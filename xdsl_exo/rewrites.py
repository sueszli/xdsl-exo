from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from xdsl.dialects import arith, func, llvm, memref, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IndexType, IntegerAttr, MemRefType, VectorType, f32, f64, i64
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern

from xdsl_exo import patches as llvm_extra

# `vec_*` intrinsic lowering: `func.CallOp` -> LLVM/vector dialect ops
#
# Naming:
# -------
#     vec_<op>_<type>          - plain version. all lanes written
#     vec_<op>_<type>_pfx      - prefix version. only lanes 0..n-1 written (loop tails)
#
#     <type> = f32x8 | f64x4
#     <op>  = add, mul, neg, abs, fmadd1, fmadd2, fmadd_red, zero, ...
#
# Plain variant:
# --------------
#     func.call @vec_add_f32x8(%dst, %a, %b)
#     =>
#     %v0  = llvm.load %a : vector<8xf32>
#     %v1  = llvm.load %b : vector<8xf32>
#     %r   = llvm.fadd %v0, %v1
#            llvm.store %r, %dst
#
# Prefix (_pfx) variant:
# ----------------------
# First arg is a lane-count `n`. A boolean mask selects which lanes get written.
#
#     func.call @vec_add_f32x8_pfx(%n, %dst, %a, %b)      e.g. n=3
#     =>
#     %idx  = arith.constant   [0, 1, 2, 3, 4, 5, 6, 7]
#     %bc   = vector.broadcast [3, 3, 3, 3, 3, 3, 3, 3]   (n splatted to all lanes)
#     %mask = llvm.icmp "slt"  [T, T, T, F, F, F, F, F]   (idx < bc)
#     %v0   = llvm.load %a : vector<8xf32>
#     %v1   = llvm.load %b : vector<8xf32>
#     %r    = llvm.fadd %v0, %v1
#             llvm.masked_store %r, %dst, %mask


MaskResult: TypeAlias = tuple[list[Operation], SSAValue]
BuildResult: TypeAlias = tuple[list[Operation], SSAValue, SSAValue]
Builder: TypeAlias = Callable[..., BuildResult]
MaskFn: TypeAlias = Callable[[SSAValue], MaskResult]


def _make_mask(lane_count: SSAValue, n_lanes: int, *, extend_lane_count: bool = False) -> MaskResult:
    # mask[i] = (i < lane_count)
    # e.g. for lane_count=3 -> [T, T, T, F, F, ...]
    ops = []
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [n_lanes]), list(range(n_lanes))))
    ops.append(indices)
    if extend_lane_count:
        # upcast i32 -> i64 to match `VectorType(i64, ...)`
        ext = arith.ExtSIOp(lane_count, i64)
        ops.append(ext)
        lane_count = ext.result
    broadcast = vector.BroadcastOp(operands=[lane_count], result_types=[VectorType(i64, [n_lanes])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return ops + [broadcast, mask], mask.res


def _mask_f32x8(lane_count: SSAValue) -> MaskResult:
    # 256 AVX2 register width / 32 bit float = 8 lanes
    return _make_mask(lane_count, 8)


def _mask_f64x4(lane_count: SSAValue) -> MaskResult:
    # 256 AVX2 register width / 64 bit double = 4 lanes
    return _make_mask(lane_count, 4)


def _mask_f64x4_ext(lane_count: SSAValue) -> MaskResult:
    # lane_count is i32 at call site, needs upcasting to i64
    return _make_mask(lane_count, 4, extend_lane_count=True)


def _build_identity(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src[:]
    load = llvm.LoadOp(src, vec_type)
    return [load], load.dereferenced_value, dst


def _build_abs(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [abs(x) for x in src]
    load = llvm.LoadOp(src, vec_type)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vec_type)
    return [load, fabs], fabs.result, dst


def _build_abs_pfx(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src[:]
    # dst[:n] = [abs(x) for x in src[:n]]
    load = llvm.LoadOp(src, vec_type)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vec_type)
    return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result, dst


def _build_neg(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [-x for x in src]
    load = llvm.LoadOp(src, vec_type)
    neg = llvm_extra.FNegOp(load.dereferenced_value)
    return [load, neg], neg.res, dst


def _build_binary(binary_op: Callable[[SSAValue, SSAValue], Operation]) -> Builder:
    # dst[:] = [op(a, b) for a, b in zip(src_a, src_b)]
    def build(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
        load_a = llvm.LoadOp(src_a, vec_type)
        load_b = llvm.LoadOp(src_b, vec_type)
        result = binary_op(load_a.dereferenced_value, load_b.dereferenced_value)
        return [load_a, load_b, result], result.res, dst

    return build


def _build_add_red(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [d + s for d, s in zip(dst, src)]
    load_dst = llvm.LoadOp(dst, vec_type)
    load_src = llvm.LoadOp(src, vec_type)
    add = llvm.FAddOp(load_dst.dereferenced_value, load_src.dereferenced_value)
    return [load_dst, load_src, add], add.res, dst


def _build_fma(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, src_c: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [a*b + c for a, b, c in zip(src_a, src_b, src_c)]
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    load_c = llvm.LoadOp(src_c, vec_type)
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_c.dereferenced_value], result_types=[vec_type])
    return [load_a, load_b, load_c, fma], fma.res, dst


def _build_fma_red(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [acc + a*b for acc, a, b in zip(dst, src_a, src_b)]
    load_acc = llvm.LoadOp(dst, vec_type)
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value], result_types=[vec_type])
    return [load_acc, load_a, load_b, fma], fma.res, dst


def _build_broadcast(dst: SSAValue, scalar: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [scalar] * n_lanes
    broadcast = vector.BroadcastOp(operands=[scalar], result_types=[vec_type])
    return [broadcast], broadcast.vector, dst


def _build_zero(dst: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [0.0] * n_lanes
    zero = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(vec_type, [0.0] * vec_type.get_shape()[0]))
    return [zero], zero.result, dst


# name -> (builder_fn, vector_type, mask_fn | None)
_VEC_INTRINSICS: dict[str, tuple[Builder, VectorType, MaskFn | None]] = {}
for _name, _builder, _pfx_builder, _f64_mask in [
    ("abs", _build_abs, _build_abs_pfx, _mask_f64x4_ext),
    ("add_red", _build_add_red, None, _mask_f64x4_ext),
    ("copy", _build_identity, None, _mask_f64x4_ext),
    ("load", _build_identity, None, _mask_f64x4_ext),
    ("store", _build_identity, None, _mask_f64x4),
    ("add", _build_binary(llvm.FAddOp), None, _mask_f64x4),
    ("mul", _build_binary(llvm.FMulOp), None, _mask_f64x4),
    ("neg", _build_neg, None, _mask_f64x4),
    ("brdcst_scl", _build_broadcast, None, _mask_f64x4),
    ("fmadd2", _build_fma, None, _mask_f64x4),
    ("fmadd1", _build_fma, None, _mask_f64x4),
    ("fmadd_red", _build_fma_red, None, _mask_f64x4),
    ("zero", _build_zero, None, _mask_f64x4),
]:
    _maybe_pfx_builder = _pfx_builder or _builder
    _VEC_INTRINSICS[f"vec_{_name}_f32x8"] = (_builder, VectorType(f32, [8]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f32x8_pfx"] = (_maybe_pfx_builder, VectorType(f32, [8]), _mask_f32x8)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4"] = (_builder, VectorType(f64, [4]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4_pfx"] = (_maybe_pfx_builder, VectorType(f64, [4]), _f64_mask)


def _build_mm256_storeu_ps(dst: SSAValue, src: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = src[:]
    # (mm256 = 256-bit AVX register, storeu = store unaligned, ps = packed singles of f32x8)
    load = llvm.LoadOp(src, VectorType(f32, [8]))
    return (load, llvm.StoreOp(load.dereferenced_value, dst))


def _build_mm256_fmadd_ps(dst: SSAValue, src_a: SSAValue, src_b: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    # (fmadd = fused multiply-add, ps = packed singles of f32x8)
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    load_acc = llvm.LoadOp(dst, VectorType(f32, [8]))
    load_a = llvm.LoadOp(src_a, VectorType(f32, [8]))
    load_b = llvm.LoadOp(src_b, VectorType(f32, [8]))
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value], result_types=[VectorType(f32, [8])])
    return (zero, load_acc, load_a, load_b, fma, llvm.StoreOp(fma.res, dst, [zero.result]))


def _build_mm256_broadcast_ss(dst: SSAValue, scalar_ptr: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = [*scalar_ptr] * 8
    # (broadcast_ss = broadcast scalar single = splat one f32 to all lanes)
    # (scalar_ptr is a memref / scalar pointer, unlike the llvm ptrs used elsewhere)
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    load = memref.LoadOp.get(scalar_ptr, [zero.result])
    broadcast = vector.BroadcastOp(operands=[load.results[0]], result_types=[VectorType(f32, [8])])
    return (zero, load, broadcast, llvm.StoreOp(broadcast.results[0], dst, [zero.result]))


_MM256_INTRINSICS: dict[str, Callable[..., tuple[Operation, ...]]] = {
    "mm256_storeu_ps": _build_mm256_storeu_ps,
    "mm256_fmadd_ps": _build_mm256_fmadd_ps,
    "mm256_broadcast_ss": _build_mm256_broadcast_ss,
    "mm256_loadu_ps": _build_mm256_storeu_ps,  # same load-then-store operation
}


def _reduce(op: func.CallOp, rewriter: PatternRewriter, vec_type: VectorType) -> None:
    # acc_scalar += sum(src_vector)
    assert isinstance(op.arguments[0].owner, llvm.LoadOp)
    acc_load = op.arguments[0].owner
    src_load = llvm.LoadOp(op.arguments[1], vec_type)
    reduce = vector.ReductionOp(src_load.dereferenced_value, vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]), acc=op.arguments[0])
    rewriter.replace_matched_op((src_load, reduce, llvm.StoreOp(reduce.dest, acc_load.ptr)))


class ConvertVecIntrinsic(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        callee = op.callee.root_reference.data

        if callee == "vec_reduce_add_scl_f32x8":
            return _reduce(op, rewriter, VectorType(f32, [8]))
        if callee == "vec_reduce_add_scl_f64x4":
            return _reduce(op, rewriter, VectorType(f64, [4]))

        mm256_builder = _MM256_INTRINSICS.get(callee)
        if mm256_builder is not None:
            rewriter.replace_matched_op(mm256_builder(*op.arguments))
            return

        entry = _VEC_INTRINSICS.get(callee)
        if entry is None:
            return

        builder, vec_type, mask_fn = entry
        if mask_fn is not None:
            lane_count, dst, *srcs = op.arguments
            mask_ops, mask = mask_fn(lane_count)
            core_ops, result, out_dst = builder(dst, *srcs, vec_type=vec_type)
            rewriter.replace_matched_op((*mask_ops, *core_ops, llvm_extra.MaskedStoreOp(result, out_dst, mask)))
        else:
            dst, *srcs = op.arguments
            core_ops, result, out_dst = builder(dst, *srcs, vec_type=vec_type)
            rewriter.replace_matched_op((*core_ops, llvm.StoreOp(result, out_dst)))


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    # must run last: after shape/element info is consumed by earlier passes into ops
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()
