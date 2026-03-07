# RUN: uv run python %s | filecheck %s
"""Lit test for ConvertVecIntrinsic: llvm.call @vec_*/mm256_* -> LLVM/vector ops."""

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, f32, i64
from xdsl.ir import Block, Region
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.printer import Printer

from xdsl_exo.patches_intrinsics import ConvertVecIntrinsic

ptr = llvm.LLVMPointerType()
ext = llvm.LinkageAttr("external")


def lower(name, arg_types):
    """Build module with llvm.call @name, apply ConvertVecIntrinsic, print result."""
    extern = llvm.FuncOp(name, llvm.LLVMFunctionType(arg_types), linkage=ext)
    block = Block(arg_types=arg_types)
    block.add_ops([llvm.CallOp(name, *block.args), llvm.ReturnOp()])
    func = llvm.FuncOp("test", llvm.LLVMFunctionType(arg_types), linkage=ext, body=Region([block]))
    module = ModuleOp([extern, func])
    PatternRewriteWalker(GreedyRewritePatternApplier([ConvertVecIntrinsic()])).rewrite_module(module)
    Printer().print_op(module)
    print()


# CHECK-LABEL: @test_add
# CHECK:       %{{.*}} = "llvm.load"(%{{.*}}) {{.*}} -> vector<8xf32>
# CHECK-NEXT:  %{{.*}} = "llvm.load"(%{{.*}}) {{.*}} -> vector<8xf32>
# CHECK-NEXT:  %{{.*}} = llvm.fadd %{{.*}}, %{{.*}} : vector<8xf32>
# CHECK-NEXT:  "llvm.store"(%{{.*}}, %{{.*}}) {{.*}}
# CHECK-NEXT:  llvm.return
lower("vec_add_f32x8", [ptr, ptr, ptr])

# CHECK-LABEL: @test_mul
# CHECK:       %{{.*}} = "llvm.load"
# CHECK:       %{{.*}} = "llvm.load"
# CHECK-NEXT:  %{{.*}} = llvm.fmul %{{.*}}, %{{.*}} : vector<8xf32>
# CHECK-NEXT:  "llvm.store"
lower("vec_mul_f32x8", [ptr, ptr, ptr])

# CHECK-LABEL: @test_neg
# CHECK:       %{{.*}} = "llvm.load"(%{{.*}}) {{.*}} -> vector<8xf32>
# CHECK-NEXT:  %{{.*}} = llvm.fneg %{{.*}} : vector<8xf32>
# CHECK-NEXT:  "llvm.store"
lower("vec_neg_f32x8", [ptr, ptr])

# CHECK-LABEL: @test_copy
# CHECK:       %{{.*}} = "llvm.load"(%{{.*}}) {{.*}} -> vector<8xf32>
# CHECK-NEXT:  "llvm.store"(%{{.*}}, %{{.*}}) {{.*}}
# CHECK-NEXT:  llvm.return
lower("vec_copy_f32x8", [ptr, ptr])

# CHECK-LABEL: @test_fmadd
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK-NEXT:  %{{.*}} = vector.fma %{{.*}}, %{{.*}}, %{{.*}} : vector<8xf32>
# CHECK-NEXT:  "llvm.store"
lower("vec_fmadd1_f32x8", [ptr, ptr, ptr, ptr])

# fmadd_red: dst is loaded as accumulator, result stored back
# CHECK-LABEL: @test_fmadd_red
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       %{{.*}} = "llvm.load"{{.*}} -> vector<8xf32>
# CHECK-NEXT:  %{{.*}} = vector.fma
# CHECK-NEXT:  "llvm.store"
lower("vec_fmadd_red_f32x8", [ptr, ptr, ptr])

# CHECK-LABEL: @test_zero
# CHECK:       %{{.*}} = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>)
# CHECK-NEXT:  "llvm.store"
lower("vec_zero_f32x8", [ptr])

# CHECK-LABEL: @test_brdcst
# CHECK:       %{{.*}} = vector.broadcast %{{.*}} : f32 to vector<8xf32>
# CHECK-NEXT:  "llvm.store"
lower("vec_brdcst_scl_f32x8", [ptr, f32])

# f64x4 variant: loads/stores use vector<4xf64>
# CHECK-LABEL: @test_add_f64
# CHECK:       "llvm.load"{{.*}} -> vector<4xf64>
# CHECK:       "llvm.load"{{.*}} -> vector<4xf64>
# CHECK-NEXT:  llvm.fadd %{{.*}}, %{{.*}} : vector<4xf64>
# CHECK-NEXT:  "llvm.store"
lower("vec_add_f64x4", [ptr, ptr, ptr])

# pfx: mask preamble, then core ops, then masked store
# CHECK-LABEL: @test_add_pfx
# CHECK:       llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7]>
# CHECK-NEXT:  vector.broadcast %{{.*}} : i64 to vector<8xi64>
# CHECK-NEXT:  llvm.icmp "slt"
# CHECK-NEXT:  "llvm.load"{{.*}} -> vector<8xf32>
# CHECK-NEXT:  "llvm.load"{{.*}} -> vector<8xf32>
# CHECK-NEXT:  llvm.fadd
# CHECK-NEXT:  llvm.intr.masked.store %{{.*}}, %{{.*}}, %{{.*}} {alignment = 32 : i32}
# CHECK-NEXT:  llvm.return
lower("vec_add_f32x8_pfx", [i64, ptr, ptr, ptr])

# mm256_fmadd_ps: load acc from dst, load a, load b, fma, store to dst
# CHECK-LABEL: @test_mm256_fmadd
# CHECK:       "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       "llvm.load"{{.*}} -> vector<8xf32>
# CHECK:       "llvm.load"{{.*}} -> vector<8xf32>
# CHECK-NEXT:  vector.fma
# CHECK-NEXT:  "llvm.store"
lower("mm256_fmadd_ps", [ptr, ptr, ptr])

# mm256_broadcast_ss: load scalar f32, broadcast to vector<8xf32>, store
# CHECK-LABEL: @test_mm256_brdcst
# CHECK:       "llvm.load"(%{{.*}}) {{.*}} -> f32
# CHECK-NEXT:  vector.broadcast %{{.*}} : f32 to vector<8xf32>
# CHECK-NEXT:  "llvm.store"
lower("mm256_broadcast_ss", [ptr, ptr])

# unknown call is left untouched
# CHECK-LABEL: @test_passthrough
# CHECK:       "llvm.call"
# CHECK-NEXT:  llvm.return
lower("some_other_func", [ptr])
