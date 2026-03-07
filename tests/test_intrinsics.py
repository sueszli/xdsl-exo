import pytest
from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import f32, i64
from xdsl.dialects.llvm import FNegOp, MaskedStoreOp
from xdsl.utils.test_value import create_ssa_value

from xdsl_exo.patches_intrinsics import _make_intrinsics


def _ptr():
    return create_ssa_value(llvm.LLVMPointerType())


def _op_types(ops):
    return [type(op).__name__ for op in ops]


INTRINSICS = _make_intrinsics()

EXPECTED_PLAIN_NAMES = [
    "vec_abs",
    "vec_add",
    "vec_add_red",
    "vec_brdcst_scl",
    "vec_copy",
    "vec_fmadd1",
    "vec_fmadd2",
    "vec_fmadd_red",
    "vec_load",
    "vec_mul",
    "vec_neg",
    "vec_store",
    "vec_zero",
]


class TestRegistryCompleteness:
    def test_total_entry_count(self):
        assert len(INTRINSICS) == 58

    @pytest.mark.parametrize("name", EXPECTED_PLAIN_NAMES)
    def test_plain_f32x8_exists(self, name):
        assert f"{name}_f32x8" in INTRINSICS

    @pytest.mark.parametrize("name", EXPECTED_PLAIN_NAMES)
    def test_plain_f64x4_exists(self, name):
        assert f"{name}_f64x4" in INTRINSICS

    @pytest.mark.parametrize("name", EXPECTED_PLAIN_NAMES)
    def test_pfx_f32x8_exists(self, name):
        assert f"{name}_f32x8_pfx" in INTRINSICS

    @pytest.mark.parametrize("name", EXPECTED_PLAIN_NAMES)
    def test_pfx_f64x4_exists(self, name):
        assert f"{name}_f64x4_pfx" in INTRINSICS

    def test_reduce_entries(self):
        assert "vec_reduce_add_scl_f32x8" in INTRINSICS
        assert "vec_reduce_add_scl_f64x4" in INTRINSICS

    def test_mm256_entries(self):
        for name in ["mm256_storeu_ps", "mm256_loadu_ps", "mm256_fmadd_ps", "mm256_broadcast_ss"]:
            assert name in INTRINSICS

    def test_all_handlers_callable(self):
        for name, handler in INTRINSICS.items():
            assert callable(handler), f"{name} is not callable"


class TestPlainOpWiring:
    # each name maps to the right op class
    @pytest.mark.parametrize(
        "key,args,expected_op",
        [
            ("vec_copy_f32x8", [_ptr(), _ptr()], "LoadOp"),
            ("vec_neg_f32x8", [_ptr(), _ptr()], "FNegOp"),
            ("vec_abs_f32x8", [_ptr(), _ptr()], "FAbsOp"),
            ("vec_add_f32x8", [_ptr(), _ptr(), _ptr()], "FAddOp"),
            ("vec_mul_f32x8", [_ptr(), _ptr(), _ptr()], "FMulOp"),
            ("vec_fmadd1_f32x8", [_ptr(), _ptr(), _ptr(), _ptr()], "FMAOp"),
            ("vec_fmadd_red_f32x8", [_ptr(), _ptr(), _ptr()], "FMAOp"),
        ],
    )
    def test_core_op_type(self, key, args, expected_op):
        types = _op_types(INTRINSICS[key](args))
        assert expected_op in types
        assert types[-1] == "StoreOp"


class TestPlainStoreTarget:
    # final StoreOp must write to dst (args[0])
    @pytest.mark.parametrize(
        "key,args_fn",
        [
            ("vec_copy_f32x8", lambda d: [d, _ptr()]),
            ("vec_add_f32x8", lambda d: [d, _ptr(), _ptr()]),
            ("vec_fmadd1_f32x8", lambda d: [d, _ptr(), _ptr(), _ptr()]),
            ("vec_fmadd_red_f32x8", lambda d: [d, _ptr(), _ptr()]),
            ("vec_zero_f32x8", lambda d: [d]),
        ],
    )
    def test_stores_to_dst(self, key, args_fn):
        dst = _ptr()
        store = INTRINSICS[key](args_fn(dst))[-1]
        assert isinstance(store, llvm.StoreOp)
        assert store.ptr is dst


class TestPlainDataFlow:
    # compute op result is the value fed to the final store
    def test_add(self):
        ops = INTRINSICS["vec_add_f32x8"]([_ptr(), _ptr(), _ptr()])
        assert isinstance(ops[2], llvm.FAddOp)
        assert ops[3].value is ops[2].res

    def test_neg(self):
        ops = INTRINSICS["vec_neg_f32x8"]([_ptr(), _ptr()])
        assert isinstance(ops[1], FNegOp)
        assert ops[2].value is ops[1].res

    def test_copy(self):
        ops = INTRINSICS["vec_copy_f32x8"]([_ptr(), _ptr()])
        assert isinstance(ops[0], llvm.LoadOp)
        assert ops[1].value is ops[0].dereferenced_value

    def test_fma(self):
        ops = INTRINSICS["vec_fmadd1_f32x8"]([_ptr(), _ptr(), _ptr(), _ptr()])
        assert isinstance(ops[3], vector.FMAOp)
        assert ops[4].value is ops[3].res


class TestPlainLoadSources:
    def test_add_loads_from_srcs(self):
        dst, src_a, src_b = _ptr(), _ptr(), _ptr()
        ops = INTRINSICS["vec_add_f32x8"]([dst, src_a, src_b])
        assert ops[0].ptr is src_a
        assert ops[1].ptr is src_b

    def test_add_red_loads_dst_and_src(self):
        dst, src = _ptr(), _ptr()
        ops = INTRINSICS["vec_add_red_f32x8"]([dst, src])
        assert {ops[0].ptr, ops[1].ptr} == {dst, src}

    def test_fma_red_loads_dst_as_acc(self):
        dst, a, b = _ptr(), _ptr(), _ptr()
        ops = INTRINSICS["vec_fmadd_red_f32x8"]([dst, a, b])
        assert {ops[i].ptr for i in range(3)} == {dst, a, b}


class TestPfxStructure:
    # prefix handlers: mask ops, then core ops, then MaskedStoreOp
    def test_add_pfx_f32x8(self):
        lane = create_ssa_value(i64)
        types = _op_types(INTRINSICS["vec_add_f32x8_pfx"]([lane, _ptr(), _ptr(), _ptr()]))
        assert types[:3] == ["ConstantOp", "BroadcastOp", "ICmpOp"]
        assert types[3:6] == ["LoadOp", "LoadOp", "FAddOp"]
        assert types[-1] == "MaskedStoreOp"

    def test_pfx_stores_to_dst(self):
        lane, dst = create_ssa_value(i64), _ptr()
        masked_store = INTRINSICS["vec_add_f32x8_pfx"]([lane, dst, _ptr(), _ptr()])[-1]
        assert isinstance(masked_store, MaskedStoreOp)
        assert masked_store.data is dst


class TestPfxMaskVariants:
    def test_f64x4_no_sext(self):
        # _mask_f64x4: no SExtOp
        lane = create_ssa_value(i64)
        assert "SExtOp" not in _op_types(INTRINSICS["vec_add_f64x4_pfx"]([lane, _ptr(), _ptr(), _ptr()]))

    def test_f64x4_ext_has_sext(self):
        # _mask_f64x4_ext: SExtOp to upcast lane_count i32 -> i64
        lane = create_ssa_value(i64)
        assert "SExtOp" in _op_types(INTRINSICS["vec_add_red_f64x4_pfx"]([lane, _ptr(), _ptr()]))


class TestPfxAbsSpecial:
    def test_abs_pfx_has_extra_store(self):
        # _build_abs_pfx pre-stores src to all lanes, then MaskedStoreOp overwrites active lanes
        lane = create_ssa_value(i64)
        types = _op_types(INTRINSICS["vec_abs_f32x8_pfx"]([lane, _ptr(), _ptr()]))
        assert types.count("StoreOp") == 1
        assert types[-1] == "MaskedStoreOp"


class TestReduceHandler:
    def _make_acc(self, elem_type):
        # acc_val.owner must be llvm.LoadOp (asserted in _reduce_handler)
        acc_ptr = _ptr()
        load = llvm.LoadOp(acc_ptr, elem_type)
        return load.dereferenced_value, acc_ptr

    def test_structure(self):
        acc_val, _ = self._make_acc(f32)
        assert _op_types(INTRINSICS["vec_reduce_add_scl_f32x8"]([acc_val, _ptr()])) == ["LoadOp", "ReductionOp", "StoreOp"]

    def test_stores_back_to_acc_ptr(self):
        acc_val, acc_ptr = self._make_acc(f32)
        store = INTRINSICS["vec_reduce_add_scl_f32x8"]([acc_val, _ptr()])[2]
        assert isinstance(store, llvm.StoreOp)
        assert store.ptr is acc_ptr

    def test_uses_add_combining_kind(self):
        acc_val, _ = self._make_acc(f32)
        reduction = INTRINSICS["vec_reduce_add_scl_f32x8"]([acc_val, _ptr()])[1]
        assert isinstance(reduction, vector.ReductionOp)
        assert vector.CombiningKindFlag.ADD in reduction.kind.data


class TestMm256:
    def test_fmadd_ps(self):
        assert _op_types(INTRINSICS["mm256_fmadd_ps"]([_ptr(), _ptr(), _ptr()])) == ["LoadOp", "LoadOp", "LoadOp", "FMAOp", "StoreOp"]

    def test_fmadd_stores_to_dst(self):
        dst = _ptr()
        store = INTRINSICS["mm256_fmadd_ps"]([dst, _ptr(), _ptr()])[-1]
        assert isinstance(store, llvm.StoreOp)
        assert store.ptr is dst

    def test_broadcast_ss_loads_scalar(self):
        # loads scalar f32, not a vector
        ops = INTRINSICS["mm256_broadcast_ss"]([_ptr(), _ptr()])
        assert isinstance(ops[0], llvm.LoadOp)
        assert ops[0].dereferenced_value.type == f32
