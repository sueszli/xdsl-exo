from __future__ import annotations

import ctypes
import runpy
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_match
from exo import *
from exo.API import Procedure
from exo.backend.prec_analysis import get_default_prec, set_default_prec
from exo.core.LoopIR import LoopIR
from exo.libs.memories import StaticMemory
from llvmlite import binding
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp

from exojit import FFI, LLVMBackend, jit, to_mlir


@proc
def allocating_increment(x: f32[1]):
    value: f32
    value = x[0] + 1.0
    x[0] = value


@proc
def repeated_calls(x: f32[1]):
    value: f32
    value = 40.0
    allocating_increment(x)
    allocating_increment(x)
    x[0] += value


@pytest.mark.parametrize("explicit_callee", [False, True])
def test_repeated_allocating_calls_and_exports(explicit_callee):
    assert_match(repeated_calls, x=[0.0])
    module = to_mlir([repeated_calls, allocating_increment, allocating_increment] if explicit_callee else repeated_calls)
    assert isinstance(module, ModuleOp)
    module.verify()
    functions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.body.blocks]
    assert [op.sym_name.data for op in functions] == ["allocating_increment", "repeated_calls"]
    for func, expected, allocations in zip(functions, (1.0, 42.0), (1, 3)):
        callees = [op.callee.root_reference.data for op in func.walk() if isinstance(op, llvm.CallOp)]
        assert callees.count("malloc") == callees.count("free") == allocations
        assert set(callees) == {"malloc", "free"}
        compiled = LLVMBackend._jit_backend.jit(module, func.sym_name.data, LLVMBackend._context())
        x = FFI().new("float[1]", [0.0])
        compiled.c_func(x)
        assert x[0] == expected


@proc
def fractional_value(out: f64[1]):
    out[0] = 1.0 / 2.0


@proc
def call_fraction(out: f64[1]):
    fractional_value(out)


def test_no_generic_precision_folding():
    out = [0.0]
    jit(call_fraction)(out)
    assert out == [0.5]
    assert_match(call_fraction, out=[0.0])


@proc
def local_windows(A: f32[4, 5]):
    w = A[1:4, 1:5]
    z = w[1:3, 1:3]
    for i in seq(0, 2):
        for j in seq(0, 2):
            z[i, j] = 7.0


@proc
def branch_windows(N: size, x: f32[1]):
    if N == 1:
        allocating_increment(x)
    else:
        w = x[0:1]
        w[0] += 2.0


@pytest.mark.parametrize("N", [1, 2])
def test_calls_and_windows_in_branches(N):
    assert_match(branch_windows, N=N, x=[0.0])


def test_local_chained_windows():
    assert_match(local_windows, A=[0.0] * 20)
    A = np.zeros((4, 5), dtype=np.float32)
    jit(local_windows, raw=True)(A)
    expected = np.zeros_like(A)
    expected[2:4, 2:4] = 7.0
    np.testing.assert_array_equal(A, expected)


@pytest.mark.parametrize("filename, name, shape, written", [("window_row", "window_row", (4, 4), slice(None)), ("window_col", "window_col", (4, 4), slice(None)), ("window_chained", "outer", (4, 4, 4), (2, 1, 0))])
def test_window_snapshot_execution(filename, name, shape, written):
    # In particular, column accesses must use the source's stride, not unit stride.
    source = Path(__file__).parents[1] / "filecheck" / f"{filename}.py"
    kernel = runpy.run_path(str(source))[name]
    assert_match(kernel, A=np.full(shape, 3.0).tolist())
    A = np.full(shape, 3.0, dtype=np.float32)
    jit(kernel, raw=True)(A)
    expected = np.full_like(A, 3.0)
    expected[written] = 1.0 if name == "outer" else 0.0
    np.testing.assert_array_equal(A, expected)


@proc
def requires_static(x: f32[1] @ StaticMemory):
    pass


@proc
def wrong_memory(x: f32[1]):
    requires_static(x)


@proc
def requires_f32(x: f32[1]):
    pass


@proc
def wrong_precision(x: f64[1]):
    requires_f32(x)


@proc
def transitive_wrong_memory(x: f32[1]):
    wrong_memory(x)


@proc
def wrong_window(x: [f32][1]):
    requires_f32(x)


@pytest.mark.parametrize("kernel, error", [(wrong_memory, "expected argument in StaticMemory"), (transitive_wrong_memory, "expected argument in StaticMemory"), (wrong_precision, "expected precision f32"), (wrong_window, "expected a non-window tensor")])
def test_original_call_boundaries_validated(kernel, error):
    # These invalid arguments disappear completely if validation follows inlining.
    with pytest.raises(TypeError, match=error):
        to_mlir(kernel)


@instr("not_the_specification({x_data});")
def opaque_instruction(x: [f32][1], value: f32):
    x[0] = 1.0


@proc
def calls_instruction(x: f32[1]):
    value: f32
    value = 7.0
    opaque_instruction(x, value)
    opaque_instruction(x, value)


@proc
def transitively_calls_instruction(x: f32[1]):
    calls_instruction(x)


@proc
def calls_instruction_window(x: f32[2, 3]):
    value: f32
    value = 7.0
    w = x[1, 1:2]
    opaque_instruction(w, value)


@pytest.mark.parametrize("kernel, size, offset, expected", [(calls_instruction, 1, 0, 14.0), (transitively_calls_instruction, 1, 0, 14.0), (calls_instruction_window, 6, 4, 7.0)])
def test_instruction_specifications_are_not_inlined(kernel, size, offset, expected):
    @ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_float)
    def native(x, value):
        x[0] += value

    binding.add_symbol("opaque_instruction", ctypes.cast(native, ctypes.c_void_p).value)
    module = to_mlir(kernel)
    declarations = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == "opaque_instruction"]
    assert len(declarations) == 1 and not declarations[0].body.blocks
    x = [0.0] * size
    jit(kernel)(x)
    assert x == [expected if i == offset else 0.0 for i in range(size)]


def test_instruction_definitions_are_not_exports():
    assert "opaque_instruction" not in str(to_mlir([fractional_value, opaque_instruction]))


def test_repeated_call_graph_expansion():
    # No simplify pass (or arbitrary statement/call budget) is needed for expansion.
    kernel = allocating_increment
    for depth in range(6):
        ir = kernel._loopir_proc
        arg = ir.args[0]
        call = LoopIR.Call(ir, [LoopIR.Read(arg.name, [], arg.type, arg.srcinfo)], ir.srcinfo)
        kernel = Procedure(ir.update(name=f"double_{depth}", body=[call, call]))
    x = [0.0]
    jit(kernel)(x)
    assert x == [64.0]


@proc
def a_fixed(x: f32[4, 4]):
    x[1, 2] = 7.0


@proc
def z_expression(x: f32[4, 2 * 2]):
    a_fixed(x)


@proc
def z_predicate(N: size, x: f32[4, N]):
    assert N == 4
    a_fixed(x)


@proc
def a_large(x: f32[2, 65536]):
    x[1, 2] = 7.0


@proc
def z_large(N: size, x: f32[2, N]):
    assert N == 65536
    a_large(x)


@pytest.mark.parametrize("kernel, sizes, shape", [(z_expression, (), (4, 4)), (z_predicate, (4,), (4, 4)), (z_large, (65536,), (2, 65536))])
def test_inlined_callee_shape_preserved(kernel, sizes, shape):
    # The original separate static callee works even with computed/constrained caller shapes.
    rows, width = shape
    x = [[0.0] * width for _ in range(rows)]
    jit(kernel)(*sizes, x)
    expected = [[0.0] * width for _ in range(rows)]
    expected[1][2] = 7.0
    assert x == expected
    raw = np.zeros(shape, dtype=np.float32)
    jit(kernel, raw=True)(*sizes, raw)
    np.testing.assert_array_equal(raw, expected)
    assert_match(kernel, **({"N": sizes[0]} if sizes else {}), x=[[0.0] * width for _ in range(rows)])


@proc
def a_generic(x: R[1]):
    tmp: R
    tmp = 1.0 / 2.0
    x[0] = x[0] + tmp + 1.0000000001


@proc
def z_caller64(x: f64[1]):
    a_generic(x)


@proc
def a_double(x: f64[1]):
    tmp: f64
    tmp = 1.0000000001
    x[0] = tmp + 1.0 / 2.0


@proc
def z_double(x: f64[1]):
    a_double(x)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("kernel", [z_caller64, z_double])
def test_generic_and_explicit_f64_precision(kernel, precision):
    previous = str(get_default_prec())
    set_default_prec(precision)
    try:
        if kernel is z_caller64 and precision == "f32":
            with pytest.raises(TypeError, match="expected precision f32"):
                to_mlir(kernel)
        else:
            x = [0.0]
            jit(kernel)(x)
            assert x == pytest.approx([1.5000000001], rel=0, abs=1e-15)
            raw = np.zeros(1, dtype=np.float64)
            jit(kernel, raw=True)(raw)
            assert raw.tolist() == x
            assert_match(kernel, x=[0.0])
    finally:
        set_default_prec(previous)
