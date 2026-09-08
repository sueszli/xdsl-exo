from __future__ import annotations

import ctypes

import pytest
from exo import *
from exo.libs.memories import DRAM_STACK
from xdsl.dialects.builtin import ModuleOp

from exojit import IRGenerator, jit, to_mlir


@proc
def increment_window(x: [f32][4] @ DRAM):
    for i in seq(0, 4):
        x[i] += 1.0


@proc
def local_row_window(x: f32[2, 4] @ DRAM):
    row = x[1, :]
    increment_window(row)


def test_contiguous_window_callee_is_standalone():
    x = [2.0, 3.0, 4.0, 5.0]
    jit(increment_window)(x)
    assert x == [3.0, 4.0, 5.0, 6.0]
    matrix = [float(i) for i in range(8)]
    jit(local_row_window)(matrix)
    assert matrix == [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]


def test_generator_emits_llvm_memory_without_lowering():
    module = IRGenerator().generate([local_row_window._loopir_proc])
    assert isinstance(module, ModuleOp)
    module.verify()
    names = {op.name for op in module.walk()}
    assert {"llvm.getelementptr", "llvm.load", "llvm.store", "llvm.call"} <= names
    assert all(name == "builtin.module" or name.startswith("llvm.") for name in names)
    assert "llvm.noalias" not in str(module)


@proc
def overlapping_copy(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = x[i] + 1.0


def test_raw_pointers_may_overlap():
    x = (ctypes.c_float * 5)(10.0, 20.0, 30.0, 40.0, 50.0)
    ptr = ctypes.addressof(x)
    jit(overlapping_copy, raw=True)(ptr + ctypes.sizeof(ctypes.c_float), ptr)
    assert list(x) == [10.0, 11.0, 12.0, 13.0, 14.0]


@proc
def dynamic_allocation(N: size, out: f32[N] @ DRAM):
    tmp: f32[N]
    for i in seq(0, N):
        tmp[i] = out[i]
    for i in seq(0, N):
        out[i] = tmp[i]


def test_dynamic_allocations_remain_unsupported():
    with pytest.raises(AssertionError, match="dynamic-sized allocs are not supported"):
        to_mlir(dynamic_allocation)


@proc
def stack_copy(out: f32[4] @ DRAM):
    tmp: f32[4] @ DRAM_STACK
    for i in seq(0, 4):
        tmp[i] = out[i]
    for i in seq(0, 4):
        out[i] = tmp[i]


def test_stack_allocation_policy():
    ir = str(to_mlir(stack_copy))
    assert "llvm.alloca" in ir
    assert "llvm.call @malloc" not in ir and "llvm.call @free" not in ir
    x = [1.0, 2.0, 3.0, 4.0]
    jit(stack_copy)(x)
    assert x == [1.0, 2.0, 3.0, 4.0]


@proc
def requires_stack(x: f32[4] @ DRAM_STACK):
    x[0] = 1.0


@proc
def wrong_memory(x: f32[4] @ DRAM):
    requires_stack(x)


def test_call_memory_policy_is_still_checked():
    with pytest.raises(TypeError, match="expected argument in DRAM_STACK but got an argument in DRAM"):
        to_mlir(wrong_memory)


@proc
def floor_shape_copy(N: size, out: f32[2, 2 + (N - 3) / 2], x: f32[2, 2 + (N - 3) / 2]):
    for i in seq(0, 2):
        for j in seq(0, (N + 1) / 2):
            out[i, j] = x[i, j]


@proc
def call_floor_shape_copy(N: size, out: f32[2, 2 + (N - 3) / 2], x: f32[2, 2 + (N - 3) / 2]):
    floor_shape_copy(N, out, x)


@proc
def modulo_shape_copy(N: size, out: f32[2, 1 + (N - 3) % 2], x: f32[2, 1 + (N - 3) % 2]):
    for i in seq(0, 2):
        for j in seq(0, 1 + (N + 1) % 2):
            out[i, j] = x[i, j]


@proc
def indexed_floor_shape_copy(N: size, n: size[4], out: f32[2, n[2 + (N - 3) / 2]], x: f32[2, n[2 + (N - 3) / 2]]):
    assert N <= 4
    assert n[2 + (N - 3) / 2] == 3
    for i in seq(0, 2):
        for j in seq(0, 3):
            out[i, j] = x[i, j]


@proc
def indexed_modulo_shape_copy(N: size, n: size[4], out: f32[2, n[1 + (N - 3) % 2]], x: f32[2, n[1 + (N - 3) % 2]]):
    assert n[1 + (N - 3) % 2] == 3
    for i in seq(0, 2):
        for j in seq(0, 3):
            out[i, j] = x[i, j]


@proc
def nested_floor_shape_copy(N: size, n: size[4], out: f32[2, n[3 + -n[2 + -((N - 3) / 2)]]], x: f32[2, n[3 + -n[2 + -((N - 3) / 2)]]]):
    assert N <= 4
    assert n[2 + -((N - 3) / 2)] <= 3
    assert n[3 + -n[2 + -((N - 3) / 2)]] == 3
    for i in seq(0, 2):
        for j in seq(0, 3):
            out[i, j] = x[i, j]


@proc
def nested_modulo_shape_copy(N: size, n: size[4], out: f32[2, n[3 + -n[2 + (-(N - 1)) % 2]]], x: f32[2, n[3 + -n[2 + (-(N - 1)) % 2]]]):
    assert n[2 + (-(N - 1)) % 2] <= 3
    assert n[3 + -n[2 + (-(N - 1)) % 2]] == 3
    for i in seq(0, 2):
        for j in seq(0, 3):
            out[i, j] = x[i, j]


@pytest.mark.parametrize(
    "kernel, width, sizes",
    [
        (floor_shape_copy, lambda n: (n + 1) // 2, None),
        (call_floor_shape_copy, lambda n: (n + 1) // 2, None),
        (modulo_shape_copy, lambda n: 1 + (n + 1) % 2, None),
        (indexed_floor_shape_copy, lambda n: 3, [5, 3, 4, 6]),
        (indexed_modulo_shape_copy, lambda n: 3, [4, 5, 3, 6]),
        (nested_floor_shape_copy, lambda n: 3, [4, 3, 1, 2]),
        (nested_modulo_shape_copy, lambda n: 3, [4, 3, 1, 2]),
    ],
)
def test_shape_arithmetic_matches_python_buffer_size(kernel, width, sizes):
    normal, raw = jit(kernel), jit(kernel, raw=True)
    for n in (2,) if sizes is not None else (1, 2, 3, 4, 6):
        args = (n,) if sizes is None else (n, (ctypes.c_int64 * len(sizes))(*sizes))
        values = [float(i + 10) for i in range(2 * width(n))]
        # Padding makes wrong row strides observable without an out-of-bounds raw access.
        array = ctypes.c_float * (len(values) + 4)
        out = array(*([-1.0] * (len(values) + 4)))
        raw(*args, out, array(*values, *([99.0] * 4)))
        assert list(out) == values + [-1.0] * 4
        # Indexed-size dimensions use the raw ABI; the list converter only handles scalar sizes.
        if sizes is None:
            out = [-1.0] * len(values)
            normal(n, out, values)
            assert out == values


@proc
def mixed_tensor_store(out: f32[2], x: f64[1]):
    out[0] = x[0]


@proc
def mixed_scalar_store(out: f64[1], x: f32[1]):
    tmp: f64
    tmp = x[0]
    out[0] = tmp


@pytest.mark.parametrize("kernel", [mixed_tensor_store, mixed_scalar_store])
def test_mixed_width_stores_remain_rejected(kernel):
    with pytest.raises(AssertionError, match="mixed-width stores are not supported"):
        to_mlir(kernel)


@proc
def captured_shapes(n: size[1], mutate: i32[1], out: f32[2, n[0]], x: f32[2, 3]):
    assert n[0] == 3
    w = x[:, 0 : n[0]]
    mutate[0] = 2
    for i in seq(0, 2):
        for j in seq(0, 3):
            out[i, j] = w[i, j]


def test_argument_and_window_shapes_capture_binding_values():
    n = (ctypes.c_int64 * 1)(3)
    out, x = (ctypes.c_float * 6)(*([-1.0] * 6)), (ctypes.c_float * 6)(*range(6))
    # Raw LLVM pointers have no noalias promise: mutate overlaps the size buffer.
    jit(captured_shapes, raw=True)(n, ctypes.addressof(n), out, x)
    assert n[0] == 2
    assert list(out) == list(x)
