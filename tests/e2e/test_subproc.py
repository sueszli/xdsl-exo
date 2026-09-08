from __future__ import annotations

import pytest
from _utils import compile_mlir
from conftest import assert_match
from exo import *
from exo.API_scheduling import unroll_loop
from exo.backend.prec_analysis import get_default_prec, set_default_prec
from exo.libs.externs import select

from exojit import jit, to_mlir


@proc
def add_one_scalar(x: f32[1] @ DRAM):
    x[0] = x[0] + 1.0


@proc
def call_add_one(x: f32[1] @ DRAM):
    add_one_scalar(x)


def test_call_scalar_subproc():
    assert_match(call_add_one, x=[5.0])


@proc
def increment_scalar_value(x: f32):
    x = x + 1.0


@proc
def forward_scalar_value(x: f32):
    increment_scalar_value(x)


@proc
def call_scalar_value_subproc(x: f32[1] @ DRAM):
    value: f32
    value = x[0]
    forward_scalar_value(value)
    x[0] = value


def test_call_scalar_value_subproc():
    assert_match(call_scalar_value_subproc, x=[5.0])


@proc
def double_elements(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        x[i] = x[i] * 2.0


@proc
def call_double(x: f32[8] @ DRAM):
    double_elements(8, x)


def test_call_array_subproc():
    assert_match(call_double, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def increment(x: f32[1] @ DRAM):
    x[0] = x[0] + 1.0


@proc
def double_val(x: f32[1] @ DRAM):
    x[0] = x[0] * 2.0


@proc
def inc_then_double(x: f32[1] @ DRAM):
    increment(x)
    double_val(x)


def test_chained_subprocs():
    assert_match(inc_then_double, x=[5.0])


@proc
def add_buffers(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + b[i]


@proc
def call_add_buffers(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    add_buffers(4, out, a, b)


def test_call_multi_buffer_subproc():
    assert_match(call_add_buffers, out=[0.0] * 4, a=[1.0, 2.0, 3.0, 4.0], b=[10.0, 20.0, 30.0, 40.0])


@proc
def z_generic_expressions(out: R[5], x: R[1]):
    value: R
    value = 1.25
    value += 1.5
    out[0] = 1.0 / x[0]
    out[1] = (1.0 + 2.0) / x[0]
    out[2] = -(1.0 + 2.0)
    out[3] = select(0.0, x[0], value, -2.5)
    out[4] = select(0.0, 1.0, 2.5, 3.0)


@proc
def a_call_generic_f32(out: f32[5], x: f32[1]):
    z_generic_expressions(out, x)
    z_generic_expressions(out, x)


@proc
def a_call_generic_f64(out: f64[5], x: f64[1]):
    z_generic_expressions(out, x)
    z_generic_expressions(out, x)


@pytest.mark.parametrize("precision,caller", [("f32", a_call_generic_f32), ("f64", a_call_generic_f64)])
def test_analyzed_callee_expressions(precision, caller):
    # Caller sorts first: forward calls must use the analyzed callee body/signature,
    # including default R precision and the allocation's analysis-inserted free.
    previous = get_default_prec()
    set_default_prec(precision)
    try:
        module = to_mlir([caller, caller])
        names = [op.sym_name.data for op in module.ops]
        assert names == [caller.name(), z_generic_expressions.name(), "malloc", "free"]
        assert str(module).count("llvm.call @free(") == 1
        out = [0.0] * 5
        jit(caller)(out, [4.0])
        assert out == [0.25, 0.75, -3.0, 2.75, 2.5]
        assert_match(caller, out=[0.0] * 5, x=[4.0])
    finally:
        set_default_prec(str(previous))


@proc
def z_local_window(out: f32[1]):
    buf: f32[4]
    buf[0] = 23.0
    w = buf[0:4]
    out[0] = w[0]


@proc
def a_window_caller(out: f32[1]):
    z_local_window(out)


@pytest.mark.parametrize("target", [a_window_caller, z_local_window])
def test_local_window_lifetime(target):
    module = to_mlir(target)
    callee = str(next(op for op in module.ops if op.sym_name.data == "z_local_window"))
    assert callee.count("llvm.call @free(") == 1
    assert callee.index("llvm.load") < callee.index("llvm.call @free(")
    out = [0.0]
    jit(target)(out)
    assert out == [23.0]
    # Exo C has the same local-alias lifetime bug; use an explicit native oracle.
    assert compile_mlir(target, module)(out=[0.0])["out"].tolist() == [23.0]


@proc
def long_block(x: f32[1200]):
    for i in seq(0, 1200):
        x[i] = 23.0


@proc
def long_window_block(x: f32[1200]):
    w = x[0:1200]
    v = w[0:1200]
    for i in seq(0, 1200):
        v[i] = 23.0


@pytest.mark.parametrize("target", [long_block, long_window_block])
def test_long_block_window_normalization(target):
    # Cursor discovery must not recurse per sibling, even with aliases to inline.
    # Chained aliases also exercise forwarding after the first window is removed.
    target = unroll_loop(target, "i")
    assert len(target.body()) >= 1200
    out = [0.0] * 1200
    jit(target)(out)
    assert out == [23.0] * 1200
