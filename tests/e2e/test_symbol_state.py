from __future__ import annotations

from conftest import assert_match
from exo import *

from exojit import jit


@proc
def nested_symbol_scopes(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    tmp: f32
    tmp = 100.0
    for i in seq(0, 4):
        out[i] = tmp
        tmp: f32
        tmp = x[i]
        for i in seq(0, 2):
            tmp: f32
            tmp = 3.0
            out[i] += tmp
        out[i] += tmp
    out[3] += tmp


def test_nested_same_spelling_symbols():
    assert_match(nested_symbol_scopes, out=[0.0] * 4, x=[1.0, 2.0, 3.0, 4.0])
    out = [0.0] * 4
    jit(nested_symbol_scopes)(out, [1.0, 2.0, 3.0, 4.0])
    assert out == [113.0, 111.0, 103.0, 204.0]


@proc
def zz_symbol_leaf(x: f32, tmp: f32):
    x = x + tmp
    for i in seq(0, 2):
        tmp: f32
        tmp = 1.0
        x = x + tmp
    x = x + tmp


@proc
def z_symbol_middle(x: f32, tmp: f32):
    for i in seq(0, 2):
        zz_symbol_leaf(x, tmp)
    x = x + tmp


@proc
def a_symbol_caller(out: f32[4] @ DRAM):
    for i in seq(0, 4):
        x: f32
        tmp: f32
        x = out[i]
        tmp = 3.0
        z_symbol_middle(x, tmp)
        out[i] = x + tmp
    for i in seq(0, 4):
        out[i] += 1.0


def test_recursive_callee_symbol_and_builder_restore():
    # Caller sorts first, so both callees are emitted recursively inside its loop.
    assert_match(a_symbol_caller, out=[0.0, 1.0, 2.0, 3.0])
    out = [0.0, 1.0, 2.0, 3.0]
    jit(a_symbol_caller)(out)
    assert out == [23.0, 24.0, 25.0, 26.0]
