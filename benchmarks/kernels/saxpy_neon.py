from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@cache
def saxpy_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0, "N must be divisible by 16 for 4x unrolled f32x4"
    n16 = n // 16

    @proc
    def _saxpy_neon(y: f32[n] @ DRAM, x: f32[n] @ DRAM, a: f32[1] @ DRAM):
        a_vec: f32[4] @ NEON
        neon_broadcast_f32x4(a_vec, a[0:1])
        for i in seq(0, n16):
            x0: f32[4] @ NEON
            x1: f32[4] @ NEON
            x2: f32[4] @ NEON
            x3: f32[4] @ NEON
            y0: f32[4] @ NEON
            y1: f32[4] @ NEON
            y2: f32[4] @ NEON
            y3: f32[4] @ NEON
            neon_loadu_f32x4(x0, x[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(x1, x[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(x2, x[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(x3, x[16 * i + 12 : 16 * i + 16])
            neon_loadu_f32x4(y0, y[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(y1, y[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(y2, y[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(y3, y[16 * i + 12 : 16 * i + 16])
            neon_fmadd_f32x4(y0, a_vec, x0)
            neon_fmadd_f32x4(y1, a_vec, x1)
            neon_fmadd_f32x4(y2, a_vec, x2)
            neon_fmadd_f32x4(y3, a_vec, x3)
            neon_storeu_f32x4(y[16 * i + 0 : 16 * i + 4], y0)
            neon_storeu_f32x4(y[16 * i + 4 : 16 * i + 8], y1)
            neon_storeu_f32x4(y[16 * i + 8 : 16 * i + 12], y2)
            neon_storeu_f32x4(y[16 * i + 12 : 16 * i + 16], y3)

    name = f"_saxpy_neon_{n}"
    p = rename(_saxpy_neon, name)
    return compile_jit(p)[name]
