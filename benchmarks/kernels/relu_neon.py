from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import select
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


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_fmax_acc_f32x4({acc_data}, {src_data});")
def neon_fmax_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = select(acc[i], src[i], src[i], acc[i])


@cache
def relu_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    n16 = n // 16

    @proc
    def _relu_neon(out: f32[n] @ DRAM, inp: f32[n] @ DRAM):
        zero_buf: f32[1] @ DRAM
        zero_buf[0] = 0.0

        for i in seq(0, n16):
            x0: f32[4] @ NEON
            x1: f32[4] @ NEON
            x2: f32[4] @ NEON
            x3: f32[4] @ NEON
            neon_loadu_f32x4(x0, inp[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(x1, inp[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(x2, inp[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(x3, inp[16 * i + 12 : 16 * i + 16])

            z0: f32[4] @ NEON
            z1: f32[4] @ NEON
            z2: f32[4] @ NEON
            z3: f32[4] @ NEON
            neon_broadcast_f32x4(z0, zero_buf[0:1])
            neon_broadcast_f32x4(z1, zero_buf[0:1])
            neon_broadcast_f32x4(z2, zero_buf[0:1])
            neon_broadcast_f32x4(z3, zero_buf[0:1])

            neon_fmax_acc_f32x4(z0, x0)
            neon_fmax_acc_f32x4(z1, x1)
            neon_fmax_acc_f32x4(z2, x2)
            neon_fmax_acc_f32x4(z3, x3)

            neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], z0)
            neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], z1)
            neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], z2)
            neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], z3)

    name = f"_relu_neon_{n}"
    p = rename(_relu_neon, name)
    return compile_jit(p)[name]
