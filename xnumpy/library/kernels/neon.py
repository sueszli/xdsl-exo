from __future__ import annotations

from exo import *  # noqa: F403,F401

from xnumpy.patches_exo import NEON  # noqa: F401


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_add_f32x4({dst_data}, {a_data}, {b_data});")
def neon_add_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("neon_sub_f32x4({dst_data}, {a_data}, {b_data});")
def neon_sub_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] - b[i]


@instr("neon_mul_f32x4({dst_data}, {a_data}, {b_data});")
def neon_mul_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("neon_neg_f32x4({dst_data}, {a_data});")
def neon_neg_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    for i in seq(0, 4):
        dst[i] = -a[i]


@instr("neon_vadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_vadd_f32x4(dst: [f32][4] @ DRAM, a: [f32][4] @ DRAM, b: [f32][4] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("neon_vsub_f32x4({dst_data}, {a_data}, {b_data});")
def neon_vsub_f32x4(dst: [f32][4] @ DRAM, a: [f32][4] @ DRAM, b: [f32][4] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] - b[i]


@instr("neon_vmul_f32x4({dst_data}, {a_data}, {b_data});")
def neon_vmul_f32x4(dst: [f32][4] @ DRAM, a: [f32][4] @ DRAM, b: [f32][4] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("neon_vneg_f32x4({dst_data}, {a_data});")
def neon_vneg_f32x4(dst: [f32][4] @ DRAM, a: [f32][4] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    for i in seq(0, 4):
        dst[i] = -a[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):  # type: ignore[valid-type]
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]
