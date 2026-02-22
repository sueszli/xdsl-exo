from __future__ import annotations

from exo import *


@proc
def add_vectors_4(lhs: [f32][4] @ DRAM, rhs: [f32][4] @ DRAM):
    for i in seq(0, 4):
        lhs[i] += rhs[i]


@proc
def add_vectors_16(lhs: f32[16] @ DRAM, rhs: f32[16] @ DRAM):
    for i in seq(0, 4):
        add_vectors_4(lhs[i * 4 : (i + 1) * 4], rhs[i * 4 : (i + 1) * 4])
