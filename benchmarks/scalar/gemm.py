from __future__ import annotations

from exo import proc
from exo.libs.memories import *


@proc
def gemm(
    N: size,
    M: size,
    K: size,
    out: f32[N, K] @ DRAM,
    a: f32[N, M] @ DRAM,
    b: f32[M, K] @ DRAM,
):
    for i in seq(0, N):
        for j in seq(0, K):
            out[i, j] = 0.0
            for k in seq(0, M):
                out[i, j] += a[i, k] * b[k, j]
