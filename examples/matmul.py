from __future__ import annotations

from exo import *


@proc
def matmul(M: size, N: size, K: size, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM, C: f32[M, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]
