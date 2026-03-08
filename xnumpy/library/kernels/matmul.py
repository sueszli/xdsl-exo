from __future__ import annotations

from collections.abc import Callable

from exo import *
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify, unroll_loop

from xnumpy.backends import compile_jit


@proc
def _mm(M: size, K: size, N: size, C: f32[M, N] @ DRAM, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            C[i, j] = 0.0
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


def _schedule_matmul(n: int):
    def transform(p):
        p = fission(p, p.find("for k in _: _").before(), n_lifts=2)
        p = reorder_loops(p, "j k")
        if n % 4 == 0:
            p = divide_loop(p, "j #1", 4, ["jo", "ji"], perfect=True)
            p = unroll_loop(p, "ji")
        p = simplify(p)
        return p

    return transform


def matmul(m: int, k: int, n: int) -> Callable[..., None]:
    return compile_jit(
        _mm.partial_eval(M=m, K=k, N=n),
        f"_mm_{m}_{k}_{n}",
        schedule=_schedule_matmul(n),
    )
