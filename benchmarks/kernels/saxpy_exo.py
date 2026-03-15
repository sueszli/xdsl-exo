from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from exojit.main import compile_jit

_PAR_MIN_ELEMENTS = 524288


@proc
def _saxpy(N: size, y: f32[N] @ DRAM, x: f32[N] @ DRAM, a: f32[1] @ DRAM):
    for i in seq(0, N):
        y[i] += a[0] * x[i]


@proc
def _saxpy_par(N: size, y: f32[N] @ DRAM, x: f32[N] @ DRAM, a: f32[1] @ DRAM):
    for i in par(0, N):
        y[i] += a[0] * x[i]


@cache
def saxpy_exo(n: int) -> Callable[..., None]:
    p = (_saxpy_par if n >= _PAR_MIN_ELEMENTS else _saxpy).partial_eval(N=n)
    p = simplify(p)
    name = f"_saxpy_{n}"
    return compile_jit(rename(p, name))[name]
