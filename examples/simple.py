from __future__ import annotations

from exo import *


@proc
def helper(x: f32[1] @ DRAM):
    x[0] = 0.0


@proc
def caller(x: f32[1] @ DRAM):
    helper(x)
    x[0] = x[0] + 1.0
