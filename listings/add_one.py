from __future__ import annotations

from exo import *


@proc
def add_one(a: f32[16] @ DRAM):
    for i in seq(0, 16):
        a[i] += 1.0
