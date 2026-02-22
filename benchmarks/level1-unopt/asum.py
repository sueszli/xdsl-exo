from __future__ import annotations

from exo import *
from exo.libs.externs import select
from exo.platforms.x86 import *


@proc
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])
