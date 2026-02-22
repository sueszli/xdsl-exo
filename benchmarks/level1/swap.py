from __future__ import annotations

from exo import *
from exoblas.blaslib import *
from exoblas.codegen_helpers import *


@proc
def swap(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        tmp: R
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp


variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(swap, "i", 4, globals=globals())
