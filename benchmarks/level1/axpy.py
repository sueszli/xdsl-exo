from __future__ import annotations

from exo import *
from exoblas.blaslib import *
from exoblas.codegen_helpers import *


@proc
def axpy(n: size, alpha: R, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] += alpha * x[i]


variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(axpy, "i", 8, globals=globals())
