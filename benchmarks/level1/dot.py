from __future__ import annotations

from exo import *
from exoblas.blaslib import *
from exoblas.codegen_helpers import *


@proc
def dot(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]


variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(dot, "i", 4, globals=globals())
