from __future__ import annotations

from exo import *
from exoblas.blaslib import *
from exoblas.codegen_helpers import *


@proc
def copy(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] = x[i]


variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(copy, "i", 4, globals=globals())
