from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename
from kernels.add_neon import neon_loadu_f32x4, neon_storeu_f32x4

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


@cache
def embedding_neon(d: int) -> Callable[..., None]:
    assert d % 16 == 0, "D must be divisible by 16 for 4x unrolled f32x4"
    d16 = d // 16

    @proc
    def _embedding_neon(out: f32[d] @ DRAM, row: f32[d] @ DRAM):
        for i in seq(0, d16):
            v0: f32[4] @ NEON
            v1: f32[4] @ NEON
            v2: f32[4] @ NEON
            v3: f32[4] @ NEON
            neon_loadu_f32x4(v0, row[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(v1, row[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(v2, row[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(v3, row[16 * i + 12 : 16 * i + 16])
            neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], v0)
            neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], v1)
            neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], v2)
            neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], v3)

    name = f"_embedding_neon_{d}"
    p = rename(_embedding_neon, name)
    return compile_jit(p)[name]
