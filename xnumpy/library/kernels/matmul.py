from __future__ import annotations

from collections.abc import Callable

from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify, unroll_loop

from xnumpy.library.kernels.jit import _jit


def _schedule_matmul(n: int) -> Callable[..., object]:
    def transform(p: object) -> object:
        p = fission(p, p.find("for k in _: _").before(), n_lifts=2)  # type: ignore[union-attr]
        p = reorder_loops(p, "j k")
        if n % 4 == 0:
            p = divide_loop(p, "j #1", 4, ["jo", "ji"], perfect=True)
            p = unroll_loop(p, "ji")
        p = simplify(p)
        return p

    return transform


def matmul(m: int, k: int, n: int) -> Callable[..., None]:
    name = f"_mm_{m}_{k}_{n}"
    return _jit(
        f"""@proc
def {name}(C: f32[{m},{n}] @ DRAM, A: f32[{m},{k}] @ DRAM, B: f32[{k},{n}] @ DRAM):
    for i in seq(0, {m}):
        for j in seq(0, {n}):
            C[i,j] = 0.0
            for k in seq(0, {k}):
                C[i,j] += A[i,k] * B[k,j]
""",
        name,
        transform=_schedule_matmul(n),
    )
