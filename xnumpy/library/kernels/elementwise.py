from __future__ import annotations

from collections.abc import Callable

from exo.stdlib.scheduling import divide_loop, replace, simplify

from xnumpy.library.kernels.jit import _jit
from xnumpy.library.kernels.neon import (
    neon_vadd_f32x4,
    neon_vmul_f32x4,
    neon_vneg_f32x4,
    neon_vsub_f32x4,
)


# ---------------------------------------------------------------------------
# scheduling transform — vectorize via DRAM-accepting intrinsics
# ---------------------------------------------------------------------------


def _schedule_vec(intrinsic: object) -> Callable[..., object]:
    def transform(p: object) -> object:
        p = divide_loop(p, "i", 4, ["io", "ii"], perfect=True)
        p = replace(p, "for ii in _: _", intrinsic)
        p = simplify(p)
        return p

    return transform


# ---------------------------------------------------------------------------
# code templates — scalar
# ---------------------------------------------------------------------------


def _scalar_binary(n: int, name: str, expr: str) -> str:
    return f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = {expr}
"""


def _scalar_unary(n: int, name: str, expr: str) -> str:
    return f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = {expr}
"""


def _scalar_with_scalar(n: int, name: str, expr: str) -> str:
    return f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        out[i] = {expr}
"""


def _scalar_inplace_binary(n: int, name: str, expr: str) -> str:
    return f"""@proc
def {name}(a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        a[i] = {expr}
"""


def _scalar_inplace_scalar(n: int, name: str, expr: str) -> str:
    return f"""@proc
def {name}(a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        a[i] = {expr}
"""


# ---------------------------------------------------------------------------
# kernel factories
# ---------------------------------------------------------------------------


def _make_binary(prefix: str, expr: str, vec_intrinsic: object) -> Callable[[int], Callable[..., None]]:
    def kernel(n: int) -> Callable[..., None]:
        name = f"_{prefix}_{n}"
        code = _scalar_binary(n, name, expr)
        transform = _schedule_vec(vec_intrinsic) if n % 4 == 0 else None
        return _jit(code, name, transform=transform)

    return kernel


def _make_unary(prefix: str, expr: str, vec_intrinsic: object) -> Callable[[int], Callable[..., None]]:
    def kernel(n: int) -> Callable[..., None]:
        name = f"_{prefix}_{n}"
        code = _scalar_unary(n, name, expr)
        transform = _schedule_vec(vec_intrinsic) if n % 4 == 0 else None
        return _jit(code, name, transform=transform)

    return kernel


def _make_scalar_op(prefix: str, expr: str) -> Callable[[int], Callable[..., None]]:
    def kernel(n: int) -> Callable[..., None]:
        return _jit(_scalar_with_scalar(n, f"_{prefix}_{n}", expr), f"_{prefix}_{n}")

    return kernel


def _make_inplace_binary(prefix: str, expr: str, vec_intrinsic: object) -> Callable[[int], Callable[..., None]]:
    def kernel(n: int) -> Callable[..., None]:
        name = f"_{prefix}_{n}"
        code = _scalar_inplace_binary(n, name, expr)
        transform = _schedule_vec(vec_intrinsic) if n % 4 == 0 else None
        return _jit(code, name, transform=transform)

    return kernel


def _make_inplace_scalar(prefix: str, expr: str) -> Callable[[int], Callable[..., None]]:
    def kernel(n: int) -> Callable[..., None]:
        return _jit(_scalar_inplace_scalar(n, f"_{prefix}_{n}", expr), f"_{prefix}_{n}")

    return kernel


# ---------------------------------------------------------------------------
# public kernels
# ---------------------------------------------------------------------------

add = _make_binary("add", "a[i] + b[i]", neon_vadd_f32x4)
sub = _make_binary("sub", "a[i] - b[i]", neon_vsub_f32x4)
mul = _make_binary("mul", "a[i] * b[i]", neon_vmul_f32x4)
neg = _make_unary("neg", "-a[i]", neon_vneg_f32x4)

scalar_add = _make_scalar_op("sadd", "a[i] + s[0]")
scalar_sub = _make_scalar_op("ssub", "a[i] - s[0]")
scalar_mul = _make_scalar_op("smul", "a[i] * s[0]")
scalar_rsub = _make_scalar_op("srsub", "s[0] - a[i]")

iadd = _make_inplace_binary("iadd", "a[i] + b[i]", neon_vadd_f32x4)
isub = _make_inplace_binary("isub", "a[i] - b[i]", neon_vsub_f32x4)
imul = _make_inplace_binary("imul", "a[i] * b[i]", neon_vmul_f32x4)

iscalar_add = _make_inplace_scalar("isadd", "a[i] + s[0]")
iscalar_sub = _make_inplace_scalar("issub", "a[i] - s[0]")
iscalar_mul = _make_inplace_scalar("ismul", "a[i] * s[0]")


def sum_reduce(n: int) -> Callable[..., None]:
    name = f"_sum_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[1] @ DRAM, a: f32[{n}] @ DRAM):
    out[0] = 0.0
    for i in seq(0, {n}):
        out[0] += a[i]
""",
        name,
    )
