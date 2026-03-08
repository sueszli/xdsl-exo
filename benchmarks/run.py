from __future__ import annotations

import timeit

import numpy as np
import polars as pl

import xnumpy as xnp
from xnumpy.backends import JIT_CACHE
from xnumpy.library.kernels.elementwise import add, mul, neg

REPEATS = 50
BATCH = 1000

bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []

EW_SIZES = [64, 256, 1024, 4096, 16384, 65536]
MM_SIZES = [32, 64, 128, 256]

for n in EW_SIZES:
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    a_xnp = xnp.array(a_np)
    b_xnp = xnp.array(b_np)

    assert xnp.allclose(a_xnp + b_xnp, a_np + b_np)
    assert xnp.allclose(a_xnp * b_xnp, a_np * b_np)
    assert xnp.allclose(-a_xnp, -a_np)

    # trigger compilation and get repeat wrappers
    add(n)
    mul(n)
    neg(n)
    add_r = JIT_CACHE[f"_add_{n}_repeat"]
    mul_r = JIT_CACHE[f"_mul_{n}_repeat"]
    neg_r = JIT_CACHE[f"_neg_{n}_repeat"]

    # pre-allocate output for kernel-only benchmark
    out = np.empty(n, dtype=np.float32)
    op, ap, bp = out.ctypes.data, a_np.ctypes.data, b_np.ctypes.data

    for op_name, np_fn, xnp_fn in [
        ("add", lambda a=a_np, b=b_np: a + b, lambda o=op, a=ap, b=bp, r=add_r: r(o, a, b, BATCH)),
        ("mul", lambda a=a_np, b=b_np: a * b, lambda o=op, a=ap, b=bp, r=mul_r: r(o, a, b, BATCH)),
        ("neg", lambda a=a_np: -a, lambda o=op, a=ap, r=neg_r: r(o, a, BATCH)),
    ]:
        t_np = bench(np_fn)
        t_xnp = bench(xnp_fn) / BATCH
        rows.append({"op": op_name, "n": n, "numpy_us": t_np * 1e6, "xnumpy_us": t_xnp * 1e6, "speedup": t_np / t_xnp})

for n in MM_SIZES:
    A_np = np.random.randn(n, n).astype(np.float32)
    B_np = np.random.randn(n, n).astype(np.float32)
    A_xnp = xnp.array(A_np)
    B_xnp = xnp.array(B_np)

    assert xnp.allclose(A_xnp @ B_xnp, A_np @ B_np, atol=1e-3)

    t_np = bench(lambda A=A_np, B=B_np: A @ B)
    t_xnp = bench(lambda A=A_xnp, B=B_xnp: A @ B)
    rows.append({"op": "matmul", "n": n, "numpy_us": t_np * 1e6, "xnumpy_us": t_xnp * 1e6, "speedup": t_np / t_xnp})

df = pl.DataFrame(rows)
df = df.with_columns(pl.col("numpy_us").round(1), pl.col("xnumpy_us").round(1), pl.col("speedup").round(2))
print(df)
df.write_csv("benchmarks/results.csv")
