from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from kernels.matmul import matmul
from kernels.matmul_neon import matmul_neon
from kernels.saxpy import saxpy
from kernels.saxpy_neon import saxpy_neon
from kernels.softmax import _jit_max_neon, softmax
from kernels.softmax_neon import softmax_neon

REPEATS = 200


bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []


#
# matmul
#


matmul_sizes = [32, 64, 128, 256]


for n in tqdm(matmul_sizes, desc="matmul"):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    expected = A @ B
    flops = 2 * n**3

    # numpy
    t_np = bench(lambda A=A, B=B: A @ B)

    # exo auto-vectorized
    fn_exo = matmul(n, n, n)
    C_exo = np.zeros((n, n), dtype=np.float32)
    fn_exo(C_exo, A, B)
    assert np.allclose(C_exo, expected, atol=1e-3)
    t_exo = bench(lambda C=C_exo, A=A, B=B: fn_exo(C, A, B))

    # exo neon intrinsics
    fn_neon = matmul_neon(n, n, n)
    C_neon = np.zeros((n, n), dtype=np.float32)
    fn_neon(C_neon, A, B)
    assert np.allclose(C_neon, expected, atol=1e-3)
    t_neon = bench(lambda C=C_neon, A=A, B=B: fn_neon(C, A, B))

    rows.append(
        {
            "kernel": "matmul",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 1),
            "exo_gflops": round(flops / t_exo / 1e9, 1),
            "neon_gflops": round(flops / t_neon / 1e9, 1),
        }
    )


#
# saxpy (y += a*x)
#


saxpy_sizes = [4096, 16384, 65536, 262144]


for n in tqdm(saxpy_sizes, desc="saxpy"):
    x = np.random.randn(n).astype(np.float32)
    y_orig = np.random.randn(n).astype(np.float32)
    a_val = np.float32(2.5)
    a_arr = np.array([a_val], dtype=np.float32)
    expected = y_orig + a_val * x
    flops = 2 * n  # n multiplies + n adds

    # numpy
    t_np = bench(lambda y=y_orig.copy(), a=a_val, xv=x: np.add(a * xv, y, out=y))

    # exo auto-vectorized
    fn_exo = saxpy(n)
    y_test = y_orig.copy()
    fn_exo(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy exo wrong"
    t_exo = bench(lambda fn=fn_exo, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    # exo neon intrinsics
    fn_neon = saxpy_neon(n)
    y_test = y_orig.copy()
    fn_neon(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy neon wrong"
    t_neon = bench(lambda fn=fn_neon, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    rows.append(
        {
            "kernel": "saxpy",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# softmax (fused: exp(x-max) + sum + normalize)
#


softmax_sizes = [256, 1024, 4096, 16384]


for n in tqdm(softmax_sizes, desc="softmax"):
    inp = np.random.randn(n).astype(np.float32)
    out_np = np.empty(n, dtype=np.float32)
    tmp_np = np.empty(n, dtype=np.float32)

    # numpy reference: max + sub + exp + sum + div (pre-allocated temporaries)
    def numpy_softmax(x=inp, out=out_np, tmp=tmp_np):
        m = x.max()
        np.subtract(x, m, out=tmp)
        np.exp(tmp, out=out)
        s = out.sum()
        out *= 1.0 / s

    numpy_softmax()
    expected = out_np.copy()
    flops = 4 * n  # sub + exp + sum + div (4 ops per element, standard counting)

    t_np = bench(numpy_softmax)

    # exo auto-vectorized (jit max + fused exp/sum/normalize)
    fn_max, fn_core = softmax(n)
    out_exo = np.zeros(n, dtype=np.float32)
    mx = np.array([0.0], dtype=np.float32)
    fn_max(mx, inp)
    fn_core(out_exo, inp, mx)
    assert np.allclose(out_exo, expected, atol=1e-3), f"softmax exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"

    def bench_exo(fn_m=fn_max, fn_c=fn_core, out=out_exo, x=inp, mx=mx):
        fn_m(mx, x)
        fn_c(out, x, mx)

    t_exo = bench(bench_exo)

    # exo neon intrinsics (neon max + fused exp/sum/normalize with explicit neon)
    fn_neon = softmax_neon(n)
    fn_max_neon = _jit_max_neon(n)
    out_neon = np.zeros(n, dtype=np.float32)
    fn_max_neon(mx, inp)
    fn_neon(out_neon, inp, mx)
    assert np.allclose(out_neon, expected, atol=1e-3), f"softmax neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"

    def bench_neon(fn_m=fn_max_neon, fn=fn_neon, out=out_neon, x=inp, mx=mx):
        fn_m(mx, x)
        fn(out, x, mx)

    t_neon = bench(bench_neon)

    rows.append(
        {
            "kernel": "softmax",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


if __name__ == "__main__":
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("exo_gflops") / pl.col("numpy_gflops")).round(2).alias("exo_speedup"),
        (pl.col("neon_gflops") / pl.col("numpy_gflops")).round(2).alias("neon_speedup"),
    )
    with pl.Config(tbl_rows=-1):
        print(df)
    df.write_csv(Path(__file__).parent / "results.csv")
