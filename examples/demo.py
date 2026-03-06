from __future__ import annotations

import ctypes

import numpy as np
from exo import *

from xdsl_exo.main import compile_procs
from xdsl_exo.patches_llvmlite import jit_compile, to_llvmlite


@proc
def add(n: size, a: f32[n] @ DRAM, b: f32[n] @ DRAM, out: f32[n] @ DRAM):
    for i in seq(0, n):
        out[i] = a[i] + b[i]


engine = jit_compile(to_llvmlite(compile_procs([add])))
fn = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(engine.get_function_address("add"))

n = 4
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([10, 20, 30, 40], dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

fn(n, a.ctypes.data, b.ctypes.data, out.ctypes.data)
print(out)
