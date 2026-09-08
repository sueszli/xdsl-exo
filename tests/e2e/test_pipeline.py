from __future__ import annotations

import numpy as np
import pytest
from exo import *
from exo.libs.externs import expf

from exojit import FPTruncOp, jit, to_mlir


@proc
def expf_f64(out: f64[4] @ DRAM, src: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = expf(src[i])


@pytest.mark.parametrize("raw", [False, True])
def test_repeated_lowering_and_jit_with_custom_llvm_op(raw):
    module = to_mlir(expf_f64)
    module.verify()
    assert any(isinstance(op, FPTruncOp) for op in module.walk())
    functions = [jit(expf_f64, raw=raw) for _ in range(2)]
    repeated = to_mlir(expf_f64)
    assert repeated is not module
    assert str(repeated) == str(module)
    src = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float64)
    for fn in functions:
        out = np.zeros_like(src) if raw else [0.0] * 4
        fn(out, src if raw else src.tolist())
        np.testing.assert_allclose(out, np.exp(src.astype(np.float32)), rtol=1e-6)
