# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_comparisons(%0 : !llvm.ptr, %1 : i64, %2 : i64) {
# CHECK-NEXT:   %3 = arith.cmpi eq, %1, %2 : i64
# CHECK-NEXT:   cf.cond_br %3, ^bb0, ^bb1
# CHECK:        %7 = arith.cmpi slt, %1, %2 : i64
# CHECK-NEXT:   cf.cond_br %7, ^bb2, ^bb3
# CHECK:        %11 = arith.cmpi sgt, %1, %2 : i64
# CHECK-NEXT:   cf.cond_br %11, ^bb4, ^bb5
@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
