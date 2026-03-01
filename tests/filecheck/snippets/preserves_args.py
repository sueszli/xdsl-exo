# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @preserves_args(%0 : !llvm.ptr, %1 : i64) {
# CHECK-NEXT:   %2 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:   %3 = "llvm.getelementptr"(%0, %1) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%2, %3) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
