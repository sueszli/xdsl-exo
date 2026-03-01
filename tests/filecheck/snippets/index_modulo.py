# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @index_modulo(%0 : !llvm.ptr, %1 : i64) {
# CHECK-NEXT:   %2 = arith.constant 10 : i64
# CHECK-NEXT:   %3 = arith.remsi %1, %2 : i64
# CHECK-NEXT:   %4 = arith.constant 42 : i32
# CHECK-NEXT:   %5 = "llvm.getelementptr"(%0, %3) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%4, %5) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
@proc
def index_modulo(out: i32[10] @ DRAM, n: index):
    assert n >= 0
    assert n < 10
    out[n % 10] = 42
