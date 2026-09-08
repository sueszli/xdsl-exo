# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @index_modulo
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[I:[0-9]+]]: i64)
# CHECK: %[[TEN:[0-9]+]] = llvm.mlir.constant(10) : i64
# CHECK: %[[INDEX:[0-9]+]] = llvm.srem %[[I]], %[[TEN]] : i64
# CHECK: %[[VALUE:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][%[[INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT: llvm.store %[[VALUE]], %[[PTR]] : i32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def index_modulo(out: i32[10] @ DRAM, i: index):
    assert i >= 0
    out[i % 10] = 42
