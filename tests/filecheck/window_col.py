# RUN: uv run exojit --mlir %s | filecheck %s

# Legacy lowering-only fixture: A[:, j] passes its origin, not a column stride.
# The bare-pointer ABI assumes contiguous windows. This is NOT an execution
# oracle for strided columns, which remain unsupported (no hidden stride args).
# CHECK-LABEL: llvm.func @set_col
# CHECK-SAME: (%[[COL:[0-9]+]]: !llvm.ptr)
# CHECK: ^{{bb[0-9]+}}(%[[I:[0-9]+]]: i64):
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK: %[[CP:[0-9]+]] = llvm.getelementptr inbounds %[[COL]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[ZERO]], %[[CP]] : f32, !llvm.ptr
# CHECK: llvm.return
# CHECK-LABEL: llvm.func @window_col
# CHECK-SAME: (%[[A:[0-9]+]]: !llvm.ptr)
# CHECK: %[[WIDTH:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: ^{{bb[0-9]+}}(%[[J:[0-9]+]]: i64):
# CHECK: %[[ROW:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[R:[0-9]+]] = llvm.mul %[[ROW]], %[[WIDTH]] : i64
# CHECK: %[[OFFSET:[0-9]+]] = llvm.add %[[R]], %[[J]] : i64
# CHECK: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[A]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.call @set_col(%[[PTR]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def set_col(col: [f32][4] @ DRAM):
    for i in seq(0, 4):
        col[i] = 0.0


@proc
def window_col(A: f32[4, 4] @ DRAM):
    for j in seq(0, 4):
        set_col(A[:, j])
