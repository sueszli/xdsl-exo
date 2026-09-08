# RUN: uv run exojit --mlir %s | filecheck %s

# Contiguous row origins use i * 4 + 0; no descriptor or pointer-to-integer roundtrip.
# CHECK-LABEL: llvm.func @set_row
# CHECK-SAME: (%[[ROW:[0-9]+]]: !llvm.ptr)
# CHECK: ^{{bb[0-9]+}}(%[[J:[0-9]+]]: i64):
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK: %[[RP:[0-9]+]] = llvm.getelementptr inbounds %[[ROW]][%[[J]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[ZERO]], %[[RP]] : f32, !llvm.ptr
# CHECK: llvm.return
# CHECK-LABEL: llvm.func @window_row
# CHECK-SAME: (%[[A:[0-9]+]]: !llvm.ptr)
# CHECK: %[[WIDTH:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: ^{{bb[0-9]+}}(%[[I:[0-9]+]]: i64):
# CHECK: %[[COL:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[R:[0-9]+]] = llvm.mul %[[I]], %[[WIDTH]] : i64
# CHECK: %[[OFFSET:[0-9]+]] = llvm.add %[[R]], %[[COL]] : i64
# CHECK: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[A]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.call @set_row(%[[PTR]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def set_row(row: [f32][4] @ DRAM):
    for i in seq(0, 4):
        row[i] = 0.0


@proc
def window_row(A: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        set_row(A[i, :])
