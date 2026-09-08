# RUN: uv run exojit --mlir %s | filecheck %s

# Contiguous views pass just their origin pointer; callees remain standalone functions.
# CHECK-LABEL: llvm.func @set_first
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr)
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[ZERO]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[ONE]], %[[XP]] : f32, !llvm.ptr
# CHECK: llvm.return
# CHECK-LABEL: llvm.func @inner
# CHECK-SAME: (%[[INNER:[0-9]+]]: !llvm.ptr)
# CHECK: %[[WIDTH:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: %[[ROW:[0-9]+]] = llvm.mlir.constant(1) : i64
# CHECK: %[[COL:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[R:[0-9]+]] = llvm.mul %[[ROW]], %[[WIDTH]] : i64
# CHECK: %[[OFFSET:[0-9]+]] = llvm.add %[[R]], %[[COL]] : i64
# CHECK: %[[IP:[0-9]+]] = llvm.getelementptr inbounds %[[INNER]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.call @set_first(%[[IP]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return
# CHECK-LABEL: llvm.func @outer
# CHECK-SAME: (%[[OUTER:[0-9]+]]: !llvm.ptr)
# CHECK: %[[W:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: %[[PLANE:[0-9]+]] = llvm.mlir.constant(2) : i64
# CHECK: %[[Z:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[P:[0-9]+]] = llvm.mul %[[PLANE]], %[[W]] : i64
# CHECK: %[[PR:[0-9]+]] = llvm.add %[[P]], %[[Z]] : i64
# CHECK: %[[PC:[0-9]+]] = llvm.mul %[[PR]], %[[W]] : i64
# CHECK: %[[OFF:[0-9]+]] = llvm.add %[[PC]], %[[Z]] : i64
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUTER]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.call @inner(%[[OP]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def set_first(x: [f32][4] @ DRAM):
    x[0] = 1.0


@proc
def inner(A: [f32][4, 4] @ DRAM):
    set_first(A[1, :])


@proc
def outer(A: f32[4, 4, 4] @ DRAM):
    inner(A[2, :, :])
