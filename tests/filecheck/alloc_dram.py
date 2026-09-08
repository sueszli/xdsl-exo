# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @alloc_dram
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr)
# CHECK: %[[SIZE:[0-9]+]] = llvm.mlir.constant(16) : i64
# CHECK: %[[TMP:[0-9]+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[ZERO]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[V:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[TP:[0-9]+]] = llvm.getelementptr inbounds %[[TMP]][%[[ZERO]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[V]], %[[TP]] : f32, !llvm.ptr
# CHECK: %[[COPY:[0-9]+]] = llvm.load %[[TP]] : !llvm.ptr -> f32
# CHECK: llvm.store %[[COPY]], %[[XP]] : f32, !llvm.ptr
# CHECK: llvm.call @free(%[[TMP]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return
# CHECK: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK: llvm.func @free(!llvm.ptr)


from __future__ import annotations

from exo import *


@proc
def alloc_dram(x: f32[8] @ DRAM):
    tmp: f32[4]
    tmp[0] = x[0]
    x[0] = tmp[0]
