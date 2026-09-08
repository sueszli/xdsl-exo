# RUN: uv run exojit --mlir %s | filecheck %s

# Scalar storage is an ordinary pointer, not a one-element temporary memref.
# CHECK-LABEL: llvm.func @assign_from_scalar_memref
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr)
# CHECK: %[[SIZE:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: %[[TMP:[0-9]+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
# CHECK: %[[VALUE:[0-9]+]] = llvm.mlir.constant(4.200000e+01 : f32) : f32
# CHECK-NEXT: llvm.store %[[VALUE]], %[[TMP]] : f32, !llvm.ptr
# CHECK: ^{{bb[0-9]+}}(%[[I:[0-9]+]]: i64):
# CHECK: %[[V:[0-9]+]] = llvm.load %[[TMP]] : !llvm.ptr -> f32
# CHECK-NEXT: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.store %[[V]], %[[PTR]] : f32, !llvm.ptr
# CHECK: llvm.call @free(%[[TMP]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def assign_from_scalar_memref(x: f32[8] @ DRAM):
    tmp: f32
    tmp = 42.0
    for i in seq(0, 8):
        x[i] = tmp
