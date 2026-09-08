# RUN: uv run exojit --mlir %s | filecheck %s

# Row-major addressing is i * 4 + j, in element units.
# CHECK-LABEL: llvm.func @assign_2d
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr)
# CHECK: %[[WIDTH:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: ^{{bb[0-9]+}}(%[[I:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[I]], {{%[0-9]+}} : i64
# CHECK: ^{{bb[0-9]+}}(%[[J:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[J]], {{%[0-9]+}} : i64
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK: %[[ROW:[0-9]+]] = llvm.mul %[[I]], %[[WIDTH]] : i64
# CHECK-NEXT: %[[OFFSET:[0-9]+]] = llvm.add %[[ROW]], %[[J]] : i64
# CHECK-NEXT: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.store %[[ZERO]], %[[PTR]] : f32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def assign_2d(x: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            x[i, j] = 0.0
