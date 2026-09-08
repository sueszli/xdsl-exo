# RUN: uv run exojit --mlir %s | filecheck %s

# Check the three row-major addresses and the reduction, not SSA numbering.
# CHECK-LABEL: llvm.func @fixed_matmul
# CHECK-SAME: (%[[C:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: !llvm.ptr, %[[B:[0-9]+]]: !llvm.ptr)
# CHECK: %[[WIDTH:[0-9]+]] = llvm.mlir.constant(16) : i64
# CHECK: %[[AW:[0-9]+]] = llvm.mlir.constant(16) : i64
# CHECK: %[[BW:[0-9]+]] = llvm.mlir.constant(16) : i64
# CHECK: ^{{bb[0-9]+}}(%[[I:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[I]], {{%[0-9]+}} : i64
# CHECK: ^{{bb[0-9]+}}(%[[J:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[J]], {{%[0-9]+}} : i64
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK: %[[ROW:[0-9]+]] = llvm.mul %[[I]], %[[WIDTH]] : i64
# CHECK: %[[IJ:[0-9]+]] = llvm.add %[[ROW]], %[[J]] : i64
# CHECK: %[[CP:[0-9]+]] = llvm.getelementptr inbounds %[[C]][%[[IJ]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[ZERO]], %[[CP]] : f32, !llvm.ptr
# CHECK: ^{{bb[0-9]+}}(%[[K:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[K]], {{%[0-9]+}} : i64
# CHECK: %[[AROW:[0-9]+]] = llvm.mul %[[I]], %[[AW]] : i64
# CHECK: %[[IK:[0-9]+]] = llvm.add %[[AROW]], %[[K]] : i64
# CHECK: %[[AP:[0-9]+]] = llvm.getelementptr inbounds %[[A]][%[[IK]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[AV:[0-9]+]] = llvm.load %[[AP]] : !llvm.ptr -> f32
# CHECK: %[[BROW:[0-9]+]] = llvm.mul %[[K]], %[[BW]] : i64
# CHECK: %[[KJ:[0-9]+]] = llvm.add %[[BROW]], %[[J]] : i64
# CHECK: %[[BP:[0-9]+]] = llvm.getelementptr inbounds %[[B]][%[[KJ]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[BV:[0-9]+]] = llvm.load %[[BP]] : !llvm.ptr -> f32
# CHECK: %[[PRODUCT:[0-9]+]] = llvm.fmul %[[AV]], %[[BV]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: %[[CROW:[0-9]+]] = llvm.mul %[[I]], %[[WIDTH]] : i64
# CHECK: %[[CIJ:[0-9]+]] = llvm.add %[[CROW]], %[[J]] : i64
# CHECK: %[[CP2:[0-9]+]] = llvm.getelementptr inbounds %[[C]][%[[CIJ]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[CURRENT:[0-9]+]] = llvm.load %[[CP2]] : !llvm.ptr -> f32
# CHECK: %[[SUM:[0-9]+]] = llvm.fadd %[[CURRENT]], %[[PRODUCT]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT: llvm.store %[[SUM]], %[[CP2]] : f32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]
