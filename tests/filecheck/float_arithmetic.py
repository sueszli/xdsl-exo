# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @float_arithmetic
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[X:[0-9]+]]: !llvm.ptr, %[[Y:[0-9]+]]: !llvm.ptr)
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[XV:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[YP:[0-9]+]] = llvm.getelementptr inbounds %[[Y]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[YV:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> f32
# CHECK: %[[ADD:[0-9]+]] = llvm.fadd %[[XV]], %[[YV]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[ADD]], %[[OP]] : f32, !llvm.ptr
# CHECK: %[[X2:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[Y2:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> f32
# CHECK: %[[SUB:[0-9]+]] = llvm.fsub %[[X2]], %[[Y2]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: llvm.store %[[SUB]], %[[OP]] : f32, !llvm.ptr
# CHECK: %[[X3:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[Y3:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> f32
# CHECK: %[[MUL:[0-9]+]] = llvm.fmul %[[X3]], %[[Y3]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: llvm.store %[[MUL]], %[[OP]] : f32, !llvm.ptr
# CHECK: %[[X4:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[Y4:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> f32
# CHECK: %[[DIV:[0-9]+]] = llvm.fdiv %[[X4]], %[[Y4]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: llvm.store %[[DIV]], %[[OP]] : f32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def float_arithmetic(out: f32[1] @ DRAM, x: f32[1] @ DRAM, y: f32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
