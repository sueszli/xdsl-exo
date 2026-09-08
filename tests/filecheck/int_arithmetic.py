# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @int_arithmetic
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[X:[0-9]+]]: !llvm.ptr, %[[Y:[0-9]+]]: !llvm.ptr)
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: %[[XV:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i32
# CHECK: %[[YP:[0-9]+]] = llvm.getelementptr inbounds %[[Y]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: %[[YV:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> i32
# CHECK: %[[ADD:[0-9]+]] = llvm.add %[[XV]], %[[YV]] : i32
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: llvm.store %[[ADD]], %[[OP]] : i32, !llvm.ptr
# CHECK: %[[X2:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i32
# CHECK: %[[Y2:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> i32
# CHECK: %[[SUB:[0-9]+]] = llvm.sub %[[X2]], %[[Y2]] : i32
# CHECK: llvm.store %[[SUB]], %[[OP]] : i32, !llvm.ptr
# CHECK: %[[X3:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i32
# CHECK: %[[Y3:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> i32
# CHECK: %[[MUL:[0-9]+]] = llvm.mul %[[X3]], %[[Y3]] : i32
# CHECK: llvm.store %[[MUL]], %[[OP]] : i32, !llvm.ptr
# CHECK: %[[X4:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i32
# CHECK: %[[Y4:[0-9]+]] = llvm.load %[[YP]] : !llvm.ptr -> i32
# CHECK: %[[DIV:[0-9]+]] = llvm.sdiv %[[X4]], %[[Y4]] : i32
# CHECK: llvm.store %[[DIV]], %[[OP]] : i32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def int_arithmetic(out: i32[1] @ DRAM, x: i32[1] @ DRAM, y: i32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
