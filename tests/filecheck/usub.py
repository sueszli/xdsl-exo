# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @usub_float
# CHECK-SAME: (%[[OF:[0-9]+]]: !llvm.ptr, %[[XF:[0-9]+]]: !llvm.ptr)
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[XF]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[V:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK: %[[NEG:[0-9]+]] = llvm.fneg %[[V]] {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OF]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[NEG]], %[[OP]] : f32, !llvm.ptr
# CHECK: llvm.return
# CHECK-LABEL: llvm.func @usub_int
# CHECK-SAME: (%[[OI:[0-9]+]]: !llvm.ptr, %[[XI:[0-9]+]]: !llvm.ptr)
# CHECK: %[[XIP:[0-9]+]] = llvm.getelementptr inbounds %[[XI]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: %[[IV:[0-9]+]] = llvm.load %[[XIP]] : !llvm.ptr -> i32
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
# CHECK: %[[INEG:[0-9]+]] = llvm.sub %[[ZERO]], %[[IV]] : i32
# CHECK: %[[OIP:[0-9]+]] = llvm.getelementptr inbounds %[[OI]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: llvm.store %[[INEG]], %[[OIP]] : i32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def usub_float(out: f32[1] @ DRAM, x: f32[1] @ DRAM):
    out[0] = -x[0]


@proc
def usub_int(out: i32[1] @ DRAM, x: i32[1] @ DRAM):
    out[0] = -x[0]
