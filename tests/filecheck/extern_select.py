# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @uses_select
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: !llvm.ptr, %[[B:[0-9]+]]: !llvm.ptr)
# CHECK: %[[AP:[0-9]+]] = llvm.getelementptr inbounds %[[A]][{{%[0-9]+}}]
# CHECK: %[[AV:[0-9]+]] = llvm.load %[[AP]] : !llvm.ptr -> f32
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK: %[[AV2:[0-9]+]] = llvm.load %[[AP]] : !llvm.ptr -> f32
# CHECK: %[[BP:[0-9]+]] = llvm.getelementptr inbounds %[[B]][{{%[0-9]+}}]
# CHECK: %[[BV:[0-9]+]] = llvm.load %[[BP]] : !llvm.ptr -> f32
# CHECK: %[[COND:[0-9]+]] = llvm.fcmp "olt" %[[ZERO]], %[[AV]] : f32
# CHECK: %[[RESULT:[0-9]+]] = llvm.select %[[COND]], %[[AV2]], %[[BV]] : i1, f32
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[RESULT]], %[[OP]] : f32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *
from exo.platforms.x86 import *


@proc
def uses_select(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = select(0.0, a[0], a[0], b[0])
