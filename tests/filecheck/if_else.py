# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @if_else
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: i64, %[[B:[0-9]+]]: i64)
# CHECK: %[[COND:[0-9]+]] = llvm.icmp "slt" %[[A]], %[[B]] : i64
# CHECK: llvm.cond_br %[[COND]], ^[[TRUE:bb[0-9]+]], ^[[FALSE:bb[0-9]+]]
# CHECK: ^[[TRUE]]:
# CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK: %[[P1:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[ONE]], %[[P1]] : f32, !llvm.ptr
# CHECK: llvm.br ^[[MERGE:bb[0-9]+]]
# CHECK: ^[[FALSE]]:
# CHECK: %[[TWO:[0-9]+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK: %[[P2:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[TWO]], %[[P2]] : f32, !llvm.ptr
# CHECK: llvm.br ^[[MERGE]]
# CHECK: ^[[MERGE]]:
# CHECK-NEXT: llvm.return


from __future__ import annotations

from exo import *


@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0
