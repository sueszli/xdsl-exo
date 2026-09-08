# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @bool_ops
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: i64, %[[B:[0-9]+]]: i64, %[[C:[0-9]+]]: i64)
# CHECK: %[[AB:[0-9]+]] = llvm.icmp "slt" %[[A]], %[[B]] : i64
# CHECK: %[[BC:[0-9]+]] = llvm.icmp "slt" %[[B]], %[[C]] : i64
# CHECK: %[[AND:[0-9]+]] = llvm.and %[[AB]], %[[BC]] : i1
# CHECK: llvm.cond_br %[[AND]], ^[[TRUE:bb[0-9]+]], ^{{bb[0-9]+}}
# CHECK: ^[[TRUE]]:
# CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK: %[[P1:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[ONE]], %[[P1]] : f32, !llvm.ptr
# CHECK: %[[AB2:[0-9]+]] = llvm.icmp "slt" %[[A]], %[[B]] : i64
# CHECK: %[[BC2:[0-9]+]] = llvm.icmp "slt" %[[B]], %[[C]] : i64
# CHECK: %[[OR:[0-9]+]] = llvm.or %[[AB2]], %[[BC2]] : i1
# CHECK: llvm.cond_br %[[OR]], ^[[TRUE2:bb[0-9]+]], ^{{bb[0-9]+}}
# CHECK: ^[[TRUE2]]:
# CHECK: %[[TWO:[0-9]+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK: %[[P2:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[TWO]], %[[P2]] : f32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    if a < b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
