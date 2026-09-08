# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @int_comparisons
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: i64, %[[B:[0-9]+]]: i64)
# CHECK: %[[EQ:[0-9]+]] = llvm.icmp "eq" %[[A]], %[[B]] : i64
# CHECK: llvm.cond_br %[[EQ]], ^[[EQTRUE:bb[0-9]+]], ^{{bb[0-9]+}}
# CHECK: ^[[EQTRUE]]:
# CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
# CHECK: %[[P1:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[ONE]], %[[P1]] : i32, !llvm.ptr
# CHECK: %[[LT:[0-9]+]] = llvm.icmp "slt" %[[A]], %[[B]] : i64
# CHECK: llvm.cond_br %[[LT]], ^[[LTTRUE:bb[0-9]+]], ^{{bb[0-9]+}}
# CHECK: ^[[LTTRUE]]:
# CHECK: %[[TWO:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
# CHECK: %[[P2:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[TWO]], %[[P2]] : i32, !llvm.ptr
# CHECK: %[[GT:[0-9]+]] = llvm.icmp "sgt" %[[A]], %[[B]] : i64
# CHECK: llvm.cond_br %[[GT]], ^[[GTTRUE:bb[0-9]+]], ^{{bb[0-9]+}}
# CHECK: ^[[GTTRUE]]:
# CHECK: %[[THREE:[0-9]+]] = llvm.mlir.constant(3 : i32) : i32
# CHECK: %[[P3:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}]
# CHECK: llvm.store %[[THREE]], %[[P3]] : i32, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
