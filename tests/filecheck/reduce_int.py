# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @reduce_int
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr, %[[OUT:[0-9]+]]: !llvm.ptr)
# CHECK: %[[N:[0-9]+]] = llvm.mlir.constant(8) : i64
# CHECK: ^[[HEADER:bb[0-9]+]](%[[I:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[I]], %[[N]] : i64
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0) : i64
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: %[[V:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i32
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][%[[ZERO]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: %[[CURRENT:[0-9]+]] = llvm.load %[[OP]] : !llvm.ptr -> i32
# CHECK: %[[SUM:[0-9]+]] = llvm.add %[[CURRENT]], %[[V]] : i32
# CHECK-NEXT: llvm.store %[[SUM]], %[[OP]] : i32, !llvm.ptr
# CHECK: llvm.br ^[[HEADER]]
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def reduce_int(x: i32[8] @ DRAM, out: i32[1] @ DRAM):
    for i in seq(0, 8):
        out[0] += x[i]
