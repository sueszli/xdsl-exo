# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @i8_copy
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[X:[0-9]+]]: !llvm.ptr)
# CHECK: %[[N:[0-9]+]] = llvm.mlir.constant(8) : i64
# CHECK: ^[[HEADER:bb[0-9]+]](%[[I:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[I]], %[[N]] : i64
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
# CHECK: %[[V:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> i8
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
# CHECK: llvm.store %[[V]], %[[OP]] : i8, !llvm.ptr
# CHECK: llvm.br ^[[HEADER]]
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def i8_copy(out: i8[8] @ DRAM, x: i8[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = x[i]
