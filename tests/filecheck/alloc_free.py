# RUN: uv run exojit --mlir %s | filecheck %s

# Scalar references use the allocation pointer directly, within the loop lifetime.
# CHECK-LABEL: llvm.func @alloc_free
# CHECK-SAME: (%[[N:[0-9]+]]: i64, %[[X:[0-9]+]]: !llvm.ptr)
# CHECK: ^[[HEADER:bb[0-9]+]](%[[I:[0-9]+]]: i64):
# CHECK: llvm.icmp "slt" %[[I]], %[[N]] : i64
# CHECK: %[[SIZE:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: %[[TMP:[0-9]+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
# CHECK: %[[XP:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: %[[V:[0-9]+]] = llvm.load %[[XP]] : !llvm.ptr -> f32
# CHECK-NEXT: llvm.store %[[V]], %[[TMP]] : f32, !llvm.ptr
# CHECK-NEXT: %[[COPY:[0-9]+]] = llvm.load %[[TMP]] : !llvm.ptr -> f32
# CHECK: %[[XP2:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[COPY]], %[[XP2]] : f32, !llvm.ptr
# CHECK: llvm.call @free(%[[TMP]]) : (!llvm.ptr) -> ()
# CHECK: llvm.br ^[[HEADER]]
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
