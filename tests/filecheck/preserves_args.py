# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @preserves_args
# CHECK-SAME: (%[[X:[0-9]+]]: !llvm.ptr, %[[INDEX:[0-9]+]]: i64)
# CHECK: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT: %[[PTR:[0-9]+]] = llvm.getelementptr inbounds %[[X]][%[[INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT: llvm.store %[[ZERO]], %[[PTR]] : f32, !llvm.ptr
# CHECK-NEXT: llvm.return


from __future__ import annotations

from exo import *


@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
