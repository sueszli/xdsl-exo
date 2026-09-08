# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK-LABEL: llvm.func @f64_arithmetic
# CHECK-SAME: (%[[OUT:[0-9]+]]: !llvm.ptr, %[[A:[0-9]+]]: !llvm.ptr, %[[B:[0-9]+]]: !llvm.ptr)
# CHECK: %[[AP:[0-9]+]] = llvm.getelementptr inbounds %[[A]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK: %[[AV:[0-9]+]] = llvm.load %[[AP]] : !llvm.ptr -> f64
# CHECK: %[[BP:[0-9]+]] = llvm.getelementptr inbounds %[[B]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK: %[[BV:[0-9]+]] = llvm.load %[[BP]] : !llvm.ptr -> f64
# CHECK: %[[SUM:[0-9]+]] = llvm.fadd %[[AV]], %[[BV]] {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK: %[[OP:[0-9]+]] = llvm.getelementptr inbounds %[[OUT]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK: llvm.store %[[SUM]], %[[OP]] : f64, !llvm.ptr
# CHECK: %[[AV2:[0-9]+]] = llvm.load %[[AP]] : !llvm.ptr -> f64
# CHECK: %[[BV2:[0-9]+]] = llvm.load %[[BP]] : !llvm.ptr -> f64
# CHECK: %[[PRODUCT:[0-9]+]] = llvm.fmul %[[AV2]], %[[BV2]] {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK: llvm.store %[[PRODUCT]], %[[OP]] : f64, !llvm.ptr
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] * b[0]
