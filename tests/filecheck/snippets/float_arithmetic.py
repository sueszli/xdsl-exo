# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @float_arithmetic(%0 : !llvm.ptr, %1 : !llvm.ptr, %2 : !llvm.ptr) {
# CHECK-NEXT:   %3 = arith.constant 0 : i64
# CHECK-NEXT:   %4 = "llvm.getelementptr"(%1, %3) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %5 = "llvm.load"(%4) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %6 = "llvm.getelementptr"(%2, %3) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %7 = "llvm.load"(%6) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %8 = arith.addf %5, %7 : f32
# CHECK:        %12 = arith.subf %10, %11 : f32
# CHECK:        %15 = arith.mulf %13, %14 : f32
# CHECK:        %18 = arith.divf %16, %17 : f32
@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
