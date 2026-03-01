# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import DRAM, proc


# CHECK: builtin.module {
# CHECK-NEXT:   func.func @fixed_matmul(%0 : !llvm.ptr, %1 : !llvm.ptr, %2 : !llvm.ptr) {
# CHECK-NEXT:     %3 = arith.constant 0 : i64
# CHECK-NEXT:     %4 = arith.constant 16 : i64
# CHECK-NEXT:     %5 = arith.constant 1 : i64
# CHECK-NEXT:     %6 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     cf.br ^bb0(%3 : i64)
# CHECK-NEXT:   ^bb0(%7 : i64):
# CHECK-NEXT:     %8 = arith.cmpi slt, %7, %4 : i64
# CHECK-NEXT:     cf.cond_br %8, ^bb1(%3 : i64), ^bb2
# CHECK-NEXT:   ^bb1(%9 : i64):
# CHECK-NEXT:     %10 = arith.cmpi slt, %9, %4 : i64
# CHECK-NEXT:     cf.cond_br %10, ^bb3, ^bb4
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %11 = arith.constant 16 : i64
# CHECK-NEXT:     %12 = arith.muli %7, %11 : i64
# CHECK-NEXT:     %13 = arith.addi %12, %9 : i64
# CHECK-NEXT:     %14 = "llvm.getelementptr"(%0, %13) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"(%6, %14) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     cf.br ^bb5(%3 : i64)
# CHECK-NEXT:   ^bb5(%15 : i64):
# CHECK-NEXT:     %16 = arith.cmpi slt, %15, %4 : i64
# CHECK-NEXT:     cf.cond_br %16, ^bb6, ^bb7
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %17 = arith.constant 16 : i64
# CHECK-NEXT:     %18 = arith.muli %7, %17 : i64
# CHECK-NEXT:     %19 = arith.addi %18, %15 : i64
# CHECK-NEXT:     %20 = "llvm.getelementptr"(%1, %19) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     %21 = "llvm.load"(%20) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %22 = arith.constant 16 : i64
# CHECK-NEXT:     %23 = arith.muli %15, %22 : i64
# CHECK-NEXT:     %24 = arith.addi %23, %9 : i64
# CHECK-NEXT:     %25 = "llvm.getelementptr"(%2, %24) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     %26 = "llvm.load"(%25) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %27 = arith.mulf %21, %26 : f32
# CHECK-NEXT:     %28 = arith.constant 16 : i64
# CHECK-NEXT:     %29 = arith.muli %7, %28 : i64
# CHECK-NEXT:     %30 = arith.addi %29, %9 : i64
# CHECK-NEXT:     %31 = "llvm.getelementptr"(%0, %30) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     %32 = "llvm.load"(%31) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %33 = arith.addf %32, %27 : f32
# CHECK-NEXT:     %34 = arith.constant 16 : i64
# CHECK-NEXT:     %35 = arith.muli %7, %34 : i64
# CHECK-NEXT:     %36 = arith.addi %35, %9 : i64
# CHECK-NEXT:     %37 = "llvm.getelementptr"(%0, %36) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"(%33, %37) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     %38 = arith.addi %15, %5 : i64
# CHECK-NEXT:     cf.br ^bb5(%38 : i64)
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %39 = arith.addi %9, %5 : i64
# CHECK-NEXT:     cf.br ^bb1(%39 : i64)
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %40 = arith.addi %7, %5 : i64
# CHECK-NEXT:     cf.br ^bb0(%40 : i64)
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]
