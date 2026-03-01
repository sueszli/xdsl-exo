# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @alloc_free(%0 : i64, %1 : !llvm.ptr) {
# CHECK-NEXT:   %c0 = arith.constant 0 : i64
# CHECK-NEXT:   %2 = arith.constant 1 : i64
# CHECK-NEXT:   cf.br ^bb0(%c0 : i64)
# CHECK-NEXT: ^bb0(%3 : i64):
# CHECK-NEXT:   %4 = arith.cmpi slt, %3, %0 : i64
# CHECK-NEXT:   cf.cond_br %4, ^bb1, ^bb2
# CHECK-NEXT: ^bb1:
# CHECK-NEXT:   %5 = arith.constant 1 : i64
# CHECK-NEXT:   %6 = "llvm.call"(%5) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %7 = "llvm.getelementptr"(%1, %3) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %8 = "llvm.load"(%7) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %9 = arith.constant 0 : i64
# CHECK-NEXT:   %10 = "llvm.getelementptr"(%6, %9) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%8, %10) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   %11 = arith.constant 0 : i64
# CHECK-NEXT:   %12 = "llvm.getelementptr"(%6, %11) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %13 = "llvm.load"(%12) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %14 = "llvm.getelementptr"(%1, %3) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%13, %14) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   "llvm.call"(%6) <{callee = @free, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   %15 = arith.addi %3, %2 : i64
# CHECK-NEXT:   cf.br ^bb0(%15 : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
