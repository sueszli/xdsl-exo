# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @conv1d(%0 : i64, %1 : i64, %2 : i64, %3 : i64, %4 : !llvm.ptr, %5 : !llvm.ptr, %6 : !llvm.ptr) {
# CHECK-NEXT:   %c0 = arith.constant 0 : i64
# CHECK-NEXT:   %7 = arith.constant 1 : i64
# CHECK-NEXT:   %8 = arith.constant 0 : i32
# CHECK-NEXT:   cf.br ^bb0(%c0 : i64)
# CHECK-NEXT: ^bb0(%9 : i64):
# CHECK-NEXT:   %10 = arith.cmpi slt, %9, %1 : i64
# CHECK-NEXT:   cf.cond_br %10, ^bb1(%c0 : i64), ^bb2
# CHECK-NEXT: ^bb1(%11 : i64):
# CHECK-NEXT:   %12 = arith.cmpi slt, %11, %2 : i64
# CHECK-NEXT:   cf.cond_br %12, ^bb3, ^bb4
# CHECK-NEXT: ^bb3:
# CHECK-NEXT:   %13 = arith.muli %9, %2 : i64
# CHECK-NEXT:   %14 = arith.addi %13, %11 : i64
# CHECK-NEXT:   %15 = "llvm.getelementptr"(%6, %14) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%8, %15) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   cf.br ^bb5(%c0 : i64)
# CHECK-NEXT: ^bb5(%16 : i64):
# CHECK-NEXT:   %17 = arith.cmpi slt, %16, %0 : i64
# CHECK-NEXT:   cf.cond_br %17, ^bb6(%c0 : i64), ^bb7
# CHECK-NEXT: ^bb6(%18 : i64):
# CHECK-NEXT:   %19 = arith.cmpi slt, %18, %3 : i64
# CHECK-NEXT:   cf.cond_br %19, ^bb8, ^bb9
# CHECK-NEXT: ^bb8:
# CHECK-NEXT:   %20 = arith.constant 1 : i64
# CHECK-NEXT:   %21 = "llvm.call"(%20) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %22 = arith.addi %11, %18 : i64
# CHECK-NEXT:   %23 = arith.cmpi slt, %22, %2 : i64
# CHECK-NEXT:   cf.cond_br %23, ^bb10, ^bb11
# CHECK-NEXT: ^bb10:
# CHECK-NEXT:   %24 = arith.muli %16, %2 : i64
# CHECK-NEXT:   %25 = arith.addi %24, %22 : i64
# CHECK-NEXT:   %26 = "llvm.getelementptr"(%4, %25) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %27 = "llvm.load"(%26) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %28 = arith.constant 0 : i64
# CHECK-NEXT:   %29 = "llvm.getelementptr"(%21, %28) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%27, %29) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   cf.br ^bb12
# CHECK-NEXT: ^bb11:
# CHECK-NEXT:   %30 = arith.constant 0 : i64
# CHECK-NEXT:   %31 = "llvm.getelementptr"(%21, %30) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%8, %31) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   cf.br ^bb12
# CHECK-NEXT: ^bb12:
# CHECK-NEXT:   %32 = arith.muli %3, %0 : i64
# CHECK-NEXT:   %33 = arith.muli %9, %32 : i64
# CHECK-NEXT:   %34 = arith.muli %16, %3 : i64
# CHECK-NEXT:   %35 = arith.addi %33, %34 : i64
# CHECK-NEXT:   %36 = arith.addi %35, %18 : i64
# CHECK-NEXT:   %37 = "llvm.getelementptr"(%5, %36) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %38 = "llvm.load"(%37) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %39 = arith.constant 0 : i64
# CHECK-NEXT:   %40 = "llvm.getelementptr"(%21, %39) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %41 = "llvm.load"(%40) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %42 = arith.muli %38, %41 : i32
# CHECK-NEXT:   %43 = arith.muli %9, %2 : i64
# CHECK-NEXT:   %44 = arith.addi %43, %11 : i64
# CHECK-NEXT:   %45 = "llvm.getelementptr"(%6, %44) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   %46 = "llvm.load"(%45) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %47 = arith.addi %46, %42 : i32
# CHECK-NEXT:   %48 = arith.muli %9, %2 : i64
# CHECK-NEXT:   %49 = arith.addi %48, %11 : i64
# CHECK-NEXT:   %50 = "llvm.getelementptr"(%6, %49) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%47, %50) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   "llvm.call"(%21) <{callee = @free, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   %51 = arith.addi %18, %7 : i64
# CHECK-NEXT:   cf.br ^bb6(%51 : i64)
# CHECK-NEXT: ^bb9:
# CHECK-NEXT:   %52 = arith.addi %16, %7 : i64
# CHECK-NEXT:   cf.br ^bb5(%52 : i64)
# CHECK-NEXT: ^bb7:
# CHECK-NEXT:   %53 = arith.addi %11, %7 : i64
# CHECK-NEXT:   cf.br ^bb1(%53 : i64)
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   %54 = arith.addi %9, %7 : i64
# CHECK-NEXT:   cf.br ^bb0(%54 : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def conv1d(
    IC: size,
    OC: size,
    N: size,
    W: size,
    data: i32[IC, N],
    kernels: i32[OC, IC, W],
    out: i32[OC, N],
):
    for i in seq(0, OC):
        for j in seq(0, N):
            out[i, j] = 0
            for c in seq(0, IC):
                for r in seq(0, W):
                    y: i32
                    if j + r < N:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
