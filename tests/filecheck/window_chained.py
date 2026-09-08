# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @inner(%0: !llvm.ptr) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %4 = llvm.mul %1, %3 : i64
# CHECK-NEXT:     %5 = llvm.mul %1, %4 : i64
# CHECK-NEXT:     %6 = llvm.mul %2, %1 : i64
# CHECK-NEXT:     %7 = llvm.add %5, %6 : i64
# CHECK-NEXT:     %8 = llvm.mul %7, %3 : i64
# CHECK-NEXT:     %9 = llvm.ptrtoint %0 : !llvm.ptr to i64
# CHECK-NEXT:     %10 = llvm.add %9, %8 : i64
# CHECK-NEXT:     %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @set_first(%11) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @outer(%12: !llvm.ptr) {
# CHECK-NEXT:     %13 = llvm.mlir.constant(2) : i64
# CHECK-NEXT:     %14 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %15 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %16 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %17 = llvm.mul %15, %16 : i64
# CHECK-NEXT:     %18 = llvm.mul %17, %16 : i64
# CHECK-NEXT:     %19 = llvm.mul %13, %18 : i64
# CHECK-NEXT:     %20 = llvm.mul %14, %17 : i64
# CHECK-NEXT:     %21 = llvm.add %19, %20 : i64
# CHECK-NEXT:     %22 = llvm.mul %14, %15 : i64
# CHECK-NEXT:     %23 = llvm.add %21, %22 : i64
# CHECK-NEXT:     %24 = llvm.mul %23, %16 : i64
# CHECK-NEXT:     %25 = llvm.ptrtoint %12 : !llvm.ptr to i64
# CHECK-NEXT:     %26 = llvm.add %25, %24 : i64
# CHECK-NEXT:     %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @inner(%27) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @set_first(%28: !llvm.ptr) {
# CHECK-NEXT:     %29 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %30 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %31 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %32 = llvm.mul %29, %31 : i64
# CHECK-NEXT:     %33 = llvm.getelementptr inbounds %28[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %30, %33 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def set_first(x: [f32][4] @ DRAM):
    x[0] = 1.0


@proc
def inner(A: [f32][4, 4] @ DRAM):
    set_first(A[1, :])


@proc
def outer(A: f32[4, 4, 4] @ DRAM):
    inner(A[2, :, :])
