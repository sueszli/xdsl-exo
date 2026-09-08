# RUN: uv run exojit --mlir %s | filecheck %s

# Both scalars occupy four bytes; element types belong to loads/stores, not pointers.
# CHECK-LABEL: llvm.func @multi_type_alloc
# CHECK-SAME: (%[[OUTF:[0-9]+]]: !llvm.ptr, %[[OUTI:[0-9]+]]: !llvm.ptr)
# CHECK: %[[SIZE:[0-9]+]] = llvm.mlir.constant(4) : i64
# CHECK: %[[F:[0-9]+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
# CHECK: %[[I:[0-9]+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
# CHECK: %[[FV:[0-9]+]] = llvm.mlir.constant(3.140000e+00 : f32) : f32
# CHECK-NEXT: llvm.store %[[FV]], %[[F]] : f32, !llvm.ptr
# CHECK: %[[IV:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT: llvm.store %[[IV]], %[[I]] : i32, !llvm.ptr
# CHECK: %[[FL:[0-9]+]] = llvm.load %[[F]] : !llvm.ptr -> f32
# CHECK: %[[FP:[0-9]+]] = llvm.getelementptr inbounds %[[OUTF]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK: llvm.store %[[FL]], %[[FP]] : f32, !llvm.ptr
# CHECK: llvm.call @free(%[[F]]) : (!llvm.ptr) -> ()
# CHECK: %[[IL:[0-9]+]] = llvm.load %[[I]] : !llvm.ptr -> i32
# CHECK: %[[IP:[0-9]+]] = llvm.getelementptr inbounds %[[OUTI]][{{%[0-9]+}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK: llvm.store %[[IL]], %[[IP]] : i32, !llvm.ptr
# CHECK: llvm.call @free(%[[I]]) : (!llvm.ptr) -> ()
# CHECK: llvm.return


from __future__ import annotations

from exo import *


@proc
def multi_type_alloc(out_f: f32[1] @ DRAM, out_i: i32[1] @ DRAM):
    tmp_f: f32
    tmp_i: i32
    tmp_f = 3.14
    tmp_i = 42
    out_f[0] = tmp_f
    out_i[0] = tmp_i
