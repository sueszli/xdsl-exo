from __future__ import annotations

from exo import *

# write through a window, read back from the parent buffer.
# window = alias, not a copy. lowered as:
#   WindowExpr -> memref.subview -> ptr arithmetic -> llvm.getelementptr
#
# only correct for contiguous slices (row windows). column windows like
# A[:, j] have stride > 1, but we lower to a flat pointer and lose the
# stride — callee indexes with stride 1 and reads wrong memory. exo's C
# backend passes struct { data*, strides[] } to fix this. we'd need
# strided memref types to do the same.


@proc
def write_via_window(row: [f32][4] @ DRAM):
    row[2] = 42.0


@proc
def window_roundtrip(A: f32[4, 4] @ DRAM, out: f32[1] @ DRAM):
    write_via_window(A[1, :])  # write 42.0 into A[1,2] through a window
    out[0] = A[1, 2]  # read back from parent -> should be 42.0
