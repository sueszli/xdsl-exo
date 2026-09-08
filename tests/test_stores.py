from __future__ import annotations

import pytest
from exo import *
from exo.core.LoopIR import LoopIR
from xdsl.dialects import llvm, memref

from exojit import IRGenerator


@pytest.mark.parametrize("statement", [LoopIR.Assign, LoopIR.Reduce])
def test_store_evaluation_order(statement):
    @proc
    def indexed_store(out: f64[4, 4] @ DRAM, src: f64[1] @ DRAM, row: index, col: index):
        assert 0 <= row and row < 3
        assert 0 <= col and col < 3
        out[row + 1, col + 1] += src[0]

    ir = indexed_store._loopir_proc
    stmt = ir.body[0]
    stmt = statement(stmt.name, stmt.type, stmt.idx, stmt.rhs, stmt.srcinfo)
    module = IRGenerator().generate([ir.update(body=[stmt])])
    block = next(op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == ir.name).body.block
    ops = list(block.ops)
    loads = [op for op in ops if isinstance(op, memref.LoadOp)]
    store = next(op for op in ops if isinstance(op, memref.StoreOp))
    # Inspect before CSE: each index expression must be evaluated once, left to right, before the RHS.
    index_adds = [op for op in ops[: ops.index(loads[0])] if isinstance(op, llvm.AddOp)]
    assert len(index_adds) == 2
    assert [op.lhs for op in index_adds] == list(block.args[2:])
    for access in [*loads[1:], store]:
        assert [index.owner.inputs[0] for index in access.indices] == [op.res for op in index_adds]
