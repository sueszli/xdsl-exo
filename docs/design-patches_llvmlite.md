#### 2. Phi insertion logic is duplicated between `BrOp` and `CondBrOp` handlers (L36-49)

```python
case BrOp():
    cur = builder.block
    for a, v in zip(op.successor.args, op.operands):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    builder.branch(block_map[op.successor])

case CondBrOp():
    cur = builder.block
    for a, v in zip(op.successors[0].args, op.then_arguments):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    for a, v in zip(op.successors[1].args, op.else_arguments):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    builder.cbranch(...)
```

The 3-line `for a, v in zip(...): if a in phi_map: phi_map[a].add_incoming(...)` pattern appears three times (once in BrOp, twice in CondBrOp).

**Proposed change:** Extract a helper:

```python
def _add_phis(phi_map: PhiMap, val_map: ValMap, block_args, operands, cur_block):
    for a, v in zip(block_args, operands):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur_block)
```

Then:
```python
case BrOp():
    _add_phis(phi_map, val_map, op.successor.args, op.operands, builder.block)
    builder.branch(block_map[op.successor])

case CondBrOp():
    _add_phis(phi_map, val_map, op.successors[0].args, op.then_arguments, builder.block)
    _add_phis(phi_map, val_map, op.successors[1].args, op.else_arguments, builder.block)
    builder.cbranch(...)
```
