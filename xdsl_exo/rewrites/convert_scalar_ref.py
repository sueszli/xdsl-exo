from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import FunctionType, IntegerAttr, MemRefType, ModuleOp, NoneAttr, i64
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo


class ConvertAssignToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        if not isinstance(op.input.type, MemRefType) or op.input.type.get_shape() != (1,) or len(op.indices) != 0:
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, i64)),
                exo.AssignOp(op.value, op.input, [zero_op.result], [1]),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertScalarFuncArgsToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        for idx, arg in enumerate(op.args):
            # ignore tensor types
            if isinstance(arg.type, MemRefType):
                continue

            mutated = any(isinstance(use.operation, exo.AssignOp) and use.operation.input == arg for use in arg.uses)

            # ignore unmutated scalar types, these can stay as is
            if not mutated:
                continue

            func_type = FunctionType.from_lists(
                (
                    *(arg.type for arg in op.args[:idx]),
                    MemRefType(arg.type, [1], NoneAttr()),
                    *(arg.type for arg in op.args[idx + 1 :]),
                ),
                op.function_type.outputs,
            )

            # rewrite function signature
            body = op.detach_region(op.body)
            new_arg = rewriter.insert_block_argument(body.block, idx, (MemRefType(arg.type, [1], NoneAttr())))
            rewriter.replace_all_uses_with(
                arg,
                new_arg,
            )
            rewriter.erase_block_argument(arg, idx)

            rewriter.replace_matched_op(func.FuncOp(op.sym_name.data, func_type, body, op.sym_visibility))


class ConvertScalarRefPass(ModulePass):
    name = "convert-scalar-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertAssignToTensor(),
                    ConvertScalarFuncArgsToTensor(),
                ]
            ),
        ).rewrite_module(m)
