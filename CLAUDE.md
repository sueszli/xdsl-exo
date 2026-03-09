# xnumpy

JIT compiler for numerical kernels. Exo DSL procedures are lowered through xDSL MLIR to LLVM machine code via llvmlite, then executed directly from Python with ~1us FFI overhead.

## Compilation pipeline

```
Exo @proc (Python DSL)
  → LoopIR (Exo's internal IR)
  → xDSL MLIR (LLVM dialect)          # IRGenerator in main.py
  → llvmlite IR (LLVM text)           # LLVMLiteGenerator in main.py
  → LLVM MCJIT (native ARM/x86)       # llvmlite.binding
  → JitFunc (Python callable)         # jitcall.c FFI bridge
```

Key entry point: `compile_jit(proc)` in `xnumpy/main.py` — takes an Exo `Procedure`, returns `dict[str, callable]`.

IR is disk-cached at `.cache/xnumpy/{hash}/` keyed by a hash of all `xnumpy/*.py` files. Changing any compiler source auto-invalidates the cache. Delete `.cache/xnumpy/` to force recompilation.

## Project structure

```
xnumpy/
  main.py                    # Core: IRGenerator, LLVMLiteGenerator, compile_jit, CLI
  patches_xdsl_llvm.py       # Custom xDSL ops (FCmpOp, SelectOp, BrOp, CondBrOp,
                              #   VectorFMaxOp) + memref→ptr lowering patterns
  patches_xdsl_intrinsics.py # vec_*/neon_* intrinsic handlers (ConvertVecIntrinsic)
  patches_exo.py             # Exo memory classes: Stack (alloca), NEON (register)
  jitcall.c                  # C extension: JitFunc with vectorcall, buffer protocol
benchmarks/
  run.py                     # Benchmark runner (matmul, saxpy, softmax)
  kernels/                   # Kernel implementations (auto-vec + NEON variants)
tests/
  e2e/                       # 142 end-to-end functional tests
  filecheck/                 # ~26 MLIR/asm inspection tests (lit + FileCheck)
  _utils.py                  # Test helpers (compile_exo, compile_mlir)
  conftest.py                # GC disabled to avoid xDSL pointer issues
```

## Commands

```bash
.venv/bin/python -m pytest tests/           # Run all 142 tests
.venv/bin/python -m pytest tests/ -x -q     # Stop on first failure, quiet
.venv/bin/python benchmarks/run.py          # Run full benchmark suite
make benchmark                              # Same as above via Makefile
```

Use `.venv/bin/python` — system python3 lacks numpy and other deps.

## Writing a new kernel

### Auto-vectorized (let LLVM generate SIMD)

```python
from exo import *
from exo.stdlib.scheduling import rename, simplify
from xnumpy.main import compile_jit
from xnumpy.patches_exo import Stack

@proc
def _my_kernel(N: size, out: f32[N], inp: f32[N]):
    for i in seq(0, N):
        out[i] = inp[i] * 2.0

@cache
def my_kernel(n: int) -> Callable[..., None]:
    p = _my_kernel.partial_eval(N=n)
    p = simplify(p)
    name = f"_my_kernel_{n}"
    return compile_jit(rename(p, name))[name]

# Usage: fn = my_kernel(256); fn(out_array, inp_array)
```

LLVM will auto-vectorize the inner loop if: (1) all float ops have fast-math, (2) pointer args are noalias, (3) target triple is set. All three are handled by the pipeline automatically.

### NEON intrinsics (explicit SIMD)

1. Define the Exo instruction with `@instr`. The Python function name becomes the xDSL call name and **must match** the handler key in `patches_xdsl_intrinsics.py`:

```python
from xnumpy.patches_exo import NEON

@instr("neon_my_op_f32x4({dst_data}, {src_data});")
def neon_my_op_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i] * src[i]  # Semantic body (reference, not executed)
```

2. Add the xDSL lowering handler in `patches_xdsl_intrinsics.py`:

```python
def _build_neon_my_op(dst, src, *, vec_type):
    load = llvm.LoadOp(src, vec_type)
    result = llvm.FMulOp(load.dereferenced_value, load.dereferenced_value)
    return (load, result, llvm.StoreOp(result.res, dst))

# In _make_intrinsics():
entries["neon_my_op_f32x4"] = lambda args: _build_neon_my_op(*args, vec_type=_F32X4)
```

3. If you need a new xDSL op type (like `VectorFMaxOp` for `llvm.maxnum`), define it in `patches_xdsl_llvm.py`, register it in `_context()`, and add a case in `LLVMLiteGenerator._convert_op()`.

### Existing NEON intrinsics available

Defined across kernel files and lowered in `patches_xdsl_intrinsics.py`:

- `neon_loadu_f32x4(dst, src)` — load 4 floats from DRAM to NEON register
- `neon_storeu_f32x4(dst, src)` — store NEON register to DRAM
- `neon_broadcast_f32x4(dst, scalar_ptr)` — splat scalar to all 4 lanes
- `neon_fmadd_f32x4(acc, a, b)` — acc += a * b (fused multiply-add)
- `neon_add_f32x4(dst, a, b)` / `neon_sub_f32x4` / `neon_mul_f32x4`
- `neon_square_f32x4(dst, src)` — dst = src * src (avoids aliasing issue)
- `neon_add_acc_f32x4(acc, src)` — acc += src
- `neon_fmax_acc_f32x4(acc, src)` — acc = max(acc, src) via `llvm.maxnum.v4f32`
- f64x2 variants: `neon_loadu_f64x2`, `neon_fmadd_f64x2`, `neon_broadcast_f64x2`, etc.

### Scheduling (Exo stdlib)

```python
from exo.stdlib.scheduling import *

p = divide_loop(p, "i", 64, ["io", "ii"])  # Tile loop
p = reorder_loops(p, "j k")                # Swap loop order
p = unroll_loop(p, "ii")                   # Unroll (use sparingly!)
p = fission(p, p.find("for k in _: _").before(), n_lifts=2)
p = simplify(p)
p = rename(p, "new_name")
```

### Memory spaces

- `@ DRAM` — heap allocated (malloc/free). Use for input/output arrays.
- `@ Stack` — stack allocated (alloca). Use for scalar temporaries — enables LLVM register promotion. Defined in `patches_exo.py`.
- `@ NEON` — NEON vector register (float32x4_t / float64x2_t). Use for explicit SIMD. Defined in `patches_exo.py`.

## Critical gotchas

### Exo language limitations

- **No float comparisons in `if`**: Exo's `if` only supports index/size types. For float branching, use the `select` extern: `select(a, b, c, d)` returns `c` if `a < b`, else `d`. Import from `exo.libs.externs`.
- **Aliasing checker**: Exo rejects `neon_mul_f32x4(dst, x, x)` — same buffer in multiple args. Create a dedicated intrinsic instead (e.g., `neon_square_f32x4(dst, src)`).
- **Read-modify-write in Horner chains**: `e = e * t + c` can produce wrong results due to aliasing. Use separate variables for each step: `e5 = ...; e4 = e5 * y + c; e3 = e4 * y + c`.

### Intrinsic naming

The Python function name of an `@instr` becomes the xDSL `llvm.call` callee name. It **must exactly match** the key in `_make_intrinsics()` in `patches_xdsl_intrinsics.py`. If it doesn't match, the call passes through as an unresolved external function and segfaults at runtime.

### Auto-vectorization killers

- **`unroll_loop` on the vectorizable dimension**: Creates explicit scalar ops that prevent LLVM's loop vectorizer from emitting SIMD. Remove the unroll and let LLVM vectorize automatically. Only unroll non-vectorizable dimensions.
- **`ptrtoint`/`inttoptr` for scalar access**: Breaks LLVM alias analysis. Use GEP (`getelementptr`) for scalar load/store. The pipeline uses `_offset_ptr_gep` for loads/stores and `_offset_ptr_raw` (ptrtoint) only for subviews.
- **Missing fast-math on `fcmp`**: Without `fcmp fast olt`, LLVM can't recognize min/max reduction patterns. The pipeline adds `flags=("fast",)` to all fcmp instructions.
- **Scalar max reduction**: Even with fast-math, LLVM doesn't auto-vectorize `acc = select(acc, x[i], x[i], acc)` loops. Use explicit NEON `neon_fmax_acc_f32x4` with multiple accumulators for ILP.

### Combining select-based max with other loops

A select-based max loop combined with an exp loop in the same JIT kernel produces NaN at n >= 256 due to an LLVM optimization interaction. Split them into separate JIT kernels (separate `compile_jit` calls).

## Performance optimization checklist

1. **Fuse operations**: Single-pass kernels avoid intermediate memory traffic. The softmax kernel fuses exp+sum+normalize into one loop.
2. **Use Stack memory for temporaries**: `@ Stack` uses alloca, enabling LLVM register promotion. `@ DRAM` uses malloc which prevents it.
3. **Tile for cache**: For matmul-like kernels, bk=64/bj=64 keeps B-tiles (64x64x4=16KB) in L1. Reorder inner loops to maximize spatial locality.
4. **ILP with multiple accumulators**: Use 2-4 independent accumulator registers to hide pipeline latency (e.g., NEON max with 2x4-wide accumulators processes 8 floats/iter).
5. **Don't unroll the SIMD dimension**: Let LLVM's loop vectorizer handle it. Only use Exo scheduling for tiling and loop reordering.
6. **Polynomial approximations**: For transcendentals (exp, log), use range reduction + short polynomial. Example: `exp(x) = exp(x/32)^32` with degree-5 Taylor for `exp(x/32)` gives ~1e-6 relative error.

## Adding a new xDSL op

If an intrinsic needs an LLVM operation not available in xDSL (like `llvm.maxnum`):

1. Define the op in `patches_xdsl_llvm.py` (follow `VectorFMaxOp` pattern)
2. Register it in `_context()` in `main.py`: `ctx.load_op(MyNewOp)`
3. Import it in `main.py`
4. Add a case in `LLVMLiteGenerator._convert_op()` to emit the llvmlite IR (follow the `vector.FMAOp` → `llvm.fma` pattern for intrinsic calls)
5. Use it in intrinsic handlers in `patches_xdsl_intrinsics.py`

## Debugging

- **Dump llvmlite IR**: Read the cached `.ll` file from `.cache/xnumpy/{hash}/{kernel_name}.ll`
- **Dump optimized LLVM IR**: Use `llvmlite.binding.parse_assembly(ir)` → optimize → `str(mod_ref)`
- **Check vectorization**: Look for `<4 x float>` operations, `fmla.4s` in assembly, or `llvm.maxnum.v4f32` calls in optimized IR
- **Segfault debugging**: Usually means an intrinsic name doesn't match a handler (unresolved external call) or a buffer size mismatch

## Plugins

Prefer LSP over Grep/Glob/Read for code navigation:
- `goToDefinition` / `goToImplementation` to jump to source
- `findReferences` to see all usages across the codebase
- `workspaceSymbol` to find where something is defined
- `documentSymbol` to list all symbols in a file
- `hover` for type info without reading the file
- `incomingCalls` / `outgoingCalls` for call hierarchy

Before renaming or changing a function signature, use `findReferences` to find all call sites first.

Use Grep/Glob only for text/pattern searches (comments, strings, config values) where LSP doesn't help.

After writing or editing code, check LSP diagnostics before moving on. Fix any type errors or missing imports immediately. Note: Pyright will flag false positives on Exo DSL code (`@proc`, `f32`, `size`, `seq`, `stride`, `@instr`) — these are valid Exo syntax, not Python type errors.
