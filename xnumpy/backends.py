from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from enum import Enum, auto
from functools import cache
from pathlib import Path
from typing import Any, Callable

import llvmlite.binding as llvm_binding
import numpy as np
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.core.LoopIR import LoopIR
from exo.stdlib.scheduling import rename
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp

from xnumpy.main import JITEmitter
from xnumpy.main import compile_procs as xdsl_compile_procs

_TYPES: dict[str, tuple[type, type]] = {
    "f16": (np.float16, ctypes.c_uint16),
    "f32": (np.float32, ctypes.c_float),
    "f64": (np.float64, ctypes.c_double),
    "i8": (np.int8, ctypes.c_int8),
    "ui8": (np.uint8, ctypes.c_uint8),
    "i16": (np.int16, ctypes.c_int16),
    "ui16": (np.uint16, ctypes.c_uint16),
    "i32": (np.int32, ctypes.c_int32),
}


@cache
def _find_llvm_bin() -> Path:
    # $LLVM_BIN > which mlir-opt > brew --prefix llvm
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    prefix = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()
    return Path(prefix) / "bin"


def _compile_exo_c(procs: list[Procedure]) -> ctypes.CDLL:
    # exo C codegen -> clang -shared -> .so
    d = Path(tempfile.mkdtemp())
    exo_compile_procs(procs, d, "o.c", "o.h")
    subprocess.run(["clang", "-shared", "-fPIC", "-O0", "-I", str(d), "-o", str(d / "lib.so"), str(d / "o.c")], check=True)
    return ctypes.CDLL(str(d / "lib.so"))


def _compile_xdsl_mlir(procs: list[Procedure]) -> ctypes.CDLL:
    # xdsl -> mlir-translate --mlir-to-llvmir -> clang -shared -> .so
    mlir_text = str(xdsl_compile_procs(procs))
    d = Path(tempfile.mkdtemp())
    mlir, so = d / "o.mlir", d / "lib.so"
    mlir.write_text(mlir_text)
    subprocess.run(f"{_find_llvm_bin()}/mlir-translate --mlir-to-llvmir {mlir} | clang -shared -x ir -o {so} -", shell=True, check=True)
    return ctypes.CDLL(str(so))


def _call(lib: ctypes.CDLL, proc_ir: Any, kwargs: dict[str, Any], *, has_ctxt: bool) -> dict[str, np.ndarray]:
    # marshal args from numpy/python into ctypes, call fn, return output buffers
    fn = getattr(lib, proc_ir.name)
    argtypes: list = []
    args: list = []
    bufs: dict[str, np.ndarray] = {}

    if has_ctxt:
        argtypes += [ctypes.c_void_p]
        args += [None]

    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]

        match arg.type:
            case LoopIR.Size() | LoopIR.Index():
                argtypes += [ctypes.c_long]
                args += [int(val)]
            case LoopIR.Tensor():
                np_dtype, c_type = _TYPES[str(arg.type.basetype())]
                arr = np.array(val, dtype=np_dtype)
                bufs[name] = arr
                argtypes += [ctypes.POINTER(c_type)]
                args += [arr.ctypes.data_as(ctypes.POINTER(c_type))]

    fn.argtypes = argtypes
    fn.restype = None
    fn(*args)
    return bufs


def _call_jit(fn: Callable[..., None], proc_ir: Any, kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    # marshal numpy kwargs into raw pointers and call a JIT-compiled function
    args: list = []
    bufs: dict[str, np.ndarray] = {}

    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]

        match arg.type:
            case LoopIR.Size() | LoopIR.Index():
                args += [int(val)]
            case LoopIR.Tensor():
                arr = np.array(val, dtype=_TYPES[str(arg.type.basetype())][0])
                bufs[name] = arr
                args += [arr.ctypes.data]

    fn(*args)
    return bufs


# 
# llvmlite stuff (confusing api, needs some rework)
# 


llvm_binding.initialize_native_target()
llvm_binding.initialize_native_asmprinter()


@cache
def _create_target_machine() -> llvm_binding.TargetMachine:
    # host cpu + features never change
    target = llvm_binding.Target.from_default_triple()
    cpu = llvm_binding.get_host_cpu_name()
    features = llvm_binding.get_host_cpu_features().flatten()
    return target.create_target_machine(cpu=cpu, features=features, opt=2)


def _optimize_module(llvm_mod: llvm_binding.ModuleRef, target_machine: llvm_binding.TargetMachine) -> None:
    # run llvm -O2 pass pipeline
    pto = llvm_binding.PipelineTuningOptions()
    pto.speed_level = 2
    pb = llvm_binding.create_pass_builder(target_machine, pto)
    pm = pb.getModulePassManager()
    pm.run(llvm_mod, pb)


def _lower(module: ModuleOp) -> tuple[llvm_binding.ModuleRef, llvm_binding.TargetMachine]:
    # xdsl module -> llvmlite ir -> parsed llvm module, optimized at -O2
    llvm_module = JITEmitter.emit(module)
    llvm_mod = llvm_binding.parse_assembly(str(llvm_module))
    target_machine = _create_target_machine()
    _optimize_module(llvm_mod, target_machine)
    return llvm_mod, target_machine


def jit_compile(module: ModuleOp) -> dict[str, ctypes._CFuncPtr]:
    # lower + jit. returns {name: cfunc} for each func with a body
    llvm_mod, target_machine = _lower(module)
    engine = llvm_binding.create_mcjit_compiler(llvm_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    fns: dict[str, ctypes._CFuncPtr] = {}
    for op in module.ops:
        if not isinstance(op, llvm.FuncOp) or not op.body.blocks:
            continue
        name = op.sym_name.data
        n = len(op.function_type.inputs)
        fn = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * n))(engine.get_function_address(name))
        fn._engine = engine
        fns[name] = fn
    return fns


def emit_assembly(module: ModuleOp) -> str:
    # lower + emit native assembly text (no repeat wrappers)
    llvm_mod, target_machine = _lower(module)
    return target_machine.emit_assembly(llvm_mod)


def compile_jit(proc: Procedure | Sequence[Procedure], name: str | Sequence[str]) -> Callable[..., None] | dict[str, Callable[..., None]]:
    # compile an exo procedure (+ any co-compiled procs) via JIT
    if isinstance(proc, Procedure):
        proc = [proc]
    single = isinstance(name, str)
    if single:
        name = [name]
    proc = [rename(p, n) for p, n in zip(proc, name)] + list(proc[len(name) :])
    fns = jit_compile(xdsl_compile_procs(proc))
    return fns[name[0]] if single else {n: fns[n] for n in name}


class Backend(Enum):
    EXO_C = auto()  # exo's native C codegen -> clang -> .so (reference)
    MLIR = auto()  # xdsl -> mlir-translate + clang -> .so
    JIT = auto()  # xdsl -> llvmlite JIT (in-memory)


def compile_and_load(proc: Procedure, backend: Backend) -> Callable[..., dict[str, np.ndarray]]:
    # compile a procedure and return a callable.
    # fn(**kwargs) -> {buffer_name: np.ndarray} with mutated output buffers.
    ir = proc._loopir_proc

    match backend:
        case Backend.EXO_C:
            lib = _compile_exo_c([proc])
            return lambda **kwargs: _call(lib, ir, deepcopy(kwargs), has_ctxt=True)

        case Backend.MLIR:
            lib = _compile_xdsl_mlir([proc])
            return lambda **kwargs: _call(lib, ir, deepcopy(kwargs), has_ctxt=False)

        case Backend.JIT:
            fn = compile_jit(proc, ir.name)
            return lambda **kwargs: _call_jit(fn, ir, deepcopy(kwargs))

        case _:
            assert False
