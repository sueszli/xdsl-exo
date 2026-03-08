from __future__ import annotations

import linecache
from collections.abc import Callable

from xnumpy.main import compile_procs
from xnumpy.patches_llvmlite import jit_compile

_cache: dict[str, Callable[..., None]] = {}

# names injected into every @proc exec namespace (NEON intrinsics + memory type)
_neon_ns: dict[str, object] | None = None


def _get_neon() -> dict[str, object]:
    global _neon_ns
    if _neon_ns is None:
        import xnumpy.library.kernels.neon as _neon_mod

        _neon_ns = {
            name: getattr(_neon_mod, name)
            for name in dir(_neon_mod)
            if name.startswith("neon_") or name == "NEON"
        }
    return _neon_ns


def _jit(code: str, name: str, transform: Callable[..., object] | None = None) -> Callable[..., None]:
    if name not in _cache:
        code = code.lstrip("\n")
        ns: dict[str, object] = {}
        exec("from exo import *", ns)
        ns.update(_get_neon())
        filename = f"<xnumpy:{name}>"
        compiled = compile(code, filename, "exec")
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        exec(compiled, ns)
        p = ns[name]
        if transform is not None:
            p = transform(p)
        fns = jit_compile(compile_procs(p))
        _cache[name] = fns[name]
        _cache[f"{name}_repeat"] = fns[f"{name}_repeat"]
    return _cache[name]
