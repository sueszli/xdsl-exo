# RUN: uv run xdsl-exo --target exo -o - %s | filecheck %s
# RUN: uv run xdsl-exo --target llvm -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @noop() {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def noop():
    pass
