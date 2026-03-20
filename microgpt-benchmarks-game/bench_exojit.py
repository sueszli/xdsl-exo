# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

from __future__ import annotations

import ctypes
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from exo.stdlib.scheduling import simplify
from microgpt_kernels import BLOCK_SIZE, N_EMBED, attn_bwd_fused, attn_fwd_fused, build_token_kernels, mlp_bwd_fused, mlp_fwd_fused
from utils.exo_alloc import Tensor, alloc_layout, empty, full, view_layout, zeros_like
from utils.exo_kernels import adam
from utils.times import save_times
from utils.weights import assert_weights_match

from exojit.main import jit

N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD


@dataclass(frozen=True)
class TokenBatch:
    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray
    inv_sum_mask: np.ndarray


@dataclass(frozen=True)
class Params:
    wte: Tensor
    wpe: Tensor
    lm_head: Tensor
    attn_wq: Tensor
    attn_wk: Tensor
    attn_wv: Tensor
    attn_wo: Tensor
    mlp_fc1: Tensor
    mlp_fc2: Tensor


@dataclass(frozen=True)
class Scratch:
    emb: Tensor
    rms_init: Tensor
    x0: Tensor
    x1: Tensor
    logits: Tensor
    attn_xn: Tensor
    attn_rms: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    attn_w: Tensor
    out_flat: Tensor
    mlp_xn: Tensor
    mlp_rms: Tensor
    h_pre: Tensor
    h: Tensor
    dx0: Tensor
    dx1: Tensor
    dattn_out: Tensor


@dataclass(frozen=True)
class Scalars:
    opt_lr: Tensor
    opt_bc1: Tensor
    opt_bc2: Tensor
    zero: Tensor
    one: Tensor
    rms_inv_n: Tensor
    rms_eps: Tensor
    adam_b1: Tensor
    adam_b2: Tensor
    adam_eps: Tensor


def init_normal_(tensor: Tensor, *, scale: float) -> None:
    flat = tensor.view((tensor.numel,))
    for i in range(tensor.numel):
        flat[i] = random.gauss(0.0, scale)


def param_layout(vocab_size: int) -> dict[str, tuple[int, ...]]:
    return {
        "wte": (vocab_size, N_EMBED),
        "wpe": (BLOCK_SIZE, N_EMBED),
        "lm_head": (vocab_size, N_EMBED),
        "attn_wq": (N_EMBED, N_EMBED),
        "attn_wk": (N_EMBED, N_EMBED),
        "attn_wv": (N_EMBED, N_EMBED),
        "attn_wo": (N_EMBED, N_EMBED),
        "mlp_fc1": (4 * N_EMBED, N_EMBED),
        "mlp_fc2": (N_EMBED, 4 * N_EMBED),
    }


def scratch_layout(vocab_size: int) -> dict[str, tuple[int, ...]]:
    return {
        "emb": (BLOCK_SIZE, N_EMBED),
        "rms_init": (BLOCK_SIZE, 1),
        "x0": (BLOCK_SIZE, N_EMBED),
        "x1": (BLOCK_SIZE, N_EMBED),
        "logits": (BLOCK_SIZE, vocab_size),
        "attn_xn": (BLOCK_SIZE, N_EMBED),
        "attn_rms": (BLOCK_SIZE, 1),
        "q": (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        "k": (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        "v": (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        "attn_w": (N_HEAD, BLOCK_SIZE, BLOCK_SIZE),
        "out_flat": (BLOCK_SIZE, N_EMBED),
        "mlp_xn": (BLOCK_SIZE, N_EMBED),
        "mlp_rms": (BLOCK_SIZE, 1),
        "h_pre": (BLOCK_SIZE, 4 * N_EMBED),
        "h": (BLOCK_SIZE, 4 * N_EMBED),
        "dx0": (BLOCK_SIZE, N_EMBED),
        "dx1": (BLOCK_SIZE, N_EMBED),
        "dattn_out": (N_HEAD, BLOCK_SIZE, HEAD_DIM),
    }


def bind_params(views: dict[str, Tensor]) -> Params:
    return Params(
        wte=views["wte"],
        wpe=views["wpe"],
        lm_head=views["lm_head"],
        attn_wq=views["attn_wq"],
        attn_wk=views["attn_wk"],
        attn_wv=views["attn_wv"],
        attn_wo=views["attn_wo"],
        mlp_fc1=views["mlp_fc1"],
        mlp_fc2=views["mlp_fc2"],
    )


def bind_scratch(views: dict[str, Tensor]) -> Scratch:
    return Scratch(
        emb=views["emb"],
        rms_init=views["rms_init"],
        x0=views["x0"],
        x1=views["x1"],
        logits=views["logits"],
        attn_xn=views["attn_xn"],
        attn_rms=views["attn_rms"],
        q=views["q"],
        k=views["k"],
        v=views["v"],
        attn_w=views["attn_w"],
        out_flat=views["out_flat"],
        mlp_xn=views["mlp_xn"],
        mlp_rms=views["mlp_rms"],
        h_pre=views["h_pre"],
        h=views["h"],
        dx0=views["dx0"],
        dx1=views["dx1"],
        dattn_out=views["dattn_out"],
    )


def params_dict(params: Params) -> dict[str, Tensor]:
    return {
        "wte": params.wte,
        "wpe": params.wpe,
        "lm_head": params.lm_head,
        "layer0.attn_wq": params.attn_wq,
        "layer0.attn_wk": params.attn_wk,
        "layer0.attn_wv": params.attn_wv,
        "layer0.attn_wo": params.attn_wo,
        "layer0.mlp_fc1": params.mlp_fc1,
        "layer0.mlp_fc2": params.mlp_fc2,
    }


def tokenize(doc: str, c2i: dict[str, int], bos: int) -> TokenBatch:
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    inputs = np.zeros(BLOCK_SIZE, dtype=np.int64)
    targets = np.zeros(BLOCK_SIZE, dtype=np.int64)
    loss_mask = np.zeros(BLOCK_SIZE, dtype=np.float64)
    for i in range(n):
        inputs[i] = tokens[i]
        targets[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    inv_sum_mask = np.array([1.0 / max(1, n)], dtype=np.float64)
    return TokenBatch(inputs, targets, loss_mask, inv_sum_mask)


def wrap_state_dict(params: Params) -> dict[str, list[list[object]]]:
    class W:
        __slots__ = ("data",)

        def __init__(self, data: float):
            self.data = data

    return {name: [[W(float(tensor[i, j])) for j in range(tensor.shape[1])] for i in range(tensor.shape[0])] for name, tensor in params_dict(params).items()}


def main() -> None:
    random.seed(42)
    num_steps = 1000
    attn_fwd, attn_bwd, mlp_fwd, mlp_bwd = (jit(simplify(proc))._raw for proc in (attn_fwd_fused, attn_bwd_fused, mlp_fwd_fused, mlp_bwd_fused))

    docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
    random.shuffle(docs)
    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1

    flat_params, param_views = alloc_layout(param_layout(vocab_size))
    params = bind_params(param_views)
    for tensor in params_dict(params).values():
        init_normal_(tensor, scale=0.08)

    flat_grads = zeros_like(flat_params)
    grads = bind_params(view_layout(flat_grads, param_layout(vocab_size)))
    opt_m = zeros_like(flat_params)
    opt_v = zeros_like(flat_params)

    _, scratch_views = alloc_layout(scratch_layout(vocab_size))
    scratch = bind_scratch(scratch_views)
    scalars = Scalars(
        opt_lr=empty((1,)),
        opt_bc1=empty((1,)),
        opt_bc2=empty((1,)),
        zero=full((1,), 0.0),
        one=full((1,), 1.0),
        rms_inv_n=full((1,), 1.0 / N_EMBED),
        rms_eps=full((1,), 1e-5),
        adam_b1=full((1,), 0.85),
        adam_b2=full((1,), 0.99),
        adam_eps=full((1,), 1e-8),
    )

    lm_head_step_proc, embed_rms_fwd_proc, embed_rms_bwd_proc = build_token_kernels(vocab_size)
    lm_head_step = jit(lm_head_step_proc, raw=True)
    embed_rms_fwd = jit(embed_rms_fwd_proc, raw=True)
    embed_rms_bwd = jit(embed_rms_bwd_proc, raw=True)
    adam_step = jit(simplify(adam.partial_eval(N=flat_params.numel)))._raw

    attn_fwd_args = (
        scratch.x1.ptr,
        scratch.attn_xn.ptr,
        scratch.attn_rms.ptr,
        scratch.q.ptr,
        scratch.k.ptr,
        scratch.v.ptr,
        scratch.attn_w.ptr,
        scratch.out_flat.ptr,
        scratch.x0.ptr,
        params.attn_wq.ptr,
        params.attn_wk.ptr,
        params.attn_wv.ptr,
        params.attn_wo.ptr,
        scalars.zero.ptr,
        scalars.one.ptr,
        scalars.rms_inv_n.ptr,
        scalars.rms_eps.ptr,
    )
    mlp_fwd_args = (
        scratch.dx0.ptr,
        scratch.mlp_xn.ptr,
        scratch.mlp_rms.ptr,
        scratch.h_pre.ptr,
        scratch.h.ptr,
        scratch.x1.ptr,
        params.mlp_fc1.ptr,
        params.mlp_fc2.ptr,
        scalars.zero.ptr,
        scalars.one.ptr,
        scalars.rms_inv_n.ptr,
        scalars.rms_eps.ptr,
    )
    mlp_bwd_args = (
        scratch.dx0.ptr,
        grads.mlp_fc1.ptr,
        grads.mlp_fc2.ptr,
        scratch.dx1.ptr,
        scratch.x1.ptr,
        scratch.mlp_xn.ptr,
        scratch.mlp_rms.ptr,
        scratch.h_pre.ptr,
        scratch.h.ptr,
        params.mlp_fc1.ptr,
        params.mlp_fc2.ptr,
        scalars.zero.ptr,
        scalars.rms_inv_n.ptr,
    )
    attn_bwd_args = (
        scratch.dx1.ptr,
        grads.attn_wq.ptr,
        grads.attn_wk.ptr,
        grads.attn_wv.ptr,
        grads.attn_wo.ptr,
        scratch.dattn_out.ptr,
        scratch.dx0.ptr,
        scratch.x0.ptr,
        scratch.attn_xn.ptr,
        scratch.attn_rms.ptr,
        scratch.q.ptr,
        scratch.k.ptr,
        scratch.v.ptr,
        scratch.attn_w.ptr,
        scratch.out_flat.ptr,
        params.attn_wq.ptr,
        params.attn_wk.ptr,
        params.attn_wv.ptr,
        params.attn_wo.ptr,
        scalars.zero.ptr,
        scalars.rms_inv_n.ptr,
    )
    adam_args = (
        flat_params.ptr,
        flat_grads.ptr,
        opt_m.ptr,
        opt_v.ptr,
        scalars.adam_b1.ptr,
        scalars.adam_b2.ptr,
        scalars.adam_eps.ptr,
        scalars.opt_lr.ptr,
        scalars.opt_bc1.ptr,
        scalars.opt_bc2.ptr,
    )

    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = vocab_size - 1
    tokenized = [tokenize(doc, c2i, bos) for doc in docs]

    g_wte_bytes = grads.wte.numel * grads.wte.itemsize
    g_wpe_bytes = grads.wpe.numel * grads.wpe.itemsize
    lr_t = [0.01 * (1.0 - step / num_steps) for step in range(num_steps)]
    bc1 = [1.0 - 0.85 ** (step + 1) for step in range(num_steps)]
    bc2 = [1.0 - 0.99 ** (step + 1) for step in range(num_steps)]
    memset = ctypes.memset
    perf_counter = time.perf_counter
    step_times = []

    for step in range(num_steps):
        scalars.opt_lr[0] = lr_t[step]
        scalars.opt_bc1[0] = bc1[step]
        scalars.opt_bc2[0] = bc2[step]
        batch = tokenized[step % len(tokenized)]
        embed_args = (
            scratch.emb.ptr,
            scratch.x0.ptr,
            scratch.rms_init.ptr,
            params.wte.ptr,
            params.wpe.ptr,
            scalars.zero.ptr,
            scalars.one.ptr,
            scalars.rms_inv_n.ptr,
            scalars.rms_eps.ptr,
            batch.input_ids.ctypes.data,
        )
        lm_head_args = (
            scratch.dx1.ptr,
            grads.lm_head.ptr,
            scratch.logits.ptr,
            scratch.dx0.ptr,
            params.lm_head.ptr,
            batch.loss_mask.ctypes.data,
            batch.inv_sum_mask.ctypes.data,
            scalars.zero.ptr,
            scalars.one.ptr,
            batch.target_ids.ctypes.data,
        )
        embed_bwd_args = (
            grads.wte.ptr,
            grads.wpe.ptr,
            scratch.dx1.ptr,
            scratch.emb.ptr,
            scratch.rms_init.ptr,
            scalars.zero.ptr,
            scalars.rms_inv_n.ptr,
            batch.input_ids.ctypes.data,
        )
        memset(grads.wte.ptr, 0, g_wte_bytes)
        memset(grads.wpe.ptr, 0, g_wpe_bytes)
        t0 = perf_counter()
        embed_rms_fwd(*embed_args)
        attn_fwd(*attn_fwd_args)
        mlp_fwd(*mlp_fwd_args)
        lm_head_step(*lm_head_args)
        mlp_bwd(*mlp_bwd_args)
        attn_bwd(*attn_bwd_args)
        embed_rms_bwd(*embed_bwd_args)
        adam_step(*adam_args)
        step_times.append(perf_counter() - t0)

    save_times(step_times)
    assert_weights_match(wrap_state_dict(params))


if __name__ == "__main__":
    main()
