from __future__ import annotations

from exo import *
from exo.libs.externs import expf, select
from utils.exo_kernels import matmul, matmul_left_t, matmul_right_t, rmsnorm

from exojit.patches_exo import Stack


@proc
def lm_head_step_fused(BLOCK_SIZE: size, N_EMBED: size, V: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[V, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, V] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[V, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM):
    matmul_right_t(BLOCK_SIZE, V, N_EMBED, logits, x, lm_head, zero)
    for t in seq(0, BLOCK_SIZE):
        mx: f64 @ Stack
        sum_val: f64 @ Stack
        scale: f64 @ Stack
        val: f64 @ Stack
        inv_denom: f64 @ Stack
        mx = logits[t, 0]
        for v_idx in seq(1, V):
            mx = select(mx, logits[t, v_idx], logits[t, v_idx], mx)
        sum_val = zero[0]
        for v_idx in seq(0, V):
            val = expf(logits[t, v_idx] - mx)
            logits[t, v_idx] = val
            sum_val += val
        inv_denom = one[0] / sum_val
        scale = loss_mask[t] * inv_sum_mask[0] * inv_denom
        for v_idx in seq(0, V):
            logits[t, v_idx] = logits[t, v_idx] * scale
            if v_idx == target_ids[t]:
                logits[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]
    matmul_left_t(BLOCK_SIZE, V, N_EMBED, dweight, logits, x, zero)
    matmul(BLOCK_SIZE, N_EMBED, V, dx, logits, lm_head, zero)


@proc
def embed_rms_fwd_tokens(BLOCK_SIZE: size, N_EMBED: size, V: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[V, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            emb[t, e] = wpe[t, e]
            for v in seq(0, V):
                if v == input_ids[t]:
                    emb[t, e] += wte[v, e]
    rmsnorm(BLOCK_SIZE, N_EMBED, out, rms, emb, zero, one, inv_n, eps)


@proc
def embed_rms_bwd_tokens(BLOCK_SIZE: size, N_EMBED: size, V: size, g_wte: f64[V, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = zero[0]
        scale = rms[t, 0]
        for e in seq(0, N_EMBED):
            dot += dout[t, e] * x[t, e]
        corr = scale * scale * scale * inv_n[0] * dot
        for e in seq(0, N_EMBED):
            dx: f64 @ Stack
            dx = dout[t, e] * scale - x[t, e] * corr
            g_wpe[t, e] = dx
            for v in seq(0, V):
                if v == input_ids[t]:
                    g_wte[v, e] += dx
