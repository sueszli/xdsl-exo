from __future__ import annotations

from exo import *
from exo.libs.externs import expf, select
from utils.exo_kernels import add, fill, matmul, matmul_left_t, matmul_right_t, relu, rmsnorm, rmsnorm_bwd

from exojit.patches_exo import Stack

N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD
INV_SCALE = 1.0 / HEAD_DIM**0.5
CAUSAL_MASK_VALUE = -1e10


def build_token_kernels(vocab_size: int):
    @proc
    def lm_head_step_fused(dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM):
        matmul_right_t(BLOCK_SIZE, vocab_size, N_EMBED, logits, x, lm_head, zero)
        for t in seq(0, BLOCK_SIZE):
            mx: f64 @ Stack
            sum_val: f64 @ Stack
            scale: f64 @ Stack
            val: f64 @ Stack
            inv_denom: f64 @ Stack
            mx = logits[t, 0]
            for v_idx in seq(1, vocab_size):
                mx = select(mx, logits[t, v_idx], logits[t, v_idx], mx)
            sum_val = zero[0]
            for v_idx in seq(0, vocab_size):
                val = expf(logits[t, v_idx] - mx)
                logits[t, v_idx] = val
                sum_val += val
            inv_denom = one[0] / sum_val
            scale = loss_mask[t] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, vocab_size):
                logits[t, v_idx] = logits[t, v_idx] * scale
                if v_idx == target_ids[t]:
                    logits[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]
        matmul_left_t(BLOCK_SIZE, vocab_size, N_EMBED, dweight, logits, x, zero)
        matmul(BLOCK_SIZE, N_EMBED, vocab_size, dx, logits, lm_head, zero)

    @proc
    def embed_rms_fwd_tokens(emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
        for t in seq(0, BLOCK_SIZE):
            for e in seq(0, N_EMBED):
                emb[t, e] = wpe[t, e]
                for v in seq(0, vocab_size):
                    if v == input_ids[t]:
                        emb[t, e] += wte[v, e]
        rmsnorm(BLOCK_SIZE, N_EMBED, out, rms, emb, zero, one, inv_n, eps)

    @proc
    def embed_rms_bwd_tokens(g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
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
                for v in seq(0, vocab_size):
                    if v == input_ids[t]:
                        g_wte[v, e] += dx

    return lm_head_step_fused, embed_rms_fwd_tokens, embed_rms_bwd_tokens


@proc
def attn_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    rmsnorm(BLOCK_SIZE, N_EMBED, xn, rms, x, zero, one, inv_n, eps)

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_q: f64 @ Stack
                acc_k: f64 @ Stack
                acc_v: f64 @ Stack
                acc_q = 0.0
                acc_k = 0.0
                acc_v = 0.0
                for e in seq(0, N_EMBED):
                    acc_q += xn[t, e] * wq[h * HEAD_DIM + d, e]
                    acc_k += xn[t, e] * wk[h * HEAD_DIM + d, e]
                    acc_v += xn[t, e] * wv[h * HEAD_DIM + d, e]
                q[h, t, d] = acc_q
                k[h, t, d] = acc_k
                v[h, t, d] = acc_v

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            mx: f64 @ Stack
            sum_val: f64 @ Stack
            logit: f64 @ Stack
            t0: f64 @ Stack

            mx = CAUSAL_MASK_VALUE
            for s in seq(0, BLOCK_SIZE):
                if s > t:
                    logit = CAUSAL_MASK_VALUE
                else:
                    logit = 0.0
                    for d in seq(0, HEAD_DIM):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * INV_SCALE
                attn_w[h, t, s] = logit
                mx = select(mx, attn_w[h, t, s], attn_w[h, t, s], mx)

            sum_val = 0.0
            for s in seq(0, BLOCK_SIZE):
                t0 = attn_w[h, t, s] - mx
                attn_w[h, t, s] = expf(t0)
                sum_val += attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                attn_w[h, t, s] = attn_w[h, t, s] / sum_val

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out_flat[t, h * HEAD_DIM + d] = acc

    matmul_right_t(BLOCK_SIZE, N_EMBED, N_EMBED, out, out_flat, wo, zero)
    add(BLOCK_SIZE, N_EMBED, out, x)


@proc
def mlp_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    rmsnorm(BLOCK_SIZE, N_EMBED, xn, rms, x, zero, one, inv_n, eps)
    matmul_right_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, h_pre, xn, fc1, zero)
    relu(BLOCK_SIZE, 4 * N_EMBED, h, h_pre, zero)
    matmul_right_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, h, fc2, zero)
    add(BLOCK_SIZE, N_EMBED, out, x)


@proc
def mlp_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dw1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dw2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    fill(BLOCK_SIZE, N_EMBED, out, zero)
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dw2, dx, h, zero)
    fill(4 * N_EMBED, N_EMBED, dw1, zero)

    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, 4 * N_EMBED):
            dh: f64 @ Stack
            dh_pre: f64 @ Stack
            dh = 0.0
            for j in seq(0, N_EMBED):
                dh += dx[t, j] * fc2[j, e]
            dh_pre = select(0.0, h_pre[t, e], dh, 0.0)
            for k in seq(0, N_EMBED):
                dw1[e, k] += dh_pre * xn[t, k]
                out[t, k] += dh_pre * fc1[e, k]

    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, out, dx, x_pre, rms, zero, inv_n)


@proc
def attn_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
    fill(BLOCK_SIZE, N_EMBED, out, zero)
    matmul_left_t(BLOCK_SIZE, N_EMBED, N_EMBED, dwo, dx, out_flat, zero)
    fill(N_EMBED, N_EMBED, dwq, zero)
    fill(N_EMBED, N_EMBED, dwk, zero)
    fill(N_EMBED, N_EMBED, dwv, zero)

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for j in seq(0, N_EMBED):
                    acc += dx[t, j] * wo[j, h * HEAD_DIM + d]
                dattn_out[h, t, d] = acc

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            dq_acc: f64[HEAD_DIM] @ Stack
            dot: f64 @ Stack
            dot = 0.0
            for d in seq(0, HEAD_DIM):
                dq_acc[d] = 0.0
            for s in seq(0, BLOCK_SIZE):
                dattn_w: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * v[h, s, d]
                attn_tmp[s] = dattn_w
                dot += dattn_w * attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                dlogit: f64 @ Stack
                dlogit = attn_w[h, t, s] * (attn_tmp[s] - dot) * INV_SCALE

                for d in seq(0, HEAD_DIM):
                    dk_contrib: f64 @ Stack
                    dv_contrib: f64 @ Stack
                    dq_acc[d] += dlogit * k[h, s, d]
                    dk_contrib = dlogit * q[h, t, d]
                    dv_contrib = attn_w[h, t, s] * dattn_out[h, t, d]

                    for e in seq(0, N_EMBED):
                        out[s, e] += dk_contrib * wk[h * HEAD_DIM + d, e]
                        out[s, e] += dv_contrib * wv[h * HEAD_DIM + d, e]
                        dwk[h * HEAD_DIM + d, e] += dk_contrib * xn[s, e]
                        dwv[h * HEAD_DIM + d, e] += dv_contrib * xn[s, e]

            for d in seq(0, HEAD_DIM):
                for e in seq(0, N_EMBED):
                    out[t, e] += dq_acc[d] * wq[h * HEAD_DIM + d, e]
                    dwq[h * HEAD_DIM + d, e] += dq_acc[d] * xn[t, e]

    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, out, dx, x_pre, rms, zero, inv_n)
