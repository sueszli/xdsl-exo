# /// script
# requires-python = "==3.14.*"
# dependencies = ["tqdm"]
# ///

import math
import random
import time
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


def rmsnorm_fwd(x_2d):
    n = len(x_2d)
    d = len(x_2d[0])
    inv_d = 1.0 / d
    out = [[0.0] * d for _ in range(n)]
    rms = [0.0] * n
    for i in range(n):
        x = x_2d[i]
        ms = sum(xi * xi for xi in x) * inv_d
        scale = (ms + 1e-5) ** -0.5
        rms[i] = scale
        out_i = out[i]
        for j in range(d):
            out_i[j] = x[j] * scale
    return out, rms


def rmsnorm_bwd(dout_2d, x_2d, rms):
    n = len(x_2d)
    d = len(x_2d[0])
    dx = [[0.0] * d for _ in range(n)]
    for i in range(n):
        do = dout_2d[i]
        x = x_2d[i]
        scale = rms[i]
        scale3_over_d = (scale**3) / d
        dot = sum(doi * xi for doi, xi in zip(do, x))
        dx_i = dx[i]
        for j in range(d):
            dx_i[j] = do[j] * scale - scale3_over_d * x[j] * dot
    return dx


def transpose_2d(A):
    return [list(col) for col in zip(*A)]


def forward_backward(params, input_ids, target_ids, loss_mask):
    _exp = math.exp
    _log = math.log
    _sum = sum
    _range = range

    n = len(input_ids)
    grads = {k: [[0.0 for _ in row] for row in mat] for k, mat in params.items()}
    head_dim = N_EMBED // N_HEAD
    vocab_size = len(params["lm_head"])

    wte = params["wte"]
    wpe = params["wpe"]
    emb = [[0.0] * N_EMBED for _ in _range(n)]
    for i in _range(n):
        wte_i = wte[input_ids[i]]
        wpe_i = wpe[i]
        emb_i = emb[i]
        for j in _range(N_EMBED):
            emb_i[j] = wte_i[j] + wpe_i[j]

    x, rms_init = rmsnorm_fwd(emb)

    layer_cache = []
    for li in _range(N_LAYER):
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        x_pre_attn = x
        xn_attn, rms_attn = rmsnorm_fwd(x)

        q = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        k = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        v = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        _range_N_EMBED = _range(N_EMBED)
        for i in _range(n):
            xn_i = xn_attn[i]
            for j in _range_N_EMBED:
                wq_j = wq[j]
                wk_j = wk[j]
                wv_j = wv[j]
                val_q = _sum(xn_i[d] * wq_j[d] for d in _range_N_EMBED)
                val_k = _sum(xn_i[d] * wk_j[d] for d in _range_N_EMBED)
                val_v = _sum(xn_i[d] * wv_j[d] for d in _range_N_EMBED)
                h_idx = j // head_dim
                d_idx = j % head_dim
                q[h_idx][i][d_idx] = val_q
                k[h_idx][i][d_idx] = val_k
                v[h_idx][i][d_idx] = val_v

        attn_w = [[[0.0] * n for _ in _range(n)] for _ in _range(N_HEAD)]
        attn_out_flat = [[0.0] * N_EMBED for _ in _range(n)]
        scale = head_dim**0.5
        inv_scale = 1.0 / scale
        _range_head_dim = _range(head_dim)

        for h in _range(N_HEAD):
            q_h, k_h, v_h = q[h], k[h], v[h]
            h_off = h * head_dim
            for i in _range(n):
                q_h_i = q_h[i]
                qk_i = [0.0] * n
                for j in _range(i + 1):
                    qk_i[j] = _sum(q_h_i[d] * k_h[j][d] for d in _range_head_dim) * inv_scale
                for j in _range(i + 1, n):
                    qk_i[j] = -1e10

                max_val = max(qk_i)
                exps = [_exp(val - max_val) for val in qk_i]
                total = _sum(exps)
                aw_i = [e / total for e in exps]
                attn_w[h][i] = aw_i

                attn_out_flat_i = attn_out_flat[i]
                for d in _range_head_dim:
                    val = _sum(aw_i[j] * v_h[j][d] for j in _range(n))
                    attn_out_flat_i[h_off + d] = val

        x = [[0.0] * N_EMBED for _ in _range(n)]
        for i in _range(n):
            attn_out_flat_i = attn_out_flat[i]
            x_pre_i = x_pre_attn[i]
            x_i = x[i]
            for j in _range_N_EMBED:
                wo_j = wo[j]
                x_i[j] = _sum(attn_out_flat_i[d] * wo_j[d] for d in _range_N_EMBED) + x_pre_i[j]

        x_pre_mlp = x
        xn_mlp, rms_mlp = rmsnorm_fwd(x)

        mlp_dim = 4 * N_EMBED
        _range_mlp = _range(mlp_dim)
        h_pre = [[0.0] * mlp_dim for _ in _range(n)]
        h_val = [[0.0] * mlp_dim for _ in _range(n)]
        for i in _range(n):
            xn_mlp_i = xn_mlp[i]
            h_pre_i = h_pre[i]
            h_val_i = h_val[i]
            for j in _range_mlp:
                fc1_j = fc1[j]
                val = _sum(xn_mlp_i[d] * fc1_j[d] for d in _range_N_EMBED)
                h_pre_i[j] = val
                h_val_i[j] = val if val > 0.0 else 0.0

        x = [[0.0] * N_EMBED for _ in _range(n)]
        for i in _range(n):
            h_val_i = h_val[i]
            x_pre_mlp_i = x_pre_mlp[i]
            x_i = x[i]
            for j in _range_N_EMBED:
                fc2_j = fc2[j]
                x_i[j] = _sum(h_val_i[d] * fc2_j[d] for d in _range_mlp) + x_pre_mlp_i[j]

        layer_cache.append({"x_pre_attn": x_pre_attn, "xn_attn": xn_attn, "rms_attn": rms_attn, "q": q, "k": k, "v": v, "attn_w": attn_w, "attn_out_flat": attn_out_flat, "x_pre_mlp": x_pre_mlp, "xn_mlp": xn_mlp, "rms_mlp": rms_mlp, "h_pre": h_pre, "h": h_val})

    lm_head = params["lm_head"]
    logits = [[0.0] * vocab_size for _ in _range(n)]
    probs = [[0.0] * vocab_size for _ in _range(n)]
    _range_vocab = _range(vocab_size)
    for i in _range(n):
        x_i = x[i]
        logits_i = logits[i]
        for j in _range_vocab:
            lm_j = lm_head[j]
            logits_i[j] = _sum(x_i[d] * lm_j[d] for d in _range_N_EMBED)
        max_val = max(logits_i)
        exps = [_exp(val - max_val) for val in logits_i]
        total = _sum(exps)
        probs[i] = [e / total for e in exps]

    sum_mask = _sum(loss_mask)
    if sum_mask == 0:
        sum_mask = 1.0
    inv_sum_mask = 1.0 / sum_mask

    loss = -_sum(_log(probs[i][target_ids[i]]) * loss_mask[i] for i in _range(n)) * inv_sum_mask

    dlogits = [[(p * inv_sum_mask) * loss_mask[i] for p in p_row] for i, p_row in enumerate(probs)]
    for i in _range(n):
        dlogits[i][target_ids[i]] -= inv_sum_mask * loss_mask[i]

    dlm_head = [[0.0] * N_EMBED for _ in _range(vocab_size)]
    for j in _range_vocab:
        dlm_j = dlm_head[j]
        for d in _range_N_EMBED:
            dlm_j[d] = _sum(dlogits[i][j] * x[i][d] for i in _range(n))
    grads["lm_head"] = dlm_head

    dx = [[0.0] * N_EMBED for _ in _range(n)]
    for i in _range(n):
        dlogits_i = dlogits[i]
        dx_i = dx[i]
        for d in _range_N_EMBED:
            dx_i[d] = _sum(dlogits_i[j] * lm_head[j][d] for j in _range_vocab)

    for li in reversed(_range(N_LAYER)):
        cache = layer_cache[li]
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]
        cache_h = cache["h"]
        cache_h_pre = cache["h_pre"]
        cache_xn_mlp = cache["xn_mlp"]
        cache_attn_out_flat = cache["attn_out_flat"]
        cache_attn_w = cache["attn_w"]
        cache_v = cache["v"]
        cache_k = cache["k"]
        cache_q = cache["q"]
        cache_xn_attn = cache["xn_attn"]

        mlp_dim = 4 * N_EMBED
        _range_mlp = _range(mlp_dim)

        dx_res_mlp = dx
        dfc2 = [[0.0] * mlp_dim for _ in _range(N_EMBED)]
        for j in _range_N_EMBED:
            dfc2_j = dfc2[j]
            for d in _range_mlp:
                dfc2_j[d] = _sum(dx[i][j] * cache_h[i][d] for i in _range(n))
        grads[f"layer{li}.mlp_fc2"] = dfc2

        dh_pre = [[0.0] * mlp_dim for _ in _range(n)]
        for i in _range(n):
            dx_i = dx[i]
            h_pre_i = cache_h_pre[i]
            dh_pre_i = dh_pre[i]
            for j in _range_mlp:
                val = _sum(dx_i[d] * fc2[d][j] for d in _range_N_EMBED)
                dh_pre_i[j] = val if h_pre_i[j] > 0.0 else 0.0

        dfc1 = [[0.0] * N_EMBED for _ in _range(mlp_dim)]
        for j in _range_mlp:
            dfc1_j = dfc1[j]
            for d in _range_N_EMBED:
                dfc1_j[d] = _sum(dh_pre[i][j] * cache_xn_mlp[i][d] for i in _range(n))
        grads[f"layer{li}.mlp_fc1"] = dfc1

        dxn_mlp = [[0.0] * N_EMBED for _ in _range(n)]
        for i in _range(n):
            dh_pre_i = dh_pre[i]
            dxn_mlp_i = dxn_mlp[i]
            for d in _range_N_EMBED:
                dxn_mlp_i[d] = _sum(dh_pre_i[j] * fc1[j][d] for j in _range_mlp)

        dx_rmsnorm = rmsnorm_bwd(dxn_mlp, cache["x_pre_mlp"], cache["rms_mlp"])
        dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rmsnorm, dx_res_mlp)]

        dx_res_attn = dx
        dwo = [[0.0] * N_EMBED for _ in _range(N_EMBED)]
        for j in _range_N_EMBED:
            dwo_j = dwo[j]
            for d in _range_N_EMBED:
                dwo_j[d] = _sum(dx[i][j] * cache_attn_out_flat[i][d] for i in _range(n))
        grads[f"layer{li}.attn_wo"] = dwo

        dattn_out_flat = [[0.0] * N_EMBED for _ in _range(n)]
        for i in _range(n):
            dx_i = dx[i]
            dattn_i = dattn_out_flat[i]
            for d in _range_N_EMBED:
                dattn_i[d] = _sum(dx_i[j] * wo[j][d] for j in _range_N_EMBED)

        dv = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        dattn_w = [[[0.0] * n for _ in _range(n)] for _ in _range(N_HEAD)]

        for h in _range(N_HEAD):
            aw_h = cache_attn_w[h]
            v_h = cache_v[h]
            h_off = h * head_dim
            dv_h = dv[h]
            dattn_w_h = dattn_w[h]
            for i in _range(n):
                dv_h_i = dv_h[i]
                dattn_i = dattn_out_flat[i]
                for d in _range_head_dim:
                    dv_h_i[d] = _sum(aw_h[j][i] * dattn_out_flat[j][h_off + d] for j in _range(n))

            for i in _range(n):
                dattn_i = dattn_out_flat[i]
                dattn_w_h_i = dattn_w_h[i]
                for j in _range(n):
                    dattn_w_h_i[j] = _sum(dattn_i[h_off + d] * v_h[j][d] for d in _range_head_dim)

        dlogits_attn = [[[0.0] * n for _ in _range(n)] for _ in _range(N_HEAD)]
        for h in _range(N_HEAD):
            aw_h = cache_attn_w[h]
            d_aw_h = dattn_w[h]
            dl_attn_h = dlogits_attn[h]
            for i in _range(n):
                aw_h_i = aw_h[i]
                d_aw_h_i = d_aw_h[i]
                sum_d_aw = _sum(d_aw_h_i[j] * aw_h_i[j] for j in _range(n))
                dl_attn_h_i = dl_attn_h[i]
                for j in _range(n):
                    dl_attn_h_i[j] = aw_h_i[j] * (d_aw_h_i[j] - sum_d_aw) * inv_scale

        dq = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        dk = [[[0.0] * head_dim for _ in _range(n)] for _ in _range(N_HEAD)]
        for h in _range(N_HEAD):
            dl_h = dlogits_attn[h]
            k_h = cache_k[h]
            q_h = cache_q[h]
            dq_h = dq[h]
            dk_h = dk[h]
            for i in _range(n):
                dl_h_i = dl_h[i]
                dq_h_i = dq_h[i]
                dk_h_i = dk_h[i]
                for d in _range_head_dim:
                    dq_h_i[d] = _sum(dl_h_i[j] * k_h[j][d] for j in _range(n))
                    dk_h_i[d] = _sum(dl_h[j][i] * q_h[j][d] for j in _range(n))

        dwq = [[0.0] * N_EMBED for _ in _range(N_EMBED)]
        dwk = [[0.0] * N_EMBED for _ in _range(N_EMBED)]
        dwv = [[0.0] * N_EMBED for _ in _range(N_EMBED)]
        for j in _range_N_EMBED:
            h_idx = j // head_dim
            d_idx = j % head_dim
            dq_h = dq[h_idx]
            dk_h = dk[h_idx]
            dv_h = dv[h_idx]
            dwq_j = dwq[j]
            dwk_j = dwk[j]
            dwv_j = dwv[j]
            for k_dim in _range_N_EMBED:
                dwq_j[k_dim] = _sum(dq_h[i][d_idx] * cache_xn_attn[i][k_dim] for i in _range(n))
                dwk_j[k_dim] = _sum(dk_h[i][d_idx] * cache_xn_attn[i][k_dim] for i in _range(n))
                dwv_j[k_dim] = _sum(dv_h[i][d_idx] * cache_xn_attn[i][k_dim] for i in _range(n))

        grads[f"layer{li}.attn_wq"] = dwq
        grads[f"layer{li}.attn_wk"] = dwk
        grads[f"layer{li}.attn_wv"] = dwv

        dxn_attn = [[0.0] * N_EMBED for _ in _range(n)]
        for i in _range(n):
            dxn_attn_i = dxn_attn[i]
            dq_i_by_head = [dq[j // head_dim][i][j % head_dim] for j in _range_N_EMBED]
            dk_i_by_head = [dk[j // head_dim][i][j % head_dim] for j in _range_N_EMBED]
            dv_i_by_head = [dv[j // head_dim][i][j % head_dim] for j in _range_N_EMBED]
            for k_dim in _range_N_EMBED:
                val = 0.0
                for j in _range_N_EMBED:
                    val += dq_i_by_head[j] * wq[j][k_dim] + dk_i_by_head[j] * wk[j][k_dim] + dv_i_by_head[j] * wv[j][k_dim]
                dxn_attn_i[k_dim] = val

        dx_rmsnorm = rmsnorm_bwd(dxn_attn, cache["x_pre_attn"], cache["rms_attn"])
        dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rmsnorm, dx_res_attn)]

    demb = rmsnorm_bwd(dx, emb, rms_init)
    grads_wte = grads["wte"]
    grads_wpe = grads["wpe"]
    for i in _range(n):
        tid = input_ids[i]
        demb_i = demb[i]
        gwte_tid = grads_wte[tid]
        gwpe_i = grads_wpe[i]
        for j in _range_N_EMBED:
            d_ij = demb_i[j]
            gwte_tid[j] += d_ij
            gwpe_i[j] += d_ij

    return loss, grads


def step_fn(params, opt_state, input_ids, target_ids, loss_mask, step):
    loss, grads = forward_backward(params, input_ids, target_ids, loss_mask)

    learning_rate = 0.01
    beta1 = 0.85
    beta2 = 0.99
    eps_adam = 1e-8
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    m = opt_state["m"]
    v = opt_state["v"]

    inv_bias1 = 1.0 / (1 - beta1 ** (step + 1))
    inv_bias2 = 1.0 / (1 - beta2 ** (step + 1))
    lr_scaled = lr_t * inv_bias1
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    _pow = pow

    new_params = {}
    new_m = {}
    new_v = {}

    for k in params:
        p_mat = params[k]
        g_mat = grads[k]
        m_mat = m[k]
        v_mat = v[k]

        nrow = len(p_mat)
        ncol = len(p_mat[0])

        new_p = [[0.0] * ncol for _ in range(nrow)]
        new_m_mat = [[0.0] * ncol for _ in range(nrow)]
        new_v_mat = [[0.0] * ncol for _ in range(nrow)]

        for i in range(nrow):
            p_i = p_mat[i]
            g_i = g_mat[i]
            m_i = m_mat[i]
            v_i = v_mat[i]
            new_p_i = new_p[i]
            new_m_i = new_m_mat[i]
            new_v_i = new_v_mat[i]
            for j in range(ncol):
                g = g_i[j]
                m_ij = beta1 * m_i[j] + one_minus_beta1 * g
                v_ij = beta2 * v_i[j] + one_minus_beta2 * g * g
                v_hat = v_ij * inv_bias2
                new_p_i[j] = p_i[j] - lr_scaled * m_ij / (_pow(v_hat, 0.5) + eps_adam)
                new_m_i[j] = m_ij
                new_v_i[j] = v_ij

        new_params[k] = new_p
        new_m[k] = new_m_mat
        new_v[k] = new_v_mat

    return loss, new_params, {"m": new_m, "v": new_v}


def tokenize(doc: str, uchars: list[str]) -> tuple[list[int], list[int], list[float]]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    input_ids = [0] * BLOCK_SIZE
    target_ids = [0] * BLOCK_SIZE
    loss_mask = [0.0] * BLOCK_SIZE

    for i in range(n):
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
        loss_mask[i] = 1.0

    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    "wte": matrix(len(uchars) + 1, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(len(uchars) + 1, N_EMBED),
    **{f"layer{i}.attn_wq": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

opt_state = {"m": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()}, "v": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()}}

tokenized = [tokenize(doc, uchars) for doc in docs]

step_times = []
for step in tqdm(range(NUM_STEPS)):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
