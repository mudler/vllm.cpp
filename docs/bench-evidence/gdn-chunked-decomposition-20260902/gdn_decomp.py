#!/usr/bin/env python3
"""GDNDECOMP probe: split vLLM's chunked gated-delta-rule prefill error into
its REASSOCIATION term and its BF16-INTERMEDIATE term.

Reference algorithm read at vLLM pin 5559679229bc961848b121ccdeaa8fa5d79bec98,
vendored FLA tree vllm/third_party/flash_linear_attention/ops/:
  chunk.py:37-82  cumsum.py  chunk_scaled_dot_kkt.py:86-116  solve_tril.py
  wy_fast.py:70-116  chunk_delta_h.py:132-302,352-357  chunk_o.py:88-138

Every arm below consumes the SAME inputs, so input rounding is common mode and
cancels; what is measured is the algorithm and the placement of its rounding.
"""
import numpy as np

# ---------------------------------------------------------------- bf16 RTNE
def bf16(x):
    """f32/f64 -> bf16 -> f32, round-to-nearest-even. Mirrors Triton's
    fp_downcast_rounding='rtne' and torch's .to(bfloat16)."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> np.uint64(16)) & np.uint64(1)
    rounded = (u + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFF0000)
    out = rounded.astype(np.uint32).view(np.float32)
    out = np.where(np.isnan(a), a, out)
    return out.reshape(a.shape)

def tf32(x):
    """f32 -> tf32 (1-8-10) -> f32, RTNE. Triton's default tl.dot input
    precision on NVIDIA when allow_tf32 is not set to False."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> np.uint64(13)) & np.uint64(1)
    rounded = (u + np.uint64(0xFFF) + lsb) & np.uint64(0xFFFFE000)
    out = rounded.astype(np.uint32).view(np.float32)
    return np.where(np.isnan(a), a, out).reshape(a.shape)

IDENT = lambda x: x

# ---------------------------------------------------------------- reference
def seq_scan(q, k, v, g, beta, h0, scale, D, prescale_q=False):
    """The exact sequential gated delta rule, all arithmetic in dtype D.
    D=float64 is the oracle; D=float32 is vllm.cpp's CPU arm
    (src/vt/cpu/cpu_ops.cpp:1765 GdnHeadTokenStep)."""
    T, Dk = k.shape
    Dv = v.shape[1]
    S = h0.astype(D).copy()
    o = np.zeros((T, Dv), dtype=D)
    q = q.astype(D); k = k.astype(D); v = v.astype(D)
    g = g.astype(D); beta = beta.astype(D)
    sc = D(scale)
    for t in range(T):
        qt = q[t] * sc if prescale_q else q[t]
        S *= np.exp(g[t])
        vt = (v[t] - S @ k[t]) * beta[t]
        S = S + np.outer(vt, k[t])
        o[t] = S @ qt if prescale_q else sc * (S @ qt)
    return o, S

# ---------------------------------------------------------------- chunked
def chunked(q, k, v, g, beta, h0, scale, D, BT=64, R=None):
    """FLA's chunked gated delta rule. `D` is the working dtype for every
    accumulation; `R` maps each of FLA's storage/operand rounding sites to a
    rounding function (bf16 / tf32 / identity), so a site can be ablated.

    R keys, each naming the exact upstream line it models:
      'kkt_op'  chunk_scaled_dot_kkt.py:103 tl.dot operand precision for K.Kt
      'A_store' chunk_scaled_dot_kkt.py:161 A dtype (upstream f32 -> identity)
      'Ai'      chunk.py:50 solve_tril(output_dtype=k.dtype) -> bf16
      'wu_op'   wy_fast.py:92,114 (v*beta) and (k*beta*exp G) cast to k.dtype
      'wu_st'   wy_fast.py:94,116 u,w stores -> bf16
      'h_snap'  chunk_delta_h.py:352 h = k.new_empty(...) -> bf16 (read by chunk_o)
      'h_dot'   chunk_delta_h.py:178 tl.trans(b_h1).to(b_w.dtype) -> bf16
      'vnew_st' chunk_delta_h.py:206 v_new store -> bf16
      'vdec'    chunk_delta_h.py:274 b_v.to(k.dtype) after the decay
      'Ao'      chunk_o.py:137 b_A.to(b_v.dtype) -> bf16
      'o_st'    chunk_o.py:138 o store -> bf16
    """
    if R is None:
        R = {}
    r = lambda key: R.get(key, IDENT)
    T, Dk = k.shape
    Dv = v.shape[1]
    q = q.astype(D); k = k.astype(D); v = v.astype(D)
    g = g.astype(D); beta = beta.astype(D)
    H = h0.astype(D).copy()          # running state, f32 registers upstream
    o = np.zeros((T, Dv), dtype=D)
    nchunks = 0
    for c0 in range(0, T, BT):
        c1 = min(c0 + BT, T)
        L = c1 - c0
        nchunks += 1
        kc = k[c0:c1]; vc = v[c0:c1]; qc = q[c0:c1]
        bc = beta[c0:c1]
        G = np.cumsum(g[c0:c1]).astype(D)              # cumsum.py, f32 out
        eG = np.exp(G)
        # --- A = tril(beta_i (k_i.k_j) exp(G_i-G_j), -1)
        kb = kc * bc[:, None]
        A = (r('kkt_op')(kb).astype(D) @ r('kkt_op')(kc).astype(D).T)
        A = A * np.exp(G[:, None] - G[None, :])
        A = np.tril(A, -1)
        A = r('A_store')(A).astype(D)
        # --- Ai = (I+A)^-1, unit lower triangular; f32 forward substitution
        Ai = np.linalg.solve((np.eye(L, dtype=D) + A), np.eye(L, dtype=D))
        Ai = r('Ai')(Ai).astype(D)
        # --- WY: u = Ai @ (beta v), w = Ai @ (beta exp(G) k)
        vb = r('wu_op')(vc * bc[:, None]).astype(D)
        kbg = r('wu_op')(kc * bc[:, None] * eG[:, None]).astype(D)
        u = r('wu_st')(Ai @ vb).astype(D)
        w = r('wu_st')(Ai @ kbg).astype(D)
        # --- delta_h
        h_snap = r('h_snap')(H).astype(D)              # chunk-start snapshot
        v_new = r('vnew_st')(u - w @ r('h_dot')(H).astype(D).T).astype(D)
        glast = G[L - 1]
        v_dec = r('vdec')(v_new * np.exp(glast - G)[:, None]).astype(D)
        H = H * np.exp(glast) + (v_dec.T @ kc)         # H[vv,kk] += sum_i v_dec[i,vv] k[i,kk]
        # --- chunk_o
        o_cross = (qc @ h_snap.T) * eG[:, None]
        Ao = qc @ kc.T
        Ao = Ao * np.exp(G[:, None] - G[None, :])
        Ao = np.tril(Ao, 0)
        oc = o_cross * D(scale) + (r('Ao')(Ao).astype(D) @ v_new) * D(scale)
        o[c0:c1] = r('o_st')(oc).astype(D)
    return o, H, nchunks

# ---------------------------------------------------------------- upstream map
UPSTREAM = {                 # vLLM pin 5559679229, prefill path (g,beta f32)
    'kkt_op': IDENT,         # f32 operands (ieee); see TF32 sensitivity below
    'A_store': IDENT,        # chunk.py:47 output_dtype=torch.float32
    'Ai': bf16,              # chunk.py:50 output_dtype=k.dtype
    'wu_op': bf16, 'wu_st': bf16,
    'h_snap': bf16, 'h_dot': bf16, 'vnew_st': bf16, 'vdec': bf16,
    'Ao': bf16, 'o_st': bf16,
}
UPSTREAM_TF32 = dict(UPSTREAM, kkt_op=tf32)

# ---------------------------------------------------------------- inputs
def make_inputs(rng, T, Dk=128, Dv=128):
    """Mirrors fused_post_conv_prep: q,k L2-normalised bf16; v bf16;
    g = -exp(A_log) softplus(a + dt_bias) (f32, <=0); beta = sigmoid(b) (f32)."""
    q = rng.standard_normal((T, Dk)).astype(np.float32)
    k = rng.standard_normal((T, Dk)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    q = bf16(q); k = bf16(k)
    v = bf16(rng.standard_normal((T, Dv)).astype(np.float32))
    A_log = np.log(rng.uniform(1.0, 16.0)).astype(np.float32)   # Qwen GDN init
    a = rng.standard_normal(T).astype(np.float32)
    dt_bias = np.float32(rng.standard_normal())
    sp = np.log1p(np.exp(a + dt_bias)).astype(np.float32)
    g = (-np.exp(A_log) * sp).astype(np.float32)
    b = rng.standard_normal(T).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-b))).astype(np.float32)
    h0 = bf16(rng.standard_normal((Dv, Dk)).astype(np.float32) * 0.1).astype(np.float32)
    return q, k, v, g, beta, h0

def rel(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))

def relsum(x, ref):
    a = float(np.abs(x).sum()); b = float(np.abs(ref).sum())
    return abs(a - b) / max(abs(a), abs(b))
