# GDN (Gated DeltaNet) semantics — pinned oracle notes

Recorded from the pinned upstream checkout `/home/mudler/_git/vllm` @
`e24d1b24` (see .agents/upstream-sync.md). Every formula below was read from
source, not memory. Cites are `file:line` relative to `vllm/` in that tree.
This is the M0.7 op contract reference AND the M0.9 layer-assembly reference.

Pinned sources read:

- `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (orchestration)
- `model_executor/layers/mamba/gdn/base.py` (activation, eps, state dtype)
- `model_executor/layers/mamba/ops/causal_conv1d.py` (conv fwd + update)
- `model_executor/layers/fla/ops/chunk.py` (+ sub-ops `chunk_delta_h.py`,
  `chunk_o.py`, `chunk_scaled_dot_kkt.py`, `cumsum.py`, `solve_tril.py`,
  `wy_fast.py`, `index.py`, `op.py`, `utils.py`)
- `model_executor/layers/fla/ops/fused_recurrent.py`
- `model_executor/layers/fla/ops/fused_sigmoid_gating.py`
- `model_executor/layers/fla/ops/l2norm.py`
- `model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`
- `model_executor/layers/fla/ops/layernorm_guard.py`
- `model_executor/layers/layernorm.py` (`RMSNormGated`)
- `v1/attention/backends/gdn_attn.py` (prefill/decode segmentation)
- `model_executor/layers/mamba/mamba_utils.py` (state shapes/dtypes)

## 1. Dims, head layout, state shapes/dtypes

Config fields (`transformers_utils/configs/qwen3_next.py:204-208` defaults):
`Hk = linear_num_key_heads` (16), `Hv = linear_num_value_heads` (32),
`Dk = linear_key_head_dim` (128), `Dv = linear_value_head_dim` (128),
`K = linear_conv_kernel_dim` (4). Derived (qwen_gdn_linear_attn.py:443-449,
466): `key_dim = Hk*Dk`, `value_dim = Hv*Dv`,
`conv_dim = 2*key_dim + value_dim`.

- **mixed_qkv channel layout**: `[ q (Hk*Dk) | k (Hk*Dk) | v (Hv*Dv) ]`
  (qwen_gdn_linear_attn.py:825, 936-939; fused_gdn_prefill_post_conv.py:77-119
  uses offsets `i_h*K`, `H*K + i_h*K`, `2*H*K + i_hv*V`).
- **GQA broadcast (Hv > Hk)**: v-head `i_hv` reads q/k head
  `i_h = i_hv // (Hv // Hk)` (fused_recurrent.py:64,
  fused_sigmoid_gating.py:64). Hv must be a multiple of Hk. Heads only
  replicate the math; dims drive it.
- **SSM state**: per sequence `[Hv, Dv, Dk]` — v-major: element `(v, k)` at
  offset `i_hv*V*K + v*K + k` (fused_recurrent.py:119,160). Shape from
  `MambaStateShapeCalculator.gated_delta_net_state_shape`
  (mamba_utils.py:229-233).
- **Conv state**: per sequence `[conv_dim, K-1]` (mamba_utils.py:223-227);
  DS layout = (dim, state_len) directly, SD layout needs transpose
  (qwen_gdn_linear_attn.py:1309-1315, mamba_utils.py:46-48).
- **State dtypes** (mamba_utils.py:84-96): with the default
  `mamba_cache_dtype="auto"`/`mamba_ssm_cache_dtype="auto"`, BOTH conv state
  and ssm state are the **model dtype** (bf16 for our gates); ssm dtype is
  independently overridable to f32. Kernels always *compute* the state in
  f32 (`b_h` is `tl.float32`, fused_recurrent.py:102) and round to the cache
  dtype on store (fused_recurrent.py:161; qwen_gdn_linear_attn.py:1532
  `.to(ssm_state.dtype)`). M0.7 C++ ops keep caller-allocated **f32 states**;
  the bf16-state rounding point is an M0.9 assembly concern — recorded here
  so it is not lost.
- **Scale**: everywhere `scale = Dk ** -0.5` (chunk.py:228-229 default;
  fused_sigmoid_gating.py:220-221 default; passed explicitly as
  `self.head_k_dim**-0.5` in the packed decode path,
  qwen_gdn_linear_attn.py:1690). Applied to **q only**, after l2norm
  (fused_recurrent.py:130, fused_sigmoid_gating.py:141).

## 2. causal_conv1d_fn (prefill conv) — `kCausalConv1dFwd`

Call site (qwen_gdn_linear_attn.py:1362-1375): x = `mixed_qkv.transpose(0,1)`
→ `(dim, cu_seqlen)` channel-last (`x.stride(0)==1` required,
causal_conv1d.py:620), `weight = conv1d.weight.view(dim, K)`
(qwen_gdn_linear_attn.py:1325-1327; checkpoint weight is `[dim, 1, K]`,
line 473), `bias = conv1d.bias` = **None** (conv1d built with `bias=False`,
line 470 — Qwen3-Next/3.5; the kernel supports bias so the op keeps it
optional), `activation = self.activation = config.hidden_act = "silu"`
(base.py:37), `conv_states`, `has_initial_state`, `cache_indices`,
`query_start_loc`.

Math per channel c, token t of a sequence of length T
(causal_conv1d.py:411-465, weights preloaded 399-409):

    window = [ p_{K-1-j} ] where p_i = x[c, t-i] for t-i >= 0,
              else conv_state[c, (K-1)+(t-i)] if has_initial_state
              else 0                                      (lines 151-192)
    acc    = bias[c] + sum_{j=0}^{K-1} w[c, j] * window[j] (f32 accum, 386-442)
    out[c,t] = silu(acc) = acc / (1 + exp(-acc)) if activation in
               {silu, swish} else acc                     (lines 454-455, 736)

i.e. `w[:, K-1]` multiplies the current token, `w[:, 0]` the oldest.
Input is cast to `conv_states.dtype` on entry and output back to the input
dtype (lines 543-545, 745).

**State write-back** (lines 196-235 for T >= K-1; 237-307 otherwise):
`conv_state[cache_idx] <- last K-1 RAW x tokens` (pre-activation), left-padded
with zeros (no init state) or shifted old state (init state) when T < K-1.

Continuous batching: `cache_indices` value 0 is the NULL block and is skipped
(`null_block_id = NULL_BLOCK_ID = 0`, v1/attention/backends/utils.py:46;
causal_conv1d.py:136-139); `pad_slot_id = -1` skips padded grid entries.

## 3. causal_conv1d_update (decode conv, single step) — `kCausalConv1dUpdate`

Call site (qwen_gdn_linear_attn.py:1376-1388, 1674-1682): `x [num_tokens, dim]`
(one token per sequence in the non-spec decode path), same weight/bias/
activation, `conv_state_indices` per token.

For seqlen == 1 (causal_conv1d.py:856-933, 972-1066):

    window = [ conv_state[c, 0], ..., conv_state[c, K-2], x[c] ]
    out[c] = silu(bias[c] + sum_j w[c, j] * window[j])
    conv_state[c, :] <- [ conv_state[c, 1:], x[c] ]   (rolled left; raw x)

The state roll is stored before the output loop (line 933) but the output
window uses the pre-roll `col*` registers, so semantics are
"read-old-then-roll". x is cast to `conv_state.dtype` (line 1131), the output
overwrites x's buffer upstream (`out = x`, line 1163) and is cast back
(line 1239). NULL block id 0 skipped (lines 814-817).

## 4. l2norm_fwd — `kL2Norm`

l2norm.py:88-92 (kernel2 is the default path, `USE_DEFAULT_FLA_NORM=0`,
l2norm.py:18,114; kernels 1/l2norm_fwd_kernel compute the same formula,
lines 41-47, 71-73):

    y = x * rsqrt( sum(x^2 over last dim) + eps ),  eps = 1e-6 (default,
    l2norm.py:96)

**Plain sum, not mean** (this is NOT an rmsnorm). Computed in f32, stored in
the input dtype. Rowwise over the last dim.

**Application points** (q/k, per (token, head), over Dk):

- prefill: applied INSIDE `fused_post_conv_prep` (`apply_l2norm=True`,
  qwen_gdn_linear_attn.py:1437; kernel math identical: `1/sqrt(sum+1e-6)`,
  fused_gdn_prefill_post_conv.py:84-91, L2NORM_EPS=1e-6 line 238), so
  `chunk_gated_delta_rule` is called with `use_qk_l2norm_in_kernel=False`
  (qwen_gdn_linear_attn.py:1529). q/k entering the chunk kernel are ALREADY
  normalized.
- decode: applied INSIDE the fused recurrence via
  `use_qk_l2norm_in_kernel=True` (qwen_gdn_linear_attn.py:1557):
  `b_q * rsqrt(sum(b_q^2) + 1e-6)` (fused_sigmoid_gating.py:138-140,
  fused_recurrent.py:127-129). Identical formula/eps, so decomposing decode
  into L2Norm + norm-free recurrence is exact.
- v is never l2-normalized.

## 5. RMSNormGated — `kRmsNormGated`

Construction (qwen_gdn_linear_attn.py:529-543): `hidden = Dv`,
`eps = config.rms_norm_eps` (base.py:38), `group_size=None`,
`norm_before_gate=True`, `activation = output_gate_type` — default "silu"
("swish" mapped to "silu"; "sigmoid" allowed). No bias (layernorm.py:210).

Applied on flattened rows (qwen_gdn_linear_attn.py:863-866):
`x = core_attn_out.reshape(-1, Dv)`, `z = z.reshape(-1, Dv)` (the gate from
the qkvz projection), `out = norm(x, z)`.

Math (native reference layernorm.py:218-266 `forward_static`; the CUDA path
`rmsnorm_fn` → `layer_norm_fwd_kernel` in layernorm_guard.py computes the
same, lines 111-165):

    x, z, w in f32
    var  = mean(x^2 over last dim)          # sum/N — a MEAN here, unlike §4
    out  = x * rsqrt(var + eps) * w
    out *= act(z)          # act = silu (or sigmoid if output_gate_type says)
    return out.to(orig_dtype)

Order: norm FIRST, then gate (norm_before_gate=True; the `False` branch
`norm(x * act(z))` is not used by Qwen GDN). Group path unused
(group_size=None → single group).

## 6. beta / g derivation (M0.9 territory; ops take g/beta as inputs)

From the ba projection: `b, a = ba.chunk(2, dim=-1)`, each `[T, Hv]`
(qwen_gdn_linear_attn.py:941). Learned params `A_log [Hv] f32`,
`dt_bias [Hv]` (lines 516-524). Then (fused_gdn_prefill_post_conv.py:136-145;
identical in fused_sigmoid_gating.py:128-136 and `fused_gdn_gating`
qwen_gdn_linear_attn.py:1781-1789):

    x     = a + dt_bias                          (f32)
    sp    = softplus(x) = log(1 + exp(x)),  numerically stabilized;
            sp = x where x > threshold (threshold = 20.0)
    g     = -exp(A_log) * sp                     (f32, <= 0; "log-space decay")
    beta  = sigmoid(b)                           (f32, in (0,1))

`fused_post_conv_prep` (fla/ops/fused_gdn_prefill_post_conv.py:152-248) is
the prefill packager: takes conv output `[T, conv_dim]` + a/b + A_log/dt_bias
and emits contiguous `q [T,Hk,Dk]`, `k [T,Hk,Dk]` (both l2-normalized when
`apply_l2norm=True`), `v [T,Hv,Dv]` (copied, model dtype), `g [T,Hv] f32`,
`beta [T,Hv] f32`. `output_g_exp=False` on the Triton/FLA path (g stays in
log space; True only for FlashInfer).

## 7. The gated-delta-rule recurrence — `kGdnPrefill` / `kGdnDecode` core math

Per (sequence, v-head i_hv) with q/k head `i_h = i_hv // (Hv//Hk)` and f32
state `S [Dv, Dk]` (fused_recurrent.py:102-175; fused_sigmoid_gating.py:
102-178 is the same update with in-kernel gating):

    S = initial_state (f32; zeros if none)       (lines 103-120)
    for each token t:
        q_t' = l2norm(q_t) if in-kernel norm     (line 127-129)
        q_t' = q_t' * scale                      (line 130; scale = Dk^-0.5)
        S    = S * exp(g_t[i_hv])                (line 133-134; scalar per v-head)
        v_t' = v_t - S @ k_t                     (line 139; [Dv] = [Dv,Dk]@[Dk])
        v_t' = v_t' * beta_t[i_hv]               (line 144)
        S    = S + outer(v_t', k_t)              (line 146; [Dv,Dk])
        o_t  = S @ q_t'                          (line 148; [Dv])

k is NOT scaled and (on the prefill path) arrives already l2-normalized.
Output dtype = q.dtype (`o = q.new_empty(...)`, fused_recurrent.py:201);
all arithmetic f32. Final state stored to the cache dtype (line 161).

**Prefill (chunked oracle)**: `chunk_gated_delta_rule` (chunk.py:138-245) is
a chunk-parallel (WY-representation) evaluation of exactly this recurrence:
`chunk_local_cumsum(g, 64)` → `chunk_scaled_dot_kkt` → `solve_tril` →
`recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` → `chunk_fwd_o`
(chunk.py:37-82, FLA_CHUNK_SIZE=64, fla/ops/utils.py:31). Call signature at
the Qwen call site (qwen_gdn_linear_attn.py:1515-1530): q/k/v `[1, T, H*, D*]`
(varlen, batch dim 1), g/beta `[1, T, Hv]` f32, `initial_state [N, Hv, Dv,
Dk]` (rows zeroed where no initial state, lines 1513-1514),
`output_final_state=True`, `cu_seqlens [N+1]`,
`use_qk_l2norm_in_kernel=False`. Output `o [1, T, Hv, Dv]` cast to q.dtype
(chunk.py:134). **f32 q/k/v is rejected** by the pinned wrapper (chunk.py:
213-215 `assert q.dtype != torch.float32`) — bf16 is the supported input
dtype. There is NO torch-native reference implementation of the chunked path
at this pin (all five sub-ops are Triton-only); the sequential Triton kernel
`fused_recurrent_gated_delta_rule` (same file family, accepts f32) is the
pinned sequential statement of the same recurrence and is what the M0.7 C++
implements directly.

**Decode**: `fused_sigmoid_gating_delta_rule_update` (fused_sigmoid_gating.py:
181-279) = the recurrence above with g/beta computed in-kernel from
A_log/a/b/dt_bias (§6) and `use_qk_l2norm_in_kernel=True`. Call site
(qwen_gdn_linear_attn.py:1540-1559): q/k/v `[1, B, H*, D*]` (B single-token
seqs), `initial_state = ssm_state` (whole cache), `inplace_final_state=True`,
`cu_seqlens = non_spec_query_start_loc [B+1]`, `ssm_state_indices [B]`
(cache-line per seq; index 0 = NULL block → skipped, fused_sigmoid_gating.py:
110-115), scale default `Dk^-0.5`. Output `o [1, B, Hv, Dv]` in q.dtype.
An optional packed fast path (`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`,
off by default, envs) calls `fused_recurrent_gated_delta_rule_packed_decode`
(fused_recurrent.py:339-478) — same single-token math (lines 313-336)
reading q/k/v straight out of packed mixed_qkv.

## 8. Prefill/decode segmentation (v1/attention/backends/gdn_attn.py)

`GDNAttentionMetadataBuilder.build` splits the batch with
`split_decodes_and_prefills(m, decode_threshold=1)` (gdn_attn.py:211-213):
sequences with query_len == 1 are decodes, the rest prefills. Batch is
reordered decode-first (`reorder_batch_threshold = 1`, line 85). Per layer
(`_forward_core`, qwen_gdn_linear_attn.py:1260-1576):

- decode-only → `causal_conv1d_update` + `fused_sigmoid_gating_delta_rule_update`.
- any prefill → `causal_conv1d_fn` over the whole non-spec token stream, then
  decodes (if mixed) are peeled off the front through
  `fused_sigmoid_gating_delta_rule_update` (lines 1480-1499) and the prefill
  tail goes through `fused_post_conv_prep` + `chunk_gated_delta_rule`
  (lines 1422-1530) with rebased `prefill_query_start_loc`
  (gdn_attn.py:340-354). `prefill_has_initial_state = context_lens > 0`
  (gdn_attn.py:390); initial-state rows for fresh sequences are zeroed
  (qwen_gdn_linear_attn.py:1514). Final chunk state written back:
  `ssm_state[idx] = last_recurrent_state.to(ssm_state.dtype)` (line 1532).
- spec-decode branches use the same fused sigmoid-gating kernel with
  multi-token cu_seqlens and `num_accepted_tokens` (lines 1341-1357,
  1455-1475) — out of M0.7 scope; recorded for M1.x.

`has_initial_state` (conv) = `context_lens_tensor > 0` (gdn_attn.py:390);
FLA chunk metadata (`chunk_indices/offsets`) is precomputed on CPU
(gdn_attn.py:374-387) but may be None — the FLA ops recompute it on the fly
(index.py:22-37 via tensor_cache).

## 9. M0.7 golden-dump decisions (tools/parity/dump_gdn.py)

- Oracle = pinned checkout executed on dgx GPU (venv torch 2.11 + triton);
  manifests record `oracle.source = "pinned:e24d1b24"` and the exact
  callable per case.
- `gdn_prefill` bf16 cases: `chunk_gated_delta_rule` (the real prefill
  oracle). f32 prefill cases: `fused_recurrent_gated_delta_rule` (pinned
  sequential kernel) because the pinned chunk wrapper hard-rejects f32
  (chunk.py:213). The sequential-vs-chunked gap is measured on the shared
  bf16 case and recorded in its manifest (`args.chunk_vs_sequential_*`).
  **Measured on GB10 (2026-07-03)**: max|diff| out `2.44e-4`, state
  `2.25e-3` — bf16-rounding scale, comfortably inside the 1.5e-2 golden
  tolerance. (The plan's "f32 sequential vs f32 chunked ~1e-4" check is not
  runnable at this pin — chunked has no f32 path — so the bf16-input gap is
  the recorded evidence.) Decode decomposition (pinned l2norm_fwd + g/beta
  + norm-free `fused_recurrent` vs the fused sigmoid-gating kernel)
  validated at dump time to ~2e-7.
- `gdn_decode` oracle: `fused_sigmoid_gating_delta_rule_update` exactly as
  the decode call site invokes it (in-kernel l2norm + gating); derived
  l2norm-ed q/k and g/beta are ALSO stored (computed by the pinned
  `fused_post_conv_prep`) so the C++ runner can test the decomposed op chain
  L2Norm → GdnDecode(q,k,v,g,beta).
- Budget: real-dims cases keep Dk=Dv=128 (dims drive the math) but slice
  heads to Hk=1/Hv=2 (GQA ratio 2 preserved); synthetic-small cases use
  Hk=2/Hv=4 with small dims. States dumped as gathered per-sequence slices,
  f32. Total ≤ 2MB.
