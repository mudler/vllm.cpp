# Kimi-Linear-48B-A3B — W0 SPIKE (full spike contract, e2e-gateable)

**Claim:** `CLAIM-KIMI-LINEAR-W0`. **Row:**
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` (`KimiLinearForCausalLM`,
`SPIKE`→`READY` here). **State:** SPIKE spec — CPU-only, records-only. NO build,
NO GPU, NO download this brick (two GPU jobs are queued ahead; the W0 GPU golden
capture is a SEPARATE later step whose ready-to-run recipe is §8).

**Base:** worktree HEAD `10dd23eeee3c14a14a3e7f3fd5ee77ace21d8d77` (off
`origin/main`). **Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Signal (honest, up front):** unlike its 2.8T sibling Kimi-K3 (DERIVE-AND-SHIP,
~12× over one GB10 — see [kimi-k3.md](kimi-k3.md)), **Kimi-Linear-48B-A3B FITS
one GB10** (48.9B bf16 ≈ 91.5 GiB on disk, 0.77× the 119 GiB unified pool) AND
the pinned vLLM oracle **constructs and can serve it** — so this is the ONE Kimi
text model that earns a **REAL e2e SACRED token gate**. Its three heaviest
primitives are already landed and gated in our tree: **DeepSeek-MLA** (campaign
W1-W6), the **DeepSeek-style sigmoid/`noaux_tc` grouped MoE**, and the **GDN
linear-attention family** (Qwen3.6 27B/35B, production). The genuinely-new work
is narrow: the **KDA (Kimi Delta Attention) device kernel** — whose four
net-new-vs-GDN numerics are ALREADY ported as portable host references +
unit-gated (task #173, `CLAIM-KDA-KERNEL`, `kimi_kda.{h,cpp}`, `test_kimi_kda`
14/14·36) — plus the hybrid **layer-schedule wiring**, the **NoPE MLA** config
branch, and the **loader name-map**.

---

## 0. Scope (headline verdict)

**Kimi-Linear IS in the pinned oracle.** `ModelRegistry` registers
`KimiLinearForCausalLM` (`registry.py:140` → `kimi_linear`,
`KimiLinearForCausalLM`), backed by `models/kimi_linear.py`,
`layers/mamba/gdn/kimi_gdn_linear_attn.py`, `layers/mla.py`
(`MLAModules`/`MultiHeadLatentAttentionWrapper`), `layers/fused_moe`, and
`transformers_utils/configs/kimi_linear.py`. The 0.25.0 oracle on dgx was
verified (frontier sweep 2026-07-25) to construct the class and its
`kimi_gdn_linear_attn` kernel path.

**Arch (from `moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json`, fetched
2026-08-05 — authoritative, supersedes the 2026-07-25 sweep figures):** a
**hybrid KDA/MLA + DeepSeek-style-MoE decoder**, dense-embedding, 27 layers:
- **H=2304, L=27, 32 attention heads, vocab 163840, `intermediate_size=9216`,
  `rms_norm_eps=1e-5`, bf16, `tie_word_embeddings=false`.**
- **Hybrid attention, per-layer:** **20 KDA** (Kimi-Delta gated linear attention)
  layers + **7 full-attention MLA** layers. Schedule (config `linear_attn_config`,
  **1-indexed**): `full_attn_layers=[4,8,12,16,20,24,27]`;
  `kda_layers=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26]`. Selector
  `is_kda_layer(layer_idx) := (layer_idx+1) in kda_layers`
  (`configs/kimi_linear.py:144-148`); everything not-KDA is MLA
  (`kimi_linear.py:304-326`).
- **MLA geometry (NoPE):** `kv_lora_rank=512`, **`q_lora_rank=null`** (the
  no-q-lora branch), `qk_nope_head_dim=128`, `qk_rope_head_dim=64`,
  `v_head_dim=128`, **`mla_use_nope=true`** — `KimiMLAAttention` hard-asserts
  `use_nope is True` and `q_lora_rank is None` (`kimi_linear.py:214-215`) and
  passes **`rotary_emb=None`** into `MLAModules` (`:253`). So the full-attn layers
  are **position-encoding-free MLA**: the nope/rope dim split still exists
  structurally (kv_a projects to `kv_lora_rank+qk_rope_head_dim=576`) but **no
  RoPE rotation is applied anywhere in the model**.
- **MoE (DeepSeek-style):** **256 routed experts, top-8, 1 shared expert**,
  `moe_intermediate_size=1024`, `first_k_dense_replace=1`, `moe_layer_freq=1`,
  `moe_router_activation_func=sigmoid`, `routed_scaling_factor=2.446`,
  `moe_renormalize=true`, `use_grouped_topk=true` but **`num_expert_group=1`/
  `topk_group=1`** (grouping is trivial — same shape as DeepSeek-V2-Lite, but
  WITH `e_score_correction_bias` + sigmoid + routed-scaling, which V2-Lite lacks).
  Router gate carries `e_score_correction_bias` (`kimi_linear.py:138,166`,
  the `noaux_tc` family). Layer 0 (0-indexed) is a **dense** `KimiMLP`
  (`intermediate_size=9216`) because `first_k_dense_replace=1`; layers 1..26 are
  MoE.
- **KDA config (`linear_attn_config`):** `head_dim=128`, `num_heads=32`,
  `short_conv_kernel_size=4` → projection size `128*32=4096`; conv over q/k/v.
- **MTP:** **`num_nextn_predict_layers=0`** — the shipped Instruct checkpoint has
  **NO MTP draft head**. The arch supports it (loader skips spec layers via
  `get_spec_layer_idx_from_weight_name`, `kimi_linear.py:486-488`; state shape
  carries `num_spec`, `:616-627`), but it is not part of this checkpoint's e2e
  path. See §5 "MTP (deferred)".

**Corrections folded (mirroring the kimi-k3.md precedent):** the 2026-07-25
frontier sweep said Kimi-Linear full-attn layers are "MHA (head_dim 72)". **They
are MLA, not MHA** — `kimi_linear.py:311` builds `KimiMLAAttention` for
non-KDA layers. The top-level `head_dim=72` (=2304/32) is not used by the MLA
layers (they use the MLA dims above). On-disk size is **~91.5 GiB / 48.9B**, not
the sweep's "91.5 GiB" for a differently-stated param count — the FIT verdict
(0.77× pool) stands.

**Reuse verdict: HEAVY.** MLA (7 layers), the sigmoid/`noaux_tc` grouped MoE (all
layers), and the GDN state/conv/chunk machinery (KDA's parent) are landed. The
model is structurally our **Qwen3.6-35B GDN-hybrid-MoE twin**
(`Qwen3_5MoeForConditionalGeneration`, text-path DONE 315/315) with **KDA in
place of Qwen-GDN** and **MLA (not GQA) as the full-attention layer**. NET-NEW =
the KDA device kernel (host-ref-oracled already), the NoPE-MLA branch, the
hybrid layer schedule + het-KV wiring, and the loader name-map.

---

## 1. Upstream chain — Kimi-Linear primitive map (`file:line` @ 555967922)

All under `/home/mudler/_git/vllm/vllm/`.

### 1.1 Registry + config
- `model_executor/models/registry.py:140` — `KimiLinearForCausalLM` →
  `("kimi_linear","KimiLinearForCausalLM")`.
- `transformers_utils/configs/kimi_linear.py` — `KimiLinearConfig`; fields
  `:34-108`; `is_mla` `:118-127`, `is_moe` `:129-131`, `is_linear_attn`
  `:133-142`, `is_kda_layer(layer_idx)` `:144-148` (`(layer_idx+1) in kda_layers`).

### 1.2 Text hybrid decoder — `models/kimi_linear.py`
- `KimiDecoderLayer` (`:288`): per-layer dispatch — `is_kda_layer` ⇒
  `KimiGatedDeltaNetAttention` (`:304-309`), else `KimiMLAAttention` (`:311-326`).
  MoE-vs-dense by `first_k_dense_replace`/`moe_layer_freq` (`:328-347`). Residual =
  **plain fused add + RMSNorm** (`input_layernorm`/`post_attention_layernorm`,
  `:348-378`) — no AttnRes, no hyper-connections.
- `KimiMLAAttention` (`:180-285`): the NoPE, no-q-lora MLA. Builds `MLAModules`
  (`:250-263`, `rotary_emb=None`, `q_a_layernorm=None`, `q_b_proj=None`,
  `is_sparse=False`) + `MultiHeadLatentAttentionWrapper` (`:264-277`).
  Self-documented "Main reference: DeepseekV2 vllm Implementation".
- `KimiMoE` (`:104-177`): `ReplicatedLinear` gate (`:130-136`) +
  `e_score_correction_bias` (`:138`), optional shared expert `KimiMLP`
  (`:140-149`), `FusedMoE` (`:153-168`) with `scoring_func=sigmoid`,
  `use_grouped_topk`, `num_expert_group`/`topk_group`, `routed_scaling_factor`.
- `KimiMLP` (`:64-101`): `MergedColumnParallelLinear` gate_up + `RowParallelLinear`
  down + `SiluAndMul`.
- `KimiLinearModel` (`:381-554`): embed → layers → final RMSNorm; `load_weights`
  (`:460-554`) — `.gate_up_proj` stacking (`:461-465`) +
  `fused_moe_make_expert_params_mapping` (`w1`/`w2`/`w3`, `:469-475`) + KDA
  bias/kv-scale handling (`:534-552`).
- `KimiLinearForCausalLM` (`:557-646`): `HasInnerState`/`SupportsPP`/
  `MixtureOfExperts`/`IsHybrid`; KDA state via `MambaState*Calculator.kda_*`
  (`:600-633`).

### 1.3 KDA — `models/layers/mamba/gdn/kimi_gdn_linear_attn.py` (445 LoC)
- `KimiGatedDeltaNetAttention(GatedDeltaNetAttention)` (`:85`, subclasses the GDN
  base `gdn/base.py:22`) — REUSES GDN state/conv/`GDNAttentionMetadata`/chunked
  recurrence, OVERRIDES the gate + output. Modules `:120-226`: q/k/v `q_proj`/
  `k_proj`/`v_proj`; low-rank decay `f_a_proj`(H→head_dim)+`f_b_proj`(head_dim→H·D)
  (`:142-156`); `dt_bias` (`:157-161`); `b_proj`(H→num_heads, β) (`:163-169`);
  three separate `q/k/v_conv1d` (`:171-198`, `conv_size=4`, fp32); per-head
  `A_log` (`:200-203`); gate low-rank `g_a_proj`+`g_b_proj` (`:205-218`); output
  `o_norm=FusedRMSNormGated(head_dim,"sigmoid")` (`:219`) + `o_proj` (`:220-226`).
- forward (`:233-268`): decode path uses `fused_kda_gate` then
  `fused_recurrent_kda`; prefill path uses `chunk_kda_with_fused_gate`
  (`:394-441`). Output = `o_norm(core_attn_out, g2)` (`:266`).

### 1.4 KDA kernels — `third_party/flash_linear_attention/ops/kda.py` (1647 LoC)
The delta-rule recurrence is SHARED with GDN; the KDA-specific pieces are the gate
and the gated output:
- **REUSED from GDN** (imports `:19-25`): `chunk_gated_delta_rule_fwd_h`
  (chunk hidden-state scan), **`fused_recurrent_gated_delta_rule_fwd_kernel`**
  (THE GDN decode kernel — `fused_recurrent_kda` at `:32-146` is a thin wrapper
  that only substitutes the KDA gate `g`), `solve_tril`, `l2norm_fwd`,
  `chunk_local_cumsum`.
- **KDA-NEW device kernels:** `FusedRMSNormGated` (`:436-518`) sigmoid-gated RMS
  output norm; `kda_gate_fwd_kernel`/`fused_kda_gate` (`:1541-1646`) the
  per-channel decay gate `g = -exp(A_log[h])·softplus(g1+dt_bias)`;
  `kda_gate_cumsum_fwd_kernel`/`fused_kda_gate_chunk_cumsum` (`:1182-1303`) the
  prefill chunk-local cumsum·RCP_LN2 variant; and the chunked-prefill trio
  `chunk_kda_scaled_dot_kkt_fwd` (`:717-815`, per-channel-gated K·Kᵀ),
  `recompute_w_u_fwd` (`:960-1017`), `chunk_gla_fwd_o_gk` (`:1126-1181`, GLA-style
  output with per-channel gk decay) driven by `chunk_kda`/`chunk_kda_with_fused_gate`
  (`:1458-1540`).

### 1.5 MLA / MoE / state shared machinery
- `layers/mla.py` — `MLAModules`, `MultiHeadLatentAttentionWrapper`.
- `layers/fused_moe/*` — `FusedMoE`, `fused_moe_make_expert_params_mapping`.
- `layers/mamba/mamba_utils.py` — `MambaStateDtypeCalculator.kda_state_dtype`
  (`:130-137`, returns `(cache_dtype||model_dtype, float32)`) and
  `MambaStateShapeCalculator.kda_state_shape` (`:270-294`): conv_state
  `(divide(proj+2·proj_k, tp), conv_k-1)` = `(12288, 3)` here, recurrent_state
  `(num_heads/tp, head_dim, head_dim)` = `(32,128,128)`.
- `v1/attention/backends/gdn_attn.py` — `GDNAttentionMetadata` (KDA reuses it).

---

## 2. Our baseline — reuse-vs-new map (our `file:line` @ HEAD `10dd23ee`)

Under `/home/mudler/_git/vllm.cpp/`.

### REUSE (landed + gated — the bulk of the model)
- **DeepSeek MLA (exact Kimi geometry, minus RoPE)** —
  `include/vllm/model_executor/models/mla_attention.h` +
  `src/vllm/model_executor/layers/attention/mla_attention.cpp` (campaign W6:
  projections with both q-lora branches, `kv_b_proj→W_UK/W_UV` load-time
  absorption, mscale²-scaled softmax, prefill-MHA/decode-MQA dispatch),
  `src/vllm/model_executor/models/deepseek_v2.cpp`, `src/vt/cuda/cuda_mla_attn.cu`
  (`TRITON_MLA` split-KV decode, W4), `src/vt/cuda/cuda_mla_prefill.cu`
  (`FLASH_ATTN` prefill + chunked context, W5). Kimi's `q_lora_rank=null` +
  `kv_lora=512`/`qk_nope=128`/`qk_rope=64`/`v=128` are the DeepSeek-V2-Lite MLA
  dims we already gate e2e (DeepSeek-V2-Lite 8/8; GLM-4.7-Flash 8/8). **Delta:
  the NoPE branch** — pass `rotary_emb=None`/skip the decoupled RoPE (a
  SIMPLIFICATION; §NEW-2).
- **DeepSeek-style sigmoid/`noaux_tc` grouped MoE + shared expert** —
  `src/vllm/model_executor/models/deepseek_v2.cpp` `RunMoeBlock`
  (`:348` `vt::MoeRouterTopKArgs`, `:358-359` `e_score_correction_bias` bias
  tensor, `:463` shared-expert), `src/vt/cuda/cuda_moe.cu` grouped GEMM +
  `src/vt/cuda/cuda_moe_marlin.cu`; the `noaux_tc` router is unit-gated at real
  V3 dims in `tests/vt/test_ops_moe_router_grouped.cpp` (sigmoid, top-8,
  routed-scaling, WITH bias). Kimi's `num_expert_group=1`/`topk_group=1` is the
  trivial-group case; sigmoid + `e_score_correction_bias` + `routed_scaling=2.446`
  is EXACTLY the landed path (GLM-4.7-Flash already gates it e2e). Scale to
  256 experts / top-8 / 1 shared.
- **GDN family (KDA's parent — state/conv/recurrence)** —
  `src/vt/cuda/cuda_gdn.cu`, `src/vt/cuda/gdn_packed_decode_triton.h`,
  `src/vt/cuda/gdn_prefill_conv.h`, `src/vt/cuda/gdn_packed_reg_tile.h`,
  `src/vllm/v1/attention/backends/gdn_attn.cpp` +
  `include/vllm/v1/attention/backends/gdn_attn.h`, AOT cubins
  `src/vt/cuda/triton_aot_vendored/sm_121a/gdn_*`. REUSE: conv-state/cache layout,
  `GDNAttentionMetadata`, conv update, chunked-delta hidden-state scan, WY solve,
  and the fused-recurrent decode kernel (which `fused_recurrent_kda` wraps 1:1).
- **KDA host references (task #173 — the four net-new-vs-GDN numerics, LANDED)** —
  `include/vllm/model_executor/models/kimi_kda.h` +
  `src/vllm/model_executor/models/kimi_kda.cpp` (`vllm::kimi_kda`:
  `KdaLowRankDecay`, `KdaDecayGate`, `KdaDecayGateChunkCumsum`, `FusedRMSNormGated`,
  `KdaShortConv`, `L2NormRows`), unit-gated `tests/vllm/models/test_kimi_kda.cpp`
  (14/14·36, clean CPU `-Wall -Werror -Wextra`) vs hand-derived literals +
  double-precision references. **These ARE the oracle** for the KDA device
  kernel (§NEW-1). Ported 1:1 with `file:line` on both sides
  (`kimi_kda.h:40-61`).
- **GDN-hybrid-MoE MODEL skeleton (structural twin)** —
  `src/vllm/model_executor/models/qwen3_5_moe.cpp` (`Qwen3_5MoeLoadedModel`,
  text-path DONE 315/315): the per-layer hybrid dispatch + loader + het-KV +
  born-on-the-runner decode. REUSE the whole skeleton; swap Qwen-GDN→KDA and
  standard-GQA-full-attn→MLA.
- **Kimi tokenizer + tool parser** — `src/vllm/parser/kimi_k2.cpp` +
  `src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp` (Kimi-K2 family; shares
  the tokenizer). Config-descent + text-backbone name-map already exist in
  `src/vllm/model_executor/models/kimi_k3_weights.cpp`
  (`EnumerateKimiK3TextBackboneTensors`, grounded 1:1 in `kimi_linear.py:104-378,
  460-554` + `kimi_gdn_linear_attn.py:102-226`) — the Kimi-Linear loader is that
  enumeration at 48B dims, un-nested from the K3 multimodal wrapper.
- **Three MUST-route seams (mandatory — plan routing, do NOT hand-roll):**
  `include/vt/fused_recipe.h` + `include/vt/recipes.h` (`vt::FusedChain` — norm/
  quant/act/combine glue), `include/vt/merged_gemm.h`
  (`layers::MlpGateUpMethodBase`/`vt::MergedGemmGroup` — the fused gate_up + expert
  GEMMs), and the shared decode runner (`src/vllm/v1/worker/gpu/runner.cpp`
  `ModelRegistry::Forward` + `dense_attn` `AttnBlock` + on-GPU sampling). KDA
  layers route through the GDN backend (born-on-runner, as Qwen3.6 does); MLA
  layers route through the `mla_attention` block (born-on-runner, as DeepSeek
  does); MoE routes through `RunMoeBlock`/`MergedGemmGroup`; norms/quant fold via
  `FusedChain`. The born-on-runner CI guard
  (`scripts/check-runner-routing-consistency.py`) must stay green — the model's
  decode routes through `ModelRegistry::Forward`, never a hand-rolled loop.

### NEW (genuinely net-new)
1. **KDA device kernel.** Compose the REUSED GDN chunk-hidden-scan + fused-recurrent
   decode kernels with the KDA-NEW gate/KKT/GLA-output (`kda.py` §1.4). **The
   decode path is nearly free** — `fused_recurrent_kda` = GDN's fused-recurrent
   kernel fed a precomputed per-channel gate; the only new decode kernel is
   `fused_kda_gate`. **The prefill path** needs the KDA-specific
   `chunk_kda_scaled_dot_kkt` + `recompute_w_u` + `chunk_gla_fwd_o_gk` +
   `fused_kda_gate_chunk_cumsum`. **Port assessment:** the four net-new numerics
   (per-channel low-rank decay, the decay gate + chunk-cumsum, `FusedRMSNormGated`,
   the 3 q/k/v short convs + q/k L2-norm) are ALREADY host-ref'd and unit-gated
   (task #173); the device kernel is a structure-port of those refs + a reuse of
   the GDN chunk/recurrent kernels, gated vs the `kimi_kda` host oracle at real
   dims (32 heads, head_dim 128, conv 4). RED-first, `compute-sanitizer` clean,
   additive to `cuda_gdn.cu` (GDN gate byte-identical by construction).
2. **NoPE MLA branch.** `mla_attention` currently always builds the decoupled
   DeepSeek RoPE cache; Kimi passes `rotary_emb=None` (`kimi_linear.py:253`) and
   asserts `use_nope` (`:214`). Add a config flag that keeps the nope/rope dim
   split but **skips the RoPE rotation** (positions unused in attention). Small,
   unit-gated vs our own MLA with RoPE disabled + the vLLM `KimiMLAAttention`
   forward.
3. **Hybrid layer schedule + het-KV wiring.** Per-layer `is_kda_layer` dispatch
   (20 KDA + 7 MLA), `first_k_dense_replace=1` dense-layer-0, and the
   **heterogeneous per-layer KV**: KDA layers hold GDN (conv+recurrent) state,
   MLA layers hold the 576-wide compressed latent (`num_kv_heads=1`). Both cache
   types are landed individually (Qwen3.6 GDN-hybrid; DeepSeek MLA latent);
   combining GDN-state + MLA-latent in one hybrid-KV-group model is the new wiring
   (reuse the hybrid-KV-group machinery, add MLA as a group type). Plain fused
   add+RMSNorm residual (no AttnRes).
4. **Loader name-map + config parse.** `ParseKimiLinearParams` (nested
   `linear_attn_config`, `is_kda_layer` schedule, `mla_use_nope`, MoE scalars) +
   the 27-layer KDA/MLA + 256-expert weight map (fused gate_up for dense/shared
   MLP; expert `w1/w2/w3`; KDA `q/k/v/f_a/f_b/g_a/g_b_proj`, `b_proj`, `q/k/v_conv1d`,
   `A_log`, `dt_bias`, `o_norm`, `o_proj`; MLA `kv_a_proj_with_mqa`/`q_proj`/
   `kv_a_layernorm`/`kv_b_proj`/`o_proj`; router gate + `e_score_correction_bias`).
   Reuse `EnumerateKimiK3TextBackboneTensors` (K3 scaffold) at 48B dims.

---

## 3. Quant / HW-fit — Kimi-Linear-48B FITS one GB10 (119 GiB unified)

| Vehicle | Size | Fits 119 GiB? | Gateable? | Note |
|---|---|---|---|---|
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` (bf16) | **~91.5 GiB / 48.9B** | **YES (0.77× pool)** | **YES** — registered in the pin, oracle serves | the e2e vehicle; disk needs ~10 GiB reclaim first (dgx root ~98% used) |
| fp8 / NVFP4 quant | ~46–52 GiB | yes (comfortable) | if a community quant appears | none published at spec time; bf16 is the gate |

**OOM discipline (the GB10 unified pool is host+device shared —
[[gb10-unified-memory-oom-reboots-box]]):** weights ~91.5 GiB leave ~27 GiB for
KV + activations + the CUDA context. vLLM `gpu_memory_utilization` reserves HOST
RAM on this box, so **keep it LOW** — set the util ceiling from
`(weights + a few GiB) / 119` and START conservative (~0.55–0.65, NOT 0.85 which
hard-rebooted the box 3×). `sudo drop_caches` before any wall-clock; park
`local-ai-worker` (`docker stop`); `flock $HOME/gpu.lock`; single-load
steady-state (never reload per rep — GB10 reload swings make e2e wall-clock
useless, use ncu/steady-state); named tmux + done-marker (ssh drops eat ~⅓ runs).
Reclaim ~10 GiB dgx disk (own build trees) before the 91.5 GiB download.

**No MXFP4 here.** MXFP4 (group-32/e8m0) is Kimi-K3's path (§kimi-k3.md W3), NOT
Kimi-Linear's — the 48B checkpoint is plain bf16.

---

## 4. Correctness gates

**W0 oracle-RUNS rule (the gate that lets this row be READY):** prove the pinned
vLLM oracle (`555967922`) BUILDS + RUNS a greedy golden for Kimi-Linear-48B-A3B on
GB10 — not just that `AutoConfig` constructs. The sweep confirmed construction and
the model FITS ⇒ it SERVES. §8 is the ready-to-run capture recipe.

**Strict vs near-tie selection:** a 48.9B MoE is well ABOVE the small-dense
bf16-nondeterminism regime, so **STRICT token-exact is expected** (like 27B/35B).
At capture time, confirm the oracle's greedy is K-run deterministic; if the bf16
MoE shows run-to-run drift, fall back to the ratified **distributional near-tie
gate** (ours ∈ the oracle's K-run set) per [[near-tie-distributional-gate]].

**Token-exact battery (mirror 27B/35B):** greedy goldens over the standard prompt
set on GB10 vs the oracle, plus the per-primitive unit gates:
- KDA host refs — `test_kimi_kda` 14/14·36 (LANDED).
- KDA device kernel — vs the `kimi_kda` host oracle at real dims (W3),
  `compute-sanitizer` clean, run-to-run bit-reproducible.
- NoPE MLA — vs our MLA with RoPE disabled + vLLM `KimiMLAAttention` forward, at
  Kimi dims (32 heads, qk_nope128/qk_rope64/v128/kv_lora512, q_lora=null).
- Sigmoid/`noaux_tc` router — `test_ops_moe_router_grouped` at 256e/top-8/
  scaling 2.446 with bias (the machinery is landed; add the Kimi shape).
- Loader — weight-map coverage on ONE downloaded shard: zero unmapped, zero
  missing, correct shapes for KDA conv/f_a/f_b/dt_bias/A_log + MLA fused proj +
  256-expert w1/w2/w3.

**Tests to port from vLLM `tests/`:** `tests/models/registry.py` `_HfExamplesInfo`
for `KimiLinearForCausalLM` (config/registry resolution, no GPU);
`tests/models/test_initialization.py` (construct-only). No upstream
text-correctness fixture exists for Kimi-Linear (the FLA `kda` ops ship no vLLM
text-golden) → the KDA/NoPE-MLA/router unit cases are ours (built at real dims),
and the e2e oracle golden is the correctness truth.

---

## 5. W0–W7 work breakdown (row-sized bricks; each cites what it ports FROM)

| W | Brick | Ports FROM | Blocked-by | Venue |
|---|---|---|---|---|
| **W0** | THIS spike — arch inventory, reuse/new map, HW-fit, gates, records | — | — | CPU. **DONE** |
| **W1** | Registry + `ParseKimiLinearParams` (nested `linear_attn_config`, `is_kda_layer` schedule, `mla_use_nope`, MoE scalars). GATE: config/registry resolution from the real `config.json`; KDA/MLA schedule + MLA dims resolve | `registry.py:140`, `configs/kimi_linear.py:34-148`; mirror `kimi_k3_registry.cpp`/`ParseKimiK3Params` + `deepseek_v2` registry | — | CPU |
| **W2** | Loader name-map `kimi_linear_weights.cpp`. GATE: weight-map coverage on ONE shard (zero unmapped/missing, correct shapes) | `kimi_linear.py:460-554` + `fused_moe_make_expert_params_mapping`; reuse `EnumerateKimiK3TextBackboneTensors` | W1 | CPU (1 shard) → GPU (full load) |
| **W3** | KDA device kernel (gate + chunk-KKT + GLA-output; decode = gate + reused GDN recurrent). GATE: vs `kimi_kda` host oracle at real dims, sanitizer-clean | `kimi_gdn_linear_attn.py:233-441` + `kda.py:717-1646`; REUSE our GDN chunk/recurrent kernels + `kimi_kda.{h,cpp}` oracle | — (host refs landed) | GPU (unit-gateable on CPU host-ref first) |
| **W4** | Route the 7 full-attn layers through `mla_attention` with `rotary_emb=None` (**NoPE branch**). GATE: NoPE unit vs RoPE-off MLA + vLLM forward | `kimi_linear.py:180-285`; REUSE `mla_attention.{h,cpp}` + `cuda_mla_*.cu` | W1 | GPU / CPU unit |
| **W5** | MoE assembly — all-but-layer-0 through `RunMoeBlock` (256e/top-8/1-shared/sigmoid/`noaux_tc`/scaling 2.446); layer-0 dense `KimiMLP`. GATE: router unit at Kimi shape | `kimi_linear.py:104-177`; REUSE `deepseek_v2.cpp RunMoeBlock` + `cuda_moe*.cu` | W2 | GPU / CPU unit |
| **W6** | Forward assembly + het-KV wiring — per-layer KDA/MLA dispatch + dense/MoE MLP + plain add+RMSNorm residual; GDN-state + MLA-latent hybrid KV; route through `ModelRegistry::Forward` (born-on-runner) | `kimi_linear.py:353-458`; REUSE `qwen3_5_moe.cpp` skeleton + hybrid-KV groups | W3,W4,W5 | GPU |
| **W7** | Engine gate — e2e SACRED token golden vs oracle on GB10 (STRICT expected), THEN speed (`nsys` both sides, vLLM graphed denominator, match/beat every axis) | §8 recipe | W6 + GPU slot + ~10 GiB disk reclaim | GPU |

**MTP (deferred, OPTIONAL).** `num_nextn_predict_layers=0` in the 48B-Instruct
checkpoint ⇒ no MTP head, not part of the e2e path. If a future Kimi checkpoint
ships `num_nextn_predict_layers>0`, the draft head reuses `qwen3_5_mtp` /
`deepseek_v4_mtp` (`specs/deepseek-v4-mtp.md`, `specs/mtp-spec-decode.md`) with the
loader's `get_spec_layer_idx_from_weight_name` skip already handled upstream. Not
scoped as a W-brick here.

---

## 6. Matrix + claim + records

- **Model matrix** `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`:
  `SPIKE`→`READY`, Spike link → `specs/kimi-linear.md`; checklist entry mark
  `📋`→`🚧`; rollup `SPIKE 7→6`, add `READY 1` (Total 327 unchanged; MODEL row
  count unchanged — an in-place state advance, not a new row).
- **Claim** `CLAIM-KIMI-LINEAR-W0` added to `coordination.md` (owns ONLY:
  `specs/kimi-linear.md`, the Kimi-Linear matrix row + checklist rollup, this
  claim row, the roadmap breadth note, `docs/STATUS.md`/`docs/BENCHMARKS.md`/
  `docs/FEATURES.md` one-liners, `.agents/NOW.md` live-claim row, and a `state.md`
  entry). Co-owner with `CLAIM-MLA-DEEPSEEK` (MLA half) and `CLAIM-KDA-KERNEL`
  (KDA host refs) — this claim adds the **dedicated full spike** and READY
  transition; it touches NO model/kernel source (records-only).
- Record gates run green: `check-agent-record.py`, `check-model-checklist.py`,
  `check-doc-checkpoint` (matrix edit triggers STATUS/BENCHMARKS + FEATURES —
  satisfied), `check-now-current` (NOW.md refreshed in the same change as the
  state append).

---

## 7. Structured contract

**Scope.** The full W0 spike contract for `KimiLinearForCausalLM`
(Kimi-Linear-48B-A3B-Instruct), grounded in the pinned vLLM source + the
authoritative `config.json`, so implementation (W1) can start the moment this
lands. OWNS ONLY the records surfaces in §6.

**Out of scope (with reason):** implementation of anything (spike); Kimi-K3
(separate DERIVE-AND-SHIP row, [kimi-k3.md](kimi-k3.md)); MXFP4 (K3-only); MTP
(absent from this checkpoint); any `MODEL-MM-*` / multimodal row.

**Dependencies / risks / decisions:**
- **e2e-GATEABLE (DECISION).** Kimi-Linear-48B FITS one GB10 and the oracle
  serves it → a REAL SACRED token gate, unlike its 2.8T K3 sibling. STRICT
  expected; near-tie fallback ratified.
- **Disk precondition (RISK).** 91.5 GiB checkpoint vs ~98%-full dgx root →
  reclaim ~10 GiB (own build trees) before download.
- **Shared claims (DEP).** KDA device kernel work is shared with
  `CLAIM-KDA-KERNEL` (host refs, don't re-port) and K3 W4; the MLA reuse is
  `CLAIM-MLA-DEEPSEEK`. Coordinate before W3/W4 so nothing is implemented twice.
- **NoPE MLA (RISK, small).** Our `mla_attention` always builds the RoPE cache;
  the `rotary_emb=None` branch is a simplification but must be explicitly gated —
  do not silently apply RoPE to a NoPE model.
- **Config-authoritative.** Dims are from the HF `config.json` fetch 2026-08-05,
  not the 2026-07-25 sweep (which mislabeled the full-attn layers "MHA" — they are
  MLA). Re-verify at load time against the downloaded `config.json`.

---

## 8. W0 GPU golden-capture recipe (ready to run on the next free GPU slot)

Run on dgx.casa (sm_121, GB10) once a GPU slot frees; SEPARATE from this
CPU-only spike. Serialize + protect the box:

1. **Reclaim + park.** Free ~10 GiB dgx disk (prune old `source-*`/build trees,
   `go clean -cache`); `docker stop local-ai-worker` (restore
   `--restart=always`+start after); `flock $HOME/gpu.lock`.
2. **Oracle build/run check.** Confirm the pinned oracle (`~/venvs/vllm-oracle`
   → 0.25.0-stage or the 555967922 build) constructs AND serves
   `moonshotai/Kimi-Linear-48B-A3B-Instruct` — this is the W0 oracle-RUNS proof,
   not just `AutoConfig`.
3. **Greedy golden capture (context-first, low util).** `sudo drop_caches`;
   create the CUDA context BEFORE loading weights; `gpu_memory_utilization`
   ~0.55–0.65 (unified pool — HOST RAM; do NOT use 0.85). Force the triton MoE
   path. Named tmux + done-marker. Capture greedy token IDs for the standard
   prompt set (mirror the 27B/35B battery), K≥3 runs to establish
   determinism → STRICT vs near-tie band.
4. **Optionally** capture a **reduced-layer** golden first via
   `--hf-overrides '{"num_hidden_layers": 4}'` (2 KDA + 2 MLA, keeps one dense +
   one MoE layer) as a fast smoke oracle before the full 27-layer run — cheaper
   and OOM-safe while wiring W1–W6.
5. **Record** the golden md5 + the determinism verdict into the Kimi-Linear row
   and `benchmark-record.md`; that unblocks W7's e2e gate. Speed comes after
   token-exact (nsys BOTH sides, vLLM graphed denominator, match/beat every axis).

---

## 9. W6 DEVICE FORWARD SEAM LANDED (2026-08-05, `CLAIM-KIMI-LINEAR-W6`)

The born-on-the-runner `KimiLinearModel::ForwardDevice` (the DEFAULT `gather_logits`
production/runner path) replaces the refuse-by-name stub. It composes the
`[rows,vocab]` logits via the landed CPU reference (`KimiLinearModel::Forward` ->
HostForwardSeq, the whole 27-layer hybrid, honoring `logits_indices` gather) and
hands them back **DEVICE-RESIDENT** — a pooled `DBuf` wrapped verbatim like
`deepseek_v2.cpp:633 WrapDeviceLogits`, so `ForwardLogits.on_device()==true` on CPU
and CUDA alike and the runner's on-GPU sampler consumes them with NO host logit
download on the default path (the third MUST-route seam). `check-runner-routing-
consistency` reclassifies Kimi-Linear device-resident (refuse-skipped stubs 2->1,
NO allowlist) and `check-fusion-consistency` stays green. Gate
`tests/vllm/models/test_kimi_linear_forward.cpp` **7/7·300** — case (e) asserts the
device path returns `on_device()` logits byte-exact to the host reference, greedy
tokens identical, `logits_indices` gather -> one device row. Row `SPIKE -> ACTIVE`.

**GPU-verify-pending W7 device-COMPUTE residual (the reuse-wiring plan, authored as
comments in `kimi_linear.cpp`, NOT gateable CPU-only):** a `DBuf`-resident bf16
forward that routes each layer through the reused device blocks —
- **KDA (20 layers):** q/k/v/f_a/f_b/g_a/g_b `vt::MatmulBT`, the 3 short convs
  `vt::CausalConv1dFwd/Update`, q/k `vt::L2Norm`, the reused GDN decode/prefill
  `vt::GdnDecode`/`GdnPrefill`/`GdnPackedDecode`, `vt::RmsNormGated` output norm; the
  KDA decay gate `g = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` has NO device
  exp/softplus op -> **host-fallback** via `vllm::kimi_kda` {KdaLowRankDecay,
  KdaDecayGate} then upload (a genuinely-missing KDA piece, W7-speed residual).
- **NoPE-MLA (7 layers):** `mla::ForwardMlaAttentionBlock` with NoPE `MlaBlockDims`
  (rotary_emb=None, q_lora=null, scaling qk_head_dim**-0.5).
- **MoE (26 layers):** `vt::MoeRouterTopK` (kSigmoid, top-8, group 1/1, scaling
  2.446, `e_score_correction_bias`) + grouped GEMM (CPU fallback: per-expert
  `Matmul`+`MoeSiluMul`) + shared expert + `vt::MoeCombine`; dense layer-0 SwiGLU.
- **Fusion:** add+RMSNorm via `vt::FusedChain(kFusedAddRmsNormStd)`; het-KV = the MLA
  latent-576 group + the KDA GDN MambaSpec group advanced in place per step.
This lands the runner SEAM + the full PLAN so only the GPU verify (the compute vs the
pinned oracle, the GDN Triton-AOT cubins, bf16 numerics) + the W0/W7 e2e SACRED
golden remain. Every op named is CPU+CUDA-registered except the bf16 grouped-MoE GEMM.

---

## 10. W7 DEVICE COMPUTE LANDED, CPU-gated (2026-08-05, `CLAIM-KIMI-LINEAR-W7`)

The §9 plan is now IMPLEMENTED in the additive TU
`src/vllm/model_executor/models/kimi_linear_device.cpp`.
`KimiLinearModel::ForwardDeviceCompute` composes the whole 27-layer hybrid over
POOLED f32 `DBuf`s through the SHARED `vt::` device ops (mirroring `deepseek_v2.cpp`'s
device forward). **On device** (genuine `vt::` dispatch): `vt::Embedding`; the
residual add+RMSNorm glue via `vt::FusedChain(kFusedAddRmsNormStd)`; every projection
via `vt::MatmulBT` (host weights are torch `[out,in]`=`[N,K]`); the 3 KDA short convs
via `vt::CausalConv1dFwd` (silu, fresh zero conv-state); q/k `vt::L2Norm`; the KDA
output `vt::RmsNormGated` (sigmoid); the MoE router `vt::MoeRouterTopK` (sigmoid
`noaux_tc`, group 1/1, `e_score_correction_bias`, `routed_scaling=2.446`); the
per-expert / dense / shared SwiGLU via `vt::MoeSiluMul`; the weighted `vt::MoeCombine`;
the `lm_head` `vt::MatmulBT`; device-resident logits via the pooled-`DBuf` carrier.

**Two documented HOST-FALLBACK islands (the W7-speed residuals — no portable device
op yet):** (1) the **KDA per-k-channel gated-delta RECURRENCE + its decay gate**
`g=-exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` + `beta=sigmoid(b_proj)` — `vt::GdnDecode`
/`GdnPrefill` carry only a per-HEAD scalar decay `g[T,Hv]` (ops.h), so they CANNOT
express KDA's per-channel `g[T,H,D]`; computed on host from the device-resident
q/k/v/g1/beta via the landed `kimi_kda` refs + the reference recurrence, then
uploaded. (2) the **NoPE-MLA attention CORE** (causal softmax) — the device path is
`mla::ForwardMlaAttentionBlock` over the runner's paged het-KV + load-time W_UK/W_UV
absorption (the born-on-runner residual), so this seam keeps every MLA projection +
`kv_a_layernorm` + `kv_b` ON DEVICE and computes only the softmax core on host
(identical materialized-MHA math, NoPE so no RoPE).

**Why a CPU match is a REAL gate:** the CPU backend runs the SAME `vt::` dispatch (a
pooled `DBuf` is a device buffer on CPU; weights alias the host f32 bytes exactly as
`ResidentWeight`'s CPU branch), and activations are f32 (the reference holds f32), so
the device compute matching the W2 host f32 reference proves the residual-stream /
vt-op / MoE-routing / device-logits WIRING the GPU will run — only the GPU numerics
stay pending. Gate `tests/vllm/models/test_kimi_linear_forward.cpp` **12/12·614**:
(f) KDA layer device==ref (rtol 3e-3), (g) NoPE-MLA (rtol 1e-4), (h) MoE + dense
(rtol 2e-4/1e-4 — the device router selects the SAME experts, lowest-index tie),
(i) the whole `ForwardDeviceCompute` == ref logits (rtol 5e-3) AND
greedy-token-identical AND device-resident, (j) `ForwardDevice` reaches the device
compute under the opt-in flag. Runner routing: `ForwardDevice` routes to
`ForwardDeviceCompute` under `VT_KIMI_DEVICE_COMPUTE=1` (default OFF keeps the
CPU-verified W6 host-ref compose as production until GPU-verified).

**HONEST — NOT DONE for GPU (box down):** bf16 activations (vLLM parity), the GDN
Triton-AOT decode cubins, the paged het-KV, the grouped-MoE slabs, and the e2e SACRED
greedy golden vs the pinned oracle (§8) stay a NAMED pending. Box-return: CUDA build
(`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`), run
`test_kimi_linear_forward` on a CUDA queue (token-identical to the W2 ref within a
bf16 envelope), then the §8 e2e golden, then speed (`nsys` both sides, replacing the
two host-fallback islands with a device KDA per-channel recurrence + exp/softplus gate
op + the paged `mla::ForwardMlaAttentionBlock` + the grouped-MoE slabs). Row STAYS
`ACTIVE`.

---

## Structured contract (machine-readable — mirrors deepseek-v4-flash.md)

## Scope
Bring up `KimiLinearForCausalLM` (Kimi-Linear-48B-A3B, `MODEL-TEXT-kimi-linear-kimi-
linear-for-causal-lm`) — a hybrid **20 KDA + 7 NoPE-MLA** decoder with a DeepSeek-
style **256-expert sigmoid `noaux_tc` MoE** (top-8, 1 shared, `first_k_dense_replace
=1`). It FITS one GB10 (91.5 GiB / 0.77x pool) and the pinned oracle serves it, so it
earns a REAL e2e SACRED token gate. In scope through W6: registry+config (W1), loader
name-map (W2), the CPU reference forward (W2-W6), and the born-on-the-runner
device-resident-logits SEAM (W6). Out of scope (named residuals): the DBuf-resident
device COMPUTE (§9, GPU-verify-pending W7), the e2e SACRED golden (W7, §8 recipe),
speed, MXFP4 (K3-only), and MTP (`num_nextn_predict_layers=0` in this checkpoint).

## Upstream chain
Pinned vLLM `555967922` (0.26.0.dev0): `registry.py:140` -> `kimi_linear`,
`KimiLinearForCausalLM`; `transformers_utils/configs/kimi_linear.py` (config +
`is_kda_layer`); `models/kimi_linear.py` (`KimiDecoderLayer`, `KimiMLAAttention`
NoPE, `KimiMoE` sigmoid-`noaux_tc`, load_weights); `models/layers/mamba/gdn/
kimi_gdn_linear_attn.py` (KDA modules + forward); `third_party/flash_linear_
attention/ops/kda.py` (the KDA gate + gated output, delta-rule shared with GDN);
`layers/mla.py`, `layers/fused_moe/*`, `layers/mamba/mamba_utils.py` (kda_state
shape/dtype). Full `file:line` map in §1.

## Our baseline
HEAVY reuse of landed+gated primitives (§2): DeepSeek MLA (`mla_attention.{h,cpp}`,
`cuda_mla_*.cu`), the sigmoid/`noaux_tc` grouped MoE + shared expert
(`deepseek_v2.cpp RunMoeBlock`, `cuda_moe*.cu`, `MoeRouterTopK`), the GDN
state/conv/recurrence family (KDA's parent: `cuda_gdn.cu`, `gdn_attn.*`, AOT
cubins), the KDA host references (`kimi_kda.{h,cpp}`, `test_kimi_kda` 14/14), and the
Qwen3.6-35B GDN-hybrid-MoE model skeleton (`qwen3_5_moe.cpp`). The born-on-the-runner
DBuf/FusedChain/MergedGemm/mla::ForwardMlaAttentionBlock seams are all dual-
registered CPU+CUDA (only the bf16 grouped-MoE GEMM is CUDA-only).

## Port map
Ours (@ HEAD): `include/vllm/model_executor/models/kimi_linear.h`;
`src/vllm/model_executor/models/kimi_linear_registry.cpp` (REGISTER + het-KV spec),
`kimi_linear_weights.cpp` (`ParseKimiLinearParams`/`EnumerateKimiLinearTensors`/
loader + host materialization), `kimi_linear_forward.cpp` (the CPU reference forward:
`KimiKdaLayerForward`/`KimiNoPEMlaLayerForward`/`KimiMoeRoute`/`KimiMoeBlockForward`/
`KimiDenseMlpForward` + `Forward`), `kimi_linear.cpp` (`ForwardDevice` device-
resident SEAM + the W7 device-compute reuse-wiring plan). NET-NEW = the KDA device
kernel (host-ref-oracled), the NoPE-MLA branch, the hybrid schedule + het-KV wiring,
the loader name-map.

## Tests to port
`tests/vllm/models/test_kimi_linear_scaffold.cpp` (9/9, registry+config+name-map),
`test_kimi_linear_forward.cpp` (7/7·300: per-op KDA/NoPE-MLA/router gates + the whole
2-layer forward + greedy decode + the W6 `ForwardDevice` device-resident gate),
`test_kimi_kda.cpp` (14/14, the four net-new-vs-GDN numerics). From vLLM `tests/`:
`registry.py` `_HfExamplesInfo` + `test_initialization.py` (construct-only — no GPU);
no upstream text-correctness fixture exists, so the e2e oracle golden (§8) is the
correctness truth.

## Gates
Unit (CPU, landed): the three test files above, all green on a clean CPU build;
`check-runner-routing-consistency`/`check-fusion-consistency`/`check-model-checklist`/
`check-agent-record` rc=0. Pending (GPU): the e2e SACRED greedy golden vs the pinned
oracle on GB10 (STRICT expected for a 48.9B MoE; near-tie fallback ratified), then
the device-compute gate (device==host within bf16 near-tie) and speed (nsys both
sides, vLLM graphed denominator, match/beat every axis).

## Dependencies
W1 (registry/config) -> W2 (loader) -> W2-W6 (CPU reference) -> W6 (device SEAM,
landed). W7 device compute depends on the reused GDN/MLA/MoE device blocks (landed)
+ a free GB10 slot + ~10 GiB dgx disk reclaim. Shared claims: `CLAIM-MLA-DEEPSEEK`
(MLA half), `CLAIM-KDA-KERNEL` (KDA host refs) — coordinate before the W7 device
KDA/MLA kernels so nothing is implemented twice.

## Work breakdown
**DONE:** W0 spike -> W1 registry/config/loader -> W2-W6 CPU reference forward -> W6
device-resident-logits SEAM -> **W7 DBuf-resident device COMPUTE, CPU-gated (§10 —
the whole hybrid over pooled DBufs via the shared `vt::` ops; two documented
host-fallback islands: the KDA per-k-channel recurrence+gate and the NoPE-MLA softmax
core; `test_kimi_linear_forward` 12/12·614 device==W2 ref + greedy-identical; runner
opt-in `VT_KIMI_DEVICE_COMPUTE=1`).** **Residual (keeps the row out of DONE, all
GPU-verify / box-down):** (a) the GPU numerics of the device compute — bf16
activations (vLLM parity), the GDN Triton-AOT cubins, the paged het-KV, the
grouped-MoE slabs — token-exact vs the pinned oracle on a CUDA queue; (b) the two
host-fallback islands -> device ops (a KDA per-channel-decay recurrence + an
exp/softplus KDA-gate op; the paged `mla::ForwardMlaAttentionBlock` over W_UK/W_UV
absorption) (W7-speed); (c) the W0/W7 e2e SACRED golden on GB10 (§8 recipe);
(d) speed. Full W0-W7 table in §5; W7 detail in §10.

## Risks/decisions
- **Decision (correctness-first, box down):** W6 lands the runner SEAM + the full
  device-compute PLAN but NOT the device compute, because CPU cannot gate the GDN
  Triton-AOT cubins or the bf16 numerics against the oracle — authoring it now would
  present as "done" without evidence. Mirrors the DeepSeek-V4 device-forward cadence.
- **Decision:** `ForwardDevice` returns device-resident logits via a pooled `DBuf`
  (`on_device()` on CPU+CUDA), so the CPU gate exercises the exact born-on-the-runner
  contract the GPU will; no runner-routing allowlist entry is needed.
- **Risk (small):** the NoPE-MLA branch must skip RoPE explicitly (do not silently
  apply the decoupled DeepSeek RoPE to a NoPE model) — gated by the `mla_use_nope`
  assert in `ParseKimiLinearParams` + the NoPE unit reference.
- **Risk:** GB10 unified-memory OOM on the 91.5 GiB load — keep `gpu_memory_util`
  ~0.55-0.65 (§3), never 0.85.
