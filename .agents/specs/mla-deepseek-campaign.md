# SPIKE — MLA (Multi-head Latent Attention) + the DeepSeek / Kimi / MiniMax families

**Status:** SPIKE ONLY. No implementation, no kernels, no build. Nothing in this
change claims `READY`/`ACTIVE`/`DONE`.
**Base:** `b4f14ee` (`origin/main`).
**Pinned oracle:** `/home/mudler/_git/vllm` @ `e24d1b24` (v0.25.0 audit target
`702f4814fe54`). The executable oracle venv `~/venvs/vllm-oracle` lives on
**dgx.casa**, not on the dev box — every claim below is grounded in repo source;
the two claims that need a RUNTIME observation are flagged in §9/§12.
**Rows covered:** `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-for-causal-lm`,
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`,
`MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm`.
**Claim:** `CLAIM-MLA-DEEPSEEK`.
**Parent plan:** [breadth-sweep-plan.md](breadth-sweep-plan.md) §B.3 Tier 3 named
"MLA family = new attention, new campaign, needs its own leaf spike". This is
that leaf spike.

---

## 0. Headline findings (read this first)

1. **We have NO MLA at all.** The only occurrences of the string in our tree are
   three *deferral comments* (`src/vllm/platforms/cuda.cpp:54-56`,
   `include/vllm/v1/kv_cache_interface.h:26,44`,
   `include/vllm/v1/attention/backend.h:28`) and one non-MLA test title
   (`tests/vllm/v1/attention/test_attn_backend_registry.cpp:121`). Zero code.

   **Layout note for anyone following older vLLM paths:** at this pin
   `vllm/attention/` no longer exists. `vllm/attention/layer.py` is gone,
   `vllm/attention/backends/abstract.py` is now `vllm/v1/attention/backend.py`,
   `MLACommonBackend`/`MLACommonImpl` are **not** in
   `vllm/v1/attention/backends/mla/common.py` but in
   `vllm/model_executor/layers/attention/mla_attention.py` (2435 lines), and
   `deepseek_v4` is a platform-dispatched *package* (`vllm/models/deepseek_v4/`),
   not a module. Ports written against the old layout will not find their
   targets.

2. **On GB10 (sm_121, `device_capability.major == 12`) vLLM's MLA backend list is
   exactly two entries** — read, not inferred, from
   `vllm/platforms/cuda.py:129-133`:
   ```
   elif device_capability.major == 12:
       return [
           AttentionBackendEnum.TRITON_MLA,
           AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
       ]
   ```
   The second is *sparse* MLA (V3.2/DSA only). **So the dense-MLA decode path we
   must mirror on our only hardware is `TRITON_MLA`** —
   `vllm/v1/attention/backends/mla/triton_mla.py:81` (`TritonMLABackend`,
   `supports_compute_capability` returns `True` unconditionally at `:130-131`),
   whose impl `TritonMLAImpl:134` runs `decode_attention_fwd` from
   `vllm/v1/attention/ops/triton_decode_attention.py`. **FlashMLA, CUTLASS MLA,
   FlashInfer MLA and TokenSpeed MLA are NOT reachable on GB10** — they only
   appear in the `major == 10` (sm_100) and `else` (sm_90 and older) branches.
   This is the single most plan-shaping fact in the spike: it removes an entire
   class of sm90/sm100-only kernel ports from scope.

3. **MLA prefill on GB10 is FlashAttention**, likewise read not inferred:
   `vllm/v1/attention/backends/mla/prefill/selector.py:66-76` — `major == 10`
   gets `[FLASH_ATTN, TRTLLM_RAGGED, FLASHINFER, TOKENSPEED_MLA]`, **everything
   else (including major 12) gets `[FLASH_ATTN]` alone.** We already vendor an
   FA-2 varlen prefill launcher; the delta is head-dim generalization
   (qk 192 / v 128), not a new kernel family.

4. **The cross-cutting cost is the KV cache, and it is a real structural change,
   not a parameter.** MLA stores ONE latent vector plus the rope part per token,
   `num_kv_heads == 1`, `head_size == kv_lora_rank + qk_rope_head_dim == 576`,
   and **no separate V**. Our allocator hardcodes the factor 2 for K+V
   (`src/vllm/v1/worker/gpu/runner.cpp:487-492`) and derives shape from the HF
   config rather than from the KV spec; our `vt::ReshapeAndCache` /
   `vt::PagedAttention` op signatures take separate `k_cache`/`v_cache` tensors
   (`include/vt/ops.h:1167,1201`). See §4 — this is the spike's most important
   finding.

5. **The MLA unlock is bigger than DeepSeek.** `KimiLinearForCausalLM` is a
   *hybrid whose full-attention layers are MLA* — `kimi_linear.py:310` builds
   `KimiMLAAttention` (`:179`, self-documented "Main reference: DeepseekV2 vllm
   Implementation" at `:181`) around the shared `MLAModules` /
   `MultiHeadLatentAttentionWrapper` (`vllm/model_executor/layers/mla.py:14,34`).
   It is **NoPE MLA** — `kimi_linear.py:213` hard-asserts `use_nope is True`, so
   no rotary on the attention layers at all; position information comes from the
   KDA recurrence. Kimi K2/K2.5's text backbone is DeepSeek-V3 by config
   composition (`vllm/transformers_utils/configs/kimi_k25.py:10` imports
   `DeepseekV3Config`; `kimi_k25.py:360-364` re-enters the registry). So one MLA
   implementation unlocks DeepSeek V2/V3/V3.2 (+V4 partially) **and** the Kimi
   line.

6. **MiniMax is DISJOINT from MLA — the matrix inventory is wrong on M3.**
   `MiniMaxM2ForCausalLM` is plain dense GQA + MoE with partial RoPE and
   per-layer sliding window (`minimax_m2.py:139,178,196-199,206-214`), no MLA and
   no linear attention. `MiniMaxM3SparseForCausalLM` is GQA + a learned sparse
   *indexer*, **not** MLA (`vllm/models/minimax_m3/nvidia/model.py:673-690,
   286, 386`; indexer subsystem `vllm/models/minimax_m3/common/indexer.py:63,120,
   519`). The `MODEL-TEXT-minimax-m3-*` and `MODEL-MM-minimax-m3-*` rows list
   `MLA/latent KV` as a dependency; that cell is a source-scan hypothesis and is
   **corrected in this change** (the row contract explicitly says dependency
   cells are hypotheses the target spike must correct). Lightning attention is
   dead in the MiniMax line entirely — `MiniMaxText01ForCausalLM` /
   `MiniMaxM1ForCausalLM` are listed as REMOVED at `registry.py:721-724`.

7. **`DeepseekForCausalLM` is NOT an MLA model.**
   `deepseek_v2.py:1201-1211`: `use_mha = config.model_type == "deepseek" or all
   (dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))`, and `use_mha`
   selects `DeepseekAttention` (`:133`) — plain MHA. The original DeepSeek-MoE
   16B/67B therefore needs **no MLA at all**; it is the cheapest DeepSeek row and
   is essentially "our existing dense attention + a DeepSeek MoE router". Its
   matrix dependency cell (`MLA/latent KV`) is likewise corrected here.

8. **USER-PREMISE CORRECTION — there is no "Kimi K3".** The pinned registry's
   newest Moonshot entries are `KimiLinearForCausalLM` (`registry.py:139`),
   `KimiVLForConditionalGeneration` (`:447`), `KimiK25ForConditionalGeneration`
   (`:448`) and `MoonshotKimiaForCausalLM` (`:449`, which maps to module
   `kimi_audio`, class `KimiAudioForConditionalGeneration` — a registry
   name/target mismatch worth knowing). **K2.5 is the newest Kimi.** Recorded per
   the user's request.

9. **Hardware verdict is stark and decides the plan: exactly one family member
   is e2e-gateable on GB10.** DeepSeek-V2-Lite (~15.7B, ~29.3 GiB bf16) fits
   comfortably. Everything else in the campaign is HW-BLOCKED e2e (§5).

---

## 1. Scope

Design (not build) **MLA attention + the compressed-latent KV cache** as a new,
additive attention family in our engine, plus the DeepSeek-V2/V3 model family
that consumes it, and determine honestly what of the DeepSeek / Kimi / MiniMax
portfolio can actually be gated on GB10.

**In scope of this spike:** the full upstream/dependency chain with `file:line`
on both sides; the GB10 backend-selection answer (read from source, both dense
and prefill selectors); the exact KV-cache/allocator/op/runner impact; weight
absorption and the prefill-vs-decode MLA forms; the DeepSeek MoE router
(`noaux_tc` grouped-topk, shared experts, routed scaling); MTP inventory; the
per-family hardware-fit verdict with real numbers; the shared-vs-per-model
factoring; the upstream test inventory; and a row-sized W0..W9 breakdown.

**Out of scope of this spike (and named as such):** sparse MLA / DSA (V3.2
indexer, `FLASHINFER_MLA_SPARSE_SM120`), DeepSeek-V4 (`vllm.models.deepseek_v4`,
its 584-byte `fp8_ds_mla` page layout and DSpark drafting), MiniMax-M3's sparse
indexer, every multimodal Kimi row (`kimi_vl`, `kimi_k25`, `kimi_audio`),
MTP/Eagle draft *implementation* (inventoried only), fp8 KV cache
(`fp8_ds_mla`), and multi-GPU/EP. Each keeps its `INVENTORIED` row.

---

## 2. Upstream chain

Every anchor below was read at the pin. vLLM is orchestration; where the real
kernel lives outside vLLM it is named.

### 2.1 Model layer — `vllm/model_executor/models/deepseek_v2.py` (1936 lines)

| Anchor | What |
|---|---|
| `:133 DeepseekAttention` | plain MHA — the `model_type == "deepseek"` path (§0.7) |
| `:229 DeepseekV2MLP` | dense SwiGLU MLP (first `first_k_dense_replace` layers) |
| `:276 DeepseekV2MoE` | the MoE block (router + routed experts + shared experts) |
| `:446 DeepseekV2Attention` | the **non-MLA** V2 form — materializes full per-head K/V from the projections (used when `model_config.use_mla` is False) |
| `:613 DeepseekV32IndexerCache`, `:642 Indexer` | V3.2 DSA sparse indexer — OUT OF SCOPE |
| `:905-950 DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear)` | fused `q_a_proj`+`kv_a_proj_with_mqa`. **NOT optional** — when `q_lora_rank is not None` there is no standalone `q_a_proj` module at all; `:1812-1820` sets `packed_modules_mapping["fused_qkv_a_proj"] = ["q_a_proj", "kv_a_proj_with_mqa"]`, so a port must un-fuse the checkpoint names |
| `:952 DeepseekV2MLAAttention` | **the MLA model-side module** |
| `:1172 DeepseekV2DecoderLayer` | attention-class selection (`:1201-1211`) + MoE-vs-dense layer selection |
| `:1347 DeepseekV2Model`, `:1739 DeepseekV2MixtureOfExperts` | model body + MoE introspection mixin |
| `:1780 DeepseekV2ForCausalLM` | base class |
| `:1909 DeepseekForCausalLM`, `:1913 DeepseekV3ForCausalLM`, `:1917 GlmMoeDsaForCausalLM` | all three subclass the V2 base — **flat inheritance, one file** |

Registry routing, quoted from `vllm/model_executor/models/registry.py:90-94`:
```
90:    "DeepseekForCausalLM": ("deepseek_v2", "DeepseekForCausalLM"),
91:    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
92:    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
93:    "DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
94:    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
```
**V3 and V3.2 resolve to the SAME class.** V3.2's delta is the indexer, reached
through config, not a separate model class. V4 is the only one in the newer
`vllm.models.*` namespace.

**MLA projections** (`deepseek_v2.py:453-526`), with the config fields:
`qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `q_lora_rank`,
`kv_lora_rank`; derived `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`
(`:469`).
- `:1003-1034` (ctor) / `:484-494` — **`q_lora_rank is not None` branches the
  query path**: with a rank it builds `fused_qkv_a_proj` (q_a fused with kv_a) ->
  `q_a_layernorm` (RMSNorm over `q_lora_rank`, `:1019`) -> `q_b_proj`
  (`:1020-1026`); **without** (`q_lora_rank is None`) it builds a standalone
  `kv_a_proj_with_mqa` (`:1010-1016`, `ReplicatedLinear`) plus a single direct
  `q_proj` (`:1028-1034`). **DeepSeek-V2-Lite has `q_lora_rank=null`, so our gate
  vehicle exercises the SIMPLER branch** — see §5.1 for the coverage gap this
  creates.
- Attention scale: `:981-996` sets `scaling = qk_head_dim ** -0.5`, then
  `:1067-1075` multiplies it by `mscale` **squared** where
  `mscale = yarn_get_mscale(factor, mscale_all_dim)` (`:428-433`:
  `0.1 * mscale * log(scale) + 1.0`, identity for `scale <= 1`). Getting this
  squared factor wrong is a silent accuracy bug, not a crash.
- `:1053-1065` — rope type is normalized to `deepseek_yarn` /
  `deepseek_llama_scaling`, and rope is constructed over **`qk_rope_head_dim`
  only** with **`is_neox_style=False`**. Kernel side:
  `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:20,26,57-58`;
  dispatch `rotary_embedding/__init__.py:284`. Application to only the rope slice
  is in the wrapper: `vllm/model_executor/layers/mla.py:160-167` (`k_pe` carries
  a single shared head — the "decoupled"/MQA part), with the Llama-4-style
  position-dependent q scaling at `deepseek_v2.py:436-443` applied at `mla.py:172-173`.
- `:511` — `kv_a_proj_with_mqa` produces `kv_lora_rank + qk_rope_head_dim` (=576)
  per token: the latent plus the decoupled rope part, **one head, not per-head**.
- `:516` — `kv_a_layernorm` = RMSNorm over `kv_lora_rank` only (the rope part is
  NOT normed).
- `:518-519` — `kv_b_proj`: `kv_lora_rank -> num_heads * (qk_nope_head_dim +
  v_head_dim)`; this is `[W_UK ; W_UV]` concatenated per head.
- `:526` — `o_proj` from `num_heads * v_head_dim`.

**Decoupled RoPE** (`:580-595`): `q` is split
`[qk_nope_head_dim, qk_rope_head_dim]`; only the `_pe` halves of q and k go
through rotary, then are written back into the tail of the concatenated head
(`:592`, `:594-595`). `k_pe` is the shared single-head rope part sliced from the
latent at `:588`. `:603-608` shows the V-padding convention: `v` is padded from
`v_head_dim` to `qk_head_dim` for kernels that require symmetric head dims, then
sliced back. YaRN/`rope_scaling` supplies the `mscale` softmax-scale correction.

**MoE + the `noaux_tc` router** (`:276-393`):
- `:288 routed_scaling_factor`; `:313-318` — when `topk_method == "noaux_tc"` a
  learned `e_score_correction_bias` parameter is attached to the gate; otherwise
  it is `None`.
- `:370-378` — `use_grouped_topk=True`, `num_expert_group=n_group`,
  `topk_group=topk_group`, `scoring_func` (`sigmoid` for V3, `softmax` for V2),
  `routed_scaling_factor`, `e_score_correction_bias`.
- `:1305` — `hidden_states *= 1.0 / self.routed_scaling_factor` (the MTP-side
  scaling mirror).
- The router math itself:
  `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:80` —
  `scores = sigmoid|softmax(gating_output)` (`:114-115`); bias is added to a
  *copy* used only for SELECTION while the unbiased score supplies the WEIGHT
  (`:120-124`); group score = **sum of the top-2 experts in each group** when a
  bias is present, else the group max (`:125-131`); then top-`topk_group` groups
  are selected and masked (`:135-139`) before the per-token top-k. A fully fused
  CUDA path `ops.grouped_topk` exists for `num_expert_group <= 32` +
  `scoring_func == "sigmoid"` + bias present (`:28-64`, `:94-104`).
  **This whole selection rule is new work for us** (§3).
- Shared experts: gated on `n_shared_experts`; V2-Lite has 2, V3 has 1.

**MTP:** `vllm/model_executor/models/deepseek_mtp.py::DeepSeekMTP`, registered
`registry.py:611` as `DeepSeekMTPModel`. Related draft rows:
`registry.py:608-609` `Eagle3DeepseekV2ForCausalLM` / `Eagle3DeepseekV3ForCausalLM`
(-> `deepseek_eagle3`), `:610` `EagleDeepSeekMTPModel` (-> `deepseek_eagle`),
`:612` `DeepSeekV4MTPModel`, `:592` `DSparkDraftModel`. **Inventoried, not
spiked here** — they keep their `MODEL-SPEC-*` rows at `INVENTORIED`.

### 2.2 Attention layer — `vllm/model_executor/layers/attention/mla_attention.py` (2435 lines)

This file is BOTH the shared layer and the common backend/impl base; it is the
single densest port target in the campaign.

- `:339 MLAAttention(nn.Module, AttentionLayerBase)` — the layer; `:553 forward`.
- `:55-73` — the canonical weight glossary (`W_DQ`, `W_UQ`, `W_QR`, `W_DKV`,
  `W_UK`, `W_KR`, `W_UV`, `W_O`) with shapes. Port comments should cite it.
- `:66-89` — **"Compute Friendly Approach" (`forward_mha`)**: materialize
  `k_nope = kv_c @ W_UK` and `v = kv_c @ W_UV`, then run ordinary MHA with
  **QK head dim `P+R` (=192) and V head dim `V` (=128)**. `:2344 forward_mha`.
- `:1216-1224 get_kv_cache_shape -> (num_blocks, block_size, head_size)` (3D — no
  K/V axis); `:1237-1238 get_supported_head_sizes() -> [320, 576]`;
  `:1240-1242 is_mla() -> True`.
- **Prefill/decode dispatch** `forward_impl:624-874`: `:700-709` computes
  `num_mqa_tokens = attn_metadata.num_decode_tokens` and the rest are MHA;
  `:722-737` runs `forward_mha` on the tail, `:739+` absorbs and runs
  `forward_mqa` on the head. **Decode tokens are packed FIRST in the batch.** The
  split is purely the scheduler's prefill/decode label — `:19-22` explicitly
  notes this is a heuristic they may want to tune, which is worth knowing before
  we treat it as a hard invariant.
- `:94-117` — **"Data-Movement Friendly Approach" (`forward_mqa`)**: absorb
  `W_UK` into the query (`ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)`), run
  **MQA with QK head dim `Lkv+R` (=576) and V head dim `Lkv` (=512)** directly
  against the cached latent, then project the output back with `W_UV`.
  `:2428 forward_mqa`. This is why decode reads the latent with no per-head K/V
  and why the KV cache can be 1-head.
- `:125-175` — **chunked prefill**: the compute-friendly form would OOM on
  `k_nope = (kv_c @ W_UK)` for large `Skv`, so the context is processed in
  bounded chunks with FA returning `lse`, merged across chunks. Our port needs
  the same workspace-bounded loop plus an LSE-merge (we already have an
  attn-states merge concept in the FA-2 split-KV combine).
- **Weight absorption** — `:875 process_weights_after_loading`: `:899` splits the
  loaded `kv_b_proj` into `W_UK, W_UV`; `:960` stores `self.W_UV =
  W_UV.transpose(0, 1)`; `:962` stores `self.W_UK_T = W_UK.permute(1, 2, 0)`.
  The comment at `:877` records that these are kept as plain fp16/bf16 copies
  rather than quantized. Runtime consumers: `:780-789`
  (`torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)` — the query-side
  absorption) and `:1034` (`torch.bmm(x, self.W_UV, ...)` — the output-side
  un-absorption). **Absorption is a load-time weight transform plus two batched
  GEMMs; it is not a fused kernel.** That is good news for a portable port.
  Exact transforms (`:892-900`, `:959-962`): reshape to
  `(kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim)`, split on the last
  axis, then `W_UV = W_UV.transpose(0,1)` -> `(N, L, V)` and
  `W_UK_T = W_UK.permute(1,2,0)` -> `(N, P, L)`. For V3, `W_UK_T` is
  `(128,128,512)` and `W_UV` is `(128,512,128)`.
  **The two compute shapes, concretely** — this is what "changes the compute
  shape" means in practice:
  - **decode / MQA:** q absorbed to `[B, N, 576]`; K = the raw latent `[S, 576]`
    with **1 KV head**; V = the first 512 columns of that same tensor; QK head
    dim **576**, V head dim **512**; output `[B, N, 512]`, then `W_UV`
    up-projects to `[B, N, 128]`. **K/V are never materialized.**
  - **prefill / MHA:** `kv_b_proj` materializes K `[S, 128, 192]` (128 nope +
    64 rope broadcast across heads) and V `[S, 128, 128]`; QK head dim **192**,
    V head dim **128**, **128 KV heads**. `_concat_k_nope_k_pe` (`:2063-2092`)
    pre-allocates and copies rather than `torch.cat`-ing an expanded
    non-contiguous tensor — worth mirroring for performance.
- `:1206 MLACommonBackend`, `:1246 MLACommonPrefillMetadata` (+
  `:1250 ChunkedContextMetadata`), `:1281 MLACommonDecodeMetadata`,
  `:1291 MLACommonMetadata`, `:1338 MLADims`, `:1401 MLACommonMetadataBuilder`,
  `:1988 MLACommonImpl` — the prefill/decode split every concrete MLA backend
  derives from.
- `:1145 QueryLenSupport`, `:1177 _DecodeConcatQuantFP8` — spec-decode query-len
  and fp8 concerns; not needed for the first gate.
- Shared model-facing wrapper: `vllm/model_executor/layers/mla.py:14 MLAModules`,
  `:34 MultiHeadLatentAttentionWrapper(PluggableLayer)` — this is what
  `kimi_linear.py:263` reuses, i.e. the seam that makes Kimi Linear's MLA half
  free once DeepSeek's is done.

### 2.3 Backend selection and the GB10 answer

- `vllm/platforms/cuda.py:83-142 _get_backend_priorities(use_mla, ...)`. MLA
  branch `:93-142`. **`major == 12` -> `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`
  (`:129-133`).** Call site `:374` passes `attn_selector_config.use_mla`. Our
  port implements only the `else` (non-MLA) branch today —
  `src/vllm/platforms/cuda.cpp:54-71` says so in the comment.
- Backends present at the pin (`vllm/v1/attention/backends/mla/`): `triton_mla`,
  `cutlass_mla`, `flashattn_mla`, `flashattn_mla_sparse`, `flashinfer_mla`,
  `flashinfer_mla_sparse`, `flashinfer_mla_sparse_sm120`, `flashmla`,
  `flashmla_sparse`, `tokenspeed_mla`, `aiter_triton_mla`, `rocm_aiter_mla`,
  `rocm_aiter_mla_sparse`, `xpu_mla_sparse`, `sparse_swa`, plus `indexer.py`,
  `compressor_utils.py`, `sparse_utils.py` and the `prefill/` subpackage.
  **Only `triton_mla` is reachable on our hardware for dense MLA.**
- `triton_mla.py:81 TritonMLABackend` — `:92 get_supported_head_sizes`,
  `:96 get_supported_kernel_block_sizes`, `:100 supports_block_size`,
  `:106 get_kv_cache_stride_order`, `:118 supports_batch_invariance`,
  `:130-131 supports_compute_capability -> True`. Impl `:134 TritonMLAImpl`
  (`can_return_lse_for_decode = True`), which **rejects `alibi_slopes`,
  `sliding_window`, `logits_soft_cap`** (`:165-171`) and non-decoder attn types
  (`:173-179`) — so the reachable configuration surface is small.
  `:189 forward_mqa` is the decode entry point.
- **The real decode kernel is not vLLM's.**
  `vllm/v1/attention/ops/triton_decode_attention.py:1-11` records the chain:
  vLLM <- **SGLang** `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
  <- **lightllm** `lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py`
  and `..._stage2.py`. It is a two-stage split-KV flash-decode (per-split partial
  + fixed-order combine) — **structurally the same shape as our existing FA-2
  split-KV decode + combine**, which is the single most important reuse finding
  for the decode kernel. `triton_mla.py:41 _compute_num_kv_splits` and
  `:57 _reserve_attn_logits_workspace` define the split heuristic and the
  workspace we must mirror.
- **MLA prefill selection:**
  `vllm/v1/attention/backends/mla/prefill/selector.py:48-76` —
  `major == 10` -> `[FLASH_ATTN, TRTLLM_RAGGED, FLASHINFER, TOKENSPEED_MLA]`;
  **everything else -> `[FLASH_ATTN]`**. Backends in
  `mla/prefill/{flash_attn,flashinfer,trtllm_ragged,tokenspeed_mla,aiter_flash_attn}.py`
  behind `base.py`/`registry.py`/`selector.py`. `MLADimensions(qk_nope_head_dim,
  qk_rope_head_dim, v_head_dim)` at `selector.py:34-39` is the selection key.
  Two GB10-specific consequences: (a) `FlashAttnPrefillBackend.is_available()` is
  `is_flash_attn_varlen_func_available()` (`prefill/flash_attn.py:48-49`), and if
  it is False the selector **hard-raises** `"No valid MLA prefill backend found"`
  (`selector.py:191-194`) rather than falling back — there is no safety net below
  FA on sm_121; (b) `requires_v_padding` is **True** on GB10
  (`prefill/flash_attn.py:88-99` requires FA3-on-SM90 or FA4), so **V is
  zero-padded from 128 to 192** to match the QK head dim and sliced back after.
  Our W5 must mirror that padding exactly. Note also
  `backend_supports_prefill_query_quantization()` is False here — it needs
  `is_device_capability_family(100)` (`mla_attention.py:1382-1385`).
- **Per-backend capability gates, for the record** (none reachable on major 12,
  and none even PROBED since they are absent from the priority list):
  `FLASH_ATTN_MLA` `major == 9` (`flashattn_mla.py:80-81`); `FLASHMLA`
  `major in [9,10]` (`flashmla.py:82-83`); `CUTLASS_MLA` `major == 10`
  (`cutlass_mla.py:73-74`); `FLASHINFER_MLA` `major == 10`
  (`flashinfer_mla.py:89-90`); `TOKENSPEED_MLA` `major == 10` + hard-coded R1
  dims (`tokenspeed_mla.py:83-84,106-125`); `FLASHMLA_SPARSE` `major in [9,10]`
  (`flashmla_sparse.py:129-130`).
- **Why `TRITON_MLA` wins on GB10, check by check** (`backend.py:307-360`
  `validate_configuration`, with `head_size=576`, `bf16`, `kv_cache_dtype="auto"`,
  `use_mla=True`, `use_sparse=False`): `supports_head_size` passes because
  `TritonMLABackend.get_supported_head_sizes()` returns **`[]`**
  (`triton_mla.py:92-93`) and `backend.py:158-160` treats empty as
  universal-accept — note it *widens* `MLACommonBackend`'s `[320,576]`;
  `supports_dtype` `:82`; `supports_kv_cache_dtype` `:83-89`; `supports_block_size`
  = `block_size % 16 == 0` (`:99-103`); `is_mla()` True; `is_sparse()` False
  matches; **`supports_compute_capability` returns `True` unconditionally**
  (`:130-131`); attn type DECODER ok; `supports_combination` not overridden
  (returns `None`). **Zero invalid reasons -> selected.** The only other candidate,
  `FLASHINFER_MLA_SPARSE_SM120`, is rejected on `is_sparse()` True vs
  `use_sparse=False` (`flashinfer_mla_sparse.py:67`) and additionally requires
  `has_flashinfer_sparse_mla_sm120()` (`:177-200`, bf16-only, blocks `[64,256]`).
- **The KV write op is a dedicated custom op, not our ReshapeAndCache:**
  `vllm/_custom_ops.py:2532 concat_and_cache_mla` (and `:2545
  concat_and_cache_mla_rope_fused`), invoked from
  `vllm/v1/attention/backend.py:995` and `:1075`. It concatenates `kv_c` and
  `k_pe` into ONE cache row. A compile-time fusion pass folds RoPE into it:
  `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:40`.

### 2.4 KV-cache spec — `vllm/v1/kv_cache_interface.py`

- `:363 MLAAttentionSpec(FullAttentionSpec)` with `cache_dtype_str` and the
  V4-only `alignment` / `compress_ratio` / `model_version` fields (`:365-369`).
- `:376-377 storage_block_size = block_size // compress_ratio`.
- `:380-398 real_page_size_bytes` — **the decisive formula**:
  ```
  return self.storage_block_size * self.num_kv_heads * head_dim * get_dtype_size(self.dtype)
  ```
  i.e. `block_size * 1 * 576 * 2` for bf16 V2-Lite. **No factor 2, no separate
  V.** The special cases above it are `fp8_ds_mla` (V3.2 = 656 bytes/token,
  V4 = 584 bytes/token, `:381-388`) and INT4 per-token-head (`:389-390`) — both
  out of scope.
- `:400 merge` asserts every layer in an MLA group is an `MLAAttentionSpec`, and
  `FullAttentionSpec.merge` at `:277-279` asserts the converse. **MLA and
  full-attention layers can never share a KV group** — that is what makes Kimi
  Linear a THREE-group model (MLA + KDA-mamba, and no plain-full-attention group).
- `:327 _apply_alignment_padding` (V4 only), `:337 TQFullAttentionSpec` (unrelated).

### 2.5 Kimi and MiniMax (family adjacency)

- `KimiLinearForCausalLM` — `registry.py:139` -> `kimi_linear.py`. Layer
  selection `:303-325`: `config.is_kda_layer(layer_idx)` -> `KimiGatedDeltaNet
  Attention` else `KimiMLAAttention` (`:179`, MLA geometry at `:189-193`,
  `MLAModules` at `:249`, wrapper at `:263`, **`assert self.use_nope is True` at
  `:213`**, `q_lora_rank: int | None` at `:192` — same optional-rank branch as
  DeepSeek). The KDA/MLA interleave is **data-driven from the checkpoint**, not a
  hardcoded ratio: `vllm/transformers_utils/configs/kimi_linear.py:144-148
  is_kda_layer` reads `linear_attn_config["kda_layers"]`, with both `kda_layers`
  and `full_attn_layers` asserted non-None at `:106-107`. MoE `kimi_linear.py:103
  KimiMoE`, gated by `first_k_dense_replace`/`moe_layer_freq` (`:326-331`); model
  class `:556` declares `HasInnerState, IsHybrid, MixtureOfExperts, SupportsPP`.
- **KDA vs our GDN — honest assessment.** `KimiGatedDeltaNetAttention`
  *subclasses* `GatedDeltaNetAttention`
  (`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py:85`; base
  `mamba/gdn/base.py:22`, `mamba_type -> GDN_ATTN` at `base.py:50-51`), so it
  reuses the **same attention backend + `GDNAttentionMetadata`** we already
  built, and reuses GDN's own chunked primitives —
  `vllm/model_executor/layers/fla/ops/kda.py:19-26` imports
  `chunk_gated_delta_rule_fwd_h`, `chunk_local_cumsum`, `l2norm_fwd`,
  `solve_tril` (the WY/UT triangular solve we already ported), and
  `prepare_chunk_indices`. **But the kernel is a different algorithm**: GDN uses
  a per-head SCALAR decay, KDA uses a **per-channel** (`[H, D]`) decay produced
  by a low-rank `f_a_proj`/`f_b_proj` bottleneck
  (`kimi_gdn_linear_attn.py:142-161`; `kda.py:1603-1617 fused_kda_gate` returns
  `[..., H, D]`), which forces the **gated-linear-attention** output kernel
  (`kda.py:1019 chunk_gla_fwd_kernel_o`, `:1126 chunk_gla_fwd_o_gk`) that plain
  GDN does not have. It also uses **three separate q/k/v short convs**
  (`kimi_gdn_linear_attn.py:171-198`) vs our single fused conv over concatenated
  qkv. New KDA-only ops: `chunk_kda_with_fused_gate`, `fused_kda_gate`,
  `fused_recurrent_kda`, plus `chunk_kda_scaled_dot_kkt_fwd` (`kda.py:717`),
  `recompute_w_u_fwd` (`:960`), `kda_gate_cumsum_fwd_kernel` (`:1182`),
  `FusedRMSNormGated` (`:436`). State shapes via
  `mamba/mamba_utils.py:237-260 kda_state_shape`.
  **Verdict: the STATE MACHINERY (cache layout, metadata, conv-update, chunked
  delta recurrence, WY solve) is genuinely reusable; the DECAY/GATE path and the
  output kernel are NOT — they are a new gated-linear-attention kernel.** Calling
  our GDN kernels "reusable for KDA" without that split would be the kind of
  superficial-similarity claim [[ground-premises-before-dispatching]] warns about.
- `KimiK25ForConditionalGeneration` — `registry.py:448` -> `kimi_k25.py`. It does
  **not** import `deepseek_v2` at all; the link is config composition
  (`configs/kimi_k25.py:10` imports `DeepseekV3Config`, `:83` `text_config`,
  `:101-103` instantiate, `:72` documents "Configuration for the text model
  (DeepseekV3)"), with `kimi_k25.py:360-364 init_vllm_registered_model`
  re-entering the registry to reach `DeepseekV3ForCausalLM`. Multimodal parts
  (`:279-286`, `MoonViT3dPretrainedModel` `:337`, projector `:349`, image + video
  chunk `:287-292`) keep the MM row out of scope.
- `MiniMaxM2ForCausalLM` — `registry.py:153` -> `minimax_m2.py`. `:139
  MiniMaxM2Attention` with `QKVParallelLinear` (`:178`), GQA (`:162-174`),
  standard `get_rope` with **partial rotary** (`:196-199`), per-layer sliding
  window (`:211`), plain `Attention` (`:206-214`, import `:41`). MoE `:72
  MiniMaxM2MoE`, `num_experts=num_local_experts`, `top_k=num_experts_per_tok`
  (`:99-100`), with a DeepSeek-V3-style expert-bias weight loader (`:93`), no
  shared expert. Model class `:432` (`SupportsLoRA, SupportsPP, SupportsEagle3`).
  Draft: `registry.py:597 Eagle3MiniMaxM2ForCausalLM` reuses the **generic Llama
  Eagle3** impl; there is no M2 MTP.
- `MiniMaxM3SparseForCausalLM` — `registry.py:154-157` ->
  `vllm/models/minimax_m3/` (per-vendor `nvidia/`, `amd/`, shared `common/`).
  Per-layer dense-vs-sparse selection `nvidia/model.py:673-690`; `:286
  MiniMaxM3Attention` (GQA, q/k norms `:337-338`, partial RoPE `:340-342`);
  `:386 MiniMaxM3SparseAttention` adds indexer heads (`:430-445`). Indexer
  subsystem `common/indexer.py:63,120,175,187,199,218,261,337,397,466,519`; main
  sparse path `common/sparse_attention.py:61,281,336,400`. MTP
  `registry.py:613 MiniMaxM3MTP`. **GQA + sparse indexer, NOT MLA.**

---

## 3. Our baseline

**MLA-relevant code we have: none.** The four deferral sites are listed in §0.1.

What we DO have that is genuinely reusable, assessed honestly:

| Asset | Anchor | Reusable for MLA? |
|---|---|---|
| Model self-registration seam (`REGISTER_VLLM_MODEL` TU, factory, config parse) | `src/vllm/model_executor/models/qwen3_moe_registry.cpp` | **Yes, fully.** A new model is a new TU; this is the additive win we already proved twice. |
| Shape-agnostic runner (`ENG-RUNNER-MODELSHAPE`) | `src/vllm/v1/worker/gpu/runner.cpp` | **Partly.** It is agnostic to layer *counts* and to "has GDN or not", but NOT to KV *layout* — see §4. MLA is the first model to force a real KV-shape generalization. |
| Dense attention block + step-input glue | `include/vllm/model_executor/models/dense_attn_block.h` (556 lines) | **Structure yes, math no.** The qkv->rope->cache->paged-attn->o_proj skeleton and `BuildStepInputs` are the template; MLA replaces the middle with the two-form (MHA-prefill / MQA-decode) path. |
| bf16 grouped-MoE GEMM | `vt::MoeGroupedGemmBf16` + `MoeGroupedGemmBf16WmmaPipe` + deterministic split-K (`src/vt/cuda/cuda_matmul_nvfp4.cu`) | **Yes — this is the biggest single reuse.** It runs at ~1.2x vLLM's Triton `fused_moe` rate. DeepSeek's experts are exactly this shape. |
| `RunMoeBlock` / `MoeBlockOutput` | `include/vllm/model_executor/models/qwen3_5_moe_block.h` | **Yes for the expert half.** The ROUTER half must be extended (§3.1). |
| Shared-expert path | `qwen3_5.cpp` `SharedExpert` + the W1 no-shared-expert guard | **Yes** — DeepSeek has 1-2 shared experts, so the guard's *other* branch is exercised. |
| Decode CUDA-graph driver (3 siblings already) | `Qwen3MoeDecodeGraph` in `src/vllm/model_executor/models/qwen3_moe.cpp`; sizes in `decode_graph_sizes.h` | **Yes, as a pattern.** A 4th sibling. Note the W7 lesson: capture exposes stack-local host uploads. |
| FA-2 varlen prefill launcher (d128) | `src/vt/cuda/cuda_flash_attn_fa2.cu` | **Yes with a head-dim generalization** — MLA prefill on GB10 is FLASH_ATTN at qk 192 / v 128 (§2.3). We already generalized this launcher once (d128 for Qwen3). |
| FA-2 split-KV decode + fixed-order combine | same TU | **Yes as the structural template** for the two-stage Triton MLA decode (§2.3): same split-partial + deterministic-combine shape. |
| `cuda_arch_tactics` dispatch seam (`b4f14ee`) | `src/vt/cuda/cuda_arch_tactics.{h,cu}` | **Yes** — the registration point for an MLA decode tactic; keeps the new kernel additive. |
| Fusion catalog / `vt::FusedChain` | `include/vt/ops.h` | **Yes** — the MLA RoPE+cache-concat fusion (`mla_rope_kvcache_cat_fusion.py:40`) is a natural recipe once the unfused path is byte-exact. |
| `vt::ReshapeAndCache` / `vt::PagedAttention` | `include/vt/ops.h:1167,1201` | **NO.** Signatures hardcode separate `k_cache`/`v_cache`. New ops needed (§4). |
| `FullAttentionSpec` / KV allocator | `include/vllm/v1/kv_cache_interface.h:157-173`; `src/vllm/v1/worker/gpu/runner.cpp:449-520` | **NO.** Hardcodes K+V. New spec + allocator generalization (§4). |
| MoE router | `vt::MoeRouterTopK` + `MoeRouterTopKArgs{top_k, renormalize}` (`include/vt/ops.h:100,311-318,1108`) | **NO — too narrow.** Softmax + top-k + renorm only. §3.1. |

### 3.1 The router gap, precisely

`MoeRouterTopKArgs` carries exactly two fields. DeepSeek needs, additionally:
`scoring_func` (sigmoid vs softmax), `e_score_correction_bias` (a device tensor,
used for SELECTION only while the UNBIASED score becomes the weight),
`num_expert_group` / `topk_group` (the two-level group mask, with group score =
top-2 sum when a bias is present, else group max), and `routed_scaling_factor`.
This is an additive extension of one args struct plus one kernel — small, but it
is genuinely new numerics and needs its own unit test against the upstream
formula (`grouped_topk_router.py:80-145`).

---

## 4. THE CROSS-CUTTING COST — the compressed-latent KV cache

This is the spike's most important finding and the reason MLA is a *campaign*
rather than a model row.

### 4.1 What MLA actually stores

Per token, per layer, MLA caches **one** row of `kv_lora_rank + qk_rope_head_dim`
elements (512 + 64 = **576** for every DeepSeek variant and for Kimi Linear's MLA
layers), with `num_kv_heads == 1` and **no V tensor at all** — V is reconstructed
on the fly from the same latent via `W_UV`. Upstream page size
(`kv_cache_interface.py:380-398`):
`storage_block_size * num_kv_heads(1) * head_size(576) * dtype_size`.

Three independent places encode `head_size = kv_lora_rank + qk_rope_head_dim`
(`mla_attention.py:387`,
`vllm/transformers_utils/model_arch_config_convertor.py:48-58`,
`get_mla_dims` `mla_attention.py:1364-1370`) and three encode `num_kv_heads = 1`
(`mla_attention.py:390`, `:1004-1009`, and `vllm/config/model.py:1270-1274`
"When using MLA during decode it becomes MQA"). The `use_mla` flag itself is
`model.py:1638-1639` = `is_deepseek_mla and not VLLM_MLA_DISABLE`, with the
model-type allowlist at `model_arch_config_convertor.py:255-285` — **which
includes `kimi_k2` and `kimi_linear` alongside the DeepSeek types**, independent
confirmation of §0.5.

**The shape is 3-dimensional, and that is the tell.**
`MLACommonBackend.get_kv_cache_shape` returns
`(num_blocks, block_size, head_size)` (`mla_attention.py:1216-1224`) — no K/V
axis at all, where every non-MLA backend returns a leading `2`. The
`num_kv_heads` argument is accepted and ignored (`:1219`: "assumed to be 1 for
MLA"). And decode reads V as a *slice of K*: `triton_mla.py:236`
`kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]`, then
`decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, ...)` at `:242-259`
**passes the same buffer twice**, with `layer._k_scale` used as both `k_scale`
and `v_scale` (`:256-257`; cf. `calc_kv_scales` `:975-996`, where one abs-max
feeds both). Our op signatures cannot express that.

Concretely for DeepSeek-V2-Lite (27 layers, bf16, block 16):
`16 * 1 * 576 * 2 = 18,432 B` per block per layer — versus a hypothetical
per-head MHA cache of `16 * 16heads * 192 * 2 * 2(K+V) = 393,216 B`. **MLA is
~21x smaller here.** That compression is the entire point of the architecture and
it is *why* a 671B model is servable at all; it is also why we cannot fake it
with our existing cache and expect representative memory or speed numbers.

### 4.2 Exactly what breaks in our tree

1. **`include/vllm/v1/kv_cache_interface.h`.** No `MLAAttentionSpec`. Line 44
   already names it as deferred. `FullAttentionSpec::real_page_size_bytes`
   (`:157-173`) is `block_size * num_kv_heads * (head_size + head_size_v) *
   dtype_size` — the `head_size_v` split exists (the header comment at `:24-26`
   says it was put there for exactly this) but the K+V duality is baked in.
   **New:** `MLAAttentionSpec : FullAttentionSpec` overriding
   `real_page_size_bytes` to the single-tensor formula, plus a `kind()`
   (`KVCacheSpecKind::kMLA`) so the runner can branch, plus the merge assertion
   that MLA and full-attention layers never share a group
   (`kv_cache_interface.py:277-279, 400-403`).

2. **`src/vllm/v1/worker/gpu/runner.cpp:487-492` — the hardcoded factor 2.**
   ```
   full_attn_buf_.push_back(std::make_unique<CacheBuffer>(
       dev, queue_,
       static_cast<size_t>(num_blocks_ * 2 * fa_block_size * Hkv * Dh) *
           static_cast<size_t>(vt::SizeOf(kv_dtype)), ...));
   ```
   Note also `:399-400`: `Hkv`/`Dh` come from `config_.num_key_value_heads` /
   `config_.head_dim` — **the runner reconstructs the shape from the HF config
   instead of asking the spec**. That is the actual defect MLA exposes. The
   correct generalization (and the one that mirrors upstream) is to **size every
   cache buffer from `spec->page_size_bytes()` and let the spec own the layout**,
   which is behaviour-preserving for the existing models by construction (their
   spec computes the same number) and is the additive seam every future
   asymmetric-V or quantized-KV layer needs. This is a `ENG-*`-class change and
   should land as its OWN behaviour-preserving W-step with the 27B/35B/Coder
   gates unchanged, before any MLA math exists.

3. **`PagedKvCache` (the view struct, `runner.cpp:495-508`)** carries
   `num_kv_heads` + `head_size` and is consumed as a K/V pair. MLA needs a
   sibling view (`MlaKvCache { data, dtype, num_blocks, block_size, kv_lora_rank,
   qk_rope_head_dim }`) or a discriminated variant.

4. **`vt::ReshapeAndCache` (`include/vt/ops.h:1167`, `OpId::kReshapeAndCache`
   `:103`)** takes `(k, v, k_cache, v_cache, ...)`. MLA's write is upstream's
   `concat_and_cache_mla` (`_custom_ops.py:2532`, called from
   `v1/attention/backend.py:995,1075`) — concatenate `kv_c` (post-`kv_a_layernorm`)
   with `k_pe` into ONE row at the slot. **New op `vt::ConcatAndCacheMla`.**
   Kernel-wise it is trivial (a strided copy of 576 elements per token); the work
   is the op/registry/CPU-reference/CUDA/plumbing quartet, not the math.

5. **`vt::PagedAttention` (`:1201`, `OpId::kPagedAttention` `:104`)** takes
   `k_cache` and `v_cache` separately. MLA decode reads ONE cache as both K and V
   (MQA, QK dim 576, V dim 512). **New op `vt::MlaDecodeAttention`** mirroring
   the two-stage `decode_attention_fwd` (split partials + deterministic combine).
   Our FA-2 split-KV decode is the structural template; determinism must follow
   our fixed-ascending-order reduce convention, never `atomicAdd`
   (the W6 Qwen3-Coder precedent).

6. **Block manager / allocator / prefix caching — CONFIRMED cheap, not assumed.**
   Upstream registers MLA against the *ordinary* full-attention manager:
   `vllm/v1/core/single_type_kv_cache_manager.py:1539` maps
   `MLAAttentionSpec -> FullAttentionManager` with
   `uniform_type_base_spec=FullAttentionSpec`. **Block table, prefix caching and
   eviction are therefore unchanged by MLA** — MLA-ness is a page-SIZE and
   tensor-SHAPE concern only. Equally telling: `vllm/v1/worker/gpu_model_runner.py`
   contains **no `use_mla` branch at all** — the only MLA mentions are `:58`
   (import), `:977` (`isinstance(module, (Attention, MLAAttention))`), `:1085`
   and `:7133` (comments). Everything is dispatched through the spec/backend
   abstraction. **This is the strongest argument for the W1 design**: mirror that
   abstraction (size from the spec, dispatch through the backend) and the runner
   needs almost nothing MLA-specific. Our
   `src/vllm/v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,
   single_type_kv_cache_manager,kv_cache_utils}.cpp` deal in block ids and hashes
   and should likewise need no change — to be confirmed in W1, but now with a
   cited upstream precedent rather than a guess.
   Grouping-side exceptions to be aware of: `kv_cache_utils.py:1521` (the
   `isinstance(spec, MLAAttentionSpec)` grouping branch) and `:1582-1590`
   (`_get_kv_cache_groups_uniform_groups` asserts the FIRST group is all-MLA).

7. **Chunked prefill workspace.** `mla_attention.py:125-175` bounds memory by
   chunking the context for the compute-friendly prefill and merging with LSE
   (`_compute_prefill_context:2094-2200`, merge at `forward_mha:2412-2419`).
   Sizing is `MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size`
   (`:1423-1452`):
   `min(max(8*max_model_len, 4*max_num_seqs*block_size), 64*1024)`, then
   `max(..., max_num_seqs*block_size)`. The inline arithmetic at `:1435-1441`
   states the stakes: at 64k tokens the latent workspace is
   `2*576*65536 = 144 MB` but the **up-projected** context would be
   `2*(192*128)*65536 = 3 GB`. **That 3 GB blow-up is precisely why chunking
   exists** — a port that skips it will OOM on long contexts rather than merely
   run slow. We have no workspace-bounded attention loop; we do have an
   attention-states combine. This is genuinely new engine machinery and is the
   part most likely to be under-estimated.

9. **Feature restrictions to mirror (all cited, all narrowing — good news).**
   KV sharing across layers raises `NotImplementedError` for MLA
   (`mla_attention.py:2038-2039`). **Cascade attention is never used** — no MLA
   metadata builder overrides `use_cascade_attention` (`backend.py:708`), so it
   defaults False. Prefix caching is force-disabled for `TRITON_MLA`/`FLASHINFER`
   under `VLLM_BATCH_INVARIANT` (`mla_attention.py:437-451`). `TritonMLAImpl`
   additionally rejects `alibi_slopes`, `sliding_window`, `logits_soft_cap`
   (`triton_mla.py:165-171`) and non-decoder attention (`:173-179`). The
   reachable configuration surface on GB10 is therefore genuinely small.

8. **Cache dtype interaction.** Our KV cache defaults to bf16 with a
   `VT_KV_CACHE_F32` A/B (`runner.cpp:401-406`). MLA at bf16 is fine; the
   `fp8_ds_mla` layouts (656 B / 584 B per token) are explicitly out of scope and
   must throw loudly rather than silently mis-size.

### 4.3 Consequence for sequencing

Items 1-2 are **behaviour-preserving engine work with no MLA math in them** and
must land first, gated on the existing models being byte-identical. Items 4-5 are
two new `vt::` ops. Item 7 is the largest unknown. Nothing about MLA is a
"config parse plus a new model TU" — that framing, which worked for Qwen3-Coder,
does not apply here, and the W-plan in §9 reflects that.

---

## 5. Hardware reality on GB10 — per-family verdict

**Box:** dgx.casa (note: the hostname `dgx` does not resolve; use `dgx.casa`),
self-identifying as `promaxgb10-4ad8`, GB10 / DGX Spark, ~119 GiB unified memory
(~110 GiB usable after KV cache + activations + page cache).

**Disk, measured this session:** single root filesystem
`/dev/nvme0n1p2 3.6T, 3.2T used, 238G available, 94% full`. There is **no
separate model volume**. `~/models` holds ~120 GiB (three diffusiongemma GGUFs
account for 113 GiB of it); the HF hub cache holds the Qwen/Parakeet/OPT
checkpoints. **None of the DeepSeek / Kimi / MiniMax checkpoints are present.**
Nothing was downloaded in this spike.

| Family / checkpoint | Params (total / active) | bf16 size | Fits ~110 GiB? | Fits 238 GiB free disk? | Verdict |
|---|---|---|---|---|---|
| **DeepSeek-V2-Lite** (`deepseek-ai/DeepSeek-V2-Lite[-Chat]`) | 15.7B / 2.4B | **~29.3 GiB** | **YES, comfortably** | yes, trivially | **THE MLA GATE VEHICLE** |
| DeepSeek-V3 / V3.2-Exp | 671B / 37B | ~1250 GiB (native fp8 ~640 GiB) | no | **no** | **HW-BLOCKED e2e** |
| Kimi-K2 / K2.5 | ~1.03T / ~32B | ~2000 GiB (fp8 ~960 GiB) | no | **no** | **HW-BLOCKED e2e** |
| Kimi-Linear-48B-A3B-Instruct | 48B / ~3B | ~89.4 GiB | marginal (~20 GiB left) | yes, but pushes root past 97% | **HW-MARGINAL** |
| MiniMax-M2 | ~230B / ~10B | ~428 GiB | no (~4x over) | **no** | **HW-BLOCKED e2e** |
| MiniMax-M3-Sparse | not confidently known; official release is MXFP8 | >>110 GiB | no | no | **HW-BLOCKED e2e** |

**Provenance caveat, stated plainly.** vLLM references these checkpoints only by
HF model id and fetches configs at runtime, so almost none of these constants are
in-repo. The two genuine in-repo corroborations found were
`tests/v1/attention/test_mla_prefill_quant_output.py:158` ("v = v_head_dim(128);
DeepSeek-V2-Lite has 16 query heads") and
`tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py:154` (DeepSeek-V3 routing
config). **Every other figure above is from model knowledge and must be verified
against the real `config.json` at W0** before any of it becomes load-bearing.
DeepSeek-V2-Lite ids appear widely in upstream tests (`tests/test_config.py:135,
1259,1284`; `tests/distributed/test_context_parallel.py:34`;
`tests/v1/distributed/test_dbo.py:29`;
`examples/ray_serving/elastic_ep/serve_deepseek_v2.sh:8`), which is independent
evidence that it is upstream's own small-MLA smoke vehicle — the same role we
want it for.

### 5.1 DeepSeek-V2-Lite vs V3 — and what the gate does NOT cover

**Config comparison** (V2-Lite to be confirmed at W0; V3 column corroborated by
upstream test usage):

| Field | V2-Lite | V3 / R1 / V3.2 |
|---|---|---|
| `hidden_size` | 2048 | 7168 |
| `num_attention_heads` | 16 | 128 |
| **`q_lora_rank`** | **`null`** -> direct `q_proj` | 1536 -> `fused_qkv_a_proj` + `q_a_layernorm` + `q_b_proj` |
| `kv_lora_rank` / `qk_nope` / `qk_rope` / `v_head_dim` | 512 / 128 / 64 / 128 | identical |
| **MLA cache `head_size`** | **576** | **576** |
| `n_routed_experts` / `n_shared_experts` / top-k | 64 / 2 / 6 | 256 / 1 / 8 |
| **`n_group` / `topk_group`** | **1 / 1** | 8 / 4 |
| **`scoring_func` / `topk_method`** | **softmax / greedy** | **sigmoid / `noaux_tc`** |
| `routed_scaling_factor` | 1.0 | 2.5 |
| `first_k_dense_replace` | 1 | 3 |

**The MLA geometry is IDENTICAL between V2-Lite and V3** — same 576-wide latent,
same nope/rope/v dims. That is what makes V2-Lite a legitimate MLA gate vehicle
rather than a toy.

**But it leaves two real coverage gaps, and they must be stated in the row rather
than papered over:**

1. **`q_lora_rank=null`** -> the gate never exercises `fused_qkv_a_proj`,
   `q_a_layernorm` or `q_b_proj`, nor the `packed_modules_mapping` un-fusing at
   `deepseek_v2.py:1812-1820`. All of that must be BUILT for V3 and is
   **unit-gated only**.
2. **`n_group=1, topk_group=1, softmax, greedy`** -> **V2-Lite has no
   `e_score_correction_bias` at all** (`deepseek_v2.py:315-320` only creates it
   when `topk_method == "noaux_tc"`), so the gate exercises **none** of the
   `noaux_tc` machinery: not sigmoid scoring, not the biased-select /
   unbiased-weight asymmetry, not the two-level group mask. The single subtlest
   piece of new router numerics is therefore **unit-gated only**, against the
   upstream formula at `grouped_topk_router.py:112-161` (scoring `:112-117`,
   top-2 group score `:120-127`, non-bias group max `:129-131`, mask/`-inf`
   `:135-145`, **select-with-biased / weight-with-unbiased `:147-150`**,
   renormalize `:156-157`, routed scaling `:159-160`).

Both gaps are acceptable — they are unit-testable and dimension-free — but a row
that claims "MLA + DeepSeek MoE gated" without naming them would be overclaiming.

**Free bring-up oracle worth knowing:** `VLLM_MLA_DISABLE=1` is not a perf
switch. It flips `use_mla` (`config/model.py:1638-1639`), changes the reported
`head_size` from **576 to 192** (`model_arch_config_convertor.py:48-58`), and
routes the model to a completely different attention class,
`DeepseekV2Attention` (`deepseek_v2.py:446-610`, selected at `:1207-1213`), which
materializes full MHA from the same weights. **That gives us a same-checkpoint,
same-oracle A/B between the latent and materialized forms** — an unusually good
debugging asset for W6, and it should be used before chasing any numerics
discrepancy into the kernels.

### 5.2 What is still gateable where e2e is blocked — say it plainly

For V3/V3.2, Kimi-K2.5, MiniMax-M2/M3 **we cannot run an end-to-end token-exact
or speed gate on GB10, now or after any amount of software work.** We will not
propose one. What CAN honestly be gated on the blocked rows:

- **Config/registry resolution** — parse the real `config.json` (a few KB
  download, not weights) and assert the right factory, KV spec and MLA geometry.
- **Weight-map / loader gate on a SLICE** — vLLM's own `safetensors` index is a
  JSON manifest; we can assert every expected tensor name/shape maps, and load a
  single-layer slice, without materializing the model.
- **Unit parity on the new numerics** — the grouped-topk router with V3's
  `sigmoid` + `e_score_correction_bias` + `n_group/topk_group`, and the MLA
  absorption/decode/prefill ops at V3's dimensions (`q_lora_rank=1536`, 128
  heads), against dumped upstream references. Dimensions are free; weights are
  not.
- Anything beyond that is **HW-BLOCKED** and the matrix row must say so rather
  than sit in a state that implies a gate we cannot run.

---

## 6. Port map

New files (all additive; the model TU pattern is already proven):

| Path | Contents |
|---|---|
| `include/vllm/v1/kv_cache_interface.h` (EDIT) | `MLAAttentionSpec` + `KVCacheSpecKind::kMLA` + the merge assertions |
| `include/vllm/model_executor/models/mla_attention.h` | the MLA block: projections, decoupled RoPE, absorption state, the MHA-prefill / MQA-decode forms — the analog of `dense_attn_block.h`, ported from `mla_attention.py` |
| `src/vllm/model_executor/models/mla_attention.cpp` | its implementation |
| `include/vllm/model_executor/models/deepseek_v2.h` | `DeepseekV2Weights`, model decls |
| `src/vllm/model_executor/models/deepseek_v2_registry.cpp` | `REGISTER_VLLM_MODEL(deepseek_v2, "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM", "DeepseekForCausalLM")` + `ParseDeepseekV2Config` + `MakeDeepseekV2KVCache` |
| `src/vllm/model_executor/models/deepseek_v2_weights.cpp` | loader: the optional `q_lora_rank` branch, `kv_a_proj_with_mqa`, `kv_b_proj` **split into `W_UK`/`W_UV` at load time** (`mla_attention.py:899,960,962`), the two norms, per-expert MoE + shared experts, `e_score_correction_bias` |
| `src/vllm/model_executor/models/deepseek_v2.cpp` | forward + a 4th `DecodeGraph` sibling |
| `src/vt/cuda/cuda_mla_attn.cu` (+ CPU reference in `src/vt/cpu/`) | `ConcatAndCacheMla`, `MlaDecodeAttention` (two-stage split + deterministic combine), registered through `cuda_arch_tactics` |
| `include/vt/ops.h`, `src/vt/ops.cpp` (EDIT) | new `OpId`s + args + validation; `MoeRouterTopKArgs` extension (§3.1) |
| `src/vllm/platforms/cuda.cpp` (EDIT) | the MLA branch of `_get_backend_priorities` (`cuda.py:93-142`), completing the port the comment at `:54-56` defers |
| `src/vllm/v1/worker/gpu/runner.cpp` (EDIT) | spec-driven cache sizing (§4.2 item 2) + the MLA cache view + the MLA group id |
| `tests/...` | per §7 |

Shared-file touches are deliberately few and each is a designed seam
(in-TU REGISTER, CMake TU glue, the ops header, the platform priority list, the
runner's spec-driven sizing). The runner edit is the only one that is not a pure
addition, and it is behaviour-preserving by construction.

---

## 7. Tests to port

Per [test-porting.md](../test-porting.md), each W-step carries its upstream
module. Inventory from the pin (`/home/mudler/_git/vllm/tests`):

| Upstream | Asserts | Our tier / target | W |
|---|---|---|---|
| `tests/v1/attention/test_mla_backends.py` | MLA backend correctness vs a reference across batch/seq shapes | parity `tests/vt/test_ops_mla_attn.cpp` | W4/W5 |
| `tests/v1/attention/test_mla_prefill_selector.py` | MLA **prefill** backend selection per capability | doctest `tests/vllm/v1/attention/test_mla_backend_registry.cpp` | W2 |
| `tests/v1/attention/test_mla_prefill_registry.py` | prefill backend registry/enum integrity | same | W2 |
| `tests/v1/attention/test_attention_backends_selection.py` (MLA cases) | `_get_backend_priorities` MLA branch ordering | extend `tests/vllm/v1/attention/test_attn_backend_registry.cpp` (today non-MLA only, `:121`) | W2 |
| `tests/v1/attention/test_mla_prefill_quant_output.py` | prefill output vs reference incl. the V2-Lite dims (`:158`) | parity | W5 |
| `tests/kernels/attention/test_mla_decode_cpu.py:47` | MLA decode against a CPU reference — **device-independent, so it is the numerical ORACLE to port FIRST**, before any CUDA work | doctest + parity | W4 |
| `tests/kernels/moe/test_grouped_topk.py:46` and `tests/kernels/moe/test_routing.py:441` | the dedicated grouped-topk correctness cases | doctest `tests/vt/test_ops_moe_router_grouped.cpp` | W3 |
| `tests/evals/gsm8k/configs/DeepSeek-V2-Lite-Instruct-FP8.yaml` | upstream's own V2-Lite accuracy target | secondary e2e sanity (our SACRED gate remains token-exact) | W8 |
| `tests/kernels/test_concat_mla_q.py` | the q concat helper | doctest | W4 |
| `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py` | fused RoPE + MLA cache write | doctest (unfused first, fusion later) | W3/W8 |
| `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` | the RoPE+concat-cache fusion pass | fusion-catalog recipe test | W8 |
| `tests/compile/passes/test_fuse_mla_dual_rms_norm.py` | dual-RMSNorm fusion (`q_a_layernorm`+`kv_a_layernorm`) | fusion-catalog recipe test | W8 |
| `tests/compile/passes/test_mla_attn_quant_fusion.py` | MLA attn+quant fusion | SKIPPED (quant out of scope), tracked | — |
| grouped-topk router tests under `tests/kernels/moe/` | sigmoid + bias + group masking selection | doctest `tests/vt/test_ops_moe_router_grouped.cpp` | W3 |
| `tests/lora/test_deepseekv2_tp.py` | DeepSeek-V2 layer wiring under TP | SKIPPED (no multi-GPU), tracked | — |
| `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`, `test_sparse_mla_backends.py`, `test_flashinfer_sparse_mla_sm120_api.py`, `test_indexer_deepseek_v4_slot_mapping.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | V3.2/V4 sparse + DSA | **OUT OF SCOPE** — checked in SKIPPED with the reason "sparse MLA / V4 not spiked" | — |
| `tests/kernels/attention/test_flashmla.py`, `test_cutlass_mla_decode.py`, `tests/rocm/aiter/*`, `test_rocm_aiter_mla_decode_metadata.py` | sm90/sm100/ROCm-only MLA kernels | **NOT PORTED** — unreachable on GB10 per §2.3; recorded, not silently dropped | — |
| tool/reasoning parsers (`tests/tool_parsers/test_deepseekv3_tool_parser.py` and siblings, `tests/reasoning/test_deepseekr1_reasoning_parser.py`) | DeepSeek chat/tool/reasoning formats | separate serving row, inventoried not spiked | — |

New tests with no single upstream twin (our own gate protocol):
`tests/vllm/models/test_deepseek_v2_load.cpp` (every tensor mapped, none left
over), `test_deepseek_v2_forward.cpp` (real-checkpoint prefill argmax),
`test_deepseekv2lite_paged_engine.cpp` (the SACRED gate), plus
`scripts/deepseekv2-oracle-capture.py` and `scripts/deepseekv2-neartie-gap.py`
mirroring the Qwen3-Coder pair.

---

## 8. Gates

**Correctness (SACRED).** Per [[near-tie-distributional-gate]]: first establish
by K=5 per-prompt capture whether vLLM's own greedy is deterministic on
DeepSeek-V2-Lite; if yes (expected — it is a 16B model but MoE routing is
deterministic), the STRICT-where-well-posed near-tie-robust gate applies —
our token within 0.5 nats of vLLM's teacher-forced argmax, strict where equal,
0 forward-divergent. **The vehicle is DeepSeek-V2-Lite bf16.** V3/V3.2/K2.5/M2
get **no e2e gate — HW-BLOCKED (§5.2)**, only the config/loader-slice/unit gates.

**Per-step gates.** Every W-step: dgx CUDA `-Werror` 0-warn on a clean build;
`compute-sanitizer memcheck` 0 errors; and the **regression set UNCHANGED** —
Qwen3-dense 16/16, Qwen3-Coder 6/6, 27B 235/235, 35B 315/315. The KV-allocator
generalization (W1) and the platform-priority completion (W2) are
**behaviour-preserving** and must prove it by leaving all four byte-identical.

**Op-level gates.** New `vt::` ops get CPU-reference-vs-CUDA parity tests before
any model uses them (the `test_ops_moe_grouped_bf16` precedent), including a
run-to-run bit-reproducibility assertion on any split/reduce kernel.

**Speed.** The bar is unchanged and non-negotiable: match or beat **graphed**
vLLM 0.25.0 (`enforce_eager=False`, `FULL_AND_PIECEWISE`) on **every axis**
(median TTFT, TPOT, ITL, output throughput, peak memory) at c1/c2/c4/c8, fresh
server per concurrency with a verified 0.0% prefix-cache hit rate, idle box under
one `flock /tmp/gpu`, 2 reps with the cold leg discarded. Per
[[parity-enablers-ship-as-defaults]], any lever parity depends on ships
DEFAULT-ON (with a `VT_*=0` rollback) **before** the binding run.

**Nothing may claim `DONE`** until the row links exact code + test anchors, ledger
evidence and the closing commit.

---

## 9. Dependencies

- **Checkpoint:** DeepSeek-V2-Lite must be downloaded to dgx (~29.3 GiB against
  238 GiB free — comfortable, but the root filesystem is at 94% and should be
  watched per [[grid-per-sha-trees-fill-disk]]). Nothing else in the campaign
  should be downloaded: Kimi-Linear at ~89 GiB would push root past 97%, and
  MiniMax-M2 at ~428 GiB does not fit on disk at all.
- **Oracle:** vLLM 0.25.0 in `~/venvs/vllm-oracle` must resolve
  `DeepseekV2ForCausalLM` on GB10 and actually select `TRITON_MLA` — **verify at
  W0 by running it, do not infer** (this spike read the selection logic; a
  runtime confirmation is still required per [[profile-vllm-actual-kernels-port-1to1]]).
- **Engine:** the spec-driven KV allocator (§4.2 item 2) blocks everything else.
  It is arguably an `ENG-*` row in its own right and should be treated as one if
  it grows.
- **No multi-GPU, no EP, no TP** — all single-GPU.
- **GPU serialization:** `flock /tmp/gpu` for every dgx run.

---

## 10. Work breakdown

Ranked so the shared unlock lands first on the only vehicle that fits our
hardware. **Nothing below is implemented; this is the plan.**

| W | Item | Depends on | Gate |
|---|---|---|---|
| **W0** | **Ground the facts.** Download DeepSeek-V2-Lite; confirm its real `config.json` against §5 (especially `q_lora_rank is None`); run the vLLM oracle on it and CONFIRM from its logs that `TRITON_MLA` + FA-2 MLA prefill are selected on sm_121; capture a K=5 greedy determinism probe to fix the gate form. | checkpoint | oracle runs; selection confirmed by observation |
| **W1** | **Spec-driven KV allocation (behaviour-preserving, no MLA).** Size every cache buffer from `spec->page_size_bytes()`; stop reconstructing shape from `config_`; add `KVCacheSpecKind::kMLA` + `MLAAttentionSpec` with the single-tensor formula + merge assertions. | — | 27B 235/235, 35B 315/315, Coder 6/6, Qwen3 16/16 **byte-identical**; new spec unit tests |
| **W2** | **Platform MLA backend priorities + registry.** Port the `use_mla` branch of `_get_backend_priorities` (`cuda.py:93-142`) and the MLA prefill selector; register an `MLA` attention backend name resolving to TRITON_MLA on major 12. | W1 | ports of `test_attention_backends_selection.py` (MLA cases), `test_mla_prefill_selector.py`, `test_mla_prefill_registry.py`; existing non-MLA selection unchanged |
| **W3** | **New `vt::` ops — cache write + router.** `vt::ConcatAndCacheMla` (CPU ref + CUDA) mirroring `concat_and_cache_mla`; extend `MoeRouterTopKArgs` with `scoring_func` / `e_score_correction_bias` / `num_expert_group` / `topk_group` / `routed_scaling_factor` and implement the grouped-topk selection rule. | W1 | op parity tests vs CPU reference; grouped-topk unit test vs the upstream formula; existing router byte-identical |
| **W4** | **MLA decode attention op.** `vt::MlaDecodeAttention` — the two-stage split-KV MQA decode over the latent (QK 576 / V 512), structurally mirroring `decode_attention_fwd` (vLLM <- SGLang <- lightllm) and reusing our FA-2 split+deterministic-combine pattern; registered via `cuda_arch_tactics`. | W3 | port of `test_mla_decode_cpu.py`; CPU-ref parity; run-to-run bit-reproducibility |
| **W5** | **MLA prefill.** Generalize the vendored FA-2 varlen launcher to qk 192 / v 128 (the GB10 MLA prefill backend is FLASH_ATTN, §2.3) and implement the compute-friendly materialized form plus the workspace-bounded chunked-context loop with LSE merge. | W4 | ports of `test_mla_backends.py`, `test_mla_prefill_quant_output.py` |
| **W6** | **MLA attention block + weight absorption.** `mla_attention.{h,cpp}`: projections (both `q_lora_rank` branches), the two RMSNorms, decoupled RoPE, the load-time `kv_b_proj -> W_UK/W_UV` split, and the prefill-MHA / decode-MQA dispatch. | W5 | forward doctest; absorption round-trip vs the unabsorbed form |
| **W7** | **DeepSeek-V2 model: registry + loader + forward.** New TU, config parse, KV spec (MLA-only group), per-expert bf16 loader + shared experts + `e_score_correction_bias`, forward composing the MLA block + `RunMoeBlock` + the first `first_k_dense_replace` dense layers. Registers all four aliases; `DeepseekForCausalLM` takes the **MHA** branch (§0.7). | W6, W3 | load gate (all tensors mapped, none left over); real-checkpoint prefill argmax |
| **W8** | **SACRED correctness gate — DeepSeek-V2-Lite.** Paged-engine greedy vs the vLLM 0.25.0 oracle under the W0-determined gate form; goldens + capture/near-tie scripts. | W7 | the gate; regression set UNCHANGED; memcheck 0 |
| **W9** | **Speed close.** Decode CUDA-graph sibling; the MLA fusion recipes (RoPE+concat-cache, dual-RMSNorm) as byte-exact catalog entries; then the binding every-axis grid vs graphed vLLM at c1/c2/c4/c8. | W8 | every-axis parity or an honest attributed miss |
| **W10** | **Blocked-row honesty pass.** Config/registry resolution + weight-map + unit parity at V3 dimensions (§5.2) for V3/V3.2; record `HW-BLOCKED` where e2e cannot run. Kimi-Linear and MiniMax-M2 assessments per §11 stay separate rows. | W8 | config/loader-slice/unit gates only |

**Ranked rationale.** W1-W2 are pure engine seams with zero MLA math and must not
be entangled with model work. W3-W5 are the three genuinely new kernels/ops. W6-W8
are the model itself and the only e2e gate our hardware supports. W9 is the speed
bar. W10 is the honest disposition of everything we cannot run.

---

## 11. Shared vs per-model factoring

**SHARED (built once, in W1-W6) — the MLA unlock:**
the `MLAAttentionSpec` + spec-driven allocator, the MLA branch of the backend
priorities, `ConcatAndCacheMla`, `MlaDecodeAttention`, the FA-2 prefill
generalization + chunked-context loop, the MLA attention block with weight
absorption, and the grouped-topk router extension.

**This unlocks, in order of how much is left afterwards:**

| Model | What remains after the shared unlock | Gateable on GB10? |
|---|---|---|
| **DeepSeek-V2 / V2-Lite** | config + loader + forward only (W7) | **YES — the vehicle** |
| **DeepSeek-V3 / V3.2** | config deltas (`q_lora_rank=1536`, sigmoid routing, 256+1 experts, `n_group/topk_group`) — V3.2 additionally needs the DSA indexer, out of scope | e2e **HW-BLOCKED**; config/loader-slice/unit only |
| **DeepSeek (v1) MoE** | **no MLA at all** — plain MHA (§0.7) + the DeepSeek MoE router; the cheapest row, largely covered by assets we already have | e2e blocked by checkpoint size (67B), 16B variant would fit |
| **Kimi-K2 / K2.5 (text)** | DeepSeek-V3 by config composition — essentially free once V3's config path exists | **HW-BLOCKED** (~1T) |
| **Kimi-Linear-48B** | the MLA half is free **minus the NoPE variant** (`use_nope=True`, i.e. skip rotary — a *simplification*); the KDA half is a **separate kernel campaign** (§2.5): reuses our GDN cache/metadata/conv/chunked-delta/WY-solve, but needs a NEW per-channel-decay gate + gated-linear-attention output kernel + three separate q/k/v convs | **HW-MARGINAL** (~89 GiB of ~110) |
| **DeepSeek-V4** | new namespace, sliding window, 584 B `fp8_ds_mla` page layout, DSpark drafting — its own spike | HW-BLOCKED |

**DISJOINT — MiniMax needs none of the above:**
`MiniMaxM2ForCausalLM` is dense GQA + MoE + **partial RoPE** + **per-layer sliding
window**. Against our current tree it needs: partial-rotary support in the RoPE
op, per-layer sliding-window attention (a `SlidingWindowSpec` KV group we have
declared but never exercised), and the expert-bias router (shared with DeepSeek's
`e_score_correction_bias` — the one genuine overlap). **At ~428 GiB bf16 it does
not fit in memory OR on disk, so it is HW-BLOCKED e2e and should be sequenced
behind the sliding-window track (`ROAD-V1-C5`), not behind MLA.**
`MiniMaxM3SparseForCausalLM` needs a sparse-attention indexer subsystem and is a
third, separate campaign.

---

## 12. Structured spike contract

The prose above (§0-§11) is the full spike; this restates it in the
record-checker's structured fields. Rows:
`MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-for-causal-lm`,
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`,
`MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm`.

### Scope
Design MLA (Multi-head Latent Attention) + the compressed-latent KV cache as a
new additive attention family, plus the DeepSeek-V2/V3 model family that consumes
it, and determine what of the DeepSeek / Kimi / MiniMax portfolio is actually
gateable on GB10 (§1). In scope: the upstream chain with `file:line` on both
sides; the GB10 dense-MLA and MLA-prefill backend selection read from source; the
KV-cache/allocator/op/runner impact; weight absorption and the prefill-MHA vs
decode-MQA forms; the `noaux_tc` grouped-topk router; the hardware verdict per
family; the test inventory; a W0-W10 breakdown. Out of scope and left
`INVENTORIED`: sparse MLA / DSA (V3.2 indexer), DeepSeek-V4, MiniMax-M3's sparse
indexer, all multimodal Kimi rows, MTP/Eagle implementation, fp8 KV cache
(`fp8_ds_mla`), TP/EP.

### Upstream chain
`vllm/model_executor/models/deepseek_v2.py` @ `e24d1b24` — `:952
DeepseekV2MLAAttention` (projections `:453-526`, optional `q_lora_rank` branch
`:484-494`, `kv_a_proj_with_mqa` `:511`, `kv_a_layernorm` `:516`, `kv_b_proj`
`:518-519`), decoupled RoPE `:580-595`, MoE + `noaux_tc` router `:276-393`,
decoder-layer attention selection `:1201-1211` (`use_mha` -> `:133
DeepseekAttention`), subclasses `:1780/:1909/:1913/:1917`; registry
`registry.py:90-94`. Attention layer
`vllm/model_executor/layers/attention/mla_attention.py` — glossary `:55-73`,
`forward_mha` `:66-89, :2344`, `forward_mqa` `:94-117, :2428`, chunked prefill
`:125-175`, absorption `:875,899,960,962` consumed at `:780-789,:1034`, common
backend/impl `:1206,1246,1281,1291,1338,1401,1988`; wrapper
`vllm/model_executor/layers/mla.py:14,34`. Backend selection
`vllm/platforms/cuda.py:93-142` — **`major == 12` -> `[TRITON_MLA,
FLASHINFER_MLA_SPARSE_SM120]` at `:129-133`**; prefill selection
`vllm/v1/attention/backends/mla/prefill/selector.py:48-76` (major 12 ->
`[FLASH_ATTN]`). Backend `vllm/v1/attention/backends/mla/triton_mla.py:81,130,134,
189,41,57`. **Real kernel outside vLLM:**
`vllm/v1/attention/ops/triton_decode_attention.py:1-11` records the chain vLLM <-
SGLang `decode_attention.py` <- lightllm `deepseek2/triton_kernel/
gqa_flash_decoding_stage{1,2}.py`. Cache write `vllm/_custom_ops.py:2532,2545`
from `vllm/v1/attention/backend.py:995,1075`; fusion
`vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:40`. KV spec
`vllm/v1/kv_cache_interface.py:363-403` (page formula `:380-398`). Router
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:28-64,80-145`.
Adjacent families: `kimi_linear.py:179,213,249,263,303-325` +
`transformers_utils/configs/kimi_linear.py:144-148`;
`mamba/gdn/kimi_gdn_linear_attn.py:85,142-198` + `fla/ops/kda.py:19-26,436,717,
960,1019,1126,1182,1603-1617`; `configs/kimi_k25.py:10,72,83,101-103` +
`kimi_k25.py:360-364`; `minimax_m2.py:72,139,178,196-214,432`;
`vllm/models/minimax_m3/nvidia/model.py:286,386,673-690` +
`common/indexer.py:63,120,519`; removals `registry.py:721-724`.

### Our baseline
**Zero MLA** — only deferral comments at `src/vllm/platforms/cuda.cpp:54-56`,
`include/vllm/v1/kv_cache_interface.h:26,44`,
`include/vllm/v1/attention/backend.h:28`. Reusable (§3): the model
self-registration seam, `dense_attn_block.h` as structural template,
`vt::MoeGroupedGemmBf16`(+`WmmaPipe`+split-K) at ~1.2x vLLM's `fused_moe` rate,
`RunMoeBlock`/`qwen3_5_moe_block.h` + the shared-expert path, the decode
CUDA-graph driver (3 siblings, `decode_graph_sizes.h`), the vendored FA-2 varlen
prefill + split-KV decode/combine, the `cuda_arch_tactics` seam (`b4f14ee`), and
the fusion catalog. NOT reusable: `vt::ReshapeAndCache`/`vt::PagedAttention`
(`include/vt/ops.h:1167,1201` — separate K/V), `FullAttentionSpec` + the
allocator (`runner.cpp:399-400,487-492` — hardcoded factor 2, shape from HF
config), and `MoeRouterTopKArgs` (`include/vt/ops.h:311-318` — softmax+top-k+
renorm only, no sigmoid/bias/group).

### Port map
Per §6. New: `mla_attention.{h,cpp}`, `deepseek_v2.h`,
`deepseek_v2_registry.cpp`, `deepseek_v2_weights.cpp`, `deepseek_v2.cpp`,
`src/vt/cuda/cuda_mla_attn.cu` + its CPU reference, tests. Edits (all designed
seams): `include/vllm/v1/kv_cache_interface.h` (`MLAAttentionSpec` +
`KVCacheSpecKind::kMLA`), `include/vt/ops.h`/`src/vt/ops.cpp` (new `OpId`s +
router args), `src/vllm/platforms/cuda.cpp` (the MLA priority branch),
`src/vllm/v1/worker/gpu/runner.cpp` (spec-driven cache sizing + the MLA cache
view + group id), CMake TU glue.

### Tests to port
Per §7. Ported: `test_mla_backends.py`, `test_mla_prefill_selector.py`,
`test_mla_prefill_registry.py`, the MLA cases of
`test_attention_backends_selection.py`, `test_mla_prefill_quant_output.py`,
`test_mla_decode_cpu.py`, `test_concat_mla_q.py`,
`test_rotary_embedding_mla_cache_fused.py`,
`test_mla_rope_kvcache_cat_fusion.py`, `test_fuse_mla_dual_rms_norm.py`, and the
grouped-topk router cases. Checked in SKIPPED with tracked reasons:
`test_mla_attn_quant_fusion.py` (quant OOS), `test_deepseekv2_tp.py` (no
multi-GPU), and the V3.2/V4 sparse set (`test_deepseek_v4_mega_moe.py`,
`test_dspark_noncausal_sparse_mla.py`, `test_sparse_mla_backends.py`,
`test_flashinfer_sparse_mla_sm120_api.py`,
`test_indexer_deepseek_v4_slot_mapping.py`,
`test_fused_deepseek_v32_norm_rope.py`,
`test_fused_deepseek_v4_qnorm_rope_kv_insert.py`). NOT ported and recorded as
unreachable on GB10: `test_flashmla.py`, `test_cutlass_mla_decode.py`, the ROCm
aiter MLA set. New (no upstream twin): `test_deepseek_v2_load.cpp`,
`test_deepseek_v2_forward.cpp`, `test_deepseekv2lite_paged_engine.cpp`,
`scripts/deepseekv2-oracle-capture.py`, `scripts/deepseekv2-neartie-gap.py`.

### Gates
Per §8. SACRED correctness on **DeepSeek-V2-Lite bf16** under the
[[near-tie-distributional-gate]] form determined at W0 (K=5 determinism probe;
strict where well-posed, else our token within 0.5 nats of vLLM's teacher-forced
argmax, 0 forward-divergent). Every W-step: dgx CUDA `-Werror` 0-warn on a clean
build, `memcheck` 0, and the regression set UNCHANGED (Qwen3-dense 16/16,
Qwen3-Coder 6/6, 27B 235/235, 35B 315/315). W1/W2 are behaviour-preserving and
must prove byte-identity. New `vt::` ops gate against a CPU reference before any
model uses them, with run-to-run bit-reproducibility on any split/reduce. Speed:
match or beat **graphed** vLLM 0.25.0 on **every** axis at c1/c2/c4/c8, fresh
server per concurrency at verified 0.0% prefix-cache hit rate, idle box under
`flock /tmp/gpu`, 2 reps cold-leg-discarded; parity levers ship DEFAULT-ON before
the binding run. **V3/V3.2/K2.5/M2/M3 get NO e2e gate — HW-BLOCKED (§5.2);** only
config/registry resolution, weight-map + single-layer-slice loading, and unit
parity at their dimensions.

### Dependencies
Per §9. DeepSeek-V2-Lite must be downloaded to dgx (~29.3 GiB against **238 GiB
free on a single 94%-full root filesystem**; nothing else in the campaign should
be downloaded — Kimi-Linear ~89 GiB pushes root past 97%, MiniMax-M2 ~428 GiB
does not fit at all). The vLLM 0.25.0 oracle must resolve `DeepseekV2ForCausalLM`
on GB10 and be **observed** selecting `TRITON_MLA` + FA-2 MLA prefill at W0 (this
spike read the logic; runtime confirmation is still owed). W1's spec-driven KV
allocator blocks every other W. Single-GPU only (no TP/EP/multi-GPU). All dgx
runs take `flock /tmp/gpu`. Upstream pin stays `e24d1b24`.

### Work breakdown
W0 ground the facts (checkpoint + real config + observed backend selection +
determinism probe). W1 spec-driven KV allocation + `MLAAttentionSpec` +
`KVCacheSpecKind::kMLA` (behaviour-preserving, zero MLA math). W2 platform MLA
backend priorities + prefill selector + registry. W3 `vt::ConcatAndCacheMla` +
the grouped-topk router extension. W4 `vt::MlaDecodeAttention` (two-stage
split-KV MQA over the latent, deterministic combine, `cuda_arch_tactics`-
registered). W5 MLA prefill (FA-2 varlen generalized to qk 192 / v 128 + the
workspace-bounded chunked-context loop with LSE merge). W6 the MLA attention
block + load-time weight absorption (`kv_b_proj -> W_UK`/`W_UV`) + the
prefill-MHA / decode-MQA dispatch. W7 the DeepSeek-V2 model TU (registry all four
aliases, loader, forward). W8 the SACRED gate on DeepSeek-V2-Lite. W9 speed close
(decode-graph sibling + MLA fusion recipes + the binding every-axis grid). W10
the blocked-row honesty pass (config/loader-slice/unit parity for V3/V3.2;
`HW-BLOCKED` recorded where e2e cannot run). Full table with per-W gates in §10.

### Risks/decisions

**Decision — the gate vehicle is DeepSeek-V2-Lite, and it is the ONLY one.** It
is the single campaign member that fits GB10 (~29.3 GiB of ~110 GiB usable), it
exercises the complete MLA geometry (`kv_lora_rank=512`, `qk_nope=128`,
`qk_rope=64`, `v_head_dim=128`, so the same 576-wide latent as V3), it exercises
MoE with shared experts and grouped-topk routing, and upstream itself uses it as
the small-MLA smoke vehicle across many test modules. It has **two** coverage gaps (§5.1),
both of which must be stated in the row rather than papered over: (a)
`q_lora_rank=null`, so `fused_qkv_a_proj`/`q_a_layernorm`/`q_b_proj` and their
`packed_modules_mapping` un-fusing are built for V3 but unit-gated only; and (b)
`n_group=topk_group=1` with `softmax`/`greedy`, so V2-Lite has **no
`e_score_correction_bias` at all** and the entire `noaux_tc` router — sigmoid
scoring, the biased-select/unbiased-weight asymmetry, the two-level group mask —
is likewise unit-gated only. Both are dimension-free and unit-testable, so this
is an acceptable trade, but "MLA + DeepSeek MoE gated" without naming them would
be overclaiming.

**Decision — use `VLLM_MLA_DISABLE=1` as the bring-up oracle before touching
kernels.** It is not a perf switch: it flips `use_mla`
(`config/model.py:1638-1639`), moves the reported `head_size` 576 -> 192, and
routes the SAME checkpoint through `DeepseekV2Attention`
(`deepseek_v2.py:446-610`, selected at `:1207-1213`), which materializes full
MHA. That is a same-checkpoint, same-oracle A/B between the latent and
materialized forms — the cheapest possible way to localize a W6 numerics
discrepancy, and it should be exhausted before any kernel is suspected.

**Decision — target `TRITON_MLA`, not FlashMLA/CUTLASS MLA.** Read from
`cuda.py:129-133`, not inferred. This deletes a large class of sm90/sm100-only
kernel ports from scope and points the decode work at a two-stage split-KV
flash-decode whose structure we already have. The corresponding **risk** is that
this is a source read: W0 must OBSERVE the oracle selecting it before W4 spends
kernel effort ([[profile-vllm-actual-kernels-port-1to1]] — never infer
"at-parity" or "this is what runs" from source alone).

**Risk (headline) — the KV cache is a structural change, not a parameter.** Our
allocator hardcodes K+V and derives shape from the HF config
(`runner.cpp:399-400,487-492`); both `vt::ReshapeAndCache` and
`vt::PagedAttention` have K/V-pair signatures. The mitigation is to land the
spec-driven sizing FIRST as a behaviour-preserving W1 with all four existing
gates byte-identical, so the MLA math never lands entangled with an allocator
refactor.

**Risk — chunked-prefill workspace is the most under-estimated item.**
`mla_attention.py:125-175` bounds memory by chunking the context and merging with
LSE. We have no workspace-bounded attention loop; we have a combine. This is
genuinely new engine machinery and W5 should be expected to be the largest
kernel-side W.

**Risk — near-tie determinism is unprobed on this model.** The gate FORM is
chosen at W0 by measurement, not assumed. If vLLM's own greedy is
self-inconsistent on V2-Lite the distributional gate applies; if deterministic,
strict-where-well-posed applies. Do not pick before measuring.

**Risk — MoE numerics move tokens.** Swapping accumulation order re-resolves
bf16 near-ties (the Qwen3-Coder W5 precedent, where one token moved and the gate
then got STRICTER). Any change to the expert GEMM or router requires re-dumping
goldens through the ratified teacher-forcing procedure, never re-baselining.

**Honest statement — most of this campaign cannot be gated here.** DeepSeek V3 /
V3.2 (671B), Kimi-K2/K2.5 (~1T), MiniMax-M2 (~230B) and MiniMax-M3 are
**HW-BLOCKED on GB10 end-to-end** — not "pending optimization", not "needs a
smaller batch": they do not fit in 119 GiB unified memory, and two of them do not
fit on the 238 GiB of free disk either. We will gate config resolution, weight
mapping on a slice, and unit parity at their dimensions, and record the rest as
`HW-BLOCKED`. We will not propose an e2e gate we cannot run.

**Correction recorded — "Kimi K3" does not exist** at the pin; K2.5 is the newest
(`registry.py:448`). **Correction recorded — MiniMax is not MLA**: M2 is dense
GQA + MoE + partial RoPE + sliding window, M3 is GQA + a sparse indexer; the
matrix dependency cells claiming `MLA/latent KV` for the M3 rows are hypotheses
and are corrected in this change. **Correction recorded —
`DeepseekForCausalLM` is not MLA either** (`deepseek_v2.py:1201-1211` selects
plain MHA for `model_type == "deepseek"`). **Correction recorded — Kimi Linear
IS MLA** (NoPE variant), which is why the MLA unlock reaches the Kimi line and
not only DeepSeek.

**Decision — KDA is adjacent to GDN, not a re-use of it.** The state machinery
(cache layout, `GDNAttentionMetadata`, conv update, chunked delta recurrence, WY
solve) is genuinely shared; the per-channel decay gate, its low-rank
`f_a_proj`/`f_b_proj` parameterization, the gated-linear-attention output kernel
and the three separate q/k/v convs are NOT. Kimi-Linear is therefore two
campaigns (MLA-NoPE + a KDA kernel), not one, and at ~89 GiB of ~110 GiB usable
it is HW-MARGINAL even then.
