# SPIKE — MLA (Multi-head Latent Attention) + the DeepSeek / Kimi / MiniMax families

**Status:** **THE W-PLAN IS COMPLETE — W0 through W10 have landed** (see `### Work breakdown` and
the per-step sections below). The campaign's ONE e2e-gateable member, **DeepSeek-V2-Lite, is
CORRECTNESS-COMPLETE (SACRED gate 8/8) and SPEED-SHORT (0.86-0.95x vLLM tok/s)**, so its row is
`ACTIVE`, not `DONE`, with one named open lever (batch-1 dense projections off cuBLAS `gemvx` onto a
tensor-core GEMM). W10 set every other row to its final honest state: V3/V3.2, MiniMax-M2 and (cross-
claim) GLM-5 are **`BLOCKED`** — hardware, and for the DSA models a dependency dead-end as well;
`DeepseekForCausalLM` (plain MHA, not MLA) and Kimi-Linear stay `SPIKE`. **The block is NOT closeable
and this spec stays LIVE.** Two coverage gaps are PERMANENT on this hardware (`noaux_tc` router and
the `q_lora` query branch, both unit-gated only); **GLM-4.7-Flash (31.2B, 58.2 GiB) fits GB10 and
would close both** — it is the named next vehicle. Historical detail follows. W0 grounded
every fact the spike flagged as an unverified source read — all CONFIRMED by
runtime observation, nothing contradicted. W1 landed the behaviour-preserving
spec-driven KV allocation + `MLAAttentionSpec`. W2 landed the MLA branch of the
backend selector as a DATA table plus the `is_mla()`/`is_sparse()` filter that
leaves a zero-edit seam for DSA. W3 landed the two new `vt::` ops —
`ConcatAndCacheMla` and the grouped-topk (`noaux_tc`) router extension — each
CPU-reference-gated. W4 landed the MLA DECODE kernel `vt::MlaDecodeAttention` and filled
`TritonMLABackend::get_impl_cls()`. W5 landed MLA PREFILL (the FA-2 launcher
generalized to QK 192 / V 128) plus the workspace-bounded CHUNKED-CONTEXT loop
and its two supporting ops. W6 landed the MLA ATTENTION BLOCK + LOAD-TIME WEIGHT
ABSORPTION — the layer that finally COMPOSES W3's cache write, W4's decode and
W5's prefill, with both `q_lora_rank` query branches, the decoupled RoPE, the
`kv_b_proj -> W_UK/W_UV` split, and the prefill-MHA / decode-MQA dispatch; the
absorbed-vs-unabsorbed equivalence is proven numerically three ways. W7 landed
the DEEPSEEK-V2 MODEL — registry (`DeepseekV2ForCausalLM` only), config parse,
the MLA-only KV group, the BF16 loader with the `kv_b_proj -> W_UK/W_UV`
absorption applied at LOAD time, and the forward composing W6's MLA block with a
DeepSeek MoE block (grouped router + SHARED experts) and the
`first_k_dense_replace` dense layers. **The real DeepSeek-V2-Lite checkpoint now
LOADS (5291/5291 tensors, none leftover) and FORWARDS: "The capital of France
is" -> ` Paris`.**
**It is still not a SUPPORTED model.** A loading, forwarding model is not a
supported one: the SACRED token-exact gate vs the vLLM 0.25.0 oracle is W8, so
**no model row moves off `SPIKE`** and W8-W10 remain plan only.
**Base:** `b4f14ee` (spike) / `fb3fd5d` (W0+W1) / `a05437f` (W2+W3) / `ed2c342` (W4) / `5395203` (W5) / `2846467` (W6) / `ce43c51` (W7).
**Pinned oracle:** `/home/mudler/_git/vllm` @ `e24d1b24` (v0.25.0 audit target
`702f4814fe54`). The executable oracle venv `~/venvs/vllm-oracle` lives on
**dgx.casa**. Every claim below is grounded in repo source; the two claims that
needed a RUNTIME observation (§9/§12) were OBSERVED at W0 on 2026-07-21 and are
recorded in `### Work breakdown`.
**Rows covered:** `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-deepseek-for-causal-lm`,
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`,
`MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm`.
**Claim:** `CLAIM-MLA-DEEPSEEK`.
**Parent plan:** [breadth-sweep-plan.md](breadth-sweep-plan.md) §B.3 Tier 3 named
"MLA family = new attention, new campaign, needs its own leaf spike". This is
that leaf spike.

**Post-campaign fix — 2026-07-23 (`CLAIM-MLA-PREFIX-CACHE-ASSERT`, base `6abe09c`).** The DeepSeek-V2
SACRED gate `test_deepseek_v2_paged_engine` aborted (SIGABRT) at
`single_type_kv_cache_manager.cpp:293` under **asserts-enabled** builds (empty/Debug `CMAKE_BUILD_TYPE` —
no NDEBUG). ROOT CAUSE: `FullAttentionManager::find_longest_cache_hit` asserted the group's spec
`kind() == kFullAttention`, but the MLA group's spec is `MLAAttentionSpec` (`kind() == kMlaAttention`),
which W1 correctly routes to the ORDINARY `FullAttentionManager` (registry, mirroring upstream). DeepSeek-V2
defaults automatic prefix caching ON, so the single-group `UnitaryKVCacheCoordinator` calls the manager's
`find_longest_cache_hit` on the FIRST request → the strict precondition aborts. DETERMINISTIC (fires before
any prompt forwards); LATENT since the original `ec6f4be`; masked in every prior gate because the canonical
build is `Release` (NDEBUG folds `assert()` out) — L7/S3/G7/L6 genuinely passed 8/8. NOT a regression from
LMCache/Mistral/SentencePiece. FIX (grounded in vLLM `single_type_kv_cache_manager.py:578-582`: upstream
asserts `isinstance(spec, FullAttentionSpec | ChunkedLocalAttentionSpec)` and `MLAAttentionSpec(FullAttentionSpec)`
is a subclass): relax our precondition to `kFullAttention || kMlaAttention || kChunkedLocalAttention`. MLA
prefix caching IS valid — blocks are opaque `page_size_bytes` rows hashed by token, so the cache-hit walk is
spec-kind-agnostic and runs byte-for-byte as upstream. Two new CPU unit cases prove the MLA lookup does not
abort AND the restored prefix restores the byte-identical KV pages (the previously untested MLA restart-hit).
Restores DeepSeek-V2 8/8 (3/3, asserts-on, incl. Phase-3 prefix reuse byte-identity); full-attention models
byte-identical (Release changed-TU object unchanged); `-Werror` 0 warnings. Does not change any W0-W10
conclusion or state.

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
| `include/vllm/model_executor/models/mla_attention.h` **(W6 — LANDED)** | the MLA block: projections (both `q_lora_rank` branches), decoupled RoPE + the YaRN cos\|sin cache + the mscale^2 scale, the load-time absorption transform, and the MHA-prefill / MQA-decode forms — the analog of `dense_attn_block.h`, ported from `mla_attention.py` + `mla.py` |
| `src/vllm/model_executor/layers/attention/mla_attention.cpp` **(W6 — LANDED)** | its implementation (it sits under `layers/attention/` beside W5's `mla_chunked_context.h`, not under `models/`) |
| `include/vt/ops.h`, `src/vt/ops.cpp`, `src/vt/cpu/cpu_ops.cpp`, `src/vt/cuda/{cuda_matmul.cu,cuda_mla_attn.cu}` **(W6 — EDIT)** | `vt::BatchedMatmul` (`torch.bmm`, the two absorption GEMMs) + `vt::ConcatMlaNopeRope` (`concat_mla_q` generalized to serve `_concat_k_nope_k_pe`), plus the two additive relaxations `vt::RopeFromCache` (stride-driven q/k) and `vt::MatmulBT` (row-strided activation) |
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
| `tests/v1/attention/test_mla_backends.py` | MLA backend correctness vs a reference across batch/seq shapes | parity `tests/vt/test_ops_mla_attn.cpp` (W4, decode) + `tests/vt/test_ops_mla_prefill.cpp` and `tests/vt/test_ops_mla_chunked_context.cpp` (W5, prefill) — **PORTED** | W4/W5 |
| `tests/v1/attention/test_mla_prefill_selector.py` | MLA **prefill** backend selection per capability | doctest `tests/vllm/v1/attention/test_mla_backend_registry.cpp` | W2 |
| `tests/v1/attention/test_mla_prefill_registry.py` | prefill backend registry/enum integrity | same | W2 |
| `tests/v1/attention/test_attention_backends_selection.py` (MLA cases) | `_get_backend_priorities` MLA branch ordering | extend `tests/vllm/v1/attention/test_attn_backend_registry.cpp` (today non-MLA only, `:121`) | W2 |
| `tests/v1/attention/test_mla_prefill_quant_output.py` | prefill output vs reference incl. the V2-Lite dims (`:158`) | parity `tests/vt/test_ops_mla_prefill.cpp` — **PORTED** for the bf16 arm; its QUANT (fp8-output) arms are NOT ported and say so in the file header: they require `is_device_capability_family(100)` (`mla_attention.py:1382-1385`) and are UNREACHABLE on sm_121 | W5 |
| `tests/kernels/attention/test_mla_decode_cpu.py:47` | MLA decode against a CPU reference — **device-independent, so it is the numerical ORACLE to port FIRST**, before any CUDA work | doctest + parity | W4 |
| `tests/kernels/moe/test_grouped_topk.py:46` and `tests/kernels/moe/test_routing.py:441` | the dedicated grouped-topk correctness cases | doctest `tests/vt/test_ops_moe_router_grouped.cpp` | W3 |
| `tests/evals/gsm8k/configs/DeepSeek-V2-Lite-Instruct-FP8.yaml` | upstream's own V2-Lite accuracy target | secondary e2e sanity (our SACRED gate remains token-exact) | W8 |
| `tests/kernels/test_concat_mla_q.py` | the q concat helper | doctest `tests/vt/test_ops_mla_absorb.cpp` — **PORTED** (W6). BOTH arms: `test_concat_mla_q_contiguous` (`:22-34`) and `test_concat_mla_q_transposed_nope` (`:37-63`), the latter being the case that exists because the real nope operand is the NON-CONTIGUOUS transposed `torch.bmm` output. Its `atol=0, rtol=0` comparison is ported as bit-exactness, since a concat is a pure copy. Our op is the generalization that ALSO serves `_concat_k_nope_k_pe` (broadcast-rope arm added) | W6 |
| `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py` | fused RoPE + MLA cache write | doctest (unfused first, fusion later) | W3/W8 |
| `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` | the RoPE+concat-cache fusion pass | fusion-catalog recipe test | W8 |
| `tests/compile/passes/test_fuse_mla_dual_rms_norm.py` | dual-RMSNorm fusion (`q_a_layernorm`+`kv_a_layernorm`) | fusion-catalog recipe test | W8 |
| `tests/compile/passes/test_mla_attn_quant_fusion.py` | MLA attn+quant fusion | SKIPPED (quant out of scope), tracked | — |
| grouped-topk router tests under `tests/kernels/moe/` | sigmoid + bias + group masking selection | doctest `tests/vt/test_ops_moe_router_grouped.cpp` | W3 |
| `tests/lora/test_deepseekv2_tp.py` | DeepSeek-V2 layer wiring under TP | SKIPPED (no multi-GPU), tracked | — |
| `tests/models/test_deepseek_v4_mega_moe.py`, `tests/v1/attention/test_dspark_noncausal_sparse_mla.py`, `test_sparse_mla_backends.py`, `test_flashinfer_sparse_mla_sm120_api.py`, `test_indexer_deepseek_v4_slot_mapping.py`, `tests/kernels/test_fused_deepseek_v32_norm_rope.py`, `test_fused_deepseek_v4_qnorm_rope_kv_insert.py` | V3.2/V4 sparse + DSA | **OUT OF SCOPE** — checked in SKIPPED with the reason "sparse MLA / V4 not spiked" | — |
| `tests/kernels/attention/test_flashmla.py`, `test_cutlass_mla_decode.py`, `tests/rocm/aiter/*`, `test_rocm_aiter_mla_decode_metadata.py` | sm90/sm100/ROCm-only MLA kernels | **NOT PORTED** — unreachable on GB10 per §2.3; recorded, not silently dropped | — |
| tool/reasoning parsers (`tests/tool_parsers/test_deepseekv3_tool_parser.py` and siblings, `tests/reasoning/test_deepseekr1_reasoning_parser.py`) | DeepSeek chat/tool/reasoning formats | separate serving row, inventoried not spiked | — |

| `torch.bmm` at `mla_attention.py:789`, `:1034` (no dedicated upstream test module — absorption is covered only through `test_mla_backends.py`) | the two batched GEMMs weight absorption is made of | doctest `tests/vt/test_ops_mla_absorb.cpp` (`vt::BatchedMatmul` vs a double-precision oracle, both absorption shapes at V2-Lite AND V3 dims, the transposed views both call sites pass) + the block-level equivalence gate in `tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp` — **PORTED** | W6 |
| `mla_attention.py:875-962 process_weights_after_loading` (upstream exercises it only indirectly via the backend tests) | the load-time `kv_b_proj -> W_UK/W_UV` split | `tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp` — **PORTED**: `AbsorbKvBProjBf16` vs an independent literal `.T -> view -> split -> permute` transcription, plus the three-way absorbed-vs-unabsorbed equivalence proof | W6 |

New tests with no single upstream twin (our own gate protocol):
`tests/vllm/models/test_deepseek_v2_load.cpp` (every tensor mapped, none left
over), `test_deepseek_v2_forward.cpp` (real-checkpoint prefill argmax),
`test_deepseekv2lite_paged_engine.cpp` (the SACRED gate), plus
`scripts/deepseekv2-oracle-capture.py` and `scripts/deepseekv2-neartie-gap.py`
mirroring the Qwen3-Coder pair.

---

## 8. Gates

**Correctness (SACRED). SETTLED AT W0 — the gate is the STRICT form.** The K=5
probe was run and, **at batch=1 (the gate regime), vLLM's own greedy on
DeepSeek-V2-Lite is DETERMINISTIC on 8/8 prompts**. So the STRICT-where-
well-posed near-tie-robust gate applies — our token within 0.5 nats of vLLM's
teacher-forced argmax, strict where equal, 0 forward-divergent. **Caveat that
must not be lost:** the SAME battery run as ONE BATCH looks non-deterministic on
3/8 prompts. That is a batching artifact (re-ordered reductions), not model
non-determinism; every determinism claim on this model must be made at batch=1. **The vehicle is DeepSeek-V2-Lite bf16.** V3/V3.2/K2.5/M2
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
| **W0** ✅ **DONE 2026-07-21** | **Ground the facts.** Download DeepSeek-V2-Lite; confirm its real `config.json` against §5 (especially `q_lora_rank is None`); run the vLLM oracle on it and CONFIRM from its logs that `TRITON_MLA` + FA-2 MLA prefill are selected on sm_121; capture a K=5 greedy determinism probe to fix the gate form. | checkpoint | **PASSED** — checkpoint fetched (29.26 GiB, 4 shards) and generating in the oracle; `TRITON_MLA` + `FLASH_ATTN` MLA prefill OBSERVED on sm_121 (3 runs) and the two-stage decode kernels seen executing; every §5.1 config number confirmed, none corrected; determinism = **DETERMINISTIC 8/8 at batch=1** ⇒ STRICT gate |
| **W1** ✅ **DONE 2026-07-21** | **Spec-driven KV allocation (behaviour-preserving, no MLA).** Size every cache buffer from `spec->page_size_bytes()`; stop reconstructing shape from `config_`; add `KVCacheSpecKind::kMlaAttention` + `MLAAttentionSpec` with the single-tensor formula (merge assertions deferred with `merge()` itself; the distinct `kind()` is what a future `merge()` keys on). | — | **PASSED** — 27B 235/235, 35B 315/315, Coder 6/6, Qwen3-dense 16/16 all UNCHANGED; CUDA `-Werror` 0 warn/0 err; `test_kv_cache_interface` 21/21 (4 new MLA-spec cases), `test_runner` 15/15, `test_llm_engine` 5/5; new path proven EXERCISED via `fa_page_size_bytes()` + a `page_size_padded` case |
| **W2** ✅ **DONE 2026-07-21** | **Platform MLA backend priorities + registry.** Port the `use_mla` branch of `_get_backend_priorities` (`cuda.py:93-142`) and the MLA prefill selector; register an `MLA` attention backend name resolving to TRITON_MLA on major 12. | W1 | **PASSED** — ports of `test_attention_backends_selection.py` (MLA cases), `test_mla_prefill_selector.py`, `test_mla_prefill_registry.py` in `tests/vllm/v1/attention/test_attn_backend_registry.cpp`; non-MLA selection byte-identical; `use_mla=true` on sm_121 RESOLVES to `TRITON_MLA` |
| **W3** ✅ **DONE 2026-07-21** | **New `vt::` ops — cache write + router.** `vt::ConcatAndCacheMla` (CPU ref + CUDA) mirroring `concat_and_cache_mla`; extend `MoeRouterTopKArgs` with `scoring_func` / `e_score_correction_bias` / `num_expert_group` / `topk_group` / `routed_scaling_factor` and implement the grouped-topk selection rule. | W1 | **PASSED** — `tests/vt/test_ops_mla_cache.cpp` + `tests/vt/test_ops_moe_router_grouped.cpp`; CPU-vs-CUDA parity (exact on the cache write, exact ids on the router); grouped-topk gated at REAL V3 dims (256 experts, n_group=8, topk_group=4, sigmoid, WITH `e_score_correction_bias`); ungrouped router byte-identical |
| **W4** ✅ **DONE 2026-07-22** | **MLA decode attention op.** `vt::MlaDecodeAttention` — the two-stage split-KV MQA decode over the latent (QK 576 / V 512), structurally mirroring `decode_attention_fwd` (vLLM <- SGLang <- lightllm) and reusing our FA-2 split+deterministic-combine pattern; registered via `cuda_arch_tactics`. | W3 | **PASSED** — [`tests/vt/test_ops_mla_attn.cpp`](../../tests/vt/test_ops_mla_attn.cpp) (port of `test_mla_decode_cpu.py` incl. its `ref_mla` two-pass oracle, its bs=4/256/16/576/512/16 parametrization, both `varlen` arms and its NaN-padding out-of-bounds detector; plus the `test_mla_backends.py` shape sweep). dgx sm_121: 11/11 cases, 2,303,193 assertions at the REAL V2-Lite geometry (576/512/64, block 16, mscale^2 scale) over ragged / multi-block / single-block / EVERY split boundary / 128-head V3 / non-BLOCK_H head counts / a 288-256 block-32 geometry / bf16 + f32; run-to-run BIT-exact over 5 runs; memcheck **0**, racecheck **0 hazards**, synccheck **0**; clean CUDA build 0 warn/0 err; 27B 235/235 + 35B 315/315 + Coder 6/6 + Qwen3-dense 16/16 + OPT 6/6 UNCHANGED. Deviation from the plan cell: registered via the ordinary `RegisterOp` table (the W3 pattern), NOT `cuda_arch_tactics` — there is no per-arch tactic to choose between yet, and adding an empty selector would be ceremony; the tactic seam is a W9 concern when a second kernel exists |
| **W5** ✅ **DONE 2026-07-22** | **MLA prefill + the chunked-context loop.** Generalize the vendored FA-2 varlen launcher to qk 192 / v 128 (the GB10 MLA prefill backend is FLASH_ATTN, §2.3) and implement the compute-friendly materialized form plus the workspace-bounded chunked-context loop with LSE merge. | W4 | **PASSED** — three new ops + the driver, all `file:line`-cited both sides: `vt::MlaPrefillAttention` <- `mla/prefill/flash_attn.py:153-248` over the vendored FA-2; `vt::GatherMlaCache` <- `cache_kernels.cu:992-1064`; `vt::MergeAttnStates` <- `merge_attn_states.cu:18-192`; `BuildMlaChunkedContext`/`ComputeMlaPrefillContext`/`ForwardMlaPrefillMha` <- `mla_attention.py:1422-1451,1667-1745,2094-2199,2344-2425` in the new header [`mla_chunked_context.h`](../../include/vllm/model_executor/layers/attention/mla_chunked_context.h). **The FA-2 launcher WAS generalized, and it was tractable exactly as W4 predicted** — because upstream does not ask FA-2 for asymmetric head dims either: `requires_v_padding` is TRUE on GB10 (`flash_attn.py:88-99`), so V is ZERO-PADDED 128 -> 192 and the output sliced back (`:164-168`, `:196-197`), leaving a plain SYMMETRIC head_dim-192 kernel. The generalization is therefore (a) two new explicit instantiations of the UNCHANGED generic `run_mha_fwd_splitkv_dispatch<bf16,192,{true,false}>` template, (b) a NEW launcher entry `LaunchMlaPrefillFA2Bf16` for the CONTIGUOUS-varlen mode (`cu_seqlens_k` instead of `block_table`+`seqused_k`, already supported by the vendored kernel at `flash_fwd_kernel.h:584-590`), (c) the pad/slice pair. **Existing models proven byte-identical structurally AND by gate:** the diff of `cuda_flash_attn_fa2.cu` is 211 insertions / **0 deletions**, the vendored tree gains only 2 new files, `LaunchPrefillFA2Bf16` (the paged launcher every non-MLA prefill calls) is textually untouched — and 27B 235/235 + 35B 315/315 + Coder 138/138 + Qwen3-dense 664/664 + OPT 36/36 all UNCHANGED. Evidence: [`test_ops_mla_prefill.cpp`](../../tests/vt/test_ops_mla_prefill.cpp) 4/4 cases / **2,377,052 assertions** and [`test_ops_mla_chunked_context.cpp`](../../tests/vt/test_ops_mla_chunked_context.cpp) 5/5 / **306,037 assertions** on dgx sm_121, at the REAL V2-Lite geometry (QK 192 / V 128 / latent 576, block 16, mscale²-corrected scale), against an INDEPENDENT double-precision two-pass oracle; the chunked loop is gated against a SINGLE-SHOT whole-sequence oracle that never chunks, over exact / +1 / −1 chunk boundaries, a request with no context, ragged multi-chunk, and V3's 128 heads; ADVERSARIAL reverse-interleaved block tables throughout; NaN-poisoned outputs; run-to-run BIT-exact over 5 runs. `compute-sanitizer` memcheck **0**, racecheck **0 hazards**, synccheck **0** on both binaries. Clean dgx CUDA build **0 warn / 0 err**. **Deviations, recorded not glossed:** (1) the FA-2 kernel is driven with `unpadded_lse = false` into a `[b,h,max_seqlen_q]` scratch which we then convert to the caller's `[h,total_q]` — because upstream FA-2's EMPTY-K early exit (`flash_fwd_kernel.h:1030-1043`) ignores `unpadded_lse` and always writes at the PADDED offset, and a zero-key request IS reachable in the chunked loop (it is the very case `merge_attn_states.cu:100-106` documents), so the mixed layout would clobber valid rows and write out of bounds; the conversion also normalizes FA-2's `+INFINITY` empty-row LSE to `-inf`, the value `merge_attn_states.cu:97-98` normalizes it to anyway. (2) `MergeAttnStates` is SCALAR, not upstream's 128-bit-packed form — same arithmetic, any head_size/stride; vectorization is a W9 concern. (3) The up-projection (`kv_b_proj` + `_concat_k_nope_k_pe`) is a CALLBACK, not inlined: it is the MODEL's weight, i.e. W6. (4) DCP/context-parallel (`_context_parallel_compute_prefill_context`, `reorg_kvcache`) and the fp8-prefill arms NOT ported — single-GPU only, and fp8 prefill needs device-capability family 100 (`mla_attention.py:1382-1385`), unreachable on sm_121. **NO MLA MODEL and NO MLA FORWARD — rows stay `SPIKE`** |
| **W6** ✅ **DONE 2026-07-22** | **MLA attention block + weight absorption.** `mla_attention.{h,cpp}`: projections (both `q_lora_rank` branches), the two RMSNorms, decoupled RoPE, the load-time `kv_b_proj -> W_UK/W_UV` split, and the prefill-MHA / decode-MQA dispatch. | W5 | **PASSED** — the block ([`mla_attention.h`](../../include/vllm/model_executor/models/mla_attention.h) + [`mla_attention.cpp`](../../src/vllm/model_executor/layers/attention/mla_attention.cpp)) plus TWO new `vt::` primitives, all `file:line`-cited both sides: `vt::BatchedMatmul` <- `torch.bmm` at `mla_attention.py:789` (the q-side W_UK fold) and `:1034` (`_v_up_proj`'s W_UV un-projection), and `vt::ConcatMlaNopeRope` <- `concat_mla_q` (`csrc/libtorch_stable/concat_mla_q.cuh` + `cache_kernels.cu:1555-1600`) generalized to also serve `_concat_k_nope_k_pe` (`:2063-2092`). **ABSORPTION IS A LOAD-TIME TRANSFORM PLUS TWO BATCHED GEMMs, exactly as §2.2 predicted** — no new attention kernel was needed for it. **The equivalence is PROVEN NUMERICALLY, three independent ways** (see the `### Work breakdown` entry): the identity itself in double precision (< 1e-11 rel), ours vs the UNABSORBED double oracle (< 2e-4 rel in f32), and — the strongest — the SAME batch driven once through our ABSORBED MQA decode kernel and once through our UNABSORBED materialized-MHA prefill path, agreeing to < 3e-4 (CPU f32) / < 4e-2 (CUDA bf16). Evidence: [`test_mla_attention_block.cpp`](../../tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp) **10/10 cases / 2,372,644 assertions** and [`test_ops_mla_absorb.cpp`](../../tests/vt/test_ops_mla_absorb.cpp) **9/9 / 1,644,807 assertions** on dgx sm_121. Clean dgx CUDA build **0 warn / 0 err**; `compute-sanitizer` memcheck / racecheck / synccheck all **0** on both binaries; regression set UNCHANGED. **Deviations, recorded not glossed:** (1) the A-projections are issued as one GEMM per weight ROW-SLICE rather than one fused GEMM, because `vt::RmsNorm` requires contiguous inputs and relaxing the hottest op in every model for no MLA-specific gain is the wrong trade — the checkpoint PACKING is unchanged (`fused_qkv_a_proj` stays one weight per `packed_modules_mapping`, `deepseek_v2.py:1812-1820`) and the dense block already defaults to exactly this 3-shard form; (2) `vt::ConcatMlaNopeRope` is SCALAR and width-generic where upstream's is 128/256-bit vectorized and templated on NOPE_DIM=512 (a concat is a pure copy, so the bytes are identical; vectorization is W9, same disposition as W5's `MergeAttnStates`); (3) two ADDITIVE relaxations of existing ops — `vt::RopeFromCache` is now stride-driven on q/k and `vt::MatmulBT` accepts a row-strided ACTIVATION — both integer-identical for contiguous tensors, hence bit-identical for every existing model by construction and by gate. **NO MLA MODEL and NO MLA FORWARD — rows stay `SPIKE`** |
| **W7** ✅ **DONE 2026-07-22** | **DeepSeek-V2 model: registry + loader + forward.** New TU, config parse, KV spec (MLA-only group), per-expert bf16 loader + shared experts + `e_score_correction_bias`, forward composing the MLA block + the MoE block + the first `first_k_dense_replace` dense layers. | W6, W3 | **PASSED** — see the `### Work breakdown` W7 entry. Loader gate on the REAL 4-shard DeepSeek-V2-Lite: **5291/5291 checkpoint tensors accounted for, none unmapped and none leftover**, every shape asserted incl. the load-time `W_UK [16,128,512]` / `W_UV [16,512,128]` absorption split and the shared-expert MLP; **forward gate: `The capital of France is` -> argmax ` Paris`** (top-5 ` Paris`, ` the`, ` a`, ` one`, ` also`), run-to-run bit-exact. Batch-ordering gate + shared-expert gates + a CUDA-vs-CPU agreement case (0.0061 worst relative logit error, bit-exact run to run on device). memcheck / racecheck / synccheck all **0**. **DEVIATION from the plan cell: only `DeepseekV2ForCausalLM` is REGISTERED, not all four aliases** — `DeepseekForCausalLM` is plain MHA (§0.7) and would be a false support claim, V3 is fp8/671B (no bf16 loader path, no hardware), V3.2 needs the DSA indexer we do not have; W10 owns those rows. Second deviation: the MoE block is written against the `vt::` ops DIRECTLY rather than reusing `RunMoeBlock`, because DeepSeek's shared expert has NO sigmoid gate and its router is grouped — reusing Qwen's block would have applied Qwen's semantics, and writing it separately also means ZERO edit to the 27B/35B/Coder MoE paths |
| **W8** | **SACRED correctness gate — DeepSeek-V2-Lite.** Paged-engine greedy vs the vLLM 0.25.0 oracle under the W0-determined gate form; goldens + capture/near-tie scripts. | W7 | the gate; regression set UNCHANGED; memcheck 0 |
| **W9** ⚠️ **LANDED 2026-07-22 — ATTRIBUTED MISS, row stays `ACTIVE`** | **Speed close.** Decode CUDA-graph sibling; the MLA fusion recipes (RoPE+concat-cache, dual-RMSNorm) as byte-exact catalog entries; then the binding every-axis grid vs graphed vLLM at c1/c2/c4/c8. | W8 | every-axis parity or an honest attributed miss — **RESULT: HONEST MISS** — see the `### Work breakdown` W9 entry and [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) § "Binding DeepSeek-V2-Lite (MLA) every-axis grid". Decode throughput went **0.50x -> 0.87x** of vLLM at c1 and TTFT now BEATS vLLM at c4/c8, but throughput is short at every concurrency, so the row does NOT reach `DONE`. **DEVIATION from the plan cell: the two levers actually driven were NOT the ones planned.** The planned MLA fusion recipes were not built, because `nsys` said they were not the lever; what the trace found instead was `MlaDecodeStage1` at **44.7% of all GPU time and ~180x off its own memory-bound floor** (a 2-CTA grid at batch 1), fixed by an occupancy-fill of the split-KV count for **+69.5%/+53.3%/+32.0%/+19.5%** at c1/c2/c4/c8 — an order of magnitude more than the planned work would have been worth. The planned decode CUDA-graph sibling WAS built and is worth only ~+2%: this model's decode is GPU-bound, not host-bound |
| **W10** ✅ **DONE 2026-07-22 — the campaign's W-plan is COMPLETE** | **Blocked-row honesty pass.** Config/registry resolution + weight-map + unit parity at V3 dimensions (§5.2) for V3/V3.2; record `HW-BLOCKED` where e2e cannot run. Kimi-Linear and MiniMax-M2 assessments per §11 stay separate rows. | W8 | config/loader-slice/unit gates only — **RESULT: RECORDS ONLY, no code, no build, no GPU work, nothing downloaded, no number claimed.** Two campaign rows move `SPIKE` -> `BLOCKED` (V3/V3.2 HW+DEP; MiniMax-M2 HW+disjoint), one cross-claim row moves `SPIKE` -> `BLOCKED` (GLM-5, HW+DEP), and the two rows that are neither blocked nor supported (`DeepseekForCausalLM`, Kimi-Linear) keep `SPIKE` with their records repaired. The DeepSeek-V2 row stays `ACTIVE`. **The block is NOT closeable** and nothing is archived — see `## W10` below. **Deviation from the plan cell: the config/loader-slice/unit gates for V3 were NOT newly built.** The only one that was ever cheap enough to be honest — unit parity of the `noaux_tc` router at V3's real dimensions — ALREADY EXISTS from W3 (`tests/vt/test_ops_moe_router_grouped.cpp` at 256 experts / `n_group=8` / `topk_group=4` / sigmoid / bias) and the `q_lora` branch is likewise already unit-gated at V3 dims from W6; adding a config-resolution test for an architecture whose config parse deliberately REFUSES it by name would assert our own refusal, not upstream parity, and a weight-map slice test needs a 642 GiB checkpoint's shard we have neither downloaded nor can justify downloading. What CAN still be gated is therefore stated in the row as a named, unbuilt option rather than silently claimed as coverage |

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
Per §8. SACRED correctness on **DeepSeek-V2-Lite bf16**. The W0 K=5 probe
SETTLED the form: at **batch=1 (the gate regime) vLLM's own greedy is
DETERMINISTIC 8/8**, so the **STRICT** where-well-posed gate applies (our token
within 0.5 nats of vLLM's teacher-forced argmax, strict where equal, 0
forward-divergent) — NOT the distributional form. The same battery run as one
BATCH looks non-deterministic 3/8; that is a batching artifact and every
determinism claim here must be made at batch=1. Every W-step: dgx CUDA `-Werror` 0-warn on a clean
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

**W0 — LANDED 2026-07-21. Every flagged source read CONFIRMED by observation;
nothing contradicted.** Evidence root on dgx: `~/scratch_mla_w1/w0_probe.{py,log,json}`.

*(a) Checkpoint.* `deepseek-ai/DeepSeek-V2-Lite` fetched to dgx
(`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite`, snapshot
`604d5664dddd88a0433dbae533b7fe9472482de0`, 4 safetensors shards, **30 GB on
disk**; root went 94% -> 95%, 207 GiB free). It loads in the vLLM 0.25.0 oracle
(`~/venvs/vllm-oracle`) as `DeepseekV2ForCausalLM`, bf16, V2 Model Runner.
**Operational note:** the oracle needs `ninja` on `PATH` for the FlashInfer JIT —
`PATH=$HOME/venvs/vllm-oracle/bin:$PATH`, else engine-core init dies with
`FileNotFoundError: 'ninja'` during `determine_available_memory`.

*(b) BACKEND SELECTION — OBSERVED, matches the source read exactly.* From the
oracle's own `VLLM_LOGGING_LEVEL=DEBUG` startup on sm_121, reproduced on two
independent runs:
```
INFO [platforms/cuda.py:476] Using TRITON_MLA attention backend
                             out of potential backends: ['TRITON_MLA'].
INFO [v1/attention/.../prefill/selector.py:174] Using FLASH_ATTN MLA prefill backend.
```
This confirms §0.2 and §0.3 as OBSERVATION, not inference: dense-MLA decode is
`TRITON_MLA` and MLA prefill is `FLASH_ATTN` on GB10, so the **entire
sm90/sm100-only MLA kernel class (FlashMLA, CUTLASS MLA, FlashInfer MLA,
TokenSpeed MLA) stays out of scope** and W4's two-stage split-KV decode plan
stands. Note the candidate list printed is `['TRITON_MLA']` — a SINGLE entry:
`FLASHINFER_MLA_SPARSE_SM120` is filtered out before selection on
`is_sparse()` vs `use_sparse=False`, exactly as §2.3 predicted.
**Bonus observation for W9 (not a W0 deliverable):** the MoE backend resolves to
`FlashInfer CUTLASS Unquantized MoE` out of
`['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']` — i.e.
the DeepSeek MoE denominator is NOT vLLM's Triton `fused_moe`. The §3 claim that
`vt::MoeGroupedGemmBf16` runs "~1.2x vLLM's Triton fused_moe rate" therefore does
NOT transfer to this model's speed bar and must be re-measured at W9.

*(c) REAL CONFIG — every §5.1 V2-Lite number CONFIRMED against the shipped
`config.json`, none corrected.* `hidden_size=2048`, `num_attention_heads=16`,
`num_hidden_layers=27`, **`kv_lora_rank=512`**, **`qk_nope_head_dim=128`**,
**`qk_rope_head_dim=64`** (so the latent is **512+64 = 576** wide, matching V3),
**`v_head_dim=128`**, **`q_lora_rank=null`**, `n_routed_experts=64`,
`n_shared_experts=2`, `num_experts_per_tok=6`, **`n_group=1`,
`topk_group=1`**, **`scoring_func="softmax"`, `topk_method="greedy"`**,
`routed_scaling_factor=1.0`, `first_k_dense_replace=1`, `moe_intermediate_size=1408`,
`intermediate_size=10944`, `vocab_size=102400`, `rope_theta=10000`,
`max_position_embeddings=163840`, `torch_dtype=bfloat16`.
`rope_scaling = {type: yarn, factor: 40, mscale: 0.707, mscale_all_dim: 0.707,
beta_fast: 32, beta_slow: 1, original_max_position_embeddings: 4096}`.
Code-side constants re-read at the pin and confirmed: `is_neox_style=False`
(`deepseek_v2.py:1059-1064`, rope built over `qk_rope_head_dim` only) and the
**mscale² softmax-scale correction** (`:1071-1074`
`self.scaling = self.scaling * mscale * mscale`) — for this checkpoint
`yarn_get_mscale(40, 0.707) = 0.1*0.707*ln(40)+1 ≈ 1.2608`, so the effective
scale is `192**-0.5 * 1.5896`. **Concrete MLA page size for the W1 spec:
`16 * 1 * 576 * 2 = 18,432 B` per block per layer.**

*(d) COVERAGE GAPS — both §5.1 gaps CONFIRMED against the real config, neither
disappeared.* `q_lora_rank` really is `null`, so the gate vehicle never
exercises `fused_qkv_a_proj` / `q_a_layernorm` / `q_b_proj` nor the
`packed_modules_mapping` un-fusing (`deepseek_v2.py:1812-1820`). And
`n_group == topk_group == 1` with `softmax`/`greedy` means there is **no
`e_score_correction_bias` parameter in the checkpoint at all**, so the whole
`noaux_tc` router — sigmoid scoring, biased-select/unbiased-weight asymmetry,
two-level group mask — is **unit-gated only**. Both remain stated in the rows.

*(e) DETERMINISM PROBE — the W8 gate is **STRICT token-exact**, and the batching
trap was hit and avoided.* A first K=5 greedy probe over an 8-prompt battery
generated all 8 prompts in ONE batch and reported vLLM self-inconsistent on
**3/8** prompts (2 distinct outputs each), which per
[[near-tie-distributional-gate]] would have forced the weaker distributional
form. That is the SAME artifact the Qwen3-dense razor hit: batched generation
re-orders reductions, and the SACRED gate runs **one prompt at a time**.
Re-probed at **batch=1, K=5 — the actual gate regime — vLLM's own greedy is
DETERMINISTIC on 8/8 prompts** (`ALL_PROMPTS_DETERMINISTIC: True`).
**=> The W8 SACRED gate is the STRICT token-exact form** (strict-where-
well-posed, per-prompt batch=1), NOT the distributional one. Recorded here so W8
cannot quietly loosen it. Any future determinism claim on this model MUST be
made at batch=1 or it is measuring the wrong thing.
Evidence: `~/scratch_mla_w1/w0_probe{,_b1}.{py,log,json}` on dgx.

*(f) BONUS OBSERVATION — the W4 target kernels were seen EXECUTING, not just
selected.* vLLM's JIT monitor named them during generation:
`_fwd_grouped_kernel_stage1` and `_fwd_kernel_stage2` — the **two-stage split-KV
flash-decode** of `vllm/v1/attention/ops/triton_decode_attention.py`
(vLLM <- SGLang <- lightllm). This is direct runtime confirmation of the §2.3
finding that the decode kernel is structurally the same split-partial +
deterministic-combine shape as our existing FA-2 split-KV decode, i.e. W4's
single most important reuse premise is now observed rather than read.

*(g) Engine-resolved KV geometry, straight from the oracle's own config:*
`use_mla=True`, `get_head_size()=576`, `get_num_kv_heads()=1`, `block_size=16` —
so upstream's page is `16 * 1 * 576 * 2 = 18,432 B`, the exact number the W1
`MLAAttentionSpec` unit test asserts.

**W1 — LANDED 2026-07-21 (behaviour-preserving, ZERO MLA math).** Allocation and
cache shape now derive from the **KV spec** instead of hardcoded `2 * block *
Hkv * Dh` arithmetic and HF-config reconstruction:
- `MLAAttentionSpec : FullAttentionSpec` (`include/vllm/v1/kv_cache_interface.h`,
  math in `src/vllm/v1/kv_cache_interface.cpp`) overriding
  `real_page_size_bytes()` to upstream's single-tensor formula
  `storage_block_size * num_kv_heads * head_size * dtype_size`
  (`vllm/v1/kv_cache_interface.py:397-398` — **no factor 2, no separate V**),
  with `KVCacheSpecKind::kMlaAttention`. The out-of-scope `fp8_ds_mla` / int4
  layouts (`:381-390`) throw rather than mis-size.
- Registered against the **ORDINARY** full-attention manager with
  `uniform_type_base_spec = FullAttentionSpec`
  (`src/vllm/v1/kv_cache_spec_registry.cpp`), mirroring
  `vllm/v1/core/single_type_kv_cache_manager.py:1539`. This is the spike's key
  finding made executable: because upstream maps MLA onto
  `FullAttentionManager` and `vllm/v1/worker/gpu_model_runner.py` has **no
  `use_mla` branch at all** (only an import at `:58`, an `isinstance` at `:977`,
  comments at `:1085`/`:7133`), block manager, prefix caching and eviction need
  NO change — the cost concentrates in the allocator and the ops.
- `src/vllm/v1/worker/gpu/runner.cpp`: the attention buffer is now
  `num_blocks * spec->page_size_bytes()`, and `block_size` / `num_kv_heads` /
  `head_size` / `dtype` for the `PagedKvCache` view all come from the spec
  instead of `config_.num_key_value_heads` / `config_.head_dim`. An
  asymmetric-`head_size_v` full-attention spec is REFUSED (the single-`head_size`
  view cannot express it) rather than silently mis-viewed.
- The paged-KV storage dtype moved into the spec, resolved once by
  `include/vllm/v1/kv_cache_dtype.h::ResolveKvCacheDType()` (our `VT_KV_CACHE_F32`
  A/B, default bf16 — unchanged semantics), and every producer (the three model
  KV-cache factories + the runner tests) builds its spec with it.
- **Proof the new path is EXERCISED, not merely compiled** (the W7 decode-graph
  lesson): `GPUModelRunner::fa_page_size_bytes()` reports the byte cost the
  allocator actually used, sourced from the spec, plus an opt-in
  `VT_KV_ALLOC_LOG=1` stderr line. The new `test_runner` case asserts (i) the
  default spec reproduces the pre-refactor arithmetic EXACTLY and (ii) a
  `page_size_padded` spec — a value **no HF-config formula can produce** — is
  honoured, which is only possible if the allocator asked the spec. Observed:
  `[kv-alloc] source=spec ... page_size_bytes=512` then `...=1024`.
- **Behaviour-preservation PROVEN on GB10**, all four gates byte-identical on a
  clean `-Werror` build (0 warnings, 0 errors): 27B `235/235`, 35B `315/315`,
  Qwen3-Coder `6/6` (STRICT 5/6, near-tie-band 1/6, 0 forward-divergent —
  identical to the pre-change record), Qwen3-dense `16/16` (STRICT 11/16,
  max gap 0.25 nats, 0 forward-divergent — identical). `test_runner` 15/15,
  `test_kv_cache_interface` 21/21 (incl. 4 new MLA-spec cases),
  `test_llm_engine` 5/5.

**W2 — LANDED 2026-07-21 (behaviour-preserving for every dense model).** The MLA
branch of `_get_backend_priorities` that `src/vllm/platforms/cuda.cpp:54-56`
deliberately skipped is now ported, and it is DATA:
- `include/vllm/platforms/cuda_attn_priority.h` (NEW) holds the whole of
  `vllm/platforms/cuda.py:84-176` as a TABLE — one row per upstream
  `if device_capability.major == N` arm, keyed on `(use_mla, major)`, with
  `kAnyMajor` for the `else` arm. **BOTH** branches are ported, not just ours:
  MLA sm_100 (`:117-131`, including the `:96-115` adaptive sparse tail — the
  FlashInfer-vs-FlashMLA swap on fp8 KV or `num_heads <= 16`), MLA sm_12x
  (`:129-133`), MLA `else` (`:134-142`), and the two pre-existing non-MLA arms
  moved verbatim from the old inline if-chain. A new arch is a ROW, not a code
  path. The table lives in a header (not the CUDA-only TU) so the CPU test tier
  asserts the REAL table — the pre-W2 `FakeCudaPlatform` was a hand-copied
  duplicate carrying a "keep in sync" comment, and that duplicate is now gone.
- `AttnSelectorConfig` (`include/vllm/platforms/interface.h`) carries upstream's
  `use_mla` / `num_heads` / `kv_cache_dtype` inputs plus `use_sparse`. Every
  field defaults to the pre-W2 dense answer, so the zero-argument call sites are
  unchanged.
- **The sparse/DSA SEAM.** GB10's MLA list keeps BOTH upstream entries
  (`[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`). The sparse one is eliminated not
  by omission but by a real FILTER: `AttentionBackend::is_mla()` / `is_sparse()`
  (mirroring `vllm/v1/attention/backend.py:307-360 validate_configuration`) are
  compared against the request in `SelectAttentionBackendName`. That is exactly
  why the oracle's candidate list printed a SINGLE entry at W0. A future DSA
  backend therefore needs **no edit to the table and no edit to the selector** —
  it registers with `is_sparse() == true` and a `use_sparse=true` request selects
  it. This is unit-proven, not asserted: the test registers a stand-in sparse
  backend and shows the dense request still gets `TRITON_MLA` while a sparse
  request gets the sparse entry. (The concurrent `CLAIM-GLM-DSA-LATEST-DEEPSEEK`
  spike scopes DSA itself; none of it is implemented here.)
- `TritonMLABackend` (`include/vllm/v1/attention/backend.h`,
  `src/vllm/v1/attention/backend.cpp`) — the NAME + the selection-relevant
  surface only, ported from `triton_mla.py:81` over `mla_attention.py:1206`. Its
  `get_kv_cache_shape` returns upstream's **3-D** `(num_blocks, block_size,
  head_size)` (`mla_attention.py:1216-1224`) with no K/V axis, and REFUSES
  `num_kv_heads != 1` rather than ignoring it as upstream does. `get_impl_cls()`
  stays the base `nullptr` — the impl is W4/W6.
- MLA PREFILL priority (`get_mla_prefill_backend_priority`,
  `mla/prefill/selector.py:47-76`): sm_100 gets the four-entry list, everything
  else — including GB10 — gets `[FLASH_ATTN]` ALONE, matching the W0 observation.
  The ROCm arm is recorded as not-ported rather than dropped.

**W3 — LANDED 2026-07-21. Two new numerics, both CPU-reference-gated.**

*(a) `vt::ConcatAndCacheMla`* — `OpId::kConcatAndCacheMla`, validation in
`src/vt/ops.cpp`, CPU reference in `src/vt/cpu/cpu_cache.cpp`, CUDA kernel in
`src/vt/cuda/cuda_cache.cu`. Ported 1:1 from
`vllm/csrc/libtorch_stable/cache_kernels.cu:401-442`
(`concat_and_cache_mla_kernel`) + its host wrapper `:842-905`.
**WHAT ACTUALLY RUNS UPSTREAM — verified, not assumed** (AGENTS.md "ground every
check in the whole execution chain"): unlike the GEMM/attention families, this op
is NOT delegated to a dependency. `vllm/_custom_ops.py:2532` binds straight to
`torch.ops._C_cache_ops.concat_and_cache_mla` (`:2540-2542`), registered from
vLLM's OWN `csrc/libtorch_stable/torch_bindings.cpp` over that kernel. No
flashinfer / cutlass / TRT-LLM variant exists in the dense-bf16 path; the only
sibling in that TU is `concat_and_cache_ds_mla_kernel` (`:445+`), the fp8_ds_mla
656-byte V3.2 layout, which is out of campaign scope and which our wrapper
REFUSES via its dtype check rather than mis-sizing. The one thing that can
displace it is the COMPILE-TIME pass
`compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:40` folding RoPE into
`concat_and_cache_mla_rope_fused` — same math, one launch; our fusion-catalog
analogue stays deferred to W9 exactly as §10 sequences it.
Semantics: the latent (`kv_lora_rank`) and the decoupled rope part
(`qk_rope_head_dim`) are CONCATENATED into ONE 576-wide entry — this is the write
`vt::ReshapeAndCache`'s `(k, v, k_cache, v_cache)` signature cannot express.
Indexing is STRIDE-driven on every operand, so (i) a per-layer slice of a
multi-layer allocation and (ii) the two column halves of the single
`kv_a_proj_with_mqa` output (`deepseek_v2.py:511`) both work with no copy — both
exercised in the tests.

*(b) grouped-topk (`noaux_tc`) router* — `MoeRouterTopKArgs` EXTENDED additively
with `scoring_func` (new `MoeScoringFunc` enum), `num_expert_group`,
`topk_group`, `routed_scaling_factor`, plus an optional trailing
`e_score_correction_bias` tensor argument on `vt::MoeRouterTopK` (mirroring how
`MoeCombine` takes its optional `shared`). Ported 1:1 from
`fused_moe/router/grouped_topk_router.py:106-161` (the `forward_native` path; the
`ops.grouped_topk` CUDA path at `:28-70` is an optimization of the SAME formula,
gated on an env flag + sigmoid + a bias, not a different result).
**The existing router is untouched**: `num_expert_group == 0` (the default)
dispatches to the ORIGINAL kernel on both devices — a separate
`MoeRouterGroupedTopKKernel` implements the new path — so the 27B / 35B / Coder /
Qwen3-dense routers are byte-identical by construction, not by measurement.
Recorded DETERMINISM DEVIATION: upstream uses `torch.topk`, whose tie order is
unspecified unless `VLLM_BATCH_INVARIANT` forces `sorted=True`; we keep the house
lowest-index-wins rule for both the group and the expert selection, which is what
makes CPU and CUDA agree bit-for-bit.
**The `noaux_tc` correctness evidence is UNIT-ONLY, and this is stated plainly
rather than papered over.** DeepSeek-V2-Lite — the campaign's only e2e vehicle —
has `n_group=topk_group=1`, softmax/greedy and therefore **no
`e_score_correction_bias` parameter at all** (§5.1, W0-confirmed), so the e2e gate
exercises NONE of this: not sigmoid scoring, not the biased-select /
unbiased-weight asymmetry, not the two-level group mask, not routed scaling.
`tests/vt/test_ops_moe_router_grouped.cpp` is the whole of the evidence, and it
runs at DeepSeek-V3's REAL dimensions (256 experts, `n_group=8`,
`topk_group=4`, `top_k=8`, sigmoid, `routed_scaling_factor=2.5`, WITH the bias)
against an INDEPENDENT transcription of the upstream formula that uses a
sort-based top-k rather than the kernel's greedy scan — so agreement is evidence,
not a restatement of the same code. Each new rule is additionally isolated in its
own hand-computed case: the bias selects but the unbiased score weights; the
group score is top-2-SUM with a bias and MAX without (a case where the two rules
disagree about which group survives); the mask really excludes the GLOBAL argmax
when it sits in a losing group; renormalize happens BEFORE routed scaling.
**Measured CPU-vs-CUDA agreement, corrected mid-gate rather than assumed:** the
first version of the test asserted the router weights were BIT-identical across
devices on the sigmoid path, reasoning that sigmoid has no cross-expert
reduction. dgx disproved it — 7 of 192 weights differed by exactly one ULP. The
cause is not reduction order but the TRANSCENDENTAL: the CPU reference uses the
host `std::exp` and the kernel uses the device `expf`, separately-rounded
implementations of the same function. The test now asserts the accurate property
(a 1-ULP bound), and the claim that IS exact is the one that matters for routing
— the SELECTION (expert ids), which is pure comparison over the same values and
cannot legitimately differ, plus run-to-run bit-reproducibility on a device.
**COORDINATION:** `.agents/coordination.md` records the
`vt::MoeRouterTopK` extension as SHARED with `CLAIM-GLM-DSA-LATEST-DEEPSEEK`
("must not be implemented twice"). It is landed HERE; that claim consumes it.

**A real bug memcheck caught, recorded because the lesson generalizes.** The
grouped-router CUDA kernel's dynamic shared memory is laid out
`[sel(e) | orig(e) | gscore(G) | gkeep(G)]`, but the launch initially sized it as
`2*e + G` floats — one per-group array short — so every write to the `gkeep` mask
ran off the end of the allocation. `compute-sanitizer memcheck` reported 26
`Invalid __shared__ write` errors. **The unit tests passed both before and after
the fix**, because the stray write landed outside the live data. That is the
argument for memcheck being a per-step gate rather than a formality, and it is
the same shape as the standing "prove the new path RAN, a green gate does not
prove it" rule: a passing test does not prove memory safety either.

**W4 LANDED 2026-07-22** (base `ed2c342`). `vt::MlaDecodeAttention` — the MQA
decode over the compressed latent (QK 576 / V 512, `num_kv_heads == 1`), reading
the paged 3-D cache W3 writes.

*(a) WHAT WAS PORTED, both sides.* `MlaDecodeStage1` <-
`triton_decode_attention.py:278-458 _fwd_grouped_kernel_stage1` (the `IS_MLA`
branch); `MlaDecodeStage2` <- `:575-639 _fwd_kernel_stage2`; the launch pair <-
`:470-573 _decode_grouped_att_m_fwd` + `:642-682 _decode_softmax_reducev_fwd` +
`:719-754 decode_attention_fwd_grouped`; `ComputeNumKvSplits` <-
`triton_mla.py:40-47 _compute_num_kv_splits`; the caller contract <-
`:189-260 forward_mqa`; the split workspace <- `:57-78
_reserve_attn_logits_workspace`, realized as the house grow-only per-stream
scratch with `RetireGraphScratch` (upstream's workspace manager has no C++
twin, and a captured decode graph bakes the pointer). The ACTUAL Triton source
was read, not the Python wrapper — which is how the one non-obvious thing in
the kernel was ported correctly: at `:424-431`, under `IS_MLA` the kernel loads
NO V tile at all, it does `v = tl.trans(k)`. V is the leading 512 columns of the
SAME latent row already loaded as K, so our one shared-memory tile is both the K
and the V tile. Upstream's 512+64 `BLOCK_DMODEL`/`BLOCK_DPE` two-tile split
(`:494-496`) exists only because Triton needs power-of-two tile shapes; the
arithmetic is one 576-wide dot either way, so we load one 576-wide row.

*(b) WHERE OUR FA-2 SPLIT+COMBINE FIT, AND WHERE IT DID NOT — stated plainly
rather than forced.* It FIT at the ALGORITHM level, exactly as W0's observation
predicted: the split-KV schedule (partition `[0, seq_len)` into `num_kv_splits`
contiguous chunks, one normalized partial + its LSE each), the combine algebra
(online-softmax rescale), and our fixed-ascending / never-atomicAdd determinism
rule all transfer unchanged. It did NOT fit at the CODE level and the vendored
launcher is not reused: (1) `cuda_flash_attn_fa2.cu` consumes SEPARATE 4-D
`k_cache`/`v_cache` tensors, while MLA's cache is 3-D with no K/V and no head
axis and K and V are the same bytes; (2) its CUTLASS instantiations are compiled
for symmetric head_dim {128, 256}, and MLA needs the ASYMMETRIC QK 576 / V 512
(576 is not a supported FA-2 head_dim at all); (3) its combine addresses the
output through the FA-2 params struct, which has no notion of a V width
different from the QK width. Generalizing that launcher is tractable for W5's
PREFILL (qk 192 / v 128) and is W5's job. This verdict is recorded in the TU
header so the next reader does not re-litigate it.

*(c) THE EVIDENCE, unit-level and deliberately strong* (there is no e2e model
until W7, so this is the whole spine). `tests/vt/test_ops_mla_attn.cpp` ports
`tests/kernels/attention/test_mla_decode_cpu.py`: its `ref_mla` (`:13-33`)
becomes an INDEPENDENT TWO-PASS softmax oracle — a different algorithm from the
streaming online-softmax BOTH of our impls use, so a bug in the streaming
rescale cannot hide behind a matching reference — and its NaN-padding trick
(`:71-73`, every cache row past `seq_len` poisoned, then "Likely read out of
bounds") is ported verbatim as a real out-of-bounds detector. Its parametrization
is ported too (bs=4, mean_seq_len=256, h_q=16, d=576, dv=512, block_size=16, BOTH
`varlen` arms), and the `test_mla_backends.py` shape sweep is covered by: ragged
lengths straddling every boundary (1/15/16/17/255/256/257/300), multi-block, the
single-block single-token case, EVERY split boundary (`num_kv_splits` in
{1,2,3,4,5,8,16,17,64,300,512} — including splits > seq_len, i.e. the empty-split
path both stages must skip identically), 128-head DeepSeek-V3 geometry (which
exercises head_tiles > 1, and V2-Lite's 16 heads does not), head counts 1/3/17
that do not fill a `BLOCK_H` tile (the `mask_h` path, `:321-323`), a 288/256
block-32 non-V2-Lite geometry, bf16 AND f32, and run-to-run BIT-exactness over 5
runs. The block table is deliberately REVERSE-INTERLEAVED (descending,
non-contiguous pages) so any stride assumption fails; upstream's own test uses a
plain arange and cannot catch that. The `scale` used is the REAL mscale^2-
corrected value, not `1/sqrt(576)`. dgx/sm_121: **11/11 cases, 2,303,193
assertions**; the 7 CUDA cases were confirmed to EXECUTE (2,106,080 assertions
when run alone), not skip, and the output buffer is pre-poisoned with NaN so a
kernel that fails to write an element is caught.

*(d) MEMORY SAFETY, run deliberately per the W3 lesson.* `compute-sanitizer
memcheck` **0 errors** over the CUDA case set, and because the new kernel is a
`__syncthreads`-coordinated shared-memory kernel — a hazard class memcheck does
NOT cover — `racecheck` (**0 hazards**) and `synccheck` (**0 errors**) were run
too. The W3 note that "unit tests passing does not mean the kernel is
memory-safe" is exactly why the shared-memory tile bounds, the `mask_h` early
return placement relative to both `__syncthreads` calls, and the opt-in dynamic
shared-memory ceiling are all checked by tooling and not by inspection.

*(e) THE ENGINE SEAM.* `TritonMLABackend::get_impl_cls()` is no longer
`nullptr`: it returns a real `TritonMLAImpl` (`triton_mla.py:126-128,134`) whose
`forward_mqa` is the 1:1 counterpart of `:189-260`, driven by a new
`MLACommonMetadata` carrying exactly the two device tensors upstream reads
(`:245-246`) plus the host `max_seq_len` that sizes the split heuristic. It
refuses what upstream refuses (`window_size`, `:165-171`) and — importantly — a
prefill-shaped batch throws BY NAME ("MLA prefill is campaign step W5") rather
than silently producing wrong numbers. Recorded deviations: (i) upstream launches
on torch's AMBIENT CUDA stream, which has no C++ twin, so `TritonMLAImpl::queue`
is a settable field defaulting to the (correct but serializing) default stream,
wired properly at W7; (ii) fp8 KV cache (`:390-391`) and `logit_cap` (`:404-405`)
are NOT ported — the former is out of campaign scope and refused loudly by the op
wrapper, the latter is unreachable because `TritonMLAImpl` rejects
`logits_soft_cap`; (iii) `seq_len == 0` writes zeros where upstream's stage 2
would divide by zero — unreachable upstream, and matched exactly by our CPU
reference.

*(f) NOT IN W4, on purpose.* No weight absorption (folding `W_UK` into the query
and un-projecting with `W_UV` is W6 — `forward_mqa` takes the query already in
latent space and returns the output still in latent space, exactly like
upstream). No prefill. No model. No speed number: decode perf is W9, and this
kernel is correctness-graded, not tuned.

**W5 — LANDED 2026-07-22. MLA PREFILL + the CHUNKED-CONTEXT LOOP.** Evidence root
on dgx: `~/w5mla/` (`build.log`, `memcheck_{prefill,chunked}.log`,
`racecheck_{prefill,chunked}.log`, `synccheck_{prefill,chunked}.log`,
`regressions.log`, `ctest.log`).

*(a) WHAT ACTUALLY RUNS UPSTREAM, per the whole-chain rule — and it is THREE
different answers.* The MLA prefill ATTENTION is **not** vLLM's csrc: it is
FlashAttention via `vllm_flash_attn`, selected by
`mla/prefill/selector.py:66-76` (every non-sm100 device gets the single entry
`[FLASH_ATTN]`, and `:191-194` HARD-RAISES if it is unavailable — there is no
fallback below FA on sm_121), which W0 OBSERVED the oracle logging. The chunked
loop's two supporting kernels ARE vLLM's own `libtorch_stable` csrc
(`gather_and_maybe_dequant_cache`, `merge_attn_states`) with no
flashinfer/cutlass/TRT-LLM variant in the dense-bf16 path; their fp8 siblings
(`cp_gather_cache`, `USE_FP8_OUTPUT`) are out of scope and refused.

*(b) The FA-2 LAUNCHER GENERALIZATION W4 left as W5's job — DONE, and the reason
it was tractable is that upstream does not ask FA-2 for asymmetric head dims
either.* `requires_v_padding` is TRUE on GB10 (`flash_attn.py:88-99` exempts only
FA3-on-SM90 and FA4), so upstream ZERO-PADS V from 128 to 192
(`:164-168`) and slices the output back (`:196-197`) — the kernel stays a plain
SYMMETRIC head_dim-192 instantiation. So the generalization is exactly three
things: two new explicit instantiations of the **untouched** generic
`run_mha_fwd_splitkv_dispatch<bf16, 192, {true,false}>`; a new launcher entry
`LaunchMlaPrefillFA2Bf16` for the CONTIGUOUS-varlen mode (`cu_seqlens_k` instead
of `block_table`+`seqused_k`, which the vendored kernel already supports at
`flash_fwd_kernel.h:584-590`); and the pad/slice pair. `LaunchPrefillFA2Bf16` —
the PAGED launcher every non-MLA prefill calls — is not touched: the diff of
`cuda_flash_attn_fa2.cu` is 211 insertions / **0 deletions**, and the vendored
tree gains only two new files. That is the structural byte-identity argument; the
gate is the empirical one (§(e)).

*(c) The chunked-context loop, and the three boundary hazards it hides.* Ported
as `mla_chunked_context.h`: workspace sizing (`:1422-1451`), the chunk grid
(`:1667-1745`, `:1837-1855`), `_compute_prefill_context` (`:2094-2199`) and
`forward_mha` (`:2344-2425`). The hazards, all covered: (i) `max_context_chunk`
is rounded DOWN to a page multiple (`:1687-1690`) because the gather kernel
indexes `(seq_starts[b] + within)/block_size` into the block table, so an
unaligned start silently gathers the WRONG PAGES; (ii) a request whose context is
shorter than the chunk grid contributes ZERO tokens to later chunks
(`:1699 clamp(min=0)`), giving a `-inf` LSE, and merging two `-inf` partials is
0/0 — the case `merge_attn_states.cu:100-106` exists for; (iii) the merge is a
PING-PONG over two buffers (`:2196-2197`), so chunk k's result must not live in
the buffer chunk k+1 writes.

*(d) An upstream FA-2 QUIRK found and worked around, recorded as a deviation.*
With `unpadded_lse = true` the MAIN path writes LSE at the unpadded offset
(`flash_fwd_kernel.h:1038-1041`) but the EMPTY-K EARLY EXIT (`:1030-1043`)
ignores the flag and always writes at the PADDED
`((split*b+bidb)*h+bidh)*seqlen_q` offset. A zero-key request is REACHABLE in the
chunked loop (see (c)(ii)), so that mixed layout would both clobber valid rows
and write past an `[h, total_q]` buffer. We therefore run with
`unpadded_lse = false` into a `[b, h, max_seqlen_q]` scratch and convert, which
also normalizes FA-2's `+INFINITY` empty-row LSE to `-inf` — the value
`merge_attn_states.cu:97-98` normalizes it to anyway.

*(e) Evidence — unit-level, held to the W4 bar.* `test_ops_mla_prefill.cpp`
**4/4 cases / 2,377,052 assertions**; `test_ops_mla_chunked_context.cpp`
**5/5 / 306,037 assertions**; both on dgx sm_121 at the REAL V2-Lite prefill
geometry (QK 192 = 128 nope + 64 rope, V 128, latent 576, block 16, mscale²
scale). The oracle is INDEPENDENT: a double-precision TWO-PASS softmax, a
different algorithm from FlashAttention's streaming online-softmax. The chunked
loop is additionally gated against a SINGLE-SHOT whole-sequence oracle that never
chunks — exact / +1 / −1 chunk boundaries, a request with no context at all,
ragged multi-chunk, V3's 128 heads. ADVERSARIAL reverse-interleaved block tables
everywhere (upstream's own `arange` tables cannot catch a page-stride
assumption). NaN-poisoned outputs so a kernel that fails to write fails the gate.
Run-to-run BIT-exact over 5 runs. `compute-sanitizer` memcheck **0 errors**,
racecheck **0 hazards**, synccheck **0 errors** on BOTH binaries. Clean dgx CUDA
build **0 warnings / 0 errors**. Regression set UNCHANGED: 27B 235/235, 35B
315/315, Qwen3-Coder 138/138, Qwen3-dense 664/664, OPT 36/36.

*(f) NOT IN W5, on purpose.* The `kv_b_proj` up-projection is a CALLBACK, not
inlined — it is a model weight, i.e. W6. No weight absorption, no model, no
forward. DCP/context-parallel (`_context_parallel_compute_prefill_context`,
`reorg_kvcache`) is single-GPU-out-of-scope; the fp8-prefill arms need
device-capability family 100 (`mla_attention.py:1382-1385`) and are unreachable
on sm_121. No speed number: prefill perf is W9.

**W6 — LANDED 2026-07-22. THE MLA ATTENTION BLOCK + WEIGHT ABSORPTION.** Evidence
root on dgx: `~/w6mla/` (`build.log`, `memcheck_*.log`, `racecheck_*.log`,
`synccheck_*.log`, `regressions.log`, `ctest.log`). The attention family is no
longer a pile of ops: W3's cache write, W4's MQA decode and W5's MHA prefill are
now composed by ONE layer.

*(a) WHAT WAS PORTED, both sides.* The block is
[`mla_attention.h`](../../include/vllm/model_executor/models/mla_attention.h) +
[`mla_attention.cpp`](../../src/vllm/model_executor/layers/attention/mla_attention.cpp):
`AbsorbKvBProjBf16` <- `mla_attention.py:875-962 process_weights_after_loading`
(the split at `:892-900`, the two permutes at `:959-962`); `MakeMlaUpProjectFn`
<- `:2141-2170` — **the `kv_b_proj` callback W5 deliberately left open, now
wired**; `ForwardMlaAttentionBlock` <- `mla.py:119-181`
(`MultiHeadLatentAttentionWrapper.forward` — the projections, both query
branches, the two RMSNorms and the decoupled RoPE) over `mla_attention.py:553-620`
(`forward`, which fixes the cache-update-BEFORE-attention ORDER), `:624-874`
(`forward_impl`, the dispatch and the absorbed decode) and `:2344-2425`
(`forward_mha`); `BuildDeepseekRopeCosSinCache` <- `deepseek_scaling_rope.py:76-118`
over `rotary_embedding/common.py:34-70`; `YarnGetMscale` <- `:20-23`;
`MlaAttentionScale` <- `deepseek_v2.py:995` + `:1067-1075`.

*(b) ABSORPTION NEEDED NO NEW ATTENTION KERNEL — §2.2's most useful prediction
held.* It is a LOAD-TIME weight transform plus TWO BATCHED GEMMs, so the whole
new-kernel surface is two general primitives:
- **`vt::BatchedMatmul`** <- `torch.bmm` at `mla_attention.py:789`
  (`torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)`, the q-side W_UK fold)
  and `:1034` (`_v_up_proj`'s W_UV un-projection). **Whole-chain check:** on CUDA
  `torch.bmm` over bf16 resolves to ATen's `baddbmm_out_cuda_impl` -> cuBLAS
  `gemmStridedBatchedEx` (CUDA_R_16BF, CUBLAS_COMPUTE_32F); our CUDA impl is the
  cuBLASLt strided-batched form of that same GEMM on the SAME handle/workspace as
  `vt::Matmul`. The only alternatives upstream has are the ROCm-only aiter
  fp8/fp4 bmm branches (`:766-776`, `:1024-1032`), unreachable on CUDA. It is
  STRIDE-DRIVEN because BOTH upstream call sites pass `.transpose(0,1)` views.
- **`vt::ConcatMlaNopeRope`** <- `ConcatMLAQKernel`
  (`csrc/libtorch_stable/concat_mla_q.cuh`) + host wrapper
  `cache_kernels.cu:1555-1600` (bound at `torch_bindings.cpp:841,905`, reached
  from `_custom_ops.py:2696-2708`), GENERALIZED to arbitrary nope/rope widths and
  a head-BROADCAST rope operand so the SAME op also serves upstream's OTHER
  head-concat site, `_concat_k_nope_k_pe` (`mla_attention.py:2063-2092`).

*(c) THE ABSORBED-vs-UNABSORBED EQUIVALENCE, PROVEN NUMERICALLY THREE WAYS.* This
is the heart of W6, and it is measured rather than argued
([`test_mla_attention_block.cpp`](../../tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp)):
1. **The identity itself.** An INDEPENDENT double-precision block oracle computes
   the attention BOTH ways from the same weights — unabsorbed (materialize
   `k_nope = W_UK^T kv_c` and `v = W_UV^T kv_c`, then MHA at QK 192 / V 128) and
   absorbed (fold W_UK into the query, MQA against the raw 576-wide latent, then
   un-project with W_UV) — and they agree to **< 1e-11 relative**, at BOTH query
   branches. That is `q_nope.(W_UK^T kv_c) == (q_nope W_UK).kv_c` and
   `sum_s p_s (W_UV^T kv_c_s) == W_UV^T (sum_s p_s kv_c_s)` made executable.
2. **Ours vs the UNABSORBED oracle.** Our block's absorbed decode reproduces the
   double oracle to **< 2e-4 relative in f32** across decode-only, prefill-only,
   chunked-prefill-with-context and MIXED batches (**< 3e-2 in bf16**, where bf16
   storage of every intermediate over ~2k-wide reductions is the binding limit —
   the f32 arm is what pins the MATH).
3. **Ours vs ours, THROUGH TWO GENUINELY DIFFERENT CODE PATHS.** The SAME batch —
   every request one new token over cached context, so it is legally either shape
   — is run once labelled all-DECODE (our `vt::MlaDecodeAttention` MQA kernel,
   QK 576 / V 512, one KV head, K/V never materialized, W_UK folded into the
   query and W_UV un-projecting the output) and once labelled all-PREFILL (our
   `vt::MlaPrefillAttention` MHA at QK 192 / V 128 over K/V materialized by
   `kv_b_proj`, plus the chunked-context loop over the cached latent). They agree
   to **< 3e-4 (CPU f32)** and **< 4e-2 (CUDA bf16)**. Nothing but the weights is
   shared between those paths, so an absorption bug cannot cancel out — this is
   the strongest of the three, and it is the one that would catch a transposed
   permute that happened to be self-consistent.
`AbsorbKvBProjBf16` is additionally gated on its own against an INDEPENDENT
transcription that performs `.T -> view -> split -> permute` literally, step by
step, rather than the folded index arithmetic production uses.

*(d) BOTH QUERY BRANCHES — and the one that has NO e2e coverage, said plainly.*
With `q_lora_rank` set the path is `fused_qkv_a_proj` (q_a FUSED with kv_a — there
is no standalone `q_a_proj` module upstream, `deepseek_v2.py:905-950`, and
`:1812-1820` registers the `packed_modules_mapping` un-fusing) ->
`q_a_layernorm` -> `q_b_proj`; without it, a direct `q_proj`
(`:1010-1016`, `:1028-1034`). Both are implemented and both are gated at REAL
dimensions — the non-lora branch at DeepSeek-V2-Lite's geometry and the lora
branch at DeepSeek-V3's (hidden 7168, 128 heads, `q_lora_rank=1536`). **The lora
branch has NO END-TO-END COVERAGE and cannot get any on this hardware**:
DeepSeek-V2-Lite, the campaign's only e2e vehicle, has `q_lora_rank=null`
(W0-confirmed against the shipped `config.json`), so W8's gate will exercise none
of it. It is UNIT-GATED ONLY, exactly like the `noaux_tc` router W3 landed.
GLM-4.7-Flash (58.2 GiB, `q_lora_rank=768`) is the checkpoint that would close it
e2e and it FITS GB10 — noted for a later step, deliberately NOT attempted here
(`CLAIM-GLM-DSA-LATEST-DEEPSEEK` scopes that family).

*(e) DISPATCH, mirrored exactly.* `num_mqa_tokens = attn_metadata.num_decode_tokens`
and the rest are MHA (`mla_attention.py:700-709`), with DECODE TOKENS PACKED
FIRST, so MHA takes the tail `q[num_mqa_tokens:]` (`:722-737`) and MQA the head
`q[:num_mqa_tokens]` (`:739+`); the MHA call is issued FIRST, as upstream does.
Upstream's own note at `:19-22` that this split is a tunable heuristic rather
than an invariant is recorded in the header rather than silently hardened. The
MIXED case is gated (2 decode + 2 prefill, decode first, ragged query lengths,
one prefill with context and one without), as are decode-only, prefill-only-with-
no-context (the `:2421-2425` path that skips the chunk loop entirely) and
chunked prefill over multiple chunks. **One upstream ORDERING INVARIANT was found
the hard way and is now documented:** within the prefill tail the WITH-CONTEXT
requests must come first, because `prefill_tokens_with_context` is
`query_start_loc[num_prefills_with_context]` (`:1806-1810`) and every query row
past it takes the suffix result verbatim. A test batch that violated it produced
a wrong answer (0.86 relative error) — the gate caught it, and W7 must build its
batches accordingly.

*(f) DECOUPLED RoPE + the mscale^2 scale.* Only the `qk_rope_head_dim` (64) slice
rotates, `is_neox_style=False` (adjacent-pair GPT-J), applied to
`q[..., qk_nope_head_dim:]` and to the SINGLE shared `k_pe` head
(`mla.py:158-167`, `deepseek_v2.py:1053-1064`). The YaRN cos|sin cache carries the
ROTATION mscale `yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim) *
attn_factor` (`deepseek_scaling_rope.py:55-59`), while the SOFTMAX scale carries
the SQUARE of `yarn_get_mscale(f, mscale_all_dim)` (`deepseek_v2.py:1067-1075`).
Conflating those two mscales is the classic DeepSeek RoPE bug; they are separate
functions here and each is gated against an independent transcription of its
formula.

*(g) THE EVIDENCE.* On dgx sm_121:
[`test_mla_attention_block.cpp`](../../tests/vllm/model_executor/layers/attention/test_mla_attention_block.cpp)
**10/10 cases / 2,372,644 assertions** and
[`test_ops_mla_absorb.cpp`](../../tests/vt/test_ops_mla_absorb.cpp) **9/9 /
1,644,807 assertions**. The CUDA cases were confirmed to EXECUTE rather than skip
(2 cases / 124,941 assertions and 3 / 290,835 when run alone). Every output buffer
is NaN-POISONED before the call, so a path that fails to write a token fails the
gate. Run-to-run BIT-exactness over repeated runs on both devices. Full dgx `ctest`
**181/182**, the single failure being the DOCUMENTED `test_capi` flake — verified
by signature rather than assumed: its `:353` case fails on
`M-oM-?M-=oollllll` vs `loollllll` (a U+FFFD partial-UTF-8 first byte, exactly the
recorded signature) and its `:453` streaming case fails alongside it, and BOTH
appear identically in the W5 baseline log (`~/w5mla/ctest.log`, "1 tests failed out
of 180"), so nothing about it moved. Upstream test
modules ported: `tests/kernels/test_concat_mla_q.py` (BOTH arms, including
`test_concat_mla_q_transposed_nope:37-63`, whose docstring exists precisely because
the real nope operand is the non-contiguous transposed bmm output — compared
EXACTLY, `atol=0, rtol=0`, since a concat is a pure copy), plus the MLA-geometry
sweep of `tests/v1/attention/test_mla_backends.py` and the two-pass-oracle
discipline of `tests/kernels/attention/test_mla_decode_cpu.py`.

*(h) MEMORY SAFETY — and a compute-sanitizer LIMITATION worth recording.*
`compute-sanitizer` memcheck **0 errors** and racecheck **0 hazards** on both new
binaries. synccheck initially reported `1 error` on the block binary, and the
cause was NOT a barrier defect: the tool printed
`Warning: Detected overflow of tracked cuda::barrier structures. Results might be
incorrect. Try using --num-cuda-barriers to fix the issue`, after which the run
died with `unspecified launch failure`. The block drives far more distinct kernels
per process than any previous MLA test (cuBLASLt dense + strided-batched, the
vendored FA-2, the MLA decode pair, the cache write, the gather/merge), which
overflows synccheck's default barrier table and corrupts its own instrumentation.
With `--num-cuda-barriers 65536` synccheck reports **0 errors**. Recorded because
the failure mode is indistinguishable from a real defect at first glance, and
because every future test that composes this many kernel families will hit it.
**A second operational lesson from the same run:** `ctest -j4` on dgx is
OOM-KILLED partway through the heavy tail — it schedules two ~30 GiB paged-engine
model tests concurrently against 119 GiB of UNIFIED memory, and the whole ctest
process disappears with a truncated log that looks like a hang rather than a
failure. `-j2` completes. Recorded next to the disk-space hazard already in the
operational notes, because the symptom (a log that simply stops) invites the wrong
diagnosis.

*(i) DEVIATIONS, recorded not glossed.*
1. **The A-projections are issued per weight ROW-SLICE, not as one fused GEMM.**
   Upstream runs one GEMM per A-projection module and `.split(...)`s the result
   into views; we slice the single fused WEIGHT's output rows and issue one GEMM
   per slice, so every downstream consumer gets a CONTIGUOUS buffer. The reason is
   concrete: `vt::RmsNorm` requires contiguous inputs, and relaxing the hottest op
   in every existing model to save one MLA copy is the wrong trade. The checkpoint
   PACKING is unchanged — `fused_qkv_a_proj` remains ONE weight, exactly as
   `packed_modules_mapping` (`deepseek_v2.py:1812-1820`) requires — so only the
   launch granularity differs, and that is the same trade the dense block already
   makes BY DEFAULT (`dense_attn_block.h`'s 3-shard qkv path,
   `VT_QWEN3_QKV_MERGE` default OFF). A truly fused A-GEMM is a W9 A/B.
2. **`vt::ConcatMlaNopeRope` is SCALAR and width-generic** where upstream's is
   128/256-bit vectorized and instantiated only for NOPE_DIM=512 / rope 64
   (`concat_mla_q.cuh:13,21-24,50-53`). A concat is a pure copy, so the bytes
   written are identical; the generality is what lets one op serve the prefill K
   concat (nope 128) too. Vectorization is W9 — the same disposition as W5's
   scalar `MergeAttnStates`.
3. **Two ADDITIVE relaxations of existing ops**, both integer-identical for
   contiguous tensors and therefore bit-identical for every existing model BY
   CONSTRUCTION (and confirmed BY GATE): `vt::RopeFromCache` is now stride-driven
   on q/k (DeepSeek rotates the TRAILING slice of each query head and its `k_pe`
   is a column block of the fused kv_a projection — both strided views), and
   `vt::MatmulBT` accepts a ROW-STRIDED activation (upstream applies `kv_b_proj`
   to a 512-column slice of the 576-wide chunked-prefill workspace,
   `mla_attention.py:2160`, and `F.linear` takes exactly that).
4. **fp8 output quant, DCP/context-parallel and the sparse/indexer branches are
   NOT ported** — out of campaign scope or unreachable on sm_121, and refused
   loudly rather than mis-computed.

*(j) NOT IN W6, on purpose.* No model, no registry entry, no loader, no config
parse, no forward over layers — that is W7. No CUDA-graph sibling and no speed
number: W9 owns tuning, and the A-projection granularity above is explicitly left
as one of its A/Bs. **NO MODEL ROW MOVES OFF `SPIKE`.**

**W7 — LANDED 2026-07-22. THE DEEPSEEK-V2 MODEL: REGISTRY + CONFIG PARSE +
LOADER + FORWARD.** Evidence root on dgx: `~/w7mla_src` (build + gates).
The first MLA MODEL in this tree. It composes everything W1-W6 built into
something a checkpoint can actually run through.

*(a) What landed.* Four files, one new registry entry:
[`deepseek_v2.h`](../../include/vllm/model_executor/models/deepseek_v2.h) (the
whole port map, `file:line` on both sides),
[`deepseek_v2_weights.cpp`](../../src/vllm/model_executor/models/deepseek_v2_weights.cpp)
(config resolution + the checkpoint name map + the load-time absorption),
[`deepseek_v2.cpp`](../../src/vllm/model_executor/models/deepseek_v2.cpp) (the
forward + the batch split), and
[`deepseek_v2_registry.cpp`](../../src/vllm/model_executor/models/deepseek_v2_registry.cpp)
(one `REGISTER_VLLM_MODEL`, the MLA KV spec, `is_dense_model=false`). The ONLY
edit to shared code is a two-line additive condition in `runner.cpp` so a
`KVCacheSpecKind::kMlaAttention` group is recognised as the model's attention
group — upstream registers MLA against the ordinary `FullAttentionManager`
(`single_type_kv_cache_manager.py:1539`), so nothing else about block tables,
prefix caching or eviction changes.

*(b) THE LOADER GATE — PASSED on the real checkpoint.*
[`test_deepseek_v2_load.cpp`](../../tests/vllm/models/test_deepseek_v2_load.cpp)
on dgx: **4/4 cases, 37,331 assertions**. Every expected tensor mapped with the
right shape, and the exhaustive set comparison shows **5291 checkpoint tensors,
zero unmapped, zero leftover** (1 dense layer x 3 MLP tensors + 26 MoE layers x
[1 router + 3 shared + 64x3 routed] + 27 x 7 attention/norm + embed/norm/lm_head).
Asserted per layer: `q_proj [3072,2048]`, `kv_a_proj_with_mqa [576,2048]`,
`kv_b_proj [4096,512]`, `o_proj [2048,2048]`, `kv_a_layernorm [512]`, and the
**load-time absorption split `W_UK_T [16,128,512]` / `W_UV [16,512,128]`**.

*(c) THE QUERY BRANCH V2-Lite TAKES, asserted not assumed.* `q_lora_rank: null`
-> `has_q_lora() == false` -> the DIRECT `q_proj` path (`deepseek_v2.py:1028-1034`);
`fused_qkv_a_proj` / `q_a_layernorm` / `q_b_proj` are asserted EMPTY on every
layer. The fused branch is implemented (the loader un-fuses
`packed_modules_mapping["fused_qkv_a_proj"] = [q_a_proj, kv_a_proj_with_mqa]`
into one merged raw-NK owner) but, as §5.1 already recorded, gets no e2e coverage
on this box.

*(d) THE FORWARD GATE — PASSED, and it is obviously right, not merely finite.*
The real DeepSeek-V2-Lite, single-sequence CPU prefill of
`The capital of France is` (ids `549,6077,280,7239,317`):
**argmax = 8913 = ` Paris`**, top-5 `[' Paris', ' the', ' a', ' one', ' also']`,
all logits finite, re-run BIT-identical. This is the direct analogue of the
Qwen3-Coder W3 sanity case and it means the whole chain — embed, the A
projections, both RMSNorms, the decoupled YaRN RoPE with the mscale² scale, the
MLA cache write, the materialized-MHA prefill, the grouped router, 64 routed
experts, the shared experts, `first_k_dense_replace`, the final norm and the
untied lm_head — is composed correctly. **The token-exact bar remains W8**; a
sane argmax is a sanity result, not a correctness gate.

*(e) THE BATCH-ORDERING GATE — the invariant W6 found the hard way, now
enforced.* `BuildMlaBatchSplit` is a PURE HOST function reproducing
`split_decodes_and_prefills` (`mla_attention.py:1640-1649`, `reorder_batch_
threshold == 1` at `:1420`) and `prefill_tokens_with_context` (`:1806-1810`), and
it VALIDATES both orderings instead of assuming them: a decode following a
prefill THROWS, and a with-context prefill following a context-free one THROWS —
naming the offending request index and citing the upstream line. Four cases cover
the mixed batch (2 decodes + 1 with-context prefill + 1 context-free, checking
every derived count and the relative `prefill_cu_seqlens_q`), pure-decode,
pure-prefill, and both violations. W6 measured **0.86 relative error** from
exactly the second violation; it is now unrepresentable rather than silent.

*(f) SHARED EXPERTS — new for this family, and gated two ways.* Qwen3-Coder has
none and Qwen3.6's carries a SIGMOID GATE that DeepSeek's does not
(`deepseek_v2.py:344-357`: a plain `DeepseekV2MLP` whose output is ADDED to the
routed sum, `moe_runner.py:407`). (1) **Equivalence:** with every routed expert
weight exactly zero the routed term is exactly zero, so a MoE layer must be
BIT-IDENTICAL to a DENSE layer holding the same MLP weights — asserted with
`memcmp`. (2) **Exercised:** turning the shared expert off (`n_shared_experts: 0`)
changes the logits, so the path demonstrably runs.

*(g) THE CUDA PATH IS EXERCISED, NOT MERELY COMPILED.* A CUDA case runs the same
tiny model at DeepSeek-V2-Lite's REAL MLA head geometry (QK 192, V 128, 576-wide
latent — the only head_dim the W5 FA-2 MLA prefill is instantiated for) through
the CUDA MLA kernels, `vt::ConcatAndCacheMla`, `vt::BatchedMatmul` AND the
CUDA-only grouped bf16 MoE GEMM branch (`vt::MoeGroupedGemmBf16` x3 +
`vt::MoeSiluMul`): bit-exact run to run on device, and agreeing with the CPU
reference path to **0.0061 worst relative logit error** (a bf16 GEMM
accumulation-order band, not a bit comparison — the CPU arm runs the per-expert
reference loop). `compute-sanitizer` memcheck **0 errors**, racecheck **0
hazards**, synccheck **0 errors** (with `--num-cuda-barriers 65536`).
Total: [`test_deepseek_v2_forward.cpp`](../../tests/vllm/models/test_deepseek_v2_forward.cpp)
**11/11 cases**. The KV spec itself is gated too: exactly ONE group, kind
`kMlaAttention`, `num_kv_heads == 1`, `head_size == 576`, and
`page_size_bytes == block_size * 576 * dtype_size` — the factor 2 every other
attention spec carries simply does not appear.

**Full `ctest` on dgx: 182/184.** Both failures are explained and neither is a
W7 defect. (1) `test_capi` is the documented pre-existing flake, signature
CONFIRMED at `test_capi.cpp:353` — a partial-UTF-8 first byte
(`\ufffd oollllll` vs `loollllll`). (2) `test_qwen36_gguf_engine` failed with
`vt cuda combine_tokens: CombineKernel launch: out of memory` under `ctest -j2`,
i.e. the documented unified-memory hazard of scheduling two ~30 GB model tests
together — it PASSES standalone (334 s), so it is a scheduling artifact, not a
defect.

*(h) A LATENT TREE-WIDE HAZARD FOUND (pre-existing, NOT MLA).* The shared
`DevicePool` (`device_pool.h`) is a process-wide singleton keyed ONLY on a byte
size class and documents itself "backend-agnostic". That is safe for the engine,
which drives one device per process — but a TEST BINARY that runs a CPU forward
and a CUDA forward hands the second backend the first backend's recycled
pointers. Observed as a SIGSEGV in the CPU arm dereferencing a CUDA block. Worked
around test-locally by giving each arm its own `DevicePool` via `ActivePoolScope`
(the mechanism the aux-stream MoE overlap already uses); the underlying hazard is
recorded here and in the ledger rather than papered over, and it is unrelated to
this campaign.

*(i) DEVIATIONS, recorded not glossed.*
1. **Only ONE architecture is registered, not the four the plan cell named.**
   `DeepseekForCausalLM` is plain MHA (`deepseek_v2.py:1201-1211` -> `:133
   DeepseekAttention`) — §0.7's own finding — so registering it would be a false
   support claim; the config parse REFUSES that branch by name. `DeepseekV3ForCausalLM`
   resolves to the same Python class but every shipped V3 checkpoint is fp8
   block-quantized and 671B (neither our bf16 loader nor GB10 can take it), and
   `DeepseekV32ForCausalLM` additionally needs the DSA indexer we do not have —
   both are refused by name in the config parse. W10 owns their honest rows.
2. **The MoE block does NOT reuse `RunMoeBlock`.** DeepSeek's router is grouped
   and its shared expert is ungated; Qwen's block hardcodes an ungrouped softmax
   router and a sigmoid-gated shared expert. Reusing it would have applied Qwen's
   semantics. Writing the DeepSeek block directly over the same underlying `vt::`
   ops also means ZERO edit to the 27B/35B/Coder MoE paths — which is why the
   regression set is unchanged despite "the shared-expert path now activates".
3. **`routed_scaling_factor` is applied to the ROUTING WEIGHTS**
   (`grouped_topk_router.py:159-160`) rather than to the combined routed OUTPUT
   (`moe_runner.py:402-406`, which vLLM's CUDA path selects via
   `apply_routed_scale_to_output=True`). The routed combine is linear in the
   weights so the two are the same function, differing only in rounding order,
   and the shared term — which must NOT be scaled — is added after either way.
   V2-Lite has `routed_scaling_factor: 1.0`, so on the W8 vehicle they are
   bit-identical.
4. **The chunked-prefill workspace is sized with `max_model_len :=
   config.max_position_embeddings` and `max_num_seqs :=` this step's `num_reqs`**,
   mirroring `determine_chunked_prefill_workspace_size`
   (`mla_attention.py:1422-1451`) as closely as a model forward with no
   `VllmConfig` can. With an unclamped `--max-model-len` these are the same
   numbers upstream uses.
5. **The A-projections stay per-row-slice** (W6's deviation, unchanged) and there
   is **no decode CUDA-graph sibling and no speed number** — both are W9.

*(j) NOT IN W7, on purpose.* No paged-engine wiring of the batch REORDER itself
(the model VALIDATES the ordering; producing it in the scheduler is W8's job), no
token-exact gate, no goldens, no benchmark. **NO MODEL ROW MOVES OFF `SPIKE`** — a
loading, forwarding model is not a supported model.

**W8 — LANDED 2026-07-22. THE SACRED CORRECTNESS GATE — PASSES 8/8.** Evidence
root on dgx: `~/w8mla` (goldens, build, gate, teacher-forcing). The increment
that decides whether DeepSeek-V2 is correctness-complete. It is.

*(a) THE RESULT.*
[`test_deepseek_v2_paged_engine.cpp`](../../tests/vllm/models/test_deepseek_v2_paged_engine.cpp)
drives the 8-prompt battery through the FULL paged `LLMEngine`
(`LoadedEngine::FromModelDir` -> InputProcessor -> Scheduler -> decode-first
reorder -> MLA cache write + MLA decode/prefill + DeepSeek MoE -> Sampler ->
OutputProcessor) and compares greedy decode to the pinned vLLM 0.25.0 oracle:
**8/8 prompts PASS — STRICT token-exact 5/8, near-tie band 3/8, 92/128 tokens
strictly exact, max teacher-forced gap 0.25 nats, 0 forward-divergent**
(224 assertions). The bar applied is the RATIFIED NEAR-TIE-ROBUST form, and it
was ARRIVED AT rather than chosen — see (b).

*(b) HOW THE BAR WAS DETERMINED. The STRICT form was applied first and it came
out 5/8; the near-tie form was then EARNED by measurement, not assumed.*
1. **vLLM is deterministic here, re-confirmed.** W0's batch=1 K=5 result (8/8) is
   re-confirmed by W8's own capture at T=16: **0 multi-valued (prompt,pos)
   cells** over K=5 (`greedy_dist.npy`, re-asserted at run time by the gate).
   The batched-probe artifact is NOT used, and the gate says so in-line.
2. **Ran the STRICT gate: 5/8 (92/128 tokens).** Prompts 2, 3 and 5 diverged.
3. **Ran the ratified TEACHER-FORCING diagnosis**
   (`scripts/deepseek-v2-neartie-gap.py`, the
   `qwen3coder-neartie-gap.py` template): feed vLLM OUR exact sequence, record
   per position how many nats vLLM's OWN argmax beats OUR token by GIVEN OUR
   PREFIX. Result: **36 divergent positions, 35 of them at gap EXACTLY 0.0000
   nats** — vLLM's own logits on our prefix pick OUR token, so those are the
   downstream tail of a single earlier flip, not independent errors. **Exactly
   one root flip carries any gap at all: prompt[3] tok 9 at 0.2500 nats**, inside
   the ratified 0.5-nat band and equal to the worst gap the already-landed
   Qwen3-dense 4B gate carries. **ZERO tokens outside vLLM's top-20.**
   Two of the three root flips (prompt[2] tok 1, prompt[5] tok 1) sit at gap
   0.0000 — i.e. vLLM's incremental DECODE argmax disagrees with vLLM's OWN
   teacher-forced PREFILL argmax, and OUR token is the one vLLM's logits prefer.
4. **The nats evidence is COMMITTED** (`neartie_gap_mnats.npy` + `our_ids.npy`
   as its anchor), so the claim is auditable and an engine change that moves a
   token trips the anchor check and forces a re-capture rather than silently
   inheriting a stale band. Anything beyond `kNearTieMnats` (0.5 nats) or outside
   vLLM's top-K still FAILS the gate. This is strict where the bar is well-posed
   and tolerant only where vLLM cannot separate the tokens from itself.

*(c) THE SCHEDULER/RUNNER WIRING — W8's first job — and the honest finding that
NO NEW REORDER CODE WAS NEEDED.* The plan expected W8 to make the engine produce
the order W7's `BuildMlaBatchSplit` validates. Reading the runner first (rather
than writing a reorder and declaring victory) showed the order was ALREADY
correct, and for a non-accidental reason: `runner.cpp:671` calls
`reorder_batch_to_split_decodes_and_prefills` unconditionally with
`decode_threshold = 1` — **exactly MLA's `reorder_batch_threshold`
(`mla_attention.py:1420`)** — and its four-way target ordering is
`decode(0) -> short_extend(1) -> long_extend(2) -> pure_prefill(3)`. Mapped onto
MLA's two invariants: regions 0 and 1 both have `query_len <= 1` and are
therefore `split_decodes_and_prefills` DECODES, so decodes form a batch PREFIX;
and region 2 (has context) precedes region 3 (context-free), so **with-context
prefills lead the prefill tail** — the `prefill_tokens_with_context` prefix-length
invariant W6 measured 0.86 relative error from violating. The port made for GDN
happens to be exactly MLA's contract.
**That is a claim, so W8 PROVES it end to end instead of asserting it**, with a
non-vacuity bar (a legal order over a stream of batch-of-1 steps would prove
nothing):
- new DIAGNOSTIC counters `MlaBatchSplitStats` (deepseek_v2.h) accumulate the
  shapes the engine actually produced, per forward — the W1
  `fa_page_size_bytes()` pattern, "proof the path is EXERCISED, not compiled";
- **phase 2** admits the battery CONCURRENTLY with STAGGERED admission and
  measures **steps=30, MIXED=7, max_num_reqs=8, max_num_decodes=8** — seven
  genuinely mixed decode+prefill steps at up to 8 concurrent requests, and
  `BuildMlaBatchSplit` (which THROWS naming the request and citing the upstream
  line on either violation) never fired;
- **phase 3** submits a 91-token prompt twice so the PREFIX CACHE produces a real
  **with-context prefill** (`with_context_prefill_steps=1`), exercising the
  prefix-length invariant through the engine, with the cached run reproducing the
  uncached one exactly;
- **phase 0** asserts the engine really allocated an MLA cache:
  `fa_page_size_bytes = 36864 = block 32 x 576 x 2B`, with the factor 2 every
  other attention spec carries simply absent.

*(d) THE REAL BLOCKER W8 HIT WAS THE TOKENIZER, NOT THE MODEL — a whole new
pre-tokenizer family.* The first gate run did not diverge; it REFUSED to load:
`tokenizer: unsupported normalizer "Sequence"`. W7 never exercised the tokenizer
(its forward gate uses hard-coded ids). Behind the normalizer sat a pre-tokenizer
we could not express at all. **DeepSeek's is not another alternation regex like
Qwen/Llama-3/GPT-2 — it is a HF `Sequence` PIPELINE of SEVEN stages**: five
`Split(Isolated)` over ENUMERATED codepoint ranges (newlines; cased letters;
ASCII+fullwidth+CJK punctuation; trailing whitespace; CJK/Hangul), then
`Digits(individual_digits=true)`, then `ByteLevel(use_regex=false)`. Landed as
`SplitPattern::kDeepSeek` (pretokenizer.{h,cpp}) plus verbatim-pattern
recognition in the loader, with three findings worth keeping:
1. **Stage ORDER is load-bearing.** Stage 2's punctuation class spans 0x3A-0x7E
   and therefore CONTAINS A-Z/a-z; it is only correct because stage 1 already
   isolated the letter runs. A single-pass alternation scanner is the wrong
   shape; this is a genuine pipeline.
2. **Stage 3's `\s+$` anchors to end-of-PIECE.** `$` in onig is end-of-LINE, but
   stage 0 has already isolated every `\r`/`\n` into its own piece, so no piece
   reaching stage 3 contains a newline and the two readings coincide. Recorded
   rather than silently relied on.
3. **The empty `Sequence` normalizer is accepted; a non-empty one still fails.**
   DeepSeek ships `{"type":"Sequence","normalizers":[]}` — a genuine no-op.
**It is MEASURED, not argued:** new
[`test_tokenizer_parity_deepseek.cpp`](../../tests/vllm/test_tokenizer_parity_deepseek.cpp)
checks Encode / Decode round-trip / incremental detokenization against the REAL
HF `tokenizers` library over a 98-entry corpus extended with pipeline-stress
entries (individual-digit splitting, the letter/punct class overlap, fullwidth
and CJK punctuation, CJK/Hangul, the cased-letter class boundaries — Greek,
Cyrillic, Armenian, Georgian, Cherokee, Coptic, Deseret are IN the class while
Arabic, Hebrew, Devanagari and Han are NOT — newline isolation, trailing
whitespace, and `\s?`-prefix attachment across every class boundary):
**6/6 cases, 2461 assertions, token-for-token.** A transcription hazard was hit
and is now designed out: the class literals were first copied as literal UTF-8
and silently picked up **U+03CE where the checkpoint has U+1F7D** — visually
identical glyphs — so they are written as explicit `\u`/`\U` escapes and compared
VERBATIM against the checkpoint, which makes a DeepSeek variant shipping
different ranges fail loudly instead of mis-tokenizing.

*(e) THE TOKENIZATION GOLDENS EARNED THEIR PLACE — by REFUTING a fix I was about
to make.* `tokenizer_config.json` declares `tokenizer_class: LlamaTokenizerFast`
and **`add_bos_token: true`** with `bos_token <|begin of sentence|>` (id 100000),
while tokenizer.json's post_processor is a plain ByteLevel declaring no special
tokens. That reads as a certain OPT-style missing-BOS bug, and a
tokenizer_config.json reader was written to fix it. **The oracle capture then
showed vLLM's own `prompt_token_ids` carry NO BOS** (`[549, 6077, 280, 7239,
317]`), and a direct probe confirmed why: `AutoTokenizer.from_pretrained(...,
trust_remote_code=True)` resolves to class `TokenizersBackend` with
`add_bos_token` **False**, and `encode(add_special_tokens=True)` adds nothing.
Our loader already matched bit-for-bit; **the "fix" would have BROKEN a passing
gate.** It was reverted, and the measured behaviour is now pinned by a guard case
in `test_bpe.cpp` so the next reader has to re-measure before "fixing" it
([[ground-premises-before-dispatching]]). The committed `p{i}_prompt.i32`
goldens are asserted with `REQUIRE` (not advisory) and all 8 match.

*(f) BATCH INVARIANCE — REPORTED, deliberately NOT a bar, and grounded.* The
concurrent run reproduces 6/8 of the batch=1 sequences. Asserting 8/8 would hold
us to a standard **the oracle itself does not meet**: W0 measured vLLM's OWN
greedy changing on **3/8** of this same battery under batched generation while
being deterministic 8/8 at batch=1. Batching re-orders reductions (the grouped
MoE GEMM batches tokens across requests) and this model sits on bf16 near-ties —
3 of 8 prompts are within 0.25 nats. So it is reported with the oracle's own
number beside it, while the thing that WOULD be a real defect — the batch ORDER —
is asserted hard.

*(g) GATES.* Clean full CUDA rebuild on dgx sm_121: **RC=0, 0 warnings / 0
errors**. Local CPU suite **151/151** (one parallel-contention flake,
`test_openai_api_server`, passes standalone). Tokenizer regressions unchanged
(`test_tokenizer_parity` Qwen3.6 corpus, `test_bpe` 17/17,
`test_pretokenizer`, `test_detokenizer`, `test_input_processor`,
`test_op_parity`).

*(h) DEVIATIONS, recorded not glossed.*
1. **The oracle runs `moe_backend='triton'`.** vLLM auto-selects the FlashInfer
   CUTLASS unquantized MoE backend, whose expert workspace **REBOOTED dgx three
   times** on 2026-07-22 (GB10's 119 GiB is UNIFIED, so weights + the 0.40
   reservation + profiling activations + the 30 GB checkpoint page cache all draw
   on one pool; capping `max_num_batched_tokens` to 2048 did NOT help, and the
   third attempt died alone on a freshly-rebooted box with an empty page cache).
   `triton` is a first-class vLLM backend computing the SAME bf16 grouped MoE
   with a small footprint — the identical legitimate-oracle substitution
   `scripts/qwen3coder-*.py` already make. It is used CONSISTENTLY by the capture
   AND the teacher-forcing run, so both arms see one oracle configuration.
2. **`gpu_memory_utilization=0.40` is held**, as mandated; the memory lever used
   was the MoE backend, not the reservation.
3. **The capture caps `max_model_len=2048` / `max_num_batched_tokens=2048` /
   `max_num_seqs=4`.** The battery's longest prompt is ~20 tokens generating 16,
   and nothing about the greedy result depends on these (DeepSeek's YaRN scaling
   reads `original_max_position_embeddings`/`factor` from config.json, never from
   `max_model_len`).
4. **No new reorder code** — see (c). The existing GDN-motivated reorder already
   satisfies MLA's contract; W8 proves that end to end instead of duplicating it.

*(i) WHAT THIS DOES AND DOES NOT LICENSE.* The row moves `SPIKE` -> **`ACTIVE`
with speed explicitly PENDING**. It does NOT move to `DONE`: `DONE` requires
vLLM-speed parity on every axis, which is W9 and has no number at all yet. Both
§5.1 coverage gaps remain exactly as stated (`q_lora_rank=null` and
`n_group=topk_group=1`/softmax/greedy, so the `fused_qkv_a_proj` query branch and
the whole `noaux_tc` router stay unit-gated only on this box).

W9 speed close (decode-graph sibling + MLA fusion recipes + the binding
every-axis grid). W10 the blocked-row honesty pass (config/loader-slice/unit
parity for V3/V3.2; `HW-BLOCKED` recorded where e2e cannot run). Full table with
per-W gates in §10.

---

## W9 — the speed close (LANDED 2026-07-22): an ATTRIBUTED MISS

*(a) VERDICT FIRST.* **The row does NOT reach `DONE`.** It stays `ACTIVE` with a
measured, attributed residual. Two optimizations landed; together they moved
decode throughput from **0.50x to 0.87x** of vLLM at c1 (+73.6% over the W8
baseline) and made our **TTFT BEAT vLLM at c4 and c8**. But output / total /
request throughput is short at all four concurrencies (0.86-0.95x) and TPOT/ITL
is short at c1/c4/c8, so "match or beat on EVERY axis" is NOT met. The full grid,
the denominator justification, the nsys diff and the repro recipe are the binding
record in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md).

*(b) THE DENOMINATOR — settled, with evidence, not assumed.* vLLM AUTO-selects
the **FlashInfer CUTLASS Unquantized MoE** backend for DeepSeek-V2-Lite. It
**cannot run on GB10**: it has now hard-rebooted dgx **five times** (three at W8,
two more at W9). W9's two attempts deliberately applied the Qwen3-Coder
mitigations — `gpu_memory_utilization=0.40`, `max-model-len 2048`,
`max-num-batched-tokens` 2048 then 1024, `max-num-seqs` 8 then 4 — and the second
ran on a **pristine freshly-rebooted box with a 0 GiB page cache and 116 GiB
free**, which is the strongest form of the page-cache-evictor mitigation
available (`sudo` is password-gated here, so `drop_caches` itself never worked).
Both died at the IDENTICAL phase: immediately after `torch.compile`, in the
memory-profiling dummy run / FlashInfer autotune where the CUTLASS MoE expert
workspace is first allocated. So **`--moe-backend triton` IS vLLM's best STABLE
graphed/production configuration on this hardware and is the legitimate bar** —
and it is worth saying plainly that this substitution does not flatter us: we
LOSE to it. Both sides otherwise resolve identically (`TRITON_MLA` decode +
`FLASH_ATTN` MLA prefill), and vLLM runs GRAPHED
(`cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes `[1,2,4,8,16]`).

*(c) THE LEVER `nsys` FOUND — and it was NOT the planned one.* §10's W9 cell
planned "the MLA fusion recipes (RoPE+concat-cache, dual-RMSNorm)". The trace
refuted that priority before a line of it was written. `nsys profile
--cuda-graph-trace=node` on OURS at c1 showed **`MlaDecodeStage1` at 44.7% of ALL
GPU time, 837 us per instance** — 27 layers x 837 us = ~22.6 ms of a ~47.7 ms
TPOT. A batch-1 step reads ~1.25 MiB of KV latent per layer, i.e. ~5 us at GB10's
memory rate, so the kernel was running about **two orders of magnitude off its own
memory-bound floor**. The cause is pure occupancy: `_compute_num_kv_splits`
(`triton_mla.py:40-47`) derives the split count from `max_seq_len` ALONE and
returned **2**; MLA has exactly ONE kv head so `min(BLOCK_H, kv_group_num)`
collapses the head dimension to a single tile; at batch 1 the grid was therefore
**2 CTAs on the entire GPU**.

The fix is **upstream's own occupancy target, actually applied**:
`_compute_num_kv_splits` already names `maximum = sm_count *
_SPLIT_OCCUPANCY_MULTIPLIER` as its occupancy bound but never REACHES it on short
sequences. `ComputeNumKvSplitsFilled` (`cuda_mla_attn.cu`) raises the split count
until the grid actually has that many CTAs given what the other two grid
dimensions contribute, never exceeding upstream's own `maximum` and never
splitting finer than one `kNTile` of work. Result: **837 us -> 45.8 us, an 18.3x
kernel speedup**, 44.7% -> 3.9% of GPU time.

This is a **RECORDED DEVIATION** from upstream, not a mirror: upstream can afford
the `max_seq_len`-only heuristic because it targets a large-batch serving regime
where `batch` alone fills the grid, and because its Triton CTA does the whole head
block with `tl.dot`. It is numerics-visible (the stage-2 online-softmax reduction
order changes), so it was GATED, not assumed — see (e). Rollback:
`VT_MLA_SPLIT_FILL=0`.

*(d) THE DECODE CUDA-GRAPH SIBLING — built as planned, worth ~+2%.*
`DeepseekV2DecodeGraph` (`deepseek_v2.cpp`) is the MLA member of the driver family
(`Qwen3_5DecodeGraph` / `Qwen3_5DenseDecodeGraph` / `Qwen3MoeDecodeGraph`): same
cold -> warm -> capture -> replay machine, same `decode_graph_sizes.h` padded
capture set, same persistent fixed-address host inputs. It required splitting
`ForwardBody` into `EmbedInto` + a capturable `ForwardLayers`, and it measures
**+2.5% / +2.5% / +1.8% / +2.0%** at c1/c2/c4/c8. **That is an order of magnitude
less than the Qwen3-Coder analogue** (which was worth that model's entire c1
deficit), and the reason is now measured rather than guessed: DeepSeek-V2-Lite's
decode step is GPU-BOUND, so there is only ~1.2 ms/step of host launch tax to
remove, not ~5 ms.

*(e) A REAL CUDA-GRAPH SAFETY BUG THE GRAPH EXPOSED — and the guard that now
catches its whole class.* The first capture produced a graph that was CORRECT on
the replay issued immediately after capture and then diverged by ~19 logits on
replay #2 onwards, eventually faulting with "an illegal memory access was
encountered". `compute-sanitizer` serialized enough to HIDE the fault and showed
only wrong numbers — 0 memory errors, wrong tokens. A `VT_DEEPSEEK_GRAPH_VERIFY=1`
diagnostic (kept in the tree) that runs the eager forward alongside each replay
localized it in one run.

Root cause: `BuildMlaStep` uploaded the decode half's `block_table` and `seq_lens`
from **function-local temporary vectors**. Eagerly that is fine — the
`cudaMemcpyAsync` is issued before the temporary dies. Under CAPTURE it is a
use-after-free: the copy becomes a graph node that BAKES the source address, and
every later replay copies whatever now occupies that freed heap block into the
decode metadata. The fix is `UploadRange`, which uploads a contiguous range of the
CALLER's persistent vector with no temporary — legitimate here because the decode
half is a batch PREFIX by the scheduler's own reorder contract
(`mla_attention.py:1420`), and the prefill half is a SUFFIX, so neither needs a
copy at all. After the fix the verify hook reports **`max abs(graph - eager) = 0`**:
the replay is BIT-IDENTICAL to eager.

A second, independent capture hazard surfaced once the occupancy fill made the
split-KV workspace size batch-dependent: it GREW inside a capture region, which
turns `cudaMallocAsync` into a graph-owned allocation node whose memory the graph
frees. Two changes close it: `EnsureMidScratch` now carries the house capture
guard (mirroring `cuda_dropin.cu:124` and the FA-2 decode launcher at
`cuda_flash_attn_fa2.cu:717-720`) so growth-under-capture is a loud error instead
of a silent fault, and the workspace is sized at the CONSTANT bound `sm_count * 2`
splits rather than at the call's chosen `num_splits`, which makes the allocation
depend only on `(batch, num_heads)` — constant for a captured size, so the
pre-warm step's allocation always suffices.

*(f) CORRECTNESS IS INTACT — the SACRED gate is 8/8 with EVERY optimization
DEFAULT-ON, with IDENTICAL results.* `test_deepseek_v2_paged_engine`: **8/8
prompts PASS, STRICT token-exact 5/8, near-tie band 3/8, 92/128 tokens strictly
exact, max teacher-forced gap 0.25 nats @ prompt[3] tok=9, 0 forward-divergent,
vLLM self-determinism 0 multi-valued cells** — character-for-character the W8
result, so neither lever moved a token. The `MlaBatchSplitStats` diagnostics are
also unchanged (phase 1 steps=128 / decode_only=120; phase 2 MIXED=7,
max_num_reqs=8; phase 3 with_context_prefill_steps=1), which required the graph
driver to record the split on the REPLAY path too, since `BuildMlaStep` does not
run there. The capture set exercised is `S = {1, 2, 4, 8}`.
`test_ops_mla_attn` stays **11/11 / 2,303,193 assertions**.

*(g) THE RESIDUAL, ATTRIBUTED TO KERNELS.* From the same nsys diff, after the fix:
1. **MoE grouped GEMM is now our top kernel at 40.5%** — ~1.2x its own bandwidth
   floor, so close but not free. vLLM runs one `fused_moe_kernel` family.
2. **Dense projections at batch 1 fall to cuBLAS `gemvx` for 31.8% of our GPU
   time; vLLM splits the same work between `gemvx` (12.7%) and tensor-core
   `nvjet_sm121_tst_mma_*` (6.6%).** This is the LARGEST single attributable
   difference and the **next lever by gain/effort** — it is a dispatch change, not
   a new kernel.
3. **vLLM runs Inductor-codegenned FUSED glue we do not have**
   (`triton_red_fused_add_fused_add_rms_norm_1`,
   `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0`,
   `triton_poi_fused_mul_silu_slice_0`, `triton_poi_fused_2`), ~7-8% of its
   profile — the same glue-fusion theme the 35B campaign recorded.
4. Our MLA decode attention is still ~2.3x vLLM's per call (45.8 + 22.6 us vs its
   `_fwd_grouped_kernel_stage1` at 29.6 us) — small in absolute terms now.

*(h) WHAT WAS NOT DONE, said plainly.* The planned MLA fusion catalog entries
(RoPE+concat-cache, dual-RMSNorm) were NOT built — the trace ranked them far below
the occupancy fix, and building them anyway would have been ceremony. The three
W4-W7 deviations listed as W9 candidates were re-examined and left alone on the
same evidence: `vt::MergeAttnStates` is chunked-prefill-only and does not appear
in the decode profile at all; `vt::ConcatMlaNopeRope` is 0.1% of GPU time
(vectorizing it cannot pay); the fused A-projection GEMM addresses part of the
`gemvx` line but the bigger half of that line is the tensor-core dispatch, so it
is folded into lever (2) above rather than done blind. `cuda_arch_tactics`
registration stays deferred: there is still exactly one MLA decode kernel.

*(i) WHAT THIS LICENSES.* The row stays **`ACTIVE`** with the residual named per
axis and attributed to specific kernels. It reaches `DONE` when the c1..c8 grid
passes on EVERY axis. Both §5.1 coverage gaps are unchanged.

## W10 — the blocked-row honesty pass (DONE 2026-07-22): the campaign's W-plan is COMPLETE

*(a) WHAT THIS STEP IS.* Records only — **no code, no kernels, no build, no GPU work, nothing
downloaded, and no benchmark number produced or claimed.** Its job is to leave every row this campaign
touched in the state the EVIDENCE supports, and to mark the ones that are hardware- or
dependency-blocked as such rather than leaving them in a state that implies a gate we cannot run.

*(b) ROW DISPOSITIONS — final honest state.*

| Row | Was | Now | Why |
|---|---|---|---|
| `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` | `ACTIVE` | `ACTIVE` | Correctness COMPLETE (SACRED 8/8), speed SHORT (0.86-0.95x tok/s). `DONE` needs BOTH. One named open lever |
| `MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm` (V3, V3.2, and K2/K2.5's text backbone) | `SPIKE` | **`BLOCKED`** | **HW** ~642 GiB fp8 / ~1250 GiB bf16 (671B) vs 119 GiB unified; **DEP** (V3.2 only) DSA has no working sm_121 backend |
| `MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm` | `SPIKE` | **`BLOCKED`** | **HW** ~428 GiB vs 119 GiB; and DISJOINT from MLA — it belongs behind the sliding-window track `ROAD-V1-C5` |
| `MODEL-TEXT-deepseek-v2-deepseek-for-causal-lm` | `SPIKE` | `SPIKE` | **NOT an MLA row.** Plain MHA (`deepseek_v2.py:1201-1211`); needs our existing dense attention + the DeepSeek MoE block. The 16B variant would fit GB10, so it is not blocked — just not started |
| `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` | `SPIKE` | `SPIKE` | MLA half unlocked (NoPE = a simplification of what we built); KDA is a SEPARATE kernel campaign; HW-MARGINAL at ~89.4 GiB of ~110 |
| `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` (GLM-5) — **cross-claim** | `SPIKE` | **`BLOCKED`** | **HW** 1404.2 GiB; **DEP** GLM-5.x is V3.2 verbatim (`deepseek_v2.py:1917-1918`), so the same DSA dead-end. Edited from this claim with the disposition recorded in BOTH coordination rows |

Per the record contract a claim may hold only `SPIKE`/`ACTIVE` rows, so each `BLOCKED` row is
RELEASED from its claim's row list in the same change (the checker enforces this, and it is the right
semantics: a blocked row is not work anyone is doing).

*(c) THE DSA DEP-BLOCK, stated once so both rows can cite it.* For a SPARSE model the backend filter
(`is_sparse()` XOR `use_sparse`) eliminates `TRITON_MLA`, so of the two entries in sm_121's MLA
priority list (`cuda.py:129-133`) exactly ONE candidate survives:
`FLASHINFER_MLA_SPARSE_SM120`. That path is non-functional on the flashinfer available here — its
sm12x dispatch goes to **XQA** (`flashinfer/mla/_core.py:1169-1172`), which is DENSE-ONLY and
hardcodes `0` for `sparse_mla_top_k` (`_core.py:1483`), while vLLM's probe
`has_flashinfer_sparse_mla_sm120()` (`vllm/utils/flashinfer.py:216-231`) only checks that three
symbols import and never the function actually called; upstream's single test
(`tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py:37`) MONKEYPATCHES that probe to `True`
and asserts nothing numerical. **This is a DEPENDENCY gap to watch, not work we close by porting
harder.** (Full four-failure analysis: [glm-dsa-latest-deepseek](glm-dsa-latest-deepseek.md) §0.2.)
Recorded alongside it so nobody ports the wrong file: **`vllm/models/deepseek_v32/` is UNREGISTERED
DEAD CODE** upstream — `registry.py:92-93` routes both V3 and V3.2 to `deepseek_v2.py`.

*(d) WHAT CAN STILL BE GATED ON A BLOCKED ROW, versus what cannot.* CAN: config/registry resolution
against the real `config.json` (kilobytes, no weights); a weight-map/loader assertion on a single-layer
SLICE from the safetensors index; unit parity at the real dimensions. CANNOT: any end-to-end
token-exact or speed gate, now or after any amount of software work. Of the "CAN" list, the two pieces
that matter are ALREADY GATED from W3/W6 (the `noaux_tc` router and the `q_lora` branch, both at V3's
real dimensions); the rest is recorded as a named, UNBUILT option rather than claimed as coverage —
see the W10 row of §10 for why building them now would have asserted our own refusal rather than
upstream parity.

*(e) THE TWO PERMANENT COVERAGE GAPS — now stated in the ROWS, not only here.* Both are properties of
this hardware, not of our port: (1) the `noaux_tc` grouped router has **no e2e coverage** — V2-Lite is
`n_group=topk_group=1` with softmax/greedy and has no `e_score_correction_bias` at all
(`deepseek_v2.py:315-320`); (2) the `q_lora` query branch has **no e2e coverage** — V2-Lite is
`q_lora_rank=null`, so `fused_qkv_a_proj`/`q_a_layernorm`/`q_b_proj` and the `packed_modules_mapping`
un-fusing exist only under unit test. **Neither closes by porting harder — it needs a different
checkpoint, and exactly one reachable checkpoint closes BOTH:**

> **NAMED NEXT VEHICLE — GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, `zai-org/GLM-4.7-Flash`, 31.2B,
> 58.2 GiB bf16) FITS GB10** (~110 GiB usable) and has **`q_lora_rank=768`** AND **`noaux_tc` with an
> `e_score_correction_bias`**, so it converts both unit-only gaps into e2e coverage. Upstream it is a
> zero-override subclass of DeepSeek-V2's MLA attention, so it lands as a pure addition on top of
> W1-W7. Row: `MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm`, spike
> [glm-dsa-latest-deepseek](glm-dsa-latest-deepseek.md).

*(f) THE DEVIATIONS LEDGER, swept into the rows.* Every deviation this campaign recorded is now
discoverable from the DeepSeek-V2 matrix row itself rather than only from this spec: the upstream FA-2
empty-K / `unpadded_lse` defect W5 works around (`flash_fwd_kernel.h:1030-1043`); scalar
`vt::MergeAttnStates` and `vt::ConcatMlaNopeRope` where upstream is vectorized; per-row-slice
A-projections rather than one fused GEMM; the `cuda_arch_tactics` registration deferral; DCP/
context-parallel and fp8-prefill not ported; the W9 split-KV occupancy fill as a recorded deviation
from upstream's `max_seq_len`-only heuristic; and the PRE-EXISTING, tree-wide, non-MLA `DevicePool`
singleton hazard (byte-size-class keying -> cross-backend recycled pointers -> SIGSEGV in a mixed
CPU/CUDA test binary; worked around test-locally with `ActivePoolScope`).

*(g) NUMBERS CORRECTED IN PASSING.* §5's disk figure of "238 GiB free" was measured before the
V2-Lite download; dgx's root filesystem was re-measured at **184 GiB free / 95% full** on 2026-07-21.
Neither number changes any verdict — every HW-BLOCKED member misses on MEMORY first — but the smaller
one is the live figure for any future download, per [[grid-per-sha-trees-fill-disk]].

*(h) BLOCK CLOSURE — NOT YET, and nothing is archived.* Every row in the block is at its final honest
state, but `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` is `ACTIVE`, not `DONE`: `DONE` requires
BOTH token-exact correctness (have it) AND vLLM-speed parity on EVERY axis (0.86-0.95x on throughput,
so no). The coordination rule is "when EVERY row in an execution block is `DONE`, move the block
plan/report to `.agents/completed/`", so **this spec and its claim STAY LIVE**. Archiving now would
put load-bearing content under `completed/`, which the doc-lifecycle rule forbids. The claim remains
open with exactly ONE open item: route the batch-1 dense projections off cuBLAS `gemvx` (31.8% of our
GPU time) onto a tensor-core GEMM — vLLM splits the same work `gemvx` 12.7% + `nvjet_sm121_tst_mma_*`
6.6%. That is a DISPATCH change, not a new kernel.

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
