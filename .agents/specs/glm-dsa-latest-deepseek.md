# SPIKE: GLM family + DSA (sparse MLA) + latest DeepSeek (V3.2, V4)

**G2 LANDED 2026-07-24 — `Glm4ForCausalLM` (GLM-4-9B-0414), the FIRST GLM-family
model. SACRED gate 16/16 vs vLLM 0.25.0** (STRICT token-exact 13/16 + near-tie band
3/16, max gap 0 nats, 0 forward-divergent; vLLM K=5 self-deterministic ⇒ STRICT bar).
Row `MODEL-TEXT-glm4-glm4-for-causal-lm` SPIKE→ACTIVE (correctness DONE, speed
PENDING). **The spike's §0.4.3 "two genuinely new primitives" over-estimated the
work — BOTH reduced to EXISTING infrastructure:** (1) partial + INTERLEAVED rope is
already implemented in `RopeFromCache` (both backends — `cuda_ops.cu:697-698` /
`cpu_ops.cpp:744-746`), which honors partial `rotary_dim` (leading-slice rotate, tail
passthrough) AND `is_neox_style=false`; it is the SAME path DeepSeek-V2 decoupled rope
is gated on. GLM just routes it with `is_neox_style=false` over `rotary_dim=64`. No new
kernel. (2) Sandwich norms are standalone `vt::RmsNorm` (nullptr residual) on the
sublayer output — the existing op. New files: `glm4.{h,cpp}` + `glm4_weights.cpp` +
`glm4_registry.cpp` (one `REGISTER_VLLM_MODEL`), reusing the shared dense glue; a
GLM-specific attention block (biased qkv via `vt::Add`+1-D `LoadMergedBf16Vector`, no
QK-norm) per the OPT precedent (D4). Loader 523 tensors, zero missing/unmapped; GLM-4-
9B-0414 ships `mlp.gate_up_proj` PRE-MERGED. Runs EAGER (bf16). Remaining G1/G3/G4/G5
per §0.7 unchanged.**

**G1 LANDED 2026-07-24 — `Glm4MoeLiteForCausalLM` (GLM-4.7-Flash, 31.2B), the SECOND MLA
model. SACRED gate 8/8 vs vLLM 0.25.0** (STRICT token-exact 1/8 + near-tie band 7/8,
69/128 tokens strictly exact, max teacher-forced gap 0.0 nats, 0 forward-divergent; vLLM
K=5 self-deterministic ⇒ STRICT bar). Row `MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm`
SPIKE→ACTIVE (correctness COMPLETE, speed PENDING). **The scope was MUCH smaller than
§0.4.1 estimated — the MLA campaign already built nearly everything:** the noaux_tc
grouped router (sigmoid + `e_score_correction_bias` + group masking + `routed_scaling_factor`)
had ALREADY landed in the campaign's W3, and GLM-4.7-Flash reuses the ENTIRE DeepSeek-V2 MLA
stack over the SAME `DeepseekV2Weights` (W6 MLA block incl. q_lora branch, W7 loader + fused_qkv_a_proj
merge, W9 decode graph). Genuinely-new work reduced to FOUR additive pieces: (1) a `head_dim=256`
dispatch in `LaunchMlaPrefillFA2Bf16` (GLM qk 256/v 256; the 256 split-KV kernel was already
compiled for the 27B/35B paged prefill, so the 192 path is byte-identical); (2) MTP-tolerant
parse/loader (`allow_mtp_tail`, defaulted false → DeepSeek-V2 byte-identical); (3) the GLM registry
TU `glm4_moe_lite_registry.cpp`; (4) a scoring-func fix (GLM's config OMITS `scoring_func` and its
model class hardcodes sigmoid, so `noaux_tc`+absent-key now defaults to sigmoid; DeepSeek-V2-Lite
greedy→softmax UNCHANGED — caught by the loader gate before the SACRED gate). **C2 CLOSED:** the
q_lora query branch AND the whole noaux_tc router now have E2E coverage (were unit-gated-only on
DeepSeek-V2-Lite). Residual per R4: `n_group=topk_group=1` on GLM-4.7-Flash, so multi-group masking
stays unit-gated only. Regressions ALL byte-identical (incl. DeepSeek-V2 223/223 Release + asserts-on
exit 0, the shared-TU canary). **Status:** SPIKE + G2 + G1 IMPLEMENTED (Glm4ForCausalLM + Glm4MoeLiteForCausalLM landed; G3/G4/G5 still design-only blocked-honesty passes).
**Base:** `aa65ce7`. **Oracle pin:** `/home/mudler/_git/vllm` @ `e24d1b24`.
**Claim:** `CLAIM-GLM-DSA-LATEST-DEEPSEEK`.
**Parent plan:** [`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B.3 Tier 3.
**Sibling campaign (DO NOT EDIT — owned by `CLAIM-MLA-DEEPSEEK`):**
[`mla-deepseek-campaign.md`](mla-deepseek-campaign.md). This spike CROSS-REFERENCES
it and, in three named places, CORRECTS or EXTENDS it (§0.1). Reconciliation is
the user's; this spec never edits that file.

Rows covered by this spike:
`MODEL-TEXT-chatglm-chat-glmfor-causal-lm`,
`MODEL-TEXT-glm-glm-for-causal-lm`,
`MODEL-TEXT-glm4-glm4-for-causal-lm`,
`MODEL-TEXT-glm4-moe-glm4-moe-for-causal-lm`,
`MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm`,
`MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm`.

DeepSeek-V3.2 itself lives in `MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm`,
which is ALREADY `SPIKE` under `CLAIM-MLA-DEEPSEEK`. This spike does not claim,
edit, or re-own that row. It supplies the DSA surface that row's owning spike
declared out of scope, as cross-referenced input.

---

## 0. Headline findings

### 0.1 What this spike CORRECTS or EXTENDS in the MLA campaign spike

Three items, each grounded, listed here so the user can reconcile them in one place.

**C1 — CORRECTION (scope, not fact). `FLASHINFER_MLA_SPARSE_SM120` was dismissed
too early, but the dismissal's CONCLUSION survives for a different reason.**
The MLA campaign spike says at its L388-391 that the second sm_121 MLA backend is
"rejected on `is_sparse()` True vs `use_sparse=False`", and at L142-146 puts sparse
MLA / DSA out of scope. That reasoning is correct *for a dense model* but it is not
a statement about DSA. For a DSA model `use_sparse=True`, the XOR filter at
`vllm/v1/attention/backend.py:345-350` ELIMINATES `TRITON_MLA` instead — `TritonMLABackend`
(`vllm/v1/attention/backends/mla/triton_mla.py:81-131`) has zero indexer/topk handling
and inherits `is_sparse() == False` (`backend.py:253-254`). So on sm_121 the sparse
backend is not merely "also present", it is **the sole candidate** for DSA, and there
is no non-flashinfer fallback. The campaign spike's framing ("sparse-only, out of
scope") understates this: it is not an optional extra, it is the entire DSA path.
**However**, §0.2 below shows the path is BROKEN on the flashinfer we have, so the
practical out-of-scope call stands. The correction is to the reasoning and to the
hardware verdict's basis, not to the plan.

**C2 — EXTENSION (material, plan-shaping). The MLA campaign's statement that
DeepSeek-V2-Lite is the ONLY MLA gate vehicle, with the `q_lora_rank` and
`noaux_tc` router paths unit-gated only, is now superseded by a second vehicle
that FITS GB10 and closes BOTH named coverage gaps.**
The campaign spike §12 (L1106-1120) names two gaps in its single gate vehicle:
(a) DeepSeek-V2-Lite has `q_lora_rank=null`, so `fused_qkv_a_proj` /
`q_a_layernorm` / `q_b_proj` are unit-gated only; (b) it has `n_group=topk_group=1`
with softmax/greedy routing and **no `e_score_correction_bias`**, so the whole
`noaux_tc` router is unit-gated only.
**`zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`) closes both.** Verified from
its live `config.json` (fetched 2026-07-21, metadata only, nothing downloaded):
`q_lora_rank: 768` (non-null -> gap (a) closed e2e), `topk_method: "noaux_tc"` ->
`e_score_correction_bias` present (gap (b) closed e2e), `kv_lora_rank: 512`,
`qk_nope_head_dim: 192`, `qk_rope_head_dim: 64`, `v_head_dim: 256`,
47 layers, 64 routed experts + 1 shared, `num_experts_per_tok: 4`,
`first_k_dense_replace: 1`, `routed_scaling_factor: 1.8`, `n_group: 1`,
`topk_group: 1`, `num_nextn_predict_layers: 1`, **no `index_topk`** (so
`is_v32 == False`: this is DENSE MLA, not DSA). Size: **31.2B params, 58.2 GiB
bf16** — fits GB10's ~119 GiB unified memory with room for KV, and fits the
184 GiB free on dgx. It is upstream-registered at `registry.py:115` and in the
upstream test registry at `tests/models/registry.py:292-295`.
This does not change the campaign's W-order; it adds a SECOND gate vehicle at W8+
that raises the campaign's own correctness coverage. See §0.4 and W5 below.

**C3 — CORRECTION (fact, small). Free disk on dgx is 184 GiB, not 238 GiB.**
Measured 2026-07-21: `/dev/nvme0n1p2 3.6T 3.3T 184G 95% /`. The campaign spike's
L1163-1169 blocked-row reasoning cites 238 GiB of free disk. The conclusion is
unaffected (everything it called blocked is still blocked, by more), but the
number should not be re-cited. Also verified on dgx: **no GLM checkpoint of any
kind is present** in `~/.cache/huggingface/hub/`. Per the standing lesson that
`breadth-sweep-plan.md` §B.1's "present" column was wrong for OPT, presence was
checked directly rather than read from a table. `deepseek-ai/DeepSeek-V2-Lite`
**is** present (the campaign's gate vehicle).

### 0.2 DSA on GB10: the verdict is NO, and the reason is not the vLLM layer

DSA (DeepSeek Sparse Attention) is what DeepSeek-V3.2 and GLM-5.x use. The gate is
duck-typed: `self.is_v32 = hasattr(config, "index_topk")` at
`vllm/model_executor/models/deepseek_v2.py:1075`. vLLM's selection layer on sm_121
does offer and would select `FLASHINFER_MLA_SPARSE_SM120` — the backend's own gates
all pass for a real V3.2 config: `capability.major == 12`
(`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:172-173`), bf16 (`:147`),
block size in `[64, 256]` (`:162`), head size 576, and `index_topk == 2048` exactly
(`:208-219`).

**But the capability probe is a liar and the call underneath does not work.**
`has_flashinfer_sparse_mla_sm120()` (`vllm/utils/flashinfer.py:216-231`) only checks
that three symbols are importable; it never checks the function vLLM actually calls.
The impl calls `flashinfer_trtllm_batch_decode_with_kv_cache_mla`
(`flashinfer_mla_sparse_sm120.py:141-156`), which on sm12x dispatches to flashinfer's
**XQA** backend (`flashinfer/mla/_core.py:1169-1172`: `backend = "trtllm-gen" if
major == 10 else "xqa"`, and vLLM never passes `backend=`). Four independent hard
failures follow on the flashinfer available here (0.6.12 at
`/home/mudler/_git/flashinfer-ref`, vs vLLM's pinned 0.6.13 in
`requirements/cuda.txt:13-14`):

1. `kv_scale_format` is passed by vLLM (`flashinfer_mla_sparse_sm120.py:155`) but is
   not a parameter of `_core.py:1054-1077` and there is no `**kwargs` -> `TypeError`.
2. XQA is **dense-only**: it discards `sparse_mla_top_k` and hardcodes `0` into the
   shape checker (`_core.py:1483`). The sparse branch (`_core.py:179-184`) is
   unreachable via XQA.
3. dtype rejection: vLLM passes `kv_cache.view(torch.uint8)`
   (`flashinfer_mla_sparse_sm120.py:143`); XQA requires bf16 or fp8_e4m3 for both
   query and kv (`_core.py:1186-1192`).
4. `seq_lens=None` is passed (`:149`); XQA does `seq_lens.unsqueeze(1)`
   unconditionally (`_core.py:1510`).

Confidence signal from upstream itself: the only test of this path,
`tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py:37`, **monkeypatches the
capability probe to `True`** and asserts only that `validate_configuration` returns
`[]`. There is no numerical or e2e test of the sm120 sparse kernel anywhere.

**Verdict: GB10/sm_121 cannot run DSA end-to-end today, in vLLM or (therefore) as a
gateable mirror target.** This is a DEPENDENCY gap (flashinfer needs sparse top-k
plumbed through its sm12x XQA backend), not a vLLM-layer gap and not something we
close by porting harder. It is recorded as an upstream-watch item, not as work.

### 0.3 What DSA actually is (so the port is designed, not guessed)

DSA = dense MLA **plus a second, much smaller attention-like module** (the "indexer" /
"Lightning Indexer") whose only job is to pick which `index_topk` past tokens each
query attends to. Delta over dense MLA, per non-skipped layer:

- **Extra weights** (`deepseek_v2.py:642-817`, prefix `…self_attn.indexer.`):
  `wq_b` `ReplicatedLinear(q_lora_rank -> index_n_heads*index_head_dim)` (`:666-672`,
  replicated, no TP); `wk` + `weights_proj` fused by vLLM into one
  `wk_weights_proj MergedColumnParallelLinear(hidden -> [head_dim, n_head])`
  (`:675-682`, mapping at `:1521-1526`); `k_norm` LayerNorm(128) (`:683`).
  `softmax_scale = head_dim**-0.5` (`:684`) plus a head scale `n_head**-0.5` (`:813`).
- **Extra KV cache**: `DeepseekV32IndexerCache` (`:613-639`), an `AttentionLayerBase`
  that self-registers into `compilation_config.static_forward_context` (`:624-626`)
  and returns an `MLAAttentionSpec` with `num_kv_heads=1` (`:628-634`). Row width is
  `head_dim + head_dim//quant_block_size*4` uint8 (`:693-698`) = **132 B/token**
  (128 fp8 values + one fp32 scale), against 656 B/token for the main `fp8_ds_mla`
  MLA cache. It lands in its OWN kv-cache group because `MLAAttentionSpec.merge`
  (`vllm/v1/kv_cache_interface.py:399-429`) forces group-wide agreement.
- **Extra selection plumbing**: one model-wide
  `topk_indices_buffer: int32[max_num_batched_tokens, index_topk]` allocated once
  (`:1361-1368`), threaded to `MLAAttention` via `extra_impl_args` when `use_sparse`
  (`vllm/model_executor/layers/attention/mla_attention.py:457-459`), mutated in place
  by the custom op (`sparse_attn_indexer.py:678`), read by the sparse backend
  (`flashinfer_mla_sparse_sm120.py:116`). `-1` is the padding sentinel (`:397-398`).
- **Extra kernels** (`vllm/model_executor/layers/sparse_attn_indexer.py`, 826 lines):
  a Triton fused q-RoPE+ue8m0-quant (`:126-250`); `indexer_k_quant_and_cache` and
  `cp_gather_indexer_k_quant_cache` in `csrc/libtorch_stable/cache_kernels.cu:550`
  and `:613`; `top_k_per_row_{prefill,decode}` in `csrc/libtorch_stable/sampler.cu:545,569`;
  `persistent_topk` / `cooperative_topk` (`csrc/libtorch_stable/topk.cu`,
  `cooperative_topk.cu`); and **the logits GEMMs live in vendored DeepGEMM**, not
  in csrc — `fp8_fp4_mqa_logits` (`:466-473`) and `fp8_fp4_paged_mqa_logits`
  (`:562-571`). CUDA construction hard-fails without DeepGEMM (`:728-732`).
- **Extra attention semantics**: sparse impls force ALL tokens through `forward_mqa`
  with `num_mha_tokens = 0` (`mla_attention.py:697-702`) — no prefill/decode split,
  no chunked-prefill MHA path. KV dtype is auto-canonicalized to `fp8_ds_mla`
  (`mla_attention.py:323-337`).
- **Extra config**: `index_topk` (the gate), `index_n_heads`, `index_head_dim`, and
  optionally `index_topk_freq`, `index_topk_pattern`, `index_skip_topk_offset`,
  `indexer_rope_interleave`, `index_share_for_mtp_iteration`.
- **Per-layer skip**: `deepseek_v2.py:1079-1103` — layers may reuse the previous
  layer's selection (`_skip_topk`), and their checkpoint indexer weights are
  silently dropped (`:1553-1569`). MTP layers ALWAYS build an indexer (`:1100-1103`).
- **MTP interaction**: the proposer overwrites the draft's buffer with the target's
  (`vllm/v1/spec_decode/llm_base_proposer.py:1548-1562`);
  `index_share_for_mtp_iteration` runs the indexer once per draft chain
  (`:1564-1575`, `:566-570`, `:596-600`).

### 0.4 The GLM family reduces to three genuinely new things

Everything else is a subclass or a composition of parts we already have or already
plan. In dependency order:

1. **`Glm4MoeLiteForCausalLM` is DeepSeek-V2 with GLM's MoE block bolted in.**
   `glm4_moe_lite.py:94-95` and `:98-99` are literal zero-override subclasses of
   `DeepseekV2Attention` / `DeepseekV2MLAAttention`; the decoder layer, model, and
   `load_weights` (incl. the `fused_qkv_a_proj` merge at `:330-335`, `:544-551`) are
   structural copies of deepseek_v2. The only GLM-specific piece is
   `Glm4MoeLite = Glm4MoE` (`:86-87`). **If the MLA campaign lands, this is nearly
   free** — and per C2 it is the better gate vehicle.
2. **`GlmMoeDsaForCausalLM` is DeepSeek-V3.2, verbatim.** `deepseek_v2.py:1917-1918`
   is `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`. The only
   behavioural special case anywhere is fp32 router dtype forced by
   `model_type == "glm_moe_dsa"` (`deepseek_v2.py:120-130`), because older GLM-5
   configs omit `moe_router_dtype`. Two numerical deltas live in the newer tree:
   interleaved (adjacent-pair) indexer RoPE vs DeepSeek's NeoX split-half
   (`vllm/models/deepseek_v32/nvidia/kernels.py:300,697`;
   `nvidia/attention.py:297`) and `index_topk_freq=4` (`nvidia/attention.py:206`).
   Confirmed in the live `zai-org/GLM-5` config: `indexer_rope_interleave: true`,
   `index_topk: 2048`, `index_n_heads: 32`, `index_head_dim: 128`.
3. **`Glm4ForCausalLM` needs two primitives we do not have: partial rotary and
   sandwich norms.** `glm4.py:180-187` gives the decoder layer FOUR RMSNorms —
   `input_layernorm`, `post_attention_layernorm`, plus `post_self_attn_layernorm`
   and `post_mlp_layernorm` applied to the sublayer OUTPUT before the residual add
   (`:206`, `:211`), the Gemma2 sandwich pattern. Attention is GQA with
   `partial_rotary_factor` 0.5 and `is_neox_style=False` (`:86-92`, `:119`), no
   QK-norm. Verified live: `GLM-4-9B-0414` has `partial_rotary_factor: 0.5`,
   `attention_bias: true`, `num_key_value_heads: 2`, 40 layers, hidden 4096.
   **We have neither primitive today** — no `partial_rotary_factor` anywhere in
   `src/`/`include/`, and zero hits for `post_self_attn_layernorm` /
   `post_mlp_layernorm`.

`GlmForCausalLM` (`glm.py`, 24 lines) is `LlamaForCausalLM` plus three
post-construction deltas (`:14`, `:22-24`): `partial_rotary_factor = 0.5`,
`is_neox_style = False`, and `o_proj` bias dropped with `skip_bias_add`. It shares
exactly one new primitive with Glm4 (partial rotary) and needs no sandwich norms.
It is also double-registered as an embedding model (`registry.py:218` — an entry
missing from the task's list, tracked by the separate `MODEL-EMBED-glm-glm-for-causal-lm`
row, which this spike does NOT claim).

`Glm4MoeForCausalLM` is **Qwen3-MoE attention + DeepSeek-V2 router**: GQA with
optional QK-norm (`glm4_moe.py:305-307`, `:316-322`) and partial NeoX rope
(`:289`), but a router that is a near-verbatim `DeepseekV2MoE` port — fp32
`nn.Linear` gate (`:147-152`), `e_score_correction_bias` (`:153-155`),
`scoring_func="sigmoid"` (`:204`), `use_grouped_topk=True` (`:200-202`),
`routed_scaling_factor` (`:206-207`), `first_k_dense_replace` (`:362-379`).
The file's own comments say so at `:172-173`, `:205`, `:486`.

`ChatGLMForCausalLM` is the legacy ChatGLM2/3 lineage: hand-written attention with
`multi_query_group_num` (`chatglm.py:60-63`) and a hardcoded partial rotary factor
of 0.5 (`:103`, applied `:131`), on an out-of-tree config requiring
`trust_remote_code` (`:34`). It shares the partial-rotary primitive and nothing else.

### 0.5 Hardware fit — measured, per variant

Sizes fetched from the HF API 2026-07-21 (metadata only; **nothing downloaded**).
GB10 budget: ~119 GiB unified memory. dgx free disk: **184 GiB**.

| Variant | Smallest genuine checkpoint | Params | bf16/native on disk | GB10 verdict |
|---|---|---|---|---|
| `Glm4MoeLiteForCausalLM` (MLA+MoE) | `zai-org/GLM-4.7-Flash` | 31.2B | **58.2 GiB** | **FITS — recommended gate vehicle** |
| `Glm4ForCausalLM` (dense, sandwich) | `zai-org/GLM-4-9B-0414` | 9.4B | **17.5 GiB** | **FITS** |
| `GlmForCausalLM` (Llama + partial rope) | `zai-org/glm-4-9b-chat-hf` | 9.4B | **17.5 GiB** | **FITS** |
| `ChatGLMModel` (legacy) | `zai-org/chatglm3-6b` | 6.2B | 23.3 GiB (fp16) | FITS (needs `trust_remote_code` config) |
| `Glm4MoeForCausalLM` (GQA+DS router) | `zai-org/GLM-4.5-Air` | 110.5B | **205.8 GiB** | **HW-BLOCKED at bf16.** No smaller genuine checkpoint exists. `zai-org/GLM-4.5-Air-FP8` is 104.8 GiB and would be HW-MARGINAL, but depends on an fp8 checkpoint-loading row we do not own |
| `GlmMoeDsaForCausalLM` (DSA) | `zai-org/GLM-5` | 753.9B | **1404.2 GiB** | **HW-BLOCKED — by ~12x memory and ~7.6x disk.** Also DSA-blocked per §0.2 |
| `DeepseekV4ForCausalLM` | `deepseek-ai/DeepSeek-V4-Flash` | 158.1B stored | **148.7 GiB** (fp4 experts) | **HW-BLOCKED e2e** — fits the 184 GiB disk, does NOT fit 119 GiB memory |
| DeepSeek-V3.2 (row owned by MLA campaign) | `deepseek-ai/DeepSeek-V3.2-Exp` | 685.4B | **642.1 GiB** (fp8) | **HW-BLOCKED** — exceeds memory and disk both |

Multimodal GLM rows (not claimed by this spike, inventory only): `zai-org/GLM-OCR`
1.3B/2.5 GiB and `zai-org/GLM-ASR-Nano-2512` 2.3B/4.2 GiB are the two smallest
checkpoints in the entire GLM family and both fit trivially — but they need the
vision/audio tower tracks, which we have not started.
`zai-org/GLM-4.1V-9B-Thinking` and `GLM-4.6V-Flash` are 19.2 GiB (dense Glm4
backbone); `zai-org/GLM-4.5V` is 200.6 GiB (HW-BLOCKED).

**No proposed gate depends on hardware we do not have.** Every checkpoint named as
a gate vehicle below fits, mirroring the discipline the MLA campaign spike applied
to V3/K2.5/M2.

### 0.6 DeepSeek V3.2 and V4 — what is genuinely new

**V3.2 = V3 + DSA.** Nothing else. Critically, **`vllm/models/deepseek_v32/` is
unregistered dead code**: `registry.py:93` maps `DeepseekV32ForCausalLM` to
`("deepseek_v2", "DeepseekV3ForCausalLM")` — the OLD tree — and a full-repo grep
finds nothing outside the new package importing it. Only its Triton kernels are
live-tested (`tests/kernels/test_fused_deepseek_v32_norm_rope.py:30`). Port
`deepseek_v2.py` for semantics; read `deepseek_v32/nvidia/attention.py` only as a
better-commented restatement.

**V4 is a different architecture, not an increment.** `registry.py:94` DOES point at
the new tree (`vllm.models.deepseek_v4`), resolved by
`registry.py:1382-1390` (a value starting with `vllm.` is a fully-qualified module).
Four subsystems are redesigned, ranked by porting cost:

- **Manifold Hyper-Connections (MHC) — the highest risk item in either family.**
  The residual stream stops being a vector: `hidden_states.unsqueeze(-2).repeat(1, hc_mult, 1)`
  (`vllm/models/deepseek_v4/nvidia/model.py:1066`), `hc_mult = 4`. Six extra fp32
  params per sublayer (`:817-861`), and the mixing weights are **Sinkhorn-normalized
  at runtime, 20 iterations, every forward** (`hc_sinkhorn_iters`), inside TileLang
  kernels that also swallow the RMSNorms (`:878`, `:894`, `:986`, `:1091`, `:1113`;
  the comment at `:892` confirms `attn_norm` is fused in and never applied standalone).
  There is **no eager reference path in the repo and no upstream numerical test** of
  the MHC kernels. Porting V4 means building our own reference for its most novel part.
- **CSA/HCA compressor + Lightning Indexer replace V3.2's MLA.**
  `DeepseekV4Attention` (`attention.py:97`) does NOT subclass `MLAAttention`. No
  `kv_lora_rank`, no nope/v split; instead a grouped **output** LoRA
  (`wo_a`/`wo_b`, `o_groups`/`o_lora_rank`), per-head fp32 attention sinks
  (`:194-197`), and sliding window 128 with its own `DeepseekV4SWACache` (`:288`).
  Per-layer `compress_ratio` alternates 4 (CSA, gets an indexer — `:246-248`) and
  128 (HCA, compressor only). `DeepseekCompressor` (`compressor.py:177`) carries a
  **stateful fp32 recurrent `CompressorStateCache`** (`:121`) paged through the KV
  manager via its own `CompressorBackend` (`:37`) — a genuinely new cache kind.
- **MoE: `sqrtsoftplus` scoring** (`nvidia/model.py:541`), **hash-routed layers**
  (the first `num_hash_layers = 3` have no learned gate at all, routing via a
  `tid2eid[vocab_size, top_k]` lookup on the token id — `:563-577`, which is why
  `input_ids` is threaded into the FFN at `:938`), and clamped SwiGLU
  (`SiluAndMulWithClamp(swiglu_limit)`, `:130-133`). Experts default to fp4
  (`:546`).
- **Dual-theta RoPE** with YaRN mscale explicitly disabled (`common/rope.py:18-20`,
  `:27-28`) — 36 lines, and easy to get silently wrong.

**Tokenizer risk: LOW, and this is the answer to the OPT lesson.** The concern was
that a new tokenizer format would silently mis-encode and score 0/6 while emitting
fluent text. It is not a new format: `vllm/tokenizers/deepseek_v4.py:95` is literally
`PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)` — ordinary HF fast BPE
over a standard `tokenizers.json`. Corroborated empirically by the existence of
`unsloth/DeepSeek-V4-Flash-GGUF` (13 quant variants), i.e. the vocab already
round-trips through GGUF conversion. What IS new is the ~700-line hand-written
**chat template** replacing Jinja (`deepseek_v4_encoding.py`, marked
`# ruff: noqa` / `# fmt: off` at `:3-4`, i.e. vendored verbatim from DeepSeek):
BOS is prepended once and only when context is empty (`:551`); EOS is appended per
assistant turn by template concatenation (`:50`, with a no-EOS variant at `:51` for
the trailing generation prefix); plus tool-result reordering (`:407`, `:466`),
thinking-history dropping (`:581`), a DSML tool-call markup dialect (`:57-62`,
`:145`, `:175`), and reasoning-effort injection (`deepseek_v4.py:43-52`).
`vllm/renderers/deepseek_v4.py` is a thin 91-line delegator with no tokenization
logic. **Upstream ships golden fixtures** —
`tests/tokenizers_/test_deepseek_v4.py:286
::test_deepseek_v4_matches_reference_golden_fixtures` — so the BOS/EOS placement
that bit us on OPT is directly and offline-gateable here. This is explicitly NOT the
critical path; MHC and the compressor are.

### 0.7 Recommended order

The shared unlock lands first on hardware we actually have:

```
  [MLA campaign W1..W8, DeepSeek-V2-Lite]      <- already running, not ours
        |
        +--> G1  Glm4MoeLite / GLM-4.7-Flash  (pure ADD on dense MLA; FITS;
        |        closes the campaign's OWN two router/q_lora coverage gaps)
        |
  independent, no MLA dependency, can run in parallel:
        +--> G2  partial-rotary + sandwich-norm primitives
        |         -> Glm4 (GLM-4-9B-0414) and Glm (glm-4-9b-chat-hf); both FIT
        |
  blocked / deferred, honesty passes only:
        +--> G3  Glm4Moe        HW-BLOCKED at bf16 (205.8 GiB)
        +--> G4  DSA + GlmMoeDsa DEP-BLOCKED (flashinfer) and HW-BLOCKED (1404 GiB)
        +--> G5  DeepSeek-V4    HW-BLOCKED e2e (148.7 GiB > 119 GiB)
```

G1 is the highest-value item in this spike and the only one that both adds coverage
to the running campaign and fits. G2 is fully independent of MLA and unblocks four
rows' worth of shared primitives (Glm, Glm4, ChatGLM, and later MiniMax-M2's partial
RoPE). G3/G4/G5 are honesty passes, not implementation.

---

## 1. Structured contract

### Scope

Design — not build — the GLM model family end to end, the DSA (sparse MLA) surface
shared by DeepSeek-V3.2 and GLM-5.x, and the two newest DeepSeek architectures
(V3.2, V4), and determine honestly what is gateable on GB10. This spike covers the
rows `MODEL-TEXT-chatglm-chat-glmfor-causal-lm`, `MODEL-TEXT-glm-glm-for-causal-lm`,
`MODEL-TEXT-glm4-glm4-for-causal-lm`, `MODEL-TEXT-glm4-moe-glm4-moe-for-causal-lm`,
`MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm`,
`MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm`, and
`MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm`.

In scope: the complete GLM registry inventory with per-architecture characterization;
the two new shared primitives GLM needs (partial rotary factor, sandwich norms); the
full DSA delta over dense MLA (indexer weights, indexer KV cache, top-k selection
plumbing, kernels, config, MTP interaction); the sm_121 sparse-MLA backend verdict
with its dependency-chain grounding; V3.2-vs-V3 and V4-vs-V3 architectural deltas
including the V4 tokenizer/renderer risk assessment; measured per-variant hardware
fit against GB10 and dgx disk; the shared-vs-independent factoring against the
running MLA campaign; and the upstream test inventory for all of the above.

OUT of scope, each with a reason: **implementation of anything** (this is a spike —
no code, no kernels, no build, no benchmark). **DSA implementation**, because §0.2
shows the only sm_121 backend that can run it is non-functional on the flashinfer
in this environment — it is a dependency gap we watch, not work we can gate.
**DeepSeek-V4 implementation**, because at 148.7 GiB it does not fit GB10's 119 GiB
and its most novel subsystem (MHC/Sinkhorn) has no upstream numerical test to gate
against. **`Glm4MoeForCausalLM` e2e**, because its smallest genuine checkpoint is
205.8 GiB bf16; the 104.8 GiB FP8 variant depends on an fp8 checkpoint-loading row
this spike does not own. **All GLM multimodal rows** (`glm4v`, `glm4_1v`, `glm_ocr`,
`glmasr`), because the vision/audio tower tracks have not started; they stay
`INVENTORIED`. **All GLM MTP rows** (`Glm4MoeMTPModel`, `Glm4MoeLiteMTPModel`,
`GlmOcrMTPModel`), because speculative decoding is a separate track; inventoried
only, they stay `INVENTORIED`. **`MODEL-EMBED-glm-glm-for-causal-lm`**, because
pooling/embedding tasks are a separate modality row. **DeepSeek-V3.2's own row**
(`MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm`), because it is already `SPIKE`
under `CLAIM-MLA-DEEPSEEK` and this spike must not collide with it — the DSA
findings here are supplied to that row's owner as cross-referenced input.

### Upstream chain

Registry, all in `vllm/model_executor/models/registry.py` @ `e24d1b24`:
`:82-83` ChatGLM (`chatglm.py::ChatGLMForCausalLM`); `:112` `GlmForCausalLM`
(`glm.py`); `:113` `Glm4ForCausalLM` (`glm4.py`); `:114` `Glm4MoeForCausalLM`
(`glm4_moe.py`); `:115` `Glm4MoeLiteForCausalLM` (`glm4_moe_lite.py`); `:116`
`GlmMoeDsaForCausalLM` -> `("deepseek_v2", "GlmMoeDsaForCausalLM")`; `:94`
`DeepseekV4ForCausalLM` -> `("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")`.
Also `:218` (Glm as embedding), `:397-401` (multimodal), `:620-622` (MTP),
`:592`/`:612` (V4 DSpark/MTP). Fully-qualified-module resolution: `:1382-1390`,
with the module-hash special case at `:918-931`.

Model layer: `glm.py:11,14,22-24`; `glm4.py:55,86-92,111,119,180-187,206,211,295-305`;
`glm4_moe.py:121,136-137,147-155,172-173,200-211,232,284-322,355,362-384,501-665,668`;
`glm4_moe_lite.py:82-99,120,127-135,145,157,172-175,223-231,330-335,544-551`;
`chatglm.py:34,46,60-63,103,131,137,184,261`;
`deepseek_v2.py:120-130,613-639,642-817,820-858,1075-1125,1143-1144,1359-1368,1385,1521-1527,1553-1569,1917-1918`.

Attention/selection: `vllm/model_executor/layers/mla.py:95,169-180`;
`vllm/model_executor/layers/attention/mla_attention.py:323-337,404-411,457-459,697-702`;
`vllm/platforms/cuda.py:100-142` (sparse candidates), `:130-134` (the cc-12 list);
`vllm/v1/attention/backend.py:253-254,345-350` (the sparse XOR filter);
`vllm/v1/attention/backends/registry.py:76-79,88,91,95`;
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:65-67,144-219,228-231`;
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:50-58,65-69,98,116,141-156`;
`vllm/v1/attention/backends/mla/triton_mla.py:81-131`;
`vllm/v1/attention/backends/mla/indexer.py:37-68,69-115,126-157,200-206,224-235,238-303,360-364`;
`vllm/utils/flashinfer.py:216-231`.

Indexer kernels: `vllm/model_executor/layers/sparse_attn_indexer.py:43-44,47-70,83-92,101-123,126-250,253-291,294-339,373-398,422-484,512-521,552-624,639-642,675-681,684-826`;
`csrc/libtorch_stable/cache_kernels.cu:550,594-608,613,1461,1508`;
`csrc/libtorch_stable/sampler.cu:545,569,661,723`; `csrc/libtorch_stable/topk.cu`;
`csrc/libtorch_stable/cooperative_topk.cu:65,69,131`; bindings
`csrc/libtorch_stable/torch_bindings.cpp:478-493,684-688,837-844,904-907`.
KV grouping: `vllm/v1/kv_cache_interface.py:380-397,399-429`;
`vllm/v1/core/single_type_kv_cache_manager.py:1539`;
`vllm/v1/worker/gpu_model_runner.py:7466-7500`.

DeepSeek-V4: `vllm/models/deepseek_v4/nvidia/model.py:130-133,160,303,511,541,546,563-582,817-861,866-939,1066,1091,1113,1370`;
`attention.py:97,170-234,181-185,194-197,246-248,288,306-315,643-655,663,691`;
`compressor.py:37,121,177,184-188`; `quant_config.py:29`; `sparse_mla.py:37`;
`common/rope.py:9-36`; config `vllm/transformers_utils/configs/deepseek_v4.py:8,13-22`.
Tokenizer/renderer: `vllm/tokenizers/deepseek_v4.py:25-88,43-52,92,95`;
`vllm/tokenizers/deepseek_v4_encoding.py:3-4,17,23-41,50-51,57-62,76,145,175,229,407,466,551,557-559,581,612,636,693`;
`vllm/renderers/deepseek_v4.py:23,35-36,51-55`; registration
`vllm/tokenizers/registry.py:41-42`, `vllm/renderers/registry.py:23-24`,
`vllm/config/model.py:80,612-616`.
GLM-5 numerics in the newer tree: `vllm/models/deepseek_v32/__init__.py:5-8,13-14`;
`nvidia/attention.py:39,160,206,297,373`; `nvidia/kernels.py:300,697`.
Speculative allowlist: `vllm/config/speculative.py:39-40,326,412,423`.
Arch-config conversion: `vllm/transformers_utils/model_arch_config_convertor.py:265-267,649`.
Dependency chain outside vLLM: flashinfer `mla/_core.py:171-177,179-184,462-620,1054-1077,1169-1172,1186-1192,1470-1476,1483,1510`
(observed at `/home/mudler/_git/flashinfer-ref`, version 0.6.12 per `_build_meta.py:2-3`,
against vLLM's pin 0.6.13 in `requirements/cuda.txt:13-14`); DeepGEMM supplies the
indexer logits GEMMs; TileLang supplies the V4 MHC kernels.

### Our baseline

Landed seams this work would reuse as-is:
`src/vt/cuda/cuda_arch_tactics.h` + `.cu` (`TacticFamily` `:52`,
`ArchTacticSupportsFn` `:83`, `ArchTactic` `:86`, `RegisterArchTactic` `:96`,
`SelectArchTactic` `:100`) — the additive registration point for any new decode
tactic. `vt::MoeGroupedGemmBf16` (`include/vt/ops.h:158,384,789`;
`src/vt/ops.cpp:582-604`; impl `src/vt/cuda/cuda_matmul_nvfp4.cu`), which already
runs ~1.2x vLLM's Triton `fused_moe` and is deterministic (fixed-ascending split-K,
never `atomicAdd`) — directly reusable for every GLM MoE variant.
`RunMoeBlock` (`include/vllm/model_executor/models/qwen3_5_moe_block.h:45`), the
cross-TU bf16 MoE block with router + experts + optional shared expert — GLM's MoE
has a shared expert (`n_shared_experts: 1` in all three live configs), so this is a
direct fit. The decode CUDA-graph driver pattern, which now has three siblings
(`Qwen3MoeDecodeGraph` `include/vllm/model_executor/models/qwen3_moe.h:117` +
`src/vllm/model_executor/models/qwen3_moe.cpp:333-462`; `Qwen3_5DecodeGraph`
`src/vllm/model_executor/models/qwen3_5.cpp:5902`; `Qwen3_5DenseDecodeGraph`
`include/vllm/model_executor/models/qwen3_5_dense.h:238`) over shared sizes in
`include/vllm/model_executor/models/decode_graph_sizes.h` — a fourth is a pattern
application, not new design. The model-factory seam (`REGISTER_VLLM_MODEL`, 5 models
registered today) and the runner generalization `ENG-RUNNER-MODELSHAPE`.

Honestly NOT reusable, and why:
**`dense_attn_block.h`** (`include/vllm/model_executor/models/dense_attn_block.h`,
556 lines) — its `AttnBlock` hard-codes q/k RMSNorm + NeoX RoPE and asserts
`VT_CHECK(qkv_bias.Empty())`. The OPT row already proved this does not stretch
across families and deliberately did not reuse it. GLM needs biased qkv
(`attention_bias: true` in both 9B configs), partial + non-NeoX rope, and sandwich
norms — three independent violations. Only its glue (`Dev`/`DBuf`/`DevicePool`/
`ResidentWeight`/`KvSlice`/`BuildStepInputs`) and its GQA paged path (`:97`, `:387`,
`:505`) are reusable.
**The MoE router.** `vt::MoeRouterTopK` (`include/vt/ops.h:100,334-341,461-462,1168-1169`;
`src/vt/cuda/cuda_moe.cu:62,212,422-423,438`) has exactly TWO fields — `top_k` and
`renormalize`. No sigmoid scoring, no `e_score_correction_bias`, no group masking,
no `routed_scaling_factor`. Every GLM MoE variant needs all four. This is the same
gap the MLA campaign spike identified in its §3.1; it is shared work, and whoever
lands it first satisfies both campaigns.
**MLA: we have zero.** Only four deferral comments exist —
`src/vllm/platforms/cuda.cpp:54-56`, `include/vllm/v1/kv_cache_interface.h:26,44`,
`include/vllm/v1/attention/backend.h:28` — plus a non-MLA test title at
`tests/vllm/v1/attention/test_attn_backend_registry.cpp:121`. Everything MLA in
this spike is therefore downstream of the MLA campaign's W1-W7.
**Partial rotary: we have none.** No `partial_rotary_factor` anywhere in `src/` or
`include/`; `rotary_dim` exists only inside
`src/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.cpp` as a
full-head parameter for YaRN, with no path that ropes a leading slice and passes the
tail through. New work, shared by GLM, ChatGLM and MiniMax-M2.
**Sandwich norms: we have none.** Zero hits for `post_self_attn_layernorm`,
`post_mlp_layernorm`, or `sandwich` in `src/` or `include/`. Fully new.
**No MODEL row is `DONE`** anywhere in the matrix today; the best states are
`ACTIVE` with correctness complete and speed pending (OPT, Qwen3-dense,
Qwen3-Coder). Nothing in this spike claims otherwise.

Precedent specs: [`mla-deepseek-campaign.md`](mla-deepseek-campaign.md) (the parent
MLA work this depends on), [`sweep-qwen3-coder-30b.md`](sweep-qwen3-coder-30b.md)
(BF16 MoE bring-up shape), [`sweep-opt-125m.md`](sweep-opt-125m.md) (the
cross-family additivity canary and the BOS/tokenizer lesson),
[`first-additive-model-qwen3-dense.md`](first-additive-model-qwen3-dense.md).

**Anchor-drift warning.** Line anchors quoted in the MLA campaign spike have moved
since its base `b4f14ee`. Verified current values in this tree:
`src/vllm/v1/worker/gpu/runner.cpp` — the hardcoded factor-2 KV allocation is at
`:489-491` (campaign cites `:487-492`); `include/vt/ops.h` — `ReshapeAndCache`
`:1227` (cited `:1167`), `PagedAttention` `:1261` (cited `:1201`),
`MoeRouterTopKArgs` `:334` (cited `:311-318`), `MoeRouterTopK` `:1168` (cited
`:1108`). Re-anchor before citing; `check_links` validates line ranges.

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:112` `GlmForCausalLM` (`glm.py:11,14,22-24`) | **NEW** `src/vllm/model_executor/models/glm_registry.cpp` — one `REGISTER_VLLM_MODEL`, full-attention KV spec, `is_dense_model=true` |
| `registry.py:113` `Glm4ForCausalLM` (`glm4.py:55,180-187`) | **NEW** `include/vllm/model_executor/models/glm4.h` + `src/vllm/model_executor/models/{glm4_registry,glm4_weights,glm4}.cpp` |
| `registry.py:114` `Glm4MoeForCausalLM` (`glm4_moe.py:121,232`) | **NEW** `glm4_moe.{h,cpp}` — composes the new GLM attention block + `RunMoeBlock` + the extended router. HW-BLOCKED e2e; loader/unit only |
| `registry.py:115` `Glm4MoeLiteForCausalLM` (`glm4_moe_lite.py:82-99`) | **NEW** `glm4_moe_lite.{h,cpp}` — MLA attention from the campaign's W6 block + `RunMoeBlock` + extended router. **The gate vehicle** |
| `registry.py:116` `GlmMoeDsaForCausalLM` (`deepseek_v2.py:1917-1918`) | **NEW** alias registration over the DeepSeek-V3 TU + the `glm_moe_dsa` fp32-router special case (`deepseek_v2.py:120-130`). DEP-BLOCKED + HW-BLOCKED; registry/config resolution only |
| `registry.py:82-83` ChatGLM (`chatglm.py:46,60-63`) | **NEW** `chatglm.{h,cpp}` + out-of-tree config handling. Lowest priority |
| `glm4.py:86-92,119` partial + non-NeoX rope; `chatglm.py:103` | **NEW** `partial_rotary_factor` support in `src/vllm/model_executor/layers/rotary_embedding/` — rope a leading `rotary_dim` slice, pass the tail through, both NeoX and interleaved. **Shared primitive** |
| `glm4.py:180-187,206,211` sandwich norms | **NEW** post-sublayer norm hooks in the GLM attention/MLP block (output-normed before residual add). **Shared primitive** |
| `glm4_moe.py:147-155,200-211` sigmoid + `e_score_correction_bias` + grouped top-k + `routed_scaling_factor` | **EXTEND** `vt::MoeRouterTopKArgs` (`include/vt/ops.h:334`) and `src/vt/cuda/cuda_moe.cu:62,212`. **Shared with the MLA campaign's §3.1 router gap** — one implementation satisfies both |
| `glm4_moe_lite.py:330-335,544-551` `fused_qkv_a_proj` merge | Reuses the MLA campaign's W7 loader mapping unchanged |
| `deepseek_v2.py:642-817` `Indexer` | **NOT PORTED** — DEP-BLOCKED (§0.2). Inventoried in §0.3 so the port is designed when flashinfer lands sm12x sparse |
| `deepseek_v2.py:613-639` `DeepseekV32IndexerCache` | **NOT PORTED** — would need a second KV-cache group at 132 B/token uint8 alongside the main MLA group; recorded as the design, not built |
| `sparse_attn_indexer.py` + `csrc` top-k/cache kernels + DeepGEMM logits GEMMs | **NOT PORTED** — DEP-BLOCKED; note the logits GEMMs are in vendored DeepGEMM, not csrc, so "port the csrc kernels" would be an incomplete port |
| `vllm/models/deepseek_v4/**` (MHC, compressor, hash MoE, dual-theta rope) | **NOT PORTED** — HW-BLOCKED e2e (148.7 GiB > 119 GiB) and MHC has no upstream numerical test |
| `vllm/tokenizers/deepseek_v4.py:95` | No new tokenizer needed — standard HF fast BPE, our existing loader path |
| `vllm/tokenizers/deepseek_v4_encoding.py` (~700 lines) | **Deferred** chat-template port; offline-gateable against upstream golden fixtures when V4 becomes reachable |

### Tests to port

Per [`.agents/test-porting.md`](../test-porting.md), the upstream test modules that
are the executable spec for these rows. Nothing below is ported by this spike (it is
spec-only); this is the inventory that binds the implementing Ws.

| Upstream test | Tier | Ours |
|---|---|---|
| `tests/models/language/generation/test_common.py:66` (`zai-org/chatglm3-6b` — the ONLY GLM text-correctness entry upstream) | T-parity | `tests/vllm/models/test_chatglm_paged_engine.cpp` (deferred with the ChatGLM row) |
| `tests/models/registry.py:221-227,289-298` `_HfExamplesInfo` for every GLM arch | T-unit | config/registry resolution cases per GLM row |
| `tests/models/test_initialization.py:117-125` (skips `DeepseekV32ForCausalLM`/`GlmMoeDsaForCausalLM` below cc 9.0) | T-unit | SKIPPED with reason "DSA DEP-BLOCKED on sm_121 flashinfer" |
| `tests/config/base_model_arch_groundtruth.json:223-228` + `tests/config/test_model_arch_config.py:42` (`zai-org/GLM-4.5` -> `Glm4MoeForCausalLM`) | T-unit | arch-config resolution, gateable with NO checkpoint |
| `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py:34-39` `test_v32_glm_sm120_backend_accepts_glm_block_size` | T-unit | SKIPPED — and note it monkeypatches the capability probe, so it does not evidence a working kernel |
| `tests/v1/attention/test_mla_prefill_selector.py:260-274` `selector_config_glm5` | T-unit | MLA prefill selector cases (shared with the MLA campaign's W2) |
| `tests/kernels/test_fused_deepseek_v32_norm_rope.py:36,86` (GLM-5.2 adjacent-pair vs DeepSeek NeoX split-half indexer rope) | T-unit | SKIPPED — DSA DEP-BLOCKED; the rope-layout parametrization is the spec for the GLM-5 vs V3.2 numerical delta |
| `tests/v1/attention/test_sparse_mla_backends.py:185,591,646,720,775,787,800` | T-unit | SKIPPED — DSA DEP-BLOCKED |
| `tests/v1/attention/test_indexer_dcp_localize.py` (15 cases, deepest DSA coverage) | T-unit | SKIPPED — DSA DEP-BLOCKED; single-GPU, DCP not in scope |
| `tests/kernels/moe/test_topk_softplus_sqrt.py:49,83,139` (V4 `sqrtsoftplus` + hash routing) | T-unit | SKIPPED — V4 HW-BLOCKED |
| `tests/kernels/test_compressor_kv_cache.py:63,191,276,341,394,544,681` (V4 compressor cache) | T-unit | SKIPPED — V4 HW-BLOCKED |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py:235,288,383,452,619,712` | T-unit | SKIPPED — V4 HW-BLOCKED |
| `tests/tokenizers_/test_deepseek_v4.py:286` `test_deepseek_v4_matches_reference_golden_fixtures` (11 cases) | T-unit | **Gateable offline with no GPU and no weights** — the direct answer to the OPT BOS lesson; port when the V4 chat template is ported |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` (5 cases), `test_glm47_moe_tool_parser.py` (12 cases), `tests/reasoning/test_glm4_moe_reasoning_parser.py:14` | T-unit | Serving-layer parsers; deferred to the tool-calling track, inventoried here |
| `tests/tool_parsers/test_deepseekv4_tool_parser.py` (9 cases, DSML) | T-unit | Deferred with the V4 chat template |
| `tests/lora/test_chatglm3_tp.py:60,83,108`; `tests/distributed/test_pipeline_parallel.py:107,172` | T-unit | Out of scope (LoRA / multi-GPU tracks) — recorded, not ported |
| `tests/models/multimodal/processing/test_glm4_1v.py:14,71`; `generation/test_common.py:446-506` (glm4v/glm4_1v/glm_ocr, two `pytest.mark.skip` upstream); `processing/test_common.py:63-84,280-281` (glmasr) | T-parity | Out of scope — multimodal rows stay `INVENTORIED` |

**Upstream coverage gap to state honestly:** text-side e2e correctness for GLM is
almost nonexistent upstream — only `chatglm3-6b` appears in
`tests/models/language/generation/test_common.py`. No GLM-4, 4.5, 4.7 or 5 text
model has an upstream output-correctness test. **Our correctness oracle for
GLM-4.7-Flash must therefore be built by us**, against the pinned pip-vLLM oracle
per [`.agents/gates.md`](../gates.md), exactly as the sweep models did — it cannot
be inherited from an upstream test list.

### Gates

1. **Correctness (SACRED), `Glm4MoeLiteForCausalLM` on `zai-org/GLM-4.7-Flash`
   bf16.** Token-exact against the pinned vLLM oracle on the identical prompt set,
   greedy. Gate form selected BY MEASUREMENT per
   [`near-tie-distributional-gate`](../gates.md): run vLLM's own greedy K=5 times
   first; if vLLM is self-deterministic, the bar is STRICT token-exact — a 31.2B MoE
   is well above the small-dense near-tie regime, so STRICT is the expectation and a
   distributional fallback must be justified by measurement, not assumed.
2. **Correctness, `Glm4ForCausalLM` on `zai-org/GLM-4-9B-0414`** and
   **`GlmForCausalLM` on `zai-org/glm-4-9b-chat-hf`**, same protocol. These two are
   the gates that actually prove the partial-rotary and sandwich-norm primitives;
   a unit test alone is insufficient because a silently-unapplied rope slice emits
   fluent text (the OPT BOS failure mode).
3. **New ops.** Extended `vt::MoeRouterTopK` (sigmoid, `e_score_correction_bias`,
   group masking, `routed_scaling_factor`) unit-gated at REAL model dimensions
   against a CPU reference: GLM-4.7-Flash (64 experts, top-4, 1 shared,
   `routed_scaling_factor=1.8`, `noaux_tc`) and GLM-4.5-Air (128 experts, top-8).
   Partial rotary unit-gated at `partial_rotary_factor=0.5` in both NeoX and
   interleaved layouts, asserting the non-roped tail is passed through bit-exactly.
4. **Loader.** Weight-map coverage with zero unmapped and zero missing tensors for
   every gated checkpoint, including the `fused_qkv_a_proj` merge for the MLA
   variant and the `num_nextn_predict_layers` MTP-tail skip
   (`glm4.py:295-305`, `:308`).
5. **Regression, non-negotiable.** 27B 235/235, 35B 315/315, Qwen3-Coder 6/6,
   Qwen3-dense, OPT-125m 6/6 all UNCHANGED. Every model added by this campaign is
   additive; a regression on any of these voids the change.
6. **Build.** Clean full rebuild `-Werror`, zero warnings. Per
   [`incremental-build-masks-werror`](../workflow.md), incremental builds may report
   green while a clean build is red — header changes here are certain, so the gate is
   a clean rebuild, not an incremental one.
7. **memcheck.** `compute-sanitizer` zero errors on the new kernels.
8. **Record.** `scripts/check-agent-record.py` and
   `scripts/check-doc-checkpoint.py --staged` both green.
9. **SPEED.** Explicitly PENDING and unclaimed for every row. Per the acceptance
   rule, a model is DONE only at token-exact AND vLLM throughput on every axis; no
   row in this campaign may reach `DONE` on correctness alone.
10. **Blocked-row honesty gates.** For `Glm4MoeForCausalLM`, `GlmMoeDsaForCausalLM`
    and `DeepseekV4ForCausalLM` — where e2e is impossible — the gateable subset is:
    config/registry resolution from the real `config.json` (no weights needed);
    weight-map coverage on a downloaded SLICE (a single shard) proving the mapping is
    complete and correctly shaped; and unit parity at the REAL dimensions from the
    live config. These rows record `HW-BLOCKED` or `BLOCKED` with the measured number
    and never claim more.

### Dependencies

**Hard upward dependency on the running MLA campaign.** G1 (`Glm4MoeLite`) cannot
start before the campaign's W1 (spec-driven KV allocation, `KVCacheSpecKind::kMLA`),
W3 (`vt::ConcatAndCacheMla`), W4 (`vt::MlaDecodeAttention`), W5 (MLA prefill) and W6
(MLA attention block + weight absorption) land. This spike adds no MLA design of its
own and deliberately does not duplicate it — it consumes it. Rows:
`MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` (`SPIKE`, `CLAIM-MLA-DEEPSEEK`).

**Shared, either-order dependency: the MoE router extension.** The MLA campaign's
§3.1 and this spike's §Our baseline identify the same two-field
`vt::MoeRouterTopKArgs` gap. Whoever implements sigmoid + `e_score_correction_bias` +
group masking + `routed_scaling_factor` first unblocks the other. This must be
coordinated, not implemented twice.

**No dependency on MLA at all** for G2: partial rotary and sandwich norms are pure
dense-path primitives. G2 can proceed fully in parallel with the entire MLA campaign
and is the correct thing to start first if the campaign is mid-flight.

**Downward dependencies this introduces:** `ENG-RUNNER-MODELSHAPE` (already landed)
for any new model shape; the model-factory `REGISTER_VLLM_MODEL` seam; and, for the
MoE variants, `vt::MoeGroupedGemmBf16` + `RunMoeBlock`.

**Checkpoint dependencies (downloads, not yet performed).** `zai-org/GLM-4.7-Flash`
58.2 GiB, `zai-org/GLM-4-9B-0414` 17.5 GiB, `zai-org/glm-4-9b-chat-hf` 17.5 GiB.
Total 93.2 GiB against 184 GiB free on dgx — feasible, but only just, and the disk
is at 95%. **A download plan must free space first or stage the three sequentially**;
per [`grid-per-sha-trees-fill-disk`](../workflow.md) an ENOSPC mid-download produces
bogus test failures that look like correctness bugs. No GLM checkpoint is present
today (verified, not read from a table).

**External dependency that is BLOCKING and outside our control:** flashinfer sparse
top-k support plumbed through its sm12x XQA backend. Until that exists, DSA is not
runnable on GB10 by vLLM OR by us. Watch item on `flashinfer` releases past 0.6.13.

**No dependency on:** multi-GPU/EP, DCP (`sparse_attn_indexer.py:47-70` restricts DCP
to CuteDSL and `index_topk in {512,1024,2048}`), fp8 KV cache, quantized checkpoint
loading (all gates are bf16), speculative decoding/MTP, or any multimodal tower.

### Work breakdown

- **W0 — Ground the facts on hardware.** Free disk on dgx, download
  `zai-org/GLM-4-9B-0414` (17.5 GiB, the cheapest of the three), confirm the real
  `config.json` matches what §0.5 fetched, run the pinned vLLM oracle on it, and run
  the K=5 greedy self-determinism probe that selects the gate form per gate 1. No
  code. Cheap, and it de-risks every later W. *Gate: oracle produces reference
  outputs; determinism verdict recorded.*
- **W1 — Partial rotary factor.** Extend
  `src/vllm/model_executor/layers/rotary_embedding/` to rope a leading `rotary_dim`
  slice and pass the tail through, in both NeoX and interleaved layouts. Additive;
  no existing model sets it, so all five current models must be bit-identical.
  *Gate: unit parity at `partial_rotary_factor=0.5` + all regressions UNCHANGED.*
- **W2 — Sandwich norms + the GLM dense attention block.** New
  `glm4.{h,cpp}` with output-normed sublayers, biased qkv, and the W1 rope. Does NOT
  extend `dense_attn_block.h` — reuses only its glue, per the OPT precedent.
  *Gate: `Glm4ForCausalLM` SACRED gate on GLM-4-9B-0414.*
- **W3 — `GlmForCausalLM`.** Llama-shaped, reuses W1's rope, drops the `o_proj`
  bias. Small once W1/W2 land. *Gate: SACRED gate on glm-4-9b-chat-hf.*
- **W4 — MoE router extension.** Sigmoid scoring, `e_score_correction_bias`, group
  masking, `routed_scaling_factor` in `vt::MoeRouterTopKArgs` + `cuda_moe.cu`.
  **Coordinate with `CLAIM-MLA-DEEPSEEK` before starting — this is shared.**
  *Gate: unit parity at both GLM MoE configurations + all regressions UNCHANGED.*
- **W5 — `Glm4MoeLiteForCausalLM` on GLM-4.7-Flash.** The campaign's MLA block +
  `RunMoeBlock` + W4's router + the `fused_qkv_a_proj` loader merge. **Gated on the
  MLA campaign reaching W6.** This is the highest-value W in the spike: it is the
  second MLA gate vehicle and closes the campaign's own two coverage gaps (§0.1 C2).
  *Gate: SACRED gate on GLM-4.7-Flash, form per W0.*
- **W6 — `Glm4MoeForCausalLM` honesty pass.** Config/registry resolution and
  weight-map coverage on a single downloaded shard of `zai-org/GLM-4.5-Air`; unit
  parity of the router at its real 128-expert/top-8 dimensions. Record
  `HW-BLOCKED` with the measured 205.8 GiB. *Gate: gate 10.*
- **W7 — DSA + `GlmMoeDsaForCausalLM` honesty pass.** Registry/config resolution for
  `glm_moe_dsa` including the fp32-router special case; record the flashinfer
  dependency gap with its four concrete failure modes (§0.2) as the reopen
  condition. No indexer implementation. *Gate: gate 10; row records
  `BLOCKED` with the dependency named.*
- **W8 — DeepSeek-V4 honesty pass.** Config resolution against the real V4-Flash
  `config.json`; record `HW-BLOCKED` at 148.7 GiB vs 119 GiB. Optionally port the
  V4 chat template against upstream golden fixtures — offline, no GPU, no weights —
  since that is the one V4 deliverable that is fully gateable today. *Gate: gate 10.*
- **W9 — ChatGLM.** Lowest priority: legacy lineage, out-of-tree config,
  `trust_remote_code`. Reuses W1's rope. Deferrable indefinitely without blocking
  anything. *Gate: SACRED gate on chatglm3-6b.*
- **W10 — Speed close.** Decode-graph sibling for each landed model + the binding
  every-axis grid vs vLLM. Nothing reaches `DONE` before this. *Gate: acceptance
  rule — match or beat vLLM on every axis.*

### Risks/decisions

**D1 — Do not implement DSA; record it as designed and dependency-blocked.**
The temptation is to port the indexer because it is well-documented and the csrc
kernels are readable. Resist it: the logits GEMMs are in vendored DeepGEMM, not
csrc, so a csrc-only port is structurally incomplete; and there is no sm_121 backend
that can consume the selection (§0.2). Building it would produce untestable code on
our only hardware. §0.3 exists so the port is DESIGNED and can be built quickly the
day flashinfer lands sm12x sparse top-k. The reopen condition is explicit and
falsifiable: a flashinfer release whose `mla/_core.py` XQA path accepts
`sparse_mla_top_k` and `kv_scale_format`.

**D2 — Promote GLM-4.7-Flash to a second MLA gate vehicle rather than treating GLM
as a separate campaign.** It is a zero-override subclass of DeepSeek-V2's attention
(`glm4_moe_lite.py:94-99`), it fits GB10, and it closes both coverage gaps the MLA
campaign named in its own spec. Treating it as "GLM work" would waste that. The cost
is a coordination dependency on `CLAIM-MLA-DEEPSEEK` reaching W6, and the honest
consequence is that W5 cannot start early.

**D3 — Start with G2 (partial rotary + sandwich norms), not G1.** G1 is more
valuable but is blocked behind someone else's W6. G2 is fully independent, unblocks
four rows' worth of primitives (Glm, Glm4, ChatGLM, and later MiniMax-M2's partial
RoPE), and both its gate vehicles fit and are cheap to download. Sequencing G2 first
keeps this campaign off the critical path of the campaign it depends on.

**D4 — Do not extend `dense_attn_block.h` for GLM.** GLM violates three of its
hard-coded assumptions at once (biased qkv, partial non-NeoX rope, sandwich norms).
The OPT row already proved this header does not stretch across families and chose new
files instead; that precedent holds. Reuse the glue and the GQA paged path, write a
new block. The tracked debt is that we will then have three attention blocks; a
consolidation pass is a later, separate question and must not be smuggled into a
model bring-up.

**R1 — Disk is the most likely operational failure.** dgx is at 95% with 184 GiB
free; the three gate checkpoints total 93.2 GiB. An ENOSPC mid-download presents as
bogus test failures, and this has bitten the project before. Mitigation: stage
downloads one at a time, verify free space before each, and prune before starting.
This also means the 148.7 GiB DeepSeek-V4-Flash cannot be downloaded even for a
weight-map slice check without freeing space first — W8 must use a single-shard
fetch, not a full clone.

**R2 — The sandwich norm is a silent-corruption hazard.** `glm4.py:206,211` applies
the extra norms to the sublayer OUTPUT before the residual add. Getting the order
wrong (norm after the add, or norming the residual) produces a model that still
emits fluent text while being numerically wrong — precisely the OPT BOS failure
mode, which scored 0/6 while reading fine. Mitigation: gate 2 requires an e2e
token-exact gate for this primitive, not a unit test alone.

**R3 — `Glm4MoeForCausalLM` may have no viable gate ever on this hardware.** Its
smallest genuine checkpoint is 205.8 GiB bf16. The 104.8 GiB FP8 variant would fit
memory but depends on an fp8 checkpoint-loading row we do not own, and would change
the gate's dtype semantics. This row is honestly `HW-BLOCKED` and should not be
planned as if a vehicle will appear. Tracked debt, not a plan.

**R4 — The `n_group=1` blind spot survives even with the new vehicle.** All three
live GLM MoE configs (4.7-Flash, 4.5-Air, GLM-5) have `n_group: 1, topk_group: 1`,
so group-limited greedy routing is still degenerate at every gate. `noaux_tc` and
`e_score_correction_bias` ARE exercised e2e by GLM-4.7-Flash (that is the C2 win),
but the group-masking code path remains unit-gated only. This is a smaller gap than
the campaign's original one, not a closed one, and must be stated as such rather
than rounded to "coverage complete".

**R5 — DeepSeek-V4's MHC is unportable-by-inspection.** Sinkhorn normalization runs
inside TileLang kernels every forward, the RMSNorms are fused into them, there is no
eager reference in the repo, and upstream has zero numerical tests for those kernels.
Even if V4 fit in memory, we would have to build our own reference for its most novel
subsystem before any port could be trusted. This compounds the hardware block and is
why W8 is an honesty pass, not a bring-up.

**R6 — Upstream gives us almost no GLM text-correctness oracle.** Only `chatglm3-6b`
appears in upstream's text-generation correctness list; GLM-4/4.5/4.7/5 have
initialization and registry smoke tests only. Our gates therefore rest entirely on
our own pinned-vLLM oracle comparison. That is the project's standard practice and
is sufficient, but it must not be mistaken for inheriting upstream guarantees.
