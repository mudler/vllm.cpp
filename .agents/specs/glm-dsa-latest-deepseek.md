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
2. **`GlmMoeDsaForCausalLM` is DeepSeek-V3.2, verbatim at the pin this section was
   written against.** `deepseek_v2.py:1917-1918` @ `e24d1b24`
   is `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`. At the CURRENT
   parity pin `555967922` the same class is `:1930-1931`, and on vLLM `main` the
   architecture is no longer verbatim at all. See §2. The only
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
| `registry.py:116` `GlmMoeDsaForCausalLM` (`deepseek_v2.py:1917-1918`) @ `e24d1b24`; `:117` and `:1930` at the current pin `555967922`, see §2 | **NEW** alias registration over the DeepSeek-V3 TU + the `glm_moe_dsa` fp32-router special case (`deepseek_v2.py:120-130`). DEP-BLOCKED + HW-BLOCKED; registry/config resolution only |
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

Per [`.agents/porting.md`](../porting.md), the upstream test modules that
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
per [`.agents/verification.md`](../verification.md), exactly as the sweep models did — it cannot
be inherited from an upstream test list.

### Gates

1. **Correctness (SACRED), `Glm4MoeLiteForCausalLM` on `zai-org/GLM-4.7-Flash`
   bf16.** Token-exact against the pinned vLLM oracle on the identical prompt set,
   greedy. Gate form selected BY MEASUREMENT per
   [`near-tie-distributional-gate`](../verification.md): run vLLM's own greedy K=5 times
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

---

## 2. Reconcile of `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` against `zai-org/GLM-5.3` (2026-08-28)

**Issue:** [#2194](https://github.com/mudler/vllm.cpp/issues/2194).
**Scope:** records only. The row stays `BLOCKED`. No product code, no pin advance,
no second matrix row. `GlmMoeDsaForCausalLM` has exactly one row and keeps it.
**Not touched:** `MODEL-MM-GLM53-FLASH` and
[`glm5-next-flash.md`](glm5-next-flash.md), which own the different architecture
`glm5_next` and are edited by another claim.

**Secondary oracle:** `llama-cpp`

### 2.1 Both upstream anchors were stale at our own parity pin

The row carried `registry.py:116` and `deepseek_v2.py:1917-1918`. Measured at the
current parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`:

| The row said | What is there at the current pin | Correct at the current pin |
|---|---|---|
| `registry.py:116` | `"Glm4MoeLiteForCausalLM": ("glm4_moe_lite", …)`, a DIFFERENT model | `registry.py:117` |
| `deepseek_v2.py:1917-1918` | `def load_weights(...)` / `loader = AutoWeightsLoader(self)` | `deepseek_v2.py:1930-1931` |

The registry anchor is the dangerous shape. Line 116 holds a plausible GLM entry
for another architecture, so a reader who checks it casually confirms it and
stops.

Both corrected anchors are UNIQUE, which is the property that matters. At the pin
`grep -n GlmMoeDsaForCausalLM` over `registry.py` returns one line, `:117`, and
over `deepseek_v2.py` returns one line, `:1930`. `grep -n glm_moe_dsa` over
`deepseek_v2.py` also returns one line, `:127`, inside `_get_moe_router_dtype`.
That last anchor sharpens the row's older `:120-130` range.

**Neither number was wrong when it was written.** Both are exact at the PRIOR pin
`e24d1b24`: `registry.py:116` and `deepseek_v2.py:1917`. That is the revision
this spike's `### Upstream chain` names, and that section is still honest. The
2026-07-26 advance to `555967922` moved `registry.py` by one line and
`deepseek_v2.py` by thirteen. The matrix row copied the coordinates without the
revision label, so nothing could see them drift. Every anchor this section adds
therefore carries the revision it was measured at.

Commands:

```sh
git show 5559679229bc961848b121ccdeaa8fa5d79bec98:vllm/model_executor/models/registry.py \
  | grep -n 'GlmMoeDsaForCausalLM\|Glm4MoeLiteForCausalLM'
git show 5559679229bc961848b121ccdeaa8fa5d79bec98:vllm/model_executor/models/deepseek_v2.py \
  | grep -n 'GlmMoeDsa\|glm_moe_dsa'
git show e24d1b24fe:vllm/model_executor/models/registry.py | grep -n 'GlmMoeDsaForCausalLM'
```

Read every object with `git show <revision>:<path>`, never from a working tree.
The local vLLM checkout can be dirty, and a dirty tree is not a revision.

### 2.2 "GLM-5.x is DeepSeek-V3.2 VERBATIM" holds at the pin and nowhere later

At `555967922` the claim is exact. `deepseek_v2.py:1930-1931` is
`class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`, and the only
behavioural special case is the fp32 router dtype forced by
`model_type == "glm_moe_dsa"` in `_get_moe_router_dtype` (`deepseek_v2.py:127`).

On vLLM `main` `d1922cb5a7` (read 2026-08-28) three things diverge:

1. `registry.py:118` re-homes the alias to `vllm.models.deepseek_v32`. That
   package's `__init__.py:17-29` binds `GlmMoeDsaForCausalLM` to
   `DeepseekV32ForCausalLM` from `nvidia/model.py` under CUDA, and keeps the
   `deepseek_v2` subclass on ROCm, XPU and CPU. So under CUDA the architecture no
   longer resolves to the `deepseek_v2` class at all.
2. The architecture gains its own `VerifyAndUpdateConfig`,
   `vllm/model_executor/models/config.py::GlmMoeDsaForCausalLM` at `:43`,
   registered in the dispatch table at `:936`. It calls
   `parallel_config.set_dcp_defaults(comm_backend="a2a", q_replicate=True)`, which
   is a decode-context-parallel default this architecture alone selects.
3. `vllm/config/vllm.py:81` names it in
   `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`.

None of the three exists at the pin: `git show <pin>:…/config.py` and
`git show <pin>:vllm/config/vllm.py` return no `GlmMoeDsa` match at all. Reaching
any of it needs a pin advance, and this reconcile takes none. The row's
DEP-blocked reason now says "verbatim AT THE PIN" and names what diverges.

### 2.3 The GLM-5.3 checkpoint, and the blocker arithmetic

`zai-org/GLM-5.3`, revision `935644c05e76fc198714f4cca449fd8b970ff6d7`, read from
the HuggingFace API on 2026-08-28.

| Property | Value |
|---|---|
| `model_type` / `architectures` | `glm_moe_dsa` / `["GlmMoeDsaForCausalLM"]` |
| `dtype` | `bfloat16` |
| `quantization_config` | `quant_method: fp8`, `fmt: e4m3`, dynamic activations, `weight_block_size [128, 128]` |
| layers / hidden / vocab | 78 / 6144 / 154880 |
| routed + shared experts, top-k | 256 + 1, top-8 |
| `q_lora_rank` / `kv_lora_rank` | 2048 / 512 |
| `qk_rope_head_dim` | 64 |
| indexer | `index_topk 2048`, `index_n_heads 32`, `index_topk_freq 4`, `indexer_rope_interleave: true` |
| MTP | `num_nextn_predict_layers: 1` |
| shards / on-disk size | 141 safetensors / 755,632,050,320 B = **703.74 GiB** |
| parameters (API `safetensors.total`) | **753,329,940,480** = 751,226,191,872 `F8_E4M3` + 2,103,729,152 `BF16` + 19,456 `F32` |

This is NOT the `glm5_next` of `MODEL-MM-GLM53-FLASH`. The two differ on
`model_type`, layer count, hidden size, expert count, `q_lora_rank`, RoPE
presence, the mHC stream, the indexer k-pool and modality. The rows stay
separate.

The measured parameter count confirms the 753.9B this row already carried. It
also gives **1403.2 GiB** at bf16, where the row said 1404.2 GiB. The difference
is small and the corrected value is the one measured here.

One detail retires part of the fp32-router special case for this checkpoint.
GLM-5.3 DOES expose `moe_router_dtype: float32`, so the upstream comment "Older
GLM-5/5.2 configs require fp32 routing but do not expose `moe_router_dtype` yet"
no longer describes the newest artifact. The forced branch still fires first, so
behaviour is unchanged.

**The arithmetic, so nobody redoes it.** `dgx:gpu0` reports
128,452,956,160 B = 119.631 GiB from `cudaMemGetInfo`. Fitting 753,329,940,480
parameters in that budget needs **1.3641 bits per weight**, before any KV cache.

| Rate | GLM-5.3 size | Fits `dgx:gpu0` |
|---|---|---|
| 2.32 bpw, the rate of the smallest arm that fits anything comparable | 203.5 GiB | no |
| 1.70 bpw, aggressive sub-IQ1 | 149.1 GiB | no |
| 1.50 bpw, below anything published | 131.5 GiB | no |

The 2.32 bpw reference is `unsloth/GLM-5.3-Flash-GGUF` `UD-IQ1_S`, 86.69 GiB over
321.32B parameters, which is a different model and is cited only for the rate.

GGUF conversion of GLM-5.3 has started and does not change the verdict. Re-read
on 2026-08-28: `unsloth/GLM-5.3-GGUF` (revision `8cf52b13b130`, modified
16:14 UTC) holds one complete arm, `UD-Q3_K_XL`, 9 files, **319.41 GiB**, which is
3.64 bpw. `AtomicChat/GLM-5.3-GGUF` and `MaliAir/GLM-5.3-MXFP4-MOE-Q8_0-GGUF`
hold ZERO `.gguf` files. That repository is being populated live, so re-read it
rather than quoting this line.

`rc devices` on 2026-08-28 listed `dgx:gpu0`, `orin:gpu0`, `strix:gpu0` and
`thor:gpu0`. None is larger than `dgx:gpu0`. No single fleet device holds this
model at any published or plausible quantization, so the row's `🚫` is correct
rather than stale.

### 2.4 Oracles: both already registered, and neither needs a new file

**vLLM is the primary and reaches this architecture AT OUR PIN.** The entry is
`registry.py:117` and the class is `deepseek_v2.py:1930`, both at
`555967922`. No pin advance is needed, and no new oracle file is written. Three
reasons, in order of force:

1. `scripts/check-oracle-pins.py` requires exactly one `role = primary` record
   and requires it to be `vllm`. A second vLLM file is refused by the gate.
2. `.agents/oracles/README.md` states that `vllm.md` points at
   `.agents/upstream-sync.md` rather than restating the pin, "because a pin
   transcribed twice is a pin that drifts". A per-model vLLM file would be that
   second transcription. §2.1 above is a live instance of exactly that failure.
3. The registry files carry oracle IDENTITY, not per-model reach. Adding an id
   would also force an edit to the AGENTS.md admissible-oracle table, which is a
   shared file this reconcile has no reason to lock.

**llama.cpp reaches it at our STOCK release pin `b10451`.** Verified in a fresh
bare clone of `ggml-org/llama.cpp`, never a working tree:

```sh
git init --bare && git remote add origin https://github.com/ggml-org/llama.cpp.git
git fetch --depth 1 origin 10bf611e533d81f739128304991c5e133c6aebd8
git ls-remote --tags origin refs/tags/b10451   # -> 10bf611e533d81f739128304991c5e133c6aebd8
git show 10bf611e533d81f739128304991c5e133c6aebd8:src/llama-arch.cpp | grep -n GLM_DSA
```

| What | Where, at `b10451` |
|---|---|
| architecture name | `src/llama-arch.cpp:85`, `{ LLM_ARCH_GLM_DSA, "glm-dsa" }` |
| the case that reaches it | `src/llama-arch.cpp:1051` |
| graph | `src/models/glm-dsa.cpp` |
| GGUF constant | `gguf-py/gguf/constants.py:534`, name at `:1249` |
| converter registration | `conversion/glm.py:274-276`, `@ModelBase.register("GlmMoeDsaForCausalLM")`, `class GlmMoeDsaModel(DeepseekV2Model)`, `model_arch = GLM_DSA`; dispatch row `conversion/__init__.py:99` |

`grep -n GLM_DSA` over `src/llama-arch.cpp` returns exactly those two lines, so
both anchors are unique. The converter mirrors the vLLM structure: a
`DeepseekV2Model` subclass, the same relation the vLLM class has to
`DeepseekV2ForCausalLM`.

**So no scoped PR-oracle file is needed here, and that contrast is the useful
finding.** `llama-cpp-qwen4exp` exists because `qwen4exp` is defined by an open
llama.cpp PR and by no release, and the `llama-cpp-glm5next` proposed in
[#2178](https://github.com/mudler/vllm.cpp/issues/2178) exists for the same
reason. A scoped file buys a denominator that the stock pin cannot supply. Here
the stock pin supplies it, so a scoped file would add a second llama.cpp pin that
answers a question the first one already answers.

**Both oracles are `gateable = no` FOR THIS MODEL, and the reason is MEMORY.**
Not missing support: both build the architecture and would run it on a device
large enough to hold it. That distinction is the whole point of this section. It
separates this row, blocked on hardware, from `MODEL-MM-GLM53-FLASH`, which is
blocked because no SERVING oracle registers `glm5_next` at any revision — not
vLLM, not vllm-omni, not SGLang, not llama.cpp. That row is not without a
reference: `transformers` **v5.16.1** implements the architecture, and W4
([#2098](https://github.com/mudler/vllm.cpp/issues/2098), landed on `main`
2026-08-28 as `6c715de00`) gates its mHC arm by RUNNING that module. So the two
rows are blocked on opposite things — GLM-5.3-Flash can be gated piecewise
against a reference nobody serves, and this row cannot be gated at all, because
the oracles that DO serve it cannot fit it.

This per-model verdict lives on the row and in this spec, never in
`.agents/oracles/*.md`. The `gateable` key there is a property of the ORACLE, and
both oracles are `gateable = yes` as oracles. Writing `no` into either file to
express one blocked checkpoint would retract a measured property of the oracle
across every other row that uses it.

### 2.5 What this reconcile did NOT do

- It did not advance any pin, and it measured nothing on a GPU.
- It did not add a matrix row. The architecture has one row and keeps it.
- It did not add an oracle file, for the reasons in §2.4.
- It did not change the row's state. `BLOCKED` is still correct, and §2.3 is the
  arithmetic that keeps it correct.
- It did not touch `MODEL-MM-GLM53-FLASH` or its spec.

---

## 3. Port plan: `GlmMoeDsaForCausalLM` / `zai-org/GLM-5.3` under expert streaming (2026-08-29)

**Issue:** [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
**Row:** `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm`, `BLOCKED` -> `SPIKE`.
**Claim:** `CLAIM-MODEL-GLM-MOE-DSA`.
**Scope of THIS section:** a committed port plan and nothing else. No product
code, no pin advance, no GPU lease, no download. Every number below was
recomputed here from primary sources — the published `config.json`, the
published GGUF shard headers over HTTP range requests, and the local tree — and
none of it was copied from #2214 or from §2 above. Where a recomputation
DISAGREES with a figure already on record, the disagreement is stated.

**What changes versus §2.** §2 concluded `BLOCKED` on resident capacity, and
that conclusion was correct for the frame it used. This section changes the
frame: for a model that is 97.4% routed experts, the question is not whether the
weights fit but whether the **step working set** fits, and that is a different
and much smaller number. The row therefore moves to `SPIKE` — scoped in a
committed spec, not implemented — and stays there until W1 lands.

### 3.1 The streaming arithmetic, recomputed

**Method.** Fetch `https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json`
(29,464 B, HTTP 200, read 2026-08-29). Sum the parameter count of every tensor
group analytically from the config's own fields, then check the total against
the checkpoint's own `model.safetensors.index.json`
(`metadata.total_size = 755,617,140,416` over 118,629 tensors, fetched through
the `resolve` endpoint because the `raw` endpoint serves the 11,359,251-byte
LFS pointer) and against the HuggingFace API's `safetensors.total`. A model that
does not reproduce the published total is a model of some other checkpoint.

The config's own layer schedule is read, not assumed. `mlp_layer_types` is 78
entries, 3 `dense` then 75 `sparse`, which agrees with
`first_k_dense_replace = 3`. `indexer_types` is 78 entries, **21 `full` and 57
`shared`**, in the pattern `full,full,full` then `(shared,shared,shared,full)`
repeating. `num_nextn_predict_layers = 1` adds a 79th block.

Per-group formulae, all from `config.json`:

| Group | Formula | Params |
|---|---|---|
| one routed expert | `3 * hidden * moe_inter` = `3 * 6144 * 2048` | 37,748,736 |
| routed experts, one MoE layer | `256 *` the above | 9,663,676,416 |
| **routed experts, 75 MoE layers** | `75 *` the above | **724,775,731,200** |
| **routed experts, the MTP block** | `1 *` the above | **9,663,676,416** |
| MLA, one layer | `H*q_lora + q_lora*n_h*qk_head + H*(kv_lora+qk_rope) + kv_lora*n_h*(qk_nope+v_head) + n_h*v_head*H` | 165,019,648 |
| MLA, 78 layers | | 12,871,532,544 |
| indexer, one layer | `q_lora*idx_n_h*idx_head + H*idx_head + H*idx_n_h` | 9,371,648 |
| indexer, 21 `full` layers | | 196,804,608 |
| shared expert, 75 MoE layers | `75 * 1 *` one expert | 2,831,155,200 |
| dense MLP, 3 layers | `3 * 3 * 6144 * 12288` | 679,477,248 |
| router gates, 75 layers | `75 * 6144 * 256` | 117,964,800 |
| embed + lm_head | `2 * 154880 * 6144` | 1,903,165,440 |
| MTP block, non-expert | MLA + indexer + shared expert + gate + `eh_proj(2H x H)` | 289,210,368 |

| | params | share |
|---|---|---|
| routed experts, **streamable** | **734,439,407,616** | **97.49%** |
| everything else, **must be resident** | **18,889,310,208** | **2.51%** |
| model total, this arithmetic | 753,328,717,824 | — |
| API `safetensors.total`, measured | 753,329,940,480 | — |
| **residual** | **-1,222,656** | **-0.00016%** |

The residual is 1.2M parameters over 753.3B — the bias terms and the 79
`k_norm.bias` / layernorm vectors this model does not enumerate. **This is a
tighter reconciliation than #2214's, and the numbers differ, which is why it was
redone.** #2214 models 724.8B streamable / 21.0B resident / 745.8B total and
calls that "within 1%". The 7.5B gap is the MTP block, whose 256 experts are
themselves streamable; folding it in moves the streamable share from 97.2% to
**97.49%** and the resident total from 21.0B **down** to 18.89B. Both figures
favour the argument, so the correction does not change the verdict — but the
resident dtype table below is materially different and the difference is 4 GiB.

**Resident footprint by dtype.** `dgx:gpu0` reports 128,452,956,160 B =
**119.631 GiB** from `cudaMemGetInfo` (measured 2026-08-28, §2.3; not
re-measured here, because this section took no GPU lease).

| resident dtype | bpw | resident | whole model | fits `dgx:gpu0` resident-only |
|---|---:|---:|---:|---|
| bf16 | 16.0000 | **35.18 GiB** | 1403.18 GiB | yes |
| Q8_0 | 8.5000 | 18.69 GiB | 745.44 GiB | yes |
| Q6_K | 6.5625 | 14.43 GiB | 575.52 GiB | yes |
| Q5_K | 5.5000 | 12.09 GiB | 482.34 GiB | yes |
| Q4_K | 4.5000 | 9.90 GiB | 394.65 GiB | yes |
| Q2_K | 2.6250 | 5.77 GiB | 230.21 GiB | yes |

#2214 gives bf16 resident as 39.19 GiB; recomputed it is **35.18 GiB**, because
its resident set was 21.0B and the correct one is 18.89B. Neither number changes
the answer. **The measured resident figure that actually matters is neither of
these, and it is in §3.4: the published UD arms carry the non-expert tensors at
mixed Q4_K/Q5_K/Q6_K/Q8_0/F32 and weigh 14.51 GiB.**

**One decode step, batch 1.** `num_experts_per_tok = 8` over 75 MoE layers
touches `8 * 75 * 37,748,736` = 22,649,241,600 parameters, i.e. 3.0% of the
routed set:

| dtype | per decode step |
|---|---:|
| bf16 | 42.19 GiB |
| Q8_0 | 22.41 GiB |
| Q4_K | 11.87 GiB |
| Q2_K | 6.92 GiB |
| IQ1_S | 4.12 GiB |

#2214's "~6.86 GiB at 2.6 bpw" reproduces as 6.92 GiB at Q2_K's exact 2.625 bpw.
**The step figure is per token and it scales with batch**: at concurrency `c` the
distinct set is bounded by `min(256, 8c)` experts per layer, so the touched bytes
grow until they saturate at the whole 187 GiB tower set. This is a paging
problem at `c = 1` and a capacity problem well before `c = 32`, and §3.6 keeps
that inside the gate.

### 3.2 What `expert_streamer.cpp` provides today, and what this model needs

Read at base `60a6dd97b`. The capability is real and it is **not turnkey for this
model**; five of the eight gaps below are load-bearing.

**What exists.** `include/vllm/model_executor/expert_streamer.h` (154 lines) and
`src/vllm/model_executor/expert_streamer.cpp` (224 lines), plus
`expert_slot_cache.{h,cpp}` (the policy) and
`host_expert_slot_store.h` / `device_expert_slot_store.{h,cpp}` (the
destinations).

- `ExpertSlotStore` (`expert_streamer.h:43`) is a pure-virtual destination seam:
  `slot_bytes()`, `slot_count()`, `WriteSlot`, `SlotForWrite`, `CommitSlot`.
  There is deliberately no virtual `SlotForRead`; the read is the concrete
  `HostExpertSlotStore::Slot()`.
- `ExpertStreamer` (`expert_streamer.h:91`) offers `Ensure`, `EnsureSpan`,
  `EnsureFile(key, fd, file_offset, bytes)` and `EndStep()`.
- `ExpertSlotCache` (`expert_slot_cache.h:61`) is a hotness-decayed LFU with LRU
  tiebreak (`expert_slot_cache.cpp:19-44`, default decay 0.98), a dense slot
  table with an `unordered_map<ExpertKey,int32_t>` logical->physical remap, and a
  **per-step protection rule**: every `Acquire` marks the entry protected
  (`expert_slot_cache.cpp:91`) and only `EndStep()` clears it
  (`:142-145`). If every slot is protected, `Acquire` returns `-1` and sets
  `capacity_exhausted_` (`:105-113`).
- The backing store is the **GGUF file on disk, read by `pread(2)`** against the
  model fd (`expert_streamer.cpp:85-100`), or a memcpy out of the mmap when no fd
  is available. The resident store is host RAM: a plain `std::vector<uint8_t>`
  arena of `slots * slot_bytes` (`host_expert_slot_store.h:40`).
- Admissible weight formats are **GGUF keep-quant / keep-f16 stacked
  `[E, out, in]` towers only** (`gguf_device_fit.cpp:95` refuses anything that is
  not `kKeepQuant` or `kKeepF16`). Slices are **pure byte offsets, never a
  repack**, which is a layout precondition stated at `gguf_expert_span.h:12-16`:
  whole rows of the same K, no block ever cut.
- Config surface, live and reachable from production: `VT_MOE_EXPERT_STREAM`,
  `_SLOTS` (default **64**, `weight_residency.cpp:1035-1039`), `_SLOT_BYTES`,
  plus the JSON `{"vllm_cpp":{"expert_stream":{...}}}` schema at
  `include/vllm.h:502-506`, installed at
  `model_loader.cpp:2251` inside `LoadedEngine::FromModelDir`, parsed by the
  OpenAI server (`server_main.cpp:654-655`, `:1088-1089`, `:1326`) and the C ABI
  (`vllm_c.cpp:666-667`). Default is OFF.
- Tests: six binaries, `tests/CMakeLists.txt:1562-1633`. The end-to-end suite is
  `tests/vllm/model_executor/test_expert_stream_wiring.cpp`, which proves decode
  reaches the streamer, that a streamed slice and the tower view produce
  identical logits, and that a file-backed tower is served by `pread` at a
  deliberately unaligned offset.

**What is missing for GLM-5.3.** Each of these is work, not configuration.

1. **The wiring is not a seam. It lives inside `qwen3_5.cpp`.**
   `Qwen35ExpertStream` (`qwen3_5.cpp:5725`), `KqExpertSlice` (`:6180`),
   `KqHostSliceView` (`:6169`), `Reserve` (`:6284`) and the step guard are all in
   that one translation unit, and it is the **only** model TU that constructs
   `HostExpertSlotStore` / `ExpertSlotCache` / `ExpertStreamer` (`:6038-6040`).
   `deepseek_v2.cpp` has zero references to any streamer symbol.
   `qwen3_moe.cpp:195-197` holds only the step guard. A new architecture cannot
   include a header and get streaming; the mechanism has to be lifted into a
   shared seam first. **This is W2 and it is the largest single item.**
2. **The default slot budget fails closed and quietly.** The decode working set
   is `75 layers * 3 towers * 8 experts = 1800` distinct slices, every one
   protected until `EndStep`. The default is 64 slots. Below the working set,
   `Slice` returns `nullptr`, `exhausted_` increments (`qwen3_5.cpp:5824`), and
   every slice falls back to reading the mmap in place — **counted on stderr, not
   an error**. On this model that fallback is a 187 GiB random read per token.
3. **No prefetch, no double buffering, no async I/O**, stated verbatim at
   `expert_streamer.h:25-29`. A miss is a blocking `pread` inline in front of the
   GEMM. 1800 serialized syscalls per token in the cold case.
4. **Eviction is an O(resident) linear scan per miss**
   (`expert_slot_cache.cpp:26-44`). At the slot counts §3.3 needs (thousands)
   and ~1800 misses per step, that is a host cost nobody has profiled.
5. **No device destination is wired.** `DeviceExpertSlotStore` exists, is filled
   correctly through `EnsureFile`, is gated by
   `tests/vllm/model_executor/test_device_expert_slot_store.cpp`, and **is
   selected by nothing** (`expert_streamer.h:13-23`, and `qwen3_5.cpp:6067`
   holds the concrete host store). The production predicate is
   `qwen3_5.cpp:6199`: `cpu || host_memory_is_device_addressable()`. A discrete
   CUDA GPU answers false and falls through. **`dgx:gpu0` is a GB10 with unified
   memory and answers TRUE**, which is precisely why this row is viable there and
   would not be on a discrete part.
6. **Streaming and the grouped keep-quant MoE path are mutually exclusive**
   (`qwen3_5.cpp:6307-6312`); enabling one disables the other, with one line on
   stderr.
7. **`pread` streaming has never run on a real checkpoint.**
   `.agents/specs/expert-streaming.md` `## Owed`, verbatim: "**The `pread` path
   has never run on the model.** ... It is still unmeasured on a real
   checkpoint." No test model has more than 4 experts or 4 layers
   (`tests/support/expert_stream_model.h:130-131`).
8. **Windows has no streaming at all**: `EnsureFile` throws
   `"expert streamer: EnsureFile needs pread"` (`expert_streamer.cpp:31-36`).

**Row states, read rather than assumed.** `ENG-EXPERT-STREAM`
(`engine-matrix.md:117`) is `READY`, owner `-`, and its "Our code" and "Our
tests/evidence" columns are both a bare `-` despite ~700 shipped lines and six
test binaries; its row text describes "fixed contiguous Marlin slots" and **no
Marlin code is on this path**. `ENG-HYBRID-PLACEMENT` (`:119`) is `ACTIVE` and
is the *inverse* mechanism — it moves expert COMPUTE to the CPU — not a
substitute. `ENG-RESIDENCY-CONFIG` (`:120`) is `ACTIVE`, is the only one of the
three with populated code/evidence columns, and owns the config surface this row
uses unchanged. `ENG-EXPERT-STREAM-DEVICE` (`:122`, `ACTIVE`, #1124) is the row
that owns gap 5; its `## Now` says W1 "lands UNREACHED" and W2 owns the wiring.
**This row does not take any of those four rows' work.** It consumes them, and
where it needs more than they provide it says so under `## Owed`.

### 3.3 The residency plan

Grounded in what §3.2 measured, not in what the streaming row claims.

**Two tensor classes, and the split is the GGUF tensor name.** The streamer's
own admission rule is the `_exps.weight` suffix (`model_loader.cpp:2472`,
`kStreamedExpertSuffix`; `gguf_device_fit.h:98-99`), and GLM-5.3's GGUF
conveniently draws the same line: `blk.N.ffn_{gate,up,down}_exps.weight` are the
228 stacked `[256, out, in]` towers and every other tensor is per-layer.

| class | tensors | UD-IQ1_S size | placement |
|---|---:|---:|---|
| **resident** | 1581 | **14.511 GiB** | device pool, whole run |
| **streamed** | 228 | **187.312 GiB** | slot cache, paged from the file |

The resident class is: `token_embd`, `output`, `output_norm`, and per block
`attn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_a_mqa`,
`attn_kv_a_norm`, `attn_k_b`, `attn_v_b`, `attn_output`, `ffn_norm`,
`ffn_gate_inp`, `exp_probs_b`, the three shared-expert projections, the five
`indexer.*` tensors, the three dense-MLP projections on blocks 0-2, and the four
`nextn.*` tensors on block 78. Full census in §3.4.

**The resident expert cache budget.** Slots are uniform and sized to the
LARGEST slice (`host_expert_slot_store.h:30-33`; a bigger slice is refused by
name, `expert_streamer.cpp:181-186`), so on UD-IQ1_S `slot_bytes` is set by the
IQ4_XS `ffn_down_exps` slice:

| slice encoding | bytes | MiB |
|---|---:|---:|
| IQ1_S gate/up | 2,457,600 | 2.344 |
| IQ2_XXS gate/up | 3,244,032 | 3.094 |
| IQ3_XXS down | 4,816,896 | 4.594 |
| **IQ4_XS down (the max)** | **6,684,672** | **6.375** |

| slots | arena | note |
|---:|---:|---|
| 1800 | 11.21 GiB | the bare decode working set at `c = 1`; **the floor, not a budget** |
| 4096 | 25.50 GiB | ~2.3 steps of history |
| 8000 | 49.80 GiB | the shape `benchmarks/expert_stream_device_w0e.cpp` already uses |

**Proposed default for the first run: 4096 slots = 25.50 GiB.** Resident 14.51 +
slots 25.50 = **40.01 GiB**, against 119.631 GiB on `dgx:gpu0`, leaving ~79 GiB
for the KV cache, activations, scratch pools and the CUDA context. The KV
arithmetic, from the config: the MLA latent row is `kv_lora + qk_rope = 576`
elements per token per layer, so 78 layers at bf16 is 89,856 B/token = 87.75
KiB/token, and the DSA indexer cache adds 132 B/token/indexer-layer over 22
layers = 2,904 B/token. At 8192 context that is **0.71 GiB**; at 131,072 context,
11.32 GiB. Even the long-context case fits inside the headroom, and the
`max_position_embeddings` of 1,048,576 does not, which is a configuration limit
to refuse rather than a surprise.

**Uniform slots waste 46% of the arena on this artifact.** 1800 slices at their
real sizes are 6.03 GiB; at the uniform 6,684,672 B they are 11.21 GiB. That is
the price of the pure-byte-offset design, it is a known cost rather than a
defect, and W6 records it as a measured lever rather than fixing it
speculatively.

**On a cache miss mid-step: the step stalls, synchronously, per slice.** There is
no other behaviour available (§3.2 gap 3). The chain is
`ExpertMlpKq -> MatmulBf16Slice -> KqExpertSlice -> Qwen35ExpertStream::Slice ->
EnsureFile -> ::pread`, blocking, immediately before `vt::MatmulBT` runs on that
weight. On a throw the acquisition is undone (`expert_streamer.cpp:108-111`,
`:163-166`, `:214-217`) so nothing half-filled becomes resident.

**On cache EXHAUSTION — every slot protected this step — the model does not
fail. It silently degrades**, and on this artifact that degradation is fatal to
any measurement: `Slice` returns `nullptr` and the caller reads the tower in
place out of a 201.83 GiB mmap. **W1 therefore owes a refusal, not a fallback,**
when the configured slot count is below the model's computed decode working set.
A model that quietly reads 187 GiB per token through the page cache is the exact
shape of measurement this repository has been burned by, and a `capacity <
75*3*num_experts_per_tok` check at load costs one comparison.

### 3.4 The artifact, and its encodings

**Re-measured 2026-08-29, and the repository has changed completely since
2026-08-28.** §2.3 recorded `unsloth/GLM-5.3-GGUF` at revision `8cf52b13b130`
holding ONE arm, `UD-Q3_K_XL` at 319.41 GiB. At revision
`346b3591c7f28d1a23716f97a065ecf12ec14771` (`lastModified`
`2026-08-29T02:35:58Z`) it holds **twelve arms, 140 `.gguf` files, 5542.40 GiB**:

| arm | files | size |
|---|---:|---:|
| **UD-IQ1_S** | 6 | **201.83 GiB** |
| UD-IQ1_M | 6 | 212.80 GiB |
| UD-IQ2_M | 6 | 222.19 GiB |
| UD-Q2_K_XL | 7 | 236.44 GiB |
| UD-IQ3_XXS | 7 | 262.34 GiB |
| UD-Q3_K_XL | 9 | 319.41 GiB |
| UD-IQ4_XS | 9 | 340.22 GiB |
| UD-Q4_K_XL | 11 | 435.20 GiB |
| UD-Q5_K_XL | 13 | 523.84 GiB |
| UD-Q6_K_XL | 16 | 637.37 GiB |
| Q8_0 | 17 | 746.32 GiB |
| BF16 | 33 | 1404.42 GiB |

Re-read this table rather than quoting it. The repository was being populated
live on both days this row looked at it.

**The census, and why a name is not a format.** Method: HTTP range requests
against the six `UD-IQ1_S` shards, parsing only the GGUF header — magic,
version, `tensor_count`, the KV block, then each `tensor_info`'s name, dims,
`ggml_type` and offset. Header sizes are 9,428,677 B for shard 1 (metadata only,
0 tensors, carrying the 20 MB tokenizer) and 25-30 kB for shards 2-6. **Nothing
was downloaded**; the four arms below cost ~9.6 MB of range reads in total.
`split.tensors.count` is 1809 and the shards sum to 455+419+412+397+126 = 1809,
so the census is complete rather than sampled.

`UD-IQ1_S`, 1809 tensors:

| ggml type | n | GiB | of which experts | expert GiB | resident | resident GiB |
|---|---:|---:|---:|---:|---:|---:|
| IQ3_XXS | 71 | 81.539 | 71 | 81.539 | 0 | 0.000 |
| IQ1_S | 106 | 62.109 | 106 | 62.109 | 0 | 0.000 |
| IQ2_XXS | 44 | 34.031 | 44 | 34.031 | 0 | 0.000 |
| Q5_K | 312 | 7.154 | 0 | 0.000 | 312 | 7.154 |
| **IQ4_XS** | **4** | **6.375** | **4** | **6.375** | 0 | 0.000 |
| Q8_0 | 476 | 4.852 | 0 | 0.000 | 476 | 4.852 |
| Q2_K | 2 | 1.969 | 2 | 1.969 | 0 | 0.000 |
| Q3_K | 1 | 1.289 | 1 | 1.289 | 0 | 0.000 |
| Q6_K | 82 | 1.000 | 0 | 0.000 | 82 | 1.000 |
| Q4_K | 2 | 0.997 | 0 | 0.000 | 2 | 0.997 |
| F32 | 709 | 0.508 | 0 | 0.000 | 709 | 0.508 |
| **TOTAL** | **1809** | **201.823** | **228** | **187.312** | **1581** | **14.511** |

**`UD-IQ1_S` contains 106 IQ1_S tensors out of 1809.** The name is a target
average, exactly as #2214 warned from the Flash row's `UD-Q2_K_XL`. The same
census over three neighbours:

| arm | expert encodings | resident encodings | resident GiB |
|---|---|---|---:|
| UD-IQ1_S | 106 IQ1_S, 71 IQ3_XXS, 44 IQ2_XXS, 4 **IQ4_XS**, 2 Q2_K, 1 Q3_K | Q8_0/Q5_K/Q6_K/Q4_K/F32 | 14.511 |
| UD-IQ1_M | 76 **IQ1_M**, 74 IQ2_XXS, 71 IQ3_XXS, 4 **IQ4_XS**, 2 Q2_K, 1 Q3_K | same | 14.511 |
| UD-IQ2_M | 148 IQ2_XXS, 71 IQ3_XXS, 4 **IQ4_XS**, 2 IQ2_S, 2 Q2_K, 1 Q3_K | same | 14.621 |
| UD-Q2_K_XL | 148 **IQ2_XS**, 73 IQ3_XXS, 4 **IQ4_XS**, 2 Q2_K, 1 Q3_K | same | 14.621 |

Two facts fall straight out. **The resident class is ~14.5 GiB in every arm** —
the UD recipe keeps every non-expert tensor at Q4_K or better regardless of the
name on the tin — so the residency plan in §3.3 is arm-independent. And
`UD-Q2_K_XL` contains **two** Q2_K tensors out of 1809, both on the MTP block.

**The verdict against our decoders and `vec_dot` lists. This section was
rewritten after `origin/main` moved under it, and the correction inverts the
answer.** At this branch's base `60a6dd97b`, `IQ4_XS` and `IQ2_XS` had neither a
`vt` block dtype nor a decoder, so both were a hard refusal. On 2026-08-29 at
`94de63ff5` ([#2245](https://github.com/mudler/vllm.cpp/issues/2245)) main landed
**the dequantizers for both**, for the sibling `MODEL-MM-GLM53-FLASH` row's own
staged artifact. `kIQ2_XS` and `kIQ4_XS` now exist in `include/vt/dtype.h::DType`,
`gguf_reader.cpp` sizes id 17 at `{256, 74}`, and `gguf_dequant.cpp` cases 17 and
23 decode. **Neither gained a keep-quant `vec_dot`, and that is the half that
decides this row.**

Three lists decide it, and they are not the same list:

1. `gguf_reader.cpp::FindGgmlTraits` — the ggml ids we can SIZE. An id outside it
   throws `"gguf: unknown ggml type id N"` at file OPEN. **17 and 23 are now in.**
2. `vt::BlockDTypeFromGgmlTypeId` + `gguf_dequant.cpp` — the ids we can DECODE.
   **17 and 23 are now in.**
3. `src/vt/cpu/cpu_quant_dot.cpp::BlockVecDot`, read through
   `vt::cpu::HasQuantDotKernel` — the ids that stay COMPRESSED.
   `Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ2_S,
   IQ1_S, IQ1_XXXS, IQ4_NL, MXFP4`. **17 and 23 are NOT in, and nothing else in
   the four censused arms is missing.**

`gguf_keep_quant.cpp::KeepQuantDType` is the gate: it resolves the block dtype
and then `if (!vt::cpu::HasQuantDotKernel(dt)) return false;`. **A type with a
decoder and no `vec_dot` therefore EXPANDS TO bf16 at load** — exactly the
failure mode #2214 named, arriving here through the door that had just been
opened.

| type | traits | decoder | `vec_dot` | what happens |
|---|---|---|---|---|
| Q4_K, Q5_K, Q6_K, Q8_0, F32 | yes | yes | yes | resident class stays compressed |
| IQ1_S, IQ2_XXS, IQ3_XXS, IQ2_S, Q2_K, Q3_K | yes | yes | yes | expert towers stay compressed |
| **IQ4_XS (23)** | **yes** | **yes, since `94de63ff5`** | **NO** | **expands to bf16** |
| **IQ2_XS (17)** | **yes** | **yes, since `94de63ff5`** | **NO** | **expands to bf16** |
| IQ1_M (29) | NO | NO | NO | `gguf: unknown ggml type id 29` at file OPEN |

**And an expanded tower does not merely cost bytes — it leaves the streaming lane
entirely.** `gguf_device_fit.cpp:85-100` walks every `*_exps.weight` tensor,
asks `PeekRoute` for its residency, and returns **false for the whole arm** the
moment one of them is not `kKeepQuant` or `kKeepF16`. The eligibility is
per-MODEL, not per-tensor. So four IQ4_XS tensors out of 228 disqualify all 228.

The cost, computed exactly. One `*_exps` tower is
`2048 * 6144 * 256 = 3,221,225,472` elements, **6.000 GiB at bf16**:

| arm | offending type | compressed | expanded to bf16 | delta |
|---|---|---:|---:|---:|
| UD-IQ1_S | 4 x IQ4_XS | 6.375 GiB | **24.000 GiB** | +17.6 GiB |
| UD-IQ2_M | 4 x IQ4_XS | 6.375 GiB | **24.000 GiB** | +17.6 GiB |
| UD-Q2_K_XL | 148 x IQ2_XS | 128.344 GiB | **888.000 GiB** | +759.7 GiB |

And the slot arithmetic collapses with it: a bf16 expert slice is
`6144 * 2048 * 2 = 25,165,824 B = 24.00 MiB` against the IQ4_XS slice's 6.375
MiB, and slots are uniform at the largest, so §3.3's 4096-slot cache would be
**96.00 GiB** instead of 25.50 GiB — more than three quarters of the device on
its own.

**So the verdict changes shape but not sign, and it is sharper than it was.**

- **The row is blocked on ONE kernel and it is a `vec_dot`, not a decoder:
  `VecDotIQ4_XSQ8_K` against the Q8_K activation encoding.** Four tensors,
  `blk.{8,75,76,77}.ffn_down_exps.weight`. With it, UD-IQ1_S loads entirely
  compressed at 201.823 GiB and every tower is streamable. Without it, the arm
  loads at 219.4 GiB, cannot stream at all, and is dead on this fleet.
- The port is small and well-precedented, and it is smaller today than it was at
  this branch's base: `94de63ff5` already ported the 136-byte `block_iq4_xs`
  layout and its decoder from llama.cpp `b10451`, so what remains is the dot
  product itself over a codebook this tree already carries for `IQ4_NL`
  (`kValuesIq4nl`, `cpu_quant_dot.cpp::VecDotIQ4_NLQ8_0`, anchored `quants.c:1254`).
  Upstream's is `ggml_vec_dot_iq4_xs_q8_K`.
- `UD-IQ2_M` needs the same one and nothing else. `UD-Q2_K_XL` needs
  `VecDotIQ2_XSQ8_K` as well, and `UD-IQ1_M` is still rejected outright on
  `IQ1_M`, which has no traits at all.
- The row already exists: `QUANT-GGUF-IQ4_XS`
  (`.agents/quantization-matrix.md:78`, `INVENTORIED`).

**The general lesson this section paid for, and the reason it is written out
rather than quietly corrected: a decoder and a `vec_dot` are two different
obligations, and landing only the first turns a loud refusal into a silent 3.4x
memory multiplier.** At `60a6dd97b` this arm refused at load with a message
naming the type. At `94de63ff5` it loads, and the only symptom is that a
119.631 GiB device runs out of memory for reasons the log does not name.
`gguf_device_fit`'s all-or-nothing rule is what converts the same defect from
"+17.6 GiB" into "no streaming at all", and neither is visible to a token gate.

**The GGUF's own metadata, and one thing it does NOT carry.** Shard 1's KV block
declares `general.architecture = glm-dsa`, `glm-dsa.block_count = 79`,
`context_length = 1048576`, `embedding_length = 6144`, `expert_count = 256`,
`expert_used_count = 8`, `expert_feed_forward_length = 2048`,
`expert_shared_count = 1`, `expert_gating_func = 2` (sigmoid),
`expert_weights_scale = 2.5`, `expert_weights_norm = true`,
`leading_dense_block_count = 3`, `attention.q_lora_rank = 2048`,
`attention.kv_lora_rank = 512`, `attention.key_length = 576`,
`attention.value_length = 512`, `attention.key_length_mla = 256`,
`attention.value_length_mla = 256`, `rope.dimension_count = 64`,
`rope.freq_base = 8e6`, `nextn_predict_layers = 1`,
`attention.indexer.head_count = 32`, `attention.indexer.key_length = 128`,
`attention.indexer.top_k = 2048`, `tokenizer.ggml.pre = glm4`,
`general.file_type = 24`, and an imatrix provenance block
(`quantize.imatrix.entries_count = 1065`, `chunks_count = 209`).

**It does NOT carry `glm-dsa.attention.indexer.types`, and that is a trap with a
known workaround** — see §3.5, D3.

**Fleet and staging.** `rc devices` on 2026-08-29 lists `dgx:gpu0` (busy),
`orin:gpu0`, `strix:gpu0`, `thor:gpu0`; none is larger than `dgx:gpu0`.
`/mnt/nas_share` has **2.2 TiB free** of 7.3 TiB, so the 201.83 GiB arm stages
there. **`dgx.casa`'s local disk had 184 GiB free when last measured (§0.1 C3,
2026-07-21), which is LESS than the arm**, so W7 must either free local disk or
`pread` across CIFS — and a CIFS-backed `pread` of 1800 slices per token is a
different measurement from a local-NVMe one. §3.9 O7 owes that number.

### 3.5 The delta against `DeepseekV2ForCausalLM`

#### 3.5.1 Upstream, at the pin — and §2.2's premise needed one correction

`registry.py:117` and `deepseek_v2.py:1930-1931` are as §2.1 records, both unique
at `555967922`. `_get_moe_router_dtype` (`deepseek_v2.py:123-133`) forces
`torch.float32` on `model_type == "glm_moe_dsa"` at `:127` before the generic
`moe_router_dtype == "float32"` branch at `:131`, so the special case is
redundant on THIS checkpoint and still fires first. That much §2.2 had right.

**What §2.2 left open, and what is now measured: the pinned class CAN load this
checkpoint, and it does not read `indexer_types` to do it.** At the pin,
`grep -c indexer_types` over `deepseek_v2.py` is 0, and over every `*.py` in the
tree it is 0. `mlp_layer_types` is likewise unread by this model (it exists only
in `cohere2_moe.py` and `mellum.py`). The schedule is DERIVED, at
`deepseek_v2.py:1092-1103`:

```python
_index_topk_freq        = getattr(config, "index_topk_freq", 1)
_index_topk_pattern     = getattr(config, "index_topk_pattern", None)
_index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
if _index_topk_pattern is None:
    _skip_topk = max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq != 0
```

with the indexer built at `:1115` when `self.is_v32 and (not _skip_topk or
is_mtp_layer)`. Evaluated on GLM-5.3 (`freq = 4`, `offset = 3`, 78 layers) that
yields full layers `{0,1,2} ∪ {6,10,…,74}` = **21**, plus the MTP layer forced
full at `:1110-1115`, = **22 indexers**.

**Three independent derivations agree, and they agree bit for bit.**

| source | schedule |
|---|---|
| the checkpoint's `config.json` `indexer_types` | `111000100010001000…` (78 entries, 21 ones) |
| vLLM at the pin, `deepseek_v2.py:1097-1101`, evaluated | identical |
| llama.cpp `b10451`, `src/models/glm-dsa.cpp:6-27` `GLM_5_2_DEFAULT_INDEXER_TYPES` | identical over all 78 |
| the checkpoint's own tensor index | `self_attn.indexer.*` present on **22** of 79 blocks |

Those 22 are the 21 trunk full layers plus block 78, the MTP block — exactly what
the pin builds. The pin also anticipates a checkpoint that ships MORE indexer
weight than it builds, dropping the surplus at `deepseek_v2.py:1566-1582`
("With index_topk_freq>1 only some layers build an indexer, yet the checkpoint
ships indexer weights for all of them"). GLM-5.3 does not need that path, but the
PUBLISHED GGUF does — see D3.

**`n_shared_experts = 1`** is an ordinary read (`deepseek_v2.py:299`, `:349`,
`:352`, `:385`) and needs nothing special.

**The `indexers_proj` question from §2 is now answered, and the answer is that it
names no tensor.** `grep -n indexers_proj` over every `*.py` at the pin returns
zero. The checkpoint's own `model.safetensors.index.json` (118,629 tensors,
`metadata.total_size = 755,617,140,416`, fetched 2026-08-29) ships the upstream
spellings and only those: `self_attn.indexer.{wq_b,wk,weights_proj,k_norm}`, 22
of each, with `wq_b` and `wk` carrying `weight_scale_inv` sidecars and
`weights_proj` and `k_norm` carrying none. `modules_to_not_convert`'s 22
`self_attn.indexers_proj` entries are a quantization-skip shorthand that matches
no shipped tensor name; the tensor it means (`indexer.weights_proj.weight`) is
unquantized anyway. **It is a config-level string, not a naming divergence, and
a loader must not mirror it.** vLLM at the pin fuses `wk` + `weights_proj` into
one `MergedColumnParallelLinear` named `wk_weights_proj` through the stacked
mapping at `deepseek_v2.py:1536-1540`, with an fp8 dequant-into-the-fused-param
helper `_try_load_fp8_indexer_wk` at `:820-860`.

`vllm/models/deepseek_v32/nvidia/attention.py` exists at the pin and implements
the same skip schedule at `:211-219`, but `registry.py:117` routes this
architecture to `deepseek_v2`, so that tree is **not** reached at the pin. The
re-homing §2.2 describes is a `main`-only change and stays out of scope.

#### 3.5.2 Our side — what is free, what is adjacent, what is new

**Free from the existing DeepSeek-V2 + shared MLA stack.** Verified at
`60a6dd97b`:

- **The MLA geometry is already supported and already exercised.**
  `mla::MlaBlockDims::Validate` (`mla_attention.cpp:89-192`) requires
  `v_head_dim <= qk_head_dim()`; GLM-5.3 is `256 <= 192+64 = 256`, which passes,
  and there is no rule forcing `qk_nope_head_dim == v_head_dim`. The prefill
  head-dim switch (`src/vt/cuda/cuda_mla_prefill.cu:194-209`) hits the native
  FA-2 256 instantiation with no padding — the same instantiation GLM-4.7-Flash
  already uses. Decode runs in latent space at `head_size = 576` /
  `v_head_dim = 512` and takes the `<= 512` arm, byte-identical to DeepSeek-V3
  (`cuda_mla_attn.cu:671-682`). **No MLA refusal fires for this model.**
- Load-time `kv_b_proj` absorption at the asymmetric 192/256 split
  (`AbsorbKvBProjBf16`, `mla_attention.cpp:205-229`, splitting at `row = p + v`).
- **Interleaved (GPT-J) RoPE**, which is DeepSeek's default here:
  `MlaBlockDims::is_neox_style` defaults `false` (`mla_attention.h:136`). Upstream
  passes `is_neox_style=False` unconditionally (`deepseek_v2.py:1073`) and reads
  no top-level `rope_interleave`, so our default is parity-correct — **but it is
  correct by default rather than by a read, and W2 writes that down**.
- The **noaux_tc grouped router** at `n_group = 1` / `topk_group = 1`, sigmoid
  scoring, `norm_topk_prob`, `routed_scaling_factor 2.5` and
  `e_score_correction_bias` (`deepseek_v2_weights.cpp:286-341`,
  `deepseek_v2.cpp:355-366`). This is exactly the configuration GLM-4.7-Flash
  already gates end-to-end (§0.1 C2).
- The MoE expert layout has **no hardcoded expert-count limit**
  (`vt::MoeGroupedGemmBf16` validation, `ops.cpp:904-928`, requires only
  `weight_ptrs.Numel() == e`), so 256 x 75 is representable.
- `first_k_dense_replace`-driven dense/MoE layout
  (`DeepseekV2Params::is_moe_layer`, `deepseek_v2.h:126-129`) reproduces
  upstream's rule and is arithmetically identical to the checkpoint's
  `mlp_layer_types` for this config. The batch split, decode CUDA graph and
  paged engine come along unchanged.

**Adjacent and already landed, but not wired to DeepSeek-V2.** This is the
finding that most changes the size of the port:

- **A device-native DSA lightning indexer already lives inside the SHARED MLA
  block**, `mla_attention.cpp:598-745`, landed for `dots3-note`. It is a port of
  upstream's non-fused `Indexer.forward` (`deepseek_v2.py:803-842`): `wq_b` GEMM
  (`:646`), split `wk` / `weights_proj` GEMMs (`:655`, `:658`), `k_norm` as a
  real **LayerNorm with bias at eps 1e-6** (`:663-664`), leading-slice rope under
  an independent `dims.indexer_rope_is_neox_style` (`:667-673`, upstream's
  `not indexer_rope_interleave` at `deepseek_v2.py:1120`), chunked logits under a
  16 Mi-element budget (`:698-712`), then `vt::DsaIndexerLogits` +
  `vt::DsaTopkSelect` per request (`:741-742`), handed to decode at `:880-883`.
  Both ops are implemented and registered on **CPU** (`cpu_dsa_indexer.cpp:184,186`)
  and **CUDA** (`cuda_dsa_indexer.cu:320,322`). Geometry fields
  `index_n_heads` / `index_head_dim` / `index_topk` /
  `indexer_rope_is_neox_style` already exist on `MlaBlockDims`
  (`mla_attention.h:210-232`), as do the five indexer tensors
  (`mla_attention.h:426-430`). **This is a much stronger starting point than
  §0.2's "GB10 cannot run DSA" verdict suggests** — that verdict was about
  vLLM's flashinfer path, not about ours, and ours has since been built.
- The freq/offset + pattern + explicit-list indexer schedule parser is already
  written and gated, in the WRONG model's translation unit:
  `glm5_next.cpp:287-338`, whose fallback at `:322-330` is line-for-line
  upstream's `:1097-1101`. The `mlp_layer_types` reader is at
  `glm5_next.cpp:262-284`. Both are liftable.
- The block-fp8 config reader exists (`fp8_block_quant.{h,cpp}`, reading
  `weight_block_size`, `activation_scheme`, `modules_to_not_convert`), with
  exactly one consumer, the Qwen3.5 **dense** loader.

**Genuinely net-new, in order of size.**

1. **The indexer KV side cache.** Sparse decode today refuses any step in which
   any request RESUMES (`dots3_note_device.cpp:1147-1180`), because the indexer's
   `k` comes from the step's own hidden states and a resumed request needs the
   indexer's own 128-wide cache. Upstream's is `DeepseekV32IndexerCache`
   (`deepseek_v2.py:696-701`), a 132 B/token row in its OWN kv-cache group.
   Tracked as `KV-DSV4-MULTICACHE` ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)).
   **Without it there is no multi-step decode, so there is no gate.** Largest item.
2. **The expert-streaming seam.** §3.2 gap 1: the mechanism is welded into
   `qwen3_5.cpp` and has to be lifted before a second model can reach it.
3. **Sparse prefill.** `MlaPrefillAttentionArgs` has no topk member at all
   (`ops.h` through `:1737`); `MlaPrefillAttention` (`ops.cpp:4159-4230`) has no
   selection arm. Upstream forces ALL tokens through `forward_mqa` for a sparse
   impl (`mla_attention.py:697-702`), so this is not optional at long context.
4. **Per-layer heterogeneous `MlaBlockDims`** — 22 indexer-bearing blocks out of
   79 — plus the **`shared` / `skip_topk` selection-reuse** semantics
   (`vllm/model_executor/layers/mla.py:180`: a skip layer runs no indexer but
   stays `is_sparse` and attends through the preceding full layer's
   `topk_indices_buffer`). Nothing in this tree reuses a prior layer's top-k.
5. **The `IQ4_XS` encoding** (§3.4), owned by `QUANT-GGUF-IQ4_XS`.
6. **A `"glm-dsa"` GGUF arm.** `kGgufArchArms` (`model_loader.cpp:1029-1037`)
   knows `deepseek4`, `muse-glimmer`, `qwen35`, `qwen35moe`, `qwen3next`,
   `qwen4exp`, `glm5next` — and no `deepseek2` and no `glm-dsa`.
   `deepseek_v2_registry.cpp:68-71` throws
   `"Model architecture DeepseekV2ForCausalLM does not support GGUF weights"`.
   The whole GGUF path for this family is net-new.
7. **The fp32 router GEMM.** `deepseek_v2.cpp:350` hardcodes
   `DBuf dlog(d, DType::kBF16, {T, E});`. The softmax/top-k stage is already f32;
   only the gate GEMM is bf16. Small and real.
8. **Registration and the lifting of the tripwire.** `ParseDeepseekV2Params`
   refuses any checkpoint carrying `index_topk`
   (`deepseek_v2_weights.cpp:358-364`) and any `quantization_config`
   (`:365-369`), and refuses `num_nextn_predict_layers > 0` unless
   `allow_mtp_tail` (`:353-357`, which only `Glm4MoeLiteForCausalLM` passes).
   `GlmMoeDsaForCausalLM` appears nowhere under `src/` or `include/`.

**Deliberately NOT in scope.** The safetensors arms. The published bf16/fp8
checkpoint is 703.74 GiB across 141 shards and the DeepSeek-V2 loader holds
`OwnedTensor` host bytes with no streaming path — 57,600 host tensors for the
routed experts alone. There is no MoE-expert block-fp8 rung anywhere in the tree.
**This row ships a GGUF arm and refuses safetensors by name**, which inverts the
usual polarity and is the correct inversion here: the quantized arm is the only
one that can be fed. Recorded as D1. MTP is skipped through `allow_mtp_tail`,
following `glm4_moe_lite_registry.cpp:161,169`; there is no MTP drafter in the
tree at all (`src/vllm/v1/spec_decode/` holds three files, none of them an MTP
proposer). Recorded as O5.

### 3.6 The gate

**The honest headline: no end-to-end token gate against vLLM is reachable on this
fleet, and this section says so before any wave promises one.**

vLLM at the pin implements this architecture and, per §3.5.1, would load this
checkpoint. It cannot RUN it here. The published weights are 703.74 GiB at fp8;
`dgx:gpu0` is 119.631 GiB of unified memory, which is also its host RAM, so
`--cpu-offload-gb` offloads into the same pool it is offloading out of. No fleet
device is larger (`rc devices`, 2026-08-29). vLLM has no GGUF path for this
architecture either. **The denominator does not exist, and that is a measured
absence rather than a missing effort.** AGENTS.md's rule applies directly: say so
plainly, gate against what can actually be run, and do not call the result
token-exact against the runtime.

Four gates ARE reachable, and together they are the row's spine.

**G1 — module parity against the pinned vLLM, on CPU, at small shapes.** vLLM's
`Indexer`, `_get_moe_router_dtype`, the skip-topk schedule formula and the
noaux_tc router are all importable and runnable without the checkpoint. Capture
goldens out of `555967922` on synthetic inputs and compare numerically, not by
token. This gates the primitives in items 4 and 7 of §3.5.2 and it is the ONLY
place vLLM is the reference. Precedent: `MODEL-MM-GLM53-FLASH` W3
([#2213](https://github.com/mudler/vllm.cpp/issues/2213)) gated its indexer this
way against transformers, asserting **SET equality of the selected indices** with
the margin printed — the right shape for a discrete selection, where the error is
bimodal and a tolerance bounds nothing.

**G2 — the structural loader gate, headers only, env-gated.** Every tensor in the
real `UD-IQ1_S` shards is enumerated and accounted: 1809 == 1809, zero
unaccounted, and every `ggml_type` in the file is one this tree can decode. This
is the gate that would have caught `IQ4_XS` before a wave was planned, and it
costs ~9.6 MB of range reads, so CI can run it against the published repository
without the asset.

**G3 — the streaming self-consistency gate, and it needs no oracle at all.** The
row's novelty is the streaming mechanism, and its correctness question is
internal: **a streamed slice and the resident tower must produce identical
logits.** `tests/vllm/model_executor/test_expert_stream_wiring.cpp:215` already
asserts exactly this for Qwen3.5 through `SetForceFallback`, inside one process.
Extended to a GLM-5.3-shaped synthetic model it gates the seam lift, the capacity
refusal and the slot arithmetic, on CPU, with no checkpoint. **This is the gate
that decides whether W3 landed correctly**, and it is available from W3 onward.

**G4 — an end-to-end floor against llama.cpp `b10451`, on the IDENTICAL
artifact, labeled as a secondary floor and never as the bar.** llama.cpp reaches
this architecture at our stock pin (§2.4) and, unlike vLLM, can run it: it mmaps
the GGUF and pages from disk. Run `llama-cli` on the same `UD-IQ1_S` shards, same
prompts, greedy, and compare. **Expect a near-tie band and not token-exactness**,
because two independent i-quant implementations agree on the dequantized values
but not on reduction order, and `bf16` stores absorb the difference unevenly.
Ratify the band before running, or the run becomes an argument. Two preconditions
this section does NOT wave away: the artifact must be staged (O7), and the
llama.cpp side must itself be shown to load and generate before a single number
from it is quoted — `gateable = yes` is a property of the oracle, and running
THIS model on it is a separate measurement.

**What no gate here does.** None of the four is token-exact against vLLM, and no
wave may report one as if it were. No speed axis has a denominator: vLLM cannot
run the model, so the only comparable is llama.cpp on the same artifact, and that
is a labeled secondary floor. Per AGENTS.md the speed axis is therefore an **open
gap by construction**, not a waiver and not silence.

### 3.7 Work breakdown

Eight waves. Each is a separate `row/MODEL-TEXT-GLM-MOE-DSA-W<n>` branch, a
separate pull request, a fresh implementer and a fresh reviewer. **W1-W4 and W6
are CPU-gateable and need no GPU. W5, W7 and W8 need a GPU.** Sizes are the
author's estimate of reviewable diff, not a budget. W1 and W2 are independent of
each other; everything else is ordered.

#### W1 — the `IQ4_XS` encoding (CPU, medium) — **LANDED, by another row**

**DISCHARGED on `origin/main` before this row reached it, and verified at the
three sites rather than taken on report.** `2e9f4d88d`
([#2247](https://github.com/mudler/vllm.cpp/issues/2247),
[#2256](https://github.com/mudler/vllm.cpp/issues/2256)) landed the keep-quant
`vec_dot` for the sibling Flash row: `VecDotIQ4_XSQ8_K` is defined at
`cpu_quant_dot.cpp:844` and dispatched at `:1004`, its `QuantTypeTraits` row is
`MakeTraits(DType::kIQ4_XS, DType::kQ8_K)` at `cpu_quant_traits.cpp:121-122`,
and `KeepQuantDType` gates on `HasQuantDotKernel` at `gguf_keep_quant.cpp:176`.
So `HasQuantDotKernel(kIQ4_XS)` is TRUE and the type is no longer expanded to
bf16. `QUANT-GGUF-IQ4_XS` records the same at `quantization-matrix.md:78`
(`C = Y` since #2247). **`kIQ2_XS` landed in the same commit** (`:1003`), which
discharges the IQ4_XS and IQ2_XS halves of O2 and O3 together.

The original scope is kept below as history. Nothing in it remains to do.

**Scope:** the keep-quant `VecDotIQ4_XSQ8_K` and its `QuantTypeTraits` row, so
`vt::cpu::HasQuantDotKernel(kIQ4_XS)` becomes true and
`gguf_keep_quant.cpp::KeepQuantDType` stops expanding the type to bf16. The
dtype, the 136-byte block layout and the decoder already landed at `94de63ff5`
([#2245](https://github.com/mudler/vllm.cpp/issues/2245)); this wave is the half
that was not in it. **Owned by `QUANT-GGUF-IQ4_XS`**
(`.agents/quantization-matrix.md:78`, `INVENTORIED`), consumed here; this row
does not steal that row's state.
**Exclusions:** no model code. `VecDotIQ2_XSQ8_K` is the same shape and is NOT
in scope, because no arm this row targets needs it; `IQ1_M` stays unimplemented
and `UD-IQ1_M` stays refused.
**Anchors:** llama.cpp `b10451` `ggml/src/ggml-common.h::block_iq4_xs` (256
elements, 136 bytes) and `ggml/src/.../quants.c::ggml_vec_dot_iq4_xs_q8_K`; the
shared 16-entry `kValuesIq4nl` codebook this tree already carries for `IQ4_NL`
(`cpu_quant_dot.cpp::VecDotIQ4_NLQ8_0`, anchored `quants.c:1254`); the reader
already sizes it at `gguf_reader.cpp` case 23, `{256, 136}`.
**Tests:** RED first — `HasQuantDotKernel(kIQ4_XS)` is false today and
`KeepQuantDType(23, ...)` returns false, so a test asserting a real IQ4_XS tensor
loads COMPRESSED fails before the change and passes after. Then the `vec_dot`
against the existing dequant-composite fallback on the same blocks, and the
LOWER bound a quantized arm needs: the kept-quant result must not merely
correlate with the expanded one, it must agree to the encoding's own error.
`tests/vt/iq2xs_iq4xs_golden_vectors.h` already carries `94de63ff5`'s reference
vectors.
**Gate:** focused ctest, full preflight. **Reachability:** the type must arrive
through `GgufFile::OpenOne` on a real header, not through a hand-built block.
**Stop:** if the 136-byte layout does not reproduce llama.cpp byte for byte,
return `NEEDS_DECISION` rather than widening a tolerance.

#### W2 — config, registration, GGUF arch arm, refuse-by-name (CPU, medium)

**Scope:** a `glm_moe_dsa` config parser that resolves the indexer schedule by
upstream's DERIVED rule (`index_topk_freq` / `index_skip_topk_offset` /
`index_topk_pattern`, `deepseek_v2.py:1092-1103`) with the explicit
`indexer_types` list as an override, lifting the parser at
`glm5_next.cpp:287-338` rather than writing a second one; the `mlp_layer_types`
reader (`glm5_next.cpp:262-284`) with its `first_k_dense_replace` fallback and a
refusal when the two disagree; `GlmMoeDsaForCausalLM` registered from its own
translation unit; a `"glm-dsa"` row in `kGgufArchArms`
(`model_loader.cpp:1029-1037`); and a `Forward` that refuses by name, naming
every unimplemented primitive and this section.
**Exclusions:** no forward math, no loader materialization, no change to
`DeepseekV2Params` or to the DeepSeek-V2 refusals — GLM-5.3 gets its own params
struct, because sharing one would make the `index_topk` tripwire
(`deepseek_v2_weights.cpp:358-364`) a choice rather than a wall for DeepSeek-V2.
**Anchors:** `deepseek_v2.py:1092-1103`, `:1110-1115`, `:127`;
`glm5_next.cpp:262-338`; registration pattern
`glm4_moe_lite_registry.cpp:18-38`; refusal pattern `kimi_k3.cpp:44-51`.
**Tests:** the derived schedule equals the checkpoint's `indexer_types` for all
78 entries, as a committed fixture from the real `config.json` (this is the test
that makes §3.5.1's three-way agreement executable); the `mlp_layer_types`
disagreement refusal; `is_neox_style == false` asserted rather than defaulted;
a `glm-dsa` GGUF header reaches the config builder through
`LoadedEngine::FromModelDir`; the refusal message names each missing primitive.
**Gate:** CPU build, focused ctest, full preflight. **Evidence:** the registry
contract test's architecture count moves by exactly one.
**Reachability:** deleting the `REGISTER_VLLM_MODEL` line, or the `kGgufArchArms`
row, must red the focused gate.

#### W3 — lift the expert-streaming seam out of `qwen3_5.cpp` (CPU, large)

**Scope:** move `Qwen35ExpertStream`, `KqExpertSlice`, `KqHostSliceView`,
`Reserve` and the step guard (`qwen3_5.cpp:5725`, `:6180`, `:6169`, `:6284`) into
a shared header + translation unit that a second model TU can include, with
Qwen3.5 rewritten as its first client and byte-identical behaviour. **Plus the
capacity refusal §3.3 argues for**: a configured slot count below
`n_moe_layers * 3 * num_experts_per_tok` refuses at load, by name, instead of
degrading to the mmap fallback.
**Exclusions:** no policy change (the LFU stays), no prefetch, no async I/O, no
device store — those are `ENG-EXPERT-STREAM` W6 and
`ENG-EXPERT-STREAM-DEVICE` W2 and this row does not take them.
**Anchors:** `expert_streamer.{h,cpp}`, `expert_slot_cache.{h,cpp}`,
`host_expert_slot_store.h`, `gguf_expert_span.h:12-16`,
`gguf_device_fit.cpp:95`, `model_loader.cpp:2472`.
**Tests:** **G3** — a streamed slice and the resident tower produce identical
logits, extended from `test_expert_stream_wiring.cpp:215` to a model with more
than 4 experts and more than 4 layers; the capacity refusal RED first; Qwen3.5's
six existing streaming binaries stay green and its goldens byte-identical.
**Gate:** focused ctest, full preflight, Qwen3.5 SACRED inertness.
**Reachability:** deleting the seam's call site in `qwen3_5.cpp` must red the
Qwen3.5 streaming suite.
**Stop:** if the lift cannot preserve Qwen3.5 byte-identity, return
`NEEDS_DECISION`; a behaviour change to a gated model is not this wave's to make.

#### W4 — the heterogeneous indexer schedule and selection reuse (CPU, medium)

**Scope:** per-layer `MlaBlockDims` so 22 of 79 blocks carry an indexer and 57 do
not; the `skip_topk` semantics — a shared layer runs no indexer, stays
`is_sparse`, and attends through the preceding full layer's selection
(`vllm/model_executor/layers/mla.py:180`); the fp32 router gate GEMM
(`deepseek_v2.cpp:350`).
**Exclusions:** no KV cache work, no prefill work.
**Anchors:** `deepseek_v2.py:1115`, `:1134-1135`, `:1175`;
`vllm/model_executor/layers/mla.py:180`; ours
`mla_attention.cpp:414`, `:598-745`, `:880-883`;
`_get_moe_router_dtype` `deepseek_v2.py:123-133`.
**Tests:** **G1** — the selection a shared layer uses is byte-identical to the
one its owning full layer produced, mutation-proven by re-pointing it at a
different layer; the router GEMM's output dtype asserted as f32 against a vLLM
golden; a full layer and a shared layer produce DIFFERENT attention outputs (the
tautology guard).
**Gate:** focused ctest, full preflight. **Reachability:** the schedule must
arrive from the config parsed in W2, not be constructed in the test.

#### W5 — the indexer KV side cache (GPU, large) — [#1925](https://github.com/mudler/vllm.cpp/issues/1925)

**STRUCK FROM THIS ROW 2026-08-30, and RE-SEQUENCED as a consumption site.**
This is not a deferral: the wave does not belong to this row at any date, and
what remains here is the site that consumes `KV-DSV4-MULTICACHE` W5 once that
row lands it. The conditional below is retained because reading how it resolved
is the point.

**Scope:** the indexer's own 132 B/token cache in its own kv-cache group, so a
resumed request no longer refuses. This is `KV-DSV4-MULTICACHE`'s work and this
row consumes it; if that row does not schedule it, this row's W5 is where it
lands and the ownership is recorded in both places before a line is written.

**The condition resolved FALSE.** `KV-DSV4-MULTICACHE` HAS scheduled W5. It
carries a `### W5 design` section (W5-1 through W5-6) tracked by
[#2323](https://github.com/mudler/vllm.cpp/issues/2323), and its first
implementation commit — W5-2's "the refusal becomes a gated dispatch" — is
already written on `row/KV-DSV4-W5-IMPL`. Three further branches are working the
same by-name multi-KV plumbing concurrently (`ENG-MULTIKV-BYNAME`,
`ENG-MULTIKV-FORWARD-1925`, `MODEL-MM-GLM53-FLASH-MULTIKV`). So this row
CONSUMES that channel and takes no cache-plumbing scope of its own; the decision
is recorded in `.agents/specs/kv-dsv4-multicache.md` beside its W5 boundary
table, per the sentence above.

**Two further reasons, either of which is independently sufficient**, recorded
so a later reader does not reopen this on the ownership point alone:

1. **The test this wave named would have deleted another row's guard.** The
   refusal below guards `Dots3NoteForCausalLM` and belongs to
   `MODEL-DOTS3-NOTE` ([#699](https://github.com/mudler/vllm.cpp/issues/699)) —
   a different model. Its own comment already says the indexer cache "is
   `KV-DSV4-MULTICACHE` (#1925), **not this row**", and `kv-dsv4-multicache.md`
   W5-2 says **"deleting the refusal is the one thing W5 must not do"**,
   because it would restore the silent-discard failure W3 built it to prevent.
   The replacement is a DISPATCH on a declared capability, not an absence.
2. **There is no GLM-5.3 forward to wire a cache into.**
   `glm_moe_dsa.cpp` is a config parser plus `kForwardRefusal`; W1 (`IQ4_XS`)
   and W7 (the loader and the streamed towers) are both undone, so no GLM-5.3
   weight can be materialized. A cache routed into a refusal stub is a shell
   under `## Nothing lands dead`, gateable only by a unit test that constructs
   the type by hand — which proves the class works, never that anything reaches
   it.

**What is left here, and its shape.** Not a wave — a CONSUMPTION SITE that
opens when `KV-DSV4-MULTICACHE` W5 lands, built on the form
`MODEL-MM-GLM53-FLASH` already proved: `glm5_next_kv.{h,cpp}`,
`ModelFactory::consumes_multi_kv_cache`, and each layer resolving its own caches
by the names they were published under, where an unresolved name refuses rather
than falling back. It is ordered after W7, because a consumption site needs a
forward and a loaded weight to consume anything, and W7 supplies both.
**Exclusions:** sparse prefill, which is W6.
**Anchors:** `DeepseekV32IndexerCache` `deepseek_v2.py:696-701`; the
`MLAAttentionSpec` merge rule `vllm/v1/kv_cache_interface.py:399-429` that forces
it into a separate group; our refusal `dots3_note_device.cpp:1147-1180`.
**Tests:** a two-step decode with a resumed request produces the same tokens as
the same prompt decoded in one step; the refusal at
`dots3_note_device.cpp:1147-1180` is deleted and its replacement is gated, not
merely absent.
**Gate:** focused ctest on GPU, full preflight, dots3-note inertness.
**Needs a GPU.**

#### W6 — sparse prefill (GPU, large)

**Scope:** a topk/selection arm on `MlaPrefillAttentionArgs` and
`MlaPrefillAttention`, mirroring upstream's rule that a sparse impl forces ALL
tokens through the MQA path with no prefill/decode split
(`vllm/model_executor/layers/attention/mla_attention.py:697-702`).
**Exclusions:** no change to the dense prefill path any other model takes.
**Anchors:** `mla_attention.py:697-702`; ours `ops.h` `MlaPrefillAttentionArgs`,
`ops.cpp:4159-4230`.
**Tests:** prefill selection SET-equal to decode selection on the same context;
DeepSeek-V2 and GLM-4.7-Flash prefill byte-identical.
**Needs a GPU.**

#### W7 — the loader, the streamed towers, and the first load (GPU + large asset)

**Scope:** the `glm-dsa` GGUF weight loader; `_exps.weight` towers routed to the
W3 seam; the resident class staged to device; safetensors refused by name (D1);
`allow_mtp_tail` skipping block 78. Stage `UD-IQ1_S` (201.83 GiB, 6 shards) to
`/mnt/nas_share` — 2.2 TiB free — and record the sha256 of each shard.
**Exclusions:** no speed number.
**No decoder or keep-quant question remains on this path (verified 2026-08-30):**
all six encodings the `UD-IQ1_S` arm uses — IQ1_S, IQ3_XXS, IQ2_XXS, IQ4_XS,
Q2_K, Q3_K — have a `vec_dot` row in `cpu_quant_dot.cpp` on `origin/main`, 6 of
6, so the arm keeps its blocks and `gguf_device_fit.cpp`'s all-or-nothing rule
admits it to the streaming lane. `IQ1_M` is still absent, which refuses only the
`UD-IQ1_M` arm this wave does not stage.
**Tests:** **G2** structurally over the real shard headers, env-gated; the model
loads and produces a first token; the resident footprint measured against the
14.511 GiB this section predicts, and the difference explained if it is not
within a few percent.
**Gate:** the load itself, under an `rc` lease on `dgx:gpu0`.
**Needs a GPU and the asset.** **Stop:** if `dgx.casa`'s local disk cannot hold
201.83 GiB, do NOT quietly `pread` across CIFS and report the result as a
streaming measurement — record it as a CIFS number and open O7's measurement.

#### W8 — the gates, once and only once a load exists (GPU + asset)

**Scope:** G4 against llama.cpp `b10451` on the identical artifact, with the band
ratified in advance and the oracle's own ability to run this model demonstrated
first. Then, and only then, the speed axis — recorded as an open gap with a
labeled secondary floor and no vLLM denominator (§3.6).
**Exclusions:** no correctness claim that names vLLM as the runtime denominator.
**Needs a GPU and the asset.**

#### W9 — the forward, the first token, and the seam's second client (GPU + asset)

**Why this block exists at all.** §3.7 was authored as eight waves and none of
them is the forward. W5 was struck to `KV-DSV4-MULTICACHE` (#2323), W6 is sparse
prefill and W7 was the loader — whose TEST list said "the model loads and
produces a first token" and whose SCOPE paragraph said only the loader. W7
delivered the scope and not the second half of the test line, and `## Owed` O21
records that plainly. This block is the wave O21 names, written before its code
as `## Spec before code` requires, and it is deliberately numbered W9 rather
than folded into W7, because W7 has landed and a landed wave's scope is not
edited afterwards to cover work it did not do.

**Scope.** `GlmMoeDsaModel::Forward` / `ForwardDevice` replacing
`kForwardRefusal` with logits, composed only of seams that already exist:

1. a per-layer `RunLayer` driving `GlmMoeDsaMlaSchedule(p)`'s 78 `MlaBlockDims`
   in order, over ONE `mla::MlaSharedSelection` allocated per forward and handed
   to every layer — which is what makes a `kShared` layer read the selection its
   owning `kFull` layer wrote (`mla.py:180`). This discharges O19: the eleventh
   argument of `ForwardMlaAttentionBlock` gets its first production caller.
2. a sparse-step builder that populates `meta.indexer_cu_seqlens_q`, and the
   refusal that is the exact complement of the route predicate;
3. the 256-expert MoE routed through `expert_stream::ExpertSlice`, which makes
   GLM-5.3 the seam's SECOND client and discharges O15;
4. the fp32 router gate GEMM, sized from `router_dtype_is_f32` exactly as
   `deepseek_v2.cpp:363` sizes it;
5. `ModelFactory::streams_routed_experts = true` on `kGlmMoeDsaFactory` — O22,
   which is a claim about the forward and may only be made once the forward
   reads through the slot seam. Without it `CheckDeviceWeightFit` charges
   187.312 GiB against 119.631 GiB and refuses the load, so the flag and the
   forward land together or neither is testable on `dgx:gpu0`.

**Exclusions.** No speed number and no denominator (O10 stands). No sparse
prefill beyond what a FIRST token needs — W6 owns it, and this wave does not
touch `MlaPrefillAttentionArgs` or `MlaPrefillAttention`. No indexer KV side
cache: a resumed request is REFUSED BY NAME and O4 keeps its owner. No MTP. No
safetensors arm. No change to `deepseek_v2.cpp`, `qwen3_5.cpp` or the Qwen3.5
streamed lane.

**Upstream anchors, verified at parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`.**
`DeepseekV2DecoderLayer.forward` `deepseek_v2.py:1262-1345`;
`DeepseekV2MoE.forward` `:395-424`; the shared `topk_indices_buffer`
`:1372-1377` allocated once per model and handed to every layer at `:1395`; the
reuse itself `vllm/model_executor/layers/mla.py:180`; `_get_moe_router_dtype`
`:123-133` with the `glm_moe_dsa` special case at `:127`; the whole-step MQA
promotion `vllm/model_executor/layers/attention/mla_attention.py:829-851` and
its `use_dense_mha` source `sparse_mla_attention.py:296-299`; the absorbed
decode form `mla_attention.py:739-830`; `process_weights_after_loading`
`mla_attention.py:875-962` and its own reason for keeping the absorbed halves
unquantized at `:876-878`.
Ours: `mla::ForwardMlaAttentionBlock` and `mla::MlaSharedSelection`
(`mla_attention.h`), `mla::AbsorbKvBProjBf16`, `expert_stream::ExpertSlice` /
`ExpertStreamStepGuard` (`expert_stream_seam.h`), `vt::MoeRouterTopK`,
`vt::FusedChain`, `BuildMlaStep` (`deepseek_v2.h`), and the sparse-step shape
`BuildDots3NoteSparseStep` already builds (`dots3_note_device.cpp`).

**Five findings this wave measured before writing a line, each of which changes
the work.** O21 named two of them and asked for both to be verified; they are,
and three more came with them.

- **F1 — `ForwardMlaAttentionBlock`'s `shared` parameter has no production
  caller.** VERIFIED. `git grep ForwardMlaAttentionBlock` over `src` returns
  four production call sites — `deepseek_v2.cpp:515`, `dots3_note_device.cpp`,
  `kimi_linear_device.cpp`, `minicpm3.cpp` — and every one of them passes ten
  arguments. O21 was right, and this wave is the caller.
- **F2 — `BuildMlaStep` never sets `indexer_cu_seqlens_q`.** VERIFIED. The only
  producer in the tree is `BuildDots3NoteSparseStep`. GLM therefore needs its
  own builder, and it needs it for the same reason dots3-note does: the field
  being non-empty is what declares the whole step MQA.
- **F3 — THE PUBLISHED FILE'S `attn_q_a` AND `attn_kv_a_mqa` CARRY DIFFERENT
  GGML TYPES, so the fused A-projection the MLA seam requires cannot be built by
  concatenation.** Read off the real shard 2 header (`GLM-5.3-UD-IQ1_S-00002-of-
  00006.gguf`, 455 tensors): `blk.0.attn_q_a.weight` is `[2048, 6144]` **Q5_K**
  and `blk.0.attn_kv_a_mqa.weight` is `[576, 6144]` **Q8_0**. Upstream fuses them
  because `packed_modules_mapping` fuses them (`deepseek_v2.py:1812-1820`) out of
  a bf16 checkpoint where a row concatenation is meaningful; here the two halves
  are two different quantizations and no byte concatenation exists. The seam is
  therefore EXTENDED, additively: `MlaBlockWeights` grows a `q_a_proj`, and the
  `has_q_lora()` branch takes `{q_a_proj, kv_a_proj_with_mqa}` when it is present
  and `fused_qkv_a_proj` otherwise. The arithmetic is IDENTICAL — the fused arm
  already slices the weight's output rows and issues one GEMM per slice
  (`mla_attention.cpp`, the recorded DEVIATION above the A-projections), so
  supplying the two slices as two weights is the same two GEMMs on the same
  bytes. Exactly one of the two must be present; both, or neither, is refused by
  name. Nothing else in the tree sets `q_a_proj`, so every existing registration
  is byte-identical.
- **F4 — the GGUF's `attn_k_b` and `attn_v_b` are the PER-HEAD TRANSPOSES of the
  seam's `w_uk_t` and `w_uv`, and no quantized batched GEMM exists to consume
  them in place.** `blk.0.attn_k_b.weight` reads `[64, 512, 192]` =
  `[heads, kv_lora_rank, qk_nope_head_dim]` and `blk.0.attn_v_b.weight` reads
  `[64, 256, 512]` = `[heads, v_head_dim, kv_lora_rank]`, both Q8_0 — llama.cpp's
  `ggml_mul_mat` contracts over `ne[0]`, so its rows run along the contraction
  axis. The seam's decode arm needs `w_uk_t` `[heads, qk_nope, kv_lora]` and
  `w_uv` `[heads, kv_lora, v_head]` for `vt::BatchedMatmul`, whose contract is
  "a/b share f32 or bf16" and "only the innermost dimension must be unit-stride"
  — so neither a transposed view nor a quantized operand is admissible. This is
  upstream's own situation and upstream's own answer: `W_UK_T` and `W_UV` are
  plain bf16 copies produced by `process_weights_after_loading`, and
  `mla_attention.py:876-878` says why ("we currently do not have quantized bmm's
  which are needed for W_UV and W_UK_T ... the extra memory overhead of this is
  fairly low"). So this wave adds upstream's own stage, under upstream's own
  name — a POST-LOAD ABSORPTION that rebuilds the checkpoint-layout `kv_b_proj`
  from the two GGUF halves and then calls the SHARED `mla::AbsorbKvBProjBf16` on
  it rather than writing a second absorber. It runs in the LOADER, which is
  where `deepseek_v2_weights.cpp:138-153` already calls that same absorber for
  the safetensors arm: the stage needs the dequantizer and the open `GgufFile`,
  and both live there.
  **It costs bytes and the number is stated rather than discovered:** 58.8 MB per
  layer over 78 layers = **4.48 GiB on top of §3.3's 14.511 GiB resident**, so the
  resident class this wave actually loads is ~18.99 GiB and O9's prediction is
  restated against it rather than quietly failed.
- **F5 — the published file still states no indexer schedule, confirmed by
  reading its own KV block a second time.** Shard 1 is 9,428,677 bytes, 0
  tensors, **64 keys**, and none of `glm-dsa.attention.indexer.types`,
  `index_topk_freq` or `index_skip_topk_offset` is among them. O17's diagnosis
  holds on the staged bytes, so the run in this wave feeds the file
  `scripts/glm-dsa-write-indexer-types.py` repaired, and records the sha256 of
  the published shard 1 and of the derived one side by side. The loader's
  refusal is not weakened.

**Tests.**

- **The model produces a first token, through the production entry point.**
  `LoadedEngine::FromModelDir` on the W7 synthetic `glm-dsa` fixture, then
  `engine().generate(prompt_ids, greedy, id)` — the same two calls a user makes.
  A unit test that called `GlmMoeDsaModel::Forward` directly would prove the
  class works and nothing about whether anything reaches it.
- **Every float comparison is guarded by `isfinite`, and the logits are PRINTED.**
  The sibling Flash row read an all-NaN forward as a PERFECT match, because every
  comparison against a NaN is false, and then emitted token id 0 eight times. So
  the gate asserts the logits are finite BEFORE it asserts anything about their
  values, and the run report carries NaN/Inf counts, min/max/mean/sd and the
  top-5 `(id, logit, decoded piece)` whether or not the token looks sensible.
  Uniform logits and NaN logits both argmax to 0; neither is a token.
- **The selection reaches attention, and a shared layer reuses it.** A `kShared`
  layer's selection is byte-identical to the one its owning `kFull` layer wrote,
  and the two layers produce DIFFERENT attention outputs (the tautology guard).
  Mutation: re-point the shared buffer at a different layer.
- **The MoE routes through the seam.** `ExpertStreamLane::SetForceFallback`
  proves the streamed slice and the resident tower produce identical logits
  inside one process — G3, extended to a GLM-5.3-shaped model, which is what
  makes GLM the seam's second client rather than a claim that it is.
- **The router GEMM's output dtype is f32.** O18 records that no numerical gate
  can SEE this on a tiny fixture; what is gated is that `router_dtype_is_f32`
  reaches the buffer that is allocated, proven by mutation rather than by a
  tolerance.
- **Reachability.** The production call site of the forward is deleted in a
  scratch copy and the focused gate must red. The tree is restored
  byte-for-byte and hashed. A mutation killed by the COMPILER is weaker than one
  killed by an assertion and is rewritten to compile.

**Gate.** `scripts/agent-preflight.sh --fail-on-skip` with zero skips; the
focused C++ suites run by hand with their counts; Qwen3.5 streaming inertness,
because this wave adds a second client to a lane Qwen3.5 owns; dots3-note
inertness, because the sparse-step eligibility predicate is lifted into the MLA
seam and dots3-note is made to call the lifted one rather than keeping a second
copy. Then the real load on `dgx:gpu0` under an `rc` lease, `--max-tokens 1`
first, resumable.

**Stop conditions.**

- The artifact is incomplete or fails its sha256: report and stop. Do not start
  a second download and do not kill the running `curl`.
- The forward refuses on the real artifact: report the refusal VERBATIM and
  stop. An honest refusal naming the missing piece has been worth more than a
  speculative patch on every wave of this row.
- A degenerate token (id 0, a repeated character, uniform logits): print the
  distribution first and theorise second.
- `dgx:gpu0` cannot hold the resident class at F4's restated 18.99 GiB: record
  it against O9 as a measured miss rather than adjusting the prediction.
- Anything that would need the indexer KV side cache: refuse by name and leave
  O4 with `KV-DSV4-MULTICACHE`. A FIRST token on a fresh prompt does not need
  one; a SECOND does.

### 3.8 Risks and decisions taken in this section

**D1 — the GGUF arm ships and the safetensors arms are refused by name.** This
inverts `porting-a-model.md`'s usual polarity, which treats bf16 as the base arm
and the quantized arms as the obligation. Here the bf16/fp8 checkpoint is 703.74
GiB with no streaming loader and no MoE block-fp8 rung, and the GGUF arm is the
only one that can be fed on this fleet. The refusal names the missing pieces so a
reader meets it at load rather than discovering it.

**D2 — `UD-IQ1_S` is the target arm.** Smallest at 201.83 GiB, needs exactly one
keep-quant `vec_dot` (`IQ4_XS`), and its resident class is the same 14.5 GiB as
every larger arm. `UD-IQ2_M` (222.19 GiB) is the fallback and needs the same
single kernel, so W1 unlocks both. `UD-Q2_K_XL` additionally needs
`VecDotIQ2_XSQ8_K`, and `UD-IQ1_M` is rejected outright because `IQ1_M` has no
reader traits. None of the four is rejected on size.

**D3 — the published GGUF's indexer schedule cannot be read out of the file, and
the port must not try.** The file declares indexer weights on **all 79 blocks**
while the checkpoint ships them on 22, so the conversion broadcast the shared
layers' weights — ~770 MB of duplicated Q8_0 — and it does **not** write
`glm-dsa.attention.indexer.types`, which `b10451`'s converter would have written
(`conversion/glm.py:337-340`). llama.cpp survives this by falling back to a
HARDCODED table: `is_pre_5_2 = n_ctx_train < 1048576` is false for this model
(`max_position_embeddings` is exactly 1048576), so it uses
`GLM_5_2_DEFAULT_INDEXER_TYPES` (`src/models/glm-dsa.cpp:6-27`), which §3.5.1
verified is bit-identical to GLM-5.3's list. **We do not copy that table.** W2
derives the schedule from `index_topk_freq` / `index_skip_topk_offset` the way
vLLM does, reads `indexer.types` when present, and refuses when a file declares
neither and the derivation is unavailable. A hardcoded 78-entry constant that
happens to be right is the shape that silently becomes wrong on GLM-5.4.

**D4 — the row moves to `SPIKE`, not to `READY` or `ACTIVE`.** Scoped in a
committed spec, not implemented. It leaves `SPIKE` when W2 lands.

**R1 — the slot cache has never run at this scale.** 1800 protected slices per
step against a policy whose eviction is an O(resident) linear scan
(`expert_slot_cache.cpp:26-44`) and whose fills are 1800 serialized blocking
`pread`s. Nothing in the tree has run the `pread` path on a real checkpoint at
all. W7 is where this becomes a number, and it may be the number that reopens the
blocked verdict on throughput grounds rather than capacity grounds.

**R2 — batch is the capacity axis, not context.** At concurrency `c` the distinct
expert set per layer is bounded by `min(256, 8c)`, so the working set grows to the
whole 187 GiB tower set well before `c = 32`. The row's viability claim is a
`c = 1` claim and W8 must say so beside every number.

**R3 — `dgx:gpu0`'s viability depends on it being a GB10.** The production
predicate is `cpu || host_memory_is_device_addressable()`
(`qwen3_5.cpp:6199`); a discrete CUDA part answers false and falls through to
`KqResidentSlice`. Unified memory is what makes the host slot arena readable by
the device without a device store, and `ENG-EXPERT-STREAM-DEVICE` W2 — the
virtual `SlotForRead` — is what a discrete part would need. This row does not
take that work; it records that the port is GB10-shaped until that lands.

**R4 — the streamed and grouped MoE paths are mutually exclusive**
(`qwen3_5.cpp:6307-6312`). Every speed number on this row is a
grouped-MoE-disabled number, and that has to be said each time rather than once.

### 3.9 Owed

- **O1 — no end-to-end token gate against vLLM exists or can exist on this
  fleet** (§3.6). Owed against a device that can hold 703.74 GiB, or against a
  multi-device execution path this project does not have. Tracked by
  [#2214](https://github.com/mudler/vllm.cpp/issues/2214). Discharged by either
  of those two things and by nothing else.
- **O2 — DISCHARGED 2026-08-30 by `2e9f4d88d`** (#2247, #2256), which landed
  `VecDotIQ4_XSQ8_K` (`cpu_quant_dot.cpp:844`, dispatched `:1004`) and its
  traits row (`cpu_quant_traits.cpp:121-122`). `HasQuantDotKernel(kIQ4_XS)` is
  true, so the arm keeps its blocks and stays in the streaming lane. Verified at
  the three sites on `origin/main`, not taken on report. Original text: `IQ4_XS`
  has a decoder and no keep-quant `vec_dot`, so the target arm
  loads by EXPANDING four expert towers from 6.375 GiB to 24.000 GiB and, worse,
  drops out of the streaming lane entirely** (`gguf_device_fit.cpp:85-100` is
  all-or-nothing across a model's `*_exps` tensors). Discharged by W1 landing
  `VecDotIQ4_XSQ8_K`. Owned by `QUANT-GGUF-IQ4_XS`.
- **O3 — HALF DISCHARGED 2026-08-30. `IQ2_XS` (id 17) landed in the same
  commit as O2** (`2e9f4d88d`, dispatched `cpu_quant_dot.cpp:1003`), so
  `UD-Q2_K_XL` no longer expands. **`IQ1_M` (id 29) still has no reader traits
  at all** — zero occurrences in `cpu_quant_dot.cpp` and `gguf_reader.cpp` on
  `origin/main`, re-measured 2026-08-30 — so `UD-IQ1_M` still refuses at file
  open and that half stands. Original text: `IQ2_XS` (id 17) is in the same
  state and `IQ1_M` (id 29) has no reader traits at all.** `UD-Q2_K_XL` would expand 148 towers from 128.344 GiB to
  888.000 GiB; `UD-IQ1_M` refuses at file open. Discharged by a
  `VecDotIQ2_XSQ8_K` and an `IQ1_M` port, or by this row permanently recording
  those two arms as unreachable. Nothing here needs either; they are named so a
  later reader does not rediscover them as defects.
- **O3b — `94de63ff5` left `IQ2_XS` and `IQ4_XS` decodable but not keep-quant,
  for every row, not only this one.** That is a silent 3.4x memory multiplier on
  any artifact carrying them, invisible to a token gate, and it is not this row's
  record to repair. Named here because a reader who checks `gguf_dequant.cpp` and
  stops will conclude both types are supported.
- **O4 — the indexer KV side cache does not exist**, so sparse decode refuses any
  resumed request — the refusing `VT_CHECK(!elig.prunes || elig.Active(), ...)`,
  at `dots3_note_device.cpp:1204-1227` on `origin/main` `03e0dcd19` and
  `:1205-1228` on this row's integration branch, which is W4's own three-line
  edit moving it. The anchor read `:1147-1180` until 2026-08-30, see O20. Discharged by `KV-DSV4-MULTICACHE`
  W5 ([#1925](https://github.com/mudler/vllm.cpp/issues/1925),
  [#2323](https://github.com/mudler/vllm.cpp/issues/2323)) and consumed here,
  NOT by a wave of this row — the §3.7 W5 conditional resolved FALSE on
  2026-08-30 and the decision is recorded in both specs. The refusal is not
  discharged by DELETION under any owner: it guards `Dots3NoteForCausalLM`
  (`MODEL-DOTS3-NOTE`, #699) and its replacement is a gated dispatch.
- **O5 — MTP is skipped, not implemented.** `num_nextn_predict_layers: 1` and
  `index_share_for_mtp_iteration: true` are dropped through `allow_mtp_tail`.
  There is no MTP drafter in the tree (`src/vllm/v1/spec_decode/` holds three
  files, none of them one). Discharged by a drafter row that does not exist yet.
- **O6 — sparse prefill does not exist** (§3.5.2 item 3). Discharged by W6.
- **O7 — PARTIALLY DISCHARGED by W7, 2026-08-30; the half that remains is the
  measurement rather than the staging.** The `UD-IQ1_S` arm is being staged to
  `/mnt/nas_share/rc/ckpt/GLM-5.3-UD-IQ1_S/` (which IS `/workspace` on a leased
  worker), sha256-verified per shard by the fetch script as each one lands.
  Recorded so far, at revision `346b3591c7f28d1a23716f97a065ecf12ec14771`:
  shard 1 `ff3adab0853dfb00bdf3889ec3f5556196f56b65783115720d57767bbd760dd9`,
  shard 2 `659d04cf4fc0b6026944f34c0b590a635803bff06c1775361e28490db7b168f8`.
  Shards 3-6 were still downloading when W7 ended (43% of 201.83 GiB staged) and
  their hashes are owed to the wave that drives the load.
  **THE STAGING HALF IS DISCHARGED, 2026-08-31.** All six shards are staged and
  all six sha256 are recorded, and they were re-verified INDEPENDENTLY of the
  fetch script — read a second time off the same CIFS mount from the devbox — and
  all six are identical to the values recorded as each shard landed:
  shard 3 `433302bac0e2d54da64c7c2f28509fa1b235aeccdf5b215a8a446ebaad1b5b27`,
  shard 4 `d0a6f19452d5b5cd498e1eb8fbe856e00aed7da1f80c27c095301eabe81e9bc1`,
  shard 5 `2ea1537ffab40fa8b8584a8647ec10fbaa6199dfed45e4019b822da2b319db37`,
  shard 6 `42a76ef04ffc5e321e1240f4e572b6fa6fc3315da5bea22fb598d7460db210fe`,
  beside shards 1 and 2 above; the six total 216,715,365,893 B, the published
  count exactly. **AND THE FILESYSTEM IS NAMED: CIFS.** `/workspace` is
  `//192.168.68.102/Data`, `dgx.casa`'s local disk was NOT used and was not
  re-measured, and §3.7 W7's stop condition therefore binds on every read rate
  this row states. The 866-second load below is a CIFS number and is labelled
  one. **NO `pread` NUMBER EXISTS AND NONE IS CLAIMED** (`expert-streaming.md` `## Owed`, verbatim: "The
  `pread` path has never run on the model"). `/mnt/nas_share` is CIFS, and §3.7
  W7's stop condition binds: a `pread` served across CIFS is a CIFS number and
  must be labelled one, never reported as a streaming measurement. Whether
  `dgx.casa`'s local disk can hold 201.83 GiB was NOT re-measured — the last
  figure is 184 GiB free from 2026-07-21 (§0.1 C3), which is stale rather than
  current. Discharged by a wave that records the remaining four hashes, drives
  the load, and states which filesystem served the reads.
- **O8 — the expert-streaming mechanism has no shared seam**, so it is reachable
  from exactly one model TU (§3.2 gap 1). **DISCHARGED by W3**, 2026-08-30:
  `include/vllm/model_executor/expert_stream_seam.h` and
  `src/vllm/model_executor/expert_stream_seam.cpp` carry `ExpertStreamLane` (was
  `Qwen35ExpertStream`), `ExpertStreamStepGuard`, `HostSliceView` and
  `ExpertSlice`; `qwen3_5.cpp` is the first client and keeps four alias
  declarations. The resident-tower fallback is a PARAMETER, not a call — see O13.
- **O9 — the resident 14.511 GiB is arithmetic from the shard headers, not a
  measurement.** It excludes KV cache, activations, scratch pools and the CUDA
  context, which is the same omission `expert-streaming.md` `## Owed` already
  records for its own fit bound. **STILL OWED after W7**, which loaded no real
  weight: no `rc` lease was obtained and the artifact was 43% staged when the
  wave ended, so the 14.511 GiB stands as arithmetic. When it is measured, note
  what may and may not establish it — and it is now MEASURED, on `thor:gpu0`,
  2026-08-31: `VmHWM` peaks at 24,216,892 kB = **23.10 GiB** on a load that
  materializes every weight and sizes the engine's caches, against about
  20.3 GiB predicted (§3.3's 14.511 + O27's 4.48 + O31's 1.772 less the 0.5 of
  Q4_K it replaces). The prediction is met to +2.8 GiB, which is the CUDA
  context, the KV pool and the scratch the arithmetic explicitly excludes. The
  DEVICE-POOL half is still owed and could not be taken here: this device's
  driver reports `memory.used` as `[N/A]`. Original note: **peak RSS is NOT a
  residency measurement while the towers are mmap-resident**, because `VmHWM` then tracks page-cache
  pressure — the sibling Flash row read 99.47, 85.18 and 72.05 GiB for the same
  model on different boxes. The number that answers this item is the DEVICE pool
  the resident class occupies, not the process's high-water mark. Discharged by a
  wave reporting that beside this prediction, and naming what established it.
- **O10 — no speed axis has a denominator** (§3.6). vLLM cannot run the model, so
  the only comparable is llama.cpp on the same artifact, a labeled secondary
  floor. Open gap by construction, not a waiver.
- **O11 — `docs/USAGE.md` carries no weights row for this model**, because
  nothing is reachable yet. Owed in the same change that makes the capability
  reachable, i.e. W7: file names, sizes, `unsloth/GLM-5.3-GGUF` at its exact
  revision, per-shard sha256, and the refused arms named beside them.
- **O12 — the `ENG-EXPERT-STREAM` row (`engine-matrix.md:117`) carries `-` in
  both its "Our code" and "Our tests/evidence" columns**, and its row text
  describes "fixed contiguous Marlin slots" when no Marlin code is on that path.
  Not this row's record to fix, and named here because a reader who checks that
  row before this section will conclude the capability does not exist.
- **O13 — this tree has TWO `ResidentWeight` definitions and they are not
  interchangeable.** `qwen3_5.cpp` defines one in its unnamed namespace which
  SHADOWS `dense_attn_block.h:181`'s, and the local one additionally refuses a
  streamed tower, an `elem_kn_repacked` weight and a `repacked` weight at device
  staging, and carries the `MakeHostBytesDeviceAliasable` /
  `StageOwnedWeightsToDevice` host-alias arm. W3 found this while lifting the
  seam: a shared `ExpertSlice` that called `ResidentWeight` itself would bind to
  the header's definition and silently drop all of those guarantees from
  Qwen3.5's streamed lane, with no test in the tree able to see it. W3 works
  around it by injecting the fallback (`ResidentSliceFn`), which is correct and
  is not a repair. Discharged by reconciling the two definitions, which is not
  this row's work and has no owner yet.
- **O14 — the load-time capacity refusal's production call site is not reachable
  from any CPU gate.** `RequireSlotCapacity` is called from
  `LoadedEngine::FromModelDir`'s streamed-lane block, which is guarded by
  `target.needs_weight_staging() && target.host_memory_is_device_addressable()`
  — true on `dgx:gpu0` and false on every CPU. So W3 gates the refusal, its
  arithmetic and its GGUF-header inputs directly
  (`tests/vllm/model_executor/test_expert_stream_capacity.cpp`, 7 cases), and
  the call site itself is proven only by reading it. Discharged by W7, which is
  the first wave that loads through that block under an `rc` lease; the same
  device-only reachability problem `test_expert_stream_device_slot` solves with
  a fake platform, which W3 did not extend because the block under test is the
  loader's and not the seam's. **STILL OWED after W7.** W7 loaded a model through
  `FromModelDir`, but on a CPU and on a synthetic fixture, so the device-gated
  block that calls `RequireSlotCapacity` was not entered: it needs
  `needs_weight_staging() && host_memory_is_device_addressable()`, which is a
  GB10 and not a CI runner, and it additionally needs
  `factory->streams_routed_experts`, which W7 deliberately did NOT set (O22).
  **DISCHARGED 2026-08-31 on `thor:gpu0`, and by the load SUCCEEDING rather than
  by a message.** All five conditions of that block held, so it was entered and
  `RequireSlotCapacity` ran on this file's own geometry — 228 streamed towers x 8
  experts per token = 1824 against the configured 4096 — and returned. The proof
  that it was entered is that the load completed at all: without the lane
  `CheckDeviceWeightFit` charges the device the whole 187.312 GiB of towers and
  refuses (O22). The refusal ARM of `RequireSlotCapacity` is still gated only by
  W3's seven direct cases, which is the right place for it.
- **O15 — nothing measures the lifted seam from a SECOND model.** W3 makes the
  lane reachable by a second TU and Qwen3.5 remains its only client, so what is
  gated is that the mechanism still works, not that another architecture can
  take it. This is deliberate: W3's scope is the lift, and a second client
  without a model to attach it to would be the shell several waves on this row
  stopped rather than ship. **STILL OWED after W7, and this is the most important
  thing W7 did not do.** W7 loads GLM-5.3's 228 towers and holds them in exactly
  the shape the seam consumes — stacked keep-quant, borrowed from the mmap, flat
  `[E*out, K]`, the form `LoadExpertsStackedKq` produces for the seam's only
  gated client — but nothing calls `expert_stream::ExpertSlice` on them, because
  the forward that would does not exist (O21). Qwen3.5 therefore remains the
  seam's only client. Discharged by the wave that writes the forward and routes
  the towers through it.
  **HALF DISCHARGED 2026-08-31, and the half that remains is the one that
  counts.** W9's forward routes through `expert_stream::ExpertSlice`, and the
  2026-08-31 load on `thor:gpu0` proved the LOADER side end to end on the real
  201.83 GiB artifact: the streamed-lane block was entered, the slot arena was
  built from this model's own geometry, and 187.312 GiB of towers stayed out of
  the resident class. **No slice was ever served**, because the forward threw in
  MLA prefill before the first expert, so `ExpertStreamLane` was never
  constructed, no `[expert-stream]` line was printed, and there are no
  `fills`/`hits`/`bytes` counters to quote.
  **DISCHARGED on `dgx:gpu0` the same day, and by counters rather than by an
  argument.** With FlashAttention-2 compiled for `121a` the step reaches the
  experts, `ExpertStreamLane` is constructed —
  `[expert-stream] ON slots=4096 slot_bytes=4816896 resident=18.38 GiB` — and one
  step reports `steps=1 hits=0 misses=527 evictions=0 fills=527
  bytes=1876328448 exhausted=0 advised=0`. GLM-5.3 is therefore the seam's second
  client in the only sense that matters: 527 slices of a second architecture's
  towers were paged out of a real file into slots. `exhausted=0` is the part to
  read twice — no slice fell back to reading the tower in place, so nothing in
  those bytes is a page-cache number wearing a streaming label. It is also the
  first time the `pread` path has run on a real checkpoint at all.
  **FULLY DISCHARGED the same day, by a step that COMPLETED.** With O33's slot
  budget corrected the same command runs to `rc=0` and a token, reporting
  `steps=1 hits=0 misses=6399 evictions=0 fills=4096 bytes=13939408896
  exhausted=2303 advised=0`: 4096 slices served from slots, 12.98 GiB moved. GLM-5.3
  is the seam's second client in the strongest available sense — a second
  architecture's towers, a real 201.83 GiB file, and a token out the other end.
  Read `exhausted=2303` with it; O34 owns that number.
- **O16 — W2 lands three surfaces that NOTHING reaches yet**, and it says so
  rather than letting a reader infer reachability from a green suite. The
  registration, its config hook, the `glm-dsa` `kGgufArchArms` row and
  `GlmMoeDsaHfConfigFromGguf` ARE reached, from `LoadedEngine::FromModelDir`
  through `HfConfigFromGgufDispatch` and `ModelRegistry::Resolve`, and the
  focused suite enters through that entry point. `GlmMoeDsaModel::Forward` /
  `ForwardDevice`, the registry's `forward` hook and `MakeGlmMoeDsaKVCache` are
  NOT: every arm of `load_weights` refuses, so no `LoadedModel` of this type can
  exist and neither the forward nor the KV-cache hook can be called through
  production. Owned by this row. **HALF DISCHARGED by W7, 2026-08-30**: the GGUF
  arm of `load_weights` loads, so a `GlmMoeDsaLoadedModel` DOES exist and
  `MakeGlmMoeDsaKVCache` is reached through it —
  `tests/vllm/models/test_glm_moe_dsa_gguf_load.cpp` builds one through
  `LoadedEngine::FromModelDir`, and three mutations (the `kGgufArchArms` row, the
  `REGISTER_VLLM_MODEL` line, the GGUF branch of `LoadGlmMoeDsaForCausalLM`) each
  red it. The FORWARD hook is still not reached, because `GlmMoeDsaModel::Forward`
  still refuses — that half is O21. Tracked by
  [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
- **O17 — the one staged GGUF arm cannot be fed, because it states no indexer
  schedule.** D3 records that the published conversion writes no
  `glm-dsa.attention.indexer.types`, and W2 refuses such a file by name rather
  than substituting llama.cpp's hardcoded table. So the `glm-dsa` arm is
  implemented and the published artifact still does not load.
  **W7 CONFIRMED THE DIAGNOSIS ON THE REAL FILE AND SHIPPED THE REPAIR,
  2026-08-30, and the repair is to the FILE and not to the loader.** Shard 1's KV
  block was read key by key at revision
  `346b3591c7f28d1a23716f97a065ecf12ec14771`: **64 keys, and none of
  `glm-dsa.attention.indexer.types`, `index_topk_freq` or
  `index_skip_topk_offset` is among them.** The file also declares `indexer.*` on
  all 79 blocks, so its own tensor list cannot be read as the schedule either —
  D3's broadcast, confirmed by census.
  `scripts/glm-dsa-write-indexer-types.py` writes the key llama.cpp's own
  converter would have written (`b10451:conversion/glm.py:337-339`),
  TRANSCRIBING it from the model author's `config.json` and deriving nothing. It
  refuses a config that states no `indexer_types`, refuses a shard that carries
  tensor payload (growing that header would move every tensor's data without
  moving its recorded offset), refuses a file that already states its schedule,
  and reads its own output back before reporting success. It rewrites only the
  metadata shard, which on this artifact is shard 1: 9,428,677 bytes, zero
  tensors, header ending exactly at end-of-file. A dry run over the real shard
  produces a 9,428,810-byte file, 64 keys becoming 65, sha256
  `b3e9838651a5c279533c98390ab4bc03cf1d8c176d5be0754180f07d9ed85c01`.
  **CONFIRMED ON THE REAL BYTES BY W9, 2026-08-30.** The dry run above was a dry
  run; W9 ran the script against the STAGED shard 1 with the committed
  `config.json` and wrote the output: 9,428,677 -> 9,428,810 bytes, 64 keys ->
  65, `21 full of 78`, full layers `{0,1,2} u {6,10,...,74}`, read back and
  re-parsed by the script itself, sha256
  `b3e9838651a5c279533c98390ab4bc03cf1d8c176d5be0754180f07d9ed85c01` — bit for
  bit the dry run's predicted hash. Both hashes are now recorded side by side in
  `docs/USAGE.md`. What remains owed is only the LOAD: the repaired shard has
  never been fed to the loader, because shards 4-6 are not staged.
  **FULLY DISCHARGED 2026-08-31.** The script has now been run against the
  COMPLETE six-shard artifact and the derived file has been FED to the loader,
  which is the last thing this item asked for. Three independent runs — W7's dry
  run, the devbox, and inside the lease — produce the same 9,428,810 bytes, the
  same 64 keys becoming 65, the same `21 full of 78` at layers
  `{0,1,2} u {6,10,...,74}`, and the same sha256
  `b3e9838651a5c279533c98390ab4bc03cf1d8c176d5be0754180f07d9ed85c01`. The five
  payload shards are HARD-LINKED beside it rather than copied, which this CIFS
  share supports and which is what makes a 9.4 MB rewrite of a 201.83 GiB
  artifact cheap. Fed as
  `--model <derived>/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf`, the loader accepts
  the schedule and materializes the model.
  **The output is a DERIVED artifact and must never be quoted as the published
  one.** The loader's refusal is NOT weakened, which is D3's whole point: a
  hardcoded 78-entry constant that happens to be right is the shape that silently
  becomes wrong on GLM-5.4. Fully discharged when a wave runs the script against
  the complete artifact and records the derived shard's hash beside the published
  ones.
- **O18 — the fp32 router GEMM's dtype SELECTION has no discriminating
  numerical gate, and this is measured rather than suspected.** W4 makes
  `MoeBlock` size `dlog` from `DeepseekV2Params::router_dtype_is_f32`
  (`deepseek_v2.cpp:363`), and the line is REACHED — `MoeBlock` is on
  `DeepseekV2Model::Forward`, which `test_deepseek_v2_forward` drives on the CPU.
  What no test can see is the dtype itself. Over the tiny fixture's 500 output
  logits the f32 arm and the bf16 arm are BIT-IDENTICAL (`differing = 0/500`,
  `maxabs = 0`): the router logits feed an f32 softmax and top-k, and every
  activation downstream of the combine is stored at bf16, so the ~4e-3 relative
  rounding the wider store removes is re-introduced two ops later. Forcing
  `DType::kF32` unconditionally at `:363` also leaves every case in that file
  green, which was measured as mutation M7. So what W4 gates is the PARSE
  (against the pinned oracle's own return values on eight configs), that an f32
  and a bf16 store of one GEMM genuinely differ (13 of 24 exact-integer
  products), and that the f32 arm RUNS end to end through `vt::MoeRouterTopK`;
  the selection at `:363` is proven by reading. Discharged by a fixture whose
  routing is precision-sensitive enough to separate the two arms, or by an
  end-to-end gate against a checkpoint that declares `moe_router_dtype:
  "float32"` — neither of which this row needs, because GLM-5.3's own forward is
  W7's. Named so a reader who sees a green suite does not conclude the dtype is
  gated. This is the `## Gates` hazard AGENTS.md states — "a token gate cannot
  detect a dtype that is too wide" — landing as a concrete instance.
- **O19 — the `skip_topk` reuse arm of `ForwardMlaAttentionBlock` has no
  PRODUCTION caller yet.** W4 lands the semantics, the refusals and the shared
  buffer, and the only thing that sets `dims.skip_topk` on a real forward is a
  GLM-5.3 decoder layer, which does not exist: `GlmMoeDsaModel::Forward` still
  refuses by name (O16). What IS reached is the SCHEDULE — `ParseGlmMoeDsaConfig`
  runs `GlmMoeDsaMlaSchedule` for every `GlmMoeDsaForCausalLM` config resolved
  through `ModelRegistry::Resolve`, and deleting that call reds the focused gate.
  The reuse arm itself is a staged slice under `## Nothing lands dead`, owned by
  W7, tracked by [#2214](https://github.com/mudler/vllm.cpp/issues/2214), and
  discharged when W7's forward drives the 78 layers in order over one
  `MlaSharedSelection`.
- **O20 — this section's `dots3_note_device.cpp:1147-1180` anchor was WRONG,
  and it had already propagated into product code.** The refusing `VT_CHECK` is
  at `:1204-1227` on `origin/main` `03e0dcd19` (`:1205-1228` on this row's
  integration branch, one line down under W4's edit to the same file);
  `:1147-1180` is the explanatory comment block above it, so the anchor pointed
  at prose rather than at the guard it claimed to cite. **Cite it by its
  predicate — `!elig.prunes || elig.Active()` — rather than by a number**, which
  is the durable form: the range moved by one line inside this very branch. The
  same wrong range is compiled into `kForwardRefusal`
  (`glm_moe_dsa.cpp`), which means a GLM-5.3 user meeting the refuse-by-name
  forward is handed a line range that does not contain a refusal. Corrected in
  O4 and §3.7 here; the product-code string is W2's surface and is left to the
  wave that next edits that file, named rather than silently repaired, because
  editing a refusal message is a behaviour change to a registered model and not
  this record's to make. Verified at parity pin `5559679229`.
  **DISPOSITION, decided 2026-08-30: repaired IN FLOW by the next wave that
  touches `glm_moe_dsa.cpp`, referencing O20 — not by a row of its own.** It is
  a wrong `file:line` inside a user-visible refusal string, so a user who hits
  the refusal is sent to a comment rather than to the check; worth fixing, and
  not worth its own branch, gate run and fresh review. Discharged when
  `kForwardRefusal` cites the predicate `!elig.prunes || elig.Active()` rather
  than a line range — cite the predicate, not the line.
  **DISCHARGED by W7, 2026-08-30.** `kForwardRefusal` now cites the predicate
  `!elig.prunes || elig.Active()` — verified unique repo-wide over `src`,
  `include` and `tests`, one hit at `dots3_note_device.cpp:1205` — and no line
  range. `git grep 1147-1180` over `src` and `include` returns only the comment
  in `glm_moe_dsa.cpp` that records the repair.
  **The same string had gone stale in four further places and W7 fixed those
  too**, because a refusal that keeps naming finished work sends its reader
  looking for something they will not find: it named the expert-streaming seam
  (landed, W3), the per-layer dims and `skip_topk` reuse (landed, W4), the
  `IQ4_XS` keep-quant `vec_dot` (landed, `2e9f4d88d`) and the fp32 router GEMM
  (landed) as missing. `test_glm_moe_dsa_config.cpp` now gates BOTH halves: the
  two primitives that are genuinely missing, and that the four discharged ones
  are no longer named.
  **The other three anchors in §3.7 W5 verified CLEAN at the same pin**, and
  are recorded so they are not re-checked: `DeepseekV32IndexerCache`
  (`deepseek_v2.py:696-701`) is exact; the **132 B/token** figure is exact and
  DERIVED rather than quoted — `head_dim + head_dim // quant_block_size * 4` is
  `128 + 128//128*4 = 132` at `dtype=torch.uint8` (`:697-698`); the
  `MLAAttentionSpec` merge rule (`kv_cache_interface.py:399-429`) is imprecise
  rather than wrong — the class opens at `:381` and `merge` at `:419`, so the
  cited range's first half is `real_page_size_bytes`. That property carries **no
  factor 2**, which is the sibling Flash row's hard-won lesson: an MLA latent is
  ONE vector per token, not a K+V pair, and reading it pair-strided yields
  finite, correctly-shaped, WRONG numbers.
- **O16 and O17 were authored by W2 as O13 and O14, and are renumbered here.**
  W2 and W3 were developed on parallel branches and each appended two owed
  items to this list, so both claimed O13 and O14. W3's three items landed on
  this row's integration branch first and keep their numbers; W2's two move to
  O16 and O17, and the `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` row
  in `.agents/model-matrix.md` cites the new numbers. Nothing else changed in
  either item.

- **O21 — THE FORWARD DOES NOT EXIST, SO NO TOKEN HAS BEEN OBSERVED, and W7 did
  not produce one.** This is the plainest thing this record has to say. W7's
  scope paragraph is the loader and W7 delivered the loader; W7's TEST list
  additionally says "the model loads and produces a first token", and the second
  half did not happen. `GlmMoeDsaModel::Forward` still refuses by name.
  What the forward needs, read out of the tree rather than guessed: a per-layer
  `RunLayer` that passes `GlmMoeDsaMlaSchedule`'s 78 dims instead of one scalar
  `p.mla`, and threads an `mla::MlaSharedSelection` into
  `ForwardMlaAttentionBlock`'s eleventh argument — which exists and defaults to
  `nullptr`, and which **no production model passes today**; a sparse-step
  builder that populates `meta.indexer_cu_seqlens_q`, which `BuildMlaStep` never
  sets and only `BuildDots3NoteSparseStep` does; and a MoE block that slices the
  stacked towers through `expert_stream::ExpertSlice` rather than DeepSeek-V2's
  per-expert vectors. **A FIRST token is reachable and a SECOND is not**: a
  sparse step routes every token through MQA (`mla_attention.cpp:448-453`), which
  a fresh prompt can do and a resumed request cannot until O4 lands. Owned by
  this row.
  **Scheduled as §3.7 W9, written before its code as `## Spec before code`
  requires. W9 verified both findings this item states and measured three more;
  see W9's F1..F5.**
- **O22 — `ModelFactory::streams_routed_experts` is deliberately NOT set on
  `kGlmMoeDsaFactory`, and the load on `dgx:gpu0` cannot succeed until it is.**
  The flag asserts that THIS model's forward reads experts through the slot seam,
  and `model_registry.h:552-589` argues at length that it is a property of the
  forward and lives beside it. Scheduled as part of §3.7 W9, which is
  the change that lands the forward. W7 has no forward, so setting it would be a claim
  about code that is not there. The cost is exact, and stating it is the point of
  this item: without the flag `model_loader.cpp:2487-2494` never builds the lane,
  so `CheckDeviceWeightFit` charges the device the full 187.312 GiB of towers
  against a 119.631 GiB budget and REFUSES the load. Discharged in the same
  change that lands the forward, and not before.
- **O23 — `docs/USAGE.md`'s GLM-5.3 row states a load that has not been driven.**
  O11 asked for the weights row in the change that makes the capability
  reachable. W7 makes the LOADER reachable and adds the row, and the row says
  exactly that: the published arm is unfeedable as shipped (O17), no load of the
  real artifact has run, and no token exists. **DISCHARGED 2026-08-31**: the row
  now states a DRIVEN load — `thor:gpu0`, 866 s, `VmHWM` 23.10 GiB, the derived
  shard's hash beside the six published ones, and the repair step named so a user
  meets it as an instruction rather than as a mystery. It still says that no
  token exists, because none does.

- **O24 — the routed-expert MoE runs the PER-EXPERT reference loop, and the
  grouped keep-quant arm is deliberately not taken.** `vt::MoeGateUpSwiGLUGrouped`
  consumes the WHOLE stacked tower, so it bypasses the slot lane entirely —
  the same conflict `qwen3_5.cpp` resolves by turning grouping OFF whenever
  streaming is on, and saying so on stderr. This row has no speed axis (O10) and
  a grouped MoE that silently un-streams a 187.312 GiB expert set is the
  invisible-fallback shape this campaign keeps finding, so W9 takes the loop.
  Discharged by a row that wants the lever and can gate it, which is
  `ENG-EXPERT-STREAM` W6 / `ENG-EXPERT-STREAM-DEVICE` W2 territory rather than
  this row's.
- **O25 — W9 adds a THIRD host-slice helper beside the two `ResidentWeight`
  definitions O13 names.** `GlmResidentExpertSlice` (`glm_moe_dsa_forward.cpp`)
  is the resident fallback `expert_stream::ExpertSlice` takes when streaming was
  never requested or the device is discrete. It is NOT a third `ResidentWeight`:
  it aliases the borrowed mmap at a row offset and REFUSES a discrete device by
  name, where both `ResidentWeight`s would stage. But it is a third place in
  this tree that decides how a weight becomes a device tensor, which is the
  count O13 was already unhappy about. Discharged with O13, which still has no
  owner.
- **O26 — the router GEMM is deliberately WIDER than vLLM, on one op, and
  nothing gates the difference.** Upstream's `GateLinear` holds the gate at the
  model dtype and takes a bf16 x bf16 -> f32 tier
  (`fused_moe/router/gate_linear.py`); the published GGUF stores
  `ffn_gate_inp` at F32, and W9's forward keeps it there and widens the
  ACTIVATION to match, so the GEMM is f32 x f32 -> f32. Three reasons are
  recorded at the call site: it is the smallest GEMM in the model, vLLM's own
  `force_fp32_compute` arm stores this exact weight in fp32 when no specialized
  kernel is available, and the output feeds a DISCRETE top-k where narrowing the
  artifact's own f32 would be us discarding precision the file carries. O18
  already records that no numerical gate on a tiny fixture can SEE a router
  dtype; this is the same blindness pointed the other way, and it is named
  rather than left for a reader to find. Discharged by a fixture whose routing
  is precision-sensitive enough to separate the two arms.
- **O27 — the absorbed MLA trio costs 4.48 GiB that §3.3's residency plan does
  not contain, and O9's prediction is restated rather than met.** Per layer:
  `kv_b_proj` 29.4 MB + `w_uk_t` 12.6 MB + `w_uv` 16.8 MB = 58.8 MB, over 78
  layers. So the resident class this port actually holds is ~18.99 GiB against
  the 14.511 GiB §3.3 computed from the shard headers, and the §3.10 arithmetic
  that put resident + a 4096-slot cache at 40.01 GiB becomes ~44.5 GiB against
  119.631 GiB. It is not avoidable at this pin: `vt::BatchedMatmul` shares
  "f32 or bf16" across its operands and there is no quantized bmm, which is the
  same constraint `mla_attention.py:876-878` records upstream. Discharged by a
  quantized batched GEMM, or by W7's measured-footprint number landing against
  this figure instead of against 14.511 GiB.

- **O28 — the post-load absorption's ORIENTATION has no gate, and this is
  measured rather than suspected.** W9's mutation M5 dropped the per-head
  transpose entirely — reading `attn_k_b` verbatim into `kv_b_proj`'s nope rows
  — and every shape, every byte count, all 7 forward cases and all 4 load cases
  stayed GREEN. Only the logit VALUES moved (the fixture's top-2 margin went
  0.00482 -> 0.00068), and nothing on this row compares a value to anything,
  because there is no oracle (O1). The orientation is therefore established by
  READING two conventions against each other: llama.cpp's `ggml_mul_mat`
  contracts over `ne[0]`, so `attn_k_b`'s rows run along `qk_nope_head_dim`,
  while `vt::BatchedMatmul` is `torch.bmm`, so `w_uk_t`'s rows run along
  `kv_lora_rank`. The dimensions cannot disambiguate it either: the wrong index
  order stays in bounds on both the real checkpoint and the fixture.
  `test_glm_moe_dsa_gguf_load.cpp` now carries a DRIFT LOCK that fires on M5,
  and its own comment says what it is — a transcription of the rule the loader
  implements, which catches a later edit that changes one without the other and
  proves nothing about which rule is right. Discharged by G4, llama.cpp `b10451`
  on the identical artifact (§3.6, W8), which needs the complete checkpoint.
- **O29 — `streams_routed_experts` is set and NOTHING on a CPU can see it, and
  W9 measured that rather than reading it.** Mutation M4 deleted the flag from
  `kGlmMoeDsaFactory` and all 7 forward cases stayed green, because the only
  reader is `model_loader.cpp`'s streamed-lane block, which is guarded on
  `needs_weight_staging() && host_memory_is_device_addressable()` — true on
  `dgx:gpu0` and false on every CI runner. This is O14's statement for O22's
  flag, and it converts both from "proven by reading" into a measured negative.
  **DISCHARGED 2026-08-31 on `thor:gpu0`**, with O14 and by the same evidence: the
  flag's only production reader is that block, the block was entered, and a load
  that entered it is a load `CheckDeviceWeightFit` did not refuse. What is still
  NOT measured is the flag's CLAIM — that the forward reads experts through the
  slot seam — because no step ran; that is the counters O15 owes.

- **O30 — THE LOADER PREFAULTED THE 187.312 GiB IT EXISTS NOT TO READ, and the
  first drive of the real artifact is what found it.** `OwnGgufQuantBlocks` and
  `OwnGgufF16` end their BORROW arm with `PrefaultBorrowedSpan(src, bytes)`,
  which `madvise(MADV_WILLNEED)`s the span and then touches one byte per page —
  L7's repair for a page trap landing in the timed prefill
  (`gguf_keep_quant.cpp:267`). For a weight the forward reads in place that is
  right. For a routed-expert tower it is not a latency trade at all: the slot
  lane `pread`s each slice into its own slot and never reads the tower through
  the mapping, so the prefault reads the whole expert set off disk to populate
  pages nothing will look at.
  **Measured, not inferred.** `thor:gpu0`, 2026-08-31, `--device cuda`,
  `VT_MOE_EXPERT_STREAM=1`, `VT_MOE_EXPERT_STREAM_SLOTS=4096`, the derived
  artifact: the load's `VmHWM` rose LINEARLY through 6.16, 12.05, 17.54, 21.45,
  26.75, 32.99, 38.77, 44.49 and 48.62 GiB at 158-second intervals — ~35 MB/s,
  the CIFS read rate — with no plateau anywhere near the 18.99 GiB resident class
  O27 predicts. The run was killed at 23 minutes rather than left to read all
  201.83 GiB, because what it was measuring was the page-cache path under a
  streaming label, which is exactly what §3.3 refuses to publish.
  **Fixed in the same flow** ([#2214](https://github.com/mudler/vllm.cpp/issues/2214)):
  both functions take a `prefault` parameter defaulted to `true`, so every
  existing caller is byte-identical, and `LoadStackedExperts`
  (`glm_moe_dsa_loader.cpp`) passes `false`. Gated by
  `test_glm_moe_dsa_gguf_load.cpp`'s "the streamed towers are borrowed and NOT
  prefaulted", which asserts the towers took the BORROW arm (the positive
  control, `mmap_fd >= 0`) and that the prefaulted-span count is zero — sound in
  this fixture because every non-tower tensor in it is F32 and therefore
  dequantized into an owned buffer, so the Q8_0 towers are the only spans that
  reach the borrow arm at all.
  **What is NOT discharged: the same defect on the Qwen3.5 lane.**
  `qwen3_5_gguf_weights.cpp`'s own `LoadExpertsStackedKq` still prefaults the
  towers it hands to the same seam. It is invisible there because no Qwen3.5
  artifact is large enough for anyone to notice, and it is `ENG-EXPERT-STREAM`'s
  to generalise rather than this row's to reach into another model's loader.
  Named here so a reader does not conclude the lane is clean.

- **O31 — `token_embd.weight` WAS ASKED THE WRONG ROLE, AND ONLY A DEVICE CAN
  SEE IT.** The GGUF loader routed the vocabulary table through `LoadMatmul`,
  which asks `GgufTensorRole::kMatmulWeight`. A GEMM weight's device gate is
  `DeviceKeepQuantSupported`, TRUE on CUDA because the CUDA backend falls back to
  the CPU kernel for anything it lacks; a GATHER's gate is
  `DeviceQuantGatherSupported`, true ONLY on the CPU, because
  `EmbeddingKernelCuda` accepts f32 and bf16 tables and nothing else and has no
  fallback tier. So on a device queue this checkpoint's `[154880, 6144]` Q4_K
  table stayed Q4_K and the FIRST forward threw, with all 201.83 GiB of the model
  already resident and the engine's caches already sized:
  `vt: cuda embedding: unsupported table dtype (f32/bf16 only) at
  src/vt/cuda/cuda_ops.cu:861`. Measured on `thor:gpu0`, 2026-08-31, at 892 s of
  wall clock into the run.
  **Fixed in the same flow** by asking the role the tensor actually has:
  `LoadEmbeddingTable` routes `kEmbeddingTable`, and the shared policy then
  expands to bf16 on a device queue and keeps the blocks on a CPU one. The tied
  case takes the intersection of both roles, because a gather needs only a row
  decoder while a GEMM needs a `vec_dot`. The cost is exact and stated rather than
  discovered: 1.772 GiB of bf16 where 0.5 GiB of Q4_K stood, on top of O27's
  ~18.99 GiB.
  **NO CPU GATE CAN SEE THIS, and that is measured rather than assumed.** On a CPU
  runner `DeviceQuantGatherSupported` is true, so the gather role and the GEMM
  role reach the SAME residency for every encoding in this tree — the decoder set
  and the `vec_dot` set differ only on `Q8_K`, which is an activation encoding
  and never a file weight. There is therefore no fixture that separates the two
  arms without a fake non-CPU platform, which is a second test binary and a
  second registered platform. This is O14's and O29's statement for a third
  surface. What DOES gate it is the row's own declared gate, the load on a leased
  device, before and after, on the same artifact and the same command.
  Discharged by a fake-platform loader suite, which belongs with O14's, or by the
  CUDA quantized gather `gguf_keep_quant.cpp` already records as owed.

- **O32 — `test_gguf_keep_quant`'s gather case ASSERTS THE CPU'S ANSWER AND IS
  RED ON ANY CUDA BUILD RUNNING ON A GPU.** Found while running it as the
  regression control for O30's `prefault` default:
  `tests/vllm/test_gguf_keep_quant.cpp:589`, `:608` and `:625` assert
  `RouteGgufTensor(..., kEmbeddingTable, ...) == kKeepQuant`, and that is true
  only where `DeviceQuantGatherSupported` is true, which is the CPU alone. On
  `thor:gpu0`, built `-DVLLM_CPP_CUDA=ON` and run on the device, the suite reads
  43 cases / 42 passed / 1 failed and 6480 assertions / 9 failed, every failure in
  that one case. It is **base-caused**: nothing on this row touches
  `RouteGgufTensor`, and the same binary's other seven suites are green.
  It is also the same fact O31 is about, pointed at a test instead of at a
  loader — which is why it is worth naming rather than routing around. Not
  repaired here: making the expectation device-conditional is a semantic change
  to another row's gate and wants that row's spec and a fresh review. Owned by
  `ENG-RESIDENCY-CONFIG` / `QUANT-GGUF-*`, whichever claims
  `tests/vllm/test_gguf_keep_quant.cpp`.

- **O33 — THE SLOT STORE WAS SIZED BY THE FIRST LAYER AND THE ARENA BOUND BY THE
  WHOLE FILE, SO THE TWO WERE DIFFERENT NUMBERS.** `ExpertStreamLane` is a
  process-lifetime singleton built on the FIRST slice anyone asks for
  (`expert_stream_seam.cpp`, `Get` -> `std::max(slot_bytes, Reserved())`), and
  W9's forward reserves the maximum of the THREE slices of the layer it is
  running (`glm_moe_dsa_forward.cpp::ExpertMlp`). That is the right maximum for
  one layer and it is not the model's: a `UD-*` arm mixes encodings ACROSS layers
  by design. On `UD-IQ1_S` the first MoE block is `blk.3`, whose `ffn_down_exps`
  is IQ3_XXS at **4,816,896 B**, while `blk.{8,75,76,77}`'s are IQ4_XS at
  **6,684,672 B** — §3.3 computed exactly that and said slots are sized to the
  LARGEST slice.
  **Measured on `dgx:gpu0` (GB10), 2026-08-31.** The lane came up
  `[expert-stream] ON slots=4096 slot_bytes=4816896 resident=18.38 GiB`, streamed
  **527 slices / 1,876,328,448 bytes with 0 evictions and 0 exhaustions**, and
  then refused by name at `blk.8`: `expert stream: a slice of 6684672 bytes
  exceeds the slot budget of 4816896`. The refusal is correct and its message is
  actionable; what was wrong is the budget it was given. Note the second symptom,
  which is the one worth keeping: `CheckDeviceWeightFit` had already charged the
  device `4096 * 6,684,672 = 25.5 GiB` while the store allocated
  `4096 * 4,816,896 = 18.38 GiB`. The bound and the store were pricing different
  arenas.
  **Fixed in the same flow** by reserving, at load, the same number the bound
  charges: `model_loader.cpp`'s streamed-lane block already computes
  `GgufLargestExpertSliceBytes` over the whole file and now passes it to
  `ExpertStreamLane::Reserve` before `CheckDeviceWeightFit` uses it, so the two
  cannot disagree. It is in the loader rather than in a model because that is the
  one place with the whole FILE; `Reserve` takes a maximum and is inert unless
  streaming was requested, so Qwen3.5's own reservation is unaffected except by
  gaining a floor that is at least correct. W9's per-layer `Reserve` stays, and is
  what the CPU suites (which never enter the device-gated loader block) still
  rely on.
  **Not CPU-gateable, for the third time on this row.** The call site is inside
  the block O14 records as device-only, and the synthetic fixture's towers are all
  Q8_0, so every layer's maximum is the model's and no CPU fixture can separate
  them. What gates it is the device run before and after, on the same artifact and
  the same command.
  **DISCHARGED 2026-08-31, by that before-and-after.** The next run on `dgx:gpu0`,
  same artifact and same command, came up
  `[expert-stream] ON slots=4096 slot_bytes=6684672 resident=25.50 GiB` — the
  IQ4_XS slice and §3.3's arena to the byte, where the run before it read
  `slot_bytes=4816896` and `18.38 GiB` — passed `blk.8` instead of refusing at it,
  and finished the step at `rc=0` with a token. The bound and the store now price
  the same arena.

- **O34 — A PREFILL STEP EXHAUSTS ANY SLOT BUDGET, AND 36% OF THE FIRST TOKEN'S
  SLICES WERE READ THROUGH THE MAPPING RATHER THAN STREAMED.** Measured on
  `dgx:gpu0`, 2026-08-31, at `VT_MOE_EXPERT_STREAM_SLOTS=4096`:
  `misses=6399 fills=4096 exhausted=2303`. The step needed 6399 distinct slices
  and the cache holds 4096, so 2303 `Acquire` calls returned -1 and
  `expert_stream::ExpertSlice` fell through to a `HostSliceView` on the tower's
  own bytes.
  **This is §3.8 R2 arriving as a number, not a defect.** `RequireSlotCapacity`
  bounds a `c = 1` DECODE step at `228 towers x 8 experts = 1824`; a PREFILL of
  `T` tokens touches up to `228 x min(256, 8T)`, and at `T = 5` that measured
  6399. `expert_stream_seam.cpp`'s `ExpertSlice` says so in as many words and
  chooses the in-place read deliberately on a host-addressable device, because
  the alternative — staging the whole tower — is what the load-time refusal
  exists to prevent. The fallback is COUNTED, which is the property that makes
  this reportable at all rather than a silent page-cache number.
  **What it costs the row's claim, exactly:** 64% of the first token's slices were
  served from streamed slots and 36% were not, so no figure here may be quoted as
  a fully-streamed step. **A confirmation at 8192 slots is queued** — 6399 needed
  against 8192 held, a `8192 x 6,684,672 = 51.0 GiB` arena plus ~20.3 GiB resident
  against 119.631 GiB — and it is the run that can state `exhausted=0`.
  Discharged either by that run, or by a DECODE-only measurement, where the
  working set is 1824 and the existing budget covers it. Not discharged by raising
  the default: an arena is device memory the operator did not ask for, which is
  the argument `RequireSlotCapacity` already makes for refusing rather than
  clamping.

### 3.10 Now

**GLM-5.3 EMITS ` Paris` FROM ITS REAL 201.83 GiB ARTIFACT ON GB10, THROUGH THE
EXPERT-STREAMING LANE, 2026-08-31**
([#2214](https://github.com/mudler/vllm.cpp/issues/2214)). The row's question is
answered, and the answer carries one number that must travel with it.

**The run.** `dgx:gpu0` — `NVIDIA GB10`, 20 cores, 119 GB, compute capability
12.1, driver 580.173.02 — under an `rc` lease, base `00940ad13`, `vllm-cli`
built `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
-DVLLM_CPP_FLASH_ATTN=ON` with CUTLASS 4.5.0, sha256
`ce80fd3edd756adf0ec17a0e5791d98c061a1e72b78b87b98ebdec18b9581900`, run
`--device cuda --prompt "The capital of France is" --max-tokens 1
--temperature 0` with `VT_MOE_EXPERT_STREAM=1` and
`VT_MOE_EXPERT_STREAM_SLOTS=4096`, against the derived metadata shard beside the
five published payload shards.

```text
### LOAD_RC=0   wall=1154s
--- stdout (the emitted text, verbatim) ---
 Paris
vllm-cli: run=1/1 finish_reason=length prompt_tokens=5 completion_tokens=1 secs=852.330
```

`stdout` is seven bytes, `0x20 P a r i s 0x0a`. `VmHWM` peaks at 60,512,268 kB =
**57.71 GiB**, against 119.631 GiB on the device and 201.83 GiB of artifact.

**The streaming evidence, which is the half `VmHWM` cannot supply.**

```text
[expert-stream] ON slots=4096 slot_bytes=6684672 resident=25.50 GiB
[expert-stream] steps=1 hits=0 misses=6399 evictions=0 fills=4096
                bytes=13939408896 exhausted=2303 advised=0
```

`slot_bytes=6684672` is the IQ4_XS `ffn_down_exps` slice §3.3 predicted, and
`4096 * 6,684,672 = 25.50 GiB` is §3.3's arena to the byte. **4096 slices were
paged out of the file into slots and 13,939,408,896 bytes = 12.98 GiB moved
through them, with zero evictions.** The 187.312 GiB of towers were never
materialized.

**`exhausted=2303`, and it is not a defect — it is §3.8 R2 arriving as a
number.** The step needed **6399** distinct slices and the budget holds 4096, so
2303 were refused by the cache and read IN PLACE out of the mapping. That is the
documented behaviour of a host-addressable device
(`expert_stream_seam.cpp`'s `ExpertSlice`: a slot the cache cannot serve becomes
a `HostSliceView` on the tower's own bytes rather than a staged tower), it is
COUNTED rather than silent, and the arithmetic is R2's: `RequireSlotCapacity`
bounds a `c = 1` DECODE step at `228 towers x 8 experts = 1824`, while a PREFILL
of `T` tokens can touch `228 x min(256, 8T)`, here `228 x 28 = 6399`. So
**64% of this step's slices were served from streamed slots and 36% were read
through the mapping**, and no number on this row may be quoted as if it were
100%. O34 records it and the 8192-slot confirmation.

**No speed number is claimed** (O10): `secs=852.330` is one token, on a
CIFS-backed artifact, with 2303 in-place fallbacks in it.

**Three defects were fixed to get here, all ours, each measured before it was
repaired.** O30, the loader prefaulting all 228 towers, caught because `VmHWM`
climbed linearly past 48.62 GiB at the filesystem's read rate. O31,
`token_embd.weight` routed as a GEMM weight rather than as a gather, caught by
`EmbeddingKernelCuda` refusing a Q4_K table by name. O33, the slot store sized
from the first layer instead of the file, caught when the lane streamed 527
slices and then refused at `blk.8`'s IQ4_XS tower. The run above is the same
command as the run that found each of them.

---

**The earlier arms, kept because they are what the answer was built out of.**

**`--device cpu` on the same box and artifact also emits ` Paris`** — `rc=0`,
`prompt_tokens=5 completion_tokens=1`, `generate` 950.249 s, `VmHWM` 44.46 GiB —
and **that token is NOT a streaming result**: a CPU queue makes
`needs_weight_staging()` false, no lane is built, and every routed-expert slice
is read in place. It is kept because two independent arms agreeing on ` Paris`
is worth more than either alone.

**`thor:gpu0` cannot reach a token on the CUDA arm at all**, and that is the arch
rather than the recipe: MLA prefill on this family IS FlashAttention, the
vendored FA2 covers `8.0,8.6,8.7,8.9,12.0a,12.1a`, and thor is `sm_110a`. Its
legs found O30 and O31.

**One operational lesson, because it cost two legs.** The runner on the share was
edited while two leases were executing it; `bash` reads a script incrementally,
so both resumed at a shifted offset. Stage a new path for a changed recipe.

---

**SUPERSEDED, 2026-08-31 — kept for the sequence it records.** The heading below
read "and that token is not a streaming result" when the CPU arm was the only one
that had produced one.
([#2214](https://github.com/mudler/vllm.cpp/issues/2214)). Both halves are the
record and neither may be quoted without the other.

**The token, exactly.** `dgx:gpu0` — `NVIDIA GB10`, `aarch64`, 20 cores, 119 GB,
compute capability 12.1, driver 580.173.02 — under an `rc` lease, base
`65a821980`, `vllm-cli --model <derived shard 1> --device cpu --prompt "The
capital of France is" --max-tokens 1 --temperature 0`. `rc=0`,
`prompt_tokens=5 completion_tokens=1`, and `stdout` is seven bytes,
`0x20 P a r i s 0x0a`:

```text
 Paris
```

Wall 1198 s for the whole process, of which `generate` is 950.249 s. **No speed
number is claimed and none may be read off that** (O10): it is one token, on a
CPU queue, reading expert slices out of a CIFS-backed mmap. `VmHWM` peaks at
46,618,820 kB = **44.46 GiB**.

**Why that token is NOT the goal, said before anything else quotes it.**
`--device cpu` makes `needs_weight_staging()` false, so `model_loader.cpp` never
builds the streamed-expert lane at all: `expert_stream::ExpertSlice` takes the
RESIDENT fallback and every routed-expert slice is read IN PLACE out of the
201.83 GiB mapping. That is the page-cache path §3.3 refuses to publish under a
streaming label, and the 44.46 GiB above is a page-cache figure. What this arm
establishes is the other half of the question — this port computes a coherent
first token from this checkpoint — and nothing about streaming.

**THE STREAMING LANE RAN, ON GB10, AND ITS COUNTERS EXIST.** Same box, same
artifact, `--device cuda`, `VT_MOE_EXPERT_STREAM=1`,
`VT_MOE_EXPERT_STREAM_SLOTS=4096`:

```text
[expert-stream] ON slots=4096 slot_bytes=4816896 resident=18.38 GiB
[expert-stream] steps=1 hits=0 misses=527 evictions=0 fills=527
                bytes=1876328448 exhausted=0 advised=0
```

**527 slices were paged out of the file into slots and 1,876,328,448 bytes moved,
with zero evictions and zero exhaustions.** That is the direct evidence the
loader-side argument below could only approximate, and it is the first time the
`pread` path has run on a real checkpoint at all (`expert-streaming.md` `## Owed`
said it never had). The load itself took 349 s and `VmHWM` peaked at 42.04 GiB —
the 18.38 GiB arena plus about 20.3 GiB of resident weight. The step then refused
by name at `blk.8`, on the slot BUDGET rather than on anything it computed, and
that is O33: the store was sized from the first layer's largest slice instead of
the file's, so `blk.8`'s IQ4_XS `ffn_down_exps` at 6,684,672 B did not fit the
4,816,896 B slots. Fixed here; the re-run is queued.

**One operational lesson, because it cost two legs.** The runner on the share was
EDITED while two leases were executing it. `bash` reads a script incrementally,
so both jobs resumed at a shifted byte offset: one died with
`syntax error near unexpected token '('` immediately after printing its result,
and the other re-ran a stale block and refused on a hard link that already
existed. Neither corrupted a measurement, and both lost their remaining legs.
Stage a NEW path for a changed recipe instead.

**The earlier `thor:gpu0` legs, kept because they are what found two of the
three defects.** `thor:gpu0` under an `rc` lease — `NVIDIA Thor`,
`aarch64`, 14 cores, 122 GB, compute capability 11.0, CUDA arch `sm_110a`, driver
595.78 — built `-DVLLM_CPP_CUDA=ON -DCMAKE_BUILD_TYPE=Release` from base
`65a821980`, `--device cuda`, `VT_MOE_EXPERT_STREAM=1`,
`VT_MOE_EXPERT_STREAM_SLOTS=4096`, prompt `The capital of France is`,
`--max-tokens 1`. **Every number in the rest of this section is a thor number**, and thor cannot
reach a token at all: `cmake/CudaArchFeatures.cmake`'s `fa2` row covers
`8.0,8.6,8.7,8.9,12.0a,12.1a`, thor is `sm_110a`, and MLA prefill on this family
IS FlashAttention with no fallback below it. The recipe is
[`.agents/scripts/glm53-dsa-streamed-load.sh`](../scripts/glm53-dsa-streamed-load.sh),
which is the script that ran rather than a description of one.

**The load itself: 866 s wall, and it ends with a sized engine.** All six shards
open and merge, 1809 tensors resolve, the MTP block is dropped, the absorption
runs, the KV cache is built — `[kv-alloc] source=spec kind=1 block_size=32
num_kv_heads=1 head_size=576 dtype=2 page_size_bytes=36864 num_blocks=256`, with
`max_model_len` auto-fitted from the file's 1,048,576 down to 8192 — and the
engine reaches its first step. That is the thing W7's test list asked for and W7
did not do.

**The residency number, and what establishes it.** `VmHWM` peaks at
**24,216,892 kB = 23.10 GiB**. Read the caveat before quoting it: while the
towers are mmap-resident `VmHWM` tracks page-cache pressure and is NOT the device
pool (O9), and `nvidia-smi --query-gpu=memory.used` reports `[N/A]` on this
device, so no device-pool figure was obtainable here at all. What the 23.10 GiB
DOES establish is the negative that matters: against §3.3's 14.511 GiB shard-header
resident class, plus O27's 4.48 GiB absorption, plus O31's 1.772 GiB bf16
vocabulary table less the 0.5 GiB of Q4_K it replaces — about 20.3 GiB predicted —
the process peaked 2.8 GiB above the prediction and **187.312 GiB below the
whole-artifact figure**. The towers were not materialized.

**Two defects the drive found, both ours, both fixed in the same flow.** O30: the
loader prefaulted the 228 expert towers, so the first attempt's `VmHWM` climbed
LINEARLY through 48.62 GiB at the filesystem's read rate with no plateau, and was
killed at 23 minutes rather than allowed to publish a page-cache number under a
streaming label. O31: `token_embd.weight` was routed as a GEMM weight rather than
as a gather, so on a device queue the Q4_K table survived to the first forward and
`EmbeddingKernelCuda` refused it by name. The plateau above is the first one's
after; the step below is the second one's after.

**Where it stops, verbatim, and it is the RECIPE and the ARCH rather than the
tree.**

```text
engine-fatal: EngineCore busy loop threw: cuda mla_prefill_attention: built
without the vendored FlashAttention-2 (VLLM_CPP_FLASH_ATTN). MLA prefill on
sm_121 IS FlashAttention — the upstream selector has no fallback below it
(mla/prefill/selector.py:191-194) — so there is nothing to degrade to.
```

The configure had said so in a line that carries no error —
`CUDA FA2 compiled-arch manifest: []` — because CUTLASS is not in the leased
container and the FA2 target needs its headers. The runner now stages the
CUTLASS 4.5.0 tarball already on the share and reads that manifest back in words
before any result is believed. **That repairs `dgx:gpu0` and it cannot repair
`thor:gpu0`:** `cmake/CudaArchFeatures.cmake`'s `fa2` row covers
`8.0,8.6,8.7,8.9,12.0a,12.1a`, `thor` is `sm_110a` and is not in that set, and
`dgx` is `sm_121a` and is. A first token on this row is therefore a GB10 result
by construction, which is what the goal already said.

**No token was observed on THIS box.** `load.stdout` is empty. The token above
is GB10's, on a different arm, and is labelled there.

**The streaming evidence, stated exactly, because this is the claim most easily
overstated.** No `[expert-stream]` line was printed, and that is not a failure of
the lane: `ExpertStreamLane` is constructed on the FIRST expert slice, the
forward threw in attention before reaching one, so there are no per-step
`fills`/`hits`/`bytes` counters and none are claimed. What IS established is
upstream of the counters and is not weak: `model_loader.cpp`'s streamed-lane
block was ENTERED — it is guarded on `needs_weight_staging() &&
host_memory_is_device_addressable() && factory->streams_routed_experts &&
ResolveExpertStreamRequested() && GgufExpertTowersReachSlotLane(...)`, all five —
and the load then SUCCEEDED, which it cannot do without the lane, because
`CheckDeviceWeightFit` would otherwise charge the device all 187.312 GiB of towers
(O22). So the lane was built, `RequireSlotCapacity` was reached and passed 4096
against the 1824 this model needs (O14, discharged), `streams_routed_experts`'s
only production reader ran (O29, discharged), and the resident footprint stayed
23.10 GiB. The counters are owed by the wave that takes a step.

**The gates, run by hand on the same binary.** `test_glm_moe_dsa_gguf_load` 5/228,
`test_glm_moe_dsa_forward` 7/5258, `test_glm_moe_dsa_config` 16/380,
`test_glm_moe_dsa_schedule` 12/533, `test_glm_moe_dsa_gguf_census` 3/3831 (the
third case reads the real published shards), `test_expert_stream_wiring` 4/882,
`test_expert_stream_capacity` 7/29 — all green. `test_gguf_keep_quant` reads
42 of 43 cases and 9 failed assertions, base-caused and recorded as O32.
`test_glm_moe_dsa_schedule` did not COMPILE on `origin/main` at all before this
wave, which is why it had not been run by hand since W9.

**The mutation, on the device that found the defect.** Restoring the tower
prefault at its call site and rebuilding reds the new case with
`prefaulted := 9` — exactly the fixture's 3 MoE layers x 3 towers — at 4 of 5
cases passing; the tree is then restored byte-for-byte, verified by sha256, and
rebuilt to 5/5 and 228/228.

**Next action: the streaming arm on `dgx:gpu0` with O33's fix.** Everything else
is in hand — the artifact, the repair, the lane, the counters and a token — and
what is owed is one run in which the slot budget is the file's own maximum, so the
step that streamed 527 slices runs to a token instead of refusing at `blk.8`.

---

**W7 LANDED ITS LOADER AND DID NOT PRODUCE A TOKEN, 2026-08-30**
([#2214](https://github.com/mudler/vllm.cpp/issues/2214)). Both halves of that
sentence are the record. `GlmMoeDsaForCausalLM` now materializes weights from a
`glm-dsa` GGUF through `LoadedEngine::FromModelDir`, and
`GlmMoeDsaModel::Forward` still refuses by name. W7's scope paragraph asked for
the loader and got it; W7's test list also asked for a first token, and that is
O21 and is owed.

**The census reproduces, and it reproduces exactly.** §3.4's numbers were
recomputed here from the six published shard headers rather than trusted: 1809
tensors, 216,705,819,648 payload bytes = 201.823 GiB, **228 expert towers at
187.312 GiB against 1581 resident tensors at 14.511 GiB**, and the per-type table
to the tensor — IQ3_XXS 71, IQ1_S 106, IQ2_XXS 44, Q5_K 312, IQ4_XS 4, Q8_0 476,
Q2_K 2, Q3_K 1, Q6_K 82, Q4_K 2, F32 709. The payload plus the six headers plus
62 bytes of 32-byte alignment padding is 216,715,365,893, the published byte
count exactly. The largest per-expert slice is 6,684,672 B, which §3.3 predicted.
`test_glm_moe_dsa_gguf_census` carries all of it; two of its three cases need no
artifact.

**The accounting closes at 1782 + 27 = 1809, and that is the gate.** There is no
token gate on this row and there cannot be one on this fleet (O1), so a tensor
the port does not claim is weight that is silently absent from the model with
nothing to catch it. The loader refuses rather than returning a model when the
sum does not close, and the negative case proves it by adding a tensor.

**Three things the file gets wrong are now handled by name rather than by luck.**
The MTP block is read, counted and dropped (27 tensors, spec O5). The 285
`indexer.*` tensors the conversion broadcast onto the 57 `shared` blocks are
dropped by SCHEDULE, and the mutation that makes the loader believe the file
instead — build an indexer on every layer — is killed by the gate; that failure
would otherwise load clean and carry 57 indexers the reference does not build.
And the file states its indexer schedule nowhere at all, which was confirmed key
by key on shard 1: **64 metadata keys and not one of `indexer.types`,
`index_topk_freq`, `index_skip_topk_offset`**. O17 is now diagnosed on the real
file rather than inferred, and `scripts/glm-dsa-write-indexer-types.py` repairs
the FILE from the model author's own `config.json` — the loader's refusal is
untouched, because D3 is right that a hardcoded table which happens to be correct
is the shape that silently becomes wrong on GLM-5.4.

**The i8mm quant repack is declined, and the reason is correctness rather than
speed.** It keeps the dtype and the byte count identical, so every assertion in
this tree passes on a repacked buffer while a consumer reading it as plain blocks
reads wrong values silently. The sibling Flash row paid for exactly that once
(#2241) on an artifact from the same publisher, and this arm's 476 Q8_0 tensors
are the MLA and indexer projections whose values decide the attention selection.
W7 claims no speed number, so the lever buys nothing here and no gate on this row
could see it go wrong.

**What did NOT happen, stated before anything else claims otherwise.** No `rc`
lease was taken: `dgx:gpu0` and `thor:gpu0` were both busy for the whole wave.
The artifact was **43% staged** when the wave ended, so the load was never
driven, no resident footprint was measured (O9 stands as arithmetic), no `pread`
number exists (O7), and **no token was observed** (O21). Every gate reported here
is a CPU gate on a synthetic model or a header read of the real one.

**One decision is recorded as a refusal to claim rather than as work.**
`kGlmMoeDsaFactory` does not set `streams_routed_experts`, because that flag
asserts this model's forward reads experts through the slot seam and this model
has no forward. The consequence is exact and is O22: without it
`CheckDeviceWeightFit` charges the device all 187.312 GiB of towers against
119.631 GiB and refuses. The flag and the forward land together or neither does.

**Next action: the forward (O21).** It is the last thing between this row and a
token, the pieces it needs are enumerated in O21, and a first token is reachable
on a fresh prompt while a second waits on O4. Then O22's flag, then the load on
`dgx:gpu0` under a lease, which discharges O7, O9, O14, O15 and O23 together.

---

**W5 IS STRUCK FROM THIS ROW AND RE-SEQUENCED, 2026-08-30**
([#1925](https://github.com/mudler/vllm.cpp/issues/1925),
[#2214](https://github.com/mudler/vllm.cpp/issues/2214)). W5 was dispatched as a
wave of this row, and the first obligation in its own scope paragraph — settle
the ownership before a line is written — resolved against it. No product code
was written. §3.7 W5 carries the evidence and `kv-dsv4-multicache.md` carries
the other half, as that paragraph requires.

**The short version: the conditional was already false when it was written
down.** §3.7 W5 says this wave lands here "if that row does not schedule it".
`KV-DSV4-MULTICACHE` had scheduled it — a `### W5 design` section, W5-1 through
W5-6, tracked by [#2323](https://github.com/mudler/vllm.cpp/issues/2323), with
W5-2's gated dispatch already implemented. Four branches are working that
plumbing right now. What made this worth an hour rather than a glance is that
the roadmap sentence and the code disagreed in the ordinary direction: the
record said "if nobody schedules it", and `git log --grep` said somebody had,
two days earlier.

**The near miss is the reusable part.** This wave's test list said *"the refusal
at `dots3_note_device.cpp:1147-1180` is deleted"*. Three things were wrong with
that one line and each was found by opening the file rather than trusting the
citation:

1. **It is not our refusal.** It guards `Dots3NoteForCausalLM` —
   `MODEL-DOTS3-NOTE`, [#699](https://github.com/mudler/vllm.cpp/issues/699) —
   and its own comment says the indexer cache "is `KV-DSV4-MULTICACHE` (#1925),
   **not this row**".
2. **Deleting it is forbidden by the owning row in as many words.** W5-2:
   "deleting the refusal is the one thing W5 must not do", because it restores
   the silent-discard failure W3 built it to prevent — a wrong-answer-not-a-crash
   invisible to a token gate, since the tokens stay right while the decode
   recomputes. The replacement is a DISPATCH on a declared capability.
3. **The line range does not contain the refusal.** The `VT_CHECK` is at
   `:1204-1227` on `origin/main` `03e0dcd19`, and at `:1205-1228` here — W4's
   own edit to that file moved it one line, inside this branch, which is why
   the durable citation is its predicate `!elig.prunes || elig.Active()` and
   not a range. `:1147-1180` is the comment above it. That wrong range is
   compiled into `kForwardRefusal` in `glm_moe_dsa.cpp`, so it is shipping to
   users today. Recorded as O20 rather than repaired here, because editing a
   registered model's refusal message is a behaviour change and not a record's
   to make.

**Anchors verified at parity pin `5559679229`**, so the next reader does not
re-run them. `DeepseekV32IndexerCache` (`deepseek_v2.py:696-701`) — exact. The
**132 B/token** figure — exact, and derived rather than quoted:
`128 + 128//128*4 = 132` at `dtype=torch.uint8`. The `MLAAttentionSpec` merge
rule (`kv_cache_interface.py:399-429`) — imprecise, not wrong; `merge` opens at
`:419` and the range's first half is `real_page_size_bytes`, which carries no
factor 2. One of four anchors wrong, one drifted, two clean.

**Even with the ownership question set aside, there is nothing to build.**
`glm_moe_dsa.cpp` is a config parser and a `kForwardRefusal` string: W2
registered the architecture with a forward that refuses by name, and **W7** (the
loader and the streamed towers) is undone, so no GLM-5.3 weight can be
materialized. A cache wired into a refusal stub is a shell under `## Nothing
lands dead` — provable only by a unit test that constructs the type by hand,
which shows the class works and never that anything reaches it.

**W1 IS ALREADY DONE, and this section said otherwise for one commit.** The
first draft of this entry carried "W1 and W7 are both undone" from the wave
table without checking the tree. `2e9f4d88d` (#2247, #2256) had already landed
`VecDotIQ4_XSQ8_K` for the sibling Flash row: defined `cpu_quant_dot.cpp:844`,
dispatched `:1004`, traits `cpu_quant_traits.cpp:121-122`, and
`KeepQuantDType` gates on `HasQuantDotKernel` at `gguf_keep_quant.cpp:176`.
`kIQ2_XS` landed with it at `:1003`. That is the SAME defect this entry was
written to record — a wave's status read off a record instead of off the code —
committed by the same session that had just documented it, one screen further
down. It is kept rather than quietly fixed for that reason. O2 is discharged,
O3's IQ2_XS half with it, and `IQ1_M` remains absent.

**A second thing falls out of that table, and it removes a question from W7's
path:** every encoding the `UD-IQ1_S` arm uses — IQ1_S, IQ3_XXS, IQ2_XXS,
IQ4_XS, Q2_K and Q3_K — now has a `vec_dot` row on `origin/main`, verified 6 of
6 on 2026-08-30. W7 therefore has no decoder or keep-quant question left to
answer; the whole arm keeps its blocks and stays in the streaming lane.

**Next action: W7**, which is now the critical path — it is what turns O19 from
a staged slice into a reached one, and what gives this row's re-sequenced W5 a
forward to consume caches into. Then W6 (sparse prefill). This row's W5 opens as
a consumption site once `KV-DSV4-MULTICACHE` W5 lands, on the shape
`MODEL-MM-GLM53-FLASH` already proved (`glm5_next_kv.{h,cpp}`,
`ModelFactory::consumes_multi_kv_cache`).

---

**W4 LANDED, 2026-08-30** ([#2214](https://github.com/mudler/vllm.cpp/issues/2214)).
The heterogeneous indexer schedule, the `skip_topk` selection reuse and the fp32
router gate GEMM are on this row's integration branch, on top of W2 and W3.
`GlmMoeDsaMlaSchedule` turns the parsed `indexer_types` into 78
`mla::MlaBlockDims`, 21 of them carrying an indexer and 57 carrying `skip_topk`;
with the MTP block upstream forces full at `deepseek_v2.py:1110-1115` that is the
22 of 79 §3.5.1 counted, and the test asserts the split from the checkpoint's own
`config.json` rather than from a literal.

**The reuse is the ABSENCE of a write, and reading it as a copy is how a port
gets it wrong.** Upstream allocates ONE `topk_indices_buffer` per model
(`deepseek_v2.py:1372-1377`) and hands the same tensor to every layer (`:1395`,
`mla.py:120`). `mla.py:180` — `if self.indexer and self.is_sparse and not
self.skip_topk:` — runs the indexer only on a full layer, and a shared layer's
indexer does not exist at all (`:1134-1135`). So the bytes a shared layer attends
through are the ones its owning full layer left there earlier in the same forward
pass; nothing is copied, cached or carried across steps.
`sparse_mla_attention.py:303-305` says it in upstream's own words. Mirrored as
`mla::MlaSharedSelection`: a full layer writes INTO it, a shared layer reads it,
and a `skip_topk` layer handed no buffer is REFUSED rather than falling through
to the dense contiguous key loop — which would have produced a finite, plausible,
wrong output on 57 of 79 blocks that no token gate could see.

**One polarity in this seam now points both ways, deliberately.**
`mla_attention.cpp:943-945` already said the decode metadata is copied so that "a
sliding layer must not inherit a full layer's selection". That is still true and
still dots3-note's. GLM's shared layer inherits BY CONFIGURATION, from a buffer
the caller allocated for it, and never from leftover metadata. Both statements
are in the file, next to each other.

**The fp32 router turned out to be a DeepSeek-V2 parity repair as well as a GLM
need.** `_get_moe_router_dtype` (`deepseek_v2.py:123-133`) returns f32 for
`model_type == "glm_moe_dsa"` at `:127` AND for any config declaring
`moe_router_dtype: "float32"` at `:131`; the tree hardcoded bf16, so a DeepSeek-V2
or V3 checkpoint asking for an f32 router silently did not get one.
`DeepseekV2Params::router_dtype_is_f32` is read in `ParseDeepseekV2Params` and
consumed at `deepseek_v2.cpp:363`. The dtype answers are the pinned oracle's own
return values, not a transcription: `_get_moe_router_dtype` was extracted from
`5559679229` and EXECUTED on eight configs with torch 2.11.0+cu130. The order of
its two arms is what the table gates — `:127` wins even against an explicit
`"bfloat16"`, and a rule written the other way round passes every other row.

**And the dtype selection at `:363` is not gateable, which is measured.** The f32
and bf16 arms of the tiny DeepSeek-V2 forward are bit-identical over all 500
logits, because every activation downstream of the router combine is stored at
bf16. Forcing f32 unconditionally leaves that whole file green. That is O18, and
it is AGENTS.md's "a token gate cannot detect a dtype that is too wide" arriving
as a concrete instance rather than a warning.

**What W4 does NOT reach.** The reuse arm has no production caller: only a GLM
decoder layer sets `skip_topk` on a real forward, and `GlmMoeDsaModel::Forward`
still refuses by name. The SCHEDULE is reached — `ParseGlmMoeDsaConfig` runs it
for every config `ModelRegistry::Resolve` sees, and deleting that call reds the
focused gate — but the block arm is a staged slice, owned by W7 and recorded as
O19. W4 also closed a hole W2 left and W4 would have been first to fall into: a
non-`default` `rope_type` is now refused, because `MlaAttentionScale` would
otherwise have handed a YaRN checkpoint the unscaled softmax scale.

Gates: 24 cases and 2594 assertions across the two new/extended suites, plus
`test_mla_attention_block` (18 cases, 2,255,433 assertions), `test_dots3_note_attn`
(51 / 6888), `test_glm_moe_dsa_config` (15 / 380) and `test_deepseek_v2_load`
(4 / 14) all green and unmoved. Seven mutations: deleting the reuse read kills
G1; making the full layer keep its selection local kills G1 and the tautology
guard; deleting the schedule's production call site kills the reachability case;
neutering the router-dtype parse kills two; dropping either refusal kills one
each; and forcing f32 at the GEMM call site kills NOTHING, which is O18.

**Next action:** W1, still independent and still unlocking two arms at once, then
W5 (the indexer KV side cache, GPU) and W6 (sparse prefill, GPU). W7 is what
turns O19 from a staged slice into a reached one.

---

W2 LANDED, 2026-08-30. The row moves `SPIKE` -> `ACTIVE` (`📋` -> `🚧`),
which is what D4 said would happen when W2 landed. `ACTIVE` rather than
`PARTIAL` because `CLAIM-MODEL-GLM-MOE-DSA` is still open over seven remaining
waves, and `scripts/check-agent-record.py` holds an active claim's rows to
`SPIKE` or `ACTIVE`; the row moves to `PARTIAL` when the claim closes. What is on `main`
now: `GlmMoeDsaForCausalLM` is registered from its own translation unit, its
config resolves from a `config.json` and from a `glm-dsa` GGUF header through
ONE validator, `kGgufArchArms` carries a `glm-dsa` row, and the forward refuses
by name and lists all seven missing primitives. It loads no weight and computes
no token, and O16 names the three surfaces nothing reaches yet.

**The three-way agreement of §3.5.1 is now executable, and it holds.** The
checkpoint's own 78-entry `indexer_types` (committed verbatim as
`tests/vllm/models/glm_moe_dsa_config_glm53.inc`, `zai-org/GLM-5.3` revision
`935644c05e76fc198714f4cca449fd8b970ff6d7`), upstream's derived rule at
`deepseek_v2.py:1097-1101` evaluated at `freq = 4` / `offset = 3`, and
llama.cpp's `GLM_5_2_DEFAULT_INDEXER_TYPES` (`b10451:src/models/glm-dsa.cpp:6-27`)
agree on all 78 entries, 21 of them `full`. The same case proves the derivation
is not a constant: three other (freq, offset) pairs produce three other
schedules. `mlp_layer_types` likewise reproduces exactly from
`first_k_dense_replace = 3`, and a config whose two statements disagree is
refused rather than resolved to either.

**Everything else in this section still holds.** The port is smaller than §0.2
implies, the blocker that remains is one quantization kernel, and the gate is
the honest cost: vLLM at the pin implements this architecture and cannot run it
on any device this project can reach, so no wave may promise a token-exact
number against it.

**Next action:** W1, which is independent of W2 and unlocks two arms at once.
W3 has since landed on this row's integration branch; its own record is the
`W3 LANDED` paragraph further down this section. W2's own residue is O16 (the
unreached forward) and O17 (the staged artifact states no indexer schedule, so
the arm that exists still cannot be fed).

---

The state this section recorded when the spike was written, kept because the
arithmetic is what makes the row's position defensible:

`SPIKE`, 2026-08-29. The row moves off `🚫 BLOCKED` because the blocker was
computed in the wrong frame, and the correct frame is measured here: **97.49% of
this model's parameters are routed experts, the resident class is 14.511 GiB in
every published UD arm, and one decode step at `c = 1` touches 1800 expert slices
= 11.21 GiB of uniform slots.** Resident plus a 4096-slot cache is 40.01 GiB
against 119.631 GiB on `dgx:gpu0`. Nothing is implemented; `GlmMoeDsaForCausalLM`
appears nowhere under `src/` or `include/`, and `ParseDeepseekV2Params` refuses
this checkpoint at the `index_topk` tripwire before anything else runs.

Three findings shape what happens next, and each corrects something this
repository previously believed.

**The port is smaller than §0.2 implies.** That section's verdict — "GB10 cannot
run DSA end-to-end" — was a statement about vLLM's flashinfer sm120 path, and it
is still true of that path. It is no longer a statement about ours: a
device-native DSA lightning indexer now lives in the shared MLA block
(`mla_attention.cpp:598-745`) with CPU and CUDA `DsaIndexerLogits` /
`DsaTopkSelect`, reached in production by `Dots3NoteForCausalLM`. The MLA
geometry this model needs already validates and already dispatches to native
kernel instantiations. What is left is the indexer KV side cache (O4), sparse
prefill (O6), and the schedule/reuse semantics.

**The blocker that remains is one quantization kernel, it is named, and
`origin/main` changed which half of it is missing while this spec was being
written.** The census of `UD-IQ1_S` over its own shard headers says the arm is
106 IQ1_S + 71 IQ3_XXS + 44 IQ2_XXS + 4 IQ4_XS + 3 K-quant expert tensors. At
this branch's base `60a6dd97b`, `IQ4_XS` had no decoder and the arm refused
loudly at load. At `94de63ff5`, landed 2026-08-29 for the sibling Flash row
([#2245](https://github.com/mudler/vllm.cpp/issues/2245)), it has a decoder and
still no keep-quant `vec_dot` — so the arm now LOADS, expands those four towers
from 6.375 GiB to 24.000 GiB, and drops out of the expert-streaming lane
altogether, because `gguf_device_fit.cpp:85-100` is all-or-nothing across a
model's `*_exps` tensors. **The missing piece is `VecDotIQ4_XSQ8_K`, and the
failure it prevents is now silent rather than loud.** That is the sharper form of
this finding and it is why the census had to be redone against the merged tree
rather than trusted from an hour earlier.

**The gate is the honest cost.** vLLM at the pin implements this architecture and
cannot run it on any device this project can reach, so no wave may promise a
token-exact number against it. What W3 onward can prove is that a streamed slice
and a resident tower produce identical logits, which is the row's actual novelty
and needs no oracle at all.

**W3 LANDED 2026-08-30** ([#2214](https://github.com/mudler/vllm.cpp/issues/2214)),
and it is the first wave of this section with product code on `main`. The
expert-streaming wiring is now `expert_stream_seam.{h,cpp}`: `ExpertStreamLane`
(was `Qwen35ExpertStream`), `ExpertStreamStepGuard`, `HostSliceView` and
`ExpertSlice`, moved rather than rewritten. `qwen3_5.cpp` shrinks by 556 lines
and keeps four alias declarations. O8 is discharged; O13, O14 and O15 are new
and named.

Three things the lift found or decided, each of which a later reader would
otherwise rediscover.

**The resident fallback could not be a call, and that is O13.** `qwen3_5.cpp`
defines its own `ResidentWeight` in its unnamed namespace, shadowing
`dense_attn_block.h:181`'s, and the two differ: the local one refuses a streamed
tower, an `elem_kn_repacked` weight and a `repacked` weight at device staging,
and carries the host-alias arm. A shared `ExpertSlice` that called
`ResidentWeight` itself would have bound to the header's definition and dropped
every one of those guarantees from Qwen3.5's streamed lane, with no gate in the
tree able to see it — the tokens would still be tokens. So `ExpertSlice` takes a
`ResidentSliceFn` and each model passes its own. Byte-identity then holds by
construction rather than by inspection.

**Qwen3.5's byte-identity is measured, not argued.** Two separately-built
libraries differing only in `qwen3_5.cpp` — the lifted file and `origin/main`'s,
same build directory, same flags, both relinked and confirmed distinct by
sha256 — run the same synthetic forward through `Qwen3_5Model::Forward` and
produce logits digest `84ae1a52ee64d117` over 160 floats, with streaming OFF and
with streaming ON, and with a character-identical
`[expert-stream] steps=1 hits=0 misses=42 evictions=0 fills=42 bytes=45696
exhausted=0 advised=42` line. The one deliberate text change is the step guard's
refusal prefix, `qwen3_5:` to `expert stream:`, because the guard is no longer
that model's; `test_expert_stream_steps` matches on `must not nest` and is
unaffected.

**The capacity refusal is at LOAD, and it fires on every streaming load rather
than only this row's.** §3.3 argued for it and W3 ships it: below
`streamed_towers * experts_per_tok` the loader refuses by name instead of
letting `Slice` return nullptr and the caller read the tower in place out of the
mmap. Both terms are read off the GGUF header the lane will serve
(`GgufStreamedExpertLaneGeometry`), so the refusal and the forward cannot
disagree about a checkpoint. This DOES change one behaviour outside this row: a
`VT_MOE_EXPERT_STREAM=1` load of a large Qwen3.5 at the default 64 slots now
refuses rather than silently degrading. That is the intended polarity and it is
not a numerical change — streaming is default OFF, the six existing streaming
binaries do not load through `FromModelDir`, and all of them stay green. An
unknown geometry is inert rather than a refusal, so an architecture whose
metadata the reader did not understand is never refused by a number the reader
invented.

**Next action:** W1 and W2, both CPU, both independent. W1 belongs to
`QUANT-GGUF-IQ4_XS` and unlocks two arms at once. W4 is now unblocked on W3.

### 3.11 The second decode step, and the read-key registered before it runs

**Status: the legs are QUEUED on `dgx:gpu0` and nothing below is a result.**
Issue [#2596](https://github.com/mudler/vllm.cpp/issues/2596). This section is
written before the measurement on purpose, so that what it predicts cannot be
fitted to what it later reports.

**The gap §3.10 leaves open, stated as the counter that says it.** §3.10's
landed run is `--max-tokens 1`, and its token comes out of prefill. Its own
final line is `steps=1 hits=0 misses=6399 evictions=0 fills=4096
exhausted=2303`. `hits=0` and `evictions=0` are not measurements of slot reuse;
they are the absence of any. The lane was filled and never read, because nothing
asked it for a second step. **No second decode step has ever run on this model,
on this box or any other**, so the slot cache — the whole mechanism this row
depends on — is exercised here at its weakest possible point.

**Why the first multi-token run is also a correctness question.**
[#2544](https://github.com/mudler/vllm.cpp/issues/2544) names `glm_moe_dsa`
among five architectures that never read
`ModelForwardInput::device_token_ids`, and it is explicit that its list is filed
on a `grep` and that each entry is "a candidate, not a conviction". Two facts
make the candidate reachable on precisely the box this row is gated on, and both
are read off the tree at `5649e07d2`:

1. No `glm_moe_dsa` translation unit mentions `device_token_ids`.
   `ForwardGlmMoeDsaForCausalLM` (`glm_moe_dsa_registry.cpp:96-107`) passes
   `input.token_ids` through with no `detail::DeviceTokenIdsScope`, where
   `qwen3_moe_registry.cpp:100` takes one.
2. `GPUModelRunner::async_device_mirror()` (`runner.cpp:4577-4601`) engages on
   **integrated OR discrete** CUDA, and its own comment names the integrated arm
   "GB10, UnifiedMemory AND is_integrated_gpu" as DEFAULT ON. `dgx:gpu0` is a
   GB10, so the mirror is live and the host `token_ids` a decode row reads is
   the stale one.

**The read-key.** Two legs, one variable, the same binary and the same artifact,
with expert streaming ON in BOTH (`VT_MOE_EXPERT_STREAM=1`,
`VT_MOE_EXPERT_STREAM_SLOTS=8192`), `--device cuda`, prompt
`The capital of France is`, `--max-tokens 4`, `--temperature 0`:

| leg | env |
|---|---|
| C | default |
| E | `VT_ASYNC_DEVICE_MIRROR=0` |

- **C and E differ ⇒ CONVICTION.** `glm_moe_dsa` embeds the stale host
  identifiers, so every decode step after the first generates from token id 0.
  This is the same isolation that convicted
  `Glm5NextForConditionalGeneration` (#2544's comment, `dgx:gpu0`, 2026-09-01),
  and leg E is then the only arm whose text may be quoted.
- **C and E agree ⇒ FALSIFICATION.** #2544's grep-based candidacy is wrong about
  this model, and that is recorded as such rather than left as a suspicion.

**Neither outcome says anything about expert streaming**, which is ON in both
legs. Whichever leg emits text, that text is streamed text. The variable moved
is the async device mirror and nothing else.

**The recipe is the script that runs, not a description of one.**
[`.agents/scripts/glm53-dsa-second-step.sh`](../scripts/glm53-dsa-second-step.sh)
is byte-identical to the file the lease executes (sha256
`b811f4b54c43e830f2990e93e6fde45905e888f17b25c87add6ee4174a18038f`), and it
carries the read-key above in its own header so the prediction travels with
the numbers. Every verdict it prints is an expression over bytes it reads
back off disk; no conclusion string in it is a constant. That is deliberate:
a job on this box has reported `3 runs all rc=0` over its own captured
`rc=1`, from a label that never read its input.

**Bytes are compared as bytes.** The sibling's divergence was ` Paris.` against
` Paris Paris`, which agrees for five characters; a `diff` of rendered text is
not the instrument, and the runner records `od -An -tx1` of each leg's stdout.

**What the run may NOT produce.** A rate. This row has no vLLM denominator at
any revision, and `## Gates` binds: correctness first, and a decode whose
identifiers are in question is not a correctness result. If the legs diverge, no
tok/s from either arm is admissible.

**Owed if the lease does not free.** `dgx:gpu0` is the only GB10 the project can
reach, and it is the only part where `host_memory_is_device_addressable()`
answers true for this artifact's 187.312 GiB of towers. If the legs do not run,
the gap above stays exactly as written and is owned by
[#2596](https://github.com/mudler/vllm.cpp/issues/2596): the reuse counters stay
unmeasured, and #2544's candidacy for this model stays unresolved. It is not
inferrable from the one-token run, and this section does not infer it.

### 3.12 The candidacy is CONVICTED on CPU, and the repair is two lines

**Status: measured and fixed here. The `dgx:gpu0` legs of §3.11 are still
QUEUED and are still owed**, because what they answer is not what this section
answers. Issue [#2596](https://github.com/mudler/vllm.cpp/issues/2596).

**What was measured.** A new case in `test_glm_moe_dsa_forward.cpp` drives four
pure-decode steps through `ModelRegistry::Forward` — the production entry point,
not a hand-built type — on the tiny fixture, as the A/B/C triple
`test_moe_async_device_ids.cpp` established for the same contract:

| run | host ids | mirror | expected |
|---|---|---|---|
| A | true | none | the reference |
| B | zeros | none | must DIFFER from A |
| C | zeros | carries A's | must EQUAL A |

B is the control. Without it a forward that ignored its identifiers entirely
would satisfy C, and so would a gate whose two runs shared a buffer.

**RED, at `d0f7fd2c6` plus the test alone:**

```text
CHECK( differing == 0 ) is NOT correct!  values: CHECK( 128 == 0 )
[doctest] test cases:  1 |  0 passed | 1 failed | 7 skipped
[doctest] assertions: 81 | 80 passed | 1 failed |
```

**Both counts are quoted because only the pair separates a failed assertion from
a thrown case.** 80 of 81 assertions passed, so run B's control fired correctly
and the case FAILED rather than threw: all 128 logits (4 steps x 32 vocab)
differ. `glm_moe_dsa` did not read `device_token_ids` at all.

**The repair is two lines and neither is new.** The registry publishes the field
for the duration of the forward (`detail::DeviceTokenIdsScope`, exactly as
`qwen3_moe_registry.cpp:100` does), and the embed consumes it
(`detail::ApplyDeviceTokenIds`, exactly as `deepseek_v2.cpp:586` does). Both
helpers already exist in `qwen3_5_internal.h` beside the publisher, because
#1305 put them there to stop a fifth hand-rolled copy. Nothing here is a new
mechanism; this registration simply never joined the nine that consume it.

**GREEN, whole suite:** `8 | 8 passed | 0 failed`, `assertions: 5339 | 5339
passed | 0 failed`, `0 differing`. Seven adjacent suites are green on the same
build: `test_glm_moe_dsa_{config,gguf_load,gguf_census,schedule}`,
`test_moe_async_device_ids`, `test_expert_stream_{wiring,steps}` — 60 cases and
11,319 assertions in total, none failing.

**Three mutations, each REBUILT and each restored byte-for-byte.** A mutation
that does not rebuild reads as a passing test, so the build rc is recorded
beside the verdict and each mutant binary's sha256 differs from the green one's
`f367624476c5b584961c3d1d00c41d03220df85be9256828aa6ad74f704450c6`:

| mutation | build rc | result |
|---|---|---|
| delete the CONSUMER (`ApplyDeviceTokenIds`) | 0 | RED, `CHECK( 128 == 0 )` |
| delete the PUBLISHER (`DeviceTokenIdsScope`) | 0 | RED, `CHECK( 128 == 0 )` |
| neuter the CONTROL (run B stops staling the host ids) | 0 | RED, `CHECK( 0 > 0 )` on every step |

The first two are the reachability proof `## Nothing lands dead` asks for:
deleting either production call site reds the gate, so the case measures a
capability and not a class. The third proves the control is not vacuous — it is
the arm that stops a forward which ignores identifiers entirely from passing.
All three source files restored to their pristine sha256, verified by `diff`
against the hashes taken before the first mutation.

**WHAT THIS DOES NOT SETTLE, and why §3.11's legs still owe their run.** This is
a CPU gate on a 1/1000th-scale fixture. It proves the CONTRACT was unmet and is
now met. It does not show the symptom on the published 201.83 GiB checkpoint, it
does not exercise the slot cache at the real model's working-set size, and it
says nothing about the reuse counters §3.11 exists to measure. The staged
runner deliberately builds from **base `main` without this repair**, so its leg
C still carries the defect and its leg E is still the discriminator: that pair
is what convicts on the real artifact, as it did for the sibling model. A CPU
fixture agreeing with a two-line fix is not a 753.33B model generating text.

**No rate is claimed here either.** Nothing in this section is a performance
measurement, and the row still has no vLLM denominator at any revision.

### 3.13 The legs ran: four tokens, the lane reused slots, and the read-key CONVICTED

**Status: measured.** `dgx:gpu0` (`NVIDIA GB10`, `sm_121a`, CUDA arch `121a`,
FA2 manifest `[121a]`), one `rc` lease, job
`c7f4c7ea-73d1-4c00-82af-61aae5617979`, worker `rc-worker-4b8lj`, base
`5649e07d2`, both legs from ONE binary, 2026-09-02.
Issue [#2596](https://github.com/mudler/vllm.cpp/issues/2596).

**THE GENERATED TEXT, verbatim, from the real 201.83 GiB artifact.**

| leg | env | rc | wall | stdout | bytes |
|---|---|---|---|---|---|
| C | default | 0 | 1059 s | ` Paris The Paris Paris` | `20 50 61 72 69 73 20 54 68 65 20 50 61 72 69 73 20 50 61 72 69 73 0a` |
| E | `VT_ASYNC_DEVICE_MIRROR=0` | 0 | 1061 s | ` Paris, which is` | `20 50 61 72 69 73 2c 20 77 68 69 63 68 20 69 73 0a` |

**The read-key §3.11 registered BEFORE the run is resolved, and it CONVICTS.**
The bytes differ on one variable, and that variable is not the streaming lane.
`glm_moe_dsa` read `token_ids` and ignored `device_token_ids`, so every decode
step after the first was generated from token id 0. Leg C's ` Paris The Paris
Paris` is that symptom; leg E's ` Paris, which is` is the same model, same
binary, same artifact, with the mirror off. **This is the behavioural half of
§3.12's CPU conviction, on the published checkpoint**, and it is the same
isolation that convicted `Glm5NextForConditionalGeneration` under #2544.

**THE LANE WAS ON IN BOTH LEGS, AND FOR THE FIRST TIME IT REUSED SLOTS.**

```text
                    [expert-stream] ON slots=8192 slot_bytes=6684672 resident=51.00 GiB
leg C  steps=1 hits=0    misses=6393 evictions=0   fills=6393 bytes=22026289152 exhausted=0
       steps=2 hits=789  misses=7404 evictions=0   fills=7404 bytes=25511657472 exhausted=0
       steps=3 hits=1791 misses=8202 evictions=10  fills=8202 bytes=28255223808 exhausted=0
       steps=4 hits=3156 misses=8637 evictions=445 fills=8637 bytes=29729193984 exhausted=0
leg E  steps=4 hits=2688 misses=9105 evictions=913 fills=9105 bytes=31334006784 exhausted=0
```

`steps=4` and `exhausted=0` are the two `docs/ENVIRONMENT.md` names as the test
of a live lane, and both hold in both legs. **`hits` moved 0 -> 789 -> 1791 ->
3156**, which is the slot reuse §3.10's one-token run could not measure and
§3.11 was written to obtain: at 8192 slots the working set fits, nothing is
refused to the mmap, and eviction begins only at step 3. 29.73 GB moved through
a 51.00 GiB arena in leg C against 187.312 GiB of towers.

**Peak resident: 76,092,816 kB = 72.57 GiB (leg C) and 76,442,608 kB = 72.90 GiB
(leg E), against the pool's 119.631 GiB.** `nvidia-smi memory.used` reads 0 MiB
on this part and is not a residency figure here; `VmHWM` tracks page-cache
pressure while the towers are mmap-resident and is not one either (O9).

**Reference-tier hits: `exhausted=0` on every line of both legs.** No slice fell
back to reading the tower in place, which is the §3.10 run's `exhausted=2303`
resolved by the larger budget.

#### THIS WAS ALSO A PLACEMENT RUN, AND THAT IS NOT WHAT THE ROW ASKED FOR

`VT_CPU_MOE` was unset and the runner asserts that rather than claiming it.
The engine placed experts anyway, on its own `--fit`, in BOTH legs:

```text
engine: device placement INSTALLED: 33 layers run their routed experts on cpu,
        the rest on cuda (resolved against 78 layers, origin fit)
engine: device placement: --fit placed 33 layer(s) (89288343552 B) to bring a
        216433205760 B footprint under a 128452960256 B budget
```

**So these numbers are a HYBRID and must be read as one.** The streaming lane is
genuinely live — the banner, `steps=4`, the climbing `hits` and `exhausted=0`
are its own counters, and 83.14 GiB of expert bytes moved through slots that no
placement performs. But 33 of 78 layers ran their routed experts on the HOST
CPU, so this is not the pure streamed configuration, and no line above may be
quoted as one. The pure arm needs the fit suppressed, and it has not run.

**#2568 is live on exactly this shape.** `quant_repack` is a per-load flag read
off the ENGINE device, so on this aarch64 CUDA engine the 33 host-executed
layers loaded WITHOUT the i8mm repack. That is throughput, not correctness, and
it is another reason no rate below is publishable.

**NO RATE IS CLAIMED, and `tok_s=0.005` in the logs is not one.** Leg C's tokens
are wrong, so its timing measures a defect. Leg E's are coherent but it has no
vLLM denominator at any revision, it ran the hybrid placement above, it is n=1,
and its reads came off CIFS. `## Gates` binds: correctness first.

#### What is still owed

1. **The repair has not been driven on the artifact.** §3.12's fix landed after
   the tarball this lease built, deliberately, so leg C carried the defect and
   the pair stayed a discriminator. What is owed is one leg on a FIXED binary
   proving the DEFAULT arm now emits leg E's bytes without the env var.
2. **The pure streaming arm**, with the `--fit` placement suppressed, so the
   lane's numbers are not entangled with 33 host layers.
3. **A second slot budget**, since 8192 never exhausted and 4096 exhausted on a
   third of its misses at one token: the floor between them is unmeasured.

## §3.14 -- the #2596 fix is DRIVEN ON THE ARTIFACT, and the default arm is correct

`devids2544`, `dgx:gpu0`, finished 2026-09-03T18:38:56Z, on the real 201.83 GiB
`UD-IQ1_S` artifact (six shards, shard 1 sha256 `b3e9838651a5c279...`).

**This is the validation §3.13 recorded as OWED.** The `device_token_ids` repair
landed at `d8683402b` but the lease that found the defect deliberately built base
`main` to keep its two legs a discriminator, so "the default arm now emits leg E's
bytes unaided" was an INFERENCE from a CPU gate. It is now measured.

| leg | env | rc | bytes | stdout |
|---|---|---|---|---|
| D1 | **no mirror variable** | 0 | 17 | `20 50 61 72 69 73 2c 20 77 68 69 63 68 20 69 73 0a` = ` Paris, which is` |
| D2 | `VT_ASYNC_DEVICE_MIRROR=0` | 0 | 17 | identical |

D1 and D2 agree byte-for-byte, so the mirror variable no longer changes the
answer for this model. That is the close, not D1 alone.

**EXPERT STREAMING WAS LIVE WHILE IT RAN, and the counters say so rather than the
configuration:** `steps=4 hits=2688 misses=9105 evictions=913 fills=9105
bytes=31334006784 exhausted=0`. Climbing `hits` at `exhausted=0` is slot REUSE;
`VT_MOE_EXPERT_STREAM_SLOTS=8192`, stats every step. Configuration alone cannot
distinguish a streamed run from a fallback (#2505), which is why the counters and
not the env are quoted here.

**Not claimed.** No speed number: `wall=1549 s` (D1) and `1487 s` (D2) are
DURATIONS, ~85% GGUF load off CIFS. This says nothing about the pure streamed arm
either -- the engine's own `--fit` places layers on the host, so these legs remain
a HYBRID, exactly as §3.13 records.

The FA2 build requirement was asserted twice before any leg ran
(`VLLM_CPP_FLASH_ATTN:BOOL=ON` plus 1296 TUs carrying the define), because this
model is MLA and a silently non-FA2 build throws in the first forward in a way
that reads as a model defect.
