# SPIKE: recent-dense TEXT batch (Phi / Command-R / Granite / StableLM / InternLM2 / MiniCPM / Phi-3-4)

**SPIKE ONLY — no implementation, no kernels, no build, no benchmark, nothing
downloaded (only HF API metadata read).** A BATCH-SCOPING triage of the next tier
of recent dense (and small-MoE) text families, so the sweep has a RANKED, GROUNDED
implementation queue and near-additive families can be brought up rapidly, one impl
agent each, OLMo-2-style. Design + records only.

**Base:** `origin/main` `5c00fc4` (OLMo-2 `Olmo2/Olmo3ForCausalLM` W0-W4 landed).
**Oracle pin:** `/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:**
`~/venvs/vllm-oracle` = vLLM **0.25.0**. **Claim:** `CLAIM-SWEEP-RECENT-DENSE`.
**Parent plan:** [`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B (the ranked
model sweep). **Gold-standard spike shapes mirrored:** [`sweep-olmo2.md`](sweep-olmo2.md)
(the ZERO-new-kernel dense bring-up pattern), [`sweep-gemma.md`](sweep-gemma.md)
(the multi-row family batch + per-version delta table).

Rows advanced `INVENTORIED` -> `SPIKE` by this spike (8):
`MODEL-TEXT-phi3-phi3-for-causal-lm`,
`MODEL-TEXT-granite-granite-for-causal-lm`,
`MODEL-TEXT-stablelm-stablelm-for-causal-lm`,
`MODEL-TEXT-minicpm-mini-cpmfor-causal-lm`,
`MODEL-TEXT-internlm2-intern-lm2-for-causal-lm`,
`MODEL-TEXT-commandr-cohere-for-causal-lm`,
`MODEL-TEXT-phi-phi-for-causal-lm`,
`MODEL-TEXT-minicpm3-mini-cpm3-for-causal-lm`.

Called out but NOT a new row here (each near-free on a LANDED row): **OLMo-3**
(`Olmo3ForCausalLM`, rides the ACTIVE `MODEL-TEXT-olmo2-olmo2-for-causal-lm` row as
its W5 sliding-window follow-on — the cheapest next win) and **InternLM3 / Yi**
(both map to the ACTIVE `MODEL-TEXT-llama-llama-for-causal-lm` row — registry-alias
additions, not separate architectures).

Left `INVENTORIED` (characterized below with a reason, deprioritized): Falcon
(`FalconForCausalLM`, older + alibi/parallel branches), Falcon-H1
(`FalconH1ForCausalLM`, Mamba2 SSM hybrid — campaign), GraniteMoe /
GraniteMoeShared / GraniteMoeHybrid, Cohere2Moe (`Cohere2MoeForCausalLM`), PhiMoE
(`PhiMoEForCausalLM`) — all MoE/SSM campaigns. And the pin-REMOVED names (see §0.4).

---

## 0. Headline findings

### 0.0 The batch is dominated by ZERO-NEW-KERNEL near-additives over the LANDED substrate

The dense + LayerNorm + scalar-multiplier + partial-rope + MLA infrastructure that
Qwen3/Llama/Mistral/OPT/GLM/Gemma/OLMo-2/DeepSeek-V2 already landed covers most of
this tier with **new files only**. Of the 8 newly-scoped rows, **4 are
ZERO-NEW-KERNEL near-additive** (Phi-3/Phi-4, Granite-3, StableLM, MiniCPM), and two
more zero-new-kernel bring-ups are available off LANDED rows (OLMo-3 W5; InternLM3/Yi
Llama aliases) — **7 zero-new-kernel bring-ups total**. The 4 `small-new-op` rows add
at most ONE small thing each (a plain non-gated GELU unary for Phi-1/2; a scalar +
parallel-residual wiring for Command-R; a fused-`wqkv` interleaved split for InternLM2;
MLA re-wiring for MiniCPM3). No row in this batch needs a genuinely-new compute kernel
of the fp4/GDN/attention class.

### 0.1 The landed primitive inventory this batch reuses (verified in `src/`/`include/` @ `5c00fc4`)

- **Plain RMSNorm** `vt::RmsNorm {gemma=false}` (`include/vt/recipes.h`, `ops.h:90`) —
  Phi-3/4, Granite, MiniCPM/3, InternLM2.
- **`nn.LayerNorm` (mean+variance, weight AND optional bias)** — `vt::LayerNorm`
  (`ops.h:181`, `include/vt/ops.h:1215`), landed for OPT. Covers StableLM (weight+bias
  optional), Phi-1/2 (`input_layernorm`/`final_layernorm`), and Command-R's Cohere
  `LayerNorm` (no-bias -> null bias pointer).
- **Partial NeoX RoPE** (`rotary_dim < head_dim`) — the partial in-place + from-cache
  paths (`ops.h:1243,1249,1310`), landed for GLM-4 decoupled rope. Covers StableLM +
  Phi-1/2 partial rotary, and Phi-3 su/longrope's partial application if configured.
- **SiLU-SwiGLU** `kSiluAndMul` (`ops.h:91`) — Phi-3/4, Granite, StableLM, MiniCPM,
  InternLM2 MLP.
- **`kMulScalar`** (bf16 scalar multiply, `ops.h:201`, landed for Gemma embed-scale) —
  the Granite/MiniCPM scalar multipliers ride this + folded scalars.
- **MLA / latent-KV attention** (`DeepseekV2*` block, q_lora/kv_lora, decoupled rope,
  head_dim-256 prefill; landed W6 for DeepSeek-V2-Lite + GLM-4.7-Flash) — MiniCPM3.
- **Interleaved sliding-window / local+global attention** (FA-2 finite window +
  `SlidingWindowSpec`/`ChunkedLocalAttentionSpec` + per-layer `layer_types` routing,
  landed for Gemma-3 / OLMo-3) — Command-R (Cohere2/v2) + OLMo-3.
- **Merged-column loader** (`qkv_proj`, `gate_up_proj` packed_modules_mapping),
  **tied-or-untied embeddings**, the **shape-agnostic runner**, the
  `REGISTER_VLLM_MODEL` seam, the decode-graph sibling pattern — all reused.
- **Tokenizers:** ByteLevel BPE (Isolated + Removed-inverted) covers Granite /
  StableLM / Command-R / Phi-2 / Phi-4; SentencePiece(partial) covers Phi-3 / MiniCPM /
  MiniCPM3 / InternLM2 (same `LOAD-SENTENCEPIECE` path landed for Mistral/Gemma). BOS
  handling is a per-family W0 verify (the OPT/Llama trap: a silently mis-placed BOS
  scores 0/n while emitting fluent text).

### 0.2 The genuinely-NEW delta per family (small, additive)

| # | Family | Row | Dense/MoE/hybrid | The specific NEW delta over LANDED | Class |
|---|---|---|---|---|---|
| 1 | **Phi-3 / Phi-4** | `phi3-phi3-for-causal-lm` | dense | `Phi3ForCausalLM` is a `LlamaForCausalLM` subclass (`phi3.py:10`) with pre-fused `qkv_proj`/`gate_up_proj` checkpoints; only delta is optional su/longrope rope-scaling on long-context variants (base 4k = plain NeoX). Reuses the DONE Llama forward VERBATIM. | **ZERO-NEW-KERNEL** |
| 2 | **Granite-3** | `granite-granite-for-causal-lm` | dense | Llama + FOUR scalar multipliers (`embedding_multiplier` `granite.py:313`, `residual_multiplier` `:240,245`, `attention_multiplier`->scaling `:137`, `logits_scaling` `:371-372`); tied lm_head. All default-1 scalars threaded like Gemma qpas/embed-scale. | **ZERO-NEW-KERNEL** |
| 3 | **StableLM** | `stablelm-stablelm-for-causal-lm` | dense | `nn.LayerNorm` (not RMS; REUSE `kLayerNorm`) + partial rotary (`partial_rotary_factor`; REUSE partial NeoX) + optional `use_qkv_bias` (REUSE biased qkv) + SiLU-SwiGLU; standard pre-norm (separate input/post norms `stablelm.py:190-191`). | **ZERO-NEW-KERNEL** |
| 4 | **MiniCPM** | `minicpm-mini-cpmfor-causal-lm` | dense (MoE variant gated off) | Llama + `scale_emb` embedding scale (`minicpm.py:443`), `scale_depth/sqrt(num_layers)` residual scaling (`:384-385,392-393`), `dim_model_base` logit scaling. All scalars. Tied. (The `MiniCPMMoE` class `:82` is config-gated; the dense checkpoint is scalar-only.) | **ZERO-NEW-KERNEL** |
| 5 | **InternLM2** | `internlm2-intern-lm2-for-causal-lm` | dense | RMSNorm+NeoX+SiLU pre-norm = Llama, BUT the fused `wqkv` packs q/k/v INTERLEAVED by kv-group (`split_qkv` `internlm2.py:158-188`) — a LOADER/split-layout delta only, NO compute kernel. `w1/w3 -> gate_up`. | **small-new-op** (loader layout) |
| 6 | **Command-R / Command-R7B** | `commandr-cohere-for-causal-lm` | dense (one file for Cohere+Cohere2) | `logit_scale` on the logits proc (`commandr.py:376`); Cohere `LayerNorm` (no learnable bias, mean-centred -> REUSE `kLayerNorm` null-bias, verify mean-subtract at W0); **parallel-residual** block (single input norm, attn+MLP both off the normed input, summed `:264-272`); Cohere2/v2 adds per-head QK-`LayerNorm` (`:200-213`) + interleaved sliding window (`:184-197`, REUSE Gemma-3) + rope-on-sliding/v1-only; tied embeds asserted (`:372`), no biases. | **small-new-op** (logit_scale scalar + parallel-residual wiring + per-head-LN qk-norm) |
| 7 | **Phi-1 / Phi-2** | `phi-phi-for-causal-lm` | dense | Parallel residual (attn + mlp + residual, single input `nn.LayerNorm` `phi.py:201`; REUSE `kLayerNorm`), partial rotary (REUSE), qkv **bias**=True (`:98`; REUSE), **plain non-gated GELU MLP** (`fc1`->NewGELU->`fc2` `:166-168`) — the ONE new op: a `kGelu`/NewGELU UNARY (we only have gated `kGeluAndMul`); lm_head **bias** + UNTIED (`:279,291`). | **small-new-op** (NewGELU unary) |
| 8 | **MiniCPM3** | `minicpm3-mini-cpm3-for-causal-lm` | dense + MLA | MiniCPM scalars (subclasses `MiniCPMDecoderLayer` `minicpm3.py:186`) + **MLA attention** (q_lora/kv_lora, `q_a_layernorm`/`kv_a_layernorm`, decoupled rope `:52-156`) — REWIRES the LANDED DeepSeek-V2 MLA block; no new kernel. | **small-new-op** (MLA re-wire, reuses campaign) |

### 0.3 Hardware fit + tokenizer + oracle-risk (HF API 2026-07-24, metadata only, nothing downloaded)

bf16 on disk ~= 2 x params. GB10: ~119 GiB unified pool, ~113 GiB free now — every
smallest genuine checkpoint fits with room for a build tree. Presence is NOT read from
this table (the OPT lesson): W0 verifies weight FILES on dgx before a row is picked up.

| Family | Smallest genuine checkpoint | Params | bf16 on disk | Gated? | Tokenizer + BOS note | Oracle-support risk (0.25.0) |
|---|---|---|---|---|---|---|
| **Phi-3 / Phi-4** | `microsoft/Phi-4-mini-instruct` | 3.836B | ~7.7 GiB | ungated | Phi-4 = ByteLevel BPE (o200k-ish); Phi-3-mini = SentencePiece (Llama, 32064); BOS varies (Phi-3 `<s>`; Phi-4 `<|endoftext|>`) — W0 verify. Both REUSE. | LOW — `Phi3ForCausalLM` well-established; bigger STRICT `microsoft/phi-4` 14.66B (~29.3 GiB) also `Phi3ForCausalLM` |
| **Granite-3** | `ibm-granite/granite-3.3-2b-instruct` | 2.53B | ~5.1 GiB | ungated | ByteLevel BPE (StarCoder/GPT-2 family, 49k); BOS W0-verify. REUSE. | LOW — Granite well-established |
| **StableLM** | `stabilityai/stablelm-2-1_6b` | 1.64B | ~3.3 GiB | ungated | Arcade100k GPT-NeoX ByteLevel BPE. REUSE. | LOW; `stablelm-3b-4e1t` 2.8B a bigger strict |
| **MiniCPM** | `openbmb/MiniCPM-2B-sft-bf16` | ~2.7B | ~5.4 GiB | ungated | Llama SentencePiece. REUSE. NOTE: `.bin` (no safetensors), needs `trust_remote_code`. | MEDIUM — needs `trust_remote_code=True`; W0 confirms the 0.25.0 oracle constructs it |
| **InternLM2** | `internlm/internlm2-chat-1_8b` | 1.89B | ~3.8 GiB | ungated | SentencePiece. REUSE. | LOW |
| **Command-R7B** | `CohereLabs/c4ai-command-r7b-12-2024` | 8.03B | ~16 GiB | **gated:auto** (HF click-through) | Cohere ByteLevel BPE; `<BOS_TOKEN>` prepended — W0 verify. REUSE. | LOW arch (`Cohere2ForCausalLM`); accept the click-through gate at W0. Command-R-v01 35B (~70 GiB) deprioritized |
| **Phi-1 / Phi-2** | `microsoft/phi-2` | 2.78B | ~5.6 GiB | ungated | GPT-2 ByteLevel BPE (codegen); no BOS. REUSE. | LOW; lower recency |
| **MiniCPM3** | `openbmb/MiniCPM3-4B` | ~4B | ~8 GiB | ungated | Llama SentencePiece. REUSE. `trust_remote_code`. | MEDIUM — MLA config needs `trust_remote_code`; W0 probes oracle construction |
| **OLMo-3** (landed row W5) | `allenai/OLMo-3-1025-7B` | ~7.3B | ~13.6 GiB | verify | ByteLevel BPE (no BOS, OLMo-2 family). REUSE. | **NONE** — 0.25.0 CONSTRUCTS `Olmo3Config` (verified in the OLMo-2 spike). Reuses landed Gemma-3 sliding window |

### 0.4 Oracle-support: names that are pin-REMOVED (no SACRED bar available)

These appear in requests but are in vLLM's `_PREVIOUSLY_SUPPORTED_MODELS`
(`registry.py:701-712`) — the pinned 0.25.0 oracle CANNOT construct them, so there is
no SACRED gate vehicle; flag any of them for a W0 probe and treat as DEP-blocked until
the oracle advances:
- `Phi3SmallForCausalLM` (removed 0.9.2), `Phi4FlashForCausalLM` (0.10.2),
  `Phi4MultimodalForCausalLM` (0.12.0), `InternLM2VEForCausalLM` (0.23.0),
  `MotifForCausalLM` (0.10.2).
- There is **no bare `Phi4ForCausalLM` text row** in the pin — Phi-4 *dense* uses the
  `Phi3ForCausalLM` architecture (verified: `microsoft/phi-4` and `Phi-4-mini-instruct`
  both report `architectures: ["Phi3ForCausalLM"]`). The registered `Phi4*` names are
  all multimodal (`Phi4ForCausalLMV` siglip `:518`, `Phi4MMForCausalLM` `:519`) — out
  of scope (vision towers not started).

### 0.5 RANKED implementation queue (recency x GB10-fit x additivity)

**Tier 1 — cheapest zero-new-kernel, do first (one impl agent each, OLMo-2-style):**

| Rank | Family / arch | Row | Why here | One-line W-order |
|---|---|---|---|---|
| 1 | **OLMo-3** (`Olmo3ForCausalLM`) | (landed olmo2 row, W5) | THE cheapest next win: config-gated sliding window on the ALREADY-LANDED `olmo2.cpp` class; 0.25.0 CONSTRUCTS `Olmo3Config` (verified); reuses the landed Gemma-3 interleaved sliding window verbatim | W5 only: enable `layer_types` sliding-window + full-attn rope-scaling on the olmo2 class -> SACRED on `OLMo-3-1025-7B` (verify presence at W0) |
| 2 | **Phi-3 / Phi-4** (`Phi3ForCausalLM`) | `phi3-phi3-for-causal-lm` | Most mainstream recent dense; a `LlamaForCausalLM` subclass -> reuses the DONE Llama forward VERBATIM; ungated Phi-4-mini fits; oracle-certain | W0 config/BOS/tokenizer(SP vs BPE) -> W1 registry+config (su/longrope flag) -> W2 loader (packed qkv/gate_up, tied Phi-4 / untied Phi-3) -> W3 forward (reuse Llama block) -> W4 SACRED on Phi-4-mini + STRICT on phi-4-14B |
| 3 | **Granite-3** (`GraniteForCausalLM`) | `granite-granite-for-causal-lm` | Recent (IBM 2025); ZERO new kernel (4 default-1 scalar multipliers); ungated 2.5B fits | W0 -> W1 registry+config (4 multipliers as default-1 scalars) -> W2 loader -> W3 forward (Llama block + threaded multipliers) -> W4 SACRED on granite-3.3-2b |
| 4 | **StableLM** (`StableLmForCausalLM`) | `stablelm-stablelm-for-causal-lm` | ZERO new kernel (LayerNorm dense + partial rope + optional qkv bias, all REUSE); ungated 1.6B fits | W0 -> W1 registry+config (partial_rotary_factor, use_qkv_bias, LayerNorm eps) -> W2 loader -> W3 forward (LayerNorm block, reuse partial rope + kLayerNorm) -> W4 SACRED on stablelm-2-1_6b |
| 5 | **MiniCPM** (`MiniCPMForCausalLM`) | `minicpm-mini-cpmfor-causal-lm` | ZERO new kernel (scale_emb / scale_depth / dim_model_base scalars); dense checkpoint; MEDIUM oracle-risk (trust_remote_code + `.bin`) | W0 (confirm oracle constructs it + `.bin` loader path) -> W1 config scalars -> W2 loader -> W3 forward (Llama block + 3 scalars) -> W4 SACRED on MiniCPM-2B |

**Tier 2 — small-new-op, one small thing each:**

| Rank | Family / arch | Row | New op / delta | One-line W-order |
|---|---|---|---|---|
| 6 | **InternLM2** (`InternLM2ForCausalLM`) | `internlm2-intern-lm2-for-causal-lm` | fused-`wqkv` interleaved kv-group split (loader only) | W0 -> W1 config -> W2 loader (the `split_qkv` interleave -> merged qkv) -> W3 forward (Llama block) -> W4 SACRED on internlm2-chat-1_8b |
| 7 | **Command-R7B** (`Cohere2ForCausalLM`) | `commandr-cohere-for-causal-lm` | logit_scale scalar + parallel-residual block + per-head-LN qk-norm + interleaved sliding (REUSE) | W0 (BOS + LN mean-subtract + accept gate) -> W1 config -> W2 loader (tied, no bias) -> W3 parallel-residual block + qk-LN + logit_scale + sliding routing -> W4 SACRED on command-r7b |
| 8 | **Phi-1 / Phi-2** (`PhiForCausalLM`) | `phi-phi-for-causal-lm` | ONE new op: `kGelu`/NewGELU UNARY (non-gated); parallel-residual + biased qkv/lm_head + untied | W0 -> W1 `kGelu` unary op (unit-gated) -> W2 config+loader (biased qkv, untied lm_head bias) -> W3 parallel-residual LayerNorm block + partial rope + plain-GELU MLP -> W4 SACRED on phi-2 |
| 9 | **MiniCPM3** (`MiniCPM3ForCausalLM`) | `minicpm3-mini-cpm3-for-causal-lm` | MLA re-wire (reuses the landed DeepSeek-V2 MLA block) + MiniCPM scalars | W0 (oracle-construct probe) -> W1 config (q_lora/kv_lora dims + scalars) -> W2 loader (MLA fused_qkv_a + scalars) -> W3 forward (DeepSeek-V2 MLA block + MiniCPM scalar wiring) -> W4 SACRED on MiniCPM3-4B |

**Tier 3 — campaigns (sequence after Tier 1-2), each needs its own leaf spike:**
Falcon-H1 (`FalconH1ForCausalLM`, Mamba2 SSM hybrid — new state kernel),
GraniteMoeHybrid (SSM), GraniteMoe / GraniteMoeShared / Cohere2Moe / PhiMoE (FusedMoE
+ family-specific router/expert structure — reuse the BF16 grouped-GEMM but a distinct
row each), Falcon (`FalconForCausalLM`, older; alibi + `new_decoder_architecture`
parallel branches).

### 0.6 Recommended TOP 3 to implement next

1. **OLMo-3** (W5 on the landed OLMo-2 row) — nearly-free; W-order = W5 only.
2. **Phi-3 / Phi-4** (`Phi3ForCausalLM`) — W0->W1->W2->W3(reuse Llama)->W4 SACRED (Phi-4-mini + strict phi-4-14B).
3. **Granite-3** (`GraniteForCausalLM`) — W0->W1(4 scalars)->W2->W3(Llama block+multipliers)->W4 SACRED (granite-3.3-2b).

**ZERO-NEW-KERNEL near-additive count:** 4 of the 8 newly-scoped rows (Phi-3/Phi-4,
Granite-3, StableLM, MiniCPM); plus OLMo-3 (landed row W5) and the InternLM3/Yi Llama
aliases are also zero-new-kernel -> **7 zero-new-kernel bring-ups available**.

---

## 1. Structured contract

### Scope

Design — not build — a RANKED, grounded implementation queue for the next tier of
recent dense (and small-MoE) TEXT families, factoring each honestly into what REUSES
the landed substrate vs the specific new delta, with per-family GB10 fit, tokenizer,
and 0.25.0-oracle-support risk. Advances the 8 rows listed at the top
`INVENTORIED` -> `SPIKE`.

In scope: the triage table (§0.2) + hardware/tokenizer/oracle table (§0.3); the
pin-removed / no-bare-Phi4 oracle facts (§0.4); the ranked queue with per-row W-orders
and the TOP 3 (§0.5-0.6); the OLMo-3 (landed-row W5) and InternLM3/Yi (Llama-alias)
call-outs. Every reuse claim is anchored to a landed `src/`/`include/` op (§0.1).

OUT of scope, each with a reason: **implementation of anything** (spike — no code, no
kernels, no build, no benchmark, nothing downloaded; only HF API metadata read).
**MoE/SSM/older families** (Falcon, Falcon-H1, GraniteMoe*, Cohere2Moe, PhiMoE) — each
a distinct campaign row, stay `INVENTORIED` (§0.5 Tier 3). **Pin-removed names**
(Phi3Small, Phi4Flash, Phi4Multimodal, InternLM2VE, Motif) and **Phi-4 multimodal**
(`Phi4ForCausalLMV`, `Phi4MMForCausalLM`) — no constructible 0.25.0 oracle / vision
towers not started (§0.4). **The embedding/reward/MM rows** for these families
(`MODEL-EMBED-phi3-*`, `MODEL-REWARD-internlm2-*`, `MODEL-MM-minicpmv-*`,
`MODEL-MM-cohere*`, `MODEL-MM-granite*`, `MODEL-MM-phi*`) — separate modalities.

### Upstream chain

Registry (`vllm/model_executor/models/registry.py` @ `e24d1b24`): Phi-3 `:185`
(`phi3.py::Phi3ForCausalLM`, a `LlamaForCausalLM` subclass); Phi-1/2 `:184`
(`phi.py::PhiForCausalLM`); Cohere/Cohere2 `:84-85` (`commandr.py::CohereForCausalLM`);
Granite `:121` (`granite.py::GraniteForCausalLM`); StableLM/Epoch `:200-201`
(`stablelm.py::StablelmForCausalLM`); InternLM2 `:133` (`internlm2.py::InternLM2ForCausalLM`);
MiniCPM `:151` (`minicpm.py::MiniCPMForCausalLM`); MiniCPM3 `:152`
(`minicpm3.py::MiniCPM3ForCausalLM`); InternLM3 `:134` -> `llama.py::LlamaForCausalLM`
(alias). Removed: `registry.py:701-712`.

Model-layer anchors: `phi3.py:10-18` (subclass + packed map); `phi.py:77-320`
(parallel-residual `:201`, plain GELU `:166-168`, biased qkv/dense `:98,102`, untied
lm_head bias `:279,291`); `commandr.py:76-417` (Cohere LayerNorm `:76-87`, qk-norm
`:200-213`, sliding `:184-197`, parallel-residual `:264-272`, logit_scale `:376`, tie
`:372`); `granite.py:64-414` (multipliers `:137,240,245,313,371-372`, tie `:411-414`);
`stablelm.py:60-277` (partial rotary + `use_qkv_bias` `:124`, LayerNorm `:190-191,238`,
gate_up `:71-89`); `internlm2.py:53-327` (`wqkv`/`split_qkv` `:126-189`, `w1/w3->gate_up`
`:254-255`); `minicpm.py:82-576` (scale_emb `:443`, scale_depth `:384-393`, MoE class
`:82`, tie); `minicpm3.py:52-224` (MLA `:52-156`, subclasses MiniCPM `:186-224`).
Shared layers: `layernorm.py::RMSNorm`, `activation.py::SiluAndMul` /
`get_act_fn("gelu_new")`, `rotary_embedding/get_rope` (partial + su/longrope),
`attention/Attention` (`per_layer_sliding_window`, `logits_soft_cap` unused here),
`logits_processor.py::LogitsProcessor(scale=...)`.

### Our baseline

REUSED as-is (anchor + why) — see §0.1 for the full list: `vt::RmsNorm`
(`recipes.h`), `vt::LayerNorm` (`ops.h:181,1215`, landed for OPT), partial NeoX RoPE
(`ops.h:1243,1249,1310`, landed for GLM-4), `kSiluAndMul` (`ops.h:91`), `kMulScalar`
(`ops.h:201`, landed for Gemma), the DeepSeek-V2 MLA block (landed W6), the Gemma-3 /
OLMo-3 interleaved sliding-window + KV specs, the merged-column loader, tied/untied
embeddings, the `ByteLevel` + `LOAD-SENTENCEPIECE` tokenizer paths, the shape-agnostic
runner, `REGISTER_VLLM_MODEL`, the decode-graph sibling.

Honestly NOT reusable, and why (the per-family new delta): a plain **non-gated
NewGELU unary** for Phi-1/2 (we only have gated `kGeluAndMul`) — the ONE new compute op
in the batch; the **`wqkv` interleaved kv-group split** loader layout for InternLM2; the
**parallel-residual block** wiring + `logit_scale` scalar for Command-R; the MiniCPM3
**MLA re-wire** (reuses the landed block, new layer composition). Per the OPT/GLM/Gemma/
OLMo-2 precedent (their D2/D4), each family gets a NEW `<family>.{h,cpp}` block that
reuses only the glue + attention path — it does NOT extend `dense_attn_block.h::AttnBlock`
(which hard-codes pre-norm + Qwen per-head qk-norm). **No MODEL row is `DONE`** anywhere;
nothing here claims otherwise.

Precedent specs: [`sweep-olmo2.md`](sweep-olmo2.md), [`sweep-gemma.md`](sweep-gemma.md),
[`sweep-llama-3.2.md`](sweep-llama-3.2.md) (the su/longrope + Llama-subclass baseline),
[`sweep-opt-125m.md`](sweep-opt-125m.md) (LayerNorm + BOS trap),
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (MLA re-wire for MiniCPM3).

**Anchor-drift warning.** Re-anchor every cited `file:line` against the tree at
implementation time; anchors are against pin `e24d1b24` / base `5c00fc4`.

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:185` `Phi3ForCausalLM` (`phi3.py`, Llama subclass) | **NEW (impl)** `phi3.{h,cpp}`/`_weights`/`_registry` reusing the Llama forward VERBATIM + optional su/longrope; ZERO new kernel |
| `registry.py:121` `GraniteForCausalLM` (`granite.py`) | **NEW (impl)** `granite.{h,cpp}` — Llama block + 4 default-1 scalar multipliers threaded (REUSE `kMulScalar`/folded scalars); ZERO new kernel |
| `registry.py:200-201` `StablelmForCausalLM` (`stablelm.py`) | **NEW (impl)** `stablelm.{h,cpp}` — LayerNorm dense (REUSE `kLayerNorm`) + partial rope (REUSE) + optional qkv bias; ZERO new kernel |
| `registry.py:151` `MiniCPMForCausalLM` (`minicpm.py`) | **NEW (impl)** `minicpm.{h,cpp}` — Llama block + scale_emb/scale_depth/dim_model_base scalars; ZERO new kernel |
| `registry.py:133` `InternLM2ForCausalLM` (`internlm2.py`) | **NEW (impl)** `internlm2.{h,cpp}` — Llama block; loader does the `wqkv` interleaved kv-group split -> merged qkv |
| `registry.py:84-85` `Cohere/Cohere2ForCausalLM` (`commandr.py`) | **NEW (impl)** `commandr.{h,cpp}` — parallel-residual block + Cohere LayerNorm (REUSE null-bias) + per-head qk-LN + `logit_scale` + interleaved sliding (REUSE Gemma-3) |
| `registry.py:184` `PhiForCausalLM` (`phi.py`) | **NEW (impl)** `phi.{h,cpp}` + ONE new op `kGelu` (NewGELU unary) — parallel-residual LayerNorm block + partial rope + biased qkv/untied-bias lm_head |
| `registry.py:152` `MiniCPM3ForCausalLM` (`minicpm3.py`) | **NEW (impl)** `minicpm3.{h,cpp}` — REWIRE the landed DeepSeek-V2 MLA block + MiniCPM scalars |
| `registry.py:134` `InternLM3ForCausalLM` -> `llama` | **REUSE** — a registry-alias string on the ACTIVE Llama row (not a new arch) |
| Falcon / Falcon-H1 / GraniteMoe* / Cohere2Moe / PhiMoE / removed names | **NOT PORTED** — out of scope (§0.4-0.5 Tier 3); stay `INVENTORIED` |

### Tests to port

Per [`.agents/test-porting.md`](../test-porting.md). Nothing is ported by THIS spike
(design only); this is the inventory that binds the implementing Ws.

| Upstream test | Tier | Ours (at impl time) |
|---|---|---|
| `tests/models/language/generation/test_common.py` (Phi-3, Granite, StableLM, InternLM2, MiniCPM, Command-R text-gen entries) | T-parity | `tests/parity/test_<family>_paged_engine.cpp` — the per-family SACRED token-exact gate (W4) |
| `tests/models/language/generation/test_granite.py` | T-parity | Granite text-gen correctness (W4) |
| `tests/models/registry.py` `_HfExamplesInfo` for each arch (+ pin-removed check) | T-unit | config/registry resolution + the removed-name loud-fail cases (W1) |
| `tests/models/test_initialization.py` (init smoke) | T-unit | init/registry resolution; MiniCPM/MiniCPM3 gate the trust_remote_code oracle-construct verdict (W0) |
| `tests/test_config.py` (arch-config resolution) | T-unit | arch-config resolution, gateable with NO checkpoint |
| `tests/v1/e2e/general/test_correctness_sliding_window.py` | T-e2e | Command-R7B (Cohere2) + OLMo-3 interleaved sliding window (W3/W5) |

### Gates (bind the implementing Ws; NONE run in this spike)

1. **Correctness (SACRED), per family** — token-exact vs the pinned vLLM 0.25.0
   oracle, greedy, identical prompts, gate form selected BY MEASUREMENT per
   [`near-tie-distributional-gate`](../gates.md) (run vLLM's own K=5 greedy first;
   distributional/near-tie-band fallback ONLY if measured, else STRICT; add a bigger
   STRICT model where the small one is a near-tie, e.g. phi-4-14B for Phi-3).
2. **New-op unit gate** — only Phi-1/2 has one: `kGelu`/NewGELU UNARY bit-exact vs the
   vLLM `gelu_new` reference at real dims. The other rows add no `vt::` op; their new
   facts (scalars, parallel-residual, wqkv split, MLA re-wire) are proven by gate 1
   (a wrong scalar/placement/split emits fluent-WRONG tokens — the OPT mode).
3. **Loader** — zero unmapped, zero missing on the gated checkpoint (incl. the
   InternLM2 `wqkv` interleave asserted, Granite/MiniCPM multipliers loaded, Command-R
   tied-no-bias, Phi biased-untied, MiniCPM3 MLA fused_qkv_a).
4. **Regression, non-negotiable** — all current SACRED gates UNCHANGED (each new family
   is new-files-only + at most one additive default-inert op, so existing gates are
   byte-identical by construction).
5. **Build** — clean full rebuild `-Werror`, zero warnings.
6. **memcheck** — `compute-sanitizer` zero errors on the new forward (+ `kGelu` for Phi).
7. **Records** — all five CI checkers green.
8. **SPEED** — explicitly PENDING and unclaimed; a row is `DONE` only at token-exact
   AND vLLM throughput on every axis.

### Dependencies

**No hard upward dependency** for the ZERO-NEW-KERNEL rows (Phi-3/4, Granite, StableLM,
MiniCPM) and OLMo-3 (W5 on landed infra). Phi-1/2 depends on the small `kGelu` unary.
MiniCPM3 depends on the landed MLA block (present). **Oracle preconditions (W0 probes):**
MiniCPM / MiniCPM3 need `trust_remote_code`; Command-R7B needs the HF click-through gate
accepted; OLMo-3 needs its checkpoint present (`Olmo3Config` construction already
verified). **Checkpoint downloads (not performed):** the §0.3 smallest genuine
checkpoints; stage sequentially ([[grid-per-sha-trees-fill-disk]]). **Downward
dependencies introduced:** the `kGelu` NewGELU unary (reusable by any non-gated-GELU
model, e.g. GPT-2/GPT-NeoX/Falcon later).

### Work breakdown (per-row W-orders are in §0.5; this spike delivers only the scoping)

- **This spike (DONE):** the triage + ranked queue + records; advance 8 rows
  `INVENTORIED` -> `SPIKE`; register `CLAIM-SWEEP-RECENT-DENSE`.
- **Next (separate impl claims, one agent each):** rank-1 OLMo-3 W5, rank-2 Phi-3/Phi-4,
  rank-3 Granite-3 — then StableLM, MiniCPM, InternLM2, Command-R7B, Phi-1/2, MiniCPM3,
  each per its §0.5 W-order and the gates above.

### Risks/decisions

**D1 — the queue is ordered by (recency x fit x additivity); OLMo-3 leads because it is
nearly-free on a LANDED row.** Phi-3/Granite lead the NEW rows because they reuse the
DONE Llama forward with zero (Phi-3) or scalar-only (Granite) deltas.

**D2 — do NOT extend `dense_attn_block.h::AttnBlock`.** Each family gets a new
`<family>.{h,cpp}` block reusing only the glue + attention path (the OPT/GLM/Gemma/OLMo-2
precedent). Command-R's parallel-residual and Phi's parallel-residual+plain-GELU are the
clearest cases that the shared header does not stretch.

**D3 — scalars/placement are silent-corruption hazards.** Granite's 4 multipliers,
MiniCPM's scale_emb/scale_depth/dim_model_base, Command-R's logit_scale, and the
parallel-residual re-join order all emit fluent-WRONG tokens if mis-wired (the OPT mode).
Mitigation: gate 1 is an e2e token-exact gate; each W3 grounds the exact scalar/residual
order line-by-line against the model file.

**D4 — BOS + tokenizer family is a per-row W0 verify.** SP vs ByteLevel varies within the
batch (even within Phi: Phi-3 SP, Phi-4 BPE). A silently mis-placed BOS scores 0/n while
emitting fluent text (OPT/Llama). W0 verifies BOS vs the oracle before any forward; the
gate can run tokenizer-free (feed the oracle's exact ids) if our loader does not validate
that family's tokenizer.json.

**D5 — oracle-support must be probed for MiniCPM/MiniCPM3 (trust_remote_code) and the
pin-removed names (§0.4).** A removed/unconstructible arch has NO SACRED bar; W0 records
the verdict and the row stays honestly blocked rather than claiming a gate it cannot run.

**D6 — MoE/SSM/older families stay `INVENTORIED` by design, not omission.** Falcon-H1 /
GraniteMoeHybrid are SSM campaigns; GraniteMoe / Cohere2Moe / PhiMoE add a family-specific
FusedMoE router/expert layout; Falcon is older with alibi + parallel `new_decoder_arch`
branches. Each is a distinct leaf spike (§0.5 Tier 3).
