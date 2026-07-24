# SPIKE: OLMo-2 family (`Olmo2ForCausalLM` / `Olmo3ForCausalLM`)

**SPIKE ONLY — no implementation, no kernels, no build, no benchmark, nothing
downloaded.** Grounds the breadth-sweep's next recent dense family after GLM-4 and
Gemma landed ([`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B.3 Tier-2 rank 8,
"GLM4 / Olmo2-3" — GLM-4 and the Gemma family are in; Olmo2 is next). Design +
records only.

**Base:** `a60b88b` (Gemma-2 + Gemma-1 W3-W6 landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` = vLLM
**0.25.0**. **Claim:** `CLAIM-SWEEP-OLMO2`. **Parent plan:**
[`breadth-sweep-plan.md`](breadth-sweep-plan.md) §B.2 ("`Olmo2/3` reuse the Qwen3
dense forward almost verbatim") + §B.3 Tier-2 rank 8. **Gold-standard spike shapes
mirrored:** [`sweep-gemma.md`](sweep-gemma.md),
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md).

Row advanced `INVENTORIED` -> `SPIKE`:
`MODEL-TEXT-olmo2-olmo2-for-causal-lm` (`Olmo2ForCausalLM`, `Olmo3ForCausalLM` — one
vLLM class, one row).

Explicitly LEFT `INVENTORIED` (out of scope, each with a reason in §Scope):
`MODEL-TEXT-olmo-olmo-for-causal-lm` (OLMo-1 — a DIFFERENT norm: non-parametric
`nn.LayerNorm`, not RMSNorm, and pre-norm placement),
`MODEL-TEXT-olmoe-olmoe-for-causal-lm` (OLMoE — MoE + pre-norm),
`MODEL-TEXT-flex-olmo-flex-olmo-for-causal-lm` (FlexOlmo — MoE + OLMo-2-style
post-norm), `MODEL-TEXT-olmo-hybrid-olmo-hybrid-for-causal-lm` (OlmoHybrid —
Mamba/GatedDeltaNet SSM hybrid, a separate kernel campaign).

---

## 0. Headline findings

### 0.0 OLMo-2 is the cleanest dense bring-up yet: ZERO new compute kernels

`olmo2.py` is **421 lines** and is a plain dense GQA decoder — SiLU SwiGLU MLP,
NeoX RoPE, no biases, tied-or-untied embeddings. Everything OLMo-2 computes we
ALREADY have as a kernel. The two things that make OLMo-2 *look* distinctive —
its **reordered post-norm placement** (`norm_after`) and its **QK-norm** — both
resolve to **wiring over existing ops**, not new kernels. OLMo-2 therefore needs
**no new `vt::` op at all** (contrast Gemma, which needed `kGeluAndMul` +
`kSoftCap`). It is a stricter additivity proof than Gemma: a genuinely-recent dense
family that lands as *new files only* over the landed dense substrate.

### 0.1 The norm-placement finding (the load-bearing analysis)

OLMo-2's `Olmo2DecoderLayer.forward` (`olmo2.py:261-277`) is **pure post-norm**:

```
residual = h
h = self_attn(h)                       # attention on the RAW residual stream — NO input norm
h = post_attention_layernorm(h)        # standalone RMSNorm on the ATTENTION OUTPUT
h = h + residual                       # plain residual add

residual = h
h = mlp(h)                             # MLP on the RAW residual stream — NO pre-FF norm
h = post_feedforward_layernorm(h)      # standalone RMSNorm on the MLP OUTPUT
h = residual + h                       # plain residual add
```

This is genuinely distinct from **all three** norm layouts already in the tree, and
the distinction must be gotten exactly right (a wrong placement emits fluent-WRONG
tokens, the OPT/GLM/Gemma silent-corruption mode):

| Layout | Where the norm sits | Landed example |
|---|---|---|
| **Pre-norm** | `h = h + Attn(input_norm(h))` — norm BEFORE the sublayer | Qwen3/Llama/Mistral (`dense_attn_block.h`) |
| **Sandwich** | `h = h + post_norm(Attn(input_norm(h)))` — norm BEFORE **and** after | GLM-4 (`glm4.cpp:158-188`), Gemma-2/3 (`gemma3.cpp`) |
| **Pure post-norm (`norm_after`)** | `h = h + post_norm(Attn(h))` — **NO pre-norm**, norm ONLY on the sublayer output | **OLMo-2 (NEW wiring)** |

**The crux:** OLMo-2's `norm_after` is a strict SUBSET of the GLM-4/Gemma sandwich —
it keeps only the post (output) norms and DROPS the input/pre norms entirely. So the
underlying PRIMITIVE the task asked about — "a standalone RMSNorm on a sublayer's
output before the residual add" — is EXACTLY the landed GLM-4
`post_self_attn_layernorm`/`post_mlp_layernorm` op (`glm4.cpp:174-178,186-188`),
byte-for-byte reusable. **What is new is the layer wiring, not the op:** OLMo-2 omits
the two pre-norms and its residual re-join is a *plain add* (`h + residual`), whereas
GLM-4/Gemma fold the add into the NEXT pre-norm's `fused_add_rms_norm`. There is no
`input_layernorm`/`pre_feedforward_layernorm` at all. So the post-norm infra COVERS
OLMo-2's placement at the op level; the delta is a new `olmo2.{h,cpp}` block that
composes those standalone norms in the post-norm order with plain residual adds.

### 0.2 The QK-norm finding — OLMo-2's is FULL-WIDTH, not per-head (cannot reuse the fused recipe)

OLMo-2 has QK-norm (RMSNorm on q and k before RoPE, `olmo2.py:160-182`), which the
task flagged as possibly-already-ours via `kAttnQkNormRopeGate`. **It is NOT the same
shape.** OLMo-2 norms the ENTIRE q and k projection outputs across all heads at once:

```python
self.q_norm = RMSNorm(config.hidden_size, eps=...)                       # = num_heads * head_dim  (:117)
self.k_norm = RMSNorm(total_num_kv_heads * head_dim, eps=...)            # = kv_size               (:113-116)
# forward: q,k = qkv.split(...); q = q_norm(q); k = k_norm(k); q,k = rope(q,k)   (:179-182)
```

vLLM's `QKNormRoPEFusionPass` / our `kAttnQkNormRopeGate` (`recipes.h:231-244`)
apply a **per-head** RMSNorm over `head_dim` (the Qwen3/Gemma recipe — its operands
are `q_norm[Dh]`, `k_norm[Dh]`). OLMo-2 applies **one RMSNorm over the whole
`q_size`/`kv_size` vector** (all heads folded into the normalization statistic). That
is a *different* normalization and CANNOT use the fused per-head recipe. But it IS a
plain `vt::RmsNorm` over a 2-D `[T, q_size]` / `[T, kv_size]` tensor — an op we have.
So the QK-norm REUSES the `vt::RmsNorm` op at a new (full-width) shape, applied as two
standalone norms BEFORE a standard `RopeNeox`, with NO fusion. This mirrors how the
GLM-4.7-Flash spike found its "new primitives" reduced to existing ops at a different
wiring — same lesson, different family.

Also note: OLMo-2 uses **plain RMSNorm** (not the Gemma `(1+w)` variant) at every
norm — `RmsNormArgs{eps, gemma=false}`, exactly the form GLM-4 uses. REUSE.

### 0.3 What reduces to landed infrastructure (REUSE) vs genuinely NEW

Verified in `src/`/`include/` on base `a60b88b`:

**REUSE (all landed):**
- **Plain RMSNorm** (`gemma=false`) — `vt::RmsNorm` (`recipes.h`, `ops.h`). Every
  OLMo-2 norm. Same op GLM-4 uses.
- **Standalone RMSNorm on the sublayer OUTPUT before the residual add** — the GLM-4
  `post_self_attn_layernorm`/`post_mlp_layernorm` pattern (`glm4.cpp:174-178,186-188`).
  OLMo-2's `post_attention_layernorm`/`post_feedforward_layernorm` are the SAME op —
  cross-family reuse (the exact primitive the task flagged).
- **SiLU SwiGLU MLP** — `kSiluAndMul` (`ops.h:91`), fused gate_up + SiLU + down.
  OLMo-2 `Olmo2MLP` (`olmo2.py:188-230`) is identical to Qwen3/Llama's.
- **NeoX RoPE** — `RopeNeox`/`RopeFromCache`. OLMo-2 uses `get_rope(... default)`
  (`olmo2.py:145-149`); Olmo-3 adds rope-scaling on full-attention layers only.
- **GQA paged attention + dense device glue** — `dense_attn_block.h` glue
  (`Dev`/`DBuf`/`ResidentWeight`/`KvSlice`/`BuildStepInputs` + the GQA paged forward),
  the `REGISTER_VLLM_MODEL` model-factory seam, and the `ENG-RUNNER-MODELSHAPE`
  model-shape-agnostic runner (full-attention-only KV, no hybrid path).
- **Interleaved sliding-window / local+global attention (Olmo-3 only)** — the FA-2
  finite window + `SlidingWindowSpec`/`ChunkedLocalAttentionSpec` + per-layer routing
  by `layer_types[i] == "sliding_attention"`, ALL landed for Gemma-3. Olmo-3's
  `layer_types` gate (`olmo2.py:122-126`) + full-attn-only rope-scaling
  (`olmo2.py:139-149`) is the SAME wiring Gemma-3 uses. REUSE.
- **Tied-or-untied embeddings** — the `tie_word_embeddings` skip-`lm_head` loader path
  (opt/qwen3_dense/glm4/gemma). OLMo-2 ties conditionally (`olmo2.py:374-382,417`).
- **Packed loader mapping** — `qkv_proj <- q/k/v_proj`, `gate_up_proj <- gate/up_proj`
  (`olmo2.py:351-364`), identical to Qwen3/Gemma `packed_modules_mapping`. REUSE the
  merged-column loader.
- **ByteLevel BPE tokenizer** — OLMo-2's tokenizer is GPT-NeoX-derived byte-level BPE
  (`tokenizer.json`; the file header cites "EleutherAI's GPT-NeoX", `olmo2.py:9`), the
  SAME family as Qwen/Llama-3/OPT/GPT-2 (`tokenizer.cpp` ByteLevel path). NOT
  SentencePiece — so it does NOT hit the Mistral/Gemma `LOAD-SENTENCEPIECE` blocker.
  W0 verifies BOS handling directly (the OPT lesson: OLMo tokenizers commonly do NOT
  prepend a BOS, using `<|endoftext|>` as eos/pad — a silently-wrong BOS scores 0/n
  while emitting fluent text).
- **The decode CUDA-graph driver pattern** (the qwen3-dense/glm4 bf16 dense sibling)
  for the W5 speed close.

**GENUINELY NEW — and it is TWO wiring facts, ZERO kernels:**
1. **Pure post-norm (`norm_after`) block placement** (§0.1) — a new
   `olmo2.{h,cpp}` block that applies standalone `vt::RmsNorm` to each sublayer's
   OUTPUT and re-joins the residual with a plain add, with NO pre-attention/pre-FF
   norm. Reuses the standalone-norm op + `vt::Add`/residual; adds no kernel.
2. **Full-width QK-norm** (§0.2) — two standalone `vt::RmsNorm` over `[T, q_size]` /
   `[T, kv_size]` before a standard `RopeNeox` (cannot use the fused per-head
   `kAttnQkNormRopeGate`). Reuses the `vt::RmsNorm` op at a new shape; adds no kernel.

### 0.4 The recent-first gate vehicle that FITS + is oracle-certain: **OLMo-2-1B**

`allenai/OLMo-2-0425-1B` (1.485B, `Olmo2ForCausalLM`) is a clean pure-text dense
checkpoint, ~2.77 GiB bf16 — small enough to fit GB10's TIGHT ~30 GiB free alongside
a build tree, and certain to run on the 0.25.0 oracle (`Olmo2Config` is well
established in transformers, unlike the very new Olmo3/Gemma-4). This is the **primary
gate vehicle**, exactly the pattern GLM (HW-blocked flagship, gate vehicle = the one
that fit) and Gemma (Gemma-4 leads, Gemma-3-1b gates) used.

### 0.5 The OLMo registry set (per-arch characterization)

Six OLMo-family architectures at the pin (`registry.py`):

| Arch | Registry | File | Norm placement | QK-norm | Dense/MoE | This spike |
|---|---|---|---|---|---|---|
| **`Olmo2ForCausalLM`** | `:172` | `olmo2.py` | **pure post-norm (`norm_after`)** | **full-width** | dense | **SPIKE (gate vehicle)** |
| **`Olmo3ForCausalLM`** | `:173` | `olmo2.py` (SAME class) | pure post-norm | full-width | dense + interleaved sliding window | **SPIKE (same row, config-driven)** |
| `OlmoForCausalLM` (OLMo-1) | `:171` | `olmo.py` | pre-norm, **non-parametric `nn.LayerNorm`** (mean-centering, no weight, `olmo.py`) | none | dense | INVENTORIED (different norm) |
| `OlmoeForCausalLM` (OLMoE) | `:175` | `olmoe.py` | **pre-norm** (fused add+RMSNorm) | full-width (`olmoe.py:166-167`) | **MoE** (`FusedMoE`) | INVENTORIED (MoE) |
| `FlexOlmoForCausalLM` | `:104` | `flex_olmo.py` | OLMo-2-style post-norm (`post_attention`+`post_feedforward`) | full-width (extends `OlmoeAttention`) | **MoE** | INVENTORIED (MoE) |
| `OlmoHybridForCausalLM` | `:174` | `olmo_hybrid.py` | — | — | **Mamba/GatedDeltaNet SSM hybrid** | INVENTORIED (SSM campaign) |

The key inventory fact: **`Olmo3ForCausalLM` maps to the SAME `olmo2.py`
`Olmo2ForCausalLM` class** (`registry.py:173`), gated on `Olmo3Config` vs
`Olmo2Config`. Olmo-3 = Olmo-2 + interleaved sliding-window (`layer_types`) + rope
scaling on full-attention layers only — the Gemma-3 sliding pattern. Both are one row.
OLMo-1 is genuinely different (non-parametric LayerNorm — mean-subtracting, NOT
RMSNorm — and no QK-norm); OLMoE/FlexOlmo add the FusedMoE path (landed for the 35B /
Qwen3-Coder, so additive later); OlmoHybrid is an SSM campaign. All four stay
`INVENTORIED`, characterized here.

### 0.6 Hardware fit — measured per variant

Params from the HF API 2026-07-24 (`safetensors.total`; metadata only, **nothing
downloaded**). bf16 on-disk ≈ 2 x params. GB10 budget: ~119 GiB unified memory, but
the sweep gate vehicle must fit the **TIGHT ~30 GiB free** alongside a build tree
(unified-memory OOM reboots the box — [[gb10-unified-memory-oom-reboots-box]]).

| Variant | Checkpoint | Params | bf16 on disk | Arch on the checkpoint | GB10 verdict |
|---|---|---|---|---|---|
| `Olmo2ForCausalLM` (dense) | **`allenai/OLMo-2-0425-1B`** | 1.485B | **~2.77 GiB** | `Olmo2ForCausalLM` (pure text) | **FITS — recommended PRIMARY gate vehicle** |
| `Olmo2ForCausalLM` (bigger strict) | `allenai/OLMo-2-1124-7B` | 7.30B | ~13.6 GiB | `Olmo2ForCausalLM` (pure text) | **FITS the tight budget — bigger-model strict check (only if 1B is a near-tie)** |
| `Olmo2ForCausalLM` | `allenai/OLMo-2-1124-13B` | 13.72B | ~25.5 GiB | `Olmo2ForCausalLM` (pure text) | **FITS full budget, TIGHT vs ~30 GiB free — not needed if 7B suffices** |
| `Olmo2ForCausalLM` | `allenai/OLMo-2-0325-32B` / `-32B-Instruct` | 32.23B | ~60 GiB | `Olmo2ForCausalLM` (pure text) | **HW-MARGINAL** vs the ~30 GiB free-alongside-build budget; fits the full 119 GiB pool alone but not the sweep vehicle |
| `Olmo3ForCausalLM` (sliding window) | `allenai/OLMo-3-*` (verify presence + 0.25.0 oracle-support at W0) | — | — | `Olmo3ForCausalLM` (pure text) | **oracle-support UNVERIFIED** (Olmo3 is newer than the 0.25.0 pin's transformers; if the oracle can't construct it there is no SACRED bar — W5 reopens) |

**No proposed gate depends on hardware we do not have.** The primary vehicle
(`OLMo-2-0425-1B`, ~2.77 GiB) fits any budget. Presence is NOT read from a table (the
OPT lesson): W0 verifies weight FILES on dgx before the row is picked up.

### 0.7 Recommended W-order

```
  W0  Ground facts on HW: verify the 0.25.0 oracle constructs Olmo2ForCausalLM and runs
      it; fetch OLMo-2-0425-1B; confirm live config (post-norm, full-width q/k norm dims,
      tie_word_embeddings, rope_theta); run oracle + K=5 greedy self-determinism (selects
      the gate form); VERIFY BOS placement vs the oracle (the OPT lesson — OLMo tokenizers
      often prepend NO BOS); probe whether the oracle's transformers can construct Olmo3Config.
      No code.
        |
  W1  Registry stub + config parse: new TU `olmo2_registry.cpp` + `olmo2.h`, one
      REGISTER_VLLM_MODEL, full-attention-only KV spec, is_dense_model=true, W2/W3 stubs.
      Gate: registry resolves Olmo2ForCausalLM (+ Olmo3 alias); regressions UNCHANGED.
        |
  W2  Weight loader (`olmo2_weights.cpp`): merged qkv/gate_up (packed_modules_mapping),
      the FULL-WIDTH q_norm[hidden]/k_norm[kv_size] weights, the two post-norms/layer,
      final norm, tied-or-untied lm_head. Gate: zero unmapped, zero missing on the 1B ckpt.
        |
  W3  Forward — new `olmo2.{h,cpp}` block: the PURE POST-NORM placement (standalone
      vt::RmsNorm on each sublayer output + plain residual add, NO pre-norm) + FULL-WIDTH
      qk-norm (two standalone vt::RmsNorm before RopeNeox) + SiLU SwiGLU + GQA paged.
      Reuses only the dense_attn_block.h glue + GQA paged path (per D2). ZERO new kernel.
        |
  W4  SACRED gate on OLMo-2-0425-1B (form per W0) + bigger strict on OLMo-2-7B if 1B is a
      near-tie. This is the primitive-proof gate (a wrong norm placement or qk-norm shape
      emits fluent-WRONG tokens — the OPT failure mode).
        |
  W5  Olmo-3 sliding-window variant (IF the oracle constructs Olmo3Config AND a checkpoint
      fits): reuse the Gemma-3 interleaved sliding-window + full-attn rope-scaling wiring.
      Otherwise an honesty pass recording the oracle-support verdict.
        |
  W6  Speed close: decode-graph sibling + the every-axis grid vs vLLM. Nothing reaches DONE
      before this (correctness-complete + vLLM-parity on every axis is the DONE bar).
```

W3 is the highest-value item and the first token-exact gate. There is no independent
W1-style new-ops step (unlike Gemma) because OLMo-2 adds no op — the two "new" facts
(§0.3) are both realized inside the W3 block wiring.

---

## 1. Structured contract

### Scope

Design — not build — `Olmo2ForCausalLM` (and its `Olmo3ForCausalLM` alias) end to
end, determine the recent-first gate vehicle that runs on GB10 against the pinned
oracle, and factor the work honestly into REUSE vs genuinely-NEW. Advances
`MODEL-TEXT-olmo2-olmo2-for-causal-lm` `INVENTORIED` -> `SPIKE`.

In scope: the complete OLMo registry inventory with per-arch characterization; the
reuse-vs-new factoring (plain RMSNorm, the GLM-4 standalone post-norm op, SiLU SwiGLU,
NeoX rope, GQA paged glue, sliding-window (Olmo-3), tied embeddings, packed loader,
ByteLevel BPE all REUSE; the pure post-norm placement + the full-width QK-norm are the
two NEW wiring facts, ZERO new kernels); the norm-placement analysis (pure post-norm
`norm_after` is a strict subset of the landed GLM-4/Gemma sandwich, reusing its
standalone-output-norm op); the QK-norm shape analysis (full-width, not per-head ->
cannot use `kAttnQkNormRopeGate`); measured per-variant GB10 hardware fit; the SACRED
gate design (form BY MEASUREMENT); and the upstream test inventory.

OUT of scope, each with a reason: **implementation of anything** (this is a spike —
no code, no kernels, no build, no benchmark, nothing downloaded).
**`OlmoForCausalLM` (OLMo-1)**, because it uses a non-parametric `nn.LayerNorm`
(mean-centering, no weight) and pre-norm placement — a different norm than RMSNorm
that no OLMo-2 machinery covers; it stays `INVENTORIED`. **`OlmoeForCausalLM` /
`FlexOlmoForCausalLM`**, because they add the `FusedMoE` grouped-GEMM path (landed for
the 35B / Qwen3-Coder, so additive later, but a distinct row); they stay `INVENTORIED`.
**`OlmoHybridForCausalLM`**, because it is a Mamba/GatedDeltaNet SSM hybrid — a
separate state-kernel campaign; it stays `INVENTORIED`. **Olmo-3's e2e**, gated behind
a W0/W5 oracle-support probe (Olmo3Config may be too new for the 0.25.0 oracle); the
`Olmo2ForCausalLM` row covers it structurally when a checkpoint + oracle exist.

### Upstream chain

Registry (`vllm/model_executor/models/registry.py` @ `e24d1b24`): `:171`
`OlmoForCausalLM` (`olmo.py`); `:172` `Olmo2ForCausalLM` (`olmo2.py`); `:173`
`Olmo3ForCausalLM` -> `olmo2.py::Olmo2ForCausalLM` (SAME class); `:174`
`OlmoHybridForCausalLM` (`olmo_hybrid.py`); `:175` `OlmoeForCausalLM` (`olmoe.py`);
`:104` `FlexOlmoForCausalLM` (`flex_olmo.py`).

Model layer (`olmo2.py` @ `e24d1b24`): `Olmo2Attention` `:67-185` (full-width q/k norm
ctor `:113-117`, `_apply_qk_norm` full-width `:160-172`, qk-norm-then-rope forward
`:179-182`, Olmo-3 sliding-window/`layer_types` gate `:121-126`, full-attn-only rope
scaling `:139-149`, scaling `head_dim**-0.5` `:119`, no biases); `Olmo2MLP` `:188-230`
(SiLU SwiGLU, merged gate_up); `Olmo2DecoderLayer` `:233-277` (the **pure post-norm**
forward `:261-277`, two standalone `post_attention_layernorm`/
`post_feedforward_layernorm` `:253-259`, NO pre-norm); `Olmo2Model` `:280-343` (final
`norm` `:297-300`); `Olmo2ForCausalLM` `:346-421` (`hf_to_vllm_mapper`/
`packed_modules_mapping` `:351-364`, tie `:374-382,417`, `AutoWeightsLoader`
`:413-420`). Norm: `vllm/model_executor/layers/layernorm.py::RMSNorm` (plain, NOT the
Gemma `(1+w)` subclass). Activation: `activation.py::SiluAndMul`. RoPE:
`rotary_embedding/get_rope` (NeoX default; Olmo-3 rope scaling on full-attn layers).
Attention + sliding window: `attention/Attention` (`per_layer_sliding_window`).

### Our baseline

REUSED as-is (each with the anchor and why):
- **Plain RMSNorm** (`gemma=false`) — `vt::RmsNorm` (`include/vt/recipes.h`,
  `include/vt/ops.h`); the same call GLM-4 uses. Every OLMo-2 norm.
- **Standalone RMSNorm on the sublayer OUTPUT before the residual add** —
  `src/vllm/model_executor/models/glm4.cpp:174-178,186-188`
  (`post_self_attn_layernorm`/`post_mlp_layernorm`). OLMo-2's two post-norms are the
  SAME op (the primitive the task flagged). Cross-family reuse GLM -> OLMo-2.
- **SiLU SwiGLU MLP** — `kSiluAndMul` (`include/vt/ops.h:91`). OLMo-2 `Olmo2MLP` is
  identical to Qwen3/Llama's fused gate_up + SiLU + down.
- **NeoX RoPE** — `RopeNeox`/`RopeFromCache` (used by qwen3/llama/glm4).
- **Interleaved sliding-window / local attention (Olmo-3)** — FA-2 finite window
  `src/vt/cuda/cuda_flash_attn_fa2.cu:397-407`; layer plumbing
  `src/vllm/model_executor/layers/attention/attention.cpp:13-23`
  (`per_layer_sliding_window`); KV specs `SlidingWindowSpec`/`ChunkedLocalAttentionSpec`
  (`src/vllm/v1/kv_cache_interface.cpp:76-95`,
  `src/vllm/v1/kv_cache_spec_registry.cpp:35-38`); per-layer routing by `layer_types`
  — ALL landed for Gemma-3 (`gemma3.cpp`).
- **GQA paged attention + dense glue** —
  `include/vllm/model_executor/models/dense_attn_block.h` (`Dev`/`DBuf`/
  `ResidentWeight`/`KvSlice`/`BuildStepInputs` + the GQA paged forward), the
  `REGISTER_VLLM_MODEL` model-factory seam, and the `ENG-RUNNER-MODELSHAPE`
  model-shape-agnostic runner.
- **Tied-or-untied embeddings** — the `tie_word_embeddings` skip-`lm_head` path on
  opt/qwen3_dense/glm4/gemma loaders.
- **Packed loader mapping** — the merged-column loader (`qkv_proj`, `gate_up_proj`)
  used by qwen3/gemma.
- **ByteLevel BPE tokenizer** — `src/vllm/tokenizer/tokenizer.cpp` ByteLevel path
  (Qwen/Llama-3/OPT/GPT-2 family; OLMo-2's GPT-NeoX BPE is the same family). W0
  verifies BOS.
- **The decode CUDA-graph driver pattern** (qwen3-dense/glm4 bf16 dense sibling) for
  W5.

Honestly NOT reusable, and why:
- **`kAttnQkNormRopeGate` (the fused per-head QK-norm+rope recipe)** — its RMSNorm is
  over `head_dim` per head (`recipes.h:231-244`, operands `q_norm[Dh]`/`k_norm[Dh]`);
  OLMo-2's is FULL-WIDTH over `q_size`/`kv_size`. The fused recipe cannot express it.
  OLMo-2 uses two STANDALONE `vt::RmsNorm` (reused op, new shape) before a standard
  `RopeNeox` — no fusion.
- **`dense_attn_block.h::AttnBlock` body** — it hard-codes PRE-norm placement and
  Qwen-style per-head qk-norm; OLMo-2 needs pure POST-norm and full-width qk-norm. Per
  the OPT/GLM/Gemma precedent (their D2/D4), OLMo-2 gets a NEW `olmo2.{h,cpp}` block
  that reuses only the glue + GQA paged path — it does NOT extend `AttnBlock`.
- **The OLMo-1 non-parametric LayerNorm, OLMoE/FlexOlmo MoE, OlmoHybrid SSM** — all
  out of scope (§Scope), each a distinct row.
- **No MODEL row is `DONE`** anywhere in the matrix; nothing here claims otherwise.

Precedent specs: [`sweep-gemma.md`](sweep-gemma.md) (the sandwich/post-norm + shared
dense bring-up shape), [`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (the
standalone sandwich-norm op + the "new primitives reduce to existing ops" lesson),
[`first-additive-model-qwen3-dense.md`](first-additive-model-qwen3-dense.md) (the dense
per-row W0..W5 protocol), [`sweep-llama-3.2.md`](sweep-llama-3.2.md) (dense new-files
additivity + the BOS tokenizer trap), [`sweep-opt-125m.md`](sweep-opt-125m.md) (the
BOS/tokenizer lesson).

**Anchor-drift warning.** Line anchors move between bases. Re-anchor every cited
`file:line` against the tree at implementation time; `check_links` validates line
ranges. The `olmo2.py`/`glm4.cpp` anchors above are against pin `e24d1b24` / base
`a60b88b`.

### Port map

| Upstream | Ours |
|---|---|
| `registry.py:172-173` `Olmo2ForCausalLM`/`Olmo3ForCausalLM` (`olmo2.py:346-421`) | **NEW** `include/vllm/model_executor/models/olmo2.h` + `src/vllm/model_executor/models/{olmo2_registry,olmo2_weights,olmo2}.cpp` — one `REGISTER_VLLM_MODEL` (covers both arch strings), full-attention-only KV spec (+ sliding-window for Olmo-3), `is_dense_model=true`. **The gate vehicle.** |
| `olmo2.py:261-277` pure post-norm (`norm_after`) forward | **NEW WIRING** in `olmo2.cpp` — standalone `vt::RmsNorm` on each sublayer OUTPUT + plain residual add, NO pre-norm. REUSES the GLM-4 standalone-norm op (`glm4.cpp:174-178`); adds no kernel |
| `olmo2.py:113-117,160-182` full-width QK-norm + rope | **NEW WIRING** — two standalone `vt::RmsNorm` over `[T,q_size]`/`[T,kv_size]` then `RopeNeox`. REUSES `vt::RmsNorm` at a new shape; does NOT use `kAttnQkNormRopeGate`; adds no kernel |
| `layernorm.py::RMSNorm` (plain) | **REUSE** `vt::RmsNorm {gemma=false}` |
| `olmo2.py:188-230` `Olmo2MLP` (SiLU SwiGLU) | **REUSE** `kSiluAndMul` + merged gate_up |
| `olmo2.py:145-149` NeoX rope (Olmo-3: full-attn rope scaling) | **REUSE** `RopeNeox`/`RopeFromCache` |
| `olmo2.py:121-126` Olmo-3 `layer_types` sliding window | **REUSE** the Gemma-3 interleaved-sliding-window + KV-spec wiring (W5) |
| `olmo2.py:351-364` packed qkv/gate_up + `:374-382,417` tie | **REUSE** the merged-column loader + `tie_word_embeddings` skip path |
| ByteLevel BPE tokenizer + BOS | **REUSE** `tokenizer.cpp` ByteLevel path; W0 verifies BOS vs oracle |
| `olmo.py` (OLMo-1 non-parametric LayerNorm), `olmoe.py`/`flex_olmo.py` (MoE), `olmo_hybrid.py` (SSM) | **NOT PORTED** — out of scope; stay `INVENTORIED` (§Scope) |

### Tests to port

Per [`.agents/test-porting.md`](../test-porting.md). Nothing below is ported by this
spike (spec only); this is the inventory that binds the implementing Ws.

| Upstream test | Tier | Ours |
|---|---|---|
| `tests/models/language/generation/test_common.py` (OLMo-2 text-generation entry, `allenai/OLMo-2-*`) | T-parity | `tests/vllm/models/test_olmo2_paged_engine.cpp` (W4) — the SACRED token-exact gate |
| `tests/models/registry.py` `_HfExamplesInfo` for `Olmo2ForCausalLM`/`Olmo3ForCausalLM`/`OlmoForCausalLM`/`OlmoeForCausalLM` | T-unit | config/registry resolution cases (W1) |
| `tests/models/test_initialization.py` (OLMo-2/3 init smoke) | T-unit | init/registry resolution; the Olmo-3 case gates the 0.25.0 oracle-support verdict (W0/W5) |
| `tests/test_config.py` (OLMo arch-config resolution) | T-unit | arch-config resolution, gateable with NO checkpoint |
| `tests/v1/e2e/general/test_correctness_sliding_window.py` | T-e2e | Olmo-3 interleaved sliding-window correctness (W5), if the oracle constructs Olmo3Config |

**Upstream coverage note:** OLMo-2 has a standard text-generation correctness entry
(the same family Llama/Gemma use), directly portable. Our SACRED oracle remains our
own pinned-vLLM comparison per [`.agents/gates.md`](../gates.md).

### Gates

1. **Correctness (SACRED), `Olmo2ForCausalLM` on `allenai/OLMo-2-0425-1B` bf16.**
   Token-exact vs the pinned vLLM 0.25.0 oracle, greedy, identical prompt set. Gate
   form selected BY MEASUREMENT per [`near-tie-distributional-gate`](../gates.md): run
   vLLM's own greedy K=5 first. A 1.485B dense model may be in the small-dense near-tie
   regime — a distributional / near-tie-band fallback (our token in vLLM's K-run set,
   or within a measured nats band with 0 forward-divergent) is permitted ONLY if
   measured; otherwise STRICT. If 1B is a near-tie, add a bigger STRICT check on
   `OLMo-2-1124-7B` (where vLLM greedy is expected deterministic).
2. **New-ops unit gate — NONE required.** OLMo-2 adds no `vt::` op (§0.3); the two new
   facts are block wiring, covered by the e2e forward gate. (Contrast Gemma's
   `kGeluAndMul`/`kSoftCap`.) The full-width-qk-norm and post-norm placement are proven
   by gate 1 — a wrong placement or shape emits fluent-WRONG tokens (the OPT mode), so
   the e2e token-exact gate is the primitive proof.
3. **Loader.** Weight-map coverage with zero unmapped and zero missing tensors on the
   gated checkpoint, including: the FULL-WIDTH `q_norm[hidden]`/`k_norm[kv_size]`
   weights (asserted at the right dims, NOT per-head), the two post-norms per layer,
   the merged qkv/gate_up, and the tied-or-untied `lm_head` skip.
4. **Regression, non-negotiable — all 13 current SACRED gates UNCHANGED.** 27B 235/235,
   35B 315/315, Qwen3-Coder 6/6, Qwen3-dense (0.6B+4B 16/16), GLM-4-9B-0414 16/16,
   GLM-4.7-Flash 8/8, Gemma-1 48/48, Gemma-2 48/48, Gemma-3 48/48, OPT-125m 6/6,
   DeepSeek-V2-Lite 8/8, Llama-3.2-1B 16/16, Mistral-7B-v0.3 16/16. OLMo-2 is
   new-files-only (one forward/loader/registry TU + one `REGISTER_VLLM_MODEL` line, no
   shared kernel/runner/sampler edit), so every existing gate is byte-identical by
   construction; the fast dense gates are the empirical witness.
5. **Build.** Clean full rebuild `-Werror`, zero warnings (a clean rebuild, not
   incremental — header changes are certain; [[incremental-build-masks-werror]]).
6. **memcheck.** `compute-sanitizer` zero errors on the OLMo-2 forward. (No new kernel,
   so this is the block wiring over existing kernels, but the full forward is run under
   memcheck.)
7. **Records.** `scripts/check-agent-record.py`, `scripts/check-doc-checkpoint.py`,
   `scripts/check-device-leakage.py`, `scripts/check-readme-structure.py`,
   `scripts/check-model-checklist.py` all green.
8. **SPEED.** Explicitly PENDING and unclaimed. Per the acceptance rule, a model is
   `DONE` only at token-exact AND vLLM throughput on every axis; no row here reaches
   `DONE` on correctness alone.

### Dependencies

**No hard upward dependency.** Every OLMo-2/3 gate is a pure dense-path bring-up on
landed infrastructure (plain RMSNorm, the GLM-4 standalone post-norm op, SiLU SwiGLU,
NeoX rope, the model-shape-agnostic runner, the Gemma-3 sliding-window wiring for
Olmo-3, tied embeddings, ByteLevel BPE). There is NO genuinely-new compute kernel.

**Checkpoint dependencies (downloads, not yet performed):** `OLMo-2-0425-1B`
~2.77 GiB (primary), `OLMo-2-1124-7B` ~13.6 GiB (bigger strict, only if 1B is a
near-tie). Small totals; stage sequentially ([[grid-per-sha-trees-fill-disk]], dgx
disk tight). Presence verified directly, not from a table (the OPT lesson).

**Blocking precondition for the Olmo-3 sliding-window variant only:** the 0.25.0
oracle must construct `Olmo3Config` (newer than the pin's transformers — unverified).
Until proven, W5 is an honesty pass. `Olmo2ForCausalLM` itself is oracle-certain.

**Downward dependencies this introduces:** none new (no new op). The `olmo2.{h,cpp}`
post-norm block is a reference for any future pure-`norm_after` arch (FlexOlmo's MoE
post-norm layout would reuse it).

### Work breakdown

- **W0 — Ground the facts on hardware.** Verify the 0.25.0 oracle constructs +
  runs `Olmo2ForCausalLM`; fetch `OLMo-2-0425-1B`; confirm the real `config.json`
  matches §0.1-0.6 (pure post-norm, `q_norm`=hidden / `k_norm`=kv_size dims,
  `tie_word_embeddings`, `rope_theta`, GQA head counts, `rms_norm_eps`); run the K=5
  greedy self-determinism probe that selects the gate form; **verify BOS placement vs
  the oracle** (the OPT lesson — OLMo tokenizers commonly prepend NO BOS); probe
  whether the oracle's transformers can construct `Olmo3Config`. No code. *Gate: oracle
  reference outputs produced; determinism + BOS + Olmo-3-oracle verdicts recorded.*
- **W1 — Registry stub + config parse.** New `olmo2_registry.cpp` + `olmo2.h`, one
  `REGISTER_VLLM_MODEL` covering both `Olmo2ForCausalLM` and `Olmo3ForCausalLM`,
  full-attention-only KV spec, `is_dense_model=true`, W2/W3 throwing stubs. *Gate:
  registry resolves both arch strings; regressions UNCHANGED.*
- **W2 — Weight loader.** `olmo2_weights.cpp`: merged qkv/gate_up, the FULL-WIDTH
  `q_norm`/`k_norm` weights, the two post-norms + final norm, tied-or-untied lm_head.
  *Gate: zero unmapped, zero missing on the 1B checkpoint; the full-width norm dims
  asserted (NOT per-head).*
- **W3 — Forward + the two new wiring facts.** New `olmo2.{h,cpp}` block: the PURE
  POST-NORM placement (standalone `vt::RmsNorm` on each sublayer output + plain
  residual add, NO pre-norm) + the FULL-WIDTH qk-norm (two standalone `vt::RmsNorm`
  before `RopeNeox`) + SiLU SwiGLU + GQA paged. Reuses only the `dense_attn_block.h`
  glue + GQA paged path (D2). ZERO new kernel. *Gate: forward finite + deterministic;
  feeds into W4.*
- **W4 — SACRED gate.** Token-exact on `OLMo-2-0425-1B` (form per W0) + bigger strict
  on `OLMo-2-7B` if 1B is a near-tie. *Gate: gates 1, 3, 4, 5, 6.*
- **W5 — Olmo-3 sliding-window variant (conditional).** If W0 proved the oracle
  constructs `Olmo3Config` AND a checkpoint fits: reuse the Gemma-3 interleaved
  sliding-window + full-attn rope-scaling wiring; SACRED gate. Else: honesty pass
  recording the oracle-support verdict. *Gate: gate 1 on an Olmo-3 checkpoint, or the
  recorded verdict.*
- **W6 — Speed close.** Decode-graph sibling + the binding every-axis grid vs vLLM.
  Nothing reaches `DONE` before this. *Gate: acceptance rule — match or beat vLLM on
  every axis.*

### Risks/decisions

**D1 — OLMo-2 is the recent-first gate vehicle; OLMo-3 rides the same row.** `Olmo3ForCausalLM`
maps to the SAME `olmo2.py` class (config-gated). The `OLMo-2-0425-1B` vehicle is
oracle-certain and fits; Olmo-3 e2e is gated behind a W0 oracle-support probe
(Olmo3Config may be too new for 0.25.0). This mirrors GLM (flagship blocked, vehicle =
what fit) and Gemma (Gemma-4 leads, Gemma-3-1b gates).

**D2 — Do NOT extend `dense_attn_block.h::AttnBlock` for OLMo-2.** It hard-codes
PRE-norm placement and per-head qk-norm; OLMo-2 needs pure POST-norm and full-width
qk-norm. The OPT/GLM/Gemma precedent (their D2/D4) already proved this header does not
stretch across norm layouts; reuse only the glue + GQA paged path, write a new
`olmo2.{h,cpp}` block. Tracked debt: a later block-consolidation pass, never smuggled
into a bring-up.

**D3 — The pure post-norm placement is a silent-corruption hazard.** `olmo2.py:261-277`
applies the norm to the sublayer OUTPUT with NO pre-norm and re-joins the residual
with a plain add. Getting the order wrong (e.g. reusing the sandwich layout's fused
pre-norm add, or normalizing the input) emits fluent-WRONG tokens (the OPT/GLM/Gemma
mode). Mitigation: gate 1 is an e2e token-exact gate; W3 grounds the exact residual/add
order against `olmo2.py:266-276` line by line.

**D4 — The full-width QK-norm must NOT be wired as the per-head fused recipe.** OLMo-2
norms the whole q/k vector, not per-head (§0.2). Reusing `kAttnQkNormRopeGate` (per-head
`head_dim` norm) would be a small, uniform, silent numeric drift. Mitigation: W2
asserts the loaded `q_norm`/`k_norm` dims are `hidden`/`kv_size` (not `head_dim`); W3
applies them as two standalone full-width `vt::RmsNorm` before `RopeNeox`; gate 1
proves it end to end.

**D5 — BOS handling is a tokenizer trap.** OLMo tokenizers (GPT-NeoX BPE) commonly
prepend NO BOS and use `<|endoftext|>` as eos/pad. A silently-wrong BOS (added or
omitted against the oracle) scores 0/n while emitting fluent text (the OPT 0/6 mode,
re-hit on Llama-3.2's missing 128000). Mitigation: W0 verifies BOS placement vs the
oracle BEFORE any forward work; the gate can run tokenizer-free (feed the oracle's
exact ids into the CUDA prefill) if our loader does not validate OLMo's tokenizer.json,
mirroring the Gemma/Mistral `LOAD-SENTENCEPIECE` handling.

**D6 — OLMo-1 / OLMoE / FlexOlmo / OlmoHybrid stay `INVENTORIED` by design, not
omission.** OLMo-1 uses a non-parametric `nn.LayerNorm` (mean-centering, no weight —
a different norm than RMSNorm); OLMoE/FlexOlmo add the FusedMoE path (additive later,
distinct rows); OlmoHybrid is a Mamba/GatedDeltaNet SSM campaign. Folding any in would
misrepresent the OLMo-2 dense bring-up as covering it.
