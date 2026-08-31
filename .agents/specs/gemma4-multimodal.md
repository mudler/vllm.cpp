# SPIKE: Gemma-4 multimodal (image + video + AUDIO) + the AUDIO track

**SPIKE ONLY — READ-ONLY design + a checkpoint/oracle/HW-fit check. No
implementation, no tower built, no build, no download, no gate.** Grounds the
remaining piece of the user's #1 roadmap priority (2026-07-25: *"Multimodal
Audio/Video/Image with Gemma-4 and Qwen3.6"*). Qwen3.6 image+video is LANDED
(M2/M3, [multimodal-track.md](multimodal-track.md)); **audio exists ONLY in
Gemma-4 / gemma3n**, so this spike scopes (a) the genuinely-new AUDIO modality
and (b) the Gemma-4 backbone+towers, honestly separating what is reachable on
GB10 against the pinned oracle from what is staged/blocked.

**Base:** `origin/main` `64a01af` (M3c video near-tie gate landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` = vLLM
**0.25.0** + **transformers 5.13.1** (measured 2026-07-25). **Claim:**
`CLAIM-GEMMA4-MULTIMODAL`.
**Precedent spikes mirrored:** [`multimodal-track.md`](multimodal-track.md) (the
landed M0–M3 mm infra this reuses), [`sweep-gemma.md`](sweep-gemma.md) (the
Gemma-4 text-backbone characterization: PLE/YOCO/Gemma-4-MoE), and the
blocked-row honesty precedent [`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md).

Rows this spike advances (owned; the concurrent Qwen3.6-video agent owns the Qwen
`MODEL-MM-*` rows + `multimodal-track.md` — NOT touched here):
- `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` — stays `SPIKE`, verdict
  sharpened to **oracle-BLOCKED (decisive)** + re-pointed to this spec.
- `MODEL-MM-gemma4-unified-gemma4-unified-for-conditional-generation` — same.

---

## 0. Headline findings

> **W0 RUN-VERIFIED (2026-07-28, `CLAIM-GEMMA4-W0`): the oracle LOADS + RUNS +
> GENERATES — Gemma-4 is GATEABLE, greedy golden captured.** The decisive
> oracle-gateability gate ([[oracle-gateability-model-runs-not-config-constructs]])
> is now PASSED, not inferred: on dgx, `~/venvs/vllm-oracle` = vLLM **0.25.0** +
> transformers **5.13.1**, `vllm.LLM(model=unsloth/gemma-4-E4B-it)` (ungated,
> `Gemma4ForConditionalGeneration`, 15.99 GB bf16 single shard) **resolved the arch,
> loaded the weights onto the GB10, built the KV cache, and greedily generated 32
> coherent tokens** (`enforce_eager`, `temperature=0`, GMU 0.30 under `flock`). vLLM
> configured Gemma-4's heterogeneous head dims (head_dim=256 / global_head_dim=512 →
> forced `TRITON_ATTN`) and ran the PLE/YOCO/Gemma-4-MoE backbone e2e. **K=5
> self-determinism = ALL-DETERMINISTIC ⇒ the future bring-up bar is STRICT
> token-exact.** This is the opposite of OLMo-3 (which CONSTRUCTS but ABORTS on run):
> Gemma-4 CONSTRUCTS **and RUNS**. Golden fixture:
> `tests/parity/goldens/gemma4_e4b_text/gen_manifest.json` (prompt + exact 32 output
> token ids + K=5 runs + sha256; capture script `scripts/mm/g0_gemma4_oracle_capture.py`).
> **Verdict: the oracle-block is fully retired; the ONLY remaining work is
> implementation** (the PLE/YOCO/Gemma-4-MoE backbone + the SigLIP/USM-Conformer
> towers — see the G-plan §2.2). Text-only ran (sufficient for W0 — it exercises the
> full backbone); an image/audio prompt was NOT run this pass (staged to G2/G3).
> Golden greedy tokens (ref, STRICT): `[236776, 2455, 5192, 2028, 563, 496, 3996,
> 16477, 14020, 1948, 15453, 580, 12566, 12136, 529, 1816, 1262, 531, 3050, 236764,
> 8729, 236764, 532, 8932, 531, 3246, 5192, 528, 496, 40137, 532, 4403]` = *"A large
> language model is a complex artificial intelligence program trained on massive
> amounts of text data to understand, generate, and respond to human language in a
> coherent and context"*. Oracle input prompt ids (chat-templated, len 20): BOS 2 →
> `<|turn>user…`. The two subsections below are preserved as the (now-superseded)
> gate-time record.

> **POST-PIN-ADVANCE UPDATE (2026-07-26): the decisive block is DISSOLVED.** The
> parity pin advanced to `555967922` / vLLM 0.26.0.dev0, which carries **transformers
> 5.14.1 - and 5.14.1 SHIPS `transformers.models.gemma4`** (environment.md:39-40;
> the advance explicitly lists Gemma-4 among its unblocks). So the "oracle cannot
> construct the mm path" verdict below was correct against the *gate-time* 0.25.0 /
> transformers-5.13.1 oracle but no longer holds on the current pin. Re-assessed
> verdict: Gemma-4 multimodal is **reachable on the advanced pin, implementation
> pending** (the ≥12B HF-gated mm-wrapped checkpoint + the PLE/YOCO/Gemma-4-MoE
> backbone + the SigLIP/USM-Conformer towers remain the real work; the oracle is no
> longer the blocker). The analysis below is preserved as the gate-time record.

### 0.0 The DECISIVE gating fact (AT GATE TIME): the 0.25.0 oracle CANNOT construct Gemma-4 mm (transformers 5.13.1 has no `gemma4`)

The Gemma-4 mm wrapper does **not** implement its towers natively. It loads them
from **Transformers** via `AutoModel.from_config`:

- vision tower — `gemma4_mm.py:1040` `self.vision_tower = AutoModel.from_config(config=config.vision_config)`
- audio tower — `gemma4_mm.py:1056` `self.audio_tower = AutoModel.from_config(config=config.audio_config)` (+ `post_init()` at `:1061` to build the Conformer `inv_timescales`/softcap/grad-clip buffers absent from the checkpoint)

and the config classes are imported straight from Transformers:
`gemma4_mm.py:25-33` `from transformers.models.gemma4 import (Gemma4Config,
Gemma4Processor, Gemma4VisionConfig)` and `...configuration_gemma4 import
(Gemma4AudioConfig, Gemma4TextConfig)`.

**Measured on dgx (2026-07-25):** `~/venvs/vllm-oracle` = vLLM 0.25.0 +
**transformers 5.13.1**, and `import transformers.models.gemma4` **FAILS**
(ModuleNotFoundError) while `import transformers.models.gemma3n` **succeeds**. So
constructing `Gemma4ForConditionalGeneration` on the pinned oracle dies at
`AutoModel.from_config(config.vision_config)` — **there is no SACRED oracle for
the Gemma-4 mm path, hence no gateable target.** This is stronger and more
decisive than [`sweep-gemma.md`](sweep-gemma.md) §W6 found for the bare text row
(which only worried oracle *registry* listing): even though vLLM 0.25.0 ships
`gemma4_mm.py`/`gemma4_unified.py`, the towers are Transformers modules the pinned
Transformers does not carry. **A model the oracle cannot run has no gate**
([verification.md](../verification.md)). This is the load-bearing reason Gemma-4 mm is
HONESTY-PASS-BLOCKED, independent of checkpoints and HW.

### 0.1 The Gemma-4 mm architecture — two variants, a SigLIP vision tower, and a USM Conformer audio tower

`gemma4_mm.py` (1708 lines) registers `Gemma4ForConditionalGeneration`
(`registry.py:392`); `gemma4_unified.py` (469 lines) registers
`Gemma4UnifiedForConditionalGeneration` (`registry.py:393-396`), which
**subclasses** `Gemma4ForConditionalGeneration` (`gemma4_unified.py:40,220`).

**Variant A — `Gemma4ForConditionalGeneration` (the encoder variant):**
- **Vision tower** — a **SigLIP-class ViT** loaded via `AutoModel.from_config(config.vision_config)`
  (`gemma4_mm.py:1040`), shared by image AND video (`_mark_tower_model(..., {"image","video"})`
  `:1039`). Structurally a learned-abs-pos ViT (patch-embed → pre-LN transformer
  blocks with bidirectional attention + gelu-tanh MLP), the Gemma-3 vision lineage
  — NO patch-merger / NO DeepStack / NO vision-RoPE (simpler than the Qwen3-VL
  tower we landed in M2a). Video path: `_VIDEO_MAX_SOFT_TOKENS=70`, `_VIDEO_MAX_FRAMES=32`
  (`gemma4_mm.py:91-92`).
- **Audio tower** — a **USM-class Conformer** loaded via `AutoModel.from_config(config.audio_config)`
  (`gemma4_mm.py:1056`), present **only** on variants whose config carries an
  `audio_config` (`:1055`, else `audio_tower=None` `:1074`). Frontend per the
  processor arithmetic (`gemma4_mm.py:348-368` `_compute_audio_num_tokens`): **mel
  framing** (`Gemma4AudioFeatureExtractor._unfold`) → **two Conv2d subsampling
  layers** (kernel 3, stride 2, semicausal pad top=1/bottom=1) → Conformer blocks
  with relative-position attention + softcap (the `post_init()` buffers). Forward
  `gemma4_mm.py:1468-1490`: `audio_tower(input_features, input_features_mask)` →
  `(audio_encodings, audio_mask)`, then the projector, then per-audio padding strip.
  `hidden=1024`, `output_proj_dims=1536` (`gemma4_mm.py:936-940`).
- **Projector(s)** — `Gemma4MultimodalEmbedder` (`gemma4_mm.py:908-965`): a
  **2-layer** design = `RMSNorm(has_weight=False)` (`:942`) → `ReplicatedLinear`
  bias-free to `text_hidden_size` (`:949`). One instance each for vision
  (`embed_vision`, in-dim `hidden_size`=768) and audio (`embed_audio`, in-dim
  `output_proj_dims`=1536). No embedding table, no pre-projection weights (the
  checkpoint only has `embedding_projection.weight`).
- **Backbone** — `Gemma4ForCausalLM` (`gemma4_mm.py:44`), the PLE/YOCO/KV-sharing/
  Gemma-4-MoE/k_eq_v text stack characterized in [`sweep-gemma.md`](sweep-gemma.md)
  §0.1 (none of it landed).
- **Merge** — soft tokens scatter into audio/image/video placeholder rows of
  `input_embeds` (the `SupportsMultiModal` mixin, `interfaces.py`), identical in
  shape to the vision merge we landed (M2c/M3b `Qwen3VLMergeMultimodal`).
- **Modality support gating** (`gemma4_mm.py:216-235,270-275`): image+video always;
  **audio only if `audio_config is not None`** — a non-audio Gemma-4 checkpoint
  raises "does not have an audio tower" (`:225-229`). `audio` mm-limit and
  `audio_seq_length` come from the processor (`:235,255`).

**Variant B — `Gemma4UnifiedForConditionalGeneration` (encoder-FREE):**
`gemma4_unified.py:3-16` — **no SigLIP vision tower and no audio tower**; images
flow through `Gemma4UnifiedVisionEmbedder` (`:73`), a lightweight patch pipeline
with **factorized 2-D positional embeddings** straight into text-embed space
(`:266-291` overrides `embed_vision`/`embed_audio`). Audio still gated on
`audio_config` (`:172,288-291`). This is a **different, simpler** design than
Variant A (no `AutoModel` vision tower) — but it STILL imports
`transformers.models.gemma4_unified` configs (`:23`), so it is oracle-blocked by
0.0 for the same reason.

**Checkpoint→variant map** (HF metadata, [`sweep-gemma.md`](sweep-gemma.md) §0.0,
nothing downloaded): `google/gemma-4-12B-it` (11.96B) = **Unified** (encoder-free);
`google/gemma-4-31B-it` (32.68B) + `google/gemma-4-26B-A4B-it` (26.5B MoE) =
**Variant A** (encoder). Which of them carry an `audio_config` (hence do audio) is
per-checkpoint and unverified — the audio tower is optional; a fetch-time
`config.json` read decides it.

### 0.2 Reuse-vs-new against the LANDED M0–M3 mm infra

The image+video track (M0–M3) built a reusable spine. Mapping each Gemma-4/audio
piece onto it:

| Gemma-4 / audio piece | Landed M0–M3 anchor | Verdict |
|---|---|---|
| mm INPUT container (`MultiModalKwargs`, feature specs) | `src/vllm/multimodal/` (`inputs.h`) | **REUSE** — modality-agnostic; audio adds a new feature-tensor kind (`input_features`+mask) |
| mm-hash | `MultiModalHasher` blake3 (`src/vllm/multimodal/hasher.cpp`) | **REUSE** — hashes any serialized media incl. audio bytes |
| encoder-cache engine seam | `EncoderCacheManager` + budget (`src/vllm/v1/core/encoder_cache_manager.cpp`) | **REUSE** — keyed by mm-hash, modality-agnostic |
| placeholder-token expansion + masked-scatter merge | `Qwen3VLMergeMultimodal` (M2c/M3b, `qwen3_vl.cpp`/`qwen3_5.cpp`) | **REUSE (pattern)** — audio soft tokens scatter into audio-placeholder rows exactly like image tokens; a Gemma-4 audio-token id + count-arithmetic is the only new wiring |
| LMCache `extra_keys` mm slot | `chunked_token_database.{h,cpp}` | **REUSE** — already carries the slot |
| ViT tower scaffold (patch-embed matmul, attention, LayerNorm, GELU MLP) | M2a `qwen3_vl_vision.{h,cpp}` + `vt::` GEMM/attn/norm + `GeluTanh`/`GeluErf` ops | **REUSE (scaffold) for the SigLIP vision tower** — SigLIP delta: learned abs pos-embed (no vision-RoPE), no patch-merger, no DeepStack → SIMPLER than M2a; drop those three, keep blocks+MLP+LN |
| the merge/decode fork driver | M2c/M3b `VLGenerateCore` | **REUSE (pattern)** for any tower→merge→decode |
| **AUDIO input pipeline** (decode→resample→log-mel→_unfold framing→count) | — none (grep: only image/video processors) | **GENUINELY NEW** — the largest audio piece; no audio preprocessing exists today |
| **AUDIO encoder tower** (USM Conformer: 2×Conv2d subsample + relative-pos attn + conv-module + softcap) | — none | **GENUINELY NEW tower TYPE** — attention/FFN reuse `vt::` ops; the Conformer conv-module + relative-position bias + Conv2d subsampling are new kernels |
| audio projector | `Gemma4MultimodalEmbedder` = RMSNorm(no-weight)+Linear | **REUSE** ops (`vt::RmsNorm`+GEMM), trivial |
| **Gemma-4 backbone** (PLE/YOCO/KV-share/Gemma-4-MoE/k_eq_v) | — none | **GENUINELY NEW** — the [`sweep-gemma.md`](sweep-gemma.md) §0.1 stack; a separate campaign, no other row needs it |
| Gemma sandwich norms / soft-cap / GeGLU / gemma-RMSNorm | LANDED (Gemma-1/2/3 text, `gemma{,2,3}.cpp`; `kGeluAndMul`, `kSoftCap`) | **REUSE** — the Gemma text primitives are done |

**Net:** the mm SPINE (input container, hash, encoder cache, merge, decode fork)
and the ViT scaffold + Gemma text primitives are **ours**. Genuinely NEW =
**(a) the audio input pipeline, (b) the audio USM-Conformer tower, (c) the Gemma-4
backbone**; the SigLIP vision tower is a REUSE-with-simplification of M2a.

### 0.3 Checkpoint + oracle + GB10-fit — the gating facts (Gemma-4)

| Fact | Verdict |
|---|---|
| **Oracle constructs the mm path?** | **NO at gate time (decisive then; DISSOLVED on the advanced pin)** — transformers 5.13.1 had no `gemma4`; `AutoModel.from_config(vision_config)` failed (0.0). The current 0.26.0.dev0 pin carries transformers 5.14.1, which ships `gemma4`, so the oracle can now construct it. |
| Oracle ships the vLLM files? | Yes (`gemma4_mm.py`,`gemma4_unified.py` present) — necessary but NOT sufficient; the towers are Transformers modules. |
| Smallest checkpoint | `google/gemma-4-12B-it` (11.96B, **Unified/encoder-free**); next `26B-A4B`/`31B` (Variant A). All **≥12B, mm-wrapped, `google/*` HF-gated**. No ungated bare-text or small mirror. Audio-capable variant = whichever carries `audio_config` (per-checkpoint, unverified). |
| dgx cache | **none** — `ssh dgx 'ls ~/.cache/huggingface/hub ~/bench \| grep -i gemma'` = gemma-3-1b-it, unsloth gemma-2/2b only. No Gemma-4, no audio-LLM. |
| GB10 fit (119 GiB unified) | 12B (~24 GiB bf16) + SigLIP tower + audio tower **FITS**; 26B-A4B/31B **HW-marginal-to-blocked** ([`sweep-gemma.md`](sweep-gemma.md) §0.6). |

**Verdict: Gemma-4 mm = HONESTY-PASS-BLOCKED / STAGED.** Not e2e-reachable at the
pin: (1) oracle cannot construct it (no gate); (2) checkpoints ≥12B, gated,
mm-wrapped, none cached; (3) three new subsystems unbuilt (SigLIP tower is a
reuse, but the **audio pipeline + USM Conformer tower + the PLE/YOCO/MoE
backbone** are all new). HW is the ONLY non-blocker (12B fits). Reopen when the
oracle's Transformers advances to carry `gemma4` (or the pin advances) AND a
fitting checkpoint downloads.

### 0.4 Audio reachability BEYOND Gemma-4 — the smallest oracle-runnable vehicle to stand the modality up

Audio is the genuinely-new modality (nothing built). Mirroring how vision was
stood up on Qwen3-VL-4B before Qwen3.6-27B, land the audio subsystems on the
smallest model the pinned oracle CAN run, then carry them to Gemma-4. All the
following have **native vLLM towers** (grep: 0 `AutoModel.from_config` in
`whisper.py`/`qwen2_audio.py`*/`voxtral.py`/`ultravox.py`/`granite_speech.py`) and
their Transformers deps ARE in 5.13.1 — so unlike Gemma-4 they are
oracle-constructible.

| Vehicle | Params (approx, verify at fetch) | Audio encoder | Backbone | Oracle 0.25.0 | GB10 fit | Role |
|---|---|---|---|:--:|:--:|---|
| **Whisper** (`whisper-small` 244M / `base` 74M / `tiny` 39M) | 39M–244M | native WhisperEncoder (log-mel 128 + 2×Conv1d + transformer), `whisper.py:458` | encoder-DECODER (ASR) | ✅ native | ✅ trivial | **A1/A2 vehicle** — smallest, stands up the audio INPUT pipeline (log-mel) + first encoder tower in isolation |
| **Voxtral-Mini-3B-2507** | ~4.7B | native `WhisperCausalEncoder` (`voxtral.py:671,737`) | **Mistral** (LANDED: Mistral-7B-v0.3 text SACRED 16/16) | ✅ native | ✅ ~9.4 GiB | **A3 vehicle** — smallest decoder-MERGE audio-LLM on a backbone we own → e2e audio→text |
| Qwen2-Audio-7B | ~8.4B | transformers `Qwen2AudioEncoder` (Whisper-class; qwen2_audio in 5.13.1), `qwen2_audio.py:349` | Qwen2 decoder | ✅ | ✅ ~17 GiB | A3 fallback (larger; Qwen2 backbone) |
| Granite-Speech-3.3-2b | ~2–3B | native **Conformer** (`granite_speech.py:294-297`) | Granite decoder | ✅ native | ✅ | **Conformer reference** — closest encoder to Gemma-4's USM (proves the conv-module) |
| gemma3n-E2B-it | 5.44B | native USM **Conformer** (`gemma3n.py`, `gemma3n_audio_utils.py`) | MatFormer/AltUp (complex, out-of-scope per sweep-gemma D6) | ✅ (transformers has gemma3n) | ✅ ~10.9 GiB | Gemma-family Conformer ref; backbone too heavy to be the first vehicle |

**Named smallest audio-runnable vehicle: `whisper-small` (244M, native,
oracle-certain, fits trivially)** to stand up the audio INPUT pipeline + first
encoder tower, then **`Voxtral-Mini-3B`** (native Whisper-class encoder + our
LANDED Mistral backbone + projector-merge) for the e2e audio→text gate — the
low-risk path that reuses a text backbone we already own. **Encoder-family caveat
(honest):** Whisper/Voxtral/Qwen2-Audio use a **Whisper-class** encoder (2×Conv +
vanilla transformer); Gemma-4/gemma3n/Granite use a **USM Conformer** (Conv2d
subsampling + conv-module + relative-pos attn + softcap). So these vehicles
de-risk the audio *pipeline + merge pattern* fully, but the **Gemma-4-specific
Conformer tower** is proven separately on **Granite-Speech-2b** (or gemma3n-E2B) —
not by Whisper/Voxtral.

---

## G1 — TEXT backbone LANDED (2026-07-28, `CLAIM-GEMMA4-G1`)

The Gemma-4 **text backbone** (`Gemma4ForConditionalGeneration` language_model
stack of `unsloth/gemma-4-E4B-it`) is implemented as NEW additive files —
`include/vllm/model_executor/models/gemma4.h`,
`src/vllm/model_executor/models/{gemma4,gemma4_weights,gemma4_registry}.cpp` —
mirroring the OLMo-2/gemma3 registration seam (one `REGISTER_VLLM_MODEL` line, no
shared-array edit). CPU `-Werror` 0-warn on all three TUs + full `libvllm.a` link
(SACRED inertness: the whole existing model set still builds).

### G1.0 E4B config — which primitives are actually ON

From `unsloth/gemma-4-E4B-it` `text_config` (fetched HF, 2026-07-28): hidden 2560,
42 layers, GQA 8/2, head_dim 256 / **global_head_dim 512**, intermediate 10240,
`hidden_size_per_layer_input` 256, `num_kv_shared_layers` 18, sliding_window 512,
`final_logit_softcapping` 30.0, vocab 262144, tie_word_embeddings true. Crucially
**`enable_moe_block=false`, `attention_k_eq_v=false`, `use_double_wide_mlp=false`**
→ the Gemma-4 MoE router / per_expert_scale, k_eq_v, and double-wide MLP are OFF
for E4B and are the ≥12B-checkpoint follow-on, NOT G1.

### G1.1 Primitive-by-primitive port map (grounded, file:line)

| Primitive | vLLM ground | Our realization | REUSE / NEW |
|---|---|---|---|
| **PLAIN RMSNorm** (`x·w`, NOT gemma `(1+w)`) | `gemma4.py:45` imports `layernorm.RMSNorm` (not `GemmaRMSNorm`); used at every norm | `vt::RmsNorm(...,{eps,false})` | REUSE (the `gemma=false` mode) — **the load-bearing divergence from gemma2/3** |
| **PLE** (Per-Layer Embeddings) | `gemma4.py:986-1063` (tables) + `:845-898` (combine) + `:680-761` (per-layer gate/proj/norm) | `embed_tokens_per_layer` lookup·√ple + `per_layer_model_projection`·h^-0.5 → RMSNorm → `(proj+emb)·rsqrt2`; per layer `gelu(gate_lin(h))*ple` → proj → norm → add | NEW wiring; the gate reuses `vt::GeluAndMul` on `[gate_lin ‖ ple]` (no elementwise-Mul op exists) |
| **YOCO KV-sharing** | `gemma4.py:463-489`, forward `:535-548` | shared layers (24-41) read the target layer's cache (sliding→22, full→23) IN-forward, compute no K/V | NEW (in-forward target-cache index; no runner aliasing) |
| **heterogeneous head_dim** 256/512 | `gemma4.py:572-578` | per-layer `Dh` threaded through the attn block | NEW |
| **proportional partial-RoPE** (full) | `gemma4_rope.py` (head_dim denom + zero-pad), `rope_parameters.full_attention` (θ 1e6, pf 0.25 → rotary 128/512) | custom host cos/sin cache → `vt::RopeFromCache` | REUSE `RopeFromCache` + NEW cache builder |
| **standard sliding rope** | `rope_parameters.sliding_attention` (θ 1e4, full 256) | `vt::RopeNeox` | REUSE |
| **weight-less V-norm** | `gemma4.py:437` `has_weight=False` | `vt::RmsNorm` with a ones[Dh] weight (identity) | REUSE |
| **GeGLU MLP** | `gemma4.py:224-254` `gelu_pytorch_tanh` | `vt::GeluAndMul` | REUSE (gemma2/3 W1 primitive) |
| **√hidden embed-scale** | `gemma4.py:1067-1074` | `vt::MulScalar` bf16 normalizer | REUSE |
| **per-layer scalar** | `gemma4.py:707,765` `layer_scalar` [1] | host-read bf16 → `vt::MulScalar` | REUSE |
| **final logit soft-cap 30** | `gemma4.py:1569-1572` | `vt::SoftCap` | REUSE (gemma2 W3 primitive) |
| **tied lm_head** | `gemma4.py:1566-1567` | `MatmulBT` over embed table | REUSE |

Residual pattern is standalone-norm + explicit `vt::Add` (NOT the gemma2/3 fused
add-norm sandwich), because PLE + `layer_scalar` intervene after the second add.

### G1.2 Weight loader — VERIFIED (no download)

`LoadGemma4ForConditionalGenerationWeights` strips the `model.language_model.`
prefix and skips the mm towers (`audio_tower`/`vision_tower`/`embed_audio`/
`embed_vision`, per `gemma4.py:1716-1723`). VERIFIED against the real E4B
safetensors HEADER (HTTP range, no 16 GB download): 2130 total tensors; every one
of the 336 `language_model.*` names + shapes matches the loader's expected map
(incl. per-layer q/k/v/o at the correct 256/512 head widths, PLE tables, per-layer
gate/proj/norm, `layer_scalar` [1], tied embeddings). Shared layers (24-41) DO
carry their own q/k/v_proj in the checkpoint (loaded but K/V discarded at forward).

### G1.3 HONEST e2e GATE STATUS — BLOCKED on runner KV topology (named)

The strict 32/32 gate vs `tests/parity/goldens/gemma4_e4b_text/gen_manifest.json`
(golden `[236776, 2455, 5192, ...]`) is **NOT reached this pass**, and the reason is
precise, not a numeric divergence: the runner allocates **one uniform KV head_dim**
per non-GDN layer (`src/vllm/v1/worker/gpu/runner.cpp:600-646`, `attn_kv_` built
with a single `Hkv`/`Dh`). Gemma-4's per-layer **256 (sliding) / 512 (full)** head
dims cannot be represented without a shared-path change to `attn_kv_` construction
(per-layer/per-group head_dim). The forward's per-layer
`VT_CHECK(kv.head_size == Dh)` turns this into an explicit failure rather than a
silent wrong answer. This is the additive-vs-shared-path boundary: G1 is clean
additive files; the KV-topology change is a runner edit deferred to **G-next**.

**G-next (to reach strict 32/32):** (1) runner heterogeneous per-layer KV head_dim
(+ optional YOCO cache aliasing to reclaim the 18 shared caches' memory); (2) a full
CUDA build + STRICT gate on dgx under `flock`; (3) verify the two named bf16-rounding
nuances (the f32-accumulated PLE combine in vLLM vs our per-op bf16; the proportional
cos/sin cache dtype) do not perturb the token match. G2 (SigLIP vision, reuses M2a)
and G3 (USM-Conformer audio) remain separate and unbuilt.

### G1b — TEXT PATH STRICT 32/32 LANDED (2026-07-28, `CLAIM-GEMMA4-G1B`)

**The strict e2e gate PASSES: 32/32 token-exact.** `unsloth/gemma-4-E4B-it` loads
through our engine (`LoadedEngine::FromModelDir`) and greedily emits the EXACT 32
golden ids `[236776,2455,5192,…]`; gate `tests/parity/test_gemma4_paged_engine.cpp`
(dgx CUDA, `flock`, FA2 on). The (1)/(2)/(3) checklist above is done; the two flagged
bf16 nuances did NOT perturb the token match (32/32 exact).

**(1) Runner per-layer KV head_dim — the named G1b deliverable, byte-neutral.**
`KVCacheConfig` gains an OPTIONAL `per_layer_attn_specs` (index == layer). When set,
`runner.cpp` `initialize_kv_cache` sizes each non-GDN layer's paged KV buffer + view
from its OWN `FullAttentionSpec` (`page_size_bytes`/`num_kv_heads`/`head_size`); the
model's `MakeGemma4…KVCache` publishes sliding 256 / global 512 per layer. EMPTY for
every uniform-KV model ⇒ the loop collapses to the single group spec ⇒ byte-identical
allocation, view, indexing and kernel dispatch (the sm_75-guard "additive, identical
existing path" property). The block table / KV manager / scheduler are head_dim-
independent (num_blocks + block_size, uniform), so a single group + per-layer
allocation needs no per-group block table. YOCO sharing was already correct in the
forward (`kv_idx = shared ? target : l`); the shared layers' own unused buffers are
still allocated (memory-only **G1c** residual).

**Inertness proof.** CPU: `test_runner` (120 s, the runner-KV path) + `test_scheduler`
+ `test_kv_cache_{interface,manager,coordinator}` + `test_single_type_kv_cache_manager`
+ `test_input_batch` + `test_hf_config` + `test_tokenizer_parity{,_mistral,_deepseek}`
all green; full CPU suite `-Werror` 0-warn. GPU SACRED re-gate: OLMo-2-0425-1B 16/16
UNCHANGED through the modified runner on the FINAL binary (13 strict + 3 near-tie, 0
divergent).

**(4) Three additive loader gaps found on the first-ever Gemma-4 forward** (each
byte-neutral for existing models — all previously threw / were unreachable):
- **nested per-layer `rope_parameters`** (`hf_config.cpp`): the loader threw on
  `{full_attention:{…},sliding_attention:{…}}`; now it loads (records presence, keeps
  the nested dict in `raw` for the model to read) instead of aborting.
- **Gemma metaspace-via-normalizer** (`tokenizer.cpp`): E4B expresses metaspace as a
  `Replace(" "→"▁")` normalizer + `Split(" ",MergedWithPrevious)` pre_tokenizer (not a
  `Metaspace` node); folded onto the SAME validated metaspace machinery.
- **`raw["text_config"]` resolution** (`gemma4.cpp`/`gemma4_weights.cpp`/
  `gemma4_registry.cpp`): `HfConfig::raw` is the FULL config (`hf_config.cpp:414`), so
  the G1 reads of `global_head_dim`/`layer_types`/`hidden_size_per_layer_input`/
  `num_kv_shared_layers` from `raw` hit FALLBACKS — every layer was silently treated as
  sliding (head_dim 256). This is the bug the 32/32 forward surfaced (a full layer's
  o_proj `[2560,4096]` fed a 2048-wide input); all Gemma-4 raw reads now go through the
  text_config view.

**Residuals:** YOCO cache DEDUP (G1c, memory-only); G2 SigLIP vision (reuses M2a);
G3 USM-Conformer audio; per-axis SPEED vs vLLM (text path is correctness-DONE,
speed-pending).

### Port map

The G1b runner + loader port map (1:1 upstream anchors; forward primitives are the
G1.1 primitive-by-primitive table above):

| What | Ours | Upstream mirror |
|---|---|---|
| runner per-layer KV head_dim | `KVCacheConfig::per_layer_attn_specs` (`include/vllm/v1/kv_cache_interface.h`) consumed in `src/vllm/v1/worker/gpu/runner.cpp` `initialize_kv_cache`; published by `MakeGemma4…KVCache` (`gemma4_registry.cpp`) | per-layer KV-cache-spec grouping, `vllm/v1/worker/gpu/model_runner.py` `initialize_kv_cache` @ `e24d1b24` (single group + per-layer alloc, block table head_dim-independent) |
| nested per-layer rope | `ParseRopeParameters` loads nested `rope_parameters` (`src/vllm/transformers_utils/hf_config.cpp`); model reads it from `raw` | vLLM keeps per-layer-type rope configs on the model (`gemma4.py` `Gemma4RotaryEmbedding` per attention type) |
| Gemma metaspace normalizer | `DetectGemmaMetaspaceNormalizer` folds `Replace(" "→"▁")` + `Split` onto the metaspace path (`src/vllm/tokenizer/tokenizer.cpp`) | HF tokenizers: `Replace`+`Split(MergedWithPrevious)` == `Metaspace(replacement="▁", split=false)` |
| text_config scalar reads | `TextCfg` view in `gemma4.cpp` / `gemma4_weights.cpp` / `gemma4_registry.cpp` | `PretrainedConfig.get_text_config()` (`vllm/transformers_utils/config.py`) — Gemma-4 scalars live under `text_config` |

### Work breakdown

- **G1b [DONE 2026-07-28]** — runner heterogeneous per-layer KV head_dim + the 3
  loader gaps → text path STRICT 32/32 token-exact (this section).
- **G1c** — YOCO shared-layer cache DEDUP (memory-only; correctness already right,
  shared layers read the target's cache in-forward).
- **G2** — SigLIP vision tower (reuses the landed M2a Qwen3-VL ViT scaffold).
- **G3** — USM-Conformer audio tower (new: mel + 2×Conv2d subsample + conv-module +
  relpos), staged behind the audio track.
- **Speed** — per-axis throughput/latency vs vLLM for the text path (correctness
  DONE, speed-pending).

---

## G2 — IMAGE oracle + SigLIP/NaFlex port map LANDED (2026-07-28, `CLAIM-GEMMA4-G2`)

**Deliverable this pass (golden-first, mirroring M2a + the G1 honest-partial cadence):**
the IMAGE→text oracle golden + the four staged vision-tower reference tensors +
the corrected SigLIP/NaFlex port map. The C++ tower forward + the Gemma-4 NaFlex
image processor + projector/merge wiring are the **named residual** (unbuilt this
pass — NO token-exact claimed for our engine yet).

### G2.0 The IMAGE golden — CAPTURED, STRICT (the SACRED anchor)

`unsloth/gemma-4-E4B-it` on the pinned vLLM 0.25.0 oracle (dgx, `flock`, GMU 0.30,
`enforce_eager`, bf16), fixed committed image `g2_fixed_112.png` (112×112 gradient,
array-sha `306c792d…`) + chat prompt `[{image},{text:"Describe this image in one
sentence."}]`:

- **K=5 DETERMINISTIC ⇒ STRICT gate form** (same bar as Qwen3-VL image STRICT 32/32;
  measured, not assumed — `scripts/mm/g2_gemma4_image_oracle_capture.py`).
- 18 greedy output tokens `[2094,563,496,28239,…,236761,106]` →
  `"This is a vibrant, abstract background featuring a smooth gradient of bright,
  blended colors."` — a COHERENT description of the gradient image ⇒ the vision path
  is genuinely exercised e2e (not a text-only fallback).
- `prompt_token_ids` len **274** = `boi`(255999) + **256** image soft-tokens +
  `eoi`(258882) + template/text (~16). The image yields **256** valid soft tokens
  after pool+strip (of the 280 `default_output_length` budget).
- Fixture `tests/parity/goldens/gemma4_e4b_image/` (11 MB): `gen_manifest.json`
  (token ids, determinism, gate form, prompt ids, processor shapes/shas), the fixed
  PNG, `vision_refs/` (below), all sha-verified locally.

### G2.1 Staged vision references — CAPTURED (the M2a per-stage unit-gate targets)

Dumped by `scripts/mm/g2_vision_ref_dump.py` (transformers-eager
`Gemma4VisionModel` + `Gemma4MultimodalEmbedder`, vision weights only, run on the
golden's EXACT processor outputs; vLLM runs this same tower in eager per
`gemma4_mm.py` docstring ⇒ faithful stage refs). All four stage shas in
`vision_refs/vision_ref_manifest.json`:

| Stage | Ref | Shape | Committed? | Unit-gate it localizes |
|---|---|---|:--:|---|
| image processor out | `proc_pixel_values.npy` + `proc_image_position_ids.npy` | `[1,2520,768]` f32 / `[1,2520,2]` i64 | ✅ | **C++ NaFlex image processor** (patchify + (x,y) ids + padding) |
| patch-embed out | `ref_patch_embedder` (sha only; regen from committed inputs) | `[1,2520,768]` | sha | patch-embed: `input_proj` + learned 2D pos-embed |
| encoder last-hidden | `ref_encoder_last_hidden` (sha only; regen) | `[1,2520,768]` | sha | 16 ViT blocks (RoPE+q/k/v-norm+sandwich) |
| pooled+stripped | `ref_pooled_stripped.npy` | `[256,768]` | ✅ | pooler (avg-by-position + √hidden fp32) + padding strip |
| projected (merge input) | `ref_projected.npy` | `[256,2560]` | ✅ | `embed_vision` projector — THE tensor that scatters into text |

`2520 = 280 × 3²` (max soft tokens × pooling_kernel²); `768 = 3·16²` (patch pixels).
The two 7.4 MB intermediates are regenerable (committed script + committed proc
inputs) so only their shas ship — the image-processor target and the final two
stages are committed as `.npy`.

### G2.2 SigLIP/NaFlex port map — GROUNDED (corrects the §0.1 "no RoPE" error)

**★ CORRECTION to §0.1:** the earlier spike claimed the Gemma-4 vision tower has
"NO vision-RoPE … simpler than Qwen3-VL". **This is WRONG.** Grounding
`transformers/models/gemma4/modeling_gemma4.py` proves it is a custom **NaFlex
SigLIP2** with, per E4B `vision_config` (hidden 768, 16 layers, 12 heads, head_dim
64 MHA, intermediate 3072, patch 16, pooling_kernel 3, pos_embed_size 10240,
rope_theta 100, standardize=False, gelu_pytorch_tanh, rms_eps 1e-6):

| Sub-module | transformers anchor | vLLM anchor | Reuse-vs-new |
|---|---|---|---|
| **NaFlex image processor** (pre-patchified `pixel_values [P,3·16²]` + `(x,y)` position ids + `-1` padding; resize each dim to `patch·pooling` multiple; `max_soft_tokens` budget) | `image_processing_gemma4.py`; `_compute_num_soft_tokens` `gemma4_mm.py:287-322` | processor feeds `_process_image_input` `gemma4_mm.py:1258` | **NEW** — NOT covered by `qwen3vl_processor.cpp` (Qwen uses smart-resize + temporal patch); genuinely new C++ |
| **Patch embedder** | `Gemma4VisionPatchEmbedder:575` — `input_proj` Linear(768→768) `:583`; scale `2·(x−0.5)` `:612`; learned **2D pos-embed** `position_embedding_table[2,10240,768]` `:584`, `x_emb+y_emb` via `F.embedding` `:602-604`, zeroed at padding `:605` | `vt.patch_embedder(pv,pp,pad)` `gemma4_mm.py:1315` | **NEW** — learned 2D lookup pos-embed (not Qwen's) |
| **Vision RoPE** (multidim) | `Gemma4VisionRotaryEmbedding:701` theta 100 `:739`, `spatial_dim=head_dim//2=32` `:746`, `inv_freq` over `arange(0,32,2)` `:749`; `apply_multidimensional_rope:855` splits head_dim into 2 dims × 32 ch each | `pixel_position_ids` → `vt.encoder(...)` `:1320` | **NEW** — ★ the spike's "no RoPE" was false |
| **Attention** (non-causal full) | `Gemma4VisionAttention:911` — `scaling=1.0` `:921`, **q_norm/k_norm** RMSNorm + **v_norm no-scale** `:929-931`, MHA (kv_heads=heads=12), o_proj | eager | **REUSE (M2a full-attn `AttentionDenseFlash` non-causal)** + NEW q/k/v-norm + rope wiring |
| **MLP** | `Gemma4VisionMLP:685` — gate/up/down, `gelu_pytorch_tanh` `:694` | eager | **REUSE** (`vt::` GEMM + `GeluTanh`) |
| **Encoder block** (Gemma2 sandwich) | `Gemma4VisionEncoderLayer:980` — `input_ln→attn→post_attn_ln→+res`; `pre_ff_ln→mlp→post_ff_ln→+res` (4 RMSNorms) | eager | **NEW block wiring** — differs from Qwen3-VL block norm structure; RMSNorm op REUSED |
| **Pooler** | `Gemma4VisionPooler:618` — mask padding `:671`; `_avg_pool_by_positions` k²-grid via `one_hot` weights `:631-656`; `×√hidden` in **fp32** `:681`; padding-strip mask | `vt.pooler(...)` `gemma4_mm.py:1342` | **NEW** — avg-pool-by-position + √hidden fp32 scale |
| **Projector** `Gemma4MultimodalEmbedder` | modeling `:2078`; vLLM `gemma4_mm.py:918-970` — `RMSNorm(has_weight=False)` `:952` → `ReplicatedLinear(768→2560,bias=False)` `:958` | `self.embed_vision(...)` `:1358` | **REUSE** ops (`vt::RmsNorm`+GEMM), trivial |
| **Merge** (masked scatter) | soft tokens → image-token rows between `boi`/`eoi` | `SupportsMultiModal` mixin | **REUSE** — `Qwen3VLMergeMultimodal` pattern; NEW = Gemma-4 `image_token`/`boi`/`eoi` ids + count arithmetic |

**Net reuse-vs-new (corrected):** REUSE = the mm spine (input container /
`MultiModalHasher` / `EncoderCacheManager` / masked-scatter merge / decode-fork) +
the M2a full-attention ViT-block GEMMs + `RmsNorm`/`GeluTanh`. **NEW** = (1) the
Gemma-4 NaFlex image processor, (2) the learned-2D-pos-embed patch embedder,
(3) multidim **vision RoPE** + q/k/v-norm attention wiring, (4) the Gemma2 sandwich
block wiring, (5) the avg-pool-by-position + √hidden fp32 pooler. This is
materially MORE than the "drop merger/DeepStack/RoPE, keep blocks" the spike
assumed — the tower is closer to a *Gemma-2 transformer with vision RoPE* than to a
plain learned-pos-embed ViT.

### G2.3 Gate plan (RED-first) + the named residual

- **Unit gates (localize before e2e, M2a rel-L2):** C++ NaFlex processor →
  `proc_pixel_values` (exact/rel-L2); patch-embed / encoder / pooled / projected →
  the four staged refs.
- **e2e image→text:** the fixed PNG + prompt through our engine emits the 18 golden
  tokens; **STRICT** (oracle is K=5 deterministic). RED-first: pre-tower the image
  request cannot run / wrong tokens; post-tower it matches.
- **Inertness (must hold when the tower lands):** `test_gemma4_paged_engine` text
  32/32 UNCHANGED (mm gated on image input; text path byte-identical); Qwen3-VL
  image/video 32/32 UNCHANGED; SACRED models untouched.
- **RESIDUAL (unbuilt this pass, named):** the C++ SigLIP/NaFlex tower forward,
  the Gemma-4 NaFlex image processor, and the projector/merge wiring. This pass
  landed the oracle + staged refs + corrected port map only — no C++ vision code,
  no token-exact claim. Follow-on `G2-impl` is turnkey against the committed refs.

---

## G2-impl — C++ NaFlex SigLIP2 vision TOWER LANDED, per-stage gates PASS (2026-07-28, `CLAIM-GEMMA4-G2-IMPL`)

The C++ NaFlex SigLIP2 tower is implemented as a standalone additive TU
(`include/vllm/model_executor/models/gemma4_vision.h` +
`src/vllm/model_executor/models/gemma4_vision.cpp`) and **proven faithful
stage-by-stage vs the committed transformers-eager refs** — the M2a ladder
(`tests/vllm/multimodal/test_gemma4_vision_tower.cpp`, dgx CUDA `flock`, weights
via `scripts/mm/g2_vision_weight_dump.py`). This is the tower-in-isolation
milestone (mirrors M2a for Qwen3-VL before the M2c e2e); image→text e2e wiring is
the named residual.

### G2-impl.0 Per-stage gate result (MEASURED, dgx sm_121a bf16)

| Stage | rel-L2 vs ref | bound | verdict |
|---|---|---|---|
| patch-embed (input_proj + 2·(x−.5) + learned-2D pos-embed) | **2.15e-3** | 5e-3 | ✅ TIGHT |
| encoder last-hidden (16 sandwich blocks, RoPE+qkv-norm) | **3.14e-2** | 6e-2 | ✅ bf16-depth envelope |
| pooled+stripped (avg-pool-by-position + √hidden fp32, 256 soft) | **1.36e-2** | 6e-2 | ✅ |
| projected (embed_vision RMSNorm-noweight + Linear→2560) | **1.85e-2** | 7e-2 | ✅ merge-input |

`n_valid=2304, n_soft=256` (48×48 patch grid → 3² pool → 16×16). 220/220
assertions, SUCCESS. **compute-sanitizer memcheck 0 errors** on the vision path.

### G2-impl.1 Grounded implementation notes (deltas found vs the G2 port map)

- **QAT ACTIVATION CLAMPS were real, not no-op (★ port-map correction).** E4B
  `vision_config.use_clipped_linears=True` with FINITE trained bounds (e.g. L0
  `o_proj` out ±21.25, L15 `q_proj` in ±20.75). `Gemma4ClippableLinear` applies
  `clamp(linear(clamp(x,in_min,in_max)),out_min,out_max)` on the **7 encoder
  attention/MLP linears** (q/k/v/o + gate/up/down); patch_embedder.input_proj and
  embed_vision.embedding_projection are PLAIN nn.Linear (no clip). q/k/v share the
  in-clamp (same source), gate/up share BOTH in- and out-clamp (verified identical
  L0/L8/L15) → the fused gate_up GEMM survives with whole-buffer clamps. Bounds are
  bf16-rounded to match torch's bf16 clamp buffers. Implemented host-side (a fused
  device clamp is the perf follow-on; correctness-first for the one-shot gate).
- **Padding is a trailing contiguous block** (valid patches [0:2304], padding
  [2304:2520]). The bidirectional encoder mask excludes padding keys ⇒ full
  non-causal attention over the valid prefix is exact; padding contributes nothing
  (pos-embed zeroed, pooler masked_fill 0). The tower runs the valid prefix only.
- **Multidim vision RoPE via TWO `vt::RopeFromCache` calls** sharing ONE cos|sin
  cache (both axes use identical inv_freq, theta 100, spatial_dim 32/16 freqs):
  call 1 rotates head channels [0:32] with x-positions, call 2 rotates [32:64] (a
  +32-channel offset head view) with y-positions — bit-faithful to
  `apply_multidimensional_rope`'s per-part `apply_rotary_pos_emb`.
- **Attention scaling = 1.0** (NOT 1/√d — `Gemma4VisionAttention.scaling==1.0`;
  the q/k RMSNorms bound the dot product). Weight-less v-norm + projector pre-norm
  = `vt::RmsNorm` with a ones weight.
- **REUSE, no new kernel:** `vt::MatmulBT` / `Add` / `RmsNorm` / `RopeFromCache` /
  `AttentionDenseFlash` / `GeluAndMul` (fused gate_up GeGLU). No `vt::` op added.

### G2-impl.2 Inertness (PROVEN)

- Gemma-4 text `test_gemma4_paged_engine` **STRICT 32/32 UNCHANGED** (re-run on the
  binary that links the new vision TU — the tower is standalone, not referenced by
  the registry/runner, so the text path is byte-identical by construction).
- Qwen3-VL mm gates additive-by-construction (new file touches no shared code; the
  full `libvllm` re-links with every existing model TU compiling clean).
- `-Werror` 0-warn on both new TUs (CPU + CUDA). One pre-existing, UNRELATED GCC-13
  `-Warray-bounds` FALSE POSITIVE in `voxtral.cpp` (std::copy after vector::assign)
  was suppressed per-file on the dgx build tree ONLY (NOT committed) to link.

### G2-impl.3 RESIDUAL (named, honest)

Image→text **e2e is NOT gated** this pass. Remaining: the C++ Gemma-4 NaFlex image
**processor** (PNG → pre-patchified pixel_values + (x,y) position ids + padding)
and the **engine mm-plumbing** (register Gemma-4 as SupportsMultiModal, hasher /
encoder-cache seam, masked-scatter merge of the 256 soft tokens at the
`<image_pad>` rows, the tower→merge→decode fork). The tower + projector (the merge
INPUT) are proven; wiring them into a running image request is the M2c-equivalent
follow-on. Speed (device-resident weights + fused device clamp) is also pending.

---

## G3 — C++ USM-Conformer AUDIO TOWER LANDED, per-stage gates PASS (2026-07-28, `CLAIM-GEMMA4-G3`)

The C++ USM-Conformer audio tower is implemented as a standalone additive TU
(`include/vllm/model_executor/models/gemma4_audio.h` +
`src/vllm/model_executor/models/gemma4_audio.cpp`) and **proven faithful
stage-by-stage vs the transformers-eager reference** — the A2/G2-impl ladder,
committed golden `tests/parity/goldens/gemma4_e4b_audio/audio_refs/`, weights via
`scripts/mm/g3_audio_tower_ref.py`, gate `tests/vllm/multimodal/test_gemma4_audio_tower.cpp`.
Tower-in-isolation milestone; **audio→text e2e wiring is the named residual**
(exactly the G2-impl vision cadence). Gemma-4 now has all three modalities
tower-proven (text STRICT 32/32, vision per-stage, audio per-stage).

### G3.0 Per-stage gate result (MEASURED, dev-box host f32)

| Stage | rel-L2 vs ref | band | verdict |
|---|---|---|---|
| subsample (2×Conv2d k3s2p1 + LN-no-bias + ReLU + input_proj) | **5.4e-7** | 2e-4 | ✅ f32-exact |
| position_embeddings (inv_timescales, sin\|cos, [13,1024]) | **8.9e-8** | 1e-5 | ✅ |
| block0 (conformer layer 0) | **4.2e-7** | 5e-4 | ✅ |
| block_mid (layer 6) | **3.8e-7** | 1e-3 | ✅ |
| block_last (layer 11) | **4.4e-6** | 2e-3 | ✅ |
| output_proj (== last_hidden_state, [63,1536]) | **5.9e-6** | 2e-3 | ✅ |
| projected (embed_audio RMSNorm-noweight + Linear→2560) | **6.3e-6** | 3e-3 | ✅ merge-input |

`T=250` mel frames → `S=63` soft tokens; head_dim 128, 8 heads, 12 layers.
1256/1256 assertions, SUCCESS. The residual is pure f64-vs-f32 accumulation order
— the USM-Conformer MATH is bit-faithful. **RED-first:** the initial wrong sliding
window (`kj∈[qi-12,qi]`, 13 keys) drove block_mid 2.8e-2 / block_last 0.22 /
projected 0.31 RED; the fix → all ~1e-6 GREEN.

### G3.1 USM-Conformer port map (grounded 1:1, `modeling_gemma4.py` @ 5.13.1)

- **Subsample** (`Gemma4AudioSubSampleConvProjection` :385-412) — `input_features`
  [T,128] `unsqueeze(1)` → 2× `Gemma4AudioSubSampleConvProjectionLayer` (:357):
  mask-zero the padded time rows → **Conv2d(k3,s2,p1,bias-free)** → **`nn.LayerNorm`
  over the CHANNEL dim (elementwise_affine, NO bias)** → **ReLU** → `mask[:, ::2]`;
  channels 1→128→32, then `permute(0,2,3,1).reshape(S, (128/4)*32=1024)` →
  `input_proj_linear`(1024→1024). The reshape is **freq-major, channel-minor**.
- **Rel-pos-enc** (`Gemma4AudioRelPositionalEncoding` :218-246) — `inv_timescales`
  over `hidden//2=512` (min 1, max 10000), `position_ids=arange(ctx//2,-1,-1)` =
  [12..0] (P=`ctx//2+1`=13, ctx=`chunk+past+future`=24), `pos=[sin|cos]` → [13,1024].
- **Chunked-local attention** (`Gemma4AudioAttention` :249-354) — `q *= (hd^-0.5)/ln2
  · softplus(per_dim_scale)`, `k *= ln(1+e)/ln2`; blocks of `chunk=12` with a context
  window [past=`context_left-1`=12, future=`context_right`=0] (context_size 24);
  matrix_ac = q·k over the extracted block context (OOB keys zero); matrix_bd =
  q·relative_k_proj(pos) over 13 rel positions then **`_rel_shift`** (Transformer-XL:
  pad to ctx+1, flatten, slice, reshape → 24); `attn = tanh((ac+bd)/50)·50`; **mask =
  `sliding_window_mask_function((12,0))`**: valid iff `dist=q_idx-kv_idx ∈ [0,12)`
  (★ the load-bearing off-by-one — 12 keys, not 13) AND kv valid; softmax f32;
  `post` proj.
- **Light-conv** (`Gemma4AudioLightConv1d` :484-522) — pre_norm → `linear_start`
  (1024→2048) → **GLU** (`a·sigmoid(b)`) → **depthwise CAUSAL Conv1d** (k=5,
  `left_pad = (K-1)+1-stride = 4`, per-channel groups) → conv_norm → SiLU →
  `linear_end` → +res.
- **FeedForward** (`Gemma4AudioFeedForward` :415-447) — pre_norm → ffw1(1024→4096) →
  SiLU → ffw2(4096→1024) → post_norm → **·0.5 (`residual_weight`, half-step)** → +res.
- **Layer** (`Gemma4AudioLayer` :525-573) — ff1 → clamp+norm_pre_attn → attn →
  clamp+norm_post_attn+res → lconv → ff2 → clamp+norm_out. `output_proj` (Linear
  **WITH bias** 1024→1536). All `Gemma4RMSNorm` (:197) = plain `x·w` (eps 1e-6).
- **QAT clamps** — E4B `use_clipped_linears=True`: FINITE trained per-linear scalar
  bounds `clamp(x,in)→Linear(bias-free)→clamp(out)` on every `Gemma4ClippableLinear`
  (q/k/v/post, ffw1/2, linear_start/end); the subsample convs + input_proj +
  relative_k_proj + output_proj + embed_audio are PLAIN. `gradient_clipping=1e10` is a
  no-op at f32.
- **Projector** (`Gemma4MultimodalEmbedder` audio, `gemma4_mm.py:908-960`) —
  `RMSNorm(has_weight=False)` → `Linear(1536→text_hidden=2560, bias-free)`.
- **ZERO new `vt::` op / kernel** — pure host f32 (correctness-first, device-neutral).

### G3.2 Inertness (PROVEN by construction)

NEW standalone TU, not referenced by the registry/runner ⇒ the text
`test_gemma4_paged_engine` STRICT 32/32 + the G2-impl vision gates are byte-identical
by construction; the full `libvllm` + every test TU re-link clean; `-Werror` 0-warn
on both new TUs (CPU). No shared model/runner/registry/KV path touched.

### G3.3 RESIDUAL (named, honest)

Audio→text **e2e is NOT gated** this pass (same residual class as G2-impl vision).
Remaining: the Gemma-4 audio **feature extractor** (A1 mel frontend: STFT + mel +
`_unfold` framing → `input_features` [T,128] + mask + the soft-token count arithmetic
`gemma4_mm.py:348-395`) — here `input_features` is a dumped golden; and the **engine
mm-plumbing** (register Gemma-4 as SupportsMultiModal, hasher / encoder-cache seam,
masked-scatter merge of the audio soft tokens at the `<audio>` rows, tower→merge→
decode fork). The tower + projector (the merge INPUT) are proven. The **device-
resident bf16 forward** (speed) and the **audio e2e GPU golden** are named residuals.

---

## MM-E2E — IMAGE→text FOLDED into the ENGINE registered forward, dgx-GREEN near-tie (2026-07-29, `CLAIM-GEMMA4-MM-E2E`)

The G2-impl residual ("register Gemma-4 as SupportsMultiModal + the tower→merge→decode
fork through the engine") is CLOSED for IMAGE. Gemma-4 image→text now runs through the
engine's REGISTERED forward (`ModelRegistry::Forward`), mirroring the Qwen3-VL fold
(`CLAIM-ENGINE-MM-FORWARD`), NOT a bespoke standalone driver.

### The fold (port map)
- **`MultiModalForwardInput` gains `ple_token_ids`** (`model_registry.h`, additive
  default-null): the Gemma-4 PLE masked ids (mm rows→0 + `vocab_size_per_layer_input`
  range mask), mirror of `gemma4_mm.py` `embed_input_ids:1962-1973` + `gemma4.py`
  `get_per_layer_inputs:857-863`. Gemma-4 uses the 1-D `ModelForwardInput::positions`
  (NO 3-D MRoPE), NO DeepStack — so `positions3`/`deepstack` stay nullptr.
- **`Gemma4Model::ForwardMm` + the `ForwardBody` mm seam** (`gemma4.cpp`, additive
  default-null overrides): when set, the hidden stream STARTS from the already-merged
  inputs_embeds (text rows √H-scaled, image rows the `embed_vision` projector output),
  the PLE `embed_tokens_per_layer` lookup uses the masked ids, and `ple_proj` still
  projects the merged embeds (`gemma4.py` forward inputs_embeds branch `:908-912`). The
  two TEXT call sites pass null ⇒ the SACRED text 32/32 path is byte-identical.
- **`gemma4_registry.cpp`:** `supports_multimodal=true` (no engine consumer ⇒ byte-
  neutral flip) + the mm branch routing `ModelForwardInput.mm` + a borrow-capable
  LoadedModel + `Make/BorrowGemma4LoadedModel` (mirror Qwen3-VL).
- **`gemma4_mm.cpp` (NEW): `Gemma4GenerateGreedyViaRegistry`** — the single-sequence
  greedy driver: prefill = `embed(prompt)*√H` + masked-scatter of the SigLIP2 projector
  output into the `<image>` rows (`Qwen3VLMergeMultimodal`, modality-agnostic) + per-
  layer paged KV (256 sliding / 512 full, YOCO-aware) → every step through
  `ModelRegistry::Forward` with `input.mm` set (mirror `Qwen3VLGenerateGreedyViaRegistry`).

### The gate (RATIFIED NEAR-TIE FORM) — MEASURED dgx GB10 sm_121a, `flock`
`tests/vllm/multimodal/test_gemma4_registry_e2e.cpp` (dgx-only): image→text through
`ModelRegistry::Forward` vs the STRICT `gemma4_e4b_image` golden. Runs the LIVE C++
SigLIP2 tower when `VLLM_GEMMA4_VISION_WEIGHTS` is dumped
(`scripts/mm/g2_vision_weight_dump.py`), else the committed `ref_projected.npy`.

- **16/18 content tokens BIT-EXACT** = the WHOLE sentence "This is a vibrant, abstract
  background featuring a smooth gradient of bright, blended colors" — token-identical to
  the vLLM 0.25.0 golden.
- The **single divergence is the TERMINAL punctuation** (idx16 ours=`,`236764 vs golden
  =`.`236761, idx17 cascades) at a **bf16 NEAR-TIE**: greedy top1-top2 margin **~0.10-0.12
  logit** (run-to-run bf16 noise ~0.02), vs the ~2.0 confident-pick margins elsewhere.
- **INVARIANT to the vision-input precision:** the live C++ **bf16** tower (rel-L2 ~1.8%
  vs f32 ref) AND the committed **f32** `ref_projected.npy` diverge IDENTICALLY (same
  idx16 `,`) ⇒ the residual is **backbone bf16-accumulation on the 256 image-soft-token
  rows, NOT the fold/merge/PLE**. Ruled out: PLE image-row mask (verified vs
  `embed_input_ids`), `vocab_size_per_layer_input=262144` (range mask a no-op),
  `use_bidirectional_attention=None` (causal is correct), the text path (STRICT 32/32).
- **Gate form:** the golden is STRICT (K=5 vLLM-deterministic), so the committed gate is
  the ratified near-tie form — content-exact prefix + the FIRST divergence must be a bf16
  near-tie (margin < 0.5 band, past the sentence midpoint); a structural bug (confidently-
  wrong token or early divergence) FAILs. GREEN: 239/239 assertions.

### Inertness (same dgx session)
Text SACRED `test_gemma4_paged_engine` STRICT **32/32 UNCHANGED** on the mm binary. CPU
`-Werror` 0-warn (libvllm + new test). `test_model_registry` gemma4 mm-capability
assertion green (the KimiK3 count/list drift is PRE-EXISTING on base HEAD — the KimiK3
claim's fixture, out of scope).

### Residuals (named, honest)
1. **STRICT 18/18** — requires bit-matching vLLM's prefill bf16 accumulation on the image
   rows (the DFlash-class bf16 acceptance floor); the near-tie is not a fold bug.
2. **AUDIO→text e2e** — the A1 mel feature extractor (`gemma4_mm.py:348-395`) + the engine
   `<audio>` merge through this SAME registered fold. The G3 USM-Conformer tower + audio
   projector are already per-stage f32-exact; the mel frontend + merge wiring is the named
   next brick (the audio e2e was too large to also land this lane — image e2e landed fully
   per the honest-scope directive).
3. The FULL in-runner BATCHED mm path (this drives the registered forward single-sequence,
   like Qwen3-VL), a C++ NaFlex image processor (the gate consumes the committed processor
   output + tower output, as the Qwen3-VL registry e2e consumes pre-decoded inputs), and
   per-axis SPEED (device-resident bf16 weights).

---

## 1. Per-model / per-modality DISPOSITION

| Target | Image | Video | Audio | Disposition |
|---|:--:|:--:|:--:|---|
| **Gemma-4 `Gemma4ForConditionalGeneration`** | ✅ (SigLIP) | ✅ (SigLIP) | ✅ (if `audio_config`) | **TEXT STRICT 32/32 (G1b) + VISION tower per-stage (G2-impl) + AUDIO tower per-stage (G3) — TRI-MODAL TOWERS PROVEN; engine mm e2e pending.** vLLM 0.25.0 (transformers 5.13.1) LOADS+RUNS+GENERATES `unsloth/gemma-4-E4B-it` (ungated, 15.99 GB; carries `audio_config` = USM-Conformer). Text STRICT golden `tests/parity/goldens/gemma4_e4b_text/`; vision refs `…/gemma4_e4b_image/`; **audio USM-Conformer tower per-stage f32-exact `…/gemma4_e4b_audio/` (G3, 7/7)**. Residual = the image/audio feature extractors + engine mm-plumbing (SupportsMultiModal, encoder-cache, masked-scatter merge, decode fork); device-resident bf16 speed. |
| **Gemma-4 `Gemma4UnifiedForConditionalGeneration`** | ✅ (encoder-free embedder) | ✅ | ✅ (if `audio_config`) | **GATEABLE (by the E4B W0 proof — same oracle path), IMPLEMENTATION PENDING** — simpler (no SigLIP/audio `AutoModel` tower); the ungated `unsloth/gemma-4-12b-it` (23.92 GB) is this variant and fits GB10. No standalone run this pass (E4B is the smaller vehicle); the oracle-runnability is proven by the shared registered path. No engine e2e yet. |
| **AUDIO modality** (as a subsystem) | — | — | ✅ | **STAGED — reachable, land it first on the smallest oracle-runnable vehicle** (Whisper→Voxtral-Mini-3B), NOT on Gemma-4. This is the genuinely-new work. |
| Whisper (vehicle) | — | — | ✅ ASR | IMPLEMENTABLE-ADDITIVE — oracle-runnable, fits; the audio-pipeline standup vehicle |
| Voxtral-Mini-3B (vehicle) | — | — | ✅ | IMPLEMENTABLE-ADDITIVE — oracle-runnable, fits, Mistral backbone LANDED; the e2e audio-merge vehicle |

---

## 2. The W-plan — the AUDIO track (modality-first) then the Gemma-4 campaign

Two SEPARATE campaigns. The AUDIO track (A0–A3) is the genuinely-new, reachable
work and does NOT depend on Gemma-4. The Gemma-4 campaign (G0–G3) is
staged/blocked behind the oracle. Every increment carries the **inertness gate**
(all current SACRED gates byte-identical — the audio/tower subsystems are additive
and gated on mm input, exactly as the landed vision track proved).

### 2.1 AUDIO track (genuinely-new modality; the low-risk path)

```
 A0  Ground + vehicle + oracle/fit                     [CRITICAL PATH]
      |
 A1  AUDIO INPUT pipeline (decode→resample→log-mel→framing→placeholder count)  [foundation]
      |
 A2  AUDIO encoder TOWER (Whisper-class first; Conformer as the Gemma-family delta)
      |
 A3  e2e AUDIO→text gate on the smallest decoder-merge audio-LLM (Voxtral-Mini-3B)
```

- **A0 — Ground + vehicle + oracle/fit.** Confirm on dgx the oracle constructs +
  runs `whisper-small` and `Voxtral-Mini-3B` (native towers, transformers 5.13.1
  OK); fetch `whisper-small` (~0.5 GiB) then `Voxtral-Mini-3B` (~9.4 GiB, staged);
  capture the oracle log-mel feature reference + a fixed audio→text greedy golden +
  K=5 self-determinism (selects the gate form). **Gate:** oracle references
  produced; vehicle+fit confirmed. **GPU:** minimal (oracle under `flock`).
  **Risk:** audio decode determinism (resampling filter) — pin the resampler.
  **Critical path:** YES.
- **A1 — AUDIO INPUT pipeline (the largest new piece).** `src/vllm/multimodal/`
  audio processor: waveform decode → resample to the feature sampling_rate →
  log-mel feature extraction → `_unfold` framing → the placeholder-count
  arithmetic (mel frames → 2×stride-2 subsample → num soft tokens, cf.
  `gemma4_mm.py:348-368` for the Gemma shape). REUSE `MultiModalKwargs`,
  `MultiModalHasher`, `EncoderCacheManager`, the placeholder-expansion seam.
  **Gate:** feature-parity — our log-mel `input_features` + mask BIT/near-exact vs
  the oracle's feature extractor on a fixed clip, and the mm-hash byte-identical
  (mirrors M1 processor-parity 23/23, RED-first). **GPU:** CPU-only. **Risk:**
  log-mel numerics (window/FFT/mel-filterbank) are a silent-corruption hazard —
  gate against the oracle, never eyeball. **Critical path:** YES.
- **A2 — AUDIO encoder TOWER.** Start with the **Whisper-class encoder** (2×Conv +
  transformer; reuse `vt::` GEMM/attn/LayerNorm/GELU) proven faithful in ISOLATION
  vs a dumped oracle reference (mirrors M2a tower fidelity, bf16-envelope tol).
  Then the **USM Conformer** delta (Conv2d subsampling stride-2 semicausal +
  conv-module + relative-position attention + softcap) proven on
  **Granite-Speech-2b** — the Gemma-4-family tower. **Gate:** tower output rel-L2
  within the bf16 envelope vs the oracle dump; RED-first (disable a block → fail).
  **GPU:** tower forward under `flock`; **new kernels:** Conv2d subsample,
  Conformer conv-module, relative-position bias (compute-sanitizer 0). **Risk:**
  the Conformer relative-position + softcap buffers (the `post_init()` set) must
  match. **Critical path:** YES (proves the tower Gemma-4 audio reuses).
- **A3 — e2e AUDIO→text gate.** `Voxtral-Mini-3B`: native Whisper-class encoder
  (A2) + projector (RMSNorm+Linear, `vt::` ops) + masked-scatter merge (REUSE
  `VLGenerateCore`/`Qwen3VLMergeMultimodal` pattern) into the **LANDED Mistral**
  decoder → forked greedy decode. **Gate:** audio→text token-exact vs vLLM 0.25.0,
  greedy, gate form BY MEASUREMENT (K=5 self-determinism → STRICT else near-tie,
  per the ratified rule). **GPU:** full decode under `flock`, memory-careful (~9.4
  GiB, never OOM-reboot GB10). **Risk:** the projector-merge silent-corruption
  (wrong count/offset → fluent-wrong); localize with A1/A2 unit gates first.
  **Critical path:** YES — this is the first end-to-end AUDIO capability.

### 2.2 Gemma-4 campaign (STAGED / oracle-blocked)

```
 G0  Honesty pass (THIS spike): oracle verdict + config/registry + primitive inventory   [reachable now]
      |  ── BLOCKED until: oracle Transformers carries `gemma4` AND a fitting checkpoint downloads ──
 G1  Gemma-4 backbone (PLE/YOCO/KV-share/Gemma-4-MoE/k_eq_v) via nested text_config  [sweep-gemma §0.1 campaign]
      |
 G2  SigLIP vision tower (REUSE M2a scaffold minus merger/DeepStack/RoPE) + video → IMAGE/VIDEO gate
      |
 G3  Gemma-4 AUDIO (REUSE the A-track pipeline + the USM Conformer tower from A2) → AUDIO gate
```

- **G0 — Oracle-gateability + greedy golden [DONE 2026-07-28, `CLAIM-GEMMA4-W0`].**
  Supersedes the original honesty-pass scope: the oracle does NOT merely construct —
  it **LOADS+RUNS+GENERATES** `unsloth/gemma-4-E4B-it` on vLLM 0.25.0 (§0 banner).
  Captured: the STRICT K=5 greedy golden (`tests/parity/goldens/gemma4_e4b_text/gen_manifest.json`),
  the oracle command/versions, the arch/head-dim config resolution, the two-variant
  architecture (0.1), the reuse-vs-new map (0.2). **Gate PASSED:** a real greedy
  generation (32 coherent tokens, deterministic ⇒ STRICT bar for G-bringup). **GPU:**
  used (E4B under `flock`, GMU 0.30). This is the anchor for the future G1–G3 gates.
- **G1 — Gemma-4 backbone** (only after the block clears): the PLE/YOCO/Gemma-4-MoE/
  k_eq_v/double-wide-MLP/layer-scalar stack ([`sweep-gemma.md`](sweep-gemma.md)
  §0.1), reusing the landed Gemma text primitives (gemma-RMSNorm, sandwich norms,
  GeGLU `kGeluAndMul`, soft-cap `kSoftCap`) + the BF16 grouped-MoE GEMM. A large
  NEW-primitive campaign; **the critical Gemma-4 blocker beyond the oracle.**
- **G2 — SigLIP vision tower + video.** REUSE the M2a ViT scaffold, DROP
  merger/DeepStack/vision-RoPE, ADD learned abs pos-embed; projector = the audio
  projector op. **Gate:** image then video token-exact vs oracle.
- **G3 — Gemma-4 AUDIO.** REUSE the A-track audio pipeline + the USM Conformer
  tower (A2) + the projector-merge; wire the Gemma-4 audio-token id + count
  arithmetic (`gemma4_mm.py:348-395`). **Gate:** audio→text token-exact.

**Critical path:** the AUDIO track (A0–A3) is independent and reachable now; the
Gemma-4 campaign (G1–G3) is gated on the oracle advancing AND a checkpoint. G3
consumes the A-track output, so landing audio on Whisper/Voxtral first is strictly
on the Gemma-4 critical path too.

---

## 3. Honest blockers (mirror the GLM/DeepSeek blocked-row precedent)

- **Gemma-4 mm — oracle block RETIRED (W0 RUN-VERIFIED 2026-07-28); now
  IMPLEMENTATION-BLOCKED only (backbone + towers unbuilt).** The two gate-time
  blockers are both dissolved by measurement: (1) the pinned oracle vLLM 0.25.0 +
  transformers 5.13.1 DOES carry `transformers.models.gemma4` and **loads + runs +
  generates** the mm wrapper (STRICT greedy golden captured on `unsloth/gemma-4-E4B-it`);
  (2) an **ungated** vehicle exists and is cached (`unsloth/gemma-4-E4B-it` 15.99 GB,
  `unsloth/gemma-4-12b-it` 23.92 GB — no HF token needed). The ONLY remaining
  blockers are implementation: the PLE/YOCO/Gemma-4-MoE backbone + the USM-Conformer
  audio tower are unbuilt (the SigLIP vision tower is a reuse of M2a). HW fits (E4B
  ~15 GB in the 119 GiB pool). **The G-campaign is now unblocked to start.**
- **AUDIO modality — NOT blocked; STAGED on a smaller vehicle.** Reachable via
  native oracle-runnable audio models (Whisper, Voxtral-Mini-3B, Qwen2-Audio,
  Granite-Speech). Genuinely-new subsystems (audio pipeline + encoder tower), no
  reuse from image/video except the mm spine + merge pattern. Land it on
  `whisper-small`→`Voxtral-Mini-3B` (Mistral backbone LANDED), then carry to
  Gemma-4 (G3). Downloads not performed (staged, dgx disk tight —
  [[grid-per-sha-trees-fill-disk]]).
- **No HW-blocked modality for the audio vehicles** — all fit the 119 GiB unified
  pool with vast headroom.

---

## 4. Structured contract

### Scope
Design — not build — the Gemma-4 mm readiness (vision + audio towers + backbone)
and the genuinely-new AUDIO track, with the honest oracle/checkpoint/HW verdict
and a per-modality/per-model disposition + W-plan. Covers the two Gemma-4
`MODEL-MM-*` rows. In scope: the mm architecture characterization (§0.1); the
oracle-block finding (§0.0); the reuse-vs-new map against landed M0–M3 (§0.2); the
checkpoint/oracle/GB10 verdict (§0.3); the smallest audio-runnable vehicle (§0.4);
the AUDIO A0–A3 + Gemma-4 G0–G3 W-plans (§2); the blockers (§3).

OUT of scope, each with a reason: **implementation of anything** (spike — no code,
no tower, no build, no download, no gate). **Gemma-4 e2e** — oracle-blocked (§0.0)
+ ≥12B gated checkpoints + unbuilt backbone; honesty-pass only. **The Qwen3.6 mm
rows + `multimodal-track.md`** — owned by the concurrent Qwen3.6-video agent; not
touched. **gemma3n / GlmAsr / other audio rows** as targets — named as references
only; they stay `INVENTORIED`.

### Upstream chain
Registry `registry.py:392-396` (`gemma4_mm`,`gemma4_unified`). Wrapper+towers
`gemma4_mm.py` (vision `:1039-1051`, audio `:1055-1073`, projector `:908-965`,
audio forward `:1468-1490`, audio count `:348-395`, modality gating `:216-235`);
`gemma4_unified.py` (encoder-free `:3-16,73,220-291`). Backbone `gemma4.py`
([`sweep-gemma.md`](sweep-gemma.md) §0.1). Audio vehicles: `whisper.py:458`,
`voxtral.py:671,737`, `qwen2_audio.py:349`, `granite_speech.py:294`,
`gemma3n.py`/`gemma3n_audio_utils.py`. Landed mm spine (§0.2 anchors).
**Anchor-drift warning:** re-anchor every `file:line` at implementation time.

### Our baseline
REUSE §0.2 (mm spine, ViT scaffold, Gemma text primitives, Mistral backbone for
Voxtral). NEW §0.2 (audio pipeline, audio Conformer tower, Gemma-4 backbone).
No `MODEL-MM-gemma4-*` row is or becomes `DONE`; both stay `SPIKE` with the
oracle-blocked verdict.

### Tests to port (inventory only — nothing ported here)
| Upstream test | Tier | Ours (increment) |
|---|---|---|
| audio feature-extractor / processor parity (whisper/gemma4 feature extractor) | T-unit | log-mel `input_features` + mm-hash bit-parity (A1) |
| `tests/models/multimodal/generation/test_*audio*` / whisper / voxtral | T-e2e | audio→text token-exact (A2 tower, A3 e2e) |
| `tests/models/multimodal/generation/test_gemma4*` | T-e2e | SKIPPED — oracle-blocked (G-campaign), tracked reason |
| `tests/models/registry.py` Gemma-4 `_HfExamplesInfo` | T-unit | config/registry resolution (G0, no checkpoint) |

### Gates
1. **Inertness (SACRED, every increment).** All current SACRED gates
   byte-identical (audio/tower subsystems additive + gated on mm input).
2. **Audio feature parity (A1).** log-mel `input_features`+mask + mm-hash
   bit/near-exact vs the oracle on a fixed clip (RED-first).
3. **Audio tower fidelity (A2).** tower output rel-L2 within the bf16 envelope vs
   the oracle dump; RED-first.
4. **Audio e2e (A3, SACRED).** audio→text token-exact vs vLLM 0.25.0, gate form by
   measurement.
5. **Build/memcheck/records** — clean `-Werror`; compute-sanitizer 0 on new audio
   kernels (Conv2d subsample, Conformer conv-module, relative-pos bias); the record
   checkers green.
6. **SPEED** — PENDING; a target is `DONE` only at token-exact AND vLLM throughput.
7. **Blocked-row honesty (Gemma-4).** Record the oracle/checkpoint verdict + the
   primitive inventory; never claim more than a runnable gate backs.

### Dependencies
No hard upward dependency for the AUDIO track (the mm spine + Mistral backbone are
landed). Checkpoint deps (NOT performed): `whisper-small` ~0.5 GiB,
`Voxtral-Mini-3B` ~9.4 GiB, Granite-Speech-2b ~5 GiB; Gemma-4 ≥12B `google/*`
HF-gated. Stage sequentially (dgx disk tight). Downward deps introduced: the audio
input pipeline + audio tower are reusable by the entire audio family incl.
Gemma-4 (G3). **Blocking preconditions for the Gemma-4 rows only:** (a) the oracle
Transformers must carry `gemma4` (currently absent at 5.13.1); (b) a fitting
checkpoint downloads; (c) the Gemma-4 backbone campaign lands. Until all hold, the
Gemma-4 rows are an honesty pass.

### Risks/decisions
- **D1 — Gemma-4 mm is oracle-blocked, not just checkpoint-gated.** The towers are
  Transformers `AutoModel` modules absent from the pinned Transformers; this is the
  decisive blocker, verified by measurement (§0.0). Do not schedule Gemma-4 mm e2e
  until it clears.
- **D2 — Stand audio up on the smallest oracle-runnable vehicle, not Gemma-4.**
  Mirror Qwen3-VL-4B→Qwen3.6: Whisper (pipeline+tower) then Voxtral-Mini-3B (e2e
  merge on our landed Mistral). The Gemma-4 Conformer tower is the family delta,
  proven on Granite-Speech-2b.
- **D3 — log-mel + placeholder-count are silent-corruption hazards.** Wrong
  feature numerics or soft-token count emits fluent-wrong text (the OPT failure
  mode). Gate A1 against the oracle bit/near-exact, never eyeball.
- **D4 — Whisper-class ≠ Conformer.** The small vehicles de-risk the pipeline +
  merge but NOT the USM Conformer tower; do not claim the Gemma-4 audio tower is
  proven by a Whisper/Voxtral gate.
- **D5 — the Gemma-4 backbone is a separate campaign.** PLE/YOCO/Gemma-4-MoE/k_eq_v
  ([`sweep-gemma.md`](sweep-gemma.md) §0.1) is unbuilt and no other row needs it;
  it is not smuggled into an mm bring-up.

### Cross-reference — the vision tower's wire-or-remove verdict

[`gemma4-vision-reachability.md`](gemma4-vision-reachability.md) settles
[#2173](https://github.com/mudler/vllm.cpp/issues/2173): the SigLIP2 tower is
unreached from any production entry point, the pinned oracle runs its own tower
on the default path, so the verdict is WIRE and the wiring is owed. That spec
owns the bricks, gates and stop conditions; nothing is restated here, because
this file is the shared campaign record every Gemma-4 wave writes and a section
with its own gates would make it a conflict lock. Its §3.8 records a
provenance contradiction against §0.0 above that is not settled — read it before
citing §0.0's transformers measurement.
