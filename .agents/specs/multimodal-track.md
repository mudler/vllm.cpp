# SPIKE: Multimodal track (Audio / Video / Image) — Gemma-4 + Qwen3.6

**STATUS 2026-07-25: M0-M3 LANDED (image + video). The design below is the
original spike and is preserved as reference.** Landed: the mm input pipeline
(M0/M1), the Qwen3-VL vision tower (M2a), the MRoPE/DeepStack text backbone
(M2b/M2c), Qwen3-VL-4B image + video e2e, and Qwen3.6-27B image (M3-b) + video
(M3d) STRICT 32/32 vs vLLM 0.25.0. The audio track (A0-A3) is also up (Voxtral-Mini-3B
e2e 14/14). Speed and Gemma-4 remain. Original spike scope: READ-ONLY design +
oracle/checkpoint availability check, grounding the user's TOP `roadmap_v1` priority
(2026-07-25): *"Multimodal support Audio/Video/Image with Gemma-4 and Qwen3.6"*,
ahead of GPU-arch expansion, model expansion, KV-to-disk, LMCache. Framing memory:
[[multimodal-is-top-priority-gate-models-already-mm]].

**Base:** `72f9fb1` (`origin/main`, SPEC-MTP I7 landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` =
vLLM **0.25.0**. **Claim:** `CLAIM-MULTIMODAL-TRACK`.
**Gold-standard spike shape mirrored:** [`sweep-gemma.md`](sweep-gemma.md),
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (the BLOCKED-row
honesty precedent), [`mtp-spec-decode.md`](mtp-spec-decode.md) (the
campaign-decomposition + falsifiable-per-brick pattern).

Rows this spike advances:
- `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` (Qwen3.6-27B) — stays
  `PARTIAL` (text-only), narrative advanced with the mm-completion plan.
- `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` (Qwen3.6-35B) — same.
- `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` — `INVENTORIED` →
  `SPIKE` (BLOCKED-for-now honesty verdict).
- `MODEL-MM-gemma4-unified-gemma4-unified-for-conditional-generation` — same.

Named but LEFT at their current state (the vision family this UNLOCKS, not this
spike's targets): every other `MODEL-MM-*` row (Qwen2.5-VL, Qwen3-VL, GLM-4V,
InternVL, gemma3_mm, gemma3n_mm, …). Qwen3-VL-4B is named below as the first
*vehicle* but its row is not advanced until M2 owns it.

---

## 0. Headline findings (lead with the high-leverage framing)

### 0.0 Our two production GATE models are ALREADY multimodal — brought up text-only

The MVP gate models are not text architectures we would *add* vision to; they are
multimodal architectures we shipped with the vision half switched off:

- **Qwen3.6-27B** = `Qwen3_5ForConditionalGeneration`
  (`qwen3_5.py:389`), which **subclasses `Qwen3VLForConditionalGeneration`**
  (`qwen3_5.py:81-82,389`) and instantiates the **`Qwen3_VisionTransformer`**
  (`qwen3_vl.py:519`) over modalities `{"image", "video"}`
  (`qwen3_5.py:412-414`).
- **Qwen3.6-35B-A3B** = `Qwen3_5MoeForConditionalGeneration`
  (`qwen3_5.py:604`), same vision tower and modalities (`qwen3_5.py:624-626`).

So **completing Qwen3.6 multimodal completes models we already ship + benchmark**
(text-gen token-exact 235/235 and 315/315 vs vLLM 0.25.0). The vision half is the
*same* `Qwen3_VisionTransformer` used by the whole Qwen3-VL family, so the same
tower port unlocks Qwen3-VL-{2B,4B,8B,30B-A3B} as a side effect. The PRIMARY
vehicle question is therefore: **what does completing Qwen3.6-27B image+video
take, given we already own its text path + GDN-hybrid backbone?**

### 0.1 The oracle CAN construct the multimodal path (0.25.0), but our cached gate checkpoints are TEXT-ONLY

Two independently-measured gating facts (dgx, 2026-07-25):

1. **vLLM 0.25.0 (`~/venvs/vllm-oracle`) ships the multimodal model files** —
   `qwen3_5.py`, `qwen3_vl.py`, `qwen2_5_vl.py`, `qwen2_vl.py`, `gemma4_mm.py`,
   `gemma4_unified.py`, `gemma3_mm.py`, `gemma3n_mm.py` are all present in
   `site-packages/vllm/model_executor/models/`. So the SACRED oracle for the mm
   path EXISTS for every target here — this is a stronger starting position than
   Gemma-4-text had (there the worry was oracle construction; here the files are
   present). **Actual mm-forward run** on each is a W0/M0 verification, not
   assumed.
2. **Our cached NVFP4 gate checkpoints do NOT contain the vision tower.** Read
   the safetensors headers directly (metadata only, files already cached):
   - `unsloth/Qwen3.6-27B-NVFP4`: `architectures=["Qwen3_5ForConditionalGeneration"]`,
     `config.json` DOES declare a `vision_config` (depth 27, hidden 1152,
     out_hidden_size 5120, `deepstack_visual_indexes`, `spatial_merge_size`,
     `temporal_patch_size`), **but the weights contain 2111 tensors, ZERO named
     `visual.*`** — only `model.language_model.*`, `lm_head`, `mtp.*`. The unsloth
     NVFP4 quant is TEXT-ONLY (which is exactly why we could bring it up text-only).
   - `nvidia/Qwen3.6-35B-A3B-NVFP4`: same — `Qwen3_5MoeForConditionalGeneration`,
     `vision_config` declared, vision weights absent.
   - **No vision/VL checkpoint of any family is cached on dgx** (`ls ~/.cache/
     huggingface/hub` — only `wavlm-base-plus`, an unrelated audio model).

   **Consequence:** Qwen3.6 mm is NOT hardware-blocked and NOT oracle-blocked —
   it is **CHECKPOINT-blocked** until a vision-inclusive checkpoint is fetched
   (the full-precision Qwen3.6 release, or a mm-inclusive quant). M0 owns that
   fetch + the matching-checkpoint oracle run. The vision tower itself is tiny
   (depth 27 × hidden 1152 ≈ 0.5–0.7 B params, ~1–1.4 GiB bf16) — it fits the
   119 GiB GB10 unified pool trivially alongside the 27B/35B LLM.

### 0.2 Per-target modality — which of Audio/Video/Image is actually reachable

| Target | Image | Video | Audio | Vision tower | Oracle file (0.25.0) | Checkpoint on dgx | GB10 fit |
|---|:--:|:--:|:--:|---|:--:|:--:|---|
| **Qwen3.6-27B** (`Qwen3_5ForConditionalGeneration`) | ✅ | ✅ | ❌ | `Qwen3_VisionTransformer` (DeepStack) | ✅ `qwen3_5.py`+`qwen3_vl.py` | text-only quant cached; **vision-inclusive needed** | tower ~1 GiB + 27B — **FITS** |
| **Qwen3.6-35B-A3B** (`Qwen3_5MoeForConditionalGeneration`) | ✅ | ✅ | ❌ | same | ✅ | text-only quant cached; vision-inclusive needed | tower ~1 GiB + 35B MoE — **FITS** (per landed text run) |
| **Gemma-4** (`Gemma4ForConditionalGeneration` / `Gemma4Unified…`) | ✅ | ✅ | ✅ | SigLIP-class + **audio tower** | ✅ `gemma4_mm.py`,`gemma4_unified.py` | **none cached**; all public ckpts ≥12B, mm-wrapped, `google/*` HF-gated | 12B fits; 26B-A4B/31B HW-marginal-to-blocked (sweep-gemma §0.6) |
| **Qwen3-VL-4B** (first vehicle, `Qwen3VLForConditionalGeneration`) | ✅ | ✅ | ❌ | **`Qwen3_VisionTransformer` (identical to Qwen3.6)** | ✅ `qwen3_vl.py` | none cached (~9 GiB bf16 download) | **FITS trivially** |
| Qwen2.5-VL-3B (fallback vehicle, `Qwen2_5_VLForConditionalGeneration`) | ✅ | ✅ | ❌ | Qwen2.5-VL ViT (older, no DeepStack) | ✅ `qwen2_5_vl.py` | none cached (~7 GiB) | FITS |

**Audio honesty:** among the two named model targets, **only Gemma-4 does audio**
(image+video+audio; `gemma4_mm.py` wires an audio/ASR frontend). **Qwen3.6 has NO
audio path at all.** Audio is therefore reachable ONLY through the Gemma-4 lift
(or the separate `gemma3n_mm`/`*Speech*`/`*ASR*` families), and it needs a
genuinely NEW subsystem (audio encoder + feature frontend + audio preprocessing)
that no image/video work produces. It is deferred to M5, behind M4's Gemma-4
backbone, and called out as a large separate lift, not a near-term gate.

### 0.3 The first vehicle: **Qwen3-VL-4B-Instruct** (stands up the EXACT tower Qwen3.6 needs)

`Qwen/Qwen3-VL-4B-Instruct` (`Qwen3VLForConditionalGeneration`,
`tests/models/registry.py:1293-1294`) is the smallest genuinely-gateable vehicle
that de-risks the target directly: it instantiates the **same
`Qwen3_VisionTransformer`** (DeepStack, `qwen3_vl.py:519`) that
`Qwen3_5ForConditionalGeneration` reuses, is image+video, is oracle-runnable
(0.25.0 has `qwen3_vl.py`), is ~9 GiB bf16 (fits GB10 with vast headroom), and its
LLM backbone is a plain Qwen3-VL **dense text** decoder — a near-sibling of our
landed Qwen3-dense. So the ONLY new subsystem on the first gate is the vision +
input pipeline; the LLM half is already ours. Once M2 gates Qwen3-VL-4B image,
M3 reuses the tower **verbatim** and re-wires it onto the landed Qwen3_5
GDN-hybrid backbone for the actual Qwen3.6 target. Fallback vehicle
(`Qwen/Qwen2.5-VL-3B-Instruct`) is smaller and older but its ViT is NOT the
DeepStack tower — use it only if the Qwen3-VL-4B download is blocked.

### 0.4 Reuse-vs-new — most of the ENGINE is ours; the mm SUBSYSTEMS are all new

**REUSE (landed, unchanged):**
- **Paged attention + KV cache + the model-shape-agnostic runner**
  (`ENG-RUNNER-MODELSHAPE`) — the LLM decode over vision-produced embeds is the
  same paged path we run today; mm only changes how `input_embeds` are populated
  before the first decode.
- **Both LLM backbones for the target** — Qwen3_5 dense + MoE GDN-hybrid text is
  landed and token-exact; Qwen3-VL-4B's dense text is a Qwen3-dense sibling.
- **Tokenizer, sampling, logits path** — unchanged; mm tokens are placeholder ids
  the tokenizer already emits.
- **The LMCache key structure is already mm-AWARE** — our
  `chunked_token_database.{h,cpp}` mirrors vLLM's `ChunkedTokenDatabase` whose
  per-chunk key hashes the 3-tuple `(prefix, tokens, extra_keys)`; we currently
  pass `extra_keys=()` (text-only scope, `KV-EXTERNAL-CACHE` W4). Populating
  `extra_keys` with the mm-hash is a small, localized addition, not a new key
  scheme.

**GENUINELY NEW (nothing in `src/`/`include/` today — grep confirms zero mm
subsystems: no `MultiModalKwargs`, no encoder cache, no vision tower, no pixel
preprocessing):**
1. The **mm INPUT pipeline** — `MultiModalKwargs`, the `BaseMultiModalProcessor`
   (image/video preprocessing → pixel/grid features), placeholder-token
   expansion, and `MultiModalHasher` (mm-hash).
2. The **vision TOWER** — `Qwen3_VisionTransformer` (patch embed, ViT blocks,
   patch merger, DeepStack multi-level injection) + the projector/merge into
   `input_embeds`.
3. The **encoder-cache ENGINE seam** — `EncoderCacheManager`, `EncoderRunner`,
   the scheduler encoder-budget + chunked-prefill-mm hooks.
4. The **serving mm ingestion** — `image_url`/video parsing on the OpenAI server,
   `--limit-mm-per-prompt`.
5. (M5) the **audio encoder + ASR frontend** — Gemma-4/gemma3n only.

---

## 1. The multimodal SEAM MAP — vLLM `file:line` → what we build

### 1.1 Input pipeline (`vllm/multimodal/`)

| vLLM seam | file:line @ `e24d1b24` | What we build |
|---|---|---|
| `MultiModalKwargs` / mm inputs container | `vllm/multimodal/inputs.py` (32 KB) | NEW C++ `MultiModalKwargs` (typed per-modality feature tensors + grid metadata) |
| `BaseMultiModalProcessor` (the input processor) | `vllm/multimodal/processing/processor.py:972`; `apply` `:1663`; `_get_prompt_updates` `:1020`; `_get_mm_fields_config` `:1011`; `_call_hf_processor` `:1097` | NEW processor: HF image/video preprocess → pixel/grid features + placeholder-token expansion |
| `BaseProcessingInfo` | `vllm/multimodal/processing/context.py:296` | NEW per-model processing-info (num-mm-tokens per item, grid sizing) |
| `MultiModalHasher` (mm-hash) | `vllm/multimodal/hasher.py:50` (`serialize_item`) | NEW mm-hash (blake3/sha256 over serialized media) → feeds encoder cache + LMCache `extra_keys` |
| image / video / audio preprocessing | `vllm/multimodal/image.py`, `video.py` (64 KB), `audio.py` | NEW image preprocess (M1); video frame sampling (M3); audio (M5) |
| placeholder-token expansion + merge target | `vllm/model_executor/models/utils.py::_merge_multimodal_embeddings:524` (masked scatter `inputs_embeds[is_multimodal]=…:545`) | NEW `_merge_multimodal_embeddings` (scatter mm embeds into the placeholder positions of `input_embeds`) |

### 1.2 The `*ForConditionalGeneration` wrapper + vision tower

| vLLM seam | file:line | What we build |
|---|---|---|
| `SupportsMultiModal` protocol (`embed_multimodal`, `get_input_embeddings`, `get_language_model`, `get_placeholder_str`) | `vllm/model_executor/models/interfaces.py:94,147,176,390-404` | NEW `SupportsMultiModal` mixin contract on the wrapper |
| Qwen3.6 wrapper | `qwen3_5.py:389-453` (`_mark_tower_model {"image","video"}` `:412`; `self.visual = Qwen3_VisionTransformer` `:413`; merge `:447`) | NEW `Qwen3_5ForConditionalGeneration` wrapper over the LANDED text model |
| Qwen3-VL vision tower | `qwen3_vl.py`: `Qwen3_VisionPatchEmbed:347`, `Qwen3_VisionBlock:413`, `Qwen3_VisionPatchMerger:467`, `Qwen3_VisionTransformer:519`, DeepStack `:543-608,821-840` | NEW vision tower (patch embed, ViT blocks, merger, DeepStack) — reuse `vt::` GEMM/attn/norm |
| image/video feature → embeds | `qwen3_vl.py::_process_image_input:2143`, `_process_video_input:2165`, `embed_multimodal:2731`, `get_input_embeddings`/forward `:2843` | NEW per-modality feature-extraction glue |
| tower weight loading | `qwen3_vl.py::load_weights:2905`, vision-tower `load_weights:843` | NEW `visual.*` weight map (added to the landed text loader) |

### 1.3 Engine seams (scheduler / encoder cache / runner / serving)

| vLLM seam | file:line | What we build |
|---|---|---|
| `EncoderCacheManager` (allocate/free/budget/mm-hash) | `vllm/v1/core/encoder_cache_manager.py:17` (`check_and_update_cache:94`, `can_allocate:123`, `allocate:184`, `get_freed_mm_hashes:255`, `compute_mm_encoder_budget:269`) | NEW encoder-cache manager (keyed by mm-hash) |
| scheduler mm hooks | `vllm/v1/core/sched/scheduler.py:205-228` (budget wiring), `:616,1004-1011` (allocate), `:1356-1467` (mm-hash schedule + chunked-mm), `:1905-1935` (free) | NEW scheduler encoder-budget + per-item scheduling + free path |
| chunked-prefill × mm placeholders | `scheduler.py:1405-1420` (`disable_chunked_mm_input`, don't split a mm item) | NEW chunked-prefill mm-placeholder guard |
| `MultiModalBudget` | `vllm/multimodal/encoder_budget.py` | NEW encoder-budget sizing |
| `EncoderRunner` (execute encoder, gather embeds, splice) | `vllm/v1/worker/gpu/mm/encoder_runner.py:13` (`prepare_mm_inputs:35`, `execute_mm_encoder:52`, `gather_mm_embeddings:64`, `get_inputs_embeds:148`) | NEW worker-side encoder runner + `EncoderCache` |
| model-runner mm wiring | `vllm/v1/worker/gpu/model_runner.py:181-186,315,668-673,737-754` | NEW: build encoder cache + reset/remove-request hooks (inert when `supports_mm_inputs=False`) |
| serving `image_url`/video ingestion | `vllm/entrypoints/chat_utils.py:726` (`MultiModalDataDict`), `:873,955` (`parse_image`, `fetch_image`) | NEW OpenAI-server mm content parsing |
| `--limit-mm-per-prompt` + mm config | `vllm/config/multimodal.py` | NEW mm config + CLI/server flag |

### 1.4 mm-hash ↔ LMCache

`MultiModalHasher` (`hasher.py:50`) produces the per-item mm-hash that keys the
encoder cache AND flows into the KV-cache key as `extra_keys`. vLLM's
`ChunkedTokenDatabase` (LMCache) hashes `(prefix, tokens, extra_keys)`; our
landed `src/vllm/v1/kv_offload/lmcache/chunked_token_database.{h,cpp}` already
carries the `extra_keys` slot (currently `()`, text-only). Populating it with the
mm-hash is the ONLY LMCache change the mm track needs — the key scheme is
unchanged.

---

### 1.5 Multimodal CONFIG and input limits — the seam this map missed (#607, 2026-08-13)

The seam map above covers the input *pipeline* and the tower, but not the
**config that decides whether either runs**. That gap surfaced from the
`recipes.vllm.ai` sweep: `--language-model-only` is used by **43 of 157** official
recipes, and we reject it.

| vLLM seam | file:line @ `555967922` | What we build |
|---|---|---|
| `MultiModalConfig.limit_per_prompt` | `vllm/config/multimodal.py:81` | NEW per-modality input-count limits on the model config |
| `get_limit_per_prompt(modality)` | `vllm/config/multimodal.py:321-336` (returns **0** when `language_model_only`, else the map, else the 999 default) | NEW accessor; the single place every consumer asks |
| `language_model_only` | `vllm/config/multimodal.py:78` | NEW flag — **sugar**, see below |
| `--limit-mm-per-prompt` | `vllm/engine/arg_utils.py:556,1279,1692` | serve flag — **the primary one**; the limits are the mechanism |
| `--language-model-only` | `vllm/engine/arg_utils.py:555,1276,1691` | serve flag over the boolean |
| **`validate_num_items` — what makes a limit a limit** | `vllm/multimodal/processing/context.py:409-428` — raises `VLLMValidationError("At most {limit} {modality}(s) may be provided in one prompt.")`, appending `" Set --limit-mm-per-prompt to increase this limit."` when the MODEL supports more (`:425-426`) | NEW — the **enforcement** point; without it a limit is a number nothing reads |
| its two call sites | `context.py:461` inside `parse_mm_data` (`:430`), and `vllm/entrypoints/chat_utils.py:662` per tracked item, whose only escape is `enable_mm_embeds` + a `*_embeds` modality at limit 0 (`chat_utils.py:653-660`) | NEW — mirror both, including that escape |
| `allowed_mm_limits` | `vllm/multimodal/processing/context.py:392-405` — folds the USER limit with the model's own `supported_mm_limits` by `min()` | NEW — a user limit never raises a model's ceiling |
| further limit consumers | `vllm/multimodal/registry.py:126` (every supported modality at limit 0 ⇒ mm processing disabled, unless `enable_mm_embeds`); `vllm/v1/worker/encoder_cudagraph.py:139` (`video` limit 0 ⇒ `max_frames_per_batch = 0`) | NEW — the limits reach past the tower |
| **tower skip when all limits are 0** | `vllm/model_executor/models/interfaces.py:293` — builds the tower inside `no_init_weights(..., StageMissingLayer)` when `all(get_limit_per_prompt(m) == 0)` | the memory win; a CONSEQUENCE of zero limits, reachable by either flag |
| kernel gate | `vllm/model_executor/models/qwen3_next.py:325` — `text_only` feeds `use_fused_qk_norm_rope_gate` | our fused path is currently unconditional; see below |
| LoRA interaction | `vllm/lora/model_manager.py:233` | deferred with `LORA-RUNTIME` |

**The correction that matters.** `--language-model-only` is **not** a
"skip the encoder" boolean. Its own docstring is explicit: *"disables all
multimodal inputs by setting all modality limits to 0. Equivalent to setting
`--limit-mm-per-prompt` to 0 for every modality."* The encoder skip falls out of
`interfaces.py:293` because the limits are zero — **any** route to zero limits
gets it. So porting the boolean alone would be a bespoke path that does not exist
upstream, which the mirror rule forbids: port `limit_per_prompt` +
`get_limit_per_prompt` first, and the flag becomes three lines on top.

**The consequence that matters more: upstream `--language-model-only` REFUSES
every multimodal request.** This was missing from the first draft of this
section, which mapped the memory consequence and the kernel gate but not the
enforcement. Follow the chain: `get_limit_per_prompt` returns **0** for every
modality (`multimodal.py:321-327`), so `validate_num_items` computes `limit = 0`
and raises `VLLMValidationError` for any request carrying one or more items
(`context.py:409-428`), on both the `parse_mm_data` path (`:461`) and the OpenAI
chat path (`chat_utils.py:662`). The user-visible behaviour of the flag is not
"the same server, minus some VRAM" — it is a server that answers an image
request with *"At most 0 image(s) may be provided in one prompt."*

That is what makes the limits the mechanism and the flag the sugar, and it is
the half a port can most easily leave out, because omitting it breaks nothing
that a text-only workload would notice.

**Our baseline was nothing** (as of 2026-08-13, before L1): `grep -rn
'limit_per_prompt\|MultimodalConfig' src/ include/` returned no hits; no
multimodal config surface existed, and nothing gated tower construction on
config. This is a port, not an exposure. L1 has since landed the config and the
refusal (see the wave list below); tower construction is still ungated by
config, which is L3.

**A second-order consequence worth naming.** `qwen3_next.py:325` proves the flag
is not purely a memory knob — upstream uses it to select the *fused* QK-norm+RoPE+
gate path. Ours takes the fused route unconditionally (`qwen3_5.cpp`, "true for
BOTH the 27B and the 35B"). That asymmetry is the serving-side twin of
[#414](https://github.com/mudler/vllm.cpp/issues/414), which found our published
ratios flattered because the oracle ran unfused while we ran fused. Any gate
comparing the two arms must set the flag on both sides or state that it did not.

**Waves** (additive to §3; none blocks M1's pipeline work):

- **L1** — `limit_per_prompt` + `get_limit_per_prompt` on the model config, with
  upstream's precedence exactly: `language_model_only` ⇒ 0, else the explicit map,
  else 999 — **and the refusal that gives those numbers effect**: mirror
  `allowed_mm_limits` (`min()` against the model's supported limits) and
  `validate_num_items`, message text included, at both call sites, with
  upstream's `enable_mm_embeds` escape. Unit-gated, no serve surface yet. The
  refusal belongs here, not in L3: it is the limits' own semantics, and a limit
  nothing enforces is not a limit.

  **LANDED 2026-08-13 (#607, `row/mm-limits-l1`).** `include/vllm/config/multimodal.h`
  carries `vllm::MultiModalConfig` with the three ported fields and
  `GetLimitPerPrompt`, mirroring `multimodal.py:78,81,98,321-336` — the config
  directory is a 1:1 mirror of `vllm/config/<name>.py`, so `multimodal.h` is the
  only home that is not bespoke. The enforcement is
  `include/vllm/multimodal/processing/context.h` + `src/…/context.cpp`:
  `BaseProcessingInfo::{AllowedMmLimits,ValidateNumItems,ValidateParsedMmData,
  ValidateTrackedChatItem}`, mirroring `processing/context.py:392-405,409-428,
  441-461` and `chat_utils.py:630-662` including the `enable_mm_embeds` escape in
  both of its spellings. The refusal throws `vllm::v1::InputValidationError` —
  the type `api_server.cpp:185,252` already maps to HTTP 400, which is what
  upstream's `VLLMValidationError` gets; a second refusal class would have landed
  a too-many-images request as a 500. That type moved from
  `v1/engine/input_processor.h` into `v1/engine/validation_error.h` (same name,
  same namespace, no behaviour change) so the multimodal layer can throw it
  without including the input processor and without a header cycle at L2's call
  site. Gated `test_multimodal_config` 7/7 (21 assertions) +
  `test_processing_limits` 19/19 (78 assertions); both suites port the upstream
  parametrizations verbatim from `tests/multimodal/test_processing.py:902-941,
  944-985`, `tests/entrypoints/multimodal/llm/test_mm_embeds_only.py:41-49` and
  `tests/entrypoints/unit_tests/test_chat_utils.py:1498-1560`.

  **What L1 deliberately does NOT do, so L2 knows what it inherits.** Nothing
  constructs a `MultiModalConfig` yet and nothing calls the validators on a live
  request: no model config owns one, so `process_inputs_mm`
  (`input_processor.cpp:321,352` — our `context.py:461`) and the chat seam
  (`chat_mm.cpp`, our `chat_utils.py:662`) still validate nothing. Wiring those
  two call sites is L2's, together with the flags and the C-ABI field, because it
  is the config reaching them that the flags exist to set. Separately noted for
  whoever takes L2: `chat_mm.cpp:256-270` takes the FIRST image part and silently
  ignores the rest, so today a 5-image request is neither served nor refused — it
  is quietly truncated to one. That is the divergence L2 closes by calling
  `ValidateTrackedChatItem` per tracked item.
- **L2** — `--limit-mm-per-prompt` and `--language-model-only` serve flags plus
  the C-ABI field, over L1. This is the point at which the 43 recipes stop
  aborting.

  **LANDED 2026-08-14 (#607, #686, `row/mm-limits-l2`).** Three surfaces, and
  the third is the one that makes the other two mean anything:

  1. **The flags** (`server_main.cpp`). `--language-model-only` /
     `--no-language-model-only`, mirroring `arg_utils.py:555,1276,1691` — both
     spellings, because `_compute_kwargs` gives a bool field
     `argparse.BooleanOptionalAction` (`arg_utils.py:346-348`) and a recipe that
     turns the flag off explicitly must not die on an unknown argument.
     `--limit-mm-per-prompt '<json>'`, mirroring `arg_utils.py:556,1279,1692`,
     whose dict type resolves to `type=parse_type(json.loads)`
     (`arg_utils.py:379-381` — the plain-`dict` branch; the `union_dict_and_str`
     branch immediately above at `:374-378` is a different rule and needs a `str`
     arm or a non-builtin type hint, which `limit_per_prompt: dict[str,
     BaseDummyOptions]` does not have because `dict[…]` reports
     `__module__ == "builtins"`) — so the value is a JSON object.
     `ParseLimitMmPerPromptJson` (`src/vllm/config/multimodal.cpp`) ports
     `_validate_limit_per_prompt` (`multimodal.py:212-236`) and the DummyOptions
     dataclasses behind it (`:17-45`): the legacy count-only, the configurable
     `{"count": N, …}` and the mixed spellings all parse; a non-object document,
     a negative count (`count: int = Field(999, ge=0)`, `:21`), and — **for the
     three builtin modalities only** — an unknown per-modality option
     (`extra="forbid"`, `:24,33,41`) or a non-positive one (`Field(None, gt=0)`,
     `:28-30,37-38,45`) is REFUSED before the model load rather than defaulted,
     because a mistyped limit that silently became 999 is a limit that is not
     there. A modality OUTSIDE image/video/audio falls to the bare
     `BaseDummyOptions` at `:233`, the one dummy-options dataclass declared
     without `extra="forbid"` (`:17-21`), so pydantic's default `extra='ignore'`
     applies and its unknown keys are dropped rather than refused — re-derived
     under pydantic 2.12.5 against the pinned declarations:
     `BaseDummyOptions(count=2, foo=3)` → `BaseDummyOptions(count=2)`, while
     `ImageDummyOptions(count=2, foo=3)` raises. We mirror both halves; refusing
     the second would refuse a document upstream accepts (repaired in the #749
     review round — the original L2 landing refused it for every modality).
     The profiling options are validated and then dropped
     (only `.count` feeds `get_limit_per_prompt`, `:335`) and the drop is
     ANNOUNCED per key. **NAMED RESIDUAL:** upstream's dotted spelling
     (`--limit-mm-per-prompt.image 2`) is a `FlexibleArgumentParser` feature
     (`argparse_utils.py:389-425`) that applies equally to `--kv-transfer-config`
     and `--speculative-config`, which this server also takes as JSON only.
     Adding it for one flag would be the bespoke path; it belongs to a parser
     brick covering all three.
  2. **The C-ABI field** — `vllm_model_params.language_model_only` +
     `.limit_mm_per_prompt`, `VLLM_ABI_VERSION` 18 → **19**, appended so a
     zero-initialised v18 struct is byte-identical. `limit_mm_per_prompt` is the
     same JSON object the flag takes, following the v9 `kv_transfer_config`
     precedent that a dict-valued vLLM flag crosses the ABI as its own JSON
     rather than as a fixed struct of modalities the ABI would then owe forever.
     Malformed input fails `vllm_engine_load` with `VLLM_ERR_INVALID_ARGUMENT`.
     Both land on `EngineParams::multimodal` → `LoadedEngine::mm_config()`, so
     the server flags and the ABI resolve ONE config object per engine and
     cannot drift.
     **What the ABI fields do NOT do, corrected in the #749 review round.** The
     v19 header initially said the fields are "ENFORCED, not recorded" and that
     an engine loaded with `language_model_only` answers a multimodal request
     with `At most 0 image(s)…`. That is true of the OpenAI-server path and false
     of the C ABI: `set_multimodal_chat_fn` has exactly one caller,
     `server_main.cpp`, and `serving_chat.cpp` gates the whole multimodal branch
     on that seam being set — `vllm_chat` / `vllm_chat_stream` never install one.
     On a C-ABI engine the two fields are therefore RECORDED on the config and
     consulted by nothing the ABI can reach: an `image_url` content part parses,
     is dropped, and the request is answered as text. The header now says so, and
     `tests/capi/test_capi.cpp` pins it behaviourally (a `language_model_only`
     engine + an `image_url` body → `VLLM_OK` and a `chat.completion`) so the
     wording cannot go stale on a permanent public contract. Adding a C-ABI
     multimodal REQUEST path is a new capability, not an L2 repair.
  3. **The call site — this is what L2 is for.** `chat_mm.cpp` now calls
     `ValidateChatMmLimits` (the port of the `chat_utils.py:648-662` tracker)
     as step 0 of `MakeQwen3VLImageChatFn`, over a `BaseProcessingInfo` folding
     the engine's config with the seam's own declared ceiling,
     `Qwen3VLChatSupportedMmLimits() == {"image": 1}`. That declaration is the
     answer to the open question #686 left ("the model's own supported limit —
     today nothing declares one"): this seam locates a single image part and
     handles no video or audio, so its honest ceiling is one image and every
     other modality is absent, which `context.py:414-415` reads as limit 0. A
     user limit can only LOWER it (the fold is a `min`), so
     `--limit-mm-per-prompt image=99` still refuses the second image, and the
     refusal then carries no `--limit-mm-per-prompt` hint because raising the
     user's limit would not help.
     **The ceiling is a number, not yet a message (#758).** The L2 landing
     claimed this declaration satisfies AGENTS.md's "an unimplemented arm is
     refused with a message naming the missing piece"; the #749 review found it
     does not. The text a client receives is upstream's generic `At most 0
     video(s) may be provided in one prompt.`, which is indistinguishable from an
     operator having configured that limit. The only present signal is by
     OMISSION — the withheld `--limit-mm-per-prompt` hint. Naming the arm means
     diverging from a verbatim-ported message that three suites assert
     byte-for-byte, so it is owed to #758 with its own spec rather than repaired
     in review. Also corrected there: `get_supported_mm_limits` is not defined on
     `Qwen3VLProcessingInfo`; it is inherited from `Qwen2VLProcessingInfo`
     (`qwen2_vl.py:851-852`, `{"image": None, "video": None}`), which
     `qwen3_vl.py:848` subclasses.

  **#686 is CLOSED by this.** A three-image request is answered
  `400 BadRequestError "At most 1 image(s) may be provided in one prompt."`
  instead of being served with its first image. One correction to the issue's
  own text, found by the RED run and recorded rather than quietly fixed: against
  the production seam the pre-L2 behaviour was not the truncated 200 the issue
  describes but an **HTTP 500** — `MakeQwen3VLImageChatFn` injects one
  placeholder marker per image part while routing only the first image, so
  `ExpandImagePlaceholders` raised "more image placeholders than grids" and the
  client saw a server fault carrying an internal message. The truncated 200 is
  real for the validate-then-build shape and is pinned as its own leg. Both are
  wrong the same way — neither is upstream's refusal — so the issue's diagnosis
  stands and only its consequence was understated.

  **Gates** (aarch64, the `build-test-cpu-arm64` lane; `-DVLLM_CPP_CUDA=OFF`).
  At the #749 REVIEW-REPAIR head, on `kairos-4db2`: clean rebuild 1355/1355
  targets, 0 warnings under `-Werror`; `ctest -j 6` 456/457 with 2 skipped, the
  single failure being `test_op_parity` (#737, reproduced from pristine main);
  `test_serve_mm_limits` 11/11 (109 assertions), `test_chat_mm` 11/11 (126),
  `test_openai_api_server` 56/56, `test_capi` 58/58 (536),
  `test_processing_limits` 19/19 (78), `test_multimodal_config` 7/7 (21).
  Mutations, `cp` + `touch` + rebuild + re-verified green between each and both
  files md5-identical afterwards: `IsBuiltinModality` forced to `true` (the
  pre-repair "forbid extras everywhere") takes `test_serve_mm_limits` to
  10 passed / 1 failed, and wiring an image refusal into `vllm_chat` takes
  `test_capi` to 57 passed / 1 failed on `REQUIRE(1 == 0)`.
  **`test_openai_api_server`'s ASSERTION count is not a pin** — measured 632,
  648 and 651 across three runs of one binary, because SSE-chunk loops assert
  per chunk received. Its CASE count, 56/56, is stable and is the number to
  quote. At the L2 landing the assertion count was recorded as 638.
  RED before the L2 change: `test_chat_mm`
  2 cases failing (`CHECK_THROWS_AS ... threw a DIFFERENT exception: "Expand
  ImagePlaceholders: more image placeholders than grids"` and `FATAL ERROR:
  expected --language-model-only to refuse an image request`), and the HTTP legs
  failing `CHECK(500 == 400)`, `CHECK("InternalServerError" == BadRequestError)`
  and `CHECK(200 == 400)` — the last of those being the truncated 200 itself.

  **What L2 does NOT do, so L3 knows what it inherits.** No memory claim is made
  or implied: nothing gates tower CONSTRUCTION on the limits, so
  `--language-model-only` today refuses multimodal requests and frees nothing.
  That is L3 and it is owed with a MEASURED RSS reduction. Separately owed and
  named rather than assumed: the SECOND call site,
  `process_inputs_mm` (`input_processor.cpp:321,352`, upstream's
  `context.py:461`), is still unwired. It needs the per-model
  `get_supported_mm_limits()` hook that L1 already recorded as absent (the models
  that would implement it are the M2 towers); wiring it before that hook exists
  would mean inventing a supported-limits source inside the engine, which is the
  bespoke path the mirror rule forbids. The chat call site is wired because it
  HAS a concrete source — the seam's own implemented arm. Every path the OpenAI
  server can reach today goes through the wired one.
- **L3** — the tower skip: construct-without-initialising when every limit is 0,
  gated on **measured** RSS reduction against a multimodal checkpoint, plus
  token-exactness of the text path with and without the flag.
- **L4** — mirror the kernel gate, or record an explicit tracked exception with
  the #414 cross-reference.

**L2 is shippable without L3 only because L1 carries the refusal.** The first
draft of this section said "L2 without L3 is honest and shippable: the flag
would be accepted and would correctly zero the limits". That was too generous,
and the caveat it carried guarded the wrong thing — it guarded "frees VRAM"
while leaving a much larger divergence unguarded. Zeroing a limit that nothing
enforces is not correct behaviour: we would ACCEPT an image request that
upstream REFUSES, which is the flag's main observable effect, and the flag would
be inert on exactly the axis a user would test first.

So the honest statement is conditional. **With L1's refusal, L2 ships honestly**
and only the memory win is owed; it must still NOT be described as "frees VRAM"
until L3 lands and is measured. **Without the refusal, L2 is not shippable at
all** — it is a flag that is accepted and inert, which is worse than the abort
it replaces, because an abort is visible and a silently-served image request is
not.

---

## 2. Structured contract

### Scope
Design — not build — the multimodal track (Image → Video → Audio) with the
primary vehicle question framed on completing Qwen3.6-27B/35B (already-shipped
text-only mm architectures), plus a smaller first vehicle (Qwen3-VL-4B) to stand
the track up on, and the honest Gemma-4 verdict. Covers the four rows named in
the header. In scope: the seam map (§1); per-target modality + oracle + GB10-fit
+ checkpoint gateability (§0.2); the reuse-vs-new factoring (§0.4); the ordered
M0–M5 W-plan (§3); per-increment gates, GPU/CPU, risk, critical path; and the
HW/oracle/checkpoint-blocked honesty (§0.1, §4).

OUT of scope, each with a reason: **implementation of anything** (spike — no code,
no tower, no build, no download, no gate). **Audio e2e** beyond a design + honesty
verdict — it needs a NEW audio encoder + ASR frontend reachable only via Gemma-4
(M5), a large separate lift. **Gemma-4 e2e** beyond a characterization/honesty
pass — its only checkpoints are ≥12B mm-wrapped + `google/*` HF-gated and it needs
the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1) PLUS vision PLUS audio
towers; staged in M4. **The rest of the vision family** (GLM-4V, InternVL,
gemma3_mm, …) — unlocked by the track but not this spike's targets; they stay at
their current state.

### Upstream chain
Registry: `qwen3_5.py:389,604` (`registry.py:556,557-560`), `qwen3_vl.py`
(`registry.py:551`), `qwen2_5_vl.py` (`:526-529`), `gemma4_mm.py` (`:392`),
`gemma4_unified.py` (`:393-396`), `gemma3_mm.py` (`:383`). Input pipeline
`vllm/multimodal/` (§1.1). Wrapper + tower `qwen3_5.py`+`qwen3_vl.py` (§1.2).
Engine `vllm/v1/core/encoder_cache_manager.py`, `sched/scheduler.py`,
`v1/worker/gpu/mm/encoder_runner.py`, `model_runner.py`, `entrypoints/chat_utils.py`,
`config/multimodal.py` (§1.3). **Anchor-drift warning:** re-anchor every cited
`file:line` at implementation time.

### Our baseline
REUSE §0.4 (paged path, both landed LLM backbones, tokenizer, sampling, the
LMCache `extra_keys` seam). NEW §0.4 (the five mm subsystems). No `MODEL-MM` row
is or becomes `DONE`; the two Qwen rows stay `PARTIAL` and the Gemma-4 rows move
to `SPIKE` with a BLOCKED-for-now verdict.

### Tests to port (inventory only — nothing ported here)
Per [`.agents/porting.md`](../porting.md):
| Upstream test | Tier | Ours (increment) |
|---|---|---|
| `tests/multimodal/test_processing.py` (processor + placeholder expansion) | T-unit | processor-parity vs vLLM's processor output (M1) |
| `tests/multimodal/test_hasher.py` | T-unit | mm-hash byte-agreement (M1) |
| `tests/v1/core/test_encoder_cache_manager.py` | T-unit | encoder-cache manager (M1) |
| `tests/models/multimodal/generation/test_qwen*_vl.py` (or the pinned equivalent) | T-e2e | image token-exact on Qwen3-VL-4B (M2), Qwen3.6 (M3) |
| `tests/models/multimodal/generation/test_common.py` (video) | T-e2e | video token-exact (M3) |
| `tests/models/multimodal/generation/test_gemma*` | T-e2e | SKIPPED — Gemma-4 (M4) / audio (M5), tracked reason |

### Gates
1. **Inertness (SACRED, non-negotiable, EVERY increment).** With no mm input, all
   current SACRED gates byte-identical (27B 235/235, 35B 315/315, Qwen3-Coder,
   Qwen3-dense, OPT, DeepSeek-V2-Lite, Llama-3.2-1B, Mistral-7B, GLM, Gemma-1/2/3,
   OLMo-2). The mm subsystems are additive + default-inert (`supports_mm_inputs`
   gates every hook).
2. **Processor parity (M1).** Our processor's placeholder-token ids + `mm_kwargs`
   tensor shapes/values match vLLM's `BaseMultiModalProcessor.apply` on a fixed
   image, and the mm-hash is byte-identical.
3. **Image token-exact (M2/M3, SACRED).** Greedy token-exact vs vLLM 0.25.0 on a
   FIXED image+prompt — Qwen3-VL-4B (M2), then Qwen3.6-27B (M3). Gate form by
   measurement per [[near-tie-distributional-gate]] (a vision-conditioned decode
   may sit in a bf16 near-tie band → distributional fallback only if measured).
4. **Video token-exact (M3).** Same, on a fixed short video (frame-sampling
   parity is the risk).
5. **Build / memcheck / records** — clean `-Werror`; `compute-sanitizer` 0 on new
   vision/encoder kernels; the five record checkers green.
6. **SPEED.** PENDING per increment; a target is `DONE` only at token-exact AND
   vLLM throughput on every axis (mm encoder-cache reuse + chunked-prefill mm are
   the throughput levers).
7. **Blocked-row honesty (Gemma-4, audio).** Record the oracle/checkpoint verdict
   and the primitive inventory; never claim more than a runnable gate backs.

### Dependencies
No hard upward dependency on unlanded engine work — the paged path + both LLM
backbones are landed. Checkpoint dependencies (downloads, NOT performed): a
vision-inclusive Qwen3.6-27B/35B checkpoint (our NVFP4 caches are text-only, §0.1);
`Qwen/Qwen3-VL-4B-Instruct` (~9 GiB); `Qwen/Qwen2.5-VL-3B-Instruct` (~7 GiB,
fallback); Gemma-4 (≥12B, `google/*` HF-gated). Stage sequentially
([[grid-per-sha-trees-fill-disk]] — dgx disk tight). Downward dependency this
introduces: the mm input pipeline + encoder cache are reusable by the ENTIRE
vision family; the mm-hash `extra_keys` closes the LMCache mm gap.

---

## 3. The dispatch-sized W-PLAN (M0–M5)

Each increment is independently gateable; the inertness gate (Gate 1) rides on
every one.

```
 M0  Ground + vehicle + oracle/fit/checkpoint         [CRITICAL PATH]
      |
 M1  mm INPUT pipeline + encoder-cache engine seam     [CRITICAL PATH, foundation]
      |
 M2  first vision TOWER + merge on Qwen3-VL-4B -> IMAGE gate   [CLOSED 2026-07-25: STRICT 32/32]
      |
 M3  complete Qwen3.6 IMAGE (reuse M2 tower) -> then VIDEO     [CLOSED 2026-07-25: image STRICT 32/32 (M3-b) + video STRICT 32/32 (M3d)]
      |
 M4  Gemma-4 (staged: vision + PLE/YOCO/MoE backbone; honesty-pass blocked pieces)
      |
 M5  AUDIO if reachable (Gemma-4 / gemma3n audio encoder + ASR frontend)
```

**M0 — Ground the facts; pick the first vehicle; confirm oracle + fit + checkpoint.
[LANDED 2026-07-25, `CLAIM-MULTIMODAL-M1`]** Downloaded `Qwen/Qwen3-VL-4B-Instruct` (8.3 GiB) to
dgx; `scripts/mm/m0_oracle_capture.py` captured the vLLM 0.25.0 `BaseMultiModalProcessor.apply`
reference for a fixed (image, prompt) into committed fixtures
`tests/vllm/multimodal/fixtures/qwen3vl/` (pixel_values bf16 784x1536 sha256 `2c908796...`, grid_thw
[1,28,28], expanded ids 9->204 N=196, mm-hash `ef6f5bea...`). RCA locked the bit-exact pixel
contract: fused rescale+normalize `(raw-127.5)/127.5` -> transformers patchify -> bf16 model-dtype
cast (vLLM casts mm_kwargs to model dtype in `call_hf_processor`).
- Builds: nothing (design + measurement). Confirm (DONE in this spike): our cached
  NVFP4 gate checkpoints are text-only (no `visual.*`); 0.25.0 has the mm model
  files. Remaining M0 work: fetch `Qwen/Qwen3-VL-4B-Instruct`; run the 0.25.0
  oracle on it on a FIXED image+prompt to produce reference tokens; measure the
  vision-tower param count + peak-mem fit; decide the vision-inclusive Qwen3.6
  checkpoint source for M3.
- Gate: oracle mm reference outputs produced; vehicle selected; fit measured;
  checkpoint plan recorded. GPU: minimal (one oracle run under `flock`).
- Hardest risk: the Qwen3.6 vision-inclusive checkpoint may only exist as a large
  bf16 release (no mm-inclusive NVFP4) → M3 loads a bf16 tower alongside the NVFP4
  LLM (mixed-precision load) or downloads more.
- Critical path: YES (unblocks everything).

**M1 — The mm INPUT pipeline + encoder-cache engine seam (the foundation, inert
without mm input). [LANDED 2026-07-25, `CLAIM-MULTIMODAL-M1`]** Built `src/vllm/multimodal/`
(MultiModalKwargs/FeatureSpec/Inputs, MultiModalHasher blake3, Qwen3VLImageProcessor
smart_resize+normalize+patchify, ExpandImagePlaceholders) + `EncoderCacheManager`+budget; additive
inert `mm_features` on Request/EngineCoreRequest; `extra_keys` seam on ChunkedTokenDatabase. Gate 2
(processor parity) PASS 23/23 BIT-identical vs the M0 oracle (RED-first proven); encoder-cache 32/32;
Gate 1 (text-inertness) CPU-green STANDALONE + `check-device-leakage` OK; the 27B/35B/Coder SACRED
CUDA gates are the mandatory GPU inertness proof. NO vision tower / embed-merge (that is M2).
Tracked as engine-matrix row `ENG-MM-INPUT-PIPELINE` (`ACTIVE`, `CLAIM-MULTIMODAL-M1`).

#### Port map
`ENG-MM-INPUT-PIPELINE` vLLM `file:line` -> our code:
- `multimodal/inputs.py` -> `include/vllm/multimodal/inputs.h` (MultiModalKwargs/FeatureSpec/Inputs).
- `multimodal/hasher.py:50` + `processing/inputs.py:62` -> `src/vllm/multimodal/hasher.cpp` (blake3 mm-hash, exact byte stream).
- transformers `image_processing_qwen2_vl.py:62` + `image_processing_backends.py:327`; `qwen3_vl.py::_get_prompt_updates:1400` -> `src/vllm/multimodal/qwen3vl_processor.cpp` (smart_resize + fused normalize + patchify + placeholder expansion).
- `v1/core/encoder_cache_manager.py:17` -> `src/vllm/v1/core/encoder_cache_manager.cpp`.
- `Request`/`EngineCoreRequest` mm_features -> `include/vllm/v1/request.h`, `src/vllm/v1/request.cpp`, `include/vllm/v1/engine/types.h` (additive, inert).
- LMCache `extra_keys` -> `include/vllm/v1/kv_offload/lmcache/chunked_token_database.h` + `.cpp` (empty -> byte-identical).

#### Work breakdown
- M0 [DONE]: fetch Qwen3-VL-4B; `scripts/mm/m0_oracle_capture.py` -> committed oracle fixtures.
- M1 [DONE]: image processor + placeholder expansion + mm-hash + MultiModalKwargs; EncoderCacheManager + budget; inert `mm_features` + `extra_keys`; processor-parity gate 23/23 + encoder-cache 32/32; CPU inertness + `check-device-leakage`; SACRED CUDA 27B/35B/Coder.
- M2 [NEXT]: `Qwen3_VisionTransformer` forward + `_merge_multimodal_embeddings` -> image token-exact gate.

#### Risks/decisions
- pixel_values is the bf16 MODEL-DTYPE cast (vLLM `call_hf_processor::_postprocess_output`), NOT the raw f32 processor output — the parity gate compares bf16 (RCA-locked). Decision: C++ computes f32 `(raw-127.5)/127.5` then round-to-nearest-even bf16.
- The bicubic RESIZE path is deferred: M1 uses images already conformant to smart_resize (a genuine resize throws). Risk retired for the gate image (448x448); M2/M3 owns arbitrary sizes.
- `extra_keys` non-empty byte-format vs a full mm LMCache oracle is an M2/M3 gate; the empty (text) path is byte-identical (proven by `test_lmcache_key_agreement`).
- Placeholder expansion is a silent-corruption hazard (wrong count -> fluent-wrong text); gated bit-exact against the oracle, never by eyeball.
- Builds: `MultiModalKwargs`; a `BaseMultiModalProcessor` port (image preprocess +
  placeholder expansion + `_get_mm_fields_config`); `MultiModalHasher`;
  `EncoderCacheManager` + `EncoderRunner` scaffold + scheduler encoder-budget /
  chunked-mm hooks; `_merge_multimodal_embeddings` (masked scatter). All gated on
  `supports_mm_inputs` so text engines are byte-identical.
- Gate: Gate 1 (inertness) + Gate 2 (processor parity: placeholder ids +
  `mm_kwargs` shapes + mm-hash byte-identical vs vLLM on a fixed image);
  encoder-cache-manager unit vs `test_encoder_cache_manager.py`. GPU: mostly CPU
  (preprocessing/hashing/scheduling); no vision compute yet.
- Hardest risk: placeholder-token expansion is a SILENT-corruption hazard (wrong
  count/position emits fluent-but-wrong text, the OPT failure mode) — gate the
  processor output against vLLM, not by eyeball. Chunked-prefill must not split a
  mm item (`disable_chunked_mm_input`).
- Critical path: YES.

**M2 — First vision TOWER + projector/merge on Qwen3-VL-4B → first IMAGE gate.**
DECOMPOSED (2026-07-25) into three independently-gateable bricks after the M2a
scope audit found the LLM side is NOT the landed plain Qwen3-dense (it needs MRoPE
3-D positions + DeepStack decoder injection):
- **M2a — vision TOWER proven faithful (LANDED 2026-07-25, `CLAIM-MULTIMODAL-M2A`, engine-matrix row `ENG-MM-VISION-TOWER`).**
  Built `src/vllm/model_executor/models/qwen3_vl_vision.{h,cpp}` (patch-embed
  matmul+bias; host pos-embed bilinear-interp+reorder; 24 ViT blocks = LayerNorm +
  vision attention with partial-rotary vision RoPE [`vt::RopeFromCache` neox,
  rotary_dim=64, `[cos|sin]` cache] + non-causal varlen `vt::Attention(causal=false)`
  + tanh-GELU MLP; patch merger = LayerNorm + exact-erf-GELU + 2 FCs; DeepStack 3
  post-shuffle mergers at layers 5/11/17 → concat `[196,10240]`). Added 2 additive
  vt ops (`GeluTanh`/`GeluErf`). 4 RED-first unit gates vs the dumped vLLM-0.25.0
  tower reference PASS (patch-embed relL2 2.1e-3, block0 6.8e-3, merger 6.5e-2,
  DeepStack taps, full tower 5.1e-2 — bf16-depth envelope, RCA'd; RED = rope
  disabled → block0 0.149, tower 0.75, 6 fails). Tower proven faithful in
  ISOLATION — NOT the e2e image result. Additive-only (no runner/model TU) → text
  SACRED byte-identical by construction. compute-sanitizer 0.
- **M2b — the Qwen3-VL text backbone. NUMERIC CONTRACTS UNIT-GREEN (LANDED 2026-07-25,
  `CLAIM-MULTIMODAL-M2BC`, engine-matrix row `ENG-MM-TEXT-BACKBONE`).** `src/vllm/
  model_executor/models/qwen3_vl_text.{h,cpp}`: `Qwen3VLGetRopeIndex` (MRoPE 3-D
  `get_rope_index` positions `[3,T]`, `qwen3_vl.py:2567`/`2482`) + established that the
  DeepStack decoder injection (`qwen3_vl.py:1589-1594`) is a plain add of
  `Qwen3VLComputeDeepstack`'s `[L,T,H]` at layers 0/1/2. KEY: the 3-section MRoPE
  APPLICATION needs NO new kernel — it is the EXISTING `vt::RopeFromCache` mrope path
  (positions `[3,T]` + `mrope_section` + `mrope_interleaved`), proven faithful to
  `MRotaryEmbedding.forward_native` for Qwen3-VL's config (interleaved,
  `section=[24,20,20]`, `rotary_dim=128`, `theta=5e6`). Gate
  `tests/vllm/multimodal/test_qwen3vl_text.cpp` — 4 RED-first CPU gates PASS 85/85 vs
  vLLM 0.25.0 (`scripts/mm/m2b_text_ref_dump.py`): get_rope_index BIT-exact, MRoPE
  q/k rel-L2 1.5e-3 (RED interleaved-off >5e-2), DeepStack + merge BIT-exact.
  Additive-only ⇒ text SACRED byte-identical by construction. The FORKED VL decode
  (inputs_embeds + MRoPE + DeepStack inject in a running forward) is part of M2c below.
- **M2c — merge + e2e IMAGE gate. [LANDED 2026-07-25, `CLAIM-MULTIMODAL-M2C`, M2 CLOSED].**
  The forked VL decode is wired and the STRICT e2e image gate PASSES: our full
  pipeline greedy tokens == the committed vLLM 0.25.0 golden **32/32 token-exact**
  on Qwen3-VL-4B (fixture image + "What is in this image?"). Built `src/vllm/
  model_executor/models/qwen3_vl.{h,cpp}`: the VL weight loader (`model.language_model.*`
  onto the landed Qwen3-dense bf16 helpers + `model.visual.*` into the M2a tower) +
  the forked greedy decode (embed text ids → `Qwen3VLMergeMultimodal` scatter of the
  tower merger `[:, :2560]` into image rows → 3-section MRoPE via `vt::RopeFromCache`
  over a global absolute-position cos|sin cache + positions `[3,T]` → DeepStack add
  after decoder layers 0/1/2 → paged greedy, MRoPE decode positions = `idx+delta`).
  Gate `tests/vllm/multimodal/test_qwen3vl_e2e.cpp` (dgx-only) + input-ids fixture
  `scripts/mm/m2c_e2e_inputs.py`. RCA (one bug fixed): the M2a tower's `cap==nullptr`
  DeepStack concat path was an explicit deferred-to-M2c stub (deepstack features were
  only stashed on the capture struct) → completed in `qwen3_vl_vision.cpp` (M2a
  capture output byte-identical). No near-tie needed: the STRICT gate passed exactly,
  first try after the stub fix — the M2a bf16-envelope drift did NOT flip any argmax.
  **M3 next:** Qwen3.6-27B image reusing this tower+backbone+loader on the GDN-hybrid
  backbone, then video. HISTORICAL M2c pre-wire notes below.
- **M2c (pre-wire, superseded by LANDED above).** `Qwen3VLMergeMultimodal` masked scatter
  (`_merge_multimodal_embeddings`, `utils.py:524-545`) of the tower's `[:, :2560]` into
  `input_embeds` is UNIT-GREEN (BIT-exact, landed with M2b). **REMAINING:** the
  `visual.*`/`language_model.*` weight loading (a name-remap over the landed
  `LoadQwen3ForCausalLMWeights` layer helpers + the M2a tower weights) + the forked VL
  decode forward (inputs_embeds instead of embed-from-ids + `vt::RopeFromCache` MRoPE +
  DeepStack add after layers 0/1/2) + a single-seq greedy loop (scaffold =
  `tests/vllm/models/test_qwen3_forward.cpp`) → Gate 3 (image token-exact vs vLLM 0.25.0
  on Qwen3-VL-4B, fixed image+prompt, greedy). **GATE FORM DECIDED BY MEASUREMENT
  (2026-07-25, `scripts/mm/m2c_e2e_golden.py`): STRICT token-exact.** vLLM 0.25.0
  greedy `enforce_eager` on the fixed (image, prompt) is DETERMINISTIC across K=5
  (first_divergence=None) → the M2c e2e gate is STRICT, not near-tie. The golden is
  captured + committed (`tests/vllm/multimodal/fixtures/qwen3vl_text/gen_tokens_i32.bin`,
  32 greedy tokens, sha256 `3ec5f2b7…`; the model correctly reads the random-noise
  fixture as "a noise pattern / static"). This is the first end-to-end image→text
  proof once wired. RISK: the M2a tower is bf16-envelope faithful (rel-L2 ~5e-2) and
  the gate is STRICT, so any tower/merge/MRoPE numeric drift that flips a near-tie
  argmax fails the gate — localize with the M2b unit gates (all green) before blaming
  the tower.
- Hardest risk: DeepStack multi-level injection; vision-tower attention numerics +
  vision RoPE (M2a: PROVEN faithful); MRoPE positions (M2b); the placeholder-merge
  silent-corruption (M2c).
- Critical path: YES (proves the tower Qwen3.6 reuses).

**M3 — Complete Qwen3.6-27B IMAGE (reuse M2 tower verbatim), then VIDEO.**

**M3-W0 (LANDED 2026-07-25, `CLAIM-MULTIMODAL-M3`) — the GATING FACTS, MEASURED on dgx.**
- **Vision-inclusive checkpoint FOUND + FITS + downloaded.** The real repo is
  **`Qwen/Qwen3.6-27B`** (HF, NOT gated, `language_model_only:false`): 51.7 GiB
  **fully-bf16** (LLM + vision both bf16 — NOT the "NVFP4-LLM + bf16-vision mixed"
  layout the spike guessed; the checkpoint is uniform bf16), 15 safetensors, 1199
  tensors of which **333 `model.visual.*`** are present + `preprocessor_config.json`
  + `video_preprocessor_config.json`. dgx free was 55 GiB (99% full) → reclaimed
  mine-only `~/work/{mixed-batch,mm-m0m1-cuda,spec-i3,wt-marlin-ctmp-pool}` (~42 GiB;
  apex/darwin_36b_opus untouched) → 95 GiB free → downloaded. GB10 fit: 54 GiB bf16
  weights + ~1.4 GiB bf16 tower + KV in the 119 GiB unified pool (measured 115 GiB
  available, 4 GiB used) — FITS with headroom; oracle `gpu_memory_utilization` kept
  low, run ALONE, never OOM-rebooted.
- **The 27B vision config DIFFERS from Qwen3-VL-4B — parametrize, do not hardcode.**
  From `Qwen/Qwen3.6-27B/config.json`: vision **depth 27** (vs 24), **hidden 1152**
  (vs 1024), **out_hidden 5120** (== 27B text hidden, vs 2560), **num_heads 16**,
  **intermediate 4304**, num_position_embeddings 2304, patch_size 16,
  spatial_merge_size 2, temporal_patch_size 2, `hidden_act` gelu_pytorch_tanh, and
  **`deepstack_visual_indexes: []` EMPTY** — Qwen3.6-27B has **NO DeepStack**
  (confirmed in the vision-inclusive checkpoint, not a quant-stripping artifact; the
  `model.visual.*` tensor list has NO `deepstack_merger_list.*`). So M3 REUSES the
  M2a `Qwen3VLVisionForward` VERBATIM with a 27B `Qwen3VLVisionConfig` (empty
  deepstack ⇒ tower output `[N,5120]`, no multiscale concat, NO decoder-layer
  DeepStack injection — SIMPLER than the 4B). IMAGE_TOKEN=248056, VIDEO_TOKEN=248057,
  vision_start=248053.
- **MRoPE config (27B):** `rope_parameters.mrope_section=[11,11,10]`,
  `mrope_interleaved=true`, `partial_rotary_factor=0.25` (head_dim 256 ⇒ rotary_dim
  **64**), `rope_theta=1e7`. (vs 4B's section=[24,20,20], rotary_dim 128, theta 5e6
  — so the M2b MRoPE application is REUSED with the 27B section/rot/theta, no new
  kernel.) Text backbone: 64 layers = **48 linear_attention (GDN) + 16 full_attention**
  (`full_attention_interval:4`), hidden 5120, 24 heads / 4 kv, head_dim 256,
  `attn_output_gate:true`, vocab 248320, `tie_word_embeddings:false` (owns lm_head).
- **The bf16 GDN-hybrid LLM loader ALREADY EXISTS.** `LoadQwen3_5Dense`
  (`qwen3_5_dense_weights.cpp:369`) routes each Linear bf16-vs-NVFP4 by the presence
  of `.weight_packed`; on `Qwen/Qwen3.6-27B` (no `.weight_packed`) it loads every
  projection bf16 (`LoadBf16RawNK`/`LoadMergedBf16RawNK`) into the SAME
  `Qwen3_5DenseWeights` the existing forward reads, and the GDN-hybrid forward's
  bf16 path (`fp8→fp4→bf16` fallback, exercised by the GGUF/synthetic loaders) runs
  it. So M3 needs NO new LLM loader — only the `model.visual.*` vision loader (the
  M2c `LoadQwen3VLWeights` vision half, verbatim, with the 27B vision config).
- **Oracle golden + gate form:** `scripts/mm/m3_oracle_capture.py` (fixed fixture
  image 448x448 + "What is in this image?" via the chat template) captured the vLLM
  0.25.0 greedy `enforce_eager` golden + K=5 self-determinism + the
  placeholder-expanded model input ids (image span 196 tokens) → committed fixtures
  `tests/vllm/multimodal/fixtures/qwen3_5_27b/`. **MEASURED VERDICT:** vLLM
  CONSTRUCTS + LOADS + RUNS the 27B mm path (resolves `Qwen3_5ForConditionalGeneration`,
  loads 15 bf16 shards, initializes the encoder cache + profiles 1 image item) — NOT
  oracle-blocked. Input = 214 tokens / **196 image tokens** at offset 4; the greedy
  32-token golden is **K=5 DETERMINISTIC (first_divergence=None) ⇒ GATE FORM = STRICT**
  (sha256 `ead4b484…`; coherent image-conditioned text). GB10 held the 54 GiB bf16
  model + encoder + KV (GMU 0.6, no OOM-reboot).

**M3-b (LANDED 2026-07-25, `CLAIM-MULTIMODAL-M3B`) — the GDN-hybrid VL forward + STRICT
image gate PASS 32/32. Our own gate model's image path now works end-to-end.**
[engine-matrix row `ENG-MM-QWEN36-VL-FORWARD` `SPIKE`→`ACTIVE`.] BUILT: vision-only
loader `LoadQwen3VLVisionWeights` (`src/vllm/model_executor/models/qwen3_vl.{h,cpp}`,
factored from the 4B `LoadQwen3VLWeights`, reused with the 27B vision config, empty
deepstack ⇒ tower `[196,5120]`); the forked greedy driver `Qwen3_5VLGenerateGreedy`
+ host MRoPE builder `BuildMropeCosSinHost` + a default-null `mrope_cos_sin` param on
`DenseForwardLayers` (`src/vllm/model_executor/models/qwen3_5.cpp`); the STRICT gate
`tests/vllm/multimodal/test_qwen3_5_vl_e2e.cpp`. RESULT: **32/32 token-exact vs the
golden `ead4b484…`** (54/54 assertions); text-inertness re-run cutlass-ON **27B
235/235, 35B 315/315, Coder 138/138** (the mm path is gated on mm input ⇒
`mrope_cos_sin==nullptr` on every text caller = byte-identical); clean `-Werror` 0
warn; compute-sanitizer on the 27B VL forward. The MRoPE injects a per-token `[T,64]`
cos|sin cache (interleaved 3-section selection baked in) into the fused
`AttnQkNormRopeGate` preamble (default-ON), NOT a new kernel. Weights via
`LoadQwen3_5Dense(shards,cfg,&queue)` direct device load + host release (no
unified-pool OOM). VIDEO = M3c (owed); SPEED pending. Original design (as-built):
Reuse:
M2a tower (27B config) + M2c vision loader + `LoadQwen3_5Dense` bf16 LLM. NEW: fork
the landed `Qwen3_5DenseModel` GDN-hybrid forward (`qwen3_5.cpp`
`DenseForwardLayers:6383`/`DenseForwardBody:6474`) on THREE gated points, all
default-off so a text-only 27B request stays byte-identical:
  (a) `inputs_embeds` entry — `DenseForwardLayers` already takes an embedded
      `hidden_in`; build it OUTSIDE = embed(prompt_ids) then `Qwen3VLMergeMultimodal`
      scatter of the tower merger `[N,5120]` into the image_token(248056) rows (NO
      deepstack add — empty for 27B);
  (b) MRoPE positions — the full-attn rope reads a per-token `[T,rot]` cos|sin cache
      (`AttnQkNormRopeGate` / `sdi.attn_cos_sin`, built by `MaybeBuildAttnCosSin`
      from 1-D positions and applied ROW-PER-TOKEN, base unused). Inject MRoPE by
      building that same `[T,rot]` cache from `Qwen3VLGetRopeIndex` positions `[3,T]`
      + `mrope_section=[11,11,10]` interleaved (a small host builder mirroring the
      proven M2b interleaved selection) — NO attn-block/kernel change, byte-identical
      for text. GDN (linear_attention) layers have NO rope, so only the 16 full-attn
      layers are touched.
  (c) NO DeepStack (empty `deepstack_visual_indexes`).
The forked VL forward lives IN `qwen3_5.cpp` (it must reuse the anon-namespace GDN
machinery `RunDenseLayerPaged`/`BuildStepDevInputs`, which the M2c plain-dense fork
could re-implement standalone but the GDN-hybrid cannot). A single-seq greedy driver
mirrors `tests/vllm/models/test_qwen27_paged_forward.cpp`'s `KVStatePool` +
`PrefillGdnMeta`/`PrefillAttnMeta` (GDN state cache + attn KV + prefill→decode meta).
- Gate: Gate 3 on Qwen3.6-27B image — **PASSED STRICT 32/32** (our full pipeline greedy
  == the vLLM golden, first run, no near-tie). Flipped the 27B row narrative
  `PARTIAL`(text-only)→IMAGE-e2e-working (correctness; speed pending). Then Gate 4 video (M3c).
- Hardest risk: the MRoPE `[T,rot]` cos|sin layout (interleaved section) must match
  vLLM bit-for-a-near-tie — localize with the M2b unit gates (proven) before blaming
  the tower; the 54 GiB bf16 memory-careful gate (never OOM-reboot GB10).
- Critical path: YES — this is the user's stated target. Text-inertness (27B 235/235,
  35B 315/315, Coder 138/138) is MANDATORY on every increment (the fork is gated on
  mm input).

**M3c (LANDED 2026-07-25, `CLAIM-MULTIMODAL-M3C`, engine-matrix row `ENG-MM-VIDEO-FORWARD`
`ACTIVE`) — VIDEO understanding on Qwen3-VL-4B: preprocessing + full wiring LANDED +
unit-gated; video e2e NEAR-TIE-ROBUST PASS (gate form RESOLVED BY MEASUREMENT 2026-07-25,
`CLAIM-MULTIMODAL-TOWER-FIDELITY`).** The genuinely-new piece is
video PREPROCESSING — the tower already handles temporal patches (`temporal_patch_size=2`)
and MRoPE the temporal axis, both verified. BUILT (all additive to qwen3_vl*/multimodal
TUs; ZERO text-path TU ⇒ text SACRED byte-identical BY CONSTRUCTION — unlike M3-b which
edited `qwen3_5.cpp`):
- **Video processor** `Qwen3VLImageProcessor::ProcessVideo` + `VideoSmartResize`
  (`video_processing_qwen3_vl.py:35`) + `ComputeVideoTimestamps` (`qwen3_vl.py:975`) +
  `BuildVideoRepl` (`get_video_repl:1479`) + `VideoKwargs`
  (`src/vllm/multimodal/qwen3vl_processor.cpp`, `include/.../inputs.h`). Video patchify
  mirrors transformers `video_processing_qwen3_vl.py:249` reshape/permute: rows C-order
  `[grid_t,Gh,Gw,mh,mw]`, cols `[c,tp,ph,pw]`, **source frame = grid_t_idx*temporal_patch_size
  + t (2 REAL frames per patch-row, NOT the image duplicate)** — the one video-specific
  subtlety. `BuildVideoRepl` owns the timestamp-interleaved expansion (per frame
  `[ts_ids]+vision_start+video_token*Nf+vision_end`).
- **Tower per-frame windowed attention** (`qwen3_vl_vision.cpp`): vLLM cu_seqlens per frame
  (`qwen3_vl.py:744`, dumped `[0,64,128,192,256]`) ⇒ frames never attend across each other;
  looped `vt::Attention` over grid_t windows; grid_t==1 (image) == byte-identical single
  window. pos-embed + vision RoPE already loop over t.
- **Video MRoPE** `Qwen3VLGetRopeIndexVideo` (`qwen3_vl_text.{h,cpp}`): per-frame scan
  (vision_start→video_token→vision_end) mirroring `_iter_mm_grid_hw`+`_get_mrope_input_positions`;
  timestamp/marker tokens get sequential text positions, each frame's video tokens grid
  positions off the running max.
- **e2e video driver** `Qwen3VLGenerateGreedyVideo` (`qwen3_vl.{h,cpp}`): refactored the image
  driver into a shared `VLGenerateCore`; video wrapper builds the mask on video_token_id +
  positions via the video get_rope_index. Image driver byte-identical (proven by the image gate).

GATES (Qwen3-VL-4B, vLLM 0.25.0 oracle, dgx GB10 arch 121a, cutlass+FA2 banner, clean
`-Werror` RC=0):
- Oracle `scripts/mm/m3c_video_oracle_capture.py` on a fixed 8×128×128 synthetic video
  (do_sample_frames=false): grid_thw `[4,8,8]`, pixel_values_videos `[256,1536]` (precast-bf16
  == production, same contract as image), timestamps `[0.25,1.25,2.25,3.25]`, 64 video tokens,
  K=5 DETERMINISTIC ⇒ **GATE FORM STRICT**, golden 32 tokens sha256 `9f78f8a0…`. Committed
  `tests/vllm/multimodal/fixtures/qwen3vl_video/`.
- **Gate 4 (video-processor UNIT gate, localize-first) `test_qwen3vl_video_processor` — 41/41:
  pixel_values_videos BIT-exact 0/393216 mismatches**, grid + timestamps + interleaved expansion
  exact. **RED-first PROVEN:** image-duplicate frame mapping → 195838/393216 mismatch → FAILURE.
- **Video MRoPE positions BIT-exact vs vLLM** (`scripts/mm/m3c_mrope_check.py`, delta −48).
- **Video TOWER faithful** (never gated before — M2a only tested grid_t==1): per-frame
  windowed-attention output rel-L2 **0.072** vs the dumped vLLM 0.25.0 video tower
  (`scripts/mm/m3c_video_tower_ref_dump.py`), within the bf16 envelope (image ~0.05; tol <0.1).
- **Gate 3/4 (video e2e) `test_qwen3vl_video_e2e`: NEAR-TIE-ROBUST PASS** — gate form selected
  BY MEASUREMENT (`CLAIM-MULTIMODAL-TOWER-FIDELITY`, 2026-07-25), mirroring the ratified
  olmo2/qwen3-dense/glm4 near-tie gates. **The prior RCA above was WRONG: it mislocated the flip
  (claimed token 24 / "noise") and NEVER teacher-forced.** The DECISIVE measurement
  (`scripts/mm/m3c_video_neartie_gap.py` — teacher-force vLLM 0.25.0 on OUR exact sequence with
  the identical video mm input, read per-position gaps): the REAL first divergence is at
  **tok22**, ours ' colorful' (33866) vs vLLM ' static' (1099) at a **0.125-nat** gap — our token
  is vLLM's OWN 2nd choice among 4 tokens tied within 0.25 nats (a textbook bf16 tie), and EVERY
  downstream token (tok23–31) IS vLLM's teacher-forced argmax at gap **0.0000** (the 22/32-vs-
  greedy is the deterministic one-token shift from that single tie). vLLM is fully self-consistent
  on the golden (teacher-forced argmax == golden 32/32) ⇒ a clean STRICT target; our forward
  reproduces vLLM's logits everywhere except the one tie. **Tower-accumulation analysis (the
  "fixable f32 vs bf16" question):** our tower ALREADY accumulates in f32 everywhere — cuBLASLt
  GEMMs use `CUBLAS_COMPUTE_32F` (bf16-in/f32-accum, = vLLM's cuBLAS), the vision attention is an
  online-softmax kernel entirely in f32 (QK dot, softmax stats, output accumulator; stores bf16 =
  vLLM's FlashAttention f32-softmax), and LayerNorm/merger accumulate in f32. So rel-L2 0.072 is
  the IRREDUCIBLE inter-op bf16 rounding envelope (kernel reduction-order differences rounded to
  bf16 between ops), NOT a fixable numeric choice ⇒ **NO kernel change** (the correct path per the
  DATA; doing tower work would be unnecessary). Gate: anchor `our_ids_i32.bin` +
  `neartie_gap_mnats_i32.bin`, PASS iff all gaps ≤ 0.5 nats (max 0.125 << 0.5). **NOT a loosened
  STRICT gate** — it is the measured near-tie equivalence class, identical in form to the text
  near-tie gates. VIDEO understanding now WORKS e2e (correctness complete; speed pending).
- **Gate 1 (inertness): NO REGRESSION — image e2e 4B STRICT 32/32** (the deterministic strict-pass
  proof — same 4B, same tower, rel-L2 0.05; the tower windowing [identical for grid_t==1] + driver
  refactor are byte-identical on image); CPU units image-processor 23/23, video-processor 41/41,
  text 85/85. Text SACRED (27B/35B/Coder) byte-identical BY CONSTRUCTION — the tower-fidelity
  resolution touched ONLY the video TEST + a new script + 2 fixtures (zero src/kernel/shared-op),
  so a text-SACRED re-run is not required and compute-sanitizer is N/A (no kernel changed).
- Critical path: YES (video is part of the user's stated Qwen modalities). NEXT: 27B-video (reuse
  the 4B video path on the GDN-hybrid backbone) + speed; then Gemma-4 (M4).

**M3d (LANDED 2026-07-25, `CLAIM-MULTIMODAL-M3D`, engine-matrix row `ENG-MM-QWEN36-VL-FORWARD`)
— VIDEO on Qwen3.6-27B: the GDN-hybrid VL video driver, STRICT e2e gate PASS 32/32. This
COMPLETES the Qwen video modalities on our own gate model (image+video both work e2e).** A pure
REUSE increment — image works on 27B (M3-b), video works on Qwen3-VL-4B (M3c); M3d runs video
through the 27B VL forward. The processor (`ProcessVideo`, bit-exact `pixel_values_videos` +
temporal grid + timestamps + `BuildVideoRepl`), the per-frame windowed tower attention, and
`Qwen3VLGetRopeIndexVideo` were all landed + unit-gated by M3c and are REUSED verbatim (verified,
not modified) — the only new wiring is the 27B video driver.
- **The 27B video driver `Qwen3_5VLGenerateGreedyVideo`** (`src/vllm/model_executor/models/
  qwen3_5.cpp`, decl `include/vllm/model_executor/models/qwen3_5_dense.h`). The M3-b image driver
  `Qwen3_5VLGenerateGreedy` was refactored into a shared **`VLGenerateCoreGdn`** (embed + scatter
  `mm_main` `[N,5120]` into the masked rows → GDN-hybrid prefill/decode with the `[T,64]` MRoPE
  cos|sin cache via `BuildMropeCosSinHost` → paged greedy), mirroring how M3c split the 4B path
  into `VLGenerateCore`. The image and video wrappers differ ONLY in how `mask`/`pos3_prefill`/
  `delta` are built: image = `image_token`(248056) mask + `Qwen3VLGetRopeIndex`; video =
  `video_token`(248057) mask across ALL frames + `Qwen3VLGetRopeIndexVideo` (per-frame,
  timestamp-interleaved). NO DeepStack (27B `deepstack_visual_indexes` empty ⇒ tower `[N,5120]`,
  no multiscale, no decoder inject). The GDN backbone, KV/GDN state, MRoPE application, and decode
  continuation are IDENTICAL to the image path — so the image path is byte-identical across the
  refactor (proven by the image gate re-run below). The video driver is purely additive (a new
  function); the shared TEXT forward (`DenseForwardLayers`/`DenseForwardBody`/`DenseEmbedInto`/
  `RunLayer`/`RunDenseLayer` + `Qwen3_5DenseModel::Forward*`) is UNTOUCHED (`git diff --stat`) ⇒
  text SACRED byte-identical BY CONSTRUCTION.
- **Oracle golden + gate form (`scripts/mm/m3d_video_oracle_capture.py`).** Reuses the M3c
  synthetic clip byte-identically (same RNG seed/NF/HW/FPS ⇒ raw-video sha256 `8a111599…` ==
  the M3c fixture) retargeted to `Qwen/Qwen3.6-27B`: grid_thw `[4,8,8]`, 64 video tokens, gen
  input 113 tokens (64 video at offset 10). vLLM 0.25.0 greedy `enforce_eager` K=5 **DETERMINISTIC
  (first_divergence=None) ⇒ GATE FORM STRICT** — golden 32 tokens (coherent video-conditioned
  text: "The user wants me to describe the video…I see a sequence of 6 images…"). Committed
  `tests/vllm/multimodal/fixtures/qwen3_5_27b_video/`.
- **Gate 4 (27B video e2e) `test_qwen3_5_vl_video_e2e`: STRICT PASS 32/32** (near-tie-robust gate
  FORM, mirroring M3c since the tower bf16 envelope is shared — but MEASURED STRICT: our tokens ==
  vLLM greedy golden 32/32, and the teacher-forced near-tie gaps `neartie_gap_mnats_i32.bin` are
  **0.0000 nats at every position** — 0 divergent positions; the near-tie band never engages). The
  full pipeline (C++ `ProcessVideo` → M2a tower 27B config per-frame windowed attn `[64,5120]` →
  merge into video_token(248057) rows → temporal MRoPE `[11,11,10]` on the 16 full-attn layers →
  GDN-hybrid backbone → greedy) == golden, first run (27/27 assertions). Proof it RAN: MESSAGE
  "video token-exact vs vLLM greedy golden: 32/32", ours[:8]==golden[:8]==760,1156,6587,728,310,
  7276,279,2678.
- **Inertness.** 27B IMAGE e2e re-run `test_qwen3_5_vl_e2e` **STRICT 32/32** (54/54) — the driver
  refactor preserved the image path exactly. Text SACRED (27B 235/235, 35B 315/315, Coder 138/138)
  byte-identical BY CONSTRUCTION — the shared text forward is untouched (`git diff --stat`: the
  qwen3_5.cpp change is confined to the VL-only driver region; the video path is gated on mm input
  ⇒ `mrope_cos_sin==nullptr` on every text caller).
- **Build/memcheck.** Clean CUDA `-Werror` 0 warnings (Release, arch 121a, cutlass NVFP4 + FP8 +
  Marlin + FA2 all ENABLED banner); compute-sanitizer memcheck 0 errors on the 27B video forward.
- Critical path: YES — COMPLETES the user's stated Qwen3.6 video modality. Qwen3.6-27B is now
  image+video e2e (audio N/A for Qwen; SPEED pending — the row stays `PARTIAL`). NEXT: speed
  (encoder-cache reuse + chunked-prefill mm); then Gemma-4 (M4, image+video+AUDIO).

**M4 — Gemma-4 (staged: vision tower + backbone stack; honesty-pass the blocked
pieces).**
- Builds (only if the preconditions clear): the Gemma-4 SigLIP-class vision tower
  + the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1 — a large NEW-primitive
  stack no other row uses) via the nested `text_config`. Until a fitting checkpoint
  downloads AND the 0.25.0 oracle runs the mm forward, this is a
  characterization/honesty pass (the row records `SPIKE`/BLOCKED-for-now).
- Gate: Gate 7 (honesty) first; Gate 3 image only after the preconditions clear.
  GPU: heavy (12B mm). 
- Hardest risk: ALL public Gemma-4 checkpoints are ≥12B mm-wrapped +`google/*`
  HF-gated + the PLE/YOCO/MoE backbone is unbuilt; audio tower on top.
- Critical path: NO — parallel/deferred; does not block the Qwen3.6 target.

**M5 — AUDIO if reachable (Gemma-4 / gemma3n).**
- Builds: a NEW audio encoder + ASR/feature frontend + audio preprocessing —
  reachable ONLY via Gemma-4 (or `gemma3n_mm`/`*Speech*`/`*ASR*`); Qwen3.6 has no
  audio path. A large separate lift with no reuse from image/video.
- Gate: Gate 7 honesty; then audio token-exact on the smallest oracle-runnable
  audio+text model that fits. GPU: model-dependent.
- Hardest risk: it is a genuinely new modality subsystem gated behind M4's
  Gemma-4 backbone; honestly deferred until image+video land.
- Critical path: NO — the last, largest, most-deferred lift.

---

## 4. Blocked / deferred honesty (mirror the GLM/DeepSeek blocked-row precedent)

- **Qwen3.6-27B/35B multimodal — NOT blocked, CHECKPOINT-gated.** Oracle present
  (0.25.0 `qwen3_5.py`+`qwen3_vl.py`), tower fits GB10 trivially; the only gate to
  a run is fetching a vision-inclusive checkpoint (our NVFP4 caches are text-only,
  §0.1). Reachable modalities: **image + video** (no audio). This is the primary,
  achievable target.
- **Gemma-4 (`Gemma4ForConditionalGeneration`, `Gemma4Unified…`) — SPIKE /
  BLOCKED-for-now.** Oracle FILE present (0.25.0 `gemma4_mm.py`), but no checkpoint
  cached, all public checkpoints ≥12B mm-wrapped + `google/*` HF-gated, and it
  needs the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1) + vision + audio
  towers. Advanced `INVENTORIED` → `SPIKE`; NOT implemented; M4 is a
  characterization pass until a fitting checkpoint downloads and the oracle runs
  the mm forward.
- **Audio — deferred (M5), reachable only via Gemma-4/gemma3n.** A large separate
  lift (new audio encoder + ASR frontend); Qwen3.6 has no audio. Honestly not a
  near-term gate.
- No HW-blocked modality for the Qwen3.6 target: the vision tower is ~1 GiB and
  fits the 119 GiB unified pool alongside the 27B/35B LLM.
