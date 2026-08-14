# Multimodal serving — wiring image/audio/video into the OpenAI server (`ROAD-V1-MM` serving)

Row IDs: `MM-SERVE-PARSE` (W1), `MM-SERVE-ENGINE` (W2), `MM-SERVE-E2E` (W3, this brick).
Owner claim: `CLAIM-MM-SERVING-E2E` (W1 `CLAIM-MM-SERVING-W1`, W2 `CLAIM-MM-SERVING-W2`).
Pinned vLLM oracle: `${VLLM_SOURCE}` @ `555967922` (0.26.0.dev0).

## Problem

The multimodal INPUT pipeline (image/video/audio → processed features +
placeholder-expanded prompt) is correctness-complete on the single-sequence path
(`MODEL-MM`: Qwen3-VL-4B + Qwen3.6-27B image/video STRICT 32/32, Voxtral audio
14/14). It is **NOT wired into the OpenAI server**: the chat request `content` was
a bare string and multimodal content-part arrays were dropped
(`protocol.cpp` `from_json(ChatMessage)` handled only `content.is_string()`;
`serving_chat.{h,cpp}` deferred "multimodal").

## Ground truth (both sides, file:line)

### Ours — the mm INPUT pipeline that already exists (unwired)
- `include/vllm/multimodal/inputs.h` — `MultiModalInputs` (placeholder-expanded
  `prompt_token_ids` + `mm_features`), `MultiModalFeatureSpec`
  (mm_hash/modality/offset/length + `ImageKwargs`/`AudioKwargs`).
- `include/vllm/multimodal/qwen3vl_processor.h:87` `ProcessImage(rgb,H,W)` →
  `ImageKwargs`; `:113` `ExpandImagePlaceholders(...)`; `:90` `HashImage`. Needs
  only `Qwen3VLProcessorConfig` (no model weights).
- `include/vllm/multimodal/audio_processor.h:50` `DecodeWavPcm16Mono`; `:81`
  `ProcessWaveform` → `AudioKwargs`; `:98` `ExpandAudioPlaceholders`; `:86`
  `HashAudio`. Needs `AudioProcessorConfig` + the golden mel filterbank (no model
  weights).
- How the single-sequence path drives these (the reference): the e2e test
  `tests/vllm/multimodal/test_qwen3vl_e2e.cpp:112-158` — fixture RGB →
  `ProcessImage` → grid `[1,28,28]`/196 → tower → greedy. **Note:** even the
  single-seq e2e consumes PRE-DECODED raw RGB (`image_rgb_uint8_448x448x3.bin`)
  and a PRE-TOKENIZED prompt (`input_ids_i32.bin`); it never decodes PNG/JPEG and
  never tokenizes. The audio parity gate `test_audio_processor.cpp:91-107` decodes
  a committed PCM16 WAV → processor entirely on CPU.
- `include/vllm/v1/core/encoder_cache_manager.h` — the scheduler/encoder-cache
  seam that budgets/allocates per mm item (consumer of `mm_features`).

### The serving gap
- `include/vllm/entrypoints/openai/protocol.h:352` (`ChatMessage`) — content was a
  bare-string T0; `src/.../protocol.cpp` `from_json(ChatMessage)` only read
  `content.is_string()`.
- `include/vllm/entrypoints/openai/serving_chat.h` — `ChatPromptFn` seam renders
  `messages` → a prompt STRING; `create_chat_completion`
  (`serving_chat.cpp:577`) builds `prompt` then `add_request(prompt, params)` /
  `generate(prompt, ...)`. The engine `add_request` (`llm_engine.h:87,96`) takes a
  string OR pre-tokenized ids — **no mm-features overload**.

### vLLM's mm chat handling (MIRROR)
- `vllm/entrypoints/chat_utils.py:1478` `MM_PARSER_MAP` + `:1524`
  `_parse_chat_message_content_mm_part` — the content-part schema and dispatch:
  - `"text"` → `part["text"]`;
  - `"image_url"` → `part["image_url"]["url"]` (a `data:image/…;base64,…` URI or
    an http URL);
  - `"input_audio"` → `part["input_audio"]` = `{data:<base64>, format:str}`;
  - `"audio_url"` → `part["audio_url"]["url"]`;
  - others (`image_embeds`/`video_url`/`prompt_embeds`/`image_pil`/…) — residual.
- `vllm/multimodal/utils.py:35-113` — the `data:{mimetype};base64,{b64}` media
  encoding (we mirror the decode). Decode/route to a `MultiModalDataDict`, then
  the placeholder string is inserted into the templated prompt by the per-model
  parser (`_add_placeholder`, `chat_utils.py:886`).
- The OpenAI serving layer passes the `MultiModalDataDict` to the engine
  alongside the rendered prompt; the engine's input processor runs the mm
  processor and the encoder cache holds the features.

## Full wiring path (decomposed into bricks)

```
chat request content-part array          [MM-SERVE-PARSE — CPU, THIS BRICK]
  → parse typed parts (text/image_url/input_audio/audio_url)   protocol.cpp
  → decode data:/base64 payload → raw bytes                    chat_mm.cpp
  → route bytes through the EXISTING processor                 chat_mm.cpp
      · audio: WAV → ProcessWaveform → ExpandAudioPlaceholders (FULL on CPU)
      · image: RGB → ProcessImage    → ExpandImagePlaceholders (RGB in; codec = residual)
  → MultiModalInputs (expanded prompt ids + mm_features)       [asserted, CPU]

  → attach MultiModalInputs to the engine request              [MM-SERVE-ENGINE — CPU-ish]
      · new LLMEngine::add_request(prompt_ids, mm_features, params) overload
      · thread mm_features onto Request / EngineCoreRequest (fields already exist
        on Request for APC extra-keys) + encoder-cache budgeting
      · placeholder-string insertion into the chat-template prompt (chat_mm ↔
        ChatPromptFn), mirroring chat_utils `_add_placeholder`

  → mm model forward on the GPU worker                         [MM-SERVE-E2E — GPU, DGX]
      · encoder tower consumes mm_features; DeepStack/MRoPE inject
      · a real image+prompt OpenAI request → token-correct output
```

### Bricks

| Row | Scope | vLLM mirror | Tests to port | HW |
|-----|-------|-------------|---------------|----|
| `MM-SERVE-PARSE` (**LANDED, this commit**) | content-part parse + base64/data-URI decode + route to the existing processor → `MultiModalInputs` (shapes asserted) | `chat_utils.py:1478,1524`; `multimodal/utils.py:35-113` | `tests/entrypoints/test_chat_utils.py` (`test_parse_chat_messages_*`, the `image_url`/`input_audio` cases) → `test_chat_mm.cpp` | CPU |
| `MM-SERVE-ENGINE` (**LANDED, brick 2**) | attach `MultiModalInputs` to the engine request (`add_request(MultiModalInputs)` overload on `LLMEngine`+`AsyncLLM` via `InputProcessor::process_inputs_mm` → `mm_features` on `EngineCoreRequest`/`Request`; placeholder-STRING helpers mirroring `get_placeholder_str`; serving_chat `MultiModalChatFn` seam) | `v1/engine/input_processor.py:333-379`; `chat_utils.py _add_placeholder:886` + `get_placeholder_str` (qwen3_vl.py:1714 / qwen2_audio.py:333) | `test_input_processor` (`process_inputs_mm`) + `test_chat_mm` (placeholder strings + full chain) | CPU-verifiable (shapes/counts); no forward |
| `MM-SERVE-E2E` (**W3 — CPU SEAM BODY LANDED; GPU forward DEFERRED**) | the `MultiModalChatFn` seam BODY (`MakeQwen3VLImageChatFn`: messages → marker-inject → chat template → tokenize (single image_pad id) → RouteImageRgb EXPAND to N + mm_features), wired in `examples/server/main.cpp`. GPU forward = the closing gate (a real image+prompt request → token-correct on Qwen3-VL-4B vs the M2c golden) — **architecturally blocked** (see below) | `entrypoints/openai/chat_completion/serving.py` full mm path | `test_chat_mm` seam-body (real tokenizer + chat template → 196 image tokens, RED=0) + `test_openai_serving` (the production seam is invoked + routed) | CPU landed; **GPU e2e needs the engine mm-forward first** |

## This brick (`MM-SERVE-PARSE`) — landed

- `include/vllm/entrypoints/openai/protocol.h` — `struct ChatContentPart` (mirrors
  MM_PARSER_MAP fields) + `ChatMessage.content_parts` (nullopt for bare-string ⇒
  T0 byte-identical).
- `src/.../protocol.cpp` `from_json(ChatMessage)` + `ParseChatContentPart` — the
  array form fills `content_parts` and sets `content` to the joined text spans so
  the existing text prompt path is unchanged.
- `include/vllm/entrypoints/openai/chat_mm.h` + `src/.../chat_mm.cpp` —
  `DecodeBase64` (RFC 4648), `DecodeDataUri` (RFC 2397 `;base64`), the per-part
  decoders, and `RouteAudioWav` / `RouteImageRgb` reusing the existing processor
  seams to produce `MultiModalInputs`.
- Gate `tests/vllm/entrypoints/openai/test_chat_mm.cpp` (reuses the M1 image + A1
  audio processor-parity fixtures): inertness (bare-string unchanged); base64 /
  data-URI decode vectors; audio `input_audio` part → features `[80,3000]` + 1500
  placeholder tokens + byte-exact mm-hash; image `image_url` part → grid
  `[1,28,28]` + 196 merged tokens.

### Named residuals (out of this brick)
- **Container-format image decode (PNG/JPEG → RGB + dims):** no codec is vendored;
  `RouteImageRgb` consumes raw RGB (as the single-seq e2e path itself does). A
  real `data:image/png;…` needs a decoder — belongs to `MM-SERVE-ENGINE`/E2E or a
  small codec brick.
- **http(s) media-URL fetch** (vs inline `data:`/base64) — `DecodeDataUri`/
  `DecodeInputAudioPart` handle only inline payloads.
- **Engine plumbing** (`MM-SERVE-ENGINE`): **LANDED (brick 2).** `LLMEngine`/
  `AsyncLLM` `add_request(MultiModalInputs)` + `generate(MultiModalInputs)`
  overloads via `InputProcessor::process_inputs_mm` carry the expanded prompt +
  `mm_features` onto `EngineCoreRequest`/`Request`; serving_chat routes mm
  requests through them via the `MultiModalChatFn` seam. See "Brick 2 — landed".
- **Streaming mm, video parts, image_embeds/prompt_embeds** — the
  parse tolerates them (empty payload under their `type`) but they are not routed.
- **Multiple images — NOT "not routed": SILENTLY TRUNCATED** (#686). This line
  previously grouped them with the above, which understated it.
  `entrypoints/openai/chat_mm.cpp:256-262` locates the **first** `image_url` part
  and `break`s, so a three-image request is *served with one*, with no error and no
  warning. "Not routed" is accurate for video, audio and `*_embeds` — nothing is
  sent and the user notices. Truncation is the case where they do not.
  AGENTS.md requires an unimplemented arm to be "refused with a message naming the
  missing piece", and upstream does exactly that:
  `MultiModalProcessingInfo.validate_num_items`
  (`vllm/multimodal/processing/context.py:409-428`) raises `VLLMValidationError`.
  #607's L1 has now ported that refusal (`BaseProcessingInfo::ValidateNumItems`);
  wiring this call site to it is L2, and this is the reason L2 exists.

## Brick 2 (`MM-SERVE-ENGINE`) — landed

- `include/vllm/v1/engine/input_processor.{h,cpp}` — `process_inputs_mm`: the
  MULTIMODAL request builder (mirror of upstream `input_processor.py:333-379`).
  Takes the placeholder-EXPANDED prompt ids + `mm_features` and carries BOTH onto
  the `EngineCoreRequest`; otherwise byte-for-byte `process_inputs_tokens`
  (validate / default-max_tokens / eos+stop). Empty `mm_features` ⇒ identical to
  the tokens path (text never perturbed).
- `include/vllm/v1/engine/llm_engine.{h,cpp}` + `async_llm.{h,cpp}` — strictly
  ADDITIVE `add_request(MultiModalInputs)` + `generate(MultiModalInputs)`
  overloads on both engines. `mm_features` rides through
  `Request::FromEngineCoreRequest` (already wired) to the scheduler/encoder-cache
  seam. Parallel-sampling fan-out shares the mm inputs (child copies carry them).
- `include/vllm/entrypoints/openai/chat_mm.{h,cpp}` — the chat-template
  placeholder-STRING helpers: `ImagePlaceholderString` (`<|vision_start|><|image_pad|><|vision_end|>`,
  qwen3_vl.py:1716), `VideoPlaceholderString`, `AudioPlaceholderString(i)`
  (`Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>`, qwen2_audio.py:335),
  `ChatPlaceholderFor` / `CollectChatPlaceholders` (mirror `get_placeholder_str`
  + `_add_placeholder`). The single marker is what `ExpandImagePlaceholders`
  (brick 1) expands to N = grid/feature-count copies.
- `include/vllm/entrypoints/openai/serving_chat.{h,cpp}` — `MultiModalChatFn`
  seam (`set_multimodal_chat_fn`). Default UNSET ⇒ the text path is byte-identical
  (mm parts fall back to the joined-text content, brick-1 behavior). When set AND
  a request carries a mm part, `create_chat_completion` builds the
  `MultiModalInputs` and drives the non-stream engine mm `generate` overload;
  streaming mm is rejected (named residual).
- Gate: `test_input_processor` +2 (`process_inputs_mm` carries mm_features + the
  expanded prompt; empty == tokens path) + `test_chat_mm` +2 (placeholder strings
  mirror vLLM; the full chain parse→route→`process_inputs_mm`→`FromEngineCoreRequest`
  asserts the engine request carries the mm handles + the expanded prompt with the
  196 image_pad feature count). Text-path inertness suites byte-identical; clean
  CPU `-Werror` library + server build.

## Brick 3 (`MM-SERVE-E2E`) — CPU seam BODY landed; GPU forward deferred

The `MultiModalChatFn` seam body is now IMPLEMENTED on CPU:
- `include/vllm/entrypoints/openai/chat_mm.{h,cpp}` — `BuildMarkerInjectedContent`
  (walk content_parts in order, inject the placeholder MARKER at the mm part
  position; mirror `chat_utils.py:886 _add_placeholder`) + `MakeQwen3VLImageChatFn`,
  the seam body the server sets. Given the image processor + the model tokenizer +
  the chat-template renderer + an image codec, it turns chat `messages` into the
  engine's placeholder-EXPANDED `MultiModalInputs`:
  (1) marker-inject + render the templated prompt;
  (2) `tokenizer.EncodeWithSpecialTokens` → the single `<|image_pad|>` marker maps
      to ONE `image_token_id` (added tokens matched leftmost-longest, tokenizer.h:55);
  (3) decode the image (codec) + `RouteImageRgb` → EXPAND that single id to
      N = prod(grid_thw)/merge² copies + build `mm_features`.
  IMAGE only; video/audio/multiple-image are named residuals.
- `examples/server/main.cpp` — production wiring: when the model dir carries
  `preprocessor_config.json`, construct the Qwen3-VL image processor and
  `set_multimodal_chat_fn(MakeQwen3VLImageChatFn(...))`. A text-only model leaves
  the seam UNSET ⇒ the chat path is byte-identical. The container-format codec
  (PNG/JPEG → RGB) stays a NAMED residual: the production codec accepts
  `image/x-raw-rgb` (the fixture format) and rejects encoded images with a clear
  message.

### The GPU e2e gate is ARCHITECTURALLY BLOCKED (not box contention)
A real token-exact image `/v1/chat/completions` CANNOT run through the production
engine seam yet — the engine MODEL RUNNER has no multimodal forward:
- `include/vllm/model_executor/models/model_registry.h:142` `ModelForwardInput`
  (the per-forward struct `ModelRegistry::Forward` consumes) has NO vision /
  mm-embedding / inputs_embeds field — token_ids + positions + attn/kv only.
- `src/vllm/v1/worker/gpu/runner.cpp` never reads `Request.mm_features` (it is
  carried onto Request for the scheduler/encoder-cache extra-keys, but no forward
  consumes it).
- Qwen3-VL is NOT engine-registered — there is no `REGISTER_VLLM_MODEL` in
  `qwen3_vl*.cpp`; the M2c vision forward is the STANDALONE `Qwen3VLGenerateGreedy`
  (`qwen3_vl.cpp:502`), a bespoke greedy driver that does embed+merge+MRoPE+DeepStack
  itself, OUTSIDE `ModelRegistry::Forward`.

Closing the GPU gate (the exact residual, DGX GB10 + Qwen3-VL-4B checkpoint):
1. add a vision-embedding / inputs_embeds field to `ModelForwardInput`;
2. `runner.cpp`: run the vision tower on `Request.mm_features` via the
   `EncoderCacheManager` seam, `_merge_multimodal_embeddings` into the token
   embeddings, and pass MRoPE 3-D positions + DeepStack multiscale — i.e. fold the
   `Qwen3VLGenerateGreedy` forward INTO the registered engine forward;
3. `REGISTER_VLLM_MODEL(qwen3_vl, "Qwen3VLForConditionalGeneration", …)`;
4. then a `test_openai_serving_chat_mm_e2e.cpp` (dgx-only) drives a real image
   `/v1/chat/completions` through the running server / the constructed ApiServer
   harness and asserts token-exact vs the committed M2c golden `gen_tokens_i32.bin`
   (STRICT, K-deterministic). Recipe: `flock $HOME/gpu.lock`, `df` disk-guard
   (tree ~21G; box was 99% full 2026-07-28), `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
   `ssh dgx.casa`.

## Correctness / gates
- CPU (bricks 1+2+3): `test_chat_mm` 8/8, 100 asserts (W3 +1 seam-body test:
  the real tokenizer + chat template render 196 image tokens + mm_features; RED
  line = the text-only path renders 0 image tokens); `test_openai_serving` +1 (the
  production seam is invoked on an image request + routed to the engine mm generate
  overload; text-only never touches the seam; streaming+mm rejected). Inertness
  suites byte-identical for bare-string; clean `-Werror` full-library + `server`
  build (0 warnings).
- **GPU closing gate (`MM-SERVE-E2E`):** DEFERRED — architecturally blocked on the
  engine mm-forward (above), not on the DGX being unavailable. The CPU seam body is
  wired + gated; the forward integration is the named residual.

## MM-ENGINE-FORWARD — the block is RESOLVED (2026-07-28, `CLAIM-ENGINE-MM-FORWARD`)

The three-point architectural block above is CLOSED. Multimodal now runs through the
engine's REGISTERED forward (`ModelRegistry::Forward`), not only the standalone
`Qwen3VLGenerateGreedy` driver.

1. **`ModelForwardInput` gains a vision field.** `MultiModalForwardInput`
   (`model_registry.h`) carries the ALREADY-MERGED host bf16 inputs_embeds, the 3-D
   MRoPE positions `[3,T]`, and the DeepStack `[L,T,H]` (borrowed handles); it hangs
   off `ModelForwardInput` as `std::optional<MultiModalForwardInput> mm =
   std::nullopt`. ADDITIVE + default-nullopt ⇒ every TEXT step leaves it nullopt, the
   registered text forwards never read it, and the shared runner path is
   byte-identical **by construction** (proven: `test_runner` 16/16 + `test_scheduler`
   36/36 + `test_model_registry` 24/24 + `test_chat_mm` 8/8 + `test_openai_serving`
   41/41, all green).
2. **The M2c forward is FOLDED into the registered per-step contract.**
   `VLForwardLastLogits` is exposed as the shared `Qwen3VLForwardStepLastLogits`;
   `VLGenerateCore` is refactored to a shared `VLStepFn` driven by BOTH the standalone
   driver AND `Qwen3VLGenerateGreedyViaRegistry`. The registered `ForwardQwen3VL...`
   consumes `input.mm` and calls the SAME step ⇒ registered == standalone numerically,
   no duplicated decode. `Qwen3VLGenerateGreedyViaRegistry` (the engine mm-forward
   entry the MM-SERVE seam uses) drives every step through `ModelRegistry::Forward`.
3. **Qwen3-VL is REGISTERED.** `REGISTER_VLLM_MODEL(qwen3_vl,
   "Qwen3VLForConditionalGeneration", …)` in the new `qwen3_vl_registry.cpp`
   (`supports_multimodal=true`, non-hybrid dense backbone, full-attention KV spec).
   The registered LoadedModel owns the persistent cos|sin MRoPE cache (built with the
   SAME RopeArgs/Pmax as the driver ⇒ bit-identical).

**GPU closing gate (this pass):** `test_qwen3vl_registry_e2e` (dgx.casa GB10 +
cached `Qwen3-VL-4B-Instruct`) runs image→text THROUGH `ModelRegistry::Forward` and
asserts token-exact vs the M2c golden `gen_tokens_i32.bin` (STRICT, K-deterministic),
RED-first (unregistered ⇒ `Resolve` throws). Recipe: git-archive the commit →
`dgx.casa`, `export PATH=/usr/local/cuda/bin:$PATH`,
`cmake -B build-cuda -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DVLLM_CPP_CUDA_ARCHITECTURES=121a`,
`flock $HOME/gpu.lock ./build-cuda/tests/test_qwen3vl_registry_e2e`.

**Named residual (still open):** (a) the FULL in-runner path — `runner.cpp` reading
`Request.mm_features`, running the tower via the `EncoderCacheManager`, and building
the `mm` field from staged encoder outputs for the BATCHED scheduler loop (this pass
drives the registered forward via `Qwen3VLGenerateGreedyViaRegistry`, single-sequence,
so batched-runner mm + cross-step per-request MRoPE-delta state is the residual);
(b) the real server `/v1/chat/completions` GPU e2e (the W3 seam → the registered
forward end-to-end); (c) video / multi-image / audio / Gemma-4-image through the
registered path. README's "not yet wired into the OpenAI server end-to-end" line is
now RESOLVABLE for the registered-forward half.
