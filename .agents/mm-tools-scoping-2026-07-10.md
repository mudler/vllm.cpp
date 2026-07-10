# Multimodality + tool-calling scoping (task #49, four sub-agent reports synthesized 2026-07-10)

Sources: vLLM V1 MM architecture map, gate-model family/template scan, vLLM
parallel-tool-call map, our-server tool-call survey (all grounded file:line;
vLLM pin `e24d1b24`, deps `~/venvs/vllm-oracle`).

## Headlines (both change previous beliefs)

1. **The gate checkpoints ARE the multimodal models.** There is no separate
   "Qwen3.6-VL": `Qwen3_5[Moe]ForConditionalGeneration` subclasses
   `Qwen3VLForConditionalGeneration` (`vllm/model_executor/models/qwen3_5.py:389,604`).
   Both gate checkpoints ship the full 27-block SigLIP-shaped ViT — **333
   `model.visual.*` tensors, bf16 — which our loader silently drops**
   (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:210`). vLLM
   serves these exact checkpoints as VLMs by default → vision enablement on
   the gate models is a MIRROR obligation, not an expansion.
2. **Our Qwen3 tool parser targets the WRONG format.** The shipped gate
   template (byte-identical in both checkpoints) instructs and renders the
   **Qwen3-Coder XML format** (`<function=NAME><parameter=K>V</parameter>…`),
   and vLLM's own docs serve Qwen3.6-27B with `--tool-call-parser qwen3_coder
   --reasoning-parser qwen3` (`docs/serving/integrations/codex.md:30`). Our
   `Qwen3ToolParser` is a Hermes-JSON subclass — it will fail on real gate
   output, and our `tool_choice=required/named` structural tags would force
   JSON against an XML-trained model. The Hermes inference came from tokenizer
   special tokens; the XML markers are plain text, so their absence proved
   nothing (recorded as a grounding lesson).

## A. Multimodal — what stands between us and vision on the gate models

vLLM V1 MM pipeline anatomy (all upstream, mirror targets):
- **Preprocessing**: vLLM does NOT reimplement image ops — resize/normalize/
  patchify happen inside the HF `AutoProcessor` call
  (`vllm/multimodal/processing/context.py:270`, chain from
  `BaseMultiModalProcessor.apply()` processor.py:1663→1097). **This is the
  hard C++ problem**: port Qwen smart-resize/normalize + placeholder
  expansion (`<|vision_start|><|image_pad|><|vision_end|>`, video timestamp
  tokens) with numeric parity. Placeholder/count validation at three layers
  (processor.py:1581-1619; models/utils.py:545-556 masked scatter).
- **Caching/scheduling**: blake3 content-hash mm items (hasher.py:50,
  processor cache cache.py), `EncoderCacheManager` keyed by mm_hash with
  cross-request reuse + LRU freeable set (encoder_cache_manager.py:17-255),
  encoder budget co-scheduled with chunked prefill
  (`_try_schedule_encoder_inputs`, scheduler.py:1317 — atomic mm items,
  rollback of num_new_tokens).
- **Runner**: `_execute_mm_encoder` (gpu_model_runner.py:2917) batches by
  modality → `model.embed_multimodal`; `_gather_mm_embeddings` (:3126) +
  `embed_input_ids(…, is_multimodal=mask)` scatter into `inputs_embeds`
  (:3476) — MM models always run the decoder on embeddings, not ids.
- **Audio**: decoder-only audio (Qwen2-Audio/Ultravox/Omni thinker) reuses the
  exact same scatter path; only Whisper needs true enc-dec machinery
  (`EncoderDecoderCacheManager`). Audio rides the same plumbing later.

What the gate models specifically need (from `qwen3_vl.py`, reused by
`qwen3_5.py:412-418`):
Conv3d patch embed (2×16×16) · learned 48×48 pos-embed + bilinear interp ·
27× pre-LN ViT blocks (LayerNorm, qkv+bias varlen full attention per image
with 2-D rope, GELU-tanh MLP — all primitives exist in our stack; bf16,
unquantized, ≈0.4B params) · 4→1 patch merger · embedding scatter ·
**interleaved M-RoPE** (`(3,T)` positions, `mrope_section [11,11,10]`,
`mrope.py:190`; our RoPE path is 1-D today, degenerate-equal for pure text) ·
MM input pipeline above. **Deepstack: NOT needed** — both gate configs have
`deepstack_visual_indexes: []` (compiled out, `qwen3_vl.py:1709-1716`).
Everything else (GDN hybrid, MoE, NVFP4, KV, sampler, server) is 100% shared.
Caveat: unsloth 27B lacks `preprocessor_config.json` — image-processor
defaults need care.

**Ranked route:** ① vision on the gate checkpoints themselves (max reuse, max
demand, mirror obligation) → ② Qwen3-VL-MoE (adds deepstack + a plain
full-attention Qwen3MoeModel text stack — simpler than what we built) → ③
Qwen3-Omni thinker (audio tower; defer).

## B. Tool calls — vLLM vs ours

Ours (verified): parallel calls work end-to-end at parser/streaming-index/
serialization layers with tests (hermes.cpp:136-159, 212-252; DeltaToolCall
index always serialized protocol.cpp:437-441); `finish_reason="tool_calls"`
correct both paths; tool_choice auto(lazy)/required(one-or-more)/named(one)
via native structural tags (serving_chat.cpp:98-148).

vLLM semantics to mirror (deltas):
1. **`parallel_tool_calls` honored** (default `true`): post-generation filter
   only — truncate to first call / index-0 deltas when `false`
   (`entrypoints/serve/utils/tool_calls_utils.py:19-37`); grammar NEVER
   tightened (no maxItems anywhere). Ours parses-and-ignores → small mirror.
2. **`required` = JSON ARRAY grammar** (`tool_parsers/utils.py:247-262`,
   `minItems:1`, no max; streaming index = `len(obj)-1`, streaming.py:203,222);
   named = single object. Our structural-tag one-or-more is semantically
   equivalent — but must be retargeted to the XML tag shape (headline 2).
3. **Round-tripping is broken on our side** (the real serving gap): incoming
   `ChatMessage` never parses `tool_calls` (protocol.cpp:213-219) and the
   template context gets only role+content (chat_template.cpp:1147-1153) — an
   assistant turn with tool_calls can't be re-rendered; multi-turn tool
   conversations are impossible today.
4. **No reasoning parser at all** — the gate template FORCES generation to
   open inside `<think>`; vLLM splits `reasoning_content` via the same
   `Qwen3Parser` state machine (`vllm/parser/qwen3.py:92`). Without it, our
   gate-model chat responses would leak think-text as content.
5. **Our Jinja engine cannot render the shipped gate template.** Finite
   missing-construct list (template-driven): `macro` w/ default args ·
   `namespace()` + `set ns.x` · filters `trim/items/string/safe` (+tuple
   unpack `for k, v in x|items`) · tests `is string/iterable/mapping/none/
   defined/true` · reverse slice `[::-1]` · `loop.previtem/nextitem` ·
   `.startswith/.endswith/.split()[-1]` · `raise_exception()` · tolerated
   kwargs (`enable_thinking`, …). Tool-role messages render as
   `<tool_response>` merged into user turns (template lines 131-142).
6. `OpenAIServingChat` still isn't wired into a shipping server main; no HTTP
   test asserts >1 tool call through the full path.

## C. Delegable specs (feature-matrix rows → .agents/specs/, per §E + test-porting.md)

| Spec | Scope | Rank |
|---|---|---|
| `specs/qwen3-tool-parser-xml.md` | Port `Qwen3Parser` XML state machine + reasoning split; retarget structural tags to `qwen_3_coder` shape; keep Hermes for Qwen2.5-era | **1 — corrects a wrong-format ship-blocker** |
| `specs/chat-template-jinja-gate.md` | The ~10 missing Jinja constructs; gate = render the shipped template byte-exact vs transformers' Jinja2 on the same messages | **1 (pairs with above)** |
| `specs/tool-call-round-trip.md` | Parse incoming `tool_calls`/tool-role; expose to template; `parallel_tool_calls` post-filter; wire serving into server main + HTTP multi-call test | 2 |
| `specs/vision-gate-models.md` | ViT + M-RoPE + scatter for the gate checkpoints (stop dropping `model.visual.*`) | 3 (big, high demand) |
| `specs/mm-input-pipeline.md` | C++ HF-processor parity (smart-resize/normalize/patchify), mm hashing, encoder cache + scheduler budget, runner scatter | 3 (prereq of vision serving) |
| `specs/audio-models.md` | Decoder-only audio (Omni/Qwen2-Audio path) | 4 (defer) |

**Tests to port** (test-porting.md): `tests/models/multimodal/processing/
test_common.py` (cached-vs-uncached + text-vs-token bit-exactness harness) +
`test_qwen3_vl.py` · `tests/multimodal/{test_processing,test_hasher,test_cache,
test_embedding_shape_validation}.py` · v1 encoder-cache/scheduler MM cases ·
`tests/entrypoints/openai/` tool-parser + parallel-tool-call suites ·
`vllm/parser/` qwen3 state-machine tests.
