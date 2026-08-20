# C API reference

Use the public C ABI in `include/vllm.h` for external applications.

Link `libvllm` (static or shared) and include [`include/vllm.h`](../../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 23`,
`include/vllm.h:329`; **47** exported functions, the count of `^VLLM_API `
declarations in that header) suitable for `dlopen` / FFI / LocalAI integration.
This line read `19` and `36` until 2026-08-17 and `21`, `273` and `46` until the
W0 phase log added `vllm_video_last_phase_log`; every one of those numbers was
last true several ABI additions ago, and none of the three is derived by any
gate, the version, the line and the count each drift independently, and the
line number drifts on an edit that adds no ABI at all. The version moved twice
in one day: `mmproj_path` took v22 and the phase log, written as v22 on its own
branch, landed as v23.

On native Windows/MSVC, the shared-library packaging lane keeps the runtime DLL
name at `vllm` and gives the import/static archive the distinct name
`vllm_shared`, so one build tree can hold the shared C ABI package and the
static `vllm` archive without a filename collision. The same ABI smoke test
therefore resolves the exported symbols through `LoadLibraryA` /
`GetProcAddress` on Windows and `dlopen` / `dlsym` on POSIX.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               /* sp.temperature = 0.0 means greedy */

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle, blocking and streaming completion, non-blocking
concurrent requests, memory helpers, and diagnostics. Later ABI versions add:

| ABI | Adds |
|---:|---|
| v2 | Structured output (JSON schema, JSON object, regex, choice, GBNF) |
| v3 | Chat with tools and chat templates |
| v4 | Tool-parser selection |
| v5 | Reasoning-parser selection |
| v6 | Speculative decoding |
| v7 | Prefix caching (tri-state) |
| v8 | Custom logits processors |
| v9 | Engine sizing: chunked-prefill token budget, scheduling policy, external KV connector / LMCache |
| v10 | Jump-forward decoding (tri-state, default off) |
| v11 | Audio transcription through `vllm_transcribe` |
| v12 | Video and audio generation through `vllm_video_*` |
| v13 | Pre-tokenized completion through `vllm_complete_tokens` |
| v14 | Explicit device selection (`auto`, CPU, or CUDA) |
| v15 | Embeddings through `vllm_embed` |
| v16 | Absolute KV-cache memory sizing |
| v17 | The OpenAI server as a thin ABI client through `vllm_server_main` |
| v18 | Video model-family selection (`family`, `vllm_video_engine_family`) and family-specific `extra_keys`/`extra_values` on `vllm_video_*` |
| v19 | Per-modality multimodal input limits |
| v20 | Speech and music generation through `vllm_speech_*` / `vllm_synthesize` |
| v21 | Device selection on the speech lane (`vllm_speech_model_params.device`, `vllm_speech_engine_device`) |
| v22 | A second GGUF for the multimodal projector (`vllm_model_params.mmproj_path`) |
| v23 | The render phase table: `vllm_video_last_phase_log` names the `phase-log.json` a completed `vllm_video_generate` wrote |

Chat templates render through the vendored google/minja engine, the same
renderer llama.cpp ships.

## Consuming it from C++

The higher-level surface lives under [`include/vllm/`](../../include/vllm/).
`LoadedEngine::FromModelDir(...)`
([`entrypoints/model_loader.h`](../../include/vllm/entrypoints/model_loader.h))
hands back either the synchronous `LLMEngine`
([`v1/engine/llm_engine.h`](../../include/vllm/v1/engine/llm_engine.h)) or the async
`AsyncLLM` ([`v1/engine/async_llm.h`](../../include/vllm/v1/engine/async_llm.h)) that
the server itself uses.

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;
ep.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

The underlying portable tensor runtime is `vt::` ([`include/vt/`](../../include/vt/)),
which carries no ggml or PyTorch dependency.

Video and audio generation is reached through `vllm::multimodal::VideoEngine`
([`multimodal/video_engine.h`](../../include/vllm/multimodal/video_engine.h)).
`LoadVideoEngine` resolves the model family from what the checkpoint HOLDS, never
from a filename, and refuses rather than guessing: zero claimants, several
claimants, and an unregistered declared `family` are all errors that name what was
seen and what is registered. A caller who supplies no `dit_path` is told which
artifact is missing rather than being advised to declare a family, which would not
help. A family adds itself with `RegisterVideoFamily`, which refuses a name that
is already registered, because two families under one name would collapse into a
single claimant and leave the choice of loader to link order.

Two families are registered. `minimax-h3` is detected by `video_patch_proj` plus
`audio_patch_proj`; `ltx-2.5` by `patchify_proj` plus `audio_patchify_proj`, with
or without the ComfyUI `model.diffusion_model.` prefix. Each family reads its own
knobs from `extras`. H3 takes `partition`. LTX-2.5 takes
`audio_prompt_embeds_path` (the audio stream's conditioning, the twin of the
seam's `prompt_embeds_path`, which carries the video stream), `pipeline_kind`
(default `distilled_two_stage`; also `one_stage`, `res2s_two_stage`, `dmd2`,
`dfr`, `retake` and `t2a_one_stage`), `model_version` (only for a checkpoint that
declares none), `dit_config_path`, `encoder_config_path`,
`negative_prompt_embeds_path` and `negative_audio_prompt_embeds_path` (the
negative half of the same fallback, for the unconditional forward),
`allow_unported_modules`, `max_phase`, `prompt_embeds_valid_rows`,
`upsampler_path`, `duration_head_path`, `lora_path` and `lora_strength`, twelve
keys, which is `kKnownLoadExtras` (`ltx2_video.cpp:377-383`) in order. The two
LoRA keys landed with issue #923 and were missing from this list until
2026-08-17; the array's own neighbouring comment still says "nine of these ten",
which is [#1097](https://github.com/mudler/vllm.cpp/issues/1097).
An extra a family does not define is
refused, never ignored. One caveat inside that set: `duration_head_path` is
defined but UNSERVED, the duration head is ported and gated as a brick, and
nothing in the video engine constructs one, so supplying it is **refused by
name** at load rather than accepted. It used to be accepted and read by nothing,
which silently substituted the recipe default for the file you named. Give
`num_frames` (or `duration`, which is exact arithmetic against the recipe's frame
rate) instead. Every other key in that list reaches a reader.

One LTX-2.5 arm is refused where a render would otherwise silently downgrade:
the spatiotemporal latent upsampler. It is reachable, supplying that checkpoint
as `upsampler_path` gets a refusal naming the arm you actually supplied. The
spatiotemporal upsampler is the arm with `spatial_upsample` AND
`temporal_upsample` set, which upstream builds as a different operator
(`Conv3d(mid, 8*mid)` + `PixelShuffleND(3)`). The temporal-only x2 upsampler is
**ported** and is not refused; nothing shipped drives it yet, so it is gated
rather than served. **Three** more are
recorded as out of scope but are **not requestable**, so no flag or extra can
reach them: `int8-convrot`, single-node multi-GPU, and
`BetaScheduler`. (LoRA fusion was in that list until 2026-08-15 and is now
SERVED - see `--lora` above - so its marker was retired rather than moved. This
sentence still said "Four more" until 2026-08-17, counting the retired marker in
the same breath as it explained the retirement.) That is four
`Ltx2UnportedPipelineFeature` enumerators in total, one reachable and three
markers (`ltx2_pipeline.h:768-803`), and the split is derived from the tree by
`test_ltx2_pipeline` rather than restated here. Their messages
say `DECLARED, NOT REQUESTABLE` so the two kinds are not confused.
`BetaScheduler` is in that group rather than the reachable one because upstream
selects it nowhere: every `ltx-pipelines` entry point hard-codes
`LTX2Scheduler()`, so there is no scheduler-kind field to mirror and nothing here
carries one either. `int8-convrot`
in particular is a ComfyUI-ecosystem format: upstream LTX-2's own inference
quantization kinds are `fp8-cast`, `fp8-scaled-mm`, `nvfp4-cast` and
`nvfp4-prequant`, and nothing wired upstream reaches int8 at all.

What is **not** on that list, and why: **multi-shot or multi-scene generation.**
A request that composes several camera takes into one output has no flag here
because upstream LTX-2 has no such mode to mirror, its `shot` is one continuous
take, and its own prompt-enhancement prompts instruct the model to keep a "single
continuous take" and not to describe scene cuts. `scene` does appear across the
upstream tree, in three unrelated senses (`scene-linear` HDR colour, PySceneDetect
in the trainer's dataset preprocessor, and that prompt-writing guidance); none of
them is a generation mode. This port carried a `multishot` refusal until
2026-08-13, which was a defect in our own record rather than a gap, and it was
retired. Generate one take per request.

`prompt_embeds_valid_rows` is how many of the supplied conditioning rows are real
tokens; absent, every row is. It matters because the embeddings connector
substitutes its learnable register table at PADDED positions, so padding decides
which of the connector's inputs are learned constants rather than caption
features. Upstream always knows this because its tokenizer produced the mask;
this seam reads conditioning from a file, which carries none.

`dit_config_path` names a JSON file holding the DiT's `{"transformer": {...}}`
configuration, and it exists because only one of the two shipped LTX-2.5 DiTs
carries one. The first-party NVFP4 file embeds it in `__metadata__["config"]`;
the ungated `vonkaiser/LTX-2.5-FP8-NVFP4` FP8 DiT has no `__metadata__` at all.
Tensor shapes resolve the geometry but not the values no shape encodes, so
without a config `double_precision_rope` would default to false and
`av_ca_timestep_scale_multiplier` to 1, where LTX-2.5 declares `float64` and
`1000`. Both move every RoPE angle and every audio-to-video modulation, so a DiT
that declares no config is refused until one is supplied rather than rendered
under defaults that contradict the model family. A supplied config is adopted
only when it reproduces the identical weight contract the shapes describe, and
supplying one for a checkpoint that already declares its own is refused rather
than ordered.

`vllm_video_model_params.device` is `0` for the CPU and `1` for **the
accelerator this build resolves**, not for CUDA. The value is unchanged and it
is CUDA on a CUDA build, but it is read through the platform seam rather than as
an enum value, so the same `1` selects Metal, Vulkan or Tenstorrent on a build
that registers one of those, and is refused by name on a build that registers
none. The C ABI's text-generation `vllm_model_params.device` is a separate,
later selector with its own `0 = auto / 1 = cpu / 2 = cuda` numbering.

The LTX-2.5 arm runs on the CPU in f32 and on CUDA in bf16. `device = 0` takes
the f32 parity forward; `device = 1` stages the DiT to the GPU one tensor at a
time and runs the device-resident forward, so a CUDA handle means a CUDA forward.
On a build with no accelerator backend, `device = 1` is refused by name rather
than served the CPU forward behind an accelerator handle. It is also refused when the build's
accelerator is a PARTIAL backend that declines this architecture, Metal and
Tenstorrent each register the kernels for a named short list of models, and a
backend that has not registered this one now says so by name instead of binding
a queue and failing later inside a kernel. The same three questions decide
`minimax-h3`'s `device = 1`, which resolves through the platform seam rather
than reading the ABI selector as an enum value, so on a CPU-only build it throws
instead of naming CUDA. `encoder_path` loads the Gemma-4
text tower, and the request's own `prompt` then conditions the render; the tower
itself runs on the CPU in f32 whichever device the DiT is on. Without one,
conditioning comes from the two prompt-embeds files, which must agree on their
row count.

`Sampler`'s `logprobs_mode` selects which tensor the returned logprobs are read
from, and all four of vLLM's values now work: `raw_logprobs` (the default) and
`raw_logits` are snapshotted before any logits processor runs, so they describe
the MODEL's distribution; `processed_logprobs` and `processed_logits` are taken
after temperature and top-k/top-p, so they describe the distribution actually
SAMPLED from, a token top-k masked away reads `-inf` there and its true value
under the raw pair. It is selectable by constructing a `Sampler` directly; there
is no config, CLI or request field for it yet.

`LogprobsTensors::slice_request(req_idx, request_num_positions)` cuts that
batch-wide payload by rows. The second argument is the requested row count;
each row keeps the source tensor's independent `num_tokens_per_position`
width.

(That brick is the TEXT decode path and is a different mechanism from LTX-2.5's
IC-LoRA, which fuses into the weights at load and IS served - see `--lora`.)
The LoRA adapter headers ([`lora/lora_weights.h`](../../include/vllm/lora/lora_weights.h),
[`lora/punica.h`](../../include/vllm/lora/punica.h),
[`lora/layers.h`](../../include/vllm/lora/layers.h)) are present but **not yet wired
to any engine path**: they are the in-progress runtime (`LORA-RUNTIME`), not a
supported way to serve an adapter. There is no CLI flag, server flag, config key
or C-ABI field for LoRA, and adding one is a later work item, see
[`.agents/specs/lora-adapter.md`](../../.agents/specs/lora-adapter.md).

`SamplingParams::logprobs` accepts `-1` for "every vocab entry", as vLLM's does;
it returns the same gathered shape a finite count returns, one entry per vocab id
per position.

Over HTTP the same `-1` reaches the chat surface: `{"logprobs": true,
"top_logprobs": -1}` is accepted, as in vLLM, and returns every vocab entry for
each generated token. No numeric range is enforced on either surface, vLLM's
`check_logprobs` request validation and its `max_logprobs` model cap are not
ported yet. Two consequences: `{"logprobs": -1}` on the **completion** surface
returns empty `top_logprobs` maps where vLLM answers `400`, and an out-of-range
count is not rejected. Both are tracked by
[issue #249](https://github.com/mudler/vllm.cpp/issues/249).

`SamplingParams::logprob_token_ids` scores an EXPLICIT set of vocab ids instead,
vLLM's generative-scoring path, and what to reach for when you only need a few
labels compared, since it avoids the full-vocab sort `logprobs=-1` costs:

```cpp
vllm::SamplingParams sp;
sp.max_tokens = 1;
sp.logprob_token_ids = std::vector<int32_t>{yes_id, no_id};  // `logprobs` unset
```

Each returned position then carries exactly those ids plus the sampled token,
whose `rank` is still its rank over the WHOLE vocabulary, so it stays comparable
across requests. At most 128 ids (vLLM's `MAX_LOGPROB_TOKEN_IDS`); setting
`logprobs` as well is allowed only when it equals the id count, and the explicit
ids win. This is a library-API field today, the OpenAI request field is not
wired yet.

## KV-cache events, and `kv_cache_report_mode`

`SamplingParams::extra_args` is a per-request string map mirroring vLLM's
`extra_args`, and the one key read from it today is `kv_cache_report_mode`:

```cpp
vllm::SamplingParams params;
params.extra_args = std::map<std::string, std::string>{
    {"kv_cache_report_mode", "full"}};
```

It controls how much of that request's prefix-cache activity reaches the
KV-cache event stream. `"incremental"`, the default and what you get whenever the
key is absent, reports only blocks the request newly STORED. `"full"` also
re-reports the blocks it REUSED from the cache, which is what a prefix-cache-aware
router needs to learn that this engine already holds a prefix.

Events are OFF unless a `vllm::distributed::KVEventsConfig` with
`enable_kv_cache_events = true` is passed to the `Scheduler`, so
`kv_cache_report_mode` changes nothing by itself. With events on, each engine step
publishes at most one `KVEventBatch`, a wall-clock `ts`, that step's
`BlockStored` / `BlockRemoved` / `AllBlocksCleared` events, and the data-parallel
rank, to the configured publisher, and its msgpack encoding is byte-identical to
what vLLM puts on the wire.

Two limits to know. The **`zmq` publisher is not ported**: asking for it throws
rather than silently downgrading, because the live socket transport needs a
dependency this project does not carry, so `publisher` must be `"null"` today,
and it must be set explicitly, since an unset value is not yet resolved the way
vLLM resolves it ([issue #353](https://github.com/mudler/vllm.cpp/issues/353)).
And `extra_args` is reachable **only from the C++ API**: the HTTP door to it
(`vllm_xargs`) is not ported, so an OpenAI request cannot set the report mode.
