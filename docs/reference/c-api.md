# C API reference

Use the public C ABI in `include/vllm.h` for external applications.

Link `libvllm` (static or shared) and include [`include/vllm.h`](../../include/vllm.h).
It exposes a flat, exception-free C ABI for `dlopen`, FFI, and LocalAI
integration. Read `VLLM_ABI_VERSION` in the header for the current ABI version.

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

The ABI covers engine lifecycle, completion, chat, embeddings, transcription,
media generation, speech generation, memory helpers, and diagnostics. It also
exposes blocking, streaming, and concurrent request interfaces. The current
version is `VLLM_ABI_VERSION 23`.

Read [`include/vllm.h`](../../include/vllm.h) for the fields and functions in
the current ABI. Call `vllm_abi_version()` at runtime to detect a header and
library mismatch.

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


For video-family load options, defaults, and refused arms, see the
[LTX 2.5 recipe](../models/ltx-2-5.md) and
[MiniMax-H3 recipe](../models/minimax-h3.md).

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
each generated token. The **completion** surface is different, and that is
vLLM's own difference rather than ours: `{"logprobs": -1}` there answers `400`
`` `logprobs` must be a positive value.`` The chat count additionally requires
`"logprobs": true` whenever it would emit a payload.

vLLM's `check_logprobs` request validation is ported
([#1815](https://github.com/mudler/vllm.cpp/issues/1815)); its `max_logprobs`
model cap is not. The remaining consequence is that an out-of-range count is
capped at the vocabulary size rather than rejected against a configured maximum,
which is why `{"prompt_logprobs": -1}` is served here and refused by a vLLM whose
`--max-logprobs` defaults to 20. Tracked by
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
