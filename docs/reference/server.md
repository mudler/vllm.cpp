# Server reference

Use this page to look up server behavior, flags, and defaults.

`vllm-server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../../src/vllm/entrypoints/openai/).

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```


For build and packaging commands, see [Building vllm.cpp](../BUILD.md). For
published artifacts and verification steps, see [Binary releases](../RELEASES.md).

## Selecting an x86 CPU ISA tier

The x86_64 CPU library is one adaptive binary: portable, SSE2,
SSE2+F16C, AVX2, and AVX-512 elementwise matmul kernels are isolated in their
own translation units and selected only after CPUID plus the required XCR0 OS
state are checked. Leave `VT_CPU_MATMUL_TIER` unset for automatic selection, or
set it to `portable`, `sse2`, `sse2+f16c`, `avx2`, or `avx512` for a same-binary
correctness/performance check. A forced tier that the current CPU or OS cannot
execute fails closed instead of silently narrowing or risking an illegal
instruction. Release builds never use `-march=native`.

On arm64, leave the same variable unset to select between portable and NEON
elementwise matmul, or force `portable`/`neon`. DotProd and i8mm kernels are
independently selectable with `VT_CPU_Q8_DOT`, `VT_CPU_QUANT_MMLA`, and
`VT_CPU_QUANT_REPACK`; `auto` uses Linux HWCAP/HWCAP2 or Darwin feature sysctls,
while an unavailable forced tier fails closed. The exact accepted values are
listed in [the environment reference](../ENVIRONMENT.md).


NVFP4 optimization defaults and current measurements are listed in
[the environment reference](../ENVIRONMENT.md) and
[the benchmark report](../BENCHMARKS.md).
### Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../../src/vllm/entrypoints/openai/api_server.cpp).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4), recorded per engine step by the engine that serves your requests. Series and families keep stable addresses as new ones register (#330), so a long-lived scrape target does not read through a reallocated registry |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`) |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |
| POST | `/v1/embeddings` | Embeddings. Registered **only** when an embedder is attached, so a text server answers 404 at the route table |
| POST | `/v1/audio/transcriptions` | Speech to text (multipart: audio as `file`, `response_format` as a form field). Registered **only** when a transcriber is attached |
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |
| POST | `/v1/audio/speech` | Text (or lyrics + a music description) to audio; responds with `audio/wav` bytes. Registered **only** when a synthesizer is attached (`--speech-model`) |

The reference-audio side of IndexTTS-2.5 is complete in the library -- a 16 kHz
clip goes through the SeamlessM4T feature extractor, the w2v-bert Conformer, the
layer-17 hidden-state tap, the checkpoint's stored-statistics normalization and
the semantic codec to discrete codes, and the talker's prompt is assembled from
that conditioning plus the text -- but none of it is reachable from a command or
a route yet. The greedy generate loop that turns the prompt into mel codes is
ported too, and so is the STATED-emotion path -- eight weights selecting rows
from the checkpoint's own speaker and emotion matrices by cosine similarity -- so
text plus a reference clip and an emotion reaches mel CODES in the library. What
is still missing is a COMMAND or ROUTE. TEXT DOES REACH AUDIO in the library:
`test_indextts2_e2e` tokenizes with the checkpoint's own vocabulary, runs the
talker to mel codes, and drives those through the length regulator, the CFM loop
and BigVGAN to samples. Point it at all four checkpoint paths:

```sh
VLLM_CPP_INDEXTTS2_S2MEL=... VLLM_CPP_INDEXTTS2_BIGVGAN=... \
VLLM_CPP_INDEXTTS2_GPT=... VLLM_CPP_INDEXTTS2_TIKTOKEN=... \
  ./build/tests/test_indextts2_e2e
```

A REAL LIMITATION to know before using it: the reference clip is required and
then IGNORED. Its encoders are ported and their checkpoints are staged, but the
conditioning rows are zeros, so two different reference voices give the same
output today. `campplus::LoadCampplus` reads its weights but
`campplus::Forward` returns NaN on them, which is an open defect recorded in
the spec and blocks the wiring.

It asserts STRUCTURE, not quality: nothing is compared against vLLM-Omni, which
is unpinned (#633). The TOKENIZER it uses:
`tiktoken::LoadRanks` reads the shipped `.tiktoken` vocabulary and
`tiktoken::Encode` reproduces python tiktoken's ids exactly on the cases
gated, CJK included. The checkpoint now
LOADS through `vllm::multimodal::SpeechRegistry`, reports its family and its
22.05 kHz output rate, and states that a reference clip is required; asking
it to synthesize refuses by naming the one gap between text and the render
path, which is that the shipped vocabulary is tiktoken and this tree has no
reader for one. The pipeline itself renders on the real
checkpoints: the talker emits its own mel codes, the length regulator resamples
them to the mel frame rate, a classifier-free guided CFM Euler loop integrates
the S2Mel estimator, and BigVGAN turns the mel into a bounded 22.05 kHz
waveform. `indextts2::Render` is the entry point, and
`test_indextts2_render` drives it end to end when the three checkpoint
environment variables are set. It is NOT yet measured against the vLLM-Omni
oracle, which is unpinned (#633), so nothing here is a quality claim. Inferring the emotion from a clip instead of stating it needs a
Conformer and a Perceiver that are not ported.

`/v1/audio/speech` is served, but **only** by a server started with
`--speech-model`, and what it can render depends on the family that flag loads
(#1112). MiniMax-Music3 renders: a composed request returns a real 44100 Hz
stereo WAV (#852). **IndexTTS-2.5 does not**: its stages are ported and gated at
reduced dimensions, further stages are named as missing by the checkpoint's own
manifest, and loading the family refuses with a message naming the missing
pieces (#634). Without `--speech-model` the route is a 404 at the route table
rather than a runtime error, which is the accurate signal: the endpoint is opt
in, not absent. See
[the MiniMax-Music3 recipe](../models/minimax-music3.md).

`prompt_logprobs` is accepted on `/v1/completions` and `/v1/chat/completions`
and the engine computes it, every prompt position is scored against the token
that followed it, accumulated across chunked prefill, but the **response body
does not carry it yet**: emitting it needs the OpenAI `echo` wiring, which is
not done. Until then it is reachable through the library
(`RequestOutput.prompt_logprobs`), not over HTTP. `logprobs`/`top_logprobs` on
GENERATED tokens are emitted normally.

That computation is gated on the **CPU** backend only. A step that owes prompt
logits takes the full-logits route, and on that route the sampler is handed a
host-resident logits buffer carrying the accelerator's device label, sound on
unified memory, and **not yet verified on CUDA at all, discrete or otherwise**.
Treat `prompt_logprobs` on a GPU build as unverified until that gate runs; the
mechanism and the exact owed invocation are in
[`.agents/specs/prompt-logprobs.md`](../../.agents/specs/prompt-logprobs.md)
(risk 4 and the `PENDING` CUDA smoke gate). Requests that do NOT set it are
unaffected on every backend, the route is only taken for a step where some
request asked.

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. See
Use the [model recipe index](../USAGE.md#find-a-model-recipe) to open the current combined
MiniMax-H3 video and audio workflow.

`/v1/audio/speech` is registered **only** when the server was started with
`--speech-model`; without it the route is absent (404) and the server is
identical to one built before it existed. See the
[MiniMax-Music3 recipe](../models/minimax-music3.md).

### `max_tokens`: what a non-positive value means

Some clients (Hermes among them) send `max_tokens: -1` to mean "no client-side
limit". A non-positive `max_tokens`, or `max_completion_tokens` on
`/v1/chat/completions`, which takes precedence, is treated as **unset**, not as
an error and not as a clamp to some constant. Unset then generates up to
`max_model_len` minus the prompt length, mirroring vLLM.

That distinction is load-bearing for long-context requests: substituting a
constant would cap exactly the request that asked to be left unlimited, and the
client would see `finish_reason: length` with no way to tell it apart from a
limit it set itself. Use `VT_SERVER_MAX_NEW_TOKENS` when you want a serving-side
ceiling.

### Which token ids stop a generation

Stop ids come from two files in the checkpoint, not one. `config.json`'s
`eos_token_id` supplies the **primary** eos id, and the sibling
`generation_config.json` supplies **secondary** stop ids that are usually a
superset of it. Gemma-4-26B is the clearest case:

```
config.json             eos_token_id: [1, 106]
generation_config.json  eos_token_id: [1, 106, 50]
```

Both are read, mirroring vLLM's default `--generation-config auto`. The
secondary ids are merged into the request's `stop_token_ids`, so a chat model
stops on its turn-level token rather than running to the length cap. A missing
or malformed `generation_config.json` is a silent no-op.

`ignore_eos: true` suppresses **all** of them, primary and secondary alike, and
generation then runs to the token budget. The ids still count toward
`min_tokens` masking either way, so `min_tokens` cannot be satisfied by emitting
a stop token early.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size. **Must be a multiple of 16**, the attention backends' `get_kv_cache_shape` refuses anything else, and the server now rejects it at startup rather than throwing during engine init |
| `--num-blocks N` | `0` (auto, resolves to `256`) | KV block count, and vLLM's `num_gpu_blocks_override`. It wins over every other sizing knob. `0` means auto, which uses `--kv-cache-memory` when that is set and otherwise falls back to `256` blocks |
| `--kv-cache-memory BYTES` | `0` (unset) | Absolute KV-pool size in bytes, vLLM's `kv_cache_memory_bytes`. The block count is this budget divided by the model's own bytes per block, summed across its KV groups, so it is correct on MLA and heterogeneous-KV architectures too. It ignores `--gpu-memory-utilization`, as vLLM does. A budget smaller than one KV block is refused at startup |
| `--gpu-memory-utilization F` | `0.92` | **Accepted, and it does not size anything yet.** See [What `--gpu-memory-utilization` does not do yet](#what---gpu-memory-utilization-does-not-do-yet) |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Max concurrent sequences (also sizes the HTTP worker pool). Was `8`, which put a c8 client exactly on the batch ceiling; vLLM's own default is 1024, which we do not mirror because this also caps the padded decode-graph set. On a GDN/Mamba model under speculative decoding this also multiplies the recurrent state, which is sized `max-num-seqs x (k+1)`; an unservable budget is refused at load with the arithmetic |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [the SGLang compatibility guide](../SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (42 names over 38 families). `auto` detects from the chat template, `none` disables. For `gemma4`, OpenAI chat uses the text-seam parser (wrapped `<\|tool_call>` **or** bare `call:NAME{ARGS}`) so free-form / detokenized tool bodies still become `tool_calls`. **`inkling` needs `"skip_special_tokens": false` on the request today**, its whole grammar is special tokens and we have no `adjust_request` seam to force the flag off for you, so at the `true` default the detokenizer strips the markers before the parser runs ([#695](https://github.com/mudler/vllm.cpp/issues/695)). `--reasoning-parser inkling` is not registered at all ([#703](https://github.com/mudler/vllm.cpp/issues/703)) |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`, `muse_glimmer`, `qwen3`, `mimo`). `auto` detects, `none` disables. `qwen3` and its `mimo` alias are the engine-backed adapter (one upstream class, two registry names): thinking is ON, so a marker-less stream is reasoning and a `<tool_call>` ends reasoning with no `</think>`. `auto` never selects it, a generic `<think>` template resolves to `think_auto`, which is the right default for hybrid-thinking models that may answer with no think block at all |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [the KV offload guide](../KV-OFFLOAD.md) |
| `--offload-config '<json>'` | (unset) | Weight offload, the same JSON vLLM's `OffloadConfig` takes (distinct from `--kv-transfer-config`, which offloads KV blocks). Parsed and validated at startup, so a malformed document, an unknown backend, an unknown TOP-LEVEL key (the four legal ones are `offload_backend`, `uva`, `prefetch` and `vllm_cpp`) or a validator violation is refused before any model I/O; a backend/field mismatch is a warning, as upstream. **Enabling it fails startup on every model today**: no loader consults the offloader, so the engine refuses the configuration by architecture name rather than accept a budget that frees nothing. A config that leaves offloading disabled still parses and reports normally. On unified memory such as GB10 offload cannot help at all, because host and device share one pool. See [weight offload](../WEIGHT-OFFLOAD.md). The same document also carries the **`vllm_cpp` key**, which governs the tier BELOW this one, weights borrowed out of the file mapping rather than moved to host RAM, and which is live rather than refused: see [expert streaming](../guides/expert-streaming.md). A `vllm_cpp`-only document does not enable vLLM's offload backends and is not subject to the refusal above. The flag is accepted by `vllm-server` (the generate/chat and the pooling/embedding paths), by `vllm-cli`, and by the C ABI; the server's transcription-only path REFUSES it by name, because that path builds no engine and could only accept the document and ignore it ([#1195](https://github.com/mudler/vllm.cpp/issues/1195)) |
| `--speculative-config '<json>'` | (unset) | Configure `mtp`, `dflash`, `ngram`, or `dspark` speculative decoding. Accepted fields are `method`, `num_speculative_tokens`, `model`, `prompt_lookup_min`, `prompt_lookup_max`, `draft_sample_method`, and `rejection_sample_method`. The sampling defaults are `greedy` and `standard`; other values refuse at startup. Unknown fields, unsupported methods, incompatible targets, and invalid draft depths also refuse at startup. MTP defaults its depth to the checkpoint's `mtp_num_hidden_layers`. DSpark depth must cover the draft checkpoint's block. DFlash2 drafts refuse because the grouped convolution and candidate selector are missing. GGUF targets cannot use DSpark. See the [speculative decoding guide](../SPECULATIVE-DECODING.md) for method commands, validation rules, current gates, benchmarks, and tracked limitations. |
| `--language-model-only` / `--no-language-model-only` | off | Disable all multimodal input by setting **every** modality limit to 0, mirroring vLLM's flag of the same name. It is not a "skip the encoder" switch: the server then **refuses** a multimodal request with ``400 At most 0 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit.`` It does **not** free VRAM yet, nothing gates tower construction on it ([#607](https://github.com/mudler/vllm.cpp/issues/607) wave L3) |
| `--limit-mm-per-prompt '<json>'` | (unset ⇒ 999 per modality) | Maximum multimodal input items per prompt, per modality, as the same JSON object vLLM's flag takes: `'{"image": 2, "video": 0}'`, or with profiling options `'{"video": {"count": 1, "num_frames": 32}}'` (the options are validated and ignored, they size dummy inputs for memory profiling, which this engine does not do). A limit can only **lower** what the model/seam supports, never raise it. Malformed JSON, a negative count, or an unknown option on `image` / `video` / `audio` is refused at startup rather than defaulted. An unknown option on any other modality name is dropped rather than refused, mirroring upstream, whose fallback `BaseDummyOptions` is the one such dataclass without `extra="forbid"`. Upstream's dotted spelling (`--limit-mm-per-prompt.image 2`) is not accepted here, as for `--kv-transfer-config` and `--speculative-config` |
| `--mmproj <mmproj-*.gguf>` | (unset) | The SECOND GGUF file: a `clip`-architecture multimodal projector beside a `.gguf` `--model`, spelled as llama.cpp spells it. It is read, validated and REFUSED BY NAME before the tokenizer and before any language-model weight byte, so a wrong file costs a message instead of a 17 GB map. Refused when `--model` is not a `.gguf` (a safetensors checkpoint carries its tower in its own shards), when the file's `general.architecture` is not `clip`, when its `clip.projector_type` is not `qwen3vl_merger`, and when it carries `v.patch_embd.weight` without `v.patch_embd.weight.1`, half the input features the temporal patch embedding needs, which cannot be completed without inventing the other half. A `muse-glimmer` projector gets MuseGlimmer's own recorded refusal. **The tower is loaded and held, and nothing runs it yet**: there is no multimodal request path over HTTP for a GGUF model, so today the flag buys you validation and a loaded tower, not an image answer ([#821](https://github.com/mudler/vllm.cpp/issues/821)). Auto-discovery of a sibling `mmproj*.gguf` is deliberately not implemented, a directory holding two unrelated models must not silently fuse them |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | off | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it (Gemma-4 and friends). Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking` |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

#### Accepted for recipe compatibility, these flags have NO effect

A published `vllm serve` line has to reach model load. The flags below appear in
most official [vllm-project/recipes](https://github.com/vllm-project/recipes)
commands, mean nothing to this engine, and are therefore **accepted and ignored**
rather than rejected. Each one prints a notice on startup naming itself and the
reason it does nothing, so a log never implies it took effect.

| Flag | Effect here | Why it is inert |
|---|---|---|
| `--enable-auto-tool-choice` | **none** | Tool parsing is already unconditional once `--tool-call-parser` resolves; there is no second gate to open. Note `--tool-call-parser` defaults to `hermes` here, where upstream's defaults to unset, so the two flags do not line up when the parser is omitted. Upstream's validation is still mirrored: combining it with `--tool-call-parser none` is refused, as in `vllm/entrypoints/openai/cli_args.py:395` |
| `--trust-remote-code` | **none** | It authorizes executing Python from the checkpoint. This engine has no Python runtime, so there is nothing to authorize, N/A by construction, not unimplemented |

The notice is on stderr at startup, one line per flag actually passed, so what
you see in a log matches this table:

```text
server: accepted '--trust-remote-code' for published-recipe compatibility; it has no effect here: no Python runtime, so there is no remote code to trust
```

The mirrored validation is reported before the parser dialect is checked, so a
contradiction is named as a contradiction rather than passing silently (`none` is
itself a valid selection):

```text
server: Error: --enable-auto-tool-choice requires --tool-call-parser
server: (--tool-call-parser none selects NO parser; name a parser, or drop --tool-call-parser to keep the hermes default)
```

This list is **enumerated, not a catch-all**. Any other unrecognized flag still
aborts with `server: unknown argument '<flag>'`, including flags that are inert
only because the capability is missing (`--tensor-parallel-size` and the other
parallelism flags), silently accepting those would let you believe you got
tensor parallelism when you did not.

#### What `--gpu-memory-utilization` does not do yet

The flag is accepted, keeps vLLM's exact name and fraction semantics, and is
then discarded. It does not size the KV pool. Passing
`--gpu-memory-utilization 0.85` gives the same 256-block pool as passing
nothing.

Turning a free-memory fraction into a block count needs a profile run that
measures what the weights and activations cost on the device first. That run is
not implemented. It is `ROAD-V1-MEM` M3, tracked by
[issue #83](https://github.com/mudler/vllm.cpp/issues/83), and it needs a GPU to
gate.

The flag is accepted rather than refused so that a published `vllm serve`
command line runs here unchanged. Setting it prints this warning at startup, so
a log never implies it took effect:

```text
vllm.cpp: WARNING --gpu-memory-utilization 0.85 was accepted but did NOT size the KV cache.
vllm.cpp:   The profile run that turns a free-memory fraction into a block count is not
vllm.cpp:   implemented yet (ROAD-V1-MEM M3, https://github.com/mudler/vllm.cpp/issues/83).
vllm.cpp:   The pool fell back to 256 blocks. To size it today, pass
vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or --num-blocks <n> for an
vllm.cpp:   exact block count.
```

To size the pool today, use `--kv-cache-memory` for an absolute byte budget or
`--num-blocks` for an exact count. A run that never sets the flag prints
nothing.

**Warning.** On a unified-memory board such as NVIDIA GB10, a fraction of
"device" memory is a fraction of the one pool the host shares, so it reserves
host RAM as well. A value of 0.85 has hard-rebooted a GB10 box. When M3 lands
and this flag starts to bind, choose the fraction on such a board against the
whole 119 GiB pool and leave the host its headroom. Until then the flag reserves
nothing, on any board.

#### Context length vs the KV pool

The KV pool holds `--num-blocks × --block-size` tokens, `256 × 32 = 8192` by
default. A request longer than that can never be scheduled, so the engine
refuses it early rather than leaving it in the waiting queue forever. Two checks
do that, mirroring vLLM:

- **At startup.** If `--max-model-len` is given and the pool cannot hold one
  sequence that long, the server exits with the sizes and the flags that close
  the gap (vLLM's `_check_enough_kv_cache_memory`). If it is **not** given, the
  serving length is auto-fitted down to what the pool holds and logged
  (vLLM's `_auto_fit_max_model_len`), so raising `--num-blocks` is what buys a
  longer context.
- **At admission.** A prompt at or past the resolved `max_model_len` is
  rejected with **HTTP 400** (`BadRequestError`) naming both lengths, exactly as
  vLLM's `_validate_prompt_len` does. It is never a finish reason and never a
  500.

Set `VT_ENGINE_STEP_LOG=1` to print a per-step engine heartbeat if you need to
confirm that a quiet engine is idle rather than stalled.

For a production deployment, use [LocalAI](https://localai.io), which can embed
engines like this behind a model gallery, multi-model serving, the full OpenAI
API surface, auth, and metrics.
