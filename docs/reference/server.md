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
## Endpoints

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

`/v1/audio/speech` is registered only when you start the server with
`--speech-model`. Without that flag, the route returns 404. MiniMax-Music3
returns a 44.1 kHz stereo WAV. See the
[MiniMax-Music3 recipe](../models/minimax-music3.md).

IndexTTS-2.5 loads from `--speech-model` and returns a 22.05 kHz mono WAV. Each
request must include `reference_audio` as a data URL for a 16-bit PCM mono WAV.
The current engine does not use the reference clip for conditioning, so the
clip does not select the output voice. The vLLM-Omni quality comparison is also
pending. See the
[owning model specification](../../.agents/specs/indextts-2-5.md) for the owned
limitations and verification evidence.

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
to one built without video support. Use the
[MiniMax-H3 recipe](../models/minimax-h3.md) for the combined video and audio
workflow.

## `max_tokens`: what a non-positive value means

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

## Which token ids stop a generation

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

## Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir\|file.gguf\|org/repo\|org/repo:QUANT>` | Required except when `--speech-model` selects a speech/music-only server | A local directory or `.gguf` file, opened as before, or a Hugging Face repository, which is fetched into the cache. The local forms are probed first, so a network call can never shadow a path on disk. See [Access Hugging Face checkpoints](../guides/hugging-face-access.md) |
| `--revision <ref>` | repository default branch | A branch, a tag, or a 40 character commit for a `--model org/repo`. vLLM's own flag, and there is no inline `org/repo@rev` syntax |
| `--download-dir <path>` | the resolved Hugging Face cache root | The directory that holds the `models--org--repo` folders. vLLM's own flag. Fetching a repository needs a build that carries transport layer security. See [Access Hugging Face checkpoints](../guides/hugging-face-access.md) for the three build options, the environment variables, and which release lanes carry it |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename, or the `org/repo` you typed | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size. **Must be a multiple of 16**, the attention backends' `get_kv_cache_shape` refuses anything else, and the server now rejects it at startup rather than throwing during engine init |
| `--num-blocks N` | `0` (auto, resolves to `256`) | KV block count, and vLLM's `num_gpu_blocks_override`. It wins over every other sizing knob. `0` means auto, which uses `--kv-cache-memory` when that is set and otherwise falls back to `256` blocks |
| `--kv-cache-memory BYTES` | `0` (unset) | Absolute KV-pool size in bytes, vLLM's `kv_cache_memory_bytes`. The block count is this budget divided by the model's own bytes per block, summed across its KV groups, so it is correct on MLA and heterogeneous-KV architectures too. It ignores `--gpu-memory-utilization`, as vLLM does. A budget smaller than one KV block is refused at startup |
| `--gpu-memory-utilization F` | `0.92` | **Accepted, and it does not size anything yet.** See [What `--gpu-memory-utilization` does not do yet](#what---gpu-memory-utilization-does-not-do-yet) |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Maximum concurrent sequences and HTTP workers. For GDN or Mamba speculative decoding, it also sizes recurrent state as `N x (k+1)`. The loader refuses an unsupported state budget. |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [the SGLang compatibility guide](../SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Select one of 42 registered names across 38 dialect families. `auto` detects the dialect from the chat template, and `none` disables parsing. Gemma-4 accepts wrapped or bare text calls. Inkling requests require `"skip_special_tokens": false`. The `inkling` reasoning parser is unavailable. |
| `--reasoning-parser <name>` | `none` | Select `think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`, `muse_glimmer`, `qwen3`, or `mimo`. `auto` detects from the template. The `qwen3` and `mimo` names share an adapter. |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [the KV offload guide](../KV-OFFLOAD.md) |
| `--offload-config '<json>'` | (unset) | Validate vLLM weight-offload fields and the `vllm_cpp` disk-residency tier. vLLM offload backends currently refuse at startup because loaders do not use them. A `vllm_cpp`-only config works. Transcription refuses this flag. See [Weight offload](../WEIGHT-OFFLOAD.md) and [Expert streaming](../guides/expert-streaming.md). |
| `--speculative-config '<json>'` | (unset) | Configure `mtp`, `dflash`, `ngram`, or `dspark`. Unknown fields, unsupported methods, incompatible targets, and invalid depths refuse at startup. Sampling defaults to `greedy` with `standard` rejection. The document also accepts a `vllm_cpp` extension object whose only key is `drafter_chain`, a preference-ordered list of speculators; it is validated but refused at startup, because nothing resolves a chain yet. See [Speculative decoding](../SPECULATIVE-DECODING.md). |
| `--language-model-only` / `--no-language-model-only` | off | Set every multimodal limit to zero. Multimodal requests then return HTTP 400. This option does not skip tower construction or free its memory. See [Multimodal input](../guides/multimodal-input.md). |
| `--limit-mm-per-prompt '<json>'` | `999` per modality | Set lower per-prompt limits with a JSON object such as `'{"image": 2, "video": 0}'`. Malformed JSON, negative counts, and unknown image, video, or audio options refuse at startup. Dotted flag syntax is unavailable. See [Multimodal input](../guides/multimodal-input.md). |
| `--mmproj <mmproj-*.gguf>` | (unset) | Load and validate a `clip` GGUF projector for a GGUF model. The server refuses incompatible model types, architectures, projector types, and incomplete temporal patch weights. HTTP multimodal inference for GGUF is unavailable, so this option does not produce image answers. See [Multimodal input](../guides/multimodal-input.md). |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | off | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it (Gemma-4 and friends). Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking` |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

### Accepted for recipe compatibility, these flags have NO effect

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

### What `--gpu-memory-utilization` does not do yet

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

### Context length vs the KV pool

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
