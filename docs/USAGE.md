# Using vllm.cpp

The complete surface: the CLI, the OpenAI-compatible server, and the library
(C ABI and C++). The [README](../README.md) carries the quickstart; this page is
the reference behind it. Per-capability lifecycle state is
[docs/STATUS.md](STATUS.md); measured numbers are
[docs/BENCHMARKS.md](BENCHMARKS.md).

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`.
- `tokenize` ([`examples/tokenize/main.cpp`](../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.

## OpenAI-compatible server

`server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../src/vllm/entrypoints/openai/).

```sh
build/examples/server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

Any OpenAI client works by pointing its `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

### Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../src/vllm/entrypoints/openai/api_server.cpp).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4) |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`) |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. See
[MiniMax-H3: video + audio generation](#minimax-h3-video--audio-generation).

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `8` | Max concurrent sequences (also sizes the HTTP worker pool) |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [docs/SGLANG-COMPAT.md](SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (40 names over 36 families). `auto` detects from the chat template, `none` disables |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`). `auto` detects, `none` disables |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

For a production deployment, use [LocalAI](https://localai.io), which can embed
engines like this behind a model gallery, multi-model serving, the full OpenAI
API surface, auth, and metrics.

## MiniMax-H3: video + audio generation

`/v1/videos` generates video with sound through the MiniMax-H3 diffusion model.
It speaks **OpenAI's Sora video shape**, so an OpenAI client works against it
unmodified, and it keeps the richer native knobs alongside.

```sh
build/examples/server --model /path/to/Qwen3.6-27B \
  --video-dit /path/to/h3-dit.gguf --video-vae /path/to/video-vae.safetensors \
  --audio-vae /path/to/audio-vae.safetensors \
  --video-vae-config video_vae/config.json --audio-vae-config audio_vae/config.json \
  --video-encoder /path/to/h3-encoder.gguf
```

```python
video = client.videos.create(model="sora-2-pro", prompt="a cat on a skateboard",
                             size="1280x720", seconds="8")
while client.videos.retrieve(video.id).status not in ("succeeded", "failed"):
    time.sleep(5)
open("out.mp4", "wb").write(client.videos.download_content(video.id).read())
```

### Request fields

| Field | Spelling | Meaning |
|---|---|---|
| `prompt` | both | Required. The text conditioning |
| `model` | OpenAI | Recorded and echoed back. A name this server does not serve is a `warning` on the job, never a rejection: the video model is chosen at startup |
| `size` | OpenAI | `"<width>x<height>"`, e.g. `"1280x720"`. Whole pixels, both positive |
| `seconds` | OpenAI | Duration, as a number or a numeric string (`8` and `"8"` both work) |
| `input_reference` | OpenAI | The image the video starts from. A filesystem path or a `data:` URL |
| `metadata` | OpenAI | Free-form string map, passed through untouched. Two keys are acted on: `input_reference_video` and `input_reference_audio` (see below) |
| `width`, `height` | native | Output geometry in pixels |
| `duration` | native | Duration in seconds |
| `task` | native | `t2va`, `fl2va`, `ref2va`; resolved from the inputs when omitted |
| `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed` | native | The H3 generation knobs. Accepted at the top level or nested under `extra_params` |

**Precedence.** When a body carries both spellings of one value, the **native
field wins**: `width`/`height` beat `size`, `duration` beats `seconds`. That
direction keeps every request that parses today meaning exactly what it meant
before. Both spellings are validated either way, so a malformed `size` is a 400
even when explicit `width`/`height` would have overridden it.

**`input_reference` maps to fl2va first-frame conditioning.** OpenAI documents
it as the image the generated video starts from, which is what fl2va expresses:
the supplied image is pinned as frame 0 of the output. H3's other image mode,
ref2va, prepends whole reference images as their own blocks (subject or style
guidance that never becomes a frame), so it stays reachable only through the
native `task` field and the `minimax-h3-gen` CLI. Two limits: the image must be
a **binary PPM (P6)** (no PNG or JPEG codec is vendored, the same residual the
chat multimodal path carries), and it must already be at the output resolution
(no image resampler is vendored). A mismatch is refused with the resolved
geometry in the message.

### Video and audio references (`metadata`)

H3 supports three reference modalities and OpenAI's schema has a slot for one,
so the other two enter through `metadata`, the standard OpenAI free-form string
map. Strict clients tolerate it, and no invented top-level field breaks their
schema validation. Unknown metadata keys are passed through untouched.

```jsonc
{
  "prompt": "the same scene, at dusk",
  "metadata": {
    "input_reference_video": "/tmp/vllm_h3_videos/job0",  // DIR of frame_%06d.ppm
    "input_reference_audio": "/tmp/voice.wav"             // 16-bit PCM WAV, or a data: URL
  }
}
```

`input_reference_video` is a **directory of `frame_%06d.ppm`**, which is exactly
what this server and `minimax-h3-gen` write, so one run's frames chain straight
into the next request. It is not a container file: no demuxer is vendored.

**A video reference is SILENT.** `MiniMaxH3EncodeReferenceVideo` emits a
`kVideoAudio` block with `ref_audio_t == 0`, so the clip contributes no sound of
its own. Supplying `input_reference_audio` alongside it attaches the audio to
that same block (one block carrying both, the layout upstream builds); without
it the reference is picture only. That is a real limitation, not an omission.

**Legal combinations.** fl2va keyframes and ref2va reference blocks are
exclusive in the pipeline itself
([`minimax_h3_pipeline.cpp`](../src/vllm/model_executor/models/minimax_h3_pipeline.cpp)),
so the request parser enforces the same rule and returns a 400 naming the
offending pair rather than dropping a reference you supplied.

| `input_reference` | `metadata.input_reference_video` | `metadata.input_reference_audio` | |
|---|---|---|---|
| (none) | (none) | (none) | t2va, prompt only |
| image | (none) | (none) | fl2va, the image is frame 0 |
| (none) | clip | (none) | ref2va, silent video reference |
| (none) | (none) | WAV | ref2va, audio reference |
| (none) | clip | WAV | ref2va, one block carrying both |
| image | clip and/or WAV | | **400**: keyframe and reference conditioning are exclusive |

The video reference needs `--video-vae` (the encoder half of the same file) and
the audio reference needs `--audio-vae`; both load lazily, once, on the first
request that asks for them.

### The job lifecycle

`POST /v1/videos` returns immediately with `{"id": "vid_1", "status": "queued"}`;
generation is minutes long, so the synchronous twin `POST /v1/videos/sync` exists
for scripts that would rather block. `GET /v1/videos/{id}` reports `queued`,
`running`, `succeeded` (with `output_path`) or `failed` (with `error`).

`GET /v1/videos/{id}/content` returns the finished MP4 with
`Content-Type: video/mp4`. An unknown id is a 404; a job that has not finished is
a **409** naming its current status rather than a truncated file; a failed job is
a 500 carrying its failure; an output that has since vanished from disk is a 500
rather than a 200 with zero bytes.

The library never spawns a process, so generation and muxing enter through a
caller-supplied `VideoRunner` callback (`examples/server/main.cpp` supplies one
that invokes `ffmpeg`, path configurable with `--video-ffmpeg`).

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 10`,
19 exported symbols) suitable for `dlopen` / FFI / LocalAI integration.

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

Chat templates render through the vendored google/minja engine, the same
renderer llama.cpp ships.

## Consuming it from C++

The higher-level surface lives under [`include/vllm/`](../include/vllm/).
`LoadedEngine::FromModelDir(...)`
([`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h))
hands back either the synchronous `LLMEngine`
([`v1/engine/llm_engine.h`](../include/vllm/v1/engine/llm_engine.h)) or the async
`AsyncLLM` ([`v1/engine/async_llm.h`](../include/vllm/v1/engine/async_llm.h)) that
the server itself uses.

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;
ep.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

The underlying portable tensor runtime is `vt::` ([`include/vt/`](../include/vt/)),
which carries no ggml or PyTorch dependency.

## Multimodal input (image, video, audio to text)

Multimodal input is served over the **OpenAI API**, not the CLI. `vllm-cli` is text-only:
`--model --prompt --max-tokens --temperature --top-k --top-p --seed --stream
--speculative-config --tokenizer-config`.

Start the server with a multimodal model, then send content parts on
`/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.chat.completions.create(model="Qwen3.6-27B", messages=[{"role": "user", "content": [
    {"type": "text",      "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
]}])
```

Accepted part types (`src/vllm/entrypoints/openai/chat_mm.cpp`):

| part type | modality |
|---|---|
| `image_url` | image |
| `video_url` | video |
| `input_audio` / `audio_url` | audio |

## MiniMax-H3: video + audio generation


Renders an MP4 with a stereo track. Weights: a GGUF DiT (use **Q4_K_M**), the Qwen3-VL-32B
encoder, and both VAEs.

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "A golden retriever runs across a sunlit beach, waves crashing behind it" \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 480 --width 864 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

Conditioning modes, all optional and mutually exclusive where noted:

```sh
--first-frame start.ppm --last-frame end.ppm   # pin the first and/or last frame (fl2va)
--ref-image subject.ppm                        # reference image, repeatable (ref2va)
--ref-video prev_workdir/                      # reference clip, reads frame_%06d.ppm
--ref-audio voice.wav                          # reference audio
--noise-aug 0.9                                # how hard a keyframe is pinned (1.0 = exact)
```

Reference frames are binary PPM, which is what this tool also **writes**, so one run's `--workdir`
feeds straight back in as `--ref-video` and clips chain. Convert anything else with
`ffmpeg -i in.png -pix_fmt rgb24 out.ppm`.

Useful for measurement: `--denoise-only` times the DiT loop without loading the VAEs,
`--dump-params` prints the geometry a checkpoint implies (manifest only, no weights),
`--save-embeds` writes the text conditioning so a second run can replay it with
`--prompt-embeds` and compare checkpoints on identical conditioning.

Served over HTTP too: pass `--video-dit` (plus the VAEs and configs) to `examples/server` and
`POST /v1/videos`, `POST /v1/videos/sync` and `GET /v1/videos/{id}` register. Without it the
routes stay unregistered.

