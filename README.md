# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch, no ggml at runtime. Same V1 architecture, same
algorithms, same serving surface; usable as a library (llama.cpp-style) with a
C API, an example CLI, and an OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The engine and serving layer are
> built and behaviorally tested on CPU; end-to-end validation on the real
> benchmark model (Qwen3.6-35B-A3B) and the throughput/parity gates run on
> NVIDIA GB10 (DGX Spark) and are **not yet confirmed** — see *Status &
> caveats*. The tables track real, tested support and are kept current as work
> lands (see `.agents/`).

## What's implemented (CPU, behaviorally tested)

The full vLLM **V1 engine** runs end-to-end on CPU, ported 1:1 from upstream:

- **Engine core** — the unified token-budget `Scheduler` (chunked prefill, FCFS
  preemption), hash-based prefix-caching `KVCacheManager` + hybrid KV coordinator
  (full-attn + GDN/mamba-state groups), the persistent `InputBatch` + step-input
  build, `EngineCore.step()` loop, and the `LLMEngine` (`add_request`/`step`/
  `generate`).
- **Model forward** — Qwen3.6-35B-A3B hybrid (GDN×3 + gated full-attention, 256-
  expert MoE + shared expert), with **paged attention** (block-paged KV cache) +
  batched GDN. Loads from **safetensors** (NVFP4/FP8→bf16) and **GGUF**
  (k-quant→bf16 for the qwen35moe APEX files).
- **Sampler** — the exact V1 pipeline: greedy (bit-exact vs `torch.argmax`),
  temperature, top-k/top-p, penalties, min-p/logit-bias/min-tokens, allowed/
  bad-words, logprobs.
- **OpenAI server** — `/v1/completions` and `/v1/chat/completions` (SSE streaming
  + non-streaming), `/v1/models`, `/health`, `/version`; chat templates (a
  minja-subset Jinja engine); **tool/function calling** (Hermes/Qwen `<tool_call>`
  parsers, streaming tool-call deltas, `tool_choice` auto/required/named — `auto`
  is *relaxed*: the model may reply in plain text or call a tool, constrained only
  once it starts a call); **grammars / structured output** (JSON-schema,
  `json_object`, regex, choice, GBNF — a native constrained-decoding engine behind
  vLLM's structured-output seam).
- **Library packaging** — a stable C ABI (`include/vllm.h`), `libvllm` shared +
  static, an example CLI and OpenAI server, and a `dlopen`/FFI consumption path
  (for LocalAI-style embedding via cgo/purego).

Every unit is adversarially reviewed and behaviorally tested; the CPU test suite
is green (`ctest`). See the [parity ledger](.agents/parity-ledger.md) for the
per-change record vs upstream vLLM.

## Quick start

```sh
# Build (CPU; add -DVLLM_CPP_CUDA=ON on an NVIDIA box, -DVLLM_CPP_SERVER=ON for the server)
cmake -S . -B build -DVLLM_CPP_SERVER=ON
cmake --build build -j
ctest --test-dir build            # the behavioral suite

# Serve an OpenAI-compatible endpoint (safetensors dir or a .gguf file)
./build/examples/server --model /path/to/Qwen3.6-35B-A3B --port 8000
# then: curl localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'

# Or a one-shot completion via the CLI (drives the C ABI end-to-end)
./build/examples/vllm-cli --model /path/to/model --prompt "The capital of France is"
```

As a library, link `libvllm` (or `dlopen` it) and drive the C API in
`include/vllm.h` (`vllm_engine_load` / `vllm_complete` / `vllm_complete_stream`).

## Supported model architectures

| Architecture | Families | Safetensors | GGUF | Status |
|---|---|---|---|---|
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ forward parity (greedy-verified on GB10, M0 exit) | 🚧 load path done (k-quant dequant + qwen35moe loader; real-file parity dgx-pending) | 🚧 engine + server + tools + grammars done on CPU; real-model + throughput gates on GB10 pending |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | ✅ engine + serving end-to-end |
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121) | 🚧 kernels ported (attention/GDN/MoE/sampler/cache); baseline ops parity-verified on GB10; the full paged stack + 35B greedy gate + throughput parity are **dgx bring-up pending** |
| Metal | Apple Silicon (MLX vs native MSL under exploration) | 🗓 planned (post-MVP) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | 🚧 W4A16 dequant + forward done (35B safetensors); W4A4 dense ~M2.2 |
| GGUF quants (Q4_K/Q5_K/Q6_K/Q3_K, Q8_0, F32) | 🚧 dequant (byte-exact vs ggml) + qwen35moe load path done on CPU; real-file greedy parity dgx-pending; IQ2_S/IQ4_XS i-quants pending |
| fp8, MXFP4 | 🗓 planned (post-MVP T1) |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Behavioral vs numerical:** the engine, sampler, server, tools and grammars are
  validated on CPU with a synthetic model + adversarial review (correct shapes,
  framing, semantics, streaming, error contracts). The **real Qwen3.6-35B-A3B**
  greedy/logits parity through the *paged batched engine*, all CUDA kernels, and
  the **throughput-parity-vs-vLLM** benchmark run on **GB10** and are the
  outstanding acceptance gates — `scripts/dgx-bringup.sh` runs the CUDA suite +
  the 35B gate on dgx.casa.
- No PyTorch / no ggml at build or runtime (ggml is a *format* reference for GGUF
  only). Original runtime/packaging components (the `vt` op runtime, the minja
  chat-template engine, the native grammar backend, the C API) are documented as
  deviations in the [porting inventory](.agents/porting-inventory.md) §9.

## Project record

Development is fully documented under [`.agents/`](.agents/) (canonical index in
[AGENTS.md](AGENTS.md)): the [gates](.agents/gates.md), the [porting inventory](.agents/porting-inventory.md)
vs upstream vLLM, the [parity ledger](.agents/parity-ledger.md), the [roadmap](.agents/roadmap.md),
and the [upstream sync protocol](.agents/upstream-sync.md).
