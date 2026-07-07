# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch, no ggml at runtime. Same V1 architecture, same
algorithms, same serving surface; usable as a library (llama.cpp-style) with a
C API, an example CLI, and an OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** Both NVFP4 gate models —
> **Qwen3.6-35B-A3B** (MoE hybrid) and **Qwen3.6-27B** (dense W4A4) — now run the
> full paged engine end-to-end on **NVIDIA GB10** (DGX Spark, sm_121a) with
> **token-exact greedy gates passing**, and throughput is measured against vLLM
> on real hardware. We **beat vLLM run eager** on both models; against vLLM's
> *production* config (CUDA graphs + torch.compile) we are at **near-parity**
> (≈0.90× on the 27B, ≈0.96× on the 35B) and are **actively closing the last few
> percent** to full parity via a measured, execution-traced roadmap — see
> *Status*. The tables track real, tested support and are kept current as work
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
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ **both run end-to-end on GB10; token-exact greedy gates pass**; throughput near-parity vs vLLM | 🚧 load path done (k-quant dequant + qwen35moe loader; real-file parity dgx-pending) | ✅ engine + server + tools + grammars on CPU; ✅ **full paged stack + both greedy gates on GB10**; 🚧 throughput parity being closed |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | ✅ engine + serving end-to-end |
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121a) | ✅ **full paged stack running on GB10**: vendored torch-free kernels (cutlass NVFP4/FP8, cuBLASLt, WMMA attention/GDN, CUDA-graph decode); both 27B + 35B greedy gates pass token-exact; throughput measured vs vLLM (near-parity, beats eager) — 🚧 **closing the last few % to full parity** |
| Metal | Apple Silicon (MLX vs native MSL under exploration) | 🗓 planned (post-MVP) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | ✅ **both running on GB10**: W4A16 MoE (35B, Marlin + fp4-resident) and W4A4 dense (27B, cutlass sm120a fp4×fp4 + fp8-W8A8 attn/GDN); token-exact greedy gates pass |
| GGUF quants (Q4_K/Q5_K/Q6_K/Q3_K, Q8_0, F32) | 🚧 dequant (byte-exact vs ggml) + qwen35moe load path done on CPU; real-file greedy parity dgx-pending; IQ2_S/IQ4_XS i-quants pending |
| fp8, MXFP4 | 🗓 planned (post-MVP T1) |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Where we are (GB10 / DGX Spark, sm_121a):** both NVFP4 gate models run the full
  paged engine end-to-end with **token-exact greedy gates passing**, and all CUDA
  kernels are validated on real hardware. **Throughput, measured vs vLLM on the same
  workload:** we **beat vLLM run `--enforce-eager`** on both models; against vLLM's
  *production* config (CUDA graphs + torch.compile — the honest bar) we are at
  ≈**0.90×** (27B, conc-16) / **0.96×** (35B), i.e. near-parity, consistent across
  concurrency.
- **The MVP is FULL throughput parity** (≥1.0× vs *production* vLLM on every axis —
  total/output throughput, TTFT, TPOT, memory — both gate models, at their large-
  concurrency operating point). We do not stop at "near parity." The remaining gap is
  being closed by an **execution-traced roadmap**: we `nsys`-profile vLLM's *actual*
  kernels (not just its source) and mirror what it runs — FlashAttention, cuBLASLt/
  `nvjet_sm121` GEMM steering, hardware `cvt.e2m1x2` fp4 quant, merged projection
  GEMMs, GDN chunk tuning — which an execution trace shows is enough to reach parity.
  Every step is A/B-measured and gate-checked; the full record is in the
  [parity ledger](.agents/parity-ledger.md). `scripts/dgx-bringup.sh` runs the CUDA
  suite + the gates on dgx.casa.
- No PyTorch / no ggml at build or runtime (ggml is a *format* reference for GGUF
  only). Original runtime/packaging components (the `vt` op runtime, the minja
  chat-template engine, the native grammar backend, the C API) are documented as
  deviations in the [porting inventory](.agents/porting-inventory.md) §9.

## Project record

Development is fully documented under [`.agents/`](.agents/) (canonical index in
[AGENTS.md](AGENTS.md)): the [gates](.agents/gates.md), the [porting inventory](.agents/porting-inventory.md)
vs upstream vLLM, the [parity ledger](.agents/parity-ledger.md), the [roadmap](.agents/roadmap.md),
and the [upstream sync protocol](.agents/upstream-sync.md).
