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
> *production* config (CUDA graphs + torch.compile) **both gate models measure
> ≥1.0× total throughput at their large-concurrency operating points**: the
> **35B at 1.02×** (Triton-AOT GDN build; 0.99× in the default pure-C++ build)
> and the **27B at 1.007× conc16 / 1.007× conc32** (vendored FA-2 prefill,
> default-on; per-rep spreads in *Status*) — with better TTFT/TPOT and less
> peak memory on both. The tables track real, tested support and are kept
> current as work lands (see `.agents/`).

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
  once it starts a call; ⚠️ the Qwen3.6 gate checkpoints' shipped chat template
  uses the **Qwen3-Coder XML** tool format and forced `<think>` blocks, which the
  current Hermes-JSON parsers and template engine do not yet handle — the
  `qwen3_coder` XML parser + reasoning parser + template-engine extensions are
  scoped in `.agents/mm-tools-scoping-2026-07-10.md`); **grammars / structured output** (JSON-schema,
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

# Fast-GDN CUDA build (the MVP-parity numbers): adds the Triton-AOT GDN kernels.
# They are VENDORED as pre-compiled cubins (src/vt/cuda/triton_aot_vendored/) —
# building needs ONLY a C compiler, no Python/Triton (regenerating the vendored
# artifacts is a maintainer task: scripts/regen-triton-aot.sh).
cmake -S . -B build -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON

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
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ **both run end-to-end on GB10; token-exact greedy gates pass**; ✅ throughput parity vs production vLLM at the MVP gates | ✅ 35B from real APEX k-quant `.gguf` on GB10 (greedy parity vs same-file llama.cpp oracle); 27B GGUF pending (no file exists) | ✅ engine + server + tools + grammars on CPU; ✅ **full paged stack + both greedy gates on GB10**; ✅ throughput gates passed |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | ✅ engine + serving end-to-end |
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121a) | ✅ **full paged stack running on GB10**: vendored torch-free kernels (cutlass NVFP4/FP8, cuBLASLt, WMMA attention/GDN, Triton-AOT GDN, FA-2 prefill, CUDA-graph decode); both 27B + 35B greedy gates pass token-exact; ✅ throughput gates passed vs production vLLM at the MVP operating points (default pure-C++ 35B remains 0.99×; see Status) |
| Metal | Apple Silicon (MLX vs native MSL under exploration) | 🗓 planned (post-MVP) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

### Kernels & target hardware

Every op has a portable **CPU reference** (the correctness oracle + the backbone
other backends port from) and a **CUDA** implementation that mirrors the kernel
vLLM actually runs (cited to upstream / its deps — flashinfer, cutlass, cuBLASLt).

| Op / kernel | Implementation (upstream it mirrors) | CPU | CUDA · Blackwell (GB10, sm_121a) |
|---|---|:--:|---|
| Dense **W4A4** GEMM | cutlass fp4×fp4 (sm120a) + fp8 W8A8 — vLLM `cutlass_scaled_mm` | ✅ ref | ✅ |
| MoE **W4A16** GEMM | Marlin + fp4-resident — vLLM `marlin` / `fused_moe` | ✅ ref | ✅ |
| FP8 / bf16 projection GEMM | cuBLASLt col-major-TN → `nvjet_sm121` | ✅ ref | ✅ |
| Prefill attention | **vendored FlashAttention-2** `flash_fwd_splitkv` (vllm-project/flash-attention @ 2c839c33, the exact kernel vLLM runs; default-on for the bf16 head-256 path, 3.7× vs our WMMA) with the flash-style WMMA kernel as the portable fallback | ✅ ref | ✅ |
| Decode attention (paged) | FlashInfer-style paged, GQA-fused | ✅ ref | ✅ |
| GDN / linear-attn (chunk) | tensor-core WY solve — FLA `chunk_delta` | ✅ ref | ✅ |
| RMSNorm(+residual) → fp4 quant | 2 kernels, PARITY with production vLLM (measured: its Inductor `fused_add_rms_norm` triton kernel + extern `scaled_fp4_quant`; no fp4 norm-quant fusion exists in vLLM 0.24) · fp8: fused `RmsNormQuantFp8` (35B) | ✅ ref | ✅ |
| Activation fp4 quant | HW `cvt.e2m1x2` PTX (vLLM `nvfp4_utils`) / software ladder | ✅ ref | ✅ ladder · HW-PTX 🚧 A/B |
| CUDA-graph decode | captured decode step (vLLM cudagraph) | — | ✅ (both models) |
| Sampling (greedy/top-k/top-p/penalties) | vLLM V1 sampler, on-GPU sort-free | ✅ | ✅ |
| RMSNorm / RoPE / SwiGLU | fused elementwise | ✅ | ✅ |

Arch fallbacks: cutlass fp4×fp4 needs **sm_120a+**; WMMA paths need **sm_80+**;
cuBLASLt and the software fp4 ladder are portable across CUDA arches. Non-CUDA
backends (Metal/Vulkan/ROCm/XPU) are 🗓 post-MVP and will port from the CPU ref.

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | ✅ **both running on GB10**: W4A16 MoE (35B, Marlin + fp4-resident) and W4A4 dense (27B, cutlass sm120a fp4×fp4 + fp8-W8A8 attn/GDN); token-exact greedy gates pass |
| GGUF quants (Q4_K/Q5_K/Q6_K/Q3_K, Q8_0, F32) | ✅ 35B (qwen35moe) loads + serves from a single real `.gguf` on GB10 (dequant byte-exact vs ggml; greedy token-for-token vs a same-file llama.cpp oracle); 27B GGUF + IQ2_S/IQ4_XS i-quants + NVFP4 GGUF extension type pending (no real files exist yet) |
| fp8, MXFP4 | 🗓 planned (post-MVP T1) |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Where we are (GB10 / DGX Spark, sm_121a):** both NVFP4 gate models run the full
  paged engine end-to-end with **token-exact greedy gates passing**, and all CUDA
  kernels are validated on real hardware. **Throughput, measured vs vLLM on the same
  workload:** we **beat vLLM run `--enforce-eager`** on both models; against vLLM's
  *production* config (CUDA graphs + torch.compile — the honest bar) **both gate
  models measure ≥1.0×**: the **35B at 1.02× total throughput with better TTFT (−4%)
  and TPOT (−2%)** in the Triton-AOT GDN build (`-DVLLM_CPP_TRITON=ON`; 0.99× in the
  default pure-C++ build), and the **27B at 1.007× conc16 / 1.007× conc32** (see
  below), while we use less peak memory on both (27B run: 61.8 vs 76.2 GB at the
  identical workload/recipe). The GDN chunk-kernel
  **codegen** gap (~1.9× vs vLLM's Triton/FLA) was closed by a sanctioned, bounded
  **Triton AOT fast-path** (FLA kernels verbatim → cubin, **vendored per-arch as
  generated C** so the build needs no Python/Triton — regen is a maintainer task
  via `scripts/regen-triton-aot.sh`; the runtime stays Python/Triton-free and the
  portable C++ kernels and CPU reference remain the fallback — see the porting
  inventory §9). The 27B measures **1.0072× at conc16
  and 1.0071× at conc32** (764.3 vs 758.8; 1051.2 vs 1043.9 tok/s; 7/5-rep means vs
  fresh same-hour graphed denominators; conc32 5/5 reps ≥1.0×, conc16 6/7 with one
  0.996 rep disclosed): a production-codegen dump + kernel profile of vLLM settled
  the residual — the norm→quant sites were already at parity; the real gaps were
  the **bf16 GEMM layout** (the GDN in_proj now runs the cuBLASLt TN `nvjet TNNN`
  class vLLM's `F.linear` gets), the **fused attention preamble** (default-on for
  the 27B, reading a cos/sin cache), the **tiled causal-conv** (default-on), and
  finally the prefill **attention kernel** — closed by the **vendored
  FlashAttention-2** `flash_fwd_splitkv` port (the exact kernel vLLM runs on GB10),
  wired natively-bf16 (zero cast kernels) and sync-free, measured **3.7× per-kernel
  vs our WMMA** and **+1.5%/+0.5% e2e** (default-on; `VT_FA2_PREFILL=0` falls back
  to the portable WMMA kernel).
- **The MVP throughput gate is passed**, and the rule remains full parity: ≥1.0× vs
  *production* vLLM on every axis — total/output throughput, TTFT, TPOT, memory —
  for both gate models at their large-concurrency operating point. We do not stop at
  "near parity." Future performance work stays on the same **execution-traced
  roadmap**: `nsys`- and torch-profile vLLM's *actual* kernels (not just its source),
  mirror what it runs, then A/B-measure and gate-check. Recent measured,
  token-exact, default-on wins: register-tiled GDN `delta_h` + `cp.async` ring, a
  blocked tensor-core WY triangular inverse, fused fp8 RMSNorm→quant (quantize-once),
  bf16 GDN input I/O, per-shape fp4-GEMM autotune, and FA-2 prefill attention.
  The full record is in the [parity ledger](.agents/parity-ledger.md).
  `scripts/dgx-bringup.sh` runs the CUDA suite + the gates on dgx.casa.
- No PyTorch / no ggml at build or runtime (ggml is a *format* reference for GGUF
  only). Original runtime/packaging components (the `vt` op runtime, the minja
  chat-template engine, the native grammar backend, the C API) are documented as
  deviations in the [porting inventory](.agents/porting-inventory.md) §9.

## Project record

Development is fully documented under [`.agents/`](.agents/) (canonical index in
[AGENTS.md](AGENTS.md)): the [gates](.agents/gates.md), the [porting inventory](.agents/porting-inventory.md)
vs upstream vLLM, the [parity ledger](.agents/parity-ledger.md), the [roadmap](.agents/roadmap.md),
and the [upstream sync protocol](.agents/upstream-sync.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [`NOTICE`](NOTICE)
for third-party attributions (this project is a C++ port of the Apache-2.0-licensed
[vLLM](https://github.com/vllm-project/vllm) and vendors additional components under
their respective licenses).
