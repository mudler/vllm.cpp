# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch, no ggml at runtime. It mirrors the V1 architecture and
targets the same algorithms and serving surface; the tables below distinguish
the implemented subset from the remaining parity work. It is usable as a
library (llama.cpp-style) with a C API, an example CLI, and an
OpenAI-compatible server.

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
> current as work lands (see `.agents/`). CPU multithreaded dispatch is now
> correctness-gated, but its real-file throughput/RSS checkpoint is still open.

## What's implemented (CPU, behaviorally tested)

The vLLM **V1 engine path needed by the two Qwen gate-model text workloads**
runs end-to-end on CPU:

- **Engine core** — the unified token-budget `Scheduler` (chunked prefill, FCFS
  preemption), hash-based prefix-caching `KVCacheManager` + hybrid KV coordinator
  (full-attn + GDN/mamba-state groups), the persistent `InputBatch` + step-input
  build, `EngineCore.step()` loop, and the `LLMEngine` (`add_request`/`step`/
  `generate`).
- **Local-attention KV + sliding-window operator** — `SlidingWindowSpec` and
  `ChunkedLocalAttentionSpec` now have registry-backed managers, bounded
  admission, local prefix reuse/recycling, and hybrid-disabled full-allocation
  fallbacks, all CPU/property/sanitizer-gated. The generic sliding window also
  propagates through config, the CPU reference, separately specialized
  portable-CUDA kernels, and vendored FA2 local dispatch; the sm_121a
  translation units compile cleanly. Chunked-local attention now has its cached
  ordinary-backend wrapper, exact virtual Q/K batches, reusable block-table
  gather/update plan, cudagraph rejection and generic spec seam; all six pinned
  shapes plus randomized causal-mask properties are CPU/sanitizer-gated.
  Neither mode is yet user-visible model support: GPU runtime, positive-model
  e2e, oracle, trace, and performance/memory gates remain open.
- **Long-context RoPE foundation (YaRN/MRoPE/Llama 3/Phi-3 LongRoPE)** — modern
  and legacy Hugging Face RoPE parameters now normalize into a typed, memoized
  factory with f32/bf16 caches.
  Plain YaRN and its `mrope_section` branch apply caller-supplied caches for
  NeoX/GPT-J layouts and 1-D or 3-axis contiguous/interleaved positions; Llama 3
  adds its exact unchanged/smoothed/scaled frequency bands, including equal
  low/high factors. Phi-3 LongRoPE builds its short and long factor-array caches
  and selects one globally from runtime max length, including explicit mscale
  overrides. Twelve exact pinned-source oracle fixtures, CPU references, and
  ASan/UBSan pass. The CUDA implementation is source-complete but has not yet
  been compiled or run on an uncontended GPU, and no feature-positive model
  path is claimed yet.
- **Model forward** — Qwen3.6-35B-A3B hybrid (GDN×3 + gated full-attention, 256-
  expert MoE + shared expert), with **paged attention** (block-paged KV cache) +
  batched GDN. Loads from **safetensors** (NVFP4/FP8→bf16) and **GGUF**
  (k-quant→bf16 for the qwen35moe APEX files). Model selection now uses a
  central ordered, type-erased registry over the full Hugging Face
  `architectures` list for both implemented Qwen IDs; unsupported,
  previously-supported, and out-of-tree IDs are rejected explicitly instead of
  being guessed from `num_experts`. The registry's CPU suite is green; its
  unchanged two-model GPU path is awaiting the queued no-regression rerun.
- **CPU execution** — persistent ggml-derived native threadpool with chunked
  GEMM and row/batch op dispatch; `VLLM_CPP_CPU_THREADS` selects the worker
  count. Outputs are byte-identical at 1/3/20 threads and the concurrency suite
  is ThreadSanitizer-clean. The real-file ≥10x speed/RSS checkpoint still needs
  an exclusive idle-host rerun, and GGUF compute-in-quant remains open.
- **Sampler** — greedy (bit-exact vs `torch.argmax`) plus temperature,
  top-k/top-p, penalties, min-p and internal logit-filter/logprob primitives.
  `n`, full random/logprob payload behavior, and request/API wiring for the
  long-tail controls remain open; some paths synchronize results to the host.
- **OpenAI server** — basic `/v1/completions` and `/v1/chat/completions`
  transport (non-streaming plus **live incremental SSE** over an `AsyncLLM`
  engine thread), concurrent requests through one scheduler, and disconnect
  abort; `/v1/models`, `/health`, and `/version` are present, while `/health`
  currently reports process liveness rather than probing engine health. The
  async path passes the full CPU suite and ThreadSanitizer; its post-change
  GB10 token/latency/throughput/memory gates are still pending. Chat templates
  use a bounded minja-subset Jinja engine.
  **Tool/function calling** provides Hermes-style `<tool_call>` parsing,
  streaming tool-call deltas, and `tool_choice` auto/required/named — `auto`
  is *relaxed*: the model may reply in plain text or call a tool, constrained only
  once it starts a call; ⚠️ the Qwen3.6 gate checkpoints' shipped chat template
  uses the **Qwen3-Coder XML** tool format and forced `<think>` blocks, which the
  current parser (including the local `qwen3` Hermes alias) and template engine
  do not yet handle — the
  `qwen3_coder` XML parser + reasoning parser + template-engine extensions are
  scoped in `.agents/specs/mm-tools-scoping-2026-07-10.md`); **grammars / structured output** (JSON-schema,
  `json_object`, regex, choice, GBNF — a native constrained-decoding engine with
  an explicitly bounded JSON-schema/structural-tag subset behind vLLM's
  structured-output seam).
- **Library packaging** — a stable 17-symbol C ABI (`include/vllm.h`),
  `libvllm` shared + static, an example CLI and OpenAI server, and a
  `dlopen`/FFI consumption path (for LocalAI-style embedding via cgo/purego).
  Alongside the blocking completion calls, additive request handles provide
  nonblocking submit/cancel/wait/done/error/free with per-delta callbacks.

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

# Serve a supported Qwen text checkpoint (safetensors dir or supported 35B .gguf)
./build/examples/server --model /path/to/Qwen3.6-35B-A3B --port 8000
# then: curl localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'

# Or a one-shot completion via the CLI (drives the C ABI end-to-end)
./build/examples/vllm-cli --model /path/to/model --prompt "The capital of France is"
```

As a library, link `libvllm` (or `dlopen` it) and drive the C API in
`include/vllm.h`: use `vllm_complete` / `vllm_complete_stream` for blocking
calls, or `vllm_request_submit` plus the request lifecycle functions for
nonblocking concurrent streams.

## Supported model architectures

| Architecture | Families | Safetensors | GGUF | Status |
|---|---|---|---|---|
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ **text submodels** run end-to-end on GB10; token-exact greedy gates pass; ≥1.0× throughput parity passed. The upstream wrappers are multimodal; their vision path is not implemented. | ✅ 35B text path from real APEX k-quant `.gguf` on GB10 (greedy parity vs same-file llama.cpp oracle); 27B GGUF pending (no file exists) | ✅ paged text engine + basic server/tool/grammar subsets; ✅ greedy and production-vLLM throughput gates on GB10 |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | 🟡 gate-model text engine + basic serving path end-to-end; multithreaded op dispatch (ggml-threadpool port, `VLLM_CPP_CPU_THREADS`) is 1/3/20-thread bit-identical and TSAN-clean. Its B4 real-file speed/RSS gate is pending an idle-host rerun; compute-in-quant GGUF speed remains open |
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121a) | ✅ **gate-model paged text stack running on GB10**: vendored torch-free kernels (cutlass NVFP4/FP8, cuBLASLt, FA-2, Triton-AOT GDN, Qwen-specific CUDA-graph decode); both 27B + 35B greedy gates pass token-exact and meet the production-vLLM throughput gate |
| Other CUDA targets | vLLM's sm70/75/80/86/87/89/90/100/101/103/110/120 targets | 🗓 inventoried, **not yet built or validated here**; per-target kernel dispatch/AOT/build/correctness/trace/performance gates remain |
| Metal | Apple Silicon via MLX; custom MSL/MLX primitives for paged ops | 🗓 planned (M4 bring-up host available) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

### Kernels & target hardware

The ops exercised by the two GB10 text gates have portable references and CUDA
implementations grounded in vLLM or its dependencies. This is not a claim of
complete vLLM-kernel or cross-architecture coverage: the canonical kernel and
CUDA-target inventories track unimplemented and untraced families separately.

| Op / kernel | Implementation (upstream it mirrors) | CPU | CUDA · Blackwell (GB10, sm_121a) |
|---|---|:--:|---|
| Dense **W4A4** GEMM | cutlass fp4×fp4 (sm120a) + fp8 W8A8 — vLLM `cutlass_scaled_mm` | ✅ ref | ✅ |
| MoE **W4A16** GEMM | Marlin + fp4-resident — vLLM `marlin` / `fused_moe` | ✅ ref | ✅ |
| FP8 / bf16 projection GEMM | cuBLASLt col-major-TN → `nvjet_sm121` | ✅ ref | ✅ |
| Prefill attention | **vendored FlashAttention-2** `flash_fwd_splitkv` (vllm-project/flash-attention @ 2c839c33, the exact kernel vLLM runs; default-on for the bf16 head-256 path, 3.7× vs our WMMA) with the flash-style WMMA kernel as the portable fallback | ✅ ref | ✅ |
| Decode attention (paged) | FlashInfer-style paged, GQA-fused | ✅ ref | ✅ |
| GDN / linear-attn (chunk) | tensor-core WY solve — FLA `chunk_delta` | ✅ ref | ✅ |
| RMSNorm(+residual) → fp4 quant | the **traced gate workload** dispatches Inductor `fused_add_rms_norm` + external `scaled_fp4_quant`; FlashInfer also contains fused Add/RMSNorm+FP4 kernels that remain in the upstream kernel inventory · fp8: fused `RmsNormQuantFp8` (35B) | ✅ ref | ✅ gate path |
| Activation fp4 quant | HW `cvt.e2m1x2` PTX (vLLM `nvfp4_utils`) / software ladder | ✅ ref | ✅ ladder · HW-PTX 🚧 A/B |
| CUDA-graph decode | Qwen-specific captured decode step (vLLM cudagraph) | — | 🟡 explicit 35B capture gate; 27B evidence backfill open |
| Sampling (greedy/top-k/top-p/penalties) | vLLM V1 ordering subset; some token/logprob paths synchronize to host | ✅ | 🟡 bounded subset |
| RMSNorm / default RoPE / SwiGLU | fused elementwise | ✅ | ✅ |
| Scaled RoPE supplied-cache rotation | pinned vLLM typed cache/factory + YaRN/MRoPE/Llama 3/Phi-3 LongRoPE construction and cache lookup/rotation | ✅ ref + twelve oracle fixtures | 🚧 source present; compile/runtime gate open |

Only **GB10/sm_121a is built, traced and gated today**. Source-level fallbacks
suggest routes for other SMs, but none counts as support until its full build,
correctness, trace and performance block passes. Non-CUDA backends
(Metal/Vulkan/ROCm/XPU) are post-MVP and will port from the CPU reference.

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | ✅ **both running on GB10**: W4A16 MoE (35B, Marlin + fp4-resident) and W4A4 dense (27B, cutlass sm120a fp4×fp4 + fp8-W8A8 attn/GDN); token-exact greedy gates pass |
| GGUF materialization (F32, Q4_0, Q8_0, Q3_K/Q4_K/Q5_K/Q6_K) | 🟡 load-time bf16 materialization; synthetic layout tests plus real 35B APEX Q3/Q4/Q5/Q6/Q8 greedy parity vs same-file llama.cpp. CPU ops now use correctness-gated multithreaded dispatch, but its real-file speed/RSS gate and direct compute-in-quant path remain open; F16/BF16, Q2_K, IQ/TQ/Q1, MXFP4 and NVFP4 execution remain open. |
| FP8 | 🟡 the 35B ModelOpt static per-tensor W8A8 projection slice is native and gate-passing; generic FP8 modes/dispatch and FP8 KV remain planned |
| MXFP4 / MXFP8 | 🗓 planned, including MLX-native modes on Apple |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Where we are (GB10 / DGX Spark, sm_121a):** both NVFP4 gate-model **text paths** run the full
  paged engine end-to-end with **token-exact greedy gates passing**, and the CUDA
  kernel paths exercised by those workloads are validated on real hardware. **Throughput, measured vs vLLM on the same
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
  via `scripts/regen-triton-aot.sh`; generation is pinned to the destination
  architecture and byte-reproducible across source paths). The runtime stays
  Python/Triton-free and the portable C++ kernels and CPU reference remain the
  fallback. The upstream-faithful bf16 `chunk_o` artifacts are available behind
  `VT_GDN_OUT_BF16=1`; f32 remains the default pending a fresh current-main
  two-model A/B. GDN reuses grow-only, queue-owned scratch, with the two
  `VT_GDN_TRITON_*_POOL=0` toggles retaining same-binary fallbacks (see the
  porting inventory §9). The 27B measures **1.0072× at conc16
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
- **The MVP throughput gate is PASSED** (≥1.0× vs *production* vLLM on every axis —
  total/output throughput, TTFT, TPOT, memory — both gate models, at their large-
  concurrency operating point). Post-MVP optimization remains execution-traced:
  we `nsys`- and torch-profile vLLM's *actual* kernels (not just its source) and
  mirror what it runs. Recent measured,
  token-exact, default-on wins: register-tiled GDN `delta_h` + `cp.async` ring, a
  blocked tensor-core WY triangular inverse, fused fp8 RMSNorm→quant (quantize-once),
  and bf16 GDN input I/O. A per-shape fp4-GEMM autotune (**+5.8%** on the 27B at high
  concurrency) is **default-on** (`VT_FP4_AUTOTUNE=0` opts out). Every step is
  A/B-measured and
  gate-checked; the full record is in the [parity ledger](.agents/parity-ledger.md).
  `scripts/dgx-bringup.sh` runs the CUDA suite + the gates on dgx.casa.
- **CPU status:** the native threadpool and chunked op dispatch pass the complete
  suite at 1/3/20 workers and a focused ThreadSanitizer run. This is not yet a
  speed claim: the required same-binary 1-vs-20 Qwen3.5-2B GGUF throughput/RSS
  series was deferred because the recovered 20-core host was contended by
  unowned inference services. It remains `GATING` until an exclusive idle-host
  run reproduces the ≥10x prefill/decode floor and refreshes llama.cpp.
- No PyTorch / no ggml at build or runtime (ggml is a *format* reference for GGUF
  only). Original runtime/packaging components (the `vt` op runtime, the minja
  chat-template engine, the native grammar backend, the C API) are documented as
  deviations in the [porting inventory](.agents/porting-inventory.md) §9.
- The CUDA low-concurrency comparison against SGLang is specified and pinned to
  v0.5.13, but **no SGLang performance result is claimed yet**. The unchanged
  27B checkpoint has a plausible native path; exact 35B mixed-quant loading must
  be proven. True incremental async HTTP streaming now exists, but binding
  TTFT/ITL numbers wait for the post-W2 GB10 online gate
  (`SERVE-ASYNC-LLM` remains `GATING`, not `DONE`).
- **Speculative decoding is not user-visible yet.** The first MTP leaf now has
  safetensors loaders and a standalone dense/MoE Qwen3.5 head with CPU tests,
  but its exact 27B+35B oracle gate is still queued and the scheduler,
  rejection sampler, GDN state snapshots, and API/config wiring remain open.
  GGUF files do not currently carry `mtp.*`; normal non-speculative execution
  is unchanged.
- **Local attention is not user-visible yet.** Sliding-window KV bookkeeping
  and backend-neutral CPU/portable-CUDA/FA2 compute-window leaves are
  implemented; chunked-local KV sizing, registry/grouping, fixed-chunk prefix
  reuse/recycling, admission and fallback allocation are also implemented, as
  is the virtual-batch wrapper that delegates each fixed chunk to an ordinary
  causal backend. Their CPU/property/sanitizer gates pass, and the
  sliding-window sm_121a translation units compile. GPU runtime, supported
  StarCoder2/Gemma3/Llama4-class model paths, oracle, trace, throughput, latency
  and memory evidence remain open; all four execution rows remain `GATING`, not
  supported.
- **Scaled long-context RoPE is not user-visible yet.** Typed config, memoized
  f32/bf16 cache construction, plain/MRoPE YaRN, Llama 3 frequency-band, and
  Phi-3 LongRoPE short/long formulas, plus the CPU supplied-cache operator,
  pass twelve exact pinned-source oracle fixtures and sanitizer checks. CUDA
  compile/runtime, Nomic/Qwen-VL YaRN, Llama-3.1 and Phi-3 consumers, both-
  engine traces, and correctness/performance/latency/memory closure remain
  open; `ATTN-YARN`, `ATTN-ROPE-LLAMA3`, and `ATTN-ROPE-LONGROPE` are
  `GATING`, not supported.

## Project record

Development is fully documented under [`.agents/`](.agents/) (canonical index in
[AGENTS.md](AGENTS.md)): the [gates](.agents/gates.md), the [porting inventory](.agents/porting-inventory.md)
vs upstream vLLM, the [parity ledger](.agents/parity-ledger.md), the [roadmap](.agents/roadmap_v1.md),
the [engine](.agents/engine-matrix.md), [model](.agents/model-matrix.md),
[quantization](.agents/quantization-matrix.md), [kernel](.agents/kernel-matrix.md),
and [backend](.agents/backend-matrix.md) matrices,
and the [upstream sync protocol](.agents/upstream-sync.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [`NOTICE`](NOTICE)
for third-party attributions (this project is a C++ port of the Apache-2.0-licensed
[vLLM](https://github.com/vllm-project/vllm) and vendors additional components under
their respective licenses).
