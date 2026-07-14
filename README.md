# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. Production performance parity is still open: the
> binding 27B comparison against vLLM v0.25.0 passes **55/124** required axes.
> At c2 our TTFT is better, but decode TPOT is **114.841 vs 108.274 ms**
> (**6.1% slower**). Exact trace plus source inspection finds one complete
> structural gap: our 48 GDN layers launch **193** BF16 GEMMs per B=2 window,
> while vLLM's merged qkvz/ba topology launches **97**. The accepted
> [implementation spike](.agents/specs/gdn-merged-input-projections.md) now has
> a `GATING` BA-only W1 implementation: one weight owner and one F32-output BA
> GEMM pass the clean pushed-SHA merged/split 27B gates **235/235 + 16/16**,
> packed-view CUDA replay/memcheck, and native-35B inertness. An upstream-dtype
> BF16-output experiment instead reproduced the rejected emulation stream
> (**233/235**), so exact rounding parity, the 145-vs-193 trace, and component
> performance remain open; no speed credit exists yet. Host memory also retains
> a **22.92 GiB CPU weight mirror**. No 35B performance result is claimed. See
> [Benchmarks](docs/BENCHMARKS.md) for the exact checkpoint.

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | ❌ FAILED / `GATING` | Immutable `3f256ab`: **55/124 pass, 69 fail**; pushed `581d335` closes BA W1 core correctness/safety with F32 output, but exact structure/performance and BF16-output parity are pending | Trace and component-gate merged BA, resolve its output rounding, then claim merged qkvz |
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | ⏸ BLOCKED | No current v0.25.0 performance result | Run only after all 27B axes pass |
| Host-memory parity | ❌ FAILED / diagnosed | Persistent host tensors account for **22.92 GiB**; source mmap pages overlap them during load | After the merged-projection component gates, stream weights into final device storage and re-run all memory axes |

The binding cache-off workload is input 1,024 → output 128, greedy, closed
loop, with three interleaved repetitions. Ratios are direction-normalized so
**1.0 or higher passes**.

| Concurrency | Axes passing | Total throughput: ours / vLLM | Ratio |
|---:|---:|---:|---:|
| 1 | 5/20 | 81.645 / 82.179 tok/s | **0.993504×** |
| 2 | 4/20 | 150.561 / 157.744 tok/s | **0.954464×** |
| 4 | 5/20 | 280.291 / 290.025 tok/s | **0.966438×** |
| 8 | 4/20 | 495.700 / 505.466 tok/s | **0.980678×** |
| 16 | 17/20 | 812.303 / 790.264 tok/s | **1.027889×** |
| 32 | 18/20 | 1121.955 / 1079.407 tok/s | **1.039417×** |

Total throughput beats vLLM at c16/c32, but parity requires every throughput,
latency, and memory axis to pass. Host PSS/RSS remain large open gaps; sampled
GPU memory and available-memory drop pass. The full per-axis table and exact
reproduction recipe are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Current performance track

| Work item | Present disposition |
|---|---|
| Binding gate | `3f256ab` remains **55/124**; c1–c8 decode-shaped axes and host PSS/RSS are open |
| Selected GPU work | `KERNEL-GEMM-BF16` W1 is `GATING`: pushed `581d335` closes the merged-owner, split rollback, F32/BF16 strided-consumer, 27B/native-35B, and strict-safety gates. BF16 output failed the near-tie gate; exact trace/component evidence is pending and qkvz remains excluded |
| Remaining kernel queue | Finalized c2 evidence ranks equal-count RMSNorm/generated partitions after the merge; FP4 tactics already match **128 Stream-K + 80 static-persistent** and are not the positive residual |
| Host-memory repair | Direct-to-final-device streaming is the complete fix; page eviction or post-prepare host release alone addresses only half of the peak/steady-state problem |

## What is implemented

The implemented subset is intentionally narrower than vLLM's full feature
surface.

| Area | Implemented scope |
|---|---|
| Engine | V1 scheduler, unified token budget, chunked prefill, FCFS preemption, persistent input batch, engine step loop, and batched generation |
| KV cache | Block-paged full-attention KV, hybrid full-attention + GDN state groups, prefix-cache manager, allocation/recycling, and device-resident gate-model state |
| Models | Qwen3.6-35B-A3B and Qwen3.6-27B text forwards with GDN, full attention, dense/MoE layers, and paged generation |
| Loading | Safetensors for both gate models; supported 35B GGUF k-quant files materialized to BF16 |
| Sampling | Greedy, temperature, top-k, top-p, min-p, repetition/frequency/presence penalties, allowed-token and bad-word masks, and internal logprob primitives |
| Serving | Basic `/v1/completions` and `/v1/chat/completions`, non-streaming and incremental SSE, concurrent scheduling, cancellation, usage frames, models/health/version endpoints |
| Structured output | Bounded JSON schema, JSON object, regex, choice, GBNF, and Hermes-style tool-call subset |
| Library | Shared/static `libvllm`, stable 17-symbol C ABI, blocking and nonblocking request lifecycles, example CLI, and OpenAI server |

Behavioral CPU tests run under CTest. CUDA correctness, sanitizer, trace, and
performance evidence is recorded per feature rather than inferred from source.

## Quick start

```sh
# CPU build. Add -DVLLM_CPP_CUDA=ON on NVIDIA and
# -DVLLM_CPP_SERVER=ON for the HTTP server.
cmake -S . -B build -DVLLM_CPP_SERVER=ON
cmake --build build -j
ctest --test-dir build

# GB10 fast-GDN build. Triton-AOT cubins are vendored; Python/Triton is only
# needed to regenerate them, not to build or run them.
cmake -S . -B build-cuda \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON \
  -DVLLM_CPP_SERVER=ON
cmake --build build-cuda -j

# Serve a supported Qwen text checkpoint.
./build-cuda/examples/server \
  --model /path/to/Qwen3.6-35B-A3B \
  --port 8000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192

# One-shot completion through the C ABI.
./build-cuda/examples/vllm-cli \
  --model /path/to/model \
  --prompt "The capital of France is"
```

Link `libvllm` or load it dynamically and use `include/vllm.h`.
`vllm_complete` / `vllm_complete_stream` provide blocking calls;
`vllm_request_submit` and the request lifecycle functions provide nonblocking
concurrent streams.

## Supported model architectures

| Architecture | Families | Safetensors | GGUF | Status |
|---|---|:---:|:---:|---|
| Qwen3.5/3.6 hybrid text | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ | 35B only | 🟡 Token-exact correctness passes on GB10; 27B performance is `GATING` at 55/124 axes; vision paths are not implemented |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 Post-parity roadmap |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 Post-parity roadmap |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 Post-parity roadmap |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference | 🟡 Correctness/CI implementation with native threadpool; real-file GGUF speed/RSS and compute-in-quant gates remain open |
| CUDA | GB10 / DGX Spark, sm_121a | 🟡 Gate-model correctness passes; 27B v0.25.0 performance remains `GATING` at 55/124; merged GDN BA is implemented/correctness-gating and qkvz remains pending |
| Other NVIDIA SMs | sm70 through sm120 families inventoried from vLLM | 🗓 Not yet fully built, traced, or gated here |
| ROCm / Intel XPU | AMD / Intel GPUs | 🗓 Post-parity roadmap |
| Metal / ANE | Apple Silicon | 🗓 Post-parity roadmap; M4 bring-up host available |
| Vulkan | Portable GPU | 🗓 Post-parity roadmap |

Only GB10/sm_121a counts as CUDA hardware support today. Source-level fallback
paths do not become support claims until their build, correctness, trace, and
performance gates pass.

### Kernel coverage on the gate path

| Kernel family | CPU | CUDA · GB10 | Status |
|---|:---:|:---:|---|
| Dense NVFP4 W4A4 GEMM | ✅ ref | ✅ | CUTLASS/FlashInfer-compatible tactics, frozen plan cache, packed QKV |
| MoE NVFP4 W4A16 GEMM | ✅ ref | ✅ | Marlin/fp4-resident gate path |
| BF16/FP8 projection GEMM | ✅ ref | ✅ | cuBLASLt TN / `nvjet_sm121` path |
| Prefill attention | ✅ ref | ✅ | Vendored FlashAttention-2 with portable fallback |
| Paged decode attention | ✅ ref | 🟡 | FA2 ratio-6 route is correctness/structure-green but strict performance-failed |
| GDN / linear attention | ✅ ref | ✅ | Vendored Triton-AOT and tensor-core WY gate path |
| RMSNorm, RoPE, SwiGLU, FP4/FP8 quant | ✅ ref | ✅ | Gate-path coverage; broader variant inventory remains open |
| CUDA-graph decode | — | 🟡 | Gate-model path runs; complete cross-model evidence remains open |

## Quantization

| Format | Status |
|---|---|
| NVFP4 W4A4 / W4A16 | 🟡 Both gate-model paths run on GB10 and pass token-exact correctness. The current 27B performance gate fails 69/124 axes; exact c2 evidence shows FP4 tactic parity, so the active speed leaf is the separate BF16 GDN projection topology |
| GGUF F32, Q4_0, Q8_0, Q3_K/Q4_K/Q5_K/Q6_K | 🟡 Supported 35B files load through BF16 materialization and pass same-file llama.cpp greedy checks; direct compute-in-quant and several formats remain open |
| FP8 | 🟡 The 35B ModelOpt static per-tensor W8A8 projection slice is implemented; generic FP8 modes and FP8 KV remain open |
| MXFP4 / MXFP8 | 🗓 Planned, including MLX-native modes |

Legend: ✅ supported and tested · 🟡 partial / gating · 🗓 planned.

## Serving and API caveats

- The Qwen3.6 checkpoints' shipped tool template uses Qwen3-Coder XML and
  forced reasoning blocks. The current Hermes-style parser does not yet fully
  implement that format.
- `/health` currently reports process liveness rather than a full engine-health
  probe.
- Prefix caching is configurable and mirrors the supported default policy, but
  the binding SGLang/vLLM shared-prefix competitor gate is still pending.
- External KV-cache connectors, including LMCache interoperability, are
  roadmap-only and are not implemented or benchmarked yet.
- Speculative decoding is not user-visible yet. MTP foundations exist, while
  MTP integration, DFlash, DSpark, TLI, n-gram, and EAGLE3 remain roadmap work.
- Multimodal/vision, LoRA, multi-GPU, local attention model consumers, and
  scaled long-context RoPE consumers are not supported yet.

The next execution order is fixed: immutable trace/component-gate merged BA and
resolve its BF16 rounding gap → implement/gate merged qkvz → all-axis 27B
parity → 35B parity → the SGLang shared-prefix gate → the rest of
[roadmap v1](.agents/roadmap_v1.md), including DSpark and external KV cache /
LMCache support.

## Project record

The canonical project record lives under [`.agents/`](.agents/), indexed by
[AGENTS.md](AGENTS.md):

- [roadmap v1](.agents/roadmap_v1.md)
- [benchmark scoreboard](docs/BENCHMARKS.md)
- [gates](.agents/gates.md) and [benchmark protocol](.agents/benchmark-protocol.md)
- [engine](.agents/engine-matrix.md), [model](.agents/model-matrix.md),
  [quantization](.agents/quantization-matrix.md),
  [kernel](.agents/kernel-matrix.md), and
  [backend](.agents/backend-matrix.md) matrices
- [append-only parity ledger](.agents/parity-ledger.md) and
  [state log](.agents/state.md) for detailed chronology and evidence
- [upstream sync protocol](.agents/upstream-sync.md) and
  [v0.25.0 audit](.agents/sync/2026-07-12-702f481.md)

The README, benchmark scoreboard, roadmap, matrices, and live specs are compact
current-state surfaces, not chronological logs. Git history plus the
append-only parity ledger and state log retain detailed attempt evidence.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE)
for third-party attributions.
