# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. Production parity is still open: the binding 27B
> comparison against vLLM v0.25.0 passes **55/124** required axes, and c2 TPOT
> is **114.841 vs 108.274 ms** (**6.1% slower**). The order-0
> [packed-decode port](.agents/specs/gdn-packed-decode.md) (the GDN decode
> kernel) is now **CLOSED on equivalence**: correctness-green at clean `f344dec`
> (default and rollback each **235/235 + 16/16**), structurally proven at
> `7ff713e`/`24cea4f` (packed **915** nodes vs rollback's **963**, 48 packed
> calls replacing 48 decomposed + 48 post-conv), and the earlier c16 HTTP-500
> slot defect fixed and proven at `c172336` (the runner's compact GDN
> state-slot pool now keys on the request identity, not the mamba block-id that
> collapsed two long concurrent sequences onto one recurrent-state slot — this
> also removes latent silent cross-request GDN state corruption in pre-validator
> binaries; see [Benchmarks](docs/BENCHMARKS.md)). The W1D3 performance/component
> gate is resolved over **eight sealed components** plus an **8-pair locked c16
> A/B** and a **24-window trace**: **packed is GPU-cheaper** (c16 paired mean
> **−0.205%, sd 0.30, <1σ**; cuBLASLt algo selection process-deterministic; the
> trace attributes no packed-side cost — kernel compute −1.30..−1.58%/step). The
> eighth seal (`e47b4d6`, `complete-failed`) reached **38/40 axes + 8/8 memory**,
> stable, `validation_error=None`, paired-consistency PASS at both concurrencies,
> with c16 at equivalence (packed med 801.97 vs rollback 802.95, −0.12%); the two
> failing axes (c2 `median_tpot_ms` 0.9899, c2 pooled `p99_ttft_ms` 0.8464) are
> sign-flipping band-edge statistics of a true-zero effect. **Disposition:
> EQUIVALENCE PROVEN — no stable regression on any axis.** The packed path stays
> the default (`VT_GDN_PACKED_DECODE=0` rollback); there is **no `complete-pass`
> marker and no speed credit**. **qkvz** (merged qkv+z projection packing) is now
> **implemented** (one BF16 in_proj_qkvz GEMM per GDN layer, `VT_GDN_MERGED_QKVZ=0`
> rollback from the same owner; CPU gates green) and **`GATING` on its DGX gates**,
> followed by the authorized **fresh binding/exact-grid rerun**. On the memory axis, the
> failing binding **peak** (48.3 GB) was localized to LOAD-time double-residency;
> the **windowed-load** release (`madvise(MADV_DONTNEED)` on each copied-then-dead
> source range, default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback) is **measured
> on GB10**: 27B load-to-ready VmHWM **48.29 GB off vs 24.75 GB on (−23.54 GB)**,
> load transient fully eliminated, ON-arm serving smoke 6/6 — the binding memory
> axes flip only at the authorized exact-grid rerun (projected PASS). No 35B
> performance result is claimed. See [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | ❌ FAILED / `GATING` | Immutable `3f256ab`: **55/124 pass**. The order-0 packed GDN decode leaf is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`, `e47b4d6`): c2 decode and ALL memory axes at packed-rollback equivalence (component peak PSS **24.86 GB**; c2 TPOT ~108.5 ms). W1D3/G3 closed over **eight sealed components** + the `00bf484` **8-pair locked c16 A/B** (paired mean **−0.205%, sd 0.30, <1σ**; cuBLASLt algo selection process-deterministic → algo-lottery REFUTED) + the 24-window trace (**packed is GPU-cheaper**; no attributable packed-side cost). The eighth seal (`e47b4d6`, `complete-failed`) reached **38/40 + 8/8 memory**, stable, paired-consistency PASS at both concurrencies; the 2 fails are band-edge statistics of a true-zero effect. **Disposition: EQUIVALENCE PROVEN — no stable regression**; no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2 merged qkv+z packing) is **implemented** (one BF16 GEMM/layer, strided mixed/z views, `VT_GDN_MERGED_QKVZ=0` rollback; CPU gates green: CTest 107/107, tools 162/162) | Next: qkvz **DGX gates** (27B default+rollback model gates, GDN suite, memcheck, 145→97 BF16-GEMM/window trace), then the authorized **fresh binding/exact-grid rerun** (fresh vLLM denominators; `--mamba-ssm-cache-dtype float32`; cite `702f481`) → all-axis 27B parity → 35B
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | ⏸ BLOCKED | No current v0.25.0 performance result | Run only after all 27B axes pass |
| Host-memory parity | ❌ FAILED on the binding grid / fix MEASURED | The failing **peak** was load-time double-residency (22.92 GiB mirror built while the full source mmap stayed resident). The `LOAD-SAFETENSORS` windowed release (default on; `VT_LOAD_WINDOWED_RELEASE=0` rollback) is now **measured on GB10**: 27B load-to-ready VmHWM **48.29 GB off vs 24.75 GB on (−23.54 GB, −48.7%)** — the load transient is fully eliminated and the ON peak equals steady RSS, below vLLM's 28.5 GB binding peak; ON-arm serving smoke 6/6 | Binding Peak PSS/RSS axes flip only at the next authorized exact-grid rerun (projected PASS). Direct-to-device streaming remains the deeper fix (still wanted for 35B) |

The binding cache-off workload is input 1,024 → output 128, greedy, closed
loop, with three interleaved repetitions. Arm equivalence is audited: batch
cap, token budget, sampling, corpus, cache dtypes and kernel families all
match, and the client commands are identical to one token — see the
[equivalence audit](.agents/specs/benchmark-equivalence-audit-2026-07-15.md). Ratios are direction-normalized so
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
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is **`DONE`** — W1D3 **CLOSED on equivalence** (owner `e47b4d6`). The c16 HTTP-500 slot defect (the runner keyed the compact GDN state-slot pool on the mamba block-id, collapsing 2 long c16 sequences onto 1 recurrent-state slot; also latent silent cross-request corruption) was fixed test-first (request-identity keying) and proven at `c172336`. G3 closed over eight sealed components + the 8-pair A/B (−0.205% ± 0.30, <1σ) + the trace attribution (packed GPU-cheaper): no stable regression, no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2) is implemented CPU-side; selected next GPU work is its DGX gate battery (model gates both arms, memcheck, 145→97 BF16 trace) + the authorized exact-grid rerun |
| Remaining gap diagnosis | The 2026-07-14 [parity rescan](.agents/specs/parity-rescan-2026-07-14.md) grounds the failing mass as **host-side**: TTFT passes 24/24, our GPU kernels are net faster on the measured window, and the open axes are c2–c8 decode latency plus host memory. The prior RMSNorm/generated-partitions residual is **disproven** (vLLM's norm-quant fusion is FP8-only; cross-profiler artifact). Parallel host workstreams: TCP_NODELAY (DONE, measured neutral on loopback — ruled out as the decode-gap cause), memory precheck → weight streaming, and nsys c2 attribution before async-sched W3 |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity and the decode-gap attribution moves to the nsys c2 full-step diff |
| Host-memory repair | **MEASURED**: the `LOAD-SAFETENSORS` windowed release (progressive `madvise(MADV_DONTNEED)` on each copied-then-dead source range; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback) cuts the 27B load-to-ready VmHWM from **48.29 GB to 24.75 GB (−48.7%)** on GB10 — the load transient is fully eliminated (peak = steady RSS) and serving stays healthy (smoke 6/6). Projected to flip both binding memory axes at the next authorized exact-grid rerun (vLLM peaks 28.17/28.53 GB); no credit until then. Direct-to-final-device streaming stays the complete fix (also removes the steady mirror, wanted for 35B) |

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
| CUDA | GB10 / DGX Spark, sm_121a | 🟡 Gate-model correctness passes; 27B v0.25.0 performance remains `GATING` at 55/124. Packed GDN decode is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`): the c16 slot defect was fixed test-first (request-identity keying) and proven at `c172336`, and W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ) + a trace showing packed is GPU-cheaper. qkvz is implemented (`GATING` on its DGX gates); those gates + the authorized exact-grid rerun are next |
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
| GDN / linear attention | ✅ ref | 🟡 | Prefill AOT is gated; the packed pure-decode kernel is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`). The c16 slot defect (compact state-slot pool keyed on the mamba block-id collapsed two long c16 sequences onto one recurrent-state slot) was fixed test-first (request-identity keying) and proven at `c172336`; W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ). Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); no speed credit |
| RMSNorm, RoPE, SwiGLU, FP4/FP8 quant | ✅ ref | ✅ | Gate-path coverage; broader variant inventory remains open |
| CUDA-graph decode | — | 🟡 | Gate-model path runs; complete cross-model evidence remains open |

## Quantization

| Format | Status |
|---|---|
| NVFP4 W4A4 / W4A16 | 🟡 Both gate-model paths run on GB10 and pass token-exact correctness. The current 27B performance gate fails 69/124 axes; FP4 tactics match, the non-quantized packed GDN decode leaf is now closed on equivalence, and the active speed leaf — merged qkvz (`KERNEL-GEMM-BF16` W2) — is implemented (one BF16 in_proj_qkvz GEMM per GDN layer, DGX gates pending) |
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

The next execution order is fixed: with the packed-GDN decode leaf now CLOSED on
equivalence and merged **qkvz** implemented, run the qkvz DGX gates → the
authorized fresh binding/exact-grid rerun → all-axis 27B parity → 35B parity → the SGLang
shared-prefix gate → the rest of
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
- [parity ledger](.agents/parity-ledger.md) and
  [state log](.agents/state.md) for the current record era's detailed
  chronology and evidence
- [upstream sync protocol](.agents/upstream-sync.md) and
  [v0.25.0 audit](.agents/sync/2026-07-12-702f481.md)

The README, benchmark scoreboard, roadmap, matrices, and live specs are compact
current-state surfaces, not chronological logs. Detailed state/ledger records
are append-only within an open era, then frozen under `.agents/completed/` and
replaced by a concise carry-forward when that era closes.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE)
for third-party attributions.
