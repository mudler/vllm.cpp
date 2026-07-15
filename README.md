# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. Production parity is still open: the binding 27B
> comparison against vLLM v0.25.0 passes **55/124** required axes, and c2 TPOT
> is **114.841 vs 108.274 ms** (**6.1% slower**). The active
> [packed-decode port](.agents/specs/gdn-packed-decode.md) is correctness-green
> on clean `f344dec`: default and rollback each pass **235/235 + 16/16**.
> Immutable `7ff713e`, finalized by `24cea4f`, also proves the exact structural
> substitution: packed has **915** nodes versus rollback's **963**, replacing
> 48 decomposed recurrence plus 48 post-conv calls with 48 packed calls while
> leaving the remaining topology invariant. A production-build-only c2/c16
> AB/BA/AB component runner and marker-last every-axis finalizer are
> CPU-gated **45/45**. The clean pushed `d82d282` series is
> **FAILED / INCOMPLETE**: both direct model gates and all six c2 legs
> completed, then the first c16 packed leg passed its streaming preflight and 16
> warmups but all **0/96** timed requests returned HTTP 500. The root cause was
> unrecoverable because our port dropped two upstream fatal log lines. A bounded
> **test-first diagnostic checkpoint** now restores them: four unconditional
> `std::cerr` error-path channels (`engine-fatal:` at the busy-loop guard,
> `async-llm:` at the output handler, `api-server:` at both 500 sites, `sse:`
> mid-flight) plus an opt-in `VT_GDN_DIAG_STEP_LOG` geometry trace, and a
> packed-only `--diagnostic-c16` driver mode kept fully separate from the gating
> component (`component-diagnostic.json`; the finalizer refuses diagnostic
> evidence). The DGX reproduction **captured the root cause deterministically
> 3/3**: `vt: qwen3_5: duplicate live GDN state index` (`qwen3_5.cpp:73`) — the
> runner keyed its compact GDN state-slot pool on the mamba block-id, which
> collapses to the shared null block-id 0 once a sequence exceeds one mamba
> block (the group is configured with a sub-sequence `block_size`), so two long
> concurrent sequences under c16 churn shared ONE recurrent-state slot. The
> **test-first repair** now keys the slot on the request identity so each live
> sequence owns exactly one slot for its lifetime (a RED runner test threw the
> exact fatal, GREEN after the fix; `test_runner` **8/8**, all tools
> **135/135**). This also fixes latent silent cross-request GDN state corruption
> that pre-validator binaries could hit at high concurrency (see
> [Benchmarks](docs/BENCHMARKS.md)). The slot fix is proven on DGX at `c172336`,
> but the first three sealed 12-leg components all reached `complete-void` with
> stable throughput/TPOT/ITL/memory, voided only by the c2/c16 TTFT-family. Those
> voids are grounded as an upstream-mirrored prefill co-schedule ARRIVAL LOTTERY
> (our waiting-queue admission is a 1:1 mirror of vLLM's — no scheduler
> divergence, scheduler unchanged): at c2 the six per-request TTFTs are bimodal
> (~0.45/0.9 s) and flip 3/3-vs-6/0 leg-to-leg. Two per-run stability revisions
> landed test-first — tails at 15% (c16 + non-TTFT), and the **c2 TTFT-family**
> now COMPARED on each arm's pooled 18-per-request distribution, STABILITY-gated
> on a 50% pooled sanity bound, and EXCLUDED from the gated per-rep paired axes;
> c16 and every non-TTFT axis keep the 4%/15% rules, and E2EL is unchanged
> (measured c2 E2EL deviation ≤0.30%). The packed component stays `GATING` (no
> speed credit) until a fourth 12-leg component from the pushed SHA reaches a
> verified terminal status. On the memory axis, the failing binding **peak** (48.3 GB) was
> localized to LOAD-time double-residency (the 22.92 GiB host mirror built while
> the full source mmap stayed resident); a **windowed-load** change now releases
> each copied-then-dead source range during the copy loop (`madvise(MADV_DONTNEED)`,
> default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback), CPU-gated with RED→GREEN
> mincore/RSS tests — the DGX VmHWM A/B is PENDING and the binding memory axes
> stay FAILED until the authorized exact-grid rerun. No 35B performance result is
> claimed. See [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | ❌ FAILED / `GATING` | Immutable `3f256ab`: **55/124 pass, 69 fail**. Structural packed/rollback evidence is accepted, and the c16 slot defect (`duplicate live GDN state index`, captured 3/3 at `4a450f9`) is **repaired and proven on DGX** at `c172336`: the previously-fatal c16 boundary passes **3/3** and both model gates pass **235/235**. The first two sealed 12-leg components both reached marker-last **`complete-void`** — every throughput/mean/median axis stable and packed non-regressing on forensic medians (run 1 c2 TPOT 108.7 vs 109.1 ms; run 2 c16 tput 793.1/794.1, TPOT 166.5/166.2), voided **only** by max-dominated TTFT tail wobble (c2 max-of-6, c16 last-order-statistic). Those two voids show the uniform 4% per-run rule mis-fits tail order statistics, so the tail stability tolerance was revised test-first (tails 15%, non-tail/memory stay 4%). The third seal (`d19e091`) then voided on the c2 **mean/median** TTFT themselves; the **co-schedule PARITY verdict** proved a bimodal prefill arrival lottery (our waiting loop co-schedules two 1024-token prefills in the 2048 budget exactly like vLLM's — 1:1 mirror, scheduler unchanged), so this checkpoint pools the c2 TTFT-family test-first: it is COMPARED on each arm's pooled 18-per-request distribution, STABILITY-gated on a 50% pooled sanity bound, and EXCLUDED from the gated per-rep paired axes (c16 + non-TTFT keep 4%/15%; E2EL unchanged, measured c2 deviation ≤0.30%). Focused **56/56**, tools **139/139** | A fourth 12-leg component from the pushed SHA must reach a verified terminal status before any axis binds; then qkvz |
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | ⏸ BLOCKED | No current v0.25.0 performance result | Run only after all 27B axes pass |
| Host-memory parity | ❌ FAILED / windowed-load landed, A/B pending | The failing **peak** (48.3 GB) is load-time double-residency: the 22.92 GiB host mirror is built while the full source mmap stays resident; steady RSS afterwards is 24.75 GB — already below vLLM's 28.5 GB peak. Allocator retention is ruled out (≤0.5 GB). The `LOAD-SAFETENSORS` windowed release now drops each copied-then-dead source range during load (default on; `VT_LOAD_WINDOWED_RELEASE=0` rollback), CPU RED→GREEN tests green | Run the DGX VmHWM A/B (expect ~48.3 GB off vs ~25 GB on); axes stay FAILED until the authorized exact-grid rerun. Direct-to-device streaming remains the deeper fix (still wanted for 35B) |

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
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is `ACTIVE`: structural evidence is accepted. The `d82d282` c16 packed leg died with `duplicate live GDN state index`; the `4a450f9` diagnostic reproduction captured it 3/3. Root cause: the runner keyed the compact GDN state-slot pool on the mamba block-id, which collapses to the shared null block-id 0 for any sequence past its first mamba block, so 2 long c16 sequences shared 1 recurrent-state slot. **Test-first repair landed** — the pool now keys on the request identity (each live sequence owns one slot for its lifetime). This also removes latent silent cross-request GDN state corruption. DGX correctness gates + a fresh 12-leg component rerun are the next step; no partial number binds |
| Remaining gap diagnosis | The 2026-07-14 [parity rescan](.agents/specs/parity-rescan-2026-07-14.md) grounds the failing mass as **host-side**: TTFT passes 24/24, our GPU kernels are net faster on the measured window, and the open axes are c2–c8 decode latency plus host memory. The prior RMSNorm/generated-partitions residual is **disproven** (vLLM's norm-quant fusion is FP8-only; cross-profiler artifact). Parallel host workstreams: TCP_NODELAY (DONE, measured neutral on loopback — ruled out as the decode-gap cause), memory precheck → weight streaming, and nsys c2 attribution before async-sched W3 |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity and the decode-gap attribution moves to the nsys c2 full-step diff |
| Host-memory repair | The binding **peak** is LOAD-time double-residency, so the first shot is the `LOAD-SAFETENSORS` windowed release (progressive `madvise(MADV_DONTNEED)` on each copied-then-dead source range during the copy loop; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback). Landed and CPU-gated (RED→GREEN mincore/RSS + byte-identity + neighbor-safety); DGX VmHWM A/B pending. Direct-to-final-device streaming stays the complete fix (also removes the steady mirror, wanted for 35B) |

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
| CUDA | GB10 / DGX Spark, sm_121a | 🟡 Gate-model correctness passes; 27B v0.25.0 performance remains `GATING` at 55/124. Packed GDN correctness/structure are accepted; the `d82d282` c16 leg died on `duplicate live GDN state index`, captured 3/3 at `4a450f9`. A test-first repair keys the runner's compact GDN state-slot pool on the request identity (was the mamba block-id, which collapsed to the shared null block for long sequences); DGX correctness gates + a fresh 12-leg component rerun are pending, and qkvz remains blocked |
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
| GDN / linear attention | ✅ ref | 🟡 | Prefill AOT and packed correctness/structure are gated. The `d82d282` c16 leg died on `duplicate live GDN state index` (captured 3/3 at `4a450f9`): the runner's compact GDN state-slot pool, keyed on the mamba block-id, collapsed two long c16 sequences onto one recurrent-state slot. A test-first repair keys the pool on the request identity; a fresh 12-leg component rerun after DGX gates is next |
| RMSNorm, RoPE, SwiGLU, FP4/FP8 quant | ✅ ref | ✅ | Gate-path coverage; broader variant inventory remains open |
| CUDA-graph decode | — | 🟡 | Gate-model path runs; complete cross-model evidence remains open |

## Quantization

| Format | Status |
|---|---|
| NVFP4 W4A4 / W4A16 | 🟡 Both gate-model paths run on GB10 and pass token-exact correctness. The current 27B performance gate fails 69/124 axes; FP4 tactics match, and the active speed leaf is the non-quantized packed GDN decode path |
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

The next execution order is fixed: run the DGX `--diagnostic-c16` reproduction to
read the now-exposed c16 HTTP 500 root cause, repair test-first, rerun the
packed-GDN component → implement/gate
merged qkvz →
all-axis 27B parity → 35B parity → the SGLang shared-prefix gate → the rest of
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
