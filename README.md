# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. Production parity is still open. The **new binding 27B
> result** is the fresh, fully-interleaved exact-grid rerun at `246a23c`:
> **49/124** required axes pass against vLLM v0.25.0. It **supersedes** the
> `3f256ab` grid (**55/124**, retained immutable). The nominal 49 < 55 is a
> **structural recomposition**, not a plain regression: **memory (4/4)**, **c1
> (20/20)** and **every TTFT axis** now sweep clean for the first time, and the
> entire remaining failure mass is the decode-coupled family at c2–c32
> (TPOT/ITL/E2EL **2.2–6.5% slower**) plus two ITL tail anomalies (c8 p99, c32
> p90). **Honest regression:** ours lost **−2.67% total throughput at c16** and
> **−3.64% at c32** vs `3f256ab` while vLLM held, so the old c16/c32 wins are
> gone. **Hypothesis (labeled, unproven):** `3f256ab` silently carried the GDN
> slot-sharing defect (two long requests could share one recurrent-state slot at
> high concurrency), which the `c172336` correctness fix removed and may have
> traded that inflated throughput away, alongside every other change between the
> SHAs; an **era A/B** (`3f256ab` vs `246a23c` binary, interleaved c16) is running
> now as the diagnostic. A source+arithmetic bisect (`.agents/state.md` 2026-07-16)
> now **grounds this hypothesis and refutes a "host-machinery" reading**: the
> per-step validators/remap are µs-scale (n≤32), while de-collapsing shared GDN
> slots restores ~3.4 GB/step of correctness-required recurrent-state DRAM traffic
> (3 MB/slot/layer) ≈ the 8 ms — so `9ad8fb7`'s 825 was a silent-corruption
> artifact and ~790 is the correct floor; the honest 790→794 residual is decode
> **kernel** efficiency, not a recoverable host cost. The `246a23c` binary carries the correctness slot-fix
> (`c172336`), the **windowed-load** release (`cb2d310`, which flips both memory
> axes to PASS — ours peak PSS 24.88 GB vs vLLM 28.18 GB), merged **qkvz**
> (`45f9e6d`, DGX gates green), and packed GDN decode as the default. The order-0
> [packed-decode port](.agents/specs/gdn-packed-decode.md) is **CLOSED on
> equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`, `e47b4d6`): correctness
> immutable at `f344dec` (default and rollback each **235/235 + 16/16**),
> structure at `7ff713e`/`24cea4f`, W1D3/G3 closed over eight sealed components +
> the 8-pair locked c16 A/B (**−0.205% ± 0.30, <1σ**) + the 24-window trace
> (**packed is GPU-cheaper**), with **no `complete-pass` marker and no packed
> speed credit**. No 35B performance result is claimed until 27B reaches 124/124.
> **Correct-state c16 kernel traces (2026-07-16)** now attribute the ~8 ms/step gap:
> ~4.65 ms GPU-busy + ~3.25 ms host/idle, and the busy part splits ~2.06 ms GDN
> recurrence tiling (ours 21.3 vs vLLM 19.2 ms/step; state r/w FUSED in-kernel on
> both sides — NOT separate state-I/O) + ~2 ms unfused norm/quant glue, with
> GEMM/MoE/attention at parity. The ~2.06 ms recurrence-tiling lever is now
> **ported (test-first, CPU-gated, DGX-pending)**: a register-resident
> single-warp `num_stages=3` FLA packed-decode kernel (`GdnPackedDecodeRegTileKernel`,
> default-on behind `VT_GDN_PACKED_REG_TILE`, =0 restores the legacy kernel
> bit-for-bit) — expected ~+2 ms/step (~+10 tok/s) at c16 once the DGX A/B
> confirms. See [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | ❌ FAILED / `GATING` | **NEW BINDING `246a23c`: 49/124 pass** (fresh interleaved exact-grid rerun; supersedes `3f256ab`'s 55/124, retained immutable). Per concurrency: c1 **20/20**, c2 4, c4 5, c8 4, c16 6, c32 6, memory **4/4**. Memory + c1 + every TTFT axis sweep clean; failure mass is the decode-coupled family at c2–c32 (mean TPOT 2.2–6.5% slower) + two ITL tail anomalies (c8 p99_itl 0.5599, c32 p90_itl 0.7925). HONEST regression: ours c16/c32 total throughput −2.67%/−3.64% vs `3f256ab` while vLLM held; old c16/c32 wins gone. The binary carries slot-fix `c172336`, windowed-load `cb2d310` (memory now PASS), qkvz `45f9e6d` (DGX gates green), packed-decode default. The order-0 packed leaf is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`, `e47b4d6`; eight seals + 8-pair A/B −0.205% ± 0.30 <1σ + 24-window trace, packed GPU-cheaper); no `complete-pass` marker, no packed speed credit | ATTRIBUTED + VALIDATED (era A/B, four rounds + probe3 at `6dd24df`): `3f256ab`'s high-concurrency throughput WAS inflated by the slot-sharing defect (collapsed slots = 1/16th the GDN state traffic ≈ 8–12 ms/step); the honest correct-state c16 floor is ~790–799 vs vLLM 794, gates 235/235×2, no recoverable host cost (≤15 µs/step; O(n) validation landed). Top lever (MEASURED 2026-07-16, correct-state c16 kernel traces): the ~8 ms/step wall gap (c16 TPOT 159.6 vs 167.5) = ~4.65 ms GPU-busy + ~3.25 ms host/idle; the busy part is ~2.06 ms GDN recurrence tiling (ours 21.31 vs vLLM 19.24 ms/step, state r/w fused in-kernel, ~83% vs ~92% of peak BW) + ~2 ms unfused norm/quant glue, GEMM/MoE/attention at parity. Port vLLM's register-resident single-warp num_stages=3 FLA packed-decode tiling (`fused_recurrent.py:282-336`) into `GdnPackedDecodeKernel`, plus the Inductor add+RMSNorm+FP4-quant / SiLU-quant decode fusion. Then the c8 p99 / c32 p90 ITL tail. 35B only after 27B 124/124
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | ⏸ BLOCKED | No current v0.25.0 performance result | Run only after all 27B axes pass |
| Host-memory parity | ✅ PASS on the new binding grid | All four memory axes now PASS at `246a23c` (windowed-load `cb2d310` binding): ours peak PSS/RSS 24,879,201/24,881,800 KiB vs vLLM 28,184,400/28,563,020 KiB (1.1329×/1.1479×), GPU 40,996 vs 70,531 MiB, MemAvailable-drop 68,346,844 vs 80,660,556 KiB. The prior `3f256ab` peak (48.3 GB) was load-time double-residency, eliminated by the windowed release (−23.54 GB load-to-ready VmHWM) | Memory parity holds; direct-to-device streaming remains the deeper fix (removes the 22.92 GiB steady mirror, wanted for 35B) |

The binding cache-off workload is input 1,024 → output 128, greedy, closed
loop, with three interleaved repetitions. Arm equivalence is audited: batch
cap, token budget, sampling, corpus, cache dtypes and kernel families all
match, and the client commands are identical to one token — see the
[equivalence audit](.agents/specs/benchmark-equivalence-audit-2026-07-15.md). Ratios are direction-normalized so
**1.0 or higher passes**.

| Concurrency | Axes passing | Total throughput: ours / vLLM | Ratio |
|---:|---:|---:|---:|
| 1 | **20/20** | 84.149 / 82.779 tok/s | **1.016543×** |
| 2 | 4/20 | 156.325 / 158.977 tok/s | **0.983320×** |
| 4 | 5/20 | 286.896 / 292.396 tok/s | **0.981189×** |
| 8 | 4/20 | 499.150 / 508.958 tok/s | **0.980730×** |
| 16 | 6/20 | 790.625 / 794.356 tok/s | **0.995303×** |
| 32 | 6/20 | 1081.098 / 1082.750 tok/s | **0.998474×** |

All four memory axes and every TTFT axis now pass, and c1 sweeps 20/20; the
remaining gaps are the decode-coupled family (throughput, TPOT/ITL/E2EL) at
c2–c32, where decode is 2.2–6.5% slower, plus two ITL tail anomalies. The old
c16/c32 total-throughput wins are gone (see the honest regression above). The
full per-axis table, memory table, and exact reproduction recipe are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Current performance track

| Work item | Present disposition |
|---|---|
| Binding gate | **NEW BINDING `246a23c`: 49/124** (supersedes `3f256ab`'s 55/124). Memory (4/4), c1 (20/20) and every TTFT axis now pass; the open axes are the decode-coupled family at c2–c32 (2.2–6.5% slower) plus two ITL tail anomalies |
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is **`DONE`** — W1D3 **CLOSED on equivalence** (owner `e47b4d6`). The c16 HTTP-500 slot defect (the runner keyed the compact GDN state-slot pool on the mamba block-id, collapsing 2 long c16 sequences onto 1 recurrent-state slot; also latent silent cross-request corruption) was fixed test-first (request-identity keying) and proven at `c172336`. G3 closed over eight sealed components + the 8-pair A/B (−0.205% ± 0.30, <1σ) + the trace attribution (packed GPU-cheaper): no stable regression, no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2) DGX gates closed green at `45f9e6d` (−48 BF16 GEMMs/window confirmed) and is in the `246a23c` binding binary. The authorized exact-grid rerun has now RUN (new binding, 49/124); the era A/B fully attributed the c16/c32 delta (corruption-subsidized bandwidth; pre-fix GDN kernel evidence now contamination-suspect); the fresh correct-state GDN kernel trace vs vLLM is now DONE (2026-07-16): the ~8 ms/step gap is ~4.65 ms busy + ~3.25 ms host/idle, busy = ~2.06 ms GDN recurrence tiling + ~2 ms unfused norm/quant glue (state-I/O fused in-kernel, not a separate op; GEMM/MoE/attn at parity); selected next GPU work is porting vLLM's register-resident single-warp num_stages=3 FLA packed-decode tiling into `GdnPackedDecodeKernel` |
| Remaining gap diagnosis | With memory now passing, the failing mass is the **c2–c32 decode-coupled family** (throughput inversely coupled to TPOT/ITL). The 2026-07-14 [parity rescan](.agents/specs/parity-rescan-2026-07-14.md) already grounded decode as host-side (TTFT passes, our GPU kernels net faster on the measured window). The prior RMSNorm/generated-partitions residual is **disproven** (vLLM's norm-quant fusion is FP8-only). Open questions this binding raises: the labeled c16/c32 throughput regression vs `3f256ab` (era A/B in-flight) and the c8 p99 / c32 p90 ITL tail mechanism; nsys c2/c8 full-step attribution precedes async-sched W3 |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity and the decode-gap attribution moves to the nsys c2 full-step diff |
| Host-memory repair | **BINDING PASS**: the `LOAD-SAFETENSORS` windowed release (progressive `madvise(MADV_DONTNEED)` on each copied-then-dead source range; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback) is now in the `246a23c` binding binary — all four memory axes PASS (ours peak PSS 24.88 GB vs vLLM 28.18 GB). Direct-to-final-device streaming stays the complete fix (also removes the steady mirror, wanted for 35B) |

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
| Qwen3.5/3.6 hybrid text | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ | 35B only | 🟡 Token-exact correctness passes on GB10; 27B performance is `GATING` at 49/124 axes (new binding `246a23c`); vision paths are not implemented |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 Post-parity roadmap |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 Post-parity roadmap |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 Post-parity roadmap |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference | 🟡 Correctness/CI implementation with native threadpool; real-file GGUF speed/RSS and compute-in-quant gates remain open |
| CUDA | GB10 / DGX Spark, sm_121a | 🟡 Gate-model correctness passes; 27B v0.25.0 performance is `GATING` at **49/124** (new binding `246a23c`; memory + c1 + TTFT clean, decode c2–c32 open). Packed GDN decode is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`): the c16 slot defect was fixed test-first (request-identity keying) and proven at `c172336`, and W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ) + a trace showing packed is GPU-cheaper. qkvz DGX gates closed green at `45f9e6d` and ride this binding |
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
| NVFP4 W4A4 / W4A16 | 🟡 Both gate-model paths run on GB10 and pass token-exact correctness. The new-binding 27B performance gate (`246a23c`) fails 75/124 axes (memory + c1 + TTFT now pass; decode c2–c32 open); FP4 tactics match, the non-quantized packed GDN decode leaf is closed on equivalence, and merged qkvz (`KERNEL-GEMM-BF16` W2, one BF16 in_proj_qkvz GEMM per GDN layer) is DGX-green and in the binding binary |
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

The next execution order is fixed: with the packed-GDN decode leaf CLOSED on
equivalence, merged **qkvz** DGX-green, and the authorized exact-grid rerun now
RUN (new binding `246a23c`, 49/124), the open front is the c2–c32 decode gap →
the in-flight era A/B verdict + nsys c2/c8 full-step attribution → `ENG-ASYNC-SCHED`
W3 if confirmed → all-axis 27B parity → 35B parity → the SGLang shared-prefix gate →
the rest of [roadmap v1](.agents/roadmap_v1.md), including DSpark and external KV
cache / LMCache support.

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
