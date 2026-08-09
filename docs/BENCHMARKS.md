# Benchmarks

## At a glance: W5 manifest guarded; release artifacts pending

| Reference | Workload | Headline | Tokens |
|---|---|---|---|
| **Binary release matrix (ACTIVE; W5 implemented)** | `ENG-RELEASE-BINARIES`: versioned deterministic manifest for primary host-ABI fat-CUDA + adaptive-CPU static-core bundles, optional per-SM diagnostics and experimental literal-static musl CPU | **W5 GREEN:** 19/19 + ten review mutations killed; contract 30/30; PR-size classes/budget GREEN. Synthetic fixtures only. **PENDING:** W1-W4/W6-W13 and all archive/staged-smoke/runtime/correctness/performance gates | n/a |
| **Container images (inventoried)** | `ENG-RELEASE-CONTAINERS`: GHCR images from GitHub Actions; lanes `-cuda`/`-vulkan`/`-cpu`, amd64+arm64 manifests | **No number owed:** record-only. No Dockerfile, workflow, registry package or image exists; the image is the unimplemented `ENG-RELEASE-BINARIES` bundle | n/a |
| **Developer agent entry point (implemented)** | `DOCS-AGENT-PROTOCOL-ENTRYPOINT`: public contribution guide + synchronized, mutation-gated pre-claim intake rule | Rebased documentation/protocol only; benchmark void | n/a |
| **ARCH audit: ABI is text-only** | 4 capabilities (H3 video, Laguna, Kimi-Linear, DeepSeek-V4) reachable only from `examples/`, none registry-backed. No gate asks whether a CONSUMER can reach a capability. Documentation only |
| **DSR fix: async readback capability (2026-08-08)** | **No number owed**: behavior-neutral (CPU/CUDA async-ON, discrete non-CUDA async-OFF, unchanged); moves a `kCUDA` check onto `Backend`, unblocking red CI on #127/#154/#155 |
| **`ROAD-V1-MEM` M1+M2 (2026-08-08)** | KV auto-sizing CPU brick: `--kv-cache-memory` sizes the pool from a byte budget via the group-aware `KVBytesPerBlock` divisor (ABI v16, CPU-gated). M3 profile run dgx-gated |
| **Record/checker repair 2026-08-07–08** | Restored red record/env gates; made release AST semantic pins Python 3.12/3.13-stable; recorded merged Gemma-4 MoE as known merged-GEMM drift and closed the stale embeddings claim. No runtime/performance change |
| **vLLM** | Qwen3.6-27B NVFP4, GB10 | ahead 4.5% at c1, **tie** at c2 to c32 | identical |
| **vLLM** | Qwen3.6-35B-A3B NVFP4, GB10 | 0.93x to 1.03x: ahead at c4, worst c16 0.93x | identical |
| **vLLM** | DeepSeek-V2-Lite (MLA), GB10 | 0.86x to 0.95x throughput, TTFT wins at c4/c8 | identical |
| **vLLM** | Laguna-S-2.1 NVFP4 (118B/8B MoE), GB10 | **parity+, 1.03x** (44.46 vs 43.10 tok/s, byte-exact, default config; bf16 weights now device-resident) | near-tie |
| **llama.cpp** | Qwen3.5-2B GGUF, CPU aarch64 | 20-core Arm/i8mm: prefill **1.18x ahead**, decode tie, memory parity. RPi5/A76: vllm.cpp is **0.461x prefill / 0.653x decode+E2E**, but uses **24.2% less RSS** | byte-identical on both Arm lanes |
| **MLX-LM** | Qwen3-0.6B, Apple M4 | 97.6% warm total, prefill ahead | near-tie |
| **DwarfStar** | DeepSeek-V4-Flash GGUF, GB10 | **beats ds4, 1.144x** (18.69 vs 16.33 tok/s, byte-exact, default config) | n/a, GGUF peer |

Reading the ratios: throughput is ours/reference, latency is reference/ours, so
**1.0 or higher is a win** everywhere on this page. Which architecture each number
is measured on, and the per-architecture correctness gate behind it, is the
registry-bound list in [FEATURES.md](FEATURES.md).

## vLLM, online serving

The binding comparison. vLLM runs its **production graphed config**, never
`--enforce-eager`, because the graphed config is the honest denominator.

| Model | Quant | vLLM pin | Axes passing | Disposition |
|---|---|---|---:|---|
| Qwen3.6-27B | NVFP4 | 0.25.0 | **115/124** | Effective parity-or-better, two-grid totality. Measured on `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` (BF16 head); @`ccdaab7e` re-quantized the head to FP8 |
| Qwen3.6-35B-A3B | NVFP4 `modelopt_mixed` | 0.25.0 | 2/18 | 3-rep grid 2026-08-05 @`1ea26427`: 0.93-1.03x (c4 wins), c16 0.93x. Both c16 levers A/B'd NEG: drain event -1.9%, mirror 0.999x. ★ probe found a prod async batch-1 greedy DEGENERATION bug the mirror fixes |
| DeepSeek-V2-Lite | bf16 MLA | 0.25.0 | 4/25 | Attributed miss, row stays `ACTIVE` |
| Qwen3.5-4B | bf16 direct-load | 0.26.0.dev0 | throughput + host PSS | Exact chunks ON: total **1.021x PASS**; TTFT **1.086x**, TPOT **1.025x**, VRAM **1.018x OPEN**; local A/B **+2.152%** ([evidence](bench-evidence/qwen35-4b-sm120-main-20260807.md)) |

### GDN prefill causal-convolution by GPU

| GPU | Workload and basis | vllm.cpp | vLLM | Ratio | Status |
|---|---|---:|---:|---:|---|
| RTX 5070 Ti (`sm_120`) | Qwen3.5-4B BF16, c32, steady-interval total | 233.955 ms | 145.421 ms | **1.609x slower** | Rebased-main exact chunks ON; rollback 718.704 ms, so local is **3.072x faster** ([evidence](bench-evidence/qwen35-4b-sm120-main-20260807.md)) |
| GB10 (`sm_121a`) | Qwen3.6-27B NVFP4, historical normalized prefill | 0.43 us/token/layer | 0.18 us/token/layer | **2.39x slower** | Directional only: unequal token clusters, older pin ([ledger](../.agents/parity-ledger.md)) |
| GB10 (`sm_121a`) | Qwen3.6-35B NVFP4, later local kernel A/B | 321.148 us c1; 960.313 us c6 | - | `PENDING` | Register vs tiled improved 4.7%/7.3%; no paired vLLM denominator ([record](../.agents/specs/gdn-prefill-conv-reg-2026-07-18.md)) |
| Jetson Thor (`sm_110`), AGX Orin (`sm_87`) | No matched GDN workload | - | - | `PENDING` | Runtime correctness only; no causal-conv speed trace |

### Qwen3.6-27B by concurrency

Medians of three interleaved repetitions, 1,024 in / 128 out, cache off, closed
loop. Output is token-for-token identical to vLLM at every point.

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s | **86.05** | 159.68 | 292.34 | 508.77 | 801.76 | 1095.01 |
| vLLM tok/s | 82.32 | 158.03 | 290.31 | 505.46 | 789.16 | 1076.25 |
| **Ratio** | **1.045x** | 1.011x | 1.007x | 1.007x | 1.016x | 1.017x |
| Axes passing | 20/20 | 20/20 | 18/20 | 15/20 | 19/20 | 18/20 |

We are nominally ahead at all six, but only c1 means anything. Our run-to-run
noise band is 0.5% and c2 through c32 land between 0.7% and 1.7%, so **treat
those five as ties**, not as wins. The nine axes that fail in both grids are one
tradeoff, not nine problems: our synchronous deterministic forward loses on
low-concurrency *median* decode and TTFT, and wins the corresponding *tail* and
the same metric at higher concurrency (c8 p99 ITL 0.86x, but 1.055x at c16 and
1.078x at c32).

### Qwen3.6-35B-A3B by concurrency

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s | 593.7 | 882.9 | **1417.6** | 1845.2 | 2327.4 | 2937.8 |
| vLLM tok/s | 607.4 | 911.9 | 1374.4 | 1922.1 | 2497.2 | 3012.9 |
| **Ratio** | 0.977x | 0.964x | **1.025x** | 0.964x | 0.932x | 0.971x |
| Mean TPOT | 0.975x | 0.966x | **1.036x** | 0.976x | 0.928x | 0.988x |
| Mean TTFT | 0.980x | 0.947x | 0.976x | 0.939x | 0.933x | 0.929x |

Fresh 3-rep grid 2026-08-05 at `1ea26427` (post async-UAF fix), medians, SACRED
gate passed pre-bench. The prior low-batch gap closed hard (c1 0.817x→0.977x);
the open mass is now TTFT 0.93-0.98x everywhere and a NEW c16 dip to 0.932x
(prior binding won c16). **c16 drain-sync lever, MEASURED NEGATIVE 2026-08-05**
(3+3 reps, single load/arm, ours only): the UAF fix's full-stream drain vs a
blocking-event drain (mirrors vLLM's `prepare_inputs_event`, blocking=True) gave
2308.0 vs 2263.2 total tok/s medians, non-overlapping bands, the event arm
**-1.9%** (TPOT 47.6 to 49.0 ms). So the c16 cost is NOT driver-lock spin but the
depth-2 **serialization** the drain guards: `update_states`' host `condense`
read-after-writes the previous step's device scatter of `last_sampled_tokens`, a
true data dependency on the integrated host-array combine path that cannot be
overlapped without moving sampled tokens GPU-resident (vLLM's `prev_sampled_
token_ids` device gather). The full drain is byte-exact and UAF-safe and is
**kept**.

**Device-resident sampled tokens on integrated (`VT_ASYNC_DEVICE_MIRROR`) A/B'd
2026-08-06, speed-NEUTRAL** (same-binary): c16 OFF median 2305.8 vs ON 2303.3
(0.999x), c32 2928.9 vs 2919.1 (0.997x), bands overlap. It is a drain MOVE (relocate
the drain past the host prep, not remove it), so it overlaps only the small host
prep; the drain still serializes GPU input staging, so c16 does not recover. The
drain REMOVAL, `VT_ASYNC_EXECUTOR` Option A (`row/SERVE-ASYNC-OPTION-A`, default-OFF,
input H2D staged out of capture, the faithful vLLM structure), is GREEN + RED-reproducing
but binding-A/B speed-NEUTRAL (c8/c16/c32 +0.0-0.1%), so the c16 gap is NOT the async
input path. Detail in `.agents/benchmark-record.md`.

**But the mirror FIXES a shipping correctness bug, so it is now DEFAULT ON
(ROW-SERVE-ASYNC-LLM, 2026-08-06).** Baseline async (AsyncLLM depth-2) batch-1 greedy
decode degenerated into nondeterministic token-0 garbage; the mirror is deterministic
and coherent. The missing gate now exists, `test_qwen36_async_serving` (depth-2
AsyncLLM, batch-1 + concurrency, token-exact vs the SACRED oracle): RED on `=0`, GREEN
on the default. c16 re-checked on the default: 2312.9/2303.9/2294.4 (median
**2303.9**), c32 2942.7 (no regression). Root cause + file:line in the benchmark
record. The same P0 hit classic dense `Qwen3ForCausalLM` (quant-independent), fixed by `ROW-SERVE-ASYNC-DENSE-MIRROR` (see the MXFP4 Qwen3-8B row). The intake-drain lever likewise measured NEUTRAL (2026-08-06, `VT_INTAKE_DRAIN` A/B 3+3 reps): admitting during the forward wait collapses intake -91% but shifts it into queued, arrival-to-scheduled invariant, so the recorded INTAKE term is an attribution boundary over a GPU-bound prefill wait, not reducible; lever reverted, byte-exact `VT_LOOP_TRACE` probe kept.

### DeepSeek-V2-Lite (MLA)

Medians of 3 reps, 1,024 in / 128 out. The vLLM arm runs `--moe-backend triton`,
which is its best **stable** graphed config on GB10: the auto-selected FlashInfer
CUTLASS MoE backend hard-rebooted the box five times. The substitution does not
flatter us, we lose against it.

| Concurrency | Output tok/s ours / vLLM | Ratio | Median TTFT | Median TPOT |
|---:|---|---:|---:|---:|
| 1 | 33.18 / 38.17 | 0.87x | 0.95x | 0.90x |
| 2 | 52.63 / 55.27 | 0.95x | 0.88x | **1.03x** |
| 4 | 70.36 / 81.51 | 0.86x | **1.04x** | 0.86x |
| 8 | 102.37 / 116.35 | 0.88x | **1.12x** | 0.86x |

Peak memory is the decisive win: **31.38 GiB against vLLM's 68.5 GiB**, with the
caveat that vLLM pre-reserves a fixed fraction up front while we allocate the KV
blocks the workload needs. Real difference in operating footprint, not evidence
of a lower per-token KV cost.

### Laguna-S-2.1 (NVFP4)

`poolside/Laguna-S-2.1-NVFP4`, 118B total / ~8B active MoE, 48 layers,
256 experts, ~67 GiB. Both arms NVFP4, single request, batch 1, GB10. The local
checkpoint directory is named `laguna-xs-nvfp4`, which is where the "Laguna-XS"
label in earlier revisions of this page came from; the measured model is S-2.1.

| Arm | Decode tok/s | Ratio |
|---|---:|---:|
| vLLM NVFP4, graphed | 43.10 | 1.00x |
| **vllm.cpp NVFP4**, resident decode + CUDA graph | **44.46** | **1.03x** |

The gap is CLOSED (2026-08-04, same-tool nsys graph-node tracing on both
engines). Root cause: the bf16 M=1 projection GEMVs (o_proj, qkv, router,
dense, lm_head) read their weights from GB10 UNIFIED/ATS host memory (a
`w.View()` device retag, no `cudaMalloc` staging), which the GPU reads slower
than true device memory. `VT_LAGUNA_RESIDENT_BF16W` (default-ON parity enabler)
stages every projection device-resident (one H2D copy at load); byte-exact
(ids bit-identical to the `=0` retag arm). Binding new-default decode is 44.46
tok/s (median-of-3), 1.03x vs vLLM's graphed 43.10.

Per-call the residency recovers o_proj 194 to 131 us (about 168 to 249 GB/s),
qkv 245 to 225 us, and lm_head 2410 to 1620 us/call. This supersedes the earlier
invocation / bf16-output-`cublasGemmEx` framing (measured a wash): the
invocation was never the cause, the weight's memory RESIDENCY was. Full
forensics in `.agents/benchmark-record.md`.

Memory: the roughly 2.6 GB of bf16 device copies sit in the 119 GB GB10 unified
pool; peak host RSS is unchanged at about 40.7 GiB (device/unified allocations
are not counted in `ru_maxrss`), so there is no memory regression.

## Memory

Qwen3.6-27B NVFP4, GB10, whole serving window.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.

## llama.cpp, CPU

| RPi5/A76 arm (R4-R6, `GATING`) | Result |
|---|---|
| Scope | 4-core A76, DotProd, no i8mm; 20-core binding arm does NOT transfer. buildx/QEMU-built, Pi-executed unthrottled; hashes pinned in the [campaign spec](../.agents/specs/rpi5-cortex-a76-cpu-optimization.md) |
| Assembly vs compiler SDOT (one GCC 13.3 binary) | AAPCS64 leaf wall +3.66% M1/T1, +5.08% M128/T1, +3.69% M128/T4, with 9.74-10.24% fewer instructions; M1/T4 is the -2.43% residual ([assembly evidence](bench-evidence/rpi5-a76-q8-dot-20260806.md)) |
| 64-token Qwen model gate | Byte-identical across x86, portable, SDOT and assembly arms; asm vs SDOT median TTFT -1.55%, TPOT neutral, E2E -0.13%; vs portable TTFT -33.40%, E2E -2.67%. Cortex-A76+DotProd selects assembly by default |
| Same-file llama.cpp floor (pp17/tg64) | **NOT MET on speed**: prefill 12.81 vs 27.77 tok/s (0.461x), decode 2.55 vs 3.91 (0.653x), E2E 26,018.39 vs 16,998.49 ms ([competitor evidence](bench-evidence/rpi5-a76-llamacpp-20260806.md)) |
| Peak RSS | **2.841 vs 3.747 GiB, 24.2% less**; 3 clean unthrottled reps; same-text 64-token greedy output byte-identical after trailing-newline normalization |
| `PENDING` | Pi concurrency; BF16 GEMM / speed closure (the 2.17x prefill, 1.53x decode gap profiling is W6) |

Same GGUF file both arms, `dgx.casa` GB10 aarch64 (20 cores), idle, 3 reps,
llama.cpp `237ad9b96` built fresh on the same host.

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| Prefill | **223.8 tok/s** | 177.3 | **1.18x** | **PASS** |
| Decode | 24.7 tok/s | 25.4 | 0.97x | tie |
| Peak memory | 2.83 GiB | 2.80 GiB | 1.01x | **PARITY** |

Decode lands inside llama.cpp's own run-to-run spread, and the memory difference
is 30 MiB on a 2.8 GiB working set. Prefill is the only axis with a real gap and
it goes our way. Output tokens are **byte-identical** to llama.cpp's greedy
decode and to our own CPU reference path. Single-stream only: we have not
measured concurrent serving against llama.cpp's server.

## MLX-LM, Apple M4

Qwen3-0.6B, warm, batch 1, 6 interleaved runs.

| Axis | vllm.cpp | MLX-LM | Ratio |
|---|---:|---:|---:|
| Prefill TTFT | **524.5 ms** | 532.6 ms | **1.015x** |
| Decode | 27.23 tok/s | 27.85 | 0.978x |
| Warm total | 24.37 tok/s | 24.96 | 0.976x |

The 2.4% is a real gap, not noise: our spread was 0.12% and MLX-LM's 0.34%. All
of it sits in decode, 0.81 ms per token. Indicative rather than binding: two
models, 18 of 75 ops native, and the 97.6% needs the optional MLX GEMM provider
shape-gated to prefill (95.9% on the default build).

## DwarfStar, GGUF

DeepSeek-V4-Flash cannot run on vLLM on a single GB10 at all: every
vLLM-loadable checkpoint is 156 GB or larger against a 119 GiB unified pool, so
the only quant that fits is extreme-low-bit GGUF, which vLLM cannot load here.
GGUF was forced by the hardware. A policy-correct vLLM comparison needs 2x GB10
Sparks with TP2 and is owed.

| Engine | Quant | Decode tok/s | Ratio |
|---|---|---:|---:|
| DwarfStar (`ds4`) | IQ2_XXS mixed | 16.33 | 1.00x |
| **vllm.cpp** (default) | same GGUF | **16.28** | **0.997x, parity** |
| **vllm.cpp** (`VT_V4_RESIDENT_W` default-ON) | same GGUF | **18.69** | **1.144x, byte-exact** |

The default arm is parity, measured same-session clean (2026-08-04, single-load
steady both arms); the earlier 15.87/96% and 17.13 figures are superseded.

Weight residency is the beat-path (2026-08-05, `VT_V4_RESIDENT_W`, default-ON). Env var allowlisted (env-doc gate green).
The dense Q8_0 MLA/shared-expert/lm_head projection tower is read from the GGUF
mmap over ATS/unified memory, which the GB10 GPU reads about 20% slower per-GEMV
than `cudaMalloc`'d device memory. Staging that ~6 GiB tower device-resident once
at load (same bytes, same kernel, same invocation) lifts decode 16.23 to 18.69
(median-of-3, drop_caches), generated ids byte-identical. It is the same lever
that took Laguna to vLLM parity+ (`VT_LAGUNA_RESIDENT_BF16W`).

PEAK RESIDENT is flat at 86.68 GiB in both arms: the staged copy is additive but
the clean mmap file pages evict under the unified pool, so net usage does not
grow. An nsys A/B (identical instance counts) confirms the mechanism is
residency-bound, not latency-bound: per-launch time drops about 20% on every
dense Q8_0 kernel (`QuantDotGemmQ8_0Kernel` 184 to 147 µs, `Q8_0GroupDiagKernel`
212 to 166 µs, `Q8_0PairKernel` 74 to 60 µs). This corrects the earlier
"per-launch GEMV parity / Q8_0 weight-stream floor" framing: our GEMV was
ATS-bound, not at ds4 parity.

Phase-2 staged the routed-expert slabs too (the ~70 GiB IQ2/Q2_K bulk,
`VT_V4_RESIDENT_EXPERTS`, first-touch `cudaMalloc` plus immediate
`madvise(MADV_DONTNEED)` per slab so the transient stays ~flat). It was **measured
NEGATIVE (2026-08-05) and is HELD default-OFF** as a characterized artifact.
Same-binary median-of-3, warm-cancelled steady, drop_caches: OFF (Phase-1) **19.43
tok/s** vs ON **18.76 tok/s** (0.966x, ~3.4% slower), generated ids byte-identical
(md5 equal across all 6 runs), PEAK RESIDENT flat at 86.6 GiB. The move itself
works: host RSS drops 86 to 14 GiB as the mmap pages are reclaimed.

The regression matches the roofline. Unlike the dense Q8_0 tower (63% of DRAM
peak, bandwidth-bound), the grouped-MoE `QuantDotGemmGrouped<IQ2_XXS>`/`<Q2_K>`
kernels run at only ~19-24% of peak (dequant/latency-bound), so weight residency,
a bandwidth lever, cannot help them. It also adds a large one-time graph-capture
cost, and pinning the 70 GiB as `cudaMalloc` (vs evictable mmap file cache) cuts
the unified-pool reclaimable headroom from ~103 to ~30 GiB avail. The lever stays
in the tree, default-OFF, for reproducibility; detail in the benchmark record.

## Speculative decoding

| Speculator | Model | Result | Status |
|---|---|---|---|
| MTP | Qwen3.6-27B NVFP4 | token-identical to vLLM MTP, **~4% faster at c1**; on-par at c2-c8 | `DONE` |
| DFlash | Qwen3.6-27B NVFP4 | **2.9x over spec-off** (10.16 → 29.32 tok/s), at/above vLLM DFlash-on (**1.003x**, non-overlapping bands) | `DONE` |
| n-gram | Qwen3.6-27B NVFP4 | draft-free (`SPEC-NGRAM`); 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted (correctness only, no speed row yet) | `DONE` |
| Breadth (EAGLE1/3, suffix, ngram-gpu, dspark, dynamic-k, ...) | n/a | enumerated from vLLM source + `INVENTORIED` 2026-08-06 (`.agents/specs/spec-decode-inventory.md`), unmeasured | `INVENTORIED` |

## How we measure


Record dates are CI-guarded: state anchors dated in the future are rejected
(`check-state-order`), so scoreboard stamps trace to real landing dates. The
review protocol behind these numbers is guarded the same way: the reviewer and
implementer sub-agent prompts are tracked artifacts checked by
`check-protocol-consistency` (orchestration harness step 5/5), and
`check-gate-commands` pins the 25 record rows that name a gate command able to
FAIL. That pin is exact, not shrink-only: gaining a gate command reddens it too,
so the set is never re-pinned silently in either direction. Since 2026-08-07,
a PR verified green merges in that same session (disposition rule).

**Hardware.** NVIDIA GB10 / DGX Spark (sm_121a) for CUDA, `dgx.casa` aarch64 for
CPU, Apple M4 for Metal. GB10's 119 GiB pool is unified, so host and device
memory compete; end-to-end wall-clock on a cold page cache is unusable there,
and steady-state per-step timing or `nsys` GPU-busy is the anchor. The
2026-08-06 #77-slip tree-revert changed no benchmark content or number.

**Oracle pin.** vLLM 0.26.0.dev0 (`55596792`) plus transformers 5.14.1, built from
source for sm_121a. Speed figures labelled 0.25.0 are the last binding run; the
engine is unchanged by the pin advance and a 0.26 re-benchmark is pending.
Correctness re-validated bit-identical across the advance, zero golden drift.

**Protocol.** Greedy, closed loop, three interleaved repetitions per point, one
`flock` across the whole series, same-binary A/B for every lever, cold legs
discarded. Workload equivalence between arms is audited, not assumed: batch cap,
token budget, context, corpus bytes, KV and SSM dtypes, kernel family, and
graphed decode all match, and the audit is
[recorded](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md). The
2026-08-04/08 governance checkpoints (record/CI substrate, anchor backfill,
operator/helper W0-W5, upstream/device inventory, onboarding probe, the
review-hardened `agent-start.py` entrypoint, and CI-bound review-until-PASS
policy) touched no engine code and moved no number: **NOT APPLICABLE**, nothing
to reproduce.

The PR #28 sanitizer repair is also NOT APPLICABLE to performance: both full
333-test CPU detector lanes pass after merging upstream `main`, while the
ASan+UBSan build footprint falls from 93 GiB to 5.7 GiB and TSan occupies
1.9 GiB. Reproduce with the sanitizer
CTest commands recorded in `.agents/state.md`. The 2026-08-06 live-state audit, its standing preflight+CI gate (`scripts/audit-live-rows.py --check`) and that gate's review follow-up (evidence-rule limit recorded, `ACTIVE` precondition written into `.agents/workflow.md`, count/scope fixes) are likewise NOT APPLICABLE: bookkeeping, a record checker over matrices and Git refs, prose. No engine code, no kernel, no number on this page.

**Vocabulary.** *Token-exact* means our output ids equal the reference's, byte
for byte. *Near-tie* means the reference's own greedy decode is not deterministic
at this precision, so the gate is distributional: our output must fall inside the
set the reference produces across K runs. *Tie* means the difference is inside
the measured run-to-run noise band, which is 0.5% on GB10 and 0.12% to 0.34% on
M4. We never publish a partial, contended, or stale-denominator number as
binding, and when a denominator turns out to be wrong we correct every ratio
built on it rather than keeping the flattering one.

**CPU elementwise GEMM, wide x86 tiers (2026-08-07).** INDICATIVE ONLY, not binding: the x86 dev box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. The AVX-512 tier measures 1.56x to 2.83x over SSE2 on the elementwise micro-kernels, byte-identically. A binding number needs a qualified x86 host, which the project does not have.

**CPU elementwise GEMM, transpose-free `[K,N]` path (2026-08-07).** On dgx aarch64 the `[K,N]` path beats `[N,K]` by 1.16x to 1.30x, byte-identically. The x86 arm is INDICATIVE ONLY, not binding: that box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. `VT_CPU_MATMUL_STEAL` ships default OFF and is NOT measured; it must justify itself by measurement and may measure neutral.

## Open gaps

| Track | Status | Next gate |
|---|---|---|
| Surface coverage (`ARCH-ONE-SURFACE`) | **CORRECTNESS COMPLETE:** #139 restores DSR 32 (`kcuda=0`) via registry/name resolution; ABI-v14 selection unchanged; no speed claim | Selector 2/2·11 plus execution-bound CMake/File-API/CTest + CI/preflight + manifest-integrity guard 52/52; CPU platform/loader/C-ABI tests green; CUDA A/B remains residual |
| 35B prefill TTFT | 0.93x to 0.98x at every concurrency (2026-08-05) | Attribute the residual, then close |
| 35B low-batch MoE decode | CLOSED at low batch (c1 0.975x, c4 wins); c16 0.93x. `VT_ASYNC_DEVICE_MIRROR` **default ON for correctness**. `VT_ASYNC_EXECUTOR` Option A (H2D out of capture) A/B'd speed-NEUTRAL | c16 lever is prefill glue (task #61), not the decode drain. `test_qwen36_async_serving` GREEN |
| CPU keep-quant MoE decode | **No number owed**: correctness-only P0. The grouped keep-quant GEMM read activations as f32 whatever their dtype, so CPU MoE decode emitted token-0 garbage from `b4f5610a` (2026-07-31) | Speed unmeasured and unclaimed; `test_ops_quant_dot` GREEN (150224 assertions) |
| DeepSeek-V2-Lite MLA | Attributed miss, `ACTIVE` | Throughput at every concurrency |
| Qwen3.5 upstream throughput levers (roadmap C10) | NOT MEASURED. vLLM's 2026-08-06 25K tok/s/GPU is a GB200/NVLink72 disaggregated cluster result, not comparable to one GB10, and is NOT adopted as our bar | Advance the parity pin past `555967922` so the referenced PRs exist, re-capture goldens at zero drift, then port the GDN prefill kernel |
| Laguna-S-2.1 NVFP4 | **CLOSED 2026-08-04, parity+**: `VT_LAGUNA_RESIDENT_BF16W` default-ON (bf16 weights unified/ATS → cudaMalloc device-resident) → 44.6 vs 43.1 tok/s, byte-exact (o_proj 194→131, lm_head 2410→1620 us/call) | none, closed |
| DeepSeek-V4-Flash | **Parity with ds4 (0.997x)** | Optional beat-path: f16 tensor-core DSA/router (near-tie class) |
| DeepSeek-V4-Flash vs vLLM | Infeasible on one Spark | 2x GB10 with TP2 over the NCCL seam |
| Tensor parallelism (task #287 spike, 2026-08-08) | **No number owed** (`benchmark_binding=false`): records-only scope at pin `555967922` ([spec](../.agents/specs/tensor-parallelism-spike.md)). TP-W1 landed (group table, 6/6); TP-W2..W7 gate on token-exactness, not speed | Perf gate is TP-W6: at or above vLLM TP=2 on every axis, same 2-GPU box (PENDING-HW) |
| DFlash speculative decode | **CLOSED 2026-07-27 (D14)**: warp-scoped draft attention (242.9 → 77.9 ms), c1 our-on 29.32 vs vLLM-on 29.24 tok/s, non-overlapping 3-rep bands, 1.003x | none, closed |
| Multimodal image, audio, video | Correctness gated, speed unmeasured | Per-modality speed grids |
| `/v1/videos` OpenAI + ONE-SURFACE ROW 2 | **No number owed:** ABI-v12 device selection is backend-dispatch plumbing; generation math and speed paths are unchanged | DSR 34→32; baseline/allowlist unchanged; 25/25 checker mutations; CPU fold 6/137, including one-queue/device-provenance mutations |
| Qwen3-dense decode CUDA-graph | Token-exact pass, ~4.3% e2e directional | Steady-state per-step tok/s |
| Kimi-Linear-48B-A3B (KDA+MLA+MoE) | **RUNNER FOLD LANDS (ROW 7, §21, #122): engine==CLI 128/128 byte-identical; vs golden 122/128 (near-tie profile); FA2 MLA default-ON; SACRED green.** Server 19.0 tok/s wall; CLI 18.93 reproduced | vLLM ~21 (#111 floor; in-session re-measure ABORTED by GB10 reboot at util 0.82, §21): **~0.90×**, >= vLLM NOT met; residual = KDA host islands + grouped MoE + decode graph |
| vLLM 0.26 re-benchmark | Pending | Re-run the binding grids on the advanced pin |
| MiniMax-H3 FP4 speed (W-FP4a) | **Measured GB10 (`row/H3-FP4-GPU-E2E`).** Marlin W4A16 byte-exact vs bf16; fp4 a memory win, 0.8x bf16/forward. Real-ckpt fp4-resident e2e RUNS (mp4/wav) | fp4 speed CLOSED. bf16-vs-quant A/B: ENCODER half MEASURED (§8.15), DiT half NOT (no bf16 render exists). Detail: benchmark-record + spec §8 |
| MiniMax-H3 render coherence (`row/H3-RENDER-CLOSE` #77) | **CLOSED: a COHERENT scene on GB10.** #70/#74 white was wrong-PARTITION usage (t2va on the ref2va ckpt); t2va on the FL2VA GGUF renders a prompt-matched orange cat (adj-cos 0.95 vs 0.06, no patch-grid) | Verified first: t2va inputs byte-exact vs upstream; CUDA device==host at seq 1920. Follow-up `H3-TASK-PARTITION-GUARD`: the task/partition mismatch now RAISES 1:1 with `_resolve_task` (spec §8.6-8.7) |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`) | **fl2va COHERENT; ref2va assembly bug FIXED+gated.** vision→cond scatter gated; ref2va block-dim double-division fixed + RED-first gated (128 vs 512) + a permanent ref2va DiT-forward rung (§8.10) | grid RE-ATTRIBUTED: with the fix ref2va grids in fp4 AND bf16, and t2va with no refs on the ref2va NVFP4 also grids while FL2VA-GGUF renders, so it is the **NVFP4 checkpoint/loader**, NOT assembly/fp4 (§8.10) |
| MiniMax-H3 Thor render speed (sm_110, no FA2) | **34.6 s/step** at 864x480/124f/50 steps on Q4_K_M, **16.6x** off 574.5 (render ~28 min, was ~8 h). Landed: warp-per-query, chunked warp reduce-scatter (1.76x), bf16 `mma.sync` (9.82x) | Shared-memory K/V tiling (23% SLOWER) and register Q-blocking (-0.8%) both measured and REVERTED: memory traffic is not the bound (one head's K+V is 3.9 MB against 32 MB of L2) |
| MiniMax-H3 render duration (audio halving) | **CLOSED**: a 124-frame render silently muxed as **61**. The decoded audio ran half the video's duration and the muxer passes `-shortest`. Every structural check passed: shapes were self-consistent, just halved | Gated on the duration invariant (latent steps / 40 Hz equals the video duration), not on shape self-consistency, which a halved pipeline satisfies |
| MiniMax-H3 quantization floor | **Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude outliers 3-bit cannot hold; a controlled A/B (same prompt, seed, code) turned a murky lattice-covered silhouette into a photoreal close-up | Per-tensor mixed precision, if a smaller footprint is ever owed |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`, `row/H3-NVFP4-LOADER-DIFF`, `row/H3-NVFP4-STREAM-DIFF`) | **fl2va COHERENT; ref2va NVFP4 nibble loader bug FIXED (byte-verified); grid residual DIAGNOSED §8.12** | Activation diff + fingerprints (PR #95): NO load-path defect; all weights/islands/RoPE quant-noise-close to the coherent GGUF; grid = community-NVFP4 quant fidelity, not a loader fix. See benchmark-record |
| MiniMax-H3 encoder quantization (`H3-ENC-BF16-COND-DIFF`) | **Measured Thor (`d1085374`).** Q4_K_M vs bf16 encoder, same 233-token prompt, same forward: rel RMS **0.0340** (0.0685 excl. sink), per-token cosine mean **0.99745** / min 0.909, median rotation **3.5°** | NOT a scale change (best rescale 0.0340->0.0328). Same energy as a ONE-WORD prompt edit but DIFFUSE: 232/233 tokens rotate vs 172/233 untouched. Render A/B owed. Detail: benchmark-record |
| MXFP4 Qwen3-8B (W4A16 Marlin) | **`KERNEL-MARLIN-DENSE-EXEC` x3 (dense-ON default): c1 1.020, c2/c4/c8 0.962/0.966/0.969, GPU mem 2.63x less** (beats #51 1.005/0.925/0.939/0.953 EVERY axis); #44 3/3, 32B-NVFP4A16 6/6; -Werror test-guard fixes x2 | **VT_MARLIN_DENSE default-ON** (+951us). `FLASH-PTXAS` #82: cuModule A/B ties our+vLLM PTX across ptxas 13.0/13.2/driver-JIT (~144us); +10us is engine CONTEXT not codegen, no ptxas lever/flip (retires #75) |
| Vulkan vs llama.cpp Vulkan (`BENCH-VK-LLAMA`) | **Both arms measured, same weights.** 0.6B @128-in/32-out: llama.cpp Vulkan **11,956** pp / **174.8** tg; ours **575** pp / **66.6** tg | Decode **8.59 -> 91.7 t/s** (**10.7x**), 6/6 exact; **2.62x** off llama.cpp at matched shape. CUDA arm unblocked. 27B: fallbacks **11->5**. paged_attn batching REFUTED ([plan](../.agents/specs/bench-27b-five-way.md)) |
| Memory footprint vs declared workload (`ROAD-V1-MEM`, #83) | **Never measured, and not measurable today**: there is no auto-sizing to compare against, because the KV pool is a hand-typed `--num-blocks`, so "what the run actually needed" has no number | Once M1's `MemoryBudget` lands: predicted-vs-actual bytes per allocation class, then peak footprint ours-auto vs vLLM at its 0.9 default on the same model and config |
| Startup latency (cold to first `/health`) | **36.51 s vs vLLM 0.25.0's 221.51 s = 6.07x** (medians of 3, 27B-NVFP4, GB10). PROVISIONAL: 3 of 6 legs contended, repeat killed by a host reboot. [Detail](../.agents/benchmark-record.md) | Uncontended 3-rep re-run on a quiet box |
| Speculation depth (`ROAD-V1-D3-SPEC-K`, #81) | **Never measured, MTP is k=1** (our port covers vLLM's k=1 branch only), so no acceptance-vs-depth curve exists | k=2..4 three-way greedy gate, then the c1/c>1 A/B + the per-workload (prose vs code) acceptance-vs-depth curve any dynamic or adaptive depth policy needs |
| Vulkan vs llama.cpp Vulkan (`BENCH-VK-LLAMA`) | 25 NATIVE (+8 GDN). **27B prefill 21.5x**, decode **4.285**/4.35. Host MEASURED ~3 ms/tok not 5.3; pipelined submit **-1.41 ms/tok 5/5**; fence spin REJECTED. [source](../benchmarks/demo/vulkan_27b_llamacpp.json) | `VK-C` coopmat A/B on Thor (`VT_VULKAN_COOPMAT=0` A/Bs it): **11.1x-32.9x** vs our UNTILED scalar kernel, not vs a competent GEMM. `VK-E`: llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96` on dgx, same GGUF, three columns |
| ROCm (`BACKEND-GATE-ROCM-VLLM` / `-SGLANG`) | **NOT APPLICABLE: no number measured, claimed or owed.** W0 ctest-green on 4 gfx archs (#41); gfx1201 hipBLAS + Gemma-4 MoE (#140, contributor) ran M0/M1 on 2× R9700, our side CPU-link-verified only. No AMD HW here | The approach-(b) fix (PENDING community) unblocks the first APU model run (M2); the gate becomes a same-box vLLM-ROCm oracle once a model runs ([#41](https://github.com/mudler/vllm.cpp/issues/41)); floor: vLLM |
| SGLang floor arms | Never ran | Both arms of the SGLang comparison |
| Embeddings on the ONE surface (ROW 6, `LlamaModel` + `vllm_embed` + `/v1/embeddings`) | **NO number measured, claimed or owed.** Correctness-gated only, CPU: the 2026-08-08 fold (engine path == direct registry path, f64 LAST+normalize reference on the committed fixture) is plumbing, no speed claim | A REAL embedding checkpoint (e5-mistral class) + a same-box `vllm.LLM(task="embed")` oracle; only then does an embed-throughput bar exist |
| Parakeet/FastConformer ASR (P1-P4 + ONE-SURFACE fold ROW 1) | **NO number measured, claimed or owed.** Correctness-gated only, CPU f32; the 2026-08-07 surface fold (`vllm_transcribe`, `/v1/audio/transcriptions`) is transcript-byte-identical plumbing, no speed claim. | Floor is `parakeet.cpp`, same clip and box; needs a CUDA provider and a pretrained checkpoint |
| cuBLAS invocation-parity guard | CI guard landed (CPU); `kGemvHeuristicAlgos` refactor build-verify owed | `nvcc` rebuild + SACRED gate on dgx |
| Ampere consumer (`sm_86`, RTX 3090 class) | **No number owed; no such board here.** 2026-08-06 build-verify: 7/7 FA2 TUs 0-warn, real `sm_86` SASS. [Detail](../.agents/benchmark-record.md) | External RTX 3090 report. Floor is llama.cpp on that card (GGUF, not our Blackwell-only NVFP4 grid) |
| Pre-Ampere breadth (Turing `sm_75` / Volta `sm_70` / Pascal) | **No number owed; nothing runs on these arches.** 2026-08-06 `sm_75`: 20/20 TUs PASS (0 err/warn), WMMA bodies + all 3 selectors arch-gated; GB10 SASS byte-identical. [Detail](../.agents/benchmark-record.md) | Full-library LINK at `sm_75` + `cuobjdump` SASS, then a build-supported row. The fp16 `fattn` port is speed-only now; its floor when a card exists is llama.cpp on that card |

## Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Not yet runnable (no model runs on Vulkan). Planned harness in `.agents/specs/vulkan-full-support.md` §5.2 |

Build flags, environment variables, and the full gate list are in
[BUILD.md](BUILD.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
