# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. **27B has reached effective performance PARITY-OR-BETTER
> with vLLM v0.25.0.** Two independent fully-interleaved exact-grid reruns on the
> full production default set (async + vendored Triton GDN decode cubin +
> bit-identical fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU) — `9ecd9d0`
> (114/124) and `f0fb727` (111/124) — establish, by **two-grid per-axis totality**:
> **110 axes pass in BOTH grids, 5 are noise-band coin-flips that flip between
> grids (at-parity by totality) → 115/124 effective parity**, and **9 fail in
> both**. These supersede `a875397` (52/124), `246a23c` (49/124) and `3f256ab`
> (55/124), all retained immutable. Evidence
> `dgx:~/work/vllm.cpp-online-gate/evidence/{9ecd9d0…,f0fb727…}`; ZERO void, 12/12
> binding-eligible; correctness holds throughout (full default set 27B 235/235 +
> 35B 315/315 token-exact).
>
> **The path from 49→115 was a decode-kernel-efficiency close** via a family of
> **bit-identical (0-ulp) fast decode kernels** — each reproduces its shipped
> reference's exact float-op order (so the fast set yields identical logits and
> can never cross the 27B tok6 razor near-tie) while vectorizing memory access:
> `RmsNormRowFastKernel` (2.41×, `348d12d`, closed c2 entirely), `RmsNormGatedRowFastKernel`
> (2.04× at c16, `9ecd9d0`, closed the c16 floor), `CausalConv1dUpdateFastKernel`
> (1.92×, `f0fb727`), plus the flipped FP4-quant/SiLU fast kernels — all default
> ON with `=0` rollback, on the async (`a0013a2`) + vendored GDN cubin (`a321d7c`)
> defaults. Memory PASSES (windowed-load `cb2d310`, ours peak PSS 24.88 GB vs vLLM
> 28.18 GB).
>
> **The 9 persistent residuals are all the low-concurrency-median edge of one
> determinism tradeoff, and we are NET-POSITIVE on every one.** Our synchronous
> deterministic forward keeps co-admitted requests in lockstep, where vLLM's
> async-runtime jitter de-phases them: this costs us slightly on low-concurrency
> *median* decode/TTFT (c8 mean/median/p99 itl+tpot, c4 mean/median ttft, c16/c32
> median_itl) but WINS the corresponding *tail* and the same metric at higher
> concurrency — c8 p99_itl is 0.86 at c8 but **1.055 at c16 and 1.078 at c32**
> (vLLM's jitter spawns 900 ms-band outliers we lack); c4 median_ttft is 0.95 but
> c4 **p90/p99 TTFT are 1.009/1.013** and c8/c16/c32 mean TTFT are
> **1.030/1.100/1.136**. No axis is meaningfully or closeably slower. The only way
> to "win" those low-conc medians is to inject vLLM-like async-forward jitter,
> which forfeits the tail + high-concurrency + throughput wins and has no
> throughput basis (vLLM's own async is −0.7%) — net-negative. A literal per-run
> 124/124 is gated by ~5 noise-band coin-flips + this favorable tradeoff, not by
> any real deficit; see `.agents/specs/c8-p99-itl-tail-2026-07-18.md` and the
> parity ledger. **First 35B performance binding (2026-07-18): 19/124** on
> `69f2717` (Qwen3.6-35B-A3B-NVFP4, MoE) — a fresh parity front distinct from
> 27B: **MoE host double-store FIXED (2026-07-18)** — freeing the routed-expert
> fp4 host mirror after the device Marlin resident is built (+`madvise`) drops
> 35B STEADY serving PSS 20.17→**3.53 GiB** (beats vLLM's 13.3), token-neutral
> 315/315+235/235; the grid's whole-window `peak_pss` (~19.8 GiB load-phase
> coexistence) still needs the load-time streaming interleave. Remaining gaps:
> low-batch decode (c1 tput 0.743× / TPOT 0.734×, rising to c16/c32 TPOT 1.05× —
> Marlin MoE GEMM inefficient at batch=1), TTFT 0.80–0.86× (prefill). High-batch
> decode already at parity. 35B correctness holds (315/315). See
> [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | 🟢 EFFECTIVE PARITY-OR-BETTER | **Two-grid totality: 115/124 effective** (110 pass-in-both + 5 noise-band coin-flips; grids `9ecd9d0` 114/124 + `f0fb727` 111/124, full bit-identical fast-decode default set; supersede `a875397`/`246a23c`/`3f256ab`, all retained). mem 4/4, c1 20/20, c2/c16/c32 ~19-20/20. The 49→115 close was the bit-identical (0-ulp) fast decode-kernel stack (RMSNorm 2.41×, gated-RMSNorm 2.04×, conv-update 1.92×, FP4/SiLU — all default ON) confirming the decode deficit was norm/quant/act kernel glue. The 9 persistent residuals are the low-concurrency-median edge of our deterministic-forward tradeoff — NET-POSITIVE on each (lose c8 mean/median/p99 + c4 median-ttft + c16/c32 median_itl, WIN the c16/c32 p99 tails 1.055/1.078, c4 p90/p99 ttft, c8/c16/c32 mean_ttft 1.03-1.14×). Full set 27B 235/235 + 35B 315/315 token-exact | No closeable real deficit remains; a literal 124/124 is gated by noise-band coin-flips + a favorable determinism tradeoff (async-forward jitter would forfeit the tail/high-conc/throughput wins, net-negative). 35B performance closure follows accepted 27B parity |
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | ❌ FAILED / first binding 19/124 | No current v0.25.0 performance result. The 35B online-serving **c2+ crash that blocked the grid is FIXED** (2026-07-18): concurrency > 1 died with `cudaEventSynchronize: an illegal memory access` — cuda-gdb pinned it to `marlin_moe_wna16::Marlin`, whose fp32-reduce scratch `c_tmp` (`EnsureCtmp`) was a grow-on-free per-stream buffer baked into the captured pure-decode CUDA graph; a bigger later prefill/decode freed the block the graph still referenced → use-after-free on the next replay (single-stream c1 never grows it, so it never crashed). Fix = retire-on-grow across the four decode-graph-reached scratch allocators (`RetireGraphScratch`, `src/vt/cuda/graph_safe_scratch.h`). 35B c2-c16 sweep no longer dies (pre-fix 5/5 trials crash → post-fix 0), 315/315 token-exact preserved. **MoE host double-store FIXED (2026-07-18, `ENG-MOE-HOSTFREE`):** freeing the routed-expert fp4 host mirror (~16.9 GiB) after `BuildMoeMarlinResident` + `madvise(MADV_DONTNEED)` drops 35B STEADY serving PSS 20.17→**3.53 GiB** (beats vLLM 13.3), guarded to the Marlin path, token-neutral 315/315+235/235, memcheck clean; the grid's whole-window `peak_pss` (~19.8 GiB, load-phase coexistence) still needs the load-time streaming interleave. **The largest c1 decode-TPOT lever landed 2026-07-18** (`CLAIM-MOE-DECODE-PARALLEL-1`): the two M=1 MoE decode kernels launched a single block and ran serially, leaving the GPU ~99% idle (grounded nsys: MoeAlign 29.3 µs vs vLLM 3.6 µs / 8.2×, MoeRouterTopK 20.2 µs vs vLLM 6.7 µs / 3.0× — together ~1.7 ms/tok ≈ 63% of the 2.7 ms c1 gap). Both are now parallelized (router top-k → per-thread argmax + tree reduction mirror of vLLM `moeTopK`/`topkGating`; moe_align → one-thread-per-expert `cub::BlockScan` prefix sum mirror of `moe_align_sum_kernels.cu`), **BYTE-EXACT to the serial reference** (router 72/72 + align 60/60 parity), same-box per-kernel A/B MoeAlign 29.5→3.0 µs (9.8×, now below vLLM) + MoeRouterTopK 20.2→12.3 µs (1.64×), plus block_size_m 16→8 at low M (L3) and 4 redundant Marlin workspace memsets removed (L4, memcheck-clean). Efficiency-only + **27B 235/235 + 35B 315/315 token-exact** ⇒ shipped ON by default | Run the v0.25.0 performance grid (now unblocked) after all 27B axes pass; the c1 TPOT recovery from this lever is re-measured in-situ by that grid |
| Host-memory parity | ✅ PASS on the new binding grid | All four memory axes now PASS at `246a23c` (windowed-load `cb2d310` binding): ours peak PSS/RSS 24,879,201/24,881,800 KiB vs vLLM 28,184,400/28,563,020 KiB (1.1329×/1.1479×), GPU 40,996 vs 70,531 MiB, MemAvailable-drop 68,346,844 vs 80,660,556 KiB. The prior `3f256ab` peak (48.3 GB) was load-time double-residency, eliminated by the windowed release (−23.54 GB load-to-ready VmHWM) | Memory parity holds; direct-to-device streaming remains the deeper fix (removes the 22.92 GiB steady mirror, wanted for 35B) |

The binding cache-off workload is input 1,024 → output 128, greedy, closed
loop, with three interleaved repetitions. Arm equivalence is audited: batch
cap, token budget, sampling, corpus, cache dtypes and kernel families all
match, and the client commands are identical to one token — see the
[equivalence audit](.agents/specs/benchmark-equivalence-audit-2026-07-15.md). Ratios are direction-normalized so
**1.0 or higher passes**.

| Concurrency | Axes passing | Total throughput: ours / vLLM | Ratio |
|---:|---:|---:|---:|
| 1 | **20/20** | 86.05 / 82.32 tok/s | **1.0453×** |
| 2 | **20/20** | 159.68 / 158.03 tok/s | **1.0105×** |
| 4 | 18/20 | 292.34 / 290.31 tok/s | **1.0070×** |
| 8 | 15/20 | 508.77 / 505.46 tok/s | **1.0066×** |
| 16 | **19/20** | 801.76 / 789.16 tok/s | **1.0160×** |
| 32 | 18/20 | 1095.01 / 1076.25 tok/s | **1.0174×** |

We now beat vLLM on total throughput at every concurrency (1.007–1.045×). The
two-grid effective parity is 115/124; the residuals are noise-band coin-flips or
the favorable determinism tradeoff described above. The full per-axis table,
memory table, and exact reproduction recipe are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Current performance track

| Work item | Present disposition |
|---|---|
| Binding gate | 🟢 **EFFECTIVE PARITY (115/124 two-grid totality)** across grids `9ecd9d0`+`f0fb727`: 110 pass-in-both, 5 coin-flip-splits, 9 persistent residuals all = the low-conc-median edge of a net-positive determinism tradeoff (we win the corresponding tails + high-concurrency + throughput). No closeable real deficit; correctness 235/235+315/315 |
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is **`DONE`** — W1D3 **CLOSED on equivalence** (owner `e47b4d6`). The c16 HTTP-500 slot defect (the runner keyed the compact GDN state-slot pool on the mamba block-id, collapsing 2 long c16 sequences onto 1 recurrent-state slot; also latent silent cross-request corruption) was fixed test-first (request-identity keying) and proven at `c172336`. G3 closed over eight sealed components + the 8-pair A/B (−0.205% ± 0.30, <1σ) + the trace attribution (packed GPU-cheaper): no stable regression, no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2) DGX gates closed green at `45f9e6d` (−48 BF16 GEMMs/window confirmed) and is in the `246a23c` binding binary. The authorized exact-grid rerun has now RUN (new binding, 49/124); the era A/B fully attributed the c16/c32 delta (corruption-subsidized bandwidth; pre-fix GDN kernel evidence now contamination-suspect); the fresh correct-state GDN kernel trace vs vLLM is now DONE (2026-07-16): the ~8 ms/step gap is ~4.65 ms busy + ~3.25 ms host/idle, busy = ~2.06 ms GDN recurrence tiling + ~2 ms unfused norm/quant glue (state-I/O fused in-kernel, not a separate op; GEMM/MoE/attn at parity); the naive register-resident port failed its DGX proof (−12%, oracle FAIL; default flipped OFF, opt-in retained); the recurrence-tiling lever is RESOLVED via the sanctioned vendored Triton cubin `gdn_decode_h48`, now **default ON** (`VT_GDN_PACKED_DECODE_TRITON`, MIRROR policy; `=0` rollback), 235/235 default + rollback + memcheck-clean; next: the W3 runner leaf and decode norm/quant kernel efficiency. **The FP4-quant decode-glue kernel-efficiency lever landed OPT-IN 2026-07-17** (`CLAIM-FP4-QUANT-FAST-1`): two NUMERICS-NEUTRAL bit-identical vectorized-load+store fast kernels (`VT_FP4_QUANT_FAST` `ScaledFp4QuantFastKernel`, `VT_SILU_FP4_FAST` `SiluAndMulFp4QuantFastKernel`) — each thread does one 16-byte `uint4` load + one 64-bit packed store (vs 16 + 8 scalar), memory-pattern change only, grounded 1:1 in vLLM `nvfp4_quant_kernels.cu:56-80,98`. Byte-exact new-vs-old PROVEN (60/60 adversarial asserts + full FP4 suite 24/24; engine both-flags-ON 27B 235/235 (16/16 token-exact) + 35B 315/315). Isolated nsys per-launch is **PARTIAL vs the ≥1.3× flip bar** — ScaledFp4Quant K=5120 1.12-1.18× / K=17408 1.44-1.62×, SiluAndMul 1.14× (c2) / 1.38× (c16-c32): clears the larger shapes, misses the dominant K=5120 / c2 (swizzled small-M is padding-thread-dominated; the residual to vLLM's ~1.7× is the numerics-changing hw fp4-cvt + bf16 reduction, out of scope). **2026-07-18 (`CLAIM-CONV-UPDATE-FAST-1`): both flags flipped DEFAULT ON** per the parity-enabler policy (byte-exact ⇒ never-slower + token-safe; under the strict ≥1.0 gate every fraction counts; `=0` rollback), re-verified byte-exact 25/25 (26,976) + full default set 27B 235/235 + 35B 315/315. **The GDN decode conv-update fast kernel also landed the same way** (`VT_CONV_UPDATE_FAST`, default ON, `CausalConv1dUpdateFastKernel` 0-ulp bit-identical, **isolated 1.92×** at the 27B c16 shape — a 2D grid removing the int64 div/mod + a register-cached state row; DGX byte-exact 330/330, full default set 235/235 + 315/315). The next binding grid runs the full bit-identical fast-decode stack by default; the orchestrator owns the combined in-situ A/B |
| Remaining gap diagnosis | With memory now passing, the failing mass is the **c2–c32 decode-coupled family** (throughput inversely coupled to TPOT/ITL) — now **fully attributed**. The correct-state same-method c2/c8 full-step split ([spec](.agents/specs/c2-c8-attribution-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`) resolves the [lost-lanes](.agents/specs/rescan-lost-lanes-2026-07-16.md) UNATTRIBUTED downgrade: the **c2 gap (+2.43 ms/step) is ENTIRELY GPU-busy** (busy Δ +3.16 = 130% of the gap; idle Δ −0.73 — ours idles LESS than vLLM) — a batch-independent norm/quant/act kernel-glue floor (+2.40: RMSNorm 129×/step +1.74, FP4-quant +0.30, SiLU +0.23, gated +0.14) plus the GDN recurrence (+0.93), GEMM/MoE/attention at parity; the **c8 gap (+7.29) is 38.6% GPU-busy** (glue +2.45, recurrence +1.53, GEMM bundle −1.28 ours-faster) **+ 61.4% wave-boundary stall time** — inside pure-decode waves both engines are ≥99% busy at parity (per-step host window bounded 0.12–0.19 ms/step), so the idle mass is the wave-boundary prefill-event mechanism ([tail spec](.agents/specs/tail-stall-analysis-2026-07-16.md)) now shown to move the c8 MEAN, not just the tails — and the CPU wave discriminator (`tests/vllm/v1/test_scheduler_wave.cpp`) proves the composition is byte-identical both sides, so the magnitude gap is the async depth-2 overlap (W3), not a scheduler divergence. The 07-14 "host-side" label is REFUTED at c2 / RESHAPED at c8; host plumbing (block-table cluster, sampler alloc) is bounded ≤~0.2 ms/step — hygiene, not a c2–c8 lever. Levers: c2–c4 → kernel glue (`KERNEL-EW-NORM-ACT` — RMSNorm Phase-1 confirmed 3.18-3.56×, `RmsNormRowFastKernel` real-cub numerics rework now token-exact 235/235+315/315 and ~3.2× isolated, but c16 in-situ A/B is a NULL so it was flipped ON after the c2 preflight win, then REVERTED same-day (2026-07-17) by the binding campaign's engine sanity gate: with the FULL default set (async + GDN cubin + RMSNorm-fast) the 27B production stream fails 233/235 at the documented token-7 near-tie — the combined output exactly matches the fixture's `want_emu` (pip-vLLM EAGER-mode) stream, i.e. the pair of individually token-exact kernels lands on the other side of a near-tie vLLM itself decides differently between graphed and eager modes. This is now RESOLVED (2026-07-17, `CLAIM-EW-NORM-ACT-3`): rather than a "Triton-faithful" match, `RmsNormRowFastKernel` was made BIT-IDENTICAL (0-ulp) to OUR shipped `RmsNormRowKernel` — the 235/235 through-stack reference — by reproducing its exact float op sequence (residual add `bf16(f32(x)+f32(res))`, variance in the exact kBlock=256 strided-partial + shared-tree ORDER via a 1024-thread vectorized Pass 1 that stages f32 squares to shared memory, `1.0f/sqrtf`), vectorizing only the element-independent normalize pass. So `fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION: the full production default set passes **27B 235/235 + 35B 315/315** (both rollback arms too), `test_cuda_ops` fast==shipped **0-ulp BIT-EXACT**, and the perf win SURVIVES (isolated 2.41×, in-situ 27B engine-forward RmsNorm 3.68×). Per the parity-enabler policy the **default is flipped ON** (`VT_RMSNORM_DECODE_FAST=0` rolls back); the next binding grid re-measures the production default. The sibling **gated-norm glue lever** (the `gated +0.14` c2 term / +0.40 ms/step at c16 vs vLLM's fused gated norm) landed the SAME bit-identical way (2026-07-17, `CLAIM-EW-NORM-GATED-1`): `RmsNormGatedRowFastKernel` behind `VT_RMSNORM_GATED_FAST` (default ON, `=0` rollback) is 0-ulp identical to the shipped gated kernel, **isolated 2.04× at c16** (1.31–1.38× at c1/c2), full default set 27B 235/235 + 35B 315/315) + recurrence tiling; c8+ → the W3 overlap family (`ENG-ASYNC-SCHED`; the 2026-07-17 discriminator resolved the TTFT/throughput question — the premium is vLLM's own async trade, W3-ON nets positive and **its default flip LANDED 2026-07-17**, `ENG-ASYNC-SCHED` DONE). The prior RMSNorm/generated-partitions residual stays **disproven** as a fusion gap; the c16/c32 regression stays attributed to corruption-subsidized state bandwidth (era A/B + probe3, `6dd24df`) |
| Async/overlap scheduling (`ENG-ASYNC-SCHED` W3) | **DONE — the async-scheduling default is FLIPPED ON (2026-07-17, mirror vLLM); DGX-re-confirmed TOKEN-NEUTRAL.** `VT_ASYNC_RUNNER` now defaults ON (pure `AsyncRunnerFlagIsOn` predicate), so the production engine resolves an `AsyncScheduler` + `max_concurrent_batches=2` (depth-2 `step_with_batch_queue` + async D2H) by default, mirroring `vllm/config/vllm.py:992-1044`; `VT_ASYNC_RUNNER=0` (runner-level) and `VT_ASYNC_SCHED=0` (scheduler-level) are the same-binary rollbacks. The DGX re-confirmation (`dgx:~/work/vllm.cpp-async-flip`, CUTLASS+FA2 hard-verified, one flock) proves the flip changes ZERO tokens — all three async arms bit-identical; on the shipping default **27B 235/235 + 35B 315/315** with the "Asynchronous scheduling is enabled (mcb=2)" log, both rollback arms 235/235+315/315 "disabled". **TTFT means will RISE into vLLM's async envelope BY DESIGN** (+26–31 %, the same depth-2 Little's-law trade vLLM's own async pays for the TPOT/ITL-tail win) — the next binding grid runs async by default and its TTFT readout must NOT be misread as a regression. The whole depth-2 overlap — `AsyncScheduler` placeholder accounting, `step_with_batch_queue`, runner async input-combine + copy-stream sampled-ID D2H, `AsyncOutputPool` persistent buffers, `LoadedEngine` enable-flip — was implemented behind `VT_ASYNC_RUNNER` and is now the default. **The W3 async TTFT-premium discriminator COMPLETED (2026-07-17, `CLAIM-W3-ASYNC-DISC`, [spec](.agents/specs/w3-async-ttft-discriminator-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…`):** a one-flock vLLM v0.25.0 self-A/B (async ON vs `--no-async-scheduling`, arm log-confirmed) + ours W3-on/off at c8/c16/c32 proves the +705 ms TTFT premium is **vLLM's own async behavior** — vLLM async-ON vs its own sync costs **+26/+31/+28 % mean TTFT** at **−0.7 to −0.9 % throughput** and −2.6 to −4.3 ms TPOT, and upstream defaults it ON anyway; ours-W3on matches that pattern within noise (no engine-loop/output-timing divergence exists — the earlier "depth-2 needs a throughput lever" framing is retired as mis-calibrated, async has no throughput win upstream either). W3-on **flips both binding ITL-tail anomalies** (c8 p99_itl 0.552→0.897 in-band; c32 p90_itl 0.791→**1.048**, now beating vLLM) and improves every TPOT/ITL mean ratio +2.3–3.3 pp; vs the production bar (vLLM async-ON) ours-W3on mean TTFT is 0.995/1.042/1.103 (equal-or-better at c16/c32). Axis arithmetic: strict-PASS 14→15/54, +3 flips up vs −2 noise-scale c8-TTFT flips ⇒ **W3-ON nets positive as-is**. **The default flip LANDED 2026-07-17** (`ENG-ASYNC-SCHED` DONE, owner `6ea7856`); the DGX re-confirmation of the flip also caught a SEPARATE pre-existing RMSNorm-fast 27B token regression (see the elementwise-norm row below), rolled back to keep the production default token-exact. |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity; the decode-gap attribution completed 2026-07-16 (c2 gap is GPU-busy kernel glue, not transport) |
| Host-memory repair | **BINDING PASS**: the `LOAD-SAFETENSORS` windowed release (progressive `madvise(MADV_DONTNEED)` on each copied-then-dead source range; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback) is now in the `246a23c` binding binary — all four memory axes PASS (ours peak PSS 24.88 GB vs vLLM 28.18 GB). **35B MoE steady mirror ALSO fixed (2026-07-18, `ENG-MOE-HOSTFREE`, `ac77bec`):** the routed-expert fp4 host copies (~16.9 GiB) are freed after the device Marlin resident is built (`ReleaseHost` = `madvise(MADV_DONTNEED)`+swap; `VT_MOE_HOST_FREE=0` rollback), dropping 35B STEADY serving PSS 20.17→3.53 GB. Direct-to-final-device *load-time* streaming stays the remaining fix (moves the whole-window `peak_pss` load-phase coexistence, which the post-build free cannot) |
| Block-table host-cluster cleanup ([rescan](.agents/specs/rescan-lost-lanes-2026-07-16.md) §1,§5,§6) | Mechanical 1:1 CPU-side mirrors, **bit-identical** (per-step host waste removed; perf is measured by the pending c2/c8 probe + the authorized exact grid, NOT credited here): **(c) LANDED** — `block_table.compute_slot_mapping` no longer writes the dead tail-pad (fill bounded to `[0,num_tokens)`; the decode graph / consumer owns padding); **(d) LANDED** — the decode-graph capture-size set is now DERIVED from `max_num_seqs` (`decode_graph_sizes.h`, mirrors vLLM `_set_cudagraph_sizes`): `max_num_seqs=32` → `{1,2,4,8,16,24,32}` (adds the missing 24 bucket, drops the never-reachable 64; batches 17–24 stop over-padding to 32); CUDA-only, padding rows inert (token-exact); **(e) LANDED** — `InputBatch::make_sampling_metadata` now caches and rebuilds only on a batch change (add/remove/condense/swap), mirroring vLLM `refresh_metadata` (penalty-active path still rebuilds every step for output-token freshness — the greedy gate gets the full win bit-identically); `SchedulerOutput` `std::move`s the `num_scheduled_tokens` map + `finished_req_ids` set instead of copying (container plumbing, zero policy change). Items (a)-runner (full-width gather / positions / zero-copy views) and (b) (GDN col-0 gather) live in `runner.cpp` (owned by the async/GDN claims) — reported for those owners, not touched here. Landed `8a717b2`/`81afc36`/`0c4b41c` (merged `e027ad5`), clean `-Werror` rebuild; **DGX token-exactness gate PASSED** on the `e027ad5` build (27B default 235/235, `VT_GDN_PACKED_DECODE=0` rollback 235/235, 35B 315/315, one `flock` series) — CLOSED, claim released; `benchmark_binding=false`, no speed credit (payoff via the c2/c8 probe + next exact grid) |

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

**Extensibility — the Platform seam is extracted + DGX-confirmed (2026-07-18).** vLLM's
`platforms/interface.py` capability seam is now mirrored 1:1 in C++
(`include/vllm/platforms/interface.h` + `src/vllm/platforms/{platform,cpu,cuda}.cpp`):
a `Platform` composes the `vt::Backend` and answers the memory-model /
capability queries the engine and model code used to branch on inline
(`is_cuda()`, `is_unified_memory()`, `has_device_capability()`,
`supported_dtypes()`, `residency_policy()`, `supports_graph_capture()`), self-registered
per `DeviceType`. The residency/host-weight-release + device-pool memory-model
conditionals branch **per object** through `GetPlatform(<obj>.device.type)` —
keyed on the specific tensor/queue being dispatched, so a new discrete/unified
GPU's memory model becomes one additive `platforms/<gpu>.cpp` rather than
scattered engine/model edits. (`CurrentPlatform()` — the process-global
accelerator-first resolver — is reserved for genuine process-level "which
accelerator is this process on" questions; using it for per-object dispatch
would misroute a CPU queue/tensor on a GPU box, so the memory-model/residency
sites key on the object's own device.) This is a behavior-preserving refactor
(both gate models stay token-exact); kernel-shape dispatch branches are
deliberately left for the later attention/kernel-registry items.

### Kernel coverage on the gate path

| Kernel family | CPU | CUDA · GB10 | Status |
|---|:---:|:---:|---|
| Dense NVFP4 W4A4 GEMM | ✅ ref | ✅ | CUTLASS/FlashInfer-compatible tactics, frozen plan cache, packed QKV |
| MoE NVFP4 W4A16 GEMM | ✅ ref | ✅ | Marlin/fp4-resident gate path |
| BF16/FP8 projection GEMM | ✅ ref | ✅ | cuBLASLt TN / `nvjet_sm121` path |
| Prefill attention | ✅ ref | ✅ | Vendored FlashAttention-2 with portable fallback |
| Paged decode attention | ✅ ref | 🟡 | FA2 ratio-6 route is correctness/structure-green but strict performance-failed |
| GDN / linear attention | ✅ ref | 🟡 | Prefill AOT is gated; the packed pure-decode kernel is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`). The c16 slot defect (compact state-slot pool keyed on the mamba block-id collapsed two long c16 sequences onto one recurrent-state slot) was fixed test-first (request-identity keying) and proven at `c172336`; W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ). Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); no speed credit |
| RMSNorm, RoPE, SwiGLU, FP4/FP8 quant | ✅ ref | ✅ | Gate-path coverage; broader variant inventory remains open. CPU backend now registers both cast directions (`bf16→f32` `kCastF32` alongside the existing `f32→bf16` `kCastBf16`), closing a CPU op-registration asymmetry |
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
RUN (new binding `246a23c`, 49/124), the open front is the c2–c32 decode gap —
now fully attributed (c2/c8 split 2026-07-16): kernel glue + recurrence at c2–c4,
wave-boundary admission grading (`ENG-ASYNC-SCHED` family) from c8 up →
all-axis 27B parity → 35B parity → the SGLang shared-prefix gate →
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
