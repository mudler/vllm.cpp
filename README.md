# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ — no
Python, PyTorch, or ggml at runtime. The project mirrors vLLM's V1 / Model
Runner V2 architecture and exposes a library, C API, example CLI, and
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** The text paths for
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** pass token-exact greedy correctness
> gates on NVIDIA GB10. Production parity is still open. The **new binding 27B
> result** is the fresh, fully-interleaved exact-grid rerun at `a875397` on the
> production default set (async scheduling ON + vendored Triton GDN decode cubin
> ON + RMSNorm-fast opt-in): **52/124** required axes pass against vLLM v0.25.0
> (gate NO). It **supersedes** `246a23c` (**49/124**) and the older `3f256ab`
> grid (**55/124**), both retained immutable. This is +3 axes AND a structural
> improvement: the **async lever flipped the ITL tails** that dominated the old
> failure mass — c16 p99_itl 1.024, c32 p90_itl 1.020 + p99_itl 1.026 now PASS,
> and the old catastrophic tails (c8 p99_itl 0.56, c32 p90_itl 0.79) are resolved
> at c16/c32 (c8 p99_itl improved to 0.844). **memory (4/4)** and **c1 (20/20)**
> stay clean. The entire remaining failure mass is a nearly-UNIFORM **~1–2%
> decode deficit** (of 72 failing axes, 55 are within 2% of vLLM, only 3 worse
> than 5%; the gate is strict ≥1.0, so 0.98–0.99 counts as fail) — e.g. c16
> throughput 790.95 vs 796.99 (0.9924), c32 throughput 1081.5 vs 1083.7 (0.9979).
> The named lever to reclaim ~1% and flip a batch of c2–c8 near-misses is the
> RMSNorm-fast kernel (c2 preflight: +1.446% tput / −0.887% TPOT); its
> combination-numerics blocker is now **FIXED and the default is flipped ON**
> (2026-07-17, `CLAIM-EW-NORM-ACT-3`): `RmsNormRowFastKernel`'s output is
> BIT-IDENTICAL (0-ulp) to the shipped kernel, so `fast+cubin ≡ 198` by
> construction — the full production default set now passes 27B 235/235 + 35B
> 315/315 while the kernel stays 2.41× faster in isolation. The 52/124 binding was
> measured before this flip; the next binding grid re-measures the production
> default. See [BENCHMARKS](docs/BENCHMARKS.md). **Honest regression:** ours lost **−2.67% total throughput at c16** and
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
> both sides — NOT separate state-I/O) + ~2.6 µs·k norm/quant/act glue, with
> GEMM/MoE/attention at parity. **The norm/quant glue is a kernel-EFFICIENCY gap,
> NOT a fusion gap** (reconciled 2026-07-16): vLLM's denominator does NOT fuse
> rmsnorm+fp4quant — its Inductor body stores bf16 after add+RMSNorm and quantizes
> separately (`cvt_fp16_to_fp4`, 144/win == ours), `fuse_norm_quant=False`; ours
> already matches that structure and already mirrors the one real fusion
> (silu+fp4quant, `VT_FUSE_SILU_QUANT`). The residual is cross-profiler-confounded
> (isolated ours RMSNorm is 6-9 µs, not the 15.5 µs in-trace) with only a modest,
> non-bit-exact efficiency headroom — reassigned to `KERNEL-EW-NORM-ACT`, not a
> fusion. The ~2.06 ms recurrence-tiling lever is now
> **RESOLVED (2026-07-16) via the sanctioned vendored Triton cubin.** The naive
> register-resident hand port PROOF-FAILED (oracle FAIL, ~12% slower); Phase-1
> `cuobjdump` MEASURED why (vLLM's FLA decode cubin holds the register-resident
> tile at REG:205/**0 spills**; the hand port hits REG:255 + **spills** — register
> allocation / codegen, the sanctioned-exception premise), so vLLM's exact decode
> kernel is now vendored as an AOT cubin (`gdn_decode_h48`, 27B-only) behind
> `VT_GDN_PACKED_DECODE_TRITON` (**default ON since 2026-07-16** — MIRROR policy:
> the vendored kernel IS vLLM's exact token-identical FLA kernel and vLLM runs it
> by default, so we do too, joining the sibling GDN Triton kernels; `=0` rolls
> back to the hand kernel in the same binary), **235/235 token-exact** default +
> rollback + memcheck-clean; c16 A/B **+5.48 tok/s (+0.67%), -1.26 ms TPOT, 3/3
> pairs**. The **next binding grid runs the Triton decode path by default**. 35B
> is unaffected — it never selects packed decode (MoE, excluded by the dense-only
> `ShouldUsePackedGdnDecode`) and the launcher guard rejects its `Hv=32` shape
> anyway, so no 35B specialization was added. A follow-up build repair guards the
> Triton launch-counter helper so the default Triton-less CUDA build stays
> `-Werror`-clean. See [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | ❌ FAILED / `GATING` | **NEW BINDING `a875397`: 52/124 pass** (gate NO; fresh interleaved exact-grid rerun on the production default set — async ON + Triton GDN cubin ON + RMSNorm-fast opt-in; supersedes `246a23c`'s 49/124 and `3f256ab`'s 55/124, both retained immutable; ZERO void, 12/12 binding-eligible; evidence `dgx:~/work/vllm.cpp-online-gate/evidence/a875397…`, ratios.json sha256 `4cb89b08…1069`). Per concurrency: mem **4/4**, c1 **20/20**, c2 3, c4 4, c8 5, c16 6, c32 **10**. The **async lever flipped the ITL tails** (c16 p99_itl 1.024, c32 p90_itl 1.020 + p99_itl 1.026, c4 p99_itl 1.89 — all PASS; the old catastrophic tails c8 p99_itl 0.56 / c32 p90_itl 0.79 resolved at c16/c32, c8 p99_itl 0.56→0.844). Remaining failure mass = a nearly-uniform **~1–2% decode deficit** (55/72 fails within 2% of vLLM, only 3 >5%; strict ≥1.0 gate) — c16 tput 790.95 vs 796.99 (0.9924), c32 tput 1081.5 vs 1083.7 (0.9979). Named lever: the RMSNorm-fast kernel (c2 preflight +1.446% tput / −0.887% TPOT) — its combination-numerics blocker is now FIXED via bit-identity and the default is flipped ON (2026-07-17, `CLAIM-EW-NORM-ACT-3`; 27B 235/235 + 35B 315/315 on the full default set, 2.41× isolated); the next binding grid re-measures it. The binary carries slot-fix `c172336`, windowed-load `cb2d310` (memory now PASS), qkvz `45f9e6d` (DGX gates green), packed-decode default. The order-0 packed leaf is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`, `e47b4d6`; eight seals + 8-pair A/B −0.205% ± 0.30 <1σ + 24-window trace, packed GPU-cheaper); no `complete-pass` marker, no packed speed credit | ATTRIBUTED + VALIDATED (era A/B, four rounds + probe3 at `6dd24df`): `3f256ab`'s high-concurrency throughput WAS inflated by the slot-sharing defect (collapsed slots = 1/16th the GDN state traffic ≈ 8–12 ms/step); the honest correct-state c16 floor is ~790–799 vs vLLM 794, gates 235/235×2, no recoverable host cost (≤15 µs/step; O(n) validation landed). Top lever (MEASURED 2026-07-16, correct-state c16 kernel traces): the ~8 ms/step wall gap (c16 TPOT 159.6 vs 167.5) = ~4.65 ms GPU-busy + ~3.25 ms host/idle; the busy part is ~2.06 ms GDN recurrence tiling (ours 21.31 vs vLLM 19.24 ms/step, state r/w fused in-kernel, ~83% vs ~92% of peak BW) + ~2.6 µs·k norm/quant glue (kernel-EFFICIENCY, not fusion), GEMM/MoE/attention at parity. The naive register-resident hand port FAILED its proof (oracle FAIL, −12%); Phase-1 cuobjdump MEASURED the codegen cause (vLLM decode cubin REG:205/0-spill vs hand REG:255+spill) and the recurrence lever is now RESOLVED via the sanctioned vendored Triton cubin (`gdn_decode_h48`, 27B-only, `VT_GDN_PACKED_DECODE_TRITON` **default ON since 2026-07-16** — MIRROR policy, vLLM's exact kernel; `=0` rolls back to the hand kernel), flip gates ALL EIGHT PASS (27B 235/235 default + rollback; 35B 315/315 both arms, inert; memcheck 0-errors), c16 A/B **+5.48 tok/s (+0.67%), -1.26 ms TPOT, 3/3 pairs**; the next binding grid runs the Triton decode path by default. Parallel levers: async-sched W3 runner leaf (+3.25 ms) and decode norm/quant kernel EFFICIENCY (the "fusion" framing is REFUTED — vLLM does not fuse rmsnorm+fp4quant, `fuse_norm_quant=False` + `3f256ab` body-dump stores bf16 + separate `cvt_fp16_to_fp4` (144/win == ours), silu+fp4quant already mirrored; the residual is per-launch kernel EFFICIENCY, reassigned to `KERNEL-EW-NORM-ACT`, now **Phase-1 CONFIRMED, PORTED, and DEFAULT ON via a bit-identical rework (2026-07-17, `CLAIM-EW-NORM-ACT-3`)** ([decode-fast port](.agents/specs/rmsnorm-decode-fast-2026-07-16.md)): same-profiler nsys BOTH sides isolated at the real 27B decode shape M×H=5120 gives ours 8.44-8.53 µs/launch vs vLLM 2.37-2.68 µs = **3.18-3.56×** (honest Δ ≈0.77 ms/step at c2 and c16; the cross-profiler confound was in-situ only), and `RmsNormRowFastKernel` (`VT_RMSNORM_DECODE_FAST`, default OFF) — a 1:1 port of vLLM's own CUDA `fused_add_rms_norm_kernel<bf16,8>` — had its token-exactness FIXED by the 2026-07-17 numerics rework, and the 2026-07-17 c2 preflight A/B (Phase-0 of the authorized binding-grid rerun) delivered the awaited in-situ win: pooled per-request-median TPOT **−0.912 ms (−0.887%)**, total throughput **+1.446%**, 3/3 interleaved pairs fast-better on BOTH axes (binding c2 corpus, w0-discard, one flock; evidence `dgx:~/work/vllm.cpp-online-gate/preflight-rmsnorm-c2-a321d7c…/`). It was briefly flipped ON (`696a991`), ROLLED BACK when `test_qwen27_paged_ENGINE` caught a 27B token-7 near-tie flip (234/235). The rollback's "Inductor-Triton" premise was wrong: the oracle golden is generated with `enforce_eager=True` (pip-vllm:0.24.0), so the oracle rmsnorm is the EAGER csrc `cub::BlockReduce<float,1024>` kernel, NOT Triton. The old fast kernel merely APPROXIMATED cub with a hand warp-shuffle whose reordered f32 sum flipped the near-tie; swapping in the ACTUAL `cub::BlockReduce` reproduces the oracle's exact reduction order. DGX proof: `test_qwen27_paged_ENGINE` **235/235** + `qwen36_paged_engine` **315/315** fast-ON (both rollback arms match), paged_forward 84/84+8/8, CUDA parity 132/132; perf **nsys pure-kernel 2.66 µs median vs shipped 8.66 µs (~3.2×)**, within vLLM's own 2.37-2.68 µs. The c16 in-situ A/B was a NULL within noise (−0.60% / +0.34 ms TPOT), but the 2026-07-17 c2 preflight A/B at the documented c2 target delivered the win (pooled-median TPOT −0.887%, throughput +1.446%, 3/3 pairs both axes), so it was flipped ON after the c2 preflight win, then REVERTED same-day (2026-07-17) by the binding campaign's engine sanity gate: with the FULL default set (async + GDN cubin + RMSNorm-fast) the 27B production stream fails 233/235 at the documented token-7 near-tie — the combined output exactly matches the fixture's `want_emu` (pip-vLLM EAGER-mode) stream, i.e. the pair of individually token-exact kernels lands on the other side of a near-tie vLLM itself decides differently between graphed and eager modes. This is now RESOLVED (2026-07-17, `CLAIM-EW-NORM-ACT-3`): rather than a "Triton-faithful" match, `RmsNormRowFastKernel` was made BIT-IDENTICAL (0-ulp) to OUR shipped `RmsNormRowKernel` — the 235/235 through-stack reference — by reproducing its exact float op sequence (residual add `bf16(f32(x)+f32(res))`, variance in the exact kBlock=256 strided-partial + shared-tree ORDER via a 1024-thread vectorized Pass 1 that stages f32 squares to shared memory, `1.0f/sqrtf`), vectorizing only the element-independent normalize pass. So `fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION: the full production default set passes **27B 235/235 + 35B 315/315** (both rollback arms too), `test_cuda_ops` fast==shipped **0-ulp BIT-EXACT**, and the perf win SURVIVES (isolated 2.41×, in-situ 27B engine-forward RmsNorm 3.68×). Per the parity-enabler policy the **default is flipped ON** (`VT_RMSNORM_DECODE_FAST=0` rolls back); the next binding grid re-measures the production default. The ITL tails are DISCRIMINATED (2026-07-16, CPU-only, [spec](.agents/specs/tail-stall-analysis-2026-07-16.md)): the scheduler is NOT the cause — driving the REAL vLLM sync `Scheduler` and async `AsyncScheduler` through the identical wave-boundary script yields BYTE-IDENTICAL composition (both pack the 2048 budget as 1024 + chunk at the stall step; our C++ schedulers reproduce it exactly, `tests/vllm/v1/test_scheduler_wave.cpp` 44/44). H-B/H-C dead, H-A already dead (`89b329e`). The ~860 vs ~500 ms gap is async-runtime output-timing phasing (W3), un-mirrorable in the sync path (CPU sim of the async driver still budget-packs to 2048); per MIRROR policy the fix IS async-on — the axes are expected to close only under W3-on (already implemented, default-OFF per `89b329e`'s +36 % TTFT finding). The c8+c32 W3-on/off ITL-tail A/B RAN 2026-07-17 and CONFIRMED the mechanism: both anomalies flip under W3-on (c8 p99_itl 0.552→0.897, c32 p90_itl 0.791→1.048; the ~500 ms single-prefill band appears), see the W3 discriminator row below. 35B only after 27B 124/124
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
| Binding gate | **NEW BINDING `a875397`: 52/124** (gate NO; supersedes `246a23c`'s 49/124 and `3f256ab`'s 55/124). Mem (4/4), c1 (20/20) pass; the async default flipped the c16/c32 ITL tails to PASS; the open axes are a nearly-uniform ~1–2% decode deficit at c2–c32 (55/72 fails within 2% of vLLM, only 3 >5%) — the RMSNorm-fast kernel (combination-numerics blocker now FIXED via bit-identity, default flipped ON 2026-07-17, `CLAIM-EW-NORM-ACT-3`) is the named ~1% lever, re-measured by the next binding grid |
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is **`DONE`** — W1D3 **CLOSED on equivalence** (owner `e47b4d6`). The c16 HTTP-500 slot defect (the runner keyed the compact GDN state-slot pool on the mamba block-id, collapsing 2 long c16 sequences onto 1 recurrent-state slot; also latent silent cross-request corruption) was fixed test-first (request-identity keying) and proven at `c172336`. G3 closed over eight sealed components + the 8-pair A/B (−0.205% ± 0.30, <1σ) + the trace attribution (packed GPU-cheaper): no stable regression, no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2) DGX gates closed green at `45f9e6d` (−48 BF16 GEMMs/window confirmed) and is in the `246a23c` binding binary. The authorized exact-grid rerun has now RUN (new binding, 49/124); the era A/B fully attributed the c16/c32 delta (corruption-subsidized bandwidth; pre-fix GDN kernel evidence now contamination-suspect); the fresh correct-state GDN kernel trace vs vLLM is now DONE (2026-07-16): the ~8 ms/step gap is ~4.65 ms busy + ~3.25 ms host/idle, busy = ~2.06 ms GDN recurrence tiling + ~2 ms unfused norm/quant glue (state-I/O fused in-kernel, not a separate op; GEMM/MoE/attn at parity); the naive register-resident port failed its DGX proof (−12%, oracle FAIL; default flipped OFF, opt-in retained); the recurrence-tiling lever is RESOLVED via the sanctioned vendored Triton cubin `gdn_decode_h48`, now **default ON** (`VT_GDN_PACKED_DECODE_TRITON`, MIRROR policy; `=0` rollback), 235/235 default + rollback + memcheck-clean; next: the W3 runner leaf and decode norm/quant kernel efficiency |
| Remaining gap diagnosis | With memory now passing, the failing mass is the **c2–c32 decode-coupled family** (throughput inversely coupled to TPOT/ITL) — now **fully attributed**. The correct-state same-method c2/c8 full-step split ([spec](.agents/specs/c2-c8-attribution-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`) resolves the [lost-lanes](.agents/specs/rescan-lost-lanes-2026-07-16.md) UNATTRIBUTED downgrade: the **c2 gap (+2.43 ms/step) is ENTIRELY GPU-busy** (busy Δ +3.16 = 130% of the gap; idle Δ −0.73 — ours idles LESS than vLLM) — a batch-independent norm/quant/act kernel-glue floor (+2.40: RMSNorm 129×/step +1.74, FP4-quant +0.30, SiLU +0.23, gated +0.14) plus the GDN recurrence (+0.93), GEMM/MoE/attention at parity; the **c8 gap (+7.29) is 38.6% GPU-busy** (glue +2.45, recurrence +1.53, GEMM bundle −1.28 ours-faster) **+ 61.4% wave-boundary stall time** — inside pure-decode waves both engines are ≥99% busy at parity (per-step host window bounded 0.12–0.19 ms/step), so the idle mass is the wave-boundary prefill-event mechanism ([tail spec](.agents/specs/tail-stall-analysis-2026-07-16.md)) now shown to move the c8 MEAN, not just the tails — and the CPU wave discriminator (`tests/vllm/v1/test_scheduler_wave.cpp`) proves the composition is byte-identical both sides, so the magnitude gap is the async depth-2 overlap (W3), not a scheduler divergence. The 07-14 "host-side" label is REFUTED at c2 / RESHAPED at c8; host plumbing (block-table cluster, sampler alloc) is bounded ≤~0.2 ms/step — hygiene, not a c2–c8 lever. Levers: c2–c4 → kernel glue (`KERNEL-EW-NORM-ACT` — RMSNorm Phase-1 confirmed 3.18-3.56×, `RmsNormRowFastKernel` real-cub numerics rework now token-exact 235/235+315/315 and ~3.2× isolated, but c16 in-situ A/B is a NULL so it was flipped ON after the c2 preflight win, then REVERTED same-day (2026-07-17) by the binding campaign's engine sanity gate: with the FULL default set (async + GDN cubin + RMSNorm-fast) the 27B production stream fails 233/235 at the documented token-7 near-tie — the combined output exactly matches the fixture's `want_emu` (pip-vLLM EAGER-mode) stream, i.e. the pair of individually token-exact kernels lands on the other side of a near-tie vLLM itself decides differently between graphed and eager modes. This is now RESOLVED (2026-07-17, `CLAIM-EW-NORM-ACT-3`): rather than a "Triton-faithful" match, `RmsNormRowFastKernel` was made BIT-IDENTICAL (0-ulp) to OUR shipped `RmsNormRowKernel` — the 235/235 through-stack reference — by reproducing its exact float op sequence (residual add `bf16(f32(x)+f32(res))`, variance in the exact kBlock=256 strided-partial + shared-tree ORDER via a 1024-thread vectorized Pass 1 that stages f32 squares to shared memory, `1.0f/sqrtf`), vectorizing only the element-independent normalize pass. So `fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION: the full production default set passes **27B 235/235 + 35B 315/315** (both rollback arms too), `test_cuda_ops` fast==shipped **0-ulp BIT-EXACT**, and the perf win SURVIVES (isolated 2.41×, in-situ 27B engine-forward RmsNorm 3.68×). Per the parity-enabler policy the **default is flipped ON** (`VT_RMSNORM_DECODE_FAST=0` rolls back); the next binding grid re-measures the production default) + recurrence tiling; c8+ → the W3 overlap family (`ENG-ASYNC-SCHED`; the 2026-07-17 discriminator resolved the TTFT/throughput question — the premium is vLLM's own async trade, W3-ON nets positive and **its default flip LANDED 2026-07-17**, `ENG-ASYNC-SCHED` DONE). The prior RMSNorm/generated-partitions residual stays **disproven** as a fusion gap; the c16/c32 regression stays attributed to corruption-subsidized state bandwidth (era A/B + probe3, `6dd24df`) |
| Async/overlap scheduling (`ENG-ASYNC-SCHED` W3) | **DONE — the async-scheduling default is FLIPPED ON (2026-07-17, mirror vLLM); DGX-re-confirmed TOKEN-NEUTRAL.** `VT_ASYNC_RUNNER` now defaults ON (pure `AsyncRunnerFlagIsOn` predicate), so the production engine resolves an `AsyncScheduler` + `max_concurrent_batches=2` (depth-2 `step_with_batch_queue` + async D2H) by default, mirroring `vllm/config/vllm.py:992-1044`; `VT_ASYNC_RUNNER=0` (runner-level) and `VT_ASYNC_SCHED=0` (scheduler-level) are the same-binary rollbacks. The DGX re-confirmation (`dgx:~/work/vllm.cpp-async-flip`, CUTLASS+FA2 hard-verified, one flock) proves the flip changes ZERO tokens — all three async arms bit-identical; on the shipping default **27B 235/235 + 35B 315/315** with the "Asynchronous scheduling is enabled (mcb=2)" log, both rollback arms 235/235+315/315 "disabled". **TTFT means will RISE into vLLM's async envelope BY DESIGN** (+26–31 %, the same depth-2 Little's-law trade vLLM's own async pays for the TPOT/ITL-tail win) — the next binding grid runs async by default and its TTFT readout must NOT be misread as a regression. The whole depth-2 overlap — `AsyncScheduler` placeholder accounting, `step_with_batch_queue`, runner async input-combine + copy-stream sampled-ID D2H, `AsyncOutputPool` persistent buffers, `LoadedEngine` enable-flip — was implemented behind `VT_ASYNC_RUNNER` and is now the default. **The W3 async TTFT-premium discriminator COMPLETED (2026-07-17, `CLAIM-W3-ASYNC-DISC`, [spec](.agents/specs/w3-async-ttft-discriminator-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…`):** a one-flock vLLM v0.25.0 self-A/B (async ON vs `--no-async-scheduling`, arm log-confirmed) + ours W3-on/off at c8/c16/c32 proves the +705 ms TTFT premium is **vLLM's own async behavior** — vLLM async-ON vs its own sync costs **+26/+31/+28 % mean TTFT** at **−0.7 to −0.9 % throughput** and −2.6 to −4.3 ms TPOT, and upstream defaults it ON anyway; ours-W3on matches that pattern within noise (no engine-loop/output-timing divergence exists — the earlier "depth-2 needs a throughput lever" framing is retired as mis-calibrated, async has no throughput win upstream either). W3-on **flips both binding ITL-tail anomalies** (c8 p99_itl 0.552→0.897 in-band; c32 p90_itl 0.791→**1.048**, now beating vLLM) and improves every TPOT/ITL mean ratio +2.3–3.3 pp; vs the production bar (vLLM async-ON) ours-W3on mean TTFT is 0.995/1.042/1.103 (equal-or-better at c16/c32). Axis arithmetic: strict-PASS 14→15/54, +3 flips up vs −2 noise-scale c8-TTFT flips ⇒ **W3-ON nets positive as-is**. **The default flip LANDED 2026-07-17** (`ENG-ASYNC-SCHED` DONE, owner `6ea7856`); the DGX re-confirmation of the flip also caught a SEPARATE pre-existing RMSNorm-fast 27B token regression (see the elementwise-norm row below), rolled back to keep the production default token-exact. |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity; the decode-gap attribution completed 2026-07-16 (c2 gap is GPU-busy kernel glue, not transport) |
| Host-memory repair | **BINDING PASS**: the `LOAD-SAFETENSORS` windowed release (progressive `madvise(MADV_DONTNEED)` on each copied-then-dead source range; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback) is now in the `246a23c` binding binary — all four memory axes PASS (ours peak PSS 24.88 GB vs vLLM 28.18 GB). Direct-to-final-device streaming stays the complete fix (also removes the steady mirror, wanted for 35B) |
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
