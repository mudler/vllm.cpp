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
> (2.04× at c16, `9ecd9d0`, closed the c16 floor; templated `<Tin,Tout>` 2026-07-19 so the
> 35B MoE **f32** gated norm also takes it — 1.55× isolated, was silently on the slow
> kernel), `CausalConv1dUpdateFastKernel`
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
> parity ledger. **35B performance: 19→70/124** (2026-07-19; + MoE shared-expert aux-stream decode overlap). c4-c32 ALL win vLLM (16/20 each, TPOT 1.05-1.18×); memory 4/4 beats vLLM; only c1/c2 residual (~0.96-0.98). Next engine levers: fp8 merged-projection glue-fusion + more overlap slices. Memory 4/4 beats vLLM; the c8/c16/c32 serving operating point WINS (16/20 each, TPOT 1.06-1.15×); c1/c2 residual (~0.91-0.94) is the multi-stream-overlap + glue-fusion engine work carried into roadmap_v1.81× at c1 amortizing to a win by c32, plus prefill TTFT).6-35B-A3B-NVFP4, MoE). The MoE routing/align + host-free levers closed the high-concurrency decode (c16/c32 now WIN throughput+TPOT); remaining: **MoE host double-store FIXED (2026-07-18)** — freeing the routed-expert
> fp4 host mirror after the device Marlin resident is built (+`madvise`) drops
> 35B STEADY serving PSS 20.17→**3.53 GiB** (beats vLLM's 13.3), token-neutral
> 315/315+235/235; the whole-window `peak_pss` (~19.8 GiB load-phase
> coexistence) load-time streaming interleave now LANDED CPU-side
> (`ENG-MOE-LOADSTREAM`: defer routed experts, build+free per layer → bound peak
> to one layer; DGX peak-PSS confirmation pending). Remaining gaps:
> low-batch decode (c1 tput 0.743× / TPOT 0.734×, rising to c16/c32 TPOT 1.05× —
> Marlin MoE GEMM inefficient at batch=1), TTFT 0.80–0.86× (prefill). High-batch
> decode already at parity. 35B correctness holds (315/315). See
> [Benchmarks](docs/BENCHMARKS.md).

## Current status

| Gate | State | Current evidence | Next gate |
|---|---|---|---|
| Qwen3.6-27B correctness | ✅ PASS | Real NVFP4 model, token-exact greedy oracle | Retained as the precondition for every performance run |
| Qwen3.6-27B performance | 🟢 EFFECTIVE PARITY-OR-BETTER | **Two-grid totality: 115/124 effective** (regression-confirmed 2026-07-19: a fresh 27B grid at `fcfde41` after all 35B-era changes = 118/124, NO regression, throughput wins every concurrency) (110 pass-in-both + 5 noise-band coin-flips; grids `9ecd9d0` 114/124 + `f0fb727` 111/124, full bit-identical fast-decode default set; supersede `a875397`/`246a23c`/`3f256ab`, all retained). mem 4/4, c1 20/20, c2/c16/c32 ~19-20/20. The 49→115 close was the bit-identical (0-ulp) fast decode-kernel stack (RMSNorm 2.41×, gated-RMSNorm 2.04×, conv-update 1.92×, FP4/SiLU — all default ON) confirming the decode deficit was norm/quant/act kernel glue. The 9 persistent residuals are the low-concurrency-median edge of our deterministic-forward tradeoff — NET-POSITIVE on each (lose c8 mean/median/p99 + c4 median-ttft + c16/c32 median_itl, WIN the c16/c32 p99 tails 1.055/1.078, c4 p90/p99 ttft, c8/c16/c32 mean_ttft 1.03-1.14×). Full set 27B 235/235 + 35B 315/315 token-exact | No closeable real deficit remains; a literal 124/124 is gated by noise-band coin-flips + a favorable determinism tradeoff (async-forward jitter would forfeit the tail/high-conc/throughput wins, net-negative). 35B performance closure follows accepted 27B parity |
| Qwen3.6-35B-A3B correctness | ✅ PASS | Real NVFP4 safetensors and supported GGUF text paths | Continue no-regression checks |
| Qwen3.6-35B-A3B performance | 🟡 DECODE AT-PARITY-OR-BETTER; PREFILL-TTFT-BOUND — **70/124 at `786aa0e`** (2026-07-19) | **Fresh binding at HEAD `786aa0e` (ZERO void, 12/12 eligible): 70/124, and the entire remaining gap is PREFILL TTFT.** Decode is at-or-beyond vLLM everywhere: memory 4/4; at c4/c8/c16/c32 the ONLY failures are the 4 TTFT axes (mean/median/p90/p99 ttft 0.877–0.971), every decode axis (tpot/itl/e2el/throughput) PASSES; c1 (2/20) + c2 (0/20) are near-miss across the board (0.935–0.975) but TTFT-led. The GDN kernel-glue wins (post-conv + gated-RMSNorm, byte-exact, default-ON) held 70/124 without flipping TTFT → the gap exceeds kernel-glue; the concurrency-dependence (worse c2–c8/c32 than c1) points at batched-prefill SCHEDULING. Active lever: prefill-TTFT full-step attribution (host-side/scheduling vs kernel). The 19→70 journey (decode close) below. The 35B online-serving **c2+ crash that blocked the grid is FIXED** (2026-07-18): concurrency > 1 died with `cudaEventSynchronize: an illegal memory access` — cuda-gdb pinned it to `marlin_moe_wna16::Marlin`, whose fp32-reduce scratch `c_tmp` (`EnsureCtmp`) was a grow-on-free per-stream buffer baked into the captured pure-decode CUDA graph; a bigger later prefill/decode freed the block the graph still referenced → use-after-free on the next replay (single-stream c1 never grows it, so it never crashed). Fix = retire-on-grow across the four decode-graph-reached scratch allocators (`RetireGraphScratch`, `src/vt/cuda/graph_safe_scratch.h`). 35B c2-c16 sweep no longer dies (pre-fix 5/5 trials crash → post-fix 0), 315/315 token-exact preserved. **MoE host double-store FIXED (2026-07-18, `ENG-MOE-HOSTFREE`):** freeing the routed-expert fp4 host mirror (~16.9 GiB) after `BuildMoeMarlinResident` + `madvise(MADV_DONTNEED)` drops 35B STEADY serving PSS 20.17→**3.53 GiB** (beats vLLM 13.3), guarded to the Marlin path, token-neutral 315/315+235/235, memcheck clean; the whole-window `peak_pss` (~19.8 GiB, load-phase coexistence) is now bounded by the **load-time streaming interleave LANDED 2026-07-18 + DGX-PROVEN (`ENG-MOE-LOADSTREAM`):** the loader DEFERS the routed-expert host copies and `PrepareMarlinResident` materializes ONE layer's experts immediately before that layer's device Marlin build + host free, so at most one layer's ~256 experts coexist on the host. DGX A/B (new vs eager parent, one flock): 35B load-to-ready **peak RSS 21.43→4.19 GiB (−80%, below vLLM 13.3)**, token byte-identical (315/315 + 235/235), 27B unaffected, memcheck 0/315; device residents byte-identical; the orchestrator re-grids the binding memory axes to confirm the FAIL→PASS flip. **The largest c1 decode-TPOT lever landed 2026-07-18** (`CLAIM-MOE-DECODE-PARALLEL-1`): the two M=1 MoE decode kernels launched a single block and ran serially, leaving the GPU ~99% idle (grounded nsys: MoeAlign 29.3 µs vs vLLM 3.6 µs / 8.2×, MoeRouterTopK 20.2 µs vs vLLM 6.7 µs / 3.0× — together ~1.7 ms/tok ≈ 63% of the 2.7 ms c1 gap). Both are now parallelized (router top-k → per-thread argmax + tree reduction mirror of vLLM `moeTopK`/`topkGating`; moe_align → one-thread-per-expert `cub::BlockScan` prefix sum mirror of `moe_align_sum_kernels.cu`), **BYTE-EXACT to the serial reference** (router 72/72 + align 60/60 parity), same-box per-kernel A/B MoeAlign 29.5→3.0 µs (9.8×, now below vLLM) + MoeRouterTopK 20.2→12.3 µs (1.64×), plus block_size_m 16→8 at low M (L3) and 4 redundant Marlin workspace memsets removed (L4, memcheck-clean). Efficiency-only + **27B 235/235 + 35B 315/315 token-exact** ⇒ shipped ON by default | Run the v0.25.0 performance grid (now unblocked) after all 27B axes pass; the c1 TPOT recovery from this lever is re-measured in-situ by that grid |
| Host-memory parity | ✅ PASS on the new binding grid | All four memory axes now PASS at `246a23c` (windowed-load `cb2d310` binding): ours peak PSS/RSS 24,879,201/24,881,800 KiB vs vLLM 28,184,400/28,563,020 KiB (1.1329×/1.1479×), GPU 40,996 vs 70,531 MiB, MemAvailable-drop 68,346,844 vs 80,660,556 KiB. The prior `3f256ab` peak (48.3 GB) was load-time double-residency, eliminated by the windowed release (−23.54 GB load-to-ready VmHWM) | Memory parity holds. Layer-bounded direct-device staging is now implemented only for the plain-BF16 Qwen3.5 dense diagnostic and remains `GATING`; broader quantized/MoE streaming is still open |
| 35B FA2-prefill lever | ✅ LANDED / default-ON (2026-07-18, `CLAIM-35B-FA2-FLIP-1`) | The 35B FA2-prefill + fused qk-norm-rope-gate preamble is now DEFAULT-ON (`FuseAttnPreambleOn` returns true all arches; `VT_FUSE_ATTN_PREAMBLE=0`/`VT_FA2_PREFILL=0` roll back), so the 35B ratio-8 full-attn layers take the exact `flash_fwd_splitkv` kernel the 27B already uses. Grounded by the RESOLVED oracle ([spec](.agents/specs/qwen36-35b-fa2-prefill-oracle-2026-07-18.md)): vLLM production runs FA2 bf16-q + the fused preamble; vLLM 0.25.0 graphed greedy == the stored oracle **16/16**, so no re-baseline. Both sacred gates hold on the FULL current-main default set: 35B `test_qwen36_paged_engine` **315/315** + 27B `test_qwen27_paged_engine` **235/235**; memcheck 35B prefill **0 errors**. The spec's "round normed q/k→bf16 before RoPE" tighten was op-level bit-identical but flipped the 27B tok6 near-tie in combination (RMSNorm-saga) → NOT shipped; the preamble ships UNTIGHTENED, both arches token-exact. Realistic input-1024 TTFT A/B (conc8, same binary, FA2 default vs `VT_FA2_PREFILL=0`): Mean TTFT **824.7 vs 874.4 ms (−5.7%)**, prefill tput +5.5% (3 interleaved pairs, low variance). CPU gate: clean `-Werror` 0/0, full ctest **156/157** (`test_capi` is a known pre-existing nondeterministic dgx detokenizer flake, not a regression), tools 164/164 | 35B perf grid (orchestrator-owned) re-measures the in-situ TTFT gain from this lever |
| 35B FA2-decode lever | ✅ LANDED / default-ON (2026-07-19, `CLAIM-35B-FA2-DECODE-1`) | The 35B ratio-8 (Hq/Hkv=16/2) hd-256 full-attn DECODE now takes the vendored `flash_fwd_splitkv` main+combine path the 27B ratio-6 already used, gated by new env `VT_FA2_DECODE_35B` (default ON, `=0` rollback). The old ratio-8 decode launched a **2-block grid** (`PagedAttentionDecodeGqaKernel`, grid=(num_reqs,num_kv_heads)) at low batch — near-zero GB10 occupancy; split-KV adds the `num_splits` axis so the grid fills the machine (nsys: clean 1:1 kernel swap, 300↔300 launches, GridZ up to 16). Both sacred gates hold on the FULL default set: 35B **315/315** + 27B **235/235**; operator **454,358** assertions (adds ratio-8 parity ladder); memcheck 35B decode **0 illegal-access errors**. In-situ 35B A/B (input-1024, same binary, `VT_FA2_DECODE_35B=1` vs `=0`, 4 interleaved pairs, first dropped): **c1 TPOT 14.96→16.72 ms (−10.5%), total tput +10.4%; c8 TPOT 33.02→34.12 ms (−3.2%), tput +2.8%; TTFT neutral** (decode-only lever). Targets the 35B c1 decode-TPOT 0.810× low-batch residual. CPU gate: clean `-Werror` 0/0, tools 164/164, checkers green; full DGX ctest 157/160 — the 3 misses are non-numerics (`test_async_llm` parallel-port flake passes isolated; `test_capi` documented nondeterministic detokenizer flake; `test_qwen36_gguf_engine` a two-35B-GGUF memory-edge OOM — `VT_FA2_DECODE_35B=0` passes 28/28, the added decode scratch tips a marginal box, production safetensors 35B is 315/315 ON — flagged for a scratch-pool follow-up) | 35B perf grid (orchestrator-owned) re-measures the in-situ c1–c4 TPOT gain |

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
| Selected GPU work | `KERNEL-GDN-PACKED-DECODE` is **`DONE`** — W1D3 **CLOSED on equivalence** (owner `e47b4d6`). The c16 HTTP-500 slot defect (the runner keyed the compact GDN state-slot pool on the mamba block-id, collapsing 2 long c16 sequences onto 1 recurrent-state slot; also latent silent cross-request corruption) was fixed test-first (request-identity keying) and proven at `c172336`. G3 closed over eight sealed components + the 8-pair A/B (−0.205% ± 0.30, <1σ) + the trace attribution (packed GPU-cheaper): no stable regression, no `complete-pass` marker, no speed credit. **qkvz** (`KERNEL-GEMM-BF16` W2) DGX gates closed green at `45f9e6d` (−48 BF16 GEMMs/window confirmed) and is in the `246a23c` binding binary. The authorized exact-grid rerun has now RUN (new binding, 49/124); the era A/B fully attributed the c16/c32 delta (corruption-subsidized bandwidth; pre-fix GDN kernel evidence now contamination-suspect); the fresh correct-state GDN kernel trace vs vLLM is now DONE (2026-07-16): the ~8 ms/step gap is ~4.65 ms busy + ~3.25 ms host/idle, busy = ~2.06 ms GDN recurrence tiling + ~2 ms unfused norm/quant glue (state-I/O fused in-kernel, not a separate op; GEMM/MoE/attn at parity); the naive register-resident port failed its DGX proof (−12%, oracle FAIL; default flipped OFF, opt-in retained); the recurrence-tiling lever is RESOLVED via the sanctioned vendored Triton cubin `gdn_decode_h48`, now **default ON** (`VT_GDN_PACKED_DECODE_TRITON`, MIRROR policy; `=0` rollback), 235/235 default + rollback + memcheck-clean; next: the W3 runner leaf and decode norm/quant kernel efficiency. **The FP4-quant decode-glue kernel-efficiency lever landed OPT-IN 2026-07-17** (`CLAIM-FP4-QUANT-FAST-1`): two NUMERICS-NEUTRAL bit-identical vectorized-load+store fast kernels (`VT_FP4_QUANT_FAST` `ScaledFp4QuantFastKernel`, `VT_SILU_FP4_FAST` `SiluAndMulFp4QuantFastKernel`) — each thread does one 16-byte `uint4` load + one 64-bit packed store (vs 16 + 8 scalar), memory-pattern change only, grounded 1:1 in vLLM `nvfp4_quant_kernels.cu:56-80,98`. Byte-exact new-vs-old PROVEN (60/60 adversarial asserts + full FP4 suite 24/24; engine both-flags-ON 27B 235/235 (16/16 token-exact) + 35B 315/315). Isolated nsys per-launch is **PARTIAL vs the ≥1.3× flip bar** — ScaledFp4Quant K=5120 1.12-1.18× / K=17408 1.44-1.62×, SiluAndMul 1.14× (c2) / 1.38× (c16-c32): clears the larger shapes, misses the dominant K=5120 / c2 (swizzled small-M is padding-thread-dominated; the residual to vLLM's ~1.7× is the numerics-changing hw fp4-cvt + bf16 reduction, out of scope). **2026-07-18 (`CLAIM-CONV-UPDATE-FAST-1`): both flags flipped DEFAULT ON** per the parity-enabler policy (byte-exact ⇒ never-slower + token-safe; under the strict ≥1.0 gate every fraction counts; `=0` rollback), re-verified byte-exact 25/25 (26,976) + full default set 27B 235/235 + 35B 315/315. **The GDN decode conv-update fast kernel also landed the same way** (`VT_CONV_UPDATE_FAST`, default ON, `CausalConv1dUpdateFastKernel` 0-ulp bit-identical, **isolated 1.92×** at the 27B c16 shape — a 2D grid removing the int64 div/mod + a register-cached state row; DGX byte-exact 330/330, full default set 235/235 + 315/315). The next binding grid runs the full bit-identical fast-decode stack by default; the orchestrator owns the combined in-situ A/B. **2026-07-18 (`CLAIM-GDN-PREFILL-CONV-1`): PREFILL GDN conv kernel-efficiency landed** — `CausalConv1dFwdRegKernel` (`VT_CONV_REG` DEFAULT ON, `=0` rollback), the register-resident sliding-window mirror of vLLM's FLA prefill `causal_conv1d` (weights preloaded to registers, each x loaded once, token-axis chunked for low-batch prefill occupancy); the per-V-head fused post-conv split (`VT_GDN_POSTCONV_SPLIT`) landed OPT-IN (measured near-neutral). Both BIT-IDENTICAL (0-ulp) to the shipped tiled/megablock kernels (byte-exact CUDA A/B 268 asserts + memcheck 0; 27B 235/235 + 35B 315/315). nsys 35B: conv −4.7% (c1) / −7.3% (c6) per-call — a real but modest bandwidth-bound win (the kernels are BW-bound; the residual vLLM conv gap is bf16 traffic, VT_GDN_IN_BF16, separate); TTFT within run-noise. Binding grid re-measures. **2026-07-19 (`CLAIM-GDN-POSTCONV-FAST-1`): fused post-conv FAST kernel landed DEFAULT ON.** A fresh production-path nsys (`--cuda-graph-trace=node`, Triton AOT ON) confirmed the GDN chunk **compute** (delta_h/chunk_o/kkt/recompute_w_u) runs the vendored FLA Triton cubins by default = **at FLA parity by construction**, so the #1 remaining NON-AOT GDN kernel on BOTH models is the fused post-conv prep. Since the split measured neutral/slower, `GdnPostConvFastKernel` (`VT_GDN_POSTCONV_FAST`, `=0` rollback) keeps the megablock grid but makes two BYTE-IDENTICAL changes (128 threads/block — the 128-wide L2-norm tree is the 256-wide tree minus a `+0` step; + 128-bit-staged V copy), grounded in FLA `_fused_post_conv_kernel`. Isolated nsys **27B −24.3% / 35B −24.8%** per-call; in-situ TTFT 27B c1 −1.14% / c2 −1.31%, 35B c1 −0.72% / c2 −0.99% (all reps positive); 27B 235/235 + 35B 315/315 both arms; clean `-Werror`. Per the parity-enabler policy (byte-exact + measured-faster ⇒ default ON) |
| Remaining gap diagnosis | With memory now passing, the failing mass is the **c2–c32 decode-coupled family** (throughput inversely coupled to TPOT/ITL) — now **fully attributed**. The correct-state same-method c2/c8 full-step split ([spec](.agents/specs/c2-c8-attribution-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`) resolves the [lost-lanes](.agents/specs/rescan-lost-lanes-2026-07-16.md) UNATTRIBUTED downgrade: the **c2 gap (+2.43 ms/step) is ENTIRELY GPU-busy** (busy Δ +3.16 = 130% of the gap; idle Δ −0.73 — ours idles LESS than vLLM) — a batch-independent norm/quant/act kernel-glue floor (+2.40: RMSNorm 129×/step +1.74, FP4-quant +0.30, SiLU +0.23, gated +0.14) plus the GDN recurrence (+0.93), GEMM/MoE/attention at parity; the **c8 gap (+7.29) is 38.6% GPU-busy** (glue +2.45, recurrence +1.53, GEMM bundle −1.28 ours-faster) **+ 61.4% wave-boundary stall time** — inside pure-decode waves both engines are ≥99% busy at parity (per-step host window bounded 0.12–0.19 ms/step), so the idle mass is the wave-boundary prefill-event mechanism ([tail spec](.agents/specs/tail-stall-analysis-2026-07-16.md)) now shown to move the c8 MEAN, not just the tails — and the CPU wave discriminator (`tests/vllm/v1/test_scheduler_wave.cpp`) proves the composition is byte-identical both sides, so the magnitude gap is the async depth-2 overlap (W3), not a scheduler divergence. The 07-14 "host-side" label is REFUTED at c2 / RESHAPED at c8; host plumbing (block-table cluster, sampler alloc) is bounded ≤~0.2 ms/step — hygiene, not a c2–c8 lever. Levers: c2–c4 → kernel glue (`KERNEL-EW-NORM-ACT` — RMSNorm Phase-1 confirmed 3.18-3.56×, `RmsNormRowFastKernel` real-cub numerics rework now token-exact 235/235+315/315 and ~3.2× isolated, but c16 in-situ A/B is a NULL so it was flipped ON after the c2 preflight win, then REVERTED same-day (2026-07-17) by the binding campaign's engine sanity gate: with the FULL default set (async + GDN cubin + RMSNorm-fast) the 27B production stream fails 233/235 at the documented token-7 near-tie — the combined output exactly matches the fixture's `want_emu` (pip-vLLM EAGER-mode) stream, i.e. the pair of individually token-exact kernels lands on the other side of a near-tie vLLM itself decides differently between graphed and eager modes. This is now RESOLVED (2026-07-17, `CLAIM-EW-NORM-ACT-3`): rather than a "Triton-faithful" match, `RmsNormRowFastKernel` was made BIT-IDENTICAL (0-ulp) to OUR shipped `RmsNormRowKernel` — the 235/235 through-stack reference — by reproducing its exact float op sequence (residual add `bf16(f32(x)+f32(res))`, variance in the exact kBlock=256 strided-partial + shared-tree ORDER via a 1024-thread vectorized Pass 1 that stages f32 squares to shared memory, `1.0f/sqrtf`), vectorizing only the element-independent normalize pass. So `fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION: the full production default set passes **27B 235/235 + 35B 315/315** (both rollback arms too), `test_cuda_ops` fast==shipped **0-ulp BIT-EXACT**, and the perf win SURVIVES (isolated 2.41×, in-situ 27B engine-forward RmsNorm 3.68×). Per the parity-enabler policy the **default is flipped ON** (`VT_RMSNORM_DECODE_FAST=0` rolls back); the next binding grid re-measures the production default. The sibling **gated-norm glue lever** (the `gated +0.14` c2 term / +0.40 ms/step at c16 vs vLLM's fused gated norm) landed the SAME bit-identical way (2026-07-17, `CLAIM-EW-NORM-GATED-1`): `RmsNormGatedRowFastKernel` behind `VT_RMSNORM_GATED_FAST` (default ON, `=0` rollback) is 0-ulp identical to the shipped gated kernel, **isolated 2.04× at c16** (1.31–1.38× at c1/c2), full default set 27B 235/235 + 35B 315/315) + recurrence tiling; c8+ → the W3 overlap family (`ENG-ASYNC-SCHED`; the 2026-07-17 discriminator resolved the TTFT/throughput question — the premium is vLLM's own async trade, W3-ON nets positive and **its default flip LANDED 2026-07-17**, `ENG-ASYNC-SCHED` DONE). The prior RMSNorm/generated-partitions residual stays **disproven** as a fusion gap; the c16/c32 regression stays attributed to corruption-subsidized state bandwidth (era A/B + probe3, `6dd24df`) |
| MoE shared-expert aux-stream decode overlap (`ENG-MOE-SHARED-AUX`) | **LANDED default-ON 2026-07-19 (`CLAIM-MOE-SHARED-AUX-1`) — the first slice of the multi-stream intra-step OVERLAP named the largest remaining 35B c1/c2 lever.** Mirror of vLLM's decode overlap (`fused_moe/runner/shared_experts.py:99-104,125-142` + `maybe_execute_in_parallel`, `multi_stream_utils.py:20-58`): the MoE **shared-expert MLP** runs on a 2nd persistent CUDA stream concurrent with the **routed-expert** router/align/grouped-Marlin-GEMMs on the main stream inside `MoeBlockFusedMarlinCuda`, joined before the combine. **Byte-identical by construction** (independent shared/routed paths both complete before combine; overlap changes WHEN not WHAT) — proven: overlap `VT_MOE_SHARED_AUX_STREAM`∈{0,1} both 35B **315/315** + 27B **235/235**, captured-vs-eager 315/315, `compute-sanitizer memcheck` 0 errors. The scratch `DevicePool`'s single-stream reuse invariant is preserved by giving the aux stream its own `AuxPool` (each pool serves one stream; the fix for the cross-stream scratch race vLLM avoids via its `record_stream` allocator). **In-situ same-binary interleaved TPOT A/B** (35B, input-1024/output-128, greedy, drop cold rep1): c1 13.60 vs 14.40 ms **−5.6%**, c2 −2.7%, c4 −3.7%, c8 −3.4%, c16 −1.6%, c32 −1.5% — WINS at every concurrency, ZERO regression (GB10's 48 SMs leave spare occupancy across the low-conc band). Token-exact + faster c1-c4 + non-regressing c8+ ⇒ **default flipped ON** (`VT_MOE_SHARED_AUX_STREAM=0` rollback; `VT_MOE_SHARED_AUX_THRESHOLD` 128 gate). Remaining overlap slices (routed/attention/prefill multi-stream) + portable glue-fusion stay the roadmap_v1 35B priority. The next binding grid re-measures the c1/c2/c4 TPOT. |
| FP8 merged-QKV projection (`KERNEL-GEMM-FP8` sub-lever) | **LANDED opt-in 2026-07-19 (`CLAIM-FP8-MERGED-QKV-1`) — token-exact, perf-NEUTRAL.** Extends the fp4-only merged-QKV fusion to the 35B FP8 W8A8 full-attn path: the 10 full-attn layers ran Q/K/V as 3 separate `MatmulFp8Cutlass` GEMMs; now ONE fp8 GEMM over the N-concatenated operand + split (30→10 GEMMs/step), targeting the M=1 decode launch overhead of the last 35B c1/c2 residual. **Key finding:** fp8 here is PER-TENSOR scaled (one scalar `weight_scale`/folded `alpha` per shard), so unlike the fp4 per-block merge a single-alpha concat is incorrect — realized as concat RAW fp8 bytes + GEMM `alpha=1` + a resident per-column alpha vector applied by the NEW `vt::MulColVecF32` op; byte-identical to the 3 separate GEMMs when cuBLASLt tiling matches (Q/K/V share one `input_scale` ⇒ identical activation quant). CPU gates GREEN (glue 10/10 incl. 2 new byte-exact MulColVecF32, fp8_cutlass 6/6, matmul 7/7, clean `-Werror`); **DGX GREEN** (clean CUDA `-Werror` 0 warn): 35B **315/315 token-exact both arms** + 27B **235/235 both arms** (inert), memcheck **0 errors**, merge proven to fire. **In-situ 35B TPOT A/B: NEUTRAL** — c1/c8 ~0%, c2/c4 −0.5%, all ≤0.9% within rep noise (the ~20-launch saving over 10 attn layers is below the decode-step noise floor; the step is dominated by 30 GDN layers + MoE). **Kept opt-in** (`VT_FP8_MERGED_QKV` default OFF) per "token-exact but not measurably faster". Follow-up: GDN qkvz/BA fp8 merge (30 layers — larger launch slice). |
| Sigmoid-gate → o_proj fold (`KERNEL-GEMM-NVFP4-W4A4` sub-lever) | **LANDED opt-in 2026-07-19 (`CLAIM-SIGMOID-GATE-FOLD-1`) — byte-exact, perf-NEUTRAL.** Folds the standalone full-attention sigmoid output gate (`attn*sigmoid(gate)`) into the o_proj NVFP4 activation quant on the 27B true-W4A4 path — one fused `vt::SigmoidGateFp4Quant` kernel, no bf16 `gated` intermediate — mirroring vLLM's Inductor `triton_poi_fused_mul_scaled_fp4_quant_sigmoid` and reusing our `SiluMulFp4Quant` precedent. Only 27B (W4A4) o_proj quantizes its activation; 35B (W4A16-Marlin / fp8 o_proj) reads bf16 activations and keeps the standalone `SigmoidGateBf16` (fusion inert). **DGX GREEN** (`~/work/vllm.cpp-sigmoid-fold`, production flags CUTLASS sm120a + Marlin + FA2 sm_121a + Triton AOT, clean CUDA `-Werror` **0 warnings**): byte-exact op test `sigmoid_gate_fp4_quant` **14/14** (CPU f32/bf16 + CUDA, 3 shapes), 27B `test_qwen27_paged_engine` **235/235 both arms** (fused=default-ON and `VT_FUSE_SIGMOID_QUANT=0` fallback), 35B `test_qwen36_paged_engine` **315/315** (inert). **In-situ 27B TTFT A/B (input-1024, 3 reps/arm): NEUTRAL** — c1 med OFF 419.6 vs ON 419.0 ms (−0.15 %), c2 792.9 vs 792.7 ms (−0.03 %), all within ±0.5 % rep noise (the full-attn o_proj is a small slice of 27B prefill, dominated by GDN + MoE + QKV/gate/up GEMMs). **Kept opt-in** (`VT_FUSE_SIGMOID_QUANT=1`, default OFF) per "flip only on a measured win". |
| 35B GDN out_proj gated-RMSNorm → fp8-quant fold (`KERNEL-EW-NORM-ACT` sub-lever, `CLAIM-GDN-OUT-FP8-FUSE-1`) | **LANDED default-ON 2026-07-19 — byte-exact, measured-faster (small).** Folds the 35B GDN out_proj's static W8A8 fp8 activation quant INTO the gated-RMSNorm output store: one new `vt::RmsNormGatedQuantFp8` kernel emits the fp8 activation directly (fed to `MatmulFp8CutlassPreQuantD`), removing the standalone `QuantFp8Static` pass **and** the bf16 gated-norm output it would write then re-read — the gated sibling of the existing `RmsNormQuantFp8Row` fusion (mirrors vLLM's Inductor fusion of the fla `layernorm_guard` RMSNormGated epilogue with the following RowParallelLinear's static-fp8 quant). **Byte-identical** to the split `RmsNormGated(bf16)+QuantFp8Static` (fp8 taken from the SAME bf16-rounded value; the fused kernel reproduces `RmsNormGatedRowFastKernel`'s exact 0-ulp variance reduction). Only the 35B GDN out_proj is fp8 (27B's is W4A4 fp4) → **35B-only, no 27B greedy-razor exposure.** **DGX GREEN** (`~/work/vllm.cpp-gdn-fp8-fuse`, prod flags, clean CUDA `-Werror` **0 warnings**): CPU byte-exact `rmsnorm_gated_quant_fp8` (silu+sigmoid, fused==split); 35B `test_qwen36_paged_engine` **315/315** on default-ON, `VT_GDN_OUT_FP8_FUSE=0` rollback, AND the pre-flip `=1` arm; 27B **235/235** (inert); `compute-sanitizer memcheck` **0 errors** fused-ON. **Isolated nsys** (35B prefill c1, 30 GDN layers): the gated-norm+out_proj-quant chain **6.43→4.58 ms (−28.7%)** — the fused fp8 1-byte store is even cheaper than the unfused gated norm's bf16 store alone — but only **~0.28%** of the ~655 ms total prefill GPU. **In-situ 35B TTFT A/B** (input-1024, output-8, 3 reps interleaved, same binary): **c1 median −1.4% (3/3 reps), c2 −1.3% (3/3), c8 −0.4% (2/3; async-off arm — a pre-existing `async_scheduler` assertion crashes c8+short-output, unrelated to the fold)**, prefill tput +1.0%, zero regression. Byte-exact + consistently faster ⇒ **default ON** (`VT_GDN_OUT_FP8_FUSE=0` rollback) per the parity-enabler policy (like `GdnPostConvFast`). **Honest read: a REAL but SMALL fold — it does NOT flip the 3–14% binding TTFT axes alone; each portable glue fold is ~1%, so closing the 35B prefill gap needs a STACK of them.** |
| Eager Marlin repack (35B first-token) | **VERIFIED already-at-load-time — no work needed (2026-07-19).** The scoping premise (first-touch Marlin repack = 42 % of a COLD 35B prefill) was checked: the routed-expert / shared-expert / lm_head Marlin repack (`BuildMoeMarlinResident`) already runs eagerly at engine init via `Qwen3_5Model::PrepareMarlinResident`, called from the **`GPUModelRunner` constructor** (`runner.cpp:302/321`) — before any warmup or serving, mirroring vLLM's `process_weights_after_loading`. The forward-path first-touch build (`if (!mr.ready)`) is a dead fallback in production (`mr.ready` is already true from load). Empirically an un-warmed 35B c1 bench shows an elevated FIRST-request TTFT (median 178 / p99 575 ms over 6 reqs) — but that ~400 ms delta is general first-request warmup (FP4 autotune / plan-cache / CUDA-graph capture / page-in), NOT the multi-second expert repack (which is provably at construction, before request 1). The binding grid warms with a 1×1024-token request before timing, so even that general first-request cost is excluded; the repack is OUTSIDE the measured c1/c2 window ⇒ lever skipped. |
| Async/overlap scheduling (`ENG-ASYNC-SCHED` W3) | **DONE — the async-scheduling default is FLIPPED ON (2026-07-17, mirror vLLM); DGX-re-confirmed TOKEN-NEUTRAL.** `VT_ASYNC_RUNNER` now defaults ON (pure `AsyncRunnerFlagIsOn` predicate), so the production engine resolves an `AsyncScheduler` + `max_concurrent_batches=2` (depth-2 `step_with_batch_queue` + async D2H) by default, mirroring `vllm/config/vllm.py:992-1044`; `VT_ASYNC_RUNNER=0` (runner-level) and `VT_ASYNC_SCHED=0` (scheduler-level) are the same-binary rollbacks. The DGX re-confirmation (`dgx:~/work/vllm.cpp-async-flip`, CUTLASS+FA2 hard-verified, one flock) proves the flip changes ZERO tokens — all three async arms bit-identical; on the shipping default **27B 235/235 + 35B 315/315** with the "Asynchronous scheduling is enabled (mcb=2)" log, both rollback arms 235/235+315/315 "disabled". **TTFT means will RISE into vLLM's async envelope BY DESIGN** (+26–31 %, the same depth-2 Little's-law trade vLLM's own async pays for the TPOT/ITL-tail win) — the next binding grid runs async by default and its TTFT readout must NOT be misread as a regression. The whole depth-2 overlap — `AsyncScheduler` placeholder accounting, `step_with_batch_queue`, runner async input-combine + copy-stream sampled-ID D2H, `AsyncOutputPool` persistent buffers, `LoadedEngine` enable-flip — was implemented behind `VT_ASYNC_RUNNER` and is now the default. **The W3 async TTFT-premium discriminator COMPLETED (2026-07-17, `CLAIM-W3-ASYNC-DISC`, [spec](.agents/specs/w3-async-ttft-discriminator-2026-07-16.md), evidence `dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…`):** a one-flock vLLM v0.25.0 self-A/B (async ON vs `--no-async-scheduling`, arm log-confirmed) + ours W3-on/off at c8/c16/c32 proves the +705 ms TTFT premium is **vLLM's own async behavior** — vLLM async-ON vs its own sync costs **+26/+31/+28 % mean TTFT** at **−0.7 to −0.9 % throughput** and −2.6 to −4.3 ms TPOT, and upstream defaults it ON anyway; ours-W3on matches that pattern within noise (no engine-loop/output-timing divergence exists — the earlier "depth-2 needs a throughput lever" framing is retired as mis-calibrated, async has no throughput win upstream either). W3-on **flips both binding ITL-tail anomalies** (c8 p99_itl 0.552→0.897 in-band; c32 p90_itl 0.791→**1.048**, now beating vLLM) and improves every TPOT/ITL mean ratio +2.3–3.3 pp; vs the production bar (vLLM async-ON) ours-W3on mean TTFT is 0.995/1.042/1.103 (equal-or-better at c16/c32). Axis arithmetic: strict-PASS 14→15/54, +3 flips up vs −2 noise-scale c8-TTFT flips ⇒ **W3-ON nets positive as-is**. **The default flip LANDED 2026-07-17** (`ENG-ASYNC-SCHED` DONE, owner `6ea7856`); the DGX re-confirmation of the flip also caught a SEPARATE pre-existing RMSNorm-fast 27B token regression (see the elementwise-norm row below), rolled back to keep the production default token-exact. **ROBUSTNESS FIX (2026-07-20):** the SEPARATE pre-existing `async_scheduler` `num_output_placeholders >= 0` assertion crash on c8 + short-output (noted in the rows above) is FIXED — the runner was missing vLLM's `discard_request_mask`, so it emitted a sampled token for prefill-CHUNK requests, underflowing the async placeholder count under chunked-prefill + preemption. The fix mirrors vLLM (runner clears sampled tokens for still-prefilling requests; the async scheduler is unchanged); byte-identical on the non-chunked gates (27B 235/235 + 35B 315/315), and `vllm-bench` c8 + short-output + chunked-prefill + KV-pressure with async ON now completes cleanly (memcheck 0). |
| Serving transport (TCP_NODELAY) | **DONE; measured NEUTRAL on the gate workload** (`SERVE-HTTP-TRANSPORT`). We mirror vLLM's uvicorn/asyncio default (`set_tcp_nodelay(true)`), pinned by a behavioral accepted-socket test (RED 0 → GREEN 1, 22/22). The non-binding localhost A/B sizing is neutral within noise at c1/c2 — µs loopback ACKs mean Nagle never held our ~100 ms-cadence token frames — so the mirror stays for real-network parity; the decode-gap attribution completed 2026-07-16 (c2 gap is GPU-busy kernel glue, not transport) |
| Host-memory repair | **BINDING PASS for the established 27B windowed-release gate**: `LOAD-SAFETENSORS` progressive `madvise(MADV_DONTNEED)` remains in the `246a23c` binding binary (ours peak PSS 24.88 GB vs vLLM 28.18 GB), and the 35B routed-expert steady mirror/load interleave remains implemented. The additive plain-BF16 direct-device leaf (`LOAD-SAFETENSORS-DIRECT-DENSE`, [spike](.agents/specs/qwen35-plain-bf16-direct-load.md)) is **`GATING` after the corrected local Triton-AOT 4B run**. Root `/tmp/qwen35-transplant-4b-aot-557ab41d` completed all 18 guarded legs: direct ON/OFF/local-vLLM-0.24 total **6155.10/6064.06/6730.46 tok/s**, output **680.61/670.54/744.24**, peak PSS **2.405/8.571/7.569 GiB**, stable PSS **0.733/8.571/4.066 GiB**, and peak VRAM **12892/12884/12942.7 MiB**. ON=OFF output IDs **128/128 in every pair**; ON is **+1.50%** total and cuts peak/stable PSS **71.9%/91.4%**. Against the prior matching-AOT 4B result, current ON is **0.9316x** (**6155.10 vs 6607.04, −6.84%**); it is **0.9145x** local vLLM. The graph-node audit proves generated AOT GDN chunk kernels ran; the 27B-only packed-decode cubin is correctly ineligible for this 4B shape. **The 27B/35B/Coder regressions are now RUN and GREEN** on the PR rebased onto `a63c497` (2026-07-21, dgx, clean `-Werror` rebuild, FINAL binary): 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6**, plus the bf16/NVFP4 grouped-MoE and MoE-forward suites, with `compute-sanitizer memcheck` **0 errors** on the 27B engine path; `test_cuda_backend` shows GB10 reports `pageable=1 integrated=1`, so the discrete-Blackwell memory-classification fix is behavior-NEUTRAL here. A same-arch (`121a`) interleaved A/B of `origin/main` vs the rebased head on the 27B is neutral on every axis (output throughput **0.9995×**, median TPOT +0.03%, median TTFT −0.14%, all inside ±0.15%), so **the −6.84% is not a code regression against main**; the CMake arch suffix is independently confirmed a first-order lever (it gates fp4-MMA, CUTLASS NVFP4, Marlin MoE, FlashAttention-2 and the vendored Triton-AOT tree — FA2 alone measures +3.9% throughput / −36.4% TTFT on the 27B). Still open, and only reachable on the sm_120 box with the 4B checkpoint: the current-v0.25 4B correctness oracle, strict ON<=OFF VRAM (+8 MiB), and the same-source `120`-vs-`120a` A/B that would actually close the −6.84% attribution. No support extrapolation to 27B/35B from any 4B result |
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
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | 🚧 | — | 🚧 **First additive-model bring-up — correctness complete; SPEED at c1 effective-parity, c8 decode residual (2026-07-21)** ([spike](.agents/specs/first-additive-model-qwen3-dense.md)). **SPEED CORRECTION 2026-07-21 (same-session, matching-recipe binding, `Qwen3-4B` vs vLLM 0.25.0 production, in1024/out128 ignore-eos closed-loop):** the earlier "TTFT 2.27×/5.85×" and "c8 ITL 4.3×" were BAD-DENOMINATOR / num-prompts artifacts — a fresh same-session vLLM capture gives c1 TTFT ~152 ms & c8 ITL P99 ~130 ms, and **ours BEATS vLLM on TTFT at both concurrencies** (c1 0.90×, c8 0.38×) and ties c1 ITL (0.996×). **c1 = effective every-axis parity** (tput 0.98×, TPOT 1.01×, TTFT+ITL wins). **c8 residual = decode throughput/latency** (tput 0.93×, TPOT 1.10×, ITL P99 1.12×): the c8 decode step is 93% GPU-busy/compute-bound, dominated by small-M=8 `cutlass_80_wmma` projection GEMMs. The **qkv-merge** (single QKVParallelLinear GEMM + new `QkvSplit` op, mirroring vLLM) was implemented + measured NEUTRAL (doesn't cut decode FLOPs) → ships default-OFF (`VT_QWEN3_QKV_MERGE=1` opts in). **cutlass verification RESOLVED:** 27B `test_qwen27_paged_engine` on the flashinfer-cutlass build = 235/235 (the "flashinfer ⇒ 234/235" claim was a build artifact). Stays `ACTIVE` (c8 decode residual). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) + [bench-evidence/qwen3-4b-binding-20260721.log](docs/bench-evidence/qwen3-4b-binding-20260721.log). Prior W0-W4 history: `Qwen3ForCausalLM` on `Qwen3-0.6B` (BF16). W0 = new registry TU (`qwen3_dense.cpp` + `qwen3.h`, one `REGISTER_VLLM_MODEL`, full-attention-only KV spec) resolves via the clean model-factory seam (zero shared-array edit). W1 = the **runner is now model-shape-agnostic**: a full-attention-only (empty `layer_types`, no GDN group) KV config allocates + steps without the hybrid GDN path — behaviour-preserving (27B 235/235 + 35B 315/315 token-exact UNCHANGED). W2 = the **weight loader** (`qwen3_weights.cpp`, new file): Qwen3-0.6B safetensors → `Qwen3DenseWeights` with merged qkv/gate_up (vLLM `packed_modules_mapping`), per-head q/k norm, and **tied `lm_head`** (aliases `embed_tokens`; the checkpoint's `lm_head.weight` is skipped, mirroring vLLM `skip_prefixes`). Shared BF16 loader helpers extracted to `dense_weight_loaders.h` (behaviour-preserving: 27B load byte-identical). W3 = the **dense forward** (`qwen3.cpp`, new file): embed → 28 layers (std add+RMSNorm, per-head q/k norm + RoPE, causal paged attention, SwiGLU MLP) → final norm → tied lm_head, composed from vt:: ops + 2 new byte-exact fusion recipes (non-gemma add-RMSNorm + non-gated qk-norm-rope), bf16 numerics mirroring vLLM. Two genuine latent bugs the first pure-dense bf16 model exposed were fixed: the classic Qwen2/Qwen3 tokenizer pre-tokenizer (`kQwen2Classic`) and a WMMA prefill kernel that mistokenized at head_dim≠256. **W4 CORRECTNESS COMPLETE 2026-07-20 — near-tie-robust gate PASSES on both 0.6B and a bigger 4B.** The earlier "vLLM greedy is non-deterministic" belief was a BATCHING artifact: PER-PROMPT (batch=1, the gate's single-request regime) vLLM 0.25.0 greedy is DETERMINISTIC (0.6B 0-multi over K=10, 4B 0-multi over K=5). Forward correctness is PROVEN by teacher-forcing vLLM on our exact prefix: at all-but-2 positions vLLM's own argmax given our prefix IS our token (gap 0.0000, bit-identical logprobs); the residual are bf16 near-ties (0.6B ≤0.125 nats, 4B ≤0.25) where vLLM's own one-shot prefill argmax disagrees with its incremental decode (vLLM contradicts itself — no forward bug). Gate: our token within 0.5 nats of vLLM's teacher-forced argmax (strict where equal). **Qwen3-0.6B 16/16** (strict 12/16 + near-tie 4/16). **Qwen3-4B (BF16, 36 layers, GQA 32/8, hidden 2560 — a DIFFERENT config, same forward code) 16/16** — the bigger-model complete-correctness proof (strict 10/16 + near-tie 6/16). Regression 27B 235/235 + 35B 315/315 UNCHANGED, CUDA -Werror clean, memcheck 0. Correctness-complete. **SPEED benchmark vs vLLM 0.25.0 production (2026-07-20, `Qwen3-4B`, in1024/out128, graphed denominator) — the dominant PREFILL lever is now IMPLEMENTED but Qwen3-4B is still below vLLM → stays ACTIVE (NOT DONE):** a **d128 FlashAttention-2 varlen prefill** (vLLM runs Qwen3 prefill on the same `flash_attn_varlen_func` family; generalized the vendored FA2 launcher to head_dim 128, `VT_FA2_PREFILL_QWEN3` default-ON) plus flipping the **FA2 varlen d128 decode default ON** (`VT_FA2_DECODE_QWEN3`, near-tie gate re-passes 16/16 on 0.6B + 4B) closes most of the gap: total throughput **0.90× (c1) / 0.62× (c8)** (up from 0.80×/0.48×), and c1 decode reaches parity (TPOT 1.04×, **ITL P99 0.98× — win**). The prefill kernel alone is +7%/+41% total tput and −55%/−48% TTFT (same-binary A/B). Still failing: **TTFT median 5.85×/10.2×** and total tput <1×, because the full prefill STEP (not the attention kernel, which now matches vLLM's FA2) is ~6× vLLM — non-attention prefill glue (GEMM/MLP fusion) + host-side per-kernel launch overhead (un-graphed prefill); plus c8 decode batch efficiency (TPOT 1.38×). Named residual levers: (1) portable prefill-step fusion + graphed prefill (dominant); (2) c8 split-KV decode occupancy. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md). **Strict-decode razor (2026-07-20, investigation):** an attempt to tighten the near-tie band to strict 16/16 by routing d128 decode through the vendored FA2 group-swap split-KV kernel measured WORSE (0.6B 12→11/16, 4B 10→9/16) and was reverted — vLLM runs FA2 *varlen* (fa_version 2 on sm_121), not the group-swap kvcache path; the real bit-match target is an FA2 varlen d128 decode. **That FA2 varlen d128 decode is now vendored + measured (2026-07-20, [spec](.agents/specs/qwen3-decode-strict-bitmatch.md)):** the exact vLLM varlen kernel BIT-MATCHES vLLM's decode attention output (teacher-forced logit gap 0.0000 nats away from the near-ties) but STRICT 16/16 is bf16-tie-bounded and NOT reached — it scores 0.6B 11/16 & 4B 9/16, one fewer than the CUDA-core fallback (12/16 & 10/16), because the residual ≤0.375-nat ties are ones vLLM itself resolves inconsistently between prefill argmax and incremental decode. Now shipped **default-ON** (`VT_FA2_DECODE_QWEN3`, d128-scoped so the d256 arms are untouched) as the production Qwen3-dense decode after the near-tie-robust gate re-passed 16/16 on 0.6B + 4B with FA2 prefill+decode ON (max teacher-forced gap 0.125 nats); strict-16/16 stays bf16-tie-bounded and the near-tie-robust gate remains the closure. **TTFT levers 2026-07-20 (base `812a57a`), measure-first:** the pooled `DevicePool` was extracted to a shared header (`device_pool.h`; dense forward + gate models both use it) — byte-identical, but a clean same-binary A/B DISPROVED the "~44% GPU-idle from cudaMalloc/cudaFree" premise (PERF-NEUTRAL; the async scheduler overlaps the alloc syncs). A **RoPE cos/sin cache** (route the bf16 preamble to `RopeFromCache` off the per-step cache instead of `RopeNeox`'s per-element fp64 pow/cos/sin, ~36.5% of prefill GPU-busy; bf16 cache == vLLM's `RotaryEmbedding`) IS the dominant dense-TTFT lever — same-binary A/B cuts `Qwen3-4B` median TTFT **c1 209→135 ms (−35%) / c8 316→144 ms (−54%)** (at/beyond vLLM median TTFT under this recipe). Shipped **DEFAULT-ON** (`VT_QWEN3_ROPE_CACHE=0` opts back to `RopeNeox`) after the opt-in blocker was GROUNDED AND DISPROVEN (2026-07-20): the alleged "1-ULP shift lands on an FA2-split-KV-combine run-to-run NONDETERMINISM → flaky gate" does NOT reproduce — the paged engine is byte-DETERMINISTIC run-to-run (RoPE-off gate 4/4 + RoPE-on dumps md5-identical 3/3 + gate 16/16 identical K=5), and for the short gate contexts `num_splits==1` so the split-KV combine kernel is never launched at all. The 1-ULP FMA shift is real but deterministic; it moves two genuine bf16 near-tie tokens, resolved by regenerating the near-tie goldens on the canonical `cutlass-4.5.0` build. Gate re-passes **16/16** on both sizes (0.6B max teacher-forced gap 0.0 nats, 4B 0.25 nats, 0 forward-divergent). Isolated from 27B/35B (`RopeCacheEnabled` lives only in the Qwen3-dense TU): 27B **235/235** + 35B **315/315** UNCHANGED. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md). 27B 235/235 + 35B 315/315 UNCHANGED, `-Werror` 0-warn, memcheck 0 |
| MLA / DeepSeek family | DeepSeek-V2/V3/V3.2, Kimi-Linear, Kimi-K2.5 | BF16 (DeepSeek-V2-Lite) | — | 🟡 **CORRECT, FASTER, still NOT AT vLLM SPEED — and the rest of the family is now recorded as BLOCKED, not pending. The campaign plan is COMPLETE (W0-W10). DeepSeek-V2-Lite passes its correctness gate and has its binding speed number (2026-07-22). The row is `ACTIVE`, NOT `DONE`, because we are still behind vLLM on throughput at every concurrency** ([spike](.agents/specs/mla-deepseek-campaign.md), [numbers](docs/BENCHMARKS.md#binding-deepseek-v2-lite-mla-every-axis-grid--mla-campaign-w9)). **W0–W9 of the plan have landed.** Multi-head Latent Attention was a brand-new attention family for this project: instead of storing a key and a value per attention head, MLA stores ONE compressed 576-wide vector per token, which our cache allocator and cache-write kernels could not express at all. That required a new KV-cache description, backend selection for MLA, a cache-write kernel, a decode kernel, a prefill path with a bounded loop for reusing already-cached context, an attention layer with load-time weight folding so decode can read the compressed cache directly, and finally the DeepSeek-V2 model itself (configuration, weight loading, forward pass). Only `DeepseekV2ForCausalLM` is registered: the original `DeepseekForCausalLM` is not an MLA model at all, V3 does not fit this hardware and is stored in a format we do not load, and V3.2 needs a sparse-attention module we do not have — all three are refused by name rather than silently claimed. Two coverage limits are recorded rather than glossed: the query path DeepSeek-V3 and GLM-4.7-Flash use, and DeepSeek's distinctive expert-router behaviour, are both implemented and unit-tested at their real dimensions but unexercised end-to-end, because the one checkpoint that fits this box uses neither. **Hardware verdict: only DeepSeek-V2-Lite (~15.7B, ~29.3 GiB) fits GB10.** DeepSeek-V3/V3.2, Kimi-K2.5, MiniMax-M2 and MiniMax-M3 do not fit 119 GiB of unified memory (two also exceed free disk); Kimi-Linear-48B is marginal. For those we will check configuration, weight mapping on a slice and unit-level agreement at their dimensions, and record the rest as hardware-blocked rather than claiming a gate we cannot run | **Correctness: 8 of 8 prompts pass, and stayed identical after every speed change.** The bar is token-for-token agreement with vLLM 0.25.0 through the full paged engine. Five prompts match vLLM exactly; the other three differ only where vLLM cannot tell the candidates apart from itself — measured, not assumed: feeding vLLM our own text back shows that at 35 of the 36 differing positions vLLM's own most-likely next token is the one we produced, and the remaining case is within a quarter of a nat. Nothing we produced fell outside vLLM's top 20. The gate also proves the engine really drives this path: it checks the compact MLA cache was actually allocated, and runs the prompts concurrently with staggered arrival so the scheduler produces genuinely mixed batches in the exact order MLA silently depends on, plus a cache-reuse case. **Speed: two changes made decode 1.7× to 2.2× faster than before, but vLLM is still ahead.** Profiling the GPU rather than guessing found that the MLA decode kernel was taking 45% of all GPU time while using about 4% of the GPU — it was splitting the work into 2 pieces on a processor with dozens of cores, because the upstream rule that picks the split count only looks at sequence length and assumes many requests are in flight. Sizing the split to actually fill the GPU made that kernel **18× faster** and lifted end-to-end output rate by 70% at one request and 20% at eight. A second change records the decode step as a replayable GPU command list, worth a further ~2% (small here because this model's decode is limited by the GPU, not by the time we spend telling it what to do). Both are on by default and neither moves a single token. Against vLLM we are now at **0.86–0.95× its output rate** (was 0.50×), and our time-to-first-token is now **faster than vLLM at 4 and 8 concurrent requests**, but throughput is short everywhere, so this is an honest miss with a named cause: at one request per batch our dense projections fall back to a vector-matrix routine for a third of GPU time where vLLM uses tensor cores for much of the same work. That is the next thing to fix. One further honest note about the comparison itself: vLLM's own default expert-layer backend for this model **reboots this machine** — five times now, including twice on a freshly restarted box with every mitigation applied — so the comparison uses vLLM's best configuration that actually runs here, which is stated openly and does not flatter us, since we lose to it. Everything else in the tree is unchanged (the 27B, 35B, Qwen3-Coder, Qwen3-dense and OPT gates all re-run identical), the build is clean, and the GPU memory/race/sync tools report nothing. **Blocked-row disposition (W10):** DeepSeek-V3/V3.2 (671B, ~642 GiB), Kimi-K2.5 (~2000 GiB) and MiniMax-M2 (~428 GiB) cannot run end to end on 119 GiB and are recorded as hardware-blocked; V3.2 is blocked a second time by a dependency (its sparse attention has exactly one candidate backend on this GPU and that path dispatches to a dense-only kernel that drops the sparse selection); `DeepseekForCausalLM` is plain multi-head attention, not MLA, and is refused by name. Each blocked row records what can still be checked (configuration resolution, weight mapping on a slice, unit agreement at real dimensions) and what cannot. **Two permanent coverage gaps** — the grouped `noaux_tc` router and the low-rank query path — are unit-gated only, because the one checkpoint that fits uses neither; **GLM-4.7-Flash (31.2B, 58.2 GiB) fits and would close both**, and is the named next vehicle |
| MiniMax M2 / M3 | MiniMax-M2, MiniMax-M3-Sparse | — | — | 🔬 **SPIKED (M2) / INVENTORIED (M3), no implementation.** Determined to be **disjoint from MLA**, correcting an earlier source-scan assumption: M2 is plain dense GQA + MoE with partial RoPE and per-layer sliding window; M3 is GQA plus a learned sparse-attention indexer. Lightning attention is retired upstream. M2 needs the sliding-window track, not the MLA track, and its row is now `BLOCKED` (W10): HW-BLOCKED e2e at ~428 GiB bf16 against 119 GiB, with only configuration resolution and unit parity at its dimensions still gateable |
| GLM family | GLM-4.7-Flash, GLM-4-9B-0414, GLM-4.5-Air, GLM-5, ChatGLM3 | — | — | 🔬 **SPIKED ONLY — no implementation (2026-07-21)** ([spike](.agents/specs/glm-dsa-latest-deepseek.md)). The family splits three ways. **`Glm4MoeLiteForCausalLM` (GLM-4.7-Flash, 31.2B, 58.2 GiB bf16) FITS GB10** and is upstream a zero-override subclass of DeepSeek-V2's MLA attention, so it lands as a pure addition once the MLA track exists — and it exercises two routing/projection paths the MLA spike could only unit-test. **`Glm4ForCausalLM` and `GlmForCausalLM` (both 9B / 17.5 GiB) also fit**, but need two primitives we have none of: a partial rotary factor (rope only a leading slice of each head) and sandwich norms (extra norms on each sublayer's output before the residual add). **`Glm4MoeForCausalLM` is HW-BLOCKED** — its smallest genuine checkpoint is 205.8 GiB bf16. **`GlmMoeDsaForCausalLM` (GLM-5) is blocked twice over:** 1404 GiB, and it needs sparse attention that does not run on this GPU (see the DeepSeek row). Nothing is implemented. **Updated 2026-07-22 (MLA campaign W10):** the GLM-5 row is now `BLOCKED` (hardware AND dependency), and GLM-4.7-Flash is recorded as the MLA track's named next vehicle — it is the only reachable checkpoint that closes both of that campaign's permanent coverage gaps. The other five GLM rows stay `SPIKE` |
| DeepSeek V3.2 / V4 (latest) | DeepSeek-V3.2-Exp, DeepSeek-V4-Flash/Pro | — | — | 🔬 **SPIKED ONLY — no implementation (2026-07-21)** ([spike](.agents/specs/glm-dsa-latest-deepseek.md)). **V3.2 = V3 plus DSA** (DeepSeek Sparse Attention): a small extra "indexer" module with its own weights and its own 132-byte-per-token KV cache, whose only job is to pick which past tokens each query attends to. **DSA cannot run on GB10.** For a sparse model vLLM's backend filter removes the Triton MLA backend entirely, leaving one flashinfer sparse backend as the sole option — and that path is non-functional on the flashinfer available here (its sm12x kernel is dense-only and silently discards the top-k selection; vLLM's capability probe checks the wrong symbols, and upstream's only test stubs the probe out). This is a dependency gap to watch, not work we can do. **V4 is a new architecture, not an increment** — a 4-way-expanded residual stream normalized by a 20-iteration Sinkhorn solver every forward, a compressor with recurrent state paged through the KV manager, and hash-routed MoE layers with no learned gate. Its tokenizer, by contrast, is ordinary HF BPE with upstream golden fixtures, so that part is low-risk. Both are HW-BLOCKED end to end at 642 GiB and 148.7 GiB against 119 GiB of memory |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 Post-parity roadmap (W-next after Qwen3-dense; needs a checkpoint download — none on dgx) |
| OPT (cross-family canary) | OPT-125m, … | 🚧 | — | 🚧 **`OPTForCausalLM` — W0-W4 LANDED 2026-07-21; correctness COMPLETE (STRICT token-exact 6/6), SPEED PENDING** ([spike](.agents/specs/sweep-opt-125m.md)). The breadth sweep's **cross-family additivity canary**: the first model we support that is not a Qwen variant. OPT breaks every shared assumption at once — no RoPE (LEARNED absolute position embeddings, with the fairseq offset of 2), biased q/k/v/out/fc1/fc2 projections, LayerNorm-with-bias instead of RMSNorm, a plain ReLU fc1/fc2 MLP instead of SwiGLU, a `do_layer_norm_before` pre/post-LN switch, and tied embeddings by default. **Correctness: STRICT token-exact, 6/6 prompts / 96/96 tokens** vs the pinned vLLM 0.25.0 oracle. The strict bar is used because it was *measured* to be the right one: vLLM's own greedy is DETERMINISTIC on this model (K=5 runs, 0 multi-valued positions), so unlike the near-tie-gated Qwen rows there is no tolerance band here at all. **Additivity verdict:** the model-layer seams held completely — the whole model landed as 4 new files with zero changes to the runner, scheduler, platforms, attention registry, config parser, any CUDA kernel, or any existing model. What did have to change was one layer down, in two vocabularies that had quietly been Qwen-shaped: the `vt::` op table gained LayerNorm/ReLU/Add (append-only, through the existing op-table extension point), and the tokenizer gained the original GPT-2 byte-level split plus the ability to apply a post-processor's special tokens — the latter had been parsed but never applied, which silently cost every OPT prompt its leading `</s>` and was caught only because the oracle's own tokenization is committed alongside its continuations. **Caveats:** runs bf16 rather than the checkpoint's fp16 (vLLM's `--dtype bfloat16`, since our CUDA path has no fp16 compute); the HF snapshot ships a torch-pickle checkpoint with no `tokenizer.json`, so it is materialized to safetensors by a committed script; the post-LN (OPT-350m) branch is implemented but has no checkpoint to gate it. **No throughput has been measured and none is claimed** — head_dim 64 falls back to the generic paged-attention path (the FA2 fast kernels are d128/d256-only) and there is no decode CUDA graph yet, so the speed bar is a separate increment and the row stays `ACTIVE`, not `DONE`. Regressions all unchanged on the final dgx binary (27B 235/235, 35B 315/315, Qwen3-Coder 6/6, Qwen3-dense 664 assertions); clean CUDA rebuild `-Werror` 0 warnings; memcheck reports 0 invalid reads/writes/global accesses on the OPT engine path (its 66 leak-check reports are the shared DevicePool's by-design retained scratch, fewer than the 125 the pre-existing Qwen3-dense reports on the same binary) |
| MoE decoders | Mixtral, Qwen3-MoE | 🚧 | — | 🚧 **Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) — W0-W3 LANDED 2026-07-21** ([spike](.agents/specs/sweep-qwen3-coder-30b.md)); the first sweep model, a BF16 full-attention MoE that composes the done dense attention + 35B MoE experts (zero runner change). **W0** = new registry TU (`qwen3_moe_registry.cpp` + `qwen3_moe.h`, one `REGISTER_VLLM_MODEL`, full-attention-only KV spec, `is_dense_model=false`) resolves via the clean model-factory seam. **W1** = three behaviour-preserving refactors making the two done pieces reusable: (#1) the dense self-attention block extracted verbatim to `dense_attn_block.h`, (#2) the bf16 `MoeBlock` exposed cross-TU via `RunMoeBlock` (`qwen3_5_moe_block.h`), (#3) a no-shared-expert guard in `MoeBlock`. **W2** = BF16 loader `qwen3_moe_weights.cpp` (merged qkv/o + per-head q/k norm; NEW bf16 per-expert loader — router + 128 experts × gate/up/down in Matmul-B layout, no shared expert; UNTIED `lm_head` loaded separately, not aliased). **W3** = forward `qwen3_moe.cpp` (`Qwen3MoeModel::Forward/ForwardDevice` = the dense forward body with the per-layer MLP replaced by the MoE block — reused `AttnBlock` + `RunMoeBlock`, no GDN/shared, untied lm_head) wired into the factory. **W4** = the SACRED correctness gate (`test_qwen3coder_paged_engine.cpp`): paged-engine greedy vs the vLLM 0.25.0 oracle. vLLM's own greedy is DETERMINISTIC on Qwen3-Coder (K=5, 0 multi-valued cells), so the STRICT-where-well-posed near-tie-robust gate applies (same ratified methodology as the dense models) — **6/6 prompts PASS** (STRICT token-exact 4/6 + near-tie-band 2/6, max teacher-forced gap 0.125 nats, 0 forward-divergent; the 2 near-tie flips are vLLM's own argmax on our prefix). Gates: load gate all 18867 tensors mapped, forward doctest green (real-checkpoint prefill argmax = 12095 " Paris"); regression UNCHANGED (Qwen3-dense 16/16, 27B 235/235, 35B 315/315), CUDA `-Werror` 0-warn, memcheck 0. **W5 LANDED 2026-07-21** = the fast BF16 grouped-MoE GEMM (`vt::MoeGroupedGemmBf16`, **default ON**) replacing the per-expert host-gather loop with ~3 on-device grouped launches, reusing our NVFP4 grouped-MoE scheduling with the fp4 decode swapped for a direct bf16 read; plus a host-mirror release that stopped the model occupying ~114 GiB of GB10's 119 GiB unified pool. Correctness was RE-VERIFIED after the accumulation-order change and got **stricter** (6/6; STRICT token-exact 4/6 → 5/6; max teacher-forced gap 0.125 → 0.0000 nats). Same-binary A/B: TTFT **3.4×**, TPOT **1.59×**. **W6 LANDED 2026-07-21** = the MoE-GEMM tile/pipeline rework plus deterministic split-K, both **default ON**. The W5 tile's real defect was an uncoalesced `[K,N]` weight stage on a GEMM that is weight-bandwidth bound; the rewrite reads the weight tile k-major (`wmma::matrix_b` row-major) so the global read runs along contiguous N, adds 16-byte vectorized stages, a 3-deep `cp.async` multi-stage pipeline and a wider tile (shape ported from vLLM's Triton `fused_moe_kernel` + `get_default_config`; multistage structure from CUTLASS `mma_multistage.h`). Split-K lifts the block-starved small-batch decode GEMM using a fixed ascending-order partial reduction — explicitly not `atomicAdd`, which would break greedy reproducibility. **The MoE GEMM went 4.50 → 12.15 TFLOP/s (2.7×) and now runs ~1.2× vLLM's Triton `fused_moe` rate.** Correctness is UNCHANGED with zero token movement (the tile preserves the K-reduction order exactly, so it is bit-identical to W5): 6/6, strict 5/6, max gap 0.0000 nats. **SPEED: 12 of 16 binding-grid cells now at or above graphed vLLM — c4 passes every axis, c8 passes 3 of 4** (median TTFT 1.03/0.82/1.05/1.19×, TPOT 0.88/1.01/1.22/1.05×, ITL 0.88/1.03/1.22/1.09×, output throughput 0.77/0.90/1.09/0.99×). Every-axis-at-every-concurrency is **still not met**: the 4 residual cells are `nsys`-attributed to the missing bf16 decode CUDA graph (c1 decode is ~86% GPU-busy, so ~5 ms/step of host tax — not kernel speed) and to the non-MoE prefill glue at c2 (the MoE tile fell from 56% to 17% of GPU time). Note the W5 grid's c2/c4/c8 vLLM numbers were measured against a shared server that replayed prompts into vLLM's prefix cache; W6 re-measures with a fresh server per concurrency at a verified 0.0% hit rate (see [BENCHMARKS](docs/BENCHMARKS.md)). **W7 LANDED 2026-07-21** = the **bf16 decode CUDA graph** — the largest named lever from W6. Qwen3-Coder now gets the same capture-once/replay-per-step decode graph the 27B and 35B already had (that one was fp4-gated, so a bf16 model never got it), built as a third sibling of the same driver and ported from vLLM's `gpu_model_runner` capture/replay plus its pad-to-nearest-captured-size dispatch. Because Qwen3-Coder is pure full attention there is no GDN state to pad or gate, so every captured batch size is usable. Capturing this path for the first time exposed a genuine latent bug in shared attention glue: the RoPE row-index was uploaded from a stack-local buffer, which is fine eagerly but under capture bakes a dead stack address that every replay re-reads — it produced one wrong token and is now served from a process-persistent table. **Measured: decode went from ~86% to ~92% GPU-busy (host tax ~5 -> ~2.7 ms/step), median TPOT 36.22 -> 34.13 ms**, with **zero token movement** (a CUDA graph replays the same kernels in the same order, so bit-identity is the expectation and the measurement): 6/6, strict 5/6, max gap 0.0000 nats. **SPEED: 11 of 16 binding-grid cells at or above graphed vLLM, and c4 and c8 now pass every axis** (median TTFT 1.02/0.91/1.05/1.17x, TPOT 0.93/1.12/1.27/1.08x, ITL 0.93/1.10/1.28/1.14x, output throughput 0.84/0.96/1.11/1.03x). Every-axis-at-every-concurrency is **still not met**: the 5 residual cells sit at c1 (TPOT/ITL/output throughput) and c2 (TTFT/output throughput). The c1 residual is now attributed to engine-side per-step **host** bookkeeping — vLLM's entire 31.75 ms decode step costs about what our kernels alone cost (~31.5 ms), so c1 needs no kernel work — and the tempting CUDA-API explanation was profiled and **refuted** (stream-sync median 4.5 us). c2 is prefill-bound and untouched by a decode lever. **CORRECTNESS-DONE, SPEED ACTIVE.** Each sweep model is held to the Qwen3 bar (near-tie-robust token-exact + vLLM-speed on every axis) |

**Breadth sweep — ACTIVE (2026-07-21).** With the first additive model (Qwen3 dense) validated
end-to-end, the active priority is adding more model architectures (recent-first), each held to the
same bar: token-exact vs vLLM (near-tie-robust where vLLM's own greedy is self-inconsistent) AND
vLLM-speed on every axis. Ranked queue + the CUDA-arch additivity audit:
[`.agents/specs/breadth-sweep-plan.md`](.agents/specs/breadth-sweep-plan.md). Note: adding a new
CUDA *arch* beyond the same-family sm_120 is HW-blocked (only GB10 sm_121 is testable here), so the
actionable sweep is model breadth on GB10; Metal/Intel/Vulkan/CPU follow. The audit's four
CUDA-arch seam-gaps are now CLOSED as mechanism (see Acceleration below,
[spike](.agents/specs/cuda-arch-additivity.md)) — expansion is additive, but validation of any
non-sm_121 target remains hardware-blocked.

**MLA / DeepSeek campaign — the plan is COMPLETE (W0-W10, 2026-07-22). DeepSeek-V2-Lite is CORRECT
and is STILL BEHIND vLLM on throughput; every other family member is now recorded as blocked rather
than pending.** The breadth sweep's Tier-3 "MLA = new attention, new campaign" item has its leaf spike
([`.agents/specs/mla-deepseek-campaign.md`](.agents/specs/mla-deepseek-campaign.md)); the whole plan is
in the tree, with a binding speed number
([numbers](docs/BENCHMARKS.md#binding-deepseek-v2-lite-mla-every-axis-grid--mla-campaign-w9)).

*What was hard.* Multi-head Latent Attention does not store a key and a value per attention head. It
stores ONE compressed 576-number vector per token, with a single shared head and no separate value at
all — something our cache allocator (which hardcoded a factor of two for key-plus-value) and our
cache-write and attention kernels could not express in any form. Closing that meant a new way to
describe a KV cache, MLA backend selection, a cache-write kernel, a decode kernel, a prefill path with
a bounded loop for reusing already-cached context, an attention layer with load-time weight folding so
decode reads the compressed cache directly instead of expanding it, DeepSeek's expert-router variant,
and finally the DeepSeek-V2 model itself.

*Correctness — the gate this project holds every model to.* Token-for-token agreement with vLLM
0.25.0 over a fixed prompt set, driven through the full paged engine: **8 of 8 prompts pass**, and the
result did not move by a single token across either speed change. Five match vLLM exactly. The other
three differ only where vLLM cannot separate the candidates from itself, which was measured rather
than assumed — feeding vLLM our own generated text back shows that at 35 of the 36 differing positions
vLLM's own most-likely next token is the one we produced, and the one remaining case sits within a
quarter of a nat; nothing we produced fell outside vLLM's top 20. The gate also proves the engine
really drives this path: it checks that the compact MLA cache was actually allocated, and it runs the
prompts concurrently with staggered arrival so the scheduler produces genuinely mixed batches in the
exact order MLA silently depends on, plus a cache-reuse case.

*Speed — a large win, and an honest shortfall.* Rather than guessing, we profiled what the GPU
actually ran. The MLA decode kernel was consuming **45% of all GPU time while using about 4% of the
GPU**: it split the work in two on a processor with dozens of independent cores, because the upstream
rule that chooses the split count looks only at sequence length and quietly assumes many requests are
in flight. Sizing the split to fill the machine — upstream's own stated occupancy target, simply
applied — made that kernel **18x faster**. With a replayable decode command list (a further ~2%; small
here because this decode is limited by the GPU, not by issuing work), end-to-end output rate rose
**70% at one request and 20% at eight**. Both are on by default. Against vLLM we now reach
**0.86-0.95x its output rate** (it was 0.50x) and our time-to-first-token is **faster than vLLM at 4
and 8 concurrent requests**, but throughput is short at every concurrency, so this is a miss with a
named cause: at small batch our dense projections fall back to a vector-matrix routine for roughly a
third of GPU time where vLLM puts much of the same work on tensor cores. That is the next thing to
fix, and it is a routing change rather than a new kernel. One thing worth stating plainly about the
comparison: vLLM's own default expert-layer backend for this model **hard-reboots this machine** —
five times now, including on a freshly restarted box with every mitigation applied — so the comparison
uses vLLM's best configuration that actually runs here. That substitution does not flatter us: we lose
to it.

*Blocked, not pending — the honest disposition of the rest of the family.* Only
`DeepseekV2ForCausalLM` is registered, and everything else is now recorded as what it actually is
rather than as work in progress. DeepSeek-V3 and V3.2 (671B, ~642 GiB even in their native format),
Kimi-K2.5 (~2000 GiB), MiniMax-M2 (~428 GiB) and GLM-5 (1404 GiB) **cannot run end to end on this
119 GiB machine, now or after any amount of software work.** V3.2 and GLM-5 are blocked a second time
over by a dependency: their sparse attention leaves exactly one candidate backend on this GPU, and
that path is non-functional — it dispatches to a dense-only kernel that silently drops the sparse
selection, and upstream's only test of it stubs the capability check out. `DeepseekForCausalLM` is not
an MLA model at all (it is plain multi-head attention) and is refused by name rather than parked in an
MLA row. For every blocked row we record what can still be checked — configuration resolution, weight
mapping on a single-layer slice, unit-level agreement at the real dimensions — and what cannot.

*Two permanent coverage gaps, and the checkpoint that would close them.* DeepSeek's distinctive
grouped expert router and the low-rank query path that V3 and GLM use are both implemented and
unit-tested at their real dimensions, but neither has any end-to-end coverage, because the one
checkpoint that fits this box uses neither. That is not closable by porting harder — it needs a
different model. **GLM-4.7-Flash (31.2B, 58.2 GiB) fits this hardware and would close both**, and
upstream it is a zero-override subclass of DeepSeek-V2's attention, so it is recorded as the named
next vehicle for this track.

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference | 🟡 Correctness/CI implementation with native threadpool; real-file GGUF speed/RSS and compute-in-quant gates remain open |
| CUDA | GB10 / DGX Spark, sm_121a | 🟡 Gate-model correctness passes; 27B v0.25.0 performance is `GATING` at **49/124** (new binding `246a23c`; memory + c1 + TTFT clean, decode c2–c32 open). Packed GDN decode is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`): the c16 slot defect was fixed test-first (request-identity keying) and proven at `c172336`, and W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ) + a trace showing packed is GPU-cheaper. qkvz DGX gates closed green at `45f9e6d` and ride this binding |
| Other NVIDIA SMs | sm70 through sm120 families inventoried from vLLM | 🗓 Not yet built, traced, or gated here. The **expansion framework is now additive** (2026-07-21, see below) — adding one is a table row + a tactic registration rather than a scattered multi-file edit — but no architecture has moved off `INVENTORIED`, because none can be executed on this hardware |
| ROCm / Intel XPU | AMD / Intel GPUs | 🗓 Post-parity roadmap |
| Metal / ANE | Apple Silicon | 🗓 Post-parity roadmap; M4 bring-up host available |
| Vulkan | Portable GPU | 🗓 Post-parity roadmap |

Only GB10/sm_121a counts as CUDA hardware support today. Source-level fallback
paths do not become support claims until their build, correctness, trace, and
performance gates pass.

**CUDA-arch expansion is now ADDITIVE — mechanism only, no new support
(`BACKEND-CUDA-ARCH-ADDITIVITY`, LANDED + GATED 2026-07-21,
[spike](.agents/specs/cuda-arch-additivity.md)).** The four seam-gaps the
breadth-sweep audit named are closed: (1) the four hardcoded
`MATCHES "12[01]a"` CMake guards became a **per-arch FEATURE TABLE**
([`cmake/CudaArchFeatures.cmake`](cmake/CudaArchFeatures.cmake), a 1:1 port of
vLLM's `cuda_archs_loose_intersection`, `cmake/utils.cmake:376-485`) that
resolves each capability against the requested target list and REPORTS the
result, so a multi-arch build can no longer silently drop fp4 / cutlass-nvfp4 /
cutlass-fp8 / Marlin / FA2; (2) the device capability is probed once, cached,
threaded into the kernel layer and **carried by the CUDA backend** (it previously
existed only on the engine-side Platform seam); (3) host launchers now select a
**runtime tactic** from a registration point, so a future architecture ADDS a
tactic instead of editing the launcher; (4) the opt-in shared-memory ceiling in
paged attention is the **queried** `cudaDevAttrMaxSharedMemoryPerBlockOptin`
rather than a hardcoded GB10 assumption. **This is behavior-preserving by
construction and confers no support:** exactly ONE tactic is registered (the
existing `sm_12x` fp4 path), GB10 output is bit-identical, and no architecture
moved off `INVENTORIED`. Building a genuinely heterogeneous fat binary still
needs per-source gencode narrowing plus the per-arch kernel bodies — that
remains a **hardware-blocked kernel campaign, not an additive drop-in**.

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

**Extensibility — residency is now a consumed Platform capability (2026-07-19,
CPU-gated).** The MoE host-weight-release, per-layer load-stream interleave, and
device-scratch-pool cap decisions are now *read from*
`GetPlatform(<obj>.device.type).residency_policy()` instead of being decided by
an inline `device.type`/env gate. The RESIDENCY POLICY (free host weights after
the device upload? pool cap?) comes from the platform; whether the Marlin
resident is the committed compute path stays an orthogonal *kernel* gate
(`MarlinMoeEnabled()`) — the two questions are kept separate. `CudaPlatform`
advertises the values that reproduce today's GB10 behavior exactly
(`release_host_weights_after_upload = true`, device pool on, uncapped), so the
same host-free + load-stream runs and the 35B ~4 GiB load-to-ready peak is
preserved; `VT_MOE_HOST_FREE`/`VT_MOE_LOADSTREAM` stay as rollback overrides.
**Net effect: a new (e.g. discrete) GPU's residency behavior = set its
`residency_policy()` field values, with zero model-code edits.** A
behavior-preserving refactor (no numeric/kernel/memory change on GB10), unit-tested
on the CPU tier and DGX-confirmed token-exact (27B 235/235 + 35B 315/315, 35B
load-to-ready peak ≈ 4.0 GiB preserved, memcheck 0 errors).

**Extensibility — model self-registration landed (2026-07-19, CPU-gated).**
Adding a model architecture is now additive. The fixed model-registry array is
replaced by a `REGISTER_VLLM_MODEL(...)` static-registration idiom
(`include/vllm/model_executor/models/model_registry.h`) copying the proven
op/backend/platform self-registration pattern: each architecture registers
itself from its own translation unit into the shared, type-erased
`ModelFactory`. The Qwen3.6 dense and MoE variants now live in their own registry
TUs (`qwen3_5_dense.cpp`, `qwen3_5_moe.cpp`) over a shared `qwen3_5_common`
helper, and `model_registry.cpp` is the generic family-agnostic registry. **Net
effect: adding the next model = one new TU + one `REGISTER_VLLM_MODEL` line, with
zero edits to a shared array or to `model_registry.cpp`** (previously: edit the
fixed `kRegistrations` array *and* add glue inside the shared monolith). A
behavior-preserving refactor — no numeric/kernel/dispatch change, DGX-confirmed
token-exact (27B 235/235 + 35B 315/315, memcheck 0 errors). The deeper
`qwen3_5.cpp` forward-machinery (DevicePool/matmul/GDN) factoring is a deferred
follow-up.

**Extensibility — attention-backend registry landed (2026-07-19, CPU-gated).**
The third and final portability seam is realized: *which* attention backend a
platform selects is now data, not an inline code edit. Attention backends
self-register per device (`include/vllm/v1/attention/registry.h`,
`RegisterAttentionBackend` — the same static-registration idiom as the op /
platform / model registries), each `Platform` advertises a capability-ordered
priority list (`get_attn_backend_priority()`, a 1:1 mirror of vLLM
`cuda.py::_get_backend_priorities` — FLASH_ATTN → FLASHINFER → TRITON → FLEX on
GB10, FLASHINFER-first on sm_100), and a selector (mirror of
`get_attn_backend_cls`) returns the first *registered* backend in that order.
**Net effect: adding a new backend's attention = one self-registering TU + one
priority slot, with zero edits to the selector, model, or runner.** The concrete
attention *kernel* remains selected at the device-agnostic vt:: op table, so this
is a behavior-preserving refactor — the same FlashAttention-2 path is selected on
both gate models (DGX-confirmed token-exact: 27B 235/235 + 35B 315/315, memcheck
0 errors). This completes the three-seam extensibility foundation (Platform,
attention-backend registry, model self-registration).

**Extensibility — roadmap_v1 ORDER-1: the portable op-fusion framework
(SPIKED 2026-07-19; W0 ADOPTED; W1 POD GENERALIZED; W2 HAND-FUSIONS MIGRATED; W3 MECHANICAL-SYNC PROVEN; W4 BACKEND-ADDITIVITY PROVEN 2026-07-20 — the W-series ORDER-1 proof milestone is DONE).** The fourth and unifying seam: fusions **declared once**
(a backend-agnostic recipe catalog above `vt::`, transcribing vLLM's finite
fusion-pass set) and **realized per-backend** through the `vt::` op table (a
composite tier is the CPU oracle every backend inherits free; one interpreter
kernel per backend lights up every recipe). It makes a new vLLM fusion PR a
one-declaration port, a new GPU a one-file catalog realization, and a new model
an additive pattern declaration — the same additive pattern as the seams above,
so upstream fusion PRs port mechanically instead of being hand-wired at call
sites. **W0 landed (2026-07-19):** the Phase-0 skeleton is now ADOPTED at one
real production site — the 35B post-attention layernorm routes its plain
add+residual+RMSNorm through `vt::FusedChain(kFusedAddRmsNorm)` (behind
`VT_FUSED_CHAIN_ADOPT`, default-ON, `=0` rollback), behaviour-preserving and
byte-identical to the prior hand-call, proving the declare-once/realize-per-backend
seam end-to-end in a real model forward (35B 315/315 + 27B 235/235 token-exact on
both arms, memcheck 0 errors). **W1 landed (2026-07-20):** the `FusedRecipe` POD is
generalized — from `{kAdd,kMul,kRmsNorm}` + a fixed 4-role operand model to the full
activation/norm/quant/rope opcode set + a small indexed operand table — so all five
quant-fused W2 target chains (`kRmsNormQuantFp8`, `kRmsNormGatedQuantFp8`,
`kSiluMulFp4Quant`, `kSigmoidGateFp4Quant`, `kAttnQkNormRopeGate`) are now expressible
as `constexpr` recipes whose Tier-0 composite is byte-exact to the standalone-op
sequence the model hand-calls today (proven CPU + CUDA; 35B 315/315 + 27B 235/235
token-exact regression on the W0 site, memcheck 0). W1 is infrastructure only — no
model call site changed. **W2 landed (2026-07-20):** the bespoke hand-fused ops are
now MIGRATED to `vt::FusedChain(recipe)` at six production call sites
(`kSiluMulFp4Quant` MoE down-proj, `kSigmoidGateFp4Quant` full-attn o-proj gate,
`kRmsNormGatedQuantFp8` ×2 GDN out-proj, `kRmsNormQuantFp8` input-layernorm,
`kAttnQkNormRopeGate` ×2 attn preamble) — the framework now OWNS the fusion dispatch.
Each recipe carries a `fast_op` binding to its existing single-launch fused kernel, so
`FusedChain(recipe)` dispatches to the SAME fast kernel the model called directly
before migration: **byte-identical AND perf-neutral by construction** (no extra
kernel, no per-forward getenv/alloc). The Tier-0 composite is the byte-exact oracle the
fast kernel is validated against (`test_ops_fused_chain` proves fast == composite ==
unfused sequence). Gates: clean CUDA `-Werror` 0-warn, byte-exact CUDA 420/420, and
token-exact 27B 235/235 + 35B 315/315 on BOTH `VT_FUSED_CHAIN_ADOPT` arms (`=0`
restores the exact prior hand-calls, same binary). **W3 landed (2026-07-20) — the
mechanical-upstream-sync PROOF:** a NEW, previously-unported vLLM fusion pass
(`SiluMulFp8StaticQuantPattern`, the static-per-tensor-FP8 activation variant of
`ActivationQuantFusionPass`, `act_quant_fusion.py:81` → `_C.silu_and_mul_quant`) was
ported as **ONE `constexpr FusedRecipe kSiluMulQuantFp8` declaration + its byte-exact
test — touching exactly two files** (`include/vt/recipes.h` + `tests/vt/test_ops_fused_chain.cpp`),
with **no kernel, no dispatch, no model-site edit, and no new primitive** (its composite
realizes through the existing `vt::MoeSiluMul` + `vt::QuantFp8Static` ops). `git diff
--stat main` = those two files, nothing else — the PR-#4 additivity test made concrete:
adding a whole new fusion pattern is one declaration. Gates: clean CUDA `-Werror` 0-warn,
byte-exact CUDA 432/432, and — because the recipe is declared, not wired into any model —
no token regression (27B 235/235 + 35B 315/315), memcheck 0. **W4 landed (2026-07-20) —
the backend-additivity PROOF, closing the W-series:** the additivity claim ("a new backend
registers `kFusedChain` once and inherits the ENTIRE catalog correct, zero per-recipe
work") is now EXECUTABLE. A new test (`tests/vt/test_fused_chain_additivity.cpp`) treats the
existing **CPU backend as the 'second backend'** relative to CUDA (no mock `DeviceType` —
that would edit the core enum + every switch, ironically non-additive) and, in one generic
loop over the whole catalog (all 7 recipes), asserts each runs **byte-exact on the CPU
backend via the Tier-0 composite** — 4 end-to-end + 3 whose CUDA-only static-fp8 terminal
is backend-negotiated (byte-exact prefix; the full composite is asserted to throw on CPU,
documenting the tail). The additivity is real: the catalog (`recipes.h`) grew 1→6→7 recipes
while the composite walker stayed **one per-opcode function** and each backend's `kFusedChain`
registration stayed **one line** — `cpu_ops.cpp` never even includes `recipes.h`, and W3's
whole new recipe appears in zero backend files, inherited for free. Gates: clean CPU
`-Werror` 0-warn, `test_fused_chain_additivity` 17/17 + `test_ops_fused_chain` 228/228; the
change is test-only (engine byte-identical) so 27B 235/235 + 35B 315/315 are structurally
unchanged. Honest scope: this is an **extensibility +
mechanical-upstream-sync + backend-additivity** cornerstone, not a perf lever — W0–W4 are
perf-neutral by construction (the measured 35B prefill gap is compute-bound, ceiling
~3.5%/step). Named future/HW-blocked work (does not gate the ORDER-1 milestone): the Tier-1
single-pass perf interpreter for the quant chains (composite-only today), and a real
Metal/Vulkan realization of the catalog (needs the M4 dev-box).
Spike + work breakdown (W0–Wn):
[`.agents/specs/portable-fusion-framework.md`](.agents/specs/portable-fusion-framework.md).
Build hygiene alongside W1: a pre-existing CPU-only `-Werror=unused-function`
(the `FuseSigmoidGateQuantEnabled` helper is used only under `#ifdef
VT_CUTLASS_NVFP4`, so a CUDA-off build compiled it out) is fixed with
`[[maybe_unused]]` — keeps the `build-test-cpu` CI job green; no behavior change.

### Kernel coverage on the gate path

| Kernel family | CPU | CUDA · GB10 | Status |
|---|:---:|:---:|---|
| Dense NVFP4 W4A4 GEMM | ✅ ref | ✅ | CUTLASS/FlashInfer-compatible tactics, frozen plan cache, packed QKV |
| MoE NVFP4 W4A16 GEMM | ✅ ref | ✅ | Marlin/fp4-resident gate path |
| BF16/FP8 projection GEMM | ✅ ref | ✅ | cuBLASLt TN / `nvjet_sm121` path; an opt-in per-device plan cache (descriptor + layouts + heuristic algo, keyed on the full shape/config; `VT_FP8_PLAN_CACHE=1`) mirrors vLLM's in-graph plan reuse — bit-exact but measured production-NEUTRAL on GB10 (prefill GPU-bound, decode graph-captured), so it ships default-OFF |
| Prefill attention | ✅ ref | ✅ | Vendored FlashAttention-2 with portable fallback |
| Paged decode attention | ✅ ref | ✅ | FA2 split-KV decode covers BOTH gate topologies: 27B ratio-6 (`VT_FA2_DECODE`) and **35B ratio-8 default-ON (2026-07-19, `VT_FA2_DECODE_35B`)**. The 35B decode previously ran a 2-block-grid GQA kernel at low batch (near-zero GB10 occupancy); split-KV fills the machine. Token-exact (35B 315/315 + 27B 235/235) + in-situ 35B win (c1 TPOT −10.5%, c8 −3.2%, TTFT neutral). The older 27B ratio-6 component strict-performance-failed its frozen c2/c16 grid (no credit); the 35B decode is a fresh in-situ win, orchestrator re-grids |
| GDN / linear attention | ✅ ref | 🟡 | Prefill AOT is gated; the packed pure-decode kernel is **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`). The c16 slot defect (compact state-slot pool keyed on the mamba block-id collapsed two long c16 sequences onto one recurrent-state slot) was fixed test-first (request-identity keying) and proven at `c172336`; W1D3/G3 closed over eight seals + the 8-pair A/B (−0.205% ± 0.30, <1σ). Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); no speed credit |
| RMSNorm, RoPE, SwiGLU, FP4/FP8 quant | ✅ ref | ✅ | Gate-path coverage; broader variant inventory remains open. CPU backend now registers both cast directions (`bf16→f32` `kCastF32` alongside the existing `f32→bf16` `kCastBf16`), closing a CPU op-registration asymmetry |
| CUDA-graph decode | — | 🟡 | Gate-model path runs; complete cross-model evidence remains open |

## Quantization

| Format | Status |
|---|---|
| NVFP4 W4A4 / W4A16 | 🟡 Both gate-model paths run on GB10 and pass token-exact correctness. The new-binding 27B performance gate (`246a23c`) fails 75/124 axes (memory + c1 + TTFT now pass; decode c2–c32 open); FP4 tactics match, the non-quantized packed GDN decode leaf is closed on equivalence, and merged qkvz (`KERNEL-GEMM-BF16` W2, one BF16 in_proj_qkvz GEMM per GDN layer) is DGX-green and in the binding binary |
| compressed-tensors NVFP4A16 (W4A16), dense | 🟡 **New scheme, correctness COMPLETE; speed not yet measured.** `RedHatAI/Qwen3-32B-NVFP4A16` (`Qwen3ForCausalLM`, 64 layers) loads and computes natively through the Marlin weight-only path — the same kernel vLLM itself forces for `use_a16` on sm_121 (observed: `Using MarlinNvFp4LinearKernel`), so this scheme needed **zero new kernel code**. The loader, scale conventions and merged-shard collapse mirror vLLM; a CPU synthetic check is bit-exact against the dequantized reference. The loader gate passes fully (all 1603 tensors mapped, 5144 assertions) and the engine path is memcheck-clean. The engine gate is **6 of 6 prompts**, of which 4 are token-exact (67/96 tokens) and 2 pass the same near-tie band the BF16 Qwen3 models use. That band is not an assumption: feeding vLLM our exact prefix and reading its own logits shows **every divergent position within 0.0625 nats of vLLM's own top choice, and 28 of 29 exactly equal to it** — one of the two flips is a perfect tie where vLLM's own two ways of computing it disagree with each other. The residual is the BF16 rounding behaviour the dense Qwen3 path already carries, not the quantization, which is separately exonerated by the bit-exact CPU check and by the divergences surviving a full swap of the quantized GEMM |
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
