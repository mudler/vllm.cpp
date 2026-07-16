# c2 + c8 full-step attribution — correct-state same-method split (2026-07-16)

**Kind:** diagnostic (no engine/kernel change; records only). Adjudicates whether
the c2–c8 decode-mean gap is host-window or GPU-busy, mirroring the a2329e1 c16
method (state.md 2026-07-16 "CORRECT-STATE GDN kernel comparison") at c2 and c8.
Resolves the [lost-lanes rescan](rescan-lost-lanes-2026-07-16.md) §4 UNATTRIBUTED
downgrade. `benchmark_binding=false`, no speed credit, binding stays 49/124.

## Method

- **Build:** production from `main` @ `beb8497` (fresh detached worktree, full CUDA
  `RelWithDebInfo` build, arch `121a`, production flags FLASH_ATTN/MARLIN/SERVER/
  TRITON on, `VLLM_CPP_BENCH_PROFILE_CONTROL=OFF`). Correctness gate
  `test_qwen27_paged_engine` packed-decode **235/235** before each OURS capture.
- **Corpus/client (binding-identical):** the `246a23c` binding corpus
  (`corpus/27/vllm/c{2,8}-r1.jsonl`; generated `--requests-per-partition 192
  --warmup-requests 1`, in1024/out128, greedy `temperature 0 --ignore-eos`);
  point sizes c2=6 prompts/2 warmups, c8=24/8; server `--max-num-seqs 32
  --max-num-batched-tokens 2048 --no-enable-prefix-caching --num-blocks 4736`.
- **OURS:** `nsys profile --cuda-graph-trace=node --delay 25 --duration 65` node
  trace of the production server (`VT_GDN_PACKED_DECODE=1`, full FP4
  autotune/plan-cache env), inside `/usr/bin/env -i`; pure-decode span = the
  prefill-free gap containing the most `GdnPackedDecodeKernel` launches (one clean
  decode wave; c2 **127 steps**/13.74 s, c8 **126 steps**/14.56 s); steps =
  packed-launches/48 (48 GDN layers, batch-independent).
- **vLLM:** `tools/bench/profile_vllm_online_gate.py` torch-profiler
  (`--mamba-ssm-cache-dtype float32` resolved, async-scheduling True,
  `output_digests_equal` both points), 3 measured reps, `env -i`; per-step = per
  clean `execute_context_0(0)_generation_N(N)` window (c2 **1524**, c8 **1508**
  windows). Same named families + bundled GEMM+MoE+attention+rest as the c16 record.
- **Wall anchor:** the `246a23c` binding decode means; OURS capture-measured mean
  TPOT corroborates within noise (c2 109.18 vs 109.85, −0.6%; c8 129.73 vs 131.41,
  −1.3%). idle = wall − busy per side, the c16 convention.
- **Evidence root (immutable):** `dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`
  (`SUMMARY.json` sha256 `5fa07663…e231`; ours/vllm decomp JSONs, gap analyses,
  capture/analyze scripts, raw client results, logs). One `flock /tmp/gpu` held per
  capture series; GPU verified idle at each acquisition.

## Decomposition — c2 (per c2 decode step, ms; Δ = ours − vLLM)

| family | ours | vLLM | Δ |
|---|---|---|---|
| GEMM+MoE+attention+rest | 101.154 | 101.397 | −0.243 |
| GDN packed recurrence | 2.548 | 1.617 | **+0.931** |
| RMSNorm plain (129/step) | 2.117 | 0.381 | **+1.736** |
| RMSNorm gated (48/step) | 0.135 | fused | +0.135 |
| FP4-quant (144/step) | 0.638 | 0.336 | +0.302 |
| SiLU-mul | 0.443 | 0.217 | +0.226 |
| GDN conv-update | 0.275 | 0.203 | +0.073 |
| **TOTAL GPU-busy/step** | **107.310** | **104.151** | **+3.159** |
| GPU-idle (wall−busy)/step | 2.540 | 3.269 | **−0.729** |
| **WALL (binding mean TPOT)** | **109.85** | **107.42** | **+2.43** |

busy% ours 97.7 / vLLM 97.0. Gap split: **GPU-busy +3.16 (130%), host/idle −0.73
(−30%)** — ours has *less* idle than vLLM at c2. Norm/quant/act glue Δ = +2.399.

## Decomposition — c8 (per c8 decode step, ms; Δ = ours − vLLM)

| family | ours | vLLM | Δ |
|---|---|---|---|
| GEMM+MoE+attention+rest | 100.847 | 102.124 | **−1.277** |
| GDN packed recurrence | 10.045 | 8.514 | **+1.531** |
| RMSNorm plain (129/step) | 2.028 | 0.377 | **+1.651** |
| RMSNorm gated (48/step) | 0.256 | fused | +0.256 |
| FP4-quant (144/step) | 0.635 | 0.334 | +0.301 |
| SiLU-mul | 0.478 | 0.241 | +0.237 |
| GDN conv-update | 0.418 | 0.300 | +0.118 |
| **TOTAL GPU-busy/step** | **114.706** | **111.890** | **+2.816** |
| GPU-idle (wall−busy)/step | 16.704 | 12.230 | **+4.474** |
| **WALL (binding mean TPOT)** | **131.41** | **124.12** | **+7.29** |

busy% ours 87.3 / vLLM 90.1. Gap split: **GPU-busy +2.82 (38.6%), idle +4.47
(61.4%)**. Norm/quant/act glue Δ = +2.445. The idle mass lives OUTSIDE pure-decode
spans: in-span device idle is at parity (ours 0.919 vs vLLM ~0.88 ms/step; ours
99.2% busy inside the decode wave).

## The four explicit answers

**(a) GPU-busy vs host/idle fraction.** c2: the +2.43 ms gap is **entirely
GPU-busy** — busy Δ +3.16 (130% of the gap), idle Δ **−0.73** (ours idles less
than vLLM). c8: **38.6% GPU-busy (+2.82) / 61.4% idle (+4.47)** — but the idle
delta is NOT a per-step host window: inside pure-decode waves both engines are
≥99% busy at parity; the +4.47 accrues at wave boundaries (prefill-interruption
handling smeared into the client TPOT — the same wave-boundary stall mechanism
attributed in [tail-stall-analysis](tail-stall-analysis-2026-07-16.md), here
shown to move the c8 MEAN, not just the tails; the parallel CPU wave
discriminator proved the step COMPOSITION is byte-identical both sides, so the
magnitude gap is the async depth-2 overlap = W3, not a scheduler divergence).

**(b) RMSNorm per-launch delta.** Yes — visible under the same method pair as the
c16 record and batch-INDEPENDENT as predicted: ours 2.117/2.028/2.006 ms/step at
c2/c8/c16 (129 launches, ~16 µs/launch in-trace) vs vLLM 0.381/0.377/0.391
(~3.0 µs/launch) ⇒ Δ **+1.74 (c2) / +1.65 (c8)** ms/step. Whole glue family
(RMSNorm plain+gated, FP4-quant, SiLU): Δ **+2.40 (c2) / +2.45 (c8)** —
essentially the ENTIRE c2 wall gap. Caveat (from the
[reconciliation](decode-norm-quant-fusion-reconcile-2026-07-16.md)): the profiler
pair differs per engine (nsys node vs torch CUPTI); its isolated microbench put
ours' RMSNorm at 6–9 µs, but that ran the WRONG shape (H=2048–4096; the real 27B
decode norm is H=5120 — `CLAIM-EW-NORM-ACT-1` Phase-1 grounding), and the
same-profiler Phase-1 adjudication has since CONFIRMED a 3.18–3.56× per-launch
gap and ported vLLM's kernel default-OFF (`VT_RMSNORM_DECODE_FAST`). Even at the
conservative reading the glue Δ cannot flip the c2 verdict — the idle Δ is
negative.

**(c) Block-table/prepare host window.** NOT meaningfully exposed. Ours in-span
device idle is 0.942 (c2) / 0.919 (c8) ms/step total — 99.1/99.2% busy — of which
the post-sampler step-boundary hole (where block-table/prepare/sampler-D2H host
work sits) is only **0.116 (c2) / 0.186 (c8) ms/step**; vLLM's same-position
budget is ~0.84–0.88 ms/step (in-window idle 0.79/0.81 + inter-window host gap
median 48/68 µs). Net exposed host-window delta ≤ ~0.1 ms/step — consistent with
the rescan's 50–250 µs/step block-table estimate but an order of magnitude below
the gaps. Host-plumbing levers are micro for the c2–c8 means.

**(d) Interpolation vs regime change.** REGIME CHANGE. Busy Δ is non-monotonic
(c2 +3.16 → c8 +2.82 → c16 +4.65) and idle Δ flips sign (−0.73 → +4.47 → +3.25).
Two superposed mechanisms: (i) a **batch-independent GPU-busy kernel-glue floor
(~2.4 ms/step at every concurrency)** plus a batch-growing GDN recurrence Δ
(+0.93/+1.53/+2.06) — at c2 this is the whole gap; (ii) a **wave-boundary
prefill-stall scheduling component** that appears from c8 up (+4.5/+3.3 ms/step
equivalent) and is absent (sign-negative) at c2. GEMM/MoE/attention stays at
parity to ours-slightly-faster (−0.24/−1.28/+0.08).

## Lever routing (the adjudication)

1. **c2–c4 decode means → GPU-busy KERNEL levers**, primarily the batch-independent
   norm/quant/act glue (`KERNEL-EW-NORM-ACT`, ≈2.4 ms/step everywhere — worth ~2.2%
   at c2 alone) and the GDN recurrence tiling (grows with batch; the open
   occupancy-aware redesign). Host plumbing (block-table cluster, sampler alloc)
   is bounded ≤~0.2 ms/step exposed — keep as hygiene, not as a c2–c8 lever.
2. **c8 (and c16/c32) extra mass → the wave-boundary stall mechanism** — per the
   parallel CPU discriminator this is the async depth-2 OVERLAP (`ENG-ASYNC-SCHED`
   W3 family; composition byte-identical, no sync-scheduler fix exists) — now
   shown to move the decode MEANS at c8, not just the p99/p90 tails.
3. The 07-14 "c2–c8 is host-side" attribution is **REFUTED** for c2 (it is
   GPU-busy) and **RESHAPED** for c8 (the non-busy part is scheduling at wave
   boundaries, not per-step host transport).

## Deviations

- Wall anchored on the `246a23c` binding TPOT means (the c16 record used its own
  capture TPOT); OURS capture TPOT agrees within 0.6–1.3%, and the binding is the
  authoritative decode mean the gap targets reference.
- OURS decode-span selection = "most packed kernels" prefill-free gap (not longest
  gap): at low concurrency, server startup/warmup creates longer idle gaps; the
  chosen span is one clean batch-N decode wave (batch-consistent recurrence
  2.55/10.04/21.31 ms at c2/c8/c16; batch-independent counts 129 RMSNorm / 48 GDN /
  144 FP4 per step at every concurrency).
- vLLM captures ran in a clean lock session: the first in-series back-to-back run
  failed because the OURS server's GPU memory had not released before the vLLM
  load (plus box contention with a concurrent agent series, NVRM OOM 07:30Z); the
  clean-session re-run completed with deterministic digests. No recipe defect.
- The c2 OURS nsys window started during server warmup (startup 43 s > delay 25 s);
  harmless — the analyzed span is selected inside the client's decode by content,
  not by the window edges.
