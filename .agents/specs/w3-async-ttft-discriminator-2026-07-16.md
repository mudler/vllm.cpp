# W3 async TTFT-premium discriminator (2026-07-16)

**Row:** `ENG-ASYNC-SCHED` (`W3`) diagnostic sub-probe · **Claim:**
`CLAIM-W3-ASYNC-DISC` (extends the async-serving campaign) · **Kind:** read-only
benchmark discriminator + (if a CPU-expressible fix emerges) test-first output-timing
fix. `benchmark_binding=false`, **no speed credit**; binding stays 49/124.

Evidence root (immutable): `dgx:~/work/vllm.cpp-w3-discriminator/<sha>`
(source SHA `6ea7856`). Build `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc`, CUTLASS + FA2 lines
hard-verified in the configure log before any run (void-signature guard).

## Question

The W3 c16 record (`6ea7856`, ab-fix2) shows W3-on pays a **+705 ms mean-TTFT
premium** over W3-off (2732 vs 2027 ms) at neutral throughput and a −5.4 ms/step
TPOT win. Is that premium (a) inherent to async depth-2 overlap (Little's law,
paid by vLLM too), or (b) a divergence in our engine loop / output timing that
vLLM does not pay? Four sub-questions:

1. **vLLM self-A/B (c16, c8, c32):** async scheduling ON (default) vs OFF
   (`--no-async-scheduling`). Does vLLM's own async raise throughput? Does it pay
   a TTFT premium vs its own sync?
2. **Ours W3-on vs W3-off (c8, c16, c32):** the pending tail confirmation
   (`tail-stall-analysis-2026-07-16.md` §Empirical confirmation gate). Prediction:
   c8 `p99_itl` and c32 `p90_itl` flip toward PASS (ratio ≥0.85 vs vLLM) under
   W3-on via stall amortization, at the cost of higher TTFT.
3. **Diff the two async patterns:** vLLM async-vs-sync deltas vs our W3on-vs-off
   deltas; ITL spike location (start- vs end-loaded) in all four arms; localize
   any divergent mechanism in BOTH codebases (file:line) or record its honest
   absence.
4. **Axis arithmetic:** count the binding-grid axes W3-ON would flip and state
   whether W3-ON nets positive as-is, only-with-fix, or never.

## Calibration — the task's "vLLM ~2005 ms TTFT" is a mis-citation

Read READ-ONLY from the binding evidence raw
(`~/work/vllm.cpp-online-gate/evidence/246a23c…/raw/27/{vllm,ours}/c*-r*.json`),
the actual binding per-rep numbers (async-ON vLLM vs SYNC ours) are:

| Axis (c16, 3 reps) | vLLM async-ON | ours SYNC |
|---|---|---|
| total tok/s | 793.7 / 794.4 / 794.5 | 792.5 / 788.0 / 790.6 |
| mean TPOT ms | 159.7 / 159.5 / 159.6 | 166.9 / 167.7 / 167.5 |
| **mean TTFT ms** | **2846 / 2861 / 2838** | **1980 / 2023 / 1968** |
| p99_itl ms | 897 / 901 / 895 | 876 / 876 / 870 |
| p90_itl ms | 134.0 / 133.6 / 133.9 | 135.9 / 136.3 / 136.5 |

So at c16 **vLLM's async TTFT (~2848 ms) is HIGHER than ours sync (~1990 ms)** —
the opposite of the task's framing ("vLLM async 2005 beats our sync 2027"). The
"~2005" figure does not appear in the binding raw for vLLM; it is close to *ours*
sync TTFT. Consequence: on the binding gate's TTFT axis (ours ≤ vLLM to PASS),
ours already BEATS vLLM in both arms; the +705 ms premium is ours-internal
(W3-on vs W3-off), not a gate loss. This is the single most important correction
the discriminator makes, and it reframes the axis arithmetic (see §4).

Binding tail denominators (async-ON vLLM), for the task-#2 prediction:

| Axis | vLLM async-ON (binding) | ours SYNC (binding, FAIL) | ratio |
|---|---|---|---|
| c8 `p99_itl` | 477.8 ms (r1) | 853.3 | 0.560 |
| c32 `p90_itl` | 560.2 ms (r2) | 706.8 | 0.792 |

Also note the TTFT-vs-tail trade at every concurrency in the binding raw:
ours (sync) has LOWER mean TTFT but HIGHER ITL tails; vLLM (async) the reverse
(c8: ours TTFT 1720 vs vLLM 2270, ours p99_itl 853 vs vLLM 478; c32: ours TTFT
2740 vs vLLM 3945, ours p90_itl 707 vs vLLM 560). Async trades TTFT for smoother
decode — the hypothesis the self-A/B tests directly.

## Method

One `flock /tmp/gpu` for the whole series (interleaved within each arm-set;
w0 cold discard + 3 pairs). Same frozen binding vLLM corpus
(`corpus/27/vllm/c{C}-r1.jsonl`, r1 across reps to isolate the arm delta, per the
validated ab.sh recipe), binding client params (in1024/out128 greedy,
`--num-prompts` = POINTS c8→24/c16→96/c32→192, `--num-warmups`=concurrency,
`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
--save-detailed` retaining per-request `itls[]`).

- vLLM arm: binding server command (`vllm serve … --gpu-memory-utilization 0.6
  --max-num-seqs 32 --max-num-batched-tokens 2048 --no-enable-prefix-caching
  --mamba-ssm-cache-dtype float32`), async OFF adds `--no-async-scheduling`; arm
  confirmed per-server from the "Asynchronous scheduling is enabled/disabled." log
  (`vllm/config/vllm.py:1042`). Default resolves ON for this model
  (`vllm/config/vllm.py:1040`).
- ours arm: binding server (`examples/server … --num-blocks 4736 --max-num-seqs 32
  --max-num-batched-tokens 2048 --no-enable-prefix-caching`) + the ab.sh FP4
  production env; W3-on `VT_ASYNC_RUNNER=1`, W3-off `VT_ASYNC_RUNNER=1
  VT_ASYNC_SCHED=0` (same-binary rollback).
- Token-exactness precondition: 27B + 35B paged-engine gates × {default, W3-on,
  W3-off}.

## Results — PENDING (campaign running under one flock)

<!-- filled at campaign completion: vLLM self-A/B table; ours W3 c8/c16/c32 table
with tail ratios vs prediction; four-arm spike-location table; named divergence
or its honest absence with file:line; axis arithmetic; any fix. -->
