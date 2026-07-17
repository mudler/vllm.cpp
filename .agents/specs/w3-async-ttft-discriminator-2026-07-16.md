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

## Results (2026-07-17, campaign COMPLETE — one flock, 42 legs, 0 failures)

Token-exactness precondition 6/6 PASS (27B+35B × default/W3-on/W3-off). Every
vLLM arm log-confirmed ("Asynchronous scheduling is enabled/disabled." per
server). No leg shows the void signature (all legs 504–1108 tok/s, production
scale). Deviations: campaign binary @ `6ea7856` (predates the `696a991` RMSNorm
default flip — same binary both arms, deltas internally valid); vLLM legs run
`env` PATH-only like the binding grid (not `env -i`); corpus partition r1
reused across reps per the validated ab.sh recipe.

### (1) vLLM self-A/B — vLLM's OWN async pays the same TTFT premium

Mean over 3 interleaved reps (w0 discarded):

| c | arm | tok/s | mean TPOT | mean TTFT | p99 ITL | p90 ITL |
|---|---|---|---|---|---|---|
| 8 | async ON | 515.1 | 122.6 | 2251.9 | 473.2 | 114.9 |
| 8 | async OFF | 519.2 | 125.2 | 1783.3 | 477.6 | 114.8 |
| 16 | async ON | 804.3 | 157.4 | 2834.5 | 894.0 | 129.8 |
| 16 | async OFF | 809.7 | 161.4 | 2172.0 | 875.4 | 130.2 |
| 32 | async ON | 1097.3 | 232.2 | 3935.2 | 960.5 | 552.8 |
| 32 | async OFF | 1107.4 | 236.5 | 3085.0 | 944.2 | 550.6 |

vLLM async-ON vs its own sync: throughput **−0.66 to −0.91 %** (async does NOT
raise X — it is slightly negative on this box/workload), TPOT **−2.6 to
−4.3 ms/step**, **mean TTFT +26.3 % / +30.5 % / +27.6 % (+469/+663/+850 ms)**.
So the TTFT premium IS inherent to async depth-2 scheduling and vLLM pays it
too; the task's premise ("vLLM's async pays NO TTFT premium") is REFUTED, as the
§Calibration mis-citation already indicated. Corollary: the prior W3 ship-gate
"≥+1.5 % throughput" was mis-calibrated — the mirrored feature does not raise
throughput in vLLM itself; upstream defaults it ON for the TPOT/tail win.

### (2) Ours W3-on vs W3-off — the tail prediction is CONFIRMED

| c | arm | tok/s | mean TPOT | mean TTFT | p99 ITL | p90 ITL |
|---|---|---|---|---|---|---|
| 8 | W3-on | 504.9 | 125.5 | 2263.1 | **527.4** | 117.5 |
| 8 | W3-off | 507.9 | 128.9 | 1709.3 | 856.8 | 117.6 |
| 16 | W3-on | 790.8 | 161.5 | 2721.4 | 877.5 | 133.2 |
| 16 | W3-off | 794.8 | 166.2 | 2008.6 | 877.1 | 133.6 |
| 32 | W3-on | 1084.9 | 238.2 | 3568.6 | 945.3 | **534.4** |
| 32 | W3-off | 1089.8 | 243.5 | 2739.3 | 942.3 | 698.7 |

Tail ratios (gate convention vLLM/ours, ACCEPT ≥0.85):

| axis | W3-off | W3-on | vs binding denom (477.8/560.2) | verdict |
|---|---|---|---|---|
| c8 `p99_itl` | 0.552 | **0.897** | 0.906 | FAIL → **in-band PASS** |
| c32 `p90_itl` | 0.791 | **1.034** | 1.048 | FAIL → **strict PASS (ours beats vLLM)** |

The tail-stall spec's ACCEPT criteria hold: the ~500–550 ms single-prefill band
APPEARS under W3-on (ours c8: 54 events in the 500-band vs ZERO under W3-off,
which had 131/162 events in the uniform 800-band), both ratios ≥0.85, decode
means IMPROVE (TPOT −3.5/−4.7/−5.3 ms/step), throughput −0.45 to −0.59 %
(within the vLLM-async's own −0.66 to −0.91 % envelope). TTFT rises +32/+36/+30 %
vs our own sync arm — the same premium vLLM pays (see §3).

### (3) The two async patterns are THE SAME — no divergence exists

(a) Deltas side by side (async-on minus sync, per engine):

| c | engine | ΔX tok/s | ΔTPOT ms | ΔTTFT ms |
|---|---|---|---|---|
| 8 | vLLM | −4.1 (−0.79 %) | −2.6 | +469 (+26.3 %) |
| 8 | ours | −3.0 (−0.59 %) | −3.5 | +554 (+32.4 %) |
| 16 | vLLM | −5.4 (−0.66 %) | −4.0 | +663 (+30.5 %) |
| 16 | ours | −4.0 (−0.51 %) | −4.7 | +713 (+35.5 %) |
| 32 | vLLM | −10.1 (−0.91 %) | −4.3 | +850 (+27.6 %) |
| 32 | ours | −4.9 (−0.45 %) | −5.3 | +829 (+30.3 %) |

(b) Four-arm ITL spike-location (spikes >400 ms; first/last 15 tokens; 3 reps):

| arm | spikes | START | MID | END | magnitude bands (ms:events) |
|---|---|---|---|---|---|
| ours c8 W3-on | 144 | 108 | 0 | 36 | 400:24 **500:54** 800:66 |
| ours c8 W3-off | 162 | 120 | 0 | 42 | 400:31 **800:131** (no 500-band) |
| vLLM c8 aon | 129 | 84 | 0 | 45 | graded 400–900 |
| vLLM c8 aoff | 201 | 120 | 0 | 81 | 400:120 800:81 |
| ours c32 W3-on | 7465 | 4380 | 5 | 3080 | graded, 900:6136 |
| ours c32 W3-off | 7983 | 4590 | 18 | 3375 | 900:6690 |
| vLLM c32 aon | 7965 | 3975 | 69 | 3921 | graded, 900:5236 |
| vLLM c32 aoff | 8316 | 4227 | 99 | 3990 | 500:1785 900:6024 |

The binding-era fingerprint ("ours START-loaded, vLLM END-loaded") is exactly
reproduced by the SYNC-ours vs ASYNC-vLLM pair (w3off 4590/18/3375 ≈ binding
ours 4608/0/3375; aon 3975/69/3921 ≈ binding vLLM 4002/54/3990) — it was a
property of the SCHEDULING MODE, not the engine: async shifts spike mass toward
END/graded and grades the magnitude bands on BOTH engines; sync concentrates
START-loaded uniform two-prefill stalls on BOTH.

(c) **Named divergence: HONEST ABSENCE.** Ours-W3on reproduces vLLM-async's
X/TPOT/TTFT deltas, tail structure, spike placement and stall-magnitude grading
within noise. Grounding: our depth-2 loop `EngineCore::step_with_batch_queue`
(`src/vllm/v1/engine/core.cpp`) mirrors vLLM `core.py:519-632` (v0.25.0
`vllm/v1/engine/core.py`, `step_with_batch_queue`); admission timing proven
same-step by `tests/vllm/v1/test_async_admission_timing.cpp` (89b329e) against
`core.py:1259-1298`; scheduler composition byte-identical
(`tests/vllm/v1/test_scheduler_wave.cpp`, 20fc0e1); output visibility resolves
at consume time in both (`AsyncGPUModelRunnerOutput::get_output` ==
`async_utils.py:12-70`). Nothing measurable remains to localize: the +705 ms
premium (fresh: +713 ms) matches vLLM's own +663 ms at c16 within one step.
**No fix exists to implement — the premium is the mirrored behavior.**

### (4) Axis arithmetic — W3-ON nets positive AS-IS

Per-axis ratios vs the PRODUCTION bar (vLLM async-ON), 18 axes × c8/c16/c32,
this campaign's interleaved arms (strict PASS = ratio ≥1.0; pass* = p90/p99
inside the 15 % tail band):

- strict-PASS count: W3-off **14**/54 → W3-on **15**/54.
- FAIL→PASS under W3-on: `c8 p99_tpot` (1.0087), `c16 p99_tpot` (1.0036),
  **`c32 p90_itl` (1.0344)** — one of the two binding anomalies, now beating vLLM.
- FAIL→in-band under W3-on: **`c8 p99_itl` 0.552 → 0.897** — the other binding
  anomaly, a 62 % tail reduction to within one prefill of vLLM.
- PASS→FAIL under W3-on: `c8 mean_ttft` (1.3174 → **0.9951**, −0.5 %, within
  run noise) and `c8 p99_ttft` (1.0380 → 0.9940, in-band). Every other TTFT axis
  (10 of 12) RETAINS PASS with margin (c16 mean 1.0415, c32 mean 1.1027).
- Every TPOT/ITL mean/median ratio improves +2.3–3.3 pp toward parity (none of
  the already-failing decode means flips; they are the separate kernel-glue /
  recurrence levers).
- Throughput ratio cost: −0.5 pp (0.9859→0.9800 / 0.9882→0.9832 / 0.9932→0.9887)
  — the same trade vLLM's own async makes (−0.7 to −0.9 %).

**Verdict: W3-ON nets positive AS-IS** — +3 strict flips (incl. one binding
anomaly) + the second anomaly into-band vs −2 marginal c8 TTFT flips (−0.5/−0.6 %,
noise-scale), and it is the MIRROR obligation: vLLM's production default is
async-ON paying the same premium. The "W3 needs a throughput lever before it
ships" framing is retired — vLLM's async has no throughput win either. Decision
for the next grid: **flip W3 default ON** (mirror `vllm/config/vllm.py:1040`),
owned by `CLAIM-ASYNC-SCHED-W3` (config resolution + fresh token gates + grid);
this discriminator changes no engine default itself (records only).

Evidence (immutable): `dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…/`
(42 leg JSONs with per-request `itls[]`, server/client logs, `asched-*.txt` arm
markers, gate logs; `campaign.out` driver log; `analyze.py`/`axis_table.py`).
`benchmark_binding=false`; binding stays 49/124.
