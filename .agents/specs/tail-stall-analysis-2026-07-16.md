# ITL tail-stall analysis — c8 p99_itl / c32 p90_itl attribution (2026-07-16)

**Row:** `SERVE-GATE-ONLINE` (both failing tail axes of the NEW BINDING
`246a23c`) · **Kind:** read-only diagnostic on the binding evidence root
(`benchmark_binding=false`, no speed credit). Analysis agent, Opus 4.8, on
`dgx:~/work/vllm.cpp-online-gate/evidence/246a23cfa423e8e50c65b0ff067be55f3a3c7bf9`
(immutable). Method reproduces the gate's own r1 percentiles exactly
(p99 853.3 / p90 706.8) before drawing conclusions; r2/r3 within ±1%.

## Verdict

Both failing tail axes — **c8 `p99_itl` 0.5599** (853.3 vs 477.8 ms) and
**c32 `p90_itl` 0.7925** (706.8 vs 560.2 ms) — are ONE mechanism:
**batch-wide prefill stalls at request-wave boundaries whose per-event
magnitude is uniformly ~860 ms in ours (two full 1024-token prefills filling
the 2048-token step budget) vs graded ~500 ms in vLLM (dominated by
single-prefill events)**. The decode body is at parity: tokens 16–111 of every
request are razor-flat (ours body p99 121.6/169.5 ms at c8/c32 vs vLLM
119.6/165.5) with ZERO mid-sequence stalls in ours at either concurrency.

## Evidence (all from per-request `itls[]` + `start_times[]` + `ttfts[]`)

1. **Cross-request wall-clock correlation.** 96% (c8) to 100% (c32) of spikes
   >400 ms land in 100 ms bins where ≥3 concurrent requests spike TOGETHER;
   the largest c32 stalls hit **31/32 concurrent requests simultaneously at 16
   distinct token indices**. A per-request periodic maintenance cause (GDN
   state-pool ops, block alloc every N tokens, log flush) is impossible — it
   would strike the same token index per request, not 16 different ones at one
   wall-clock instant. RULED OUT with CUDA-graph recapture / allocator GC
   (stalls recur at exactly the wave period, not randomly) and client artifacts.
2. **Wave-period recurrence.** c8 mega-stalls at t≈3.4/21.7/40.0 s (Δ≈18.3 s =
   24 req / 8 conc wave period); c32 at t≈14.4/48.4/82.2/116.2/150.1/184.0 s
   (Δ=34.0±0.1 s = 192/32 wave period).
3. **Bimodal token-index location.** Spikes >400 ms live ONLY at a request's
   first ~15 tokens (stalled by its own admission wave's sibling prefills) and
   last ~15 tokens (stalled by the NEXT wave's prefills). Ours c8: 36 start /
   0 middle / 12 end; c32: 1536 / 0 / 1125. vLLM shows the same shape (c32:
   1334 / 18 / 1330).
4. **Magnitude mechanism.** One isolated 1024-prefill costs ~440 ms (c1
   mean TTFT ours 440 vs vLLM 452 — parity). Budget 2048 = exactly two
   prompts, outputs fixed at 128 → ours admits/finishes in PAIRS and packs two
   full prefills per step: all 48 c8 spikes and 2330/2661 c32 spikes sit in
   the 800–1000 ms band. vLLM on the IDENTICAL budget/server args shows a
   graded distribution with a 400–600 ms band (c8: 18 events) that ours
   **completely lacks**.
5. **Counterfactuals (gate r1 data, band = ratio ≥0.85).** Removing spikes:
   c8 p99 → 137 ms (ratio 3.49), c32 p90 → 168 ms (3.31). Realistic bound —
   cap each stall at ONE prefill (~500–550 ms): c8 ratio 0.869–0.956 **PASS**,
   c32 ratio 0.927–1.112 **PASS**. Halving the per-event stall is
   measured-sufficient to flip BOTH axes; no other axis moves.

## Tension with the c2-era verdict (scheduler-prefill-coschedule.md)

That spike traced `schedule()` policy 1:1 against pinned vLLM and concluded
"no co-scheduling divergence — both sides co-schedule two 1024 prefills that
fit the budget". This analysis does NOT contradict the code trace; it shows the
**emergent regime differs**: ours locks into finish-in-pairs waves (uniform
two-prefill stalls, self-perpetuating because both requests start the same
step and emit exactly 128 tokens), while vLLM's completions arrive staggered
(graded, mostly single-prefill stalls). Candidate mechanisms for the stagger,
to be discriminated by the fix owner:

- **H-A (prime): vLLM's arm runs async scheduling ON by default** (confirmed
  in the binding server logs); overlap shifts admission one step relative to
  completion and can break the pair-lockstep. Our W3 stack (default OFF in the
  binding binary) was NOT active. → Once the W3 TTFT-admission fix lands
  (proof at `f086b64`: −5.4 ms/step TPOT, +36% TTFT admission bug being
  fixed), re-measure the tail structure under W3-on: W3 may fix these two
  axes as a side effect.
- **H-B: decode-first budget accounting.** With 32 decodes funded first only
  ~2016 tokens remain → the second prefill MUST chunk (992) and its completion
  staggers one step. If ours funds waiting prefills before/without the running
  decodes at wave boundary (or the wave leaves no decodes to fund), it packs
  2×1024. Discriminator: per-step composition log `{prefills, prefill_tokens,
  decodes, wall_ms}` — expect ours: 2 prefills/2048 tok/~860 ms.
- **H-C: partial-prefill policy** (`max_num_partial_prefills`,
  `long_prefill_token_threshold` defaults) differing in effect at wave
  boundaries.

MIRROR policy applies: the fix is whatever pinned vLLM actually does
(v0.25.0 `702f481`), not an invented cap.

## Disposition

- Fix owner: task #18 (blocked on the W3 scheduler-admission agent, which owns
  the sched files). First action: step-composition probe (H-B discriminator),
  then A/B under W3-on (H-A).
- Acceptance for closing the axes: interleaved c8+c32 rerun with per-event
  stall ≤1 prefill equivalents, both ratios ≥0.85, no regression to the
  passing means or the flat decode body.
