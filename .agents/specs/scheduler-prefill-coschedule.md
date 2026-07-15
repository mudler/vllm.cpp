# Scheduler prefill co-schedule parity — c2 TTFT-family void verdict

**Row:** `ENG-SCHED-CORE` (engine-matrix) · **Component:** the `KERNEL-GDN-PACKED-DECODE`
c2/c16 online component (owned by `CLAIM-GDN-BA-ROUNDING-1`). This spec is a
grounding/verdict spike, not an implementation plan: it resolves whether the
serving-gate component's c2 TTFT-family instability is a **scheduler divergence**
from pinned vLLM (an unmet mirror obligation to fix) or **arrival phasing** (a
microbenchmark lottery the harness must treat statistically).

## Verdict

**No co-scheduling divergence. Do NOT change the scheduler.** Our V1 scheduler's
waiting-queue admission and token-budget accounting is a faithful 1:1 mirror of
pinned vLLM's (both pins: `e24d1b24` and the v0.25.0 target `702f481`). Both
schedulers co-schedule two 1024-token prefills that fit the 2048-token budget
into a **single** step; both "serialize" only when the second request is not yet
in the waiting queue at the `schedule()` call (i.e. it arrived after the first
was already dispatched). That decision is identical logic on both sides, so the
component's leg-to-leg 3/3-vs-6/0 TTFT mixture is **arrival phasing**, not a
policy difference.

## Measured evidence (three sealed component roots, dgx, read-only)

27B NVFP4, closed-loop, in1024/out128, greedy, `--max-num-seqs 32
--max-num-batched-tokens 2048`, chunked-prefill era, concurrency 2 (6 requests).
Per-request TTFTs are BIMODAL: ~0.45 s (a 1024-prefill runs alone in its own
step) vs ~0.9 s (the request's prefill lands one step later, co-scheduled with
the other request's decode, or both co-schedule into one 2048 forward). Typical
legs alternate `[0.5,0.9,0.5,0.9,0.5,0.9]`; one leg (run 3) was all-`0.9`. Two
1024 prefills fit the 2048 budget EXACTLY, so a budget-filling scheduler
co-schedules them. Roots (immutable):
`~/work/vllm.cpp-gdn-packed-component/{c172336…, c172336…-r2, d19e0916…}`.
Client-side corroboration: the c2 preflight `first_chunk_s` samples are
bimodal — most ~0.48–0.52 s with elevated first-request samples (0.954, 1.29,
0.58 s). Server-side step-composition logs do NOT exist (VT_GDN_DIAG was off in
component runs; `driver.log` is a one-line status JSON), so step composition is
established from the code trace below.

## Upstream chain (both sides, file:line)

Arithmetic for the c2 arrival pattern (`token_budget = max_num_batched_tokens =
2048`, `long_prefill_token_threshold = 0`, `max_num_partial_prefills = 1`,
`enable_chunked_prefill = true`, no prefix reuse across distinct prompts):

Both-in-queue (both reqs enqueued before `schedule()`): waiting loop admits A
(`num_new = 1024`, budget 2048→1024), then B (`num_new = min(1024,1024) = 1024`,
budget 1024→0) → **both prefills in one 2048 forward** → both TTFT ≈ 0.9 s.

Staggered (B enqueued after A dispatched): step S prefills A alone (budget
2048→1024, half unused) → A TTFT ≈ 0.45 s; step S+1 co-schedules A's 1-token
decode + B's full 1024 prefill (1+1024 ≤ 2048) → B TTFT ≈ 0.9 s. The late
prefill is **co-scheduled, not refused**.

Neither side has a decode-budget reservation, a one-waiting-request-per-step
rule, or an active partial-prefill cap. Both are pure "fill the token budget
FCFS while `token_budget > 0` and `running < max_num_seqs`" loops.

| Element | Ours | Pinned vLLM |
|---|---|---|
| Waiting loop | `src/vllm/v1/core/sched/scheduler.cpp:234-298` | `vllm/v1/core/sched/scheduler.py:640-1013` |
| Budget init (`= max_num_scheduled_tokens`) | `scheduler.cpp:127` | `scheduler.py:416` |
| `max_num_seqs` break | `scheduler.cpp:235` | `scheduler.py:643-645` |
| `num_new = num_tokens - num_computed` | `scheduler.cpp:259` | `scheduler.py:810` |
| long-prefill cap (threshold 0 → inert) | `scheduler.cpp:260-263` | `scheduler.py:828-830` |
| chunked-disabled break (chunked ON → skipped) | `scheduler.cpp:265-267` | `scheduler.py:834-840` |
| `num_new = min(num_new, token_budget)` | `scheduler.cpp:268` | `scheduler.py:842` |
| allocate + `token_budget -= num_new` | `scheduler.cpp:271,295` | `scheduler.py:905-917,989` |
| Running loop (decode co-schedules with late prefill) | `scheduler.cpp:140-229` | `scheduler.py:442-591` |
| Config defaults | `src/vllm/config/scheduler.cpp:40,46-47,73-74` | `vllm/config/scheduler.py:70,80,257-259` |
| Engine input drain before `schedule()` | `src/vllm/v1/engine/core_proc.cpp:62-89` | `vllm/v1/engine/core.py:1269-1298` |

The engine busy loop drains **all** pending input items before `schedule()` on
both sides, so there is no systematic bias making us co-schedule less often. The
only thing that decides co-schedule vs staggered is whether both requests are
enqueued when `schedule()` fires — arrival phasing, identical on both sides. A
deterministic code difference would bias every leg the same way; the observed
leg-to-leg FLIP is the fingerprint of timing jitter, not policy.

## Tests to port / added

`tests/vllm/v1/test_scheduler.cpp:205` — "two budget-filling prefills
co-schedule (c2 parity)": two 1024-token prompts, budget 2048, both fully
scheduled in ONE step (each 1024, total 2048, both `scheduled_new_reqs`, waiting
drains, neither chunked). Mirrors `scheduler.py:640-1013` and the budget-fill
semantics of upstream `tests/v1/core/test_scheduler.py::test_schedule` (line 86).
`tests/vllm/v1/test_scheduler.cpp:241` — "a late prefill co-schedules with a
running decode": documents the arrival-phasing mechanism (A prefills alone, then
A's decode + B's full prefill co-schedule). **GREEN** on the current tree (would
be RED only if we serialized budget-filling prefills — we do not). Full
`test_scheduler` 31/31 (261 assertions); scheduler-family suites green.

## Harness recommendation (for the orchestrator — NOT implemented here)

Because the c2 TTFT-family (mean, median, AND tails) swings with the 3/3-vs-6/0
co-schedule mixture that flips leg-to-leg while every throughput/TPOT axis is
stable ≤0.5%, per-run TTFT-family stability at c2 is a Bernoulli lottery on
arrival phase and cannot be gated per-run at 4%. The tail-only 15% relaxation
already landed (`tools/bench/gdn_packed_component.py`) covers p90/p99 but the
c2 **mean/median** TTFT can still flip 4–24% between legs. Recommended (choose
one, statistically grounded, no scheduler change):

1. **Pool the c2 TTFT-family across the 3 reps** (18 samples) and gate the
   pooled distribution, not per-run — the mixture averages out across reps.
2. **Drop per-run c2 TTFT-family stability** entirely (keep throughput/TPOT/ITL
   and memory per-run stability), since the c2 TTFT split is arrival-phasing
   noise that is present identically in vLLM.

Honesty note on the "TTFT win": the binding grid's c2 mean TTFT ≈ 697 ms ours vs
≈ 833 ms vLLM (1.196×) is a **lottery artifact** — we happen to catch more
0.45 s "alone" prefills — not a durable, reproducible advantage. It will swing
leg-to-leg and must not be counted as a binding TTFT edge; the pooled-across-reps
comparison is the honest denominator. Since the scheduler is unchanged, the
expected TTFT profile is UNCHANGED (bimodal, arrival-phased).

## Run-3 void linkage

This verdict explains the three `complete-void` TTFT-family voids at
`c172336`, `c172336…-r2`, and `d19e0916` (run 3, the all-`0.9` leg): all three
are the same arrival-phasing co-schedule lottery, not a regression and not a
scheduler divergence. The tail-stability relaxation is therefore the correct
treatment; extending it to the c2 mean/median TTFT-family per the recommendation
above removes the remaining false-void surface.

## Gates (CPU only; the orchestrator owns any GPU rerun)

`test_scheduler` 31/31, `test_scheduler_config` 7/7, `test_sched_output` 8/8,
`test_request_queue` 26/26; full tools; clean `-Werror` rebuild; record checkers
+ doc-checkpoint. No scheduler code changed; no GPU work.

## Risks/decisions

vLLM-defined behavior is not reopened — the scheduler mirrors upstream and stays
as-is. The only open item is a harness statistics call for the orchestrator
(above); it is a benchmark-methodology decision, not a product/scope call.
