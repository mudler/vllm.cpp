# Spec: attribute the 27B decode DRAM exhaustion before fixing it

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-24).
Issue: `ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW`.
Git integration: one pull request (spec + measurement + evidence), branch
`row/TT-27B-DRAM-GROWTH`.

## Problem

The APEX-I-Nano 27B decode exhausts tt-metal DRAM on the P150:
`bank_manager.cpp:495` refuses a ~268 MB allocation against a ~63 MB
largest free block, in the decode path `Qwen3_5DenseDecodeGraph::Step` →
`DecodeKeepQuantWordsF32` → `ttnn::where`. The failure is
**growth-with-requests**: a 64-request process dies after ~60 requests,
16-request after ~14, 8-request after ~7 — deaths scale with REQUESTS
SERVED, not with concurrency or steps-per-request, pointing at a per-
request accumulation of ~4-5 MB retained each. Steady-state free DRAM is
already only ~63 MB before the growth, so the exhaustion margin is thin
even at load.

This blocks: the INT8DOT ≥64-prompt band sweep (spec
`tenstorrent-keepquant-int8dot-default.md` gate 1), the 27B Q4KM sibling
gate, and the 27B trace-capture row. The allocation-trace instruments
this spec needs now live in their own module
(`tenstorrent_capture.cpp`: `AllocTraceSnapshot`, `FreeDeviceDramBytesForTest`,
`DumpSlotCensus`, `ResetAllocTraceForTest` — the W4d wave).

## Scope

W1 — attribute, don't fix (the W7 staging-elimination precedent: measure
first):

1. **Per-request DRAM ledger.** Drive the 27B decode with
   `VT_TT_ALLOC_TRACE=1` (the existing instrument), taking an
   `AllocTraceSnapshot` at: load-complete, after each request's first
   decode step, and every K requests (K=8) until near-exhaustion. Record:
   free-DRAM total, largest-free-block, snapshot delta, slot census.
2. **Attribution classes.** For the growing region, identify the
   allocation site class via the trace's existing tagging:
   - per-request tensors held and never released (keyed caches —
     candidates: keepquant words cache, decode-ids/shadow caches, any
     map keyed by request count or uid),
   - per-STEP allocations that accumulate (unbalanced alloc/free inside
     the captured decode step),
   - fragmentation (total free stays high while largest-free-block
     shrinks — bank-level fragmentation, a different fix entirely).
3. **A committed evidence file**: the per-snapshot ledger, the growth
   rate (MB/request), the attributed class with `file:line` anchors, and
   the reproduction recipe. The interpretation table below binds which
   follow-up row owns the fix; nothing in this change fixes anything.

## Interpretation (binds the follow-up)

- **~constant MB/request retained, traced to a keyed cache** → the fix is
  a bound/eviction on that cache (the `#1486` never-destroyed pattern is
  the usual suspect) — its own row.
- **growth inside the decode step** → a per-step unbalanced alloc in the
  capture/keepquant path — its own row.
- **largest-free-block collapses while total free stays high** →
  bank fragmentation; the fix is allocation-layout or pool-residency
  work, and the 63 MB steady-state margin question becomes a row of its
  own.
- **the trace shows NO growth** (deaths without ledger growth) → the
  exhaustion is upstream of our allocations (tt-metal bookkeeping); the
  row escalates to a pinned-tt-metal defect with the ledger as evidence.

## Gates

- Read-only instruments: the trace knobs change no numerics; the e2e
  anchor tokens are byte-identical with tracing on/off.
- The ledger numbers reproduce: two legs, same binary, deltas within
  noise; the death request-count per shape reproduces within ±10%.
- The evidence file states host, build SHA, tt-metal revision, model
  artifact, and contention state — the INT8DOT session's margins moved
  with the tt-metal revision, so the revision is part of the result.

## Risks

- The known ~20 min/process JIT non-amortization makes long runs slow;
  the census runs to NEAR-exhaustion, not to death, so legs stay bounded.
- Attribution may land in tt-metal's own allocator rather than our code;
  that is still a valid row outcome (escalation with evidence).

## Non-goals

- No fix, no eviction policy, no tt-metal change on this branch.
- No INT8DOT re-run here; the sweep reopens when this row lands.

## Stop conditions

- If the instruments cannot see the growing region (snapshots flat while
  the OOM still occurs), stop and escalate to tt-metal's profiler with
  the ledger — that outcome redirects the row, it does not fail it.
