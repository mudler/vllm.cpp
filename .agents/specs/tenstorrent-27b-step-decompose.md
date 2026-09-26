# Spec: decompose the 27B captured step — where do the 35 seconds go

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-26).
Issue: files against the row that owns the 27B captured-arm economics.
Follow-up to: `tenstorrent-gemv-overhead-trace.md` (the W1 phase-
attribution pattern this row extends), `tenstorrent-capture-warmup-
redesign.md` (the live capture system this measures).
Git integration: one pull request (spec + instruments + the decompose
measurement), branch `row/TT-27B-STEP-DECOMPOSE`.

## Problem

After the warmup redesign landed the live capture system (PR #3321), the
27B APEX captured arm measures **~35 s/token (0.02 tok/s)** at the
anchor — but that number is undecomposed. It may contain: per-step
warmup passes, capture-invocation costs, replay launch overhead, device
wait, near-OOM pressure from the ttnn deferred-reader retention
(tt-metal#57970 — the ~950 MB/request staircase still caps the process
at ~5 requests), JIT, or genuine kernel time. Every performance lever
(INT8DOT re-measurement, M=2 batching, launch-overhead fixes, capture
economics) needs this table before its expected value is computable.
The old TPOT numbers (1253/642 ms) were measured on the silently
all-eager engine and are void.

## Scope

W0 — attribute, don't optimize:

1. **Phase timestamps around the captured step's structure** (the W1
   GEMV pattern extended): host-side timestamps bracketing (a) the
   warmup path (does each step re-warm? the redesign serves commits —
   count warm passes per step), (b) capture-begin/end, (c) replay
   launch, (d) completion wait. Opt-in env knob (zero-cost when unset),
   stderr per-step lines in the TT trace-knob style.
2. **The existing instruments read out**: `TT_LEDGER` allocator deltas
   per step (is the retention still stepping?), `VT_DECODE_GRAPH_STATS`
   (capture/replay accounting), `VT_TT_TRACE_DEBUG` (per-op entries),
   the alloc-trace's per-call classes.
3. **The retention-pressure axis**: one leg with the process at request
   1 (fresh, no pressure) vs request 4 (near the ceiling) — the same
   step's phase table on both isolates the near-OOM component.
4. **Per-layer sampling**: at the dominant phase, one sampling probe
   over the 64 layers (the GDN microbench pattern, `TT_GDN_BENCH`
   precedent) — the layer-level cost distribution.

## Deliverable

A dated benchmark-record section with the ranked cost table (ms per
component per step, both pressure legs), the dominant term named with
its mechanism, and the lever the table points at (which of: warmup
economics, capture economics, launch overhead, retention pressure, or
genuine compute). Interpretation binds the follow-up: the biggest term
names the next perf row.

## Gates

- Read-only instruments: with the knobs unset the anchor tokens are
  byte-identical (the W1 GEMV spec's gate, verbatim).
- Reproducible: two legs, same binary, phase deltas within noise; the
  retention state recorded per leg.
- The phase table's components sum to the measured wall (±10%) — no
  hidden residue.

## Risks

- The retention may OOM a leg before enough steps: the table reports
  what the process lifetime allows (request-1 fresh leg is the floor).
- A dominant term that is tt-metal-internal (e.g. replay launch inside
  ttnn) is still a valid outcome — it joins tt-metal#57970's lane.

## Non-goals

- No optimization, no kernel change, no capture-path change.
- No TPOT claims beyond the decomposed table.

## Stop conditions

- If the instruments perturb the measurement (tokens shift with knobs
  set), stop and redesign the instrumentation (the W1 stop condition).
- If the captured step cannot be bracketed at the 27B (the structure
  resists the phase pattern), report the closest decomposition with its
  limits stated — do not force.
