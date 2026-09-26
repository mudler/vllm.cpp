ID: ISSUE-LOCAL-01M3F7GQ4MXQT72D05A12JD9EQ
Title: Decompose the 27B captured step: where do the 35 s/token go
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

After the warmup redesign (PR #3321) landed the live capture system, the 27B captured arm measures ~35 s/token (0.02 tok/s) at the anchor — undecomposed. The number may contain per-step warmup passes, capture-invocation costs, replay launch overhead, device wait, near-OOM pressure from the ttnn deferred-reader retention (tt-metal#57970, still capping the process at ~5 requests), JIT, or genuine kernel time. Every performance lever (INT8DOT re-measure, M=2 batching, launch-overhead fixes, capture economics) needs the ranked cost table before its expected value is computable; the old TPOT numbers (1253/642 ms) were measured on the silently all-eager engine and are void. Spec: tenstorrent-27b-step-decompose.md — the W1 GEMV phase-attribution pattern extended to the full captured step.

## Resolution

-
