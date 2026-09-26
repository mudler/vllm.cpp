ID: ISSUE-LOCAL-01M3F7GQ4MXQT72D05A12JD9EQ
Title: Decompose the 27B captured step: where do the 35 s/token go
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: 2026-09-26

## Problem

After the warmup redesign (PR #3321) landed the live capture system, the 27B captured arm measures ~35 s/token (0.02 tok/s) at the anchor — undecomposed. The number may contain per-step warmup passes, capture-invocation costs, replay launch overhead, device wait, near-OOM pressure from the ttnn deferred-reader retention (tt-metal#57970, still capping the process at ~5 requests), JIT, or genuine kernel time. Every performance lever (INT8DOT re-measure, M=2 batching, launch-overhead fixes, capture economics) needs the ranked cost table before its expected value is computable; the old TPOT numbers (1253/642 ms) were measured on the silently all-eager engine and are void. Spec: tenstorrent-27b-step-decompose.md — the W1 GEMV phase-attribution pattern extended to the full captured step.

## Resolution

2026-09-26 (row TT-27B-STEP-DECOMPOSE, branch row/TT-27B-STEP-DECOMPOSE, instrument commit 46cd3e4b5 + the evidence record commit, P150a under the file mutex, pinned tt-metal vllm-cpp-pin/20260925): the ranked cost table is measured and recorded in docs/bench-evidence/tt-27b-step-decompose-20260926.md. The ~35 s/token decomposes (steady request, 3 decode steps, 104.0 s, sum check 100% at the instrument level): 64.4% the eager cold body (67.0 s/request, host dispatch of the S=1 forward — the device work is 1.27 s: the VT_TT_STEP_PHASES=sync probe measured gdn 20.3 ms / fa 18.6 ms per layer), 29.9% the trace-execution wait (31.1 s — the trace replays ~1.3 s of kernels in 31.1 s), 5.5% the capture pass (record ~2.6 s + finalize ~1.7 s), <0.1% the per-step warms/embed/replay-launch. NOT a term: retention pressure (per-request entry DRAM flat, >5 GiB free at request 4; the staircase does not step in this arm), JIT (nothing after request 1), and genuine kernel compute (~1.3 s, under 4%). The dominant term's mechanism: the #2469 continuation port to the dense driver is missing its two post-replay ++s.expected_cur_pos increments (qwen3.cpp:971/:1118 vs the dense driver's qwen3_5.cpp:12400 update and :12395 check), so boundary=1 fires at EVERY decode step, s.graph.Reset() destroys the fresh trace before it can serve, and two of three steps re-warm eagerly — VT_DECODE_GRAPH_STATS reports 4 replays for a 12-step run, every one a capture's own launch. Gates: knobs-unset anchor tokens byte-identical across all legs (sha256 13c3f70b...), two instrumented legs reproducible within noise (trace wait 31.107-31.126 s), components sum to the wall (104.0 vs 104.0 s; bench TPOT within 4.5%). The follow-up ISSUE-LOCAL-01M3FF1DJA10C7DBY17QGZACSV owns the port completion (the capture-path change this row's non-goals forbid); the next perf row the table binds is the per-op dispatch overhead (~30 ms/op over the 1,037-op captured forward, host-eager and trace-replay alike; the kernels are 1.3 s).
