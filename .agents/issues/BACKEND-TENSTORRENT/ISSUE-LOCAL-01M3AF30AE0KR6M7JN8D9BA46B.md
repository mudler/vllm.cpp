ID: ISSUE-LOCAL-01M3AF30AE0KR6M7JN8D9BA46B
Title: batch>1 GDN prefill asserts at load on current tt-metal (BH 480/192 vs 110 cores)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

On tt-metal 81f3bbf3b405 the 27B APEX decode asserts at LOAD for any concurrency above 1: chunk_gdn_phased_program_factory.cpp:137 requires BH <= ncores and the GDN prefill shape wants 480/192 against 110 cores. Concurrency 16 and 4 both died before serving a request (DRAM-ledger session, docs/bench-evidence/tt-27b-dram-ledger-20260924.md); only c=1 runs. This makes M>1 batching unreachable on the GDN decode lane at this revision and blocks the +43% M=2 batching result from reproducing. The assert is either a tt-metal program-factory grid-shape regression or a caller-side shape choice that needs the multi-core decomposition the older revision had.

## Resolution

-
