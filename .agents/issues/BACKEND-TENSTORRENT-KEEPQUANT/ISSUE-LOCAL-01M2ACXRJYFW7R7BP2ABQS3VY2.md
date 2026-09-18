ID: ISSUE-LOCAL-01M2ACXRJYFW7R7BP2ABQS3VY2
Title: TT platform reports no device memory total: the MoE fit always refuses, so Qwen3.8-27B silently lands on CPU and never reaches the P150
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

device_memory_total_bytes is 0 on the Tenstorrent platform (no probe implemented), so ResolveMoeFitFromSizes always returns UNKNOWN-budget and places nothing: 'engine: device placement: --fit resolved NO placement' (w3-gate-final.log:9, also the 09:59 OOM run's stderr:13). Every qwen38-gguf-q4km-27b attempt today — including the one that OOM'd in tt-metal bank_manager — ran with weights on CPU; the VT_DUMP_IDS capture behind it is CPU evidence and was correctly refused by the gate's device failsafe (exit 77). W4d's fits-the-card premise was never exercised. Fix: probe total DRAM (bank size x banks) on the TT platform so the fit resolves against the real 32 GiB.

## Resolution

-
