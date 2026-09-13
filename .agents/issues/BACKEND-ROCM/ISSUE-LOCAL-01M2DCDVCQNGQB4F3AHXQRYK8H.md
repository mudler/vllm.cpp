ID: ISSUE-LOCAL-01M2DCDVCQNGQB4F3AHXQRYK8H
Title: No per-kernel decode-time attribution exists on ROCm, so gfx1151 decode cost is unattributable; #3040 is read as a blanket refusal of rocprofv3 timestamps when its own committed capture is internally consistent
Row: BACKEND-ROCM
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

We own no per-kernel ROCm timing instrument. `VT_OP_PROVIDER_STATS` (`rocm_backend.hip:460`) counts selections, not time. On CUDA the equivalent exists and works -- nsys ranked `HcGroupedNormKernel` at 40.7% of kernel time on `dgx:gpu0`, which became a scoped optimization row. On ROCm nothing ranks anything, so `cc0e827dd` had to close the qwen4_exp placement question with "where our decode step time goes is unestablished". The blocker of record is [#3040](https://github.com/mudler/vllm.cpp/issues/3040), whose evidence README says "Do not derive kernel durations, time shares, or host idle bounds from these timestamps". That sentence is broader than the measurement under it: the committed kernel-only capture `docs/bench-evidence/strix-kernel-trace-3015-20260907/diag-kernel-control-asaj36ta--trace--e5367aefafa8--1_kernel_trace.csv.gz` carries 85,737 dispatch rows with ZERO negative and ZERO zero-length durations, a kernel-busy sum of 11.749 s against a 12.551 s wall span (93.6%), and 62 swap warnings -- 0.072% of rows. Needed: an instrument that produces a ranked table (name, total, count, mean, min, max, share) over DECODE-ONLY windows, states its own distortion, and bounds the residual #3040 risk instead of inheriting a blanket refusal.

## Resolution

-
