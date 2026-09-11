ID: ISSUE-GH-2164
Title: **gfx1100 decode is launch-bound, and kernel micro-optimization is exhausted as a lever.** The GFX1100-TG200 campaign swept 15 levers, adopted 11, and reached ~103 tok/s (~9.71 ms/tok) against a 200 tok/s target on Qwen3.5-4B-Q4_K_M / RX 7900 XTX. The discriminating result is a NEGATIVE one: T20's full-warp `KQuantGemvMmvqRow` rewrite is 2.38x-3.13x faster on large grids in microbenchmark, and a paired interleaved 5-rep engine A/B reads 92.9 vs 92.8 tok/s — a 0.1% wash — because the dominant Q4_K path runs at grid ~576 and is bound by fixed launch cost, not by the reduction barriers the rewrite removed, while the large-grid win lands on lm_head at one call per token (~0.04 ms/tok averaged). An earlier `rocprofv3` capture shows the mechanism directly: 97 standalone `QuantizeQ8KK` launches per token, EVERY one a single block (`m*nsb <= 128` at batch 1), mean duration 48.2-50.1 us FLAT with respect to K. The issue also separates two overhead terms the campaign's summary collapses: ~4.2 ms/tok is kernel time above the 4.38 ms/tok weight-read floor (occupancy and per-launch cost INSIDE kernels), and a further ~1.13 ms/tok is wall outside kernels entirely. Next levers are HIP graph capture ([#332](https://github.com/mudler/vllm.cpp/issues/332), which predicted this on gfx1200 from an explicitly unmeasured two-point fit and which this measures on gfx1100), a `SiluMulK` quant epilogue for the 40 of 97 launches the `RmsNorm` epilogue cannot absorb, then persistent kernels. Owed: the evidence is read from unmerged [#1936](https://github.com/mudler/vllm.cpp/pull/1936) at `b058bb752` and is NOT reproducible from `main`, so landing `docs/bench-evidence/gfx1100-tg200-*.md` and the campaign spec comes first, then a fresh capture with per-token dispatch counts to replace the budget-table arithmetic with a traced split
Row: BACKEND-ROCM
State: UNKNOWN
Kind: perf
GitHub: 2164
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:842`

### Frozen archive evidence

> | [#2164](https://github.com/mudler/vllm.cpp/issues/2164) | `BACKEND-ROCM` | **gfx1100 decode is launch-bound, and kernel micro-optimization is exhausted as a lever.** The GFX1100-TG200 campaign swept 15 levers, adopted 11, and reached ~103 tok/s (~9.71 ms/tok) against a 200 tok/s target on Qwen3.5-4B-Q4_K_M / RX 7900 XTX. The discriminating result is a NEGATIVE one: T20's full-warp `KQuantGemvMmvqRow` rewrite is 2.38x-3.13x faster on large grids in microbenchmark, and a paired interleaved 5-rep engine A/B reads 92.9 vs 92.8 tok/s — a 0.1% wash — because the dominant Q4_K path runs at grid ~576 and is bound by fixed launch cost, not by the reduction barriers the rewrite removed, while the large-grid win lands on lm_head at one call per token (~0.04 ms/tok averaged). An earlier `rocprofv3` capture shows the mechanism directly: 97 standalone `QuantizeQ8KK` launches per token, EVERY one a single block (`m*nsb <= 128` at batch 1), mean duration 48.2-50.1 us FLAT with respect to K. The issue also separates two overhead terms the campaign's summary collapses: ~4.2 ms/tok is kernel time above the 4.38 ms/tok weight-read floor (occupancy and per-launch cost INSIDE kernels), and a further ~1.13 ms/tok is wall outside kernels entirely. Next levers are HIP graph capture ([#332](https://github.com/mudler/vllm.cpp/issues/332), which predicted this on gfx1200 from an explicitly unmeasured two-point fit and which this measures on gfx1100), a `SiluMulK` quant epilogue for the 40 of 97 launches the `RmsNorm` epilogue cannot absorb, then persistent kernels. Owed: the evidence is read from unmerged [#1936](https://github.com/mudler/vllm.cpp/pull/1936) at `b058bb752` and is NOT reproducible from `main`, so landing `docs/bench-evidence/gfx1100-tg200-*.md` and the campaign spec comes first, then a fresh capture with per-token dispatch counts to replace the budget-table arithmetic with a traced split | perf |

## Resolution

-
