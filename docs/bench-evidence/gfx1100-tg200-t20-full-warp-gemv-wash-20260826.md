# T20: full-warp cooperative KQuantGemvMmvqRow — closed not-adopted (engine wash)

Date: 2026-08-26
Branch: `row/GFX1100-TG200` head `96c523d9` (T18 baseline)
Model: Qwen3.5-4B-Q4_K_M, d_model=2560, 32 layers (8 full-attn / 24 SSM)

## Hypothesis

The T18 `KQuantGemvMmvqRow` uses 8 lanes per super-block × 4 super-blocks per
pass, with 3 `__shfl_down` reduction barriers per pass. The barriers prevent
the compiler from pipelining memory loads across super-blocks, leaving memory
latency unhidden. Replacing the scheme with full-warp cooperation (all 32
threads on one super-block, per-thread float accumulation, single
`warp_reduce_sum`) eliminates the intermediate barriers and lets the GPU
overlap weight reads from multiple super-blocks.

## Implementation

Rewrote `KQuantGemvMmvqRow` in `src/vt/rocm/rocm_grouped_gemm.hip`:
- 32 threads per super-block (sub-block c=lane>>2, quarter q2=lane&3)
- Each thread handles 8 elements via 2 `amd_mixed_dot` iterations
- Per-thread float accumulation: `d*scale*sub - dmin*mn*pre` per super-block
- Single `warp_reduce_sum` at end (zero intermediate barriers)
- Q6_K scale selection: q2<2 uses `sc[2c]`, q2>=2 uses `sc[2c+1]`
- Min correction without pre-computed bsums: `amd_mixed_dot` with `0x01010101`

Correctness: integer core (v_dot4 dot products, q8 sums) is exact under any
association. Float association differs (per-thread-per-sb vs per-sb-after-
octet-reduce), so ULP differences expected. NMSE within 1e-6 oracle band.

## Microbenchmark results (test_rocm_quant_dot timing test)

| Grid shape | OFF µs | ON µs | Ratio | Speedup |
|---|---|---|---|---|
| 320×2560 Q6_K | 74.5 | 55.8 | 0.75x | 1.33x |
| 320×2560 Q4_K | 60.6 | 58.6 | 0.97x | 1.03x |
| 2304×2560 Q4_K | 73.1 | 72.6 | 0.99x | 1.01x |
| 31040×4096 Q6_K | 449.0 | 188.2 | 0.42x | 2.38x |
| 248320×2560 Q6_K | 2133.7 | 681.7 | 0.32x | 3.13x |

The kernel speedup scales with grid size: 1.01x on small Q4_K grids, 3.13x on
large Q6_K grids. The large-grid win is real — eliminating barriers lets the
GPU pipeline memory loads across super-blocks.

## Engine A/B (acceptance workload)

Paired interleaved A/B, 5 reps, 256 tokens, greedy, full campaign config
(12 flags), CLI entry point. T20 (ON) vs T18 (OFF) by reverting kernel file
to `96c523d9` and rebuilding.

| Pair | ON tok/s | OFF tok/s |
|---|---|---|
| 1 | 85.940 | 92.744 |
| 2 | 92.927 | 92.854 |
| 3 | 92.996 | 92.763 |
| 4 | 92.761 | 92.780 |
| 5 | 92.758 | 92.768 |

ON median: 92.9 tok/s. OFF median: 92.8 tok/s. **Wash** (+0.1%, within noise).

Body coherence: ON rep 1 produced a different (coherent) continuation due to
float association change. ON reps 2-5 byte-identical to OFF. Acceptable per
near-tie doctrine.

## Why the kernel win didn't reach the engine

The dominant Q4_K path (2.46 ms/token, 25% of wall) has small grids
(`ffn_gate` and `ffn_up` at about 288 super-blocks per row, grid about 576).
At small grids the kernel is launch-overhead-bound, not
reduction-barrier-bound. Removing barriers has no effect. The Q6_K path
(1.20 ms/token) is mostly small-grid `ffn_down` with 22 calls per token,
where T20 gives 1.03x.

The large-grid timing row is a standalone microbenchmark, not an engine
amortization. Its OFF-to-ON difference is
`2,133.7 - 681.7 = 1,452.0 µs`, or `1.452 ms/token`, because the production
census records one lm_head dispatch per decode token. A 256-token engine run
therefore makes 256 calls. It does not divide one call's saving by 256.
The pre-T20 T7 production trace recorded the same lm_head geometry at about
615 µs/call. That production result is already below both the standalone OFF
time and its implied saving. The standalone OFF timing therefore does not
describe the production dispatch. The five-pair engine A/B directly
establishes the wash, but these measurements do not attribute the
microbenchmark-to-engine difference.

## Conclusion

T20 closed not-adopted. The kernel-level optimization is correct and effective
on large grids, but the engine's dominant cost is small-grid Q4_K GEMV at
2.46 ms/tok, which is launch-overhead-bound. The path to 200 tok/s requires
reducing launch overhead (HIP graph capture, kernel fusion, or persistent
kernels), not further micro-optimizing individual kernel internals.

Failed-attempt ledger: 8 of 15 closed-not-adopted.
