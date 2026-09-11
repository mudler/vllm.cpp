ID: ISSUE-GH-2362
Title: ROCm: replace software Dp4a with hardware v_dot4_i32_iu8 in KQuantGemmK — 1.38-1.54x prefill speedup on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2362
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-30
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: BACKEND-ROCM
>
> ## Problem
>
> `KQuantGemmK` in `src/vt/rocm/rocm_grouped_gemm.hip` uses a software `Dp4a` implementation: 4 int8 multiplies + 4 adds in scalar instructions. A rocprofv3 trace on the optimized PP path (Qwen3.5-4B Q4_K_M, PP 228, RX 7900 XTX) shows `KQuantGemmK` at **93.5% of kernel time** (1939.6 ms of 2073.9 ms, 384 dispatches). The kernel is compute-bound on the dot product, not memory-bandwidth-bound.
>
> ## Fix
>
> Replace the software `Dp4a` function body with `__ockl_sdot4`, which generates the hardware `v_dot4_i32_iu8` instruction. One instruction instead of eight. Bit-identical: signed int8×int8→int32 dot product is exact in both forms.
>
> The change is 6 lines — the `Dp4a` function body only. No kernel structure change, no shared memory, no synchronization.
>
> ## Relationship to #2109
>
> #2109 tracks the MFMA tensor-core arm for `KQuantGemmK` — a larger effort porting llama.cpp's `mul_mat_q` MFMA path. This change is complementary: it optimizes the existing scalar kernel's dot product instruction without changing the kernel structure. The MFMA arm would supersede this optimization where it applies, but this change benefits all current `KQuantGemmK`/`Q8_0GemmK`/`GroupedKQ8K` callers immediately, including decode (m=1) where MFMA may not apply.
>
> ## Evidence
>
> A/B on Qwen3.5-4B Q4_K_M / RX 7900 XTX / ROCm 7.15, both builds clean with `--offload-arch=gfx1100`, 5 reps interleaved:
>
> | PP | Base TTFT (ms) | HW-Dp4a TTFT (ms) | Speedup | PT gain |
> |---|---|---|---|---|
> | 28 | 104.5 | 75.9 | 1.38x | 37.7% |
> | 64 | 222.7 | 158.1 | 1.41x | 40.9% |
> | 128 | 775.7 | 521.1 | 1.49x | 48.9% |
> | 228 | 744.2 | 484.2 | 1.54x | 53.7% |
> | 911 | 3441.9 | 2768.0 | 1.24x | 24.3% |
> | 1821 | 6665.4 | 5541.1 | 1.20x | 20.3% |
>
> Decode also benefits: TPOT 21.91ms → 20.71ms at PP=64, output 16.
>
> ## Correctness
>
> - `test_ops_quant_dot`: 28 cases, 210138 assertions, all pass.
> - `test_backend_cross_device` ("non-grouped keep-quant GEMM"): 4 NMSE assertions pass (Q8_0/Q4_K/Q5_K/Q6_K, all ≤ kNmseTol).
> - Token-exact: identical output tokens vs baseline at seed=42, temperature=0.
>

## Resolution

-
