ID: ISSUE-GH-3009
Title: perf(BACKEND-ROCM): non-greedy decode 22% slower than greedy — single-block sampling kernels underutilize 96 CUs
Row: BACKEND-ROCM
State: CLOSED
Kind: UNKNOWN
GitHub: 3009
Mirror: SYNCED
Availability: FULL
Created: 2026-09-06
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ## Problem
>
> Non-greedy decode (temperature + top-p + softmax + Gumbel-max sample) on ROCm runs **22% slower** than greedy decode at single-request batch size. On a Qwen3.5-4B Q4_K_M model (vocab ~152K) with fp8 KV cache on an RX 7900 XTX (gfx1100, 96 CUs):
>
> | Path | tok/s | ms/tok |
> |---|---|---|
> | Greedy (temp=0) | 97.8 | 10.2 |
> | Non-greedy (temp=0.7, top_p=0.9) | 75.6 | 13.2 |
> | **Sampling overhead** | — | **3.0 ms/tok** |
>
> ## Root cause
>
> Three per-row sampling kernels in `src/vt/rocm/rocm_sample.hip` launch as `<<<1, 256>>>` — one block of 256 threads (4 wavefronts) on **1 CU out of 96**, scanning the full ~152K vocab:
>
> | Kernel | Avg µs/call | Bottleneck |
> |---|---|---|
> | `ApplyTopKTopPRowK` | 1270 | Ternary search, up to 64 iterations over full vocab |
> | `RandomSampleK` | 1665 | `GumbelScore` → `ExpNoise` computes `log(u)` in double precision per element — compute-bound, 148 serial evaluations per thread |
> | `SoftmaxK` | 406 | 3-pass reduce (max, sum, normalize) |
> | **Total** | **3341** | 3.3 ms/tok on 1 CU |
>
> The greedy `ArgmaxK` has the same single-block shape but is cheaper per element (one compare vs one `log`), so the gap is specific to the non-greedy path.
>
> ## Evidence
>
> rocprofv3 kernel trace, 32 decode tokens, Qwen3.5-4B Q4_K_M, fp8 KV, gfx1100:
>
> ```
> ApplyTemperatureK   32 calls  avg=     3.8 us
> ApplyTopKTopPRowK   32 calls  avg=  1270.0 us
> SoftmaxK            32 calls  avg=   406.0 us
> RandomSampleK       32 calls  avg=  1665.0 us
> ```
>
> The CUDA backend (`src/vt/cuda/cuda_sample.cu`) already has a multi-block split-phase greedy argmax (`ArgBlocksPerRow`, `ArgmaxPartialKernel` + `ArgmaxFinalKernel`), but the ROCm random sample has no equivalent.
>
> ## Proposed fix
>
> Two changes, both in `src/vt/rocm/rocm_sample.hip`:
>
> 1. **Widen per-row sampling block from 256 to 1024 threads** (`kVocabBlock = 1024`). 16 wavefronts vs 4 gives 4× more latency hiding on the same CU. The `BlockRed*` helpers use `blockDim.x` instead of the compile-time constant. Top-p drops 5.5×, softmax 3.8×.
>
> 2. **Split-phase random sample** (`VT_SAMPLE_SPLIT=1`, default ON). Spread the Gumbel-score argmax across `kSampleSplitBlocks = 128` blocks (all 96 CUs), each thread handling 1–2 elements instead of 148. Phase A writes per-block `(score, index)` partials to scratch; Phase B reduces them. `ArgReduce` is associative and order-independent (same property the greedy split relies on), so the result is **bit-identical** to the single-block kernel.
>
> ## Measured result
>
> | Path | Before | After | Change |
> |---|---|---|---|
> | Greedy | 97.2 tok/s | 97.8 tok/s | unchanged |
> | Non-greedy | 75.6 tok/s | 96.5 tok/s | **+27.6%** |
> | Gap | 22% | 3.4% | |
>
> ## Correctness
>
> - `test_ops_sample`: 29/29 pass (2 CUDA-only skipped), 278K assertions
> - `test_sampler`: 21/21 pass
> - A/B test: 20 seed/temperature combinations (seeds 1–12345, temps 0.3–1.5) produce **bit-identical** output with `VT_SAMPLE_SPLIT=0` vs `VT_SAMPLE_SPLIT=1`
> - Determinism: 3 consecutive runs with same seed produce identical output
> - Full test suite: 225/225 pass, 3 consecutive runs, zero failures

## Resolution

Fixed by PR #3010: sampling block widen and split-phase implementation.
