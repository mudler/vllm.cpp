ID: ISSUE-GH-1910
Title: KQuantGemmK strides 32 lanes over nsb=K/256 superblocks: half the warp idles on 75% of decode calls, and it is 54% of ROCm decode
Row: BACKEND-ROCM
State: CLOSED
Kind: perf
GitHub: 1910
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `KQuantGemmK` gives one warp each `(i,j)` output element and strides its 32 lanes
> over the row's superblocks:
>
> ```c
> // src/vt/rocm/rocm_grouped_gemm.hip:449
> for (int64_t sb = lane; sb < nsb; sb += 32) { ... }
> #pragma unroll
> for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffULL, partial, off);
> ```
>
> `nsb = K / 256`. On a 4096-wide model that is **16**, so **lanes 16..31 execute
> nothing** while the full 5-round reduction still runs. Half of every warp, on
> three quarters of decode calls.
>
> ## Measured, by instrumenting the launcher
>
> `4b1154bc5`, `Ornith-1.5-9B-Q4_K_M.gguf` (dense `qwen35`, 4096 hidden), RX 9060
> XT (gfx1200), ROCm 7.2.3. Every `MatmulBTQuantKernelRocm` dispatch logged with
> its shape; `m=1` rows are decode.
>
> | calls | n | k | nsb | fmt | out | idle lanes |
> |---:|---:|---:|---:|---|---|---:|
> | 128 | 12288 | 4096 | 16 | Q4_K | f32 | **16 of 32** |
> | 32 | 4096 | 12288 | 48 | Q6_K | bf16 | 0 |
> | 32 | 4096 | 12288 | 48 | Q4_K | bf16 | 0 |
> | 24 | 1024 | 4096 | 16 | Q4_K | bf16 | **16** |
> | 16 | 8192 | 4096 | 16 | Q4_K | bf16 | **16** |
> | 16 | 4096 | 4096 | 16 | Q4_K | bf16 | **16** |
> | 8 | 1024 | 4096 | 16 | Q6_K | bf16 | **16** |
> | 3 | **248320** | 4096 | 16 | Q6_K | bf16 | **16** |
>
> **195 of 259 decode calls (75%) run `nsb` = 16 against a 32-lane loop.**
>
> **Not every shape is affected**, and a fix must not regress the ones that are
> not: the two `k=12288` entries have `nsb` = 48 and pack all 32 lanes. The defect
> tracks `K`, and `K` = 4096 is simply what most projections in this model use.
>
> ## Cost
>
> Decode profile after [#1876](https://github.com/mudler/vllm.cpp/issues/1876)
> (the activation-quantizer decomposition) is applied, same method as
> [#1863](https://github.com/mudler/vllm.cpp/issues/1863) — `rocprofv3
> --kernel-trace --stats`, decode isolated by differencing `--max-tokens 4`
> against `--max-tokens 36` over 32 tokens:
>
> | ms/tok | share | calls | us/call | kernel |
> |---:|---:|---:|---:|---|
> | 12.457 | 29.5% | 21 | 593 | `KQuantGemmK<bf16,…>` |
> | 7.320 | 17.3% | 64 | 114 | `KQuantGemmK<f32,0>` |
> | 3.160 | 7.5% | 44 | 72 | `KQuantGemmK<bf16,…>` |
> | **22.937** | **54.3%** | **129** | | **`KQuantGemmK` total** |
> | 10.713 | 25.4% | 72 | 149 | `wvSplitKSml<1>` |
> | 42.227 | 100% | | | total decode GPU time |
>
> Against llama.cpp `b10451` HIP on the identical workload: `mul_mat_vec_q`
> totals **21.229 ms/tok over 217 calls** where our matmul family
> (`KQuantGemmK` + `wvSplitKSml`) totals **33.65**, a 1.585x deficit that is
> **12.4 ms of the 18.76 ms/tok remaining gap — about 66% of what is left** once
> #1876 lands.
>
> ## Same defect class as #1876
>
> #1876 was a decomposition written for prefill shapes that starved at `m` = 1:
> one thread per 256-element superblock, 16 threads on a 32-CU part, 95.5 us/call
> against llama.cpp's 1.6. Fixing the decomposition took it to 2.68 us/call and
> moved end-to-end decode 1.27x. This is the same shape of problem one kernel
> downstream, and the fix likely rhymes: choose the lane→work map from `nsb`
> rather than assuming it exceeds the wave width.
>
> ## What is NOT established
>
> - **The 593 us/call instantiation is not attributed.** `n` = 248320 (the lm_head,
>   one call per token, 20x wider than any other output) is the obvious candidate,
>   but dispatches were **not** correlated against the profiler's per-kernel rows.
>   That correlation should be done before anyone sizes the win, because it
>   decides whether the fix targets lm_head specifically or the general `nsb` = 16
>   path.
> - **No fix is proposed here, and no speed claim is made.** Whether the answer is
>   fewer lanes per superblock with more `(i,j)` per warp, a split-K reduction, or
>   something closer to llama.cpp's MMVQ decomposition is a design question this
>   issue does not settle.
> - The profile above was taken on the **unmerged** `row/ROCM-Q8K-QUANT-DECOMP`
>   branch (reviewed PASS, four findings unrepaired at the time of writing). On
>   current `main` the quantizer still costs 12.321 ms/tok, so `KQuantGemmK`'s
>   *share* is smaller there while its absolute cost is the same.
> - Host was not idle (loadavg 2.2-2.3, free VRAM asserted above 13 GiB with no
>   resident model process). Per-call kernel figures are the robust ones; treat
>   end-to-end numbers as indicative.
> - One model, one prompt, batch 1, gfx1200 only. The MoE path
>   (`MatmulBTQuantGroupedKernelRocm`, `:547`) uses the same kernel and was **not**
>   measured; [#1294](https://github.com/mudler/vllm.cpp/issues/1294) is where that
>   profile lives.
>
> ## Reproducing the shape dump
>
> Add to `MatmulBTQuantKernelRocm` before the `launch` lambda:
>
> ```c
> if (std::getenv("VT_DUMP_KQ_SHAPES")) {
>   fprintf(stderr, "KQSHAPE dense m=%lld n=%lld k=%lld nsb=%lld fmt=%d\n",
>           (long long)m, (long long)n, (long long)k, (long long)nsb, fmt);
> }
> ```
>
> then `VT_DUMP_KQ_SHAPES=1 vllm-cli --model <gguf> --max-tokens 3 --device auto`
> and `grep '^KQSHAPE dense m=1' | sort | uniq -c`. The instrumentation was
> reverted and is not in the tree.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

GitHub records closing pull request #2086 (https://github.com/mudler/vllm.cpp/pull/2086) merged on 2026-08-28 as commit `72f99c048b1be268801eb13e963d8f1bde95c594`. GitHub closed issue #1910 on 2026-08-28.
