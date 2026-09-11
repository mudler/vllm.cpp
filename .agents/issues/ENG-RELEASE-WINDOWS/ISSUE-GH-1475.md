ID: ISSUE-GH-1475
Title: CUDA 13.2 on Windows rejects sampler infinity sentinels during DFlash2 W3 compilation
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 1475
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `SPEC-DFLASH2` W3 first compiled its CUDA arms on the local RTX 3090 host with CUDA 13.2 and Visual Studio 2022. The SM86 build fails in `src/vt/cuda/cuda_sample.cu:32` because CUDA 13.2/MSVC expands `INFINITY` to a cast of `1e+300`, then rejects the value as not representable by `float`.
>
> Exact failing configuration:
>
> ```text
> cmake -S . -B build-dflash2-01-w3-sm86 -G "Visual Studio 17 2022" -A x64 -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=86 -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_BUILD_TESTS=ON "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"
> cmake --build build-dflash2-01-w3-sm86 --config Release --target test_ops_topk_values_indices --parallel 1
> ```
>
> Observed error:
>
> ```text
> cuda_sample.cu(32): error: floating-point value does not fit in required floating-point type
> constexpr float kNegInf = -((float)(1e+300));
> ```
>
> The same expansion produces warnings at the positive-infinity sentinels used by the pivot-bracket search. This blocks the new `TopKValuesIndices` CUDA arm and the existing sampler translation unit on native Windows.
>
> Owner: `SPEC-DFLASH2` W3 repair flow. The repair must preserve negative-infinity semantics, add an executing regression, run the focused selector/top-k CUDA parity tests on SM86, and keep Linux behavior unchanged.

## Resolution

-
