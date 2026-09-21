ID: ISSUE-LOCAL-01M2T6Q2GSRRV0G6PTFJSJNSP4
Title: ROCm 7.1 build fails: hipDeviceAttributeGcnArch and -Werror unused variables in split-KV reduce
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: 2026-09-18

## Problem

Building with -DVLLM_CPP_HIP=ON on ROCm HIP 7.1.52802 (Fedora, clang 20) fails twice. src/vt/rocm/rocm_skinny_gemm.hip calls hipDeviceGetAttribute with hipDeviceAttributeGcnArch, which HIP 7.1 headers no longer declare, and it calls a HIP runtime function from device code. src/vt/rocm/rocm_paged_attn.hip declares 'g' and 'm_split' in PagedAttnDecodeSplitKvReduce (issue #845) without using them, which -Werror rejects. Both stop the build for every architecture, including gfx1102 and gfx1103.

## Resolution

2026-09-18: on_gfx1151(), YtileUnrl and SelectYtileUnrl are removed; #2787 left them without a caller, so the HIP 7.1 build rejects them as unused before it reaches the missing hipDeviceAttributeGcnArch. The split-KV reduce kernel drops the unused 'g' and 'm_split' locals. Evidence on a Framework 16 (RX 7700S gfx1102 + Radeon 780M gfx1103, HIP 7.1.52802, kernel 7.2.5): Release build with VLLM_CPP_HIP_ARCHITECTURES='gfx1102;gfx1103' completes; ctest -R 'rocm|cross_device' passes 31/31 on both GPUs (on 984f7265c); Qwen3-0.6B greedy output matches CPU on both GPUs with zero reference-tier hits (~61 tok/s dGPU, ~34 tok/s iGPU, 64 tokens).
