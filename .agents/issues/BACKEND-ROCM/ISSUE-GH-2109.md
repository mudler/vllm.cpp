ID: ISSUE-GH-2109
Title: KERNEL-QUANT-CIQ-GEMM-ROCM: ROCm keep-quant GEMM (KQuantGemmK) has no MFMA arm — 61.5% of GPU time, ~4.4x behind llama.cpp mul_mat_q on gfx1200
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2109
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-27
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Row
>
> `KERNEL-QUANT-CIQ-GEMM-ROCM` — the performance half of the ROCm keep-quant GEMM (mirror of `KERNEL-QUANT-CIQ-GEMM-CUDA`, which closed the same gap for NVFP4). Not yet claimed.
>
> ## Measured baseline (same-tool rocprofv3 attribution, Ornith-1.5-9B-Q4_K_M, `-n 128 -ngl 99`, gfx1200 / RX 9060 XT 16 GB)
>
> | | Oracle (llama.cpp b10451) | vllm.cpp (ours) | Ratio |
> |---|---|---|---|
> | **Prefill** (pp512) | 1691 tok/s | 73.7 tok/s | **22.9x slower** |
> | **Decode** (tg128) | 48.51 tok/s | 17.44 tok/s | **2.78x slower** |
> | **Total GPU time** | 18,812 ms | 108,817 ms | **5.78x more** |
>
> Single-kernel verdict: our `KQuantGemmK` (132,096 calls / **66,960 ms / 61.5%** of GPU budget) uses scalar `__dp4a` warp reductions, warp-per-output-row. llama.cpp's `mul_mat_q` uses MFMA tensor cores (`__builtin_amdgcn_mfma_i32_16x16x32_i8` / `16x16x16i8` in `ggml/src/ggml-cuda/mma.cuh`) and runs ~4.4x faster for the same work.
>
> ## The work
>
> Port `src/vt/rocm/rocm_grouped_gemm.hip`'s `KQuantGemmK` / `Q8_0GemmK` / `GroupedKQ8K` (lines 323-403, 446-496) to MFMA tensor-core / hipBLASLt-backed kernels, matching llama.cpp's quantized-GEMM path. Keep the block-dequant math (`DotQ4K`/`DotQ6K`/`DotQ8_0`, lines 179-290) and the `EnsureQuantScratch` stream-ordered pool (lines 421-441). Out of scope: attention, GDN scan, activation quantize (`QuantizeQ8KK` — 11.3%, not the bottleneck), memory-bandwidth (KV-FP8).
>
> ## Relationship to open work on this kernel (do not double-claim)
>
> - **#41** (`BACKEND-ROCM`) — owning feature row.
> - **#1910** — the diagnostic that pinned `KQuantGemmK` at 54-61.5% of ROCm decode/GPU time. This row is a candidate closer for the GEMM (prefill) half.
> - **PR #2086** (`ROCM-KQUANT-NWARPS-DECODE`, OPEN) — ports llama.cpp's *decode GEMV* mechanism (cooperative-warp `nwarps`, `KQuantGemmKCoopQ6K<OutT,8>`). That is the GEMV/n=1 path; this row is the *GEMM/n>1* (prefill) path that llama.cpp serves with MFMA. #2086 explicitly leaves the MFMA/row-packing/Q4_K-Q5_K/MoE work as `## Owed` and does NOT close #1910. Complement, not duplicate — but coordinate so the two branches do not both rewrite `KQuantGemmK`'s launch site.
>
> ## Gates
>
> (a) `test_backend_cross_device` NMSE ≤ 5e-4 vs CPU oracle, unchanged. (b) `test_rocm_quant_dot` green (mirror of `test_cuda_quant_dot`, NMSE ≤ 5e-4). (c) `rocprofv3 --kernel-trace` shows the MFMA replacement ≤ 20,000 ms total on the Ornith-9B trace workload (target: match oracle ≤ 19,000 ms). (d) no regression on `ctest -R 'rocm|cross_device'`.
>
> ## Risk
>
> hipBLASLt may not support per-superblock Q8_K block scales (16 independent scales per 256-element superblock). If not, write custom MFMA with explicit scale dequant in the inner loop — this is what llama.cpp does (`mma.cuh`).
>
> Spec: `.agents/specs/kernel-quant-ciq-gemm-rocm.md` (to be authored).

## Resolution

-
