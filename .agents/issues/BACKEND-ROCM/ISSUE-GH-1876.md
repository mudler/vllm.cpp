ID: ISSUE-GH-1876
Title: QuantizeQ8KK is thread-per-superblock: 95.5us/call against llama.cpp's 1.6us, and 41% of the ROCm decode gap
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1876
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-24
Updated: 2026-08-24
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `QuantizeQ8KK` decomposes one thread per 256-element superblock. At decode that
> leaves 16 threads on a 32-CU GPU, and it costs **95.5 us per call against
> llama.cpp's 1.6 us for the same work** — a 35x per-call gap that accounts for
> **41% of the entire ROCm decode deficit** on a dense model.
>
> ## Measured
>
> `rocprofv3` 1.1.0 (`rocprofiler-sdk` 7.2.3, native), `--kernel-trace --stats`,
> `Ornith-1.5-9B-Q4_K_M.gguf` (dense `qwen35`), RX 9060 XT (gfx1200), ROCm 7.2.3.
> Decode isolated by differencing a 4-token run against a 36-token run over 32
> tokens, on both sides. vllm.cpp `4b1154bc5`; llama.cpp `b10451` (`10bf611e5`)
> HIP. Free VRAM asserted above 13 GiB with no resident model process.
>
> | | ms/tok | calls/tok | us/call |
> |---|---:|---:|---:|
> | ours, `QuantizeQ8KK` | **12.321** | 129 | **95.5** |
> | llama.cpp, `quantize_q8_1` | **0.352** | 217 | **1.6** |
>
> Total kernel time per decode token is 52.735 ms against llama.cpp's 23.466 ms.
> The quantizer delta alone is 11.969 ms, **40.9% of that 29.269 ms gap**.
> llama.cpp issues *more* quantize calls than we do and still pays 35x less each.
>
> ## Cause
>
> `src/vt/rocm/rocm_grouped_gemm.hip:146`, and the comment says it outright —
> `Q8_K (thread-per-256-superblock)`:
>
> ```c
> const int64_t t = blockIdx.x * blockDim.x + threadIdx.x;   // one thread = one superblock
> const int64_t elem0 = i * a_rs + sb * kQK_K;
> float mx = 0.0f, amax = 0.0f;
> for (int j = 0; j < kQK_K; ++j) {
>     const float ax = fabsf(DLoadAct(a, adt, elem0 + j));
>     if (ax > amax) { amax = ax; mx = DLoadAct(a, adt, elem0 + j); }
> }
> ```
>
> launched at `rocm_grouped_gemm.hip:479`:
>
> ```c
> QuantizeQ8KK<<<static_cast<unsigned>((m * nsb + 127) / 128), 128, 0, s>>>(...)
> ```
>
> At dense decode `m` is 1 and `nsb` is `K/256`. For a 4096-wide hidden that is
> **16 threads, one workgroup, on a 32-CU part**, each thread walking 256
> elements serially. The MoE path at `:547` has the same kernel with `m` =
> T x top_k, which is why [#1294](https://github.com/mudler/vllm.cpp/issues/1294)
> measured it at 35% of GPU time there.
>
> **Two defects, not one.** `DLoadAct` is called **twice per element** on the
> `amax` scan — once for the comparison and once for the value kept — so the loop
> issues 512 loads where 256 would do. That is independent of the decomposition
> and is a strict improvement on any backend.
>
> ## The donor
>
> `ggml/src/ggml-cuda/quantize.cu:53` decomposes per **element**:
>
> ```c
> const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
> if (i0 >= ne0) return;
> ```
>
> launched (`quantize.cu:567`) as:
>
> ```c
> const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
> const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
> const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);   // 256
> ```
>
> with a warp reduction for the scale. Thousands of threads instead of 16.
>
> Note llama.cpp is a **secondary oracle** here and never the mirror source: it
> supplies the shape of a fix and the floor to beat, not the behaviour. vLLM
> defines behaviour, and this kernel's own header already names its vLLM donor
> (`cuda_quant_dot.cu QuantizeQ8KKernel`). Any change must stay byte-identical to
> the reference path.
>
> ## Scope
>
> - `MatmulBTQuantKernelRocm` (`:479`), the dense path — measured above.
> - `MatmulBTQuantGroupedKernelRocm` (`:547`), the MoE path — the same kernel,
>   measured at 35% by #1294. One fix moves both.
> - The arithmetic is unchanged: this is a decomposition and a redundant-load
>   removal, so every existing golden must stay byte-identical. That is the
>   correctness gate, and it is stronger than a tolerance.
>
> ## Not claimed
>
> No throughput number is predicted here. Closing the quantizer to llama.cpp
> parity would remove 11.969 ms of 52.735 on this one model and prompt, which is
> about 1.3x on the arm measured, and says nothing about other shapes, batch
> sizes, or the MoE path's own ratio. A same-binary A/B behind an environment
> flag, on an idle host, is what a speed claim would owe — and note that
> `VT_GGUF_KEEP_QUANT=0` is not available as the OFF arm on a 16 GiB card
> ([#1870](https://github.com/mudler/vllm.cpp/issues/1870)), so the A/B has to be
> against the unmodified kernel rather than against expansion.
>
> The residual GEMM gap of 1.54x (`KQuantGemmK` + `wvSplitKSml` 32.574 ms against
> MMVQ 21.229 ms) is **not** in scope here and stays with
> [#1863](https://github.com/mudler/vllm.cpp/issues/1863).
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-
