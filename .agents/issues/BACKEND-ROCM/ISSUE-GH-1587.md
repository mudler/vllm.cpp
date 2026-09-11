ID: ISSUE-GH-1587
Title: Port the upstream RDNA3 quantized-GEMM family from csrc/rocm
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1587
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-21
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Upstream surface at the pin
>
> vLLM `555967922` ships a HIP kernel family for quantized GEMM in
> `csrc/rocm/`, exposed through `csrc/rocm/torch_bindings.cpp`:
>
> | Op | Source | Note |
> |---|---|---|
> | `LLMM1` | `skinny_gemms.cu` | matrix-vector GEMM |
> | `wvSplitK` | `skinny_gemms.cu` | skinny GEMM, bf16/f16 |
> | `wvSplitKrc` | `skinny_gemms.cu` | skinny GEMM variant |
> | `wvSplitK_int4_g` | `skinny_gemms_int4.cu` | W4A16 grouped skinny GEMM, per-group scales, optional zero points |
> | `wvSplitKQ` | `skinny_gemms.cu` | FP8 skinny GEMM; gfx1100 excluded upstream (`supports_fp8()` is gfx9 or gfx12x only) |
> | `gptq_gemm_rdna3` | `q_gemm_rdna3.cu` | W4A16 GPTQ, gated `#ifdef VLLM_ROCM_GFX1100` |
> | `gptq_gemm_rdna3_wmma` | `q_gemm_rdna3_wmma.cu` | WMMA variant of the same, same gate |
> | `moe_gptq_gemm_rdna3` | `moe_q_gemm_rdna3.cu` | grouped MoE W4A16 GPTQ, same gate |
>
> The three `rdna3` kernels are built for exactly this project's target card:
> gfx1100, RDNA3, 24 GiB consumer parts.
>
> ## Local gap
>
> `src/vt/rocm/` carries 18 `.hip` files and registers roughly 44 ops. A
> search for `MatmulBTQuant`, `kMatmulBTQuant`, and `vec_dot` over
> `src/vt/rocm/` and `include/vt/` returns nothing. The ROCm backend has no
> quantized-weight GEMM provider, so quantized arms fall back off device.
>
> `AGENTS.md` makes the quantized arms of every model port a standing
> requirement, not a per-model choice. On this box the quantized arm has no
> device path at all.
>
> ## Scope questions for the spec
>
> 1. Which arm lands first: dense W4A16 GPTQ (`gptq_gemm_rdna3`), the MoE
>    variant, or a GGUF k-quant keep-quant `vec_dot` on `kROCM`.
> 2. Routing: through the `vt::` op table and the `OpProvider` seam, with the
>    selection recorded by `KERNEL-ACCEL-PROVIDER-SELECT`, never a parallel
>    path.
> 3. Evidence: invocation parity against the upstream kernels on identical
>    inputs, plus the upstream tests ported in the same change.
> 4. The GGUF k-quant question interacts with `KERNEL-QUANT-CIQ-GEMM-CUDA`,
>    which owns the CUDA keep-quant provider; mirror its contract rather than
>    inventing a second shape.
>
> ## Non-goals
>
> - FP8 on gfx1100. Upstream refuses it on this arch; do not gate this row on
>   an FP8 lane.
> - Triton-on-ROCm families (`TRITON_ATTN`, Triton AWQ, MXFP4 MoE). They stay
>   separate rows if this campaign opens them.
>

## Resolution

-
