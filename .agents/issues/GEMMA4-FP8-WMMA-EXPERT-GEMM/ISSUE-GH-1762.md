ID: ISSUE-GH-1762
Title: Gemma-4 FP8 WMMA expert GEMM for M>1 prefill on gfx1201
Row: GEMMA4-FP8-WMMA-EXPERT-GEMM
State: OPEN
Kind: perf
GitHub: 1762
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-23
Updated: 2026-08-26
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Gemma-4-26B FP8 prefill on dual R9700 still pays a dequant+BF16 Tensile tax on every T>1 expert GEMM. `ROCPROF_P11K_SHAREDK` (KEEP, PC=0) attributes **21.4%** of prefill kernel time to hipBLAS Tensile GemmEx and **10.1%** to `Fp8ChannelDequantBf16`. Combined **31.5%**. Attention (46.4%) is a separate lever.
>
> Row: `GEMMA4-FP8-WMMA-EXPERT-GEMM`
>
> Current product path is `ExpertGeGLUFp8Native` (`gemma4_moe.cpp`): T=1 stays fused FP8 GEMV; T>1 dequantizes FP8 weights to BF16 and calls `MatmulBT` / `MatmulBTAlphaBeta`. hipBLASLt FP8 W8A8 was already slower than BF16 GemmEx on gfx1201 and is not the comparator.
>
> This issue tracks a **gated, default-OFF** fused-dequant direct-FP8 WMMA expert GEMM for FP8 T>1 only. Decode M=1 is untouched. Unset / wrong-arch / wrong-dims / tail must keep today's dequant+hipBLAS path.
>
> Spec-first. No kernel, no GPU, and no product integration on this filing. A 2026-08-16 isolated scratch probe (`a70e`) is **not** a product kernel: it failed both the numerical gate and the production speed gate (0.126× at M=2048). Reopen only a materially different construction.
>
> Integration gate: **>1.3× vs the actual production dequant+Tensile path** over a **time-weighted** M histogram. Token-mass is measured; time weights are still VOID. Scalar-only uplift is insufficient.
>
> Amdahl check from the p11k profile: drop the 10.1% dequant tax and ~2× the 21.4% GEMM ≈ **1.26× overall** (~3130 t/s @11k from the KEEP class). That is the vision, not a GREEN.
>
> Relates to #41. Orthogonal to #523 (GGUF keep-quant).

## Resolution

-
