ID: ISSUE-GH-3124
Title: EXL3 GEMM: implement reconstruct+cuBLAS path for M>144 to close prefill rate gap
Row: QUANT-EXL3
State: CLOSED
Kind: UNKNOWN
GitHub: 3124
Mirror: SYNCED
Availability: FULL
Created: 2026-09-10
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BENCH-QWEN38-EXL3-VARIADIC`
>
> ExLlamaV3 uses a dual-path EXL3 GEMM strategy. vllm.cpp only has path 1,
> which explains the ~2x prefill rate gap on long prompts (298 vs 585 tok/s
> on the XL band of the variadic benchmark).
>
> ## Root cause
>
> ExLlamaV3's `LinearEXL3.forward` (`exllamav3/modules/quant/exl3.py:132-139`)
> dispatches on M (the batch token count):
>
> - **M <= 144** (`AUTO_RECONSTRUCT_THRESHOLD`): `exl3_gemm` — the EXL3 persistent
>   cooperative kernel. Both engines use this.
> - **M > 144**: `reconstruct_hgemm` — dequantizes EXL3 weights to fp16 on-device,
>   then runs cuBLAS fp16 GEMM. vllm.cpp **lacks this path entirely**.
>
> At large M (prefill), the one-time dequant cost (O(K*N), M-independent) is
> amortized, and cuBLAS is compute-bound, so this path is ~2x faster than the
> cooperative kernel.
>
> ## Evidence — benchmark data matches
>
> | Band | M (mean) | OURS/THEIRS ratio | ExLlamaV3 path |
> |---|---|---|---|
> | S | 111 | 0.90x | EXL3 kernel (<=144) |
> | M | 185 | 1.02x | reconstruct+cuBLAS, not yet amortized |
> | L | 931 | 0.67x | cuBLAS efficient |
> | XL | 2811 | 0.51x | cuBLAS at peak |
>
> ## What is missing
>
> - `Exl3MatmulD` (`dense_attn_block.h:297-337`) always calls `vt::Exl3Gemm` —
>   no M-threshold, no reconstruct path
> - No device-side EXL3 weight dequantization kernel
> - No cuBLAS fp16 GEMM dispatch for dequantized weights
>
> ## Scope
>
> 1. Device-side EXL3 weight dequantization kernel (reconstruct to fp16)
> 2. M-threshold dispatch (M > 144 -> reconstruct + cuBLAS, M <= 144 -> EXL3 cooperative)
> 3. Optionally fuse Hadamard transforms into dequant (for M >= 1024)
> 4. cuBLAS fp16 GEMM for the matmul
>
> Discovered during the c16/c32 expansion of the variadic benchmark (#3122).

## Resolution

Fixed by PR #3150: EXL3 reconstruct + cuBLAS GEMM path for M > 144
