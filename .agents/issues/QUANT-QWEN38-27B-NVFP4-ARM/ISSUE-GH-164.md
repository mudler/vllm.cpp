ID: ISSUE-GH-164
Title: Qwen3.6-27B NVFP4 checkpoints fail because dense loader requires BF16 `lm_head`
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: CLOSED
Kind: UNKNOWN
GitHub: 164
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-08
Updated: 2026-08-08
Closed: 2026-08-08

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> The Qwen3.6 dense loader fails to load two common 27B NVFP4 checkpoints:
>
> - `nvidia/Qwen3.6-27B-NVFP4`
> - `unsloth/Qwen3.6-27B-NVFP4`
>
> Both fail with:
>
> ```text
> server: fatal: vt: dense loader: expected BF16 for lm_head.weight
> at include/vllm/model_executor/models/dense_weight_loaders.h:83
> ```
>
> ## Cause
>
> `LoadQwen3_5Dense()` always loads an explicit `lm_head.weight` using the BF16-only `LoadBf16Transposed()` path.
>
> However, these checkpoints use different quantized output heads:
>
> - **NVIDIA:** ModelOpt NVFP4, `U8 [248320, 2560]`, with `weight_scale` and `weight_scale_2`
> - **Unsloth:** FP8, `F8_E4M3 [248320, 5120]`, with `weight_scale`
>
> Model revisions tested:
>
> - NVIDIA: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
> - Unsloth: `ccdaab7e68af2409599b8949a8f2685703c9bae5`
>
> ## Environment
>
> - RTX 5090 (`sm_120a`)
> - CUDA 13.0.2
> - vllm.cpp commit `0e3bf3c0`
>
> ## Expected behavior
>
> Support the quantized `lm_head` formats used by these NVFP4 checkpoints, or document the exact supported checkpoint/revision and return a clearer unsupported-format error.

## Resolution

Commit `6f3a3d05492c6f4f7ff05ed3ef5cea4158505e0b` dated 2026-08-08 loads the reported quantized `lm_head` forms and names issue #164. GitHub closed issue #164 on 2026-08-08.
