ID: ISSUE-GH-1628
Title: **The DFlash2 candidate selector could not consume a QUANTIZED target `lm_head`, so the arm was refused on the one checkpoint the `BENCH-QWEN38-27B-SOTA` campaign has.** Measured on `dgx:gpu0` 2026-08-21: this engine loads `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` and generates correctly (canary `19 x 23 -> 437`, warm decode 11.06 tok/s vs vLLM's 9.71 on the same box), and attaching the DFlash2 draft died at `vllm_engine_load: dflash: target tensor lm_head.weight is not BF16 (got U8)`. The draft owns no head — it runs the TARGET's — and `SharedHeadSource` read it with one `LoadNamedBf16`, refusing on the STORED DTYPE. That predicate cannot separate the two states `## Risks/decisions` D12 is about: a head WIDENED into something the target does not compute with, and a head kept PACKED and computed with natively. D12 is NOT reversed — it stands for the GGUF container, the only one that still widens a head — and the safetensors arm now takes the target loader's OWN routing decision (`DenseLmHeadTakesNvfp4`) so an NVFP4 head lands packed in `Qwen3DFlashWeights::lm_head_fp4` and the draft's logits GEMM is the same W4A16 dispatcher the target's head takes. The merged oracle agrees and needs no branch: at vllm-project/vllm#52816 head `b389ac29` `compute_candidates` carries no quant-method check and goes through `LogitsProcessor.get_top_k_tokens` -> `_apply_head` -> `lm_head.quant_method.apply`, so the guard this port mirrored at head `66e5414c` is gone. Gated on the PROPERTY and not the load: the draft's block forward must be BITWISE equal to `Qwen3_5MTPModel::ComputeLogits` — the other draft that shares the target's head — over the same hidden states, so the selector's top-K is the target's exactly. FP8 and true-W4A4 heads still refuse by name, DSpark still refuses (`## Owed` O28), and the CUDA arm plus the real checkpoint are owed a measurement (`## Owed` O29)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1628
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:581`

### Frozen archive evidence

> | [#1628](https://github.com/mudler/vllm.cpp/issues/1628) | `SPEC-DFLASH2` | **The DFlash2 candidate selector could not consume a QUANTIZED target `lm_head`, so the arm was refused on the one checkpoint the `BENCH-QWEN38-27B-SOTA` campaign has.** Measured on `dgx:gpu0` 2026-08-21: this engine loads `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` and generates correctly (canary `19 x 23 -> 437`, warm decode 11.06 tok/s vs vLLM's 9.71 on the same box), and attaching the DFlash2 draft died at `vllm_engine_load: dflash: target tensor lm_head.weight is not BF16 (got U8)`. The draft owns no head — it runs the TARGET's — and `SharedHeadSource` read it with one `LoadNamedBf16`, refusing on the STORED DTYPE. That predicate cannot separate the two states `## Risks/decisions` D12 is about: a head WIDENED into something the target does not compute with, and a head kept PACKED and computed with natively. D12 is NOT reversed — it stands for the GGUF container, the only one that still widens a head — and the safetensors arm now takes the target loader's OWN routing decision (`DenseLmHeadTakesNvfp4`) so an NVFP4 head lands packed in `Qwen3DFlashWeights::lm_head_fp4` and the draft's logits GEMM is the same W4A16 dispatcher the target's head takes. The merged oracle agrees and needs no branch: at vllm-project/vllm#52816 head `b389ac29` `compute_candidates` carries no quant-method check and goes through `LogitsProcessor.get_top_k_tokens` -> `_apply_head` -> `lm_head.quant_method.apply`, so the guard this port mirrored at head `66e5414c` is gone. Gated on the PROPERTY and not the load: the draft's block forward must be BITWISE equal to `Qwen3_5MTPModel::ComputeLogits` — the other draft that shares the target's head — over the same hidden states, so the selector's top-K is the target's exactly. FP8 and true-W4A4 heads still refuse by name, DSpark still refuses (`## Owed` O28), and the CUDA arm plus the real checkpoint are owed a measurement (`## Owed` O29) | bug |

## Resolution

-
