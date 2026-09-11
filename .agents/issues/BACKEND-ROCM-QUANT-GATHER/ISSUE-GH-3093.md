ID: ISSUE-GH-3093
Title: feat(BACKEND-ROCM-QUANT-GATHER): gather quantized embeddings on ROCm
Row: BACKEND-ROCM-QUANT-GATHER
State: OPEN
Kind: UNKNOWN
GitHub: 3093
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-QUANT-GATHER`
>
> ROCm has no native kEmbeddingQuant provider. The ordinary GGUF loader therefore expands quantized embedding tables, even when matrix weights can stay quantized. This consumes memory and leaves a backend capability gap on gfx1100.
>
> Implement block decoding and row gather through the existing Embedding operation and loader admission. Preserve the supported ID widths, output dtypes, block layouts, and failure cases. Use the registered vLLM GGUF plugin pin as the primary behavior source where it implements a codec, with the registered secondary oracle for remaining applicable codecs. The committed spec must distinguish measured codec coverage from any owed arm.
>
> Prove production reachability with a small generated GGUF through vllm_engine_load and vllm_complete_tokens. Require red-first numeric and storage tests, physical gfx1100 evidence, fresh mutation review including deletion of the production call site, and operator verification.
>
> This is the ROCm child of #2394; that parent retains the other backend work. It does not depend on the quantized GEMM provider added by PR #2782. No CI changes are included.
>
> Spec: `.agents/specs/rocm-quant-gather.md` (to be committed before implementation).
>

## Resolution

-
