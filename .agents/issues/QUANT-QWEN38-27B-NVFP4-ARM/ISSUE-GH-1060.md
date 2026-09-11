ID: ISSUE-GH-1060
Title: Qwen3.8-27B AutoRound W4A16: native Windows RTX 3090 integration and performance gate
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: CLOSED
Kind: UNKNOWN
GitHub: 1060
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-16
Updated: 2026-08-20
Closed: 2026-08-20

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Run the local `Qwen3.8-27B-W4A16-AutoRound` compressed-tensors checkpoint on a native Windows RTX 3090 build, reproduce the proven vLLM embedding/lm-head quantization and MTP path, and then exceed the matched vLLM throughput without changing generated-token correctness.
>
> ## Checkpoint and reference
>
> - Checkpoint: `G:\python\custom-kernel-3090\Qwen3.8-27B-W4A16-AutoRound`
> - Patch/reference setup: `G:\python\custom-kernel-3090\qwen38-27b-rtx3090`
> - Architecture: `Qwen3_5ForConditionalGeneration`, dense 64-layer Qwen3.8-27B
> - Body: compressed-tensors `pack-quantized`, signed INT4, group size 128, BF16 activations
> - Vocabulary matrices in the supplied snapshot: BF16, untied
> - MTP: one BF16 head in `model_extra_tensors.safetensors`
> - Device: local native Windows RTX 3090, sm_86, 24 GB, CUDA 13.2
>
> ## Scope
>
> - Add the exact AutoRound W4A16 loader and compute arm.
> - Add the Ampere sm_86 Marlin INT4 W4A16 execution path required by this checkpoint.
> - Reproduce the reference scripts' INT8 group-128 `embed_tokens` and `lm_head` conversion/loading, including the main and MTP embedding paths.
> - Keep MTP lossless and measure useful depths rather than assuming the deepest setting wins.
> - Provide one Qwen3.8-27B-specialized Windows executable/launcher with model-shape specialization and maximum safe VRAM defaults.
> - Gate correctness before performance, then benchmark matched C1/C2/C4/C8 workloads with at least 1,028 generated tokens, saved raw evidence, discarded warmup, and idle-GPU checks.
>
> ## Out of scope
>
> - Other models, other GPU architectures, vision, remote hosts, release publication, and pushing or merging without separate authority.
>
> ## Acceptance
>
> - Every checkpoint tensor is loaded or refused by an explicit named rule; no silent fallback.
> - Native Windows sm_86 build and real RTX 3090 execution succeed.
> - Greedy output is matched against the pinned vLLM behavior on identical prompts.
> - Long-generation tests complete without OOM or corruption.
> - Prompt/prefill and decode throughput are measured separately for C1/C2/C4/C8.
> - Optimization results use same-binary A/B tests and retain all raw logs/JSON/CSV.

## Resolution

GitHub records `state_reason: not_planned`, closed by `Don-Chad` on 2026-08-20. This disposition claims no implementation or performance result.
