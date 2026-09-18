ID: ISSUE-GH-3106
Title: ROCm unfused gated attention cannot dispatch AttnGateSplit
Row: BACKEND-ROCM-ATTN-GATE-SPLIT
State: OPEN
Kind: UNKNOWN
GitHub: 3106
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-ATTN-GATE-SPLIT`
>
> The ROCm backend does not register `kAttnGateSplit`. A full-attention Qwen3.5 forward reaches this missing operation when its rotary width is zero and the fused preamble cannot run.
>
> The operator reproduced the failure in the unchanged F16 registered-forward fixture after integrating the reviewed full-attention state prerequisite from #3098. The complete `test_gguf_keep_quant` run passed 10,455 assertions, then its explicit-head production case threw `vt: no kernel for op AttnGateSplit (id 68) on device rocm (type 5)`. This is a failed gate, not a test pass. The retained-weight implementation from #3092 does not add or remove this operation.
>
> Observed inputs: `BuildDenseF16Gguf(DenseDims{}, false, true)` has hidden width 64, vocabulary 32, two query heads, one KV head, head width 32, and two full-attention layers. The fixture omits `qwen35.rope.dimension_count`; `HfConfigFromGguf` resolves that field to zero. No fixture, GDN state, fallback permission, or runtime override was changed to avoid the failure.
>
> The executing chain is `ModelRegistry::Forward` into `FullAttnBlockPaged` in `src/vllm/model_executor/models/qwen3_5.cpp`, then `vt::AttnGateSplit` in `src/vt/ops.cpp`. `src/vt/rocm/rocm_ops.hip` has the fused preamble registration but no split registration. The shared operation accepts F32 and BF16 inputs and exposes separate query and gate outputs. The CPU and CUDA implementations already define the local contract.
>
> Evidence: `/home/vikash/.cache/rdna3-f16-repair1/operator-focused/gguf-keep-quant-control.log`, SHA-256 `f4d682bdfa800b80c7e25d75b51c31f00a0d3168579534a3e25fff2fda84752a`. HIP archive SHA-256 `dae20f3a702462461ae9edc7143e4774fe448882b16fe506441194d446f02bc7`. The frozen source map is `398ad011bd6d16d04f1aa4ada98e0058def422a2b4925b6dd39d90604f838696`. The operator ran the binary on gfx1100 under `/home/vikash/gpu.lock`, with HIP and ROCR visibility set to device 0.
>
> The owning backend row owes a committed spec and a fresh implementation/review campaign for this newly exposed provider gap. Read and execute the pinned vLLM split path, preserve all applicable shared dtypes and the current contiguous-input validation, port the focused contract, and require the existing registered-forward fixture to pass. Delete the production registration in scratch to prove the gate detects lost reachability. Inspect subsequent operations before claiming that the unfused attention path is complete.
>
> This is a separate existing backend capability gap discovered during the bounded F16 work. The F16 MR keeps its fixture and records its production gate as FAILING until this issue is resolved. It does not claim that the successful full-attention public fixture, which has nonzero rotary width, verifies the zero-rotary-width path. No quantized GEMM, PR #2782, or CI change is requested here.
>
>
> ## Scoped repair specification
>
> Parent: `BACKEND-ROCM`. Spec: `.agents/specs/rocm-attn-gate-split.md`, based on `4137b96369467e925bfdf0738e5bad013c89b58f`. The child owns the native split and the subsequent zero-width rotation refusal needed by the unchanged production fixture.
>
> Static inspection found that `FullAttnBlockPaged` calls `vt::RopeNeox` with the resolved zero rotary width after the split and RMSNorm calls. The shared wrapper currently rejects zero. The pinned Python rotation reference preserves the full tail for zero width, but the custom HIP launcher computes a zero-thread block. The operator measured identity in both dtypes through the default primitive dispatch, which selects forward_native. Explicit forward_hip fails in both dtypes. These operation measurements do not establish upstream model construction or missing-GGUF-field defaults. The selected local adaptation preserves the loader's existing zero resolution and skips only the two model rotation calls when their resolved width is zero. The shared primitive and fixture remain unchanged. A split-only implementation cannot close this production gate.

## Resolution

The [scoped specification](../../specs/rocm-attn-gate-split.md) precedes implementation.
The operator measured default primitive identity and retained the explicit HIP launch failures.
The native provider and selected local model guards remain unimplemented.
The issue stays open until the reviewed production repair lands.
