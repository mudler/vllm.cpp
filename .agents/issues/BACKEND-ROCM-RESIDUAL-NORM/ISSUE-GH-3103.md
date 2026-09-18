ID: ISSUE-GH-3103
Title: fix(BACKEND-ROCM): mirror compiled residual normalization boundaries
Row: BACKEND-ROCM-RESIDUAL-NORM
State: OPEN
Kind: UNKNOWN
GitHub: 3103
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> The default compiled vLLM Qwen3 MoE path and the native ROCm path round residual expressions at different boundaries. This is a shared normalization gap found while gating #3094. It predates the new BF16 expert providers.
>
> The reproducer uses the fixed `Qwen3MoeForCausalLM` fixture from `.agents/specs/rocm-bf16-moe.md`: hidden width 128, two layers, four experts, and top-k 2. The pinned vLLM revision is `e126687a9a828d513c01a07cd69f025f27d63280`, with default compilation, production graphs, ROCM_ATTN, and Triton experts. The native model differs at six generated positions across three repeated length-33, concurrency-2 runs. This issue proves one earlier numerical difference and does not claim that correcting it alone resolves the token gate.
>
> For layer 0, token 0, the actual QKV and attention projection outputs match exactly. Both normalization calls receive identical BF16 attention output, embedding residual, and gamma. The normalized results differ in 33 of 128 values. The first difference is element 3: native `-0.1865234375`, oracle `-0.185546875`. The same witness exists at concurrency 1 and 2. An independent operator rerun of the captured-input FP32 algebra reproduces all 128 native values with residual rounding and all 128 oracle values without that rounding.
>
> `src/vt/rocm/rocm_rmsnorm.hip:125–147` rounds the residual sum to its stored BF16 dtype before variance, then reloads that stored sum. The pinned IR at `vllm/ir/ops/layernorm.py:44–62` computes variance from the FP32 sum. The executing generated kernel confirms this distinction.
>
> The complete compiled lifetime matters. The layer-0 post-attention kernel keeps attention and embedding operands separately and does not store their intermediate residual sum. The next input normalization recomputes `MoE + (attention + embedding)` in FP32, then materializes its residual in BF16. The final layer similarly recomputes the two additions for final normalization. Every allocated residual buffer remains BF16. Blanket widening of residual storage would not mirror this chain.
>
> The complete handoff is `/home/vikash/vllm.cpp-rdna3-moe-impl/build-rdna3-moe-hip/evidence/norm-gap-handoff/handoff.md`, SHA256 `9850ee49c93bb82ddfca1a811fb422337c33ea09335e749fbf3c42dfcaa0400a`. Its manifest records the six generated modules, executed AOT artifact, exact source paths, and hashes. The operator independently checked all 14 manifest entries. Paired raw captures, `residual-row0-diagnostic.json`, and `residual-row0-operator-receipt.json` retain the arithmetic proof.
>
> The owning backend row owes a separate committed spec, fresh implementation, and independent review for the complete lifetime through shared fusion seams. Preserve physical dtype evidence, production graph defaults, and the fixed workload. Port the applicable upstream normalization tests and mutate the intermediate rounding boundary and production wiring. The earlier attention-output difference and local F32 LM-head output remain separate gaps. Keep the MoE token gate failing until its required mismatches are resolved.
>

## Resolution

The child row `BACKEND-ROCM-RESIDUAL-NORM` owns the repair under parent `BACKEND-ROCM`.
The [committed repair spec](../../specs/rocm-residual-norm.md) defines the ordered residual expression and production tests.
The GitHub issue carries the same child row. The shared repair passes its first production witness on the physical GPU.
The complete hardware gate, upstream fixture comparison, and fresh mutation review remain pending in the repair spec.
