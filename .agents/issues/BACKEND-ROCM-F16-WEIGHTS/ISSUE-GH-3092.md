ID: ISSUE-GH-3092
Title: feat(BACKEND-ROCM-F16-WEIGHTS): retain F16 dense and embedding weights on ROCm
Row: BACKEND-ROCM-F16-WEIGHTS
State: OPEN
Kind: UNKNOWN
GitHub: 3092
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-F16-WEIGHTS`
>
> ROCm currently expands GGUF F16 weights because ordinary matrix multiplication and embedding kernels reject F16 storage. This prevents those weights from staying in their checkpoint format on gfx1100.
>
> Implement F16 storage support through ordinary Matmul/MatmulBT and embedding operations, preserving the existing BF16/F32 activation and output contract. Enable loader admission for dense matrix weights and embedding tables only. Keep stacked expert weights on their existing expansion path; this issue does not enable a full F16 activation runtime.
>
> The spec must establish mixed input arithmetic from the pinned upstream sources, cover both ID widths for embeddings, and prove the default public loading and generation path reaches the new providers. Require red-first tests, physical gfx1100 correctness and memory evidence, fresh mutation review, and operator verification.
>
> This work is independent of PR #2782. That PR expands quantized GEMM kernels. F16 RMSNorm activation support remains a distinct concern tracked by #2542. No CI changes are included.
>
> Spec: `.agents/specs/rocm-f16-weights.md` (to be committed before implementation).
>

## Resolution

The local repair adds mutation-sensitive coverage for explicit registry dtype
refusal, unsupported embedding providers, and both ordinary GEMM compute
override guards. It integrates reviewed prerequisite
`6a7bcb77637e66df34429208e3a4055e0945a875` from #3098 and PR #3101.
The row spec records the exact evidence and remaining gates. The issue stays
open until the work lands. Full-suite baseline failures, oracle limitations,
fresh review, and the final operator gate remain explicit obligations.
The integrated explicit-head fixture reaches a missing native ROCm
`AttnGateSplit` provider. Issue #3106 owns that backend gap. F16-G3 remains
failing with the unchanged fixture and expected successful forward.

The repaired CPU suite passes 703 tests, skips 12, and retains only #3102
out of 716. The operator's complete HIP suite passes 692 tests, skips 12,
and fails 20 of 724. Nineteen failures exactly match the pristine baseline;
the remaining failure is #3106. The added public prerequisite test passes.
The staged preflight executes every supplied gate successfully and compiles
678 translation units. Its five argument-dependent skips and explicit
check dispositions are recorded in the spec. The row remains active.
