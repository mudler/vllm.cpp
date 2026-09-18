ID: ISSUE-GH-3116
Title: fix(BACKEND-ROCM-BF16-MOE): mirror the primary BF16 LM-head output boundary
Row: BACKEND-ROCM-BF16-MOE
State: OPEN
Kind: UNKNOWN
GitHub: 3116
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-BF16-MOE`
>
> The native Qwen3 MoE forward emits F32-local LM-head output where the compiled primary narrows the head output to BF16 (`rocm-residual-norm.md:320`). Disposition analysis of the #3096 production token gate (verdict b, recorded with the `BACKEND-ROCM-RESIDUAL-NORM` evidence) shows this boundary participates in the decode step-6 exact-tie flip: applying the primary's BF16 head narrowing to the native step-6 logits rounds the 63/118 pair to an exact tie at 0.3359375, and native tie-break then keeps 63, the primary's own cc1 answer.
>
> Hidden-state parity at ~1 BF16 ulp is additionally required: a perfect head narrowing alone cannot turn the native margin into the oracle's cc2 answer 118, so this issue depends on the decode attention and Q/K preamble parity of #3115.
>
> Evidence: `/home/vikash/.cache/residual-norm-repair1/green-cc9d4f565/production-fusion-1.json` (native step-6 logits), the primary head-6 BF16 logits capture `oracle-diagnostic-2/L33-C2-R0-head-6-logits.bin`, and the disposition findings recorded with the residual row.

## Resolution

-
