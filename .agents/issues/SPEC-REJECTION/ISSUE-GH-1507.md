ID: ISSUE-GH-1507
Title: SPEC-REJECTION: implement stochastic speculative verification
Row: SPEC-REJECTION
State: OPEN
Kind: UNKNOWN
GitHub: 1507
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Extend the shared SPEC-REJECTION verifier from greedy-only acceptance to the stochastic rejection-sampling contract required by DFlash2 and other probabilistic speculative decoders.
>
> ## Current gap
>
> The local RejectionSampler accepts drafts only when they equal the target argmax. Its API and runner call chain carry target logits plus draft token ids, but do not carry realized draft probabilities or logits q, request temperatures, deterministic per-request RNG state, or residual distribution sampling after rejection.
>
> DFlash2 W4 requires a T>0 inverse-CDF proposal arm and a realized-q draft-logit cache consumed by lossless verification. Implementing that cache without a stochastic consumer would land dead code.
>
> ## Scope
>
> Write a committed prerequisite spec under the existing SPEC-REJECTION row, then implement the pinned-vLLM stochastic acceptance and residual-resampling path through the shared API, CPU reference, CUDA/device op, runner state, and production call chain. Preserve the existing greedy path byte-for-byte when temperature is zero.
>
> ## Consumer
>
> SPEC-DFLASH2 W4 / issue #1314 depends on this work before its full path walk and realized-q cache can land.
>
> ## Integration
>
> One pull request for the prerequisite spec and implementation, as recorded in .agents/developer-preferences.md. Local commits are authorized; push, PR creation, and merge are not.

## Resolution

-
