ID: ISSUE-GH-2385
Title: VT_QWEN35_STAGE_MIN_FREE_FRAC is documented as a tuning knob and has no reader in compiled code
Row: ENG-WEIGHT-RESIDENCY
State: CLOSED
Kind: UNKNOWN
GitHub: 2385
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `docs/ENVIRONMENT.md:262` documents `VT_QWEN35_STAGE_MIN_FREE_FRAC` in the **user-facing** table — default `0.55`, an exact formula, a stated parse grammar, and tuning direction:
>
> > stage only while `free - bytes >= VT_QWEN35_STAGE_MIN_FREE_FRAC * total` ... LOWER stages more aggressively; HIGHER is more conservative
>
> Its only occurrence in `src/` or `include/` is a **comment**:
>
> ```
> $ git grep -n "VT_QWEN35_STAGE_MIN_FREE_FRAC" -- src include
> include/vllm/model_executor/models/qwen3_5_weights.h:1340:// `VT_QWEN35_STAGE_MIN_FREE_FRAC` of its total memory free (default 0.55).
> ```
>
> No `getenv`, no string literal, no macro construction. Every value a user sets is discarded.
>
> ## Why
>
> The staging policy was rewritten to decide **once** from a stable total (#2342): `SetSafetensorsWeightBudget` (`src/vllm/model_executor/models/qwen3_5_weights.cpp:211`) reads only `VT_QWEN35_STAGE_RESERVE_BYTES` and `VT_QWEN35_ALIAS_HOST_WEIGHTS`, then applies `2*model + reserve <= total` — a rule with no fraction in it. The knob's reader was deleted with the old live-free policy; its documentation row and the header comment above `StagingFitsModel` were not.
>
> ## Why this is worse than silence
>
> The doc row's stated failure mode reassures the reader that a bad value degrades gracefully ("a typo cannot silently disable the floor"), when in fact **every** value, valid ones included, is ignored. An operator tuning a staging OOM would change this and see no effect, with nothing to indicate why.
>
> ## Fix
>
> Delete the row from `docs/ENVIRONMENT.md` and correct the stale comment at `qwen3_5_weights.h:1340`, which still asserts the deleted rule directly above the function that implements the new one.
>
> Row: `ENG-WEIGHT-RESIDENCY`

## Resolution

GitHub links pull request #2438 as closing issue #2385 on 2026-08-31.
