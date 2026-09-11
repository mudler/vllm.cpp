ID: ISSUE-GH-2329
Title: **`VT_QWEN35_STAGE_MIN_FREE_FRAC` was read from `src/` and documented nowhere, so `check-env-doc` was RED on `origin/main` itself and every branch cut after `207c12932` inherited a failing preflight.** Introduced by `207c12932` ([#2327](https://github.com/mudler/vllm.cpp/issues/2327), [#2328](https://github.com/mudler/vllm.cpp/pull/2328)), read at `src/vllm/model_executor/models/qwen3_5_weights.cpp:218` and explained only in a code comment. Reproduced on `origin/main`'s own bytes, extracted with `git archive` into a clean directory with no branch involved: `check-env-doc` rc 1 naming that one variable; with the entry, rc 0 over 396 scanned names. **This is [#2312](https://github.com/mudler/vllm.cpp/issues/2312) recurring, the same class within one day** -- that row records the identical failure for `VT_DFLASH_BOUNDS_DEVICE` from [#2304](https://github.com/mudler/vllm.cpp/pull/2304), fixed by [#2313](https://github.com/mudler/vllm.cpp/pull/2313) with the same note that it is a base failure every later branch inherits. Two occurrences in a day suggests the gap is STRUCTURAL rather than an oversight: nothing forces the doc entry at the point the knob is introduced, and the gate that would catch it only runs against a base that already merged. **Documented in `docs/ENVIRONMENT.md` rather than allowlisted**, mirroring #2313 and the convention this knob's own family sets -- `VT_QWEN35_ALIAS_HOST_WEIGHTS`, whose behaviour this variable governs, is documented there, and the allowlist is for kernel-internal tuning switches. It is user-facing on its face: it decides whether a dense weight is staged as a true device copy or left aliased, which is the difference #1299 measured between a model that decodes and one that exhausts a 119.631 GiB box. Semantics read from the code rather than transcribed: stage only while `free - bytes >= frac * total`, default `0.55`, with unset, empty, unparsable, `<= 0` and `>= 1` all falling back to `0.55`
Row: ENG-WEIGHT-RESIDENCY
State: UNKNOWN
Kind: bug
GitHub: 2329
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:904`

### Frozen archive evidence

> | [#2329](https://github.com/mudler/vllm.cpp/issues/2329) | `PERF-QWEN35-STAGE-WEIGHTS` | **`VT_QWEN35_STAGE_MIN_FREE_FRAC` was read from `src/` and documented nowhere, so `check-env-doc` was RED on `origin/main` itself and every branch cut after `207c12932` inherited a failing preflight.** Introduced by `207c12932` ([#2327](https://github.com/mudler/vllm.cpp/issues/2327), [#2328](https://github.com/mudler/vllm.cpp/pull/2328)), read at `src/vllm/model_executor/models/qwen3_5_weights.cpp:218` and explained only in a code comment. Reproduced on `origin/main`'s own bytes, extracted with `git archive` into a clean directory with no branch involved: `check-env-doc` rc 1 naming that one variable; with the entry, rc 0 over 396 scanned names. **This is [#2312](https://github.com/mudler/vllm.cpp/issues/2312) recurring, the same class within one day** -- that row records the identical failure for `VT_DFLASH_BOUNDS_DEVICE` from [#2304](https://github.com/mudler/vllm.cpp/pull/2304), fixed by [#2313](https://github.com/mudler/vllm.cpp/pull/2313) with the same note that it is a base failure every later branch inherits. Two occurrences in a day suggests the gap is STRUCTURAL rather than an oversight: nothing forces the doc entry at the point the knob is introduced, and the gate that would catch it only runs against a base that already merged. **Documented in `docs/ENVIRONMENT.md` rather than allowlisted**, mirroring #2313 and the convention this knob's own family sets -- `VT_QWEN35_ALIAS_HOST_WEIGHTS`, whose behaviour this variable governs, is documented there, and the allowlist is for kernel-internal tuning switches. It is user-facing on its face: it decides whether a dense weight is staged as a true device copy or left aliased, which is the difference #1299 measured between a model that decodes and one that exhausts a 119.631 GiB box. Semantics read from the code rather than transcribed: stage only while `free - bytes >= frac * total`, default `0.55`, with unset, empty, unparsable, `<= 0` and `>= 1` all falling back to `0.55` | bug |

## Resolution

-
