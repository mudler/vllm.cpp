ID: ISSUE-GH-2307
Title: **`VT_DFLASH_BOUNDS_DEVICE` was read from `src/` and documented nowhere, so `check-env-doc` was RED on `origin/main` and every branch cut from it inherited a red preflight.** Introduced by `21ef6f053` ([#2274](https://github.com/mudler/vllm.cpp/issues/2274), [#2304](https://github.com/mudler/vllm.cpp/pull/2304)), read at `src/vllm/model_executor/models/qwen3_dflash_internal.h:375`. Reproduced on a CLEAN `origin/main` checkout with no local changes, so it was not an in-flight artifact; it failed both `check-env-doc` and `test_check_env_doc` under `scripts/agent-preflight.sh`. FOUND while landing [#2302](https://github.com/mudler/vllm.cpp/issues/2302) and FIXED IN FLOW, per AGENTS.md's rule that filing does not defer the fix. **Documented in `docs/ENVIRONMENT.md` rather than allowlisted**, following the convention its own family sets -- `VT_DFLASH_PAGED`, `VT_DFLASH_GRAPH`, `VT_DFLASH_ATTN_BLOCK` and `VT_FA2_DFLASH_BLOCK` are all documented there, and the allowlist is for kernel-internal tuning switches. The distinction is load-bearing here rather than clerical: the switch adds two `Copy` + `Synchronize` round-trips onto a path whose entire purpose is to avoid a sync, so enabling it changes the timing of the very thing `SPEC-DFLASH2` measures -- a DIAGNOSTIC run, never a speed run, and a reader has to be told that
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 2307
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:895`

### Frozen archive evidence

> | [#2307](https://github.com/mudler/vllm.cpp/issues/2307) | `SPEC-DFLASH2` | **`VT_DFLASH_BOUNDS_DEVICE` was read from `src/` and documented nowhere, so `check-env-doc` was RED on `origin/main` and every branch cut from it inherited a red preflight.** Introduced by `21ef6f053` ([#2274](https://github.com/mudler/vllm.cpp/issues/2274), [#2304](https://github.com/mudler/vllm.cpp/pull/2304)), read at `src/vllm/model_executor/models/qwen3_dflash_internal.h:375`. Reproduced on a CLEAN `origin/main` checkout with no local changes, so it was not an in-flight artifact; it failed both `check-env-doc` and `test_check_env_doc` under `scripts/agent-preflight.sh`. FOUND while landing [#2302](https://github.com/mudler/vllm.cpp/issues/2302) and FIXED IN FLOW, per AGENTS.md's rule that filing does not defer the fix. **Documented in `docs/ENVIRONMENT.md` rather than allowlisted**, following the convention its own family sets -- `VT_DFLASH_PAGED`, `VT_DFLASH_GRAPH`, `VT_DFLASH_ATTN_BLOCK` and `VT_FA2_DFLASH_BLOCK` are all documented there, and the allowlist is for kernel-internal tuning switches. The distinction is load-bearing here rather than clerical: the switch adds two `Copy` + `Synchronize` round-trips onto a path whose entire purpose is to avoid a sync, so enabling it changes the timing of the very thing `SPEC-DFLASH2` measures -- a DIAGNOSTIC run, never a speed run, and a reader has to be told that | bug |

## Resolution

-
