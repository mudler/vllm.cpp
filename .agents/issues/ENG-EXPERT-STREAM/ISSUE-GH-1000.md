ID: ISSUE-GH-1000
Title: `main` is RED on `check-env-doc` at `3ce1cf7c7`: `VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS` and `VT_MOE_EXPERT_STREAM_SLOT_BYTES` arrived with `3005447f8` (#993) and are neither in `docs/ENVIRONMENT.md` nor on `scripts/env-doc-allowlist.txt`, so `check-env-doc` and `test_check_env_doc` fail on every branch cut from current main. Measured in a clean worktree at that SHA with no local edits. Found running the preflight as the gate for #998, and NOT repaired there: choosing documented knob versus internal tuning switch for each var belongs to the row that added them
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1000
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:277`

### Frozen archive evidence

> | [#1000](https://github.com/mudler/vllm.cpp/issues/1000) | `ENG-EXPERT-STREAM` | `main` is RED on `check-env-doc` at `3ce1cf7c7`: `VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS` and `VT_MOE_EXPERT_STREAM_SLOT_BYTES` arrived with `3005447f8` (#993) and are neither in `docs/ENVIRONMENT.md` nor on `scripts/env-doc-allowlist.txt`, so `check-env-doc` and `test_check_env_doc` fail on every branch cut from current main. Measured in a clean worktree at that SHA with no local edits. Found running the preflight as the gate for #998, and NOT repaired there: choosing documented knob versus internal tuning switch for each var belongs to the row that added them | bug |

## Resolution

-
