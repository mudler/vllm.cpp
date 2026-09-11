ID: ISSUE-GH-2090
Title: **The scheduler-mirror claim does not survive a spec-decode run.** `.agents/parity-ledger.md:467` scopes itself honestly to the waiting loop plus two defaults measured with no speculator, but `specs/scheduler-prefill-coschedule.md:132`, `specs/c8-p99-itl-tail-2026-07-18.md:28`, `:64-66` and `specs/async-serving.md:224` restate it unscoped. Two upstream mechanisms are absent and neither had an issue: `pad_spec_decode` (`vllm/v1/core/sched/scheduler.py:826-843`, `:1022-1025` at pin `5559679229`), which pads a newly admitted 1-token request to `1 + num_spec_tokens` to "preserve full cudagraph for this step", and the dynamic-SD lookup (`:1122-1125`, config-gated upstream). Both are recorded as deferrals in `include/vllm/v1/core/sched/scheduler.h:54-56`. It matters here because a ragged batch is a WHOLE-STEP cliff: `GraphEligibleQueryLen` (`src/vllm/v1/worker/gpu/cudagraph_dispatch.h:161-175`) refuses the entire step if any one request has `drafts + 1 != q`. INERT for the #1574 ladder, which runs `--no-enable-prefix-caching`; filed because it becomes live the moment prefix caching does. Verified in the same read: no `O(num_running^2)` term exists in our `schedule()` that upstream lacks
Row: ENG-SCHED-CORE
State: UNKNOWN
Kind: parity
GitHub: 2090
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:804`

### Frozen archive evidence

> | [#2090](https://github.com/mudler/vllm.cpp/issues/2090) | `ENG-SCHED-CORE` | **The scheduler-mirror claim does not survive a spec-decode run.** `.agents/parity-ledger.md:467` scopes itself honestly to the waiting loop plus two defaults measured with no speculator, but `specs/scheduler-prefill-coschedule.md:132`, `specs/c8-p99-itl-tail-2026-07-18.md:28`, `:64-66` and `specs/async-serving.md:224` restate it unscoped. Two upstream mechanisms are absent and neither had an issue: `pad_spec_decode` (`vllm/v1/core/sched/scheduler.py:826-843`, `:1022-1025` at pin `5559679229`), which pads a newly admitted 1-token request to `1 + num_spec_tokens` to "preserve full cudagraph for this step", and the dynamic-SD lookup (`:1122-1125`, config-gated upstream). Both are recorded as deferrals in `include/vllm/v1/core/sched/scheduler.h:54-56`. It matters here because a ragged batch is a WHOLE-STEP cliff: `GraphEligibleQueryLen` (`src/vllm/v1/worker/gpu/cudagraph_dispatch.h:161-175`) refuses the entire step if any one request has `drafts + 1 != q`. INERT for the #1574 ladder, which runs `--no-enable-prefix-caching`; filed because it becomes live the moment prefix caching does. Verified in the same read: no `O(num_running^2)` term exists in our `schedule()` that upstream lacks | parity |

## Resolution

-
