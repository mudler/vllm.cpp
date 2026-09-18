ID: ISSUE-LOCAL-01M2TPJS61P4512QZY0B00N0MK
Title: check-rocm-dp4a-intrinsic was widened until its own live-mutation test stopped detecting the scalar expansion
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: 2026-09-18

## Problem

tests/scripts/test_check_rocm_dp4a_intrinsic.py::test_live_scalar_mutation_fails is RED on origin/main at 7fa861392, independent of any local change: it asserts 1 error and gets 0. Commit 2bde17f6c (2026-09-17) added a __has_builtin(__ockl_sdot4) conditional to src/vt/rocm/rocm_grouped_gemm.hip and widened scripts/check-rocm-dp4a-intrinsic.py to pass on 'has_intrinsic OR has_conditional_fallback'. The mutation test replaces only the '  return __ockl_sdot4(va, vb, acc, false);' line with the scalar expansion; the __has_builtin guard survives in the Dp4a body, so the widened checker returns no error for a source that no longer emits v_dot4_i32_i8 at all. AGENTS.md 'Changing the rules or a checker': never make a red gate green by widening its scope. The checker now accepts the exact defect its red-before test was written to catch. FOUND while implementing MODEL-MM-deepseek-v4-1 W3c/W3d (ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB), whose diff touches nothing under src/vt/rocm, scripts/ or tests/scripts (verified byte-identical to origin/main). NOT fixed in that change: the repair changes checker semantics, which AGENTS.md routes through its own spec and fresh review rather than the in-flow rule.

## Resolution

2026-09-18: CLOSED, falsified by the tree. `a09932f87` ("fix(BACKEND-ROCM): the dp4a gate detects a call, not a string", merged as `df87f880e`) rewrote `scripts/check-rocm-dp4a-intrinsic.py` to require a CALL rather than a string, under the spec `.agents/specs/rocm-dp4a-checker-detects-a-call.md`. Measured after rebasing this branch onto that main: `scripts/agent-preflight.sh` reports `ok check-rocm-dp4a-intrinsic` and `ok test_check_rocm_dp4a_intrinsic`, where the run that opened this issue reported both red. Two other local issues on this row, ISSUE-LOCAL-01M2TQ83JQPT2KVK2S2HAZSMNY and ISSUE-LOCAL-01M2TQYXN85A4FK48B8N2WFQ8G, describe the same defect and were already on main when this one was written; this file is the third copy and is closed as such rather than re-specced. AGENTS.md "An issue the tree falsifies closes with that evidence".
