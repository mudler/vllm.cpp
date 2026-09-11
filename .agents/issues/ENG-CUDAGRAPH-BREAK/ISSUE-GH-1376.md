ID: ISSUE-GH-1376
Title: `main` was red on `tests/scripts/test_check_gate_commands.py`, measured at `601b576c6` in a detached worktree of `origin/main`: 8 failures of 44 tests, every one a comparison between the computed runnable population and `RUNNABLE_BASELINE`. `ENG-CUDAGRAPH-BREAK` was in the first and absent from the second. Cause: W5 of that row ([#1361](https://github.com/mudler/vllm.cpp/issues/1361)) filled its spec's `## Gates` section with runnable evidence, including a named test binary with its case and assertion counts and an exit status, which is exactly what moves a row into the runnable population. The ratchet's own error text instructs a re-pin in the SAME change, and the re-pin was not made. This is the growth case the ratchet exists to force a decision about, not a defect in that row's work. **It landed with no remote verdict**: the continuous integration lane that would have caught it independently has not executed for this repository since roughly 07:43Z on 19 August 2026, with runs queueing and none starting while GitHub reports Actions operational. FIXED IN FLOW while merging `origin/main` into `row/ENG-HF-MODEL-DOWNLOAD` for [#1280](https://github.com/mudler/vllm.cpp/issues/1280), because the fix is small and clear and a red `main` blocks every other row's gate. The entry is added with a justifying comment in the form the neighbouring entries use, no checker semantics change, and no test is weakened. After the re-pin the suite reports 45 tests OK and the audit reads 39 runnable of 119 gated rows
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1376
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:465`

### Frozen archive evidence

> | [#1376](https://github.com/mudler/vllm.cpp/issues/1376) | `ENG-CUDAGRAPH-BREAK` | `main` was red on `tests/scripts/test_check_gate_commands.py`, measured at `601b576c6` in a detached worktree of `origin/main`: 8 failures of 44 tests, every one a comparison between the computed runnable population and `RUNNABLE_BASELINE`. `ENG-CUDAGRAPH-BREAK` was in the first and absent from the second. Cause: W5 of that row ([#1361](https://github.com/mudler/vllm.cpp/issues/1361)) filled its spec's `## Gates` section with runnable evidence, including a named test binary with its case and assertion counts and an exit status, which is exactly what moves a row into the runnable population. The ratchet's own error text instructs a re-pin in the SAME change, and the re-pin was not made. This is the growth case the ratchet exists to force a decision about, not a defect in that row's work. **It landed with no remote verdict**: the continuous integration lane that would have caught it independently has not executed for this repository since roughly 07:43Z on 19 August 2026, with runs queueing and none starting while GitHub reports Actions operational. FIXED IN FLOW while merging `origin/main` into `row/ENG-HF-MODEL-DOWNLOAD` for [#1280](https://github.com/mudler/vllm.cpp/issues/1280), because the fix is small and clear and a red `main` blocks every other row's gate. The entry is added with a justifying comment in the form the neighbouring entries use, no checker semantics change, and no test is weakened. After the re-pin the suite reports 45 tests OK and the audit reads 39 runnable of 119 gated rows | bug |

## Resolution

-
