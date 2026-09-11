ID: ISSUE-GH-1319
Title: The Qwen3.5 decode-graph poison hook wrote past the end of the pinned `seq_lens` block on a SPEC step. `MaybePoisonStagedInputs` (`src/vllm/model_executor/models/qwen3_5.cpp:9848-9855 @ 5c8671c50`) zeroed four pinned staging blocks over `pin.S` elements each, but `seq_lens` is allocated with **R** (`:9733`) — and `PinnedStepInputs` keeps tokens and requests apart deliberately, because SPEC-DSPARK W8 ([#442](https://github.com/mudler/vllm.cpp/issues/442)) separated them for a speculative verify with `S = R * (1 + k)`. On a pure-decode step `S == R` and the fill is exact; on a spec step it writes `(S - R)` int32s past a `cudaHostAlloc`'d block. Bounded: the hook runs only under the test-only `VT_ASYNC_EXECUTOR_POISON=1` (`scripts/env-doc-allowlist.txt:6`, never set in production), whose one consumer is the deterministic RED arm of `tests/parity/test_qwen36_async_serving.cpp:170-173` — so the corruption lands in the instrument, in the arm meant to fail loudly. Found while migrating the two Qwen3.5 drivers onto the capture seam and NOT caused by it (present at that branch's base `5c8671c50`); FIXED IN FLOW by [#1307](https://github.com/mudler/vllm.cpp/issues/1307), where each cell is memset over its own `capacity()` and so cannot disagree with its allocation
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1319
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:431`

### Frozen archive evidence

> | [#1319](https://github.com/mudler/vllm.cpp/issues/1319) | `ENG-CUDAGRAPH-BREAK` | The Qwen3.5 decode-graph poison hook wrote past the end of the pinned `seq_lens` block on a SPEC step. `MaybePoisonStagedInputs` (`src/vllm/model_executor/models/qwen3_5.cpp:9848-9855 @ 5c8671c50`) zeroed four pinned staging blocks over `pin.S` elements each, but `seq_lens` is allocated with **R** (`:9733`) — and `PinnedStepInputs` keeps tokens and requests apart deliberately, because SPEC-DSPARK W8 ([#442](https://github.com/mudler/vllm.cpp/issues/442)) separated them for a speculative verify with `S = R * (1 + k)`. On a pure-decode step `S == R` and the fill is exact; on a spec step it writes `(S - R)` int32s past a `cudaHostAlloc`'d block. Bounded: the hook runs only under the test-only `VT_ASYNC_EXECUTOR_POISON=1` (`scripts/env-doc-allowlist.txt:6`, never set in production), whose one consumer is the deterministic RED arm of `tests/parity/test_qwen36_async_serving.cpp:170-173` — so the corruption lands in the instrument, in the arm meant to fail loudly. Found while migrating the two Qwen3.5 drivers onto the capture seam and NOT caused by it (present at that branch's base `5c8671c50`); FIXED IN FLOW by [#1307](https://github.com/mudler/vllm.cpp/issues/1307), where each cell is memset over its own `capacity()` and so cannot disagree with its allocation | bug |

## Resolution

-
