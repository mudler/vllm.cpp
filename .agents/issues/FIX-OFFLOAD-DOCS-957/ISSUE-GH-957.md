ID: ISSUE-GH-957
Title: `4a183b731` (#887) turned a configured weight offload from ACCEPTED-AND-INERT into a hard startup refusal, and neither public document followed. `docs/WEIGHT-OFFLOAD.md` still said "a budget you set is accepted, reported, and does not free memory" and `docs/USAGE.md:1473` still said "Accepted and inert today", while `RefuseUnsupportedWeightOffload` (`src/vllm/model_executor/weight_offloader.cpp:72-83` @ 2daa3287f) throws from the load path (`src/vllm/entrypoints/model_loader.cpp:1410-1414` @ 2daa3287f) before any weight I/O. `ModelFactory::supports_weight_offload` defaults false and NO model sets it, so every architecture is refused; `tests/vllm/model_executor/test_weight_offloader.cpp:376-379` @ 2daa3287f asserts that count itself. Found auditing the 28 commits `documentation-checkpoint` flags: 27 needed nothing
Row: FIX-OFFLOAD-DOCS-957
State: UNKNOWN
Kind: bug
GitHub: 957
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:264`

### Frozen archive evidence

> | [#957](https://github.com/mudler/vllm.cpp/issues/957) | `FIX-OFFLOAD-DOCS-957` | `4a183b731` (#887) turned a configured weight offload from ACCEPTED-AND-INERT into a hard startup refusal, and neither public document followed. `docs/WEIGHT-OFFLOAD.md` still said "a budget you set is accepted, reported, and does not free memory" and `docs/USAGE.md:1473` still said "Accepted and inert today", while `RefuseUnsupportedWeightOffload` (`src/vllm/model_executor/weight_offloader.cpp:72-83` @ 2daa3287f) throws from the load path (`src/vllm/entrypoints/model_loader.cpp:1410-1414` @ 2daa3287f) before any weight I/O. `ModelFactory::supports_weight_offload` defaults false and NO model sets it, so every architecture is refused; `tests/vllm/model_executor/test_weight_offloader.cpp:376-379` @ 2daa3287f asserts that count itself. Found auditing the 28 commits `documentation-checkpoint` flags: 27 needed nothing | bug |

## Resolution

-
