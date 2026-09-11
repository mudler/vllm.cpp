ID: ISSUE-GH-2101
Title: **`main` does not compile under MSVC: seven range-`for` loop variables named `n` in `Glm5NextExpectedGgufTensors` hide the function-scope `const size_t n`, and `/W4 /WX` turns C4456 into `error C2220`.** Landed by [#2067](https://github.com/mudler/vllm.cpp/issues/2067) (PR [#2073](https://github.com/mudler/vllm.cpp/pull/2073), commit `47a2b35a5`), which authored `src/vllm/model_executor/models/glm5_next_weights.cpp` whole. `windows-msvc-cpu` and `windows-msvc-vulkan` both fail the build, so **every pull request that merges current `main` inherits a red Windows pair** — and it looks exactly like the long-standing [#584](https://github.com/mudler/vllm.cpp/issues/584) crash in `gh pr checks`, same two job names and same red, distinguishable only by reading the log: #584 carries exit `-1073740791` and zero `error C####`, this carries one `error C2220` and no crash code. **The issue's stated cause is not the mechanism, and the correction matters for the fix.** Sibling scopes do not hide one another, so the five loops named in #2101 do not shadow each other; every one of them shadows `const size_t n` at `glm5_next_weights.cpp:252`, the layer-count local the function's own bounds check uses. Renaming loop variables to be distinct *from each other* would therefore have left the defect in place. **CI reported four sites and there are seven**, because MSVC stops at the first `error C2220`: 276, 279, 285 and 287 reached the log; 288, 293 and 299 never did. Found with GCC's `-Wshadow`, whose `shadows a previous local` diagnostic is the exact analogue of C4456 and which names all seven at once — red-before `rc=1`, green-after `rc=0` on the same command. Fixed by naming the function-scope local `layer_count` for what it is, which removes all seven hidings at their source, and by naming the seven loop variables `tn` so no bland one-letter name can collide there again. No pragma, no `/WX` relaxation, no suppression: the warning is correct. Green-after for the Windows build itself is the CI job, which cannot be run on this fleet
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2101
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:811`

### Frozen archive evidence

> | [#2101](https://github.com/mudler/vllm.cpp/issues/2101) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **`main` does not compile under MSVC: seven range-`for` loop variables named `n` in `Glm5NextExpectedGgufTensors` hide the function-scope `const size_t n`, and `/W4 /WX` turns C4456 into `error C2220`.** Landed by [#2067](https://github.com/mudler/vllm.cpp/issues/2067) (PR [#2073](https://github.com/mudler/vllm.cpp/pull/2073), commit `47a2b35a5`), which authored `src/vllm/model_executor/models/glm5_next_weights.cpp` whole. `windows-msvc-cpu` and `windows-msvc-vulkan` both fail the build, so **every pull request that merges current `main` inherits a red Windows pair** — and it looks exactly like the long-standing [#584](https://github.com/mudler/vllm.cpp/issues/584) crash in `gh pr checks`, same two job names and same red, distinguishable only by reading the log: #584 carries exit `-1073740791` and zero `error C####`, this carries one `error C2220` and no crash code. **The issue's stated cause is not the mechanism, and the correction matters for the fix.** Sibling scopes do not hide one another, so the five loops named in #2101 do not shadow each other; every one of them shadows `const size_t n` at `glm5_next_weights.cpp:252`, the layer-count local the function's own bounds check uses. Renaming loop variables to be distinct *from each other* would therefore have left the defect in place. **CI reported four sites and there are seven**, because MSVC stops at the first `error C2220`: 276, 279, 285 and 287 reached the log; 288, 293 and 299 never did. Found with GCC's `-Wshadow`, whose `shadows a previous local` diagnostic is the exact analogue of C4456 and which names all seven at once — red-before `rc=1`, green-after `rc=0` on the same command. Fixed by naming the function-scope local `layer_count` for what it is, which removes all seven hidings at their source, and by naming the seven loop variables `tn` so no bland one-letter name can collide there again. No pragma, no `/WX` relaxation, no suppression: the warning is correct. Green-after for the Windows build itself is the CI job, which cannot be run on this fleet | bug |

## Resolution

-
