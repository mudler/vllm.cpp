ID: ISSUE-GH-1829
Title: **`src/vllm/model_executor/models/dots3_note.h:282` declares `[[noreturn]] static ForwardLogits ForwardDevice(...)` -- `[[noreturn]]` on a non-void return type -- and MSVC's C4646 plus warnings-as-errors turns that into `error C2220` at `dots3_note.cpp(606,31)`, so the whole `vllm` project fails to COMPILE on Windows.** GCC and Clang accept the declaration silently, so `build-test-cpu`, `build-test-cpu-arm64`, `build-newest-gcc` and `verify (cpu)` are all green on the same commit. Found on the `windows-msvc-cpu` job of #1821, whose own diff is four files and none of them this one; the declaration arrived with `849a7dd73` (#1805) which is an ancestor of `af320abb2`, so it is INHERITED and every pull request branched from `main` since carries it. It landed unseen because `windows-msvc-cpu` and `windows-msvc-vulkan` are PULL-REQUEST-ONLY jobs with no `main` baseline, which is the second half of the defect: `main` receives no Windows verdict at all. **NOT #584** -- that is the runtime `exit -1073740791` STATUS_STACK_BUFFER_OVERRUN in `test_openai_api_server.exe`, and a reader who stops at the job name will wave this real break through as the known one. Not repaired in flow by #1821: `ForwardDevice` overrides into a registry hook and cannot simply become `void`, so the fix is a semantic decision in an actively-developed file the `MODEL-MM-dots3-note` row owns, and #1821 has no MSVC to verify one against
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 1829
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:675`

### Frozen archive evidence

> | [#1829](https://github.com/mudler/vllm.cpp/issues/1829) | `MODEL-MM-dots3-note` | **`src/vllm/model_executor/models/dots3_note.h:282` declares `[[noreturn]] static ForwardLogits ForwardDevice(...)` -- `[[noreturn]]` on a non-void return type -- and MSVC's C4646 plus warnings-as-errors turns that into `error C2220` at `dots3_note.cpp(606,31)`, so the whole `vllm` project fails to COMPILE on Windows.** GCC and Clang accept the declaration silently, so `build-test-cpu`, `build-test-cpu-arm64`, `build-newest-gcc` and `verify (cpu)` are all green on the same commit. Found on the `windows-msvc-cpu` job of #1821, whose own diff is four files and none of them this one; the declaration arrived with `849a7dd73` (#1805) which is an ancestor of `af320abb2`, so it is INHERITED and every pull request branched from `main` since carries it. It landed unseen because `windows-msvc-cpu` and `windows-msvc-vulkan` are PULL-REQUEST-ONLY jobs with no `main` baseline, which is the second half of the defect: `main` receives no Windows verdict at all. **NOT #584** -- that is the runtime `exit -1073740791` STATUS_STACK_BUFFER_OVERRUN in `test_openai_api_server.exe`, and a reader who stops at the job name will wave this real break through as the known one. Not repaired in flow by #1821: `ForwardDevice` overrides into a registry hook and cannot simply become `void`, so the fix is a semantic decision in an actively-developed file the `MODEL-MM-dots3-note` row owns, and #1821 has no MSVC to verify one against | bug |

## Resolution

-
