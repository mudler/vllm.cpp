ID: ISSUE-GH-1575
Title: **`build-newest-gcc` has been RED on `main` since `5702d8f83`, and it is a recurrence of the class `tests/support/process_id.h` was created to close, not a missing include.** `tests/vllm/models/test_qwen3_dflash2_gguf.cpp:547` calls `::getpid()` while including nothing that declares it; the lane builds in a `gcc:16` container where libstdc++ no longer supplies the transitive `<unistd.h>`, so it fails to COMPILE (`error: '::getpid' has not been declared; did you mean 'getpt'?`) while every local toolchain at gcc 13 stays green. The seam's own header predicts this: it says the spelling "was fixed once in three files and came back in five more, because each new loader test copies the temp-directory helper from the last one", which is exactly what W5's GGUF test did. The convention is otherwise unanimous — every other file in the tree naming `getpid` either includes `<unistd.h>` or uses the helper, and the sibling `tests/vllm/entrypoints/test_dspark_draft_routing.cpp` uses the helper AND carries a comment warning not to reintroduce this class. ATTRIBUTED, not inferred: `build-newest-gcc` is absent from the baseline's failed list at `92406c620` and present at `5702d8f83`, the commit that added the file. MEASURED red-first in the lane's own container at base `947e5f648`, file sha256 `83bba319…`: unmodified `RED_RC=1` with the error byte-identical to CI; with the seam applied `GREEN_RC=0`, `compile_err=0`, `git diff --stat` confirming the edit applied; local gcc 13 `GCC13_RC=0`, so no regression on the shipped toolchain. FIXED IN FLOW with the portable spelling (`#include "support/process_id.h"`, `vllm_test::ProcessId()`) rather than `<unistd.h>`, which would work on POSIX but re-copies the idiom the helper centralises and does not compile on MSVC. Found while measuring [#1464](https://github.com/mudler/vllm.cpp/issues/1464) at `origin/main`
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1575
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:555`

### Frozen archive evidence

> | [#1575](https://github.com/mudler/vllm.cpp/issues/1575) | `SPEC-DFLASH2` | **`build-newest-gcc` has been RED on `main` since `5702d8f83`, and it is a recurrence of the class `tests/support/process_id.h` was created to close, not a missing include.** `tests/vllm/models/test_qwen3_dflash2_gguf.cpp:547` calls `::getpid()` while including nothing that declares it; the lane builds in a `gcc:16` container where libstdc++ no longer supplies the transitive `<unistd.h>`, so it fails to COMPILE (`error: '::getpid' has not been declared; did you mean 'getpt'?`) while every local toolchain at gcc 13 stays green. The seam's own header predicts this: it says the spelling "was fixed once in three files and came back in five more, because each new loader test copies the temp-directory helper from the last one", which is exactly what W5's GGUF test did. The convention is otherwise unanimous — every other file in the tree naming `getpid` either includes `<unistd.h>` or uses the helper, and the sibling `tests/vllm/entrypoints/test_dspark_draft_routing.cpp` uses the helper AND carries a comment warning not to reintroduce this class. ATTRIBUTED, not inferred: `build-newest-gcc` is absent from the baseline's failed list at `92406c620` and present at `5702d8f83`, the commit that added the file. MEASURED red-first in the lane's own container at base `947e5f648`, file sha256 `83bba319…`: unmodified `RED_RC=1` with the error byte-identical to CI; with the seam applied `GREEN_RC=0`, `compile_err=0`, `git diff --stat` confirming the edit applied; local gcc 13 `GCC13_RC=0`, so no regression on the shipped toolchain. FIXED IN FLOW with the portable spelling (`#include "support/process_id.h"`, `vllm_test::ProcessId()`) rather than `<unistd.h>`, which would work on POSIX but re-copies the idiom the helper centralises and does not compile on MSVC. Found while measuring [#1464](https://github.com/mudler/vllm.cpp/issues/1464) at `origin/main` | bug |

## Resolution

-
