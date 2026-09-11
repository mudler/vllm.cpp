ID: ISSUE-GH-1296
Title: `build-newest-gcc` was RED on `main`: `tests/vllm/entrypoints/test_dspark_block_size_guard.cpp:68` called `::getpid()` with nothing that declares it, landed in `b626be75a` (#1243), so every branch cut after 2026-08-18 14:41 UTC inherited the break. Reproduced under `gcc:16` (16.2.0): `error: '::getpid' has not been declared`. This is the sixth instance of the class `tests/support/process_id.h` exists to end, and that header's own comment predicts it ("fixed once in three files and came back in five more"). Fixed in flow by routing through `vllm_test::ProcessId()` exactly as the sibling `test_dspark_draft_routing.cpp` does, NOT by adding `<unistd.h>`, which MSVC does not ship. Red-before/green-after proven in one `gcc:16` container: reverting to the `::getpid()` spelling exits 1 with that error, restoring exits 0. Suite 14/14, 39 assertions on the normal toolchain. The `windows-msvc-*` lanes were already red BEFORE this file landed, so this is not claimed as their cause
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 1296
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:424`

### Frozen archive evidence

> | [#1296](https://github.com/mudler/vllm.cpp/issues/1296) | `ENG-RELEASE-WINDOWS` | `build-newest-gcc` was RED on `main`: `tests/vllm/entrypoints/test_dspark_block_size_guard.cpp:68` called `::getpid()` with nothing that declares it, landed in `b626be75a` (#1243), so every branch cut after 2026-08-18 14:41 UTC inherited the break. Reproduced under `gcc:16` (16.2.0): `error: '::getpid' has not been declared`. This is the sixth instance of the class `tests/support/process_id.h` exists to end, and that header's own comment predicts it ("fixed once in three files and came back in five more"). Fixed in flow by routing through `vllm_test::ProcessId()` exactly as the sibling `test_dspark_draft_routing.cpp` does, NOT by adding `<unistd.h>`, which MSVC does not ship. Red-before/green-after proven in one `gcc:16` container: reverting to the `::getpid()` spelling exits 1 with that error, restoring exits 0. Suite 14/14, 39 assertions on the normal toolchain. The `windows-msvc-*` lanes were already red BEFORE this file landed, so this is not claimed as their cause | bug |

## Resolution

-
