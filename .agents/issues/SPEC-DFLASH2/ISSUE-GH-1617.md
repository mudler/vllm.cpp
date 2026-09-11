ID: ISSUE-GH-1617
Title: **`d27639e71` re-added an unguarded `#include <unistd.h>` to `tests/vllm/models/test_qwen3_dflash2_gguf.cpp`, a file [#1581](https://github.com/mudler/vllm.cpp/pull/1581) (`13548db8f`) had already fixed through the `process_id` seam.** The include supports NOTHING: the file carries no `::getpid` call and includes `support/process_id.h` at line 68. It is the [#603](https://github.com/mudler/vllm.cpp/issues/603) shape of the [#503](https://github.com/mudler/vllm.cpp/issues/503) class -- MSVC ships no `<unistd.h>`, so an unguarded include does not fail on Windows, it does not COMPILE, and `tests/support/process_id.h` says so in its own header comment. It did not surface as a new red because both `windows-msvc-*` lanes are baseline-red and never run on `main`. ORDERING is the whole cause: the change was authored against the pre-#1581 tree, where it was correct, and landed after #1581 had removed the call it existed to support. FIXED IN FLOW by deleting the one line; found while merging external contributor pull requests, not owned by that work
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1617
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:577`

### Frozen archive evidence

> | [#1617](https://github.com/mudler/vllm.cpp/issues/1617) | `SPEC-DFLASH2` | **`d27639e71` re-added an unguarded `#include <unistd.h>` to `tests/vllm/models/test_qwen3_dflash2_gguf.cpp`, a file [#1581](https://github.com/mudler/vllm.cpp/pull/1581) (`13548db8f`) had already fixed through the `process_id` seam.** The include supports NOTHING: the file carries no `::getpid` call and includes `support/process_id.h` at line 68. It is the [#603](https://github.com/mudler/vllm.cpp/issues/603) shape of the [#503](https://github.com/mudler/vllm.cpp/issues/503) class -- MSVC ships no `<unistd.h>`, so an unguarded include does not fail on Windows, it does not COMPILE, and `tests/support/process_id.h` says so in its own header comment. It did not surface as a new red because both `windows-msvc-*` lanes are baseline-red and never run on `main`. ORDERING is the whole cause: the change was authored against the pre-#1581 tree, where it was correct, and landed after #1581 had removed the call it existed to support. FIXED IN FLOW by deleting the one line; found while merging external contributor pull requests, not owned by that work | bug |

## Resolution

-
