ID: ISSUE-GH-1206
Title: `docs/USAGE.md` invoked `./build/vllm-server` and `./build/vllm-cli` in THREE command lines (`grep -c 'build/vllm-' docs/USAGE.md` == 3 at this branch's base `fd64c76ee`, and the same at the merge base; lines 2354, 3835 and 3848), and `examples/CMakeLists.txt` builds both under `examples/`, so the binaries are at `build/examples/vllm-server` and `build/examples/vllm-cli`. The same document already used the correct spelling elsewhere (`build/examples/vllm-cli` in "Running inference (CLI)"), so a reader got two answers and one of them failed with "No such file or directory". Verified on a fresh `cmake -S . -B build -G Ninja && cmake --build build`: `ls build/vllm-server build/vllm-cli` reports no such file, and `find build -maxdepth 2` finds both under `build/examples/`. FOUND while closing [#1127](https://github.com/mudler/vllm.cpp/issues/1127) and [#1135](https://github.com/mudler/vllm.cpp/issues/1135), which add command lines to the same two sections, and FIXED IN THAT FLOW: all three spellings now name `build/examples/`, and the count is 0 at head. The row first said FIVE, which was never measured; the reviewer of [#1216](https://github.com/mudler/vllm.cpp/pull/1216) counted three and the row was corrected while that pull request was still open. The FIX was always complete — only the count was wrong. No checker covers a command line in a document, so this is a reading rather than a gate
Row: ENG-RESIDENCY-CONFIG
State: UNKNOWN
Kind: bug
GitHub: 1206
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:377`

### Frozen archive evidence

> | [#1206](https://github.com/mudler/vllm.cpp/issues/1206) | `ENG-RESIDENCY-CONFIG` | `docs/USAGE.md` invoked `./build/vllm-server` and `./build/vllm-cli` in THREE command lines (`grep -c 'build/vllm-' docs/USAGE.md` == 3 at this branch's base `fd64c76ee`, and the same at the merge base; lines 2354, 3835 and 3848), and `examples/CMakeLists.txt` builds both under `examples/`, so the binaries are at `build/examples/vllm-server` and `build/examples/vllm-cli`. The same document already used the correct spelling elsewhere (`build/examples/vllm-cli` in "Running inference (CLI)"), so a reader got two answers and one of them failed with "No such file or directory". Verified on a fresh `cmake -S . -B build -G Ninja && cmake --build build`: `ls build/vllm-server build/vllm-cli` reports no such file, and `find build -maxdepth 2` finds both under `build/examples/`. FOUND while closing [#1127](https://github.com/mudler/vllm.cpp/issues/1127) and [#1135](https://github.com/mudler/vllm.cpp/issues/1135), which add command lines to the same two sections, and FIXED IN THAT FLOW: all three spellings now name `build/examples/`, and the count is 0 at head. The row first said FIVE, which was never measured; the reviewer of [#1216](https://github.com/mudler/vllm.cpp/pull/1216) counted three and the row was corrected while that pull request was still open. The FIX was always complete — only the count was wrong. No checker covers a command line in a document, so this is a reading rather than a gate | bug |

## Resolution

-
