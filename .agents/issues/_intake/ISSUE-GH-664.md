ID: ISSUE-GH-664
Title: **FIXED 2026-08-13, `row/FIX-WINDOWS-POSIX-VIDEO-ENGINE`.** `windows-msvc-cpu` / `windows-msvc-vulkan` were RED on EVERY open PR (9 sampled across 5 unrelated lanes) from ONE file: `video_engine.cpp` reached Windows with `<sys/stat.h>`, `::stat` and `S_ISDIR` (landed `cefacd2d0`), which `check-windows-portability.py:1675-1688` flags under `full_source_posix` — every scanned source, not only the platform-boundary set. `main` was never a denominator because the Windows jobs are PR-only and `skipped` on push (#584). Repaired at the SOURCE, not the checker: `IsDir`/`Exists` now take the `std::error_code` <filesystem> overloads through a file-local `NativePath`, which preserves `::stat`'s return-false-for-an-uninspectable-path behaviour that the THROWING overloads would have turned into a `filesystem_error` escaping a registry query
Row: -
State: UNKNOWN
Kind: bug
GitHub: 664
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:191`

### Frozen archive evidence

> | [#664](https://github.com/mudler/vllm.cpp/issues/664) | — | **FIXED 2026-08-13, `row/FIX-WINDOWS-POSIX-VIDEO-ENGINE`.** `windows-msvc-cpu` / `windows-msvc-vulkan` were RED on EVERY open PR (9 sampled across 5 unrelated lanes) from ONE file: `video_engine.cpp` reached Windows with `<sys/stat.h>`, `::stat` and `S_ISDIR` (landed `cefacd2d0`), which `check-windows-portability.py:1675-1688` flags under `full_source_posix` — every scanned source, not only the platform-boundary set. `main` was never a denominator because the Windows jobs are PR-only and `skipped` on push (#584). Repaired at the SOURCE, not the checker: `IsDir`/`Exists` now take the `std::error_code` <filesystem> overloads through a file-local `NativePath`, which preserves `::stat`'s return-false-for-an-uninspectable-path behaviour that the THROWING overloads would have turned into a `filesystem_error` escaping a registry query | bug |

## Resolution

-
