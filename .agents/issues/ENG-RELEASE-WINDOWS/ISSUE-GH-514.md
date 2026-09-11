ID: ISSUE-GH-514
Title: `tests/vt/test_backend_cross_device.cpp` uses POSIX `setenv`/`unsetenv`, so both windows-msvc jobs fail on EVERY pr while `main` stays green (they are skipped on push)
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 514
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:91`

### Frozen archive evidence

> | [#514](https://github.com/mudler/vllm.cpp/issues/514) | `ENG-RELEASE-WINDOWS` | `tests/vt/test_backend_cross_device.cpp` uses POSIX `setenv`/`unsetenv`, so both windows-msvc jobs fail on EVERY pr while `main` stays green (they are skipped on push) | bug |

## Resolution

-
