ID: ISSUE-GH-558
Title: `tests/parity/hf_snapshot.h` has no guard against declaration-order breaks: the C++ build catches them, but the records-only lane that broke it never builds C++, and all 14 TUs that include the header are checkpoint-gated so `ctest` reports the break as `***Not Run`. `fafa16f0f` (#546, #551) fixed the ordering and carried no guard
Row: -
State: UNKNOWN
Kind: bug
GitHub: 558
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:181`

### Frozen archive evidence

> | [#558](https://github.com/mudler/vllm.cpp/issues/558) | — | `tests/parity/hf_snapshot.h` has no guard against declaration-order breaks: the C++ build catches them, but the records-only lane that broke it never builds C++, and all 14 TUs that include the header are checkpoint-gated so `ctest` reports the break as `***Not Run`. `fafa16f0f` (#546, #551) fixed the ordering and carried no guard | bug |

## Resolution

-
