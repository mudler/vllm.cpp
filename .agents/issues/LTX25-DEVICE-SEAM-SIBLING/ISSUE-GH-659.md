ID: ISSUE-GH-659
Title: LTX-2.5 device select adopts M3a's platform seam but not its companion capability guard: `ltx2_video.cpp` asks `CurrentPlatform().device_type()` and `TryGetBackend(...)` but never `supports_model_architecture`, so a PARTIAL backend (Metal 15/75 ops, Tenstorrent) is handed a queue and dies in a kernel bind where it used to be refused BY NAME (found while reviewing #553 for landing)
Row: LTX25-DEVICE-SEAM-SIBLING
State: UNKNOWN
Kind: bug
GitHub: 659
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:188`

### Frozen archive evidence

> | [#659](https://github.com/mudler/vllm.cpp/issues/659) | — | LTX-2.5 device select adopts M3a's platform seam but not its companion capability guard: `ltx2_video.cpp` asks `CurrentPlatform().device_type()` and `TryGetBackend(...)` but never `supports_model_architecture`, so a PARTIAL backend (Metal 15/75 ops, Tenstorrent) is handed a queue and dies in a kernel bind where it used to be refused BY NAME (found while reviewing #553 for landing) | bug |

## Resolution

-
