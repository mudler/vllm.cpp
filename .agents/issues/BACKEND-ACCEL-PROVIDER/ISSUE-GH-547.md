ID: ISSUE-GH-547
Title: GB10 reports `UnifiedMemory()` true, so `ReferenceTierEligible(kCUDA)` runs the CPU host kernel over `cudaMalloc` pointers; it needs `DeviceMemoryIsHostAddressable()`
Row: BACKEND-ACCEL-PROVIDER
State: UNKNOWN
Kind: bug
GitHub: 547
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:93`

### Frozen archive evidence

> | [#547](https://github.com/mudler/vllm.cpp/issues/547) | — | GB10 reports `UnifiedMemory()` true, so `ReferenceTierEligible(kCUDA)` runs the CPU host kernel over `cudaMalloc` pointers; it needs `DeviceMemoryIsHostAddressable()` | bug |

## Resolution

-
