ID: ISSUE-LOCAL-01M2CCNA0S74WT5WBV50B3VD0W
Title: OWED: the DeviceMemoryIsHostAddressable and Synchronize terms of MaybeReleaseStagedBorrowSource are UNGATED
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Deleting backend.DeviceMemoryIsHostAddressable() from MaybeReleaseStagedBorrowSource (qwen3_5_weights.cpp) leaves the focused suite 25/25 green with the binary proven changed, and so does deleting backend.Synchronize(queue). Both survive because of the harness, not because the code is wrong. The fake backend in tests/vllm/model_executor/test_resident_weight_host_addressable.cpp answers DeviceMemoryIsHostAddressable() false unconditionally, so only the platform half of the host-addressability pair is ever exercised; and HostBackend::Copy is a synchronous memcpy that does not override Synchronize, so no case in this tree can express a DMA still reading the source pages when they are dropped. Convicting the first needs a backend answering true while the platform answers false, a combination no fleet device presents and that the file's process-global registrar cannot hold beside the existing one. Convicting the second needs a backend whose Copy defers. Recorded as owed rather than repaired by weakening or contorting a passing test. See .agents/specs/rocm-host-residency-after-upload.md, its Ungated-guarantees section.

## Resolution

-
