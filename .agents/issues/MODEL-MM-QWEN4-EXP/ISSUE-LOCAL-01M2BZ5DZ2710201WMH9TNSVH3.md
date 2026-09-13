ID: ISSUE-LOCAL-01M2BZ5DZ2710201WMH9TNSVH3
Title: gfx1151: the whole GGUF is prefaulted and never released, so a 67.56 GiB load wedges in svm_range_set_attr
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

On gfx1151 the 67.56 GiB Qwen3.8-Flash-Next UD-IQ1_S loads with zero op refusals and then wedges forever in svm_range_set_attr (uninterruptible, gpu_busy 0) on a ~31 GiB host. Two host-residency defects compound. (1) Every borrowed GGUF span is synchronously prefaulted -- madvise(WILLNEED) plus a one-byte-per-page touch loop over the whole span (PrefaultBorrowedSpan, qwen3_5_gguf_weights.cpp) -- faulting in 65.488 GiB that is read exactly once. (2) Nothing releases those pages after the device upload: ResidentWeight's staging arm copies the borrow to the device and leaves every source page mapped for the process lifetime, and OwnedTensor::ReleaseHost() refuses to madvise a BORROWED buffer on the argument that clean file-backed pages need no help. That argument holds against the page reclaimer and fails against the KFD's resident-system-memory accounting. llama.cpp (pin 10bf611e5) does the opposite on both counts: MAP_POPULATE plus an advisory posix_madvise with no synchronous touch loop, and unmap_fragment(0, mmap_used.first) after load, which for a fully offloaded model munmaps the entire mapping.

## Resolution

-
