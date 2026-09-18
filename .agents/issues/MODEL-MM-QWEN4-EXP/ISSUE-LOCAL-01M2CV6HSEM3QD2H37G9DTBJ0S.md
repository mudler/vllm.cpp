ID: ISSUE-LOCAL-01M2CV6HSEM3QD2H37G9DTBJ0S
Title: ROCm pinned H2D ring is allocated before the decision that would use it
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

RocmBackend::StagedCopy evaluated in.ring_available = EnsureRing(chunk) before ShouldStageH2D, so the first copy of 64 MiB or more on a non-capturing stream allocated 256 MiB of pinned host memory and four events whatever the pointer kinds turned out to be. Measured on gfx1151 with VT_ROCM_MANAGED_ALLOC=1: staged=0 direct=3 chunks=0 ring_bytes=268435456, a ring built and never touched, on the arm whose managed ceiling .agents/environment.md measures as bounded by host RAM. The device case printed the number and asserted nothing about it. A second, narrower defect rides with it: in.stream_capturing was hardcoded false while a separate early return did the work, so the truth table gated a value production never supplied.

## Resolution

Fixed 2026-09-13. EnsureRing is now the LAST term StagedCopy evaluates: the header splits the four cheap terms out as StagingTermsExceptRing, production asks those first and calls the allocator only when they pass, and ShouldStageH2D is spelled as StagingTermsExceptRing(in) && in.ring_available so it stays the single authority the truth table gates. A new 480-input case in tests/vt/test_rocm_pinned_h2d.cpp asserts the two agree over the whole table so the split cannot drift. The capture probe now feeds in.stream_capturing instead of a separate early return, so the guard is the predicate term the table describes; StreamIsCapturing's own return value stays ungated on a board and .agents/specs/rocm-chunked-pinned-h2d.md section 6 says why. RED on strix:gpu0 at c794b5dda with VT_ROCM_MANAGED_ALLOC=1: FAILURE, 1 of 4 assertions failed, ring_bytes=268435456. GREEN at 33a1eaaf8: 4/4, ring_bytes=0. Gates: test_rocm_pinned_h2d 10/1527, cross-device 61/84841, -tc=*DSA* 2/273, -tc=*pinned bounce* 1/8, every selector printing the ROCm-board line. Evidence /workspace/vtchunked/repair-20260913-073233/.
