ID: ISSUE-GH-2431
Title: sanitize-cpu: 384 bytes leaked in 4 allocations from test_glm_moe_dsa_schedule's own device buffers
Row: MODEL-TEXT
State: OPEN
Kind: UNKNOWN
GitHub: 2431
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `-`
>
> Second of the two defects keeping `sanitize-cpu (address,undefined)` red on `main`.
> The other is the out-of-bounds write in `MakeTensor` (#2430). This one is a leak, and
> it is what makes the job exit 8 even after the UBSan failure is accounted for.
>
> ## The defect
>
> ```
> ==36575==ERROR: LeakSanitizer: detected memory leaks
> Direct leak of 128 byte(s) in 1 object(s) allocated from:
>     #0 aligned_alloc
>     #1 AllocAligned64 src/vt/cpu/cpu_backend.cpp:20
>     #2 Alloc          src/vt/cpu/cpu_backend.cpp:42
>     #3 operator()     tests/vllm/models/test_glm_moe_dsa_schedule.cpp:680
>     #4 operator()     tests/vllm/models/test_glm_moe_dsa_schedule.cpp:686
>     #5 DOCTEST_ANON_FUNC_39 tests/vllm/models/test_glm_moe_dsa_schedule.cpp:708
> SUMMARY: AddressSanitizer: 384 byte(s) leaked in 4 allocation(s).
> ```
>
> Four device allocations taken through `Backend::Alloc` inside the test case at
> `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:708` are never released. 384 bytes
> total, 128 in the largest.
>
> The allocations are the test's own, not product code: frames #3-#5 are all in the test
> file. The fix belongs with whatever owns that case's buffer lifetime.
>
> ## One thing to check before fixing
>
> The DevicePool deliberately RETAINS blocks, so a naive local reproduction reads as a
> leak when it is not. CI's sanitize lane sets `VT_POOL_BYPASS=1` for exactly this reason.
> Reproduce with that variable set, or the result is not comparable to the CI finding.
> A local run without it will show a much larger figure that is not a defect.
>
> ## Scope
>
> Pre-existing on `main`; observed on #2414, #2397 and #2396, none of which touch this
> file. #2414's entire diff is one workflow line and a comment, so the lane is red
> independently of what is under review.
>
> ## Adjacent observation, not filed separately
>
> In the same run `test_dots3_note_scaffold` (#198 of 686) took **1499.36 sec** -- roughly
> 25 minutes for one case, and the single largest contributor to that job's 2h09m wall
> clock. Worth a look on its own terms; a sanitizer build is slow, but 25 minutes for one
> scaffold case is out of line with its neighbours, which finish in under a second.
>

## Resolution

-
