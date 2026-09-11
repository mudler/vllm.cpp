ID: ISSUE-GH-1954
Title: test_backend_cross_device: MoeSiluMul bf16 exactness reds on gfx1200 ROCm too, the ROCm counterpart of #1802
Row: -
State: OPEN
Kind: bug
GitHub: 1954
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-26
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `test_backend_cross_device`'s `MoeSiluMul matches the CPU oracle within
> NMSE <= 5e-4` case fails on ROCm gfx1200, at `:2063`,
> `CHECK(got == ref_b)`. This is the ROCm counterpart of the bookkeeping
> [#1802](https://github.com/mudler/vllm.cpp/issues/1802) did for sm_110,
> which already carries the identical test name and the identical failing
> assertion for CUDA.
>
> [#907](https://github.com/mudler/vllm.cpp/issues/907) recorded GB10's own
> red sweep and is related but distinct, not a second match for the same
> test. It documents `test_cuda_ops`'s "CUDA silu_and_mul matches CPU" case
> failing at 439 of 440 assertions, a different test and a different
> assertion in the same defect family (silu-and-mul last-digit numerics), not
> the same test or the same assertion `test_backend_cross_device` fails here.
>
> No entry for gfx1200 or for ROCm exists anywhere in `.agents/environment.md`.
>
> ## Confirmed pre-existing, not introduced by a change in flight
>
> Found during the fresh review of
> [row/ROCM-KQUANT-NWARPS-DECODE](https://github.com/mudler/vllm.cpp/issues/1910).
> The review reverted only that row's two changed files back to the parent
> commit, in a scratch copy, and rebuilt. The failure reproduces byte for byte:
>
> | | parent commit | reviewed commit |
> |---|---|---|
> | cases | 24 passed, 1 failed of 25 | 25 passed, 1 failed of 26 |
> | assertions | 1 failed of 80195 | 1 failed of 80253 |
>
> The one new case and 58 new assertions the row adds all pass. The delta
> between the two rows is exactly the row's own test, nothing else.
>
> ## Whether this is the same root cause as #1802's CUDA entry
>
> Not established here. #1802 records the CUDA sm_110 vectors as differing "in
> the last digit of a handful of elements," consistent with an ordinary bf16
> rounding difference in a hardware-specific kernel implementation. ROCm's
> kernel is a separate implementation from CUDA's, so the two reds could be
> independent tolerance gaps in the same test rather than one shared defect.
> Nobody has compared the actual mismatching elements across the two backends.
>
> ## What is NOT established
>
> - No comparison of which elements mismatch on ROCm against which elements
>   mismatch on CUDA.
> - No attribution to a specific `MoeSiluMul` kernel line on either backend.
> - No speed or correctness claim beyond "this one assertion currently fails."
>
> ## Owed
>
> Add a gfx1200 row to `.agents/environment.md`'s known-red table, mirroring
> the existing sm_110 and GB10 rows, so a future gfx1200 run does not
> re-report this as new.
>

## Resolution

-
