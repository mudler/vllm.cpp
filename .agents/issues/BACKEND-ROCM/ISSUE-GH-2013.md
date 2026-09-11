ID: ISSUE-GH-2013
Title: index row for #1954 wrongly claims #907 documents the same test and assertion
Row: BACKEND-ROCM
State: OPEN
Kind: record
GitHub: 2013
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> The [#1954](https://github.com/mudler/vllm.cpp/issues/1954) row in
> `.agents/issue-index.md` claims issue #907 documents "the same test name and
> the same assertion" as #1954 on GB10. This is false.
>
> #907 records `test_cuda_ops`'s "CUDA silu_and_mul matches CPU" case failing
> at 439 of 440 assertions on GB10. #1954 is about
> `test_backend_cross_device`'s "MoeSiluMul matches the CPU oracle within
> NMSE <= 5e-4" case, at `:2063`, on gfx1200 ROCm. Different test, different
> assertion. Both are last-digit bf16 numerics in a silu-and-mul kernel, so
> they are the same defect family, but #907 is not a second recorded instance
> of #1954's specific test and assertion.
>
> #1954's own GitHub title and body have already been corrected to state this
> plainly. `.agents/environment.md` and `.agents/specs/rocm-kquant-nwarps-decode.md`
> `## Owed` have also been corrected on `row/ROCM-KQUANT-NWARPS-DECODE`. The
> `.agents/issue-index.md` row cannot be corrected in place: the file is
> append-only, and `scripts/check-agent-record.py` refuses a second row citing
> an issue number already in the index. This issue exists so a corrective row
> can cite it instead, following the same pattern as
> [#1339](https://github.com/mudler/vllm.cpp/issues/1339) superseding #1280 and
> [#1796](https://github.com/mudler/vllm.cpp/issues/1796) retracting #1456.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-
