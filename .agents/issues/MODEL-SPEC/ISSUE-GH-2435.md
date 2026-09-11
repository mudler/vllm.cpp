ID: ISSUE-GH-2435
Title: dense_device_glue writes past Tensor::shape[kMaxRank] for rank>4, and leaks 384 bytes
Row: MODEL-SPEC
State: OPEN
Kind: UNKNOWN
GitHub: 2435
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
> `include/vllm/model_executor/models/dense_device_glue.h:56-58` writes `t.shape[i]`,
> `t.stride[i]` and reads `t.shape[i]` for `i` up to `rank-1`, but `vt::Tensor` fixes
> `kMaxRank = 4` (`include/vt/tensor.h:12`) with `int64_t shape[kMaxRank]`. Any shape of
> rank 5 or more overflows both fixed arrays. There is no bounds check on `rank`.
>
> Measured on pure `origin/main` `7bafd4224`, no pull request applied, Debug build with
> `-fsanitize=address,undefined`:
>
> ```
> dense_device_glue.h:56:14: runtime error: index 4 out of bounds for type 'long int [4]'
> dense_device_glue.h:57:15: runtime error: index 4 out of bounds for type 'long int [4]'
> dense_device_glue.h:58:21: runtime error: index 4 out of bounds for type 'long int [4]'
> ==308780==ERROR: LeakSanitizer: detected memory leaks
> SUMMARY: AddressSanitizer: 384 byte(s) leaked in 4 allocation(s)
> ```
>
> Reproduce: build `test_qwen4_exp_layer_loop` with `address,undefined` and run it. The
> doctest assertions all pass (309/309, `SUCCESS!`); the process still exits 1 on the
> sanitizer. That combination is why no ordinary gate sees this - the test reports success
> and only the sanitizer lane disagrees.
>
> This is what reddens `sanitize-cpu (address,undefined)` on open pull requests. It is not
> caused by any of them. It was attributed here by running the same test on `main` alone.
>
> Owed: a rank bound at the write site, and the 4-allocation leak in the same path.

## Resolution

-
