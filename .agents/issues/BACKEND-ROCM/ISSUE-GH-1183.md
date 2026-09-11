ID: ISSUE-GH-1183
Title: ROCm skinny GEMM architecture eligibility caches the first device
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 1183
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-18
Updated: 2026-08-18
Closed: 2026-08-18

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> PR #506 adds the wave32-only ROCm wvSplitK skinny GEMM. Its dispatcher currently implements:
>
> ```cpp
> bool SkinnyGemmArchOk(int device_index) {
>   static const std::string arch = vt::rocm::DeviceArchName(device_index);
>   ...
> }
> ```
>
> The function accepts a device index but caches only the first device's architecture. In a heterogeneous process, a first call on gfx11/gfx12 poisons later calls on gfx9 as eligible. The adjacent source comment says the unported wave64 arm would then produce silently wrong sums.
>
> A fresh integration reviewer found this while rebasing #506 onto main's dual-slot GetBlas/device-hop seam. The existing focused test uses only device index 0, so it cannot detect the defect.
>
> ## Required fix
>
> - Cache/evaluate eligibility by device, without adding a per-token HIP property probe or a hot-path global lock.
> - Add a RED-first device-hop test using a deterministic architecture resolver/cache seam, because this host's four GPUs are all gfx1100.
> - Preserve the wave32-only gfx11/gfx12 policy, unknown-architecture refusal, main's dual-slot GetBlas behavior, and all existing wvSplitK guards.
> - Mutation-prove that collapsing eligibility back to the first device makes the new test fail.
>
> ## Ownership
>
> Owning row: `BACKEND-ROCM`. Fix in the existing PR #506 flow before its rebased head is pushed.

## Resolution

GitHub records closing pull request #506 (https://github.com/mudler/vllm.cpp/pull/506) merged on 2026-08-21 as commit `f38c1edc4ea679348f856c0c0b20fb0702f77daf`. GitHub closed issue #1183 on 2026-08-18.
