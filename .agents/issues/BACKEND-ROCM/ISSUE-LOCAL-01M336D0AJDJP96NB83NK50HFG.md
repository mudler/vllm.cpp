ID: ISSUE-LOCAL-01M336D0AJDJP96NB83NK50HFG
Title: ROCm 5.7 / clang-17 build fails on Strix (gfx1151): dead code, missing hipBLASLt guard, and __shfl_down_sync incompatibility
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: 2026-09-21

## Problem

The ROCm HIP build fails on Strix (gfx1151, ROCm 5.7, clang-17) with 10 distinct errors. Five are dead code that -Werror catches on clang-17 but not on clang-20/ROCm 7.x: unused m_split in rocm_paged_attn.hip:1333, duplicate mindiv in rocm_skinny_gemm.hip:42+68, and on_gfx1151()/SelectYtileUnrl/YtileUnrl in rocm_skinny_gemm.hip (no caller since #2787). Two are compatibility gaps: __shfl_down_sync/__shfl_sync with mask (15 call sites in rocm_grouped_gemm.hip, rocm_gdn_fused.hip, rocm_quant_dot.hip) is not defined in ROCm 5.7 HIP headers, and hipBLASLt is not installed on all ROCm systems (rocm_matmul_hipblaslt.hip:15 includes it unconditionally). Three are build configuration issues specific to the non-standard ROCm 5.7 layout at /usr: CMake 3.28 rejects hipcc (must use clang-17), --rocm-device-lib-path points to a nonexistent path, and __clang_hip_runtime_wrapper.h defines bfloat16 conversions non-inline causing multiple definition link errors. The build was confirmed working with all 10 workarounds applied via sed, producing 618/618 objects and passing the first-c1 gate (5/5 stability, ~10 tok/s). Commit af394746a added -Werror for HIP and 78725a3ce cleared 19 diagnostics on ROCm 7.2, but clang-17 raises additional diagnostics not present on clang-20. PR #3220 (KhazAkar) addresses the dead code for HIP 7.1 but is stale (213-file diff from divergence) and does not address the compatibility or build-config issues.

## Resolution

Fixed in commit 6e5272696 on branch row/BACKEND-ROCM-STRIX-BUILD. All six
categories of fix applied: dead code removed, unused m_split removed,
__shfl compat header added, hipBLASLt made optional with CMake guard,
include order fixed in ~30 .hip files, -Wl,--allow-multiple-definition
added to HIP link. CPU build verified: 2105/2105 objects, 778/779 ctest
pass (1 pre-existing failure in test_serve_low_tools, unrelated). Strix
HIP build verification pending on the GPU.
