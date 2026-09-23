ID: ISSUE-LOCAL-01M36SK0ER1S8AEHQ81QX4GED0
Title: ROCm: --allow-multiple-definition is PRIVATE on static lib vllm, never reaches vllm_shared link step
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

The --allow-multiple-definition linker flag at CMakeLists.txt:1872 is set as target_link_options(vllm PRIVATE ...). Since vllm is a STATIC library, PRIVATE link options have no effect (static archives are not linked). The flag needs to be PUBLIC so it propagates to vllm_shared (the SHARED library that links vllm), where the actual linking happens and the multiple-definition errors for bfloat16 builtins (__float2bfloat16, __bfloat1622float2, etc.) occur. Without this fix, the HIP build on Strix (gfx1151, clang-17, ROCm 5.7) compiles all 623 objects but fails at the final link step.

## Resolution

Fixed in 0da53ba17: changed target_link_options(vllm PRIVATE ...) to PUBLIC so --allow-multiple-definition propagates to vllm_shared. Verified on Strix: all 623 objects compile and link, vllm-cli produced.
