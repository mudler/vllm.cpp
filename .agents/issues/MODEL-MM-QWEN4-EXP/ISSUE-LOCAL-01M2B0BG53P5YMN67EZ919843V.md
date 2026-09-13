ID: ISSUE-LOCAL-01M2B0BG53P5YMN67EZ919843V
Title: rocm_gdn_fused.hip states kSigmoidGateBf16 has no CUDA registration; cuda_glue.cu:475 registers one
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

src/vt/rocm/rocm_gdn_fused.hip:8 says 'kSigmoidGateBf16 has NO CUDA registration - the CPU composite semantics are the donor of record (the Vulkan lane's only native sibling)'. src/vt/cuda/cuda_glue.cu:475 registers OpId::kSigmoidGateBf16 on DeviceType::kCUDA. The claim is the same class the include/vt/ops.h sweep in ISSUE-LOCAL-01M2AYNBBZPVSG7G9PSGSGF68V corrected, and it is load-bearing in the same way: it names the donor of record for a ROCm arm, so a later porter reads it and does not compare against the CUDA kernel that exists. NOT fixed in the W3 re-review repair, and the reason is scope rather than doubt: the dispatch for that repair makes any touch of a compiled file owe the full ROCm cross-device suite on strix:gpu0, and this is a one-line comment in a .hip translation unit. It needs either that lease or an operator decision that a comment-only touch of a .hip file does not owe one.

## Resolution

-
