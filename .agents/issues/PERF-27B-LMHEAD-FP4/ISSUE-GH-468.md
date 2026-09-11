ID: ISSUE-GH-468
Title: `VT_GDN_FP8_ALPHA_IN_CONV`'s model-layer wiring has no test at any tier: the fp8 matmul registers only on CUDA (`cuda_matmul.cu:827-829`) so CPU cannot reach it, and `ProjectGdnFp8QkvzForTest` is called with DEFAULT args, so nothing pins the 8 call sites forwarding `mixed_scale`. Lever parked on `row/PERF-ALPHA-IN-CONV-PARKED`
Row: PERF-27B-LMHEAD-FP4
State: UNKNOWN
Kind: bug
GitHub: 468
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:118`

### Frozen archive evidence

> | [#468](https://github.com/mudler/vllm.cpp/issues/468) | `PERF-27B-LMHEAD-FP4` | `VT_GDN_FP8_ALPHA_IN_CONV`'s model-layer wiring has no test at any tier: the fp8 matmul registers only on CUDA (`cuda_matmul.cu:827-829`) so CPU cannot reach it, and `ProjectGdnFp8QkvzForTest` is called with DEFAULT args, so nothing pins the 8 call sites forwarding `mixed_scale`. Lever parked on `row/PERF-ALPHA-IN-CONV-PARKED` | bug |

## Resolution

-
