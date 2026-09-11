ID: ISSUE-GH-2161
Title: `test_cuda_fp8_kv_cache` went red on `main` at `191f64608` (#2080, W6). Its G1b case loops `{kMETAL, kROCM}` as the backends that register `kPagedAttention` for the float path with no fp8 dequant, and asserts the named refusal in `src/vt/ops.cpp`. W6 implemented the ROCm arm and correctly widened that refusal to `kCPU || kCUDA || kROCM`, so the ROCm leg stopped measuring a refusal and saw `GetOp`'s "no kernel for op PagedAttention on device rocm" instead — neither string the case asserts. The corrected predicate WAS written, at `tests/vt/test_rocm_fp8_kv_cache.cpp:196`, which loops Metal alone; that file is registered under `if(VLLM_CPP_HIP)`, so it never builds on the CPU tier and the CPU-visible copy was missed. Fixed in flow by dropping `kROCM` from the loop, mutation-proven: permitting `kMETAL` in `ops.cpp` reds the case again at 2 of 8 assertions
Row: KV-FP8
State: UNKNOWN
Kind: bug
GitHub: 2161
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:840`

### Frozen archive evidence

> | [#2161](https://github.com/mudler/vllm.cpp/issues/2161) | `KV-FP8` | `test_cuda_fp8_kv_cache` went red on `main` at `191f64608` (#2080, W6). Its G1b case loops `{kMETAL, kROCM}` as the backends that register `kPagedAttention` for the float path with no fp8 dequant, and asserts the named refusal in `src/vt/ops.cpp`. W6 implemented the ROCm arm and correctly widened that refusal to `kCPU \|\| kCUDA \|\| kROCM`, so the ROCm leg stopped measuring a refusal and saw `GetOp`'s "no kernel for op PagedAttention on device rocm" instead — neither string the case asserts. The corrected predicate WAS written, at `tests/vt/test_rocm_fp8_kv_cache.cpp:196`, which loops Metal alone; that file is registered under `if(VLLM_CPP_HIP)`, so it never builds on the CPU tier and the CPU-visible copy was missed. Fixed in flow by dropping `kROCM` from the loop, mutation-proven: permitting `kMETAL` in `ops.cpp` reds the case again at 2 of 8 assertions | bug |

## Resolution

-
