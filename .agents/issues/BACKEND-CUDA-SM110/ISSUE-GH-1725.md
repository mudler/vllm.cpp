ID: ISSUE-GH-1725
Title: `kMatmulFp8Cutlass` and `kMatmulFp8BlockScaled` fall through to the portable CPU tier and SEGFAULT on sm_110 (`test_ops_fp8_cutlass:191`, `test_ops_matmul_fp8_block_cuda:345`, measured at `0764ded2b`). The sm_110 baseline attributed them to [#960](https://github.com/mudler/vllm.cpp/issues/960), which was CLOSED COMPLETED three days earlier by `d607fec4c` -- that fix covered `QuantFp8Static` only, while these two ops are registered from TUs `CMakeLists.txt:1790-1791` builds solely for `VT_CUTLASS_FP8_ARCHS`. `cuda_matmul_fp8_block_cutlass.cu:56-58` asserts they refuse by name instead, which the measurement contradicts; `cffe59b02` has since rewritten that dispatch, and no CI lane can see either way because `cutlass-fp8` is ENABLED on the GB10 gate host
Row: BACKEND-CUDA-SM110
State: UNKNOWN
Kind: bug
GitHub: 1725
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:646`

### Frozen archive evidence

> | [#1725](https://github.com/mudler/vllm.cpp/issues/1725) | `BACKEND-CUDA-SM110` | `kMatmulFp8Cutlass` and `kMatmulFp8BlockScaled` fall through to the portable CPU tier and SEGFAULT on sm_110 (`test_ops_fp8_cutlass:191`, `test_ops_matmul_fp8_block_cuda:345`, measured at `0764ded2b`). The sm_110 baseline attributed them to [#960](https://github.com/mudler/vllm.cpp/issues/960), which was CLOSED COMPLETED three days earlier by `d607fec4c` -- that fix covered `QuantFp8Static` only, while these two ops are registered from TUs `CMakeLists.txt:1790-1791` builds solely for `VT_CUTLASS_FP8_ARCHS`. `cuda_matmul_fp8_block_cutlass.cu:56-58` asserts they refuse by name instead, which the measurement contradicts; `cffe59b02` has since rewritten that dispatch, and no CI lane can see either way because `cutlass-fp8` is ENABLED on the GB10 gate host | bug |

## Resolution

-
