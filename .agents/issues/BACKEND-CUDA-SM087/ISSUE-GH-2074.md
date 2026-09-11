ID: ISSUE-GH-2074
Title: **`cmake/CudaArchFeatures.cmake`'s `fa2` row still labels its Ampere `sm_8x` cells "NO Ampere board ran them here", and an `sm_87` board ran the FA-2 path on 2026-08-26.** Jetson AGX Orin IS `sm_87`, one of the four cells that label covers. Row `MODEL-MM-dots3-note` ([#699](https://github.com/mudler/vllm.cpp/issues/699)) leased `orin:gpu0`, measured FA-2 ON three ways (`fa2: ENABLED for [87]`, `VLLM_CPP_FLASH_ATTN:BOOL=ON`, `VLLM_CPP_CUDA_FA2_COMPILED_ARCHS "87"`) and ran `test_ops_mla_prefill` to 2,931,678 assertions on the device against 329,772 with `CUDA_VISIBLE_DEVICES=""`, with the window biting at `gpu_win` vs `gpu_none` = 1.06055. Evidence `.agents/specs/dots3-note.md` §4.8; that record is CITED, not re-measured here. Scope: `8.7` only, so the label is now wrong for one of its four cells and right for `8.0`, `8.6` and `8.9`, which one line cannot carry. The identical claim also sits in `.agents/specs/cuda-arch-ampere-fastpath.md` WA-1 and the `BACKEND-CUDA-SM087` backend-matrix cell. Found by W4b-3a of #699 and filed rather than fixed, because re-labelling another row's verification state is that row's decision
Row: BACKEND-CUDA-SM087
State: UNKNOWN
Kind: record
GitHub: 2074
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:800`

### Frozen archive evidence

> | [#2074](https://github.com/mudler/vllm.cpp/issues/2074) | `BACKEND-CUDA-SM087` | **`cmake/CudaArchFeatures.cmake`'s `fa2` row still labels its Ampere `sm_8x` cells "NO Ampere board ran them here", and an `sm_87` board ran the FA-2 path on 2026-08-26.** Jetson AGX Orin IS `sm_87`, one of the four cells that label covers. Row `MODEL-MM-dots3-note` ([#699](https://github.com/mudler/vllm.cpp/issues/699)) leased `orin:gpu0`, measured FA-2 ON three ways (`fa2: ENABLED for [87]`, `VLLM_CPP_FLASH_ATTN:BOOL=ON`, `VLLM_CPP_CUDA_FA2_COMPILED_ARCHS "87"`) and ran `test_ops_mla_prefill` to 2,931,678 assertions on the device against 329,772 with `CUDA_VISIBLE_DEVICES=""`, with the window biting at `gpu_win` vs `gpu_none` = 1.06055. Evidence `.agents/specs/dots3-note.md` §4.8; that record is CITED, not re-measured here. Scope: `8.7` only, so the label is now wrong for one of its four cells and right for `8.0`, `8.6` and `8.9`, which one line cannot carry. The identical claim also sits in `.agents/specs/cuda-arch-ampere-fastpath.md` WA-1 and the `BACKEND-CUDA-SM087` backend-matrix cell. Found by W4b-3a of #699 and filed rather than fixed, because re-labelling another row's verification state is that row's decision | record |

## Resolution

-
