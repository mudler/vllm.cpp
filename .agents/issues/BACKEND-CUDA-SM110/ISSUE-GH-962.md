ID: ISSUE-GH-962
Title: NVFP4 marlin disagrees with itself on sm_110: `test_ops_moe_grouped:1262` fails `CHECK(bitdiff == 0)` on `NVFP4 block8-vs-block16 M=8 K=4096 N=4096 bitdiff=15/32768`, against a kernel configure reports as `marlin-nvfp4: ENABLED for [110]`. A live kernel defect on an ENABLED feature, not an absent one, and the only substantive standing sm_110 finding in the [#955](https://github.com/mudler/vllm.cpp/issues/955) baseline. Indexed late, same cause as the row above
Row: BACKEND-CUDA-SM110
State: UNKNOWN
Kind: bug
GitHub: 962
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:643`

### Frozen archive evidence

> | [#962](https://github.com/mudler/vllm.cpp/issues/962) | `BACKEND-CUDA-SM110` | NVFP4 marlin disagrees with itself on sm_110: `test_ops_moe_grouped:1262` fails `CHECK(bitdiff == 0)` on `NVFP4 block8-vs-block16 M=8 K=4096 N=4096 bitdiff=15/32768`, against a kernel configure reports as `marlin-nvfp4: ENABLED for [110]`. A live kernel defect on an ENABLED feature, not an absent one, and the only substantive standing sm_110 finding in the [#955](https://github.com/mudler/vllm.cpp/issues/955) baseline. Indexed late, same cause as the row above | bug |

## Resolution

-
