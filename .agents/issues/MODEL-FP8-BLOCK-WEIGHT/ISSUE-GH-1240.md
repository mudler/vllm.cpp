ID: ISSUE-GH-1240
Title: `test_qwen27_dense_forward` and `test_qwen27_dense_forward_glue_fuse_off` are RED on `origin/main` at `aba8d5ffb`. Both throw the block-wise-FP8 refusal `09597106e` (#1228, issue #1189) added at `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:428`: `'layer.linear_attn.in_proj_qkv.weight_scale_inv' is present ... but the checkpoint's quantization_config declares no weight_block_size`. ATTRIBUTED by measurement, not inference: `git log -1` on that file is `09597106e`, and `git show --stat 09597106e` changes it (+219) while touching neither test — the producer moved and its consumer did not. It hides behind the doctest thrown-case shape: the run reports 563 assertions with NONE failed beside 9 cases with ONE failed, because a case that THROWS records no failed assertion, so the assertions line reads clean and only `Status: FAILURE!` and the exit code tell the truth. NOT fixed in flow, deliberately: the refusal is correct in its own terms (guessing 128x128 is exactly what #1166 asked not to happen), so whether the synthetic fixture is wrong to emit `weight_scale_inv` without `weight_block_size`, or the refusal is too broad for a checkpoint that is not block-wise FP8 at all, is a semantics choice belonging to the owning row with its own red-first evidence — not a fixture edit that makes the message go away. Found by `ENG-EXPERT-STREAM-DEVICE` W0 running the full ctest suite after merging `origin/main` (523 of 525 passed; these two the only failures, and W0 touches neither file)
Row: MODEL-FP8-BLOCK-WEIGHT
State: UNKNOWN
Kind: bug
GitHub: 1240
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:401`

### Frozen archive evidence

> | [#1240](https://github.com/mudler/vllm.cpp/issues/1240) | `MODEL-FP8-BLOCK-WEIGHT` | `test_qwen27_dense_forward` and `test_qwen27_dense_forward_glue_fuse_off` are RED on `origin/main` at `aba8d5ffb`. Both throw the block-wise-FP8 refusal `09597106e` (#1228, issue #1189) added at `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:428`: `'layer.linear_attn.in_proj_qkv.weight_scale_inv' is present ... but the checkpoint's quantization_config declares no weight_block_size`. ATTRIBUTED by measurement, not inference: `git log -1` on that file is `09597106e`, and `git show --stat 09597106e` changes it (+219) while touching neither test — the producer moved and its consumer did not. It hides behind the doctest thrown-case shape: the run reports 563 assertions with NONE failed beside 9 cases with ONE failed, because a case that THROWS records no failed assertion, so the assertions line reads clean and only `Status: FAILURE!` and the exit code tell the truth. NOT fixed in flow, deliberately: the refusal is correct in its own terms (guessing 128x128 is exactly what #1166 asked not to happen), so whether the synthetic fixture is wrong to emit `weight_scale_inv` without `weight_block_size`, or the refusal is too broad for a checkpoint that is not block-wise FP8 at all, is a semantics choice belonging to the owning row with its own red-first evidence — not a fixture edit that makes the message go away. Found by `ENG-EXPERT-STREAM-DEVICE` W0 running the full ctest suite after merging `origin/main` (523 of 525 passed; these two the only failures, and W0 touches neither file) | bug |

## Resolution

-
