ID: ISSUE-GH-907
Title: Five tests are red on dgx.casa (GB10, sm_121a) at `main`, proven PRE-EXISTING by a control build at `10b8bbdaa`: `test_capi` (SIGSEGV in an ABI v8 custom-logits-processor case, plausibly [#547](https://github.com/mudler/vllm.cpp/issues/547) or [#844](https://github.com/mudler/vllm.cpp/issues/844)), `test_cuda_ops` 439/440, `test_linear_method` 83/85, `test_ops_gdn` 4899/4900 ([#614](https://github.com/mudler/vllm.cpp/issues/614)), `test_qwen3_5_gdn_spec_routing` 119/123. Three of the five had no issue at all, which is why this exists
Row: BACKEND-CUDA-COMP-CORE
State: UNKNOWN
Kind: bug
GitHub: 907
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:241`

### Frozen archive evidence

> | [#907](https://github.com/mudler/vllm.cpp/issues/907) | `BACKEND-CUDA-COMP-CORE` | Five tests are red on dgx.casa (GB10, sm_121a) at `main`, proven PRE-EXISTING by a control build at `10b8bbdaa`: `test_capi` (SIGSEGV in an ABI v8 custom-logits-processor case, plausibly [#547](https://github.com/mudler/vllm.cpp/issues/547) or [#844](https://github.com/mudler/vllm.cpp/issues/844)), `test_cuda_ops` 439/440, `test_linear_method` 83/85, `test_ops_gdn` 4899/4900 ([#614](https://github.com/mudler/vllm.cpp/issues/614)), `test_qwen3_5_gdn_spec_routing` 119/123. Three of the five had no issue at all, which is why this exists | bug |

## Resolution

-
