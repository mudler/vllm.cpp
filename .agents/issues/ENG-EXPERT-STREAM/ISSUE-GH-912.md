ID: ISSUE-GH-912
Title: Stream routed experts from NVMe so a model larger than device memory runs. Target `Qwen/Qwen3.8-2.4T-A95B`, REGISTERED against `Qwen3_5MoeForCausalLM` and blocked only on capacity: 370 GiB at UD-Q1_0 against 128 GB of unified memory on GB10. The only one of the three offload rows that helps on a unified-memory host, because `ENG-WEIGHT-OFFLOAD` and `ENG-HYBRID-PLACEMENT` both move bytes inside one physical pool. Cheaper than the spec assumed: on the GGUF path the mmap'd file already IS the bank and the per-expert slicer landed 2026-07-22 ([#824](https://github.com/mudler/vllm.cpp/issues/824))
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: feature
GitHub: 912
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:242`

### Frozen archive evidence

> | [#912](https://github.com/mudler/vllm.cpp/issues/912) | `ENG-EXPERT-STREAM` | Stream routed experts from NVMe so a model larger than device memory runs. Target `Qwen/Qwen3.8-2.4T-A95B`, REGISTERED against `Qwen3_5MoeForCausalLM` and blocked only on capacity: 370 GiB at UD-Q1_0 against 128 GB of unified memory on GB10. The only one of the three offload rows that helps on a unified-memory host, because `ENG-WEIGHT-OFFLOAD` and `ENG-HYBRID-PLACEMENT` both move bytes inside one physical pool. Cheaper than the spec assumed: on the GGUF path the mmap'd file already IS the bank and the per-expert slicer landed 2026-07-22 ([#824](https://github.com/mudler/vllm.cpp/issues/824)) | feature |

## Resolution

-
