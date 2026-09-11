ID: ISSUE-GH-824
Title: The spec's record is stale in four places: it puts GGUF expert streaming out of scope because it "needs per-expert slicing of 3D tensors" and must wait for `QUANT-GGUF-KEEPQ-LOADER`, but that slicer landed `429e19d6a` on 2026-07-22, twelve days AFTER the spec was written, and is now the production decode path (`OwnGgufQuantBlocks` row_offset at `qwen3_5_gguf_weights.cpp:57`; `GemmRowSlice` per-expert at `deepseek_v4.cpp:1004,2487` and `laguna.cpp:1225`). The correction inverts a scheduling decision: the GGUF lane needs NO Marlin bank at all (the mmap'd GGUF file already is one), so it is plausibly the CHEAPER lane, and it is the only one that can serve a Qwen3.8-class checkpoint since every such checkpoint that exists is GGUF. FIXED IN FLOW
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 824
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:201`

### Frozen archive evidence

> | [#824](https://github.com/mudler/vllm.cpp/issues/824) | `ENG-EXPERT-STREAM` | The spec's record is stale in four places: it puts GGUF expert streaming out of scope because it "needs per-expert slicing of 3D tensors" and must wait for `QUANT-GGUF-KEEPQ-LOADER`, but that slicer landed `429e19d6a` on 2026-07-22, twelve days AFTER the spec was written, and is now the production decode path (`OwnGgufQuantBlocks` row_offset at `qwen3_5_gguf_weights.cpp:57`; `GemmRowSlice` per-expert at `deepseek_v4.cpp:1004,2487` and `laguna.cpp:1225`). The correction inverts a scheduling decision: the GGUF lane needs NO Marlin bank at all (the mmap'd GGUF file already is one), so it is plausibly the CHEAPER lane, and it is the only one that can serve a Qwen3.8-class checkpoint since every such checkpoint that exists is GGUF. FIXED IN FLOW | bug |

## Resolution

-
