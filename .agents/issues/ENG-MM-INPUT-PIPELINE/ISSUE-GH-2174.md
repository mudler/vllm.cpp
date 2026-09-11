ID: ISSUE-GH-2174
Title: **`MakeDevBf16` in `gemma4_vision.cpp` allocated from the tensor's declared SHAPE and copied the host store's OWN length into it, with nothing checking that the two agree.** `bytes = numel * SizeOf(kBF16)` sizes the allocation while `b.Copy(q, d.p, bf.data(), bf.size() * sizeof(uint16_t))` sets the copy length, so `bf.size() * 2 > bytes` overruns the allocation and a loader bug — a wrong enumeration, a mis-shaped weight, a checkpoint whose config disagrees with its tensors — lands as heap corruption rather than as a named refusal; the under-full case leaves an uninitialised tail. Its line-for-line twin `qwen3_vl_vision.cpp:137` grew exactly this guard in #1359 and copies `bytes`, so the asymmetry is the defect and the Gemma-4 copy predates #1359 rather than being made worse by it. Not currently exploitable for a reason that is itself debt: nothing in production calls `Gemma4VisionForward` (#2173), so the only shapes this function sees are two tests' fixtures, which agree by construction. Found in the fresh review of #2169 and FIXED IN FLOW there by mirroring the twin's `VT_CHECK` and copying `bytes`; behaviour is unchanged on every shape the loaders produce
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 2174
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:845`

### Frozen archive evidence

> | [#2174](https://github.com/mudler/vllm.cpp/issues/2174) | `ENG-MM-INPUT-PIPELINE` | **`MakeDevBf16` in `gemma4_vision.cpp` allocated from the tensor's declared SHAPE and copied the host store's OWN length into it, with nothing checking that the two agree.** `bytes = numel * SizeOf(kBF16)` sizes the allocation while `b.Copy(q, d.p, bf.data(), bf.size() * sizeof(uint16_t))` sets the copy length, so `bf.size() * 2 > bytes` overruns the allocation and a loader bug — a wrong enumeration, a mis-shaped weight, a checkpoint whose config disagrees with its tensors — lands as heap corruption rather than as a named refusal; the under-full case leaves an uninitialised tail. Its line-for-line twin `qwen3_vl_vision.cpp:137` grew exactly this guard in #1359 and copies `bytes`, so the asymmetry is the defect and the Gemma-4 copy predates #1359 rather than being made worse by it. Not currently exploitable for a reason that is itself debt: nothing in production calls `Gemma4VisionForward` (#2173), so the only shapes this function sees are two tests' fixtures, which agree by construction. Found in the fresh review of #2169 and FIXED IN FLOW there by mirroring the twin's `VT_CHECK` and copying `bytes`; behaviour is unchanged on every shape the loaders produce | bug |

## Resolution

-
