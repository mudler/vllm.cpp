ID: ISSUE-LOCAL-01M36V2V99K4GNJ926QBPBKDPT
Title: qwen3 GGUF loader: wrong tensor name post_attention_norm should be ffn_norm
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

The qwen3_gguf_weights.cpp loader expects blk.N.post_attention_norm.weight for the post-attention layernorm, but the standard Qwen3 GGUF naming convention (used by llama.cpp convert_hf_to_gguf.py and the Qwen/Qwen3-4B-Q4_K_M.gguf HuggingFace checkpoint) uses blk.N.ffn_norm.weight. This causes model load failure: "gguf: no tensor named blk.0.post_attention_norm.weight". The test file also uses the wrong name.

## Resolution

Fixed in 023e55944: changed post_attention_norm.weight to ffn_norm.weight in qwen3_gguf_weights.cpp and test. All 5 regression tests pass (117 assertions).
