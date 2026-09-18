ID: ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB
Title: DeepSeek-V4.1 W3c/W3d: indexer-K host reference and the kv_source_layer-keyed KV topology
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

DeepSeek-V4.1-Flash has no host reference for its indexer arm or its cross-layer KV topology. Two gaps, one mechanism. (1) W3c: the net-new op indexer_k_norm_rope_store (vllm e77daef89e vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py:29,120) replaces V4's indexer-local DeepseekCompressor: index K is k_norm(wk(latent)) from the MAIN compressor latent, RoPE'd at the group's first position (pos//r)*r, emitted only at group boundaries (pos+1)%r==0, into a 132-byte FP8 or 68-byte MXFP4 paged uint8 row with segregated value and scale regions. This tree has no indexer K cache at all: deepseek_v4_dsa.cpp:485 hardcodes VT_CHECK(compress_ratio == 4) and coff = 2, which V4.1 DELETES. (2) W3d: every compressed layer resolves its compressed-KV and indexer sources as max(s for s in source_layers if s <= layer_id) (attention.py:284-286, index twin :287-289), and the per-layer RoPE selection keys on compress_ratio > 0 where V4 keyed on > 1. Inheriting V4's ratio<=1 == sliding-window-only reading silently kills the whole compressed arm of layers 20-39, because in V4.1 ratio 1 means FULL-LENGTH COMPRESSED.

## Resolution

-
