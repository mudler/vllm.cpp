ID: ISSUE-LOCAL-01M370AT7VPD1X5RAHNMKGVDJ6
Title: qwen3 GGUF: rotary_dim defaults to 0 when rope.dimension_count absent, crashing RoPE
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

The Qwen3-4B Q4_K_M GGUF file from Qwen/Qwen3-4B-GGUF does not contain a qwen3.rope.dimension_count metadata key. The qwen3_gguf_weights.cpp loader defaults rotary_dim to 0 via OptInt(gguf, p + 'rope.dimension_count', 0) at line 436. This causes the RoPE assertion at ops.cpp:1806 ('rotary_dim must be even and <= head_dim') to fail on every prompt. The correct default is head_dim (128) since Qwen3 uses full rotary (no partial_rotary_factor in config.json). llama.cpp's own converter omits rope.dimension_count when it equals head_dim.

## Resolution

Fixed in #3284 (merged as 363bfbe92). Changed default from 0 to c.head_dim in qwen3_gguf_weights.cpp:436. Added test for absent rope.dimension_count key. Mutation-verified: reverting to 0 fails the new test with CHECK(0 == 32).
