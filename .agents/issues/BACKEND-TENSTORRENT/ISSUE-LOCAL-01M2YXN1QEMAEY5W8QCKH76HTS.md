ID: ISSUE-LOCAL-01M2YXN1QEMAEY5W8QCKH76HTS
Title: TT: native BFP4/BFP8 weight residency for the 27B-class decode
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-20
Updated: 2026-09-20
Closed: -

## Problem

The f32-exact decode stack on the Tenstorrent P150 reaches ~0.028 tok/s on Qwen3.8-27B-APEX-I-Nano (B2, token-exact vs the llama.cpp b10451 greedy oracle), while Tenstorrent's native tt-metal pipeline reports ~50 tok/s on a single P150A with BFP4 weights / BFP8 KV / BF16 deltaNet state — a ~1800x decode gap. The root difference is FORMAT, not tuning: our path block-dequants GGUF weights and runs every decode GEMM through the SFPU f32-exact path (ttnn matmul truncates f32 operands to tf32 on Blackhole, measured; ComputeConfig does not lift it), while the native pipeline runs tensor-core BFP matmul where BFP precision IS the hardware's precision contract. This row commits the native BFP4/BFP8 weight-residency path: load-time conversion and device-resident ttnn matmul, with a near-tie correctness gate against the b10451 oracle on the quantized model.

## Resolution

-
