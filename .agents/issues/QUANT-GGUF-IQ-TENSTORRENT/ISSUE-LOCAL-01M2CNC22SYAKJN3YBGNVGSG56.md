ID: ISSUE-LOCAL-01M2CNC22SYAKJN3YBGNVGSG56
Title: TENSTORRENT: serve the APEX-I-Nano IQ-quant family (IQ3_XXS/IQ2_S/IQ2_XXS + Q3_K keep-quant decode)
Row: QUANT-GGUF-IQ-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Ettore's mudler/Qwen3.8-27B-APEX-I-Nano.gguf (10.7 GB) packs the 27B into the P150's 32 GiB with real headroom, but its tensor census (llama-gguf, /tmp/apex-gguf-dump.txt, 2026-09-13) is IQ-dominant: 164 IQ3_XXS + 89 IQ2_S + 44 IQ2_XXS + 78 Q3_K tensors, against only 122 Q4_K + 8 Q8_0 + 1 Q6_K. The TENSTORRENT keep-quant registered set (MatmulBTQuantKernel, tenstorrent_ops.cpp) is exactly {Q4_K, Q5_K, Q6_K, Q8_0}, so the artifact refuses by name on the P150 today. This row tracks the per-encoding device waves (each encoding needs an on-core keep-quant decode chain — the vec_dot semantics — plus route inclusion and the W4a wave-2b test pattern); the CPU-side states live on the per-encoding rows (IQ3_XXS READY, IQ2_XXS ACTIVE, IQ2_S INVENTORIED — its CPU dot is a prerequisite, Q3_K PARTIAL). The e2e gate is the APEX file generating on the P150 against the pinned oracle, with the artifact documented in docs/USAGE.md per the weights rule. Ordering note: the Q4_K_M 27B verdict (W3/W4, in flight) takes the device first — this row starts after that lands or in parallel on the CPU-side prerequisites.

## Resolution

-
