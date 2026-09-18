ID: ISSUE-LOCAL-01M2CNCNA30WWNBRA5JBC7MSMY
Title: TENSTORRENT: keep-quant decode chain for IQ3_XXS (APEX-I-Nano wave)
Row: QUANT-GGUF-IQ3_XXS
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

The APEX-I-Nano census (tracker ISSUE-LOCAL-01M2CNC22SYAKJN3YBGNVGSG56, row QUANT-GGUF-IQ-TENSTORRENT) carries IQ3_XXS tensors, but the TENSTORRENT keep-quant registered set (MatmulBTQuantKernel) is {Q4_K, Q5_K, Q6_K, Q8_0} — the encoding refuses by name on the P150. Wave: the on-core keep-quant decode chain for IQ3_XXS (vec_dot semantics mirroring vt::cpu::BlockVecDot), route inclusion (KeepQuantDType/DeviceKeepQuantSupported), the W4a wave-2b test pattern (op-level + bit-exact sweep vs vt::cpu::BlockVecDot), and census-driven reach on mudler/Qwen3.8-27B-APEX-I-Nano. Context: the IQ3_XXS CPU side is READY. The tracker owns ordering; the Q4_K_M 27B verdict (W3/W4, in flight) takes the device first.

## Resolution

-
