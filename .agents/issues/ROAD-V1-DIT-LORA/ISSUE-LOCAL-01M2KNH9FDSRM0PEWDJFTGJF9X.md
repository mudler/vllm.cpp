ID: ISSUE-LOCAL-01M2KNH9FDSRM0PEWDJFTGJF9X
Title: Shared DiT LoRA load-time fusion seam for LTX2.5 + MiniMax-H3
Row: ROAD-V1-DIT-LORA
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-15
Updated: 2026-09-15
Closed: -

## Problem

LTX2.5 has a complete LoRA load-time-fusion implementation (Ltx2LoraSpec/Ltx2LoraAdapter/Ltx2FuseLoraIntoTensor) that is LTX2-specific only in naming and one key-rewrite function. MiniMax-H3 is the same architecture class (diffusion DiT with host-side weight materialization) but has no LoRA support. LocalAI's vllm-cpp backend has zero LoRA support. Generalize the LTX2 fusion into a shared dit_lora seam, wire H3 to use it, and expose both through LocalAI config-based lora_adapters/lora_scales.

## Resolution

-
