ID: ISSUE-LOCAL-01M2RECPA70PBQN14E1VCMHJ5T
Title: Runtime prompt-activated LoRA for diffusion DiT models
Row: ROAD-V1-LORA-RUNTIME
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-17
Updated: 2026-09-17
Closed: -

## Problem

Diffusion DiT models (MiniMax-H3, LTX2.5) support load-time LoRA fusion (ROAD-V1-DIT-LORA, merged) but cannot select LoRAs per-request. Users need prompt-tag activation (<lora:name:strength>) mirroring LocalAI stable-diffusion.cpp backend, so different LoRAs apply to different requests without reloading the model. vLLM-Omni DiffusionLoRAManager implements this as single-active-adapter per-layer additive delta applied at forward time, bypassing punica. The shared seam on main is dit_lora.h/cpp (Dit-prefixed symbols).

## Resolution

-
