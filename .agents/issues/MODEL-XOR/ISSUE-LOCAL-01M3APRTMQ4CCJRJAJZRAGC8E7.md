ID: ISSUE-LOCAL-01M3APRTMQ4CCJRJAJZRAGC8E7
Title: Port juspay/xor 35B MoE SystemOne-class decision model
Row: MODEL-XOR
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

Xor (juspay/xor) is a 35B MoE SystemOne-class decision model post-trained from Qwen3.6-35B-A3B (35B total, ~3B activated). It supports noul/choice/score question types through the /v1/systemone API with deterministic single-token candidate readout, forward+reverse option-order evaluation, and probability calibration. It is multimodal (up to 8 images) with fully merged BF16 weights. The Qwen3.6-35B-A3B MoE backbone needs implementation. CPU + GPU (CUDA). Oracle: SGLang (validated serving runtime).

## Resolution

-
