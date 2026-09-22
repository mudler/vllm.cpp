ID: ISSUE-LOCAL-01M3554MJ5K9V3BJ5X84YQAZ70
Title: qwen3 GGUF arch not supported in model_loader dispatch
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-22
Updated: 2026-09-22
Closed: -

## Problem

The GGUF dispatch table in model_loader.cpp supports qwen35, qwen35moe, and qwen3next architectures, but NOT plain qwen3. qwen3_dense.cpp:69 explicitly refuses GGUF loading. This blocks the quant-matched C++-only token gate (llama.cpp Q4_K_M vs vllm.cpp Q4_K_M) needed for BACKEND-GATE-ROCM-SGLANG on Strix, where vLLM and SGLang cannot run due to ROCm 5.7 / torch 2.13.0+rocm7.2 incompatibility.

## Resolution

-
