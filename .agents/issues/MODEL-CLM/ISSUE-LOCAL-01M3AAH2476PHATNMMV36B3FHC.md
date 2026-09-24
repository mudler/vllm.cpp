ID: ISSUE-LOCAL-01M3AAH2476PHATNMMV36B3FHC
Title: Port Contrastive-LM/CLM-v0.1-8B System 1 decision model
Row: MODEL-CLM
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

CLM-v0.1-8B (Contrastive-LM/CLM-v0.1-8B) is an 8B System One decision model built on a frozen Qwen3-8B encoder with two projection heads. It uses the system_one API (Choice/Noul/Score), the same pattern as kev and laya. vLLM has no CLM implementation. The model needs porting to vllm.cpp with registration, /v1/systemone server dispatch, and C ABI integration. CPU + GPU (CUDA).

## Resolution

-
