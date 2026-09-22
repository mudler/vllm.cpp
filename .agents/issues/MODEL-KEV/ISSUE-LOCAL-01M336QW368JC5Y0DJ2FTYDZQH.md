ID: ISSUE-LOCAL-01M336QW368JC5Y0DJ2FTYDZQH
Title: Port kev-0.8b decision model (Qwen3.5 + LoRA + PointerHead) to vllm.cpp
Row: MODEL-KEV
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-22
Updated: 2026-09-22
Closed: -

## Problem

kev-0.8b (jaredpalmer/kev-0.8b) is a System 1 decision model: frozen Qwen3.5-0.8B-Base + rank-16 LoRA + PointerHead. It takes a state document + typed questions (choice/score/noul) and returns probability distributions. vllm.cpp has the Qwen3.5 backbone but lacks ForwardHidden, load-time LoRA merge, PointerHead, and /v1/systemone dispatch for kev. The /v1/systemone API exists from GLiNER2.5; the DecisionFn callback from Laya (PR #3263) merges first.

## Resolution

-
