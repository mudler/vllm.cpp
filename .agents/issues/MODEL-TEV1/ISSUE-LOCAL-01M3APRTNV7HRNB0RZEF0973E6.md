ID: ISSUE-LOCAL-01M3APRTNV7HRNB0RZEF0973E6
Title: Port togethercomputer/Tev1-4B-experimental autoregressive decision model
Row: MODEL-TEV1
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

Tev1-4B-experimental (togethercomputer/Tev1-4B-experimental) is a 4B autoregressive decision model. It is an SFT of Qwen3.5-4B-Base with a standard next-token LM head (not non-autoregressive). It is trained to choose one option from a structured state, question, and 2-24 labeled options. It uses standard chat completions (temperature=0, max_tokens=8, enable_thinking=false) and returns a single option letter. The Qwen3.5-4B backbone is already implemented in vllm.cpp. The model needs porting with chat template integration, decision prompting, and registration. CPU + GPU (CUDA). Oracle: vLLM (Qwen3.5 at pin).

## Resolution

-
