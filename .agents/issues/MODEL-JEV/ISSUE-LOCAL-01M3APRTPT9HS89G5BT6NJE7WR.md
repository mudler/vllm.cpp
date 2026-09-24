ID: ISSUE-LOCAL-01M3APRTPT9HS89G5BT6NJE7WR
Title: Port Jev-like structured generation for DiffusionGemma (vLLM PR #57250)
Row: MODEL-JEV
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

vLLM PR #57250 (merged) adds a structured generation mode for DiffusionGemma enabling Jev-like bounded-choice answers. Features include canvas seeding (diffusion_seed_canvas), read-only requests (diffusion_read_only), pinned positions, and logprobs on the converging step. The model can answer noul (yes/no), scale, and multiple-choice questions with certainty and error bars. DiffusionGemma is already inventoried in vllm.cpp; this adds the decision capability. The vLLM pin must be advanced past PR #57250. CPU + GPU (CUDA). Oracle: vLLM (PR #57250 merged to main).

## Resolution

-
