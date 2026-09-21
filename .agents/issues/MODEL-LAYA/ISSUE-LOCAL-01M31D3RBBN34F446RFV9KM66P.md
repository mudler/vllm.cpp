ID: ISSUE-LOCAL-01M31D3RBBN34F446RFV9KM66P
Title: Port Laya (convaiinnovations/laya) — ModernBERT-large + decision head, SystemOne API
Row: MODEL-LAYA
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

Laya is a System 1 decision model (choice/score/noul questions) using a ModernBERT-large backbone. Neither ModernBERT nor Laya exist in vllm.cpp. The model reuses the existing /v1/systemone API (same question types as GLiNER2.5). Need to port the ModernBERT encoder (dual RoPE, sliding window, GeGLU MLP), the Laya decision head (type_emb, transformer head, scorer, act_head), sequence construction, and wire into the SystemOne server.

## Resolution

-
