ID: ISSUE-LOCAL-01M3EQ87A8KBKRAABQSWX75GJG
Title: Port: Gemma4 KV-shared layers k_norm/v_norm conditional creation (upstream f2d45f26bd)
Row: ENG-SCHED-CORE
State: OPEN
Kind: tech-debt
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

Upstream commit f2d45f26bd conditionally creates KV projections/norms on KV-shared layers in gemma4.py. The C++ port already guards with is_kv_shared but should be reviewed for parity on the k_norm/v_norm conditional path. Classification: PORT-NOW (reviewed, deferred — C++ uses merged qkv_proj with is_kv_shared guard, no k_norm/v_norm equivalent). Sync cycle a7c23ac96d.

## Resolution

-
