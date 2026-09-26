ID: ISSUE-LOCAL-01M3EQD91494MZG0DE9XN4ZKV6
Title: Port: null RoPE parameters fix (upstream 18615ad1ee)
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

Upstream commit 18615ad1ee fixes null RoPE parameters in ModelConfig._verify_rope_params. No C++ equivalent exists yet. Classification: INVENTORY (C++ has no rope param verification). Sync cycle a7c23ac96d.

## Resolution

-
