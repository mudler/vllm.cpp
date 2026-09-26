ID: ISSUE-LOCAL-01M3EQ87MWSPPCW51HFVDWSQHC
Title: Port: Gemma4 AutoWeightsLoader migration (upstream 44dd18fe0b)
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

Upstream commit 44dd18fe0b refactors gemma4.py weight loading to use AutoWeightsLoader (475 lines changed). The C++ port loads weights via its own safetensors reader; the loading semantics should be reviewed for parity. Classification: INVENTORY (Python-side refactor, C++ has its own loader). Sync cycle a7c23ac96d.

## Resolution

-
