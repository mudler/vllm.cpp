ID: ISSUE-LOCAL-01M3EQD8PJ3HBNB7AQT6YMT38W
Title: Port: YaRN alignment fix (upstream c191787a68)
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

Upstream commit c191787a68 fixes vLLM YaRN alignment in ModelConfig._get_and_verify_max_len. No C++ equivalent exists yet. Classification: INVENTORY (C++ has no YaRN max_len verification). Sync cycle a7c23ac96d.

## Resolution

-
