ID: ISSUE-LOCAL-01M2JXGESJW624DWDHBM4SF8KS
Title: ROCm wvSplitK: add gfx1151 dispatch tuning
Row: BACKEND-ROCM
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-15
Updated: 2026-09-15
Closed: -

## Problem

wvSplitK kernel accounts for 28% of decode time on gfx1151. Fixed YTILE=2, UNRL=2 configuration is suboptimal. vLLM uses gfx1151-specific dispatch logic that selects different configurations based on K, N, and LDS fit.

## Resolution

-
