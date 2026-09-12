ID: ISSUE-LOCAL-01M29KFC958AT75GWR7JKXA54K
Title: test_serve_deepseek_v4_mm times out at 1800 s with no output on a CUDA build, cause unknown
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

On thor:gpu0 during W7-CUDA the test produced NO output before CTest killed it at the 1800 s limit, on both the red run and the green run. Nothing measured attributes the hang, and no cause is guessed here: it is not asserted to be related to the vision path, to the staging change, or to the served-image MatVec blocker. What is known is only the observation and that it reproduces across two runs at different heads. Owed: run it under a lease with per-stage output, so the hang is located before any hypothesis is written down.

## Resolution

-
