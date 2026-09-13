ID: ISSUE-LOCAL-01M2C9E4KVFT4G1T617FWGHGVC
Title: `main` is RED on the `tools suites` gate and has been for the whole V4.1 campaign: 853 tests, 49 failures, 1 error on a pristine detached checkout of `origin/main`, verified 2026-09-13 by the operator with `python3 -m unittest discover -s tests/tools -t . -p 'test_*.py'` in a throwaway worktree carrying no local change. Every V4.1 wave's `agent-preflight.sh --staged` therefore reports `1 gate(s) failed: tools suites` no matter what it contains, and three independent implementers each had to prove the failure pre-existing before their evidence could be read. The failure MODE has drifted during the campaign, which is itself worth recording: while the box was at 100% disk the failures presented as `ValueError: disk headroom exhausted` out of `test_strix_vllm_oracle.py`; with 16 GB free they present as the `c8-leg-runner` benchmark harness reporting `NOT ADMISSIBLE -- see fold.reasons` and `3 legs planned, 0 done, 3 owed`. So at least two distinct causes have hidden behind one red, and a reader who dismissed the first would have dismissed the second. The cost is not theoretical: a permanently-red shared gate trains every agent to classify a real failure as environmental, which is the inverse of what a gate is for, and it means no V4.1 wave can present a fully green preflight however correct it is
Row: ENG-RECORD-CLAIM-AGREEMENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

A shared gate is red on main, so no wave can show a green preflight and every agent must first disprove it.

## Resolution

-
