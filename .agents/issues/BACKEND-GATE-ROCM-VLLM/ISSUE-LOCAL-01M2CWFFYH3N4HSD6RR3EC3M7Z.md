ID: ISSUE-LOCAL-01M2CWFFYH3N4HSD6RR3EC3M7Z
Title: the Strix vLLM oracle suite turns a low-disk box into 49 opaque test failures
Row: BACKEND-GATE-ROCM-VLLM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

On a host with little free disk, tests/tools/test_strix_vllm_oracle.py reports 49 failures and 1 error, and scripts/agent-preflight.sh fails its 'tools suites' gate with them. The cause is not the tree: tools/bench/strix_vllm_oracle/worker.py:257 raises ValueError('disk headroom exhausted') from session.headroom(), the test's built() helper asserts returncode == 0, and every case that builds a session fails with that traceback buried in an assertEqual message. Reproduced 2026-09-13 on a dev box at 97% full (15 GiB free of 447 GiB) against a PRISTINE git archive of origin/main ee0644eab, 49 failures and 1 error, byte-identical to what the same suite reports on row/MODEL-MM-QWEN4-EXP-ROCM-CHUNKED-H2D, whose diff touches no file under tools/ or tests/tools/. So this is an environment condition that an unrelated row's preflight reads as its own red. A headroom guard that cannot run should SKIP with the named reason, the way preflight already reports its five argument-starved gates, rather than fail 49 cases whose messages do not say 'disk'. Found while repairing MODEL-MM-QWEN4-EXP-ROCM-CHUNKED-H2D; not fixed there, because the fix is this row's tooling and needs its own scope.

## Resolution

-
