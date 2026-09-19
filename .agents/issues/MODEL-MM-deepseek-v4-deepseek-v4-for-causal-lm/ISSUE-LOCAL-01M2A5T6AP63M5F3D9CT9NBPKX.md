ID: ISSUE-LOCAL-01M2A5T6AP63M5F3D9CT9NBPKX
Title: W7-CUDA driver and its mutation suite cite spec line anchors that have moved
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

tools/parity/dsv4v_w7_cuda.sh:334 cites .agents/specs/deepseek-v4-flash-vision.md:1370 for the recorded '24 of 27 passed' ctest result; the sentence is really at :1423. The same file's :335 cites :1377-1383 for the dev_attn refusal, which is really at :779 and :1432. tests/scripts/test_dsv4v_w6_compare.py:472 and :482 repeat both stale anchors in the CTEST_RECORDED and DEV_ATTN_RECORDED comments. Nothing is mis-JUDGED by this -- the classification reads steps and logs, not the spec -- but a reader sent to the wrong line cannot check the attribution the classifier rests on, and these anchors drift every time the spec grows. Found by fresh review 2026-09-12 and filed rather than fixed, because the repair wave was scoped to the fail-open and the false-red routes.

## Resolution

-
