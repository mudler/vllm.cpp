ID: ISSUE-LOCAL-01M29SAM8CKS2XQM8EPEEN9Z67
Title: DeepSeek-V4 vision W6 harness: three unproven weaknesses recorded rather than repaired
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

Three items a fresh review flagged as PLAUSIBLE without mutating them, recorded here so they are not lost. (1) tools/parity/dsv4v_w6_compare.py lines 192-199 silently re-lays-out the oracle's vit dump when the shapes disagree and a numeric coincidence holds (isqrt(rows)^2 == rows and b[1] == grid and b[0] == a[1]*grid), applying a GUESSED permutation that the emitted report does not record; a reader of report-<tag>.json cannot tell whether the vit numbers came from the file as written or from a re-indexing this code invented. (2) tools/parity/dsv4v_w6_probe.cpp hardcodes the normalisation ((raw/255) - 0.5) / 0.5 in its f32 arm instead of calling DeepSeekV4ImageProcessor::ProcessImage, so the f32 arm would keep agreeing with the oracle even if the SHIPPED processor's mean/std diverged from those constants -- the arm that is supposed to test the function shares no code with the function on that step. (3) dsv4v_w6_compare.py's best_match does not assert that the reference rows are pairwise distinct, so on a flat or low-detail image the argmax would be arbitrary and the identity-permutation result would be meaningless rather than wrong. None of the three was mutated or measured by this repair wave. Found by fresh review 2026-09-12.

## Resolution

-
