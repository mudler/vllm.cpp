ID: ISSUE-LOCAL-01M299JRG1PC7WR02RS4AZMZBQ
Title: the pinned attention-rung population omits the W2 vision tower
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: 2026-09-11

## Problem

tests/scripts/test_check_attention_rung_consistency.py::PopulationTests::test_widening_the_population_moves_no_present_verdict pins eight model paths by name. The W2 vision tower src/vllm/model_executor/models/deepseek_v4_vision.cpp, added by 8c10cad16, carries a marked vt::Attention call, so scan_models() now discovers nine and the case reds. The red is the review the case was written to force: the new site must be confirmed correctly marked before the path is added to the pinned list. The row gate never ran this suite, which is why the red sat unnoticed.

## Resolution

2026-09-11: verified FIRST that the new site is correctly marked, then widened the pin. scripts/check-attention-rung-consistency.py reads the marker on lines 683-685 of src/vllm/model_executor/models/deepseek_v4_vision.cpp and reports the call at line 686 as (686, True); deleting only those three marker lines in a scratch mutation turns the checker red at that exact call (rc=1, 'no recorded reason'), and restoring left the file byte-identical by sha256 5580cf1ec6d6113d5c1a27cc8babb859e0d10ba86b1494211492c93e8173511a. The second VT-ATTN-NAIVE comment at line 637 sits 49 lines above the call, outside MARKER_WINDOW_LINES=20, so it contributes nothing and the pass rests solely on the marker the mutation moved. The path was then added to the pinned list in sorted position. RED before: 45 tests, 1 failure on the population case. GREEN after: 45 tests, OK. Checker: 'OK (attention rung): 9 vt::Attention call site(s) in 9 source file(s) under 3 scanned root(s); 9 carry a recorded reason, 0 unmarked and excused by 0 allowlisted in-flight stem(s).'
