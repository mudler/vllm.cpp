ID: ISSUE-LOCAL-01M2E18YDQ8M5Y5P5SPFXD1D99
Title: test_qwen4_exp_qsa_block's CUDA-vs-CPU selection-margin case is RED on thor:gpu0 at W9's own head
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

MEASURED on `thor:gpu0` (sm_110), CUDA 13.0.88, Release, -DVLLM_CPP_CUDA_ARCHITECTURES=110, `rc` jobs `686350d8` and `b58e2aee`. The case "qwen4_exp qsa block: a CUDA queue enters the block, and its selection is the CPU's" fails its last value assertion, `CHECK(logit_diff < worst_margin / 2.0)` at test_qwen4_exp_qsa_block.cpp:1737, with INFO "the CUDA logits must stay far enough from the CPU's that no selection can flip". The suite reports 13 cases, 12 passed, 1 failed; 7439 assertions, 1 failed. NOT CAUSED BY THE W9 REPAIR, and that is measured rather than argued: ONE job (`b58e2aee`), ONE clone, ONE build directory, two checkouts, built and run the same way. BASE 7d0d74c2c (W9's own head, binary md5 14137d19250ed76b941db7de9b05e7b8) fails 3 of 3 reps and the full suite; FIX a6ad48295 (the W9 repair, binary md5 fe65429ca506246c9bddc92b06d64b23) fails 3 of 3 reps and the full suite, identically. Every filtered run asserted selected_cases=1, so no run measured nothing. It is deterministic on this worker rather than flaky: 6 of 6 reps red across the two arms. The assertion is on the INDEXER logits (vt::DsaIndexerLogits + vt::DsaTopkSelect) and the gather is not on that path. UNKNOWN and not assumed: whether the operator's green reading of the same four suites (`rc` job d80f84a7) used a different architecture, build type or toolkit, and whether the guard is a near-tie whose margin this fixture no longer clears. Resolve that before scoping a fix, because the difference between an environment-dependent margin and a real device-arm defect is the whole question.

## Resolution

-
