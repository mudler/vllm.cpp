ID: ISSUE-LOCAL-01M2DXWBV1ZZ56GAY2704WB2Y5
Title: The QSA gather's ascending softmax fold is enforced by NOTHING: two reassociations pass all three suites
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

MEASURED by a fresh review on `thor:gpu0` (`rc` job `95053659-0f2d-47f1-8f6a-b6af25e2173b`) against W9 at `7d0d74c2c`. Two reassociations of `QsaGatherAttentionKernel`'s softmax denominator -- a DESCENDING fold, and the WARP-SHUFFLE TREE the kernel header explicitly declines -- each pass ALL THREE committed suites unmodified: test_qwen4_exp_cuda_reductions 19/19, test_qwen4_exp_qsa_device 12/12, test_qwen4_exp_qsa 14/14. Both demonstrably change the output: the suite's own not-bitwise-equal counter moves on every fixture (grid stride 53,379 -> 95,488 of 166,400 floats). The ascending fold is the property W9's spec calls the whole argument for choosing its lever, and nothing measures it. TIGHTENING THE BOUND IS NOT THE REPAIR, and that is measured: at the multi-tile shapes the reassociated result sits at 0.01-0.02% of the derived bound (|diff| 5.2e-08 against 9.8e-04 at |sel| = 2050), and the worst ratio on a CORRECT kernel (0.2041, over_budget at |sel| = 2) is HIGHER than on a reassociated one (0.1136, grid stride), so no single threshold separates them. test_qwen4_exp_cuda_reductions.cpp:37-46 already records a fitted kUlpTol = 1.20e-7 that failed correct kernels by 112-290%; re-creating that is a stop condition, not a repair.

## Resolution

-
