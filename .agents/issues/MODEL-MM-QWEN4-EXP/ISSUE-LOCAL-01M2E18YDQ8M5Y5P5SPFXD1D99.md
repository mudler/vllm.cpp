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

### THE OPEN QUESTION IS ANSWERED, AND THE ANSWER IS "IT PREDATES W9" (operator, 2026-09-13)

This issue asked, correctly, whether the operator's green four-suite reading
(`rc` job `d80f84a7`) used a different architecture, build type or toolkit, and
said to resolve that before scoping a fix. Both halves are now settled.

**1. There was never a contradiction. `d80f84a7` DID NOT RUN THIS SUITE.** Its
`TARGETS` line is `test_qwen4_exp_cuda_reductions test_qwen4_exp_qsa
test_qwen4_exp_qsa_device test_qwen4_exp_cuda` -- four suites, and
`test_qwen4_exp_qsa_block` is not among them; the string does not appear anywhere
in that job's log. So the green and the red are not two readings of one thing.
**That was a hole in the operator's gate**, not a conflict, and it is closed: the
gate now carries `test_qwen4_exp_qsa_block` as a fifth target.

**2. THE RED PREDATES W9, measured on the arm nobody had run.** Both arms this
issue compared (`7d0d74c2c` and `a6ad48295`) CONTAIN W9, so they cannot separate
a regression from standing debt. `rc` job `a9de65c6` on `thor:gpu0` ran W9's
PARENT -- `6a47d4370`, which is `main` without W9 -- against `7d0d74c2c`, one
clone, one toolkit, one build type, only the checkout differing:

| arm | HEAD | binary md5 | reps | result |
|---|---|---|---|---|
| base, NO W9 | `6a47d4370` | `426aa066dd5d` | 3 | rc=1, 13 selected, **12 passed** |
| fix, W9 | `7d0d74c2c` | `0b00abde081e` | 3 | rc=1, 13 selected, **12 passed** |

Identical, 3 of 3 reps each, with DIFFERENT binary md5s proving the two arms are
genuinely different builds rather than one binary run twice. The operator's
five-suite gate on the final landed head (`af417e4ee`) reproduces it a third
time: 13 cases, 12 passed, 7439 assertions, 1 failed.

**W9 did not cause this and W9 merged on that basis** (`054a8910c`). What remains
open is the original question the title asks: whether the guard is a near-tie
whose margin this fixture no longer clears, or a real device-arm defect on the
indexer-logits path. Nothing here touches that, and the ANCHOR for it is
unchanged: the assertion is on `vt::DsaIndexerLogits` + `vt::DsaTopkSelect`, and
the gather is not on that path.

**Owed:** the near-tie question, on `dgx:gpu0` as well as `thor:gpu0`, since a
margin that depends on the architecture is exactly what an sm_110-only reading
cannot distinguish from a defect. See
[[a-discrete-selection-gate-has-bimodal-error-not-a-tolerance]] in spirit: a
selection guard's error flips or it does not, so "close to the margin" is the
thing to measure, not the verdict.
