ID: ISSUE-LOCAL-01M2DXWBV1ZZ56GAY2704WB2Y5
Title: The QSA gather's ascending softmax fold is enforced by NOTHING: two reassociations pass all three suites
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

MEASURED by a fresh review on `thor:gpu0` (`rc` job `95053659-0f2d-47f1-8f6a-b6af25e2173b`) against W9 at `7d0d74c2c`. Two reassociations of `QsaGatherAttentionKernel`'s softmax denominator -- a DESCENDING fold, and the WARP-SHUFFLE TREE the kernel header explicitly declines -- each pass ALL THREE committed suites unmodified: test_qwen4_exp_cuda_reductions 19/19, test_qwen4_exp_qsa_device 12/12, test_qwen4_exp_qsa 14/14. Both demonstrably change the output: the suite's own not-bitwise-equal counter moves on every fixture (grid stride 53,379 -> 95,488 of 166,400 floats). The ascending fold is the property W9's spec calls the whole argument for choosing its lever, and nothing measures it. TIGHTENING THE BOUND IS NOT THE REPAIR, and that is measured: at the multi-tile shapes the reassociated result sits at 0.01-0.02% of the derived bound (|diff| 5.2e-08 against 9.8e-04 at |sel| = 2050), and the worst ratio on a CORRECT kernel (0.2041, over_budget at |sel| = 2) is HIGHER than on a reassociated one (0.1136, grid stride), so no single threshold separates them. test_qwen4_exp_cuda_reductions.cpp:37-46 already records a fitted kUlpTol = 1.20e-7 that failed correct kernels by 112-290%; re-creating that is a stop condition, not a repair.

## Resolution

### GATED 2026-09-13 on `thor:gpu0` (sm_110), `rc` job `686350d8`

The repair is `Qwen4ExpQsaAttnArgs::softmax_probe_weights` /
`softmax_probe_denom` / `softmax_probe_stride`, an optional instrument on the
`keys_visited` precedent, and the case
`vt::Qwen4ExpQsaGatherAttention W9: the DENOMINATOR is the ASCENDING fold,
BITWISE` in `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp`. The test
refolds the arm's OWN weights ascending on the host and requires BIT equality
with the arm's own denominator, for the CUDA arm and the CPU arm, over three
multi-tile shapes (1200 rows / 38 tiles, 2050 rows / 65 tiles, 66 rows with a
2-row ragged tail).

Built at `a6ad48295` from a clean clone, CUDA 13.0.88, `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110`, 41 `.cu.o`. Every filtered run asserts its
SELECTED CASE COUNT, because a doctest filter that matches nothing exits 0.

| arm | binary md5 | fold gate | all 9 gather cases |
|---|---|---|---|
| unmutated | `88754e4de2969c0ae8dbdba79b49697d` | **PASS**, 1 case selected | PASS, 9 selected |
| M1 descending fold | `a893bc76ef886df522eb45f5bd67510a` | **RED** | RED |
| M2 warp-shuffle tree | `d8b196cb07d23bb7fa200800274db8a7` | **RED** | RED |
| restored | `457e729e3280ada143d1240782d232a9` | PASS, 1 selected | PASS, 9 selected |

Both reassociations the review ran are DETECTED, each from a binary proved
changed. M1 replaces the tile fold with `u = n-1 .. 0`; M2 replaces it with the
warp-shuffle tree the kernel header declines. The restored binary's md5 differs
from the fix's while its source is byte-identical (`git checkout --`), so the
link is not reproducible on this worker; the source sha256 and the green result
are what the restore establishes.

**The case validates its own fixture.** It computes both reassociations on the
host and requires them to differ from the ascending fold, so a fixture that
could not see a reassociation fails instead of passing. Measured on the
unmutated tree: a descending fold would move 4 of 4, 2 of 2 and 9 of 12 pairs,
a tree 4, 2 and 9 (CPU: 4, 2, 8).

**The instrument does not perturb what it measures**, which W9's bit-identity
bar requires: the case runs the same gather with the probe unset and
`memcmp`s the outputs, and they are equal on every shape.

**Not repaired by a tolerance, and the reason is in the numbers this run
printed.** At the multi-tile shapes the correct kernel's arm-vs-arm ratio is
0.0000-0.0038, while the two golden fixtures at `|sel| = 2` read 0.1783 and
0.2041 -- so the worst ratio on a CORRECT kernel is an order of magnitude above
the band a reassociation lives in, exactly as the review measured.

The other three suites at the same binary: `test_qwen4_exp_qsa` 14/14,
`test_qwen4_exp_qsa_device` 12/12, `test_qwen4_exp_cuda_reductions` 20/20.
`test_qwen4_exp_qsa_block` reported 12 of 13 with
`the CUDA logits must stay far enough from the CPU's that no selection can flip`
red; that assertion is on the INDEXER logits, which this change does not touch,
and the base-vs-fix A/B below says so with numbers.

### THAT BLOCK-SUITE RED IS PRE-EXISTING, MEASURED, not argued (`rc` job `b58e2aee`)

ONE job, ONE clone, ONE build directory, two checkouts, built and run the same
way, three reps each plus the full suite, every filtered run asserting
`selected_cases=1`:

| arm | binary md5 | the one case | full suite |
|---|---|---|---|
| BASE `7d0d74c2c` (W9's head) | `14137d19250ed76b941db7de9b05e7b8` | RED 3 of 3 reps | 12 of 13 |
| FIX `a6ad48295` (this repair) | `fe65429ca506246c9bddc92b06d64b23` | RED 3 of 3 reps | 12 of 13 |

Identical on both arms and deterministic rather than flaky (6 of 6 reps). It is
tracked as its own issue under `.agents/issues/MODEL-MM-QWEN4-EXP/` and is NOT
repaired here.

### STATE

CLOSED by this change. The gate is committed, red against both reassociations
and green unmutated. What it does NOT establish is recorded in the issue above
and in the row's kernel-time issue: the block suite's selection-margin case, and
the re-rank that issue still owes.
