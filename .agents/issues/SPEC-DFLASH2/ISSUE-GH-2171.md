ID: ISSUE-GH-2171
Title: **`DFlashAttnMmaKernel`'s multi-block QUERY path has never executed under test: every D1 case stops at `Tq=27` and the tile is 64 rows.** The kernel tiles the query axis at `kMmaWarps * kMmaQ = 4 * 16 = 64` (`src/vt/cuda/cuda_ops.cu:2372-2373`, grid `:2676`), and `RunD1Bf16Parity`'s four cases in `tests/vt/test_ops_dflash_block_attn.cpp` carry `Tq` of 18, 18, 18 and 27, so `mgrid.x` has always been 1. Production crosses the boundary on EVERY step — 8 concurrent requests at k=8 is `Tq = 8*9 = 72`, two query blocks, the second holding only the last request's nine rows. The comment above those cases reasons about walking several `kMmaKeys` tiles, which is the KEY axis; the query axis had no coverage past its first block. THIRD instance of this shape in one file, one axis over each time: a tiled CUDA path guarded to `num_reqs == 1` that "shipped never-executed while the suite stayed green", then an f32 harness that "by dispatch can never reach `DFlashAttnMmaKernel`" (the reason `RunD1Bf16Parity` exists). **The kernel PASSES at the missing shapes** — six added cases run on dgx:gpu0 GB10 sm_121a give 10 cases / 89886 assertions / ZERO failures, max|diff| 1.3e-4 — so this is a coverage gap, not a live defect, and a future regression there would have landed green. Controls are chosen for ATTRIBUTION: an 8-request red beside a 7-request `Tq=63` green isolates the query-block boundary, and a production-scale red (8 reqs, ctx ~1200, `Ncomb` ~9.7k) beside a single-request control at the same key extent isolates the many-request key union from context length. Found while investigating [#2154](https://github.com/mudler/vllm.cpp/issues/2154), where the query tile was a candidate mechanism for the acceptance collapse; these cases REFUTED that hypothesis
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 2171
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:846`

### Frozen archive evidence

> | [#2171](https://github.com/mudler/vllm.cpp/issues/2171) | `SPEC-DFLASH2` | **`DFlashAttnMmaKernel`'s multi-block QUERY path has never executed under test: every D1 case stops at `Tq=27` and the tile is 64 rows.** The kernel tiles the query axis at `kMmaWarps * kMmaQ = 4 * 16 = 64` (`src/vt/cuda/cuda_ops.cu:2372-2373`, grid `:2676`), and `RunD1Bf16Parity`'s four cases in `tests/vt/test_ops_dflash_block_attn.cpp` carry `Tq` of 18, 18, 18 and 27, so `mgrid.x` has always been 1. Production crosses the boundary on EVERY step — 8 concurrent requests at k=8 is `Tq = 8*9 = 72`, two query blocks, the second holding only the last request's nine rows. The comment above those cases reasons about walking several `kMmaKeys` tiles, which is the KEY axis; the query axis had no coverage past its first block. THIRD instance of this shape in one file, one axis over each time: a tiled CUDA path guarded to `num_reqs == 1` that "shipped never-executed while the suite stayed green", then an f32 harness that "by dispatch can never reach `DFlashAttnMmaKernel`" (the reason `RunD1Bf16Parity` exists). **The kernel PASSES at the missing shapes** — six added cases run on dgx:gpu0 GB10 sm_121a give 10 cases / 89886 assertions / ZERO failures, max\|diff\| 1.3e-4 — so this is a coverage gap, not a live defect, and a future regression there would have landed green. Controls are chosen for ATTRIBUTION: an 8-request red beside a 7-request `Tq=63` green isolates the query-block boundary, and a production-scale red (8 reqs, ctx ~1200, `Ncomb` ~9.7k) beside a single-request control at the same key extent isolates the many-request key union from context length. Found while investigating [#2154](https://github.com/mudler/vllm.cpp/issues/2154), where the query tile was a candidate mechanism for the acceptance collapse; these cases REFUTED that hypothesis | bug |

## Resolution

-
