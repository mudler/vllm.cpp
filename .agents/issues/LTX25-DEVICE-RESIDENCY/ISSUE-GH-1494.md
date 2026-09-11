ID: ISSUE-GH-1494
Title: **A SECOND LTX-2.5 phase-coverage ratio decides by box load, and it is NOT the one [#1439](https://github.com/mudler/vllm.cpp/issues/1439) tracks.** `ltx2 video: the three carrying phases contain their work and the load keeps its order` asserts `CHECK_MESSAGE(covered >= c.min_coverage * leaf_seconds, ...)` at `tests/vllm/multimodal/test_ltx2_video.cpp:3696`; #1439 is `CHECK(leaves >= 0.95 * wall)` in a DIFFERENT case, now at `:3259`. Closing one does not close the other. Measured 2026-08-20, x86_64 `Release` `VLLM_CPP_CUDA=OFF`, three consecutive full-suite runs of ONE binary (`sha256 8fdbc31d...`) with no source change: loadavg 10.45 gives 94.6039% RED, a quieter run gives 96.8506% green, loadavg 16.53 gives 94.6039% RED. The comparison prints as `CHECK( 0.00414483 >= 0.00416218 )` - `denoise` is 0.00438124 s, its eight named sub-scopes cover 0.00414483 s, so the un-named residue is 0.00023641 s and **the margin is a quarter of a millisecond**. Same scheduling polarity #1439 recorded, which is the tell that this is the instrument and not the code: the run that PASSED is the run where `denoise` took 0.00940481 s, more than twice the failing runs', because the residue grows more slowly than the leaf it is divided by. NOT FIXED IN FLOW: bounding the residue in SECONDS beside the ratio, so the assertion says the same thing at fixture and production scale, changes a gate's semantics and needs its own row, spec and red-first evidence per `AGENTS.md` `## Changing the rules or a checker` - the same conclusion #1439 reached, and the two should be repaired together because one seconds bound would serve both. Found by the fresh implementer repairing the review findings of [#1481](https://github.com/mudler/vllm.cpp/pull/1481); pre-existing and not that PR's defect
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: bug
GitHub: 1494
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:521`

### Frozen archive evidence

> | [#1494](https://github.com/mudler/vllm.cpp/issues/1494) | `LTX25-DEVICE-RESIDENCY` | **A SECOND LTX-2.5 phase-coverage ratio decides by box load, and it is NOT the one [#1439](https://github.com/mudler/vllm.cpp/issues/1439) tracks.** `ltx2 video: the three carrying phases contain their work and the load keeps its order` asserts `CHECK_MESSAGE(covered >= c.min_coverage * leaf_seconds, ...)` at `tests/vllm/multimodal/test_ltx2_video.cpp:3696`; #1439 is `CHECK(leaves >= 0.95 * wall)` in a DIFFERENT case, now at `:3259`. Closing one does not close the other. Measured 2026-08-20, x86_64 `Release` `VLLM_CPP_CUDA=OFF`, three consecutive full-suite runs of ONE binary (`sha256 8fdbc31d...`) with no source change: loadavg 10.45 gives 94.6039% RED, a quieter run gives 96.8506% green, loadavg 16.53 gives 94.6039% RED. The comparison prints as `CHECK( 0.00414483 >= 0.00416218 )` - `denoise` is 0.00438124 s, its eight named sub-scopes cover 0.00414483 s, so the un-named residue is 0.00023641 s and **the margin is a quarter of a millisecond**. Same scheduling polarity #1439 recorded, which is the tell that this is the instrument and not the code: the run that PASSED is the run where `denoise` took 0.00940481 s, more than twice the failing runs', because the residue grows more slowly than the leaf it is divided by. NOT FIXED IN FLOW: bounding the residue in SECONDS beside the ratio, so the assertion says the same thing at fixture and production scale, changes a gate's semantics and needs its own row, spec and red-first evidence per `AGENTS.md` `## Changing the rules or a checker` - the same conclusion #1439 reached, and the two should be repaired together because one seconds bound would serve both. Found by the fresh implementer repairing the review findings of [#1481](https://github.com/mudler/vllm.cpp/pull/1481); pre-existing and not that PR's defect | bug |

## Resolution

-
