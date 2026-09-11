ID: ISSUE-GH-776
Title: `test_op_parity` THREW `json.exception.type_error.302` out of the CPU golden pass instead of failing an assertion, so the walker's `no runner for op` guard — the check that caught #559's missing runner arm — stopped running for every golden after the offender. The artifact was `tests/parity/goldens/minimax_music3_oracle/manifest.json`, and that half is #755, already fixed by `043e56862`. #755 closed the walker's INPUT set; it did not close its EXCEPTION surface, and reproducing #776 on the fixed tree shows the difference: nulling one tensor `dtype` in `rmsnorm_f32_8x128` still threw `type_error.302` at the TEST_CASE line and cut the pass from 142 assertions to 37, leaving 45 committed goldens unchecked. Both remaining throw sites — `json::parse` and any runner field read — now funnel through `GuardGoldenStage`, which turns a `std::exception` into a `FAIL_CHECK` naming `goldens/<case>/manifest.json` and continues. `doctest::detail::TestFailureException` is deliberately not a `std::exception` (`third_party/doctest/doctest.h:2563`), so the #559 `FAIL` still aborts loudly and the widening cannot mute it. FIXED IN FLOW
Row: ENG-PREFLIGHT-COMPILES
State: UNKNOWN
Kind: bug
GitHub: 776
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:213`

### Frozen archive evidence

> | [#776](https://github.com/mudler/vllm.cpp/issues/776) | `GATE-OP-PARITY-MANIFEST` | `test_op_parity` THREW `json.exception.type_error.302` out of the CPU golden pass instead of failing an assertion, so the walker's `no runner for op` guard — the check that caught #559's missing runner arm — stopped running for every golden after the offender. The artifact was `tests/parity/goldens/minimax_music3_oracle/manifest.json`, and that half is #755, already fixed by `043e56862`. #755 closed the walker's INPUT set; it did not close its EXCEPTION surface, and reproducing #776 on the fixed tree shows the difference: nulling one tensor `dtype` in `rmsnorm_f32_8x128` still threw `type_error.302` at the TEST_CASE line and cut the pass from 142 assertions to 37, leaving 45 committed goldens unchecked. Both remaining throw sites — `json::parse` and any runner field read — now funnel through `GuardGoldenStage`, which turns a `std::exception` into a `FAIL_CHECK` naming `goldens/<case>/manifest.json` and continues. `doctest::detail::TestFailureException` is deliberately not a `std::exception` (`third_party/doctest/doctest.h:2563`), so the #559 `FAIL` still aborts loudly and the widening cannot mute it. FIXED IN FLOW | bug |

## Resolution

-
