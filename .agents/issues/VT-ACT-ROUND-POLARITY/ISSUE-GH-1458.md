ID: ISSUE-GH-1458
Title: **`4712dac40` reds FOUR suites on `main` — `test_ltx2_text_encoder`, `test_muse_glimmer_text`, `test_muse_glimmer_text_fallback`, `test_minimax_music3_ar` — by exceeding bf16 error floors none of them had re-derived.** Found while gating [#1403](https://github.com/mudler/vllm.cpp/issues/1403) (PR #1457) and filed in flow, not that row's defect. Deterministic, not a load artifact: first seen under `-j4` and re-measured SERIALLY, 101 s across the four. `test_ltx2_text_encoder.cpp:2407` reads `CHECK( 0.1323 <= 0.109394 )` and `:2409` reads `CHECK( 0.0752773 <= 0.0573374 )`, both over by 20-30%, with the file at `26 passed / 1 failed`, `4118 assertions`, `Status: FAILURE!`. ATTRIBUTED BY MUTATION rather than inferred, both directions, in one build directory changing only `src/vt/cpu/cpu_ops.cpp`, compile rc 0 on every arm and `sha256sum` taken before and after: at `0adeb8b0e` four fail; reverted to `4712dac40^` four pass and `test_ltx2_text_encoder` is 27/27 `SUCCESS!`; restored to a re-matched `sha256` four fail again. They also passed at `b537a5344`, three commits earlier, in a full 567-test `ctest` run at 3-6 s each. `4712dac40` is +42/-3 in `src/vt/cpu/cpu_ops.cpp` plus a new `tests/vt/test_ops_activation.cpp`, and touches none of the four. NOT FIXED IN FLOW, deliberately: the question is a numerics decision, not a defect with an obvious repair — either the floors were calibrated against the rounding polarity that commit corrected and need re-deriving against the oracle, or the narrowing is wider than upstream's — and editing the four floors to pick the first branch is the scope-widening `AGENTS.md` prohibits
Row: VT-ACT-ROUND-POLARITY
State: UNKNOWN
Kind: bug
GitHub: 1458
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:504`

### Frozen archive evidence

> | [#1458](https://github.com/mudler/vllm.cpp/issues/1458) | `VT-ACT-ROUND-POLARITY` | **`4712dac40` reds FOUR suites on `main` — `test_ltx2_text_encoder`, `test_muse_glimmer_text`, `test_muse_glimmer_text_fallback`, `test_minimax_music3_ar` — by exceeding bf16 error floors none of them had re-derived.** Found while gating [#1403](https://github.com/mudler/vllm.cpp/issues/1403) (PR #1457) and filed in flow, not that row's defect. Deterministic, not a load artifact: first seen under `-j4` and re-measured SERIALLY, 101 s across the four. `test_ltx2_text_encoder.cpp:2407` reads `CHECK( 0.1323 <= 0.109394 )` and `:2409` reads `CHECK( 0.0752773 <= 0.0573374 )`, both over by 20-30%, with the file at `26 passed / 1 failed`, `4118 assertions`, `Status: FAILURE!`. ATTRIBUTED BY MUTATION rather than inferred, both directions, in one build directory changing only `src/vt/cpu/cpu_ops.cpp`, compile rc 0 on every arm and `sha256sum` taken before and after: at `0adeb8b0e` four fail; reverted to `4712dac40^` four pass and `test_ltx2_text_encoder` is 27/27 `SUCCESS!`; restored to a re-matched `sha256` four fail again. They also passed at `b537a5344`, three commits earlier, in a full 567-test `ctest` run at 3-6 s each. `4712dac40` is +42/-3 in `src/vt/cpu/cpu_ops.cpp` plus a new `tests/vt/test_ops_activation.cpp`, and touches none of the four. NOT FIXED IN FLOW, deliberately: the question is a numerics decision, not a defect with an obvious repair — either the floors were calibrated against the rounding polarity that commit corrected and need re-deriving against the oracle, or the narrowing is wider than upstream's — and editing the four floors to pick the first branch is the scope-widening `AGENTS.md` prohibits | bug |

## Resolution

-
