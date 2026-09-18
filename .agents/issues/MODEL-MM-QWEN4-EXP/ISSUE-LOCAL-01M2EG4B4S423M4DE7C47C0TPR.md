ID: ISSUE-LOCAL-01M2EG4B4S423M4DE7C47C0TPR
Title: eight MODEL-MM-QWEN4-EXP gates still reduce with the NaN-blind std::max spelling and FAIL OPEN
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

Found by the sweep the fresh review of the ROCm grouped-norm row (F1) asked for.
Repairing `test_qwen4_exp_rocm_reductions.cpp` case 5 -- the THIRD recurrence of
issue #449's form B, after #449 itself and #1988 in this same model family --
does not exhaust the spelling in this row's gates. `grep` over every test file
that mentions `qwen4_exp` finds it in 13 more places, and EIGHT of them are
fail-OPEN: an all-NaN operand reduces to 0.0 and the bound passes.

The eight, each with the assertion that consumes the blind reduction:

- `tests/vllm/models/test_qwen4_exp_hc_device.cpp:450-454` -- `worst`/`scale`
  form B, `CHECK(worst <= kRealWidthRel * std::max(scale, 1e-3))`.
- `tests/vllm/models/test_qwen4_exp_hc_device.cpp:457-461` -- `winj`,
  `CHECK(winj <= kRealWidthRel * 2.0)`.
- `tests/vllm/models/test_qwen4_exp_hc_device.cpp:589-608` --
  `CHECK(worst < kAccumBound)`. This is the accumulator-width case at `:504`
  that the ROCm row transcribes, so the ORIGINAL of the repaired case carries
  the defect the transcription just had removed.
- `tests/vllm/models/test_qwen4_exp_forward.cpp:76-81` -- a local `MaxAbsDiff`
  that is form B, consumed by `CHECK(... < 1e-5f)` at `:232` and `:234`. The
  `:165` call is fail-closed, because `== doctest::Approx(1.0f)` rejects a 0.
- `tests/vllm/models/test_qwen4_exp_matmul_bt_dtype.cpp:438-448` -- form A
  (`if (diff > worst) worst = diff;`), `CHECK(worst < 1e-4)`.
- `tests/vllm/models/test_qwen4_exp_cuda_reductions.cpp:613-632` -- the derived
  bound helper. `diff` is form-A-adjacent: a NaN `got[o]` makes `diff > bound`
  false, `n_over` stays 0, and `CHECK(n_over == 0)` passes.
- `tests/vllm/models/test_qwen4_exp_qsa_block.cpp:269-280` -- `MaxRelDiff` folds
  `worst` with form B. Three call sites (`:946`, `:1116`, `:1626`) add an
  explicit finiteness REQUIRE first and are fail-closed; the others are not.
- `tests/vllm/models/test_qwen4_exp_qsa_block.cpp:417-428` -- block logits vs
  golden, `CHECK(worst / scale < kLogitsTol)` with `worst` folded form B.

FAIL-CLOSED, spelling only, recorded so a later sweep does not re-derive them:
`qwen4_exp_hc_synth.h:130` and `test_qwen4_exp_cuda.cpp:277` (both are `scale`
folds whose `a.worst` already comes from the hardened `Compare`);
`test_qwen4_exp_hc.cpp:793` (`REQUIRE(scale > 0.1)` plus a hardened
`MaxAbsDiff`); `test_qwen4_exp_ple_block.cpp:850-862` (guarded by
`REQUIRE(std::isfinite(...))` two lines above); the five `sep`/`tail`
separations in `test_qwen4_exp_cuda_reductions.cpp` (`:770`, `:1062`, `:1109`,
`:1396`, `:1793`) and all four in `tests/vt/test_ops_rms_norm_group_cuda.cpp`
(every one asserts `> bound`, which a blind 0 misses);
`test_qwen4_exp_qsa_block.cpp:1643` (MESSAGE, never asserted);
`test_backend_cross_device.cpp:6941` (a softmax maximum, not a difference).

NOT FIXED IN THE ROW THAT FOUND IT, and the reason is named rather than
deferred silently. Six of the eight are CUDA-arm or CPU-golden gates belonging
to other waves of MODEL-MM-QWEN4-EXP; each repair owes its own red-first
NaN-poison mutation on the arm it gates, and four of them need a CUDA lease that
the ROCm row had no reason to take. Repairing them blind, without the paired red,
would be the same "prose is not a gate" failure in a new place. The ROCm row
repaired the two cases its own change owns and reports the rest here.

## Resolution

-
