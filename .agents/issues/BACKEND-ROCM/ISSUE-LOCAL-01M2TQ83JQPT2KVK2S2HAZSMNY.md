ID: ISSUE-LOCAL-01M2TQ83JQPT2KVK2S2HAZSMNY
Title: test_check_rocm_dp4a_intrinsic is red on main: the live scalar mutation no longer makes the dp4a checker fire
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: 2026-09-18

## Problem

`python3 -m unittest tests.scripts.test_check_rocm_dp4a_intrinsic` fails on a clean checkout of `origin/main` (`7fa861392`), in a worktree whose `src/vt/rocm/rocm_grouped_gemm.hip`, `scripts/check-rocm-dp4a-intrinsic.py` and `tests/scripts/test_check_rocm_dp4a_intrinsic.py` are byte-identical to main:

```
AssertionError: 0 != 1 : []
Ran 6 tests in 0.003s
FAILED (failures=1)
```

The failing case is `test_live_scalar_mutation_fails` (`tests/scripts/test_check_rocm_dp4a_intrinsic.py:108`). Its comment states what it exists for: "prove the gate fails when v_dot4_i32_i8 is not emitted". It mutates the live source by replacing the line `  return __ockl_sdot4(va, vb, acc, false);` with the scalar expansion, and requires the checker to report one error. The checker now reports none.

Mechanism, read in the tree: `2bde17f6c fix(ROCM): add conditional __ockl_sdot4 intrinsic with scalar fallback` (2026-09-17) rewrote `Dp4a` (`src/vt/rocm/rocm_grouped_gemm.hip:82-98`) so that the intrinsic sits inside `#if __has_builtin(__ockl_sdot4)` and a scalar loop follows it as a permanent fallback. The token `__ockl_sdot4` therefore appears in the `#if` line as well as in the call. Deleting the call alone leaves the token in the file, so a presence-based check still passes and the mutation is no longer detected.

Two things follow, and the second is the reason this is filed rather than left:

1. The suite is red on main, so `scripts/agent-preflight.sh` is not green for any change in this tree, and every session now has to know which red is "the known one". The `tools suites` disk-headroom red already occupies that slot.
2. The gate's own guarantee is weaker than it reads. A tree that stops emitting `v_dot4_i32_i8` — for example because the builtin probe is false on some ROCm, which is exactly the case `2bde17f6c` added the fallback for — now passes the checker. The mutation this test performs is the one that proves otherwise, and it no longer fires.

Not established here: whether the fallback should exist at all on the gated target, what `2bde17f6c` measured, and whether the checker should assert the emitted ISA instead of the source token. Those are the owning row's call. This issue records the red and its cause; per CLAUDE.md a checker-semantics change needs its own spec, a red-before test and green-after evidence, so it is not an in-flow fix.

Found while running the full harness suite for `BENCH-QWEN38-EXL3-LONGCTX`, which touches nothing under `src/vt/rocm` or `scripts/check-rocm-*`.

## Resolution

2026-09-18: fixed on main by df87f880e (merge: row/ROCMDP4A-checker-blind, make the dp4a checker require a call and not a mention), by another session, independently of this record. Verified here on a detached checkout of origin/main at df87f880e: python3 -m unittest tests.scripts.test_check_rocm_dp4a_intrinsic runs 10 tests, OK, where the same command at 7fa861392 failed test_live_scalar_mutation_fails with AssertionError: 0 != 1. The suite also grew from 6 cases to 10. This issue therefore closes on the tree that falsifies it rather than on work of its own.
