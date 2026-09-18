ID: ISSUE-LOCAL-01M2TPZ5XKNPG1Z352YGHDQ0H7
Title: check-rocm-dp4a-intrinsic's own mutation test is green on the mutated source: the scalar-expansion arm no longer produces an error
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

MEASURED 2026-09-18 at base bca4ab09fcaa6e718fbc25740816f60be417bddb (clean detached worktree, no local edits): 'python3 tests/scripts/test_check_rocm_dp4a_intrinsic.py' gives 'Ran 6 tests ... FAILED (failures=1)'. The failure is test_live_scalar_mutation_fails at tests/scripts/test_check_rocm_dp4a_intrinsic.py:126, 'AssertionError: 0 != 1 : []'. The test replaces the '__ockl_sdot4' call in the LIVE src/vt/rocm/rocm_grouped_gemm.hip text with the scalar expansion and asserts the checker reports exactly one error; the checker now reports NONE. So the gate that is supposed to prove v_dot4_i32_i8 is emitted cannot detect the mutation it was written against, and check-rocm-dp4a-intrinsic.py is currently an instrument that succeeds at the wrong question: it is green on this tree while its own red-arm is green too. Either the live source no longer contains the exact string the test replaces (so the mutation silently does not apply, although the test asserts 'mutation did not apply' against that) or checker.check() stopped looking at the path the test writes. FOUND while porting per-group attention-backend dispatch for MODEL-MM-deepseek-v4 (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F); it is PRE-EXISTING and unrelated to that change, reproduced at the untouched base commit above. NOT FIXED THERE: repairing it changes checker semantics, which CLAUDE.md routes through its own spec and fresh review rather than through an in-flow fix.

## Resolution

-
