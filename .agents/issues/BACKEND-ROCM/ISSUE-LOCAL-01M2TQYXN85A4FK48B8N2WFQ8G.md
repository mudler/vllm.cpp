ID: ISSUE-LOCAL-01M2TQYXN85A4FK48B8N2WFQ8G
Title: check-rocm-dp4a-intrinsic cannot detect its own mutation
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

tests/scripts/test_check_rocm_dp4a_intrinsic.py::test_live_scalar_mutation_fails is RED on main: it replaces the __ockl_sdot4 call in the real src/vt/rocm/rocm_grouped_gemm.hip with the scalar expansion and the checker returns zero errors. The mutated Dp4a body contains ZERO call sites of the intrinsic, yet both of the checker's disjuncts pass on the surviving '#if __has_builtin(__ockl_sdot4)' probe line. The gate therefore measures a string, not a call, and the v_dot4_i32_i8 performance lever can regress silently.

## Resolution

-
