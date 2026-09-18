# The dp4a gate has to see a call, and it only sees a string

Issue: `ISSUE-LOCAL-01M2TQYXN85A4FK48B8N2WFQ8G`
Row: `BACKEND-ROCM`

## Why

`scripts/check-rocm-dp4a-intrinsic.py` exists because a CPU-only `ctest` run
cannot compile `src/vt/rocm/rocm_grouped_gemm.hip`, so nothing else in the tree
notices if the hardware dot product leaves `Dp4a`. The checker reads the source
and is the whole guarantee that `v_dot4_i32_i8` is still emitted.

It no longer is one. `tests/scripts/test_check_rocm_dp4a_intrinsic.py:126`
mutates the real source and the checker returns an empty error list.
`scripts/agent-preflight.sh` reports `1 gate(s) failed` on `main` for this.

The mutation replaces the call line, and the mutated `Dp4a` body reads:

```
#if defined(__has_builtin)
#if __has_builtin(__ockl_sdot4)
  int sum = 0;
  ...                        <- the scalar expansion, no intrinsic anywhere
#endif
#endif
  // Fallback: scalar expansion
  ...
```

Zero call sites. Both of the checker's disjuncts pass anyway, each on its own,
because the token `__ockl_sdot4` survives on the `__has_builtin` **probe** line
and both tests are substring tests:

- `INTRINSIC in body` matches the probe.
- `"__has_builtin(__ockl_sdot4)" in body` matches the probe by construction.

So the second disjunct can never fail while the first passes, and the first
passes on a body that calls nothing. Measured in-tree before any change:
`body.count("__ockl_sdot4") == 1`, real call sites `== 0`, both flags `True`.

## Which half is stale

The checker. `git log --follow` on the three files settles it:

| File | Last touched |
|---|---|
| `tests/scripts/test_check_rocm_dp4a_intrinsic.py` | `6fd5443fa` |
| `scripts/check-rocm-dp4a-intrinsic.py` | `2bde17f6c` |
| `src/vt/rocm/rocm_grouped_gemm.hip` | `2bde17f6c` (for this hunk) |

`2bde17f6c` ("fix(ROCM): add conditional `__ockl_sdot4` intrinsic with scalar
fallback") changed the source and the checker in one commit and did not touch
the test. Its own message says *"Update checker to accept conditional
fallback pattern"*, and the diff adds the `has_conditional_fallback` disjunct.
That is the shape AGENTS.md names: *"Never make a red gate green by deleting an
assertion or widening its scope."* The test is the half that did not move, so
the test is not the stale one.

The **product** intent of `2bde17f6c` is legitimate and is kept: `__ockl_sdot4`
is absent from some ROCm installs, so the call is guarded and a scalar fallback
follows it. The defect is that the checker was widened to accept the guard
*instead of* the call rather than *around* it.

## Scope

- `scripts/check-rocm-dp4a-intrinsic.py`: detect a genuine call site.
- `tests/scripts/test_check_rocm_dp4a_intrinsic.py`: pin the new shape, in both
  directions, and repair one case that does not test what it is named for.

Out of scope: `src/vt/rocm/rocm_grouped_gemm.hip`. The live source is correct
under the restored rule and is not edited. No kernel, build or ROCm behaviour
changes; this is a checker-only change and needs no GPU.

## Design

The rule, restated so the checker and this file say the same thing: **the
`Dp4a` body must contain at least one genuine call to `__ockl_sdot4`.** A
`#if __has_builtin(__ockl_sdot4)` guard around that call is allowed, and a
scalar fallback after it is allowed. Neither is a substitute for the call.

Detection, in order:

1. Strip `//` and `/* */` comments from the extracted body, so an intrinsic
   named only in prose cannot satisfy the gate.
2. Require `__ockl_sdot4\s*\(` to match what is left.

A third step was written and then removed: an explicit pass deleting every
`__has_builtin( __ockl_sdot4 )` probe. It was **unreachable**, and the
mutation matrix is what proved it — deleting the pass changed no verdict and
the test file stayed green, the only surviving arm of the run. The reason is
that inside the probe the intrinsic is followed by `)`, so step 2 already
rejects it. Shipping the pass would have added exactly the shape this row
exists to remove, so it is gone and
`test_has_builtin_probe_alone_fails` pins the behaviour instead.

The `has_conditional_fallback` disjunct is removed. It is not an assertion
being deleted: it is the widening, and after step 2 it cannot be satisfied by
anything the first test does not already cover.

`_SCALAR_MARKERS` is removed in the same change. It is dead — defined, never
read — and it describes a rule (`a * b` means regression) the checker has never
enforced, so a reader of the checker is misled about what the gate does.

## Risks

- **Over-tight regex refuses the live tree.** Mitigated by
  `test_live_tree_passes` and by a new case that builds the exact live shape
  (guarded call plus scalar fallback) in a scratch tree.
- **A future legitimate spelling is refused**, for example a call through a
  macro or a function pointer. Accepted: the checker is a source-shape gate by
  construction, and a spelling change to `Dp4a` should re-read this gate.
- **Vacuous green elsewhere in the file.** Addressed below rather than assumed.

## Tests

Red before: `test_live_scalar_mutation_fails` fails with `0 != 1 : []`.

Green after, plus these new cases:

| Case | Asserts |
|---|---|
| `test_has_builtin_probe_alone_fails` | the probe without a call is red — pins the removed disjunct dead |
| `test_intrinsic_in_comment_only_fails` | prose is not a call |
| `test_guarded_call_with_scalar_fallback_passes` | `2bde17f6c`'s real shape is still accepted |
| `test_missing_dp4a_function_fails` | repaired: it now removes the signature, so it reaches the `Dp4a function not found` branch it is named for |
| `test_empty_dp4a_body_fails` | the case the old `test_missing_dp4a_function_fails` actually ran, kept under an honest name |

Audit of the five pre-existing cases for vacuity is recorded in `## Outcome`.

## Gates

`python3 tests/scripts/test_check_rocm_dp4a_intrinsic.py`,
`python3 scripts/check-rocm-dp4a-intrinsic.py`, and
`scripts/agent-preflight.sh`. No GPU lease is required and none is taken.

## Evidence

Both mutation directions, run by hand and recorded in the pull request body:

1. Mutate the **source**: replace the guarded call with the scalar expansion in
   the real `.hip` and confirm the repaired checker reports one error.
2. Mutate the **checker**: defeat the call detection and confirm the test file
   goes red, then restore and verify with `sha256sum -c`.

A restored `.py` can still run mutant `__pycache__` bytecode, so
`__pycache__` is purged between arms and every restore is hash-verified.

## Stop conditions

- Stop and return `NEEDS_DECISION` if the repaired checker refuses the live
  tree, because that would mean the source, not the checker, is the stale half.
- Stop if repairing detection would require editing
  `src/vt/rocm/rocm_grouped_gemm.hip`; that is a different change with a
  different reviewer.

## Outcome

**Which half was stale: the checker.** `2bde17f6c` widened it and left the
test behind. The repaired checker detects its own mutation; the A/B on one
byte-identical mutated source is `rc=1` new against `rc=0` old.

**Every one of the ten cases has teeth, and nothing is vacuous.** Nine
checker mutations were run against the test file, with `__pycache__` purged
between arms and each restore verified by sha256. Each arm was asserted to
have changed the file before the arm ran, and the selected count was asserted
at 10 on every arm, so no arm is a green over zero cases.

| Arm | Checker mutation | Result |
|---|---|---|
| B | drop the comment stripping | detected |
| C | call regex to bare substring (the pre-repair polarity) | detected |
| D | restore both old disjuncts verbatim | detected |
| E | `_has_call` always True | detected |
| F | `_has_call` always False | detected |
| G | delete the `if not _has_call` guard | detected |
| H | `_extract_dp4a` never finds the function | detected |
| I | drop the missing-source-file early return | detected |
| J | drop the `Dp4a function not found` error | detected |

Zero survivors after the probe pass was removed, and every line of the
checker is load-bearing under some arm.

`test_missing_source_file_fails` and `test_missing_dp4a_function_fails` fired
on no arm until `I` and `J` were written for them. They are not vacuous, but
neither was covered by a mutation until this row, and each is now pinned by
exactly one arm.

**One pre-existing case was misnamed, not vacuous.** The old
`test_missing_dp4a_function_fails` passed an empty *body*, which still has a
`Dp4a` signature, so it never reached the `Dp4a function not found` branch it
is named for; that branch had no test at all. It is split into
`test_empty_dp4a_body_fails` (what it measured) and a new
`test_missing_dp4a_function_fails` (what it claimed), and arm `J` shows the
new one reaches the branch.

**`_SCALAR_MARKERS` was dead and is removed.** It was defined, never read,
and described an `a * b` rule the checker never enforced.

**No source, kernel, build or ROCm behaviour changed**, and no GPU was used.
`src/vt/rocm/rocm_grouped_gemm.hip` is byte-identical to `origin/main` after
every mutation arm, verified with `sha256sum -c`.

## Now

`BACKEND-ROCM` does not change lifecycle state. This is a gate repair inside
the row, not a capability move.
