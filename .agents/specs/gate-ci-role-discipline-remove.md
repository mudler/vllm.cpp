# Remove check-role-discipline.py: the gate is permanently red and protects nothing

Issue: [#3191](https://github.com/mudler/vllm.cpp/issues/3191)
Row: `GATE-CI-ROLE-DISCIPLINE-REMOVE`

## Scope

Remove `scripts/check-role-discipline.py`, the enforcement floor mechanism
(`scripts/ci-enforcement-floor.txt`, `scripts/ci-walk-base.py`'s floor
clamping), the floor spec, and every CI step and test that invokes or asserts
the checker. The worktree/row-branch rule stays in `AGENTS.md` as guidance; it
is no longer gated on push-to-main history.

## Why

`check-role-discipline.py` enforces that every commit touching tracked files
arrives on `main` through a `row/<ID>` branch or PR. The enforcement floor has
been advanced 3 times (44 commits forgiven), and 45 more violations have
accumulated since, all by the project owner pushing directly to main. The
spec itself admits: "a job that is always red trains every reader to skip it,
and a skipped job protects nothing."

`commit-protocol-tag` already solved the same problem (trailer enforcement on
landed history) by moving to the PR lane. Role discipline follows the same path:
the rule stays in `AGENTS.md`, but the gate stops reading landed history.

## Design

### Files deleted

- `scripts/check-role-discipline.py`
- `scripts/ci-enforcement-floor.txt`
- `.agents/specs/ci-enforcement-floor.md`
- `tests/scripts/test_check_role_discipline.py`

### Files edited

- `.github/workflows/ci.yml` — remove the "Agent role machinery and role
  discipline" step from `agent-record`; remove the `check-role-discipline.py`
  invocation from `documentation-checkpoint`; remove floor comments from the
  three remaining `ci-walk-base.py` call sites.
- `scripts/ci-walk-base.py` — strip the floor logic (`read_floor`,
  `FloorError`, `DEFAULT_FLOOR_FILE`, `SHA`, the `--floor` and `--floor-file`
  args, the clamping block in `resolve_base`). The base resolver stays; it
  still provides last-green lossless base selection for `check-now-current.py`
  and the trailer walks.
- `scripts/check-pr-size.py` — remove `load_role_discipline()` and the
  `CHECKER_EVIDENCE_OVERRIDES` entry for `check-role-discipline.py`.
- `scripts/agent-preflight.sh` — remove `check-role-discipline` from the
  `SUITES` array.
- `tests/scripts/test_ci_walk_base.py` — remove floor-specific tests; keep
  base-resolution tests.
- `tests/scripts/test_agent_role.py` — remove the `RoleDiscipline` and
  `MergeLandedPrContent` classes and the `discipline` import.
- `tests/scripts/test_agent_gates.py` — remove the role-discipline step
  assertion.
- `tests/scripts/test_main_baseline.py` — remove `check-role-discipline.py`
  from `RANGE_SCOPED`; remove the `--floor ""` arg from the base-resolution
  test.
- `tests/scripts/test_check_pr_size.py` — remove tests that call
  `load_role_discipline()`.
- `tests/scripts/test_check_commit_trailers.py` — remove the
  `test_the_enforcement_floor_no_longer_claims_the_trailer_steps` test.

## Risks

- A `ci-walk-base.py` caller that still passes `--floor` will get an argparse
  error. No caller does after this change.
- `check-pr-size.py` tests that imported `load_role_discipline()` are removed;
  the PR size checker itself never called it.

## Tests

- The full test suite runs green with the checker, floor file, and floor tests
  deleted.
- `ci-walk-base.py` still resolves the base correctly for `check-now-current.py`
  and the trailer walks.
- `documentation-checkpoint` and `agent-record` jobs stay green (they still
  run other checkers).

## Stop conditions

- If removing the checker reveals a non-obvious dependency that cannot be
  resolved in this change, stop and report.

## Outcome

Removed `check-role-discipline.py`, `ci-enforcement-floor.txt`,
`ci-enforcement-floor.md`, and `test_check_role_discipline.py`. Stripped the
floor logic from `ci-walk-base.py` (removed `read_floor`, `FloorError`,
`DEFAULT_FLOOR_FILE`, `SHA`, `--floor`/`--floor-file` args, and the clamping
block in `resolve_base`). Removed the role-discipline step from `agent-record`
and the `check-role-discipline.py` invocation from `documentation-checkpoint`
in `ci.yml`. Removed `load_role_discipline()` and the evidence override from
`check-pr-size.py`. Removed `check-role-discipline` from `agent-preflight.sh`.
Removed the `RoleDiscipline` and `MergeLandedPrContent` classes and the
`discipline` import from `test_agent_role.py`. Updated floor-specific tests in
`test_ci_walk_base.py`, `test_main_baseline.py`, `test_check_pr_size.py`, and
`test_check_commit_trailers.py`. Updated `CONTRIBUTING.md` and `ci.yml`
comments that referenced the deleted checker or the enforcement floor.

Measured: 251 tests passed (562 subtests) across all affected test files.
`ci-walk-base.py --help` confirms the clean interface with no floor args. The
worktree/row-branch rule stays in `AGENTS.md` as guidance; it is no longer
gated on push-to-main history.

Rejected: moving role discipline to the PR lane (as `commit-protocol-tag`
did). The checker reads landed history to verify branch arrival, which is a
property of `main`'s graph, not of a PR's diff. A PR-lane check would verify
the PR's own branch name, which is trivially satisfied and protects nothing.
The rule stays as prose guidance in `AGENTS.md`.
