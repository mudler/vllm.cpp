# Spec — `CHECKER-STYLE-MERGE-BASE`: the style checker walks from the merge base, so preflight stops forcing a merge

Issue: [#2366](https://github.com/mudler/vllm.cpp/issues/2366). Owner row:
`CHECKER-STYLE-MERGE-BASE`. Branch `row/CHECKER-STYLE-MERGE-BASE`, base
`d574f514a` (origin/main, pinned at worktree creation).

## Scope

One behavior change, two files, plus the tests that pin them:

1. `scripts/check-commit-style.py` `validate_range` (lines 114-131) ports
   the #773 repair the trailers checker already carries
   (`check-commit-trailers.py:402-407`): resolve the range base to
   `merge_base(base, head)` instead of refusing a non-ancestor base with
   `range base must be an ancestor of range head` (lines 125-126).
   Unrelated histories still fail closed: no merge base means no range.
   The cutover comparison logic is untouched.
2. `scripts/agent-preflight.sh` drops the ancestry SKIP arm for status 1
   (the `TRAILER_BEHIND` block, lines 569-576). The gates then run for a
   branch behind main over `${BASE_SHA}..HEAD`, whose commit set is the
   branch's own commits under any ancestry. The unresolved-base arm, the
   ancestry-ERROR arm (status > 1, unborn HEAD), and both range-count
   arms stay exactly as they are, with their verbatim-pinned messages.
3. The comment above the dropped arm (lines 550-554) cites "#999" for the
   owed repair; that issue does not exist on the forge. The comment is
   rewritten to cite #2366 and the landed repair.

Exclusions: `check-commit-trailers.py` is already correct (#773, #2157)
and is not modified. `BASE_UNRESOLVED`, `ANCESTRY_UNKNOWN`, and
`RANGE_UNKNOWN` texts are verbatim-pinned and untouched. No change to
what the gates enforce — only to when they run.

## Design

`rev-list BASE..HEAD` selects exactly "commits reachable from HEAD, not
from BASE" regardless of ancestry, so the commit set under judgement is
already correct for a behind branch. The refusal lived only in the style
checker's validation and in the preflight guard compensating for it. With
the merge-base walk ported, a behind branch's gates run over the same
commits a post-rebase branch would present, and `agent-ready` stops
forcing a merge to reach green.

## Risks

- CI hands `pull_request.base.sha`-style ranges whose base stops being an
  ancestor as soon as main advances (#773's original finding). The style
  checker gains the same tolerance the trailers checker has had, so the
  two walks converge instead of diverging.
- A genuinely unrelated history has no merge base; both checkers fail
  closed with a stated error rather than inventing a range.
- The skip-count semantics of `agent-ready.py` are unchanged: fewer
  SKIPs on behind branches is the intended effect, not a weakening of
  any assertion.

## Tests

- `tests/scripts/test_check_commit_style.py`: a fixture repo whose range
  base is behind head (shared history, base not an ancestor) currently
  raises; the new test asserts `validate_range` returns failures computed
  over `merge_base..HEAD`. RED before the fix.
- `tests/scripts/test_agent_preflight_skip_report.py`: a behind-base
  branch case asserting both gates RUN — no `SKIP commit-trailers` /
  `SKIP commit-style` lines, exit 0 on a clean range. RED before the fix
  (both gates skip today).
- Existing suites stay green unchanged, including every verbatim message
  pin that this change does not touch.

Each new assertion mutated red per the house rule; the reviewer re-derives
them by mutation (re-add the ancestor refusal; re-add the SKIP arm) and
expects both new tests to red.

## Gates

Focused: `test_check_commit_style.py`,
`test_agent_preflight_skip_report.py`, `test_check_commit_trailers.py`.
Then the full `scripts/agent-preflight.sh`, then
`python3 scripts/agent-ready.py`. Evidence: run logs captured in the PR
description; the red-before captures in the implementer report.

## Stop conditions

- If the merge-base port cannot keep the cutover comparison total (an
  incomparable commit case), STOP and report `NEEDS_DECISION`; do not
  weaken the cutover contract to land the range fix.
- If a verbatim-pinned message outside the dropped arm must change for
  the suite to pass, that is a scope violation — STOP and report.

## Git integration

One PR for spec and implementation (repository default, recorded
policy). Spec committed before implementation on
`row/CHECKER-STYLE-MERGE-BASE`. The PR body carries `Closes #2366`, and
No `.agents/issue-index.md` row: the index is retired upstream by
`7dc2ef1ea` (ENG-RECORD-CONFLICT-SURFACES W6 — GitHub is the issue
index), and this PR does not resurrect the file.
