# GATE-FORK-ANCESTRY — diff a PR from its merge base, not from a moved branch tip

**Row:** `GATE-FORK-ANCESTRY`
**Issue:** [#773](https://github.com/mudler/vllm.cpp/issues/773)
**Base:** `origin/main` `5ddfca6f6`
**Status:** ACTIVE, 2026-08-14

## 1. Scope

Two checkers stop requiring the base revision to be an *ancestor* of head, and
instead compute the range from `git merge-base base head`:

- `scripts/check-pr-size.py` — `require_ancestor()` / `changed_paths()`
- `scripts/check-commit-trailers.py` — `validate_range()`

**In scope.** Those two range computations, the two existing tests that pin the
current behaviour, and new tests for both halves of the distinction in §3.

**Out of scope.** Every rule either checker enforces once the range exists: path
classification, the checker-evidence contract, the role checks, the trailer
contract itself, cutover handling. None of them changes. This alters *which
commits are examined*, never *what is demanded of them*.

Also out of scope: `audit-live-rows` (#726) and `check-windows-portability`
(#774) are being repaired concurrently and this change touches neither.

## 2. Anchors

Local governance checkers; no vLLM counterpart.

| What | Where |
|---|---|
| The pr-size guard | `scripts/check-pr-size.py` `require_ancestor()` |
| Its caller | `scripts/check-pr-size.py` `changed_paths()` |
| The trailer guard | `scripts/check-commit-trailers.py` `validate_range()` |
| What CI passes as base | `.github/workflows/ci.yml`, `github.event.pull_request.base.sha` |
| Test pinning current pr-size behaviour | `tests/scripts/test_check_pr_size.py::test_missing_and_nonancestor_objects_fail_closed` |
| Test pinning current trailer behaviour | `tests/scripts/test_check_commit_trailers.py::test_missing_unreachable_and_non_ancestor_revisions_fail_closed` |

## 3. Design

`github.event.pull_request.base.sha` is the **tip of the base branch**. It stops
being an ancestor of head the moment `main` moves after the branch was cut,
which on this repo is continuous. Measured on three open PRs (#506, #523, #559):
base is not an ancestor in any of them, and a merge base exists in all three.

Both checkers therefore abort **before validating anything**. The consequence is
not merely a red check:

> CI has never validated commit trailers on an external contribution. The check
> that enforces `FOLLOWING_AGENTS_PROTOCOL` and `Assisted-by:` exits before it
> reads a single commit.

The fix is to use what a pull request actually *is*. `git diff A...B` (three-dot)
is defined as `git diff $(git merge-base A B) B`, and it is what GitHub shows.
Two-dot diffing against a moved `main` is not stricter, it is **wrong**: main's
own commits appear as reversions inside the contributor's diff, so paths the
contributor never touched get classified and charged to them.

**The distinction that must survive.** "Base is not an ancestor" currently
conflates two situations:

| Situation | Merge base | Correct behaviour |
|---|---|---|
| Ordinary divergence — branch cut from main, main moved on | exists | **Examine `merge_base..head`.** This is every PR. |
| Unrelated histories — orphan branch, wrong repo, garbage revision | none | **Fail closed.** Absence of information is not absence of work. |

Only the first changes. The second keeps raising, and the existing orphan-branch
test keeps asserting it — which is why that test is preserved verbatim rather
than relaxed.

**Why not "fetch more in CI instead".** That would make the base an ancestor
again only by luck of timing, and would leave the checkers wrong for anyone
running them locally against a branch cut before the last merge. The range
computation is the defect; the fetch is not.

## 4. Risks and decisions

| Risk | Assessment |
|---|---|
| Relaxing an assertion to make a gate pass | The orphan/unrelated-histories case still raises, and its test is kept unchanged. What is removed is a demand that no pull request in this repository can satisfy — a rule nothing can meet is not enforcing a standard. Argued in the commit message per the no-waiver-registry rule. |
| A contributor hides a change behind an old merge base | They cannot. `merge_base..head` contains exactly the commits the PR adds; anything they touch is in it. What leaves the range is main's own work, which is precisely what should not be charged to them. |
| The trailer checker examines fewer commits and misses one | It examines *more* correctly-scoped commits: today it examines **zero** on a fork PR, because it aborts. Any commit the PR introduces is reachable from head and not from the merge base, so it is in range by construction. |
| Cutover handling drifts | `cutover` is validated against `head`, not against `base`, and that check is untouched. |
| Divergent-history test now passes where it failed | Deliberate, and the reason the row exists — see §3's table. Split into two cases so the surviving half is asserted explicitly rather than deleted. |

## 5. Tests

RED-first in both suites.

1. `test_pr_size_uses_the_merge_base_when_main_moved` — build a repo, cut a
   branch, advance `main` past it, then call `changed_paths(main_tip, branch)`.
   **RED before:** raises `base must be an ancestor of head`. **After:** returns
   exactly the branch's own paths, and *not* the paths main added.
2. `test_trailers_validate_from_the_merge_base_when_main_moved` — same shape
   against `validate_range()`. **RED before:** raises. **After:** validates the
   branch's commits, and a bad trailer among them is still reported — proving
   the range change did not disarm the contract.
3. Existing `test_missing_and_nonancestor_objects_fail_closed` (pr-size) is
   **kept unchanged**: it uses an `--orphan` branch, so there is no merge base
   and it must still raise.
4. `test_missing_unreachable_and_non_ancestor_revisions_fail_closed` (trailers)
   is **split**. Its missing-revision half is unchanged. Its divergent-branch
   half moves to a new unrelated-histories case, because the divergent pair it
   built *shares* a merge base and is the situation §3 says must now work.

## 6. Gates

- Both suites green, with cases 1 and 2 shown RED on the unmodified checkers.
- `check-pr-size.py --base <base> --head <head>` classifies this PR's own diff.
- `check-commit-trailers.py` validates this PR's own range.
- Both are `governance_checker` paths, so this PR must itself carry executable
  mutation evidence in both recognized test files.
- `scripts/agent-preflight.sh`, and `pytest tests/scripts/` with
  `test_cpu_kernel_bench.py` ignored (it needs a built benchmark binary and
  fails collection on main).

## 7. Evidence

Recorded on completion: RED output from both checkers before the change, GREEN
after, the orphan case still raising, and each checker's verdict on this PR's
own range.

## 8. Stop conditions

- Stop if the orphan/unrelated-histories case stops raising. That is the half
  that must not move.
- Stop if a trailer defect inside the new range goes unreported — the range may
  change, the contract may not.
- Stop if either checker cannot classify or validate its own diff after the
  change.
- Stop if making the range correct requires touching any rule in §1's
  out-of-scope list.
