# GATE-AUDIT-BRANCH-EVIDENCE — the audit must not read "no branch information" as "no branch work"

**Row:** `GATE-AUDIT-BRANCH-EVIDENCE`
**Issue:** [#726](https://github.com/mudler/vllm.cpp/issues/726)
**Base:** `origin/main` `af026e524`
**Status:** ACTIVE, 2026-08-14

## 1. Scope

`scripts/audit-live-rows.py` classifies every `ACTIVE` row as `LANDED`,
`IN-FLIGHT` or `ABANDONED`. In CI the `IN-FLIGHT` arm is **unreachable**,
because the `agent-record` job fetches `main` and nothing else, so no ref named
`row/<ID>` exists in the checkout. A row whose work is genuinely in flight and a
row nobody is working on produce the same verdict.

**In scope.**

- One new evidence source in `scripts/audit-live-rows.py`: commits reachable
  from `HEAD` but not from `origin/main` whose message names the row ID.
- One new precondition in the same file, `require_branch_information()`,
  mirroring `require_origin_main()`: refuse to classify when the checkout holds
  no `row/*` ref at all.
- One extra refspec on the `agent-record` job's existing `git fetch`, so the
  branch evidence the classifier was built around is actually present in CI.
- The cases in `tests/scripts/test_audit_live_rows.py` that pin all three.

**Out of scope.** The `PARTIAL` gap heuristic, the duplicate-ID report, the
matrix set, `CHECK_FAILS_ON`, and every other job in `.github/workflows/ci.yml`.
No row's lifecycle state changes in this row: the audit "PROPOSES and REPORTS"
by design, and a state transition carries obligations only a reader of that row
can satisfy (`scripts/audit-live-rows.py:7-10`). In particular
`ENG-FORGE-COAUTHOR`, the row `main` is red on today, is **not** reconciled
here; see §7.

## 2. Anchors

Local, not upstream — this is project governance machinery with no vLLM
counterpart. `AGENTS.md` §"Changing the rules or a checker" governs it.

| What | Where |
|---|---|
| The classifier | `scripts/audit-live-rows.py:154-179` (`classify_active`) |
| The branch evidence source | `scripts/audit-live-rows.py:90-103` (`row_branches`) |
| The guard this one mirrors | `scripts/audit-live-rows.py:128-142` (`require_origin_main`) |
| The principle it states | *"Absence of work and absence of information must never look the same."* |
| The CI step that starves it | `.github/workflows/ci.yml:316` |
| The reported instance | [#726](https://github.com/mudler/vllm.cpp/issues/726), found landing [#431](https://github.com/mudler/vllm.cpp/pull/431) |

## 3. Design

### 3.1 `HEAD` is an evidence source

```python
def head_commits(item_id: str) -> list[str]:
    out = git("log", "--oneline", "-E", f"--grep={id_grep_pattern(item_id)}",
              "-n", "20", "origin/main..HEAD")
```

Same anchored, whole-token grep as `main_commits`, over `origin/main..HEAD`
instead of `origin/main`. It reads the commit *body*, not just the subject,
because a row ID frequently appears only there — `25df7468f` is the shipped
example, and a `--oneline`-then-filter shortcut would miss it.

`HEAD` is what makes the **fork** case reachable at all. A contributor's branch
lives on their fork; `origin` in the runner's checkout is the base repository,
so no refspec can ever fetch it. On a pull request `HEAD` is the merge commit
GitHub builds, which already contains that contributor's commits. This is the
first fix the issue proposes and it costs no extra fetch.

Placed in `classify_active` **after** the branch `IN-FLIGHT` arm and **before**
both `LANDED` arms, so the existing rule — *IN-FLIGHT wins over LANDED whenever
both are present* — extends to it unchanged. A row with landed groundwork and a
follow-up commit in this PR is in flight, not finished.

### 3.2 The missing guard

`require_origin_main` exists because `git()` maps every failure to `""`, which
downstream is indistinguishable from "no evidence". Exactly the same is true of
`row_branches()`: an empty mapping means either "nobody has a branch" or "this
checkout was never told about branches", and the classifier reads both as the
former. `require_branch_information()` closes that, mirroring the existing
guard's shape: abort with a message naming the refspec to fetch.

A precondition, not a result-dependent check, because the degradation is not
confined to `ABANDONED`. With no branch refs a row that is `IN-FLIGHT` *and* has
landed groundwork silently reports `LANDED` — a live claim reported as finished,
the exact false negative `classify_active`'s own comment names. The whole census
is untrustworthy, not just its abandoned rows.

The condition is "at least one ref named `row/*` exists (local or remote)".
`HEAD` deliberately does **not** satisfy it: `HEAD` carries information about
the row this PR advances and no other, so counting it would make the guard
vacuous while every other row stayed silently misclassified.

### 3.3 The refspec

```yaml
git fetch -q origin +refs/heads/main:refs/remotes/origin/main \
                    '+refs/heads/row/*:refs/remotes/origin/row/*'
```

Without this the new guard aborts on every lane, which is honest but useless.
With it, the runner's checkout holds the same branch evidence a developer's does
— which is what `row_branches()` was written to read, and which is the only
source that covers a row being worked on in a *different* PR. Several
coordinators run at once here, so that is the common case, not a corner.

Measured cost: `origin` carries 267 `row/*` heads holding **648** commits not on
`origin/main` (`git rev-list --count --stdin --not origin/main`). The job
already checks out at `fetch-depth: 0`, so main's history is present and the
fetch negotiates down to those 648 commits' objects.

### 3.4 What this deliberately does not fix

`main` is squash-only, so a merged branch keeps commits that are not ancestors
of `origin/main` until someone deletes it. `unmerged(branch)` therefore reports
a landed-and-undeleted branch as live work, and the row reads `IN-FLIGHT`
forever. That weakness is **pre-existing and unchanged here** — it is why a
developer's local run has always disagreed with CI in the *other* direction —
but §3.3 makes CI inherit it. It is recorded as owed rather than silently
absorbed: [#788](https://github.com/mudler/vllm.cpp/issues/788).

## 4. Risks / decisions

| Risk | Decision |
|---|---|
| The guard aborts in a fresh clone that has only fetched `main` | Accepted, and identical to `require_origin_main`'s existing behaviour. The message names the exact refspec. An abort is the loud alternative to a wrong census. |
| Fetching 267 refs slows the job | Accepted: 648 commits over a `fetch-depth: 0` checkout. Rejected `--filter=blob:none`, which would convert the whole job's checkout into a promisor repo to save objects that are mostly deltas of `main`. |
| `HEAD` widens `IN-FLIGHT` too far | It does not: the match is the same anchored whole-token ID grep, restricted to `origin/main..HEAD`. A PR that mentions no row ID rescues no row. Pinned by a test that an unrelated PR leaves the row `ABANDONED`. |
| Turning a red gate green by widening scope | The scope that widens is *evidence*, not the failing condition. `CHECK_FAILS_ON` is unchanged and is asserted unchanged; the `ABANDONED` verdict and its exit code are unchanged; the no-evidence-at-all case still fails. |

## 5. Tests

`tests/scripts/test_audit_live_rows.py`, all executable and all RED at BASE:

1. `classify_active` returns `IN-FLIGHT` on HEAD commits alone.
2. HEAD `IN-FLIGHT` beats a `LANDED` main commit, and beats a fully merged
   branch.
3. HEAD evidence naming a *different* row leaves this row `ABANDONED`.
4. `head_commits` greps `origin/main..HEAD`, anchored on ID boundaries, and is
   not a prefix match.
5. `require_branch_information` raises when no `row/*` ref exists, and the
   message names the refspec.
6. `audit()` runs the guard **before** it reads any row, like the origin/main
   guard.
7. The `agent-record` step fetches the `row/*` refspec.

## 6. Gates

- `python3 tests/scripts/test_audit_live_rows.py`
- `python3 -m pytest tests/scripts/ --ignore=tests/scripts/test_cpu_kernel_bench.py`
- `scripts/agent-preflight.sh`
- An end-to-end reproduction in a checkout built to CI's shape — `origin/main`
  and no `row/*` ref — RED before, guarded after; and the same checkout with the
  refspec fetched, green.

CPU-only. No GPU, no oracle, no model artifacts: nothing here executes a model,
so runtime/performance/parity axes are `VOID`.

### 6.1 Evidence

RED at `af026e524` with the spec committed and the tests added: 26 of 57 cases
fail, 25 as `TypeError`/`AttributeError` on the absent parameter and the two
absent functions, 1 as the workflow assertion. GREEN after: `Ran 57 tests … OK`.

End-to-end, in a checkout built to the `agent-record` job's shape — a tree
holding only `scripts/`, `.agents/`, `.github/` and `tests/scripts/`, a git dir
whose objects are the real repository's, and exactly the refs the job would
have:

| # | Refs present | HEAD | Verdict for `ENG-FORGE-COAUTHOR` | Exit |
|---|---|---|---|---|
| A | `origin/main` only — today's job | `= origin/main` | **before:** `ABANDONED`, "no branch, no commit on main mentioning the row ID" / **after:** aborts, naming the refspec | 1 / 1 |
| B | `origin/main` + 270 `origin/row/*` — the job after §3.3 | `= origin/main` | `IN-FLIGHT`, `unmerged commits on origin/row/ENG-FORGE-COAUTHOR` | 0 |
| C | as B, minus `origin/row/ENG-FORGE-COAUTHOR` — the **fork** shape | `refs/pull/N/merge` over a commit naming the row | `IN-FLIGHT`, `unmerged commits on HEAD: 8f2c940e0 feat(ENG-FORGE-COAUTHOR): …` | 0 |
| D | as C — **control** | same merge shape over `docs(readme): fix a typo` | `ABANDONED` | **1** |

A and D are what make the rest mean something. A shows the failure this row
exists to fix, and that it is now loud instead of wrong. D holds every variable
of C fixed except whether a commit names the row, and the gate still fails —
so the fix widened the *evidence*, not the failing condition.

C's merge subject is `Merge <sha> into <sha>`, which is what GitHub actually
writes for `refs/pull/N/merge`; it names no row, so the verdict has to come from
the contributor's own commit, as it would on a real fork PR.

## 7. `ENG-FORGE-COAUTHOR`, and why it is not fixed here

`main` at `af026e524` reports `1 abandoned ACTIVE`, and reproducing #726's CI
shape shows it is `ENG-FORGE-COAUTHOR` (`.agents/engine-matrix.md:219`), not the
`MODEL-MUSIC-…` row named when this work was scoped — that one now reads
`LANDED` on `25df7468f`.

It is a genuine record defect and the audit is right to fire: the row's own spec
says *"`ACTIVE`; W1-W3 implemented. Next: land, then close the row"*, the work
landed on `2026-08-11` as `902b0e394` (#419), and issue #418 is still open. The
verdict's *wording* is inaccurate — the commit exists, it just does not name the
row ID as a token — but its conclusion is correct.

**Filed separately as [#787](https://github.com/mudler/vllm.cpp/issues/787),
not fixed in flow**, and that is the deliberate choice.
Closing it is a lifecycle transition: `DONE` in the matrix, an `## Outcome`
section on its spec, `docs/STATUS.md`, `docs/BENCHMARKS.md`, the spec's `## Now`,
and closing #418. `AGENTS.md` admits an in-flow repair for "the small and
obvious"; this is neither, and the audit's own header says a correction "carries
contract obligations … that only a reader of the row can satisfy". Bundling it
would also mean a checker change and an unrelated row's closure arriving in one
reviewable unit.

## 8. Stop conditions

Return `NEEDS_DECISION` rather than widening if closing #726 appears to require
relaxing `CHECK_FAILS_ON`, dropping the `ABANDONED` verdict, or making
`--check` exit 0 on an unknown. None of those is this fix; all three would make
the gate stop gating.

## 9. Now

`ACTIVE`; implementation and gates in this PR.

## 10. Outcome

Pending.
