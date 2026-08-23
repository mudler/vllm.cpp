# Spec — the issue index is keyed on the issue AND its owning row

Issue: [#1731](https://github.com/mudler/vllm.cpp/issues/1731)
Row: `GATE-ISSUE-INDEX-OWNER-KEY` (unplaced record/gate defect; the tracked tree
is a checker, not a matrix row, the same placement
[`gate-windows-portability-target-scope.md`](gate-windows-portability-target-scope.md)
uses)
State: `ACTIVE`

## 1. Scope

`check_issue_index` in `scripts/check-agent-record.py` refuses a second row for
an issue number. This spec narrows the uniqueness key to the pair
`(issue number, owning row ID)`, reports both offending line numbers, and pins
that the narrowing admits a second OWNER and nothing else.

In scope: the duplicate rule in `check_issue_index`, its message, the
[`issue-index.md`](../issue-index.md) preamble sentence that tells an appending
agent which second row is legal, and the checker's copy of that preamble.

Out of scope, each with its reason in §4: the unowned ratchet's per-row
counting, validation of the owner ID against the row inventory, a bound on how
many rows one issue may accumulate, and every other rule in
`check-agent-record.py`.

## 2. The defect, measured

`python3 scripts/check-agent-record.py` exits 1 on `origin/main` at
`038ff61e5`:

```
ERROR: .agents/issue-index.md: issue #1649 listed twice. Under `merge=union` a duplicate is what two branches appending the same issue look like
```

`python3 -m unittest tests.scripts.test_agent_record` on the same tree reports
109 tests and 1 failure, `test_the_tracked_index_is_valid`, for the same
duplicate. Both are inherited by every branch that merges `main`.

The two rows are not a bad merge and not a hand edit:

| Line at `038ff61e5` | Added by | Owning row | What it records |
|---|---|---|---|
| `:592` | `a7bb3130b` | `ENG-HF-MODEL-DOWNLOAD` | the bug, and the attribution to `a50c57d69` that introduced it |
| `:632` | `2f2a70925` | `GATE-WINDOWS-PORTABILITY-TARGET-SCOPE` | the fix, plus a second red that `:592` did not record |

The line numbers are read at `038ff61e5` and move on the next append, which is why the rows are identified by their commit here.

One lane found and filed the bug while repairing something else. A different
lane fixed it and recorded its own ownership. Neither branch could see the
other, and `merge=union` combines two appends silently.

Counted directly at `038ff61e5`: 620 rows, one duplicate number (`#1649`), and
**zero** duplicate `(number, owner)` pairs. The narrowed key therefore turns
`main` green on the tree that exists, without an edit to a single row.

## 3. Why the ban is wrong, not merely inconvenient

**It is a leftover from a different record surface.** The rule arrived in
`8dd6508da` (2026-08-09), whose message says the checker "checks that the number
and its URL agree, that no issue is listed twice, and that the table is not
empty". At that commit the surface was a KEYED intake table inside
`.agents/roadmap_v1.md`, where a row could be edited in place and one row per
issue was the whole design. `51e0cb5b1` moved the table into
[`issue-index.md`](../issue-index.md), made it append-only, and gave it
`merge=union`. Uniqueness by number survived that move unexamined.

**In an append-only log, an update is an append.** The preamble forbids editing
a row and deleting one. A rule that permits one row per issue therefore permits
one statement per issue for the life of the repository. Ownership that moves,
a fix that lands under a different row, and a re-scoped issue then have no
legal way to reach the index.

**The tree already pays for this, twice, in prose.**
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md) records that `#1365` was
re-scoped in place and that its index row now under-describes its own issue:
"Both available edits are gate failures, so the reconciliation is PROSE, it
lives here, and this paragraph is it."
[`serve-request-length-guard.md`](serve-request-length-guard.md) records the
same refusal for `#1541`. Two specs carry a fact that belongs in the index
because the gate would refuse it.

**The remedy the ban forces is the operation the file forbids.**
[#1731](https://github.com/mudler/vllm.cpp/issues/1731) proposes merging the two
`#1649` rows into one, which is an edit to an append-only record.
`scripts/check-issue-index-append-only.py` refuses exactly that against the
merge base, so the ban and the append-only rule point at each other.

**This change has to append its own second row to land.**
[`issue-index.md`](../issue-index.md) already carries a `#1731` row owned by
`ENG-RECORD-CONFLICT-SURFACES`, the standing owner of record-surface semantics.
That row's spec,
[`retire-shared-record-surfaces.md`](retire-shared-record-surfaces.md), states
"No product source, kernel or gate semantic moves", and the `#1543` row it owns
states that a checker-semantics change "owes its own row, spec, red-before test
or mutation, and green-after evidence". So the fixing row is this one, and
`AGENTS.md` requires the index, the spec, and the pull request body to agree on
the owner. Recording that agreement means appending a second `#1731` row under
`GATE-ISSUE-INDEX-OWNER-KEY`. Under the old key that append is red. It is the
change's own first legitimate case.

## 4. Design

`check_issue_index` keys `seen` on the pair `(number, row_id)` and stores the
line number of the first row for each pair. A repeat reports the repeating line,
the first line, and the shared owner:

```
.agents/issue-index.md:640: issue #1649 is listed twice under the same owner
`ENG-HF-MODEL-DOWNLOAD`, first at .agents/issue-index.md:592. ...
```

The line numbers are derived at read time and stored nowhere, which is the
record shape `AGENTS.md` `## Records` admits. A line number written INTO an
append-only file would go stale on the next append; a line number computed by
the checker cannot.

Four decisions, each of which could have gone the other way:

**A dash and an owner are different keys, and both may stand.** A row that names
no owner is owned through a spec's `## Owed`. A later row that names an owning
row ID records adoption, and adoption can only be recorded by appending. The
opposite order loses information, and it is already gated: an added dashed row
raises the unowned count and reds the `UNOWNED_HIGH_WATER` ratchet.

**The unowned count stays per-row.** `AGENTS.md` states the obligation per row:
"Every index row names an owning row ID, or names a spec that lists the issue
under `## Owed`." Treating an issue as owned because SOME row names an owner
would lower the count, trip the ratchet's lower arm, and change a second rule in
a change that owes one. `UNOWNED_HIGH_WATER` is therefore untouched at 33, and
the count is unchanged by this change because the row it appends names an owner.

**No cap on rows per issue.** A cap is a shared-file budget at the scale of one
issue, which `AGENTS.md` `## Records` names as the anti-pattern: "Limit an
entry, not a shared file. A shared-file budget forces each addition to remove
another entry." The third lane to touch an issue would have to delete an earlier
row, which is the forbidden edit again.

**The owner ID is not checked against the row inventory.** It is not checked
today, and adding that check would red `main` immediately:
`GATE-WINDOWS-PORTABILITY-TARGET-SCOPE` and this row are unplaced gate rows with
no matrix entry, and both are legitimate owners.

## 5. Risks

The narrowing widens what the checker accepts, so the hole is named rather than
asserted away.

**What the gate still catches, and why that is the case it was built for.**
[#1619](https://github.com/mudler/vllm.cpp/issues/1619) measured the one
corruption this refusal has ever caught: the `merge=union` driver duplicated a
row when both sides appended before the same trailing anchor, producing 538
lines where the correct union is 537, with `#1546` BYTE-IDENTICAL at lines 523
and 533. `git merge-tree --write-tree` called it clean and
`check-issue-index-append-only.py` passed, because a duplicate is an addition
and that checker collects removals only. `check-agent-record.py` was the single
gate that refused it. A driver duplicate copies a LINE, so both copies carry the
same owner, so the pair key collides and the refusal is unchanged. The same
holds for a rebase or a copy-paste that appends one row twice.

**What the gate no longer catches.** Two rows for one issue under two DIFFERENT
owners, where the second owner was a mistake rather than a hand-off. No reading
of the file can separate that from the legitimate case, because the two are the
same bytes. The bound is that the mistake is visible: the row names a row ID, a
reader can follow it, and the owner ID is a reviewed field in a reviewed diff.

**The preamble edit carries the risk the preamble describes.** This change adds
a paragraph to the [`issue-index.md`](../issue-index.md) preamble and to
`INDEX_PREAMBLE`. Under `merge=union` two branches that edit the preamble at the
same time duplicate the lines rather than merge them. The edit inserts whole new
lines and changes no existing line, so `check-issue-index-append-only.py` sees
no removal, and the preamble drift check in `check_issue_index` reports the
duplication if a second branch edits the preamble before this lands. The
alternative, leaving the rule in the checker message alone, was rejected: an
agent reads the preamble BEFORE appending and reads the message only after
redding the gate, and the wrong repair the message costs is an edit to an
append-only record.

### The other repair landed first, and this row is now preventive

A concurrent lane took the opposite approach on branch
`row/FIX-ISSUE-INDEX-1649-DUP`: DELETE the `:592` row as an append-only
exception argued in its own commit message. **It landed while this row was
being gated**, as `6354755ba` (PR #1742), which removed the
`ENG-HF-MODEL-DOWNLOAD` `#1649` row, appended a `#1733` row and closed
[#1731](https://github.com/mudler/vllm.cpp/issues/1731).

That changes what this row is for, and the change is stated rather than glossed.
`main` is GREEN on `check-agent-record.py` by its own deletion, verified here by
running `main`'s checker against `main`'s index. This row no longer CURES a red.
It removes the CLASS: the narrowed key means the next filing-and-fixing hand-off
does not have to spend a deleted row and an argued exception, and the deletion
that landed is a per-instance cost that recurs every time two lanes touch one
issue. Nothing here reverts that deletion — the merge takes `main`'s index
wholesale and the `:592` row stays gone.

The evidence for the class is no longer hypothetical, because this branch's own
index is the first live instance. After merging `main`,
[#1731](https://github.com/mudler/vllm.cpp/issues/1731) appears twice: at `:644`
under `ENG-RECORD-CONFLICT-SURFACES`, the row that filed it, and at `:646` under
`GATE-ISSUE-INDEX-OWNER-KEY`, the row that fixed it. Duplicate `(number, owner)`
pairs: zero. `main`'s checker refuses this branch's index with `issue #1731
listed twice`; this branch's checker passes it. The hand-off the old key could
not express is the one this change had to make about itself.

Two findings below are that lane's, credited to it. Both were re-measured here
before being written down, because relaying an unverified finding is
publishing it.

**[#1733](https://github.com/mudler/vllm.cpp/issues/1733) is a duplicate of
[#1731](https://github.com/mudler/vllm.cpp/issues/1731), and its central claim
is false.** It was filed 17 minutes and 22 seconds after #1731 against the same
red (2026-08-22 21:08:35Z and 21:25:57Z) and closed `NOT_PLANNED`. Its body
states in bold that "**The repair is measured and it is NOT blocked**". It is
not, and the reason is an instrument artifact rather than a mistake in reading
the code.

**`check-issue-index-append-only.py` reads COMMITS, so an uncommitted deletion
is invisible to it.** `scripts/check-issue-index-append-only.py:50-51` diffs
`merge-base(origin/main, HEAD)..HEAD`, a commit range. #1733 measured its repair
by "removing one of the two rows in a worktree and running both checkers", which
is a WORKING-TREE removal the range cannot see. Both halves were reproduced
here:

| Deletion of `:599`, the `ENG-HF-MODEL-DOWNLOAD` `#1649` row | working tree `--numstat` | checker's own range `--numstat` | checker |
|---|---|---|---|
| uncommitted | `0 1` | `8 0` | `OK: issue index append-only`, rc 0 |
| the same bytes, committed | `0 1` | `8 1` | `FAIL: ... is append-only, and this range removes or edits lines`, rc 1 |

The committed run was made on a throwaway branch that was deleted afterwards;
this branch's head was `c93258f49` before and after it, and the index file was
byte-identical to its pre-probe hash. #1733's own body contains the mechanism
one sentence before the wrong conclusion, noting that a removal "does red it
only when the duplicate is also reachable from the base of the branch making the
repair" — which is exactly this case, since both `#1649` rows are on `main` and
therefore reachable from any branch's merge base.

## 6. Tests

`tests/scripts/test_agent_record.py`, `IssueIntakeTable`. Four cases added, one
rescoped, one already present and now passing:

| Case | Before | After |
|---|---|---|
| `test_a_second_row_under_a_different_owner_is_a_record` | RED (`listed twice`) | green |
| `test_a_dashed_row_and_an_owned_row_are_not_a_duplicate` | RED (`listed twice`) | green |
| `test_a_duplicate_under_one_owner_names_both_line_numbers` | RED (no line numbers in the message) | green |
| `test_a_byte_identical_duplicate_row_is_rejected` | green | green |
| `test_a_duplicated_issue_is_rejected` (rescoped) | green | green |
| `test_the_tracked_index_is_valid` | RED (`#1649`) | green |

`test_a_duplicated_issue_is_rejected` is rescoped rather than deleted. It
appended a duplicate `#201` row under a DASH while the first `#201` row is owned
by `BACKEND-ROCM`, so it asserted the old key by accident. It now appends the
duplicate under the SAME owner, which is the guarantee the case names. The
different-owner half it used to cover is not dropped; it becomes
`test_a_second_row_under_a_different_owner_is_a_record`, with the opposite
expectation and a stated reason.

`test_a_byte_identical_duplicate_row_is_rejected` reproduces the shape
[#1619](https://github.com/mudler/vllm.cpp/issues/1619) measured, so the one
corruption this refusal has caught in the field has a case of its own.

The three cases that pass in both directions are the guard properties. They are
proved discriminating by mutation in §7 rather than by reading.

## 7. Gates

Every number below was re-measured for this section at `66f055248`, the merge of
`origin/main` `1a1d17e53` into this branch. The section previously named ONE
"before" tree and quoted two numbers that came from two different ones, so each
column now says which tree it was taken on. Three trees are involved:

- **T-main**: `origin/main` at `1a1d17e53`. The checker, the suite, the index
  and the engine matrix all from `main`.
- **T-mixed**: this branch's suite, index and matrix with `main`'s CHECKER in
  place. The red-first tree: the new cases against the old predicate.
- **T-branch**: this branch as it stands.

| Gate | T-main | T-mixed | T-branch |
|---|---|---|---|
| `python3 tests/scripts/test_agent_record.py` | 109 tests, 1 failure | 113 tests, 5 failures | 113 tests, `OK` |
| `python3 -m unittest tests.scripts.test_agent_record.IssueIntakeTable` | 7 tests, 1 failure | 11 tests, 4 failures | 11 tests, `OK` |
| `python3 scripts/check-agent-record.py` | rc 1, `issue #1649 listed twice` | — | rc 0, `agent record OK: ENGINE=170 MODEL=377 QUANT=84 KERNEL=57 BACKEND=85 ANCHOR-ROT=37` |
| `python3 scripts/check-issue-index-append-only.py` | `OK` | — | `OK` |
| `git diff origin/main --numstat -- .agents/issue-index.md` | n/a | n/a | `8 0`, additions only |

The correction matters because the earlier "11 tests, 4 failures" was labelled
`origin/main`, where `IssueIntakeTable` has SEVEN cases and exactly one of them
fails. Eleven cases exist only once this branch's suite is present, so the row
described T-mixed under T-main's name.

T-mixed is the RED-FIRST result. Its five failures are the three cases this
change adds, the real index, and the preamble consistency case:

```
FAIL: test_a_second_row_under_a_different_owner_is_a_record
FAIL: test_a_dashed_row_and_an_owned_row_are_not_a_duplicate
FAIL: test_a_duplicate_under_one_owner_names_both_line_numbers
FAIL: test_the_tracked_index_is_valid
FAIL: test_real_index_matches_the_checkers_preamble
```

`test_real_index_matches_the_checkers_preamble` is red on T-mixed by
construction and is not a defect: the index carries the new preamble paragraph
while `main`'s `INDEX_PREAMBLE` does not, which is the drift this very check
exists to report.

The refusal each side produces, verbatim, on the same fixture — one issue, two
DIFFERENT owning rows:

```
old: .agents/issue-index.md: issue #201 listed twice. Under `merge=union` a
     duplicate is what two branches appending the same issue look like
new: (no errors)
```

and on a byte-identical row appended twice under the SAME owner, which is the
[#1619](https://github.com/mudler/vllm.cpp/issues/1619) shape:

```
old: .agents/issue-index.md: issue #201 listed twice. Under `merge=union` a
     duplicate is what two branches appending the same issue look like
new: .agents/issue-index.md:28: issue #201 is listed twice under the same owner
     `BACKEND-ROCM`, first at .agents/issue-index.md:26. ...
```

Against the REAL index rather than a fixture, `main`'s checker reports
`issue #1649 listed twice` on `main`'s own index and this branch's checker
reports nothing on this branch's index, with both `#1649` rows byte-identical to
their `main` copies (sha256 `6dfc1fbb...` at `:599` and `74fe3230...` at `:639`,
matching `main`'s `:592` and `:632`).

**`UNOWNED_HIGH_WATER` is unaffected and needs no adjustment.** It is 33 before
and after, and the branch's index has exactly 33 unowned rows, a delta of 0. The
appended row names `GATE-ISSUE-INDEX-OWNER-KEY` as its owner, so it never enters
that population.

`git merge-base --is-ancestor origin/main HEAD` exits 0 at `66f055248`.
`scripts/agent-preflight.sh:452-459` takes a `TRAILER_BEHIND` arm when the head
is behind the base and then reports NOTHING, so `BASE_SHA` and `RANGE_COUNT` are
reported beside every trailer green: a count of 0 is a vacuous pass, not a pass.

MUTATION. `scripts/check-agent-record.py` is
`7abe4aa4b3a8b0d776364207396be146e58dd9d34067b9f258c106a96e12d593` before and
after every mutation. The hashes below are of the exact bytes named in the
Mutation column, so a reader can reproduce each one with a single substitution
and check the hash. The two hashes this table previously carried could NOT be
reproduced from the mutation text beside them and were replaced rather than
re-quoted.

| Mutation | sha256 while mutated | Red |
|---|---|---|
| `first = seen.get(key)` → `first = None`: the refusal can never fire | `cf8eea17...` | 3 of 113 — `test_a_duplicated_issue_is_rejected`, `test_a_byte_identical_duplicate_row_is_rejected`, `test_a_duplicate_under_one_owner_names_both_line_numbers` |
| `key = (number, row_id)` → `key = (number, None)`: the pre-#1731 predicate | `a14213fc...` | 3 of 113 — `test_a_second_row_under_a_different_owner_is_a_record`, `test_a_dashed_row_and_an_owned_row_are_not_a_duplicate`, `test_the_tracked_index_is_valid` |
| the message drops both line numbers | `fcddaa08...` | 1 of 113 — `test_a_duplicate_under_one_owner_names_both_line_numbers` |

The two predicate mutations red DISJOINT sets, which is the property worth
having: the first proves the refusal still refuses, the second proves the
narrowing is what admits the hand-off. Neither alone would show both.

Each mutant was import-checked before its run, because a mutant that fails to
BUILD reads as a passing test. The import check itself was validated in both
directions first: it must report `IMPORT OK` on the pristine file and must exit
9 on a deliberately broken one. Its first form did neither — it reported
`AttributeError: 'NoneType' object has no attribute '__dict__'` on the PRISTINE
file, because `@dataclass` resolves `__module__` through `sys.modules` and the
module had not been registered there. Every mutant would have read as "does not
build", which is a broken instrument failing toward a code verdict.

The mutation run also left the tree MUTATED once, when the harness was killed
by a two-minute timeout part-way through the third mutation. The tree was
`fcddaa08...` and not the pristine hash. It is restored, and `git status
--porcelain` and `git diff` are both empty at `66f055248`; the restoration is
proved by hash rather than by the harness having finished.

### Re-measured after merging `6354755ba`

`main` changed under this branch while it was being gated, which disarms part of
the evidence above rather than merely dating it. Every affected number was
taken again at `0d2d69e5c`.

| Gate | `origin/main` `6354755ba` | this branch at `0d2d69e5c` |
|---|---|---|
| `main`'s checker on `main`'s index | GREEN — the duplicate NUMBER is gone, deleted | n/a |
| `main`'s checker on THIS branch's index | n/a | RED: `issue #1731 listed twice`, plus the expected preamble drift |
| this branch's checker on this branch's index | n/a | PASS |
| `python3 scripts/check-agent-record.py` | n/a | rc 0, `agent record OK: ENGINE=170 MODEL=377 QUANT=84 KERNEL=57 BACKEND=85 ANCHOR-ROT=37` |
| `python3 scripts/check-issue-index-append-only.py` | n/a | `OK`, rc 0 |
| `python3 tests/scripts/test_agent_record.py` | n/a | 113 tests, `OK` |
| index rows | 620 | 621 |
| `UNOWNED_HIGH_WATER` vs actual | n/a | 33 vs 33, delta 0 |

The row that had to be re-checked is `test_the_tracked_index_is_valid` under the
`key = (number, None)` mutation. Before the merge it red because the real index
carried two `#1649` rows; `main` deleted one, so that reason is gone. It reds
anyway, for a NEW reason: this branch's own index now carries two `#1731` rows
under two different owners. Re-measured post-merge, both predicate mutations red
the same three cases each, with the same hashes:

| Mutation | sha256 | Red, post-merge |
|---|---|---|
| `first = None` | `cf8eea17...` | 3 of 113 — unchanged |
| `key = (number, None)` | `a14213fc...` | 3 of 113 — unchanged, but `test_the_tracked_index_is_valid` now reds on `#1731` rather than on `#1649` |

Had this not been re-run, the mutation table would have kept a green that a
later commit on `main` had quietly stopped earning.

## 8. Stop conditions

Stop and ask before repairing the STALE index rows this change makes
repairable. `#1365` and `#1541` can now be reconciled by appending a row under
the owning row, and
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md) and
[`serve-request-length-guard.md`](serve-request-length-guard.md) then carry
prose that has a better home. Each is a record edit that belongs to its own
row, and doing them here would hide the checker change under a records sweep.

Stop and return `NEEDS_DECISION` rather than deleting a row if any tree state
appears where the narrowed key cannot turn `main` green. Deleting a row is the
operation the index forbids, and it needs an exception argued in its own commit
message.

## 9. Now

`main` was made green by `6354755ba`, the other repair, which DELETED one of the
two `#1649` rows. This row does not undo that and does not claim the green. What
it carries is the narrowed key, so the next hand-off between two lanes costs an
append rather than a deleted row and an argued exception. Its own `#1731` row,
sitting beside the one `ENG-RECORD-CONFLICT-SURFACES` already owns, is the first
case the old key would have refused.

## 10. Outcome

Recorded when the row reaches `DONE`. The measurements are in §7, and the two
decisions most likely to be revisited are in §4: no cap on rows per issue, and
no owner-ID existence check.
