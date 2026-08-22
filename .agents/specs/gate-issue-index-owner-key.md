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

| Line | Added by | Owning row | What it records |
|---|---|---|---|
| `:592` | `a7bb3130b` | `ENG-HF-MODEL-DOWNLOAD` | the bug, and the attribution to `a50c57d69` that introduced it |
| `:632` | `2f2a70925` | `GATE-WINDOWS-PORTABILITY-TARGET-SCOPE` | the fix, plus a second red that `:592` did not record |

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

Each command is run on the real tree. The BEFORE column is measured at
`038ff61e5`. The AFTER column is filled by the implementation commit, which is
why it names the command rather than a number here.

| Gate | Before | After |
|---|---|---|
| `python3 -m unittest tests.scripts.test_agent_record` | 109 tests, 1 failure (`test_the_tracked_index_is_valid`) | every case green, four cases more |
| `python3 scripts/check-agent-record.py` | rc 1, `issue #1649 listed twice` | rc 0 |
| `python3 scripts/check-issue-index-append-only.py --base origin/main` | `OK` | `OK`, so the preamble insert and the appended row remove no line |
| `scripts/agent-preflight.sh` | not applicable at the base | all gates green |

`git merge-base --is-ancestor origin/main HEAD` must exit 0 before any preflight
trailer result is read. `scripts/agent-preflight.sh:452-459` takes a
`TRAILER_BEHIND` arm when the head is behind the base and then reports nothing,
so a green there would mean "not checked".

MUTATION, recorded in the pull request body with the `git diff --stat` that
proves each mutation applied and an empty `git status --porcelain` after
restoring: forcing the key back to the number alone must red exactly the cases
the narrowing is for, and forcing the key to the owner alone must red the
duplicate cases. A mutation that fails to run reads as a passing test, so the
interpreter output is quoted for each one.

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

The narrowed key and its cases are written. `main` is green on
`check-agent-record.py` without any row being edited or deleted, and the two
`#1649` rows both stand, each under the row that wrote it.

## 10. Outcome

Recorded on landing.
