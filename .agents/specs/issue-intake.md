# The issue index becomes an append-only file, and ownership becomes a gate

Issue: [#840](https://github.com/mudler/vllm.cpp/issues/840)
Row: `POLICY-ISSUE-INTAKE`

Three changes land together because they are one loop seen at three points. An
issue is filed. A record is written about it. Nothing checks that anyone owns
it. None of this touches product code.

## Why

### The intake table is the lock its own policy forbids

`AGENTS.md` says "Do not create a surface that every pull request must write. If
N concurrent pull requests edit file F, that file is a lock." The `## Open
issues` table lives inside `.agents/roadmap_v1.md`. That file is touched by 51
of the last 60 commits.

The table is not the only writer of that file, and this spec does not claim it
is. Over the last 40 commits touching `roadmap_v1.md`, issue rows are 64 of the
added lines and row lifecycle edits are the other 103. Moving the table removes
about a third of the contention. The lifecycle half is a keyed record and stays
exactly where it is.

### Appending is not the problem, merging two appends is

The obvious objection to moving the table is that pull requests would keep
appending to the end of the new file. That objection is correct about the
behavior and wrong about the cost. The measurement below shows why. Two branches
that each append one row conflict under the default merge and merge cleanly
under `merge=union`.

A `.gitattributes` entry binds a path and never a section. This is the whole
reason the table must move. `roadmap_v1.md` also holds keyed lifecycle records,
and union merge would silently duplicate those. `AGENTS.md` names that failure:
"Never accept an automatic three-way merge of a keyed record." Union merge is
safe for the index and unsafe for the roadmap, so the two cannot share a path.

### Filing an issue defers the fix, and no gate sees it

`AGENTS.md` says "Filing the issue does not defer the fix. File it, fix it in
the same flow, reference it in the commit, and close it." Of 116 issues opened
on 13 and 14 August 2026, 88 are still open. The intake table holds 185 rows and
33 of them name no owning row.

`check_issue_table` in `scripts/check-agent-record.py` validates the form of the
rows that are present. It never requires an owner. It never requires a row to be
added at all. The rule is prose only, so the practice drifted to filing without
fixing.

### A two-line record gets a full branch

`AGENTS.md` says "Do every unit of work in its own linked worktree and task
branch. This rule includes a feature, fix, policy, document, record, and
one-line gate repair." That enumeration makes a record edit a unit of work by
itself. Five of the last 60 merged pull requests change fewer than 50 lines, and
the smallest changes two lines in one file. Each one still costs a branch, a
pull request, a gate run, and a fresh review.

## What is actually there, measured before touching anything

### The merge behavior, tested rather than assumed

Two scratch repositories, one row appended at the end of the file on each of two
branches, then merged.

| Configuration | Result |
|---|---|
| No merge driver | `CONFLICT (content): Merge conflict in index.md` |
| `index.md merge=union` | Clean merge, both rows present |

The test covers a local `git merge`, which is the path this project uses. An
agent fetches, merges `origin/main` into the branch, reruns the gate, and
pushes. Whether the GitHub mergeability check honors the driver is untested. The
pull request page may still report a conflict until the agent merges locally.
This spec claims nothing about the forge behavior.

### The table is already append-only in practice

Over the last 40 commits touching `roadmap_v1.md`, 64 issue rows were added and
8 were removed. Rows are also annotated in place. Issue #824 carries the text
`FIXED IN FLOW` inside its row. Both the removals and the annotations are edits
to an existing row, and both are what stop the file from being union-mergeable.

### The heading already misdescribes the content

The section is called `## Open issues` and it contains closed ones. It is an
index of issue to owner, not a queue of open work.

### The mapping is worth keeping, and cannot yet be derived

152 of the 185 rows name an owning row. That makes the file the network-free
answer to "who owns this issue", which is what lets the ownership gate run
without querying GitHub.

`AGENTS.md` allows a third record shape, "a value that is derived at read time
and is not stored". Deriving this index from the specs is the better end state
and is not reachable today. 148 of 304 specs cite an issue. Of 15 sampled rows
that name an owner, 3 do not cite their issue in the owning spec. Deriving now
would silently drop about a fifth of the mapping.

### No gate in this repository queries the network

`scripts/check-release-workflow.py` is the only checker that does. The
`check_issue_table` docstring states the reason: "Querying GitHub would make
this gate fail on connectivity, which is exactly the class of flake this
protocol exists to remove." The ownership gate in this spec therefore reads the
index and the specs, never GitHub.

## Design

### 1. Move the index and give it a merge driver

Create `.agents/issue-index.md`. It holds a frozen preamble and the 185 rows,
moved verbatim. Add to `.gitattributes`:

```
.agents/issue-index.md merge=union
```

Replace the `## Open issues` section in `.agents/roadmap_v1.md` with a short
pointer to the new file. `AGENTS.md` keeps all three link sites. The index, the
row spec, and the pull request body still have to agree.

The file becomes append-only by rule. A row is appended at the end. A row is
never edited and never deleted. GitHub holds the open and closed state, so
closing an issue costs no edit at all. The `FIXED IN FLOW` style annotation
stops being written.

Union merge has two hazards and both are gated rather than argued away.

Two branches that append the same issue number produce a duplicate row after a
union merge. The existing duplicate check in `check_issue_table` already catches
that, so it is kept and becomes load bearing rather than incidental.

An edited preamble line is duplicated by a union merge in the same way. The
checker therefore asserts the preamble matches a literal held in the checker.
The preamble cannot drift without a deliberate change to both.

### 2. Narrow the unit of work

Rewrite the paragraph at `AGENTS.md` §*Work happens in a worktree*. A unit of
work becomes one issue's change together with the records that change
invalidates. A record edit rides in the pull request whose change made the
record stale.

A record-only pull request stays correct when the record is the work. A stale
row, a corrected pin, and a newly filed gap are all the work. A record that only
restates what just landed is not, and it belongs in the pull request that landed
it.

The worktree rule itself does not change. Every unit of work still gets its own
worktree and task branch.

### 3. Gate ownership without a network call

Extend `scripts/check-agent-record.py`, which `scripts/agent-preflight.sh` and
`.github/workflows/ci.yml` already run. No new registration point is needed.

A row is owned when either condition holds. Column two names a row ID in
backticks. Or the issue number appears under an `## Owed` heading in any
`.agents/specs/*.md`, read with a glob.

The checker counts rows that satisfy neither. It compares that count to a
high-water mark held as a literal in the checker, starting at 33. The count may
never exceed the mark. When the count drops, the mark must be lowered in the
same change, which is the ratchet `ENG-RECORD-ANCHOR-RATCHET` already
establishes. A new row that names no owner therefore fails the gate, because it
pushes the count above the mark.

Append-only needs a base to compare against, so it is a separate range check
next to the existing `now-current range` invocation in `agent-preflight.sh`. The
diff of `.agents/issue-index.md` against the merge base must contain no removed
row and no modified row. When no base ref is resolvable the check reports
`SKIP` and says so, rather than passing silently.

## Scope

In scope:

- `.agents/issue-index.md`, new, holding the moved table.
- `.gitattributes`, one added line.
- `.agents/roadmap_v1.md`, the section replaced by a pointer.
- `AGENTS.md`, the unit-of-work paragraph and the three-link sentence.
- `scripts/check-agent-record.py`, retargeted path, ownership count, preamble
  assertion, append-only range mode.
- `tests/scripts/test_check_agent_record.py`, the red-before cases.

Out of scope and stated as owed:

- Deriving the index from specs and deleting the stored file. It needs the
  backfill measured above. It gets its own issue and is not started here.
- The 88 currently open issues. This change stops the count from growing and
  does not retire the backlog.
- The lifecycle half of `roadmap_v1.md`, which stays a keyed record.

## Gates

Every gate is red before and green after.

| Gate | Red before | Green after |
|---|---|---|
| Duplicate row after union merge | Two branches append `#900`, merge, checker fires | Checker reports the duplicate |
| Preamble drift | Edit one preamble line, checker fires | Checker names the drifted line |
| Unowned row | Append a row with `—`, count reaches 34, checker fires | Checker names the row and the mark |
| Ratchet not lowered | Own a row, count drops to 32, mark still 33, checker fires | Checker demands the lower mark |
| Row removed | Delete a row, range check fires | Range check names the removed row |
| Row edited | Annotate a row, range check fires | Range check names the edited row |
| No base ref | Run with no merge base | Reports `SKIP`, never `PASS` |

Full gate before the push: `scripts/agent-preflight.sh --staged` and
`python3 scripts/agent-integration.py --base origin/main`.

The union merge behavior is re-tested in the repository itself rather than in a
scratch directory, because `.gitattributes` has to be in the tree for the driver
to bind.

## Risks

**The forge may not honor the driver.** Untested and stated as such. If the pull
request page reports a conflict the agent still resolves it with a local merge,
which is the documented path. The cost is the status indicator, not the merge.

**Union merge is a silent operation.** It never reports that it combined two
sides. The duplicate check is the only thing that catches a bad combination, so
that check must stay in CI and not only in preflight.

**The high-water mark can be raised instead of respected.** A mark is a number
in a file and an agent can edit it. The mark is a literal in the checker rather
than in the data file, so raising it is a checker semantic change. `AGENTS.md`
already requires a spec and a red-before test for that.

**Narrowing the unit of work can be read as permission to bundle.** It is not. A
record rides along only when the pull request it rides in caused the record to
go stale. Unrelated records still need their own unit.

## Stop conditions

Stop and return `NEEDS_DECISION` if the union driver fails to bind in the real
repository, because the split then buys nothing and the table should stay in
`roadmap_v1.md`.

Stop if the ownership definition cannot reach 33 on the current tree. A mark
that does not match the measured count means the definition is wrong, not that
the mark needs adjusting.

## Outcome

**What was measured.** The union driver binds in this repository, not only in a
scratch directory. Two branches each appended one row to the real index and
merged with both rows present and no conflict. The same shape conflicts without
the driver.

**What the mutations found.** Nine guarantees were mutated and all nine fire:
preamble drift, a deleted preamble line, a duplicate row, a row whose number and
URL disagree, a row that lost its link, an extra unowned row, a ratchet that did
not fall, an empty index, and both range cases for a deleted and an edited row.

One mutation exposed a real defect rather than confirming a guarantee. The first
version of the row loop skipped any line not starting with `| [#`, so a row that
lost its link was invisible instead of malformed. The pre-existing
`test_a_bare_issue_number_without_a_link_is_rejected` case caught it. The loop
now matches any table line that is not the header or the separator.

**What was rejected.** Deleting the index outright. 152 of its 185 rows name an
owning row, which is what lets the ownership gate run without a network call.
Deriving the index from specs was rejected for the same round, because 148 of
304 specs cite an issue and 3 of 15 sampled rows are absent from their owning
spec. Deriving would have dropped about a fifth of the mapping silently.

**Why the mark lives in the checker.** A number in the data file is an ordinary
edit. A number in the checker is a checker semantic change, which `AGENTS.md`
already gates behind a spec and a red-before test. The cost is that every
legitimate ratchet-down edits the checker, and that cost is the point.

**Why `high_water` is injectable.** Without it every small fixture is red for
having the wrong number of unowned rows, which would hide whatever the fixture
is actually about. That is the mute-switch shape this protocol exists to avoid.

**A vacuous pass, found and understood.** The range check reported OK after a
committed row was deleted on this branch. It is not a mute switch. The index is
added in this range, so its net diff has no removals at all. Re-run against a
base where the file exists, both the deleted-row and the edited-row cases fail
as intended. The committed suite builds real repositories for exactly this
reason.

**What this does not do.** The 88 issues already open stay open. This change
stops the unowned count from growing and does not retire the backlog.

## Now

Implemented and gated. Awaiting review on the pull request.
