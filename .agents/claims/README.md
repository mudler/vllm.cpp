# Active claims — one file per claim

A claim lives in its own file here, named for its claim ID:
`.agents/claims/CLAIM-<SOMETHING>.md`. `scripts/check-agent-record.py` reads
every `CLAIM-*.md` in this directory, and it reads the legacy table in
[`../coordination.md`](../coordination.md) too, so both are equally valid to the
gate. **New claims go here.**

## Why

The claims TABLE in `coordination.md` is insert-at-one-anchor: every concurrent
claim appends a row at the same line, so every concurrent claim collides. It was
the single largest conflict source in the repository — 8 of the 16 conflicting
open PRs at `origin/main` `d928e2c3`, including six PRs from ONE author's
sequential ROCm GDN stack (#334 #336 #341 #343 #345 #348) whose *only* conflict
was this file. Their work did not overlap at all.

A claim in its own file has exactly one writer, so it cannot collide with
another claim. That is the same shape `.agents/specs/` already has, which is why
specs took **zero** conflicts across the whole measured sample. See `AGENTS.md`
§"Records" — *no surface that every PR must write* — and
[`../specs/retire-shared-record-surfaces.md`](../specs/retire-shared-record-surfaces.md)
(issue [#364](https://github.com/mudler/vllm.cpp/issues/364)).

Existing rows are deliberately **not** migrated. They stay in `coordination.md`
and are removed as their claims close, so no record is rewritten in bulk and
nothing can be lost in the move. The table shrinks to nothing on its own.

## Shape

One markdown table row, in a file of its own. The columns are the ones
`coordination.md` already uses, and the checker requires the same two things it
always did: a `CLAIM-*` ID in the first cell, and at least one stable row ID in
the second, each of which must name a real row in `SPIKE` or `ACTIVE` state.

```markdown
# CLAIM-EXAMPLE-W1

| Claim | Row IDs | Agent | Worktree / remote dir | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-EXAMPLE-W1` | `ENG-EXAMPLE` | who | where | `row/ENG-EXAMPLE` | what it owns and excludes | `ACTIVE` | date — what changed |
```

Delete the file when the claim closes. Git keeps the history; that is the point.
