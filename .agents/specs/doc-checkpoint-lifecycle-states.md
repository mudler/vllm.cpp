# The lifecycle gate could not see `PARTIAL`, so a third of the rows moved unobserved

Issue: [#1434](https://github.com/mudler/vllm.cpp/issues/1434)
Row: `GATE-DOC-CHECKPOINT-STATES`

`AGENTS.md` says `docs/STATUS.md`, `docs/BENCHMARKS.md` and the moved row spec's
`## Now` are owed when a row changes lifecycle state.
`scripts/check-doc-checkpoint.py` decides what a lifecycle state *is* by
matching a backticked token against a fixed tuple. `PARTIAL` was not in it, and
`PARTIAL` is the second most used state in the matrices. This row adds it,
and records why `ANCHOR-BACKFILL` is deliberately left out.

## The defect

`scripts/check-doc-checkpoint.py:56-66` at `947e5f648`:

```python
STATES = ("TODO", "READY", "ACTIVE", "GATING", "BLOCKED", "DONE", "DROPPED", "N/A")
STATE_CELL = re.compile(r"`(" + "|".join(re.escape(s) for s in STATES) + r")`")
```

`row_states` keeps a row only when `STATE_CELL` matches somewhere on its line.
`lifecycle_moves` and `moved_rows` then iterate the AFTER map, so a row whose
destination state is unmatchable is not merely mis-labelled — it is absent, and
an absent row is never compared against its predecessor. Leaving the matched set
is therefore silent by construction.

`.agents/feature-matrix.md:19-20` names the states the project actually uses:

> **Lifecycle:** `INVENTORIED -> SPIKE -> READY -> ACTIVE -> GATING -> DONE`, with
> `PARTIAL`, `BLOCKED`, and `ANCHOR-BACKFILL` as explicit non-done states.

`BLOCKED` is in the tuple. `PARTIAL` and `ANCHOR-BACKFILL` are not.

### Re-derived counts

Measured at `947e5f648` (the issue measured `63d87805c`, and rows have moved
since; `PARTIAL` 115 -> 118, `DONE` 73 -> 77, `ANCHOR-BACKFILL` and `BLOCKED`
unchanged). Backticked state cells across `.agents/*-matrix.md`, using the
checker's own `STATE_CELL` shape:

| state | cells | in `STATES` at `947e5f648` |
|---|---:|---|
| `INVENTORIED` | 503 | no (deliberate: pre-claim) |
| `ACTIVE` | 179 | yes |
| `PARTIAL` | **118** | **no** |
| `SPIKE` | 90 | no (deliberate: pre-claim) |
| `DONE` | 77 | yes |
| `ANCHOR-BACKFILL` | **73** | **no** |
| `READY` | 38 | yes |
| `GATING` | 28 | yes |
| `BLOCKED` | 9 | yes |
| `TODO`, `DROPPED`, `N/A` | 0 | yes |

`PARTIAL` per table: `engine` 31, `model` 28, `feature` 25, `quantization` 16,
`backend` 11, `kernel` 7.

Over the seven tables the checker actually reads (`ROW_TABLES`, which excludes
`.agents/sglang-matrix.md` — see `## Owed`), the resolved-row population is
**153 rows today and 226 after adding `PARTIAL`**, a 48 % widening. Adding
`ANCHOR-BACKFILL` as well would take it to 281.

### The count is not the whole defect

Two of the transitions the issue names behave differently from its description,
and the difference matters for the test design. Measured at `947e5f648` with
scratch commits, checker unmodified:

| transition | today | why |
|---|---|---|
| `READY -> PARTIAL` | **rc 0** | the row leaves the AFTER map; nothing iterates it |
| `PARTIAL -> READY` | **rc 0** | absent from BEFORE, and `READY` is not in the new-row claim set |
| `PARTIAL -> ACTIVE` | rc 1 | absent from BEFORE, so it reports **`added as ACTIVE`** — right verdict, wrong reason, for a row that has existed for months |
| `READY -> ACTIVE` | rc 1 | control: the instrument is live |

So the issue's suggested reproduction (`PARTIAL -> ACTIVE`) already reds, by
accident, and the reproductions that hold are `READY -> PARTIAL` — which is the
`LOAD-GGUF-MMPROJ` case the issue was filed from — and `PARTIAL -> READY`.
The tests pin both, and pin the message correction on the third.

### Two rows resolve to the wrong state today

`row_states` takes the LAST state cell on a line, because earlier ones belong to
evidence prose. That heuristic silently inverts when the true state is a token
the tuple cannot see but the prose mentions one it can:

| row | State column | resolved at `947e5f648` | resolved after |
|---|---|---|---|
| `KV-BLOCK-POOL` (`engine-matrix.md:97`) | `PARTIAL` | `DONE` | `PARTIAL` |
| `KV-EXTERNAL-CACHE` (`engine-matrix.md:108`) | `ANCHOR-BACKFILL` | `ACTIVE` | `ACTIVE`, unchanged |

`KV-BLOCK-POOL`'s row says `` `PARTIAL` (not `DONE`) `` in its prose, and the
gate believed the parenthesis. It is the ONLY row in the seven tables whose
resolved state moves under this change, and the move is a repair.
`KV-EXTERNAL-CACHE` is listed because it shows the same heuristic failing for a
state this change does not admit: it stays wrong, and it stays wrong in a way
that still produces the right move/no-move verdict, because both blobs are read
with the same tuple.
## Design

One string, one comment, and a decision recorded beside it.

```python
STATES = ("TODO", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED", "DONE", "DROPPED", "N/A")
```

Nothing else changes. `STATE_CELL` is derived, `REQUIRED` is untouched, and the
new-row claim set `{ACTIVE, GATING, DONE}` is untouched (see `## Risks`).

### Ruling: `ANCHOR-BACKFILL` does not go in the same tuple

It is admitted that `.agents/feature-matrix.md:20` lists the two side by side.
The tuple is not that sentence, though; it is the trigger for
`REQUIRED["lifecycle"] = (STATUS, BENCHMARKS)`, and the question is whether each
state's moves are things `docs/STATUS.md` projects.

**`PARTIAL` is a public term.** `docs/STATUS.md:39` carries it verbatim in the
lifecycle table — `| Partial | A usable path exists with named missing behavior
or evidence |` — and the user-facing table below it uses that exact value for
speculative decoding (`:53`) and for LoRA and adapters (`:57`), with three more
surfaces described as partial in prose. A row entering or leaving `PARTIAL` is
exactly the claim that page exists to project.

**`ANCHOR-BACKFILL` is not, and by construction cannot be.**
`.agents/feature-matrix.md:14-17` defines it as a property of the RECORD, not of
the capability: *"a legacy implemented row without exact code, test and
real-spec anchors is `ANCHOR-BACKFILL`, not protocol-complete `DONE`"*. The
capability is already implemented; what is missing is the row's anchors.
`docs/STATUS.md` carries no corresponding term and would have nothing true to
write on a `DONE <-> ANCHOR-BACKFILL` move.

`REQUIRED["lifecycle"]` cannot demand the spec's `## Now` alone — the class
carries all three surfaces or none — so admitting `ANCHOR-BACKFILL` would demand
a public-document edit with nothing to say. That is precisely the failure this
file's own header records as the reason it was rewritten: the previous trigger
"produced 16 of the last 20 red CI runs, and it had accreted SIX hardcoded
exact-path-set escape hatches -- one per legitimate change it had blocked"
(`scripts/check-doc-checkpoint.py:4-17`). Re-importing that shape to catch a
record-hygiene marker is a bad trade, and a gate that over-fires gets worked
around.

The `## Now` half of the obligation is genuinely owed for
`ANCHOR-BACKFILL` moves and is genuinely not paid by this change. It is filed
rather than dropped — see `## Owed`. Splitting `REQUIRED` into a spec-only class
is a larger semantic change than #1434 describes and wants its own argument.

## Scope

In:

- `scripts/check-doc-checkpoint.py`: `PARTIAL` added to `STATES`, with a dated
  comment naming the issue, the counts, and the `ANCHOR-BACKFILL` exclusion.
- `tests/scripts/test_doc_checkpoint.py`: the red-before cases, the mutation
  that proves the addition is load-bearing, and a test that pins the ruling.
- `.agents/issue-index.md`: one appended row for #1434.
- this spec.

Out:

- `ANCHOR-BACKFILL`, `INVENTORIED`, `SPIKE` (each argued above or below).
- the new-row claim set `{ACTIVE, GATING, DONE}`.
- `.agents/sglang-matrix.md` joining `ROW_TABLES`.
- reporting a row that disappears from the AFTER map entirely.
- any change to `REQUIRED`, to the public documents, or to the matrices. No row
  moves lifecycle state in this change, so no checkpoint surface is owed by it.

## Upstream anchors

None. `check-doc-checkpoint.py` is a repository governance gate with no vLLM
counterpart, so there is no upstream behaviour to mirror. The authorities are
`AGENTS.md` `## Public documents` (the prose rule) and
`.agents/feature-matrix.md:16-20` (the state vocabulary).

## Risks

**The ratchet shape.** Widening the gate's population is the same shape as
[#1376](https://github.com/mudler/vllm.cpp/issues/1376), where filling a spec's
`## Gates` moved a row into the runnable population without re-pinning
`RUNNABLE_BASELINE`, and `main` went red on 8 of 44 tests. Checked here rather
than assumed:

- `scripts/check-gate-commands.py` has its OWN `GATED_STATES` (`:56`) and does
  not import this tuple; `RUNNABLE_BASELINE` (`:424`) is keyed on matrix rows
  linking a spec whose `## Gates` names a runnable command. This change adds no
  matrix row, so no row enters that population. Verified by running
  `--check` before and after.
- `scripts/check-agent-record.py`'s `UNOWNED_HIGH_WATER` (`:1899`) counts index
  rows naming neither an owning row ID nor a spec `## Owed` entry. The appended
  #1434 row names `GATE-DOC-CHECKPOINT-STATES`, so the count does not move.
- The record-anchor ratchet (`scripts/record-anchor-baseline.json`) counts
  citations on matrix rows. No matrix row changes.
- `docs/STATUS.md`'s prose ratchet is not touched: no public document changes.

No counter is re-pinned by this change, and that is a measured result rather
than an omission.

**The new-row claim set stays `{ACTIVE, GATING, DONE}`.** A row absent from the
BEFORE map and present as `PARTIAL` in the AFTER map is still not a move. That
is deliberate. The dominant real cause of a row being absent from BEFORE in this
repository is a record RELOCATION between matrices, not a new capability, and
widening the claim set would red legitimate record moves for 118 more rows while
fixing no reported defect. Named under `## Owed`.

**A phantom move from prose.** A row whose evidence prose contains a trailing
`` `PARTIAL` `` after its State column now resolves to `PARTIAL`, so editing
that prose could read as a move. Two rows resolve differently after this change
and both are audited above; one is a repair and the other is unchanged. This
hazard is pre-existing, generic to the last-match heuristic, and not widened in
kind.

**Retroactive reds.** Running the widened checker over historical commits can
report moves that were green when they landed. CI runs it diff-scoped over the
pull-request range only (`.github/workflows/ci.yml:540`), and this branch's
commits touch no row table, so nothing in this change is retroactively gated.

## Tests

`tests/scripts/test_doc_checkpoint.py`, which `scripts/check-pr-size.py:302`
already binds to this checker.

1. `test_a_move_into_partial_is_a_lifecycle_move` — `READY -> PARTIAL` demands
   `STATUS`, `BENCHMARKS` and the spec. RED before.
2. `test_a_move_out_of_partial_is_a_lifecycle_move` — `PARTIAL -> READY`, same.
   RED before.
3. `test_a_move_out_of_partial_names_the_transition_not_a_new_row` —
   `PARTIAL -> ACTIVE` must report the transition, not `added as ACTIVE`. RED
   before on the message, green on the verdict.
4. `test_removing_partial_from_states_restores_the_blind_spot` — MUTATION.
   Rebuilds `STATE_CELL` without `PARTIAL` and proves case 1 goes green, so the
   addition is what makes the tests fire.
5. `test_anchor_backfill_is_deliberately_excluded` — pins the ruling, so
   admitting it later is a deliberate edit with a reason, not a drift.
6. `test_partial_is_a_public_status_term` — pins the ruling's premise against
   `docs/STATUS.md` itself, so the argument stays executable.
7. `test_prose_partial_does_not_beat_the_state_cell` — the last-match heuristic
   still holds for the new token.

## Gates

```sh
python3 -m pytest -q tests/scripts/test_doc_checkpoint.py
python3 -m pytest -q tests/scripts/test_agent_record.py tests/scripts/test_check_gate_commands.py
python3 scripts/check-gate-commands.py --check
python3 scripts/check-agent-record.py
scripts/agent-preflight.sh --staged
```

No C++ build is involved: nothing under `src/`, `include/` or `tests/*.cpp`
changes.

## Evidence

- Red-before scratch-commit table under `## The defect`, each run printing the
  checker's real exit code and `git diff --stat`, with the tree restored against
  a pre-taken sha256 after every mutation.
- The green-after rerun of the same four transitions.
- Before/after `--check` output for `scripts/check-gate-commands.py`.

## Stop conditions

- Stop and return `NEEDS_DECISION` if the `ANCHOR-BACKFILL` ruling is rejected:
  admitting it means either accepting `STATUS`/`BENCHMARKS` edits with nothing
  to say, or splitting `REQUIRED["lifecycle"]`, and the second is a different
  row.
- Stop if adding `PARTIAL` moves any pinned counter. It does not, and if a later
  measurement says otherwise the re-pin belongs in the same change with its
  reason.
- Stop if the widening reds a commit already on `main` inside this branch's
  range.

## Owed

- [#1434](https://github.com/mudler/vllm.cpp/issues/1434) is closed by this
  change for `PARTIAL` only. The remainder is listed here so it is visible debt:
  - `ANCHOR-BACKFILL` moves still owe their spec's `## Now` and no gate observes
    them. Paying it needs `REQUIRED` to carry a spec-only class.
  - `.agents/sglang-matrix.md` is a keyed row table with State cells and is
    absent from `ROW_TABLES`, so none of its rows are observed at all.
  - A row that leaves the matched set entirely — `READY -> INVENTORIED`,
    `READY -> ANCHOR-BACKFILL` — is still silent, because `moved_rows` iterates
    the AFTER map. Reporting disappearances would red legitimate relocations
    between matrices and needs its own design.
  - A new row added directly as `PARTIAL` is still not a claim.

## Now

Landed on `row/RECORDS-LIFECYCLE-STATES-1434`. Next: fresh scoped review of the
immutable head, mutating case 4 above and confirming no pinned counter moved.
