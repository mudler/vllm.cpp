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

Everything from here to `## Stop conditions` describes W1, the wave that added
`PARTIAL`. It is left standing as the record of what that wave decided and
measured. `## W2` below states its own scope and says which of these rulings it
keeps, which it pays, and which one it overturns.

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

## W2 — the four residuals

W1 (`4c193bd55`, #1577) added `PARTIAL` and filed four residuals below. W2 closes
all four in one change, because they are one issue, one checker and one spec.
Two of them turned out to rest on premises this wave measured and falsified.

### The premise W1 got wrong, and it was the whole blocker

W1 wrote that paying the `ANCHOR-BACKFILL` obligation "needs `REQUIRED` to carry
a spec-only class". It does not. `errors_for` calls `spec_now_errors` DIRECTLY:

```python
def errors_for(paths, before, after):
    classes, reasons = classify(paths, before, after)
    errors = spec_now_errors(paths, before, after)      # never routed via REQUIRED
```

`REQUIRED` maps a CLASS to public surfaces, and `classify()` is what assigns a
class. The spec `## Now` obligation was never a member of that mapping, so the
expressiveness W1 said was missing already existed. The only reason an
`ANCHOR-BACKFILL` move was unobserved is that `row_states` DROPPED the row before
any of it ran.

So the fix is RESOLUTION, not a new `REQUIRED` class:

```python
RECORD_STATES = ("INVENTORIED", "SPIKE", "ANCHOR-BACKFILL")
LIFECYCLE_STATES = STATES + RECORD_STATES
CLAIM_ON_ARRIVAL = frozenset({"ACTIVE", "GATING", "DONE", "PARTIAL"})
```

`transitions()` replaces the duplicated loop in `moved_rows` and
`lifecycle_moves` and labels each move CLAIM or RECORD. A move is CLAIM when
both endpoints are states `docs/STATUS.md` has a term for; otherwise RECORD.
`lifecycle_moves` reports only CLAIM moves, so only those reach
`REQUIRED["lifecycle"] = (STATUS, BENCHMARKS)`. `spec_now_errors` reports both,
so every lifecycle move owes its row spec's `## Now` — which is what AGENTS.md
`## Public documents` says, without qualification by state.

This answers residual 1 as asked: `REQUIRED` **cannot** express a `## Now`-only
obligation, and it does not need to, because that obligation does not live
there. Adding a class would have been the disproportionate change; nothing was
added.

### Residual 1 — `ANCHOR-BACKFILL` owes `## Now`: **PAID**

`DONE <-> ANCHOR-BACKFILL` is now a RECORD move. It owes the row spec's `## Now`
and owes `docs/STATUS.md` and `docs/BENCHMARKS.md` nothing, so W1's ruling — that
page has no term for it and would have nothing true to write — stands unchanged
and is now enforced rather than only argued.

### Residual 3 — leaving the matched set: **PAID, without reporting deletions**

W1 recorded that `READY -> INVENTORIED` "is still silent, because `moved_rows`
iterates the AFTER map", and that reporting disappearances "would red legitimate
relocations between matrices". Both halves are addressed without touching the
AFTER-map iteration: once the destination state RESOLVES, the row never
disappears, so it is compared against its predecessor like any other. No
disappearance is reported and no relocation can red.

The genuinely remaining hole is a row ID that leaves a table entirely. It is
still unobserved, and it is now sized: over the 400 non-merge commits ending at
`e2a9e035d`, **zero** departures occurred.

### Residual 4 — a new row added as `PARTIAL`: **OVERTURNED, and admitted**

W1 excluded it and pinned the exclusion in a test. The stated reason was that
"the dominant real cause of a row being absent from the BEFORE map in this
repository is a record RELOCATION between matrices, not a new capability". That
was never measured. Replayed over the same 400 commits, 126 of which touch a row
table:

| arrivals (row absent from BEFORE in that table) | count |
|---|---:|
| genuinely new (the ID was in no `ROW_TABLES` file beforehand) | **45** |
| relocations from another matrix | **0** |
| departures of any kind | **0** |

By arrival state: `ACTIVE` 11, `INVENTORIED` 14, `READY` 12, `SPIKE` 5,
`PARTIAL` **2**, `GATING` 1. The case the exclusion protected does not occur in
400 commits, and the case it hid occurred twice. `Partial` is a `docs/STATUS.md`
term (`:39`), so a row arriving in it asserts exactly what that page projects.

Admitting it costs nothing retroactively: `5498b4aea`, the one commit carrying
`PARTIAL` arrivals, already paid `STATUS`, `BENCHMARKS` and the spec, and stays
green. The record states stay out of the arrival set for the opposite and still
sound reason — a row arriving as `INVENTORIED` has no prior position and claims
nothing.

### Residual 2 — `.agents/sglang-matrix.md` in `ROW_TABLES`: **DECLINED**

Measured, not assumed. The file has **46 keyed rows**. Its own header says it
carries "a **classification** in place of a lifecycle state", and that axis reads
`FUSED` 24, `OUT-OF-SCOPE` 8, `INVENTORIED` 6, `SGLANG-DISTINCT` 5 — written
unbackticked, so `row_states` matches none of them.

What it DOES match is three rows, and all three are phantoms quoted from the
"our implementation" column about OTHER matrices' rows:

| row | matched | what the line actually says | its own classification |
|---|---|---|---|
| `SGLANG-SCHED-OVERLAP` (`:80`) | `DONE` | `ENG-ASYNC-SCHED` `DONE` | `FUSED` |
| `SGLANG-SPEC-EAGLE` (`:102`) | `DONE` | our `SPEC-MTP` `DONE` | `FUSED` |
| `SGLANG-SPEC-DFLASH` (`:104`) | `DONE` | `SPEC-DFLASH` `DONE` | `FUSED` |

Admitting the file would therefore observe **zero of 46 rows on their own axis
and three on prose about other rows**. Rewording one of those three sentences —
say, updating `ENG-ASYNC-SCHED` to `PARTIAL` where it is quoted — would read as
`SGLANG-SCHED-OVERLAP` changing lifecycle state and would demand `STATUS`,
`BENCHMARKS` and a spec `## Now` for a move that never happened. That is a
false-positive generator, which is the failure mode this file's header records.

The decline is pinned executably rather than only argued:
`TheSglangMatrixIsNotALifecycleTable` asserts the classification vocabulary is
disjoint from `LIFECYCLE_STATES` and that every row the checker CAN resolve there
still carries its own classification cell. A real State cell added later leaves
that row without one, the test fails, and someone decides again.

### The resolution asymmetry, and why it is measured

`row_states` takes the LAST backticked state on a line, because earlier ones
belong to the evidence columns. Applying that unchanged to the record states
regresses two real rows: `MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm` and
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` both narrate "the row stays
`SPIKE`" in prose that follows State cells reading `BLOCKED` and `ACTIVE`.

So `RECORD_CELL` requires the token to OPEN a markdown cell, and a cell-anchored
record state wins outright over a later claim state. Against a column-position
proxy over every row:

| resolution | population | rows resolved | disagreeing with the proxy |
|---|---:|---:|---:|
| before this change | 226 | 226 | 12 |
| record states matched anywhere | 793 | 793 | 11 |
| record states cell-anchored (shipped) | 793 | 793 | **10** |

Exactly two previously-resolved rows change state, and both are REPAIRS:
`engine-matrix.md` `KV-EXTERNAL-CACHE` `ACTIVE -> ANCHOR-BACKFILL` (the row W1's
audit table listed as staying wrong) and `quantization-matrix.md`
`QUANT-GGUF-PRESETS` `READY -> INVENTORIED`, whose `READY` came from the prose
"split exact preset IDs before `READY`". No previously-correct row regresses.

Those two rows are also the only place this change is a NARROWING, and it is
stated rather than buried: their true states are record states, so moves
involving them now owe the spec's `## Now` instead of `STATUS` and `BENCHMARKS`.
Demanding the public pages for a row whose State cell reads `ANCHOR-BACKFILL` was
the wrong demand, so the narrowing is the repair, not a relaxation of it.

The second narrowing is the pre-claim carve-out in `spec_now_errors`: a RECORD
move on a row that links NO spec is not an error. `.agents/feature-matrix.md`
gives a pre-claim row a `CLAIM-*` rather than a spec, and **416 of the 463**
`INVENTORIED` rows link none, so the error would demand a document the protocol
says does not exist yet. It loosens nothing: before this change the move was not
observed at all. A CLAIM move on an unlinked row is still an error, and
`test_a_claim_move_on_a_row_linking_no_spec_is_still_an_error` holds that line.

### What it costs, replayed

400 non-merge commits ending at `e2a9e035d`; 126 touch a row table. Each commit
run through the W1 checker and the W2 checker on the same paths and blobs:

| verdict | commits |
|---|---:|
| red under both | 12 |
| **newly red under W2** | **1** |
| newly green under W2 | **0** |

The one newly red commit is a true positive: `ab6e65216` moved
`ENG-CUDAGRAPH-BREAK` `INVENTORIED -> READY`, a real lifecycle move that no gate
observed. Its message names the wrong spec — see `## Owed`.

### W2 counters, re-measured rather than inherited

W1 measured its counters and W2 measures them again, because adding a table to
`ROW_TABLES` would have moved a different population than adding a state does.

| counter | before | after | why |
|---|---|---|---|
| `check-gate-commands.py --check` | rc 0 | rc 0 | `GATED_STATES` (`:56`) is its own tuple and it does not import this one; `RUNNABLE_BASELINE` (`:424`) keys on matrix rows, and `GATE-DOC-CHECKPOINT-STATES` has no matrix row at all |
| `check-agent-record.py` | rc 0, `ENGINE=169 MODEL=377 QUANT=84 KERNEL=57 BACKEND=85 ANCHOR-ROT=37` | identical | no matrix row and no index row changes; #1434's index row already names this owning row, so `UNOWNED_HIGH_WATER = 33` cannot move |
| record-anchor ratchet | unchanged | unchanged | no matrix row changes |
| `docs/STATUS.md` prose ratchet | untouched | untouched | no public document changes |

No counter is re-pinned, and no row moves lifecycle state, so this change owes no
checkpoint surface.

### W2 tests

`RecordStatesAreResolvedButNotClaims`, `ANewPartialRowIsAClaim` and
`TheSglangMatrixIsNotALifecycleTable` in `tests/scripts/test_doc_checkpoint.py`,
plus the flipped `test_a_new_row_added_as_partial_is_a_claim`. Four mutations,
each rebinding every derived name so it cannot apply to nothing:

1. `RECORD_STATES` emptied AND `RECORD_CELL` rebound — the gate goes blind on
   `DONE -> ANCHOR-BACKFILL`.
2. `transitions()` forced to label every move CLAIM — the same edit starts
   demanding `docs/STATUS.md`, the outcome the ruling rejects. This is the
   mutation for the SPLIT, which resolution alone does not prove.
3. `RECORD_CELL` unanchored — the prose line resolves `SPIKE` instead of
   `BLOCKED`, which is the two-row regression above.
4. `PARTIAL` removed from `CLAIM_ON_ARRIVAL` — a new `PARTIAL` row stops being a
   claim.

## W3 — the repair: W2 loosened the gate on nine transitions

W2 landed on `main` as `ba4634204` (#1609) **without an independent review**. A
post-landing review returned FAIL. This section is the repair, on `main` rather
than a revert: W2 makes 52 transitions observable and 9 of them regress, and the
9 are one line.

### The defect

`transitions()` keyed the move class on **both** endpoints:

```python
kind = CLAIM if previous in STATES and state in STATES else RECORD
```

Before W2 a row sitting in `INVENTORIED`, `SPIKE` or `ANCHOR-BACKFILL` was
**absent from the BEFORE map**, so `previous` was `None`, the ARRIVAL rule fired
and pulled in `REQUIRED["lifecycle"] = (STATUS, BENCHMARKS)`. Resolving the
record states made `previous` resolve, which made the arrival rule unreachable
(`previous is not None`), and the both-endpoints test then sent the move down
the RECORD branch and dropped both surfaces.

**569 of the 793 resolved rows — 72% — sit in a record state**, so this is the
exit path of nearly every row in the tree.

W2's own justification for the polarity was that "the arrival rule below already
governs a row that appears as `DONE`". It cannot: the arrival rule is guarded by
`previous is None`, and resolving the record states is exactly what stopped that
being true.

### Nine transitions, red before `ba4634204`, green after

Constructed scratch commits on `503e45900`, one pair per transition, each
changing only `.agents/kernel-matrix.md` and the row's spec, with the spec's
`## Now` **paid** and `docs/STATUS.md` / `docs/BENCHMARKS.md` **withheld**. Exit
code taken from the checker directly, never through a pipe.

| transition | `e2a9e035d` (pre-W2) | `ba4634204` == `503e45900` | W3 |
|---|---|---|---|
| `INVENTORIED -> ACTIVE` | rc 1 | **rc 0** | rc 1 |
| `INVENTORIED -> GATING` | rc 1 | **rc 0** | rc 1 |
| `INVENTORIED -> DONE` | rc 1 | **rc 0** | rc 1 |
| `SPIKE -> ACTIVE` | rc 1 | **rc 0** | rc 1 |
| `SPIKE -> GATING` | rc 1 | **rc 0** | rc 1 |
| `SPIKE -> DONE` | rc 1 | **rc 0** | rc 1 |
| `ANCHOR-BACKFILL -> ACTIVE` | rc 1 | **rc 0** | rc 1 |
| `ANCHOR-BACKFILL -> GATING` | rc 1 | **rc 0** | rc 1 |
| `ANCHOR-BACKFILL -> DONE` | rc 1 | **rc 0** | rc 1 |

Pre-W2 reports `added as ACTIVE`; W3 reports `INVENTORIED -> ACTIVE`. The
verdict is restored and the message is better than either.

### The three real commits, and a correction to the report

The review named `2a976eb9f` (`ENG-CUDAGRAPH-DEDUP` `INVENTORIED -> ACTIVE`),
`678fc672c` (`ENG-RECORD-ANCHOR-RATCHET` `SPIKE -> ACTIVE`) and `7a0e6c82b`
(`MODEL-MUSIC-minimax-music3…` `SPIKE -> ACTIVE`) as "ERROR under base, OK under
W2". Re-derived: **all three are rc 0 under all three checkers**, because all
three PAID `docs/STATUS.md` and `docs/BENCHMARKS.md` — the pre-W2 gate made
them. The commits still prove the point, on the classification rather than the
exit code. Withhold those two paths from the same tree change:

| commit | pre-W2 | W2 | W3 |
|---|---|---|---|
| `2a976eb9f` | `added as ACTIVE`, demands STATUS | **no lifecycle move at all** | `INVENTORIED -> ACTIVE`, demands STATUS |
| `678fc672c` | `READY -> ACTIVE`, demands STATUS | **no lifecycle move at all** | `SPIKE -> ACTIVE`, demands STATUS |
| `7a0e6c82b` | `added as ACTIVE`, demands STATUS | **no lifecycle move at all** | `SPIKE -> ACTIVE`, demands STATUS |

### Why W2's safety argument could not see this

W2 replayed 400 commits and reported **0 newly green**. That number is an
artifact, not evidence. The replay measures commits that were red under the base
gate; the base gate **enforced** the obligation W2 removed, so every commit that
would have gone green had already paid it. **A replay of already-gated history
is structurally incapable of detecting the removal of the obligation that gated
it.** The nine transitions are therefore pinned as CONSTRUCTED cases in
`tests/scripts/test_doc_checkpoint.py`, and the replay is used only for the
cost, which is what it can measure.

### The fix, and what it costs

```python
kind = CLAIM if state in STATES else RECORD
```

Key on the **destination**. It closes all nine, preserves W1's
`ANCHOR-BACKFILL` ruling (`DONE -> ANCHOR-BACKFILL` stays RECORD, and so do
`DONE -> INVENTORIED` and `DONE -> SPIKE`), and matches the polarity
`CLAIM_ON_ARRIVAL` already uses — a rule that reads where the row landed and
ignores where it came from.

Replayed over the same 400 non-merge commits ending at `e2a9e035d`:

| verdict | commits |
|---|---:|
| red under pre-W2 | 44 |
| red under W2 | 45 |
| red under W3 | 47 |
| **newly red, W3 against pre-W2** | **3** |
| **newly green, W3 against pre-W2** | **0** |
| newly red, W3 against W2 | 2 |
| newly green, W3 against W2 | 0 |

Each of the three is a genuinely unpaid surface:

| commit | move | what is missing |
|---|---|---|
| `67e53e716` | `TOOLS-PARSER-BREADTH` `INVENTORIED -> PARTIAL` | `docs/BENCHMARKS.md` (STATUS paid) |
| `33f570ea9` | `ENG-WEIGHT-OFFLOAD` `INVENTORIED -> READY` | `docs/BENCHMARKS.md` (STATUS paid) |
| `ab6e65216` | `ENG-CUDAGRAPH-BREAK` `INVENTORIED -> READY` | `docs/BENCHMARKS.md`, `docs/STATUS.md` |

**The `ab6e65216` accounting is corrected.** W2 counted it as one newly-red true
positive. Its *lifecycle* half is a true positive under W3. Its `## Now` half —
the only error W2 itself raised on it — is a **false positive**: the row's
`Spike/spec` column links `specs/eng-cudagraph-break.md`, which that commit
CREATES, 834 lines, with a populated `## Now` at `:818`. `spec_for_row` takes
the FIRST link on the line, which is an evidence link to
`specs/decode-graph-scratch-uaf-2026-07-18.md`, so the gate demanded a file the
change had no reason to touch. The move being unobserved is true; the message is
wrong. Pinned by `SpecForRowTakesTheFirstLink` and carried in `## Owed`.

### The public-term criterion, re-derived per state

W2's `RECORD_STATES` comment said the three are "real lifecycle positions that
the public pages carry **NO** term for", and W2 made that criterion executable
in `test_partial_is_a_public_status_term` /
`test_anchor_backfill_is_not_a_public_status_term`. Applying W2's own criterion
to `INVENTORIED` returns the **opposite** answer to the one W2 recorded, and
neither the W2 spec nor the W2 tests contain the string `Inventoried` — it was
derived for `ANCHOR-BACKFILL` and extended to two more states without being
re-derived. The criterion is falsified in both directions:

- **Not sufficient.** `docs/STATUS.md:42` defines
  `| Inventoried | The gap has a stable record but no accepted implementation |`
  and `:56` uses it in a live projection cell.
- **Not necessary.** `READY`, `TODO`, `BLOCKED`, `DROPPED` and `N/A` are all in
  `STATES` and `docs/STATUS.md` names none of them (grep count 0 each).

The criterion that survives is whether the position belongs to the CAPABILITY or
to the ROW'S RECORD. Re-derived separately for each of the three:

**`INVENTORIED` — on the page, and still a record state.** `STATUS.md`'s own
definition describes the RECORD ("the gap has a stable record") together with
the ABSENCE of an implementation, so carrying the word does not make arriving
there a capability claim. The two obvious costs were measured rather than
argued, and neither decides it: moving `INVENTORIED` into `STATES` changes **0
of 793** row resolutions, and the 400-commit replay is **identical** either way
(47 red, 2 newly red vs W2, 0 newly green). What decides it is that
`INVENTORIED` is where a PRE-CLAIM row and a DEMOTED row both sit. With it in
`STATES`, `SPIKE -> INVENTORIED` would demand `docs/STATUS.md` and
`docs/BENCHMARKS.md` for a row that has never claimed anything — the
public-document edit with nothing true to write that
`check-doc-checkpoint.py:4-17` records as the reason the file was rewritten. It
stays a record state.

**`SPIKE` — pre-claim by protocol.** `.agents/feature-matrix.md` gives a `SPIKE`
row a `CLAIM-*` and not a spec, and `docs/STATUS.md` carries no term (grep 0).
Neither limb argues for admitting it. Note that unlike `INVENTORIED`, `SPIKE`
also carries a measured resolution cost: admitting it moves 2 of 793 rows to the
wrong state (`MODEL-TEXT-deepseek-v2…` `BLOCKED -> SPIKE`,
`MODEL-TEXT-kimi-linear…` `READY -> SPIKE`), because their evidence prose says
"the row stays `SPIKE`". That is the measurement W2 recorded for the
cell-anchoring asymmetry, and it belongs to `SPIKE` alone.

**`ANCHOR-BACKFILL` — W1's ruling, re-derived independently and unchanged.** It
is a property of the record by definition, `docs/STATUS.md` carries no term
(grep 0), and a `DONE <-> ANCHOR-BACKFILL` move changes nothing a reader of that
page could be told.

### The residual this leaves, sized

Destination-keying does not observe a **demotion**: `DONE -> INVENTORIED` is a
retraction that `docs/STATUS.md` genuinely projects, and it stays a RECORD move
owing only the spec's `## Now`. Sized rather than asserted: over the 400 commits
ending at `e2a9e035d` the transition census is **11 claim -> claim, 6 record ->
claim, 0 claim -> record, 0 record -> record**, plus 14 arrivals. The 6 are
exactly the population W2 un-gated. The demotion has zero traffic. Carried in
`## Owed`.

### W3 tests

`TheDestinationDecidesTheMoveClass` in `tests/scripts/test_doc_checkpoint.py`
pins all nine transitions constructed, that paying both surfaces discharges
them, and that the arrival rule is unreachable once `previous` resolves. Its
mutation restores W2's both-endpoints line and requires all nine to go green,
then re-asserts the red after restoring, so a mutation that never applied cannot
read as a pass. `test_a_move_into_a_record_state_is_still_a_record_move` holds
the other polarity for all three record states.

Three W2 tests are repaired rather than added:

1. `test_a_record_move_reaching_a_claim_state_stays_a_record_move` pinned the
   defect on a false premise. Renamed to
   `test_a_record_move_reaching_a_claim_state_is_a_claim_move`, with the premise
   corrected in the docstring.
2. `test_anchor_backfill_stays_out_of_the_tuple` asserted
   `"ANCHOR-BACKFILL" in spec.split("## Owed", 1)[1]`. The first literal
   `## Owed` is inline prose at `:59`, not the section at `:479`, so it scanned
   420 lines of spec body and could not fail — and it did not: W2 rewrote
   `## Owed` without `ANCHOR-BACKFILL` and the test stayed green. Re-anchored to
   the section body through `owed_section()`, which is itself mutation-tested by
   `test_the_owed_section_reader_is_not_the_whole_file`. **Red-before evidence:
   the re-anchored assertion FAILS against W2's `## Owed` at `503e45900`.**
3. `test_every_row_the_checker_could_resolve_there_is_a_phantom` catches
   REPLACING a `FUSED` cell with `` `ACTIVE` `` and does not catch ADDING
   `` `ACTIVE` `` as a new column beside it — 3 passed, rc 0 — because it
   asserts `len(classification) == 1`. Given the file's header says the
   classification stands "in place of a lifecycle state", the additive shape is
   the likelier way it changes.
   `test_no_resolved_row_carries_a_bare_state_cell` asks the other question:
   every resolved row's only bare backticked cell is its own ID cell, so a real
   State column is a second one however it arrives.

### Two corrections to W2 record keeping

- `spec_now_errors`'s comment said "52 of 52 `SPIKE` rows". Re-derived over the
  seven `ROW_TABLES` at `503e45900`: **50 of 50**. The `ANCHOR-BACKFILL` figure
  in the same sentence, 50 of 56, is correct, as is 416 of 463 for
  `INVENTORIED`.
- `TheSglangMatrixIsNotALifecycleTable` reported 46 keyed rows and four
  classification values summing to 43 without reconciling the two.
  Reconciled: 46 = `FUSED` 24 + `SGLANG-DISTINCT` 5 + `OUT-OF-SCOPE` 8 +
  unbackticked `INVENTORIED` 6 + 3 rows whose axis cell is formatted differently
  again (`SGLANG-SCHED-INBATCH` "ACTIVE (order-only)", `SGLANG-CONSTRAIN-JUMP`,
  `SGLANG-ORACLE-PERF`).

### W3 counters, re-measured

W2 measured its counters and W3 measures them again on the same tree, base
version restored from the index and the W3 version restored byte-for-byte
afterwards against a pre-taken sha256.

| counter | at `503e45900` | with W3 | why it cannot move |
|---|---|---|---|
| `check-gate-commands.py --check` | rc 0 | rc 0 | `GATED_STATES` (`:56`) is its own tuple and does not import this one; `RUNNABLE_BASELINE` (`:424`) keys on matrix rows, and `GATE-DOC-CHECKPOINT-STATES` has no matrix row |
| `check-agent-record.py` | rc 0, `ENGINE=169 MODEL=377 QUANT=84 KERNEL=57 BACKEND=85 ANCHOR-ROT=37` | identical | no matrix row and no index row changes; #1434's index row already names this owning row, so `UNOWNED_HIGH_WATER` cannot move |
| record-anchor ratchet | unchanged | unchanged | no matrix row changes |
| `docs/STATUS.md` prose ratchet | untouched | untouched | no public document changes |

The `## Gates` section of this spec is NOT edited by W3, which is the #1376
shape: filling a spec's `## Gates` moves its row into the runnable population.
No row moves lifecycle state in this change either, so it owes no checkpoint
surface.

## Owed

- [#1434](https://github.com/mudler/vllm.cpp/issues/1434) is closed by W1 for
  `PARTIAL`, by W2 for resolution of the record states, and by W3 for the move
  classification. What remains is narrower than what it replaced, and each item
  is sized rather than asserted:
  - **A DEMOTION out of a claim state is still not projected.** `DONE ->
    INVENTORIED` is a retraction `docs/STATUS.md` carries a term for, and
    destination-keying makes it a RECORD move owing only the spec's `## Now`.
    Admitting it means either putting `INVENTORIED` in `STATES`, which makes
    `SPIKE -> INVENTORIED` demand both public surfaces for a row that has never
    claimed anything, or splitting the class by direction — a different rule and
    a different row. Measured traffic: **0 of the 17 non-arrival transitions in
    400 commits**.
  - `ANCHOR-BACKFILL` and `SPIKE` moves owe their spec's `## Now` and nothing
    else, by the W1 ruling re-derived per state in `## W3`. That is a decision,
    not debt, and it is listed here so the state names stay in this section.
  - `spec_for_row` returns the FIRST `specs/*.md` link on a row line, which is
    not always the `Spike/spec` column. **97 of 793 rows (12%)** link more than
    one spec. `ab6e65216` is exactly this: the move is real, the spec named is
    an evidence link, and the file the row actually links is created by that same
    commit with a populated `## Now`. Pinned by `SpecForRowTakesTheFirstLink`
    so a repair is a deliberate edit. Fixing it means deciding which column is
    authoritative per matrix, or accepting any linked spec — the second is a
    loosening and needs its own argument.
  - `row_states` lets a cell-anchored RECORD state win OUTRIGHT and `continue`
    past `STATE_CELL`, so an evidence cell that OPENS with a backticked record
    state demotes a claim row wherever the real State column sits. **Zero rows
    do that today.** The repair is already sized: taking whichever of the last
    `RECORD_CELL` and the last `STATE_CELL` match ends LATER in the line changes
    **0 of 793** resolutions. Not made here because it is a separate semantic
    change to the resolver and wants its own red-before case.
  - A row ID that leaves a table entirely is still unobserved. Measured at
    **zero occurrences in 400 commits**, so this is a hole with no known traffic
    rather than a live gap.
  - The last-match heuristic still resolves **10 of 793** rows differently from a
    column-position proxy. A column-aware resolver would change 11 rows at once
    and is a different row: it moves what counts as a move for rows this change
    does not touch.
  - `.agents/sglang-matrix.md` stays out of `ROW_TABLES` by decision, not by
    omission. If it ever gains a real lifecycle State column, either
    `test_every_row_the_checker_could_resolve_there_is_a_phantom` or
    `test_no_resolved_row_carries_a_bare_state_cell` fails and the decision is
    made again.

## Now

W2 landed on `main` as `ba4634204` (#1609) from base `e2a9e035d`, **without an
independent review**. The post-landing fresh review reproduced the resolution
work as sound — 52 transitions become observable — and returned **FAIL** on one
finding: keying the move class on both endpoints removed
`REQUIRED["lifecycle"]` from the nine
`{INVENTORIED, SPIKE, ANCHOR-BACKFILL} -> {ACTIVE, GATING, DONE}` transitions,
which the arrival rule had gated before resolution made `previous` non-`None`.
The review also found the `INVENTORIED` public-term criterion applied without
being re-derived, the `ab6e65216` cost accounting counting a false positive as a
true one, an additive-column blind spot in the `sglang-matrix` decline, and a
tautological `## Owed` assertion.

W3 is the repair, on `main` rather than a revert, on branch
`row/GATE-DOC-CHECKPOINT-STATES-W3` from base `503e45900`: one line in
`transitions()`, the per-state re-derivation recorded in `## W3`, and the four
test repairs. Next: fresh scoped review of the immutable head — mutate the
destination-keyed line back to the both-endpoints form and confirm all nine go
green, re-run the 400-commit replay for 3 newly red / 0 newly green against
`e2a9e035d`, and confirm no pinned counter moved.
