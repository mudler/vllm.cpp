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

## Owed

- [#1434](https://github.com/mudler/vllm.cpp/issues/1434) is closed by W1 for
  `PARTIAL` and by W2 for all four residuals it filed. What remains is narrower
  than what it replaced, and each item is sized rather than asserted:
  - `spec_for_row` returns the FIRST `specs/*.md` link on a row line, which is
    not always the `Spike/spec` column. **97 of 793 rows (12%)** link more than
    one spec, so on those the error can name a spec the change did not have to
    touch. `ab6e65216` above is exactly this: the move is real, the spec named is
    an evidence link. Pre-existing and unchanged by W2, which raises its
    exposure. Fixing it means deciding which column is authoritative per matrix,
    or accepting any linked spec — the second is a loosening and needs its own
    argument.
  - A row ID that leaves a table entirely is still unobserved. Measured at
    **zero occurrences in 400 commits**, so this is a hole with no known traffic
    rather than a live gap.
  - The last-match heuristic still resolves **10 of 793** rows differently from a
    column-position proxy. A column-aware resolver would change 11 rows at once
    and is a different row: it moves what counts as a move for rows this change
    does not touch.
  - `.agents/sglang-matrix.md` stays out of `ROW_TABLES` by decision, not by
    omission. If it ever gains a real lifecycle State column,
    `test_every_row_the_checker_could_resolve_there_is_a_phantom` fails and the
    decision is made again.

## Now

W2 landed on `row/GATE-DOC-CHECKPOINT-STATES-W2` from base `e2a9e035d`. Next:
fresh scoped review of the immutable head, re-running the four W2 mutations and
the 400-commit replay to confirm the newly-red count is still 1.
