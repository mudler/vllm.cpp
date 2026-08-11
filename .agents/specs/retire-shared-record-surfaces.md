# Shared record surfaces are a lock, and the lock is the conflicts

Issue: [#364](https://github.com/mudler/vllm.cpp/issues/364).
Row: `ENG-RECORD-CONFLICT-SURFACES`.
Measured against `origin/main` `d928e2c3` on 2026-08-11.

Three tracked files are shaped so that concurrent pull requests must collide in
them. That shape, not the volume of parallel work, is what makes most open PRs
unmergeable. This spec records the measurement, names the defect in each
surface, and defines the one invariant that prevents the class.

## Scope

**In scope.** The `STATUS_RATCHET` global in `scripts/check-public-doc-tables.py`;
the fixed character budget in `scripts/check-now-current.py`; the active-claims
table in `.agents/coordination.md`; the row ordering of the keyed tables in
`.agents/roadmap_v1.md`; and one new invariant in `AGENTS.md` covering the class.

**Out of scope, deliberately.** `.agents/specs/`, the `*-matrix.md` inventories,
`.agents/benchmark-record.md`, the public documents themselves, and every
per-cell or per-paragraph cap. These are measured below as *not* implicated;
removing them would be scope this evidence does not support.

**Not a correctness change.** No product source, kernel, or gate semantic moves.
`docs/` content is unchanged by this row; only what *gates* it changes.

## Our baseline — what was measured

`git merge-tree --write-tree origin/main <head>` was run for every open PR
GitHub reports as `CONFLICTING`. **16 of 29 open PRs conflict (55%). 13 of the
16 conflict in bookkeeping files only, with no product code involved.**

| File | conflicting PRs |
|---|---|
| `.agents/coordination.md` | 8 (#334 #336 #341 #343 #345 #348 #361 #282) |
| `.agents/NOW.md` | 5 (#361 #282 #266 #171 #155) |
| `.agents/roadmap_v1.md` | 4 (#361 #346 #324 #266) |
| `scripts/check-public-doc-tables.py` | 4 (#361 #282 #267 #266) |
| `docs/STATUS.md` | 4 (#266 #171 #158 #155) |
| `docs/BENCHMARKS.md`, `docs/FEATURES.md`, `.agents/benchmark-record.md` | 2 each |
| any `src/` or `tests/` path | 3 (#158, #266, and #171 a stale mega-branch) |

Churn over the last 300 commits: 95 of 252 non-merge commits (38%) touched only
bookkeeping; 22 commits are re-merges of `main` into a landing head, and
`row/MODEL-MUSE-GLIMMER` alone re-merged 7 times.

### Defect 1 — `NOW.md` is a fixed-size buffer with zero headroom

`check-now-current.py:31` sets `MAX_CHARS = 6000`. The file measures **exactly
6000 characters** and 91 of 100 permitted lines: tuned to the byte, with no room
left. Adding a row therefore requires evicting a different row, so every PR
performs a read-modify-write of one shared global.

Concurrent read-modify-write on a shared global loses updates. The git conflict
is the *lucky* outcome — a clean three-way merge would apply both PRs' evictions
and both PRs' additions, silently dropping two live rows and blowing the budget
the checker exists to defend. The budget does not merely cause conflicts; it
makes a successful merge unsafe.

### Defect 2 — a ratchet that couples every PR to lines it does not own

`check-public-doc-tables.py:557` holds `STATUS_RATCHET = {"chars": 243245}` — a
hardcoded byte count of `docs/STATUS.md`, stored in a second file, permitted to
move in one direction only. A PR that owes `STATUS.md` one lifecycle line must
therefore delete unrelated prose from some other row's paragraph to pay for it,
and edit the checker in the same change. Two edits per PR to two shared files,
both of which every other PR is also editing.

The design already knows. `check-public-doc-tables.py:331` reads: *"a ratchet
pinned to the byte turns every concurrently merged row's one-line status edit
into a spurious failure."* That was answered by adding slack to the constant
rather than by removing the coupling, so the failure returned at the next
cadence of parallel work.

### Defect 3 — `coordination.md` claims are insert-at-one-anchor

The six ROCm GDN PRs (#334 #336 #341 #343 #345 #348) are one author's sequential
stack. They conflict on **nothing but `.agents/coordination.md`**: each appends a
roughly 1,500-character claim row at the same anchor. The content of that row —
what the branch owns, what it excludes, what state it is in — is the pull
request description, transcribed into a tracked file that every other claim also
writes.

### It contradicts the protocol it serves

`AGENTS.md` states *"History is git"*, *"There is no state log"*, and *"A
registry of exceptions is a state log, and this protocol has none."* The
active-claims table and the `NOW.md` live-claims table **are** state logs,
hand-maintained, duplicating `gh pr list`, `row/<ID>` branch names, and issue
state. The argument that refuses a waiver registry applies unchanged to a claims
registry.

Precedent, twice: the `policy.csv` registry was retired in `0f3e44ee`, and the
per-class line budgets were retired on 2026-08-10 because the gate fired on
ordinary work. This is the third instance of one failure mode.

### What the evidence exonerates

Every low-conflict surface shares one property — **one writer per file**:

- `.agents/specs/<slug>.md`: one file per row, **zero conflicts** in the sample.
  This is the shape the rest should copy.
- `*-matrix.md` and the roadmap inventory: stable row IDs and the 362-arch
  inventory are not derivable from any other source. The roadmap's 4 conflicts
  are the insert-point defect, repairable by ordering rather than deletion.
- `.agents/benchmark-record.md`: append-only and already union-mergeable.
- The public documents are user-facing and stay. The ratchets on them are the
  defect; the pages are not.

## Upstream chain

None, and this is recorded rather than omitted. vLLM has no counterpart to this
protocol machinery, so the mirror rule does not apply and no upstream `file:line`
exists to port from — there is no executing chain on the other side to trace
against. The change is local-protocol work governed instead by
[`AGENTS.md`](../../AGENTS.md) §"Changing the rules or a checker", which requires
a spec, a red-before test or mutation, and green-after evidence, and forbids
turning a red gate green by deleting an assertion or widening a scope.

## Port map

Nothing is ported; every item below is a local deletion or rewrite, recorded here
as from-scratch protocol work rather than left implicit.

| Item | Local anchor | Motion |
|---|---|---|
| `STATUS_RATCHET` constant and its comparison loop | `scripts/check-public-doc-tables.py:323,557,595` | delete |
| `MAX_CELL_CHARS`, `MAX_PARAGRAPH_CHARS` | `scripts/check-public-doc-tables.py:35,38` | **retain unchanged** |
| `MAX_CHARS` byte budget | `scripts/check-now-current.py:31` | delete |
| `MAX_LINES`, structure and stamp rules | `scripts/check-now-current.py:30,32` | retain |
| Active-claims table | `.agents/coordination.md` | remove; derive from open PRs + `row/<ID>` branches |
| Canonical hierarchy, row contract | `.agents/coordination.md` | retain |
| Keyed table row order | `.agents/roadmap_v1.md` | sort by stable ID |
| The invariant | `AGENTS.md` §"Records" | add |

## Design

**The invariant.** *No surface that every PR must write.* If N concurrent PRs all
edit file F, F is a lock. A surface is admissible only if it is one of: one file
per row (globbed for reading), genuinely append-only (union-mergeable), or
derived at read time from git and GitHub. Anything else is rewritten into one of
those three shapes.

**W1 — delete the `chars` key of `STATUS_RATCHET`.** *(Narrowed during
implementation; the original text said delete the whole constant.)* Reading the
checker showed the four keys are not alike. `chars` is a LENGTH, so it moves on
every edit — that is what made it the conflict driver and a merge hotspot in 4
of the 16 conflicting PRs. The other three (`h2_sections`, `long_paragraphs`,
`oversized_cells`) count QUALITY DEFECTS — sections, paragraphs over
`MAX_PARAGRAPH_CHARS`, cells over `MAX_CELL_CHARS` — so an ordinary lifecycle
line moves none of them and two concurrent PRs do not collide on them. They are
kept, and they carry the whole anti-decay obligation: BENCHMARKS.md's
11,127-line decay would have tripped `long_paragraphs` and `oversized_cells`
alone. Deleting them too would have dropped a real obligation for no conflict
benefit, which is exactly the case this spec's stop condition names.

Retain `MAX_CELL_CHARS` and `MAX_PARAGRAPH_CHARS`: local to the cell or
paragraph being edited, so they couple no two PRs.

**W2 — one file per claim in `.agents/claims/`.** *(Redesigned during
implementation. The original text said "derive claims from open PRs and branch
names"; that is wrong and is recorded here rather than quietly changed.)*

Two things came out of reading the consumers. First, deriving from GitHub would
put a **network call inside an offline gate** — `check-agent-record.py` runs in
preflight with no network, and a checker that cannot answer without `gh` fails
closed on every disconnected run. Second, the table is not redundant with the PR
list: `check_row_contracts` uses it for a **bidirectional cross-check** — every
`SPIKE`/`ACTIVE` matrix row must be claimed by a live claim, and every claim must
name rows in those states. A PR list cannot supply that, so deriving would have
silently dropped a real guarantee. That is exactly the case this spec's stop
condition names.

The shape that satisfies the invariant without either cost is the one that
already works here: **one file per claim**, globbed. `.agents/claims/CLAIM-*.md`
holds one claim each, `parse_active_claims` reads that directory *and* the legacy
table, and the cross-check is unchanged because the parsed data is unchanged.

It is **additive on purpose**: no existing row is migrated. Rewriting 115 rows
spread across several interleaved tables, struck-through released entries, and
prose blocks — some of them below a later `##` heading — is how a record gets
silently lost, and it would be a large unreviewable diff. New claims go in their
own file; the table shrinks as claims close. The conflict source is closed for
all future claims from the day this lands, which is what the measurement is
about.

Keep the canonical hierarchy and the row contract: policy prose, edited rarely,
implicated in no conflict. `scripts/agent-role.py` keeps the coordinator record
it already owns.

**W3 — drop `NOW.md`'s hard character budget.** `MAX_LINES` and
`MAX_ENTRY_CHARS` stay and carry the obligation: a line cap bounds the page the
same way a byte cap does, but a row costs ONE line rather than a variable number
of bytes, and `MAX_ENTRY_CHARS` bounds each entry locally.

*Correction found by the W4 regression test, recorded because the original text
overclaimed:* removing the budget does **not**, on its own, make two concurrent
row additions merge. Two appends land on the same anchor and conflict either way
— that is W4's job. What the budget added on top was a forced DELETION of
unrelated content, and that is the part with teeth. It made every PR edit lines
it did not own, and it made a *successful* merge unsafe: git resolving two such
branches without complaint applies both evictions, so both victims vanish and no
gate notices. The budget was defended by losing the content it defended. W3
removes the eviction; W4 removes the collision; neither alone is sufficient.

**W4 — order the roadmap's keyed tables by ID.** Sorting the issue and portfolio
tables by a stable key turns "append at the same anchor" into "insert at a key",
which resolves distinct keys without a conflict. The keyed-record merge rule in
`AGENTS.md` is unchanged and still forbids accepting a three-way merge blindly.

**W5 — record the invariant in `AGENTS.md`**, in §"Records", with the measurement
that motivated it, so the class cannot be reintroduced by the next surface.

## Tests to port

No upstream tests exist to port, for the reason given in Upstream chain; the
suites below are written from scratch against our own checkers.

Red-before, green-after, per `AGENTS.md` §"Changing the rules or a checker". No
assertion is deleted to turn a gate green; each removal below deletes a *rule*
whose motivating obligation is proven still enforced elsewhere.

1. `tests/scripts/test_check_public_doc_tables.py`: a RED case asserting a
   `STATUS.md` that grows by one lifecycle line passes the checker; today it
   fails on the ratchet. Existing per-cell and per-paragraph cases must stay
   green unchanged, proving the retained caps still fire.
2. `tests/scripts/test_check_now_current.py`: a RED case for a `NOW.md` over
   6000 characters but within the line and structure rules. Structure, required
   headings, and stamp cases stay green.
3. A mutation case per removed rule, proving the obligation survives: delete a
   required `STATUS.md` section and confirm `check-public-doc-tables.py` still
   fails; skip the `NOW.md` refresh on a lifecycle change and confirm
   `check-doc-checkpoint.py` still fails.
4. A merge-shape regression: construct two branches that each add one row to the
   roadmap issue table and one row to `NOW.md`, and assert
   `git merge-tree --write-tree` reports no conflict. This is the defect under
   test; it must be RED before W3/W4 and GREEN after.

## Gates

- `scripts/agent-preflight.sh` and `scripts/agent-preflight.sh --staged`.
- Full `tests/scripts/` suite green, checked by `Status:` and test-case count,
  not by the `assertions:` line.
- `python3 scripts/agent-integration.py --base origin/main`.
- No CUDA, GPU, or SACRED gate is implicated: no product source is touched. This
  is stated so the absence of a GPU gate is a recorded decision rather than an
  omission.

## Evidence

The `merge-tree` conflict table above, reproducible at `origin/main` `d928e2c3`;
the two byte measurements (`NOW.md` at 6000/6000, `STATUS_RATCHET` at 243245);
the checker's own comment at `check-public-doc-tables.py:331`; and the
before/after count of `CONFLICTING` open PRs, which is the binding result.

## Work breakdown

Non-overlapping, one fresh implementer each, in this order. W1 and W2 are
independent of one another; W3 and W4 both touch the merge-shape regression and
so share an implementer; W5 lands last so the invariant is recorded against the
evidence the earlier items produce.

| ID | Work | Files owned | Done when |
|---|---|---|---|
| W1 | Delete `STATUS_RATCHET` and its loop; keep the local caps | `scripts/check-public-doc-tables.py`, `tests/scripts/test_check_public_doc_tables.py` | growth case GREEN; cap cases still GREEN; mutation proves a missing required section still fails |
| W2 | Remove the active-claims table; derive claims | `.agents/coordination.md`, `scripts/check-agent-record.py` if it reads the table | claim derivation documented; record gates green with no claims table present |
| W3 | Drop `NOW.md`'s byte budget, keep line/structure/stamp | `scripts/check-now-current.py`, `tests/scripts/test_check_now_current.py` | over-6000 case GREEN; structure cases still GREEN |
| W4 | Sort the roadmap's keyed tables by stable ID | `.agents/roadmap_v1.md`, merge-shape regression test | two-branch `git merge-tree` reports no conflict; RED before, GREEN after |
| W5 | Record the invariant with its measurement | `AGENTS.md` §"Records" | prose states the rule and the three admissible shapes |

Each W is its own worktree, its own `row/ENG-RECORD-CONFLICT-SURFACES-W<n>`
branch, and its own fresh-reviewer round. A reviewer that wrote the code does not
review it.

## Dependencies

None on product rows. It touches files that many in-flight PRs also touch, so it
should land as one small change and the affected branches rebase onto it once —
the last conflict of this class rather than a new one.

## Risks / decisions

- **The pages could decay once the ratchet is gone.** Accepted, with the caps and
  `check-doc-checkpoint.py` retained as the actual guard. A ratchet on total
  bytes never measured quality; it measured length, and it purchased that proxy
  by making every PR edit another PR's lines.
- **Derived claims are less rich than the table.** Accepted. The table's content
  was the PR description; the PR is the better home for it, and it cannot go
  stale there.
- **This spec's own change touches shared surfaces.** Unavoidable and
  acknowledged: the fix must edit the files it is fixing.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if: removing `STATUS_RATCHET`
would drop an obligation not demonstrably carried by the retained caps or
`check-doc-checkpoint.py`; or if the merge-shape regression test stays RED after
W3/W4, which would mean the conflict source was misidentified and the
measurement above needs redoing before any further deletion.

## Outcome

Pending. To be written when the row reaches `DONE`, recording the measured
`CONFLICTING` count after the change, what was rejected, and why each retained
cap is set where it is.
