# NOW.md is derived, and the freshness obligation moves to the row's spec

Issue: [#374](https://github.com/mudler/vllm.cpp/issues/374).
Row: `ENG-NOW-DERIVED`. Follow-up to
[#364](https://github.com/mudler/vllm.cpp/issues/364) /
[retire-shared-record-surfaces.md](retire-shared-record-surfaces.md).

`.agents/NOW.md` is a surface every PR must write. #364 removed its byte budget —
the forced *eviction* — but not the forced *write*, so it remains a lock under
the invariant that same change added to `AGENTS.md`. This row removes the last
writer.

## Scope

**In scope.** The `NOW` entry in `check-doc-checkpoint.py`'s lifecycle triple and
its replacement by a per-row obligation; a `## Now` line in row specs;
`scripts/now.py`; the live-claims table in `.agents/NOW.md`; a regrowth guard in
`check-now-current.py`; the cold-start step in `AGENTS.md`; and the compression
of `AGENTS.md`'s §Records paragraph.

**Out of scope.** The legacy claims table in `coordination.md` (already handled
additively by #364 and emptying on its own), every other public document, and
all product source. No kernel, model or gate semantic moves.

## Upstream chain

None. vLLM has no counterpart to this protocol machinery, so the mirror rule does
not apply and there is no upstream `file:line` to port from. Governed instead by
`AGENTS.md` §"Changing the rules or a checker", which requires a spec, a
red-before test or mutation, and green-after evidence, and forbids turning a red
gate green by deleting an assertion.

## Our baseline — why the file still conflicts

`.agents/NOW.md` conflicted in 5 of the 16 conflicting open PRs measured at
`origin/main` `d928e2c3` (#364). The cause is one line:

```python
REQUIRED = {"lifecycle": (STATUS, BENCHMARKS, NOW), ...}
```

Any row that changes lifecycle state must edit `.agents/NOW.md`, so the gate
itself is what marches every row-advancing PR into a single shared file. This is
not a matter of discipline; no amount of care stops it while the requirement
stands.

The evidence that the requirement is what does it: `d72ac5a3`,
*"docs(now): keep the ROCm live-claim row inside the NOW.md budget"* — a commit
whose entire content is one author paying the shared-file tax, landing while #364
was removing the constant that demanded it.

### What the file is made of

Almost every line is a projection of a record that already has a per-row home.

| Content | Canonical home | Derivable? |
|---|---|---|
| row state, owner | the matrix row | yes |
| who is working on it | `.agents/claims/CLAIM-*.md`, open PRs | yes |
| the row's next command/step | — | **no: this is the gap** |
| current gate, cross-row priorities | authored, operator cadence | no |

Only two things are genuinely authored. The per-row next step has no per-row home
today, which is exactly why it is written into the shared file. Give it one and
the shared write disappears.

## Port map

Nothing is ported; every item is local protocol work, recorded as from-scratch.

| Item | Local anchor | Motion |
|---|---|---|
| `NOW` in the lifecycle triple | `scripts/check-doc-checkpoint.py` `REQUIRED` | remove |
| moved row must carry `## Now` in its spec | `scripts/check-doc-checkpoint.py` | add |
| `NOW` in `PUBLIC_SURFACES` | `scripts/check-doc-checkpoint.py` | remove |
| digest renderer | `scripts/now.py` | new |
| live-claims table | `.agents/NOW.md` | delete (generated instead) |
| regrowth guard | `scripts/check-now-current.py` | add |
| line/entry caps, headings, stamp | `scripts/check-now-current.py` | retain |
| cold-start step 3 | `AGENTS.md` | point at `scripts/now.py` |
| §Records measurement paragraph | `AGENTS.md` | compress to the rule |

## Design

**The obligation is relocated, not deleted.** "The live position must be current"
survives intact; it is paid in a file with one writer instead of one with all of
them.

**W1 — the row's next step lives in the row's spec.** A `## Now` section, one
short line: what the next command or step is. The row's author is the only person
who touches that file, and `.agents/specs/` took **zero** conflicts across the
whole #364 sample.

**W2 — `check-doc-checkpoint.py` requires the spec, not the digest.** `NOW` leaves
the lifecycle triple and `PUBLIC_SURFACES`. In its place, every row named by
`lifecycle_moves()` must have its spec among the changed paths *and* that spec
must carry a non-empty `## Now`. The row's spec is found through the matrix row's
existing `Spike/spec` cell, so no new mapping is introduced. A row whose spec
link is absent reports that, rather than silently passing.

**W3 — `scripts/now.py` renders the digest on demand.** Sources: the matrices for
`SPIKE`/`ACTIVE` rows and their owners; `.agents/claims/CLAIM-*.md`; each row
spec's `## Now`; and open PRs for live branch state. It is **offline-first**: the
PR half degrades to `REMOTE_UNVERIFIED` exactly as `claim-view.py` already does,
and never fails the render. A digest that needs the network to print is a worse
NOW.md than the one it replaces.

**W4 — `.agents/NOW.md` keeps only what is authored.** Current gate, next actions,
and the invariants that bite: operator cadence, not per-PR. Its live-claims table
is deleted and comes from `now.py`. `check-now-current.py` gains a guard that the
file may not carry a per-row claims table again, so it cannot regrow into the log
it replaced — the structural half of the fix, since the removal alone would decay
back.

**W5 — `AGENTS.md`.** Step 3 points at `scripts/now.py`. The §Records paragraph
loses its dated measurement: a rules file that accumulates "measured on DATE at
SHA" is a state log, which this protocol does not have, and it is the same
accumulating-justification pattern #364 removed from `STATUS_RATCHET`. The
evidence lives in this spec and in the commit, where `git log --grep` is the
record.

## Tests to port

No upstream tests exist to port, for the reason in Upstream chain. Written from
scratch against our own checkers, red-before and green-after, each in the changed
checker's **paired** suite — the lesson #364's `pr-size` finding taught, since a
new standalone test file is easy to miss or delete wholesale.

1. `tests/scripts/test_doc_checkpoint.py`: a lifecycle move that edits `STATUS`
   and `BENCHMARKS` but **not** `.agents/NOW.md` passes (RED before W2); the same
   move without the row's spec `## Now` fails; a spec with an empty `## Now`
   fails. Mutation: restore `NOW` to the triple and the first case must go red.
2. `tests/scripts/test_check_now_current.py`: a `NOW.md` carrying a per-row claims
   table fails the regrowth guard; the authored sections pass. Mutation: remove
   the guard and the case goes green, proving it is load-bearing.
3. `tests/scripts/test_now_render.py` for `scripts/now.py`: renders with the
   network stubbed out and still lists every `SPIKE`/`ACTIVE` row; an unreachable
   PR source yields `REMOTE_UNVERIFIED` and a still-successful render, never an
   exception or an empty claim set.
4. Every existing `check-now-current` case for headings, stamp, line cap and entry
   cap must stay green unchanged, proving the retained obligations survive.

## Gates

- `scripts/agent-preflight.sh` and `--staged`.
- Full `tests/scripts/` suite, read by `Status:` and test-case COUNT, never the
  `assertions:` line, and with `if __name__ == "__main__"` at the END of every
  file touched — #364 found two suites reporting OK while never running their
  newly appended cases.
- `python3 scripts/agent-integration.py --base origin/main`.
- No CUDA, GPU or SACRED gate is implicated: no product source is touched. Stated
  so the absence is a recorded decision rather than an omission.

## Evidence

The `REQUIRED` line above; NOW.md's 5-of-16 conflict count at `d928e2c3`;
`d72ac5a3` as an unprompted instance of the tax being paid; and the before/after
of a lifecycle PR's changed-path set, which is the binding result — a row
advancing must touch **no** shared surface.

## Dependencies

None on product rows. Depends on `.agents/claims/` from #364, already landed in
`87308dea`.

## Work breakdown

| ID | Work | Files owned | Done when |
|---|---|---|---|
| W1 | `## Now` in row specs | `.agents/specs/*.md` for live rows | live rows carry the line |
| W2 | doc-checkpoint requires the spec | `scripts/check-doc-checkpoint.py`, `tests/scripts/test_doc_checkpoint.py` | NOW-free lifecycle move passes; missing/empty `## Now` fails; mutation red |
| W3 | `scripts/now.py` | `scripts/now.py`, `tests/scripts/test_now_render.py` | renders offline; `REMOTE_UNVERIFIED` degrade proven |
| W4 | strip the claims table, guard regrowth | `.agents/NOW.md`, `scripts/check-now-current.py`, its test | table gone; regrowth case red without the guard |
| W5 | `AGENTS.md` | `AGENTS.md` | step 3 points at `now.py`; §Records is the rule alone |

## Risks / decisions

- **A generated digest loses authored judgement.** Accepted and bounded: only the
  derivable half is generated. Current gate and next actions stay authored, which
  is why `.agents/NOW.md` survives at all rather than being deleted.
- **`now.py` could become slow or network-bound**, making cold start worse than
  the file it replaces. Mitigated by the offline-first requirement and gated by
  the stubbed-network test; a render that needs the network is a defect.
- **NOW.md could regrow a claims table.** This is the likely decay path, so it is
  a checker rule rather than a convention.
- **The cold-start ritual changes for every agent.** Deliberate, and the reason
  this is its own row with its own review rather than a rider on #364.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if: removing `NOW` from the
lifecycle triple would drop an obligation not demonstrably carried by the per-row
`## Now` requirement; or if a row's spec cannot be resolved from its matrix row
for a material number of live rows, which would mean the relocation has no
reliable target and the mapping must be fixed first.

## Now

DONE at `dbd0d51c`: no implementation step remains; use `scripts/now.py` for the
derived live position.

## Outcome

Landed `dbd0d51c` (PR #376). The performance/parity result is honestly `VOID`:
this is local protocol/checker machinery with no vLLM analogue, and it changes
no product source, model output, latency, throughput, or memory behavior.

**What was measured.** `.agents/NOW.md` left the lifecycle triple and
`PUBLIC_SURFACES`. The binding lifecycle-path test passes with exactly the moved
matrix, that row's own spec, `docs/STATUS.md`, and `docs/BENCHMARKS.md`;
`.agents/NOW.md` is absent. The paired suite ran 26 cases: omitting the spec,
omitting either public projection, removing `## Now`, or leaving it empty fails,
and the mutation that restores `NOW` to the lifecycle triple makes the NOW-free
case fail again. `scripts/now.py` rendered 105 live rows in ~50 ms with no
network; its 11-case suite preserves the roster while reporting
`REMOTE_UNVERIFIED`. The 15-case digest suite rejects table regrowth, and its
mutation proves `ROW_TABLE_LINE` is load-bearing.

The stronger draft shorthand "touches no shared surface" is rejected as
inaccurate: `STATUS` and `BENCHMARKS` intentionally remain lifecycle
projections. The measured result is narrower and sufficient: no lifecycle PR
writes the shared *live-position digest*.

**What the implementation found that the plan had not.** Two consumers of the
per-row table were not in the issue's inventory: `check-release-binary-contract.py`
pinned a literal Release row inside NOW.md, and two of #364's own tests asserted
that adding a ROW to NOW.md is free. Both were relocated rather than deleted. This
is the second time scoping this class of change from analysis alone missed a
consumer — the first was #364's `pr-size` finding — which is an argument for
grepping every reader of a surface before removing it, not for a bigger spec.

**What was rejected.** Deleting `.agents/NOW.md` outright: the current gate and
the cross-row next actions are authored judgement no generator produces. Merely
enlarging its budget keeps the write lock and postpones the same eviction;
dropping freshness loses the obligation instead of relocating it. A digest that
needs a network to print would be worse than the file it replaced. Deriving
claims solely from `gh pr list` was rejected for the same offline reason in #364
and stays rejected here.

**The cost, recorded because it was not flagged before landing.** Removing a
shared surface imposes a ONE-TIME conflict on every in-flight branch that touched
it. Measured after the merge: 5 of the 8 still-conflicting open PRs conflict on
`.agents/NOW.md` and 4 on `check-public-doc-tables.py`, and #360 was `MERGEABLE`
before this work and `CONFLICTING` after. The resolution is mechanical — take
main's version, move the row's next step into its own spec's `## Now` — but the
cost is real and should have been stated in the spec before the change landed.

**Why the backfill is progressive.** Requiring `## Now` in every existing spec at
once would be a bulk edit across ~100 files, which is exactly the unreviewable
shared-surface rewrite this row exists to avoid. The requirement therefore binds
on movement, and `now.py` shows a dash until a row moves. This is the selected
compatibility policy, not unfinished work in this row.

**Why the retained defaults remain.** Offline-first rendering preserves cold
start and degrades explicitly to `REMOTE_UNVERIFIED`. The freshness stamp and
the three authored headings say when the snapshot was known true and retain the
non-derivable operator context. The 100-line cap keeps that context readable in
one pass; the 400-character per-entry cap bounds narrative growth locally. The
old 6,000-byte cap stays removed because variable-size edits forced unrelated
evictions. The per-row-table regrowth guard stays enabled so the eliminated
shared writer cannot silently return.
