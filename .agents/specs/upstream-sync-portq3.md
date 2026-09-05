# Sync cycle `e126687a9a`, wave PORTQ-3

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`. `scripts/check-agent-record.py` passes on a `Row:` line whether or
not the row resolves, so this note is here to stop a reader taking it for a
matrix reference. The issues this wave cites are carried under `## Owed` below,
which is the route AGENTS.md gives when no row owns the work.
Issue: [#2647](https://github.com/mudler/vllm.cpp/issues/2647).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq1.md`](upstream-sync-portq1.md),
whose method this wave continues unchanged.

## Now

**Done.** 40 of 40 re-derived: **5 ALREADY_SATISFIED, 8 REAL_GAP,
27 NOT_APPLICABLE** (20 `surface-absent`, 7 `inert`). **32 of the 40 disagree
with the recorded PORT-NOW disposition.** Seven issues carry the eight gaps
([#2648](https://github.com/mudler/vllm.cpp/issues/2648),
[#2649](https://github.com/mudler/vllm.cpp/issues/2649),
[#2650](https://github.com/mudler/vllm.cpp/issues/2650),
[#2651](https://github.com/mudler/vllm.cpp/issues/2651),
[#2652](https://github.com/mudler/vllm.cpp/issues/2652) which covers two entries,
[#2653](https://github.com/mudler/vllm.cpp/issues/2653), and the pre-existing
[#2527](https://github.com/mudler/vllm.cpp/issues/2527)); the seven deferred
entries are carried on [#2655](https://github.com/mudler/vllm.cpp/issues/2655),
and a citation defect found on the way is
[#2654](https://github.com/mudler/vllm.cpp/issues/2654).
Report: [`../sync/2026-09-03-portq3.md`](../sync/2026-09-03-portq3.md).

**Nothing was executed.** No build, no test run, no GPU, no lease.

The pin did **not** advance and nothing read here is a reason to move it. The
active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked forty times.** For each of PORT-NOW entries 81 to 120 of
`5559679229..e126687a9a` in upstream commit order, what is true **of this tree**?

`.agents/sync/2026-09-02-e126687.md` §4 says in as many words that **no
disposition in it is re-derived**: all 1002 are carried by SHA from
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md). A `PORT-NOW`
label is a prior wave's reading of an upstream diff. It is evidence about
upstream and says nothing about what this tree has.

PORTQ-1 measured that difference on entries 1 to 40 and published the number:
**34 of 40 labels did not hold**, and the record was accurate about upstream in
all forty. It never asked what is already here. This wave asks, for forty more.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 and 1465 counts before anything is read off it.
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming a row that **exists**.

Out of scope: porting anything, advancing the pin, entries 1-80 and 121-290, and
any measured number. Nothing in this wave is executed.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l          # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                              # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | sed -n '81,120p'
```

`portnow.txt` is the 315 SHAs of `../sync/2026-09-01-cdefd9d.md` §4. The `1465`
and the `290` are the counts `../sync/2026-09-02-e126687.md` §4 publishes, so the
extraction reproduces a committed number before it is used. The ordering is
checked against PORTQ-1's published entries: positions 1, 9, 21 and 40 come out
`3f1d40960f`, `bf2b45b5d6`, `0b6aa3c47c`, `b49eaf205a`, which are exactly the
SHAs that report names at those indices. If either denominator or the ordering
had failed to reproduce, this wave would have stopped, because it would have
invalidated PORTQ-1 too.

### 2.2 Three labels, and a named sub-shape for the third

`ALREADY_SATISFIED` cites the `file:line` or symbol that satisfies the entry.
`REAL_GAP` says what is missing, roughly how big, and who owns it.
`NOT_APPLICABLE` splits, because the two halves have different futures:

- **`surface-absent`** — the commit edits something this tree does not carry.
  Discarded work; there is nothing to port until the underlying feature is.
- **`inert:<gate>`** — the surface is here and the new arm is gated on something
  unreachable in every configuration this tree builds. **Deferred** work: it
  becomes real the day that gate is ported, so the gate is named, and the note
  goes where the person porting that gate will meet it rather than only in this
  report.

### 2.3 Upstream is read at a revision, never in a working tree

Every upstream citation is `git -C ${VLLM_SOURCE} show <rev>:<path>` or
`git show <sha>` for the diff. An anchor read in a working tree is wrong at the
pin.

### 2.4 Absence is never concluded from one grep

This is PORTQ-1's own retraction, and it is repeated here because the wave that
wrote the rule broke it. It grepped six upstream-named symbols, got zero hits,
and concluded "no such buffer" — while this tree carried seven of them under
different names (`g_slot_map` for `_slot_mapping_buffer`, `g_dpos` for
`positions`). This tree renames on the way in, so a failed grep on an upstream
identifier is the *expected* result even where the behaviour is present.

Each entry is searched under at least two spellings and in at least two
locations before anything is called missing, **and both searches are written
down beside the claim**. That last clause is new. PORTQ-1 recorded its method's
known hole: re-reading citations cannot catch a claim of absence, because
absence has no citation to re-read. Recording the searches gives the reviewer
something to falsify.

### 2.5 Git is asked whether it already triaged the entry

`git log --oneline --all --grep '<sha>'`, `git log -S'<symbol>'`, and a grep of
`.agents/` for the SHA and the upstream PR number, before any novelty is
claimed. `b7e414cc4` had already triaged three of PORTQ-1's forty by name, and
one of those three readings had gone stale under a moving tree. In this tranche
`.agents/sync/2026-09-01-cdefd9d.md` §13 — the #2524 worked list — already reads
two entries against the tree, and `.agents/specs/upstream-sync-portq1.md` names a
third as a worked counter-example. Each is re-verified against the current tree
rather than carried.

### 2.6 The reading is delegated, and its citations are re-checked

Six fresh readers take four to eight entries each, clustered by surface so a
reader builds context once. The wave operator prints the citations again and
resolves each against the tree before it is written down.

## 3. Risks

- **A carried label is anchoring.** Reading the record's sentence before the diff
  invites confirming it. Mitigated by reading the upstream diff first and the
  record's sentence last, and by treating a disagreement as the expected result.
- **`inert` is a judgement about reachability, not a grep.** A gate called
  unreachable that is in fact reachable turns a real gap into a deferral.
  Mitigated by naming the gate so a later reader can falsify it, and by putting
  the note on the issue that owns the gate.
- **A named owner row may not exist.** `check-agent-record.py` passes on a `Row:`
  line whether or not the row is real. Mitigated by checking every owner against
  `roadmap_v1.md` and the matrices before the issue is filed.
- **Forty is not 290.** Any rate computed here is an extrapolation from a
  non-random slice, not a count. Recorded as such, beside the number.
- **A sibling wave is live.** PORTQ-2 works entries 41-80 concurrently. This wave
  writes its own report and its own spec and touches no shared record surface.
- **Disk.** The host is at 96%. Nothing in this wave builds; a worktree that
  compiled would be the defect.

## 4. Gates

- `git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l` prints
  `1465`, and `comm -12 <(sort portnow.txt) range.txt | wc -l` prints `290`. The
  derivation is not used until both reproduce.
- Positions 1, 9, 21 and 40 of the derived order match the SHAs
  `.agents/sync/2026-09-03-portq1.md` §3 publishes at those indices.
- Every `path:line` in the report resolves in the tree at the merge commit, and
  every citation past the pin names its revision.
- Every owner row named by an issue resolves in `roadmap_v1.md` or a
  `*-matrix.md`, or the issue says plainly that none does.
- `scripts/agent-preflight.sh --staged`, read by grepping its output for
  `gate(s) failed` and `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at 40. The tranche is bounded; finishing the queue is not this wave's.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing read here is a reason to move it.
- Stop before entries 41-80. PORTQ-2 owns them.

## Owed

- The remaining PORT-NOW entries of `5559679229..e126687a9a` outside 1-120
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. They are classified here and
  implemented elsewhere; each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave reads and does not discharge.

## Outcome

**What the delegated readers got wrong, and it was one citation in twenty.** The
operator re-read all eight real gaps and eleven further citations. One named a
file that does not exist — `include/vllm/v1/worker/gpu/cudagraph_dispatch.h:163`,
where the symbol is at `src/vllm/v1/worker/gpu/cudagraph_dispatch.h:162`. Wrong
directory, one line off, substance correct; the same shape as the three PORTQ-1
corrected. No citation named a symbol that does not exist. §2.4 of the report
carries the correction rather than quietly repairing it.

**Why 20% is written down as an estimate and not a rate.** This tranche's 8 of 40
sits between tranche 1's 6 of 40 and #2524's 11 of 63, and the three agree to
within five percentage points. That agreement is the only reason the extrapolation
to 50-60 gaps over the 290 is worth stating at all, and §10 of the report gives
four reasons not to treat it as a measurement — the slice is contiguous rather
than random, four of the eight gaps sit in two subsystems, this tranche shares two
worked gaps with #2524's population, and two entries landing either way moves the
figure from 15% to 25%.

**Why `[107]` and `[114]` share one issue.** They are two upstream commits over
eight adjacent keys of one transition table, and both are the same missing
`REASONING_END`. AGENTS.md warns that intake without an exit is how 701 open
issues accumulated; two issues over ~8 lines of the same map is that shape. The
issue names both SHAs and all eight keys, so nothing is lost by the merge.

**Why the deferred entries were collected rather than filed one by one.** Six of
the seven `inert` entries have no open issue owning their gate — searched for
`EPLB`, `DBO`, `ubatch`, `tokwise`, `pooling runner` and `OffloadingConnector
worker`, all empty. Six new issues that each say "do not port this yet" would be
six issues nobody can close. One issue with six rows, each naming its gate and
the `path:line` that establishes it, is findable and closes when the gates land.

**The shape this wave met that tranche 1 did not, and it is a property of the
queue rather than of the tree.** `[89]` and `[120]` carry PORT-NOW labels for
commits that edit `_mamba_block_aligned_split` and `shift_computed_tokens` —
both of which upstream **already had at the pin**. The tree's distance from those
commits is not the commit; it is a hole that predates the pin. A queue derived
from a commit range structurally cannot see this, so the cheapest check for it,
`git show 5559679229:<upstream path>`, is recorded in the report's §11 for the
remaining tranches.
