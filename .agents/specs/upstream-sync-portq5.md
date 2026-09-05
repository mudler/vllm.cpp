# Sync cycle `e126687a9a`, wave PORTQ-5

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, verified again at `a700e8da6`. `scripts/check-agent-record.py`
passes on a `Row:` line whether or not the row resolves, so this note is here to
stop a reader taking it for a matrix reference. The issues this wave cites are
carried under `## Owed` below, which is the route AGENTS.md gives when no row
owns the work.
Issue: [#2679](https://github.com/mudler/vllm.cpp/issues/2679).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq3.md`](upstream-sync-portq3.md),
whose method this wave continues.

## Now

**Done, and repaired after a fresh review returned NOT PASS.** 40 of 40
re-derived: **2 ALREADY_SATISFIED, 11 REAL_GAP, 27 NOT_APPLICABLE**
(26 `surface-absent`, 1 `inert`). **29 of the 40 disagree with the recorded
PORT-NOW disposition.**

The review falsified one label and two claims, all corrected in place in the
report rather than quietly repaired: `[178]` was published ALREADY_SATISFIED and
is a live HTTP 500 on `/tokenize` (§6.6, now [#2707](https://github.com/mudler/vllm.cpp/issues/2707));
the stale-`sed` hazard was stated as an over-read to 343 and is an **under**-read
to 314 that silently drops a SHA (§6.5); and the pull-request body's compressed
"byte-identical across every path" was false, where 4 of 15 paths are identical
and the exact claim is a line-level inverse (§5.2). The gate run this record had
declared but never published is now §8b, including an incomplete-log misread.

Eleven issues carry the eleven gaps: ten new
([#2686](https://github.com/mudler/vllm.cpp/issues/2686),
[#2687](https://github.com/mudler/vllm.cpp/issues/2687),
[#2690](https://github.com/mudler/vllm.cpp/issues/2690),
[#2691](https://github.com/mudler/vllm.cpp/issues/2691),
[#2692](https://github.com/mudler/vllm.cpp/issues/2692),
[#2695](https://github.com/mudler/vllm.cpp/issues/2695),
[#2698](https://github.com/mudler/vllm.cpp/issues/2698),
[#2699](https://github.com/mudler/vllm.cpp/issues/2699),
[#2701](https://github.com/mudler/vllm.cpp/issues/2701),
[#2707](https://github.com/mudler/vllm.cpp/issues/2707)) and the pre-existing
[#2657](https://github.com/mudler/vllm.cpp/issues/2657) for `[162]`. The one
`inert` entry is deferred under `.agents/feature-matrix.md:140`, which already
owns its gate.
Report: [`../sync/2026-09-03-portq5.md`](../sync/2026-09-03-portq5.md).

**Nothing in this wave is executed.** No build, no test run, no GPU, no lease.

The pin does **not** advance and nothing read here is a reason to move it. The
active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked forty times.** For each of PORT-NOW entries 161 to 200 of
`5559679229..e126687a9a` in upstream commit order, what is true **of this tree**?

`.agents/sync/2026-09-02-e126687.md` §4 says in as many words that **no
disposition in it is re-derived**: all 1002 are carried by SHA from
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md). A `PORT-NOW`
label is a prior wave's reading of an upstream diff. It is evidence about
upstream and says nothing about what this tree has.

Three tranches have now measured that difference and published the numbers:
**97 of 120 recorded labels did not hold**, and the record was accurate about
upstream in essentially every case. This wave asks the question for forty more.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 and 1465 counts, and the landed tranches' own positions, before
   anything is read off it.
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches and positive control that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming a row that **exists**.
5. Any source record this wave falsifies, corrected in place by annotation.

Out of scope: porting anything, advancing the pin, entries outside 161-200, and
any measured number. Nothing in this wave is executed.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l          # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                              # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | sed -n '161,200p'
```

`portnow.txt` is the 315 SHAs of `../sync/2026-09-01-cdefd9d.md` §4. The `1465`
and the `290` are the counts `../sync/2026-09-02-e126687.md` §4 publishes, so the
extraction reproduces a committed number before it is used.

**The ordering is checked twice, positionally and in absolute time.**
Positionally: positions 1, 9, 21, 40 of the derived order are `3f1d40960f`,
`bf2b45b5d6`, `0b6aa3c47c`, `b49eaf205a`, exactly what
[`../sync/2026-09-03-portq1.md`](../sync/2026-09-03-portq1.md) §3 names; 41, 60,
80 land in PORTQ-2's table and 81, 95, 116, 120 in PORTQ-3's. The three landed
tables are contiguous, disjoint and correctly ordered against this derivation, so
the slice at 161-200 is the same extraction they were.

In absolute time: the range contains **zero merges** and committer timestamps
(`%ct`) are **strictly monotone across all 290** entries, checked entry by entry.
This is PORTQ-3's retracted-and-repaired check. Rendered local dates (`%cd`)
invert without meaning and must never be compared; PORTQ-3's first draft read two
such inversions as merge topology in a range that has none.

If either denominator, the positional check or the monotonicity had failed, this
wave would have stopped, because that would have invalidated three landed
tranches as well as this one.

### 2.2 Three labels, and a named sub-shape for the third

`ALREADY_SATISFIED` cites the `file:line` or symbol that satisfies the entry.
`REAL_GAP` says what is missing, roughly how big, and who owns it.
`NOT_APPLICABLE` splits, because the two halves have different futures:

- **`surface-absent`** — the commit edits something this tree does not carry.
  Discarded work; there is nothing to port until the underlying feature is.
- **`inert:<gate>`** — the surface is here and the new arm is gated on something
  unreachable in every configuration this tree builds. **Deferred** work: it
  becomes real the day that gate is ported, so the gate is named.

`inert` requires the surface to actually be present. Where half a commit has no
surface at all, that half is `surface-absent`: flipping the gate would land on
nothing.

### 2.3 Upstream is read at a revision, never in a working tree

Every upstream citation is `git -C ${VLLM_SOURCE} show <rev>:<path>` or
`git show <sha>` for the diff. An anchor read in a working tree is wrong at the
pin.

### 2.4 Absence is never concluded from one grep, and every zero carries a control

Four distinct mechanisms produced false zeros across the first three tranches:
grepping upstream's identifier where this tree renames on the way in
(`g_slot_map` for `_slot_mapping_buffer`); a malformed ERE, where
`grep -rniE "a\|b"` reads `\|` as a literal pipe; an unquoted `--include=*.h`
eaten by zsh, returning silence at rc 0; and source text wrapping across a line
break so the pattern cannot match.

So: each entry is searched under at least two spellings and in at least two
locations before anything is called missing; the searches are written down beside
the claim; **every zero is paired with a positive control run through the same
probe form**, pointed at something known to exist; and every zero states its
scope, because an unstated scope makes a zero unfalsifiable.

### 2.5 Git is asked whether it already triaged the entry, and whether the hole predates the pin

`git log --oneline --grep '<sha>'`, `git log --grep '<upstream PR number>'`,
`git log -S'<symbol>'`, and a grep of `.agents/` for the SHA, before any novelty
is claimed. Five of these forty have such a hit and each is read and cited.

Then PORTQ-3's cheaper check: `git show 5559679229:<upstream path>`. PORTQ-3
found entries whose surfaces upstream **already had at the pin**, so the tree's
distance is not the commit at all. A queue derived from a commit range cannot
structurally see that shape.

### 2.6 A falsified source record is corrected where it is written

PORTQ-3 established this: the same false carried claim was independently
rediscovered three times because every wave wrote a new report and none edited
the one that misled it. A correction **annotates and does not rewrite**, keeping
the original text and its date, and must not perturb the source record's SHA
extraction — `.agents/sync/2026-09-01-cdefd9d.md` §4 must still yield 315.

### 2.7 The reading is delegated, and its citations are re-checked

Six fresh readers take six or seven entries each, clustered by surface so a
reader builds context once. The wave operator prints the citations again and
resolves each against the tree before it is written down. One citation in twenty
was wrong in PORTQ-3; the re-check is what caught it.

## 3. Risks

- **A carried label is anchoring.** Reading the record's sentence before the diff
  invites confirming it. Mitigated by treating a disagreement as the expected
  result — three tranches running, it is the outcome in 81% of entries.
- **`inert` is a judgement about reachability, not a grep.** A gate called
  unreachable that is in fact reachable turns a real gap into a deferral.
  Mitigated by naming the gate so a later reader can falsify it.
- **A named owner row may not exist.** `check-agent-record.py` passes on a `Row:`
  line whether or not the row is real. Mitigated by checking every owner against
  `roadmap_v1.md` and the matrices before the issue is filed.
- **Forty is not 290.** Any rate computed here is an extrapolation from a
  contiguous, non-random slice, not a count. Recorded as an estimate beside the
  number, with its reasons inline.
- **A sibling wave is live.** PORTQ-4 works entries 121-160 concurrently. This
  wave writes its own report and its own spec, and the one shared record it
  annotates is `2026-09-01-cdefd9d.md` §4, per-entry and append-shaped.
- **Disk.** The host is at 96%, 21 GB free. Nothing in this wave builds; a
  worktree that compiled would be the defect.

## 4. Gates

- `git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l` prints
  `1465`, and `comm -12 <(sort portnow.txt) range.txt | wc -l` prints `290`. The
  derivation is not used until both reproduce.
- Positions 1, 9, 21, 40 of the derived order match the SHAs
  `.agents/sync/2026-09-03-portq1.md` §3 publishes; 41, 60, 80 resolve in
  PORTQ-2's table and 81, 95, 116, 120 in PORTQ-3's.
- `git rev-list --merges 5559679229..e126687a9a | wc -l` prints `0`, and `%ct` is
  strictly increasing across all 290 derived entries.
- Every `path:line` in the report resolves in the tree at the merge commit, and
  every citation past the pin names its revision.
- Every owner row named by an issue resolves in `roadmap_v1.md` or a
  `*-matrix.md`, or the issue says plainly that none does.
- After any annotation of `.agents/sync/2026-09-01-cdefd9d.md`, its §4 SHA
  extraction still yields **315**.
- `scripts/agent-preflight.sh --staged`, read by grepping its output for
  `gate(s) failed` and `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at 40. The tranche is bounded; finishing the queue is not this wave's.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing read here is a reason to move it.
- Stop before entries 121-160. PORTQ-4 owns them.

## Owed

- The remaining PORT-NOW entries of `5559679229..e126687a9a` outside 1-120 and
  161-200 ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. They are classified here and implemented
  elsewhere; each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave reads and does not discharge.
