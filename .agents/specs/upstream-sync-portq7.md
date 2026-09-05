# Sync cycle `e126687a9a`, wave PORTQ-7 — the last tranche

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, re-verified at `bb2da6f97` against a positive control
(`KV-MAMBA-ALIGN`, which resolves in four of the nine files probed).
`scripts/check-agent-record.py` passes on a `Row:` line whether or not the row
resolves, so this note is here to stop a reader taking it for a matrix
reference. The issues this wave cites are carried under `## Owed` below.
Issue: [#2717](https://github.com/mudler/vllm.cpp/issues/2717).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq5.md`](upstream-sync-portq5.md),
whose method this wave continues.

## Now

**Done. All fifty reached; none left unclassified.** 50 of 50 re-derived:
**4 ALREADY_SATISFIED, 10 REAL_GAP, 36 NOT_APPLICABLE** (32 `surface-absent`,
4 `inert`). **40 of the 50 disagree with the recorded PORT-NOW disposition.**

**The queue ends exactly at 290**, at `e126687a9a` — the sync target itself —
with position 291 empty, re-derived at the merge commit and byte-identical to
the pre-annotation derivation. This tranche is the queue's last slice, but the
queue is **not** fully counted here: PORTQ-6 ran 201-240 concurrently and its
counts are not in this record, so what is known is **55 real gaps in 250
entries** (the landed tranches' 45 across 1-200 plus this wave's 10), not a
290-entry total.

Ten issues carry the ten gaps, all new:
[#2724](https://github.com/mudler/vllm.cpp/issues/2724),
[#2725](https://github.com/mudler/vllm.cpp/issues/2725),
[#2727](https://github.com/mudler/vllm.cpp/issues/2727),
[#2728](https://github.com/mudler/vllm.cpp/issues/2728),
[#2729](https://github.com/mudler/vllm.cpp/issues/2729),
[#2731](https://github.com/mudler/vllm.cpp/issues/2731),
[#2733](https://github.com/mudler/vllm.cpp/issues/2733),
[#2734](https://github.com/mudler/vllm.cpp/issues/2734),
[#2735](https://github.com/mudler/vllm.cpp/issues/2735),
[#2736](https://github.com/mudler/vllm.cpp/issues/2736). The four `inert`
entries, two unreached surfaces and one flagged coordinator divergence are
deferred under [#2737](https://github.com/mudler/vllm.cpp/issues/2737).

**The pairing check found what a single tranche cannot see**: entries 229 and
241 are byte-identical changes, the queue's only duplicate, with KV-layout stage
6 clobbering the first between them; six further supersession pairs follow, two
of them amendments no scan can detect.

**One landed record is corrected in place** — `2026-09-03-portq5.md` §5.3 says
stage 6 is outside the range and it is inside, at position 230 — and
[#2695](https://github.com/mudler/vllm.cpp/issues/2695) is commented with the
same correction. **Four citations in this wave's own reading were wrong**, found
by a mechanical anchor resolve; none moved a label. **Two labels were changed
from what the readers returned**, `inert` to `surface-absent`.
Report: [`../sync/2026-09-03-portq7.md`](../sync/2026-09-03-portq7.md).

**Nothing in this wave is executed.** No build, no test run, no GPU, no lease.

The pin does **not** advance and nothing read here is a reason to move it. The
active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked fifty times.** For each of PORT-NOW entries 241 to 290 of
`5559679229..e126687a9a` in upstream commit order — the last fifty of the queue
— what is true **of this tree**?

`.agents/sync/2026-09-02-e126687.md` §4 says in as many words that **no
disposition in it is re-derived**: all 1002 are carried by SHA from
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md). A `PORT-NOW`
label is a prior wave's reading of an upstream diff. It is evidence about
upstream and says nothing about what this tree has.

Five tranches have measured that difference: **155 of 200 recorded labels did not
hold**, and the record was accurate about upstream in essentially every case.
This wave asks the question for the remaining fifty in its range.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 and 1465 counts and all five landed tranches' own positions
   before anything is read off it. **Show that the queue ends exactly at 290.**
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches, scope and positive control that found
   nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. **A pairing check over the whole 290, not only this tranche.** PORTQ-5 found
   that entry 200 reverts entry 45 — net upstream delta zero, structurally
   invisible from inside any single forty. As the tranche that sees the queue
   complete, this wave runs that check globally and reports every pairing it
   finds, in its own range or elsewhere.
5. One issue per real gap, naming a row that **exists**.
6. Any source record this wave falsifies, corrected in place by annotation.

Out of scope: porting anything, advancing the pin, entries outside 241-290, and
any measured number. Nothing in this wave is executed.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l          # 1465
$ comm -12 portnow.txt range.txt | wc -l                                      # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow.txt | sed -n '241,290p'
```

`portnow.txt` is the 315 SHAs of `../sync/2026-09-01-cdefd9d.md` §4, extracted
**by pattern** — `awk` between the `## 4.` and `## 5.` headers, then the
`^- \`<sha>\` vllm#` line shape. PORTQ-5 §6.5 measured what a hard-coded line
range now does: the stale `sed -n '205,563p'` under-reads by one line and
silently drops a real SHA. A bare 10-hex scan over the same span yields **317**,
because two SHAs appear in §4's prose preamble; the line shape is what makes the
extraction 315.

**The ordering is checked positionally against all five landed tables and in
absolute time.** Positions 1/9/21/40 (PORTQ-1), 41/60/80 (PORTQ-2),
81/95/116/120 (PORTQ-3), 121/140/160 (PORTQ-4) and 161/180/200 (PORTQ-5) must
each match the SHA that tranche published. The range contains **zero merges** and
`%ct` must be strictly monotone across all 290. Rendered local dates (`%cd`)
invert without meaning and are never compared.

**And the slice must end exactly at 290.** If it does not, the queue's size is
wrong, which is a larger finding than any entry, and the wave stops.

### 2.2 Three labels, and a named sub-shape for the third

`ALREADY_SATISFIED` cites the `file:line` or symbol that satisfies the entry, in
its **post**-commit shape. `REAL_GAP` says what is missing, roughly how big, and
who owns it. `NOT_APPLICABLE` splits:

- **`surface-absent`** — the commit edits something this tree does not carry.
- **`inert:<gate>`** — the surface **is** here and the new arm is gated on
  something unreachable in every configuration this tree builds. Deferred work,
  so the gate is named. Where half a commit has no surface at all, that half is
  `surface-absent`: flipping the gate would land on nothing.

### 2.3 Upstream is read at a revision, never in a working tree

Every upstream citation is `git show <rev>:<path>` or `git show <sha>` for the
diff, in `/home/mudler/_git/vllm`. An anchor read in a working tree is wrong at
the pin.

### 2.4 Absence is never concluded from one grep, and every zero carries a control

**Seven mechanisms have produced a false zero across the five landed tranches**,
and each reader was sent all seven: upstream's spelling versus this tree's; a
malformed ERE where `\|` under `-E` is a literal pipe; an unquoted
`--include=*.h` eaten by zsh; source text wrapping across a line break; `grep`
here wrapping `ugrep`, which **without `-r` skips a directory argument silently
and exits 1** — the canonical not-found rc, where GNU grep would exit 2 with a
diagnostic; `-F` not doing alternation, so `grep -rnF 'a\|b'` searches for one
literal string containing a backslash; and an unquoted `"$VAR"` holding several
paths, which zsh does not word-split.

So: each entry is searched under at least two spellings and in at least two
locations; the searches are written beside the claim; **every zero is paired with
a positive control run through the identical probe form**; and every zero states
its scope.

### 2.5 Git is asked whether it already triaged the entry, and whether the hole predates the pin

`git log --oneline --grep '<sha>'`, `--grep 'vllm#<PR>'`, `git log -S'<symbol>'`
and a grep of `.agents/` for the SHA, before any novelty is claimed. Then
PORTQ-3's cheaper check, `git show 5559679229:<upstream path>`: a queue derived
from a commit range cannot structurally see a hole that predates its own base.

### 2.6 The whole-queue pairing check

Two scans, because they find different shapes and both are needed.

**Exact inverse.** For every ordered pair in the 290, compare the sign-swapped
sorted line multisets of `git show -U0`. A pair where A's additions equal B's
deletions and A's deletions equal B's additions is an exact revert whose net
contribution to the range is zero. This reproduces PORTQ-5's `[45]`/`[200]`
finding and is the strictest test.

**Directional supersession.** For every ordered pair, compute how much of A's
added content B later deletes, over substantive lines only (blank, comment and
short lines dropped). A ratio at or near 1.0 means the later commit removed what
the earlier one added, whether or not it is a formal revert.

Neither scan sees an **amendment** — a later commit that narrows the earlier one
by adding a condition rather than by deleting its lines. Those are found by
reading the diffs of entries that touch the same file, and this wave reports the
ones it read.

### 2.7 A falsified source record is corrected where it is written

A correction **annotates and does not rewrite**, keeping the original text and
its date, citing the prior finding, and must not perturb the source record's SHA
extraction — `.agents/sync/2026-09-01-cdefd9d.md` §4 must still yield 315 by the
pattern extraction of §2.1.

### 2.8 The reading is delegated, and its citations are re-checked

Seven fresh readers take four to ten entries each, clustered by surface. The wave
operator prints every citation again and resolves it against the tree before it
is written down. PORTQ-5's one false label was not a bad grep but an
over-generalisation from four correctly-read sites to a fifth that behaves
differently, so a claim of the form "this port always does X" is expanded into
its enumeration before it is accepted.

## 3. Risks

- **A carried label is anchoring.** Mitigated by treating a disagreement as the
  expected result — five tranches running, it is the outcome in 78% of entries.
- **A false `ALREADY_SATISFIED` is more dangerous than a false `REAL_GAP`.** A
  gap gets an issue and a reader; a satisfied entry closes the question and
  nobody re-reads it. Mitigated by §2.8's enumeration rule and by the operator
  re-reading every satisfied claim rather than only every gap.
- **Fifty is more than any predecessor carried.** Mitigated by saying exactly
  where the reading stopped rather than thinning every entry. A partial tranche
  with honest boundaries beats fifty shallow readings.
- **A named owner row may not exist.** `check-agent-record.py` passes on a `Row:`
  line whether or not the row is real. Mitigated by resolving every owner by line
  number before its issue is filed.
- **A pairing scan's negative is weak.** The exact-inverse test finds only formal
  reverts and the directional test only deletions; neither sees an amendment.
  Mitigated by stating that limitation beside the result rather than reporting a
  clean scan as "no other pairs exist".
- **Fifty is not 290.** Any rate computed here is an extrapolation from a
  contiguous, non-random slice. Recorded as an estimate with its reasons inline.
- **A sibling wave is live.** PORTQ-6 works entries 201-240 concurrently.
- **Disk.** The host is at 97%, 16 GB free. Nothing in this wave builds; a
  worktree that compiled would be the defect.

## 4. Gates

- `git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l` prints
  `1465`, and the intersection with the 315 prints `290`. The derivation is not
  used until both reproduce.
- The §4 extraction of `../sync/2026-09-01-cdefd9d.md`, by pattern, prints `315`
  — before and after any annotation this wave writes.
- Positions 1/9/21/40, 41/60/80, 81/95/116/120, 121/140/160 and 161/180/200 of
  the derived order match the SHAs the five landed tranches publish, and
  **position 290 is the last**.
- `git rev-list --merges 5559679229..e126687a9a | wc -l` prints `0`, and `%ct` is
  strictly increasing across all 290 derived entries.
- Every `path:line` in the report resolves in the tree at the merge commit, and
  every citation past the pin names its revision.
- Every owner row named by an issue resolves in `roadmap_v1.md` or a
  `*-matrix.md`, or the issue says plainly that none does.
- `scripts/agent-preflight.sh --staged`, read by grepping ANSI-stripped output
  for `gate(s) failed` and `NOT a green` **on a log that reached its terminal
  summary**, never by its exit code and never on a truncated log.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at 50, and say where the reading stopped if it did not reach all fifty.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing read here is a reason to move it.
- Stop before entries 201-240. PORTQ-6 owns them.

## Owed

- Advancing the pin, and whatever the completed queue then implies
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. They are classified here and implemented
  elsewhere; each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave reads and does not discharge.
