# Sync cycle `e126687a9a`, wave PORTQ-1

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor wave's spec.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`. The issues this wave cites are carried under `## Owed` below,
which is the route AGENTS.md gives when no row owns the work.
Issue: [#2632](https://github.com/mudler/vllm.cpp/issues/2632).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue this wave takes its first tranche from, and
[`upstream-sync-headpin-runhalf.md`](upstream-sync-headpin-runhalf.md), whose
`## Owed` names that queue as its first item.

## Now

**Done, and corrected once after review.** 40 of 40 re-derived:
**9 ALREADY_SATISFIED, 6 REAL_GAP, 25 NOT_APPLICABLE** (22 `surface-absent`,
3 `inert`). **34 of the 40 disagree
with the recorded PORT-NOW disposition.** Six issues carry the gaps; one of the
six ([31], [#2531](https://github.com/mudler/vllm.cpp/issues/2531)) is a live
defect on a reachable path and the rest are ports or mirror repairs.
Report: [`../sync/2026-09-03-portq1.md`](../sync/2026-09-03-portq1.md).

The pin did **not** advance and nothing measured here is a reason to move it.
The active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked forty times.** For each of the first 40 PORT-NOW entries of
`5559679229..e126687a9a` in upstream commit order, what is true **of this tree**?

Advancing the pin is the last unmet clause of the cycle, and AGENTS.md permits it
only "after you reconcile every affected row and gate". The queue is what remains
to reconcile. It is 290 entries; this wave takes 40 and does not attempt more.

**Why the queue cannot be read off the record.**
[`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md) §4 says in as
many words that **no disposition in it is re-derived**: all 1002 are carried by
SHA from [`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md),
including the 202 that report carried unread. A `PORT-NOW` label is a prior
wave's reading of an upstream diff. It is evidence about upstream and says
nothing about what this tree has.

#2524 measured the size of that difference on a stratified sample of 63 of the
315 entries, re-derived against the tree by seven readers: **11 REAL_GAP,
12 ALREADY_SATISFIED, 40 NOT_APPLICABLE, 0 UNCERTAIN**. `021b7d985b` is the
worked counter-example already on record — its report claim reproduced as
**false**, because `include/vllm/model_executor/models/decode_graph_sizes.h`
bounds its candidate set by `max_num_seqs` and applies no 512 cap at all.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 / 1465 counts before using it.
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming the row that owns it.

Out of scope:

- **Porting.** Classification is the deliverable. AGENTS.md forbids mixing
  feature work into a sync record, and a real gap earns an issue, not a patch.
- **Advancing the pin**, and the remaining 250 entries.
- Any throughput, latency or memory number. Nothing here runs on a GPU, so
  nothing here may be quoted as a speed axis.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u > range.txt   # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                          # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | head -40                          # the tranche
```

`portnow.txt` is the 315 SHAs of
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md) §4. The 1465 and
the 290 are the counts `../sync/2026-09-02-e126687.md` §4 records, so the
extraction is verified against a published number before anything is read off it.
Commit order, not the record's SHA order: `--reverse` yields oldest first, and the
40 commit dates come out monotone, which is the cheap check that the order is the
one intended.

### 2.2 Three labels, and a named sub-shape for the third

`ALREADY_SATISFIED`, `REAL_GAP`, `NOT_APPLICABLE`. #2524 found the third is where
the queue's mass sits, and that it hides two different futures, so it is split:

- `surface-absent` — the commit edits something this tree deliberately does not
  carry. There is nothing to port until the underlying feature is. Discarded work.
- `inert:<gate>` — the surface is here, and the new arm is gated on a backend or a
  spec field unreachable in every configuration this tree can build. **Deferred**
  work: it becomes real the day that gate is ported, so the gate is named.

### 2.3 Upstream is read at a revision, never in a working tree

Every read is `git -C /home/mudler/_git/vllm show <sha>:<path>` or
`git show <sha> -- <path>`. An anchor read from a working tree is an anchor at
whatever that tree happens to be checked out to, which is the failure
[`../porting.md`](../porting.md) names. Every citation past `5559679229` is a
forward reference and carries its revision.

### 2.4 Absence is never concluded from one grep

A false `REAL_GAP` costs a porting wave; a false `ALREADY_SATISFIED` hides one.
So a missing symbol is reported only after at least two spellings and two
locations, and the report says which were tried. This tree renames on the way in
— vLLM's `snake_case` module attribute becomes a C++ member or a `vt::` seam — so
one failed grep on the upstream name is the expected result even when the
behavior is present.

### 2.5 The reading is delegated, and its citations are re-checked

Four fresh readers take ten entries each from one shared brief. Every `path:line`
they return is printed again here before it is written down. Subagent reports on
this row have previously named symbols that do not exist, so the citations are
evidence only once they have been re-read.

## 3. Risks

- **A carried label is anchoring.** Reading the record's sentence before the diff
  invites confirming it. Mitigated by reading the upstream diff first and the
  record's sentence last, and by treating a disagreement as the expected result
  rather than an error.
- **`inert` is a judgement about reachability, not a grep.** A gate called
  unreachable that is in fact reachable turns a real gap into a deferral.
  Mitigated by naming the gate so the claim is falsifiable by a later reader.
- **Forty is not 290.** Any rate computed here is an extrapolation from a
  non-random prefix, not a count. Recorded as such, beside the number.
- **Disk.** The host is at 95%. Nothing in this wave builds; a worktree that
  compiled would be the defect.

## 4. Gates

- `comm -12 <(sort portnow.txt) range.txt | wc -l` prints `290`, and
  `git rev-list 5559679229..e126687a9a | wc -l` prints `1465`. The derivation is
  not used until both reproduce.
- Every `path:line` in the report resolves in the tree at the merge commit, and
  every citation past the pin names its revision.
- `scripts/agent-preflight.sh --staged`, read by grepping its output for
  `gate(s) failed` and `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at 40. The tranche is bounded; finishing the queue is not this wave's.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing measured here is a reason to move it.

## Owed

- The remaining 250 PORT-NOW entries of `5559679229..e126687a9a`
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. They are classified here and
  implemented elsewhere; each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave reads and does not discharge, and the two live defects it found that
  are inside this tranche's range:
  [#2527](https://github.com/mudler/vllm.cpp/issues/2527) and
  [#2531](https://github.com/mudler/vllm.cpp/issues/2531).

## Outcome

**What review corrected, and it was this wave's own rule.** Entry [21]
`0b6aa3c47c` was first recorded `NOT_APPLICABLE` on the ground that "there is no
persistent buffer to under-size" and "no CUDA graph covers any draft propose".
**Both are false.** `src/vllm/model_executor/models/qwen3_dflash.cpp:1237-1262`
carries seven persistent device buffers and `:1533-1536` makes the captured draft
step default ON. The ground was six greps for UPSTREAM spellings
(`max_query_tokens`, `slot_mapping_buffer`, …) returning zero, read as absence —
the exact error §2.4 of this spec exists to prevent, committed by the wave that
wrote it. The label survives on a different and correct argument: `:1852-1867`
reallocates every buffer when the block width changes, so upstream's padding
under-allocation cannot arise. The entry moves to `inert`. Recorded rather than
quietly repaired, because the retraction is the more useful record.

**The lesson, stated so the next tranche inherits it.** §2.5's re-check of every
citation cannot catch a claim of ABSENCE, which has no citation to re-check.
Absence needs a different instrument: search for the CONCEPT under this tree's
naming, not for the upstream symbol. A fresh reviewer caught it; nothing in this
wave's own method would have.

**What was measured.** The tranche derivation reproduced both published counts
(`1465` range commits, `290` PORT-NOW) before any entry was read, and the 40
commit dates came out monotone. 13 citations, covering all six real gaps, six of
the nine `ALREADY_SATISFIED` and both `inert` calls, were re-read by this wave's
operator; none named a symbol that does not exist, and three anchors were
off by a few lines or named the wrong directory and are corrected in the report.

**What was rejected, and why.** Treating 6/40 as *the* rate for the remaining
250. It agrees closely with #2524's 11/63, which is the only reason it is quoted
at all, but this tranche is a contiguous oldest-first prefix rather than a random
draw, its population overlaps #2524's worked list, and the four batches returned
1/1/1/3 gaps — a spread where one entry moves the figure by a quarter. The report
states 40-50 as an estimate from two samples and says all three reasons out loud.

**Why `NOT_APPLICABLE` is split.** #2524 found the label carries most of the
queue's mass and hides two different futures. `surface-absent` is discarded work;
`inert` is deferred work that becomes real the day its gate is ported, so §2.2
requires the gate to be named. Both `inert` calls here ([33], [39]) name theirs,
which is what makes them falsifiable by a later reader instead of permanent.

**Why the reading was delegated to four readers and then re-checked.** Forty
entries at two spellings and two locations each is mechanical breadth, which
parallelises; judging a citation is not, and subagent reports on this row have
previously named symbols that do not exist. Splitting the two is why the
verification pass found nothing wrong rather than proving nothing.

**The finding worth carrying forward.** The 34 disagreements are three shapes,
not 34 mistakes, and the largest is Shape B: the surface is recorded *in this
repository* as deliberately absent, in a DEFERRED comment block or a matrix cell.
That text settled the majority of this tranche without opening a build or, in
several cases, an upstream diff. It is a minutes-long first pass that the 202
carried entries never received, and the next tranche should run it first.
