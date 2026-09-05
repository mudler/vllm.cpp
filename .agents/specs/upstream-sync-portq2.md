# Sync cycle `e126687a9a`, wave PORTQ-2

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, as [`upstream-sync-portq1.md`](upstream-sync-portq1.md) records.
The issues this wave cites are carried under `## Owed` below, which is the route
AGENTS.md gives when no row owns the work.
Issue: [#2646](https://github.com/mudler/vllm.cpp/issues/2646).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [`upstream-sync-portq1.md`](upstream-sync-portq1.md) (#2632, landed
as #2642), which did entries 1-40 of the same queue.
Parent: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns the
290-entry queue.

## Now

**Done.** 40 of 40 re-derived: **4 ALREADY_SATISFIED, 9 REAL_GAP,
27 NOT_APPLICABLE** (23 `surface-absent`, 4 `inert`). **31 of the 40 disagree
with the recorded PORT-NOW disposition.** Thirteen issues carry the work: nine
for the gaps, four for the deferred entries. **Nothing was executed** — no build,
no test run, no GPU — so every label is a static reading of source and
`ALREADY_SATISFIED` means the code implements the behaviour, not that a gate
observed it.
Report: [`../sync/2026-09-03-portq2.md`](../sync/2026-09-03-portq2.md).

The pin did **not** advance and nothing read here is a reason to move it. The
active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked forty times.** For each of PORT-NOW entries **41 through
80** of `5559679229..e126687a9a` in upstream commit order, what is true **of this
tree**?

Advancing the parity pin is the last unmet clause of the cycle, and AGENTS.md
permits it only "after you reconcile every affected row and gate". The queue is
what remains to reconcile. PORTQ-1 took the first 40; this wave takes the next 40
and does not attempt more. A sibling wave, PORTQ-3, takes 81-120 concurrently and
this wave stays off its range and its record surfaces.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 / 1465 counts before using it.
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming an owning row **that exists**.

Out of scope:

- **Porting.** Classification is the deliverable.
- **Advancing the pin**, and the remaining 210 entries.
- Any throughput, latency or memory number. Nothing here is executed, so nothing
  here may be quoted as a speed axis.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u > range.txt   # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                          # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | sed -n '41,80p'                   # the tranche
```

`portnow.txt` is the 315 SHAs of
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md) §4. Both counts
are the ones `../sync/2026-09-02-e126687.md` §4 publishes, so the extraction is
verified against a committed number before anything is read off it. The
tranche's first and 40th entries reproduce PORTQ-1's published endpoints
(`3f1d40960f`, `b49eaf205a`) at offsets 1 and 40, which is the second check that
the ordering is the same one PORTQ-1 used.

### 2.2 Three labels, and a named sub-shape for the third

Identical to PORTQ-1 §2.2, deliberately, so the two tranches compose:
`ALREADY_SATISFIED`, `REAL_GAP`, `NOT_APPLICABLE` split into `surface-absent`
(discarded work) and `inert:<gate>` (deferred work, gate named).

### 2.3 Upstream is read at a revision, never in a working tree

Every read is `git -C /home/mudler/_git/vllm show <sha>:<path>` or
`git show <sha>`. An anchor read from a working tree is an anchor at whatever
that tree is checked out to.

### 2.4 Absence is never concluded from one grep, and the instrument is different

PORTQ-1's retraction is this wave's binding lesson. Re-reading a citation cannot
catch a false claim of absence, because absence has no citation to re-read. So a
missing symbol is reported only after at least two spellings and two locations,
**searched for the concept in this tree's vocabulary rather than upstream's**,
and the report says which searches were tried and returned nothing.

### 2.5 Git is asked before novelty is claimed

`git log --oneline --grep '<sha>'` and `--grep 'vllm#<PR>'`, plus a grep of
`.agents/` for both, before any entry is called newly read. PORTQ-1 found three
of its forty already triaged by `b7e414cc4`, one of those readings stale. This
tranche's first pass found four entries already re-derived against the tree by
#2524's stratified sample, recorded in `../sync/2026-09-01-cdefd9d.md` §13; each
is re-verified at this tree rather than carried.

### 2.6 The reading is delegated, and its citations are re-checked

Four fresh readers take ten entries each from one shared brief. Citations are
printed again by this wave's operator before they are written down.

## 3. Risks

- **A carried label is anchoring.** Mitigated by treating disagreement as the
  expected result.
- **`inert` is a judgement about reachability, not a grep.** Mitigated by naming
  the gate so a later reader can falsify it.
- **A prior REAL_GAP can be stale in either direction.** #2524's §13 anchors were
  taken at `63889449c`; the tree has moved. Each is re-verified here.
- **Forty is not 290.** Any rate computed here is an extrapolation from a
  non-random contiguous slice, recorded as an estimate beside the number.
- **Disk.** The host is at 95%. Nothing in this wave builds; a worktree that
  compiled would be the defect.

## 4. Gates

- `git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l` prints
  `1465` and the PORT-NOW intersection prints `290`. The derivation is not used
  until both reproduce.
- Every `path:line` in the report resolves in the tree at the merge commit.
- Every row named as an owner resolves in `.agents/roadmap_v1.md` or a
  `.agents/*-matrix.md`.
- `scripts/agent-preflight.sh --staged`, read by grepping its output for
  `gate(s) failed` and `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at entry 80. Entries 81-120 belong to PORTQ-3.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing read here is a reason to move it.

## Owed

- The remaining 210 PORT-NOW entries of `5559679229..e126687a9a`
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. Each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave re-verifies for the four items inside this tranche and does not
  discharge.
- The nine real gaps, each on its own issue with an owning row that resolves in
  `roadmap_v1.md` or a `*-matrix.md`:
  [#2656](https://github.com/mudler/vllm.cpp/issues/2656) `ENG-RUNNER-MODELSHAPE`,
  [#2657](https://github.com/mudler/vllm.cpp/issues/2657) `SPEC-DSPARK-BLOCK-SIZE-GUARD`,
  [#2658](https://github.com/mudler/vllm.cpp/issues/2658) `SERVE-COMPLETION-LONGTAIL`,
  [#2659](https://github.com/mudler/vllm.cpp/issues/2659) `QUANT-FP8-GENERIC`,
  [#2660](https://github.com/mudler/vllm.cpp/issues/2660) `KV-MAMBA-ALIGN`,
  [#2661](https://github.com/mudler/vllm.cpp/issues/2661) `QUANT-GGUF-CPU-THREADPOOL`,
  [#2662](https://github.com/mudler/vllm.cpp/issues/2662) `KV-OFFLOAD`,
  [#2663](https://github.com/mudler/vllm.cpp/issues/2663) `KV-PREFIX-MATCH-UNIT`,
  [#2664](https://github.com/mudler/vllm.cpp/issues/2664) `LOAD-CONFIG-SURFACE`.
- The four deferred (`inert`) entries, each with its gate named and an owner:
  [#2665](https://github.com/mudler/vllm.cpp/issues/2665) `ENG-MM-INPUT-PIPELINE`,
  [#2666](https://github.com/mudler/vllm.cpp/issues/2666) `KERNEL-ATTN-MLA-SPARSE`,
  [#2667](https://github.com/mudler/vllm.cpp/issues/2667) `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`,
  [#2668](https://github.com/mudler/vllm.cpp/issues/2668) `SPEC-ACCEPT-VARIANTS`.
- **Two committed records that the PIN ADVANCE will falsify**, correct today and
  therefore not edited here: `.agents/porting.md:78` (ModelOpt fp8 `out_dtype`),
  and `calculate_kv_scales`'s `config/cache.py:111` anchor in SIX places, one of
  them inside a thrown message a user can read
  (`include/vllm/model_executor/layers/quantization/kv_cache.h:108`). Report §9
  enumerates all six. A third item claimed here in an earlier draft — the DFlash2
  draft-logits layout — was **false and is retracted in report §9.1**: DFlash2
  keeps `(torch.float32, -inf)` at the advance target, so that record is correct
  and nothing is owed.

## Outcome

**What was measured.** The tranche derivation reproduced both published counts
(`1465` range commits, `290` PORT-NOW) before any entry was read, and offsets 1
and 40 of the same ordering reproduced PORTQ-1's published endpoints
`3f1d40960f` and `b49eaf205a` — a third check that this tranche continues that
one rather than a differently-ordered list. This wave's operator independently
derived sixteen of the forty before the readers returned, and re-read every
`file:line` behind all nine real gaps and both prior-triage re-verifications. No
citation named a symbol that does not exist; one anchor was off by one line and
is corrected in the report.

**What the method caught in itself.** An operator probe used
`grep -rniE "a\|b"`, where ERE reads `\|` as a LITERAL pipe, and it returned `0`
for a symbol this tree plainly carries. It was caught only because the zero
contradicted a citation PORTQ-1 had published. A zero from a malformed pattern is
indistinguishable from a zero from an absent symbol, which is PORTQ-1's
retraction in a new costume: the instrument, not the tree, produced the absence.

**What was rejected, and why.** Treating 9/40 as *the* rate for the remaining
210. It is higher than PORTQ-1's 6/40 and #2524's 11/63, and all three are
contiguous or overlapping populations rather than independent draws — four of
this tranche's nine gaps ARE #2524's own §13 items. The report states 45-60 as an
estimate from three samples and gives all four reasons not to trust the point
value, including that three of the nine are `yes (part)`.

**One label was adjudicated against its reader.** `[41]` came back `REAL_GAP` at
~150-250 lines "dominated by vendoring SHA-2". A port whose bulk is work the
commit does not contain, guarded by an algorithm this tree does not implement, is
`inert` under §2.2's own definition. The disagreement is recorded in the report
rather than absorbed, because the reader's facts were right and only the
label boundary was at issue.

**What the fresh review corrected, and what it did not.** Twenty of twenty
dispositions it re-derived held, and none of the six findings moved a label. Four
were in the hand-off sections, where they would have misled the wave that acts on
them: a false staleness claim about the DFlash2 draft-logits record that had
already reached an issue body (report §9.1, retracted, and #2668 corrected in the
same change); an undercount of the `config/cache.py:111` citations that omitted
the instance in product output; a reversed precedence gloss in §6.1, where
upstream reads `block_size` first and this tree reads `dspark_block_size` first;
and an over-scoped `ALREADY_SATISFIED` on `[77]`, which holds for the
index-range defect the commit fixes and not for its NaN-selection side effect.
One cell, `[46]`, argued from a missing symbol where a positive citation was
available and stronger — this spec's own §2.4 standard, unmet in exactly one
place. Three zero-hit lists left their `include/`+`src/` scope unstated.

**The instrument rule, promoted from an incident to a rule.** §2.4's guard is now
stated as: pair every zero with a positive control run through the SAME probe
form. This wave produced one malformed probe (`grep -rniE "a\|b"`, where ERE
reads `\|` as a literal pipe); the review of it produced two more with a
different mechanism (an unquoted `--include=*.h` eaten by zsh globbing, silent at
rc 0) and caught both by the control. The incident recurs; the control is what
does not.

**The finding worth carrying to the next tranche.** Reading the commit is not
enough; read the TARGET HEAD. `[55]`'s commit deletes a DSpark refusal, and
carry-over C9 says a later commit re-lands the removal — but at `e126687a9a` a
NARROWER, Qwen3-Omni-scoped equality check stands in its place, reading
`block_size` before `dspark_block_size`, which is this tree's own widened
fallback order arrived at independently. Sizing that deletion from the commit
alone would have produced the wrong unit of work.
