# Sync cycle `e126687a9a`, wave RECONCILE — the queue against itself, and the pin verdict

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the PORTQ waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, re-verified at `f5b5b2c3c` against a positive control
(`KV-MAMBA-ALIGN`, which resolves). `scripts/check-agent-record.py` passes on a
`Row:` line whether or not the row resolves, so this note is here to stop a
reader taking it for a matrix reference. The issues this wave cites are carried
under `## Owed` below.
Issue: [#2764](https://github.com/mudler/vllm.cpp/issues/2764).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq7.md`](upstream-sync-portq7.md),
whose §5 this wave reproduces, corrects and extends.

## Now

**Done. Two deliverables, and the second is a verdict, not a measurement.**

**The pin verdict.** Nothing about the PORT-NOW queue blocks the advance, and
three measured obligations do. `.agents/oracles/vllm.md:58-70` already names
four; this wave rules that item 1 of those four is satisfied by the record the
seven PORTQ waves built, and that items 2, 3 and 4 are not. The blockers are a
declared token-exact gate at the target, step-6 re-measurement of the four
denominators that move under it, and a reading on `dgx:gpu0`. `qwen4_exp` not
running at the candidate is **not** a blocker, and §6 of the report says why and
what it costs. The full argument is §2 and §5 of the report.

**The queue-against-itself pass.** All 290 entries compared with each other and
with the 1175 in-range commits that are **not** in the queue. PORTQ-7 §5 already
ran the intra-queue half; every one of its numbers reproduces here from an
independent extraction (§3.1). Three findings are new, and all three come from
axes PORTQ-7's two scans could not reach:

1. **`[9]` → `[175]` → `7156c63bef`**, where the third step is in range and
   **outside** the 315, classified `INVENTORY`. Net at the target is neither
   `[9]`'s gate nor `[175]`'s absence.
2. **`[185]`'s FlashInfer half is deleted by `d6c2fec9fd`**, also out of queue
   and classified `IGNORE`. `[185]` is filed `inert`, which says the work becomes
   real when its gate lands; half of it never becomes real.
3. **`[54]` → `[115]`**, a restoration. PORTQ-7's directional scan measures only
   "A's added lines that B deletes". The mirror — "A's deleted lines that B
   re-adds" — was never run, and it is the shape of the C9 chain. `[54]`→`[115]`
   scores 1 of 46 on the directional scan and 16 of 29 on the mirror.

**The brief for this wave called the queue-against-itself pass one nobody had
done. That is not accurate and the report says so in §4.1.** PORTQ-7 §5 ran two
of the four scans over all 290; PORTQ-6 §5 ran the out-of-queue check for its own
tranche. Every number they published reproduces here from an independent
extraction, including PORTQ-6's self-correction on `[201]` from 11 to 7. What
this wave adds is the two axes they did not cover and a non-pairwise survival
probe whose median is **1.000**: the queue is net-stable as a body, and §4.5
enumerates the small population that is not.

**Nothing in this wave is executed.** No build, no test run, no GPU, no lease.
Disk on the developer box was at 98-99% throughout and no build tree was created.

The pin does **not** advance and this wave is not the change that advances it.
The active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.
Report: [`../sync/2026-09-03-portq-reconcile.md`](../sync/2026-09-03-portq-reconcile.md).

## 1. Scope

Two questions.

**Q1.** Over all 290 PORT-NOW entries of `5559679229..e126687a9a`, which pairs
or chains cancel, amend, supersede or reorder one another, including where the
other member is a commit in range but outside the 315? Produce a net-effect
ordering: must-port-together, fixed-order, and non-work.

**Q2.** Applying AGENTS.md §"vLLM is the reference" — "Advance the pin only
after you reconcile every affected row and gate" — and §"Gates" to the measured
facts, what stands between the current state and advancing the parity pin to
`e126687a9a`?

Out of scope: porting any entry, editing `porting-inventory.md`, editing the
`parity-pin` block, and advancing the pin.

## 2. Method

**The queue is extracted by pattern, never by line range.**
`.agents/sync/2026-09-01-cdefd9d.md` §4's 315 SHAs are taken from lines matching
`^- ` inside the section delimited by its own headings, then intersected with
`git rev-list --reverse 5559679229..e126687a9a`. The result is 290, and its
positional numbering is **byte-identical** to the numbering the seven tranche
reports use — checked by `diff` against every `| N | \`sha\` |` row across all
seven, which is a positive control on both the extraction and the reports.

**Diffs are read at revision, never in a working tree**, from `git show -U0
--no-renames` in `/home/mudler/_git/vllm`.

**Four scans, and the parser defect that would have hidden a result.** The first
extraction dropped every hunk of a file a commit deletes, because it keyed the
current path off `+++ b/` alone and a deletion writes `+++ /dev/null`. That made
`[45]`'s revert read as 239 lines where PORTQ-7 §5.3 measured 332. The corrected
parser falls back to `--- a/`, and it reproduces PORTQ-7's 332/332, 66/66 and
52/52 independently. **A scan that silently loses a class of hunk fails toward
"no relationship found", which is the same shape as a green gate that never
ran.**

The four scans, all filtered by shared file, which §3.1 shows is load-bearing:

1. exact inverse (`A.plus == B.minus` and `A.minus == B.plus`);
2. byte-identical (`A.plus == B.plus` and `A.minus == B.minus`);
3. directional (of A's added lines, how many a later B deletes);
4. **mirror** (of A's deleted lines, how many a later B re-adds) — new here.

Scans 3 and 4 are run twice: over the 290, and over the 1175 in-range commits
outside the queue.

**A fifth probe answers the question the pairwise scans cannot.** For every
entry, what fraction of the lines it adds is present verbatim in the same file
at `e126687a9a`? This is a net-effect measure that needs no pair, so it catches
an entry undone by two commits, or by a rewrite no single commit reverts.

**Every zero carries a positive control through the same probe form**, and the
controls are printed beside the zeros in the report. Every exit status is read
directly, never after a pipe.

## 3. Risks

- **A relocation reads as a revert.** `--no-renames` makes a moved file's old
  path look deleted. Four hits are this shape and are named as such in §3.3 of
  the report, each checked by confirming both paths exist at the target.
- **A duplicated idiom reads as a pairing.** PORTQ-7 §5.3 measured this at 57%
  and 37.5%; the shared-file filter is therefore part of the test.
- **A survival fraction cannot see a semantic revert with no shared lines.** The
  probe is stated as a lower bound on disturbance, not an upper one.
- **This wave reads records, not the tree.** Where a report's label is cited it
  is cited as that report's finding, not re-derived.

## 4. Tests and gates

No product code changes, so no focused suite applies. The gates are:

- `scripts/agent-preflight.sh --staged` before commit, read for `gate(s) failed`
  and `NOT a green preflight` in the text, never off the exit status alone;
- `python3 scripts/agent-pr-body.py --pr <N>` on the bytes about to be squashed;
- every count in the report reproducible from a command printed beside it.

## 5. Owed

This wave files nothing new and closes nothing but its own issue. It carries
forward, unchanged, the obligations `.agents/oracles/vllm.md:58-70` records
against a pin advance, and the 55 issues the seven PORTQ waves filed against the
queue's real gaps. Its own reading of "reconcile every affected row" is argued
in the report and is not a rule change; no checker is added or altered.

- [#2764](https://github.com/mudler/vllm.cpp/issues/2764) — this wave.
- [#2611](https://github.com/mudler/vllm.cpp/issues/2611) — the 290-entry queue.
- [#2626](https://github.com/mudler/vllm.cpp/issues/2626) — `qwen4_exp` does not
  run at the candidate on `thor:gpu0`; owned by `MODEL-MM-QWEN4-EXP`.

## 6. Stop conditions

Stop and report rather than proceed if: the pin block, `.agents/oracles/vllm.md`
or `.agents/upstream-sync.md` is edited by another session while this wave runs;
a scan result contradicts a committed PORTQ number and the contradiction cannot
be resolved to a parser or normalisation difference; or the developer directs
the pin to advance, which is a separate change with its own review.
