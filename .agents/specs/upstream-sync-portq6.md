# Sync cycle `e126687a9a`, wave PORTQ-6

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, verified again at `bb2da6f97`. `scripts/check-agent-record.py`
passes on a `Row:` line whether or not the row resolves, so this note is here to
stop a reader taking it for a matrix reference. The issues this wave cites are
carried under `## Owed` below, which is the route AGENTS.md gives when no row
owns the work.
Issue: [#2718](https://github.com/mudler/vllm.cpp/issues/2718).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq5.md`](upstream-sync-portq5.md),
whose method this wave continues.
Sibling: PORTQ-7 ([#2717](https://github.com/mudler/vllm.cpp/issues/2717)) takes
entries 241-290 concurrently. Together the two finish the classification.

## Now

**Done, and repaired after a fresh review returned NOT PASS.** 40 of 40
re-derived: **3 ALREADY_SATISFIED, 6 REAL_GAP, 31 NOT_APPLICABLE**
(29 `surface-absent`, 2 `inert`). **34 of the 40 disagree with the recorded
PORT-NOW disposition.**

The review re-derived 22 of the 23 NOT_APPLICABLE and ALREADY_SATISFIED calls and
found **no false absence anywhere**, so the classification stands. What failed was
a layer of quantitative and causal claims on top of it, and they failed in one
direction: toward drama. Six material corrections are recorded in place, each
beside the claim it replaces -- §10 (the gate lesson, which was itself an
after-a-pipe error), §6.5 (`[214]`'s pre-pin excuse, refuted), §5.1 (11 markers
overstated; 7), §4.1 and §3 (`[214]`'s site counts), §1 (21 wholly pre-pin
inflated; 18), and §6.1 (the citation mechanism's causal story). **`[221]` moved
from ALREADY_SATISFIED to REAL_GAP**, because the report called its residual "not
portable work" and sized it at ~5 lines one sentence later.

Six issues carry the six gaps, all new:
[#2719](https://github.com/mudler/vllm.cpp/issues/2719) (`KV-SIZING`),
[#2721](https://github.com/mudler/vllm.cpp/issues/2721) (`LORA-RUNTIME`),
[#2722](https://github.com/mudler/vllm.cpp/issues/2722) (`LOAD-SAFETENSORS`),
[#2723](https://github.com/mudler/vllm.cpp/issues/2723) (no row exists),
[#2726](https://github.com/mudler/vllm.cpp/issues/2726) (`SERVE-REQUEST-LENGTH-GUARD`),
[#2747](https://github.com/mudler/vllm.cpp/issues/2747) (`SPEC-DFLASH2`).

**One landed record is falsified and corrected in place.**
`.agents/sync/2026-09-03-portq5.md` §5.3 says stage 6 of the KV-cache layout
refactor, `8bdc70ec7b`, is outside the pin target. It is inside, at PORT-NOW
queue position 230, and PORTQ-5's own citation span contained the annotation that
says so. §6.1 of the report gives the mechanism;
[#2695](https://github.com/mudler/vllm.cpp/issues/2695) is retitled and carries a
correcting comment.

The report is [`../sync/2026-09-03-portq6.md`](../sync/2026-09-03-portq6.md).

**Nothing was executed.** No build, no test run, no GPU, no lease.

## Scope

Classify PORT-NOW entries **201 to 240** of `5559679229..e126687a9a` against the
current tree. The queue is the 290 SHAs that are both in the range and in the 315
of [`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md) §4, extracted
by heading pattern and ordered oldest first.

Additionally, and new in this wave: check each of the forty against the **whole
290 and the whole range**, not only against the tree, because a self-cancelling
or superseding pair is invisible from inside a tranche of forty.

## Exclusions

Port nothing. Advance no pin. Run nothing. Do not build. No GPU lease. Do not
touch entries outside 201-240 except to record a cross-tranche pairing.

## Design

Seven fresh readers, five or six entries each, clustered by surface, each working
from a written brief carrying the eight false-zero mechanisms, the per-file
labelling rule, the pre-pin check and the post-commit-shape rule. The operator
derived the slice twice in two directories, validated it positionally against all
five landed tables, ran the whole-queue pairing scan itself, and re-read every
`file:line` behind every `REAL_GAP` plus the anchors behind two record findings.

## Risks

- **A false `ALREADY_SATISFIED` is more dangerous than a false `REAL_GAP`**,
  because nobody re-reads one. All four here rest on enumerated call sites, not
  on a sampled generalisation. This is PORTQ-5's `[178]` lesson applied.
- **A carried claim can be false at the current target while true where it was
  written.** Two were, and one of them was already annotated. Read to the end of
  every citation span, not to the line the claim quotes.
- **A gap-scoped re-read cannot find a gap filed as a non-gap.** Mitigated by
  clustering readers by surface rather than by label, so each reader met adjacent
  entries and could contradict a sibling's disposition.

## Tests

None. This wave writes no product code and no test. It is a classification pass
whose output is a record, five issues and two corrections.

## Gates

```sh
scripts/agent-preflight.sh --staged
python3 scripts/check-agent-record.py
python3 scripts/agent-pr-body.py --pr <N>
```

The gate run is published in [`../sync/2026-09-03-portq6.md`](../sync/2026-09-03-portq6.md)
§10 and in the pull-request body.

## Evidence

The forty labels with citations, the five gaps with sizes and owners, the five
non-work findings, the pre-pin holes, the absence-control table and the
undetermined list are all in
[`../sync/2026-09-03-portq6.md`](../sync/2026-09-03-portq6.md).

## Stop conditions

Stop if the §4 extraction stops yielding 315, if the positional check disagrees
with any landed table, or if `%ct` monotonicity fails — any of those would
invalidate the landed tranches as well as this one. None fired.

## Owed

- [#2719](https://github.com/mudler/vllm.cpp/issues/2719) — `KV-SIZING`: the null
  block is not reserved when `max_model_len` is sized or auto-fitted.
- [#2721](https://github.com/mudler/vllm.cpp/issues/2721) — `LORA-RUNTIME`:
  `ExpandPackedLora` refuses a packed group with an unadapted member.
- [#2722](https://github.com/mudler/vllm.cpp/issues/2722) — `LOAD-SAFETENSORS`:
  24 loaders trust the config `tie_word_embeddings` over the checkpoint.
- [#2723](https://github.com/mudler/vllm.cpp/issues/2723) — **no row exists**:
  stage 6 of the KV-cache layout refactor, ~900-1400 product lines. The series
  (stages 4-6, #2693 / #2695 / #2723) needs a row; this wave did not invent one.
- [#2726](https://github.com/mudler/vllm.cpp/issues/2726) —
  `SERVE-REQUEST-LENGTH-GUARD`: four media entry points decode or accept
  unbounded bytes.
- [#2747](https://github.com/mudler/vllm.cpp/issues/2747) — `SPEC-DFLASH2`:
  `output_multiplier` accepts any number where upstream refuses a non-positive
  scale.

One residual carries no issue because it is not portable work, and it is recorded
in the report §6.2: `[222]`'s stale prose at `modelopt_mixed_precision.h:56-62`.
The other, `[221]`'s missing `output_multiplier` guard, **was** portable work; the
first version of this spec said otherwise and #2747 now carries it. Two more are owned
elsewhere and are recorded rather than filed: `[203]`'s mirror-source move
(`.agents/model-matrix.md:246` already carries it) and `[211]`'s denominator
consequence (`.agents/oracles/vllm.md:63-68` already carries it).

## Outcome

Recorded on completion in [`../sync/2026-09-03-portq6.md`](../sync/2026-09-03-portq6.md)
§1 and §5. The measured result: **six gaps in forty, 34 of 40 disagreeing with
the record, and five entries that are non-work or partly non-work against the
target for reasons only a whole-queue scan could reach.** Extrapolating the
running rate across the six landed tranches — 51 gaps in 240 entries — gives
**roughly 62 gaps in the 290**, which is **an estimate**, not a measurement: the
tranches are contiguous slices of a SHA-ordered queue, not a random sample, and
this tranche's rate (6 of 40) is at the low end while PORTQ-5's was 11 of 40. The
remaining fifty entries are PORTQ-7's and will settle it.

**The durable lesson of this wave is not in the classification.** The
classification survived review intact. What did not survive was a layer of
derived numbers and causal stories laid on top of it, every one of which erred
toward the more dramatic reading: a gate that "failed while exiting 0" (it exited
1; the 0 was `$?` after a pipe), 11 markers "removed as incorrect" (7 were), a
reader who "could have looked and didn't" (the annotation was not in their tree).
Each was cheap to check and none was checked, because each confirmed a point the
report was already making.
