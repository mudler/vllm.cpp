# LTX25-ADHERENCE-RESCORE — does the sigma-shift anchor explain the adherence gap?

Row `LTX25-ADHERENCE-RESCORE`. Issue
[#2576](https://github.com/mudler/vllm.cpp/issues/2576). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Base: `origin/main` at `6974557b02f822d9a1d603e8639091f735b7ecbe`, pinned when
the worktree was created. The four cross-document anchors the review repairs
added — `ltx25-adherence-detail-loss.md:454-461`, `ltx25-oracle-absolute.md:54`,
`ltx25-prompt-adherence.md:698-706` and `ltx25-render-compare.py:369` — were
re-read at `origin/main` **`e24ec8bfd`** as well as in this worktree, and hold
in both.

Oracle: Lightricks `LTX-2` at `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
`gateable = yes`; its reference render, manifest and `SHA256SUMS` are committed
at `tests/parity/goldens/ltx2_oracle/`. Scorer: `openai/clip-vit-base-patch16`
at `57c216476eefef5ab752ec549e440a49ae4ae5f3`, an INSTRUMENT and not an oracle,
pinned by `ltx25-prompt-adherence.md` §5.

## Now

`DONE` with this change. The answer is NO, and it is worse than no: the sigma
anchor does not explain the adherence gap, and mirroring upstream MOVED THE
SCORE DOWN. S1 falls 35.2719 -> 35.1799 and S2, which passed before, now FAILS
with a named decoy winning all 25 frames. `## Outcome` carries the four
readings, the third arm that makes them attributable, and the lead this leaves.
PR #2528 is correct, stays landed, and this row does not ask for a revert.

## Scope

**In.** One render at the pinned request on a binary built from a tree that
carries `c9366de65` (`LTX25-SIGMA-SHIFT-MIRROR`), scored against the same
reference and the same scorer that produced the standing 35.2719, and the
comparison of the two readings. Order 1 of
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) `## 8`.

**Out.** Closing the gap. Tuning anything toward the score. Any change to the
engine, the harness's verdict block, the request, the scorer, the reference, or
the bound. A second seed. This row RETURNS `NEEDS_DECISION` rather than moving
a number.

## The question, stated so that both answers publish

`ltx25-prompt-adherence.md` `## Outcome` records `[FAIL]
absolute.ours.adherence_clip 35.2719 >= 36.0087 ... margin -0.7368`, reference
mean 38.1278, on frames produced by `main` at `7905607af`.

`LTX25-SIGMA-SHIFT-MIRROR` (#2521) then changed the sigma schedule that render
used. Its own arm table puts `one_stage` x 2.5 `generate` on the flipped side:
`schedule_tokens` moved from `kTargetLatent` to `kSchedulerDefault`, so at this
geometry the shift moved from 240 tokens / 0.669271 to 4096 / 2.050000 and every
interior sigma moved with it, by up to 0.194617 at step 5. The loop leaves the
high-noise regime later. CFG sets composition and prompt semantics there, and
prompt semantics is what CLIP measures.

**That is a hypothesis and this row tests it. It is not a validation of #2521,
and a non-improvement is not a regression of it.** #2521's justification is that
the engine did not match upstream, and it stands whichever way the score moves;
that row said so in advance, in
[`ltx25-sigma-shift-mirror.md`](ltx25-sigma-shift-mirror.md)
`## Why this is not gated on the adherence number`: "if the score gets worse it
is still correct and is reported as such". The negative answer is the more
informative one here, because **no mechanism currently explains the gap at
all**: #2513 falsified smoothness and withdrew its replacement lead as a DFT
border artifact.

**A third answer is available and it is the cheapest to detect.** If the new
frames are BYTE-IDENTICAL to the old ones, the flip did not reach this
configuration, and that is the finding rather than a failed measurement.

## Why one render is enough, and what it cannot buy

`ltx25-render-confirm.md` `## Outcome` records three renders on one binary at
one seed producing byte-identical frames — the same `sha256sum frame_*.ppm |
sha256sum` — differing only in wall time. So one render on the new binary
establishes the new value EXACTLY.

**The same fact forbids the obvious next step.** Repeating a bit-deterministic
render cannot produce an error bar, and an `n` quoted over identical inputs is
not a sample size. The standing reading has no error bar
(`ltx25-prompt-adherence.md` `## Owed`, #2514) and this row does not give it
one. Uncertainty here would have to come from DIFFERENT SEEDS, which is a
different question and is out of this scope. `N=1` is therefore the honest
setting rather than a saving, and the harness's own `n = 3` speed axis is not
collected.

## Method

1. **The old reading is re-derived rather than transcribed.** The 25 retained
   PPM frames of `rc` job `93a60151-7d4d-4718-842c-ef724208be0e` are copied off
   CIFS, their `sha256sum frame_*.ppm | sha256sum` is checked against the
   committed `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae`,
   and the tool is re-run on them on this devbox. This is the control on the
   scoring pipeline: a reading that does not reproduce 35.2719 means the
   instrument moved, and no comparison is possible until it is explained.
2. **The render.** `scripts/ltx25-render-confirm.sh` inside an `rc` lease on
   `dgx:gpu0`, `N=1 KEEP_FRAMES=1`, at the harness's own pinned request. The
   binary is built in-lease from a tarball of this branch, and its
   `binary_sha256`, `library_sha256` and `source_sha` are recorded.
3. **The score.** `scripts/ltx25-render-compare.py --adherence-model` on the
   retained frames, against BOTH reference forms — the committed
   `upstream-render.mp4` and the lossless `upstream_frames` — because
   `ltx25-prompt-adherence.md` reports the codec's 0.1254 contribution from
   exactly that pair and dropping one form would make this reading
   incomparable with the standing one.
4. **The identity check.** `sha256sum frame_*.ppm | sha256sum` over the new
   frames against the old digest.

## What would falsify each answer

| Answer | What it takes | What it does not license |
|---|---|---|
| the mechanism EXPLAINS the gap | S1 margin >= 0 at this request | calling adherence closed at any other request, seed or geometry — n is still 1 |
| the mechanism PARTLY explains it | margin moves toward 0 and stays negative | attributing the remainder to anything; the residue is unexplained |
| the mechanism DOES NOT explain it | margin unchanged or worse | any statement that #2521 was wrong |
| the flip did not REACH this arm | frames byte-identical to the old digest | reading the arm table as falsified; it is a statement about recipes, not renders |

## Risks

- **Two binaries, 454 commits apart, and the inherited no-drift scan does NOT
  reach across them.** The standing 35.2719 came from `main` at `7905607af`;
  the new reading comes from this branch, whose lease `PROVENANCE` records
  `source_sha = c3b4da804`. Re-derived over that exact range rather than
  inherited: `git rev-list --count 7905607af..c3b4da804` is **454**, **129**
  files under `src/` and `include/` changed, and **5 of them match
  `ltx|video|vae|diffus`** — `ltx2_conditioning.h`, `ltx2_pipeline.h`,
  `ltx2_video.h`, `ltx2_pipeline.cpp` and `ltx2_video.cpp`.

  **`7905607af` here and the `source_sha 790c582bb` in `## Outcome`'s arm table
  are the same engine**, so a reader who checks does not have to leave the
  document. `790c582bb` is the dangling pre-squash tip of
  `row/LTX25-RENDER-CONFIRM`, reachable from no ref, and its entire delta over
  `7905607af` is `.agents/specs/ltx25-render-confirm.md` plus
  `scripts/ltx25-render-confirm.sh` — no engine code
  (`ltx25-prompt-adherence.md:698-706` establishes this; re-verified for this
  bullet: `git merge-base --is-ancestor 7905607af 790c582bb` is YES,
  `git rev-list --count 7905607af..790c582bb` is **2**, and
  `git diff --name-only` over that range lists exactly those two files). The
  engine sha is therefore the right endpoint to scan from, which is why this
  bullet uses it.

  Every number in `## Outcome` names the SHA it came from, and the difference is
  not attributed to the sigma flip on the strength of this scan.

  **An earlier draft of this bullet read "112+ commits apart", and it was wrong
  in two ways.** 112 is `ltx25-prompt-adherence.md`'s own figure and it is
  exact at ITS anchor: `7905607af..855905f59` is 112 commits, 56 `src`/`include`
  files, and 0 matching the LTX path names. Quoting it here imported a count
  pinned to a different, older endpoint — and, the part that mattered, imported
  the `0 matching` property with it. That property does not survive the
  extension: over this row's range, five LTX-path files moved. `origin/main` is
  no anchor either, because it moves: `git rev-list --count
  7905607af..origin/main` read 460 while this row's lease ran, 613 while
  `## Outcome` was being written, and 648 by the time this file was committed —
  three values on one day, none of them a fact about the binary that produced
  the pixels.

  So the drift confound was real, a path-name scan could not retire it, and a
  two-arm before/after would have been uninterpretable. `## Outcome` retires it
  by measurement instead, with a third arm.
- **The harness gates blockiness before it will time anything, and exits on a
  non-zero verdict.** That branch is untouched. Render 1's frames are never
  deleted by the harness at any exit, so an early exit still leaves the pixels
  this row scores.
- **`dgx` crashes under long sequences.** `N=1` is one render, and the source
  tarball plus the `confirm-bin` cache make a re-run cheap.

## Gates

Run from a worktree at this row's base, with `MODEL` the pinned CLIP snapshot,
`REFPPM` a local copy of `/mnt/nas_share/rc/ltx2-oracle/out/upstream_frames`,
`OLD` a local copy of the retained pre-flip frames
(`ltx25-render-confirm/run/20260901T075837Z/r1`), and `NEW` and `ABL` local
copies of the lease's `r1` and `abl`
(`ltx25-adherence-rescore/run/20260902T145000Z/`). No GPU and no lease: the
frames and the goldens are all this needs.

```sh
# the control: the old frames are the old frames, and the ABLATION is those
# same frames, to the byte, 454 commits and three builds later
( cd "$OLD" && sha256sum frame_*.ppm | sha256sum )   # 1166b2869400...93ae
( cd "$ABL" && sha256sum frame_*.ppm | sha256sum )   # 1166b2869400...93ae
for f in "$OLD"/frame_*.ppm; do cmp "$f" "$ABL/$(basename "$f")"; done

# the three readings, each against both reference forms
for A in "$OLD" "$ABL" "$NEW"; do
  python3 scripts/ltx25-render-compare.py --a "$A" --label-a ours \
    --reference tests/parity/goldens/ltx2_oracle/upstream-render.mp4 \
    --adherence-model "$MODEL"
  python3 scripts/ltx25-render-compare.py --a "$A" --label-a ours \
    --reference "$REFPPM" --adherence-model "$MODEL"
done

# did the flip reach this arm at all: it did, on all 25 frames
( cd "$NEW" && sha256sum frame_*.ppm | sha256sum )   # 065687a01f59...27a7
```

Each `ltx25-render-compare.py` invocation exits 1 with
`READING WORSE_THAN_ORACLE` and `VERDICT FAIL`, on `OLD` and `ABL` for S1 alone
and on `NEW` for S1 and S2 both. That non-zero exit IS the recorded result of
this row; `## Outcome` carries every printed figure.

## Stop conditions

- Return `NEEDS_DECISION` rather than changing the request, the seed, the
  scorer, the reference, the bound or any engine value to move the score.
- Return `REMOTE_UNVERIFIED` rather than guessing if the controller cannot be
  reached. Never fall back to `ssh` plus a file mutex on a fleet device: the
  fleet cannot see that mutex, and `minimax-music3.md` §13.10 retains a whole
  speed axis as VOID because that was done once.
- Stop and report if the control in `## Method` step 1 does not reproduce
  35.2719.

## Outcome

`DONE`, measured 2026-09-02 and re-derived 2026-09-03. The mechanism is
**REFUTED**, and the refutation is sharper than "no change": mirroring
upstream's sigma anchor made the render WORSE by this instrument, on both of
the gate's adherence checks rather than one.

### What was run

One `rc` lease, job `964d415e-e22e-41e5-b0ad-7f00ae221c5b` on `dgx:gpu0`,
writing `/mnt/nas_share/rc/ltx25-adherence-rescore/run/20260902T145000Z/`. Its
`PROVENANCE` records `source_sha = c3b4da804`, `binary_sha256 =
600cf798c48ebabebc1fa25fb4891fe0b550f31f995501105aea856cced4c54d`, the four
BF16 checkpoint digests, `geometry=320x192/25f steps=8 seed=42
video_tokens=240` and `prompt_sha256=a65a14fe11dc5296b6747e62f412c949d00f455e4ffabce794f3f3d939f4cb93`.
`renders_completed=1 verdict=PASS` on the harness's own blockiness gate;
`phase1_harness_exit=0`.

Phase 2 then patched exactly ONE line —
`src/vllm/model_executor/models/ltx2_pipeline.cpp:1158`, `OneStagePhase`'s
`phase.schedule_tokens`, from `kSchedulerDefault` back to `kTargetLatent` —
rebuilt in place (`build-ablation.log`: five ninja targets, only
`ltx2_pipeline.cpp.o` recompiled, `phase2_build_seconds=7`) and rendered a
second time. That is the ABLATION arm.

**`## Method` planned two arms. Three were taken, and two would not have been
interpretable.** The third arm is a deviation from the committed method and it
is recorded as one. It is not a scope breach: the patch was a throwaway
ablation build inside the lease, never committed and never landed, and
`## Scope`'s exclusion is on changing the engine to MOVE the number. This
changed the engine to hold the number still.

### The readings, re-derived rather than transcribed

Every figure below was recomputed on the devbox, CPU only, no GPU and no lease,
from the retained PPM frames and the committed goldens. Both committed golden
files verify against `SHA256SUMS`, and so do all 25 lossless reference frames
plus `audio.wav` at `/mnt/nas_share/rc/ltx2-oracle/out/upstream_frames`.

| arm | source | `sha256sum frame_*.ppm \| sha256sum` | `cat frame_*.ppm \| sha256sum` |
|---|---|---|---|
| old | job `93a60151`, `source_sha 790c582bb` | `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae` | `7c74a2b074e557b60935cbc2d2f990ef19545e73236761e2d7d7297c1e77faae` |
| ablation | job `964d415e`, `c3b4da804` + the one-line revert | `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae` | `7c74a2b074e557b60935cbc2d2f990ef19545e73236761e2d7d7297c1e77faae` |
| new | job `964d415e`, `c3b4da804` | `065687a01f59d6d3d109d4565e385af3c31b6a53326d4a44492745c0df9527a7` | `e98dc514cec9043833d53c95c33ffa477c8a9eac1ab99fd5aca945dbcedc7147` |

The scores, `scripts/ltx25-render-compare.py --adherence-model` with the pinned
`openai/clip-vit-base-patch16` at `57c216476eefef5ab752ec549e440a49ae4ae5f3`,
eight file digests verified on every run. Both reference forms, as
`## Method` step 3 requires:

| | ablation / old | new |
|---|---:|---:|
| S1 CLIP mean, ours | **35.2719** | **35.1799** |
| S1 bound, mp4 reference (mean 38.1278) | 36.0087, margin **−0.7368** | 36.0087, margin **−0.8288** |
| S1 bound, lossless reference (mean 38.0024) | 35.9286, margin **−0.6567** | 35.9286, margin **−0.7487** |
| S2 argmax over 7 prompts | `true` | **`near:1`** |
| S2 margin to the best decoy | **+0.3370** | **−0.7868** |
| S2 per-frame wins | 15/25 | **0/25** |
| S2 verdict | `[PASS]` | **`[FAIL]`** |
| sharpness | 10.637435899739584 | 10.123432387695312 |
| `blockiness_grid8` | 1.0301103174717752 `[PASS]` | 1.0282447301026165 `[PASS]` |
| `blockiness_grid32` | 1.0248094630021185 `[PASS]` | 0.9757147618162578 `[PASS]` |
| `pixel_mean` — reference **121.704516** | 126.544109 | 103.150315 |
| HSV mean saturation — reference **0.243190** | 0.117272 | 0.154290 |
| `audio_rms` | 121.79029972068476 | 160.8030039983292 |
| reading / verdict | `WORSE_THAN_ORACLE` / `FAIL` (exit 1) | `WORSE_THAN_ORACLE` / `FAIL` (exit 1) |

S0 PASSED on the reference in all four runs — the true prompt ranks first over
the six committed decoys, by +1.8240 on the mp4 form and +1.8068 on the
lossless one — so the instrument was demonstrably able to say no on every
reading taken here.

**Three properties of the instrument fell out of collecting both forms, and
each is a check on it.** Our own S1 value is INVARIANT to the reference form,
because the reference supplies only the bound; the form moves the reference
mean by 38.1278 − 38.0024 = **+0.1254**, which is exactly the codec
contribution `ltx25-prompt-adherence.md` records, reproduced here
independently; and S2 is invariant to the form entirely, margins and per-frame
wins alike, because it ranks decoys on OUR frames and never on the reference's.

**Exposure moved four times further from the reference than colour moved toward
it, and the standing table reported neither against the value that makes them
readable.** The reference's own `pixel_mean` is **121.704516**. So the ablation
arm sat **+4.839593** from the reference's exposure and the new arm sits
**−18.554201** — mirroring upstream's sigma anchor moved global exposure
**3.83x further** from the reference than it already was: 15.25% below the
reference, 18.49% below the arm it replaced. The colour axis moves the OTHER
way. Mean HSV saturation goes **0.117272 → 0.154290** against the reference's
**0.243190**, which closes 29.4% of the ablation arm's shortfall. The obvious
joint mechanism therefore does **NOT** hold: the new arm is not washed out
toward grey, it is darker AND more saturated, and the next reader should not
spend a lease chasing desaturation. This is a real result of the flip, not a
side note, and §"What to chase next" ranks it accordingly.

**How those four numbers were computed**, so a reader can repeat them with no
GPU and no lease. `pixel_mean` is the tool's own
(`scripts/ltx25-render-compare.py:369`): the mean over the 25 frames of each
frame's mean over all three 8-bit channels. Saturation is the HSV S channel,
`(max(R,G,B) − min(R,G,B)) / max(R,G,B)` with `S = 0` where `max = 0`, averaged
over every pixel of every frame — `ltx25-render-compare.py` emits no saturation
metric, so this one is DEFINED here rather than quoted from an instrument. Both
were read from the retained PPMs at
`ltx25-adherence-rescore/run/20260902T145000Z/{abl,r1}` and
`/mnt/nas_share/rc/ltx2-oracle/out/upstream_frames`, which is the same lossless
reference set this section verifies against `SHA256SUMS` above.

### The control, which is the most valuable line in this row

**The ablation's frames are byte-identical to the render taken 454 commits
earlier.** Not equal in digest by coincidence of hashing: all 25 files compare
byte-for-byte with `cmp`, under both digest forms, and the digest-of-digests
equals the value this tree already commits in four places — grep
`1166b28694001c52a` and it is `OURS_SET_DIGEST` in
`scripts/ltx25-adherence-detail-loss.py`, the `set_digest` key of
`tests/parity/goldens/ltx25_detail_loss/detail-loss.json`, and one line in each
of `ltx25-prompt-adherence.md` and `ltx25-adherence-detail-loss.md` — as the
frames the standing 35.2719 was scored on. Re-running the tool on them
reproduces
`ltx25-prompt-adherence.md`'s printed block to every digit:
`sharpness=10.637435899739584 block8=1.0301103174717752
block32=1.0248094630021185 clipped=0.001076171875`, `[FAIL]
absolute.ours.adherence_clip 35.2719 >= 36.0087 ... margin -0.7368`, `[PASS]
absolute.ours.adherence_argmax ... margin +0.3370, wins 15/25`.

That equality holds across **454 commits, two committed `source_sha`s plus the
one-line ablation patch, three separate builds carrying three different
`library_sha256` values (`c4692db9…`, `f4ead0e4…`, `d29f6c05…`), and two `rc`
jobs a day apart**. 129 files under `src/` and `include/` changed over the
range, five of them matching `ltx|video|vae|diffus` by path name — and the
pixels did not move by one byte.

**So the drift confound is measured at exactly zero, and every difference in
the new arm belongs to the sigma anchor alone.** A two-arm before/after over
this range could not have said that: it would have carried 454 commits of
unmeasured difference and five touched LTX-path files inside its delta, and no
amount of reading the diff would have retired them. The third arm retired them
by measurement, for a 7-second rebuild and a 313-second render.

It also strengthens the determinism record. `ltx25-render-confirm.md`
`## Outcome` established bit-determinism across three executions of ONE binary.
This establishes it across three DIFFERENT builds of an evolving tree, which is
a stronger statement and a cheaper one to have taken.

### The answer

The sigma anchor does not explain the adherence gap. It is the third mechanism
proposed for that gap and the third to be falsified:

1. **Smoothness** — falsified by #2513 / PR #2525, which also found the premise
   false: our render is not smoother than upstream's.
2. **The separable-upsampler lead** that replaced it — withdrawn by that same
   row's `## CORRECTION` as a DFT border artifact of a Hann window, the
   periodic component showing a crossover the first draft said did not exist.
3. **The sigma-shift anchor** — this row. Refuted, and in the unhelpful
   direction.

**No mechanism currently explains the gap, and the candidate list is now
shorter by three.**

### What this row does NOT say, stated because it is the easy misreading

**PR #2528 (issue #2521, commit `c9366de65`) is correct and stays landed. This
row is not a reason to revert it and does not ask for one.** It matches
vLLM-Omni's `_official_ltx_sigmas` (`ltx2_denoise.py:186-188` at `a4ea67a2`,
under the comment naming the max-sequence anchor) and Lightricks'
`ti2vid_one_stage.py:207`, which passes no latent and so takes
`schedulers.py:32`'s `default_number_of_tokens` branch at
`MAX_SHIFT_ANCHOR = 4096`. Both oracles agree independently. Mirroring vLLM is
the rule, and a score is not a vote on it.

That change **said so in advance, before any number existed.** Its own commit
body records "The prompt-adherence score was NOT measured and is deliberately
not offered as justification", and `ltx25-sigma-shift-mirror.md` `## Why this
is not gated on the adherence number` says "if the score gets worse it is still
correct and is reported as such". This row is that report. The framing is what
makes the result a finding rather than a regression: because the justification
never rested on the score, the score is free to be informative.

**The reading this supports is that a second defect was being partly masked by
the first.** Our render diverges from upstream's for some reason this row does
not name, and the wrong schedule was compensating for it — buying back enough
of the prompt's semantics to keep S2 above water while S1 stayed below the
bound. Remove the compensation and the underlying divergence is visible at full
size. That is a hypothesis, it is one reading consistent with the data, and it
is not evidence for itself. It is not the only one: the exposure axis in
`## Outcome` supplies a competitor from this row's own table.

**One obvious counter to the masking reading is closed, not open.** It would be
that our `one_stage` arm is mapped to the wrong upstream recipe, so #2521's flip
never should have reached this render and the score change is a mismatch rather
than a mechanism. It is not: the #1864 reference render this row scores against
was itself taken through `python -m ltx_pipelines.ti2vid_one_stage`
(`ltx25-oracle-absolute.md:54`, the `Entry point` row of its anchor table). The
oracle render ran at the 4096 anchor, which is exactly what
`schedulers.py:32`'s `default_number_of_tokens` branch gives
`ti2vid_one_stage.py:207`. Our arm and the reference are the same recipe, and
the flip moved us ONTO the reference's schedule. That is what makes the score
getting worse informative rather than a mapping error.

### What to chase next, and why it is a better handle than the number that fell

**S2 is the finding, not S1.** S1 moved by 0.0920, which is 0.10 of the
reference's own per-frame sd — a number that on `n = 1` argues for very little.
S2 moved from a `[PASS]` with 15/25 per-frame wins to a `[FAIL]` with **0/25**,
and the ranking inverted: `near:1 > true` on every single frame, where before
it was `true > near:1`.

`near:1` is `decoys.json` index 1, **"A grey wolf walks slowly through a snowy
pine forest at sunrise, cinematic."** — the SPECIES decoy, the one that changes
exactly one attribute of the true prompt and changes the animal. Season
(`near:0`) and action (`near:2`) both stay below it. So the instrument is
naming a specific, discrete, attributable failure: **the render's animal now
reads as a wolf rather than a fox, unanimously.**

That is a categorical signal with a per-frame count behind it, on a decoy set
committed before any of these scores were seen. It is worth far more than a
scalar mean that fell by a tenth of a standard deviation. It is also the axis
this campaign already treated as the tight one: `ltx25-adherence-detail-loss.md`
names "the fox-against-wolf margin" three times as the thing its own hypothesis
was built to explain.

**Two candidates come out of this row with equal standing, and it does not
choose between them.**

1. **What carries species identity through this pipeline** — the conditioning
   path and the high-noise steps where CFG sets subject semantics. This is what
   the `near:1` flip names directly.
2. **Global exposure** — the axis in `## Outcome`, where the new arm sits
   **−18.554201** from the reference's `pixel_mean` against the ablation's
   **+4.839593**.

The second is not a separate defect to file behind the first. It is a
**competing explanation for it**: the true prompt says *at sunrise*, and an
18.49% drop in mean pixel value on a sunrise prompt can move a CLIP argmax on
its own, with the render's animal unchanged. It also has the better provenance
of the two. The exposure figure is measured against the ORACLE's own frames,
while the `near:1` flip is an instrument reading with no reference value behind
it — S2 ranks decoys on our frames only, as three paragraphs above establish.
And the campaign already treats this axis as live:
`ltx25-adherence-detail-loss.md:454-461` fits a gain-and-gamma map from our
luminance quantiles onto the reference's and reports gain **0.99144**, gamma
**1.04654** with an rms residual of **14.9379** levels. That fit was taken on
the frames whose digest is `1166b28694001c52a`, which this row proves are the
ablation arm's, byte for byte — the arm that sat **+4.839593**. Nobody has
retaken it on the arm that sits **−18.554201**, and doing so is cheap.

Both candidates should be gated on S2's per-frame win count, which
discriminates, rather than on S1's mean, which on this evidence barely does.

### Why each default has the value it has

- **`N = 1`.** Renders here are bit-deterministic, now demonstrated across
  three builds rather than three executions. Repeating one cannot produce an
  error bar, and an `n` quoted over identical inputs is not a sample size. The
  harness's own `n = 3` speed axis was not collected because it would measure
  wall time, which this row does not ask about.
- **Seed 42 and the pinned request.** It is the request the #1864 reference
  render was taken at. The tool reads the prompt from
  `ltx2_oracle_manifest.json` rather than from a flag, so the prompt and the
  reference cannot disagree. Changing either is a `## Stop conditions` breach.
- **Both reference forms.** `ltx25-prompt-adherence.md` reports the codec's
  contribution from exactly this pair, so dropping one would have made this
  reading incomparable with the standing one. Collecting both also produced the
  independent +0.1254 confirmation above.
- **The bound is untouched.** It stays the reference's own per-frame minimum,
  recomputed on every run. #1854 exists to keep a chosen constant out of the
  verdict, and a row that failed a bound is the last one that should move it.
- **The ablation is a one-line source patch, not a checkout of the pre-#2521
  tree.** A checkout would have re-introduced the 454 commits this arm exists
  to exclude and would have measured drift plus anchor together — the exact
  confound it was built to remove. Same source, same toolkit sonames, same four
  checkpoints, same lease, one line different.
- **CLIP stays an INSTRUMENT.** Its role is unchanged by this row and no
  reading here is consumed except as a comparison against the `ltx-2` oracle's
  own render.

### What was rejected

- **Reverting the sigma anchor.** Refused above, with the oracle citations. A
  `NEEDS_DECISION` would have been the correct exit if this row had been asked
  to act on the score; it was asked to measure, and it measured.
- **Attributing the drop to any named mechanism.** `## What would falsify each
  answer` licenses no such statement on this row's evidence, and the masking
  reading above is offered as a hypothesis and labelled one.
- **Quoting an error bar, or calling the 0.0920 movement significant.**
  Determinism forbids the first and `n = 1` forbids the second.
- **`origin/main` as the drift anchor.** It moves, and it moved three times
  while this row was being recorded: `git rev-list --count
  7905607af..origin/main` read 460, then 613, then 648, all on 2026-09-02/03.
  None of them is a fact about the binary that produced these pixels.
  `7905607af..c3b4da804` — the render's own recorded `source_sha` — is 454 and
  does not move, which is why it is the figure this spec carries.
- **The "112+ commits apart" figure this spec's `## Risks` carried.** Corrected
  in place; the reasoning is in that bullet.

## Owed

- **SEED VARIANCE is the one experiment on this axis worth a lease, and this
  row makes the case for it rather than closing it.** `n` is still 1, #2514
  owns it, and the closing move is DIFFERENT SEEDS. This row hardened the
  reason: renders here are bit-deterministic across three separate BUILDS of an
  evolving tree, not merely across three executions of one binary
  (`## Outcome`, "The control"). **So no number of repeats can ever produce an
  error bar.** A lease spent re-rendering seed 42 is a lease wasted, and the
  only remaining source of uncertainty on this axis is the seed. The shape to
  fund is k distinct seeds at the pinned request, both arms, scored the same
  way — which would also say whether the 0/25 S2 inversion this row found is a
  property of the schedule or of one draw.
- **`scripts/ltx25-render-confirm.sh` still does not pass `--adherence-model`**,
  so all four adherence readings in `## Outcome` were taken by hand on a second
  pass after the lease. `ltx25-prompt-adherence.md` `## Owed` records it as
  order 2 of `ltx25-completion-scope.md` `## 8`, and this row inherits it
  rather than fixing it: wiring the scorer into the harness changes what the
  lease gates and needs its own red-first case.

  **It is worth less than it looks, and this row is the evidence.** Because the
  frames are bit-deterministic and the harness retains them at every exit, the
  hand pass is free, repeatable, runs on a devbox with no GPU and no lease, and
  produced identical numbers to the digit on a second pass a day later. What
  the wiring would buy is the lease FAILING on adherence instead of only on
  blockiness — a scheduling convenience, not a measurement. Rank it below the
  seed sweep.
