# SPEC — `LTX25-ORACLE-ABSOLUTE`: the absolute quality panel becomes a gate

Issue: [#1854](https://github.com/mudler/vllm.cpp/issues/1854).
Owner row: `LTX25-ORACLE-ABSOLUTE`. Until this row landed, #1854 was owed by
`LTX25-DIT-ATTN-FLASH` under `## Owed` in
[`ltx25-dit-attn-flash.md`](ltx25-dit-attn-flash.md) §11.5.

## Scope

#1854 says that nothing in this tree asks whether an LTX-2.5 render is GOOD,
only whether two renders are the SAME, and it names the one thing that would
change that: "an absolute reference render from an oracle that runs this
pipeline". It then records that `.agents/oracles/` has no such entry.

**That blocker is gone.** [#1864](https://github.com/mudler/vllm.cpp/issues/1864)
made `ltx-2` `gateable = yes` (`.agents/oracles/ltx-2.md`) and committed a real
reference render of a fixed request to
`tests/parity/goldens/ltx2_oracle/`. This row spends that reference: it turns
part of the REPORTED-ONLY absolute panel in
`scripts/ltx25-render-compare.py` into a check whose bound is recomputed from
the reference's own frames.

IN scope:

- `scripts/ltx25-render-compare.py` gains `--reference`, an oracle render whose
  identity is asserted against the committed `SHA256SUMS`, and gains checks that
  hold each arm's blockiness against a bound derived from that reference.
- The tool learns to judge ONE render, because the absolute question is about
  one render and requiring a second would make the tool demand a comparison it
  does not use.
- One render of our engine at the reference's exact request, on `dgx:gpu0`, and
  the measured verdict, whatever it says.

OUT of scope, and each is declared rather than approximated:

- **Prompt adherence**, #1854's sub-question 1. It needs a vision-language
  scoring model pinned as an oracle. There is none in this tree, this row does
  not invent one, and #1854's first sub-question stays open. See `## Owed`.
- **The identity, correspondence and coherence checks #1743 landed.** Not one
  value, computation or printed line of them moves. This row only ADDS.
- **Any other statistic in the panel.** §5 records, with the measurement, why
  sharpness, clipped fraction and audio RMS stay REPORTED.

## Upstream chain

The reference is upstream's own runtime, not vLLM: vLLM does not register
LTX-2.5, and `ltx-2` is the registered secondary oracle for exactly that reason
(`AGENTS.md` §"When vLLM has no implementation", `.agents/oracles/ltx-2.md`).

| Anchor | Value |
|---|---|
| Oracle | `ltx-2`, `https://github.com/Lightricks/LTX-2` |
| Pin | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| Entry point | `python -m ltx_pipelines.ti2vid_one_stage` |
| Recipe | `tools/oracle/ltx2_oracle.py`, the committed script that produced the manifest |
| Reference artefacts | `tests/parity/goldens/ltx2_oracle/{upstream-render.mp4,ltx2_oracle_manifest.json,SHA256SUMS}` |

The request is fixed in `tools/oracle/ltx2_oracle.py:77-90` and recorded in the
manifest: prompt `A red fox walks slowly through a snowy pine forest at
sunrise, cinematic.`, 320x192, 25 frames, 8 inference steps, seed 42,
`--offload cpu`, `--device cuda`, on `NVIDIA GB10`. The four checkpoints and
their sha256 are in the manifest's `checkpoints` block; every one of them is
BF16, and the text encoder is `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`,
NOT the `nvfp4-torchao` file `scripts/ltx25-dit-attn-flash-pixel-ab.sh:783`
uses. A comparison across quantization arms would measure the arm, so this row
matches the oracle's arm.

## Our baseline

`scripts/ltx25-render-compare.py` computes `absolute_quality()` per arm — mean
sharpness, the 8-grid and 32-grid blockiness ratios, the clipped-pixel fraction,
and audio RMS — prints it under the heading `absolute quality: REPORTED, and NOT
CHECKED (#1854)`, records it in the JSON with `"checked": false`, and checks
none of it. That is #1854's own instrumentation and it is the thing this row
promotes.

Our engine's own render of this request does not exist. Every LTX-2.5 pixel
figure in the tree is from `768x448/49f`, seed `20260820`, the golden-retriever
prompt (`scripts/ltx25-dit-attn-flash-pixel-ab.sh:697-698`), so no existing
number is comparable to the reference and none is reused here.

## Why a comparison is admissible where a threshold is not

#1854 refuses a proxy in terms this row has to satisfy: "a proxy for perceptual
quality that measures nothing is worse than a declared gap", and a hand-rolled
statistic "that correlates with nothing would be the
`a-shape-valid-gate-passes-a-wrong-artefact` failure". The admissible shape it
names is one shape only — **"worse than the oracle on this statistic", because
that is a comparison and not a convention.**

So every number this row gates on is recomputed from the reference render at run
time. Nothing is transcribed, and no constant is chosen. A statistic whose bound
cannot be derived from the reference stays REPORTED, and §5 gives the derivation
that failed for each one.

## 1. The reference, and how its identity is asserted

A "reference" a caller can point anywhere is not a reference. Pointed at our own
render, the gate would pass by construction, which is the
`oracle-identity-must-be-asserted` failure.

`--reference` therefore accepts only bytes whose sha256 is in the committed
`tests/parity/goldens/ltx2_oracle/SHA256SUMS`, resolved from the script's own
location, and it refuses anything else BEFORE it reads a pixel. Two forms:

- **an `.mp4`**, whose whole-file digest must equal the `upstream-render.mp4`
  line. This is the committed artefact, so the gate needs nothing outside the
  tree but `ffmpeg`.
- **a directory** of `frame_*.ppm`, where EVERY frame's digest must appear in
  `SHA256SUMS`. This is the exact form: those 25 digests are committed and the
  frames themselves live on the NAS at `/workspace/ltx2-oracle/out/upstream_frames`,
  which `SHA256SUMS`'s own preamble says are recorded "so a later copy of them is
  checkable against this run rather than trusted".

## 2. The committed mp4 is a usable source, and that is MEASURED, not assumed

The obvious objection to gating blockiness off an H.264 file is that H.264 is a
block codec, so the container could supply the very artefact the statistic
looks for. That objection is correct in general and is false here, and the
difference is a measurement.

Decoding `upstream-render.mp4` (h264, `yuv420p`, 320x192, 25 frames, 225,151
bytes ≈ 9 KB/frame) and running the panel on the decoded frames, against the
same panel on the 25 PPM frames verified byte-for-byte against `SHA256SUMS`:

| statistic | from the PPM frames | from the committed mp4 | relative difference of the derived bound |
|---|---|---|---|
| `blockiness_grid8` mean | 1.042812 | 1.042928 | — |
| `blockiness_grid8` per-frame max | 1.143393 | 1.143697 | **2.66e-04** |
| `blockiness_grid32` mean | 1.037230 | 1.037375 | — |
| `blockiness_grid32` per-frame max | 1.148672 | 1.147804 | **7.56e-04** |
| `sharpness_mean` | 11.274039 | 11.273355 | — |
| `clipped_fraction` | 0.00165039 | 0.00139063 | **0.157** |

The two gated bounds agree to better than 0.08%. `clipped_fraction` does NOT
survive the `yuv420p` round trip — it moves by 16% — which is one of the two
reasons §5 leaves it reported.

## 3. Which statistic gates: blockiness, and only blockiness

Of the four panel statistics, blockiness is the one whose value is anchored by
construction rather than by content. `blockiness_bands()` is the ratio of the
mean luma step ON the block grid to the mean step off it, and its own docstring
states the null: "A render with no block structure sits near 1.0 because the grid
has no special status in it."

The reference confirms that null empirically rather than by assertion. Over its
25 frames, `blockiness_grid8` reads mean 1.042812 with per-frame values from
0.947454 to 1.143393 — a healthy excess over the null of +0.0428 against a
per-frame spread of 0.0528, so the reference's own excess is SMALLER than its
own scatter. `blockiness_grid32` reads 1.037230 over 0.920299 to 1.148672. Both
straddle 1.0.

That is the answer #1854 asked for in its own words: "A blockiness ratio of 1.14
is meaningless without knowing what this VAE produces when it is working." It
produces 1.0428 and 1.0372, and its frames straddle the statistic's structural
null.

## 4. The bound, and why it contains no chosen number

For a one-sided statistic where higher is worse, the reference supplies exactly
one number that is entirely measured: the largest value its own frames reach.

    ours_mean  <=  ref_frame_max        recomputed per grid, on every run

Every digit is measured off the oracle render. Nothing is rounded, nothing is a
convention.

**A TWO-SIDED BAND WAS THE FIRST DESIGN AND A TEST KILLED IT.** The lower edge
was to be `ref_frame_min`, justified as a guard against the collapse §4 describes
below. Held that way, "much LESS blocky than the reference" is a FAILURE, and
less blocky is not worse. The case that found it is in the suite: one render
reading `1.185808` against a deliberately blocky reference whose band was
`[1.892608, 2.161415]` failed, on the side where it was better. The quality claim
is therefore one-sided, and the degeneracy it needed a floor for is checked
directly instead — see below. This is recorded rather than quietly corrected
because the rejected design is the one a reader would otherwise propose.

**Why our MEAN against their per-frame MAX**, and not mean against mean or max
against max. Mean-against-mean has no margin at all and would fire on the
difference in CONTENT between two renders. Max-against-max is a comparison of
two single order statistics: with 25 frames on each side and no real difference,
the probability that our max exceeds theirs is about one half, so that gate
would fire on a coin toss. Our mean against their max is the asymmetric form
whose looseness can be stated: the reference's per-frame max sits about two
standard deviations above its mean, and our mean has one fifth of the per-frame
standard error, so the check fires when our render's blockiness exceeds the
reference's by roughly two of the reference's own per-frame standard deviations.
It is deliberately the LOOSE side of the two, because a red from this gate is a
published claim that our render is worse than upstream's, and that claim must
not rest on scatter.

**The ceiling alone is a mute switch, and what closes it is a COUNT rather than
a second edge.** `blockiness_bands` divides the on-grid step by the off-grid step
and returns `0.0` for a band whose denominator collapsed. A render whose blocks
are FULLY flat — the worst possible block artefact — therefore reads `0.0` and
CLEARS any ceiling. Measured: flattening the reference's own frames completely
onto the 8x8 grid takes `blockiness_grid8` to exactly `0.0000`. So each grid
carries a second check, `absolute.<arm>.blockiness_gridN_defined`, requiring that
no band collapsed. It is a count and not a threshold: the ratio of two
non-negative means is `0.0` only when one of them has collapsed, and neither
collapses in a render with a picture in it. The suite asserts the hole as well as
the guard — the fully flattened fixture PASSES the ceiling and FAILS the count —
so a later reader can see which check is doing the work.

**The reference is held to the same precondition.** A reference whose own bands
collapsed has a ceiling of `0.0`, against which every render fails: a broken
instrument reporting a code verdict. `reference_bounds` refuses it at
`EXIT_UNREADABLE` before any bound is published. It has never happened to the
#1864 render, and it is checked anyway, because the cost of discovering it inside
a GPU lease is a lease.

**What the bound catches, measured.** Flattening each 8x8 block of the reference
frames toward its own mean by a fraction `alpha` gives a dose-response for a
real, visible block artefact on this exact content:

| alpha | `blockiness_grid8` mean | `blockiness_grid32` mean | `sharpness_mean` |
|---|---|---|---|
| 0.00 (the reference) | 1.0428 | 1.0372 | 11.2740 |
| 0.02 | 1.0584 | 1.0527 | 10.9759 |
| 0.05 | 1.0678 | 1.0639 | 10.7244 |
| 0.10 | 1.0955 | 1.0942 | 10.2028 |
| 0.20 | 1.1796 | 1.1830 | 9.1571 |
| 0.35 | 1.3969 | 1.3994 | 7.6230 |
| 0.50 | 1.8079 | 1.7719 | 6.1320 |
| 1.00 | **0.0000** (denominator collapse) | 11.5969 | 1.5107 |

The `grid8` upper bound 1.143393 is crossed between `alpha = 0.10` and
`alpha = 0.20`, so this gate detects a block artefact of roughly 14% flattening
strength or worse and does not detect one weaker than that. **That is the gate's
sensitivity and it is stated rather than implied.** A weaker artefact is not
covered by anything here, and pretending otherwise would be the
`a-floor-below-the-real-count-is-a-mute-switch` failure in the other direction.

## 5. What stays REPORTED, with the derivation that failed

Each of these is left where #1854 put it, and the reason is a measurement rather
than a preference.

- **`sharpness_mean`.** It has no structural null. `blockiness` is a ratio whose
  1.0 means "the grid is not special"; sharpness is a gradient magnitude in
  8-bit levels whose value is set by what is in the picture. Our render and the
  reference are NOT the same picture — different engine, different sampler noise
  — so a bound taken from the reference's 10.84-11.76 band would be measuring
  content. The dose-response above shows it does respond to a blur (11.27 down
  to 6.13 at `alpha = 0.5`), so it is useful instrumentation and it is printed;
  it is not a bound.
- **`clipped_fraction`.** Two independent reasons, both measured. It is
  content-driven, and a sunrise prompt clips; and §2 measures that it does not
  survive the committed mp4's `yuv420p` round trip (0.001650 to 0.001391, 16%
  relative), so the in-tree reference cannot supply a stable bound for it at all.
- **`audio_rms_mean` / `audio_rms_min`.** The reference's `audio.wav` is not
  committed — `SHA256SUMS` records its digest and says the file stays on the
  NAS — and the mp4's audio stream is AAC, which is lossy in amplitude. There is
  no in-tree bytes-exact audio reference to derive a bound from. The relative
  audio question is already gated by `coherence.audio_rms` (#1743) and is not
  touched.

## Port map

| File | Change |
|---|---|
| `scripts/ltx25-render-compare.py` | `--reference`, `--reference-sums`; reference identity assertion; `reference_bounds()`; `absolute.<arm>.blockiness_grid{8,32}` checks; `--b` optional when `--reference` is given; the panel's printed heading becomes conditional |
| `tests/scripts/test_ltx25_absolute_reference.py` | new; the red-before suite for every new assertion, on synthetic fixtures |
| `.agents/specs/ltx25-oracle-absolute.md` | this file |
| `.agents/specs/ltx25-dit-attn-flash.md` | §11.5 GAP 2 and `## Owed` reworded: the artefact-freedom half of #1854 moves here; prompt adherence stays open |
| `.agents/oracles/ltx-2.md` | one line: the reference now has a consumer |
| `docs/USAGE.md` | the new flags, and the checkpoints this row rendered against |

No product code. The engine is not changed by this row; it is MEASURED by it.

### `--b` becomes optional, and that is forced rather than preferred

The absolute question is about ONE render. Two ways to avoid the change were
considered and both are worse:

- **Pass our render as both `--a` and `--b`.** Every check passes trivially:
  `bit_identical` short-circuits the identity block, `coherence()` returns
  `k = 0.0` on a zero difference (`scripts/ltx25-render-compare.py:708-716`),
  and the alignment checks match a frame to itself. Landing that as the gate's
  invocation is `gate-comparing-shared-helper-proves-consistency-not-correctness`
  with the two sides of the comparison identical.
- **Make the reference arm B.** The identity bounds are already relocated out of
  the verdict by #1743, but `align.*` and `coherence.*` are NOT: they judge the
  verdict, and two different pictures of the same prompt fail them by
  construction. The run would exit 1 for a reason that is not a finding.

So the A/B blocks are guarded and skipped when `--b` is absent. When `--b` is
present, not one byte of their behaviour changes, and the tests hold that.

## Tests to port

There is no upstream test for this; upstream renders, it does not grade. The
suite is `tests/scripts/test_ltx25_absolute_reference.py`, hermetic, numpy-only,
no GPU, no network, and every case is red before the change:

| ID | Assertion | Red-before |
|---|---|---|
| T1 | A reference whose digest is not in `SHA256SUMS` is REFUSED, exit `EXIT_UNREADABLE`, and no report is written | no `--reference` flag exists |
| T2 | `--reference` absent leaves the panel `"checked": false` and adds NO check | — (holds the no-op) |
| T3 | A clean synthetic render passes all four blockiness checks against a synthetic reference | the checks do not exist |
| T4 | The same render with its 8x8 blocks flattened FAILS `absolute.<arm>.blockiness_grid8` and reads `WORSE_THAN_ORACLE` | the checks do not exist |
| T5 | A FULLY flattened render PASSES the ceiling — the hole, asserted — and FAILS `..._defined` | the checks do not exist |
| T6 | The bound in the JSON equals the reference's own recomputed per-frame max, not a literal | the checks do not exist |
| T7 | `--b` absent runs, gates, and does not emit any `align.*`, `coherence.*` or identity check | `--b` is required |
| T8 | `--reference` changes NO arm-to-arm check: the two reports agree entry for entry | the change could alter the A/B path |
| T9 | ONE render, TWO references, OPPOSITE verdicts: the ENFORCED bound moves with the reference | the checks do not exist |
| T10 | A degenerate REFERENCE is refused rather than used | the checks do not exist |
| T11 | A pure-noise render PASSES, and the case says so: that is the gate's limit | the checks do not exist |

T8 is the mutation-resistant form of "#1743's checks are untouched": it compares
reports rather than reading the diff.

**T9 exists because a mutation found the suite without it green.** Replacing
`frame_max` inside `reference_checks` with the real reference's literal
`1.1433929206406797` left every other case passing: the JSON still REPORTED a
computed bound, because `reference_bounds` still computed it, and the pass/fail
fixtures happened to agree with the literal. A gate enforcing a transcribed
number while reporting a computed one is worse than an honest literal, because
the report vouches for it. T9 runs one render against a clean reference and
against a blocky one and requires opposite verdicts, so no bound that is not read
from the reference in hand can produce both.

## Gates

1. `python3 tests/scripts/test_ltx25_absolute_reference.py` — the new suite.
2. `python3 tests/scripts/test_ltx25_render_compare.py` — the existing suite, green and unchanged.
3. `python3 tests/scripts/test_ltx2_oracle_goldens.py` — the reference's own digests still recompute.
4. `scripts/agent-preflight.sh`.
5. **The measurement.** One render of our engine at the manifest's request on
   `dgx:gpu0`, compared against the committed reference, and the resulting
   verdict recorded in `## Outcome` whether it passes or fails.

Gate 5 is the one that can only be run inside an `rc` lease. Everything else
runs anywhere.

## Dependencies

- `.agents/oracles/ltx-2.md` at `gateable = yes` and the committed goldens (#1864). Landed.
- `dgx:gpu0` through `rc`, for gate 5 only.
- The BF16 checkpoint set on the NAS. Present and size-matched: the transformer,
  the conv video VAE and the audio VAE at `/workspace/ltx25-fullmodel/ckpt/`,
  and the BF16 text encoder at `/workspace/ckpt/ltx-2.5/text_encoders/`.
- `ffmpeg`, already required by `ltx2-gen`.

## Work breakdown

- **W1** — this spec, committed before any code.
- **W2** — the tool change and the red-before suite. No GPU.
- **W3** — the lease: build, render at the reference's request, run the gate,
  record the verdict in `## Outcome`.

W2 does not wait on W3. If W3's render cannot be taken, W2 still lands the gate
and W3's absence is recorded under `## Owed` as `PENDING` — a gate with no
reading yet is visible debt, and it is not the same as no gate.

## Risks/decisions

- **The gate is loose, on purpose.** §4 states its measured sensitivity: about
  14% block-flattening strength. A subtler artefact passes. The alternative was
  a tighter bound with a chosen constant in it, which #1854 forbids by name.
- **Our render is a different picture from the reference.** Same prompt, same
  seed integer, different engine, so the sampler's noise is not the same draw.
  Every statistic that depends on content is therefore excluded (§5), and the
  one that gates is a grid ratio whose null does not depend on content.
- **`--offload cpu` has no exact counterpart on our side.** The oracle streamed
  the DiT from host memory; we do not. This changes memory traffic and not
  arithmetic, so it is recorded rather than matched.
- **N = 25 frames, one prompt, one seed, one geometry, bf16 only.** The gate is
  a gate over that request and claims nothing outside it.
- **A blockiness gate cannot see a render that is bad in another way, and the
  clearest case is NOISE.** `blockiness_bands` is a ratio: a render of pure noise
  has a huge step on the grid and an equally huge step off it, so it reads near
  1.0, inside the reference's band, and PASSES. C0 does not catch it either --
  `ltx25-dit-attn-flash.md` §10.8 already records that two identical sequences of
  pure noise clear all three C0 checks. This is why the passing reading is named
  `NO_WORSE_THAN_ORACLE_ON_BLOCKINESS` and not "as good as the oracle", why the
  panel still PRINTS mean sharpness beside it (pure noise reads far above the
  reference's 11.27), and why #1854's prompt-adherence half is the thing that
  closes the case rather than a fifth statistic.
  `tests/scripts/test_ltx25_absolute_reference.py` states this limit as an
  executable case rather than only here, so a later reader who assumes the gate
  is broader is contradicted by a test rather than by a paragraph.
- **A future re-render of the oracle would move the bound.** That is correct
  behaviour — the bound is the reference — but it means the reference's digests
  and this spec's numbers must move together. `test_ltx2_oracle_goldens.py`
  already recomputes those digests, so a silently swapped reference is red.

## Evidence

- The reference panel and its per-frame distribution, recomputed from the 25
  NAS PPM frames after verifying all 26 digests against the committed
  `SHA256SUMS` (26 of 26 matched).
- §2's mp4-versus-PPM table.
- §4's dose-response table.
- W3's render: `rc` job id, lease runtime, the binary and library sha256, the
  four checkpoint sha256 checked against the manifest, and the comparison JSON.

## Stop conditions

Stop and report, do not work around:

- a checkpoint sha256 that does not match the manifest;
- a reference digest that does not match `SHA256SUMS`;
- an unhealthy or unreachable fleet device (clearing one is a human's call);
- any state in which the only way to make a check green is to weaken it. A red
  that is true is this row's deliverable.

## Owed

- **Gate 5's READING is still PENDING, and the reason moved from the code to the
  fleet.** [#2140](https://github.com/mudler/vllm.cpp/issues/2140) is CLOSED
  (`8bfd3a542`) and its refusal is verified gone on the real bytes, so nothing in
  this port is now known to block the render. What blocks it is `dgx:gpu0`, the
  GB10 the reference was rendered on, which has been `unhealthy` and out of the
  pool with `worker_lost` since a heartbeat 3h20m before this was written.
  Clearing it needs an admin token and is a human's call, and no other fleet
  device can carry the run: `thor:gpu0` is `sm_110`, which the FA-2 architecture
  set excludes, `orin:gpu0` has no NAS mount for 70 GB of checkpoints, and
  `strix:gpu0` is AMD and has no CUDA at all. Substituting any of them would
  measure the box or the arm rather than the render. Owner: this row. §Outcome
  records what was established without the GPU, and what remains.
- **`--steps` is WIRED AND UNPROVEN END TO END, and that is the one thing this
  change lands without an executed path through it** ([#2130](https://github.com/mudler/vllm.cpp/issues/2130)
  closes the absence of the flag, not the absence of its proof). Every link is
  verified by inspection and none by execution: `main.cpp` assigns `vp.steps`,
  `vllm_c.cpp:1664` forwards it, `ltx2_video.cpp:4027` reads it. The lease DID
  pass `--steps 8`, and the render refused at the checkpoint load 76 s in, before
  the sampler ever resolved a sigma schedule, so no run in this tree has yet
  observed the value arrive. Nothing gates it: no test builds `ltx2-gen`.
  Unblocking it needs the render, so it is owed together with gate 5's reading
  and not separately. Owner: this row.

  **ONE FAILURE MODE IS NOW RULED OUT BY INSPECTION OF THE RECIPE, AND IT WAS THE
  DANGEROUS ONE.** A step override reaches two branches
  (`ltx2_video.cpp:4025-4073`): the schedule is computed from `steps` only when
  `phase.sigmas` is EMPTY, and a phase that carries its own sigmas either REFUSES
  the override or, if `allow_request_sigmas` is true, keeps its own schedule and
  ignores it. The second is silent, and a silent 30-step render against an
  8-step reference would carry a 3.75x denoise-budget confound while reporting
  success. `OneStagePhase` (`ltx2_pipeline.cpp:1124-1147`) sets `name`,
  `video_guidance`, `audio_guidance` and `noise_scale` and NO sigmas, and
  `OneStageRecipe` (`:1149-1163`) never assigns `allow_request_sigmas`, so it
  keeps the header default `true`. The harness's `--pipeline-kind one_stage`
  therefore takes the empty-sigmas branch, where `steps` is read. What is still
  owed is the RUNTIME observation, and the harness now extracts it rather than
  leaving it in a log: `VLLM_RENDER_PROGRESS` is on by default and `PhaseLog::Tick`
  (`render_phase_log.cpp:546-560`) writes `[render]   dit forward N  phase P step
  k/M`, whose denominator `M` is `sigmas.size() - 1` — the RESOLVED count.
  `ltx25-oracle-absolute-render.sh` collects the distinct denominators into
  `steps-observed.txt` and into `PROVENANCE`, and says so loudly when the set is
  not exactly `{8}`.
- **Five line anchors into `examples/ltx2_gen/main.cpp` are now STALE and cannot
  be repaired, because they live in the append-only issue index.** Adding
  `--steps` moved that file's later lines by +12, and
  `.agents/issue-index.md` rows 324, 325, 356, 373 and **821** cite it by line.
  Row 821 is this row's own #2130 entry, so one of the five is mine: the guidance
  `never-cite-a-line-number-in-an-append-only-file` names exactly this, and I
  wrote a line number into an append-only row anyway. AGENTS.md forbids editing a
  row, so the correction lives here instead: **`tools/oracle/ltx2_oracle.py:88` in
  that row should read `:89`**, and its `examples/ltx2_gen/main.cpp:306-451`
  should read `:318-476`. Owner: this row.
- **[#1854](https://github.com/mudler/vllm.cpp/issues/1854) sub-question 1,
  prompt adherence, stays OPEN and is not narrowed by this row.** It needs a
  vision-language model scoring frames against the prompt, pinned as an oracle
  with its own gateability measurement. This tree has none. Owner: this row.
- The three REPORTED statistics of §5 stay reported. Each has a stated
  derivation that failed, not an absence of effort. Owner: this row.

## Now

`ACTIVE`. W1 and W2 are complete. W3's first attempt refused at the checkpoint
load and that refusal became
[#2140](https://github.com/mudler/vllm.cpp/issues/2140), which is now CLOSED.
W3's second attempt did not reach a lease: `dgx:gpu0` is `unhealthy` and out of
the pool. Everything W3 can establish without a GPU is established and is in
`## Outcome` — both bf16 towers resolve, all four checkpoints are digest-verified
— and the reading itself stays `PENDING` under `## Owed`.

## Outcome

**What the gate is.** `--reference` admits only the #1864 render, by digest,
and holds the 8-grid and 32-grid blockiness ratios against a ceiling recomputed
from that render's own frames, with a collapsed-band count beside each. Twenty-one
cases, every one red against `origin/main`'s tool before the change and green
after. Seven mutations, each red: the identity assertion removed, the ceiling
widened tenfold, the collapsed-band guard disarmed, the bound transcribed as the
reference's own literal, the reference precondition removed, and the call site
deleted from each of the two paths. None of them is a build-matrix configuration
— `grep -rn ".github/"` finds only `VT_POOL_BYPASS: "1"` at `ci.yml:1605`, which
this row does not touch.

**Why the CI registration is one chained line rather than a commented block.**
It was six lines, and six lines inserted at `ci.yml:495` moved every later line by
six — silently staling **32 cited `ci.yml` line anchors across 30 files**, none of
which any checker validates, and every one of which was accurate before. The
registration is therefore written as a single `&&` chain that adds ZERO net lines,
and the explanation lives in `scripts/agent-preflight.sh` beside the same suite
rather than in the workflow. Verified after the change: for all 32 cited lines,
`origin/main`'s line N and this branch's line N are byte-identical. This is
`AGENTS.md` §Records in miniature — "never store a measurement of one file inside
another file" — and a line number in prose is that measurement.

**What the reference reads.** Recomputed from the 25 NAS PPM frames after all 26
committed digests verified, 26 of 26:

| statistic | mean | per-frame sd | per-frame min | per-frame max |
|---|---|---|---|---|
| `blockiness_grid8` | 1.042812 | 0.052844 | 0.947454 | **1.143393** |
| `blockiness_grid32` | 1.037230 | 0.059956 | 0.920299 | **1.148672** |
| `sharpness_mean` | 11.274039 | 0.278711 | 10.839144 | 11.760068 |
| `clipped_fraction` | 0.00165039 | 0.00022774 | 0.00122613 | 0.00210503 |

The two bold values are the gate's ceilings. Neither is written down anywhere:
they are printed above so a reader can see them, and recomputed on every run so
that changing `blockiness_bands` moves them.

**What the render did, and it is a finding.** `rc` job
`001c36e9-76b1-432c-9536-2d24c0e613d0` on `dgx:gpu0`, 44m45s of lease. All four
checkpoints staged to local disk and **all four sha256 matched the manifest**:
transformer `792a2bad...`, text encoder `ef724361...`, video VAE `685b06ee...`,
audio VAE `c52733d3...`. The CUDA unit gate ran first and passed 23 cases /
**806 assertions**, so the correctness floor is established rather than assumed.
Then the render refused after 76 s, at the load:

    'text_embedding_projection.video_aggregate_embed.weight' unpacks to
    in_features 376320 but the Gemma geometry gives 188160

The BF16 tower itself LOADED — `load.text_encoder` completed in 34.815 s at 76.64
GiB host — and the two caption projections did not.
[#2140](https://github.com/mudler/vllm.cpp/issues/2140) locates it:
`LoadProjection` hard-assumes torchao-NVFP4 packing, doubles a width that is
already logical, and its message blames a caller for the reading the function
itself made.

**What was NOT done, and why.** The NVFP4 tower was not substituted. It would
have produced a number, and the number would have compared a quantized text arm
against a bf16 reference — measuring the arm. `AGENTS.md` calls that state
`PENDING`, "not skipped, and not substituted", and this row takes it.

**What the timings cost, for whoever runs W3 again.** Build 1192 s (`ninja -j 4`,
two named targets). Staging 70.1 GB over CIFS: 803 s for the 42 GB DiT, 475 s for
the 26.3 GB tower, 28 s and 6 s for the VAEs. sha256 of all four: 91 s. The
render's own load reached the refusal in 76 s. A second lease with a warm
`$W/absref-bin` and `/root/ckpt` reaches the render in minutes.

**Two defects this row found in its own work, both by mutation rather than by
reading.** The first gate design was a two-sided band and it failed a render for
being BETTER than the reference; §4 records the measurement that killed it. The
second was that the ENFORCED bound was not pinned, only the reported one, so a
transcribed literal left the whole suite green; T9 closes it. Both are in the
history rather than quietly corrected, because the rejected designs are the ones
a later reader would propose.

**What this row does not claim.** One request, one geometry, one seed, bf16 only,
25 frames. Two of four panel statistics. Prompt adherence is untouched and open.
A pure-noise render passes, and a test says so.

### W3, second attempt: no lease, and the port is no longer what blocks it

The GPU was never reached. `dgx:gpu0` — the GB10, the box #1864 rendered on —
read `unhealthy (no contact 3h20m)` with `out of the pool  worker_lost`, and it
stayed there for the whole session. No job was queued against it: a queued job
against a dead worker is a lease held on a hope. It was not cleared, because
clearing needs an admin token and is the developer's call, and no `ssh` was
attempted, because a device that is unschedulable through `rc` is never a reason
to reach it another way.

**So the deliverable of this attempt is the elimination of every REMAINING
non-GPU unknown, on the real bytes, at `fe21faf63`.** The point is that the next
lease spends its wall on the render rather than on discovering a refusal, which
is exactly what the first attempt spent 44m45s doing.

**The four checkpoints are digest-verified, from the NAS, against the manifest.**
Not sizes: sha256, all four, all matching, 15m51s of CIFS reads.

| checkpoint | sha256 | verdict |
|---|---|---|
| `ltx-2.5-22b-dev-transformer-bf16.safetensors` | `792a2bad…c8e7584` | matches |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `ef724361…d16561d1` | matches |
| `ltx-2.5-video-vae-conv-bf16.safetensors` | `685b06ee…97dfce8d` | matches |
| `ltx-2.5-audio-vae-bf16.safetensors` | `c52733d3…0d54837a5` | matches |

**#2140's refusal is GONE, re-run on the bytes that produced it.**
`scripts/probe_ltx2_text_encoder_load.cpp` against the bf16 tower resolves
`video out=4096 in=188160` and `audio out=2048 in=188160` — the logical width,
not the doubled 376320 the old `LoadProjection` computed — with
`quantized_modules = 0`, in 32.9 s at 8.68 GiB peak, exit 0. The first attempt's
verbatim message was `'text_embedding_projection.video_aggregate_embed.weight'
unpacks to in_features 376320 but the Gemma geometry gives 188160`. It does not
occur.

**THE 42 GB BF16 DiT WAS THE OPEN QUESTION AND IT RESOLVES.** Every LTX-2.5
render this project has taken loaded the NVFP4 or the FP8 transformer; `8bfd3a542`
fixed the TEXT ENCODER, and nothing had established that the dev bf16 transformer
is not refused in turn. `scripts/probe_ltx2_dit_load.cpp` is new and answers it in
1.8 s off the header:

    resolved_arm    kNone
    contract        4091 tensors
    contract_bytes  37985180160 (35.38 GiB, what a load materializes)
    bound           4091 of 4349 file tensors
    unbound         258 tensors: audio_embeddings_connector video_embeddings_connector
    unported        none: the load does NOT need allow_unported_modules

Every one of the 4091 contract tensors is present under the ComfyUI prefix, at
the contract's shape, in a dtype this loader materializes, holding exactly the
bytes that shape requires — which is `MaterializeDitTensor`'s own BF16 check
(`ltx2_loader.cpp:499-506`). The 258 unbound tensors are the two
`*_embeddings_connector` families, which `UnportedFamilies` skips via
`LoadedElsewhere` (`:618-631`) and which `RefuseUnported`'s own message says
"are not in this list either and never will be" (`:654-656`), so the load needs
no `allow_unported_modules` and `ltx2-gen` does not pass one.

**WHY A PROBE AND NOT A LOAD, stated rather than left to be inferred.**
`Ltx2LoadDitFromSafetensors` and `Ltx2StreamDitToDevice` share their whole
prologue and differ only in what the per-tensor loop does with the bytes
(`:703-806`). The prologue is header-only and is where every DiT refusal in this
tree has happened, including #1148's. The loop is 35.38 GiB, and the CPU box this
ran on had 23-26 GiB available, so materializing would have OOM-ed a shared box
to re-measure a memcpy. **The probe therefore does NOT establish that the render
runs, and it prints that sentence itself before its `OK`.**

**The probe can say no, and that is measured rather than assumed.** Pointed at
the text encoder — a real safetensors file that is not a DiT — it exits 1 with
`REFUSED: ltx2 loader: 'hf_asset__chat_template.jinja' is U8 (NVFP4-packed) but
rank 1`. Pointed at the DiT it exits 0. An instrument that only ever passes is
not evidence.

**Independently confirmed by the tree's own case.** `ltx2 video: the SHIPPED
Lightricks checkpoints parse and load` / `the FULL bf16 dev DiT resolves onto the
L2 contract`, run against the same file with `LTX2_CHECKPOINT_ROOT` set: 1 case,
**18 assertions**, 0 failed, `quant=kNone, 4349 tensors, 4059 BF16 / 290 F32`.
That case and the probe are separate readers of the same header, and they agree.

**The three non-GPU gates are green at this head**: `test_ltx25_absolute_reference.py`
21 tests OK, `test_ltx25_render_compare.py` 65 tests OK,
`test_ltx2_oracle_goldens.py` PASSED.

**What is still owed is the render, and only the render.** The harness is
committed and unchanged in its request; the reference frames are on the NAS (25
PPM plus `audio.wav`, 26 files); the previous lease's binary cache is at
`$W/absref-bin` and will rebuild once, because its `SRC_SHA` predates `8bfd3a542`.
