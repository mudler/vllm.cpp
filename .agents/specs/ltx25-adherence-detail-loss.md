# SPEC — `LTX25-ADHERENCE-DETAIL-LOSS`: does the smoothness CAUSE the adherence gap?

Issue: [#2513](https://github.com/mudler/vllm.cpp/issues/2513).
Owner row: `LTX25-ADHERENCE-DETAIL-LOSS`.
Related: [#1854](https://github.com/mudler/vllm.cpp/issues/1854),
[#2295](https://github.com/mudler/vllm.cpp/issues/2295).

## Scope

`LTX25-PROMPT-ADHERENCE` W3 measured our render and it FAILS the adherence
bound. `.agents/specs/ltx25-prompt-adherence.md` `## Owed` says two things this
row exists for: **nothing owns closing that gap**, and the only hypothesis on
offer — that the render's smoothness costs it the fox-against-wolf margin — is
**untested, with the ablation that would test it specified nowhere**.

This row specifies and runs that measurement. It DIAGNOSES. It does not repair,
and it does not touch `src/` or `include/`. A repair is a separate row that this
diagnosis has to justify first, and if the diagnosis falsifies the hypothesis
the repair that would have been written is the wrong one.

In scope:

- the four trivial tone explanations, tested BEFORE anything about frequency;
- a radially averaged power spectrum per frame on both renders;
- the per-frame correlation between high-frequency energy and CLIP score;
- the per-frame table, and whether the frames that clear the bound are the
  sharp ones.

Out of scope: any change to the engine; any change to the gate or its bound; any
new render; any GPU work. Both frame sets already exist.

## Inputs, and what makes them the right ones

| Set | Location | Identity |
|---|---|---|
| ours | `/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z/r1/frame_*.ppm` | `sha256sum frame_*.ppm \| sha256sum` = `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae`, which is the digest `ltx25-prompt-adherence.md` records for the frames W3 scored |
| reference | `/mnt/nas_share/rc/ltx2-oracle/out/upstream_frames/frame_*.ppm` | 25 digests in `tests/parity/goldens/ltx2_oracle/SHA256SUMS` |

Both are verified before a pixel is read. This repository has gated against the
wrong checkpoint before, and the share is CIFS and `soft`-mounted, so a
streaming read can fail mid-file: the frames are copied to local disk and hashed
there rather than scored across the mount.

Geometry `320x192`, 25 frames, `steps=8`, `seed=42`, prompt
`A red fox walks slowly through a snowy pine forest at sunrise, cinematic.`

The engine that produced our frames is `main` at `7905607af`. **The claim this
row was dispatched with -- that no file whose PATH matches `ltx|video|vae|diffus`
changed since -- was true when it was made and is NOT true today**, which is why
it is re-measured here against a named revision rather than inherited. Against
`origin/main` at `63889449c`, **four** such files changed:
`ltx2_conditioning.h`, `ltx2_pipeline.h`, `ltx2_pipeline.cpp` and
`ltx2_video.cpp`, all in one commit, `565359bed`
(`fix(LTX25-ANCHOR-REPAIR)`).

That commit changes **no executable line**. Three of the four files have zero
non-comment changed lines, and the fourth has two, which are the tail of one
assertion's message string: `latent_cond.py:38-39` becomes
`latent_cond.py:40-41`. It is a sweep of upstream anchor line numbers in
comments. So the reading is not stale, and the reason it is not stale is a
measurement rather than an absence.

This is still a path-name scan and not a call-graph one, so it does not exclude
a shared primitive moving under the LTX path, and it is not enough to say
today's engine would score the same.

## Design

### The order is the design

**The trivial explanations run FIRST.** A global gain, a per-channel level
difference, a gamma difference or a compressed dynamic range would depress
sharpness, clipped fraction and CLIP score together **without one detail being
lost**, and the repair for that is not the repair for a lost band. Testing it
last is how a frequency story gets written about a tone difference.

Measured per frame on both sets: mean and per-channel mean, standard deviation,
5th/95th percentiles, min and max, and the best-fit gamma that maps our
luminance histogram onto the reference's. A tone difference is admitted as the
explanation only if correcting it also removes the sharpness and CLIP gaps.

### The spectrum, because "smoother" is a word

A radially averaged power spectrum per frame, on luminance, with a separable
window so the frame's own edges do not manufacture the high-frequency energy
being measured. The frames are `320x192`, so the radial bins run to the Nyquist
of the shorter axis and the two renders share a grid exactly (identical
geometry, so no resampling enters the comparison).

Reported: the per-bin ratio ours/reference, the crossover frequency at which
that ratio leaves unity, and the fraction of total energy we hold above it
against the fraction the reference holds. A uniform ratio across all bins is a
SCALE difference and belongs to the section above, not to this one. That
distinction is the reason both are measured in one run.

### The correlation, which is the actual test

The hypothesis says our frames score low BECAUSE they carry less fine detail.
That predicts a positive association between a frame's high-frequency energy and
its CLIP score against the true prompt. Computed three ways:

- within our 25 frames (n = 25);
- within the reference's 25 frames (n = 25);
- across all 50, where the between-render difference dominates and the
  coefficient is therefore not independent evidence — it is reported for
  completeness and read with that caveat.

The CLIP scores come from the landed instrument itself. The tool's own
`ClipAdherenceScorer` and `discrimination` are imported from
`scripts/ltx25-render-compare.py` rather than reimplemented, so the per-frame
numbers are the same instrument that produced the verdict, after the same
`assert_scorer_identity` check against the committed pin. The tool publishes
per-prompt reductions and not the per-frame matrix, which is why this row calls
the scorer directly instead of parsing a log.

**A correlation here is a hypothesis test, not a gate.** The coefficient, its n
and its null are stated together, and nothing in this row converts one into a
threshold. **If the within-render correlations are absent or point the wrong
way, the smoothness hypothesis is FALSIFIED**, and that is this row's most
valuable available result, because it redirects a repair nobody has started.

## Risks

- **n = 1 on our side.** The line `[ "$i" = 1 ] || rm -f "$D"/frame_*.ppm` in
  `scripts/ltx25-render-confirm.sh` deletes renders 2 and 3 by design, so no
  run-to-run spread of our own render exists in pixels. It is quoted rather than
  cited by number because the number is not stable: it is `:505` in `592e224e7`,
  which is the revision that produced these frames, and `:551` on `origin/main`
  at `63889449c`.
  Every statement about our render says n = 1. No single-render statistic is
  presented as a between-render one, which is the error
  `ltx25-prompt-adherence.md` corrected once already when a mean gap was divided
  by a within-render dispersion and called sigma.
- **Windowing manufactures or destroys high-frequency energy.** The window is
  applied identically to both sets, and the unwindowed spectra are computed too,
  so the conclusion cannot rest on the window choice.
  **[THIS RISK WAS REAL AND THIS MITIGATION DID NOT WORK.** The unwindowed
  spectra were computed and never compared against the windowed ones, and the
  conclusion did rest on the window choice: the two disagreed by a factor of
  thirteen and in sign. Computing a control is not checking it. See
  `## CORRECTION`, which replaces the estimator, and `## CORRECTION 2`, which
  finds the same shape a second time on a non-spectral axis.]**
- **A correlation over 25 frames of one continuous shot is not 25 independent
  samples.** Adjacent frames of a video are strongly dependent, so the effective
  n is smaller than 25 and any p-value computed against an i.i.d. null is
  optimistic. The row states the coefficient and this limit rather than a
  significance claim.
- **CLIP resizes a `320x192` frame to a `224x224` centre crop.** So the scorer
  never sees the frame's own Nyquist, and a band the spectrum finds may be one
  the instrument cannot see. This is a real limit on how far the mechanism can
  be traced and it is stated with the result.

## Tests and evidence

The measurement is a script under `scripts/`, run on CPU on the devbox, with its
full JSON committed under `tests/parity/goldens/` beside the frames' digests so
a reader can recompute any number in the report. The script verifies both
digest sets before it reads a pixel and refuses rather than scoring a set it
cannot identify.

**COMMITTING THE JSON DID NOT MAKE THE REPORT RECOMPUTABLE, and this sentence
was false for as long as it stood alone.** Nothing read the file -- no test, no
script, no CI step, no CMake target -- so `## Outcome` drifted away from it
figure by figure and the artifact that falsified the prose sat beside it in the
same commit. `tests/scripts/test_ltx25_adherence_detail_loss.py::TheCommittedGoldenIsREAD`
is the consumer: it asserts each published figure against the committed JSON, so
changing either side alone reds the suite. A golden with no reader is a
transcription that cannot fail.

## Gates

None. This row adds no gate and moves no bound. The adherence gate belongs to
`LTX25-PROMPT-ADHERENCE` and stays exactly as it landed. **Fitting our output to
raise a gate score is what AGENTS.md forbids and what this row must not become**,
which is the reason the row is scoped to diagnosis with the repair explicitly
elsewhere.

## Stop conditions

- `NEEDS_DECISION` rather than modifying the engine.
- `NEEDS_CONTEXT` if either frame set fails its digests.
- A falsified hypothesis is a RESULT and not a stop condition. It gets recorded
  as the finding, in full, and the row does not go looking for a second story to
  tell instead.

## Outcome

Measured on 2026-09-01 on the devbox: CPU only, no GPU, no lease, both frame
sets already on disk. `scripts/ltx25-adherence-detail-loss.py` takes the whole
reading and its full JSON is at
`tests/parity/goldens/ltx25_detail_loss/detail-loss.json`.

> **THIS SECTION IS THE WITHDRAWN FIRST DRAFT. READ `## CORRECTION` AND
> `## CORRECTION 2` INSTEAD.** It opened by saying "every number below can be
> recomputed rather than trusted", and that sentence is the reason it is marked
> rather than quietly rewritten: recomputed against its own committed JSON, this
> section's spectral table, its band shares, its correlation table, its per-frame
> high-frequency column, its temporal growth figures, its axis-to-ring table and
> its decisive intervention row are each falsified, **one of them by a change of
> SIGN**. `763ae655e` corrected the headline and the conclusion and re-derived
> none of the evidence under either.
>
> Every falsified figure below now carries an inline `[WITHDRAWN: ...]` marker
> giving the value the committed artifact actually holds.
> `tests/scripts/test_ltx25_adherence_detail_loss.py::TheCommittedGoldenIsREAD`
> asserts the corrected figures against that artifact, so the two can no longer
> drift apart in silence -- which is what happened here, because until that class
> existed the 2980-line golden had no consumer at all.
>
> **What survives is in `## CORRECTION 2` under "The falsification stands".**
> The hypothesis this row was created to test is still refuted, and it is refuted
> on legs that do not depend on any withdrawn number.

### THE HYPOTHESIS IS FALSIFIED, and its premise is false as well

The hypothesis was that our render is smoother than upstream's and that the lost
fine detail is what costs it the fox-against-wolf margin. Three independent
measurements refute it, and they refute it in the same direction.

**Our render is not spectrally smoother. It carries MORE fine-scale energy than
the reference, not less.** The radially averaged power spectrum has no crossover
at all: there is no frequency above which our profile stays below the
reference's. The ratio ours/reference rises with frequency instead of falling.

**[WITHDRAWN: every figure in the table below, and the "no crossover at all"
claim with them. All six are the HANN-windowed profile, which `## CORRECTION`
withdrew as the frame's own border. The committed `spectrum/ours_mean_profile`
is Moisan's periodic component, and against it the profile ratio is nearly FLAT
and slightly below unity across the range -- 0.9987, 0.9968, 0.9196, 0.9624 and
0.9567 at the five frequencies listed, with a minimum of 0.4358 at f = 0.0273
and a maximum of 1.2167 at f = 0.3789. `spectrum/crossover_cycles_per_pixel`
records **0.43359375**, so a crossover exists and the row's own harness found
it. "The ratio rises with frequency instead of falling" is not a property of the
estimator this row settled on.]**

| band | ours / reference |
|---|---|
| f = 0.0039 c/px (the lowest populated bin) | **1.1927** |
| f = 0.0898 c/px | **0.6076**, the minimum over all 64 bins |
| f = 0.15 c/px | 1.4074 |
| f = 0.25 c/px | 1.4834 |
| f = 0.3945 c/px | **2.8582**, the maximum |

That is not a blur. A blur is a monotone rolloff and this is a **band-shaped**
difference: more energy at the very bottom, a **hole in the middle**, and a large
excess near Nyquist. As energy shares of the frame,

**[WITHDRAWN: both rows. These are the Hann shares. `## CORRECTION` replaced the
high-band figure with **+22.68%** and never restated the table it came from. The
committed border-free (Welch) shares are mid 0.178624 against 0.196926,
**-9.29%**, and high 0.162515 against 0.132465, **+22.68%**.]**

| band | ours | reference | difference |
|---|---|---|---|
| mid, 0.04 to 0.14 c/px | 0.081937 | 0.089686 | **-8.64%** |
| high, 0.20 c/px and above | 0.081641 | 0.045918 | **+77.79%** |

The mid band is the one that carries object-scale structure at 320x192 -- 7 to
25 pixel periods, which is the scale of a fox's markings in this frame. We are
short there by 8.6% and long at fine scale by 78%.

**Sharpness was measuring the mid band, and "less sharp" was the right number
read as the wrong thing.** `sharpness_mean` is a mean absolute gradient, which
is dominated by mid-frequency contrast, so a mid-band deficit lowers it even
while the high band is nearly doubled. The reading in
`ltx25-oracle-absolute.md` was correct; the inference "smoother" that was drawn
from it was not, and no spectrum had ever been taken.

### THE CORRELATION POINTS THE WRONG WAY, in every form

The hypothesis predicts that within our render, the frames with more fine detail
score better. They score **worse**.

**[WITHDRAWN: all six figures. These are the Hann shares' coefficients, and the
row's reported estimator is Welch. The committed `correlation/*` holds ours
**-0.6195** / rho **-0.5985**, the reference **+0.2640** / rho **+0.2254**, and
pooled **-0.7963** / rho **-0.7812**. The SIGN of every row survives, which is
why the sign flip is what `## CORRECTION 2` keeps and the magnitudes are not.]**

| population | n | r(high-frequency share, CLIP) | rho |
|---|---|---|---|
| within OURS | 25 | **-0.5862** | -0.5723 |
| within the REFERENCE | 25 | **+0.3943** | +0.3938 |
| pooled | 50 | -0.8122 | -0.7582 |

The null is no association, r = 0. **No p-value is quoted**: 25 frames of one
continuous shot are not 25 independent samples, so the effective n is below 25
and any i.i.d. test would be optimistic. The pooled coefficient is reported for
completeness and is **not** independent evidence, because the between-render
difference dominates that pool. None of these is a gate and none of them is
compared against a threshold.

The **sign flip between the two renders** is the finding, not the magnitude. In
upstream's render more fine-scale energy goes with a better score, which is what
detail does. In ours it goes with a worse one, which is what noise does.

**[WITHDRAWN, and this one is a change of SIGN.** The mid-band coefficients are
**+0.3682** for ours and **+0.4513** for the reference in
`correlation/*/pearson_mid_band_vs_clip`. The published `-0.3690` is the same
magnitude with the opposite sign, so the sentence read as a second sign flip
where the committed artifact records the two renders AGREEING. The sharpness
pair `-0.4450` and `-0.1889` is correct and is the axis on which the two renders
do NOT contrast: both are negative, so nothing in it separates them.**]**

Against sharpness directly the coefficients are `-0.4450` for ours and `-0.1889`
for the reference; against the mid band, `-0.3690` and `+0.1715`.

### THE PER-FRAME TABLE INVERTS THE PREDICTION

Six of our 25 frames clear the bound of 35.9286 (the reference's own per-frame
minimum on its lossless frames). The question §"Design" asked was whether those
are the sharpest frames.

**They are the least sharp frames.** The overlap between the five best-scoring
frames and the five sharpest is **empty**.

**[WITHDRAWN: the `high-frequency share` COLUMN ONLY, in all eleven rows. Those
values are Hann shares, 0.047 to 0.065; the committed `per_frame_table`
`hf_energy_fraction` is the Welch share and runs 0.1517 to 0.1773 -- 0.154504 for
frame 0, 0.161151 for frame 2, 0.177252 for frame 21. Every other column, and the
finding this table exists for, are correct as printed and are asserted against
the artifact by `test_the_sharpest_five_and_the_best_scoring_five_share_no_member`.]**

| frame | CLIP | clears | sharpness | sharpness rank | high-frequency share |
|---|---|---|---|---|---|
| 0 | 36.7943 | YES | 10.0400 | **25 of 25** | 0.047195 |
| 2 | 36.4033 | YES | 10.2967 | **24** | 0.051021 |
| 1 | 36.2714 | YES | 10.3067 | **23** | 0.050562 |
| 12 | 37.0408 | YES | 10.4929 | 18 | 0.047528 |
| 15 | 36.7858 | YES | 10.6020 | 12 | 0.052270 |
| 16 | 35.9588 | YES | 10.4966 | 17 | 0.050114 |
| ... | | | | | |
| 21 | 34.9132 | no | **11.3850** | **1, our sharpest** | 0.064793 |
| 17 | 35.2323 | no | 11.1354 | 2 | 0.057456 |
| 5 | 33.4080 | no | 11.0838 | 3 | 0.055693 |
| 9 | 34.0052 | no | 11.0639 | 4 | 0.053751 |
| 22 | 33.1911 | no | 10.6927 | 9 | 0.060651 |

Our three best-scoring frames are our three LEAST sharp, and our sharpest frame
fails the bound. The full 25 rows are in the JSON.

### THE INTERVENTION SETTLES THE DIRECTION

A correlation within one render is an association. These two are interventions,
and they answer opposite questions with the same instrument.

**Removing detail from the reference does not reproduce our gap.** The observed
shortfall against the lossless reference is **2.7305** CLIP points.

| sigma | reference sharpness | reference CLIP | delta | argmax | wins |
|---|---|---|---|---|---|
| 0 | 11.2740 | 38.0024 | — | true | 25/25 |
| 0.30 | 11.2450 | 38.0098 | +0.0074 | true | 25/25 |
| 0.50 | 7.7847 | 37.2884 | -0.7140 | true | 25/25 |
| 0.70 | 5.2870 | 36.6043 | -1.3981 | true | 25/25 |
| 1.00 | 3.7785 | 37.0296 | -0.9728 | true | 25/25 |
| 1.50 | 2.7095 | 36.5300 | -1.4724 | true | 25/25 |
| 2.00 | 2.1885 | 36.0337 | -1.9687 | true | 23/25 |

Blurring the reference until its sharpness matches OURS costs it **0.1192** CLIP
points by interpolation, which is **4.4%** of the shortfall. Blurring it until
its sharpness is **a fifth** of ours -- a visibly destroyed frame -- costs
1.9687, still short of 2.7305, and it still ranks the true prompt first on 23 of
25 frames. **No achievable amount of smoothing reproduces our gap.** The sweep
is also non-monotonic (0.70 costs more than 1.00), which says a small detail
difference sits inside this instrument's own noise on this axis.

**THIS IS THE DECISIVE ROW, and it is the only SUBTRACTIVE intervention in this
section that holds its own precondition.** Every sigma above ranks the true
prompt FIRST, with a positive margin, on 25 of 25 frames down to sigma 1.5 and 23
of 25 at 2.0. It survives because the reference starts with 1.8068 of separation
against our 0.3370, so there is room to destroy the picture before the scorer
stops measuring. The table below has no such room, and it ran out. The ADDITIVE
intervention, the unsharp arm further down, holds its precondition too, which an
earlier version of this sentence denied by saying "only".

**Removing OUR excess fine-scale energy RAISES our score, sharply.**

**[WITHDRAWN: every row of this arm except sigma 0.30, and the paragraph under
it. The two columns this table omitted are the ones that decide whether its
numbers are readings at all.** The harness computed and recorded `argmax_label`,
`margin_to_best_decoy` and `per_frame_true_wins` for each row and printed none of
them, so the table was copied from a console line that could not show them. With
them, from the same committed `ours_lowpass_ablation.sweep`:

| sigma | delta | argmax | margin to best decoy | wins | readable? |
|---|---|---|---|---|---|
| 0.30 | -0.0029 | `true` | +0.3318 | 15/25 | **yes** |
| 0.50 | -0.4252 | `near:1` | **-0.0436** | 12/25 | **NO** |
| 0.70 | +0.8305 | `near:1` | **-0.3660** | 6/25 | **NO** |
| 1.00 | **+1.9131** | `near:1` | **-0.4036** | 5/25 | **NO** |

`scripts/ltx25-render-compare.py`'s S0 states the rule this row wrote itself and
then applied once: a scorer that cannot rank the true prompt first "measures
nothing here and no adherence number is published at all". From sigma 0.50 the
decoy `near:1` outranks the true prompt on our blurred frames, and the margin
walks off monotonically, so no larger sigma rescues it either. **The `+1.9131`
that decided this section comes from an arm on which the instrument had stopped
telling a fox from a wolf.** The only readable row is sigma 0.30, and its delta
is **-0.0029** -- zero. Blurring our frames by an amount the scorer can still
read does nothing to our score.

Why the reference's sweep survives the same operation and ours does not is
measured, not argued: separation. Ours starts at 0.3370, so half a pixel of blur
exhausts it; the reference starts at 1.8068.]**

| sigma | our sharpness | our CLIP | delta | frames clearing the bound |
|---|---|---|---|---|
| 0 | 10.6374 | 35.2719 | — | **6 of 25** |
| 0.30 | 10.6052 | 35.2690 | -0.0029 | 5 |
| 0.50 | 7.3703 | 34.8468 | -0.4252 | 4 |
| 0.70 | 5.0082 | 36.1025 | **+0.8305** | 12 |
| 1.00 | 3.5381 | **37.1850** | **+1.9131** | **23 of 25** |

This is the decisive row. The same operation costs the reference 0.97 CLIP
points and gains us 1.91, and it moves us from 6 frames clearing the bound to
23. **A render that is failing because it is too smooth cannot be repaired by
smoothing it further.** The hypothesis predicted the opposite sign, and 70% of
the 2.73 shortfall is recovered by throwing our fine-scale energy away.

The symmetric diagnostic is the FIFTH LEG of the falsification, and it is the
arm of this pair that holds its own precondition. An unsharp mask at sigma 1.0,
amount 1.0 raises our sharpness from 10.6374 to **19.0357** -- 69% above the
reference's own 11.2740 -- and buys **+0.4542** CLIP.

**[WITHDRAWN: "a quarter of what the blur buys", and "only one of them helps".**
Both clauses compared this arm against the `+1.9131` the table above withdraws,
so the ratio divides by a disqualified number and the polarity is inverted. The
only readable blur row is sigma 0.30 at **-0.0029**, which is no denominator at
all, and it is the BLUR arm that fails from sigma 0.50 up.
`arm_validity.arms["ours_unsharp_1.0"]` records argmax `true`, margin **+0.6547**
and **19 of 25** per-frame wins, so the sharpened arm is the one the scorer can
still read.]**

Read against its own committed validity, the arm says more than the withdrawn
clause did. Sharpening our frames **69% past the reference's own sharpness**
recovers **0.4542 of the 2.7305 gap -- 16.6%**. Adding fine-scale energy far
beyond what the oracle itself carries leaves five sixths of the shortfall in
place, on an arm that separates better than our unblurred render does. **That arm
is a DIAGNOSTIC and is not proposed as a repair**; it is here because adding
detail and removing it had to be tried in the same run, and the readable half of
that pair is the one that adds it.

### TONE IS NOT THE EXPLANATION EITHER, and it was tested first

| statistic | ours | reference |
|---|---|---|
| luma mean | 126.0390 | 120.4358 |
| luma sd | **46.4470** | 39.4230 |
| luma p05 / p50 / p95 | 64.352 / 114.032 / 191.070 | 63.113 / 114.990 / 190.336 |
| r, g, b means | 127.708 / 125.021 / 126.903 | 119.948 / 119.709 / 125.456 |
| clipped fraction | 0.001076 | 0.001650 |

The fitted map from our luminance quantiles onto the reference's is gain
**0.99144**, gamma **1.04654**: essentially the identity. Applying it and
rescoring moves our mean by **+0.1846**, to 35.4565, and takes 6 frames to 9. So
tone accounts for at most **6.8%** of the shortfall, and there is no global gain
or gamma error to find. **The fit's rms residual is 14.9379 levels**, which is
large, and that is itself informative: the two renders' tone curves do not
differ by a gain and a gamma, they differ in shape. Our render has **18% more
luminance spread** at a nearly identical median, which is the low-frequency
excess the spectrum's first bin shows, seen in the histogram.

### WHAT THE MEASUREMENT POINTS AT, stated as a candidate and not a finding

> **THIS WHOLE SUBSECTION IS RETRACTED.** `## CORRECTION` withdrew the
> separable-upsampler lead and `## Now` says this row names NO cause, but this
> text was left standing and unmarked, so it still reads as a live candidate
> naming three rows to dispatch against: `LTX25-TEMPORAL-UPSAMPLER`,
> `LTX25-TILED-DECODE` and `LTX25-DECODE-DTYPE`. **No row should be dispatched
> against anything below.** The axis-to-ring table is the Hann one, which was
> reading the frame's own wrap step -- ours is the larger, 73.39 against 49.48 --
> and border-free the contrast is 2.46x/2.22x for ours against 2.23x/1.78x for
> the reference, a difference too weak to name a stage. The temporal
> high-frequency growth figures beside it are wrong as well, and in the order
> they claim.
>
> The three ablations in "WHAT WOULD CONFIRM IT" remain the right way to
> attribute the excess to a stage. They are not confirmation of the candidate
> below, because there is no longer a candidate; they are how one would be found.

The excess is **not isotropic**, and that is the sharpest structural clue.
Near-Nyquist energy on the two frequency axes, against the radial ring around
them:

**[WITHDRAWN: every figure. Border-free (`bands/*_nyquist_axes` in the committed
JSON) the ratios are **2.46x / 2.22x** for ours and **2.23x / 1.78x** for the
reference. Both renders carry a comparable modest axis structure and the
difference supports naming no stage.]**

| | horizontal Nyquist | vertical Nyquist | radial ring | axis / ring |
|---|---|---|---|---|
| ours | 70.058 | 69.322 | 9.530 | **7.35x / 7.27x** |
| reference | 25.043 | 20.105 | 5.148 | 4.86x / 3.91x |

Both renders put more energy on the axes than on the ring, which a picture with
horizontal and vertical structure in it does. **Ours puts 2.8 to 3.4 times as
much there as the reference does, and its axis-to-ring contrast is half again
as large.** Energy concentrated on the sampling lattice's own axes at half the
sampling rate is the signature of a **separable upsampling stage** -- a
transposed convolution, a pixel shuffle, or a nearest-neighbour rung -- in the
VAE decoder or the spatial upsampler, rather than of scene texture, which
spreads that ring evenly.

**The temporal axis says the same thing from another direction.** High-frequency
share grows along the clip in both renders (+22.8% ours, +17.9% reference, so
the growth itself is not the anomaly), but the scores diverge:

**[WITHDRAWN: the growth figures, and their ORDER. `correlation/temporal`
records `ours_hf_last5_over_first5` = 1.0481 and
`reference_hf_last5_over_first5` = 1.0698, so the growth is **+4.8% ours against
+7.0% reference** -- ours grows LESS, where the published pair has it growing
more. The CLIP columns in the table below are correct and are the FFT-free part
of this paragraph; only the high-frequency sentence above them is withdrawn.]**

| | r(frame index, CLIP) | first 5 frames | last 5 frames |
|---|---|---|---|
| reference | **+0.6987** | 36.9820 | 38.9851, **+2.0030** |
| ours | **-0.3351** | 35.8053 | 34.2158, **-1.5895** |

**Upstream's render improves along the clip and ours decays.** Our first three
frames clear the bound and our last five are among our worst. A defect that
grows with frame index is not in the per-frame spatial path.

**[RETRACTED: the sentence below names a cause and three rows to dispatch
against it. This row names NO cause. The axis excess it rests on was the frame's
border, the temporal growth it rests on has the wrong sign of difference, and the
"8.6% short" mid band is **-9.30%** as a share at **1.0373x** absolute power,
which is equal rather than short. Do not open work against
`LTX25-TEMPORAL-UPSAMPLER`, `LTX25-TILED-DECODE` or `LTX25-DECODE-DTYPE` on the
strength of this paragraph.]**

So the candidate this measurement names is: **an upsampling or decode stage that
injects axis-aligned near-Nyquist energy, growing along the temporal axis, while
the mid band it should be filling stays 8.6% short.** The rows that own the
paths this would live on are `LTX25-TEMPORAL-UPSAMPLER`, `LTX25-TILED-DECODE`
and `LTX25-DECODE-DTYPE`, and nothing here decides between them.

**WHAT WOULD CONFIRM IT, and none of it is in this row.** Render the same
request with the temporal upsampler and the tiled decode each disabled, and take
the same spectrum: a stage that injects the excess removes it when it is
bypassed. Compare our decoder's output against the oracle's at the SAME latent,
which removes the sampler and the prompt from the comparison entirely and is the
only form that can attribute a band to a stage. And re-render at n > 1, because
this whole reading rests on one render.

### WHAT THIS DOES NOT SAY

**n IS 1 ON OUR SIDE.** Every number here about our render is one render, and
`[ "$i" = 1 ] || rm -f "$D"/frame_*.ppm` in `ltx25-render-confirm.sh` is why. The run-to-run spread of our own adherence
reading remains UNMEASURED, so the S1 verdict itself could still move; the
falsification above does not depend on that, because it rests on interventions
applied to the same 25 frames rather than on the gap's exact size.

**The instrument sees a 224x224 centre crop.** CLIP resizes a 320x192 frame, so
it never sees the frame's own Nyquist directly, and the horizontal downscale
folds our near-Nyquist excess back into the band it does see. That aliasing is a
plausible route from the excess to the score and this row does not measure it.
It is a limit on how far the mechanism is traced, not on whether the hypothesis
was refuted.

**[WITHDRAWN: "recovers 70% of the shortfall". The blur arm recovers nothing
the instrument can read. From sigma 0.50 a decoy outranks the true prompt on our
blurred frames, and the one readable row moves our mean by -0.0029. The
prohibition the sentence carries is UNCHANGED and now has a second reason: an
operation that raises a CLIP mean by destroying the scorer's ability to rank the
prompt first is not a repair, it is the gate-fitting this row forbids, wearing a
number.]**

**Nothing here says our render looks bad to a viewer**, and nothing here is a
repair. The blur arm recovers 70% of the shortfall and **must not become one**:
it would be fitting the output to the gate, on an instrument whose absolute
value is uncalibrated and a gate that is one sample deep.

## CORRECTION, 2026-09-01: the high-band headline was an ARTEFACT, and the upsampler lead is WITHDRAWN

A reviewer recomputed the band shares without a window and got the **opposite
sign** for the high band. Both readings were wrong, and chasing the disagreement
found a defect that changes two of this row's published claims. The reviewer's
figures reproduce on this row's own code to six digits, so nothing below is a
dispute about arithmetic.

**n is 1 on our side here too.** `scripts/ltx25-render-confirm.sh` deletes
renders 2 and 3, so every corrected figure in this section is a statement about
one render of ours, exactly as `## Risks` says of the original ones. Two other
specs route readers straight to this section, so the caveat is stated here
rather than left to be found in `## Risks` or in `## CORRECTION 2`.

### What the defect is

The DFT treats a frame as periodic, so the jump from its last row to its first is
a step the picture does not contain. That step leaks as `1/f^2` into the **low**
bins, which inflates the denominator of every band SHARE, and it lands on the
**fx and fy axes**, which is exactly where a separable-upsampling artefact would
be read.

**The two renders do not carry the same step, so it does not cancel in a ratio.**
Our frames' top-to-bottom wrap jump is **73.39** against the reference's
**49.48**, on near-identical interior steps (10.68 and 11.32).

Three whole-frame conventions therefore give three answers, two of them with
different signs:

| high band (>= 0.20 c/px) | ours | reference | delta |
|---|---|---|---|
| raw, unwindowed | 0.069311 | 0.092724 | **-25.25%** |
| Hann-windowed (what this row published) | 0.081442 | 0.045880 | **+77.51%** |
| Moisan periodic-plus-smooth | 0.105520 | 0.099757 | +5.78% |
| **Welch, interior tiles (now reported)** | **0.162547** | **0.132493** | **+22.68%** |

The raw figure is our own larger border. The Hann figure tapers roughly half the
frame's area to nothing, so it weights the picture's middle and discards its
edges. Moisan removes the wrap step exactly but leaves a low-frequency
counter-term whose size scales with the boundary jump, which the two renders
again do not share -- `test_the_periodic_counter_term_is_real_and_scales_with_the_boundary`
pins that as a measured property rather than a suspicion.

**Welch tiling has none of the three.** Interior 64-pixel tiles at 32-pixel stride
never touch the frame border, so no wrap step enters and no counter-term is
created; every tile is weighted equally, so there is no whole-frame taper bias.
It costs frequency resolution, which a band comparison does not need.
`test_welch_never_sees_the_frame_border` paints a cliff into the outermost rows
and columns and requires the Welch share to be unchanged to nine places while a
whole-frame estimator's share moves.

### What changes

**"77.79% MORE high-frequency energy" is WITHDRAWN.** The border-free figure is
**+22.68%** as a share, and the cleanest statement avoids shares altogether,
because a share moves when either band moves:

| absolute band power, Welch interior tiles | ours / reference |
|---|---|
| mid band, 0.04-0.14 c/px | **1.0373x** -- equal |
| high band, >= 0.20 c/px | **1.4031x** |
| total | 1.1437x |

So our render carries about **40% more absolute high-frequency power at
essentially equal mid-band power**, with 14% more total variance. The DIRECTION
of the original claim survives -- there is no high-frequency rolloff, so "our
render is smoother" is still false -- but the magnitude was inflated more than
threefold against the best estimator and the sign was never the reviewer's to
lose.

**THE SEPARABLE-UPSAMPLER LEAD IS WITHDRAWN, and this is the more serious of the
two.** The wrap step lands on the axes, and ours is the larger one, so the probe
was reading a border difference as a decoder signature. Border-free:

| near-Nyquist axis / radial ring | ours | reference |
|---|---|---|
| published (Hann) | 7.35x / 7.27x | 4.86x / 3.91x |
| **Welch, interior tiles** | **2.46x / 2.22x** | **2.23x / 1.78x** |

Both renders carry a comparable modest axis structure. The difference is weak and
it does not support naming a stage. **No reader should act on that lead**, and
this row names no replacement cause: it was wrong once about a mechanism and the
data it has does not identify a second one.

**One more quantity flipped and was never quoted, which is luck rather than
care.** The high-band temporal churn read ours 0.6892 against the reference's
0.6222 windowed, and reads ours **0.6415** against **0.7297** border-free -- the
order reverses. It appears in the committed JSON and in no prose, so no record
had to be corrected for it. It is recorded here because a reader comparing the
old JSON with the new one would otherwise find an unexplained sign change.

### What is UNCHANGED

> **TWO ITEMS IN THIS LIST DID NOT SURVIVE, and `## CORRECTION 2` replaces
> them.** The first bullet's `+1.9131` comes from an arm whose scorer had
> FAILED, and the estimator table below was transcribed rather than computed.
> The three middle bullets stand, and they are what the falsification now rests
> on.

Everything that decided the row. The mechanism falsification is pixel- and
CLIP-domain and no FFT convention enters it:

- **[WITHDRAWN]** the blur intervention, **+1.9131** on our frames, 6 of 25
  clearing the bound to **23 of 25**, against **-0.9728** on the reference's. A
  decoy outranks the true prompt on our frames from sigma 0.50 upward, so that
  arm is not a reading; see `## CORRECTION 2`;
- the reference blur sweep, which never reproduces the 2.7305 gap;
- tone, ruled out at **+0.1846**;
- the per-frame table, whose five best-scoring frames share **no member** with its
  five sharpest;
- the temporal split, ours **-0.3351** against the reference's **+0.6987**.

**[WITHDRAWN: "survives every estimator", and two of the six figures under it.**
That table was a transcription. The harness computed exactly ONE correlation --
Welch -- because `_conv_shares` returned whole-render scalars and a correlation
needs one number per frame, so no per-frame Hann or periodic coefficient existed
to quote. `-0.6216 / +0.1754` happens to be the periodic pair and is right;
`-0.5862 / +0.3943` is neither Hann nor Welch but the earlier radial-profile
reading this row's own headline used before Welch replaced it. The **raw**
convention -- the only one whose band delta reverses sign -- was missing from the
table altogether. All four are now computed per frame and committed under
`correlation/*/pearson_hf_vs_clip_by_convention`, and the claim is half true:

| r(high-band share, CLIP), per frame | ours | reference |
|---|---|---|
| raw | -0.5946 | **+0.0009** |
| Hann | -0.6097 | +0.4514 |
| periodic | -0.6216 | +0.1754 |
| **Welch (reported)** | **-0.6195** | **+0.2640** |

**Our negative coefficient is robust** -- four estimators inside a 0.03 band.
**The reference's positive one is not**: it runs from +0.0009 to +0.4514, a
factor of 500, and under the raw convention there is no association on the
reference side at all. The sign FLIP therefore depends on which estimator reads
the reference, so `## CORRECTION 2` does not lean on it.]**

And the **mid-band deficit is robust in sign across all four estimators**: -31.15%
raw, -8.88% Hann, -11.83% periodic, **-9.30% Welch**. That is a deficit in SHARE.
In absolute power the mid band is **1.0373x**, which is equal or fractionally
above, and the two must not be quoted as one quantity.

### The corrected reading

**[SUPERSEDED IN ONE CLAUSE by `## CORRECTION 2`: "Smoothing our frames raises
their score" rests on the withdrawn arm. Everything else in this paragraph
stands.]**

Our render carries **more** absolute high-frequency power than the reference
(1.40x) at **equal** mid-band power (1.04x), and within our own frames more of
that high-frequency energy predicts a **worse** CLIP score. Smoothing our frames
raises their score. So the smoothness hypothesis stays refuted, and what replaces
it is not a decoder-geometry story but an unattributed excess of fine-scale energy
that behaves like noise rather than detail. **What stage produces it is not
identified by this row**, and the ablations named above -- bypassing the temporal
upsampler and the tiled decode, and comparing decoder output at the same latent
-- remain the way to find out.

### The lesson this row owes its reader

A spectral SHARE is a ratio whose denominator is the whole frame, so any artefact
anywhere in the spectrum moves it. This row published a headline from a single
convention, asserted in its own `## Risks` that "the unwindowed spectra are
computed too, so the conclusion cannot rest on the window choice", and never
compared the two. They disagreed by a factor of thirteen and in sign. Every run
now prints all four conventions and the wrap-jump statistic beside the verdict,
so the next reader sees the spread instead of discovering it in review.


## CORRECTION 2, 2026-09-01: the decisive arm's SCORER had failed, and `## Outcome` was never re-derived

`## CORRECTION` corrected this row's headline band figure and withdrew its
mechanism. It re-derived neither the evidence under the headline nor the evidence
under the conclusion, and both were wrong. This section does that work. It is
written against
`tests/parity/goldens/ltx25_detail_loss/detail-loss.json`, and
`tests/scripts/test_ltx25_adherence_detail_loss.py::TheCommittedGoldenIsREAD`
asserts every figure below against that file, so "recomputable rather than
trusted" is now enforced rather than promised.

### The defect: a precondition asked of one arm is not a precondition of a run

This row's decisive result was **+1.9131**: blurring our own frames at sigma 1.0
raises their CLIP mean and takes 6 of 25 frames clearing the bound to 23 of 25.
`scripts/ltx25-render-compare.py`'s S0 already forbade publishing such a number.
It requires the scorer to rank the TRUE prompt first, and says that a scorer
which cannot "measures nothing here and no adherence number is published at all".

S0 ran **once**, on the unblurred reference. On our own blurred frames:

| sigma | delta vs ours | argmax | margin to best decoy | wins | readable |
|---|---|---|---|---|---|
| 0.30 | **-0.0029** | `true` | +0.3318 | 15/25 | **yes** |
| 0.50 | -0.4252 | `near:1` | -0.0436 | 12/25 | no |
| 0.70 | +0.8305 | `near:1` | -0.3660 | 6/25 | no |
| 1.00 | **+1.9131** | `near:1` | **-0.4036** | 5/25 | no |

From sigma 0.50 a decoy -- a near-miss prompt, the wolf against the fox -- scores
higher than the prompt the render was made from, and the margin walks off
monotonically, so no larger sigma recovers it. **The only readable row of this
arm is sigma 0.30, whose delta is -0.0029: zero.** Blurring our frames by an
amount the instrument can still read does nothing to our score.

The reference's sweep survives the identical operation to sigma 2.0 because it
starts with **1.8068** of separation against our **0.3370**. That is not a
difference in the operation; it is headroom.

Both arms recorded `argmax_label`, `margin_to_best_decoy` and
`per_frame_true_wins`. The reference's console line printed all three. Ours
printed none, and the spec table was copied from that line. The data was
present at every layer a machine reads and absent at both layers a human reads.

**The repair is the scope of the check, not its logic.** `adherence_arm_valid`
in `ltx25-render-compare.py` is now the single predicate; `scorer_precondition`
calls it and raises, every scored arm records its verdict under `scorer_valid`,
the run prints `READABLE` or `INVALID` beside every delta, and `arm_validity` in
the JSON is a ledger of all fourteen arms. `test_S0_AND_the_arm_check_are_the_SAME_predicate`
pins the two together, because two copies of a refusal rule drift and the copy
that drifts is the one nothing calls on the failing path.

### The falsification STANDS, on five legs, none of them withdrawn

The hypothesis was that our render is smoother than upstream's and that the lost
fine detail costs it the fox-against-wolf margin. It is still refuted.

1. **The reference blur sweep, which is the strongest leg and the only
   surviving SUBTRACTIVE intervention.** Every row holds its precondition:
   argmax `true`, a positive margin, 25 of 25 frames down to sigma 1.5 and 23
   of 25 at sigma 2.0. Blurring
   upstream's frames until their sharpness is **a fifth** of ours -- a visibly
   destroyed picture -- costs **1.9687** CLIP points against an observed gap of
   **2.7305**, and blurring them only to our own sharpness costs **0.1192**, or
   4.4% of it. No achievable amount of smoothing reproduces our gap. **This leg
   alone carries the falsification.**
2. **The empty overlap between our five sharpest frames and our five
   best-scoring ones.** FFT-free, and reproducible from `per_frame_table` with no
   spectral convention involved at all. Our three best-scoring frames are our
   three least sharp; our sharpest frame, 21, fails the bound.
3. **Tone, ruled out at +0.1846** on an arm that holds its precondition -- argmax
   `true`, margin +0.3701. The fitted gain 0.99144 and gamma 1.04654 are the
   identity, and correcting them takes 6 frames to 9 rather than to 25.
4. **The temporal split, ours -0.3351 against the reference's +0.6987**, also
   FFT-free. Upstream's render improves along the clip and ours decays.
5. **The unsharp arm, at +0.4542 on an arm that holds its precondition** --
   argmax `true`, margin +0.6547, 19 of 25 frames. Sharpening our frames 69%
   past the reference's own sharpness recovers 16.6% of the 2.7305 gap. It was
   framed against the withdrawn `+1.9131` and so read as the weak half of a
   pair; read against the gap it is the symmetric intervention, and it points
   the same way as leg 1. Adding detail does not close the gap, and removing it
   does not either.

### What this row does NOT lean on any more

- **The blurred-ours arm.** Invalid from sigma 0.50. Withdrawn.
- **"The within-render correlation survives every estimator."** Computed per
  frame rather than transcribed, ours is robustly negative (-0.5946 to -0.6216
  across four estimators) and the reference's positive coefficient is NOT: it
  runs +0.0009 raw, +0.1754 periodic, +0.2640 Welch, +0.4514 Hann. The sign
  flip depends on which estimator reads the reference.
- **The sharpness axis as a contrast.** Both renders are negative there,
  -0.4450 ours against -0.1889 the reference: the same sign, no contrast.
- **Any named cause.** See the retraction in `## Outcome`.

### What is measured, and what it is not

Our render carries **1.4031x** the reference's absolute high-band power at
**1.0373x** its mid-band power, with **1.1437x** the total. As shares, the mid
band is **-9.30%** and the high band **+22.68%**. A share and an absolute power
are different quantities and this row has already been read wrong once for
quoting one as the other: **there is no absolute mid-band deficit** -- 1.0373x is
equal or fractionally above -- and the deficit that exists is a share, which
moves when either band moves.

So there is no high-frequency rolloff, and "our render is smoother" describes a
rolloff. What the excess IS remains unattributed. The three ablations named in
`## Outcome` -- bypassing the temporal upsampler, bypassing the tiled decode, and
comparing decoder output against the oracle's at the SAME latent -- remain the
way to attribute it, and none of them is in this row.

### n IS 1 ON OUR SIDE, and that applies to every figure in this section too

`## Outcome` said "every statement about our render says n = 1" and then
`## CORRECTION` and this section moved the corrected figures out from under that
sentence. It is restated here so it covers them.
`[ "$i" = 1 ] || rm -f "$D"/frame_*.ppm` in `scripts/ltx25-render-confirm.sh`
deletes renders 2 and 3, so the run-to-run spread of our own render does not
exist in pixels and the S1 verdict itself could still move. The falsification
does not depend on the gap's exact size: legs 1 to 5 are interventions and
orderings applied to the same 25 frames.

The instrument also sees a 224x224 centre crop of a 320x192 frame, so it never
sees the frame's own Nyquist, and the horizontal downscale folds our
near-Nyquist excess into the band it does see. That is a limit on how far the
mechanism is traced, not on whether the hypothesis was refuted.

### The lesson, and it is not the one `## CORRECTION` drew

`## CORRECTION` concluded that a spectral share is a fragile statistic and made
the harness print four conventions. That was right and insufficient, because the
figure that then decided the row was not spectral at all.

The pattern underneath both is the same: **this row corrected its headline and
its conclusion twice and never re-derived the evidence under either.** The
committed JSON held the falsifying values the whole time -- the mid-band sign
flip, the crossover, the per-frame shares, the four invalid sweep rows -- and
nothing read it. A 2980-line artifact with no consumer is a transcription that
cannot fail. It has one now.

## Now

`DONE`. The hypothesis this row was created to test is refuted. **The repair
remains unowned and this row does NOT name a cause for it**, and the
separable-upsampler lead an earlier version of this line pointed at is withdrawn
in `## CORRECTION` and retracted in place in `## Outcome`.

What this row leaves, on **n = 1 for our render**:

- a **refuted** smoothness hypothesis, resting on the reference blur sweep, the
  empty sharpest-versus-best-scoring overlap, tone at +0.1846, the temporal
  split and the unsharp arm at **+0.4542** -- five legs, none of which depends
  on a withdrawn figure. The unsharp arm was framed against the withdrawn
  `+1.9131` and so read as the weak half of a pair; it holds its own
  precondition and recovers 16.6% of the gap while sharpening 69% past the
  reference, which is a leg rather than a footnote;
- **no measured absolute mid-band deficit**: the mid band is **1.0373x**, equal;
  the deficit is a **share**, **-9.30%** border-free;
- a **1.4031x absolute high-band excess** whose stage is unidentified;
- three named ablations that would attribute it, owned by no row;
- and one instrument repair that outlives the row: `adherence_arm_valid`, so
  that no future arm publishes a CLIP delta the scorer could not read.
