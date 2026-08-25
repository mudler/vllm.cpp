# LTX-2.5: the guided audio arm at the model's own defaults

Row: `LTX25-AUDIO-GUIDANCE-DEFAULTS`. Owning matrix row:
[`MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`](../model-matrix.md).
Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned, and **not edited by this
row**). Issue: [#1510](https://github.com/mudler/vllm.cpp/issues/1510).

This row does not edit [`ltx25-guided-video.md`](ltx25-guided-video.md). That
spec has a concurrent writer on another branch, and a second writer would make
it a lock under `AGENTS.md` `## Records`.

Upstream pins:

| Reference | Registry id | Revision | Gateable |
|---|---|---|---|
| Lightricks/LTX-2 | `ltx-2` ([`oracles/ltx-2.md`](../oracles/ltx-2.md)) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` | no ([#1864](https://github.com/mudler/vllm.cpp/issues/1864)) |
| vllm-project/vllm-omni | `vllm-omni` ([`oracles/vllm-omni.md`](../oracles/vllm-omni.md)) | `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`, the local checkout's `HEAD` | no, and `pin = UNPINNED` ([#633](https://github.com/mudler/vllm.cpp/issues/633)) |

Both revisions were read from local checkouts. `git rev-parse HEAD` ran on each
before any anchor below was taken, and the Lightricks checkout's `origin` is
`https://github.com/Lightricks/LTX-2.git` with a clean worktree.

**Read the second row exactly as written.** `.agents/oracles/vllm-omni.md`
records `pin = UNPINNED`, so `a4ea67a2` is the revision this row READ and not a
recorded parity pin. `.agents/oracles/ltx-2.md` names the same revision for
vLLM-Omni's `ltx2` registration, which is why the two agree, and that is a
citation rather than a pin as well. Section 3 uses vLLM-Omni as corroboration
only, never as a gate.

---

## 0. Honesty statement

This row **records a verdict and adds one instrument**. It changes no render
behavior. No default moves, no clamp is added, no normalization is added, and no
arm is refused.

Three limits apply to the verdict, and none of them is closable today:

1. **This is a source comparison, not an oracle run.** `.agents/oracles/ltx-2.md`
   records `gateable = no`, because nothing in this tree has ever run the
   Lightricks model on real weights. [#1864](https://github.com/mudler/vllm.cpp/issues/1864)
   owes that measurement. vLLM proper registers no LTX at the parity pin, so
   there is no primary oracle to reach for.
2. **Nobody has run upstream on this prompt.** There is no reference audio for
   the render #1510 reports. This engine's own two arms HAVE now been run
   head to head on one binary (section 5.4). That controls the port against
   itself, and it is not a comparison against upstream.
3. **Nobody has listened to either file.** Both verdicts in #1510 are instrument
   readings from one instrument.

---

## 1. Scope

In scope:

- The verdict on [#1510](https://github.com/mudler/vllm.cpp/issues/1510): the
  guided audio arm mirrors upstream at every step of the chain, so the loudness
  #1510 measures is upstream behavior and not a port defect.
- Four render-level trace fields for the **audio** guider's resolved scales,
  mirroring the four the video guider already has.
- One assertion block in the existing one-stage X0-space case that pins those
  four values against the upstream literals.
- Five more trace fields for the audio guider's **inputs and output** at the
  first guided evaluation, and the replay that turns them into a gate on what
  the DENOISER WAS HANDED rather than on what the engine resolved. Added by the
  fresh review's finding F1; section 6, decision D4 says why the resolved-scale
  fields could not answer that question.
- One new case that overrides all four audio guidance scales through the request
  and reads them back, which is what gates the "after `ApplyGuidanceOverrides`"
  claim and what makes a full cross-wire to the video row observable on four
  fields instead of one. Findings F2 and F3; decision D5.
- One new issue for the unversioned audio verify instrument, listed under
  [`## Owed`](#owed).

Out of scope:

- Any change to a rendered sample. See section 6, decision D1.
- Any edit to [`ltx25-guided-video.md`](ltx25-guided-video.md) or to
  [`ltx-2-5.md`](ltx-2-5.md).
- [`docs/USAGE.md`](../../docs/USAGE.md). A trace field crosses none of that
  document's triggers: no command, C application programming interface (API),
  configuration key, install step, or workflow changes.
- Re-filing the confound and the transcription error in section 5. The operator
  comments on #1510 directly about both.

---

## 2. Upstream anchors, the audio chain step by step

Every step from the guider to the bytes on disk was read at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` and compared with this tree. Each
upstream anchor was asserted for uniqueness with `git grep`, not only for
existence.

**The right-hand column is re-derived at this branch's head, not at the declared
base.** It was written against `d0d4f1f60fc4765dd4118dead8bcc1778b1df9b1` and
this row's own commits then moved two of its lines: `e9cd0fa8f` added four lines
above `PhaseGuidance` and the review repair added thirteen above the first
guided-step recorder. `ToDenoised` moved from `:4349-4350` to `:4366-4367` and
the video trace assignment from `:4065-4068` to `:4078-4081`. Every local anchor
in this spec was re-taken by content and asserted UNIQUE -- one match for
`out.video = ToDenoised(`, one for `im.trace.video_guidance_cfg_scale`, one for
each of the two `denoise_in.*_guider` assignments -- rather than by arithmetic on
the old numbers. An anchor a change's own diff staled is the ordinary shape here,
not an unusual one.

| Step | Lightricks/LTX-2 at `fd4ded7f` | This tree, at this branch's head |
|---|---|---|
| The four-term guidance sum | `packages/ltx-core/src/ltx_core/components/guiders.py:261-266` | `ltx2_pipeline.cpp:547-560`, `Ltx2MultiModalGuidance` |
| The standard-deviation renormalization | `guiders.py:268-271` | `ltx2_pipeline.cpp:562-579`, `Ltx2MultiModalGuidance` |
| The audio guider's own defaults | `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:59-68` | `ltx2_pipeline.cpp:956-961`, `Ltx2Params20` |
| Combination in x0 space, not velocity space | `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:480-482` and `packages/ltx-core/src/ltx_core/model/transformer/model.py:590-604`, the guider call at `utils/denoisers.py:202-203` | `ltx2_video.cpp:4366-4367`, `ToDenoised`, and `ltx2_denoisers.cpp:362-365` |
| Vocoder output activation | `packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py:436` | `ltx2_audio_vae.cpp:707-708` |
| Bandwidth-extension sum clamp, then the int16 write | `vocoder.py:630` and `packages/ltx-core/src/ltx_core/color/audio_mux.py:71-73` | `ltx2_audio_vae.cpp:851-852` and `minimax_h3_wav.cpp:71-75` |

The formula upstream computes, verbatim from `guiders.py:261-266`:

```python
pred = (
    cond
    + (self.params.cfg_scale - 1) * (cond - uncond_text)
    + self.params.stg_scale * (cond - uncond_perturbed)
    + (self.params.modality_scale - 1) * (cond - uncond_modality)
)
```

and the renormalization at `:268-271`:

```python
if self.params.rescale_scale != 0:
    factor = cond.std() / pred.std()
    factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
    pred = pred * factor
```

`Ltx2MultiModalGuidance` sums the same four terms left to right and applies the
same factor. It uses the unbiased `N-1` estimator, because that is what
`torch.std` uses by default.

**Three of the six upstream anchors need a name beside the line number, because
the token alone matches twice.** They are recorded here rather than left for the
next reader to trip over:

- `model.py:590-604` is `X0Model.forward`, which takes per-token `timesteps`.
  `LegacyX0Model.forward` at `:556-571` is a second `to_denoised` call site that
  takes a scalar `sigma`. `blocks.py:482` builds `X0Model`, so `:590-604` is the
  one the pipeline runs.
- `denoisers.py:203` is the pipeline's `audio_guider.calculate`. A second call
  site exists in a different package, `ltx-trainer`
  (`validation_runner.py:952`), and it is not on the render path.
- `constants.py:61` is `PipelineParams.audio_guider_params`. `cfg_scale=7.0`
  appears a second time at `:108`, inside `LTX_2_3_HQ_PARAMS`.

The remaining three anchors match exactly once over the whole upstream tree:
`guiders.py:265` for the modality term, `vocoder.py:436` and `:630`, and
`audio_mux.py:73`.

**The audio `cfg_scale` of 7.0 is constant across upstream's presets, and the
renormalization is what varies.** `LTX_2_3_HQ_PARAMS` at `constants.py:107-114`
also ships audio `cfg_scale = 7.0`, with `stg_scale = 0.0` and
`rescale_scale = 1.0`, that is with the renormalization applied at full strength.
Upstream therefore treats the audio arm's high `cfg_scale` as fixed and reaches
for `rescale_scale` when it wants a different amount of variance correction,
which is the same knob section 2.3 names.

### 2.1 Two absences that close the "something is missing" question

Both were searched for over the whole upstream tree, and both are negative
results rather than unchecked assumptions.

- **Upstream has no peak normalization anywhere.** `audio_mux.py:22-24` is named
  `normalize_audio_waveform`, and the name is misleading: its whole body is
  `return samples.T if samples.shape[1] != 2 else samples`, which transposes a
  stereo tensor to channel-last. It touches no amplitude.
- **Upstream has no clamp on the latent and none on the guided prediction.** The
  only clamps in the audio chain are the two this tree already mirrors: the
  vocoder's final activation at `vocoder.py:436` and the bandwidth-extension sum
  at `vocoder.py:630`, plus the int16 conversion clip at `audio_mux.py:71-73`.

### 2.2 The renormalization hypothesis in #1510 is false

#1510 asks whether the port might be missing the renormalization. It is not.
`rescale_scale` is `0.7` on **both** the audio and the video arms of the 2.5
one-stage recipe, in this tree and upstream. `Ltx2Params20` sets `0.7` twice
(`ltx2_pipeline.cpp:952` video, `:958` audio) against `constants.py:53` and
`:63`, and `Ltx2Params23` changes only the step count and the perturbed block.

### 2.3 Upstream names the phenomenon, and ships the audio arm above its own band

Three facts from the upstream documentation, which turn "upstream might sound
the same" into "upstream chose this trade and wrote it down":

- `packages/ltx-pipelines/docs/multimodal-guidance.md:14` describes
  `rescale_scale` as rescaling the guided prediction to match the variance of
  the conditional prediction, says it "Helps prevent over-saturation", and gives
  typical values of 0.5 to 0.7. Upstream names the exact phenomenon #1510
  reports and names the knob for it. The shipped value, 0.7, is the top of that
  band.
- The same file at `:11` gives `cfg_scale` typical values of 2.0 to 5.0, while
  `:42` and `constants.py:61` ship the **audio** guider at **7.0**. Upstream runs
  its audio arm above its own documented typical band on purpose. `:50` states
  the trade: a higher `cfg_scale` gives stronger prompt adherence.
- `packages/ltx-pipelines/src/ltx_pipelines/utils/args.py:975-978` documents
  `--video-rescale-scale` with "Higher values tend to decrease oversaturation
  effects", and the audio twin at `:1030-1037` adds "Experimental."

### 2.4 The command-line defaults are the same table

Every guidance flag in `utils/args.py` takes `default=<guider>.<field>` off
`detect_params(checkpoint)`. `args.py:934-935` binds `video_guider` and
`audio_guider` to `params.video_guider_params` and `params.audio_guider_params`,
and each `add_argument` reads a field off one of them, for example `:950` and
`:1032`. So for a 2.5 checkpoint the effective upstream default is
`cfg 7.0 / stg 1.0 / rescale 0.7 / modality 3.0`, which is exactly what
`ltx2_pipeline.cpp:956-961` resolves. `ti2vid_one_stage.py:274-289` is where the
one-stage entry point builds both `MultiModalGuiderParams` from those arguments.
This closes the gap that the command line might override the table. It does not.

---

## 3. vLLM-Omni agrees, independently

`.agents/oracles/ltx-2.md` sets the precedence: prefer vLLM-Omni wherever it
implements the pipeline. It implements this part, and it agrees on every point.
Read at `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`, the same revision that oracle
file records:

- `vllm_omni/diffusion/models/ltx2/ltx2_guidance.py:157-173` computes the same
  four-term sum and the same
  `factor = rescale * (cond_std / pred_std) + (1 - rescale)` renormalization, in
  float32, cast back to the input dtype. That is `ltx2_pipeline.cpp:547-579`.
- `vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:90-106`, `_official_guidance`,
  ships video `cfg 3.0` and audio `cfg 7.0`, with `stg 1.0`, `modality 3.0` and
  `rescale 0.7` on both arms. That is `constants.py:49-68` and
  `ltx2_pipeline.cpp:950-961`.

This is corroboration, not a gate: vLLM-Omni has no parity pin
([#633](https://github.com/mudler/vllm.cpp/issues/633)). It still moves the
verdict from one source to two independent ones, and the second of them outranks
Lightricks under `AGENTS.md`.

### 3.1 Three real differences between the two upstreams, none of which is a divergence here

`combine_guided_x0` (`ltx2_guidance.py:144-174` @ `a4ea67a2`) is not a
transcription of `MultiModalGuider.calculate` (`guiders.py:244-273` @
`fd4ded7f`). It differs in three places, and an earlier draft of this section
named one. None of the three moves a value on the path this row is about, so the
corroboration in section 3 survives -- but a reader who checks the two sources
finds three differences and must not have to work out which ones were counted.

1. **The renormalization reduction axis.** vLLM-Omni reduces the standard
   deviation **per batch item** (`reduce_dims = tuple(range(1, pred.ndim))`,
   `ltx2_guidance.py:166-168`), where Lightricks takes a whole-tensor
   `cond.std()` (`guiders.py:269`). Omni's own comment at `:162-165` says the
   official one-stage entry runs a single generated sample, so the two
   definitions coincide at batch 1. `Ltx2MultiModalGuidance` takes `count` over
   the whole modality tensor, and every render on this engine is batch 1, so it
   matches both. If this engine ever batches independent requests through that
   seam, the two part company. That is recorded under [`## Owed`](#owed) rather
   than changed now.
2. **The STG term is gated by a predicate rather than by its own scale.** Omni
   writes `if guidance.do_stg:` around the term (`ltx2_guidance.py:158-159`),
   and `do_stg` is `stg_scale != 0 AND stg_blocks` (`:46-48`). Lightricks always
   adds `stg_scale * (cond - uncond_perturbed)` (`guiders.py:264`). The two
   agree wherever `do_stg` agrees with `stg_scale != 0`, and they part only for
   a configuration with a non-zero `stg_scale` and an EMPTY `stg_blocks` -- for
   which Lightricks adds `stg_scale * (cond - cond)`, which is zero, because a
   perturbed pass with no perturbed block returns the conditional pass. Both
   therefore add nothing. On the 2.5 one-stage row `stg_blocks` is `[28]` and
   `stg_scale` is 1.0, so the predicate is true on both sides.
3. **An epsilon floor and a guard on the divisor.** Omni clamps `pred_std` to
   `torch.finfo(pred.dtype).eps` and replaces the factor with 1.0 wherever
   `pred_std <= eps` (`ltx2_guidance.py:169-171`); Lightricks divides straight
   through (`guiders.py:269`). The two differ only for a guided prediction whose
   standard deviation is at or below float epsilon, that is a constant tensor,
   which no arm of the render measured in section 5 produces.

`Ltx2MultiModalGuidance` mirrors Lightricks on all three, which is the required
polarity: vLLM-Omni corroborates and never becomes the mirror source.

### 3.2 Why the 2.5 row is sourced at Lightricks

vLLM-Omni's `_PIPELINE_RECIPES` stop at `("one_stage", "2.3")`
(`ltx2_recipes.py:161-166`), so 2.5 exists only at Lightricks. Nothing rests on
that gap for this row, and the chain that says so is the version resolver rather
than one `replace` call:

- `constants.py:83-88` is `LTX_2_3_PARAMS = replace(LTX_2_PARAMS, ...)`. That is
  the **2.0 to 2.3** delta -- the step count from 40 to 30 and the perturbed
  block from 29 to 28 -- and an earlier draft of this section cited it as the 2.3
  to 2.5 one.
- A 2.5 checkpoint does not resolve to that row at all. `_PARAMS_SINCE_VERSION`
  (`constants.py:130-133`) gives a checkpoint the newest generation it is at or
  above, so 2.5 lands on `LTX_2_4_PARAMS`, which is
  `replace(LTX_2_3_PARAMS, default_image_crf=LTX_2_4_IMAGE_CRF)`
  (`constants.py:124`) and moves the image CRF and nothing else.

So the 2.3 to 2.5 delta is exactly `default_image_crf`. All four audio guidance
scales are `PipelineParams.audio_guider_params`' own defaults
(`constants.py:59-68`) on every row of the lineage, which is what the assertions
in section 7 pin. The conclusion is unchanged and now rests on the correct
citation.

---

## 4. Verdict

**#1510 is mirrored upstream behavior, not a port defect.** Every step of the
audio chain from the guider to the int16 write matches Lightricks/LTX-2 at
`fd4ded7f`, and the two load-bearing steps match vLLM-Omni at `a4ea67a2` as
well. The port applies the upstream formula with the upstream defaults and
performs the upstream clamps and no others.

The audio arm is louder than the video arm because upstream gives it a
`cfg_scale` of 7.0 against the video arm's 3.0, and holds both at
`rescale_scale = 0.7`. A user who wants a quieter mix reaches for upstream's own
knobs, and both are already accepted on this engine: `kLtx2AudioRescaleScaleExtra`
(`--audio-rescale-scale`, toward 1.0) and `kLtx2AudioCfgScaleExtra`
(`--audio-cfg-guidance-scale`, down into the documented 2.0 to 5.0 band). There
is nothing to build for that, only something to record.

**The flatter envelope is part of the verdict, not an exception to it.** Sections
5.2 and 5.4 show the guided envelope really is flatter in a scale-invariant sense
on two independent renders, the second of them a controlled A/B, and
that is not refuted by anything here. It is the behavior of the guided trajectory
at upstream's own `cfg 7.0 / stg 1.0 / rescale 0.7 / modality 3.0`, produced by a
chain that matches upstream at every step. Calling it a port defect would need
one step of that chain to differ, and none does. Whether upstream's own render
sounds the same on this prompt is open, and it is
[#1864](https://github.com/mudler/vllm.cpp/issues/1864)'s oracle run that answers
it, not a change here.

---

## 5. What the measurement actually shows

The two files are the engine's own `audio.wav` from each run. Their sha256
values were re-derived here:

| Arm | Path | sha256 |
|---|---|---|
| unguided | `/mnt/nas_share/rc/ltx25-fullmodel/out/20260819T150230Z/768x448-25f/audio.wav` | `b662dd7dbbab97896fdd15ab981c5e05ecc23f7aaa0b67c02858cdbee0ceca70` |
| guided | `/mnt/nas_share/rc/ltx25-fullmodel/out/20260820T161441Z/768x448-25f/audio.wav` | `a43dc599d9a97fc86b3f0548f25def32dd43c7ade3d2521a961520d65b52a189` |

Every value in the issue's A/B table reproduces from the runs' own `verify.json`
except one: RMS at `-15.6266` and `-11.9378` dBFS, mono peak at `-4.3243` and
`-0.2871` dBFS, spectral crest at `52.049` and `61.842`, and the guided envelope
coefficient of variation at `0.07309848`. Four findings sit on top of them. The
operator comments on #1510 about the transcription error and the confound, and
section 5.4 is the operator's own clean re-run that removes the confound.

### 5.1 The unguided envelope coefficient of variation is 0.1118, not 0.1069

The run's own `verify.json` records `0.11184659763714917`. The value in the
issue table, 0.1069, is a digit transposition of 0.109629, which is the envelope
coefficient of variation of the **muxed AAC** of that same render. That is
exactly the source the issue says it did not use. The direction is unchanged and
slightly stronger: 0.1118 to 0.0731 is a fall of 34.6 percent, not 31.6 percent.
No PASS or FAIL verdict moves.

`active_fraction` is `1.0` on **both** arms in `verify.json`, so the issue
table's dash in that cell for the unguided arm reads as absent when the value is
present and identical.

### 5.2 The compression mechanism the issue asserts is not in the samples

#1510 states that the loud parts stop getting louder while the quiet parts rise.
Measured directly over the int16 stereo stream:

| Quantity | unguided | guided |
|---|---:|---:|
| peak absolute sample | 22398 | 32767 |
| samples at full scale | 0 | 2 |
| longest run at full scale | n/a | 1 sample |

Two samples at the ceiling out of 96,960, with a longest run of one sample, is
not flat-topping. The nine deciles of the absolute sample value show a near
uniform scaling of the whole distribution by about 1.45:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unguided | 789 | 1567 | 2373 | 3215 | 4107 | 5128 | 6302 | 7800 | 9900 |
| guided | 1111 | 2242 | 3409 | 4621 | 5950 | 7452 | 9177 | 11329 | 14539 |
| ratio | 1.41 | 1.43 | 1.44 | 1.44 | 1.45 | 1.45 | 1.46 | 1.45 | 1.47 |

It is a louder render, not a compressed one. The `-0.287` dBFS peak in the issue
is the **mono mixdown** peak, because the verifier downmixes to mono before
every headline metric. The stereo stream's own peak is `32767`, which is
`-0.000265` dBFS against a 32768 full scale.

**This refutes the mechanism and not the reading.** `envelope_cv` is
`env.std() / env.mean()` over 50 ms frames, so it is SCALE-INVARIANT: the 1.45
gain measured above cannot move it, and does not. The guided envelope is
genuinely flatter in a scale-invariant sense, and the fall from `0.1118` to
`0.0731` is a real property of the guided trajectory. What the deciles refute is
the stated CAUSE, that a ceiling compressed the loud frames while the quiet ones
rose. Both statements are needed, because refuting the cause does not make the
reading go away.

### 5.3 The A/B is confounded by two uncontrolled differences

Both `render.log` files were read. The prompt, the seed `20260818`, the
`768x448` geometry and the 25-frame count agree, read off each command line. Two
things do not:

- **Different binaries.** The unguided log carries no `[render]` line at all,
  because the per-forward tick from
  [#1413](https://github.com/mudler/vllm.cpp/issues/1413) did not exist when it
  ran.
- **Different flags beyond the one under test.** The guided command passes
  `--checkpoint-class full`; the unguided command does not. The unguided command
  additionally passes the four zeroing flags
  (`--video-stg-guidance-scale 0 --audio-stg-guidance-scale 0
  --a2v-guidance-scale 1.0 --v2a-guidance-scale 1.0`), which is the difference
  the A/B is about.

**The step count agrees, and the two sides do not prove it the same way.** The
guided log states it directly: its last tick reads
`dit forward 120  phase 0 step 30/30`. The unguided log emits no tick at all, so
its step count is INFERRED from `runguard.py --total-forwards 60`, which is 30
steps at two forwards each against the guided run's 30 at four. The inference is
consistent, and it is still an inference on the arm that carries no direct
evidence. Neither command line passes an explicit step count, so both take the
recipe's own 30.

### 5.4 A clean A/B removes the confound, and the phenomenon reproduces

**The operator ran this on a leased graphics processing unit (GPU); this row did
not.** Attribute it accordingly. Every number below was re-read here from the
run's own `verify.json`, `render.log` and `audio.wav`, and every one matched.

One binary served both arms: sha256
`bdeedd0143902fe806785ec4dc5fafe9d225276c07ce2835ea8c39581da707a3`, built from
`d0d4f1f60fc4765dd4118dead8bcc1778b1df9b1`, this row's base, for `sm_121a` with
zero compile errors, announcing `op=22 device=1 selected=vt-native`, which is the
shipped FA-2 default. It ran on `dgx:gpu0` inside an `rc` lease on an idle box.
Both arms passed `--checkpoint-class full`, so the flag that confounded section
5.3 is now held constant, and the only difference is the four guidance flags.
Evidence: `/mnt/nas_share/rc/ltx25-ab-1510/out/20260825T120744Z/`.

| Metric | unguided | guided | Delta |
|---|---:|---:|---|
| envelope coefficient of variation | 0.11115 | **0.08720** | fall of 21.5 percent, still under the 0.10 floor |
| RMS, dBFS | -15.559 | **-11.961** | **+3.60 dB** |
| peak, dBFS, verifier mono downmix | -4.041 | -0.693 | +3.35 dB |
| peak, dBFS, direct stereo samples | -2.462 | **-0.153** | +2.31 dB |
| samples at full scale | 0 | **0** | no clipping on either arm |
| longest run at full scale | 0 | 0 | n/a |
| spectral crest | 54.06 | 66.86 | n/a |
| active fraction | 1.0 | 1.0 | n/a |
| audio verdict | PASS | **FAIL** | n/a |
| video verdict | PASS | PASS | video is fine on both |

Four conclusions follow, and the fourth is the uncomfortable one.

1. **The phenomenon is real and reproduces on a controlled A/B.** RMS rises 3.60
   dB against the issue's claimed 3.7 dB, the envelope coefficient of variation
   falls and still fails the floor, and video passes on both arms. It is not an
   artifact of the confound.
2. **It is upstream's behavior, because the code is a verified mirror.**
   Reproduction plus the chain in section 2 gives the verdict in section 4: the
   model's own defaults do this.
3. **The correction in section 5.1 is confirmed on a second render.** This run's
   unguided arm reads `0.11115`, against the `0.11184` measured on the retained
   WAV. The issue's `0.1069` is now wrong on two independent renders.
4. **Part of the reported magnitude WAS the confound.** The confounded pair
   reported a guided coefficient of variation of `0.0731` and a mono peak of
   `-0.287` dBFS. The clean pair reads `0.0872` and `-0.693` dBFS. The effect is
   real and it is **smaller** than #1510 states once the binary and the
   `--checkpoint-class` difference are controlled. The old figures are not
   carried forward anywhere in this spec.

**On "near-clipping", with numbers.** The phrase holds: `-0.153` dBFS is about
half a percent from the ceiling. "Pushed into its ceiling", with the loud parts
compressed, does not: **zero samples reach full scale on either arm** and the
longest run at full scale is zero. It is a hot mix, not a saturated one. The two
peak rows differ because the verifier downmixes to mono before its headline
metrics, which is the same instrument property section 5.2 names.

### 5.5 The verifier that produced the FAIL verdict is not in this repository

`verify_render.py` has one copy, on a Common Internet File System (CIFS) share:
`/mnt/nas_share/rc/ltx25-fullmodel/job/verify_render.py`, sha256
`57cf92846506be961e3c6ab3c9198de5608d0e85bfd1eb992eb91dda1c0fa563`. It is
unversioned, untested and unreachable from any gate in this
tree. `grep -rn envelope_cv src include tests scripts` returns nothing and exits
1.

**The scope in that command is load-bearing, and the unscoped form was
self-invalidating.** `grep -rn 'envelope_cv'` over the whole repository now
matches this file and the #1905 row in
[`../issue-index.md`](../issue-index.md), because both describe the instrument.
The claim worth checking is that no GATE in this tree reaches it, and the four
directories above are where a gate could live.

Two properties of that instrument bear on the verdict it produced:

- Its `envelope_cv` rests on 20 non-overlapping 50 ms frames over a 1.01 s clip.
- Its `active_fraction` threshold is **relative**, at 40 dB below the loudest
  frame, so a constant-level signal reads `1.0` at any absolute level. That
  explains why both arms read `1.0`.

This is the new issue this row files. See [`## Owed`](#owed).

---

## 6. Design, risks and decisions

### D1. No render behavior changes

**Rejected: adding a peak normalization, a latent clamp or a lower default.**
The verdict is that this port mirrors upstream. Changing what it renders would
break the mirror, and `AGENTS.md` `## vLLM is the reference` forbids diverging
because a secondary source would do it differently. Upstream ships the knob for
this exact phenomenon and this engine already accepts it.

### D2. The gap that is real: the audio arm is uninstrumented at the render level

`Ltx2ConditioningTrace` carries `video_guidance_cfg_scale`,
`video_guidance_stg_scale`, `video_guidance_rescale_scale` and
`video_guidance_modality_scale`
(`include/vllm/multimodal/ltx2_video.h:1092-1095`), set on the shipped render
path at `src/vllm/multimodal/ltx2_video.cpp:4078-4081`. **Before this row there
was no audio counterpart** -- decisions D4 and D5 carry what the four this row
adds can and cannot see. The existing case "ltx2 one_stage: all four guidance arms are
combined in X0 space (#1092)" pins the video row's four scales at
`tests/vllm/multimodal/test_ltx2_video.cpp:9453-9456` and asserts nothing about
the audio row.

So a change that moved the **audio** guider's resolved scales on the shipped
render path is invisible at the render level today. Section 8 measures that
claim rather than asserting it.

**What already exists, and why it is not the same pin.** Two cases pin the audio
guider parameters: `tests/vllm/models/test_ltx2_pipeline.cpp:1040-1047` and
`tests/vllm/multimodal/test_ltx2_video.cpp:8185`. Both read
`Ltx2DetectPipelineParams` directly. Neither goes through a render, so neither
can see a render that resolves different scales than the params function
returns. This row adds the render-level pin and does not duplicate the two
params-level ones.

### D3. The new assertions compare against literals, not against the recipe row

The video block asserts `t.video_guidance_cfg_scale == row.cfg_scale`, where
`row` is read from the same recipe the engine resolved. That form proves
consistency between two reads of one source. It cannot see a change that moves
both. The audio block therefore asserts the four **literal** upstream values
from `constants.py:59-68` directly against the trace, and pins the recipe row's
own four values beside them. Mutation M2 in section 8 is the measurement that
says the weaker form would have stayed green.

### D4. The resolved scales cannot gate the guider's CONSUMPTION, so the arms are recorded too

The four `audio_guidance_*` fields are read at
`src/vllm/multimodal/ltx2_video.cpp:4082-4085`, off the resolved `PhaseGuidance`
entry. `denoise_in.audio_guider = audio_guidance` at `:4386` is a second
assignment, and no gate observed it. **Measured by the fresh review:** replacing
that one line with a copy carrying `cfg_scale = 1.0, rescale_scale = 0.0` -- CFG
off and the renormalization disabled, which is #1510's own subject -- left
`test_ltx2_video`, `test_ltx2_dfr`, `test_ltx2_retake`, `test_video_engine` and
`test_diffusion_device_seam` GREEN, while the same replacement on
`denoise_in.video_guider` at `:4385` red three cases and five assertions.

The asymmetry was not a property of the two arms. It was the video arm's
**replay**: `RecordFirstGuidedStep` records the four video passes and
`video_denoised`, and the #1092 case re-combines them through the shipped
`Ltx2MultiModalGuidance` on the recipe's own params and requires the result to
equal `video_denoised` exactly. Nothing recorded the audio passes, so the same
replay had no inputs on that side.

**Rejected: asserting on the rendered `audio.wav`.** A byte comparison against a
golden would gate the whole chain including the VAE and the vocoder, would move
on any unrelated numerical change, and would need a checked-in artifact. The
replay is exact, is 1152 floats wide on the fixture, and names the one function
whose inputs it reproduces.

**Rejected: replaying with `t.audio_guidance_*`.** That is decision D3's trap one
level down: a replay fed the trace's own scales agrees with the pipeline whenever
a change moves both, which is exactly the shape a cross-wired trace field has.
The replay uses `recipe.phases[0].audio_guidance`.

### D5. The default render discriminates ONE field of four, and only a request can fix that

At the shipped defaults the two arms differ in `cfg_scale` alone -- 3.0 video
against 7.0 audio -- while `stg 1.0`, `rescale 0.7` and `modality 3.0` are
identical on both arms across the whole 2.3-to-2.5 lineage (`constants.py:49-68`,
inherited unchanged through `LTX_2_3_PARAMS` and `LTX_2_4_PARAMS`). **Measured:**
wiring all four audio trace fields to `video_guidance` red exactly ONE assertion
in the #1092 case. Three quarters of that cross-wire read as correct, because the
wrong source carried the right number.

No assertion over a default render can close that, because there is nothing to
observe: the two rows agree on three fields. The only way to make them differ
without moving a shipped default -- which decision D1 forbids -- is a request
override. The new case sends `--audio-cfg-guidance-scale 2.0`,
`--audio-stg-guidance-scale 0.5`, `--audio-rescale-scale 0.25` and
`--v2a-guidance-scale 2.0`, each different from BOTH rows, and asserts the
precondition that they are before it asserts anything about the render. The same
cross-wire reds four assertions there. The four values keep every pass running
(`cfg != 1`, `stg != 0`, `modality != 1`), so the render assembles the same four
passes the default one does.

That case also carries the "after `ApplyGuidanceOverrides`" claim the header
makes. **Measured:** pointing the four fields at `phase.audio_guidance`, the row
BEFORE the overrides, left the whole `test_ltx2_video` binary green; the
identical mutation on the video four reds in the a2vid override case, which
overrides a video scale and reads it back. Nothing did that on the audio side.

### R1. A trace field is an instrument, and an instrument can be wired wrongly

The mitigation is the reachability mutation in section 8, M3: delete the
production assignment and confirm the case reds. A field that is declared and
never set reads `0.0`, which is not `7.0`, so the assertion is not satisfiable
by an unwired field.

### R2. The batch-1 coincidence in section 3.1

Recorded under [`## Owed`](#owed). It is not reachable today, because every
render on this engine is batch 1.

---

## 7. Tests

Both live in `tests/vllm/multimodal/test_ltx2_video.cpp`, and no new executable
is added. Each enters the engine through `vllm::multimodal::LoadVideoEngine` and
`engine->Generate`, so every assertion below reads a completed render rather than
a hand-built type.

**One existing case changes:** "ltx2 one_stage: all four guidance arms are
combined in X0 space (#1092)". It gains two blocks.

1. The resolved-scale block. It pins `recipe.phases[0].audio_guidance`'s four
   scales against the upstream literals `cfg 7.0`, `stg 1.0`, `rescale 0.7`,
   `modality 3.0` (`constants.py:59-68`), and pins the four
   `audio_guidance_*` trace fields, read off the render, against the same
   literals.
2. The **replay** block, added by the fresh review's finding F1 and argued in
   decision D4. It re-combines `audio_first_cond`, `audio_first_uncond`,
   `audio_first_perturbed` and `audio_first_modality` through the shipped
   `Ltx2MultiModalGuidance` on `recipe.phases[0].audio_guidance` and requires the
   result to equal `audio_first_denoised` EXACTLY, with the three
   arm-is-a-different-forward guards and the non-vacuity guards the video block
   carries. This is the block that observes what `denoise_in.audio_guider` was.

**One case is new:** "ltx2 one_stage: the four AUDIO guidance overrides reach the
render and the trace (#1510)", argued in decision D5. It renders once with all
four audio guidance extras set to values that differ from BOTH the audio row and
the video row, asserts that precondition per field before it asserts anything
about the render, then checks:

1. the four `audio_guidance_*` trace fields carry the OVERRIDDEN values;
2. the four `video_guidance_*` fields are untouched, so the extras are
   audio-scoped;
3. the replay at the OVERRIDDEN scales reproduces `audio_first_denoised`
   exactly, so the overrides reached the denoiser and not only the trace;
4. the replay at the recipe's DEFAULT scales does NOT, so the render can say
   which of the two the denoiser used.

---

## 8. Gates

Run from the worktree root.

1. **Configure and build.**

   ```sh
   cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   cmake --build build -j 4 --target test_ltx2_video
   ```

2. **Focused gate.** The case must pass with a non-zero assertion count. An
   `assertions: 0` line is a skip wearing a pass.

   ```sh
   ./build/tests/test_ltx2_video -tc='ltx2 one_stage: all four guidance arms are combined in X0 space (#1092)'
   ./build/tests/test_ltx2_video -tc='ltx2 one_stage: the four AUDIO guidance overrides reach the render and the trace (#1510)'
   ```

3. **Full suite.**

   ```sh
   ctest --test-dir build --output-on-failure
   ```

   **`PENDING` on the authoring host, for a named external resource.** Building
   the whole test suite means building every remaining test executable, and the
   box was at 98 percent with 13 GB free and falling while other sessions built
   in parallel.
   An out-of-space condition here does not fail loudly: it makes the record
   checkers emit false policy refusals, so the gate would read as a code verdict.
   What ran instead is the complete set of executables that compile the changed
   header, listed by `grep -rln 'multimodal/ltx2_video.h' src/ include/ tests/`
   and its one transitive hop through `ltx2_connector.h`:

   ```sh
   cmake --build build -j 4 --target test_ltx2_video test_ltx2_pipeline test_diffusion_device_seam
   ./build/tests/test_ltx2_video
   ./build/tests/test_ltx2_pipeline
   ./build/tests/test_diffusion_device_seam
   ```

   Re-run at the review-repair head, on the same three executables and the same
   `grep -rln` derivation of the list.

   **Read the CASE column as the result. The `test_ltx2_video` assertion figure
   is not a measurement and must not be re-quoted as one.** That binary's
   whole-run assertion total is nondeterministic on an identical binary over an
   unchanged tree ([#1885](https://github.com/mudler/vllm.cpp/issues/1885)); the
   ten draws taken across this row and its polish pass are 4714, 4715, 4716,
   4717, 4719, 4721, 4769, 4770, 4771 and 4775. Three of them -- 4770, 4771 and
   4769 -- were taken during the polish pass over trees whose only differences
   are comment text and one block's position, so no assertion was added or
   removed between them. The cell below therefore states which draw it is rather
   than a count. A `-tc`-filtered count IS stable, which is why the mutation
   rows further down carry one.

   | Executable | Exit | Cases (the result) | Assertions | Verdict |
   |---|---:|---|---|---|
   | `test_ltx2_video` | 0 | 106 of 106 passed | one draw: 4775 passed, 0 failed. Band 4714-4775 on an unchanged tree (#1885) | SUCCESS |
   | `test_ltx2_pipeline` | 0 | 60 of 60 passed | 3475 of 3475 passed | SUCCESS |
   | `test_diffusion_device_seam` | 0 | 7 of 7 passed | 49 of 49 passed | SUCCESS |

   Continuous integration runs the full suite on the pull request, which is the
   lane that closes this.

4. **Preflight.**

   ```sh
   scripts/agent-preflight.sh --staged
   ```

5. **Mutations.** Each one rebuilds the target, runs the focused gate, and
   restores the tree so that `git diff` is empty. A mutation that fails to
   build, or one the build system skips, reads exactly like a passing test, so
   each row records the compile-error count and whether the target relinked.

   | Id | Tree | Mutation | Focused case | Expected |
   |---|---|---|---|---|
   | M0 | before this row | `ltx2_pipeline.cpp:958`, audio `rescale_scale` `0.7` to `0.0` | #1092 | GREEN, the arm is invisible |
   | M1 | after this row | `ltx2_pipeline.cpp:958`, audio `rescale_scale` `0.7` to `0.0` | #1092 | RED |
   | M2 | after this row | `ltx2_pipeline.cpp:956`, audio `cfg_scale` `7.0` to `3.0` | #1092 | RED |
   | M3 | after this row | delete the production assignment of the four audio trace fields in `src/vllm/multimodal/ltx2_video.cpp` | #1092 | RED, reachability |
   | M4 | review repair | `ltx2_video.cpp:4386`, hand `denoise_in.audio_guider` a copy carrying `cfg_scale = 1.0, rescale_scale = 0.0` | #1092 | RED |
   | M5 | review repair | the same mutation | the override case | RED |
   | M6 | review repair | delete the production assignment of the five `audio_first_*` trace fields | #1092 | RED, reachability |
   | M7 | review repair | the four audio trace fields read `phase.audio_guidance`, the PRE-override row | the override case | RED |
   | M8 | review repair | the four audio trace fields read `video_guidance` | the override case | RED on all four |

   All nine are **measured**. Every one built with zero compile errors, relinked
   `tests/test_ltx2_video`, and was restored so that `git diff` reported the
   intended change and nothing else.

   | Id | Diff | Compile errors | Relinked | Exit | Cases | Assertions | Result |
   |---|---|---:|---|---:|---|---|---|
   | M0 | 1 insertion, 1 deletion | 0 | yes | 0 | 1 passed | 67 of 67 passed | **GREEN**, as predicted |
   | M1 | 1 insertion, 1 deletion | 0 | yes | 1 | 1 failed | 75, 2 failed | **RED** |
   | M2 | 1 insertion, 1 deletion | 0 | yes | 1 | 1 failed | 75, 2 failed | **RED** |
   | M3 | 4 deletions | 0 | yes | 1 | 1 failed | 75, 4 failed | **RED** |
   | M4 | 4 insertions, 1 deletion | 0 | yes | 1 | 1 failed | 1 failed, on the replay | **RED** |
   | M5 | 4 insertions, 1 deletion | 0 | yes | 1 | 1 failed | 1 failed, on the overridden replay | **RED** |
   | M6 | 5 deletions | 0 | yes | 1 | 1 failed | 1 failed, `REQUIRE(an > 0)` | **RED** |
   | M7 | 4 insertions, 4 deletions | 0 | yes | 1 | 1 failed | 4 failed | **RED** |
   | M8 | 4 insertions, 4 deletions | 0 | yes | 1 | 1 failed | 4 failed | **RED** |

   **The assertion COUNT is not a verdict in this binary and no row above rests
   on one.** `test_ltx2_video`'s whole-binary count is nondeterministic -- 4714,
   4715, 4716, 4717, 4719, 4721, 4769, 4770, 4771 and 4775 were measured on
   identical binaries over unchanged trees
   ([#1885](https://github.com/mudler/vllm.cpp/issues/1885)), and 4769, 4770 and
   4771 came from the polish pass over trees that differ only in comment text and
   one block's position. The rows read the CASE count and the pass or fail
   verdict. A `-tc`-filtered count is stable, which is why the M0 to M3 rows
   could carry one.

   **A stale binary produced a false verdict twice during this repair, in two
   different shapes, and both are worth naming because the tree was clean on disk
   each time.**

   - **A restore that preserved the mtime.** M4 to M8 restored the mutated file
     with `cp -a`. Ninja compared timestamps, skipped the rebuild, and left
     `libvllm.a` carrying the mutated translation unit, so the next case run
     reported failures against a clean tree. Every row above was re-measured with
     the restore followed by `touch`.
   - **A build that relinked a DIFFERENT target.** Building
     `test_ltx2_pipeline` and `test_diffusion_device_seam` recompiled
     `ltx2_video.cpp` and relinked `libvllm.a`, and did not relink
     `tests/test_ltx2_video`, which was still the executable a mutation had
     built. The whole-binary run then read 2 cases failed and 5 assertions
     failed on a repaired tree. Naming the target you are about to RUN in the
     build command is the fix, and reading the link line out of the build log is
     how you see that it happened.

   Each build log above was checked for the `Linking CXX executable
   tests/test_ltx2_video` line rather than assumed, and each mutation recorded
   its compile-error count, because a mutation that fails to build and one that
   never applied both read exactly like a passing test.

   M0 ran at `d0d4f1f60fc4765dd4118dead8bcc1778b1df9b1` and is the reason this
   row exists: with the audio renormalization disabled, the focused case passed
   every assertion it had. M1 and M2 each fail two assertions, one on the recipe
   row and one on the trace. M2 is the one that separates this block from the
   video block's form: the recipe row and the trace still AGREE with each other
   at `cfg_scale = 3.0`, and the literal comparison is what reds. M3 fails all
   four trace assertions at `0.0`, which is the reachability proof: the case
   enters the new code through `LoadVideoEngine` and `Generate` and not through a
   hand-built type.

   The green tree passes at 1 case and **75 of 75 assertions**, up from 67.

   One measurement did not complete and is recorded as such. The whole
   `test_ltx2_video` binary was started under M0 to ask whether ANY other case
   caught the mutation. It was stopped by `SIGTERM` at case 42 of 105 to free the
   box for the required gates, having reached **41 complete cases and 2632 of
   2632 assertions passed, with zero assertion failures**. It is a partial
   negative result, not a complete one. The complete statement rests on the
   focused M0 above and on a search: before this change no `audio_guidance_`
   identifier existed anywhere under `include/`, `src/` or `tests/`, so no
   assertion could read the render-resolved audio scales.

6. **The polish pass, and the two mutations re-run after it.** A second fresh
   review of the repair returned `PASS` with four LOW findings. Three are prose
   -- an audio-side `REQUIRE` that truncated video coverage, an imprecise
   citation, and a header sentence that claimed more than the replay holds --
   and the fourth is this section's own assertion figure. None of them moves a
   render.

   **The `REQUIRE` finding is the one with a measurement, and the reviewer took
   it.** The audio replay block sat in the MIDDLE of the `#1092` case, and
   doctest aborts the whole case at a failed `REQUIRE`, so an audio-side absence
   truncated every video assertion after it: the review reports that under
   mutation M6 the case ran **64 of its 91 assertions**, with the
   rebuilt-velocity, stepper-input and Euler blocks never reporting at all. That
   64 is the reviewer's number, carried here as their measurement; what this
   session measured is the 76 below, on the moved tree.

   The block is now the LAST thing in the case. The guard was NOT downgraded to
   a `CHECK`: it exists so the four `.data()` calls under it do not read an
   empty vector, and trading a clean abort for undefined behaviour is not a
   repair. The same shape at the override case's `REQUIRE` needed no
   move, because that block is already the last one in its case and truncates
   nothing.

   Re-measured at the polish head, on the build that carries it:

   | Id | Mutation | Diff | Compile errors | Relinked | Case | Exit | Cases | Assertions | Result |
   |---|---|---|---:|---|---|---:|---|---|---|
   | P1 | delete the five `audio_first_*` assignments in `RecordFirstGuidedStep` | 5 deletions | 0 | yes | #1092 | 1 | 1 failed | 76 run, 75 passed, 1 failed | **RED** |
   | P2 | `denoise_in.audio_guider` gets a copy of the resolved row with `cfg_scale = 1.0` and `rescale_scale = 0.0` | 4 insertions, 1 deletion | 0 | yes | #1092 | 1 | 1 failed | 91 run, 90 passed, 1 failed | **RED** |
   | P2 | the same mutation | 4 insertions, 1 deletion | 0 | yes | override | 1 | 1 failed | 32 run, 31 passed, 1 failed | **RED** |

   **P1's assertion count is the proof that the move worked, and it is a
   `-tc`-filtered count, so it is stable.** The green case runs 91 assertions and
   the audio block holds 16 of them. Aborting at the block's first assertion
   therefore admits `91 - 16 + 1 = 76`, which is what P1 reports: every one of
   the 75 video assertions ran and passed before the abort. The identity is what
   makes this a structural result rather than a bigger number -- 76 is not "more
   coverage", it is ALL of the video coverage, and no arithmetic short of that
   would have been the repair.

   P2 was deliberately NOT run in its stronger form. Handing the denoiser a
   default-constructed guider also zeroes `stg_scale`, which stops the audio
   perturbation and reds `video_audio_perturbed_blocks` as well -- a real
   failure, but a confounded one, because it is no longer a statement about the
   four scales the replay reads. Measured, and recorded here as the negative
   result it is: that form reds 3 assertions in the `#1092` case, of which one
   is the unrelated perturbation record. The constrained form above moves
   exactly the replay.

   Each restore was `cp` from a scratch copy followed by `touch` and a rebuild,
   never `cp -a`, and each was confirmed byte-for-byte: the mutated file returned
   to sha256 `3e4a33202151bb662a4399499ac4f604d4335f79203ad257d775a3710621bfdd`
   after both mutations, and the focused case returned to 91 of 91 passed.

   The polish head's own whole-binary run is **106 of 106 cases passed, exit 0**,
   on all three runs taken (assertion draws 4770, 4771 and 4769), the last of
   them on the tree this commit carries. Only `test_ltx2_video` was rebuilt: the
   polish pass changes two comments and the ORDER of one block inside one test
   case, so nothing it touches can move `test_ltx2_pipeline` or
   `test_diffusion_device_seam`. That
   is a reasoned exclusion, not a skipped gate: continuous integration runs the
   full suite on the pull request.

   **The other three findings, and what each cost.** The citation
   `ltx2_denoisers.cpp:57-58, :359-365` named the wrong mechanism twice: `:57-58`
   is the `PositiveOnlyGuider()` substitution, which makes an ABSENT modality
   guide with the identity and is not what leaves a vector empty, and `:359-365`
   starts INSIDE the `if (in.audio != nullptr)` branch rather than at its guard
   on `:357`. Both were bare line numbers, which `scripts/check-symbol-anchors.py`
   states outright that it cannot verify. It is now
   `ltx2_denoisers.cpp::Ltx2GuidedDenoise` in symbol form, which the gate does
   check and which reports `in-repo checked 355 (fresh 355, stale 0)`. The header
   sentence "what a wrongly handed `audio_guider` moves" overstated the replay by
   two fields of six; it now names which four `Ltx2MultiModalGuidance` reads,
   points `stg_blocks` at `video_audio_perturbed_blocks`, and names `skip_step`
   as held by nothing, which is
   [#1920](https://github.com/mudler/vllm.cpp/issues/1920) under `## Owed`. The
   fourth finding is the assertion figure, restated at the head of this section.

---

## 9. Evidence

| What | Where |
|---|---|
| Upstream anchors and their uniqueness | section 2, read at `fd4ded7f` |
| vLLM-Omni corroboration | section 3, read at `a4ea67a2` |
| Re-measured audio statistics | section 5 |
| Render commands and the confound | the two `render.log` files named in section 5.3 |
| Clean single-binary A/B on a leased GPU, run by the operator | `/mnt/nas_share/rc/ltx25-ab-1510/out/20260825T120744Z/`, section 5.4 |
| Mutation results | section 8, table M0 to M3, recorded in this row's pull request |

---

## 10. Stop conditions

Stop and return `NEEDS_DECISION` rather than continuing, in any of these cases:

1. **A step of the chain does not match upstream.** The verdict in section 4
   rests on every step matching. One that does not turns this row from a record
   into a bug fix with a different scope, a different test and a different
   reviewer question.
2. **Somebody asks for a render-behavior change.** A new default, a clamp, a
   normalization or a refusal breaks the mirror this row proves. It needs its
   own row, its own spec and an explicit developer decision, because it is a
   deliberate divergence from the reference and not a repair.
3. **A gate needs a graphics processing unit (GPU) lease, an `ssh` session or a
   large download.** This row has none of those authorities, and the oracle run
   that would close section 0 limit 1 belongs to
   [#1864](https://github.com/mudler/vllm.cpp/issues/1864).
4. **The audio-side render pin turns out to already exist.** Report the existing
   pin and add nothing. Section 6, decision D2 records the search that says it
   does not: the two cases that pin the audio guider params both read
   `Ltx2DetectPipelineParams` and neither goes through a render.
5. **A mutation fails to build, or the target does not relink.** Either one reads
   exactly like a passing test. Fix the mutation, do not record the green.

---

## Now

`DONE`. The verdict is recorded, the render-level audio pin is in, and the
uninstrumented verify script is filed. Nothing on this row is in flight.

## Owed

- [#1905](https://github.com/mudler/vllm.cpp/issues/1905). `verify_render.py`,
  the instrument that produced the FAIL verdict #1510 rests on, is not in this
  repository. It lives only on a CIFS share, is unversioned and untested, and no
  gate here reaches it: `grep -rn envelope_cv src include tests scripts` returns
  nothing and exits 1. Its `envelope_cv` rests on 20 frames of a 1.01 s clip and
  its `active_fraction` threshold is relative, so a constant-level signal reads
  `1.0` at any absolute level.

  The index row for #1905 states the unscoped `grep -rn 'envelope_cv'` instead,
  which now matches that row itself. It is **not corrected**, and deliberately:
  the index is append-only, so the only correction available is a second row
  keyed to one issue, which is a duplicate key and a second record that can
  disagree with the first. The load-bearing fact -- no gate in this tree reaches
  the instrument -- is unchanged, and this spec is where the reproducible command
  belongs.
- **The renormalization reduction axis is batch-1 only.** Lightricks reduces the
  standard deviation over the whole tensor and vLLM-Omni reduces it per batch
  item. `Ltx2MultiModalGuidance` reduces over the whole modality tensor, which
  matches both at batch 1 and matches only Lightricks above it. If this engine
  ever batches independent requests through that seam, the two definitions part
  company and this owes a decision. Not reachable today.
- **[#1920](https://github.com/mudler/vllm.cpp/issues/1920). The audio guider's
  `skip_step` is the one field of `denoise_in.audio_guider` nothing observes.**
  The replay this row added holds the four fields `Ltx2MultiModalGuidance`
  reads -- `cfg_scale`, `stg_scale`, `rescale_scale`, `modality_scale` -- and
  `stg_blocks` is held separately by `video_audio_perturbed_blocks`. That is
  five of six. `skip_step` is read only by `ltx2_pipeline.cpp::ShouldSkipStep`,
  which is `step % (skip_step + 1) != 0` and is FALSE at step 0 for every
  `skip_step`, and `Ltx2ConditioningTrace` records the first guided step only.
  So a guider mis-handed on that field alone moves nothing any gate here sees,
  and both shipped rows carry `skip_step = 0`, which makes a cross-wire between
  the arms invisible at the defaults for the same reason three of the four
  scales were before the override case existed. Closing it needs a trace field
  for the skip decision -- `Ltx2GuidedDenoiseResult` already carries
  `audio_skipped` and `RecordFirstGuidedStep` copies it nowhere -- and a case
  that reaches a step the guider CAN skip, through the existing
  `audio_skip_step` extra. NOT FIXED IN FLOW: it adds a field to a shared struct
  and needs its own red-before evidence. No gate that exists today is
  invalidated.
- **[#1864](https://github.com/mudler/vllm.cpp/issues/1864) owes the oracle
  run.** Until Lightricks/LTX-2 demonstrably runs the model here, section 4's
  verdict is a source comparison and cannot become a gate. That issue belongs to
  the oracle row, and this row cites it rather than claiming it.

## Outcome

**Measured: the port mirrors upstream, and the loudness is upstream's own trade.**
Section 2 walks the chain step by step and section 3 confirms it against a second
reference that outranks the first. The two hypotheses #1510 raises are both
refuted: the renormalization is present at `0.7` on both arms (section 2.2), and
the isolated-modality pass is not double counted, because the combination is the
one four-term sum at `guiders.py:261-266` applied once per denoiser call.

**Rejected: changing a default, adding a normalization, adding a clamp.** Each
would break the mirror the rest of the row proves. Upstream documents
`rescale_scale` as the anti-oversaturation knob and ships it at the top of its
own typical band while running the audio `cfg_scale` above its own typical band.
That is a deliberate upstream trade, not an accident to correct downstream.

**The MECHANISM in the issue does not survive re-measurement, and the FLATNESS
does.** #1510 reads saturation from a falling envelope coefficient of variation
beside a near full-scale peak, and says the loud parts stop getting louder while
the quiet parts rise. That causal story is refuted: two samples of 96,960 reach
the ceiling, the longest run at the ceiling is one sample, and the whole
amplitude distribution scales by about 1.45. Nothing is flat-topping.

The falling coefficient of variation is a separate fact and it stands.
`envelope_cv` is `env.std() / env.mean()` over 50 ms frames, which is
SCALE-INVARIANT, so the 1.45 gain cannot produce the fall and does not explain
it. The guided render's temporal envelope really is relatively flatter. What
changes is what that means: it is a property of the guided trajectory at
upstream's own defaults, not evidence of a ceiling, and the instrument reads it
over 20 frames without a confidence interval ([#1905](https://github.com/mudler/vllm.cpp/issues/1905)).

**The phenomenon REPRODUCES on a controlled A/B, and part of its reported size
did not.** The operator's leased single-binary run (section 5.4) holds
`--checkpoint-class` and the binary constant and still measures RMS up 3.60 dB
and the envelope coefficient of variation down 21.5 percent to `0.0872`, under
the instrument's 0.10 floor, with video passing on both arms. It also shows the
confounded pair overstated the effect: `0.0872` against the reported `0.0731`,
and a mono peak of `-0.693` dBFS against the reported `-0.287`. Zero samples
reach full scale on either arm. Reproduction plus the mirror in section 2 is what
makes the verdict a statement about upstream's defaults rather than about this
port.

**What the investigation actually found was an instrument gap, and that is the
part this row builds.** The video guider's resolved scales are pinned at the
render level and the audio guider's were not, so the audio arm of a guided
one-stage render could move without any gate noticing. Section 8's M0 measures
that, rather than asserting it.

**The first version of this instrument measured the RESOLVED scales and not the
CONSUMED ones, and a fresh review caught it.** The four `audio_guidance_*` fields
pin what the engine resolved; `denoise_in.audio_guider` is a second assignment,
and handing it CFG off with the renormalization disabled -- #1510's own subject
-- left five engine binaries green. The video arm was not green under the same
mutation, and the difference was its REPLAY rather than anything about the two
modalities. Five `audio_first_*` fields and the audio replay close it, and
decision D4 records why a golden WAV and a trace-fed replay were both rejected.

**Two further limits of that first version were measured rather than argued, and
both are closed.** The trace's "after `ApplyGuidanceOverrides`" guarantee had no
test: reading the PRE-override row left the whole binary green. And a full
cross-wire of the four fields to the video row red exactly ONE assertion, because
at the shipped defaults only `cfg_scale` differs between the arms. Neither is
fixable on a default render without moving a shipped default, which decision D1
forbids, so both are closed by one override case that sends four audio extras
differing from BOTH rows. The same two mutations now red four assertions each.

**Two record errors were found and are not re-filed here.** The unguided envelope
coefficient of variation in the #1510 table is a transposition of the AAC value
(section 5.1), and the A/B differs in binary and in `--checkpoint-class` beside
the flag under test (section 5.3). The operator comments on #1510 about both.
