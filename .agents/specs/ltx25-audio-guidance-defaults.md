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
   the render #1510 reports.
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
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` and compared with this tree at
`d0d4f1f60fc4765dd4118dead8bcc1778b1df9b1`. Each upstream anchor was asserted
for uniqueness with `git grep`, not only for existence.

| Step | Lightricks/LTX-2 at `fd4ded7f` | This tree at `d0d4f1f6` |
|---|---|---|
| The four-term guidance sum | `packages/ltx-core/src/ltx_core/components/guiders.py:261-266` | `ltx2_pipeline.cpp:547-560`, `Ltx2MultiModalGuidance` |
| The standard-deviation renormalization | `guiders.py:268-271` | `ltx2_pipeline.cpp:562-579`, `Ltx2MultiModalGuidance` |
| The audio guider's own defaults | `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:59-68` | `ltx2_pipeline.cpp:956-961`, `Ltx2Params20` |
| Combination in x0 space, not velocity space | `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:480-482` and `packages/ltx-core/src/ltx_core/model/transformer/model.py:590-604`, the guider call at `utils/denoisers.py:202-203` | `ltx2_video.cpp:4349-4350`, `ToDenoised`, and `ltx2_denoisers.cpp:362-365` |
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

### 3.1 One real difference between the two upstreams, which is not a divergence here

vLLM-Omni reduces the renormalization standard deviation **per batch item**
(`reduce_dims = tuple(range(1, pred.ndim))`, `ltx2_guidance.py:166-168`), where
Lightricks takes a whole-tensor `cond.std()` (`guiders.py:269`). Omni's own
comment at `:162-165` says the official one-stage entry runs a single generated
sample, so the two definitions coincide at batch 1. `Ltx2MultiModalGuidance`
takes `count` over the whole modality tensor, and every render on this engine is
batch 1, so it matches both. If this engine ever batches independent requests
through that seam, the two part company. That is recorded under
[`## Owed`](#owed) rather than changed now.

### 3.2 Why the 2.5 row is sourced at Lightricks

vLLM-Omni's `_PIPELINE_RECIPES` stop at `("one_stage", "2.3")`
(`ltx2_recipes.py:161-166`), so 2.5 exists only at Lightricks. Nothing rests on
that gap for this row: `constants.py:83-88` changes only the step count and the
perturbed block between 2.3 and 2.5, and leaves all four guidance scales alone.

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

**The flatter envelope is part of the verdict, not an exception to it.** Section
5.2 shows the guided envelope really is flatter in a scale-invariant sense, and
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
coefficient of variation at `0.07309848`. Three findings sit on top of them, and
the operator comments on #1510 about the first and the third.

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

### 5.4 The verifier that produced the FAIL verdict is not in this repository

`verify_render.py` has one copy, on a Common Internet File System (CIFS) share:
`/mnt/nas_share/rc/ltx25-fullmodel/job/verify_render.py`, sha256
`57cf92846506be961e3c6ab3c9198de5608d0e85bfd1eb992eb91dda1c0fa563`. It is
unversioned, untested and unreachable from any gate in this
tree. `grep -rn 'envelope_cv'` over this repository returns nothing.

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
path at `src/vllm/multimodal/ltx2_video.cpp:4065-4068`. There is no audio
counterpart. The existing case "ltx2 one_stage: all four guidance arms are
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

One case changes: "ltx2 one_stage: all four guidance arms are combined in X0
space (#1092)" in `tests/vllm/multimodal/test_ltx2_video.cpp`. It gains an
assertion block that:

1. pins `recipe.phases[0].audio_guidance`'s four scales against the upstream
   literals `cfg 7.0`, `stg 1.0`, `rescale 0.7`, `modality 3.0`
   (`constants.py:59-68`);
2. pins the four new `Ltx2ConditioningTrace` audio fields, read off a completed
   render, against the same literals.

No new case, and no new executable. The case already loads the engine through
`vllm::multimodal::LoadVideoEngine` and renders through `engine->Generate`, so
the new assertions enter through the production entry point rather than
constructing the type by hand.

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

   | Id | Tree | Mutation | Expected |
   |---|---|---|---|
   | M0 | before this row | `ltx2_pipeline.cpp:958`, audio `rescale_scale` `0.7` to `0.0` | GREEN, the arm is invisible |
   | M1 | after this row | `ltx2_pipeline.cpp:958`, audio `rescale_scale` `0.7` to `0.0` | RED |
   | M2 | after this row | `ltx2_pipeline.cpp:956`, audio `cfg_scale` `7.0` to `3.0` | RED |
   | M3 | after this row | delete the production assignment of the four audio trace fields in `src/vllm/multimodal/ltx2_video.cpp` | RED, reachability |

   All four are **measured**. Every one built with zero compile errors, relinked
   `tests/test_ltx2_video`, and was restored so that `git diff` reported the
   intended change and nothing else.

   | Id | Diff | Compile errors | Relinked | Exit | Cases | Assertions | Result |
   |---|---|---:|---|---:|---|---|---|
   | M0 | 1 insertion, 1 deletion | 0 | yes | 0 | 1 passed | 67 of 67 passed | **GREEN**, as predicted |
   | M1 | 1 insertion, 1 deletion | 0 | yes | 1 | 1 failed | 75, 2 failed | **RED** |
   | M2 | 1 insertion, 1 deletion | 0 | yes | 1 | 1 failed | 75, 2 failed | **RED** |
   | M3 | 4 deletions | 0 | yes | 1 | 1 failed | 75, 4 failed | **RED** |

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

---

## 9. Evidence

| What | Where |
|---|---|
| Upstream anchors and their uniqueness | section 2, read at `fd4ded7f` |
| vLLM-Omni corroboration | section 3, read at `a4ea67a2` |
| Re-measured audio statistics | section 5 |
| Render commands and the confound | the two `render.log` files named in section 5.3 |
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
  gate here reaches it. Its `envelope_cv` rests on 20 frames of a 1.01 s clip and
  its `active_fraction` threshold is relative, so a constant-level signal reads
  `1.0` at any absolute level.
- **The renormalization reduction axis is batch-1 only.** Lightricks reduces the
  standard deviation over the whole tensor and vLLM-Omni reduces it per batch
  item. `Ltx2MultiModalGuidance` reduces over the whole modality tensor, which
  matches both at batch 1 and matches only Lightricks above it. If this engine
  ever batches independent requests through that seam, the two definitions part
  company and this owes a decision. Not reachable today.
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

**What the investigation actually found was an instrument gap, and that is the
part this row builds.** The video guider's resolved scales are pinned at the
render level and the audio guider's were not, so the audio arm of a guided
one-stage render could move without any gate noticing. Section 8's M0 measures
that, rather than asserting it.

**Two record errors were found and are not re-filed here.** The unguided envelope
coefficient of variation in the #1510 table is a transposition of the AAC value
(section 5.1), and the A/B differs in binary and in `--checkpoint-class` beside
the flag under test (section 5.3). The operator comments on #1510 about both.
