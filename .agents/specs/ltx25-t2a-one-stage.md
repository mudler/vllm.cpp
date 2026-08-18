# LTX-2.5 — text-to-audio (`T2AOneStagePipeline`)

Row: `LTX25-T2A-ONE-STAGE`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned;
**not edited by this row**). Issue:
[#1005](https://github.com/mudler/vllm.cpp/issues/1005). Parent campaign issues:
[#644](https://github.com/mudler/vllm.cpp/issues/644),
[#435](https://github.com/mudler/vllm.cpp/issues/435).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Read from a local checkout at that revision. `git rev-parse HEAD` was run on a
clean tree (`git status --short` empty) before any anchor below was taken.

---

## 0. Honesty statement — what this row does and does not claim

This row makes LTX-2.5 render **audio with no video**, from a prompt, through
the shared video seam. It is the first path in this tree that returns a
`VideoResult` carrying zero frames.

**Upstream ships no tests at this pin.** `find /home/mudler/_git/LTX-2 -name
'test_*.py'` returns **0**; the broader `find . -iname '*test*' -not -path
'./.git/*'` that row `LTX25-A2V-AUDIO-INPUT` ran returns empty as well. So "port
the upstream tests in the same change" has nothing to port, and the obligation
becomes what §5 does instead: pin upstream's *behaviours* — its constants, its
predicates, its refusals — each against a `file:line` anchor, and tie at least
one assertion to a **local** fact so the suite can see a claim about this tree go
stale. This is stated rather than quietly skipped.

**No render on real weights is claimed.** The row is gated on the reduced
fixture. The GPU is out of bounds for this row (a render ladder holds
`dgx.casa`, and the box OOM-reboots when its 119 GiB unified pool is exhausted),
so the real-checkpoint T2A render is owed and named under `## Owed`.

**The device arm is refused by name, not served.** See §7.

---

## 1. Does the audio-only shape fit `Ltx2VideoEngine`? YES — and here is the
## derivation, because the opposite answer was the expected one

The dispatch that opened this row expected a possible `NEEDS_DECISION` on the
entry point. It is not needed, and the reason is upstream's own shape rather
than a convenience here.

**Upstream expresses T2A's duration through a VIDEO pixel shape.**
`t2a_one_stage.py:37-40` declares `_AUDIO_ONLY_PLACEHOLDER_RES = 512` and says in
as many words that "Audio-only generation reads ``frames`` and ``fps`` from the
pixel shape via ``AudioLatentShape.from_video_pixel_shape`` (height/width are
unused)". `__call__` then passes `width=512, height=512, frames=num_frames,
fps=frame_rate` into the SAME `DiffusionStage.__call__` every video pipeline
calls (`:163-167`, against `utils/blocks.py:501-512`). So the request shape T2A
needs is exactly the request shape `VideoGenParams` already carries:
`num_frames`, `frame_rate` (through `duration_seconds` or the recipe), `steps`,
`seed`, `prompt`, `output_dir`.

**And `VideoResult` can already say "no picture".** It carries `frame_dir`,
`audio_path`, `frame_count`, `sample_rate` and `mux_argv` as independent fields
(`include/vllm/multimodal/video_engine.h`, `struct VideoResult`). An audio-only
result is `frame_count = 0`, an empty `frame_dir`, an `audio_path`, and an EMPTY
`mux_argv` — there is nothing to mux, and composing an ffmpeg argv over a frame
pattern that matches no file would hand the caller a command that fails.

So the seam represents the behaviour, and AGENTS.md §"Shared seams" then binds
the other way round: a distinct entry point would be the parallel path that
section forbids. What this row does NOT do is bend the video path: the audio-only
render lives in its own translation unit (§3), mirroring upstream's own file
structure, and `Generate` branches to it before it resolves any video geometry.

---

## 2. The gap, derived at `332aed738`

Three blockers, each named by the symbol that has to change. None of them is
"the pipeline file is missing".

### 2.1 `Ltx2DitForward` refuses a one-stream call, and its stated reason does not describe T2A

`src/vllm/model_executor/models/ltx2_dit.cpp:765 @ 332aed738` is

```
VT_CHECK(video != nullptr && audio != nullptr, "... LTXModelType.VideoOnly and
         LTXModelType.AudioOnly carry a different weight contract and are not ported");
```

Re-derived at this tree rather than inherited. The reason is a claim about the
**checkpoint**, and it is true of a checkpoint that was SAVED as an `AudioOnly`
model. T2A does not load one. `t2a_one_stage.py:81-93` calls
`DiffusionStage.from_checkpoint(model_paths.transformer(), ...,
model_configurator=LTXAudioOnlyModelConfigurator,
model_sd_ops=LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP)`: the FILE is the ordinary
AudioVideo transformer, and the SDOps map (`model_configurator.py:228-239`)
merely restricts which of its keys are read — `audio_attn1`, `audio_attn2`,
`audio_ff`, `audio_patchify`, `audio_proj_out`, `audio_adaln_single`,
`audio_prompt`, `audio_scale_shift_table`. The comment at `:78-82` of the
pipeline says why: so "the video weights are never even read from disk". The
weight contract `EnumerateLtx2DitTensors` describes is therefore the one T2A's
file satisfies, and the refusal's cause is not this row's blocker.

**What is left of that refusal after the lift is stated rather than dropped.** A
checkpoint containing ONLY the audio subset still cannot be loaded here, because
`Ltx2LoadDitFromSafetensors` enumerates the AV contract. That half stays, and it
moves to where it is true — the loader — instead of guarding a forward that
already handles the case.

**The forward already handles it.** Every line of `Ltx2DitForward` below the
check is written against `video != nullptr` / `have_both`
(`ltx2_dit.cpp:786-869 @ 332aed738`), and `Ltx2TransformerBlockForward` computes
`run_vx` / `run_a2v` / `run_v2a` from `video_x != nullptr && tv > 0`
(`ltx2_dit.cpp:248-251 @ 332aed738`). Lifting the check reaches a path that was already
written; it does not add one.

### 2.2 `enabled = false` is NOT the same shape, and the difference renders

The refusal's own message says to "use `enabled` to run one stream of an AV
model", and `include/vllm/model_executor/models/ltx2.h:506-508 @ 332aed738` repeats it. That
advice is correct for the A2V and joint pipelines. **It is wrong for T2A**, and
this is the detail that fails silently if guessed.

Upstream's predicate is

```
run_v2a = run_ax and (video is not None and vx.numel() > 0)     # transformer.py:269
```

It tests `video is not None`, **not** `video.enabled`. So an AV forward handed a
present-but-disabled video stream still runs video->audio cross attention, taking
its context from a video latent that T2A never intended to exist — and returns a
finished waveform. Our port mirrors that polarity exactly at
`ltx2_dit.cpp:251 @ 332aed738`, so the same trap is live here.

`video = nullptr` is the only shape that reproduces `LTXModel.forward(video=None,
...)` (`model.py:492-538`, the `video_args = ... if video is not None else None`
at `:505`). It is also what makes `_init_preprocessors`' AudioOnly arm
(`model.py:351-365`, a plain `TransformerArgsPreprocessor` with no
`cross_scale_shift_adaln` and no `cross_gate_adaln`) equivalent to our
`have_both == false` path, which passes `other = nullptr` into `PrepareStream`
and therefore builds neither.

### 2.3 The engine has NO guided denoiser, and T2A's defaults turn one on

This is the finding the dispatch did not anticipate and it is the largest part of
the row.

`Ltx2VideoEngine::Generate` runs exactly ONE DiT forward per step
(`ltx2_video.cpp`, the `Ltx2DitForward` / `Ltx2DitForwardDevice` call in the
denoise loop) and never reads a guider parameter: `git grep -n
'guid\|Guider\|cfg_scale' src/vllm/multimodal/ltx2_video.cpp` returns zero hits
against a positive control of 66 for `ltx2` in the same file. That is correct for
`distilled_two_stage`, which builds a `SimpleDenoiser` upstream too. It is not
correct for T2A.

T2A builds a `FactoryGuidedDenoiser` (`t2a_one_stage.py:154-161`) over
`MultiModalGuiderParams` whose CLI defaults are the params table's
`audio_guider_params` — `cfg_scale=7.0`, `stg_scale=1.0`, `rescale_scale=0.7`,
`stg_blocks=[28]` on the 2.3/2.4/2.5 lineage (`utils/constants.py:58-66`,
`:82-87`, `:118`) — with `modality_scale` pinned to **1.0** by the CLI itself
(`t2a_one_stage.py:200-202`: "Audio-only generation has no video modality, so the
video->audio (v2a) cross-modal guidance is meaningless here. 1.0 disables it").

Read against `guiders.py:275-287`, those defaults mean
`do_unconditional_generation()` is TRUE (`cfg_scale != 1.0`),
`do_perturbed_generation()` is TRUE (`stg_scale != 0.0`) and
`do_isolated_modality_generation()` is FALSE. **Three forwards per step: cond,
uncond-text, uncond-perturbed.** Shipping T2A on the engine's single-forward path
would be a different trajectory from upstream's default, and no shape, token
count or file length could see it.

The bricks exist and are gated; the WIRING does not. `Ltx2MultiModalGuidance`
(`ltx2_pipeline.h:323 @ 332aed738`), `Ltx2CfgDelta`, `Ltx2StgDelta`,
`Ltx2GuiderParamsForSigma`, `Ltx2PerturbationConfig` and
`Ltx2BatchedPerturbationConfig` are all ported with goldens and have **no product
caller** — the "test-only driver" shape `.agents/reachability.md` enumerates. What
is genuinely absent is one thing: the DiT forward runs upstream's
`perturbations=None` path only, and says so at
`include/vllm/model_executor/models/ltx2.h:42-46 @ 332aed738`.

---

## 3. Design

### 3.1 A new translation unit, mirroring upstream's file

`include/vllm/model_executor/models/ltx2_t2a.h` +
`src/vllm/model_executor/models/ltx2_t2a.cpp`, holding `Ltx2T2aGenerate` — the
port of `T2AOneStagePipeline.__call__` (`t2a_one_stage.py:109-172`).
`Ltx2VideoEngine::Generate` branches to it at the top, before it resolves any
video geometry. This mirrors upstream's file structure, which AGENTS.md §"Shared
seams" requires, and it follows the precedent
`.agents/specs/ltx25-a2v-audio-input.md` §2 set ("A new translation unit per
concern, mirroring upstream's file structure rather than growing
`ltx2_video.cpp`").

It is NOT a parallel path: it is reached only through
`VideoEngine::Generate`, it consumes the same `Impl`-owned weights, and every
numeric it uses is an already-gated brick — `Ltx2SigmaSchedule`,
`Ltx2AudioPatchify` / `Ltx2AudioUnpatchify` / `Ltx2AudioPatchTimings`,
`Ltx2DitForward`, `Ltx2MultiModalGuidance`, `Ltx2EulerStep`,
`Ltx2AudioDecoderForward`, `Ltx2VocoderWithBweForward`, `MiniMaxH3WriteWav`.

The duplication it does accept is the audio-only denoise loop. That is deliberate:
the joint loop in `ltx2_video.cpp` is ~200 lines of video-stream construction the
audio-only path has no counterpart for, and threading an `is_t2a` flag through it
would put nine new branches inside a function that already runs 1935 lines.

### 3.2 Call order, mirroring `t2a_one_stage.py:123-172`

1. `require_num_frames_source` (`:123`, `utils/blocks.py:894-905`). Auto duration
   with no `DurationHead` is upstream's own refusal. Mirrored: this engine
   constructs no duration head at all (`duration_head_path` is refused at load by
   `CheckUnservedExtras`, #611), so an absent `num_frames` is refused with the
   message naming the head.
2. Encode the prompt AND the negative prompt (`:127-135`). Upstream encodes both
   in one `PromptEncoder` call and takes `.audio_encoding` from each
   (`:134-135`). Here both go through `Ltx2EncodePromptToConditioning` and the
   connector, and only the AUDIO half of each is kept — the video half is
   computed and discarded, exactly as upstream computes both encodings and uses
   only the audio one.
3. `resolve_num_frames` (`:137-139`) — an explicit count is returned verbatim
   (`utils/blocks.py:920-921`).
4. `sigmas = self._scheduler.execute(steps=num_inference_steps)` (`:141-143`).
   `LTX2Scheduler()` is HARD-CODED at `:67`, so this is `Ltx2SigmaSchedule` and
   the recipe carries no distilled sigma table. The token count the shift is
   derived from is the AUDIO latent's own (`schedulers.py:32` reads
   `math.prod(latent.shape[2:])` of the unpatchified target).
5. `create_multimodal_guider_factory(params, negative_context)` (`:149-152`).
   A plain `MultiModalGuiderParams` becomes a sigma-independent guider; the
   sigma-binned factory arm is not reachable from any surface here and is owed.
6. The stage call with `video=None` and `audio=ModalitySpec(context=a_context_p)`
   (`:154-170`). No initial latent, no freeze: the audio starts as pure noise.
7. `return self.audio_decoder(audio_state.latent)` (`:172`) — the audio VAE
   decode and the BWE vocoder, which is what the ordinary render already does.

### 3.3 The guided step

Per step, mirroring `FactoryGuidedDenoiser` over `MultiModalGuider.calculate`
(`guiders.py:244-273`):

| Pass | Run when | Inputs |
|---|---|---|
| `cond` | always | positive audio context, no perturbation |
| `uncond_text` | `DoUnconditionalGeneration()` | NEGATIVE audio context, no perturbation |
| `uncond_perturbed` | `DoPerturbedGeneration()` | positive context, audio self-attn perturbed on `stg_blocks` |
| `uncond_modality` | `DoIsolatedModalityGeneration()` | — **refused by name**, see §7 |

`ShouldSkipStep(step)` (`guiders.py:287-291`) runs NO forward at all and reuses
the previous step's denoised prediction (`utils/denoisers.py:85-91`), which is
not the same as "skip the guidance and keep the conditional pass" — see M9 in §5.
`skip_step` defaults to 0, so it never fires on the default path and is reachable
only from the extra.

The combination is `Ltx2MultiModalGuidance`, which already carries the two
details a re-derivation gets wrong: the `(scale - 1)` polarity on CFG and modality
against a bare `scale` on STG, and torch's UNBIASED `std` in the rescale.

### 3.4 STG — the one genuinely new numeric

`SKIP_AUDIO_SELF_ATTN` on a perturbed block replaces the attention output with the
raw value projection BEFORE `to_out`: `attention.py:558-577` computes `v =
self.to_v(context)` and, when `all_perturbed`, sets `out = v` without ever
projecting `q` or `k`; the blended form `out * mask + v * (1 - mask)` at `:571-572`
is the partial-batch case and reduces to the same thing at batch 1.

Mirrored as an `all_perturbed` flag on `Ltx2AttentionArgs`, set per block from a
`Ltx2BatchedPerturbationConfig` handed to `Ltx2DitForward` — which is upstream's
own `perturbations` parameter on `LTXModel.forward` (`model.py:492`), so this is
mirroring a signature rather than inventing a seam. `nullptr` is upstream's
`perturbations=None` and every existing caller keeps its current behaviour
byte-for-byte.

### 3.5 Where the request enters

`pipeline_kind = t2a_one_stage` is a LOAD extra
(`kLtx2PipelineKindExtra`, already defined and already reachable from
`ltx2-gen --pipeline-kind` and from `vllm_video_model_params::extras`). This
matters for the reach claim: #928 records that `/v1/videos` forwards no
PER-GENERATION extra, and this row's selector is not one. The reach claim in §6b
is worded against what was checked rather than against the file the knob lives in.

Two per-generation extras are added, both upstream CLI arguments:

- `negative_prompt` (`--negative-prompt`, `utils/args.py:1083-1088`, defaulting
  to `DEFAULT_NEGATIVE_PROMPT`, `utils/constants.py:186`). Absent means the
  recipe's own default, which is `LightricksNegativePrompt()` — already in the
  tree and already what `one_stage` on 2.4/2.5 resolves.
- `audio_cfg_guidance_scale` / `audio_stg_guidance_scale` /
  `audio_rescale_scale` / `audio_skip_step` / `audio_stg_blocks`
  (`utils/args.py:1089-1119`). Absent means the params table's own value.

`modality_scale` gets NO extra, and that is not an omission: the CLI pins it to
1.0 at `t2a_one_stage.py:200-202` and there is no upstream surface that varies it
on this pipeline.

### 3.6 The recipe

`t2a_one_stage` rows for `2`, `2.3`, `2.4` and `2.5`, mirroring the `one_stage`
rows one for one, because the schedule is the same object: `LTX2Scheduler()` at
`t2a_one_stage.py:67` against `ti2vid_one_stage.py:81`, and the same
`detect_params` step count. What differs is what the recipe DECLARES about video:
`video_output_phase = -1` and a new `audio_only` flag, so a T2A recipe cannot be
run down the video path by accident.

The 2.0 and 2.3 rows take `kOmniNegativePrompt` and the 2.4/2.5 rows take
`LightricksNegativePrompt()`, which is what `ResolveLtx2PipelineRecipe`'s
`one_stage` arm already does (`ltx2_pipeline.cpp:1239-1247 @ 332aed738`) — the negative prompt
travels with the generation, not with the pipeline.

### 3.7 Load

`video_vae_path` becomes optional **only** on a `t2a_one_stage` engine.
Upstream's `T2AOneStagePipeline.__init__` never calls `model_paths.video_vae()`
(`:53-107` constructs a `PromptEncoder`, a `DiffusionStage`, an `AudioDecoder`
and a `DurationPredictor`, and nothing else), so requiring one here would demand
a checkpoint the pipeline cannot use. `audio_vae_path` stays required, because
`AudioDecoder` is constructed unconditionally at `:94-100`.

---

## 4. Risks

**The `READER ANCHORS` gate.** `ltx2_video.cpp` carries a derived line-number
list re-derived and string-compared by `test_ltx2_video.cpp`. Any line inserted
above the last anchored line shifts it, and a clean `git merge` will not warn.
Mitigation: the list is re-derived at the final tree after the last merge of
`origin/main` with the test's own walk rather than by arithmetic, and it is named
as a merge hazard in the PR body.

**AND THE ANCHORS DID MOVE.** The mitigation above once claimed the engine edit
is "a branch at the TOP of `Generate`, which is below every anchor", and the
list is `781 791 792 854 950 966 968 1046 1071 1176 1217` on `origin/main`
against `782 792 793 855 951 967 969 1060 1085 1190 1231` here. Two hunks sit
ABOVE the last anchor: the `ltx2_t2a.h` include at `@@ -36,6 +36,7`, which shifts
every anchor by one, and the audio-only video-VAE exception at
`@@ -974,8 +975,21`, which adds thirteen more and moves the last four by
fourteen. `@@ -1018,7 +1032,7` is above 1231 as well and is net zero. The
anchors were correctly RE-DERIVED with the test's own walk and the gate passes
23/23, so the outcome is right; only the stated reason was false, and a false
reason is what makes the next reader skip the re-derivation.

**A silently wrong render.** Every failure mode here produces a playable WAV of
the right length: a guider that never runs its uncond pass, an STG pass that
perturbs the wrong block, a `video = nullptr` path that silently reintroduces v2a
cross attention, an audio latent that decoded from zeros. None is visible in the
output. Mitigation: §5's trace fields observe the EFFECT rather than the
bookkeeping, plus a lower bound so a zeroed tensor cannot pass.

**A recorded value is not a reached one.** A sibling row's suite recorded a
latent digest and inferred the latent reached the phase; a build that started from
zeros passed everything. So every assertion below is written against the state the
loop actually ran over, read after the operation it claims, and at least one is a
CONTROL that must move in the opposite direction.

**Concurrent edits.** `ltx2_video.cpp`, `ltx2.h`, `ltx2_pipeline.{h,cpp}`,
`docs/FEATURES.md` and `.agents/issue-index.md` are being edited by sibling rows
(`LTX25-RETAKE`, `LTX25-DFR-PIPELINE` both landed within the hour before this
row started). Mitigation: new behaviour lives in new TUs; keyed records are
reapplied by key with unrelated keys proven byte-identical; the index is appended
to and never edited.

---

## 5. Tests and evidence

Focused gate: `test_ltx2_video`, `test_ltx2_pipeline`, `test_ltx2`, `test_capi`.

1. **RED first, through the production entry point.** The smallest failing test
   loads the reduced fixture with `pipeline_kind = t2a_one_stage` and calls
   `Ltx2VideoEngine::Generate` (reached from `vllm_video_generate`). It fails
   first because `ResolveLtx2PipelineRecipe` refuses the kind by name.
   Reachability is then proven by deleting the production call site and showing
   the test goes RED (`.agents/reachability.md`).
2. **The render has NO picture and DOES have sound.** `frame_count == 0`, an
   empty `mux_argv`, an `audio.wav` on disk whose sample rate is the vocoder's
   own, and NO `frame_000000.ppm` in the output directory. The last one is the
   half a field check cannot make: a build that wrote frames and reported zero
   passes every other assertion here.
3. **The video stream never reaches the DiT.** A trace field records whether the
   forward was handed a video stream at all, and it must be false on every step
   of a T2A render and true on an ordinary one. This is the §2.2 trap, and a
   count- or shape-shaped check cannot see it: an `enabled=false` video stream
   produces a waveform of exactly the right length.
4. **The guider actually ran, and each arm is separable.** Counters for the
   cond / uncond-text / uncond-perturbed forwards. At the default params the
   counts must be `steps`, `steps`, `steps`; with `audio_cfg_guidance_scale=1`
   the second must be 0 and the render must still complete; with
   `audio_stg_guidance_scale=0` the third must be 0. A single "guidance ran"
   boolean would be satisfied by a build that ran the uncond pass and then
   ignored it, so the guided output is ALSO required to differ from the
   unguided one on the same seed.
5. **STG perturbs the block it was told to.** Asking for `audio_stg_blocks` that
   name a block index out of range is upstream's own failure and is refused; and
   perturbing block N produces a different velocity from perturbing block M. A
   test asserting only that "a perturbed pass ran" cannot tell a config that
   perturbs everything from one that perturbs nothing.
6. **A LOCAL fact is tied to the lifted refusal**, per this campaign's standing
   rule. A test asserting only upstream symbol names cannot see staleness: a
   sibling proved this by replacing a refusal's local claim with a self-declared
   falsehood and watching the suite stay green at 44/44. So the test re-derives,
   from THIS tree, that `Ltx2DitForward` still guards its remaining half and that
   the AudioVideo weight contract is still what the loader enumerates.
7. **A lower bound** on the decoded waveform's magnitude, so a silently zeroed or
   constant buffer fails. A correlation or count-based check cannot see a scale
   error and is not used alone.

Every mutation records three facts: `git diff --stat` after applying, whether it
**BUILT** with the compile-error count beside it, and the **exit code** captured
directly rather than through a pipe. A mutation that fails to compile establishes
nothing and is recorded as such. `-tc` filters are comma-free and the case count
is asserted non-zero, because doctest splits on a comma and runs UNRELATED cases
to a green SUCCESS.

### What the mutation pass actually found

Focused gate `./build/tests/test_ltx2_video "--test-case=*t2a*"`. Each mutation
applied to ONE file, rebuilt, run, restored in a `finally` and the restore
verified by **sha256** rather than assumed. The harness rebuilds the restored
tree before anything else measures it.

**Measured at 8 cases / 505 assertions / exit 0 unmutated**, on the tree this
pull request ships. The table was re-run at that tree rather than carried forward
from the earlier 6-case measurement, because a count carried forward while the
suite grows is exactly how this campaign's previous mutation records went stale.

| Mutation | `git diff --stat` | BUILT | exit | verdict |
|---|---|---|---|---|
| M1 delete the production call site (the reachability mutation) | `ltx2_video.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED, 4 of 8 cases red |
| M2 hand the forward a present-but-DISABLED video stream instead of `nullptr` | `ltx2_t2a.cpp` (see note) | YES (0 errors) | 1 | DETECTED, 3 red |
| M3 never run the unconditional forward | `ltx2_t2a.cpp` (see note) | YES (0 errors) | 1 | DETECTED, 2 red |
| M4 ignore `stg_blocks` and perturb EVERY block | `ltx2_t2a.cpp` (see note) | YES (0 errors) | 1 | DETECTED, 3 red |
| M5 `all_perturbed` falls through to ordinary attention | `ltx2.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED, 2 red |
| M6 revert the `one_stage` `noise_scale` to the struct default (#1013) | `ltx2_pipeline.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED, 1 red |
| M7 scale the initial latent by `sigmas[0]` | `ltx2_t2a.cpp` (see note) | YES (0 errors) | 0 | **SURVIVED** — see below |
| M8 write a frame on the audio-only path | `ltx2_video.cpp \| 1 +` | YES (0 errors) | 1 | DETECTED, 1 red |
| M9 a skipped step RECOMPUTES the conditional forward instead of reusing | `ltx2_t2a.cpp \| 39 +++---` | YES (0 errors) | 1 | DETECTED, 1 red |

**A note on the first fact for the `ltx2_t2a.cpp` rows, because it reported
something misleading and that is worth writing down rather than tidying away.**
`git diff --stat` is measured against `HEAD`, not against the pre-mutation
working tree, so on a run where that file also carried an uncommitted change the
stat reports 45-47 lines rather than the mutation's own 1-3. The number is
therefore not a measurement of the mutation on those rows. It is kept, with this
note, instead of being replaced by a prettier one: the fact the protocol asks for
is what the command printed. The M1, M5, M6, M8 and M9 rows were measured against
a clean file and their stats are the mutations'.

**M9 is the mutation for a defect this port ACTUALLY SHIPPED in its first
draft**, rather than an invented one. `should_skip_step` (`guiders.py:287-291`)
does not mean "skip the guidance and keep the conditional prediction". Upstream
returns `DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)`
(`utils/denoisers.py:85-91`) BEFORE it assembles any pass, so a skipped step runs
**no DiT forward at all** and reuses the previous step's denoised prediction. The
first draft ran the conditional forward and used it: a whole extra forward per
skipped step, on a different trajectory, producing a finished waveform of exactly
the right length. Nothing about the output separates the two, and the FORWARD
COUNT is the only thing that does.

**M7 SURVIVED, and the resolution is the useful part.** It is the mutation a
reader coming from another flow-matching sampler expects to be REQUIRED — scaling
the initial noise by the first sigma — and it changed nothing. The reason is not
a blind gate: it is an identity, and it STAYS survived after the pin below,
because a pin on an identity cannot turn one arm red. `LTX2Scheduler` starts at `linspace(1, 0, steps
+ 1)[0] == 1`; the shift map sends 1 to `exp(s)/(exp(s) + (1/1 - 1))`, exactly 1
(`schedulers.py:41-45`); the stretch sends it to `1 - (1 - 1)/scale_factor`,
again exactly 1 (`:47-55`). So `sigmas[0]` is 1.0 for every step count.

Rather than record "a mutation survived", the identity is now GATED — the case
"the schedule starts at exactly 1.0" pins `sigmas.front() == 1.0F` across four
step counts. If upstream ever moves the first sigma off 1, that fires and the two
forms stop agreeing.

**And that case found a second thing.** `steps = 1` returns `-nan`, on both
sides: the non-zero sigma list is `[1.0]`, so `one_minus_z` is `[0.0]`,
`scale_factor = 0 / (1 - terminal)` is 0, and the stretch computes `1 - 0/0`
(`schedulers.py:49-54`). It is upstream's own arithmetic, not a defect here, and
it is excluded from the pin with the reason written beside it rather than
silently skipped. A one-step schedule is recorded under `## Owed`.

**Two harness notes, because both would otherwise read as verdicts about the
code.** A `.pyc` for `scripts/agent-start.py` was truncated to exactly 4096 bytes
on this shared box, and `scripts/agent-preflight.sh` reported
`FAIL test_agent_start` with an `EOFError: marshal data too short` — a corrupt
byte-cache presenting as a failing gate. Removing the file made it pass 20/20.
And M4's first form asserted the STG perturbation on a latent filled with a
constant: self-attention over identical rows returns a weighted average of
identical values, which IS the value projection, so the perturbation was a
numeric no-op and the case reported "the perturbation changed nothing" about a
correct build. The fixture latent now varies per element, and the reason is in
the test.

---

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported with `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the full pass/fail line, positive controls for `No space left` and
`BFD assertion`, the box load and free disk. Known-red on `main` proven
pre-existing rather than asserted: `windows-msvc-*` (#584), `check-env-doc`
(#995). A load-dependent failure is re-run alone with the load recorded before it
is charged to this row.

## 6b. Reachability — the sentence the records must carry

**A production entry point reaches this, and the test enters through it.**

```
include/vllm.h  vllm_video_generate
  -> src/capi/vllm_c.cpp                        engine->Generate(gen)
    -> vllm::multimodal::VideoEngine::Generate
      -> Ltx2VideoEngine::Generate              (the t2a branch at the top)
        -> Ltx2T2aGenerate                      (ltx2_t2a.cpp)
          -> Ltx2DitForward(..., /*video=*/nullptr, &ain, ...)
          -> Ltx2MultiModalGuidance -> Ltx2EulerStep
          -> Ltx2AudioDecoderForward -> Ltx2VocoderWithBweForward -> audio.wav
```

The command-line arm is the same call: `ltx2-gen --pipeline-kind t2a_one_stage`
sets the load extra and calls `vllm_video_generate`, as a thin ABI client that
includes no internal header.

**What is NOT reachable, stated rather than left to be found.** `/v1/videos`
carries LOAD parameters, so `pipeline_kind` reaches it in principle; the
per-generation extras this row adds do NOT, because
`VideoGenParamsFromRequest` never forwards `VideoRequest::metadata` to
`VideoGenParams::extras` (#928). A T2A render over that route therefore takes the
recipe's own guider defaults and cannot override them. That is a pre-existing
defect of the route rather than of this row, and the reach claim above is worded
to exclude it. Whether the route's LOAD path was exercised end to end here is
recorded in the final report as measured or unverified — it is not asserted.

**This row ends ONE test-only driver, and leaves three standing.** An earlier
draft of this section claimed all four, and the tree contradicts it:

| Symbol | Production call site after this row |
|---|---|
| `Ltx2MultiModalGuidance` | **yes** — `src/vllm/model_executor/models/ltx2_t2a.cpp`, in the guided step. This is the one this row ends |
| `Ltx2CfgDelta` | no — reached only through `Ltx2Guidance` (`ltx2_pipeline.cpp:532`) and from `tests/vllm/models/test_ltx2_pipeline.cpp:615` |
| `Ltx2StgDelta` | no — the same, at `ltx2_pipeline.cpp:534` and `test_ltx2_pipeline.cpp:630` |
| `Ltx2BatchedPerturbationConfig` | no — constructed nowhere outside `test_ltx2_pipeline.cpp:832-859`. T2A uses `Ltx2DitPerturbation`, which is a different type |

`Ltx2Guidance` itself is the reason the two deltas stay dead: its only caller in
the tree is `tests/vllm/models/test_ltx2_pipeline.cpp:710`, so the switch that
would route a configured guider kind to them is reached from no product path.

**This row does not owe that wiring.** All four landed with #641, before this row
existed, and none of them is on the T2A path — T2A resolves a
`Ltx2MultiModalGuiderParams` and calls `Ltx2MultiModalGuidance` directly, exactly
as `_guided_denoise` does upstream. Recorded here because the spec must not
assert what `git grep` refutes, and listed under `## Owed` with the issue that
tracks it.

## 7. Quantized arms

| Arm | Disposition |
|---|---|
| bf16 / f32 safetensors | **ported** — the reduced fixture is this, and the gate runs on it |
| NVFP4, FP8 (the DiT tower) | **ported by inheritance, and that is a claim about the load rather than about a render.** The T2A path consumes `Impl::dit.weights`, which is whatever arm `Ltx2LoadDitFromSafetensors` materialized; this row adds no GEMM, changes no dtype and selects no arm. It is recorded as UNMEASURED on real quantized weights, because the GPU was out of bounds |
| GGUF k-quants | **not applicable**, and not merely undone. Upstream ships no GGUF arm for any LTX-2 component: `quantization_factory.py:23-26` enumerates the inference kinds exhaustively as fp8-cast, fp8-scaled-mm, nvfp4-cast and nvfp4-prequant, with `assert_never` at `:50`. There is no upstream behaviour to mirror and no quant-matched llama.cpp comparison to serve, because llama.cpp does not carry this architecture |
| int8-convrot | **out of scope**, unchanged by this row and already refused by name through `kInt8ConvRot` |

Dtype polarity: the T2A path is f32 throughout on the host arm, which is what
`Ltx2DitForward` declares and refuses to widen, and the audio VAE and vocoder are
f32 by upstream's own choice (`vocoder.py:585-595`, mirrored in
`ltx2_audio_vae.cpp:1-12`). No buffer here is wider than the arm it feeds.

---

## 7b. The bug this row found and fixed in flow

[#1013](https://github.com/mudler/vllm.cpp/issues/1013). `OneStagePhase`
(`ltx2_pipeline.cpp`) left `Ltx2PhaseRecipe::noise_scale` at the struct default
of **0.0**, and 0.0 is not "no extra noise": `Ltx2GaussianNoise` is
`latent + noise_scale * (noise - latent)`, so at 0.0 the state stays exactly as
`create_initial_state` wrote it, which with no initial latent is **all zeros**.
A `one_stage` render therefore denoised a zero tensor on both streams.

Upstream's `ModalitySpec.noise_scale` defaults to 1.0
(`ltx-pipelines/utils/types.py:110`) and `TI2VidOneStagePipeline.__call__`
constructs both specs without it (`ti2vid_one_stage.py:233-239`). The two
neighbouring recipes set it explicitly, which is what made the omission legible.

Fixed here rather than deferred, per AGENTS.md § *Every change starts from an
issue*, because the `t2a_one_stage` rows are built FROM `OneStageRecipe` and
would have inherited it. No gate saw it because every end-to-end test loads
`distilled_two_stage`, and a zero-initialized denoise still returns a finite clip
of the right size, frame count and sample rate. M6 above is the mutation that now
holds it.

**NOT claimed for `dmd2`.** `PositiveOnlyRecipe` leaves the same field at 0.0 and
its source is vLLM-Omni's `LTX_POSITIVE_ONLY_RECIPE`, which is not checked out
here. Left as-is and named rather than corrected by analogy: a recipe whose
upstream nobody read is exactly where a plausible fix lands wrong. Recorded in
#1013.

## 7c. The second bug this row found and fixed in flow

[#1039](https://github.com/mudler/vllm.cpp/issues/1039). **The guidance passes
were combined in VELOCITY space, and upstream combines x0.** Found by review of
pull request #1032 at `3d9d9c9bb`, before the branch landed.

Upstream never hands the denoiser the raw velocity model. `DiffusionStage`
builds `X0Model(self._prepared_builder().build(device=target, **kwargs))`
(`ltx-pipelines/utils/blocks.py:480-482`), and `X0Model.forward` returns
`to_denoised(audio.latent, ax, audio.timesteps)` (`ltx-core
model/transformer/model.py:590-604`), which is `sample - velocity * sigma`
(`ltx-core utils.py:39-52`). So `_guided_denoise`'s
`all_v, all_a = transformer(...)` (`ltx-pipelines/utils/denoisers.py:188`)
already carries DENOISED tensors, and `audio_guider.calculate(cond_a, uncond_a,
ptb_a, mod_a)` at `:203` combines those.

`Ltx2T2aGenerate` took `Ltx2DitForward`'s velocities straight into
`Ltx2MultiModalGuidance` and applied `ToDenoised` once to the result.

**Why no existing gate saw it, and why the row's own §5 could not.**
`MultiModalGuider.calculate`'s linear terms (`guiders.py:261-266`) are invariant
under `x0 = latent - sigma*v`, so the two forms agree EXACTLY while
`rescale_scale == 0`. The rescale at `:268-271` is not invariant: upstream's
`factor` is `std(x0_cond)/std(x0_pred)` and it scales the whole x0, giving
`factor*(latent - sigma*v)`, where scaling the velocity gives
`latent - sigma*factor*v`. The two differ by `(factor - 1) * latent` — zero
only where the latent is zero, which on this path it never is, because the state
IS the unit-variance noise (§3.2 item 6). `rescale_scale = 0.7` is the shipped
default (`utils/constants.py:63`, `utils/args.py:1101-1106`), so the DEFAULT arm
took the divergent branch. Everything §5 observes — the three forward counters,
`t2a_video_stream_present`, `t2a_perturbed_blocks`, the latent absmax, the
waveform's length, channel count and sample rate — is identical between the two
forms.

**Fixed by moving the conversion, not by moving the rescale.** The mirror is
structural: the per-pass `x0_model` lambda in `ltx2_t2a.cpp` IS `X0Model`, it
applies `ToDenoised` on the way out of every forward, and
`Ltx2MultiModalGuidance` stays a faithful port of `calculate` over whatever the
model returned. Reaching the same numbers by moving the rescale inside the
guidance seam would put `to_denoised` inside `calculate`, where upstream does not
have it, and would make the seam correct only for this one composition.

**The VIDEO arm is unaffected, and that was checked rather than assumed.**
`git grep -n Ltx2MultiModalGuidance -- src include` returns exactly one
production call site, `ltx2_t2a.cpp`. `Ltx2PipelineParams::video_guider` and
`Ltx2PhaseRecipe::video_guidance` are carried by the recipe and read by nothing:
the joint driver in `ltx2_video.cpp` runs ONE unguided `Ltx2DitForward` per step
and applies `ToDenoised` to that single velocity (`:3034-3036`), which is the
same tensor in both spaces because there is no combination to be invariant
under. There is therefore no second instance of this defect to fix, and there
will be one the moment a guided video denoiser is wired — noted here because
that wiring is a live campaign item.

**What the gate is, and what it deliberately is not.** The reduced fixture
CANNOT resolve the rescale's numeric consequence, and that is measured rather
than assumed: its DiT responds to the conditioning at ~1e-5 of its own output,
so `std(cond)/std(pred)` is 1.0 to 1e-5 in BOTH spaces, both factors land within
1e-5 of 1.0, and the two candidate step-0 predictions sit 7.6e-07 apart against
a span of 3.41. The first draft of the test asserted exactly that difference and
its own separation guard refused it — a case that would have been green either
way. So the row gates the defect at two places instead:

1. `test_ltx2_video` "the guider is handed x0 predictions and not raw
   velocities" — end to end through `LoadVideoEngine` and
   `VideoEngine::Generate`, on the recipe's own guider with no extra touched. It
   pins the EQUATION `cond == latent - sigma*velocity` between three recorded
   step-0 tensors. That is exact in x0 space and off by the whole sample in
   velocity space, so no fixture scale satisfies it by accident; a zeroed
   velocity or a zero sample fails the two `REQUIRE`s that precede it rather
   than passing it.
2. `test_ltx2_video` "rescale_scale 0 is the control because both spaces agree
   there" — the numeric consequence, on the real `Ltx2MultiModalGuidance` seam
   with a latent that makes it visible. MEASURED: relative disagreement between
   the two spaces is **1.50e-07 at `rescale_scale = 0.0`** and **0.352 at the
   shipped 0.7**. This is what makes 0.0 the control rather than the assertion
   site.

The observability this needed is four step-0 fields on `Ltx2T2aResult` and the
trace: the sample, the conditional pass's RAW velocity, the tensor handed to the
guider, and the guider's result, plus step 0's sigma. `first_step_cond` is
upstream's own `DenoisedLatentResult.cond` (`utils/denoisers.py:206`).

### Mutations for #1039

Focused gate: three comma-free `--test-case` filters, each asserting a non-zero
case count. Each mutation applied to ONE file, rebuilt, run, restored in a
`finally` and the restore verified by **sha256**. `git diff --stat` is scoped to
the mutated file and measured against the committed fix, so the numbers are the
mutation's own.

| Mutation | `git diff --stat` | BUILT | exit | verdict |
|---|---|---|---|---|
| N1 revert to velocity-space guidance (the defect) | `ltx2_t2a.cpp \| 4 ++--` | YES (0 errors) | 1 | DETECTED by case 1 |
| N2 delete the production call site | `ltx2_video.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED by case 1 AND the render case |
| N3 take x0 against a ZERO sample instead of the latent | `ltx2_t2a.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED by case 1 |
| N4 drop the rescale branch entirely (`guiders.py:268-271`) | `ltx2_pipeline.cpp \| 2 +-` | YES (0 errors) | 1 | DETECTED by the control case |

N4 is the row that proves the control case is not decorative: it is the only one
of the four that case 1 does not see, and the only one the control does.

**N1 is the RED-before**, and this is what it printed:

```
tests/vllm/multimodal/test_ltx2_video.cpp:5371: ERROR:
  CHECK( err_x0 <= 1e-5 * latent_span ) is NOT correct!
  values: CHECK( 3.43642 <= 3.38677e-05 )
  logged: sigma = 1  max|latent| = 3.38677  max|velocity| = 0.415609
          |cond - (latent - sigma*velocity)| = 3.43642  |cond - velocity| = 0
          elements = 3328
tests/vllm/multimodal/test_ltx2_video.cpp:5378: ERROR:
  CHECK( err_v > 1e-2 * latent_span ) is NOT correct!
  values: CHECK( 0 >  0.0338677 )
[doctest] test cases:  1 |  0 passed | 1 failed | 66 skipped
[doctest] assertions: 16 | 14 passed | 2 failed |
[doctest] Status: FAILURE!    exit 1
```

`|cond - velocity| = 0` **exactly** is the whole finding: the tensor handed to
`Ltx2MultiModalGuidance` WAS the raw DiT velocity. Green after, on the same
filter: 1 case, 16 assertions, 0 failed, exit 0.

### The #1039 gate covered ONE of the three guidance arms

The fresh review of `c1fe35592` passed on the correctness of the fix and
returned one blocking finding: the case above recorded `first_step_velocity` and
`first_step_cond` for the CONDITIONAL pass, nothing observed the other two arms,
and nothing pinned what `Ltx2EulerStep` consumed. `ltx2_t2a.cpp:41-43` claims the
conversion is applied to EVERY PASS, and the gate held that claim for one third
of them.

Reproduced before the repair, on the same filter as the green run
(`--test-case=ltx2 t2a*`, comma-free, 10 cases / 526 assertions at
`c1fe35592`). Each mutation applied to ONE file, `git diff --stat` taken against
the pre-mutation working tree rather than against `HEAD`, rebuilt with the
`: error:` count printed beside the verdict, exit code captured directly, and
restored from a content snapshot with `os.utime(now)` and a sha256 compare.

| Mutation | before the repair | after |
|---|---|---|
| A1 the PERTURBED (STG) pass alone left in velocity space | SURVIVED, exit 0, 10/526 | DETECTED, exit 1 |
| A2 the UNCONDITIONAL pass alone left in velocity space | SURVIVED, exit 0, 10/526 | DETECTED, exit 1 |
| R1b `ToDenoised` applied a SECOND time to the guider's output, between the step-0 record and the Euler step | SURVIVED, exit 0, 10/526 | DETECTED, exit 1 |
| R1c the same double application ABOVE the step-0 record, so the recorded `t2a_first_denoised` is itself doubly converted | SURVIVED, exit 0, 10/526 | DETECTED, exit 1 |
| A4 the perturbed arm's recorded velocity ZEROED — the guard, not a defect | n/a (the field did not exist) | DETECTED, exit 1, by the `REQUIRE` |

R1c is not from the review. It was found while closing R1b: the reviewer's R1b
sits between the record and the step, so the recovered-Euler-input check sees
it, and moving the same edit one statement earlier does not. That is why the
repair adds a second, independent check rather than one.

**The repair is observability plus three checks, not a change to the fix.**
`Ltx2T2aResult` and the trace gain a (velocity, x0) pair for the unconditional
and perturbed arms and the latent the Euler step wrote. The case then applies the
SAME equation to every arm, replays `Ltx2MultiModalGuidance` over the three
recorded arms and requires bit equality with `t2a_first_denoised`, and recovers
`t2a_first_next_latent` from `t2a_first_denoised` through the Euler formula.

**Non-vacuity, per arm rather than once.** `latent_span > 1e-3` stays shared —
a zero sample makes the two candidate tensors coincide on every arm. Its partner
`sigma * velocity_span > 1e-6` moves INSIDE the per-arm loop, because a zero
velocity makes `to_denoised` the identity for that arm alone, and "expected zero,
and a stub also produces zero" is the trap this campaign has hit twice. A4 is the
mutation that proves that guard is armed: zeroing one arm's recorded velocity
takes the case red through the `REQUIRE`, at 538 assertions rather than 548
because the `REQUIRE` aborts the case. The replay check adds its own
(`t2a_first_denoised != t2a_first_cond`, the guider MOVED what it was handed) and
the Euler check adds two (`|dt| > 1e-3`, so the step is not the identity, and
`scale > 1e-3`, so the residual bounds something).

**The rescale's numeric difference is still NOT asserted, and the reason was
re-measured.** `std(cond)/std(pred)` is 1 to printed precision on this fixture,
so `factor = 0.7*1 + 0.3 = 1`, the rescale is an exact no-op in BOTH spaces, and
the difference term `(factor - 1) * latent` is identically zero. Owed against the
real-checkpoint render, as before.

Green after: `--test-case=ltx2 t2a*` at 10 cases / **548** assertions / 0 failed
/ exit 0, up from 526.

## Owed

- **The rescale's numeric consequence END TO END.** Gated at the seam (0.352
  relative at the shipped 0.7) and at the space (exactly, through the engine),
  and NOT on a render, because the reduced fixture's guidance deltas are ~1e-5
  of the prediction and both rescale factors land within 1e-5 of 1.0. What would
  close it is the real-checkpoint render already owed below, where the DiT's
  velocity is comparable to the sample. Tracked by
  [#1039](https://github.com/mudler/vllm.cpp/issues/1039).
- **The DEVICE arm.** `Ltx2DitForwardDevice` dereferences `*video`
  unconditionally from its first `PrepareStreamDev` call onward
  (`src/vllm/model_executor/models/ltx2_device.cpp`, the two `PrepareStreamDev`
  calls and the per-block `a.batch = video->batch`), so a one-stream device
  forward is a rewrite of that function rather than a lifted check. T2A on
  `device != 0` is REFUSED BY NAME rather than served the host forward behind a
  device handle, because that substitution is what would make every later timing
  claim false. Tracked by #1005.
- **STG on the DEVICE forward**, for the same reason.
- **The sigma-BINNED guider factory** (`guiders.py:294-342`,
  `MultiModalGuiderFactory.from_dict`). `Ltx2GuiderParamsForSigma` is ported and
  gated; nothing here constructs bins, because no upstream surface on this
  pipeline varies params by sigma — `t2a_one_stage.py:196-205` passes a plain
  `MultiModalGuiderParams`. A caller may pass a factory (`:116`), and no CLI
  does.
- **`uncond_modality` (isolated-modality guidance).** REFUSED BY NAME on this
  pipeline, and the refusal is upstream's own reasoning rather than a local
  limit: `modality_scale` is pinned to 1.0 for T2A because there is no video
  modality to isolate (`t2a_one_stage.py:200-202`). A caller who reaches it
  through another pipeline gets a message naming the missing fourth forward.
- **AUTO duration.** `resolve_num_frames`' predicting arm needs a constructed
  `DurationPredictor`; `Ltx2DurationHeadForward` is ported and gated (including
  the audio-only case, `kLtx2DurAudioOnlyGolden`) and nothing constructs one.
  Inherited from the engine, not introduced here.
- **A one-step schedule.** `Ltx2SigmaSchedule(1, ...)` returns `-nan` as its
  first sigma, mirroring upstream's own division by zero
  (`schedulers.py:49-54`). Neither side is checked against it; a T2A request with
  `steps = 1` therefore produces NaN. Named rather than defended against,
  because the correct behaviour is upstream's to decide and this port does not
  get to invent one.
- **The `dmd2` recipe's `noise_scale`**, see §7b.
- **A real-checkpoint T2A render.** Gated on the reduced fixture only; the GPU
  was out of bounds for this row.
- **THE LTX-2.5 CHECKPOINT PIN.** `docs/USAGE.md` names six LTX-2.5 artifacts by
  bare file name and gives no HuggingFace repo, no revision and no sha256 for
  any of them — `:663-670` and `:2183-2188` on `origin/main` at `d1b0ea3a8`,
  plus the text-to-audio recipe this row added at `:853-857`. AGENTS.md
  § *Say which weights, and from where* requires all three, per arm. It is
  campaign-wide and pre-existing rather than introduced here: `grep -n sha256
  docs/USAGE.md` returns two checkpoint hashes and both belong to
  MiniMax-Music3 (`:3127`, `:3269`), while MiniMax-H3 (`:1950-1993`) and
  MiniMax-Music3 (`:3123-3149`) each carry a full table and LTX-2.5 carries
  none. RECORDED AND NOT FABRICATED: this row claims no render on real weights,
  so there is no checkpoint it was gated against to pin, and inventing a repo id
  would be worse than the gap. The real-checkpoint render owed above is what
  closes it. Tracked by
  [#1048](https://github.com/mudler/vllm.cpp/issues/1048).
- **`Ltx2Guidance` and the two deltas it gates are DEAD in production**, and so
  is `Ltx2BatchedPerturbationConfig`. See §6b for the measured table. All four
  landed with #641, none is on the T2A path, and this row ends only
  `Ltx2MultiModalGuidance`'s test-only-driver state. Tracked by
  [#1049](https://github.com/mudler/vllm.cpp/issues/1049).
- **`test_engine_core_proc`'s immediate-shutdown case is load-dependent**, and
  no issue named it until now. Its abort frame is searched for over a FIXED 1000
  dequeues while a `max_tokens=100000` request keeps producing token deltas, so
  the budget is a bet on scheduling. MEASURED at `37e680cab`, same binary
  throughout: 2 failures in 3 `ctest -j4` runs of the full suite, 0 in 25 solo
  runs on an idle box, 0 in 25 solo runs against 20 spinning processes, and 0 in
  two `ctest -R` runs. The third `-j4` run failed `test_cpu_threadpool` INSTEAD,
  so the identity of the failing test rotates between runs of an unchanged
  binary. This branch touches no file under `tests/vllm/v1/` or `src/vllm/v1/`.
  The earlier revision of the PR body attributed it to #294, which is a
  DIFFERENT defect in `test_async_llm`. Tracked by
  [#1052](https://github.com/mudler/vllm.cpp/issues/1052).
- **The guider rescale's `std` comment states an impossible consequence.**
  `ltx2_pipeline.cpp:505-506` and `ltx2_pipeline.h:319-322` say torch's unbiased
  (N-1) `std` matters and the biased one "would be a small, everywhere,
  resolution-dependent gain error". `factor = std(cond)/std(pred)` divides two
  `std`s over the same count, so the `(n-1)` cancels exactly and the two
  estimators give the same ratio. A review mutation from biased to unbiased
  survived because it is an IDENTITY, not because the gate is blind — which is
  worth writing down, because a survivor at that site otherwise reads as a blind
  instrument and costs another investigation. The CODE is right; the COMMENT is
  the defect. Pre-existing from `cefacd2d0` (#641) and out of this row's scope.
  Tracked by [#1050](https://github.com/mudler/vllm.cpp/issues/1050).
- **Value goldens from executed upstream for the T2A COMPOSITION.** The bricks
  either side have them; the chain does not. What would close it is a section in
  `scripts/gen-ltx2-pipeline-goldens.py` that instantiates
  `LTXAudioOnlyModelConfigurator` at reduced dimensions and runs one guided step.
- **`max_batch_size`** (`t2a_one_stage.py:120`, `:169`), the prompt enhancer
  (`:118-119`), LoRA/quantization/compilation/offload constructor arguments
  (`:56-63`) — the engine's own surfaces, unchanged by this row.

## 8. Stop conditions

Stop and report rather than widening scope if: lifting the `video != nullptr`
half of the DiT refusal moves any existing LTX-2.5 test; the STG perturbation
cannot be expressed without changing an existing forward's numerics; or the
`READER ANCHORS` gate cannot be re-derived deterministically.

## 9. Now

`ACTIVE` — spec committed before implementation, per `AGENTS.md` § *Spec before
code*.
