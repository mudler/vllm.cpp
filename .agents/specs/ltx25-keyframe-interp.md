# LTX25-KEYFRAME-INTERP — the interpolation pipeline, and the conditioning branch it deletes

Row `LTX25-KEYFRAME-INTERP`. Issue
[#1096](https://github.com/mudler/vllm.cpp/issues/1096). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Upstream pin: Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified with `git rev-parse HEAD` in `/home/mudler/_git/LTX-2` on 2026-08-18.

Base: `8daf58e77`, pinned when the branch was created.

## Now

`ACTIVE` -> `DONE` with this change. `KeyframeInterpolationPipeline`
(`keyframe_interpolation.py:55`) becomes `pipeline_kind = "keyframe_interpolation"`
on the four generations the recipe table keys.

## Scope

**In.** One recipe row, `KeyframeInterpolationRecipe`, and its four
`(kind, version)` keys. One new recipe field,
`Ltx2PipelineRecipe::image_conditioning`, because this pipeline is the first arm
that uses upstream's OTHER image-conditioning builder and this engine can only
express the one. The docs and the CLI help that list the kinds.

**Out.** The three arms #1150 leaves divergent on the schedule anchor — this row
sets its own phase and touches nothing else. Per-phase adapter STRENGTH (#1144).
An N-image request surface with per-image `frame_idx` — see `## Owed`. The
real-weights render, which needs a GPU lease — see `## What is NOT verified`.

## What was genuinely missing, against what #1096 recorded

`.agents/issue-index.md:321` is the audit this row inherited. It named two
blockers and a missing artifact. **Two of the three are stale, one is real but
is not what makes this pipeline different, and the difference that DOES make it
different is not in the audit at all.** Each is re-derived at the pin below,
because "the audit said so" is a search result and not a finding.

### (a) "no multi-keyframe request surface" — TRUE, and not the blocker

Still true as stated: the ABI carries two scalar slots, `first_frame` and
`last_frame` (`include/vllm.h:947-948`), against upstream's repeatable
`--image PATH FRAME_IDX STRENGTH [CRF]` (`utils/args.py:805-817`). The indices
are still fixed — `latent_idx=0` and `frame_idx=frames-1`.

But two pinned keyframes at the two ENDS of the clip is what
"interpolate between keyframes" means at its default configuration, and this row
serves that. An arbitrary interior `frame_idx` is a request-surface row of its
own, owed below with an issue. **A pipeline that cannot be asked for a third
keyframe is narrower than upstream; a pipeline that maps its first keyframe onto
the wrong conditioning item is WRONG**, and only the second stops the row.

### (b) "a per-sigma guided denoiser" — STALE on this pipeline's default path

The audit is right that `FactoryGuidedDenoiser` resolves guiders per step from
sigma (`utils/denoisers.py:332-343`). It does not follow that this pipeline ever
uses more than one, and at the pin it does not:

- `main()` builds plain `MultiModalGuiderParams` for both streams
  (`keyframe_interpolation.py:325-340`), never a `MultiModalGuiderFactory`.
- `create_multimodal_guider_factory` therefore takes its last line,
  `MultiModalGuiderFactory.constant(params, ...)` (`guiders.py:360`).
- `constant` builds `_params_by_sigma = ((inf, params),)` (`guiders.py:312-315`)
  — ONE bin, so `build_from_sigma` returns the same guider at every sigma.
- `FactoryGuidedDenoiser.__call__` then delegates to `_guided_denoise`
  (`denoisers.py:345-358`), which is what `Ltx2GuidedDenoise` ports.

So on the configuration this row ships, `FactoryGuidedDenoiser` IS
`GuidedDenoiser`. `ltx25-ti2vid-recipe.md` recorded the same reduction for
`ti2vid_two_stages.py`, which selects the identical parser. The seam landed at
`daeff67f2` (#1092/#1102) and `kGuided` reaches it.

The sigma-BINNED arm is real and is reachable only by a caller who constructs a
`MultiModalGuiderFactory.from_dict` and passes it as `video_guider_params` — no
CLI flag builds one. It is owed below, under the issue that owns it for every
arm rather than for this one.

### (c) "neither file is on the NAS" — STALE

Both are, and byte-verified: the dev transformer at 42,018,190,584 B and the
distilled adapter at 8,899,889,568 B, under
`/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/`. `PlanDit`'s refusal of
a pure-BF16 DiT (#1148) closed at `40a796aa9`. What is owed is the RUN.

### (d) THE DIFFERENCE THE AUDIT DOES NOT NAME: the conditioning builder

`ti2vid_two_stages.py:231` and `:276` call `combined_image_conditionings`; this
pipeline's `:211` and `:260` call `image_conditionings_by_adding_guiding_latent`.
(Both files carry a `:211`, and this line named the wrong one until the review
repair: `ti2vid_two_stages.py:211` is blank.) They are
different functions and the difference is one branch:

| | `combined_image_conditionings` (`helpers.py:272-308`) | `image_conditionings_by_adding_guiding_latent` (`helpers.py:343-367`) |
|---|---|---|
| `frame_idx == 0` | `VideoConditionByLatentIndex` (`:295-300`) | `VideoConditionByKeyframeIndex` |
| any other `frame_idx` | `VideoConditionByKeyframeIndex` (`:301-305`) | `VideoConditionByKeyframeIndex` |
| what frame 0 does to the state | REPLACES latent frame 0's clean tokens; the token count NEVER changes (`latent_cond.py:40-41`) | APPENDS a latent frame of tokens at the end (`keyframe_cond.py:79-82`) |

The second function has no branch at all: it is one loop and one item type. That
is the whole content of the diff between the two files' conditioning, and it is
this pipeline's name — the first image is a KEYFRAME to interpolate FROM, not a
frame to overwrite.

**This engine hard-codes the branch.** `ltx2_video.cpp`'s `wants_first_frame` arm
calls `Ltx2ConditionVideoByLatentIndex` unconditionally and its comment cites
`helpers.py:295-300` by line. Both conditioning primitives are ported and
gated — `Ltx2ConditionVideoByKeyframe` is what the `wants_last_frame` arm below
it already drives — so what is missing is the SELECTION, not a capability.

**And it is invisible to every gate this tree owns.** A `keyframe_interpolation`
render built on the replace arm returns a clip of the right size, the right frame
count and the right sample rate, with the supplied image pinned into it. It is
conditioned; it is conditioned as a different pipeline. The only observable is
the sequence LENGTH the DiT ran over, which is `Ltx2ConditioningTrace::video_tokens`,
and no pixel comparison and no shape check reads it.

## Port map, each line read at the pin

Anchors are
`packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py` unless
another file is named.

### Stage 1 — `stage_1` (`:100-110`, called at `:231-252`)

| Field | Value | Upstream |
|---|---|---|
| `spatial_downscale` | 2 | `:203-209`, `width // 2` / `height // 2` |
| `sigmas` | empty (derived) | `:199-200`, `self._scheduler.execute(steps=num_inference_steps)` |
| `schedule_tokens` | `kSchedulerDefault` | the same call passes NO latent; `schedulers.py:32` reads that as `default_number_of_tokens` = `MAX_SHIFT_ANCHOR` = 4096 (`:11`, `:29`) |
| `noise_scale` | 1.0 | `:244-247` sets none; `ModalitySpec.noise_scale` defaults 1.0 (`utils/types.py:110`) |
| `video_guidance` | `params.video_guider` | `:222-225`, from `MultiModalGuiderParams` the CLI fills (`:325-332`) |
| `audio_guidance` | `params.audio_guider` | `:226-229`, filled from six `--audio-*` / `--v2a-guidance-scale` flags (`:333-340`) |
| `denoiser` | `kGuided` | `:232`, `FactoryGuidedDenoiser` — see (b) above for why that is `Ltx2GuidedDenoise` here |
| `allow_guidance_override` | true | `:301` selects `default_2_stage_arg_parser`, which carries the six guider flags (`utils/args.py:947-1006`) |
| `loras` | `kNoAdapters` | `:104`, `loras=tuple(loras)`, against `:111`'s `stage_2_loras` |
| `stepper` | `kEuler` | derived, below |

### Stage 2 — `stage_2` (`:112-122`, called at `:271-290`)

| Field | Value | Upstream |
|---|---|---|
| `sigmas` | `Stage2DistilledSigmas()` | `:166`, `stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS` — a DEFAULT ARGUMENT, so frozen |
| `use_official_sigma_schedule` | false | the schedule is explicit |
| `noise_scale` | `Stage2DistilledSigmas().front()` | `:282` and `:287`, `stage_2_sigmas[0].item()` on BOTH modality specs |
| `input_transform` | `kSpatialUpsample` | `:255`, `self.upsampler(video_state.latent[:1])` |
| `denoiser` | `kSimple` | `:272`, `SimpleDenoiser(v_context_p, a_context_p)` — takes no params |
| `allow_guidance_override` | true | the flags are legal on this parser and reach stage 1's guider alone; `kSimple` is what makes them inert here |
| `loras` | `kAllAdapters` (default) | `:111`, `(*tuple(loras), *tuple(distilled_lora))`, passed at `:116` |
| `stepper` | `kEuler` | derived, below |

### Recipe (`:147-168`, `:297-358`)

| Field | Value | Upstream |
|---|---|---|
| `height` / `width` | `params.stage_2_*()` | `:301` sets the request geometry to the FINAL output (`utils/args.py:1128`) |
| `negative_prompt` | per version | `:178-186` encodes `[prompt, negative_prompt]` and reads `ctx_n` into both guider factories (`:224`, `:228`) |
| `video_output_phase` | 1 | `:292` decodes the name `:271` rebound |
| `audio_output_phase` | **1** | `:271` is `video_state, audio_state = self.stage_2(...)` — NOT a discard — and `:293` decodes that name |
| `allow_request_sigmas` | true | `:165` `stage_1_sigmas` is a real parameter and `:200` honours it |
| `allow_request_latents` | false | no `__call__` parameter carries one (`:147-168`) |
| `allow_negative_prompt` | true | `:150` |
| `requires_distilled_lora` | **true** | `distilled_lora` is POSITIONAL and non-defaulted (`:68`), and `--distilled-lora` is `required=True` (`utils/args.py:1140-1155`) on the parser `:301` selects |
| `requires_audio_input` | false (default) | there is no `--audio-path`; the soundtrack is generated |
| `audio_only` | false (default) | `:292-293` decodes both |
| `image_conditioning` | **`kAddGuidingLatent`** | `:211` and `:260` — the new field, section (d) above |

### `audio_output_phase = 1`, and it is the opposite of `ti2vid_two_stage`

These two pipelines share a parser, a stage layout, a stepper and a sigma set,
and they disagree here. `ti2vid_two_stages.py:287-289` carries upstream's own
comment — "Stage 2 refines video only; discard its audio" — and binds
`video_state, _`. `keyframe_interpolation.py:271` binds `video_state,
audio_state` and `:293` decodes it. There is no comment either way in this file;
the binding IS the statement.

**So the field most likely to be copied from `ti2vid_two_stage` is the one that
must not be.** Writing 0 here decodes stage 1's soundtrack, which is finite, the
right length, at the right sample rate, and the wrong take. The recipe case
asserts both polarities side by side for that reason.

### `stepper = kEuler` is derived, not assumed

Neither `self.stage_1(...)` (`:231-252`) nor `self.stage_2(...)` (`:271-290`)
passes `stepper` or `loop`. `DiffusionStage.__call__` declares both as `None`
defaults (`utils/blocks.py:512-513`) and fills them at `:524-527` with
`euler_denoising_loop` and `EulerDiffusionStep()`. `distilled.py:76-84` selects
the ANCESTRAL stepper on generation 2.5 and reaches this pipeline through
nothing.

### Four version keys

`2`, `2.3`, `2.4`, `2.5`, mirroring `ti2vid_two_stage` and `a2vid_two_stage`.
`main()` calls `resolve_cli_params()` (`:300`) and hands the result to
`default_2_stage_arg_parser(params=params)` (`:301`), so the generation comes off
the CHECKPOINT and there is no "which generations support this pipeline"
question upstream. The 2 and 2.3 rows carry `kOmniNegativePrompt`; 2.4 and 2.5
carry `LightricksNegativePrompt()`.

### `num_frames` is REQUIRED here and optional on `ti2vid_two_stage`

`:154` is a positional, non-defaulted `num_frames: int`, and this pipeline builds
no `DurationPredictor` — `ti2vid_two_stages.py:174` takes
`int | AutoDuration = DEFAULT_AUTO_DURATION` and resolves it from the caption.
Recorded rather than expressed: this engine has never auto-predicted a duration,
so every recipe already behaves as this pipeline does, and a field would have one
value on every row.

## The new field, and why it is on the RECIPE and not the phase

`Ltx2PipelineRecipe` gains
`Ltx2ImageConditioningBuilder image_conditioning = kCombined`:

- `kCombined` — `combined_image_conditionings` (`helpers.py:272-308`). Frame 0
  REPLACES. **The DEFAULT, and it is today's behaviour, so no landed arm moves.**
- `kAddGuidingLatent` — `image_conditionings_by_adding_guiding_latent`
  (`helpers.py:343-367`). Frame 0 APPENDS, like every other frame.

**Per recipe, because upstream calls the same builder for both of this
pipeline's stages** (`:211` for stage 1 and `:260` for stage 2 — the only
difference between the two calls is the height and width). A per-phase field
would offer a combination upstream has no site for, and every phase field this
engine carries exists because some upstream pipeline sets it per stage.

**A preserving default cannot fail silently.** With `kCombined` default, an arm
moves only where a line says so. The upstream-faithful-everywhere alternative
does not arise: `combined_image_conditionings` is what the other ten pipelines
call, so `kCombined` IS the majority as well as the incumbent.

The engine reads it in one place, the `wants_first_frame` arm of the phase loop
in `ltx2_video.cpp`, which becomes a two-way branch on the field. The
`wants_last_frame` arm below it is unchanged: `frame_idx != 0` takes the keyframe
item under BOTH builders.

`causal_fix` stays `true` at the call. `VideoConditionByKeyframeIndex` gates it
itself — `latent_tools.causal_fix if self.frame_idx == 0 else False`
(`keyframe_cond.py:49`) — and `Ltx2ConditionVideoByKeyframe` already mirrors that
at `ltx2_conditioning.cpp:548`. So passing `true` with `frame_idx = 0` is the
one combination in which the fix is APPLIED, which is upstream's, and passing
`false` would silently drop it.

### The trace has to follow the tokens

`im.trace.image_tokens` / `image_digest` / `image_absmax` are digested from the
tokens the item WROTE, not from the encoder's output, and the replace arm reads
them off the FRONT of `video.clean`. Under `kAddGuidingLatent` the written tokens
are at the TAIL. Reading the front there would digest unconditioned tokens, and
the instrument would report a healthy conditioning for a build that placed
nothing — the exact failure the field's own comment in `ltx2_video.h` says it
exists to prevent. The digest slice is therefore taken from the range the append
actually grew.

## Tests

Upstream ships no unit test over a pipeline's recipe — the recipe IS the
constructor — so these are ported in the sense that every assertion cites the
line it mirrors, and the harness is ours.

### 1. The recipe, field by field (`test_ltx2_pipeline.cpp`)

Mirrors `"ltx2 ti2vid: the recipe is the PLAIN two-stage pipeline, not the HQ
one"` (`:3521`). Every field in the Port map, each against a CONTROL drawn from
the recipe it would otherwise be confused with — `ti2vid_two_stage` above all,
which it is TWO fields from:

- `audio_output_phase == 1`, control `ti2vid_two_stage` at 0 and
  `res2s_two_stage` at 0. The two-against-two split is what stops this passing
  because every two-stage recipe happens to agree;
- `image_conditioning == kAddGuidingLatent`, control `ti2vid_two_stage`,
  `a2vid_two_stage`, `res2s_two_stage`, `distilled_two_stage` and `one_stage` all
  at `kCombined`;
- stage 1 `loras == kNoAdapters`, control `res2s_two_stage` stage 1 at
  `kAllAdapters`;
- stage 1 `schedule_tokens == kSchedulerDefault`, control `res2s_two_stage` at
  `kTargetLatent`;
- stage 1 `audio_guidance.cfg_scale` == the params table's 7.0, control
  `a2vid_two_stage` at the positive-only default;
- `stepper == kEuler` on both phases, controls `distilled_two_stage` at 2.5 on
  `kEulerAncestral` and `res2s_two_stage` on `kRes2s`;
- `requires_distilled_lora` true, `requires_audio_input` FALSE, control
  `a2vid_two_stage` true/true.

Plus a version-key case: all four resolve, and `keyframe_interpolation_two_stage`,
`keyframe`, `keyframe_interp` and version `2.6` refuse by name.

### 2. The conditioning builder REACHES the state (`test_ltx2_video.cpp`)

The row's identity, gated through `LoadVideoEngine` + `Generate` rather than on
the recipe struct: case 1 proves the field is set, this one proves it is
CONSUMED (#1013).

The SAME image, the SAME geometry and the SAME seed on two loads that differ
only in `pipeline_kind`. `Ltx2ConditioningTrace::video_tokens` is the length the
DiT ran over on the last phase, recorded before the trim:

- `keyframe_interpolation` reports `video_tokens == ti.video_tokens +
  kf.image_tokens` — one latent frame of tokens MORE than the plain arm, which
  is what an append costs;
- `ti2vid_two_stage` reports `video_tokens` equal to the bare target grid, taken
  from a third render of the same kind with NO image at all. Without that third
  render the equality above would pass on a tree where both arms append.
- `image_tokens > 0` on both, so neither passes by placing nothing.

The rendered pixels are compared too, and must differ from an unconditioned
render of the same kind — the arms place the same content, so a pixel comparison
between THEM is not the discriminator, and saying which comparison carries which
claim is the point.

### 3. The per-arm x0 invariant on stage 1 (`test_ltx2_video.cpp`)

The correctness trap this arm inherits: guidance combines **x0**, not velocity.
Every linear term is invariant under `x0 = latent - sigma*v`, so cfg, stg and
modality cannot see the difference; the RESCALE branch is not invariant and
`rescale_scale` defaults to 0.7, which is the default path (#1039, #1092).

A magnitude assertion cannot gate it — on a reduced fixture `std(cond)/std(pred)`
is 1.0 to 1e-5 in BOTH spaces. The gate is the per-arm equation
`x0 == latent - sigma*velocity` over three recorded tensors, exact in x0 space
and off by the whole sample in velocity space, where it degenerates to
`|x0 - velocity| = 0`. Both residuals are printed on every arm so a RED says
which space it landed in. Over **all four** passes — cond, uncond, perturbed,
modality — each with its own non-vacuity guard, because a zeroed velocity
collapses the equation to `x0 == latent` for that arm alone.

### 4. The schedule anchor REACHES the sigmas (`test_ltx2_video.cpp`)

`trace.schedule_tokens == 4096` on a `keyframe_interpolation` render at two
geometries, and on the same fixture a `res2s_two_stage` render reports a
DIFFERENT, non-4096 count at each. Both halves are load-bearing: the equalities
alone pass on a build that hard-codes 4096, the inequalities alone on today's
tree.

**The trajectory half must render at three steps or more and must recompute at
the RENDER's step count.** At two steps the schedule is `{1, 0.1, 0}` for every
token count — `stretch` pins sigma[0] at 1.0 and renormalises the last non-zero
sigma to exactly `terminal` = 0.1 (`schedulers.py:48-55`), and a 2-step schedule
has only those two non-zero entries. Three steps is the shortest schedule with an
interior sigma. The step count comes back OUT of the render lambda, and
`rendered_steps > 2` is asserted by name, so lowering it is refused rather than
silently turning the comparison into a value against itself. `ltx25-ti2vid-recipe.md`
shipped this case with a literal `3` and a fresh review found it vacuous; the
mutation table below re-runs that exact mutation here.

### 5. Reachability (`test_ltx2_video.cpp`)

`LoadVideoEngine` with `pipeline_kind = keyframe_interpolation` -> `Generate` ->
pixels, which is `include/vllm.h` plus the documented load extras and is what
`ltx2-gen --pipeline-kind keyframe_interpolation --lora-path ... --upsampler-path
...` does through the ABI. Deleting the dispatch row in
`ResolveLtx2PipelineRecipe` REDs it at the LOAD.

`dit_forwards`, not an evaluation count, distinguishes the guided stage: a
denoiser call is ONE evaluation whether or not guidance ran, and only
`Ltx2ConditioningTrace::dit_forwards` counts actual `Ltx2DitForward` calls.

All three knobs this recipe needs are LOAD extras, which a server supplies
through `--video-extra KEY=VALUE`, and `requires_audio_input` is false — so #928
does not exclude the HTTP route, as it does on `a2vid_two_stage`. That is a claim
about the request SURFACE. Nothing here drives the HTTP route end to end, so it
is not measured and the reach claim rests on the ABI path alone.

### 6. The `requires_distilled_lora` refusal

A `keyframe_interpolation` load with no `lora_path` refuses BY WHAT IS MISSING,
and the message names the pipeline. Mirrors the positional `distilled_lora`
(`:68`) and `--distilled-lora required=True` (`utils/args.py:1140-1155`). Two
controls: the same load WITH `lora_path` renders, and the DEFAULT kind still
loads without one — otherwise the case passes on any load failure or on a new
global requirement.

### Mutations required to pass

1. `recipe.image_conditioning = kAddGuidingLatent` deleted. Test 2 RED.
2. The reader branch in `ltx2_video.cpp` forced to the replace arm. Test 2 RED.
3. `recipe.audio_output_phase = 1` -> 0. The recipe case RED.
4. `stage1.schedule_tokens = kSchedulerDefault` deleted. Test 4 RED.
5. `stage1.loras = kNoAdapters` deleted. The recipe case RED.
6. `recipe.requires_distilled_lora = true` deleted. Test 6 RED.
7. The `keyframe_interpolation` branch deleted from `ResolveLtx2PipelineRecipe` —
   the standing reachability mutation. Tests 2, 3, 4, 5, 6 RED.
8. `Ltx2GuidedDenoise` left in velocity space on one arm. Test 3 RED.
9. Test 4's `gen.steps` 3 -> 2 — the mutation the ti2vid row's FIRST head passed.
   Test 4 RED on both `at_anchor != at_target` and `rendered_steps > 2`.

Every mutation prints FOUR facts: `git diff --stat`, whether it BUILT, the
compile-error count, and the exit code captured directly. The expected OLD line
content is asserted before each edit, because a mutation that landed on a comment
line printed a clean one-line diff, BUILT, 0 errors, exit 0, and read as a pass.

**`stg_blocks = []` is legal.** Upstream defaults it to `[]` and validates it
nowhere. No refusal is added for it; one had to be removed already.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Whole binaries, never a `--test-case` filter: a filter matching zero cases prints
`SUCCESS!` at exit 0, and LTX case names contain commas, which doctest `-tc`
splits on. Assert a non-zero case AND assertion count. A thrown case prints
`0 failed` beside `Status: FAILURE!`, so the exit code is the authority.

Report `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the pass/fail line, and `No space left` / `BFD` greps WITH live
positive controls, plus load and free disk. 511 tests registered at this head.

`READER ANCHORS` (`ltx2_video.cpp`) is gated by `test_ltx2_video` and shifts
whenever the readers above it move. This row edits the conditioning block, which
is BELOW the last anchor, so the list is not expected to move — re-derive with
the test's own walk anyway, and arm the instrument first by inserting a line and
confirming MISMATCH.

**No GPU.** Another agent holds `dgx:gpu0` for a render, and a recipe row is
correctly gated by the CPU goldens.

`ltx2-gen` gains `--last-frame` in this change ([#1191](https://github.com/mudler/vllm.cpp/issues/1191)),
fixed in flow. The ABI field and the engine's reader both predate it; only the
CLI was missing, and this is the first row a missing CLOSING keyframe narrows.

## What is NOT verified

**No real-weights render, and the reason is contention, not a missing artifact.**

Upstream marks this arm `2 | Full + distilled LoRA | Euler | Keyframe
interpolation` (`packages/ltx-pipelines/CLAUDE.md:24`). Stage 1's identity is CFG
on the UNADAPTED model, so the checkpoint it needs is
`ltx-2.5-22b-dev-transformer-bf16.safetensors`. That file is on the NAS and
byte-verified, the distilled adapter is beside it, and #1148 made the pure-BF16
DiT loadable at `40a796aa9`. What remains owed is the RUN: a GPU lease, a build
on that host, upstream's render and ours on the same checkpoint, prompt and seed,
and the comparison.

**Running this arm against a DISTILLED checkpoint instead would be worse than not
running it.** The distilled scales are trained INTO those weights, so a
CFG-guided stage 1 on top samples a trajectory they were never trained for, and
it renders — right size, right frame count, right sample rate, plausible picture,
no diagnostic (#1137).

So the claim this row makes is: gated on CPU goldens, correct against the
upstream SOURCE line by line, and NOT measured against upstream's own render.

## Outcome

Row `DONE`. Recorded here because neither the code nor the Git history carries
it: what the gate MEASURED, and the two things this row found that its scoping
pass did not know.

### The audit was wrong about WHICH blocker, in both directions

#1096 named a per-sigma denoiser and two absent checkpoints, both stale, and a
multi-keyframe surface, which is true and is not what stops the pipeline. It did
not name the conditioning builder, which is the one difference that makes
`KeyframeInterpolationPipeline` a different pipeline rather than a second
`ti2vid_two_stage`.

Both errors have the same shape and it is worth naming: the audit read the
pipeline's IMPORTS and the machinery they reach, and inferred what would be hard.
`FactoryGuidedDenoiser` resolves guiders per sigma, so a per-sigma guider looked
required — and the same file's `main()` passes plain params, so the factory has
one bin. `image_conditionings_by_adding_guiding_latent` and
`combined_image_conditionings` are both one-line imports of a helper, so neither
looked like anything. **The import list says which machinery is reachable; only
the call site says which behaviour runs.**

### What the gate measured

Focused, at the head this row pushes:

| Binary | cases | assertions | exit |
|---|---|---|---|
| `test_ltx2_pipeline` | 56 | 3316 | 0 |
| `test_ltx2_video` | 87 | 2713 | 0 |

RED before the recipe landed, captured on the same binaries: `test_ltx2_pipeline`
56 cases / 54 passed / 3183 assertions, `Status: FAILURE!`, exit 1, both new
cases throwing `Unsupported LTX pipeline kind/version:
'keyframe_interpolation'/'2.5'`; `test_ltx2_video` 87 / 83 passed / 2593, exit 1.

Full gate at the same head: `CONFIGURE_EXIT=0`, `BUILD_EXIT=0`, `: error:` count
0, `ctest -N` **511**, `CTEST_EXIT=0`, `100% tests passed, 0 tests failed out of
511`, `No space left` and `BFD` each 0 against injected controls that returned 1.
Load average 14 to 36 across the run and 13 GiB free at its start, and none of
the four load-dependent gates (#618, #294, #1052, #428) went red.

The conditioning case's own `MESSAGE` line is the row in one measurement:

```text
video_tokens  keyframe 12  ti2vid 8  bare 8
```

`image_tokens` is 4 on both arms — the same image, the same encode, 8 + 4 on one
and 8 on the other. And the anchor case reports `keyframe: 4096 / 4096   res2s:
2 / 8`, the same split `ltx25-ti2vid-recipe.md` measured.

### The reachability mutation FAILED TO BUILD on its first shape, and that is the finding

Deleting the whole `keyframe_interpolation` arm from `ResolveLtx2PipelineRecipe`
left `KeyframeInterpolationRecipe` unreferenced in an anonymous namespace, and
`-Werror` killed the build: `BUILT=NO`, 1 error, **no test result at all**. A
harness that printed only a diff stat and an exit code would have shown a clean
26-line deletion and nothing else, which is the third shape of green-that-proves-
nothing this campaign has paid for.

The mutation that measures the thing is a rename of the dispatch KEY —
`"keyframe_interpolation"` to `"keyframe_interpolation__UNREACHABLE"`. Every line
stays compiled and referenced; the recipe simply stops being selectable through
the request surface, which is what "delete the production call site" is asking.
Under it all six new cases go RED **by name**, four in `test_ltx2_video` and two
in `test_ltx2_pipeline`.

Both runs are in the pull request body. The first is not a failed attempt to be
tidied away: it is the reason the harness prints four facts.

### The trace digest had to follow the tokens, and nothing would have said so

`im.trace.image_tokens` / `image_digest` / `image_absmax` are digested from the
tokens the conditioning item WROTE, and the replace arm reads them off the FRONT
of `video.clean`. Under the append arm the written tokens are at the TAIL.

Left as it was, the digest would have described unconditioned tokens on this
recipe and reported a healthy conditioning for a state the keyframe never
reached — which is precisely what the field's own comment in `ltx2_video.h` says
it exists to prevent, one arm down from where it was written. No test in this row
compares the digest's VALUE, so nothing here would have caught it; it is correct
because the range was derived, not because a gate held it. Recorded so the next
reader knows which half is measured.

### Two things a later reader should not re-derive

- **`causal_fix` is passed `true` at the frame-0 keyframe call and that is not a
  copy of the last-frame arm.** `VideoConditionByKeyframeIndex` gates it itself
  — `latent_tools.causal_fix if self.frame_idx == 0 else False`
  (`keyframe_cond.py:49`) — and `Ltx2ConditionVideoByKeyframe` mirrors that gate
  at `ltx2_conditioning.cpp:548`. So `frame_idx = 0` is the ONE combination in
  which the fix applies, and passing `false` would silently drop it rather than
  being refused.
- **`ltx2-gen` gained `--last-frame` (#1191) and the ABI did not change.**
  `vllm_video_params::last_frame` has existed since #930 and the engine has
  served it since; the CLI had simply never read the field. Nothing about this
  row required an ABI change, and a reader who sees the flag appear in the same
  commit should not infer one.

### A number this row moved that no checker owns

`docs/USAGE.md` says how many `(kind, version)` pairs resolve. It said
**twenty-four** and now says **twenty-eight**. It is derived by hand from
`ResolveLtx2PipelineRecipe` and nothing recomputes it, so the next four-key kind
has to move it again.

### The review repair, and the one defect it found

The fresh review returned eight findings. Six are repaired on this branch, one is
inherited and tree-wide, and one is filed above under `## Owed`.

**The blocking one is [#1219](https://github.com/mudler/vllm.cpp/issues/1219),
and it is worth stating as a shape rather than as a line number.** This row's own
`## Tests` §2 gated that frame 0 APPENDS on this builder, and every case it
describes pins ONE end. Pinning both — which is the pipeline's name, the
`docs/USAGE.md` worked example and the `ltx2-gen --help` instruction — aborted the
render, because the LANDED last-frame arm located its own appended tokens at
`positions[target_tokens * 2]`. That index is the first token past the target
grid: correct while that arm owned the first append, and false the moment this row
put a second appending item in front of it. The generated-keyframe-slot arm
carried the same derivation.

**Nothing in the diff touched either arm.** The row added an append two hundred
lines above them and changed what "the first appended token" meant. A derived
index that is right by ORDER survives every review of the code that contains it,
because the code that contains it is not the code that breaks it. What the case
this repair adds gates is the ORDER, on both builders, with `ti2vid_two_stage` —
replace plus one append — as the control that never reached the defect.

**A second measurement the repair owes the next reader**: `causal_fix = true` at
the first-frame call site is INERT, not merely ungated. The reason and the probe
numbers are under `## Owed`, and the gate that could exist lives on the seam in
`test_ltx2_vae` rather than at the call site.

## Dependencies

- #1118 (`LTX25-PHASE-LORA`), landed at `4ae0f54ab`. Supplies
  `Ltx2PhaseRecipe::loras` and `Ltx2RebindDitLoras`, without which stage 1 could
  not run unadapted.
- #1092 / #1102 (`LTX25-GUIDED-VIDEO`), landed at `daeff67f2`. Supplies
  `Ltx2GuidedDenoise`, without which stage 1's CFG has no seam.
- #1093 (`LTX25-TI2VID-RECIPE`), landed at `affc2a7fd`. Supplies
  `Ltx2PhaseRecipe::schedule_tokens` and `Ltx2PhaseScheduleTokens`, and is the
  reviewed template this row follows.
- #1117 (`LTX25-A2VID-RECIPE`), landed at `d1e5e9bc0`. Supplies
  `requires_distilled_lora` and its refusal, keyed on the flag rather than on the
  kind string.
- #1148 (`LTX25-BF16-DIT`), landed at `40a796aa9`. Makes the dev transformer
  loadable, which is what the owed run needs.
- #930 (`LTX25-TOKEN-APPEND`), landed at `c7cb59fbb`. Supplies the append seam —
  `Ltx2ExtendKeyframesMask` and `Ltx2ClearConditioning` — which the frame-0
  keyframe now drives at a second call site.

## Owed

- **An N-image request surface with per-image `frame_idx`, `strength` and
  `crf`**, [#1187](https://github.com/mudler/vllm.cpp/issues/1187). Upstream's
  `--image PATH FRAME_IDX STRENGTH [CRF]` is repeatable (`utils/args.py:805-817`)
  and expands per keyframe (`helpers.py:343-367`); this ABI carries two scalar
  slots at fixed indices 0 and `frames - 1`. Every pipeline that takes `images`
  is narrowed by it, which is why it is filed against the request surface rather
  than against this row. This row serves the two-endpoint case, which is
  keyframe interpolation's default configuration.
- **The sigma-BINNED guider factory**, `MultiModalGuiderFactory.from_dict`
  (`guiders.py:317-330`) resolved per step by `FactoryGuidedDenoiser`
  (`denoisers.py:332-343`). No CLI flag builds one, so no upstream default path
  reaches it — but a caller who passes a factory as `video_guider_params` does.
  Owed under [#1187](https://github.com/mudler/vllm.cpp/issues/1187) as the
  second half of the same request-surface gap: this engine's guider is one struct
  per phase and there is no sigma-keyed spelling to fill.
- **The real-weights comparison against upstream's own render** on the dev
  transformer, same checkpoint, prompt and seed. #644 owns the standing "close
  every refused arm" sweep and every LTX arm shipped to date carries the same
  debt.
- **Per-phase adapter STRENGTH**, [#1144](https://github.com/mudler/vllm.cpp/issues/1144).
  Not needed here: upstream gives this pipeline `loras=tuple(loras)` against
  `(*tuple(loras), *tuple(distilled_lora))`, i.e. ABSENT vs PRESENT, which
  `Ltx2PhaseLoraScope` expresses exactly.
- **The other three divergent arms on the schedule anchor**, owned by
  [#1150](https://github.com/mudler/vllm.cpp/issues/1150). This row sets its own
  phase and touches none of them.
- **A trace-derived step count in the two schedule-anchor cases**,
  [#1220](https://github.com/mudler/vllm.cpp/issues/1220). Both cases — the
  `ti2vid` one and the copy this row made of it — return `gen.steps` from their
  render lambda under a comment that says the count is read back out of the
  render, so the four `REQUIRE(x.steps == rendered_steps)` lines compare four
  copies of one request field. The guard is weakened rather than vacuous: it
  still reds when the pinned `gen.steps = 3` is lowered, which is the mutation it
  was built for. `Ltx2ConditioningTrace::dit_evaluations` already carries the
  observation. Not repaired in this row's review-repair flow because it changes
  what a LANDED case measures on both pipelines and the res_2s control, so it
  owes its own red-first evidence rather than a quiet re-derivation of a passing
  assertion.
- **The `causal_fix` argument at the first-frame call site is INERT**, and the
  gate for it lives on the seam rather than at the call site. MEASURED on a probe
  of `Ltx2ConditionVideoByKeyframe`: at `num_pixel_frames = 1`, which both
  production arms pass, flipping `causal_fix` moves 0 of 48 position values at
  `frame_idx` 0 and at `frame_idx` 8 alike, because `get_pixel_coords` leaves a
  one-latent-frame keyframe's temporal START at 0 either way
  (`max(0 + 1 - time, 0)`, patchifiers.py:166-169) and the `num_pixel_frames == 1`
  narrow at `keyframe_cond.py:56-57` then overwrites the END the fix had moved.
  So no call-site check can detect a flip, and one that appeared to would be a
  false gate. The `frame_idx == 0` gate the argument passes through
  (`keyframe_cond.py:49`) is what carries the risk, and it is gated in
  `test_ltx2_vae` at `num_pixel_frames != 1`, where the fix shows: 4 of 48
  differing at `frame_idx` 0 and 0 of 48 at `frame_idx` 8. The pre-existing
  negative-half assertion beside it (`no_fix.positions == state.positions` at
  `frame_idx` 5, `num_pixel_frames = 1`) holds VACUOUSLY and is not this row's to
  rewrite.

- **The tree-wide correction of two off-by-N upstream anchors**,
  [#1230](https://github.com/mudler/vllm.cpp/issues/1230). Re-derived at the
  LTX-2 pin `fd4ded7f` by reading the pinned files rather than inheriting the
  citation: `latent_cond.py:38` is `latent_state = latent_state.clone()` and
  `:39` is blank, so the two writes are `:40-41`; `schedulers.py:31` is the
  return annotation `) -> torch.FloatTensor:`, so the
  `tokens = math.prod(latent.shape[2:])` read is `:32`. Twenty-two citations
  carry the stale form and eight already carry the corrected one, inside the
  same files: `ltx2_video.cpp` cites `schedulers.py:31` at `:3618` and `:32` at
  `:3017` and `:3600`. This row's review repair corrected the seven its own new
  lines restated and those corrections were then REVERTED, because a PARTIAL
  correction is strictly worse than none. It left `ltx2_video.cpp` reading
  `latent_cond.py:38-39` at `:2208` and `:3174` and `:40-41` a hundred lines
  later with nothing recording which a reader should believe, and a uniform
  wrong anchor is correctable by one grep while a mixed one is not. One of the
  seven also lived in `include/vllm/`, a `USER_USAGE_PREFIXES` path in
  `scripts/check-doc-checkpoint.py:99` -- a pure path match with no content
  analysis -- so a comment-only anchor edit in a public header demanded a
  `docs/USAGE.md` edit this change did not owe, and manufacturing one to turn a
  gate green is what AGENTS.md forbids. Both effects are properties of doing it
  piecemeal. The records this row writes state the anchors CORRECTLY -- the
  table at `:97` and `:129` above, and #1219's index row -- so the record says
  what is true while the source stays uniformly stale until #1230 sweeps it in
  one commit with its own reviewer.

## Stop conditions

Return `NEEDS_DECISION` rather than narrowing scope if the conditioning-builder
field cannot be added without moving a landed arm, or if serving frame 0 through
the keyframe item turns out to need a second latent-state seam this engine does
not have.
