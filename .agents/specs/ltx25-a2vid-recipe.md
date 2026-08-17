# LTX25-A2VID-RECIPE — the audio-to-video recipe, and the take it has to consume

Row `LTX25-A2VID-RECIPE`, under the campaign [`ltx-2-5.md`](ltx-2-5.md).
Issue [#1117](https://github.com/mudler/vllm.cpp/issues/1117).
Base: `origin/main` @ `daeff67f2`.
Upstream: Lightricks `LTX-2` @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, verified
with `git rev-parse HEAD` in `/home/mudler/_git/LTX-2` before any anchor below was
taken. Paths are relative to `packages/ltx-pipelines/src/ltx_pipelines/` and
`packages/ltx-core/src/ltx_core/`, as every other LTX-2.5 spec here uses them.

vLLM implements nothing in this class and neither does vLLM-Omni, whose recipe
table stops at the two-stage distilled row. Lightricks is the reference, and it
is the only one.

## 0. Honesty statement — what this row claims and what it does not

It claims: `pipeline_kind = a2vid_two_stage` resolves a recipe that mirrors
`A2VidPipelineTwoStage.__call__` phase for phase; that the recipe is reachable
from `include/vllm.h` and from `ltx2-gen` on its own default configuration; that
an a2vid render without a driving take is refused rather than rendered
unconditioned; and that the gate can see whether the frozen take was CONSUMED by
the DiT rather than merely carried in the recipe.

It does not claim a render on real weights. The GPU is out of bounds for this
row, `dgx.casa` is not answering ssh, and every number below is measured on this
tree's own reduced fixture. It does not claim an oracle-run comparison: no
LTX-2.5 checkpoint here has a recorded sha256 (#1048) and the pinned secondary
oracle for this class is `vllm-omni`, which is UNPINNED (#633) and carries no
LTX-2.5 recipe at all. Every value below is read off upstream SOURCE at
`fd4ded7f`.

It does not claim upstream's adapter placement. See §4.4 and
[#1118](https://github.com/mudler/vllm.cpp/issues/1118).

## 1. The gap, verified on this tree at `daeff67f2`

`git grep -n '"a2vid' -- src include tests docs examples` returns TWO hits, both
upstream anchors inside `Fail`-message assertions
(`tests/vllm/multimodal/test_ltx2_video.cpp:4363`, `:4427`). The control is
`git grep -c '"one_stage"' -- src include`, which returns 4 in the header and 1
in the recipe table. So the pipeline kind genuinely does not exist, and the grep
is not failing on the wrong term.

Issue [#922](https://github.com/mudler/vllm.cpp/issues/922) is CLOSED and closed
the **conditioning**, not the recipe. What it left is written down in its own
spec ([`ltx25-a2v-audio-input.md`](ltx25-a2v-audio-input.md):438-446): "This row
ports the audio-conditioning mechanism and rides the checkpoint's own resolved
recipe", which in practice is `distilled_two_stage`, "and no claim is made that a
render here reproduces upstream's A2Vid output."

The blocker that entry named is gone. `Ltx2GuidedDenoise` landed at `daeff67f2`
(#1092/#1102) with production callers in `src/vllm/multimodal/ltx2_video.cpp`,
and [`ltx25-guided-video.md`](ltx25-guided-video.md) `## Owed` records
`a2vid_two_stage` as "its own row; none is blocked on this seam any more".

**Two anchors in the dispatch that sent this row were wrong and are corrected
here rather than propagated.** The a2v guidance default is NOT 0.0 and is not at
`a2vid_two_stage.py:318-323`; `:318-323` is `--audio-start-time`, whose default
is 0.0. `--a2v-guidance-scale` lives at `utils/args.py:986-995` — the
`parser.add_argument(` call opens at `:986`, the flag is named at `:987` and the
default is set at `:989` — and it defaults to
`video_guider.modality_scale`, which is **3.0** (`utils/constants.py:54`,
`:64`). And "audio conditioning built at `:53`, called at `:143`" names the class
statement and the `__call__` signature; the `AudioConditioner` is constructed at
`:96-102` and called at `:200`.

## 2. The port map, phase by phase

Upstream's two stages, and the `Ltx2PhaseRecipe` each becomes.

| Field | stage 1 | stage 2 | Upstream |
|---|---|---|---|
| `name` | `stage_1` | `stage_2` | the pipeline's own attribute names, `:103` and `:115` |
| `spatial_downscale` | 2 | 1 | `width // 2, height // 2` at `:206-212`; full shape at `:264` |
| `sigmas` | empty (derived) | `Stage2DistilledSigmas()` | `self._scheduler.execute(steps=num_inference_steps)` at `:225-227`; `stage_2_sigmas=STAGE_2_DISTILLED_SIGMAS` at `:164` |
| `use_official_sigma_schedule` | true | false | as above |
| `noise_scale` | 1.0 | 0.909375 | `ModalitySpec.noise_scale` default (`utils/types.py:110`) since `:247-250` sets none; `stage_2_sigmas[0].item()` at `:288` |
| `input_transform` | `kInitial` | `kSpatialUpsample` | `self.upsampler(video_state.latent[:1])` at `:261` |
| `video_guidance` | the params table's video row | defaults | `MultiModalGuider(params=video_guider_params, negative_context=v_context_n)` at `:233-236`, fed from `utils/args.py:947-1006` (the six video-guider flags, `--video-cfg-guidance-scale` at `:948` through `--video-skip-step` at `:997`; the audio group starts at `:1007`); `SimpleDenoiser(v_context_p, a_context_p)` at `:278` |
| `audio_guidance` | defaults | defaults | `MultiModalGuider(params=MultiModalGuiderParams())` at `:237-239` |
| `allow_guidance_override` | true | false | the CLI passes six video guider fields at `:353-360`; stage 2 takes no guider at all |
| `stepper` | `kEuler` | `kEuler` | `:229-258` and `:277-297` pass no `stepper`, so `EulerDiffusionStep()` applies (`utils/blocks.py:526-527`) |

**The audio guider on stage 1 is the DEFAULT one and not the params table's audio
row.** `MultiModalGuiderParams()` at `:237-239` is cfg 1.0 / stg 0.0 / rescale
0.0 / modality 1.0 / stg_blocks `[]` (`ltx-core components/guiders.py:200-210`),
which is `_POSITIVE_ONLY_GUIDER`. The params table's audio row carries cfg 7.0
(`utils/constants.py:61`). Taking the table's row here — which is what
`OneStagePhase` does, correctly, for `ti2vid_one_stage.py:215-218` — would run an
unconditional audio forward on a stream that is frozen, and would spend a text
encode on a negative context the audio guider never asked for. It renders either
way.

**Stage 1 is plain Euler and this is the one place a neighbouring recipe is
actively misleading.** `DistilledTwoStageRecipe` selects `kEulerAncestral` for
stage 1 on generation 2.5, and that comes from `distilled.py:76-84`, which
reaches a2vid through nothing: `a2vid_two_stage.py` never imports
`should_use_ancestral_sampler` and never passes a `stepper`. Selecting it here
because the stage is "the two-stage first stage" would be inferring a sampler
from a neighbour.

Recipe-level fields:

| Field | Value | Upstream |
|---|---|---|
| `height` / `width` | `params.stage_2_*` | `parser.set_defaults(height=params.stage_2_height, ...)` at `utils/args.py:1128` |
| `num_inference_steps` | `params.num_inference_steps` | `:152`, consumed at `:226` |
| `negative_prompt` | `kOmniNegativePrompt` on the `2` and `2.3` rows, `LightricksNegativePrompt()` on `2.4` and `2.5` | `--negative-prompt` default `DEFAULT_NEGATIVE_PROMPT` (`utils/args.py:937-946`), consumed at `:176-183`. Which string is a question of WHICH REFERENCE owns the row, per the header's "which upstream owns which value": vLLM-Omni supplies the pre-2.4 rows and Lightricks the 2.4 and 2.5 ones. `t2a_one_stage` splits the same way at the same four versions (its arm of `ResolveLtx2PipelineRecipe`), and these rows mirror it one for one |
| `allow_negative_prompt` | true | `:146` is a parameter and `:183` reads `ctx_n` into the video guider |
| `allow_request_sigmas` | true | stage 1's schedule IS `num_inference_steps` (`:226`); stage 2's is a constant and carries its own explicit `sigmas`, which the engine reads before the override branch |
| `fixed_num_inference_steps` | false | as above |
| `video_output_phase` / `audio_output_phase` | 1 | `:299` decodes `video_state.latent` after stage 2 |
| `allow_request_latents` | false | `:229-297` construct every `ModalitySpec` from pipeline state; no request latent surface exists |
| `requires_audio_input` | true | `--audio-path` `required=True` at `:312-317` |
| `requires_distilled_lora` | true | `--distilled-lora` `required=True` at `utils/args.py:1140-1153` |

**Which versions.** All four this table KEYS — `2`, `2.3`, `2.4`, `2.5` —
mirroring the `t2a_one_stage` rows and for the same reason:
`A2VidPipelineTwoStage` takes whatever `resolve_cli_params()` read off the
checkpoint (`:311`), exactly as `T2AOneStagePipeline` does at
`t2a_one_stage.py:178-179`. There is no "which generations support A2V" question
upstream, and restricting the rows would be a local invention. This differs from
`distilled_two_stage`'s two rows, which are two rows because two DIFFERENT
references supply them.

Four KEYS is not four params objects, and the earlier wording — "all four the
params table distinguishes" — claimed the second. `_PARAMS_SINCE_VERSION`
(`utils/constants.py:130-133`) carries exactly TWO rows, `(2,4)` and `(2,3)`,
with `LTX_2_PARAMS` as the fall-through at `:179`. So 2.5 does not have a params
row of its own: it is at or above `(2,4)` and resolves onto the 2.4 one.
`Ltx2DetectPipelineParams` (`ltx2_pipeline.cpp:947-956`) mirrors that shape
exactly and its own comment already says so — "this is what gives LTX-2.5 the 2.4
params". The four keys exist because the RECIPE table refuses an unknown
`(kind, version)` by name rather than defaulting, not because upstream reads four
different parameter sets.

## 3. What already exists and is reused unchanged

Nothing in the denoise loop moves. The row is a recipe row plus two refusals.

| Piece | Where | Reached by this row how |
|---|---|---|
| decode → encode → truncate the take | `Ltx2DecodeAudioWav` / `Ltx2EncodeAudioToLatent`, `ltx2_video.cpp:2497-2516` | the `audio_path` extra, now REQUIRED on this kind |
| the freeze, both halves | the zeroed denoise mask at `ltx2_video.cpp:2994` and the scalar `ain.sigma` at `:3414` | unchanged; §5.2 gates that this arm consumes it |
| the guided seam | `Ltx2GuidedDenoise`, `ltx2_video.cpp:3483` | stage 1's guider is the params table's video row, so all four passes run |
| the spatial upsample | `Ltx2PhaseInputTransform::kSpatialUpsample`, `ltx2_video.cpp:2739` | stage 2 |
| the caller's own waveform as the soundtrack | `ltx2_video.cpp:3853` | keyed on the take being present, which this kind now guarantees |
| the resolution guard's divisor | `max_spatial_downscale()` | stage 1's downscale of 2 makes it 64, which is `assert_resolution(is_two_stage=True)` (`:168`) |

`phase.noise_scale` is applied to both streams at `ltx2_video.cpp:3244-3248` and
that is NOT a divergence from upstream's per-modality `noise_scale=0.0`:
`ApplyGaussianNoise` is masked, and a frozen stream's mask is all zeros, so the
audio latent stays at `clean` whatever the phase's scale is. Stage 2's 0.909375
therefore cannot reach the frozen take. Checked rather than assumed — §5.2's
digest is taken after the noiser, on both phases.

## 4. Design

### 4.1 Two recipe flags, not two string compares

`Ltx2PipelineRecipe` grows `requires_audio_input` and
`requires_distilled_lora`. Both are flags on the recipe for the reason
`audio_only` already gives in the header: the engine has to answer the question
in a place that is not the recipe table, and an `im.pipeline_kind ==
"a2vid_two_stage"` test at each site is one chance per site to miss the next
recipe that needs it. `ti2vid_two_stages` (#1093) and `keyframe_interpolation`
(#1096) both come with `--distilled-lora required=True`, so the second flag has a
second user before it lands.

### 4.2 The audio refusal

`requires_audio_input` is checked in `Generate`, beside the `audio_path` read,
because `pipeline_kind` is a LOAD knob and `audio_path` is a per-generation
extra: the question is only decidable once a request exists. The message names
the extra, the upstream line that makes it required, and what happens without it
— an unconditioned render that returns a clip of the right size, frame count and
sample rate, which is the shape of defect this file keeps finding.

### 4.3 The LoRA refusal

`requires_distilled_lora` is checked at LOAD, where `lora_path` is read
(`ltx2_video.cpp:808-820`), because both are load-time. Upstream cannot run this
pipeline without the adapter; neither can this. Refusing there rather than at
generate time means a caller learns before paying for a 22B load.

The refusal deliberately does not try to verify that the supplied adapter IS the
distilled one. Upstream does not either: `--distilled-lora` takes any path.

### 4.4 What this cannot mirror, and why it is filed rather than commented

`stage_2_loras = (*loras, *distilled_lora)` at `:114` puts the distilled adapter
on stage 2 ALONE; stage 1 gets `loras=tuple(loras)` at `:107`. This engine fuses
at load into ONE weight set — `ltx2_video.cpp:816-820` is the only
`dit_options.loras.push_back` in the tree — so the adapter reaches both phases.
Stage 1's guided schedule therefore runs against base + distilled LoRA where
upstream runs it against the base alone — 30 steps of it on 2.5, since
`LTX_2_3_PARAMS` sets `num_inference_steps=30` (`utils/constants.py:85`) and 2.4
inherits it at `:124`, which is the row 2.5 resolves onto.

That divergence RENDERS, and the PIXELS it renders are not upstream's. It changes
the trajectory, so the frames themselves differ; what it leaves untouched is the
frame count, the shapes, the sample rate and the errors — nothing in the SHAPE of
the result says anything is wrong. It is therefore
[#1118](https://github.com/mudler/vllm.cpp/issues/1118) and `## Owed` rather than
a comment. It bounds #1093 and #921 the same way.

**Say what a gate could see, not that none could.** "Changes nothing a caller can
read" would be false and would be the more damaging kind of false, because it
implies no instrument could ever detect this. A caller reads pixels. The gate
that WOULD detect it is the real-weights comparison against upstream's own render
that §0 and `## Owed` record this row as not having: same checkpoint, same take,
same seed, upstream's stage 1 on the base weights against ours on base +
distilled. That comparison is owed, not impossible, and #1118 is the row that
owes it.

Refusing the whole kind instead was considered and rejected: it would land the
recipe dead, and the rule that forbids dead code is not satisfied by a capability
nobody can reach. Rendering with NO adapter was also rejected — a 3-step
distilled refinement on a checkpoint that was never distilled is the
plausible-and-wrong shape [`ltx25-a2v-audio-input.md`](ltx25-a2v-audio-input.md)
`## Owed` warned this row about by name.

### 4.5 What the implementation added that §4 did not foresee

`Ltx2PhaseDenoiser { kGuided, kSimple }`, a new phase field naming which of
upstream's two denoiser classes (`utils/denoisers.py`) a phase constructs. It
exists because **`allow_guidance_override` cannot express a2vid's stage 2**, and
that only became visible once the recipe was written.

That boolean answers "does this pipeline's CLI carry the guider flags at all".
`distilled.py` selects `default_2_stage_distilled_arg_parser`
(`utils/args.py:1188`; `:1187` is blank), which never adds them, so an override
there names a knob the pipeline has no surface for and the engine refuses it —
correctly, and that refusal is landed and gated. `a2vid_two_stage.py:311` selects
`default_2_stage_arg_parser` (`utils/args.py:1123`), which DOES carry them
(`utils/args.py:947-1006`),
and they reach stage 1's guider alone (`:233-236`) because stage 2 constructs
`SimpleDenoiser(v_context_p, a_context_p)` (`:278`) and takes no params at all.

Neither value of the boolean says that. `false` refuses a request upstream
accepts. `true` applies the override to stage 2's positive-only params and
switches on a guidance pass upstream's stage 2 does not run — invisibly, since an
extra forward changes no output shape, frame count or sample rate. So a2vid's
stage 2 is `allow_guidance_override = true` **and** `kSimple`, and
`ApplyGuidanceOverrides` skips a `kSimple` phase AFTER the refusal check. The
order matters: every existing recipe that refuses is also `kSimple`, so testing
the skip first would silently turn three landed refusals into silent ignores.

The field is populated on every phase in the table from the upstream line that
constructs the denoiser, and it is read in exactly one place.

**How it is gated, and why the obvious gate is vacuous.** The claim is that an
override reaches stage 1 and not stage 2, and no trace field records what the
second phase did. The instrument is a pair of renders whose only difference is a
value that is ALREADY stage 1's own: `video_stg_scale = 1.0` is what this
recipe's stage 1 carries (`utils/constants.py:52`), so applying it there changes
nothing, while stage 2's own STG scale is 0.0 and applying it THERE adds a
perturbed forward per step. Equal artifact bytes therefore mean the override
stopped at stage 1. The case `REQUIRE`s both of those recipe values first, so a
table change that made the restated value differ from stage 1's turns the
comparison into a failure rather than into a tautology.

## 5. Tests and gates

Focused: `ctest --test-dir build -R 'ltx2' --output-on-failure`, and the whole
`test_ltx2_video` / `test_ltx2_pipeline` binaries run with NO `--test-case`
filter, because a truncated filter matched zero cases and printed `SUCCESS!` with
exit 0 on a sibling row and many case names here contain commas. The case and
assertion counts are asserted non-zero.

Full:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Reported with `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the pass/fail line, positive controls for `No space left` and
`BFD assertion`, plus load and free disk. Known-red cited by the issue naming the
test, never by family: `windows-msvc-*` (#584), `test_async_llm` (#294),
`test_engine_core_proc` (#1052), `test_serve_low_tools` (#428),
`test_cpu_x86_llamacpp_floor` (#618).

### 5.1 RED first, through the production entry point

The smallest failing test loads an engine with
`pipeline_kind = a2vid_two_stage` and renders. It fails first because
`ResolveLtx2PipelineRecipe` refuses the pair by name. That is the intended
reason, and it is quoted in the pull request body.

### 5.2 The frozen take is CONSUMED, not carried

A recipe-level assertion proves the value is set. What this row asserts is that
the DiT saw it, on THIS arm, at every phase:

- `trace.audio_frozen`, derived from the denoise mask the loop uses and read
  AFTER the noiser (`ltx2_video.cpp:3261`) — the position that a mutation
  survived when it was read before;
- `trace.audio_sigma_max == 0.0`, the scalar `Modality.sigma` half of upstream's
  `frozen` (`utils/types.py:104-106`), which the mask cannot reach;
- `trace.audio_latent_digest` equal to the digest of the SAME render's take
  encoded once, and `audio_latent_absmax > 0` so a zeroed latent cannot pass a
  digest comparison against another zeroed latent.

Two controls, because one of them alone is passable by a constant:

- **the same take at a different SEED gives a bit-identical audio latent.** It is
  the encoded file, not a sample. A build that noised the audio stream, or that
  generated it and let the take decorate the trace, moves this digest and moves
  nothing a caller can see.
- **a different `audio_start_time` gives a different one.** Without this the first
  control passes on any constant tensor.

### 5.3 The guidance arms, per arm and in x0 space

The a2vid stage-1 guider is the params table's video row — cfg 3.0, stg 1.0,
rescale 0.7, modality 3.0 — so all four passes run and the rescale branch is
live. The per-arm invariant `x0 == latent - sigma*velocity` is asserted for
`cond`, `uncond`, `perturbed` and `modality` off `Ltx2ConditioningTrace`'s
recorded pass tensors. In velocity space the residual is the whole sample and the
RED prints `|x0 - velocity| = 0` exactly.

Non-vacuity is `REQUIRE`d: a zero step-0 sample makes the two candidate tensors
coincide.

### 5.4 The mutations this gate must survive

| # | Mutation | Must go RED at |
|---|---|---|
| M1 | the `a2vid_two_stage` dispatch row deleted (reachability) | the whole e2e case |
| M2 | stage 1's `spatial_downscale` set to 1 | the phase-shape assertions |
| M3 | stage 1 given `DistilledSigmas()` instead of the derived schedule | the schedule assertions |
| M4 | stage 1's `allow_guidance_override` set to false | the override case |
| M5 | stage 1's audio guider taken from the params table | the audio-guider assertion |
| M6 | stage 1's stepper set to `kEulerAncestral` | the stepper assertion |
| M7 | `requires_audio_input` never checked | the no-take refusal case |
| M8 | `requires_distilled_lora` never checked | the no-LoRA refusal case |
| M9 | the frozen take's denoise mask left at 1 on this arm | §5.2 `audio_frozen` |
| M10 | the frozen scalar sigma left at the schedule's | §5.2 `audio_sigma_max` |
| M11 | every guidance arm left in velocity space | §5.3, all four rows |
| M12 | the `kSimple` skip deleted, so an override reaches stage 2 | §4.5's artifact comparison |

Each mutation reports three facts: `git diff --stat`, whether it BUILT with the
compile-error count, and the exit code captured directly. A mutation that fails
to compile, and a mutation that never applied, both read exactly like a passing
test. The mutations are run against the COMMITTED head, so that first fact is
the mutation's own diff rather than the whole uncommitted change.

**Measured: twelve applied, twelve DETECTED, and one of them by one binary
only.** M6 — the stepper — is SURVIVED by the end-to-end case and DETECTED by the
recipe case (`test_ltx2_pipeline`, exit 1, 44 cases / 2598 assertions). That is
recorded as it measured rather than as it would read better: the stepper is a
recipe field, the end-to-end case has no baseline to compare a trajectory
against, and this recipe's `noise_seed_offset` is 0, so the ancestral arm moves
no digest the trace carries.

**One harness defect, found and fixed in flow.** §4.5's artifact comparison was
written as `CHECK(a == b)` over PPM pixels and a WAV. A failing one dumps raw
bytes into the doctest report, and that killed the harness with a
`UnicodeDecodeError` between applying M12 and restoring it — the shape
[`ltx25-a2v-audio-input.md`](ltx25-a2v-audio-input.md) §5 already records. The
`finally` restored the tree, the comparison is now a differing-byte COUNT, and
the harness decodes with `errors="replace"`.

## 5b. Reachability — the sentence the records must carry

Entry point: `vllm_video_engine_load` → `LoadVideoEngine` with
`pipeline_kind = a2vid_two_stage` (a documented value of a documented load
extra), then `vllm_video_generate` → `VideoEngine::Generate` with the
`audio_path` extra. The chain is
`include/vllm.h` → `src/capi/vllm_c.cpp` → `Ltx2VideoEngine::Generate` →
`ResolveLtx2PipelineRecipe`'s a2vid row → the phase loop → `Ltx2GuidedDenoise`.
`ltx2-gen` is the same call through `--pipeline-kind` and `--audio-path` as a
thin ABI client that includes no internal header.

The reachability mutation is M1: delete the `a2vid_two_stage` dispatch row so the
table refuses the pair, and rerun the focused gate.

**What does NOT reach it, stated rather than left to be found.** The OpenAI
`/v1/videos` route cannot drive this: `VideoGenParamsFromRequest`
(`src/vllm/multimodal/video_engine.cpp:349-384`) never writes `gen.extras`, so no
per-generation extra reaches any engine over HTTP
([#928](https://github.com/mudler/vllm.cpp/issues/928)). `pipeline_kind` is a
LOAD extra and IS reachable over `--video-extra`, so a server can be started on
this kind — and every such request would then be refused for the missing take,
which is the correct behaviour and not a way to use the route. The honest
statement is: reachable from `include/vllm.h` and `ltx2-gen`; NOT drivable over
`/v1/videos` until #928 is fixed.

## 6. Quantized arms

This row adds no GEMM, no kernel and no dtype. Arm by arm so none is left to be
discovered:

| Arm | Disposition |
|---|---|
| bf16 / f32 safetensors | ported; the shipped audio VAE and the fixture are this |
| NVFP4, FP8 on the DiT tower | unaffected. A recipe row selects sigmas, guiders and phase shapes; both arms reach it exactly as they reach `distilled_two_stage` |
| GGUF k-quants | not applicable and not merely undone. Upstream enumerates its inference quantization kinds exhaustively — fp8-cast, fp8-scaled-mm, nvfp4-cast, nvfp4-prequant (`utils/quantization_factory.py:23-26`, `assert_never` at `:50`) — so there is no upstream GGUF behaviour to mirror, and llama.cpp does not carry this architecture, so there is no quant-matched comparison to serve |
| int8-convrot | out of scope, already refused by name at `ltx2_pipeline.cpp`'s `kInt8ConvRot` |
| the device-resident arm | REFUSED by name for this kind's default guidance, and that refusal already exists: stage 1's `stg_scale = 1.0` and `modality_scale = 3.0` both need a perturbed forward, and `Ltx2DitForwardDevice` takes no `perturbations` (`ltx2_video.cpp:2598-2618`). Unchanged by this row and named here so it is not discovered later |

## 7. Risks

**R1 — a recipe that renders whatever it says.** Every field below is a number
that produces a finished clip whether it is right or wrong. Mitigated by
asserting the recipe's fields directly against the upstream anchors in §2, and by
mutations M2 to M6, which each move exactly one of them.

**R2 — the `READER ANCHORS` gate.** `ltx2_video.cpp:304-305` carries a
line-number list re-derived and string-compared by a case in
`test_ltx2_video.cpp`. Any line inserted above it shifts the list and a clean
`git merge` will not warn. Mitigated by re-deriving after the final merge of
`origin/main` and naming it as a merge hazard in the pull request body.

**R3 — concurrent edits.** `ltx2_video.cpp`, `ltx2_pipeline.{h,cpp}`,
`docs/FEATURES.md` and `.agents/issue-index.md` are edited by sibling rows.
`docs/BENCHMARKS.md` and `docs/FEATURES.md` sit at their prose-paragraph budgets
(#1055), so this row puts its content in TABLE ROWS and runs
`check-public-doc-tables.py` before pushing. The issue index is taken from
`origin/main` wholesale and re-appended on any conflict, never auto-merged.

**R4 — the adapter divergence in §4.4.** Named, filed as #1118, and listed under
`## Owed`.

## 8. Stop conditions

Report `NEEDS_DECISION` rather than narrowing silently if:

- the fixture cannot supply a LoRA, because then `requires_distilled_lora` makes
  the kind unreachable in-tree and the row becomes seam-only;
- stage 1's derived schedule cannot be exercised on the fixture without the
  guided seam refusing (the fixture DiT has two blocks and the params row names
  block 28 — `LTX_2_3_PARAMS` overrides 2.0's `[29]` to `[28]` at
  `utils/constants.py:86`, and 2.4, which 2.5 resolves onto, inherits it at
  `:124`), because then the guided arm of this recipe is gated by nothing.

## Owed

- **[#1118](https://github.com/mudler/vllm.cpp/issues/1118) — the per-phase
  adapter.** §4.4. Owned by this row. It also bounds
  [#1093](https://github.com/mudler/vllm.cpp/issues/1093) and
  [#921](https://github.com/mudler/vllm.cpp/issues/921).
- **A real-checkpoint A2V render.** Gated on fixtures only. The artifacts exist
  now — `/usr/local/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/` holds the
  NVFP4 DiT, both video VAEs, the audio VAE, both upscalers and
  `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` — and the host does
  not: `dgx.casa` is network-alive and its sshd will not handshake.
- **An oracle-run comparison.** `vllm-omni` is UNPINNED (#633) and carries no
  LTX-2.5 recipe; no LTX-2.5 checkpoint here has a recorded sha256 (#1048). The
  recipe is gated against upstream SOURCE, not upstream OUTPUT, and that is the
  ceiling on this row's evidence.
- **`/v1/videos` cannot drive this** until
  [#928](https://github.com/mudler/vllm.cpp/issues/928) forwards per-generation
  extras. §5b.

## Now

`ACTIVE` — spec committed before implementation, per `AGENTS.md` § *Spec before
code*.
