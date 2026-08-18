# LTX25-TI2VID-RECIPE — the plain two-stage pipeline, and the schedule anchor it exposed

Row `LTX25-TI2VID-RECIPE`. Issue
[#1093](https://github.com/mudler/vllm.cpp/issues/1093). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Upstream pin: Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified with `git rev-parse HEAD` in `/home/mudler/_git/LTX-2` on 2026-08-17.

Base: `c83b96934`, pinned when the worktree was created.

## Now

`ACTIVE` -> `DONE` with this change. `TI2VidTwoStagesPipeline`
(`ti2vid_two_stages.py:61`) becomes `pipeline_kind = "ti2vid_two_stage"` on the
four generations the recipe table keys.

## Scope

**In.** One recipe row, `Ti2VidTwoStageRecipe`, and its four `(kind, version)`
keys. One new per-phase field, `Ltx2PhaseRecipe::schedule_tokens`, because this
pipeline is the first arm whose upstream schedule anchor this engine cannot
already express. The docs and the CLI help that list the kinds.

**Out.** The three arms #1150 leaves divergent on that same anchor — see
`## Owed`. Per-phase adapter STRENGTH (#1144). The real-weights render, which
needs a GPU lease — see `## What is NOT verified`. `ti2vid_two_stages_mgpu.py`,
which is `kMultiGpuParallelism`.

## Why it is not a recipe we already ship

`ti2vid_two_stages.py` sits between two arms that do ship, and the three-way
diff is what makes it its own row rather than a version key on either.

| | `distilled_two_stage` (`distilled.py`) | **this row** | `res2s_two_stage` (`ti2vid_two_stages_hq.py`) |
|---|---|---|---|
| stages built | ONE, reused (`distilled.py:131`) | TWO (`:136`, `:147`) | TWO (`:151`, `:162`) |
| stage-1 denoiser | `SimpleDenoiser` (`:265-266`) | `FactoryGuidedDenoiser` (`:248-259`) | `GuidedDenoiser` (`:271-281`) |
| stage-1 sigmas | frozen `DISTILLED_SIGMAS` (`:200`) | derived (`:243-245`) | derived (`:260-267`) |
| stage-1 adapter | the one stage set | NONE (`:140`) | distilled @ 0.25 (`:92-96`, `:154`) |
| stage-2 adapter | the one stage set | distilled (`:151`) | distilled @ 0.5 (`:97-101`, `:165`) |
| stepper | ancestral on 2.5 (`:76-84`) | Euler (derived, below) | `Res2sDiffusionStep` (`:258`) |
| schedule anchor | n/a, frozen | **4096** (`execute(steps=)`) | target latent (`latent=empty_latent`) |

The row is therefore three fields away from `A2VidTwoStageRecipe` and four from
`Res2sTwoStageRecipe`, and every one of those differences renders.

## Port map, each line read at the pin

Anchors are `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`
unless another file is named.

### Stage 1 — `stage_1`

| Field | Value | Upstream |
|---|---|---|
| `spatial_downscale` | 2 | `:223-229`, `width // 2` / `height // 2` |
| `sigmas` | empty (derived) | `:243-245`, `self._scheduler.execute(steps=num_inference_steps)` |
| `schedule_tokens` | `kSchedulerDefault` | the same call passes NO latent; `schedulers.py:31` |
| `noise_scale` | 1.0 | `:266-267` sets none; `ModalitySpec.noise_scale` defaults 1.0 (`utils/types.py:110`) |
| `video_guidance` | `params.video_guider` | `:251-254`, from `MultiModalGuiderParams` the CLI fills (`:343-350`) |
| `audio_guidance` | `params.audio_guider` | `:255-258`, filled from `--audio-*` / `--v2a-guidance-scale` (`:351-358`) |
| `denoiser` | `kGuided` | `:248`, `FactoryGuidedDenoiser` |
| `allow_guidance_override` | true | `:319` selects `default_2_stage_arg_parser`, which carries the six guider flags (`utils/args.py:947-1006`) |
| `loras` | `kNoAdapters` | `:140`, `loras=tuple(loras)` against `:151` |
| `stepper` | `kEuler` | derived, below |

**The audio guider is the params table's row here, and on `a2vid_two_stage` it
is NOT.** That is not an inconsistency between the two rows; it is the
difference between the two pipelines. A2Vid's audio stream is the caller's
frozen take, so it builds `MultiModalGuiderParams()`
(`a2vid_two_stage.py:237-239`); this pipeline GENERATES its soundtrack and
`:255-258` hands the audio guider factory the real params, which
`main()` fills from six `--audio-*` flags at `:351-358`. Copying a2vid's line
here would silently drop audio CFG 7.0 on a stream that is being sampled.

### Stage 2 — `stage_2`

| Field | Value | Upstream |
|---|---|---|
| `sigmas` | `Stage2DistilledSigmas()` | `:178`, `stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS` — a DEFAULT ARGUMENT, so frozen |
| `use_official_sigma_schedule` | false | the schedule is explicit |
| `noise_scale` | `Stage2DistilledSigmas().front()` | `:300` and `:305`, `stage_2_sigmas[0].item()` on BOTH modality specs |
| `input_transform` | `kSpatialUpsample` | `:272`, `self.upsampler(video_state.latent[:1])` |
| `denoiser` | `kSimple` | `:290`, `SimpleDenoiser(v_context_p, a_context_p)` — takes no params |
| `allow_guidance_override` | true | same argument as `a2vid_two_stage.py`: the flags are legal on this parser and simply reach stage 1's guider alone. `kSimple` is what makes them inert here |
| `loras` | `kAllAdapters` (default) | `:151`, `(*tuple(loras), *distilled_lora)` |
| `stepper` | `kEuler` | derived, below |

`STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]`
(`utils/constants.py:19-20`), already ported as `Stage2DistilledSigmas()`.

### Recipe

| Field | Value | Upstream |
|---|---|---|
| `height` / `width` | `params.stage_2_*()` | `:319` sets the request geometry to the FINAL output (`utils/args.py:1128`) |
| `negative_prompt` | per version, below | `:194-202` encodes `[prompt, negative_prompt]` and reads `ctx_n` into both guider factories (`:253`, `:257`) |
| `video_output_phase` | 1 | `:310` decodes the name `:289` rebound |
| `audio_output_phase` | **0** | `:289` is `video_state, _ = self.stage_2(...)` — the audio is DISCARDED — and `:311` decodes the `audio_state` that `:247` bound |
| `allow_request_sigmas` | true | `:177` `stage_1_sigmas` is a real parameter and `:244` honours it |
| `allow_request_latents` | false | no `__call__` parameter carries one (`:159-181`); stage 1 has no `initial_latent` and stage 2's is the upsampler's output |
| `allow_negative_prompt` | true | `:162` |
| `requires_distilled_lora` | **true** | `--distilled-lora` is `required=True` (`utils/args.py:1140-1155`) on the parser `:319` selects |
| `requires_audio_input` | false (default) | there is no `--audio-path`; the soundtrack is generated |
| `audio_only` | false (default) | `:310-311` decodes both |

**`audio_output_phase = 0` is the field most likely to be "fixed" to 1**, and
`:287-288` is upstream's own comment saying why not: "Stage 2 refines video
only; discard its audio." Writing 1 would decode a soundtrack that is finite,
the right length, at the right sample rate, and the wrong take. `res2s_two_stage`
already carries 0 for the identical reason (`ltx2_pipeline.cpp:1468`).

### `stepper = kEuler` is derived, not assumed

Neither `self.stage_1(...)` (`:247-269`) nor `self.stage_2(...)` (`:289-308`)
passes `stepper` or `loop`. `DiffusionStage.__call__` declares both as
`None` defaults (`utils/blocks.py:512-513`) and fills them at `:524-527` with
`euler_denoising_loop` and `EulerDiffusionStep()`.

**Do not inherit `distilled.py`'s 2.5 ancestral selection.** That is
`distilled.py:76-84`, inside `DistilledPipeline`, and nothing routes it here —
this pipeline never constructs a stepper at all. `retake` and `a2vid_two_stage`
already carry `kEuler` on the same argument.

### `FactoryGuidedDenoiser` vs `GuidedDenoiser` is a no-op on the default path

`main()` passes plain `MultiModalGuiderParams` (`:343-358`), never a factory, so
`create_multimodal_guider_factory` returns a constant factory and both classes
reduce to `_guided_denoise` (`utils/denoisers.py:61-211`). Our `kGuided` reaches
`Ltx2GuidedDenoise`, which IS that function. Recorded because the class names
differ and the difference is not behavioural.

### Four version keys

`2`, `2.3`, `2.4`, `2.5`, mirroring `a2vid_two_stage` and `t2a_one_stage` line
for line. `main()` calls `resolve_cli_params()` (`:318`) and hands the result to
`default_2_stage_arg_parser(params=params, ...)` (`:319`) — the same two calls
`a2vid_two_stage.py:310-311` makes — so the generation comes off the checkpoint
and there is no "which generations support this pipeline" question upstream.
Restricting the rows would be a local invention.

The 2 and 2.3 rows carry `kOmniNegativePrompt` and the 2.4 / 2.5 rows carry
`LightricksNegativePrompt()`, which is the split every four-key kind already
uses: the prompt travels with the GENERATION, not with the pipeline.

## The divergence this row resolves

`ti2vid_two_stages.py:243-245` calls `execute(steps=num_inference_steps)` with
**no latent**, and `schedulers.py:31` reads that as
`default_number_of_tokens` = `MAX_SHIFT_ANCHOR` = **4096** (`:11`, `:29`).
This engine derives the shift from `target_tokens` on every phase
(`ltx2_video.cpp:3442-3443`). So a faithful `ti2vid_two_stage` cannot be written
with the fields that exist.

**The population is seven call sites and the split is six to one**, which
inverts how #1093 and `ltx25-res2s-loop.md:80-88` both framed it.
`grep -rn "\.execute(" packages/ltx-pipelines/src/ltx_pipelines/` at the pin:

| call site | latent | our arm | correct today |
|---|---|---|---|
| `ti2vid_one_stage.py:207` | no | `one_stage` x4 | **no** |
| `t2a_one_stage.py:141` | no | `t2a_one_stage` | yes — passes 0 at `ltx2_t2a.cpp:178` |
| `retake.py:287` | no | `retake`, non-distilled arm | **no** |
| `a2vid_two_stage.py:226` | no | `a2vid_two_stage` stage 1 | **no** |
| `ti2vid_two_stages.py:244` | no | **this row** | this row makes it yes |
| `keyframe_interpolation.py:200` | no | `keyframe_interpolation` stage 1 | yes, since row LTX25-KEYFRAME-INTERP (#1096) |
| `ti2vid_two_stages_hq.py:267` | **yes** | `res2s_two_stage` stage 1 | yes |

So the engine mirrors the exception and diverges from the rule. That is #1150,
filed with the arithmetic; at the recipe default geometry the target latent is
6144 tokens and `sigma_shift` is 2.78 against upstream's 2.05, so every sigma
moves.

### The resolution, and why the default does not flip

`Ltx2PhaseRecipe` gains
`Ltx2PhaseScheduleTokens schedule_tokens = kTargetLatent`:

- `kTargetLatent` — `math.prod(latent.shape[2:])` of this phase's target grid.
  `ti2vid_two_stages_hq.py:267`'s `latent=empty_latent`. **The DEFAULT.**
- `kSchedulerDefault` — `default_number_of_tokens`, i.e. 4096. The six sites
  that pass no latent.

**The default is today's behaviour, so this row moves exactly one phase.** The
upstream-faithful default would be `kSchedulerDefault`, and it is rejected here
on blast radius, not on principle: flipping it re-samples `one_stage` at four
version keys, `a2vid_two_stage` stage 1 and `retake`, all of them shipped and
gated, and rewrites their goldens. A row scoped to add one recipe must not move
five others on a finding made inside it. #1150 owns the flip, and this field is
the seam it will use — three assignments plus goldens.

**A preserving default also cannot fail silently.** With `kTargetLatent`
default, an arm moves only where a line says so. With the flip, an arm I failed
to pin would move with nothing naming it — and the whole class of defect this
campaign gates for is the one that still renders.

The engine reads it in one place, `ltx2_video.cpp:3442`, and resolves it to a
CONCRETE token count rather than passing 0:

```cpp
const int64_t schedule_tokens =
    phase.schedule_tokens == Ltx2PhaseScheduleTokens::kSchedulerDefault
        ? Ltx2SchedulerParams{}.default_number_of_tokens
        : target_tokens;
```

`Ltx2SigmaSchedule(steps, 4096)` and `(steps, 0)` are identical by
`ltx2_pipeline.cpp:110`, and the concrete form keeps the existing "ONE local
feeds both the schedule and the trace" property — `im.trace.schedule_tokens`
then reports 4096 rather than 0, which is what makes the gate below an equality
rather than a sentinel check.

## Tests

Upstream ships no unit test over a pipeline's recipe — the recipe IS the
constructor — so these are ported in the sense that every assertion cites the
line it mirrors, and the harness is ours.

### 1. The recipe, field by field (`test_ltx2_pipeline.cpp`)

Mirrors `"ltx2 a2vid: the recipe is upstream's TWO stages, not the distilled
one"` (`:3335`). Every field in the Port map above, each asserted against a
CONTROL drawn from the recipe it would otherwise be confused with, so no
assertion can pass by two values coinciding:

- stage 1 `loras == kNoAdapters`, control `res2s_two_stage` stage 1 at
  `kAllAdapters`;
- stage 1 `sigmas` empty and stage 2's == `Stage2DistilledSigmas()`, control
  `distilled_two_stage` stage 1 non-empty;
- `stepper == kEuler` on both phases, control `distilled_two_stage` at 2.5 on
  `kEulerAncestral` and `res2s_two_stage` on `kRes2s`;
- `audio_output_phase == 0` with `video_output_phase == 1`, control
  `a2vid_two_stage` at 1/1;
- stage 1 `audio_guidance.cfg_scale` == the params table's, control
  `a2vid_two_stage` stage 1 at the positive-only default. This is the field the
  Port map flags as most likely to be copied wrongly;
- `requires_distilled_lora` true, `requires_audio_input` FALSE, control
  `a2vid_two_stage` true/true;
- `schedule_tokens == kSchedulerDefault` on stage 1, control `res2s_two_stage`
  stage 1 at `kTargetLatent`.

Plus a version-key case mirroring `:3483`: all four resolve, and
`{"ti2vid_two_stages", "2.5"}` (upstream's PLURAL file name) and
`{"ti2vid_two_stage", "2.6"}` refuse by name.

### 2. The schedule anchor REACHES the sigmas (`test_ltx2_video.cpp`)

Through `LoadVideoEngine` + `Generate`, not the recipe struct: the recipe test
above proves the field is set, and this one proves it is CONSUMED (#1013).

`trace.schedule_tokens == 4096` on a `ti2vid_two_stage` render, and on the same
fixture geometry a `res2s_two_stage` render reports a DIFFERENT, non-4096 count.
Both halves are load-bearing. The equality alone passes on a build that
hard-codes 4096 everywhere; the inequality alone passes on today's tree.

The stronger half is recomputation rather than comparison, and its limit has to
be stated because it is easy to overclaim. `Ltx2ConditioningTrace` exposes
`schedule_tokens`, a COUNT, and no sampled sigmas, so there is nothing on the
render side to compare sigma values against. What the case can do, and does, is
rebuild TWO schedules — `Ltx2SigmaSchedule(steps, 4096)` and
`Ltx2SigmaSchedule(steps, hq_small)` — and require them to DIFFER, which fails
on a build whose shift arithmetic has been flattened and on which selecting the
anchor would be inert. The render-to-trajectory link therefore runs through the
counter alone.

Two things keep that from being vacuous. `steps` is the count the RENDER ran at,
returned out of the render lambda rather than restated as a literal, so the
recomputation cannot drift away from what was sampled. And `steps > 2` is
asserted directly, because at two steps `stretch` pins both non-zero sigmas and
the two schedules are equal for every token count.

### 3. The per-arm x0 invariant on stage 1 (`test_ltx2_video.cpp`)

The correctness trap this arm inherits: guidance combines **x0**, not velocity.
Every linear term is invariant under `x0 = latent - sigma*v`, so cfg, stg and
modality cannot see the difference; the RESCALE branch is not invariant and
`rescale_scale` defaults to 0.7, which is the default path (#1039, #1092).

A magnitude assertion cannot gate it — on a reduced fixture
`std(cond)/std(pred)` is 1.0 to 1e-5 in BOTH spaces. The gate is the per-arm
equation `x0 == latent - sigma*velocity`, three recorded tensors, exact in x0
space and off by the whole sample in velocity space, where it degenerates to
`|x0 - velocity| = 0`. Both residuals are printed on every arm so a RED says
which space it landed in.

Ported from `"ltx2 one_stage: all four guidance arms are combined in X0 space
(#1092)"` (`:6245`) onto a `ti2vid_two_stage` load, over **all four** passes —
cond, uncond, perturbed, modality — each with its own non-vacuity guard, because
a zeroed velocity collapses the equation to `x0 == latent` for that arm alone.

### 4. Reachability (`test_ltx2_video.cpp`)

`LoadVideoEngine` with `pipeline_kind = ti2vid_two_stage` -> `Generate` ->
pixels, which is `include/vllm.h` plus the documented load extras and is what
`ltx2-gen --pipeline-kind ti2vid_two_stage --lora-path ... --upsampler-path ...`
does through the ABI.

**#928 does NOT exclude the HTTP route here, and the a2vid wording would have
said it did.** That recipe needs `audio_path`, a PER-GENERATION extra, and
`VideoGenParamsFromRequest` writes none. All three knobs this recipe needs are
LOAD extras, which a server supplies through `--video-extra KEY=VALUE`
(`server_main.cpp` applies `args.video_extras` onto `vmp.extras`), and
`requires_audio_input` is false. That is a claim about the request SURFACE.
Nothing here drives the HTTP route end to end, so it is not measured and the
reach claim rests on the ABI path alone.

`dit_forwards`, not an evaluation count, distinguishes the guided stage: a
denoiser call is ONE evaluation whether or not guidance ran, and only
`Ltx2ConditioningTrace::dit_forwards` counts actual `Ltx2DitForward` calls.
Stage 1 guided at the default cfg must show more forwards per evaluation than
stage 2, which is `kSimple`.

### 5. The `requires_distilled_lora` refusal

A `ti2vid_two_stage` load with no `lora_path` refuses BY WHAT IS MISSING, and
the message names the pipeline. Mirrors `--distilled-lora required=True`
(`utils/args.py:1140-1155`). The control is the same load WITH `lora_path`,
which renders — otherwise the case passes on any load failure.

### Mutations required to pass

1. `stage1.schedule_tokens = kSchedulerDefault` deleted. Test 2 RED.
2. `stage1.loras = kNoAdapters` deleted. The recipe test RED.
3. `recipe.audio_output_phase = 0` -> 1. The recipe test RED.
4. `recipe.requires_distilled_lora = true` deleted. Test 5 RED.
5. The `ti2vid_two_stage` branch deleted from `ResolveLtx2PipelineRecipe` — the
   standing reachability mutation. Tests 2, 3, 4, 5 RED.
6. `Ltx2GuidedDenoise` left in velocity space on one arm. Test 3 RED.
7. Test 2's `gen.steps` 3 -> 2 — the mutation that the FIRST head passed. Test 2
   RED, on both `at_anchor != at_target` and `rendered_steps > 2`. A version of
   test 2 that recomputes at a literal is green under it and gates nothing about
   the step count it renders at.

Every mutation prints FOUR facts: `git diff --stat`, whether it BUILT, the
compile-error count, and the exit code captured directly. A mutation that fails
to build, and a mutation that never applied, both read as a passing test.
Anchor uniqueness is asserted before each one.

**`stg_blocks = []` is legal.** Upstream defaults it to `[]` and validates it
nowhere. No refusal is added for it; one had to be removed already.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Whole binaries, never a `--test-case` filter: a filter matching zero cases
prints `SUCCESS!` at exit 0, and LTX case names contain commas, which doctest
`-tc` splits on. Assert a non-zero case AND assertion count. A thrown case
prints `0 failed` beside `Status: FAILURE!`, so the exit code is the authority.

Report `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the pass/fail line, and `No space left` / `BFD` greps WITH
positive controls, plus load and free disk. ~504 tests registered.

`READER ANCHORS` (`ltx2_video.cpp`) is gated by `test_ltx2_video` and shifts
whenever the readers above it move. This row edits the schedule block, so the
list may move: re-derive with the test's own walk and arm the instrument first
by inserting a line and confirming MISMATCH.

**No GPU.** The fleet is leased with `rc`, `dgx:gpu0` reads `unhealthy`, and a
recipe row is correctly gated by the CPU goldens.

## What is NOT verified

**No real-weights render, and the reason is a live blocker, not a choice.**

Upstream marks this arm `Full + distilled LoRA`
(`packages/ltx-pipelines/CLAUDE.md:17-30`). Stage 1's identity is CFG on the
UNADAPTED model, so the checkpoint it needs is
`ltx-2.5-22b-dev-transformer-bf16.safetensors`. That file is now on the NAS and
byte-verified (42,018,190,584 B, 4349 tensors, 21.004 B params, pure BF16;
`ltx25-phase-lora.md` `## Owed` records the header read), and the distilled
adapter is beside it (8,899,889,568 B). So the artifacts are no longer missing.

**`PlanDit` used to refuse a pure-BF16 DiT**, which was
[#1148](https://github.com/mudler/vllm.cpp/issues/1148) and was named here as
this row's blocker. **It closed while this row was in flight**, at `40a796aa9`
(row `LTX25-BF16-DIT`), which this branch merged before pushing. So the load
path is no longer the obstacle and the paragraph that said it was would have
shipped stale.

**What remains owed is therefore the RUN, not a fix and not an artifact**: a GPU
lease, a build on that host, upstream's render and ours on the same checkpoint,
prompt and seed, and the comparison. `dgx:gpu0` reads `unhealthy` and no GPU
work is in this row's scope, so it is owed rather than attempted.

**Running this arm against a DISTILLED checkpoint instead would be worse than
not running it.** The distilled scales are trained INTO those weights, so a
CFG-guided stage 1 on top samples a trajectory they were never trained for, and
it renders — right size, right frame count, right sample rate, plausible
picture, no diagnostic (#1137). Presenting that as verification would be the
exact silent-wrongness this campaign's gates exist to catch. A sibling agent
refused the same substitution and was right to.

So the claim this row makes is: gated on CPU goldens, correct against the
upstream SOURCE line by line, and NOT measured against upstream's own render.

## Outcome

Row `DONE`. Recorded here because neither the code nor the Git history carries
it: what the gate MEASURED, and the two things this row found that its dispatch
did not know.

### The schedule anchor is INVISIBLE at two steps, and the gate was written at two

The 2x2 in test 2 reports, from the case's own `MESSAGE` line:

```text
ti2vid: 4096 / 4096   res2s: 2 / 8
```

Exactly the intended split — this arm's anchor does not move with resolution and
the HQ arm's does, 2 tokens at 64x64 and 8 at 128x128.

**And the trajectory half of the same case went RED, with the counters already
correct.** `Ltx2SigmaSchedule(2, 4096)` and `Ltx2SigmaSchedule(2, 2)` are both
`{1, 0.1, 0}`. The scheduler's `stretch` pins sigma[0] at 1.0 and renormalises
so the LAST non-zero sigma is exactly `terminal` = 0.1 (`schedulers.py:48-55`),
and a 2-step schedule has only those two non-zero entries — so the shift is
entirely absorbed and **every token count produces the same schedule**. The
fixture's usual step count is 2.

So a version of this case that asserted only the counters would have been green,
correct, and unable to see whether the anchor reached anything. Three steps is
the shortest schedule with an interior sigma.

**The first fix for that was itself vacuous, and a fresh review caught it.** The
case pinned the degeneracy by asserting that a 2-step schedule is
token-independent — true, load-bearing against a future scheduler change, and
silent about this case's own step count, because the trajectory half recomputed
at the LITERAL 3 while the render read `gen.steps`. Mutating `gen.steps` from 3
to 2 was NOT DETECTED: 83 cases, 2577 assertions, exit 0. Nothing the case
asserted depended on the step count it rendered at, and the only render-derived
observable in it is `schedule_tokens`, a count that does not move with steps at
all. The pull request body claimed the opposite, which under
`squash_merge_commit_message = PR_BODY` would have become the permanent commit
message of the row whose headline finding is about vacuous gates.

The step count now comes back OUT of the render lambda and the recomputation
runs at it, and `steps > 2` is asserted by name. Both assertions fire on the
mutation, so lowering the step count to make the case faster is refused rather
than deprecated.

**This bounds #1150 as well.** A short distilled schedule cannot see the anchor
either, which is part of why three shipped arms carried the divergence with
every gate green.

### The refusal message was a third stale #1118 site

`requires_distilled_lora`'s refusal still told callers that this engine fuses
once at load so stage 1 sees the adapter — false since `4ae0f54ab` — and it
hard-coded `a2vid_two_stage.py`'s line numbers into a message deliberately keyed
on the flag so that this row would inherit it. `ltx25-phase-lora.md` repaired the
reference-conditioning refusal, which carried the identical claim about 1100
lines away, and named only that one. The `ltx2-gen --help` text carried it too.
Filed as #1151 and fixed in flow, because this row is the first caller the
wrong-pipeline half would have misled.

**The a2vid test asserted the stale string was PRESENT** (`message.find("1118")
!= npos`), so the record was gated into place. It now asserts absence.

### What the gate measured

Focused, at the head this row pushes:

| Binary | cases | assertions | exit |
|---|---|---|---|
| `test_ltx2_pipeline` | 54 | 3182 | 0 |
| `test_ltx2_video` | 83 | 2581 | 0 |

The `test_ltx2_video` row was recorded as 82 / 2496 through the review and was
wrong on both numbers at every head this row ever pushed; the fresh review
caught it. 83 / 2577 was the head under review, and the four assertions that
close the step-count decoupling above bring it to 2581.

Full gate at the same head: `CONFIGURE_EXIT=0`, `BUILD_EXIT=0`, `: error:` count
0, `ctest -N` 506, `CTEST_EXIT=0`, `100% tests passed, 0 tests failed out of
506`, `No space left` and `BFD` each 0 against injected controls that returned 1.

RED before the recipe landed, captured on the same binaries: `test_ltx2_pipeline`
54 cases / 3063 assertions, `Status: FAILURE!`, exit 1, both new cases throwing
`Unsupported LTX pipeline kind/version: 'ti2vid_two_stage'/'2.5'`;
`test_ltx2_video` 82 / 2432, exit 1.

### Two things a later reader should not re-derive

- **`allow_request_latents = false` here and `true` on `res2s_two_stage`** is not
  an inconsistency this row introduced. Neither pipeline's `__call__` takes an
  initial latent, and the field has **no reader in `src/`** — it is asserted in
  tests only. This row matched `a2vid_two_stage`, whose value is the one derived
  from the signature. The res_2s row's `true` is unexamined rather than wrong,
  and moving it is not this row's to do.
- **`READER ANCHORS` moved by 13 lines** and the instrument caught it unprompted,
  which is the arming evidence: the gate went RED on a real shift before it was
  re-derived, and the printed list was pasted rather than computed by hand.

## Dependencies

- #1118 (`LTX25-PHASE-LORA`), landed at `4ae0f54ab`. Supplies
  `Ltx2PhaseRecipe::loras` and `Ltx2RebindDitLoras`. Without it stage 1 could
  not run unadapted and this row would return `NEEDS_DECISION`, which is what
  the scoping pass on #1093 did.
- #1117 (`LTX25-A2VID-RECIPE`), landed at `d1e5e9bc0`. Supplies
  `requires_distilled_lora` and its refusal, keyed on the flag rather than on
  the kind string — `ltx2_pipeline.h:727-729` names this row as the waiting
  second user.
- #1092 / #1102 (`LTX25-GUIDED-VIDEO`), landed at `daeff67f2`. Supplies
  `Ltx2GuidedDenoise`, without which stage 1's CFG has no seam.

## Owed

- **The other three divergent arms on the schedule anchor**, owned by
  [#1150](https://github.com/mudler/vllm.cpp/issues/1150): `one_stage` at four
  version keys (`ti2vid_one_stage.py:207`), `a2vid_two_stage` stage 1
  (`a2vid_two_stage.py:226`) and `retake`'s non-distilled arm
  (`retake.py:287`). The seam they need is the field this row adds; what they
  additionally need is their own goldens and their own fresh review, because
  flipping them re-samples five shipped arms.
- **The real-weights comparison against upstream's own render** on the dev
  transformer, same checkpoint, prompt and seed. Neither the artifacts nor the
  load path is missing any more: #1148 closed at `40a796aa9` while this row was
  in flight, and this branch merged it. What is owed is a GPU lease and the two
  renders. No owning issue is filed for it, because #644 owns the standing
  "close every refused arm" sweep and every LTX arm shipped to date carries the
  same debt; filing one per arm would be six issues saying one thing.
- **Per-phase adapter STRENGTH**, [#1144](https://github.com/mudler/vllm.cpp/issues/1144).
  Not needed here: upstream gives this pipeline `loras=tuple(loras)` against
  `(*loras, *distilled_lora)`, i.e. ABSENT vs PRESENT, which
  `Ltx2PhaseLoraScope` expresses exactly. It IS needed by
  `ti2vid_two_stages_hq.py`, whose two stages are both fused at 0.25 and 0.5 —
  and `Ltx2RebindDitLoras` early-returns on `currently_fused == fuse`, a
  BOOLEAN, so that transition would silently no-op.
- **Upstream's separate user `loras` list**, which rides BOTH stages
  (`:151`'s `*tuple(loras)`). This engine's one adapter slot is upstream's
  `distilled_lora`; the second list has no spelling here until the adapter arity
  refusal lifts (`ltx2_lora.h:167-172`).
- ~~**`keyframe_interpolation`** (#1096), the fourth pipeline on this parser.~~
  Landed as row `LTX25-KEYFRAME-INTERP`, which took this row as its template and
  set `schedule_tokens = kSchedulerDefault` on its own stage 1. That moved the
  keyframe line of the table above from `n/a` to `yes` and left the count owed to
  #1150 at **three**, where it already stood: the six-to-one split is the
  `latent` column, which no port can change, and the keyframe arm was never one
  of the divergent three because it was unported rather than wrong.
- **`allow_request_latents` has no reader in `src/`**, on any recipe, owned by
  [#1152](https://github.com/mudler/vllm.cpp/issues/1152). Four assignments on
  `origin/main` and five with this row's, zero readers in either count, against
  a positive control (`allow_request_sigmas`, declared one line above) that has
  a real reader at `ltx2_video.cpp:3476`. So `res2s_two_stage`'s `true` against
  everything else's `false` is a disagreement nothing can detect. This row
  writes `false`, which is what `ti2vid_two_stages.py:159-181` supports, and
  adds a fifth write to a dead field rather than leaving the default `true`
  standing as a wrong record.
  Not fixed in flow: both closes — give it a reader, or delete it and its gated
  assertions — touch five landed recipes.

## Stop conditions

Return `NEEDS_DECISION` rather than narrowing scope if the schedule field cannot
be added without moving a landed arm, or if `requires_distilled_lora`'s refusal
turns out to be keyed on the kind string after all.
