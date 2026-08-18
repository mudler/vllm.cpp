# LTX25-GUIDED-VIDEO — the guided video denoiser, as a shared seam

Row `LTX25-GUIDED-VIDEO`, under the campaign [`ltx-2-5.md`](ltx-2-5.md).
Issue [#1092](https://github.com/mudler/vllm.cpp/issues/1092).
Base: `origin/main` @ `b5756ea8c`.
Upstream: Lightricks `LTX-2` @ `fd4ded7f` (the revision every anchor below is
read at), secondary oracle `vllm-omni` (UNPINNED, #633) for the recipe tables
this tree already mirrors.

Paths are relative to `packages/ltx-pipelines/src/ltx_pipelines/` and
`packages/ltx-core/src/ltx_core/` in that checkout, as the rest of the LTX-2.5
specs use them.

## 0. Honesty statement — what this row does and does not claim

It claims: the LTX-2.5 **video** denoise path now runs upstream's guided
denoiser, that the guidance is combined in **x0** space, that one production
pipeline (`pipeline_kind = one_stage`) reaches it on its **own default
configuration**, and that the gate can see the space error that #1039 was.

It does not claim: a numerical comparison against a running oracle. There is
none — vLLM-Omni is UNPINNED (#633) and carries no LTX-2.5 recipe at all, and no
LTX-2.5 checkpoint on this host has a recorded sha256 (#1048). Every anchor
below is **source read** at `fd4ded7f`, and every number below is measured on
**this tree's own reduced fixture**. That is what an ungateable lane looks like
when it is stated instead of implied.

It does not claim to retire [#1049](https://github.com/mudler/vllm.cpp/issues/1049).
See section 6c: one of that issue's four symbols is reached by this row and three
are not, and forcing the other three would mean inventing a dispatch upstream
does not have.

## 1. Scope

**In:**

- `Ltx2GuidedDenoise` — `_guided_denoise` (`utils/denoisers.py:61-211`) in its
  own translation unit mirroring upstream's own file.
- The four passes it assembles: `cond`, `uncond`, `ptb`, `mod`.
- The `SKIP_A2V_CROSS_ATTN` / `SKIP_V2A_CROSS_ATTN` halves of
  `Ltx2DitPerturbation`, without which the `mod` pass cannot run and every video
  guider default is unreachable.
- The negative conditioning for the video path — the second half of the encode
  `GenerateAudioOnly` already performs and discards.
- The video guidance request extras that `default_1_stage_arg_parser`
  (`utils/args.py:930-1067`) exposes, gated by `allow_guidance_override`.
- The `one_stage` pipeline as the reachable consumer.

**Out, and owed rather than silently absent:**

- The other video pipelines that need this seam — `a2vid_two_stage`,
  `ti2vid_two_stages` ([#1093](https://github.com/mudler/vllm.cpp/issues/1093)),
  `ti2vid_two_stages_hq` (owned by #921),
  `keyframe_interpolation` ([#1096](https://github.com/mudler/vllm.cpp/issues/1096)),
  and beside them `hdr_ic_lora` ([#1094](https://github.com/mudler/vllm.cpp/issues/1094))
  and `dubit` ([#1095](https://github.com/mudler/vllm.cpp/issues/1095)), which are
  blocked on other things. This list named four pipelines and no issue numbers
  when the spec was written; `281e6a120` (#1099) filed them while this branch was
  open, and #1093 names this row's seam as one of the two things it waits on.
  Each is its own row; this row exists so that they are ordinary porting work
  rather than blocked.
- The **device-resident** arm of the `ptb` and `mod` passes.
  `Ltx2DitForwardDevice` (`ltx2_device.h:136`) takes no `perturbations`
  argument. Refused by name on that arm rather than run unperturbed, which would
  produce a legal-looking render whose STG term is identically zero. See §4.3.
- `BatchedPerturbationConfig`'s partial blend (`attention.py:572-573`) and
  batch > 1, which stay degenerate at the one batch size this port runs — the
  statement `ltx2.h` already carries, unchanged.

## 2. Upstream chain

The executing chain for one guided step, top to bottom:

| Step | Upstream | What it decides |
|---|---|---|
| the stage builds the model | `utils/blocks.py:480-482` — `X0Model(self._prepared_builder().build(...))` | the transformer the loop is handed is **already** an x0 model |
| the loop calls the denoiser | `utils/samplers.py:73-74` | one denoiser call per step |
| the denoiser assembles passes | `utils/denoisers.py:100-137` | `cond`, `uncond`, `ptb`, `mod`, in that order |
| the forward converts | `model/transformer/model.py:590-604` — `to_denoised(video.latent, vx, video.timesteps)` | **every** pass is x0 before any combination |
| the guider combines | `components/guiders.py:244-273`, per modality at `denoisers.py:203-204` | `cond + (cfg-1)(cond-uncond) + stg(cond-ptb) + (mod-1)(cond-modpass)`, then the rescale at `:268-271` |
| the loop post-processes | `utils/samplers.py:35` — `post_process_latent(denoised, ...)` | the conditioned tokens are pinned back **after** the guider, not per arm |
| the stepper steps | `utils/blocks.py:524-527` / `samplers.py:488-558` | Euler or ancestral, unchanged by this row |

The pass list is **shared between the two modalities and the guiders are not**.
`denoisers.py:103-137` takes the union — one `uncond` pass if *either* guider
wants one, one `ptb` pass carrying *both* modalities' perturbations, one `mod`
pass if *either* wants one — and then `:203-204` combines each modality with its
**own** guider over the same splits. A per-modality pass list would run up to six
forwards where upstream runs four, and would give the audio stream a different
video state to cross-attend to on the video-only passes. That is the single
structural fact this port has to get right, and it is why the seam takes both
guiders rather than being called twice.

The perturbation types are per direction, not per modality
(`guidance/perturbations.py:8-16`, applied at `model.py:443-458`):

| Pass | Perturbations | Reaches |
|---|---|---|
| `cond` | none | — |
| `uncond` | none | negative context on both streams |
| `ptb` | `SKIP_VIDEO_SELF_ATTN` on `video_guider.stg_blocks`, `SKIP_AUDIO_SELF_ATTN` on `audio_guider.stg_blocks` | `attention.py:557` `use_attention = not all_perturbed` |
| `mod` | `SKIP_A2V_CROSS_ATTN` and `SKIP_V2A_CROSS_ATTN`, **all blocks** | `transformer.py:335,367` `cross_attn_skip_all` |

## 3. Our baseline, derived at `b5756ea8c`

`src/vllm/multimodal/ltx2_video.cpp:3036-3045` runs **one** forward per step and
converts its velocity:

```
const Ltx2DitOutputs velocity = im.on_device ? Ltx2DitForwardDevice(...) : Ltx2DitForward(...);
const std::vector<float> v_denoised = PostProcessLatent(ToDenoised(video.latent, velocity.video, ...), video);
const std::vector<float> a_denoised = PostProcessLatent(ToDenoised(audio.latent, velocity.audio, ...), audio);
```

Everything the recipe resolved for that step is set and read by nothing:

```
$ git grep -n 'video_guidance' -- src include            # @ b5756ea8c
include/vllm/model_executor/models/ltx2_pipeline.h:526:  Ltx2MultiModalGuiderParams video_guidance;
src/vllm/model_executor/models/ltx2_pipeline.cpp:1069:  phase.video_guidance = params.video_guider;
```

The positive control for that grep is the same command for `audio_guidance`,
which returns the T2A consumer at `ltx2_video.cpp:3527`. The term and the path
set are right; the video consumer is genuinely absent. `allow_guidance_override`
(`ltx2_pipeline.h:534`) is the same shape: three recipes set it `false` and
nothing reads it.

**What is already correct and is reused unchanged:**

- `Ltx2MultiModalGuidance` (`ltx2_pipeline.cpp:479-522`) — `calculate` including
  the unbiased-`std` rescale. Reviewed under #1032/#1039.
- `Ltx2BatchedPerturbationConfig` (`ltx2_pipeline.h:380-405`) — the full
  four-type keep-mask, ported under #641 and, per #1049, constructed only by its
  own test until this row.
- `ToDenoised` (`ltx2_video.cpp:277`) and `PostProcessLatent` (`:234`).
- The T2A driver (`ltx2_t2a.cpp:322-368`), which is the **template**: it is the
  one place in this tree that already converts to x0 inside the model wrapper.

**What is missing and why nothing noticed:** a token gate cannot see it, and this
path has no token gate. An unguided render returns a finite clip of the right
size, frame count and sample rate. It is #1039's family of defect one level up:
not the wrong space, the wrong number of forwards.

## 4. Design

### 4.1 The seam — `ltx2_denoisers.{h,cpp}`

A new translation unit mirroring `ltx-pipelines/utils/denoisers.py`, rather than
another block inside `ltx2_pipeline.cpp`. Two reasons, and only the first is
about this row:

1. Upstream has that file. `AGENTS.md` §Shared seams: new capability is additive
   files mirroring the upstream structure.
2. `ltx2_pipeline.{h,cpp}` is concurrently edited by #921. A seam that four
   later rows will extend does not want to live in the file with the most
   contention.

The transformer is a **callable**, exactly as `_guided_denoise(transformer, ...)`
takes one:

```
using Ltx2X0Model = std::function<Ltx2X0Outputs(const Ltx2ModalityInput* video,
                                                const Ltx2ModalityInput* audio,
                                                const Ltx2DitPerturbation* perturbations)>;
```

That is the structural claim this row is graded on. The x0 conversion happens
**inside the caller's lambda**, which is upstream's `X0Model` wrapper
(`blocks.py:480-482` builds it; `model.py:590-604` is its forward), so the seam
combines already-denoised tensors. Converting once after the guider instead is a
**different function on the default arm** — `rescale_scale` is 0.7 on every
video row — and that is #1039, on the audio arm, in this tree, six days ago.

**That is caller discipline, not a type guarantee**, and this section claimed the
stronger thing ("cannot be handed a velocity") until 2026-08-17.
`Ltx2X0Outputs` carries `video_velocity` and `audio_velocity` beside `video` and
`audio` (`ltx2_denoisers.h`), because the per-arm invariant is an equation
between three tensors and cannot be checked from the denoised one alone. A lambda
that fills `video` with the velocity therefore compiles and renders. What stops
it is the gate, not the signature: mutations M1 to M4 hand the seam a velocity on
one arm each and all four are red. The claim is restated rather than the code
changed, because dropping the velocity from the struct would delete the evidence
the invariant is checked against.

It also means the host forward and the device forward are the same seam with two
lambdas, and that the four later pipelines supply their own conditioning without
the seam knowing anything about keyframes, reference clips or two-stage
schedules.

### 4.2 The passes

`Ltx2GuidedDenoise` mirrors `denoisers.py:84-207` line for line:

- `v_skip`/`a_skip` from `ShouldSkipStep` (`:84-85`); both skipping returns the
  previous step's denoised pair with **no forward at all** (`:87-90`).
- `cond` always (`:100`).
- `uncond` when either guider asks or `force_uncond_pass` (`:102-109`), with the
  negative context substituted per modality and `v_neg = v_context` when a
  modality has none (`:107-108`).
- `ptb` when either guider perturbs, carrying both modalities' `stg_blocks`
  (`:111-119`).
- `mod` when either guider isolates, all blocks, both cross directions
  (`:121-137`).
- `enabled = not skip` per modality (`:158,168`), which is
  `Ltx2ModalityInput::enabled` here.
- the combination per modality with that modality's own guider (`:203-204`).

Perturbations route through `Ltx2BatchedPerturbationConfig`: one config built
over all N passes (`denoisers.py:182-187`), then `BatchSlice(i, i+1)` per pass,
then flattened into the `Ltx2DitPerturbation` the forward takes. At batch 1 the
slice is the pass's own mask, which is exactly the degeneracy `ltx2.h` already
records.

### 4.3 The cross-attention perturbation

`Ltx2DitPerturbation` grows two booleans, `video_cross_attn_skip_all` and
`audio_cross_attn_skip_all`, mirroring `TransformerArgs.cross_attn_skip_all`
(`transformer_args.py:70`). They gate the A2V and V2A branches at
`ltx2_dit.cpp`'s `if (run_a2v)` / `if (run_v2a)`, mirroring
`transformer.py:335` and `:367`. Note the polarity: `video.cross_attn_skip_all`
gates **A2V** (audio into video) and `audio.cross_attn_skip_all` gates **V2A**,
because the flag rides on the stream being *written*.

The snapshot of `vx_pre`/`ax_pre` stays outside both guards, as upstream's
`vx_pre_av = vx` at `:333` does, so a build where only one direction is skipped
still reads the pre-cross state for the other.

`ltx2.h:41-49`'s NOT-PORTED entry is corrected in the same change. Its stated
reason — "nothing upstream that this port serves constructs them" — was true for
text-to-audio, which pins `modality_scale = 1.0` (`t2a_one_stage.py:202`),
and is false for every video pipeline, all of which default it to 3.0.

**The device arm is refused, not degraded.** `Ltx2DitForwardDevice` has no
`perturbations` parameter, so a `ptb` or `mod` pass on that arm would have to
run unperturbed. The result is a finite clip whose STG and modality terms are
identically zero — indistinguishable from a working render. The refusal names
the missing function and the owed issue. CFG alone (a different context, no
perturbation) is served on both arms.

### 4.4 The negative conditioning

`Generate` already encodes the positive prompt into both streams
(`ltx2_video.cpp:1771-1806`). The negative half is the same chain with
`recipe.negative_prompt` (or the `negative_prompt` extra), through the same
connector, and is encoded **only when a guider asks for it**
(`do_unconditional_generation`, `guiders.py:275-277`) — at `cfg_scale = 1.0`
there is no unconditional forward and encoding it would be a wasted host-side
12B pass per request.

Two fallbacks exist for an engine with no text tower, matching the two that
already exist for the positive stream (`prompt_embeds_path`,
`audio_prompt_embeds_path`): `negative_prompt_embeds_path` and
`negative_audio_prompt_embeds_path`. This is a **local adaptation**, recorded as
one: upstream encodes `[prompt, negative_prompt]` in one `PromptEncoder` call
(`ti2vid_one_stage.py:166-174`) and has no embeds surface at all. The adaptation
is the existing one applied to the second of upstream's two encodings, not a new
concept. Without a tower and without those files, a guider that asks for the
unconditional pass is refused by name, exactly as T2A is at
`ltx2_video.cpp:3583-3593`.

### 4.5 The request extras

Mirroring `default_1_stage_arg_parser` (`utils/args.py:947-1066`), one extra per
flag, each overriding one field:

| Extra | Upstream flag | Field |
|---|---|---|
| `video_cfg_guidance_scale` | `--video-cfg-guidance-scale` | `cfg_scale` |
| `video_stg_guidance_scale` | `--video-stg-guidance-scale` | `stg_scale` |
| `video_rescale_scale` | `--video-rescale-scale` | `rescale_scale` |
| `video_stg_blocks` | `--video-stg-blocks` | `stg_blocks` |
| `a2v_guidance_scale` | `--a2v-guidance-scale` | video `modality_scale` |
| `v2a_guidance_scale` | `--v2a-guidance-scale` | audio `modality_scale` |

The audio row already exists for T2A (`ltx2_video.h:456-460`) and is reused for
the joint path. Every override is refused on a phase whose
`allow_guidance_override` is `false` — the distilled and retake recipes, whose
guidance is trained in — which is the first read that field has ever had.

An extra that is PRESENT and empty is upstream's empty list for `nargs="*"`, and
stays distinct from an ABSENT extra, which takes the params table's own value.
That distinction is already made for `audio_stg_blocks` (`ltx2_video.cpp:3540`)
and is made the same way here.

## 5. Port map

| Upstream | Here |
|---|---|
| `utils/denoisers.py:61-211` `_guided_denoise` | `Ltx2GuidedDenoise`, `src/vllm/model_executor/models/ltx2_denoisers.cpp` |
| `utils/denoisers.py:25-28` `_POSITIVE_ONLY_GUIDER` | the default-constructed `Ltx2MultiModalGuiderParams`, whose defaults are already `cfg 1.0 / stg 0.0 / modality 1.0` |
| `model/transformer/model.py:590-604` `X0Model.forward` | the caller's `Ltx2X0Model` lambda, `ltx2_video.cpp` |
| `guidance/perturbations.py:8-16` cross types | `Ltx2DitPerturbation::{video,audio}_cross_attn_skip_all` |
| `model/transformer/transformer.py:335,367` `cross_attn_skip_all` | `ltx2_dit.cpp` A2V / V2A guards |
| `components/guiders.py:244-273` | `Ltx2MultiModalGuidance` (unchanged) |
| `utils/args.py:947-1066` | the six extras in §4.5 |
| `ti2vid_one_stage.py:211-226` | the `one_stage` consumer in `ltx2_video.cpp` |

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Focused: `ctest --test-dir build -R 'ltx2' --output-on-failure`.

Known-red and cited by the issue that names the test, not the family:
`windows-msvc-*` (#584, no `main` baseline), `test_async_llm` (#294),
`test_engine_core_proc` (#1052), `test_serve_low_tools` (#428),
`test_cpu_x86_llamacpp_floor` exit 4 (#618).

## 6b. Reachability — the sentence the records must carry

Entry point: `vllm_video_generate` → `VideoEngine::Generate`
(`include/vllm.h`) on an engine loaded with `pipeline_kind = one_stage`, which
is a documented value of a documented load extra and needs no other flag. The
chain is `Generate` → the phase loop → `Ltx2GuidedDenoise`. No test constructs a
guider, a DiT or a modality by hand to reach it.

The reachability mutation is the deletion of that call — the
`Ltx2GuidedDenoise(...)` line in the phase loop, replaced by the single unguided
forward this row removes — with the focused gate rerun. A green gate there would
mean the suite measures the seam and not the pipeline.

## 6c. #1049 — partly retired, and the rest argued rather than deferred

| Symbol | Before | After |
|---|---|---|
| `Ltx2BatchedPerturbationConfig` | test-only | **reached**, `ltx2_denoisers.cpp` |
| `Ltx2Guidance` | test-only | still test-only |
| `Ltx2CfgDelta` | reached only via `Ltx2Guidance` | unchanged |
| `Ltx2StgDelta` | reached only via `Ltx2Guidance` | unchanged |

`Ltx2Guidance` is a **kind dispatch upstream does not have**. Every LTX-2
pipeline builds a `MultiModalGuider` and calls `calculate`; there is no object
that holds a `GuiderKind` and selects between CFG-only, STG-only and multi-modal
arms. Routing the production combination through `Ltx2Guidance(kMultiModal, ...)`
to make the symbol live would add a switch statement between the caller and the
function upstream actually calls, and would still leave `Ltx2CfgDelta` and
`Ltx2StgDelta` — the two arms nothing can select — dead. #1049 stays open, its
scope narrows to those three symbols, and the honest disposition is that they
are ported-but-unreachable arms of `guiders.py:11-27,56-74`, not a wiring gap
this row can close.

## 7. Tests to port

Upstream's own tests for this path are `pytest` over `torch` and cannot be run
here; the harness adaptation is the whole of it. What is preserved is the
**structure of what they assert** plus the four defects this tree has already
had on the sibling arm.

### 7.1 The per-arm invariant, on every arm

`cond == latent - sigma*velocity`, per pass, from the trace the render records.
Exact in x0 space; in velocity space the residual is the whole sample. The RED
prints `|x0 - velocity| = 0` **exactly**, which is unambiguous.

Non-vacuity is `REQUIRE`d, not assumed, twice: a zero latent makes the two
candidate tensors coincide, and a zero velocity on a given arm makes
`to_denoised` the identity for that arm alone.

**Every arm**, because #1039's first gate covered only the conditional pass and
three mutations survived it. This path has four arms, so it needs four rows plus
the two double-application positions.

### 7.2 The seam-level rescale control

`rescale_scale = 0.0` against `0.7` on the shipped
`Ltx2MultiModalGuidance`, measuring the disagreement between combining in x0
space and combining in velocity space. At 0.0 the linear terms are invariant and
the two are the same function; at 0.7 they are not. A gate that fires at 0.0 is
not about this defect. The existing T2A case measured 1.50e-07 against 0.352.

**The modality term is in this control and is inert in it**, which is a weaker
statement than this section made until 2026-08-17. It said the video case "adds
the modality term, which the T2A control could not carry" and left a reader to
infer that the control therefore covers the modality arm. It does not: pinning
`modality_scale` to 1.0 moves the shipped-rescale disagreement from
`4.054e-01` to `4.118e-01`, a 1.6% change, so the fourth linear term changes
what the rescale is computed over and does not change whether the rescale is
what breaks the equivalence. Presence is coverage; it is not discriminating
power. **The modality arm's gate is the per-arm invariant in §7.1**, whose
`modality` row is what mutation M4 turns red. The two numbers are now asserted
inside the case rather than argued here, so a later reader who leans on this
control for modality coverage is contradicted by an assertion instead of by a
paragraph.

### 7.3 The pass count

`std(cond)/std(pred)` is 1.0 to 1e-5 on this fixture in **both** spaces, so a
naive numeric assertion on the rescale difference passes whether or not the bug
exists (7.6e-07 against a span of 3.41, measured under #1039). The instrument
that works is the **count of forwards by kind**, recorded at the call and not
asserted in prose: an arm silently skipped changes a counter no output does.

### 7.4 The mutations this gate must survive

| # | Mutation | Must go RED at |
|---|---|---|
| M1 | `cond` pass left in velocity space | §7.1 cond row |
| M2 | `uncond` pass left in velocity space | §7.1 uncond row |
| M3 | `ptb` pass left in velocity space | §7.1 ptb row |
| M4 | `mod` pass left in velocity space | §7.1 mod row |
| M5 | second `ToDenoised` **below** the step-0 record | the Euler-recovery check |
| M6 | second `ToDenoised` **above** the step-0 record | the guider-replay check |
| M7 | `uncond` pass given the positive context | the replay check / a uncond≠cond check |
| M8 | `mod` pass given no cross-attn perturbation | a mod≠cond check |
| M9 | `ptb` pass given no self-attn perturbation | a ptb≠cond check |
| M10 | `PostProcessLatent` applied per arm instead of after the guider | the replay check |
| M11 | the production call site deleted (reachability) | the whole case |
| M12 | the DiT ignores `video_cross_attn_skip_all` | §7.5 A2V row |
| M13 | the DiT ignores `audio_cross_attn_skip_all` | §7.5 V2A row |
| M14 | the DiT ignores BOTH cross flags | §7.5, and the shipped-path `mod != cond` check |
| M15 | the DiT SWAPS which flag gates which direction | §7.5 both rows |

Each mutation reports three facts: `git diff --stat`, whether it **BUILT** with
the compile-error count, and the exit code captured directly. A non-building
mutation reads exactly like a passing test.

### 7.5 The two cross directions, gated per direction

M12, M13 and M15 were all **GREEN** against the first draft of this row's gate,
which had no direct DiT-level cross case: only M14 was caught. A build that
plumbs both flags and applies exactly one, or applies both to the wrong
directions, renders — on the DEFAULT video arm, whose `modality_scale` is 3.0 —
with the isolated-modality term half wrong.

Two things made the shipped-path case blind to it. The end-to-end
`MaxAbsDiffOf(video_first_modality, video_first_cond)` still fires with one
direction applied, because the modality pass still differs from `cond`. And
`Ltx2ConditioningTrace::video_modality_skipped_{a2v,v2a}` is assigned from the
perturbation struct **the seam built** (`ltx2_denoisers.cpp:315-316`), so it
records what was handed over and nothing about what the DiT did with it — while
its message claimed the latter.

The instrument is a **direct `Ltx2DitForward` case per direction**, mirroring the
self-attention one this file already had. Separation comes from upstream's own
predicates (`transformer.py:265-269`): `run_a2v` needs the VIDEO stream enabled
and the audio stream merely PRESENT, and `run_v2a` the reverse. So a forward with
`audio->enabled = false` runs A2V alone, and one with `video->enabled = false`
runs V2A alone. Each row asserts both halves — the flag for that direction MOVES
the written stream, and the flag for the other direction leaves it BIT-IDENTICAL
— which is what makes the swap detectable rather than only the omission. The
per-block-pair coupling that defeats a both-enabled forward (block 1's V2A reads
what block 0's A2V wrote) never arises, because only one direction runs at all.

## 8. Risks and decisions

**R1 — the space error, on the default arm.** `rescale_scale` is 0.7 on the
2.4/2.5 video row and 0.45 on the HQ row; both are non-zero, so a space error
hits the default. Mitigated by §7.1 and §7.2, and by the conversion living in
the caller's lambda where the seam cannot receive a velocity.

**R2 — the shared pass list.** Assembling per modality is the plausible wrong
design and it renders. Mitigated by the pass-count trace and by the seam taking
both guiders.

**R3 — cost.** Four forwards per step where there was one. That is upstream's
own cost — `denoisers.py` batches them into one call, this port runs them
serially — and it is a **correctness** row, so the throughput axis is not traded
against it. Recorded, not hidden: a `one_stage` render is now up to 4x the DiT
work per step. `distilled_two_stage`, the default recipe and the one every
benchmark on this row's campaign used, denoises with `SimpleDenoiser` upstream
(`distilled.py:266,295`) and is **unchanged** by this row.

**R4 — the device arm.** §4.3. Refused by name, owed by a new issue, rather than
run unperturbed.

**R5 — concurrent edits.** #921 touches `ltx2_pipeline.{h,cpp}` and the stepper
enum. This row's new code is in a new file; its edits to `ltx2_pipeline.h` are
additive constants only.

## 9. Stop conditions

Stop and report `NEEDS_DECISION` rather than narrowing silently if:

- the `mod` pass cannot be made to differ from the `cond` pass on the fixture,
  because then §7.4 M8 cannot go red and the isolated-modality arm is gated by
  nothing;
- the guided `one_stage` render cannot be reached without a text tower **and**
  the negative-embeds adaptation is judged out of scope, because then the
  consumer is unreachable in-tree and this becomes a seam-only row.

## Owed

- **The device-resident `ptb` and `mod` passes.** `Ltx2DitForwardDevice` takes
  no `perturbations`. Owned by this row's follow-up issue; refused by name until
  then.
- **The other four pipelines** — `a2vid_two_stage`, `ti2vid_two_stages`
  ([#1093](https://github.com/mudler/vllm.cpp/issues/1093)),
  `ti2vid_two_stages_hq` (#921),
  `keyframe_interpolation` ([#1096](https://github.com/mudler/vllm.cpp/issues/1096)).
  Each needs its own row; none is blocked on this seam any more.
- **#1049's remaining three symbols** — see §6c.
- **[#1111](https://github.com/mudler/vllm.cpp/issues/1111) — the T2A arm still
  refuses an EMPTY `audio_stg_blocks`.** The same divergence this row's review
  found on the video path, in `ltx2_t2a.cpp:203-214`, which landed with #1032 and
  is not this row's code. Fixing it moves a landed row's gated behaviour and one
  of its cases, so it takes the normal row-spec-review path. Until then the two
  arms disagree about the same request.
- **An oracle-run comparison.** vLLM-Omni is UNPINNED (#633) and carries no
  LTX-2.5 recipe; no LTX-2.5 checkpoint here has a recorded sha256 (#1048). The
  guidance arithmetic is gated against upstream **source**, not against upstream
  **output**, and that is the ceiling on this row's evidence.

## 10. What the implementation actually did, against §4

Three things moved from the design, and each is recorded here rather than left
for a reader to diff.

**The seam runs on EVERY video render, not only the guided ones.** §4 left this
open. The implementation routes the phase loop through `Ltx2GuidedDenoise`
unconditionally, because a phase whose recipe sets no guidance keeps
`Ltx2MultiModalGuiderParams`'s own defaults and those ARE
`_POSITIVE_ONLY_GUIDER` (`denoisers.py:25-28`): one pass, and a `calculate` whose
every term is zero. Upstream selects `SimpleDenoiser` by PIPELINE
(`distilled.py:266,295`) rather than by params, and the two agree here because
the recipes that select it are exactly the recipes whose guidance is the no-op
one. **Measured, not argued:** every existing golden in `test_ltx2_video` held
unchanged across the change, including the `distilled_two_stage` renders and the
DFR and retake ones.

**One refusal that upstream does not have.** It is a case where upstream's
behaviour is a silent zero rather than an error, and where the render finishes:

- an `stg_blocks` list naming no block this checkpoint has. Upstream's
  `Perturbation.is_perturbed` is a membership test (`perturbations.py:26-33`), so
  `[28]` on a two-block DiT perturbs nothing and leaves
  `stg_scale * (cond - perturbed)` at exactly zero. Upstream never meets it
  because it only runs 48-block checkpoints; this port runs reduced ones. The
  refusal is about a request that disagrees with the **checkpoint**, which is a
  local condition rather than an upstream one.

**A second refusal was landed on this branch and is retired in the review
repair.** An EMPTY `stg_blocks` beside a non-zero STG scale was refused for the
same "the term would be exactly zero" reason. Every clause of that reason is
true, and none of it makes the configuration illegal upstream — which is the only
question a mirror gets to ask. Measured at `fd4ded7f`:

| Evidence | Where |
|---|---|
| "Set to `[]` to disable STG", in the same table and idiom as `stg_scale` → 0.0 | `ltx-pipelines/docs/multimodal-guidance.md:13` |
| `MultiModalGuiderParams.stg_blocks` DEFAULTS to `[]` | `guiders.py:204` |
| `--video-stg-blocks` / `--audio-stg-blocks` are `nargs="*"`, so `[]` has a CLI spelling; `nargs="+"` was the one-character way to forbid it | `args.py:979-985`, `:1039-1045`, `:1107-1113` |
| `LTX_2_3_HQ_PARAMS` SHIPS `stg_blocks=[]` on both modalities | `constants.py:105`, `:113` |
| no validation of `stg_blocks` anywhere in that tree — no emptiness, length or range check | measured by a whole-tree search with the null results recorded |

`blocks=None` means EVERY block and `blocks=[]` means NO block
(`perturbations.py:26-33`), so the empty list is how a caller says the second
thing — and `ApplyStgBlocksExtra` exists precisely to keep PRESENT-and-empty
distinct from ABSENT. Refusing it made that distinction unreachable, which is
the shape of the defect rather than a matter of taste. Upstream does not skip
the pass either: `do_perturbed_generation` reads `stg_scale` alone
(`guiders.py:279-281`), so the `ptb` entry is appended and its result equals
`cond`. This port now does the same, and the case asserts the exact equality
rather than a tolerance.

**The same refusal exists on the T2A arm and is NOT changed here.**
`ltx2_t2a.cpp:203-214` computes the block mask and refuses when no bit is set,
which catches the empty list as well as the out-of-range one. It landed with
[#1032](https://github.com/mudler/vllm.cpp/issues/1032) and is not this row's
code; changing it would alter a landed row's gated behaviour and one of its
cases. Filed and listed under `## Owed`.

**A cross perturbation that differs between blocks is refused rather than
widened.** `Ltx2DitPerturbation` carries one boolean per direction, which is
sound only while the config says the same thing on every block — which
`blocks=None` guarantees. That is checked in the seam rather than assumed.

## 11. Outcome

Landed on `row/LTX25-GUIDED-VIDEO`. Twelve mutations, eleven RED and one proven
an IDENTITY; the table with all three facts per mutation is in the pull request
body.

**The one green, and what chasing it changed.** `post_process_latent` ADDED per
arm, with the after-guider application left in place, runs the whole suite to
71 cases / 2145 assertions / exit 0. Two readings of that green were tried and
the first one was wrong.

The first reading was that the gate could not see it, so a check was added that
replays the guider over arms REBUILT from `latent` and the four raw velocities --
independent of anything applied to the arms alike. It did not move the green
either, which was the clue.

The second reading, measured rather than argued, is that the mutation is a NO-OP.
`post_process_latent` is `x*mask + clean*(1-mask)`, so it can only touch a token
whose denoise mask is 0; such a token's per-token sigma is 0
(`timesteps_from_mask`, `utils/helpers.py:494-503`), so `X0Model` returns
`latent - 0*v`, which is `latent`; and a conditioned token's `latent` IS its
clean value. Every arm already equals what post-processing would write.

**What that produced is a better test, not a waived mutation.** The tempting
proof of the above -- "so applying it anywhere is the same" -- is false for the
guider's RESULT, because the rescale (`guiders.py:268-271`) is a scalar over the
whole tensor and multiplies the conditioned tokens too. Take the after-guider
application away, or move it down into the denoiser, and those tokens leave the
step scaled by a number nobody asked for. The row therefore carries a CONDITIONED
`one_stage` case which asserts both halves: that no arm was touched, and that the
guider's result was, on exactly the mask-0 tokens. The MOVE mutation is RED
against it (71 cases / 2133 assertions / exit 1); the ADD mutation stays green
and the case now says why in an assertion instead of leaving it unexplained.

**A residual, stated.** Nothing here compares against a running oracle; see §0.

### 11.1 The fresh review, and what it moved

One BLOCKING finding and ten non-blocking ones. The core was reproduced and is
not revisited: the seam, the x0 space, all four arms, the reachability, and the
gate. The reviewer also confirmed the live defect independently — `video_guidance`
had exactly two hits at the merge base, a declaration and a write with no reader,
against a positive control where `audio_guidance` finds its T2A consumer.

| Finding | Disposition |
|---|---|
| B1 (blocking) — the two cross booleans were not gated per direction: M12, M13 and M15 were GREEN | REPAIRED. §7.5, a direct DiT-level case per direction. All four mutations now RED |
| B2 — `docs/FEATURES.md` said T2A was "the only GUIDED arm", which this row's own new row made false | REPAIRED in the existing cell, 202 of 220 chars, prose paragraphs unchanged at 21 of 21 |
| B3 — the EMPTY `stg_blocks` refusal diverges from upstream | REPAIRED on this row's paths; the T2A one is #1111. See §10 |
| B4 — `INFO("arm = " << arm.name)` printed `arm = 1` | REPAIRED at all three sites in the file |
| B5 — the rescale control's modality claim was numerically inert | RESTATED, and now asserted rather than argued. See §7.2 |
| B6 — upstream anchor drift | REPAIRED, 41 scripted replacements with the hit count asserted per edit. See §11.2 |
| B7 — "the seam cannot be handed a velocity" is caller discipline, not a type guarantee | RESTATED in §4.1 and in `ltx2_denoisers.h`; the code is unchanged, which is the right outcome |
| B8 — the branch was behind `origin/main` | MERGED and re-gated; §1 now carries #1093 to #1096 from `281e6a120` |
| B9, B11 | recorded by the reviewer as not this repair's |
| B10 — the new `docs/USAGE.md` section documented flags and not extra keys | REPAIRED: the `/v1/videos` caveat, a flag-to-extra table, and the empty-list behaviour |

**One sub-claim was REJECTED on evidence.** B3 argued an asymmetry: that
`audio_stg_blocks=""` is still accepted on `t2a_one_stage`, because that path
returns before `ApplyGuidanceOverrides`. It does return there, and the request is
still refused — by `ltx2_t2a.cpp:203-214`, which builds the block mask and fails
when no bit is set, and which `git log -S` puts on `main` at `0b0b8900f` with
#1032 rather than on this branch. So there is no asymmetry today; the two arms
agree, and both diverge from upstream. Fixing the video half creates the
asymmetry, which is why #1111 exists and is listed under `## Owed` rather than
left implied.

### 11.2 The anchors

Re-derived against `fd4ded7f` from the sentence making each claim, never by
reading text out of the cited span. Corrected: `_guided_denoise` **61-211** (was
62-207); `enabled = not skip` at **158, 168** (was 151, 161 — the `= None`
initializers); the V2A `cross_attn_skip_all` guard at **367** (366 is blank); the
batched config built at **182-187** (172-176 is the comment plus the per-sample
replication at :175); the partial blend at **572-573**; the one `PromptEncoder`
call at **166-174**; `default_1_stage_arg_parser` **930-1067** and its guider
flags **947-1066**; the two `--*-stg-blocks` flags at **979-985** and
**1039-1045**; `cross_attn_skip_all` DECLARED at `transformer_args.py:70` (118 is
a call site); `modality_scale = 3.0` at `constants.py:54, :64` with
`_PARAMS_SINCE_VERSION` at **130-133** (the cited 40-80 covers neither);
`CFGGuider` / `STGGuider` at **11-27** and **56-74**; the `perturbations`
ARGUMENT at `model.py:493` (492 is the `def`); `t2a_one_stage.py:202` (200-201
are its comment); `ti2vid_one_stage.py:211-226` (210 is blank).

No gate protects a spec anchor ([#632](https://github.com/mudler/vllm.cpp/issues/632)),
so the edits were applied by a script that asserts the expected hit count per
replacement and refuses the whole run on a mismatch — two of the 43 planned edits
were caught that way and re-derived.

## Now

The fresh review returned CHANGES REQUESTED; the blocking finding and the nine
non-blocking ones this repair owns are answered on the branch (§11.1), the branch
is merged up to `origin/main`, and the row awaits a second fresh review.
