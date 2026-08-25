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

- ~~**The device-resident `ptb` and `mod` passes.**~~ CLOSED by §12 on
  2026-08-19. `Ltx2DitForwardDevice` takes `perturbations` and honours all four
  fields, and the two refusals that stood in for it are gone.
- ~~**[#1426](https://github.com/mudler/vllm.cpp/issues/1426) — link B of
  §12.8.**~~ BOTH HALVES MEASURED on 2026-08-25. The device branch of the x0
  model's ternary was entered by no test on a box without an accelerator, so
  mutation D10 (drop `p` from that one branch) was GREEN while its sibling D11
  (drop it from both) was RED. That was a missing RUNNER and not a missing test.
  **Software half** (§12.8.1):
  `tests/vllm/multimodal/test_ltx2_video_device_forward.cpp` renders `one_stage`
  on `device = 1` through `LoadVideoEngine` against a synthetic unified-memory
  accelerator, and D10 is now RED on it while `test_ltx2_video` keeps its 105
  cases. **Hardware half** (§12.8.2): the OPERATOR ran
  `one_stage --device cuda` in a lease on `dgx:gpu0` at this branch's base
  commit and recorded 120 `dit forward` ticks across 30 steps against the
  unguided control's 60, which is the `dit_forwards == 4 * dit_evaluations` this
  entry named, plus a guided-against-unguided output A/B that a dropped `p`
  could not produce. Neither half replaces the other: the leased run cannot
  execute in continuous integration, and the synthetic accelerator cannot prove
  a GPU ran anything. The same shape
  [#1131](https://github.com/mudler/vllm.cpp/issues/1131) records for
  MiniMax-Music3.
- **The WALL-CLOCK cost of §12.5, unmeasured on hardware.** The forward COUNT is
  exact — 120 where [#1375](https://github.com/mudler/vllm.cpp/issues/1375)
  measured 60 — and needs no lease. The **2.0x is an UPPER BOUND**, because it
  prices all four passes at #1375's `cond` cost while `ptb` and especially `mod`
  do strictly less work (§12.5). The true figure is above 1.5x and below 2.0x and
  nothing here ran it. Closes with the same leased run #1426 needs, and
  [#1413](https://github.com/mudler/vllm.cpp/issues/1413)'s per-forward `last=`
  tick supplies the PER-PASS duration directly (§12.10), which is what separates
  the bound from the number.
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
  **output**, and that is the ceiling on this row's evidence. Sharper than it
  read: the source this row cites is `Lightricks/LTX-2` at `fd4ded7f`, a THIRD
  repository with no registry row, no `.agents/oracles/` file and no pin at all —
  filed as [#1433](https://github.com/mudler/vllm.cpp/issues/1433) and owned by
  `MODEL-DIFFUSION-LTX25`, because adding a row to the `AGENTS.md` oracle table
  is a policy edit.

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

## 12. The device-resident perturbed forward — the owed half of #1092

Added 2026-08-19, base `origin/main` @ `96ed8346f`. Upstream pin unchanged:
Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, verified with
`git rev-parse HEAD` in `/home/mudler/_git/LTX-2` on 2026-08-19.

§11 landed the guided denoiser. It landed with one arm missing, recorded under
`## Owed` and refused by name: `Ltx2DitForwardDevice` took no `perturbations`
argument, so on `device != 0` the `ptb` and `mod` passes could not run and the
render was refused rather than served an unperturbed forward. This section is
that arm.

### 12.1 What the refusal cost, measured

The first full-model render reached a DiT forward on 2026-08-19 (run
`20260819T150230Z`, `dgx:gpu0`, GB10, [#1375](https://github.com/mudler/vllm.cpp/issues/1375)).
Its configuration line reads `one_stage --device cuda`, 1024x576/25f, **"STG off
and modality 1.0 (the device arm refuses a perturbed forward, #1092)"**. So the
only arm that can hold the 21.004 B model ran with two of upstream's four
guidance terms pinned to their identity values:

| field | model default (`ltx2_pipeline.cpp:947-963`) | what the render had to use |
|---|---|---|
| `video_guider.stg_scale` | 1.0 | 0.0 |
| `audio_guider.stg_scale` | 1.0 | 0.0 |
| `video_guider.modality_scale` | 3.0 | 1.0 |
| `audio_guider.modality_scale` | 3.0 | 1.0 |

That is the same defect this row exists for, one residency down: a finite clip of
the right size and frame count, denoised along a different trajectory than
upstream's.

### 12.2 Scope

**In.** `Ltx2DitForwardDevice` grows the `const Ltx2DitPerturbation*` parameter
the host forward already has, and honours all four of its fields on the device
arm. The two refusals that stood in for it are removed. The refusals the HOST arm
carries for the same states are mirrored rather than dropped.

**Out.** Nothing new. The device arm inherits exactly the port boundary the host
arm already has: no `BatchedPerturbationConfig` partial blend, no batch > 1
(both degenerate at `Ltx2ModalityInput::batch == 1`), and the prompt-K/V cache
stays refused on this arm for the reason `ltx2_device.h` already gives.

### 12.3 Upstream chain, re-derived at the pin

Read at `fd4ded7f` in `/home/mudler/_git/LTX-2`, from the sentence making each
claim rather than out of a cited span:

| Upstream | Line | What it says |
|---|---|---|
| `ltx-core .../transformer/attention.py` | `:556` | `context = x if context is None else context` |
| | `:557` | `use_attention = not all_perturbed` |
| | `:559` | `v = self.to_v(context)` — computed on BOTH paths |
| | `:561-562` | `if not use_attention: out = v` |
| | `:564-565` | `q = self.to_q(x)`, `k = self.to_k(context)` — the skipped half |
| | `:573` | `out = out * perturbation_mask + v * (1 - perturbation_mask)` — the partial blend, unported, degenerate at batch 1 |
| | `:576` | the per-head gate, applied on BOTH paths |
| | `:579` | `return self.to_out(out)` — on BOTH paths |
| `ltx-core .../transformer/transformer.py` | `:268-269` | `run_a2v` / `run_v2a`, which test PRESENCE and not `enabled` |
| | `:335` | `if run_a2v and not video.cross_attn_skip_all` |
| | `:367` | `if run_v2a and not audio.cross_attn_skip_all` |
| `ltx-core .../transformer/transformer_args.py` | `:99-102, :112-118` | `BlockPerturbationsProcessor` — `cross_attn_skip_all` is `all_in_batch(cross_type, block)`, so at batch 1 it is the flag itself |
| `ltx-core .../transformer/model.py` | `:442-458` | which perturbation type reaches which stream: `SKIP_A2V_CROSS_ATTN` rides the VIDEO args, `SKIP_V2A_CROSS_ATTN` the AUDIO args |
| | `:493`, `:509-511` | the `perturbations` argument, and `None` meaning "perturb nothing" |
| `ltx-pipelines utils/denoisers.py` | `:100-137` | the four passes, in that order |
| | `:182-187` | the `BatchedPerturbationConfig` built over the pass list |

**Upstream does not share work between the legs.** `_guided_denoise`'s own
docstring says it "batches all guidance passes into one transformer call": it
`torch.cat`s the per-pass contexts, `_repeat_state`s the latent `n` times and
issues ONE forward at batch `n * B` (`denoisers.py:145-190`), then `chunk(n)`s
the result. The FLOPs are `n` times the conditional pass either way, and
`BatchSplitAdapter` is upstream's own documented option to "split this batch into
sequential chunks internally" (`denoisers.py:66-70`). This port runs the passes
sequentially at batch 1, which is that shape. Nothing is shared and nothing is
lost.

**And batch 1 is upstream's OWN DEFAULT, not a degenerate case of it.** Every
shipped pipeline declares `max_batch_size: int = 1` — `ti2vid_one_stage.py:149`,
`ti2vid_two_stages.py:176`, `ti2vid_two_stages_hq.py:191`,
`keyframe_interpolation.py:164`, `a2vid_two_stage.py:162`,
`t2a_one_stage.py:120`, `retake.py:170` — and the shared denoise helper both
declares the same default and ALWAYS wraps the transformer in the adapter:
`wrapped = BatchSplitAdapter(transformer, max_batch_size=max_batch_size)`
(`utils/blocks.py:514`, `:565`). So an unflagged upstream run splits the batched
call back into one forward per pass before the transformer sees it. Two
consequences the earlier argument left on the table:

- **`all_in_batch` IS the per-pass flag at batch 1.** It reduces to
  `(block_masks[type, block] == 0).all()` over the batch dimension
  (`perturbations.py:130-132`), so with one sample it is that sample's own bit.
- **The partial blend is unreachable at upstream's default.** The three-way
  self-attention path is selected by `(any_in_batch and not all_in_batch)`
  (`transformer_args.py:99-102`), which is false for every single-sample batch,
  and `attention.py:573`'s `out * mask + v * (1 - mask)` never runs.

So this port's all-or-nothing `all_perturbed` is upstream's EXECUTING path on
its own defaults, and the unported partial blend is an arm that upstream's
shipped configurations do not enter. That is a stronger claim than "degenerate at
batch 1" and it is the one the anchors support.

### 12.4 Design

**One parameter, mirroring the host signature.**

```
Ltx2DitOutputs Ltx2DitForwardDevice(vt::Queue&, const Ltx2DitParams&,
                                    const Ltx2DitWeights&, const Ltx2ModalityInput* video,
                                    const Ltx2ModalityInput* audio, vt::DType compute_dtype,
                                    Ltx2PromptKvCache* cache = nullptr,
                                    const Ltx2DitPerturbation* perturbations = nullptr);
```

Three mechanical changes behind it, each the device twin of a host site that
already exists and is gated:

1. `AttnArgsDev::all_perturbed`, read in `AttentionDev` as `attention.py:557`.
   The value projection is computed first (`:559`), and when the flag is set the
   function returns through a shared epilogue carrying `v` — so `to_q`, `to_k`,
   the q/k RMSNorms, RoPE and the scores never run, and the gate and `to_out`
   still do. The epilogue is FACTORED, not duplicated, for the reason
   `ltx2.cpp:825-829` already gives on the host arm: a perturbed pass that
   skipped `to_out` returns a tensor of the right shape in the wrong width-space
   and the block would add it to the residual and render.
2. `BlockArgsDev` grows the same four fields `Ltx2BlockArgs` has, and
   `BlockForwardDev`'s `if (run_a2v)` / `if (run_v2a)` become
   `if (run_a2v && !args.video_cross_attn_skip_all)` /
   `if (run_v2a && !args.audio_cross_attn_skip_all)`. The `vx_pre` / `ax_pre`
   snapshot stays OUTSIDE both guards, as `transformer.py:333`'s `vx_pre_av = vx`
   does, so a pass that skips one direction still lets the other read the
   pre-cross state.
3. `Ltx2DitForwardDevice` validates the vectors exactly as `ltx2_dit.cpp:837-842`
   does — neither empty nor `num_layers` long is refused — and fills the block
   args per layer.

**Two refusals go away and one arrives.**

- `ltx2_video.cpp`'s pre-loop `if (im.on_device && wants_perturbation) Fail(...)`
  is deleted, because the sentence it makes is no longer true.
- The `VT_CHECK(!im.on_device || p == nullptr, ...)` inside the x0 model is
  deleted for the same reason, and the argument it guarded is passed.
- `AttentionDev` gains the host arm's own refusal: `all_perturbed` on a CROSS
  call. `Ltx2DitPerturbation` has no cross-attention self-perturbation type, and
  a cross call carrying the flag would be served the self-attention rule.

Nothing is weakened into silence. Every state the host arm refuses, the device
arm refuses with the same text.

### 12.5 Cost, stated before it is discovered

**Forward count is EXACT. Wall clock is an UPPER BOUND.** `#1375` measured 60
forwards as structural for a 30-step `one_stage` render: 30 steps x 2, cond +
uncond, because the refusal pinned the other two guiders to identity. With the
model's own defaults every step assembles all four passes
(`denoisers.py:100-137`), so the same render becomes **30 x 4 = 120 forwards**.
That count is exact, it is derived from upstream's own pass list, and it is the
quantity this row changes.

The wall clock is not exact, and the arithmetic below it does not measure what it
looks like it measures. Pricing all four passes at the ~162 s/forward `#1375`
measured on GB10 at 1024x576 gives ~5.4 h of denoise where it was ~2.7 h. That
number is **at most 2.0x**, not 2.0x, because `#1375` timed a `cond` pass and two
of the four passes do strictly less work than a `cond` pass does:

- **`ptb` is close to full price.** At the `one_stage` defaults `stg_blocks =
  {29}` is ONE block (`ltx2_pipeline.cpp:955,961`), and in that one block the
  perturbed self-attention skips `to_q`, `to_k` and the attention itself and
  keeps only `to_v` (`attention.py:556-565`). One block's partial saving against
  a whole DiT rounds to nothing.
- **`mod` is materially cheaper.** Its two perturbations carry `blocks=None`
  (`denoisers.py:125-138`), and `blocks=None` means EVERY block
  (`perturbations.py:24`, `is_perturbed` at `:26-33`), so `mod` skips BOTH
  cross-modality directions — `audio_to_video_attn` and its V2A twin,
  `transformer.py:335`, `:367` — in every block of the forward.

So the true multiplier is above 1.5x and below 2.0x, and **nobody has measured
it**: this host has no accelerator and the only per-forward number in the record
is `#1375`'s `cond` cost. Quote the count, not the wall, until a leased run
prices a `ptb` and a `mod` pass separately. A render planned off 5.4 h is planned
off a ceiling, and 2.0x has already been repeated as if it were measured.

This is upstream's own default and correctness comes first, so it is not traded
away. The two extras that were the workaround remain the way to buy the time
back: `video_stg_guidance_scale=0` / `audio_stg_guidance_scale=0` drops the `ptb`
pass and `a2v_guidance_scale=1` / `v2a_guidance_scale=1` drops `mod`, taking the
render back to 60 forwards at the trajectory it has today. What changes is which
of the two is the DEFAULT.

**Device memory: no increase, and the direction is down.** A perturbed
self-attention allocates strictly FEWER device buffers than an unperturbed one —
no `qb`, `qn`, `kb`, `kn` and no `attn`; the value projection is reused — and a
skipped cross direction allocates none of its six. The passes run one at a time,
so the forward's peak concurrent scratch is unchanged for `cond`/`uncond` and
lower for `ptb`/`mod`. `#1375` recorded `OWN_DEMAND=74.907 GiB` and
`min_avail=39.97 GiB` for the two-pass configuration; the four-pass one asks the
device for no more than that. Not measured on hardware here — see §12.8.

**Host memory: two more pass slots.** `Ltx2GuidedDenoiseResult` already carries
four slots and the device arm fills two of them; it will fill four. The delta is
`2 x (video_tokens x out_channels + audio_tokens x audio_out_channels) x 4 bytes`
for the x0 tensors and the same again for the velocities. At the shipped latent
geometry that is single-digit MiB against a 21 B model, and it is the same
quantity the host arm has carried since `daeff67f2`.

### 12.6 Tests

The gate is the one `test_ltx2_device.cpp:486` already argues for: "Registering
on kCPU is what lets the whole device-forward CODE PATH be covered without a GPU,
so a GPU gates the KERNELS and not the port."

**T1 — the device arm tracks the HOST arm on every perturbation shape.**
`Ltx2DitForward` (f32, host) against `Ltx2DitForwardDevice` (f32, CPU backend) on
identical inputs, for five configurations: unperturbed, video self-attn on block
0, audio self-attn on block 0, `SKIP_A2V_CROSS_ATTN`, `SKIP_V2A_CROSS_ATTN`, and
the `mod` pass's own both-directions form. Each row asserts TWO things, because
either alone is passable by a defect:

- **agreement** — `max|device - host| < kDeviceRoundOff`, the same 2e-5 the
  existing host-tracking case uses;
- **non-vacuity** — `max|perturbed - unperturbed| > kDeviceRoundOff` on the
  stream the perturbation writes, so a device arm that accepted the argument and
  ignored it cannot pass by agreeing with a host arm it never diverged from.

The second half is the one that is RED before the change.

**T2 — each CROSS perturbation gates ITS OWN direction and no other.** The
device twin of §7.5, and it exists for the same measured reason: against the
host arm, mutations M12, M13 and M15 (apply one flag, or swap which flag gates
which direction) were all GREEN until a direct per-direction case existed.
Separation comes from upstream's own predicates (`transformer.py:268-269`): with
`audio->enabled = false` and the audio stream still PRESENT, A2V runs and V2A
does not, and the reverse. Each row asserts that the flag for that direction
MOVES the written stream and that the flag for the other direction leaves it
**bit-identical**, which is what makes the swap detectable and not only the
omission.

**T3 — the refusals.** A wrong-length perturbation vector is refused by name. A
CROSS call carrying `all_perturbed` is refused by name. Both mirror the host
arm's text.

**T4 — reachability, and its limit.** See §12.8. The end-to-end device case is
conditional on an accelerator backend and this host has none.

### 12.7 The mutations, and what each measured

Eleven, each carrying four facts: whether it BUILT, the compiler error count,
`git diff --stat` proving it applied, and the exit status captured directly. All
eleven built clean at `compile_err=0`. Every one restored the file byte-for-byte,
verified by sha256, **and rebuilt the target** — see the note below.

| # | Mutation | Gate | diff | Result |
|---|---|---|---|---|
| D1 | `Ltx2DitForwardDevice` accepts `perturbations` and IGNORES it — the pre-change state | `test_ltx2_device` | 1+/1- | **RED** exit 1, 2 cases / 31 assertions. `CHECK(mv > kDeviceRoundOff)`, `CHECK(ma > kDeviceRoundOff)`, `CHECK(av < kDeviceRoundOff)` |
| D2 | the device self-attention ignores `all_perturbed` | `test_ltx2_device` | 1+/1- | **RED** exit 1, 1 case / 12 assertions. `CHECK(mv > kDeviceRoundOff)` |
| D3 | the device arm ignores `video_cross_attn_skip_all` | `test_ltx2_device` | 1+/1- | **RED** exit 1, 2 cases / 9 assertions. `CHECK(bv < kBf16RoundOff)`, `CHECK(mv > kDeviceRoundOff)` |
| D4 | the device arm ignores `audio_cross_attn_skip_all` | `test_ltx2_device` | 1+/1- | **RED** exit 1, 2 cases / 7 assertions. `CHECK(MaxAbsDiff(v2a_off.audio, base.audio...) > 0.0)` |
| D5 | the device arm SWAPS which flag gates which direction | `test_ltx2_device` | 2+/2- | **RED** exit 1, 2 cases / 8 assertions. `CHECK(MaxAbsDiff(a2v_off.video, base.video...) > 0.0)` AND `CHECK(v2a_off.video == base.video)` — both halves, which is what separates a swap from an omission |
| D6 | the perturbed path returns `v` WITHOUT `to_out` | `test_ltx2_device` | 1+/1- | **RED** exit 1, 1 case / 6 assertions. `CHECK(av < kDeviceRoundOff)` |
| D7 | the perturbed path skips the per-head gate | `test_ltx2_device` | 1+/1- | **RED** exit 1, 1 case / 4 assertions. `CHECK(av < kDeviceRoundOff)` |
| D8 | the wrong-length vector check deleted | `test_ltx2_device` | 1+/2- | **RED** exit 1, 1 case / 2 assertions. `CHECK(msg.find("neither empty nor one entry per block") != npos)` and `REQUIRE(sentence(msg).size() > 20)` |
| D9 | the `vx_pre` / `ax_pre` snapshot moved INSIDE the direction guards | `test_ltx2_device` | 4+/0- | **RED** exit 1, 4 cases / 13 assertions. Caught by the PRE-EXISTING golden cases at `:338`, `:347`, `:356`, `:368`, `:380`, `:426` — `CHECK(r.audio < kDeviceRoundOff)` |
| D10 | the production call site drops `p` on the DEVICE branch alone | `test_ltx2_video` | 1+/1- | **GREEN** exit 0, 88 cases / 2755 assertions. The residual; §12.8 |
| D11 | the production call site drops `p` on BOTH branches — the reachability mutation | `test_ltx2_video` | 3+/2- | **RED** exit 1, 2 cases / 3 assertions. `CHECK(MaxAbsDiffOf(t.video_first_perturbed, t.video_first_cond) > 1e-6 * latent_span)`, the same for `video_first_modality`, and `CHECK(n.video_first_perturbed != n.video_first_cond)` |

**D11 had to be re-formed to be a result at all.** Its naive form — delete `p`
from both call sites — leaves the lambda parameter unused and
`-Werror=unused-parameter` refuses the build (`ltx2_video.cpp:3851`, 1 compile
error). A non-building mutation reads exactly like a passing test, so it was run
in the form that builds: `(void)p;` plus `nullptr` at both call sites. The
compiler diagnostic is a second, weaker instrument on the same defect and is
recorded as one rather than counted as the gate.

**Restoring the source is not enough, and this was measured.** After D1 restored
`ltx2_device.cpp` byte-for-byte, a direct run of `test_ltx2_device` still read
2 failed / 31 assertions, because the BINARY still carried the mutation. The
harness now rebuilds the target after every restore. A stale binary reports the
mutation as if it were the tree.

### 12.8 Reachability, and what this host cannot prove

**The production chain, by hand.** `vllm_video_engine_load` ->
`vllm::multimodal::LoadVideoEngine` -> `Ltx2VideoEngine::Load`, which sets
`im.on_device = params.device != 0` (`ltx2_video.cpp:745`) and stages the DiT
with `Ltx2StreamDitToDevice` (`:847`) -> `Generate` -> the phase loop ->
`Ltx2GuidedDenoise(x0_model, denoise_in)` -> the `x0_model` lambda ->
`Ltx2DitForwardDevice(..., p)`. Every hop is on the DEFAULT configuration of
`pipeline_kind = one_stage`, whose guiders are `stg_scale = 1.0` and
`modality_scale = 3.0`, so `p` is non-null on the `ptb` and `mod` passes without
any request extra.

**What a test can enter through, here, and what it cannot.** The chain is entered
end to end by the existing `one_stage` guidance case, and the perturbation
argument on it is gated: mutation D11 deletes `p` from the x0 model's argument
list and the focused gate goes RED. What is NOT executable here is the RESIDENCY
branch. `Ltx2VideoEngine::Load` refuses `device != 0` unless the platform seam
resolves a REGISTERED accelerator backend (`ltx2_video.cpp:749-758`), and it
refuses a partial backend beside that (`:779-788`). This build registers neither,
no CI job in `.github/workflows/ci.yml` has a GPU runner, and this row is
explicitly barred from taking a lease. So the ternary that selects
`Ltx2DitForwardDevice` over `Ltx2DitForward` always takes its second branch here,
and D10 — which drops `p` from the first branch alone — is GREEN.

"Not executable here" means in THIS binary as it is built. A dedicated test
executable that registers its own accelerator could enter that branch, and the
closing paragraph of this section says what that would take and what about it is
still unverified. It is deferred on cost, not closed on impossibility.

**That deferral ended on 2026-08-25, and everything above this line describes the
state before it.** The executable was built, the route worked, and §12.8.1 below
records the measurement. The paragraphs above are kept because they carry the
argument and the two unknowns that the build resolved; reading them as the
current position is the one mistake this note exists to prevent.

**Stated as a result, not as an omission.** Mutation D10 — delete `p` from the
device branch of the x0 model's ternary — is **GREEN** here (exit 0, 88 cases /
2755 assertions), and it is green for lack of hardware rather than for lack of a
test. Its sibling D11, which deletes `p` from BOTH branches, is **RED** (exit 1,
2 cases / 3 assertions). The three links, separately:

| Link | What it is | Gated by |
|---|---|---|
| A | entry point -> the x0 model, carrying `p` | the end-to-end `one_stage` guidance case, which enters through `LoadVideoEngine` + `Generate` — what `include/vllm.h`'s `vllm_video_engine_load`, the server and `ltx2-gen` all reach (`test_ltx2_video.cpp:365-372`). Mutation D11 deletes `p` from the lambda's argument list and it is RED. |
| B | the x0 model -> `Ltx2DitForwardDevice(..., p)` | **`tests/vllm/multimodal/test_ltx2_video_device_forward.cpp`**, which supplies the missing RUNNER and turns D10 RED. It was "nothing in this binary" until #1426; §12.8.1 records what it now proves and what stays owed. |
| C | `Ltx2DitForwardDevice` honours `p` | T1, T2, T3 — new, and RED before this change (D1-D9) |

**D11 is the reachability mutation this row can run, and it is RED.** Deleting
the perturbation from the production call site — the argument list the entry-point
chain reaches — reds the focused gate, so the test enters through the production
path rather than constructing the seam by hand. D10 narrows that to the one thing
it cannot cover: deleting `p` from the DEVICE branch alone, which no runner here
executes.

Note that `-Werror=unused-parameter` is a second, weaker instrument on D11: the
first form of that mutation did not build at all, because dropping `p` from both
branches leaves the lambda parameter unused. That is a compiler diagnostic and not
a gate, so D11 is run in the form that BUILDS — otherwise the mutation reads
exactly like a passing test.

**A synthetic accelerator is DEFERRED, not impossible.** The first draft of this
paragraph closed it on three numbered facts. One of them — the closed enum — is
true and is not a blocker. The other two are FALSE against the source, and they
carry three separate false claims between them. They are corrected below rather
than deleted, because a wrong reason is worse than an open item: the next reader
would trust it and not build the instrument. What the source actually says,
re-read at this head:

1. **The `kLtx2` glue table HAS a public registration API.**
   `vt::RegisterOp(OpId, DeviceType, void*)` is declared at
   `include/vt/op_provider.h:127` and is documented there as
   `RegisterOpProvider` at `kNativeProviderName` and priority 0 — a NATIVE
   registration. It is exactly the call `src/vt/cpu/cpu_ltx2.cpp:201-205` makes
   to install `&kKernels` for `kCPU`. `OpId::kLtx2` is public and
   `ltx2_device_resolve.cpp:11,15` already names it, so a test can read the CPU
   table back with `vt::GetOp(OpId::kLtx2, kCPU)` and register that same pointer
   for a second device type.
2. **`Glue()` does not return `nullptr`.**
   `src/vllm/model_executor/models/ltx2_device.cpp:81-86` `VT_CHECK`s
   `Ltx2DeviceKernelsAvailable` and refuses BY NAME. There is no null for the
   forward to dereference, so the failure mode the first draft named does not
   exist.
3. **`RegisterReferenceTier` is not `vt::`-op-specific.**
   `src/vt/op_provider.cpp:795-807` loops `for (size_t o = 0; o < kOpCount; ++o)`
   over EVERY OpId, and `MaybeInstallReferenceTier` (`:204-225`) copies
   `src->fn` verbatim with no op-kind discrimination — and `kLtx2`'s CPU `fn` IS
   the table pointer. What the tier does NOT do is satisfy this seam's own
   predicate: `Ltx2DeviceKernelsAvailable` reads `vt::OpRegistered`, which
   EXCLUDES the reference tier deliberately and by name
   (`op_provider.cpp:677-701`, "a FALLBACK, not a native kernel"). So the tier
   is the route for the `vt::` ops the staging and the forward need, and the
   direct `RegisterOp` of point 1 is the route for the glue table. Neither alone
   is enough. That is a design fact the next reader should have, not a reason to
   stop.

**The ENUM argument is true and is not a blocker.** `vt::DeviceType`
is a closed 7-value enum (`include/vt/device.h:16-28`), so a fake must
impersonate a real vendor and would flip every `CurrentPlatform()` answer in the
binary. The tree already solved that, for this same family:
`tests/vllm/multimodal/test_diffusion_device_seam.cpp` is a SEPARATE EXECUTABLE
for exactly this reason (`:19-23`, `tests/CMakeLists.txt:317`), and it already
registers a `FakeXpuBackend` whose `UnifiedMemory()` returns true (`:60-73`), a
platform in the XPU AND CUDA slots (`:116-121`), and a `device = 1` load of
`kLtx2VideoFamily` itself (`:127-133`). A separate executable leaves
`test_ltx2_video`'s 88-case baseline byte-for-byte untouched, which is the whole
of what that argument was protecting.

**The PREDICATE argument charges one measurement for another.** "It would prove
the predicate, not the residency" conflates link B with hardware residency. Link B as
the table above DEFINES it is "the x0 model -> `Ltx2DitForwardDevice(..., p)`",
and a fake accelerator DOES execute that call and would turn D10 RED. What a fake
cannot prove is that a GPU runs it. Both halves are owed; they are not the same
measurement, and `ltx2_video.cpp:749-758` refuses the substitution for the
SECOND one.

**What it would cost, and what is NOT verified.** A new test executable, for the
enum reason above, plus a fixture DiT staged through the fake unified-memory
backend — `tests/vllm/multimodal/ltx2_video_fixture.h` already writes a complete
reduced model. Two of the `Load` gates on that route were checked while writing
this and PASS by construction: `vt::RegisterBackend(kXPU, ...)` writes slot 0
(`src/vt/backend.cpp:81-84`), so the third check's `TryGetBackend(Device{kXPU,
0})` at `ltx2_video.cpp:800` resolves for `params.device = 1`; and
`vt::CreateQueue(Device)` takes its `entry.backend->CreateQueue()` arm
(`backend.cpp:146-153`), whose `q.device == device` assertion the fake already
satisfies. What is NOT verified is whether `Ltx2StreamDitToDevice`
(`ltx2_video.cpp:847`) stages the checkpoint onto that queue, and whether the
bf16 render that follows (`im.compute_dtype = kBF16`, `:812`) completes through
the reference tier. **Nobody has built this. The route is PLAUSIBLE, not
proven**, and stating it as proven would be the same defect as the three claims
above, one confident sentence in the other direction.

Link C is what did not exist at all before this section, and it is the whole of
what a box without a GPU can hold. Link B is one argument in one ternary whose
other branch IS gated, and it is listed under `## Owed` with the issue that owns
measuring it on hardware. This is the shape
[#1131](https://github.com/mudler/vllm.cpp/issues/1131) already records for
MiniMax-Music3 — "the DiT device arm's production SWITCH is unreachable from any
test" — and naming it is the difference between a staged link and dead code.

### 12.8.1 Link B, SOFTWARE half — CLOSED 2026-08-25 (#1426)

`tests/vllm/multimodal/test_ltx2_video_device_forward.cpp` is the runner §12.8
deferred, registered in `tests/CMakeLists.txt` beside
`test_diffusion_device_seam` and for the same process-global reason. It stands up
a synthetic accelerator — a `FakeUnifiedBackend` in the `kXPU` slot and a
platform registered over `kXPU` AND `kCUDA` — then loads `one_stage` with
`device = 1` through `LoadVideoEngine` and renders. The load, the per-tensor
staging and the whole bf16 render run to completion and write artifacts.

**Both unknowns §12.8 refused to assert are now MEASURED, and both work.**
`Ltx2StreamDitToDevice` stages the fixture checkpoint onto the fake queue, and
the bf16 render completes through the portable reference tier. The tier
announced itself in the run's own output, which is what makes this an
observation rather than an inference:

```text
[vt] first non-CPU vt::Conv3d dispatch (device type 4). This arm has never been
     run on real hardware; see issue #1452.
[vt reference-tier] op=Conv3d device=xpu has NO native kernel; running the
     PORTABLE CPU host kernel, ... It is SLOW: this run is not a performance
     measurement
```

**Two registrations, and neither alone is enough.** `vt::RegisterOp(OpId::kLtx2,
kXPU, vt::GetOp(OpId::kLtx2, kCPU))` installs the LTX glue table NATIVELY,
because `Ltx2DeviceKernelsAvailable` asks `vt::OpRegistered`, which excludes the
reference tier by name (`src/vt/op_provider.cpp::OpRegistered`, whose contract
comment says "a FALLBACK, not a native kernel" at `:756-757`).
`vt::RegisterReferenceTier(kXPU)` then covers every other op, and it runs SECOND
so it skips `kLtx2` (`src/vt/op_provider.cpp::RegisterReferenceTier`, the
`OpRegistered` `continue` at `:900`). One correction to §12.8's account: the
tier's eligibility is `DeviceMemoryIsHostAddressable()`, not `UnifiedMemory()` —
the two are different claims (`backend.h:59`) and GB10 is the box where they
differ — so the fake backend reports both, honestly, since its allocator is
`std::malloc`.

**That ORDER is a redundancy question, not a correctness one, and the earlier
claim here was refuted by mutation.** An earlier draft of this section, and of
the test's own comment, said the reverse order would leave `kLtx2` on a provider
`vt::OpRegistered` refuses to count, and that `Glue()` would then refuse the
forward by name. The swap was measured. It compiles clean, relinks the
executable, and exits 0 at 1 case / 30 assertions: nothing is refused. It is not
a no-op mutation either — `RegisterReferenceTier` returns **112** tier-first
against **111** native-first, and `vt::OpRegistered(kLtx2, kXPU)` reads **0**
immediately after the tier-first call, so `kLtx2` genuinely is covered by the
tier at that point. The `vt::RegisterOp` that follows installs the priority-0
native provider, `OpRegistered` reads 1, and `GetOp(kLtx2, kXPU)` is the CPU
pointer again, so `Ltx2DeviceKernelsAvailable` answers true either way. What the
reverse order actually costs is ONE redundant negative-priority fallback left
registered under `kLtx2` and outranked by the native entry — the exact waste the
tier's own skip comment says it exists to avoid.

**The mutation table was taken at `d0d4f1f6`, and D10 was RE-RUN on the merged
head.** The rerun is the same verdict on the same two assertions:
`compile_err = 0`, `libvllm.a` and the executable both relinked, exit 1,
1 case / 30 assertions with 2 failed, and the tree restored byte-for-byte
(`sha256` of `src/vllm/multimodal/ltx2_video.cpp` equal before and after,
`git diff` empty). So the table below is not stale against the pushed head.

**One recorded merge, not two.** `bc9f486f4`'s message says the branch merged
`origin/main` twice, and the tree cannot support that count:
`git log --graph origin/main..HEAD` contains exactly ONE merge commit,
`ba1235abf`. A second `git merge` that was already up to date leaves no commit
behind, so a second merge is UNVERIFIABLE rather than refuted — there is nothing
either way to read. Read the count as one recorded merge. The landed commit
message cannot be amended, which is why the correction lives here.

**The mutation table.** Both mutations built at `compile_err = 0`,
both relinked `libvllm.a` and both executables, and the tree was restored
byte-for-byte after each (`sha256` of `src/vllm/multimodal/ltx2_video.cpp` equal
before and after, `git diff` empty).

| Mutation | Diff | `test_ltx2_video_device_forward` | `test_ltx2_video` |
|---|---|---|---|
| none (this head) | — | **GREEN** exit 0, 1 case / 30 assertions | GREEN exit 0, 105 cases |
| **D10** — drop `p` from the DEVICE arm alone | 1 insertion / 1 deletion | **RED** exit 1, 1 case / 30 assertions, 2 failed | **GREEN** exit 0, 105 cases — unchanged, which is the defect this row filed |
| **reachability** — delete the device arm, leaving the host call | 2 insertions / 4 deletions | **RED** exit 1, 1 case / 10 assertions, thrown | not a clean read, see below |
| **instrument** — the new test's own `mp.device`, 1 -> 0 | 1 insertion / 1 deletion | **RED** exit 1, 1 case / 30 assertions, 1 failed | not run |

The third row is the instrument's own precondition, and it is the one no other
mutation covers. `CHECK(xpu.selections > 0)` is what says the DEVICE arm ran, and
nothing above proves that check can fail — the reachability mutation throws
before it is reached. Loading the same render on `device = 0` fails that ONE
assertion and leaves the other 29 green, including all three perturbation
differences, because the host arm honours `p` too. So the counter separates the
two arms rather than reporting a constant.

D10 fails on exactly the two assertions it is aimed at and on no others:
`|perturbed - cond|` and `|modality - cond|` both read `0` against a threshold of
`2.06e-06`. The unconditional arm is the CONTROL and stays non-zero under D10,
which separates "the perturbation did not arrive" from "the denoise loop died".

The reachability mutation reds by a THROW rather than by an assertion, and the
throw is itself the second statement: with the device arm deleted the render
enters `Ltx2DitForward` with `compute_dtype = kBF16` and gets
`"phase L2 ships only the f32 parity forward; the bf16 / FP8 / NVFP4 stream
dtypes are phase L6 and are refused"` (`ltx2_dit.cpp:766`). Nothing but the
device arm can serve this configuration, so the case cannot pass without the call
site.

`test_ltx2_video` under that same mutation is recorded as NOT A CLEAN READ, and
the reason is external. It reported exit 1 with 2 thrown cases and 4665
assertions, and both throws were file errors on its own `/tmp` workspaces
(`cannot write .../stg_empty/audio.wav`, `safetensors: open failed:
.../fixture/dit.safetensors`). Three other sessions were running
`test_ltx2_video` on this box at the time, one of them investigating
[#1885](https://github.com/mudler/vllm.cpp/issues/1885) with a sentinel on
`/tmp/vllm_ltx2_video_*`. The verdict is therefore withheld rather than
attributed to the mutation. It is also not needed: the focused gate for this
mutation is the new executable, and that read RED. The new executable's own
workspace prefix is `vllm_ltx2_dev_fwd_`, which does not collide with that glob.

**`test_ltx2_video`'s baseline is UNCHANGED by this addition, and the claim rests
on the CASE COUNT rather than the assertion count.** It reads 105 cases / exit 0
before and after, at `d0d4f1f6` and again after merging `origin/main`. That is
what the separate executable was for: this change adds no case to that binary.
The `88 / 2755` quoted earlier in this section is the figure at the head where
§12.8 was written, and the difference is other rows' cases, not this one's.

**The assertion count is NOT a stable baseline on this box, and
[#1885](https://github.com/mudler/vllm.cpp/issues/1885) reproduced here.**
TWELVE runs were taken across this row's heads and the count moved without a
diff. Four runs at `d0d4f1f6` read 4719. After `origin/main` was merged, seven
back-to-back runs of ONE binary over an UNCHANGED tree read 4721 once and 4719
six times. A twelfth run, later on the merged head, read 4716. The CASE count
read 105 on every one of the twelve. (`bc9f486f4`'s message says nine, and it
enumerates twelve in the same breath; twelve is the number that was taken, and a
landed message cannot be amended.) So a comparison made from the assertion count
alone would have reported changes that no diff caused, in both directions. Quote
the case count
when comparing across a change here, and treat an assertion delta as
unexplained until #1885 lands.

**Every `ltx2_video.cpp` line anchor in §12.8 above is STALE at `d0d4f1f6`, and
this section's `ltx2_video.cpp` anchors were re-derived and land.**
`im.on_device = params.device != 0` is at `:825`, not `:745`; the two `Load`
refusals are at `:829-838` and `:859-868`, not `:749-758`
and `:779-788`; `im.compute_dtype = kBF16` is at `:919`, not `:812`;
`Ltx2StreamDitToDevice` is called at `:962`, not `:847`; and the ternary is at
`:4325`. The old numbers are left where they are because §12.8 is the record of
what was measured then. Read them as history, and read these when you go to the
file.

**That claim first read "and this section's are not", and it did not hold.** A
fresh review checked the `op_provider.cpp` anchors written in the same breath
and found four of them off by 3 to 7 lines: `ReferenceTierEligible`'s predicate
is at `:888-889` and was cited `:884-885`; the tier's native-skip is at `:900`
and was cited `:896`; its OpId loop is at `:895-902` and was cited `:888-900`;
and `OpRegistered`'s "a FALLBACK, not a native kernel" is at `:756-757` and was
cited `:749-757`. All four are repaired, here and in the test file, and the
repaired citations are written as `src/vt/op_provider.cpp::<Symbol>`. That form
is the point rather than a flourish: `scripts/check-symbol-anchors.py`
validates a symbol citation and CANNOT see a bare line number, which is exactly
how four stale line anchors rode into a section that claimed its anchors were
fresh.

**The full gate on this head.** `ctest --test-dir build -j 4 --output-on-failure`
reports `100% tests passed, 0 tests failed out of 614`, exit 0, in 604 s, with
five pre-existing skips (`test_modelopt_mixed_precision_checkpoint`,
`test_minimax_music3_device_arm_real`, `test_minimax_music3_depth_arm_real`,
`test_voxtral_e2e`, `test_qwen35_paged_engine`). The new executable is test #77
and passes in 6.51 s; `test_ltx2_video` is test #74 and passes in 543 s.
`scripts/agent-preflight.sh` exits 0 with one failing gate,
`test_cpu_x86_llamacpp_floor`, which is
[#618](https://github.com/mudler/vllm.cpp/issues/618) and is load-dependent — the
box carried a load average above 90 throughout, from three concurrent sessions —
and with `commit-trailers` and `commit-style` SKIPPED because the branch was
behind `origin/main` at the time.

**The limit of this half.** A fake accelerator whose memory is host memory
answers "does the device arm execute, and does it honour its arguments". It
answers nothing about residency, bandwidth, a real kernel, or numerics on a GPU,
and `ltx2_video.cpp:829-838` refuses that substitution in as many words. That is
the hardware half, and §12.8.2 is where it was measured — by the OPERATOR, not
here. This row holds no lease authority and took no lease.

### 12.8.2 Link B, HARDWARE half — measured by the OPERATOR, 2026-08-25

**Attribution first, because it decides how to read this section.** The run below
was made by the operator inside an `rc` lease on `dgx:gpu0`. The helper that
wrote §12.8.1 did not run it, did not hold a lease, and has not independently
verified any figure in it. Every number here is transcribed from the operator's
report and its evidence directory. Read it as the operator's measurement with a
path a reader can check, not as a second measurement by this row.

**Provenance.** Built from `d0d4f1f60fc4765dd4118dead8bcc1778b1df9b1` — the exact
base of this branch — binary `sha256
bdeedd0143902fe806785ec4dc5fafe9d225276c07ce2835ea8c39581da707a3`, `CFG_RC=0`,
`BUILD_RC=0`, `compile_errors=0`, CUDA arch `sm_121a`, CUTLASS resolved. The box
was `ADMITTED` idle with 40.56 GiB headroom. Checkpoints staged to local disk,
`--checkpoint-class full`, `one_stage`, `--device cuda`, 768x448 at 25 frames,
seed 20260818. The accelerator announced `op=22 device=1 selected=vt-native`,
which is FA-2 and the shipped default. Evidence:
`/mnt/nas_share/rc/ltx25-ab-1510/out/20260825T120744Z/`, run id
`20260825T120744Z`.

**The counter this section's closing condition names.** The guided arm, on the
model's own guider defaults and no flags, issued **120 `dit forward` ticks across
30 distinct steps**, which is 4 per denoiser evaluation — cond, uncond, ptb, mod.
The unguided control (`--video-stg-guidance-scale 0 --audio-stg-guidance-scale 0
--a2v-guidance-scale 1.0 --v2a-guidance-scale 1.0`) issued **60 ticks across 30
steps**, which is 2 — cond and uncond. So `dit_forwards == 4 * dit_evaluations`
on hardware, through the production command-line entry point.

**And an A/B that the counter cannot give.** A count of four forwards does not
prove `p` was honoured: a dropped `p` still issues four. What a dropped `p`
implies is stronger and is testable. With no perturbation the `ptb` and `mod`
passes return exactly `cond`, so `stg_scale * (cond - ptb)` and
`(modality_scale - 1) * (cond - mod)` are identically zero and the guided
prediction collapses to `cond + (cfg_scale - 1) * (cond - uncond)` — which is
exactly what the unguided arm computes. At one seed, one binary and one geometry
the two arms would then be identical outputs.

They are not. The operator measured the two `audio.wav` files at RMS **-15.5593
dBFS unguided against -11.9614 dBFS guided, 3.598 dB apart**, envelope
coefficient of variation **0.1111 against 0.0872**, and video pixel means
**59.065 against 60.888**. That is the on-hardware analogue of D10, and it is an
observable rather than a count.

**The peak is TWO numbers, because it is two measurements, and an earlier draft
of this paragraph quoted one of them without saying which.** The verifier
DOWNMIXES the stereo track to mono before every headline metric, so the
`verify.json` files in the evidence directory record `peak_dbfs` **-4.0411**
unguided against **-0.6931** guided. The figures **-2.4621** against **-0.1530**
are the peak taken directly over the STEREO `int16` samples; they were handed
over labelled "(stereo)" and the qualifier was dropped in transcription. Both
are real, and both were recomputed from the two `audio.wav` files while
repairing this paragraph. A reader opening `verify.json` finds the mono pair
and only the mono pair, so quote the mono pair when citing that file. The RMS
and envelope CV above are the verifier's own mono figures and match
`verify.json` exactly.

**The guided artifact FAILED the harness's own audio check, and this A/B does
not rest on it passing.** The same `verify.json` this section quotes the
envelope CV from records `"audio_verdict": "FAIL"` and `"verdict": "FAIL"` for
the guided arm, on one failure: `envelope CV 0.0872 < 0.10 - amplitude never
varies (constant tone or stationary noise)`. The unguided arm passes both. That
threshold is a quality judgement on the artifact, not on the perturbation, and
the argument above is a DIFFERENCE between two arms rather than a claim that
either arm sounds good — a reader must not, however, meet the failure for the
first time in the file. It is the subject of
[#1510](https://github.com/mudler/vllm.cpp/issues/1510), "LTX-2.5 guided audio
is near-clipping and dynamically flat at the model's own defaults; video on the
same run is fine", whose row `LTX25-AUDIO-GUIDANCE-DEFAULTS` records the flat
envelope as MIRRORED UPSTREAM BEHAVIOUR rather than a port defect. That row had
not reached `main` when this was written — its spec is on
`row/LTX25-AUDIO-GUIDANCE-DEFAULTS`, not in this tree — so read the verdict as
that row's and check it there, not as a landed conclusion of this one.

**It does not replace §12.8.1, and saying why matters.** The leased run cannot
execute in continuous integration: there is no GPU runner, so nothing re-runs it
on the next change. The synthetic-accelerator executable is what keeps D10 red
from now on. The two halves answer different questions and neither substitutes
for the other, which is the distinction §12.8 spent its length establishing.

### 12.9 Stop conditions

- Stop and report `NEEDS_DECISION` if honouring the model defaults on the device
  arm is judged too expensive to be the default. The row does not make that
  trade itself: correctness first is the rule, and the two extras that buy the
  time back already exist and are documented.
- Stop if any perturbation shape cannot be served on the device arm. Refuse it by
  name and record it; never run an unperturbed forward where the recipe asked for
  a perturbed one.

### 12.10 The concurrent live-tick lane, and whether a perturbed forward is its own tick

`LTX25-RESIDENCY-W0-LIVE` ([#1413](https://github.com/mudler/vllm.cpp/issues/1413),
pull request [#1419](https://github.com/mudler/vllm.cpp/pull/1419), stacked on
[#1408](https://github.com/mudler/vllm.cpp/pull/1408)) adds `phase::Tick`
immediately ABOVE the `Ltx2DitForwardDevice` call this section edits — one stderr
line per DiT forward, emitted BEFORE the forward so a run killed inside forward 37
says 37. It is on `origin/row/LTX25-RESIDENCY-W0-LIVE` at `bca9b2911`, based on
`row/LTX25-RESIDENCY-W0`, and neither is on `origin/main` yet.

`git merge-tree` against this branch reports **exactly one conflict**, in
`src/vllm/multimodal/ltx2_video.cpp`, and it is textual rather than semantic:
their 24-line tick block lands at `:3984-4007` of their tree, in the same lines
this change deleted the `VT_CHECK(!im.on_device || p == nullptr, ...)` from.

**Resolution: keep both, tick first.** Take their whole tick block verbatim, drop
the `VT_CHECK` this change removes, and take this change's argument list on the
forward below it. The tick must stay immediately above the forward, which is the
ordering their spec argues for and this row has no reason to move. This change
contributes one argument to the call below and the deletion of a refusal above,
and nothing else in that lambda — deliberately, so the conflict stays resolvable
by inspection rather than by re-deriving either side.

**A perturbed forward is the SAME tick, not a separate one, and not an
attribute.** Three reasons, and the first is decisive:

1. **Their code already says so.** The tick's own comment refuses a denominator
   on the forward counter precisely because "`Ltx2GuidedDenoise` decides how many
   forwards each call is (one to four — cond, uncond, ptb, mod,
   `denoisers.py:100-137`)". It counts `im.trace.dit_forwards + 1`, which is the
   counter this row's passes increment. Nothing needs to change on either side.
2. **Upstream counts passes, not perturbation kinds.** `_guided_denoise` runs one
   transformer call over a batch of `n` passes and `chunk(n)`s it back; a
   perturbed pass is one row of that batch and not a different operation.
3. **The tick exists to project remaining time, and a non-uniform unit breaks
   that.** A perturbed forward costs almost the same as an unperturbed one: it
   skips `to_q`, `to_k`, the q/k RMSNorms and the scores of the perturbed
   self-attentions only, which is a fraction of one block, and the `mod` pass
   skips two cross-attention branches out of a block's six attentions. Splitting
   the counter would make `last=` compare unlike things.

**What this row DOES change for that lane, and it is the important half:** the
structural forward count. #1375's 60 (`30 x 2`) was a property of the
configuration the refusal forced, not of the sampler. On a guided arm at the
model's own defaults it is `30 x 4 = 120`, so a projection built on 60 will
under-read by half. Their tick is already denominator-free for this exact reason,
so nothing there is wrong — but any reader who has 60 memorised from #1375 now
has a stale number.

**What their lane will finally give this row:** `last=` on each tick is a
per-forward duration measured live, which is the number §12.5 currently has to
compute rather than measure. Once both land, one leased run reads the perturbed
and unperturbed forward costs directly and closes the §12.5 owed line.

## Now

§1-§11 landed at `daeff67f2`. [#1092](https://github.com/mudler/vllm.cpp/issues/1092)
stayed OPEN for the arm §1 recorded as owed: the device-resident `ptb` and `mod`
passes. §12 is that arm, and it is what this change carries. The row stays
`ACTIVE` rather than moving to `DONE`, because §12.8 link B and the §12.5 wall
clock are both open and both listed under `## Owed`. Neither is closed by this
change, and this change took no GPU lease. Be exact about what a lease is needed
FOR: the §12.5 wall clock needs one, and so does the HARDWARE half of link B,
but link B's SOFTWARE half — does anything at all enter the device branch of the
ternary — is closable on this box by a dedicated test executable with a synthetic
unified-memory accelerator. §12.8 says what that costs and which two steps of it
nobody has verified.

**Link B closed on 2026-08-25
([#1426](https://github.com/mudler/vllm.cpp/issues/1426)), in two pieces by two
sessions.** The SOFTWARE half is
`tests/vllm/multimodal/test_ltx2_video_device_forward.cpp`: the executable the
paragraph above describes exists, the two unverified steps both work, and
mutation D10 is RED on it (§12.8.1). The HARDWARE half is the operator's leased
`--device cuda` run (§12.8.2), which this row's helper neither ran nor verified.
The row still stays `ACTIVE`, because §12.5's wall clock is still unmeasured and
still needs a lease.
