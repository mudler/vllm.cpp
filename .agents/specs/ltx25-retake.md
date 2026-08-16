# LTX-2.5 — `RetakePipeline`, and the frame directory that was already in the ABI

Row: `LTX25-RETAKE`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned; **not
edited by this row**). Issue:
[#924](https://github.com/mudler/vllm.cpp/issues/924). Parent campaign issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Verified at the local checkout `/home/mudler/_git/LTX-2` before any anchor below
was taken: `git rev-parse HEAD` = `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
`git status --short` empty. Base: `origin/main` at
`0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`.

---

## 0. Honesty statement — what this row does and does not claim

This row serves **regenerating a chosen time window of an existing clip**. The
clip enters as a directory of `frame_%06d.ppm`, which is upstream's
**frame-folder** arm rather than its container arm, and the difference is
material enough to state before anything else: upstream reads containers with
PyAV (`media_io/decode.py:226`), no demuxer is vendored in this tree
(`src/vllm/entrypoints/openai/video_api.cpp:115-121` says so in as many words),
and a `.mp4` retake is therefore **refused by name** rather than attempted.

That is not a workaround invented here. Upstream carries a second ingestion arm
for exactly the case where there is no container: a directory of frames, whose
frame rate cannot be read and must be supplied
(`media_io/decode.py:213-215`), and which has **no audio stream at all**
(`utils/helpers.py:261-262` returns `None` before it opens anything). Every
consequence this row inherits — `fps` is required, `initial_audio_latent` is
absent, and the audio modality is therefore neither masked nor frozen — is
upstream's own behaviour on that arm, not a local simplification. Section 1.4
pins each one to its line.

**No render on real weights is claimed.** The GPU is out of bounds for this row
(a render ladder holds `dgx.casa`, which OOM-reboots under a second job), so the
gate is reduced-dimension fixtures. The real-checkpoint retake is owed and named
under `## Owed`.

**Upstream ships no tests at this pin.** `find /home/mudler/_git/LTX-2 -name
'test_*.py'` returns 0 lines; so does `find . -iname '*test*' -not -path
'./.git/*'`. "Port the upstream tests in the same change" therefore has nothing
to port, and the obligation becomes what section 5 does instead: pin upstream's
*behaviours* — its bounds arithmetic, its overlap polarity, its conform
polarity, its four-way modality plan — each against a `file:line` anchor, and
tie at least one assertion to a **local** fact so the suite can see a refusal
going stale.

### Two record corrections this row owes before it writes code

**`ref_video_dir` is not read by nothing.** The dispatching brief and #975 both
say so. It is read, fully, by MiniMax-H3:
`src/vllm/multimodal/minimax_h3_video.cpp:647,650` calls
`ReadReferenceClipChw` (defined `:135`), which walks `frame_%06d.ppm`. What is
true is narrower and is the sentence this row uses: **the LTX-2.5 engine never
reads its contents**, testing only emptiness at
`src/vllm/multimodal/ltx2_video.cpp:1566` in order to refuse. The stronger claim
is what #975's own message says (`ltx2_video.cpp:1608`), and it is wrong about
the tree rather than about LTX.

**The anchor in #924's body has drifted.** It cites the recipe refusal at
`ltx2_pipeline.cpp:1131-1136`. At `0e1bee42f` the chain is `:1159-1174` and the
refusal is `:1175-1179`. The issue body cannot be corrected without editing an
append-only record, so this spec carries the current anchor and the drift is
recorded here rather than silently re-derived by the next reader.

---

## 1. What upstream does, with anchors on both sides

### 1.1 The pipeline

`RetakePipeline.__call__` (`ltx-pipelines/src/ltx_pipelines/retake.py:151-329`):

| step | upstream | this row |
|---|---|---|
| refuse an empty or inverted window | `retake.py:211-212` | `Ltx2RetakeAssertWindow` |
| read the source geometry | `retake.py:220` -> `media_io/decode.py:203-237` | the frame directory's own frame count and PPM header |
| encode the clip to latents | `retake.py:238-247` -> `utils/helpers.py:165-233` | `Ltx2ConvVideoEncode` over the whole clip |
| conform the latent length | `utils/helpers.py:149-162` via `:233` | `Ltx2ConformLatentLength` |
| encode the audio | `retake.py:249-257` -> `utils/helpers.py:236-269` | **absent on the frame-folder arm** (`helpers.py:261-262`) |
| build the two modality specs | `retake.py:268-283` | `Ltx2RetakePlanModalities` |
| resolve sigmas | `retake.py:286-288` | the `retake` recipe's `DistilledSigmas()` |
| run one diffusion stage | `retake.py:313-324` | one phase, full resolution |
| decode video and audio | `retake.py:326-327` | the engine's existing decode and vocoder |

The constructor picks the scheduler at `retake.py:95-96`: `LTX2Scheduler()` is
built **only when `distilled` is false**, and `distilled` defaults to `True`
(`:85`) while the CLI hard-codes it (`retake.py:359`). So the served arm is the
distilled one, its schedule is `DISTILLED_SIGMAS`
(`retake.py:287`, `utils/constants.py:17,22`), and its denoiser is
`SimpleDenoiser` with no negative prompt and no guidance
(`retake.py:290-294`, and `prompts_to_encode = [prompt]` at `:259`).

### 1.2 `TemporalRegionMask` — the item, and its two coordinate conventions

`ltx-core/src/ltx_core/conditioning/types/noise_mask_cond.py:9-45`. It writes
`denoise_mask = 1` inside the window and `0` outside, in **patchified** space.

**Video** (`:30-35`). `get_patch_grid_bounds` returns *latent* `[start, end)`
bounds (`components/patchifiers.py:64-134`); the temporal patch size is fixed at
1 (`:14-18`), so latent frame `f` yields `[f, f+1)`. Those go through
`get_pixel_coords` (`:137-171`) with `causal_fix` **defaulting to `True` at the
call site** (`noise_mask_cond.py:33` reads
`getattr(latent_tools, "causal_fix", True)`) even though `get_pixel_coords`'s
own signature defaults it to `False` (`:140`). That disagreement is a trap: the
mask's default and the function's default are opposites, and taking the
function's would move every boundary by seven pixel frames. With
`scale_factors.time = 8` the causal rewrite `(x + 1 - time).clamp(min=0)`
(`:169`) gives frame `f` the pixel span `[max(8f-7, 0), 8f+1)`, so latent frame
0 spans one pixel frame and every later one spans eight. The result is divided
by `fps` (`noise_mask_cond.py:35`).

**Audio** (`:27-29`). `AudioPatchifier.get_patch_grid_bounds` already returns
**seconds** (`patchifiers.py:334-353` -> `:251-285` -> `:216-249`), so there is
no `get_pixel_coords` and no division. Latent frame `t` spans
`[max(4t-3, 0) * 160/16000, (4t+1) * 160/16000)`.

**The selection is a half-open overlap test**, identical on both sides:
`in_region = (t_end > start_time) & (t_start < end_time)`
(`noise_mask_cond.py:39`). Not containment. A token that straddles either edge
is **inside**.

**The write is `copy_`, not a conjunction** (`:44`). It replaces whatever the
mask held. A reading that ANDs it with an existing mask would silently keep any
earlier conditioning's zeros and is wrong.

### 1.3 `_conform_latent_length` — truncate **or** zero-pad

`utils/helpers.py:149-162`. Longer than expected: slice. Shorter: concatenate a
zero block on the frame axis. Retake calls it on both the video latent (`:233`)
and the audio latent (`:269`).

**This must not share a helper with A2Vid, and the reason is on the record.**
`A2VidPipelineTwoStage` truncates and never pads (`a2vid_two_stage.py:202`), and
`AudioLatentTools.create_initial_state` **asserts** the shape
(`ltx-core/tools.py:146-148`), so on that path a short take is an error rather
than a short latent — which is what row `LTX25-A2V-AUDIO-INPUT` mirrored as a
refusal (`.agents/specs/ltx25-a2v-audio-input.md`, and the reasoning is repeated
in `src/vllm/model_executor/models/ltx2_audio_input.cpp:209`). One shared helper
would have to pick a polarity and would be wrong for one of the two callers. So
this row adds a **second, differently named** function and says why beside it.

### 1.4 The frame-folder arm, and everything it decides

Upstream's `is_exr_dir` branch is not an exotic corner; it is the arm this row
lands on, and it settles four questions that would otherwise be guesses:

1. **The frame rate must be supplied.** `get_videostream_metadata` raises when
   `fps is None` for a folder (`media_io/decode.py:213-215`), because there is
   no container to read it from. Mirrored as a refusal naming the knob.
2. **There is no audio.** `audio_latent_from_file` returns `None` for a folder
   *before opening anything* (`utils/helpers.py:261-262`).
3. **Therefore the audio modality is neither masked nor frozen.** Both of
   retake's audio predicates are conjunctions with
   `initial_audio_latent is not None` (`retake.py:279`, `:282`), so with no
   audio latent the audio spec gets an empty conditioning list and
   `frozen=False` — the soundtrack is generated fresh. **`regenerate_audio` has
   no observable effect on this arm.** That is surprising, it is upstream's, and
   this row mirrors it rather than inventing a refusal upstream does not have.
   Section 5 pins it with a test so a later reader finds the behaviour asserted
   rather than assumed.
4. **The CLI's geometry validation still applies.** `retake.py:340-353` runs
   before the pipeline is constructed: `start_time >= end_time` refuses
   (`:340-341`), a frame count failing `8k+1` refuses **naming the snapped
   value** (`:347-351`), and a width or height not a multiple of 32 refuses
   naming both (`:352-353`). All three are mirrored, including the snapped value,
   because a refusal that does not say what would have worked costs a round trip.

### 1.5 The quantized arms

Upstream's inference quantization kinds are exhaustive and `assert_never`-closed:
`fp8-cast`, `fp8-scaled-mm`, `nvfp4-cast`, `nvfp4-prequant`
(`ltx-pipelines/utils/quantization_factory.py:23-26`, `assert_never` at `:50`).
Retake takes a `QuantizationPolicy` like every other pipeline
(`retake.py:83`, `:125`) and does nothing quantization-specific with it, so the
arm matrix for this row is exactly the engine's: whatever `Ltx2VideoEngine::Load`
already resolves for FP8 and NVFP4 serves retake unchanged, because retake adds
no new weight-bearing module. **GGUF is not applicable to this family** and the
reason is upstream's list above, not a local preference: there is no GGUF arm to
port, and refusing to state that would leave the next reader to re-derive it.

---

## 2. Scope

**In.** A new translation unit, `ltx2_retake.{h,cpp}`, carrying the four ported
pieces as pure functions over the shapes `ltx2_pipeline.h` already defines. A
`retake` row in `ResolveLtx2PipelineRecipe`. The minimum seam edit in
`ltx2_video.cpp`: the request knobs, the clip read, the encode, the conform, the
mask, and the seeding of the video stream's initial latent. Reachability through
`include/vllm.h`'s existing per-generation extras array and through `ltx2-gen`.

**Deliberately not a new ABI field.** The knobs ride the v18 parallel-array
extras that `include/vllm.h:917-927` documents as existing for this, and the
clip rides `vllm_video_params::ref_video` (`include/vllm.h:912`), which already
means "DIRECTORY of frame_%06d.ppm" and which the LTX engine has been refusing
rather than reading. No struct grows a field, so sibling rows editing that header
concurrently are unaffected.

**Out, and refused by name rather than dropped.**

| refused | why | upstream anchor |
|---|---|---|
| a container source (`.mp4`, `.mov`, …) | no demuxer is vendored | `video_api.cpp:115-121`; upstream uses PyAV at `decode.py:226` |
| an EXR-frame folder and any HDR colour space | no OpenImageIO, no scene-linear colour path | `retake.py:178`, `helpers.py:197-220`; declared out of scope by `.agents/specs/ltx25-retire-dead-arms.md:167` |
| `retake` against a `(kind, version)` pair the table does not carry | the table is exact and never defaulted | `ltx2_pipeline.cpp:1175-1179` |
| a retake request that also supplies `audio_path` | retake's audio comes from the source file only (`retake.py:250-256`); accepting both would silently pick one | — |
| a source clip whose frame count is not `8k+1`, or whose width or height is not a multiple of 32 | upstream refuses at the CLI stage | `retake.py:347-353` |

**Owed** (listed again under `## Owed`): a container demuxer, so the audio half
of retake can exist at all; the real-checkpoint render; and the sub-1.0
attention-strength arm, which this row does not touch.

---

## 3. Design

### 3.1 The `retake` recipe

One phase, at the source clip's own resolution, `DISTILLED_SIGMAS`, plain Euler,
no guidance, no negative prompt — each of those read off `retake.py` rather than
assumed:

- one `DiffusionStage` call (`retake.py:313-324`), so one phase;
- `spatial_downscale = 1` and no input transform, because `__call__` passes
  `output_shape.width` / `.height` straight through (`:317-318`);
- `sigmas = DISTILLED_SIGMAS` (`:287`);
- `EulerDiffusionStep()` and `euler_denoising_loop`, which is what
  `DiffusionStage.__call__` defaults to (`utils/blocks.py:524-527`) and retake
  overrides neither — **not** the ancestral sampler, which is `distilled.py`'s
  and reaches retake through nothing;
- `SimpleDenoiser`, so `negative_prompt` is empty and
  `allow_guidance_override = false` (`retake.py:290-294`).

`recipe.height` / `.width` / `.num_frames` / `.frame_rate` come from the
**source clip**, not from the params table, because upstream's do
(`retake.py:317-320`). The recipe therefore carries the params-table values as
defaults and the engine overrides all four from the clip, refusing a request that
also names a conflicting explicit geometry rather than silently preferring one.

### 3.2 Where the mask lands

The engine already builds the video denoise mask as all-ones at
`src/vllm/multimodal/ltx2_video.cpp:1995`
(`video.mask.assign(video.tokens, 1.0F)`), which is
`create_initial_state`'s `denoise_mask = ones` (`ltx-core/tools.py:158-161`).
`TemporalRegionMask` **replaces** that vector. Nothing else in the phase loop has
to change, and that is the design rather than a coincidence — every consumer of
`mask` already broadcasts it, as the A2V row's comment at `:2054-2081` works
through: `ApplyGaussianNoise` leaves the latent at `clean` where the mask is 0,
`TimestepsFromMask` yields per-token timestep 0 there, and `PostProcessLatent`
blends `clean` back. Seeding `video_initial` with the encoded source clip and
zeroing the mask outside the window is therefore the whole of the mechanism.

`video.clean = video.latent` at `:1994` is what makes the blend restore the
**source** rather than zeros, and it mirrors
`clean_latent = initial_latent.clone()` (`ltx-core/tools.py:156`).

### 3.3 The clip reader

**This section changed during implementation, and the change is recorded rather
than quietly made.** It first said: lift `ReadReferenceClipChw`
(`src/vllm/multimodal/minimax_h3_video.cpp:135`) out of MiniMax-H3's anonymous
namespace into a shared surface, because writing a second walker by hand is what
`AGENTS.md` `## Shared seams` forbids.

Reading both sides made that the wrong shared seam. H3's walker returns raw
`[-1, 1]` pixels at whatever size the files are; upstream's video ingestion runs
each frame through `video_preprocess` -> `resize_and_center_crop` to the target
height and width (`utils/helpers.py:228`, and `:202` says the EXR arm's
centre-crop is chosen to match it). This tree already has that chain, on the LTX
side, as `Ltx2LoadImageAndPreprocess` (`ltx2_image_preprocess.h:87`): decode ->
CRF -> f32 -> resize-and-centre-crop -> `/127.5 - 1`, returning exactly the
`[3, H, W]` layout `Ltx2ConvVideoEncode` takes. So the LTX seam to reuse is that
one, and `Ltx2ReadFrameDirectory` composes it per frame and transposes
frame-major to channel-major.

Lifting H3's walker would have shared the *file loop*, which is six lines, and
NOT shared the preprocessing, which is the part with an upstream anchor and the
part that fails silently. It would also have touched a second engine's
translation unit for no behavioural gain. The duplication that remains is a
`frame_%06d.ppm` probe-until-missing loop in two files, and it is named here so
the next reader can see it was weighed rather than missed.

`crf` is 0 and is not a parameter: `crf` is a knob on an *image* conditioning
input (`ImageConditioningInput.crf`, `blocks.py:977-983`) and upstream's video
ingestion path never applies one.

### 3.4 Where the request enters

Per-generation extras, refused by name when unknown at `ltx2_video.cpp:1359-1368`:

| key | meaning | upstream |
|---|---|---|
| `retake_start_time` | window start, seconds, inclusive | `retake.py:155`, `noise_mask_cond.py:19` |
| `retake_end_time` | window end, seconds, exclusive | `retake.py:156`, `noise_mask_cond.py:20` |
| `retake_frame_rate` | the folder's fps; required, no container to read it from | `decode.py:213-215` |
| `regenerate_video` | default `1` | `retake.py:164` |
| `regenerate_audio` | default `1` | `retake.py:165` |

The clip itself is `vllm_video_params::ref_video`. A window knob supplied without
a clip is refused, following the precedent the audio window knobs set at
`ltx2_video.cpp:1370-1382`: a knob that silently does nothing is the defect that
surface exists to prevent.

---

## 4. Risks

**The `READER ANCHORS` gate.** `ltx2_video.cpp:363-364` carries a line list
re-derived and compared by `test_ltx2_video`. Adding an `#include` at the top of
that file shifts every anchor, and a clean `git merge` will not warn. Mitigation:
re-derive at the final tree after the last merge of `origin/main`, and treat it
as a merge hazard in the PR body.

**A silently plausible render.** Every failure here returns a video. An
off-by-one in the causal rewrite shifts the window by seven pixel frames; the
opposite `causal_fix` default shifts it the other way; a containment test instead
of an overlap test drops the boundary tokens; a padded-instead-of-truncated
latent welds a black tail on. None is visible in a frame count or a token count.
Mitigation: value-level assertions on the mask itself at known boundaries, with
both `causal_fix` polarities distinguishable, plus a lower bound so an all-zero
or all-one mask cannot pass.

**A refusal going stale.** This campaign has six of those on the record
(`ltx2_video.cpp:1519-1543`). This row *lifts* one — the LTX side of the
`ref_video_dir` refusal at `:1566` — and it must lift only the retake half,
leaving the reference-conditioning arm (#975) refused, because that arm consumes
the same directory in a completely different way (downscale factor, temporal
subsample, appended tokens, a stage-1-only adapter). Mitigation: the refusal is
re-derived at current main and rewritten in the `WHAT IS *NOT* THE REASON` shape,
and one assertion is tied to a **local** fact — that the LTX engine now reads the
directory — so a test asserting only upstream symbol names cannot go quietly out
of date.

**Concurrent edits.** `ltx2_video.cpp`, `ltx2_pipeline.{h,cpp}`,
`docs/FEATURES.md`, `docs/USAGE.md` and `.agents/issue-index.md` are being
edited by sibling rows. Mitigation: new behaviour lives in a new TU; keyed
records are reapplied by key with unrelated keys proven byte-identical;
`MAX_CELL_CHARS = 220` binds, so this row trims **its own** wording rather than
any measurement or host qualifier.

---

## 5. Tests and evidence

Focused gate: `test_ltx2_video`, `test_ltx2_pipeline`, `test_capi`, and
`test_minimax_h3_video_fold` (because 3.3 moves code that engine calls).

1. **RED first, through the production entry point.** The smallest failing test
   drives `vllm_video_generate` with `ref_video` naming a frame directory and the
   retake window extras, and asserts the rendered latent is bit-identical to the
   encoded source outside the window and different inside it. It fails first
   because the extras are refused as unknown. Reachability is then proven the way
   [`reachability.md`](../reachability.md) requires: delete the production call
   site, rerun the focused gate, and record the RED.
2. **The two coordinate conventions, separately.** The video mask is asserted at
   its boundary latent frames against the spans section 1.2 derives, and the
   audio mask against the seconds section 1.2 derives. A single shared assertion
   would pass whichever convention were used for both.
3. **`causal_fix` polarity.** A case that distinguishes `true` from `false`,
   because the default disagrees between the call site and the function.
4. **Overlap, not containment.** A window strictly inside one latent frame's span
   selects that frame.
5. **Conform polarity, both halves.** A long latent is truncated from the front;
   a short one is zero-padded at the tail rather than refused — which is the
   half that distinguishes this helper from A2Vid's, and the half a reader who
   knows only A2Vid would get backwards.
6. **The four-way modality plan**, all four `(regenerate_video,
   regenerate_audio)` combinations, and the frame-folder consequence of 1.4.3
   asserted rather than assumed.
7. **Refusals, each asserting the missing part is named**: a container path, an
   inverted window, a frame count off the `8k+1` grid with the snapped value in
   the message, a resolution off the 32 grid, a missing frame rate, a window knob
   with no clip, and retake together with `audio_path`.
8. **The lifted refusal.** One assertion tied to the LOCAL fact that the LTX
   engine now reads `ref_video_dir`, so the suite can see the refusal go stale.

Every mutation records three facts: `git diff --stat` after applying, whether it
**BUILT** with the compile-error count, and the **exit code**. A mutation that
fails to build establishes nothing and is recorded as such. Doctest filters are
comma-free and the case count is asserted non-zero, because `-tc` splits on a
comma and runs unrelated cases to a green `SUCCESS`.

---

## 6. Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported with `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the pass/fail line, positive controls for `No space left` and
`BFD assertion`, the load average, and free disk.

---

## 7. Stop conditions

- Return `NEEDS_DECISION` rather than widening scope if serving retake turns out
  to require a second resident DiT or a phase-scoped adapter, which is #975's
  piece 2 and a different row's memory budget.
- Return `NEEDS_CONTEXT` if the frame-directory convention is not in fact the
  one `/v1/videos` and the C ABI agree on.
- Do not use the GPU. A render ladder holds `dgx.casa` and the box OOM-reboots
  under a second job.

---

## 8. What the mutation pass actually found

Focused gate on the unmutated merged tree: `test_ltx2_video --test-case=*retake*`
5 cases / 188 assertions, exit 0; `test_ltx2_retake` 4 cases / 69 assertions,
exit 0. Each mutation was applied to one file, built, run, then restored, and the
restore verified by `sha256sum` of both product files rather than assumed. Exit
codes were captured directly, never through a pipe. Filters are comma-free, and
the case count is printed beside every exit code, so a filter that matched
nothing cannot read as a pass.

| # | mutation | `git diff --stat` | BUILT | exit (video / retake) | verdict |
|---|---|---|---|---|---|
| M1 | delete the production call site `video.mask = Ltx2TemporalRegionMaskVideo(...)` | `ltx2_video.cpp \| 2 --` | YES, `compile_err=0` | 1 / 0 | DETECTED |
| M2 | never seed `video_initial` with the encoded source clip | `ltx2_video.cpp \| 2 +-` | YES, `compile_err=0` | **0 / 0**, then 1 / 0 | **SURVIVED**, then DETECTED |
| M3 | take `get_pixel_coords`' own `causal_fix=false` at the call site | `ltx2_video.cpp \| 2 +-` | YES, `compile_err=0` | **0 / 0**, then 1 / 0 | **SURVIVED**, then DETECTED |
| M4 | leave the frozen video stream at the schedule's scalar sigma | `ltx2_video.cpp \| 2 +-` | YES, `compile_err=0` | 1 / 0 | DETECTED |
| M5 | conform truncates only, never pads | `ltx2_retake.cpp \| 1 +` | YES, `compile_err=0` | 0 / 1 | DETECTED |
| M6 | containment instead of overlap | `ltx2_retake.cpp \| 2 +-` | YES, `compile_err=0` | 1 / 1 | DETECTED |
| M7 | ignore `initial_audio_latent is not None` in the audio predicates | `ltx2_retake.cpp \| 4 ++--` | **NO, `compile_err=1`**, then YES `compile_err=0` | — , then 0 / 1 | **NOT ESTABLISHED**, then DETECTED |
| M8 | drop the `!wants_retake` guard on the reference refusal | `ltx2_video.cpp \| 2 +-` | YES, `compile_err=0` | 1 / 0 | DETECTED |
| M9 | the `8k+1` refusal rounds the snapped value UP | `ltx2_retake.cpp \| 2 +-` | YES, `compile_err=0` | 1 / 1 | DETECTED |
| M10 | conform pads the whole buffer once instead of per channel | `ltx2_retake.cpp \| 4 ++++` | YES, `compile_err=0` | 0 / 1 | DETECTED |

**M2 and M3 survived on the first run, and that is the useful part of this
section.** Both were claims made in a comment and observed by nothing.

M2 is the sharper of the two. The suite recorded `retake_latent_absmax`, which
observes the ENCODE, and inferred from it that the latent reached the phase. It
does not follow: a build that reads the clip, encodes it, records the digest and
then starts the stream from zeros passes every other assertion and renders a clip
of the right length with the right mask. The repair renders two DIFFERENT sources
at the same seed and the same window and requires the pixels to differ.

M3 is narrower and is the one a reader would predict was already covered. The
unit case for `causal_fix` polarity existed and was already red under the flip —
but it calls the function directly, so it says nothing about the argument the
PRODUCTION call site passes. The end-to-end case used the window [0.05, 0.10),
which selects the same single latent frame under both polarities, so it could not
see the flip. A second window, [0.30, 0.40), separates them: one latent frame
with the causal rewrite, both without it.

M7 failed to build, because dropping `has_audio_latent` leaves a parameter unused
under `-Werror`. That is recorded as NOT ESTABLISHED rather than counted as a
pass, and re-run with the parameter still referenced.

### Reachability

[`reachability.md`](../reachability.md) asks two questions and they are answered
separately.

**Does a production entry point reach this?** `vllm_video_generate`
(`include/vllm.h:969`) -> `src/capi/vllm_c.cpp` -> `Ltx2VideoEngine::Generate`
(`src/vllm/multimodal/ltx2_video.cpp`), where the retake knobs are parsed, the
clip is read through `Ltx2ReadFrameDirectory`, encoded, conformed, and the mask
assigned into `video.mask` inside the phase loop. The clip rides
`vllm_video_params::ref_video` (`include/vllm.h:912`), an ABI field that already
existed and that this engine previously only tested for emptiness. `ltx2-gen`
reaches the same path on its default configuration through `--pipeline-kind
retake`, `--ref-video` and the three window flags, added here. `/v1/videos`
forwards no engine extras today (#928), so the CLI and the C ABI are what this
row claims and the HTTP surface is not.

**Does a test enter through it?** Yes — the five `test_ltx2_video` cases start at
`LoadVideoEngine` and `Generate`, not at `Ltx2TemporalRegionMaskVideo`. M1 is the
reachability mutation: deleting the production call site turns that suite RED
(exit 1) while `test_ltx2_retake` stays green (exit 0), which is exactly the
split the guide predicts and the reason the unit suite is not the proof.

---

## Owed

- **A container demuxer.** Without it the audio half of retake cannot exist:
  every audio predicate is a conjunction with `initial_audio_latent is not None`
  (`retake.py:279,282`) and a frame folder never produces one. Owned by this row,
  tracked by [#924](https://github.com/mudler/vllm.cpp/issues/924) until a
  demuxer row exists.
- **The real-checkpoint retake render.** The gate here is reduced-dimension
  fixtures; the GPU was out of bounds. Owned by this row.
- **The EXR / HDR arm** (`retake.py:178`, `helpers.py:197-220`), refused by name
  and declared out of scope by
  [`ltx25-retire-dead-arms.md:167`](ltx25-retire-dead-arms.md).
- **#975's reference-conditioning arm** stays refused and is not touched here.

## Now

The arm is implemented, gated and reachable, on one pull request that carries the
spec first. The campaign row `ROAD-V1-LTX25` does NOT change lifecycle state
here and is not edited: it stays `SPIKE` because the render on real weights it is
waiting for is still owed, and this row adds one more capability to the same
spike rather than completing it. `docs/STATUS.md` therefore has nothing to
record, which is why this change does not touch it.
