# LTX-2.5 — the token-APPEND seam, and the three arms one fixed-width loop blocks

Row: `LTX25-TOKEN-APPEND`
Issue: [#930](https://github.com/mudler/vllm.cpp/issues/930)
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644)
Pin: `Lightricks/LTX-2 @ fd4ded7fa` — verified at the working checkout
`/home/mudler/_git/LTX-2`, `git rev-parse HEAD` =
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`.
Base: `origin/main` at `bc6433d1bc4da3062b846b753cca6e57c20fcb41`.

## 0. Why this is a row and not a line inside another one

Two rows hit the same wall hours apart without talking to each other. Row
`LTX25-IC-LORA` filed [#930](https://github.com/mudler/vllm.cpp/issues/930) from
the IC-LoRA port; row `LTX25-GENERATED-KEYFRAMES` filed
[#920](https://github.com/mudler/vllm.cpp/issues/920) from the generated-slots
port. Both stopped at the same sentence: *this engine's phase loop is fixed at
one `Ltx2VideoTokenCount(vshape, 1)`*. A limitation two independent rows reach
by different routes is a shared seam, so it gets its own spec.

**Sizing correction, carried deliberately.** #930's own accounting names **two**
blocked arms. There are **three**, and the row that wrote that number has asked
for the correction rather than the inheritance: #930 predates #920. The three
are the reference-video / reference-image arm, the last-frame keyframe arm, and
generated keyframe slots. This spec is the record that supersedes the two.

## 1. What upstream does, with anchors on both sides

A conditioning item that *appends* grows the token sequence, the loop runs over
the grown sequence, and a trim on the way out restores the target grid. Three
pieces, and this tree has the first one already.

### 1.1 The append — ported, and not the gap

| upstream | ours |
|---|---|
| `VideoConditionByKeyframeIndex.apply_to` — `ltx-core/conditioning/types/keyframe_cond.py:36-90`, concatenation at `:79-82` | `Ltx2ConditionVideoByKeyframe` |
| `VideoConditionByReferenceLatent.apply_to` — `ltx-core/conditioning/types/reference_video_cond.py:46-108`, concatenation at `:97-100` | `Ltx2ConditionVideoByReference` |
| `AudioConditionByReferenceLatent.apply_to` — `ltx-core/conditioning/types/reference_audio_cond.py:33-65` | `Ltx2ConditionAudioByReference` |

All three already concatenate onto `latent`, `clean`, `mask` and `positions`
through the shared `AppendTokens` helper in
`src/vllm/model_executor/models/ltx2_conditioning.cpp`, and all three are gated
against executed upstream by `scripts/gen-ltx2-vae-goldens.py` section 9. The
items are not what is missing.

### 1.2 The per-token marker the append must carry — MISSING

`extend_keyframes_mask` (`ltx-core/conditioning/mask_utils.py:74-105`). Every
appending item calls it, and upstream's own docstring says why: *"Every
conditioning item that appends tokens must call this, otherwise the per-token
marker goes out of sync with the token sequence."* `marked=False` for both video
items ported here (`keyframe_cond.py:85-86`, `reference_video_cond.py:103-105`);
`marked=True` has exactly one upstream caller, `VideoGeneratedKeyframeSlots`
(`keyframe_slots.py:121`), which is #920's arm and not this row's.

`Ltx2LatentState` carries no `keyframes_mask` at all today, so the engine holds
it beside the state on `StreamState::keyframes_mask` and an append would
desynchronise the two. This row moves the field onto `Ltx2LatentState` and
extends it inside `AppendTokens`, which is the only place that can guarantee the
call upstream's docstring demands.

### 1.3 The trim — MISSING

`LatentTools.clear_conditioning` (`ltx-core/tools.py:88-117`), called at
`ltx-pipelines/utils/blocks.py:576` and `:579`, immediately before `unpatchify`.
It truncates `latent`, `clean_latent` and `positions` to
`self.patchifier.get_token_count(self.target_shape)`, replaces `denoise_mask`
with `torch.ones_like(...)[:, :num_tokens]` — **ones, not the original mask** —
and drops `attention_mask` and `keyframes_mask` to `None`.

The engine's comment at the unpatchify site already says *"There are no
conditioning tokens on this path, so the clear is the identity."* This row makes
it not the identity.

### 1.4 The attention mask — NOT the gap, and this is load-bearing

Both ported appending video items call `update_attention_mask` with a literal
`attention_mask=None` (`keyframe_cond.py:68-76`, `reference_video_cond.py:88-96`),
and `update_attention_mask` (`mask_utils.py:110-143`) returns `None` when the
argument is `None` and `latent_state.attention_mask is None`. The only upstream
route to a non-`None` mask is `ConditioningItemAttentionStrengthWrapper`, whose
sole application site is `ltx-pipelines/iclora_utils.py:169` on the IC-LoRA
path — `combined_image_conditionings` (`ltx-pipelines/utils/helpers.py:272-308`),
which is the route this engine mirrors, never wraps.

So `Ltx2LatentState` still grows **no** attention-mask field. Adding one here
would be a field no ported item can ever populate, which is the unpassed-parameter
shape `.agents/reachability.md` enumerates. The refusal messages that cite the
absent field stay literally true; what changes is that the absence stops being
offered as a blocker.

### 1.5 The sigma schedule reads the TARGET count, never the grown one

`LTX2Scheduler.execute` (`ltx-core/components/schedulers.py:21-57`) derives its
shift from `tokens = math.prod(latent.shape[2:])` — the **unpatchified** target
latent's `F*H*W`, which by construction cannot see appended tokens. And the
pipelines compute `sigmas` before the state exists at all: `ti2vid_one_stage.py:207`
calls `self._scheduler.execute(steps=num_inference_steps)` with no latent, and
`distilled.py:200-201` uses frozen `DISTILLED_SIGMAS` constants.

This engine calls `Ltx2SigmaSchedule(steps, video.tokens)` *after* the
conditioning block. Today that is the target count because nothing appends.
The moment something appends, that line silently re-shifts the whole schedule.
It is the one behaviour-preserving-today edit this row must make anyway, and it
gets its own mutation.

## 2. Scope

**Build the seam. Lift one arm as the demonstration. Leave the other two
refused.**

In scope:

* `Ltx2LatentState::keyframes_mask`, `Ltx2ExtendKeyframesMask`, and the call to
  it from `AppendTokens`.
* `Ltx2ClearConditioning`.
* The engine's phase loop: a `target_tokens` that the sigma schedule and the
  trim both read, a grown `video.tokens` through the DiT, and the trim before
  `Ltx2VideoUnpatchify`.
* The **last-frame keyframe** arm, lifted, driven from
  `VideoGenParams::last_frame_path` **alone**. There is no `last_frame_ppm`
  field: `video_engine.h:90-91` declares `first_frame_path, last_frame_path` and
  a `first_frame_ppm` for the server's `data:` URLs, and the in-memory
  alternative exists on the first-frame side only.

Out of scope, and refused as today:

* **Reference video / reference image.** At this row's base, `row/LTX25-IC-LORA`
  (PR [#938](https://github.com/mudler/vllm.cpp/pull/938)) is **open and
  unmerged**, so `--lora` does **not** read the IC-LoRA scale factors here and
  `git log --grep '#923'` is empty. The refusal at `ltx2_video.cpp` naming
  `downscale_factor` / `temporal_scale_factor` in LoRA metadata is **still the
  true and current cause** on this base. #930's body describes that refusal as
  already rewritten onto token-append; **that is not the state of `origin/main`
  at `bc6433d1b`.** This row therefore leaves that refusal byte-identical.
  Rewording it to name only token-append would ship the exact defect a sibling
  row nearly shipped — a refusal naming the first of two causes.
* **Generated keyframe slots** (#920, PR
  [#929](https://github.com/mudler/vllm.cpp/pull/929), also open). Needs
  `marked=True`, `GeneratedKeyframeLayout`, and a standalone one-frame decode
  per slot (`types.py:269-273`). The seam this row builds is its prerequisite,
  not its implementation.
* **Reference audio.** Blocked on the audio VAE encoder key filter, untouched.
* Any `include/vllm.h` growth. `last_frame` already exists on the ABI
  (`vllm.h:910`); nothing new is exposed.

## 3. Design

### 3.1 `ltx2_conditioning.h` / `.cpp`

```
Ltx2LatentState  += std::vector<float> keyframes_mask;   // types.py LatentState.keyframes_mask
Ltx2ExtendKeyframesMask(state, num_new_tokens, marked)   <- mask_utils.py:74-105
Ltx2ClearConditioning(state, target_tokens)              <- tools.py:88-117
```

`AppendTokens` calls `Ltx2ExtendKeyframesMask(..., marked=false)` before it grows
`state->tokens`, mirroring the fact that upstream passes the **pre-append**
`latent_state`. `Ltx2CreateVideoLatentState` fills `keyframes_mask` from the
existing `Ltx2FirstFrameKeyframesMask`, which is `tools.py:184`'s own
`replace(state, keyframes_mask=...)`.

Empty vector is upstream's `None`. `extend_keyframes_mask` returns `None` when
there is no existing mask and `marked` is false, and zero-fills a fresh mask when
there is no existing mask and `marked` is true; both branches are mirrored, so
#920's arm finds the function it needs already correct.

### 3.2 `ltx2_video.cpp`

The engine keeps `StreamState` and converts at the conditioning boundary, as it
already does for the first-frame arm. Two local converters replace the existing
hand-rolled six-line copy so there is **one** statement of the mapping rather
than two. `StreamState::positions` is `double` and `Ltx2LatentState::positions`
is `float`; the round trip is exact because the engine builds those doubles by
widening floats (`ltx2_video.cpp`, the `video.positions[i] = temporal ? ...`
line), and that is stated at the converter.

```
target_tokens = Ltx2VideoTokenCount(vshape, 1)   // fixed, per phase
video.tokens  = target_tokens                    // then GROWN by any append
sigmas        = Ltx2SigmaSchedule(steps, target_tokens)   // §1.5
...denoise over video.tokens...
Ltx2ClearConditioning(&state, target_tokens)     // §1.3
Ltx2VideoUnpatchify(...)
```

The last-frame arm mirrors `combined_image_conditionings`
(`helpers.py:272-308`): `frame_idx == 0` takes `VideoConditionByLatentIndex`,
anything else takes `VideoConditionByKeyframeIndex`. The last frame of the output
is pixel frame `frames - 1`, and `num_pixel_frames` stays at upstream's default
of `1`. It shares the first-frame arm's CRF resolution, strength polarity and
encoder-shape checks, because upstream shares them too — one loop over
`images`, one `load_image_and_preprocess`, one `video_encoder(image)`.

## 4. Risks

| risk | why it is real | what catches it |
|---|---|---|
| The sigma schedule silently re-shifts once something appends | the call site sits after the conditioning block and reads `video.tokens` | mutation M4: point it back at `video.tokens` and the last-frame render must move |
| `keyframes_mask` desynchronises from the token count | it lives on `StreamState`, the append lives on `Ltx2LatentState` | the engine's existing `VT_CHECK` on `keyframes_mask.size() == video.tokens` fires; mutation M2 |
| The trim is forgotten or trims to the wrong count | `Ltx2VideoUnpatchify` would read past the target grid or reshape appended tokens into pixels | mutation M3; and the unpatchify's own size arithmetic |
| A blind pixel witness reads as a weak-but-real effect | this campaign has already had exactly that: every arm identical *including* the control, which is what made it diagnosable | the witness carries a **no-op control arm** (§5) |
| The trace cannot see any of it | `Ltx2ConditioningTrace` is filled **before** denoise, so it cannot observe the loop | the witness is on **rendered artifact bytes**, never on the trace |

## 5. Tests

Upstream ships **no tests**: `find /home/mudler/_git/LTX-2 -name 'test_*.py'`
returns 0 across the whole repository at the pin, confirmed at this row. So there
is no suite to port and every case below is written **against upstream anchors**
instead — each one cites the `file:line` that justifies the behaviour it
asserts. The bar does not move: each fails for the intended reason first, and
the engine-level cases enter through `LoadVideoEngine` / `VideoEngine::Generate`,
which is the production entry point.

1. **`test_ltx2_vae` — the seam, at the unit.** `Ltx2ExtendKeyframesMask` on
   both `marked` polarities and on the empty-mask branch (`mask_utils.py:74-105`).
   `Ltx2ClearConditioning` restores the target token count, restores an
   **all-ones** mask rather than the conditioned one (`tools.py:104`), trims
   positions per dimension, and drops the keyframes mask (`tools.py:113`).
   `AppendTokens` keeps `keyframes_mask.size() == tokens` across an append.
2. **`test_ltx2_video` — the arm, through the entry point.** A last-frame
   keyframe renders instead of refusing, and the refusal case's last-frame
   subcase is replaced by a served-arm case. The reference-video and
   reference-audio subcases stay exactly as they are.
3. **`test_ltx2_video` — the pixel witness, with a no-op control.** Three renders
   through `Generate`, byte-compared on the artifact:
   - `noop` — no image, no keyframe: nothing is appended.
   - `kf_a` — a last-frame keyframe.
   - `kf_b` — a *different* last-frame keyframe.

   `kf_a != noop` proves the append reached the maths. `kf_a != kf_b` proves the
   appended **content** reached it rather than merely the token count. And
   `noop == noop` re-rendered proves the instrument is not simply noisy. All
   three arms identical *including* the control would mean the instrument is
   blind, which is the reading this control exists to separate — without it a
   blind instrument reads as a real-but-subtle effect, and that is the direction
   that ships.
4. **Token count observed, not assumed.** The last-frame render's grown count is
   `target + tokens_per_latent_frame`, and the finished latent is back at
   `target` — otherwise `Ltx2VideoUnpatchify` could not have produced the frame
   count the artifact carries. Asserted through the artifact's frame count.

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Focused: `ctest --test-dir build -R 'ltx2' --output-on-failure`.

No GPU. This row's evidence is CPU-only and needs none: the seam is host-side
sequence bookkeeping, and the fixture DiT is reduced-dimension.

**Mutation discipline.** Every mutation records three facts — `git diff --stat`
after applying it, whether it **BUILT** with the compile-error count, and the
**exit code**. A mutation that fails to build establishes nothing and is re-run
compiling; a harness that reports exit status alone cannot tell "the guard is
load-bearing" from "the compiler rejected my edit", because a mutation that never
builds prints no test failures at all. A thrown doctest case prints `0 failed`,
so the exit code is the authority. The tree is restored byte-for-byte with a
`sha256` check after each.

Planned mutations:

| id | mutation | expected |
|---|---|---|
| M1 | reachability: delete the `Ltx2ConditionVideoByKeyframe` call site in the engine | RED |
| M2 | drop the `Ltx2ExtendKeyframesMask` call from `AppendTokens` | RED |
| M3 | make `Ltx2ClearConditioning` the identity | RED |
| M4 | point the sigma schedule back at `video.tokens` | RED |
| M5 | append with `marked=true` instead of `false` | RED |

Two more were added by the fresh review, and both were **GREEN** on the first
implementation. They are recorded here because each names a guarantee this row
advertises, and a guarantee no mutation can move is a comment rather than a
gate:

| id | mutation | expected |
|---|---|---|
| M6 | `Ltx2ClearConditioning` SLICES the mask instead of restoring all ones | RED |
| M10 | the last-frame arm's `frame_idx` becomes `0` instead of `frames - 1` | RED |

**Why M6 was green.** `Ltx2CreateVideoLatentState` already fills every target
token's mask with `1.0`, and the append writes `1 - strength` only at the TAIL,
which the trim drops under either implementation. So over the range the test
walked, "restore all ones" and "slice" produce identical bytes. The repair
conditions a token INSIDE the target first, with
`Ltx2ConditionVideoByLatentIndex` at `latent_idx = 0`, which writes
`1 - strength` at `start .. start + count` (`latent_cond.py:41`); a slice then
leaves `0.4` at token 0. The case also `REQUIRE`s that value BEFORE the trim, so
a future change that stops arming the instrument fails loudly rather than
returning the case to gating nothing.

**Why M10 was green.** Both renders still differed from the no-op control and
from each other, and the token count was identical, because a keyframe pinned to
the FIRST frame appends exactly as many tokens as one pinned to the last. The
pixel witness can see THAT an append happened and WHAT was appended; it cannot
see WHERE. The repair asserts the first appended token's temporal position at
the engine, recomputed from `frames` and `fps` rather than read back from the
`frame_idx` argument, so the check and the thing it checks are independent
expressions.

## 7. Stop conditions

* Stop and report `NEEDS_DECISION` rather than lifting the reference-video arm:
  its second cause is another row's, and PR #938 is open against it.
* Stop rather than growing `include/vllm.h`; the arm's fields already exist.
* Stop rather than editing `docs/FEATURES.md` or `docs/USAGE.md` outside the
  keys this row's arm changes.

## 8. Owed

* **Reference video / reference image** — needs the IC-LoRA metadata scale
  factors. Owned by row `LTX25-IC-LORA`, issue
  [#930](https://github.com/mudler/vllm.cpp/issues/930) for the append half,
  PR [#938](https://github.com/mudler/vllm.cpp/pull/938) for the metadata half.
  This row removes the append half of that blocker and touches neither the
  refusal nor its test.
* **Generated keyframe slots** — needs `marked=True`, `GeneratedKeyframeLayout`,
  and per-slot standalone decode. Owned by row `LTX25-GENERATED-KEYFRAMES`,
  issue [#920](https://github.com/mudler/vllm.cpp/issues/920).
* **Reference audio** — needs the audio VAE encoder key filter. Owned by the
  campaign, [#644](https://github.com/mudler/vllm.cpp/issues/644).
* **`Ltx2ExtendKeyframesMask`'s `marked=true` branch** lands with a unit driver
  and **no production caller**: the only upstream construct that passes `true` is
  `VideoGeneratedKeyframeSlots`, which is #920's arm. Declared here rather than
  discovered later, per `.agents/reachability.md` `## Landing a slice that is not
  reached yet`. Owner: row `LTX25-GENERATED-KEYFRAMES`, issue #920.

## 9. What the mutation pass could NOT reach

Recorded because a reviewer should press on it rather than rediscover it.

**The sigma-schedule binding has a residual.** `schedule_tokens` is a local that
feeds both `Ltx2SigmaSchedule` and the trace, so the ordinary mutation — changing
what the local is initialised from — moves both and REDs (M4). A mutation that
edited only the *call argument* and left the local alone would not be caught by
that field. Nothing local can close this: the instrument and the thing it
measures would have to be the same expression, and then it would measure nothing.
The pixel witness does not close it either, because it has no
correct-schedule render to compare against.

**The trim is gated on a guard, not on pixels.** Appended tokens sit at the tail
of a contiguous `[tokens, width]` buffer and `Ltx2VideoUnpatchify` takes a bare
pointer, so an un-trimmed state unpatchifies the same head bytes and renders
pixel-identical frames. M3 therefore REDs on the `VT_CHECK` at the unpatchify
boundary rather than on any output difference. That is the honest description:
the trim's correctness on this engine's path is an invariant this row asserts,
not a difference this row can observe.

**`Ltx2ExtendKeyframesMask(marked=true)` has a unit driver and no production
caller.** See `## Owed`.

## 10. Now

`ACTIVE` — spec committed before implementation; implementation, docs and tests
on the same branch; PR [#948](https://github.com/mudler/vllm.cpp/pull/948) open.

A fresh review returned `FAIL` on six findings and all six are repaired. The
seam itself survived: every upstream design decision checked out and the pixel
witness reproduced exactly. What did not survive was the evidence around it —
two advertised guarantees had no mutation that could move them (§6, M6 and M10),
four records carried anchors that resolve to real-but-different upstream
statements, and `origin/main` advanced to `c2019b0e3` (#935) mid-review so the
branch stopped merging. The anchor sweep was redone from the claims rather than
trusted, and it found two more the review had not: a citation to
`ltx2_recipes.py`, which exists neither upstream nor here, and a bare `:100-101`
whose nearest named file was the wrong one.
