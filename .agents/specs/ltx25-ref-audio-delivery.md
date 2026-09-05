# LTX-2.5 — reference-AUDIO conditioning delivery (A18)

Row: `LTX25-REF-AUDIO-DELIVERY`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue: [#2634](https://github.com/mudler/vllm.cpp/issues/2634).
Plan: [`ltx25-completion-scope.md`](ltx25-completion-scope.md) §8 order **10**,
gap **A18**, itself issue
[#2526](https://github.com/mudler/vllm.cpp/issues/2526).

Base read: `origin/main` at **`2850314e3`**. The dispatch named `be12dc71a`;
`main` had moved before the first anchor below was taken, and every anchor in
this document was re-derived at `2850314e3` with `git show origin/main:<path>`.
The review repairs applied afterwards touch four anchors —
`ltx25-a2v-audio-input.md:24` and `:86` (§3.1),
`ltx25-completion-scope.md:545` (§0) and
`ltx2_conditioning.h:61-63` (§2.1) — and each of those four was re-read at
`origin/main` **`e24805924`**, which is where they hold.

Upstream pins:

| Reference | Registry id | Revision |
|---|---|---|
| Lightricks/LTX-2 (`ltx-core` + `ltx-pipelines` + `ltx-trainer`) | `ltx-2` | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Identity asserted, not assumed: `/home/mudler/_git/LTX-2` has `origin`
`https://github.com/Lightricks/LTX-2.git`, a clean worktree and
`HEAD = fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, which is the revision
[`.agents/oracles/ltx-2.md`](../oracles/ltx-2.md) pins. Every upstream anchor
below was read with `git show fd4ded7f:<path>` inside that clone.

---

## 0. VERDICT FIRST — A18 as scoped is NOT a clean port

**Read this section before reading any other.** This row returns
`NEEDS_DECISION`. It does not recommend implementing A18 as
`ltx25-completion-scope.md` §4 states it, and §1 is written as the decision it
owes rather than as a scope to fund.

The scope document sizes A18 as **"M — the op is ported and undriven"** and
gives it exactly one dependency, order 7 (A19, which landed as
[#2583](https://github.com/mudler/vllm.cpp/issues/2583)). Three verified facts
contradict that sizing.

**1. Upstream has no delivery from a standalone reference-audio file, and the
refusal's own second anchor points at a different mechanism.** At `fd4ded7f`,
`git grep -n "AudioConditionByReferenceLatent"` returns nine lines in five
files, and the nine enumerate as **4 + 1 + 1 + 3**: four `__init__` re-export
lines across two files (`ltx-core .../conditioning/__init__.py:6` and `:16`,
`.../conditioning/types/__init__.py:8` and `:12`); one class definition
(`.../conditioning/types/reference_audio_cond.py:12`); one documentation line
(`packages/ltx-pipelines/CLAUDE.md:85`); and three inside `dubit.py` alone — its
own import (`:13`), the return annotation of `build_audio_ref_conditioning`
(`:266`) and the constructor call (`:272`).
**Exactly one site constructs it**, and it is
`packages/ltx-pipelines/src/ltx_pipelines/dubit.py:272`, inside
`DubItPipeline` — gap **A3**, which this tree does not ship
(`include/vllm/model_executor/models/ltx2_pipeline.h:876`: "`DubItPipeline`,
which this tree does not ship"). The cross-check agrees: `git grep -n
"conditionings=audio\|audio_conditionings"` over `packages/` returns
`dubit.py:277`, `dubit.py:294`, `dubit.py:305`, `dubit.py:323` and two lines of
`ltx-trainer`'s validation runner. Every other pipeline hands its audio
`ModalitySpec` no conditionings at all — `a2vid_two_stage.py:251-256` and
`:291-296` freeze the stream on an `initial_latent` instead, which is the arm
this tree already serves.

`utils/helpers.py:264-269`, the second anchor in both the refusal
(`src/vllm/multimodal/ltx2_video.cpp:2939`) and the scope table, is
`audio_latent_from_file`. Its **only** caller at the pin is
`packages/ltx-pipelines/src/ltx_pipelines/retake.py:250`, and there it produces
retake's *initial audio latent*, not a conditioning item. It never touches
`AudioConditionByReferenceLatent`. The two anchors the refusal pairs describe
two unrelated mechanisms, and pairing them is what made A18 look like a wiring
job with a file at one end.

**2. Where upstream does build the item, the source is not a waveform the caller
supplied.** `DubItPipeline` takes `reference_video_path` and encodes **that
clip's own audio stream** (`dubit.py:276`, calling `:186-191`, which raises
`ValueError(f"No audio stream found in {video_path}")` when the clip is silent).
The same path also feeds the IC-LoRA *video* reference conditioning
(`dubit.py:169-183`), so the audio and video references are one argument, not
two. Stage 2's reference is not a file at all: it is the **stage-1 audio
latent**, cloned at `dubit.py:298` and rebuilt into a second item at `:305`.
Serving `gen.ref_audio_path` into the ordinary phase loop would therefore
deliver an item upstream never constructs, from a source upstream never uses, on
a pipeline upstream never applies it to.

**3. The token geometry is pipeline-specific, and upstream says so in terms.**
`patchify_dubit_audio_reference_latent` (`dubit.py:335-354`) shifts every
position by `-(aud_dur + 0.04)` when `negative_positions=True` (`:351-353`), and
`packages/ltx-pipelines/CLAUDE.md:85` describes it as "Dub-It-only **audio**
patchify/negative positions in `dubit.py`". `ltx-trainer`'s validation runner
appends a reference-audio block with **plain, non-negative**
`get_patch_grid_bounds` positions instead
(`packages/ltx-trainer/src/ltx_trainer/validation_runner.py:759-784`). Two
upstream consumers, two different position conventions, neither of them a
default. There is no "the" reference-audio geometry to port; there is DubIt's
and there is the trainer's.

**What A18 actually is.** `Ltx2ConditionAudioByReference` is a correctly ported,
correctly gated leaf of **A3**. It is undriven here because A3 is unported, in
the same way that `Ltx2ConditionVideoByReference` is a leaf of A15. Splitting it
out as its own row produces one of two outcomes, and both are worse than not
splitting it:

* Implement it against DubIt's semantics without DubIt, and the delivery is
  unreachable in the only shape upstream gives it (no IC-LoRA, no reference
  video, no stage-1 latent to be stage 2's reference). That is the shape
  `.agents/reachability.md` calls *the flag with no default path*.
* Implement it against `gen.ref_audio_path` on the distilled phase loop, and the
  engine grows a conditioning mode upstream does not define. AGENTS.md's rule is
  "mirror vLLM"; the completion scope's own class-A test is "does upstream
  construct this?". This construction fails that test, and the scope document
  already carries the class that failure names:
  `.agents/specs/ltx25-completion-scope.md:545`, "**Class E — the one gap that
  fails class A's own test**", opened for A12 on precisely this ground. Under
  (b) or (c) the reclassification target is therefore **E**, not class **B** —
  B is for a correct mirror of an absent upstream feature, and a delivery
  upstream never performs is not a mirror of anything. Under (a) the question is
  moot: the obligation moves into A3, which carries its own class, and A18 stops
  being a classified gap.

**This is the A8 failure again.** A8 was sized `S` because a second consumer was
never counted ([#2584](https://github.com/mudler/vllm.cpp/issues/2584)). A18 is
sized `M — the op is ported and undriven` because the *delivery site* was never
read: the scope table quotes `reference_audio_cond.py:34-65` (the item) and
`utils/helpers.py:264-269` (an unrelated loader) and never quotes a constructor,
because at this pin the only constructor is inside a pipeline the same document
lists two rows above as unported.

**The decision owed.** One of:

* **(a) Withdraw A18 as an independent row.** Fold the delivery into A3
  (`LTX25-DUBIT`, §8 order 9) and record `Ltx2ConditionAudioByReference` as owed
  *by that row*, which is where its only upstream caller lives. This row
  recommends (a). It costs nothing today, removes a row whose gate cannot be
  two-sided, and puts the op behind the entry point that will actually reach it.
* **(b) Fund A18 as an engine-defined extension** — reference-audio conditioning
  from `ref_audio_path` on the distilled/one-stage phase loop, with no upstream
  referent. Then it needs a ratified product decision naming which position
  convention it takes and why, an explicit divergence note, and reclassification
  out of class A. §4 below is what that would cost, written out so the decision
  is priced rather than guessed.
* **(c) Repair the refusal and the scope record only.** Keep the refusal but
  correct its anchors and its account of what is missing (§5), and correct the
  A18 row to say "a leaf of A3". This is a records change and rides in whichever
  pull request lands the decision.

§1 through §10 are written for **(b)**, because a decision needs a priced
alternative. Nothing below authorises starting it.

---

## 1. Scope

**In (only under decision (b)):** construct an `AudioConditionByReferenceLatent`
equivalent from an LTX-2.5 generation request, apply it to the engine's audio
stream inside the phase loop, carry the appended tokens through the DiT forward,
and trim them before the audio unpatchify — reached from `include/vllm.h`.

**In, unconditionally:** the finding in §0, the corrected anchors in §5, and the
refusal-text repair they imply.

**Out:**

* `DubItPipeline` itself (A3) — its recipe, its IC-LoRA arm, its two-stage
  audio freeze, its `--reference-video` surface.
* The IC-LoRA reference-*video* conditioning (A15, A16, A17) that DubIt's stage
  conditionings need.
* N-adapter LoRA fusion (A17). `dubit.py:364-365` requires exactly one `--lora`,
  and `src/vllm/model_executor/models/ltx2_lora.cpp:256` already records that.
* Any change to the `a2vid_two_stage` frozen-audio arm, which is a different
  mechanism and already lands.

---

## 2. Upstream anchors

All at `Lightricks/LTX-2 @ fd4ded7f`. Paths are relative to the repository root.

### 2.1 The conditioning item

`packages/ltx-core/src/ltx_core/conditioning/types/reference_audio_cond.py`

| Anchor | What it fixes |
|---|---|
| `:12-22` | `class AudioConditionByReferenceLatent`. Docstring: patchified reference latent `[B, T_ref, C]`, positions `[B, 1, T_ref, 2]`, "1.0 keeps reference clean". |
| `:24-32` | Constructor. `strength: float = 1.0` is the **default**; `positions` is cast to `torch.float32` on the way in. |
| `:36-41` | `denoise_mask` for the appended block is `1.0 - strength`, filled, shape `(*tokens.shape[:2], 1)`. |
| `:43-51` | `update_attention_mask(attention_mask=None, num_noisy_tokens=patchifier.get_token_count(target_shape), ...)`. With no incoming mask **and no mask already on the state** this resolves to `None` (`mask_utils.py:141-143`); with an existing mask on the state it instead pads the new tokens with ones and rebuilds the 2-D mask (`mask_utils.py:144-156`). This tree carries the same two-part condition at `include/vllm/model_executor/models/ltx2_conditioning.h:61-63`. |
| `:54-57` | Concatenation order: `latent` gets `zeros_like(tokens)`, `clean_latent` gets `tokens`, `positions` concatenate on `dim=2`. The reference tokens go at the **end**. |
| `:59-61` | `keyframes_mask=extend_keyframes_mask(latent_state, tokens.shape[1], marked=False)` — called unconditionally, "to keep the invariant that every appending conditioning item extends the per-token fields". |
| `:62-64` | `generated_keyframe_layout`, `generated_keyframes` and `frozen` pass through unchanged. |

### 2.2 The only inference-side constructor

`packages/ltx-pipelines/src/ltx_pipelines/dubit.py`

| Anchor | What it fixes |
|---|---|
| `:1` | Module docstring: "Two-stage Dub-It pipeline with IC-LoRA and appended audio reference conditioning." |
| `:186-191` | `_encode_reference_audio_vae_latent(video_path)`: `decode_audio_from_file`, **raise** `ValueError` when the clip has no audio stream, then `vae_encode_audio(audio, enc, None)`. |
| `:266-272` | `build_audio_ref_conditioning`: patchify with `negative_positions=True`, then `AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=1.0)`. |
| `:276-277` | Stage 1's reference is the **reference video's** audio stream. |
| `:292-295` | Stage 1 audio `ModalitySpec(context=..., conditionings=audio_conditionings)` — not frozen. |
| `:298`, `:305` | Stage 2's reference is the **stage-1 audio latent**, cloned. |
| `:321-327` | Stage 2 audio is `frozen=True, noise_scale=0.0, initial_latent=s1_audio_latent` **and** carries a reference item. |
| `:331` | The delivered soundtrack is decoded from the stage-1 latent, not stage 2's. |
| `:335-354` | `patchify_dubit_audio_reference_latent`: `AudioPatchifier(patch_size=1)`, `get_patch_grid_bounds` over `AudioLatentShape(batch=b, channels=c, frames=seq_len, mel_bins=mel_bins)`, cast to `float32`. |
| `:351-353` | The negative shift: `positions - aud_dur - 0.04`, with `aud_dur = positions[:, :, -1, 1].max()`. |
| `:364-365` | "Dub-It requires exactly one `--lora`". |

`packages/ltx-pipelines/CLAUDE.md:85` names `patchify_dubit_audio_reference_latent`
as **Dub-It-only** and states the negative positions are "training-compatible".

### 2.3 The divergent second consumer

`packages/ltx-trainer/src/ltx_trainer/validation_runner.py:733-784` — the
training-validation runner appends a reference-audio block inline (not through
the class) with **plain** `get_patch_grid_bounds` positions (`:763-766`) and a
**zeroed** denoise mask (`:768-773`). It is driven by a dataset
`sample.conditions` entry, not by a request.

### 2.4 What the refusal's second anchor really is

`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:236-269`,
`audio_latent_from_file`: `:261-262` returns `None` for an EXR directory,
`:263` defaults `max_duration` to `output_shape.frames / output_shape.fps`,
`:264-266` returns `None` when the file carries no audio stream, `:267` calls
`encode_audio`, `:268-269` conforms to
`AudioLatentShape.from_video_pixel_shape(output_shape).frames`. Its **only**
caller at the pin is `packages/ltx-pipelines/src/ltx_pipelines/retake.py:250`.

`packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:249-274` is
`encode_audio`: builds an `AudioProcessor` from the encoder's own
`sample_rate` / `mel_bins` / `mel_hop_length` / `n_fft` when none is passed
(`:263-269`), runs `waveform_to_mel` (`:271`), then the encoder (`:273`).

---

## 3. The local side, as it stands at `2850314e3`

### 3.1 The refusal

`src/vllm/multimodal/ltx2_video.cpp:2934` is the guard
(`if (!gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {`); the `Fail`
body runs `:2935-2946` and the block closes at `:2947`. Both the guard string
and the unpatchify line cited in §3.4 occur **exactly once** in the file
(`grep -c` returns 1 for each), so the anchors are unique, not merely present.

The completion scope cites `ltx2_video.cpp:2754-2766`, and
`.agents/specs/ltx25-a2v-audio-input.md` cites `ltx2_video.cpp:1404` at its
`:24` and again at its `:86`. That anchor is **not unique in that file**
(`grep -c` returns 2 at `origin/main`), so both of its lines are named here
rather than one. Both citations are stale, by 180 and 1530 lines. The scope
document also cites `ltx2_pipeline.h:837` for the DubIt comment; it is at
`:876`. Anchors in this
tree drift within weeks, which is why §7 gates on a symbol grep rather than a
line number.

### 3.2 The op

* Declaration: `include/vllm/model_executor/models/ltx2_conditioning.h:308-310`,
  under a comment at `:304-307` anchoring `reference_audio_cond.py:33-65`.
* Definition: `src/vllm/model_executor/models/ltx2_conditioning.cpp:615-622`.
  It validates non-null state and `width == state->width`, then delegates to
  `AppendTokens`.
* `AppendTokens` is `src/vllm/model_executor/models/ltx2_conditioning.cpp:51-96`.
  It checks `tokens.size() == token_count * width` (`:54-55`) and
  `positions.size() == pos_dims * token_count * 2` (`:56-57`), calls
  `Ltx2ExtendKeyframesMask(state, token_count, marked=false)` **before** the
  resize (`:75`), appends `1.0 - strength` to the mask (`:79-80`) and
  concatenates positions **per dimension** (`:85-94`). This is a faithful
  `reference_audio_cond.py:36-61` and needs no change.
* `Ltx2CreateAudioLatentState`:
  `include/vllm/model_executor/models/ltx2_conditioning.h:153`,
  `src/vllm/model_executor/models/ltx2_conditioning.cpp:472`.

**Both are test-only.** The only call sites in the tree are
`tests/vllm/models/test_ltx2_vae.cpp:3454`, `:3469`, `:3562`
(`Ltx2CreateAudioLatentState`) and `:3581`
(`Ltx2ConditionAudioByReference`). `.agents/specs/ltx25-a2v-audio-input.md`
already records this under `## Owed` — "`Ltx2CreateAudioLatentState` and
`Ltx2ConditionAudioByReference` remain test-only" — and names the reason §4.1
prices below. (That entry's own anchors, `test_ltx2_vae.cpp:2412, :2431`, have
also drifted.)

### 3.3 The template A18 was meant to follow

The audio-to-video encode path in the same function:

* `src/vllm/multimodal/ltx2_video.cpp:1643` loads the encoder half through
  `Ltx2AudioVaeEncoderKeyRules()`.
* `:3175` declares `a2v_audio_volume`; `:3182` reads the `audio_path` extra.
* `:3198-3207` refuses a recipe that requires a take when none was supplied.
* `:3208-3266` is the block: `:3209-3217` refuses a decoder-only checkpoint,
  `:3221-3226` derives `target_frames`, `:3231-3242` resolves and validates
  `start_time` / `max_duration`, `:3246-3248` decodes the WAV, `:3250-3252`
  calls `Ltx2EncodeAudioToLatent`, `:3259-3264` checks `channels` and `mel_bins`
  separately rather than by product, `:3265` publishes the volume.

Everything from file bytes to an audio latent therefore already exists and is
reached. §0 point 1 is why that is not the missing half.

### 3.4 What the phase loop would have to grow

* `struct StreamState` is `src/vllm/multimodal/ltx2_video.cpp:181-192`. It is a
  **different type** from `Ltx2LatentState`
  (`include/vllm/model_executor/models/ltx2_conditioning.h:94` onward): `double`
  positions against `float`, no `pos_dims`, and `keyframes_mask` present on
  both. A bridge exists — the comment at `:194` onward records row
  `LTX25-TOKEN-APPEND` ([#930](https://github.com/mudler/vllm.cpp/issues/930))
  building it in both directions — but it was built for and exercised by the
  **video** stream.
* The audio state is built at `:3833-3903`: `width` at `:3834`, `tokens` at
  `:3835`, patchify at `:3872`, `clean` at `:3873`, mask at `:3901`, positions
  from `Ltx2AudioPatchTimings` at `:3902-3903`. Nothing between `:3833` and
  `:3904` applies a conditioning item.
* **There is no audio-side trim.** `target_tokens` is video-only
  (`:3753`, `const int64_t target_tokens = Ltx2VideoTokenCount(vshape, 1);`),
  `Ltx2ClearConditioning` is called on the video state alone (`:5045`), and the
  post-loop invariant at `:5071-5072` asserts only `video.tokens ==
  target_tokens`. The audio stream is unpatchified straight out of the live
  buffer at `:5090`,
  `audio_latent_volume = Ltx2AudioUnpatchify(audio.latent.data(), ashape);`
  — with `ashape` describing the *target* audio latent. Appending reference
  tokens without adding an audio trim would hand `Ltx2AudioUnpatchify` a longer
  buffer than its shape argument describes.
* The DiT forward takes the live count (`:4614`, `ain.tokens = audio.tokens;`),
  which is correct for an appending item and needs no change.

### 3.5 The ABI

Reachable, and reached today only to be refused:

* `include/vllm.h:1097`, `const char* ref_audio;`.
* `src/capi/vllm_c.cpp:1682`, `gen.ref_audio_path = OrEmpty(params->ref_audio);`.
* `include/vllm/multimodal/video_engine.h:97-98`
  (`ref_audio_path`, `ref_audio_wav`); the HTTP surface fills them at
  `src/vllm/multimodal/video_engine.cpp:376-381`.
* A second LTX-2.5 refusal reads the same fields on the text-to-audio path,
  `src/vllm/multimodal/ltx2_video.cpp:5885-5891`, and that one **mirrors
  upstream correctly** (`t2a_one_stage.py` passes `video=None`). It must not be
  touched.

So the entry point exists. What does not exist is a *meaning* for it on this
architecture — which is §0's finding, not a wiring gap.

---

## 4. Design, priced for decision (b)

Only if the developer selects (b). Four parts, in order.

### 4.1 The `StreamState` <-> `Ltx2LatentState` bridge for audio

Reuse the `LTX25-TOKEN-APPEND` mapping at
`src/vllm/multimodal/ltx2_video.cpp:194` onward with `pos_dims = 1` instead of
3, and prove the audio round trip is exact the way the existing comment proves
the video one. The audio positions are produced by `Ltx2AudioPatchTimings`
(`:3902`) as `float` and widened to `double` at `:3903`, so the narrowing back
is exact for the same reason the video path's is. **This is the cost the scope
document's "M — the op is ported and undriven" does not name**, and
`ltx25-a2v-audio-input.md`'s `## Owed` already called it "its own change".

### 4.2 The reference latent

Encode the caller's waveform on the path §3.3 already runs, without the
`target_frames` conform: the reference is not conformed to the target duration
upstream (`dubit.py:191` returns the encoder output directly; the conform at
`helpers.py:268-269` belongs to the *initial* latent, not the reference). Refuse
a decoder-only audio VAE with the message at `:3209-3217` rather than a second
one.

### 4.3 The positions

**This is the decision inside the decision.** DubIt shifts by
`-(aud_dur + 0.04)` (`dubit.py:351-353`); the trainer does not
(`validation_runner.py:763-766`). Under (b) the row must pick one, say which
upstream site it took it from, and record the divergence — because upstream
applies the DubIt convention *only* on the DubIt pipeline, and this row is not
that pipeline. A row that silently takes DubIt's shift is claiming a
correspondence it cannot demonstrate.

### 4.4 The trim

Add an audio `target_tokens` beside the video one at `:3753`, and trim the audio
state back to it before `:5090` — the audio analogue of `Ltx2ClearConditioning`
at `:5045`. `Ltx2ClearConditioning` itself takes a `Ltx2VideoLatentShape*` for
its generated-keyframe extraction
(`include/vllm/model_executor/models/ltx2_conditioning.h:299-302`), so either it
grows an audio arm or the audio side gets a narrower helper. Upstream's audio
states carry no keyframe layout, so the narrower helper is the smaller change,
and the choice must be recorded rather than made silently.

---

## 5. The refusal-text repair (owed under every decision)

`src/vllm/multimodal/ltx2_video.cpp:2935-2946` is wrong in one way and
incomplete in another, and both survive whichever of (a), (b) or (c) is chosen.

* It names `ltx-pipelines/utils/helpers.py:264-269` as "what it needs". That
  function is `audio_latent_from_file`, whose only caller is retake's initial
  latent (§2.4). The anchor is not wrong so much as **one wrapper away from the
  function it names**: `helpers.py:267` does call `encode_audio`, but what
  `audio_latent_from_file` returns is retake's *initial* audio latent, not a
  conditioning item. Point the refusal at `encode_audio` itself,
  `ltx-core .../model/audio_vae/audio_vae.py:249-274` (§2.4), so a reader who
  follows the anchor lands on the encode the message is actually about.
* It says "nothing in this phase loop constructs one from a request" and stops
  there, which reads as a wiring gap. What it must say is that **upstream
  constructs this item at exactly one site, `dubit.py:272`, inside a pipeline
  this tree does not ship**, and name A3 as the owner. That is the sentence a
  reader needs in order not to re-file A18 as cheap work a fourth time.

Follow the file's own convention for a corrected refusal: keep the
`WHAT IS *NOT* THE REASON` paragraph about the audio VAE encoder (it is true and
was earned), and add the constructor fact beside it.

---

## 6. Risks

| Risk | Why it is real here | Mitigation |
|---|---|---|
| **Inventing a capability.** | Under (b) the engine grows a conditioning mode with no upstream referent. AGENTS.md: "Mirror vLLM." | Do not proceed on (b) without a recorded developer decision, and reclassify A18 out of class A when it lands. |
| **A one-sided gate.** | If the delivery has no upstream site, there is no oracle to gate the *values* against — only self-consistency. §7 says what this can and cannot prove. | State it in the row's `## Outcome`; do not present a self-consistency gate as parity. |
| **Silent length drift.** | Without §4.4, `Ltx2AudioUnpatchify` at `:5090` reads a grown buffer with the target `ashape` and still returns a correctly shaped volume. A shape check cannot see it. | The RED-FIRST test in §7 asserts the **decoded audio duration**, not the shape. |
| **Anchor drift.** | Three anchors for this exact refusal were already stale in three documents (§3.1). | Gate on symbols, not lines (§8). |
| **The bridge silently drops `keyframes_mask`.** | `AppendTokens` extends it (`ltx2_conditioning.cpp:75`); an audio `StreamState` carries it empty. Round-tripping through a bridge that forgets it desynchronises a marker no shape check reads — the failure the comment at `ltx2_video.cpp:187-191` names. | Assert the audio `keyframes_mask` is empty on both sides of the bridge. |

---

## 7. Tests

### 7.1 RED-FIRST, through a production entry point

**Under decision (a) or (c), the red-first test is §7.3 and there is no product
test, because there is no product change.**

Under (b): one test, in `tests/vllm/multimodal/test_ltx2_video.cpp`, entering
through `vllm::multimodal::VideoEngine::Generate` — the same entry the C ABI
reaches through `src/capi/vllm_c.cpp:1682` — on a reduced-dimension fixture
checkpoint, with `gen.ref_audio_path` set to a written WAV fixture.

It must fail **for the intended reason before the change**: at `2850314e3` the
call reaches `src/vllm/multimodal/ltx2_video.cpp:2934` and throws with
"reference-AUDIO conditioning is not served". Capture that message as the red
result. A test that passes because the engine ignored the field is not a red
result; assert on the message text.

After the change the assertions are, in order of what they can detect:

1. **The reference tokens reached the DiT.** `im.trace.audio_tokens` (written at
   `src/vllm/multimodal/ltx2_video.cpp:3907`) exceeds `ashape.frames` by exactly
   the reference latent's frame count. This is the only assertion that
   distinguishes "the item was applied" from "the file was read and discarded".
2. **The trim happened.** The returned soundtrack's sample count matches the
   unconditioned render's, byte-for-byte in length. Without §4.4 this is the
   assertion that goes red, and it goes red on a *duration*, which no shape
   check reads.
3. **The render changed.** The audio latent digest
   (`im.trace.audio_latent_digest`, `:4370`) differs from the same request run
   without `ref_audio_path`. A differential, because a self-consistency gate is
   all that exists (§6).

A unit test on `Ltx2ConditionAudioByReference` alone is explicitly **not** the
proof. One already exists at `tests/vllm/models/test_ltx2_vae.cpp:3581` and it
has been green throughout the period in which nothing reached the op — which is
`.agents/reachability.md`'s *test-only driver*, demonstrated in this tree rather
than argued.

### 7.2 The mutation a fresh reviewer applies

In a scratch copy, never in the reviewed worktree, restored byte-for-byte after:

**Reachability mutation.** Delete the single line in
`src/vllm/multimodal/ltx2_video.cpp` that applies the item to the audio state —
the `Ltx2ConditionAudioByReference(...)` call added in the audio-state block
(the region `:3833-3903` today). Delete the **call**, not the implementation,
and leave the encode above it intact so the file is still read. Rerun the
focused gate.

* Assertion 1 must go **red**. If it stays green, the trace is not measuring the
  applied item and the gate is one-sided.
* `tests/vllm/models/test_ltx2_vae.cpp:3581` must stay **green** through this
  mutation. That is the point: it proves the unit test cannot carry the
  reachability claim, and the reviewer should record that it did stay green.

**Second mutation, for the trim.** Delete the audio trim added per §4.4 and
rerun. Assertion 2 must go red. Shape assertions will not, which is why
assertion 2 is on a duration.

**Third mutation, for the bridge.** In the `StreamState` -> `Ltx2LatentState`
audio conversion, drop `keyframes_mask`. The §6 assertion must go red.

### 7.3 The test this row owes under every decision

An anchor-integrity test, because §3.1 found three stale anchors for one
refusal: assert that the strings
`conditioning/types/reference_audio_cond.py` and
`ltx-pipelines/utils/helpers.py:264-269` in
`src/vllm/multimodal/ltx2_video.cpp` are accompanied by the corrected
`dubit.py` constructor statement from §5. This is a text gate on the tree's own
prose and it is weak by construction — it proves the sentence is present, never
that it is true. Recorded as weak rather than presented as parity.

---

## 8. Gates

Every command prints its own denominator. Run from the repository root.

```sh
# G1 — the refusal is where this document says, and there is exactly one.
# Measured at 2850314e3: 1, 2 and 1 respectively.
grep -c 'if (!gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {' \
  src/vllm/multimodal/ltx2_video.cpp          # 1 — the LTX-2.5 phase-loop guard
grep -c 'gen.ref_audio_' src/vllm/multimodal/ltx2_video.cpp
                                              # 2 — that guard plus the T2A one
grep -c 'reference-AUDIO conditioning is not served' \
  src/vllm/multimodal/ltx2_video.cpp          # 1

# G2 — the op's call sites. Before this row: 1, and it is a test.
git grep -n 'Ltx2ConditionAudioByReference' -- src include tests | cat

# G3 — the upstream constructor count. Run inside the ltx-2 pin.
cd /home/mudler/_git/LTX-2 && \
  git grep -c 'AudioConditionByReferenceLatent(' fd4ded7f -- 'packages/*'
# expect: only packages/ltx-pipelines/src/ltx_pipelines/dubit.py constructs it

# G4 — the audio stream has no trim today.
git grep -n 'target_tokens' -- src/vllm/multimodal/ltx2_video.cpp | cat
git grep -n 'Ltx2AudioUnpatchify' -- src/vllm/multimodal/ltx2_video.cpp | cat

# G5 — full gate, only under decision (b).
scripts/agent-preflight.sh
```

**G1's second command is the one that matters and its value is 2, not 1.** The
T2A guard at `:5885-5891` spells the same two fields inside a seven-term
condition, so the single-line form finds only the phase-loop guard. A gate
written on the single-line form alone would report one reader of `ref_audio` on
this path when there are two, and the second one is a correct mirror that must
never change.

**No gate in this row runs a build.** The dispatch that produced it held no
compiling slot. Every figure above is a grep or a read, and §7's assertions are
specified, not executed — see §10.

---

## 9. Evidence

| Claim | Evidence | Verified |
|---|---|---|
| The refusal is at `ltx2_video.cpp:2934-2947`, unique | `grep -n` on `git show origin/main:` output at `2850314e3` | yes |
| The op has exactly one call site and it is a test | `git grep -n Ltx2ConditionAudioByReference -- src include tests` | yes |
| Upstream constructs the item at exactly one site | `git grep -n AudioConditionByReferenceLatent fd4ded7f` (9 hits, 5 files, 1 constructor) | yes |
| No other pipeline gives audio a conditioning list | `git grep -n 'audio=ModalitySpec' fd4ded7f -- 'packages/ltx-pipelines/*'` (18 hits) cross-checked against `conditionings=audio\|audio_conditionings` (6 hits, dubit + trainer) | yes |
| `helpers.py:264-269` is `audio_latent_from_file`, called only by retake | `git grep -n audio_latent_from_file fd4ded7f` (3 hits: def, import, `retake.py:250`) | yes |
| DubIt's stage-1 reference is the reference video's audio | `dubit.py:276` -> `:186-191` | yes |
| DubIt's stage-2 reference is the stage-1 latent | `dubit.py:298`, `:305` | yes |
| The negative shift is Dub-It-only | `dubit.py:351-353`, `packages/ltx-pipelines/CLAUDE.md:85` | yes |
| The trainer uses non-negative positions | `validation_runner.py:759-784` | yes |
| The audio stream has no trim | `ltx2_video.cpp:3753`, `:5045`, `:5071-5072`, `:5090` | yes |
| `StreamState` != `Ltx2LatentState` | `ltx2_video.cpp:181-192` against `ltx2_conditioning.h:94+` | yes |
| No open issue tracks A18 | `gh issue list --state open --search "reference audio"` (2 hits, neither LTX) ; `gh issue list --state all --search "A18 in:body"` (1 hit, #2583, closed — `--state all` because `--state open` excludes it) | yes |
| The a2v spec already records the op as test-only | `.agents/specs/ltx25-a2v-audio-input.md` `## Owed` | yes |
| The scope doc's `ltx2_video.cpp:2754-2766` is stale | it is `:2934-2947` | yes |
| The scope doc's `ltx2_pipeline.h:837` is stale | the DubIt comment is at `:876` | yes |
| The a2v spec's `ltx2_video.cpp:1404` is stale | same refusal, now `:2934` | yes |

---

## 10. OPEN QUESTIONS

1. **Which decision.** (a), (b) or (c) of §0. This row cannot make it: it is a
   product decision about whether the engine carries a conditioning mode
   upstream does not define.
2. **Under (b), which position convention** — DubIt's negative shift
   (`dubit.py:351-353`) or the trainer's plain grid
   (`validation_runner.py:763-766`). Unresolvable from upstream, because
   upstream never applies either on a pipeline resembling this engine's default.
3. **Whether the LTX-2.5 checkpoints were trained with a reference-audio
   condition on the non-DubIt pipelines at all.** If they were not, a delivered
   item on the distilled loop is an out-of-distribution input and the render is
   uninterpretable. **UNVERIFIED** — it needs a real-checkpoint render, which
   needs a GPU lease this row did not hold.
4. **Whether `Ltx2ClearConditioning` should grow an audio arm or the audio side
   should get a narrower helper** (§4.4). Decidable from the code, not decided
   here.
5. **Everything in §7 is specified and not executed.** No build, no `ctest`, no
   render was run: the dispatch that produced this document held no compiling
   slot and no GPU lease. Treat every assertion as a design, not a result.
6. **Whether A3's own spec, when written, already absorbs A18.** No
   `.agents/specs/ltx25-dubit.md` exists at `2850314e3`; under decision (a) the
   A18 `## Owed` entry has to land somewhere, and that file is where.

---

## 11. Records this row would owe

* **The issue is [#2634](https://github.com/mudler/vllm.cpp/issues/2634)**,
  opened with this spec and carrying the same derivation. It stays OPEN, because
  what it asks for is a decision and not a fix. Before it existed, searched
  read-only:
  `gh issue list --state open --search "reference audio"` returned #2613 and
  #387, neither LTX. The other two searches report CLOSED issues, so they were
  run with `--state all`, which is the flag recorded here because `--state open`
  excludes exactly what they found: `--state all --search "A18 in:body"`
  returned only #2583 (A19, closed), and `--state all --search
  "dubit OR DubIt"` returned #2526 and #2583, both closed. Each of the three now
  also returns #2634 itself, which did not exist when they were run. An issue is
  required before work starts (AGENTS.md, "Every change starts from an issue");
  the spec-drafting dispatch was forbidden writes, so the operator opened it
  alongside this landing.
* Under (a): `ltx25-completion-scope.md` §4's A18 row and §8's order 10 change;
  the A18 obligation moves to A3's `## Owed`.
* Under (b): a new row in the campaign matrix, the refusal repair of §5,
  `docs/FEATURES.md` (a conditioning surface changes) and `docs/USAGE.md`
  (`ref_audio` acquires a meaning for LTX-2.5, with the checkpoints it was gated
  against).
* Under any decision: the three stale anchors of §3.1, each corrected in the
  document that carries it.

---

## 12. Stop conditions

* **Stop now and return `NEEDS_DECISION`** if the developer has not chosen among
  §0 (a), (b) and (c). This is the expected exit.
* Stop and re-derive if `git grep -c 'AudioConditionByReferenceLatent('` inside
  the `ltx-2` pin ever returns a constructor outside `dubit.py`. The whole of §0
  rests on that count.
* Stop if an upstream pin advance registers reference-audio conditioning on a
  non-DubIt pipeline. A18 then becomes a genuine class-A gap and this document
  is superseded rather than amended.
* Stop before any build or render: neither a compiling slot nor a GPU lease was
  held, and §7's assertions are unexecuted (§10 question 5).
* Stop if a fresh reviewer's reachability mutation (§7.2) leaves assertion 1
  green. That means the gate measures the class, not the capability, and the
  design in §4 is wrong rather than merely unfinished.

---

## Owed

* **A decision between (a), (b) and (c) of §0**, tracked by
  [#2634](https://github.com/mudler/vllm.cpp/issues/2634). Owner: the developer.
  This row does not choose, because (b) would commit this engine to a
  conditioning geometry that upstream does not define — `dubit.py:351-353` and
  `ltx-trainer/.../validation_runner.py:759-784` disagree, and neither is a
  default.
* **`Ltx2ConditionAudioByReference`'s delivery**, under decision (a), moves to
  A3 (DubIt). No `.agents/specs/ltx25-dubit.md` exists yet, so the entry has
  nowhere to land until that row is written; it is named here so the obligation
  is not lost with this document.
* **The refusal-text repair of §5.** Owed under every decision, because the
  message names `utils/helpers.py:264-269` as its blocker and that is not the
  mechanism. It is not repaired here: the correct replacement text depends on
  which decision is taken.
* **The three stale anchors of §3.1**, each corrected in the document that
  carries it, not from here. `ltx25-completion-scope.md` is operator-owned.
* **A measurement, not a decision:** whether the LTX-2.5 checkpoints carry a
  reference-audio condition on non-DubIt pipelines. It needs a real-checkpoint
  render and a GPU lease, and no arm of §0 is safe to fund without it.

## Now

`SPEC`. No implementation is owed until [#2634](https://github.com/mudler/vllm.cpp/issues/2634)
is answered. The operator's next action on this row is to relay the three
options, not to schedule work.
