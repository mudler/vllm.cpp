# LTX-2.5 IMAGE CONDITIONING — the VAE encoder's load path, and the first arm off `ltx2_video.cpp:1122`

**Row:** `LTX25-IMAGE-COND` (row 1 of the `#644` full-port campaign).
**Issue:** [#644](https://github.com/mudler/vllm.cpp/issues/644).
**Branch:** `row/LTX25-IMAGE-COND`.
**Parent spec:** [`ltx-2-5.md`](ltx-2-5.md) — operator-owned, NOT edited by this row.
**Upstream root (primary):** Lightricks/LTX-2 @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
`packages/ltx-core/src/ltx_core/` and `packages/ltx-pipelines/src/ltx_pipelines/`.
**Upstream (cross-check only):** `huggingface/diffusers` `pipeline_ltx2_condition.py`,
`ltx2/utils.py`. Where the two disagree, §3.3 records it and we follow `ltx_core`.

---

## 0. What is claimed, and what is not

This row closes exactly ONE conditioning kind — an **image at latent frame 0**, at
**CRF 0** — and it does so by building the thing that was actually absent: a
**weight-loading path for the video VAE encoder**. Keyframes, reference video and
reference audio stay refused, with a message that names what is now true rather
than repeating a reason that has gone stale.

Three things are stated up front so they cannot be discovered later.

1. **`crf = 0` is OUT OF DISTRIBUTION for LTX-2.5 and is labelled as such.**
   Upstream resolves an unset CRF from the checkpoint generation
   (`ImageConditioner.resolve_crf`, `blocks.py:977-983`), and a 2.5 checkpoint
   resolves to **18** — `detect_params` maps `(2,5)` onto the newest row at or
   below it, `((2, 4), LTX_2_4_PARAMS)` (`constants.py:130-133`), whose
   `default_image_crf` is `LTX_2_4_IMAGE_CRF = 18` (`constants.py:37, 124`). So
   the DEFAULT request refuses, and a caller must ask for `image_crf = 0`
   explicitly. That is upstream-legal — `preprocess` short-circuits at
   `if crf == 0: return image` (`decode.py:425-426`), and an explicit `0` is
   documented as "skip re-compression entirely" (`args.py:58-59`) — but it
   conditions on pixels the model was not trained to see. It is not silently
   rendered.
2. **No render-quality claim.** This row's evidence is numeric parity against
   executed upstream at reduced dimensions plus a mutation gate. A finite,
   correctly-shaped clip is not a quality result; the parent spec §0 already
   records why.
3. **No speed claim.** Nothing here is measured for throughput; the parent spec's
   `PENDING` speed axis is untouched.

## 1. The gap, re-verified against the current tree (not against the record)

The refusal at `src/vllm/multimodal/ltx2_video.cpp:1122` is honestly written and
its anchors verify. Its internal claim was re-checked, and it is TRUE:

| claim | check | result |
|---|---|---|
| the encoder MATH landed | `Ltx2ConvVideoEncode`, `src/vllm/model_executor/models/ltx2_video_vae.cpp:988` | present, gated (`test_ltx2_vae`) |
| the conditioning ITEMS landed | `Ltx2ConditionVideoByLatentIndex`, `src/vllm/model_executor/models/ltx2_conditioning.cpp:137` | present, gated |
| the engine loads the DECODER filter only | `src/vllm/multimodal/ltx2_video.cpp:752` | `Ltx2LoadVaeWeights(f, Ltx2VideoVaeDecoderKeyRules())` |
| an encoder key filter exists anywhere | `grep -rn Ltx2VideoVaeEncoderKeyRules src include tests` | **NO MATCH** |
| an encoder CONFIG parser exists anywhere | `grep -rn ParseConvVideoEncoder src include tests` | **NO MATCH** |

So the encoder is a brick with no delivery route: nothing turns a checkpoint into
an `Ltx2ConvVideoEncoderConfig`, and nothing puts `encoder.*` tensors in a
`Ltx2VaeWeights`. That, not the math, is what this row builds.

The CRF round trip is **not** on the critical path, contrary to what the refusal
text implies by listing it alongside the encoder. `preprocess` returns the image
unchanged at `crf == 0` (`decode.py:425-426`; note `:427-428` is a *different*
early return, for a degenerate < 2px side), and this environment has no `av`
module at all, which is itself the demonstration: a `crf = 0` path needs no
codec.

## 2. Upstream anchors — every stage, both sides

| stage | upstream `file:line` | ours |
|---|---|---|
| encoder key filter | `ltx-core/.../video_vae/model_configurator.py:267-276` (`VAE_ENCODER_COMFY_KEYS_FILTER`) | `Ltx2VideoVaeEncoderKeyRules` |
| encoder config | `model_configurator.py:37-69` (`_prepare_video_encoder_kwargs`) + `:72-78` (`VideoEncoderConfigurator`) | `Ltx2ParseConvVideoEncoderConfig` |
| encoder lifecycle | `ltx-pipelines/.../utils/blocks.py:936-991` (`ImageConditioner`; build `:985-986`, build-and-free `:988-991`) | engine load path, `ltx2_video.cpp` |
| CRF resolution | `blocks.py:966-983` + `constants.py:36-37, 124, 130-133` | `Ltx2ResolveDefaultImageCrf` |
| CRF round trip | `decode.py:413-435`, `encode_single_frame:386-400` | **REFUSED BY NAME** (§3.4) |
| image decode | `decode.py:139-170` (`decode_image`: EXIF rotate, ICC→sRGB, uint8 RGB) | `Ltx2DecodePpmRgb` (PPM only, §3.2) |
| aspect-fill resize | `media_io/resize.py:41-73` (`resize_and_center_crop`) | `Ltx2ResizeAndCenterCrop` |
| normalize | `media_io/range_map.py:8-9` (`normalize_images`) | in `Ltx2LoadImageAndPreprocess` |
| the whole preprocess | `decode.py:46-79` (`load_image_and_preprocess`) | `Ltx2LoadImageAndPreprocess` |
| VAE encode | `ltx-pipelines/.../utils/helpers.py:285-294` | `Ltx2ConvVideoEncode` (already landed) |
| latent composition | `ltx-core/.../conditioning/types/latent_cond.py:32-43` | `Ltx2ConditionVideoByLatentIndex` (already landed) |
| noise composition | `ltx-core/.../components/noisers.py:30-37` | `ApplyGaussianNoise` (already landed) |

## 3. Design

### 3.1 The encoder load path

`Ltx2VideoVaeEncoderKeyRules()` mirrors `VAE_ENCODER_COMFY_KEYS_FILTER`
(`model_configurator.py:267-276`) in the first-match-wins prefix form the
already-gated `Ltx2VideoVaeDecoderKeyRules()` uses. It is a TRANSLATION, not a
rule-for-rule copy: upstream's `SDOps` (`loader/sd_ops.py:101-122`) admits a key
by `any()` over four matchings and then chains three substring replacements,
while this port matches and replaces in one pass — so the fourth rule below is an
identity that exists to carry upstream's fourth matching. Equivalent on every key
a shipped checkpoint carries; see the comment on the function for exactly when
they would part.

```
{"vae.encoder.",                ""}
{"vae.per_channel_statistics.", "per_channel_statistics."}
{"encoder.",                    ""}
{"per_channel_statistics.",     "per_channel_statistics."}
```

`per_channel_statistics` is in BOTH filters upstream and must be in both here:
`Ltx2ConvVideoEncode` reads `per_channel_statistics.{std,mean}-of-means` to
normalize its output (`video_vae.py:336`), so an encoder bag without them cannot
produce a latent in the DiT's space.

`Ltx2ParseConvVideoEncoderConfig` mirrors `_prepare_video_encoder_kwargs` key for
key, including the two-layout split (`:46-53`): a nested `vae.encoder` object
takes its latent width from `encoder.out_channels`, a flat
`CausalVideoAutoencoder` from `vae.latent_channels` — and the top-level
`out_channels` is the DECODER's RGB count and must never be read as the latent
width (`:41-43`). The encoder's `spatial_padding_mode` default is `zeros` where
the decoder's is `reflect` (`:63-67` vs `:90`); both read the same checkpoint key
on a flat config, so they diverge only when it is ABSENT — silently, by a
half-pixel border, in opposite directions. That is why the default lives in the
parser and is pinned by a test.

A checkpoint carrying no `encoder.*` tensors is a decoder-only (Comfy-split)
file. The engine then holds no encoder and an image request is refused BY NAME —
never served by falling back to something else.

### 3.2 Image preprocessing at `crf = 0`

`load_image_and_preprocess` (`decode.py:74-78`) is four steps, and the ORDER is
load-bearing:

```
image = decode_image(path)           # uint8 [H, W, 3], sRGB
image = preprocess(image, crf)       # identity at crf == 0
image = torch.tensor(image, float32) # values still 0..255
image = resize_and_center_crop(image, height, width)   # <- IN 0..255 SPACE
image = normalize_images(image, ...)                   # /127.5 - 1.0
```

The resize happens BEFORE the normalize. Bilinear interpolation is affine so the
two commute mathematically, but not bit-for-bit, and this project's existing PPM
reader (`minimax_h3_video.cpp:87-120`) normalizes at decode time. Reusing it
would put the affine first. So this row decodes to **uint8** and keeps upstream's
order; `Ltx2DecodePpmRgb` is a separate reader for that reason and the reason is
recorded here rather than left as an unexplained duplicate.

`resize_and_center_crop` (`resize.py:41-73`) is aspect-FILL then centre crop:
`scale = max(h/src_h, w/src_w)`, `new = ceil(src * scale)` (upstream comments the
`ceil` as avoiding negative crop offsets), `interpolate(mode="bilinear",
align_corners=False)`, then `crop_top = (new_h - h) // 2`.

The bilinear kernel mirrors PyTorch's `align_corners=False` index map in **f32**,
which is what `HelperInterpLinear::compute_indices_weights` uses
(`aten/src/ATen/native/cpu/UpSampleKernel.cpp`, dispatched over `scalar_t`):
`scale = src / dst`, `real = scale * (i + 0.5) - 0.5`, clamped to `>= 0`,
`idx = floor(real)`, `lambda = clamp(real - idx, 0, 1)`, and the right tap is
`idx + (idx < src - 1)`. Width and height passes are separated, height first —
matching the generic Nd kernel's dimension order.

**Only PPM (P6) is read**, and only `maxval == 255`. No PNG/JPEG codec is
vendored — the same NAMED residual `minimax_h3_video.cpp:84-86` already carries —
and a `maxval != 255` PPM is REFUSED rather than rescaled, because upstream's
decode is PIL's and mirroring PIL's rescale semantics is a separate port. EXIF
rotation and ICC→sRGB conversion (`decode.py:143-168`) do not apply: PPM carries
neither tag, so there is nothing to honour and nothing is silently dropped.

### 3.3 Composition — and where the two references DISAGREE

`ltx_core` writes the encoded latent into `clean_latent` and sets
`denoise_mask = 1 - strength` (`latent_cond.py:40-41`), leaving the NOISY tensor
alone; the noiser then composes the two with a DOUBLE lerp
(`noisers.py:32-33`):

```
latent = lerp(latent, noise, noise_scale)
latent = lerp(clean_latent, latent, denoise_mask)
```

diffusers instead writes the clean tokens into the noisy tensor as well
(`pipeline_ltx2_condition.py:1002-1004`, `:1229-1231`). The two agree ONLY at
`noise_scale == 1`. The two-stage distilled recipe's second phase does not run at
`noise_scale == 1`, so this is a live divergence, not a theoretical one.

**We follow `ltx_core`.** `Ltx2ConditionVideoByLatentIndex` already implements
exactly that and `ltx2_conditioning.h` already records the divergence; this row
adds no new composition, it only reaches the existing one. Do not silently switch
to the diffusers form.

Placement in `Generate`: the conditioning is applied to the video `StreamState`
AFTER `clean` is seeded from the patchified initial volume and BEFORE
`ApplyGaussianNoise` — which is `create_noised_state` order (`helpers.py:428-445`:
initial state, then the conditioning items, then the noiser). NOT `blocks.py:576-580`,
which this spec cited until the review of #657: those lines are the TEARDOWN
(`clear_conditioning` + `unpatchify`) and say nothing about conditioning order.
It is applied on EVERY phase, because every phase rebuilds its state from the
recipe and a conditioning dropped on phase 2 would be re-noised away.

`strength` comes from the seam's existing `VideoGenParams::noise_aug`
("keyframe pinning strength; <= 0 => 1.0", `include/vllm.h:762`), whose polarity
already matches upstream's `ImageConditioningInput.strength` (`args.py:64`): 1.0
pins, mask 0.

### 3.4 What is still refused, and in what words

Four distinct refusals, each naming a different missing piece so a later reader
can re-check its cause rather than trust it. This campaign has had FIVE refusals
whose stated reason went stale; each message below names the exact symbol or
`file:line` that would have to change for it to become false.

1. **Non-zero CRF.** Names `encode_single_frame` / `decode_single_frame`
   (`decode.py:386-410`) as unported, states that `crf = 0` is the supported
   value and that it is out of distribution for a 2.5 checkpoint whose resolved
   default is 18, and says which extra to set.
2. **No encoder in the checkpoint.** Names `Ltx2VideoVaeEncoderKeyRules` and says
   the file carried no `encoder.*` / `vae.encoder.*` tensors.
3. **Keyframes** (`last_frame_path`, or an image at a non-zero frame index).
   `Ltx2ConditionVideoByKeyframe` EXISTS and is gated; what is missing is the
   TOKEN-APPEND machinery. `VideoConditionByKeyframeIndex.apply_to`
   (`keyframe_cond.py:36-90`) appends tokens — concatenating onto `latent`,
   `denoise_mask`, `positions` and `clean_latent` (`:79-82`), giving them their
   own coordinates offset to `frame_idx` (`:46-59`) and rebuilding the attention
   mask via `update_attention_mask` (`:68-76`) — and `clear_conditioning`
   (`ltx_core/tools.py:88-105`) trims them back before unpatchify. `Ltx2LatentState`
   carries no attention mask, and the engine's phase loop is fixed at the target
   grid's token count from one `Ltx2VideoTokenCount` through the sigma schedule,
   the `Ltx2ModalityInput` and `Ltx2VideoUnpatchify`. The first-frame arm needs
   none of it because `VideoConditionByLatentIndex` REPLACES existing tokens.

   **This spec and the shipped message previously named
   `keyframes_abs_pos_embedding` as the blocker. That was FALSE at pin
   `fd4ded7f`, and a test had been written to assert it by name.** A supplied
   keyframe is appended with `marked=False` (`keyframe_cond.py:84-86`), and its
   sole consumer adds `mask * embedding` with `mask = keyframes_mask > 0`
   (`model/transformer/transformer_args.py:42-43`, called once at `:269`), so the
   embedding contributes nothing to those tokens and porting it would not serve
   this arm. The tokens that DO reach it are the target's own first latent frame,
   marked unconditionally by `_first_frame_keyframes_mask`
   (`ltx_core/tools.py:184-196`) — the frame the SERVED first-frame arm writes
   into. That is a real gap on the served arm and is tracked as
   [#658](https://github.com/mudler/vllm.cpp/issues/658); it is not what blocks a
   last-frame keyframe.
4. **Reference video / reference image / reference audio.**
   `Ltx2ConditionVideoByReference` and `Ltx2ConditionAudioByReference` also
   EXIST; what is missing is that both need the IC-LoRA's `downscale_factor` /
   `temporal_scale_factor`, which upstream stores in LoRA metadata this project
   does not read (already recorded at `ltx2_conditioning.h:110-114`), and the
   audio arm additionally needs the audio VAE ENCODER, whose key rules this row
   does not add.

## 4. The things that fail silently here

* **Reading `vae.out_channels` as the encoder's latent width.** It is the
  decoder's RGB count (3). An encoder built with `out_channels = 3` still runs
  and still produces a latent.
* **The encoder's `zeros` vs the decoder's `reflect` padding default.** Same
  checkpoint key, different defaults, and they only diverge when the key is
  absent (`model_configurator.py:63-68` vs `:92`).
* **Defaulting a `res_x` block's `num_layers`.** Upstream subscripts
  `block_config["num_layers"]` (`video_vae.py:55`) and raises `KeyError`; no
  other block kind reads the field. `ParseEncoderBlocks` defaulted it to 1, which
  builds a one-layer `UNetMidBlock3D` out of a config upstream refuses. Made
  strict in the review of #657, matching how `multiplier`'s sentinel two lines
  below already treats an absent value.
* **Normalizing before resizing.** Same answer to ~1e-7 — and §8.1 records the
  outcome: NO golden here sees it either, because the two orders are
  algebraically equal. Mirrored because it is upstream's, and written down
  rather than assumed to be covered.
* **Applying the conditioning after the noiser.** Produces a pinned first frame
  that is pinned to the NOISED latent. Shapes, masks and finiteness all pass.
* **Inverting the mask (`strength` instead of `1 - strength`).** Renders an
  unconditioned clip that looks like the feature not working.
* **A conditioning that is loaded but never read.** The whole class this row's
  mutation evidence exists to exclude: `last_conditioning()` scaled x1.5 and
  row-reversed both passed every assertion in an earlier phase.

## 5. Tests and evidence

**Goldens.** `scripts/gen-ltx2-image-cond-goldens.py` imports and EXECUTES
upstream under the pinned SHA and emits
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc`. Sections:

1. `resize_and_center_crop` alone, over shapes covering upscale, downscale,
   wider-than-target and taller-than-target, including one where `ceil`
   disagrees with both `round` and `floor` (§8.1 corrects what that case
   actually demonstrates).
2. `load_image_and_preprocess`'s full chain at `crf = 0` (resize then normalize).
3. `VideoEncoder(image)` at reduced dims over deterministic weights.
4. `VideoConditionByLatentIndex.apply_to` over that encoded latent — `clean` and
   `denoise_mask`.
5. `GaussianNoiser` over the conditioned state at a NON-unit `noise_scale`, which
   is the arm at which `ltx_core` and diffusers disagree (§3.3).

**Harness adaptations, recorded:** (a) `ltx_pipelines.utils.media_io.__init__`
imports `av`, which is absent in this environment, so `resize.py` and
`range_map.py` are loaded by FILE PATH with `importlib` rather than as package
members. Both import only `torch`/`einops`/stdlib, so nothing about the math
changes, and the workaround is itself the evidence for §0.1. (b) Weights are
filled from the same deterministic stream `gen-ltx2-vae-goldens.py` uses, so no
weight byte is checked in.

**Mutation evidence (required, not optional).** Each of these is applied to a
scratch copy, rebuilt, run, and the tree restored byte-for-byte:

| mutation | must RED |
|---|---|
| conditioned latent scaled x1.5 | golden §4/§5 |
| mask set to `strength` instead of `1 - strength` | golden §4 |
| conditioning applied AFTER the noiser | golden §5 |
| resize/normalize order swapped | golden §2 |
| encoder key rules replaced by decoder key rules | the load test (weights absent) |
| the encoder config's latent width read from `vae.out_channels` | the config test |

**Encoder weights are LOADED AND USED, proven separately from "loadable":** the
engine test perturbs ONE tensor of the encoder half of the fixture checkpoint and
asserts the conditioning trace's image digest moves, with every other byte of the
request identical.

## 6. Gates

* `ctest -R 'test_ltx2_image_cond|test_ltx2_video|test_ltx2_vae'` — focused, with
  CASE and ASSERTION counts recorded on both sides of every mutation.
* Full `ctest` with the case count asserted against `ctest -N`.
* `scripts/agent-preflight.sh --staged` before commit.

## 7. Stop conditions

* If closing the image arm would require editing the prompt-AdaLN path in
  `ltx2_loader.cpp` / `ltx2.cpp`, STOP and return `NEEDS_DECISION` — that is
  `row/LTX25-PROMPT-ADALN`'s surface.
* If executed upstream disagrees with a ported stage by more than
  `kLtx2GoldenTol`, the port is wrong; do not widen the band.

## 8. Outcome — what was measured, and what was refuted

Recorded here rather than in the code, because none of it is derivable from the
tree. All measurements: CPU Release, gcc, `-ffp-contract=off` (the tree's own
flag), `build-lic`, this box, at the head this spec landed on.

### 8.1 Three claims in §3 and §5 were WRONG, and are corrected here

* **The `ceil` reason.** §3.2 repeated upstream's own comment — that `ceil`
  guards against `src * scale` landing just above an integer and producing a
  negative crop offset. Swept every source/target pair in `3..40 -> {16, 24}`:
  **no pair does that** in IEEE double. The `ceil` is still load-bearing, for
  the ordinary reason that it disagrees with `round` and `floor` at a
  non-integer scale (case 1, `32 * 16/24 = 21.333` → 22 rows and a 3-row crop,
  against 21 and a 2-row crop), and the generator asserts THAT rather than the
  claim upstream makes.
* **A golden can see the resize/normalize ORDER.** It cannot, and the reason is
  structural: resize is a convex combination and normalize is affine, so the two
  orders are equal in exact arithmetic and their f32 gap is pure rounding —
  **1.94e-07 measured**, below the golden band and below this port's own distance
  from torch. It cannot be amplified by choosing a different image. The order is
  mirrored because it is upstream's; that is now written down in three places
  rather than assumed to be covered, and `kLtx2ImgPreOrderGap` asserts the gap
  stays below the band so a future change that makes it gateable is visible.
* **The port can match torch's bilinear bit for bit.** It cannot, portably. The
  index map and the lambdas were probed with basis images and match EXACTLY
  (`0.61111104`, `0.35185182`, `0.09259248` … reproduced to the bit). The
  residual is in the ACCUMULATION and appears on an output element whose width
  weights are `(1, 0)` — a pure two-term `a*h0 + c*h1` — which rules out
  dimension order. Plain-f32, f64-accumulate and premultiplied-weight orderings
  all land 1 ulp away on the same elements: FMA contraction inside torch's
  kernel, which this tree compiles with `-ffp-contract=off` and so cannot
  reproduce. Hence `kLtx2ImgPixelTol`.

### 8.2 The bands, derived rather than picked

Measured by setting both to `1e-12` and reading the reported `worst`:

| section | space | worst | band |
|---|---|---|---|
| 1 resize | 0..255 | `6.10352e-05` (identity case: 0) | `kLtx2ImgPixelTol = 2e-4` |
| 2 preprocess | [-1, 1] | `4.76837e-07` | `kLtx2ImgGoldenTol = 2e-6` |
| 3 encoded latent | latent | `2.68221e-07` | same |
| 4 conditioned clean | latent | `2.68221e-07` | same |
| 5 noised latent | latent | `1.78814e-07` | same |

### 8.3 Mutation evidence — RED, with counts

Each applied to the tree, rebuilt, run, and restored byte-for-byte (md5 checked).
Green baseline: `test_ltx2_image_cond` **15 cases / 198 assertions**,
`test_ltx2_video` **32 cases / 550 assertions**, both exit 0.

| mutation | file | result |
|---|---|---|
| `ceil` → `llround` in the resize | `ltx2_image_preprocess.cpp` | image_cond 14/15, 197/198, exit 1 |
| encoder key rules → the DECODER's | `ltx2_video_vae_encoder_load.cpp` | image_cond 14/15, 195/198; video 31/32 (THREW, assertions 517 — the COUNT itself moved) |
| mask `1 - strength` → `strength` | `ltx2_conditioning.cpp` | image_cond 13/15, 193/198, exit 1 |
| latent width from `vae.out_channels` | `ltx2_video_vae_encoder_load.cpp` | image_cond 14/15, 192/198; video 9/32, 36/53 |
| CRF refusal removed | `ltx2_image_preprocess.cpp` | image_cond 14/15, 193/198; video 31/32 |
| encode the image, never PLACE it | `ltx2_video.cpp` | video 31/32, 548/550, exit 1 |

### 8.4 A mutation that SURVIVED, and what that says

`video.latent = state.clean` inserted after the placement left `test_ltx2_video`
fully green (32/32, 550/550). Two findings, and the first is not a gap:

1. **Phase 0 runs at `noise_scale = 1.0`** (`ltx2_pipeline.cpp:1068`), where
   `lerp(latent, noise, 1)` discards `latent` entirely — so on that phase the
   mutation is genuinely inert. AGENTS.md's rule applies: a mutation that moves
   nothing is not evidence of unreachability. It IS live on phase 1
   (`noise_scale = 0.909375`).
2. **`test_ltx2_video` gates no VALUE of the composed latent.** It gates that the
   conditioning is placed, that it depends on the image, and that it depends on
   the ENCODER'S OWN WEIGHTS — not the arithmetic. That arithmetic is gated
   against executed upstream in `test_ltx2_image_cond`, over the identical
   functions. This is recorded as a named residual rather than closed with an
   unanchored digest, which would detect change without pinning anything.

That mutation is also what caused `Ltx2ConditioningTrace::image_digest` to be
taken over the TOKENS AS WRITTEN rather than over the encoder's output: the
first version digested `encoded.data`, which stays healthy for a build that
encodes an image and never places it. The "never place it" mutation above is RED
only because of that change.

### 8.5 Why the defaults are what they are

* **`image_crf` has no default that renders.** Absent resolves 18 and refuses.
  A default of 0 would silently condition every request out of distribution.
* **The encoder stays RESIDENT**, where `ImageConditioner` builds and frees it
  per call (built at `blocks.py:985-986`, built-and-freed around `fn` at
  `:988-991`). A conditioning image arrives per request and
  the encoder is small next to the DiT; the divergence is lifecycle only.
* **The conditioning is applied PER PHASE**, because the two-stage recipe renders
  its stages at different resolutions and upstream passes each stage's own
  height/width. One encode would either be re-noised away or placed at the wrong
  scale. **This reason was gated by nothing until the review of #657.** MEASURED:
  changing the guard to `wants_image && phase_index == 0` left `test_ltx2_video`
  at 32 cases / 550 assertions / exit 0, because `image_tokens` and `image_digest`
  are overwritten each phase and the suite only asked `image_tokens > 0`. The
  trace now pins the LAST phase's per-latent-frame token count (4 in the fixture,
  1 at phase 0) and contrasts it with a `max_phase = 0` engine over the same
  request, which REDs that mutant at `1 == 4`. It matters because hoisting the
  per-phase decode+encode out of the loop is the obvious optimization.

## 9. Now

`ACTIVE` — spec committed before implementation; implementation landed with the
outcome above.
