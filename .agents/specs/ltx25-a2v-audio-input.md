# LTX-2.5 — audio-to-video (`A2VidPipelineTwoStage`) and the audio input path

Row: `LTX25-A2V-AUDIO-INPUT`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#922](https://github.com/mudler/vllm.cpp/issues/922). Parent campaign issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Read from a local checkout at that revision, verified with `git rev-parse HEAD`
before any anchor below was taken.

---

## 0. Honesty statement — what this row does and does not claim

This row makes a supplied audio file **drive** an LTX-2.5 render. It does not
claim reference-audio conditioning, which is a different upstream mechanism
(`ReferenceAudioConditioning`, appends tokens) and stays refused by name at
`src/vllm/multimodal/ltx2_video.cpp:1404`. It does not claim `RetakePipeline`
(#924) or text-to-audio (`t2a_one_stage.py:43,109`), both recorded as owed under `## Owed`.

**Upstream ships no tests at this pin.** `find . -iname '*test*' -not -path
'./.git/*'` over `Lightricks/LTX-2 @ fd4ded7f` returns **empty**; `pytest~=9.0`
is declared in `pyproject.toml:30` and no test file exists to run. So "port the
upstream tests in the same change" has nothing to port, and the obligation
becomes what §5 does instead: pin upstream's *behaviours* — its constants,
its truncation polarity, its error paths — each against a `file:line` anchor,
plus a golden generator that EXECUTES the upstream modules. This is stated
rather than quietly skipped, because a reader who sees no ported test file
should be able to tell "upstream had none" from "the porter did not look".

**No render on real weights is claimed.** The row is gated on reduced-dimension
fixtures and executed-upstream goldens. The GPU is out of bounds for this row
(a long render holds `dgx.casa` under `flock`), so the real-checkpoint A2V
render is owed and named under `## Owed`.

---

## 1. The grounding claim, verified on this tree

Upstream's pipeline is `A2VidPipelineTwoStage` (`ltx-pipelines/src/ltx_pipelines/
a2vid_two_stage.py:53`, called at `:143`). Stage 1 denoises video at half
resolution with the audio latent **frozen**; stage 2 upsamples 2x and refines
with a distilled LoRA, audio still frozen. The decoded audio is never returned
from the VAE — the caller's original waveform is passed through untouched
(`:301-303`).

Four things are missing here, and three things people expect to be missing are
not.

**Already present, so this row does not rebuild them:**

| Piece | Ours | Upstream |
|---|---|---|
| `AudioEncoder.forward` | `ltx2_audio_vae.cpp:1114` | `audio_vae.py:190-246` |
| mel front-end | `Ltx2WaveformToLogMel`, `ltx2_audio_vae.cpp:1019` | `ops.py:44-55` |
| slaney filterbank | `Ltx2SlaneyMelFilterbank`, `ltx2_audio_vae.cpp:970` | `torchaudio`, reached from `ops.py:20-34` |
| latent frame count | `ltx2_video.cpp:1610` | `types.py:164-181` |
| a stereo-capable PCM16 WAV reader | `MiniMaxH3ReadWav`, `minimax_h3_wav.cpp:89` | `media_io/decode.py:240-300` |

`MiniMaxH3ReadWav` matters: `ltx2_video.cpp` already includes `minimax_h3.h`
(`ltx2_video.cpp:42`) and already calls its sibling `MiniMaxH3WriteWav`
(`ltx2_video.cpp:2292`), so reusing it adds no dependency edge. Both anchors are
written out in full rather than as a bare `:NN` continuation, because a bare one
reads as a continuation of the LAST file named — here `minimax_h3.h`, not
`ltx2_video.cpp` — and that is precisely the class of stale anchor #911 records. It emits **channel-major** float, which is exactly
the layout `Ltx2WaveformToLogMel` takes, and it refuses a sample-rate mismatch
rather than resampling (`minimax_h3_wav.cpp:128-131`) — the same policy
`Ltx2WaveformToLogMel` already declares (`ltx2_audio_vae_encoder.h:171-177`).
The other two WAV readers in the tree are mono-only
(`audio_processor.cpp:67`, `speech_api.cpp:41-79`) and cannot serve the shipped
encoder's `in_channels = 2`.

**Missing:**

1. **The audio VAE encoder load path.** `Ltx2AudioVaeDecoderKeyRules`
   (`ltx2_loader.cpp:1296`) materializes `audio_vae.decoder.` and
   `audio_vae.per_channel_statistics.` only. No `Ltx2AudioVaeEncoderKeyRules`
   and no `Ltx2ParseAudioEncoderConfig` exist anywhere. The ported encoder
   therefore has no weights, which is precisely what the reference-audio refusal
   names at `ltx2_video.cpp:1404`.
2. **Ingestion.** Nothing turns a file path into a waveform for LTX-2.
3. **Per-modality noise control.** The engine applies **one** `phase.noise_scale`
   to both streams (`ltx2_video.cpp:1708-1712 @ 5a0ffe9e`). Upstream's `ModalitySpec`
   carries `noise_scale` and `frozen` per modality
   (`ltx-pipelines/utils/types.py:99-112`), and A2Vid needs the difference: at
   stage 2 video is noised to `stage_2_sigmas[0]` while audio is
   `noise_scale=0.0, frozen=True` (`a2vid_two_stage.py:285-296`).
4. **The recipe and the reachable surface.** `ResolveLtx2PipelineRecipe`
   (`ltx2_pipeline.cpp:1111-1136`) accepts exactly `one_stage`,
   `distilled_two_stage`, `dmd2` and refuses anything else by name.

### The two upstream details that fail silently if guessed

**Truncation is one-sided, and the line after it decides what that means.**
A2Vid does `encoded_audio_latent[:, :, : audio_shape.frames]`
(`a2vid_two_stage.py:202`): it **truncates and never pads**. Retake's helper
`_conform_latent_length` (`ltx-pipelines/utils/helpers.py:149-162`) truncates
*or* zero-pads, and A2Vid does not call it.

Read alone, that says a short take yields a short latent. It does not, and this
spec said it did until the next hop was read. `AudioLatentTools.create_initial_state`
**asserts** `initial_latent.shape == self.target_shape.to_torch_shape()`
(`ltx-core/tools.py:253-255`). The slice can only shorten, so a take shorter than
the clip trips that assertion: **short audio is an ERROR upstream, not a short
latent.** The two lines have to be read together, and truncate-only is not
permission to under-fill.

So this row truncates a long take, and **refuses** a short one by name with the
shortfall in seconds. Zero-padding is the tempting repair and is wrong twice: it
diverges from upstream, and it welds silence onto the end of the take while the
render finishes normally. Both halves are pinned by tests.

**`frozen` is not the same as `noise_scale = 0`.** Upstream's own docstring says
`frozen=True` "zeros the denoise mask and marks the resulting `LatentState` so
`Modality.sigma` is forced to 0 (not only per-token timesteps)"
(`utils/types.py:104-106`). Setting only the noise scale leaves the stream being
stepped by the sampler. Both are mirrored.

---

## 2. Scope

**In.** A new translation unit per concern, mirroring upstream's file structure
rather than growing `ltx2_video.cpp`. The audio VAE encoder load path as its own
TU, mirroring the precedent `ltx2_video_vae_encoder_load.cpp` set (that file's
`:5-12` states it is a separate TU precisely to avoid locking `ltx2_loader.cpp`).
Ingestion with upstream's `start_time` / `max_duration` window. The per-modality
freeze. Reachability through `include/vllm.h` and `ltx2-gen`.

**In scope and NOT delivered, so it is owed rather than implied.** This row ports
the audio-conditioning *mechanism* of `A2VidPipelineTwoStage` — decode, encode,
truncate, freeze — and it does **not** add an `a2vid_two_stage` entry to
`ResolveLtx2PipelineRecipe`. Audio conditioning is orthogonal to the recipe here:
a supplied take rides whichever `(kind, version)` the checkpoint resolves, which
in practice is `distilled_two_stage`.

That is not the same schedule upstream's A2Vid runs, and the difference is
nameable. Upstream's stage 1 is **caller-configured and guided** —
`self._scheduler.execute(steps=num_inference_steps)` with a `GuidedDenoiser` and
`MultiModalGuiderParams` whose `modality_scale` is the CLI's `a2v_guidance_scale`
(`a2vid_two_stage.py:226-240`, `:353-360`) — while stage 2 is
`STAGE_2_DISTILLED_SIGMAS` with a `SimpleDenoiser` (`:164`, `:278`). Our
`distilled_two_stage` recipe fixes both stages' sigmas. So the conditioning is
upstream's and the trajectory is not, and no claim is made that a render here
reproduces upstream's A2Vid output. Recorded under `## Owed`.

**Out, and refused by name rather than dropped.** Resampling: upstream resamples
with `torchaudio.functional.resample` (`ops.py:40`), an arbitrary-ratio polyphase
kaiser resampler; this project ports only the integer-ratio hann-sinc variant,
so a sample rate other than the encoder's is refused with the two rates in the
message. Non-PCM16 and non-RIFF containers: upstream reads anything PyAV opens
(`decode.py:252`), and no demuxer is vendored here
(`video_api.cpp:115-121` says so explicitly). MP3/FLAC/OGG are therefore refused
by name. `RetakePipeline` (#924) and text-to-audio.

### Why `RetakePipeline` is a separate row

Judged from both upstream files, not from the names. They share the audio VAE
encoder and nothing else of substance:

| Needed by | A2Vid | Retake |
|---|---|---|
| audio VAE **encoder** | yes | yes |
| audio VAE **decoder** on output | **no** — returns the caller's waveform (`a2vid:301-303`) | yes (`retake.py:327`) |
| `TemporalRegionMask` | no | yes (`noise_mask_cond.py:10-47`), **zero hits in this tree** |
| video-file ingestion + video VAE encode | no, still images only | yes (`helpers.py:165-233`) |
| latent length policy | truncate only (`a2vid:202`) | truncate **or pad** (`helpers.py:149-162`) |
| stage count | two, with the spatial upsampler | one, distilled |

The two latent-length policies are the decisive point: a shared ingestion helper
would have to pick one and would be wrong for the other. `TemporalRegionMask`
alone is a new conditioning item evaluated in patchified space under two
different coordinate conventions (seconds for audio, latent bounds pushed
through `get_pixel_coords` then divided by fps for video). Bundling buys one
shared function and costs a second, unrelated conditioning mechanism inside the
same review. Filed as #924 and listed under `## Owed`.

---

## 3. Design

Mirroring upstream call order exactly (`a2vid_two_stage.py:196-202`):

1. `decode_audio_from_file(path, device, start_time, max_duration)`
   (`media_io/decode.py:240-300`) → `Ltx2DecodeAudioFile`. Upstream normalizes
   integer PCM to `[-1, 1]` and de-interleaves to `(channels, samples)`
   (`decode.py:173-183`), then trims `round((start_time - first_frame_time) *
   sample_rate)` leading samples and caps at `round(max_duration * sample_rate)`
   (`:290-296`). With a RIFF container there is no codec frame boundary, so
   `first_frame_time` is exactly `0.0` and the leading trim reduces to
   `round(start_time * sample_rate)`. That simplification is recorded here
   because it is only valid for the uncompressed container this row accepts.
2. Upstream **raises** when the decode yields nothing
   (`a2vid_two_stage.py:198`); mirrored as a refusal naming the path.
3. `encode_audio` (`audio_vae.py:249-274`) → `Ltx2WaveformToLogMel` then
   `Ltx2AudioEncoderForward`. The encoder already applies `_normalize_latents`,
   so its output is what `Ltx2AudioPatchify` expects.
4. Truncate to `AudioLatentShape.from_duration(batch=1, duration=num_frames /
   frame_rate, channels=8, mel_bins=16).frames` (`a2vid_two_stage.py:201-202`),
   `latents_per_second = 16000 / 160 / 4 = 25.0` (`types.py:174`). The engine
   already computes this count at `ltx2_video.cpp:1610`; the row reuses it
   rather than re-deriving it.
5. Seed the audio stream with the result and freeze it for **every** phase.
6. **Return the caller's own waveform as the soundtrack, and do not run the audio
   VAE decoder or the vocoder at all.** Upstream states the reason in as many
   words: "Return the original input audio instead of VAE-decoded audio to
   preserve fidelity" (`a2vid_two_stage.py:301-303`). The latent was frozen the
   whole way, so the reconstruction can at best equal the file the caller already
   has, and in practice loses an encode and a vocoder pass to it. Skipping the
   chain is upstream's behaviour rather than an optimisation, and it is
   observable: the BWE vocoder emits 48 kHz where the take went in at 16 kHz, so
   `VideoResult::sample_rate` is the give-away a test can hold. What is returned
   is the **windowed** take, because that is what `decode_audio_from_file`
   produced (`:196`, `:303`).

The phase recipe grows a per-modality shape. `phase.noise_scale` keeps its
meaning for video; audio gains `audio_noise_scale` and `audio_frozen`, defaulted
so every existing recipe is byte-identical in behaviour. The denoise loop skips
the audio Euler step when the stream is frozen, and the audio denoise mask is
zeroed, mirroring both halves of upstream's `frozen`.

### Where the request enters

The v18 parallel-array extras mechanism, which `vllm.h:917-926` documents as
existing for exactly this ("No ABI change was needed for it, which is what this
parallel-array shape exists for"). Keys: `audio_path`, `audio_start_time`,
`audio_max_duration`. This keeps the footprint on `include/vllm.h` to the
LTX-2.5 extras doc comment, which matters because sibling rows are editing that
header concurrently. `CheckUnservedExtras` (`ltx2_video.cpp:357`) already
refuses an unknown key, so a typo is refused rather than ignored.

`audio_max_duration` defaults to `num_frames / frame_rate` when absent, which is
what upstream's CLI passes (`a2vid_two_stage.py:369-371`) — note it is the
*CLI* that supplies this default, not the pipeline, whose own default is `None`
(`:157`). Mirrored at the same layer.

---

## 4. Risks

**The `READER ANCHORS` gate.** `ltx2_video.cpp:304-305` carries a line-number
list re-derived and string-compared by `test_ltx2_video.cpp:873`. Any line
inserted above 1100 shifts it, and a clean `git merge` will not warn. Mitigation:
re-derive the list at the final tree after the last merge of `origin/main`, and
treat it as a merge hazard in the PR body.

**Concurrent edits.** `ltx2_video.cpp`, `ltx2.h`, `docs/FEATURES.md` and
`.agents/issue-index.md` are being edited by sibling rows. Mitigation: new
behaviour lives in new TUs; the touch on `ltx2_video.cpp` is the minimum seam
edit; keyed records are reapplied by key with unrelated keys proven
byte-identical.

**A silently wrong latent.** Every failure mode here renders a plausible video:
a half-bin lattice shift from symmetric padding, a wrong channel packing, a
padded-instead-of-truncated latent, a mono file fed to a two-channel encoder.
None is visible in the output. Mitigation: value-level goldens from executed
upstream, plus a lower bound on the latent so a zero or constant tensor cannot
pass (a token-shaped check cannot see a dequant-shaped fallback).

**The fixture writes no encoder tensors** (`ltx2_video_fixture.h:805-810 @
5a0ffe9e`), so an end-to-end test needs it extended; forgetting that yields a
load failure that reads as a code defect. SHA-anchored because this row EDITS
that file, which is the case #911 records: an anchor into a file the row is
itself changing is stale the moment the change lands.

---

## 5. Tests and evidence

Focused gate: `test_ltx2_video`, `test_ltx2_vae`, `test_ltx2_pipeline`,
`test_ltx2_loader`, `test_capi`.

1. **RED first, through the production entry point.** The smallest failing test
   drives `vllm_video_generate` with an `audio_path` extra and asserts the
   render consumed it. It fails first because the extra is refused as unknown.
   Per [`reachability.md`](../reachability.md), reachability is then proven by
   deleting the production call site and showing the test goes RED.
2. **Truncation polarity**, in two halves, because the second one is the half a
   digest cannot make. A take shorter than the clip is refused rather than
   padded. A take LONGER than the clip keeps its *leading* frames
   (`a2vid_two_stage.py:202`), and that is pinned by encoding the same file at a
   longer `audio_max_duration` and requiring the truncated latent to be
   bit-identical to the shorter window's — which a tail slice cannot produce,
   because the two encodes it slices have different lengths.
3. **Freeze**: the audio latent after the last phase is bit-identical to the
   encoded input. A mutation that noises the audio stream must turn this red.
4. **Refusals, each asserting the missing part is named**: wrong sample rate
   (both rates in the message), non-RIFF/compressed container, no audio stream,
   a mono file against a two-channel encoder, and `a2vid_two_stage` requested
   against a version the table does not carry.
5. **Config-parse observability.** The fixture's audio VAE deliberately declares
   a `sampling_rate`, an FFT size and a mel-bin count that differ from both the
   shipped values and the parser's own defaults, with a decoy on each
   obvious-looking neighbour. Without that, all three readings resolve to the
   same number whether they are right or wrong and the parser is unmutatable.
   The fixture carries the reasoning.
6. **A lower bound** on the encoded latent's magnitude, so a silently zeroed or
   constant tensor fails. A correlation or count-based check cannot see a scale
   error and is not used alone.

Every mutation records three facts: `git diff --stat`, whether it BUILT with any
compile error beside it, and the exit code. A mutation that fails to compile is
recorded as such and never counted as a passing test.

### What the mutation pass actually found

Focused gate `./build/tests/test_ltx2_video --test-case=*audio*`, 3 cases /
54 assertions green on the unmutated tree. Each mutation applied to one file,
built, run, then restored and the restore **verified by sha256** rather than
assumed.

| Mutation | `git diff --stat` | BUILT | exit | verdict |
|---|---|---|---|---|
| M1 delete the production call site (`a2v_audio_volume = encoded.data;`) | `ltx2_video.cpp \| 2 +-` | YES | 1 | DETECTED |
| M2 never zero the audio denoise mask | `ltx2_video.cpp \| 2 +-` | YES | 1 | DETECTED |
| M3 leave the frozen stream at the schedule's scalar sigma | `ltx2_video.cpp \| 2 +-` | YES | 1 | DETECTED |
| M4 zero-pad a short take instead of refusing | `ltx2_audio_input.cpp \| 2 +-` | YES | 1 | DETECTED |
| M5 accept any channel count | `ltx2_audio_input.cpp \| 2 +-` | YES | 1 | DETECTED |
| M6 accept any sample rate | `ltx2_audio_input.cpp \| 2 +-` | YES | 1 | DETECTED |
| M7 accept a window knob with no file | `ltx2_video.cpp \| 2 +-` | YES | 1 | DETECTED |

**M2 and M3 SURVIVED on the first run, and that is the useful part of this
section.** Both are the freeze claim, and both were made in a comment and
observed by nothing:

- **M2** survived because the trace was written *above* the noiser and reported
  the REQUEST's `audio_frozen` flag rather than the state's mask. So a build that
  noised and denoised the caller's take left the suite at 3 cases / 51
  assertions / exit 0. Fixed by moving the trace below `ApplyGaussianNoise` and
  deriving `audio_frozen` from the mask the loop actually uses.
- **M3** survived because upstream's `frozen` sets the scalar `Modality.sigma`
  as well as the per-token timesteps (`utils/types.py:104-106`), and the denoise
  mask cannot reach that input. Nothing observed it. Fixed by adding
  `audio_sigma_max` to the trace and asserting it is 0 on the frozen path and
  positive on the ordinary one — a control, so a build that pinned the sigma to
  zero unconditionally is caught too.

Two further notes on the harness, because both are the kind of failure that
reports as a verdict about the code:

- The first run crashed on non-UTF-8 output (a `CHECK` comparing two binary PPM
  buffers dumps raw pixels) **between applying M1 and restoring it**, leaving the
  mutation in the tree. The next gate run then showed 8 failures that read as the
  change's own. The harness now restores in a `finally`, and the test compares a
  count of differing bytes rather than two binary strings.
- `git diff --stat` read `(no diff)` for every mutation to `ltx2_audio_input.cpp`
  while that file was still untracked — the first of the three facts silently
  reporting nothing. Re-run after `git add`.

---

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported with `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the full pass/fail line, and positive controls for
`No space left` and `BFD assertion`. Known-red on `main` proven pre-existing
rather than asserted: the #873 checker family, `windows-msvc-*` (#584). A
load-dependent failure is re-run alone with the box load recorded before it is
charged to this row.

---

## 6b. Reachability — the sentence the records must carry

**A production entry point reaches this, and the test enters through it.**

The chain, by hand and with no unreached hop in it:

```
include/vllm.h  vllm_video_generate                    (:969)
  -> src/capi/vllm_c.cpp                               (:1646, engine->Generate(gen))
    -> vllm::multimodal::VideoEngine::Generate         (video_engine.h:157)
      -> Ltx2VideoEngine::Generate                     (ltx2_video.cpp)
        -> Ltx2DecodeAudioWav / Ltx2EncodeAudioToLatent
          -> a2v_audio_volume -> the audio StreamState -> Ltx2DitForward
```

The command-line arm is the same call: `ltx2-gen --audio-path` sets the
per-generation extra and calls `vllm_video_generate`, as a thin ABI client that
includes no internal header.

**The reachability mutation is M1 above.** Deleting the production call site —
the one line where the encoded latent reaches the stream — turns the focused gate
RED (exit 1). The gate therefore measures a capability rather than a class. That
matters here specifically: `Ltx2AudioEncoderForward` has had six test call sites
and **no** production one since `cefacd2d0`, which is exactly the "test-only
driver" shape `.agents/reachability.md` names, and this row is what ends it.

**What is NOT reachable, stated rather than left to be found.** The OpenAI
`/v1/videos` route cannot carry this, because
`VideoGenParamsFromRequest` (`video_engine.cpp:349-385`) never forwards
`VideoRequest::metadata` to `VideoGenParams::extras` — no per-generation LTX
extra reaches that route today, `image_crf` included. That is a pre-existing
defect of the route rather than of this row, it is recorded as S5 in §9, and the
reach claim above is worded to exclude it.

## 7. Quantized arms

The audio VAE is f32 on both halves by upstream's own choice
(`vocoder.py:585-595`, mirrored in `ltx2_audio_vae.cpp:1-12`), so the audio
encoder introduces no new quantized arm and the encoder loads through the same
`Ltx2LoadVaeWeights` path the decoder already uses.

Arm by arm, so none is left to be discovered:

| Arm | Disposition |
|---|---|
| bf16 / f32 safetensors | **ported** — the shipped audio VAE is this, and the encoder reads it |
| NVFP4, FP8 (the DiT tower) | **unaffected and unbroken.** This row seeds a latent and freezes a stream; it adds no GEMM and changes no dtype on the DiT path. Both arms reach the A2V path exactly as they reach the ordinary one |
| GGUF k-quants | **not applicable**, and not merely undone. Upstream ships no GGUF arm for any LTX-2 component: `quantization_factory.py:23-26` enumerates the inference kinds exhaustively as fp8-cast, fp8-scaled-mm, nvfp4-cast and nvfp4-prequant, with `assert_never` at `:50`. There is no upstream behaviour to mirror and no quant-matched llama.cpp comparison to serve, because llama.cpp does not carry this architecture |
| int8-convrot | **out of scope**, unchanged by this row and already refused by name at `ltx2_pipeline.cpp`'s `kInt8ConvRot` |

The one dtype question this row does owe an answer to: the ingestion and the
encoder are **f32 throughout**, which matches the decoder half it shares every
primitive with and matches upstream, where `encode_audio` runs the mel front-end
in the waveform's own dtype (`ops.py:54`) and the encoder in the parameters'
(`audio_vae.py:260`, `:273`). No buffer here is wider than the arm it feeds.

---

## Owed

- **`RetakePipeline`** — [#924](https://github.com/mudler/vllm.cpp/issues/924).
  Reasoning in §2.
- **Text-to-audio** (`t2a_one_stage.py:43,109`) — absent, not absorbed here.
- **The `a2vid_two_stage` RECIPE.** This row ports the audio-conditioning
  mechanism and rides the checkpoint's own resolved recipe; upstream's A2Vid
  stage 1 is a caller-configured guided schedule with `a2v_guidance_scale` as the
  guider's `modality_scale` (`a2vid_two_stage.py:226-240`, `:353-360`), which
  `distilled_two_stage` does not reproduce. Reasoning in §2. Needs the recipe
  table entry, the two guider-parameter sets, and a `steps`/`cfg` surface to
  carry them — and it should land with a real-checkpoint comparison rather than
  before one, because a sigma table with no render to check it against is the
  kind of plausible-and-wrong this campaign has already paid for twice.
- **A real-checkpoint A2V render.** Gated on fixtures only; the GPU was out of
  bounds for this row.
- **Value goldens from executed upstream for the INGESTION path.** §5 promised
  these and this row does not deliver them, so they are owed here rather than
  left as a claim nobody checked. What already exists is not the same thing:
  `Ltx2AudioEncoderForward` and the mel front-end are golden-gated against
  executed upstream by `scripts/gen-ltx2-vae-goldens.py` (cases 7a–7d) from an
  earlier row, and this row rides those. What has NO executed-upstream golden is
  the part this row added — the config parse, the `start_time`/`max_duration`
  window, and the truncation. Those are pinned instead by upstream `file:line`
  anchors plus the differential assertions in §5.2, which is weaker: an anchor
  proves the line was read and a differential proves the direction, and neither
  proves the value. Needs a generator that instantiates
  `AudioEncoderConfigurator.from_metadata` and `decode_audio_from_file` at
  reduced dimensions, which needs PyAV and a written fixture file, and is why it
  did not ride here.
- **Arbitrary-ratio resampling** — refused by name; needs the polyphase kaiser
  resampler upstream uses at `ops.py:40`.
- **Compressed audio containers** — refused by name; no demuxer is vendored.
- **`Ltx2CreateAudioLatentState` and `Ltx2ConditionAudioByReference` remain
  test-only.** Both have exactly one call site each and it is a test
  (`test_ltx2_vae.cpp:2412`, `:2431`). This row does not wire them, because the
  engine's `StreamState` (`ltx2_video.cpp:168`) is a different type from
  `Ltx2LatentState` (`ltx2_conditioning.h:61-69`) — `double` positions, no
  `pos_dims`, an extra `keyframes_mask` — and bridging them is its own change.
  Recorded so the next reader can tell a deliberate deferral from an oversight.
- **The silent-ignore paths found while surveying**, each filed rather than
  fixed in flow because each needs a contract decision this row has no mandate to
  make: [#927](https://github.com/mudler/vllm.cpp/issues/927) (seven typed
  `VideoGenParams` / `VideoModelParams` fields LTX-2.5 accepts and never reads,
  all seven of which H3 does read) and
  [#928](https://github.com/mudler/vllm.cpp/issues/928) (`/v1/videos` forwards no
  per-generation extra to any engine). Detail and read/write sites in §9.

## 9. Silent-ignore paths found in the audio surface (see `## Owed` above)

Recorded here because the row that finds them owes a statement of who owns them.

| # | Path | Status |
|---|---|---|
| S2 | `VideoModelParams::audio_vae_config_path` is written by `ltx2-gen --audio-vae-config` and `server_main.cpp:1231` and **read by nothing** in the LTX-2 engine, which takes the config from the checkpoint's `__metadata__` instead. H3 does read it. Same defect for `video_vae_config_path`. | [#927](https://github.com/mudler/vllm.cpp/issues/927) |
| S4 | `audio_flow_shift` is parsed, validated positive (`video_api.cpp:227-236`), threaded to `gen.audio_flow_shift` (`video_engine.cpp:360`) and **read by nothing** in `ltx2_video.cpp`. Same for `flow_shift` and `task`. H3 reads all three. Seven such fields in total. | [#927](https://github.com/mudler/vllm.cpp/issues/927) |
| S5 | `VideoRequest::metadata` is **never forwarded** to `VideoGenParams::extras` (`video_engine.cpp:349-385`), so no per-generation LTX extra is reachable over HTTP at all. `image_crf` included, which makes image-to-video unreachable over the route entirely, since the only served value cannot be sent. | [#928](https://github.com/mudler/vllm.cpp/issues/928) |

S5 bounds this row's reach claim honestly: the capability is reachable from
`include/vllm.h` and from `ltx2-gen`, and **not** from `/v1/videos` until S5 is
fixed, because that route drops every per-generation extra today. Claiming the
route without saying so would be the exact defect this section exists to catch.

## 10. Stop conditions

Stop and report rather than widening scope if: the audio encoder's config cannot
be resolved from the shipped checkpoint's metadata; the phase-recipe change
cannot be made without altering an existing recipe's behaviour; or the
`READER ANCHORS` gate cannot be re-derived deterministically.

## 11. Now

`ACTIVE` — spec committed before implementation, per `AGENTS.md` § *Spec before
code*.
