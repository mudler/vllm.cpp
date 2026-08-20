# LTX 2.5

Use this page for LTX 2.5 checkpoints, commands, supported arms, and current limitations.

The two-stage pipelines use `--upsampler-path` and `--lora-path`. The
corresponding upstream flag is `--distilled-lora`. Set the output geometry with
`--num-frames` and write the result with `--output-dir`.

Retake requests use `--retake-start-time`, `--retake-end-time`, and
`--retake-frame-rate`. Use `--regenerate-video` and `--regenerate-audio` to
select which part of the requested window changes.

Set `VLLM_CPP_LTX2_TEXT_ENCODER` when the text encoder is outside
`CHECKPOINT_ROOT`. Set `VLLM_CPP_LTX2_TOWER_E2E=1` to opt in to the full text
tower gate.

## LTX-2.5: what runs, and what it cannot do

LTX-2.5 is reachable as video family `ltx-2.5`, through the same
`vllm_video_engine_load` / `vllm_video_generate` C ABI that serves MiniMax-H3,
and through the `ltx2-gen` example that drives it. Its two VAE decoders, its two
VAE ENCODERS with the mel front-end, the conditioning items that place encoded
latents into the token stream, and its pipeline layer (the sigma schedule, the
diffusion steps, guidance, the latent spatial x2 upsampler, the duration head and
the embeddings connector) are implemented and gated. The latent **temporal** x2
upsampler is implemented and gated too, but no pipeline here drives it — see the
`--upsampler` note below. Several limits decide what you can actually ask for,
and each refuses by name rather than rendering something else.

**Image conditioning (image-to-video) runs at `image_crf=0`, and only there.**
Pass a first frame as binary PPM (`first_frame_path` / `first_frame_ppm`) plus
the per-generation extra `image_crf=0`; the engine decodes it, aspect-fills and
centre-crops it to each phase's own resolution, VAE-encodes it, and replaces
latent frame 0's clean tokens. `noise_aug` is the pinning strength (`1.0`, the
default, pins the frame exactly).

`image_crf=0` must be asked for **explicitly**, and it is **out of
distribution**. Upstream re-compresses a conditioning image through H.264 at the
CRF the checkpoint's generation was trained with, and an LTX-2.5 checkpoint
resolves that to **18**. That round trip needs libx264 and no codec is vendored
here, so a non-zero CRF — including the default a caller gets by saying nothing —
is refused by name. `image_crf=0` is upstream-legal (upstream short-circuits it
and documents an explicit `0` as "skip re-compression entirely") but conditions
the model on pixels it was not trained to see. That is a render-quality cost, and
it is stated rather than applied silently.

A **last-frame keyframe is served** as of the token-APPEND seam. A keyframe is
*appended* to the token sequence with its own pixel positions, denoised as part
of a longer sequence, and trimmed back off before the latent is unpatchified,
where the first-frame arm only REPLACES tokens that already exist. It takes the
same `image_crf=0` and `noise_aug` as the first-frame arm, and both may be
supplied at once. Two things a previous version of this paragraph got wrong are
worth naming, because a reader may have acted on them: there is no rebuilt
attention mask — a supplied keyframe passes `attention_mask=None` and upstream
returns no mask for it — and the sigma schedule keeps reading the TARGET token
count rather than the grown one, because upstream derives its shift from the
unpatchified target. (Until 2026-08-13 this paragraph said a last-frame keyframe
needs the DiT's unported `keyframes_abs_pos_embedding`. That was wrong: a
supplied keyframe is appended unmarked, so the embedding never applies to it.
Where the embedding does bite is the FIRST latent frame of every render, which
was a separate gap; it was closed on 2026-08-14 under issue #658, so the marker
is now applied on every render.)

**Generated keyframe slots are a different feature, and they are now SERVED.**
Upstream also lets the model *generate* extra frames at interior positions,
`--num-generated-keyframes N` there and the per-generation extra
`num_generated_keyframes` here. That is not a keyframe you supply; it is one you
ask the model to invent, and each slot buys one pixel frame at the cost of a
full latent frame of tokens. `0` is upstream's own default and means off, so
passing it explicitly renders normally. A positive count places that many
evenly spaced INTERIOR slots: both endpoints are dropped, because frame 0
already spans a single pixel frame under causal encoding and the last frame is
the clip's own end. The slots are marked with the trained keyframe embedding,
denoised with the video, and read back out of the state before the extra tokens
are trimmed away.

Two refusals remain, and they are upstream's own rather than ours. A negative
count is refused, and so is a count the clip is too short for: every slot is an
interior position, so `N + 2` frames are the minimum.

**This page said until 2026-08-16 that a positive count was refused, and it is
recorded rather than deleted** because a reader may have planned around it. The
refusal named the readback as its one blocker, and it was right: what landed
under issue #986 is the layout that locates the slots exactly and the extraction
that runs before the trim. One third of what that refusal named is still owed,
and it is a different surface rather than a smaller version of this one, the
standalone single-frame decode that would hand you slot PIXELS. Nothing here
returns those: the slots stay in latent space, which is what DFR below wants
from them.

Reference-image, reference-video and reference-audio conditioning are still
refused, each naming a different missing piece. **Two reasons this page used to
give are now false and are recorded rather than deleted**, because a reader may
have planned around them: the IC-LoRA scale factors are read as of `--lora`
(2026-08-15), and the token-APPEND machinery landed with the last-frame keyframe
above (2026-08-16). What is left for reference VIDEO and reference IMAGE is a
pixel path and a stage split. Nothing here turns a clip into latents: upstream
decodes the reference at `height/downscale x width/downscale`, keeps frame 0 and
then every Nth frame, and encodes the result
(`ltx_pipelines/iclora_utils.py:87-89`, `:112-148`), and this engine's only
pixel-to-latent route encodes exactly one frame at the phase's full resolution.
And the reference item belongs to stage 1 only: upstream fuses the adapter into
stage 1 and gives stage 2 `loras=()` and no reference item at all
(`ic_lora.py:108`, `:119`, `:314-321`), while this engine holds ONE DiT, fused
at load, that every phase runs. Reference audio additionally needs the AUDIO
VAE's encoder key filter, which is not built. Three encoder-level limits are worth
stating in advance because they are refusals rather than approximations. A
reference waveform whose sample rate differs from the audio VAE's is refused
rather than resampled, since upstream uses a polyphase kaiser resampler this
project does not carry. A VAE configured with `latent_log_var: none` is
refused, because upstream itself raises on it. And a video-VAE `res_x` encoder
block that declares no `num_layers` is refused rather than defaulted, because
upstream subscripts that key and raises `KeyError` on it; no other encoder block
kind reads it.

**A typed prompt works.** `--encoder` names the Gemma-4 12B text tower and
`--prompt` carries the words. The tower tokenizes them with its OWN embedded
tokenizer — the shipped encoder stores `tokenizer.json` as a TENSOR, so there is
no sibling file to point at — runs, aggregates all 49 hidden states, projects
them to 4096 and 2048, and passes both streams through the embeddings connector
before cross-attention. The tower is ~24 GB of host bf16 and stays resident,
because a prompt arrives per request.

One tokenization detail is a KNOWN DIVERGENCE rather than a mirror, and it is
checkpoint-conditional: upstream tokenizes through the HuggingFace `__call__`
with its default `add_special_tokens=True`, so it runs the tokenizer's
post_processor, while this port calls the plain encode and prepends BOS by hand.
On the shipped checkpoint the two are identical — its post_processor declares an
EMPTY special-token map, measured on the shipped file rather than assumed — so
nothing is lost today. A checkpoint whose post_processor DID add tokens would
tokenize differently here.

`--encoder-config` supplies the Gemma config, and it is required for the only
shipped encoder: `vonkaiser`'s
`gemma4-12b-with-proj-nvfp4-torchao.safetensors` carries no `__metadata__` at
all. An encoder that declares one (the official bf16 build does, under
`__metadata__["gemma_config"]`) needs no flag, and supplying both is refused
rather than resolved — `layer_types`, `global_head_dim`,
`num_global_key_value_heads` and `attention_k_eq_v` each resolve a different
tower out of a byte-identical tensor set.

Without `--encoder`, conditioning comes from `--prompt-embeds` plus
`--audio-prompt-embeds`: rows of little-endian f32, 4096 wide for the video
stream and 2048 for the audio stream, with the same row count in both. A
`--prompt` with no tower is refused, and supplying only one of the two files is
refused, because a stream left unconditioned renders instead of failing.

**Asking what a clip was conditioned on.** `Ltx2VideoEngine::last_conditioning()`
returns the trace of the last `Generate()` — whether the conditioning came from a
prompt or from embeds, the prompt string, the row count and both stream widths, an
FNV-1a digest over the exact f32 buffers cross-attention read, and each stream's
absmax. When the request carried an image it also reports the CRF and strength it
was conditioned at, how many tokens the encoded image replaced, and a digest over
**those tokens as written into the state** — not over the encoder's output, so a
build that encoded an image and never placed it reads as unconditioned rather
than healthy. It is returned **by value, under the engine's own lock**, so it is safe to
call from a server thread while another thread renders — but `Generate` holds that
same lock for the WHOLE render, so such a call blocks for minutes rather than
returning a stale answer immediately. `completed` is true only if that
`Generate()` returned: the trace is filled before the denoise loop, so a
render that throws later leaves a populated trace behind, and this flag is what
separates the two.

It is a **change detector, not a quality measure**. It answers "did this render
depend on this prompt, through these weights" and nothing else — it does not say
the conditioning values are the ones upstream would produce.

The text path runs on the CPU even when `--device cuda` puts the DiT on the GPU:
everything in the text encoder is f32 by declaration and its device arm is owed.
That is one host-side 12B forward over the prompt's own tokens per request,
against a denoise loop of many 21B forwards.

**Either source goes through the embeddings connector.** Both shipped LTX-2.5 DiTs
carry two `*_embeddings_connector` families, 129 tensors each, and they are the
8-layer 1-D transformer upstream runs between the caption projections and the
DiT's cross-attention. The render applies it with the checkpoint's own weights,
under the checkpoint's own `connector_*` configuration. Two consequences for the
command line: the row count must be a multiple of the connector's learnable
register count (128 on the shipped files), and `--prompt-valid-rows N` says how
many of those rows are real tokens. The rest are padding, and padding is not
inert here: the connector REPLACES it with its learnable register table, so a
run that leaves the default renders as if every supplied row were caption.
`--prompt-valid-rows` applies to the embeds path only — with `--encoder` the
tokenizer supplies the mask, which is what that flag exists to stand in for.

**The DiT config is required when the checkpoint does not carry one.** The
shipped `vonkaiser` FP8 transformer has no `__metadata__` at all, and the values
a config decides are ones no tensor shape encodes: `frequencies_precision` and
`av_ca_timestep_scale_multiplier` move every RoPE angle and every audio/video
modulation. Defaulting them resolves a different model from the same file, so
the loader refuses and `--dit-config` supplies LTX-2.5's declared values.

```sh
ltx2-gen --dit  ltx-2.5-22b-distilled-fp8.safetensors \
         --dit-config ltx-2.5-transformer-config.json \
         --model-version 2.5 \
         --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
         --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
         --upsampler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
         --encoder gemma4-12b-with-proj-nvfp4-torchao.safetensors \
         --encoder-config ltx-2.5-gemma4-text-config.json \
         --prompt "a red fox running through deep snow at sunrise" \
         --frames 25 --width 320 --height 192 --seed 20260812 \
         --device cuda --workdir /tmp/ltx25 --out /tmp/ltx25/video.mp4
```

Swap the two `--encoder*` flags and `--prompt` for `--prompt-embeds` +
`--audio-prompt-embeds` to condition from files instead.


## Where the render spent its wall: `phase-log.json`

Every completed LTX-2.5 render writes **`<workdir>/phase-log.json`** beside the
frames, on the shipped default and behind no flag, and `ltx2-gen` prints its
path on the line after `wrote N frames`. An embedder gets the same path from
`vllm_video_last_phase_log(engine)` (ABI v23) rather than by guessing a filename.

**"Completed" is the whole of it, and it is worth reading literally.** The table
is written by the success path and by nothing else: the two write sites sit
immediately before a successful return, and every guard above them throws past
them. A render that is killed, that a lease governor aborts, that refuses on a
guard, or that is still running leaves **no `phase-log.json` at all** — not a
truncated one and not an empty one — and `vllm_video_last_phase_log` returns
`NULL`. A 2.5-hour render stopped at 2.4 hours tells you nothing about where it
was. Making a running or killed render legible is
[#1413](https://github.com/mudler/vllm.cpp/issues/1413), and it is a separate
lane from this file.

It is a flat, non-overlapping timeline of named phases — the DiT load, the two
VAE loads, the text tower and the connector, the denoise, the video and audio
decodes, the frame and WAV writers — each carrying a start and end measured from
the load, a duration, a **peak host byte** count and a **peak device byte**
count. These fields say how complete it is, and how far it carries:

| Field | What it means |
|---|---|
| `wall_seconds` | from the engine load to the end of this generation |
| `sum_leaf_seconds` | the named phases, which do not overlap |
| `unaccounted_seconds` | the difference — time inside no named phase |
| `sum_rule` | which records the sum adds: `span=false` **and** `nested=false` |
| `sampler_enabled` | whether the 100 ms sampler ran, or the peaks are boundary-only |
| `notice` | **NOT A BENCHMARK**, and why — carried in the file rather than in a document a later reader would have to know to look for |

Some phases are **decomposed rather than partitioned**. `denoise` carries one
`denoise.step` per denoiser evaluation, `decode.video` carries
`decode.video.chunk` per streamed chunk, `decode.audio` carries
`decode.audio.mel` and `decode.audio.vocoder`, and a two-stage recipe's
`phase.prepare` carries `phase.upsample_latent`. Those records are marked
`nested`, are printed for the reader, and are **excluded from
`sum_leaf_seconds`** — they are inside a leaf that is already counted, so adding
them would make `unaccounted_seconds` the residue of double counting instead of
time nobody named.

A nested record is also what makes a phase NAME checkable. A leaf that claims to
cover the denoise must enclose its own `denoise.step` records; one that stops
short of the loop, or that hands the back half of it to a neighbouring name, no
longer does. The three phases that carry a render each carry such an anchor for
that reason.

**Do not read a duration here as a measurement of this machine.** Every number
is wall clock under whatever else the box was doing, which the file does not
record: on one contended host the same binary at the same geometry has moved
from 0.147 s to 4.463 s of wall between two runs a minute apart, and the rank of
its two largest phases has reversed between such runs. The ratio
`sum_leaf_seconds / wall_seconds` is stable across all of them; the seconds are
not.

`unaccounted_seconds` is emitted rather than distributed over the phases,
because a table whose parts do not add up has a phase nobody named, and a
plausible-looking table is worse than an obviously incomplete one. On a
completed 64x64/9-frame render it is under 2% of wall.

`peak_device_bytes` is the **driver's** in-use figure through the backend's
`DeviceMemoryInfo`, and it is `-1` where no probe answers. It is not zero there,
because a byte count of zero and a byte count nobody took are different facts.

**Today it answers `-1` on CUDA as well as on the CPU**, and that is worth
knowing before you read a table full of `-1` as a finding: `CudaBackend` does
not implement `DeviceMemoryInfo`
([#1126](https://github.com/mudler/vllm.cpp/issues/1126)); ROCm is the only
backend that does. On a unified-memory board such as GB10 the `peak_host_bytes`
column is not a poor substitute — host and device are one pool there, and
`nvidia-smi --query-gpu=memory.used` reports `[N/A]` on that board while
`--query-compute-apps=used_memory` answers.

Two environment variables, both measurement lanes rather than configuration:
`VLLM_RENDER_PHASE_LOG_STDERR=1` also prints the table as a fixed-width block —
including when the file itself cannot be written, which is the case that lane
exists for — and `VLLM_RENDER_PHASE_SAMPLER=0` stops the 100 ms sampler thread,
which narrows the per-phase peaks to what the phase boundaries saw and removes
nothing else.

The timeline starts at the **engine load**, because on a 22B checkpoint the DiT
staging is minutes paid at the front of every render. A process that loads a
second engine starts a new timeline, so the table describes the last load.


## While the render runs: the `[render]` lines

The table above is written by a generation that **returns**. A render that is
killed, aborted by a lease governor, or still going writes none, so LTX-2.5 also
narrates itself on stderr as it goes, on the shipped default and behind no flag:

```text
[render] + load                     t=0.000s
[render] + load.dit                 t=0.001s
[render] - load.dit                 t=0.002s dur=0.001s host=0.01GiB
...
[render] - load                     t=0.027s dur=0.027s host=0.02GiB
[render] + generate                 t=0.027s
...
[render] + denoise                  t=0.027s
[render] + denoise.step             t=0.027s
[render]   dit forward 1  phase 0 step 1/8  t=0.027s
[render] - denoise.step             t=0.065s dur=0.038s host=0.02GiB
[render] + denoise.step             t=0.066s
[render]   dit forward 2  phase 0 step 2/8  t=0.066s last=0.038s
[render] - denoise.step             t=0.069s dur=0.003s host=0.02GiB
...
[render] - denoise                  t=0.146s dur=0.118s host=0.02GiB
[render] + decode.video             t=0.146s
[render] - decode.video             t=0.147s dur=0.001s host=0.02GiB
[render] + artifacts.frames         t=0.147s
...
[render] + decode.audio             t=0.148s
[render] + decode.audio.mel         t=0.148s
[render] - decode.audio.mel         t=0.155s dur=0.008s host=0.02GiB
[render] + decode.audio.vocoder     t=0.155s
[render] - decode.audio.vocoder     t=0.236s dur=0.081s host=0.02GiB
[render] - decode.audio             t=0.236s dur=0.089s host=0.02GiB
[render] - generate                 t=0.237s dur=0.209s host=0.02GiB
```

**That is a real capture**, from the CPU gate render at 64x64 over 9 frames — the
seconds and the byte counts are that render's, on a contended box, and nothing
here is a benchmark. It is shown at this scale on purpose. The same lines from a
21.004 B render would carry a `load.dit` of minutes and a `last=` on the order of
[#1375](https://github.com/mudler/vllm.cpp/issues/1375)'s measured 162 s per DiT
forward, and **no such render has been captured yet** — that is W1's lease. A
worked example at that scale would be a projection, and a projection printed in a
public document gets quoted back as a measurement.

Read it as three things:

* **The last line names what is running.** A phase prints when it opens, not
  only when it finishes, so a run that stops inside a phase still says which one.
  Between the banner and `wrote N frames` there was previously nothing at all,
  and a working render and a hung one were the same observation.
* **`last=` is the per-forward cost.** Seconds since the previous DiT forward,
  measured by the process doing the work rather than inferred from outside it.
* **`step k/N` is exact; the forward counter has no denominator.** The sampler
  decides how many denoiser calls a step takes and the guider decides how many
  forwards each call is (one to four), so a total would be a guess. Two forwards
  per step is what `cfg_scale != 1.0` alone buys; a guider that also runs the
  STG and modality legs does four, which is what the device-resident arm does
  now that [#1092](https://github.com/mudler/vllm.cpp/issues/1092) gave
  `Ltx2DitForwardDevice` its `perturbations` argument. The `k/N` fraction is
  unaffected either way: it reads the recipe phase's own `sigmas`.

`VLLM_RENDER_PROGRESS=0` silences them. It is a measurement lane so an A/B over
what the emitter costs runs on one binary, not a setting to turn off: the cost is
one flushed `fprintf` per phase boundary and per forward — on the order of a
hundred writes against hours of wall — and nothing is emitted per token or per
VAE tile.

Add `--first-frame frame.ppm --image-crf 0` for image-to-video. The PPM is
binary P6 at maxval 255 (no PNG/JPEG codec is vendored); `--image-crf 0` is
required and is not the default, because omitting it resolves the checkpoint's
own CRF 18 and refuses — see the out-of-distribution note above.

Add `--audio-path take.wav` for **audio-to-video**: the render is conditioned on
a soundtrack you supply rather than one the model invents. The take is encoded
through the audio VAE's encoder and then held frozen through every denoise
phase, and the `audio.wav` that comes back is your own input rather than a VAE
round trip. `--audio-start-time` seeks into the file and `--audio-max-duration`
caps how much is read; both default to covering exactly the clip's duration, and
either without `--audio-path` is refused rather than ignored.

What is upstream's here is the **conditioning mechanism** — decode, encode,
truncate to the clip, freeze — and not the denoise schedule. Upstream's
audio-to-video stage 1 is a caller-configured guided one, with its
`a2v_guidance_scale` acting as the guider's modality scale, while a take here
rides whichever recipe the checkpoint resolves, in practice `distilled_two_stage`
with fixed sigmas. So the audio drives the render, and no claim is made that the
result reproduces upstream's own audio-to-video output.

The WAV has to match the checkpoint already: 16-bit PCM RIFF/WAVE, the audio
VAE's own sample rate (16 kHz on the shipped one), its encoder's channel count
(2), and at least as long as the clip. None of the four is converted. There is
no resampler for an arbitrary ratio here and no demuxer at all, and a take
shorter than the clip is an error upstream too, so each mismatch is refused with
both numbers in the message — a resampled-wrong, upmixed or silence-padded take
renders a finished clip conditioned on audio nobody supplied. This needs an
audio VAE that carries encoder weights; a decoder-only one refuses by name.

### The supported resolution envelope

**`--width` and `--height` are enforced, and an unsupported value is refused by
name.** Both must be multiples of the VAE's spatial factor (32) times the worst
downscale the recipe's phases apply — so **64 on the distilled two-stage recipe**,
whose first phase runs at half resolution, and **32 on a one-stage recipe**. Those
are upstream's own two numbers (`assert_resolution`,
`ltx-pipelines utils/helpers.py:540-551`), reached by upstream's derivation rather
than hardcoded, so a recipe that downscaled further would tighten the divisor with
it. The refusal names the offending axis — width, height, or both — the divisor,
and a size you can actually pass: the nearest legal one at or below the request,
or, when an axis is smaller than the divisor and no such size exists, the
smallest legal size there is.

Until 2026-08-15 nothing enforced this and the engine floored instead: a
two-stage request of width 80 rendered 64 and returned success, and a one-stage
request of width 100 rendered 96 ([#919](https://github.com/mudler/vllm.cpp/issues/919)).

**`--frames` is NOT enforced, and it rounds.** A frame count is floored onto the
VAE's temporal grid, `(frames - 1) / 8 * 8 + 1`, so 100 frames renders 97. This
mirrors upstream, which floors an explicit `num_frames` identically
(`ltx_core/types.py:113`) and validates it nowhere: its `snap_frames_to_grid`
helper is called from the auto-duration path and from the dubbing pipeline, and
that pipeline takes no frame count at all — it snaps one read from a reference
video's container. No frame count a caller supplies is snapped or checked, in
either project. Pass a value of the form `8k + 1` to get exactly what you asked
for. The rounding is observable either way: `result.frame_count`, `result.width`
and `result.height` report what was actually rendered, not what was requested.

Omitting all three renders the recipe default, which is 1024x1536 at 121 frames
and is a much larger request than it looks.

**What is legal is not what fits.** The first two rows below are a property of
this port and are enforced. The rest are scale markers, and the last three are
measurements of one box rather than limits of the code:

| | Value |
|---|---|
| Legal sizes | any multiple of 64 (two-stage) or 32 (one-stage), on both axes |
| Legal frame counts | any; non-`8k + 1` values floor onto the temporal grid |
| Upstream's default output | 1024x1536 at 121 frames (`utils/constants.py:42-76`) |
| Upstream's HQ preset output | 1088x1920 at 121 frames (`utils/constants.py:95-98`) |
| **Measured to complete on one GB10** | **704x448 at 25 frames** in 4231 s, 448x256 at 25 frames in 3085 s, and 320x192 at 25 frames. One run each, 16 to 17 August 2026, `main` `0b0b8900f` |
| Largest size tried | 704x448 at 25 frames. 1024x576 was not attempted to completion because another session claimed the box. That is scheduling and not an envelope, so 704x448 is not a ceiling |
| Superseded, kept for the record | 448x256 at 25 frames was published here as *not* completing, on a run that lost about 59 GB in 24 s after its denoise. It completes, and that loss did not recur |

Those three completions are one run each on one contended box, with no oracle on
either side, so read them as what has been observed and not as a limit. There is
no maximum-size check anywhere in this path.

The 59 GB stays on the page because it is the reason the old row gave, and it
belongs to its own run: a prompt-embeds render with no text tower that an armed
watchdog ended at 13.77 GiB against an 18 GiB floor, rather than the engine
failing. That run is rung F1 in `.agents/benchmark-record.md`. The loss was never
attributed to the decode, whose own heap peak at that size is 361.72 MiB, some
170x too small, and attributing it is still open as
[#1014](https://github.com/mudler/vllm.cpp/issues/1014). It did **not** reproduce
on `0b0b8900f` under a 2 s memory guard that would have seen it: the 448x256 rung
floors `MemAvailable` at 38.96 GiB over 1289 samples and the 704x448 rung at
38.89 GiB over 1743 samples, with no sample under 34 GiB on either and a peak use
of 80 of 119 GiB. See the note below on what bounds a render, and
`.agents/specs/ltx25-tiled-decode.md` and
`.agents/specs/ltx25-resolution-envelope.md`.

`--lora ic-lora.safetensors [STRENGTH]` fuses an IC-LoRA adapter into the DiT at
load, mirroring upstream's `--lora PATH [STRENGTH]`
(`ltx-pipelines/utils/args.py:600-611`). The strength is optional and defaults to
1.0. It is a LOAD-time flag, not a per-request one, because the adapter is fused
into the weights and cannot vary between generations - upstream takes it as a
`DiffusionStage.from_checkpoint` constructor argument for the same reason
(`ic_lora.py:104-114`).

The adapter is a safetensors file of `.lora_A.weight` / `.lora_B.weight` pairs,
with or without ComfyUI's `diffusion_model.` prefix. It works on every arm the
DiT loads - bf16, FP8 and NVFP4 alike - because those are all dequantized to
bf16 before the delta is added. Two things REFUSE by name rather than
proceeding quietly: an adapter naming a module this port does not bind (upstream
would skip it, and a skip cannot be told apart from a typo), and an adapter that
fuses into nothing at all.

**A second `--lora` does NOT refuse, and this page said it did until 2026-08-17.**
Only one adapter is accepted, and the library enforces that
(`ltx2_lora.cpp:243-248` fails on more than one, citing `dubit.py:364-365` and
`hdr_ic_lora.py:271-272`). But `ltx2-gen` cannot construct the two-adapter vector
that trips it: `SetExtra` (`examples/ltx2_gen/main.cpp:212-221`) overwrites an
existing key in place, so `--lora a --lora b` leaves one `lora_path` extra
holding `b`, silently fuses `b`, and exits 0. Pass one adapter.

The C ABI cannot reach it either, and that is the wider half of the finding:
`Ltx2VideoEngine::Load` carries the ONLY `dit_options.loras.push_back` in the
tree and it runs at most once, under `if (!lora_path.empty())` — named by symbol
rather than by line, because the line moved with #1118 and a stale anchor is what
this paragraph already had to correct once. So `loras.size()` is 0 or 1
on every production path — CLI, `vllm_video_engine_load` and the server alike —
and the more-than-one refusal is reached only by `test_ltx2_lora`. It is correct
code guarding a state nothing can currently construct, which is the shape
N-adapter fusion ([#932](https://github.com/mudler/vllm.cpp/issues/932)) will
need. Tracked as [#1097](https://github.com/mudler/vllm.cpp/issues/1097).

Supplying an adapter also reads its `reference_downscale_factor` and
`reference_temporal_scale_factor` metadata (`iclora_utils.py:30-49`). Those are
what a reference video needs, and reading them was what the reference refusal
used to say was missing. It no longer says that, and it does not say
token-append either: that seam landed too. What it names now is the reference
CLIP's own pixel path and the stage split, both above.

`--upsampler` is what the distilled recipe's second phase needs. Without it that
phase refuses rather than skipping: its three-step refinement is what makes the
upscaled latent valid, and decoding the half-resolution latent instead would hand
back a smaller clip that looks like a completed request. `--max-phase 0` stops
after the first phase deliberately.

It must be the **spatial** upsampler,
`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`. Lightricks also ships
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`, which is the same
class with `temporal_upsample: true` in its config and the same
`upsampler.0.*` tensor names — so it loads and runs, and returns a latent with
`2f - 1` frames at the ORIGINAL resolution where this phase needs the original
frame count at double resolution. It is `2f - 1` and not `2f` because that arm
doubles the frame axis and then drops the first frame, which upstream encodes as
a single pixel frame. Passing it is refused by name rather than
reported as a shape mismatch. The temporal arm itself is implemented and gated
against upstream, but **nothing drives it**: its only upstream consumer is
`DFRPipeline`'s multi-round loop. The DFR pipeline's BASE is ported as of issue
#986 and is described below, and the rounds loop is not, so there is still no
flag that makes a request use that file and no reason to pass it today. The
checkpoint is also not published beside the spatial one on the mirror this port
was built against, so nothing here has run it on real weights.


## The DFR pipeline: `--pipeline-kind dfr`

Detail-fidelity rendering. It is upstream's `DFRPipeline`, and it differs from
the ordinary distilled two-stage recipe in its CONDITIONING rather than in its
schedule: both stages run the same sigmas, and stage 1 is the same half
resolution. What DFR adds is a keyframe grid.

**The canvas is padded, and this is the part that surprises people.** DFR lays
keyframes on a segment grid, 24 or 32 frames per segment, whichever pads least,
and it pads `num_frames - 1` up to a whole number of segments before it renders
anything. A 9-frame request therefore denoises a 25-frame canvas and is trimmed
back to 9 before you see it. Ask for 121 frames and you get 121; ask for 9 and
the machine does about three times the work you might expect.

**Frame counts are refused here rather than floored.** Everywhere else in this
engine a frame count that is not `8k + 1` is floored onto the latent grid, and
that is documented above as the behaviour. DFR cannot live with it: every
keyframe position it emits has to land on a latent border, so `--frames 10` is
refused with the reason rather than quietly rendered as 9.

**`num_generated_keyframes` is refused on this pipeline.** DFR chooses its own
slot positions from the canvas, one per segment boundary, and the whole pipeline
is indexed by that grid. An override would leave the slots and the canvas
describing different frames, and the render would still finish. Use
`--pipeline-kind distilled_two_stage` or `one_stage` if you want to place slots
by count. An explicit `0` still passes, because that is upstream's default.

**How to reach it.** `pipeline_kind` is a LOAD knob, not a per-generation one, so
all three surfaces carry it: `ltx2-gen --pipeline-kind dfr`, the C ABI's
`vllm_video_model_params.extra_keys` / `extra_values`, and the server's
`--video-extra pipeline_kind=dfr` at launch. A server started that way renders
every `/v1/videos` request through DFR.

The two knobs beside it are per-GENERATION and therefore **ABI only**, because
`/v1/videos` forwards no per-generation extra to any engine yet (issue #928):
`num_generated_keyframes` on the other pipelines, and `temporal_upsample_rounds`
below. This paragraph said "CLI and ABI only" until 2026-08-17, and the CLI half
was never true — `examples/ltx2_gen/main.cpp` carries no flag for either name, so
`vllm_video_gen_params.extra_keys` is the only surface that reaches them.


## LTX-2.5 text-to-audio: a render with no picture

`--pipeline-kind t2a_one_stage` runs upstream's `T2AOneStagePipeline`, which
generates a soundtrack and no video at all. The result carries an `audio.wav`,
`frame_count = 0`, an empty frame directory and **no ffmpeg argv**, because there
is nothing to mux.

```sh
ltx2-gen --dit ltx-2.5-dit.safetensors \
         --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
         --encoder gemma4-12b-with-proj.safetensors --encoder-config gemma4.json \
         --pipeline-kind t2a_one_stage --device cpu \
         --frames 121 --prompt "rain on a tin roof, distant thunder" \
         --workdir /tmp/t2a
```

**`ltx2-gen` has no `--steps` flag, and this recipe carried one until 2026-08-17.**
The step count comes from the resolved recipe (`ltx2_video.cpp:2900`), and the
`vllm_video_gen_params.num_inference_steps` field that would override it
(`include/vllm.h:1072`) has no flag on this binary — `minimax-h3-gen` and
`music3-gen` both expose `--steps`, which is where the published line came from.
An unknown argument is not ignored here: `examples/ltx2_gen/main.cpp:318-321`
prints `unknown argument` and exits 2, so the command as published could not run
at all. Overriding the step count needs the C ABI today.

**These file names are not a checkpoint pin, and no LTX-2.5 recipe in this
document is.** None of them names a HuggingFace repo, a revision or a sha256,
which AGENTS.md § *Say which weights, and from where* requires; MiniMax-H3 and
MiniMax-Music3 below each carry a full table and LTX-2.5 carries none. That is
campaign-wide and pre-existing rather than particular to this recipe, and it is
recorded rather than invented, because no LTX-2.5 arm here has been rendered on
real weights yet. Tracked by
[#1048](https://github.com/mudler/vllm.cpp/issues/1048); read `--dit` above as
"the LTX-2.5 transformer", which the other recipes on this page spell as
`ltx-2.5-22b-distilled-fp8.safetensors` together with the `--dit-config` its
missing `__metadata__` requires.

**No `--video-vae` is needed**, and none is loaded: upstream's pipeline never
constructs a video VAE. `--width` and `--height` are **refused** rather than
ignored — upstream passes a 512x512 placeholder whose height and width it
documents as unused, and only the frame count and the recipe's frame rate are
read, to derive the duration.

**It is a GUIDED arm, and that changes what it costs and what it needs.** The
distilled video recipes run one DiT forward per step. This one runs **three** by
default — conditional, unconditional, and one with the audio
self-attention perturbed (STG) — so it is roughly 3x the work per step, and it
**requires a text tower**, because the unconditional pass conditions on the
negative prompt. Loading with `prompt_embeds_path` alone gets a refusal naming
`--audio-cfg-guidance-scale 1.0` as the way to turn the unconditional pass off.

It was the only guided arm here until row LTX25-GUIDED-VIDEO
([#1092](https://github.com/mudler/vllm.cpp/issues/1092)) gave the joint video
path its own denoiser; see *LTX-2.5 video guidance* below.

Six per-generation knobs mirror upstream's own CLI, and each takes the
checkpoint generation's value when absent: `--negative-prompt`,
`--audio-cfg-guidance-scale` (7.0), `--audio-stg-guidance-scale` (1.0),
`--audio-rescale-scale` (0.7), `--audio-skip-step` (0) and `--audio-stg-blocks`
(28 on the 2.3-and-later lineage), which is comma separated. A block index
outside the DiT's own layer count is refused rather than clamped. There is no
`modality_scale` knob: upstream pins it to 1.0 for this pipeline, because
audio-only generation has no video modality to isolate.

`--audio-rescale-scale` acts on the **denoised (x0) prediction**, not on the
DiT's velocity, because upstream's guider sits behind an `X0Model` and combines
already-converted tensors. The distinction is invisible at `0.0`, where the two
readings agree exactly, and it changes the render at every other value — so a
recipe or a script that was tuned against the velocity reading will not
reproduce here at the default `0.7` (issue #1039).

Being per-generation, those six reach the CLI and the C ABI and **not**
`/v1/videos`, which forwards no per-generation extra to any engine (issue #928).
`pipeline_kind` is a LOAD knob and does reach the server, so a server started
with `--video-extra pipeline_kind=t2a_one_stage` renders every request as audio
at the recipe's own guider values.

**The accelerator is refused by name.** `device = 1` gets a refusal on this
pipeline: the device forward takes both streams by reference and this pipeline
has no video stream to give it. Use `--device cpu`.


## LTX-2.5 video guidance: `--pipeline-kind one_stage`

`one_stage` mirrors upstream's `TI2VidOneStagePipeline`, which builds a
`FactoryGuidedDenoiser` from the params table's own video and audio guiders. On
the 2.4/2.5 lineage those resolve to `cfg_scale = 3.0`, `stg_scale = 1.0`,
`rescale_scale = 0.7` and `modality_scale = 3.0`.

Until [#1092](https://github.com/mudler/vllm.cpp/issues/1092) this port read none
of it: the joint denoise loop ran one unguided forward per step. A `one_stage`
render therefore finished, at the right size and frame count, along a different
trajectory than upstream's. It now runs **four** forwards per step and combines
them per modality:

| Pass | What differs | Selected by |
|---|---|---|
| conditional | nothing | always |
| unconditional | the negative conditioning | `cfg_scale != 1.0` |
| perturbed | video/audio self-attention skipped on `stg_blocks` | `stg_scale != 0.0` |
| isolated modality | the audio<->video cross attention off in every block | `modality_scale != 1.0` |

Seven per-generation knobs mirror upstream's `default_1_stage_arg_parser` and
each takes the checkpoint generation's value when absent. The audio row and
`--negative-prompt` are shared with text-to-audio and are no longer refused on a
video pipeline; upstream's parser carries both rows side by side, and the old
refusal rested on a reading of upstream that was wrong and harmless only while
nothing here read them.

| `ltx2-gen` flag | per-generation extra | meaning |
|---|---|---|
| `--video-cfg-guidance-scale` | `video_cfg_guidance_scale` | video `cfg_scale`; `1.0` turns the unconditional forward off |
| `--video-stg-guidance-scale` | `video_stg_guidance_scale` | video `stg_scale`; `0.0` turns the perturbed forward off |
| `--video-rescale-scale` | `video_rescale_scale` | video `rescale_scale`, applied to the DENOISED prediction |
| `--video-skip-step` | `video_skip_step` | `0` never skips; `n` runs every `n+1`-th step |
| `--video-stg-blocks` | `video_stg_blocks` | comma separated block indices; EMPTY disables STG, see below |
| `--a2v-guidance-scale` | `a2v_guidance_scale` | video `modality_scale`; `1.0` turns the isolated-modality forward off |
| `--v2a-guidance-scale` | `v2a_guidance_scale` | audio `modality_scale` |
| `--negative-prompt` | `negative_prompt` | the unconditional forward's conditioning |

The audio row is the same six spellings with `audio_` in place of `video_`:
`audio_cfg_guidance_scale`, `audio_stg_guidance_scale`, `audio_rescale_scale`,
`audio_skip_step`, `audio_stg_blocks`, and `v2a_guidance_scale` for its
`modality_scale`.

Those extras ride the per-generation `extra_keys` / `extra_values` array on
`vllm_video_params`, so the C ABI reaches the same path with no new field. They
are per-GENERATION and therefore reach the CLI and the C ABI and **not**
`/v1/videos`, which forwards no per-generation extra to any engine
([#928](https://github.com/mudler/vllm.cpp/issues/928)). `pipeline_kind` is a
LOAD knob and does reach the server, so a server started with
`--video-extra pipeline_kind=one_stage` renders every request through the guided
denoiser at the recipe's own guider values and no request can change them.

**An EMPTY `--video-stg-blocks` is accepted and means "perturb no block".** That
is upstream's own idiom — `docs/multimodal-guidance.md:13` says "Set to `[]` to
disable STG", the field defaults to `[]`, the flags are `nargs="*"`, and the
shipped HQ params row uses it — and it stays distinct from OMITTING the flag,
which takes the params table's value. It disables the STG signal and not the STG
cost: upstream selects the perturbed pass from `stg_scale` alone, so the forward
still runs and contributes exactly zero. Set the scale to `0.0` to skip the
forward as well. This page and this port refused the empty list until
2026-08-17.

**The unconditional forward needs a negative conditioning, and there are two
ways to supply one.** With a text tower, `--negative-prompt` (or the recipe's
own default) is encoded through the same chain as the positive prompt. Without
one, `--negative-prompt-embeds` and `--negative-audio-prompt-embeds` — the LOAD
extras `negative_prompt_embeds_path` and `negative_audio_prompt_embeds_path` —
are the negative half of the `prompt_embeds_path` fallback: two files at the
DiT's two cross-attention widths, the same row count as the positive pair. Being
LOAD extras they DO reach the server, through `--video-extra`. With neither, a
`cfg_scale` other than 1.0 is **refused by name** rather than served the positive
context twice, which would leave the whole classifier-free term at exactly zero.

**A block index the checkpoint does not have is refused**, which is the case the
empty list above is NOT. `stg_blocks` is a membership test upstream, so naming
block 28 on a model with fewer blocks perturbs nothing and leaves
`stg_scale * (cond - perturbed)` at exactly zero — the same zero, reached by a
request that disagrees with the checkpoint rather than by a caller who asked for
no perturbation. Upstream never meets it because it only ships 48-block
checkpoints, so this refusal is local to this port and is named as such.

**The distilled and retake recipes refuse every one of these flags.** Their
guidance is distilled into the weights, so honouring an override would sample a
trajectory the weights were never trained for. Their guiders are upstream's
positive-only one, so they still issue one forward per step and their output is
unchanged by this row.

**All four passes run on the accelerator too, and that costs up to twice the
render.** `Ltx2DitForwardDevice` took no perturbation argument until 2026-08-19,
so `device = 1` refused the perturbed and isolated-modality passes rather than
run them unperturbed with both terms at zero. It takes one now, and the refusal
is gone. What replaces it is a cost: at the model's own guider defaults a step
assembles four forwards on `device = 1` where it assembled two, so a 30-step
render is **120 forwards rather than 60**. That count is exact. The denoise TIME
is **at most 2.0x** and no measurement of it exists — two of the four passes do
less work than the two that were already running, because the isolated-modality
pass skips both cross-modality attentions in every block. Plan against 2.0x as a
ceiling, not as an estimate.
`--video-stg-guidance-scale 0 --audio-stg-guidance-scale 0 --a2v-guidance-scale 1
--v2a-guidance-scale 1` buys that back and is the trajectory the accelerator arm
had before, at the cost upstream's defaults are there to avoid.

**What is not served.** `temporal_upsample_rounds` is defined and refused above
`0`: the rounds loop that temporally doubles the latent, re-tiles the canvas and
stitches it back is not ported. The refusal names it, and it names three things
that are NOT the reason, because each is the one a reader reaches for first: the
temporal upsampler operator is ported and gated, the canvas and tiling
arithmetic is ported and gated in this same change, and the generated keyframe
slots are served. What has no counterpart here is the per-tile denoise pass as a
callable. Stage 2's x2 spatial detailing IC-LoRA is refused separately, for the
reasons the reference-video arm is refused above.

On the server, `--video-family ltx-2.5` pins the family instead of detecting it,
and `--video-extra KEY=VALUE` (repeatable) carries the same family-specific load
knobs the flags above map onto. Both are described under
[the server's video flags](#video-family-and-family-specific-load-knobs).

**Three things about that command are worth knowing before you run it.**

*It is bounded by HOST WALL CLOCK, well below the recipe's own defaults.*
Staging the 21.00B FP8 transformer costs about 44 GB on a 119 GB GB10, and
`--encoder` adds the text tower on top of that — roughly 24 GB of host bf16 that
stays resident, because a prompt arrives per request. Every memory figure here
was measured WITHOUT the tower, on the prompt-embeds path, so budget for both.
**320x192, 448x256 and 704x448 at 25 frames all complete** through both distilled
phases. The upper two took 3085 s and 4231 s, measured on 16 to 17 August 2026 at
`0b0b8900f`. This page used to say 448x256 did not complete, and that is what
changed. Unified memory makes those host bytes and this class of box reboots
rather than OOM-killing, so start small and grow, and put a memory watchdog in
front of anything larger. Those runs kept one at a 2 s cadence and it never came
near firing: the `MemAvailable` floor was 38.9 GiB and no sample fell under
34 GiB. The recipe default of 1024x1536 at 121 frames is far beyond what one
GB10 holds today.

Expect tens of minutes, not seconds, and expect much of that to be independent of
the resolution you asked for. Most of a render is no longer the host VAE decode.
[#1041](https://github.com/mudler/vllm.cpp/issues/1041) threaded that decode, and
what dominates now is a **single-threaded phase of about 1731 s that barely moves
with size**: 1731 s and 1732 s across two rungs whose voxel counts differ by
2.75x, which is 57 to 66% of wall on each. Which phase that is has not been
identified, and [#1087](https://github.com/mudler/vllm.cpp/issues/1087) owns
naming it. The decode itself still has no device arm and still runs at 0% GPU
([#1007](https://github.com/mudler/vllm.cpp/issues/1007)).

Read every figure in the last two paragraphs as one run per geometry on a shared
box that was contended, with no oracle on either side. Two rungs establish no
scaling law, and 704x448 is not a ceiling: the next rung up was stopped by
another session claiming the box, not by the machine.

The decode is no longer *single-threaded*, which is what this section used to
say. The decode's convolutions now dispatch across `VLLM_CPP_CPU_THREADS` workers
(default `hardware_concurrency`), bit-identical at every worker count —
[#1009](https://github.com/mudler/vllm.cpp/issues/1009), measured at **roughly
9x on 16 to 20 workers** against one. Take the band rather than a decimal: the
medians are 9.15x at 16 and 9.14x at 20, but those two counts spread 21-23% run
to run on a box that was not idle, where every count at or below 8 spreads under
7%. Read it as a decode figure and not a render one: the ~9x was taken on a
synthetic decode shape on a contended 20-core x86 host, and end to end it does
not appear, because the phase #1041 never touched is now most of the wall
(#1087). The renders above are the post-change re-measurement of that wall. Set
`VLLM_CPP_CPU_THREADS` lower if the render has to share the box.

*The render behind those numbers was NOT prompted, and it renders a scene without
rendering YOUR scene.* It was the EMBEDS path — `--prompt-embeds` with
`--prompt-valid-rows 24`, over synthetic N(0, 0.2) rows, with no text tower on the
path at all. With the connector wired the shipped 21.00B FP8 transformer produced
a temporally coherent photorealistic clip at 320x192 / 25 frames: consistent
subject, consistent background, frame-to-frame motion, where before the connector
the same weights at the same settings produced smooth colour fields. But 104 of
its 128 connector rows were the connector's own trained `learnable_registers`
table, which is what upstream substitutes at PADDED positions, and the other 24
were noise. So what conditioned that clip is the checkpoint's own learned default,
not a depiction of anything anyone asked for — and on the embeds path it could not
be otherwise, because rows read from a file are whatever you put in them rather
than an encoded caption. Ask a `--prompt-embeds` run for a subject and you will
not get it.

*Nobody has yet run the command above end to end, and this page claims nothing
about what it renders.* The typed-prompt path is gated all the way through —
tokenizer, Gemma-4 tower, connector, cross-attention — but the gate is a
REDUCED-DIMENSION synthetic encoder under CPU Release, with no real checkpoint
anywhere in it. A real-checkpoint prompted render is OWED. Until it runs, neither
claim is available: not that `--prompt "a red fox…"` puts a fox on the screen, and
not that it fails to. `last_conditioning()` answers a narrower question — that the
render depended on your prompt, through these weights — which is not the same
question as whether the frames depict it.

LTX-2.5 ships two video decoders behind one checkpoint field. The convolutional
one is implemented; the higher quality diffusion one (`NADiffusionDecoder`) is
not, and asking for it fails with a message naming the missing
neighborhood-attention kernel. It never falls back to the convolutional decoder,
because that would hand back a lower quality render as if it were the one you
asked for.

**The sentence that used to follow was stale and is retired here.** It said
keyframe and reference conditioning were refused because "only the decoder is
ported". The video VAE **encoder** is ported and is kept resident
(`ltx2_video.cpp:1007-1012`), the first-frame and last-frame keyframe arms are
SERVED — the same page says so at the image-conditioning section above — and what
remains refused is REFERENCE conditioning, for reasons that have nothing to do
with the encoder: the reference clip has no pixel path and stage 2 must run
unfused (`ltx2_video.cpp:1955-1990`,
[#975](https://github.com/mudler/vllm.cpp/issues/975)). Reference AUDIO is refused
separately (`ltx2_video.cpp:1991-2004`). A refusal whose stated reason has been
removed is worse than no reason, because a reader plans around it.

**The convolutional decode is TILED and STREAMED, on upstream's own defaults, and
there is no knob.** The layout is the one `ltx_pipelines` builds for a Conv VAE
when you pass `AUTO_TILING`: a 768 px tile with a 64 px overlap on the long side,
aspect coupled to the short one, and 80 frame temporal chunks overlapping by 24.
Each temporal chunk is written to its PPM files and dropped, so the full pixel
volume never exists at once. Two consequences worth knowing before you read a
memory number:

- **Below a 768 px long side and 81 frames the layout does not tile at all.** A
  single tile comes out, and that path reproduces the untiled decode bit for bit
  (`test_ltx2_tiling`'s one tile control, on both causality settings). So
  448x256/25f renders byte identically to how it rendered before tiling existed,
  and its memory is unchanged. Tiling starts doing something at 896x512, and
  temporal chunking at 81 frames.
- **A tiled render is not the same image as an untiled one**, and that is
  upstream's behaviour, not a defect here. Each tile decodes a crop of the latent,
  the decoder's receptive field is wider than the 64 px overlap, and the seam is
  blended rather than eliminated. Do not compare a 1920x1088 render against a
  hypothetical untiled one and read the difference as an error.
- **81 to 120 frames is already the tiled regime, and the recipe default is
  inside it.** The default request is 1024x1536 at 121 frames. At 81 frames the
  latent is 11 frames deep against a 10 frame temporal tile, so it splits into two
  chunks. Measured on the shipped conv VAE at 64x64 / 81 frames: max abs diff
  0.0503 against the untiled decode, on an output whose own max is 0.7513 — 6.70%
  of that range — with 962983 of 995328 channel values (96.75%) not bit identical.
  So nearly every value moves, by a few percent of the signal. If you need the pre
  tiling render back, ask for 73 frames or fewer.

**The refusal that used to stand here is gone, and what replaced it is an owed
ORACLE rather than an owed feature.** Through L10 this page said a prompt was
refused because the `Embeddings1DConnector` weights, which ship inside the DiT
file, were among the modules the DiT loader would not load. They are loaded
(`Ltx2LoadConnectorWeights`, `ltx2_loader.cpp:1292 @ b5756ea8c`, enumerates their
own contract at `:1295`, outside the DiT's),
so `encoder_path` is accepted, `has_encoder()` is true, and a prompt no longer
needs a matching pair of embeds files. The gap that remains is a numeric one: the
tower, the connector's forward and both caption projections each have an oracle
against executed upstream, and the two JOINS between them —
`create_embeddings`, and the render composition that chains it onto the tower's
output — have none. Upstream's `EmbeddingsProcessor.process_hidden_states` is
that whole chain in one function and is the oracle this owes; until it is
executed, the composition's VALUES rest on the per-brick oracles either side of
it. That is also why `last_conditioning()` is described above as a change
detector and not as a check on the conditioning.

## LTX-2.5: reproducing the DiT parity gate

**This section is the DiT's own parity gate, not the way to run LTX-2.5.** The
render path ships and is documented above under
[LTX-2.5: what runs, and what it cannot do](#ltx-25-what-runs-and-what-it-cannot-do):
`ltx-2.5` is one of the two registered video families
(`REGISTER_VLLM_VIDEO_FAMILY` at `src/vllm/multimodal/ltx2_video.cpp:3723 @ b5756ea8c`), the
Gemma-4 text tower loads from `--encoder` (`ltx2_video.cpp:1149`) and sets
`has_encoder` (`ltx2_video.cpp:1191`), both VAEs and the pipeline layer are implemented
(`ltx2_video_vae.cpp`, `ltx2_audio_vae.cpp`, `ltx2_pipeline.cpp`), and the
`/v1/videos` routes register for whatever family `--video-dit` resolves —
`server_main.cpp` calls the family-agnostic `LoadVideoEngine` and then prints the
resolved family. What follows here is how to regenerate the DiT's goldens. The
C++ surface is `include/vllm/model_executor/models/ltx2.h`, and it refuses by
name every arm it does not carry (a non-f32 stream dtype, the 19B
caption-projection checkpoint form, keyframe absolute-position embeddings, the
video-only / audio-only model types).

Provenance, so this can be re-checked rather than trusted: the paragraph above
replaces one that arrived at `3d89f6fc4` — the first LTX commit, where it was
true — and was never revisited as L3 through L13 built each of the six pieces it
denied.

The prompt-K/V cache (`Ltx2PromptKvCache`) is reusable across the DENOISE STEPS of
one prompt, and only those. It records a fingerprint of the prompt it was filled
for, and a forward whose context tensors, context geometry or prompt masks differ
from that prompt is refused by name rather than served K/V that would render the
cached prompt. Call `Ltx2PromptKvCache::Reset()` to rebind the same allocation to
a new request.

The gate runs the UPSTREAM modules at reduced dimensions on CPU, so it needs a
Lightricks LTX-2 checkout and the system `python3` with torch — **no checkpoint, no
venv and no gated download**. Regenerate the goldens and run it:

```sh
git clone https://github.com/Lightricks/LTX-2 ~/_git/LTX-2
python3 scripts/gen-ltx2-goldens.py \
  --ltx2 ~/_git/LTX-2 \
  --out tests/vllm/models/ltx2_goldens.inc
cmake --build build --target test_ltx2 && ./build/tests/test_ltx2
```

The generator asserts the `ltx_core` it imported came from that checkout and not
from anything installed in site-packages, and it writes the upstream revision it
executed into the generated header. Neither side checks in a weight byte: both
rebuild every tensor from one deterministic stream keyed by the parameter's name.

The pipeline layer has its own gate, and it needs a second checkout: the recipe
table is read from vLLM-Omni, which is the binding oracle for LTX even though it
carries no 2.5 row of its own. Both checkouts must be CLEAN, because a revision
anchor read from a tree with uncommitted edits stamps a SHA the goldens do not
come from.

```sh
git clone https://github.com/vllm-project/vllm-omni ~/_git/vllm-omni
python3 scripts/gen-ltx2-pipeline-goldens.py \
  --ltx2 ~/_git/LTX-2 \
  --vllm-omni ~/_git/vllm-omni \
  --out tests/vllm/models/ltx2_pipeline_goldens.inc
cmake --build build --target test_ltx2_pipeline && ./build/tests/test_ltx2_pipeline
```

If you regenerate that `.inc` against a moved upstream, expect the goldens to
carry the change rather than only the pin cases. The pipeline goldens reach the
GroupNorm eps and group count in the latent upsampler, the connector's
`rms_norm` eps, the `BlurDownsample` width (on the 1.5 arm only, since the blur
runs on the rational denominator) and the Res2s `sigma_up` clamp — that last one
on the eta = 1 arm, where the clamp binds on every step. A regeneration that
moves one of those constants alone reds a value comparison; one that moves the
constant AND the tensors together passes it, and is caught only by the cases that
compare each constant against upstream's own signature. Both layers are there
deliberately, and neither is redundant.
## The Gemma-4 text tower gate, and the interpreter it needs

The text tower is gated against the UPSTREAM HuggingFace implementation built and
run at reduced dimensions. It needs a `transformers` that registers
`gemma4_unified` in `CONFIG_MAPPING` — **5.8 or newer; 5.3.0 does not have it and
fails in a way that reads exactly like "Gemma-4 is unsupported"**. The generator
refuses such an interpreter by name rather than emitting goldens from a tower it
could not build.

```sh
/path/to/venv/bin/python scripts/gen-ltx2-gemma-tower-goldens.py \
  --out tests/vllm/models/ltx2_gemma_tower_goldens.inc
cmake --build build --target test_ltx2_text_encoder && ./build/tests/test_ltx2_text_encoder
```

No checkpoint and no download: the reduced config comes from
`tests/vllm/models/ltx2_gemma4_text_config.json`, which is the
`__metadata__["gemma_config"]` of the official bf16 text encoder, and every weight
is rebuilt on both sides from the deterministic stream. The tolerance is not a
constant — the generator MEASURES how far upstream's own answer moves between f32
and bf16 and emits that per state as the bound.

Two more gates want the real checkpoint. The prompt-token goldens are regenerated
from the tokenizer the text encoder ships **as a tensor**, and the end-to-end case
dequantizes the 12B tower to roughly 24 GB of host bf16, so it is opt-in rather
than checkpoint-presence gated:

```sh
TE=$CHECKPOINT_ROOT/ltx-2.5/vonkaiser-fp8-nvfp4/text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors
/path/to/venv/bin/python scripts/gen-ltx2-prompt-tokens-goldens.py \
  --text-encoder "$TE" \
  --out tests/vllm/models/ltx2_prompt_tokens_goldens.inc

# Gate the real vocabulary against the generated token goldens.
CHECKPOINT_ROOT=... ./build/tests/test_ltx2_text_encoder --test-case="ltx2 prompt: REAL*"

# The full 12B tower needs approximately 33 GB of host memory and minutes of CPU time.
CHECKPOINT_ROOT=... VLLM_CPP_LTX2_TOWER_E2E=1 \
  ./build/tests/test_ltx2_text_encoder --test-case="ltx2 e2e*"
```

## LTX-2.5 quantized loaders

`include/vllm/model_executor/models/ltx2_loader.h` materializes the shipped
LTX-2.5 checkpoints: the FP8 DiT, both NVFP4 DiTs, and the torchao-NVFP4 Gemma-4
text encoder with its embedded tokenizer. These are the entry points the render
path itself drives: `--dit` (`--video-dit` on the server) reaches
`Ltx2StreamDitToDevice` / `Ltx2LoadDitFromSafetensors` at
`ltx2_video.cpp:815-816 @ b5756ea8c`, and `--encoder` (`--video-encoder`) reaches
`Ltx2LoadTextEncoderFromSafetensors` at `ltx2_video.cpp:1149`. This section
documents them at the library level, where the gate below runs.

**Ten coordinates into `ltx2_video.cpp` and `ltx2_loader.cpp` were wrong, at
eleven citation sites on this page** — `ltx2_video.cpp:893` was cited twice.
Five of the replacements carry `@ b5756ea8c`, one per affected passage; the bare
`:NNN` beside a pinned one belongs to the same file at the same revision.
Nothing else on this page is pinned, so read an unpinned coordinate as
unverified.

They were re-derived on 2026-08-17 from the sentence making each claim rather
than by reading whatever sat at the cited line, and they were off by 40 to 2200
lines: the family registry was cited at `:1529` and lives at `:3723`, and
`has_encoder` was cited at `:893` where the assignment is at `:1191`. Every
symbol existed, so every citation looked plausible; the tell was only that
nothing at the cited line mentioned it. No gate here checks a documentation
anchor ([#632](https://github.com/mudler/vllm.cpp/issues/632),
[#911](https://github.com/mudler/vllm.cpp/issues/911)), so a pin is the only
thing that lets a reader tell a stale coordinate from a moved one.

The two NVFP4 checkpoints were written by different producers that disagree about
both the group-scale framing and which nibble holds which weight, so the loader
resolves the producer from the `torchao_nvfp4` marker: present means torchao
(`to_blocked` framing, low-nibble-first), absent means the Lightricks
`nvfp4-prequant` tool (cuBLAS-padded framing, high-nibble-first). A marker whose
stored scale shape contradicts it, and a marker-less file whose shape is the
`to_blocked` framing or neither framing, are refused by name rather than guessed,
because both readings type-check and produce finite, correctly scaled, wrong
weights.

The refusal cannot cover everything, and the limit is worth knowing before you
point this loader at a checkpoint it was not built for. A marker-less NVFP4 file
whose `weight_scale` is stored **linear** `[N, K/16]` — what ModelOpt,
llm-compressor and compressed-tensors write, none of which emit a
`torchao_nvfp4` sidecar — has, whenever `N % 128 == 0` and `K/16 % 4 == 0`, a
shape indistinguishable from the cuBLAS-padded one. Such a file is resolved as
`nvfp4-prequant` and read swizzled and high-first: it loads, and it is wrong.
Only the LTX-2.5 DiT is gated against an independent oracle here, so treat any
other marker-less NVFP4 checkpoint as unsupported until it is. See
`.agents/specs/nvfp4-nibble-order.md`.

Two behaviours a caller has to know. `Ltx2LoadDitFromSafetensors` ACCEPTS both
shipped DiTs with no opt-in as of 2026-08-14. `Ltx2DitLoadOptions::allow_unported_modules`
still exists, and still loads the ported subset while reporting every dropped
family in `Ltx2DitCheckpoint::unported`, but neither shipped LTX-2.5 checkpoint
needs it any more. `keyframes_abs_pos_embedding` was the last family on that
list; it is PORTED (issue #658), and `prompt_adaln_single` /
`audio_prompt_adaln_single` left the list the same way on 2026-08-13. The two
DiTs used to be refused from OPPOSITE directions — the vonkaiser FP8 copy for
carrying a trained `keyframes_abs_pos_embedding` this port did not apply, and the
first-party NVFP4 copy for declaring `use_keyframes_abs_pos_embedding` while
carrying no tensor at all. The second case is upstream-legal and means "apply
nothing": upstream builds the parameter on the meta device and
`supports_keyframes_abs_pos_embedding` stays False, so
`Ltx2AdoptDeclaredDitParams` resolves the declared flag against what the file
actually carries rather than refusing it or inventing a zero. The two
`*_embeddings_connector` towers are
**not** among them and never will be:
`UnportedFamilies` (`ltx2_loader.cpp:573 @ b5756ea8c`) filters them out at `:582`
through `LoadedElsewhere` (`ltx2_loader.cpp:569`), `RefuseUnported`
(`ltx2_loader.cpp:592`) says so in its own message at `ltx2_loader.cpp:608-611`,
and `Ltx2LoadConnectorWeights` loads them under their own contract — which is
what the video engine calls, so a checkpoint this port reads completely is never
made to ask for `allow_unported_modules` on their account. (The "five" this
paragraph used to say arrived at `5966ffef3` and was true until `e48c86253`
added `LoadedElsewhere` — the same claim the "what runs" section above already
retired, which survived here because it was never swept for.) And loading is
**bf16** by default, the checkpoint's own model dtype; `widen_to_f32` is opt-in
and exists only for the f32 parity forward.

`Ltx2StreamDitToDevice` is the GB10 arm. It dequantizes and uploads one tensor at
a time so peak residency is the device copy plus one tensor, and it stages at
load because host-resident weights measure 20 to 30 percent slower there.

## The DiT is not always quantized, and the FULL model never is

**`--dit` accepts an UNQUANTIZED bf16 transformer as of 2026-08-17**
([#1148](https://github.com/mudler/vllm.cpp/issues/1148)). Until then `PlanDit`
refused any DiT carrying neither `U8` nor `F8_E4M3`, and the file it refused is
the one most of these pipelines need: upstream's table
(`packages/ltx-pipelines/CLAUDE.md:17-30` @ `fd4ded7f`) marks `Full` or
`Full + distilled LoRA` for `TI2VidOneStagePipeline`, `T2AOneStagePipeline`,
`TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `A2VidPipelineTwoStage`
and `KeyframeInterpolationPipeline`. `one_stage`, `t2a_one_stage`,
`res2s_two_stage` and `a2vid_two_stage` are all reachable here, so all four
could previously only run against a *distilled* checkpoint — a different
sampling regime that renders plausibly and says nothing.

Nothing about the arm is a new decoder. Unquantized is upstream's ordinary case:
`_DTYPE_CASTABLE` (`single_gpu_model_builder.py:51-57` @ `fd4ded7f`) is
float32/float64/float16/bfloat16, and uint8-NVFP4 and float8 are what that file
calls "quantized payloads". `Ltx2DitCheckpoint::quant` reports which of the
three the file was, and a BF16 weight is stored as it is, so the memory format
is what the checkpoint chose.

**A dtype this loader cannot read is still refused, by name.** The refusal now
lists the dtypes the file holds and the four encodings the loader materializes
(BF16, F32, F8_E4M3 with an F32 `<name>_scale`, and U8 with an F8_E4M3
`<name>_weight_scale` plus an F32 `<name>_weight_scale_2`). An `F16` DiT is the
live case: upstream's castable set lists `torch.float16` and this port has no
F16 materialization. The message it replaced said "use the L2 path", which was
advice a reader could not follow — `Ltx2LoadDitFromSafetensors` *is* the L2 path
and calls the refusing function on its first line.

**The full model costs ~42 GB resident.** It is 21.004 B parameters at two
bytes, not a widening: no path in this loader turns a bf16 weight into anything
else, and `widen_to_f32` stays opt-in. That does not fit one GB10 beside a
24 GB text tower, so the arm has been gated on reduced fixtures and on the real
file's *header*; a full materialization and a render on real weights are still
owed ([#1048](https://github.com/mudler/vllm.cpp/issues/1048)).

## LTX-2.5 DiT weights: which file, and how to tell them apart

Repo [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) at
revision `6c7e5e573ac1667efc83407806fe9b0b93730e60`, read from
`/api/models/Lightricks/LTX-2.5` on 2026-08-17. Sizes below come from the same
API's tree listing.

| Arm | File under `diffusion_models/` | Bytes | sha256 |
|---|---|---:|---|
| unquantized bf16, FULL (dev) | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` (the local copy; see below) |
| unquantized bf16, distilled | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | not obtainable here |
| NVFP4 (`nvfp4-prequant`), distilled | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | not obtainable here |
| `int8-convrot`, REFUSED (ComfyUI-only) | `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | not obtainable here |
| `int8-convrot`, REFUSED (ComfyUI-only) | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | not obtainable here |

The distilled two-stage recipes also require
`loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`. The file is
8,899,889,568 bytes and contains 3,320 BF16 tensors. They form 1,660
`lora_A`/`lora_B` pairs. Its metadata sets `lora_rank` and `lora_alpha` to
`450`, and sets `model_version` to `2.5.0`.

This distilled LoRA is not the 327,322,640-byte
`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`. The IC-LoRA
serves a different pipeline arm.

**The hub will not give you a content hash for this repo, and it does not say
so.** `Lightricks/LTX-2.5` is gated — an unauthenticated `resolve` returns
`Access to model Lightricks/LTX-2.5 is restricted` — and the tree API answers an
unauthenticated caller with an `lfs.oid` that is **one character repeated 64
times**, for every LFS file in the repo. It is the right length, it is
lowercase hex, and `len(oid) == 64` passes. All 14 LFS files share it, which is
the only cheap tell. So a pinning script that reads that field records five
different checkpoints under one fabricated digest and reports success. Pinning
the other four by content needs an authenticated fetch and is owed
([#1048](https://github.com/mudler/vllm.cpp/issues/1048)); the dev row above is
the sha256 of the copy on this project's NAS, computed locally, and it has not
been compared against the published artifact because there is nothing here to
compare it to.

**The two bf16 transformers are exactly the same SIZE**, and the file name is
the only cheap thing that separates them. Both are 4349 tensors, both carry the
same four `__metadata__` keys with `model_version` `2.5.0`. So a mislabelled or
re-downloaded copy cannot be caught by `ls -l`, and nothing here validates the
checkpoint *class* at load
([#1137](https://github.com/mudler/vllm.cpp/issues/1137)): pointing
`--pipeline-kind res2s_two_stage` at the distilled file renders in the wrong
sampling regime with no diagnostic.

Read from the FULL model's own header on 2026-08-17, by parsing its
677,616-byte JSON prologue and no payload: 4349 tensors, every one
`model.diffusion_model.`-prefixed, **4059 BF16 and 290 F32**, zero names ending
in `_scale`, `_scale_2` or `torchao_nvfp4`, 48 blocks,
`keyframes_abs_pos_embedding` present and TRAINED as `BF16 [1, 4096]`, the 290
F32 tensors being exactly the six `*scale_shift_table*` families, and the data
end plus the 8-byte length plus the header equal to the file size.

The `vonkaiser/LTX-2.5-FP8-NVFP4` FP8 DiT is a separate repo and is pinned where
the FP8 recipes name it; it carries no `__metadata__` at all, which is why those
recipes need `--dit-config`.

The gate needs the three checkpoint headers, a vLLM checkout and an LTX-2
checkout (the two nibble-order authorities); it reads a few hundred bytes at
their own offsets and never a payload:

```sh
python3 scripts/gen-ltx2-quant-goldens.py --vllm ~/_git/vllm --ltx2 ~/_git/LTX-2 --checkpoint-root "$CHECKPOINT_ROOT" --out tests/vllm/models/ltx2_quant_goldens.inc
cmake --build build --target test_ltx2_loader && ./build/tests/test_ltx2_loader
```


## LTX-2.5 text conditioning

This documents **one brick of the shipped render path** — the text conditioning
the DiT consumes — and how to reproduce its gate. The render itself is above
under [LTX-2.5: what runs, and what it cannot do](#ltx-25-what-runs-and-what-it-cannot-do);
`--encoder` is what puts this brick on that path, and `has_encoder` is set at
`ltx2_video.cpp:1191 @ b5756ea8c` once the tower loads.

LTX-2.5 does not condition on a text encoder's last hidden state. It takes every
Gemma-4 hidden state (the embedding output plus all 48 decoder outputs, 49 in
total), normalizes them, concatenates across the layer axis, and projects the
result twice: a 4096-wide video caption projection and a 2048-wide audio one.
That is why the shipped projections take 3840 x 49 = 188160 inputs.

Two things about the shipped checkpoint are easy to trip over:

* the tokenizer is stored **as a tensor**, `tokenizer_json`, alongside
  `hf_asset__*` sidecars, so a loader that expects a sibling `tokenizer.json`
  file cannot read it;
* `vonkaiser/LTX-2.5-FP8-NVFP4`'s text encoder carries **no** safetensors
  `__metadata__` block, so the Gemma config has to be supplied out of band.
  `Ltx2LoadGemmaAssets(file, /*require_config=*/false)` is the opt-out; the
  default refuses, exactly as upstream does.

Reproduce the parity gate (CPU only, no checkpoint and no gated download; needs
torch, numpy and einops plus a Lightricks LTX-2 checkout):

```sh
python3 scripts/gen-ltx2-text-goldens.py \
    --ltx2 ~/_git/LTX-2 \
    --out tests/vllm/models/ltx2_text_goldens.inc
cmake --build build --target test_ltx2_text_encoder
./build/tests/test_ltx2_text_encoder
```

The generator imports the upstream modules by path and executes them at reduced
dimensions; both sides rebuild every weight from one deterministic stream, so no
weight byte is checked in. It also runs four degenerate inputs through upstream
and emits each one's full output tensor, not a "still finite" flag, because the
normalization epsilons and the width they are added in are invisible to a random
fixture. The mean's denominator is one of those: upstream adds it in float32
(`sequence_lengths * d` is an int64 tensor and `eps` a python float, which
promotes to the default dtype), so computing it in float64 is finer arithmetic
and the wrong answer.

A third thing to know if you are wiring a loader to it: the feature extractor
refuses, by name, any disagreement between what the checkpoint config declares
and what the weights actually carry. That covers the declared bias against
`bias.empty()`, the declared `out_features` against the weight's own width, and
`embedding_dim x (num_hidden_layers + 1)` against the weight's `in_features`. The
case worth naming is a loader that binds `video_aggregate_embed.weight` (U8,
NVFP4) and misses `.bias` (BF16, so a different unpack path) while the config
still says the projection is biased. Without the refusal that renders a plausible
video for the wrong prompt: every conditioning row is shifted by the missing bias
and every padded row projects to 0 instead of to the bias.
