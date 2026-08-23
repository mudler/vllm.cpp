# MiniMax-H3

MiniMax-H3 generates video with joint audio. A render needs five files: a DiT
and an encoder as community GGUF quantizations, two VAEs, and a tokenizer.

**Read two sections before your first render, because a render takes hours before
it tells you anything.** The checkpoint you load fixes which tasks it can serve,
and a mismatch renders a wrong picture instead of failing: see
[which tasks a checkpoint serves](#which-tasks-a-checkpoint-serves). The prompt
has to ask for speech explicitly, or you get ambience: see
[writing the prompt](#writing-the-prompt).

## Video and audio workflow

### Get the exact weights

Five files. The DiT and encoder are community GGUF quantisations; the two VAEs and
the tokenizer come from the official checkpoint.

| file | size | source |
|---|---|---|
| `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19.9 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `video_vae/source/model.safetensors` | 10,415,548,320 bytes | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/video_vae/` |
| `audio_vae/model.safetensors` | 605,429,308 bytes | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/audio_vae/` |
| `tokenizer/tokenizer.json` | 7,032,403 bytes | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/` |

Take each VAE's `config.json` from the same directory as its weights: they carry the
per-channel `latents_mean` / `latents_std` and the temporal `clip_length` /
`token_drop`, and the decode is wrong without them.

The `realrebelai/MiniMax-H3_GGUFs` files above are pinned at revision
`daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7`. Their exact sizes and SHA-256
values are in the [checkpoint registry](../USAGE.md#checkpoint-registry). The
official `MiniMaxAI/MiniMax-H3` files are pinned at revision
`42ed227ee7df40d41602854ae760620d6eb651fe`.

**Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude
outliers that 3-bit cannot hold. In a controlled A/B (same prompt, seed, code and
VAEs, only the DiT quantisation changed) Q3_K_M gave a murky silhouette under a
visible lattice and Q4_K_M gave a photoreal close-up. The full bf16 release is
66.3 GB across 13 shards if you want to go further.

Higher-precision arms that exist but are not the default: NVFP4
([lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4))
and the original bf16 weights under `FL2VA/transformer/`.
The NVFP4 repository is pinned at revision
`8c5abfed61e1b6a170240792b65253fba1a65b7b`.

## The PRUNED checkpoints — more precision for the same footprint

The community `pruned` variants **are supported and are drop-in**: pass one to
`--dit` exactly as you would an unpruned file. Nothing else about the command
changes.

They are not lossily pruned. AdaLN modulation dominates the unpruned parameter
count — `adaln_proj` alone is 13.04B of 33.12B (39.4%) — because the model
projects a 5376-wide conditioning vector into modulation parameters in every one
of the 50 blocks. But modulation depends only on the timestep, so that projection
is almost entirely redundant, and the pruned form replaces it with a `[1025, 8]`
timestep table feeding an 8-wide `adaln_proj.linear`. 13.04B parameters become
0.04B and the DiT drops from 33.12B to 20.11B, with the modulation path kept at
full precision.

The practical consequence: **a pruned Q8_0 costs about what our unpruned Q4_K_M
costs.**

| file | size | source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q8_0.gguf` | 21.4 GB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| `minimax_h3_ref2va_pruned-Q8_0.gguf` | 21.4 GB | same repo — the `ref2va` partition |
| `minimax_h3_{fl2va,ref2va}_pruned-{Q2_K,Q3_K,Q4_K,Q5_0,Q6_K}.gguf` | 6.7-16.6 GB | same repo |
| `minimax_h3_{fl2va,ref2va}_pruned_nvfp4.safetensors` | 12.5 GB | [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) |

The `unsloth/MiniMax-H3-GGUF` files are pinned at revision
`d629413c2e5b51b38c453668b75ca3b06ca92703`. The checkpoint registry records
the exact sizes and SHA-256 values for the two rendered Q8_0 artifacts.

The partition rule below still applies: a `fl2va` file serves `t2va` and `fl2va`,
a `ref2va` file serves `ref2va`.

**What is actually verified, and what merely exists.** The distinction matters
because a render takes hours before it tells you anything:

| arm | status |
|---|---|
| **Q4_K_M** | **VERIFIED end to end** — every render in this doc, on BOTH partitions (t2va + fl2va on FL2VA, ref2va on REF2VA). Use this. |
| Q3_K_M | verified BAD (the A/B above): murky silhouette under a lattice |
| bf16 (66.3 GB, 13 shards) | loader + device streamer implemented and gated, but **CPU-only** verification — no end-to-end GPU render has been done |
| NVFP4 | exists; loads (unpruned and pruned) |
| **pruned Q8_0** | **loads and renders** — the A/B is in `.agents/specs/minimax-h3.md` section 8.21 |
| pruned Q6_K / Q5_0 / Q4_K / Q3_K / Q2_K ([unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)) | load through the same path; only Q8_0 has been rendered |

## Which tasks a checkpoint serves

**`MiniMax-H3-FL2VA-Q4_K_M.gguf` is the FL2VA partition. It serves `t2va` and
`fl2va` — NOT `ref2va`.** H3 ships two independently-served DiT partitions and
the task must match the one you loaded; upstream's `_resolve_task` raises on the
mismatch.

Pass a reference image against this file and you get a task/partition mismatch.
It does not fail loudly — it renders, and the render is *wrong*: a coloured
diagonal lattice over the whole frame, worse the larger the canvas. Measured on
one prompt and canvas (1344x768 / 124f), as a period-16 seam ratio where 1.15 is
clean:

| configuration | seam ratio |
|---|---|
| ref2va against FL2VA (the mismatch) | 2.28 |
| t2va against FL2VA (correct) | **1.19** |

The small-canvas case is what makes this expensive to spot: at 864x480 the same
mismatch measures 1.15 and looks acceptable, so the bug only becomes obvious at
the resolution you actually want.

Pass `--partition fl2va` explicitly. The driver mirrors upstream's raise, so a
mismatch is rejected at the CLI rather than silently rendered.

For a reference-image render you need the **Ref2VA** partition instead, and the
one to use is **`MiniMax-H3-REF2VA-Q4_K_M.gguf`** (19.9 GB,
[realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs)) —
the same quantisation as the FL2VA file above, and verified coherent:

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-REF2VA-Q4_K_M.gguf --dequant-bf16 --partition ref2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "..." --ref-image subject.ppm \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 512 --width 512 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

**Do NOT use the NVFP4 Ref2VA weights.** `minimax_h3_ref2va_nvfp4_full` renders the
multicolour patch grid, and it took three investigations to establish that this is the
QUANTISATION and not the ref2va path: the identical reference-row assembly, packed-block
layout and denoise loop render coherently on Q4_K_M (period-16 seam **1.13**, VAE-input
latent adjacent-cell cosine **0.8526**). Ref2VA on Q4_K_M is a working mode; Ref2VA on
NVFP4 is not.

## Writing the prompt

Two things decide whether you get what you asked for, and neither is obvious.

**To get SPEECH, ask for it and supply the line.** The model generates video and
audio jointly, so a prompt describing a silent performance produces room tone and
ambience, which is correct but not what most people expect. Say that the character
talks, describe the voice, and put the words in the prompt:

```
It is TALKING to the camera: its mouth moves clearly in sync with its speech,
in a dry, deadpan tone.

It says, clearly and audibly: "Michael scheduled another all-hands.
It is about the printer. Again."

Audio: a single clear voice, close-miked, with quiet room tone underneath.
```

That prompt produced audio an ASR pass transcribed back word for word. A prompt
that only described expressions and sighs produced ambience at about 13 dB lower
level and no speech at all.

**Refer to references BY TAG in the prompt text.** A reference is bound by naming
it, not merely by being passed on the command line. Use `<Picture i>`, `<Video k>`
and `<Audio j>`, numbered from 1 per type, matching the order you pass them:

```
<Picture 1> is a cyan llama mascot wearing white sunglasses.

A talking-head interview. The subject is the llama from <Picture 1>, sitting in a
grey office chair ...
```

Other prompt notes: frame count runs on the 17n+5 grid at 24 fps, and the trained
range is roughly 124 to 362 frames (about 5 to 15 seconds). Text rendered *inside*
the video (signage, wordmarks) is the model's weakest area and will often come out
malformed; composite real logos in afterwards.

`/v1/videos` generates video with sound through the MiniMax-H3 diffusion model.
It speaks **OpenAI's Sora video shape**, so an OpenAI client works against it
unmodified, and it keeps the richer native knobs alongside.

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B \
  --video-dit /path/to/h3-dit.gguf --video-vae /path/to/video-vae.safetensors \
  --audio-vae /path/to/audio-vae.safetensors \
  --video-vae-config video_vae/config.json --audio-vae-config audio_vae/config.json \
  --video-encoder /path/to/h3-encoder.gguf
```

```python
video = client.videos.create(model="sora-2-pro", prompt="a cat on a skateboard",
                             size="1280x720", seconds="8")
while client.videos.retrieve(video.id).status not in ("succeeded", "failed"):
    time.sleep(5)
open("out.mp4", "wb").write(client.videos.download_content(video.id).read())
```

## Request fields

| Field | Spelling | Meaning |
|---|---|---|
| `prompt` | both | Required. The text conditioning |
| `model` | OpenAI | Recorded and echoed back. A name this server does not serve is a `warning` on the job, never a rejection: the video model is chosen at startup |
| `size` | OpenAI | `"<width>x<height>"`, e.g. `"1280x720"`. Whole pixels, both positive |
| `seconds` | OpenAI | Duration, as a number or a numeric string (`8` and `"8"` both work) |
| `input_reference` | OpenAI | The image the video starts from. A filesystem path or a `data:` URL |
| `metadata` | OpenAI | Free-form string map, passed through untouched. Two keys are acted on: `input_reference_video` and `input_reference_audio` (see below) |
| `width`, `height` | native | Output geometry in pixels |
| `duration` | native | Duration in seconds |
| `task` | native | `t2va`, `fl2va`, `ref2va`; resolved from the inputs when omitted |
| `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed` | native | The H3 generation knobs. Accepted at the top level or nested under `extra_params` |

**Precedence.** When a body carries both spellings of one value, the **native
field wins**: `width`/`height` beat `size`, `duration` beats `seconds`. That
direction keeps every request that parses today meaning exactly what it meant
before. Both spellings are validated either way, so a malformed `size` is a 400
even when explicit `width`/`height` would have overridden it.

**`input_reference` maps to fl2va first-frame conditioning.** OpenAI documents
it as the image the generated video starts from, which is what fl2va expresses:
the supplied image is pinned as frame 0 of the output. H3's other image mode,
ref2va, prepends whole reference images as their own blocks (subject or style
guidance that never becomes a frame), so it stays reachable only through the
native `task` field and the `minimax-h3-gen` CLI. Two limits: the image must be
a **binary PPM (P6)** (no PNG or JPEG codec is vendored, the same residual the
chat multimodal path carries), and it must already be at the output resolution
(no image resampler is vendored). A mismatch is refused with the resolved
geometry in the message.

## Video and audio references (`metadata`)

H3 supports three reference modalities and OpenAI's schema has a slot for one,
so the other two enter through `metadata`, the standard OpenAI free-form string
map. Strict clients tolerate it, and no invented top-level field breaks their
schema validation. Unknown metadata keys are passed through untouched.

```jsonc
{
  "prompt": "the same scene, at dusk",
  "metadata": {
    "input_reference_video": "/tmp/vllm_h3_videos/job0",  // DIR of frame_%06d.ppm
    "input_reference_audio": "/tmp/voice.wav"             // 16-bit PCM WAV, or a data: URL
  }
}
```

`input_reference_video` is a **directory of `frame_%06d.ppm`**, which is exactly
what this server and `minimax-h3-gen` write, so one run's frames chain straight
into the next request. It is not a container file: no demuxer is vendored.

**A video reference is SILENT.** `MiniMaxH3EncodeReferenceVideo` emits a
`kVideoAudio` block with `ref_audio_t == 0`, so the clip contributes no sound of
its own. Supplying `input_reference_audio` alongside it attaches the audio to
that same block (one block carrying both, the layout upstream builds); without
it the reference is picture only. That is a real limitation, not an omission.

**Legal combinations.** fl2va keyframes and ref2va reference blocks are
exclusive in the pipeline itself
([`minimax_h3_pipeline.cpp`](../../src/vllm/model_executor/models/minimax_h3_pipeline.cpp)),
so the request parser enforces the same rule and returns a 400 naming the
offending pair rather than dropping a reference you supplied.

| `input_reference` | `metadata.input_reference_video` | `metadata.input_reference_audio` | |
|---|---|---|---|
| (none) | (none) | (none) | t2va, prompt only |
| image | (none) | (none) | fl2va, the image is frame 0 |
| (none) | clip | (none) | ref2va, silent video reference |
| (none) | (none) | WAV | ref2va, audio reference |
| (none) | clip | WAV | ref2va, one block carrying both |
| image | clip and/or WAV | | **400**: keyframe and reference conditioning are exclusive |

The video reference needs `--video-vae` (the encoder half of the same file) and
the audio reference needs `--audio-vae`; both load lazily, once, on the first
request that asks for them.

## The job lifecycle

`POST /v1/videos` returns immediately with `{"id": "vid_1", "status": "queued"}`;
generation is minutes long, so the synchronous twin `POST /v1/videos/sync` exists
for scripts that would rather block. `GET /v1/videos/{id}` reports `queued`,
`running`, `succeeded` (with `output_path`) or `failed` (with `error`).

`GET /v1/videos/{id}/content` returns the finished MP4 with
`Content-Type: video/mp4`. An unknown id is a 404; a job that has not finished is
a **409** naming its current status rather than a truncated file; a failed job is
a 500 carrying its failure; an output that has since vanished from disk is a 500
rather than a 200 with zero bytes.

The library never spawns a process, so generation and muxing enter through a
caller-supplied `VideoRunner` callback (`examples/server/main.cpp` supplies one
that invokes `ffmpeg`, path configurable with `--video-ffmpeg`).

## Video family, and family-specific load knobs

`/v1/videos` serves whichever video family the `--video-dit` checkpoint belongs
to. By default the family is **detected** from what the checkpoint holds, and
that is unchanged.

`--video-family NAME` pins it instead. Two registered families exist,
`minimax-h3` and `ltx-2.5`, and a name outside that set is refused at argument
parsing, before the text model loads, with the registered names printed. It is
never a hint: a declared family that cannot load the checkpoint fails loudly
rather than falling back to detection, because a checkpoint handed to the wrong
family does not fail, it renders noise.

`--video-extra KEY=VALUE`, repeatable, carries a family's own load knobs. LTX-2.5
cannot load without `dit_config_path`, and it needs `encoder_config_path` beside
`--video-encoder` when the text encoder declares no `gemma_config` (the shipped
one does not); MiniMax-H3
defines `partition`, for which `--video-partition` remains the documented alias.
A bare `KEY` with no `=` is refused rather than read as an empty value, and a
`--video-extra partition=X` contradicting `--video-partition Y` is refused rather
than resolved by whichever assignment ran last. A family refuses any key it does
not define, so a mistyped knob is an error instead of a silently defaulted
render.

```sh
vllm-server --model /path/to/text-model \
  --video-family ltx-2.5 \
  --video-dit ltx-2.5-22b-distilled-fp8.safetensors \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --video-encoder gemma4-12b-with-proj-nvfp4-torchao.safetensors \
  --video-extra encoder_config_path=ltx-2.5-gemma4-text-config.json \
  --video-extra dit_config_path=ltx-2.5-transformer-config.json \
  --video-extra model_version=2.5
```

`allow_unported_modules=1` is no longer needed for either shipped LTX-2.5 DiT —
`keyframes_abs_pos_embedding`, the last family that demanded it, was ported on
2026-08-14 (issue #658). The flag still exists for a checkpoint that carries
something else this port does not.

## Inspect a render

Set `VT_H3_DUMP_DIR=<dir>` to write the video and audio latents for a numerical
comparison.

The pre-fold driver flags `--denoise-only`, `--dump-params`, and `--save-embeds`
are retired. The thin ABI client no longer accepts them.


## MiniMax-H3 browser console (`vllm-video-studio`)

A standalone browser console for MiniMax-H3, deliberately **separate** from the
OpenAI-compatible API server: `examples/server` is the API surface and a UI does
not belong in it. The studio owns its own endpoints and drives the public C ABI
(`vllm_video_*`) like any other FFI consumer, so it is also a worked example of
that ABI.

Built with the server (`-DVLLM_CPP_SERVER=ON`), because it shares the same
vendored HTTP transport.

```sh
vllm-video-studio --models-dir /path/to/h3 --port 8080
```

Then open `http://localhost:8080`. It discovers the five H3 files under
`--models-dir`, or each can be pointed at explicitly with `--dit`, `--encoder`,
`--video-vae`, `--video-vae-config`, `--audio-vae`, `--audio-vae-config` and
`--tokenizer`. Other flags: `--host`, `--device`, `--workdir`, `--ffmpeg`,
`--partition`, `--keep-quant`, `--prompt-embeds`, and `--ui` to serve a custom
web root.

The weights, and why each one is needed, are in the MiniMax-H3 section below.


### Generate video and audio


Renders an MP4 with a stereo track. Weights: a GGUF DiT (use **Q4_K_M**), the Qwen3-VL-32B
encoder, and both VAEs.

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 --partition fl2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "A golden retriever runs across a sunlit beach, waves crashing behind it" \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 768 --width 1344 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

`--partition` is REQUIRED and names the partition the checkpoint you passed
actually serves — see the trap above. This is the command every render in this
document was produced with: Q4_K_M DiT and encoder, `--dequant-bf16`, task
**t2va** (no reference image), the 1344x768 default canvas, 124 frames, 50 steps.

Cost, so you can plan: **~176 s per step** at 1344x768 / 124f on a 20-SM sm_110
device, so a 50-step render is about **2.5 hours** plus roughly 30 minutes of
weight loading. Dropping to 512x512 costs ~15 s/step (~13 minutes end to end),
which is the right canvas for iterating on a prompt before committing to a full
render. `--dequant-bf16` holds the DiT as bf16 (~66 GB resident); `--keep-quant`
is the low-memory arm.

Conditioning modes, all optional and mutually exclusive where noted:

```sh
--first-frame start.ppm --last-frame end.ppm   # pin the first and/or last frame (fl2va)
--ref-image subject.ppm                        # reference image, repeatable (ref2va)
                                               # NOT served by the FL2VA checkpoint above --
                                               # needs a Ref2VA partition (see the trap)
--ref-video prev_workdir/                      # reference clip, reads frame_%06d.ppm
--ref-audio voice.wav                          # reference audio
--noise-aug 0.9                                # how hard a keyframe is pinned (1.0 = exact)
```

Reference frames are binary PPM, which is what this tool also **writes**, so one run's `--workdir`
feeds straight back in as `--ref-video` and clips chain. Convert anything else with
`ffmpeg -i in.png -pix_fmt rgb24 out.ppm`.

Worked reference renders, all on the **Ref2VA** checkpoint (`--partition ref2va`); the flags
below replace `--ref-image` in the command above:

```sh
