# LTX 2.5

LTX 2.5 generates video, audio, or both through `ltx2-gen`, the C ABI, or the
OpenAI-compatible video endpoint.

Pick a pipeline below, check that your DiT is one of the supported formats under
[checkpoint support](#checkpoint-support), then run
[generate video](#generate-video).

## Choose a pipeline

| Pipeline | Use |
|---|---|
| `res2s_two_stage` | High-quality two-stage generation with the `res_2s` sampler |
| `ti2vid_two_stage` | Plain two-stage text or image-to-video generation |
| `keyframe_interpolation` | Motion between supplied first and last frames |
| `one_stage` | One-stage video generation and video guidance |
| `t2a_one_stage` | Text-to-audio generation |
| `a2vid_two_stage` | Video generation around supplied audio |
| `dfr` | Dynamic frame-rate generation with generated keyframe slots |

The plain two-stage, keyframe-interpolation, and audio-to-video pipelines
require `--upsampler` and `--lora`. Upstream calls the adapter
`--distilled-lora`. The `res2s_two_stage` preset requires the upsampler, but it
does not require the LoRA. If you supply one, the current loader applies one
load-time strength to both phases. It cannot reproduce the upstream HQ strengths
of `0.25` in stage 1 and `0.5` in stage 2. Use `--max-phase 0` to stop after the
first phase.

Retake requests use `--retake-start-time`, `--retake-end-time`, and
`--retake-frame-rate`. Select the regenerated streams with
`--regenerate-video` and `--regenerate-audio`.

## Checkpoint support

The loader supports BF16, F32, FP8 E4M3, Lightricks NVFP4, and TorchAO NVFP4
DiTs. FP8 tensors use an F32 `<name>_scale`. TorchAO files carry the
`torchao_nvfp4` marker. F16 and ComfyUI `int8-convrot` DiTs are unsupported.

Use `Lightricks/LTX-2.5` at revision
`6c7e5e573ac1667efc83407806fe9b0b93730e60` for the first-party assets.

| Arm | File under `diffusion_models/` | Bytes | SHA-256 |
|---|---|---:|---|
| BF16 full | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` |
| BF16 distilled | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | pending authenticated fetch |
| NVFP4 distilled | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | pending authenticated fetch |
| INT8 full, unsupported | `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | pending authenticated fetch |
| INT8 distilled, unsupported | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | pending authenticated fetch |

The full BF16 DiT uses about 42 GB. Its real-weight materialization and render
remain pending. The two BF16 files have identical sizes, so select them by file
name and sampling regime.

Distilled two-stage recipes also use
`loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`. This file is
8,899,889,568 bytes and contains 1,660 rank-450 adapter pairs. It is distinct
from the 327,322,640-byte IC-LoRA spatial upscaler.

The Gemma-4 text encoder stores its tokenizer in the `tokenizer_json` tensor.
Pass `--encoder-config` when the checkpoint has no safetensors metadata. The
full 12B text tower uses about 24 GB in BF16 and about 33 GB during its host
gate.

Set `VLLM_CPP_LTX2_TEXT_ENCODER` when the text encoder is outside
`CHECKPOINT_ROOT`. Set `VLLM_CPP_LTX2_TOWER_E2E=1` to run the full tower gate.

## Generate video

```sh
LTX_ROOT="$CHECKPOINT_ROOT/ltx-2.5"
ltx2-gen \
  --dit "$LTX_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --model-version 2.5 \
  --video-vae "$LTX_ROOT/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae "$LTX_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --upsampler "$LTX_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --prompt-embeds "$LTX_VIDEO_EMBEDS" \
  --audio-prompt-embeds "$LTX_AUDIO_EMBEDS" \
  --prompt-valid-rows "$LTX_PROMPT_ROWS" \
  --pipeline-kind res2s_two_stage \
  --frames 25 --width 320 --height 192 --seed 20260812 \
  --device cuda --workdir /tmp/ltx25 --out /tmp/ltx25/video.mp4
```

The high-quality preset sets `num_inference_steps` to 15. Stage 1 derives its
schedule from that value; stage 2 uses its fixed refinement schedule. Both
stages use the `res_2s` sampler. Video CFG is `3.0`, audio CFG is `7.0`, and
modality guidance is `3.0`. STG is disabled and video guidance rescaling is
`0.45`.

Use dimensions divisible by `64` for any two-stage pipeline. Use dimensions
divisible by `32` for a one-stage pipeline. The engine refuses invalid width or
height values instead of rounding them. Frame counts use the temporal VAE grid:
pass `8k + 1` frames to get the requested count. Other positive counts floor to
that grid. The default geometry is `1024x1536` at 121 frames.

## Generate audio without video

```sh
LTX_ROOT="$CHECKPOINT_ROOT/ltx-2.5"
ltx2-gen \
  --dit "$LTX_ROOT/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --model-version 2.5 \
  --audio-vae "$LTX_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --prompt-embeds "$LTX_VIDEO_EMBEDS" \
  --audio-prompt-embeds "$LTX_AUDIO_EMBEDS" \
  --prompt-valid-rows "$LTX_PROMPT_ROWS" \
  --pipeline-kind t2a_one_stage --device cpu \
  --audio-cfg-guidance-scale 1.0 --frames 121 \
  --workdir /tmp/t2a
```

The duration follows the frame count at the pipeline frame rate. This pipeline
returns a WAV file and no picture. It runs only on the CPU and refuses
`--device cuda`. Do not pass `--video-vae`, `--width`, or `--height`.

The audio-only defaults are CFG `7.0`, STG `1.0`, rescale `0.7`, skip step `0`,
and STG block `28`. The negative prompt supplies the sixth guidance input.
Audio-only generation fixes modality guidance at `1.0` because it has no video
stream. The prompt-embeds path needs a text tower for negative conditioning
unless you set audio CFG to `1.0`.

## Dynamic frame-rate temporal rounds

`--temporal-upsample-rounds N` on a `dfr` render drives the temporal x2 latent
upsampler, and nothing else in this project does. Each round doubles the latent
along time, re-tiles the canvas into keyframe-seam windows, invents a
mid-segment keyframe slot per window, densifies each window with ancestral Euler
at eta `0.5` under its own noise seed, and stitches the windows back.

The caller gets `(num_frames - 1) * 2**rounds + 1` frames. A 9-frame request
returns 17 at one round and 33 at two. The playback frame rate scales with the
count, so the clip's duration in seconds does not change. The conditioning frame
rate is capped at 60 while playback is not.

Rounds need a second checkpoint. Pass `--temporal-upsampler`, which is a
different file from `--upsampler`: stage 2's input transform takes the spatial x2
upscaler and the rounds take the temporal one, and `dfr` holds both at once. The
loader reads `spatial_upsample` and `temporal_upsample` from the checkpoint's own
config and refuses a file supplied in the wrong slot, because the two share a
class name and a tensor layout and the wrong file otherwise loads, runs, and
returns a plausible latent of the wrong shape. Asking for rounds without the
checkpoint is refused rather than run at zero rounds, because a silently skipped
round returns a clip a fraction of the requested length.

The flag is per generation. The C ABI reaches it through
`vllm_video_gen_params.extra_keys` as `temporal_upsample_rounds`, and
`--temporal-upsampler` maps to the `temporal_upsampler_path` load extra.
`/v1/videos` does not carry it: that endpoint forwards no per-generation extra to
any engine yet (#928).

Two limits this arm does not hide. The tile count is `2**round` clamped to the
canvas's segment count, which is upstream's `min(num_tiles, n_segments)` and not
a shortfall, but it means a short canvas denoises in fewer windows than the round
asks for. No test in this tree reaches the unclamped arm either, so that
expression is ungated and #1493 owns closing it. And the whole arm is gated on
reduced-dimension fixtures only, because the `keyframe_slot_sft` base that DFR
needs is unpublished (#1137). The real
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` exists and loads; the
transformer it would run beside is what is missing, so no real-weight DFR result
exists or is implied.

## Conditioning and current limits

Image conditioning accepts a binary PPM first frame. Set `image_crf=0`
explicitly. Other CRF values, including the trained default of 18, require an
H.264 round trip that this project does not provide. `noise_aug=1.0` pins the
supplied frame exactly.

You can also supply a last-frame keyframe. The CLI has fixed first-frame and
last-frame slots. It does not support an arbitrary interior `FRAME_IDX`.

Set `num_generated_keyframes` to add evenly spaced interior latent slots. A
negative count is invalid. The clip must contain at least `N + 2` frames. The
engine does not return these slots as separate decoded images.

Reference-image, reference-video, and reference-audio conditioning remain
unsupported. These paths still need multi-frame pixel encoding, stage-specific
adapter application, or the audio-VAE encoder filter. The engine also refuses
sample-rate conversion and a VAE configured with `latent_log_var: none`.

The DFR pipeline refuses a frame-count input. It derives the frame count from
the reference path. It also refuses `num_generated_keyframes`; DFR creates its
own keyframe grid. Other supported pipelines accept generated keyframe slots,
but reject a negative count or a target shorter than `N + 2` frames.

The server does not forward a per-generation LoRA path. Load the adapter when
you load the engine. `--negative-prompt-embeds` and
`--negative-audio-prompt-embeds` apply to the embeds path only.

The `one_stage` pipeline exposes video and audio CFG, STG, rescale, skip-step,
and STG-block controls. It also exposes `--a2v-guidance-scale` and
`--v2a-guidance-scale`. A CFG scale other than `1.0` requires negative
conditioning from the text tower or both negative embeds files. Both embeds
files must have the same row count as the positive pair. The engine refuses an
STG block outside the DiT layer count. Distilled two-stage and retake pipelines
refuse these one-stage guidance controls.

Prompt embeds are little-endian F32 rows. Video rows have width 4,096 and audio
rows have width 2,048. Supply both files with the same row count. On checkpoints
with the embeddings connector, that count must be a multiple of 128. Set
`--prompt-valid-rows` to the number of real caption rows. A command with
`--prompt` but no `--encoder`, or with only one embeds file, is invalid.

Pass `--dit-config` only when the selected DiT has no safetensors metadata. The
loader refuses a config beside a checkpoint that already declares one.

Declare the checkpoint family with `--checkpoint-class`: use `full` for the
development transformer, `distilled` for the distilled transformer, and
`keyframe_slot_sft` for that specialized arm. Every pipeline except `dmd2`
requires a declaration. The loader checks it against the checkpoint rather than
using it as an unchecked hint, and refuses a missing or mismatched class before
generation.

## Where video VAE decode runs

The video VAE convolution dispatches through `vt::Conv3d` on the queue selected
when the engine loads. `--device cuda` therefore selects the CUDA convolution;
the default selects the CPU implementation, which remains byte-identical to the
previous host loop.

No GPU has run the CUDA arm yet, so this path has no accelerator correctness or
speed claim. Norms, activations, upsampling, and attention still run on the
host, which also means a device queue currently crosses the host-device boundary
for each convolution ([#1451](https://github.com/mudler/vllm.cpp/issues/1451),
[#1452](https://github.com/mudler/vllm.cpp/issues/1452)).

The first non-CPU convolution prints this notice once per process:

```text
[vt] first non-CPU vt::Conv3d dispatch (device type 4). This arm has never been
run on real hardware; see issue #1452.
```

The notice reports that the unverified arm was reached. It does not report a
fallback or a degraded result.

## Inspect a render

Each completed render writes `<workdir>/phase-log.json`. The C ABI function
`vllm_video_last_phase_log(engine)` returns the same JSON. A failed or active
render does not leave a partial file.

Use `sum_leaf_seconds` for the accounted total. `unaccounted_seconds` reports
time outside named phases. The file labels itself as diagnostic output, not a
benchmark.

`gaps` says WHERE that un-named time is. It holds one interval before each named
leaf and one after the last, each carrying the two names it lies between, and
they add to `unaccounted_seconds` exactly. Sort it and read the top entry: the
largest gap is the next region worth naming. `<origin>` and `<end>` are the ends
of the timeline.

`instrument_seconds` says how much of the residue the instrument itself spent —
the mutex wait before a phase starts, and the flushed progress line after it
ends. Subtract it before calling what is left a phase nobody named. Every record
carries its own `instrument_seconds` too, which is what that phase paid for the
boundaries of its own sub-scopes.

Set `VLLM_RENDER_PHASE_LOG_STDERR=1` to print the phase table. Set
`VLLM_RENDER_PHASE_SAMPLER=0` to disable the 100 ms memory sampler. The normal
`[render]` lines print phase boundaries and DiT-forward progress.

Set `VLLM_RENDER_PROGRESS=0` to hide the normal progress lines. Set
`VLLM_CPP_CPU_THREADS` to control the shared CPU worker pool used by host-side
render stages.

See [benchmarks](../BENCHMARKS.md) for accepted measurements.

## Run the parity gates

The reduced DiT gate needs a clean LTX-2 checkout:

```sh
git clone https://github.com/Lightricks/LTX-2 ~/_git/LTX-2
python3 scripts/gen-ltx2-goldens.py \
  --ltx2 ~/_git/LTX-2 --out tests/vllm/models/ltx2_goldens.inc
cmake --build build --target test_ltx2
./build/tests/test_ltx2
```

The pipeline gate also needs a clean vLLM-Omni checkout:

```sh
git clone https://github.com/vllm-project/vllm-omni ~/_git/vllm-omni
python3 scripts/gen-ltx2-pipeline-goldens.py \
  --ltx2 ~/_git/LTX-2 --vllm-omni ~/_git/vllm-omni \
  --out tests/vllm/models/ltx2_pipeline_goldens.inc
cmake --build build --target test_ltx2_pipeline
./build/tests/test_ltx2_pipeline
```

The reduced Gemma-4 tower gate requires Transformers 5.8 or later:

```sh
/path/to/venv/bin/python scripts/gen-ltx2-gemma-tower-goldens.py \
  --out tests/vllm/models/ltx2_gemma_tower_goldens.inc
cmake --build build --target test_ltx2_text_encoder
./build/tests/test_ltx2_text_encoder
```

See the [LTX 2.5 specification](../../.agents/specs/ltx-2-5.md), the
[BF16 DiT specification](../../.agents/specs/ltx25-bf16-dit.md), and the
[NVFP4 nibble-order specification](../../.agents/specs/nvfp4-nibble-order.md)
for implementation chronology, oracle evidence, and open parity work.
