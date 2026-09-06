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

Every two-stage pipeline needs `--upsampler` for its second stage. The ones
whose checkpoint is not the distilled one need `--lora` as well:
`ti2vid_two_stage`, `keyframe_interpolation`, `a2vid_two_stage`,
`res2s_two_stage`, and `dfr`. Upstream calls the adapter `--distilled-lora` and
makes it `required=True`. A load that omits it is refused by name rather than
rendered, because these pipelines run a distilled refinement schedule that
undistilled weights were never trained for, and the clip that comes back is the
right size, the right length, and wrong.

`distilled_two_stage`, `one_stage`, `t2a_one_stage`, and `retake` do not need
the adapter. The first runs on weights that are already distilled; the last two
are one-stage; `retake` takes it as a CONDITION on a full checkpoint rather than
a flat requirement.

`res2s_two_stage` and `dfr` apply the adapter to BOTH stages; the other three
apply it to stage 2 only. The loader applies one load-time strength to every
phase, so it cannot reproduce the upstream HQ strengths of `0.25` in stage 1 and
`0.5` in stage 2. Use `--max-phase 0` to stop after the first phase.

Retake requests use `--retake-start-time`, `--retake-end-time`, and
`--retake-frame-rate`. Select the regenerated streams with
`--regenerate-video` and `--regenerate-audio`.

## Checkpoint support

The loader supports BF16, F32, FP8 E4M3, Lightricks NVFP4, and TorchAO NVFP4
DiTs. FP8 tensors use an F32 `<name>_scale`. TorchAO files carry the
`torchao_nvfp4` marker. F16 and ComfyUI `int8-convrot` DiTs are unsupported.

The first-party assets come from `Lightricks/LTX-2.5`. The byte counts in the
next table are what that repository published at revision
`6c7e5e573ac1667efc83407806fe9b0b93730e60`, read from its tree listing on
17 August 2026. **The SHA-256 column is deliberately not here.** The
[checkpoint registry in `docs/USAGE.md`](../USAGE.md) is its one home — that is
the surface AGENTS.md names for "which weights, and from where" — and this table
exists to say which ARMS the loader supports. The registry carries the BF16 full
DiT's digest and the caveat that goes with it: it was derived from this project's
own copies and **has never been compared against the artifact Lightricks
published**. Four of the arms below have no digest anywhere, in either document.

| Arm | File under `diffusion_models/` | Bytes | SHA-256 |
|---|---|---:|---|
| BF16 full | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | [`docs/USAGE.md` registry](../USAGE.md) |
| BF16 distilled | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | pending authenticated fetch |
| NVFP4 distilled | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | pending authenticated fetch |
| INT8 full, unsupported | `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | pending authenticated fetch |
| INT8 distilled, unsupported | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | pending authenticated fetch |

Warning: revision `6c7e5e573ac1667efc83407806fe9b0b93730e60` does not give you
the NVFP4 DiT this project measured. That revision publishes 18,721,548,408
bytes and 7877 tensors under
`ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`. Every NVFP4 measurement
here used a file of 18,721,432,024 bytes and 7876 tensors whose SHA-256 is
`f9c4c2ae9a6aa8f732eb02a1c4c3b34888caad3dd35bb65deaf3b5043cda78fa`, hashed from
the bytes on the shared checkout. Its `huggingface_hub` sidecar records
`8a4ff96f581e72bedc1b44367581c49d544a05f1` as the revision it was fetched from,
which is the best available statement of origin and not a re-derived one,
because nothing has fetched from that revision since. Check any NVFP4 checkpoint
by its SHA-256 rather than by revision alone. The checkpoint registry in
[Usage](../USAGE.md) carries both value sets, their provenance, and the limit on
the revision half ([#1723](https://github.com/mudler/vllm.cpp/issues/1723)).

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
  --lora "$LTX_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --frames 25 --width 320 --height 192 --seed 20260812 \
  --device cuda --workdir /tmp/ltx25 --out /tmp/ltx25/video.mp4
```

`--lora` is repeatable, and each repetition takes an adapter path with an
optional strength, exactly as upstream's own flag does: `--lora
first.safetensors 0.8 --lora second.safetensors`. An omitted strength is
upstream's `DEFAULT_LORA_STRENGTH`. The adapters fuse into the weights in the
order given, and that order is observable in the render, because the aggregator
rounds the first product differently from every later one.

Through the C ABI and the server the same adapters are load extras rather than
flags, because the fusion happens once when the engine loads and cannot vary per
request. The first adapter is `lora_path` and `lora_strength` with no index; the
second onward are `lora_path_2` and `lora_strength_2`, `lora_path_3`, and so on.
A gap in the numbering is refused by name rather than closed up, so
`lora_path_3` without `lora_path_2` is an error and not a two-adapter render. So
is `lora_path_1`: one adapter has one spelling.

A second adapter is refused on `a2vid_two_stage`, `ti2vid_two_stage` and
`keyframe_interpolation`. Upstream runs the first stage of those three on the
user adapters alone and the second on the user adapters plus the distilled one,
which is a per-stage subset this engine cannot hold: it keeps one resident DiT
that every phase runs fused or bare, and no load extra says which of the
adapters is the distilled one. Supply one adapter to those pipelines, or use
`dfr`, which composes the user adapters and the distilled one onto the single
stage both its phases share.

Pass `--steps N` to set the denoise step count (#2130). Omit it, or pass a value
of 0 or less, and the resolved recipe decides: `one_stage` on model version 2.5
runs 30. The flag reaches `vllm_video_params.steps`, which the C ABI has always
carried and the engine has always honoured; until #2130 nothing shipped could set
it, so every render this project took ran its recipe default and no render could
be matched to a reference captured at another step count. A recipe whose schedule
is distilled into the weights refuses an override by name rather than applying
it.

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

## Auto duration

The duration head predicts how long the shot implied by the caption should be,
so a request need not state a frame count. Point the load at the head and pass
the window per generation:

```
--duration-head /path/to/duration_head.safetensors
```

reaches the `duration_head_path` load extra; `auto_duration` is a
per-generation extra spelled `"MIN,MAX"` in seconds, which mirrors upstream's
`--auto-duration MIN_SECONDS MAX_SECONDS`. `MIN` greater than `MAX` is refused,
as it is upstream, and combining it with an explicit `--frames` or `duration` is
refused rather than letting one silently win.

**Omitting every frame source auto-predicts, but only when a head is loaded.**
That is upstream's own default -- its `num_frames` defaults to `AutoDuration` --
reached without changing any existing caller: an engine loaded without
`duration_head_path` keeps the recipe default exactly as before.

The predicted seconds are clamped to the window and converted to a frame count
snapped down to the VAE's causal `8k + 1` temporal grid. Two consequences are
upstream's and are worth stating because they surprise people. The clamp is
applied BEFORE the snap, so a prediction outside the window lands on the grid
point below the bound rather than on the bound. And when the window itself
cannot sit on the grid -- `MIN` and `MAX` equal, at a value that is not
`8k + 1` -- the window wins and the returned count is off-grid.

**A checkpoint with no duration head is not an error.** Every LTX-2 checkpoint
predating 2.5 / gemma4 lacks one, and `duration_head_path` pointed at such a
file loads normally and simply leaves no predictor; a head missing some of its
tensors behaves the same way. Only a tensor present at the wrong shape is
refused. Asking for an auto duration when no predictor is available fails
immediately, at the top of the call, before any prompt encoding or diffusion
work is paid for.

**The WINDOW is ABI-only.** `--duration-head` is a command-line flag and
omitting `--frames` is how the command line asks for an auto duration, so the
capability itself is reachable from the CLI on its defaults. Overriding the
window is not: `auto_duration` has no flag of its own and is set only through
the C ABI's per-generation extras (`vllm_video_params`). A command-line render
against a loaded head therefore uses upstream's own defaults, 1 s and 20 s.

`/v1/videos` does not carry `auto_duration` either: that endpoint forwards no
per-generation extra to any engine yet (#928).

The head runs in f32 here where upstream builds it in bfloat16. That arm is
owed under `## Owed` in `.agents/specs/ltx25-duration-head-wire.md`, as the
eighth component of gap A24.

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
requires a declaration, and the loader refuses a missing, unknown or
pipeline-mismatched class before generation.

It is a declaration, not a check against the file. The loader compares what you
declare with the class the selected pipeline needs; it cannot compare it with the
checkpoint, because nothing in a checkpoint header separates the classes. The
full and the distilled bf16 transformers are the same size and their safetensors
headers agree on every tensor name, dtype, shape, offset and metadata value.
Declaring a class the file does not hold is therefore undetectable here, and it
renders a clip of the requested size in a sampling regime the weights were never
trained for. The refusal quotes that measurement when it fires.

## Where video VAE decode runs

The video VAE convolution dispatches through `vt::Conv3d` on the queue selected
when the engine loads. `--device cuda` therefore selects the CUDA convolution;
the default selects the CPU implementation, which remains byte-identical to the
previous host loop.

A GPU has now run the CUDA arm. It was compiled for `sm_121a` and executed on a
GB10, where `tests/vt/test_ops_conv3d.cpp` measured it `memcmp`-identical to the
CPU provider over the whole shape table and under catastrophic cancellation
([#1452](https://github.com/mudler/vllm.cpp/issues/1452)). **Read that as a
measurement taken once on leased hardware, not as continuous gating**: no CI
lane here has a GPU, so those cases skip on every automated run. **There is
still no SPEED claim**, and there is no end-to-end pixel comparison from a real
render yet; both stay owed under `## Owed` in
[`ltx25-device-residency.md`](../../.agents/specs/ltx25-device-residency.md).
The arm also serves f32 storage only and refuses f16 and bf16 by name, while the
op contract and the CPU provider admit all three.

Norms, activations, upsampling, and attention still run on the host, which also
means a device queue currently crosses the host-device boundary for each
convolution ([#1451](https://github.com/mudler/vllm.cpp/issues/1451)).

The first non-CPU convolution prints a notice once per process. On a device type
that has never been run it still reads:

```text
[vt] first non-CPU vt::Conv3d dispatch (device type 4). This arm has never been
run on real hardware; see issue #1452.
```

and on CUDA it now reports the gated state instead. The notice reports that the
arm was reached. It does not report a fallback or a degraded result.

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
