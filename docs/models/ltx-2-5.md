# LTX 2.5

LTX 2.5 generates video, audio, or both through `ltx2-gen`, the C ABI, or the
OpenAI-compatible video endpoint.

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

Two-stage pipelines require `--upsampler` and `--lora`. The upstream name for
the adapter is `--distilled-lora`. Use `--max-phase 0` to stop after the first
phase.

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
ltx2-gen \
  --dit ltx-2.5-22b-distilled-fp8.safetensors \
  --dit-config ltx-2.5-transformer-config.json \
  --model-version 2.5 \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --upsampler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --encoder gemma4-12b-with-proj-nvfp4-torchao.safetensors \
  --encoder-config ltx-2.5-gemma4-text-config.json \
  --pipeline-kind res2s_two_stage \
  --prompt "a red fox running through deep snow at sunrise" \
  --frames 25 --width 320 --height 192 --seed 20260812 \
  --device cuda --workdir /tmp/ltx25 --out /tmp/ltx25/video.mp4
```

The high-quality preset uses these defaults:

- first phase: 20 steps, guidance 4.0, STG 1.0, and STG rescaling 0.7
- second phase: three steps, guidance 1.0, and STG disabled
- sampler: `res_2s`

## Generate audio without video

```sh
ltx2-gen --dit ltx-2.5-dit.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --encoder gemma4-12b-with-proj.safetensors \
  --encoder-config gemma4.json \
  --pipeline-kind t2a_one_stage --device cpu \
  --frames 121 --prompt "rain on a tin roof, distant thunder" \
  --workdir /tmp/t2a
```

The duration follows the video-frame count at the pipeline frame rate. This
pipeline returns a WAV file and no picture.

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

The DFR pipeline does not accept a frame-count input. It chooses the latent
layout from the reference path and generated keyframe slots.

The server does not forward a per-generation LoRA path. Load the adapter when
you load the engine. `--negative-prompt-embeds` and
`--negative-audio-prompt-embeds` apply to the embeds path only.

## Inspect a render

Each completed render writes `<workdir>/phase-log.json`. The C ABI function
`vllm_video_last_phase_log(engine)` returns the same JSON. A failed or active
render does not leave a partial file.

Use `sum_leaf_seconds` for the accounted total. `unaccounted_seconds` reports
time outside named phases. The file labels itself as diagnostic output, not a
benchmark.

Set `VLLM_RENDER_PHASE_LOG_STDERR=1` to print the phase table. Set
`VLLM_RENDER_PHASE_SAMPLER=0` to disable the 100 ms memory sampler. The normal
`[render]` lines print phase boundaries and DiT-forward progress.

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
