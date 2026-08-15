# LTX-2.5 — first end-to-end render on real weights

`ltx25-first-e2e-128px.mp4` (h264 + AAC, 512x512 upscaled from 128x128, 0.29 s)
and `ltx25-first-e2e-128px-frames.png` (all nine frames, 4x nearest-neighbour so
the actual pixels are visible rather than an interpolation).

## Provenance

| | |
|---|---|
| Tree | `98f8e046d` (`main`, the merge of #880) |
| Host | `dgx.casa`, NVIDIA GB10, 119 GiB unified memory |
| How | container `vllmcpp-build:gb10`, CUDA 13.0.1, `--gpus all`, under `flock $HOME/gpu.lock` |
| DiT | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, 18 721 432 024 bytes, header length 1179408 |
| Encoder | `gemma4-12b-with-proj-nvfp4-torchao.safetensors` |
| Flags | `--frames 9 --width 128 --height 128 --seed 20260814 --device cuda`, **no `--allow-unported`** |
| Prompt | `a red fox trotting through tall grass at sunrise, cinematic` (8 words) |

The `.mp4` was muxed after the fact on another host: `ffmpeg` is absent from the
CUDA image, which is the `EXIT=127` in the run log. The render itself completed
and wrote every frame plus `audio.wav`.

## What this shows

The chain executes on real quantised weights and produces structured, moving,
non-noise output with synchronised audio: Gemma-4-12B NVFP4 text tower -> 22B
NVFP4 DiT -> video VAE + audio VAE -> frames + 48 kHz stereo. It is the first
render for which `--allow-unported` was **not** required, which is the
user-visible result of row `LTX25-KEYFRAMES-ABS-POS` (#658, #880).

Measured rather than assumed: nine distinct frame md5s, mean adjacent-pixel
difference 6.1-7.3 against ~78-81 for i.i.d. noise at the observed standard
deviation, audio 31678 of 31680 samples non-zero.

## What this does NOT show

**It is not a quality sample, and the content does not match the prompt.** The
frames are a maroon and blue lattice of panel-like shapes, not a fox in grass.
Three reasons, none of them yet excluded:

- **128x128 is far below the model's design point.** Upstream's quick start runs
  **121 frames** at the pipeline's native resolution. Nine frames at 128x128 is a
  smoke test, not a render.
- **The prompt is eight words.** Upstream documents a ~200-word cinematographic
  structure and this repository has never exercised one.
- Only the **NVFP4** arm on one checkpoint has ever run e2e.

A periodic structure in the frames suggested a 2x2 tile-replication defect. That
was **tested and refuted**: the four 64x64 quadrants differ by 27-47 mean
absolute difference against a control of 34 for two genuinely different frames.
The apparent periodicity came from a one-dimensional row-mean profile, which two
similar horizontal bands reproduce without any copying.

No oracle comparison exists for this render, and no throughput number was taken.

Tracking: #435. The keyframe bias this render's DiT declares but does not carry
is #902.
