ID: ISSUE-GH-2405
Title: decode.audio.mel is 47 s of a 518 s LTX-2.5 render, and it is a serial scalar Conv2d
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2405
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `LTX25-AUDIO-DECODE-COST`
>
> `LTX25-RENDER-SPEED-PARITY` decomposed an LTX-2.5 render at the pinned oracle's
> own request (320x192, 25f, 8 steps, seed 42, bf16) on `dgx:gpu0`, n = 3, coverage
> 99.89%. `decode.audio.mel` read **47.171 s — 9.13% of a 518.4 s wall, at a 0.16%
> spread — for 1.02 s of audio.** It is 3.1x the whole 21B denoise loop and 3x
> `decode.video`. That row recorded it under `## Owed` with no hypothesis, by name:
> "`decode.audio.mel` at 47 s has no hypothesis, and saying so is better than
> inventing one."
>
> This issue owns the attribution and the repair.
>
> **What it is.** `decode.audio.mel` is one call to `Ltx2AudioDecoderForward`
> (`src/vllm/model_executor/models/ltx2_audio_vae.cpp`). At the shipped
> checkpoint's geometry the decoder issues 28 two-dimensional convolutions
> totalling 31.46 GMAC, and 99.7% of the leaf is inside the file-private `Conv2d`
> loop — a serial quintuple-nested scalar loop with a `double` accumulator, on one
> core, sustaining ~1.2 GMAC/s. That is one multiply-accumulate per ~4 cycles,
> which is the latency of the dependent accumulator chain and nothing else.
>
> **Upstream does not do this.** `CausalConv2d.forward`
> (Lightricks/LTX-2 @ `fd4ded7f`,
> `packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py:61-64`) is
> `F.pad` then `torch.nn.Conv2d`, and `ti2vid_one_stage.py:117-123` builds the
> decoder with `device=self.device` (cuda), reaching
> `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1190`. The oracle
> renders the entire request, load included, in 93.8 s, so 47 s of mel is not
> inherent to the task.
>
> **The asymmetry is internal too.** The video VAE's convolution was moved to f32,
> parallelised and put behind `vt::Conv3d` by #1007 / #1008 / #1009. #1009 excluded
> the audio VAE in as many words. The audio half has had none of it since.
>
> **Owed after this issue**, and not carried by it, because it changes goldens
> rather than being bit-exact: the `double` accumulator, which is a divergence from
> `nn.Conv2d`'s tensor-dtype accumulation exactly as #1008 measured for the video
> half, and the device arm through the existing `vt::Conv2d` op.
>

## Resolution

GitHub records closing pull request #2425 (https://github.com/mudler/vllm.cpp/pull/2425) merged on 2026-08-31 as commit `79ee71b710ee28464f910db13ad508f68ac47138`. GitHub closed issue #2405 on 2026-08-31.
