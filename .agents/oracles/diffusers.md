# `diffusers` — schedulers, VAEs and diffusion pipelines

The diffusion lane's counterpart to `transformers`: where a model ships a
`diffusers` pipeline and vLLM-Omni does not implement it, `diffusers` is the
reference for the flow-matching or denoising schedule, the VAE decode, and the
pipeline's step-by-step composition. Several checkpoints ship a `diffusers`
integration before, or instead of, any serving-engine one — `MiniMax-Music3`
recommends `diffusers` alongside SGLang-Omni, and its flow-matching and Flow-VAE
stages are the parts a scheduler oracle answers exactly.

Prefer vLLM-Omni wherever it implements the pipeline; reach here when it does
not, and record which of the two a given stage was gated against, because they
do not always agree on scheduler details.

**The pin is an OPEN pull request, deliberately.** MiniMax-Music3's diffusers
integration is PR [#14456](https://github.com/huggingface/diffusers/pull/14456)
(`Add MiniMax Music 3`), branch `huggingface:minimax-music3-integration`, which is
not merged to `main`. A branch name is not a pin — it can be rebased or
force-pushed under us — so the recorded revision is the exact head commit. If the
PR merges, moving to the merge commit is a pin advance with its own
reconciliation, not a silent follow. The same shape applies to any future
diffusers-only model: the H3 integration landed the same way (#14355, refactored
by #14371).

**Gateable since 2026-08-14, because it generated audio.** The bar is that the
oracle demonstrably *builds and runs the model* — constructing a pipeline object
proves nothing — and it now does:
[`tools/oracle/music3_oracle.py`](../../tools/oracle/music3_oracle.py) loads all
seven MiniMax-Music3 components from the local diffusers-arm checkpoint and
generates a 44100 Hz stereo waveform of shape `(1, 2, 44032)` from a fixed seed.
The per-stage reference tensors, the resolved environment and the request that
produced them are in
[`tests/parity/goldens/minimax_music3_oracle/manifest.json`](../../tests/parity/goldens/minimax_music3_oracle/manifest.json).

The script **asserts the installed revision** against the pin below before it
loads a weight, reading the distribution's recorded VCS commit rather than
trusting the venv, because a venv silently holding a different revision than the
record claims is a failure this project has already paid for once.

Two facts that record scope rather than success. The capture ran **on CPU**, so
nothing here is a speed measurement — it is a correctness reference. And the
on-disk dtypes are **not** a runnable configuration: upstream's pipeline casts
only condition→transformer and latents→vocoder, so the condition encoder and the
depth decoder must share the language model's dtype. The gated configuration is
bf16 autoregressive half / fp32 acoustic half; `tools/oracle/README.md` records
the exact error the alternative raises.

```oracle-pin
id = diffusers
role = secondary
upstream = https://github.com/huggingface/diffusers
scope = schedulers, VAEs and diffusion pipelines vLLM-Omni does not implement, including models whose only reference implementation is an unmerged diffusers PR
pin = c6da9936e4bda83107943a16eb8682e9a37d8527
pin_label = PR #14456 head (minimax-music3-integration)
pinned_on = 2026-08-14
gateable = yes
evidence = tests/parity/goldens/minimax_music3_oracle/manifest.json
```
