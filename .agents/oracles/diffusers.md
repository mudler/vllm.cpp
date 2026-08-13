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

**Not gateable yet:** nothing in this repository has executed `diffusers` as an
oracle, so there is no measured revision to pin. #647 owes both.

```oracle-pin
id = diffusers
role = secondary
upstream = https://github.com/huggingface/diffusers
scope = schedulers, VAEs and diffusion pipelines vLLM-Omni does not implement
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
