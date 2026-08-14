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

**Not gateable yet:** nothing in this repository has executed `diffusers` as an
oracle. #647 owes the measurement; [`../specs/minimax-music3.md`](../specs/minimax-music3.md)
W0 is where it gets taken, and this record flips to `gateable = yes` when the
oracle demonstrably builds and runs the model.

```oracle-pin
id = diffusers
role = secondary
upstream = https://github.com/huggingface/diffusers
scope = schedulers, VAEs and diffusion pipelines vLLM-Omni does not implement, including models whose only reference implementation is an unmerged diffusers PR
pin = c6da9936e4bda83107943a16eb8682e9a37d8527
pin_label = PR #14456 head (minimax-music3-integration)
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
