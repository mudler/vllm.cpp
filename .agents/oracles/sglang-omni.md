# SGLang-Omni — the omni/speech/TTS serving runtime

A **third** repository, `sgl-project/sglang-omni`, distinct from both
`sgl-project/sglang` ([`sglang.md`](sglang.md)) and from vLLM-Omni. It owns the
multi-stage pipeline topology for omni, speech and TTS models and composes with
SGLang for the autoregressive scheduling underneath. It gets its own record
because it has its own cadence and its own model registry: a model can be in
SGLang-Omni and in neither of the other two.

**Why it exists here.** `MiniMaxAI/MiniMax-Music3` is served by SGLang-Omni
(`sglang_omni/models/minimax_music3/`) and is registered nowhere in the pinned
vLLM. It is the cross-check for the e2e output and the speed axis of
[`../specs/minimax-music3.md`](../specs/minimax-music3.md), where `diffusers` is
the per-component primary.

**It serves the NATIVE checkpoint layout, not the diffusers one.**
`sglang_omni/models/minimax_music3/checkpoint.py:35-56` resolves exactly
`qwen_7B/qwen_7B`, `flowmatching_vae.pth` and `dav.pth`, and pulls the RVQ depth
decoder out of the Qwen shards by the `model.audio_decoder.` /
`model.audio_extra_embedding` prefixes. That is a different packaging of the same
weights (the diffusers conversion script is a pure re-layout), so a comparison
against it is meaningful — but only at the whole-pipeline level, since its tensor
names do not line up component-by-component with ours.

**Speed comparisons use its production defaults**, which its own README lists as
on without further flags: backbone decode CUDA graph, RVQ depth CUDA graph,
compiled DIT blocks, compiled DAV decoder, and batched seeded sampling. Measuring
against it with those disabled would be the dishonest-denominator mistake the
benchmark protocol forbids.

**Not gateable yet:** cloned and read, never executed here. It requires CUDA and
its own pinned `sglang.multimodal_gen` / `flashinfer-python==0.6.14` stack.

```oracle-pin
id = sglang-omni
role = secondary
upstream = https://github.com/sgl-project/sglang-omni
scope = omni, speech, TTS and music models served by the SGLang-Omni pipeline runtime and absent from vLLM; e2e output and the speed axis, not per-component parity
pin = 748a0b437e4a8faad44d7bbfd5a0ae55d1fef830
pin_label = main @ 2026-08-13
pinned_on = 2026-08-13
gateable = no
evidence = #672
```
