# HuggingFace `transformers` — the reference vLLM itself mirrors

Where vLLM implements a model, `transformers` is not the authority: vLLM is,
including where the two differ. What `transformers` answers is the layer beneath
that — a model's own reference implementation, and the processors, feature
extractors and tokenizers vLLM delegates to rather than reimplements. That is
why it is a legitimate oracle for a preprocessing golden even on models where
vLLM is the behavior spec.

**The pin is the one resolved inside the pinned vLLM environment**, not an
independently chosen release: `transformers` 5.14.1 is what
[`../upstream-sync.md`](../upstream-sync.md) records alongside vLLM
`555967922`. Pinning it separately would let the oracle environment hold two
different `transformers` at once, which is the drift this registry exists to
stop.

Precedent for it running as an oracle: the AUDIO track's A0 captured the
`WhisperFeatureExtractor` log-mel reference, the mel filterbank and the audio
placeholder expansion directly from it
([`../specs/audio-track.md`](../specs/audio-track.md) §A0), and the C++ STFT is
gated against those goldens.

**It cannot adjudicate a near-tie, and one attempt is recorded here so the next
agent does not repeat it.** A CPU `transformers` probe of the three Qwen3.8-27B
divergences ([`../specs/qwen38-27b-bf16-gate.md`](../specs/qwen38-27b-bf16-gate.md),
[#915](https://github.com/mudler/vllm.cpp/issues/915)) reported all three as
exact ties — and every runner-up gap it printed was an exact multiple of
**0.125**, one bf16 ULP in that exponent range. An instrument quantized to one
ULP cannot resolve a gap below one ULP, so "tied in bf16" does not imply "tied
in the model" and agreement with it is not confirmation. vLLM computes logprobs
from **fp32** and separates pairs bf16 collapses, so a logit-margin verdict
comes from the pinned vLLM oracle; this one stands only as a secondary
cross-check, with that limitation part of the result.

```oracle-pin
id = transformers
role = secondary
upstream = https://github.com/huggingface/transformers
scope = a model, processor, feature extractor or tokenizer's own reference implementation, at the version the pinned vLLM environment resolves
pin = 5.14.1
pin_label = 5.14.1
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/specs/audio-track.md
```
