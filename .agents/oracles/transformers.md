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
