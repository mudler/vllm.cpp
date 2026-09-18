# `vllm-factory` — GLiNER2.5 on vLLM, which vLLM does not implement

vLLM has no DeBERTa encoder, no disentangled attention, and no GLiNER2
pooler head. Upstream PRs for DeBERTa (#42094, #20215) are open and unmerged,
and maintainers stated "no plan support for DebertaV2." The disentangled
attention mechanism (content-to-content, content-to-position,
position-to-content terms) is architecturally incompatible with vLLM's
standard attention path, so the C++ port must implement it from scratch.

`ddickmann/vllm-factory` is a vLLM plugin that fills exactly this gap. Its
`plugins/deberta_gliner2/` directory carries `GLiNER2VLLMModel` (a custom
`DebertaV2EncoderModel` backbone plus `GLiNER2Pooler` head wrapped in a
`VllmPoolerAdapter`), `DeBERTaGLiNER2IOProcessor` (schema-based preprocessing,
`task="plugin"`), and a parity test harness. The plugin uses
`poolers.gliner2.GLiNER2Pooler` and a custom Flash-DeBERTa Triton encoder.

This oracle answers: what correct GLiNER2.5 output looks like on a model
architecture vLLM cannot run. It is the reference for the encoder, the pooler
head, the preprocessing pipeline, and the expected span/entity output format.
HuggingFace `transformers` (via `trust_remote_code` on the `gliner2` library)
is the secondary cross-check for the DeBERTa encoder and the GLiNER2 layers.

```oracle-pin
id = vllm-factory
role = secondary
upstream = https://github.com/ddickmann/vllm-factory
scope = GLiNER2.5 on vLLM (DeBERTa v2 encoder + disentangled attention + GLiNER2 pooler head + IO processor), which vLLM does not implement at the pin
pin = 7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0
pin_label = HEAD, 2026-09-18
pinned_on = 2026-09-18
gateable = no
evidence = #3216
```

`gateable = no` because the oracle has not yet been run on this tree's
hardware. Issue [#3216](https://github.com/mudler/vllm.cpp/issues/3216) owes
the measurement. The pin is the latest commit on `main` at the time of
writing, verified via the GitHub API.
The `plugins/deberta_gliner2/` directory and its dependencies
(`poolers/gliner2.py`, `models/deberta_v2/deberta_v2_encoder.py`) are the
source files to port.

## Source files to port

| File | What it provides |
|---|---|
| `plugins/deberta_gliner2/model.py` | `GLiNER2VLLMModel`: backbone + pooler wiring, `VllmPoolerAdapter` |
| `plugins/deberta_gliner2/io_processor.py` | `DeBERTaGLiNER2IOProcessor`: schema-based preprocessing, `task="plugin"` |
| `plugins/deberta_gliner2/processor.py` | GLiNER2 processor: span candidate generation, entity label encoding |
| `plugins/deberta_gliner2/config.py` | Model config: `GLiNER2Config` fields and defaults |
| `plugins/deberta_gliner2/lora.py` | LoRA on the encoder backbone |
| `plugins/deberta_gliner2/parity_test.py` | Parity test harness against `transformers` |
| `poolers/gliner2.py` | `GLiNER2Pooler`: SpanRep, count embed, classifier, count predictor |
| `models/deberta_v2/deberta_v2_encoder.py` | `DebertaV2EncoderModel`: disentangled attention encoder |
