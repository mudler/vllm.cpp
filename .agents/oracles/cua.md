# `cua` — cua-s1-forms TinyTransformerScorer, which vLLM does not register

vLLM has no `TinyTransformerScorer` architecture, no byte-level collator, and
no cross-attention option-scoring head. The model is a custom PyTorch
implementation that lives outside both vLLM and HuggingFace `transformers`.

`trycua/cua` is the model author's own runtime. Its `libs/cua-s1/` directory
carries the `TinyTransformerScorer` model definition (`model.py`), the
`ByteCollator` tokenizer, the `AttentionHead` cross-attention scorer, and the
inference pipeline (`planner.py`). The checkpoint is hosted at
`cua-ai/cua-s1-forms` on HuggingFace (`cua-s1-forms.safetensors` +
`cua-s1-forms.json` config sidecar).

This oracle answers: what correct cua-s1-forms output looks like (probabilities
per option given a context string and a list of option strings) on an
architecture vLLM cannot run.

```oracle-pin
id = cua
role = secondary
upstream = https://github.com/trycua/cua
scope = cua-s1-forms TinyTransformerScorer (byte-level 2-layer transformer encoder + cross-attention scoring head) for GUI form-filling decisions, which neither vLLM nor vLLM-Omni registers
pin = 9bbfa7dd3e27ca7f1861ede70aaca390174493f9
pin_label = main, 2026-09-19
pinned_on = 2026-09-21
gateable = no
evidence = MODEL-CUA-S1-FORMS spec owes the measurement
```

`gateable = no` because the oracle has not yet been run against this tree. The
spec for `MODEL-CUA-S1-FORMS` owes the measurement. The pin is the latest
commit on `main` at the time of writing, verified via the GitHub API.

Key source files at the pin:

- `libs/cua-s1/cua_s1/model.py` — `TinyTransformerScorer`, `ByteEmbedding`,
  `TransformerBlock`, `OptionEncoder`, `AttentionHead`
- `libs/cua-s1/cua_s1/planner.py` — inference pipeline, `ByteCollator`
- `libs/cua-s1/cua_s1/config.py` — config defaults (width=128, rank=128,
  context_tokens=224, option_tokens=96, layers=2, heads=4)

Checkpoint: `cua-ai/cua-s1-forms` on HuggingFace:
- `cua-s1-forms.safetensors` — 706K params, 2.8 MB
- `cua-s1-forms.json` — custom config sidecar (not standard HF `config.json`)

License: MIT (source); check the HuggingFace model card for artifact terms.
