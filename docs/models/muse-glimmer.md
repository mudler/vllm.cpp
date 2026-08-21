# Muse Glimmer

Use this page for Muse Glimmer checkpoints, commands, supported arms, and current limitations.

## Muse Glimmer: exactly what has been checked

`MuseGlimmerForCausalLM` / `MuseGlimmerForConditionalGeneration` are not in that
table: both towers forward and the perception encoder is wired, so an image or
video prompt runs instead of refusing. What has been *measured* is much narrower
than "it works", so it is worth stating precisely.

- The text tower ran on real tensors from the released 30B checkpoint at
  **reduced depth — 4 of its 52 layers.** Its **5 prefill argmax positions** are
  identical to a standalone torch transcription of the upstream source and to
  HF's own `muse_glimmer` implementation. The full-depth 52-layer arm of our
  forward has **never run**.
- Those are argmax positions from a single prefill, not generated tokens.
  **Multi-step decode is untested**, and so is the sliding window across steps.
- The perception encoder normalizes merged multimodal embeddings again as of
  #405. Its config key is absent from the released checkpoint and defaults on,
  which we had read as off — so image and video prompts before that fix skipped
  a normalization step. Still no reference decode for the vision path either
  way, so this corrects the code without changing what has been verified.
- **A config key that is absent takes the architecture's value**
  ([#412](https://github.com/mudler/vllm.cpp/issues/412)), not a neutral one:
  `qk_scale_factor` 43.784 (→ 3.87 at head_dim 128), `sliding_window` 2048,
  `output_multiplier` 0.196…, `final_logit_softcapping` 20.0, `rms_norm_eps`
  1e-5, `post_norm_eps` 1e-8. The released 30B `config.json` carries all six, so
  the text tower above is unchanged; the released GGUF and the DFlash drafter's
  `config.json` each omit some, and both used to run a quietly different model.
  Only an explicit `null` still disables the window or the soft-cap.
- Even at reduced depth this is agreement with independent transcriptions of the
  same upstream source, not agreement with the model's own runtime: the pinned
  oracle cannot load `muse_glimmer` at all.
- The perception encoder has **no reference check of any kind** — the wiring gate
  proves the tower is reachable and that its output lands on the image/video
  placeholder rows, not that an image produces the right tokens.
- Nothing has run end to end through the server, and **no speed number exists for
  this model on any axis**; there is no denominator to state one against.
- The ATEM reasoning and tool parsers are ported and unit-gated, but at the
  server's default `skip_special_tokens: true` the framing tokens they key on
  (`<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`) are stripped before the
  parser sees the text. Channel scoping is therefore an **open gap at server
  defaults** — see [FEATURES.md](../FEATURES.md) and
  [the spec](../../.agents/specs/muse-glimmer.md) §6.7.


## Muse Glimmer 30B from a GGUF k-quant

The text tower loads from a `muse-glimmer`-architecture GGUF, so the 30B model
runs from a ~17 GB k-quant instead of a ~60 GB bf16 checkpoint. Point `--model`
straight at the file; the config comes from the GGUF's own metadata, so no
`config.json` is needed:

```sh
./build/examples/vllm-server --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
```

Both published k-quants load (`muse-glimmer-30B-kquant-17gb.gguf` and the mixed
per-tensor `muse-glimmer-30B-kquant-dynamic.gguf`). Standard GGUF residency
knobs apply (`VT_GGUF_KEEP_QUANT`, `VT_GGUF_MMAP`, `VT_CPU_REF`); `o_proj`, the
attention output gate, `down_proj` and the merged `gate_up` stay quantized, while
the merged QKV, `lm_head` and the embedding table expand to bf16 because the
shared forward consumes them in a form a block encoding cannot take.

Four caveats:

- **A key the GGUF omits falls back to Muse Glimmer's own constant, not to a
  neutral one** ([#412](https://github.com/mudler/vllm.cpp/issues/412)). The
  released file's 32 metadata keys include no post-norm epsilon, so both sandwich
  post-norms used to run at `attention.layer_norm_rms_epsilon` (1e-5) where the
  architecture says 1e-8 — a factor of 1000. The same rule now covers
  `sliding_window` (2048, not "no window at all"), `output_multiplier`,
  `final_logit_softcapping` and the query pre-scale. This changes GGUF
  activations, though a same-binary A/B on the released k-quant produced
  **token-identical** greedy output on both of the prompts on record. The
  safetensors arm is unaffected: its `config.json` carries every one of those
  keys. A converter that emits
  `muse-glimmer.attention.post_norm_rms_epsilon` or `muse-glimmer.attention.scale`
  is honoured over the default.
- **The k-quant generates coherent text, but is not token-exact against
  llama.cpp.** Two defects had to be fixed to get there: the GGUF tokenizer gap
  ([#347](https://github.com/mudler/vllm.cpp/issues/347), pre `llama4` = the
  GPT-4o / o200k family) and the converter's Q/K RoPE row permutation
  ([#359](https://github.com/mudler/vllm.cpp/issues/359), which produced
  `" is is is ..."`). `"The capital of France is"` at `--temperature 0` now
  continues `" Paris. The capital of France is Paris. ..."`. llama.cpp on the
  same file agrees on the first token and then diverges; whether that residual is
  quantization drift or a second defect is open.
- **Image and video need the bf16 safetensors.** The released
  `mmproj-kquant.gguf` ships its patch embedding without the `patch_temporal`
  axis, so half the weight is not in the file; loading it is refused by name.
- **No speed number exists for this model in any weight format.** The pinned
  vLLM oracle cannot load `muse_glimmer` at all, so there is no denominator to
  quote and none is claimed.

Set `VLLM_MUSE_GGUF=<file>` (or `VLLM_MUSE_GGUF_LOAD=<file>` for the full
materialization) to run `test_muse_glimmer_gguf` against a real checkpoint;
without them the gate runs off committed header-only manifests.
