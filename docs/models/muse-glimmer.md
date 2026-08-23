# Muse Glimmer

Muse Glimmer is a 30B multimodal model. `MuseGlimmerForCausalLM` and
`MuseGlimmerForConditionalGeneration` both forward, and the perception encoder is
wired, so an image or video prompt runs instead of being refused.

**What has been measured is much narrower than "it works."** Read
[what has actually been checked](#what-has-actually-been-checked) before you rely
on any of it. Nothing has run end to end through the server, and no speed number
exists for this model on any axis.

## Run the text tower from a GGUF

The text tower loads from a `muse-glimmer`-architecture GGUF, so the 30B model
runs from a roughly 17 GB k-quant instead of a roughly 60 GB BF16 checkpoint.
Point `--model` straight at the file. The config comes from the GGUF's own
metadata, so no `config.json` is needed:

```sh
./build/examples/vllm-server --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
```

Both published k-quants load: `muse-glimmer-30B-kquant-17gb.gguf` and the mixed
per-tensor `muse-glimmer-30B-kquant-dynamic.gguf`.

The standard GGUF residency knobs apply, which are `VT_GGUF_KEEP_QUANT`,
`VT_GGUF_MMAP`, and `VT_CPU_REF`. `o_proj`, the attention output gate,
`down_proj`, and the merged `gate_up` stay quantized. The merged QKV, `lm_head`,
and the embedding table expand to BF16, because the shared forward consumes them
in a form a block encoding cannot take.

**Image and video need the BF16 safetensors.** The released
`mmproj-kquant.gguf` ships its patch embedding without the `patch_temporal`
axis, so half the weight is not in the file. Loading it is refused by name.

### The GGUF k-quant is not token-exact against llama.cpp

`"The capital of France is"` at `--temperature 0` continues
`" Paris. The capital of France is Paris. ..."`. llama.cpp on the same file
agrees on the first token and then diverges. Whether that residual is
quantization drift or a second defect is open.

Two defects had to be fixed to get that far: the GGUF tokenizer gap
([#347](https://github.com/mudler/vllm.cpp/issues/347), where `pre llama4` is the
GPT-4o and o200k family) and the converter's Q and K RoPE row permutation
([#359](https://github.com/mudler/vllm.cpp/issues/359), which produced
`" is is is ..."`).

### An absent config key takes the architecture's value

A key that a config or a GGUF omits falls back to Muse Glimmer's own constant,
not to a neutral one ([#412](https://github.com/mudler/vllm.cpp/issues/412)):

| Key | Value taken when absent |
|---|---|
| `qk_scale_factor` | 43.784, which is 3.87 at `head_dim` 128 |
| `sliding_window` | 2048, not "no window at all" |
| `output_multiplier` | 0.196... |
| `final_logit_softcapping` | 20.0 |
| `rms_norm_eps` | 1e-5 |
| `post_norm_eps` | 1e-8 |

The released 30B `config.json` carries all six, so the safetensors arm is
unaffected. The released GGUF and the DFlash drafter's `config.json` each omit
some, and both used to run a quietly different model. The released file's 32
metadata keys include no post-norm epsilon, so both sandwich post-norms used to
run at `attention.layer_norm_rms_epsilon` (1e-5) where the architecture says
1e-8, a factor of 1000.

Correcting this changes GGUF activations, though a same-binary A/B on the
released k-quant produced **token-identical** greedy output on both prompts on
record. Only an explicit `null` still disables the window or the soft-cap. A
converter that emits `muse-glimmer.attention.post_norm_rms_epsilon` or
`muse-glimmer.attention.scale` is honored over the default.

## What has actually been checked

- The text tower ran on real tensors from the released 30B checkpoint at
  **reduced depth, 4 of its 52 layers.** Its **5 prefill argmax positions** are
  identical to a standalone torch transcription of the upstream source and to
  Hugging Face's own `muse_glimmer` implementation. The full-depth 52-layer arm
  of this forward has **never run**.
- Those are argmax positions from a single prefill, not generated tokens.
  **Multi-step decode is untested**, and so is the sliding window across steps.
- Even at reduced depth, this is agreement with independent transcriptions of the
  same upstream source, not agreement with the model's own runtime. The pinned
  oracle cannot load `muse_glimmer` at all.
- The perception encoder has **no reference check of any kind**. The wiring gate
  proves the tower is reachable and that its output lands on the image and video
  placeholder rows. It does not prove that an image produces the right tokens.
  The encoder normalizes merged multimodal embeddings again as of #405. Its
  config key is absent from the released checkpoint and defaults on, which had
  been read as off, so image and video prompts before that fix skipped a
  normalization step.
- **Nothing has run end to end through the server**, and **no speed number exists
  for this model in any weight format**. The pinned vLLM oracle cannot load
  `muse_glimmer`, so there is no denominator to quote and none is claimed.
- The ATEM reasoning and tool parsers are ported and unit-gated. At the server's
  default `skip_special_tokens: true` the framing tokens they key on
  (`<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`) are stripped before the
  parser sees the text, so channel scoping is an **open gap at server defaults**.
  See [`docs/FEATURES.md`](../FEATURES.md) and
  [the spec](../../.agents/specs/muse-glimmer.md) section 6.7.

## Run the gate against a real checkpoint

Set `VLLM_MUSE_GGUF=<file>`, or `VLLM_MUSE_GGUF_LOAD=<file>` for the full
materialization, to run `test_muse_glimmer_gguf` against a real checkpoint.
Without them the gate runs off committed header-only manifests.
