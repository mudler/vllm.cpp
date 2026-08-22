# Qwen3.5

Qwen3.5 is a Gated DeltaNet (GDN) architecture. Its MoE members are also the
family that the expert streaming lane serves, which is how
[Qwen3.8 2.4T](qwen3-8-2-4t.md) runs from a file mapping.

Qwen3.5 runs through the shared paths, so [the quickstart](../QUICKSTART.md) and
[the usage guide](../USAGE.md) cover starting a server and sending a request.
This page carries the one config key that changes the arithmetic, and one load
refusal that is worth recognizing.

## The `output_gate_type` key

A Gated DeltaNet checkpoint, meaning the Qwen3.5 and Qwen3-Next family, chooses
its output-gate activation in `config.json`:

| `output_gate_type` | Gate applied |
|---|---|
| absent | `silu`, the upstream default |
| `"silu"` or `"swish"` | `silu`. `swish` is an alias, collapsed at load |
| `"sigmoid"` | `sigmoid` |
| present but `null`, `""`, or not a string | refused |

The key is read from the **resolved text config**, so a flat text-only
`config.json` and a multimodal wrapper that nests the text model under
`text_config` behave identically.

Any other value is **refused at load**, with a message naming the key and the
accepted set. It is never silently defaulted, because the wrong gate is a
numerics change that still emits plausible tokens
([#489](https://github.com/mudler/vllm.cpp/issues/489)).

Only an **absent** key takes the default. A key that is present but `null` or
empty is a value, not an absence. Upstream hands it straight to its
`assert output_gate_type in ["silu", "swish", "sigmoid"]` and errors, so this
loader refuses it as well rather than quietly reading it as `silu`.

## One load refusal that is about this code, not your checkpoint

Almost every load refusal names something your `config.json` or your tensors
actually declare. Exactly one does not:

```text
dense loader: LoadQwen3_5DenseLayer was given a tensor-presence probe that
answered YES for '__vllm_cpp__a_tensor_no_checkpoint_carries__', a name no
checkpoint carries.
```

That name is not in your checkpoint and is not supposed to be. The loader asks
about it to find out whether its own "is this tensor present?" predicate can
answer `no`, and this message means it cannot. Your checkpoint is fine. Please
report it with the model you were loading
([#1258](https://github.com/mudler/vllm.cpp/issues/1258)).

The check exists because a predicate that only ever said yes shipped twice in
one file, and what a reader saw was the opposite of the truth: a refusal naming
a block-wise FP8 scale tensor the checkpoint had never contained
([#1256](https://github.com/mudler/vllm.cpp/issues/1256)). A message that blames
the wrong side costs more than the failure does.
