# Qwen3-Next

Qwen3-Next is a Gated DeltaNet (GDN) architecture. It runs through the shared
paths, so [the quickstart](../QUICKSTART.md) and
[the usage guide](../USAGE.md) cover starting a server and sending a request.
Qwen3.5 shares the architecture and [its page](qwen3-5.md) carries the same gate
rule.

This page carries the one config key that changes the arithmetic and that a
checkpoint can get wrong.

## The `output_gate_type` key

A Gated DeltaNet checkpoint (the Qwen3.5 / Qwen3-Next family) chooses its
output-gate activation in `config.json`:

| `output_gate_type` | Gate applied |
|---|---|
| absent | `silu` — the upstream default |
| `"silu"` or `"swish"` | `silu` — `swish` is an alias, collapsed at load |
| `"sigmoid"` | `sigmoid` |
| present but `null`, `""`, or not a string | refused |

The key is read from the **resolved text config**, so a flat text-only
`config.json` and a multimodal wrapper that nests the text model under
`text_config` behave identically. Any other value is **refused at load** with a
message naming the key and the accepted set — never silently defaulted, because
the wrong gate is a numerics change that still emits plausible tokens
([#489](https://github.com/mudler/vllm.cpp/issues/489)).

Only an **absent** key takes the default. A key that is present but `null` or
empty is a value, not an absence: upstream hands it straight to its
`assert output_gate_type in ["silu", "swish", "sigmoid"]` and errors, so this
loader refuses it as well rather than quietly reading it as `silu`.
