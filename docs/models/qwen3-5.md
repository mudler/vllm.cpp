# Qwen3.5

Use this page for Qwen3.5 checkpoints, commands, supported arms, and current limitations.

## One load refusal that is about this code, not your checkpoint

Almost every load refusal in this document names something your `config.json`
or your tensors actually declare. Exactly one does not:

```text
dense loader: LoadQwen3_5DenseLayer was given a tensor-presence probe that
answered YES for '__vllm_cpp__a_tensor_no_checkpoint_carries__', a name no
checkpoint carries.
```

That name is not in your checkpoint and is not supposed to be. The loader asks
about it to find out whether its own "is this tensor present?" predicate is
capable of answering `no`, and this message means it is not. Your checkpoint is
fine; please report it with the model you were loading
([#1258](https://github.com/mudler/vllm.cpp/issues/1258)).

The check exists because a predicate that only ever said yes shipped twice in one
file, and what a reader saw was the *opposite* of the truth: a refusal naming a
block-wise FP8 scale tensor the checkpoint had never contained
([#1256](https://github.com/mudler/vllm.cpp/issues/1256)). A message that blames
the wrong side costs more than the failure does.


## GDN checkpoints: the `output_gate_type` key

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


## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../../examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. Every key is checked and none is dropped: an unknown or misspelled name is refused at startup by name, and a real vLLM key this engine does not implement is refused as such ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)). See [docs/SPECULATIVE-DECODING.md](../SPECULATIVE-DECODING.md) |
| `--offload-config '<json>'` | (unset) | Weight placement, the same JSON document `vllm-server` takes and the same C ABI field. Both halves: vLLM's mirrored `uva`/`prefetch` device-to-host weight offload, and vllm.cpp's `vllm_cpp` key for the host-to-disk residency tier that makes a checkpoint larger than host RAM loadable. An unknown key at any level of the document is refused at startup by name. Added by [#1135](https://github.com/mudler/vllm.cpp/issues/1135); see [Streaming routed experts from disk](#streaming-routed-experts-from-disk-capacity-mode) |
| `--max-num-seqs N` | engine default (32) | Max concurrent sequences. Under speculative decoding on a GDN model the recurrent state is `max-num-seqs x (k+1)` per slot, so this is the knob to lower when a run is refused for state budget |
| `--repeat N` | `1` | Load once, then run N blocking completions. Use it to read a warm decode tok/s without paying model load each time. Not supported with `--stream`, which falls back to 1 |
| `-h`, `--help` | | Print usage and exit |

`--model` resolves a Qwen3.5-family checkpoint's backbone under EITHER weight
namespace. The multimodal wrappers (`Qwen3_5ForConditionalGeneration`,
`Qwen3_5MoeForConditionalGeneration`) publish the text backbone nested under
`model.language_model.`; the text-only arms (`Qwen3_5ForCausalLM`,
`Qwen3_5MoeForCausalLM`) publish it flat under `model.`. The loader decides which
ONCE per checkpoint from the shard index, and REFUSES a checkpoint that carries
backbone tensors under both rather than binding half the model from each.

**Resolving the namespace is not the same as loading the checkpoint, and the
MoE and dense arms differ.** The dense loader routes each projection to BF16,
FP8 or NVFP4 by tensor presence, so a flat bf16 `Qwen3_5ForCausalLM` checkpoint
is expected to load. The **MoE** loader reads two ROUTED-EXPERT layouts and
decides between them ONCE per checkpoint from the shard index: per-expert NVFP4
(`experts.<e>.<proj>.weight` U8 + `.weight_scale` + `.weight_scale_2`, what an
NVFP4 requant ships) and the 3-D stacked BF16
`experts.{gate_up_proj,down_proj}` the published repos (`Qwen/Qwen3.8-2.4T-A95B`,
`Qwen/Qwen3.6-35B-A3B`) ship. A checkpoint carrying BOTH spellings under its
backbone is refused rather than half-bound.

**Outside the routed experts the MoE arm routes by tensor presence too.** The GDN
tower (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`) and the attention tower
(`self_attn.{q,k,v,o}_proj`) read BF16 or per-tensor FP8; the shared expert
(`mlp.shared_expert.{gate,up,down}_proj`) and `lm_head` read BF16 or NVFP4. Each
of the four is resolved ONCE per checkpoint, and a component whose own
projections disagree — layer 0's `q_proj` BF16 beside layer 4's F8_E4M3 — is
refused naming both sides rather than bound half from each. Different components
MAY disagree with each other: a `modelopt_mixed` checkpoint really does ship an
FP8 tower beside an NVFP4 MLP, and the dense arm reads exactly that.

**Which code runs an FP8 projection is no longer a Qwen3.5 detail.** The
per-tensor FP8 W8A8 residency and GEMM entry points live in
`include/vllm/model_executor/models/dense_fp8_gemm.h`, with the scheme policy in
`include/vllm/model_executor/layers/quantization/fp8.h`, so any model binds them
through `layers::MakeLinearMethod(bf16_weight, fp8_weight)` — the same shape the
NVFP4 W4A16 seam already had. The bound method exposes two arms: `Apply`, which
quantizes the activation itself with the checkpoint's `input_scale`, and
`ApplyPreQuantized`, which takes an activation a preceding fused epilogue already
quantized and runs only the GEMM. Nothing about running Qwen3.5 changes: the
levers (`VT_DENSE_NATIVE`, `VT_DENSE_CUBLASLT_FP8`) keep their names and
defaults, and the path stays CUDA-only.

Still OWED for the MoE arm, and refused BY NAME rather than discovered as a dtype
complaint: an NVFP4 attention or GDN tower, an FP8 shared expert, an FP8
`lm_head`, a per-expert-but-unquantized routed layout, and a non-BF16 stacked
expert tensor.

**The MoE arm's VISION TOWER.** `LoadQwen3_5Moe` reads the text backbone only.
`Qwen/Qwen3.6-35B-A3B` ships 333 `model.visual.*` tensors alongside it, and until
issue #891 they were dropped without a word — the load succeeded and produced a
text-only model. `LoadQwen3_5MoeVision` now reads them, through the SAME
`LoadQwen3VLVisionWeights` the dense `Qwen3_5ForConditionalGeneration` arm uses,
with the tower geometry from the checkpoint's `vision_config` (depth 27, hidden
1152, 16 heads, intermediate 4304, patch 16, spatial merge 2, EMPTY
`deepstack_visual_indexes`) and `out_hidden_size` taken from the text hidden size
because the merger writes into the text residual stream. A checkpoint carrying NO
`model.visual.*` tensor is REFUSED naming them, rather than quietly loading a
model that answers image prompts from text alone — `nvidia/Qwen3.6-35B-A3B-NVFP4`
declares `vision_config` and ships no `visual.*` weights, and is exactly that
case.

**What is and is not proven about a published bf16 MoE repo.** Every arm is
byte-exact on synthetic fixtures, and the real published `Qwen/Qwen3.6-35B-A3B`
and `Qwen/Qwen3.8-2.4T-A95B` indices satisfy the load plan completely — every
name, dtype and enforced shape the reader asks for
(`tests/vllm/models/test_qwen3_8_text_only.cpp`). That reads NO weight byte and
is NOT a token claim: a wrong dtype path or a missing dequant produces wrong
logits rather than an error, so only a token-exact gate closes it. No text-only
Qwen3.5 checkpoint has been RUN here — see [STATUS.md](../STATUS.md) for the owed
run gates.

GGUF and safetensors mapped-payload paths, plus safetensors index paths, use the
host's native filesystem encoding, including Unicode paths on Windows. Native
Windows release artifacts are not published yet; they will remain unavailable
until the `v0.0.3-pre.1` prerelease build and publication gates succeed.

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`. It pretokenizes before timing
  and atomically publishes each concurrency wave. Set
  `VT_BENCH_PRETOKENIZE=0` for the timed-string rollback; the report names the
  resolved mode.
- `tokenize` ([`examples/tokenize/main.cpp`](../../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.
  GGUF `tokenizer.ggml.pre` names accepted: `qwen35`, `qwen2`, `llama-bpe`,
  `gpt-4o` / `llama4` / `kanana2` / `talkie` (the GPT-4o / o200k family),
  `joyai-llm`, `deepseek-llm`, `deepseek-v3`, `laguna`. Any other name is
  refused by name rather than aliased onto a near-miss regex.
