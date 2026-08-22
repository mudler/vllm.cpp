# Model loading reference

Use this page to look up shared loader behavior and diagnostics.

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
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. Every key is checked and none is dropped: an unknown or misspelled name is refused at startup by name, and a real vLLM key this engine does not implement is refused as such ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)). See [the speculative decoding guide](../SPECULATIVE-DECODING.md) |
| `--offload-config '<json>'` | (unset) | Weight placement, the same JSON document `vllm-server` takes and the same C ABI field. Both halves: vLLM's mirrored `uva`/`prefetch` device-to-host weight offload, and vllm.cpp's `vllm_cpp` key for the host-to-disk residency tier that makes a checkpoint larger than host RAM loadable. An unknown key at any level of the document is refused at startup by name. Added by [#1135](https://github.com/mudler/vllm.cpp/issues/1135); see the [expert streaming guide](../guides/expert-streaming.md) |
| `--max-num-seqs N` | engine default (32) | Max concurrent sequences. Under speculative decoding on a GDN model the recurrent state is `max-num-seqs x (k+1)` per slot, so this is the knob to lower when a run is refused for state budget |
| `--repeat N` | `1` | Load once, then run N blocking completions. Use it to read a warm decode tok/s without paying model load each time. Not supported with `--stream`, which falls back to 1 |
| `-h`, `--help` | | Print usage and exit |


See the [Qwen3.8 27B model recipe](../models/qwen3-8-27b.md) for checkpoint-specific
weights, supported arms, and current limitations. See
[project status](../../README.md#project-status) for pending run gates.

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

## Which HF tokenizers load

A checkpoint's `tokenizer.json` is accepted when its `pre_tokenizer` is one this
build recognises. Recognition is by exact regex or pipeline shape, not by model
name, so a checkpoint from any vendor loads if it carries one of these:

| family | shape | examples |
|---|---|---|
| Qwen3.6 | one `Split` regex, single-codepoint `\p{N}`, `\p{M}` folded into letter runs | Qwen3.6-27B |
| Qwen2/Qwen3 classic | as above without `\p{M}` awareness | Qwen3-0.6B, Qwen3-Coder |
| Llama-3 | `\p{N}{1,3}` digit groups, no `\p{M}` awareness | Llama-3 family |
| Tekken (Mistral) | case-aware letter runs, single-codepoint `\p{N}`, `/` in the punct tail | Mistral-Nemo-Instruct-2407 |
| GPT-4o / o200k | the same case-aware letter runs, plus o200k's contraction SUFFIX and `\p{N}{1,3}` | Muse Glimmer (pre `llama4`), GPT-4o |
| GPT-2 byte-level | `ByteLevel(use_regex=true)` with no explicit `Split` | OPT, GPT-2 |
| DeepSeek | a seven-stage `Sequence` pipeline, not one alternation | DeepSeek-V2/V3 |
| SentencePiece | `Metaspace` + byte-fallback vocab | Mistral-7B-v0.3 |

An unrecognised one fails loudly at load with `tokenizer: unrecognized
pre-tokenizer split regex: <regex>`, rather than tokenizing incorrectly. If you
hit that, the printed regex is what a new pattern would have to match.

Note that Mistral ships **two** unrelated tokenizer families: Mistral-7B-v0.3 is
SentencePiece, while Mistral-Nemo is Tekken, a byte-level BPE whose regex is
tiktoken's `o200k_base` with the contraction group removed and `\p{N}{1,3}`
reduced to `\p{N}`. Support for one says nothing about the other. Putting those
two edits back gives the GPT-4o row above, so the two share one scanner's
character classes but stay separate patterns: they disagree on `don't` and on
every digit run longer than one.

## Load memory and diagnostics

On a unified-memory device (a DGX Spark) the Vulkan heap and system RAM are the
same bytes. Include the checkpoint and the KV pool when you estimate capacity.
Reading a checkpoint also fills the reclaimable page cache. Use `MemAvailable`,
not `MemFree`, to decide whether a model fits.

Set `VT_VULKAN_ALLOC_STATS=1` to print the running device allocation total and
the `/proc` memory context. See [Benchmarks](../BENCHMARKS.md) for accepted
measurements and the [Vulkan support spec](../../.agents/specs/vulkan-full-support.md)
for measurement history.

### Inspect load phases and moved bytes

`VT_LOAD_STATS=1` prints one line per load phase with its wall time, plus the
bytes the load actually MOVED: `host_copy` (materialized into a host buffer),
`borrowed` (read in place from the file mapping) and `device_upload`. The byte
line is printed twice, once when the weights are loaded and once at exit, because
the device uploads are lazy and happen at first use.

```
$ VT_LOAD_STATS=1 build/examples/vllm-cli --model /path/to/Qwen3.6-27B --prompt hi --max-tokens 1
[vt load] mmap+header       0.027 s
[vt load] weights          12.268 s
[vt load] bytes@load-end  host_copy=31.162 GiB borrowed=18.936 GiB device_upload=0.000 GiB
[vt load] bytes@exit      host_copy=31.162 GiB borrowed=18.936 GiB device_upload=50.098 GiB
```

`borrowed` counts weights that the device reads from the checkpoint mapping.
Merged, transposed, and dequantized tensors use `host_copy` instead. Set
`VT_LOAD_DIRECT_UPLOAD=0` to disable direct upload.

Safetensors payloads are byte-addressed and do not promise natural scalar
alignment. Borrowed BF16/F16/F32 inputs therefore use defined byte-copy loads;
an odd payload offset neither forces a host copy nor changes the loaded bits.

`device_upload` counts single-source BF16, FP8, NVFP4, and MXFP4 uploads. It does
not count merged FP4 operands or Marlin repack buffers. The loader releases the
source pages after upload, regardless of `VT_ADOPT_DEVICE_BYTES`. See the
[direct-upload spec](../../.agents/specs/load-direct-upload.md) for measurement
history and counter coverage.

### A per-tensor scale has to be one F32 number

Every scale this build reads as a single number is required to be exactly one
element and exactly `F32`. That covers `weight_scale`, `input_scale`,
`weight_scale_2`, `weight_global_scale`, `input_global_scale`, `k_scale` and
`v_scale`. A checkpoint that stores one of them as an array, or in a narrower
dtype, is refused at load with a message naming the tensor, the shape it
shipped, and the dtype it shipped:

```text
dense loader: 'model.layers.0.self_attn.q_proj.weight_scale' ships shape
[12288, 1] (12288 elements), not the ONE element a per-tensor scale is
```

The loader refuses per-output-channel and block-wise FP8 layouts at these
single-value fields. Per-output-channel FP8 is not implemented. `lm_head` has
its own per-output-channel scale path. See the
[scalar-scale guard spec](../../.agents/specs/read-f32-scalar-guard.md) for the
defect history.

### Attention backend refusals

Starting an engine resolves an attention backend for each KV-cache group, and
that backend is now asked whether it can serve the request before it is chosen.
When none of the backends this build registers can, the engine refuses at
initialization rather than later, and the message names every candidate with
every reason it lost:

```text
No valid attention backend for device type 1 from
{FLASH_ATTN: [head_size not supported, block_size not supported]}
(use_mla=false, use_sparse=false)
```

The message uses the checkpoint's resolved head size, block size, and KV-cache
dtype. CPU accepts model dtypes `f32`, `f16`, and `bf16`. Its KV cache accepts
`auto`, `fp8`, and `fp8_e4m3`. NVIDIA GPUs accept `auto`, `float16`, `bfloat16`,
`fp8`, and `fp8_e4m3`. Both devices refuse `fp8_e5m2` by name.

This validation checks the backend's declared capabilities. It does not prove
that the binary contains code for the current GPU. A later kernel launch error,
such as `cudaErrorUnsupportedPtxVersion`, indicates a build-architecture
mismatch. See the [compiled-architecture spec](../../.agents/specs/cuda-compiled-arch-manifest.md)
for that diagnostic path and the [attention validation spec](../../.agents/specs/attn-validate-configuration.md)
for the validation boundary.

Selecting a backend by name is not exposed yet; the engine always resolves one.

### Architectures that resolve but refuse to run

A few architectures are registered so their config and weight layout are
accounted for, while their forward is deliberately not implemented. Pointing the
CLI or server at one of these loads far enough to resolve the architecture and
then fails with a message naming the missing piece, rather than emitting wrong
tokens quietly.

| Architecture | Why it refuses |
|---|---|
| `KimiK3ForConditionalGeneration` | Needs ~1.56 TB (MXFP4); no host here can run it |
| `NemotronHForCausalLM` | Single-request paged decode runs. Batched decode refuses when `num_reqs > 1`. The token gate remains pending, `lm_head` reaches the device on the paged forward (A2-Q2b, implemented and unmeasured) while the FP8 Mamba2 projections still run on the host, and GGUF is unavailable. See the [Nemotron-3.5-Lightning model recipe](../models/nemotron-3-5-lightning.md) and [benchmark state](../BENCHMARKS.md). |

This is a deliberate state, not a bug: registering the architecture is what lets
the config parse and weight-name mapping be tested before the forward exists.

A refusal here is always a thrown message you can read. Every registered
architecture also refuses when it is handed a model some other architecture
loaded, naming both itself and the architecture the passed model claims, instead
of reading that model as though it were its own (#775, swept across the
remaining 34 entry points in #847). Where two architecture names share one
implementation, `Olmo2ForCausalLM` and `Olmo3ForCausalLM`, or
`LlamaForCausalLM` and `InternLM3ForCausalLM`, the refusal names the family's
primary architecture as the one that refused, and the alias you asked for as
what the passed model claimed.
