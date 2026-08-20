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
[project status](../STATUS.md) for pending run gates.

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

## Timing an encode on your own box

`tools/bench/bpe_encode_cost.cpp` times `Tokenizer::Encode` on one synthetic
input, at the sizes you name, through a `tokenizer.json` you name. Use it when
you want to know what a prompt of some shape costs to tokenize here, or to
re-derive a figure somebody else recorded instead of trusting it.

Nothing RUNS it: it is registered as no test and it is not a gate. Both halves
of that are deliberate, a growth ratio over these timings is not stable enough
to gate on a shared machine, and one leg on a long single-class input can cost
tens of seconds of one core. It IS compiled, as the never-linked OBJECT library
`vllm_bpe_encode_cost`, so it cannot rot behind a `Tokenizer::Encode` or
`FromHfJson` signature change while still being the artifact those figures are
reproducible from. Its header carries the exact `g++` and run lines; it builds
from the four tokenizer translation units directly and needs no `libvllm.a`.

It prints one row per case and size, with the ids it produced and the
1/5/15-minute load average sampled around each row, under a banner saying the
output is a session reading and not a bound. Read it that way: on a 20-core box
the same input on the same binary has read 1.7x apart on load alone, while the
id counts came back identical. Quote a number from it only with its load beside
it, and take the minimum of several repetitions rather than one shot.

## How much memory a Vulkan load needs

On a unified-memory device (a DGX Spark) the Vulkan heap and system RAM are the
same bytes, so budget roughly **the checkpoint size plus about 5%**, plus your KV
pool. Measured on GB10: Qwen3.6-27B bf16 (50.89 GiB on disk) peaks at 53.4 GiB of
process RSS. Reading the checkpoint also fills the page cache with about the file
size; that is reclaimable and does not need to be budgeted, but it does make
`MemFree` look alarming during a load. Use `MemAvailable`, not `MemFree`, to
decide whether a model fits. `VT_VULKAN_ALLOC_STATS=1` prints the running device
total and the `/proc` context if you need to see where it goes.
A Tenstorrent build (`-DVLLM_CPP_TENSTORRENT=ON`) needs TT-Metalium and TT-NN
on `CMAKE_PREFIX_PATH`. Blackhole currently runs OPT-125m through the shared
engine and has the Qwen3-0.6B correctness gate wired with device-specific
goldens. The full Qwen3 16x16 gate remains pending because paged attention is
still host-bound. This is an active correctness backend, not a performance
backend. See [the current project status](../STATUS.md) and the
[Tenstorrent backend spec](../../.agents/specs/tenstorrent-backend.md).

A Vulkan build (`-DVLLM_CPP_VULKAN=ON`) adds three kernel-measurement binaries.
They exist so a Vulkan tuning knob can be A/B'd in ONE binary, which is this
project's benchmark protocol, and each one prints WHICH kernel variant it ran so
a silent fallback cannot post a plausible number:

- `vulkan-gemm-ab`, cooperative-matrix versus the portable scalar GEMM
  (`VT_VULKAN_COOPMAT=0` picks the arm). Takes `M K N [reps]`.
- `vulkan-dispatch-floor`, one op swept across a 65,536x range of element counts,
  to separate per-dispatch overhead from real kernel cost.
- `vulkan-gemv-ab`, the decode GEMV swept over the (k, n) shapes a 27B model
  actually dispatches, with `VT_VULKAN_GEMV_ROWS` / `VT_VULKAN_GEMV_PACK` /
  `VT_VULKAN_GEMV_UNROLL` selecting the arm. Takes `[reps] [warmup] [GB/s roof]`
  and reports GB/s against that roof. Set `VT_VULKAN_DISPATCH_STATS=1` so it
  reports GPU-timestamp time rather than wall clock; see
  [the environment reference](../ENVIRONMENT.md) for what each knob does and what it measured.

  Audio note: the Voxtral/Whisper encoder attention has an opt-in FlashAttention-2
  tensor-core path, `VT_WHISPER_ENC_FA2=1`, which makes the encoder forward 5.50x
  faster, from 15.90x down to 2.89x vLLM's whole time-to-first-token. Those are
  encoder-forward-versus-TTFT ratios, not TTFT ratios: our projector, merge and
  prefill are not yet measured. It is off by default because it differs numerically
  from the shipping kernel and shifts three tokens within the ratified near-tie band
  on the gate clip, so turn it on only where encoder latency matters more than exact
  reproduction of the default output.

Every build, not only a Vulkan one, additionally gets `vocoder-conv-ab`, the
same-binary A/B for the shared 1-D BigVGAN vocoder convolution chain that
MiniMax-Music3, MiniMax-H3's audio VAE, LTX-2.5's audio VAE and IndexTTS-2.5 all
decode through. `VLLM_CPP_VOCODER_DEVICE` is the only variable, and the binary
prints the arm it RESOLVED rather than the one that was asked for, so a silent
fallback to the host cannot post a plausible pair of timings:

```sh
VLLM_CPP_VOCODER_DEVICE=cpu  ./build/vocoder-conv-ab --frames 96 --reps 3
VLLM_CPP_VOCODER_DEVICE=cuda ./build/vocoder-conv-ab --frames 96 --reps 3
```

It runs the four upsample stages at the shipped decoder's real channel counts and
strides, and prints a per-stage checksum so two arms that report the same time can
still be told apart if one of them computed something else. The transposed
convolution it times is 88.5 % of MiniMax-Music3's acoustic-half profile.
### Quantized checkpoints: which weight forms load
### How long a load takes, and how to see where it goes

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

A weight the device consumes verbatim is READ FROM the checkpoint mapping rather
than copied into a host buffer first, so it is moved once instead of twice; that
is `borrowed` above, and on this 27B it is 37.8% of the model and worth 1.54x on
the load phase warm, 1.61x cold. Tensors that are merged (`qkv`, `gate_up`),
transposed (`lm_head`) or dequantized at load are not verbatim and still copy.
`VT_LOAD_DIRECT_UPLOAD=0` turns the direct path off in the same binary; the
loaded bytes, and therefore the tokens, are identical either way.

Safetensors payloads are byte-addressed and do not promise natural scalar
alignment. Borrowed BF16/F16/F32 inputs therefore use defined byte-copy loads;
an odd payload offset neither forces a host copy nor changes the loaded bits.

`device_upload` counts every single-source weight upload: the bf16/fp8 weights
through `ResidentWeight` and the compressed-tensors NVFP4/MXFP4 `packed`/`scale`
residents through `ResidentNvfp4`. It does NOT yet count the merged fp4 operands
(`qkv`, `gate_up`) or the Marlin repack residents, which build one device buffer
out of several host tensors; on a bf16 checkpoint like the one above there are
none, so the line is the whole model. Once a weight has been uploaded its source
pages are released, and that release is independent of `VT_ADOPT_DEVICE_BYTES` --
switching the adoption off leaves the release on.

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

The two layouts this refuses in practice are per-output-channel FP8, which
stores one scale per output row, and block-wise FP8, which stores a grid. Both
used to load. The reader took the first four bytes and used them as the scale
of the whole matrix, which is a finite plausible number and therefore fluent
plausible wrong output rather than a failure. Issue
[#1181](https://github.com/mudler/vllm.cpp/issues/1181) has the detail, and the
per-output-channel arm itself is not implemented yet.

`lm_head` is not affected. It has always read a per-output-channel scale
correctly, as the table above records.

### A refusal that names the attention backend, and what it cannot tell you

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

The reason strings are vLLM's own, so a refusal here and a refusal from the
reference engine read the same. `head_size`, `block_size` and the KV-cache dtype
come from the geometry the engine has just resolved for your checkpoint, so a
refusal is about that checkpoint on this build.

**A device is only ever offered the backends built for it.** On CPU the engine
resolves `CPU_ATTN`, which is what the reference engine resolves on a CPU too. It
is worth saying out loud because it was briefly untrue: `CPU_ATTN` was named as
the CPU's preference while being registered nowhere, so CPU runs quietly fell
through to `FLASH_ATTN`, harmless until `FLASH_ATTN` was taught FlashAttention-2's
rule that a head size must be a multiple of 8. A CPU model with a head size of 6
then had no backend at all and was refused at initialization, on hardware that
runs it perfectly well ([#1371](https://github.com/mudler/vllm.cpp/issues/1371)).
If you see the refusal above naming `FLASH_ATTN` alone on a device that is not an
NVIDIA GPU, that is the shape to report: the rule quoted at you is about a kernel
your device never runs.

One consequence is worth stating on its own, because it widens what a CPU run
accepts. `CPU_ATTN` serves **`f32` as well as `f16` and `bf16`**, which is what
the reference engine's CPU backend serves. `FLASH_ATTN` declares the two half
dtypes only, so while the CPU was borrowing it an `f32` model was refused at
initialization with `dtype not supported`. It now runs. The KV-cache dtypes the
CPU accepts are `auto`, `fp8` and `fp8_e4m3`; `fp8_e5m2` is refused by name,
because the CPU kernel's fp8 arm reads e4m3 alone. On an NVIDIA GPU the list is
`auto`, `float16`, `bfloat16`, `fp8` and `fp8_e4m3`, so `fp8_e5m2` is refused
there too. That second refusal is the reference engine's own and is not
something this project trimmed away.

**What this check cannot tell you.** It reports what a backend *claims*, never
what your binary contains and never whether the kernel will launch. A backend
whose declared floor is compute capability 8.0 is accepted on any newer GPU, even
when the build carries no compiled code for that GPU.

That is a real failure mode, not a hypothetical one, and it surfaces as a launch
error rather than as the refusal above. It has been measured on a GB10 board
(compute capability 12,1) against the reference engine, same wheel and same
prompt: asking for its `FLASHINFER` backend generates text and exits cleanly,
while the default, which resolves `FLASH_ATTN`, the reference engine's *first*
preference for that device, dies at the first attention call with
`cudaErrorUnsupportedPtxVersion`. The first preference could not run and the
second could, and no capability check on either side could tell them apart.

So if a run dies inside attention rather than being refused before it starts,
the backend was accepted on a claim your build does not honour. Confirming which
architectures a build actually targets is a separate question, answered under
"Confirming which CUDA architecture a build targets" above. Tracked as
[#1332](https://github.com/mudler/vllm.cpp/issues/1332).

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
| `NemotronHForCausalLM` | **Only BATCHED decode still refuses.** A2-P (#810) narrowed this: `ForwardNemotronHForCausalLM` now selects the paged forward whenever the runner supplies paged KV and recurrent state, so K/V go into the runner's pages and the conv/SSM rows are carried across steps, and `examples/nemotron_h_gen` reaches all of it through `include/vllm.h` alone. What is left is `num_reqs > 1`, refused by name because one request's pages and one request's recurrent state are carried per step and a multi-request step would be decoded as ONE concatenated causal sequence, plausible wrong tokens rather than a failure. Owed to A2-B. **The end-to-end token gate against the pinned oracle has NOT run**, so no claim is made here about what this checkpoint emits; `docs/BENCHMARKS.md` records that as pending rather than as silence. `lm_head` and the FP8 Mamba2 projections still compute on the host, and a GGUF file is refused by name since no GGUF arm exists for it. See *Nemotron-3.5-Lightning-30B: the exact weights, and which arms run* below |

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
