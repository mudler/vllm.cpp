# Using vllm.cpp

The complete surface: the CLI, the OpenAI-compatible server, and the library
(C ABI and C++). The [README](../README.md) carries the quickstart; this page is
the reference behind it. Per-capability lifecycle state is
[docs/STATUS.md](STATUS.md); measured numbers are
[docs/BENCHMARKS.md](BENCHMARKS.md).

## Building

Full recipes are in [docs/BUILD.md](BUILD.md); the one rule worth stating here
is that the build must be **out-of-source**. Every command on this page assumes
a separate build directory:

```sh
cmake -S . -B build
cmake --build build -j
```

`cmake .` in the checkout is refused at configure time. It cannot work: the
example targets are named after the directories they are built from, so an
in-source build makes the linker write each executable over its own source
directory (issue #85).

### Setting the compiled build identity

`vllm-server --version` reports the CMake project version by default. Release
packaging passes the complete release identity, including any prerelease
component, with `-DVLLM_CPP_BUILD_VERSION=<version>`:

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_VERSION=0.0.3-pre.1
```

The value must not be empty. CUDA builds append their existing `+cuda`
qualifier to this identity. This option controls only the compiled binary
identity; release archives must still use the repository release workflow so
their manifest, `VERSION` record, archive name, and executable are validated as
one version.

### One ROCm-specific behaviour

ROCm builds register the full V1 sampler surface (temperature, top-k/top-p, min-p,
penalties, allowed-token masks, logprobs, random sample) so EngineCore does not
fatal with `no kernel for op` after prefill on AMD. Non-positive chat
`max_tokens` is treated as unset on all backends (Hermes `max_tokens=-1`).

Worth knowing before you read a hang as a bug in the tests: a build that sets no
`CMAKE_BUILD_TYPE` floors **HIP device code** at `-O1` and prints a configure
line saying so. At `-O0` the ROCm runtime starts a hostcall listener the kernels
never use, and its teardown can deadlock at process exit — every test passes,
`Status: SUCCESS!` prints, and the process never returns
([#132](https://github.com/mudler/vllm.cpp/issues/132)). Setting a build type,
or putting your own `-O` in `CMAKE_HIP_FLAGS`, overrides it.

### ROCm op coverage is incremental (and throws are by design)

The ROCm backend registers native ops family by family
([#41](https://github.com/mudler/vllm.cpp/issues/41)); landed GDN slices so far:
the indexed state I/O pair (`kGdnStateGather`/`kGdnStateScatter`), the causal
conv1d pair (`kCausalConv1dFwd`/`kCausalConv1dUpdate`, incl. the exact-chunks
descriptor form Qwen3.5 prefill passes), the fused post-conv glue
(`kGdnPostConv`), the gated-delta recurrence (`kGdnPrefill`/`kGdnDecode`,
portable scan), and the norm-gate/preamble ops (`kRmsNormGated`,
`kSigmoidGateBf16`, `kAttnQkNormRopeGate`) — the full set Qwen3.5-class
GDN-hybrid models call. Compressed conv/SSM state (bf16, the vLLM
`mamba_cache_dtype` default) is advertised via the
`SupportsCompressedConvState`/`SupportsCompressedGdnState` backend probes.
MoE-path coverage is partial: `MoeRouterTopK` (f32/bf16 logits, ungrouped
softmax, no bias) and `MoeSiluMul` are native; the remaining chain
(`kSharedExpertGate`, `kMoeCombine`/`kMoeCombineGate`, and the grouped quant
expert GEMM) is not registered yet, so MoE-bearing models still throw on
those ops. On a
discrete card there is no CPU fallback tier, so a model whose layers call an op
that is not registered yet fails loudly with `vt: no kernel for op N on device
type 5` — that is the memory-safety design working, not a crash. Run with
`VT_OP_PROVIDER_STATS=1` to see which ops resolve native.

### CUTLASS is fetched as headers only

`-DVLLM_CPP_CUTLASS_FETCH=ON` downloads CUTLASS v4.5.0 and stops there: the
sources are populated, but CUTLASS's own CMake project is never configured. Every
consumer in this tree `-isystem`s `${VLLM_CPP_CUTLASS_DIR}/include`, and nothing
links a CUTLASS CMake target, so its `tools/`, `library/`, `examples/` and
`tests/` targets are never built.

This is why no `-DCUTLASS_ENABLE_TOOLS=OFF` is needed. Configuring those targets
used to be required and could fail on its own — building for `sm_80` under CUDA
13 dies inside CUTLASS `tools/library` with duplicate `sm_100f` flags
([#193](https://github.com/mudler/vllm.cpp/issues/193)) — for a build product we
never used.

## Confirming which CUDA architecture a build targets

`CMakeCache.txt` is now a reliable answer. Configuring with
`-DVLLM_CPP_CUDA_ARCHITECTURES=<arch>` writes that value into
`CMAKE_CUDA_ARCHITECTURES` in the cache, so the two agree:

```sh
grep '^CMAKE_CUDA_ARCHITECTURES' build-cuda/CMakeCache.txt
```

Which fast paths a given architecture compiles is decided by the CUDA feature
table, not by the arch string alone. `110` (Jetson Thor) builds the portable
kernels plus the vendored Marlin NVFP4 W4A16 GEMM; the CUTLASS FP4/FP8 paths and
`fp4-mma` stay off there because no kernel body exists for it. `cmake -P
cmake/CudaArchFeaturesTest.cmake` prints the resolution for any target list
without a GPU or a CUDA toolkit.

It previously reported the toolkit's detected default (typically `75`) no matter
what was requested, because the project set the variable without writing it back
to the cache. Only the report was wrong — the emitted gencode always followed the
requested value — but it sent a contributor looking in the wrong place
([#168](https://github.com/mudler/vllm.cpp/issues/168)). The `build.ninja`
gencode line remains the ground truth if you want to double-check.

## Using more than one engine in a process

Constructing a `LoadedEngine`, destroying it, and constructing another in the
same process is supported, including on CUDA. Each engine's device-resident MoE
and Marlin constants are owned by the weights they describe and are released
with them.

Before, that state lived in process-lifetime caches keyed on the *address* of a
weights block, so a second engine could land on a freed block's address and
reuse device pointers that had already been freed. Nothing crashed — the CUDA
context is never torn down, so the pointers stayed mapped — it simply produced
corrupted or zeroed output tokens, intermittently
([#237](https://github.com/mudler/vllm.cpp/issues/237)).

More than one **backend** in one process is likewise supported — a CPU forward
running beside a CUDA one, which is what a diffusion pipeline with a host-side
stage does. Until
[#516](https://github.com/mudler/vllm.cpp/issues/516) it was not: the shared
device-scratch pool was a single process-wide free list keyed by byte size class
with no device in the key, so a block allocated through one backend was handed
to the next caller of that size class on another. It has two symptoms and the
direction picks which: a `cudaMalloc` block reaching a CPU forward segfaults in
the host `memcpy`, and a host block reaching a CUDA forward produces output that
is uniformly NaN rather than wrong. Neither can happen now — a scratch pool is
bound to one backend and refuses any other with a `std::logic_error` naming both
— and no user-facing flag or env var selects the behaviour: it is unconditional.

One consequence is worth knowing before you add a backend. The scratch pool's
residency cap now comes from *that device's* platform rather than from whichever
device resolved first, so constructing a buffer on a backend whose platform was
never registered raises instead of silently inheriting another platform's cap. A
cap read off the wrong platform is a wrong number, not a default, and every
backend the tree ships registers one.

`VT_POOL_BYPASS=1` and `VT_POOL_EXACT=1` keep exactly the meanings
[ENVIRONMENT.md](ENVIRONMENT.md) records for them. They are debugging lanes, not
timing configurations, and the pool's own test suite is green under both, so
either one stays usable as a discriminator when something else is under
suspicion.

## Starting an agent-assisted contribution

Run `scripts/agent-start.py` first. It reports an inherited worktree role or,
for a new contributor with no declared role or explicit intent, prints the
welcome that the agent should relay. An explicit request can use
`--intent operator|helper|read-only` and a helper `--row ID`. Follow its printed
claim action, rerun it after declaration, then run `scripts/agent-preflight.sh`.
The entrypoint is non-interactive and does not mutate the checkout.

The operator role is a coordinator, and **several may run at once**:
`scripts/agent-role.py claim operator` records this worktree and is never
refused, `scripts/agent-role.py show` lists the other live coordinators, and
`scripts/agent-role.py release` removes only this worktree's record. What keeps
concurrent coordinators safe is that `main` is never force-pushed, so a plain
`git push` refuses any non-fast-forward.

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../examples/cli/main.cpp).

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
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
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
is expected to load. The **MoE** loader reads only PER-EXPERT NVFP4 routed
experts, while the published MoE repos (`Qwen/Qwen3.8-2.4T-A95B`,
`Qwen/Qwen3.6-35B-A3B`) ship 3-D stacked, unquantized experts — that arm is
**not implemented**, and such a checkpoint is refused at load with a message
naming what is missing. Use an NVFP4 requant (e.g.
`nvidia/Qwen3.6-35B-A3B-NVFP4`) for the MoE path. No text-only Qwen3.5
checkpoint has been RUN here at all — see [STATUS.md](STATUS.md) for the owed
run gates.

GGUF and safetensors mapped-payload paths, plus safetensors index paths, use the
host's native filesystem encoding, including Unicode paths on Windows. Native
Windows release artifacts are not published yet; they will remain unavailable
until the `v0.0.3-pre.1` prerelease build and publication gates succeed.

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`. It pretokenizes before timing
  and atomically publishes each concurrency wave. Set
  `VT_BENCH_PRETOKENIZE=0` for the timed-string rollback; the report names the
  resolved mode.
- `tokenize` ([`examples/tokenize/main.cpp`](../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.
  GGUF `tokenizer.ggml.pre` names accepted: `qwen35`, `qwen2`, `llama-bpe`,
  `gpt-4o` / `llama4` / `kanana2` / `talkie` (the GPT-4o / o200k family),
  `joyai-llm`, `deepseek-llm`, `deepseek-v3`, `laguna`. Any other name is
  refused by name rather than aliased onto a near-miss regex.

### Which HF tokenizers load

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

### How much memory a Vulkan load needs

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
backend. See [STATUS.md](STATUS.md) and the
[Tenstorrent backend spec](../.agents/specs/tenstorrent-backend.md).

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
  [ENVIRONMENT.md](ENVIRONMENT.md) for what each knob does and what it measured.

  Audio note: the Voxtral/Whisper encoder attention has an opt-in FlashAttention-2
  tensor-core path, `VT_WHISPER_ENC_FA2=1`, which makes the encoder forward 5.50x
  faster — from 15.90x down to 2.89x vLLM's whole time-to-first-token. Those are
  encoder-forward-versus-TTFT ratios, not TTFT ratios: our projector, merge and
  prefill are not yet measured. It is off by default because it differs numerically
  from the shipping kernel and shifts three tokens within the ratified near-tie band
  on the gate clip, so turn it on only where encoder latency matters more than exact
  reproduction of the default output.

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

### Quantized checkpoints: which `lm_head` forms load

Publishers do not agree on how weights are stored, and a single repo can change
it between revisions (one 27B "NVFP4" repo silently became FP8 throughout).
The table below is about `lm_head`; the same three forms are accepted for the
attention, MLP and `linear_attn` projections, in both compressed-tensors
(`weight_packed` + `weight_global_scale`) and ModelOpt (`weight` +
`weight_scale_2`) naming. For the Qwen3.6 dense family we accept all three
forms in use, so pick a checkpoint by its quality, not by its head:

| `lm_head.weight` | Companion tensors | Seen in |
|---|---|---|
| `BF16` | none | `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` |
| `F8_E4M3` | `lm_head.weight_scale` (per-output-channel or per-tensor) | `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` |
| `U8` NVFP4 | `lm_head.weight_scale` + `weight_scale_2` (ModelOpt) or `weight_global_scale` (compressed-tensors) | `nvidia/Qwen3.6-27B-NVFP4` |

The head is dequantized to BF16 at load, so all three cost the same memory once
running. Any other dtype fails at load with a message naming what it saw.

A `modelopt_mixed` checkpoint (`nvidia/Qwen3.6-27B-NVFP4`, and the 35B-A3B that
shares the tower) keeps its `linear_attn` input projections in FP8 W8A8, and
those two per-layer projections are packed into ONE merged `in_proj_qkvz` GEMM,
mirroring vLLM's `MergedColumnParallelLinear`. The merge only fires when the two
shards carry a bitwise-identical per-tensor `input_scale`, since one GEMM
quantizes the activation once; a checkpoint whose scales differ keeps the two
separate GEMMs automatically. `VT_GDN_MERGED_QKVZ_FP8=0` restores the two GEMMs
in the same binary.

### Architectures that resolve but refuse to run

A few architectures are registered so their config and weight layout are
accounted for, while their forward is deliberately not implemented. Pointing the
CLI or server at one of these loads far enough to resolve the architecture and
then fails with a message naming the missing piece, rather than emitting wrong
tokens quietly.

| Architecture | Why it refuses |
|---|---|
| `KimiK3ForConditionalGeneration` | Needs ~1.56 TB (MXFP4); no host here can run it |
| `NemotronHForCausalLM` | The hybrid forward is ported (#517 W4) but there is no weight LOADER yet, so a checkpoint still cannot be run: loading leaves the weights unmaterialized and the forward refuses by name. Safetensors resolve and parse; a GGUF file is refused by name, since no GGUF arm exists for it |

This is a deliberate state, not a bug: registering the architecture is what lets
the config parse and weight-name mapping be tested before the forward exists.

### LTX-2.5: what runs, and what it cannot do

LTX-2.5 is reachable as video family `ltx-2.5`, through the same
`vllm_video_engine_load` / `vllm_video_generate` C ABI that serves MiniMax-H3,
and through the `ltx2-gen` example that drives it. Its two VAE decoders, its two
VAE ENCODERS with the mel front-end, the conditioning items that place encoded
latents into the token stream, and its pipeline layer (the sigma schedule, the
diffusion steps, guidance, the latent spatial x2 upsampler, the duration head and
the embeddings connector) are implemented and gated. Several limits decide what
you can actually ask for, and each refuses by name rather than rendering
something else.

In particular, the encoders being present does NOT mean image, keyframe,
reference-video or reference-audio conditioning is usable: the video engine
still refuses every one of those by name, because the request-side work between
a file on disk and a tensor the encoder accepts — image decode, aspect-fill
resize, and the H.264 CRF re-compression upstream performs before encoding
whenever the resolved CRF is not `0` and the image is at least 2 pixels on its
shorter side — is not ported. The engine also holds no
encoder to call: it materializes the VAE DECODER key filters only, so no
encoder weights are ever in memory, and the refusal names that rather than
claiming the encoder itself is missing. Two encoder-level limits are worth
stating in advance because they are refusals rather than approximations. A
reference waveform whose sample rate differs from the audio VAE's is refused
rather than resampled, since upstream uses a polyphase kaiser resampler this
project does not carry. And a VAE configured with `latent_log_var: none` is
refused, because upstream itself raises on it.

**A typed prompt works.** `--encoder` names the Gemma-4 12B text tower and
`--prompt` carries the words. The tower tokenizes them with its OWN embedded
tokenizer — the shipped encoder stores `tokenizer.json` as a TENSOR, so there is
no sibling file to point at — runs, aggregates all 49 hidden states, projects
them to 4096 and 2048, and passes both streams through the embeddings connector
before cross-attention. The tower is ~24 GB of host bf16 and stays resident,
because a prompt arrives per request.

One tokenization detail is a KNOWN DIVERGENCE rather than a mirror, and it is
checkpoint-conditional: upstream tokenizes through the HuggingFace `__call__`
with its default `add_special_tokens=True`, so it runs the tokenizer's
post_processor, while this port calls the plain encode and prepends BOS by hand.
On the shipped checkpoint the two are identical — its post_processor declares an
EMPTY special-token map, measured on the shipped file rather than assumed — so
nothing is lost today. A checkpoint whose post_processor DID add tokens would
tokenize differently here.

`--encoder-config` supplies the Gemma config, and it is required for the only
shipped encoder: `vonkaiser`'s
`gemma4-12b-with-proj-nvfp4-torchao.safetensors` carries no `__metadata__` at
all. An encoder that declares one (the official bf16 build does, under
`__metadata__["gemma_config"]`) needs no flag, and supplying both is refused
rather than resolved — `layer_types`, `global_head_dim`,
`num_global_key_value_heads` and `attention_k_eq_v` each resolve a different
tower out of a byte-identical tensor set.

Without `--encoder`, conditioning comes from `--prompt-embeds` plus
`--audio-prompt-embeds`: rows of little-endian f32, 4096 wide for the video
stream and 2048 for the audio stream, with the same row count in both. A
`--prompt` with no tower is refused, and supplying only one of the two files is
refused, because a stream left unconditioned renders instead of failing.

**Asking what a clip was conditioned on.** `Ltx2VideoEngine::last_conditioning()`
returns the trace of the last `Generate()` — whether the conditioning came from a
prompt or from embeds, the prompt string, the row count and both stream widths, an
FNV-1a digest over the exact f32 buffers cross-attention read, and each stream's
absmax. It is returned **by value, under the engine's own lock**, so it is safe to
call from a server thread while another thread renders — but `Generate` holds that
same lock for the WHOLE render, so such a call blocks for minutes rather than
returning a stale answer immediately. `completed` is true only if that
`Generate()` returned: the trace is filled before the denoise loop, so a
render that throws later leaves a populated trace behind, and this flag is what
separates the two.

It is a **change detector, not a quality measure**. It answers "did this render
depend on this prompt, through these weights" and nothing else — it does not say
the conditioning values are the ones upstream would produce.

The text path runs on the CPU even when `--device cuda` puts the DiT on the GPU:
everything in the text encoder is f32 by declaration and its device arm is owed.
That is one host-side 12B forward over the prompt's own tokens per request,
against a denoise loop of many 21B forwards.

**Either source goes through the embeddings connector.** Both shipped LTX-2.5 DiTs
carry two `*_embeddings_connector` families, 129 tensors each, and they are the
8-layer 1-D transformer upstream runs between the caption projections and the
DiT's cross-attention. The render applies it with the checkpoint's own weights,
under the checkpoint's own `connector_*` configuration. Two consequences for the
command line: the row count must be a multiple of the connector's learnable
register count (128 on the shipped files), and `--prompt-valid-rows N` says how
many of those rows are real tokens. The rest are padding, and padding is not
inert here: the connector REPLACES it with its learnable register table, so a
run that leaves the default renders as if every supplied row were caption.
`--prompt-valid-rows` applies to the embeds path only — with `--encoder` the
tokenizer supplies the mask, which is what that flag exists to stand in for.

**The DiT config is required when the checkpoint does not carry one.** The
shipped `vonkaiser` FP8 transformer has no `__metadata__` at all, and the values
a config decides are ones no tensor shape encodes: `frequencies_precision` and
`av_ca_timestep_scale_multiplier` move every RoPE angle and every audio/video
modulation. Defaulting them resolves a different model from the same file, so
the loader refuses and `--dit-config` supplies LTX-2.5's declared values.

```sh
ltx2-gen --dit  ltx-2.5-22b-distilled-fp8.safetensors \
         --dit-config ltx-2.5-transformer-config.json \
         --model-version 2.5 --allow-unported \
         --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
         --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
         --upsampler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
         --encoder gemma4-12b-with-proj-nvfp4-torchao.safetensors \
         --encoder-config ltx-2.5-gemma4-text-config.json \
         --prompt "a red fox running through deep snow at sunrise" \
         --frames 25 --width 320 --height 192 --seed 20260812 \
         --device cuda --workdir /tmp/ltx25 --out /tmp/ltx25/video.mp4
```

Swap the two `--encoder*` flags and `--prompt` for `--prompt-embeds` +
`--audio-prompt-embeds` to condition from files instead.

`--frames` must satisfy `(frames - 1) % 8 == 0` and width/height must divide by
64 (32 for the VAE, twice that because the distilled recipe's first phase runs at
half resolution). Omitting all three renders the recipe default, which is
1024x1536 at 121 frames and is a much larger request than it looks.

`--upsampler` is what the distilled recipe's second phase needs. Without it that
phase refuses rather than skipping: its three-step refinement is what makes the
upscaled latent valid, and decoding the half-resolution latent instead would hand
back a smaller clip that looks like a completed request. `--max-phase 0` stops
after the first phase deliberately.

On the server, `--video-family ltx-2.5` pins the family instead of detecting it,
and `--video-extra KEY=VALUE` (repeatable) carries the same family-specific load
knobs the flags above map onto. Both are described under
[the server's video flags](#video-family-and-family-specific-load-knobs).

**Three things about that command are worth knowing before you run it.**

*It is bounded by the VIDEO DECODE, well below the recipe's own defaults.*
Staging the 21.00B FP8 transformer costs about 44 GB on a 119 GB GB10, and
`--encoder` adds the text tower on top of that — roughly 24 GB of host bf16 that
stays resident, because a prompt arrives per request. Every memory figure here
was measured WITHOUT the tower, on the prompt-embeds path, so budget for both.
**320x192 at 25 frames completes** through both distilled phases; 448x256 at 25
frames finishes its denoise and then loses about 59 GB in 24 seconds inside the decode
and has to be stopped. The denoise itself is flat at either size. Unified memory
makes those host bytes and this class of box reboots rather than OOM-killing, so
start small and grow, and put a memory watchdog in front of anything larger. The
recipe default (1024x1536 at 121 frames) is far beyond what one GB10 holds today.
Expect minutes, not seconds: most of a 320x192/25f render is spent single-threaded
in the host VAE decode at 0% GPU.

*The render behind those numbers was NOT prompted, and it renders a scene without
rendering YOUR scene.* It was the EMBEDS path — `--prompt-embeds` with
`--prompt-valid-rows 24`, over synthetic N(0, 0.2) rows, with no text tower on the
path at all. With the connector wired the shipped 21.00B FP8 transformer produced
a temporally coherent photorealistic clip at 320x192 / 25 frames: consistent
subject, consistent background, frame-to-frame motion, where before the connector
the same weights at the same settings produced smooth colour fields. But 104 of
its 128 connector rows were the connector's own trained `learnable_registers`
table, which is what upstream substitutes at PADDED positions, and the other 24
were noise. So what conditioned that clip is the checkpoint's own learned default,
not a depiction of anything anyone asked for — and on the embeds path it could not
be otherwise, because rows read from a file are whatever you put in them rather
than an encoded caption. Ask a `--prompt-embeds` run for a subject and you will
not get it.

*Nobody has yet run the command above end to end, and this page claims nothing
about what it renders.* The typed-prompt path is gated all the way through —
tokenizer, Gemma-4 tower, connector, cross-attention — but the gate is a
REDUCED-DIMENSION synthetic encoder under CPU Release, with no real checkpoint
anywhere in it. A real-checkpoint prompted render is OWED. Until it runs, neither
claim is available: not that `--prompt "a red fox…"` puts a fox on the screen, and
not that it fails to. `last_conditioning()` answers a narrower question — that the
render depended on your prompt, through these weights — which is not the same
question as whether the frames depict it.

LTX-2.5 ships two video decoders behind one checkpoint field. The convolutional
one is implemented; the higher quality diffusion one (`NADiffusionDecoder`) is
not, and asking for it fails with a message naming the missing
neighborhood-attention kernel. It never falls back to the convolutional decoder,
because that would hand back a lower quality render as if it were the one you
asked for. Keyframe and reference conditioning is refused for the same reason: it
runs through the video VAE's encoder, and only the decoder is ported.

**The refusal that used to stand here is gone, and what replaced it is an owed
ORACLE rather than an owed feature.** Through L10 this page said a prompt was
refused because the `Embeddings1DConnector` weights, which ship inside the DiT
file, were among the modules the DiT loader would not load. They are loaded
(`ltx2_loader.cpp:416-427` carries them as their own contract, outside the DiT's),
so `encoder_path` is accepted, `has_encoder()` is true, and a prompt no longer
needs a matching pair of embeds files. The gap that remains is a numeric one: the
tower, the connector's forward and both caption projections each have an oracle
against executed upstream, and the two JOINS between them —
`create_embeddings`, and the render composition that chains it onto the tower's
output — have none. Upstream's `EmbeddingsProcessor.process_hidden_states` is
that whole chain in one function and is the oracle this owes; until it is
executed, the composition's VALUES rest on the per-brick oracles either side of
it. That is also why `last_conditioning()` is described above as a change
detector and not as a check on the conditioning.
### GDN checkpoints: the `output_gate_type` key

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

### Muse Glimmer: exactly what has been checked

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
  defaults** — see [FEATURES.md](FEATURES.md) and
  [the spec](../.agents/specs/muse-glimmer.md) §6.7.

## OpenAI-compatible server

`vllm-server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../src/vllm/entrypoints/openai/).

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

The install component and deterministic archive target both stage from install
rules rather than copying the build tree:

```sh
cmake --build build --target vllm-server-stage
cmake --build build --target vllm-server-archive
build/release/stage/bin/vllm-server --help
```

At the current numeric project version, `vllm-server-archive` emits exactly one
deterministic developer tarball named
`build/release/vllm.cpp-0.0.3-<configured-artifact-id>.tar.gz`. The target
selects `tar.gz` explicitly; it does not infer the format from the filename.
This is separate from the release workflow, whose `0.0.3-pre.1` asset names and
per-tuple formats come from the release matrix, including `.zip` for Windows.

On native Windows, run the release-bundle gate from a Visual Studio 2022 x64
developer PowerShell. It builds with MSVC/UCRT `/MT` and `/W4 /WX`, installs
`bin/vllm-server.exe`, runs the focused Win32 tests, exercises the portable and
AVX2 tiers, verifies an unsupported forced tier is refused, and smokes
`--help`, `/health`, `/version`, and a clean CTRL_BREAK shutdown:

The MSVC build defines `NOMINMAX` and the portable ISO CRT contract centrally,
and compiles C++ sources as UTF-8. Do not add those definitions per target or
disable `/WX`; both CPU and Vulkan release configurations share this contract.

```powershell
$env:SOURCE_SHA = git rev-parse HEAD
$env:VERSION = "0.0.3-pre.1"
$env:SOURCE_DATE_EPOCH = git show -s --format=%ct HEAD
$env:EVIDENCE_URL = "https://github.com/mudler/vllm.cpp/actions/runs/EXAMPLE"
pwsh -File scripts/build-windows-release.ps1 -Backend cpu
pwsh -File scripts/build-windows-release.ps1 -Backend vulkan `
  -BuildDir build-release-windows-vulkan `
  -StageDir build-release-windows-vulkan/stage
```

The adaptive binary keeps its F16C translation unit at `/arch:AVX`; AVX2 and
AVX-512 remain separate runtime-selected translation units. The gate derives
the complete server source set from CMake's generated codemodel, recursively
checks its project-local header closure, and refuses required runtime sources
that are not reachable from the shipped target. After installation it audits
project COFF directives for static `LIBCMT` and rejects dynamic/debug CRT
imports before running the staged executable's `--help`, forced-tier, or HTTP
shutdown smokes. The Win32 console-control regression uses bounded waits so a
teardown failure reports an error instead of hanging the gate.

The CUDA graph-replay profiler and its FIFO diagnostic controls remain
POSIX-only and are not exposed by native Windows server builds. Native Windows
process launch, environment updates, process IDs, and console shutdown stay on
the direct CRT/Win32 adapters; they do not require a POSIX compatibility layer
or a command shell.

Each invocation emits a deterministic `.zip` plus its exact `.sha256` and
`.provenance.json` sidecars. ZIP members are sorted, use the
`SOURCE_DATE_EPOCH` timestamp, and reject traversal, drive-qualified paths,
backslashes, symlinks, and reparse points. The PE audit requires AMD64, `/MT`,
system DLL imports, and no build/debug/MSYS paths. The Vulkan archive bundles no
loader, ICD, or driver: `vulkan-1.dll` and a working host Vulkan stack remain
external, and runtime evidence stays absent unless the extracted server is
actually probed against a real ICD.

The default smoke model is the committed tiny embedding fixture; pass
`-SmokeModel C:\path\to\model` to use another complete model directory. This
command produces a staged developer tree only. The Windows CPU and Vulkan ZIP
downloads do not exist until the `v0.0.3-pre.1` prerelease workflow and
post-publication audit succeed. <!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

The basic CMake archive under `build/release/` includes the version, configured
backend, OS, and host architecture in its name. It is a developer package. The
release workflow separately produces host-ABI-specific archives with a
manifest, `VERSION`, SPDX SBOM, notices, licenses, and detached checksum and
provenance sidecars; no release download is claimed until that workflow has
completed on a release tag.

To reproduce the W1 heterogeneous CUDA archive candidate, configure the exact
release architecture set. Portable translation units compile for all ten SMs;
architecture-specific kernels compile only for their supported intersection.
`VLLM_CPP_TRITON` is left to its default, which is `ON` here — a fat CUDA build
embeds every vendored per-arch cubin tree and selects one by exact SM at
runtime, which is what the released archive contains:

```sh
cmake -S . -B build-cuda-fat -G Ninja \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES='80;86;87;89;90a;100a;103a;110;120a;121a' \
  -DVLLM_CPP_CUTLASS_FETCH=ON
cmake --build build-cuda-fat --target vllm
python3 scripts/check-cuda-fat-gencode.py \
  --compile-commands build-cuda-fat/compile_commands.json \
  --library build-cuda-fat/libvllm.a
```

The release workflow applies this audit to independently linked x86_64 and
arm64 host executables, packages each as a preview `cuda` archive, and then
runs the extracted-archive validator. Each archive must contain all ten SM
images and the six available exact-SM Triton AOT namespaces; the manifest keeps
runtime evidence separate per SM. These build-only preview candidates are not
a downloadable release claim until the tagged workflow publishes them.

The complete primary download matrix and its runtime boundaries are documented
in [RELEASES.md](RELEASES.md). A manual workflow dispatch runs all eight tuples
without publication. An exact version tag runs the same build, produces
`release-index.json` and `RELEASE_INDEX.md` from the verified archive manifests,
attests the archive bytes, and publishes every archive/checksum/provenance
triplet through the protected release environment.

Inside the workflow, generated archives live under `release-assets` (and then
`unverified/release-assets` / `verified/release-assets`). This transient root is
deliberately separate from the checkout's tracked `assets/` directory, so exact
handoff validation sees only the planned archive/checksum/provenance triplets.
The release filenames and published eight-tuple inventory are unchanged.

### Selecting an x86 CPU ISA tier

The x86_64 CPU library is one adaptive binary: portable, SSE2,
SSE2+F16C, AVX2, and AVX-512 elementwise matmul kernels are isolated in their
own translation units and selected only after CPUID plus the required XCR0 OS
state are checked. Leave `VT_CPU_MATMUL_TIER` unset for automatic selection, or
set it to `portable`, `sse2`, `sse2+f16c`, `avx2`, or `avx512` for a same-binary
correctness/performance check. A forced tier that the current CPU or OS cannot
execute fails closed instead of silently narrowing or risking an illegal
instruction. Release builds never use `-march=native`.

On arm64, leave the same variable unset to select between portable and NEON
elementwise matmul, or force `portable`/`neon`. DotProd and i8mm kernels are
independently selectable with `VT_CPU_Q8_DOT`, `VT_CPU_QUANT_MMLA`, and
`VT_CPU_QUANT_REPACK`; `auto` uses Linux HWCAP/HWCAP2 or Darwin feature sysctls,
while an unavailable forced tier fails closed. The exact accepted values are
listed in [ENVIRONMENT.md](ENVIRONMENT.md).

### NVFP4 dense sinks

The `E=1` dense NVFP4 projections run on vLLM's own dense Marlin GEMM rather
than the single-expert grouped-MoE route, which pays `moe_align` bookkeeping and
row padding for a problem that has neither. `VT_MARLIN_DENSE` covers the single
projections and `VT_MARLIN_DENSE_PAIR` the fused shared-expert gate_up sink;
both default ON, opt out with `=0`. The pair sink was the last one still on the
MoE route: enabling it measured **+1.31% at c8 and +1.38% at c4** on
`nvidia/Qwen3.6-35B-A3B-NVFP4` with both SACRED gates unmoved. Only the
throughput changes; the routed experts still use the grouped MoE kernel, which
is where they belong.

The **dense** MLP's W4A16 gate/up pair takes that same fused gate_up GEMM
(`VT_DENSE_MARLIN_GATEUP`, **default ON**, opt out with `=0`). vLLM's dense
Qwen3.6 MLP is one `MergedColumnParallelLinear` `gate_up_proj`, so one
`[T,H]x[2I,H]` GEMM per layer is the mirrored topology; ours used to launch two,
which was 193 Marlin calls per decode step against the oracle's 129. The default
moved on a same-binary A/B: interleaved 4 reps per arm on
`nvidia/Qwen3.6-27B-NVFP4`@`0893e160` (GB10) with the toggle as the only
variable measured **+2.12% at c1 and +1.70% at c8**, every fused rep beating
every split rep at both concurrencies, and the 64-token greedy continuation
identical on both arms. It is still only ~29% of a measured +4.40 ms/step gap on
the 27B and does not reach parity on its own. It applies only to an **NVFP4**
W4A16 pair whose two shards share a global scale; a true-W4A4 checkpoint already
takes the merged CUTLASS path instead, and a **dense MXFP4** pair is refused and
keeps the split pair. That MXFP4 refusal is deliberate: the fused entry point the
dense MLP reaches is NVFP4-only — it sizes the merged block-scale grid at K/16
and pins `group_size = 16` — so admitting group-32 E8M0 scales would misread them
as group-16 fp8-e4m3, the defect this project already recorded for the sibling
implementation. No dense loader produces MXFP4 today, so the refusal changes no
shipped configuration; it stops one future loader line from silently selecting a
mis-scaled kernel.

The shared expert's `down_proj` keeps its bf16 output rather than upcasting to
f32 (`VT_SHARED_DOWN_BF16`, default ON, opt out with `=0`). Both consumers widen
bf16 in-kernel — which is exact — and re-round through bf16 on store, so the
f32 form was writing and re-reading a whole `[T,H]` buffer for a value it
already had. The change is bit-identical and worth **+2.05% at c8**.

### The NVFP4 output head

On a Qwen3.6 dense checkpoint whose `lm_head` is stored NVFP4 (ModelOpt
`weight`/`weight_scale`/`weight_scale_2`, or compressed-tensors
`weight_packed`/`weight_global_scale`) the head is kept **packed** and the logits
GEMM runs on it directly, as vLLM does. Nothing is dequantized at load, so the
head costs `K*N/2 + K*N/16` bytes instead of `2*K*N`, about 0.715 GB instead of
2.543 GB on `nvidia/Qwen3.6-27B-NVFP4` (measured peak host RSS 21.06 to 19.36
GiB, a 1.70 GiB saving on CUDA; the figure is owed a re-measurement after
`ENG-LOAD-DIRECT-UPLOAD` changed the RSS accounting).

That accounting is CUDA's. A backend with no fp4 GEMM (CPU, Vulkan, Metal, HIP,
Tenstorrent) has to multiply against a dequantized bf16 copy, so on those the
head costs the packed bytes **plus** one `2*K*N` operand, built once when the
model is prepared rather than per call — 0.666 + 2.368 = 3.034 GiB on the same
checkpoint. The sign of the change therefore depends on the backend: on Vulkan,
which used to stage a host bf16 head *and* a device copy of it, the head goes
4.736 to 3.034 GiB, the same **-1.70 GiB**; on plain CPU it goes 2.368 to 3.034,
a **+0.67 GiB** regression, paid once instead of rebuilding 2.368 GiB on every
decode step as that backend did before. Only the head is kept that way; every
other NVFP4 projection dequantizes per call, so a quantized tower is never
expanded in memory. The head runs W4A16 under both namings: the on-disk
activation divisor next to it (`input_scale`, or `input_global_scale` in the
compressed-tensors spelling) is NOT consumed unless `VT_MODELOPT_W4A4=1`,
matching vLLM, which deletes it on this path. Set `VT_LMHEAD_FP4=0` for a
same-binary A/B that restores the old dequantize-at-load owner. BF16, FP8, GGUF
and `tie_word_embeddings` heads are unaffected by either setting.

### Validating a staged release archive

Release verification reads only a freshly extracted archive, never files from
the build tree. Pass the archive together with its final-byte SHA256 and SLSA
provenance sidecars:

```sh
python3 scripts/validate-release-archive.py \
  --archive vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz \
  --archive-format tar.gz \
  --checksum vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz.sha256 \
  --provenance vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz.provenance.json \
  --forbid-path "$PWD/build"
```

The validator checks the content allowlist, executable and host ABI, manifest,
`VERSION`, SPDX SBOM, licenses, ELF dependencies and RPATH/RUNPATH, extracted
`--help`/`--version` smokes, and backend-specific CUDA or adaptive-CPU claims.
The digest and provenance are sidecars because both describe the final archive
bytes; placing either inside those bytes would create a self-reference.

The CPU release helper is the reproducible entry point used by CI. It requires
an explicit artifact tuple, architecture, channel, build directory, libc ABI,
a feature-poor QEMU userspace emulator, and a feature-rich runner. x86_64 uses
the SHA256-pinned Intel SDE installed by `scripts/install-intel-sde.sh` so the
AVX-512 tier is really executed even when the host lacks AVX-512. The gate then
executes the baseline and proves rich-tier refusal under the feature-poor QEMU
model before metadata can be generated:

```sh
SOURCE_SHA=$(git rev-parse HEAD) \
VERSION=0.0.2 \
SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD) \
EVIDENCE_URL=https://github.com/mudler/vllm.cpp/actions/runs/EXAMPLE \
scripts/build-cpu-release.sh \
  linux-x86_64-glibc-cpu x86_64 stable build-release-cpu-x86 \
  2.39 /usr/bin/qemu-x86_64 /tmp/intel-sde/sde64
```

The corresponding arm64 tuple is `linux-aarch64-glibc-cpu`. The only literal
static tuple is the CPU-only `linux-x86_64-musl-cpu-static` experiment; normal
CPU and accelerator archives are static-core bundles with audited host runtime
dependencies.

## Container images

Published to one GHCR package with the lane in the tag. Every lane is a
`linux/amd64` + `linux/arm64` manifest, so the same tag works on both.

| tag | what it is |
|---|---|
| `:<version>-cuda` / `-vulkan` / `-cpu` | **immutable.** Never republished |
| `:latest-cuda` / `-vulkan` / `-cpu` | moves to the newest **release** |
| `:latest` | the **cpu** lane, so pulling it on a machine with no accelerator gets a working server rather than a library-load failure |
| `:main-cuda` / `-vulkan` / `-cpu` | moves with **main**: rebuilt when container infrastructure changes and nightly otherwise. Convenience, not a release — no support claim |

The entrypoint is `vllm-server`, so flags go straight after the image name and
the server keeps its own default of `0.0.0.0:8000`:

```sh
docker run --rm -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest \
  --model /models/Qwen3.6-35B-A3B
```

For the CUDA lane, the GPU driver comes from the host through the container
runtime; the image carries only the CUDA *runtime* libraries it links:

```sh
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda \
  --model /models/Qwen3.6-35B-A3B
```

`/models` is the weights mount and `/cache` is the tokenizer/HF cache. The
container runs as **uid 1000**, so `/cache` must be writable by it and the
weights under `/models` must be READABLE by it. A model file with mode `0600`
owned by another uid fails as `safetensors: cannot open file`, which reads like
a corrupt checkpoint rather than a permissions problem.

### Picking the right flags for your GPU

The two NVIDIA families need **different** invocations, and this is verified on
both rather than inferred:

| host | verified on | flags |
|---|---|---|
| SBSA / datacenter arm64, x86_64 | GB10 `sm_121a` | `--gpus all` |
| Jetson / Tegra (L4T) | AGX Orin `sm_87`, L4T R36.4.3 | `--runtime nvidia --gpus all` |

On Jetson, `--gpus all` **alone is refused** ("invoking the NVIDIA Container
Runtime Hook directly ... is not supported"), and `--runtime nvidia` **alone**
starts a container with no driver that dies on `libcuda.so.1: cannot open
shared object file` — which looks like a broken image rather than a missing
flag. Use both:

```sh
docker run --rm --runtime nvidia --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda \
  --model /models/Qwen3-0.6B
```

That exact recipe was run on an AGX Orin with `Qwen/Qwen3-0.6B`: the server
serves `/v1/completions` and `tegrastats` shows `GR3D_FREQ` at 95-97% during
generation, so decode is on the GPU.

### If the server exits at startup

| symptom | cause |
|---|---|
| `safetensors: cannot open file` | the weights are not readable by **uid 1000**. The container runs as uid 1000; a `0600` model owned by another user fails here and looks like a corrupt checkpoint |
| `libcuda.so.1: cannot open shared object file` | no driver in the container — on Jetson, add `--gpus all` alongside `--runtime nvidia` |
| `--model <dir> is required` | the server takes flags directly; everything after the image name goes to `vllm-server` |

### Building and validating an image locally

One Dockerfile, one target per lane. The builder stage runs the same
`scripts/build-*-release.sh` the release workflow runs, so there is no second
build definition to drift:

```sh
docker build -f docker/Dockerfile --target cpu \
  --build-arg VERSION=0.0.1 \
  --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  --build-arg JOBS=$(nproc) \
  -t vllm-cpp:local-cpu .
```

Then gate it. Without `--model` the validator checks configuration and layout
and says plainly that the image has no runtime evidence; with one it also boots
the server, requires `/health` and `/version`, runs the image's own declared
healthcheck, and requires a clean SIGTERM shutdown:

```sh
python3 scripts/validate-container-image.py \
  --image vllm-cpp:local-cpu --lane cpu --version 0.0.1 \
  --model /path/to/opt-125m
```

`scripts/check-container-matrix.py` keeps `release/container-matrix.json` and
the Dockerfile agreeing about lanes, tags and digest-pinned bases;
`scripts/check-container-workflow.py` holds the publish workflow to its
least-privilege stages. Both run in preflight and CI.

To exercise the release pipeline without publishing anything, trigger its
manual entry point:

```sh
gh workflow run release.yml --ref main
```

Manual runs are always dry runs. Publication additionally requires the exact
tag declared in `release/release-version.json` (currently
`v0.0.3-pre.1`), a release matrix whose required lanes are all marked
ready, successful verification and attestation jobs, and approval of the
protected `release` environment. Build and verification jobs have read-only
repository permissions; only attestation receives OIDC authority, and only the
final protected job receives `contents: write`. The current declaration is a
prerelease; the publisher must pass GitHub's prerelease flag and a manual dry
run cannot publish.

Any OpenAI client works by pointing its `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

### Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../src/vllm/entrypoints/openai/api_server.cpp).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4), recorded per engine step by the engine that serves your requests. Series and families keep stable addresses as new ones register (#330), so a long-lived scrape target does not read through a reallocated registry |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`) |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |
| POST | `/v1/embeddings` | Embeddings. Registered **only** when an embedder is attached, so a text server answers 404 at the route table |
| POST | `/v1/audio/transcriptions` | Speech to text (multipart: audio as `file`, `response_format` as a form field). Registered **only** when a transcriber is attached |
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |

The reference-audio side of IndexTTS-2.5 is complete in the library -- a 16 kHz
clip goes through the SeamlessM4T feature extractor, the w2v-bert Conformer, the
layer-17 hidden-state tap, the checkpoint's stored-statistics normalization and
the semantic codec to discrete codes, and the talker's prompt is assembled from
that conditioning plus the text -- but none of it is reachable from a command or
a route yet. The greedy generate loop that turns the prompt into mel codes is
ported too, and so is the STATED-emotion path -- eight weights selecting rows
from the checkpoint's own speaker and emotion matrices by cosine similarity -- so
text plus a reference clip and an emotion reaches mel CODES in the library. What
is still missing before audio is reading those matrices out of the converted
checkpoint, BigVGAN's separately-downloaded checkpoint, and any command or route
to call it from. Inferring the emotion from a clip instead of stating it needs a
Conformer and a Perceiver that are not ported.

There is **no `/v1/audio/speech`**. Text to speech is not servable: the
IndexTTS-2.5 stages are ported and gated at reduced dimensions, with further
stages named as missing by the checkpoint's own manifest, and no route is
registered, the public ABI carries no synthesis entry point, and loading the
family refuses with a message naming the missing pieces (#634). Asking a running server for speech
today is a 404 at the route table, not a runtime error, and that is the accurate
signal: the capability does not reach any surface yet.

`prompt_logprobs` is accepted on `/v1/completions` and `/v1/chat/completions`
and the engine computes it — every prompt position is scored against the token
that followed it, accumulated across chunked prefill — but the **response body
does not carry it yet**: emitting it needs the OpenAI `echo` wiring, which is
not done. Until then it is reachable through the library
(`RequestOutput.prompt_logprobs`), not over HTTP. `logprobs`/`top_logprobs` on
GENERATED tokens are emitted normally.

That computation is gated on the **CPU** backend only. A step that owes prompt
logits takes the full-logits route, and on that route the sampler is handed a
host-resident logits buffer carrying the accelerator's device label — sound on
unified memory, and **not yet verified on CUDA at all, discrete or otherwise**.
Treat `prompt_logprobs` on a GPU build as unverified until that gate runs; the
mechanism and the exact owed invocation are in
[`.agents/specs/prompt-logprobs.md`](../.agents/specs/prompt-logprobs.md)
(risk 4 and the `PENDING` CUDA smoke gate). Requests that do NOT set it are
unaffected on every backend — the route is only taken for a step where some
request asked.

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. See
[MiniMax-H3: video + audio generation](#minimax-h3-video--audio-generation).

### `max_tokens`: what a non-positive value means

Some clients (Hermes among them) send `max_tokens: -1` to mean "no client-side
limit". A non-positive `max_tokens` — or `max_completion_tokens` on
`/v1/chat/completions`, which takes precedence — is treated as **unset**, not as
an error and not as a clamp to some constant. Unset then generates up to
`max_model_len` minus the prompt length, mirroring vLLM.

That distinction is load-bearing for long-context requests: substituting a
constant would cap exactly the request that asked to be left unlimited, and the
client would see `finish_reason: length` with no way to tell it apart from a
limit it set itself. Use `VT_SERVER_MAX_NEW_TOKENS` when you want a serving-side
ceiling.

### Which token ids stop a generation

Stop ids come from two files in the checkpoint, not one. `config.json`'s
`eos_token_id` supplies the **primary** eos id, and the sibling
`generation_config.json` supplies **secondary** stop ids that are usually a
superset of it. Gemma-4-26B is the clearest case:

```
config.json             eos_token_id: [1, 106]
generation_config.json  eos_token_id: [1, 106, 50]
```

Both are read, mirroring vLLM's default `--generation-config auto`. The
secondary ids are merged into the request's `stop_token_ids`, so a chat model
stops on its turn-level token rather than running to the length cap. A missing
or malformed `generation_config.json` is a silent no-op.

`ignore_eos: true` suppresses **all** of them, primary and secondary alike, and
generation then runs to the token budget. The ids still count toward
`min_tokens` masking either way, so `min_tokens` cannot be satisfied by emitting
a stop token early.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Max concurrent sequences (also sizes the HTTP worker pool). Was `8`, which put a c8 client exactly on the batch ceiling; vLLM's own default is 1024, which we do not mirror because this also caps the padded decode-graph set. On a GDN/Mamba model under speculative decoding this also multiplies the recurrent state, which is sized `max-num-seqs x (k+1)`; an unservable budget is refused at load with the arithmetic |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [docs/SGLANG-COMPAT.md](SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (42 names over 38 families). `auto` detects from the chat template, `none` disables. For `gemma4`, OpenAI chat uses the text-seam parser (wrapped `<\|tool_call>` **or** bare `call:NAME{ARGS}`) so free-form / detokenized tool bodies still become `tool_calls`. **`inkling` needs `"skip_special_tokens": false` on the request today** — its whole grammar is special tokens and we have no `adjust_request` seam to force the flag off for you, so at the `true` default the detokenizer strips the markers before the parser runs ([#695](https://github.com/mudler/vllm.cpp/issues/695)). `--reasoning-parser inkling` is not registered at all ([#703](https://github.com/mudler/vllm.cpp/issues/703)) |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`, `muse_glimmer`, `qwen3`, `mimo`). `auto` detects, `none` disables. `qwen3` and its `mimo` alias are the engine-backed adapter (one upstream class, two registry names): thinking is ON, so a marker-less stream is reasoning and a `<tool_call>` ends reasoning with no `</think>`. `auto` never selects it — a generic `<think>` template resolves to `think_auto`, which is the right default for hybrid-thinking models that may answer with no think block at all |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. `dspark` speculates on the Qwen3.6 gate models (native + Speculators drafts), token-identically to speculative-off, but is not gated on speed (currently ~2% behind at c1). A GGUF target, or a target with no aux multi-tap, is refused by name (`SPEC-DSPARK`). Its sequential Markov sampling runs on device by default; `VT_DSPARK_DEVICE_SAMPLE=0` restores the host loop (token-identical, cost only). The speculative verify runs from a captured CUDA graph, worth +12.2%/+3.5% on the 35B cells; `VT_SPEC_DECODE_GRAPH=0` restores the eager verify (also token-identical). See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | off | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it (Gemma-4 and friends). Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking` |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

#### Accepted for recipe compatibility — these flags have NO effect

A published `vllm serve` line has to reach model load. The flags below appear in
most official [vllm-project/recipes](https://github.com/vllm-project/recipes)
commands, mean nothing to this engine, and are therefore **accepted and ignored**
rather than rejected. Each one prints a notice on startup naming itself and the
reason it does nothing, so a log never implies it took effect.

| Flag | Effect here | Why it is inert |
|---|---|---|
| `--enable-auto-tool-choice` | **none** | Tool parsing is already unconditional once `--tool-call-parser` resolves; there is no second gate to open. Note `--tool-call-parser` defaults to `hermes` here, where upstream's defaults to unset, so the two flags do not line up when the parser is omitted. Upstream's validation is still mirrored: combining it with `--tool-call-parser none` is refused, as in `vllm/entrypoints/openai/cli_args.py:395` |
| `--trust-remote-code` | **none** | It authorizes executing Python from the checkpoint. This engine has no Python runtime, so there is nothing to authorize — N/A by construction, not unimplemented |

The notice is on stderr at startup, one line per flag actually passed, so what
you see in a log matches this table:

```text
server: accepted '--trust-remote-code' for published-recipe compatibility; it has no effect here: no Python runtime, so there is no remote code to trust
```

The mirrored validation is reported before the parser dialect is checked, so a
contradiction is named as a contradiction rather than passing silently (`none` is
itself a valid selection):

```text
server: Error: --enable-auto-tool-choice requires --tool-call-parser
server: (--tool-call-parser none selects NO parser; name a parser, or drop --tool-call-parser to keep the hermes default)
```

This list is **enumerated, not a catch-all**. Any other unrecognized flag still
aborts with `server: unknown argument '<flag>'`, including flags that are inert
only because the capability is missing (`--tensor-parallel-size` and the other
parallelism flags) — silently accepting those would let you believe you got
tensor parallelism when you did not.

#### Context length vs the KV pool

The KV pool holds `--num-blocks × --block-size` tokens — `256 × 32 = 8192` by
default. A request longer than that can never be scheduled, so the engine
refuses it early rather than leaving it in the waiting queue forever. Two checks
do that, mirroring vLLM:

- **At startup.** If `--max-model-len` is given and the pool cannot hold one
  sequence that long, the server exits with the sizes and the flags that close
  the gap (vLLM's `_check_enough_kv_cache_memory`). If it is **not** given, the
  serving length is auto-fitted down to what the pool holds and logged
  (vLLM's `_auto_fit_max_model_len`) — so raising `--num-blocks` is what buys a
  longer context.
- **At admission.** A prompt at or past the resolved `max_model_len` is
  rejected with **HTTP 400** (`BadRequestError`) naming both lengths, exactly as
  vLLM's `_validate_prompt_len` does. It is never a finish reason and never a
  500.

Set `VT_ENGINE_STEP_LOG=1` to print a per-step engine heartbeat if you need to
confirm that a quiet engine is idle rather than stalled.

For a production deployment, use [LocalAI](https://localai.io), which can embed
engines like this behind a model gallery, multi-model serving, the full OpenAI
API surface, auth, and metrics.

## Muse Glimmer 30B from a GGUF k-quant

The text tower loads from a `muse-glimmer`-architecture GGUF, so the 30B model
runs from a ~17 GB k-quant instead of a ~60 GB bf16 checkpoint. Point `--model`
straight at the file; the config comes from the GGUF's own metadata, so no
`config.json` is needed:

```sh
./build/vllm-server --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
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

## MiniMax-H3: video + audio generation

### The exact weights (so a render is reproducible)

Five files. The DiT and encoder are community GGUF quantisations; the two VAEs and
the tokenizer come from the official checkpoint.

| file | size | source |
|---|---|---|
| `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19.9 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `vae/diffusion_pytorch_model.safetensors` | 5.2 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/video_vae/` |
| `audio_vae/model.safetensors` | 0.6 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/audio_vae/` |
| `tokenizer.json` | 7 MB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/tokenizer/` |

Take each VAE's `config.json` from the same directory as its weights: they carry the
per-channel `latents_mean` / `latents_std` and the temporal `clip_length` /
`token_drop`, and the decode is wrong without them.

**Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude
outliers that 3-bit cannot hold. In a controlled A/B (same prompt, seed, code and
VAEs, only the DiT quantisation changed) Q3_K_M gave a murky silhouette under a
visible lattice and Q4_K_M gave a photoreal close-up. The full bf16 release is
66.3 GB across 13 shards if you want to go further.

Higher-precision arms that exist but are not the default: NVFP4
([lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4))
and the original bf16 weights under `FL2VA/transformer/`.

### The PRUNED checkpoints — more precision for the same footprint

The community `pruned` variants **are supported and are drop-in**: pass one to
`--dit` exactly as you would an unpruned file. Nothing else about the command
changes.

They are not lossily pruned. AdaLN modulation dominates the unpruned parameter
count — `adaln_proj` alone is 13.04B of 33.12B (39.4%) — because the model
projects a 5376-wide conditioning vector into modulation parameters in every one
of the 50 blocks. But modulation depends only on the timestep, so that projection
is almost entirely redundant, and the pruned form replaces it with a `[1025, 8]`
timestep table feeding an 8-wide `adaln_proj.linear`. 13.04B parameters become
0.04B and the DiT drops from 33.12B to 20.11B, with the modulation path kept at
full precision.

The practical consequence: **a pruned Q8_0 costs about what our unpruned Q4_K_M
costs.**

| file | size | source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q8_0.gguf` | 21.4 GB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| `minimax_h3_ref2va_pruned-Q8_0.gguf` | 21.4 GB | same repo — the `ref2va` partition |
| `minimax_h3_{fl2va,ref2va}_pruned-{Q2_K,Q3_K,Q4_K,Q5_0,Q6_K}.gguf` | 6.7-16.6 GB | same repo |
| `minimax_h3_{fl2va,ref2va}_pruned_nvfp4.safetensors` | 12.5 GB | [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) |

The partition rule below still applies: a `fl2va` file serves `t2va` and `fl2va`,
a `ref2va` file serves `ref2va`.

**What is actually verified, and what merely exists.** The distinction matters
because a render takes hours before it tells you anything:

| arm | status |
|---|---|
| **Q4_K_M** | **VERIFIED end to end** — every render in this doc, on BOTH partitions (t2va + fl2va on FL2VA, ref2va on REF2VA). Use this. |
| Q3_K_M | verified BAD (the A/B above): murky silhouette under a lattice |
| bf16 (66.3 GB, 13 shards) | loader + device streamer implemented and gated, but **CPU-only** verification — no end-to-end GPU render has been done |
| NVFP4 | exists; loads (unpruned and pruned) |
| **pruned Q8_0** | **loads and renders** — the A/B is in `.agents/specs/minimax-h3.md` section 8.21 |
| pruned Q6_K / Q5_0 / Q4_K / Q3_K / Q2_K ([unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)) | load through the same path; only Q8_0 has been rendered |

### The trap: this checkpoint does not serve every task

**`MiniMax-H3-FL2VA-Q4_K_M.gguf` is the FL2VA partition. It serves `t2va` and
`fl2va` — NOT `ref2va`.** H3 ships two independently-served DiT partitions and
the task must match the one you loaded; upstream's `_resolve_task` raises on the
mismatch.

Pass a reference image against this file and you get a task/partition mismatch.
It does not fail loudly — it renders, and the render is *wrong*: a coloured
diagonal lattice over the whole frame, worse the larger the canvas. Measured on
one prompt and canvas (1344x768 / 124f), as a period-16 seam ratio where 1.15 is
clean:

| configuration | seam ratio |
|---|---|
| ref2va against FL2VA (the mismatch) | 2.28 |
| t2va against FL2VA (correct) | **1.19** |

The small-canvas case is what makes this expensive to spot: at 864x480 the same
mismatch measures 1.15 and looks acceptable, so the bug only becomes obvious at
the resolution you actually want.

Pass `--partition fl2va` explicitly. The driver mirrors upstream's raise, so a
mismatch is rejected at the CLI rather than silently rendered.

For a reference-image render you need the **Ref2VA** partition instead, and the
one to use is **`MiniMax-H3-REF2VA-Q4_K_M.gguf`** (19.9 GB,
[realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs)) —
the same quantisation as the FL2VA file above, and verified coherent:

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-REF2VA-Q4_K_M.gguf --dequant-bf16 --partition ref2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "..." --ref-image subject.ppm \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 512 --width 512 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

**Do NOT use the NVFP4 Ref2VA weights.** `minimax_h3_ref2va_nvfp4_full` renders the
multicolour patch grid, and it took three investigations to establish that this is the
QUANTISATION and not the ref2va path: the identical reference-row assembly, packed-block
layout and denoise loop render coherently on Q4_K_M (period-16 seam **1.13**, VAE-input
latent adjacent-cell cosine **0.8526**). Ref2VA on Q4_K_M is a working mode; Ref2VA on
NVFP4 is not.

### Writing the prompt (read this first)

Two things decide whether you get what you asked for, and neither is obvious.

**To get SPEECH, ask for it and supply the line.** The model generates video and
audio jointly, so a prompt describing a silent performance produces room tone and
ambience, which is correct but not what most people expect. Say that the character
talks, describe the voice, and put the words in the prompt:

```
It is TALKING to the camera: its mouth moves clearly in sync with its speech,
in a dry, deadpan tone.

It says, clearly and audibly: "Michael scheduled another all-hands.
It is about the printer. Again."

Audio: a single clear voice, close-miked, with quiet room tone underneath.
```

That prompt produced audio an ASR pass transcribed back word for word. A prompt
that only described expressions and sighs produced ambience at about 13 dB lower
level and no speech at all.

**Refer to references BY TAG in the prompt text.** A reference is bound by naming
it, not merely by being passed on the command line. Use `<Picture i>`, `<Video k>`
and `<Audio j>`, numbered from 1 per type, matching the order you pass them:

```
<Picture 1> is a cyan llama mascot wearing white sunglasses.

A talking-head interview. The subject is the llama from <Picture 1>, sitting in a
grey office chair ...
```

Other prompt notes: frame count runs on the 17n+5 grid at 24 fps, and the trained
range is roughly 124 to 362 frames (about 5 to 15 seconds). Text rendered *inside*
the video (signage, wordmarks) is the model's weakest area and will often come out
malformed; composite real logos in afterwards.

`/v1/videos` generates video with sound through the MiniMax-H3 diffusion model.
It speaks **OpenAI's Sora video shape**, so an OpenAI client works against it
unmodified, and it keeps the richer native knobs alongside.

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B \
  --video-dit /path/to/h3-dit.gguf --video-vae /path/to/video-vae.safetensors \
  --audio-vae /path/to/audio-vae.safetensors \
  --video-vae-config video_vae/config.json --audio-vae-config audio_vae/config.json \
  --video-encoder /path/to/h3-encoder.gguf
```

```python
video = client.videos.create(model="sora-2-pro", prompt="a cat on a skateboard",
                             size="1280x720", seconds="8")
while client.videos.retrieve(video.id).status not in ("succeeded", "failed"):
    time.sleep(5)
open("out.mp4", "wb").write(client.videos.download_content(video.id).read())
```

### Request fields

| Field | Spelling | Meaning |
|---|---|---|
| `prompt` | both | Required. The text conditioning |
| `model` | OpenAI | Recorded and echoed back. A name this server does not serve is a `warning` on the job, never a rejection: the video model is chosen at startup |
| `size` | OpenAI | `"<width>x<height>"`, e.g. `"1280x720"`. Whole pixels, both positive |
| `seconds` | OpenAI | Duration, as a number or a numeric string (`8` and `"8"` both work) |
| `input_reference` | OpenAI | The image the video starts from. A filesystem path or a `data:` URL |
| `metadata` | OpenAI | Free-form string map, passed through untouched. Two keys are acted on: `input_reference_video` and `input_reference_audio` (see below) |
| `width`, `height` | native | Output geometry in pixels |
| `duration` | native | Duration in seconds |
| `task` | native | `t2va`, `fl2va`, `ref2va`; resolved from the inputs when omitted |
| `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed` | native | The H3 generation knobs. Accepted at the top level or nested under `extra_params` |

**Precedence.** When a body carries both spellings of one value, the **native
field wins**: `width`/`height` beat `size`, `duration` beats `seconds`. That
direction keeps every request that parses today meaning exactly what it meant
before. Both spellings are validated either way, so a malformed `size` is a 400
even when explicit `width`/`height` would have overridden it.

**`input_reference` maps to fl2va first-frame conditioning.** OpenAI documents
it as the image the generated video starts from, which is what fl2va expresses:
the supplied image is pinned as frame 0 of the output. H3's other image mode,
ref2va, prepends whole reference images as their own blocks (subject or style
guidance that never becomes a frame), so it stays reachable only through the
native `task` field and the `minimax-h3-gen` CLI. Two limits: the image must be
a **binary PPM (P6)** (no PNG or JPEG codec is vendored, the same residual the
chat multimodal path carries), and it must already be at the output resolution
(no image resampler is vendored). A mismatch is refused with the resolved
geometry in the message.

### Video and audio references (`metadata`)

H3 supports three reference modalities and OpenAI's schema has a slot for one,
so the other two enter through `metadata`, the standard OpenAI free-form string
map. Strict clients tolerate it, and no invented top-level field breaks their
schema validation. Unknown metadata keys are passed through untouched.

```jsonc
{
  "prompt": "the same scene, at dusk",
  "metadata": {
    "input_reference_video": "/tmp/vllm_h3_videos/job0",  // DIR of frame_%06d.ppm
    "input_reference_audio": "/tmp/voice.wav"             // 16-bit PCM WAV, or a data: URL
  }
}
```

`input_reference_video` is a **directory of `frame_%06d.ppm`**, which is exactly
what this server and `minimax-h3-gen` write, so one run's frames chain straight
into the next request. It is not a container file: no demuxer is vendored.

**A video reference is SILENT.** `MiniMaxH3EncodeReferenceVideo` emits a
`kVideoAudio` block with `ref_audio_t == 0`, so the clip contributes no sound of
its own. Supplying `input_reference_audio` alongside it attaches the audio to
that same block (one block carrying both, the layout upstream builds); without
it the reference is picture only. That is a real limitation, not an omission.

**Legal combinations.** fl2va keyframes and ref2va reference blocks are
exclusive in the pipeline itself
([`minimax_h3_pipeline.cpp`](../src/vllm/model_executor/models/minimax_h3_pipeline.cpp)),
so the request parser enforces the same rule and returns a 400 naming the
offending pair rather than dropping a reference you supplied.

| `input_reference` | `metadata.input_reference_video` | `metadata.input_reference_audio` | |
|---|---|---|---|
| (none) | (none) | (none) | t2va, prompt only |
| image | (none) | (none) | fl2va, the image is frame 0 |
| (none) | clip | (none) | ref2va, silent video reference |
| (none) | (none) | WAV | ref2va, audio reference |
| (none) | clip | WAV | ref2va, one block carrying both |
| image | clip and/or WAV | | **400**: keyframe and reference conditioning are exclusive |

The video reference needs `--video-vae` (the encoder half of the same file) and
the audio reference needs `--audio-vae`; both load lazily, once, on the first
request that asks for them.

### The job lifecycle

`POST /v1/videos` returns immediately with `{"id": "vid_1", "status": "queued"}`;
generation is minutes long, so the synchronous twin `POST /v1/videos/sync` exists
for scripts that would rather block. `GET /v1/videos/{id}` reports `queued`,
`running`, `succeeded` (with `output_path`) or `failed` (with `error`).

`GET /v1/videos/{id}/content` returns the finished MP4 with
`Content-Type: video/mp4`. An unknown id is a 404; a job that has not finished is
a **409** naming its current status rather than a truncated file; a failed job is
a 500 carrying its failure; an output that has since vanished from disk is a 500
rather than a 200 with zero bytes.

The library never spawns a process, so generation and muxing enter through a
caller-supplied `VideoRunner` callback (`examples/server/main.cpp` supplies one
that invokes `ffmpeg`, path configurable with `--video-ffmpeg`).

### Video family, and family-specific load knobs

`/v1/videos` serves whichever video family the `--video-dit` checkpoint belongs
to. By default the family is **detected** from what the checkpoint holds, and
that is unchanged.

`--video-family NAME` pins it instead. Two registered families exist,
`minimax-h3` and `ltx-2.5`, and a name outside that set is refused at argument
parsing, before the text model loads, with the registered names printed. It is
never a hint: a declared family that cannot load the checkpoint fails loudly
rather than falling back to detection, because a checkpoint handed to the wrong
family does not fail, it renders noise.

`--video-extra KEY=VALUE`, repeatable, carries a family's own load knobs. LTX-2.5
cannot load without `dit_config_path`, and it needs `encoder_config_path` beside
`--video-encoder` when the text encoder declares no `gemma_config` (the shipped
one does not); MiniMax-H3
defines `partition`, for which `--video-partition` remains the documented alias.
A bare `KEY` with no `=` is refused rather than read as an empty value, and a
`--video-extra partition=X` contradicting `--video-partition Y` is refused rather
than resolved by whichever assignment ran last. A family refuses any key it does
not define, so a mistyped knob is an error instead of a silently defaulted
render.

```sh
vllm-server --model /path/to/text-model \
  --video-family ltx-2.5 \
  --video-dit ltx-2.5-22b-distilled-fp8.safetensors \
  --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
  --video-encoder gemma4-12b-with-proj-nvfp4-torchao.safetensors \
  --video-extra encoder_config_path=ltx-2.5-gemma4-text-config.json \
  --video-extra dit_config_path=ltx-2.5-transformer-config.json \
  --video-extra model_version=2.5 --video-extra allow_unported_modules=1
```

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 18`,
36 exported functions) suitable for `dlopen` / FFI / LocalAI integration.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               /* sp.temperature = 0.0 means greedy */

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle, blocking and streaming completion, non-blocking
concurrent requests, memory helpers, and diagnostics. Later ABI versions add:

| ABI | Adds |
|---:|---|
| v2 | Structured output (JSON schema, JSON object, regex, choice, GBNF) |
| v3 | Chat with tools and chat templates |
| v4 | Tool-parser selection |
| v5 | Reasoning-parser selection |
| v6 | Speculative decoding |
| v7 | Prefix caching (tri-state) |
| v8 | Custom logits processors |
| v9 | Engine sizing: chunked-prefill token budget, scheduling policy, external KV connector / LMCache |
| v10 | Jump-forward decoding (tri-state, default off) |
| v11 | Audio transcription through `vllm_transcribe` |
| v12 | Video and audio generation through `vllm_video_*` |
| v13 | Pre-tokenized completion through `vllm_complete_tokens` |
| v14 | Explicit device selection (`auto`, CPU, or CUDA) |
| v15 | Embeddings through `vllm_embed` |
| v16 | Absolute KV-cache memory sizing |
| v17 | The OpenAI server as a thin ABI client through `vllm_server_main` |
| v18 | Video model-family selection (`family`, `vllm_video_engine_family`) and family-specific `extra_keys`/`extra_values` on `vllm_video_*` |

Chat templates render through the vendored google/minja engine, the same
renderer llama.cpp ships.

## Consuming it from C++

The higher-level surface lives under [`include/vllm/`](../include/vllm/).
`LoadedEngine::FromModelDir(...)`
([`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h))
hands back either the synchronous `LLMEngine`
([`v1/engine/llm_engine.h`](../include/vllm/v1/engine/llm_engine.h)) or the async
`AsyncLLM` ([`v1/engine/async_llm.h`](../include/vllm/v1/engine/async_llm.h)) that
the server itself uses.

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;
ep.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

The underlying portable tensor runtime is `vt::` ([`include/vt/`](../include/vt/)),
which carries no ggml or PyTorch dependency.

Video and audio generation is reached through `vllm::multimodal::VideoEngine`
([`multimodal/video_engine.h`](../include/vllm/multimodal/video_engine.h)).
`LoadVideoEngine` resolves the model family from what the checkpoint HOLDS, never
from a filename, and refuses rather than guessing: zero claimants, several
claimants, and an unregistered declared `family` are all errors that name what was
seen and what is registered. A caller who supplies no `dit_path` is told which
artifact is missing rather than being advised to declare a family, which would not
help. A family adds itself with `RegisterVideoFamily`, which refuses a name that
is already registered, because two families under one name would collapse into a
single claimant and leave the choice of loader to link order.

Two families are registered. `minimax-h3` is detected by `video_patch_proj` plus
`audio_patch_proj`; `ltx-2.5` by `patchify_proj` plus `audio_patchify_proj`, with
or without the ComfyUI `model.diffusion_model.` prefix. Each family reads its own
knobs from `extras`. H3 takes `partition`. LTX-2.5 takes
`audio_prompt_embeds_path` (the audio stream's conditioning, the twin of the
seam's `prompt_embeds_path`, which carries the video stream), `pipeline_kind`
(default `distilled_two_stage`), `model_version` (only for a checkpoint that
declares none), `dit_config_path`, `allow_unported_modules`, `max_phase`,
`prompt_embeds_valid_rows`, `upsampler_path` and `duration_head_path`. An extra a
family does not define is refused, never ignored. One caveat inside that set:
`duration_head_path` is accepted but INERT — the duration head is ported and gated
as a brick, nothing in the video engine constructs one, and no code reads that
key, so supplying it neither loads a head nor enables an AUTO duration. Give
`num_frames` (or `duration`, which is exact arithmetic against the recipe's frame
rate) instead.

`prompt_embeds_valid_rows` is how many of the supplied conditioning rows are real
tokens; absent, every row is. It matters because the embeddings connector
substitutes its learnable register table at PADDED positions, so padding decides
which of the connector's inputs are learned constants rather than caption
features. Upstream always knows this because its tokenizer produced the mask;
this seam reads conditioning from a file, which carries none.

`dit_config_path` names a JSON file holding the DiT's `{"transformer": {...}}`
configuration, and it exists because only one of the two shipped LTX-2.5 DiTs
carries one. The first-party NVFP4 file embeds it in `__metadata__["config"]`;
the ungated `vonkaiser/LTX-2.5-FP8-NVFP4` FP8 DiT has no `__metadata__` at all.
Tensor shapes resolve the geometry but not the values no shape encodes, so
without a config `double_precision_rope` would default to false and
`av_ca_timestep_scale_multiplier` to 1, where LTX-2.5 declares `float64` and
`1000`. Both move every RoPE angle and every audio-to-video modulation, so a DiT
that declares no config is refused until one is supplied rather than rendered
under defaults that contradict the model family. A supplied config is adopted
only when it reproduces the identical weight contract the shapes describe, and
supplying one for a checkpoint that already declares its own is refused rather
than ordered.

The LTX-2.5 arm runs on the CPU in f32 and on CUDA in bf16. `device = 0` takes
the f32 parity forward; `device = 1` stages the DiT to the GPU one tensor at a
time and runs the device-resident forward, so a CUDA handle means a CUDA forward.
On a build with no CUDA backend, `device = 1` is refused by name rather than
served the CPU forward behind a CUDA handle. `encoder_path` loads the Gemma-4
text tower, and the request's own `prompt` then conditions the render; the tower
itself runs on the CPU in f32 whichever device the DiT is on. Without one,
conditioning comes from the two prompt-embeds files, which must agree on their
row count.

`Sampler`'s `logprobs_mode` selects which tensor the returned logprobs are read
from, and all four of vLLM's values now work: `raw_logprobs` (the default) and
`raw_logits` are snapshotted before any logits processor runs, so they describe
the MODEL's distribution; `processed_logprobs` and `processed_logits` are taken
after temperature and top-k/top-p, so they describe the distribution actually
SAMPLED from — a token top-k masked away reads `-inf` there and its true value
under the raw pair. It is selectable by constructing a `Sampler` directly; there
is no config, CLI or request field for it yet.

`LogprobsTensors::slice_request(req_idx, request_num_positions)` cuts that
batch-wide payload by rows. The second argument is the requested row count;
each row keeps the source tensor's independent `num_tokens_per_position`
width.

The LoRA adapter headers ([`lora/lora_weights.h`](../include/vllm/lora/lora_weights.h),
[`lora/punica.h`](../include/vllm/lora/punica.h),
[`lora/layers.h`](../include/vllm/lora/layers.h)) are present but **not yet wired
to any engine path**: they are the in-progress runtime (`LORA-RUNTIME`), not a
supported way to serve an adapter. There is no CLI flag, server flag, config key
or C-ABI field for LoRA, and adding one is a later work item — see
[`.agents/specs/lora-adapter.md`](../.agents/specs/lora-adapter.md).

`SamplingParams::logprobs` accepts `-1` for "every vocab entry", as vLLM's does;
it returns the same gathered shape a finite count returns, one entry per vocab id
per position.

Over HTTP the same `-1` reaches the chat surface: `{"logprobs": true,
"top_logprobs": -1}` is accepted, as in vLLM, and returns every vocab entry for
each generated token. No numeric range is enforced on either surface — vLLM's
`check_logprobs` request validation and its `max_logprobs` model cap are not
ported yet. Two consequences: `{"logprobs": -1}` on the **completion** surface
returns empty `top_logprobs` maps where vLLM answers `400`, and an out-of-range
count is not rejected. Both are tracked by
[issue #249](https://github.com/mudler/vllm.cpp/issues/249).

`SamplingParams::logprob_token_ids` scores an EXPLICIT set of vocab ids instead —
vLLM's generative-scoring path, and what to reach for when you only need a few
labels compared, since it avoids the full-vocab sort `logprobs=-1` costs:

```cpp
vllm::SamplingParams sp;
sp.max_tokens = 1;
sp.logprob_token_ids = std::vector<int32_t>{yes_id, no_id};  // `logprobs` unset
```

Each returned position then carries exactly those ids plus the sampled token,
whose `rank` is still its rank over the WHOLE vocabulary, so it stays comparable
across requests. At most 128 ids (vLLM's `MAX_LOGPROB_TOKEN_IDS`); setting
`logprobs` as well is allowed only when it equals the id count, and the explicit
ids win. This is a library-API field today — the OpenAI request field is not
wired yet.

### KV-cache events, and `kv_cache_report_mode`

`SamplingParams::extra_args` is a per-request string map mirroring vLLM's
`extra_args`, and the one key read from it today is `kv_cache_report_mode`:

```cpp
vllm::SamplingParams params;
params.extra_args = std::map<std::string, std::string>{
    {"kv_cache_report_mode", "full"}};
```

It controls how much of that request's prefix-cache activity reaches the
KV-cache event stream. `"incremental"`, the default and what you get whenever the
key is absent, reports only blocks the request newly STORED. `"full"` also
re-reports the blocks it REUSED from the cache, which is what a prefix-cache-aware
router needs to learn that this engine already holds a prefix.

Events are OFF unless a `vllm::distributed::KVEventsConfig` with
`enable_kv_cache_events = true` is passed to the `Scheduler`, so
`kv_cache_report_mode` changes nothing by itself. With events on, each engine step
publishes at most one `KVEventBatch` — a wall-clock `ts`, that step's
`BlockStored` / `BlockRemoved` / `AllBlocksCleared` events, and the data-parallel
rank — to the configured publisher, and its msgpack encoding is byte-identical to
what vLLM puts on the wire.

Two limits to know. The **`zmq` publisher is not ported**: asking for it throws
rather than silently downgrading, because the live socket transport needs a
dependency this project does not carry, so `publisher` must be `"null"` today —
and it must be set explicitly, since an unset value is not yet resolved the way
vLLM resolves it ([issue #353](https://github.com/mudler/vllm.cpp/issues/353)).
And `extra_args` is reachable **only from the C++ API**: the HTTP door to it
(`vllm_xargs`) is not ported, so an OpenAI request cannot set the report mode.

## Multimodal input (image, video, audio to text)

Multimodal input is served over the **OpenAI API**, not the CLI. `vllm-cli` is text-only:
`--model --prompt --max-tokens --temperature --top-k --top-p --seed --stream
--speculative-config --tokenizer-config`.

Start the server with a multimodal model, then send content parts on
`/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.chat.completions.create(model="Qwen3.6-27B", messages=[{"role": "user", "content": [
    {"type": "text",      "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
]}])
```

Accepted part types (`src/vllm/entrypoints/openai/chat_mm.cpp`):

| part type | modality |
|---|---|
| `image_url` | image |
| `video_url` | video |
| `input_audio` / `audio_url` | audio |

### Per-prompt input limits — the mechanism exists, the flags do not yet

vLLM caps how many items of each modality one prompt may carry
(`--limit-mm-per-prompt`), and `--language-model-only` is sugar for setting every
one of those limits to 0, which makes the server refuse multimodal requests
outright. **Neither flag is accepted yet** — `vllm-server` still exits on both,
and there is no config key or C ABI field for them.

What landed (#607, wave L1) is the mechanism underneath, as library-internal
headers only: `vllm::MultiModalConfig::GetLimitPerPrompt`
([`config/multimodal.h`](../include/vllm/config/multimodal.h)) resolving
upstream's precedence, and the refusal it carries
([`multimodal/processing/context.h`](../include/vllm/multimodal/processing/context.h)),
which raises `vllm::v1::InputValidationError` — the same type the API server
answers with HTTP 400. Nothing constructs that config on a live request, so
**today no request is limited or refused on item count**, and a chat request
carrying several images still has all but the first silently dropped by
`chat_mm.cpp`. Wave L2 adds the two flags, the C ABI field, and the call-site
wiring that makes the limits take effect.

## MiniMax-H3 browser console (`vllm-video-studio`)

A standalone browser console for MiniMax-H3, deliberately **separate** from the
OpenAI-compatible API server: `examples/server` is the API surface and a UI does
not belong in it. The studio owns its own endpoints and drives the public C ABI
(`vllm_video_*`) like any other FFI consumer, so it is also a worked example of
that ABI.

Built with the server (`-DVLLM_CPP_SERVER=ON`), because it shares the same
vendored HTTP transport.

```sh
vllm-video-studio --models-dir /path/to/h3 --port 8080
```

Then open `http://localhost:8080`. It discovers the five H3 files under
`--models-dir`, or each can be pointed at explicitly with `--dit`, `--encoder`,
`--video-vae`, `--video-vae-config`, `--audio-vae`, `--audio-vae-config` and
`--tokenizer`. Other flags: `--host`, `--device`, `--workdir`, `--ffmpeg`,
`--partition`, `--keep-quant`, `--prompt-embeds`, and `--ui` to serve a custom
web root.

The weights, and why each one is needed, are in the MiniMax-H3 section below.

## MiniMax-H3: video + audio generation


Renders an MP4 with a stereo track. Weights: a GGUF DiT (use **Q4_K_M**), the Qwen3-VL-32B
encoder, and both VAEs.

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 --partition fl2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "A golden retriever runs across a sunlit beach, waves crashing behind it" \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 768 --width 1344 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

`--partition` is REQUIRED and names the partition the checkpoint you passed
actually serves — see the trap above. This is the command every render in this
document was produced with: Q4_K_M DiT and encoder, `--dequant-bf16`, task
**t2va** (no reference image), the 1344x768 default canvas, 124 frames, 50 steps.

Cost, so you can plan: **~176 s per step** at 1344x768 / 124f on a 20-SM sm_110
device, so a 50-step render is about **2.5 hours** plus roughly 30 minutes of
weight loading. Dropping to 512x512 costs ~15 s/step (~13 minutes end to end),
which is the right canvas for iterating on a prompt before committing to a full
render. `--dequant-bf16` holds the DiT as bf16 (~66 GB resident); `--keep-quant`
is the low-memory arm.

Conditioning modes, all optional and mutually exclusive where noted:

```sh
--first-frame start.ppm --last-frame end.ppm   # pin the first and/or last frame (fl2va)
--ref-image subject.ppm                        # reference image, repeatable (ref2va)
                                               # NOT served by the FL2VA checkpoint above --
                                               # needs a Ref2VA partition (see the trap)
--ref-video prev_workdir/                      # reference clip, reads frame_%06d.ppm
--ref-audio voice.wav                          # reference audio
--noise-aug 0.9                                # how hard a keyframe is pinned (1.0 = exact)
```

Reference frames are binary PPM, which is what this tool also **writes**, so one run's `--workdir`
feeds straight back in as `--ref-video` and clips chain. Convert anything else with
`ffmpeg -i in.png -pix_fmt rgb24 out.ppm`.

Worked reference renders, all on the **Ref2VA** checkpoint (`--partition ref2va`); the flags
below replace `--ref-image` in the command above:

```sh
# a SUBJECT carried into a new scene, from one still
--ref-image subject.ppm

# a reference CLIP: a directory of frame_%06d.ppm. A previous run's --workdir already
# has that layout, so clips chain without converting anything:
--ref-video /tmp/h3/            # reads /tmp/h3/frame_000000.ppm, frame_000001.ppm, ...

# reference AUDIO: 16-bit PCM WAV. Resample first -- the audio VAE is 32 kHz:
#   ffmpeg -i voice.mp3 -ac 1 -ar 32000 -c:a pcm_s16le voice.wav
--ref-audio voice.wav
```

To build a `--ref-video` directory from an arbitrary clip:

```sh
mkdir -p /tmp/refclip && ffmpeg -i source.mp4 -pix_fmt rgb24 /tmp/refclip/frame_%06d.ppm
```

Reference conditioning is **ref2va only**. On the FL2VA checkpoint these flags are refused
rather than silently ignored, which is the guard from the task/partition mirror.

Useful for measurement: `--prompt-embeds` replays text conditioning saved earlier, so two
checkpoints can be compared on byte-identical conditioning. `VT_H3_DUMP_DIR=<dir>` writes the
latents that enter each VAE (`vae_input_video_latent.f32`, `vae_input_audio_latent.f32`) plus
the pre-denormalize audio rows — that is how a render is checked numerically rather than by
eye, and it is byte-inert when unset.

(`--denoise-only`, `--dump-params` and `--save-embeds` belonged to the pre-fold driver and
were removed when the example became a thin ABI client; see the header comment in
`examples/minimax_h3_gen/main.cpp`.)

Served over HTTP too: pass `--video-dit` (plus the VAEs and configs) to `examples/server` and
`POST /v1/videos`, `POST /v1/videos/sync` and `GET /v1/videos/{id}` register. Without it the
routes stay unregistered.

## LTX-2.5: reproducing the DiT parity gate

**This section is the DiT's own parity gate, not the way to run LTX-2.5.** The
render path ships and is documented above under
[LTX-2.5: what runs, and what it cannot do](#ltx-25-what-runs-and-what-it-cannot-do):
`ltx-2.5` is one of the two registered video families
(`REGISTER_VLLM_VIDEO_FAMILY` at `src/vllm/multimodal/ltx2_video.cpp:1529`), the
Gemma-4 text tower loads from `--encoder` and sets `has_encoder`
(`ltx2_video.cpp:893`), both VAEs and the pipeline layer are implemented
(`ltx2_video_vae.cpp`, `ltx2_audio_vae.cpp`, `ltx2_pipeline.cpp`), and the
`/v1/videos` routes register for whatever family `--video-dit` resolves —
`server_main.cpp` calls the family-agnostic `LoadVideoEngine` and then prints the
resolved family. What follows here is how to regenerate the DiT's goldens. The
C++ surface is `include/vllm/model_executor/models/ltx2.h`, and it refuses by
name every arm it does not carry (a non-f32 stream dtype, the 19B
caption-projection checkpoint form, keyframe absolute-position embeddings, the
video-only / audio-only model types).

Provenance, so this can be re-checked rather than trusted: the paragraph above
replaces one that arrived at `3d89f6fc4` — the first LTX commit, where it was
true — and was never revisited as L3 through L13 built each of the six pieces it
denied.

The prompt-K/V cache (`Ltx2PromptKvCache`) is reusable across the DENOISE STEPS of
one prompt, and only those. It records a fingerprint of the prompt it was filled
for, and a forward whose context tensors, context geometry or prompt masks differ
from that prompt is refused by name rather than served K/V that would render the
cached prompt. Call `Ltx2PromptKvCache::Reset()` to rebind the same allocation to
a new request.

The gate runs the UPSTREAM modules at reduced dimensions on CPU, so it needs a
Lightricks LTX-2 checkout and the system `python3` with torch — **no checkpoint, no
venv and no gated download**. Regenerate the goldens and run it:

```sh
git clone https://github.com/Lightricks/LTX-2 ~/_git/LTX-2
python3 scripts/gen-ltx2-goldens.py \
  --ltx2 ~/_git/LTX-2 \
  --out tests/vllm/models/ltx2_goldens.inc
cmake --build build --target test_ltx2 && ./build/tests/test_ltx2
```

The generator asserts the `ltx_core` it imported came from that checkout and not
from anything installed in site-packages, and it writes the upstream revision it
executed into the generated header. Neither side checks in a weight byte: both
rebuild every tensor from one deterministic stream keyed by the parameter's name.

The pipeline layer has its own gate, and it needs a second checkout: the recipe
table is read from vLLM-Omni, which is the binding oracle for LTX even though it
carries no 2.5 row of its own. Both checkouts must be CLEAN, because a revision
anchor read from a tree with uncommitted edits stamps a SHA the goldens do not
come from.

```sh
git clone https://github.com/vllm-project/vllm-omni ~/_git/vllm-omni
python3 scripts/gen-ltx2-pipeline-goldens.py \
  --ltx2 ~/_git/LTX-2 \
  --vllm-omni ~/_git/vllm-omni \
  --out tests/vllm/models/ltx2_pipeline_goldens.inc
cmake --build build --target test_ltx2_pipeline && ./build/tests/test_ltx2_pipeline
```

If you regenerate that `.inc` against a moved upstream, expect the goldens to
carry the change rather than only the pin cases. The pipeline goldens reach the
GroupNorm eps and group count in the latent upsampler, the connector's
`rms_norm` eps, the `BlurDownsample` width (on the 1.5 arm only, since the blur
runs on the rational denominator) and the Res2s `sigma_up` clamp — that last one
on the eta = 1 arm, where the clamp binds on every step. A regeneration that
moves one of those constants alone reds a value comparison; one that moves the
constant AND the tensors together passes it, and is caught only by the cases that
compare each constant against upstream's own signature. Both layers are there
deliberately, and neither is redundant.
### The Gemma-4 text tower gate, and the interpreter it needs

The text tower is gated against the UPSTREAM HuggingFace implementation built and
run at reduced dimensions. It needs a `transformers` that registers
`gemma4_unified` in `CONFIG_MAPPING` — **5.8 or newer; 5.3.0 does not have it and
fails in a way that reads exactly like "Gemma-4 is unsupported"**. The generator
refuses such an interpreter by name rather than emitting goldens from a tower it
could not build.

```sh
/path/to/venv/bin/python scripts/gen-ltx2-gemma-tower-goldens.py \
  --out tests/vllm/models/ltx2_gemma_tower_goldens.inc
cmake --build build --target test_ltx2_text_encoder && ./build/tests/test_ltx2_text_encoder
```

No checkpoint and no download: the reduced config comes from
`tests/vllm/models/ltx2_gemma4_text_config.json`, which is the
`__metadata__["gemma_config"]` of the official bf16 text encoder, and every weight
is rebuilt on both sides from the deterministic stream. The tolerance is not a
constant — the generator MEASURES how far upstream's own answer moves between f32
and bf16 and emits that per state as the bound.

Two more gates want the real checkpoint. The prompt-token goldens are regenerated
from the tokenizer the text encoder ships **as a tensor**, and the end-to-end case
dequantizes the 12B tower to roughly 24 GB of host bf16, so it is opt-in rather
than checkpoint-presence gated:

```sh
TE=$CHECKPOINT_ROOT/ltx-2.5/vonkaiser-fp8-nvfp4/text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors
/path/to/venv/bin/python scripts/gen-ltx2-prompt-tokens-goldens.py \
  --text-encoder "$TE" \
  --out tests/vllm/models/ltx2_prompt_tokens_goldens.inc

# real vocab, token-exact vs HuggingFace
CHECKPOINT_ROOT=... ./build/tests/test_ltx2_text_encoder --test-case="ltx2 prompt: REAL*"

# the full 12B vertical: ~33 GB host, minutes of CPU
CHECKPOINT_ROOT=... VLLM_CPP_LTX2_TOWER_E2E=1 \
  ./build/tests/test_ltx2_text_encoder --test-case="ltx2 e2e*"
```

`VLLM_CPP_LTX2_TEXT_ENCODER` names the file directly when it does not sit under
`CHECKPOINT_ROOT` at the path above.

Recipes resolve on an EXACT `(pipeline_kind, model_version)` pair and refuse
anything else by name rather than defaulting, because a plausible but wrong sigma
schedule or guidance scale renders a video instead of failing. The pairs that
resolve are `one_stage` at 2, 2.3, 2.4 and 2.5, `distilled_two_stage` at 2 and
2.5, and `dmd2` at 2 and 2.3.

`Ltx2Guidance` serves `CFGGuider`, `STGGuider` and `MultiModalGuider`. It refuses
`CFGStarRescalingGuider`, `LtxAPGGuider` and `LegacyStatefulAPGGuider` by name,
because nothing upstream constructs them: all three appear in the Lightricks tree
only at their own `class` statements. Two known gaps in the schedule are open:
`Ltx2SigmaSchedule(1, ...)` returns a NaN first sigma where upstream returns
0.10000002, and the suite's `MaxAbsDiff` drops NaN so a golden alone will not
catch it.

## LTX-2.5 quantized loaders

`include/vllm/model_executor/models/ltx2_loader.h` materializes the shipped
LTX-2.5 checkpoints: the FP8 DiT, both NVFP4 DiTs, and the torchao-NVFP4 Gemma-4
text encoder with its embedded tokenizer. These are the entry points the render
path itself drives: `--dit` (`--video-dit` on the server) reaches
`Ltx2StreamDitToDevice` / `Ltx2LoadDitFromSafetensors` at
`ltx2_video.cpp:576-577`, and `--encoder` (`--video-encoder`) reaches
`Ltx2LoadTextEncoderFromSafetensors` at `ltx2_video.cpp:851`. This section
documents them at the library level, where the gate below runs.

The two NVFP4 checkpoints were written by different producers that disagree about
both the group-scale framing and which nibble holds which weight, so the loader
resolves the producer from the `torchao_nvfp4` marker: present means torchao
(`to_blocked` framing, low-nibble-first), absent means the Lightricks
`nvfp4-prequant` tool (cuBLAS-padded framing, high-nibble-first). A marker whose
stored scale shape contradicts it, and a marker-less file whose shape is the
`to_blocked` framing or neither framing, are refused by name rather than guessed,
because both readings type-check and produce finite, correctly scaled, wrong
weights.

The refusal cannot cover everything, and the limit is worth knowing before you
point this loader at a checkpoint it was not built for. A marker-less NVFP4 file
whose `weight_scale` is stored **linear** `[N, K/16]` — what ModelOpt,
llm-compressor and compressed-tensors write, none of which emit a
`torchao_nvfp4` sidecar — has, whenever `N % 128 == 0` and `K/16 % 4 == 0`, a
shape indistinguishable from the cuBLAS-padded one. Such a file is resolved as
`nvfp4-prequant` and read swizzled and high-first: it loads, and it is wrong.
Only the LTX-2.5 DiT is gated against an independent oracle here, so treat any
other marker-less NVFP4 checkpoint as unsupported until it is. See
`.agents/specs/nvfp4-nibble-order.md`.

Two behaviours a caller has to know. `Ltx2LoadDitFromSafetensors` REFUSES the
shipped DiT by default, because that file carries **one** module family this port
does not carry (`keyframes_abs_pos_embedding`); pass
`Ltx2DitLoadOptions::allow_unported_modules`
to load the ported subset, which still reports every one of them in
`Ltx2DitCheckpoint::unported`. `prompt_adaln_single` and
`audio_prompt_adaln_single` were on that list until 2026-08-13 and are now
PORTED, so a checkpoint carrying them needs no opt-in on their account, and the
opt-in no longer disables them. The two `*_embeddings_connector` towers are
**not** among them and never will be:
`UnportedFamilies` filters them out at `ltx2_loader.cpp:439` (`LoadedElsewhere`),
`RefuseUnported`'s own message says so in capitals at `ltx2_loader.cpp:461-464`,
and `Ltx2LoadConnectorWeights` loads them under their own contract — which is
what the video engine calls, so a checkpoint this port reads completely is never
made to ask for `allow_unported_modules` on their account. (The "five" this
paragraph used to say arrived at `5966ffef3` and was true until `e48c86253`
added `LoadedElsewhere` — the same claim the "what runs" section above already
retired, which survived here because it was never swept for.) And loading is
**bf16** by default, the checkpoint's own model dtype; `widen_to_f32` is opt-in
and exists only for the f32 parity forward.

`Ltx2StreamDitToDevice` is the GB10 arm. It dequantizes and uploads one tensor at
a time so peak residency is the device copy plus one tensor, and it stages at
load because host-resident weights measure 20 to 30 percent slower there.

The gate needs the three checkpoint headers, a vLLM checkout and an LTX-2
checkout (the two nibble-order authorities); it reads a few hundred bytes at
their own offsets and never a payload:

```sh
python3 scripts/gen-ltx2-quant-goldens.py --vllm ~/_git/vllm --ltx2 ~/_git/LTX-2 --checkpoint-root /mnt/nas_share/checkpoints --out tests/vllm/models/ltx2_quant_goldens.inc
cmake --build build --target test_ltx2_loader && ./build/tests/test_ltx2_loader
```

## SSE keepalives on long prefill

Async chat/completion streams may emit SSE **comment** frames (`:\n\n`) while
waiting on the engine (long prefill / TTFT). Interval is `VT_SERVER_SSE_PING_S`
(default 15s; `0` disables). Comment frames are not `data:` events and do not
carry tokens. Token streaming still uses a timed wait on the request collector
so deltas are not collapsed by a poll loop.

## Gemma4 FP8 on ROCm (RDNA4)

Dual-GPU resident FP8 MoE and SharedK-WMMA prefill are controlled via
ENVIRONMENT.md (`VT_GEMMA4_RESIDENT_*`, `VT_ATTN_*`). Defaults stay safe off RDNA4.
This PR does **not** restructure the Gemma-4 layer loop or enable decode hipGraph
(those stay lab-only until a CUDA token-exact gate can land them).

## LTX-2.5 text conditioning

This documents **one brick of the shipped render path** — the text conditioning
the DiT consumes — and how to reproduce its gate. The render itself is above
under [LTX-2.5: what runs, and what it cannot do](#ltx-25-what-runs-and-what-it-cannot-do);
`--encoder` is what puts this brick on that path, and `has_encoder` is set at
`ltx2_video.cpp:893` once the tower loads.

LTX-2.5 does not condition on a text encoder's last hidden state. It takes every
Gemma-4 hidden state (the embedding output plus all 48 decoder outputs, 49 in
total), normalizes them, concatenates across the layer axis, and projects the
result twice: a 4096-wide video caption projection and a 2048-wide audio one.
That is why the shipped projections take 3840 x 49 = 188160 inputs.

Two things about the shipped checkpoint are easy to trip over:

* the tokenizer is stored **as a tensor**, `tokenizer_json`, alongside
  `hf_asset__*` sidecars, so a loader that expects a sibling `tokenizer.json`
  file cannot read it;
* `vonkaiser/LTX-2.5-FP8-NVFP4`'s text encoder carries **no** safetensors
  `__metadata__` block, so the Gemma config has to be supplied out of band.
  `Ltx2LoadGemmaAssets(file, /*require_config=*/false)` is the opt-out; the
  default refuses, exactly as upstream does.

Reproduce the parity gate (CPU only, no checkpoint and no gated download; needs
torch, numpy and einops plus a Lightricks LTX-2 checkout):

```sh
python3 scripts/gen-ltx2-text-goldens.py \
    --ltx2 ~/_git/LTX-2 \
    --out tests/vllm/models/ltx2_text_goldens.inc
cmake --build build --target test_ltx2_text_encoder
./build/tests/test_ltx2_text_encoder
```

The generator imports the upstream modules by path and executes them at reduced
dimensions; both sides rebuild every weight from one deterministic stream, so no
weight byte is checked in. It also runs four degenerate inputs through upstream
and emits each one's full output tensor, not a "still finite" flag, because the
normalization epsilons and the width they are added in are invisible to a random
fixture. The mean's denominator is one of those: upstream adds it in float32
(`sequence_lengths * d` is an int64 tensor and `eps` a python float, which
promotes to the default dtype), so computing it in float64 is finer arithmetic
and the wrong answer.

A third thing to know if you are wiring a loader to it: the feature extractor
refuses, by name, any disagreement between what the checkpoint config declares
and what the weights actually carry. That covers the declared bias against
`bias.empty()`, the declared `out_features` against the weight's own width, and
`embedding_dim x (num_hidden_layers + 1)` against the weight's `in_features`. The
case worth naming is a loader that binds `video_aggregate_embed.weight` (U8,
NVFP4) and misses `.bias` (BF16, so a different unpack path) while the config
still says the projection is biased. Without the refusal that renders a plausible
video for the wrong prompt: every conditioning row is shifted by the missing bias
and every padded row projects to 0 instead of to the bias.

## MiniMax-Music3: the checkpoint loader

**It loads, it does not generate.** `include/vllm/model_executor/models/`
`minimax_music3_loader.h` is phase W1 of #672 — it resolves the shipped
`diffusers` layout, parses the six component configs, and accounts every tensor
in the files against what those configs owe. No forward, no scheduler step and
no audio; those are W2-W7, and nothing below produces a song.

Point it at the **diffusers arm**, the six-component tree:

```
minimax-music3/
  modular_model_index.json
  transformer/           config.json + 2 shards + index   441 tensors  F32
  condition_encoder/     config.json + 1 file               4 tensors  F32
  rvq_depth_decoder/     config.json + 1 file              47 tensors  BF16
  vocoder/               config.json + 1 file             121 tensors  F32
  language_model/        config.json + 4 shards + index   399 tensors  BF16
  scheduler/scheduler_config.json
  tokenizer/
```

`MiniMaxMusic3ResolveCheckpoint` refuses anything else **by name**, and the
refusal you are most likely to hit is the useful one. The same repository also
ships a **native** arm — `qwen_7B/qwen_7B/`, `flowmatching_vae.pth`, `dav.pth` —
which SGLang-Omni serves and which holds every weight this port needs in a layout
nothing here reads. Pointed at that tree the loader names it as the native arm,
lists the diffusers components it lacks, and tells you to convert it with
diffusers' `scripts/convert_minimax_music3_to_diffusers.py`. It is never
silently mis-loaded.

Two things the loader enforces that a correctness gate later could not catch:

**On-disk dtype and runtime dtype are different things, and the loader keeps
them apart.** The files store F32 for the transformer, condition encoder and
vocoder and BF16 for the RVQ depth decoder and language model, and
`MiniMaxMusic3AccountTensors` refuses a file that disagrees. That set is *not* a
runnable configuration. Upstream casts in exactly two places, `denoise.py:83`
(condition encoder output into the transformer) and `decoders.py:84` (latents
into the vocoder), and never on the way in: `denoise.py:82` hands the language
model's hidden states to the condition encoder with a device move and no dtype
move. So the autoregressive half must share one dtype, and loading the on-disk
set raises `Input type (c10::BFloat16) and bias type (float) should be the same`
from `condition_embedder_minimax_music3.py:64`.

`MiniMaxMusic3ResolveRuntimeDtypes` answers the runtime question.
`kBf16ArFp32Acoustic` is the gated configuration: language model, depth decoder
and condition encoder in bf16, transformer and vocoder in fp32.
`MiniMaxMusic3CheckRuntimeDtypes` refuses a violation by name, listing all three
autoregressive components with their dtypes, because upstream's own error names
a bias dtype and never says which component disagreed with which.
`kAsStored` is kept selectable so that failure stays reproducible; it is
reported as not runnable rather than quietly repaired.

**The vocoder's weight norm is folded at load.** Its 30 weight-normed
convolutions ship as torch's legacy `weight_g`/`weight_v` pairs;
`MiniMaxMusic3LoadVocoderWeights` collapses each to a single `<module>.weight`
through `vocoder1d::MaterializeWeightNorm`, so no `_g`/`_v` name survives and
nothing downstream can read the direction `v` as if it were the weight. Four of
the thirty are `ConvTranspose1d`, whose weight is `[C_in, C_out, K]` — torch
reduces over dimension 0 either way, which for those four is the *input* channel.

### Running its gate

The suite needs no checkpoint. `tests/vllm/models/minimax_music3_manifest.inc`
carries the real checkpoint's own safetensors headers — 1012 entries of names,
dtypes and shapes, no weight bytes — and every geometry claim is asserted
against it:

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 8 --target test_minimax_music3_loader
./build/tests/test_minimax_music3_loader
```

One test case additionally exercises the real 27 GB tree when you name it, and
loudly skips when you do not:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_loader
```

Regenerate the manifest after a checkpoint revision moves — it reads headers
only, so it does not stream the weights:

```sh
python3 scripts/gen-minimax-music3-manifest.py \
  --checkpoint /path/to/minimax-music3 \
  --output tests/vllm/models/minimax_music3_manifest.inc
```

### IndexTTS-2.5 goldens and checkpoint manifests

The speech lane is not servable yet (see `/v1/audio/speech` above); these
regenerate its gates. `read-torch-manifest.py` reads a torch `.pth`'s tensor
names and shapes from its pickle header over HTTP range requests, so it inspects
a multi-GB checkpoint without downloading the weights:

```sh
python3 scripts/read-torch-manifest.py \
  https://huggingface.co/IndexTeam/IndexTTS-2.5/resolve/main/s2mel.pth
```

The stage goldens need the upstream source checked out, and emit `.inc` files
that carry no weight bytes: both sides rebuild parameters from one shared
pseudo-random stream.

```sh
WAVENET_SRC=/path/to/index-tts/indextts/s2mel/modules \
  python3 scripts/gen-wavenet-goldens.py --out tests/vllm/models/wavenet_goldens.inc

DIT_SRC=/path/to/index-tts/indextts/s2mel/modules \
  python3 scripts/gen-dit-tail-goldens.py --out tests/vllm/models/dit_tail_goldens.inc

DIT_SRC=/path/to/index-tts/indextts/s2mel/modules \
  python3 scripts/gen-dit-front-goldens.py --out tests/vllm/models/dit_front_goldens.inc

DIT_SRC=/path/to/index-tts/indextts/s2mel/modules \
  python3 scripts/gen-dit-stack-goldens.py --out tests/vllm/models/dit_stack_goldens.inc

BIGVGAN_SRC=/path/to/index-tts/indextts/s2mel/modules/bigvgan \
  python3 scripts/gen-bigvgan-goldens.py --out tests/vllm/models/bigvgan_goldens.inc

CODEC_SRC=/path/to/index-tts/indextts \
  python3 scripts/gen-codec-encoder-goldens.py --out tests/vllm/models/codec_encoder_goldens.inc

python3 scripts/gen-w2v-fbank-goldens.py --out tests/vllm/models/w2v_fbank_goldens.inc
```

The U-Net skip routing is recorded rather than generated into an `.inc`: this
prints the schedule upstream's own Transformer actually performs, at several
depths, and the expected values are quoted in `tests/vllm/models/test_dit_skip.cpp`.

```sh
python3 scripts/gen-dit-skip-schedule.py /path/to/index-tts/indextts/s2mel/modules
```

Convert the checkpoints once, then point the loader gate at the result to check
the real weights (it is skipped, loudly, when the variable is unset):

```sh
python3 scripts/convert-indextts2-checkpoint.py \
  --checkpoint $CHECKPOINT_ROOT/IndexTTS-2.5 \
  --out $CHECKPOINT_ROOT/IndexTTS-2.5-safetensors \
  --manifest tests/vllm/models/indextts2_pth_manifest.json

VLLM_CPP_INDEXTTS2_S2MEL=$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors/s2mel.safetensors \
  ./build/tests/test_indextts2_s2mel_loader

VLLM_CPP_INDEXTTS2_GPT=$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors/gpt.safetensors \
  ./build/tests/test_indextts2_talker_loader
```

## MiniMax-Music3: the autoregressive half

Phases W2 and W3 of #672.
`include/vllm/model_executor/models/minimax_music3_ar.h` is what consumes three
of W1's six components: the prompt the `language_model` is driven with, the
semantic stage's classifier-free-guidance logit pipeline, the learned 8-layer
condition mix, and the 4-layer RVQ depth decoder. **It still does not generate a
song** — the DiT, the scheduler and the vocoder are W4–W5, and the 8.6B
`Qwen3ForCausalLM` forward itself is the remainder of W2.

### The token gate the spec promised does not exist

Worth stating plainly, because the spec said otherwise until this phase measured
it. MiniMax-Music3's autoregressive stage has **no greedy path**:
`_sample_top_k` (`encoders.py:94-103`) is the only sampler either stage uses, it
has no temperature and no argmax branch, and it ends in
`torch.multinomial(probs, 1, generator=generator)`. The oracle's
`rvq_codes.npy` is a *seeded sample*, so matching it token-for-token would be
reproducing torch's RNG rather than this model. Independently: both stages sample
from a CFG mix of a conditional and an unconditional row, and the goldens store
the conditional row only, so the guided distribution is not reconstructible from
what is committed.

The codes are therefore **inputs** to these gates, and the AR half is gated on
tensors.

### Running the gates

The reduced-dimension gate needs no checkpoint. Its goldens come from executing
upstream's own `MiniMaxMusic3ConditionEncoder` and `MiniMaxMusic3RVQDepthDecoder`
at small dimensions in float32, so it isolates an algebra defect from rounding:

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 8 --target test_minimax_music3_ar
./build/tests/test_minimax_music3_ar
```

The full-scale gate drives the real bf16 weights on the oracle capture's own
inputs and skips loudly without the checkpoint:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_ar_real
```

It compares 176 128 values for the condition mix (against `condition_chunk0.npy`)
and 716 800 for the depth decoder (against `frame_hiddens[:, 4096:]`, 25 frames ×
7 depth steps), and it reports the counts rather than only a verdict.

Regenerate the reduced-dimension goldens with the pinned oracle's interpreter
(see `tools/oracle/README.md`) after an upstream change:

```sh
~/venvs/music3-oracle/bin/python scripts/gen-minimax-music3-ar-goldens.py \
  --out tests/vllm/models/minimax_music3_ar_goldens.inc
```

### Two things that will bite a later phase

**The code rows are offset by one from the frames.** `rvq_codes.npy` is `[26, 8]`
and `frame_hiddens` is `[25, ...]`: row 0 of the codes is the priming decode step,
which emits no frame (`encoders.py:342`). `rows[1:]` align with the frames.
Comparing the unshifted sequences yields two individually plausible tensors and a
wrong gate.

**`ArCompute` is not a precision knob.** The autoregressive half runs bf16, and a
bf16 torch module rounds at *every* op boundary, so an fp32 host forward is a
different computation rather than a more precise one — measured, it leaves
448 450 of 716 800 values beyond one bf16 ULP. `ArCompute::kBFloat16` mirrors the
rounding; `kFloat32` is the reduced-dimension goldens' dtype. A caller at
`kBFloat16` also owes its weights at bf16, *including* the condition encoder,
whose file is fp32 while its runtime is not.

And bit-exactness against torch is not on offer here, which is worth knowing
before a later phase spends a day chasing it. torch's bf16 `nn.Linear` on CPU
reproduces to 32 759 of 32 768 values, but its dispatched attention reproduces to
only 25 736: the CPU kernel runs a blocked online softmax, and four candidate
rounding models (pre-scaled q, bf16-rounded scores, bf16-rounded probabilities,
and their combinations) were all *worse* than the plain form. The full-scale
bound is therefore
calibrated against torch's own `sdpa_kernel(MATH)` arm on the identical inputs
(46.34% bit-identical, mean absolute error 1.659e-03) rather than against a
bit-exactness that no second implementation can reach.

## MiniMax-Music3: the acoustic half

Phases W4 and W5 of #672.
`include/vllm/model_executor/models/minimax_music3_acoustic.h` is the rest of the
pipeline: the 2.4B fp32 flow-matching DiT, the `FlowMatchEulerDiscreteScheduler`
with `invert_sigmas`, the classifier-free-guidance mix, the denoise loop's
overlapping-window bookkeeping, and the DAC Flow-VAE vocoder that turns latents
into a **44100 Hz stereo** waveform. **It still does not generate a song end to
end** — joining the two halves through `SpeechRegistry`, the `vllm_speech_*` ABI
and the example server is W6, and the 8.6B `Qwen3ForCausalLM` forward is the
remainder of W2.

Configs are W1's (`MiniMaxMusic3TransformerConfig`,
`MiniMaxMusic3VocoderConfig`, `MiniMaxMusic3SchedulerConfig`) rather than new
ones, and every convolution, transposed convolution, pad and activation is a
call into the shared `vllm::vocoder1d` primitives. Nothing in `vocoder1d` is
modified, so MiniMax-H3 and IndexTTS-2.5 are byte-identical.

### There is no token gate on this half, and that is not a gap

A flow-matching denoise loop has no logits, no vocabulary and no sampler, so no
token gate exists to have. (That is a *different* fact from the autoregressive
half's withdrawn token gate above, which was withdrawn because upstream has no
greedy path there. Two withdrawals, two causes.) What binds instead is per-stage
tensor parity against the oracle capture, each stage against its own entry.

### Running the gates

The reduced-dimension gate needs no checkpoint. Its goldens come from executing
upstream's own `MiniMaxMusic3Transformer1DModel`, `MiniMaxMusic3Vocoder`,
`FlowMatchEulerDiscreteScheduler` and `ClassifierFreeGuidance` at small
dimensions in float32:

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 8 --target test_minimax_music3_acoustic
./build/tests/test_minimax_music3_acoustic
```

The full-scale gate drives the real fp32 weights on the capture's own inputs and
skips loudly without the checkpoint. Its scheduler and vocoder cases run in
about ninety seconds:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_acoustic_real
```

The **DiT** cases are opt-in behind a second variable, because they load 9.1 GB
of fp32 weights and run four 2.4B forwards on the host — about fifteen minutes,
not ninety seconds:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 VLLM_CPP_MUSIC3_DIT=1 \
  ./build/tests/test_minimax_music3_acoustic_real
```

Regenerate the reduced-dimension goldens with the pinned oracle's interpreter
(see `tools/oracle/README.md`) after an upstream change:

```sh
~/venvs/music3-oracle/bin/python \
  scripts/gen-minimax-music3-acoustic-goldens.py \
  --out tests/vllm/models/minimax_music3_acoustic_goldens.inc
```

### Three things that will bite a later phase

**float32 here is not a precision knob either, but it is the opposite polarity
from the AR half.** The acoustic half runs fp32 because upstream does; there is
no `Compute` parameter, because there is no second configuration. Separately,
and on a different axis: every reduction accumulates in `double` and stores
`float`, which is the tree's existing host-reference convention
(`vocoder1d::Conv1d`, `music3::LinearNoBias`) and costs no memory. Short
*elementwise* expressions — the sigma shift, the Euler step, the CFG mix, the
overlap blend — are computed in `float` on purpose, because upstream computes
them in float32 and the results are bit-exact there. Widening those to double
produces a different number: `shift * s / (1 + (shift - 1) * s)` at shift 3 is
`0.100000024` in float32 and `0.100000001` in double, and the goldens say the
former.

**A close-enough bound on an exactly-reproducible quantity hides real defects.**
Measured here: at a 1e-5 relative tolerance, dropping upstream's `(1 - 1e-6)`
factor from the overlap blend moves values by only 3.3e-07 relative and the
mutation stays **green**. The blend has no reduction, so its gate is bit-exact
instead. The same reasoning makes the Euler step and the DiT-to-vocoder handoff
bit-exact assertions rather than tolerances.

**The stereo fold is a contiguous split, not an interleave.** The 128 latent
channels reshape into two 64-channel streams: the *first* 64 become the left
channel and the second 64 the right, and each stream is decoded independently by
the same weights (`minimax_music3_vocoder.py:110,115`). Interleaving them is the
other obvious reading of "fold 128 into 2 x 64" and produces a correctly shaped,
correctly ranged, wrong waveform that no length or dtype check can see.
