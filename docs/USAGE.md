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

### Host compilers

gcc 13 and 14 and clang are exercised by CI, and **gcc 16 builds the tree,
including the OpenAI server**. Before this it did not: several files, one of
them the server's own `main`, called `getpid()` without including `<unistd.h>`
and compiled only because an older libstdc++ happened to pull that header in
for them. A compile-only CI lane on the newest released gcc now guards this,
because every other Linux lane uses the distro compiler and cannot see it.

On gcc 16 the `array-bounds` warning is reported but is **not** treated as an
error, unlike on every earlier gcc. That release emits it inside libstdc++ and
the vendored JSON library for code that is correct, and no change to the
calling code avoids it (`cmake/CompilerWarnings.cmake` explains the mechanism
and cites the upstream gcc bug). A genuine out-of-bounds still fails the build
on gcc 15 and earlier, which is what the rest of CI enforces.

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

ROCm now also carries an **engine-level attention backend name**. Until #1056 the
kernels were registered (`kPagedAttention`, `kReshapeAndCache`) but
`RocmPlatform::get_attn_backend_priority` returned an empty list, so
`SelectAttentionBackendName` had nothing to resolve for `kROCM` — ROCm was the
only platform in that state. It now returns upstream's dense order verbatim, and
`ROCM_ATTN` is registered against the NHD layout this tree uses. Nothing routes
to that name until the runner asks for it (#1065), and no user-facing flag
changes: this is what the engine picks, not something you select.

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

### A DISABLED feature removes its kernels, not the ops that do not need it

`cutlass-fp8: DISABLED` means this build has no CUTLASS sm120 FP8 **GEMM**. It
does not mean the build has no FP8. The static per-tensor activation quant
`vt::QuantFp8Static` is a hardware `e4m3` convert with no CUTLASS dependency, so
it is compiled and registered on **every** CUDA architecture
(`src/vt/cuda/cuda_quant_fp8.cu`), and the cuBLASLt FP8 GEMM it feeds is
registered unconditionally too. FP8 W8A8 checkpoints therefore load and run on a
CUDA build with no CUTLASS at all: `-DVLLM_CPP_CUTLASS_DIR` and
`-DVLLM_CPP_CUTLASS_FETCH` are not required for that path.

Until [#960](https://github.com/mudler/vllm.cpp/issues/960) the quant shared a
translation unit with that CUTLASS GEMM, so it inherited the GEMM's architecture
set and was simply absent on `110`. The engine then ran the portable CPU fallback
over device pointers and the process died with `SIGSEGV` after printing

```text
[vt reference-tier] op=QuantFp8Static device=cuda has NO native kernel; running the PORTABLE CPU fallback (correct but slow)
```

If you ever see that banner naming an op on a `cuda` device, this build is
missing a kernel it needs. Report it — it is not a slow path, and the message's
"correct but slow" is not true when the device is not the CPU
([#844](https://github.com/mudler/vllm.cpp/issues/844)).

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

`scripts/agent-preflight.sh` now also runs `scripts/check-symbol-anchors.py`,
which reads every citation written as `` `path/to/file.cpp::SymbolName` `` and
requires that the file it names still contains that symbol. Write citations in
that form rather than as `file.cpp:412`: a line number is a coordinate into a
moving file, so an edit anywhere above it retargets the citation in files the
edit never opened. Add `--upstream-root <vllm-checkout>` to ask the same
question of the pinned oracle; that run is opt-in, because CI has no checkout to
resolve upstream paths against. Both runs print every bucket they left out --
frozen files, untracked files, upstream paths -- and refuse a checked count
below the recorded floor, so a run that quietly stopped examining anything
cannot report as a pass.

The operator role is a coordinator, and **several may run at once**:
`scripts/agent-role.py claim operator` records this worktree and is never
refused, `scripts/agent-role.py show` lists the other live coordinators, and
`scripts/agent-role.py release` removes only this worktree's record. What keeps
concurrent coordinators safe is that `main` is never force-pushed, so a plain
`git push` refuses any non-fast-forward.

### `.env`: your values, and what happens when it is missing

`.env` is untracked, so a fresh clone and every linked worktree start without
one. `scripts/agent-start.py` reports that as `environment: missing`,
`incomplete`, or `unreadable`, and prints what to do about it. The route is ask
and then record. It never guesses a value, and it never falls back to a host
name or a path written in a repository document, because that is another
developer's resolved value.

Record one answered value with the writer that owns the file:

```sh
scripts/agent-onboard.py --env-set GATE_HOST=my-gate-box
```

It seeds `.env` from `.env.example` on first use, so every other key survives
commented and empty, and it refuses any key `.env.example` does not declare.
Leave a key empty when your setup does not have the thing. Empty means
unavailable, and the gates that need it stay `PENDING` for you.

Three keys name where the hardware gate runs, and a gate script refuses by name
rather than guessing when one it needs is unset:

| Key | Value |
|---|---|
| `GATE_HOST` | The box the hardware gates run on |
| `GATE_DEVICE` | Its resource-controller device, as `<box>:<device>`, for example `dgx:gpu0`. `rc devices` lists the fleet |
| `GATE_CHECKOUT` | The repository checkout on that box, which remote gate commands enter before they build |

`SHARED_STORAGE_ROOT` names the mount point of shared storage when it is a
network share, and `CHECKPOINT_ROOT` names the checkpoint directory inside it.
The two are separate because a leased worker or a container can see the same
folder under a different path.

### `GPU_LOCK`: one file mutex, and only one

Copy `.env.example` to `.env` and load it with `set -a; . ./.env; set +a`. Every
key there may be left empty to mean "my setup does not have this" — **except
`GPU_LOCK`**, which ships a real default:

```sh
GPU_LOCK=$HOME/gpu.lock
```

On a shared box, every GPU job takes that file for the whole job or the whole
benchmark series:

```sh
flock "${GPU_LOCK:-$HOME/gpu.lock}" -c '<command>'
```

Do not point it somewhere else. A mutex only works if everyone opens the **same
file**, and `flock` on a different path *succeeds* — that is what a mutex does —
so a divergent value serialises you with nobody and never says so. The damage
shows up much later as timing noise, and it does not read as "my number is
wrong", it reads as "someone else misbehaved": a whole benchmark series was lost
to this, with every absolute timing downgraded to an upper bound because only
interleaved ratios survive contention (#777). Every script in this repo falls
back to the same default, so change it only if every agent and harness on the
box moves with you.

If your `.env` predates this default and names another path, fix it by hand —
`.env` is untracked, so a shipped default cannot reach it.

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
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. Every key is checked and none is dropped: an unknown or misspelled name is refused at startup by name, and a real vLLM key this engine does not implement is refused as such ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)). See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
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
Qwen3.5 checkpoint has been RUN here — see [STATUS.md](STATUS.md) for the owed
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

Every build — not only a Vulkan one — additionally gets `vocoder-conv-ab`, the
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

### Running the vocoder convolutions on the GPU

`VLLM_CPP_VOCODER_DEVICE=cuda` routes `vt::Conv1d` and `vt::ConvTranspose1d` to
their CUDA providers for every model that decodes through the shared vocoder
core. It needs a CUDA build; asking for it without one throws by name rather than
falling back silently, because a silent fallback means an operator who asked for
a device never learns they did not get one.

The knob is not CUDA-specific. It accepts any device name `vt` knows (`cpu`,
`cuda`, `metal`, `vulkan`, `xpu`, `rocm`, `tenstorrent`) and refuses one whose
device carries no registered provider in the build in front of it, so a Metal or
Vulkan provider becomes reachable here by being registered and nothing else.

The default is `cpu`, and deliberately so — not because the device arm is
approximate. The two providers are **byte-identical**: one f64 accumulator per
output element walked in the same order on both, with the host pinned
`-ffp-contract=off` and the device kernel pinned with `__dmul_rn`/`__dadd_rn`, so
`tests/vt/test_ops_conv1d_general.cpp` gates them with `memcmp` rather than a
tolerance (8 cases / 385 assertions on Jetson Thor sm_110, against 8 / 347 on a
CPU-only box — the 38-assertion difference IS the device arm). It stays opt-in
because flipping four shipped audio models onto a device arm needs its own
re-gate against each one's committed goldens, which is owed to the row that
wires it ([#672](https://github.com/mudler/vllm.cpp/issues/672),
[.agents/specs/minimax-music3.md](../.agents/specs/minimax-music3.md) §13).

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

### Block-wise FP8 runs on CPU and refuses on CUDA

Block-wise FP8, also called fine-grained FP8, keeps one scale for each 128x128
block of a weight rather than one scale for the whole weight. A block-wise
checkpoint declares `quantization_config.weight_block_size` in its
`config.json` and stores its scales under `weight_scale_inv` rather than under
`weight_scale`.

`Qwen/Qwen3.8-27B-FP8` is such a checkpoint. At revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` it declares `weight_block_size`
`[128, 128]` with `activation_scheme` `dynamic`, and it stores
`self_attn.q_proj.weight` as `F8_E4M3` `[12288, 5120]` beside
`self_attn.q_proj.weight_scale_inv` as `BF16` `[96, 40]`.

That checkpoint now RUNS on a CPU queue. Ten projections of the Qwen3.5 dense
model — `q_proj`, `k_proj`, `v_proj`, `o_proj`, the Gated-DeltaNet
`in_proj_qkv`, `in_proj_z` and `out_proj`, and the MLP's `gate_proj`, `up_proj`
and `down_proj` — quantize their activation per token per 128-wide group and
then run a block-scaled GEMM whose scales apply in the mainloop, once per
K-block, into an F32 accumulator. Each of the ten emits BF16, which is the
model dtype and what vLLM emits at the same sites.

Those ten projections are seven GEMMs, because `gate_proj` and `up_proj` run as
one and `q_proj`, `k_proj` and `v_proj` run as one — the same two merged linears
vLLM builds. A block scale belongs to a 128-row band, so the shards' scale grids
concatenate exactly and the merged GEMM is byte-identical to the separate ones.

That merge needs each projection in a group except the last to be a multiple of
128 rows wide, which is what vLLM requires of the same checkpoints. A checkpoint
that breaks the rule is refused by name, and the message says which projection
and how wide it is, rather than quietly running a different arithmetic:

```text
block-wise FP8 merged 'qkv_proj': shard 'k_proj' has out_features 64, which is
not a multiple of the quantization block's n 128. Only the LAST shard of a
merged block-quant linear may be ragged
```

On a device with no block-scaled GEMM the model refuses while it is being
prepared, before the first forward and before any CUDA graph is captured:

```text
block-wise (fine-grained) 128x128 FP8 weights LOADED for
model.layers.0.self_attn.q_proj and there is no block-wise FP8 GEMM on device
'cuda'. The linear method and the dense forward wiring are implemented and the
CPU reference GEMM executes them, so this checkpoint runs on CPU today
```

What exists on CPU is a correctness reference. It makes no speed claim, and no
token-exact comparison against vLLM on this checkpoint has been recorded,
because the GPU arm that would run one does not exist yet. Milestone M5 of
[#1189](https://github.com/mudler/vllm.cpp/issues/1189) owns the
mainloop-scaled CUTLASS kernel; [#1166](https://github.com/mudler/vllm.cpp/issues/1166)
is the original report.

One lever is incompatible with this arm. `VT_KV_CACHE_F32=1` selects an F32
paged KV cache while `v_proj` keeps emitting BF16, and the KV write requires
both to share one dtype, so it refuses. That affects every BF16 arm rather than
this one; it is tracked as
[#1249](https://github.com/mudler/vllm.cpp/issues/1249). Leave the lever unset,
which is the default.

Two block-wise configurations are refused earlier, at load, because no build
here implements them: an `activation_scheme` other than `dynamic`, and a
`weight_block_size` other than `[128, 128]`. Both messages name the key and the
value your `config.json` declares.

To run this model on a GPU today, use a per-tensor FP8, BF16, NVFP4, or GGUF
checkpoint of it.

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

### One load refusal that is about this code, not your checkpoint

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

### Architectures that resolve but refuse to run

A few architectures are registered so their config and weight layout are
accounted for, while their forward is deliberately not implemented. Pointing the
CLI or server at one of these loads far enough to resolve the architecture and
then fails with a message naming the missing piece, rather than emitting wrong
tokens quietly.

| Architecture | Why it refuses |
|---|---|
| `KimiK3ForConditionalGeneration` | Needs ~1.56 TB (MXFP4); no host here can run it |
| `NemotronHForCausalLM` | **Only BATCHED decode still refuses.** A2-P (#810) narrowed this: `ForwardNemotronHForCausalLM` now selects the paged forward whenever the runner supplies paged KV and recurrent state, so K/V go into the runner's pages and the conv/SSM rows are carried across steps, and `examples/nemotron_h_gen` reaches all of it through `include/vllm.h` alone. What is left is `num_reqs > 1`, refused by name because one request's pages and one request's recurrent state are carried per step and a multi-request step would be decoded as ONE concatenated causal sequence — plausible wrong tokens rather than a failure. Owed to A2-B. **The end-to-end token gate against the pinned oracle has NOT run**, so no claim is made here about what this checkpoint emits; `docs/BENCHMARKS.md` records that as pending rather than as silence. A2-Q1 puts the 23 FP8 W8A8 Mamba2 blocks on the device, so `lm_head` is the last arm that computes on the host, and a GGUF file is refused by name since no GGUF arm exists for it. See *Nemotron-3.5-Lightning-30B: the exact weights, and which arms run* below |

This is a deliberate state, not a bug: registering the architecture is what lets
the config parse and weight-name mapping be tested before the forward exists.

A refusal here is always a thrown message you can read. Every registered
architecture also refuses when it is handed a model some other architecture
loaded, naming both itself and the architecture the passed model claims, instead
of reading that model as though it were its own (#775, swept across the
remaining 34 entry points in #847). Where two architecture names share one
implementation — `Olmo2ForCausalLM` and `Olmo3ForCausalLM`, or
`LlamaForCausalLM` and `InternLM3ForCausalLM` — the refusal names the family's
primary architecture as the one that refused, and the alias you asked for as
what the passed model claimed.

### LTX-2.5: what runs, and what it cannot do

LTX-2.5 is reachable as video family `ltx-2.5`, through the same
`vllm_video_engine_load` / `vllm_video_generate` C ABI that serves MiniMax-H3,
and through the `ltx2-gen` example that drives it. Its two VAE decoders, its two
VAE ENCODERS with the mel front-end, the conditioning items that place encoded
latents into the token stream, and its pipeline layer (the sigma schedule, the
diffusion steps, guidance, the latent spatial x2 upsampler, the duration head and
the embeddings connector) are implemented and gated. The latent **temporal** x2
upsampler is implemented and gated too, but no pipeline here drives it — see the
`--upsampler` note below. Several limits decide what you can actually ask for,
and each refuses by name rather than rendering something else.

**Image conditioning (image-to-video) runs at `image_crf=0`, and only there.**
Pass a first frame as binary PPM (`first_frame_path` / `first_frame_ppm`) plus
the per-generation extra `image_crf=0`; the engine decodes it, aspect-fills and
centre-crops it to each phase's own resolution, VAE-encodes it, and replaces
latent frame 0's clean tokens. `noise_aug` is the pinning strength (`1.0`, the
default, pins the frame exactly).

`image_crf=0` must be asked for **explicitly**, and it is **out of
distribution**. Upstream re-compresses a conditioning image through H.264 at the
CRF the checkpoint's generation was trained with, and an LTX-2.5 checkpoint
resolves that to **18**. That round trip needs libx264 and no codec is vendored
here, so a non-zero CRF — including the default a caller gets by saying nothing —
is refused by name. `image_crf=0` is upstream-legal (upstream short-circuits it
and documents an explicit `0` as "skip re-compression entirely") but conditions
the model on pixels it was not trained to see. That is a render-quality cost, and
it is stated rather than applied silently.

A **last-frame keyframe is served** as of the token-APPEND seam. A keyframe is
*appended* to the token sequence with its own pixel positions, denoised as part
of a longer sequence, and trimmed back off before the latent is unpatchified,
where the first-frame arm only REPLACES tokens that already exist. It takes the
same `image_crf=0` and `noise_aug` as the first-frame arm, and both may be
supplied at once. Two things a previous version of this paragraph got wrong are
worth naming, because a reader may have acted on them: there is no rebuilt
attention mask — a supplied keyframe passes `attention_mask=None` and upstream
returns no mask for it — and the sigma schedule keeps reading the TARGET token
count rather than the grown one, because upstream derives its shift from the
unpatchified target. (Until 2026-08-13 this paragraph said a last-frame keyframe
needs the DiT's unported `keyframes_abs_pos_embedding`. That was wrong: a
supplied keyframe is appended unmarked, so the embedding never applies to it.
Where the embedding does bite is the FIRST latent frame of every render, which
was a separate gap; it was closed on 2026-08-14 under issue #658, so the marker
is now applied on every render.)

**Generated keyframe slots are a different feature, and they are now SERVED.**
Upstream also lets the model *generate* extra frames at interior positions,
`--num-generated-keyframes N` there and the per-generation extra
`num_generated_keyframes` here. That is not a keyframe you supply; it is one you
ask the model to invent, and each slot buys one pixel frame at the cost of a
full latent frame of tokens. `0` is upstream's own default and means off, so
passing it explicitly renders normally. A positive count places that many
evenly spaced INTERIOR slots: both endpoints are dropped, because frame 0
already spans a single pixel frame under causal encoding and the last frame is
the clip's own end. The slots are marked with the trained keyframe embedding,
denoised with the video, and read back out of the state before the extra tokens
are trimmed away.

Two refusals remain, and they are upstream's own rather than ours. A negative
count is refused, and so is a count the clip is too short for: every slot is an
interior position, so `N + 2` frames are the minimum.

**This page said until 2026-08-16 that a positive count was refused, and it is
recorded rather than deleted** because a reader may have planned around it. The
refusal named the readback as its one blocker, and it was right: what landed
under issue #986 is the layout that locates the slots exactly and the extraction
that runs before the trim. One third of what that refusal named is still owed,
and it is a different surface rather than a smaller version of this one, the
standalone single-frame decode that would hand you slot PIXELS. Nothing here
returns those: the slots stay in latent space, which is what DFR below wants
from them.

Reference-image, reference-video and reference-audio conditioning are still
refused, each naming a different missing piece. **Two reasons this page used to
give are now false and are recorded rather than deleted**, because a reader may
have planned around them: the IC-LoRA scale factors are read as of `--lora`
(2026-08-15), and the token-APPEND machinery landed with the last-frame keyframe
above (2026-08-16). What is left for reference VIDEO and reference IMAGE is a
pixel path and a stage split. Nothing here turns a clip into latents: upstream
decodes the reference at `height/downscale x width/downscale`, keeps frame 0 and
then every Nth frame, and encodes the result
(`ltx_pipelines/iclora_utils.py:87-89`, `:112-148`), and this engine's only
pixel-to-latent route encodes exactly one frame at the phase's full resolution.
And the reference item belongs to stage 1 only: upstream fuses the adapter into
stage 1 and gives stage 2 `loras=()` and no reference item at all
(`ic_lora.py:108`, `:119`, `:314-321`), while this engine holds ONE DiT, fused
at load, that every phase runs. Reference audio additionally needs the AUDIO
VAE's encoder key filter, which is not built. Three encoder-level limits are worth
stating in advance because they are refusals rather than approximations. A
reference waveform whose sample rate differs from the audio VAE's is refused
rather than resampled, since upstream uses a polyphase kaiser resampler this
project does not carry. A VAE configured with `latent_log_var: none` is
refused, because upstream itself raises on it. And a video-VAE `res_x` encoder
block that declares no `num_layers` is refused rather than defaulted, because
upstream subscripts that key and raises `KeyError` on it; no other encoder block
kind reads it.

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
absmax. When the request carried an image it also reports the CRF and strength it
was conditioned at, how many tokens the encoded image replaced, and a digest over
**those tokens as written into the state** — not over the encoder's output, so a
build that encoded an image and never placed it reads as unconditioned rather
than healthy. It is returned **by value, under the engine's own lock**, so it is safe to
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
         --model-version 2.5 \
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

Add `--first-frame frame.ppm --image-crf 0` for image-to-video. The PPM is
binary P6 at maxval 255 (no PNG/JPEG codec is vendored); `--image-crf 0` is
required and is not the default, because omitting it resolves the checkpoint's
own CRF 18 and refuses — see the out-of-distribution note above.

Add `--audio-path take.wav` for **audio-to-video**: the render is conditioned on
a soundtrack you supply rather than one the model invents. The take is encoded
through the audio VAE's encoder and then held frozen through every denoise
phase, and the `audio.wav` that comes back is your own input rather than a VAE
round trip. `--audio-start-time` seeks into the file and `--audio-max-duration`
caps how much is read; both default to covering exactly the clip's duration, and
either without `--audio-path` is refused rather than ignored.

What is upstream's here is the **conditioning mechanism** — decode, encode,
truncate to the clip, freeze — and not the denoise schedule. Upstream's
audio-to-video stage 1 is a caller-configured guided one, with its
`a2v_guidance_scale` acting as the guider's modality scale, while a take here
rides whichever recipe the checkpoint resolves, in practice `distilled_two_stage`
with fixed sigmas. So the audio drives the render, and no claim is made that the
result reproduces upstream's own audio-to-video output.

The WAV has to match the checkpoint already: 16-bit PCM RIFF/WAVE, the audio
VAE's own sample rate (16 kHz on the shipped one), its encoder's channel count
(2), and at least as long as the clip. None of the four is converted. There is
no resampler for an arbitrary ratio here and no demuxer at all, and a take
shorter than the clip is an error upstream too, so each mismatch is refused with
both numbers in the message — a resampled-wrong, upmixed or silence-padded take
renders a finished clip conditioned on audio nobody supplied. This needs an
audio VAE that carries encoder weights; a decoder-only one refuses by name.

#### The supported resolution envelope

**`--width` and `--height` are enforced, and an unsupported value is refused by
name.** Both must be multiples of the VAE's spatial factor (32) times the worst
downscale the recipe's phases apply — so **64 on the distilled two-stage recipe**,
whose first phase runs at half resolution, and **32 on a one-stage recipe**. Those
are upstream's own two numbers (`assert_resolution`,
`ltx-pipelines utils/helpers.py:540-551`), reached by upstream's derivation rather
than hardcoded, so a recipe that downscaled further would tighten the divisor with
it. The refusal names the offending axis — width, height, or both — the divisor,
and a size you can actually pass: the nearest legal one at or below the request,
or, when an axis is smaller than the divisor and no such size exists, the
smallest legal size there is.

Until 2026-08-15 nothing enforced this and the engine floored instead: a
two-stage request of width 80 rendered 64 and returned success, and a one-stage
request of width 100 rendered 96 ([#919](https://github.com/mudler/vllm.cpp/issues/919)).

**`--frames` is NOT enforced, and it rounds.** A frame count is floored onto the
VAE's temporal grid, `(frames - 1) / 8 * 8 + 1`, so 100 frames renders 97. This
mirrors upstream, which floors an explicit `num_frames` identically
(`ltx_core/types.py:113`) and validates it nowhere: its `snap_frames_to_grid`
helper is called from the auto-duration path and from the dubbing pipeline, and
that pipeline takes no frame count at all — it snaps one read from a reference
video's container. No frame count a caller supplies is snapped or checked, in
either project. Pass a value of the form `8k + 1` to get exactly what you asked
for. The rounding is observable either way: `result.frame_count`, `result.width`
and `result.height` report what was actually rendered, not what was requested.

Omitting all three renders the recipe default, which is 1024x1536 at 121 frames
and is a much larger request than it looks.

**What is legal is not what fits.** The first two rows below are a property of
this port and are enforced. The rest are scale markers, and the last three are
measurements of one box rather than limits of the code:

| | Value |
|---|---|
| Legal sizes | any multiple of 64 (two-stage) or 32 (one-stage), on both axes |
| Legal frame counts | any; non-`8k + 1` values floor onto the temporal grid |
| Upstream's default output | 1024x1536 at 121 frames (`utils/constants.py:42-76`) |
| Upstream's HQ preset output | 1088x1920 at 121 frames (`utils/constants.py:95-98`) |
| **Measured to complete on one GB10** | **704x448 at 25 frames** in 4231 s, 448x256 at 25 frames in 3085 s, and 320x192 at 25 frames. One run each, 16 to 17 August 2026, `main` `0b0b8900f` |
| Largest size tried | 704x448 at 25 frames. 1024x576 was not attempted to completion because another session claimed the box. That is scheduling and not an envelope, so 704x448 is not a ceiling |
| Superseded, kept for the record | 448x256 at 25 frames was published here as *not* completing, on a run that lost about 59 GB in 24 s after its denoise. It completes, and that loss did not recur |

Those three completions are one run each on one contended box, with no oracle on
either side, so read them as what has been observed and not as a limit. There is
no maximum-size check anywhere in this path.

The 59 GB stays on the page because it is the reason the old row gave, and it
belongs to its own run: a prompt-embeds render with no text tower that an armed
watchdog ended at 13.77 GiB against an 18 GiB floor, rather than the engine
failing. That run is rung F1 in `.agents/benchmark-record.md`. The loss was never
attributed to the decode, whose own heap peak at that size is 361.72 MiB, some
170x too small, and attributing it is still open as
[#1014](https://github.com/mudler/vllm.cpp/issues/1014). It did **not** reproduce
on `0b0b8900f` under a 2 s memory guard that would have seen it: the 448x256 rung
floors `MemAvailable` at 38.96 GiB over 1289 samples and the 704x448 rung at
38.89 GiB over 1743 samples, with no sample under 34 GiB on either and a peak use
of 80 of 119 GiB. See the note below on what bounds a render, and
`.agents/specs/ltx25-tiled-decode.md` and
`.agents/specs/ltx25-resolution-envelope.md`.

`--lora ic-lora.safetensors [STRENGTH]` fuses an IC-LoRA adapter into the DiT at
load, mirroring upstream's `--lora PATH [STRENGTH]`
(`ltx-pipelines/utils/args.py:600-611`). The strength is optional and defaults to
1.0. It is a LOAD-time flag, not a per-request one, because the adapter is fused
into the weights and cannot vary between generations - upstream takes it as a
`DiffusionStage.from_checkpoint` constructor argument for the same reason
(`ic_lora.py:104-114`).

The adapter is a safetensors file of `.lora_A.weight` / `.lora_B.weight` pairs,
with or without ComfyUI's `diffusion_model.` prefix. It works on every arm the
DiT loads - bf16, FP8 and NVFP4 alike - because those are all dequantized to
bf16 before the delta is added. Two things REFUSE by name rather than
proceeding quietly: an adapter naming a module this port does not bind (upstream
would skip it, and a skip cannot be told apart from a typo), and an adapter that
fuses into nothing at all.

**A second `--lora` does NOT refuse, and this page said it did until 2026-08-17.**
Only one adapter is accepted, and the library enforces that
(`ltx2_lora.cpp:243-248` fails on more than one, citing `dubit.py:364-365` and
`hdr_ic_lora.py:271-272`). But `ltx2-gen` cannot construct the two-adapter vector
that trips it: `SetExtra` (`examples/ltx2_gen/main.cpp:212-221`) overwrites an
existing key in place, so `--lora a --lora b` leaves one `lora_path` extra
holding `b`, silently fuses `b`, and exits 0. Pass one adapter.

The C ABI cannot reach it either, and that is the wider half of the finding:
`Ltx2VideoEngine::Load` carries the ONLY `dit_options.loras.push_back` in the
tree and it runs at most once, under `if (!lora_path.empty())` — named by symbol
rather than by line, because the line moved with #1118 and a stale anchor is what
this paragraph already had to correct once. So `loras.size()` is 0 or 1
on every production path — CLI, `vllm_video_engine_load` and the server alike —
and the more-than-one refusal is reached only by `test_ltx2_lora`. It is correct
code guarding a state nothing can currently construct, which is the shape
N-adapter fusion ([#932](https://github.com/mudler/vllm.cpp/issues/932)) will
need. Tracked as [#1097](https://github.com/mudler/vllm.cpp/issues/1097).

Supplying an adapter also reads its `reference_downscale_factor` and
`reference_temporal_scale_factor` metadata (`iclora_utils.py:30-49`). Those are
what a reference video needs, and reading them was what the reference refusal
used to say was missing. It no longer says that, and it does not say
token-append either: that seam landed too. What it names now is the reference
CLIP's own pixel path and the stage split, both above.

`--upsampler` is what the distilled recipe's second phase needs. Without it that
phase refuses rather than skipping: its three-step refinement is what makes the
upscaled latent valid, and decoding the half-resolution latent instead would hand
back a smaller clip that looks like a completed request. `--max-phase 0` stops
after the first phase deliberately.

It must be the **spatial** upsampler,
`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`. Lightricks also ships
`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`, which is the same
class with `temporal_upsample: true` in its config and the same
`upsampler.0.*` tensor names — so it loads and runs, and returns a latent with
`2f - 1` frames at the ORIGINAL resolution where this phase needs the original
frame count at double resolution. It is `2f - 1` and not `2f` because that arm
doubles the frame axis and then drops the first frame, which upstream encodes as
a single pixel frame. Passing it is refused by name rather than
reported as a shape mismatch. The temporal arm itself is implemented and gated
against upstream, but **nothing drives it**: its only upstream consumer is
`DFRPipeline`'s multi-round loop. The DFR pipeline's BASE is ported as of issue
#986 and is described below, and the rounds loop is not, so there is still no
flag that makes a request use that file and no reason to pass it today. The
checkpoint is also not published beside the spatial one on the mirror this port
was built against, so nothing here has run it on real weights.

### The DFR pipeline: `--pipeline-kind dfr`

Detail-fidelity rendering. It is upstream's `DFRPipeline`, and it differs from
the ordinary distilled two-stage recipe in its CONDITIONING rather than in its
schedule: both stages run the same sigmas, and stage 1 is the same half
resolution. What DFR adds is a keyframe grid.

**The canvas is padded, and this is the part that surprises people.** DFR lays
keyframes on a segment grid, 24 or 32 frames per segment, whichever pads least,
and it pads `num_frames - 1` up to a whole number of segments before it renders
anything. A 9-frame request therefore denoises a 25-frame canvas and is trimmed
back to 9 before you see it. Ask for 121 frames and you get 121; ask for 9 and
the machine does about three times the work you might expect.

**Frame counts are refused here rather than floored.** Everywhere else in this
engine a frame count that is not `8k + 1` is floored onto the latent grid, and
that is documented above as the behaviour. DFR cannot live with it: every
keyframe position it emits has to land on a latent border, so `--frames 10` is
refused with the reason rather than quietly rendered as 9.

**`num_generated_keyframes` is refused on this pipeline.** DFR chooses its own
slot positions from the canvas, one per segment boundary, and the whole pipeline
is indexed by that grid. An override would leave the slots and the canvas
describing different frames, and the render would still finish. Use
`--pipeline-kind distilled_two_stage` or `one_stage` if you want to place slots
by count. An explicit `0` still passes, because that is upstream's default.

**How to reach it.** `pipeline_kind` is a LOAD knob, not a per-generation one, so
all three surfaces carry it: `ltx2-gen --pipeline-kind dfr`, the C ABI's
`vllm_video_model_params.extra_keys` / `extra_values`, and the server's
`--video-extra pipeline_kind=dfr` at launch. A server started that way renders
every `/v1/videos` request through DFR.

The two knobs beside it are per-GENERATION and therefore **ABI only**, because
`/v1/videos` forwards no per-generation extra to any engine yet (issue #928):
`num_generated_keyframes` on the other pipelines, and `temporal_upsample_rounds`
below. This paragraph said "CLI and ABI only" until 2026-08-17, and the CLI half
was never true — `examples/ltx2_gen/main.cpp` carries no flag for either name, so
`vllm_video_gen_params.extra_keys` is the only surface that reaches them.

### LTX-2.5 text-to-audio: a render with no picture

`--pipeline-kind t2a_one_stage` runs upstream's `T2AOneStagePipeline`, which
generates a soundtrack and no video at all. The result carries an `audio.wav`,
`frame_count = 0`, an empty frame directory and **no ffmpeg argv**, because there
is nothing to mux.

```sh
ltx2-gen --dit ltx-2.5-dit.safetensors \
         --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
         --encoder gemma4-12b-with-proj.safetensors --encoder-config gemma4.json \
         --pipeline-kind t2a_one_stage --device cpu \
         --frames 121 --prompt "rain on a tin roof, distant thunder" \
         --workdir /tmp/t2a
```

**`ltx2-gen` has no `--steps` flag, and this recipe carried one until 2026-08-17.**
The step count comes from the resolved recipe (`ltx2_video.cpp:2900`), and the
`vllm_video_gen_params.num_inference_steps` field that would override it
(`include/vllm.h:1072`) has no flag on this binary — `minimax-h3-gen` and
`music3-gen` both expose `--steps`, which is where the published line came from.
An unknown argument is not ignored here: `examples/ltx2_gen/main.cpp:318-321`
prints `unknown argument` and exits 2, so the command as published could not run
at all. Overriding the step count needs the C ABI today.

**These file names are not a checkpoint pin, and no LTX-2.5 recipe in this
document is.** None of them names a HuggingFace repo, a revision or a sha256,
which AGENTS.md § *Say which weights, and from where* requires; MiniMax-H3 and
MiniMax-Music3 below each carry a full table and LTX-2.5 carries none. That is
campaign-wide and pre-existing rather than particular to this recipe, and it is
recorded rather than invented, because no LTX-2.5 arm here has been rendered on
real weights yet. Tracked by
[#1048](https://github.com/mudler/vllm.cpp/issues/1048); read `--dit` above as
"the LTX-2.5 transformer", which the other recipes on this page spell as
`ltx-2.5-22b-distilled-fp8.safetensors` together with the `--dit-config` its
missing `__metadata__` requires.

**No `--video-vae` is needed**, and none is loaded: upstream's pipeline never
constructs a video VAE. `--width` and `--height` are **refused** rather than
ignored — upstream passes a 512x512 placeholder whose height and width it
documents as unused, and only the frame count and the recipe's frame rate are
read, to derive the duration.

**It is a GUIDED arm, and that changes what it costs and what it needs.** The
distilled video recipes run one DiT forward per step. This one runs **three** by
default — conditional, unconditional, and one with the audio
self-attention perturbed (STG) — so it is roughly 3x the work per step, and it
**requires a text tower**, because the unconditional pass conditions on the
negative prompt. Loading with `prompt_embeds_path` alone gets a refusal naming
`--audio-cfg-guidance-scale 1.0` as the way to turn the unconditional pass off.

It was the only guided arm here until row LTX25-GUIDED-VIDEO
([#1092](https://github.com/mudler/vllm.cpp/issues/1092)) gave the joint video
path its own denoiser; see *LTX-2.5 video guidance* below.

Six per-generation knobs mirror upstream's own CLI, and each takes the
checkpoint generation's value when absent: `--negative-prompt`,
`--audio-cfg-guidance-scale` (7.0), `--audio-stg-guidance-scale` (1.0),
`--audio-rescale-scale` (0.7), `--audio-skip-step` (0) and `--audio-stg-blocks`
(28 on the 2.3-and-later lineage), which is comma separated. A block index
outside the DiT's own layer count is refused rather than clamped. There is no
`modality_scale` knob: upstream pins it to 1.0 for this pipeline, because
audio-only generation has no video modality to isolate.

`--audio-rescale-scale` acts on the **denoised (x0) prediction**, not on the
DiT's velocity, because upstream's guider sits behind an `X0Model` and combines
already-converted tensors. The distinction is invisible at `0.0`, where the two
readings agree exactly, and it changes the render at every other value — so a
recipe or a script that was tuned against the velocity reading will not
reproduce here at the default `0.7` (issue #1039).

Being per-generation, those six reach the CLI and the C ABI and **not**
`/v1/videos`, which forwards no per-generation extra to any engine (issue #928).
`pipeline_kind` is a LOAD knob and does reach the server, so a server started
with `--video-extra pipeline_kind=t2a_one_stage` renders every request as audio
at the recipe's own guider values.

**The accelerator is refused by name.** `device = 1` gets a refusal on this
pipeline: the device forward takes both streams by reference and this pipeline
has no video stream to give it. Use `--device cpu`.

### LTX-2.5 video guidance: `--pipeline-kind one_stage`

`one_stage` mirrors upstream's `TI2VidOneStagePipeline`, which builds a
`FactoryGuidedDenoiser` from the params table's own video and audio guiders. On
the 2.4/2.5 lineage those resolve to `cfg_scale = 3.0`, `stg_scale = 1.0`,
`rescale_scale = 0.7` and `modality_scale = 3.0`.

Until [#1092](https://github.com/mudler/vllm.cpp/issues/1092) this port read none
of it: the joint denoise loop ran one unguided forward per step. A `one_stage`
render therefore finished, at the right size and frame count, along a different
trajectory than upstream's. It now runs **four** forwards per step and combines
them per modality:

| Pass | What differs | Selected by |
|---|---|---|
| conditional | nothing | always |
| unconditional | the negative conditioning | `cfg_scale != 1.0` |
| perturbed | video/audio self-attention skipped on `stg_blocks` | `stg_scale != 0.0` |
| isolated modality | the audio<->video cross attention off in every block | `modality_scale != 1.0` |

Seven per-generation knobs mirror upstream's `default_1_stage_arg_parser` and
each takes the checkpoint generation's value when absent. The audio row and
`--negative-prompt` are shared with text-to-audio and are no longer refused on a
video pipeline; upstream's parser carries both rows side by side, and the old
refusal rested on a reading of upstream that was wrong and harmless only while
nothing here read them.

| `ltx2-gen` flag | per-generation extra | meaning |
|---|---|---|
| `--video-cfg-guidance-scale` | `video_cfg_guidance_scale` | video `cfg_scale`; `1.0` turns the unconditional forward off |
| `--video-stg-guidance-scale` | `video_stg_guidance_scale` | video `stg_scale`; `0.0` turns the perturbed forward off |
| `--video-rescale-scale` | `video_rescale_scale` | video `rescale_scale`, applied to the DENOISED prediction |
| `--video-skip-step` | `video_skip_step` | `0` never skips; `n` runs every `n+1`-th step |
| `--video-stg-blocks` | `video_stg_blocks` | comma separated block indices; EMPTY disables STG, see below |
| `--a2v-guidance-scale` | `a2v_guidance_scale` | video `modality_scale`; `1.0` turns the isolated-modality forward off |
| `--v2a-guidance-scale` | `v2a_guidance_scale` | audio `modality_scale` |
| `--negative-prompt` | `negative_prompt` | the unconditional forward's conditioning |

The audio row is the same six spellings with `audio_` in place of `video_`:
`audio_cfg_guidance_scale`, `audio_stg_guidance_scale`, `audio_rescale_scale`,
`audio_skip_step`, `audio_stg_blocks`, and `v2a_guidance_scale` for its
`modality_scale`.

Those extras ride the per-generation `extra_keys` / `extra_values` array on
`vllm_video_params`, so the C ABI reaches the same path with no new field. They
are per-GENERATION and therefore reach the CLI and the C ABI and **not**
`/v1/videos`, which forwards no per-generation extra to any engine
([#928](https://github.com/mudler/vllm.cpp/issues/928)). `pipeline_kind` is a
LOAD knob and does reach the server, so a server started with
`--video-extra pipeline_kind=one_stage` renders every request through the guided
denoiser at the recipe's own guider values and no request can change them.

**An EMPTY `--video-stg-blocks` is accepted and means "perturb no block".** That
is upstream's own idiom — `docs/multimodal-guidance.md:13` says "Set to `[]` to
disable STG", the field defaults to `[]`, the flags are `nargs="*"`, and the
shipped HQ params row uses it — and it stays distinct from OMITTING the flag,
which takes the params table's value. It disables the STG signal and not the STG
cost: upstream selects the perturbed pass from `stg_scale` alone, so the forward
still runs and contributes exactly zero. Set the scale to `0.0` to skip the
forward as well. This page and this port refused the empty list until
2026-08-17.

**The unconditional forward needs a negative conditioning, and there are two
ways to supply one.** With a text tower, `--negative-prompt` (or the recipe's
own default) is encoded through the same chain as the positive prompt. Without
one, `--negative-prompt-embeds` and `--negative-audio-prompt-embeds` — the LOAD
extras `negative_prompt_embeds_path` and `negative_audio_prompt_embeds_path` —
are the negative half of the `prompt_embeds_path` fallback: two files at the
DiT's two cross-attention widths, the same row count as the positive pair. Being
LOAD extras they DO reach the server, through `--video-extra`. With neither, a
`cfg_scale` other than 1.0 is **refused by name** rather than served the positive
context twice, which would leave the whole classifier-free term at exactly zero.

**A block index the checkpoint does not have is refused**, which is the case the
empty list above is NOT. `stg_blocks` is a membership test upstream, so naming
block 28 on a model with fewer blocks perturbs nothing and leaves
`stg_scale * (cond - perturbed)` at exactly zero — the same zero, reached by a
request that disagrees with the checkpoint rather than by a caller who asked for
no perturbation. Upstream never meets it because it only ships 48-block
checkpoints, so this refusal is local to this port and is named as such.

**The distilled and retake recipes refuse every one of these flags.** Their
guidance is distilled into the weights, so honouring an override would sample a
trajectory the weights were never trained for. Their guiders are upstream's
positive-only one, so they still issue one forward per step and their output is
unchanged by this row.

**The accelerator is refused for the perturbed and isolated-modality passes.**
`Ltx2DitForwardDevice` takes no perturbation argument, so those two passes on
`device = 1` would run an unperturbed forward and leave both terms at zero.
Classifier-free guidance alone is a different context and no perturbation, and
runs on both arms.

**What is not served.** `temporal_upsample_rounds` is defined and refused above
`0`: the rounds loop that temporally doubles the latent, re-tiles the canvas and
stitches it back is not ported. The refusal names it, and it names three things
that are NOT the reason, because each is the one a reader reaches for first: the
temporal upsampler operator is ported and gated, the canvas and tiling
arithmetic is ported and gated in this same change, and the generated keyframe
slots are served. What has no counterpart here is the per-tile denoise pass as a
callable. Stage 2's x2 spatial detailing IC-LoRA is refused separately, for the
reasons the reference-video arm is refused above.

On the server, `--video-family ltx-2.5` pins the family instead of detecting it,
and `--video-extra KEY=VALUE` (repeatable) carries the same family-specific load
knobs the flags above map onto. Both are described under
[the server's video flags](#video-family-and-family-specific-load-knobs).

**Three things about that command are worth knowing before you run it.**

*It is bounded by HOST WALL CLOCK, well below the recipe's own defaults.*
Staging the 21.00B FP8 transformer costs about 44 GB on a 119 GB GB10, and
`--encoder` adds the text tower on top of that — roughly 24 GB of host bf16 that
stays resident, because a prompt arrives per request. Every memory figure here
was measured WITHOUT the tower, on the prompt-embeds path, so budget for both.
**320x192, 448x256 and 704x448 at 25 frames all complete** through both distilled
phases. The upper two took 3085 s and 4231 s, measured on 16 to 17 August 2026 at
`0b0b8900f`. This page used to say 448x256 did not complete, and that is what
changed. Unified memory makes those host bytes and this class of box reboots
rather than OOM-killing, so start small and grow, and put a memory watchdog in
front of anything larger. Those runs kept one at a 2 s cadence and it never came
near firing: the `MemAvailable` floor was 38.9 GiB and no sample fell under
34 GiB. The recipe default of 1024x1536 at 121 frames is far beyond what one
GB10 holds today.

Expect tens of minutes, not seconds, and expect much of that to be independent of
the resolution you asked for. Most of a render is no longer the host VAE decode.
[#1041](https://github.com/mudler/vllm.cpp/issues/1041) threaded that decode, and
what dominates now is a **single-threaded phase of about 1731 s that barely moves
with size**: 1731 s and 1732 s across two rungs whose voxel counts differ by
2.75x, which is 57 to 66% of wall on each. Which phase that is has not been
identified, and [#1087](https://github.com/mudler/vllm.cpp/issues/1087) owns
naming it. The decode itself still has no device arm and still runs at 0% GPU
([#1007](https://github.com/mudler/vllm.cpp/issues/1007)).

Read every figure in the last two paragraphs as one run per geometry on a shared
box that was contended, with no oracle on either side. Two rungs establish no
scaling law, and 704x448 is not a ceiling: the next rung up was stopped by
another session claiming the box, not by the machine.

The decode is no longer *single-threaded*, which is what this section used to
say. The decode's convolutions now dispatch across `VLLM_CPP_CPU_THREADS` workers
(default `hardware_concurrency`), bit-identical at every worker count —
[#1009](https://github.com/mudler/vllm.cpp/issues/1009), measured at **roughly
9x on 16 to 20 workers** against one. Take the band rather than a decimal: the
medians are 9.15x at 16 and 9.14x at 20, but those two counts spread 21-23% run
to run on a box that was not idle, where every count at or below 8 spreads under
7%. Read it as a decode figure and not a render one: the ~9x was taken on a
synthetic decode shape on a contended 20-core x86 host, and end to end it does
not appear, because the phase #1041 never touched is now most of the wall
(#1087). The renders above are the post-change re-measurement of that wall. Set
`VLLM_CPP_CPU_THREADS` lower if the render has to share the box.

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
asked for.

**The sentence that used to follow was stale and is retired here.** It said
keyframe and reference conditioning were refused because "only the decoder is
ported". The video VAE **encoder** is ported and is kept resident
(`ltx2_video.cpp:1007-1012`), the first-frame and last-frame keyframe arms are
SERVED — the same page says so at the image-conditioning section above — and what
remains refused is REFERENCE conditioning, for reasons that have nothing to do
with the encoder: the reference clip has no pixel path and stage 2 must run
unfused (`ltx2_video.cpp:1955-1990`,
[#975](https://github.com/mudler/vllm.cpp/issues/975)). Reference AUDIO is refused
separately (`ltx2_video.cpp:1991-2004`). A refusal whose stated reason has been
removed is worse than no reason, because a reader plans around it.

**The convolutional decode is TILED and STREAMED, on upstream's own defaults, and
there is no knob.** The layout is the one `ltx_pipelines` builds for a Conv VAE
when you pass `AUTO_TILING`: a 768 px tile with a 64 px overlap on the long side,
aspect coupled to the short one, and 80 frame temporal chunks overlapping by 24.
Each temporal chunk is written to its PPM files and dropped, so the full pixel
volume never exists at once. Two consequences worth knowing before you read a
memory number:

- **Below a 768 px long side and 81 frames the layout does not tile at all.** A
  single tile comes out, and that path reproduces the untiled decode bit for bit
  (`test_ltx2_tiling`'s one tile control, on both causality settings). So
  448x256/25f renders byte identically to how it rendered before tiling existed,
  and its memory is unchanged. Tiling starts doing something at 896x512, and
  temporal chunking at 81 frames.
- **A tiled render is not the same image as an untiled one**, and that is
  upstream's behaviour, not a defect here. Each tile decodes a crop of the latent,
  the decoder's receptive field is wider than the 64 px overlap, and the seam is
  blended rather than eliminated. Do not compare a 1920x1088 render against a
  hypothetical untiled one and read the difference as an error.
- **81 to 120 frames is already the tiled regime, and the recipe default is
  inside it.** The default request is 1024x1536 at 121 frames. At 81 frames the
  latent is 11 frames deep against a 10 frame temporal tile, so it splits into two
  chunks. Measured on the shipped conv VAE at 64x64 / 81 frames: max abs diff
  0.0503 against the untiled decode, on an output whose own max is 0.7513 — 6.70%
  of that range — with 962983 of 995328 channel values (96.75%) not bit identical.
  So nearly every value moves, by a few percent of the signal. If you need the pre
  tiling render back, ask for 73 frames or fewer.

**The refusal that used to stand here is gone, and what replaced it is an owed
ORACLE rather than an owed feature.** Through L10 this page said a prompt was
refused because the `Embeddings1DConnector` weights, which ship inside the DiT
file, were among the modules the DiT loader would not load. They are loaded
(`Ltx2LoadConnectorWeights`, `ltx2_loader.cpp:1292 @ b5756ea8c`, enumerates their
own contract at `:1295`, outside the DiT's),
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
| POST | `/v1/audio/speech` | Text (or lyrics + a music description) to audio; responds with `audio/wav` bytes. Registered **only** when a synthesizer is attached (`--speech-model`) |

The reference-audio side of IndexTTS-2.5 is complete in the library -- a 16 kHz
clip goes through the SeamlessM4T feature extractor, the w2v-bert Conformer, the
layer-17 hidden-state tap, the checkpoint's stored-statistics normalization and
the semantic codec to discrete codes, and the talker's prompt is assembled from
that conditioning plus the text -- but none of it is reachable from a command or
a route yet. The greedy generate loop that turns the prompt into mel codes is
ported too, and so is the STATED-emotion path -- eight weights selecting rows
from the checkpoint's own speaker and emotion matrices by cosine similarity -- so
text plus a reference clip and an emotion reaches mel CODES in the library. What
is still missing is a COMMAND or ROUTE. TEXT DOES REACH AUDIO in the library:
`test_indextts2_e2e` tokenizes with the checkpoint's own vocabulary, runs the
talker to mel codes, and drives those through the length regulator, the CFM loop
and BigVGAN to samples. Point it at all four checkpoint paths:

```sh
VLLM_CPP_INDEXTTS2_S2MEL=... VLLM_CPP_INDEXTTS2_BIGVGAN=... \
VLLM_CPP_INDEXTTS2_GPT=... VLLM_CPP_INDEXTTS2_TIKTOKEN=... \
  ./build/tests/test_indextts2_e2e
```

A REAL LIMITATION to know before using it: the reference clip is required and
then IGNORED. Its encoders are ported and their checkpoints are staged, but the
conditioning rows are zeros, so two different reference voices give the same
output today. `campplus::LoadCampplus` reads its weights but
`campplus::Forward` returns NaN on them, which is an open defect recorded in
the spec and blocks the wiring.

It asserts STRUCTURE, not quality: nothing is compared against vLLM-Omni, which
is unpinned (#633). The TOKENIZER it uses:
`tiktoken::LoadRanks` reads the shipped `.tiktoken` vocabulary and
`tiktoken::Encode` reproduces python tiktoken's ids exactly on the cases
gated, CJK included. The checkpoint now
LOADS through `vllm::multimodal::SpeechRegistry`, reports its family and its
22.05 kHz output rate, and states that a reference clip is required; asking
it to synthesize refuses by naming the one gap between text and the render
path, which is that the shipped vocabulary is tiktoken and this tree has no
reader for one. The pipeline itself renders on the real
checkpoints: the talker emits its own mel codes, the length regulator resamples
them to the mel frame rate, a classifier-free guided CFM Euler loop integrates
the S2Mel estimator, and BigVGAN turns the mel into a bounded 22.05 kHz
waveform. `indextts2::Render` is the entry point, and
`test_indextts2_render` drives it end to end when the three checkpoint
environment variables are set. It is NOT yet measured against the vLLM-Omni
oracle, which is unpinned (#633), so nothing here is a quality claim. Inferring the emotion from a clip instead of stating it needs a
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

`/v1/audio/speech` is registered **only** when the server was started with
`--speech-model`; without it the route is absent (404) and the server is
identical to one built before it existed. See
[Speech and music generation](#speech-and-music-generation).

### Speech and music generation

A **music-only server**, which is what you almost certainly want:

    vllm-server --speech-model /path/to/minimax-music3 \
      [--speech-family minimax-music3] [--speech-device 0|1] [--port 8000]

**`--model` is not required here**, and that is deliberate. Upstream's own recipe
is `sgl-omni serve --model MiniMaxAI/MiniMax-Music3` and nothing else: a music
model is not an accessory to a text model. With `--speech-model` alone this
server loads the music checkpoint, registers `/v1/audio/speech`, and registers
**nothing else** — no `/v1/completions`, no `/v1/chat/completions`. That is the
same task-conditional shape a pooling checkpoint (`/v1/embeddings` only) and a
Parakeet checkpoint (`/v1/audio/transcriptions` only) already take here, and the
same one vLLM's `api_server.py:255-265` uses.

Attach it to a text server instead, and one process serves both surfaces:

    vllm-server --model /path/to/text-model \
      --speech-model /path/to/minimax-music3

`--speech-model` names the checkpoint **set** — MiniMax-Music3 ships six
component directories beside a `modular_model_index.json`, so this is not a
single model directory. `--speech-family` is optional: omitted, the family is
**detected** by inspecting the artifact, and a directory no registered family
claims is refused at startup naming every family that was tried. A name that is
not registered is refused too; it is never treated as a hint, because the wrong
family would not fail — it would render noise. `--speech-family` without
`--speech-model` is still an error: there is nothing to load it from.

`--speech-device` says **where** the family runs. `0` is the default and the CPU
arm; `1` is the accelerator this build resolves. It is refused rather than
substituted: `--speech-device 1` on a build with no accelerator backend, or on a
partial backend that has not registered this family's kernels, fails at startup
naming the piece that is missing. `--speech-device` without `--speech-model` is
an error for the same reason `--speech-family` is — a knob that applies to
nothing reads as one that was honoured. What device 1 currently moves for
MiniMax-Music3 is documented under
[What runs on the device](#what-runs-on-the-device-and-what-does-not), and it is
**not the whole model**.

In the speech-only form the served model name defaults to the **family**
(`minimax-music3`) rather than to a directory basename, because there is no
`config.json` to take one from. `--served-model-name` still wins.

A successful music-only start prints what it resolved, so you can tell a working
server from a listening one without sending a request:

    server: speech/music-only model (family=minimax-music3, 44100 Hz,
            text-only synthesis, family DETECTED, device cpu);
            serving /v1/audio/speech
    server: listening on http://0.0.0.0:8000 (model 'minimax-music3')

`family DETECTED` means the artifact was inspected; `family DECLARED` means you
passed `--speech-family`. `text-only synthesis` is the answer to
`requires_reference_audio()` — a family that needs a reference clip says
`reference clip REQUIRED` there instead, and refuses a clipless request before
anything stages. `device` is what the load **granted**, not what you asked for:
a build that cannot serve `--speech-device 1` refuses at startup rather than
printing `cuda` and running on the CPU.

Or skip HTTP entirely. `minimax-music3-gen` drives the same seam through the C
ABI and writes the WAV itself:

    minimax-music3-gen --model /path/to/minimax-music3 --out song.wav \
      --lyrics @lyrics.txt --description "Genre: acoustic pop. BPM: 96." \
      --duration 8 --steps 8 --seed 7 [--device 0|1]

`--lyrics` and `--description` take literal text or `@path` to read a file,
because lyrics are multi-line and a `[Verse]` tag inside an argv is easy to
mangle. It prints the delivered length, rate, channels, RMS, peak and wall clock
to stderr — the *delivered* length, not the requested one, because a duration
resolves to a whole number of 25 Hz frames and is therefore quantized. It also
prints the device the handle **resolved to** rather than the one `--device`
asked for, which is the difference between timing two arms and timing one arm
twice.

The route is OpenAI's `createSpeech` shape, with the two **music** inputs as
additional named fields:

    curl http://localhost:8000/v1/audio/speech \
      -H 'Content-Type: application/json' \
      -d '{"model": "minimax-music3",
           "lyrics": "[Verse]\nMorning light filtering through the pine\n",
           "description": "Genre: acoustic pop. BPM: 96. Key: C major.",
           "audio_duration": 30, "num_inference_steps": 30, "seed": 7}' \
      --output song.wav

The response body is RIFF/WAVE 16-bit PCM at the family's **native** rate
(44100 Hz stereo for MiniMax-Music3, never resampled), with content type
`audio/wav`.

**Every field, and what it does.** Anything not in this table is refused by name
rather than ignored — see below the table for why that polarity matters here.

| field | type | default | what it does |
|---|---|---|---|
| `lyrics` | string | **required** for MiniMax-Music3 | the sung text, with `[Verse]` / `[Chorus]` section tags. An empty lyric normalizes to a bare `[start]` prompt, so it is a 400 rather than an instrumental |
| `description` (alias `prompt`) | string | **required** for MiniMax-Music3 | genre, BPM, key, instrumentation, mood. NOT a voice or speaker description. Supplying both spellings with different values is a 400, never a silent winner |
| `audio_duration` (alias `duration`) | number, seconds | `60` | resolved to `int(seconds x 25)` autoregressive frames, then **clamped** to the 9000-frame ceiling — the same silent clamp upstream applies (`encoders.py:287`). Shorter than one frame (0.04 s) is a 400 |
| `num_inference_steps` | integer | `30` | flow-matching Euler steps in the acoustic half. Must be > 0 |
| `guidance_scale` | number | `1.7` | classifier-free guidance on the DiT. **0 is legal** and selects the unconditional branch, so omitting the field is how you ask for the default — not sending 0 |
| `seed` | integer | `0` | seeds the autoregressive top-k draw *and* the initial denoise latents. A fixed seed, not a random one: 0 is as deterministic as any other value |
| `model` | string | — | echoed; the route does not check it |
| `response_format` | string | `"wav"` | `"wav"` is the only accepted value |

`lyrics` and `description` are separate fields rather than one `input` behind a
separator because upstream runs a different normalizer over each — `_clean_caption`
on the description, `_normalize_lyrics` on the lyrics (`encoders.py:54-91`). A
one-utterance family keeps using OpenAI's `input`; MiniMax-Music3 refuses it, so
a request cannot half-arrive.

**We expose `guidance_scale` where neither upstream arm does.** In diffusers it
is frozen into the guider component at 1.7 (`denoise.py:180`); in SGLang-Omni it
is a serve-time knob (`dit_cfg_scale`) and not a request field. It is a genuine
per-request control here, and its default is upstream's 1.7.

**Every refusal, and the one rule behind them.** A knob the server will not
honour must not come back behind a 200. Silently dropping one returns audio the
caller did not ask for with nothing to say so — and this project has already paid
for that once (#925), which is why the list is long rather than convenient.

| refused | why |
|---|---|
| `audio_duration_s` | the name of the *field* the key fills, not a key. It is the misspelling you reach for by reading the struct instead of the docs, and dropping it silently returned the 60 s default: 0.1 s became 60 s, 2 autoregressive frames became 1500, and this project's own e2e gate spent four multi-hour runs inside a 750x job it read as a hung weight load (#852, #925) |
| `voice` | no registered family exposes named voices, and there is no enumeration endpoint to pick one from. Upstream refuses it too (`request_builders.py:90-92`) |
| `speed` | no family implements a rate control. Upstream accepts only `1.0` (`request_builders.py:83-89`) |
| `stream`, `stream_format` | MiniMax-Music3 generates the whole song before the first sample exists, so buffering it into chunks would be a stream in name only. **Upstream has no streaming either** — SGLang-Omni declares `supports_streaming_vocoder=False` and rejects `stream=true` by name (`request_builders.py:115-116`) |
| `response_format` other than `"wav"` | no mp3/opus/aac/flac encoder is vendored, and relabelling RIFF bytes is worse than refusing |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | this model's autoregressive stage has ONE sampler — a fixed top-50 draw (`encoders.py:48,94-103`). There is no temperature to set and no nucleus branch to widen, so the knob can be neither honoured nor honestly ignored. Upstream refuses all four (`request_builders.py:14-19,109-114`). Use `seed` to control the draw |
| `max_new_tokens` | SGLang-Omni's spelling of the length, counted in 25 Hz **frames** rather than seconds (`request_builders.py:56-68`). This route takes `audio_duration` in seconds — divide by 25. Two spellings of one meaning on one route is exactly what #925 was |

A family with no text-only synthesis — IndexTTS-2.5 is one — is refused
**before** anything stages: the route asks the loaded engine's
`requires_reference_audio()` and answers 400 naming the family and the missing
`reference_audio`, which is supplied as a `data:` URL carrying a 16-bit PCM mono
WAV.

**Every stage of MiniMax-Music3 is implemented and gated**, and a request
reaches all of them: the 8.6B `Qwen3ForCausalLM` autoregressive stage, the RVQ
depth decoder, the learned condition mix, the flow-matching DiT and the DAC
Flow-VAE vocoder. **A composed request has been observed to completion** — an
HTTP POST returns a real 44100 Hz stereo WAV (#852) — and the end-to-end gate
now runs it over a real socket against a music-only server. There is no by-name
refusal left: nothing here is unimplemented. IndexTTS-2.5 still refuses naming
its own missing pieces.

**What no gate compares is the music itself.** The autoregressive codes are a
seeded `torch.multinomial` draw and the denoise loop's initial latents are a
seeded normal draw, so a request's waveform can never equal the oracle's golden
— twice over, and structurally rather than by omission. Every *stage* is gated
against the capture on the capture's own recorded inputs; a **generated** song
is evidence that the pipeline runs, not that the notes are right. Believe the
stage gates, and listen with that in mind.

**Ask for less than 8 seconds while you are exploring.** `Music3ChunkPlan` only
splits past 200 autoregressive frames, which is 8 s of audio, and the
multi-window composition — the overlap blend, the carry span, the waveform crop
across windows — is this row's one named coverage gap: each primitive is gated
individually, the composition across windows is not, because the oracle capture
is a single 25-frame window.

**It runs on CPU and it is slow.** Every gate this row has was taken on CPU
(`dgx.casa` was down throughout), and the acoustic half is upstream's own fp32.
A 0.1 s request takes tens of minutes; no speed number exists and none is
claimed. Ask for a short duration and few `num_inference_steps` while you are
checking that it works.

The part that dominates is *not* the one you would guess. The 8.6B language
model goes through `vt` and uses the CPU threadpool; the RVQ depth decoder does
not — it is a scalar host loop with a double accumulator, written that way in
W2/W3 so its reduction order is reproducible against torch. In one 0.1 s request
the depth decoder alone is the majority of the wall clock.

At a *real* duration the DiT is the whole story instead, which is why it is the
stage that moved first: a 45 s clip at the default 30 inference steps runs the
DiT 660 times (30 steps x 2 CFG branches x 11 windows) for roughly 634 TFLOP
against about 29 TFLOP for the entire autoregressive half. On the host loops
that is measured in hours. `--speech-device 1` puts it on the accelerator.

#### What runs on the device, and what does not

`--speech-device 1` (or `minimax-music3-gen --device 1`, or
`vllm_speech_model_params.device = 1`) is **a partial arm, and this table is the
whole of it**. Reading it as "the model runs on the GPU" would be wrong in the
direction that matters.

| stage | where `--speech-device 1` runs it |
|---|---|
| 8.6B `Qwen3ForCausalLM` (prefill + every decode step, its paged KV) | **device** |
| guided-logit pipeline, top-k draw, frame feedback embedding | host (two 200 000-wide rows per step; not the cost) |
| **2.4B fp32 flow-matching DiT** (every denoise step, both CFG branches) | **device**, weights staged ONCE |
| 0.646B RVQ depth decoder (7 steps per frame) | **host**, scalar loops |
| condition mix (once per window), scheduler, CFG mix, Euler step | **host** |
| DAC Flow-VAE vocoder (`Conv1d` / `ConvTranspose1d`) | **host**, scalar loops |

The language model reaches the device because it is already routed through the
shared `Qwen3DenseModel` forward that five text registrations ride — nothing was
forked for it, and the only thing this option changes is which queue that
forward is handed and where its KV cache is allocated.

The DiT reaches it the same way: through shared `vt` ops only
(`MatmulBT`, `LayerNorm`, `AttentionCross`, `RopeFromCache`, `SiluAndMul`,
`Add`), with **no new kernel**. Its 9.7 GB of fp32 weights are uploaded once per
request, before the window loop, and the host copy is released as each tensor
lands — a 45 s clip runs that forward 660 times, so a per-step or even
per-window upload would cost more than the compute it enables. `fp32 stays
fp32`: the acoustic half is float32 because upstream chose float32 for it, and
this arm mirrors that rather than buying speed with a narrower dtype.

The remaining stages do not move, for two different reasons, and both are owed
rather than hidden:

* the depth decoder and the condition mix are host `std::vector<float>`
  reference loops under `-ffp-contract=off`, and they run at
  `ArCompute::kBFloat16` — every op's *result* is rounded to bf16, which is what
  upstream stores. Routing them through an f32 GEMM would silently drop that
  rounding, so mirroring them needs bf16 storage, which is a dtype decision with
  its own numeric evidence rather than a transcription;
* the vocoder needs `ConvTranspose1d`, and **`vt` has no such op at all** — the
  1-D convolutions it does have (`vt::DepthwiseConv1d`, `vt::CausalConv1dFwd`)
  are depthwise or causal-with-state, and `vt::Conv2d` and `vt::DepthwiseConv1d`
  are registered for the **CPU only**. There is no CUDA kernel behind the op
  this stage would need, so it is named here rather than hand-rolled outside the
  seam.

Because the host stages are unchanged — and because `--speech-device 0` takes
the same `DitForward` it always did, source byte for source byte — the CPU arm
is **bit-identical** to the one every Music3 correctness gate was taken on. The
device arm's output differs from it exactly where the language model's and the
DiT's own arithmetic differ: two stages, not six, and neither difference is a
shape or an ordering defect. The DiT's device forward is gated against the same
upstream goldens at the same tolerance as the host one; nothing was widened for
it, and `VLLM_CPP_MUSIC3_DEVICE=1` runs that comparison on either arm
(`tests/parity/test_minimax_music3_acoustic_real.cpp`, with
`VLLM_CPP_MUSIC3_DIT=1`).

**The two arms do not produce the same song, and that is structural.** The
autoregressive stage has no greedy path upstream: it ends every draw in a seeded
multinomial, so a different logit changes the drawn code and everything after it.
Do not compare the two WAVs sample by sample. What *is* comparable is the
language model's own hidden state against the oracle capture, and
`tests/parity/test_minimax_music3_llm_real.cpp` runs that comparison on either
arm — `VLLM_CPP_MUSIC3_DEVICE=1` selects the device one, unset is the CPU one —
at the same bounds, with the same negative control. Numbers for both are in
[BENCHMARKS](BENCHMARKS.md).

#### Where the time actually goes: `VLLM_CPP_MUSIC3_PROFILE`

The table above says which stage runs where. It does not say what each one
*costs*, and at a real duration that is the only question anyone asks. Set

```sh
VLLM_CPP_MUSIC3_PROFILE=1 minimax-music3-gen --model ... --duration 20 --steps 30 --device 1
```

and the engine prints a `MUSIC3_PROFILE` table to **stderr** when the request
finishes: one row per stage with seconds, a call count, and its share of the
request, then the resident-set size at each stage boundary.

Read it as follows.

* `leaf` rows partition the request and are the ones that add up. `span` rows
  enclose leaves — `ar.TOTAL_loop`, `denoise.TOTAL` — and are printed for
  context but never summed, so the table cannot claim more work than the run
  contained.
* `cnt` rows carry no time at all. They are the counts a split has to state to
  be readable: frames, windows, requested steps.
* `unattributed` is the glue between the leaves — chunk slicing, the overlap
  blend, the Euler step, the WAV assembly. It is printed rather than spread
  silently over the measured stages, so a bracket in the wrong place shows up as
  a number instead of as a plausible share somewhere else.
* the `calls` column on `denoise.dit_device` counts *steps*, not forwards: one
  bracket covers both classifier-free-guidance branches, so the forward count is
  twice it.
* the `load.ar.*` and `load.ac.*` rows break the two weight loads down further.
  They are spans *inside* the `load.ar_weights` / `load.acoustic_weights` leaves,
  so they are printed and never summed. They exist because the safetensors
  reader is an **mmap** whose tensors are copied out, which interleaves
  page-fault I/O with the copy inside a single call — so the load total on its
  own cannot say whether a slow load is storage or CPU, and `load.ac.dit_build`
  in particular touches no file at all.
* the `calls` column on `ar.depth_forward` and `ar.depth_projection` changed
  meaning in #672, so **two profile tables from different builds are not directly
  comparable on those two rows**. The depth decoder used to re-run the whole
  growing depth sequence at each of the seven codebook steps, separately per
  guidance branch — 14 calls per frame, 70 rows of work to read 14 of them. It
  now appends one position at a time against a K/V cache, both branches in one
  batch-2 call: **8 calls per frame, 16 rows**. The seconds fell with the rows;
  the call count fell for a different reason, and reading the drop in `calls` as
  the speedup would double-count it. The output is bit-identical either way.

It is **off unless the variable is set to `1`, `true`, `on` or `yes`**. Any
other value, including a near miss like `y`, leaves it off — an operator who
mistypes gets a run with no table rather than a run whose meaning quietly
changed. With it off, no clock is read and no `/proc` file is opened.

This is an attribution instrument, not a benchmark harness: it takes no GPU
clock window, so its rows are a within-run **split** and must not be quoted as
per-kernel or cross-box figures.

If you want to price the depth stage on your own box without generating a song,
`tools/bench/music3_depth_stage_ab.cpp` drives one frame of it directly. Nothing
RUNS it — it allocates 2.5 GB and is a two-build A/B, which one target could not
express — but both arms are COMPILED, as the never-linked OBJECT libraries
`vllm_music3_depth_stage_ab_{before,after}`, so it cannot rot behind a signature
change while still being the artifact the §16.6 measurement is reproducible from
(#1246). Its header carries the exact `g++` and run lines. Alternate the arms and
take the minimum; it prints one fingerprint per process, after its round loop, so
a "speedup" that changed the answer cannot be mistaken for one that did not.

**Measured, so expectations are calibrated rather than hoped for.** On a Jetson
Thor (sm_110, 14 cores) the device arm was *slower* on a two-frame request
(846.6 s vs 835.1 s) and 5.4 % faster on a ten-frame one (1430.4 s vs 1512.1 s).
The difference between the arms works out to about 11.7 s saved per
autoregressive frame against a fixed cost of about 35 s, so it breaks even
around three frames — roughly 0.12 s of audio. If you are generating an actual
song the device arm helps; if you are smoke-testing the shortest request that
enters every stage, it does not.

**A first sample, measured.** Two seconds of stereo music, generated by this
engine through `minimax-music3-gen` on an idle-to-busy 20-core x86 CPU box:

| property | value |
|---|---|
| duration | 1.9969 s (88 064 frames per channel) |
| rate / channels | 44 100 Hz, 2 channels, 16-bit PCM |
| RMS | 0.03169 full-scale |
| peak | 0.97437 full-scale, **0 clipped samples** |
| L != R | 84 073 of 88 064 positions, so the stereo fold is real rather than a duplicated channel |
| wall clock | **3286 s** (54.8 min) for 2.0 s of audio, at `--steps 2`, load average 40-150 throughout |

**Its samples are compared to nothing.** The token gate this row once promised
was withdrawn — upstream's autoregressive stage has no greedy path — and a
generated waveform can never equal the oracle's golden anyway, because both the
codes and the initial latents are seeded random draws. So the numbers above
demonstrate that the pipeline runs end to end and produces a well-formed,
non-silent, non-clipped, genuinely stereo signal. They say nothing about whether
the music is right. The per-stage gates are what say that.

The clip is **not committed**: `scripts/check-pr-size.py` classifies every
repository path, and no classified path accepts a `.wav` outside `tests/`, where
a file compared to nothing would sit beside the goldens and imply it was one.
Regenerate it instead — the command above is the whole recipe.

The same seam is reachable from the C ABI at v21 — `vllm_speech_engine_load`,
`vllm_speech_engine_family` / `_sample_rate` / `_requires_reference_audio` /
`_device`, `vllm_synthesize` and `vllm_speech_result_free` — so HTTP and FFI
drive one implementation. `vllm_speech_model_params.device` is the same 0 = CPU
/ 1 = accelerator selector `--speech-device` sets, and
`vllm_speech_engine_device` reports what the load granted. `vllm_speech_result` carries both the float waveform and the
RIFF/WAVE bytes, so an embedder writes a playable file without a second encoder.


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
| `--block-size N` | `32` | KV block size. **Must be a multiple of 16** — the attention backends' `get_kv_cache_shape` refuses anything else, and the server now rejects it at startup rather than throwing during engine init |
| `--num-blocks N` | `0` (auto, resolves to `256`) | KV block count, and vLLM's `num_gpu_blocks_override`. It wins over every other sizing knob. `0` means auto, which uses `--kv-cache-memory` when that is set and otherwise falls back to `256` blocks |
| `--kv-cache-memory BYTES` | `0` (unset) | Absolute KV-pool size in bytes, vLLM's `kv_cache_memory_bytes`. The block count is this budget divided by the model's own bytes per block, summed across its KV groups, so it is correct on MLA and heterogeneous-KV architectures too. It ignores `--gpu-memory-utilization`, as vLLM does. A budget smaller than one KV block is refused at startup |
| `--gpu-memory-utilization F` | `0.92` | **Accepted, and it does not size anything yet.** See [What `--gpu-memory-utilization` does not do yet](#what---gpu-memory-utilization-does-not-do-yet) |
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
| `--offload-config '<json>'` | (unset) | Weight offload, the same JSON vLLM's `OffloadConfig` takes (distinct from `--kv-transfer-config`, which offloads KV blocks). Parsed and validated at startup, so a malformed document, an unknown backend, an unknown TOP-LEVEL key (the four legal ones are `offload_backend`, `uva`, `prefetch` and `vllm_cpp`) or a validator violation is refused before any model I/O; a backend/field mismatch is a warning, as upstream. **Enabling it fails startup on every model today**: no loader consults the offloader, so the engine refuses the configuration by architecture name rather than accept a budget that frees nothing. A config that leaves offloading disabled still parses and reports normally. On unified memory such as GB10 offload cannot help at all, because host and device share one pool. See [docs/WEIGHT-OFFLOAD.md](WEIGHT-OFFLOAD.md). The same document also carries the **`vllm_cpp` key**, which governs the tier BELOW this one — weights borrowed out of the file mapping rather than moved to host RAM — and which is live rather than refused: see [Streaming routed experts from disk](#streaming-routed-experts-from-disk-capacity-mode). A `vllm_cpp`-only document does not enable vLLM's offload backends and is not subject to the refusal above. The flag is accepted by `vllm-server` (the generate/chat and the pooling/embedding paths), by `vllm-cli`, and by the C ABI; the server's transcription-only path REFUSES it by name, because that path builds no engine and could only accept the document and ignore it ([#1195](https://github.com/mudler/vllm.cpp/issues/1195)) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. For `mtp`, `num_speculative_tokens` sets the draft DEPTH and defaults to the checkpoint's `mtp_num_hidden_layers`, which is 1 on both gate checkpoints, so the default is unchanged. A value above it must be a multiple of it, mirroring vLLM. Depth cannot move the emitted tokens under greedy decoding, and no speed number is claimed above k=1 yet ([#81](https://github.com/mudler/vllm.cpp/issues/81)). What is gated on CPU at k=1..4 is that the propose runs `k-1` draft decode forwards per propose call, that k drafts reach the verify path, and that the drafts DELIVERED to the verify path vary with depth rather than repeating the first one. That last one is counted over a RUN and never per call, because a correct drafter may resample the same token and this fixture does. Two things are NOT gated there. A draft is never accepted at depth, because acceptance is zero at every depth on the synthetic gate model. And nothing here proves the draft at depth j came from the j-th forward. Both are owed to the GPU gate, which must close the second by comparing the per-depth acceptance RATE against a PADDED control rather than by asserting a non-zero acceptance count, because a padded drafter earns acceptance at depth whenever the target's own greedy continuation repeats a token. `dspark` speculates on the Qwen3.6 gate models (native + Speculators drafts), token-identically to speculative-off, but is not gated on speed: the cross-engine ratio is UNSETTLED, with a matched-and-warm paired measurement of 0.834x against the pinned oracle and the earlier 0.957x-0.989x figures taken against a single COLD oracle invocation on a machine that has since been reimaged. A GGUF target, or a target with no aux multi-tap, is refused by name (`SPEC-DSPARK`). The DRAFT is classified from its own `config.json` rather than from the method string: `Qwen3DSparkModel`, `Gemma4DSparkModel`, and — BEYOND-PIN, mirroring [vllm#52197](https://github.com/vllm-project/vllm/pull/52197) merged 2026-08-17 — `DSparkDraftModel` together with `model_type` `qwen3` all route to the Qwen3 DSpark lane, and every other DSpark draft that DECLARES an architecture is the DeepSeek-V4 variant, which is refused by name because this engine carries only a stub for it (`SPEC-DSPARK-QWEN3-ROUTING`, [#1193](https://github.com/mudler/vllm.cpp/issues/1193)). A draft config carrying no `architectures` key at all is not classified and loads as before, because an absent key is not evidence of a lane. Its sequential Markov sampling runs on device by default; `VT_DSPARK_DEVICE_SAMPLE=0` restores the host loop (token-identical, cost only). The speculative verify runs from a captured CUDA graph, worth +12.2%/+3.5% on the 35B cells; `VT_SPEC_DECODE_GRAPH=0` restores the eager verify (also token-identical). The object is admitted key by key and NOTHING is dropped ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)): the honoured keys are `method`, `num_speculative_tokens`, `model`, `prompt_lookup_min` and `prompt_lookup_max`, plus `draft_sample_method` and `rejection_sample_method` at their upstream defaults `greedy` and `standard`, which are what this engine implements. Any other value of those two names row `SPEC-ACCEPT-VARIANTS` and is refused. A name vLLM's `SpeculativeConfig` declares but this engine does not implement, such as `quantization`, is refused as exactly that, and any other name is refused as unknown with the accepted list. Before this the extra key was discarded, so `draft_sample_method=probabilistic` ran GREEDY and a misspelled `num_speculatve_tokens` took the default, both silently and both at exit 0. For `dspark`, `num_speculative_tokens` may no longer sit BELOW the draft checkpoint's block: DSpark drafts a block, our block is sized from this value alone, and a shorter one drafted a structurally wrong block in silence. It is refused now, before any weight is loaded, naming the block, the config key the block was read from, and the value given ([#1225](https://github.com/mudler/vllm.cpp/issues/1225)). The block is read from the draft config's `dspark_block_size`, or from `block_size` when that key is absent, which is the case on every published Qwen3 draft (`deepseek-ai/dspark_qwen3_4b_block7` and `RadixArk/Qwen3.8-27B-DSpark` both carry `block_size: 7`, so k must be at least 7). vLLM reads only the first key and accepts the shorter value. vLLM also builds its model config BEFORE its speculative config, so a command that names both a target directory it cannot open and a short `k` hears about the target there and about the `k` here. Those are the two recorded divergences, both argued in `.agents/specs/dspark-block-size-guard.md`. A k at or above the block behaves exactly as before. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--language-model-only` / `--no-language-model-only` | off | Disable all multimodal input by setting **every** modality limit to 0, mirroring vLLM's flag of the same name. It is not a "skip the encoder" switch: the server then **refuses** a multimodal request with ``400 At most 0 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit.`` It does **not** free VRAM yet — nothing gates tower construction on it ([#607](https://github.com/mudler/vllm.cpp/issues/607) wave L3) |
| `--limit-mm-per-prompt '<json>'` | (unset ⇒ 999 per modality) | Maximum multimodal input items per prompt, per modality, as the same JSON object vLLM's flag takes: `'{"image": 2, "video": 0}'`, or with profiling options `'{"video": {"count": 1, "num_frames": 32}}'` (the options are validated and ignored — they size dummy inputs for memory profiling, which this engine does not do). A limit can only **lower** what the model/seam supports, never raise it. Malformed JSON, a negative count, or an unknown option on `image` / `video` / `audio` is refused at startup rather than defaulted. An unknown option on any other modality name is dropped rather than refused, mirroring upstream, whose fallback `BaseDummyOptions` is the one such dataclass without `extra="forbid"`. Upstream's dotted spelling (`--limit-mm-per-prompt.image 2`) is not accepted here, as for `--kv-transfer-config` and `--speculative-config` |
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

#### What `--gpu-memory-utilization` does not do yet

The flag is accepted, keeps vLLM's exact name and fraction semantics, and is
then discarded. It does not size the KV pool. Passing
`--gpu-memory-utilization 0.85` gives the same 256-block pool as passing
nothing.

Turning a free-memory fraction into a block count needs a profile run that
measures what the weights and activations cost on the device first. That run is
not implemented. It is `ROAD-V1-MEM` M3, tracked by
[issue #83](https://github.com/mudler/vllm.cpp/issues/83), and it needs a GPU to
gate.

The flag is accepted rather than refused so that a published `vllm serve`
command line runs here unchanged. Setting it prints this warning at startup, so
a log never implies it took effect:

```text
vllm.cpp: WARNING --gpu-memory-utilization 0.85 was accepted but did NOT size the KV cache.
vllm.cpp:   The profile run that turns a free-memory fraction into a block count is not
vllm.cpp:   implemented yet (ROAD-V1-MEM M3, https://github.com/mudler/vllm.cpp/issues/83).
vllm.cpp:   The pool fell back to 256 blocks. To size it today, pass
vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or --num-blocks <n> for an
vllm.cpp:   exact block count.
```

To size the pool today, use `--kv-cache-memory` for an absolute byte budget or
`--num-blocks` for an exact count. A run that never sets the flag prints
nothing.

**Warning.** On a unified-memory board such as NVIDIA GB10, a fraction of
"device" memory is a fraction of the one pool the host shares, so it reserves
host RAM as well. A value of 0.85 has hard-rebooted a GB10 box. When M3 lands
and this flag starts to bind, choose the fraction on such a board against the
whole 119 GiB pool and leave the host its headroom. Until then the flag reserves
nothing, on any board.

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

## DSpark drafts: the exact checkpoints

A DSpark draft is a SEPARATE checkpoint named by the `model` key of
`--speculative-config`. A repo id alone is not a pin, because a checkpoint can be
re-quantized in place under an unchanged name, so the revision is part of the
identity.

| Draft | Repo and revision | File | Bytes | sha256 |
|---|---|---|---|---|
| Qwen3.8-27B, 5 layers against a 64-layer target | `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | `model.safetensors` | 2 718 576 122 | `9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786` |

That draft declares `architectures: ["DSparkDraftModel"]` with `model_type:
"qwen3"` and `block_size: 7`, which is the pair
[vllm#52197](https://github.com/vllm-project/vllm/pull/52197) routes to the Qwen3
DSpark lane and which this engine mirrors ahead of its pinned oracle
(`SPEC-DSPARK-QWEN3-ROUTING`,
[#1193](https://github.com/mudler/vllm.cpp/issues/1193)). **It has not been run
here yet**: the token-exact gate against the pinned oracle needs the 2.53 GiB
download and GPU time, and both are pending developer authority, so the routing
is gated on CPU and the decode is not.

The two layouts that already run are the native
`deepseek-ai/dspark_qwen3_*_block7` drafts and the Speculators-format
`RedHatAI/*.dspark` drafts; the DeepSeek-V4 DSpark draft, whose weights ship
inside the DeepSeek-V4 target, is refused by name.

**Which refusal you actually get today.** Point the server or the C API at a
DeepSeek-V4 DSpark draft and the message is the named DeepSeek-V4 refusal, the
one the classification produces. `LoadedEngine::FromModelDir` resolves a
`dspark` speculative config ONCE, at the top of the function
([#1225](https://github.com/mudler/vllm.cpp/issues/1225)), before it opens the
target directory and long before it loads the draft, so the classification is now
the FIRST thing a DSpark run meets. An earlier writing of this paragraph said the
draft loader's "the draft config must carry target_layer_ids and mask_token_id"
won instead; that was true while the draft load ran ahead of the resolution, and
it stopped being true when the resolution was hoisted.

Two messages still come out in front of it, and both are the resolution's own.
A `dspark` run that names no `num_speculative_tokens` against a draft whose
config carries no `n_predict` is refused for the missing `k` first, because that
check sits ahead of the classification in the same branch. And a `.gguf` target
takes the GGUF branch above the hoist, which carries its own named refusal for a
GGUF DSpark target (`SPEC-DSPARK`). Either way the draft is refused and nothing
loads it as a Qwen3 draft.

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

## Nemotron-3.5-Lightning-30B: the exact weights, and which arms run

`NemotronHForCausalLM` is a hybrid: 6 GQA attention layers over a paged KV cache
and 23 Mamba2 layers over a recurrent conv/SSM state, with MoE blocks between
them. `examples/nemotron_h_gen` (`nemotron-h-gen`) drives it through the public
C ABI and nothing else — `vllm_engine_load` + `vllm_complete_tokens` — against
the committed oracle golden:

```sh
nemotron-h-gen --model "$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4" \
               --golden tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
```

`--golden-info` parses the golden and prints its geometry without loading a
model, which is how you check the battery's shape before spending a 20.1 GiB
load. `--load-only` stops after `vllm_engine_load`.

### The checkpoint

| field | value |
|---|---|
| repo | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) — first party |
| revision | `29f2d1746d8f41e316523194b19018707749b1b1` |
| staged as | `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` (a `hf download --local-dir` tree) |
| on-disk total | 21 583 809 748 bytes (20.1 GiB) |
| weights | `model-000{01..52}-of-00052.safetensors` + `model.safetensors.index.json` |
| quantization | `config.json` (1 337 760 B) + `hf_quant_config.json` (928 085 B), the `modelopt_mixed` layout |
| tokenizer | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja` |
| sha256 (first shard) | `672c8bda10fdec0256e0819e112d2aa3a936cc3e5d311a05fd3ff773ca9a44b9` for `model-00001-of-00052.safetensors` (743 427 168 B) |

**A repo id alone is not a pin** — checkpoints get re-quantized in place under an
unchanged name — so the revision is recorded, and it was verified rather than
copied: the first shard on the gate host hashes to the value above, which is
that revision's own LFS record for the file
(`.cache/huggingface/download/model-00001-of-00052.safetensors.metadata`, whose
sidecar names commit `29f2d174`). `tests/parity/hf_snapshot.h` resolves the
directory and refuses a tree staged at any other revision, so
`VT_NEMOTRON35_SNAPSHOT` is left UNSET for a gate run: setting it takes the
explicit-directory escape, which is deliberately not revision-checked.

    hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
      --revision 29f2d1746d8f41e316523194b19018707749b1b1 \
      --local-dir "$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4"

### The arms, and what each one costs you today

The loader materializes all 18 487 tensors in the memory format the checkpoint
ships them in, so nothing is silently widened at load. What differs between arms
is **where the arithmetic happens**, and that is not something a token
comparison can see, so it is written down here instead.

| arm | state |
|---|---|
| bf16 layers, norms, the 6 GQA attention blocks | **device** |
| MoE experts, NVFP4 W4A16 g16 | **device** (Marlin arena) |
| FP8 W8A8 static Mamba2 input projections | **host** — the device arm is owed, [#940](https://github.com/mudler/vllm.cpp/issues/940) |
| `lm_head`, NVFP4 W4A16 g16 | **host** — it refuses a non-CPU queue by name, so the forward's last step is a host projection and the model still returns host logits. Owed to A2-Q2b, [#810](https://github.com/mudler/vllm.cpp/issues/810) |

And the arms that are **refused by name** rather than substituted:

| arm | the refusal |
|---|---|
| GGUF k-quants / i-quants | not ported. A GGUF path is refused at load naming `.agents/specs/nemotron-h-model.md` §5b W7, because silently dequantizing to a supported path is exactly what a token gate cannot see |
| the MTP draft head | deferred by name at load (W5) |
| batched decode (`num_reqs > 1`) | refused at the forward. One request's KV pages and one request's recurrent state are carried per step; a multi-request step would be decoded as ONE concatenated causal sequence and would return plausible wrong tokens instead of failing. Owed to A2-B, [#810](https://github.com/mudler/vllm.cpp/issues/810) |

### What has NOT been measured

**No token gate result exists for this checkpoint yet.** The example above is the
vehicle for it and the golden is committed, but the run itself is pending; the
current state is recorded in `docs/BENCHMARKS.md` rather than left as silence,
and nothing about the released checkpoint's output is claimed here until it is
green.

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
  --video-extra model_version=2.5
```

`allow_unported_modules=1` is no longer needed for either shipped LTX-2.5 DiT —
`keyframes_abs_pos_embedding`, the last family that demanded it, was ported on
2026-08-14 (issue #658). The flag still exists for a checkpoint that carries
something else this port does not.

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 21`,
`include/vllm.h:273`; **46** exported functions, the count of `^VLLM_API `
declarations in that header) suitable for `dlopen` / FFI / LocalAI integration.
This line read `19` and `36` until 2026-08-17; both numbers were last true
several ABI additions ago, and neither is derived by any gate.

On native Windows/MSVC, the shared-library packaging lane keeps the runtime DLL
name at `vllm` and gives the import/static archive the distinct name
`vllm_shared`, so one build tree can hold the shared C ABI package and the
static `vllm` archive without a filename collision. The same ABI smoke test
therefore resolves the exported symbols through `LoadLibraryA` /
`GetProcAddress` on Windows and `dlopen` / `dlsym` on POSIX.

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
(default `distilled_two_stage`; also `one_stage`, `res2s_two_stage`, `dmd2`,
`dfr`, `retake` and `t2a_one_stage`), `model_version` (only for a checkpoint that
declares none), `dit_config_path`, `encoder_config_path`,
`negative_prompt_embeds_path` and `negative_audio_prompt_embeds_path` (the
negative half of the same fallback, for the unconditional forward),
`allow_unported_modules`, `max_phase`, `prompt_embeds_valid_rows`,
`upsampler_path`, `duration_head_path`, `lora_path` and `lora_strength` — twelve
keys, which is `kKnownLoadExtras` (`ltx2_video.cpp:377-383`) in order. The two
LoRA keys landed with issue #923 and were missing from this list until
2026-08-17; the array's own neighbouring comment still says "nine of these ten",
which is [#1097](https://github.com/mudler/vllm.cpp/issues/1097).
An extra a family does not define is
refused, never ignored. One caveat inside that set: `duration_head_path` is
defined but UNSERVED — the duration head is ported and gated as a brick, and
nothing in the video engine constructs one — so supplying it is **refused by
name** at load rather than accepted. It used to be accepted and read by nothing,
which silently substituted the recipe default for the file you named. Give
`num_frames` (or `duration`, which is exact arithmetic against the recipe's frame
rate) instead. Every other key in that list reaches a reader.

One LTX-2.5 arm is refused where a render would otherwise silently downgrade:
the spatiotemporal latent upsampler. It is reachable — supplying that checkpoint
as `upsampler_path` gets a refusal naming the arm you actually supplied. The
spatiotemporal upsampler is the arm with `spatial_upsample` AND
`temporal_upsample` set, which upstream builds as a different operator
(`Conv3d(mid, 8*mid)` + `PixelShuffleND(3)`). The temporal-only x2 upsampler is
**ported** and is not refused; nothing shipped drives it yet, so it is gated
rather than served. **Three** more are
recorded as out of scope but are **not requestable**, so no flag or extra can
reach them: `int8-convrot`, single-node multi-GPU, and
`BetaScheduler`. (LoRA fusion was in that list until 2026-08-15 and is now
SERVED - see `--lora` above - so its marker was retired rather than moved. This
sentence still said "Four more" until 2026-08-17, counting the retired marker in
the same breath as it explained the retirement.) That is four
`Ltx2UnportedPipelineFeature` enumerators in total, one reachable and three
markers (`ltx2_pipeline.h:768-803`), and the split is derived from the tree by
`test_ltx2_pipeline` rather than restated here. Their messages
say `DECLARED, NOT REQUESTABLE` so the two kinds are not confused.
`BetaScheduler` is in that group rather than the reachable one because upstream
selects it nowhere: every `ltx-pipelines` entry point hard-codes
`LTX2Scheduler()`, so there is no scheduler-kind field to mirror and nothing here
carries one either. `int8-convrot`
in particular is a ComfyUI-ecosystem format: upstream LTX-2's own inference
quantization kinds are `fp8-cast`, `fp8-scaled-mm`, `nvfp4-cast` and
`nvfp4-prequant`, and nothing wired upstream reaches int8 at all.

What is **not** on that list, and why: **multi-shot or multi-scene generation.**
A request that composes several camera takes into one output has no flag here
because upstream LTX-2 has no such mode to mirror — its `shot` is one continuous
take, and its own prompt-enhancement prompts instruct the model to keep a "single
continuous take" and not to describe scene cuts. `scene` does appear across the
upstream tree, in three unrelated senses (`scene-linear` HDR colour, PySceneDetect
in the trainer's dataset preprocessor, and that prompt-writing guidance); none of
them is a generation mode. This port carried a `multishot` refusal until
2026-08-13, which was a defect in our own record rather than a gap, and it was
retired. Generate one take per request.

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

`vllm_video_model_params.device` is `0` for the CPU and `1` for **the
accelerator this build resolves** — not for CUDA. The value is unchanged and it
is CUDA on a CUDA build, but it is read through the platform seam rather than as
an enum value, so the same `1` selects Metal, Vulkan or Tenstorrent on a build
that registers one of those, and is refused by name on a build that registers
none. The C ABI's text-generation `vllm_model_params.device` is a separate,
later selector with its own `0 = auto / 1 = cpu / 2 = cuda` numbering.

The LTX-2.5 arm runs on the CPU in f32 and on CUDA in bf16. `device = 0` takes
the f32 parity forward; `device = 1` stages the DiT to the GPU one tensor at a
time and runs the device-resident forward, so a CUDA handle means a CUDA forward.
On a build with no accelerator backend, `device = 1` is refused by name rather
than served the CPU forward behind an accelerator handle. It is also refused when the build's
accelerator is a PARTIAL backend that declines this architecture — Metal and
Tenstorrent each register the kernels for a named short list of models, and a
backend that has not registered this one now says so by name instead of binding
a queue and failing later inside a kernel. The same three questions decide
`minimax-h3`'s `device = 1`, which resolves through the platform seam rather
than reading the ABI selector as an enum value, so on a CPU-only build it throws
instead of naming CUDA. `encoder_path` loads the Gemma-4
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

(That brick is the TEXT decode path and is a different mechanism from LTX-2.5's
IC-LoRA, which fuses into the weights at load and IS served - see `--lora`.)
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

### Per-prompt input limits

vLLM caps how many items of each modality one prompt may carry
(`--limit-mm-per-prompt`), and `--language-model-only` is sugar for setting every
one of those limits to 0. Both flags are accepted (#607, waves L1+L2) and both
are enforced **on this server's chat path**, which is the one place that installs
the multimodal chat seam the check runs behind.

Both are also C ABI fields (`vllm_model_params.language_model_only` /
`.limit_mm_per_prompt`, ABI v19), and there they configure the engine — including
a server built on it — but they do not change what a `vllm_chat` call returns:
the C ABI has no multimodal request path yet, so an `image_url` content part sent
through it is dropped and answered as text. The refusals below are the server's.

The limits are the mechanism and the flag is the sugar, so it is worth stating
what the flag actually does: it does not "skip the encoder", it makes the server
**refuse** multimodal requests.

```console
$ curl -s localhost:8000/v1/chat/completions -d '{... three image_url parts ...}'
{"error":{"type":"BadRequestError",
          "message":"At most 1 image(s) may be provided in one prompt."}}   # HTTP 400

$ vllm-server --model … --language-model-only     # then any image request:
{"error":{"type":"BadRequestError",
          "message":"At most 0 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit."}}
```

Two things follow from how the limit is computed
(`min(user limit, what the model/seam supports)`):

- A user limit can only **lower** the ceiling. `--limit-mm-per-prompt
  '{"image": 99}'` on this server still refuses a second image, because the
  OpenAI chat seam handles exactly one image today (video and audio parts are
  not routed at all, so their limit is 0 and they are refused by name rather
  than dropped — this is what closed
  [#686](https://github.com/mudler/vllm.cpp/issues/686)).
- The ``Set `--limit-mm-per-prompt` to increase this limit.`` hint appears only
  when raising the limit would actually help — that is, when the seam could take
  the items and the configuration is what refused them. Its absence is currently
  the only way to tell an unimplemented arm from a configured limit; the
  refusal message itself does not say which
  ([#758](https://github.com/mudler/vllm.cpp/issues/758)).

**Not yet:** `--language-model-only` frees no memory. Nothing gates vision-tower
construction on the limits, so the flag today changes what the server accepts,
not what it allocates ([#607](https://github.com/mudler/vllm.cpp/issues/607)
wave L3, owed with a measured RSS reduction).

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
(`REGISTER_VLLM_VIDEO_FAMILY` at `src/vllm/multimodal/ltx2_video.cpp:3723 @ b5756ea8c`), the
Gemma-4 text tower loads from `--encoder` (`ltx2_video.cpp:1149`) and sets
`has_encoder` (`ltx2_video.cpp:1191`), both VAEs and the pipeline layer are implemented
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
schedule or guidance scale renders a video instead of failing. **Twenty-eight**
pairs resolve, derived from `ResolveLtx2PipelineRecipe`:

| `pipeline_kind` | resolving `model_version` | what it also needs |
|---|---|---|
| `one_stage` | 2, 2.3, 2.4, 2.5 | — |
| `distilled_two_stage` | 2, 2.5 | `upsampler_path` for its second phase |
| `res2s_two_stage` | **2.5 only** | `upsampler_path` for its second phase |
| `dfr` | **2.5 only** | `upsampler_path` |
| `dmd2` | 2, 2.3 | — |
| `retake` | 2, 2.5 | a source clip as a `frame_%06d.ppm` directory |
| `t2a_one_stage` | 2, 2.3, 2.4, 2.5 | a text tower; no video VAE is asked for |
| `a2vid_two_stage` | 2, 2.3, 2.4, 2.5 | `upsampler_path`, `lora_path`, and an `audio_path` on every request |
| `ti2vid_two_stage` | 2, 2.3, 2.4, 2.5 | `upsampler_path` and `lora_path` |
| `keyframe_interpolation` | 2, 2.3, 2.4, 2.5 | `upsampler_path` and `lora_path` |

This list ran to ten until 2026-08-17, omitting `dfr` entirely and all four
`t2a_one_stage` rows. **`dfr` at 2 is refused deliberately, not by oversight**:
DFR's base stage rests on generated keyframe slots, which need a checkpoint
declaring `use_keyframes_abs_pos_embedding`, and the 2.0 distilled row predates
that parameter — so resolving DFR onto it would build a recipe the engine must
then refuse at load. Refusing at the recipe table names the version instead
(the `dfr` arm of `ResolveLtx2PipelineRecipe`, named rather than given as a line
range because this row's own insertions above it staled the range once already).

### `res2s_two_stage`: the high-quality preset, and why it is a sampler

`res2s_two_stage` is `TI2VidTwoStagesHQPipeline`. Against the plain two-stage
pipeline it changes the SAMPLER on both stages — the `res_2s` second-order
method instead of Euler — and takes `LTX_2_3_HQ_PARAMS`: 15 steps, STG off,
video rescale 0.45, cfg 3.0 video / 7.0 audio, modality 3.0. Those are not the
only differences (stage 1 also loads the distilled LoRA, derives its schedule
from the stage-1 latent shape, and runs a `GuidedDenoiser` where the plain
pipeline runs a `FactoryGuidedDenoiser`), so do not read the sampler swap as an
exhaustive list. It resolves at 2.5 only, because that preset is a plain
constant upstream with no per-generation lineage to spread it over.

Fifteen steps is not fewer forwards, and it is not even 15 model calls. The
`res_2s` loop evaluates the denoiser TWICE per step — once at the step's sigma
and once at the geometric mean of that sigma and the next — and once more at a
terminal sigma the schedule injects. Stage 1's 15 steps is therefore 31 denoiser
calls, and stage 2's frozen 3-step schedule adds 7, for **38 calls per render**.
Stage 1 is also GUIDED, so each of its calls is three transformer forwards
(conditional, unconditional, isolated-modality) against stage 2's one: **100
transformer forwards** for a full render, where `one_stage` at its own 30-step
default runs 30 calls. Expect the HQ preset to cost several times the 30-step
arm and to look better, not to be faster.

That is also why the preset cannot be reached by passing its numbers to another
kind. `--steps 15` on `one_stage` renders a finished, correctly sized, plausible
clip at a fraction of the model evaluations the preset was tuned for, and no
property of the output says so. Ask for the pipeline, not for its step count.

```sh
ltx2-gen --pipeline-kind res2s_two_stage \
         --prompt "a cinematic shot of ..." \
         --height 1088 --width 1920 --frames 121
```

`pipeline_kind` is a LOAD knob, so this reaches the C API and the server too: a
server started with `--video-extra pipeline_kind=res2s_two_stage` renders every
request on the HQ preset.

Three limits, stated rather than left to be found. The stage-2 spatial upsample
is the same one `distilled_two_stage` uses and carries the same refusal when the
checkpoint has no latent upsampler. The loop's SDE noise is drawn from this
port's own generator rather than upstream's seeded `torch.randn`, so a render is
not bit-comparable with Lightricks' — the same limit the ancestral arm already
ships with. And stage 1's guidance asks for an isolated-modality pass, which the
device-resident forward cannot perturb, so this preset is host-only until that
is closed; both are recorded in `.agents/specs/ltx25-res2s-loop.md`.

### Audio-to-video: rendering a clip around a soundtrack you supply

`a2vid_two_stage` is `A2VidPipelineTwoStage`. Stage 1 denoises video at half
resolution, guided, on a schedule derived from the recipe's own step count;
stage 2 upsamples 2x and refines with the distilled three-sigma schedule. The
soundtrack is your file throughout: it is encoded once, frozen at both stages,
and handed back unchanged rather than round-tripped through the VAE.

```sh
ltx2-gen --dit ltx-2.5-22b-distilled-fp8.safetensors \
         --dit-config ltx-2.5-transformer-config.json \
         --video-vae ltx-2.5-video-vae-conv-bf16.safetensors \
         --audio-vae ltx-2.5-audio-vae-bf16.safetensors \
         --upsampler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
         --lora ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
         --pipeline-kind a2vid_two_stage --audio-path take.wav \
         --prompt "a drummer in a small club" \
         --width 128 --height 128 --frames 25 --out out/a2v
```

**No render on real weights is claimed for this recipe.** It is gated on reduced
fixtures. Upstream's stage 1 runs the base `-dev-` transformer and puts the
distilled adapter on stage 2 only; the command above names the distilled
checkpoint this tree has measured elsewhere, so it is a shape to copy rather than
a reproduced result.

Three things this kind demands, each refused by name rather than defaulted:

| What | Why | Where upstream says so |
|---|---|---|
| `--audio-path` on **every** request | the pipeline is "denoise video around this take"; without one the soundtrack is generated and the clip looks finished | `--audio-path` is `required=True`, `a2vid_two_stage.py:312-317` |
| `--lora` naming the distilled adapter | stage 2 is a three-sigma refinement the base weights were never distilled for | `--distilled-lora` is `required=True`, `utils/args.py:1140-1155` |
| `--upsampler` | stage 2's input is the upsampled stage-1 latent | `a2vid_two_stage.py:261` |

`--audio-start-time` and `--audio-max-duration` window the take; the window
defaults to the clip's own duration. A take shorter than the clip is refused
rather than padded, and a longer one keeps its leading frames.

**The distilled adapter rides stage 2 alone**, as upstream's does: stage 1 is
built with `loras=tuple(loras)` (`a2vid_two_stage.py:107`) and stage 2 with
`(*tuple(loras), *tuple(distilled_lora))` (`:114`), and
`ltx-pipelines/CLAUDE.md:48` states the convention for TI2Vid, A2Vid and
Keyframe alike. Until 2026-08-17 this page recorded the opposite as an
unrepairable divergence, because adapters fused once at load and every phase saw
them; [#1118](https://github.com/mudler/vllm.cpp/issues/1118) closed that. The
engine still holds ONE DiT — upstream does too, since both of its
`from_checkpoint` calls name the same `model_paths.transformer()` — and
re-materializes the adapter's target tensors at the phase boundary instead of
keeping a second weight set.

**What that costs you, per render.** Moving one DiT between the two states is
paid in wall-clock rather than in memory: a two-stage render does **two**
rebinds, one at each phase boundary, and each re-opens `--lora` and reads every
`lora_A`/`lora_B` factor pair before re-materializing the tensors they target.
The adapter above is 8,899,889,568 bytes, so this is not free, and the DiT is
left in stage 2's state so the next render pays the same two. **No number is
published for it** — this recipe is gated on reduced fixtures and nothing has
timed the boundary on real weights. Upstream spends memory here instead, holding
two `DiffusionStage`s over one checkpoint, which does not fit one GB10.

**The adapter `--lora` wants**, pinned by content rather than by name, because a
LoRA repository can be re-quantized in place under an unchanged filename:
`ltx-2.5-22b-distilled-lora-450-bf16.safetensors`, 8,899,889,568 bytes, 3320
BF16 tensors forming 1660 `lora_A`/`lora_B` pairs,
`__metadata__` `lora_rank` and `lora_alpha` both `450` and `model_version`
`2.5.0`. This is upstream's `distilled_lora`, the one `--distilled-lora`
(`required=True`) names. It is **not** the IC-LoRA
(`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`, 327,322,640
bytes), which is a different adapter for a different arm. Nothing here checks
which one you passed: `requires_distilled_lora` refuses a load carrying **no**
`--lora`, and that is the whole of it, so the two are told apart by the header
facts above and not by the engine.

The guider flags (`--video-cfg-guidance-scale` and the rest, spelled as the
`video_cfg_guidance_scale` extras over the C API) reach stage 1 and are ignored
by stage 2, which runs no guider at all — unlike `distilled_two_stage` and
`retake`, which refuse them outright. `pipeline_kind` is a LOAD knob and reaches
a server through `--video-extra pipeline_kind=a2vid_two_stage`, but `audio_path`
is a per-generation extra and `/v1/videos` forwards none
([#928](https://github.com/mudler/vllm.cpp/issues/928)), so every request to such
a server is refused for the missing take. This kind is reachable from the C API
and from `ltx2-gen`, and not over HTTP.

### `ti2vid_two_stage`: the plain two-stage pipeline

`TI2VidTwoStagesPipeline` — upstream's ordinary text/image-to-video two-stage
arm. Stage 1 generates at HALF the requested resolution under classifier-free
guidance on the **unadapted** model; stage 2 upsamples the latent 2x and refines
it with the distilled adapter on a frozen three-sigma schedule and no guider.

```sh
ltx2-gen \
  --pipeline-kind ti2vid_two_stage \
  --checkpoint "$CHECKPOINT_ROOT/ltx-2.5/..." \
  --upsampler-path "$CHECKPOINT_ROOT/ltx-2.5/.../spatial-upsampler.safetensors" \
  --lora-path "$CHECKPOINT_ROOT/ltx-2.5/.../ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --prompt 'a hot-air balloon over a wheat field at dawn' \
  --height 704 --width 1216 --num-frames 121 --steps 30 \
  --output-dir out/
```

`--lora-path` is **required** and the load is refused without it, mirroring
`--distilled-lora required=True`. The adapter is the same
`ltx-2.5-22b-distilled-lora-450-bf16.safetensors` the audio-to-video section
pins by content above. There is **no** `--audio-path`: this pipeline generates
its soundtrack, and the take that leaves is **stage 1's** — stage 2 refines the
picture only and its audio is discarded, which is upstream's own behaviour.

Height and width describe the FINAL output and must divide 64, because stage 1
halves them and the result still has to land on the VAE's 32-pixel grid. A size
that does not divide is refused rather than rounded.

**Against the neighbouring kinds.** It is not `distilled_two_stage`, which
builds one stage set, freezes stage 1's sigmas and gives 2.5 the ancestral
stepper. It is not `res2s_two_stage`, which puts the adapter on **both** stages
and runs the second-order sampler at 15 steps. And it differs from
`a2vid_two_stage` in three fields: no take is required, the audio guider is the
parameter table's row rather than the positive-only default, and the soundtrack
comes from stage 1.

**One behaviour is unique to this kind.** Its stage-1 sigma shift is fitted on
the scheduler's fixed 4096-token anchor rather than on the target latent grid,
because upstream calls `execute(steps=...)` with no latent. Every other derived
arm in this engine still fits on the target grid, which for six of upstream's
seven scheduler calls is a divergence
([#1150](https://github.com/mudler/vllm.cpp/issues/1150)); `res2s_two_stage` is
the one arm where the target grid is correct.

**Which weights this was gated against: reduced CPU fixtures, and nothing else.**
Upstream runs this pipeline on the FULL model
(`ltx-2.5-22b-dev-transformer-bf16.safetensors`, 42,018,190,584 bytes, 4349
tensors, 21.004 B parameters, pure BF16, `model_version` `2.5.0`), which is on
the NAS and header-verified, and which `LTX25-BF16-DIT`
([#1148](https://github.com/mudler/vllm.cpp/issues/1148)) made loadable. **What
is owed is the run**: a comparison against upstream's own render on the same
checkpoint, prompt and seed. Nothing here has been measured against it. Do
**not** substitute a distilled transformer to try the arm out — the distilled
scales are trained into those weights, so a CFG-guided stage 1 on top samples a
trajectory they were never trained for and renders a plausible clip with nothing
in its size, frame count, sample rate or errors to show it
([#1137](https://github.com/mudler/vllm.cpp/issues/1137)).

All three knobs this arm needs are LOAD extras, so a server supplies them with
`--video-extra pipeline_kind=ti2vid_two_stage` and the same for `lora_path` and
`upsampler_path`. Unlike `a2vid_two_stage` it needs no per-generation extra, so
[#928](https://github.com/mudler/vllm.cpp/issues/928) does not stand in the way
of `/v1/videos`. That is a statement about the request surface: the gated path
is `vllm_video_engine_load` plus `vllm_video_generate`, which is what `ltx2-gen`
drives, and no test here exercises the HTTP route end to end.

### `keyframe_interpolation`: generating the motion between pinned frames

`KeyframeInterpolationPipeline` — you supply the keyframes, the model generates
what happens between them. Its two stages are `ti2vid_two_stage`'s: a guided
half-resolution stage 1 on the **unadapted** model, then a 2x latent upsample and
a distilled three-sigma refinement. It needs the same `--lora-path` and
`--upsampler-path`, for the same reasons.

```sh
ltx2-gen \
  --pipeline-kind keyframe_interpolation \
  --checkpoint "$CHECKPOINT_ROOT/ltx-2.5/..." \
  --upsampler-path "$CHECKPOINT_ROOT/ltx-2.5/.../spatial-upsampler.safetensors" \
  --lora-path "$CHECKPOINT_ROOT/ltx-2.5/.../ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --prompt 'the balloon drifts from the left ridge to the right one' \
  --first-frame open.ppm --last-frame close.ppm --image-crf 0 \
  --height 704 --width 1216 --num-frames 121 --steps 30 \
  --output-dir out/
```

**Two fields separate it from `ti2vid_two_stage`, and both render either way.**

**The first frame is a KEYFRAME, not a replacement.** Every other pipeline maps a
conditioning image at frame 0 onto a latent-index item, which overwrites the
tokens of latent frame 0 in place. This one drops that special case: the image is
appended as keyframe guidance the model interpolates *from*, and the sequence the
transformer runs over grows by one latent frame. Nothing about a rendered clip
shows which mapping was used — both return the right size, the right frame count
and the right sample rate with the image visibly present — so the difference is
gated on the token count the transformer actually ran over.

**The soundtrack that leaves is stage 2's**, where `ti2vid_two_stage` keeps stage
1's and discards its refinement stage's audio. Upstream says so by what it binds
rather than in a comment, and the two pipelines bind opposite ways.

Everything else is shared. `--lora-path` is **required** and the load is refused
without it: upstream makes the distilled adapter a positional, non-defaulted
constructor argument as well as a required flag, and the adapter rides **stage 2
alone** while stage 1 runs the base weights. There is no `--audio-path`; the
soundtrack is generated. Height and width describe the FINAL output and must
divide 64, because stage 1 halves them. Its stage-1 sigma shift is fitted on the
scheduler's fixed 4096-token anchor rather than on the target latent grid, which
is what upstream's `execute(steps=...)` with no latent resolves to.

**`--last-frame` is new with this kind** and works on every pipeline that takes
images: the ABI and the engine have served a closing keyframe since
[#930](https://github.com/mudler/vllm.cpp/issues/930), and `ltx2-gen` had never
read the field ([#1191](https://github.com/mudler/vllm.cpp/issues/1191)). Both
image slots share one `--image-crf` and one strength, and a keyframe at an
**interior** frame is not requestable — upstream's `--image PATH FRAME_IDX
STRENGTH [CRF]` is repeatable and this request surface carries two fixed slots
([#1187](https://github.com/mudler/vllm.cpp/issues/1187)).

**Which weights this was gated against: reduced CPU fixtures, and nothing else.**
Upstream runs this pipeline on the FULL model
(`ltx-2.5-22b-dev-transformer-bf16.safetensors`, 42,018,190,584 bytes, 4349
tensors, 21.004 B parameters, pure BF16, `model_version` `2.5.0`), which is on
the NAS and header-verified, and which `LTX25-BF16-DIT`
([#1148](https://github.com/mudler/vllm.cpp/issues/1148)) made loadable. **What
is owed is the run**: a comparison against upstream's own render on the same
checkpoint, prompt and seed. Do **not** substitute a distilled transformer to try
the arm out — the distilled scales are trained into those weights, so a
CFG-guided stage 1 on top samples a trajectory they were never trained for and
renders a plausible clip with nothing in its size, frame count, sample rate or
errors to show it ([#1137](https://github.com/mudler/vllm.cpp/issues/1137)).

All three knobs this arm needs are LOAD extras, so a server supplies them with
`--video-extra pipeline_kind=keyframe_interpolation` and the same for
`lora_path` and `upsampler_path`. Like `ti2vid_two_stage` and unlike
`a2vid_two_stage` it needs no per-generation extra, so
[#928](https://github.com/mudler/vllm.cpp/issues/928) does not stand in the way
of `/v1/videos` — though `/v1/videos` forwards no image either, so a server
render is unconditioned. That is a statement about the request surface: the gated
path is `vllm_video_engine_load` plus `vllm_video_generate`, which is what
`ltx2-gen` drives, and no test here exercises the HTTP route end to end.

### Retake: regenerating a time window of an existing clip

`retake` is `RetakePipeline`: it keeps the source clip outside a window and
regenerates what is inside it from the prompt. It is one diffusion stage at the
source's own resolution, so it needs `--pipeline-kind retake` — the distilled
two-stage recipe renders its first stage at half resolution and refuses a retake
by name rather than putting a full-resolution latent into a half-resolution grid.

The source is `--ref-video`, a **directory** of `frame_%06d.ppm` numbered from
000000, which is the layout `minimax-h3-gen` writes so one run's frames chain
into the next request. A container file (`.mp4`) is refused: upstream opens one
with PyAV and no demuxer is vendored here. That is upstream's own second
ingestion arm rather than a substitute, and three things follow from it:
`retake_frame_rate` is required because a folder has no container frame rate; a
folder carries no audio, so the soundtrack is generated fresh; and
`regenerate_audio` therefore has no observable effect on this arm.

| `ltx2-gen` flag | per-generation extra | meaning |
|---|---|---|
| `--ref-video` | `vllm_video_params::ref_video` | the source clip DIRECTORY |
| `--retake-start-time` | `retake_start_time` | window start in seconds, inclusive; supplying it selects the retake path |
| `--retake-end-time` | `retake_end_time` | window end in seconds, exclusive; must be greater than the start |
| `--retake-frame-rate` | `retake_frame_rate` | the source folder's frame rate; required |
| `--regenerate-video` | `regenerate_video` | `1` (default) regenerates inside the window, `0` freezes the clip |
| `--regenerate-audio` | `regenerate_audio` | `1` (default); no effect while the source is a frame folder |

The extras ride the per-generation `extra_keys` / `extra_values` array on
`vllm_video_params`, so the C ABI reaches the same path with no new field.
`/v1/videos` forwards no engine extras today ([#928](https://github.com/mudler/vllm.cpp/issues/928)),
so the CLI and the C ABI are the reachable surfaces.

A retake takes its width, height, frame count and duration from the clip and
refuses a request that also names any of them. The clip's frame count must
satisfy `8k + 1` and both axes must be multiples of 32; both refusals name the
value that would have worked. `audio_path` alongside a retake is refused rather
than resolved to one of two soundtracks.

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
`ltx2_video.cpp:815-816 @ b5756ea8c`, and `--encoder` (`--video-encoder`) reaches
`Ltx2LoadTextEncoderFromSafetensors` at `ltx2_video.cpp:1149`. This section
documents them at the library level, where the gate below runs.

**Ten coordinates into `ltx2_video.cpp` and `ltx2_loader.cpp` were wrong, at
eleven citation sites on this page** — `ltx2_video.cpp:893` was cited twice.
Five of the replacements carry `@ b5756ea8c`, one per affected passage; the bare
`:NNN` beside a pinned one belongs to the same file at the same revision.
Nothing else on this page is pinned, so read an unpinned coordinate as
unverified.

They were re-derived on 2026-08-17 from the sentence making each claim rather
than by reading whatever sat at the cited line, and they were off by 40 to 2200
lines: the family registry was cited at `:1529` and lives at `:3723`, and
`has_encoder` was cited at `:893` where the assignment is at `:1191`. Every
symbol existed, so every citation looked plausible; the tell was only that
nothing at the cited line mentioned it. No gate here checks a documentation
anchor ([#632](https://github.com/mudler/vllm.cpp/issues/632),
[#911](https://github.com/mudler/vllm.cpp/issues/911)), so a pin is the only
thing that lets a reader tell a stale coordinate from a moved one.

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

Two behaviours a caller has to know. `Ltx2LoadDitFromSafetensors` ACCEPTS both
shipped DiTs with no opt-in as of 2026-08-14. `Ltx2DitLoadOptions::allow_unported_modules`
still exists, and still loads the ported subset while reporting every dropped
family in `Ltx2DitCheckpoint::unported`, but neither shipped LTX-2.5 checkpoint
needs it any more. `keyframes_abs_pos_embedding` was the last family on that
list; it is PORTED (issue #658), and `prompt_adaln_single` /
`audio_prompt_adaln_single` left the list the same way on 2026-08-13. The two
DiTs used to be refused from OPPOSITE directions — the vonkaiser FP8 copy for
carrying a trained `keyframes_abs_pos_embedding` this port did not apply, and the
first-party NVFP4 copy for declaring `use_keyframes_abs_pos_embedding` while
carrying no tensor at all. The second case is upstream-legal and means "apply
nothing": upstream builds the parameter on the meta device and
`supports_keyframes_abs_pos_embedding` stays False, so
`Ltx2AdoptDeclaredDitParams` resolves the declared flag against what the file
actually carries rather than refusing it or inventing a zero. The two
`*_embeddings_connector` towers are
**not** among them and never will be:
`UnportedFamilies` (`ltx2_loader.cpp:573 @ b5756ea8c`) filters them out at `:582`
through `LoadedElsewhere` (`ltx2_loader.cpp:569`), `RefuseUnported`
(`ltx2_loader.cpp:592`) says so in its own message at `ltx2_loader.cpp:608-611`,
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

### The DiT is not always quantized, and the FULL model never is

**`--dit` accepts an UNQUANTIZED bf16 transformer as of 2026-08-17**
([#1148](https://github.com/mudler/vllm.cpp/issues/1148)). Until then `PlanDit`
refused any DiT carrying neither `U8` nor `F8_E4M3`, and the file it refused is
the one most of these pipelines need: upstream's table
(`packages/ltx-pipelines/CLAUDE.md:17-30` @ `fd4ded7f`) marks `Full` or
`Full + distilled LoRA` for `TI2VidOneStagePipeline`, `T2AOneStagePipeline`,
`TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `A2VidPipelineTwoStage`
and `KeyframeInterpolationPipeline`. `one_stage`, `t2a_one_stage`,
`res2s_two_stage` and `a2vid_two_stage` are all reachable here, so all four
could previously only run against a *distilled* checkpoint — a different
sampling regime that renders plausibly and says nothing.

Nothing about the arm is a new decoder. Unquantized is upstream's ordinary case:
`_DTYPE_CASTABLE` (`single_gpu_model_builder.py:51-57` @ `fd4ded7f`) is
float32/float64/float16/bfloat16, and uint8-NVFP4 and float8 are what that file
calls "quantized payloads". `Ltx2DitCheckpoint::quant` reports which of the
three the file was, and a BF16 weight is stored as it is, so the memory format
is what the checkpoint chose.

**A dtype this loader cannot read is still refused, by name.** The refusal now
lists the dtypes the file holds and the four encodings the loader materializes
(BF16, F32, F8_E4M3 with an F32 `<name>_scale`, and U8 with an F8_E4M3
`<name>_weight_scale` plus an F32 `<name>_weight_scale_2`). An `F16` DiT is the
live case: upstream's castable set lists `torch.float16` and this port has no
F16 materialization. The message it replaced said "use the L2 path", which was
advice a reader could not follow — `Ltx2LoadDitFromSafetensors` *is* the L2 path
and calls the refusing function on its first line.

**The full model costs ~42 GB resident.** It is 21.004 B parameters at two
bytes, not a widening: no path in this loader turns a bf16 weight into anything
else, and `widen_to_f32` stays opt-in. That does not fit one GB10 beside a
24 GB text tower, so the arm has been gated on reduced fixtures and on the real
file's *header*; a full materialization and a render on real weights are still
owed ([#1048](https://github.com/mudler/vllm.cpp/issues/1048)).

### LTX-2.5 DiT weights: which file, and how to tell them apart

Repo [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) at
revision `6c7e5e573ac1667efc83407806fe9b0b93730e60`, read from
`/api/models/Lightricks/LTX-2.5` on 2026-08-17. Sizes below come from the same
API's tree listing.

| Arm | File under `diffusion_models/` | Bytes | sha256 |
|---|---|---:|---|
| unquantized bf16, FULL (dev) | `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` (the local copy; see below) |
| unquantized bf16, distilled | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 | not obtainable here |
| NVFP4 (`nvfp4-prequant`), distilled | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | not obtainable here |
| `int8-convrot`, REFUSED (ComfyUI-only) | `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | not obtainable here |
| `int8-convrot`, REFUSED (ComfyUI-only) | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | not obtainable here |

**The hub will not give you a content hash for this repo, and it does not say
so.** `Lightricks/LTX-2.5` is gated — an unauthenticated `resolve` returns
`Access to model Lightricks/LTX-2.5 is restricted` — and the tree API answers an
unauthenticated caller with an `lfs.oid` that is **one character repeated 64
times**, for every LFS file in the repo. It is the right length, it is
lowercase hex, and `len(oid) == 64` passes. All 14 LFS files share it, which is
the only cheap tell. So a pinning script that reads that field records five
different checkpoints under one fabricated digest and reports success. Pinning
the other four by content needs an authenticated fetch and is owed
([#1048](https://github.com/mudler/vllm.cpp/issues/1048)); the dev row above is
the sha256 of the copy on this project's NAS, computed locally, and it has not
been compared against the published artifact because there is nothing here to
compare it to.

**The two bf16 transformers are exactly the same SIZE**, and the file name is
the only cheap thing that separates them. Both are 4349 tensors, both carry the
same four `__metadata__` keys with `model_version` `2.5.0`. So a mislabelled or
re-downloaded copy cannot be caught by `ls -l`, and nothing here validates the
checkpoint *class* at load
([#1137](https://github.com/mudler/vllm.cpp/issues/1137)): pointing
`--pipeline-kind res2s_two_stage` at the distilled file renders in the wrong
sampling regime with no diagnostic.

Read from the FULL model's own header on 2026-08-17, by parsing its
677,616-byte JSON prologue and no payload: 4349 tensors, every one
`model.diffusion_model.`-prefixed, **4059 BF16 and 290 F32**, zero names ending
in `_scale`, `_scale_2` or `torchao_nvfp4`, 48 blocks,
`keyframes_abs_pos_embedding` present and TRAINED as `BF16 [1, 4096]`, the 290
F32 tensors being exactly the six `*scale_shift_table*` families, and the data
end plus the 8-byte length plus the header equal to the file size.

The `vonkaiser/LTX-2.5-FP8-NVFP4` FP8 DiT is a separate repo and is pinned where
the FP8 recipes name it; it carries no `__metadata__` at all, which is why those
recipes need `--dit-config`.

The gate needs the three checkpoint headers, a vLLM checkout and an LTX-2
checkout (the two nibble-order authorities); it reads a few hundred bytes at
their own offsets and never a payload:

```sh
python3 scripts/gen-ltx2-quant-goldens.py --vllm ~/_git/vllm --ltx2 ~/_git/LTX-2 --checkpoint-root "$CHECKPOINT_ROOT" --out tests/vllm/models/ltx2_quant_goldens.inc
cmake --build build --target test_ltx2_loader && ./build/tests/test_ltx2_loader
```

## Streaming routed experts from disk (capacity mode)

A mixture-of-experts checkpoint larger than the box can hold can be run by
keeping the routed-expert weights on disk and paging slices into a bounded
resident cache. It is **off by default** and it is a **capacity** feature, not a
throughput one: it targets single-user and low-concurrency use, and at high
concurrency every step touches most of the experts, so there is nothing left to
save.

```sh
VT_MOE_EXPERT_STREAM=1 \
VT_MOE_EXPERT_STREAM_SLOTS=8000 \
  ./build/examples/vllm-cli --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00008.gguf \
                   --prompt "The capital of France is" --max-tokens 16
```

### The same thing as config, and which one wins

The residency knobs are also config keys, under the `vllm_cpp` key of
`--offload-config` — the flag that already carries vLLM's weight-offload
document. `vllm-cli` takes the same flag, so the two recipes here differ only in
which binary they start, not in what each one can express. One flag covers both tiers: vLLM's own `uva`/`prefetch` keys move
weights from the device to host RAM, and the `vllm_cpp` key governs the tier
below that, where weights stay borrowed out of the file mapping.

```sh
./build/examples/vllm-server --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00008.gguf \
  --offload-config '{"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},
                                 "expert_stream":{"enabled":true,"slots":8000}}}'
```

| Key | Environment equivalent | Default |
|---|---|---|
| `vllm_cpp.mmap.enabled` | `VT_GGUF_MMAP` | on when weights stay quantized |
| `vllm_cpp.mmap.prefault` | `VT_GGUF_PREFAULT` | on with mmap residency — **set it `false` for a model larger than memory** |
| `vllm_cpp.expert_stream.enabled` | `VT_MOE_EXPERT_STREAM` | off |
| `vllm_cpp.expert_stream.slots` | `VT_MOE_EXPERT_STREAM_SLOTS` | `64`; a real model wants thousands |
| `vllm_cpp.expert_stream.slot_bytes` | `VT_MOE_EXPERT_STREAM_SLOT_BYTES` | the largest gate/up/down slice of the first MoE layer reached |
| `vllm_cpp.device_fit.weight_budget_bytes` | `VT_DEVICE_WEIGHT_BUDGET_BYTES` | the device's own probe (`cudaMemGetInfo` total on CUDA; no check elsewhere). `0` suppresses the load-time device-fit refusal; it is the only key here that accepts `0`, and a negative value is refused |

Every field is optional, and an absent field means unchanged, so an
`--offload-config` without a `vllm_cpp` key behaves exactly as it did before this
surface existed — with one difference, described below: a misspelled key is now an
error rather than being ignored. The same C ABI field carries it:
`vllm_model_params.offload_config` is one string holding both halves, so a library
client needs no new field.

**A second engine in one process is legal.** "Absent means unchanged" applies to the
install as well as to the parse: a later document is merged field by field over the
installed one, so `{"vllm_cpp":{"mmap":{"enabled":true}}}` on a second engine changes
`mmap` and leaves the first engine's `expert_stream` and slot count alone. Only two
things cannot be changed once a model has used them — whether expert streaming is on,
which is cached the first time it is asked, and the slot store's `slots x slot_bytes`
reservation, which is fixed when the store is built. A document that would change
either is refused at startup, naming the field and the value in force; a document that
omits it, or asks for exactly what is in force, is accepted.

**Precedence is `environment variable > config > built-in default`**, and it is
deliberate: the `VT_*` variables exist so a benchmark arm is switchable without
restarting the server with a new document, so `VT_MOE_EXPERT_STREAM=0` beats a
config `"enabled": true`. The engine prints one line at startup naming the fields
of the document it installed, and a second naming every variable that would win
over one of them, because a configuration silently overridden by something
exported weeks ago is the one way this precedence hurts. The first line reports
what was ASKED FOR, not what the engine resolves: the streaming answer is cached the
first time it is asked, so resolving it at startup would move that decision ahead of
the weight load. That constraint binds `expert_stream` alone — `prefault` and `slots`
could be resolved at startup, and `mmap` and `slot_bytes` need a built-in default only
their caller knows — and the line reports the document for all five so it reports one
kind of thing rather than a mixture. Read the two lines together: `expert_stream=on`
beside `VT_MOE_EXPERT_STREAM (expert_stream) OVERRIDES` means the document said on and
the variable decides.

**Where the config form reaches, and where it does not.** It reaches
`vllm-server`'s generate/chat path, `vllm-server`'s pooling/embedding path,
`vllm-cli`, and the C ABI's `vllm_model_params.offload_config`, which is the whole
of the library surface. All four take BOTH halves of the document, and the server
parses it once, before it reads the model's architecture, so a typo is refused at
startup whichever path the model then takes.

It does NOT reach the server's **transcription-only** path, and that path
**refuses the flag** rather than accepting it and doing nothing:

```text
server: fatal: --offload-config is not supported on a transcription-only model
(ParakeetForCTC). THE MISSING PART: this path serves /v1/audio/transcriptions
through ParakeetTranscriber, which loads its own weights and never builds an
engine, so neither vLLM's uva/prefetch weight offload nor vllm.cpp's vllm_cpp
weight-residency tier has a call site on it. ...
```

Use the environment form above on that path, or serve a text-generation or
embedding model. Recorded under `## Owed` in
[`.agents/specs/weight-residency-config.md`](../.agents/specs/weight-residency-config.md)
with [#1195](https://github.com/mudler/vllm.cpp/issues/1195).
[#1135](https://github.com/mudler/vllm.cpp/issues/1135) is the issue this section
answered for the other three.

**A misspelled key is refused at startup, not ignored — at every level of the
document.** vLLM's own parser ignores a key it does not recognise, which is what
lets this extension share the flag, and it is also what would make
`{"vllm_cpp":{"mmapp":…}}` or `{"vllm-cpp":{…}}` start a server that quietly does
not borrow its weights, discovered later as an out-of-memory kill. The hyphenated
spelling is the likeliest typo of all, because every flag around it is hyphenated.
So the whole document is enumerated and the offender is named:

```text
offload config: unknown key "vllm_cpp.mmapp" (expected one of: mmap expert_stream device_fit)
offload config: unknown key "vllm-cpp" (expected one of: offload_backend uva prefetch vllm_cpp)
offload config: unknown key "uva.cpu_offload_GB" (expected one of: cpu_offload_gb cpu_offload_params)
```

Every level means every level, the mirrored sub-objects included. The enumeration once
stopped at the top level and inside `vllm_cpp`, which left the same hole one step down:
`{"uva":{"cpu_offload_GB":10}}` started a server with a 0 GiB offload budget the
operator believed was set.

The four legal top-level keys are `offload_backend`, `uva`, `prefetch` and
`vllm_cpp` — vLLM's three plus this extension — so a typo in the mirrored half
(`uvaa`, or `cpu_offload_gbb` inside it) is refused on the same terms. Refusing is what upstream does with its own
JSON config flags: vLLM builds its config dataclasses with a decorator that sets
`ConfigDict(extra="forbid")` (`vllm/config/utils.py:68-69`), which is why
`--kv-transfer-config` refuses an unknown key — and upstream has no
`--offload-config` at all, so no upstream-legal document is refused by this.

`VT_MOE_EXPERT_STREAM_STATS_EVERY` is **not** a config key, by decision: it
changes only how often the statistics line below is printed, so it is the
instrument rather than the configuration, and the config surface refuses it as an
unknown key rather than accepting and dropping it.

It applies to CPU keep-quant expert towers. On a device platform the expert
slice is already device-resident and is served unchanged, and turning streaming
on also disables the default-on grouped-MoE path, which stages the whole tower
and therefore cannot stream. The engine says that once on stderr rather than
silently doing no streaming.

**Read the statistics line before you believe any number you measure with it.**
The engine prints one every `VT_MOE_EXPERT_STREAM_STATS_EVERY` steps (default
16, `0` silences the periodic line), and **exactly one more when the process
ends**, whatever the run did:

```text
[expert-stream] steps=64 hits=141230 misses=37312 evictions=29312 fills=37312 bytes=92876505088 exhausted=0 advised=37312
```

**The final line is the one to read**, because it is the only one you are
guaranteed to get. The periodic line is skipped whenever the step count is not a
multiple of the interval, so a healthy five-token run prints none of them at the
default 16; and it used to be skipped on `steps == 0` as well, which meant the
one run that most needed reporting — the one where the step boundary is never
reached — printed nothing at all. Treating absence as failure therefore reported
VOID on a working lane. The final line crosses both of those skips, so it is
printed even on a run of zero steps.

Two of the fields decide whether the run is measuring anything at all:

- `steps` must advance. If the final line says `steps=0` the decode step
  boundary is not being reached, and the cache stops serving as soon as it
  fills — it will fall back to the memory mapping for the rest of the run.
- `exhausted` must stay 0. Anything above 0 means slices were refused and read
  from the memory mapping instead, which is the slow path streaming exists to
  replace. The usual cause is a budget smaller than one step's working set:
  raise `VT_MOE_EXPERT_STREAM_SLOTS`.

Read it together with the `[expert-stream] ON slots=...` banner, which is printed
once when the lane builds its store. The four shapes are:

| Banner | Final line | What happened |
|---|---|---|
| absent | absent | Nothing reached the streamed seam. A CUDA run (a device-resident expert is served unchanged), a checkpoint whose experts are not keep-quant towers, or a prompt that never reached an MoE layer |
| present | present | The lane ran. Read `steps` and `exhausted` |
| present | absent, and nothing called `ExpertStreamFlushStats` | The process did not reach its static destructors: a crash, a signal, or `_exit` |
| present | absent, because `ExpertStreamFlushStats` was called | The internal gate seam took the process's single print, so teardown had none left to make. No shipped command or server path calls it, so an operator never reaches this shape |

The last two shapes are keyed on the CALL and not on what stderr looks like,
because stderr cannot separate them. `ExpertStreamFlushStats` prints the same
line in the same shape as the periodic report, so "a statistics line already
appeared mid-run" is also what a healthy run of 16 steps that then crashes
produces. What distinguishes the two is whether the seam was called, and only a
gate calls it.

A run whose `steps` is 0, or whose `exhausted` is large, is not a measurement of
streaming, whatever the startup line said. See
[`docs/ENVIRONMENT.md`](ENVIRONMENT.md) for every knob and its parsing rules.

### `--device cuda` refuses a checkpoint it cannot hold

Streaming is a **host** capability. The GGUF mapping is borrowed in place on the
CPU path, so a routed-expert tower costs no resident bytes, which is the whole
reason a 369.96 GiB checkpoint serves on a 119.631 GiB box. A weight-staging
device has no such lane: it copies every tower into device memory, one
`cudaMalloc` per stacked `[E*N,K]` tower.

For `Qwen3.8-2.4T-A95B UD-Q1_0` that is 276 towers of 1,275,068,416 bytes plus
three of 2,818,572,288, so 335.62 GiB in total, against a pool `cudaMemGetInfo`
reports as
128,452,956,160 bytes (119.631 GiB). Until that lane exists
([#1124](https://github.com/mudler/vllm.cpp/issues/1124)), the engine **refuses
at load** and names what is missing:

```text
device 'cuda' cannot serve this GGUF: staging its weights needs at least N bytes
(X GiB) of device memory across T tensors, the largest single allocation being M
bytes (Y GiB, '<tensor>'), and this device's memory pool is B bytes (Z GiB).
THE MISSING PART: ... there is no device-side expert slot store and no device
streaming lane ... Use device=cpu, which serves this checkpoint today, or a
checkpoint that fits the pool.
```

It used to load for 26 minutes, report ready, and then die on the first request
with `vt cuda: cudaMalloc: out of memory` from inside the engine's busy loop
([#1123](https://github.com/mudler/vllm.cpp/issues/1123)).

The refusal is keyed on the measured condition and not on the device or the file
format, so **a GGUF that fits the pool still loads on `--device cuda`**. Three
things it deliberately does not do:

- it never fires on a platform that does not stage weights, so every
  `--device cpu` load is unchanged;
- it never fires when no budget is known. Today exactly one platform stages
  weights (CUDA) and exactly one probes a budget (CUDA, with `cudaMemGetInfo`),
  so **every NVIDIA GPU this build runs on — discrete or GB10 — gets both the
  probe and the refusal**, while ROCm, Vulkan and Metal answer
  `needs_weight_staging() == false`: they read the GGUF mapping where it already
  lies, so there is no staging allocation to fail and nothing for this check to
  decide. What is owed there is the `Backend::DeviceMemoryInfo` probe CUDA does
  not implement ([#1126](https://github.com/mudler/vllm.cpp/issues/1126)), which
  is a different capability;
- it counts **weights only**. The KV cache, activations, scratch pools and the
  driver context are not in the bound, so a checkpoint just under the pool
  passes this check and can still fail later;
- it can also count a little **too much**: a tensor present in the file that this
  load will not stage — the MTP / `nextn` block on a load with no speculator, 8.33
  GiB of the measured 369.96 GiB checkpoint — is still in the sum, so a budget in
  that narrow window refuses a weight set that would have fitted. Raise
  the budget if you land in it
  ([#1136](https://github.com/mudler/vllm.cpp/issues/1136)).

**Moving the budget.** Lower it when something else lives in the pool, or raise
it (or set `0`) to suppress the refusal and get the late failure back. It does
not make the model fit. Two ways to say it, and the first beats the second:

```sh
VT_DEVICE_WEIGHT_BUDGET_BYTES=68719476736 ./build/examples/vllm-server --model ...
./build/examples/vllm-server --model ... \
  --offload-config '{"vllm_cpp":{"device_fit":{"weight_budget_bytes":68719476736}}}'
```

The config key is the same `--offload-config` document the residency knobs use,
so one flag still covers weight placement
([#1127](https://github.com/mudler/vllm.cpp/issues/1127)). `0` from either input
suppresses the refusal. The environment variable takes decimal digits only: a
value with a sign, a space or trailing garbage is ignored and falls through to
the config, then to the probe, because reading a typo as `0` would silently
disable the guard. A malformed config value cannot get that far, because the
parser refuses it at startup.

**The instrument matters here.** `nvidia-smi
--query-gpu=memory.total,memory.free,memory.used` answers `[N/A], [N/A], [N/A]`
on a GB10, because host and device share one pool. `cudaMemGetInfo` answers
honestly, and its `total` is EXACTLY `/proc/meminfo MemTotal`
(125442340 kB) times 1024. Do not size this from `nvidia-smi`.

## Turning CUDA graph capture off, including the break seam

`VLLM_CPP_CUDAGRAPH=0` disables CUDA graph capture. It always did for the six
batched decode drivers that each read it, and as of `ENG-CUDAGRAPH-BREAK` W1
(#1192) it is also the switch the shared break-point seam reads, once per
process, into a function-local static — so a process is in exactly one lane for
its whole life and nothing can toggle it mid-run.

With capture off, or on a backend that reports no capture support (Vulkan,
Metal, and the CPU backend), a `vt::GraphCaptureScope` is INERT: it captures
nothing, every `vt::GraphBreak` inside it calls its function and returns, and
the forward runs eager exactly as before. That path is byte-identical to the
non-capturing forward and makes zero backend calls, which is what makes each
migration stage reversible.

Nothing about this is new configuration to learn: there is no new flag, no new
config key and no new command. The seam is a library surface
(`include/vt/breakable_graph.h`), and W1 registers one break point at the dense
attention entry of `Qwen3ForCausalLM`. No production step opens a capture scope
yet — that arrives when the decode drivers migrate onto the seam — so today the
switch changes nothing about the break point beyond what it already changed
about the decode graphs.

Building it needs no option. `src/vt/breakable_graph.cpp` is part of the core
`vllm` library on every platform, because the seam is backend-agnostic and asks
nothing new of any backend.

The switch is GATED, and it is gated in a child process, because it is read once
per process into a function-local static and no test in a running process can
toggle it. `tests/vt/test_breakable_graph.cpp` re-executes itself with
`VLLM_CPP_CUDAGRAPH=0` and requires the inert behaviour on a backend that CAN
capture — the arm that proves the switch itself is what turns capture off, rather
than the backend's own lack of support. Asserting the backend arm instead
substitutes a different condition, and dropping the switch from the seam left the
whole suite green.

## SSE keepalives on long prefill

Async chat/completion streams can emit SSE **comment** frames (`:\n\n`) while
waiting on the engine (long prefill / TTFT), so a proxy with an inactivity
timeout sees body bytes before the first token. Interval is
`VT_SERVER_SSE_PING_S`, **default `0` — off**; a positive value enables it and
is clamped to 600.

**It is off by default, and it should stay off unless a proxy forces your
hand.** vLLM's streaming endpoints emit no comment frame at any point, so a
server that sends one is putting a byte on the wire that OpenAI-compatible
clients written against vLLM have never had to parse. vLLM's own benchmark
client is one of them: `vllm bench serve` strips each network chunk before
parsing, which destroys the `\n\n` separator at chunk boundaries, and its only
resynchronisation path looks for a `data: ` prefix — so one comment frame
arriving before a request's first token makes it report
`Never received a valid chunk to calculate TTFT` and count that request
**failed**, while this server completes it normally and logs nothing. The
requests that reach a keepalive are by construction the slowest ones, so the
effect is to delete your own worst latencies from a measurement
([#931](https://github.com/mudler/vllm.cpp/issues/931),
[#577](https://github.com/mudler/vllm.cpp/issues/577)).

Comment frames are not `data:` events and carry no tokens, and neither setting
turns token streaming into a poll loop. At the `0` default both streams take the
blocking `get_output()` on that request's own collector
(`serving_completion.cpp:39-43`, `serving_chat.cpp:333-337`), which returns the
instant the engine has something for that request. A positive interval swaps in
`get_output_for()`, the same wait with a timeout attached, and the timeout only
expires when the collector produced nothing at all. Deltas are therefore never
collapsed or delayed either way.

**A value the server cannot parse disables the keepalive; it is not an error.**
`VT_SERVER_SSE_PING_S=fifteen`, an empty value and an unset variable all resolve
to `0`, so if you enable this and no comment frames appear, check the spelling
before looking anywhere else. The fallback points at OFF deliberately: under the
previous default a typo silently switched the keepalive ON, and that is the
direction that costs you requests.

**The interval bounds silence on one request's stream, not its time to first
token.** Each wait restarts whenever anything reaches that request, so a long
prefill that keeps producing intermediate results never pings however long its
first token takes, while a request whose stream goes quiet for the whole
interval does.

## Gemma4 FP8 on ROCm (RDNA4)

Dual-GPU resident FP8 MoE and SharedK-WMMA prefill are controlled via
ENVIRONMENT.md (`VT_GEMMA4_RESIDENT_*`, `VT_ATTN_*`). Defaults stay safe off RDNA4.
GetBlas keeps two per-thread hipBLAS handles (`tls_slots[2]`, device 1 → slot 1)
so a 0→1 hop does not destroy GPU0's handle. `ProductGetBlasHandle` is the
test accessor for that file-local `GetBlas`. HIP live probe is a separate CTest
target (exit 77 if `HIP_VISIBLE_DEVICES` empty); it enters capture so production `StreamIsCapturing` is load-bearing. No new env.
Prefill peer (#839) unpins dequant cache only after observed retirement; a failed fill/ready lease is retired with RetireFillLocked after the producer stream sync (never under cache.mu); restore-fail after publish retires before rethrow; failed retire quarantines the pin.
This path does **not** restructure the Gemma-4 layer loop or enable decode hipGraph
(those stay lab-only until a CUDA token-exact gate can land them).

Contributor KEEP recipe (2x R9700 gfx1201, ROCm 7.2.4, `PREFIX_CACHE=0`, unique
pads, 2026-08-13): SharedK-WMMA on, FLASH/FMHA off, `VT_GEMMA4_PREFILL_GEMM_M=2048`
(the default), `VT_GEMMA4_PREFILL_PEER_ACT=1` (the default), batch MoE `T>=64`.
Fair median prefill **2014 t/s @~11k** and **1099 t/s @~42k**; stream decode
**55 t/s** temp=0. Paris / arith `63` / `gemma4` tool_calls held. Speculative,
ngram, FMHA, and layer-split are **out of this recipe**. Details:
[spec](../.agents/specs/gemma4-rocm-fp8-moe.md).

These are contributor-lab numbers against **no denominator**: no pinned vLLM-ROCm
run on the same box, same model, same quantization and same request shape exists
for them, so `docs/BENCHMARKS.md` still records this backend as `PENDING: no
binding throughput number` and this recipe does not change that. The decode
figure is also not reproducible from the knobs above: the as-run recipe set four
further decode splits that no product code in this tree reads
([#845](https://github.com/mudler/vllm.cpp/issues/845)), and they are recorded in
the spec rather than here, because a recipe on this page has to be one a reader
can follow.

## LTX-2.5 text conditioning

This documents **one brick of the shipped render path** — the text conditioning
the DiT consumes — and how to reproduce its gate. The render itself is above
under [LTX-2.5: what runs, and what it cannot do](#ltx-25-what-runs-and-what-it-cannot-do);
`--encoder` is what puts this brick on that path, and `has_encoder` is set at
`ltx2_video.cpp:1191 @ b5756ea8c` once the tower loads.

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

## MiniMax-Music3: the exact weights (so a song is reproducible)

**The repository is 57.4 GB and the arm we load is 28.5 GB**, because
`MiniMaxAI/MiniMax-Music3` ships the same weights **twice**: a native
`AbabForCausalLM` + `.pth` layout that SGLang-Omni serves, and a `diffusers`
six-component layout. They are the same numbers in a different arrangement —
diffusers' own `scripts/convert_minimax_music3_to_diffusers.py` renames tensors
and does nothing else — and this port loads the diffusers one. So the download
is 57.4 GB unless you filter, and what has to fit is 28.5 GB.

### The arm that loads: `diffusers`, bf16 + fp32

Repository [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
revision **`fbdf52fbaaca799592917417eb05f1899f1255ec`**. First-party. A repo id
alone is not a pin — checkpoints do get re-quantized in place under an unchanged
name — so the revision is recorded, and it was verified rather than copied:
`condition_encoder/diffusion_pytorch_model.safetensors` on disk here hashes to
`83179c5eaa9a68a370affe0c1b96c2179f659ea4175666b31071490a202c2a4d`, which is
that revision's own LFS record for the file.

| component | file(s) | size | dtype on disk |
|---|---|---|---|
| `language_model/` | `model-0000{1,2,3,4}-of-00004.safetensors` + index | **17.17 GB** | BF16 |
| `transformer/` | `diffusion_pytorch_model-0000{1,2}-of-00002.safetensors` + index | **9.73 GB** | **F32** |
| `rvq_depth_decoder/` | `diffusion_pytorch_model.safetensors` | **1.29 GB** | BF16 |
| `vocoder/` | `diffusion_pytorch_model.safetensors` | **217 MB** | F32 |
| `condition_encoder/` | `diffusion_pytorch_model.safetensors` | **101 MB** | F32 |
| `tokenizer/` | `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja` | **11 MB** | — |
| `scheduler/` | `scheduler_config.json` | 483 B | — |
| the root itself | `modular_model_index.json`, `config.json`, `README.md` | 14 KB | — |
| | **resident total** | **28.5 GB** (28 517 617 303 B) | |

The transformer being 9.73 GB for a 2.4B model is **fp32 storage, not a 4.9B
model** — that is upstream's own choice for the acoustic half and we mirror it.
The download:

    hf download MiniMaxAI/MiniMax-Music3 --revision fbdf52fb \
      --local-dir "$CHECKPOINT_ROOT/minimax-music3" \
      --exclude 'qwen_7B/*' '*.pth'

Two components are BF16 and three are F32, and **that set is not runnable as
stored**. Upstream casts in exactly two places, so the language model, the RVQ
depth decoder and the condition encoder must share one dtype; the gated
configuration is bf16 for those three and fp32 for the transformer and vocoder.
The loader enforces it and refuses a violation by name. The section below has
the detail.

### The arm that is REFUSED: the native `.pth` layout

The same repository's other 28.9 GB. **We refuse it by name** — a tree in this
shape is diagnosed as the native arm, told which diffusers components it lacks,
and pointed at the conversion script. It is never silently mis-loaded.

| file | size | what it holds |
|---|---|---|
| `qwen_7B/qwen_7B/` | ~17 GB | `AbabForCausalLM` shards; the RVQ depth decoder and the audio embedding live *inside* them as `model.audio_decoder.*` / `model.audio_extra_embedding` |
| `flowmatching_vae.pth` | ~9.7 GB | the DiT plus the condition projection |
| `dav.pth` | ~0.2 GB | the DAC Flow-VAE decoder |

**SGLang-Omni serves this arm exclusively.** If you are comparing against
`sgl-omni serve`, that is the layout it reads — same weights, so the comparison
is valid, but not the same files.

### The quantized arm that IS implemented: GGUF Q4_K, one component

| field | value |
|---|---|
| repo | [audio-cpp/MiniMax-Music3-GGUF](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) — **third party**, not MiniMaxAI |
| revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| file | `rvq_depth_decoder_q4_k.gguf` |
| size | 405 752 480 bytes (406 MB, against 1.29 GB bf16) |
| sha256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |
| contents | 47 tensors: 36 Q4_K projections, 9 BF16 norms, 2 F16 embedding tables |

It is the **only** quantized arm implemented, and one component is not a
quantized model. The remaining four are refused by name and owed; the section
"MiniMax-Music3: the quantized arms" below records what each refusal says.

### The quantized arms that are REFUSED — and they are all third-party

**MiniMaxAI ships bf16/fp32 only.** A HuggingFace survey on 2026-08-14 found
**fourteen community repositories in five formats**, published within days of the
release, and none of them is from the model's authors. Every one carries
different provenance from a first-party release, and every one except the single
Q4_K file above is refused by name.

| format | repositories | coverage | state |
|---|---|---|---|
| GGUF, `audiocpp` lineage | [audio-cpp/MiniMax-Music3-GGUF](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) | all five components, bf16 and Q4_K arms | `rvq_depth_decoder_q4_k` **LOADS**; `transformer_q4_k` (1 396 MB), `language_model_q4_k` (7 184 MB), `vocoder` (217 MB) and `condition_encoder` (101 MB) are **OWED**. Note the last two are bf16 GGUF, not k-quant — same size as the safetensors, so they buy nothing |
| GGUF, `mm3` lineage | [scragnog/MiniMax-Music3-GGUF](https://huggingface.co/scragnog/MiniMax-Music3-GGUF) | 2-file split (`mm3-lm-*` / `mm3-synth-*`), 13 tiers incl. MXFP4 and NVFP4 as GGML tensor types | **REFUSED**: needs a rename table *plus* fused QKV to split and folded weight-norm to invert. Its NVFP4 tier uses GGML type id 40, which is not a standard llama.cpp id |
| GGUF, ComfyUI lineage | [Abiray](https://huggingface.co/Abiray), [realrebelai/MiniMax-Music-3_GGUFs](https://huggingface.co/realrebelai/MiniMax-Music-3_GGUFs), [molbal](https://huggingface.co/molbal), [ChrisColeTech](https://huggingface.co/ChrisColeTech) | the 2.46B **DiT alone**, Q2_K…Q8_0, 0.9-2.7 GB | **REFUSED, and it can never be a complete arm**: these files carry the DiT and condition encoder only — no language model, no depth decoder, no vocoder — so even a finished GGUF arm would not make them generate audio |
| int8 / w4a8 | [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) (`_int8_convrot`), [NidAll/MiniMax-Music3-W4A8](https://huggingface.co/NidAll/MiniMax-Music3-W4A8), [dummy9996/…-w4a8-bf16-comfyui](https://huggingface.co/dummy9996) | DiT | **REFUSED** by name |
| MLX 4/6/8-bit | [ddalcu](https://huggingface.co/ddalcu), [vanch007](https://huggingface.co/vanch007), [elishabjm](https://huggingface.co/elishabjm) | | **REFUSED**: MLX is a shared seam this project implements for no model, so it is not a per-model addition |
| proprietary | [infosave/MiniMax-Music-3-cmf](https://huggingface.co/infosave/MiniMax-Music-3-cmf) (Cortiq 4-bit) | | **not implementable**, recorded rather than owed |

**"The GGUF arm" is three mutually incompatible lineages, and
`general.architecture` cannot separate them** — it reads `audiocpp`, `mm3`,
`qwen3` and `wan` across files of the same model, and `wan` collides with genuine
Wan video GGUFs. That is why the detector keys on
`audiocpp.model_spec.family` instead, and why pointing a `.gguf` at this loader
gets a refusal naming the lineage rather than a shape error.

**NOT found** by those queries on that date: AWQ, GPTQ, compressed-tensors, fp8 /
`fp8_e4m3fn` / `fp8_scaled`, bitsandbytes. That is "not found by these queries on
this date", never "does not exist".

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

### MiniMax-Music3: the quantized arms

**One quantized arm loads: the RVQ depth decoder from a GGUF Q4_K file.**
Everything else is the bf16/fp32 diffusers checkpoint — bf16 `language_model` +
`rvq_depth_decoder` + `condition_encoder`, fp32 `transformer` + `vocoder`,
~28.5 GB resident.

The implemented arm is pinned to a specific artifact, because an unpinned
quantized checkpoint is not reproducible:

| Field | Value |
|---|---|
| repo | `audio-cpp/MiniMax-Music3-GGUF` |
| revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| file | `rvq_depth_decoder_q4_k.gguf` (405 752 480 bytes) |
| sha256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |

`MiniMaxMusic3LoadRvqDepthDecoderFromGguf` reads it: 47 tensors as 36 Q4_K
projections, 9 BF16 norms and 2 F16 embedding tables, dequantized to bf16
through the shared `gguf_dequant.h` seam. Only the **audio-cpp lineage** is
read, keyed on `audiocpp.model_spec.family == "minimax_music3"` — not on
`general.architecture`, which reads `audiocpp`, `mm3`, `qwen3` *and* `wan` across
GGUFs of this one model and collides with genuine Wan video checkpoints. The
other two published lineages are refused by name.

**The other quantized formats still refuse**, and quantized MiniMax-Music3
checkpoints do exist in five formats — a survey on 2026-08-14 found fourteen
community repositories. Rather than mis-loading one or failing with a confusing
shape error, `MiniMaxMusic3ResolveCheckpoint`, `MiniMaxMusic3AccountTensors` and
`MiniMaxMusic3LoadConfig` each refuse **by name**:

```
minimax_music3: this checkpoint is QUANTIZED -- GGUF (evidence:
condition_encoder.gguf, language_model_q4_k.gguf, ...; 5 of 5 entries examined
carry the marker). NO quantized arm is implemented for MiniMax-Music3, so this
is REFUSED rather than mis-loaded: a GGUF arm needs a name map, the
GGUF-vs-torch dim reversal, a geometry source, and k-quant dequantization routed
through vllm/model_executor/model_loader/gguf_dequant.h ...
The supported arm is the bf16/fp32 diffusers arm ... The quantized arms are owed
rather than forgotten: phase W7 of .agents/specs/minimax-music3.md, issue #672.
```

Eight formats are diagnosed — GGUF, NVFP4, MXFP4, FP8, INT8, AWQ/GPTQ,
bitsandbytes and MLX — plus an `UNIDENTIFIED` case. Each message names the
evidence found in *your* file, how many entries carried it, what a working arm
would need, and the arm that does load. Detection happens in three places,
because a quantized checkpoint announces itself in three different ways:

| You point us at | Caught by | Because |
|---|---|---|
| a directory of `.gguf` files | the tree walk (depth 2, so `diffusion_models/` and `text_encoders/` count) | there is no component directory and no config to inspect |
| a diffusers-shaped tree whose tensors are quantized | the manifest scan, from safetensors headers only | the sidecars (`weight_scale_2`, `weight_packed`, `qweight`, `absmax`) and the dtype-only formats (fp8, int8) are invisible to a shape check |
| a checkpoint that *declares* it | the config parse | `quantization_config.quant_method`, or MLX's bare `quantization` |

A bare `weight_scale` with no `weight_scale_2` and no `weight_packed` is
reported as unidentified and the message names all three candidate schemes. It
never picks one: guessing yields a finite, correctly shaped, correctly scaled,
**wrong** result that no shape gate can see.

Note if you hold a ComfyUI-format Music3 GGUF: those ship the DiT and condition
encoder only — no language model, no depth decoder, no vocoder — so they cannot
generate audio even once a GGUF arm lands.

The refusal gate needs no checkpoint and no network:

```sh
cmake --build build -j 8 --target test_minimax_music3_quant
./build/tests/test_minimax_music3_quant
```

The Q4_K arm's own gate needs the pinned GGUF and the bf16 checkpoint, and skips
loudly without them:

```sh
CHECKPOINT_ROOT=... \
  ./build/tests/test_minimax_music3_quant_real
```

It does not merely check that the numbers land inside a tolerance. It asserts
the **resident ggml type** of all 47 tensors, checks the dequantized values lie
on the **Q4_K lattice** (at most 16 distinct values per 32-element sub-block —
a structure a bf16 read cannot produce), and bounds the output **two-sidedly**.
The lower bound is the important one: a silent dequant fallback to the bf16
weights lands *closer* to the golden (mean|d| 0.00182) than the genuine
quantized arm (0.0324), so upper bounds alone cannot tell them apart.

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

VLLM_CPP_INDEXTTS2_AUX=$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors/aux.safetensors \
  ./build/tests/test_emovec

The vocoder is a SEPARATE download (`nvidia/bigvgan_v2_22khz_80band_256x`),
which IndexTTS-2.5 fetches rather than ships. Convert it the same way, then:

```sh
VLLM_CPP_INDEXTTS2_BIGVGAN=$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors/bigvgan.safetensors \
  ./build/tests/test_bigvgan
```
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
into a **44100 Hz stereo** waveform. Joining the two halves through
`SpeechRegistry`, the `vllm_speech_*` ABI and the example server is W6, and the
8.6B `Qwen3ForCausalLM` forward at the front of the pipeline is the rest of W2 —
see [the language model](#minimax-music3-the-language-model-and-the-end-to-end-path).

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

## MiniMax-Music3: the language model, and the end-to-end path

The rest of phase W2 of #672, and the piece that made the pipeline whole.
`include/vllm/model_executor/models/minimax_music3_llm.h` carries the
autoregressive loop itself (`encoders.py:299-353`) and the 8.6B
`Qwen3ForCausalLM` at its centre. With it, a request generates a song.

### The `inputs_embeds` entry the dense path did not have

Upstream calls `language_model.model(inputs_embeds=...)` twice and
`input_ids` never (`encoders.py:311`, `:353`), because the frame feedback
`_embed_audio_frame` is a *sum* of one language-model embedding row and seven
depth-decoder rows scaled by `num_codebooks^-0.5` — a continuous vector that
corresponds to no vocabulary entry and that no token id can spell.

`Qwen3DenseModel::ForwardEmbeds` is that door. The Qwen3 family already had it
on its multimodal siblings — `qwen3_vl.h` takes `inputs_embeds_bf16` after
scattering the vision tower's rows into it, and Gemma-4 and Muse-Glimmer do the
same — because upstream's own `Qwen3Model.forward` accepts either input. Only the
**dense** registration had never wired it.

It is additive, and that is asserted rather than argued: feeding the embedding
*of the same token ids* through the new entry reproduces `Forward` **bit for
bit**, in the logits and in the paged KV it wrote, and
`tests/vllm/models/test_qwen3_forward.cpp` checks both. `Qwen3ForCausalLM`,
`LlamaForCausalLM`, `MistralForCausalLM`, `InternLM2ForCausalLM` and
`InternLM3ForCausalLM` all ride that one forward, so nothing less than
bit-identity would do.

`out_hidden` is the second half of the same entry: the post-final-norm rows,
returned from the forward that produced the logits. Music3 reads
`last_hidden_state[:, -1]` and then applies `lm_head` to that very row, so
fetching the two halves with two 8.6B passes would be pure waste.

### `num_condition_layers: 8` does not mean eight transformer layers

Worth stating because it is the reading a fresh implementer reaches for. The
eight rows of a `frame_hiddens` entry are `cat(last_hidden, depth_hidden_1..7)`
(`encoders.py:343`) — **one** language-model hidden state and the **seven**
per-depth-step states of the RVQ decoder. Nothing captures per-layer outputs
from the Qwen3 stack, and nothing needs to.

### Running the gates

The language-model gate drives the real 8.6B bf16 weights **teacher-forced** on
the capture's own `rvq_codes.npy`, and skips loudly without the checkpoint:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_llm_real
```

It stages ~18.5 GB and runs 25 decode steps on CPU — several minutes, most of it
the prefill. It compares 102 400 values against `frame_hiddens[:, :4096]`, ranks
the oracle's own sampled codes under the reproduced guided logits, and pushes the
result through the condition mix to `condition_chunk0.npy`.

The end-to-end gate posts a request at `POST /v1/audio/speech` and asserts the
WAV that comes back:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  VLLM_CPP_MUSIC3_DIT=1 \
  ./build/tests/test_minimax_music3_e2e_real
```

`VLLM_CPP_MUSIC3_DIT=1` is required because the DiT arm is four to eight 2.4B
fp32 host forwards. The generated WAV is written to `build/music3/` so you can
listen to it; nothing under `tests/parity/goldens/` is created or replaced.

### Why no gate compares a generated song to the oracle's

Twice over, and both reasons are structural. The autoregressive codes are a
seeded `torch.multinomial` draw (`encoders.py:94-103`) and the denoise loop's
initial latents are a seeded `randn_tensor` (`denoise.py:117-121`). So both the
code draw and the noise draw are **parameters** — `Music3CodeSampler` and
`Music3NoiseSource` — and a gate supplies the capture's own values where the
engine supplies a seeded draw of its own. That is the only entry at which this
pipeline is comparable to the oracle at all.

What an end-to-end request can honestly be held to is therefore what the gate
asserts: that every stage runs, that the WAV is 44100 Hz 16-bit stereo, that its
length is the one the request's duration implies, and that it is **real audio** —
non-zero, unclipped, non-constant, and with two channels that differ (the stereo
fold is a contiguous split of the 128 latent channels, and an interleave produces
a correctly shaped, correctly ranged, wrong song).

Gemma-4 FP8 xdev prefill (`RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice`) is a
Launch/Finish wrapper: cache pins stay live until host-observed `ev_e` retirement.
Peer-pipe overlap stays off (slot 0 only).
