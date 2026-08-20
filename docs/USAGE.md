# Using vllm.cpp

Use this page for the common ways to run vllm.cpp. Model-specific commands and
specialized tasks have separate indexes below.

## Before you run a model

Build vllm.cpp before you use these commands. See [Building
vllm.cpp](BUILD.md) for CPU, CUDA, Metal, Vulkan, ROCm, and Tenstorrent build
instructions.

The examples use `/path/to/model` for a local model directory. Replace that
path with a compatible checkpoint for the workflow you select.

## Run a local completion

Run one completion with `vllm-cli`:

```sh
build/examples/vllm-cli \
  --model /path/to/model \
  --prompt "The capital of France is" \
  --max-tokens 64
```

Run `build/examples/vllm-cli --help` to list the flags in your build.

## Start the OpenAI-compatible server

Start the server with a local model directory:

```sh
build/examples/vllm-server \
  --model /path/to/model \
  --port 8000 \
  --max-num-seqs 32
```

Send a completion request from another terminal:

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"The capital of France is","max_tokens":64}'
```

The server also supports OpenAI clients that use
`http://localhost:8000/v1` as their base URL. The model-specific guides record
extra files and launch flags when a model needs them.

## Use the C ABI

For an installed library, use the stable public interface in
[`include/vllm.h`](../include/vllm.h). Link `libvllm` and include `vllm.h`.
This example shows the blocking completion shape:

```c
#include "vllm.h"

vllm_model_params model = vllm_model_params_default();
model.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&model, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sampling = vllm_sampling_params_default();
sampling.max_tokens = 64;

vllm_completion output;
if (vllm_complete(engine, "The capital of France is", &sampling, &output) == VLLM_OK) {
    printf("%s\n", output.text);
    vllm_completion_free(&output);
}
vllm_engine_free(engine);
```

## Use the internal C++ library in the source tree

The headers under [`include/vllm/`](../include/vllm/) are source-tree
internals. They are not an installed or stable public ABI. Repository targets
can include these headers and link the internal `vllm::vllm` CMake target.

For example, a source-tree target can load a model directory through
`LoadedEngine`:

```cpp
vllm::entrypoints::EngineParams params;
params.enable_prefix_caching = true;
params.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

See [`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h)
for `LoadedEngine`. The source-tree examples declare their link targets in
[`examples/CMakeLists.txt`](../examples/CMakeLists.txt). External consumers
must use the C ABI in `include/vllm.h`.

## First-line troubleshooting

- Run the executable with `--help` and confirm that you are using the expected
  build directory.
- Check [Environment variables](ENVIRONMENT.md) for settings that can override
  command-line or configuration values.
- Check [Features](FEATURES.md) for the current backend and model surface.
- Read the matching model or task guide before you add model-specific flags.
- If startup fails, use the exact error text to find the refused file, option,
  operation, or checkpoint arm in the focused guides.

## Find a focused guide

[Task guides](guides/README.md) cover workflows that apply to more than one
model family, including offload, compatibility, and backend-specific use.

## Find a model recipe

[Model recipes](models/README.md) route you to commands, required weights, and
known limits for each model family.

## Checkpoint registry

This table identifies the checkpoints used by the model recipes. A model page
lists other published arms when they have not been used as a gated checkpoint.

<!-- checkpoint-registry:begin -->
| Model or component | File | Size | Repository and revision | Quantized SHA-256 | Supported arms | Refused arms or missing part |
|---|---|---|---|---|---|---|
| DSpark for Qwen3.8-27B | `model.safetensors` | 2,718,576,122 bytes | `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | n/a (non-quantized) | Qwen3 DSpark routing | Token-exact decode gate is pending |
| Nemotron-3.5-Lightning-30B | `model-000{01..52}-of-00052.safetensors` | 21,583,809,748 bytes total | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `29f2d1746d8f41e316523194b19018707749b1b1` | `672c8bda10fdec0256e0819e112d2aa3a936cc3e5d311a05fd3ff773ca9a44b9` (first shard) | Device bf16, GQA, and NVFP4 experts; host FP8 Mamba2 and NVFP4 head | GGUF, MTP, and batched decode |
| MiniMax-H3 FL2VA | `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19,864,208,160 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `5e8fa6e960d5fbd547390ceec63fcead275435d8f3bd2466a8a2cbd8c2e361e3` | Q4_K_M `t2va` and `fl2va`, verified end to end | `ref2va` requires the REF2VA partition |
| MiniMax-H3 REF2VA | `MiniMax-H3-REF2VA-Q4_K_M.gguf` | 19,864,208,064 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `17925612821ea3037ffaf5f7f9789f5460e87025385bd45e9ec6c7d536684d56` | Q4_K_M `ref2va`, verified end to end | `t2va` and `fl2va` require the FL2VA partition |
| MiniMax-H3 encoder | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14,576,977,888 bytes | `realrebelai/MiniMax-H3_GGUFs` @ `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `1bf75e038c5895b97b6ea16cc1e3d32076254b06ec3df10657650d86dc82279e` | Q4_K_M text and multimodal conditioning | No separate refused arm recorded |
| MiniMax-H3 pruned FL2VA | `minimax_h3_fl2va_pruned-Q8_0.gguf` | 21,437,786,208 bytes | `unsloth/MiniMax-H3-GGUF` @ `d629413c2e5b51b38c453668b75ca3b06ca92703` | `1c77759fd30e4b41dd4fb341d684518177f544428c6186fd9f5fd96f8ebf55d4` | Pruned Q8_0 loads and renders | Other pruned quant levels load but have not been rendered |
| MiniMax-H3 pruned REF2VA | `minimax_h3_ref2va_pruned-Q8_0.gguf` | 21,414,002,784 bytes | `unsloth/MiniMax-H3-GGUF` @ `d629413c2e5b51b38c453668b75ca3b06ca92703` | `60f8a47434ec9a925f0aea41d9e0db9cb78ebc46791b7488d621dbd6905e5d89` | Pruned Q8_0 loads and renders | Other pruned quant levels load but have not been rendered |
| MiniMax-H3 video VAE | `FL2VA/video_vae/source/model.safetensors` | 10,415,548,320 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official video decode for the five-file recipe | No quantized arm is recorded |
| MiniMax-H3 audio VAE | `FL2VA/audio_vae/model.safetensors` | 605,429,308 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official audio decode for the five-file recipe | No quantized arm is recorded |
| MiniMax-H3 tokenizer | `FL2VA/tokenizer/tokenizer.json` | 7,032,403 bytes | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | n/a (non-quantized) | Official tokenizer for the five-file recipe | No separate arm is recorded |
| MiniMax-Music3 | Diffusers checkpoint tree | about 28.5 GB resident | `MiniMaxAI/MiniMax-Music3` @ `fbdf52fbaaca799592917417eb05f1899f1255ec` | n/a (non-quantized) | bf16 language model, depth decoder, condition encoder; fp32 transformer and vocoder | Native `.pth` layout |
| MiniMax-Music3 depth decoder | `rvq_depth_decoder_q4_k.gguf` | 405,752,480 bytes | `audio-cpp/MiniMax-Music3-GGUF` @ `c36aaeed683f33b05796788e4204f4eeba8fa547` | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` | GGUF Q4_K depth decoder | Other GGUF components and third-party lineages |
| LTX-2.5 full DiT | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Full bf16 DiT | Checkpoint-class validation is owed |
| LTX-2.5 distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Distilled bf16 DiT | Checkpoint-class validation is owed |
| LTX-2.5 distilled NVFP4 DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | Content hash unavailable from the gated repository; #1048 | Distilled NVFP4 DiT | Authenticated content pin is owed |
| LTX-2.5 distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Distilled two-stage recipes; rank and alpha 450; version 2.5.0 | Distinct from the 327,322,640-byte IC-LoRA |
| Qwen3.8-27B mixed FP8 and NVFP4 | `model.safetensors` | 22,568,192,096 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` | NVFP4 modules load | FP8 modules and quantized KV cache are refused |
| Qwen3.8-27B MTP drafter | `model_mtp.safetensors` | 849,400,392 bytes | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | n/a (non-quantized) | BF16 MTP artifact is present | MTP execution is owed |
| Qwen3.8-2.4T-A95B | `UD-Q1_0` ten-file GGUF split | about 370 GiB | `unsloth/Qwen3.8-2.4T-A95B-GGUF` @ `567d3e6ac26c5474b18311e619c04350fb9a5556` | `b7770552b2ac24e7334c917bc92e90e218e87cfe29484db65e62e8ef2a60334d` (shard 1); `2765517f833c736338d3ab34354e1c10eb8d79e62325f998285b435e5cf03dcd` (shard 2) | CPU expert streaming from disk | CUDA refuses a checkpoint that exceeds device capacity |
<!-- checkpoint-registry:end -->

## Look up interface details

[Reference pages](reference/README.md) collect dense lookup material such as
build settings, environment variables, feature state, and release artifacts.

## Temporary legacy reference

The remaining sections preserve the previous usage reference while the
campaign moves each model, guide, and reference topic to its public home.

<!-- legacy-reference:begin -->

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

### FlashAttention-2 is used only where the build compiled it

`--help` will not tell you which architectures your binary carries, so the engine
now checks for itself. At configure time the build records the exact architecture
list it hands nvcc for the FlashAttention-2 kernels, and at run time the CUDA
platform compares your device against that list. Only a match takes the bf16 FA2
attention path; anything else falls back to the f32 graph-captured path, which
produces correct output and is slower.

The configure step prints the list, so you can see it before you run:

```text
-- CUDA FA2 compiled-arch manifest: [121a]
```

An empty list means FlashAttention-2 was not compiled at all — either
`-DVLLM_CPP_FLASH_ATTN=OFF`, or no CUTLASS headers, or none of your requested
architectures has an FA2 kernel body.

This matters because `VLLM_CPP_CUDA_ARCHITECTURES` defaults to `121a` alone. A
default build moved to a different card previously took the FA2 path with no code
for that device; it now takes the fallback. **If FlashAttention-2 seems to have
switched off after you changed cards, rebuild with your architecture in
`VLLM_CPP_CUDA_ARCHITECTURES`** — the manifest is telling you the truth about the
binary rather than about the GPU ([#1357](https://github.com/mudler/vllm.cpp/issues/1357)).

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

### Timing an encode on your own box

`tools/bench/bpe_encode_cost.cpp` times `Tokenizer::Encode` on one synthetic
input, at the sizes you name, through a `tokenizer.json` you name. Use it when
you want to know what a prompt of some shape costs to tokenize here, or to
re-derive a figure somebody else recorded instead of trusting it.

Nothing RUNS it: it is registered as no test and it is not a gate. Both halves
of that are deliberate — a growth ratio over these timings is not stable enough
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
through to `FLASH_ATTN` — harmless until `FLASH_ATTN` was taught FlashAttention-2's
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
while the default — which resolves `FLASH_ATTN`, the reference engine's *first*
preference for that device — dies at the first attention call with
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
| `NemotronHForCausalLM` | **Only BATCHED decode still refuses.** A2-P (#810) narrowed this: `ForwardNemotronHForCausalLM` now selects the paged forward whenever the runner supplies paged KV and recurrent state, so K/V go into the runner's pages and the conv/SSM rows are carried across steps, and `examples/nemotron_h_gen` reaches all of it through `include/vllm.h` alone. What is left is `num_reqs > 1`, refused by name because one request's pages and one request's recurrent state are carried per step and a multi-request step would be decoded as ONE concatenated causal sequence — plausible wrong tokens rather than a failure. Owed to A2-B. **The end-to-end token gate against the pinned oracle has NOT run**, so no claim is made here about what this checkpoint emits; `docs/BENCHMARKS.md` records that as pending rather than as silence. `lm_head` and the FP8 Mamba2 projections still compute on the host, and a GGUF file is refused by name since no GGUF arm exists for it. See *Nemotron-3.5-Lightning-30B: the exact weights, and which arms run* below |

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

## HuggingFace cache and credentials

`--model` takes a local directory or a `.gguf` file. It does not take a
repository identifier yet, and nothing in the tree fetches a checkpoint over the
network. Row `ENG-HF-MODEL-DOWNLOAD`, issue
[#1280](https://github.com/mudler/vllm.cpp/issues/1280), adds that, and this
section records the part of it that has landed.

The library now reads the HuggingFace environment. The values below are resolved
in `vllm/transformers_utils/hf_hub` and `vllm/transformers_utils/hf_cache`, and
they mean what `huggingface_hub` means by them:

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Bearer token for a private or gated repository |
| `HF_TOKEN_PATH` | A file holding that token, read when `HF_TOKEN` is unset |
| `HF_ENDPOINT` | Alternate hub host. A missing trailing slash is added |
| `HF_HUB_OFFLINE` | Resolve from the cache and open no socket |
| `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, `HOME` | The cache root, resolved in that order. `HF_HOME` contributes `$HF_HOME/hub` |

The cache is HuggingFace's documented layout,
`{hub}/models--org--repo/` with `refs`, `blobs` and
`snapshots/{commit}/{path}`, so a host that already holds a Python
`huggingface_hub` cache is read rather than re-downloaded. A repository holding
more than one snapshot resolves to the one written most recently.

Reading that layout is what the server does today. Writing into it is landed
code with no caller yet: where the file system holds no symbolic link, which is
the case for a CIFS mount and can be the case for the `/cache` container volume,
a snapshot entry will become a real file, and the switch will be logged one time
for each cache directory it happens in. The fetcher that calls it is W3 of the
row, so nothing prints that line at this commit.

A repository listing is refused, rather than partly used, when it fails either
of two integrity checks. An object identifier given to two entries that disagree
on the size the listing reported for them is refused, because no content hash
names two sizes. That holds whether or not the two entries name different paths:
one path listed twice at two sizes is self contradictory whichever entry is
believed. An identifier whose characters are all the same, such as one character
repeated 64 times, is refused, because no content hash produces one and that is
the value the hub was measured serving for a gated repository on 17 August 2026.
Neither check depends on `HF_TOKEN`. Entries that share an identifier and agree
on size are accepted, because that is duplicate content and a repository is
allowed to hold it.

Identifiers are compared in one letter case. Hexadecimal is case-insensitive and
the hub emits lower case, so a listing that spelled one identifier `ab23...` on
one entry and `AB23...` on the next is naming one object and both checks see it
that way. A mirror named by `HF_ENDPOINT` therefore cannot switch the size check
off by changing a letter's case, and a cached blob gets the same name on a
case-sensitive file system and on a case-insensitive one.

The size check compares only the sizes a listing actually reported. It reads the
entry's top-level `size` and falls back to `lfs.size`, never to `lfs.pointerSize`
which is the size of the pointer file. An entry that reports no size is compared
against nothing, and it cannot stand in as the reference for the entries that
follow it, so a mirror named by `HF_ENDPOINT` cannot switch the check off by
omitting one field.

Two limits are worth stating plainly. No command-line surface reaches any of
this yet, so setting `HF_TOKEN` today changes nothing a server does. And the
DFlash draft path, which is the one caller that already resolves a repository
identifier against the cache, still reads `$HOME/.cache/huggingface/hub` and
ignores `HF_HOME`. Both are recorded under `## Owed` in
`.agents/specs/hf-model-download.md`.

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

`/v1/audio/speech` is served, but **only** by a server started with
`--speech-model`, and what it can render depends on the family that flag loads
(#1112). MiniMax-Music3 renders: a composed request returns a real 44100 Hz
stereo WAV (#852). **IndexTTS-2.5 does not**: its stages are ported and gated at
reduced dimensions, further stages are named as missing by the checkpoint's own
manifest, and loading the family refuses with a message naming the missing
pieces (#634). Without `--speech-model` the route is a 404 at the route table
rather than a runtime error, which is the accurate signal: the endpoint is opt
in, not absent. See
[Speech and music generation](#speech-and-music-generation).

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
Use the [model recipe index](#find-a-model-recipe) to open the current combined
MiniMax-H3 video and audio workflow.

`/v1/audio/speech` is registered **only** when the server was started with
`--speech-model`; without it the route is absent (404) and the server is
identical to one built before it existed. See
[Speech and music generation](#speech-and-music-generation).

### Speech and music generation

Use the [model recipe index](#find-a-model-recipe) to open the MiniMax-Music3
server, command-line, and HTTP workflows.


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
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. For `mtp`, `num_speculative_tokens` sets the draft DEPTH and defaults to the checkpoint's `mtp_num_hidden_layers`, which is 1 on both gate checkpoints, so the default is unchanged. A value above it must be a multiple of it, mirroring vLLM. Depth cannot move the emitted tokens under greedy decoding, and no speed number is claimed above k=1 yet ([#81](https://github.com/mudler/vllm.cpp/issues/81)). What is gated on CPU at k=1..4 is that the propose runs `k-1` draft decode forwards per propose call, that k drafts reach the verify path, and that the drafts DELIVERED to the verify path vary with depth rather than repeating the first one. That last one is counted over a RUN and never per call, because a correct drafter may resample the same token and this fixture does. Two things are NOT gated there. A draft is never accepted at depth, because acceptance is zero at every depth on the synthetic gate model. And nothing here proves the draft at depth j came from the j-th forward. Both are owed to the GPU gate, which must close the second by comparing the per-depth acceptance RATE against a PADDED control rather than by asserting a non-zero acceptance count, because a padded drafter earns acceptance at depth whenever the target's own greedy continuation repeats a token. `dspark` speculates on the Qwen3.6 gate models (native + Speculators drafts), token-identically to speculative-off, but is not gated on speed: the cross-engine ratio is UNSETTLED, with a matched-and-warm paired measurement of 0.834x against the pinned oracle and the earlier 0.957x-0.989x figures taken against a single COLD oracle invocation on a machine that has since been reimaged. A GGUF target, or a target with no aux multi-tap, is refused by name (`SPEC-DSPARK`). The DRAFT is classified from its own `config.json` rather than from the method string: `Qwen3DSparkModel`, `Gemma4DSparkModel`, and — BEYOND-PIN, mirroring [vllm#52197](https://github.com/vllm-project/vllm/pull/52197) merged 2026-08-17 — `DSparkDraftModel` together with `model_type` `qwen3` all route to the Qwen3 DSpark lane, and every other DSpark draft that DECLARES an architecture is the DeepSeek-V4 variant, which is refused by name because this engine carries only a stub for it (`SPEC-DSPARK-QWEN3-ROUTING`, [#1193](https://github.com/mudler/vllm.cpp/issues/1193)). A draft config carrying no `architectures` key at all is not classified and loads as before, because an absent key is not evidence of a lane. Its sequential Markov sampling runs on device by default; `VT_DSPARK_DEVICE_SAMPLE=0` restores the host loop (token-identical, cost only). The speculative verify runs from a captured CUDA graph, worth +12.2%/+3.5% on the 35B cells; `VT_SPEC_DECODE_GRAPH=0` restores the eager verify (also token-identical). The object is admitted key by key and NOTHING is dropped ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)): the honoured keys are `method`, `num_speculative_tokens`, `model`, `prompt_lookup_min` and `prompt_lookup_max`, plus `draft_sample_method` and `rejection_sample_method` at their upstream defaults `greedy` and `standard`, which are what this engine implements. Any other value of those two names row `SPEC-ACCEPT-VARIANTS` and is refused. A name vLLM's `SpeculativeConfig` declares but this engine does not implement, such as `quantization`, is refused as exactly that, and any other name is refused as unknown with the accepted list. Before this the extra key was discarded, so `draft_sample_method=probabilistic` ran GREEDY and a misspelled `num_speculatve_tokens` took the default, both silently and both at exit 0. For `dspark`, `num_speculative_tokens` may no longer sit BELOW the draft checkpoint's block: DSpark drafts a block, our block is sized from this value alone, and a shorter one drafted a structurally wrong block in silence. It is refused now, before any weight is loaded, naming the block, the config key the block was read from, and the value given ([#1225](https://github.com/mudler/vllm.cpp/issues/1225)). The block is read from the draft config's `dspark_block_size`, or from `block_size` when that key is absent, which is the case on every published Qwen3 draft (`deepseek-ai/dspark_qwen3_4b_block7` and `RadixArk/Qwen3.8-27B-DSpark` both carry `block_size: 7`, so k must be at least 7). vLLM reads only the first key and accepts the shorter value. vLLM also builds its model config BEFORE its speculative config, so a command that names both a target directory it cannot open and a short `k` hears about the target there and about the `k` here. Those are the two recorded divergences, both argued in `.agents/specs/dspark-block-size-guard.md`. A k at or above the block behaves exactly as before. For `dflash`, the DRAFT is likewise classified from its own `config.json`, and a draft that declares `DFlash2DraftModel` is REFUSED at startup, before any weight is read, naming both mechanisms this engine does not implement yet: the grouped dynamic depthwise convolution and the candidate selector (`SPEC-DFLASH2`, [#1314](https://github.com/mudler/vllm.cpp/issues/1314)). It is refused rather than loaded because a DFlash2 checkpoint carries DFlash1's whole tensor set, so the DFlash1 lane would load it with nothing missing and draft worse tokens with no visible symptom: the verify is lossless, so the emitted tokens stay the target's and only acceptance falls. A `DFlashDraftModel` draft is unaffected. A GGUF drafter is classified the same way but by its METADATA, because a GGUF declares no architectures and the published DFlash2 GGUF writes the same `dflash` architecture a DFlash1 one does: a file carrying `dflash.selector_rank`, `dflash.selector_top_k` or `dflash.conv_kernel_size` is refused, and a DFlash1 GGUF, which carries none of them, loads as before. A draft config may also carry a top-level `is_causal`, which now decides every layer's causality ahead of `dflash_config.causal` and ahead of the `layer_types` default, mirroring [vllm#52816](https://github.com/vllm-project/vllm/pull/52816); no published DFlash1 checkpoint declares the key, so their behaviour is unchanged. In a GGUF the same value arrives as `dflash.attention.causal` and is resolved identically. Either spelling is honoured whenever it is DECLARED, as a boolean or as a number, so `"is_causal": 0` means non-causal rather than falling through to the default; a value of any other type is now refused by name instead of being dropped, and the two containers answer alike. When NEITHER explicit key is present, a layer is causal only if its own declared `layer_types` entry is `sliding_attention`. `dflash_config.use_swa` moves the sliding WINDOW onto every layer and no longer makes any layer causal, which is what upstream does ([#1366](https://github.com/mudler/vllm.cpp/issues/1366)); such a draft previously ran every layer causal here and non-causal in vLLM, which cost acceptance and changed no emitted token, so nothing surfaced it. **No checkpoint reaches that arm here yet**, so it changes nothing you can run today: every published DFlash draft that declares `layer_types` also declares no `use_swa`, and the one published draft of the governed shape, `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash`, declares no `layer_types` at all — which this engine's draft-config builder requires, so it fails with the raw `key 'layer_types' not found` before any causality is resolved — while its target architecture `MiMoV2ForCausalLM` is not one this engine serves. A GGUF drafter cannot declare `use_swa` at all. The rule is therefore correct and INERT, and both halves of the gap are owed by `SPEC-DFLASH2` W2 (`.agents/specs/dflash2-spec-decode.md` `## Owed` O4). See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--language-model-only` / `--no-language-model-only` | off | Disable all multimodal input by setting **every** modality limit to 0, mirroring vLLM's flag of the same name. It is not a "skip the encoder" switch: the server then **refuses** a multimodal request with ``400 At most 0 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit.`` It does **not** free VRAM yet — nothing gates tower construction on it ([#607](https://github.com/mudler/vllm.cpp/issues/607) wave L3) |
| `--limit-mm-per-prompt '<json>'` | (unset ⇒ 999 per modality) | Maximum multimodal input items per prompt, per modality, as the same JSON object vLLM's flag takes: `'{"image": 2, "video": 0}'`, or with profiling options `'{"video": {"count": 1, "num_frames": 32}}'` (the options are validated and ignored — they size dummy inputs for memory profiling, which this engine does not do). A limit can only **lower** what the model/seam supports, never raise it. Malformed JSON, a negative count, or an unknown option on `image` / `video` / `audio` is refused at startup rather than defaulted. An unknown option on any other modality name is dropped rather than refused, mirroring upstream, whose fallback `BaseDummyOptions` is the one such dataclass without `extra="forbid"`. Upstream's dotted spelling (`--limit-mm-per-prompt.image 2`) is not accepted here, as for `--kv-transfer-config` and `--speculative-config` |
| `--mmproj <mmproj-*.gguf>` | (unset) | The SECOND GGUF file: a `clip`-architecture multimodal projector beside a `.gguf` `--model`, spelled as llama.cpp spells it. It is read, validated and REFUSED BY NAME before the tokenizer and before any language-model weight byte, so a wrong file costs a message instead of a 17 GB map. Refused when `--model` is not a `.gguf` (a safetensors checkpoint carries its tower in its own shards), when the file's `general.architecture` is not `clip`, when its `clip.projector_type` is not `qwen3vl_merger`, and when it carries `v.patch_embd.weight` without `v.patch_embd.weight.1` — half the input features the temporal patch embedding needs, which cannot be completed without inventing the other half. A `muse-glimmer` projector gets MuseGlimmer's own recorded refusal. **The tower is loaded and held, and nothing runs it yet**: there is no multimodal request path over HTTP for a GGUF model, so today the flag buys you validation and a loaded tower, not an image answer ([#821](https://github.com/mudler/vllm.cpp/issues/821)). Auto-discovery of a sibling `mmproj*.gguf` is deliberately not implemented — a directory holding two unrelated models must not silently fuse them |
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

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 23`,
`include/vllm.h:329`; **47** exported functions, the count of `^VLLM_API `
declarations in that header) suitable for `dlopen` / FFI / LocalAI integration.
This line read `19` and `36` until 2026-08-17 and `21`, `273` and `46` until the
W0 phase log added `vllm_video_last_phase_log`; every one of those numbers was
last true several ABI additions ago, and none of the three is derived by any
gate — the version, the line and the count each drift independently, and the
line number drifts on an edit that adds no ABI at all. The version moved twice
in one day: `mmproj_path` took v22 and the phase log, written as v22 on its own
branch, landed as v23.

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
| v19 | Per-modality multimodal input limits |
| v20 | Speech and music generation through `vllm_speech_*` / `vllm_synthesize` |
| v21 | Device selection on the speech lane (`vllm_speech_model_params.device`, `vllm_speech_engine_device`) |
| v22 | A second GGUF for the multimodal projector (`vllm_model_params.mmproj_path`) |
| v23 | The render phase table: `vllm_video_last_phase_log` names the `phase-log.json` a completed `vllm_video_generate` wrote |

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

### The second GGUF file: a `clip` multimodal projector

A GGUF multimodal model ships as **two** files: the language `.gguf` and a
`clip`-architecture `mmproj-*.gguf` carrying the vision tower. Name the second
one with `--mmproj` (`vllm-server`) or `vllm_model_params.mmproj_path` (C ABI
v22); it is never auto-discovered from a sibling filename, because a directory
holding two unrelated models must not silently fuse them.

```console
./build/examples/vllm-server \
  --model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf
```

What this does today, exactly: the projector is opened, its `clip.*` metadata
and its `v.*` / `mm.*` tensors are read into the same vision tower the
safetensors path builds, and the result is held on the engine. **No forward
consumes it yet** — there is no multimodal request path for a GGUF model on
either the server or the C ABI — so the flag buys validation and a loaded tower,
not an image answer ([#821](https://github.com/mudler/vllm.cpp/issues/821)).

Four things are refused **by name**, all of them before the tokenizer and before
any language-model weight byte is read:

- `--model` is not a `.gguf`. A safetensors checkpoint carries its tower in its
  own shards and needs no projector file.
- the file's `general.architecture` is not `clip` (this is what you get for
  passing the language file twice).
- its `clip.projector_type` is not `qwen3vl_merger`. A `muse-glimmer` projector
  is routed to MuseGlimmer's own recorded refusal instead, which names the
  missing axis.
- it carries `v.patch_embd.weight` without `v.patch_embd.weight.1`. llama.cpp
  writes the temporal patch embedding as two halves; with one of them absent,
  loading would mean inventing the other, and the result would be a fluent,
  wrong model rather than an error.

#### The exact files this was gated against

`--mmproj` was built and gated against the two files below. Both are
**third-party quantizations by Unsloth**, not first-party releases from the model
authors, and a repo id alone is not a pin, because a checkpoint gets re-quantized
in place under an unchanged name.

| Arm | Repo and revision | File | Bytes | sha256 |
|---|---|---|---|---|
| `clip` projector (`--mmproj`) | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `mmproj-BF16.gguf` | 931 146 432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| Q4_K_M language file (`--model`) | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `Qwen3.8-27B-Q4_K_M.gguf` | 17 106 775 008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |

Both sha256 values were computed locally on this project's mirrored copy, not
read back from the hub.

The projector is GGUF v3 with 334 tensors (110 BF16 + 224 F32) and 35 metadata
keys, `general.architecture = clip`, `general.type = mmproj`,
`clip.projector_type = qwen3vl_merger`. Its tower is 27 blocks of hidden 1152,
16 heads, feed-forward 4304, patch 16, spatial merge 2, projected to 5120, with
2304 position embeddings and no DeepStack tap — its
`clip.vision.is_deepstack_layers` is 27 `false` values, and it ships no
`v.deepstack.*` tensor.

To re-run the mapping against those bytes rather than against the synthetic
fixture CI uses, name the file and run the reader's own gate:

```console
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_clip_mmproj_gguf
```

That gate reads the projector alone. To confirm the other half — that a load
which COMPLETES leaves the tower on the engine, reachable through
`LoadedEngine::vision_tower()` — name both files and run the loader's gate:

```console
VLLM_CPP_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_gguf_mmproj_reach
```

This one loads the whole 17 GB language file. Measured on an x86 CPU-only build
reading both files over CIFS: 5 min 37 s and 6 min 22 s in two runs — the wall
time is bound by the share, not by the build — at 33.06 GB peak resident both
times. Do not start it on a box with less than about 40 GB of available memory.

Unset, both cases skip loudly and the gates stay hermetic; CI never reads the
file.

#### The tensor accounting, in CI and on the bytes

Both files now have a **committed manifest** — their tensor names, ggml dims and
type ids and their scalar metadata, no weight bytes — generated by
`scripts/gen-qwen38-27b-gguf-manifest.py` and frozen at
`tests/vllm/models/qwen38_27b_q4km_gguf_manifest.inc` (866 tensors, 51 keys) and
`tests/vllm/models/qwen38_27b_mmproj_gguf_manifest.inc` (334 tensors, 35 keys).
CI accounts both against the loaders' own enumerations with no asset:

```console
./build/tests/test_qwen38_27b_gguf_manifest
```

The load itself now **refuses a file carrying tensors nothing reads**, naming
them, before the tokenizer and before any weight byte. That is the direction
that was silent: a tensor the loader asks for and the file lacks already refuses
by name, and one the file ships and no loader reads was simply dropped. On this
artifact that matters concretely — `Qwen3.8-27B-Q4_K_M.gguf` declares
`qwen35.block_count = 65` with `qwen35.nextn_predict_layers = 1`, so it holds 64
decoder blocks plus an MTP drafter at `blk.64`, and a reader spending the whole
65 on the trunk would load, decode fluently, and be the wrong graph.

To account the shipped bytes instead of the frozen manifest, name either file:

```console
VLLM_CPP_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_qwen38_27b_gguf_manifest
```

That reads only the two headers — no weight data and no 17 GB map — so it costs
seconds rather than the minutes the loader gate above costs. Unset, both live
cases skip loudly.

**What is still owed on these artifacts** is the Q4_K_M arm's token gate against
the pinned llama.cpp, which is `PENDING` on
[#857](https://github.com/mudler/vllm.cpp/issues/857) because that oracle is
recorded `gateable = no`, and any image or video answer at all —
`QUANT-QWEN38-27B-GGUF-ARM`,
[#821](https://github.com/mudler/vllm.cpp/issues/821).

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
`vllm_model_params.mmproj_path` (ABI v22) is in the same position: it loads and
validates the projector, and no C-ABI call can feed the tower an image yet.

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
## Streaming routed experts from disk (capacity mode)

A mixture-of-experts checkpoint larger than the box can hold can be run by
keeping the routed-expert weights on disk and paging slices into a bounded
resident cache. It is **off by default** and it is a **capacity** feature, not a
throughput one: it targets single-user and low-concurrency use, and at high
concurrency every step touches most of the experts, so there is nothing left to
save.

```sh
VT_MOE_EXPERT_STREAM=1 \
VT_MOE_EXPERT_STREAM_SLOTS=4000 \
  ./build/examples/vllm-cli --model /models/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
                   --prompt "The capital of France is" --max-tokens 16
```

## Turning CUDA graph capture off, including the break seam

`VLLM_CPP_CUDAGRAPH=0` disables CUDA graph capture. It reached the six batched
decode drivers as six separate reads of the same name, one copied into each
driver; as of `ENG-CUDAGRAPH-BREAK` W4 (#1307) `src/` holds exactly ONE, in the
shared break-point seam (`src/vt/breakable_graph.cpp`), which reads it once per
process into a function-local static — so a process is in exactly one lane for
its whole life and nothing can toggle it mid-run. **The switch still means what
it always meant.** What changed is that the drivers now agree by construction
instead of by six copies of one parse, and that the lane is fixed at the first
read rather than re-decided whenever a driver is constructed.

With capture off, or on a backend that reports no capture support (Vulkan,
Metal, and the CPU backend), a `vt::GraphCaptureScope` is INERT: it captures
nothing, every `vt::GraphBreak` inside it calls its function and returns, and
the forward runs eager exactly as before. That path is byte-identical to the
non-capturing forward and makes zero backend calls, which is what makes each
migration stage reversible.

Nothing about this is new configuration to learn: there is no new flag, no new
config key and no new command. The seam is a library surface
(`include/vt/breakable_graph.h`), and W1 registers one break point at the dense
attention entry of `Qwen3ForCausalLM`. **Production steps now open a capture
scope, and as of W5 (#1335) ALL NINE decode and draft graphs do**:
`Qwen3DenseDecodeGraph` (W2, #1261), `Qwen3MoeDecodeGraph`, `VoxtralDecodeGraph`
and `DeepseekV2DecodeGraph` (W3, #1291), `Qwen3_5DecodeGraph` with
`Qwen3_5DenseDecodeGraph` (W4, #1307), and the DFlash draft graph, the DeepSeek
V4 decode graph and the Laguna decode graph (W5, #1335). Every one of them opens
the scope in FULL mode, mirroring the decode half of vLLM's v1 default
`CUDAGraphMode.FULL_AND_PIECEWISE`, and a `vt::GraphBreak` inside a FULL scope
takes its pass-through arm — so the switch still changes nothing about the break
point beyond what it already changed about the decode graphs. This paragraph
asserted the opposite until W4: it was written at W1, when it was true, and W2
falsified it without rewriting it here.

**W5 WIDENS WHAT THE SWITCH REACHES, and that is a user-visible change rather
than an internal one.** The three single-shape drivers never read
`VLLM_CPP_CUDAGRAPH` at all: each invented its own name — `VT_V4_DECODE_GRAPH`,
`VT_DFLASH_GRAPH` and `VT_LAGUNA_DECODE_GRAPH` — so before W5 there was no single
setting that turned capture off everywhere. There is now, and the three
per-driver names STAY, because each is a same-binary A/B lever for exactly one
driver rather than a copy of the shared one. Either turns its driver's capture
off; `VLLM_CPP_CUDAGRAPH=0` turns all nine off at once.

Turning capture off on those three does NOT return uncomputed memory, and the
distinction is worth stating because it is invisible to a token gate. An INERT
scope runs the forward eagerly, so the driver's buffers hold real values. A
capture that FAILS is the opposite: under stream capture nothing between the
begin and the failure executed, so those same buffers hold whatever the
allocator last left there. The seam reports the two states apart and every
migrated driver propagates the failure instead of returning the buffer.

**The seam also owns the auxiliary-stream rule as of W5.** A model that forks a
side stream inside a capture — the Laguna decode graph runs its FP4 shared
expert that way — registers the fork with the capture scope, and the scope joins
any fork still outstanding before it closes a segment, because ending a capture
with an unjoined fork fails. There is nothing to configure: registration is part
of the model's fork, and outside a capture both hooks do nothing at all.

**W6 CHANGES WHICH STEPS REACH A DECODE GRAPH AT ALL** (#1374, #1020), and that
is the only user-visible behaviour change in this stage. Until W6 the engine
admitted a step to a decode graph only when its uniform query length equalled
`1 + num_speculative_tokens`, the width CONFIGURED for the engine's lifetime. The
scheduler clamps a request's drafts to the step's token budget, so at
`num_speculative_tokens` above 1 a step every request entered with the same
SHORTER draft prefix -- uniform, and exactly the shape a graph can serve -- got
no graph and ran its verify eagerly, with no log and no counter. The engine now
reads the length the step actually has. Nothing about the emitted tokens changes;
what changes is that fewer steps fall out to the eager path.

`VT_SPEC_GRAPH_MAX_QLENS` bounds that, and its default of `2` is deliberate.
Every captured shape retains an `[S, vocab]` f32 logits block plus an `[S, H]`
hidden, times two ring slots, so admitting every clamped depth would multiply the
resident capture set by `1 + num_speculative_tokens`. The default admits two
distinct speculative query lengths per driver -- the steady-state `1 + k` plus
one clamped one. `0` removes the bound; a larger value widens it. A step past the
bound runs eager, which is what every clamped step did before W6.

Two things W6 does NOT change. A prefill or a mixed batch is still never
captured, on any model: every decode graph in this engine is built for a decode
shape and there is no prefill capture driver, so "graphed except at the break
points" remains a property of the seam rather than of any shipped path. And the
seven drivers that are not the two Qwen3.5 ones still admit only query length 1,
so they are byte-identical across this change.

Building it needs no option. `src/vt/breakable_graph.cpp` and, since W4,
`src/vt/persistent_step_input.cpp` — the capture-stable per-step device input
the migrated drivers stage through, so that a replayed graph reads this step's
values from the address it was captured against — are part of the core `vllm`
library on every platform, because the seam is backend-agnostic and asks nothing
new of any backend.

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
