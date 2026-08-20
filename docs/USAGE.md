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
| Qwen3.8-27B GGUF language model | `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | Q4_K_M text model loads through `--model` | GGUF multimodal forward is missing |
| Qwen3.8-27B GGUF projector | `mmproj-BF16.gguf` | 931,146,432 bytes | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` | BF16 `clip` projector loads and validates through `--mmproj` | No request path runs the loaded projector |
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

### Validating a staged release archive

Native Windows release artifacts are not published yet. The Windows CPU and
Vulkan ZIP downloads do not exist until the `v0.0.3-pre.1` prerelease workflow
and post-publication audit succeed.
<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

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

### Speech and music generation

Use the [model recipe index](#find-a-model-recipe) to open the MiniMax-Music3
server, command-line, and HTTP workflows.
