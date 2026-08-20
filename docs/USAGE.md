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

`--model` also takes a Hugging Face repository name, which the server fetches
into the cache before it binds:

```sh
build/examples/vllm-server --model Qwen/Qwen3-0.6B --port 8000
```

That form needs a build that carries transport layer security. The default
`-DVLLM_CPP_OPENSSL=ON` is the tested path, and every release lane and every
container image uses it; `-DVLLM_CPP_BUILD_BORINGSSL=ON` is offered and has
never been compiled here. A build that mixes the two states across its own
source files refuses to start with exit 2 and a message naming what disagrees,
rather than serving corrupted responses. See [Access Hugging Face
checkpoints](guides/hugging-face-access.md) for the build options,
`--revision`, `--download-dir`, the `HF_*` environment variables, and the
release lanes that carry no fetch. `vllm-cli` and the C ABI still take a local
path only.

That command is measured, not illustrative. On 2026-08-20, on x86_64 with the
default OpenSSL build and an empty `HF_HOME`, it fetched
`Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` from
`huggingface.co`: `model.safetensors` (1503300328 bytes), `tokenizer.json`
(11422654), `vocab.json` (2776833), `merges.txt` (1671853),
`tokenizer_config.json` (9732), `config.json` (726) and
`generation_config.json` (239), 1.5 GB of cache in total. The server then bound
its port and answered `/v1/completions`. A second start with the same `HF_HOME`
reports every file as `already in the cache` and transfers no bytes. Before
[#1511](https://github.com/mudler/vllm.cpp/issues/1511) this command downloaded
nothing at all, because the hub answers with a relative `Location` header that
the client read as a URL.

## Draft with a second checkpoint

Speculative decoding runs a small draft model beside the target and verifies its
proposals losslessly, so the emitted tokens do not change. Pass the draft with
`--speculative-config`:

```sh
build/examples/vllm-server   --model /path/to/target   --speculative-config '{"method":"dflash","model":"/path/to/draft","num_speculative_tokens":7}'
```

[Speculative decoding](SPECULATIVE-DECODING.md) lists the supported methods, the
draft checkpoints each was gated against, and what each one refuses by name.
Drafting is greedy: `draft_sample_method` accepts only `"greedy"`, and any other
value is refused at startup rather than silently ignored.

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

[Model recipes](models/README.md) route you to commands, required weights,
component-specific runtime settings, and known limits for each model family.

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
| LTX-2.5 full DiT | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Full bf16 DiT; declare `--checkpoint-class full` | A mismatched or missing required class is refused |
| LTX-2.5 distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42,018,190,584 bytes | `Lightricks/LTX-2.5` @ `6c7e5e573ac1667efc83407806fe9b0b93730e60` | n/a (non-quantized) | Distilled bf16 DiT; declare `--checkpoint-class distilled` | A mismatched or missing required class is refused |
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

Native Windows release artifacts are not published yet. The Windows CPU and
Vulkan ZIP downloads do not exist until the `v0.0.3-pre.1` prerelease workflow
and post-publication audit succeed. See [Binary releases](RELEASES.md).
<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->
