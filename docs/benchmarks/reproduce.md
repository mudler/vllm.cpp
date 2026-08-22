# Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| Clock-controlled pin series | `$HOME/gpu.lock` FIRST, then `sudo -n nvidia-smi -lgc 2190` under an always-fires `-rgc` trap; oracle by EXPLICIT PATH, identity asserted per leg; a `gpu_clock_state.py` window per leg |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Same GGUF both arms: ours `-DVLLM_CPP_VULKAN=ON`, llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96`, SUPERSEDED, via `llama-bench`; clean legs only, one `flock $HOME/gpu.lock`. GEMV sweep: `benchmarks/vulkan_gemv_ab.cpp` |
| Which llama.cpp a figure ran | Three revisions on this page, all SUPERSEDED (#1003): fork `237ad9b96` (GB10 CPU, Vulkan, x86, kernel matrix), stock `b9892` (Pi 5), stock `7044859` (Muse Glimmer, #391). Pin is stock `b10451`, unbuilt (#857) |
| Revisions repo-wide | **Five**, not three, enumerated in the [spec](../../.agents/specs/oracle-llamacpp-repin-stock.md). Absent here: stock `030ebb5` (NON-BINDING) and a Poolside fork BRANCH with no commit recorded, behind Laguna's `27.8 tok/s` |

Build flags, environment variables, and the full gate list are in
[BUILD.md](../BUILD.md) and [ENVIRONMENT.md](../ENVIRONMENT.md).
