# Environment & assets

- **Dev box (no GPU)**: CPU reference backend + engine logic + CI development.
- **GPU box**: `ssh dgx.casa` — DGX Spark, GB10 (Blackwell, **sm_121**),
  ~119 GB unified memory, 20 cores, CUDA toolkit 13.0.88 (nvcc); compute
  capability 12.1 → sm_121. Unified memory: both gate models fit
  in bf16; the machine is memory-bandwidth-bound (~273 GB/s class) — decode
  parity is a bandwidth/launch-overhead game, hence CUDA graphs + fused
  kernels in T0. Give each active claim its own `~/work/<claim>/` directory;
  never share a build tree between agents.
  - Non-interactive SSH does not put nvcc on PATH — prepend
    `export PATH=/usr/local/cuda/bin:$PATH` in remote build commands.
  - Oracle venv: `~/venvs/vllm-oracle` — pip vLLM 0.24.0, used as the
    parity oracle (golden op dumps via `tools/parity/`). Its serving-benchmark
    subset is completed with the pinned CUDA-test dependency set pandas 2.2.3,
    python-dateutil 2.9.0.post0, pytz 2024.2 and tzdata 2024.2; its venv also
    provides executable `bin/ninja` for FlashInfer JIT. Online-gate manifests
    hash pandas package/distribution files plus Ninja and reject missing/drifted
    dependencies before the GPU lock; profiler launches prepend the venv `bin`
    to spawned EngineCore `PATH`.
  - **GPU mutex:** every CUDA test/model/serve/benchmark/profile holds
    `flock /tmp/gpu` for the whole job or multi-arm series WHEN other agents may run GPU work concurrently (sole owner verified idle via `nvidia-smi` may skip), following
    `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`. Compilation,
    source inspection and file transfer do not need the lock. Never kill an
    unowned PID.
  - Disk cleanup 2026-07-10 reclaimed ~368 GB from unrelated cached model sets,
    April-era autoresearch logits/F16-GGUF cache artifacts, the vLLM compile
    cache and stale rebuildable CUDA build trees. Active latency/PR workspaces,
    gate checkpoints, APEX GGUF evidence and sources were preserved; the volume
    had 359 GB free afterward. Maintain at least 200 GB headroom before adding
    competitor images.
- **Apple/Metal box**: `ssh 192.168.68.103` — Mac mini, Apple M4 (10 CPU
  cores), 16 GB unified memory, arm64, macOS 26.5.2. Use it for the MLX-backed
  `vt::` backend, Metal op parity, and small-model bring-up. It cannot hold the
  27B/35B gate models; gate-scale Apple performance needs a larger-memory Mac.
  Verified 2026-07-10: Xcode is installed; CMake and MLX are not yet installed.

## Benchmark models on dgx.casa

- `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4`
  (snapshot complete, ~22G, 3 safetensors shards — re-downloaded 2026-07-03
  after the original snapshot was found incomplete)
- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`
- `~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-*.gguf` (GGUF-gate inputs)
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (fast tests)

## Gate model architecture (from GGUF metadata, arch `qwen35moe`)

40 blocks = 10 × (3 GDN + 1 full-attn); hidden 2048; full-attn GQA 16q/2kv,
partial RoPE 64 dims (MRoPE sections [11,11,10,0]), rope base 1e7; MoE 256
experts top-8 + 1 shared (expert FFN 512); GDN: conv kernel 4, 16 groups,
inner 4096, state 128; context 262144.

## Prior art on dgx.casa (mudler's llama.cpp patch series — mine for GB10 kernels)

- `~/killgate_series/` — NVFP4 W4A4 FP4 MMA prefill, qwen35moe NVFP4
  quant/dedup, MoE decode regraph
- `~/llama-phase93-qwen3next-gqa-bcast`
- `~/llama-phase84-attn-only-source`

## TODO

- Historical offline vLLM throughput baselines are non-binding after the
  sampling/token-budget audit. Exact pushed-`a531e05` 27B online evidence is
  complete and below the every-axis floor; it reproduces cpp-httplib c32 worker
  under-capacity and FP4 small-M key/tactic mismatch. Repair and repeat 27B
  under one whole-series GPU lock before starting the exact 35B campaign.
- Provision SGLang `v0.5.13` (`28b095c`) with the digest-pinned CUDA 13 image
  from [the leaf spike](specs/cuda-sglang-low-concurrency.md) in an isolated
  environment after current GPU claims; never mutate `~/venvs/vllm-oracle`.
  Exact-checkpoint load/corpus work can proceed before `SERVE-ASYNC-LLM`, but
  binding TTFT/ITL execution cannot.
- Bootstrap CMake + MLX on the M4 host before the Metal backend bring-up.
