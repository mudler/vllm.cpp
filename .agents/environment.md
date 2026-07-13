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
  - Oracle venv: `~/venvs/vllm-oracle` is now a canonical symlink to the
    validated `~/venvs/vllm-oracle-v0.25.0-stage`; the immediately restorable
    v0.24.0 directory is preserved at
    `~/venvs/vllm-oracle-v0.24.0-retired`. The active stack is pip vLLM 0.25.0,
    FlashInfer Python/cubin 0.6.13, Torch 2.11.0+cu130, CUTLASS DSL 4.5.2,
    Humming kernels 0.1.10, Transformers 5.13.1 and Ninja 1.13.0. Its serving
    dependencies are pandas 2.2.3, python-dateutil 2.9.0.post0, pytz 2024.2 and
    tzdata 2024.2. Install/serving report SHA-256 values are
    `ab786eee…c297` / `536385d8…f506`; executable vLLM/Ninja hashes are
    `ec6d76ff…96c` / `abf71487…10b`, and the sorted freeze hash is
    `cf1636cc…fa5f`.
  - The only dependency-check exception is NVIDIA's
    `nvidia-cusparselt-cu13==0.8.0` wheel: PyPI served the aarch64 wheel
    (`sha256:400c6ed1…77c`), its library is an AArch64 ELF and direct
    `ctypes.CDLL`/Torch imports pass, but its internal WHEEL tag is
    `manylinux2014_sbsa`, so `pip check` reports it unsupported. This is a
    recorded vendor-tag defect, not silently treated as a green `pip check`.
  - Lock-held production-graph validation on the exact 27B snapshot passed both
    offline generation (16 input IDs, one output ID) and the actual text-only
    server: `/health` 200 plus `/v1/completions` 200 with exact 1+1 usage and
    `finish_reason=length`. Server log/response SHA-256 are
    `f56be69a…3787` / `82307db4…8e1` under
    `~/work/vllm-oracle-v0.25.0-stage-validation/2026-07-12-server-smoke`.
    The smoke rate is non-binding. Its first offline inference emitted one
    causal-conv Triton JIT warning, which remains a warmup/trace audit item.
    Online-gate manifests hash pandas package/distribution files plus Ninja and
    reject missing/drifted dependencies before the GPU lock; profiler launches
    prepend the venv `bin` to spawned EngineCore `PATH`.
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

- All old-oracle throughput baselines are non-binding after sampling,
  token-budget and dependency drift audits. `b5c6e4f` is historical vLLM
  0.24.0/FlashInfer 0.6.12 evidence; `3cc490c` is VOID at 28/36 groups,
  1,602/2,016 requests, four returns and no trace. Binding immutable `3f256ab`
  v0.25.0 evidence completes the exact 27B cache-off grid under one lock:
  **55/124 axes pass, 69 fail**. W3-E/W3-F/W3-G strict-fail their components.
  W3-H's exact-target H1d refresh is next: H1a/H1b/H1c and H1d attempts
  through `219f4f2` are void. The latest exact 154-target build and 27B gate
  pass, 48/48 + 16/16 clients complete, the FIFO is removed and target/Nsight
  exit zero. Its continuous range still captures three eager sampler/input gaps
  and emits severity-2 possible loss despite exact intended graph-child counts.
  The revised driver/controller must use four isolated `repeat:4` ranges and
  still enforces
  plan-first evidence, exact
  Triton-AOT/FA2/CUTLASS/toolchain configuration, its own recorded target build
  and actual `nsys -> nsys-launcher -> target-session` ancestry plus a
  diagnostic-only FIFO graceful stop, FIFO removal, zero exit and zero eager
  rows. Require three lossless exact-plan captures before implementing the
  next lever, then close all 69 failed axes before any 35B performance run.
- Keep the existing SGLang v0.5.13 P1 evidence immutable. The distinct
  shared-prefix gate pins v0.5.15 `f63458b` and image digest `d0a667e`; its PX1
  deterministic 64k/256k harness/counter work is ready after the priority
  cache-off closure. Write the dedicated `KV-MAMBA-ALIGN` spike before PX2,
  then require matched BF16/no-spec capacity, native hit/no-eviction evidence,
  full axes and traces. Never mutate the vLLM oracle while provisioning SGLang.
- Bootstrap CMake + MLX on the M4 host before the Metal backend bring-up.
