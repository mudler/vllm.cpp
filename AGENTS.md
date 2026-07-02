# AGENTS.md — vllm.cpp canonical project record

This file is the **single source of truth** for project state, discipline, and
how we work. Every session: read this first, keep it updated (state log at the
bottom), commit it with your changes. Push directly to `main`.

## Mission

1:1 port of vLLM to pure C++ — no Python, no PyTorch at build or run time.
Loyal to the upstream codebase: same architecture, same class/file names, same
algorithms, same config/metric/API surfaces, so that **any future upstream vLLM
PR can be ported here mechanically**.

## The gates (definition of MVP success)

1. **Throughput parity vs vLLM** on `dgx.casa` (DGX Spark, GB10): serve
   **Qwen3.6-35B-A3B (NVFP4)** and **Qwen3.6-27B (NVFP4)** with prefill AND
   decode throughput matching vLLM at large concurrency (request-rate sweeps,
   measured with our `bench serve` equivalent, same box, same models).
2. **GGUF reading**: the same models load and serve from GGUF files (including
   NVFP4 GGUF extension types from the APEX/killgate tooling), not just
   safetensors.
3. **Library-first packaging** (llama.cpp-style): `libvllm` + stable C API
   (`include/vllm.h`, cgo/purego-friendly for LocalAI) + example CLI and
   OpenAI-server binaries.
4. **MVP features**: OpenAI completions/chat, **both streaming (SSE) and
   non-streaming** responses, **tool calling**
   (Qwen + Hermes parsers, auto tool choice), **grammars/structured outputs**
   (xgrammar C++ core: JSON schema/regex/choice/EBNF + GBNF extension),
   core sampling surface, Prometheus metrics with vLLM's metric names.
5. **E2E test suites**: op parity vs upstream dumps, engine behavioral tests
   (ported from upstream `tests/v1/core/` semantics), model parity
   (logits + greedy token-for-token), server conformance, gate benchmark
   regression tracking. CI-runnable on CPU (0.6B model); nightly on dgx.casa.

## Canonical documents

- `docs/porting-inventory.md` — the full vLLM feature/architecture inventory
  with T0 (gate) / T1 / T2 / T3 tier assignments and upstream paths. The
  porting checklist. Keep tiers updated as items land.
- `docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md` — core
  architecture design (tensor runtime `vt::`, engine mirroring, testing).
- This file — state, discipline, environment, progress log.

## Design discipline (non-negotiable)

- **Mirror upstream structure.** C++ sources live at paths that mirror the
  Python: `vllm/v1/core/sched/scheduler.py` → `src/vllm/v1/core/sched/
  scheduler.{h,cpp}`. Same class names, same method names, same field names
  (C++-case-adjusted only where unavoidable). A vLLM developer must be able to
  navigate this repo blind; a PR touching `scheduler.py` maps to exactly one
  place here.
- **Port, don't reinvent.** Read the upstream implementation before writing
  any subsystem. Algorithms, edge cases, defaults, and even comments' intent
  come from upstream. When upstream has a design quirk, we keep it (and note
  why it exists) rather than "improving" it — divergence kills PR portability.
- **Every ported file carries a header comment**: upstream path + upstream
  commit hash it was ported from. When re-syncing with upstream, diff that
  file against its recorded commit.
- **Deviations are exceptional and recorded** in `docs/porting-inventory.md`
  §9 (currently: compute layer replaces torch/Triton with `vt::`; in-process
  queues replace ZMQ; cpp-httplib replaces FastAPI; C-ABI plugins replace
  Python plugins; GGUF promoted to first-class).
- **No PyTorch, no ggml dependencies.** ggml is a design reference only.
  Header-only third-party deps preferred (cpp-httplib, nlohmann/json);
  xgrammar C++ core vendored for grammars; CUDA + cuBLAS for GPU.
- **Parity-first testing.** Nothing is "done" until it matches upstream:
  kernels vs golden dumps, engine behavior vs ported test semantics, models
  vs logits/greedy-decode, server vs OpenAI conformance. Upstream vLLM
  (Python) is a test-time oracle only — never a runtime dependency.
- **Upstream sync**: reference checkout `/home/mudler/_git/vllm`, branch
  `main`. Current sync point: **`e24d1b24` (2026-07-02)**. To advance: fetch,
  ff-pull, update this line, scan `git log <old>..<new> -- vllm/v1/` for
  core-relevant changes, port what applies, log it below.

## Environment & assets

- **Dev box (this machine)**: no GPU. CPU reference backend + engine logic +
  CI development happen here.
- **GPU box**: `ssh dgx.casa` — DGX Spark, GB10 (Blackwell, **sm_121**),
  ~119 GB unified memory, 20 cores, CUDA 12.1-era toolkit (verify with
  `nvcc --version` before kernel work). Unified memory: both gate models fit
  in bf16; the machine is memory-bandwidth-bound (~273 GB/s class) — decode
  parity is a bandwidth/launch-overhead game, hence CUDA graphs + fused
  kernels in T0.
- **Benchmark models on dgx.casa**:
  - `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4`
  - `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`
  - `~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-*.gguf` (GGUF-gate inputs)
  - `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (fast tests)
- **Gate model architecture** (from GGUF metadata, arch `qwen35moe`):
  40 blocks = 10 × (3 GDN + 1 full-attn); hidden 2048; full-attn GQA 16q/2kv,
  partial RoPE 64 dims (MRoPE sections [11,11,10,0]), rope base 1e7; MoE 256
  experts top-8 + 1 shared (expert FFN 512); GDN: conv kernel 4, 16 groups,
  inner 4096, state 128; context 262144.
- **Prior art on dgx.casa** (mudler's llama.cpp patch series — mine these for
  GB10 kernel techniques): `~/killgate_series/` (NVFP4 W4A4 FP4 MMA prefill,
  qwen35moe NVFP4 quant/dedup, MoE decode regraph),
  `~/llama-phase93-qwen3next-gqa-bcast`, `~/llama-phase84-attn-only-source`.
- **TODO**: vLLM is NOT installed on dgx.casa — install it (or container) to
  produce the parity baseline + golden dumps before benchmark work.

## vLLM V1/V2 terminology (for clarity)

There is no "V2 engine". vLLM V0 was removed in 2025; everything lives under
`vllm/v1/`. "V2" refers only to the **Model Runner V2** (`vllm/v1/worker/gpu/`
package, "MRV2" in commit messages), an in-progress rewrite of the model
runner *inside* the V1 engine, gated by `VLLM_USE_V2_MODEL_RUNNER` /
`VllmConfig.use_v2_model_runner` (default: on for an allowlist of
architectures, required for DSpark/diffusion). **We port MRV2**, not the
legacy `gpu_model_runner.py` — upstream development is converging on it.

## How we work

- Brainstorm/design before code (superpowers skills); specs in
  `docs/superpowers/specs/`; TDD + parity harness during implementation.
- Push directly to `main` (user-authorized, for now).
- Long CUDA builds/benchmarks run on dgx.casa over SSH; keep artifacts in
  `~/work/vllm.cpp/` there.
- Benchmarks are honest: same box, same model files, request-rate sweeps,
  report TTFT/ITL/throughput; no cherry-picking.

## State log (append; newest last)

- **2026-07-02** — Project kicked off. Repo created (`mudler/vllm.cpp`),
  pushed to GitHub. vLLM reference mapped (engine core + full feature
  inventory via subagents). Gates defined (Qwen3.6 NVFP4 parity on GB10 +
  GGUF + library + tools/grammars + e2e suites). `docs/porting-inventory.md`
  written with T0–T3 tiers. Core design doc v1 written (needs revision:
  CUDA-first on GB10, hybrid GDN+MoE+NVFP4 in T0 — was CPU-first). Upstream
  synced to `e24d1b24`. NOT STARTED: any implementation. NEXT: user reviews
  inventory + revised design doc → writing-plans for milestone M0.
