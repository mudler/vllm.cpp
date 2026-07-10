# Roadmap (post-MVP) — THE project roadmap
*(user-directed 2026-07-10: this document IS the roadmap now; the old
[roadmap.md](roadmap.md) M0–M3 is the archived record of the completed MVP.)*

**Context:** the MVP throughput gate is PASSED on both gate models (35B 1.02×,
27B 1.007× vs production vLLM, token-exact, ~25-35% less memory — see
state.md 2026-07-10). This document is the canonical index of what runs next:
the in-flight closing tracks, the research tracks (each is a live sub-agent /
workflow with a report due), and the T1/T2 queue. Keep it current: mark tracks
DONE with a one-line outcome + where the full report/branch lives.

## A. MVP closing tracks (in flight)

| # | Track | State |
|---|---|---|
| A1 | Serve-latency A/B vs `vllm serve` (TTFT/TPOT online, every-axis rule) | 🔄 agent on dgx |
| A2 | GGUF real-file greedy parity on GPU (MVP loader gate) | 🔄 agent on dgx |
| A3 | `test_ops_fused_chain` FMA-contraction fix | ✅ merged bf48edb (`-ffp-contract=off` host-wide) |
| A4 | De-Python the build: vendor Triton AOT artifacts per-arch (`triton_aot_vendored/<arch>/` + MANIFEST; `VLLM_CPP_TRITON_REGEN` = maintainer-only Python) | 🔄 agent |
| A5 | e2e suites per gates.md (server conformance nightly on dgx etc.) | ☐ next |

## B. Research tracks (kicked off in parallel, reports due)

| # | Track | Question | State |
|---|---|---|---|
| B1 | Apple Metal / **MLX** | vllm-metal vs wiring MLX's C++ API behind `vt::` vs native MSL | ✅ **MLX wins** (Ollama's production swap ~1.6-2×); vendor MLX under vt:: + port vllm-metal's paged MSL as MLX primitives; **hardware-gated: needs a Mac (procurement = critical path)**. → [expansion-map-2026-07-10.md](expansion-map-2026-07-10.md) |
| B2 | Arches & vendors | per-arch matrix; Triton→AMD; SYCL vs Vulkan | ✅ **NVIDIA fan-out nearly FREE** (bf16 path already sm_80+; A100/H100/4090 need ~no new kernels — Wave 1); **Triton GDN → AMD gfx942 via one flag**; Vulkan-before-SYCL call. → map |
| B3 | SGLang steals | measured wins portable to C++ | ✅ **Async/overlap scheduling is vLLM's DEFAULT at our pin — an unmet MIRROR obligation** (we run synchronous); reframed as latency/mirror, not throughput (we already ≥1.0× vs async-vLLM). RadixAttention REJECTED per mirror policy. Kernel steals: no-op. → map |
| B4 | llama.cpp CPU | vendor IF proven faster (user criterion) | ✅ research: **our CPU is a single-threaded scalar loop (zero threads!)** — threadpool = 1-wk prerequisite; **compute-in-quant** = the structural fix (~3.3× decode, ~10× prefill on GGUF). 🔄 local Qwen3.5-2B bench = the decision measurement | 
| B5 | Speculative decoding | MTP + dflash | ✅ **Both gate checkpoints SHIP MTP heads** (bf16, we currently skip `mtp.*` at load!); **DFlash** (block-diffusion drafter, ICML'26) is in-pin WITH published drafts for our exact models + a DGX-Spark community container. Route: MTP k=1 27B → GDN spec path → DFlash. → [spec-decode-scoping-2026-07-10.md](spec-decode-scoping-2026-07-10.md) |
| B6 | Feature parity matrix | the big one-by-one vLLM-feature table (TP/PP/EP, LoRA, multimodal, disagg, pooling, …) — what we have / miss / tier | ✅ **[feature-matrix.md](feature-matrix.md)** — 13 sections, ~80 rows, every row source-verified + carrying its `Spec` delegation link; headline gaps surfaced: `/metrics` (open T0-core debt), async/overlap scheduling (unmet MIRROR obligation, B3), logprobs payload + logit_bias/bad_words API wiring (sampler done, serving not) |

## C. T1 queue (from roadmap.md Post-MVP, unchanged priorities)

- **Kernel drop-in alignment** (user-mandated, unblocked now the gates passed):
  reshape vt CUDA/ROCm kernel entry points to match upstream csrc signatures so
  upstream kernels drop in (backends.md §drop-in). **Next to kick off.**
- Dense/MoE model families (Llama / Qwen3-dense / Mixtral …) · MTP spec decode
  (→ B5 scoping first) · fp8 quant · sliding window · priority scheduling ·
  YaRN · prompt_logprobs / logit_bias / bad_words · tokenize endpoints · full
  metrics · Qwen3-Next.
- Recurring: upstream **sync cycle** (upstream-sync.md) + P1 sync tooling.

## D. T2 (after T1, per porting-inventory.md)

Backend expansion per backends.md (Metal/MLX per B1, Vulkan, Intel XPU, ANE),
multi-GPU/TP, spec-decode breadth, LoRA, offload, model zoo breadth.

## Protocol evolution (user-directed, 2026-07-10) — mirror as the FLOOR, surpass beyond it

**After the B-track research lands, the protocol gets updated:** until now the
project was a strict 1:1 map of vLLM's features/benchmarks. That remains the
COMPATIBILITY floor (every vLLM feature/behavior mirrored, upstream PRs port
mechanically). But we are no longer capped by it — where our architecture lets
us go FURTHER than vLLM (deeper fusion without Inductor's constraints, single-
pass kernels where vLLM runs two, no-Python runtime overheads, memory), we
pursue beat-vLLM wins as first-class goals. Evidence this is real: we already
BEAT vLLM on peak memory (~25-35% less), our fp4 per-shape autotune beats
flashinfer's on several shapes, our fused single-pass rmsnorm→fp8-quant and
w13+shared-expert fusion exceed vLLM's kernel structure, and 27B/35B are ≥1.0×.
The updated protocol will define: (a) the mirror floor (unchanged discipline),
(b) the surpass track (fusion-first, measured vs vLLM as the baseline to BEAT,
not just match), (c) how surpass-side divergences are recorded so upstream
sync stays mechanical. Owner: after B1-B6 reports land.

## E. Feature roadmap (the delegable breakdown)

The feature-level roadmap is **[feature-matrix.md](feature-matrix.md)** — one row
per vLLM feature (parallelism, quant, spec decode, multimodal, serving, …) with
our status and a **`Spec` column** linking to `.agents/specs/<feature-slug>.md`.
Convention: a gap row becomes a unit of delegable work by WRITING its spec at the
linked path (scope, upstream file:line to mirror, gates, A/B plan, **and a
"Tests to port" inventory of the upstream test files/cases covering the feature
— see [test-porting.md](test-porting.md)**) and handing
that spec to a sub-agent. Specs live in `.agents/specs/`; the matrix row flips
☐ → 🚧 (spec written / agent running) → ✅ (merged + gated).

## Decision rules carried forward

- Every perf claim: same-box A/B vs the reference, token-exact gated, fresh
  denominators (benchmark-protocol.md). Vendoring needs a MEASURED win first
  (B4 criterion). Sub-agents on shared GPUs use the flock mutex
  (`~/.claude/skills/sharing-a-gpu-with-flock`).
