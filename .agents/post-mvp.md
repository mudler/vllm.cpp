# Post-MVP program (started 2026-07-10, the day both throughput gates passed)

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
| B1 | Apple Metal / **MLX** | vllm-metal vs wiring MLX's C++ API behind `vt::` vs native MSL; paged-KV + nvfp4 gaps; pick ONE route + first milestone | 🔄 workflow wf_24e95c3b |
| B2 | Arches & vendors | per-NVIDIA-arch support matrix (sm_90/80/consumer); **Triton AMD backend for our same 5 GDN kernels → hsaco?**; hipify/hipBLASLt; vendor vLLM ROCm kernels 1:1; Intel SYCL/XPU vs Vulkan | 🔄 workflow wf_24e95c3b |
| B3 | SGLang steals | measured wins over vLLM V1 portable to C++ (overlap scheduling, RadixAttention vs hash prefix, jump-forward grammar decode) | 🔄 workflow wf_24e95c3b |
| B4 | llama.cpp CPU | in-quant compute vs our dequant-to-bf16, repacking, threadpool; **vendor ggml-cpu kernels IF proven faster** (user criterion) | 🔄 workflow + **local Qwen3.5-2B CPU bench agent** (the decision measurement) |
| B5 | Speculative decoding | vLLM V1 spec-decode map; **MTP** (do our gate checkpoints ship heads?); what **dflash** is; ranked route + the high-concurrency caveat | 🔄 agent |
| B6 | Feature parity matrix | the big one-by-one vLLM-feature table (TP/PP/EP, LoRA, multimodal, disagg, pooling, …) — what we have / miss / tier | 🔄 agent → `.agents/feature-matrix.md` |

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

## Decision rules carried forward

- Every perf claim: same-box A/B vs the reference, token-exact gated, fresh
  denominators (benchmark-protocol.md). Vendoring needs a MEASURED win first
  (B4 criterion). Sub-agents on shared GPUs use the flock mutex
  (`~/.claude/skills/sharing-a-gpu-with-flock`).
