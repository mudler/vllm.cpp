# Roadmap v1 — post-MVP
*(user-directed 2026-07-10: this document is the live roadmap; the completed
M0–M3 record is archived at
[completed/roadmap_mvp_v0.md](completed/roadmap_mvp_v0.md).)*

**Context:** the MVP throughput gate is PASSED on both gate models (35B 1.02×,
27B 1.007× vs production vLLM, token-exact, ~25-35% less memory — see
state.md 2026-07-10). This document is the canonical index of what runs next:
the in-flight closing tracks and the ordered T1/T2 portfolio. Detailed status
lives in the area matrices; active ownership lives in `coordination.md`; closed
execution blocks live under `completed/`.

## Top-level portfolio

This is the single ordered roadmap table. Detailed capability/status rows live
only in the linked area matrix; active agents/worktrees live in
[coordination.md](coordination.md). The order preserves the post-MVP sequence:
close the MVP operational gates, establish the kernel porting seam, execute T1,
then expand backends and scale-out.

| Order | Block | Big area / outcome | Canonical detailed table | Spike coverage | State | Next gate |
|---:|---|---|---|---|---|---|
| 0 | `ROAD-V1-A` | MVP operational closure: online latency and e2e/nightlies | [engine matrix](engine-matrix.md), [benchmark protocol](benchmark-protocol.md) | A1 first campaign invalid/incomplete (vLLM startup failed; 35B ours arms aborted), rerun required; A5 unspiked | `GATING` | diagnose/rerun A1 under flock with valid denominators, then spike A5 |
| 1 | `ROAD-V1-C1` | Drop-in kernel ABI + complete kernel-family parity | [kernel matrix](kernel-matrix.md) | exhaustive kernel/dependency inventory accepted; raw-pointer ABI leaf unspiked | `PARTIAL` | spike the adapter ABI, then split family claims |
| 2 | `ROAD-V1-C2` | Model families: Llama/Qwen3/Mistral, MoE, Qwen3-Next | [model matrix](model-matrix.md) | all 353 static architecture IDs inventoried; current Qwen wrappers are partial; factory leaf unspiked | `PARTIAL` | generic architecture-to-factory + reject-unknown spike |
| 3 | `ROAD-V1-C3` | MTP k=1 + GDN speculative path, then DFlash | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | MTP and DFlash specs written | `READY` | claim 27B MTP loader/state leaf |
| 4 | `ROAD-V1-C4` | Quantization: llama.cpp breadth/speed, NVFP4/FP8/MX, MLX native | [quantization matrix](quantization-matrix.md) | comprehensive coverage spike merged; leaf specs open | `PARTIAL` | spike GGUF compute-in-quant storage/dispatch |
| 5 | `ROAD-V1-C5` | Sliding window, local attention, YaRN/long context | [engine matrix](engine-matrix.md), [coverage view §§2,11](feature-matrix.md#2-kv-cache--memory) | unspiked | `INVENTORIED` | SlidingWindowSpec + attention joint spike |
| 6 | `ROAD-V1-C6` | Priority and async/overlap scheduling | [engine matrix](engine-matrix.md), [coverage view §1](feature-matrix.md#1-engine-core--scheduling) | async research finding only; leaf specs absent | `INVENTORIED` | async execution trace + scheduler test inventory |
| 7 | `ROAD-V1-C7` | Sampling/API controls and logprobs payloads | [engine matrix](engine-matrix.md), [coverage view §6](feature-matrix.md#6-sampling--generation-controls) | implementation slices exist; all audited leaf specs absent | `PARTIAL` | split sampler-core evidence from serving wiring |
| 8 | `ROAD-V1-C8` | Tokenize/detokenize and full metrics | [engine matrix](engine-matrix.md), [coverage view §9](feature-matrix.md#9-serving-surface-openai-api-endpoints-cli-library) | unspiked | `INVENTORIED` | `/metrics` core spike, then utility endpoints |
| 9 | `ROAD-V1-C9` | Mechanical recurring upstream sync | [upstream sync](upstream-sync.md), [porting inventory](porting-inventory.md) | protocol exists; automation P1 open | `PARTIAL` | run next enumerate/classify cycle at pin advance |
| 10 | `ROAD-V1-D1` | NVIDIA target fan-out, ROCm, MLX, Vulkan, XPU, ANE | [backend matrix](backend-matrix.md), [backends strategy](backends.md) | 13 CUDA targets, component rules, platforms and native floors inventoried; leaf specs absent | `PARTIAL` | spike the architecture spine, then sm80/sm90 build gates |
| 11 | `ROAD-V1-D2` | Tensor/multi-GPU parallelism | [engine matrix](engine-matrix.md), [coverage view §3](feature-matrix.md#3-parallelism--scale-out) | TP spec written | `READY` | acquire 2-GPU target and claim Phase 0 mock/ABI |
| 12 | `ROAD-V1-D3` | Spec-decode breadth: ngram and EAGLE3 | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | unspiked | `INVENTORIED` | after MTP/DFlash gate |
| 13 | `ROAD-V1-D4` | LoRA, KV/offload, wider model zoo | [engine matrix](engine-matrix.md), [model matrix](model-matrix.md) | unspiked | `INVENTORIED` | after T1 dependencies close |

An area row cannot enter `READY` without a real spike under `specs/`, and cannot
enter `DONE` without exact code and test/evidence anchors. Closed execution
blocks move to [completed/](completed/) while permanent support rows remain in
their area matrix.

## A. MVP closing tracks (in flight)

| # | Track | State |
|---|---|---|
| A1 | Serve-latency A/B vs `vllm serve` (TTFT/TPOT online, every-axis rule) | ⚠️ first flock-held campaign ended 2026-07-10 but is **invalid/incomplete**: vLLM 27B/35B servers failed initialization and multiple ours-35B arms aborted; preserve logs in `~/work/vllm.cpp-latency/latres`, diagnose and rerun with fresh denominators |
| A2 | GGUF real-file greedy parity on GPU (MVP loader gate) | ✅ **PASSED** — real APEX 35B GGUFs (Compact+Balanced, all supported k-quants), 28/28 assertions, 16/16 greedy token-exact vs same-file llama.cpp oracle, checkpoint-gated test+goldens merged (e2b93cf); remaining breadth: no 27B GGUF exists, NVFP4-type-40 dequant + i-quants deferred |
| A3 | `test_ops_fused_chain` FMA-contraction fix | ✅ merged bf48edb (`-ffp-contract=off` host-wide) |
| A4 | De-Python the build: vendor Triton AOT artifacts per-arch (`triton_aot_vendored/<arch>/` + MANIFEST; `VLLM_CPP_TRITON_REGEN` = maintainer-only Python) | ✅ **DONE** (54367cc..a432461) — vendored sm_121a artifacts (40 files, byte-deterministic regen), fresh-clone build green with `VLLM_CPP_TRITON_PYTHON=/nonexistent/python`, 27B gate green on vendored cubins |
| A5 | e2e suites per gates.md (server conformance nightly on dgx etc.) | ☐ next |

## B. Research tracks (complete)

The B1-B7 parallel research block closed on 2026-07-10. Its frozen questions,
findings and corrections are archived at
[completed/roadmap_v1_research_spikes_2026-07-10.md](completed/roadmap_v1_research_spikes_2026-07-10.md).
Live consequences are carried by the portfolio and area matrices; the archive
is evidence, not a load-bearing decision source.

The follow-on table/model/quant/kernel/backend coverage-spike block is archived
at [completed/roadmap_v1_inventory_spikes_2026-07-10.md](completed/roadmap_v1_inventory_spikes_2026-07-10.md).

## C. T1 queue (all open v0 items carried forward)

This is the complete Post-MVP queue inherited from the
[completed v0 roadmap](completed/roadmap_mvp_v0.md), expanded into track rows so
none of its compact TODO list is lost. Feature-level state and delegation specs
live in [feature-matrix.md](feature-matrix.md).

| # | Track | State |
|---|---|---|
| C1 | **Kernel drop-in alignment**: reshape `vt::` CUDA/ROCm adapter entry points around upstream `csrc` raw-pointer/shape/stride/stream signatures so copied kernels bind with only the Torch tensor wrapper replaced ([backends.md §drop-in](backends.md#post-mvp-drop-in-kernel-compatibility-with-upstream)) | ☐ **next to scope**; MVP experience already proves the approach with CUTLASS NVFP4, Marlin, and FlashAttention-2 lifts |
| C2 | Dense/MoE model families: Llama, Qwen3 dense, Mixtral, then Qwen3-Next | ☐ T1; [feature matrix §4](feature-matrix.md#4-model-families) |
| C3 | MTP speculative decode, starting k=1 on 27B and including the GDN speculative path | 🚧 specs written: [MTP](specs/mtp-spec-decode.md), then [DFlash](specs/dflash-spec-decode.md) |
| C4 | FP8 W8A8 quantization breadth | ☐ T1; [feature matrix §5](feature-matrix.md#5-quantization) |
| C5 | Sliding-window KV/attention and YaRN long-context scaling | ☐ T1; [feature matrix §§2,11](feature-matrix.md#2-kv-cache--memory) |
| C6 | Priority scheduling | ☐ T1; [feature matrix §1](feature-matrix.md#1-engine-core--scheduling) |
| C7 | `prompt_logprobs`, `logit_bias`, `allowed_token_ids`, and `bad_words` end-to-end API wiring | 🟡 T1; sampler ops exist, engine/serving payload and protocol wiring remain ([feature matrix §6](feature-matrix.md#6-sampling--generation-controls)) |
| C8 | `/tokenize`/`/detokenize` utility endpoints and full Prometheus metrics | ☐ T1; `/metrics` core is the oldest open T0 debt ([feature matrix §9](feature-matrix.md#9-serving-surface-openai-api-endpoints-cli-library)) |
| C9 | Recurring upstream sync cycle and P1 sync tooling | 🔁 recurring; [upstream-sync.md](upstream-sync.md) |

## D. T2 (after T1, per porting-inventory.md)

| # | Track | State |
|---|---|---|
| D1 | Backend expansion: NVIDIA fan-out → ROCm; Apple Metal via MLX (B1 selected E1 over native-MSL-first; M4/16 GB dev host available); Vulkan; loyal Intel XPU port; ANE for encoder/pooling classes | ☐ staged by waves in [feature matrix §12](feature-matrix.md#12-platforms--hardware) and [backends.md](backends.md) |
| D2 | Multi-GPU / tensor parallelism | 🚧 [spec written](specs/tensor-parallelism.md); needs a 2-GPU box because GB10 is single-GPU |
| D3 | Spec-decode breadth beyond MTP/DFlash | ☐ ngram and EAGLE3 after the T1 path |
| D4 | LoRA, KV/offload breadth, and wider model zoo | ☐ T2 |

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

## E. Delegable area tables

The complete breakdown is split by ownership boundary:
[engine-matrix.md](engine-matrix.md), [feature-matrix.md](feature-matrix.md),
[model-matrix.md](model-matrix.md),
[quantization-matrix.md](quantization-matrix.md),
[kernel-matrix.md](kernel-matrix.md), and
[backend-matrix.md](backend-matrix.md). Each stable row ID is independently
claimable through [coordination.md](coordination.md).

An item moves `INVENTORIED -> SPIKE -> READY -> ACTIVE -> GATING -> DONE`.
`READY` requires a real `.agents/specs/<slug>.md` with upstream/dependency
anchors, exact tests to port, gates, dependencies and row-sized work. `DONE`
requires merged code and exact code/test/evidence anchors. A planned path is
plain text until the spike exists.

## Decision rules carried forward

- Every perf claim: same-box A/B vs the reference, token-exact gated, fresh
  denominators (benchmark-protocol.md). Vendoring needs a MEASURED win first
  (B4 criterion). Sub-agents on shared GPUs use the flock mutex at `/tmp/gpu`
  per `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`.
