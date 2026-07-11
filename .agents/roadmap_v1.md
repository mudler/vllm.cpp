# Roadmap v1 — post-MVP
*(user-directed 2026-07-10: this document is the live roadmap; the completed
M0–M3 record is archived at
[completed/roadmap_mvp_v0.md](completed/roadmap_mvp_v0.md).)*

**Context:** both gate models run end-to-end and retain token-exact greedy
correctness, but the historical MVP performance closure is **reopened**. A
2026-07-11 audit proved the vLLM `bench throughput` denominators used
temperature 1 while ours used temperature 0; the 27B arms also used token
budgets 8192 versus 2048. The old 35B 1.02× and 27B 1.007× values are
diagnostic, not acceptance evidence. This document is the canonical index of
what runs next: first restore exact same-workload performance evidence, then
the remaining operational and T1/T2 portfolio. Detailed status lives in the
area matrices; active ownership lives in `coordination.md`; closed execution
blocks live under `completed/`.

## Top-level portfolio

This is the single ordered roadmap table. Detailed capability/status rows live
only in the linked area matrix; active agents/worktrees live in
[coordination.md](coordination.md). The order preserves the post-MVP sequence:
close the MVP operational gates, establish the kernel porting seam, execute T1,
then expand backends and scale-out.

| Order | Block | Big area / outcome | Canonical detailed table | Spike coverage | State | Next gate |
|---:|---|---|---|---|---|---|
| 0 | `ROAD-V1-A` | Restore exact production-vLLM performance closure, online latency, and e2e/nightlies | [`BACKEND-GATE-CUDA-VLLM`](backend-matrix.md), [`SERVE-GATE-ONLINE`](engine-matrix.md), [`KV-PREFIX-CACHE`](engine-matrix.md), [`KV-DEVICE-RESIDENCY`](engine-matrix.md), [`SERVE-ASYNC-LLM`](engine-matrix.md), [`KERNEL-GEMM-NVFP4-W4A4`](kernel-matrix.md), [`SERVE-STREAM-USAGE`](engine-matrix.md), [`SERVE-E2E-NIGHTLY`](engine-matrix.md), [benchmark protocol](benchmark-protocol.md) | Exact pushed-`a531e05` 27B online evidence validates all 12 groups but is below the all-axis floor: c1→c32 median total ratios **0.9679/0.9338/0.9482/0.9547/1.0028/0.9268×**, only **4/2/5/3/10/8 of 20** performance axes and **2/4** memory axes pass. Two c32 repetitions prove an accepted socket can remain unread for 205–207 s because httplib under-spawns streaming workers. Direct-c16 fresh-server traces prove the FP4 M-key aliases 1/2/4/8/16: retuning M=16 moves TPOT from the standard 167.484 ms to 161.72–161.75 ms, matching vLLM's 161.698 ms. The vLLM trace also selects 128×32×256 Stream-K/static-persistent tactics absent locally. W0/W1 component wins and inherited pool debt remain recorded; 35B and nightly are intentionally deferred | `GATING` | repair and gate HTTP stream capacity, then port exact FlashInfer FP4 buckets/single-flight and separately the full tactic family; repeat exact 27B after every iteration until every axis passes. Re-rank merged projections/residual traces only then, run 35B, repair teardown, and finally spike `SERVE-E2E-NIGHTLY`. Later roadmap tracks remain blocked |
| 1 | `ROAD-V1-C1` | Drop-in kernel ABI + complete kernel-family parity | [`BACKEND-ABI-VT`](backend-matrix.md), [kernel matrix](kernel-matrix.md) | exhaustive kernel/dependency inventory and [raw-pointer adapter ABI](specs/dropin-kernel-abi.md) accepted; additive W0 implemented and CPU 94/94. `CLAIM-BACKEND-ABI-W0-GPU-1` repaired the GCC13/doctest blocker without runtime changes; exact sm_121a all-target build, focused CUDA/ABI sanitizer, and both gate-model tests pass at `1141b79`. Cross-arch/trace/A-B and scalar-forwarder/backend-shim debts remain explicit | `PARTIAL` | finish sm_80/sm_90a cross-build plus unchanged-trace/model A/B-memory proof alongside the serving window, then migrate and independently gate one kernel family at a time |
| 2 | `ROAD-V1-C2` | Model families: Llama/Qwen3/Mistral, MoE, Qwen3-Next | [model matrix](model-matrix.md) | all 353 static architecture IDs inventoried; current Qwen wrappers are partial; central type-erased registry + exact reject-unknown + both Qwen registrations implemented at `c707602`, CPU suite 94/94; factory leaf `GATING` on the deferred two-model GPU no-regression campaign ([model-factory-registry.md](specs/model-factory-registry.md)) | `PARTIAL` | run the exact `MODEL-FACTORY-registry` two-model greedy/performance/memory handoff after `CLAIM-SERVE-GATE-1` releases dgx; then mark the factory `DONE` and spike/claim Llama dense |
| 3 | `ROAD-V1-C3` | MTP k=1 + GDN speculative path, then DFlash | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | MTP and DFlash specs written; M-mtp-0 dense/MoE safetensors loaders + standalone forward and ported CPU tests complete; oracle tool/runner locally validated | `GATING` | run M-mtp-0 exact head parity on both gate checkpoints in the first GPU window after `CLAIM-SERVE-GATE-1`, then claim M-mtp-1 scheduler/rejection/GDN-spec integration |
| 4 | `ROAD-V1-C4` | Quantization: llama.cpp breadth/speed, NVFP4/FP8/MX, MLX native | [quantization matrix](quantization-matrix.md) | coverage spike merged; `QUANT-GGUF-CPU-THREADPOOL` W1-W3 implemented and correctness-gated, now `GATING` on its idle-host ≥10x/RSS checkpoint; [compute-in-quant GEMM](specs/gguf-compute-in-quant-gemm.md) and [keep-quant loader](specs/gguf-keep-quant-loader.md) remain `READY`; other leaf specs open | `PARTIAL` | reproduce the threadpool 1-vs-20-thread B4 gate on an exclusive idle CPU host, then claim `QUANT-GGUF-KEEPQ-LOADER` + `QUANT-GGUF-CIQ-GEMM` |
| 5 | `ROAD-V1-C5` | Sliding window, local attention, YaRN/long context | [engine matrix](engine-matrix.md), [coverage view §§2,11](feature-matrix.md#2-kv-cache--memory), [joint spike](specs/sliding-local-yarn-long-context.md) | accepted complete KV/manager/backend/dependency and RoPE formula/cache/apply chains; all W1-W8 implementation leaves are now `GATING`; W5-W8 carry fifteen pinned-source YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK fixtures plus CPU/sanitizer checkpoints | `PARTIAL` | after the serving lock releases, compile/run the shared scaled-RoPE CUDA path and gate restored feature-positive model consumers plus both unchanged Qwen3.6 regressions; retain every open G6-G9 handoff honestly |
| 6 | `ROAD-V1-C6` | Priority and async/overlap scheduling + AsyncLLM streaming serving — RE-PRIORITIZED 2026-07-10: `SERVE-ASYNC-LLM` re-promoted T1→T0 because it blocks order 0 and async scheduling is vLLM's default at the pin (B3) | [engine matrix](engine-matrix.md), [coverage view §1](feature-matrix.md#1-engine-core--scheduling), [async-serving spike](specs/async-serving.md) | W1 `ENG-CORE-BUSY-LOOP` and W4 `ENG-PRIORITY-SCHED` are implemented/`GATING`; W2 `SERVE-ASYNC-LLM` is reopened `ACTIVE`. The exact `a531e05` campaign confirms cpp-httplib under-capacity: two c32 repetitions strand one accepted unread SSE socket for 205–207 s while 31 requests stream normally. W3 `ENG-ASYNC-SCHED` remains READY; W0/W1 device-residency gains remain component evidence | `PARTIAL` | close the claimed HTTP capacity repair with deterministic c32 coverage and an exact ladder A/B, then finish order-0 FP4/27B/35B gates; only afterward proceed with W3 async overlap/default-ON mirror |
| 7 | `ROAD-V1-C7` | Sampling/API controls and logprobs payloads | [engine matrix](engine-matrix.md), [coverage view §6](feature-matrix.md#6-sampling--generation-controls) | implementation slices exist; all audited leaf specs absent | `PARTIAL` | split sampler-core evidence from serving wiring |
| 8 | `ROAD-V1-C8` | Tokenize/detokenize and full metrics | [engine matrix](engine-matrix.md), [coverage view §9](feature-matrix.md#9-serving-surface-openai-api-endpoints-cli-library) | unspiked | `INVENTORIED` | `/metrics` core spike, then utility endpoints |
| 9 | `ROAD-V1-C9` | Mechanical recurring upstream sync | [upstream sync](upstream-sync.md), [porting inventory](porting-inventory.md) | protocol exists; automation P1 open | `PARTIAL` | run next enumerate/classify cycle at pin advance |
| 10 | `ROAD-V1-D1` | NVIDIA target fan-out, ROCm, MLX, Vulkan, XPU, ANE | [backend matrix](backend-matrix.md), [backends strategy](backends.md) | 13 CUDA targets, component rules, platforms and native floors inventoried; SGLang preflight P1 deterministic corpus/client/summary/memory/dry-run harness is implemented and `GATING` with 16 CPU tests, while image/model P2 and binding `BACKEND-GATE-CUDA-SGLANG` evidence remain open | `PARTIAL` | after the active benchmark/queued-GPU window, provision and verify the digest-pinned image, resolve the raw E2E/TPOT detail gap without altering timed semantics, then classify both exact checkpoints in P2 |
| 11 | `ROAD-V1-D2` | Tensor/multi-GPU parallelism | [engine matrix](engine-matrix.md), [coverage view §3](feature-matrix.md#3-parallelism--scale-out) | TP spec written | `READY` | acquire 2-GPU target and claim Phase 0 mock/ABI |
| 12 | `ROAD-V1-D3` | Spec-decode breadth: ngram and EAGLE3 | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | unspiked | `INVENTORIED` | after MTP/DFlash gate |
| 13 | `ROAD-V1-D4` | LoRA, KV/offload, wider model zoo | [engine matrix](engine-matrix.md), [model matrix](model-matrix.md) | corrected expert-streaming spike accepted (`ENG-EXPERT-STREAM` READY): bank-only safetensors→Marlin bank, fixed contiguous cache slots matching Marlin dense strides, logical→slot remap after explicit router D2H, chunked C<E prefill, whole-system GB10 memory gate; mirror floor `ENG-WEIGHT-OFFLOAD` inventoried; LoRA/KV-offload/model-zoo unspiked | `PARTIAL` | claim expert-streaming W0 trace/c1 baseline, then W1 cache policy and W2 bank/reader/pread leaves; rest after T1 dependencies close |

An area row cannot enter `READY` without a real spike under `specs/`, and cannot
enter `DONE` without exact code and test/evidence anchors. Closed execution
blocks move to [completed/](completed/) while permanent support rows remain in
their area matrix.

## A. MVP closing tracks (in flight)

| # | Track | State |
|---|---|---|
| `SERVE-GATE-ONLINE` (formerly A1) | Serve-latency A/B vs `vllm serve` (TTFT/TPOT online, every-axis rule) | 🚧 active and below floor: exact pushed-`a531e05` 27B evidence validates 12/12 groups, but c1→c32 total ratios are 0.9679/0.9338/0.9482/0.9547/1.0028/0.9268×, only 4/2/5/3/10/8 of 20 performance axes and 2/4 memory axes pass. c32 HTTP worker starvation and small-M FP4 cache aliasing are reproduced root causes; direct c16 retuning matches vLLM mean TPOT. Repair those in measured iterations, expand the traced FP4 tactic surface, re-rank residual projection/attention/scheduler gaps, and close every 27B axis before running 35B. W0/W1 component wins and inherited zero-leak debt remain open context; later roadmap work is blocked |
| A2 | GGUF real-file greedy parity on GPU (MVP loader gate) | ✅ **PASSED** — real APEX 35B GGUFs (Compact+Balanced, all supported k-quants), 28/28 assertions, 16/16 greedy token-exact vs same-file llama.cpp oracle, checkpoint-gated test+goldens merged (e2b93cf); remaining breadth: no 27B GGUF exists, NVFP4-type-40 dequant + i-quants deferred |
| A3 | `test_ops_fused_chain` FMA-contraction fix | ✅ merged bf48edb (`-ffp-contract=off` host-wide) |
| A4 | De-Python the build: vendor Triton AOT artifacts per-arch (`triton_aot_vendored/<arch>/` + MANIFEST; `VLLM_CPP_TRITON_REGEN` = maintainer-only Python) | ✅ **DONE** (54367cc..a432461; reproducibility hardening `09f1d23`) — `sm_121a` now has 48 generated C/H files + MANIFEST, including both bf16 `chunk_o` shapes; normal builds remain Python-free. Regen is explicit-target (`cuda:121:32`), line-info-disabled and byte-reproducible across source paths; the pure checker makes source/contract/artifact drift fatal and mutation-tests missing/extra/changed artifacts. A4 remains closed; fresh current-main CUDA/runtime/performance validation belongs to the two ACTIVE `CLAIM-PR3` kernel rows (evidence: porting-inventory §9). |
| `SERVE-E2E-NIGHTLY` (formerly A5) | e2e suites per gates.md (server conformance nightly on dgx etc.) | ☐ next; leaf spike required |

## B. Research tracks (complete)

The B1-B7 parallel research block closed on 2026-07-10. Its frozen questions,
findings and corrections are archived at
[completed/roadmap_v1_research_spikes_2026-07-10.md](completed/roadmap_v1_research_spikes_2026-07-10.md).
Live consequences are carried by the portfolio and area matrices; the archive
is evidence, not a load-bearing decision source.

The follow-on table/model/quant/kernel/backend coverage-spike block is archived
at [completed/roadmap_v1_inventory_spikes_2026-07-10.md](completed/roadmap_v1_inventory_spikes_2026-07-10.md).
The completed control-plane audit and lifecycle-enforcement repair is archived
at [completed/roadmap_v1_control_plane_hardening_2026-07-10.md](completed/roadmap_v1_control_plane_hardening_2026-07-10.md).

## C. T1 queue (all open v0 items carried forward)

This is the complete Post-MVP queue inherited from the
[completed v0 roadmap](completed/roadmap_mvp_v0.md), expanded into track rows so
none of its compact TODO list is lost. Feature-level state and delegation specs
live in [feature-matrix.md](feature-matrix.md).

| # | Track | State |
|---|---|---|
| C1 | **Kernel drop-in alignment**: reshape `vt::` CUDA/ROCm adapter entry points around upstream `csrc` raw-pointer/shape/stride/stream signatures so copied kernels bind with only the Torch tensor wrapper replaced ([backends.md §drop-in](backends.md#post-mvp-drop-in-kernel-compatibility-with-upstream)) | 🚧 [implementation spike accepted](specs/dropin-kernel-abi.md); `BACKEND-ABI-VT` W0 is CPU-green/`GATING` with CUDA and consolidation debts named, then each family migrates with an independent checkpoint |
| C2 | Dense/MoE model families: Llama, Qwen3 dense, Mixtral, then Qwen3-Next | ☐ T1; [feature matrix §4](feature-matrix.md#4-model-families) |
| C3 | MTP speculative decode, starting k=1 on 27B and including the GDN speculative path | 🚧 specs written: [MTP](specs/mtp-spec-decode.md), then [DFlash](specs/dflash-spec-decode.md) |
| C4 | FP8 W8A8 quantization breadth | ☐ T1; [feature matrix §5](feature-matrix.md#5-quantization) |
| C5 | Sliding-window KV/attention and YaRN long-context scaling | 🚧 [joint spike accepted](specs/sliding-local-yarn-long-context.md): all W1-W8 implementation leaves are `GATING`; W5-W8 typed YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK cache construction passes fifteen pinned-source CPU oracle fixtures and sanitizers. CUDA compile/runtime and all model/trace/every-axis gates remain open; gate Qwen3.6 checkpoints are default interleaved MRoPE regressions, with separate feature-positive e2e gates |
| C6 | Priority + async/overlap scheduling + AsyncLLM streaming serving | 🚧 joint spike written: [async-serving.md](specs/async-serving.md); W1/W4 implemented and `GATING`; W2 live SSE works but is reopened `ACTIVE` for the confirmed c32 cpp-httplib capacity repair. Its deterministic transport gate and the order-0 benchmark closure precede W3 default-ON overlap; [feature matrix §1](feature-matrix.md#1-engine-core--scheduling) |
| C7 | `prompt_logprobs`, `logit_bias`, `allowed_token_ids`, and `bad_words` end-to-end API wiring | 🟡 T1; sampler ops exist, engine/serving payload and protocol wiring remain ([feature matrix §6](feature-matrix.md#6-sampling--generation-controls)) |
| C8 | `/tokenize`/`/detokenize` utility endpoints and full Prometheus metrics | ☐ T1; `/metrics` core is the oldest open T0 debt ([feature matrix §9](feature-matrix.md#9-serving-surface-openai-api-endpoints-cli-library)) |
| C9 | Recurring upstream sync cycle and P1 sync tooling | 🔁 recurring; [upstream-sync.md](upstream-sync.md) |

## D. T2 (after T1, per porting-inventory.md)

| # | Track | State |
|---|---|---|
| D1 | Backend expansion: NVIDIA fan-out → ROCm; Apple Metal via MLX (B1 selected E1 over native-MSL-first; M4/16 GB dev host available); Vulkan; loyal Intel XPU port; ANE for encoder/pooling classes | ☐ staged by waves in [feature matrix §12](feature-matrix.md#12-platforms--hardware) and [backends.md](backends.md) |
| D2 | Multi-GPU / tensor parallelism | 🚧 [spec written](specs/tensor-parallelism.md); needs a 2-GPU box because GB10 is single-GPU |
| D3 | Spec-decode breadth beyond MTP/DFlash | ☐ ngram and EAGLE3 after the T1 path |
| D4 | LoRA, KV/offload breadth, and wider model zoo | 🚧 T2 — corrected [expert streaming from disk](specs/expert-streaming.md) contract READY; W0 trace/c1 baseline first; LoRA/KV-offload/zoo still ☐ |

## Protocol evolution (user-directed, 2026-07-10) — mirror as the FLOOR, surpass beyond it

**After the B-track research lands, the protocol gets updated:** until now the
project was a strict 1:1 map of vLLM's features/benchmarks. That remains the
COMPATIBILITY floor (every vLLM feature/behavior mirrored, upstream PRs port
mechanically). But we are no longer capped by it — where our architecture lets
us go FURTHER than vLLM (deeper fusion without Inductor's constraints, single-
pass kernels where vLLM runs two, no-Python runtime overheads, memory), we
pursue beat-vLLM wins as first-class goals. Evidence this is real: we already
BEAT vLLM on peak memory for the currently reported GPU metric, our isolated
fp4 per-shape candidates beat FlashInfer on several shapes even though the
runtime small-M bucket/tactic surface is currently incomplete, and our fused single-pass rmsnorm→fp8-quant and
w13+shared-expert fusion exceed vLLM's kernel structure. The old 27B/35B
end-to-end ratios are now historical diagnostics pending exact reruns; surpass
claims require the same matched-workload evidence as mirror-floor claims.
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
  (B4 criterion). Sub-agents use the flock mutex at `/tmp/gpu`
  per `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md` only when
  2+ agents may run GPU work concurrently (no external contention exists; sole
  owner verified idle via `nvidia-smi` runs lock-free; benchmark series always
  need an uncontended GPU).
