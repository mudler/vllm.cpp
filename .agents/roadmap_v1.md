# Roadmap v1 — post-MVP
*(user-directed 2026-07-10: this document is the live roadmap; the completed
M0–M3 record is archived at
[completed/roadmap_mvp_v0.md](completed/roadmap_mvp_v0.md).)*

**Context:** both gate models run end-to-end and retain token-exact greedy
correctness. Exact performance closure is open against vLLM v0.25.0; the
immutable 27B result `3f256ab` binds at **55/124 axes pass, 69 fail**. Restore
every throughput, latency, and memory axis on 27B and then 35B before resuming
the operational and T1/T2 portfolio. Detailed status lives in the area
matrices, active ownership in `coordination.md`, and chronological evidence in
the append-only state/ledger record.

**Current order-0 substage (2026-07-14):** binding `3f256ab` remains 55/124.
At c2 ours has better TTFT but **6.1% slower TPOT**. FP4 tactics and merged BA
topology are already matched; the active leaf is
[packed GDN decode](specs/gdn-packed-decode.md). W1D2 now implements the exact
27B CUDA pure-non-spec dispatch and rollback. Clean `f344dec` closes correctness
at **235/235 + 16/16** for both arms. Clean `7ff713e`, finalized by `24cea4f`,
closes structure: packed **915** versus rollback **963** nodes, with exactly 48
packed calls replacing 48 decomposed plus 48 post-conv calls and all unrelated
topology invariant. The production-build-only c2/c16 AB/BA/AB runner and
marker-last every-axis finalizer are now implemented; focused CPU validation is
**49/49** and all tool tests pass **132/132**. It binds both corpus manifests and every partition, exactly
recomputes the raw metrics retained by vLLM, bounds its unexported-latency clock
skew, rejects impossible durations, and rejects symlinked evidence before sealing.

Clean `d82d282` completed both model gates and all c2 legs, then failed
incomplete at c16 packed r1: preflight and 16 warmups passed, but all **0/96**
timed requests returned HTTP 500 and no terminal marker was sealed. The root
cause was unrecoverable because our port dropped two upstream fatal log lines. A
bounded **test-first diagnostic checkpoint** now restores them: four
unconditional `std::cerr` error-path channels (`engine-fatal:` busy-loop guard,
`async-llm:` output handler, `api-server:` both 500 sites, `sse:` mid-flight)
grounded in vLLM `core.py:1233`+`async_llm.py:703-705`, an opt-in
`VT_GDN_DIAG_STEP_LOG` geometry trace, and a packed-only `--diagnostic-c16`
driver mode isolated from the gating component (`component-diagnostic.json`; the
finalizer refuses diagnostic evidence). Six RED→GREEN tests landed it. Next: run
the DGX `--diagnostic-c16` reproduction from the pushed SHA in a fresh
`diagnostic-c16` root, read the exposed cause, repair test-first, and rerun all
**40+8 / 144 paired** axes; the `d82d282` root stays untouched. qkvz stays
excluded. Host PSS/RSS separately retains a **22.920 GiB** CPU weight mirror plus
source mmap residency.

**Order-0 re-ranking (2026-07-14
[parity rescan](specs/parity-rescan-2026-07-14.md), diagnostic only):** the
failing mass is **host-side, not kernel compute** — TTFT passes 24/24, our GPU
kernels are collectively net faster than vLLM's on the only per-kernel window
(−3.579 ms), and the 69 failing axes decompose into c2–c8 decode latency
(52.2% + 24.3% coupled) plus host memory (23.6%). Consequences: (a) the c16
diagnostic → component chain stays first; (b) two kernel-independent host
workstreams start in parallel — **TCP_NODELAY on the SSE server** (confirmed
unset in cpp-httplib while vLLM's uvicorn enables it; spike required before
ACTIVE) and the **memory precheck → weight-streaming loader** track; (c) the
nsys full-step c2 gap diff attributes transport vs `ENG-ASYNC-SCHED` W3 before
W3 is implemented; (d) FP4-producer/PDL/fused-norm-quant micro-levers are
deprioritized — the recorded "fused RMSNorm→NVFP4" gap is **disproven** (vLLM's
fusion pass is FP8-only) and the H1d fused-producer ranking is off-axis. qkvz
stays real-but-small (~0.476 ms/window) behind the component gate.

## Top-level portfolio

This is the single ordered roadmap table. Detailed capability/status rows live
only in the linked area matrix; active agents/worktrees live in
[coordination.md](coordination.md). The order preserves the post-MVP sequence:
close the MVP operational gates, establish the kernel porting seam, execute T1,
then expand backends and scale-out.

| Order | Block | Big area / outcome | Canonical detailed table | Spike coverage | State | Next gate |
|---:|---|---|---|---|---|---|
| 0 | `ROAD-V1-A` | Restore exact performance closure against the faster applicable vLLM v0.25.0/SGLang floor before broader roadmap implementation | [`BACKEND-GATE-CUDA-VLLM`](backend-matrix.md), [`BACKEND-GATE-CUDA-SGLANG`](backend-matrix.md), [`BACKEND-GATE-CUDA-SGLANG-PREFIX`](backend-matrix.md), [`SERVE-GATE-ONLINE`](engine-matrix.md), [`KV-PREFIX-CACHE`](engine-matrix.md), [`KV-MAMBA-ALIGN`](engine-matrix.md), [`KV-DEVICE-RESIDENCY`](engine-matrix.md), [`SERVE-ASYNC-LLM`](engine-matrix.md), [`KERNEL-GEMM-BF16`](kernel-matrix.md), [`KERNEL-GEMM-NVFP4-W4A4`](kernel-matrix.md), [`KERNEL-ATTN-FA2`](kernel-matrix.md), [`KERNEL-GDN-PACKED-DECODE`](kernel-matrix.md), [`KERNEL-GDN-AOT-BF16`](kernel-matrix.md), [`SERVE-STREAM-USAGE`](engine-matrix.md), [`SERVE-E2E-NIGHTLY`](engine-matrix.md), [benchmark protocol](benchmark-protocol.md) | v0.25.0 target `702f481` is audited and `3f256ab` binds at **55/124**. Packed correctness/structure is accepted. Clean `d82d282` completed model gates/c2, then failed incomplete at c16 packed r1 with 0/96 HTTP 500 responses and no terminal marker. A test-first diagnostic checkpoint landed (four error-path log channels + `VT_GDN_DIAG_STEP_LOG` + `--diagnostic-c16` mode) to expose the dropped cause; DGX diagnostic reproduction pending. `benchmark_binding=false`; qkvz, host-memory repair, SGLang and a newer binding result remain open | `GATING` | run the DGX `--diagnostic-c16` reproduction from the pushed SHA, read the captured cause, repair test-first, then rerun the full one-lock component from a new SHA/root before qkvz |
| 1 | `ROAD-V1-C1` | Drop-in kernel ABI + complete kernel-family parity | [`BACKEND-ABI-VT`](backend-matrix.md), [kernel matrix](kernel-matrix.md) | exhaustive kernel/dependency inventory and [raw-pointer adapter ABI](specs/dropin-kernel-abi.md) accepted; additive W0 implemented and CPU 94/94. `CLAIM-BACKEND-ABI-W0-GPU-1` repaired the GCC13/doctest blocker without runtime changes; exact sm_121a all-target build, focused CUDA/ABI sanitizer, and both gate-model tests pass at `1141b79`. Cross-arch/trace/A-B and scalar-forwarder/backend-shim debts remain explicit | `PARTIAL` | finish sm_80/sm_90a cross-build plus unchanged-trace/model A/B-memory proof alongside the serving window, then migrate and independently gate one kernel family at a time |
| 2 | `ROAD-V1-C2` | Model families: Llama/Qwen3/Mistral, MoE, Qwen3-Next | [model matrix](model-matrix.md) | current pin has 353 static IDs; v0.25.0 adds three sync-target rows (MOSS-Transcribe-Diarize, Laguna DFlash, Bailing hybrid MTP), yielding 356 after pin advance. Current Qwen wrappers and the type-erased factory remain partial/`GATING` | `PARTIAL` | after performance closure and target pin advancement, run the two-model factory no-regression handoff, then spike/claim Llama dense |
| 3 | `ROAD-V1-C3` | MTP k=1 + GDN speculative path, then DFlash, DSpark and heterogeneous-vocabulary TLI | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | MTP and DFlash specs exist; M-mtp-0 loader/standalone work is `GATING`. DSpark is user-promoted scope with DeepSeek-V4/Qwen3 draft models, reduced-vocabulary handling and full-CUDA-graph behavior inventoried under `SPEC-DSPARK`; tokenizer-agnostic target↔draft mapping is separately inventoried as `SPEC-TLI`. Their dedicated spikes are not written | `GATING` | after 27B/35B speed parity, close M-mtp-0 and MTP integration, then execute DFlash, write the DSpark spike/gates and compose it with TLI where vocabularies differ |
| 4 | `ROAD-V1-C4` | Quantization: llama.cpp breadth/speed, NVFP4/FP8/MX, MLX native | [quantization matrix](quantization-matrix.md) | coverage spike merged; `QUANT-GGUF-CPU-THREADPOOL` W1-W3 implemented and correctness-gated, now `GATING` on its idle-host ≥10x/RSS checkpoint; [compute-in-quant GEMM](specs/gguf-compute-in-quant-gemm.md) and [keep-quant loader](specs/gguf-keep-quant-loader.md) remain `READY`; other leaf specs open | `PARTIAL` | reproduce the threadpool 1-vs-20-thread B4 gate on an exclusive idle CPU host, then claim `QUANT-GGUF-KEEPQ-LOADER` + `QUANT-GGUF-CIQ-GEMM` |
| 5 | `ROAD-V1-C5` | Sliding window, local attention, YaRN/long context | [engine matrix](engine-matrix.md), [coverage view §§2,11](feature-matrix.md#2-kv-cache--memory), [joint spike](specs/sliding-local-yarn-long-context.md) | accepted complete KV/manager/backend/dependency and RoPE formula/cache/apply chains; all W1-W8 implementation leaves are now `GATING`; W5-W8 carry fifteen pinned-source YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK fixtures plus CPU/sanitizer checkpoints | `PARTIAL` | after the serving lock releases, compile/run the shared scaled-RoPE CUDA path and gate restored feature-positive model consumers plus both unchanged Qwen3.6 regressions; retain every open G6-G9 handoff honestly |
| 6 | `ROAD-V1-C6` | Priority and async/overlap scheduling + AsyncLLM streaming serving — RE-PRIORITIZED 2026-07-10: `SERVE-ASYNC-LLM` re-promoted T1→T0 because it blocks order 0 and async scheduling is vLLM's default at the pin (B3) | [engine matrix](engine-matrix.md), [coverage view §1](feature-matrix.md#1-engine-core--scheduling), [async-serving spike](specs/async-serving.md) | W1 `ENG-CORE-BUSY-LOOP`, W2 `SERVE-ASYNC-LLM`, and W4 `ENG-PRIORITY-SCHED` are implemented/`GATING`; W3 remains `READY`. Accepted `3812d8` completes its order-0 control at only **+0.215%** total credit with a TTFT regression and no GPU-time reduction; no W3 claim or speed credit exists | `PARTIAL` | keep W3 as later parity work; execute order-0 low-batch kernel mapping before returning to the remaining C6 gates |
| 7 | `ROAD-V1-C7` | Sampling/API controls and logprobs payloads | [engine matrix](engine-matrix.md), [coverage view §6](feature-matrix.md#6-sampling--generation-controls) | implementation slices exist; all audited leaf specs absent | `PARTIAL` | split sampler-core evidence from serving wiring |
| 8 | `ROAD-V1-C8` | Tokenize/detokenize, unified streaming parsing and full metrics | [engine matrix](engine-matrix.md), [coverage views §§7,9](feature-matrix.md#7-structured-outputs--tool-calling) | v0.25.0 Streaming Parser Engine (`TOOLS-STREAMING-PARSER`) and opt-in response timing (`SERVE-RESPONSE-METRICS`) are inventoried alongside the older `/metrics`/utility debt; all remain unspiked | `INVENTORIED` | `/metrics` core spike, then Streaming Parser Engine/response timing and utility endpoints |
| 9 | `ROAD-V1-C9` | Mechanical recurring upstream sync | [upstream sync](upstream-sync.md), [v0.25 audit](sync/2026-07-12-702f481.md), [porting inventory](porting-inventory.md) | v0.25.0 target `702f481` enumerated/classified: 145 post-pin commits, 94 inventory and 51 ignore, no unequivalent PORT-NOW runtime delta in the implemented T0 slice. The executable DGX oracle is validated/active at the target; the porting pin remains `e24d1b24` pending target goldens/behavior/model re-verification | `PARTIAL` | refresh exact performance denominators and target goldens/tests, then advance the parity pin |
| 10 | `ROAD-V1-D1` | NVIDIA target fan-out, ROCm, MLX, Vulkan, XPU, ANE | [backend matrix](backend-matrix.md), [backends strategy](backends.md) | 13 CUDA targets, component rules, platforms and native floors inventoried. SGLang v0.5.13 preflight P1 remains implemented/`GATING` with 16 CPU tests; image/model P2 and cache-neutral binding evidence remain open. A distinct v0.5.15 shared-prefix row is now fully spiked/`READY`, with the external scalar rejected and PX1/PX2 exact long-prefix harness plus Mamba-align retention next | `PARTIAL` | after the active 27B cache-off closure, repin/provision the digest-pinned v0.5.15 image, resolve raw E2E/TPOT detail without changing timed semantics, classify exact checkpoints, and execute the shared-prefix gate only after hit/dtype/capacity equivalence |
| 11 | `ROAD-V1-D2` | Tensor/multi-GPU and MoE sequence parallelism | [engine matrix](engine-matrix.md), [coverage view §3](feature-matrix.md#3-parallelism--scale-out) | TP spec written; v0.25.0 non-DP MoE sequence-parallel path is inventoried as `PAR-SEQUENCE-MOE` but unspiked | `READY` | acquire 2-GPU target and claim Phase 0 mock/ABI; its execution trace determines whether sequence parallel is part of the first performance slice |
| 12 | `ROAD-V1-D3` | Spec-decode breadth: ngram and EAGLE3 | [engine matrix](engine-matrix.md), [coverage view §8](feature-matrix.md#8-speculative-decoding) | unspiked | `INVENTORIED` | after MTP/DFlash gate |
| 13 | `ROAD-V1-D4` | External KV-cache provider interoperability, with LMCache as the first end-to-end target | [`KV-EXTERNAL-CACHE`](engine-matrix.md), [coverage view §2](feature-matrix.md#2-kv-cache--memory), [LMCache quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) | dedicated row now inventories pinned-vLLM `KVTransferConfig`, producer/consumer/both roles, scheduler/worker connector lifecycle, dynamic external connector modules, load-failure policy, built-in `LMCacheConnectorV1` and `LMCacheMPConnector`, plus the official standalone-service/shared-prefix workflow. No local connector ABI, LMCache execution or benchmark is claimed; the implementation spike remains unwritten | `INVENTORIED` | write `specs/external-kv-cache-lmcache.md`: first port the generic connector ABI with a deterministic fake provider, then gate LMCache MP store/retrieve across two engine instances, Qwen3.6 hybrid-cache behavior, failure/recompute semantics, observability, correctness, TTFT and transfer throughput; in-process mode follows |
| 14 | `ROAD-V1-D5` | LoRA, local KV/weight offload, expert streaming, wider model zoo | [engine matrix](engine-matrix.md), [model matrix](model-matrix.md) | corrected expert-streaming spike accepted (`ENG-EXPERT-STREAM` READY): bank-only safetensors→Marlin bank, fixed contiguous cache slots matching Marlin dense strides, logical→slot remap after explicit router D2H, chunked C<E prefill, whole-system GB10 memory gate; mirror floor `ENG-WEIGHT-OFFLOAD` inventoried; LoRA/local KV-offload/model-zoo unspiked | `PARTIAL` | claim expert-streaming W0 trace/c1 baseline, then W1 cache policy and W2 bank/reader/pread leaves; rest after T1 dependencies close |

An area row cannot enter `READY` without a real spike under `specs/`, and cannot
enter `DONE` without exact code and test/evidence anchors. Closed execution
blocks move to [completed/](completed/) while permanent support rows remain in
their area matrix.

## A. MVP closing tracks (in flight)

| # | Track | State |
|---|---|---|
| `SERVE-GATE-ONLINE` (formerly A1) | Serve-latency A/B vs `vllm serve` (TTFT/TPOT online, every-axis rule) | 🚧 immutable `3f256ab` is **FAILED/open** at 55/124. Packed structure is accepted; clean `d82d282` failed incomplete at c16 packed r1 with 0/96 HTTP 500 responses and no terminal marker. A test-first diagnostic checkpoint landed (error-path log channels + `--diagnostic-c16` mode); run the DGX diagnostic reproduction, then repair/rerun before qkvz; repair the 22.920 GiB host mirror before 35B performance |
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
| C3 | MTP speculative decode, starting k=1 on 27B and including the GDN path, then DFlash, DSpark and TLI | 🚧 MTP and DFlash specs written; **DSpark explicitly in scope** under `SPEC-DSPARK`; heterogeneous-vocabulary mapping is independently inventoried under `SPEC-TLI`; their spikes queue after parity/MTP |
| C4 | FP8 W8A8 quantization breadth | ☐ T1; [feature matrix §5](feature-matrix.md#5-quantization) |
| C5 | Sliding-window KV/attention and YaRN long-context scaling | 🚧 [joint spike accepted](specs/sliding-local-yarn-long-context.md): all W1-W8 implementation leaves are `GATING`; W5-W8 typed YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK cache construction passes fifteen pinned-source CPU oracle fixtures and sanitizers. CUDA compile/runtime and all model/trace/every-axis gates remain open; gate Qwen3.6 checkpoints are default interleaved MRoPE regressions, with separate feature-positive e2e gates |
| C6 | Priority + async/overlap scheduling + AsyncLLM streaming serving | 🚧 joint spike written: [async-serving.md](specs/async-serving.md); W1/W2/W4 are implemented and `GATING`. W2 fixed capacity is CPU/sanitizer-green and GPU-classified healthy/steady-state-neutral; broader every-axis parity remains. Order-0 FP4 closure precedes W3 default-ON overlap; [feature matrix §1](feature-matrix.md#1-engine-core--scheduling) |
| C7 | `prompt_logprobs`, `logit_bias`, `allowed_token_ids`, and `bad_words` end-to-end API wiring | 🟡 T1; sampler ops exist, engine/serving payload and protocol wiring remain ([feature matrix §6](feature-matrix.md#6-sampling--generation-controls)) |
| C8 | `/tokenize`/`/detokenize`, unified Streaming Parser Engine, response timings and full Prometheus metrics | ☐ T1; `/metrics` core is the oldest open T0 debt, with `TOOLS-STREAMING-PARSER` and `SERVE-RESPONSE-METRICS` added by the v0.25.0 inventory ([feature matrix §§7,9](feature-matrix.md#7-structured-outputs--tool-calling)) |
| C9 | Recurring upstream sync cycle and P1 sync tooling | 🔁 recurring; [upstream-sync.md](upstream-sync.md) |

## D. T2 (after T1, per porting-inventory.md)

| # | Track | State |
|---|---|---|
| D1 | Backend expansion: NVIDIA fan-out → ROCm; Apple Metal via MLX (B1 selected E1 over native-MSL-first; M4/16 GB dev host available); Vulkan; loyal Intel XPU port; ANE for encoder/pooling classes | ☐ staged by waves in [feature matrix §12](feature-matrix.md#12-platforms--hardware) and [backends.md](backends.md) |
| D2 | Multi-GPU / tensor parallelism plus non-DP MoE sequence parallelism | 🚧 [TP spec written](specs/tensor-parallelism.md); `PAR-SEQUENCE-MOE` inventory added from v0.25.0; runtime work needs a 2-GPU box because GB10 is single-GPU |
| D3 | Spec-decode breadth beyond MTP/DFlash | ☐ ngram and EAGLE3 after the T1 path |
| D4 | **External KV-cache provider API and LMCache interoperability**: mirror vLLM's connector roles/lifecycle/dynamic module seam, then prove the official LMCache MP shared-prefix store/retrieve workflow, Qwen3.6 hybrid-cache behavior, failure recovery, metrics and performance | ☐ T2; explicit `KV-EXTERNAL-CACHE` inventory added; [LMCache quickstart](https://docs.lmcache.ai/getting_started/quickstart.html); spike next |
| D5 | LoRA, local KV/weight offload, expert streaming, and wider model zoo | 🚧 T2 — corrected [expert streaming from disk](specs/expert-streaming.md) contract READY; W0 trace/c1 baseline first; LoRA/KV-offload/zoo still ☐ |

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
