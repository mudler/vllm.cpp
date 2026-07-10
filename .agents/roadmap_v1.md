# Roadmap v1 — post-MVP
*(user-directed 2026-07-10: this document is the live roadmap; the completed
M0–M3 record is archived at
[completed/roadmap_mvp_v0.md](completed/roadmap_mvp_v0.md).)*

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
| A2 | GGUF real-file greedy parity on GPU (MVP loader gate) | ✅ **PASSED** — real APEX 35B GGUFs (Compact+Balanced, all supported k-quants), 28/28 assertions, 16/16 greedy token-exact vs same-file llama.cpp oracle, checkpoint-gated test+goldens merged (e2b93cf); remaining breadth: no 27B GGUF exists, NVFP4-type-40 dequant + i-quants deferred |
| A3 | `test_ops_fused_chain` FMA-contraction fix | ✅ merged bf48edb (`-ffp-contract=off` host-wide) |
| A4 | De-Python the build: vendor Triton AOT artifacts per-arch (`triton_aot_vendored/<arch>/` + MANIFEST; `VLLM_CPP_TRITON_REGEN` = maintainer-only Python) | ✅ **DONE** (54367cc..a432461) — vendored sm_121a artifacts (40 files, byte-deterministic regen), fresh-clone build green with `VLLM_CPP_TRITON_PYTHON=/nonexistent/python`, 27B gate green on vendored cubins |
| A5 | e2e suites per gates.md (server conformance nightly on dgx etc.) | ☐ next |

## B. Research tracks (kicked off in parallel, reports due)

| # | Track | Question | State |
|---|---|---|---|
| B1 | Apple Metal / **MLX** | vllm-metal vs wiring MLX's C++ API behind `vt::` vs native MSL | ✅ **MLX wins** (Ollama's production swap ~1.6-2×); vendor MLX under vt:: + port vllm-metal's paged MSL as MLX primitives. **Hardware now available:** `ssh 192.168.68.103`, M4 Mac mini/16 GB; enough for op parity + small-model bring-up, not the 27B/35B gates. → [specs/expansion-map-2026-07-10.md](specs/expansion-map-2026-07-10.md) |
| B2 | Arches & vendors | per-arch matrix; Triton→AMD; SYCL vs Vulkan | ✅ **NVIDIA fan-out nearly FREE** (bf16 path already sm_80+; A100/H100/4090 need ~no new kernels — Wave 1); **Triton GDN → AMD gfx942 via one flag**; Vulkan-before-SYCL call. → map |
| B3 | SGLang steals | measured wins portable to C++ | ✅ **Async/overlap scheduling is vLLM's DEFAULT at our pin — an unmet MIRROR obligation** (we run synchronous); reframed as latency/mirror, not throughput (we already ≥1.0× vs async-vLLM). RadixAttention REJECTED per mirror policy. Kernel steals: no-op. → map |
| B4 | llama.cpp CPU | vendor IF proven faster (user criterion) | ✅ research: **our CPU is a single-threaded scalar loop (zero threads!)** — threadpool = 1-wk prerequisite; **compute-in-quant** = the structural fix (~3.3× decode, ~10× prefill on GGUF). 🔄 local Qwen3.5-2B bench = the decision measurement | 
| B5 | Speculative decoding | MTP + dflash | ✅ **Both gate checkpoints SHIP MTP heads** (bf16, we currently skip `mtp.*` at load!); **DFlash** (block-diffusion drafter, ICML'26) is in-pin WITH published drafts for our exact models + a DGX-Spark community container. Route: MTP k=1 27B → GDN spec path → DFlash. → [specs/spec-decode-scoping-2026-07-10.md](specs/spec-decode-scoping-2026-07-10.md) — **specs written → [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md), [specs/dflash-spec-decode.md](specs/dflash-spec-decode.md)** (NB: B5's "27B pure attention" was wrong — both gates are GDN hybrids; the GDN spec path is on milestone 1, spec §0) |
| B6 | Feature parity matrix | the big one-by-one vLLM-feature table (TP/PP/EP, LoRA, multimodal, disagg, pooling, …) — what we have / miss / tier | ✅ **[feature-matrix.md](feature-matrix.md)** — 13 sections, ~80 rows, every row source-verified + carrying its `Spec` delegation link; headline gaps surfaced: `/metrics` (open T0-core debt), async/overlap scheduling (unmet MIRROR obligation, B3), logprobs payload + logit_bias/bad_words API wiring (sampler done, serving not) |
| B7 | Multimodality + tool calls | image/audio/video route; parallel tool calls | ✅ **the gate checkpoints ARE the VLMs** (full ViT shipped in both — 333 `model.visual.*` tensors we currently DROP at load; no deepstack; interleaved M-RoPE) → vision = a MIRROR obligation. **Our Qwen3 tool parser targets the WRONG format** (Hermes JSON; the shipped template is Qwen3-Coder XML — vLLM serves 27B with `qwen3_coder`) + no reasoning parser + our Jinja engine can't render the shipped template (~10 missing constructs) + tool-result round-trip broken; parallel tool calls otherwise solid (parser/index/grammar/streaming, tested). 6 specs ranked. → [specs/mm-tools-scoping-2026-07-10.md](specs/mm-tools-scoping-2026-07-10.md) |

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
