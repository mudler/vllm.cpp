# Roadmap — engineer's work breakdown

Milestones decompose into **workable units** (one unit ≈ one focused PR an
agent can pick up, implement, test, and ledger in a session or few). Order
within a milestone is roughly dependency order. Every unit's Definition of
Done (DoD) includes: mirrored-structure sources with upstream-commit headers,
tests per [discipline.md](discipline.md), inventory markers flipped, ledger
row appended, state log updated.

Status: ☐ open · 🚧 in progress · ✅ done. Keep this current.

## M0 — Model correctness on CUDA (single request, greedy)

- ✅ **M0.1 Build skeleton** (`411c072`): CMake (host + CUDA sm_121),
  third_party vendoring (doctest v2.5.2, nlohmann/json v3.12.0, cpp-httplib
  v0.49.0), CI workflow (CPU build + tests), directory tree per design §4.
  DoD: green CI on empty-ish lib.
- ✅ **M0.2 vt runtime core** (`8df527e`): Tensor/dtype/device, step arena + persistent
  buffers, op dispatch table, CPU scalar reference ops. Interface MUST meet
  the portability requirements in [backends.md](backends.md) (open device
  enum, unified-memory awareness, per-device queue handles, optional
  graph-capture hook). DoD: op unit tests.
- ☐ **M0.3 Parity harness**: `tools/parity/` dump scripts (upstream vLLM on
  dgx.casa produces golden op/layer/logits dumps); C++ test loader +
  threshold framework. DoD: harness runs end-to-end on one op both backends.
- ☐ **M0.4 Loaders**: safetensors mmap loader + HF config.json; GGUF container
  reader (metadata + tensor table + standard quants; NVFP4 extension types).
  DoD: both parse the gate checkpoints; tensor bytes verified vs Python.
- ☐ **M0.5 Tokenizer**: byte-level BPE from tokenizer.json + GGUF vocab;
  incremental detokenizer. DoD: token-for-token vs HF tokenizers on corpus;
  ported detokenizer edge-case tests.
- ☐ **M0.6 CUDA baseline ops**: cuBLASLt GEMM, rmsnorm (+Gemma/gated
  variants), rope (partial), silu_and_mul, embedding/lm_head. DoD: op parity
  CUDA vs dumps.
- ☐ **M0.7 GDN layer (eager)**: conv1d, chunked-scan prefill, recurrence
  decode — correctness-grade CUDA ports of FLA semantics. DoD: layer parity
  vs upstream dumps.
- ☐ **M0.8 MoE layer (eager, bf16)**: router top-k + shared expert + expert
  GEMMs (dequantized NVFP4→bf16 on load for now). DoD: layer parity.
- ☐ **M0.9 Qwen3.6 forward + registry**: `model_executor/models/{registry,
  qwen3_5}.{h,cpp}`, weight mapping (stacked params, GGUF names). DoD:
  logits parity (both checkpoints, safetensors + GGUF) + greedy decode match
  vs upstream on reference prompts. **← M0 exit criterion**

## M1 — The engine (concurrency, correctness under load)

- ☐ **M1.1 core/ Request + SamplingParams + outputs** (T0 field set).
- ☐ **M1.2 BlockPool + prefix caching** (hashing, refcount, eviction) +
  behavioral tests ported from upstream `tests/v1/core/`.
- ☐ **M1.3 Hybrid KVCacheManager/coordinator** (full-attn group + GDN state
  group) + specs + allocate_slots/get_computed_blocks tests.
- ☐ **M1.4 Scheduler** (unified token-budget, chunked prefill, preemption,
  SchedulerOutput diffs) + ported scheduler test suite.
- ☐ **M1.5 InputBatch/BlockTable (MRV2)** persistent batch + step-input build.
- ☐ **M1.6 Paged attention backend** (varlen prefill + paged decode,
  correctness-grade) + CommonAttentionMetadata builders.
- ☐ **M1.7 Sampler** (upstream pipeline order, seeded RNG, logprobs) + GPU
  top-k/top-p; sampler parity tests.
- ☐ **M1.8 EngineCore + step loop + queues**; InputProcessor/OutputProcessor/
  Detokenizer integration; offline C++ `LLM` API. DoD: multi-request
  behavioral suite green; deterministic outputs under concurrency.
  **← M1 exit criterion**

## M2 — Parity performance (gate #1)

- ☐ **M2.1 bench harness** (`examples/bench`, `vllm bench serve` semantics) +
  vLLM baseline installed & measured on dgx.casa (record numbers in ledger).
- ☐ **M2.2 NVFP4 W4A4 GEMM/MoE kernels** (killgate prior art; weights stay
  fp4 in memory). DoD: op parity + measured speedup.
- ☐ **M2.3 GDN kernel tuning** (fused post-conv, fused recurrence).
- ☐ **M2.4 Paged attention tuning** for GQA 16/2 on sm_121.
- ☐ **M2.5 CUDA graphs** decode capture/replay over persistent buffers.
- ☐ **M2.6 Fusions** (qk-norm+rope+gate, rmsnorm+residual) + profile-driven
  leftovers; async scheduling if CPU-bound. DoD: **gate #1 met** — parity
  table vs vLLM committed to ledger. **← M2 exit criterion**

## M3 — Serving MVP (gates #2–#5 validated)

- ☐ **M3.1 OpenAI server**: completions/chat, SSE + non-streaming, models/
  health/version/metrics endpoints, error shapes.
- ☐ **M3.2 Chat templates** (minja-subset) for Qwen/Llama families.
- ☐ **M3.3 Tool calling**: tools/tool_choice, Qwen + Hermes parsers,
  streaming tool-call deltas, grammar-forced JSON for required/named.
- ☐ **M3.4 Grammars**: xgrammar core vendored, structured-output manager +
  scheduler bitmask integration, GBNF input.
- ☐ **M3.5 C API + packaging**: `include/vllm.h`, shared/static lib,
  examples/cli. DoD: LocalAI-style consumption smoke test via dlopen/purego.
- ☐ **M3.6 Server e2e conformance suite** + nightly dgx.casa pipeline.
- ☐ **M3.7 README + docs** (house style, honest benchmarks). **← MVP done**

## Recurring (any time after M0)

- ☐ **P1 Sync tooling** (`tools/sync/`): automate sync-cycle steps 2–4 of
  [upstream-sync.md](upstream-sync.md) — enumerate `PIN..TARGET` per ported
  subtree, map commits to our mirrored files via their headers, draft the
  classification report. DoD: one command emits a ready-to-review
  `.agents/sync/` report.
- 🔁 **Sync cycle** ([upstream-sync.md](upstream-sync.md)) — run on cadence
  (weekly or on demand); each run is a bounded, self-contained task.

## Post-MVP (T1 queue, reorder as needed)

Dense/MoE model families (Llama/Qwen3/Mixtral…) · MTP spec decode · fp8 ·
sliding window · priority scheduling · YaRN · prompt_logprobs/logit_bias/
bad_words · tokenize endpoints · full metrics · Qwen3-Next. Then T2 per
[porting-inventory.md](porting-inventory.md), including backend expansion
per [backends.md](backends.md): Apple Metal (explorations E1 MLX vs E2
native MSL first), Vulkan, Intel XPU (loyal port of upstream
`platforms/xpu`), ANE for encoder/pooling classes (E3).
