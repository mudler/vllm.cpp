# vllm.cpp — vLLM V1 Engine Core, Ported to Pure C++

**Date:** 2026-07-02
**Status:** Draft — awaiting user review
**Reference:** vLLM checkout at `/home/mudler/_git/vllm`, commit `16282a9` (2026-06-10, ~v0.11+)

## 1. Goal

A faithful C++ port of vLLM's V1 engine core — the architecture that makes vLLM
fast — with **no PyTorch and no Python** at build or run time. "1:1" means the
codebase mirrors vLLM's V1 structure class-for-class (EngineCore, Scheduler,
KVCacheManager, BlockPool, ModelRunner, InputBatch, Sampler, AttentionBackend),
so that anyone who knows vLLM can navigate vllm.cpp, and future vLLM changes
can be tracked by diffing against the reference.

ggml is a **design reference, not a dependency**: we borrow its philosophy
(minimal deps, arena allocation, explicit kernels, quantization-friendly data
layout) but not its static-graph execution model, which is architecturally at
odds with vLLM's persistent-batch, paged-KV design and would cost us the
performance we're porting for.

### Non-goals (for this spec; tracked as backlog)

- All 292 model architectures. We start with one family (Llama-shaped dense
  decoders) and grow.
- Distributed serving (TP/PP/DP, Ray), spec decode, LoRA, structured output,
  multimodal, KV offload/connectors.
- torch.compile-equivalent graph compilation (`vllm/ir/`, `vllm/kernels/`
  layers) — we bind kernels directly, which is what those layers bottom out in
  anyway.

## 2. Decisions log

Decisions confirmed by the user:

- **D1 — No PyTorch.** Hard constraint.
- **D2 — No ggml dependency.** ggml's architecture is too distant from
  vLLM/PyTorch to hit target speed; use it as an example to build on.

Decisions taken by default while the user was away (flagged for review):

- **D3 — Scope: core engine parity**, not literal full-surface port. The
  essence of vLLM (~15k of its 108k engine lines) is the deliverable;
  breadth is backlog.
- **D4 — CPU-first bring-up, CUDA as the performance milestone.** This dev
  machine has no NVIDIA GPU, so the engine is developed and validated against
  a CPU backend locally. The tensor-runtime/backend boundary is designed on
  day one so CUDA kernels (lifted from vLLM history / FlashAttention) drop in
  at milestone M3 without touching engine code.
- **D5 — HF-native model format:** safetensors weights + `config.json` +
  `tokenizer.json`, exactly what vLLM loads. GGUF support is backlog (it
  matters for the LocalAI ecosystem, but 1:1 parity testing against vLLM/HF
  is simpler with the native format).
- **D6 — First model family: Llama-shaped dense decoders** (Llama 3, Qwen2.5,
  Mistral) — one forward function covers all three with small deltas.
- **D7 — Pin the reference** to commit `16282a9`. vLLM moves fast; we port a
  fixed snapshot and rebase deliberately.

## 3. Approaches considered

**A. Build on ggml directly** — rejected (user D2, and technical analysis
agrees). vLLM's speed comes from a persistent input batch mutated in place,
paged KV with per-step `slot_mapping`/`block_table` tensors, and ragged
variable-length batched attention. ggml's build-a-graph-then-schedule model
fights all of that; every step would rebuild graphs and replan memory.

**B. Own minimal tensor runtime + kernels lifted/adapted from vLLM's own
`csrc/` (chosen).** vLLM's CPU backend (`csrc/cpu/`) contains complete SIMD
attention/norm/activation kernels (x86 AVX/AMX, NEON, RVV, VSX) as template
code over raw pointers; torch appears only in orchestration wrappers, so they
can be adapted with modest surgery. For CUDA later, the classic
`paged_attention_v1/v2` kernels exist in vLLM's git history as self-contained
CUDA, and dense GEMM comes from cuBLAS. We get vLLM-grade kernels because they
*are* vLLM's kernels, wrapped in a runtime shaped like vLLM needs (eager,
persistent buffers) rather than like ggml.

**C. Assemble from heavyweight C++ deps** (oneDNN, CUTLASS, FlashAttention as
libraries) — more day-one FLOPs, but a large, slow build with deep
dependencies, against the spirit of a `.cpp` project. Individual pieces (e.g.
FlashAttention kernels) can still be vendored selectively inside approach B
when benchmarks justify it.

## 4. Architecture

```
vllm.cpp/
├── CMakeLists.txt
├── src/
│   ├── vt/                  # tensor runtime ("vllm tensor") — the ggml-inspired part
│   │   ├── tensor.h         # Tensor: dtype, shape, strides, device, data ptr
│   │   ├── alloc.h/.cpp     # arena + pooled allocators; persistent buffers
│   │   ├── ops.h            # op entry points, dispatched by device
│   │   └── cpu/             # CPU kernels (adapted from vllm csrc/cpu + own)
│   │       └── (cuda/ at M3: lifted paged-attn kernels + cuBLAS GEMM)
│   ├── core/                # Request, SamplingParams, RequestStatus, outputs
│   │                        #   ← vllm/v1/request.py, sampling_params.py, outputs.py
│   ├── sched/               # Scheduler, SchedulerOutput, RequestQueue
│   │                        #   ← vllm/v1/core/sched/
│   ├── kv/                  # KVCacheManager, BlockPool, block hashing, specs
│   │                        #   ← vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_utils}.py
│   │                        #     + vllm/v1/kv_cache_interface.py
│   ├── worker/              # ModelRunner, InputBatch (persistent), BlockTable
│   │                        #   ← vllm/v1/worker/gpu/{model_runner,input_batch,block_table}.py
│   ├── attention/           # AttentionBackend iface + CPU paged-attention impl
│   │                        #   ← vllm/v1/attention/backend.py + backends/cpu_attn.py
│   ├── sample/              # Sampler: temperature, penalties, top-k/top-p, logprobs
│   │                        #   ← vllm/v1/sample/
│   ├── engine/              # EngineCore step loop, InputProcessor, OutputProcessor,
│   │                        #   Detokenizer, LLM (sync) + AsyncLLM-style API
│   │                        #   ← vllm/v1/engine/
│   ├── models/              # explicit forward fns per architecture + registry
│   │   └── llama.cpp/.h     #   ← vllm/model_executor/models/llama.py (semantics only)
│   ├── loader/              # safetensors mmap loader, HF config.json parsing
│   ├── tokenizer/           # byte-level BPE (tokenizer.json), chat templates
│   └── server/              # OpenAI-compatible HTTP server (SSE streaming)
│                            #   ← vllm/entrypoints/openai/ (semantics only)
├── tools/parity/            # Python-side dump scripts run against ref vLLM/HF
├── tests/                   # unit + parity + engine behavioral tests
└── docs/
```

### 4.1 vt — the tensor runtime

Eager execution, no graphs, no autograd. A `vt::Tensor` is a POD-ish view:
dtype (fp32/fp16/bf16/int8/int32...), shape, strides, device, pointer. Memory
comes from two allocators: a **step arena** (bump allocator reset every engine
step — activations) and **persistent buffers** (weights, KV cache pages, the
persistent InputBatch tensors — allocated once, mutated in place, exactly
mirroring vLLM's design and keeping the door open for CUDA-graph replay at M3).

Ops are free functions with explicit outputs (`vt::matmul(out, a, b)`,
`vt::rms_norm(out, x, w, eps)`, `vt::rope(q, k, positions, ...)`,
`vt::silu_and_mul(out, x)`, `vt::reshape_and_cache(k, v, kcache, vcache,
slot_mapping)`, `vt::paged_attention(out, q, kcache, vcache, block_table,
seq_lens, ...)`, `vt::softmax_sample(...)`). Each op dispatches on device to a
backend kernel table. The op *set* is exactly what the ported models and
engine need — we do not build a general tensor library.

CPU kernels: adapt vLLM's `csrc/cpu/` SIMD template kernels (attention,
layernorm, activation, the `cpu_types_*.hpp` vector abstraction) by replacing
their torch orchestration with raw-pointer entry points; GEMM starts with a
tiled+vectorized fp32/bf16 implementation and can vendor a single-file
microkernel library later if benchmarks demand.

### 4.2 Engine — the 1:1 part

Class-for-class mapping of the subsystems the Explore map identified:

- **Scheduler** (`sched/`): vLLM V1's unified algorithm — *no prefill/decode
  distinction*; each request tracks `num_computed_tokens` vs `num_tokens`, and
  the scheduler assigns tokens per step under a `token_budget`
  (`max_num_scheduled_tokens`). Chunked prefill falls out naturally; running
  requests first, then waiting; preemption pops the FCFS tail and requeues.
  `SchedulerOutput` carries `scheduled_new_reqs` (full data) +
  `scheduled_cached_reqs` (diffs only), `num_scheduled_tokens`, and finished
  ids — same wire shape as vLLM so behavior is directly comparable.
- **KV cache** (`kv/`): `BlockPool` (free list + ref counts +
  `BlockHash → block` map), `KVCacheManager.allocate_slots()` /
  `get_computed_blocks()` for hash-based prefix reuse, eviction of cached-but-
  unreferenced blocks, `FullAttentionSpec` first (SlidingWindow etc. are
  backlog but the `KVCacheSpec` polymorphism is in from day one). Block
  hashing follows vLLM: parent-hash-chained hashes over block-sized token
  spans.
- **Worker** (`worker/`): persistent `InputBatch` — long-lived per-slot token
  ids, positions, block tables, sampling params, updated incrementally from
  `SchedulerOutput` (add new, apply diffs, swap-remove finished). Builds the
  flattened step inputs: `query_start_loc`, `seq_lens`, `slot_mapping`,
  `positions` — the same `CommonAttentionMetadata` contract vLLM feeds its
  attention backends.
- **Sampler** (`sample/`): vLLM's pipeline order preserved — logits fp32 →
  (logits processors slot, empty at first) → greedy short-circuit → temperature
  → penalties (frequency/presence/repetition) → top-k → top-p → sample with
  per-request RNG (seeded, reproducible) → optional logprobs.
- **EngineCore** (`engine/`): the canonical `step()`:
  `scheduler.schedule() → model_runner.execute_model() → sample →
  scheduler.update_from_output()`. Runs in its own thread with an MPSC input
  queue and output queue (in-process analog of vLLM's ZMQ split — same
  boundary, so a multi-process mode remains possible). `OutputProcessor` +
  incremental `Detokenizer` on the output side. Public API: synchronous `LLM`
  (offline batch) and callback/stream-based `AsyncLLM` for the server.
- **Server** (`server/`): OpenAI-compatible `/v1/completions`,
  `/v1/chat/completions` (SSE streaming), `/v1/models`, `/health`,
  `/metrics` (Prometheus text). Built on a header-only HTTP lib
  (cpp-httplib), chat templating starts with built-in per-family templates
  (full Jinja2 is explicitly out of scope; revisit if needed).

### 4.3 Models, loader, tokenizer

A model is a weights struct + an explicit forward function over `vt` ops
(no module system): `llama_forward(weights, batch_inputs, kv_cache, arena) →
logits`. A registry maps HF `config.json` `architectures` →
loader+forward, mirroring vLLM's model registry. Weights load from
safetensors via mmap (format is trivial: 8-byte header len + JSON + raw
tensors), with on-load dtype handling (bf16 native; fp16→bf16/fp32 convert).

Tokenizer: byte-level BPE reading HF `tokenizer.json` — covers Llama 3,
Qwen2.5, Mistral. Incremental detokenization (vLLM's `detokenizer.py`
semantics: withhold bytes until UTF-8-complete). SentencePiece-based models
are backlog.

## 5. Data flow (one engine step)

```
HTTP req ─→ InputProcessor (tokenize, validate, SamplingParams)
        ─→ input queue ─→ EngineCore thread:
   scheduler.schedule()
     ├─ running reqs: assign tokens under budget; allocate_slots(); preempt if OOM
     └─ waiting reqs: prefix-cache hit via get_computed_blocks(); admit under budget
   model_runner.execute_model(SchedulerOutput)
     ├─ InputBatch.update(): add new / apply diffs / remove finished
     ├─ build step inputs: token_ids, positions, query_start_loc, seq_lens,
     │                     slot_mapping, block_table
     ├─ forward: embed → N × (rmsnorm → qkv → rope → reshape_and_cache →
     │           paged_attention → o-proj → rmsnorm → mlp) → final norm → lm_head
     │           (logits only at each sequence's last scheduled token)
     └─ Sampler → sampled token ids (+ logprobs)
   scheduler.update_from_output(): append tokens, detect stop, free KV
        ─→ output queue ─→ OutputProcessor: detokenize incrementally, stream SSE
```

## 6. Error handling

- Request-level errors (bad params, prompt too long, tokenizer failure) are
  rejected at `InputProcessor` and never reach the engine — HTTP 4xx with
  OpenAI-style error JSON.
- Engine invariants (block accounting, batch consistency) are `assert`-checked
  in debug and validated by tests; a corrupted engine state is fatal by design
  (matching vLLM — the engine dies loudly rather than serving garbage).
- KV exhaustion is not an error: it's preemption + recompute, per the
  scheduler algorithm.
- The server returns 503 on engine death; clean shutdown drains the queues.

## 7. Testing strategy

Parity-first, per the house style for C++ ports:

1. **Op-level golden tests**: `tools/parity/` Python scripts dump
   inputs/outputs for each op from the reference vLLM checkout (which may use
   torch — it's a *test-time* dependency only, never linked); C++ unit tests
   replay them with max-abs / cosine thresholds.
2. **Layer/model parity**: per-layer activation dumps for one Llama-3-class
   model; end-to-end logits parity; greedy decode must match HF/vLLM
   token-for-token over reference prompts.
3. **Engine behavioral tests** (no model needed — a tiny fake model runner):
   scheduler unit tests ported from `tests/v1/core/` (chunked prefill splits,
   preemption order, budget exhaustion), BlockPool/prefix-cache tests (hash
   chains, ref counting, eviction), detokenizer UTF-8 edge cases.
4. **End-to-end**: OpenAI-endpoint tests against a small real model
   (Qwen2.5-0.5B / Llama-3.2-1B class) — streaming, stop strings, n>1,
   seeds reproducible.
5. **Benchmarks** (from M2 on): tokens/s and TTFT vs llama.cpp on CPU, vs
   vLLM itself once CUDA lands; honest methodology, published in README.

CI: build + unit + behavioral tests on CPU (GitHub Actions); parity suites run
locally/nightly where reference dumps live.

## 8. Milestones

- **M0 — Single-sequence correctness (CPU).** vt runtime + CPU kernels +
  safetensors loader + tokenizer + Llama forward + greedy decode. Parity vs
  HF. *(Deliberately llama.cpp-shaped, but structured as vLLM from day one.)*
- **M1 — The vLLM part.** BlockPool + KVCacheManager + prefix caching +
  Scheduler (continuous batching, chunked prefill, preemption) + persistent
  InputBatch + Sampler + EngineCore loop + offline `LLM` API. Behavioral test
  suite green; multi-request throughput scales on CPU.
- **M2 — Serving.** OpenAI server (completions/chat/models, SSE), metrics,
  graceful shutdown. Benchmarks vs llama.cpp published.
- **M3 — CUDA.** `vt/cuda/` backend: cuBLAS GEMM, paged-attention +
  reshape_and_cache kernels lifted from vLLM history (FlashAttention vendoring
  as a fallback/upgrade), CUDA graphs for decode. Benchmark vs vLLM on the
  same GPU. *(Requires a GPU box; not this machine.)*
- **Backlog:** more architectures (MoE, MLA, sliding window), GGUF loading,
  quantization, structured output, spec decode, LoRA, LocalAI backend wiring,
  TP.

## 9. Risks

- **Tokenizer scope creep** — HF tokenizer.json has many pretokenizer/
  normalizer variants. Mitigation: implement exactly what the first model
  family needs; hard-fail on unsupported configs with a clear error.
- **CPU GEMM performance** — matching llama.cpp's mature quantized kernels on
  CPU is a losing race short-term. Mitigation: M0–M2 optimize for
  *correctness and engine behavior*; the performance story is CUDA (M3).
  CPU perf work happens only after profiling, per the optimization loop.
- **No local GPU** — CUDA milestone needs remote hardware; engine/backend
  boundary is designed so M3 is additive, not a refactor.
- **vLLM drift** — pinned to `16282a9` (D7); rebases are deliberate events.
- **Kernel lift friction** — csrc/cpu kernels are torch-entangled at the
  wrapper level; if surgery proves costly for some op, write it fresh against
  the parity dumps instead. The dumps, not the code, are the contract.
