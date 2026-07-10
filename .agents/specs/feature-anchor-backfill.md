# Feature anchor backfill inventory spike

**Status:** accepted inventory spike, 2026-07-10
**Parity pin:** vLLM `e24d1b24fe96`
**Owned surface:** the 26 code-bearing legacy rows in
[engine-matrix.md](../engine-matrix.md)

This umbrella spike establishes an honest baseline and a leaf-spike queue. It
does **not** satisfy any missing feature-specific spike. Every row below keeps
its plain `planned: specs/<leaf>.md` entry and must pass that leaf spike before
it may become `READY` or `ACTIVE`.

## Scope

In scope:

- identify every legacy cross-cutting row that claimed complete or partial
  implementation and actually has code in the tree;
- anchor its pinned upstream implementation and tests, local code, and local
  tests at exact files and lines;
- narrow claims to what those anchors prove;
- classify a row as `ANCHOR-BACKFILL` when the bounded behavior appears present
  but lacks its mandatory leaf spike, or `PARTIAL` when a concrete upstream gap
  is already known;
- define independent leaf-spike work that agents can claim without overlapping
  ownership.

Out of scope:

- changing runtime code, tests, build files, or public support claims;
- declaring any row `READY` or `DONE`;
- replacing the per-feature upstream/dependency-chain, dispatch, trace, tests,
  hardware, correctness, performance, and memory analysis required by the
  project spike gate;
- model, quantization, kernel, or backend inventory, which their own matrices
  own.

## Upstream chain

The audit reads the pinned orchestration and executable tests together:

| Block | Upstream implementation roots | Executable-spec roots |
|---|---|---|
| Scheduler and KV | `vllm/v1/core/sched/`, `vllm/v1/core/`, `vllm/config/` | `tests/v1/core/` |
| Sampling | `vllm/v1/sample/`, `vllm/sampling_params.py` | `tests/v1/sample/` |
| Structured output and tools | `vllm/v1/structured_output/`, `vllm/tool_parsers/`, `vllm/entrypoints/openai/chat_completion/` | `tests/entrypoints/llm/`, `tests/entrypoints/openai/` |
| Serving and packaging | `vllm/entrypoints/openai/`, `vllm/renderers/`, `vllm/entrypoints/cli/` | `tests/entrypoints/openai/`, `tests/renderers/` |
| Loading and tokenization | `vllm/model_executor/model_loader/`, `vllm/tokenizers/` | `tests/model_executor/model_loader/`, `tests/tokenizers_/` |

The exact per-row upstream code and test lines are canonical in
[engine-matrix.md](../engine-matrix.md). Runtime-dispatched CUDA behavior, most
notably CUDA graphs, must additionally follow the nsys and dependency-chain
protocol in `parity-lever-protocol.md`; source availability alone is not proof
that the path ran.

GGUF is an explicit deviation: the pinned vLLM load-format registry at
`vllm/model_executor/model_loader/__init__.py:31-65` has no GGUF loader. Its
compatibility oracle is llama.cpp, not an invented vLLM path. The C ABI is also
an original packaging surface because pinned vLLM exposes no C ABI.

## Our baseline

The audit found 26 code-bearing rows: 9 `ANCHOR-BACKFILL` and 17 `PARTIAL`.
None is protocol-complete because none has an accepted leaf spike.

| ID | State | Proven local slice | Known boundary or gap |
|---|---|---|---|
| `ENG-SCHED-CORE` | `ANCHOR-BACKFILL` | Text-only running-first FCFS, token budget, scheduler output | Other request modalities and full upstream scheduling surface need a leaf audit |
| `ENG-CHUNKED-PREFILL` | `ANCHOR-BACKFILL` | Basic budget splitting and Qwen state continuity | Full policy/default/config interactions need a leaf audit |
| `KV-PREFIX-CACHE` | `PARTIAL` | Hash, lookup, manager allocation primitives | Partial-block, size-mismatch, and several eviction paths remain stubs |
| `ENG-PREEMPT-RECOMPUTE` | `ANCHOR-BACKFILL` | FCFS tail-pop recompute slice | Full upstream preemption cases need a leaf audit |
| `ENG-CUDAGRAPH` | `PARTIAL` | Qwen3.6 decode capture/replay and an explicit 35B gate | Model-specific, no generic mode matrix, no direct 27B graph assertion |
| `ENG-SCHED-KNOBS` | `PARTIAL` | Bounded config fields and reserve-full-ISL behavior | Pluggable scheduler and remaining upstream semantics absent |
| `KV-BLOCK-POOL` | `ANCHOR-BACKFILL` | Core allocation, refcount, free-list, eviction lifecycle | Full upstream mode/test inventory needs a leaf audit |
| `KV-MANAGER-ALLOC` | `ANCHOR-BACKFILL` | Named slot allocation, watermark, free paths | Full upstream mode/test inventory needs a leaf audit |
| `KV-HYBRID-COORD` | `PARTIAL` | Full-attention plus GDN group intersection | Align-mode retention is absent |
| `KV-SIZING` | `PARTIAL` | Internal fixed sizing inputs and manager watermark | No complete public memory-utilization/block-override surface |
| `SAMPLE-CORE` | `PARTIAL` | Ordered core operations and greedy path | `n` is parsed but not executed; random/logprob paths synchronize to host |
| `SAMPLE-LOGPROBS` | `PARTIAL` | Sampler gather/rank and request parsing | No engine-to-OpenAI logprobs payload |
| `SAMPLE-LOGIT-FILTERS` | `PARTIAL` | Internal logit-bias/allowlist/bad-word operations | SamplingParams and OpenAI wiring absent |
| `SERVE-COMPLETION-LONGTAIL` | `PARTIAL` | `echo` request acceptance | Echo behavior, best-of, suffix, and user semantics absent |
| `TOOLS-STRUCTURED-CORE` | `PARTIAL` | Native schema/regex/choice/grammar subset | Does not cover full xgrammar-compatible schema semantics |
| `TOOLS-STRUCTURAL-TAG` | `PARTIAL` | Tool-choice-trigger subset | Full structural-tag registry/surface absent |
| `TOOLS-CALLING-CORE` | `PARTIAL` | Tool-choice modes, deltas, Hermes parsing | Local Qwen3 parser is a Hermes alias, not pinned Qwen3 semantics |
| `SERVE-OAI-BASIC` | `ANCHOR-BACKFILL` | Basic chat/completions HTTP and SSE transport | Full upstream conformance needs the leaf audit |
| `SERVE-DISCOVERY-HEALTH` | `PARTIAL` | Models/version routes and unconditional health response | Health always returns 200 instead of reflecting engine readiness |
| `SERVE-CHAT-TEMPLATE` | `ANCHOR-BACKFILL` | Bounded Qwen3.6 template behavior | Not a general upstream renderer implementation |
| `SERVE-C-ABI` | `ANCHOR-BACKFILL` | Eleven-symbol FFI and dlopen/C11 tests | LocalAI-style integration evidence is not yet recorded as a leaf gate |
| `SERVE-CLI-BENCH` | `PARTIAL` | Separate server/bench binaries and one in-process bench | No unified subcommands or upstream benchmark-mode family |
| `LOAD-SAFETENSORS` | `PARTIAL` | Reader plus Qwen-specific weight assembly | No general model loader or WeightsMapper equivalent |
| `LOAD-GGUF` | `PARTIAL` | Reader/dequant plus real 35B Qwen load | 35B-specific execution; dense 27B and format breadth open |
| `LOAD-HF-BPE` | `ANCHOR-BACKFILL` | Bounded Qwen byte-level BPE and incremental detokenization | General HF tokenizer family needs a leaf audit |
| `LOAD-CONFIG-SURFACE` | `PARTIAL` | Scheduler subset, HF fields, limited server flags | Far short of dataclass-for-dataclass and serve-flag parity |

## Port map

Each row below is a separately accepted future spike. The mapping is a starting
boundary, not permission to port before reading the whole dependency chain.

| Row block | Upstream to inspect | Local ownership boundary | Required leaf spike |
|---|---|---|---|
| `ENG-SCHED-CORE`, `ENG-CHUNKED-PREFILL`, `ENG-PREEMPT-RECOMPUTE`, `ENG-SCHED-KNOBS` | `vllm/v1/core/sched/`, `vllm/config/scheduler.py` | `include/vllm/config/scheduler.h`, `src/vllm/config/`, `src/vllm/v1/core/sched/` | `unified-scheduler.md`, `chunked-prefill.md`, `preemption.md`, `scheduler-knobs.md` |
| `ENG-CUDAGRAPH` | compilation config, CUDA-graph helpers, runtime graph modes | `src/vt/cuda/cuda_backend.cu`, model runner, Qwen wrappers | `cuda-graphs.md` |
| `KV-PREFIX-CACHE`, `KV-BLOCK-POOL`, `KV-MANAGER-ALLOC`, `KV-HYBRID-COORD`, `KV-SIZING` | `vllm/v1/core/`, cache config, prefix tests | matching `src/vllm/v1/core/` units plus config/public API | `prefix-caching.md`, `block-pool.md`, `kv-cache-manager.md`, `hybrid-kv-coordinator.md`, `kv-sizing.md` |
| `SAMPLE-CORE`, `SAMPLE-LOGPROBS`, `SAMPLE-LOGIT-FILTERS` | sampler, sampling params, processor tests, serving payload | sampling params, sampler/ops, output processor, protocol/serving | `core-sampler.md`, `logprobs-payload.md`, `logit-bias-bad-words.md` |
| `SERVE-COMPLETION-LONGTAIL` | completion protocol and e2e tests | OpenAI protocol and completion serving | `completions-longtail-fields.md` |
| `TOOLS-STRUCTURED-CORE`, `TOOLS-STRUCTURAL-TAG`, `TOOLS-CALLING-CORE` | structured-output backends, tag registry, tool-parser registry and serving | structured-output and OpenAI tool-parser units | `structured-outputs.md`, `structural-tag.md`, `tool-calling.md` |
| `SERVE-OAI-BASIC`, `SERVE-DISCOVERY-HEALTH`, `SERVE-CHAT-TEMPLATE` | OpenAI routers/serving, instrumentator, renderer | OpenAI entrypoints and chat template | `chat-completions-endpoints.md`, `models-health-version.md`, `chat-templating.md` |
| `SERVE-C-ABI`, `SERVE-CLI-BENCH` | project packaging contract; vLLM CLI mode family | `include/vllm.h`, `src/capi/`, `examples/`, example tests | `c-api-library.md`, `cli-serve-bench.md` |
| `LOAD-SAFETENSORS`, `LOAD-GGUF`, `LOAD-HF-BPE`, `LOAD-CONFIG-SURFACE` | loaders/mappers, llama.cpp GGUF, tokenizer registry, config dataclasses | model loader/models, tokenizer, config, server flags | `safetensors-loader.md`, `gguf-loader.md`, `hf-tokenizer.md`, `config-surface.md` |

## Tests to port

Every leaf spike must enumerate individual upstream cases; the minimum module
inventory is:

| Block | Upstream tests to port | Local tier |
|---|---|---|
| Scheduler and KV | `tests/v1/core/test_scheduler.py`, `test_prefix_caching.py`, `prefix_cache/test_partial_prefix_cache_primitives.py`, `test_single_type_kv_cache_manager.py`, `test_kv_cache_utils.py` | doctest unit plus paged-engine parity |
| Sampling | `tests/v1/sample/test_sampling_params_e2e.py`, `test_sampler.py`, `test_logprobs.py`, `test_logprobs_e2e.py` | doctest unit, OpenAI e2e, oracle parity |
| Structured output and tools | `tests/entrypoints/llm/test_struct_output_generate.py` and matching OpenAI chat/tool suites | doctest backend/parser plus server e2e |
| Serving | completion/chat/model OpenAI endpoint suites and renderer HF suites | real HTTP/SSE conformance plus e2e |
| Loading and tokenizer | fast-safetensors weight utilities, `tests/tokenizers_/test_hf.py`, `test_detokenize.py`; llama.cpp GGUF cases for the deviation | doctest loader/tokenizer plus real-model parity |

An upstream case that cannot pass initially is checked in as `SKIPPED` with its
stable row ID and tracked reason. It is never silently omitted.

## Gates

Each leaf spike selects its applicable gates from this floor:

1. Unit parity: port the pinned upstream cases and compare exact structural
   results, errors, ordering, and edge cases.
2. Correctness: run the same request against the pinned vLLM oracle; generation
   paths require token-for-token parity before performance evidence counts.
3. End to end: exercise the real engine and public API, not only isolated
   helpers. Loader rows use real supported model files in addition to synthetic
   fixtures.
4. Performance and memory: any hot-path or allocation change follows
   `benchmark-protocol.md`, with identical workload, repeat runs, every-axis
   results, and vLLM plus the backend-native reference where applicable.
5. Runtime trace: CUDA graph, scheduler-overlap, KV, and sampling hot paths use
   paired nsys traces under one `/tmp/gpu` lock for the whole A/B series.
6. Architecture/backend: record the exact target rows from the backend matrix;
   GB10 success cannot be generalized to untested CUDA or non-CUDA targets.

## Dependencies

- Pinned vLLM source and tests at `e24d1b24fe96`.
- Installed oracle dependency sources for dynamically dispatched kernels.
- The stable IDs and claim protocol in `coordination.md`.
- The benchmark and GPU-lock protocols for performance-affecting rows.
- Gate model files for engine/load e2e; llama.cpp as GGUF compatibility and
  performance reference.
- Model, quantization, kernel, and backend rows named by each eventual leaf
  spike. A leaf does not duplicate those inventories.

## Work breakdown

1. Claim one row or the smallest tightly coupled row block in coordination.
2. Write only its leaf spike, expanding the upstream chain, test cases,
   dispatch modes, dependencies, hardware, gates, and non-overlapping tasks.
3. Review and merge the spike; change only that row to `READY`.
4. Claim implementation in an isolated worktree, port code and upstream tests
   together, and keep unsupported cases checked in as explicit skips.
5. Run unit, oracle, e2e, trace, performance, memory, and backend gates required
   by the leaf.
6. Merge evidence and update the engine matrix, roadmap, README where visible,
   porting inventory, ledger, and state in the same change.

The preferred first leaves follow the roadmap order: scheduler/latency closure,
MTP prerequisites, long-context/KV, sampling/API debt, then loader breadth.
Independent leaves may run in parallel once their dependencies and file
ownership do not overlap.

## Risks/decisions

- **Decision:** no legacy checkmark survives as `DONE` without a leaf spike and
  exact merged evidence.
- **Decision:** a local stub is not an implementation anchor; absent or stubbed
  open rows keep `Our code` and `Our tests/evidence` as `-` in the matrix.
- **Decision:** bounded local extensions such as the C ABI and GGUF remain
  explicit deviations with the appropriate native oracle; upstream paths are
  never fabricated.
- **Risk:** umbrella row blocks can hide unsupported modes. Leaf spikes must
  split rows when modes have different loader, dispatch, backend, or gate state.
- **Risk:** source inspection can misidentify runtime-selected CUDA behavior.
  Trace the actual execution and inspect dependency/generated kernels.
- **Risk:** line anchors drift when the parity pin advances. The upstream-sync
  cycle must refresh anchors and classification before advancing the pin.
- **Risk:** current tests often prove a narrow Qwen/gate slice. Their presence
  must not be generalized to model-family, backend, or API-wide parity.
