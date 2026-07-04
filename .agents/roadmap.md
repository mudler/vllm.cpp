# Roadmap — engineer's work breakdown

Milestones decompose into **workable units** (one unit ≈ one focused PR an
agent can pick up, implement, test, and ledger in a session or few). Order
within a milestone is roughly dependency order. Every unit's Definition of
Done (DoD) includes: mirrored-structure sources with upstream-commit headers,
tests per [discipline.md](discipline.md), inventory markers flipped, ledger
row appended, state log updated.

Status: ☐ open · 🚧 in progress · ✅ done. Keep this current.

## M0 — Model correctness on CUDA (single request, greedy)

**Status: exit criterion MET (safetensors) ✅ — Qwen3.6-35B greedy-decodes
16/16 tokens token-for-token vs the pinned oracle on GB10 (`25326fc`). GGUF
model load (M0.10, k-quant dequant) remains open (T0).**

- ✅ **M0.1 Build skeleton** (`411c072`): CMake (host + CUDA sm_121),
  third_party vendoring (doctest v2.5.2, nlohmann/json v3.12.0, cpp-httplib
  v0.49.0), CI workflow (CPU build + tests), directory tree per design §4.
  DoD: green CI on empty-ish lib.
- ✅ **M0.2 vt runtime core** (`8df527e`): Tensor/dtype/device, step arena + persistent
  buffers, op dispatch table, CPU scalar reference ops. Interface MUST meet
  the portability requirements in [backends.md](backends.md) (open device
  enum, unified-memory awareness, per-device queue handles, optional
  graph-capture hook). DoD: op unit tests.
- ✅ **M0.3 Parity harness** (`f063890`): `tools/parity/` dump scripts (upstream vLLM on
  dgx.casa produces golden op/layer/logits dumps); C++ test loader +
  threshold framework. DoD: harness runs end-to-end on one op both backends.
- ✅ **M0.4 Loaders** (`4aeae62`): safetensors mmap loader + HF config.json; GGUF container
  reader (metadata + tensor table + standard quants; NVFP4 extension types).
  DoD: both parse the gate checkpoints; tensor bytes verified vs Python.
- ✅ **M0.5 Tokenizer** (`0baa46e`): byte-level BPE from tokenizer.json + GGUF vocab;
  incremental detokenizer. DoD: token-for-token vs HF tokenizers on corpus;
  ported detokenizer edge-case tests.
- ✅ **M0.6 CUDA baseline ops** (`3750954`): cuBLASLt GEMM, rmsnorm (+Gemma/gated
  variants), rope (partial), silu_and_mul, embedding/lm_head. DoD: op parity
  CUDA vs dumps.
- ✅ **M0.7 GDN layer (eager)** (`ead59d6`): conv1d, delta-rule prefill scan
  (chunked perf kernel M2.3), recurrence decode — correctness-grade CUDA ports of FLA semantics
  (prefill is a sequential recurrence validated against the pinned chunked
  reference; chunked perf kernel lands M2.3). DoD: op parity vs pinned
  upstream dumps, CPU + CUDA.
- ✅ **M0.8 MoE layer (eager, bf16)** (`65788b3`): router top-k + shared expert + expert
  GEMMs (dequantized NVFP4→bf16 on load for now). DoD: layer parity.
- ✅ **M0.9 Qwen3.6 forward + registry** (`25326fc`, with `861b1e4` full-model
  gate + `45d3c18` tolerance): `model_executor/models/{registry,qwen3_5}.{h,cpp}`,
  weight mapping (stacked params), dense causal attention op, full forward
  (embed→40 layers GDN/full-attn+MoE→norm→lm_head), NVFP4/FP8→bf16 weight load.
  DoD MET (safetensors path): 16/16 token-for-token greedy decode vs the pinned
  oracle on the real Qwen3.6-35B on GB10 — **M0 exit criterion MET**. Accepted
  greedy-matched deviations: f32 residual stream (vs upstream bf16) + f32
  single-round MoeCombine (vs upstream bf16 double-round); max top-1000 logit
  gap 0.994 (compounding, doesn't flip greedy). 27B dense/W4A4 correctness
  DEFERRED (~M2.2, unregistered). **← M0 exit criterion (safetensors) ✅**
  - ☐ **M0.9-GGUF (open, re-scoped as M0.10 below)**: the safetensors DoD is
    met; the GGUF model-load half of the original DoD is broken out to M0.10.
- ☑ **M0.10 GGUF model load** (T0/MVP, re-scoped out of M0.9): the APEX GGUFs
  (arch `qwen35moe`) are k-quant (Q4_K/Q5_K/Q6_K/Q8_0, IQ2_S id22, IQ4_XS id23),
  NOT NVFP4. Done `6ef3f12` (k-quant dequant: F32/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/Q3_K
  → f32/bf16, byte-exact vs ggml-quants.c — reviewed PASS, line-by-line verified)
  → `1a4db5c` (GGUF → Qwen3_5MoeWeights loader: qwen tensor-name mapping, the
  convert-time transform INVERSIONS [norm w+1, ssm_a=-exp(A_log), V-head
  grouped→tiled reorder], MoE expert 3-d split, HfConfigFromGguf, model_loader
  `.gguf` routing — reviewed PASS, transforms verified vs the safetensors loader +
  forward + convert script). CPU 76/76 (synthetic-GGUF tests). **← GGUF gate (#2)
  at model level — CPU path done.** DEFERRED: IQ2_S/IQ4_XS i-quants (Mini/Quality
  variants; Compact/Balanced are pure K-quant+Q8_0+F32 → work now); the real APEX
  GGUF end-to-end greedy parity vs the safetensors 35B is dgx-pending (quant
  fidelity + real 16k/32v head dims).

## M1 — The engine (concurrency, correctness under load)

**Status: begun 🚧 — M1.1 (engine core types) done; the shared engine
vocabulary is ported.**

- ✅ **M1.1 core/ Request + SamplingParams + outputs** (T0 field set)
  (`b888645`/`fabf48f` SamplingParams, `4320dae`/`a43eaf8` Request+RequestStatus,
  `cd13ec3` EngineCore I/O types, `4d477eb` RequestOutput; close-out adds the
  `RequestOutput.prompt_logprobs` opaque placeholder):
  SamplingParams (Verify/PostInit == `__post_init__`, eos on params),
  Request + RequestStatus (12-status ordering, IsFinished, FinishReason map),
  EngineCore I/O (EngineCoreRequest/Output(s), ModelRunnerOutput, SamplerOutput),
  RequestOutput/CompletionOutput (FinishReason→string). All reviewed PASS, CI
  green throughout; behavioral CPU unit tests, no goldens (structural port).
- ✅ **M1.2 BlockPool + prefix caching** (`5ee2301`; ports `95be067`
  KVCacheBlock + intrusive LRU FreeKVCacheBlockQueue, `a0a8622` sha256_cbor
  block hashing, `5ee2301` BlockPool re-ported to pinned e24d1b24 API +
  Request block-hashes): hashing, refcount, LRU eviction; behavioral tests
  ported from upstream `tests/v1/core/`. All reviewed PASS, CI green,
  ASan-clean. Align/partial-block/events deferred behind 1:1 stubs.
- ✅ **M1.3 Hybrid KVCacheManager/coordinator** (`75caf38`; ports `bae8f7a`
  KVCacheSpec/Config, `ec6f4be`+`9f30013` SingleTypeKVCacheManager
  (FullAttention+Mamba/GDN), `5fdbb7b` KVCacheCoordinator+Hybrid, `c708753`+
  `75caf38` KVCacheManager): full-attn block group + GDN/mamba recurrent-state
  group; page_size_bytes byte-exact; cross-group MIN-intersection prefix hit;
  allocate_slots (accounting/watermark/OOM→nullopt/admission-cap) + literal
  test_evict order. All reviewed PASS, CI green, ASan-clean; behavioral CPU
  tests ported from upstream `tests/v1/core/`. Sliding-window/MLA/chunked-local
  specs + mamba align mode + enable_caching=false deferred behind 1:1 stubs.
- ✅ **M1.4 Scheduler** (`4f12158`; ports `2f0ea69` SchedulerConfig + FCFS
  RequestQueue, `c65e650` SchedulerOutput/NewRequestData/CachedRequestData,
  `a591a0d`→`f09509c` schedule() re-ported to MRV2, `4f12158`
  update_from_output + check_stop): unified token-budget (running-first,
  chunked prefill, FCFS preemption via allocate_slots→nullopt), SchedulerOutput
  new/cached diffs in the MRV2 shape (prefill_token_ids + resumed-as-new fold),
  update_from_output + check_stop (min_tokens→eos→stop_token_ids→length
  precedence, free-on-finish). All reviewed PASS, CI green; behavioral CPU tests
  ported from `tests/v1/core/test_scheduler.py`. Priority/spec-decode/
  structured/encoder/KV-connector/async deferred behind 1:1 stubs.
- ✅ **M1.5 InputBatch/BlockTable (MRV2)** persistent batch + step-input build
  (`62fdfca`; ports `f3bf0ac` BlockTable+MultiGroupBlockTable, `2d9f693`
  persistent InputBatch add/remove/condense, `62fdfca` update_states+
  prepare_inputs): V1 host-array block-id storage (slot_mapping=block_id*bs+
  offset, multi-group fanout), persistent InputBatch (condense() field-by-field
  densification, MRV2 contract token_ids=prefill_token_ids, per-slot sampling
  arrays for M1.7), step-input build (query_start_loc/seq_lens/positions/
  slot_mapping/logits_indices matched 1:1 vs `_prepare_inputs`). ARCH: MRV2 =
  scheduler-output CONTRACT (T0, V1 host-array algorithm) vs staged worker
  STORAGE (M2) — see `.agents/vllm-v1-v2.md` (`2889abd`). All reviewed PASS, CI
  green, ASan-clean; behavioral CPU tests ported from `tests/v1/worker/`. Staged
  device tensors / CUDA-graph padding / LoRA/spec/mm deferred to M2/T1.
- ☑ **M1.6 Paged attention backend** (varlen prefill + paged decode,
  correctness-grade) + CommonAttentionMetadata builders. Done `bd47ce3`
  (CommonAttentionMetadata + AttentionBackend/Impl/MetadataBuilder ABCs, flash
  NHD get_kv_cache_shape) → `e231196`+`7de4f0c` (reshape_and_cache, stride-based
  NHD write CPU+CUDA) → `c244592` (paged_attention CPU+CUDA, anchored to M0.9
  dense on the single-seq case) → `370ddaf` (GDNAttentionMetadata segmentation).
  CPU CI green (47/47); CUDA parity tests build-guarded, dgx-pending. GDN-state
  zeroing = caller obligation carried to the batched-GDN forward (see state.md).
- ☑ **M1.7 Sampler** (upstream pipeline order, seeded RNG, logprobs) + GPU
  top-k/top-p; sampler parity tests. Done `ff366c6` (SamplingMetadata +
  LogprobsTensors + make_sampling_metadata) → `f940fa3`+hardening (core ops:
  temperature/greedy-argmax/top-k/top-p/random gumbel-max) → `aac5138`
  (penalties + bad-words/allowed-ids masks + min-tokens/logit-bias/min-p
  builtins) → `38a8846`+hardening (Sampler.forward ordered pipeline + logprobs
  gather + ranks). Greedy bit-exact (the M0-exit primitive); random RNG
  distribution-correct, torch-Philox bit-parity = T1 carry. CPU CI 52/52;
  CUDA sampling kernels dgx-pending. Reviewed PASS (4 tasks).
- ☑ **M1.8 EngineCore + step loop + queues**; InputProcessor/OutputProcessor/
  Detokenizer integration; offline C++ `LLM` API. DoD: multi-request
  behavioral suite green; deterministic outputs under concurrency.
  **← M1 exit criterion ✅ MET.** Done `88821f3` (Executor pass-through +
  EngineCore step loop) → `73a9509` (InputProcessor, closes M1.1 PostInit carry)
  → `f1ae018`+multiblock (forward dense→paged refactor: ReshapeAndCache +
  PagedAttention + batched GDN persistent-state + GDN-zeroing; paged==dense
  bit-exact) → `9949f87`+3req (batched runner: KV alloc, decode-first reorder,
  execute/sample/write-back, four-way ordering identity; closes M1.7 seed carry)
  → `c7ba3a5` (OutputProcessor: detokenize + string-stop + reqs_to_abort) →
  `c1859d9`+abort-assert (LLMEngine wiring + e2e). **The full V1 engine loop
  runs end-to-end on CPU**: greedy determinism, 2-request concurrent, streaming
  ==non-streaming, max_tokens + stop-string, all green (58/58). Reviewed PASS
  (6 tasks). GDN-state zeroing WIRED (M1.6 carry closed). **35B greedy through
  the paged loop on dgx = pending (CPU-only box).** ← **M1 CLOSED.**

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

- ☑ **M3.1 OpenAI server**: completions/chat, SSE + non-streaming, models/
  health/version/metrics endpoints, error shapes. Done `9b5c2c5` (protocol types
  + to_sampling_params) → `9afc099` (serving_completion/chat, decoupled from HTTP)
  → `23d9f2c` (cpp-httplib server + routes + SSE + SanitizeUtf8 + examples/server)
  → `+`mutex (engine serialization). Reviewed PASS. CPU 63/63 incl. a real-socket
  smoke + 6-client concurrency test. (metrics endpoint deferred; logprobs payload
  M3.x.)
- ☑ **M3.2 Chat templates** (minja-subset) for Qwen/Llama families. Done `a99a65e`
  (an original minja-subset Jinja engine: for/if/elif/else/set/interpolation +
  whitespace-control incl. trim_blocks/lstrip_blocks, verified byte-identical to
  jinja2 on the Qwen3 template; loud-error on unsupported constructs). Injected
  into serving_chat via the ChatPromptFn seam. Tool/think template branches
  (namespace/tojson/reversed) = M3.3/M3.4.
- ☑ **M3.3 Tool calling**: tools/tool_choice, Qwen + Hermes parsers,
  streaming tool-call deltas, grammar-forced JSON for required/named. Done
  `b315ef8` (protocol tools/tool_choice/ToolCall/DeltaToolCall) → `a14ce92`
  (Hermes/Qwen3 `<tool_call>` parsers; gate model = Hermes format) → `fe5034d`
  (streaming parser + serving_chat wiring + chat-template tools via minja tojson)
  → `bdb4838`/`caa6fa4` (forced-JSON for required/named). Reviewed — the review
  caught a real drop (forced JSON was bare, parser needs the wrapper; fixed).
  **M3.3b — RELAXED tool_choice=auto (lazy grammar triggers)** `6ba00d0`+`bcfe9f0`
  +`e6e497b`+`18e3efb`: per user feedback (can't force `<tool_call>` on auto — the
  model may just reply). Analyzed BOTH llama.cpp (lazy grammar/awaiting_trigger) +
  vLLM (xgrammar StructuralTag) → ported vLLM's STRUCTURAL_TAG seam 1:1 +
  implemented the lazy matcher natively (awaiting no-op mask + trigger detect +
  replay). auto → lazy (free text until `<tool_call>`, then constrain); required/
  named → forced; none → nothing. Reviewed PASS. Fixed a gate-model bug: `<tool_call>`
  /`</tool_call>` are ADDED tokens in Qwen3.6 → made added tokens grammar-matchable
  by literal content. CPU 73/73 (clean rebuild). Deferred: Coder-XML/Mistral/
  pythonic parsers, parallel-tool streaming edge cases.
- ☑ **M3.4 Grammars**: structured-output manager + scheduler bitmask integration
  + GBNF/JSON-schema input. **DECISION (Path C, see plan + vllm-v1-v2 pattern):**
  ported the INTEGRATION layer 1:1 (StructuredOutputManager, get_grammar_bitmask,
  GrammarOutput, apply_grammar_bitmask, the StructuredOutputBackend/Grammar ABCs —
  the parity surface) with a from-scratch NATIVE backend (§9) behind that seam;
  **xgrammar vendoring = a later parity-completion milestone** (a 2nd backend
  behind the same proven seam, like upstream's 4). Done `8343c4c` (ABCs+key) →
  `1351f88` (manager+scheduler+EngineCore plumbing) → `9b640ee` (apply_grammar_
  bitmask in runner, set=allowed→-inf) → `74eec60`+fix (native GBNF/regex/choice
  engine: parser + push-down FSM + byte-trie sub-O(vocab) fill; fill==accept
  invariant guarded by an exhaustive differential test) → `a66eef6`+fix
  (JSON-schema→GBNF + json_object + OpenAI response_format wiring, e2e). Reviewed
  PASS (Tasks 4+5, each caught+fixed a real over-permit). CPU 71/71. Covers
  JSON-schema/json_object/regex/choice/GBNF; STRUCTURAL_TAG + reasoning-gating +
  spec-decode + xgrammar-parity deferred.
- ☑ **M3.5 C API + packaging**: `include/vllm.h`, shared/static lib,
  examples/cli. DoD: LocalAI-style consumption smoke test via dlopen/purego.
  Done `d6a3f39` (pure-C ABI: opaque handles, no-throw, thread-local error,
  vllm_complete) → `12ce21c` (vllm_complete_stream + callback + early-stop) →
  `0b252ec` (libvllm.so/.a exporting ONLY the 11 vllm_* symbols [nm-verified +
  ctest-enforced], examples/cli, dlopen smoke) → review-fix (unique per-call
  request ids + RAII abort — fixed an ASan-confirmed heap-UAF on mid-stream
  error). Reviewed PASS after fix; ASan/UBSan clean; C-header compile-checked.
  DoD MET. (Chat-via-C-API/embeddings/LoRA deferred.)
- ☑ **M3.6 Server e2e conformance suite** (`34d3d7b` — 23 cases/252 assertions
  over the real cpp-httplib socket: completions/chat stream+non-stream,
  tool_choice auto-relaxed, grammars/response_format, error shapes, endpoints,
  UTF-8 safety, concurrency; no contract violations). + nightly dgx.casa pipeline
  (dgx-side, pending GB10 bring-up).
- ☑ **M3.7 README + docs** (`this commit` — house-style README with the built
  CPU capabilities, quick-start [build/serve/CLI/C-API], honest support tables +
  a Status & caveats section flagging the dgx-pending real-model/throughput gates).
  **← MVP: CPU-side milestones DONE; the remaining acceptance gates are the dgx
  bring-up (CUDA kernels + 35B paged greedy + throughput-parity-vs-vLLM on GB10),
  which need the hardware — `scripts/dgx-bringup.sh`.**

## Recurring (any time after M0)

- ☐ **P1 Sync tooling** (`tools/sync/`): automate sync-cycle steps 2–4 of
  [upstream-sync.md](upstream-sync.md) — enumerate `PIN..TARGET` per ported
  subtree, map commits to our mirrored files via their headers, draft the
  classification report. DoD: one command emits a ready-to-review
  `.agents/sync/` report.
- 🔁 **Sync cycle** ([upstream-sync.md](upstream-sync.md)) — run on cadence
  (weekly or on demand); each run is a bounded, self-contained task.

## Post-MVP (T1 queue, reorder as needed)

- ☐ **Kernel drop-in alignment** (user-mandated): reshape vt CUDA/ROCm kernel
  entry points to match upstream csrc signatures so upstream kernels drop in;
  see [backends.md](backends.md) §drop-in. Start after MVP gates pass.

Dense/MoE model families (Llama/Qwen3/Mixtral…) · MTP spec decode · fp8 ·
sliding window · priority scheduling · YaRN · prompt_logprobs/logit_bias/
bad_words · tokenize endpoints · full metrics · Qwen3-Next. Then T2 per
[porting-inventory.md](porting-inventory.md), including backend expansion
per [backends.md](backends.md): Apple Metal (explorations E1 MLX vs E2
native MSL first), Vulkan, Intel XPU (loyal port of upstream
`platforms/xpu`), ANE for encoder/pooling classes (E3).
