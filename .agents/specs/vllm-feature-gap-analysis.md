# vLLM feature-gap analysis (`CLAIM-FEATURE-GAP-SPIKE`)

> **Re-verified 2026-08-10 against `main` `16d5c2ce` ([#243](https://github.com/mudler/vllm.cpp/issues/243)).**
> The original sweep is dated 2026-07-28 at base `308c312a`. OUR side moved a
> long way since, and nine of the sixteen HIGH/MED rows were stale, several
> listed `MISSING` while the code was already in the tree AND reachable from the
> server. Statuses below now carry a `verified 2026-08-10` note naming what was
> checked; rows that closed outright moved to "Already HAVE".
>
> **The pin did NOT move.** Upstream is still `555967922` (0.26.0.dev0), the same
> revision this spike swept, so no re-sweep of vLLM is owed and nobody should
> re-run the four parallel area agents. Only our column changed. When the pin
> advances (see [upstream-sync.md](../upstream-sync.md)) the upstream half of
> every row needs re-reading and this note must be updated to say so.
>
> **Method.** Each row was checked against the TREE, not against
> `docs/FEATURES.md` — FEATURES was itself drifting at the time
> ([#242](https://github.com/mudler/vllm.cpp/issues/242)). Where the two
> disagreed, the tree won. The lesson worth keeping: a gap list decays from the
> bottom, because the HIGH rows are ranked HIGH precisely because they are common
> and single-box, which is also why they get implemented first. Re-verify before
> scoping off this file.

An honest, prioritized map of what pinned vLLM `555967922` (0.26.0.dev0) has
that vllm.cpp does NOT. Records-only, CPU/research; no build, no GPU. Base
`main` `308c312a`. vLLM source read at `/home/mudler/_git/vllm` (HEAD
`5559679`). Every gap below is grounded in real vLLM `file:line` and
cross-checked against [feature-matrix.md](../feature-matrix.md),
[engine-matrix.md](../engine-matrix.md),
[model-matrix.md](../model-matrix.md),
[quantization-matrix.md](../quantization-matrix.md) and
[backend-matrix.md](../backend-matrix.md) so we never re-flag something we
already HAVE. Naming: `MISSING` = no implementation and (usually) an
`INVENTORIED` row; `RECORDS-GAP` = capability has no stable row at all;
`PARTIAL` = a bounded slice exists.

## Scope

- **In:** the full cross-cutting vLLM capability surface — adapters (LoRA),
  disaggregated serving / KV connectors, speculative-decode breadth, the
  quantization registry, structured-output backends, non-generative (pooling)
  tasks, the OpenAI serving surface, engine/runtime (plugins, sleep/wake,
  RLHF, offload, loaders), parallelism, and platforms.
- **Out:** implementation. This spike produces the ranked gap list, the
  roadmap rows for the material high-priority misses, and the honest "already
  HAVE" list. No code, no gate.
- **Rows touched:** none created in the counted matrices (analysis only). New
  gap rows are added to `feature-matrix.md` (uncounted coverage view); the
  three `RECORDS-GAP` items are flagged for future row creation, not created
  here.

## Upstream chain

Pinned vLLM `555967922` @ `/home/mudler/_git/vllm`. Key registries swept:
`vllm/lora/`, `vllm/v1/spec_decode/` + `vllm/config/speculative.py`,
`vllm/model_executor/layers/quantization/__init__.py`,
`vllm/v1/structured_output/`, `vllm/reasoning/`,
`vllm/model_executor/layers/pooler/` + `vllm/tasks.py`,
`vllm/entrypoints/` (modularized per-feature `api_router.py`),
`vllm/distributed/kv_transfer/` + `vllm/v1/kv_offload/`,
`vllm/config/parallel.py`, `vllm/plugins/`, `vllm/model_executor/model_loader/`,
`vllm/platforms/`. Runtime dispatch anchors: spec-decode
`vllm/v1/worker/gpu_model_runner.py:596-654`; quant registry
`__init__.py:141-171`; structured-output backend switch
`vllm/v1/structured_output/__init__.py:133-153`.

## Our baseline

The matrices already inventory most of this surface as `INVENTORIED`/`PARTIAL`
rows with `planned:` specs. What this sweep adds: (1) confirmed vLLM
`file:line` for each, (2) an honest priority ranking, (3) three capabilities
that have NO row anywhere (batch API, plugin system, generic draft-model /
Medusa spec-decode), and (4) the correction that vLLM has REMOVED prompt
adapters, so that is not a gap.

---

## Prioritized gap table

Priority key: **HIGH** = common, user-facing, single-box relevant; **MED** =
useful but narrower or cluster-leaning; **LOW** = niche / specific-HW /
already-well-scoped long tail. Effort: S ≤ ~1 claim, M = multi-claim, L =
program.

### HIGH priority

| Gap | vLLM file:line | Our status | Effort | Notes |
|---|---|---|---|---|
| LoRA / multi-LoRA runtime (punica batched apply) | `vllm/lora/lora_model.py:60`, `vllm/lora/punica_wrapper/punica_gpu.py:33`, `vllm/lora/ops/triton_ops/lora_shrink_op.py`, `vllm/v1/worker/lora_model_runner_mixin.py:30` | **PARTIAL** (was MISSING) — `LORA-RUNTIME` | L | **verified 2026-08-10:** a CPU brick exists (`src/vllm/lora/punica_cpu.cpp`, `include/vllm/lora/{punica,lora_weights}.h`) but is UNWIRED — grep finds no reference from `src/vllm/v1/worker/`, `src/vllm/entrypoints/` or `src/capi/`. So the subsystem is no longer absent, but nothing reaches it. Still the highest-demand missing USER feature, because unreachable equals unusable. |
| LoRA dynamic load/unload endpoints + resolver | `vllm/entrypoints/serve/lora/api_router.py:43,59`, `vllm/lora/resolver.py:14` | MISSING (`LORA-ENDPOINTS` INVENTORIED) | M | **verified 2026-08-10:** absent from the server route table (13 registered routes; see the endpoint row below). `POST /v1/{load,unload}_lora_adapter`; builds on the runtime being wired first. |
| Pooling task class: embedding / classify / score / rerank (models) | `vllm/model_executor/layers/pooler/abstract.py:16`, `seqwise/methods.py:37-60`, `tokwise/methods.py:35-86`, `vllm/tasks.py:10`, `vllm/v1/worker/gpu/pool/pooling_runner.py` | **PARTIAL** (was MISSING, "cannot serve at all") — `MODEL-POOLING` | M | **verified 2026-08-10:** the non-generative runner IS landed — `src/vllm/v1/worker/gpu/pool/pooling_runner.cpp` + `src/vllm/model_executor/layers/pooler/dispatch_pooler.cpp`, and `LlamaModel` is registered as a task=embed arch. **embed only.** classify/score/rerank heads are landed ops with NO registered arch, which is the real remaining gap. |
| Pooling / embeddings / classify / score / rerank endpoints | `vllm/entrypoints/pooling/embed/api_router.py:28`, `pooling/scoring/api_router.py:37,71`, `pooling/classify/api_router.py:26` | **PARTIAL** (was MISSING) — `SERVE-POOLING-ENDPOINTS` | M | **verified 2026-08-10:** `POST /v1/embeddings` is registered and live (`api_server.cpp:1029`). `/pooling`, `/classify`, `/score`, `/rerank` (+v1/v2) are still absent from the route table. |
| AWQ quantization (native compute) | registry `vllm/model_executor/layers/quantization/__init__.py:142`, `auto_awq.py:171` (marlin :414, MoE :547) | **PARTIAL** (was MISSING, "no dequant/compute path") — `QUANT-AWQ` | M | **verified 2026-08-10:** `src/vllm/model_executor/model_loader/awq_gptq_dequant.cpp` exists — a CPU dequant path. The gap is now native GPU COMPUTE (marlin/MoE), not the absence of any path. |
| GPTQ quantization (native compute) | registry `__init__.py:152`, `auto_gptq.py:97` (linear :306, MoE :467) | **PARTIAL** (was MISSING) — `QUANT-GPTQ` | M | **verified 2026-08-10:** same CPU dequant TU as AWQ, plus `src/vt/cuda/marlin/.../gptq_marlin_repack.cuh`. Native GPU compute still open. |
| xgrammar structured-output backend | `vllm/v1/structured_output/backend_xgrammar.py:36` (grammar :136, structural-tag :345) | **CLOSED** — moved to "Already HAVE" | — | **verified 2026-08-10:** `src/vllm/v1/structured_output/backend_xgrammar.cpp` + `xgrammar_json_schema.cpp` + `include/vllm/v1/structured_output/backend_xgrammar.h`. No longer a gap. |
| fp8 KV cache (`cache_dtype=fp8*`) | `vllm/config/cache.py:76` (`CacheDType`:19), `vllm/model_executor/layers/quantization/kv_cache.py:42` | **PARTIAL** (was MISSING) — `KV-FP8` / `QUANT-KV-FP8` | M | **verified 2026-08-10:** `include/vt/fp8_kv.h` is present and the CPU store/read path landed. CUDA + the memory-halving e2e are the live residual, so the memory lever itself is still unrealised. |

### MEDIUM priority

| Gap | vLLM file:line | Our status | Effort | Notes |
|---|---|---|---|---|
| Reasoning parsers (+ reasoning-gated grammar) | `vllm/reasoning/__init__.py:22` (25+ parsers: deepseek_r1 `deepseek_r1_reasoning_parser.py:10`, qwen3, granite `granite_reasoning_parser.py:18`, gpt-oss, gemma4, glm, kimi_k2, minimax, olmo3 …) | **CLOSED** — moved to "Already HAVE" | — | **verified 2026-08-10:** 11 TUs under `src/vllm/entrypoints/openai/reasoning_parsers/` (`abstract`, `deepseek_r1`, `step3`, …). Breadth vs upstream's 25+ is a separate question from existence; this row claimed non-existence. |
| Separate draft-model spec decode (generic) | `vllm/v1/spec_decode/draft_model.py:19`, config method `"draft_model"` `speculative.py:684,1195`, runner `gpu_model_runner.py:604` | ROW CREATED 2026-07-29 → `SPEC-DRAFT-MODEL` ACTIVE (`CLAIM-SPEC-DRAFT-MEDUSA`) | M | W0 spike + W1 CPU brick: `DraftModelProposeGreedy` k-step autoregressive greedy propose over a `DraftLogitsFn` oracle, reusing the LANDED `SPEC-REJECTION` verify UNCHANGED; unit 6/6 RED-first (accepted==target greedy; full-acceptance depends on feed-back); `draft_model` config accept. Residuals: real GPU draft forward + DGX e2e greedy + speed gate (W3). See [draft-model-medusa-spec.md](draft-model-medusa-spec.md). |
| Medusa spec decode | `vllm/v1/spec_decode/medusa.py:18`, method `"medusa"` `speculative.py:822`, runner `:642` | ROW CREATED 2026-07-29 → `SPEC-MEDUSA` SPIKE (`CLAIM-SPEC-DRAFT-MEDUSA`) | M | W0 spike only; the N-head single-pass proposer is deferred to W2 (needs the target's Medusa heads, a model change). Verify/accept reuses `SPEC-REJECTION`. See [draft-model-medusa-spec.md](draft-model-medusa-spec.md). |
| EAGLE (base v1) + suffix + ngram_gpu + extract_hidden_states + custom_class | `eagle.py:10` (`"eagle"` runner `:636`), `suffix_decoding.py:9`, `ngram_proposer_gpu.py:217`, `extract_hidden_states.py:29`, `custom_class_proposer.py` | MISSING/RECORDS-GAP — `SPEC-EAGLE3` covers eagle3 only; base EAGLE + others unrowed | M | Breadth beyond eagle3; `mlp_speculator` is config-valid but has NO v1 runtime branch upstream (parity floor = nothing owed). |
| Offline Batch API (JSONL file runner) | `vllm/entrypoints/openai/run_batch.py:793` (`run_batch`), input/output `:148,:213`; batched chat/embed/score/transcription/translation | `SERVE-BATCH-API` — **landed, residuals named** | S | **verified 2026-08-10:** `src/vllm/entrypoints/openai/run_batch.cpp` + `include/.../run_batch.h`. `run_batch.cpp:188-199` shows the dispatch now recognises `/v1/embeddings`, `/score`, `/rerank`, `/v1/audio/transcriptions` and explicitly names `/v1/audio/translations` as a residual in its own error text. `docs/FEATURES.md` marks the offline batch API supported. See [batch-api.md](batch-api.md). |
| `/v1/responses` (+retrieve/cancel), `/v1/messages` (Anthropic) + count_tokens | `vllm/entrypoints/openai/responses/api_router.py:48,80,110`, `vllm/entrypoints/anthropic/api_router.py:49,95` | MISSING (`SERVE-RESPONSES-MESSAGES` INVENTORIED) | M | Responses API + Anthropic-compat surface. |
| Audio transcription endpoint | `vllm/entrypoints/speech_to_text/transcription/api_router.py:31` | **CLOSED** — moved to "Already HAVE" | — | **verified 2026-08-10:** `POST /v1/audio/transcriptions` is registered in the route table, and `vllm_transcribe` is on the C ABI at v11. The row claimed "the endpoint is not wired"; it is. |
| Audio **translation** + realtime endpoints | `translation/api_router.py:31`, `realtime/api_router.py:17` | MISSING (split out of the transcription row) | M | **verified 2026-08-10:** `/v1/audio/translations` appears in the tree ONLY as a named residual inside `run_batch.cpp:190,199`; no route registers it. `/v1/audio/realtime` absent entirely. |
| Plugin system (general / io_processor / platform / endpoint plugins) | `vllm/plugins/__init__.py:18,77`, `vllm/plugins/io_processors/interface.py:19`, `vllm/plugins/endpoint_plugins/interface.py:43` | **PARTIAL** (was RECORDS-GAP, "no row") — `ENG-PLUGIN-SYSTEM` | M | **verified 2026-08-10:** `src/vllm/plugins/plugins.cpp` + `include/vllm/plugins/plugins.h` exist, and `docs/FEATURES.md` marks the in-tree factory done plus a plugin seam. Which of upstream's four plugin KINDS (general / io_processor / platform / endpoint) the seam covers was NOT established by this re-verification and is the next question for whoever picks it up. |
| Sleep/wake (CuMemAllocator L1/L2) + RLHF weight-update + profiler endpoints | `vllm/device_allocator/cumem.py:82,229`, `vllm/v1/worker/gpu_worker.py:190,1263,1103`, `vllm/entrypoints/serve/dev/{sleep,rlhf,profile}/api_router.py` | PARTIAL (`SERVE-ADMIN` — only `/abort_requests` landed) | M | Sleep/wake + `update_weights`/`collective_rpc` are the RLHF-training integration surface; profiler is dev tooling. |
| guidance / outlines / lm-format-enforcer backends | `backend_guidance.py:88`, `backend_outlines.py:53`, `backend_lm_format_enforcer.py:95` | MISSING (`TOOLS-GUIDANCE-OUTLINES` INVENTORIED) | M | Additional structured-output engines after xgrammar. |
| Data parallel (DP) + Expert parallel (EP) + EPLB | `vllm/config/parallel.py:129,165,174`, `vllm/distributed/eplb/eplb_state.py:220`, `vllm/v1/engine/coordinator.py:23` | MISSING (`PAR-DP`, `PAR-EP-EPLB` INVENTORIED) | L | Scale-out; TP/PP already spiked. EPLB rebalances routed experts. |
| KV offloading (CPU tiering, LRU/ARC) | `vllm/v1/kv_offload/cpu/manager.py:37`, `policies/lru.py:12`, `policies/arc.py:12`, `base.py:187` | **LANDED and REACHABLE** (was MISSING) — `KV-OFFLOAD` | — | **verified 2026-08-10, the single most wrong row in this file.** A full `kv_offload/` subsystem exists in both `include/vllm/v1/kv_offload/` and `src/vllm/v1/kv_offload/`: `base`, `cache_identity`, `cache_policy`, `cpu_manager`, `fs_io`, `fs_tier`, `kv_block_transfer`, `kv_connector`, `tiering_manager`. It is WIRED — `KVConnectorFactory` is consulted in `server_main.cpp:664-669` and built in `model_loader.cpp:478`. `docs/FEATURES.md` marks KV offload to host memory supported. Residual: confirm which eviction policies `cache_policy` actually implements against upstream's LRU/ARC. |
| External KV cache + connectors (LMCache, NIXL, Mooncake, PD disaggregation) | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`, `lmcache_connector.py:72`, `nixl/connector.py:322,350`, `mooncake/mooncake_connector.py:469`, `config/kv_transfer.py:41` | **PARTIAL** (was MISSING) — `KV-EXTERNAL-CACHE`, `KV-CONNECTORS` | L | **verified 2026-08-10:** an LMCache client subtree exists — `include/vllm/v1/kv_offload/lmcache/` carries `lmcache_connector.h`, `remote_client.h`, `remote_protocol.h`, `cache_engine_key.h`, `chunked_token_database.h`, `token_hasher.h`, `memory_format.h`, but `src/.../lmcache/` has only `cache_engine_key.cpp` + `chunked_token_database.cpp`, so the client is headers-ahead-of-implementation. `docs/FEATURES.md` still marks the external KV provider ABI unsupported, which is consistent. NIXL / Mooncake / PD disaggregation remain untouched. |
| compressed-tensors generic + quark + bitsandbytes + torchao + moe_wna16 + mxfp4-MoE + modelopt-generic | `compressed_tensors/compressed_tensors.py:82`, `quark/quark.py:57`, `bitsandbytes.py:49`, `torchao.py:134`, `moe_wna16.py:47`, `mxfp4.py:40,102`, `modelopt.py:370,1017` | MISSING (`QUANT-BNB`, `QUANT-QUARK`, `QUANT-TORCHAO`, `QUANT-MOE-WNA16` INVENTORIED; CT schemes rowed `QUANT-CT-*`) | L | Broad quant registry; gate-specific NVFP4/FP8 done, generic dispatch open. mxfp4 backs gpt-oss MoE. |
| MLA + Mamba/linear attention backends | `vllm/v1/attention/backends/mla/`, `.../mamba` | **LANDED** (was MISSING) — `ATTN-MLA`, `ATTN-MAMBA` | — | **verified 2026-08-10:** MLA is real — `src/vllm/model_executor/layers/attention/mla_attention.cpp`, `include/vllm/model_executor/models/mla_attention.h`, `include/.../mla_chunked_context.h`, plus CPU kernels `src/vt/cpu/cpu_mla_{attn,prefill}.cpp` — and four registered archs use it (DeepSeek-V2, MiniCPM3, GLM-4.7-Flash, Kimi-Linear NoPE-MLA). The linear/SSM side ships as GDN with vendored per-arch Triton-AOT kernels (`src/vt/cuda/triton_aot_vendored/*/gdn_*.h`) and exact-chunks Mamba (`KERNEL-SSM-MAMBA-EXACT-CHUNKS`, merged #127). Calling these MISSING was wrong at the time of writing, not merely stale. |
| Encoder-decoder / cross-attention runtime | `vllm/model_executor/models/interfaces.py:1013` (`SupportsCrossEncoding`), `config/model.py:1649`, registry Whisper/Mllama/Bart `:595,744,772` | MISSING (`ATTN-ENCODER-CROSS`, `KV-CROSS-ENCODER-SPECS` INVENTORIED) | L | General enc-dec beyond our audio-encoder special-case. |
| Weight CPU offload (UVA / Prefetch offloader, `cpu_offload_gb`) | `vllm/config/offload.py:23`, `vllm/model_executor/offloader/{uva.py:21,prefetch.py:127}` | MISSING (`ENG-WEIGHT-OFFLOAD` INVENTORIED) | M | Per-parameter UVA offload; mirror floor for expert streaming. |

### LOW priority (niche / specific-HW / long tail — listed, not roadmapped individually)

| Gap | vLLM file:line | Our status |
|---|---|---|
| Sequence-parallel MoE (non-DP) | `vllm/config/parallel.py:673` | `PAR-SEQUENCE-MOE` INVENTORIED |
| DBO / dual-batch overlap | `vllm/config/parallel.py:211` | `ENG-DBO-UBATCH` INVENTORIED |
| `step_with_batch_queue` pipelined queue | `vllm/v1/engine/core.py` | `ENG-BATCH-QUEUE` INVENTORIED |
| Partial-prefill concurrency | `vllm/config/scheduler.py` | `ENG-PARTIAL-PREFILL` INVENTORIED |
| Opt-in batch-invariant execution | `vllm/model_executor/layers/batch_invariant.py` | `ENG-BATCH-INVARIANT` INVENTORIED |
| nvfp4 / turboquant KV | `vllm/config/cache.py`, `.../turboquant/config.py` | `KV-NVFP4-TURBO` INVENTORIED |
| Philox bit-exact RNG | `vllm/v1/sample/ops/` | `SAMPLE-PHILOX` INVENTORIED |
| Thinking budget / repetition detection / routed-expert logprobs / logprob token-ids | `vllm/v1/sample/` | `SAMPLE-THINKING-BUDGET`/`-REPETITION`/`-ROUTED-EXPERTS`/`-LOGPROB-TOKEN-IDS` INVENTORIED |
| OTLP tracing | `vllm/tracing/__init__.py:66`, `config/observability.py:36` | `SERVE-OTLP` INVENTORIED |
| experts_int8 / INC / humming / online-dynamic quant | `experts_int8.py:22`, `inc/inc.py:32`, `humming.py:170`, `online/base.py:79` | folded in `QUANT-VLLM-BREADTH` (feature-matrix) |
| Loaders: tensorizer / runai-streamer / bitsandbytes / sharded-state / modelexpress | `model_loader/{tensorizer_loader.py:43,runai_streamer_loader.py:21,bitsandbytes_loader.py:56,sharded_state_loader.py:29}` | `LOAD-LONGTAIL` INVENTORIED |
| Custom ops OOT registration | `vllm/model_executor/custom_op.py:84` (`register_oot`) | no engine row (kernel-matrix has our own custom-op seam) |
| TPU / ROCm / XPU / Vulkan platforms | `vllm/platforms/{tpu,rocm,xpu}.py` | `BACKEND-TPU`/`-ROCM`/`-XPU`/`-VULKAN` INVENTORIED |
| SageMaker / elastic-EP / fault-tolerance / render-derender / token-in-token-out / DP-supervisor endpoints | `vllm/entrypoints/serve/{sagemaker,elastic_ep,fault_tolerance}/…`, `scale_out/…` | no rows (cluster-ops niche) |
| MLP-speculator / custom_class proposer | `mlp_speculator.py`, `custom_class_proposer.py` | no rows (mlp_speculator has no v1 runtime upstream) |

---

## Already HAVE — naive-scan false-positives (do NOT re-roadmap)

**Closed since the 2026-07-28 sweep** (verified in the tree 2026-08-10, moved
down from the HIGH/MED tables above rather than deleted, so the record of why
they were once flagged survives):

- **xgrammar structured-output backend** — `src/vllm/v1/structured_output/backend_xgrammar.cpp`, `xgrammar_json_schema.cpp`.
- **Reasoning parsers** — 11 TUs under `src/vllm/entrypoints/openai/reasoning_parsers/`.
- **`/v1/audio/transcriptions`** — registered route + `vllm_transcribe` on the C ABI (v11). Translation and realtime are still open and now carry their own row.
- **Offline Batch API** — `src/vllm/entrypoints/openai/run_batch.cpp`, with embeddings/score/rerank/transcription dispatch.
- **KV offload (CPU tiering)** — the whole `kv_offload/` subsystem, wired through `KVConnectorFactory` at `server_main.cpp:664`.
- **MLA and GDN/Mamba attention** — `mla_attention.cpp` + CPU MLA kernels + vendored per-arch GDN Triton-AOT; four registered MLA archs.
- **DSpark speculative decoding** — landed via #211 after this sweep; `docs/FEATURES.md` carries it with an honest "no speed claim" note.

**Flagged as false positives in the original sweep and still true:**

- **ngram, MTP, DFlash spec decode** — `SPEC-NGRAM`/`SPEC-MTP`/`SPEC-DFLASH` all DONE.
- **Prometheus `/metrics`** — `SERVE-METRICS` ACTIVE (catalog + live per-step wiring).
- **Custom logits processors** — `SAMPLE-CUSTOM-PROCESSORS` ACTIVE (C-ABI callback).
- **`n>1`, `best_of`, beam search, logprobs, prompt_logprobs, logit_bias / allowed_token_ids / bad_words** — `SAMPLE-N`/`-BEST-OF`/`-BEAM` + logprobs rows ACTIVE/DONE.
- **Priority scheduling + request `priority` param** — GATING (`vllm/entrypoints/openai/completion/protocol.py:132`).
- **Prefix caching (APC), chunked prefill, continuous batching, preemption** — present (APC DONE).
- **Tool calling + unified streaming parser engine** — many engine-backed families ACTIVE.
- **Utility endpoints** `/tokenize` `/detokenize` `/tokenizer_info` `/abort_requests` `/reset_prefix_cache` `/server_info` `/ping` — ACTIVE.
- **YaRN / llama3 / longrope / dynamic-NTK rope, sliding-window + chunked-local attention** — ACTIVE, GPU-gated.
- **CUDA graphs (decode capture/replay)** — PARTIAL (`FULL`/`PIECEWISE` mode breadth open; no torch.compile — not applicable to a torch-free engine).
- **Collective / process-group abstraction** — W1 CPU exact-gate landed.
- **Prompt adapters** — vLLM has REMOVED prompt adapters entirely (no `PromptAdapter` in `vllm/`); correctly absent here, NOT a gap.
- **structural_tag** — PARTIAL (tool-choice subset).

## Port map

Analysis deliverable → records only:
`.agents/specs/vllm-feature-gap-analysis.md` (this file) +
[roadmap_v1.md](../roadmap_v1.md) high-priority rows +
[feature-matrix.md](../feature-matrix.md) gap rows +
[coordination.md](../coordination.md) `CLAIM-FEATURE-GAP-SPIKE` note +
[docs/STATUS.md](../../README.md#project-status) + [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)
(NOT-APPLICABLE — spike) + [parity-ledger.md](../parity-ledger.md) +
[state.md](../completed/state-events/). No `src/` port in this spike.

## Tests to port

None (analysis spike). Each gap's tests are named in its own future spec when
that row transitions `READY`. The recommended new rows (`SPEC-DRAFT-MODEL`,
`SPEC-MEDUSA`, `SERVE-BATCH-API`, `ENG-PLUGIN-SYSTEM`) carry their upstream
test module pointers there.

## Gates

Records-only: the six record checkers rc=0 after commit. No correctness or
performance gate is owed by an analysis spike (docs/BENCHMARKS.md
NOT-APPLICABLE).

## Dependencies

None to implement here. Downstream: pooling endpoints depend on the pooling
runner; LoRA endpoints on the LoRA runtime; xgrammar/guidance/outlines on their
respective C++ cores; KV connectors on the external-cache ABI; DP/EP on the
`vt::Communicator` W2 transports.

## Work breakdown

1. Sweep vLLM surface (4 parallel area agents) — DONE.
2. Cross-check each candidate against the five matrices — DONE (three
   RECORDS-GAP items found; prompt-adapters confirmed removed upstream).
3. Rank + write this spec — DONE.
4. Roadmap the material HIGH gaps + feature-matrix coverage rows — this commit.
5. (future, separate claims) create the three missing stable rows and pick up
   the HIGH gaps in priority order.

## Risks / decisions

- **Do not create counted-matrix rows in an analysis spike.** New stable rows
  bump the `check-agent-record.py` inventory constants and demand the full row
  contract; the three RECORDS-GAP items are flagged for a future
  implementing claim, not created here. This keeps the sweep honest and the
  checkers green.
- **Honesty:** where a naive scan would flag a gap we already closed (ngram,
  MTP, DFlash, metrics, custom logits, beam/`n`/`best_of`, utility endpoints,
  rope breadth), it is listed under "Already HAVE" and NOT roadmapped.
- **Priority is single-box-first:** LoRA, pooling/embeddings, AWQ/GPTQ,
  xgrammar and fp8-KV rank HIGH because they are common on one GPU; KV
  connectors / DP-EP / PD-disaggregation are MED because they are cluster-scale
  and the collective abstraction is only at W1.
- **Staleness is the standing risk in this file, not inaccuracy at authoring
  time** (2026-08-10, #243). Nine of sixteen HIGH/MED rows had gone wrong in
  ~2 weeks, all in the same direction: understating what we have. Two rows
  (KV offload, MLA/Mamba) appear to have been wrong when written, not merely
  overtaken — both had substantial code the sweep did not find, which suggests
  the original agents searched by upstream concept name rather than by our own
  file layout. Anyone re-running this sweep should grep OUR tree for the
  capability before writing `MISSING`, and prefer `PARTIAL` with a named
  residual over a binary verdict.

## Post-reconcile priority (2026-08-10)

What is genuinely HIGH after the corrections above, for scoping:

1. **LoRA end to end** — the brick exists and nothing reaches it; wiring it is
   the single biggest user-visible win.
2. **AWQ / GPTQ native GPU compute** — CPU dequant is not the lever people want.
3. **fp8 KV on CUDA** — the memory halving is still unrealised.
4. **classify / score / rerank** — heads are landed ops with no registered arch;
   the endpoints follow from that, not the other way round.
5. **`/v1/responses` + Anthropic `/v1/messages`** — untouched, and the largest
   remaining OpenAI-surface gap now that transcription and embeddings are live.
