# Async serving + overlap scheduling — joint spike

Rows covered (engine-matrix): `SERVE-ASYNC-LLM` (AsyncLLM-equivalent streaming
engine API), `ENG-CORE-BUSY-LOOP` (engine busy loop + input/output queue
split), `ENG-ASYNC-SCHED` (async/overlap scheduling), `ENG-PRIORITY-SCHED`
(priority scheduling). Roadmap block: `ROAD-V1-C6`; blocking dependency of
`ROAD-V1-A` (see priority justification). Upstream pin:
`/home/mudler/_git/vllm` @ `e24d1b24`.

## Priority justification (why this outranks its old T1 slot)

1. **The order-0 gate is structurally blocked on this work.** The
   `SERVE-GATE-ONLINE` campaign found our example server executes the engine
   **synchronously per request** under a server-wide mutex
   (`src/vllm/entrypoints/openai/api_server.cpp:18-27,86,130`) and the
   "streaming" path collects every delta after generation completes, then
   replays precomputed SSE chunks
   (`include/vllm/entrypoints/openai/serving_completion.h:9-12`,
   `src/vllm/entrypoints/openai/api_server.cpp:185-202`). `vllm bench serve`
   measures TTFT at first-SSE-chunk arrival and ITL per subsequent chunk
   (`vllm/benchmarks/lib/endpoint_request_func.py:96-98,231-242`; TPOT derived
   as `(latency-ttft)/(output_len-1)`, `vllm/benchmarks/serve.py:608-611`), so
   against our server TTFT ≈ full completion latency and ITL ≈ 0:
   **TTFT/TPOT/ITL are structurally unmeasurable**, and the every-axis online
   comparison (benchmark-protocol.md) cannot even run, let alone pass.
   `SERVE-ASYNC-LLM` is therefore a blocking dependency of roadmap order 0
   (`ROAD-V1-A`) — the edge is recorded in the roadmap portfolio rows in the
   same change as this spike.
2. **Unmet mirror obligation.** The B3 research finding (archived at
   `completed/roadmap_v1_research_spikes_2026-07-10.md` row B3) established
   that async/overlap scheduling is vLLM's **DEFAULT** at our pin: a `None`
   `async_scheduling` resolves to **True** unless an incompatibility forces it
   off (`vllm/config/vllm.py:990-1038`; config field
   `vllm/config/scheduler.py:158-162`). Our engine is synchronous end to end,
   so every A/B we run compares our sync loop against vLLM's overlapped
   default — a mirror gap on the denominator itself.

### 2026-07-11 execution update: live SSE works; HTTP capacity is now binding

W1/W2/W4 subsequently landed and the exact pushed-`a531e05` online campaign
proves live/native-count streaming at every standard point. It also turns D3
from a hypothetical risk into a reproduced defect. On the 20-CPU DGX,
cpp-httplib starts 19 workers and grows only when `idle_thread_count_ == 0` at
the enqueue instant (`third_party/httplib/httplib.h:161-169,10359-10377`). A
streaming socket owns that worker through collector waits and keepalive. At c32,
repetitions 1 and 3 each left one accepted connection with 2,131 unread request
bytes and zero response bytes for roughly 205–207 seconds while the other 31
streams completed normally; the healthy repetition grew enough workers and
reached 1087.15 tok/s. `CLAIM-SERVE-HTTP-POOL-1` therefore owns a bounded W2
repair: deterministic worker capacity for the configured concurrent-stream
floor, a persistent-client regression, disconnect/teardown checks, and exact
c32/full-ladder evidence. This repair does not authorize W3 scheduler overlap.

## Scope

| Aspect | In scope | Out of scope (owning row) |
|---|---|---|
| Rows | `SERVE-ASYNC-LLM`, `ENG-CORE-BUSY-LOOP`, `ENG-ASYNC-SCHED`, `ENG-PRIORITY-SCHED` | `ENG-BATCH-QUEUE` PP-general pipelining (only the depth-2 async subset is used here); `SERVE-CPP-API` rich API packaging; `SERVE-STREAM-USAGE` usage chunk; `SERVE-METRICS` stat loggers |
| Engine API | AsyncLLM-equivalent: submit-and-stream over a dedicated engine thread, per-request output streams, abort, `n=1` | Parallel sampling fan-out (`ParentRequest`, `async_llm.py:389-397`) stays deferred as today; streaming input (`_add_streaming_input_request`, `async_llm.py:320-331`); pooling |
| Process model | In-process engine thread + `std::deque`+mutex/condvar queues (analog of the EngineCoreProc busy loop; deviation recorded in Port map) | ZMQ/multi-process split, DP/EP coordination, multi-client indexes |
| Scheduling modes | `async_scheduling` on/off with vLLM's default-ON resolution and compat fallbacks (`vllm/config/vllm.py:990-1038`); `policy` = `fcfs` and `priority` (`vllm/config/scheduler.py:109-117`) | Custom `scheduler_cls` plugin seam (`vllm/config/scheduler.py:180-203` warning path) |
| Serving | Real per-token SSE emission on `/v1/completions` + `/v1/chat/completions` (chunk content already matches upstream; the transport/cadence becomes real), concurrent requests through one engine, client-disconnect abort | New endpoints; `stream_options`/`include_usage` (`SERVE-STREAM-USAGE`) |
| C ABI | Non-blocking request submission + per-delta delivery (design decision D1 below); existing blocking `vllm_complete_stream` kept | Breaking existing ABI symbols |
| Spec decode interplay | Compat rules mirrored: async sched allowed with EAGLE/MTP-family methods, rejected otherwise (`vllm/config/vllm.py:959-1021`) | Implementing spec-decode itself (`SPEC-MTP` et al.) |

Dispatch behavior mirrored: `SchedulerConfig.get_scheduler_cls()` selects
`AsyncScheduler` when `async_scheduling` else `Scheduler`
(`vllm/config/scheduler.py:180-189`); `VllmConfig.max_concurrent_batches`
returns 2 for async scheduling on a single GPU (MRV2: `pp_size + 1`,
`vllm/config/vllm.py:490-501`), which enables the batch queue and flips
`step_fn` to `step_with_batch_queue` (`vllm/v1/engine/core.py:196-223`).

## Upstream chain

The full execution chain, host-side only (no new kernels except one small
gather/scatter pair in W3 — see Port map):

| # | Upstream (pin e24d1b24) | Role |
|---|---|---|
| U1 | `vllm/v1/engine/async_llm.py:70` (class), `:163-176` (eager output-handler start), `:280-398` (`add_request`: process_inputs → collector queue → output_processor before engine_core, `:400-410` `_add_request`), `:524-635` (`generate`: pull loop, `q.get_nowait() or await q.get()`, abort-on-cancel), `:637-707` (`_run_output_handler`: pulls `EngineCoreOutputs`, chunked `process_outputs`, aborts stop-string finishes), `:709-745` (`abort`) | The async frontend `SERVE-ASYNC-LLM` mirrors |
| U2 | `vllm/v1/engine/output_processor.py:45-105` (`RequestOutputCollector`: single-slot coalescing queue; `put` merges DELTA outputs when producer outruns consumer `:62-76`, `get` awaits `:78-86`, `get_nowait` `:88-105`), `:512-573` (`add_request`, `STREAM_FINISHED` put at `:555`; sentinel `vllm/outputs.py:192`), `:576-693` (`process_outputs` pushes to per-request queues) | Per-request stream hand-off |
| U3 | `vllm/v1/engine/core.py:915-916` (`input_queue`/`output_queue`), `:1259-1298` (`run_busy_loop` + `_process_input_queue`: block for work, drain requests), `:1300-1318` (`_process_engine_step`: `step_fn()` → `output_queue.put_nowait`), `:479-508` (`step`: schedule → `execute_model(non_block=True)` → `get_grammar_bitmask` **overlaps the forward** → `future.result()` → `sample_tokens(grammar_output)` → `update_from_output`), `:510-517` (`post_step` draft-token path skipped under async), `:519-611` (`step_with_batch_queue`: schedule N+1 before consuming N; grammar deferral via `pending_structured_output_tokens` `:559`) | The busy loop `ENG-CORE-BUSY-LOOP` mirrors; the overlap step `ENG-ASYNC-SCHED` mirrors |
| U4 | `vllm/v1/engine/core_client.py:467` (MPClient contract: "EngineCore runs in a background process busy loop"), `:779-860` (`SyncMPClient`: background output thread + blocking `outputs_queue`), `:950-1046` (`AsyncMPClient`: asyncio `outputs_queue` + `process_outputs_socket` task) | The client side of the queue split (we collapse socket→queue, keep queue semantics) |
| U5 | `vllm/v1/core/sched/async_scheduler.py:12-75` (`AsyncScheduler(Scheduler)`: `_update_after_schedule` adds `num_output_placeholders` + spec placeholder ids `:19-51`; `_update_request_with_output` drains placeholders, discards stale frames after force-preemption via `async_tokens_to_discard`, caches blocks minus placeholders `:52-75`); placeholder accounting in the base scheduler `vllm/v1/core/sched/scheduler.py:446-475,599,1184,1304,1602`; `vllm/v1/request.py:141` | Scheduler half of overlap |
| U6 | MRV2 worker: `vllm/v1/worker/gpu/async_utils.py:12-70` (`AsyncOutput`: D2H copy of sampled ids on a dedicated copy stream + event; `get_output()` blocks on the event); `vllm/v1/worker/gpu/input_batch.py:304-406` (`combine_sampled_and_draft_tokens`: next step's decode input ids read GPU-resident `last_sampled_tokens` — token values never round-trip the host), `:457-543` (`post_update` writes `last_sampled_tokens` on-GPU); `vllm/v1/worker/gpu/model_runner.py:613,954,1423-1445` (AsyncOutput created before spec propose so the copy overlaps) | Worker half of overlap |
| U7 | Executor seam: `vllm/v1/outputs.py:298-307` (`AsyncModelRunnerOutput.get_output()` contract); `vllm/v1/executor/uniproc_executor.py:26-42` (`AsyncOutputFuture`), `:80-107` (`collective_rpc(non_block=True)`) | Future/async-output plumbing between step and runner |
| U8 | Server: `vllm/entrypoints/openai/completion/serving.py:109` (`create_completion`), `:224-236` (returns generator), `:278-310` (`completion_stream_generator`: per-`RequestOutput` SSE frames, `first_iteration` at `:296-310`); `vllm/entrypoints/openai/completion/api_router.py:66` + `vllm/entrypoints/openai/chat_completion/api_router.py:74` (`StreamingResponse(media_type="text/event-stream")`); `vllm/entrypoints/openai/chat_completion/serving.py:229,404` | Real SSE emission path |
| U9 | Priority: `vllm/v1/core/sched/request_queue.py:13-17` (`SchedulingPolicy`), `:75-128` (`FCFSRequestQueue`), `:131-198` (`PriorityRequestQueue`: heap by `(priority, arrival_time)`), `:201-208` (`create_request_queue`); preemption victim = max `(priority, arrival_time)` among running (`vllm/v1/core/sched/scheduler.py:546-556`; FCFS tail pop `:1857`) | `ENG-PRIORITY-SCHED` |
| U10 | Config/defaults: `vllm/config/scheduler.py:109-117` (`policy`), `:158-162` (`async_scheduling`), `:164-168` (`stream_interval`), `:180-203` (`get_scheduler_cls`); `vllm/config/vllm.py:490-501` (`max_concurrent_batches`), `:952,959-1038` (compat validation + default-ON resolution) | Mode selection we must mirror |
| U11 | Measurement: `vllm/benchmarks/lib/endpoint_request_func.py:96-98,197-257` (OpenAI-completions request fn: TTFT on first chunk, ITL per chunk), `:376-420` (chat variant); `vllm/benchmarks/serve.py:330-341` (metrics struct), `:581-615` (TPOT/goodput calc) | Defines what the gate measures |

**Structured-output overlap falls out for free** (per the B3 note): the sync
step already computes the grammar bitmask between launching the forward and
waiting on it (`vllm/v1/engine/core.py:492-499`), and under async scheduling
the batch-queue step defers sampling only when a grammar is actually waiting
on unresolved tokens (`pending_structured_output_tokens`,
`vllm/v1/core/sched/async_scheduler.py:31-33` → `vllm/v1/engine/core.py:559-570`).
Porting W1+W3 structurally yields this; no separate grammar work item.

**Runtime trace plan** (dispatch is dynamic): before gating W3, capture one
steady-state nsys/torch-profile of `vllm serve` (async default ON) and one
with `--no-async-scheduling` at c=16 on GB10, per parity-lever-protocol.md
§Verify-the-whole-chain, to confirm the overlap pattern the port must
reproduce (schedule N+1 host work concurrent with step N GPU tail) and to
freeze the A/B denominators.

## Our baseline (captured at spike acceptance)

This table freezes the gaps that justified the work breakdown. Current
lifecycle truth lives in `engine-matrix.md`: W1, W2 and W4 are now implemented
and `GATING`; W3 remains `READY`. In particular, the synchronous/fake-SSE
statements below describe the archived pre-W2 diagnostic binary, not the new
production server path or the current online campaign.

| Area | Anchor | Honest state |
|---|---|---|
| Engine API | `include/vllm/v1/engine/llm_engine.h:70-120`; `src/vllm/v1/engine/llm_engine.cpp:21,45,71,81` | Synchronous `LLMEngine` only: `add_request`/`step`/`generate` on the caller's thread; deferred list in header names EngineCoreClient/async explicitly (`llm_engine.h:45-52`) |
| Engine core | `include/vllm/v1/engine/core.h:65-100`; `src/vllm/v1/engine/core.cpp:37` | `EngineCore::step()` mirrors `core.py:479-508` minus futures/batch queue; header records `step_with_batch_queue`, `async_scheduling`, queues as deferred (`core.h:38-45`) |
| Threading | repo-wide grep | **No engine threading exists**: the only mutexes are the server request-serializer (`src/vllm/entrypoints/openai/api_server.cpp:26`) and lazy-init locks in `src/vllm/model_executor/models/qwen3_5.cpp:109-1084`; no `std::thread` in engine/serving code |
| Server | `src/vllm/entrypoints/openai/api_server.cpp:18-27` (engine mutex), `:86,130` (per-request lock), `:185-206` (precomputed-chunk provider) | One request at a time; SSE is replayed after the fact (the fake-SSE finding); conformance test asserts frame shape only, not arrival timing (`tests/vllm/entrypoints/openai/test_conformance.cpp:502`) |
| Serving handlers | `include/vllm/entrypoints/openai/serving_completion.h:9-12,40-46`; `include/vllm/entrypoints/openai/serving_chat.h:47`; `src/vllm/entrypoints/openai/serving_completion.cpp:19` | `create_completion` drives the engine to completion and returns `sse_chunks` as a vector; chunk **content/cadence already matches** upstream's generator — only the transport is synchronous |
| Detokenize/stream content | `src/vllm/v1/engine/detokenizer.cpp:409`; `src/vllm/v1/engine/output_processor.cpp:68,101-114` | Incremental detokenization + DELTA/FINAL_ONLY + `stream_interval` hold-back already ported (`ENG-SCHED-KNOBS` row); what is missing is only the per-request queue (`RequestOutputCollector`) — listed as deferred in `include/vllm/v1/engine/output_processor.h:39` |
| C ABI | `include/vllm.h:166-194` (`vllm_token_callback` + `vllm_complete_stream`); `src/capi/vllm_c.cpp:262-325` | Per-delta callback exists but the call is **blocking** and single-request; no submit/poll surface, `include/vllm.h:23` scopes streaming-callback as-is |
| Scheduler seam | `include/vllm/v1/core/sched/scheduler.h:83` (non-virtual class), `:201-205` (`update_after_schedule` private) | No subclass seam for `AsyncScheduler`'s two overrides; needs protected-virtual hooks (W3) |
| Request queue | `include/vllm/v1/core/sched/request_queue.h:26-27,41-44,99-100` | FCFS complete; `create_request_queue(kPriority)` deliberately throws — priority is a marked T1 hole (W4) |
| Runner | `src/vllm/v1/worker/gpu/runner.cpp:645-730`; `include/vllm/v1/worker/gpu/runner.h:141-144` | `execute_model`/`sample_tokens` split already mirrored (MRV2); sampled ids D2H synchronously + host write-back into `token_ids_cpu` each step (`runner.cpp:706-730`); no GPU-resident `last_sampled_tokens`, no copy stream/event |
| Bench harness | `examples/bench/bench_core.h:96,468`; `examples/server/main.cpp:60,121` | Offline throughput modes exist; the online serve A/B harness lives with `CLAIM-SERVE-GATE-1` (DGX evidence root `~/work/vllm.cpp-online-gate`) and consumes this work |

## Port map

Leaf W1 — `ENG-CORE-BUSY-LOOP` (engine thread + queue split):

| Upstream | Ours (new/changed) | Notes / deviations |
|---|---|---|
| `vllm/v1/engine/core.py:915-916,1259-1318` (`EngineCoreProc.__init__` queues, `run_busy_loop`, `_process_input_queue`, `_process_engine_step`) | new `src/vllm/v1/engine/core_proc.cpp` + `include/vllm/v1/engine/core_proc.h` (`EngineCoreProc` over the existing `EngineCore`) | DEVIATION (recorded): in-process `std::thread` instead of a forked process; `input_queue_`/`output_queue_` are mutex+condvar deques with the same blocking `get`/`put_nowait` semantics; no ZMQ frames — payloads move as C++ values. WAKEUP/shutdown sentinel mirrors `core.py:1205-1208` |
| `vllm/v1/engine/core_client.py:467,779-860` (`SyncMPClient`: background pull + blocking `outputs_queue`) | new `src/vllm/v1/engine/core_client.cpp` (`InprocClient` with `get_output()` blocking pop; socket loops collapse into direct queue sharing) | The socket→queue relay threads (`core_client.py:808-846`) are unnecessary in-proc; keep the client API (`add_request_async`/`get_output`/`abort_requests_async` shapes) so a future multi-proc client is a drop-in |
| `vllm/v1/engine/core.py:196-223` (`step_fn` selection, batch queue alloc) | `EngineCoreProc` ctor | `max_concurrent_batches` mirrored from `vllm/config/vllm.py:490-501` |

Leaf W2 — `SERVE-ASYNC-LLM` (async frontend + real streaming surface):

| Upstream | Ours (new/changed) | Notes / deviations |
|---|---|---|
| `vllm/v1/engine/async_llm.py:70-745` | new `src/vllm/v1/engine/async_llm.cpp` + `include/vllm/v1/engine/async_llm.h` (`AsyncLLM`: `add_request` → collector, `generate` pull loop, `abort`, output-handler thread) | DEVIATION: no asyncio — the output handler is a dedicated `std::thread` running U1's `_run_output_handler` body (pull `EngineCoreOutputs` → `output_processor.process_outputs` → abort stop-string finishes); consumers block on the collector |
| `vllm/v1/engine/output_processor.py:45-105` (`RequestOutputCollector`), `:512-573` (queue registration + `STREAM_FINISHED`) | extend `src/vllm/v1/engine/output_processor.cpp` / `.h` (add the collector + per-request queue member currently listed deferred at `output_processor.h:39`) | Single-slot coalescing queue with DELTA merge (`put` merges when consumer lags) — port 1:1 including `get_nowait` fast path |
| `vllm/entrypoints/openai/completion/serving.py:278-310` + `chat_completion/serving.py:404` (stream generators) | rework `src/vllm/entrypoints/openai/serving_completion.cpp` / `serving_chat.cpp`: `create_completion` returns a pull-based chunk source (callback/generator object) instead of a precomputed `sse_chunks` vector (`serving_completion.h:40-46` union changes shape) | Chunk JSON content is already 1:1; only the delivery becomes incremental. Non-streaming path: same handler awaits final output off the collector |
| `api_router.py:66,74` (`StreamingResponse`) | `src/vllm/entrypoints/openai/api_server.cpp:183-206` chunked provider pulls from the live chunk source; drop the engine mutex (`:26,86,130`) — concurrency moves into `AsyncLLM` | httplib's `set_chunked_content_provider` callback blocks per chunk on the collector; client disconnect (`sink.write` false / provider abort) triggers `AsyncLLM::abort` mirroring `async_llm.py:590-596` |
| `vllm/entrypoints/launcher.py` serve wiring | `examples/server/main.cpp:60,121` constructs `AsyncLLM` instead of bare `LLMEngine` | CLI flags unchanged; `--no-async-scheduling`-style flag added in W3 |
| C ABI (original packaging, no upstream) | `include/vllm.h` + `src/capi/vllm_c.cpp`: add handle-based `vllm_request_submit`; a library-owned per-request delivery thread drains that collector and invokes the callback while the shared engine/output handler keep batching. Keep `vllm_complete_stream` (`include/vllm.h:190-194`) as a blocking wrapper over the new path | Decision D1 below; C11 header compile + dlopen tests extended (`tests/capi/test_capi.cpp`, `tests/capi/test_dlopen.cpp`) |

Leaf W3 — `ENG-ASYNC-SCHED` (overlap scheduling, mirror default ON):

| Upstream | Ours (new/changed) | Notes / deviations |
|---|---|---|
| `vllm/v1/core/sched/async_scheduler.py:12-75` | new `src/vllm/v1/core/sched/async_scheduler.cpp` + header; make `Scheduler::update_after_schedule` and the per-request output update protected-virtual in `include/vllm/v1/core/sched/scheduler.h:201-205` (extract `_update_request_with_output` analog from `update_from_output`) | Placeholder counting (`num_output_placeholders` on `vllm/v1/request.py:141` → our `include/vllm/v1/request.h`), `async_tokens_to_discard` stale-frame drop, cache-blocks-minus-placeholders |
| `vllm/v1/engine/core.py:519-611` (`step_with_batch_queue`) | extend `src/vllm/v1/engine/core.cpp` (`EngineCore::step_with_batch_queue`, depth = `max_concurrent_batches` = 2) | Grammar deferral branch (`core.py:559-570`) ported as-is; futures become a small result-slot struct (no `concurrent.futures`) |
| `vllm/v1/worker/gpu/async_utils.py:12-70`; `vllm/v1/outputs.py:298-307` | new `AsyncOutput` in `src/vllm/v1/worker/gpu/runner.cpp`: sampled-ids D2H enqueued on a second `vt::` stream + CUDA event; `get_output()` waits the event | Uses existing vt:: stream/event primitives (`src/vt/cuda/cuda_backend.cu:76-105` graph/stream machinery proves the seam); CPU backend degenerates to sync copy |
| `vllm/v1/worker/gpu/input_batch.py:304-406,457-543`; `model_runner.py:613,954,1423-1445` | extend `src/vllm/v1/worker/gpu/input_batch.cpp` + `runner.cpp`: GPU-resident `last_sampled_tokens`, `combine_sampled_and_draft_tokens`-style gather of next-step decode input ids on device, `post_update` writes on device; delete the host write-back at `runner.cpp:719-730` for the async path | The only device-code delta of this spike: one small combine/post-update kernel pair ported 1:1 from the cited Triton kernels (goldens per test-porting); draft-token lane degenerates to width 1 until `SPEC-MTP` |
| `vllm/config/vllm.py:952,959-1038`; `vllm/config/scheduler.py:158-162,180-189` | extend `include/vllm/config/scheduler.h` / `src/vllm/config/scheduler.cpp` (+ engine wiring): `async_scheduling` tri-state with default-ON resolution and the compat fallbacks relevant to us (pooling n/a, spec-decode method gate recorded for `SPEC-MTP`) | Mirror the log line "Asynchronous scheduling is enabled/disabled" for A/B auditability |

Leaf W4 — `ENG-PRIORITY-SCHED` (priority policy, separable sibling):

| Upstream | Ours (new/changed) | Notes / deviations |
|---|---|---|
| `vllm/v1/core/sched/request_queue.py:131-198,201-208` | extend `src/vllm/v1/core/sched/request_queue.cpp`: `PriorityRequestQueue` (heap by `(priority, arrival_time)`); un-throw `create_request_queue(kPriority)` (`include/vllm/v1/core/sched/request_queue.h:99-100`) | `prepend_request(s)` = heap push (order re-derived), mirroring `:160-173` |
| `vllm/v1/core/sched/scheduler.py:546-556` | `src/vllm/v1/core/sched/scheduler.cpp` preemption: victim = max `(priority, arrival_time)` over running when policy=priority; FCFS tail pop unchanged | `priority` plumbed through `Request`/`EngineCoreRequest` (ours: `include/vllm/v1/request.h`, `src/vllm/v1/engine/types.cpp`) and the OpenAI `priority` request field |
| `vllm/config/scheduler.py:109-117` | scheduler config + server flag | Reject-unknown policy string mirrored |

## Tests to port

Tiers per test-porting.md (T-unit doctest / T-parity / T-e2e). Every port lands
in the same change as its leaf; blocked cases are checked in SKIPPED with the
tracked reason, never dropped.

| Upstream test (pin) | Cases | Ours | Tier | Leaf |
|---|---|---|---|---|
| `tests/v1/engine/test_output_processor.py:1146,1228` | `test_request_output_collector`, `test_cumulative_output_collector_n` (n>1 part SKIPPED: parallel sampling deferred) | `tests/vllm/v1/test_output_processor.cpp` (extend) | T-unit | W2 |
| `tests/v1/engine/test_async_llm.py:109,157,228,306,340,598` | `test_load` (concurrent requests all finish, ids unique), `test_abort`, `test_multi_abort`, `test_finished_flag`, `test_mid_stream_cancellation`, `test_abort_final_output`; SKIP-not-ported: pause/resume (`:683-1030`, admin endpoints out of scope), DP/logger cases (n/a) | new `tests/vllm/v1/test_async_llm.cpp` over the synthetic CPU engine (same seam as `tests/vllm/entrypoints/openai/test_api_server.cpp`) | T-unit | W2 |
| `tests/v1/core/test_async_scheduler.py:34,67,103,139,196,261` | `test_stop_by_max_tokens`, `test_abort`, `test_preempt`, `test_prefix_caching_for_prefill_dedup`, `test_prefix_caching_for_multi_turn`, `test_abort_request_when_structured_output_fsm_cannot_advance` | new `tests/vllm/v1/test_async_scheduler.cpp` | T-unit | W3 |
| `tests/v1/core/test_scheduler.py:2382-2810,2978` | the 11 `test_priority_scheduling_*` cases + `..._preemption_and_resumption_when_out_of_kv`; SKIP: `:3769` EC-connector variant (no connector) | extend `tests/vllm/v1/test_scheduler.cpp` + `tests/vllm/v1/test_request_queue.cpp` | T-unit | W4 |
| `tests/v1/core/test_priority_scheduler_random.py` | randomized heap-property/ordering property test | `tests/vllm/v1/test_request_queue.cpp` (seeded) | T-unit | W4 |
| `tests/entrypoints/openai/completion/test_completion.py:259` (`test_completion_streaming`) and the chat streaming analog | streamed chunks concat to non-stream text AND **chunks arrive incrementally** — new arrival-time assertion: first chunk observed before generation completes, chunk count > 1 spread over the response window (the pre-W2 conformance case could only inspect precomputed frames) | `tests/vllm/entrypoints/openai/test_api_server.cpp` (AsyncLLM harness: live completion/chat arrival/cadence, disconnect abort and concurrent requests); real-model arrival remains G3 | T-unit + T-e2e gate | W2 |
| `tests/benchmarks/test_serve_cli.py:1` | `vllm bench serve` smoke against a live server | DGX smoke in the `SERVE-GATE-ONLINE` harness: run upstream's real `vllm bench serve` client against OUR server (mirror-exact measurement client), plus `tests/examples/test_bench.cpp:15,48` kept green | T-e2e | W2 |
| Local transport-capacity regression (cpp-httplib-specific adaptation; no upstream equivalent because vLLM serves through asyncio) | More simultaneous persistent SSE clients than the library's hardware-concurrency-derived initial pool; every request must be read and produce its first frame without waiting for another stream to finish | `tests/vllm/entrypoints/openai/test_api_server.cpp`: deterministic configured-capacity case, thread/lifecycle return, disconnect abort retained | T-unit + T-e2e gate | W2 repair |
| Token-exactness twins (ours) | existing greedy gates re-run with async ON — async must not change tokens | `tests/parity/test_qwen36_paged_engine.cpp:140`, `tests/parity/test_qwen27_paged_engine.cpp:110` | T-parity | W3 |
| Combine/post-update kernel goldens | goldens dumped from `vllm/v1/worker/gpu/input_batch.py:304-406,457-543` via the LLM-API torch-profiler recipe | `tests/parity/test_op_parity.cpp` op rows (+`PendingRunnerOps` discipline if goldens land first) | T-parity | W3 |

## Gates

Precondition for every leaf: correctness is never traded (16/16 token-exact).
All GPU runs under `flock /tmp/gpu` per the shared skill; A/B series hold one
lock.

| Gate | Command / evidence | Pass bar |
|---|---|---|
| G1 token-exactness (W1-W4) | both greedy engine gates, same binary, async ON vs OFF and priority vs fcfs: `flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'` + same for `qwen27_paged_engine` on DGX | 16/16 token-for-token identical in every mode; a diff = hard fail |
| G2 unit/property suites (per leaf) | `ctest` CPU tier (CI) for the ported modules above | green, skips only with tracked reasons |
| G3 streaming reality + capacity (W2) | conformance arrival-time/persistent-client cases + manual `curl -N`; then upstream `vllm bench serve` against our server at c∈{1,16,32} and the complete standard ladder | TTFT << total latency at c=1; native usage proves exact token count; every one of 32 accepted persistent clients is read and receives a first frame without another request completing; no 19-worker hardware-concurrency cliff or queued socket. Retained ITLs remain inter-choice timings and may be coalesced by the pinned single-slot collector. The binding every-axis campaign is executed with `CLAIM-SERVE-GATE-1` under one whole-series lock |
| G4 no-throughput-regression (W1/W2) | offline gate workloads re-run vs the pre-change binary (same-binary-set A/B, fresh exact matched-config vLLM denominators per benchmark-protocol.md): both 35B and 27B direct-library mirrors | ≥ 1.00× vs our own sync baseline within run-noise and ≥ fresh vLLM on every required axis. Historical 35B 1.02× / 27B 1.007× values are diagnostics only after the sampling/token-budget audit and are not standing floors |
| G5 overlap wins (W3) | same A/B with async ON vs OFF on both gate models, plus the nsys trace-plan capture | async ON ≥ async OFF on throughput axes and ≤ on TTFT/TPOT (mirroring vLLM's own default rationale); GPU-idle gap between steps shrinks in the trace |
| G6 memory | peak RSS/VRAM sampled in the same A/Bs | ≤ sync baseline + bounded queue overhead; the ~25-35%-less-than-vLLM edge not regressed |
| G7 ABI | C11 header compile + `tests/capi/*` incl. dlopen | green; existing symbols unchanged |

Architectures/backends: CUDA (GB10/sm_121) is the gated target; CPU backend
must stay correct (sync degeneration) via the CI tier. No other backend rows
move.

## Dependencies

| Dependency | Kind | Note |
|---|---|---|
| `ENG-CORE-BUSY-LOOP` ← none | row | W1 is pure host-side; first claimable |
| `SERVE-ASYNC-LLM` ← `ENG-CORE-BUSY-LOOP` | row | W2 consumes the queue split; the collector/output-processor part can start in parallel behind the client API |
| `ENG-ASYNC-SCHED` ← `ENG-CORE-BUSY-LOOP` | row | step_with_batch_queue lives in the loop; scheduler subclass + runner AsyncOutput are parallelizable behind it |
| `ENG-PRIORITY-SCHED` ← none | row | fully separable; only merge-order coupling in `scheduler.cpp` |
| `SERVE-GATE-ONLINE` (`CLAIM-SERVE-GATE-1`) | consumer | order-0 gate blocked on W2 for its latency axes; coordinate harness reuse, do not double-own the row |
| `BACKEND-GATE-CUDA-SGLANG` | consumer | already recorded as blocked on `SERVE-ASYNC-LLM` (coordination handoff) |
| `ENG-BATCH-QUEUE` | sibling | W3 ports the depth-2 async subset of `step_with_batch_queue`; the PP-general row stays `INVENTORIED` and inherits W3's code |
| `SPEC-MTP`/`SPEC-GDN-SEGMENTS` | forward-compat | upstream allows async sched only for EAGLE/MTP-family methods (`vllm/config/vllm.py:969-1021`); W3 must land the method gate so C3 composes |
| Hardware | env | GB10 (dgx.casa) for G1/G3-G6; CPU CI for G2/G7. No new models/data/licenses; no new toolchain deps (std::thread only) |

## Work breakdown

Claim-sized, non-overlapping; each row independently gateable:

| W | Row ID | Deliverable | Files (primary) | Gate |
|---|---|---|---|---|
| W1 | `ENG-CORE-BUSY-LOOP` | `EngineCoreProc` thread + input/output queues + in-proc client; sync `LLMEngine` path untouched | new `core_proc.{h,cpp}`, `core_client.{h,cpp}` | G1, G2, G4 |
| W2 | `SERVE-ASYNC-LLM` | `AsyncLLM` + `RequestOutputCollector` + output-handler thread; real SSE emission + engine-mutex removal + disconnect-abort; deterministic HTTP worker capacity for configured concurrent streams; C ABI submit/callback; server wiring | new `async_llm.{h,cpp}`; `output_processor.{h,cpp}`, `serving_completion.*`, `serving_chat.*`, `api_server.{h,cpp}`, `include/vllm.h`, `src/capi/vllm_c.cpp`, `examples/server/main.cpp`; capacity regression in `test_api_server.cpp` | G1, G2, G3, G4, G6, G7 |
| W3 | `ENG-ASYNC-SCHED` | `AsyncScheduler` + virtual hooks; `step_with_batch_queue` depth-2; runner `AsyncOutput` copy-stream + GPU-resident `last_sampled_tokens` + combine kernels; default-ON config resolution | new `async_scheduler.{h,cpp}`; `scheduler.{h,cpp}`, `core.cpp`, `runner.cpp`, `input_batch.cpp`, `config/scheduler.*` | G1, G2, G5, G6 + kernel goldens |
| W4 | `ENG-PRIORITY-SCHED` | `PriorityRequestQueue` + policy config + priority preemption + `priority` request plumbing | `request_queue.{h,cpp}`, `scheduler.cpp`, protocol/serving plumbing | G1, G2 |

Suggested order: W1 → W2 (unblocks roadmap order 0) → W3 (mirror default) →
W4 anytime after W1 merges (or before, at the queue level).

## Risks/decisions

| # | Item | Type | Resolution |
|---|---|---|---|
| D1 | C ABI streaming shape (vLLM defines no C ABI — genuine product call) | decision | Add a handle-based non-blocking surface: `vllm_request_submit` plus a library-owned per-request delivery thread that consumes only that request's collector and invokes the existing per-delta callback; `cancel`/`wait`/`done`/`error`/`free` complete the lifecycle. KEEP `vllm_complete_stream` as a blocking wrapper so LocalAI consumers don't break. The additive 17-symbol set is C11/dlopen/export tested; the engine must outlive its request handles |
| D2 | In-proc thread instead of process split | recorded deviation | Mirrors the EngineCoreProc/MPClient QUEUE semantics without ZMQ; the client API shape is kept so a future multiproc client is additive. Not a behavior change vLLM defines externally |
| D3 | httplib worker-pool blocking on slow SSE clients | confirmed defect / bounded W2 repair | Exact `a531e05` c32 repetitions 1/3 strand one accepted unread socket for 205–207 s. Size the HTTP pool from an explicit configured stream-capacity floor plus bounded control-path headroom (not the library's 19-thread hardware heuristic), preserve disconnect-abort and clean join, expose/validate the server setting if needed for reproducibility, and gate above the initial-pool threshold plus exact c32. Do not rely on opportunistic `idle_thread_count_ == 0` growth |
| D4 | Token-exactness under placeholders | risk | Preemption/stale-frame paths (`async_tokens_to_discard`, placeholder-aware `cache_blocks`) are where async can silently corrupt; covered by the ported preempt/abort async-scheduler tests + G1 on both gate models |
| D5 | CUDA-graph replay vs the new copy stream | risk | The decode graph (`ENG-CUDAGRAPH`, `src/vt/cuda/cuda_backend.cu:76-105`) must not capture the copy stream; AsyncOutput copies run outside capture exactly as upstream keeps them out of graphed regions — verified by G1 with graphs ON |
| D6 | Overlap win smaller in C++ than in Python | risk (must-measure) | vLLM's async default partly compensates Python/GIL host cost; our host step is already cheap, so G5 may show a smaller delta — the gate is "no regression + mirror behavior", the measured delta is recorded in the ledger either way |
| D7 | vLLM-defined behavior not reopened | rule | Queue semantics, placeholder math, priority ordering, SSE frame content: mirrored 1:1 from the cited lines; no local redesign |
