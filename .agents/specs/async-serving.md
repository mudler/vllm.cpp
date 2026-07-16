# Async serving + overlap scheduling — joint spike

Rows covered (engine-matrix): `SERVE-ASYNC-LLM` (AsyncLLM-equivalent streaming
engine API), `ENG-CORE-BUSY-LOOP` (engine busy loop + input/output queue
split), `ENG-ASYNC-SCHED` (async/overlap scheduling), `ENG-PRIORITY-SCHED`
(priority scheduling). Roadmap block: `ROAD-V1-C6`; blocking dependency of
`ROAD-V1-A` (see priority justification). Upstream pin:
`/home/mudler/_git/vllm` @ `e24d1b24`.

## Current checkpoint and priority

W1 `ENG-CORE-BUSY-LOOP`, W2 `SERVE-ASYNC-LLM`, and W4
`ENG-PRIORITY-SCHED` are implemented and `GATING`; the engine matrix carries
their exact code/test evidence. Fixed HTTP delivery capacity is retained and
steady-state neutral.

**W3 `ENG-ASYNC-SCHED` host-side machinery LANDED + CPU-gated (2026-07-16),
state `ACTIVE`.** The scheduler + engine-loop half of overlap scheduling is
implemented and CPU-tested RED-first: `AsyncScheduler` output-placeholder
accounting (`src/vllm/v1/core/sched/async_scheduler.cpp`, 1:1 with
`async_scheduler.py:12-75`) over the base scheduler's now-placeholder-aware
running-loop budget/`max_tokens` guard + `is_prefill_chunk` + extracted
protected-virtual `update_after_schedule`/`update_request_with_output` hooks
(all INERT while placeholders == 0, so the synchronous path is byte-identical);
`EngineCore::step_with_batch_queue` depth-2 (`core.py:519-632`) + the
`EngineCoreProc` batch-queue/step_fn selection + `has_work()` batch-queue term;
`SchedulerConfig::ResolveAsyncScheduling` default-ON-when-compatible resolution +
`MaxConcurrentBatches` (`vllm/config/vllm.py:490-501,990-1038`); and the
`VT_ASYNC_SCHED` process rollback env. Token-exactness holds by construction on
the CPU path (the eager executor materializes each batch's tokens before the
next is scheduled; the placeholder math reconciles one step later, exactly as
upstream).

**W3 runner device-input half LANDED + CPU-gated (2026-07-16), `ACTIVE`.** The
input half of the runner integration now lands: `combine_sampled_and_draft_tokens`
(`src/vllm/v1/worker/gpu/prepare_inputs.cpp`, 1:1 with `input_batch.py:304-406`
T0 non-spec subset) rebuilds each decode row's input token id from the
GPU-resident-analog `InputBatch::last_sampled_tokens` (per-req_state, seeded on
admission per `states.py:105-122`, threaded through condense/swap/add) instead of
the host `token_ids_cpu` read `prepare_inputs` used — placeholder/prefill-aware
(`seq_len <= prefill_len` rows keep the prompt), `idx_mapping` indirection for the
abort/finish reorder, and run on the HOST side of input prep BEFORE the forward
and OUTSIDE any CUDA-graph capture (capture-safe by construction; input prep
always precedes the decode graph replay — checked both sides). It is device-neutral
(the DGX leaf ports the loop to the cited Triton kernel over GPU-resident
`input_ids`/`last_sampled_tokens`). Gated behind `VT_ASYNC_RUNNER` /
`GPUModelRunner::set_async_input_combine`, **default OFF**. Tested RED→GREEN on
CPU: `tests/vllm/v1/worker/test_combine_tokens.cpp` (7 cases — pure decode,
reads-`last_sampled`-not-host, prefill/chunked-prefill-transition boundary, mixed
batch, `idx_mapping` indirection, draft-only; RED verified by splicing the stale
value → 5/7 fail), the `last_sampled_tokens`/`prefill_len` batch-lifecycle in
`test_input_batch.cpp`, and a runner token-exactness async-ON≡sync case + a
reads-`last_sampled`-over-a-corrupted-host case in `test_runner.cpp`; full CPU
ctest 110/110, tools 164/164, clean `-Werror` rebuild.

**W3 runner sampler-OUTPUT half LANDED + CPU-gated (2026-07-16), `ACTIVE`.** The
sampler-output overlap machinery now lands, mirroring `async_utils.py:12-70` /
`gpu_model_runner.py:242-332` 1:1, CPU test-first:
- **`vt::Backend` event + pinned-host primitives** (`include/vt/backend.h`,
  `src/vt/backend.cpp` base impls, `src/vt/cpu/cpu_backend.cpp` inherit,
  `src/vt/cuda/cuda_backend.cu` overrides): `AllocPinned`/`FreePinned`
  (cudaHostAlloc) + `CreateEvent`/`DestroyEvent`/`RecordEvent`/`SynchronizeEvent`/
  `QueueWaitEvent` (cudaEvent + cudaStreamWaitEvent). Synchronous CPU degeneration
  (null-handle events, host alloc) is the exact contract the CUDA override honours;
  contract-tested in `tests/vt/test_backend.cpp`.
- **`AsyncGPUModelRunnerOutput`** (`include/vllm/v1/worker/gpu/async_output.{h,cpp}`,
  1:1 with `AsyncOutput.__init__`/`get_output`): owns the device-resident sampled-id
  snapshot; the ctor makes the COPY queue wait the MAIN queue, issues the
  non-blocking D2H into an owned pinned host buffer, and records the ready event —
  the MAIN queue is NEVER synchronized. `get_output()` blocks only that event and
  materializes the `ModelRunnerOutput`. `ReadyModelRunnerOutput` is the
  already-materialized sync degeneration.
- **`Sampler::forward(..., sampled_ids_out)`** (`sampler.{h,cpp}`): device-resident
  greedy fast path — writes ids straight into the caller's device tensor via
  `GreedyArgmax`, NO host download / NO main-queue sync (`sampled_on_device`);
  default `nullptr` keeps the sync path byte-identical.
- **`GPUModelRunner::sample_tokens_async` + `runner_supports_async`** (`runner.{h,cpp}`):
  the overlap variant produces ids device-resident, does the on-GPU post_update
  (`last_sampled` scatter) with the host token write-back DELETED on the async path,
  and returns the `AsyncGPUModelRunnerOutput` after issuing the D2H on the lazily-
  created copy queue. `runner_supports_async()` advertises the compat gate
  (`async_input_combine_`, from `VT_ASYNC_RUNNER`).
- **Executor + depth-2 seam** (`executor.{h,cpp}`, `model_runner_base.h`, `core.{h,cpp}`):
  `sample_tokens_async` threads through `Executor` and `EngineCore::step_with_batch_queue`;
  `BatchQueueItem` holds an `AsyncModelRunnerOutput` whose `get_output()` is resolved
  at CONSUME time (`update_from_output`) — off the model's critical path, so the copy
  overlaps the next step's forward. Validated by the depth-2 cycle
  `test_engine_core_proc.cpp:479` (mcb=2) with a synchronous runner double.

Everything is behind `VT_ASYNC_RUNNER` / `set_async_input_combine`, **default OFF**:
`runner_supports_async()` returns false and `sample_tokens_async` degenerates to the
byte-identical sync `sample_tokens`, so production is unchanged. Tested RED→GREEN on
CPU (RED = splice a stale value into `AsyncGPUModelRunnerOutput::get_output` → the
async-output unit cases + the runner async-ON≡sync decode both fail): full CPU ctest
**111/111**, tools **164/164**, clean `-Werror` rebuild 0 warnings.

**Remaining W3 leaf (the DGX enable-flip): production depth-2 engagement + the
device combine kernel.** Two pieces stay DGX-gated because they change production
GPU behavior that cannot be CPU-validated: (1) wire `LoadedEngine` to construct an
`AsyncScheduler` + pass `max_concurrent_batches=2` to the async engine when
`ResolveAsyncScheduling(runner_.runner_supports_async())` resolves ON (member
reorder so `runner_` precedes the scheduler; the seam through `step_with_batch_queue`
is already wired + CPU-tested), and (2) port the input-half `combine` + the
`last_sampled` scatter loops to a device kernel over GPU-resident tensors for
discrete backends (the host-array form is correct on the CPU + GB10-unified gate
backends). `VT_ASYNC_RUNNER=1` is the single env the DGX A/B flips to engage full W3
(async input-combine + async output + `runner_supports_async` TRUE → depth-2);
`VT_ASYNC_SCHED=0` is the same-binary rollback. The joint spike's speed hypothesis is
closed below.

W3 is an unmet mirror obligation whose speed hypothesis is now closed. vLLM
resolves `async_scheduling=None` to **True** when compatible
(`vllm/config/vllm.py:990-1038`), and the binding v0.25.0 server log confirms
that resolution. The 27B gate has better local TTFT at c2 but local TPOT is
**114.841 vs 108.274 ms** (**6.1% slower**). The post-W3-I scan selected an
exact vLLM async ON/OFF c2 timing + Torch-trace control on the binding corpus.
That control is complete at accepted root `3812d8`: ON/OFF total throughput is
**1.002153×**, direction-normalized TPOT is **1.010291×**, TTFT is
**0.862159×**, and aggregate traced GPU time is **1.002004×**. The tactic/batch
mix changes, but total work does not improve; async-ON digests are stable while
OFF varies across default batch shapes. This is neutral against the
predeclared 1.04× speed-credit floor. W3 stays later parity work, `READY` and
unclaimed, while the immediate speed track returns to low-batch kernel mapping.

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

**Runtime trace result** (dispatch is dynamic): accepted `3812d8` runs the c2
binding corpus with vLLM async explicitly ON/OFF and captures both arms through
`tools/bench/profile_vllm_online_gate.py --async-scheduling on|off
--num-prompts 6 --max-concurrency 2`. vLLM's Torch-profiler path is the GB10
alternative where nsys breaks EngineCore startup. It freezes W3's maximum c2
speed credit at **1.002153×** total with no aggregate GPU-time reduction, so no
c4/c8 speed promotion follows. When W3 is later claimed for parity, retain the
default-ON trace as its execution target and repeat correctness/every-axis
gates without treating this neutral control as implementation evidence.

## Our baseline and current W3 delta

| Area | Current local behavior | Missing mirror behavior |
|---|---|---|
| Engine core | `EngineCore::step()` synchronously schedules, executes, samples, then updates (`src/vllm/v1/engine/core.cpp:37-85`); `EngineCoreProc` constructs one concurrent batch and rejects depth >1 (`src/vllm/v1/engine/core_proc.cpp:18-35`) | `step_with_batch_queue` depth 2, schedule N+1 before consuming N |
| Scheduler | Output placeholders are fixed at zero (`src/vllm/v1/core/sched/scheduler.cpp:143-145,587-600`) | `AsyncScheduler` placeholder/stale-frame/cache accounting |
| Sampled token | Sync path unchanged. Async path (`sample_tokens_async`): `Sampler::forward(sampled_ids_out)` writes ids device-resident (greedy `GreedyArgmax`, no download/sync), `AsyncGPUModelRunnerOutput` D2Hs on a copy queue + event, on-GPU post_update scatters `last_sampled` (host write-back deleted) — `src/vllm/v1/sample/sampler.cpp`, `src/vllm/v1/worker/gpu/async_output.cpp`, `runner.cpp::sample_tokens_async` | REMAINING (DGX device kernel): port the `last_sampled` scatter + input-half combine to a device kernel for discrete backends (host-array form is correct on CPU + GB10-unified) |
| Runtime seam | LANDED (2026-07-16): `vt::Backend` now exposes `AllocPinned`/`FreePinned` + `CreateEvent`/`DestroyEvent`/`RecordEvent`/`SynchronizeEvent`/`QueueWaitEvent` (`include/vt/backend.h`, base sync-degeneration `src/vt/backend.cpp`, CUDA `src/vt/cuda/cuda_backend.cu`); contract-tested `tests/vt/test_backend.cpp` | — |
| Diagnostic | `tools/bench/profile_vllm_online_gate.py` can force and record oracle async mode; `tools/bench/finalize_async_credit.py` validates six legs/two traces and writes the hashed marker last; CPU contracts are green | accepted `3812d8` completes timing + traces at **1.002153×** total and **1.002004×** traced GPU time; neutral for speed, digest/batch-shape sensitivity explicit, no W3 implementation claim |

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
| `vllm/v1/worker/gpu/async_utils.py:12-70`; `vllm/v1/outputs.py:298-307` | **LANDED (2026-07-16):** `vt::Backend` event/pinned primitives (`include/vt/backend.h`, `src/vt/backend.cpp`, `src/vt/cuda/cuda_backend.cu`); `AsyncGPUModelRunnerOutput` + `AsyncModelRunnerOutput`/`ReadyModelRunnerOutput` (`include/vllm/v1/worker/gpu/async_output.{h,cpp}`) with the sampled-id D2H on the copy queue + `get_output()` waiting only its event; `GPUModelRunner::sample_tokens_async` (`runner.cpp`); the `Executor`+`step_with_batch_queue` seam (`executor.{h,cpp}`, `core.{h,cpp}`, `model_runner_base.h`) resolving `get_output()` at consume time | CUDA implements the primitives; CPU degenerates to synchronous copy. The copy queue waits the main queue/event and stays outside graph capture. Contract + unit + depth-2 CPU tests: `tests/vt/test_backend.cpp`, `tests/vllm/v1/worker/test_async_output.cpp`, `test_runner.cpp` (async-ON≡sync), `test_engine_core_proc.cpp:479` |
| `vllm/v1/worker/gpu/input_batch.py:304-406,457-543`; `model_runner.py:613,954,1423-1445` | **INPUT HALF LANDED (2026-07-16):** `combine_sampled_and_draft_tokens` device-neutral port (`src/vllm/v1/worker/gpu/prepare_inputs.cpp`) + `last_sampled_tokens`/`prefill_len` on `InputBatch` (`input_batch.cpp`, seed+condense+swap) + `post_update`-style `last_sampled` write & combine wiring on the runner (`runner.cpp`, behind `VT_ASYNC_RUNNER`, default OFF). **OUTPUT HALF also landed (2026-07-16):** `sample_tokens_async` now does the on-GPU post_update (`last_sampled` scatter) and DELETES the host token write-back on the async path. REMAINING: the DGX device kernel (scatter + combine loop → Triton) for discrete backends | The device-code delta of this spike: the combine/post-update pair. Landed as the device-neutral CPU logic (host arrays; the DGX leaf ports the loop to the cited Triton kernel over GPU-resident tensors). draft-token lane degenerates to width 1 until `SPEC-MTP`; capture-safe host-side prep (checked both sides) |
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
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | consumer | the distinct shared-prefix cache-on gate also needs honest incremental TTFT/ITL after its PX1/PX2 cache work; do not blend it with the cache-neutral row |
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
| D3 | httplib worker-pool blocking on slow SSE clients | implemented; GPU-classified, broader row gating | Exact `a531e05` c32 repetitions 1/3 strand one unread socket for 205–207 s. Production fixes the pool at `max_num_seqs + 4`, logs the resolved count, and retains `VLLM_CPP_HTTP_FIXED_POOL=0` as a diagnostic fallback. The 32-client/control-reserve and sanitizer gates pass. Same-binary c32 fixed/legacy is healthy and steady-state-neutral at 0.999764×; neither bounded arm samples the rare tail. Exact `4e1d8ca` fixed c32 is healthy 3/3. The capacity claim is released without claiming a speed win or whole-row closure |
| D4 | Token-exactness under placeholders | risk | Preemption/stale-frame paths (`async_tokens_to_discard`, placeholder-aware `cache_blocks`) are where async can silently corrupt; covered by the ported preempt/abort async-scheduler tests + G1 on both gate models |
| D5 | CUDA-graph replay vs the new copy stream | risk | The decode graph (`ENG-CUDAGRAPH`, `src/vt/cuda/cuda_backend.cu:76-105`) must not capture the copy stream; AsyncOutput copies run outside capture exactly as upstream keeps them out of graphed regions — verified by G1 with graphs ON |
| D6 | Overlap win smaller in C++ than in Python | risk (must-measure) | vLLM's async default partly compensates Python/GIL host cost; our host step is already cheap, so G5 may show a smaller delta — the gate is "no regression + mirror behavior", the measured delta is recorded in the ledger either way |
| D7 | vLLM-defined behavior not reopened | rule | Queue semantics, placeholder math, priority ordering, SSE frame content: mirrored 1:1 from the cited lines; no local redesign |
