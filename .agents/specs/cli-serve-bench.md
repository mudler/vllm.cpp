# CLI serving/benchmark parity — structured spike

Stable row: `SERVE-CLI-BENCH`.  This spike covers only the benchmark half of
the row: making the local closed-loop client exercise the same production
engine mode as the pinned vLLM denominator.  The server CLI breadth remains the
existing `PARTIAL` residual.

## Problem and measured discriminator

`vllm-bench` constructs `LoadedEngine`, whose default configuration correctly
resolves asynchronous scheduling ON and `max_concurrent_batches=2`, but then
calls `loaded->engine()` and drives `LLMEngine::step()`
(`examples/bench/bench_core.h:485-512`).  That path calls
`EngineCore::step()` directly (`src/vllm/v1/engine/llm_engine.cpp:141-143`) and
therefore never selects `EngineCore::step_with_batch_queue()`.

The pinned vLLM engine makes the dispatch in its engine core: it sizes the
batch queue from `max_concurrent_batches` and selects
`step_with_batch_queue` whenever the queue exists
(`${VLLM_SOURCE}/vllm/v1/engine/core.py:200-231`).  The queued path
schedules a new batch before consuming the oldest result
(`${VLLM_SOURCE}/vllm/v1/engine/core.py:622-669`).  Our production
equivalent is already implemented behind `LoadedEngine::async_engine()`:
`AsyncLLM` constructs `EngineCoreProc` with the resolved queue depth
(`src/vllm/entrypoints/model_loader.cpp::async_engine`), and `EngineCoreProc` selects
`step_with_batch_queue` at depth greater than one
(`src/vllm/v1/engine/core_proc.cpp:22-37`).

The 2026-08-07 fresh RTX 5070 Ti run at `7ef5f1001` exposed the consequence.
The exact c32 workload completed the same 148,168 tokens in about 22.2 seconds
on both engines, but same-tool graph-node traces showed different sampling
waves: ours ran 529 sample steps, overwhelmingly batch 32; vLLM ran 650 steps
with a mean sampled batch near 25.3.  Total steady GPU-kernel time was almost
equal (about 21.7 versus 22.0 seconds), while our client-observed mean TPOT was
38.133 ms versus 33.916 ms.  These numbers are a structural discriminator, not
a binding parity result, because the frontends were mismatched.

## Upstream and dependency chain

| Layer | Pinned vLLM / dependency behavior | Our anchor |
|---|---|---|
| CLI/client | `LLM.llm_engine.add_request` + `engine.step`, DELTA output (`tools/bench/vllm_closed_loop_metrics.py:59-102`) | closed-loop admission and metrics (`examples/bench/bench_core.h:416-589`) |
| engine dispatch | queue size and `step_fn` selection (`vllm/v1/engine/core.py:200-231`) | `LoadedEngine::async_engine` passes resolved depth (`src/vllm/entrypoints/model_loader.cpp::async_engine`) |
| batch queue | schedule-before-oldest-result (`vllm/v1/engine/core.py:622-669`) | `EngineCore::step_with_batch_queue` (`src/vllm/v1/engine/core.cpp:115-185`) |
| scheduler | `AsyncScheduler` placeholder accounting (`vllm/v1/core/sched/async_scheduler.py`) | `src/vllm/v1/core/sched/async_scheduler.cpp` |
| runner/sample | non-blocking sampled-token copy and device input update (`vllm/v1/worker/gpu/async_utils.py`, `gpu_model_runner.py`) | `GPUModelRunner::sample_tokens_async` and device mirror (`src/vllm/v1/worker/gpu/runner.cpp`) |
| CUDA graph | graph replay is selected inside the runner; tracing must expose child nodes | existing `--cuda-graph-trace=node` paired `nsys` recipe |

No new kernel is proposed.  CUTLASS, cuBLASLt, FlashAttention and GDN calls
must remain byte-identical; their aggregate and per-template times are the
post-change structural control.

## Dispatch and implementation

1. `RunBench` obtains `loaded->async_engine()` for every non-legacy benchmark
   run.  This keeps async-scheduling OFF meaningful: the same frontend then
   drives a depth-1 `EngineCoreProc`, matching vLLM's `async_scheduling=False`
   control rather than silently switching frontend APIs.
2. Admission stays deterministic and sequential.  Keep a request-id-ordered map
   of `AsyncRequest` collectors, scan them with `get_output_nowait()`, process
   every ready DELTA, and refill immediately after terminal outputs.  Yield only
   when no collector is ready.  Do not introduce one client thread per request;
   racing initial submissions would change batch composition and invalidate the
   same-workload claim.
3. Report the selected frontend, resolved async flag and batch-queue depth in
   `BenchResult`/CLI output so a benchmark artifact proves which path ran.
4. Retain the existing synchronous `LLMEngine` as a library API and test seam;
   only the comparison harness changes.

## Files and tests

- Modify `examples/bench/bench_core.h` and `examples/bench/main.cpp`.
- Extend `tests/examples/test_bench.cpp` to assert the async frontend was
  exercised, all requests/tokens remain complete, and serialized output order
  remains submission order.
- Reuse the production depth-2 correctness gates
  `tests/parity/test_qwen3_dense_async_serving.cpp` and
  `tests/vllm/v1/test_engine_core_proc.cpp`; no upstream test module directly
  covers this original C++ benchmark, so there is no omitted upstream test to
  port.
- Re-run `tests/vllm/models/test_qwen35_plain_weights.cpp` on the cached 4B
  model before timing.

RED-first mutation: replace `loaded->async_engine()` with `loaded->engine()` or
force the reported frontend false; the new benchmark contract test must fail.

## Gates and hardware

CPU gate: focused benchmark, async engine/core, scheduler and output-processor
tests, followed by the staged protocol gates.  CUDA correctness: cached
Qwen3.5-4B plain-weight test, direct ON/OFF output IDs 128/128 per repetition,
and no new sanitizer finding where sanitizer support exists.

Performance gate on the local RTX 5070 Ti (`sm_120`): hold `${GPU_LOCK}` across
the entire interleaved direct-ON / pinned-vLLM / direct-OFF series; run under the
validated user-systemd scope (`MemoryHigh=22G`, `MemoryMax=25G`, swap disabled);
three memory and three timed repetitions per arm; exact cached model, ShareGPT
digest, 128 requests, 128 output tokens, c32, 2048 batched-token cap, greedy.
Trace both engines with the same `nsys` and `--cuda-graph-trace=node`.

Acceptance is token correctness plus no regression on any axis. The corrected
frontend materially changed the queue composition, so its production run
supersedes the diagnostic TPOT target. Paired same-tool traces selected batched
GDN prefill causal-conv total GPU time; the implemented exact-chunk lever and
its enclosing result are in
[the sm_120 chunking spike](sm120-qwen35-conv-chunking-2026-08-07.md).
Aggregate throughput, TTFT, TPOT/ITL, peak/stable PSS and VRAM remain guards.

## Dependencies, risks, rollback, work breakdown

Dependencies are already landed: `SERVE-ASYNC-LLM`, `ENG-CORE-BUSY-LOOP` and
`ENG-ASYNC-SCHED`.  The cached model and pinned oracle are available locally.

Risks: collector DELTA coalescing could distort ITL if the client fails to drain
promptly; the gate therefore checks output chunking and TPOT independently.
Busy polling could contend with the engine thread; the loop yields on no work
and CPU utilization is recorded.  Async production correctness is protected by
the existing token-exact tests.  Rollback is the single harness dispatch change;
the synchronous engine remains untouched.

| Work item | Lifecycle / result |
|---|---|
| B0 fresh-main build, correctness, 18-leg diagnostic and paired trace | complete; non-binding because frontend mismatch discovered |
| B1 async-front-end benchmark repair + RED-first contract test | complete: `RunBench` uses `AsyncLLM`; report records frontend/flag/depth |
| B2 contained rebuild and focused correctness | complete: RED compile failure; GREEN build; benchmark 4/4·35, async 8/8·323, core 10/10·95, cached 4B 3/3·1672; real-model async smoke |
| B3 exact interleaved rerun + same-tool traces | complete: production c32 baseline bound; causal-conv 720.954/145.421 ms selected |
| B4 choose/implement first measured lever | complete and current-main revalidated: exact chunks 3.073x kernel / +2.121% profiled enclosing; three-pair binding A/B +2.152% |
