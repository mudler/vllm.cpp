# CPU benchmark frontend wait + thread budget

**Row:** `SERVE-CLI-BENCH` · **issue:**
[#293](https://github.com/mudler/vllm.cpp/issues/293) · **leaf state:** `ACTIVE` ·
**trigger:** Raspberry Pi 5 W0 refresh for `KERNEL-GEMM-CPU-ELEM-A76` at
`3520b7f7f` · **target:** current-source causal control first, C++ only.

## Scope

Determine whether the closed-loop benchmark frontend steals CPU time from the
engine on core-count-constrained hosts, then remove only the proven overhead.
The first binding vehicle is Qwen3.5-2B Q8_K_XL on the four-core Raspberry Pi 5.

In scope:

- a same-current-source control that distinguishes benchmark polling from the
  engine, output-handler and compute-worker thread budget;
- an event-driven "any request has output" wait for the deterministic
  multi-request benchmark loop if polling is proven causal;
- explicit, auditable benchmark output-wait selection and same-binary rollback;
- a CPU compute-thread budget change only if a blocking frontend control shows
  that non-frontend engine threads still require a reserved core;
- focused synthetic tests, QEMU AArch64 build/test, Pi PMU/profile and recursive
  full-model measurement.

Out of scope:

- kernel arithmetic, model/loader, CUDA/Vulkan, C ABI or service changes;
- assembly;
- changing request order, DELTA coalescing, metric definitions, prompts,
  sampling or correctness thresholds;
- compiling on the Pi or treating the old-binary comparison as causal proof.

## Upstream chain

Pinned vLLM `555967922` uses one `asyncio.Event` per
`RequestOutputCollector`: `output_processor.py:45-96`. `put()` sets the event;
`get()` waits; `get_nowait()` is attempted once before awaiting. Its
`AsyncLLM.generate` loop at `async_llm.py:525-587` drains an already-ready
value without a task switch, otherwise suspends at `await q.get()`.

Our collector mirrors that per-request behavior with
`std::condition_variable` in `src/vllm/v1/engine/output_processor.cpp:24-70`.
Single-request `AsyncLLM::generate` already does one nowait attempt then blocks
(`src/vllm/v1/engine/async_llm.cpp:182-240`). The original C++ benchmark needs
deterministic multi-request admission, so `examples/bench/bench_core.h:535-580`
scans all collectors and calls `std::this_thread::yield()` when none is ready.
That final policy is original harness code, not an upstream mirror.

The CPU pool itself mirrors llama.cpp: worker 0 is the dispatching caller and
only workers 1..N-1 are spawned (`cpu_threadpool.cpp:74-87`). Therefore a pool
width of four does use four compute threads, but AsyncLLM and its benchmark
consumer remain additional runnable threads.

## Our baseline

The #284 W0 diagnostic at current source `0e4a1c13d` found:

| Arm | Result | Interpretation |
|---|---|---|
| current benchmark, CPU threads=4 | 5.844 prefill / 1.368 per-stream decode tok/s; 48.70 s E2E; exact tokens | diagnostic only; TTFT spread 15.268% |
| same current binary, CPU threads=3 | 2.53 per-stream decode tok/s; 26.96 s E2E; exact tokens | proves a runnable-thread/core-budget interaction, not which frontend thread causes it |
| old driver at threads=4 | 26.03 s E2E | historical control only; 141 relevant files differ, so not a one-variable arm |
| current matched profile | Barrier 42.21%, reached F16 `Bt16Neon` 33.09%, Q8 12.05% | reranks the BF16 kernel hypothesis; does not isolate polling |

Raw evidence is on the Pi under
`~/vllm-cpp-assembly/evidence/a76-bf16-20260810/`. The old/new comparison may
suggest frontend contention, but only a current-source output-wait control can
establish it.

## Port map

| Upstream / reference | Local surface | Adaptation |
|---|---|---|
| `RequestOutputCollector.ready = asyncio.Event()` and `await ready.wait()` | existing collector `std::condition_variable` | retain per-request semantics unchanged |
| `get_nowait() or await get()` per generator | `AsyncLLM::get_output_nowait/get_output` | direct current-source blocking control at concurrency 1 |
| asyncio task scheduler wakes whichever request event completes | additive AsyncLLM output epoch + condition, or equivalent any-ready wait | one shared wake only for the deterministic C++ multi-request harness; no spin/yield loop |
| vLLM/llama CPU thread-count behavior | `VLLM_CPP_CPU_THREADS` and `Threadpool::Global()` | alter default budget only if event-driven T4 still loses to T3 and matched profiles attribute the residual to noncompute host threads |

## Design and discriminator

1. Add an auditable benchmark output-wait variant. `poll` is the exact current
   scan+yield loop. `blocking-c1` is legal only at concurrency 1 and blocks on
   the sole active collector after the same initial nowait attempt. Both run in
   one binary with identical requests and metrics.
2. Interleave `poll`/`blocking-c1` on the Pi at CPU threads 4 and 3. If
   blocking-c1 restores T4 while T3 stays neutral, polling is causal.
3. Only after that positive discriminator, implement `event`: a monotonic
   output epoch plus condition wait notified after AsyncLLM publishes outputs.
   The benchmark snapshots the epoch before scanning and waits only if the
   epoch is unchanged, preventing the scan-to-wait lost-wakeup race. The wait
   must also wake on engine death/shutdown.
4. If blocking-c1 is neutral, reject the polling hypothesis. Profile
   output-handler/core/compute threads separately; test a reserved-core budget
   without changing frontend policy. Do not implement the event seam merely
   because it is aesthetically preferable.
5. Promote a default only when full-model T4 improves credibly, T3 and x86 do
   not regress, token IDs are identical, and TTFT/ITL measurement remains
   faithful. Keep the old arm as an explicit rollback.

## Tests to port

There is no upstream multi-request C++ benchmark test. Port the load-bearing
upstream event semantics and extend the local original-harness tests:

- collector wait returns only after `put`, propagates errors and never loses a
  wake between nowait and wait;
- `blocking-c1` rejects concurrency greater than one by name;
- `event` completes multiple requests in deterministic submission order,
  preserves every DELTA/token and does not busy-loop while no output exists;
- engine death wakes the wait and throws the existing `EngineDeadError`;
- mutate/remove the epoch increment or notification and prove the bounded test
  fails for the intended timeout/unfinished-output reason;
- `poll`, `blocking-c1` and `event` produce byte-identical token-ID fixtures.

## Gates

1. RED-first focused test/mutation, then the minimum current-source control.
2. `test_bench`, `test_async_llm`, `test_output_processor`,
   `test_engine_core_proc` and threadpool tests all green; bounded timeouts must
   diagnose a missing wake rather than hang.
3. Build AArch64 Release locally through `docker/Dockerfile.arm64` using
   buildx/QEMU; run focused tests under QEMU. QEMU timings are void.
4. Deploy hash-identified artifacts to the user-authorized disposable Pi.
   Record CPU affinity, governor, frequency, temperature, throttle, contenders
   and PMU lost-sample count for each interleaved arm.
5. Correctness: exact token IDs across every same-binary arm and against the
   accepted current baseline.
6. Performance: three or more interleaved clean reps, sub-3% spread for a
   binding claim. First close the current-main regression; then resume the
   same-file llama.cpp floor. No micro/harness-only win earns model credit.
7. Native non-A76 CPU gate and all record/ready gates pass.

## Dependencies

- `SERVE-CLI-BENCH`, `SERVE-ASYNC-LLM`, `ENG-CORE-BUSY-LOOP` and
  `QUANT-GGUF-CPU-THREADPOOL` are existing prerequisites.
- Issue #284 W0 commit `3520b7f7f` supplies the diagnostic, after its review
  findings narrow the causal wording.
- Official llama.cpp b9892 `ee445f93d` is the speed denominator; pinned vLLM
  supplies collector behavior.
- The model already exists on the Pi. No download or service change is owed.

## Work breakdown

| W | Deliverable | State |
|---|---|---|
| C0 | Correct #284 evidence wording and register #293 | complete (`0a07c53db`) |
| C1 | Same-current-source `poll` vs `blocking-c1` control + RED mutation | reviewed and current-main operator-gated at `960647bf5` |
| C2 | If C1 positive, general event/epoch wait + deterministic multi-request tests | unblocked by positive C1; excluded from this implementation |
| C3 | QEMU gate and interleaved Pi T4/T3 causal profile | current-main confirmed ([operator evidence](../../docs/bench-evidence/rpi5-a76-output-wait-c1-main-20260811.md); [original evidence](../../docs/bench-evidence/rpi5-a76-output-wait-c1-20260810.md)) |
| C4 | Retain event wait or reject it; test reserved-core budget only if C1 is negative/insufficient | blocked by C2 |
| C5 | Full model versus llama.cpp; return to the measured F16/kernel ranking | blocked by C4 |

## Risks/decisions

- Blocking an arbitrary request at concurrency greater than one can delay an
  already-ready peer and distort ITL. That is why `blocking-c1` is a causal
  control, not the general implementation.
- A global condition without an epoch has a lost-wakeup race. The mutation gate
  must cover the scan-to-wait boundary.
- Yielding is not sleeping; on four runnable compute threads it may consume a
  timeslice despite making scheduler progress. This remains a hypothesis until
  C1.
- Reducing compute workers can improve an oversubscribed benchmark while
  lowering a properly isolated kernel ceiling. Thread-budget and wait-policy
  candidates remain separate experiments.
- Any remaining F16 GEMM optimization belongs to the A76 elementwise row only
  after this harness/control issue is resolved and a fresh profile ranks it.

## Outcome

C1 result: **positive, reviewed and operator-gated on current main**. The
rebased `960647bf5` AArch64 binary reproduces the discriminator over three
interleaved idle/unthrottled processes per arm: T4 `blocking-c1` / `poll`
medians are 2.149x decode, 1.973x prefill and 0.465x E2E latency, while the T3
control is neutral at 1.003x decode, 0.999x prefill and 0.998x E2E latency.
All 12 token files exactly match the accepted SHA-256; RSS is neutral. T4
blocking retires 63.0% fewer instructions and uses 53.4% less task-clock. This
confirms scan-and-yield frontend contention as the cause of the four-core
collapse after the upstream rebase.

No replacement performance floor is accepted here: the pathological T4 poll
arm has 13.33-15.96% spread and fails the sub-3% binding gate, although the
effect is positive in every pair and the blocking/T3 arms are stable. The
older source/binary evidence remains preserved rather than being overwritten.
`blocking-c1` remains an explicit concurrency-one control and `poll` remains
the default. C2's lost-wakeup-safe general event wait requires a fresh
implementer; it was deliberately not started. Full latest-main hashes, native
and QEMU counts, host state, raw evidence and medians are in the
[operator checkpoint](../../docs/bench-evidence/rpi5-a76-output-wait-c1-main-20260811.md);
the [original C1 evidence](../../docs/bench-evidence/rpi5-a76-output-wait-c1-20260810.md)
retains its historical provenance.
