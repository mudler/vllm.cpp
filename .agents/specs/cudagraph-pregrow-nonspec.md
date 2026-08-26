# ENG-CUDAGRAPH-BREAK — the capture pre-grow runs only when the double buffer or a speculative step turns it on, so the DEFAULT server captures against an unprepared pool ([#2029](https://github.com/mudler/vllm.cpp/issues/2029))

Row: `ENG-CUDAGRAPH-BREAK` ([`.agents/engine-matrix.md`](../engine-matrix.md)).
Issue: [#2029](https://github.com/mudler/vllm.cpp/issues/2029).
Parent wave: [`eng-cudagraph-break.md`](eng-cudagraph-break.md), whose #1380 fix
([`7dec1d990`](https://github.com/mudler/vllm.cpp/commit/7dec1d990)) built the
pre-grow this wave makes reachable.

## The finding

With DFlash2 speculation **off**, the engine dies at concurrency 8:

```
engine-fatal: EngineCore busy loop threw: vt cuda: cudaMalloc: operation not permitted when stream is capturing
```

and every later request returns 500 `[request submitted to a stopped AsyncLLM]`.
The same binary with `--speculative-config` warms and serves c=1/2/4 cleanly, so
#2029 records that "the failing allocation is on a path taken only when
speculation is off". That is exactly true, and the path is the **absence** of the
#1380 capture pre-grow.

`Qwen3_5DenseDecodeGraph::Step` and `Qwen3_5DecodeGraph::Step` both compute

```
const bool dbuf = impl_->dbuf || spec_step;      // qwen3_5.cpp:11315, :10764
```

with `impl_->dbuf = enabled && DecodeGraphDoubleBufferEnabled()` (`:11082`,
`:10529`) and `DecodeGraphDoubleBufferEnabled()` returning false unless
`VT_ASYNC_EXECUTOR=1` (`:10302-10308`), and
`spec_step = gdn_meta.num_spec_decodes > 0` (`:11267`, `:10715`).

The pre-grow sits **inside** `if (dbuf)`:

| Driver | `if (dbuf)` | `PreGrowForCapture` |
|---|---|---|
| `Qwen3_5DecodeGraph` (35B MoE) | `qwen3_5.cpp:10885` | `qwen3_5.cpp:10907` |
| `Qwen3_5DenseDecodeGraph` (27B dense) | `qwen3_5.cpp:11439` | `qwen3_5.cpp:11461` |

The 27B dense driver is the one #2029 measured. The #1574 subject is a Qwen3.8-27B
checkpoint, which declares `Qwen3_5ForConditionalGeneration` and registers onto
`kQwen3_5DenseFactory` (`src/vllm/model_executor/models/qwen3_5_dense.cpp:283`, `:296`,
`:259-266`); its decode dispatch is `Qwen3_5DenseDecodeGraph::Step` at
`qwen3_5_dense.cpp:223-232`. #1380's own repro was this same driver reporting this same
message.

So on the **default** server — no `VT_ASYNC_EXECUTOR`, no speculation — `dbuf`
is false, the pre-grow never executes, and the driver opens
`vt::GraphCaptureScope` over a `DevicePool` nobody prepared. A `Get` miss inside
that region is `DevicePool::Get`'s `b.Alloc(key)`
(`include/vllm/model_executor/models/device_pool.h:212`), which on CUDA is a raw
`cudaMalloc` (`src/vt/cuda/cuda_backend.cu`) and produces exactly the message
#2029 reports. #1380 named the same message from the same driver.

The dependence on batch size follows: each captured `SizeSlot` retains its
`[S, vocab]` f32 logits and its `[S, H]` bf16 hidden for the life of the process,
and `DevicePool` is keyed by SIZE CLASS rather than by tensor, so every
additional padded size a ramping server captures removes blocks from classes a
later capture needs. At c=1/2 the free list happens to be deep enough; at c=8 it
is not. That is #1380's mechanism, one axis over.

`s.demand` — the cold step's per-class peak — is recorded **unconditionally**
(`:11607`, `:11046`), and `Pool(b).MarkStepBoundary()` runs unconditionally at
the top of both `Step`s (`:11251`, `:10697`). Only the consumer is gated. The
gate is an accident of history: the block that now holds the pre-grow was
introduced for the `VT_ASYNC_EXECUTOR` parity ring, and #1393 replaced the
single-block pre-grow in place without revisiting the guard it sat under.

## Scope

**In.** Make the capture pre-grow run before **every** capture in both Qwen3.5
decode-graph drivers, and gate the guarantee it delivers — that the captured
region performs no driver allocation — on a non-speculative step with the double
buffer off.

**Out.** [#2028](https://github.com/mudler/vllm.cpp/issues/2028) (the illegal
memory access with speculation ON), [#2007](https://github.com/mudler/vllm.cpp/issues/2007)
(two pools), the `P == 1` capture gate at `qwen3_dflash.cpp:1577`, the missing
backtrace at the fatal handler ([#2037](https://github.com/mudler/vllm.cpp/issues/2037)),
and the seven other decode drivers that have no pre-grow at all
([#2035](https://github.com/mudler/vllm.cpp/issues/2035)). Each is named under `## Owed`.

## Upstream

vLLM has no analogue to port. Its decode graphs come from `torch.compile` and its
activations from torch's caching allocator, which is warmed by
`capture_model()`'s dummy runs over every `cudagraph_capture_sizes` bucket before
any capture (`vllm/v1/worker/gpu_model_runner.py::_dummy_run` /
`capture_model` @ pin `5559679229`). The warm-then-capture ordering this tree's
cold step implements is that shape; the pre-grow is the part torch's allocator
does for free and ours does not, because our pool is class-keyed with no split
and no per-block best fit over the whole cache. So this is a repair of our own
seam, not a parity gap, and nothing about the pool's design changes here.

## Design

One statement moves in each driver: `Pool(b).PreGrowForCapture(b, s.demand);`
leaves the `if (dbuf)` block and runs immediately before it, on the `s.warm`
capture path.

Everything else stays inside `if (dbuf)`: `b.Synchronize` drains an in-flight
replay that only the parity ring can leave behind, and the `StepDevInputs` /
`s.pin` construction is the persistent-input path the ring and the spec capture
need. On the `!dbuf` arm `DenseForwardLayers` is called with
`persistent_sdi == nullptr` (`:11531`, `:10974`), so it builds its step inputs
from the main pool **inside** the captured region — which is why that arm needs
the pre-grow at least as much as the `dbuf` arm, and why the demand profile is
exact for it: the cold step takes the identical `nullptr` branch, so the cold
step's per-class peak IS the capture's per-class demand. On the `dbuf` arm the
capture's main-pool demand is a strict subset of the cold step's, which is the
containment #1393 recorded.

Cost on the newly covered path: `PreGrowForCapture` is idempotent and a no-op
once the free list is deep enough, so a warm server pays nothing after the first
capture at each shape.

## Risks and decisions

- **D1. Move the call rather than widen `dbuf`.** `dbuf` selects the parity ring
  and the persistent-input path; making it true off the async lever would change
  what is captured. The pre-grow is unrelated to both.
- **D2. Do not add a pre-grow to the other seven drivers here.** `qwen3.cpp`,
  `qwen3_moe.cpp`, `deepseek_v2.cpp`, `deepseek_v4.cpp`, `voxtral.cpp`,
  `laguna.cpp` and `qwen3_dflash.cpp` neither pre-grow nor record a demand
  profile, so each needs its own `MarkStepBoundary` / `StepDemandProfile` wiring
  and its own gate. Filed as
  [#2035](https://github.com/mudler/vllm.cpp/issues/2035), not fixed in flow: it is a
  different change with a different blast radius, and #2029's subject is the Qwen3.8-27B
  target.
- **D4. Two unguarded shape-keyed caches found on the same path, filed not fixed.**
  `DenseAlignFor` (`qwen3_5.cpp:2825-2849`) allocates five blocks AND calls
  `d.b.Synchronize(d.q)` at `:2846` on an `M` miss, and `EnsureCtmp`
  (`src/vt/cuda/cuda_marlin_dense.cu:74-89`) grows with `cudaMallocAsync` at `:85` — both
  with no `cudaStreamIsCapturing` refusal, unlike the six sibling caches that have one.
  Neither is #2029's cause: both are keyed by a quantity the cold step already visits at
  the same shape. [#2036](https://github.com/mudler/vllm.cpp/issues/2036).
- **D3. The soft-cap hazard #1393 recorded is unchanged.**
  `PreGrowForCapture` ignores `device_pool_cap_bytes` and every platform resolves
  it to 0. Running the pre-grow on more paths does not reach it; it widens who
  would meet it if a non-zero cap were ever set, which that platform already owes.

## Tests

`tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`, one new case per driver,
on the existing `decode_graph_seam_harness.h` CPU seam.

The assertion is the guarantee, not the call: **zero driver allocations occur
between `BeginCapture` and `EndCaptureGraph`** on a step where
`num_spec_decodes == 0` and `VT_ASYNC_EXECUTOR` is unset. The harness backend
already counts `Alloc`; this wave adds the capture-window split
(`allocs_during_capture()`), which is one counter incremented under a flag the
existing `BeginCapture` / `EndCaptureGraph` overrides set.

The pool is **drained between the cold step and the capture step**
(`vllm::Pool(b).Drain(b)`), which is what makes the case deterministic rather
than dependent on whether two of the tiny model's tensors happen to share a size
class. It is strictly stronger than the production condition — an empty free list
rather than a short one — and it is the only construction that also asserts the
premise #1393 left ungated: that the cold step's demand profile is SUFFICIENT for
the capture, not merely non-empty.

Asserting a transcription would be the failure `.agents/verification.md` names:
`CHECK(pool.misses() == ...)` against a constant, or checking that the pre-grow
was called, both stay green when the profile is wrong.

## Gates

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF
cmake --build build -j 4 --target test_qwen3_5_decode_graph_seam test_device_pool test_breakable_graph
./build/tests/test_qwen3_5_decode_graph_seam
./build/tests/test_device_pool
./build/tests/test_breakable_graph
scripts/agent-preflight.sh --staged
```

`cuda-fat-build` is the only thing that compiles the CUDA arm; this change edits
no `.cu`, so the CUDA half is a build verdict rather than a run verdict either
way.

## Evidence

Recorded in the pull request body: exit codes, red-before and green-after
readings, and the mutation results.

## Stop conditions

- If the drained-pool case does not go red before the fix, the premise is wrong
  and the wave stops: it would mean the capture makes no main-pool allocation the
  cold step did not already leave servable, and #2029's allocation is somewhere
  else. Report the three candidates and the device experiment that discriminates
  them rather than landing a speculative fix.
- If the case cannot be made green, `s.demand` does not cover the capture and the
  fix is the demand profile rather than its call site.

## Owed

- [#2029](https://github.com/mudler/vllm.cpp/issues/2029) stays open until a
  device run confirms the c=8 speculation-off rung serves. This wave removes a
  proven, non-speculative-only gap that produces exactly its message; it does not
  prove that gap was the only one. **No GPU was available to this implementer**,
  and nothing here is reported as device-verified.
- The seven decode drivers with no capture pre-grow at all —
  [#2035](https://github.com/mudler/vllm.cpp/issues/2035) (D2).
- The two unguarded Marlin-path caches —
  [#2036](https://github.com/mudler/vllm.cpp/issues/2036) (D4).
- The fatal handler prints `e.what()` and no backtrace while
  `include/vllm/v1/engine/core_client.h:63` promises "See stack trace (above)",
  which is why #2029 and #2028 were both expensive to chase —
  [#2037](https://github.com/mudler/vllm.cpp/issues/2037).
