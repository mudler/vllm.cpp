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

## The trigger is not the batch size, and a second measurement says so

Measured 2026-08-26 on `dgx:gpu0` (GB10) by the operator, same binary
`3d895a202`, `--num-blocks 3744 --max-num-seqs 16 --max-model-len 8192`, 1024 in
/ 512 out, `--speculative-config` removed, **plus `--enable-prefix-caching
--scheduling-policy lpm`**: the identical fault fires at **concurrency 1**.

```
engine-fatal: EngineCore busy loop threw: vt cuda: cudaMalloc: operation not permitted when stream is capturing
```

`stream is capturing` x4, `illegal memory access` 0, `position discontinuity` 0,
`server: prefix caching enabled` present, zero `dflash` mentions. Rung
`ok=0 failed=8`.

**This wave's finding is unchanged by it, and predicts it.** The gap is that the
pre-grow does not run when `dbuf` is false, and `dbuf` is false on EVERY
non-speculative step at EVERY padded size. Nothing about it is specific to
`S = 8`. What the two configurations differ in is only whether the free list
happens to be short when the capture opens, which is a property of the pool's
state and not of the captured shape. This wave's own gate is a **single-request,
`S = 1`** case, and it goes red.

**The rival hypothesis — a buffer sized from a quantity that varies per step —
was checked against the code and does not hold on this driver.**

| Candidate | Why not |
|---|---|
| `block_table_num_cols` / `max_blocks` | fixed once by `BlockTable`'s constructor from `max_model_len` (`src/vllm/v1/worker/gpu/block_table.cpp:49`) and read unchanged by `gather_block_table` (`runner.cpp:1185-1187`). It cannot move between two steps, and prefix caching does not touch it |
| FA-2 decode scratch | `DecodeShapeKey` is `{batch, hq, heads, groups, head_dim, max_blocks, page_size, num_splits}` (`src/vt/cuda/cuda_flash_attn_fa2.cu:1013`), all key-determined or constant — and it carries its OWN capture refusal at `:1021-1029` with a distinct message, so it cannot produce this one |
| `seq_lens` / `max_seq_len` | values and a host grid bound; `seq_lens` is a `[num_reqs]` buffer refreshed in place |
| per-request counts | `BuildPaddedDecode` rewrites `num_reqs = S` on the non-spec path (`qwen3_5.cpp:10233-10261`) |
| the forward's knowledge of prefix caching | it has none: `enable_prefix_caching` appears nowhere in `src/vllm/v1/worker/gpu/runner.cpp`'s forward path |

So the quantity that varies is the pool's free-list state. **Which SIZE CLASS is
short in either run is not determined here and cannot be from a CPU box**; see
`## Owed` for the three device experiments that would settle it, and note that
the fix does not depend on the answer — an unprepared pool is unprepared whatever
empties it.

Not the same fault as [#2042](https://github.com/mudler/vllm.cpp/issues/2042),
which is prefix caching plus DFlash2 dying at c=1 on the draft's position
invariant. With speculation ON, prefix caching fails through #2042; with
speculation OFF, through this.

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

## The lane the gate does not apply to, and the correction that came with it

`sanitize-cpu (thread)` refused the two new cases on the PRECONDITION, not on
the guarantee:

```
tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp:948:
  FATAL ERROR: REQUIRE( freed > 0 ) is NOT correct!  values: REQUIRE( 0 > 0 )
```

**It is not ThreadSanitizer and not its allocator.**
`.github/workflows/ci.yml:1598` sets `VT_POOL_BYPASS: "1"` for that job, on BOTH
the `address,undefined` and the `thread` lane. The identical `REQUIRE( 0 > 0 )`
reproduces at the same two lines on an ordinary non-sanitized Release build with
that one variable set and nothing else changed. The `address,undefined` lane on
#2047 read `pending` rather than green, so it never contradicted this; it had
not finished.

Under bypass every `Get` is a raw `Backend::Alloc` and every `Put` a real
`Free`, `Drain` reports 0, and `PreGrowForCapture` returns before it grows
anything. The guarantee is **false by design** there, identically for the fixed
and the unfixed driver, so the cases carry `doctest::skip(PoolBypassLane())`
with the reason in the case name. The predicate mirrors `DevicePool::Bypass()`
exactly. `REQUIRE(freed > 0)` is unchanged — M3 measured that it is what stops
the pooled lane from asserting nothing.

**A correction, because the first reading of the failure was backwards.** The
guard did NOT save a vacuous pass. Removed, the cases reach the capture
assertion and FAIL it, at **107** and **195** driver allocations inside the
capture, because under bypass every `Get` is a driver call. What the guard buys
is an inevitable failure that names the PRECONDITION instead of the SYMPTOM.
That is worth having and it is a smaller claim than the one first made for it.

**The guarantee is therefore not exercised by `sanitize-cpu`, and that is the
job's configuration rather than a limit of the instrument.** Measured on a real
TSan build (`-DVLLM_CPP_SANITIZE=thread`, `setarch -R`):

| Environment | Result |
|---|---|
| `VT_POOL_BYPASS=1`, as CI sets it | exit 0, 8 passed / **2 skipped**, 138 assertions, 0 TSan warnings |
| bypass unset, the pool ENABLED | exit 0, **10 passed / 0 skipped**, 156 assertions, `0` driver allocations inside both captures, **0 TSan warnings** |

So ThreadSanitizer runs this guarantee green when the pool is on. Nothing in
this file can opt one case back in — `Bypass()` is read once into a process-wide
function-local static (`device_pool.h:480-486`), so no scope, no locally
constructed `DevicePool` and no `ActivePoolScope` reaches it — and unsetting the
variable for these cases would be worse than the gap, because the pool would
then retain blocks that the job's own `ASAN_OPTIONS=detect_leaks=1` reports as
leaks. Filed as [#2059](https://github.com/mudler/vllm.cpp/issues/2059): the
bypass is an ASan requirement applied to a lane that has no leak detector.

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

Measured on `mudler-ubuntu-box` (x86-64, CPU-only), Release,
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF`, Ninja, `-j 4`.

| Reading | Result |
|---|---|
| RED, both new cases, before the fix | exit 1, 2 cases / 0 passed / 2 failed. `45` driver allocations inside the dense capture, `42` inside the MoE one |
| GREEN, both new cases | exit 0, `0` and `0` inside the capture, `49` in the step — every one of them the pre-grow, outside the region |
| `test_qwen3_5_decode_graph_seam` whole file, three runs | exit 0 each, 10 cases / 156 assertions |
| neighbours on the shared harness and the pool | `test_qwen3_decode_graph_seam` 4/231, `test_qwen3_moe_decode_graph_seam` 3/228, `test_deepseek_v2_decode_graph_seam` 3/230, `test_voxtral_decode_graph_seam` 3/230, `test_qwen3_dflash_decode_graph_seam` 4/23, `test_qwen3_dflash2_draft` 43/449, `test_moe_async_device_ids` 6/191, `test_device_pool` 11/59, `test_breakable_graph` 30/265, `test_persistent_step_input` 10/66 — all exit 0 |
| Qwen3.5 numerics neighbours | `test_qwen35_paged_forward` 7/63, `test_qwen3_5_gdn_spec_routing` 6/52, `test_qwen3_5_fa2_class` 6/15 — all exit 0 |

Three mutations, each compiled clean and each restored by sha256:

| Mutation | Result |
|---|---|
| M1, the production call site: put `PreGrowForCapture` back inside `if (dbuf)` | RED, 10 cases / 8 passed / **2 failed** — both new cases, and nothing else |
| M2, the pool half: `PreGrowForCapture` returns 0 before it grows anything | RED, `test_qwen3_5_decode_graph_seam` 8/10 and `test_device_pool` 10/11 |
| M3, the case's own construction: remove the `Drain`, on the UNFIXED driver | the capture-window assertion goes **vacuously green** — `0` allocations inside the capture and `0` in the whole step — and only the `allocs() > 0` guard fires. This is the measurement that says the drain is load-bearing rather than decorative: without it the case cannot detect the defect, and without the non-vacuity guard it would report a pass while measuring nothing |
| M4, the skip predicate: `PoolBypassLane()` returns false, run under `VT_POOL_BYPASS=1` | RED, 10 cases / 8 passed / **2 failed**, **146 assertions** — a byte-for-byte reproduction of the `sanitize-cpu (thread)` reading, on a NON-sanitized build. The skip is load-bearing and not a decorative no-op |

All four were re-run after the fixture changed, because a fixture change can
disarm a mutation proof. M1, M2 and M3 read identically before and after.

**`.cu` is untouched, so nothing here is `REMOTE_UNVERIFIED` on the CUDA arm for a
compile reason.** What IS unverified is the device behaviour: no GPU was
available to this implementer, and the c=8 speculation-off rung has not been
re-run.

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
  device run confirms the speculation-off rungs serve. This wave removes a
  proven, non-speculative-only gap that produces exactly its message; it does not
  prove that gap was the only one. **No GPU was available to this implementer**,
  and nothing here is reported as device-verified.

  Three experiments settle it, in this order. Each uses the operator's own
  invocation with `--speculative-config` removed and
  `--enable-prefix-caching --scheduling-policy lpm` added, at c=1 — the cheapest
  reproducer known.

  1. **`VT_ASYNC_EXECUTOR=1` on the UNCHANGED `3d895a202` binary.** That flips
     `impl_->dbuf` true, which turns the pre-grow on with no rebuild and no
     patch. A run that SERVES proves the missing pre-grow is the cause, on the
     device, against the exact binary the issue was measured on. A run that still
     dies is **inconclusive rather than a refutation**, and the asymmetry has to
     be stated when the result is read: the same flag also opens the parity ring,
     which retains a second `[S, vocab]` logits and `[S, H]` hidden per size and
     can therefore empty the free list by a route of its own.
  2. **The same configuration on a binary built from this branch**, environment
     untouched. Serving says the fix closes this entry point; dying says a second
     site exists.
  3. **`VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0`, as the control.** Serving confirms
     the fault is inside THIS driver's capture; dying says the capture that
     allocates is somewhere else, and 1 and 2 answered the wrong question.

  If 2 still dies, the next instrument is a backtrace at `CudaBackend::Alloc`,
  which is how #1380 was located and is exactly what
  [#2037](https://github.com/mudler/vllm.cpp/issues/2037) exists to make
  unnecessary. `VT_POOL_STATS` cannot substitute: it prints per-pool totals at
  destruction, after the engine is already dead, and names no class.
- The seven decode drivers with no capture pre-grow at all —
  [#2035](https://github.com/mudler/vllm.cpp/issues/2035) (D2).
- The two unguarded Marlin-path caches —
  [#2036](https://github.com/mudler/vllm.cpp/issues/2036) (D4).
- `sanitize-cpu` disabling the `DevicePool` on both lanes, so neither sanitizer
  executes the pool at all and this wave's gate is not exercised there —
  [#2059](https://github.com/mudler/vllm.cpp/issues/2059).
- The fatal handler prints `e.what()` and no backtrace while
  `include/vllm/v1/engine/core_client.h:63` promises "See stack trace (above)",
  which is why #2029 and #2028 were both expensive to chase —
  [#2037](https://github.com/mudler/vllm.cpp/issues/2037).
