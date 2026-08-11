# RTX 5070 Ti Qwen3.5-4B argmax scratch-growth spike

**Issue:** [#206 — RTX 5070 Ti: close Qwen3.5-4B TTFT, TPOT, and
VRAM gaps vs vLLM](https://github.com/mudler/vllm.cpp/issues/206)

**Row:** `KERNEL-SAMPLING`, feeding `ROAD-V1-C2-LOCAL-BF16`

**Parent campaign:**
[RTX 5070 Ti Qwen3.5-4B Pareto campaign](sm120-qwen35-pareto-2026-08-09.md)

**Lifecycle:** `SPIKED`; no implementation or performance credit

## Decision and objective

The first implementation candidate is an opt-in replacement for the current
exact-size, process-global argmax scratch allocator. It will use a per-device,
per-stream owner, round capacity geometrically to the next power of two,
allocate the new value/index pair with `cudaMallocAsync` on the submitting
stream, and pass replaced blocks to the existing `RetireGraphScratch` helper.
It must make no synchronous allocation or free call after the first sufficient
capacity is available. A strict selector preserves the incumbent path exactly
when unset, `0`, or invalid.

This spike is deliberately narrower than changing the two-pass argmax kernel.
The objective is to remove the measured host-blocking scratch growth storm,
improve Qwen3.5-4B TTFT without degrading throughput or TPOT, and retain exact
lowest-index argmax behavior and CUDA-graph pointer validity. It does not change
the default, the sampler ABI, scheduling, the normal device pool, or another
kernel family.

## Exact attribution and source ground

An `LD_PRELOAD` trace captured **506** total `cudaFree` calls. Exactly **30 hot
calls (15 pairs)** resolve through `src/vt/cuda/cuda_sample.cu:159-166`
`EnsureArgmaxScratch`, called from `GreedyArgmaxCuda` at `:193-217`; the other
476 have different teardown stacks and are excluded. The first free in each hot
pair blocks for about 191--200 ms while its immediately following sibling is
only microseconds. Independently, `VT_POOL_STATS` reports the normal pool
uncapped with a 99.14% hit rate. These facts reject the earlier normal-pool-cap
hypothesis and identify argmax scratch ownership as the next falsifiable lever;
they do not attribute the teardown calls to argmax.

The local path is:

- `src/vt/cuda/cuda_sample.cu:83-89` defines the exact reduction: higher value,
  then lower global index on ties.
- `ArgmaxPartialKernel` at `:102-128` and `ArgmaxFinalKernel` at `:130-151`
  implement the two-pass reduction, including the all-`-inf` sentinel rule.
- Process-global `g_argmax_val`, `g_argmax_idx`, and `g_argmax_cap` at `:153-157`
  own the current scratch. `EnsureArgmaxScratch` frees both old blocks
  synchronously, allocates exact requested sizes with `cudaMalloc`, and publishes
  them at `:159-166`.
- `GreedyArgmaxCuda` computes
  `blocks_per_row=min(ceil(vocab/256),256)`, requests
  `rows*blocks_per_row` elements, and launches both kernels at `:193-217`.
- `VT_FAST_ARGMAX=0` at `:168-201` is a slow-kernel comparison, not an allocator
  rollback, and must remain independent of this candidate.

The pinned vLLM oracle at `5559679229bc` routes all-greedy sampling through
`${VLLM_SOURCE}/vllm/v1/sample/sampler.py:239-271`, where
`greedy_sample` is `logits.argmax(dim=-1).view(-1)`. The pinned Torch reduction
defines the observable tie rule in
`.venv-vllm-pin/lib/python3.12/site-packages/torch/include/ATen/native/SharedReduceOps.h:437-447,493-496`:
equal values select the lower index. `ATen/native/ReduceOps.h:19-30` owns the
dispatched argmax stub. The matched vLLM trace uses this production path and has
only 9 allocation and 22 free calls after its warm-up/graph-capture interval;
that count is a denominator observation, not a claim about unobserved Torch
allocator internals.

Existing local parity gates are
`tests/vt/test_ops_sample.cpp:81-103` (basic and lowest-index tie), `:608-632`
(CUDA versus CPU random rows), and `:675-715` (151,936-wide cross-block tie,
high-index maximum, and all-`-inf`). The pinned sampler fixture construction is
at `${VLLM_SOURCE}/tests/v1/sample/test_sampler.py:128-166`. Commit
`5da39b0cf` introduced the fast two-pass argmax and its scratch; this work must
not restore the serialized slow scan.

## Binding behavior and ownership contract

1. Add a pure, CPU-testable selector such as
   `ArgmaxGeometricScratchEnabled(const char*)`. Only the exact string `"1"`
   selects the candidate. Unset, `"0"`, empty, `"true"`, `"01"`, and every
   invalid value invoke only the unchanged incumbent allocator. The working
   selector name is `VT_ARGMAX_GEOMETRIC_SCRATCH`; the implementer may choose a
   clearer name before RED only if this spec is amended first.
2. Key candidate scratch by `(device index, native CUDA stream)`. Each owner has
   one value pointer, one index pointer, an element capacity, and a submit mutex.
   A stream never consumes another stream's partials, and a stream handle reused
   on another device cannot alias the entry. The mutex covers growth decision,
   publication, and both kernel submissions so another host thread cannot
   replace or concurrently overwrite that owner's pair between launches.
3. For `need>capacity`, validate overflow and choose
   `new_capacity=next_power_of_two(need)`. Allocate both candidate blocks with
   `cudaMallocAsync` on that same stream before publishing either pointer or the
   capacity. If the second allocation fails, enqueue cleanup of the first on the
   same stream and leave the old pair and capacity unchanged before propagating
   the error.
4. After both allocations succeed, atomically publish the pair and capacity,
   then call `RetireGraphScratch` for both replaced non-null blocks. Never call
   `cudaFree` or `cudaFreeAsync` on an old published block: a previously captured
   graph may contain that address. Same/smaller requests allocate, free, and
   retire nothing.
5. Query `cudaStreamIsCapturing` before a capacity miss. A hit is capture-safe
   because its pointers are stable. A miss during capture fails before allocation,
   retirement, publication, or kernel launch and says that scratch must be
   warmed/reserved before capture. Growth outside capture may occur after an
   older graph was recorded because retirement keeps every baked pointer valid.
6. `RetireGraphScratch` remains the process-lifetime graph owner already used by
   other grow-only CUDA scratch. Do not add an unsafe queue-destruction free of
   retired blocks. Per-stream map cleanup is outside this small spike unless an
   independent test proves no graph can outlive it; owner cardinality and retained
   bytes must nevertheless be reported.
7. `VT_FAST_ARGMAX=0` continues to select the existing slow kernel without
   allocating candidate scratch. The new selector changes allocation ownership
   only; it may not change kernel geometry, comparator, output type, sampling
   metadata, or public ABI.

For the bound benchmark (`vocab ~= 248,320`, maximum batched rows 2,048),
`blocks_per_row=256`, so the maximum request is 524,288 elements. The exact-size
final pair uses
`524288*(sizeof(float)+sizeof(int64_t)) = 6,291,456` bytes (6 MiB). The generic
next-power-of-two ratio gives a coarse active bound below 12 MiB; at this exact
power-of-two workload ceiling it is at most 6 MiB, and the sum of all preceding
geometric capacities is less than the active capacity. The acceptance gate uses
the deliberately looser **18 MiB total active-plus-retired** bound per benchmark
owner to cover transient allocation and allocator-accounting effects. Overflow,
owner count, and total retained bytes are explicit diagnostics outside timed
execution; multiple owners are reported separately rather than hidden inside
the per-owner bound.

## Alternatives and discriminators

| Strategy | Benefit | Rejection risk / discriminator |
|---|---|---|
| Exact-size synchronous grow/free (incumbent) | Exact rollback, bounded live allocation | Observed 30 hot `cudaFree` calls and repeated 191--200 ms host stalls |
| Power-of-two capacity only | Reduces growths from distinct batch high-water marks to logarithmic growth | Still performs a synchronizing `cudaFree` on every remaining growth; insufficient if any measured hot free remains |
| Power-of-two plus `RetireGraphScratch` and same-stream `cudaMallocAsync` (**first candidate**) | Removes hot synchronous free, preserves graph pointers, and bounds geometric retention | Retained bytes and stream-owner cardinality must meet the hard memory gates; async allocation must occur only outside capture |
| Stream-ordered async free | Bounds live bytes without a process-lifetime retire list | Rejected for a published pointer that a graph may replay; reconsider only with proven graph-lifetime ownership and destruction ordering |
| Explicit pre-serve reserve | Can make the entire serving interval allocation-free with the tightest known capacity | Requires a runner-to-sampler lifecycle surface and complete coverage of every caller; attempt only if the small candidate still grows in the measured interval |

If the first candidate eliminates the attributed calls but misses the latency
gate, the next experiment is explicit pre-serve reserve using
`max_rows*min(ceil(vocab/256),256)`. Do not combine it silently with the first
measurement. If retained memory fails its bound, do not substitute async free
without first proving graph lifetime with a dedicated spec amendment.

## RED-first implementation and focused tests

A fresh implementer adds the smallest portable ownership/selector test before
product code and records the intended compile failure because the seam does not
exist. Prefer a small header-level state machine with injected allocate, retire,
capture-query, and submit callbacks; CUDA adapter code supplies the actual runtime
calls. The tests must prove:

- strict selector truth table and exactly one incumbent/candidate callback;
- requests `1,3,5,9,17,32`, shrink, and regrow produce capacities
  `1,4,8,16,32,32,32`, with no calls on hits;
- both new allocations precede publication, both previous pointers retire once,
  and no free callback exists on successful hot growth;
- failure of either allocation preserves the previous pair/capacity, cleans only
  a partially created candidate through the same-stream failure path, and does
  not retire or launch;
- device/stream keys isolate owners and a concurrent submit cannot observe a
  half-published pair or interleave between partial and final launches;
- a warmed capture hit allocates nothing; a capture miss fails before changing
  state or launching;
- overflow is rejected and the geometric byte bound holds across every tested
  growth sequence;
- unset, `0`, and invalid selectors execute byte-for-byte incumbent callback
  identity, while `VT_FAST_ARGMAX=0` executes neither fast allocator;
- the existing CUDA exactness fixtures remain exact, plus a growth sequence over
  several row counts preserves CPU token IDs, cross-block lowest-index ties,
  unique maxima, and all-`-inf` index zero.

The fresh reviewer mutates the immutable implementation in a scratch copy and
must kill at least these defects: permissive/inverted selector; invoking both
callbacks; exact rather than power-of-two growth; freeing instead of retiring;
publishing after only one successful allocation; dropping the device or stream
from the owner key; releasing the submit lock between kernels; allowing a
capture miss; leaking/double-cleaning a failed first allocation; wrong
`float`/`int64_t` byte sizes; overflow in rounding; changing `blocks_per_row`;
last-index tie selection; and an all-`-inf` sentinel that returns a nonzero index.
The tree is restored byte-for-byte after every mutation campaign.

## Correctness, trace, memory, and timing gates

Correctness precedes timing. On the exact immutable binary used for A/B:

1. Run the focused pure selector/owner tests, complete sampling CPU tests, full
   CUDA `test_ops_sample`, clean CUDA `-Werror`, and `compute-sanitizer memcheck`
   on the growth/tie/capture fixture. Port the applicable pinned sampler
   parameters rather than weakening local coverage.
2. Run Qwen3.5-4B BF16 with the parent campaign's cached model and ShareGPT
   prompts: 128 requests, 128 output tokens, concurrency 32,
   `max_num_batched_tokens=2048`, 1,280 KV blocks, greedy sampling, the accepted
   production frontend, and every other accepted toggle held fixed. Prompt IDs,
   every request's output IDs, output order, and their hashes must match across
   all six arms before interpreting performance.
3. Use one binary and an idle, cgroup-contained RTX 5070 Ti. Run the
   counterbalanced sequence
   `INC-a -> GEO-a -> GEO-b -> INC-b -> INC-c -> GEO-c`, where the new selector
   is the only difference. Record exact command, binary SHA-256, git revision,
   model snapshot/hash, driver/runtime, clocks/power, ambient contention, and
   raw outputs.
4. Record total and output throughput, mean/median/p95 TTFT, mean TPOT/ITL,
   mean/p95 end-to-end latency, GPU peak/resident memory, host PSS, and load time
   for every arm. Report each value and candidate/incumbent ratio, not only a
   headline mean.
5. Trace both local arms with the same `nsys` command and full request interval.
   The candidate passes attribution only if zero `cudaFree` calls resolve to
   `EnsureArgmaxScratch`, zero synchronous frees occur on candidate growth, and
   the remaining allocation/retire count agrees with the geometric high-water
   sequence. Record CUDA API time and argmax kernel counts/durations. Re-run the
   pinned production vLLM trace with the identical workload if its existing
   trace cannot be matched to the same host state and profiler options.
6. Hard memory gates: the final active argmax pair is at most 6 MiB for the
   declared `n<=2048`, `blocks_per_row<=256` workload, and accounted
   active-plus-retired scratch remains below the conservative 18 MiB bound per
   benchmark owner. Observed peak VRAM is no more than incumbent plus 18 MiB and
   no more than the historical 13,053.3 MiB local peak. Host PSS and owner
   cardinality may not grow unbounded across repeated queue create/destroy cycles.
7. Accept only if all three candidate legs are exact and stable, at least two of
   three paired TTFT observations improve, aggregate mean TTFT improves by at
   least 1%, and total/output throughput, TPOT/ITL, E2E, load time, peak VRAM,
   and host PSS do not regress outside the campaign's calibrated noise band.
   A trace improvement without end-to-end improvement is diagnostic evidence,
   not a default flip.

The operator reruns the full row gate after fresh implementation and mutation
review. Any later default change requires this spec's `## Outcome`, accepted raw
evidence, and the parent campaign records; this spike alone authorizes none.

## Rollback, risks, and stop conditions

Rollback is exact and same-binary: unset, `VT_ARGMAX_GEOMETRIC_SCRATCH=0`, or
an invalid value takes only the unchanged process-global exact-size
`cudaFree`/`cudaMalloc` path. The slow-kernel `VT_FAST_ARGMAX=0` remains a
separate diagnostic and receives no performance credit.

Stop and return `NEEDS_DECISION` rather than widening scope if the change needs
a public API, scheduler behavior, sampler arithmetic, default flip, normal pool
change, or keyed roadmap/matrix edit. Reject the candidate on any token mismatch,
tie/all-`-inf` semantic drift, capture allocation, dangling graph pointer,
cross-stream race, unbounded owner growth, memory-gate failure, sanitizer error,
or Pareto regression. If all attributed frees disappear but TTFT does not move,
retain the trace as a falsified end-to-end hypothesis and profile the next
largest same-tool interval; never declare a ceiling.

## Outcome

**REJECTED and removed, 2026-08-09.** Fresh RED-first implementation and
mutation review passed at `06db3bbb3454f85e20f75b96eeb299afba7036ae` (9
portable cases / 105 assertions plus the CUDA sampling gate). The immutable
same-binary series
`INC-a -> GEO-a -> GEO-b -> INC-b -> INC-c -> GEO-c` was token-exact in all
six legs (SHA-256 `be20ffbceb61f0264ca21d972bfc5fc51e855e64f2b945de71669cae666aa702`).
Raw evidence is `/tmp/qwen35-ab-argmax-geometric-06db3bbb/`.

| Arm mean (3 reps) | total tok/s | output tok/s | mean TTFT | mean TPOT / ITL | mean E2E |
|---|---:|---:|---:|---:|---:|
| Incumbent | 6862.2667 | 758.8100 | 1009.7167 ms | 34.3400 ms | 5371.0000 ms |
| Geometric | 6863.8867 | 758.9867 | 965.0133 ms | 34.6833 ms | 5369.7067 ms |
| Change | +0.0236% | +0.0233% | -4.427% | **+0.9998%** | -0.0241% |

The candidate therefore fails the no-TPOT-regression Pareto gate. Same-tool
`nsys --trace=cuda,nvtx --cuda-graph-trace=node` explains why the apparent
TTFT win is not an end-to-end compute saving: incumbent `cudaFree` was
506 calls / 2960.361 ms (max 200.004 ms), candidate was 476 calls / 27.705 ms
(max 0.688 ms), exactly removing the 30 expected argmax frees; meanwhile
`cudaStreamSynchronize` rose from 18,027.497 to 20,878.260 ms over the same
567 calls. The wait moved to the later synchronization boundary. Candidate
trace hashes are `1021b045...239e7` (`.nsys-rep`) and
`808a7516...b7be4` (SQLite); the incumbent SQLite hash is
`bab04901...be178`. No default or performance credit is taken, and the
selector/product/tests are removed while this falsified hypothesis remains as
evidence.
