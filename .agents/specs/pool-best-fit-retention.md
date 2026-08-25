# `ENG-POOL-BEST-FIT` — a freed scratch block must be reusable by a smaller request

**Issue:** [#1922](https://github.com/mudler/vllm.cpp/issues/1922) (open; this
row does not close it — see `## Owed` O1).
**Row:** `ENG-POOL-BEST-FIT`. **Base:** `main` @ `2e2b3fc1a`.
**Owning file:** this spec.
**Status at write time:** spec committed before the implementation commit, per
AGENTS.md "Spec before code".

## Now

`ACTIVE` on `row/ENG-POOL-BEST-FIT`, PR
[#1930](https://github.com/mudler/vllm.cpp/pull/1930). The change is CPU-gated. The
attribution of [#1922](https://github.com/mudler/vllm.cpp/issues/1922)'s own
`avail` curve on `dgx:gpu0` at `--max-model-len 12288` is OWED and is not
claimed anywhere here (O1).

## Honesty statement — what this row claims and what it does not

**Claimed, and measured on this tree.** `vllm::DevicePool` — the shared scratch
allocator every model forward draws from — could only ever hand a freed block
back to a request in the block's own size class. Twelve sequential requests
through the production `LoadedEngine::generate` entry, with the **largest
request first** so that every later request demanded strictly less than one
already served, still grew the pool's retained bytes 5.1× and made 240 further
driver allocations. That is measured, red-first, and it is what this row fixes.

**Not claimed.** That this is the whole of #1922. #1922 reports host
`MemAvailable` falling about 2 GiB per request on `dgx:gpu0` with a
Qwen3.8-27B NVFP4 target and a DFlash2 draft at `--max-model-len 12288`. This
session had no GPU (the box was under another lease), so nothing here was run on
that configuration, and no `avail` curve is reproduced, attributed or claimed.
What is claimed is that the mechanism named above is present, is on that exact
path, is unbounded in the number of distinct shapes the traffic shows, and
scales with `max_model_len` — which is the axis #1922 reports. The confirmation
run is O1.

**Also measured, and NOT fixed here.** After the fix the process heap still
grows across the same twelve requests, by about a tenth of what it grew before.
The residue is not this pool: it is the other shape-keyed process-lifetime
caches the forward carries (under `Upstream chain`). They are their own issue and their own row
(O2).

## Scope

IN: `vllm::DevicePool`'s reuse policy — one fallback in `Get`, the block-class
bookkeeping `Put` needs to support it, and a `stats()` accessor the gate reads.
The gate itself, its two documentation surfaces, and this row's records.

OUT: every other shape-keyed cache on the forward path
([#1926](https://github.com/mudler/vllm.cpp/issues/1926)), the runner's
`connector_stored_blocks_` ([#1927](https://github.com/mudler/vllm.cpp/issues/1927)),
the DFlash2 draft store ([#1919](https://github.com/mudler/vllm.cpp/issues/1919),
another implementer), setting `device_pool_cap_bytes` on any platform, and the
profiling forward vLLM has and we do not
([#1165](https://github.com/mudler/vllm.cpp/issues/1165)). Also OUT: any
throughput or concurrency measurement — see `## Owed` O1.

### Why this is worth a row, and not a footnote

A ratchet is worse than a high water mark, and this one lands on the axis the
campaign cannot reach.

Every published number in the [#1574](https://github.com/mudler/vllm.cpp/issues/1574)
benchmark round is taken at `--max-num-seqs 1`. The reason recorded in
[#1922](https://github.com/mudler/vllm.cpp/issues/1922) is memory behaviour, and
this is the shape of it: what the engine holds is a function of how many
distinct shapes it has SEEN, so it goes up with every request and never comes
down. A high water mark can be sized for. A ratchet cannot, and the only way to
serve under one is to keep the shape set small — one sequence, short context —
which is exactly the configuration the campaign has been forced into.

Concurrency makes the ratchet steeper rather than merely larger, and that is the
part that decides the axis. At `--max-num-seqs 1` the scheduled token count per
step comes from ONE sequence. At c4 it is a sum over four, so the set of
distinct per-step shapes the traffic can present is combinatorially larger, and
under exact-class-only reuse each new combination mints its own retained block.
The engine therefore climbs faster at the concurrency where the competitors
publish their best figures — their own c2/c4/c6 numbers are several times their
c1 — and dies sooner. That is why there is no concurrency ladder for this engine
at all.

This row does not measure a ladder and does not claim one; the c2/c4/c6 run is
`## Owed` O1 together with #1922's own curve. What it removes is the reason the
ladder could not be attempted.

## Our baseline

The defect, as `main` carries it today.

`include/vllm/model_executor/models/device_pool.h` keys its free list by a size
class (`ClassOf`, `kClassBits = 4`, so 16 classes per octave and at most 6.25%
over-allocation). `Get(bytes)` looks in `classes_[ClassOf(bytes)]` and, on a
miss, asks the backend. `Put(bytes, p)` returns the block to
`classes_[ClassOf(bytes)]`.

Nothing else is consulted. A block sitting free in a LARGER class is invisible
to a smaller request even though it satisfies it, so **retention is a function
of how many distinct shapes the traffic has shown, not of how much one step
concurrently needs**. Blocks are never returned to the driver
(`device_pool.h` header, "Blocks are never returned to the driver"), `Drain()`
is called from exactly two places and neither is on the LLM serving path
(`src/vllm/multimodal/ltx2_video.cpp`, `src/vllm/model_executor/models/minimax_h3_pipeline.cpp`),
and the soft cap that would bound it resolves to `0` — uncapped — on every
platform (`include/vllm/platforms/interface.h`, `src/vllm/platforms/cuda.cpp`).

The per-step scratch shapes are `[T, hidden]`, `[T, intermediate]`,
`[C, kdim]`, `[S, vocab]` and so on, where `T` is the step's scheduled token
count and `C` the context length. Both move with the traffic, and how far they
can move is set by `max_model_len`. So the retained set grows request by
request and the ceiling rises with `--max-model-len`.

### The measurement, red-first

`tests/vllm/v1/worker/test_engine_scratch_steady_state.cpp`, CPU, synthetic
Qwen3.5 dense target driven through `LoadedEngine::generate`. Twelve sequential
requests, prompt lengths **descending**, so request 0 is the peak and requests
1..11 each demand strictly less than it. Nothing in requests 1..11 needs a
buffer request 0 did not already allocate and return.

| arm | pool retained after req 0 | after req 11 | growth | driver allocations, reqs 1-11 |
|---|---|---|---|---|
| pre-fix (`VT_POOL_BORROW=0`) | 560 243 B | 2 867 707 B | **5.12×** | **240** |
| post-fix (default) | 536 403 B | 624 899 B | 1.16× | 27 |

Same run with a DFlash2 draft attached, which is #1922's configuration in
miniature. This arm was taken with a scratch probe over the same synthetic
engine plus the `dflash2_runner_fixture.h` draft and is NOT in the tree — it is
recorded because it shows the defect is not an artefact of the plain arm, and
the in-tree gate deliberately does not depend on the DFlash2 fixture, which
[#1919](https://github.com/mudler/vllm.cpp/issues/1919)'s implementer is editing
(see `Port map` -> `Coordination`):

| arm | retained after req 0 | after req 11 | growth | driver allocations, reqs 1-11 |
|---|---|---|---|---|
| pre-fix | 622 120 B | 3 515 972 B | **5.65×** | 280 |
| post-fix | 569 944 B | 1 199 620 B | 2.10× | 26 |

The pre-fix figure is not a warm-up cost that saturates: it is still climbing at
request 11, and the classes counter is still climbing with it (24 → 101).

## Upstream chain

### The mirror

vLLM's activations come out of torch's caching allocator. `get_free_block`
(`c10/cuda/CUDACachingAllocator.cpp`) searches the cached pool for the
**smallest cached block at least as large as the request** before it asks the
driver, and splits it when the remainder is worth keeping. That search is the
structural difference: upstream's cache converges on the peak working set, ours
converges on the sum over every shape ever seen.

vLLM additionally runs a profiling forward at the worst-case shape before it
serves anything (`determine_available_memory` → `profile_run`), so its cache is
sized for the peak on the first step and steady-state serving mints nothing.
We have no such profile run — `gpu_memory_utilization` is still inert
(`.agents/specs/gpu-mem-util-inert.md`, [#1165](https://github.com/mudler/vllm.cpp/issues/1165)) —
so production traffic is what reveals our shapes, one at a time, forever. That
is the second half of the same gap and it is not this row (O3).

### What this row does

`Get` gains one fallback, taken only after its own class misses and before the
backend is asked: walk the class ladder upward and take the block from the
first non-empty class within `kBorrowMaxRatio` (2×) of the requested class.
`kBorrowMaxSteps` (16) is one octave at `kClassBits == 4`, so the two constants
state the same bound from both ends and the loop stops at whichever it reaches
first.

The borrow is a **loan, not a demotion**: `block_class_` records the class the
driver actually allocated each live block at, and `Put` returns the block
there. The large class gets its block back and can still serve a large request,
which is what keeps a borrow from starving the class it came from. The demand
accounting (`live` / `base` / `peak`, which `StepDemandProfile` reads) stays the
REQUESTING class's, because that is what it means.

`VT_POOL_BORROW=0` restores the pre-fix exact-class-only reuse for a
same-binary A/B, in the shape `VT_POOL_EXACT` and `VT_POOL_BYPASS` already
have.

### Why this cannot break a CUDA-graph capture

A `cudaMalloc` inside a captured region aborts the capture, which is why
`PreGrowForCapture` exists: it makes the free list deep enough to serve the
whole captured region from hits. That property is preserved, and the argument is
exact rather than empirical:

`PreGrowForCapture(demand)` grows every class in the eager step's demand
profile to that class's peak transient count. The captured region runs the same
allocation sequence as the eager step, so at every instant each class's live
count is at most its measured peak, and every `Get` inside the capture **hits
its own class**. A borrow is reachable only from a MISS. Therefore no borrow can
occur inside a correctly pre-grown capture, and the pre-grown provisioning
cannot be stolen by a smaller class.

Where a capture is NOT correctly pre-grown — the captured region demands a class
the eager step did not — the pre-fix behaviour is a `cudaMalloc` and an aborted
capture. With the borrow the miss may instead be served, and the capture either
succeeds or aborts later at the next unserviceable miss. The borrow can turn an
abort into a success; it cannot turn a success into an abort.

### What the residue is, and why it is not fixed here

With the pool bypassed entirely (`VT_POOL_BYPASS=1`) the same twelve requests
still grow the heap by ~390 KB. That is the other shape-keyed,
process-lifetime, never-evicted caches on the forward path, each keyed by an
exact token count and each holding memory for every distinct count ever seen:

- `dense_attn_block.h` `row_idx_by_t` — host, `4·T` bytes per distinct `T`,
  retained deliberately because a captured graph bakes the host source address.
- `dense_nvfp4_gemm.h` `DenseAlignFor` and its second copy in `qwen3_5.cpp` —
  four device buffers per distinct token count `M`, never freed.
- `qwen3_5.cpp` `MoeFusedResident` / `MoeBf16Resident` `tok_map` — one entry per
  distinct `T`.
- `src/vt/cuda/cuda_matmul.cu` `heurs` and `plans` — keyed on a shape that
  includes `M`; `plans` values hold cuBLASLt descriptors that are never
  destroyed.

Each is smaller than the pool by an order of magnitude and each needs its own
decision about what to key on. They are O2.

## Port map

One file changes in product code: `include/vllm/model_executor/models/device_pool.h`.

- `Get`: own-class hit (unchanged) → bounded best-fit borrow (new) → backend
  allocation (unchanged). Every handed-out block is recorded in `block_class_`.
- `Put` / `Put(cap)`: the demand decrement is the requester's class; the
  free-list return uses `TakeBlockClass`, the class the driver allocated the
  block at.
- `NextClassAbove`, `BorrowEnabled`, `TakeBlockClass`, `NoteReturned`,
  `kBorrowMaxRatio`, `kBorrowMaxSteps`, `block_class_` are new and private.
- `stats()` is new and public: hits, driver allocations, retained bytes,
  distinct classes, live blocks. It exists because a memory gate written
  against process heap bytes measures every unrelated cache in the tree and
  needs a tolerance nobody can justify, while the driver-allocation count
  measures exactly one thing — a request this pool could not serve from what it
  already held. `VT_POOL_STATS` answers the same question at destruction, which
  is one request too late to gate on.

`block_class_` holds one entry per LIVE block — the step's working set, not the
history — and empties itself as blocks return. It is needed because `Put` is
told the caller's byte count and a borrowed block's own size is not derivable
from that.

### Coordination — what this row does not touch

[#1919](https://github.com/mudler/vllm.cpp/issues/1919) (the DFlash2 draft
store's hard 4096-slot cap) is in flight in another worktree and is on the same
serving path. The two changes do not overlap in any file: #1919 owns
`src/vllm/model_executor/models/qwen3_dflash.cpp`, which this row does not edit,
and this row owns `include/vllm/model_executor/models/device_pool.h`, which
#1919 has no reason to. Nor do they overlap in mechanism: #1919 is a fixed
capacity that REFUSES and kills the engine, this is a reuse failure that
allocates and never gives back. #1922 says the reproduction it reports stayed
under #1919's cap, so neither is the other.

They do meet at one place worth naming, and it is the pool: a store that
reallocates per request returns its old pools to this allocator, and whether
those bytes can then serve anything else is exactly what this row changes. If
#1919 sizes the store from `max_model_len`, the blocks it returns get larger,
and the borrow is what keeps them reachable.

The unrelated `connector_stored_blocks_` defect found in the same audit is
[#1927](https://github.com/mudler/vllm.cpp/issues/1927) and lives in
`src/vllm/v1/worker/gpu/runner.cpp`, which this row also does not edit, for the
same reason: a different behaviour with a different test surface.

### Rejected

- **Drain the pool between requests.** Bounds retention, but it is not a mirror
  of anything upstream does (torch's `empty_cache()` is explicit and vLLM never
  calls it while serving), and it pays a full re-warm on the first steps of
  every request — a throughput cost on exactly the axis [#1574](https://github.com/mudler/vllm.cpp/issues/1574)
  is measuring, which this session could not measure.
- **Set a non-zero `device_pool_cap_bytes`.** The designed knob, but it needs a
  byte value nobody has measured, and the header records an owed
  `PreGrowForCapture` repair for whoever first sets one. A number chosen without
  a measurement is the thing this protocol exists to refuse.
- **Adopt a borrowed block into the borrowing class.** Simpler (no
  `block_class_`), and wrong: it demotes large blocks permanently, so the large
  class re-allocates and the pool ping-pongs.
- **Split blocks the way torch does.** The faithful long-term answer. It needs
  the pool to own sub-allocation, which is a different allocator, and it is not
  reachable as one scoped change.

## Tests to port

`tests/vllm/v1/worker/test_engine_scratch_steady_state.cpp`, CPU, no
checkpoint:

1. **The production-entry case (the proof).** A `LoadedEngine` over synthetic
   in-memory Qwen3.5 dense weights, twelve sequential
   `engine().generate(...)` calls with descending prompt lengths. Asserts
   (a) the pool's retained bytes after the last request are at most
   `kBorrowMaxRatio` times its retained bytes after the peak request, and
   (b) the driver allocations made during requests 1..N-1 do not exceed those
   made during request 0. Both fail on the pre-fix tree by the margins in `Our baseline`.
2. **The unit case (localization).** A `DevicePool` over the CPU backend
   directly: a block released at a larger class is handed to a smaller request
   within the ratio, is NOT handed to one outside the ratio, and comes back to
   its OWN class so a later large request still hits.
3. **The A/B case.** `VT_POOL_BORROW` is read once per process, so the
   restore-the-old-behaviour arm is asserted through `BorrowEnabled`'s
   observable effect rather than by flipping it mid-run.

**Why the assertion is cumulative.** The obvious steady-state form — "request
`k` costs what request `k+1` costs, within a tolerance" — is MUTE on this
defect, and the test says so in its own header rather than leaving the next
person to rediscover it. Both arms decay per request: the broken arm's
per-request cost falls from 29 driver allocations to 17 across this run, which
is below what request 0 alone spends, so a per-request comparison PASSES on the
defect. The separation lives only in the total — eleven small requests together
cost three times what the one big request cost, and the broken arm never stops
paying. A per-request threshold here would be a floor below the real count.

## Evidence

Tree: `row/ENG-POOL-BEST-FIT`, merged onto `origin/main` @ `a73b26968`. CPU,
x86-64, configured exactly as CI's `cpu` job does
(`cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON`, no
`CMAKE_BUILD_TYPE`, so `NDEBUG` is absent and every `assert` is live).

Focused gate `test_engine_scratch_steady_state`, both cases:

| arm | result | numbers the run printed |
|---|---|---|
| default | 12/12 assertions, `SUCCESS` | retained 536 403 B after the peak request, 624 899 B after 12; 72 driver allocations for the peak request, 27 for the 11 smaller ones |
| `VT_POOL_BORROW=0` | 6/12 assertions, `FAILURE` | retained 560 243 B after the peak request, **2 867 707 B** after 12; 79 driver allocations for the peak request, **240** for the 11 smaller ones |

The same measurement was taken before any of this row's code existed, with a
scratch probe over the same synthetic engine on `main` @ `2e2b3fc1a`: the
pre-instrument tree grew the process heap 1 714 400 B -> 4 279 840 B across the
same twelve descending requests, and 3 064 832 B -> 6 313 216 B with a DFlash2
draft attached. That probe is not in the tree; the `VT_POOL_BORROW=0` arm is the
in-tree form of the same red.

### Mutations

Each applied to a scratch copy of the head, rebuilt (52/52 targets, no
`no work to do`), run, then restored to sha256
`5ac6b9831eeac030ac29fba8ce7b99088487a96b64b66f3c3c084fb1470e7b81` and rebuilt
green.

| # | mutation | expected | observed |
|---|---|---|---|
| M1 | the borrow loop cannot run (`step < kBorrowMaxSteps` -> `step < 0`) | both cases red | 6/12, `FAILURE`; the ENGINE case reports 2 867 707 B and 240 allocations — which is also the REACHABILITY proof, because that case enters only through `LoadedEngine::generate` |
| M2 | a borrowed block is DEMOTED on return (`TakeBlockClass(p, key)` -> `key`) | the unit case red | 9/12, `FAILURE`; the "goes home" reuse and the ratio bound both fire, and the engine case stays green — which is why the unit case exists |
| M3 | the instrument is dead (`s.retained_bytes = retained_` -> `= 0`) | the gate refuses rather than passing | 8/9, `FAILURE` at `REQUIRE(after_peak.retained_bytes > 0)` |
| restore | — | 12/12 green | 12/12, `SUCCESS`, after `git diff` on the file reported EMPTY against the committed head |

M3 is there because this gate reads an accessor this row added: an instrument
that returns a constant would have made every threshold trivially satisfiable,
which is the shape a memory gate fails into.

M1 IS THE REACHABILITY MUTATION, and not a second-best stand-in for one.
`.agents/reachability.md` asks for the production CALL SITE to be deleted; the
production call site of everything this row adds is the borrow branch inside
`DevicePool::Get`, which is what M1 removes. The engine case then reds. That
case constructs nothing by hand: it builds a `LoadedEngine` and calls
`engine().generate(...)`, so the only path from the test to the deleted code
runs through the loader, the scheduler, the runner and the forward's `DBuf`
allocations. A unit case over a directly-constructed pool could not have shown
that, which is why both cases exist and why only one of them is the proof.

### The full suite, and three reds that are not this change's

`ctest --test-dir build --output-on-failure`, CPU, serial: **621 of 622 passed,
1 failed, 1 skipped**. Every speculative-decoding fixture in it is green, which
is the token-identity check this change owes: a borrowed block is larger than
the request and therefore holds different stale bytes than a same-class block
would, so a kernel reading past its own extent would surface as a token flip.

The one failure is `test_cpu_threadpool`, whose oversubscribed-dispatch RATIO
guard is a function of scheduler contention. It was re-run on its own and is
green: 9 cases, 19 602 assertions, `SUCCESS`. The box was carrying load average
70 on 20 cores, with two other sessions building and running `ctest` in
parallel, which `.agents/verification.md` names directly: "Tests that starve
under `ctest -j` are re-run serially before being called a regression."

`scripts/agent-preflight.sh` was run three times as the branch moved, and the
same load shows in it. The first run reded FIVE suites and EACH is green re-run
serially: `test_check_release_binary_contract` (30/30),
`test_release_manifest` (22/22), `test_release_archive` (23/23),
`test_check_attention_rung_consistency` (39/39) and
`test_cpu_x86_llamacpp_floor` (10/10). The LAST run, after the second
`origin/main` merge, is green on every gate but that one -- including
`commit-trailers` and `commit-style`, which had SKIPPED before the merge and
therefore had reported nothing about this tree.

The one standing red is
[#618](https://github.com/mudler/vllm.cpp/issues/618) itself, and it prints its
own reason: `waiting for quiet: 15s busy=127% ... load=33.94`, exiting 4
(`NO_QUIET_WINDOW`) where the case expects 2. It is the gate this task was told
to treat as exempt, and it is exempt for exactly this: it measures a quiet box,
and the box was not quiet.

`windows-msvc-cpu` and `windows-msvc-vulkan` fail on this pull request at
`test_openai_api_server.exe exited with status -1073740791`
(`STATUS_STACK_BUFFER_OVERRUN`). PRE-EXISTING, and that is measured rather than
assumed: the SAME job fails with the SAME binary and the SAME status on
[#1918](https://github.com/mudler/vllm.cpp/pull/1918), which predates this
branch and has already MERGED (run `32860126816`, job `97841755956`). The MSVC
lane is pull-request-only, so no `main` baseline exists to say when it started.

## Risks/decisions

- **A borrowed block is dirty from a different shape.** So was every
  same-class block before this change; the pool has never zeroed. An op that
  reads scratch beyond its logical extent could see different garbage than
  before. That is a detector for a pre-existing defect, not a new one.
- **Transient over-allocation up to 2×** while a borrowed block is held. Bounded
  by `kBorrowMaxRatio` and stated in the header.
- **Two hash operations per `Get`/`Put`**, under a mutex the pool already takes.
  The measured hit rate rises (fewer backend calls), so the allocator does
  strictly less work, not more.
- **The gate's thresholds are design constants, not fitted numbers** — the
  borrow ratio and "no more than the peak request's own allocations". The
  measured margins (1.16 vs 5.12, 27 vs 240) are wide.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 4 --target test_engine_scratch_steady_state
./build/tests/test_engine_scratch_steady_state                    # must PASS
VT_POOL_BORROW=0 ./build/tests/test_engine_scratch_steady_state   # must FAIL
cmake --build build -j 4
ctest --test-dir build --output-on-failure
scripts/agent-preflight.sh --staged
```

## Dependencies

| Depends on | Why | State |
|---|---|---|
| Nothing in flight | The change is one header and one new test; it adds no seam, needs no new op, and touches no file another open row owns | met |
| `PreGrowForCapture` keeping its contract | The no-regression argument for CUDA-graph capture rests on it provisioning every class the eager step demanded | met, unchanged by this row |
| A GPU lease | Only for `## Owed` O1, which is not part of this row's gate | NOT met; `dgx:gpu0` was under the developer's lease for the whole window |

## Work breakdown

One wave. The change is not decomposable: the borrow in `Get` and the
block-class bookkeeping in `Put` are the same guarantee written from two ends,
and landing either alone is a defect — a borrow without the bookkeeping demotes
blocks permanently, and the bookkeeping without the borrow is dead code.

| Step | Surface | State |
|---|---|---|
| W1a — the bounded best-fit borrow and its bookkeeping | `include/vllm/model_executor/models/device_pool.h` | DONE |
| W1b — the `stats()` instrument the gate reads | same file | DONE |
| W1c — the unit case and the production-entry case | `tests/vllm/v1/worker/test_engine_scratch_steady_state.cpp` | DONE |
| W1d — `VT_POOL_BORROW` on both documentation surfaces | `docs/ENVIRONMENT.md`, `docs/reference/engine-lifecycle.md` | DONE |
| W1e — records | this spec, the engine-matrix row, three issue-index rows, the claim | DONE |

## Stop conditions

- Stop and report `NEEDS_DECISION` if the borrow is found to change any token
  on any fixture.
- Stop if the borrow can be shown to reach a CUDA-graph capture that was
  previously pre-grown correctly (`Upstream chain` -> "Why this cannot break a CUDA-graph capture" says it cannot).

## Owed

- **O1 — the GPU attribution of #1922.** Operator-run, on `dgx:gpu0` under an
  `rc` lease: the Qwen3.8-27B NVFP4 + DFlash2 K=8 configuration at
  `--max-model-len 12288 --kv-cache-memory 6GiB --max-num-seqs 1
  --no-enable-prefix-caching`, N sequential 384-token completions, with the
  `avail` curve recorded and `DevicePool::stats()` read per request. This row
  claims nothing about that curve. #1922 stays open until it is measured, and
  it is what decides whether this change is the whole fix or the first part.
  The same lease owes the second half, which is what `Scope` says this unblocks
  rather than delivers: a c2/c4/c6 ladder on the same weights, with the memory
  curve recorded beside the throughput at each rung. Neither number is claimed
  anywhere in this spec.
- **O2 — the other shape-keyed process-lifetime caches (`Upstream chain` -> "What the residue is").** Issue
  [#1926](https://github.com/mudler/vllm.cpp/issues/1926).
- **O3 — no profiling forward at the worst-case shape.** vLLM sizes its
  allocator cache once, before serving, and we let production traffic discover
  our shapes. Owned by [#1165](https://github.com/mudler/vllm.cpp/issues/1165)
  (`gpu_memory_utilization` inert), whose "M3 profile run" is the same missing
  step.
- **O4 — `connector_stored_blocks_` is never erased.** `runner.h` /
  `runner.cpp`: a `request-id → int` map inserted into and never erased, so it
  grows for the life of the process, one entry per request. Inert unless a KV
  connector is installed, and far too small to be #1922, but it is an
  unbounded request-keyed container. Issue
  [#1927](https://github.com/mudler/vllm.cpp/issues/1927).

## Outcome

Written when the row reaches `DONE`.
