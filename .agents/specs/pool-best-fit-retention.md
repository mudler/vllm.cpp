# `ENG-POOL-BEST-FIT` — a freed scratch block must be reusable by a smaller request

**Issue:** [#1922](https://github.com/mudler/vllm.cpp/issues/1922) (open; this
row does not close it — see `## Owed` O1).
**Row:** `ENG-POOL-BEST-FIT`. **Base:** `main` @ `2e2b3fc1a`.
**Owning file:** this spec.
**Status at write time:** spec committed before the implementation commit, per
AGENTS.md "Spec before code".

## Now

`ACTIVE` on `row/ENG-POOL-BEST-FIT`, PR
[#1930](https://github.com/mudler/vllm.cpp/pull/1930), carrying the repairs to
the four findings of its fresh review. The change is CPU-gated.

[#1922](https://github.com/mudler/vllm.cpp/issues/1922) STAYS OPEN AND STAYS
UNEXPLAINED. This row fixes a real reuse defect in `vllm::DevicePool` that
SATURATES, and #1922's reported curve does not flatten, so the two are not the
same thing — `Honesty statement` carries the measurements and `## Owed` O1
carries the hunt for the mechanism that is.

## Honesty statement — what this row claims and what it does not

**Claimed, and measured on this tree.** `vllm::DevicePool` — the shared scratch
allocator every model forward draws from — could only ever hand a freed block
back to a request in the block's own size class. Twelve sequential requests
through the production `LoadedEngine::generate` entry, with the **largest
request first** so that every later request demanded strictly less than one
already served, still grew the pool's retained bytes 5.1× and made 240 further
driver allocations. That is measured, red-first, and it is what this row fixes.

**Not claimed, and this is now stronger than it was.** That this is the whole of
#1922 — and, since the fresh review of PR #1930 and the root-cause audit on the
issue, that it is any part of #1922's reported curve at all. #1922 reports host
`MemAvailable` falling about 2 GiB per request on `dgx:gpu0` with a
Qwen3.8-27B NVFP4 target and a DFlash2 draft at `--max-model-len 12288`. This
session had no GPU (the box was under another lease), so nothing here was run on
that configuration, and no `avail` curve is reproduced, attributed or claimed.

**THE MECHANISM THIS ROW FIXES SATURATES, AND #1922's CURVE DOES NOT.** Two
measurements settle it, and neither is this row's own:

- The fresh reviewer of #1930 ran twelve IDENTICAL requests: the defect does not
  fire at all — zero further driver allocations per request and flat retention,
  in BOTH arms — and replaying the descending suite three times costs 99 / 0 / 0
  driver allocations after the fix against 319 / 0 / 0 before it. Both arms
  plateau. What this row changes is the plateau's HEIGHT and how fast it is
  reached as a function of SHAPE COUNT. A retention that tracks concurrent need
  rather than shape variety is the correct behaviour and is worth landing, but
  it is a bounded quantity either way.
- #1922's own curve falls monotonically by about 2 GiB per request with no
  flattening whatever (48 → 41 → 39 → 37 → … → 11 GiB). A mechanism that
  provably plateaus cannot be that curve.

**AND THE `max_model_len` ARGUMENT IS WITHDRAWN.** An earlier version of this
paragraph claimed the mechanism "scales with `max_model_len` — which is the axis
#1922 reports". The root-cause audit on the issue shows `max_num_batched_tokens`
resolves to a flat 2048 in BOTH the 12288 and the 3072 configuration, so the
forward's scratch shapes do not move with `max_model_len` and nothing here
scales on that axis. The sentence was an inference, it was wrong, and it is
retracted rather than softened.

So: this row fixes a real reuse defect in `vllm::DevicePool`, on the serving
path, reachable from `LoadedEngine::generate`, and gated. It does not explain
#1922, it does not close #1922, and #1922's three actual terms are `## Owed` O1.

**Where the correction of record lives.** The `.agents/issue-index.md` row for
#1922 is append-only and carries `merge=union`, so it cannot be edited and must
not be. THIS SPEC is therefore the correction of record for every overstatement
this row made about #1922, and the index row is to be read through it.

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
serving path. The two changes have NO PRODUCT-CODE AND NO MECHANISM OVERLAP:
#1919 owns `src/vllm/model_executor/models/qwen3_dflash.cpp`, which this row
does not edit, and this row owns
`include/vllm/model_executor/models/device_pool.h`, which #1919 has no reason
to. Nor do they overlap in mechanism: #1919 is a fixed capacity that REFUSES and
kills the engine, this is a reuse failure that allocates and never gives back.
#1922 says the reproduction it reports stayed under #1919's cap, so neither is
the other.

An earlier version of this paragraph said the two "do not overlap in any file",
which is false and was corrected by the fresh review. Both branches touch
`.agents/issue-index.md`, `docs/ENVIRONMENT.md` and `tests/CMakeLists.txt`. Only
the first of those carries `merge=union`, so the other two are ordinary textual
conflicts that whoever lands second resolves. That is the claim that was
actually verified, and it is the one worth making: a record and a build-file
conflict is a merge, a product-code conflict would be a coordination problem.

They do meet at one place worth naming, and it is the pool: a store that
reallocates per request returns its old pools to this allocator, and whether
those bytes can then serve anything else is exactly what this row changes. If
#1919 sizes the store from `max_model_len`, the blocks it returns get larger,
and the borrow is what keeps them reachable.

The unrelated `connector_stored_blocks_` defect found in the same audit is
[#1927](https://github.com/mudler/vllm.cpp/issues/1927) and lives in
`src/vllm/v1/worker/gpu/runner.cpp`, which this row also does not edit, for the
same reason: a different behaviour with a different test surface.

**[#1919](https://github.com/mudler/vllm.cpp/issues/1919) / PR #1932 — THESE
TWO ARE NOT INDEPENDENT, and the meeting point is this allocator.** #1932 landed
as `c113886dc` and replaced the DFlash2 draft context store's 4096-slot constant
with a capacity resolved from `max_model_len`, capped at an 8 GiB AGGREGATE
device budget divided by `max_num_reqs`. Both changes govern device memory, so
the question was asked directly rather than assumed away, and the answer has
four parts.

**The store allocates through THIS pool.** `DflashDeviceKVStore::pool_k` and
`pool_v` are `std::vector<DBuf>`, and `DBuf` allocates with
`pool_->Get(*b_, alloc_bytes_)` and releases with `pool_->Put(*b_, alloc_bytes_,
p_, cap_)` (`include/vllm/model_executor/models/dense_device_glue.h`, the
constructor and destructor). So a store's per-layer K and V pools are ordinary
pooled scratch, and every byte #1932 sizes passes through the borrow this row
added. That is the coordination note this section already anticipated before
#1932 existed, now concrete.

**The borrow LOGIC is unchanged by the new sizing, because the ladder is
logarithmic.** `ClassOf` keeps the top `kClassBits` significant bits, so which
class serves a request of S bytes -- its own, or one at most `kBorrowMaxRatio`
above -- is the same relation at 4096 slots and at `max_model_len` slots. What
scales is the ABSOLUTE bound on transient over-hold: "at most twice what you
asked for" is a ratio, so on a store pool of tens of MiB the bounded waste is
tens of MiB where the old constant made it a few. The guarantee is unchanged;
its denomination is larger.

**The budget and the pool's retention do NOT double-count at c=32.** The
resolution is per ENGINE, not per request, so every store's per-layer pool is
byte-identical and lands in the SAME size class. A freed store's blocks are
therefore reused EXACTLY by the next store's `Get` -- an own-class hit, not even
a borrow -- so peak pool retention of store-class blocks equals peak concurrent
stores, which is precisely what #1932's 8 GiB aggregate bounds. There is no
second 8 GiB term.

**One asymmetry, and it predates this row.** The pool never returns a block to
the driver, and `device_pool_cap_bytes` is 0 -- uncapped -- on every platform
today (`include/vllm/platforms/interface.h`). So after ONE burst at
`--max-num-seqs 32`, the aggregate #1932 bounds as a PEAK becomes a residency
FLOOR for the life of the process, even if concurrency later drops to 1. That is
pre-#1922 pool behaviour and this row does not change it; the borrow only makes
those retained blocks reachable by smaller requests, which strictly REDUCES
total footprint. Worth naming for whoever reads #1932's startup line and expects
the memory back when the load falls, and it is the argument for eventually
setting a non-zero cap (`Rejected`, second entry).

**One genuine coupling the borrow introduces, bounded and self-correcting.**
While a store-class block is lent to a smaller consumer, that class's free list
is empty, so a store rebuild at the same class MISSES and asks the driver -- one
block above the intended aggregate. It is bounded by one block per class per
concurrent borrow, it needs a non-store request of at least half a store pool to
miss its own class inside the window between a store's destruction and the next
store's construction, and it resolves itself because `block_class_` returns the
borrowed block to its OWN class rather than demoting it. Nothing here refuses an
allocation, so this is a transient overshoot of an intended budget and not an
OOM path, but it is the one place the two changes are genuinely entangled and it
is recorded rather than reconciled quietly. Neither change needs an edit for it;
whether it matters at c=32 is a question for the same operator ladder O1 owes,
which is the run that would see it.

**[#1946](https://github.com/mudler/vllm.cpp/issues/1946) — no interaction, and
that is checked rather than assumed.** #1946 is the DFlash2 draft holding a
second 2.54 GB device copy of the target's embed table, because `ResidentWeight`
caches its upload per-`OwnedTensor` and the draft's embed is a distinct one. It
lands in the same device-memory picture as this row and touches none of it:
`ResidentWeight` calls `d.b.Alloc(nb)` and frees through a `shared_ptr` deleter
that calls `bk->Free(q)` (`include/vllm/model_executor/models/dense_attn_block.h`,
the `if (!w.d_dev)` branch). Both are the BACKEND directly. A resident weight
therefore never enters this pool's free list and never draws from it, so the
borrow cannot hand a resident's block to a scratch request and cannot lengthen a
resident's lifetime. Named here because the question was asked, and a negative
answer that was actually looked up is worth more than silence.

**[#1945](https://github.com/mudler/vllm.cpp/issues/1945) — this row changes the
aliasing pattern it describes, and does not make it safe.** #1945 is the DFlash2
per-request CUDA-graph capture returning graph-baked scratch to this shared
pool while those device addresses stay live inside the captured graph. The
borrow changes WHICH block a `Get` of a given size is answered with, so a block
that previously went to a same-class caller can now go to a smaller one, and the
set of callers that can end up aliasing a graph-resident address moves with it.
That is a change in the pattern, not in the guarantee: the defect is that the
addresses are returned at all, and it is neither introduced nor repaired here.
Whoever takes #1945 should read the borrow as one more reason the fix belongs at
the return site rather than in the allocator. Named here so the interaction is
on the record before anyone bisects into it.

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
1b. **The boundary case (added by the fresh review of #1930).** The same
   isolated pool, with the two requests that sit on either side of the one class
   where the borrow's answer changes: `big / 2` must be served from a retained
   `big` block and `big / 2 - big / 64`, one rung of the ladder lower, must not.
   Every size is a literal and no assertion names `kBorrowMaxRatio` or
   `kBorrowMaxSteps`, because a gate that computes its expectation from the
   constant under test passes whatever that constant becomes.

1c. **The concurrent case (added by the fresh review of #1930),** in
   `tests/vllm/models/test_device_pool.cpp` rather than here, because its
   subject is the pool and not the engine. Eight threads over a four-octave
   ladder: a barrier phase in which all eight hold a block simultaneously, so
   contention is a property of the case and not of the scheduler, then
   unsynchronised churn. Asserts no double issue, no lost block (a sequential
   replay must reach the driver zero times), and that `Drain` recounts
   `retained_` to the same bytes. Registered a second time as
   `test_device_pool_concurrent` with `VT_POOL_BYPASS=0`, or the sanitizer lane
   would never run it.

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

### The `dc3bbe8cd` reconciliation — four records, three treatments

`origin/main` advanced four times while this branch's CI ran. The resolution is
recorded because three of the four files must NOT be resolved by a three-way
merge, and one of them would have been resolved wrongly with no conflict at all.

| file | treatment | result |
|---|---|---|
| `.agents/issue-index.md` | append-only, union-merged, verified by row-ID SET DIFFERENCE | base 710, ours 713, theirs 713, merged 716; 0 lost, 0 invented, 0 duplicated, 0 duplicate issue IDs |
| `.agents/engine-matrix.md` | keyed record: take `origin/main` complete, verify byte-identical, re-apply this row's key at an anchor asserted unique | 172 -> 173 keyed rows, exactly 1 key added, 0 removed, 0 unrelated key's text changed |
| `docs/ENVIRONMENT.md` | same | one added line, `VT_POOL_BORROW`, and nothing else |
| `scripts/check-agent-record.py` | CHECKER: read both sides for MEANING before resolving | neither side changed a rule; the pin is reconciled to the union and mutation-proved |

**THE COUNTER IS THE PART A THREE-WAY MERGE GETS WRONG, and it is silent.**
[#1925](https://github.com/mudler/vllm.cpp/issues/1925) and #1922 each added ONE
engine row on its own branch, and each bumped `ENGINE_ROWS` from 171 to 172. Both
sides therefore carry an **identical** `ENGINE_ROWS = 172` line, so git resolves
it with no conflict -- counting one of the two new rows and dropping the other.
The union is 173. The area cells move in DIFFERENT columns because the two rows
have different states: `KV cache and memory` goes 25 -> 27 rows with `READY`
3 -> 4 (theirs) and `ACTIVE` 4 -> 5 (ours). Taking either side whole miscounts.

The checker was mutated rather than trusted, on the merged tree:

| mutation | observed |
|---|---|
| pin 173 -> 172 (exactly what the silent merge produces) | `exit=1`, `173 engine rows; expected 172` |
| pin 173 -> 171 (the value both branches started from) | `exit=1`, `173 engine rows; expected 171` |
| `!=` relaxed to `>` with the pin at 999 | `exit=0` -- what a quietly permissive version looks like, and NOT what landed |
| restore | `exit=0`, file byte-identical (sha `19345ade582ba794`) |

`tests/scripts/test_agent_record.py` auto-merged; both sides' additions survive
and both sit ABOVE the `if __name__ == "__main__"` guard, because a test class
below it never runs.

**One consequence worth naming, because the correct-by-policy path caused it.**
Taking the matrix from `origin/main` reinstated `ENG-RECORD-ANCHOR-RATCHET`'s
citations at `tests/scripts/test_agent_record.py:1423` and `:1491`. Both sides
added cases to that file -- theirs at the end, ours a 26-line method at line 346
-- so in the MERGED tree those symbols sit at 1449 and 1517, and the anchor
ratchet caught it as a rise from 31 stale to 33. Measured on the side:
`origin/main` alone reports `stale=31` and is green, so the rise was an
interaction and not something inherited. Repaired to the merged tree's real
lines, which is what this branch already cited before the merge.
`check-agent-record.py` then exits 0 at `ENGINE=173 ... ANCHOR-ROT=37`, and
`tests/scripts/test_agent_record.py` is 116 tests `OK`.

**How the merged tree was gated, and one harness defect to know about.** The
box could not hold the whole suite's binaries (~590 x ~45 MB against 9.4 GB
free), so the suite was built and run in disk-bounded batches: build 40 targets,
run their tests, delete those binaries, repeat. Same configure, same commands,
peak disk one batch instead of the whole suite. 625 tests, `GATE_EXIT=1` on
exactly one: `test_qwen35_paged_engine_prerequisites`, which is a CMake driver
that invokes `test_qwen35_paged_engine` -- a binary an earlier batch had already
deleted. It reported `expected exit 77, got 1 ... No such file or directory`,
which is the harness's defect and not the tree's. Rebuilt and re-run alone:
`PREREQ_EXIT=0`, Passed. Anyone reusing that batching technique must keep the
binaries that other tests invoke, because a deleted dependency reads as a
product failure.

### The fifth finding, which came from CI rather than from a reviewer

**Both sanitizer lanes were RED on the reviewed head, on this row's own test,
and the review did not see it.** Run `32889401375` at `414de4a73`:
`sanitize-cpu (thread)` and `sanitize-cpu (address,undefined)` each report
`99% tests passed, 1 tests failed out of 622`, and the one failure is
`299 - test_engine_scratch_steady_state`.

The cause is not a race and not a leak. `ci.yml` gives the sanitizer job
`VT_POOL_BYPASS: "1"`, and it is right to: the pool deliberately retains scratch
blocks and ASan's leak detector cannot tell that cache from a leak. But under
bypass every `Get` is an exact driver allocation and every `Put` a real `Free`,
so there is no free list at all -- the job's log shows
`retained 0 B, driver allocations 0, distinct classes 0` for all twelve requests
and `CHECK( 0 == 1 )` at `misses_after_first == 1`. Every case in the file has
the free list as its subject, so bypass makes the whole file unanswerable.

**The first repair was a SKIP, and it was wrong.** `RequirePool()` exited 77 so
CTest reported **Skipped**. That is the move
`.agents/verification.md` warns about from the other side: a skip guard turns a
red into a green that measured nothing, and it would have left the pool's
steady-state behaviour silently unexercised under ASan and TSan for good. The
correct repair is to supply the missing precondition, not to excuse the
assertion. Recorded here because the wrong version was committed first and the
reasoning is the useful part.

**The repair that landed: the sanitizer lanes RUN this gate, pooled.**
`tests/CMakeLists.txt` pins `ENVIRONMENT "VT_POOL_BYPASS=0"` on the test's CTest
registration, the same technique `test_device_pool_concurrent` already used.

What made that possible was a second defect this exposed: **the case never gave
its blocks back.** The pool does not return a block to the driver by design, so
running the case pooled under `detect_leaks=1` reports
`SUMMARY: AddressSanitizer: 3362944 byte(s) leaked in 103 allocation(s)` while
every assertion passes -- a process that exits 1 after printing
`Status: SUCCESS!`. 55 of the 59 reports came from the engine case's
process-wide pool and the rest from the two directly-constructed pools, which
have no owner to outlive them. The engine is now scoped so it is destroyed
first, and each case drains what it owns.

| lane | command | result |
|---|---|---|
| ASan+UBSan, pooled, BEFORE the drain | `ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 VT_POOL_BYPASS=0` | `EXIT=1`, 18/18 `SUCCESS`, `3362944 byte(s) leaked in 103 allocation(s)` |
| ASan+UBSan, pooled, AFTER the drain | same | `EXIT=0`, 19/19 `SUCCESS`, `drained 624995 B` |
| TSan, pooled | `VT_POOL_BYPASS=0` | `EXIT=0`, 19/19 `SUCCESS`, **zero** ThreadSanitizer warnings |
| ASan+UBSan, `test_device_pool_concurrent` | same options | `EXIT=0`, 22/22 `SUCCESS` -- checked because that entry is registered pooled too |

**The one cost, stated rather than left implicit.** For these two binaries only,
ASan can no longer see a use-after-free of a released `DBuf`, because a pooled
block stays mapped. That is exactly the trade `device_pool.h` documents under
its BYPASS lane; it applies to no other binary in the suite; and the two
binaries in question exist to measure the pool. Nothing is owed for it, and no
issue is needed, because no coverage is lost that the rest of the suite does not
still have.

`RequirePool()` is kept as a hand-run backstop, not as the repair. In CI it
never fires. It is there for whoever runs the binary directly under the bypass
lane that `test_device_pool.cpp`'s header recommends as a debugging
discriminator: they get one named skip instead of eight confusing assertion
failures.

### The review-repair round (the four findings on PR #1930)

Tree: `row/ENG-POOL-BEST-FIT` at the repair head, same configuration as above.
The focused suite grew from 12 assertions in 2 cases to 18 in 3, and
`test_device_pool` from 10 cases to 11.

| arm | result | numbers the run printed |
|---|---|---|
| `test_engine_scratch_steady_state`, default | 18/18, `SUCCESS` | retained 536 403 B after the peak request, 624 899 B after 12; 72 driver allocations for the peak request, 27 for the 11 smaller ones — every figure identical to the pre-repair run |
| `test_engine_scratch_steady_state`, `VT_POOL_BORROW=0` | 10/18, `FAILURE` | `CHECK( 240 <= 79 )` |
| `test_device_pool`, default | 11 cases, 59/59, `SUCCESS` | — |
| `test_device_pool`, `VT_POOL_EXACT=1` | 11 cases, 60/60, `SUCCESS` | — |
| `test_device_pool`, `VT_POOL_BORROW=0` | 11 cases, 59/59, `SUCCESS` | the concurrency invariant must not be lane-specific |
| `test_device_pool`, `VT_POOL_BYPASS=1` | 11 cases, 24/24, `SUCCESS` | the concurrent case skips itself, which is why it is registered a second time |

ThreadSanitizer, `-DVLLM_CPP_SANITIZE=thread`, `test_device_pool`: 11 cases,
59/59, `SUCCESS`, and **zero** `WARNING: ThreadSanitizer` lines. The instrument
was checked rather than trusted — see M7 below, which produces 96 of them on the
same binary. The run needs `setarch x86_64 -R` on this box: without it the TSan
runtime aborts at startup with `FATAL: ThreadSanitizer: unexpected memory
mapping`, which is this kernel's ASLR entropy and not a property of the tree. CI
runners do not need it.

| # | mutation | expected | observed |
|---|---|---|---|
| M4 | `kBorrowMaxRatio` 2 -> 16, `kBorrowMaxSteps` UNTOUCHED | the tree must refuse | build `rc=1`, `device_pool.h:515: error: static assertion failed: DevicePool: kBorrowMaxSteps and kBorrowMaxRatio must state the SAME bound...`. **Before this repair the same edit was completely undetected: 12/12 `SUCCESS`.** |
| M5 | `kBorrowMaxSteps` 16 -> 32, `kBorrowMaxRatio` UNTOUCHED | the tree must refuse | build `rc=1`, the same assertion. The pair cannot now be widened from either end |
| M6 | `probe > limit` -> `probe >= limit` | the boundary case, and only it, reds | 16/18, `FAILURE` at `CHECK( inside == held )` and `CHECK( pool.stats().misses == after_alloc + 1 )`, in `device pool: the borrow reaches exactly one octave, and not one class further`. The other two cases stayed green |
| M7 | `Put`'s `std::lock_guard<std::mutex> lk(mu_)` deleted | the concurrent case reds | plain build: `SIGSEGV`, doctest reporting `test case CRASHED` in the concurrent case with the other ten cases untouched. Under TSan: 96 `data race` reports, the first naming `vllm::DevicePool::TakeBlockClass` <- `vllm::DevicePool::Put` <- the concurrent case's barrier-phase `pool.Put` (the run printed `test_device_pool.cpp:713`) — a read of the `block_class_` map this row added |
| restore | — | green | header restored to sha256 `232f440ac0c6eaa5439881b8e833ea4c6aa57ded4d8873de270620c23485caf5`, `git status` clean, rebuilt and relinked, 18/18 and 59/59 `SUCCESS` |

Every mutation above was rebuilt AND relinked before it was read. That is not
ceremony: a `.o` that rebuilt while its executable had not yet relinked gave the
fresh reviewer of this row a false red from a stale binary, and a mutation whose
BUILD fails reads as a passing test unless you look at the build status.

The registration check, because a skip here would be invisible:
`VT_POOL_BYPASS=1 ctest -R test_device_pool_concurrent -V` reports
`test cases: 1 | 1 passed`, `assertions: 22 | 22 passed`, while
`VT_POOL_BYPASS=1 ./build/tests/test_device_pool --test-case='*concurrent*'` —
the same case WITHOUT the CTest entry's override — reports the skip message and
`assertions: 0 | 0 passed | 0 failed`. The override is doing the work, and the
difference between the two lines is the whole reason the entry exists.

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

The two sanitizer reds on `414de4a73` are this row's own and are fixed above,
under `The fifth finding`. They are not pre-existing and were not inherited.

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
- **Transient over-allocation up to 2×** while a borrowed block is held. The
  bound is DELIVERED by `kBorrowMaxSteps` — one octave of the class ladder — and
  only STATED by `kBorrowMaxRatio`; the `probe > limit` test cannot fire first
  for any request of at least `1 << kClassBits` bytes, which is why the two
  constants are now held together by a `static_assert` and the gate asserts a
  literal. Before that, raising `kBorrowMaxRatio` from 2 to 16 alone left the
  focused gate 12/12 `SUCCESS` and widened its own threshold to tolerate 13.7×
  growth.
- **Two hash operations per `Get`/`Put`**, under a mutex the pool already takes.
  The measured hit rate rises (fewer backend calls), so the allocator does
  strictly less work, not more.
- **The gate's thresholds are design constants, not fitted numbers** — the
  borrow ratio and "no more than the peak request's own allocations". The
  measured margins (1.16 vs 5.12, 27 vs 240) are wide.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 4 --target test_engine_scratch_steady_state test_device_pool
./build/tests/test_engine_scratch_steady_state                    # must PASS
VT_POOL_BORROW=0 ./build/tests/test_engine_scratch_steady_state   # must FAIL
./build/tests/test_device_pool                                    # must PASS
VT_POOL_BYPASS=1 ctest --test-dir build -R test_device_pool_concurrent -V
                                # must PASS, and must report assertions > 0
ctest --test-dir build -R '^test_engine_scratch_steady_state$' -V
                                # the registration must pin VT_POOL_BYPASS=0, so
                                # this RUNS pooled even in a bypassed lane
cmake --build build -j 4
ctest --test-dir build --output-on-failure
scripts/agent-preflight.sh --staged
```

The fourth command is not decoration. `sanitize-cpu` runs the whole suite with
`VT_POOL_BYPASS=1`, under which the concurrent case skips itself and doctest
prints `assertions: 0 | 0 passed | 0 failed` and `Status: SUCCESS!` — the exact
shape `vllm_cpp_add_test`'s own comment calls out. The `test_device_pool_concurrent`
CTest entry overrides the variable back to `0` for that one case, and running it
with the job's value in the environment is how you check that the override is
still doing its work. A pass with zero assertions there is a FAILING gate.

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
| W2a — the borrow bound is derived, not asserted: `static_assert` over `kBorrowMaxRatio`/`kBorrowMaxSteps`/`kClassBits`, literal thresholds in the gate, the boundary case | `include/vllm/model_executor/models/device_pool.h`, `tests/vllm/v1/worker/test_engine_scratch_steady_state.cpp` | DONE |
| W2b — the concurrent case, and the CTest entry that makes the sanitizer lane run it | `tests/vllm/models/test_device_pool.cpp`, `tests/CMakeLists.txt` | DONE |
| W2c — the saturation correction and the O1 rescope | this spec, the test header, the pull request body | DONE |
| W2d — the sanitizer lanes' red: the gate now RUNS there with the pool pinned on, and drains so LeakSanitizer sees a clean exit | `tests/vllm/v1/worker/test_engine_scratch_steady_state.cpp`, `tests/CMakeLists.txt` | DONE |

## Stop conditions

- Stop and report `NEEDS_DECISION` if the borrow is found to change any token
  on any fixture.
- Stop if the borrow can be shown to reach a CUDA-graph capture that was
  previously pre-grown correctly (`Upstream chain` -> "Why this cannot break a CUDA-graph capture" says it cannot).

## Owed

- **O1 — find the SECOND, NON-SATURATING mechanism. #1922 is not this row.**
  The operator run is no longer a confirmation of the borrow; it is a hunt for
  something else, and it must be dispatched that way. The reason is in
  `Honesty statement`: the mechanism this row fixes provably plateaus (12
  identical requests do not fire it at all; a 3x replay of the descending suite
  costs 99 / 0 / 0 driver allocations), while #1922's curve falls about 2 GiB
  per request with no flattening. A run that goes looking for confirmation of
  the borrow will find the borrow behaving correctly and will learn nothing.

  The root-cause audit on the issue decomposes the reported curve into THREE
  separate terms, and each is a different question:

  1. **about −7 GiB inside request 0** — lazy weight residency. `WarmupKernels`
     is guarded on `uses_nvfp4_w4a4()` and this checkpoint is W4A16, so the
     warm-up never runs and the first real request pays the residency. A
     one-time step, not a per-request slope.
  2. **about −2 GiB per request** — the slope, and the term that matters. The
     leading candidate is the retired-scratch list behind
     `RetireGraphScratch`: `detail::RetiredGraphScratchList()` at
     `src/vt/cuda/graph_safe_scratch.h:44`, a `static std::vector<void*>`
     documented in place as "process-lifetime; captured-graph pointers stay
     valid", with `RetireGraphScratch` itself at `:58`. It is PUSH-ONLY, and its
     `Ensure*` callers grow to the exact requested size, so what it retains is
     the SUM of every block each site has ever outgrown rather than the maximum
     of them. Instrument it first.
  3. **about −26 GiB during a request that emitted ZERO tokens** — most likely
     the box thrashing rather than this process at all. It needs per-PID
     attribution before it is called anything; a system-wide `MemAvailable`
     cannot tell one process's growth from another's.

  Two facts constrain the search and should be carried into it. First,
  `max_num_batched_tokens` resolves to a flat 2048 in BOTH the 12288 and the
  3072 configuration, so NOTHING in the forward's scratch scales with
  `max_model_len` and any hypothesis resting on that axis is already refuted.
  Second, the terms are additive: a single per-request number cannot separate
  them, so the run records `MemAvailable`, per-PID RSS, and
  `DevicePool::stats()` per request, not just the first.

  Beyond term 2's leading candidate, the containers worth instrumenting on the
  device — all of them per-request and none of them saturating by construction:

  - per-request CUDA-graph captures keyed on a shape that varies with the
    request, so the capture set grows with shape variety;
  - the DFlash2 draft store's per-request pools
    ([#1919](https://github.com/mudler/vllm.cpp/issues/1919)'s area), at 256 MiB
    each;
  - `connector_stored_blocks_`
    ([#1927](https://github.com/mudler/vllm.cpp/issues/1927)), a request-keyed
    map that is inserted into and never erased;
  - anything else sized from `max_model_len` and allocated PER REQUEST rather
    than once at startup.

  Configuration, unchanged: on `dgx:gpu0` under an `rc` lease, Qwen3.8-27B
  NVFP4 + DFlash2 K=8 at `--max-model-len 12288 --kv-cache-memory 6GiB
  --max-num-seqs 1 --no-enable-prefix-caching`, N sequential 384-token
  completions. #1922 stays open and stays UNEXPLAINED until that run lands.

  The same lease owes the second half, which is what `Scope` says this unblocks
  rather than delivers: a c2/c4/c6 ladder on the same weights, with the memory
  curve recorded beside the throughput at each rung. Neither number is claimed
  anywhere in this spec. That ladder is also the first traffic ever to run this
  pool at `--max-num-seqs > 1`, which is why the concurrent case in
  `tests/vllm/models/test_device_pool.cpp` had to exist before it.
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
