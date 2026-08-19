# `ENG-EXPERT-STREAM-DEVICE` — a device-reachable destination for streamed expert slices

Issue [#1124](https://github.com/mudler/vllm.cpp/issues/1124). Owning row
`ENG-EXPERT-STREAM-DEVICE` ([engine-matrix.md](../engine-matrix.md), KV cache and
memory). Parent row `ENG-EXPERT-STREAM`, spec
[expert-streaming.md](expert-streaming.md), which owns the streaming MECHANISM —
the cache, the streamer, the `pread` filler and the host store — and lists this
capability under its `## Owed`. This row owns only WHERE a slice lives and WHICH
platform may read it.

## Now

`READY`, spec committed, no implementation. W0 is the critical path: today
`--device cuda` on `Qwen3.8-2.4T-A95B UD-Q1_0` refuses at load by design
(`76f2a6d84`, issue #1123), so **there is no GPU number for this checkpoint to
take at all**, and nothing downstream of W0 produces one on the hardware this
project owns. The developer's target is that GPU figure.

## Scope

**In scope.**

* **W0 — the integrated/unified path.** Let a platform whose host memory is
  device-addressable read expert slices out of the existing host slot store, so
  `--device cuda` serves this checkpoint on a GB10 instead of refusing. Keyed on
  a PROBED property, never on a device name, an architecture string or a
  compute capability.
* **W1 — the device slot store.** A second production `ExpertSlotStore` whose
  slots are device allocations, with the fill contract that makes filling one
  possible at all.
* **W2 — the device-capable read.** A virtual read on `ExpertSlotStore`, so the
  seam `Qwen35ExpertStream::Slice` reads through stops being the concrete
  `HostExpertSlotStore`.

**Out of scope, and owed elsewhere.** Asynchronous prefetch and miss/compute
overlap (`ENG-EXPERT-STREAM` W6, conditional on a measurement). Composing
streaming with the grouped keep-quant MoE path. A Windows filler. A zero-copy
device DMA filler (GPUDirect Storage / `cuFile`) — see the W1 design note, which
records why a staging bounce is chosen instead and what would justify replacing
it. Families other than Qwen3.5 MoE. The KV/activation term of the load-time fit
bound, which `KV-WARMUP-PROFILE` owns.

**Verdict on issue #1124's piece 3.** #1124 lists "the filler is `pread`-into-host"
as a fourth independent piece. It is not independent and it is not a wave: a
device store cannot be filled AT ALL through today's contract, because
`ExpertSlotStore::SlotForWrite` returns a pointer that `ExpertStreamer::EnsureFile`
hands straight to `::pread`, and `vt::Backend::DeviceMemoryIsHostAddressable`
is false for CUDA, so that pointer is not host-writable. W1 therefore carries a
fill-contract change or W1 cannot land. Planning it as a third wave after W1
would plan a wave that deadlocks its predecessor. It is folded into W1 below.

## Upstream chain

**There is no oracle for this row, and that is a measured statement rather than
a search that came up empty.** Pinned vLLM `555967922` has inference-time expert
paging nowhere: `vllm/model_executor/offloader/uva.py:21` is a CPU-blanket UVA
offloader over whole parameters, and `vllm/model_executor/offloader/prefetch.py:557-560`
is cpu-only. The secondary-oracle table does not rescue it either — llama.cpp's
`-ot` / `-ncmoe` (`common/arg.cpp:2451-2478` @ `237ad9b96`) moves expert COMPUTE
to the host, which is `ENG-HYBRID-PLACEMENT`'s inverse design and not a slot
store. `docs/BENCHMARKS.md` already records the parent row as "correct answer;
no oracle runs this".

So the reference for correctness is **our own CPU arm on the same checkpoint and
the same binary**, which is the shape `ENG-EXPERT-STREAM` already gates on, and
every gate below is stated against it. Nothing here is mirrored from vLLM, and
nothing here may claim to be.

The one upstream surface this row DOES mirror is the platform seam it extends:
`vllm/platforms/interface.py:914` + `vllm/platforms/cuda.py:675`
`is_integrated_gpu`, already ported as `platforms::Platform::is_integrated_gpu`
and recorded in that header as "unwired today". W0 is what wires that family of
predicates, and it adds one sibling to it rather than a bespoke test at the call
site — the same discipline `needs_weight_staging` and `supports_fa2_attention`
were hoisted under (accelerator-seam audit §12.3).

## Our baseline

Every number below is re-verified against the tree at `fe24a3029` and against
the census recorded in [expert-streaming.md](expert-streaming.md) (both GGUF
tensor tables at revision `567d3e6ac26c5474b18311e619c04350fb9a5556`, 1702
records parsed against 1702 declared).

| Quantity | Value |
|---|---|
| one IQ1_XXXS expert tower `[E*N,K]`, E=512 | 1,275,068,416 B = 1.1875 GiB |
| all 279 `*_exps` towers | 360,374,599,680 B = **335.62 GiB** |
| whole tensor table (the fit bound's scope) | 397,245,341,184 B = 369.96 GiB |
| non-expert remainder | 36,870,741,504 B = 34.34 GiB |
| ... minus the MTP / `nextn` block a default load never stages | 27,930,252,800 B = **26.01 GiB** |
| one expert slice = one slot | 2,490,368 B = 2.375 MiB |
| slices per decoded token (93 layers x 10 experts x 3 matrices) | 2790, i.e. 6,948,126,720 B = **6.95 GB (6.47 GiB) per token** |
| device pool, `cudaMemGetInfo` total on `dgx:gpu0` | 128,452,956,160 B = 119.631 GiB, exactly `/proc/meminfo MemTotal` x 1024 |

### The four pieces, re-derived at `fe24a3029`

1. **No device store.** `include/vllm/model_executor/host_expert_slot_store.h::HostExpertSlotStore`
   is the only production `ExpertSlotStore`; the only other subclass is a test
   double. `include/vllm/model_executor/expert_streamer.h:8-9,30-31` says "the
   production destination is a contiguous device-side slot array" and "production
   writes to device memory". **Both sentences are false today**, and W1 makes them
   true rather than adding a second claim beside them.
2. **No device-capable read.** `Qwen35ExpertStream` holds
   `std::unique_ptr<HostExpertSlotStore> store_` (`qwen3_5.cpp:5621`) and reads
   `store_->Slot(r.slot)` at `:5381` and `:5437` — the CONCRETE class. There is no
   `SlotForRead()` on the interface, so the seam cannot be swapped even once a
   device store exists.
3. **The filler is `pread`-into-host.** `ExpertStreamer::EnsureFile` takes
   `store_.SlotForWrite(acq.slot)` and passes it to `::pread`
   (`src/vllm/model_executor/expert_streamer.cpp:76-94`). Not host-writable on a
   device allocation; see the scope verdict above.
4. **The consumer is CPU-gated.** `qwen3_5.cpp::KqExpertSlice` takes the slot arm
   only under `GetPlatform(d.q.device.type).is_cpu()` (`:5714`); every other
   platform falls through to `KqResidentSlice` (`:5218`) and stages the whole
   tower.

### Two facts #1124 does not name, and W0 fails without both

**(a) The slot arm itself stages the tower.** Inside `KqExpertSlice` the slot
branch calls `ResidentWeight(d, w)` — solely to inherit dtype, device and repack
markers — and then overwrites `.data` with the slot pointer. On CPU
`ResidentWeight` aliases and costs nothing. On a staging platform it takes the
upload branch, `void* p = d.b.Alloc(nb)` at `qwen3_5.cpp:1095` with
`nb = w.bytes.size()`, which is **the whole 1.1875 GiB stacked tower** — and
memoizes it in `w.d_dev`, so the very first streamed slice allocates what
streaming exists to avoid. That is the identical allocation #1123 died on
(48 towers, partway through layer 16 of 93). Lifting the `is_cpu()` guard alone
therefore reproduces #1123 rather than fixing it. W0 must build the slot tensor
WITHOUT `ResidentWeight`, and must refuse loudly if any `*_exps` tower reaches
`ResidentWeight` while streaming is on.

**(b) The load-time refusal fires first.** `CheckDeviceWeightFit` runs in
`src/vllm/entrypoints/model_loader.cpp` before the tokenizer and before any
weight I/O, gated on `target.needs_weight_staging()`, which is `true` on CUDA
including GB10 — the header at `include/vllm/platforms/interface.h:248-258` says
so explicitly and says why (`is_unified_memory()` / `is_integrated_gpu()` are the
WRONG predicate for staging, because GB10 is physically unified and the CUDA path
still binds distinct device pointers). The bound sums the WHOLE tensor table,
including all 335.62 GiB of `*_exps`, so it refuses this checkpoint before any
forward exists to take the slot arm. **W0 must make the refusal conditional on
the streaming lane serving those towers, or W0 produces no run.** This is the
single reason "lift one guard" is not one guard, and it is scoped here rather
than discovered during implementation.

The arithmetic that makes the conditional refusal EXACT rather than a fudge
factor: with the lane on, the staged set is the non-expert remainder plus the
slot arena, both known at load. Default load, 8000 slots:
26.01 GiB + 8000 x 2,490,368 B (18.55 GiB) = **44.56 GiB against a 119.631 GiB
pool**. No headroom fraction is invented, and the over-count direction the
existing bound documents (issue #1136) is unchanged.

### The CUDA kernel exists

`src/vt/cuda/cuda_quant_dot.cu` carries `WType::kIQ1_XXXS` through
`LaunchGemm`, `LaunchGroupedGemm` and `LaunchGroupedFusedSwiGLU`
(`:1874,1967,2182`), transcribed from the pinned `llama-cpp-unsloth` fork
oracle. So the encoding holding 96.92 % of this checkpoint has a CUDA GEMM, and
W0 is not blocked on a missing kernel. This is stated because it is the first
thing that would have killed the wave.

### The prefill exhaustion fact any W0 harness must handle

The last two-arm run recorded `exhausted=7813` on an 8000-slot cache, identical
at steps 2, 3 and 4 — all of it in prefill. That is not a defect and no slot
count fixes it. One step is one forward, prefill processes the whole prompt in
one forward, and slices acquired within a step are protected from eviction, so
the peak protected set for a T-token prompt is

    93 layers x 3 matrices x min(512, 10*T) experts

which saturates at 142,848 slices = **331 GiB** for any T >= 52 — the whole
model. A decode step touches ~2790 and exhausts nothing. A gate that reads TOTAL
`exhausted` therefore reports a red for a healthy lane. See `## Gates` G0-LIVE
for the two admissible harness shapes and which one is primary.

## Port map

Nothing is ported; there is no upstream. This is the local change map.

| Wave | Surface | Change |
|---|---|---|
| W0a | scratch probe, no tree change | print `cudaDevAttrPageableMemoryAccess`, `cudaDevAttrPageableMemoryAccessUsesHostPageTables`, `cudaDevAttrIntegrated`, `cudaDevAttrConcurrentManagedAccess` on `dgx:gpu0` |
| W0b | `include/vllm/platforms/interface.h`, `src/vllm/platforms/cuda.cpp`, `src/vllm/platforms/rocm.cpp` | new `virtual bool host_memory_is_device_addressable() const { return false; }` beside `is_integrated_gpu`; CUDA overrides from a probe taken once at registration next to the existing `cudaDevAttrIntegrated` probe; ROCm overrides from the `pageable_memory_access` capability it ALREADY probes (`rocm_backend.hip:96-103`) |
| W0c | `src/vllm/model_executor/models/qwen3_5.cpp` | `KqExpertSlice` takes the slot arm under `is_cpu()` OR `host_memory_is_device_addressable()`; the slot branch builds its tensor without `ResidentWeight`; a named `VT_CHECK` in `ResidentWeight` refuses a streamed `*_exps` tower reaching device staging |
| W0d | `include/vllm/model_executor/model_loader/gguf_device_fit.h`, `src/.../gguf_device_fit.cpp`, `src/vllm/entrypoints/model_loader.cpp` | the fit bound gains an explicit "these tensors are served by the slot lane, and the arena costs this instead" input; the loader passes it when the resolved config says streaming is on and the platform can read host slots |
| W1 | `include/vllm/model_executor/expert_streamer.h`, new `include/vllm/model_executor/device_expert_slot_store.h` | `CommitSlot(int32_t, size_t)` on `ExpertSlotStore` (no-op on the host store); `DeviceExpertSlotStore` allocating slots through `vt::Backend::Alloc` with one pinned host staging slot, `SlotForWrite` returning staging and `CommitSlot` doing the H2D; correct the two false sentences in `expert_streamer.h` |
| W2 | `expert_streamer.h`, `host_expert_slot_store.h`, `device_expert_slot_store.h`, `qwen3_5.cpp` | `virtual uint8_t* SlotForRead(int32_t)`; `Qwen35ExpertStream::store_` becomes `std::unique_ptr<ExpertSlotStore>`; `:5381` and `:5437` read through the virtual; the store is selected from the platform |

## Tests to port

**None.** There is no upstream implementation, so there is no upstream test
suite, no fixture, no tolerance and no revision anchor to preserve. Recording
that explicitly is the requirement; inventing a "ported" label for locally
written tests would falsify the porting inventory. Scratch-written tests are
recorded as such in `porting-inventory.md` §9.

The tests this row WRITES, red-first, with the defect each is proven against by
mutation:

| Test | Proves | Mutation that must red it |
|---|---|---|
| `tests/vllm/platforms/test_platform.cpp` (extend) | the new predicate defaults false and the CUDA/ROCm assembly threads the probed value | flip the default to true; drop the assignment |
| `tests/vllm/model_executor/test_expert_slot_store.cpp` (new) | a device-flavoured store filled via `EnsureFile` yields byte-identical slot content to the host store | delete `CommitSlot`'s copy; return staging from `SlotForRead` |
| `tests/vllm/model_executor/test_gguf_device_fit.cpp` (extend) | with the lane on, the bound excludes `*_exps` and adds the arena; with it off, the bound is byte-identical to today | make the exclusion unconditional |
| `tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp` (extend) | the loader reaches the conditional refusal from the production entry point | delete the production call site |
| a `qwen3_5` slot-arm unit gate | the slot branch never calls `ResidentWeight`, and a streamed tower reaching device staging throws by name | remove the `VT_CHECK`; restore the `ResidentWeight` call |

## Gates

Correctness first, always. No speed number from this row is admissible until
G0-CORRECT and G0-LIVE both pass in the SAME run that produced it — that is the
whole content of #912 F1, where a published decode figure measured a dead cache.

**G0-CORRECT (W0, blocking).** `Qwen3.8-2.4T-A95B UD-Q1_0`, `--device cuda`,
streaming ON, greedy, one fixed prompt, 32 decoded tokens. The token IDs must be
**byte-identical** to `--device cpu`, streaming ON, same binary, same checkpoint,
same prompt. "It ran" is not a result and neither is prose that looks coherent.
A mismatch stops the wave and no speed number is reported.

**G0-LIVE (W0, blocking).** From the same process:

* `steps > 0` — the step clock advanced. `steps == 0` with a printed decode
  figure is exactly the #912 F1 shape.
* `forced == 0` — no slice was served by the forced-fallback switch.
* **decode-phase `exhausted` delta == 0.** Not the total. The primary harness
  snapshots `detail::ExpertStreamSnapshot()` immediately after the prefill step
  and again at the end, and gates the DIFFERENCE. That is the shape a real
  benchmark needs, because it works at any prompt length, and prefill exhaustion
  is structural (see `## Our baseline`).
* Secondary control only, not the benchmark: a prompt of T <= 4 tokens with
  >= 11,160 slots makes prefill itself fit — 11,160 x 2,490,368 B = 25.88 GiB of
  arena plus 26.01 GiB of weights = 51.89 GiB, inside the 119.631 GiB pool — so
  TOTAL `exhausted` is legitimately 0. It is a useful cross-check on the
  snapshot arithmetic and it is not a workload, so it never replaces the diff.

**G0-SPEED (W0, reporting).** Decode s/token, CUDA arm against the CPU arm,
same box, same `rc` lease, same prompt, same token count, three interleaved
reps, min and median both recorded, plus resident bytes and the fill/hit/byte
counters. **No floor is set and none may be invented.** This row's first number
is a measurement of an unknown, and a CUDA arm SLOWER than the CPU arm is a
real, publishable result that closes the unified shortcut — recorded in
`docs/BENCHMARKS.md` as a measured negative, not as a failure to be tuned away.

**G1 (W1).** `DeviceExpertSlotStore` driven through `ExpertStreamer::EnsureFile`
produces byte-identical slot contents to `HostExpertSlotStore` on the same
input, on a CPU `vt::Backend`, red-first and mutation-proven per the table above.

**G2 (W2, reachability).** Per `## Nothing lands dead`: delete the production
selection of the device store in a scratch copy and rerun the focused gate. A
gate that stays green without it measured a class, not a capability.

**G-DISCRETE (owed, cannot run here).** See `## Owed`.

## Dependencies

| Dependency | Shape |
|---|---|
| **CPU decode re-measure on a live cache** | The DENOMINATOR for G0-SPEED, not a gate that blocks starting. `docs/BENCHMARKS.md:8` records streaming-ON decode as VOID (#912 F1) with a re-measure owed, and the operator is arranging it separately. G0-CORRECT and G0-LIVE need no denominator and can run first. **If the box frees up for only one run, take the CUDA arm** — it is the one that does not exist at all today, it produces the correctness verdict as well as a number, and the CPU denominator can follow. |
| `dgx:gpu0` | The only GB10. Leased through `rc`, never bare ssh; the hold sits behind a trap that releases on every exit path. Currently held by another agent's Nemotron gate, with a second agent on thor, so W0's measurement QUEUES. W0a/W0b/W0c/W0d are all buildable and unit-gatable without it. |
| The 370 GiB checkpoint | Already staged for the parent row; recipe in [expert-streaming.md](expert-streaming.md) "Loading a 370 GiB split GGUF". |
| PRs #1200 and #1216 | Both edit `.agents/specs/expert-streaming.md`. This row does not, deliberately — see `## Risks/decisions`. |
| #1126 (`Backend::DeviceMemoryInfo` has no CUDA override) | NOT a blocker. W0 and W1 read the budget from `residency_policy().device_memory_total_bytes`, which CUDA already probes at registration; they never touch the live free/total seam #1126 owns. |
| PR #1203 | Introduces `path::Symbol` citations and `scripts/check-symbol-anchors.py`. This spec uses symbol anchors where a symbol exists and lines only where the citation is to a comment or a specific statement. |

## Work breakdown

Each wave states its own gate and its own stop condition. A stop condition is
where the wave ENDS, not where it degrades quietly into the next one.

### W0 — the integrated/unified path (critical path for the GPU number)

* **W0a — the probe.** ~30 lines, no tree change, inside an `rc` lease: print
  `cudaDevAttrPageableMemoryAccess`, `...UsesHostPageTables`, `cudaDevAttrIntegrated`
  and `cudaDevAttrConcurrentManagedAccess` on `dgx:gpu0`. This is the cheapest
  decisive experiment in the row and it runs first, because the whole W0
  mechanism rests on a GB10 CUDA kernel being able to dereference a pointer from
  the host slot arena, which is plain `std::vector<uint8_t>` storage.
  **Stop condition:** if `PageableMemoryAccess == 0`, W0b as designed is wrong.
  The alternative is a `cudaHostAlloc`/`cudaHostRegister` arena, which works on
  ANY CUDA device including discrete but changes the store's allocator and its
  ownership story. That is a different design and it comes back as
  `NEEDS_DECISION` rather than being substituted silently.
* **W0b — the predicate.** `host_memory_is_device_addressable()`, base false,
  CUDA from the W0a attribute, ROCm from its already-probed
  `pageable_memory_access`. **Why not an existing predicate:** `is_cpu()` is what
  is being lifted; `needs_weight_staging()` is true on CUDA everywhere and would
  gate nothing; `is_unified_memory()` answers the opposite question (backend.h:58-66
  records that CUDA on GB10 reports unified while a `cudaMalloc` pointer is still
  not host-dereferenceable); and `is_integrated_gpu()` alone is insufficient
  because `rocm.cpp:52-58` already documents integrated-without-pageable-access as
  a real device class. A discrete CUDA device answers false, keeps falling through
  to `KqResidentSlice`, and therefore keeps hitting the #1123 refusal — which is
  correct, and which is what W1/W2 exist to remove.
  **Gate:** `tests/vllm/platforms/test_platform.cpp`, mutation-proven.
  **Stop condition:** none; this wave is small and self-contained.
* **W0c — the slot arm without the tower.** Lift the guard, build the slot
  tensor directly instead of through `ResidentWeight`, and add the named refusal
  for a streamed `*_exps` tower reaching device staging.
  **Gate:** the `qwen3_5` slot-arm unit gate above.
  **Stop condition:** if the marker set `ResidentWeight` carries cannot be
  reproduced without staging (a repack marker that only the upload path can
  set), stop — that is a different change and it is not this wave.
* **W0d — the conditional refusal.** Teach the fit bound the streamed-tensor
  exclusion and the arena, and pass it from the loader.
  **Gate:** `test_gguf_device_fit` for the arithmetic and
  `test_gguf_device_fit_reach` for the production reach; the lane-off bound must
  be byte-identical to today.
  **Stop condition:** if the exclusion cannot be expressed without the bound
  taking a general per-tensor staging POLICY (the shape #1136 explicitly refuses
  to invent), stop and return `NEEDS_DECISION`. The lane's tensor set is
  `*_exps` and is knowable; a general policy input is not.
* **W0e — the measurement.** G0-CORRECT, G0-LIVE, G0-SPEED, on one lease.
  **Stop condition:** a token mismatch, `steps == 0`, or a non-zero decode-phase
  `exhausted` delta stops the wave and voids the number.

### W1 — the device slot store (with its fill contract)

`DeviceExpertSlotStore` allocates its slot array through `vt::Backend::Alloc`.
It cannot be filled by `pread` (scope verdict above), so W1 adds
`CommitSlot(int32_t slot, size_t bytes)` to `ExpertSlotStore`: `SlotForWrite`
returns a host-writable staging buffer, `pread` fills it as it does today, and
`CommitSlot` performs the single contiguous H2D. `HostExpertSlotStore::CommitSlot`
is a no-op, so the host path stays byte-identical and keeps its direct
`pread`-into-slot. `ExpertStreamer` calls `CommitSlot` only on the success path;
the existing `catch` that invalidates the cache entry already covers the
partial-fill case the parent spec documents at length.

**Design answer: staging bounce, not a bespoke device filler.** A true zero-copy
filler (GPUDirect Storage / `cuFile`, or an `O_DIRECT` DMA into a device BAR
mapping) moves fewer bytes, needs a driver capability probe, a mount-level
check, an aligned-I/O path and a fallback for every case that fails those. The
bounce costs one extra host-to-device copy of 2.375 MiB per miss on top of a
disk read of the same size, lands in one wave, and keeps the zero-copy filler
genuinely optional rather than load-bearing. It is chosen for that reason and
not because it is faster; the measurement that would justify replacing it is a
device-arm decode where the H2D leg is a measurable fraction of fill time, and
that measurement does not exist yet. Recorded in `## Owed`.

**Gate:** G1. **Reachability:** W1 alone lands UNREACHED — nothing selects the
store and nothing can read it, because the read is still the concrete
`HostExpertSlotStore::Slot()`. Per `## Nothing lands dead` that is admissible
only if the commit body, the PR body and this spec's `## Owed` all name it and
name W2 as the owning wiring. **The cheaper and more honest shape is to land W1
and W2 as one pull request**, and that is the recommendation here; splitting
them is a scheduling choice that costs an explicitly-declared unreached slice.

**Stop condition:** if `CommitSlot` cannot be added without changing every
existing `ExpertSlotStore` caller's contract in a way that alters host-path
behaviour, stop — the host path must stay byte-identical, and a change that
cannot preserve that is a different design.

### W2 — the device-capable read

`virtual uint8_t* SlotForRead(int32_t)` on `ExpertSlotStore`;
`Qwen35ExpertStream::store_` becomes `std::unique_ptr<ExpertSlotStore>`; the two
`store_->Slot(...)` reads go through the virtual; the concrete store is selected
from the platform. **Gate:** G2, the call-site-deletion mutation.
**Stop condition:** if the returned pointer's device-ness cannot be expressed in
the `vt::Tensor` the GEMM binds without a second change to the tensor
construction, stop and re-scope — that is W0c's territory and it must not be
re-derived here.

## Risks/decisions

| Risk / decision | Call |
|---|---|
| **The GB10 ATS penalty could erase W0's win entirely.** Device access to host-resident weights on GB10 is recorded as carrying a real penalty, and this lane reads 6.95 GB per token that way. | Accepted as the thing being measured, not assumed away. G0-SPEED is what settles it, and the settling measurement is named: the CUDA arm's decode s/token against the CPU arm's, same box, same lease, three interleaved reps. A CUDA arm at or above the CPU arm's time is a genuine negative result — it closes the unified shortcut for this box, leaves W1/W2 standing for the discrete case, and is recorded in `docs/BENCHMARKS.md` as measured. It is a few hours, not a campaign, and that asymmetry is why W0 runs first. |
| W0 is FOUR edits, not one guard. | Stated rather than discovered. The two forcing facts are in `## Our baseline` (a): the slot arm's own `ResidentWeight` call stages the tower, and (b): the load-time refusal fires before any forward. Neither is in #1124's four pieces. W0d touches the fit bound, which is the one place this wave reaches into another row's surface; it is additive and exact (an explicit tensor exclusion plus the arena), it does not touch the KV/activation term `KV-WARMUP-PROFILE` owns, and if it cannot stay that way the W0d stop condition returns `NEEDS_DECISION`. |
| W1/W2 can be built here but only VALIDATED on hardware nobody here has. | Recorded as owed with its exact measurement rather than dressed as a gate. This host has no discrete NVIDIA GPU, and on `dgx:gpu0` device memory IS host memory, so a device store there proves the plumbing and not the capability. See `## Owed`. |
| A device store is the wrong shape if the answer is "put the slots in pinned host memory". | Open, and W0a's result informs it. A `cudaHostAlloc` arena is device-readable on discrete GPUs too, over PCIe, which is almost certainly too slow to serve 6.95 GB/token — but it is the fallback if `PageableMemoryAccess == 0` on GB10, and it is written down here so it is not re-invented as a surprise mid-wave. |
| This row does not edit `.agents/specs/expert-streaming.md`. | Deliberate. PRs #1200 and #1216 both edit that file today, and its `## Owed` entry for #1124 remains TRUE as written — it names the capability and the issue, and it does not name a row ID that this row's existence falsifies. Cross-linking is one-way, from here to there. Re-pointing that entry at this row is a one-line follow-up once both PRs land, and it is not worth a conflict now. |
| `ENGINE_ROWS` in `scripts/check-agent-record.py` is a shared counter, exactly the "measurement of one file stored in another" coupling AGENTS.md warns about. | Bumped 162 -> 163 for a real new row, with its justification paragraph, per the constant's own comment history. Checked against every open PR: none bumps it (#851 carries a stale `156` as diff context, #361 does not touch it), so this addition takes the lock cleanly. |
| Splitting the device capability out of `ENG-EXPERT-STREAM` rather than adding a W7-W9 to it. | The parent row's spec is 1600 lines and has two open PRs editing it. A separate row gives this capability an independent lifecycle state and a per-row spec surface, which is the shape AGENTS.md prefers (one file per row, read with a glob). The parent keeps the MECHANISM; this row owns the DESTINATION. |
| Streaming and the grouped keep-quant MoE path remain mutually exclusive. | Unchanged by this row, and unchanged by W0. `VT_MOE_EXPERT_STREAM=1` still disables grouping and says so once on stderr. Making them compose needs a slot-aware grouped GEMM and is its own row. |

## Owed

| Owed | Why it is open |
|---|---|
| **G-DISCRETE: validate W1/W2 on a discrete NVIDIA GPU.** The measurement: on a device with VRAM V and `host_memory_is_device_addressable() == false`, load a GGUF whose `*_exps` towers exceed V, with the lane on, and gate (i) token-exactness against the CPU arm on the same checkpoint, (ii) decode-phase `exhausted` delta 0, (iii) peak device allocation <= non-expert remainder + arena. | No discrete NVIDIA GPU is reachable from this project. `dgx:gpu0` is a GB10 where device memory IS host memory, so a device store there exercises the plumbing and not the thing W1 exists for. Recorded rather than implied, because a gate nobody can run is not a gate. |
| **A zero-copy device filler (GPUDirect Storage / `cuFile`).** | W1 ships the staging bounce by choice, for the reasons in its design note. The measurement that would justify replacing it — a device-arm decode where the H2D leg is a measurable fraction of fill time — does not exist until W1 has run somewhere. |
| **The CPU arm's streaming decode figure is still VOID.** `docs/BENCHMARKS.md:8` records it as VOID (#912 F1) with a re-measure owed. | Owned by `ENG-EXPERT-STREAM` and arranged separately by the operator. It is the DENOMINATOR for G0-SPEED, not a precondition for G0-CORRECT or G0-LIVE. |
| **`.agents/specs/expert-streaming.md`'s `## Owed` entry for #1124 still names no owning row ID.** | Not edited here on purpose; PRs #1200 and #1216 both edit that file. One-line follow-up once both land. |
| **W1 may land UNREACHED if it is split from W2.** | The recommendation is one pull request. If a split is chosen, the commit body and the PR body must name what is unreached and name W2 as the owning wiring, per `## Nothing lands dead`. |
