# BACKEND-ROCM-LANE-GUARD — key the streamed-expert lane on the predicate the fit check itself is keyed on

Issue: [#2507](https://github.com/mudler/vllm.cpp/issues/2507).
Related: [#1934](https://github.com/mudler/vllm.cpp/issues/1934)
(`BACKEND-ROCM-DEVICE-FIT-BOUNDED-MEMORY`, `DONE`) — this row corrects a claim
that row's spec makes about the lane guard, and that correction rides here
because this change is what makes it stale.
[#1124](https://github.com/mudler/vllm.cpp/issues/1124) (`ENG-EXPERT-STREAM-DEVICE`
W0d, which wrote the lane guard), [#2214](https://github.com/mudler/vllm.cpp/issues/2214)
(`MODEL-TEXT-GLM-MOE-DSA`, the model that cannot load).
Base: `6bf3abb58`.

## Scope

`src/vllm/entrypoints/model_loader.cpp` builds a `StreamedExpertLane` and hands
it to `CheckDeviceWeightFit`. The lane is the EXEMPTION inside that one
computation: towers the lane serves are not staged, so they leave the bound and
the slot arena enters it instead. The refusal and the exemption are two halves
of one arithmetic.

Since #1934 they are keyed on two different predicates. The refusal reads
`target.allocates_bounded_device_memory()`; the lane guard, six lines above,
still reads `target.needs_weight_staging()`. On CUDA both are true and nothing
shows. On ROCm they deliberately diverge — `false` and `true` respectively — and
the load draws the one combination that loses: the refusal fires and the
exemption does not.

IN SCOPE:

- The lane guard's PLATFORM term in `model_loader.cpp`, and only that term.
- The correction to `.agents/specs/rocm-device-fit-bounded-memory.md`, whose
  Design section states the lane guard "genuinely asks 'is the fully-optimized
  device-resident forward what's running'". This change falsifies that.
- Tests pinning the new keying in both directions, and the CUDA/CPU inertness
  that keying implies.

OUT OF SCOPE:

- `needs_weight_staging()` itself and every one of its other consumers. This row
  moves ONE read of it and no platform's answer to it.
- The eight-op MLA/DSA arm (`kBatchedMatmul`, `kConcatAndCacheMla`,
  `kConcatMlaNopeRope`, `kDsaIndexerLogits`, `kDsaTopkSelect`, `kFusedNormRope`,
  `kGatherMlaCache`, `kMlaDecodeAttention`). Out of scope, and §`## The eight ops
  do not block text` records the measurement that says porting them is not
  required to reach generation.
- The two adjacent defects measured on the same load. See `## The two adjacent
  defects` below for the disposition and the reasoning.

## The property the lane guard is testing for

The question is not "which predicate admits ROCm". It is what the guard needs to
know, and the guard needs to know two independent things:

1. **Is a device-fit computation running at all?** The lane changes only the
   arithmetic of `CheckDeviceWeightFit`, which returns before computing anything
   when its gating argument is false. A lane built for a load that computes no
   footprint is pure side effect — `ResolveExpertStreamRequested()` latches the
   process's streaming answer and `ExpertStreamLane::Reserve()` fixes the store's
   geometry, neither of which this block has any business doing on a load whose
   fit check is inert. The predicate that answers this is, by construction, THE
   PREDICATE THE FIT CHECK IS KEYED ON: `allocates_bounded_device_memory()`.
   Any other choice lets the refusal and its exemption disagree, which is the
   defect being repaired, not a different way of spelling it.

2. **Will the lane actually serve at runtime?** A lane that exempts towers the
   forward then stages is worse than no lane: it deletes a correct refusal and
   restores the load-then-die shape #1123 exists to prevent. The runtime serving
   predicate is `ExpertSlice` (`src/vllm/model_executor/expert_stream_seam.cpp:431`):

   ```cpp
   if (cpu || p.host_memory_is_device_addressable()) {
   ```

   `host_memory_is_device_addressable()`, and **no `needs_weight_staging()`
   term**. The guard already carries that same predicate as its second condition.

`needs_weight_staging()` answers neither. Its own doc
(`include/vllm/platforms/interface.h`) defines it as the policy selecting the
fully-optimized device-resident forward — indexed GDN state I/O, merged/packed
GDN projections, fp8/bf16 GDN resident prep. Whether those GDN kernel defaults
run has no bearing on whether a footprint is being computed, and none on whether
a slot store can be read. It stood in for (1) only because before #1934 there was
no separate predicate for (1) to stand on; #1934 built one, moved the refusal
onto it, and left the exemption behind.

So the repair is: the lane guard's first term becomes
`target.allocates_bounded_device_memory()`. Its second term
(`host_memory_is_device_addressable()`) is already correct and is untouched. This
is not "widen the guard until ROCm fits" — it is the two questions the guard
actually asks, each read from the predicate that answers it, and it happens to
admit ROCm because ROCm genuinely answers yes to both.

**Not a `||`.** Reading `needs_weight_staging() || allocates_bounded_device_memory()`
would admit ROCm too and would be wrong: it re-admits a platform that stages but
reports no bounded pool, for which the fit check computes nothing and the lane is
again pure side effect. The tests pin the plain predicate in both directions.

## Upstream chain

None. `Platform::needs_weight_staging()` and
`Platform::allocates_bounded_device_memory()` are both vllm.cpp-original policy
seams with no vLLM mirror (each says so in its own doc), and the streamed-expert
lane has no upstream counterpart — vLLM's own MoE offload is a different
mechanism. There is nothing to port and nothing to compare; the oracle question
this row answers is internal consistency between two halves of one local
computation.

## Design

One term, in one expression, at `model_loader.cpp`'s single streamed-expert-lane
guard:

```cpp
-      if (target.needs_weight_staging() &&
+      if (target.allocates_bounded_device_memory() &&
             target.host_memory_is_device_addressable() &&
```

The other four conditions (`factory != nullptr`, `streams_routed_experts`,
`ResolveExpertStreamRequested()`, `GgufExpertTowersReachSlotLane(...)`) are
unchanged, in the same order, for the reason the block's own comment gives: the
latch must stay behind the platform and architecture terms.

The comment above the guard is rewritten to name the predicate and the property,
and to cross-reference the runtime serving predicate in `expert_stream_seam.cpp`
that the guard now matches. The neighbouring comment that reads "`cpu_ref` needs
no term of its own: it is a CPU-only oracle switch, and `needs_weight_staging`
above already excludes every load it could apply to" is repaired in the same
edit: the exclusion it names still holds, because `allocates_bounded_device_memory()`
also delegates to false on CPU, but the sentence names a predicate that is no
longer above it.

## Risks and decisions

- **Does this widen the population of loads that build a lane?** Yes, by exactly
  the platforms where `allocates_bounded_device_memory() != needs_weight_staging()`
  AND `host_memory_is_device_addressable()` AND the architecture declares
  `streams_routed_experts` AND streaming was requested AND every `_exps.weight`
  tower routes to a keep residency. Today that intersection is integrated ROCm
  and nothing else. CPU is excluded because the new predicate's default delegates
  to `needs_weight_staging()`, which is false there; discrete ROCm is excluded by
  the host-addressable probe, and correctly so — a slot store it cannot read is
  not a lane.
- **Could this delete a correct refusal?** Only for a load whose forward really
  does stream, because the guard's remaining terms are unchanged and
  `GgufExpertTowersReachSlotLane` asks the model loader's own routing function
  about this file under this process's policy. The #1378 finding that motivated
  that term is untouched here.
- **Should `needs_weight_staging()` be flipped on ROCm instead?** No, and #1934
  already measured why: several GDN consumers have no op-registration fallback
  and would activate on unverified kernels. That rejection stands. This row does
  not revisit it and does not need to.
- **Does the repair belong to #1934's row rather than this one?** #1934 is `DONE`
  and its scope explicitly excluded the lane guard. Reopening a `DONE` row to
  correct one sentence in its Design, rather than landing the fix under its own
  row with its own review, would put the change on a spec nobody is reviewing.
  The record correction rides here, in the change that made it stale, which is
  what `AGENTS.md` §Records requires.

## The two adjacent defects

Both were measured on the same load. Both are real. Neither is load-bearing for
the load succeeding, and both are therefore filed rather than bundled.

**A. The placement plan and the fit check disagree within one load.** The hybrid
MoE placement reports it brought a 216,433,205,760 B footprint under a
68,719,476,736 B budget by placing 56 layers on the CPU (147,798,884,352 B), and
the next line refuses quoting the un-reduced 201.56 GiB. NOT load-bearing: with
the lane engaged the fit check charges the resident class plus the arena
(predicted 14.5105 GiB against a 64.00 GiB pool), which is under budget by a
factor of four, so the disagreement changes no verdict on this model. It is still
a defect — two lines of one load contradicting each other is a fit check that
does not account for a plan it was told about — and it would decide a load on a
model sized between the two figures. Filed as its own issue; not fixed here,
because fixing it means deciding whether the placement plan is an input to the
bound or a consumer of it, which is a design question about a different seam.

**B. The budget overstates the reachable pool by 5.92 GiB.** `DeviceWeightBudgetBytes`
reads `hipMemGetInfo`'s 64.00 GiB total while every `Backend::Alloc` on this board
is `hipMallocManaged` (`src/vt/rocm/rocm_backend.hip`), whose measured ceiling is
58.000 GiB across three samples. NOT load-bearing: 14.5105 GiB plus the arena is
under both figures. It is optimistic in the direction that produces a late OOM,
which is the failure #1123 exists to prevent, so it is a genuine gap — but
correcting it means changing what `ResidencyPolicy::device_memory_total_bytes`
means on ROCm, which moves the refusal for every ROCm load and is exactly the
scope #1934 deliberately left at the probe. Filed as its own issue.

Both issues name this row's spec under `## Owed` below, so neither is filed
without an owner.

## A correction to #2507's own premise

#2507 states "the other five conditions are satisfied on this board". Two of them
were not verified by the run that filed it, and one was not satisfied.

`ResolveExpertStreamRequested()` is default OFF
(`src/vllm/config/weight_residency.cpp:1047` — a function-local static over
`VT_MOE_EXPERT_STREAM` and the offload config's `expert_stream`). The run in
`/mnt/nas_share/rc/glm53-rocm/out6/` (`probe6.sh`) sets neither. Because
`needs_weight_staging()` is the FIRST term of a short-circuiting `&&`, the
resolver was never reached, so that run could not have told the difference —
and this fix alone would not have made that run load. The GB10 recipe that does
load (`/mnt/nas_share/rc/glm53-firstload/run.sh:287`) exports
`VT_MOE_EXPERT_STREAM=1`. This row's gate does the same, and says so.

The predicate defect is unchanged by this: with streaming requested, the guard
still fails on its first term today and passes after the fix. The correction is
to the evidence, not to the diagnosis.

## The eight ops do not block text

Asked directly, because the row was raised to an end-to-end generation bar and
the answer decides whether porting is on the critical path. It is not.

The reference tier is a **generic, device-capability-gated fallback**, not a
per-op opt-in registry. `ReferenceTierEligible`
(`src/vt/op_provider.cpp:888-919`) returns true for any non-CPU device whose
registered backend answers `DeviceMemoryIsHostAddressable()`, and
`MaybeInstallReferenceTier` (`op_provider.cpp:204-222`) is called from the
dispatch-miss path for ANY `OpId`: if the device is eligible and the CPU has a
kernel for that same op, it installs the CPU function pointer as a
`kReferenceTierPriority` provider. `Resolve` throws only when that also comes
back empty (`op_provider.cpp:597-601`).

`RocmBackend::DeviceMemoryIsHostAddressable()` returns `unified_memory_`, which
the registrar sets from `managed_alloc || (pageable_memory_access && integrated)`
— and `managed_alloc` is true on this board. So the tier IS eligible on
`gfx1151`, even though `host_memory_is_device_addressable()` (the OPPOSITE
direction: may a device kernel read a HOST pointer) is false. The two predicates
are mirrors, and this board answers them differently. That is why the reference
tier works here while the host-slot expert lane cannot.

All eight ops have a CPU registration — verified individually:
`cpu_ops.cpp:3956` (`kBatchedMatmul`), `cpu_cache.cpp:188`
(`kConcatAndCacheMla`), `cpu_ops.cpp:3959` (`kConcatMlaNopeRope`),
`cpu_dsa_indexer.cpp:183` and `:185` (`kDsaIndexerLogits`, `kDsaTopkSelect`),
`cpu_ops.cpp:4017` (`kFusedNormRope`), `cpu_mla_prefill.cpp:277`
(`kGatherMlaCache`), `cpu_mla_attn.cpp:222` (`kMlaDecodeAttention`). None was
registered for ROCm when this was written. Six of the eight are now:
[#2715](https://github.com/mudler/vllm.cpp/issues/2715) W1 landed four and
[#2926](https://github.com/mudler/vllm.cpp/issues/2926) landed
`kMlaDecodeAttention` and `kMlaPrefillAttention`. The DSA indexer pair is
still owed.

So the forward reaches them, they install host kernels, and they run. Generation
would be correct at host speed for that arm, and `docs/ROCM.md:60-61` forbids any
performance result from a run whose `GetReferenceTierHits()` is non-zero. **No
kernel port is required to reach text.** Porting them is a throughput row, not a
correctness blocker, and this row does not open it.

## Tests

**`tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp`** — the harness already
carries independently-settable `needs_weight_staging_flag`,
`allocates_bounded_device_memory_flag` and `host_addressable`, and a synthetic
MoE GGUF with an expert tower whose lane-ON and lane-OFF bounds are known
constants. Three new cases, all on ROCm's real flag shape
(`staging=false, bounded=true, host_addressable=true`):

- **The RED case.** Budget at `kLaneOnBound + 1` — above the lane-ON figure,
  below the lane-OFF one. Today the load refuses quoting `kLaneOffBound`; after
  the fix it reaches the tokenizer error. Asserts the later error POSITIVELY, so
  a load that died earlier for an unrelated reason cannot pass.
- **The positive proof the lane was COMPUTED, not merely that the refusal
  vanished.** Budget at `kLaneOnBound - 1`: the message must carry the lane note,
  the streamed-tower byte count and the arena byte count. A guard that permitted
  the load for any other reason cannot produce those three numbers. This is the
  case that distinguishes "the lane engaged" from "the refusal stopped firing",
  which a tolerance-free boolean gate cannot otherwise tell apart.
- **The `||` guard, and a correction.** The first draft of this case asserted on
  the thrown MESSAGE for `staging=true, bounded=false`, and that discriminated
  NOTHING: with `bounded=false` the fit check returns before it can emit either a
  refusal or a lane note, so the load reaches the tokenizer whether a lane was
  built or not. It passed under the old predicate, the new one, and a `||` alike,
  while reading as if it excluded them. Caught by noticing it was green in the RED
  arm.
  The observable that does discriminate is the LATCH.
  `ResolveExpertStreamRequested()` is the guard's fifth term and freezes the
  process's streaming answer; `WeightResidencyLatched(ResidencyLatch::kExpertStream)`
  reports whether it was reached, and `ResetWeightResidencyConfigForTesting()`
  clears it. The guard short-circuits before that term iff its FIRST term is
  false, so the latch reads the first term directly — false under the plain new
  predicate, true under a `||` or the old one. That latch is also precisely the
  side effect this row's argument is about, so the case now tests the claim
  rather than a proxy for it.
- **The positive control for it.** ROCm's own shape with the same file, budget and
  host-addressable bit, asserting the latch IS reached. Without it, "did not
  latch" is equally true of a guard deleted outright, and the pair would pass on
  a lane block that no longer exists.

**Inertness.** Every existing case in the file constructs the platform with both
flags `true`, which is CUDA's real shape, and must be byte-for-byte unmoved. The
three existing W0d lane cases are the CUDA-behaviour pin: they set
`host_addressable` and leave the staging flags at their `(true, true)` default,
so they assert the CUDA lane arithmetic directly.

**Reachability mutation.** Delete the production lane block in `model_loader.cpp`
in a scratch copy, rebuild, rerun the suite. The lane-ON cases must go red.

## Gates

- `./build-hip/tests/test_gguf_device_fit_reach` on `strix:gpu0` (gfx1151) —
  full case and assertion counts, read on BOTH doctest lines.
- `./build-hip/tests/test_platform` on the same box — the ROCm-real platform case
  asserting `needs_weight_staging()` is still false and
  `allocates_bounded_device_memory()` still true, i.e. this row moved no
  platform's answer.
- `./build-hip/tests/test_gguf_device_fit` — the arithmetic, unmoved.
- CPU build of the same three suites for the CPU/CUDA-shape inertness half.
- **The load itself**, on `strix:gpu0`, `--device auto` (the only route to ROCm;
  #2505), `VT_MOE_EXPERT_STREAM=1`, driving the FIRST SHARD of the derived
  `GLM-5.3-UD-IQ1_S`. The bar is the `[expert-stream] ON slots=... resident=...`
  line present and the load past the refusal — not merely a different error.
- `scripts/agent-preflight.sh --fail-on-skip`.
- `python3 scripts/check-commit-trailers.py --range origin/main..HEAD`.

No speed number. Any run on this box today reaches eight reference-tier ops,
which `docs/ROCM.md:60-61` disqualifies from carrying a performance result.

## Evidence

In the pull request body: red-before and green-after for the three new cases with
both doctest counts, the reachability mutation with its kill count, the box
identity (`gfx1151`, ROCm 7.2.4, `integrated=1 managedMemory=1`), and the exact
load transcript including the lane's counters and the resident footprint.

## Owed

- **The placement-plan / fit-check disagreement** (defect A above). Filed as its
  own issue, owned by this row's spec until a row claims it. It needs a decision
  about whether the resolved placement plan is an input to
  `GgufStagedWeightFootprint` or a consumer of its verdict, which is a design
  question about the hybrid-placement seam rather than about this guard.
- **The managed-allocator budget ceiling on ROCm** (defect B above). Filed as its
  own issue, owned by this row's spec until a row claims it. It needs a measured
  `hipMallocManaged` ceiling probe to replace or bound the `hipMemGetInfo` total,
  and that moves the refusal for every ROCm load.
- **The eight-op MLA/DSA arm for `glm-dsa` on ROCm.** Not this row's, and named
  here only because a reader arriving from #2507 will ask: the load is its
  precondition, and until the load works those kernels are unreachable.

## Stop conditions

Stop and report rather than widening scope if making the lane engage turns out to
need a change to `needs_weight_staging()`'s value on any platform. It should not:
this row moves one READ of that predicate and no platform's answer to it, and the
tests assert exactly that.

Stop rather than fixing either adjacent defect if the load turns out to still
refuse after this change. A second fix stacked on an unverified first one makes
neither attributable.

## Now

`ACTIVE`. The predicate repair is implemented, gated on `strix:gpu0`, and
correct. The bar the row was raised to — GLM-5.3 generating text on that board —
is NOT met, and §`## What the hardware said` records why with evidence.

## What the hardware said

Measured on `strix:gpu0`, `gfx1151`, ROCm 7.2.4, rc job
`b36f58f0-958d-4078-93d0-6d073e2c46e2`, log
`/workspace/glm53-lanefix/out/run.log`.

**The repair works and is proven.** On a HIP build of this branch: with the one
line reverted, 2 of 20 cases fail, and they are exactly the two new lane cases,
failing because the whole 8464-byte table is charged and no lane note is emitted.
With the line restored, 20 of 20 cases and 94 of 94 assertions pass. Deleting the
production lane block reds 4 cases — the two new ones plus the two pre-existing
CUDA-shape W0d lane cases — so the tests enter through the production loader and
not through a hand-built predicate. `test_platform` is 17 of 17 and 133 of 133,
so no platform's answer to either predicate moved.

**The model still does not load, and the lane guard was never the only blocker.**
Two further conditions are false on this board, both independently of the
predicate this row repairs, and both already measured before #2507 was filed:

- `host_memory_is_device_addressable()` is FALSE. It reads
  `integrated && pageable_memory_access` (`rocm_backend.hip:474-481`), and
  `pageableMemoryAccess=0` on `gfx1151` — recorded at
  `.agents/specs/rocm-glm53-dsa.md:29` and measured at
  `/mnt/nas_share/rc/glm53-rocm/out2/probe2.log:55`. #2507 asserted this
  condition was satisfied, citing `managedMemory=1 concurrentManagedAccess=1`;
  those are the inputs to `UseManagedAlloc`, a different predicate. Filed as
  #2515.
- `GgufExpertTowersReachSlotLane()` is FALSE. ROCm registers no IQ `vec_dot`, so
  `RouteGgufTensor` answers `kExpandBf16` for the IQ2_XS/IQ4_XS
  `kStackedExpertWeight` towers this checkpoint stores its experts in. This also
  makes `test_gguf_device_fit` RED on any ROCm build, 1 case and 2 assertions,
  pre-existing on `main`. Filed as #2516.

The second of those is a kernel-coverage gap and is closable. **The first is
physical.** The same predicate gates the lane at RUNTIME — `ExpertSlice`
(`expert_stream_seam.cpp:431`) serves from a `std::vector<uint8_t>` arena only
when `cpu || host_memory_is_device_addressable()` — so a device kernel on this
board may not dereference the slot store at all. Forcing the lane on would delete
a correct refusal and restore the load-then-die shape #1123 exists to prevent.
**The host-slot streaming lane is not a mechanism that can work on Strix Halo**,
and the bar's requirement that the lane's fill/served/bytes counters be present
is therefore unreachable here. It needs a device-side slot store
(`ENG-EXPERT-STREAM-DEVICE` W2), which is a different row.

**A trap worth recording.** The refusal's printed footprint cannot be used to
infer that keep-quant applied. `StagedBytes` (`gguf_device_fit.cpp:41-57`)
charges `min(expanded, on-disk)`, and for a sub-2-bit encoding the expanded size
is far larger, so the number equals the on-disk size whether the tower is kept or
expanded. Reading 216433205760 B as evidence of keep-quant routing is wrong, and
this row made that inference before checking `StagedBytes` and had to withdraw it.

## The two adjacent defects, disposition REVISED

Filed, not fixed here — but the reasoning has changed and the reader should have
the current version rather than the one written before the hardware answered.

They were expected to be non-load-bearing because the lane would carry the model.
The lane cannot. With streaming unavailable on this board, the hybrid MoE
placement is the ONLY remaining mechanism that could fit this checkpoint on
`gfx1151`, which makes both defects part of the critical path rather than beside
it:

- **A**, the placement/fit disagreement (#2517): the plan reports the model
  63.92 GiB, under budget by 81 MiB, and the refusal on the next line quotes the
  un-reduced 201.56 GiB.
- **B**, the budget source (#2518): 68719476736 B from `hipMemGetInfo` against a
  measured 62277025792 B `hipMallocManaged` ceiling. The two interact — A's own
  reduced figure sits ABOVE B's real ceiling — so honouring the plan without
  correcting the budget would move the failure from load time to allocation time
  rather than remove it.

They stay out of this row anyway, and the reason is unchanged: A is a design
question about whether a resolved placement plan is an input to the footprint or
a consumer of its verdict, and B changes what
`device_memory_total_bytes` means for every ROCm load on every model. Each wants
its own row, spec and hardware evidence. Bundling either into a one-predicate
repair would make neither attributable.

## Owed (revised)

- #2515 — the device-side expert slot store, the only lane mechanism that does
  not depend on `pageable_memory_access`.
- #2516 — IQ `vec_dot` for ROCm, or a platform scope on the test case that is
  currently red for an unstated reason on every ROCm build.
- #2517 and #2518 — the two adjacent defects, now the critical path to a load on
  this board.
