# BACKEND-ROCM-DEVICE-FIT-BOUNDED-MEMORY — make the #1123/#1870 refusal reachable on ROCm, without moving GDN kernel defaults

Issue: [#1934](https://github.com/mudler/vllm.cpp/issues/1934).
Related: [#1870](https://github.com/mudler/vllm.cpp/issues/1870) (stays open;
this row is its actual remaining blocker), [#1928](https://github.com/mudler/vllm.cpp/issues/1928)
(unrelated ROCm gap, not touched here).
Base: `2a42cb369` (`upstream/main` at the claim).

## Scope

`CheckDeviceWeightFit`'s one production call site (`model_loader.cpp`) reads
`target.needs_weight_staging()` to decide whether the load-time device-fit
refusal (#1123, and the arithmetic fix #1870/#1935 made to it) runs at all.
`RocmPlatform::needs_weight_staging()` hardcodes `false` — a W0-era comment
says a discrete AMD card "will eventually answer true... Revisit at M2", never
revisited despite ROCm landing grouped-GEMM, MoE combine/gate, ROCM_ATTN and
hipGraph decode capture since. Measured directly on this box (gfx1200):
`VT_DEVICE_WEIGHT_BUDGET_BYTES=1` against a real checkpoint produced no
refusal at all, because `CheckDeviceWeightFit` returns before computing
anything when its `needs_weight_staging` argument is false. Meanwhile the
actual device allocation the refusal exists to guard — `d.b.Alloc(nb)` in
`qwen3_5.cpp`'s `ResidentWeight` — is NOT gated by that flag; it runs
unconditionally on any non-CPU platform (since issue #125's fix keyed the
alias-vs-upload branch on `is_cpu()`, not `needs_weight_staging()`). So #1870's
`hipMalloc: out of memory` stays reachable regardless of the arithmetic fix.

IN SCOPE:

- Make the device-fit refusal reachable on ROCm: a real, once-probed
  `ResidencyPolicy::device_memory_total_bytes` and a way for the ONE call site
  to know a budget check applies here.
- Do it WITHOUT flipping `needs_weight_staging()` itself, because that flag is
  documented (`interface.h`) to also gate the indexed/merged/packed GDN
  kernel-dispatch defaults, at least one of which (`MergedGdnBaEnabled` /
  `MergedGdnQkvzEnabled` / the packed-decode and fp8-resident-prep gates) has
  NO op-registration fallback — it would newly activate unconditionally on
  ROCm the moment the flag moves, exactly the "assume kernels a device might
  not have" failure `IndexedGdnStateIoEnabled` was written to avoid for the
  ONE consumer that already checks op registration. Auditing and (where
  needed) porting each of those kernel paths for ROCm correctness is a
  separate, larger undertaking than this row.

OUT OF SCOPE:

- `needs_weight_staging()` itself, and every one of its other consumers
  (`DirectDeviceLoadEligible`, `IndexedGdnStateIoEnabled`, `MergedGdnBaEnabled`,
  `MergedGdnQkvzEnabled`, `PackedGdnDecodeEligibility`'s eligibility struct,
  `PrepareGdnFp8Resident`, `PrepareBf16Resident`). None of their behavior moves
  in this row, and the tests prove it.
- `ResidencyPolicy::release_host_weights_after_upload` and
  `uses_device_memory_pool` for ROCm. CUDA sets both `true` for its own
  Marlin/DevicePool reasons; this row leaves ROCm's answer at the existing
  default (`false`) because those are separate, unmeasured policy questions,
  not a side effect of making the budget check reachable.
- The `kMoeGroupedGemmBf16` ROCm provider gap (#1928). Unrelated to this row.

## Upstream chain

No upstream vLLM mirror for the new predicate — `Platform::needs_weight_staging()`
itself has none (it is a vllm.cpp-original policy seam, per its own doc), and
this row adds a narrower sibling to it for the same reason. The
`device_memory_total_bytes` probe mirrors `platforms/cuda.cpp`'s own
`cudaMemGetInfo` call at registration, HIP's exact analog.

## Design

**A new `Platform` method, not a flipped flag.** `Platform::allocates_bounded_device_memory()`
(`interface.h`) answers the narrow physical question `CheckDeviceWeightFit`
actually needs: does a load here draw from a bounded, budget-checkable device
memory pool? Default implementation delegates to `needs_weight_staging()` —
byte-identical for CUDA (already true) and every platform that overrides
neither method (still false). `RocmPlatform` overrides ONLY this new method to
`true`, leaving `needs_weight_staging()` at its existing `false`.

`model_loader.cpp`'s ONE `CheckDeviceWeightFit` call site reads
`target.allocates_bounded_device_memory()` instead of
`target.needs_weight_staging()`. The OTHER `needs_weight_staging()` read in the
same function (the streamed-expert-lane condition, a few lines above) is
UNTOUCHED by THIS row.

**That last sentence's REASON was wrong, and issue
[#2507](https://github.com/mudler/vllm.cpp/issues/2507) measured the cost.** It
said the lane condition "genuinely asks 'is the fully-optimized device-resident
forward what's running'". It does not. The lane is the EXEMPTION inside this
same refusal's arithmetic — towers it serves leave the bound and the slot arena
enters it — so it needs the same predicate the refusal is keyed on, and the
runtime seam that decides whether the lane can serve at all (`ExpertSlice`,
`expert_stream_seam.cpp`) reads `cpu || host_memory_is_device_addressable()`
with no staging term whatsoever. Leaving the two halves of one sum on two
predicates is what let them disagree on the one platform where they diverge:
ROCm refused GLM-5.3 at load, charging 187.3125 GiB of towers it declares it
streams, because the refusal fired and the exemption did not. Repaired by
`BACKEND-ROCM-LANE-GUARD` (`.agents/specs/rocm-expert-lane-guard.md`), which
moves that read and no platform's answer to either predicate. This row's own
scope decision — do not touch `needs_weight_staging()` itself — still stands and
is unaffected.

**Why not just flip `needs_weight_staging()`.** Read literally:
`IndexedGdnStateIoEnabled` (`qwen3_5.cpp`) ALREADY takes ROCm's fast arm today,
independent of `needs_weight_staging()` — its `!needs_weight_staging()` branch
falls through to `IndexedGdnOpsNative(device)` for any non-CPU platform,
checking `kCausalConv1dUpdate`/`kGdnDecode`/`kGdnStateGather`/`kGdnStateScatter`
registration directly (all four ARE registered for ROCm — confirmed via
`rocm_ops.hip`). Flipping `needs_weight_staging()` would make this consumer
take the OTHER branch (`return indexed == nullptr || indexed[0] != '0';`),
which DROPS the op-registration check and assumes the ops exist — a
regression, not an improvement, for this one consumer. `MergedGdnBaEnabled`
and its siblings have no such fallback at all: they are a bare
`enabled && needs_weight_staging()`, so flipping the flag activates them on
ROCm unconditionally, and whether the underlying merged/packed GEMM and
fp8-resident-prep paths are numerically correct there is unverified. A new,
narrow predicate sidesteps all of it.

**The probe.** `RocmPlatform` (`rocm.cpp`) is deliberately HIP-header-free —
compiled as a stub object even in non-HIP builds, reaching the device only
through the `vt::rocm::` free-function seam (`rocm_runtime.h`), the same
discipline `DeviceAvailable()`/`HostMemoryIsDeviceAddressable()` already use.
A NEW free function, `vt::rocm::DeviceMemoryTotalBytes(int index)`, declared
there and implemented in `rocm_backend.hip` (`hipSetDevice` + `hipMemGetInfo`,
mirroring `Backend::DeviceMemoryInfo`'s existing body without calling it —
that method is a LIVE per-request probe with its own single consumer,
`Gemma4MoE`'s expert LRU, and reusing it for `ResidencyPolicy` would be the
exact "wake a currently-dead-on-CUDA seam" scope-widening `interface.h`'s own
comment on `device_memory_total_bytes` names as issue #1126's territory — a
DIFFERENT platform's DIFFERENT concern, not this row's). The platform
registrar calls it ONCE, device 0, and threads the result through a new
`RocmPlatform` constructor parameter into a stored member, exactly mirroring
`CudaPlatform`'s own constructor-injected `device_memory_total_bytes_`.

`RocmPlatform::residency_policy()` returns `{device_memory_total_bytes = <probed>}`
with every other field at its existing default (`false`/`0`) — see Out of
Scope.

## Risks and decisions

- **The new predicate's name.** `allocates_bounded_device_memory()` was chosen
  over reusing `needs_weight_staging` under a different value, or deriving the
  gate implicitly from `residency_policy().device_memory_total_bytes != 0`
  alone. An explicit, documented predicate matches this codebase's own
  established style (`is_integrated_gpu`/`is_unified_memory`/
  `host_memory_is_device_addressable`/`needs_weight_staging` are all
  deliberately separate, cross-referenced concepts with a paragraph each
  explaining why they are not one another) rather than an implicit
  budget-nonzero proxy a future reader would have to reverse-engineer.
- **Two predicates that look related but move independently is exactly the
  shape issue #125 already burned this codebase on once**
  (`needs_weight_staging()` vs. `is_cpu()` for `ResidentWeight`'s alias-vs-upload
  branch). This row adds a THIRD closely-related predicate rather than fixing
  the second one's scope, which could read as compounding the confusion. Argued
  against a merge: `needs_weight_staging()` and `allocates_bounded_device_memory()`
  answer genuinely different questions (device-resident FORWARD KERNELS vs. a
  BUDGET GUARD on ordinary weight upload) that happen to coincide on CUDA and
  deliberately diverge on ROCm — the tests pin the divergence rather than
  asserting it away.
- **`vt::rocm::DeviceMemoryTotalBytes` vs. `Backend::DeviceMemoryInfo`.**
  Considered reusing the existing backend method directly from the platform
  registrar via `vt::GetBackend(DeviceType::kROCM).DeviceMemoryInfo(...)` and
  rejected: static-init order across translation units is unspecified (the
  registrar's own comment states this as the reason it probes the device
  itself rather than trusting `RocmBackend`'s registrar to have already run),
  so a call through `GetBackend` at platform-registration time risks reading
  an unregistered backend. The free-function seam sidesteps the ordering
  question entirely, the same way `DeviceAvailable()` already does.
- **CPU cost of a `hipSetDevice`+`hipMemGetInfo` call at every process start
  on a ROCm build.** Bounded: it runs exactly once, in the platform registrar,
  which already makes comparable device-attribute probe calls
  (`ProbeDevice`'s own `hipGetDeviceProperties`-class calls) at the same point.

## Tests

**`tests/vllm/platforms/test_platform.cpp`**:

- CPU and the existing `FakeCapabilityPlatform` (`sm121`) cases gain a
  `CHECK(x.allocates_bounded_device_memory() == x.needs_weight_staging())`
  pin, proving the default delegation is a true no-op for every platform that
  does not override the new method — red before this row (the method does not
  exist), trivially green after.
- A new fake platform, `FakeBoundedNonStagingPlatform`, constructing ROCm's
  own real shape (`needs_weight_staging()=false`,
  `allocates_bounded_device_memory()=true`) on the CPU tier — proves the two
  predicates are independently settable, mirroring the existing
  `FakeUnifiedAddressablePlatform` pattern this file already uses for
  `host_memory_is_device_addressable()` vs. its neighbours.
- A new HIP-build-only case, skipped via `if (!HasPlatform(DeviceType::kROCM)) return;`
  (matching the existing CUDA-only case's pattern), asserting on THIS box's
  real `RocmPlatform`: `needs_weight_staging()` still false,
  `allocates_bounded_device_memory()` true, `residency_policy().device_memory_total_bytes > 0`
  (the probe actually ran), and the two policy fields this row deliberately
  left alone still read false.

**`tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp`**: `StagingPlatform`
gains two independently-settable flags
(`needs_weight_staging_flag`, `allocates_bounded_device_memory_flag`,
previously the class hardcoded `needs_weight_staging()` to `true` and had no
override for the new method). Two new cases: one with
`(false, true)` — ROCm's exact real state — proving the refusal STILL fires;
one with `(true, false)` proving it does NOT (the call site reads the new
predicate, not the old one, in both directions). Every EXISTING case in this
file is unaffected: the default-constructed flags (`true`, `true`) reproduce
today's behavior byte for byte because the new method's default delegates to
the old one, which the class already overrides to `true`.

## Gates

- `./build-hip/tests/test_platform` (ROCm build, this box, gfx1200) — the
  HIP-only case above is the one that can only run here
- `./build-hip/tests/test_gguf_device_fit_reach`
- `./build-nix-cpu/tests/test_platform` / `test_gguf_device_fit_reach` (CPU
  build, same box) — every other case
- A real hardware repro: with this row, `VT_DEVICE_WEIGHT_BUDGET_BYTES=1`
  against a real checkpoint on this box's gfx1200 card refuses (it did not,
  before this row) — the exact measurement #1934 reports as missing.
- `scripts/agent-preflight.sh --staged` (against the true fork point, not
  `origin/main` — see `.agents/developer-preferences.md`)

## Evidence

Recorded in the pull request body: red-before/green-after for both test
files, and the real-hardware before/after transcript
(`VT_DEVICE_WEIGHT_BUDGET_BYTES=1` refusing where it previously did not).

## Owed

- **The GDN kernel-dispatch audit.** `needs_weight_staging()` itself stays
  false on ROCm, so `DirectDeviceLoadEligible`, `IndexedGdnStateIoEnabled`
  (already independently safe — see Design), `MergedGdnBaEnabled`,
  `MergedGdnQkvzEnabled`, the packed-decode eligibility struct, and the fp8/bf16
  GDN resident-prep gates all keep their current ROCm behavior. Whether ROCm
  SHOULD eventually take the optimized arm for the ones with no op-registration
  fallback is a real, separate question this row does not answer — it needs a
  per-consumer op-registration and token-exact audit, not a flag flip. No
  issue filed yet for this specific follow-up beyond the standing note in
  #1934 itself; file one before starting that audit.
- **`release_host_weights_after_upload` / `uses_device_memory_pool` for ROCm.**
  Left at `false`/default deliberately (see Out of Scope); a discrete ROCm
  card may want different values once measured, same as the CUDA leg's own
  comment describes for its choices.

## Stop conditions

Stop and report rather than widening scope if closing the "Owed" GDN audit
turns out to be a precondition for something THIS row needs — it should not
be: every existing GDN dispatch path is provably unmoved by this row's tests.

Stop rather than reusing `Backend::DeviceMemoryInfo` directly from the
platform registrar (see Risks) even if it looks like less code — the
static-init-order hazard is real and already named as the reason the
registrar probes the device itself elsewhere in this same file.

## Now

`DONE`. Implementation, tests, CPU and real-hardware (gfx1200) evidence are
in. This row closes #1934 and unblocks #1870, which stays open only pending
the operator's merge of this PR (per that issue's own record).

## Outcome

**What was measured.** `RocmPlatform::needs_weight_staging()` gates the ONE
production call site of `CheckDeviceWeightFit`. It hardcodes `false`, so the
refusal never ran on ROCm — confirmed directly, before this row, with
`VT_DEVICE_WEIGHT_BUDGET_BYTES=1` (a budget nothing could satisfy) against a
real checkpoint producing no refusal and proceeding to the next load stage.
Fixed by adding `Platform::allocates_bounded_device_memory()`, a narrower
predicate `RocmPlatform` overrides to `true` while leaving
`needs_weight_staging()` itself untouched, and a real, once-probed
`ResidencyPolicy::device_memory_total_bytes` via a new HIP-free
`vt::rocm::DeviceMemoryTotalBytes` free function.

**Real hardware, both directions, same command, same file, same box.**
`vllm-cli --model /models/Bonsai-27B-Q1_0.gguf --device auto --max-tokens 4`
with `VT_DEVICE_WEIGHT_BUDGET_BYTES=1`:

- BEFORE (stashed the fix, rebuilt): no refusal, 2.72s to reach an unrelated
  dequant error three load-stages later.
- AFTER: refuses in 0.28s —
  `device 'rocm' cannot serve this GGUF: staging its weights needs at least
  3787168768 bytes (3.52 GiB) of device memory across 851 tensors, ... and
  this device's memory pool is 1 bytes (0.00 GiB)`.

**What was rejected.** Flipping `needs_weight_staging()` itself — considered
first, and rejected once reading `IndexedGdnStateIoEnabled` showed it already
takes ROCm's fast arm today via an op-registration check that flipping the
flag would DELETE (see Design), and that at least one sibling consumer
(`MergedGdnBaEnabled` and its relatives) has no such fallback and would
activate unconditionally on unverified kernels. Reusing
`Backend::DeviceMemoryInfo` directly from the platform registrar — rejected
for the static-init-order hazard the registrar's own existing comment already
names, and because it is a live per-request probe with a different, currently
CUDA-dead consumer (issue #1126) that reusing it would silently wake.

**An unrelated pre-existing build break was found and fixed separately, not
in this row.** A fresh CPU build of the claim base (`2a42cb369`) failed under
GCC 15's `-Werror=nonnull` in Tenstorrent debug-dump code
(`qwen3_5.cpp:4019`), unrelated to ROCm or this row's files. Filed as #2021
and fixed as its own tiny PR (#2022, `BACKEND-TENSTORRENT-QWEN35`) rather than
riding this branch, per the same one-issue-one-unit-of-work reasoning this
spec applies to itself — this row's own verification needed it, so it was
applied locally and reverted before every commit in this branch, never
landing here.
