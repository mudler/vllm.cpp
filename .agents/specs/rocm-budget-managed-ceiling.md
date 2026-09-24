# BACKEND-ROCM-BUDGET-MANAGED-CEILING — correct the ROCm device weight budget from physical total to managed ceiling

Issue: [#2518](https://github.com/mudler/vllm.cpp/issues/2518).
Related: [#1934](https://github.com/mudler/vllm.cpp/issues/1934) (the
`allocates_bounded_device_memory` predicate, already landed),
[#2515](https://github.com/mudler/vllm.cpp/issues/2515) (the device slot store,
the next row this unblocks).
Base: `1e53767c4` (`origin/main` at the claim).

## Scope

`RocmPlatform` is constructed with
`vt::rocm::DeviceMemoryTotalBytes(0)`, which returns `hipMemGetInfo`'s
`total` field — the physical VRAM size, 64.00 GiB on `strix:gpu0`
(gfx1151). Every `Backend::Alloc` on this board goes through
`hipMallocManaged`, whose measured ceiling is 58.0 GiB. The budget the
fit check tests against is therefore ~6 GiB higher than what the
allocator can actually provide. A load that needs 60 GiB passes the fit
check and then dies with `hipMallocManaged: out of memory` during weight
upload.

This row corrects the budget source. One number changes: the
`device_memory_total_bytes` the ROCm platform reports to the fit check.

## What does NOT change

`allocates_bounded_device_memory()` (landed under #1934) stays `true`.
`needs_weight_staging()` stays `true`. The refusal message, the fit
arithmetic, and the `ResolveDeviceWeightBudgetBytes` precedence chain
(`VT_DEVICE_WEIGHT_BUDGET_BYTES` > config > probe) are all untouched.
The `DeviceMemoryTotalBytes` free function stays as-is — other callers
may want the physical total.

## Upstream chain

vLLM does not have this problem because it uses `hipMalloc` (not
`hipMallocManaged`) on discrete GPUs. The managed-alloc path is specific
to this codebase's ROCm backend, selected by `ProbeDevice` when
`pageableMemoryAccess=0` and `managed_memory=1` (the Strix Halo case).
vLLM's budget is `torch.cuda.mem_get_info()[0]` (free bytes), which is
closer to what we need but still not right for managed memory.

## Design

Add `vt::rocm::ManagedMemoryBudgetBytes(int index)` to
`include/vt/rocm/rocm_runtime.h` and implement in
`src/vt/rocm/rocm_backend.hip`. It returns `hipMemGetInfo`'s `free`
field at call time — the bytes available to `hipMallocManaged` right
now, not the physical total. This is the same probe `DeviceMemoryTotalBytes`
already performs; it just returns a different field.

The ROCm platform registrar (`src/vllm/platforms/rocm.cpp:282`) calls
`ManagedMemoryBudgetBytes(0)` instead of
`DeviceMemoryTotalBytes(0)`.

### Why `free`, not `total - overhead`

The spec `rocm-device-fit-bounded-memory.md` considered a derived
ceiling (`total - measured_overhead`). The measured 58 GiB figure on
`strix:gpu0` is `hipMemGetInfo`'s `free` at a quiescent process start.
The 6 GiB gap is not a fixed overhead — it includes driver-reserved
memory, KFD TTM, and display buffers that vary with what else the APU
is doing. `free` is the number the allocator itself consults. Using it
directly is more honest than computing a derived ceiling from a
measurement that may drift.

### Why a new function, not changing `DeviceMemoryTotalBytes`

`DeviceMemoryTotalBytes` has callers that want the physical total (the
`Backend::DeviceMemoryInfo` report, the `--device-info` diagnostic).
Changing its return value would silently shrink those reports. A new
function with a name that says what it returns is the honest seam.

## Risks and decisions

**The budget shrinks by ~6 GiB.** Any load that previously passed the
fit check with a footprint between 58 and 64 GiB will now be refused.
This is correct — those loads were dying at allocation. The operator
override (`VT_DEVICE_WEIGHT_BUDGET_BYTES=0`) still suppresses the
refusal if the operator wants the old behavior.

**`free` varies during the process.** The registrar probes once at
static init. A load that starts when 60 GiB is free but whose
allocation runs after other GPU consumers is not protected by this
check. This was already true — the budget is a pre-load estimate, not
a live allocator. The fix makes the estimate closer to reality, not
perfect.

**Other ROCm boards.** On an APU with `pageableMemoryAccess=1`
(e.g. Phoenix), `HostMemoryIsDeviceAddressable` returns true and the
expert streaming lane uses host memory directly — the budget is less
critical because the fit check's `needs_weight_staging` path is not the
one that runs. On a discrete ROCm GPU with `hipMalloc` (not managed),
`free` is still the right budget. No board is worsened by this change.

## Tests

`test_gguf_device_fit_refusal` already exists and tests the refusal
path. Add a test case that sets the budget to a value between 58 and 64
GiB and verifies the refusal fires for a model whose staged footprint
falls in that range. The existing tests use `VT_DEVICE_WEIGHT_BUDGET_BYTES`
to set the budget directly, so this is a new fixture, not a new
mechanism.

Mutation: change `ManagedMemoryBudgetBytes` to return `total` (i.e.
behave like `DeviceMemoryTotalBytes`) and verify the test goes red.

## Gates

- CPU gate: `test_gguf_device_fit` suite, green.
- `strix:gpu0` gate: GLM-5.3 load attempt — the fit check now refuses
  with the managed ceiling, not the physical total. The refusal message
  quotes the budget it used.

## Evidence

To be recorded from `strix:gpu0` at the gate run.

## Owed

- The `rocm-device-fit-bounded-memory.md` spec's `## Owed` section
  names this correction. Update it to point here when this lands.
- The stale comment at `rocm_ops.hip:370-375` about the DSA indexer
  pair being absent — owed correction, but not this row's work.

## Stop conditions

- If `hipMemGetInfo`'s `free` field on `strix:gpu0` does not reproduce
  the ~58 GiB figure the lane guard record measured, stop and
  re-derive. The budget must match what the allocator actually provides.
- If `ManagedMemoryBudgetBytes` is not callable at static init time
(before `hipSetDevice`), fall back to `DeviceMemoryTotalBytes` and
record the fallback. This is the same defensive pattern
`DeviceMemoryTotalBytes` already uses.

## Now

Row claimed. Spec committed next. Implementation after.
