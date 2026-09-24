# `BACKEND-ROCM-LANE-GUARD` — device-capable expert slot read (W2)

## Problem

`gfx1151` (Strix Halo) reports `pageableMemoryAccess=0`. The lane guard at
`expert_stream_seam.cpp:431` admits the lane on
`cpu || p.host_memory_is_device_addressable()`, so a discrete ROCm device
that answers false to the second condition never enters the lane. Every
expert slice falls through to `resident_fallback`, which stages the whole
tower — exactly what the lane exists to prevent.

`DeviceExpertSlotStore` exists (W1, issue #1124) and is filled through
`EnsureFile`, but nothing SELECTS it. `ExpertStreamLane::store_` is
`unique_ptr<HostExpertSlotStore>`, so the device store is constructible but
unreachable from production. The `ExpertSlotStore` base has no virtual
`SlotForRead`, so the two read sites in `Slice` call the concrete
`HostExpertSlotStore::Slot` — a non-virtual that returns a host pointer a
device kernel on this board cannot dereference.

## Scope

Make `DeviceExpertSlotStore` the production store on discrete ROCm devices.
The lane enters for every platform when streaming is requested; the store
type is selected from the platform in the constructor.

This is `ENG-EXPERT-STREAM-DEVICE` W2 as specified at
`.agents/specs/expert-stream-device-slots.md:1269-1278`.

## Upstream chain

- #1124 (W0 + W1) — unified path and device slot store with fill contract.
  Landed. `DeviceExpertSlotStore` exists, is filled, but never selected.
- #2507 (lane guard fix) — guard reads `allocates_bounded_device_memory()`.
  Landed. The guard's first condition is fixed; the second
  (`host_memory_is_device_addressable()`) is the one this row addresses.
- #2515 (this issue) — `pageableMemoryAccess=0` on `gfx1151` means the
  host-slot lane can never serve on Strix Halo. A device-side slot store is
  the mechanism that does not depend on `pageable_memory_access`.
- #2518 (budget fix, Row 1) — corrects the managed memory ceiling. Independent;
  lands in a different worktree touching different files.

## Design

### 1. Virtual `SlotForRead` on `ExpertSlotStore`

Add `virtual uint8_t* SlotForRead(int32_t slot) = 0;` to the base class
(`expert_streamer.h:43-89`). The two read sites in
`ExpertStreamLane::Slice` currently call the non-virtual
`HostExpertSlotStore::Slot`. After the change they call through the virtual,
which dispatches to the concrete store's device or host pointer.

### 2. `SlotForRead` overrides

- `HostExpertSlotStore` (`host_expert_slot_store.h`): add
  `uint8_t* SlotForRead(int32_t slot) override { return Slot(slot); }`.
  The host store's read pointer is the slot itself.

- `DeviceExpertSlotStore` (`device_expert_slot_store.h`): change the existing
  non-virtual `SlotForRead` to `uint8_t* SlotForRead(int32_t slot) override`.
  The implementation in `device_expert_slot_store.cpp` already returns
  `SlotPtr(slot)` (the device arena pointer). The body is unchanged; only the
  declaration gains `override`.

### 3. `resident_bytes()` on `ExpertSlotStore`

Add `virtual int64_t resident_bytes() const = 0;` to the base. The
constructor calls `store_->resident_bytes()` for the startup banner. Both
concrete stores already have non-virtual `resident_bytes()` — add `override`
to each.

### 4. `store_` becomes `unique_ptr<ExpertSlotStore>`

In `expert_stream_seam.h`, change the member from
`std::unique_ptr<HostExpertSlotStore>` to
`std::unique_ptr<ExpertSlotStore>`. The `ExpertStreamer` constructor takes
`ExpertSlotStore&`, so `*store_` is unchanged.

### 5. Plumb `Dev` into `ExpertStreamLane::Get`

`Get` currently takes `size_t slot_bytes`. The constructor needs
`vt::Backend&` to construct `DeviceExpertSlotStore(backend, slots, slot_bytes)`.
`ExpertSlice` has `Dev d` with `d.b` (backend) and `d.q.device` (device type).

Change `Get(size_t slot_bytes)` to `Get(Dev d, size_t slot_bytes)`. The
constructor takes `Dev d` too. The backend is used only at construction time
to select and build the store; subsequent calls return the existing instance.

`Reserve(size_t slot_bytes)` is unchanged — it records the max slot size and
does not construct.

### 6. Platform-based store selection in the constructor

In `ExpertStreamLane::ExpertStreamLane(Dev d, size_t slot_bytes)`:

```cpp
const vllm::platforms::Platform& p =
    vllm::platforms::GetPlatform(d.q.device.type);
if (p.is_cpu() || p.host_memory_is_device_addressable()) {
  store_ = std::make_unique<HostExpertSlotStore>(slots, slot_bytes);
} else {
  store_ = std::make_unique<DeviceExpertSlotStore>(d.b, slots, slot_bytes);
}
```

CPU and host-addressable devices keep the host store (no change in behavior).
Discrete devices get the device store.

### 7. Read sites: `Slot` → `SlotForRead`

At `expert_stream_seam.cpp:163` and `:219`, change
`store_->Slot(r.slot)` to `store_->SlotForRead(r.slot)`. These are the only
two read sites.

### 8. Widen the `ExpertSlice` gate

At `expert_stream_seam.cpp:431`, remove the
`if (cpu || p.host_memory_is_device_addressable())` condition. The lane is
always entered when streaming is requested. The store selection happens in
the constructor.

The inner exhaustion fallback at line 449 (`if (!cpu)`) currently returns
`HostSliceView` over the tower's host bytes — correct for host-addressable
devices, wrong for discrete devices (the device kernel cannot dereference a
host pointer). Narrow it to `if (!cpu && p.host_memory_is_device_addressable())`.
On a discrete device with an exhausted cache, the slice falls through to
`resident_fallback`, same as CPU.

### 9. Include `device_expert_slot_store.h` in the .cpp

The constructor body constructs `DeviceExpertSlotStore`, so
`expert_stream_seam.cpp` includes `device_expert_slot_store.h`. The header
`expert_stream_seam.h` keeps its existing `host_expert_slot_store.h` include
(transitive consumers may depend on it).

## Stop condition

The spec at `expert-stream-device-slots.md:1275-1278` says: if the returned
pointer's device-ness cannot be expressed in the `vt::Tensor` the GEMM binds
without a second change to the tensor construction, stop and re-scope.

`HostSliceView` constructs a `vt::Tensor` via
`dense_attn::MakeTensor(static_cast<void*>(data), w.dtype, d.q.device, {N, K})`.
On ROCm, `DeviceExpertSlotStore`'s arena is allocated through
`vt::Backend::Alloc`, which goes through `hipMallocManaged`. The returned
pointer is a managed-memory pointer — accessible from both host and device.
The tensor already carries `d.q.device`, so the GEMM knows the device. No
second change to the tensor construction is needed.

The stop condition is not reached.

## Files changed

| File | Change |
|---|---|
| `include/vllm/model_executor/expert_streamer.h` | Add `virtual SlotForRead = 0` and `virtual resident_bytes = 0` to `ExpertSlotStore` |
| `include/vllm/model_executor/host_expert_slot_store.h` | Add `SlotForRead` override, add `override` to `resident_bytes` |
| `include/vllm/model_executor/device_expert_slot_store.h` | Change `SlotForRead` to `override`, add `override` to `resident_bytes` |
| `include/vllm/model_executor/expert_stream_seam.h` | `store_` → `unique_ptr<ExpertSlotStore>`, `Get`/constructor take `Dev d` |
| `src/vllm/model_executor/expert_stream_seam.cpp` | Store selection, `Slot`→`SlotForRead`, widen gate, narrow fallback, include device store header |

## Gates

### CPU gate (green)

Build and run the existing expert-stream test suite on CPU. The host store is
still selected on CPU, so behavior is unchanged. The virtual dispatch adds one
indirection but does not change the pointer returned.

- `test_glm_moe_dsa_forward` — exercises the lane through `ExpertSlice`
- `test_expert_stream_steps` — exercises the step guard
- `test_gguf_device_fit` and `test_gguf_device_fit_reach` — unchanged (different
  files, but verify no regressions from the header changes)

### Mutation gate (G2: call-site-deletion)

In a scratch copy, make `ExpertStreamLane::Slice` return `nullptr` at both
read sites (replace `store_->SlotForRead(r.slot)` with `nullptr`). The lane
appears dead: every slice fails, the test that checks streaming behavior goes
red. This proves the `SlotForRead` virtual is the production read path, not a
class that happens to compile.

### GPU gate (blocked on `strix:gpu0` controller)

GLM-5.3 loads on `strix:gpu0` with `[expert-stream] ON` and decodes tokens
without `hipMallocManaged: out of memory`. The startup banner reports the
device store's resident bytes (slots × slot_bytes) rather than the host
store's `std::vector` size. The `[expert-stream]` final stats line shows
`steps > 0` and `fills > 0`.

Blocked until the `strix:gpu0` controller (`192.168.68.162:30808`) is
reachable. The controller has been returning connection refused; the host
pings fine.

## Risks

| Risk | Call |
|---|---|
| `HostSliceView` name is misleading when the slot is a device pointer. | Naming only, not correctness. The function creates a `vt::Tensor` from any pointer + device pair. Renaming is cosmetic and deferred. |
| `DeviceExpertSlotStore` uses `hipMallocManaged`, not `hipMalloc`. The arena is managed memory, not pure device memory. | Accepted. `vt::Backend::Alloc` on ROCm goes through `hipMallocManaged` (see `rocm_backend.hip`). The pointer is device-readable. Using `hipMalloc` would require a separate allocation path and is a future optimization, not a correctness gate. |
| `ExpertStreamLane` is a process singleton. If `Get` is called with different devices, the first call's store type wins. | Accepted. In the GLM-5.3 case there is one device. The singleton is already process-scoped; the backend is used only at construction time. |
| The exhaustion fallback on a discrete device falls through to `resident_fallback`, which stages the whole tower. | Correct by design. A discrete device with an exhausted slot budget cannot read host bytes in place (`pageableMemoryAccess=0`). The fallback is the resident slice, which is what the lane exists to avoid — but only when the budget is sufficient. |
| Row 1 (#2518) must land first for the GPU gate to be meaningful. | Recorded. The CPU and mutation gates do not depend on Row 1. The GPU gate needs the corrected budget to verify the load fits. |

## Owed

- GPU gate evidence on `strix:gpu0` once the controller is reachable.
- Update `rocm-device-fit-bounded-memory.md` `## Owed` to point to the #2518
  fix when it lands (owed from Row 1, not this row).
- Stale comment at `rocm_ops.hip:370-375` claims the DSA indexer pair is absent
  — owed correction for the next row touching that file.
