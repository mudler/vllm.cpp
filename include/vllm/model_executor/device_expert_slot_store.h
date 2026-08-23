// vllm.cpp original (ENG-EXPERT-STREAM-DEVICE W1, issue #1124, spec
// .agents/specs/expert-stream-device-slots.md). A DEVICE-memory
// `ExpertSlotStore`.
//
// There is no upstream for this file. Pinned vLLM `555967922` has no
// inference-time expert paging at all — `vllm/model_executor/offloader/uva.py`
// is a CPU-blanket UVA offloader over whole parameters and
// `vllm/model_executor/offloader/prefetch.py` is cpu-only — so nothing here is
// mirrored and nothing here may claim to be. The reference for correctness is
// `HostExpertSlotStore` on the same input, which is what gate G1 compares
// against.
//
// WHY THIS EXISTS. `HostExpertSlotStore` is the only production destination
// today, so `--device cuda` serves this checkpoint only where the platform's
// kernels can dereference host storage (W0, keyed on the probed
// `host_memory_is_device_addressable()`). That is one part, a GB10. A DISCRETE
// device cannot read the host arena at all, and for it the slice has to live in
// device memory. This is that store.
//
// WHY IT CANNOT SIMPLY BE FILLED. `ExpertStreamer::EnsureFile` hands
// `SlotForWrite()`'s pointer straight to `::pread`, and
// `vt::Backend::DeviceMemoryIsHostAddressable()` is false for CUDA, so a device
// slot pointer is not a legal `pread` destination. That is why W1 carries a
// fill-contract change rather than deferring it: without `CommitSlot` this
// class could not be filled AT ALL, so planning the contract as a later wave
// would plan a wave that deadlocks its predecessor (spec, "Verdict on issue
// #1124's piece 3").
//
// THE FILL IS A STAGING BOUNCE, BY CHOICE AND NOT BY DEFAULT.
// `SlotForWrite` returns ONE pinned host slot, `pread` fills it exactly as it
// fills a host slot today, and `CommitSlot` performs the single contiguous H2D
// into the device slot. A true zero-copy filler (GPUDirect Storage / `cuFile`,
// or `O_DIRECT` DMA into a device BAR mapping) moves fewer bytes and needs a
// driver capability probe, a mount-level check, an aligned-I/O path and a
// fallback for each of those; the bounce costs one extra host-to-device copy of
// one slice per MISS, on top of a disk read of the same size. The measurement
// that would justify replacing it is a device-arm decode where the H2D leg is a
// measurable fraction of fill time, and that measurement does not exist yet.
// Recorded under `## Owed` in the spec, not decided here.
//
// ONE STAGING SLOT, NOT `slots` OF THEM. The filler is synchronous by design
// (`expert_streamer.h`: no async I/O, no prefetch, no read-ahead; overlap is
// `ENG-EXPERT-STREAM` W6 and is conditional on a measurement), so exactly one
// fill is ever in flight. A staging slot per device slot would double the
// arena's host cost — 18.55 GiB on the target checkpoint — to buffer a
// concurrency that does not exist. The single buffer is therefore also the
// thing that makes the write/commit pairing a CONTRACT rather than a
// suggestion, and `CommitSlot` refuses a slot that is not the one `SlotForWrite`
// last handed out instead of committing another expert's bytes.
#ifndef VLLM_MODEL_EXECUTOR_DEVICE_EXPERT_SLOT_STORE_H_
#define VLLM_MODEL_EXECUTOR_DEVICE_EXPERT_SLOT_STORE_H_

#include <cstddef>
#include <cstdint>

#include "vllm/model_executor/expert_streamer.h"
#include "vt/backend.h"

namespace vllm {

class DeviceExpertSlotStore final : public ExpertSlotStore {
 public:
  // `slot_bytes` must be the LARGEST expert slice the caller will stream and is
  // fixed for the store's life, exactly as it is for the host store: a ragged
  // budget makes eviction unpredictable, and a slice that does not fit is
  // refused by the streamer rather than silently truncated.
  //
  // The arena is ONE allocation of `slots * slot_bytes` through
  // `vt::Backend::Alloc`, so a slot is a fixed offset into a contiguous device
  // block — the shape `expert_streamer.h` always described. Throws
  // std::invalid_argument on a degenerate budget or one whose product overflows
  // `size_t`. An allocator that fails PROPAGATES ITS OWN EXCEPTION unchanged --
  // that is how every backend here reports failure, `VT_CHECK` on the CPU
  // backend and `Check(cudaMalloc)`/`Check(cudaHostAlloc)` on CUDA -- and the
  // std::runtime_error this constructor raises itself is reserved for a backend
  // that reports failure by returning nullptr instead, which none in this tree
  // does. Either way NOTHING IS LEAKED: the queue and the arena are given back
  // before the exception leaves, because a constructor that throws runs no
  // destructor.
  DeviceExpertSlotStore(vt::Backend& backend, int32_t slots, size_t slot_bytes);
  ~DeviceExpertSlotStore() override;

  DeviceExpertSlotStore(const DeviceExpertSlotStore&) = delete;
  DeviceExpertSlotStore& operator=(const DeviceExpertSlotStore&) = delete;

  size_t slot_bytes() const override { return slot_bytes_; }
  int32_t slot_count() const override { return slots_; }

  // Copy `bytes` from a HOST buffer into the device slot. This is the path
  // `Ensure`/`EnsureSpan` take, where the caller already holds the bytes, so no
  // staging is involved: the backend copy is itself the H2D.
  void WriteSlot(int32_t slot, const uint8_t* src, size_t bytes) override;

  // The STAGING buffer, not the slot. `pread` writes here; the bytes reach the
  // device only in `CommitSlot`. The returned pointer is the same for every
  // slot and is never inside the arena, which is what makes forgetting the
  // commit a visible defect rather than a silent one.
  uint8_t* SlotForWrite(int32_t slot) override;

  // The single contiguous H2D of the staged bytes into `slot`, then a queue
  // synchronize. The synchronize is load-bearing: `vt::Backend::Copy` is
  // `cudaMemcpyAsync` on CUDA, and both the reuse of the one staging buffer by
  // the next fill and the GEMM that reads the slot immediately after
  // `EnsureFile` returns would otherwise race the transfer.
  //
  // Throws std::logic_error when `slot` is not the slot `SlotForWrite` last
  // handed out, because with one staging buffer that mismatch commits another
  // expert's bytes under this slot's key — the same silent-and-plausible
  // corruption the streamer's `Invalidate` on a failed read exists to prevent.
  void CommitSlot(int32_t slot, size_t bytes) override;

  // The DEVICE bytes of `slot`, for a kernel to read in place. Not host
  // memory: only the backend may dereference it. Non-virtual here on purpose —
  // making the read virtual on `ExpertSlotStore`, so that
  // `Qwen35ExpertStream::Slice` stops reading the concrete
  // `HostExpertSlotStore`, is W2's change and this wave does not pre-empt it.
  uint8_t* SlotForRead(int32_t slot);

  // Device bytes held. The arena is allocated once and never grows, so this is
  // the whole device cost of the lane.
  //
  // It is NOT initialised. `vt::Backend::Alloc` returns raw device memory --
  // `cudaMalloc` on CUDA, `std::aligned_alloc` on the CPU backend -- so a
  // slot holds whatever the allocator last left there until something fills it,
  // and the host store differs here: its arena is a `std::vector<uint8_t>` and
  // is zero-filled at construction. A slot's contents therefore match the host
  // store's over the bytes a fill WROTE and are unspecified past them. Zeroing
  // the arena would cost a full write of the whole budget at load -- 18.55 GiB
  // on the target checkpoint -- to define bytes no reader may look at, since the
  // streamer never hands out a slot it has not filled. Stated because
  // "byte-identical to the host store" is this row's gate and it is true over
  // the filled prefix rather than over the slot.
  int64_t resident_bytes() const {
    return static_cast<int64_t>(static_cast<size_t>(slots_) * slot_bytes_);
  }

 private:
  uint8_t* SlotPtr(int32_t slot) const;

  vt::Backend& b_;
  // Default-constructed, then replaced in the constructor BODY: every budget
  // refusal has to happen before anything is acquired, because a constructor
  // that throws runs no destructor. What is acquired after them is released by
  // the constructor's own catch, for the same reason.
  vt::Queue q_;
  int32_t slots_;
  size_t slot_bytes_;
  uint8_t* arena_ = nullptr;
  uint8_t* staging_ = nullptr;
  // Which slot `SlotForWrite` last handed the staging buffer to, or -1 when
  // nothing is staged. Reset by every commit, so a second commit of the same
  // slot is refused too.
  int32_t staged_slot_ = -1;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_DEVICE_EXPERT_SLOT_STORE_H_
