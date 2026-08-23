// ENG-EXPERT-STREAM-DEVICE W1 (issue #1124). See device_expert_slot_store.h for
// why the destination is device memory, why the fill is a staging bounce, and
// why there is exactly one staging buffer.
#include "vllm/model_executor/device_expert_slot_store.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace vllm {

DeviceExpertSlotStore::DeviceExpertSlotStore(vt::Backend& backend,
                                             int32_t slots, size_t slot_bytes)
    : b_(backend), slots_(slots), slot_bytes_(slot_bytes) {
  // Every BUDGET refusal below happens before the queue is created,
  // deliberately: a constructor that throws has no destructor run, so anything
  // acquired above the throw leaks. Nothing is acquired until the budget is
  // known good. The acquisitions themselves throw too, and they are wrapped for
  // the same reason -- see the note above them.
  if (slots <= 0) {
    throw std::invalid_argument(
        "DeviceExpertSlotStore: slot count must be > 0");
  }
  if (slot_bytes == 0) {
    throw std::invalid_argument("DeviceExpertSlotStore: slot bytes must be > 0");
  }
  // The host store's arena is a `std::vector`, whose own length check catches
  // this; a raw `Alloc` has no such backstop, and an arena that wrapped would
  // hand out in-range slot pointers past its end — a silent overwrite rather
  // than a refusal.
  if (slot_bytes > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(slots)) {
    throw std::invalid_argument(
        "DeviceExpertSlotStore: " + std::to_string(slots) + " slots of " +
        std::to_string(slot_bytes) + " bytes overflows size_t");
  }

  const size_t total = static_cast<size_t>(slots) * slot_bytes;
  // ACQUISITION IS ALL-OR-NOTHING FROM HERE DOWN, and the failure it is written
  // for is the one that actually happens. No backend in this tree returns
  // nullptr from an allocator: `CpuBackend::Alloc` refuses with `VT_CHECK`,
  // `CudaBackend::Alloc` and `AllocPinned` refuse through `Check(...)`, and the
  // base `Backend::AllocPinned` forwards to `Alloc`. They THROW, and out of
  // memory is this class's headline failure -- issue #1123 is literally
  // `vt cuda: cudaMalloc: out of memory`. Unwrapped, a throw from `Alloc`
  // strands the queue and a throw from `AllocPinned` strands the queue and the
  // whole device arena, 18.55 GiB on the target checkpoint, at the exact moment
  // the device has no memory left to lose. The catch below gives back whatever
  // this constructor took and rethrows unchanged, so the caller still sees the
  // backend's own message.
  q_ = b_.CreateQueue();
  try {
    arena_ = static_cast<uint8_t*>(b_.Alloc(total));
    // Kept although unreachable through any backend here, because `vt::Backend`
    // is an interface and a nullptr-returning implementation would otherwise
    // hand out slot pointers off a null arena instead of being refused. It now
    // costs one branch and no cleanup code, since the catch owns the release.
    if (arena_ == nullptr) {
      throw std::runtime_error("DeviceExpertSlotStore: device allocation of " +
                               std::to_string(total) + " bytes failed");
    }
    // Page-locked, because this buffer is the source of every H2D the lane
    // issues: on CUDA a copy from pageable memory stages through a driver
    // bounce of its own, which is the one thing this design must not pay twice.
    // The base implementation returns ordinary host memory, which is correct on
    // a backend where the distinction does not exist.
    staging_ = static_cast<uint8_t*>(b_.AllocPinned(slot_bytes));
    if (staging_ == nullptr) {  // unreachable here for the same reason
      throw std::runtime_error("DeviceExpertSlotStore: pinned staging "
                               "allocation of " + std::to_string(slot_bytes) +
                               " bytes failed");
    }
  } catch (...) {
    // The destructor's body, because that is exactly what did not run.
    if (staging_ != nullptr) b_.FreePinned(staging_);
    if (arena_ != nullptr) b_.Free(arena_);
    b_.DestroyQueue(q_);
    throw;
  }
}

DeviceExpertSlotStore::~DeviceExpertSlotStore() {
  if (staging_ != nullptr) b_.FreePinned(staging_);
  if (arena_ != nullptr) b_.Free(arena_);
  b_.DestroyQueue(q_);
}

uint8_t* DeviceExpertSlotStore::SlotPtr(int32_t slot) const {
  if (slot < 0 || slot >= slots_) {
    throw std::out_of_range("DeviceExpertSlotStore: slot " +
                            std::to_string(slot) + " out of range");
  }
  return arena_ + static_cast<size_t>(slot) * slot_bytes_;
}

void DeviceExpertSlotStore::WriteSlot(int32_t slot, const uint8_t* src,
                                      size_t bytes) {
  uint8_t* dst = SlotPtr(slot);  // bounds first, as the host store does
  if (bytes > slot_bytes_) {
    throw std::invalid_argument("DeviceExpertSlotStore: write of " +
                                std::to_string(bytes) +
                                " bytes exceeds the slot");
  }
  if (src == nullptr && bytes > 0) {
    throw std::invalid_argument("DeviceExpertSlotStore: null source");
  }
  if (bytes == 0) return;
  b_.Copy(q_, dst, src, bytes);
  // Synchronous by contract: the caller binds this slot to a GEMM as soon as
  // the streamer returns, and `Copy` is asynchronous on CUDA.
  b_.Synchronize(q_);
}

uint8_t* DeviceExpertSlotStore::SlotForWrite(int32_t slot) {
  if (slot < 0 || slot >= slots_) {
    throw std::out_of_range("DeviceExpertSlotStore: slot " +
                            std::to_string(slot) + " out of range");
  }
  staged_slot_ = slot;
  return staging_;
}

void DeviceExpertSlotStore::CommitSlot(int32_t slot, size_t bytes) {
  uint8_t* dst = SlotPtr(slot);
  if (bytes > slot_bytes_) {
    throw std::invalid_argument("DeviceExpertSlotStore: commit of " +
                                std::to_string(bytes) +
                                " bytes exceeds the slot");
  }
  if (staged_slot_ != slot) {
    // With one staging buffer this is not a bookkeeping slip: the bytes in
    // staging belong to whichever slot asked for it last, so committing them
    // here would file one expert's weights under another expert's key. The
    // cache would then report a HIT for a key whose slot holds the wrong
    // expert, and the GEMM would multiply it without a symptom.
    throw std::logic_error(
        "DeviceExpertSlotStore: commit of slot " + std::to_string(slot) +
        " but SlotForWrite last staged " + std::to_string(staged_slot_));
  }
  staged_slot_ = -1;
  if (bytes == 0) return;
  b_.Copy(q_, dst, staging_, bytes);
  b_.Synchronize(q_);
}

uint8_t* DeviceExpertSlotStore::SlotForRead(int32_t slot) {
  return SlotPtr(slot);
}

}  // namespace vllm
