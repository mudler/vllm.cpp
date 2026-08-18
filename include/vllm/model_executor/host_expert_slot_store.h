// vllm.cpp original (ENG-EXPERT-STREAM W4). A host-memory `ExpertSlotStore`.
//
// This is the backing store the CPU decode path streams into. It is a fixed
// array of equal-sized slots allocated ONCE, which is the whole point: the
// budget is decided up front and never grows, so a model larger than memory
// cannot page itself to death by admitting one more expert.
//
// Why a slot array rather than letting the kernel demand-page the mmap, which
// already works: measured on `Qwen3.8-2.4T-A95B UD-Q1_0`, the mmap path serves
// each token's ~6.7 GB of expert bytes as 4 KiB faults in ROUTER order, which
// runs near 100 MB/s against an NVMe that sustains ~5 GB/s. A slot is filled by
// ONE contiguous copy of a whole expert slice, so the read is sequential and the
// same bytes can be reused across steps instead of re-faulted every step. See
// the first-run section of `.agents/specs/expert-streaming.md`.
#ifndef VLLM_MODEL_EXECUTOR_HOST_EXPERT_SLOT_STORE_H_
#define VLLM_MODEL_EXECUTOR_HOST_EXPERT_SLOT_STORE_H_

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/expert_streamer.h"

namespace vllm {

class HostExpertSlotStore final : public ExpertSlotStore {
 public:
  // `slot_bytes` must be the LARGEST expert slice the caller will stream, and
  // is fixed for the store's life. A ragged budget would make eviction
  // unpredictable, so a slice that does not fit is refused by the streamer
  // rather than silently truncated.
  HostExpertSlotStore(int32_t slots, size_t slot_bytes)
      : slots_(slots), slot_bytes_(slot_bytes) {
    if (slots <= 0)
      throw std::invalid_argument("HostExpertSlotStore: slot count must be > 0");
    if (slot_bytes == 0)
      throw std::invalid_argument("HostExpertSlotStore: slot bytes must be > 0");
    buf_.resize(static_cast<size_t>(slots) * slot_bytes);
  }

  size_t slot_bytes() const override { return slot_bytes_; }
  int32_t slot_count() const override { return slots_; }

  void WriteSlot(int32_t slot, const uint8_t* src, size_t bytes) override {
    if (slot < 0 || slot >= slots_)
      throw std::out_of_range("HostExpertSlotStore: slot " +
                              std::to_string(slot) + " out of range");
    if (bytes > slot_bytes_)
      throw std::invalid_argument("HostExpertSlotStore: write of " +
                                  std::to_string(bytes) +
                                  " bytes exceeds the slot");
    std::memcpy(Slot(slot), src, bytes);
  }

  // The slot's bytes, for a kernel to read in place. Non-const because the
  // GEMM's weight handle is non-const in this tree; the store never writes
  // through it.
  uint8_t* SlotForWrite(int32_t slot) override { return Slot(slot); }

  uint8_t* Slot(int32_t slot) {
    if (slot < 0 || slot >= slots_)
      throw std::out_of_range("HostExpertSlotStore: slot " +
                              std::to_string(slot) + " out of range");
    return buf_.data() + static_cast<size_t>(slot) * slot_bytes_;
  }

  int64_t resident_bytes() const {
    return static_cast<int64_t>(buf_.size());
  }

 private:
  int32_t slots_;
  size_t slot_bytes_;
  std::vector<uint8_t> buf_;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_HOST_EXPERT_SLOT_STORE_H_
