// ENG-EXPERT-STREAM W3 (issue #912, spec .agents/specs/expert-streaming.md).
//
// Where W1 and W2 meet: the cache says which slot an expert should occupy and
// whether it is already there, the span says which bytes it is, and this fills
// the slot on a miss.
//
// WHY THE DESTINATION IS AN INTERFACE. The production destination is a
// contiguous device-side slot array, so a concrete device dependency here would
// make the whole streaming policy untestable without a GPU, and this row's
// hardware is the scarcest thing about it. `ExpertSlotStore` is the seam:
// production writes to device memory, tests write to a host buffer, and the
// policy above it is identical in both.
//
// WHAT THIS DELIBERATELY DOES NOT DO. It performs no asynchronous I/O, no
// prefetch and no read-ahead. A miss is a synchronous fill. That is the spec's
// phase 1 (`W3 phase-1 decode dispatch`), and the overlap work is W6, which the
// spec makes conditional on a measurement rather than assumed. Building the
// synchronous version first is what produces the measurement.
#ifndef VLLM_MODEL_EXECUTOR_EXPERT_STREAMER_H_
#define VLLM_MODEL_EXECUTOR_EXPERT_STREAMER_H_

#include <cstddef>
#include <cstdint>

#include "vllm/model_executor/expert_slot_cache.h"
#include "vllm/model_executor/model_loader/gguf_expert_span.h"

namespace vllm {

// The destination for an expert's bytes. One implementation writes into a
// device slot array; the test implementation writes into host memory.
class ExpertSlotStore {
 public:
  virtual ~ExpertSlotStore() = default;

  // Bytes reserved per slot. An expert whose span exceeds this cannot be
  // stored, and the streamer refuses rather than writing past the slot.
  virtual size_t slot_bytes() const = 0;

  // Number of slots this store holds. Must match the cache's capacity, or the
  // cache would hand out slots the store does not have.
  virtual int32_t slot_count() const = 0;

  // Copy `bytes` from `src` into `slot`. Implementations may assume the
  // streamer has already validated the slot index and the size.
  virtual void WriteSlot(int32_t slot, const uint8_t* src, size_t bytes) = 0;
};

class ExpertStreamer {
 public:
  // Neither reference is owned. Throws std::invalid_argument when the store's
  // slot count disagrees with the cache's capacity, because that mismatch would
  // otherwise surface as a write to a slot that does not exist.
  ExpertStreamer(ExpertSlotCache& cache, ExpertSlotStore& store);

  struct Result {
    int32_t slot = -1;
    bool hit = false;     // already resident: no bytes moved
    bool filled = false;  // a miss that was filled from the tensor
  };

  // Make `key`'s expert resident and return its slot.
  //
  // On a hit nothing is read and nothing is written: that is the entire point
  // of the cache, and `filled` stays false so a caller can count real I/O.
  //
  // Throws std::invalid_argument when the expert's span does not fit a slot,
  // and std::out_of_range when the cache returns a slot the store does not
  // have. Returns slot -1 with neither flag set when the cache is exhausted,
  // which is a budget error the caller must report rather than work around.
  Result Ensure(const ExpertKey& key, const GgufTensorInfo& tensor,
                const GgufExpertLayout& layout);

  void EndStep() { cache_.EndStep(); }

  // Bytes actually moved, which is the number a streaming benchmark reports.
  // Hits contribute nothing, so this is I/O and not traffic.
  int64_t bytes_filled() const { return bytes_filled_; }
  int64_t fills() const { return fills_; }
  const ExpertSlotCache& cache() const { return cache_; }

 private:
  ExpertSlotCache& cache_;
  ExpertSlotStore& store_;
  int64_t bytes_filled_ = 0;
  int64_t fills_ = 0;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_EXPERT_STREAMER_H_
