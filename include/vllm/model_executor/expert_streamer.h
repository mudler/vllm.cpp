// ENG-EXPERT-STREAM W3 (issue #912, spec .agents/specs/expert-streaming.md).
//
// Where W1 and W2 meet: the cache says which slot an expert should occupy and
// whether it is already there, the span says which bytes it is, and this fills
// the slot on a miss.
//
// WHY THE DESTINATION IS AN INTERFACE. A destination may be a contiguous
// device-side slot array, so a concrete device dependency here would make the
// whole streaming policy untestable without a GPU, and this row's hardware is
// the scarcest thing about it. `ExpertSlotStore` is the seam and the policy
// above it is identical for every implementation.
//
// WHICH DESTINATION PRODUCTION ACTUALLY USES, as of ENG-EXPERT-STREAM-DEVICE W1
// (issue #1124). Two sentences here used to say "the production destination is
// a contiguous device-side slot array" and "production writes to device
// memory". Both were false: `HostExpertSlotStore` was the only production
// implementation, and it still is the only one anything SELECTS.
// `DeviceExpertSlotStore` (device_expert_slot_store.h) now exists and is
// filled through `EnsureFile` below, but `Qwen35ExpertStream` still holds the
// concrete host store and reads it through `HostExpertSlotStore::Slot`, so no
// load can reach the device one yet. Making the read virtual and selecting the
// store from the platform is W2 of the same row, and until it lands this
// comment says what is true rather than what is intended.
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

  // A HOST-WRITABLE destination for `slot`'s bytes, for a filler that produces
  // them in place: `pread` writes straight here. Must return at least
  // `slot_bytes()` writable bytes or throw.
  //
  // It is NOT required to be the slot itself, and on a device store it is not:
  // a `cudaMalloc` pointer is not a legal `pread` destination, because
  // `vt::Backend::DeviceMemoryIsHostAddressable()` is false for CUDA. The
  // host store returns the slot and the device store returns a staging buffer;
  // `CommitSlot` below is what makes the difference invisible to the filler.
  virtual uint8_t* SlotForWrite(int32_t slot) = 0;

  // Publish the `bytes` a filler just wrote through `SlotForWrite(slot)`, so
  // that a later read of `slot` sees them.
  //
  // WHY THIS EXISTS AT ALL, since a host store needs nothing here. Without it
  // `ExpertSlotStore` cannot describe a destination that is not host memory,
  // and a device store could not be filled by ANY caller — the fill contract
  // was `pread`-into-the-slot and nothing else (issue #1124, piece 3). It is
  // therefore part of W1 rather than a wave after it: deferring it would defer
  // the only thing that makes the class fillable.
  //
  // PURE, not a defaulted no-op. A default would be correct for exactly one
  // implementation, the host one, and silently wrong for every store whose
  // slots the host cannot write — which is the whole population this method was
  // added for. An implementer who does nothing must say so.
  //
  // Called by `ExpertStreamer::EnsureFile` on the SUCCESS path only: a fill
  // that threw leaves nothing to publish, and the cache entry is invalidated
  // instead.
  virtual void CommitSlot(int32_t slot, size_t bytes) = 0;
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

  // Same contract, for a caller that already holds the expert's SPAN rather
  // than the GGUF record it came from. The decode path is such a caller: by the
  // time a model runs, an expert tower is a borrowed byte range plus a row
  // offset, and the `GgufTensorInfo` that produced it is long gone. Routing
  // that caller through the tensor overload would mean reconstructing a record
  // just to be taken apart again.
  //
  // `src` must stay valid for the call; the slot owns a copy afterwards.
  Result EnsureSpan(const ExpertKey& key, const uint8_t* src, size_t bytes);

  // Fill from the FILE rather than from a mapping. This is the form the design
  // always specified ("reads are plain pread(2) against the model fd") and the
  // one that actually changes the I/O: EnsureSpan's memcpy still traps every
  // 4 KiB page of its source on the way, so it inherits the fault path the
  // whole lane exists to bypass.
  //
  // Falls back to nothing: a short read or a bad descriptor throws, because a
  // partially filled slot decodes to garbage silently.
  Result EnsureFile(const ExpertKey& key, int fd, size_t file_offset,
                    size_t bytes);

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
