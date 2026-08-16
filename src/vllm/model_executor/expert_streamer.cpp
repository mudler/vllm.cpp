// ENG-EXPERT-STREAM W3. See expert_streamer.h for why the destination is an
// interface and why the fill is synchronous.
#include "vllm/model_executor/expert_streamer.h"

#include <stdexcept>
#include <string>

namespace vllm {

ExpertStreamer::ExpertStreamer(ExpertSlotCache& cache, ExpertSlotStore& store)
    : cache_(cache), store_(store) {
  if (store.slot_count() != cache.capacity()) {
    // The cache hands out slot indices in [0, capacity). If the store holds
    // fewer, a perfectly ordinary eviction would write past the end of the slot
    // array, so this disagreement is refused at construction rather than on the
    // first unlucky step.
    throw std::invalid_argument(
        "expert streamer: store holds " + std::to_string(store.slot_count()) +
        " slots but the cache has capacity " + std::to_string(cache.capacity()));
  }
}

ExpertStreamer::Result ExpertStreamer::EnsureSpan(const ExpertKey& key,
                                                  const uint8_t* src,
                                                  size_t bytes) {
  Result out;

  // Same ordering rule as the tensor overload, and for the same reason: check
  // the size BEFORE acquiring, or a slice that cannot be stored still evicts a
  // resident expert on its way to being refused.
  if (bytes > store_.slot_bytes()) {
    throw std::invalid_argument(
        "expert streamer: expert span is " + std::to_string(bytes) +
        " bytes but a slot holds " + std::to_string(store_.slot_bytes()));
  }
  if (src == nullptr && bytes > 0) {
    throw std::invalid_argument("expert streamer: null span with bytes > 0");
  }

  const ExpertAcquisition acq = cache_.Acquire(key);
  if (acq.slot < 0) return out;  // exhausted: caller reports, never works around
  if (acq.slot >= store_.slot_count()) {
    throw std::out_of_range(
        "expert streamer: cache returned slot " + std::to_string(acq.slot) +
        " but the store holds " + std::to_string(store_.slot_count()));
  }

  out.slot = acq.slot;
  if (acq.hit) {
    out.hit = true;  // resident: no bytes move, which is the entire saving
    return out;
  }

  store_.WriteSlot(acq.slot, src, bytes);
  bytes_filled_ += static_cast<int64_t>(bytes);
  ++fills_;
  out.filled = true;
  return out;
}

ExpertStreamer::Result ExpertStreamer::Ensure(const ExpertKey& key,
                                              const GgufTensorInfo& tensor,
                                              const GgufExpertLayout& layout) {
  Result out;

  // The size check happens BEFORE the cache is touched. Acquiring first would
  // evict a resident expert to make room for one that cannot be stored, so a
  // configuration error would also destroy a good entry.
  if (layout.expert_bytes > store_.slot_bytes()) {
    throw std::invalid_argument(
        "expert streamer: expert is " + std::to_string(layout.expert_bytes) +
        " bytes but a slot holds " + std::to_string(store_.slot_bytes()) +
        " for " + tensor.name);
  }

  const ExpertAcquisition acq = cache_.Acquire(key);
  if (acq.slot < 0) {
    // The cache refused: every slot is in use by this step. Reported by
    // returning an invalid slot with neither flag set, so the caller cannot
    // mistake it for a hit or a fill.
    return out;
  }
  if (acq.slot >= store_.slot_count()) {
    throw std::out_of_range(
        "expert streamer: cache returned slot " + std::to_string(acq.slot) +
        " but the store holds " + std::to_string(store_.slot_count()));
  }

  out.slot = acq.slot;
  if (acq.hit) {
    // Resident already. No span is computed and no bytes move: this is the
    // saving the whole row exists to produce.
    out.hit = true;
    return out;
  }

  const GgufExpertSpan span = GgufExpertSpanOf(tensor, layout, key.expert);
  store_.WriteSlot(acq.slot, span.data, span.bytes);
  bytes_filled_ += static_cast<int64_t>(span.bytes);
  ++fills_;
  out.filled = true;
  return out;
}

}  // namespace vllm
