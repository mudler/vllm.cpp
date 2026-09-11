#ifndef VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_
#define VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/grow_only_stream_scratch.h"

namespace vt::rocm::detail {

enum class NormQuantCaptureState { kNone, kActive, kInvalidated };

template <typename QueueKey, typename Stream>
class NormQuantScratchPool {
 public:
  template <typename QueryCapture, typename Allocate>
  void* Ensure(QueueKey queue, Stream stream, size_t bytes,
               QueryCapture&& query_capture, Allocate&& allocate,
               size_t* capacity = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* block = pool_.Ensure(queue, bytes, [&](size_t allocation_bytes) {
      const NormQuantCaptureState capture = query_capture(stream);
      if (capture == NormQuantCaptureState::kActive) {
        throw std::runtime_error(
            "vt rocm: norm-quant scratch: pre-warm RmsNorm on this queue "
            "before graph capture");
      }
      if (capture != NormQuantCaptureState::kNone) {
        throw std::runtime_error(
            "vt rocm: norm-quant scratch: capture is invalidated; end capture "
            "before pre-warming RmsNorm");
      }
      return allocate(allocation_bytes, stream);
    });
    if (capacity != nullptr) *capacity = pool_.CapacityFor(queue);
    return block;
  }

  size_t CapacityFor(QueueKey queue) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.CapacityFor(queue);
  }
  void* BlockFor(QueueKey queue) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.BlockFor(queue);
  }

 private:
  mutable std::mutex mutex_;
  vt::GrowOnlyStreamScratch<QueueKey> pool_;
};

struct NormQuantKey {
  const void* activation = nullptr;
  int64_t rows = 0;
  int64_t hidden = 0;
  int64_t row_stride = 0;
  DType dtype = DType::kF32;
  Device device{};
  uint64_t queue_id = 0;
  const void* stream = nullptr;
  std::thread::id host_thread{};

  friend bool operator==(const NormQuantKey& a, const NormQuantKey& b) {
    return a.activation == b.activation && a.rows == b.rows &&
           a.hidden == b.hidden && a.row_stride == b.row_stride &&
           a.dtype == b.dtype && a.device == b.device &&
           a.queue_id == b.queue_id && a.stream == b.stream &&
           a.host_thread == b.host_thread;
  }
};

struct NormQuantToken {
  NormQuantKey key;
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  uint64_t scratch_generation = 0;
};

// The token is deliberately one-shot. Take removes it before comparison, so a
// mismatch and a concurrent consume both fall back instead of observing stale
// scratch.
class NormQuantTokenRegistry {
 public:
  void Record(const NormQuantToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    token_ = token;
  }

  std::optional<NormQuantToken> Take(const NormQuantKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::optional<NormQuantToken> token = std::move(token_);
    token_.reset();
    if (!token.has_value() || !(token->key == key)) return std::nullopt;
    return token;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    token_.reset();
  }

 private:
  std::mutex mutex_;
  std::optional<NormQuantToken> token_;
};

}  // namespace vt::rocm::detail

#if defined(VLLM_CPP_HIP)
namespace vt::rocm {

struct NormQuantCounts {
  uint64_t producers = 0;
  uint64_t consumers_reused = 0;
  uint64_t consumers_standalone = 0;
};

void* NormQuantProducerScratch(Queue& q, size_t bytes, size_t* scratch_capacity,
                               uint64_t* scratch_generation);
void NormQuantRecordProducer(Queue& q, const void* activation, int64_t rows,
                             int64_t hidden, int64_t row_stride, DType dtype,
                             void* scratch, size_t scratch_bytes,
                             uint64_t scratch_generation);
bool NormQuantTakeConsumer(Queue& q, const void* activation, int64_t rows,
                           int64_t hidden, int64_t row_stride, DType dtype,
                           const void** scratch);
void NormQuantInvalidate(Queue& q);

NormQuantCounts NormQuantCountsForTesting();
void NormQuantResetForTesting();
const void* NormQuantLastScratchForTesting();

}  // namespace vt::rocm
#endif

#endif  // VLLM_CPP_SRC_VT_ROCM_ROCM_NORM_QUANT_BRIDGE_H_
