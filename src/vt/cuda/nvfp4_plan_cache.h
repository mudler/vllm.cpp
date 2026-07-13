// vllm.cpp — host-side NVFP4 autotune plan identity and single-flight cache.
//
// The hybrid M mapping mirrors FlashInfer 0.6.12
// `flashinfer/fused_moe/utils.py:212-307`; the one-plan-per-complete-key and
// single-flight publication mirror its autotuner cache contract at
// `flashinfer/autotuner.py:1384-1584`. This header contains no CUDA API calls so
// the state machine and exact bucket boundaries remain CPU-unit-testable. The
// CUDA adapter is responsible for rejecting a cache miss during stream capture.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vt::cuda::nvfp4 {

inline uint32_t NextPositivePowerOfTwo(uint32_t value) {
  if (value <= 1) return 1;
  if (value > (uint32_t{1} << 31)) {
    throw std::overflow_error("NVFP4 M bucket exceeds uint32 power-of-two range");
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1;
}

// FlashInfer map_to_hybrid_bucket_uncapped(): powers of two through 256,
// 256-wide steps through 2048, 512-wide steps through 4096, then powers of two.
inline uint32_t HybridMBucket(uint32_t value) {
  if (value == 0) return 1;
  if (value <= 256) return NextPositivePowerOfTwo(value);
  if (value <= 2048) return ((value + 255) / 256) * 256;
  if (value <= 4096) return ((value + 511) / 512) * 512;
  return NextPositivePowerOfTwo(value);
}

// FlashInfer get_hybrid_num_tokens_buckets(): enumerate every optimization
// profile materialized by a max-token autotune dummy run. The final concrete
// maximum is retained even when it is not a phase boundary. This is deliberately
// separate from HybridMBucket(), which mirrors the uncapped lookup mapper.
inline std::vector<uint32_t> HybridMTuningBuckets(uint32_t max_num_tokens,
                                                  uint32_t min_num_tokens = 1) {
  std::vector<uint32_t> buckets;
  const uint32_t minimum = std::max(min_num_tokens, uint32_t{1});

  for (uint32_t value = minimum;
       value <= std::min(max_num_tokens, uint32_t{256});) {
    buckets.push_back(value);
    if (value > std::numeric_limits<uint32_t>::max() / 2) break;
    value *= 2;
  }
  for (uint32_t value = 512;
       value <= std::min(max_num_tokens, uint32_t{2048}); value += 256) {
    buckets.push_back(value);
  }
  for (uint32_t value = 2560;
       value <= std::min(max_num_tokens, uint32_t{4096}); value += 512) {
    buckets.push_back(value);
  }
  for (uint32_t value = 8192; value <= max_num_tokens;) {
    buckets.push_back(value);
    if (value > std::numeric_limits<uint32_t>::max() / 2) break;
    value *= 2;
  }
  if (buckets.empty() || buckets.back() != max_num_tokens) {
    buckets.push_back(max_num_tokens);
  }
  std::sort(buckets.begin(), buckets.end());
  buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
  return buckets;
}

// Exact same-binary fallback for VT_FP4_EXACT_BUCKETS=0.
inline uint32_t LegacyMBucket(uint32_t value) {
  const uint32_t bucket = NextPositivePowerOfTwo(value);
  return bucket < 16 ? 16 : bucket;
}

// W1 and W2 are deliberately distinct same-binary tactic ABIs. These values are
// part of the key now and become persistent-cache compatibility versions in W3.
inline constexpr uint32_t kW1TacticSetVersion = 1;
inline constexpr uint32_t kFullTacticSetVersion = 2;
inline constexpr uint32_t kTacticSetVersion = kFullTacticSetVersion;

struct PlanKey {
  uint32_t m_bucket = 0;
  int32_t n = 0;
  int32_t k = 0;
  int32_t device_ordinal = 0;
  int32_t architecture = 0;  // encoded CUDA SM, e.g. 121
  uint8_t output_dtype = 0;
  uint32_t tactic_set_version = kTacticSetVersion;

  bool operator==(const PlanKey&) const = default;
};

struct PlanKeyHash {
  size_t operator()(const PlanKey& key) const noexcept {
    size_t seed = 0;
    Combine(seed, std::hash<uint32_t>{}(key.m_bucket));
    Combine(seed, std::hash<int32_t>{}(key.n));
    Combine(seed, std::hash<int32_t>{}(key.k));
    Combine(seed, std::hash<int32_t>{}(key.device_ordinal));
    Combine(seed, std::hash<int32_t>{}(key.architecture));
    Combine(seed, std::hash<uint8_t>{}(key.output_dtype));
    Combine(seed, std::hash<uint32_t>{}(key.tactic_set_version));
    return seed;
  }

 private:
  static void Combine(size_t& seed, size_t value) noexcept {
    constexpr size_t kGolden = sizeof(size_t) == 8 ? size_t{0x9e3779b97f4a7c15ULL}
                                                  : size_t{0x9e3779b9UL};
    seed ^= value + kGolden + (seed << 6) + (seed >> 2);
  }
};

// One tuner owns a missing key. Same-key callers wait for that complete result;
// different keys never hold the map mutex while tuning. A failed attempt wakes
// every waiter and is removed atomically so a later call can retry.
template <typename Value, typename Hash = PlanKeyHash>
class SingleFlightPlanCache {
  static_assert(std::is_copy_constructible_v<Value>);

 public:
  // Install a validated persistent result without exposing an intermediate
  // tuning state. Existing live/loaded ready results retain priority, matching
  // FlashInfer's in-memory-before-file lookup order. Importing over an active
  // tuner is a lifecycle error: persistent plans must be loaded before warmup.
  bool InsertReadyIfAbsent(const PlanKey& key, Value value) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    const auto it = entries_.find(key);
    if (it == entries_.end()) {
      auto entry = std::make_shared<Entry>();
      entry->value.emplace(std::move(value));
      entry->state = State::kReady;
      entries_.emplace(key, std::move(entry));
      return true;
    }

    std::lock_guard<std::mutex> entry_lock(it->second->mutex);
    if (it->second->state == State::kTuning) {
      throw std::logic_error(
          "NVFP4 persistent plan import collided with active tuning");
    }
    if (it->second->state != State::kReady) {
      throw std::logic_error(
          "NVFP4 persistent plan import found a failed cache entry");
    }
    return false;
  }

  std::optional<Value> FindReady(const PlanKey& key) const {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      const auto it = entries_.find(key);
      if (it == entries_.end()) return std::nullopt;
      entry = it->second;
    }
    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->state != State::kReady) return std::nullopt;
    return entry->value;
  }

  // Snapshot only executable ready entries. Holding the map lock while each
  // entry is copied preserves the global->entry lock order used by failure and
  // import paths. Sorting makes persistence deterministic despite unordered_map.
  std::vector<std::pair<PlanKey, Value>> SnapshotReady() const {
    std::vector<std::pair<PlanKey, Value>> result;
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    result.reserve(entries_.size());
    for (const auto& [key, entry] : entries_) {
      std::lock_guard<std::mutex> entry_lock(entry->mutex);
      if (entry->state == State::kReady) result.emplace_back(key, *entry->value);
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
      const PlanKey& a = lhs.first;
      const PlanKey& b = rhs.first;
      return std::tie(a.m_bucket, a.n, a.k, a.device_ordinal, a.architecture,
                      a.output_dtype, a.tactic_set_version) <
             std::tie(b.m_bucket, b.n, b.k, b.device_ordinal, b.architecture,
                      b.output_dtype, b.tactic_set_version);
    });
    return result;
  }

  template <typename Tune>
  Value GetOrTune(const PlanKey& key, Tune&& tune) {
    std::shared_ptr<Entry> entry;
    bool owner = false;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      const auto it = entries_.find(key);
      if (it == entries_.end()) {
        entry = std::make_shared<Entry>();
        entries_.emplace(key, entry);
        owner = true;
      } else {
        entry = it->second;
      }
    }

    if (!owner) {
      std::unique_lock<std::mutex> lock(entry->mutex);
      if (entry->state == State::kTuning) {
        ++entry->waiters;
        entry->ready.wait(lock, [&] { return entry->state != State::kTuning; });
        --entry->waiters;
      }
      if (entry->state == State::kReady) return *entry->value;
      const std::exception_ptr failure = entry->failure;
      lock.unlock();
      std::rethrow_exception(failure);
    }

    try {
      Value value = std::forward<Tune>(tune)();
      {
        std::lock_guard<std::mutex> lock(entry->mutex);
        entry->value.emplace(std::move(value));
        entry->state = State::kReady;
      }
      entry->ready.notify_all();
      return *entry->value;
    } catch (...) {
      const std::exception_ptr failure = std::current_exception();
      {
        // Keep the global->entry lock order used by FindReady. Erasing before
        // notification makes a subsequent caller a fresh owner immediately.
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        std::lock_guard<std::mutex> entry_lock(entry->mutex);
        entry->failure = failure;
        entry->state = State::kFailed;
        const auto it = entries_.find(key);
        if (it != entries_.end() && it->second == entry) entries_.erase(it);
      }
      entry->ready.notify_all();
      std::rethrow_exception(failure);
    }
  }

  size_t SizeForTesting() const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return entries_.size();
  }

  size_t WaiterCountForTesting(const PlanKey& key) const {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      const auto it = entries_.find(key);
      if (it == entries_.end()) return 0;
      entry = it->second;
    }
    std::lock_guard<std::mutex> lock(entry->mutex);
    return entry->waiters;
  }

 private:
  enum class State { kTuning, kReady, kFailed };

  struct Entry {
    std::mutex mutex;
    std::condition_variable ready;
    State state = State::kTuning;
    size_t waiters = 0;
    std::optional<Value> value;
    std::exception_ptr failure;
  };

  mutable std::mutex map_mutex_;
  std::unordered_map<PlanKey, std::shared_ptr<Entry>, Hash> entries_;
};

// `can_tune` is lazy: a ready hit never asks CUDA whether capture is active.
// The CUDA adapter passes a predicate backed by cudaStreamIsCapturing; CPU tests
// pass a deterministic predicate to cover the fail-closed miss contract.
template <typename Value, typename Hash, typename CanTune, typename Tune>
Value ResolvePlan(SingleFlightPlanCache<Value, Hash>& cache, const PlanKey& key,
                  CanTune&& can_tune, Tune&& tune) {
  if (const auto ready = cache.FindReady(key); ready.has_value()) return *ready;
  if (!std::forward<CanTune>(can_tune)()) {
    throw std::runtime_error("NVFP4 plan cache miss while tuning is disallowed");
  }
  return cache.GetOrTune(key, std::forward<Tune>(tune));
}

}  // namespace vt::cuda::nvfp4
