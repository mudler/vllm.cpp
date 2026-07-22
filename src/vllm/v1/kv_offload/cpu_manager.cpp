// Ported from: vllm/v1/kv_offload/cpu/manager.py:36-309 @ e24d1b24
// See include/vllm/v1/kv_offload/cpu_manager.h for scope and the tier model.
#include "vllm/v1/kv_offload/cpu_manager.h"

#include <algorithm>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace vllm::v1::kv_offload {

const char* CPULoadStoreSpec::kMedium = "CPU";

namespace {
// Insertion-ordered counter map with an O(1) move-to-end and popitem(last=False),
// mirroring the `counts: OrderedDict[OffloadKey, int]` reuse tracker
// (cpu/manager.py:73-76).
class OrderedCounts {
 public:
  int64_t get(const OffloadKey& key) const {
    auto it = entries_.find(key);
    return it == entries_.end() ? 0 : it->second.count;
  }
  // Bump the key's count and make it most-recent; insert at 1 when absent,
  // evicting the least-recent entry first if the tracker is full.
  void bump(const OffloadKey& key, size_t max_size) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      order_.splice(order_.end(), order_, it->second.order_it);
      it->second.count += 1;
      return;
    }
    if (entries_.size() >= max_size && !order_.empty()) {
      entries_.erase(order_.front());
      order_.pop_front();
    }
    order_.push_back(key);
    entries_.emplace(key, Entry{std::prev(order_.end()), 1});
  }

 private:
  struct Entry {
    std::list<OffloadKey>::iterator order_it;
    int64_t count;
  };
  std::list<OffloadKey> order_;
  std::unordered_map<OffloadKey, Entry> entries_;
};
}  // namespace

struct CPUOffloadingManager::Impl {
  int64_t num_blocks = 0;
  int64_t num_allocated_blocks = 0;
  std::vector<int64_t> free_list;
  bool events_enabled = false;
  std::vector<OffloadingEvent> events;
  std::unique_ptr<CachePolicy> policy;
  // Blocks in the cache with ref_cnt == 0 (i.e. evictable).
  int64_t num_evictable_cache_blocks = 0;
  int64_t store_threshold = 1;
  int64_t max_tracker_size = 64000;
  int64_t stores_skipped_in_current_batch = 0;
  // Only materialized when store_threshold >= 2 (cpu/manager.py:73-76).
  std::unique_ptr<OrderedCounts> counts;

  int64_t get_num_free_blocks() const {
    return static_cast<int64_t>(free_list.size()) + num_blocks -
           num_allocated_blocks;
  }

  // Bump allocator over fresh ids, then the free list (cpu/manager.py:83-97).
  std::vector<BlockStatus> allocate_blocks(size_t count) {
    const int64_t num_fresh =
        std::min(static_cast<int64_t>(count), num_blocks - num_allocated_blocks);
    const int64_t num_reused = static_cast<int64_t>(count) - num_fresh;
    if (static_cast<int64_t>(free_list.size()) < num_reused) {
      throw std::logic_error(
          "CPUOffloadingManager: free list too small for the requested blocks");
    }
    std::vector<BlockStatus> blocks;
    blocks.reserve(count);
    for (int64_t i = 0; i < num_fresh; ++i) {
      blocks.emplace_back(num_allocated_blocks);
      num_allocated_blocks += 1;
    }
    for (int64_t i = 0; i < num_reused; ++i) {
      blocks.emplace_back(free_list.back());
      free_list.pop_back();
    }
    return blocks;
  }

  void free_block(const BlockStatus& block) {
    free_list.push_back(block.block_id);
  }

  static std::shared_ptr<LoadStoreSpec> make_spec(
      const std::vector<BlockStatus>& blocks) {
    std::vector<int64_t> ids;
    ids.reserve(blocks.size());
    for (const BlockStatus& b : blocks) {
      ids.push_back(b.block_id);
    }
    return std::make_shared<CPULoadStoreSpec>(std::move(ids));
  }
};

CPUOffloadingManager::CPUOffloadingManager(int64_t num_blocks,
                                           const std::string& cache_policy,
                                           bool enable_events,
                                           int64_t store_threshold,
                                           int64_t max_tracker_size)
    : impl_(std::make_unique<Impl>()) {
  impl_->num_blocks = num_blocks;
  impl_->events_enabled = enable_events;
  // make_cache_policy throws std::invalid_argument on an unknown name,
  // mirroring upstream's ValueError (cpu/manager.py:60-64).
  impl_->policy = make_cache_policy(cache_policy, num_blocks);
  impl_->store_threshold = store_threshold;
  impl_->max_tracker_size = max_tracker_size;
  if (store_threshold >= 2) {
    impl_->counts = std::make_unique<OrderedCounts>();
  }
}

CPUOffloadingManager::~CPUOffloadingManager() = default;

int64_t CPUOffloadingManager::num_blocks() const { return impl_->num_blocks; }

int64_t CPUOffloadingManager::num_free_blocks() const {
  return impl_->get_num_free_blocks();
}

LookupResult CPUOffloadingManager::lookup(const OffloadKey& key,
                                          const ReqContext& /*req_context*/) {
  if (impl_->counts) {
    impl_->counts->bump(key, static_cast<size_t>(impl_->max_tracker_size));
  }
  BlockStatus* block = impl_->policy->get(key);
  if (block == nullptr) {
    return LookupResult::kMiss;
  }
  // Present but a store is still in flight: NOT a hit, and NOT a miss (a miss
  // would cause the caller to store it a second time).
  if (!block->is_ready()) {
    return LookupResult::kHitPending;
  }
  return LookupResult::kHit;
}

std::shared_ptr<LoadStoreSpec> CPUOffloadingManager::prepare_load(
    const std::vector<OffloadKey>& keys, const ReqContext& /*req_context*/) {
  std::vector<BlockStatus> blocks;
  blocks.reserve(keys.size());
  for (const OffloadKey& key : keys) {
    BlockStatus* block = impl_->policy->get(key);
    if (block == nullptr) {
      throw std::logic_error("CPUOffloadingManager: block not found in cache");
    }
    if (!block->is_ready()) {
      throw std::logic_error(
          "CPUOffloadingManager: block is not ready for reading");
    }
    if (block->ref_cnt == 0) {
      impl_->policy->mark_non_evictable(key);
      impl_->num_evictable_cache_blocks -= 1;  // ref_cnt 0 -> 1
      if (impl_->num_evictable_cache_blocks < 0) {
        throw std::logic_error(
            "CPUOffloadingManager: evictable block count went negative");
      }
    }
    block->ref_cnt += 1;
    blocks.push_back(*block);
  }
  return Impl::make_spec(blocks);
}

void CPUOffloadingManager::touch(const std::vector<OffloadKey>& keys,
                                 const ReqContext& /*req_context*/) {
  impl_->policy->touch(keys);
}

void CPUOffloadingManager::complete_load(const std::vector<OffloadKey>& keys,
                                         const ReqContext& /*req_context*/) {
  for (const OffloadKey& key : keys) {
    BlockStatus* block = impl_->policy->get(key);
    if (block == nullptr) {
      throw std::logic_error("CPUOffloadingManager: block not found");
    }
    if (block->ref_cnt <= 0) {
      throw std::logic_error("CPUOffloadingManager: block ref_cnt already 0");
    }
    block->ref_cnt -= 1;
    if (block->ref_cnt == 0) {
      impl_->num_evictable_cache_blocks += 1;  // ref_cnt 1 -> 0
      impl_->policy->mark_evictable(key);
    }
  }
}

std::optional<PrepareStoreOutput> CPUOffloadingManager::prepare_store(
    const std::vector<OffloadKey>& keys, const ReqContext& /*req_context*/) {
  std::vector<OffloadKey> considered = keys;
  if (impl_->counts) {
    const size_t before = considered.size();
    std::vector<OffloadKey> filtered;
    filtered.reserve(before);
    for (const OffloadKey& k : considered) {
      if (impl_->counts->get(k) >= impl_->store_threshold) {
        filtered.push_back(k);
      }
    }
    impl_->stores_skipped_in_current_batch +=
        static_cast<int64_t>(before - filtered.size());
    considered = std::move(filtered);
  }

  // Filter out blocks already stored (or being stored).
  std::vector<OffloadKey> keys_to_store;
  keys_to_store.reserve(considered.size());
  for (const OffloadKey& k : considered) {
    if (impl_->policy->get(k) == nullptr) {
      keys_to_store.push_back(k);
    }
  }

  if (keys_to_store.empty()) {
    return PrepareStoreOutput{{}, Impl::make_spec({}), {}};
  }

  const int64_t num_blocks_to_evict =
      static_cast<int64_t>(keys_to_store.size()) - impl_->get_num_free_blocks();

  std::vector<OffloadKey> to_evict;
  if (num_blocks_to_evict > 0) {
    if (num_blocks_to_evict > impl_->num_evictable_cache_blocks) {
      // Eviction cannot possibly be satisfied: SKIP the store. This is a
      // control path, not an error (see base.h).
      return std::nullopt;
    }
    // Blocks from the ORIGINAL input are excluded from eviction candidates: a
    // block that was already stored must still be in the cache afterwards.
    OffloadKeySet protected_keys(keys.begin(), keys.end());
    auto evicted = impl_->policy->evict(num_blocks_to_evict, protected_keys);
    if (!evicted.has_value()) {
      // Some idle blocks turned out to be protected: still a skip, not a
      // failure.
      return std::nullopt;
    }
    impl_->num_evictable_cache_blocks -= static_cast<int64_t>(evicted->size());
    if (impl_->num_evictable_cache_blocks < 0) {
      throw std::logic_error(
          "CPUOffloadingManager: evictable block count went negative");
    }
    for (const auto& [key, block] : *evicted) {
      impl_->free_block(block);
      to_evict.push_back(key);
    }
  }

  if (!to_evict.empty() && impl_->events_enabled) {
    impl_->events.push_back(
        OffloadingEvent{to_evict, CPULoadStoreSpec::kMedium, /*removed=*/true});
  }

  std::vector<BlockStatus> blocks = impl_->allocate_blocks(keys_to_store.size());
  if (blocks.size() != keys_to_store.size()) {
    throw std::logic_error(
        "CPUOffloadingManager: block pool did not allocate the expected number "
        "of blocks");
  }
  for (size_t i = 0; i < keys_to_store.size(); ++i) {
    // Inserted with ref_cnt == -1 (NOT READY): a concurrent lookup must see
    // kHitPending, never kHit.
    impl_->policy->insert(keys_to_store[i], blocks[i]);
  }

  return PrepareStoreOutput{keys_to_store, Impl::make_spec(blocks), to_evict};
}

void CPUOffloadingManager::complete_store(const std::vector<OffloadKey>& keys,
                                          const ReqContext& /*req_context*/,
                                          bool success) {
  std::vector<OffloadKey> stored_keys;
  if (success) {
    for (const OffloadKey& key : keys) {
      BlockStatus* block = impl_->policy->get(key);
      if (block != nullptr && !block->is_ready()) {
        block->ref_cnt = 0;  // -1 -> 0: now readable AND evictable.
        impl_->num_evictable_cache_blocks += 1;
        impl_->policy->mark_evictable(key);
        stored_keys.push_back(key);
      }
    }
  } else {
    for (const OffloadKey& key : keys) {
      BlockStatus* block = impl_->policy->get(key);
      if (block != nullptr && !block->is_ready()) {
        const BlockStatus copy = *block;
        impl_->policy->remove(key);
        impl_->free_block(copy);
      }
    }
  }

  if (!stored_keys.empty() && impl_->events_enabled) {
    impl_->events.push_back(OffloadingEvent{
        stored_keys, CPULoadStoreSpec::kMedium, /*removed=*/false});
  }
}

void CPUOffloadingManager::reset_cache() {
  // Clear ALL blocks unconditionally, matching upstream: the scheduler's stale
  // job threshold guarantees complete_load / complete_store are never called
  // for pre-reset jobs, so there is no lazy cleanup to do.
  impl_->policy->clear();
  impl_->num_evictable_cache_blocks = 0;
  impl_->free_list.clear();
  impl_->num_allocated_blocks = 0;
}

std::vector<OffloadingEvent> CPUOffloadingManager::take_events() {
  std::vector<OffloadingEvent> out;
  out.swap(impl_->events);
  return out;
}

CPUOffloadingStats CPUOffloadingManager::get_stats() {
  CPUOffloadingStats stats;
  const int64_t num_used = impl_->num_allocated_blocks -
                           static_cast<int64_t>(impl_->free_list.size()) -
                           impl_->num_evictable_cache_blocks;
  stats.cpu_cache_usage =
      impl_->num_blocks > 0 ? static_cast<double>(num_used) /
                                  static_cast<double>(impl_->num_blocks)
                            : 0.0;
  if (impl_->store_threshold >= 2) {
    stats.stores_skipped = impl_->stores_skipped_in_current_batch;
    impl_->stores_skipped_in_current_batch = 0;
  }
  return stats;
}

}  // namespace vllm::v1::kv_offload
