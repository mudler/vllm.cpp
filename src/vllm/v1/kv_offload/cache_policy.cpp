// Ported from: vllm/v1/kv_offload/cpu/policies/{base,lru,arc}.py @ e24d1b24
// See include/vllm/v1/kv_offload/cache_policy.h for the two load-bearing
// contracts (the ref_cnt == -1 sentinel and the ATOMIC evict).
//
// PYTHON CONTAINER MAPPING. Both policies are written against
// collections.OrderedDict, which is an insertion-ordered map with an O(1)
// move_to_end and an O(1) popitem(last=False). `OrderedKeyMap` below is that
// container: a std::list for the order plus an unordered_map from key to the
// list node and the value. Using a plain std::map or a sorted structure would
// silently change eviction ORDER, which is the whole behaviour under test.
#include "vllm/v1/kv_offload/cache_policy.h"

#include <algorithm>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace vllm::v1::kv_offload {
namespace {

// An insertion-ordered key->value map with O(1) move-to-end, mirroring
// collections.OrderedDict. Iteration runs least-recently-inserted/used first,
// which is the order both policies scan for eviction candidates.
template <typename V>
class OrderedKeyMap {
 public:
  using Order = std::list<OffloadKey>;

  bool contains(const OffloadKey& key) const {
    return entries_.find(key) != entries_.end();
  }

  V* get(const OffloadKey& key) {
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : &it->second.value;
  }

  // Insert or overwrite. A NEW key goes to the END (most recent); an existing
  // key keeps its position, exactly like OrderedDict's `d[k] = v`.
  void put(const OffloadKey& key, V value) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      it->second.value = std::move(value);
      return;
    }
    order_.push_back(key);
    Entry entry{std::prev(order_.end()), std::move(value)};
    entries_.emplace(key, std::move(entry));
  }

  // Erase if present; returns whether anything was erased (OrderedDict.pop
  // with a default).
  bool erase(const OffloadKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return false;
    }
    order_.erase(it->second.order_it);
    entries_.erase(it);
    return true;
  }

  // OrderedDict.move_to_end(key): make `key` the most recent. No-op if absent.
  void move_to_end(const OffloadKey& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return;
    }
    order_.splice(order_.end(), order_, it->second.order_it);
  }

  // OrderedDict.popitem(last=False): drop the least-recent entry.
  void pop_front() {
    if (order_.empty()) {
      return;
    }
    entries_.erase(order_.front());
    order_.pop_front();
  }

  void clear() {
    order_.clear();
    entries_.clear();
  }

  size_t size() const { return order_.size(); }
  // Iteration order: least recent first.
  const Order& keys_in_order() const { return order_; }

 private:
  struct Entry {
    typename Order::iterator order_it;
    V value;
  };
  Order order_;
  std::unordered_map<OffloadKey, Entry> entries_;
};

// Presence-only ordered set (upstream's `OrderedDict[key, None]`).
struct Unit {};
using OrderedKeySet = OrderedKeyMap<Unit>;

// --- LRU (cpu/policies/lru.py:12-88) ----------------------------------------
//
// The eviction order comes from `evictable_blocks`, a SEPARATE ordered set
// holding only ref_cnt == 0 blocks. Pinned blocks are absent from it and
// re-enter at the END when their last load completes (mark_evictable), so a
// block that was just read is the LAST candidate — not the first.
class LRUCachePolicy final : public CachePolicy {
 public:
  explicit LRUCachePolicy(int64_t /*cache_capacity*/) {}

  BlockStatus* get(const OffloadKey& key) override { return blocks_.get(key); }

  void insert(const OffloadKey& key, BlockStatus block) override {
    const bool evictable = block.ref_cnt == 0;
    blocks_.put(key, block);
    if (evictable) {
      evictable_.put(key, Unit{});
    }
  }

  void remove(const OffloadKey& key) override {
    blocks_.erase(key);
    evictable_.erase(key);
  }

  void touch(const std::vector<OffloadKey>& keys) override {
    // REVERSE order, mirroring `for key in reversed(list(keys))`. Active
    // (pinned) blocks are deliberately untouched: they are non-evictable now
    // and will reach the end of `evictable_` when their loads finish.
    for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
      if (evictable_.contains(*it)) {
        evictable_.move_to_end(*it);
      }
    }
  }

  std::optional<std::vector<std::pair<OffloadKey, BlockStatus>>> evict(
      int64_t n, const OffloadKeySet& protected_keys) override {
    std::vector<std::pair<OffloadKey, BlockStatus>> candidates;
    if (n == 0) {
      return candidates;
    }
    for (const OffloadKey& key : evictable_.keys_in_order()) {
      if (protected_keys.count(key) != 0) {
        continue;
      }
      BlockStatus* block = blocks_.get(key);
      // Anything in evictable_ must be idle; a violation is a port bug, not a
      // runtime condition.
      if (block == nullptr || block->ref_cnt != 0) {
        throw std::logic_error(
            "LRUCachePolicy: evictable list holds a non-idle block");
      }
      candidates.emplace_back(key, *block);
      if (static_cast<int64_t>(candidates.size()) == n) {
        break;
      }
    }
    // ATOMIC: fewer than n candidates => change nothing.
    if (static_cast<int64_t>(candidates.size()) < n) {
      return std::nullopt;
    }
    for (const auto& [key, _] : candidates) {
      evictable_.erase(key);
      blocks_.erase(key);
    }
    return candidates;
  }

  void clear() override {
    evictable_.clear();
    blocks_.clear();
  }

  void mark_evictable(const OffloadKey& key) override {
    // ref_cnt reached 0 (store completed -1 -> 0, or last load 1 -> 0). In
    // both cases the key is NOT already in the evictable list.
    evictable_.put(key, Unit{});
  }

  void mark_non_evictable(const OffloadKey& key) override {
    // The key must have been evictable; upstream does an unguarded `del`.
    evictable_.erase(key);
  }

 private:
  OrderedKeySet evictable_;
  OrderedKeyMap<BlockStatus> blocks_;
};

// --- ARC (cpu/policies/arc.py:12-171) ---------------------------------------
//
// T1 = seen once (recency), T2 = seen again (frequency), B1/B2 = ghost lists of
// keys evicted from T1/T2. `target_t1_size` adapts: a hit in B1 grows T1, a hit
// in B2 shrinks it. Ghost lists are trimmed to cache_capacity only after a
// SUCCESSFUL eviction.
class ARCCachePolicy final : public CachePolicy {
 public:
  explicit ARCCachePolicy(int64_t cache_capacity)
      : cache_capacity_(cache_capacity) {}

  BlockStatus* get(const OffloadKey& key) override {
    if (BlockStatus* b = t1_.get(key)) {
      return b;
    }
    return t2_.get(key);
  }

  void insert(const OffloadKey& key, BlockStatus block) override {
    // New blocks always land in T1 and leave the ghost lists.
    t1_.put(key, block);
    b1_.erase(key);
    b2_.erase(key);
  }

  void remove(const OffloadKey& key) override {
    if (!t1_.erase(key)) {
      t2_.erase(key);
    }
  }

  void touch(const std::vector<OffloadKey>& keys) override {
    for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
      const OffloadKey& key = *it;
      if (BlockStatus* block = t1_.get(key)) {
        const BlockStatus copy = *block;
        t1_.erase(key);
        if (!copy.is_ready()) {
          // Just PREPARED for storing, not genuinely touched twice: keep it in
          // T1 and only refresh its recency. Promoting a not-yet-stored block
          // to T2 would treat a single access as two.
          t1_.put(key, copy);
        } else {
          t2_.put(key, copy);
        }
      } else if (t2_.contains(key)) {
        t2_.move_to_end(key);
      } else if (b1_.contains(key)) {
        const double delta =
            std::max(1.0, static_cast<double>(b2_.size()) /
                              static_cast<double>(b1_.size()));
        target_t1_size_ = std::min(target_t1_size_ + delta,
                                   static_cast<double>(cache_capacity_));
        b1_.move_to_end(key);
      } else if (b2_.contains(key)) {
        const double delta =
            std::max(1.0, static_cast<double>(b1_.size()) /
                              static_cast<double>(b2_.size()));
        target_t1_size_ = std::max(target_t1_size_ - delta, 0.0);
        b2_.move_to_end(key);
      }
    }
  }

  std::optional<std::vector<std::pair<OffloadKey, BlockStatus>>> evict(
      int64_t n, const OffloadKeySet& protected_keys) override {
    std::vector<std::pair<OffloadKey, BlockStatus>> result;
    if (n == 0) {
      return result;
    }
    // Select all n candidates against a SIMULATED T1 size before mutating
    // anything (the atomicity contract).
    std::vector<CandidateInternal> candidates;
    OffloadKeySet already_selected;
    int64_t virtual_t1_size = static_cast<int64_t>(t1_.size());

    for (int64_t i = 0; i < n; ++i) {
      std::optional<CandidateInternal> candidate;
      if (virtual_t1_size >= static_cast<int64_t>(target_t1_size_)) {
        candidate = pick(t1_, protected_keys, already_selected, /*t1=*/true);
        if (candidate.has_value()) {
          virtual_t1_size -= 1;
        }
      }
      if (!candidate.has_value()) {
        candidate = pick(t2_, protected_keys, already_selected, /*t1=*/false);
        if (!candidate.has_value()) {
          return std::nullopt;
        }
      }
      already_selected.insert(candidate->key);
      candidates.push_back(*candidate);
    }

    for (const CandidateInternal& c : candidates) {
      if (c.from_t1) {
        t1_.erase(c.key);
        b1_.put(c.key, Unit{});
      } else {
        t2_.erase(c.key);
        b2_.put(c.key, Unit{});
      }
      result.emplace_back(c.key, c.block);
    }
    // Trim the ghost lists, only after a successful eviction.
    for (OrderedKeySet* ghost : {&b1_, &b2_}) {
      while (static_cast<int64_t>(ghost->size()) > cache_capacity_) {
        ghost->pop_front();
      }
    }
    return result;
  }

  void clear() override {
    t1_.clear();
    t2_.clear();
    b1_.clear();
    b2_.clear();
    target_t1_size_ = 0.0;
  }

 private:
  struct CandidateInternal {
    OffloadKey key;
    BlockStatus block;
    bool from_t1;
  };

  // First idle, unprotected, not-already-selected key in `from`, in order.
  std::optional<CandidateInternal> pick(OrderedKeyMap<BlockStatus>& from,
                                        const OffloadKeySet& protected_keys,
                                        const OffloadKeySet& already_selected,
                                        bool from_t1) {
    for (const OffloadKey& key : from.keys_in_order()) {
      BlockStatus* block = from.get(key);
      if (block->ref_cnt != 0) {
        continue;
      }
      if (protected_keys.count(key) != 0 || already_selected.count(key) != 0) {
        continue;
      }
      return CandidateInternal{key, *block, from_t1};
    }
    return std::nullopt;
  }

  int64_t cache_capacity_;
  double target_t1_size_ = 0.0;
  OrderedKeyMap<BlockStatus> t1_;
  OrderedKeyMap<BlockStatus> t2_;
  OrderedKeySet b1_;
  OrderedKeySet b2_;
};

}  // namespace

std::unique_ptr<CachePolicy> make_cache_policy(const std::string& name,
                                               int64_t cache_capacity) {
  if (name == "lru") {
    return std::make_unique<LRUCachePolicy>(cache_capacity);
  }
  if (name == "arc") {
    return std::make_unique<ARCCachePolicy>(cache_capacity);
  }
  throw std::invalid_argument("Unknown cache policy: '" + name +
                              "'. Supported: lru, arc");
}

}  // namespace vllm::v1::kv_offload
