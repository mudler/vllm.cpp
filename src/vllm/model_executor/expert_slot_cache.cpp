// ENG-EXPERT-STREAM W1. See expert_slot_cache.h for the ds4 anchors.
#include "vllm/model_executor/expert_slot_cache.h"

#include <cmath>
#include <limits>

namespace vllm {

ExpertSlotCache::ExpertSlotCache(int32_t capacity_slots, double decay)
    : capacity_(capacity_slots > 0 ? capacity_slots : 0),
      decay_(decay > 0.0 && decay <= 1.0 ? decay : 0.98) {
  entries_.reserve(static_cast<size_t>(capacity_));
  free_slots_.reserve(static_cast<size_t>(capacity_));
  // Slots are handed out low-first so a cache that never fills uses a compact
  // prefix, which keeps the device-side slot array's live range small.
  for (int32_t s = capacity_ - 1; s >= 0; --s) free_slots_.push_back(s);
}

double ExpertSlotCache::DecayedScore(const Entry& e) const {
  if (decay_ >= 1.0) return e.score;
  const int64_t age = step_ - e.score_tick;
  if (age <= 0) return e.score;
  return e.score * std::pow(decay_, static_cast<double>(age));
}

int32_t ExpertSlotCache::ColdestEvictable() const {
  int32_t best = -1;
  double best_score = std::numeric_limits<double>::infinity();
  int64_t best_used = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    // ds4_metal.m:834-905. An entry the current step selected is off limits:
    // evicting it would reuse a slot whose bytes this step is about to read.
    if (e.protected_this_step) continue;
    const double s = DecayedScore(e);
    // Hotness first, then LRU as the tiebreak.
    if (s < best_score || (s == best_score && e.last_used < best_used)) {
      best = static_cast<int32_t>(i);
      best_score = s;
      best_used = e.last_used;
    }
  }
  return best;
}

std::optional<int32_t> ExpertSlotCache::SlotOf(const ExpertKey& key) const {
  auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  return entries_[static_cast<size_t>(it->second)].slot;
}

ExpertAcquisition ExpertSlotCache::Acquire(const ExpertKey& key) {
  ExpertAcquisition out;
  capacity_exhausted_ = false;
  if (capacity_ == 0) {
    // A zero budget can never serve anything. Reported as exhausted rather than
    // as a miss, because a miss implies a slot the caller can fill.
    capacity_exhausted_ = true;
    ++misses_;
    return out;
  }

  auto it = index_.find(key);
  if (it != index_.end()) {
    Entry& e = entries_[static_cast<size_t>(it->second)];
    // Bring the score up to now BEFORE adding the hit, so a hit is worth the
    // same regardless of how long the entry idled. Adding first would let an
    // old score's decay swallow the new hit.
    e.score = DecayedScore(e) + 1.0;
    e.score_tick = step_;
    e.last_used = step_;
    e.protected_this_step = true;
    ++hits_;
    out.slot = e.slot;
    out.hit = true;
    return out;
  }

  ++misses_;

  int32_t slot = -1;
  if (!free_slots_.empty()) {
    slot = free_slots_.back();
    free_slots_.pop_back();
  } else {
    const int32_t victim = ColdestEvictable();
    if (victim < 0) {
      // Every slot is protected by this step, so the budget is smaller than one
      // step's working set. Refuse rather than evict something in use: the
      // caller must raise the budget, and a silently wrong slot here would be a
      // kernel reading another expert's bytes.
      capacity_exhausted_ = true;
      return out;
    }
    Entry& e = entries_[static_cast<size_t>(victim)];
    out.evicted = e.key;
    slot = e.slot;
    index_.erase(e.key);
    ++evictions_;
    // Compact: move the last entry into the hole so entries_ stays dense.
    const size_t last = entries_.size() - 1;
    if (static_cast<size_t>(victim) != last) {
      entries_[static_cast<size_t>(victim)] = entries_[last];
      index_[entries_[static_cast<size_t>(victim)].key] = victim;
    }
    entries_.pop_back();
  }

  Entry e;
  e.key = key;
  e.slot = slot;
  e.score = 1.0;
  e.score_tick = step_;
  e.last_used = step_;
  e.protected_this_step = true;
  entries_.push_back(e);
  index_[key] = static_cast<int32_t>(entries_.size() - 1);
  out.slot = slot;
  out.hit = false;
  return out;
}

void ExpertSlotCache::EndStep() {
  for (Entry& e : entries_) e.protected_this_step = false;
  ++step_;
}

double ExpertSlotCache::hit_rate() const {
  const int64_t total = hits_ + misses_;
  return total > 0 ? static_cast<double>(hits_) / static_cast<double>(total) : 0.0;
}

void ExpertSlotCache::Clear() {
  entries_.clear();
  index_.clear();
  free_slots_.clear();
  for (int32_t s = capacity_ - 1; s >= 0; --s) free_slots_.push_back(s);
  capacity_exhausted_ = false;
}

}  // namespace vllm
