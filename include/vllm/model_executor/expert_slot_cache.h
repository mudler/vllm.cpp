// ENG-EXPERT-STREAM W1 (issue #912, spec .agents/specs/expert-streaming.md).
//
// The CPU-side residency policy for streamed routed experts. It answers ONE
// question: which fixed device slot holds a given logical expert, and which
// entry to evict when none does. It allocates nothing, touches no device, and
// reads no file.
//
// Ported from antirez/ds4 (DwarfStar), the design the spec grounds this row in:
//   * hotness-decayed LFU with an LRU tiebreak (`ds4_metal.m:8678-8776,
//     9666-9830`). The DECAY is the load-bearing part. A plain hit-count LFU
//     "penalizes experts that are repeatedly selected but evicted before a
//     second hit" (`ds4_metal.m:9679-9683`), so a expert that was hot long ago
//     outranks one that is hot now and the cache calcifies.
//   * eviction protection for entries referenced by in-flight work or selected
//     for the current step (`ds4_metal.m:834-905`).
//   * a fixed slot budget, `--ssd-streaming-cache-experts N|NGB`
//     (`ds4_ssd.c:46-78`).
//
// WHY A SLOT CACHE AND NOT A MAP. The decode path runs the unchanged
// dense-stride Marlin/grouped GEMM over a fixed contiguous array of expert
// slots, so the cache's job is to keep a STABLE small integer per resident
// expert. Handing out pointers instead would force the kernel to gather.
#ifndef VLLM_MODEL_EXECUTOR_EXPERT_SLOT_CACHE_H_
#define VLLM_MODEL_EXECUTOR_EXPERT_SLOT_CACHE_H_

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vllm {

// A logical routed expert. `layer` disambiguates: expert 7 of layer 3 and
// expert 7 of layer 40 are different weights and must not share a slot.
struct ExpertKey {
  int32_t layer = 0;
  int32_t expert = 0;
  bool operator==(const ExpertKey& o) const {
    return layer == o.layer && expert == o.expert;
  }
};

struct ExpertKeyHash {
  size_t operator()(const ExpertKey& k) const {
    return (static_cast<uint64_t>(static_cast<uint32_t>(k.layer)) << 32) ^
           static_cast<uint32_t>(k.expert);
  }
};

// What Acquire did. `hit` is the difference between a free ride and an NVMe
// read, so the caller uses it to decide whether to issue I/O, and the counters
// use it to report a hit rate that a benchmark can quote.
struct ExpertAcquisition {
  int32_t slot = -1;
  bool hit = false;
  // Set when the acquisition evicted a resident expert, so the caller knows the
  // slot's previous contents are gone and must be refilled.
  std::optional<ExpertKey> evicted;
};

class ExpertSlotCache {
 public:
  // `capacity_slots` is the number of experts that stay resident. `decay` is
  // the per-tick multiplier applied to a hotness score, in (0, 1]; 1.0 is a
  // plain LFU and is allowed so the decay can be measured against its absence.
  ExpertSlotCache(int32_t capacity_slots, double decay = 0.98);

  // Resolve `key` to a slot for the CURRENT step.
  //
  // On a hit the entry is touched and its slot returned. On a miss the coldest
  // UNPROTECTED entry is evicted and its slot reused; a free slot is preferred
  // over any eviction.
  //
  // Every acquisition protects its entry until EndStep(), because evicting an
  // expert this step is about to read would hand the kernel a slot whose bytes
  // are being overwritten. When every slot is protected and the key is absent,
  // this returns slot -1 rather than corrupting one: see `capacity_exhausted()`.
  ExpertAcquisition Acquire(const ExpertKey& key);

  // Ends the step and clears per-step protection. Advances the hotness clock,
  // which is what makes the decay a function of TIME rather than of call count.
  void EndStep();

  // Drop `key`'s entry and return its slot to the free list, if it is resident.
  // Returns true when something was dropped.
  //
  // This exists for exactly one caller: a fill that FAILED partway. Acquire has
  // to run before the read, because the read needs somewhere to land, so a read
  // that throws leaves the key marked resident over a slot holding a prefix of
  // the right bytes and a tail of whatever the slot held before. The next
  // acquisition of that key is then a HIT, and the kernel silently multiplies
  // half an expert. Undoing the acquisition is what makes a failed fill safe,
  // and it is the difference between a loud failure and a quiet one.
  //
  // Counters are NOT rewound. hits/misses/evictions describe what the run did,
  // and the miss really did happen.
  bool Invalidate(const ExpertKey& key);

  // True when the last Acquire could not be served because every slot was
  // protected by the current step. That means the budget is smaller than one
  // step's working set, which is a configuration error the caller must refuse
  // rather than work around.
  bool capacity_exhausted() const { return capacity_exhausted_; }

  // Is `key` resident RIGHT NOW, without touching its hotness or LRU order?
  //
  // A pure query, deliberately separate from Acquire: a caller that wants to
  // know whether a fill is coming (to prefetch, or to count) must not perturb
  // the eviction order by asking. Acquire is the only thing that scores.
  //
  // There is deliberately only ONE of these. An earlier revision carried a
  // byte-identical `Contains` beside it, which is how the doc comment above
  // `capacity_exhausted()` came to sit in front of the wrong declaration.
  bool IsResident(const ExpertKey& key) const {
    return index_.find(key) != index_.end();
  }
  std::optional<int32_t> SlotOf(const ExpertKey& key) const;

  int32_t capacity() const { return capacity_; }
  int32_t resident() const { return static_cast<int32_t>(index_.size()); }
  int64_t hits() const { return hits_; }
  int64_t misses() const { return misses_; }
  int64_t evictions() const { return evictions_; }
  int64_t steps() const { return step_; }

  // hits / (hits + misses), or 0 when nothing has been asked. The number a
  // benchmark quotes, so it lives here rather than being recomputed by callers
  // that might disagree about the denominator.
  double hit_rate() const;

  // Drops every entry. The counters survive, because they describe the run.
  void Clear();

 private:
  struct Entry {
    ExpertKey key;
    int32_t slot = -1;
    double score = 0.0;      // hotness at `score_tick`
    int64_t score_tick = 0;  // when `score` was last brought up to date
    int64_t last_used = 0;   // LRU tiebreak
    bool protected_this_step = false;
  };

  // `score` decayed forward to the current step.
  double DecayedScore(const Entry& e) const;
  // Index into entries_ of the coldest evictable entry, or -1 when none is.
  int32_t ColdestEvictable() const;

  int32_t capacity_ = 0;
  double decay_ = 0.98;
  int64_t step_ = 0;
  int64_t hits_ = 0;
  int64_t misses_ = 0;
  int64_t evictions_ = 0;
  bool capacity_exhausted_ = false;

  std::vector<Entry> entries_;  // dense, one per occupied slot
  std::vector<int32_t> free_slots_;
  std::unordered_map<ExpertKey, int32_t, ExpertKeyHash> index_;  // key -> entries_ idx
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_EXPERT_SLOT_CACHE_H_
