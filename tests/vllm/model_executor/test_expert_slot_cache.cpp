// ENG-EXPERT-STREAM W1 (#912) — the streamed-expert residency policy.
//
// Ported from antirez/ds4, the design the spec grounds this row in:
//   * hotness-decayed LFU with LRU tiebreak, ds4_metal.m:8678-8776, 9666-9830.
//   * the recorded rationale for the DECAY, ds4_metal.m:9679-9683: a plain
//     hit-count LFU "penalizes experts that are repeatedly selected but evicted
//     before a second hit".
//   * eviction protection for in-flight or selected entries, ds4_metal.m:834-905.
//   * the slot budget, ds4_ssd.c:46-78.
//
// ds4 has no unit test for this; its coverage is end-to-end generation. These
// cases are AUTHORED against that source.
//
// The properties worth testing are the ones a plausible reimplementation gets
// wrong: that a slot in use this step can never be evicted, that the budget
// refuses rather than corrupts, and that hotness DECAYS rather than accumulating
// forever.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/expert_slot_cache.h"

using vllm::ExpertAcquisition;
using vllm::ExpertKey;
using vllm::ExpertSlotCache;

namespace {
ExpertKey K(int32_t layer, int32_t expert) { return ExpertKey{layer, expert}; }
}  // namespace

TEST_CASE("a cold acquire misses, takes a free slot, and a repeat hits it") {
  ExpertSlotCache c(4);
  CHECK(c.capacity() == 4);
  CHECK(c.resident() == 0);

  ExpertAcquisition a = c.Acquire(K(0, 7));
  CHECK_FALSE(a.hit);
  CHECK(a.slot >= 0);
  CHECK_FALSE(a.evicted.has_value());
  CHECK(c.resident() == 1);
  CHECK(c.IsResident(K(0, 7)));

  ExpertAcquisition b = c.Acquire(K(0, 7));
  CHECK(b.hit);
  CHECK(b.slot == a.slot);
  CHECK(c.hits() == 1);
  CHECK(c.misses() == 1);
}

TEST_CASE("the layer disambiguates: same expert id in two layers is two entries") {
  // Expert 7 of layer 3 and expert 7 of layer 40 are different weights. A cache
  // keyed on the expert id alone would serve one layer's bytes to the other,
  // which is wrong output with no error anywhere.
  ExpertSlotCache c(4);
  ExpertAcquisition a = c.Acquire(K(3, 7));
  ExpertAcquisition b = c.Acquire(K(40, 7));
  CHECK_FALSE(b.hit);
  CHECK(a.slot != b.slot);
  CHECK(c.resident() == 2);
}

TEST_CASE("a slot selected THIS step is never evicted") {
  // ds4_metal.m:834-905. The step is about to read those bytes; reusing the slot
  // would overwrite them under the kernel.
  ExpertSlotCache c(2);
  const int32_t s0 = c.Acquire(K(0, 0)).slot;
  const int32_t s1 = c.Acquire(K(0, 1)).slot;
  CHECK(s0 != s1);

  // Both slots are protected, and a third expert cannot be served.
  ExpertAcquisition third = c.Acquire(K(0, 2));
  CHECK(third.slot == -1);
  CHECK(c.capacity_exhausted());
  // Neither resident expert was disturbed.
  CHECK(c.SlotOf(K(0, 0)) == s0);
  CHECK(c.SlotOf(K(0, 1)) == s1);

  // After the step ends, protection lifts and the third fits by eviction.
  c.EndStep();
  ExpertAcquisition again = c.Acquire(K(0, 2));
  CHECK(again.slot >= 0);
  CHECK_FALSE(c.capacity_exhausted());
  CHECK(again.evicted.has_value());
}

TEST_CASE("a budget smaller than one step's working set REFUSES, it does not corrupt") {
  // The failure mode this guards is a kernel reading another expert's bytes.
  // Returning a slot anyway would produce plausible wrong output; -1 makes the
  // caller refuse and tell the operator to raise the budget.
  ExpertSlotCache c(3);
  for (int e = 0; e < 3; ++e) CHECK(c.Acquire(K(0, e)).slot >= 0);
  for (int e = 3; e < 6; ++e) {
    ExpertAcquisition a = c.Acquire(K(0, e));
    CHECK(a.slot == -1);
    CHECK(c.capacity_exhausted());
  }
  // The three residents are intact.
  CHECK(c.resident() == 3);
  for (int e = 0; e < 3; ++e) CHECK(c.IsResident(K(0, e)));
}

TEST_CASE("eviction picks the COLDEST unprotected entry, not the oldest slot") {
  ExpertSlotCache c(3, /*decay=*/1.0);  // pure LFU: isolate frequency from time
  // Warm 0 heavily, 1 moderately, 2 once. Each in its own step so nothing stays
  // protected.
  for (int i = 0; i < 5; ++i) { c.Acquire(K(0, 0)); c.EndStep(); }
  for (int i = 0; i < 2; ++i) { c.Acquire(K(0, 1)); c.EndStep(); }
  c.Acquire(K(0, 2)); c.EndStep();
  CHECK(c.resident() == 3);

  // A fourth must evict expert 2, the least frequently used.
  ExpertAcquisition a = c.Acquire(K(0, 3));
  REQUIRE(a.evicted.has_value());
  CHECK(a.evicted->expert == 2);
  CHECK_FALSE(c.IsResident(K(0, 2)));
  CHECK(c.IsResident(K(0, 0)));
  CHECK(c.IsResident(K(0, 1)));
}

TEST_CASE("hotness DECAYS, so an old favourite loses to a current one") {
  // ds4_metal.m:9679-9683 is the reason this behaviour exists. Without decay,
  // expert 0's five ancient hits outrank expert 1's three recent ones forever
  // and the cache calcifies around a stale working set.
  ExpertSlotCache c(2, /*decay=*/0.5);  // aggressive, so the effect is visible
  for (int i = 0; i < 5; ++i) { c.Acquire(K(0, 0)); c.EndStep(); }

  // Many idle steps: expert 0's score decays toward zero.
  for (int i = 0; i < 20; ++i) c.EndStep();

  // Expert 1 becomes the current favourite with FEWER total hits.
  c.Acquire(K(0, 1)); c.EndStep();
  c.Acquire(K(0, 1)); c.EndStep();
  CHECK(c.resident() == 2);

  ExpertAcquisition a = c.Acquire(K(0, 2));
  REQUIRE(a.evicted.has_value());
  CHECK_MESSAGE(a.evicted->expert == 0,
                "the decayed old favourite must lose to the current one");

  // The control: with decay disabled the SAME sequence evicts the other way,
  // which is what proves the decay is doing the work and not the ordering.
  ExpertSlotCache nodecay(2, /*decay=*/1.0);
  for (int i = 0; i < 5; ++i) { nodecay.Acquire(K(0, 0)); nodecay.EndStep(); }
  for (int i = 0; i < 20; ++i) nodecay.EndStep();
  nodecay.Acquire(K(0, 1)); nodecay.EndStep();
  nodecay.Acquire(K(0, 1)); nodecay.EndStep();
  ExpertAcquisition b = nodecay.Acquire(K(0, 2));
  REQUIRE(b.evicted.has_value());
  CHECK_MESSAGE(b.evicted->expert == 1,
                "without decay the ancient favourite wins, which is the defect");
}

TEST_CASE("the old score is decayed BEFORE the hit is added") {
  // If a hit were added to the STORED score without first decaying it forward,
  // an entry would carry its whole history at full weight and a long-idle
  // expert would outrank a currently-hot one.
  //
  // Two earlier versions of this case passed under the defect because the
  // arithmetic happened to evict the same entry either way. The numbers below
  // are chosen so the two behaviours DISAGREE: with decay 0.5 a repeatedly-hit
  // entry converges to a stored score of ~2, while the defect accumulates ~30.
  ExpertSlotCache c(2, /*decay=*/0.5);
  for (int i = 0; i < 30; ++i) { c.Acquire(K(0, 0)); c.EndStep(); }
  // Expert 1 takes the other slot and is hit three times, ending hotter than a
  // correctly-decayed 0 but colder than the defect's carried 30.
  for (int i = 0; i < 3; ++i) { c.Acquire(K(0, 1)); c.EndStep(); }

  ExpertAcquisition a = c.Acquire(K(0, 2));
  REQUIRE(a.evicted.has_value());
  CHECK_MESSAGE(a.evicted->expert == 0,
                "correct: 0 decays to ~0.125 and loses to 1 at ~0.875. "
                "The defect carries 0 at ~1.875 and evicts 1 instead.");
}

TEST_CASE("equal hotness falls back to LRU, evicting the older touch") {
  // With decay disabled every single-hit entry scores exactly 1.0, so the
  // tiebreak alone decides. Without it the choice is whatever the scan order
  // happens to be, which is not a policy.
  ExpertSlotCache c(2, /*decay=*/1.0);
  c.Acquire(K(0, 0));  // touched at step 0
  c.EndStep();
  c.Acquire(K(0, 1));  // touched at step 1
  c.EndStep();

  ExpertAcquisition a = c.Acquire(K(0, 2));
  REQUIRE(a.evicted.has_value());
  CHECK_MESSAGE(a.evicted->expert == 0,
                "equal scores must evict the LEAST recently used");
  CHECK(c.IsResident(K(0, 1)));
}

TEST_CASE("counters and hit rate describe the run") {
  ExpertSlotCache c(2);
  CHECK(c.hit_rate() == 0.0);  // nothing asked yet
  c.Acquire(K(0, 0));          // miss
  c.Acquire(K(0, 0));          // hit
  c.Acquire(K(0, 0));          // hit
  CHECK(c.hits() == 2);
  CHECK(c.misses() == 1);
  CHECK(c.hit_rate() == doctest::Approx(2.0 / 3.0));
  CHECK(c.evictions() == 0);

  c.EndStep();
  c.Acquire(K(0, 1));
  c.EndStep();
  c.Acquire(K(0, 2));  // evicts one
  CHECK(c.evictions() == 1);
  CHECK(c.steps() == 2);

  // Clear drops residency and KEEPS the counters: they describe the run, not
  // the current contents.
  const int64_t h = c.hits();
  c.Clear();
  CHECK(c.resident() == 0);
  CHECK(c.hits() == h);
}

TEST_CASE("a zero budget reports exhaustion rather than a fillable miss") {
  // A miss tells the caller "fetch it into the slot I gave you". With no slots
  // there is nothing to fetch into, and saying miss would send the caller to
  // read 40 GiB into slot -1.
  ExpertSlotCache c(0);
  ExpertAcquisition a = c.Acquire(K(0, 0));
  CHECK(a.slot == -1);
  CHECK_FALSE(a.hit);
  CHECK(c.capacity_exhausted());
  CHECK(c.resident() == 0);
}

TEST_CASE("a realistic top-k step: repeated experts cost one slot, not k") {
  // The decode shape this row exists for: 8 of 256 experts per token, with the
  // same expert often selected by several tokens in a batch.
  ExpertSlotCache c(16);
  const std::vector<int32_t> selected = {3, 9, 3, 42, 9, 3, 7, 42};
  int32_t distinct_slots = 0;
  for (int32_t e : selected) {
    ExpertAcquisition a = c.Acquire(K(5, e));
    REQUIRE(a.slot >= 0);
    if (!a.hit) ++distinct_slots;
  }
  CHECK(distinct_slots == 4);   // 3, 9, 42, 7
  CHECK(c.resident() == 4);
  CHECK(c.hits() == 4);
  c.EndStep();
  // The same step again is now free.
  for (int32_t e : selected) CHECK(c.Acquire(K(5, e)).hit);
}

TEST_CASE("IsResident is a PURE probe: it must not change what gets evicted") {
  // The prefetch caller asks "will this be a fill?" before every slice. If the
  // asking scored the entry, the probe would decide the eviction order it was
  // only meant to observe, and the hotness policy would be measuring itself.
  vllm::ExpertSlotCache c(2);
  REQUIRE(c.Acquire(K(0, 1)).slot >= 0);  // A
  REQUIRE(c.Acquire(K(0, 2)).slot >= 0);  // B
  c.EndStep();

  const int64_t hits_before = c.hits();
  for (int i = 0; i < 50; ++i) CHECK(c.IsResident(K(0, 2)));
  CHECK_FALSE(c.IsResident(K(0, 99)));
  // A probe is not a hit: the counters a benchmark reads must not move.
  CHECK(c.hits() == hits_before);

  // Make A genuinely hotter by ACQUIRING it, then admit C. The victim must be
  // B, which only the probe ever touched. If IsResident had scored, those 50
  // probes would have made B the survivor and A the victim instead.
  REQUIRE(c.Acquire(K(0, 1)).hit);
  c.EndStep();
  const vllm::ExpertAcquisition ev = c.Acquire(K(0, 3));
  REQUIRE(ev.slot >= 0);
  REQUIRE(ev.evicted.has_value());
  CHECK(ev.evicted->expert == 2);
  CHECK(c.IsResident(K(0, 1)));
  CHECK_FALSE(c.IsResident(K(0, 2)));
}

TEST_CASE("N steps of K distinct slices never exhaust a K-slot cache") {
  // THE PROPERTY THE STEP CLOCK EXISTS FOR, and the one whose absence made this
  // row's decode measurement void.
  //
  // Acquire marks every entry it serves `protected_this_step`, and only EndStep
  // clears that mark. A caller that never ends a step therefore accumulates
  // protection forever: once the cache is full, ColdestEvictable finds nothing
  // evictable, Acquire returns slot -1, and every later slice is refused. The
  // cache stops serving and says so only through `capacity_exhausted()`, which
  // the production caller was not reading either.
  //
  // Stated as a property so it holds for any budget: a step whose working set
  // FITS the budget must be servable, no matter how many steps precede it.
  const int32_t slots = 8;
  const int steps = 10;
  ExpertSlotCache c(slots);

  int64_t served = 0, refused = 0;
  for (int s = 0; s < steps; ++s) {
    // Each step asks for `slots` DISTINCT experts, disjoint from every other
    // step's, so every step is a full turnover of the cache.
    for (int32_t i = 0; i < slots; ++i) {
      const ExpertAcquisition a = c.Acquire(K(0, s * slots + i));
      if (a.slot >= 0) {
        ++served;
        CHECK(a.slot < slots);
      } else {
        ++refused;
      }
    }
    c.EndStep();
  }

  CHECK(served == static_cast<int64_t>(steps) * slots);
  CHECK(refused == 0);
  CHECK_FALSE(c.capacity_exhausted());
  // The clock really advanced, which is what makes the decay a function of time.
  CHECK(c.steps() == steps);
  // Every step after the first had to evict the previous step's residents. If
  // this is zero the cache never recycled a slot and the count above is wrong
  // for a different reason.
  CHECK(c.evictions() == static_cast<int64_t>(steps - 1) * slots);

  // And the counter-case, to prove the assertion above is not vacuous: the same
  // traffic WITHOUT a step boundary dies as soon as the cache is full.
  ExpertSlotCache stuck(slots);
  int64_t stuck_served = 0, stuck_refused = 0;
  for (int s = 0; s < steps; ++s)
    for (int32_t i = 0; i < slots; ++i) {
      if (stuck.Acquire(K(0, s * slots + i)).slot >= 0)
        ++stuck_served;
      else
        ++stuck_refused;
    }
  CHECK(stuck_served == slots);                          // exactly one step's worth
  CHECK(stuck_refused == static_cast<int64_t>(steps - 1) * slots);
  CHECK(stuck.capacity_exhausted());
  CHECK(stuck.steps() == 0);
  CHECK(stuck.evictions() == 0);
}

TEST_CASE("Invalidate drops the entry and returns its slot to the budget") {
  // The undo a failed fill needs. Without it the key stays resident over a slot
  // holding a prefix of the right bytes, and the retry is a HIT that moves no
  // bytes at all.
  ExpertSlotCache c(2);
  const ExpertAcquisition a = c.Acquire(K(0, 1));
  REQUIRE(a.slot >= 0);
  REQUIRE(c.IsResident(K(0, 1)));

  CHECK(c.Invalidate(K(0, 1)));
  CHECK_FALSE(c.IsResident(K(0, 1)));
  CHECK_FALSE(c.SlotOf(K(0, 1)).has_value());
  CHECK(c.resident() == 0);
  // Invalidating something absent is not an error and changes nothing.
  CHECK_FALSE(c.Invalidate(K(0, 1)));
  CHECK_FALSE(c.Invalidate(K(9, 9)));

  // THE SLOT CAME BACK. A budget that shrank by one on every failed fill would
  // starve a long run, and would do it silently.
  const ExpertAcquisition b = c.Acquire(K(0, 2));
  const ExpertAcquisition d = c.Acquire(K(0, 3));
  REQUIRE(b.slot >= 0);
  REQUIRE(d.slot >= 0);
  CHECK(b.slot != d.slot);
  CHECK(c.resident() == 2);

  // The re-acquisition is a MISS, so the caller is told to fill it, which is the
  // whole point of undoing the acquisition.
  c.EndStep();
  CHECK(c.Invalidate(K(0, 2)));
  CHECK_FALSE(c.Acquire(K(0, 2)).hit);
}

TEST_CASE("Invalidate keeps the entry table dense for every other key") {
  // The compaction moves the LAST entry into the hole. If the index were not
  // repaired, the moved key would point at a stranger's slot -- the silently
  // wrong-expert failure this cache exists to prevent.
  ExpertSlotCache c(4);
  std::vector<int32_t> slots;
  for (int32_t e = 0; e < 4; ++e) {
    const ExpertAcquisition a = c.Acquire(K(1, e));
    REQUIRE(a.slot >= 0);
    slots.push_back(a.slot);
  }
  // Drop a MIDDLE entry, so the last one is moved into its place.
  REQUIRE(c.Invalidate(K(1, 1)));
  CHECK(c.resident() == 3);
  for (int32_t e : {0, 2, 3}) {
    CHECK(c.IsResident(K(1, e)));
    REQUIRE(c.SlotOf(K(1, e)).has_value());
    CHECK(*c.SlotOf(K(1, e)) == slots[static_cast<size_t>(e)]);
  }
  CHECK_FALSE(c.IsResident(K(1, 1)));
}
