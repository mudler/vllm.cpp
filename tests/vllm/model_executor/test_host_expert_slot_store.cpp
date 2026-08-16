// ENG-EXPERT-STREAM W4: the host slot store, and the raw-span Ensure the decode
// path uses. RED-first: both are new surfaces with no coverage.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/expert_slot_cache.h"
#include "vllm/model_executor/expert_streamer.h"
#include "vllm/model_executor/host_expert_slot_store.h"

using vllm::ExpertKey;
using vllm::ExpertSlotCache;
using vllm::ExpertStreamer;
using vllm::HostExpertSlotStore;

TEST_CASE("HostExpertSlotStore refuses a degenerate budget") {
  CHECK_THROWS_AS(HostExpertSlotStore(0, 64), std::invalid_argument);
  CHECK_THROWS_AS(HostExpertSlotStore(-1, 64), std::invalid_argument);
  CHECK_THROWS_AS(HostExpertSlotStore(4, 0), std::invalid_argument);
}

TEST_CASE("HostExpertSlotStore allocates the whole budget UP FRONT") {
  // The budget is the point: a store that grew on demand could not bound a
  // model larger than memory, which is the situation this row exists for.
  HostExpertSlotStore s(8, 1024);
  CHECK(s.slot_count() == 8);
  CHECK(s.slot_bytes() == 1024);
  CHECK(s.resident_bytes() == 8 * 1024);
}

TEST_CASE("HostExpertSlotStore writes land in the addressed slot only") {
  HostExpertSlotStore s(3, 16);
  std::vector<uint8_t> a(16, 0xAA), b(16, 0xBB);
  s.WriteSlot(0, a.data(), a.size());
  s.WriteSlot(2, b.data(), b.size());
  CHECK(s.Slot(0)[0] == 0xAA);
  CHECK(s.Slot(2)[0] == 0xBB);
  CHECK(s.Slot(1)[0] == 0x00);  // untouched neighbour
  // A slot that does not exist, and a write past the slot, are both refused
  // rather than clamped: a truncated expert decodes to garbage silently.
  CHECK_THROWS_AS(s.WriteSlot(3, a.data(), a.size()), std::out_of_range);
  CHECK_THROWS_AS(s.WriteSlot(-1, a.data(), a.size()), std::out_of_range);
  std::vector<uint8_t> big(17, 0xCC);
  CHECK_THROWS_AS(s.WriteSlot(0, big.data(), big.size()), std::invalid_argument);
  CHECK_THROWS_AS(s.Slot(3), std::out_of_range);
}

TEST_CASE("EnsureSpan fills on a miss and moves NO bytes on a hit") {
  ExpertSlotCache cache(4);
  HostExpertSlotStore store(4, 32);
  ExpertStreamer st(cache, store);

  std::vector<uint8_t> src(32);
  std::iota(src.begin(), src.end(), uint8_t{1});

  const ExpertStreamer::Result miss =
      st.EnsureSpan(ExpertKey{0, 7}, src.data(), src.size());
  CHECK(miss.slot >= 0);
  CHECK(miss.filled);
  CHECK_FALSE(miss.hit);
  CHECK(st.fills() == 1);
  CHECK(st.bytes_filled() == 32);
  CHECK(store.Slot(miss.slot)[0] == 1);
  CHECK(store.Slot(miss.slot)[31] == 32);

  const ExpertStreamer::Result hit =
      st.EnsureSpan(ExpertKey{0, 7}, src.data(), src.size());
  CHECK(hit.slot == miss.slot);
  CHECK(hit.hit);
  CHECK_FALSE(hit.filled);
  // The saving is that a hit costs NOTHING; a counter that moved here would
  // mean the cache re-read a resident expert.
  CHECK(st.fills() == 1);
  CHECK(st.bytes_filled() == 32);
}

TEST_CASE("EnsureSpan checks the size BEFORE it evicts") {
  // Ordering matters: acquiring first would evict a resident expert to make
  // room for one that cannot be stored, so a config error would also destroy a
  // good entry.
  ExpertSlotCache cache(1);
  HostExpertSlotStore store(1, 8);
  ExpertStreamer st(cache, store);
  std::vector<uint8_t> ok(8, 0x11), too_big(9, 0x22);

  const ExpertStreamer::Result a = st.EnsureSpan(ExpertKey{0, 1}, ok.data(), 8);
  REQUIRE(a.slot == 0);
  CHECK_THROWS_AS(st.EnsureSpan(ExpertKey{0, 2}, too_big.data(), 9),
                  std::invalid_argument);
  st.EndStep();
  // The resident expert survived the refused call.
  const ExpertStreamer::Result again = st.EnsureSpan(ExpertKey{0, 1}, ok.data(), 8);
  CHECK(again.hit);
  CHECK(st.fills() == 1);
}

TEST_CASE("EnsureSpan rejects a null span and reports exhaustion") {
  ExpertSlotCache cache(1);
  HostExpertSlotStore store(1, 8);
  ExpertStreamer st(cache, store);
  std::vector<uint8_t> v(8, 0x33);
  CHECK_THROWS_AS(st.EnsureSpan(ExpertKey{0, 1}, nullptr, 8),
                  std::invalid_argument);
  // One slot, two DISTINCT experts in the same step: the second cannot be
  // served, and says so with an invalid slot rather than corrupting the first.
  REQUIRE(st.EnsureSpan(ExpertKey{0, 1}, v.data(), 8).slot == 0);
  const ExpertStreamer::Result second =
      st.EnsureSpan(ExpertKey{0, 2}, v.data(), 8);
  CHECK(second.slot == -1);
  CHECK_FALSE(second.hit);
  CHECK_FALSE(second.filled);
}

int main(int argc, char** argv) {
  doctest::Context c;
  c.applyCommandLine(argc, argv);
  return c.run();
}
