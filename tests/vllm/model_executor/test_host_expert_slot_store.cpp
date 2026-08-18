#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
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

TEST_CASE("EnsureFile preads the slice STRAIGHT into the slot") {
  // The whole point of this overload: the bytes come from the file descriptor,
  // never through a mapping, so no page of the source is faulted on the way.
  char path[] = "/tmp/vllm_iq1_pread_XXXXXX";
  const int fd = ::mkstemp(path);
  REQUIRE(fd >= 0);
  std::vector<uint8_t> file(256);
  for (size_t i = 0; i < file.size(); ++i) file[i] = static_cast<uint8_t>(i);
  REQUIRE(::write(fd, file.data(), file.size()) ==
          static_cast<ssize_t>(file.size()));

  ExpertSlotCache cache(2);
  HostExpertSlotStore store(2, 64);
  ExpertStreamer st(cache, store);

  // Read the third 64-byte slice, so a wrong offset is visible in the bytes
  // rather than only in a length.
  const ExpertStreamer::Result r = st.EnsureFile(ExpertKey{0, 3}, fd, 128, 64);
  REQUIRE(r.slot >= 0);
  CHECK(r.filled);
  CHECK_FALSE(r.hit);
  CHECK(st.bytes_filled() == 64);
  for (int i = 0; i < 64; ++i)
    REQUIRE(store.Slot(r.slot)[i] == static_cast<uint8_t>(128 + i));

  // A hit costs no syscall and moves no bytes.
  const ExpertStreamer::Result hit = st.EnsureFile(ExpertKey{0, 3}, fd, 128, 64);
  CHECK(hit.hit);
  CHECK(st.bytes_filled() == 64);
  CHECK(st.fills() == 1);

  // A read that runs past EOF is a SHORT read and must throw: a partially
  // filled slot decodes to garbage silently, which is the failure this whole
  // row is built to avoid.
  CHECK_THROWS_AS(st.EnsureFile(ExpertKey{0, 4}, fd, 224, 64),
                  std::runtime_error);
  CHECK_THROWS_AS(st.EnsureFile(ExpertKey{0, 5}, -1, 0, 64),
                  std::invalid_argument);
  // Size is still checked BEFORE the cache is touched.
  CHECK_THROWS_AS(st.EnsureFile(ExpertKey{0, 6}, fd, 0, 65),
                  std::invalid_argument);

  ::close(fd);
  ::unlink(path);
}

TEST_CASE("a fill that THROWS leaves nothing resident, and the retry refills") {
  // F2. Acquire has to run before the read, because the read needs somewhere to
  // land. So when the read throws, the cache already believes the key is
  // resident -- over a slot holding `done` correct bytes and the rest of
  // whatever the slot held before.
  //
  // Nothing downstream reads the exception. The next acquisition of that key is
  // an ordinary HIT, a hit moves no bytes by contract, and the GEMM multiplies
  // half of one expert spliced onto half of another. Plausible, silent, wrong.
  char path[] = "/tmp/vllm_expert_throw_XXXXXX";
  const int fd = ::mkstemp(path);
  REQUIRE(fd >= 0);
  std::vector<uint8_t> file(96);
  for (size_t i = 0; i < file.size(); ++i) file[i] = static_cast<uint8_t>(i + 1);
  REQUIRE(::write(fd, file.data(), file.size()) ==
          static_cast<ssize_t>(file.size()));

  ExpertSlotCache cache(2);
  HostExpertSlotStore store(2, 64);
  ExpertStreamer st(cache, store);

  const ExpertKey key{7, 11};
  // 64 bytes from offset 64, but the file holds only 96: the first 32 land and
  // then pread returns 0. Exactly the documented short-read throw.
  CHECK_THROWS_AS(st.EnsureFile(key, fd, 64, 64), std::runtime_error);

  // THE ASSERTION THAT WAS MISSING. The key must not be resident, or the retry
  // is a hit over a half-filled slot.
  CHECK_FALSE(cache.IsResident(key));
  CHECK_FALSE(cache.SlotOf(key).has_value());
  CHECK(cache.resident() == 0);
  // A failed fill moved no bytes, so the counters must not claim it did.
  CHECK(st.fills() == 0);
  CHECK(st.bytes_filled() == 0);

  // The slot came back to the budget: two other experts still both fit.
  REQUIRE(st.EnsureFile(ExpertKey{7, 1}, fd, 0, 64).filled);
  REQUIRE(st.EnsureFile(ExpertKey{7, 2}, fd, 32, 64).filled);

  // And the retry, once the read can succeed, is a MISS that really refills --
  // every byte, not the 32 that survived the failure.
  ExpertSlotCache c2(2);
  HostExpertSlotStore s2(2, 64);
  ExpertStreamer st2(c2, s2);
  CHECK_THROWS_AS(st2.EnsureFile(key, fd, 64, 64), std::runtime_error);
  REQUIRE_FALSE(c2.IsResident(key));
  const ExpertStreamer::Result retry = st2.EnsureFile(key, fd, 0, 64);
  REQUIRE(retry.slot >= 0);
  CHECK(retry.filled);
  CHECK_FALSE(retry.hit);
  for (int i = 0; i < 64; ++i)
    REQUIRE(s2.Slot(retry.slot)[i] == static_cast<uint8_t>(i + 1));

  ::close(fd);
  ::unlink(path);
}

TEST_CASE("an oversized slice is refused BEFORE it can evict a resident expert") {
  // F9. The ordering is stated in all three Ensure overloads and was pinned by
  // none of them: the existing cases used a 2-slot cache, where the refused
  // acquisition would have taken a FREE slot and evicted nothing, so the
  // assertion could not fail either way.
  //
  // A FULL cache is what makes the ordering observable.
  char path[] = "/tmp/vllm_expert_order_XXXXXX";
  const int fd = ::mkstemp(path);
  REQUIRE(fd >= 0);
  std::vector<uint8_t> file(256, 0x5A);
  REQUIRE(::write(fd, file.data(), file.size()) ==
          static_cast<ssize_t>(file.size()));

  ExpertSlotCache cache(2);
  HostExpertSlotStore store(2, 64);
  ExpertStreamer st(cache, store);
  REQUIRE(st.EnsureFile(ExpertKey{0, 1}, fd, 0, 64).filled);
  REQUIRE(st.EnsureFile(ExpertKey{0, 2}, fd, 64, 64).filled);
  cache.EndStep();  // both evictable now
  REQUIRE(cache.resident() == 2);
  const int64_t evictions_before = cache.evictions();

  // 65 bytes cannot be stored. If the size were checked AFTER Acquire, this
  // would evict one of the two residents on its way to being refused.
  CHECK_THROWS_AS(st.EnsureFile(ExpertKey{0, 3}, fd, 0, 65),
                  std::invalid_argument);
  CHECK(cache.evictions() == evictions_before);
  CHECK(cache.resident() == 2);
  CHECK(cache.IsResident(ExpertKey{0, 1}));
  CHECK(cache.IsResident(ExpertKey{0, 2}));
  CHECK_FALSE(cache.IsResident(ExpertKey{0, 3}));

  // Same ordering for the span overload, on the same full cache.
  std::vector<uint8_t> big(65, 0x11);
  CHECK_THROWS_AS(st.EnsureSpan(ExpertKey{0, 4}, big.data(), big.size()),
                  std::invalid_argument);
  CHECK(cache.evictions() == evictions_before);
  CHECK(cache.resident() == 2);

  ::close(fd);
  ::unlink(path);
}
