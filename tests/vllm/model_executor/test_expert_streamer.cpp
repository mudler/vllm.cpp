// ENG-EXPERT-STREAM W3 (#912) — the miss path that fills a slot.
//
// The property this row exists for is that a HIT moves no bytes. A streamer
// that quietly refilled on every access would be correct, pass a token gate,
// and deliver none of the speedup, because the cost of this feature is entirely
// I/O. So the tests count bytes, not just correctness.
//
// The store is an interface precisely so these run without a GPU; the host
// implementation below records what was written, which is what makes "a hit
// wrote nothing" checkable at all.
#include <doctest/doctest.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/expert_streamer.h"

using vllm::ExpertKey;
using vllm::ExpertSlotCache;
using vllm::ExpertSlotStore;
using vllm::ExpertStreamer;
using vllm::GgufExpertLayoutOf;
using vllm::GgufTensorInfo;

namespace {

constexpr uint32_t kQ8_0 = 8;
constexpr int64_t kBlockElems = 32;
constexpr int64_t kBlockBytes = 34;

// A host store that records every write, so a test can assert that a hit did
// not produce one.
class RecordingStore final : public ExpertSlotStore {
 public:
  RecordingStore(int32_t slots, size_t bytes_per_slot)
      : slots_(slots), bytes_(bytes_per_slot), mem_(static_cast<size_t>(slots) * bytes_per_slot) {}

  size_t slot_bytes() const override { return bytes_; }
  int32_t slot_count() const override { return slots_; }
  void WriteSlot(int32_t slot, const uint8_t* src, size_t bytes) override {
    REQUIRE(slot >= 0);
    REQUIRE(slot < slots_);
    REQUIRE(bytes <= bytes_);
    std::memcpy(mem_.data() + static_cast<size_t>(slot) * bytes_, src, bytes);
    ++writes;
    written_bytes += static_cast<int64_t>(bytes);
    last_slot = slot;
  }

  // The pread filler writes in place rather than handing over a buffer, so the
  // double has to expose the same destination WriteSlot would have copied into.
  uint8_t* SlotForWrite(int32_t slot) override {
    REQUIRE(slot >= 0);
    REQUIRE(slot < slots_);
    ++writes;
    last_slot = slot;
    return mem_.data() + static_cast<size_t>(slot) * bytes_;
  }

  const uint8_t* slot(int32_t s) const { return mem_.data() + static_cast<size_t>(s) * bytes_; }

  int writes = 0;
  int64_t written_bytes = 0;
  int32_t last_slot = -1;

 private:
  int32_t slots_;
  size_t bytes_;
  std::vector<uint8_t> mem_;
};

// A synthetic stacked tensor backed by real bytes, so a fill can be checked
// against the source rather than merely counted.
struct FakeTensor {
  std::vector<uint8_t> bytes;
  GgufTensorInfo info;
};

FakeTensor MakeTensor(int64_t experts, int64_t rows, int64_t k) {
  FakeTensor f;
  const int64_t row_bytes = (k / kBlockElems) * kBlockBytes;
  f.bytes.resize(static_cast<size_t>(experts * rows * row_bytes));
  // Stamp each expert's region with its own index so a wrong span is visible in
  // the destination rather than merely mis-sized.
  const size_t per_expert = static_cast<size_t>(rows * row_bytes);
  for (int64_t e = 0; e < experts; ++e) {
    std::memset(f.bytes.data() + static_cast<size_t>(e) * per_expert,
                static_cast<int>(e & 0xff), per_expert);
  }
  f.info.name = "blk.0.ffn_gate_exps.weight";
  f.info.shape = {experts, rows, k};
  f.info.ggml_type = kQ8_0;
  f.info.data = f.bytes.data();
  f.info.nbytes = f.bytes.size();
  return f;
}

ExpertKey K(int32_t layer, int32_t expert) { return ExpertKey{layer, expert}; }

}  // namespace

TEST_CASE("a miss fills the slot with THAT expert's bytes") {
  FakeTensor t = MakeTensor(8, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 8);
  ExpertSlotCache cache(4);
  RecordingStore store(4, L.expert_bytes);
  ExpertStreamer s(cache, store);

  auto r = s.Ensure(K(0, 5), t.info, L);
  CHECK(r.slot >= 0);
  CHECK(r.filled);
  CHECK_FALSE(r.hit);
  CHECK(store.writes == 1);
  // The destination holds expert 5's stamp, not another expert's. This is what
  // distinguishes a correct span from a plausible one.
  CHECK(store.slot(r.slot)[0] == 5);
  CHECK(store.slot(r.slot)[L.expert_bytes - 1] == 5);
  CHECK(s.bytes_filled() == static_cast<int64_t>(L.expert_bytes));
  CHECK(s.fills() == 1);
}

TEST_CASE("a HIT moves no bytes at all") {
  // The saving this row exists to produce. A streamer that refilled on every
  // access would still be correct and would deliver nothing.
  FakeTensor t = MakeTensor(8, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 8);
  ExpertSlotCache cache(4);
  RecordingStore store(4, L.expert_bytes);
  ExpertStreamer s(cache, store);

  auto first = s.Ensure(K(0, 3), t.info, L);
  CHECK(first.filled);
  const int writes_after_fill = store.writes;
  const int64_t bytes_after_fill = s.bytes_filled();

  for (int i = 0; i < 5; ++i) {
    auto again = s.Ensure(K(0, 3), t.info, L);
    CHECK(again.hit);
    CHECK_FALSE(again.filled);
    CHECK(again.slot == first.slot);
  }
  CHECK_MESSAGE(store.writes == writes_after_fill, "a hit must not write");
  CHECK(s.bytes_filled() == bytes_after_fill);
  CHECK(s.fills() == 1);
}

TEST_CASE("an eviction refills the reused slot with the NEW expert") {
  FakeTensor t = MakeTensor(8, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 8);
  ExpertSlotCache cache(2);
  RecordingStore store(2, L.expert_bytes);
  ExpertStreamer s(cache, store);

  s.Ensure(K(0, 0), t.info, L); s.EndStep();
  s.Ensure(K(0, 1), t.info, L); s.EndStep();
  auto third = s.Ensure(K(0, 2), t.info, L);
  REQUIRE(third.slot >= 0);
  CHECK(third.filled);
  // The reused slot must hold expert 2, not the evicted expert's stale bytes.
  CHECK(store.slot(third.slot)[0] == 2);
  CHECK(s.fills() == 3);
}

TEST_CASE("an exhausted budget returns an invalid slot and writes nothing") {
  // Every slot is protected by the current step. Writing anyway would overwrite
  // bytes a kernel is about to read.
  FakeTensor t = MakeTensor(8, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 8);
  ExpertSlotCache cache(2);
  RecordingStore store(2, L.expert_bytes);
  ExpertStreamer s(cache, store);

  CHECK(s.Ensure(K(0, 0), t.info, L).slot >= 0);
  CHECK(s.Ensure(K(0, 1), t.info, L).slot >= 0);
  const int writes_before = store.writes;

  auto blocked = s.Ensure(K(0, 2), t.info, L);
  CHECK(blocked.slot == -1);
  CHECK_FALSE(blocked.hit);
  CHECK_FALSE(blocked.filled);
  CHECK_MESSAGE(store.writes == writes_before, "a refused acquire must not write");
}

TEST_CASE("an expert too large for a slot is refused BEFORE the cache is touched") {
  // Acquiring first would evict a resident expert to make room for one that
  // cannot be stored, so a configuration error would also destroy a good entry.
  FakeTensor t = MakeTensor(4, 4, 64);
  const auto L = GgufExpertLayoutOf(t.info, 4);
  ExpertSlotCache cache(2);
  RecordingStore small(2, L.expert_bytes - 1);  // one byte short
  ExpertStreamer s(cache, small);

  CHECK_THROWS_AS(s.Ensure(K(0, 0), t.info, L), std::invalid_argument);
  CHECK_MESSAGE(cache.resident() == 0, "nothing may be admitted for an expert that cannot be stored");
  CHECK(small.writes == 0);
}

TEST_CASE("a store whose slot count disagrees with the cache is refused at construction") {
  ExpertSlotCache cache(8);
  RecordingStore store(4, 1024);
  CHECK_THROWS_AS(ExpertStreamer(cache, store), std::invalid_argument);
  RecordingStore matching(8, 1024);
  CHECK_NOTHROW(ExpertStreamer(cache, matching));
}

TEST_CASE("a repeated top-k step costs one fill per DISTINCT expert") {
  // The decode shape: several tokens select overlapping experts, and only the
  // distinct ones cost I/O. This is the number the benchmark will quote.
  FakeTensor t = MakeTensor(64, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 64);
  ExpertSlotCache cache(32);
  RecordingStore store(32, L.expert_bytes);
  ExpertStreamer s(cache, store);

  const std::vector<int32_t> selected = {3, 9, 3, 42, 9, 3, 7, 42};
  for (int32_t e : selected) CHECK(s.Ensure(K(1, e), t.info, L).slot >= 0);
  CHECK(s.fills() == 4);  // 3, 9, 42, 7
  CHECK(s.bytes_filled() == 4 * static_cast<int64_t>(L.expert_bytes));

  // The same step again is free.
  s.EndStep();
  const int64_t before = s.bytes_filled();
  for (int32_t e : selected) CHECK(s.Ensure(K(1, e), t.info, L).hit);
  CHECK(s.bytes_filled() == before);
}

TEST_CASE("experts of different layers do not share a slot") {
  FakeTensor a = MakeTensor(4, 2, 64);
  FakeTensor b = MakeTensor(4, 2, 64);
  const auto L = GgufExpertLayoutOf(a.info, 4);
  ExpertSlotCache cache(4);
  RecordingStore store(4, L.expert_bytes);
  ExpertStreamer s(cache, store);

  auto r0 = s.Ensure(K(0, 1), a.info, L);
  auto r1 = s.Ensure(K(9, 1), b.info, L);
  CHECK(r0.slot != r1.slot);
  CHECK(r1.filled);
  CHECK(s.fills() == 2);
}
