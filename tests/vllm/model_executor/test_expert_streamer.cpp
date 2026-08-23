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
#include <fcntl.h>
#include <unistd.h>

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
    // A WRITE can fail for the same class of reason a publish can, and on the
    // store W1 added it for it fails for exactly that reason:
    // `DeviceExpertSlotStore::WriteSlot` calls `vt::Backend::Copy` and
    // `Synchronize`, and a real CUDA backend throws out of both. Before W1 no
    // store in the tree could do that, because every `WriteSlot` was a memcpy.
    // The throw is raised BEFORE the copy so the slot keeps the bytes of the
    // expert that used to live there, which is the state the streamer's undo
    // has to survive.
    if (throw_on_write) throw std::runtime_error("recording store: write failed");
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

  // ENG-EXPERT-STREAM-DEVICE W1 (#1124): the double writes in place like the
  // host store, so there is nothing to publish — but it COUNTS the calls, so a
  // case can assert that the streamer commits exactly the fills it performed
  // and never a hit.
  void CommitSlot(int32_t slot, size_t bytes) override {
    REQUIRE(slot >= 0);
    REQUIRE(slot < slots_);
    REQUIRE(bytes <= bytes_);
    ++commits;
    last_commit_slot = slot;
    // A PUBLISH can fail, and on the store this method was added for it fails
    // for the same class of reason a read does: `DeviceExpertSlotStore` calls
    // `vt::Backend::Copy` and `Synchronize`, and a real CUDA backend throws out
    // of both. The double can be asked to do that, because the placement of the
    // call relative to the streamer's `try` is only observable when it throws.
    if (throw_on_commit) throw std::runtime_error("recording store: commit failed");
  }

  const uint8_t* slot(int32_t s) const { return mem_.data() + static_cast<size_t>(s) * bytes_; }

  int writes = 0;
  int64_t written_bytes = 0;
  int32_t last_slot = -1;
  int commits = 0;
  int32_t last_commit_slot = -1;
  bool throw_on_commit = false;
  bool throw_on_write = false;

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


TEST_CASE("EnsureFile PUBLISHES exactly the fills, and never a hit or a failure") {
  // ENG-EXPERT-STREAM-DEVICE W1 (#1124). `CommitSlot` is what lets a store whose
  // slots the host cannot write be filled at all, and it is only correct if the
  // streamer calls it in exactly the places a fill succeeded. A commit on a HIT
  // would republish a slot whose staging buffer now holds a different expert; a
  // commit after a failed read would publish a partial slice under a key the
  // cache is about to invalidate.
  char path[] = "/tmp/vllm_expert_commit_XXXXXX";
  const int fd = ::mkstemp(path);
  REQUIRE(fd >= 0);
  std::vector<uint8_t> file(96);
  for (size_t i = 0; i < file.size(); ++i) file[i] = static_cast<uint8_t>(i + 3);
  REQUIRE(::write(fd, file.data(), file.size()) ==
          static_cast<ssize_t>(file.size()));

  ExpertSlotCache cache(2);
  RecordingStore store(2, 32);
  ExpertStreamer s(cache, store);

  const ExpertKey key{2, 5};
  const ExpertStreamer::Result first = s.EnsureFile(key, fd, 0, 32);
  REQUIRE(first.filled);
  CHECK(store.commits == 1);
  // The slot published is the slot filled, not merely some slot.
  CHECK(store.last_commit_slot == first.slot);

  // A hit moves no bytes, so there is nothing to publish.
  REQUIRE(s.EnsureFile(key, fd, 0, 32).hit);
  CHECK(store.commits == 1);

  // A short read throws; nothing is published and the key is not resident.
  CHECK_THROWS_AS(s.EnsureFile(ExpertKey{2, 6}, fd, 80, 32), std::runtime_error);
  CHECK(store.commits == 1);
  CHECK_FALSE(cache.IsResident(ExpertKey{2, 6}));

  // A slice too large is refused before the cache is touched, so no slot was
  // ever handed out to publish.
  CHECK_THROWS_AS(s.EnsureFile(ExpertKey{2, 7}, fd, 0, 33), std::invalid_argument);
  CHECK(store.commits == 1);

  // An exhausted budget returns an invalid slot with neither flag set.
  REQUIRE(s.EnsureFile(ExpertKey{2, 8}, fd, 32, 32).filled);
  CHECK(store.commits == 2);
  const ExpertStreamer::Result none = s.EnsureFile(ExpertKey{2, 9}, fd, 0, 32);
  CHECK(none.slot == -1);
  CHECK_FALSE(none.filled);
  CHECK(store.commits == 2);

  ::close(fd);
  ::unlink(path);
}


TEST_CASE("a PUBLISH that throws leaves nothing resident either") {
  // ENG-EXPERT-STREAM-DEVICE W1 (#1124), the fresh review of PR #1735 (F1).
  // `store_.CommitSlot(...)` sits INSIDE `EnsureFile`'s try, and until this case
  // existed nothing measured that: moving the call to just after the `catch`
  // left every suite green, because no store in the tree could fail a publish.
  //
  // The failure it guards is the SAME one the read arm is wrapped for, one step
  // later. Acquire must run before the fill, so by the time a publish throws the
  // cache already says the key is resident -- over a slot still holding the
  // expert that used to live there, because the bytes never left staging. If the
  // throw escapes without undoing the acquisition, the next request for that key
  // is an ordinary HIT, no read is issued because a hit moves no bytes, and the
  // GEMM multiplies the wrong expert. Silent, plausible and wrong.
  //
  // This arm becomes REACHABLE in W2, when a store whose `CommitSlot` really
  // copies to a device is selected; it is gated now because that is when the
  // placement decision was made.
  char path[] = "/tmp/vllm_expert_publish_XXXXXX";
  const int fd = ::mkstemp(path);
  REQUIRE(fd >= 0);
  std::vector<uint8_t> file(96);
  for (size_t i = 0; i < file.size(); ++i) file[i] = static_cast<uint8_t>(i + 5);
  REQUIRE(::write(fd, file.data(), file.size()) ==
          static_cast<ssize_t>(file.size()));

  ExpertSlotCache cache(2);
  RecordingStore store(2, 32);
  ExpertStreamer s(cache, store);

  const ExpertKey key{8, 1};
  store.throw_on_commit = true;
  CHECK_THROWS_AS(s.EnsureFile(key, fd, 0, 32), std::runtime_error);

  // THE ASSERTIONS THAT THE PLACEMENT BUYS. The read succeeded, so only the
  // publish can have undone the acquisition.
  CHECK(store.commits == 1);
  CHECK_FALSE(cache.IsResident(key));
  CHECK_FALSE(cache.SlotOf(key).has_value());
  CHECK(cache.resident() == 0);
  // A fill that did not publish moved no bytes, so the counters must not claim
  // it did -- the same rule the read arm follows.
  CHECK(s.fills() == 0);
  CHECK(s.bytes_filled() == 0);

  // The slot came back to the budget, and the retry is a MISS that really
  // refills rather than a hit over a stale slot.
  store.throw_on_commit = false;
  const ExpertStreamer::Result retry = s.EnsureFile(key, fd, 0, 32);
  REQUIRE(retry.slot >= 0);
  CHECK(retry.filled);
  CHECK_FALSE(retry.hit);
  CHECK(s.fills() == 1);
  for (int i = 0; i < 32; ++i)
    REQUIRE(store.slot(retry.slot)[i] == static_cast<uint8_t>(i + 5));

  ::close(fd);
  ::unlink(path);
}


TEST_CASE("a WRITE that throws leaves nothing resident either -- EnsureSpan") {
  // ENG-EXPERT-STREAM-DEVICE W1 (#1124), the SECOND fresh review of PR #1735
  // (F1) -- not the first review's F1, which was the publish arm one call later.
  // `EnsureFile`'s publish is wrapped and `EnsureSpan`'s write was not, and the
  // two are the same window one call earlier. W1 is what opens it: before this
  // wave every store's `WriteSlot` was a memcpy and could not throw, and
  // `DeviceExpertSlotStore::WriteSlot` calls `vt::Backend::Copy` and
  // `Synchronize`, which throw `std::runtime_error` out of the CUDA backend.
  //
  // `EnsureSpan` is a PRODUCTION call site -- `qwen3_5.cpp`'s
  // `Qwen35ExpertStream::Slice` reaches it from `Qwen3_5Model::Forward` -- so
  // the failure is the deployed one: Acquire has already evicted the previous
  // expert and claimed the key, the write throws, and the entry stands over a
  // slot still holding the EVICTED expert's bytes. The next request for that key
  // is an ordinary HIT, no bytes move because a hit moves none, and the GEMM
  // multiplies the wrong expert. Silent, plausible and wrong.
  ExpertSlotCache cache(1);
  RecordingStore store(1, 32);
  ExpertStreamer s(cache, store);

  // A resident expert first, so the failed write has something to have evicted:
  // the slot's stale content is the corruption, not merely an empty slot.
  std::vector<uint8_t> a(32, 0xA1);
  const ExpertKey key_a{7, 1};
  REQUIRE(s.EnsureSpan(key_a, a.data(), a.size()).filled);
  REQUIRE(store.slot(0)[0] == 0xA1);
  s.EndStep();

  std::vector<uint8_t> b(32, 0xB2);
  const ExpertKey key_b{7, 2};
  store.throw_on_write = true;
  CHECK_THROWS_AS(s.EnsureSpan(key_b, b.data(), b.size()), std::runtime_error);

  // THE ASSERTIONS THE UNDO BUYS. The acquisition happened -- the cache had to
  // hand out a destination before the write could be attempted -- so only the
  // catch can have taken it back.
  CHECK_FALSE(cache.IsResident(key_b));
  CHECK_FALSE(cache.SlotOf(key_b).has_value());
  CHECK(cache.resident() == 0);
  // The write moved no bytes, so the counters must not claim it did.
  CHECK(s.fills() == 1);
  CHECK(s.bytes_filled() == 32);
  CHECK(store.writes == 1);
  // ...and the slot really does hold the evicted expert, which is the state a
  // surviving cache entry would have made a HIT over.
  CHECK(store.slot(0)[0] == 0xA1);
  CHECK(store.slot(0)[31] == 0xA1);

  // The retry is a real MISS that refills, not a hit over the stale slot.
  store.throw_on_write = false;
  const ExpertStreamer::Result retry = s.EnsureSpan(key_b, b.data(), b.size());
  REQUIRE(retry.slot >= 0);
  CHECK(retry.filled);
  CHECK_FALSE(retry.hit);
  CHECK(s.fills() == 2);
  CHECK(s.bytes_filled() == 64);
  for (int i = 0; i < 32; ++i) REQUIRE(store.slot(retry.slot)[i] == 0xB2);
}


TEST_CASE("a WRITE that throws leaves nothing resident either -- Ensure") {
  // The tensor overload has the identical window, for the identical reason, and
  // it is gated separately because the two entry points wrap their own writes:
  // a `try` added to one of them leaves the other exactly as exposed as before.
  FakeTensor t = MakeTensor(8, 2, 64);
  const auto L = GgufExpertLayoutOf(t.info, 8);
  ExpertSlotCache cache(1);
  RecordingStore store(1, L.expert_bytes);
  ExpertStreamer s(cache, store);

  REQUIRE(s.Ensure(K(0, 4), t.info, L).filled);
  REQUIRE(store.slot(0)[0] == 4);  // MakeTensor stamps each expert with its index
  s.EndStep();

  const ExpertKey key{0, 6};
  store.throw_on_write = true;
  CHECK_THROWS_AS(s.Ensure(key, t.info, L), std::runtime_error);

  CHECK_FALSE(cache.IsResident(key));
  CHECK_FALSE(cache.SlotOf(key).has_value());
  CHECK(cache.resident() == 0);
  CHECK(s.fills() == 1);
  CHECK(s.bytes_filled() == static_cast<int64_t>(L.expert_bytes));
  CHECK(store.writes == 1);
  CHECK(store.slot(0)[0] == 4);  // still expert 4's bytes, never expert 6's

  store.throw_on_write = false;
  const ExpertStreamer::Result retry = s.Ensure(key, t.info, L);
  REQUIRE(retry.slot >= 0);
  CHECK(retry.filled);
  CHECK_FALSE(retry.hit);
  CHECK(s.fills() == 2);
  CHECK(store.slot(retry.slot)[0] == 6);
  CHECK(store.slot(retry.slot)[L.expert_bytes - 1] == 6);
}
