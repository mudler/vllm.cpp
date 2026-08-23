// ENG-EXPERT-STREAM-DEVICE W1 (issue #1124, gate G1): can a slot store whose
// slots are DEVICE allocations be filled at all, and does what lands in it match
// what lands in `HostExpertSlotStore` byte for byte?
//
// THE ORACLE IS THE HOST STORE, and that is a measured statement rather than a
// search that came up empty. Pinned vLLM `555967922` has no inference-time
// expert paging anywhere (`model_executor/offloader/uva.py` is a CPU-blanket UVA
// offloader over whole parameters, `.../prefetch.py` is cpu-only), and the
// secondary-oracle table does not rescue it — llama.cpp's `-ot`/`-ncmoe` moves
// expert COMPUTE to the host, which is a different design and not a slot store.
// So the reference is our own host arm on the same input, driven through the
// same `ExpertStreamer`, which is the shape `ENG-EXPERT-STREAM` already gates on.
//
// WHY IT CAN RUN WITHOUT A GPU. The store's device-ness is entirely
// `vt::Backend`: one `Alloc` for the arena, one `AllocPinned` for staging, and
// `Copy`/`Synchronize` for the transfer. On the CPU backend those are malloc and
// memcpy, so the bytes are checkable, and the two structural claims a CPU tier
// COULD get wrong instead of measuring — that `SlotForWrite` hands out staging
// rather than the slot, and that `SlotForRead` hands out the slot rather than
// staging — are asserted as pointer relationships rather than inferred from
// content that a unified allocator would make identical either way. Nothing here
// dereferences a device pointer from the host; every read of a slot is a
// backend copy, which is the only access a discrete part would allow.
//
// WHAT THIS FILE DOES NOT CLAIM. It does not claim anything selects this store.
// Nothing does: `Qwen35ExpertStream` still holds a concrete
// `HostExpertSlotStore` and reads it through `HostExpertSlotStore::Slot`, and
// making that read virtual and choosing the store from the platform is W2 of the
// same row (#1124). This suite gates a class, deliberately and with that said
// out loud, per `## Nothing lands dead`.
#include <fcntl.h>
#include <unistd.h>

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/device_expert_slot_store.h"
#include "vllm/model_executor/expert_slot_cache.h"
#include "vllm/model_executor/expert_streamer.h"
#include "vllm/model_executor/host_expert_slot_store.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

using vllm::DeviceExpertSlotStore;
using vllm::ExpertKey;
using vllm::ExpertSlotCache;
using vllm::ExpertStreamer;
using vllm::HostExpertSlotStore;

// A backend over ordinary host memory that COUNTS what the store asked it for.
// The same trick `test_expert_stream_device_slot` uses: malloc stands in for a
// device allocator, so an allocation is a real inspectable address, and the
// counters let a case say HOW MANY allocations and copies it observed rather
// than only that the bytes came out right.
//
// `DeviceMemoryIsHostAddressable()` stays FALSE — the default — because that is
// the honest answer for the device this store exists for, and it is the property
// that makes `pread`-into-the-slot illegal in the first place.
class CountingBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    ++allocs;
    last_alloc_bytes = bytes;
    // OUT OF MEMORY is what this reproduces, and it is this class's headline
    // failure rather than a hypothetical: #1123 is literally
    // `vt cuda: cudaMalloc: out of memory`. Every backend in the tree reports
    // it by THROWING -- `VT_CHECK` on the CPU backend, `Check(cudaMalloc)` on
    // CUDA -- and none returns nullptr, so a throw is the only shape a real
    // allocation failure takes here.
    if (throw_on_alloc) throw std::runtime_error("counting backend: alloc failed");
    last_alloc = std::malloc(bytes == 0 ? 1 : bytes);
    return last_alloc;
  }
  void Free(void* p) override {
    ++frees;
    last_freed = p;
    std::free(p);
  }
  void Memset(vt::Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    ++copies;
    copied_bytes += bytes;
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    ++queues;
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  }
  void DestroyQueue(vt::Queue&) override { ++queues_destroyed; }
  void Synchronize(vt::Queue&) override { ++syncs; }
  bool UnifiedMemory() const override { return false; }
  void* AllocPinned(size_t bytes) override {
    ++pinned_allocs;
    last_pinned_bytes = bytes;
    // The same failure one allocation later, and the expensive one: by here the
    // whole device arena is already held.
    if (throw_on_pinned_alloc)
      throw std::runtime_error("counting backend: pinned alloc failed");
    last_pinned = std::malloc(bytes == 0 ? 1 : bytes);
    return last_pinned;
  }
  void FreePinned(void* p) override {
    ++pinned_frees;
    std::free(p);
  }

  int allocs = 0;
  int pinned_allocs = 0;
  int frees = 0;
  int pinned_frees = 0;
  bool throw_on_alloc = false;
  bool throw_on_pinned_alloc = false;
  int copies = 0;
  int syncs = 0;
  int queues = 0;
  int queues_destroyed = 0;
  int64_t copied_bytes = 0;
  size_t last_alloc_bytes = 0;
  size_t last_pinned_bytes = 0;
  void* last_alloc = nullptr;
  void* last_pinned = nullptr;
  void* last_freed = nullptr;
};

// Read a device slot the only way a discrete part would allow: a backend copy
// back to host memory. Never a host dereference of the slot pointer.
std::vector<uint8_t> ReadBack(vt::Backend& b, uint8_t* device_slot,
                              size_t bytes) {
  std::vector<uint8_t> out(bytes, 0);
  vt::Queue q = b.CreateQueue();
  b.Copy(q, out.data(), device_slot, bytes);
  b.Synchronize(q);
  b.DestroyQueue(q);
  return out;
}

// A file of bytes no two slices of which are equal, so a wrong offset shows up
// in the CONTENT and not only in a length.
struct SliceFile {
  char path[64] = "/tmp/vllm_device_slot_XXXXXX";
  int fd = -1;
  std::vector<uint8_t> bytes;

  SliceFile(size_t slices, size_t slice_bytes) : bytes(slices * slice_bytes) {
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<uint8_t>((i * 31u + (i >> 5) * 7u + 1u) & 0xFFu);
    fd = ::mkstemp(path);
    REQUIRE(fd >= 0);
    REQUIRE(::write(fd, bytes.data(), bytes.size()) ==
            static_cast<ssize_t>(bytes.size()));
  }
  ~SliceFile() {
    if (fd >= 0) ::close(fd);
    ::unlink(path);
  }
  const uint8_t* slice(size_t i, size_t slice_bytes) const {
    return bytes.data() + i * slice_bytes;
  }
};

constexpr size_t kSliceBytes = 96;
constexpr size_t kSlices = 4;

}  // namespace

TEST_CASE("DeviceExpertSlotStore refuses a budget it cannot honour") {
  CountingBackend b;
  CHECK_THROWS_AS(DeviceExpertSlotStore(b, 0, 64), std::invalid_argument);
  CHECK_THROWS_AS(DeviceExpertSlotStore(b, -1, 64), std::invalid_argument);
  CHECK_THROWS_AS(DeviceExpertSlotStore(b, 4, 0), std::invalid_argument);
  // The host store's `std::vector` catches this for it; a raw `Alloc` has no
  // backstop, and a wrapped product would hand out in-range slot pointers past
  // the end of a far smaller arena.
  CHECK_THROWS_AS(DeviceExpertSlotStore(b, 4, SIZE_MAX / 2),
                  std::invalid_argument);
  // Nothing was allocated for any of the four refusals.
  CHECK(b.allocs == 0);
  CHECK(b.pinned_allocs == 0);
  // ...and no queue either: a constructor that throws runs no destructor, so
  // anything acquired above the refusal would leak.
  CHECK(b.queues == 0);
}

TEST_CASE("an ALLOCATION that throws gives back everything already acquired") {
  // The SECOND fresh review of PR #1735 (F2) -- not the first review's F2,
  // which was the file `CHECK`'s justification. The constructor's own comment says a
  // throwing constructor runs no destructor, and then guarded the failure that
  // cannot happen while leaking on the one that does. NO BACKEND IN THIS TREE
  // RETURNS NULLPTR: `CpuBackend::Alloc` refuses with `VT_CHECK`,
  // `CudaBackend::Alloc` and `AllocPinned` refuse through `Check(...)`, and the
  // base `Backend::AllocPinned` forwards to `Alloc`. They all THROW, and out of
  // memory is this class's headline failure -- #1123 is
  // `vt cuda: cudaMalloc: out of memory`. So the leak was on the live path and
  // the guard was on the dead one.
  //
  // What leaks is not small. A throw from `Alloc` strands the queue; a throw
  // from `AllocPinned` strands the queue AND the whole device arena, 18.55 GiB
  // on the target checkpoint, at the exact moment the device is out of memory.
  SUBCASE("the arena allocation throws") {
    CountingBackend b;
    b.throw_on_alloc = true;
    CHECK_THROWS_AS(DeviceExpertSlotStore(b, 8, 1024), std::runtime_error);
    // It was attempted, so this is the failure path and not an early refusal.
    CHECK(b.allocs == 1);
    // The queue was taken before it, and is given back.
    CHECK(b.queues == 1);
    CHECK(b.queues_destroyed == 1);
    // Nothing else was ever acquired, so nothing else is released.
    CHECK(b.pinned_allocs == 0);
    CHECK(b.frees == 0);
    CHECK(b.pinned_frees == 0);
  }
  SUBCASE("the pinned staging allocation throws") {
    CountingBackend b;
    b.throw_on_pinned_alloc = true;
    CHECK_THROWS_AS(DeviceExpertSlotStore(b, 8, 1024), std::runtime_error);
    CHECK(b.allocs == 1);
    CHECK(b.pinned_allocs == 1);
    // THE ARENA COMES BACK. This is the 18.55 GiB, and `last_freed` says it was
    // the arena rather than merely some pointer.
    CHECK(b.frees == 1);
    CHECK(b.last_freed == b.last_alloc);
    CHECK(b.queues == 1);
    CHECK(b.queues_destroyed == 1);
    // Nothing pinned was ever handed over, so nothing pinned is released.
    CHECK(b.pinned_frees == 0);
  }
}

TEST_CASE("the arena is ONE contiguous device allocation and staging is ONE pinned slot") {
  CountingBackend b;
  {
    DeviceExpertSlotStore s(b, 8, 1024);
    CHECK(s.slot_count() == 8);
    CHECK(s.slot_bytes() == 1024);
    CHECK(s.resident_bytes() == 8 * 1024);
    // One arena of the whole budget, decided up front and never grown: the
    // point of a slot array is that a model larger than memory cannot page
    // itself to death by admitting one more expert.
    CHECK(b.allocs == 1);
    CHECK(b.last_alloc_bytes == 8u * 1024u);
    // ONE staging slot, not eight. The filler is synchronous, so exactly one
    // fill is ever in flight; a buffer per slot would double the arena's host
    // cost to buffer a concurrency that does not exist.
    CHECK(b.pinned_allocs == 1);
    CHECK(b.last_pinned_bytes == 1024u);
    CHECK(b.queues == 1);

    // Slots are fixed offsets into that one block, in order.
    CHECK(s.SlotForRead(0) == static_cast<uint8_t*>(b.last_alloc));
    CHECK(s.SlotForRead(3) == static_cast<uint8_t*>(b.last_alloc) + 3 * 1024);
    CHECK(s.SlotForRead(7) == static_cast<uint8_t*>(b.last_alloc) + 7 * 1024);
    CHECK_THROWS_AS(s.SlotForRead(8), std::out_of_range);
    CHECK_THROWS_AS(s.SlotForRead(-1), std::out_of_range);
  }
  // The store owns both allocations and the queue, and gives all three back.
  CHECK(b.queues_destroyed == 1);
}

TEST_CASE("SlotForWrite hands out STAGING, never the device slot") {
  // This is the claim a unified allocator would hide if it were asserted
  // through content: on this backend a slot pointer IS host memory, so a
  // `SlotForWrite` that returned the slot would still produce the right bytes.
  // It is asserted as a pointer relationship for that reason.
  CountingBackend b;
  DeviceExpertSlotStore s(b, 4, 64);
  uint8_t* const arena = static_cast<uint8_t*>(b.last_alloc);

  uint8_t* w0 = s.SlotForWrite(0);
  CHECK(w0 == static_cast<uint8_t*>(b.last_pinned));
  CHECK((w0 < arena || w0 >= arena + 4 * 64));  // outside the arena entirely
  // One staging buffer means every slot gets the same address back.
  CHECK(s.SlotForWrite(1) == w0);
  CHECK(s.SlotForWrite(3) == w0);
  CHECK_THROWS_AS(s.SlotForWrite(4), std::out_of_range);
  CHECK_THROWS_AS(s.SlotForWrite(-1), std::out_of_range);
  // Handing out staging moves no bytes by itself.
  CHECK(b.copies == 0);
}

TEST_CASE("G1: a device store filled through EnsureFile is BYTE-IDENTICAL to the host store") {
  // The gate. Both arms run the same cache policy, the same streamer, the same
  // descriptor and the same offsets; the only difference is where the slot
  // lives. On a CPU `vt::Backend`, per the spec — this row has no discrete
  // NVIDIA GPU to reach, and that limitation is recorded as G-DISCRETE rather
  // than dressed up as this gate.
  //
  // "BYTE-IDENTICAL" IS OVER THE FILLED PREFIX, and the stores are asymmetric
  // beyond it: the host arena is a `std::vector<uint8_t>` and is zero-filled at
  // construction, while the device arena is a raw `vt::Backend::Alloc` and is
  // not initialised at all. Every fill here writes a whole slot, so the prefix
  // is the slot and the distinction does not reach this gate -- but a caller
  // that streamed a SHORT slice into a full-sized slot would find the two stores
  // disagreeing past the slice, and nothing promises otherwise. Zeroing the
  // device arena would cost a full write of the whole budget at load, 18.55 GiB
  // on the target checkpoint, to hide bytes no reader may look at. (Fresh review
  // of PR #1735, F3.)
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  SliceFile f(kSlices, kSliceBytes);

  ExpertSlotCache host_cache(static_cast<int32_t>(kSlices));
  HostExpertSlotStore host(static_cast<int32_t>(kSlices), kSliceBytes);
  ExpertStreamer host_st(host_cache, host);

  ExpertSlotCache dev_cache(static_cast<int32_t>(kSlices));
  DeviceExpertSlotStore dev(cpu, static_cast<int32_t>(kSlices), kSliceBytes);
  ExpertStreamer dev_st(dev_cache, dev);

  // THE DEVICE ARENA IS PUT INTO A KNOWN STATE FIRST, and a gate assertion
  // below depends on it rather than this being a tidiness habit. "Each slot
  // holds a DIFFERENT slice" compares two device slots that a
  // publish-suppressing mutation leaves UNWRITTEN, and `vt::Backend::Alloc`
  // does not initialise them (`std::aligned_alloc` on the CPU backend), so
  // whether that mutation red that assertion was decided by whatever the
  // allocator last left there -- the second fresh review of #1735 measured this
  // suite one assertion down from the recorded count for exactly that reason.
  // Writing every slot to the SAME known byte makes an unmutated fill the only
  // thing that can make two slots differ, so the assertion measures the store.
  // It says nothing about the CLASS, whose arena is uninitialised as its header
  // states; it is this test defining its own starting state.
  const std::vector<uint8_t> known(kSliceBytes, 0x5A);
  for (size_t i = 0; i < kSlices; ++i)
    dev.WriteSlot(static_cast<int32_t>(i), known.data(), known.size());

  int32_t host_slot[kSlices];
  int32_t dev_slot[kSlices];
  for (size_t i = 0; i < kSlices; ++i) {
    const ExpertKey key{3, static_cast<int32_t>(i)};
    const size_t off = i * kSliceBytes;
    const ExpertStreamer::Result h =
        host_st.EnsureFile(key, f.fd, off, kSliceBytes);
    const ExpertStreamer::Result d =
        dev_st.EnsureFile(key, f.fd, off, kSliceBytes);
    REQUIRE(h.filled);
    REQUIRE(d.filled);
    // Same cache policy, so the same key lands in the same slot index. If this
    // ever diverged the byte comparison below would be comparing the wrong
    // pair, so it is REQUIRED rather than checked.
    REQUIRE(h.slot == d.slot);
    host_slot[i] = h.slot;
    dev_slot[i] = d.slot;
  }

  for (size_t i = 0; i < kSlices; ++i) {
    const std::vector<uint8_t> got = ReadBack(cpu, dev.SlotForRead(dev_slot[i]),
                                              kSliceBytes);
    const uint8_t* want = host.Slot(host_slot[i]);
    // Byte-identical to the host store...
    CHECK(std::memcmp(got.data(), want, kSliceBytes) == 0);
    // ...and equal to the FILE, which is the one assertion in this case that can
    // see a defect in the SHARED HELPER. The comparison above is host-arm
    // against device-arm, and both arms run the same `ExpertStreamer` over the
    // same descriptor at the same `file_offset`: a streamer that read the wrong
    // offset, or read short, or read one slice twice makes both arms
    // identically wrong and passes it. The bytes on disk are the only input
    // neither arm computed, so comparing against them is what breaks that tie.
    // (An earlier draft justified this check as making a both-arms-empty red
    // DETERMINISTIC; the arena prefill above now does that job, and it was never
    // the stronger ground. Second fresh review of PR #1735, F5.)
    CHECK(std::memcmp(got.data(), f.slice(i, kSliceBytes), kSliceBytes) == 0);
    CHECK(got[0] == f.slice(i, kSliceBytes)[0]);
  }
  // Each slot holds a DIFFERENT slice, which is what a `SlotForRead` that
  // returned the staging buffer would break: staging holds only the last fill.
  CHECK(std::memcmp(ReadBack(cpu, dev.SlotForRead(dev_slot[0]), kSliceBytes).data(),
                    ReadBack(cpu, dev.SlotForRead(dev_slot[kSlices - 1]), kSliceBytes).data(),
                    kSliceBytes) != 0);

  CHECK(dev_st.fills() == host_st.fills());
  CHECK(dev_st.bytes_filled() == host_st.bytes_filled());
}

TEST_CASE("G1: the SPAN and TENSOR fills land in the device slot too") {
  // `EnsureFile` is the production filler, but `WriteSlot` is still reachable
  // through `EnsureSpan`, and a device store that only honoured one of the two
  // would be a trap for the next caller.
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  std::vector<uint8_t> src(kSliceBytes);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(0xC0u + i);

  ExpertSlotCache host_cache(2);
  HostExpertSlotStore host(2, kSliceBytes);
  ExpertStreamer host_st(host_cache, host);
  ExpertSlotCache dev_cache(2);
  DeviceExpertSlotStore dev(cpu, 2, kSliceBytes);
  ExpertStreamer dev_st(dev_cache, dev);

  const ExpertKey key{1, 9};
  const ExpertStreamer::Result h =
      host_st.EnsureSpan(key, src.data(), src.size());
  const ExpertStreamer::Result d =
      dev_st.EnsureSpan(key, src.data(), src.size());
  REQUIRE(h.filled);
  REQUIRE(d.filled);
  const std::vector<uint8_t> got =
      ReadBack(cpu, dev.SlotForRead(d.slot), kSliceBytes);
  CHECK(std::memcmp(got.data(), host.Slot(h.slot), kSliceBytes) == 0);
  CHECK(std::memcmp(got.data(), src.data(), kSliceBytes) == 0);

  // The same refusals the host store makes, so a caller cannot learn one
  // contract from one store and be surprised by the other.
  CHECK_THROWS_AS(dev.WriteSlot(2, src.data(), src.size()), std::out_of_range);
  CHECK_THROWS_AS(dev.WriteSlot(-1, src.data(), src.size()), std::out_of_range);
  std::vector<uint8_t> big(kSliceBytes + 1, 0xEE);
  CHECK_THROWS_AS(dev.WriteSlot(0, big.data(), big.size()),
                  std::invalid_argument);
}

TEST_CASE("the fill is a bounce: staging is written, then ONE copy publishes it") {
  CountingBackend b;
  SliceFile f(kSlices, kSliceBytes);
  ExpertSlotCache cache(static_cast<int32_t>(kSlices));
  DeviceExpertSlotStore s(b, static_cast<int32_t>(kSlices), kSliceBytes);
  ExpertStreamer st(cache, s);

  const ExpertKey key{5, 2};
  const ExpertStreamer::Result r = st.EnsureFile(key, f.fd, 0, kSliceBytes);
  REQUIRE(r.filled);
  // Exactly one H2D of exactly the slice, and a synchronize after it: `Copy` is
  // `cudaMemcpyAsync` on CUDA, and both the reuse of the single staging buffer
  // by the next fill and the GEMM that reads this slot as soon as the streamer
  // returns would otherwise race the transfer.
  CHECK(b.copies == 1);
  CHECK(b.copied_bytes == static_cast<int64_t>(kSliceBytes));
  CHECK(b.syncs == 1);

  // A hit publishes nothing: no copy, no synchronize, no bytes.
  const ExpertStreamer::Result hit = st.EnsureFile(key, f.fd, 0, kSliceBytes);
  REQUIRE(hit.hit);
  CHECK(b.copies == 1);
  CHECK(b.syncs == 1);
  CHECK(st.fills() == 1);
}

TEST_CASE("a fill that THROWS publishes nothing and leaves the key non-resident") {
  CountingBackend b;
  SliceFile f(1, kSliceBytes);  // one slice on disk, so a second read is short
  ExpertSlotCache cache(2);
  DeviceExpertSlotStore s(b, 2, kSliceBytes);
  ExpertStreamer st(cache, s);

  const ExpertKey key{4, 4};
  // 96 bytes from offset 64 of a 96-byte file: 32 land in staging and then
  // pread returns 0.
  CHECK_THROWS_AS(st.EnsureFile(key, f.fd, 64, kSliceBytes),
                  std::runtime_error);
  // Nothing reached the device, so the slot still holds what it held before —
  // and, crucially, the cache does not claim the key is resident over it.
  CHECK(b.copies == 0);
  CHECK_FALSE(cache.IsResident(key));
  CHECK(st.fills() == 0);
  CHECK(st.bytes_filled() == 0);
}

TEST_CASE("CommitSlot refuses to publish staging under the WRONG slot") {
  // With one staging buffer this is not bookkeeping. The bytes in staging
  // belong to whichever slot asked for the buffer last, so committing them
  // elsewhere files one expert's weights under another expert's key; the cache
  // then reports a HIT for a slot holding the wrong expert and the GEMM
  // multiplies it without a symptom.
  CountingBackend b;
  DeviceExpertSlotStore s(b, 4, 64);

  s.SlotForWrite(2);
  CHECK_THROWS_AS(s.CommitSlot(1, 64), std::logic_error);
  CHECK(b.copies == 0);
  // The right slot goes through.
  s.CommitSlot(2, 64);
  CHECK(b.copies == 1);
  // ...and once only: the staging buffer is spent, so a repeat is refused too.
  CHECK_THROWS_AS(s.CommitSlot(2, 64), std::logic_error);
  CHECK(b.copies == 1);
  // A commit with nothing staged at all is the same refusal.
  DeviceExpertSlotStore fresh(b, 4, 64);
  CHECK_THROWS_AS(fresh.CommitSlot(0, 64), std::logic_error);

  // Bounds and size are checked before the staging identity, as they are on
  // every other entry point.
  CHECK_THROWS_AS(s.CommitSlot(4, 64), std::out_of_range);
  CHECK_THROWS_AS(s.CommitSlot(-1, 64), std::out_of_range);
  s.SlotForWrite(0);
  CHECK_THROWS_AS(s.CommitSlot(0, 65), std::invalid_argument);
}

TEST_CASE("the HOST path is byte-identical across the contract change") {
  // W1's stop condition: the host path must stay exactly what it was. It writes
  // in place, so `SlotForWrite` still returns the slot itself, `CommitSlot` is a
  // no-op, and no staging buffer exists to be allocated or copied.
  HostExpertSlotStore h(3, 64);
  CHECK(h.SlotForWrite(1) == h.Slot(1));

  std::vector<uint8_t> src(64, 0x7E);
  h.WriteSlot(1, src.data(), src.size());
  std::vector<uint8_t> before(h.Slot(1), h.Slot(1) + 64);
  h.CommitSlot(1, 64);
  CHECK(std::memcmp(before.data(), h.Slot(1), 64) == 0);
  CHECK(h.Slot(1)[0] == 0x7E);
  CHECK(h.Slot(0)[0] == 0x00);  // untouched neighbour

  // It still refuses a slot it does not have, because an out-of-range commit
  // means the streamer and the store disagree about which slot was filled.
  CHECK_THROWS_AS(h.CommitSlot(3, 64), std::out_of_range);
  CHECK_THROWS_AS(h.CommitSlot(-1, 64), std::out_of_range);
  CHECK_THROWS_AS(h.CommitSlot(0, 65), std::invalid_argument);
}
