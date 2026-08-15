// vllm.cpp original (the shared scratch pool is a vt-runtime deviation, porting
// inventory §9.1); vLLM has no mirror because it never had this design — its
// allocation handle carries the device as field 0
// (`vllm/device_allocator/__init__.py:12-14` @ pin 555967922) and torch's cache
// is per-device by construction (`c10/cuda/CUDACachingAllocator.h:118-172`).
//
// THE GATE FOR #516 (.agents/specs/pool-device-key.md): `vllm::Pool()` was a
// process-wide free list keyed by BYTE SIZE CLASS ONLY. The device was not in
// the key, so a block allocated through one backend was handed to a `DBuf`
// running on another. One fault, two symptoms, chosen by direction:
//
//   cudaMalloc block -> CPU-backend forward : SIGSEGV, compute-sanitizer CLEAN
//                                             (the fault is host-side)
//   host block       -> CUDA forward        : SILENT all-NaN output
//
// This file is the DIRECT gate. It needs no GPU, no checkpoint and no NAS: two
// distinguishable fake backends stand in for two devices (the technique
// tests/vt/test_backend_multidevice.cpp and tests/vt/test_reference_tier.cpp
// already use), and every case goes through `dense_attn::DBuf` — the seam every
// production forward allocates from — so it holds the same path the LTX-2.5
// device suite crashes on rather than a paraphrase of it.
//
// Two test-side properties are load-bearing and neither is a style choice.
// (1) A fake backend NEVER returns a block to the C allocator: these cases
// compare pointer identity ACROSS a free, and a freed pointer is not a value you
// may reason about. (2) A fake backend is never destroyed before exit, so no
// stack or heap address is ever reused — the pool is keyed on the backend's
// identity, and an address reused by a later case would make one case's pool
// answer another case's question.
//
// THE TWO DEBUG LANES ARE GREEN HERE, ON PURPOSE. `VT_POOL_BYPASS=1` removes the
// free list and `VT_POOL_EXACT=1` removes size-class rounding; the spec's own
// §10 hands `VT_POOL_BYPASS=1` to whoever picks up the unattributed dgx failures
// as "the cheap discriminator that needs no second build". A suite that reds
// under the very lane it tells you to use is a suite that costs its next reader
// an hour deciding whether the red is theirs. So: a case whose subject is REUSE
// states what the ACTIVE lane does — a pooled hit by default, a fresh driver
// block under bypass — and the case whose subject is the SIZE CLASS states
// sharing by default and SEPARATION under `VT_POOL_EXACT` (spec §5 T1.3's
// second clause, which had no assertion until now). Both env vars are read ONCE
// into a function-local static inside `DevicePool`, so a process is in exactly
// one lane for its whole life and no case can toggle them.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// A host-memory backend that remembers WHICH allocations are its own, so a case
// can ask the question that matters — "did this block come from THIS device?" —
// instead of inferring it from a pointer that merely happens to differ.
class TagBackend final : public Backend {
 public:
  ~TagBackend() override {
    for (void* p : owned_) std::free(p);
  }
  void* Alloc(size_t bytes) override {
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    owned_.push_back(p);
    ++allocs_;
    return p;
  }
  // Deliberately does NOT std::free: see the file header. The block stays valid
  // and stays owned; only the fact of the Free is recorded.
  void Free(void* p) override {
    freed_.push_back(p);
    ++frees_;
  }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{}; }
  bool UnifiedMemory() const override { return true; }

  bool Owns(const void* p) const {
    for (const void* q : owned_)
      if (q == p) return true;
    return false;
  }
  bool WasFreed(const void* p) const {
    for (const void* q : freed_)
      if (q == p) return true;
    return false;
  }
  int allocs() const { return allocs_; }
  int frees() const { return frees_; }

 private:
  std::vector<void*> owned_;
  std::vector<void*> freed_;
  int allocs_ = 0;
  int frees_ = 0;
};

// Process-lifetime fakes: see the file header, property (2).
TagBackend& NewBackend() {
  static std::vector<std::unique_ptr<TagBackend>> keep;
  keep.push_back(std::make_unique<TagBackend>());
  return *keep.back();
}

// Two devices of the same TYPE. `Device{type,index}` is exactly how the backend
// registry addresses discrete devices (vt/backend.h, kMaxDevicesPerType), and
// keeping the type equal keeps the platform lookup the DBuf constructor performs
// (`ResolveDevicePoolPolicy`) on a registered platform, so these cases test the
// pool and nothing else.
Queue QueueOn(int32_t index) {
  Queue q;
  q.device = Device{DeviceType::kCPU, index};
  q.handle = nullptr;
  return q;
}

// The two debug lanes, spelled EXACTLY as `DevicePool::Bypass()` and
// `DevicePool::ClassOf()` spell them ("=1", first character only), so this file
// cannot disagree with the header about which lane a process is in.
bool PoolBypass() {
  const char* e = std::getenv("VT_POOL_BYPASS");
  return e != nullptr && e[0] == '1';
}
bool PoolExact() {
  const char* e = std::getenv("VT_POOL_EXACT");
  return e != nullptr && e[0] == '1';
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// THE DEFECT. A block freed on device 0 must never be handed to device 1.
//
// Every case below uses its OWN size class, so no case can be decided by what an
// earlier one left in a free list — including under `--order-by=rand`.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: a block freed on one device is NEVER handed to another") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);

  // Identical shape and dtype on both devices, so both land in the SAME size
  // class. That is the whole precondition: at differing size classes the two
  // arms never trade blocks, which is why the f32 LTX-2.5 arms could not reach
  // this and the bf16 ones could.
  const std::vector<int64_t> shape{1024};  // 4096 bytes
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    on_a = x.ptr();
  }  // returned to device 0's free list here

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    on_b = y.ptr();
  }

  REQUIRE(on_a != nullptr);
  REQUIRE(on_b != nullptr);
  // The pointer identity IS the defect: before the device entered the key these
  // were the same block, and device 1 then wrote through device 0's allocation.
  CHECK(on_b != on_a);
  // ...and the stronger statement that identity check stands in for: each block
  // came out of its OWN device's allocator.
  CHECK(a.Owns(on_a));
  CHECK(b.Owns(on_b));
  CHECK_FALSE(b.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// The same ordering the other way round. The direction decides the SYMPTOM (a
// host block reaching a device forward is the silent-NaN direction; a device
// block reaching a host forward is the SIGSEGV one), so a fix that separates
// them in only one direction is not a fix.
TEST_CASE("device pool: the reverse direction is separated too") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);
  const std::vector<int64_t> shape{4096};  // 8192 bytes @ bf16

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kBF16, shape);
    on_b = y.ptr();
  }
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kBF16, shape);
    on_a = x.ptr();
  }
  CHECK(on_a != on_b);
  CHECK(b.Owns(on_b));
  CHECK(a.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// ═══════════════════════════════════════════════════════════════════════════
// ...AND THE POOL MUST STILL BE A POOL. Without the two cases below, "fixed"
// would be indistinguishable from `VT_POOL_BYPASS=1`, which also passes every
// case above and reinstates the per-op cudaMalloc/cudaFree sync storm the pool
// exists to remove. The fix has to separate the devices WITHOUT ending reuse.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: reuse on ONE device still returns the identical block") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  const std::vector<int64_t> shape{4096};  // 16384 bytes @ f32

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, shape);
    second = y.ptr();
  }
  if (PoolBypass()) {
    // The bypass lane HAS no free list — every Get is an exact-size driver
    // Alloc and every Put a real Free, which is the whole reason it exists (it
    // restores the allocation boundaries compute-sanitizer needs). Reuse is
    // therefore the wrong expectation here, not a regression, and asserting the
    // lane's own behavior is what keeps the lane usable as a discriminator.
    CHECK(second != first);
    CHECK(a.allocs() == allocs_after_first + 1);
    CHECK(a.WasFreed(first));
  } else {
    CHECK(second == first);                   // a pool HIT...
    CHECK(a.allocs() == allocs_after_first);  // ...proven by the allocator counter
  }
}

TEST_CASE("device pool: size-class rounding still lets nearby sizes share a block") {
  // 32,400 and 32,768 bytes round to the same class (kClassBits=4 keeps the top
  // four significant bits), which is what makes a prefill of a different token
  // count a pool hit instead of a synchronous cudaMalloc. Preserved
  // deliberately: `VT_POOL_EXACT=1` (reuse kept, rounding removed) was MEASURED
  // still red, so the rounding is not the fault and is not what this row
  // changes.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, {8100});  // 32,400 bytes
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, {8192});  // 32,768 bytes, same class
    second = y.ptr();
  }
  const size_t lo_class = vllm::DevicePool::SizeClassForTest(32400);
  const size_t hi_class = vllm::DevicePool::SizeClassForTest(32768);
  if (PoolBypass()) {
    CHECK(second != first);  // no free list at all, so nothing to share
  } else if (PoolExact()) {
    // Spec §5 T1.3's second clause, "`VT_POOL_EXACT` still separates them",
    // which had no assertion anywhere until this one. Exact keying is the A/B
    // arm that MEASURED the rounding innocent (still red), so it has to keep
    // being a real second behavior and not merely a variable nobody reads.
    CHECK(lo_class == 32400);
    CHECK(hi_class == 32768);
    CHECK(lo_class != hi_class);
    CHECK(second != first);
    CHECK(a.allocs() == allocs_after_first + 1);
  } else {
    CHECK(lo_class == hi_class);  // ...one class...
    CHECK(lo_class == 32768);     // ...and it is the larger of the two
    CHECK(second == first);
    CHECK(a.allocs() == allocs_after_first);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// The three properties that follow from "a pool is bound to one device", each
// of which was silently wrong before and none of which the DBuf cases above
// would catch on their own.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("device pool: Drain frees ONE device's blocks, through ITS OWN backend") {
  // Before the device entered the key there was one free list, so the MiniMax-H3
  // phase-change drain (minimax_h3_pipeline.cpp) handed every retained block to
  // whichever backend it was called with — including another device's blocks, to
  // an allocator that never made them.
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);
  const std::vector<int64_t> shape{16384};  // 65,536 bytes @ f32

  void* held_by_b = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
  }
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    held_by_b = y.ptr();
  }
  if (PoolBypass()) {
    // Bypass frees straight through, so both destructors already returned their
    // block to their OWN backend and there is nothing retained to drain. That is
    // the same device-scoped property this case is about, enforced by the
    // absence of a free list rather than by the key — worth stating, because it
    // is the arm the spec sends a reader to when a drain is under suspicion.
    CHECK(a.frees() == 1);
    CHECK(b.frees() == 1);
    CHECK(vllm::Pool(a).Drain(a) == 0);
    return;
  }
  REQUIRE(a.frees() == 0);
  REQUIRE(b.frees() == 0);

  const size_t drained = vllm::Pool(a).Drain(a);
  CHECK(drained > 0);
  CHECK(a.frees() == 1);
  CHECK(b.frees() == 0);  // device 1's retained block was NOT freed by device 0

  // ...and device 1's free list is intact: its next request of that class is
  // still the same block, i.e. the drain did not quietly empty it.
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    CHECK(y.ptr() == held_by_b);
  }
}

TEST_CASE("device pool: a pool bound to another device is REFUSED, not served") {
  // `ActivePoolScope` is the one remaining way to hand a DBuf a pool that is not
  // its device's. It is a legitimate seam — the aux CUDA stream uses it — so it
  // stays, and the pool checks the backend instead. A hard throw, not an
  // `assert`: the gate builds are Release/NDEBUG, where an assert is compiled
  // out and the silent cross-device hand-off would come straight back.
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool bs_pool(b);  // a pool that belongs to device 1

  const vllm::ActivePoolScope wrong(&bs_pool);
  CHECK_THROWS_AS(DBuf(Dev{a, qa}, DType::kF32, {64}), std::logic_error);
}

TEST_CASE("device pool: ReleaseShared returns the block to the pool it CAME FROM") {
  // The cross-step carrier (device logits, MTP hidden states, MoE scratch) used
  // to be built by hand at ~28 sites, from a deleter that closed over the byte
  // count ALONE and called `Pool().Put(alloc, q)`. So it returned the block to
  // the one global pool whatever device it came from — and, separately, whatever
  // POOL it came from, which silently drained the aux-stream pool into the main
  // one. `ReleaseShared()` captures both.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  const std::vector<int64_t> shape{12288};  // 49,152 bytes @ f32

  void* raw = nullptr;
  {
    std::shared_ptr<void> carrier;
    {
      DBuf x(Dev{a, qa}, DType::kF32, shape);
      raw = x.ptr();
      carrier = x.ReleaseShared();
    }
    // The DBuf is gone but the carrier holds the block: this device's free list
    // must NOT have it yet, so a same-class request allocates afresh.
    DBuf other(Dev{a, qa}, DType::kF32, shape);
    CHECK(other.ptr() != raw);
  }
  if (PoolBypass()) {
    // No free list, so the block does not come back. What the lane still proves
    // — and it is the half that matters — is that the carrier's deleter ran
    // against the buffer's OWN backend rather than an ambient one.
    CHECK(a.WasFreed(raw));
    return;
  }
  // Carrier dropped -> the block is back in THIS device's pool.
  {
    DBuf again(Dev{a, qa}, DType::kF32, shape);
    CHECK(again.ptr() == raw);
  }
}

TEST_CASE("device pool: a scoped pool's block returns to the SCOPED pool, not the device's") {
  // The aux-stream shape, which the hand-written deleters got wrong on every
  // path that used them: a block drawn under an ActivePoolScope and handed to a
  // shared_ptr must come back to the SCOPED pool. Returning it to the device's
  // main pool is how a second stream's block ends up in the first stream's free
  // list — the race AuxPool() exists to prevent.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool scoped(a);  // same DEVICE, different pool
  const std::vector<int64_t> shape{24576};  // 98,304 bytes @ f32

  void* raw = nullptr;
  {
    std::shared_ptr<void> carrier;
    {
      const vllm::ActivePoolScope scope(&scoped);
      DBuf x(Dev{a, qa}, DType::kF32, shape);
      raw = x.ptr();
      carrier = x.ReleaseShared();
    }  // scope ends BEFORE the carrier is dropped, deliberately
  }

  if (PoolBypass()) {
    // Neither pool retains anything under bypass; the deleter freed the block
    // to the backend it was allocated from, which is all this lane can show.
    CHECK(a.WasFreed(raw));
    return;
  }
  // The device's main pool must not have acquired it...
  {
    DBuf from_main(Dev{a, qa}, DType::kF32, shape);
    CHECK(from_main.ptr() != raw);
  }
  // ...the scoped pool must have.
  {
    const vllm::ActivePoolScope scope(&scoped);
    DBuf from_scoped(Dev{a, qa}, DType::kF32, shape);
    CHECK(from_scoped.ptr() == raw);
  }
}

TEST_CASE("device pool: a backend whose PLATFORM is unregistered is REFUSED, not defaulted") {
  // The residency policy is memoized PER DEVICE TYPE now (spec §4 D5), where it
  // used to be one function-local static for the process. That closed the
  // ambient-device hole one layer above the pool, and it opened a new failure
  // mode with it: `platforms::GetPlatform` VT_CHECK-throws for an unregistered
  // type, so a DBuf on such a backend now throws where it previously inherited
  // whichever device happened to resolve FIRST. That is the correct answer — a
  // `device_pool_cap_bytes` read off another platform is a wrong number, not a
  // default — but a new throw with no test is a new throw nobody has run.
  //
  // kXPU is the type with no `RegisterPlatform` call anywhere in the tree (grep
  // src/vllm/platforms/*.cpp: cpu, cuda, rocm, vulkan, metal, tenstorrent). The
  // REQUIRE_FALSE states that as the precondition it is, so the day an XPU
  // platform lands this case goes RED and asks to be re-pointed rather than
  // quietly asserting nothing.
  REQUIRE_FALSE(vllm::platforms::HasPlatform(DeviceType::kXPU));
  TagBackend& a = NewBackend();
  Queue q;
  q.device = Device{DeviceType::kXPU, 0};
  q.handle = nullptr;
  CHECK_THROWS_AS(DBuf(Dev{a, q}, DType::kF32, {64}), std::runtime_error);
}
