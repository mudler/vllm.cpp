// ENG-EXPERT-STREAM-DEVICE W0c (issue #1124): does the expert-slice seam serve a
// WEIGHT-STAGING device that can read host memory, and does it do so WITHOUT
// staging the tower?
//
// WHY THIS FILE EXISTS, AND WHAT IT DOES NOT CLAIM. It does not claim to prove
// that decode reaches `KqExpertSlice`; `test_expert_stream_wiring` proves that
// through `Qwen3_5Model::Forward` and it is the file to read for reachability of
// the seam. What is proved here is the PLATFORM BRANCH inside that seam, and the
// distinction matters because the two conditions differ in one bit that no CPU
// tier can supply: `needs_weight_staging() && host_memory_is_device_addressable()`.
// Exactly one machine this project can reach answers that combination (a GB10),
// and a branch provable only on one leased box is the untestable-device shape
// this row has already been bitten by. A fake platform in the otherwise-unused
// kXPU slot supplies the bit.
//
// THE DEFECT UNDER TEST IS AN ALLOCATION, not an output. Before W0c the slot arm
// called `ResidentWeight` purely to inherit dtype/device/repack markers. On CPU
// that aliases and costs nothing; on a staging platform it runs
// `d.b.Alloc(w.bytes.size())` over the whole stacked `[E*N,K]` tower — 1.1875 GiB
// on the target checkpoint — and memoizes it on `w.d_dev`. So merely lifting the
// `is_cpu()` guard would reproduce issue #1123 (48 towers staged, death partway
// through layer 16 of 93) rather than fix it. `w.d_dev` is therefore the
// assertion in almost every case below: it is the observable that says whether
// the tower was uploaded, and it is nonzero for the defect and null for the fix.
//
// NO vt OP RUNS HERE, deliberately. The seam under test does memory work only —
// a platform query, a slot fill by memcpy, and a tensor construction — so the
// fake backend needs an allocator and a copy and nothing else. Adding a kernel
// dispatch would add a reason for this file to fail that has nothing to do with
// what it measures.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/test_env.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::OwnedTensor;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// Turn the lane on before anything can read the environment: the answer latches
// in a function-local static on first use, so setting it inside a case would
// work today and break the moment a case ordering changed. Same discipline, and
// the same reason, as test_expert_stream_wiring.
struct EnableExpertStreaming {
  EnableExpertStreaming() {
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM", "1");
    // Comfortably more than this file's working set (one tower, 4 experts), so
    // an exhausted slot is only ever the FORCED one a case asks for.
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOTS", "16");
    if (std::getenv("VT_MOE_EXPERT_STREAM_STATS_EVERY") == nullptr)
      vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0");
    vllm_test::SetEnv("VT_QWEN35_GROUPED_MOE", "0");
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

// A backend over ordinary host memory. It stands in for a device allocator, and
// the point of using malloc is that a STAGED tower is then a real, inspectable
// allocation whose address differs from the tower's own bytes — which is how a
// case tells "staged" from "aliased" without a GPU.
class HostBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    ++allocs;
    alloc_bytes += bytes;
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  void DestroyQueue(Queue&) override {}
  // FALSE, and that is the honest answer for the device being modelled: on a
  // GB10 a `cudaMalloc` pointer is still not host-dereferenceable. It is also
  // what keeps `AdoptDeviceBytesAsHost` inert, so a staged tower stays visibly
  // staged instead of being folded back onto the host buffer.
  bool UnifiedMemory() const override { return false; }

  // Counted so a case can say HOW MANY allocations it observed rather than only
  // that a pointer was or was not null.
  int allocs = 0;
  size_t alloc_bytes = 0;
};

HostBackend& Fake() {
  static HostBackend b;
  return b;
}

// The GB10 shape: a platform that STAGES its ordinary weights and whose kernels
// can nevertheless DEREFERENCE host storage. `host_addressable` is a settable
// field rather than a second registered platform because the platform registry
// is process-global: two registrations would fight, and one flag lets a case
// prove the predicate SELECTS by moving only the bit under test.
class FakeDevicePlatform final : public vllm::platforms::Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  vt::Backend& backend() const override { return Fake(); }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {12, 1};
  }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
  bool needs_weight_staging() const override { return true; }
  bool host_memory_is_device_addressable() const override {
    return host_addressable;
  }

  bool host_addressable = true;
};

FakeDevicePlatform& Platform_() {
  static FakeDevicePlatform p;
  return p;
}

struct Registrar {
  Registrar() {
    vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &Fake());
    vllm::platforms::RegisterPlatform(DeviceType::kXPU, &Platform_());
  }
};
const Registrar kRegistrar;

// Tower geometry. Q8_0 packs 32 elements into 34 bytes, so K must be a multiple
// of 32 and one row is 68 bytes.
constexpr int64_t kExperts = 4;
constexpr int64_t kN = 8;    // rows per expert
constexpr int64_t kK = 64;   // columns
constexpr size_t kRowBytes = 68;
constexpr size_t kSliceBytes = static_cast<size_t>(kN) * kRowBytes;  // 544

// A keep-quant STACKED expert tower, [E*N, K] Q8_0, `nk = true` — the only shape
// KqExpertSlice slices. The blocks are BUILT rather than filled with noise: a
// Q8_0 block leads with an fp16 scale, and random bytes there encode inf/NaN,
// which would make any later byte comparison vacuous.
OwnedTensor MakeTower(uint8_t tag) {
  OwnedTensor t;
  t.dtype = DType::kQ8_0;
  t.rank = 2;
  t.shape[0] = kExperts * kN;
  t.shape[1] = kK;
  t.nk = true;
  const int64_t blocks_per_row = kK / 32;
  std::vector<uint8_t> b;
  b.reserve(static_cast<size_t>(kExperts * kN) * kRowBytes);
  for (int64_t r = 0; r < kExperts * kN; ++r) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      b.push_back(0x00);  // fp16 1.0, little-endian
      b.push_back(0x3C);
      for (int q = 0; q < 32; ++q)
        b.push_back(static_cast<uint8_t>((r * 7 + blk * 3 + q + tag) & 0x7F));
    }
  }
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

Queue XpuQueue() { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }

const uint8_t* SliceStart(const OwnedTensor& w, int64_t expert) {
  return w.bytes.data() + static_cast<size_t>(expert) * kN * kRowBytes;
}

// Every case below moves a PROCESS-GLOBAL, and an earlier revision of this file
// put each one back on the case's last line. That restores it only on the path
// that REACHES the last line: a doctest `REQUIRE` throws, doctest catches the
// throw and runs the next case, and that next case then runs against the
// previous case's platform. The failure would be reported against the wrong
// case, which is the shape that makes a suite unreadable exactly when it has
// something to say. These two guards restore on every exit path, and they
// restore the PREVIOUS value rather than a hard-coded one, so nesting or
// reordering cases cannot make a guard lie either.
class HostAddressable {
 public:
  explicit HostAddressable(bool on) : prev_(Platform_().host_addressable) {
    Platform_().host_addressable = on;
  }
  ~HostAddressable() { Platform_().host_addressable = prev_; }
  HostAddressable(const HostAddressable&) = delete;
  HostAddressable& operator=(const HostAddressable&) = delete;

 private:
  bool prev_;
};

class ForcedFallback {
 public:
  ForcedFallback() { vllm::detail::ExpertStreamSetForceFallback(true); }
  ~ForcedFallback() { vllm::detail::ExpertStreamSetForceFallback(false); }
  ForcedFallback(const ForcedFallback&) = delete;
  ForcedFallback& operator=(const ForcedFallback&) = delete;
};

}  // namespace

TEST_CASE("a host-addressable staging device takes the SLOT arm and stages nothing") {
  const HostAddressable host_addressable(true);
  const OwnedTensor tower = MakeTower(/*tag=*/1);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  const Tensor t = vllm::detail::ExpertSliceForTest(q, tower, kN, kK,
                                                    /*row_off=*/2 * kN,
                                                    /*expert=*/2);

  // THE #1123 ASSERTION. `d_dev` is set by, and only by, ResidentWeight's
  // staging branch. Null means the 1.1875-GiB-shaped upload did not happen, and
  // it is what separates this change from "lift the is_cpu() guard".
  CHECK(tower.d_dev == nullptr);
  CHECK(Fake().allocs == allocs_before);

  // The tower is CLAIMED by the lane, which is what arms the refusal below.
  CHECK(tower.expert_streamed);

  // The bytes came from a SLOT, not from the tower's own mapping: same content,
  // different address. Both halves are asserted, because either alone is
  // satisfied by a defect — equal-content-same-address is the mapping view, and
  // different-address-different-content is a slot filled from the wrong offset.
  REQUIRE(t.data != nullptr);
  CHECK(t.data != static_cast<const void*>(SliceStart(tower, 2)));
  CHECK(std::memcmp(t.data, SliceStart(tower, 2), kSliceBytes) == 0);

  // The marker set ResidentWeight's aliasing branch carried, carried here.
  CHECK(t.dtype == tower.dtype);
  CHECK(t.device.type == DeviceType::kXPU);
  CHECK(t.rank == 2);
  CHECK(t.shape[0] == kN);
  CHECK(t.shape[1] == kK);
  CHECK(t.stride[0] == kK);
  CHECK(t.stride[1] == 1);
  CHECK(t.repacked == tower.repacked);
  CHECK(t.elem_kn_repacked == tower.elem_kn_repacked);

  // The lane really ran: a slot was filled, and nothing was refused.
  const vllm::detail::ExpertStreamStats s = vllm::detail::ExpertStreamSnapshot();
  CHECK(s.active);
  CHECK(s.fills > 0);
  CHECK(s.bytes_filled > 0);
  CHECK(s.exhausted == 0);
}

TEST_CASE("the exhausted fallback reads the tower IN PLACE and still stages nothing") {
  // WHY THIS CASE IS NOT OPTIONAL. Prefill exhausts the cache by construction:
  // slices acquired within a step are protected from eviction, so the peak
  // protected set for a T-token prompt is `93 x 3 x min(512, 10*T)` slices,
  // which saturates at 331 GiB — the whole model — for any T >= 52. No slot
  // budget makes it fit. If the exhausted branch staged the tower, a real
  // prefill would take that branch thousands of times and die exactly as #1123
  // did, while every "streaming works" gate stayed green.
  const HostAddressable host_addressable(true);
  const OwnedTensor tower = MakeTower(/*tag=*/2);
  Queue q = XpuQueue();

  const Tensor t = [&] {
    const ForcedFallback forced;
    return vllm::detail::ExpertSliceForTest(q, tower, kN, kK,
                                           /*row_off=*/1 * kN, /*expert=*/1);
  }();

  CHECK(tower.d_dev == nullptr);
  // The tower's OWN bytes, at the slice offset — the direct host view, which is
  // exactly what the CPU arm reads through KqResidentSlice.
  CHECK(t.data == static_cast<const void*>(SliceStart(tower, 1)));
  CHECK(t.dtype == tower.dtype);
  CHECK(t.shape[0] == kN);
  CHECK(t.shape[1] == kK);
  CHECK(t.stride[0] == kK);
}

TEST_CASE("a device that CANNOT read host memory keeps staging the whole tower") {
  // The predicate SELECTS. This is the discrete-GPU answer, and it is
  // deliberately the pre-W0c behaviour: no slot arm, a full tower upload, and
  // therefore the #1123 load-time refusal still standing in front of it. W1/W2
  // are what remove it, not this branch.
  const HostAddressable host_addressable(false);
  const OwnedTensor tower = MakeTower(/*tag=*/3);
  Queue q = XpuQueue();
  const int allocs_before = Fake().allocs;

  const Tensor t = vllm::detail::ExpertSliceForTest(q, tower, kN, kK,
                                                    /*row_off=*/3 * kN,
                                                    /*expert=*/3);

  REQUIRE(tower.d_dev != nullptr);
  CHECK(Fake().allocs == allocs_before + 1);
  // The WHOLE stacked tower, not one slice: this is the allocation whose size is
  // the defect on a real checkpoint.
  CHECK(Fake().alloc_bytes >= tower.bytes.size());
  CHECK_FALSE(tower.expert_streamed);
  // ...and the returned tensor points INTO that staged copy.
  CHECK(t.data == static_cast<void*>(static_cast<uint8_t*>(tower.d_dev.get()) +
                                     static_cast<size_t>(3) * kN * kRowBytes));
  CHECK(std::memcmp(t.data, SliceStart(tower, 3), kSliceBytes) == 0);
}

TEST_CASE("a STREAMED tower that reaches device staging is refused BY NAME") {
  // The tripwire. It has no production caller today — with the lane on, the
  // grouped-MoE route that would stage a tower is disabled, and W0c's own
  // fallback reads host bytes in place — and that is the point: the failure it
  // guards is silent until the allocator runs out, 48 towers and 16 layers
  // later. A guard nothing can reach is not a guard, so this reaches it.
  const HostAddressable host_addressable(true);
  const OwnedTensor tower = MakeTower(/*tag=*/4);
  Queue q = XpuQueue();

  // Claim it for the lane through the production seam, not by writing the flag:
  // if `KqExpertSlice` ever stopped marking the tower, this case must go red
  // rather than keep testing a flag the code no longer sets.
  (void)vllm::detail::ExpertSliceForTest(q, tower, kN, kK, /*row_off=*/0,
                                         /*expert=*/0);
  REQUIRE(tower.expert_streamed);
  REQUIRE(tower.d_dev == nullptr);

  CHECK_THROWS_WITH_AS(
      vllm::detail::StageWeightForTest(q, tower),
      doctest::Contains("a STREAMED expert tower reached device staging"),
      std::runtime_error);

  // ...and the refusal fired BEFORE the allocation, which is the whole reason it
  // is placed where it is.
  CHECK(tower.d_dev == nullptr);
}

TEST_CASE("an unclaimed tower still stages normally, so the refusal is not a blanket") {
  // The negative control for the case above. A refusal that fired for every
  // tower would pass that case and break every model, so the same helper must
  // succeed on a tower the lane never touched.
  const HostAddressable host_addressable(true);
  const OwnedTensor plain = MakeTower(/*tag=*/5);
  Queue q = XpuQueue();

  CHECK_FALSE(plain.expert_streamed);
  CHECK_NOTHROW(vllm::detail::StageWeightForTest(q, plain));
  CHECK(plain.d_dev != nullptr);
}
