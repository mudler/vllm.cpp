// #2711: does `ResidentWeightF32` still hold its copy source when the copy runs?
//
// THE DEFECT. The helper builds the f32 upcast of a bf16 weight in a
// FUNCTION-LOCAL `std::vector<float>`, hands `f.data()` to `Backend::Copy`, and
// lets the vector die at the closing brace. `Copy` is asynchronous on both
// device backends -- `cudaMemcpyAsync` (`src/vt/cuda/cuda_backend.cu:116-118`)
// and `hipMemcpyAsync` (`src/vt/rocm/rocm_backend.hip:269-271`) -- and the
// source is ordinary pageable heap memory, which a driver is free to read at
// any point before the queue drains. Nothing in the helper made it drain.
//
// The tree already states this rule for the same hazard, in
// `src/vllm/model_executor/models/glm5_next_kv.cpp:143-150`: a deferred wait
// "would hand the driver a pageable source that the next iteration has already
// overwritten." The sibling `ResidentWeight` does NOT synchronise and is right
// not to -- its source is `w.bytes`, which outlives the call. The asymmetry is
// the whole defect.
//
// WHY THE FAKE BACKEND DEFERS, AND WHAT THAT BUYS. On the CPU backend `Copy` is
// a `memcpy` and `Synchronize` is `vt::Backend`'s default no-op, so on a
// CPU-only host the race is not expressible: every ordering produces the same
// bytes. The backend below therefore implements the contract `cudaMemcpyAsync`
// actually offers -- `Copy` RECORDS the transfer and returns, `Synchronize`
// performs it -- and poisons every allocation. Under that backend the defect is
// deterministic rather than driver-dependent: with no drain, the destination
// still holds poison when the function returns.
//
// WHAT THIS FILE PROVES. That `ResidentWeightF32` does not return while its copy
// is outstanding, so the source outlives the transfer.
//
// WHAT IT DOES NOT PROVE, in those words. It does not prove that a real CUDA or
// ROCm driver defers this particular transfer, and no test on this host can:
// #2711 records that a small pageable H2D copy is usually staged eagerly, which
// is exactly why the bug has stayed latent. This is a STRUCTURAL AND ORDERING
// gate over a SIMULATED asynchronous backend, not an observation of the race.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
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

// Every allocation starts as this byte, repeated. It is the "the copy never
// happened" value, and it is what a case sees instead of the weight when the
// transfer is still queued at the moment the source is destroyed.
constexpr int kPoison = 0xA5;

// A backend whose `Copy` is DEFERRED, which is the one property of a real device
// backend that a CPU box cannot otherwise express.
class DeferringBackend final : public vt::Backend {
 public:
  struct Pending {
    void* dst;
    const void* src;
    size_t bytes;
  };

  void* Alloc(size_t bytes) override {
    const size_t n = bytes == 0 ? 1 : bytes;
    ++allocs;
    void* p = std::malloc(n);
    std::memset(p, kPoison, n);
    return p;
  }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  // RECORD ONLY. This is the whole point of the fixture.
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    ++copies;
    pending.push_back(Pending{dst, src, bytes});
  }
  void Synchronize(Queue&) override {
    ++syncs;
    for (const Pending& p : pending) std::memcpy(p.dst, p.src, p.bytes);
    pending.clear();
  }
  Queue CreateQueue() override {
    return Queue{Device{DeviceType::kXPU, 0}, nullptr};
  }
  void DestroyQueue(Queue&) override {}
  bool UnifiedMemory() const override { return false; }
  bool DeviceMemoryIsHostAddressable() const override { return false; }

  bool HasPending(const void* dst) const {
    for (const Pending& p : pending)
      if (p.dst == dst) return true;
    return false;
  }
  // ABANDON, never run. `ResidentWeight`'s raw uploads legitimately stay queued
  // across a `PrepareBf16Resident` call (their source is the weight's own long-
  // lived bytes, so nothing has to drain them here), and on the UNFIXED tree the
  // f32 entry's `src` dangles. Executing either at teardown would be the test
  // reading freed memory to observe that the code under test would have.
  void DropPending() { pending.clear(); }

  std::vector<Pending> pending;
  int allocs = 0;
  int copies = 0;
  int syncs = 0;
};

DeferringBackend& Fake() {
  static DeferringBackend b;
  return b;
}

// A staging platform that is NOT the CPU: `needs_weight_staging()` is what
// `PrepareBf16Resident` requires, and `is_cpu()` false is what sends
// `ResidentWeightF32` down the upload arm rather than the alias arm. The
// unused kXPU slot holds it, as in `test_resident_weight_host_addressable.cpp`.
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
  bool host_memory_is_device_addressable() const override { return false; }
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

Queue XpuQueue() { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
Queue CpuQueue() { return Queue{Device{DeviceType::kCPU, 0}, nullptr}; }

constexpr int64_t kDh = 12;

// A bf16 [n] norm weight -- the shape `attn.q_norm` and `attn.k_norm` arrive as,
// and one of the four tensors `PrepareBf16Resident` hands to the f32 upcast.
OwnedTensor MakeNorm(int64_t n, uint16_t tag) {
  OwnedTensor t;
  t.dtype = DType::kBF16;
  t.rank = 1;
  t.shape[0] = n;
  std::vector<uint8_t> b(static_cast<size_t>(n) * 2);
  for (int64_t i = 0; i < n; ++i) {
    // bf16 bit patterns in the normal range, distinct per element and per tag,
    // and never equal to the poison word.
    const uint16_t bits = static_cast<uint16_t>(0x3F00u + tag * 16u + i);
    b[static_cast<size_t>(i) * 2] = static_cast<uint8_t>(bits & 0xFFu);
    b[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(bits >> 8);
  }
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

std::vector<float> ExpectUpcast(const OwnedTensor& w) {
  std::vector<float> out(static_cast<size_t>(w.Numel()));
  const uint8_t* src = w.bytes.data();
  for (size_t i = 0; i < out.size(); ++i) {
    uint16_t bits = 0;
    std::memcpy(&bits, src + i * 2, sizeof(bits));
    out[i] = vt::BF16ToF32(bits);
  }
  return out;
}

// How many of the destination's floats differ from the upcast. Reported as ONE
// count rather than as N assertions, so a red says "the copy did not land"
// instead of burying that under kDh identical failures.
int Mismatches(const OwnedTensor& w, const std::vector<float>& expect) {
  const float* got = static_cast<const float*>(w.d_dev_f32.get());
  int bad = 0;
  for (size_t i = 0; i < expect.size(); ++i)
    if (got[i] != expect[i]) ++bad;
  return bad;
}

}  // namespace

// CASE 1 -- THROUGH A PRODUCTION ENTRY POINT.
//
// `Qwen3_5DenseModel::PrepareBf16Resident` is a real load-time hook, not a test
// seam: `StageAndReleaseLoadedDense` calls it (`qwen3_5_dense_weights.cpp:183`)
// and it hands `attn.q_norm` and `attn.k_norm` to the qwen3_5.cpp copy of
// `ResidentWeightF32` -- grep `f32(attn.q_norm)` in `PrepareBf16Resident` rather
// than trusting a line number, which goes stale inside its own pull request.
// Nothing here constructs the helper's inputs by hand and calls it directly.
//
// It also shows why the drain that ALREADY exists is not the repair.
// `StageAndReleaseLoadedDense` synchronises once, AFTER this function returns --
// by which point every `std::vector<float>` the loop created is long gone. That
// is the deferred wait `glm5_next_kv.cpp` rejects, spelled out over a whole
// model.
TEST_CASE("PrepareBf16Resident: each f32 upcast copy RETIRES before its source dies") {
  DeferringBackend& b = Fake();
  vllm::Qwen3_5DenseWeights weights;
  weights.layers.resize(1);
  vllm::Qwen3_5DenseLayerWeights& layer = weights.layers[0];
  layer.is_linear_attention = false;
  layer.attn.q_norm = MakeNorm(kDh, /*tag=*/1);
  layer.attn.k_norm = MakeNorm(kDh, /*tag=*/2);
  const std::vector<float> expect_q = ExpectUpcast(layer.attn.q_norm);
  const std::vector<float> expect_k = ExpectUpcast(layer.attn.k_norm);

  Queue q = XpuQueue();
  vllm::Qwen3_5DenseModel::PrepareBf16Resident(weights, q);

  REQUIRE(layer.attn.q_norm.d_dev_f32 != nullptr);
  REQUIRE(layer.attn.k_norm.d_dev_f32 != nullptr);

  // THE ORDERING. No transfer into either upcast buffer may still be queued: the
  // function returned, so its source is gone, so a queued read of that source is
  // a read of freed memory.
  CHECK_FALSE(b.HasPending(layer.attn.q_norm.d_dev_f32.get()));
  CHECK_FALSE(b.HasPending(layer.attn.k_norm.d_dev_f32.get()));

  // THE CONSEQUENCE. Having retired while the source was alive, the device
  // buffer holds the weight. Unfixed, it still holds the allocator's poison.
  CHECK(Mismatches(layer.attn.q_norm, expect_q) == 0);
  CHECK(Mismatches(layer.attn.k_norm, expect_k) == 0);

  b.DropPending();
}

// CASE 2 -- THE HEADER COPY, and this one is a DIRECT CALL.
//
// `dense_attn::ResidentWeightF32` is reached in production from `DenseAttnBlock`
// -- four call sites, `grep -n 'attn_f32 ? ResidentWeightF32' dense_attn_block.h`
// -- on every attention layer of the 49 translation units that include the
// header, and driving one of those needs full attention metadata and a paged KV
// cache. So this case calls the inline helper directly and says so rather than
// dressing it up.
//
// What makes case 1's production evidence carry to here is that after #2711 both
// helpers install through ONE body, `dense_attn::InstallResidentF32`. This case
// is the assertion that the header's caller really does route into it.
TEST_CASE("dense_attn::ResidentWeightF32 RETIRES its copy too (direct call)") {
  DeferringBackend& b = Fake();
  const OwnedTensor w = MakeNorm(kDh, /*tag=*/3);
  const std::vector<float> expect = ExpectUpcast(w);

  Queue q = XpuQueue();
  vllm::dense_attn::Dev d{b, q};
  const Tensor t = vllm::dense_attn::ResidentWeightF32(d, w, {kDh});

  REQUIRE(w.d_dev_f32 != nullptr);
  CHECK(t.data == w.d_dev_f32.get());
  CHECK(t.dtype == DType::kF32);
  CHECK_FALSE(b.HasPending(w.d_dev_f32.get()));
  CHECK(Mismatches(w, expect) == 0);

  b.DropPending();
}

// CASE 3 -- THE CPU ARM IS UNTOUCHED.
//
// On a CPU queue the helper aliases a heap buffer it owns outright: no device
// allocation, no `Copy`, and therefore nothing to drain. A repair that reached
// this arm would be adding a synchronise to a path that has no queue, so this
// case is what stops the fix from spreading.
TEST_CASE("the CPU arm still ALIASES: no allocation, no copy, no synchronise") {
  DeferringBackend& b = Fake();
  const OwnedTensor w = MakeNorm(kDh, /*tag=*/4);
  const std::vector<float> expect = ExpectUpcast(w);
  const int allocs_before = b.allocs;
  const int copies_before = b.copies;
  const int syncs_before = b.syncs;

  Queue q = CpuQueue();
  vllm::dense_attn::Dev d{b, q};
  const Tensor t = vllm::dense_attn::ResidentWeightF32(d, w, {kDh});

  REQUIRE(w.d_dev_f32 != nullptr);
  CHECK(t.data == w.d_dev_f32.get());
  CHECK(b.allocs == allocs_before);
  CHECK(b.copies == copies_before);
  CHECK(b.syncs == syncs_before);
  CHECK(Mismatches(w, expect) == 0);
}
