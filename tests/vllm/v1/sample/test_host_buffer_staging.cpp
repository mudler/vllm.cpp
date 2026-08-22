// vllm.cpp original (vt-runtime residency; vLLM has no mirror — torch owns
// residency there via .to(device)).
//
// The gate for #1313: the [rows, vocab] logits tensor the ON-DEVICE sampler runs
// on was built from a HOST std::vector pointer stamped with the queue device
// (src/vllm/v1/worker/gpu/runner.cpp, four sites). vt::GreedyArgmax does NOT copy
// its input — its CUDA arm hands the caller's pointer straight to a kernel
// (src/vt/cuda/cuda_sample.cu:199 and :207) — so on a DISCRETE GPU that is an
// illegal address. On GB10 it survives because CudaBackend::UnifiedMemory() is
// `pageable_memory_access && integrated` (src/vt/cuda/cuda_backend.cu:363), true
// there, and the driver services the host pointer through ATS.
//
// That is why this test uses FAKE backends rather than hardware: every backend
// registered on this box reports UnifiedMemory() == true, so the defect is
// invisible here by construction. The idiom — a Backend over ordinary host
// memory registered on the otherwise-unused kXPU slot, one unified instance and
// one discrete instance — is the one tests/vt/test_reference_tier.cpp already
// uses to gate the reference tier's discrete/unified split without a GPU.
//
// This file is its own executable (tests/CMakeLists.txt: one add_executable per
// test), so the kXPU registration cannot leak into another test binary.
#include "vllm/v1/sample/device_scratch.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::v1::HostBufferStaging;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// A Backend over ordinary host memory. `unified` is the only property under
// test; everything else is a plain allocator so a staged copy actually happens
// and can be read back. Alloc/Free are counted so the grow-only contract is
// observable rather than assumed.
class FakeBackend final : public Backend {
 public:
  explicit FakeBackend(bool unified) : unified_(unified) {}
  void* Alloc(size_t bytes) override {
    ++allocs;
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override {
    if (p != nullptr) ++frees;
    std::free(p);
  }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    ++copies;
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  bool UnifiedMemory() const override { return unified_; }

  int allocs = 0;
  int frees = 0;
  int copies = 0;

 private:
  bool unified_;
};

FakeBackend& Unified() {
  static FakeBackend b(true);
  return b;
}
FakeBackend& Discrete() {
  static FakeBackend b(false);
  return b;
}

constexpr Device kXpu{DeviceType::kXPU, 0};

// A [rows, vocab] host logits buffer, exactly the shape the runner assembles.
std::vector<float> HostLogits(int64_t rows, int64_t vocab) {
  std::vector<float> v(static_cast<size_t>(rows * vocab));
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i) * 0.5f - 3.0f;
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE DEFECT (#1313) — a DISCRETE device must never be handed a host address.
// ---------------------------------------------------------------------------
TEST_CASE("host staging: a discrete device gets DEVICE memory, not the host pointer") {
  vt::RegisterBackend(DeviceType::kXPU, &Discrete());
  Queue q = Discrete().CreateQueue();

  constexpr int64_t kRows = 3, kVocab = 64;
  std::vector<float> host = HostLogits(kRows, kVocab);

  HostBufferStaging staging;
  vt::Tensor t = staging.Stage(kXpu, q, host.data(), DType::kF32, {kRows, kVocab});

  // The whole point: what a kernel will dereference is NOT the std::vector.
  CHECK(t.data != static_cast<void*>(host.data()));
  // ...and it carries the bytes, so staging is a copy and not just a different
  // address. A guard that only checked the pointer would pass on an empty one.
  REQUIRE(t.data != nullptr);
  const auto* staged = static_cast<const float*>(t.data);
  for (size_t i = 0; i < host.size(); ++i) {
    CHECK(staged[i] == doctest::Approx(host[i]));
  }
  // The tensor still reports the device the sampler dispatches on.
  CHECK(t.device.type == DeviceType::kXPU);
  CHECK(t.rank == 2);
  CHECK(t.shape[0] == kRows);
  CHECK(t.shape[1] == kVocab);
  CHECK(t.IsContiguous());
}

// ---------------------------------------------------------------------------
// 2. THE WORKING PATH — a UNIFIED device keeps the zero-copy wrap byte-for-byte.
// ---------------------------------------------------------------------------
// This is the executable form of "the NemotronH A3 gate stays 96/96". GB10 is a
// unified backend, so the repair must not move it off the in-place wrap it has
// today: no copy, no allocation, the same address.
TEST_CASE("host staging: a unified device wraps IN PLACE, with no copy and no alloc") {
  vt::RegisterBackend(DeviceType::kXPU, &Unified());
  Queue q = Unified().CreateQueue();

  constexpr int64_t kRows = 3, kVocab = 64;
  std::vector<float> host = HostLogits(kRows, kVocab);

  const int allocs_before = Unified().allocs;
  const int copies_before = Unified().copies;

  HostBufferStaging staging;
  vt::Tensor t = staging.Stage(kXpu, q, host.data(), DType::kF32, {kRows, kVocab});

  CHECK(t.data == static_cast<void*>(host.data()));
  CHECK(Unified().allocs == allocs_before);
  CHECK(Unified().copies == copies_before);

  // In-place means the sampler's mutation is visible on the host buffer, which
  // is what apply_temperature / the grammar bitmask have always done here.
  static_cast<float*>(t.data)[0] = 42.0f;
  CHECK(host[0] == doctest::Approx(42.0f));
}

// ---------------------------------------------------------------------------
// 3. GROW-ONLY — no per-step allocation on the decode path.
// ---------------------------------------------------------------------------
TEST_CASE("host staging: the discrete allocation is reused across steps and grows once") {
  vt::RegisterBackend(DeviceType::kXPU, &Discrete());
  Queue q = Discrete().CreateQueue();

  std::vector<float> big = HostLogits(4, 64);
  std::vector<float> small = HostLogits(1, 64);

  HostBufferStaging staging;
  const int allocs_before = Discrete().allocs;

  vt::Tensor a = staging.Stage(kXpu, q, big.data(), DType::kF32, {4, 64});
  const int after_first = Discrete().allocs;
  CHECK(after_first == allocs_before + 1);
  void* first_ptr = a.data;

  // A SMALLER step must not allocate again — this is the per-token decode step.
  vt::Tensor b = staging.Stage(kXpu, q, small.data(), DType::kF32, {1, 64});
  CHECK(Discrete().allocs == after_first);
  CHECK(b.data == first_ptr);
  const auto* staged = static_cast<const float*>(b.data);
  for (size_t i = 0; i < small.size(); ++i) CHECK(staged[i] == doctest::Approx(small[i]));

  // A LARGER one grows exactly once, and frees the old block rather than leaking.
  const int frees_before = Discrete().frees;
  std::vector<float> bigger = HostLogits(8, 64);
  vt::Tensor c = staging.Stage(kXpu, q, bigger.data(), DType::kF32, {8, 64});
  CHECK(Discrete().allocs == after_first + 1);
  CHECK(Discrete().frees == frees_before + 1);
  const auto* staged_c = static_cast<const float*>(c.data);
  for (size_t i = 0; i < bigger.size(); ++i) CHECK(staged_c[i] == doctest::Approx(bigger[i]));
}

// ---------------------------------------------------------------------------
// 4. TWO BUFFERS DO NOT ALIAS.
// ---------------------------------------------------------------------------
// collect_prompt_logprobs stages the prompt rows while the assembled sample
// logits are still live (runner.cpp:2083-2090), so the runner holds two staging
// objects. One shared buffer would invalidate the tensor the sampler is about to
// read — this pins that they are independent.
TEST_CASE("host staging: two staging buffers are independent") {
  vt::RegisterBackend(DeviceType::kXPU, &Discrete());
  Queue q = Discrete().CreateQueue();

  std::vector<float> sample_rows = HostLogits(2, 32);
  std::vector<float> prompt_rows(static_cast<size_t>(2 * 32), 7.0f);

  HostBufferStaging sample_staging;
  HostBufferStaging prompt_staging;

  vt::Tensor s = sample_staging.Stage(kXpu, q, sample_rows.data(), DType::kF32, {2, 32});
  vt::Tensor p = prompt_staging.Stage(kXpu, q, prompt_rows.data(), DType::kF32, {2, 32});

  CHECK(s.data != p.data);
  // Staging the prompt rows must not have disturbed the sample rows.
  const auto* sd = static_cast<const float*>(s.data);
  for (size_t i = 0; i < sample_rows.size(); ++i) CHECK(sd[i] == doctest::Approx(sample_rows[i]));
  const auto* pd = static_cast<const float*>(p.data);
  for (size_t i = 0; i < prompt_rows.size(); ++i) CHECK(pd[i] == doctest::Approx(7.0f));
}
