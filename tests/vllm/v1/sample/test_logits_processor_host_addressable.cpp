// vllm.cpp original (vt-runtime residency; vLLM has no mirror, because torch
// owns residency there through `.to(device)`).
//
// The gate for #1746: `apply_logits_processors`
// (src/vllm/v1/sample/logits_processor/builtin.cpp) hands the ABI logits-processor
// callback a HOST pointer, and it chose that pointer by asking
// `Backend::UnifiedMemory()`. That is the WIDE predicate. The question the code
// has is the NARROW one — may the host DEREFERENCE what `Backend::Alloc`
// returned? — `Backend::DeviceMemoryIsHostAddressable()`. CUDA answers the wide
// one `pageable_memory_access && integrated` (true on GB10) over allocations
// `CudaBackend::Alloc` takes from `cudaMalloc`, and it never overrides the
// narrow one, so on GB10 the callback received a device pointer and
// dereferenced it. Same class as #844 / #1435 / #960, which
// `src/vt/op_provider.cpp` already warns about beside `ReferenceTierEligible`.
//
// Why FAKE backends rather than hardware: the only backend on an ordinary box is
// CPU, where the wrong predicate cannot hurt anybody. `CpuBackend` answers
// `UnifiedMemory()` true over `std::malloc` allocations, so the old code handed
// the callback a pointer that really was host memory. The defect is therefore
// invisible to a CPU suite by construction, and
// tests/vllm/v1/sample/test_logits_processors.cpp is one.
//
// The discriminating shape is the GB10 pair — unified TRUE, host-addressable
// FALSE — which is the third fake instance tests/vt/test_reference_tier.cpp
// already builds for the same reason. Its memory is ordinary host memory, so the
// wrong arm cannot segfault here; the observable proxies are POINTER IDENTITY
// and the backend's COPY COUNT, the same proxies
// tests/vllm/v1/sample/test_host_buffer_staging.cpp uses.
//
// Those two proxies are not sufficient on their own, and the third one is here
// because of that. A copy COUNT says a copy happened and an ADDRESS says it was
// somewhere else; neither reads what the buffer HOLDS. This file's callback
// overwrites the row without ever reading it, so a stage-down leg that carried
// none of the logits would satisfy both proxies. `seen.saw` therefore records
// the row's CONTENTS as the callback received them, and both cases compare it
// against the baseline.
//
// Both cases enter through `Sampler::forward` — the one production call site of
// `apply_logits_processors` (src/vllm/v1/sample/sampler.cpp), reached from the
// `vllm_sampling_params::logits_processor` ABI field — and not through the
// function directly, so deleting that call site turns this file red.
//
// This file is its own executable (tests/CMakeLists.txt: one add_executable per
// test), so the kXPU registration cannot leak into another test binary.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include "vllm/logits_processor_callback.h"
#include "vllm/v1/sample/metadata.h"
#include "vllm/v1/sample/sampler.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace {

using vllm::v1::Sampler;
using vllm::v1::SamplingMetadata;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// A Backend over ordinary HOST memory. The two predicates are SEPARATE fields
// because a real backend separates them: `UnifiedMemory()` says host and device
// address the same physical RAM, `DeviceMemoryIsHostAddressable()` says a
// pointer from `Alloc()` may be dereferenced by the host. CUDA on GB10 answers
// true then false (include/vt/backend.h). Everything else is a plain allocator,
// so a staged copy actually happens and can be read back.
class FakeBackend final : public Backend {
 public:
  FakeBackend(bool unified, bool host_addressable)
      : unified_(unified), host_addressable_(host_addressable) {}
  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    ++copies;
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  bool UnifiedMemory() const override { return unified_; }
  bool DeviceMemoryIsHostAddressable() const override { return host_addressable_; }

  int copies = 0;

 private:
  bool unified_;
  bool host_addressable_;
};

// The GB10 pair: unified memory TRUE, host-addressable FALSE.
FakeBackend& Gb10Shaped() {
  static FakeBackend b(true, false);
  return b;
}
// A genuinely host-addressable unified backend (Vulkan, Metal
// StorageModeShared, integrated ROCm): both predicates TRUE.
FakeBackend& HostAddressable() {
  static FakeBackend b(true, true);
  return b;
}

constexpr Device kXpu{DeviceType::kXPU, 0};

// kXPU has no native kernels, and the portable reference tier is WITHHELD from a
// device that is not host-addressable — that withholding IS the #844 gate, so
// the GB10-shaped fake cannot borrow it. Install the CPU greedy-argmax kernel on
// kXPU explicitly instead. Sound here and only here: this backend's allocations
// are `std::malloc`, so a host kernel over them is a host kernel over host
// memory. Greedy argmax is the ONLY vt op an all-greedy, no-logprobs
// `Sampler::forward` dispatches.
void InstallGreedyArgmaxOnXpu() {
  static const bool done = [] {
    vt::RegisterOp(vt::OpId::kGreedyArgmax, DeviceType::kXPU,
                   vt::GetOp(vt::OpId::kGreedyArgmax, DeviceType::kCPU));
    return true;
  }();
  (void)done;
}

// What the callback SAW, recorded at the moment it ran.
struct SeenState {
  FakeBackend* backend = nullptr;
  const float* row = nullptr;   // the pointer the ABI handed us
  std::vector<float> saw;       // what that pointer HELD, before we overwrote it
  int copies_at_call = 0;       // the backend's copy count when we were called
  int calls = 0;
  int32_t vocab = 0;
  int32_t forced = -1;
};

// Records the pointer, then forces `forced` to win the argmax. The force is what
// makes the COPY-BACK observable: greedy argmax runs over the DEVICE tensor, so
// the forced token can only win if the edited host buffer reached it.
void RecordAndForceCb(const int32_t* /*token_ids*/, int32_t /*n_token_ids*/, float* logits,
                      int32_t vocab_size, void* user_data) {
  auto* s = static_cast<SeenState*>(user_data);
  s->row = logits;
  s->copies_at_call = s->backend->copies;
  s->calls += 1;
  s->vocab = vocab_size;
  // Read the row BEFORE the mask below destroys it. This is the only read this
  // callback does, and without it nothing in this file can tell a staging buffer
  // that carries the logits from one that carries zeroes.
  s->saw.assign(logits, logits + vocab_size);
  for (int32_t j = 0; j < vocab_size; ++j) logits[j] = -1.0e30f;
  if (s->forced >= 0 && s->forced < vocab_size) logits[s->forced] = 1.0e30f;
}

// One decode step through the production entry point, on whichever backend is
// registered on kXPU. Returns the sampled ids; `dev` receives the device
// allocation the sampler ran over, and the caller frees it.
std::vector<std::vector<int32_t>> RunOneStep(FakeBackend& backend, SeenState* seen,
                                             const std::vector<float>& host_logits, int64_t rows,
                                             int64_t vocab, void** dev_out) {
  Queue q = backend.CreateQueue();
  const size_t bytes = host_logits.size() * sizeof(float);
  void* dev = backend.Alloc(bytes);
  std::memcpy(dev, host_logits.data(), bytes);
  *dev_out = dev;

  vt::Tensor tl = vt::Tensor::Contiguous(dev, DType::kF32, kXpu, {rows, vocab});
  SamplingMetadata sm;
  sm.all_greedy = true;
  sm.max_num_logprobs = std::nullopt;
  sm.output_token_ids = {{7, 8}};
  sm.logits_processors[0] = vllm::LogitsProcessorCallback{&RecordAndForceCb, seen};

  Sampler sampler;
  auto out = sampler.forward(q, tl, sm);
  return out.sampled_token_ids;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE DEFECT (#1746) — a backend whose device memory the HOST MAY NOT touch
//    must get the staging bounce, however unified it calls itself.
// ---------------------------------------------------------------------------
TEST_CASE("logits processors: the GB10 pair stages, and never hands over logits.data") {
  vt::RegisterBackend(DeviceType::kXPU, &Gb10Shaped());
  InstallGreedyArgmaxOnXpu();

  // Untouched argmax is token 1; the callback forces token 3.
  const std::vector<float> baseline = {0.1f, 5.0f, 0.2f, 0.3f};
  SeenState seen;
  seen.backend = &Gb10Shaped();
  seen.forced = 3;

  const int copies_before = Gb10Shaped().copies;
  void* dev = nullptr;
  const auto sampled = RunOneStep(Gb10Shaped(), &seen, baseline, 1, 4, &dev);

  REQUIRE(seen.calls == 1);
  REQUIRE(seen.row != nullptr);
  CHECK(seen.vocab == 4);

  // THE DEFECT. `UnifiedMemory()` is true here, so the old code handed the
  // callback the device allocation itself. On GB10 that is a `cudaMalloc`
  // pointer and the callback's first store is a SIGSEGV.
  CHECK(static_cast<const void*>(seen.row) != static_cast<const void*>(dev));
  // ...and it is a real BOUNCE, not merely a different address: exactly one copy
  // (the stage DOWN) happened before the callback ran.
  CHECK(seen.copies_at_call == copies_before + 1);
  // ...and that copy CARRIED THE LOGITS. The two assertions above compare an
  // address and count a copy; neither reads the buffer, so a stage-down leg that
  // copied nothing would pass both of them. Mutating the stage-down copy to
  // `staging <- staging` leaves every other assertion in this file green and
  // fails exactly the four below.
  REQUIRE(seen.saw.size() == baseline.size());
  for (size_t j = 0; j < baseline.size(); ++j) CHECK(seen.saw[j] == baseline[j]);

  // The edits reach the device tensor, so the sampler samples what the callback
  // wrote. This is the stage-BACK leg, observed through the sampled token rather
  // than asserted about the code.
  REQUIRE(sampled.size() == 1);
  REQUIRE(sampled[0].size() == 1);
  CHECK(sampled[0][0] == 3);
  const auto* device_rows = static_cast<const float*>(dev);
  CHECK(device_rows[3] > 0.0f);   // the forced token
  CHECK(device_rows[1] < 0.0f);   // the baseline argmax, masked away

  Gb10Shaped().Free(dev);
}

// ---------------------------------------------------------------------------
// 2. THE WORKING PATH — a genuinely host-addressable backend keeps the
//    zero-copy in-place wrap. The fix must not become "always stage".
// ---------------------------------------------------------------------------
TEST_CASE("logits processors: a host-addressable backend keeps the in-place wrap") {
  vt::RegisterBackend(DeviceType::kXPU, &HostAddressable());
  InstallGreedyArgmaxOnXpu();

  const std::vector<float> baseline = {0.1f, 5.0f, 0.2f, 0.3f};
  SeenState seen;
  seen.backend = &HostAddressable();
  seen.forced = 3;

  const int copies_before = HostAddressable().copies;
  void* dev = nullptr;
  const auto sampled = RunOneStep(HostAddressable(), &seen, baseline, 1, 4, &dev);

  REQUIRE(seen.calls == 1);
  // The callback edits the tensor's own memory: same address, no bounce.
  CHECK(static_cast<const void*>(seen.row) == static_cast<const void*>(dev));
  CHECK(seen.copies_at_call == copies_before);
  // The in-place arm must also hand over the real logits, not merely the right
  // address -- the same content check case 1 makes, for the same reason.
  REQUIRE(seen.saw.size() == baseline.size());
  for (size_t j = 0; j < baseline.size(); ++j) CHECK(seen.saw[j] == baseline[j]);

  REQUIRE(sampled.size() == 1);
  REQUIRE(sampled[0].size() == 1);
  CHECK(sampled[0][0] == 3);

  HostAddressable().Free(dev);
}
