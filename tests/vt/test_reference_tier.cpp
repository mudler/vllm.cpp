// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream vLLM
// test mirrors it (vLLM's forward_native fallback is implicit in torch).
//
// The portable reference tier's own gate (S5,
// .agents/specs/accelerator-seam-audit.md § Work breakdown row S5). It proves the
// three properties the tier must have, WITHOUT needing Metal or Vulkan hardware:
//
//   1. SAFETY (the load-bearing invariant) — the CPU fallback is installed ONLY
//      where host and device memory alias (Backend::UnifiedMemory()). A device
//      that reports DISCRETE memory NEVER receives a CPU fallback; GetOp still
//      throws there, exactly as before, because a CPU kernel against true device
//      memory is corruption. This is asserted against a fake DISCRETE backend.
//
//   2. CORRECTNESS-WITH-ZERO-KERNELS — a device with NO native kernel for an op
//      still produces the right answer through the reference tier. Asserted
//      against a fake UNIFIED backend (host-memory allocator standing in for
//      Metal StorageModeShared / GB10 / integrated Vulkan): vt::Relu dispatched on
//      that device falls back to the CPU kernel and returns bit-identical output.
//
//   3. NATIVE ALWAYS WINS + OBSERVABILITY — a registered native kernel outranks
//      the tier (priority), the tier is never silent (GetReferenceTierHits, and a
//      one-time stderr line), and OpRegistered still means "native kernel present"
//      so the fused-recipe ladder is unchanged.
//
// This file is its own executable (tests/CMakeLists.txt: one add_executable per
// test), so registering a backend on the otherwise-unused kXPU slot cannot leak
// into any other test binary.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::OpId;
using vt::Queue;
using vt::Tensor;

// A minimal backend over ordinary HOST memory. `unified` is the only thing under
// test: a unified instance stands in for Metal/GB10/integrated-Vulkan (host and
// device pointers alias, so the CPU reference tier is sound); a discrete instance
// stands in for a real GPU where it must be refused. Everything else is a plain
// host allocator so a fallback kernel dispatched here actually runs.
class FakeBackend final : public Backend {
 public:
  explicit FakeBackend(bool unified) : unified_(unified) {}
  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  bool UnifiedMemory() const override { return unified_; }

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

void* AsVoid(void (*f)()) { return reinterpret_cast<void*>(f); }
void NativeKernel() {}

}  // namespace

// ---------------------------------------------------------------------------
// 1. SAFETY GATE — a DISCRETE device is never given a CPU fallback.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a discrete (non-unified) device is refused and still throws") {
  vt::RegisterBackend(DeviceType::kXPU, &Discrete());

  // The gate itself.
  CHECK_FALSE(vt::ReferenceTierEligible(DeviceType::kXPU));
  // Eager install is a strict no-op: nothing registered, count 0.
  CHECK(vt::RegisterReferenceTier(DeviceType::kXPU) == 0);
  CHECK(vt::OpProviderCount(OpId::kRelu, DeviceType::kXPU) == 0);
  // And a dispatch on an op it lacks THROWS — a CPU kernel must not be allowed to
  // run against what this device claims is discrete memory.
  CHECK_FALSE(vt::OpRegistered(OpId::kRelu, DeviceType::kXPU));
  CHECK_THROWS(vt::GetOp(OpId::kRelu, DeviceType::kXPU));
}

// The CPU device is the SOURCE of the reference kernels and is never a target.
TEST_CASE("reference tier: the CPU source device is never eligible") {
  CHECK_FALSE(vt::ReferenceTierEligible(DeviceType::kCPU));
}

// ---------------------------------------------------------------------------
// 2. CORRECTNESS WITH ZERO NATIVE KERNELS — a unified device runs via the tier.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a unified device with NO native Relu falls back to the CPU kernel") {
  vt::RegisterBackend(DeviceType::kXPU, &Unified());
  CHECK(vt::ReferenceTierEligible(DeviceType::kXPU));

  // There is no native kMETAL/kVULKAN kernel here; kXPU has none registered.
  REQUIRE_FALSE(vt::OpRegistered(OpId::kRelu, DeviceType::kXPU));

  constexpr int64_t kRows = 5, kCols = 32;
  constexpr size_t kN = kRows * kCols;
  std::vector<float> in(kN);
  for (size_t i = 0; i < kN; ++i) in[i] = static_cast<float>(static_cast<int>(i) - 40) * 0.25f;

  // CPU oracle through the very same vt::Relu entry point.
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ref(kN);
  {
    std::vector<float> ci = in;
    Tensor ti = Tensor::Contiguous(ci.data(), DType::kF32, cd, {kRows, kCols});
    Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kRows, kCols});
    vt::Relu(cq, to, ti);
  }
  cpu.DestroyQueue(cq);

  const unsigned long long hits_before = vt::GetReferenceTierHits();

  // Now the same op on the UNIFIED "accelerator" — with zero native kernels.
  Backend& dev = vt::GetBackend(DeviceType::kXPU);
  Queue q = dev.CreateQueue();
  const Device d{DeviceType::kXPU, 0};
  void* pin = dev.Alloc(kN * sizeof(float));
  void* pout = dev.Alloc(kN * sizeof(float));
  dev.Copy(q, pin, in.data(), kN * sizeof(float));
  Tensor ti = Tensor::Contiguous(pin, DType::kF32, d, {kRows, kCols});
  Tensor to = Tensor::Contiguous(pout, DType::kF32, d, {kRows, kCols});
  vt::Relu(q, to, ti);  // dispatch -> reference tier installs + runs the CPU kernel
  dev.Synchronize(q);

  std::vector<float> got(kN);
  dev.Copy(q, got.data(), pout, kN * sizeof(float));
  // The tier reuses the EXACT CPU kernel, so the result is bit-identical, not
  // merely close: memcmp, no tolerance.
  CHECK(std::memcmp(ref.data(), got.data(), kN * sizeof(float)) == 0);
  dev.Free(pin);
  dev.Free(pout);
  dev.DestroyQueue(q);

  // OBSERVABILITY — the fallback is loud, not silent.
  CHECK(vt::GetReferenceTierHits() > hits_before);
  REQUIRE(vt::OpProviderCount(OpId::kRelu, DeviceType::kXPU) >= 1);
  CHECK(std::string(vt::OpProviderNameAt(OpId::kRelu, DeviceType::kXPU, 0)) ==
        vt::kReferenceProviderName);
  const auto s = vt::GetOpProviderStats(OpId::kRelu, DeviceType::kXPU);
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == vt::kReferenceProviderName);

  // And it is NOT counted as a native kernel — the meaning OpRegistered must keep
  // for the fused-recipe ladder — even though GetOp now returns a working fn.
  CHECK_FALSE(vt::OpRegistered(OpId::kRelu, DeviceType::kXPU));
}

// ---------------------------------------------------------------------------
// 3. NATIVE ALWAYS WINS + eager registration shape.
// ---------------------------------------------------------------------------
TEST_CASE("reference tier: a native kernel outranks the fallback and is left untouched") {
  vt::RegisterBackend(DeviceType::kXPU, &Unified());

  // A native provider for kEmbedding on the "accelerator".
  vt::RegisterOp(OpId::kEmbedding, DeviceType::kXPU, AsVoid(&NativeKernel));
  REQUIRE(vt::OpRegistered(OpId::kEmbedding, DeviceType::kXPU));

  // Eager install must SKIP kEmbedding (already native) and install for the ops
  // that only have a CPU kernel. It returns a positive count and never displaces
  // the native provider.
  const int installed = vt::RegisterReferenceTier(DeviceType::kXPU);
  CHECK(installed > 0);
  CHECK(vt::GetOp(OpId::kEmbedding, DeviceType::kXPU) == AsVoid(&NativeKernel));
  CHECK(vt::OpProviderCount(OpId::kEmbedding, DeviceType::kXPU) == 1);  // no fallback added

  // An op that only had a CPU kernel now resolves through the tier.
  CHECK(vt::OpRegistered(OpId::kMatmul, DeviceType::kCPU));
  CHECK(vt::GetOp(OpId::kMatmul, DeviceType::kXPU) ==
        vt::GetOp(OpId::kMatmul, DeviceType::kCPU));  // same host kernel pointer
}

// A second eager pass is idempotent — the tier providers are already present and
// RegisterOpProvider rejects the duplicate name, so nothing new is installed.
TEST_CASE("reference tier: eager registration is idempotent") {
  vt::RegisterBackend(DeviceType::kXPU, &Unified());
  (void)vt::RegisterReferenceTier(DeviceType::kXPU);
  const int second = vt::RegisterReferenceTier(DeviceType::kXPU);
  CHECK(second == 0);
}
