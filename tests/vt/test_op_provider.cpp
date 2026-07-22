// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// The acceleration-provider seam's own gate (BACKEND-ACCEL-PROVIDER,
// include/vt/op_provider.h, .agents/specs/metal-mlx-reuse-study.md §6).
//
// WHAT IS UNDER TEST, precisely. Before this seam, `src/vt/ops.cpp` held one
// `void*` per (OpId, DeviceType) and `RegisterOp` overwrote it silently. Two
// providers of one op on one device therefore resolved by STATIC-INIT ORDER
// ACROSS TRANSLATION UNITS, which is unspecified — a nondeterministic BUILD.
// These cases assert the three properties that replace it:
//   1. DETERMINISM — the same provider set selects the same winner regardless of
//      the order the registrars ran. Proved by registering the identical set in
//      OPPOSITE orders on two slots and requiring the same answer.
//   2. DECLINE-AND-FALL-BACK — a provider that cannot serve a call forwards to
//      the next provider DOWN, with the ~70 op entry points unmodified.
//   3. OBSERVABILITY — which provider ran is answerable from the process, not
//      inferred from a green assertion.
//
// All registrations here target DeviceType::kXPU, which no backend in the tree
// registers anything on, so the production CPU/CUDA/Metal/Vulkan op tables are
// untouched by this file and the cases are order-independent of each other.
#include <doctest/doctest.h>

#include <cstring>
#include <string>

#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

using vt::DeviceType;
using vt::OpId;
using vt::OpProvider;
using vt::ProviderCaps;

// Distinct kernel bodies so the selected pointer identifies the provider. Their
// signature is irrelevant to the seam (`fn` is type-erased exactly as the old
// table's `void*` was); what matters is that the addresses differ.
void KernelA() {}
void KernelB() {}
void KernelC() {}

void* Fn(void (*f)()) { return reinterpret_cast<void*>(f); }

OpProvider P(const char* name, int priority, void* fn,
             vt::OpProviderSupportsFn supports = nullptr) {
  OpProvider p;
  p.name = name;
  p.priority = priority;
  p.supports = supports;
  p.fn = fn;
  return p;
}

bool NeverSupports(const ProviderCaps&) { return false; }
bool NeedsComputeMajor9(const ProviderCaps& c) { return c.valid && c.compute_major >= 9; }

}  // namespace

// ---------------------------------------------------------------------------
// 1. DETERMINISM — the property whose absence was the bug.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: selection is independent of registration order") {
  // Same three providers, registered in OPPOSITE orders on two different ops.
  // Under the old flat table this was exactly the case that resolved by whichever
  // static initializer happened to run last.
  vt::RegisterOpProvider(OpId::kAdd, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kAdd, DeviceType::kXPU, P("accel-lo", 5, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kAdd, DeviceType::kXPU, P("accel-hi", 9, Fn(&KernelC)));

  vt::RegisterOpProvider(OpId::kRelu, DeviceType::kXPU, P("accel-hi", 9, Fn(&KernelC)));
  vt::RegisterOpProvider(OpId::kRelu, DeviceType::kXPU, P("accel-lo", 5, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kRelu, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));

  CHECK(vt::GetOp(OpId::kAdd, DeviceType::kXPU) == Fn(&KernelC));
  CHECK(vt::GetOp(OpId::kRelu, DeviceType::kXPU) == Fn(&KernelC));

  // ... and the whole ORDER, not just the winner, is the same both ways.
  for (int i = 0; i < 3; ++i) {
    const char* a = vt::OpProviderNameAt(OpId::kAdd, DeviceType::kXPU, i);
    const char* b = vt::OpProviderNameAt(OpId::kRelu, DeviceType::kXPU, i);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CAPTURE(i);
    CHECK(std::string(a) == std::string(b));
  }
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, DeviceType::kXPU, 0)) == "accel-hi");
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, DeviceType::kXPU, 1)) == "accel-lo");
  CHECK(std::string(vt::OpProviderNameAt(OpId::kAdd, DeviceType::kXPU, 2)) == "vt-native");
  CHECK(vt::OpProviderNameAt(OpId::kAdd, DeviceType::kXPU, 3) == nullptr);
}

TEST_CASE("op provider: equal priority breaks by name, not by registration order") {
  vt::RegisterOpProvider(OpId::kLayerNorm, DeviceType::kXPU, P("zzz-provider", 3, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kLayerNorm, DeviceType::kXPU, P("aaa-provider", 3, Fn(&KernelB)));
  CHECK(vt::GetOp(OpId::kLayerNorm, DeviceType::kXPU) == Fn(&KernelB));
  CHECK(std::string(vt::OpProviderNameAt(OpId::kLayerNorm, DeviceType::kXPU, 0)) == "aaa-provider");
}

TEST_CASE("op provider: a duplicate name is rejected so the order stays total") {
  vt::RegisterOpProvider(OpId::kL2Norm, DeviceType::kXPU, P("dup", 1, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kL2Norm, DeviceType::kXPU, P("dup", 7, Fn(&KernelB)));
  CHECK(vt::OpProviderCount(OpId::kL2Norm, DeviceType::kXPU) == 1);
  CHECK(vt::GetOp(OpId::kL2Norm, DeviceType::kXPU) == Fn(&KernelA));
}

// ---------------------------------------------------------------------------
// 2. CAPABILITY PREDICATE + DECLINE-AND-FALL-BACK (the two fallback axes).
// ---------------------------------------------------------------------------
TEST_CASE("op provider: a provider whose capability predicate fails is skipped") {
  vt::RegisterOpProvider(OpId::kEmbedding, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kEmbedding, DeviceType::kXPU,
                         P("never", 100, Fn(&KernelB), &NeverSupports));
  CHECK(vt::GetOp(OpId::kEmbedding, DeviceType::kXPU) == Fn(&KernelA));
  CHECK(std::string(vt::GetOpProviderStats(OpId::kEmbedding, DeviceType::kXPU).last_selected) ==
        "vt-native");
}

TEST_CASE("op provider: capability predicates re-resolve when the device caps are published") {
  vt::RegisterOpProvider(OpId::kQkvSplit, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kQkvSplit, DeviceType::kXPU,
                         P("sm9-only", 50, Fn(&KernelB), &NeedsComputeMajor9));
  // Unprobed caps: the accelerator must decline rather than guess.
  CHECK(vt::GetOp(OpId::kQkvSplit, DeviceType::kXPU) == Fn(&KernelA));

  ProviderCaps caps;
  caps.valid = true;
  caps.compute_major = 9;
  vt::SetDeviceProviderCaps(DeviceType::kXPU, caps);
  CHECK(vt::GetOp(OpId::kQkvSplit, DeviceType::kXPU) == Fn(&KernelB));

  ProviderCaps old;  // restore the unprobed record for the rest of the file
  vt::SetDeviceProviderCaps(DeviceType::kXPU, old);
  CHECK(vt::GetOp(OpId::kQkvSplit, DeviceType::kXPU) == Fn(&KernelA));
}

TEST_CASE("op provider: a provider declines a call and falls back to the one below it") {
  // This is the SECOND fallback axis — per-CALL refusal (a shape or dtype the
  // accelerator does not handle), which `supports()` cannot express because
  // `GetOp` has no shape to inspect. The declining kernel asks for the provider
  // below itself and forwards. That is what keeps the ~70 op entry points in
  // src/vt/ops.cpp free of any edit: the decline lives inside the kernel.
  vt::RegisterOpProvider(OpId::kMulColVecF32, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kMulColVecF32, DeviceType::kXPU, P("accel", 10, Fn(&KernelB)));
  vt::RegisterOpProvider(OpId::kMulColVecF32, DeviceType::kXPU, P("accel-mid", 5, Fn(&KernelC)));

  CHECK(vt::GetOp(OpId::kMulColVecF32, DeviceType::kXPU) == Fn(&KernelB));
  // "accel" declines -> the next one DOWN in the deterministic order.
  CHECK(vt::GetOpFallback(OpId::kMulColVecF32, DeviceType::kXPU, "accel") == Fn(&KernelC));
  // ... and if that one declines too, the native kernel.
  CHECK(vt::GetOpFallback(OpId::kMulColVecF32, DeviceType::kXPU, "accel-mid") == Fn(&KernelA));
  // Nothing below the native kernel: a decline there is a hard error, never a
  // silent no-op.
  CHECK_THROWS(vt::GetOpFallback(OpId::kMulColVecF32, DeviceType::kXPU, "vt-native"));
  CHECK(vt::GetOpProviderStats(OpId::kMulColVecF32, DeviceType::kXPU).declines >= 3);
}

// ---------------------------------------------------------------------------
// 3. OBSERVABILITY — "did the accelerator actually run?" must be answerable.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: selection is observable through the stats counters") {
  vt::RegisterOpProvider(OpId::kSigmoidGateBf16, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kSigmoidGateBf16, DeviceType::kXPU, P("accel", 10, Fn(&KernelB)));
  vt::ResetOpProviderStats(OpId::kSigmoidGateBf16, DeviceType::kXPU);
  vt::EnableOpProviderCallStats(true);
  for (int i = 0; i < 5; ++i) (void)vt::GetOp(OpId::kSigmoidGateBf16, DeviceType::kXPU);
  vt::EnableOpProviderCallStats(false);

  const auto s = vt::GetOpProviderStats(OpId::kSigmoidGateBf16, DeviceType::kXPU);
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == "accel");
  CHECK(s.selections == 5);
}

TEST_CASE("op provider: a provider can be disabled at run time for a same-binary A/B") {
  vt::RegisterOpProvider(OpId::kCastF32, DeviceType::kXPU, P("vt-native", 0, Fn(&KernelA)));
  vt::RegisterOpProvider(OpId::kCastF32, DeviceType::kXPU, P("ab-accel", 10, Fn(&KernelB)));
  CHECK(vt::GetOp(OpId::kCastF32, DeviceType::kXPU) == Fn(&KernelB));

  vt::DisableOpProvider("ab-accel", true);
  CHECK(vt::OpProviderDisabled("ab-accel"));
  CHECK(vt::GetOp(OpId::kCastF32, DeviceType::kXPU) == Fn(&KernelA));

  vt::DisableOpProvider("ab-accel", false);
  CHECK_FALSE(vt::OpProviderDisabled("ab-accel"));
  CHECK(vt::GetOp(OpId::kCastF32, DeviceType::kXPU) == Fn(&KernelB));
}

// ---------------------------------------------------------------------------
// 4. BEHAVIOUR PRESERVATION — every backend in the tree still registers through
//    plain RegisterOp() and must keep winning exactly as before.
// ---------------------------------------------------------------------------
TEST_CASE("op provider: RegisterOp is the priority-0 vt-native provider, unchanged") {
  vt::RegisterOp(OpId::kCastBf16, DeviceType::kXPU, Fn(&KernelA));
  CHECK(vt::OpProviderCount(OpId::kCastBf16, DeviceType::kXPU) == 1);
  CHECK(std::string(vt::OpProviderNameAt(OpId::kCastBf16, DeviceType::kXPU, 0)) ==
        vt::kNativeProviderName);
  CHECK(vt::OpRegistered(OpId::kCastBf16, DeviceType::kXPU));
  CHECK(vt::GetOp(OpId::kCastBf16, DeviceType::kXPU) == Fn(&KernelA));
}

TEST_CASE("op provider: an unrealized (op, device) still probes false and throws") {
  CHECK_FALSE(vt::OpRegistered(OpId::kPagedAttention, DeviceType::kXPU));
  CHECK_THROWS(vt::GetOp(OpId::kPagedAttention, DeviceType::kXPU));
  // The probe stays false after the throw (the negative resolution is memoized,
  // it is not a one-shot).
  CHECK_FALSE(vt::OpRegistered(OpId::kPagedAttention, DeviceType::kXPU));
}

TEST_CASE("op provider: the real CPU backend resolves to vt-native") {
  CHECK(vt::OpRegistered(OpId::kMatmul, DeviceType::kCPU));
  (void)vt::GetOp(OpId::kMatmul, DeviceType::kCPU);
  const auto s = vt::GetOpProviderStats(OpId::kMatmul, DeviceType::kCPU);
  REQUIRE(s.last_selected != nullptr);
  CHECK(std::string(s.last_selected) == vt::kNativeProviderName);
}
