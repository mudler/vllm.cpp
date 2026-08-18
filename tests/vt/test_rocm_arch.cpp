// Gates on the GCN-arch capability parse (include/vt/rocm/rocm_arch.h), the one
// piece of the ROCm skeleton that holds a decision rather than an API call.
//
// COMPILED IN EVERY BUILD, including a CPU-only build on a machine with no AMD
// GPU and no ROCm toolchain — that is the entire reason the parse lives in a
// plain header. The rest of the ROCm skeleton (rocm_backend.hip, the RmsNorm
// kernel, the platform TU) is gated on VLLM_CPP_HIP and cannot be exercised
// here; see tests/vt/test_rocm_backend.cpp.
//
// The cases are upstream's own worked examples, from the docstring of
// vllm/platforms/rocm.py:223-291 `_capability_from_gcn_arch`, plus the three
// boards offered on issue #41 (gfx1100, gfx1103, gfx1151).
#include <doctest/doctest.h>

#include "vt/rocm/rocm_arch.h"

using vt::rocm::CapabilityFromGcnArch;

namespace {
// Reads as (major, minor) at the call site instead of .first / .second.
void CheckArch(const char* gcn, int major, int minor) {
  const auto cap = CapabilityFromGcnArch(gcn);
  REQUIRE_MESSAGE(cap.has_value(), gcn);
  CHECK_MESSAGE(cap->first == major, gcn);
  CHECK_MESSAGE(cap->second == minor, gcn);
}
}  // namespace

TEST_CASE("gfx9 family parses as 1-digit major, upstream's worked examples") {
  CheckArch("gfx90a", 9, 0);  // MI210 / MI250
  CheckArch("gfx942", 9, 4);  // MI300X / MI325X
  CheckArch("gfx950", 9, 5);  // MI355
}

TEST_CASE("gfx10xx and later parse as 2-digit major") {
  CheckArch("gfx1100", 11, 0);  // RDNA3 dGPU, 7900 XTX (issue #41)
  CheckArch("gfx1101", 11, 0);
  CheckArch("gfx1103", 11, 0);  // RDNA3 iGPU, Radeon 780M (issue #41)
  CheckArch("gfx1151", 11, 5);  // Strix Halo APU (issue #41)
  CheckArch("gfx1200", 12, 0);  // RDNA4
}

TEST_CASE("the HIP feature suffix on gcnArchName is stripped") {
  // hipDeviceProp_t::gcnArchName carries target features on gfx9 parts. The
  // capability must not change because ECC or xnack is on.
  CheckArch("gfx942:sramecc+:xnack-", 9, 4);
  CheckArch("gfx90a:sramecc+:xnack+", 9, 0);
}

TEST_CASE("non-gfx strings are declined, not guessed") {
  // Upstream returns None here (not a ROCm arch string at all) so the caller can
  // fall back. Ours reports "capability unknown" the same way.
  CHECK_FALSE(CapabilityFromGcnArch("").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("sm_90a").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("Apple M4").has_value());
  CHECK_FALSE(CapabilityFromGcnArch("agfx1100").has_value());  // not anchored at 0
}

TEST_CASE("layouts outside the known MMms shape are declined, never split by guess") {
  // Upstream RAISES on each of these rather than returning a value. A skeleton
  // that invented (1, 0) for "gfx1" would hand a wrong capability to a future
  // tactic selector, which is the failure this test exists to prevent.
  CHECK_FALSE(CapabilityFromGcnArch("gfx").has_value());      // no digits
  CHECK_FALSE(CapabilityFromGcnArch("gfx9").has_value());     // too few to split
  CHECK_FALSE(CapabilityFromGcnArch("gfx11000").has_value());  // beyond 4 digits
}

TEST_CASE("major outside [9, 12] is declined") {
  // Both of upstream's sanity rails. A parse that yields major < 9 or > 12 means
  // the layout assumption did not hold, so the answer is not trustworthy.
  CHECK_FALSE(CapabilityFromGcnArch("gfx803").has_value());   // pre-gfx9 (Polaris)
  CHECK_FALSE(CapabilityFromGcnArch("gfx1300").has_value());  // no such generation
}

TEST_CASE("SharedK WMMA host gate is gfx1200/gfx1201 prefix, not substring") {
  using vt::rocm::GcnArchNameIsGfx12PrefillWmma;
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1200"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201:xnack-"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1201:sramecc+"));
  CHECK(GcnArchNameIsGfx12PrefillWmma("gfx1200:xnack-"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma(""));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1100"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1202"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx1210"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("foogfx1201"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("agfx1201"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx12010"));
  CHECK_FALSE(GcnArchNameIsGfx12PrefillWmma("gfx120"));
  static_assert(GcnArchNameIsGfx12PrefillWmma("gfx1201:xnack-"));
  static_assert(!GcnArchNameIsGfx12PrefillWmma("foogfx1201"));
  static_assert(!GcnArchNameIsGfx12PrefillWmma("gfx12010"));
}

TEST_CASE("the parse is constexpr, so a wrong answer is a compile error") {
  // Not decoration: it is what lets the capability be asserted without a device.
  static_assert(CapabilityFromGcnArch("gfx1100")->first == 11);
  static_assert(CapabilityFromGcnArch("gfx1151")->second == 5);
  static_assert(!CapabilityFromGcnArch("sm_121a").has_value());
  CHECK(true);
}
