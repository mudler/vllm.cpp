// vllm.cpp original. TEST-ONLY scoped pin of the VT_MOE_ROUTER_WARP selector
// (spec: .agents/specs/moe-router-topk-single-warp.md, issue #378).
//
// WHY THIS EXISTS. vt::cuda::MoeRouterWarpEnabled() (src/vt/cuda/cuda_moe.cu)
// does a FRESH getenv on every launch, so which router kernel
// `vt::MoeRouterTopK` runs is decided by the AMBIENT environment of whoever
// started ctest. A device parallel-vs-serial sweep that neither sets nor
// asserts that variable therefore proves nothing about the warp kernel: exported
// `VT_MOE_ROUTER_WARP=0` — which spec §9 gates 6/7 tell the operator to export
// for the A/B — silently degenerates the sweep to block-vs-serial, which has
// been green since 6a8c5cf9, and it reports the IDENTICAL case and assertion
// counts. Green then cannot distinguish "the warp kernel is byte-exact" from
// "the warp kernel never ran".
//
// So the toggle is PINNED around the sweep, both ways, and the pinned state is
// ASSERTED rather than assumed. Same scoped setenv/restore idiom as
// tests/vt/test_ops_nvfp4_fp4.cpp:1268-1287 and tests/vt/test_ops_gdn.cpp:1350-1370.
//
// It is a shared header, not a per-file copy, on purpose: the device gate that
// needs the pin cannot be compiled without nvcc and a GPU, while the portable
// companion tests/vt/test_moe_router_warp_map.cpp runs anywhere — so the host
// test executes the very same pin the device gate uses, instead of a
// character-identical copy of it that nothing checks.
#pragma once

#include <cstdlib>
#include <string>

#include "vt/cuda/moe_router_warp.h"

namespace vt_test {

// RAII: pin VT_MOE_ROUTER_WARP to `value` (nullptr = unset it) and restore
// whatever was there before — including its absence — on scope exit.
class ScopedMoeRouterWarp {
 public:
  static constexpr const char* kName = "VT_MOE_ROUTER_WARP";

  explicit ScopedMoeRouterWarp(const char* value) {
    const char* current = std::getenv(kName);
    had_ = current != nullptr;
    if (had_) saved_ = current;
    Apply(value);
  }
  ~ScopedMoeRouterWarp() { Apply(had_ ? saved_.c_str() : nullptr); }

  ScopedMoeRouterWarp(const ScopedMoeRouterWarp&) = delete;
  ScopedMoeRouterWarp& operator=(const ScopedMoeRouterWarp&) = delete;

  // Exactly what the launcher's per-call getenv will resolve to RIGHT NOW:
  // MoeRouterWarpEnabled() is MoeRouterWarpFlagIsOn(getenv(kName)) and nothing
  // else (cuda_moe.cu). true  -> vt::MoeRouterTopK dispatches the WARP kernel
  // for E in {32,64,128,256}; false -> it falls through to the block kernel.
  static bool EffectiveFlag() {
    return vt::cuda::MoeRouterWarpFlagIsOn(std::getenv(kName));
  }

 private:
  static void Apply(const char* value) {
    if (value != nullptr)
      setenv(kName, value, 1);
    else
      unsetenv(kName);
  }

  std::string saved_;
  bool had_ = false;
};

}  // namespace vt_test
