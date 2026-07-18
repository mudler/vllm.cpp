// vllm.cpp original (vt runtime env-flag plumbing); the kernel it selects is a
// GDN decode causal_conv1d state-update kernel whose output (both `out` and the
// rolled conv_state) is BIT-IDENTICAL (0-ulp) to the shipped
// CausalConv1dUpdateKernel (cuda_gdn.cu).
//
// CPU-tier contract for the env-flag plumbing that selects the fast conv-update
// decode kernel (src/vt/cuda/conv_update_fast.h): the VT_CONV_UPDATE_FAST flag
// predicate. The kernel itself is CUDA-only and lives in cuda_gdn.cu
// (CausalConv1dUpdateFastKernel); its BIT-EXACT (0-ulp) parity vs
// CausalConv1dUpdateKernel and the width-specialized 2D-grid launch guard are
// DGX-gated CUDA checks (tests/vt/test_ops_gdn.cpp). This suite pins the portable
// default-ON / '0'-rollback parse so the contract is regression-covered on every
// platform, not just DGX. (The default was flipped ON after the isolated microbench
// showed 1.92x at the 27B c16 decode shape; bit-identical ⇒ never-slower.)
#include <doctest/doctest.h>

#include "vt/cuda/conv_update_fast.h"

using vt::cuda::ConvUpdateFastFlagIsOn;

TEST_CASE("VT_CONV_UPDATE_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: CausalConv1dUpdateFastKernel's output is BIT-IDENTICAL
  // (0-ulp) to the shipped CausalConv1dUpdateKernel by construction (same bias
  // init, same left-to-right conv accumulation, same silu/identity epilogue +
  // round-to-store, same rolled conv_state bytes; only the 2D-grid index math and
  // the register-cached state row differ) AND 1.92x faster in isolation, so it is
  // the production default.
  CHECK(ConvUpdateFastFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON (the default).
  CHECK(ConvUpdateFastFlagIsOn(""));
  CHECK(ConvUpdateFastFlagIsOn("1"));
  CHECK(ConvUpdateFastFlagIsOn("off"));
  CHECK(ConvUpdateFastFlagIsOn("false"));
  CHECK(ConvUpdateFastFlagIsOn("2"));
  CHECK(ConvUpdateFastFlagIsOn(" 0"));  // leading space, not '0'
  // Roll back to the shipped CausalConv1dUpdateKernel: FIRST character '0'.
  CHECK_FALSE(ConvUpdateFastFlagIsOn("0"));
  CHECK_FALSE(ConvUpdateFastFlagIsOn("0abc"));
  CHECK_FALSE(ConvUpdateFastFlagIsOn("00"));
}
