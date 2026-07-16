// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CPU-tier contract for the env-flag plumbing that selects the register-resident
// packed-decode tiling (src/vt/cuda/gdn_packed_reg_tile.h): the
// VT_GDN_PACKED_REG_TILE flag predicate. The register-resident kernel itself is
// CUDA-only and lives in cuda_gdn.cu (GdnPackedDecodeRegTileKernel); its
// bit-exactness and the rollback-selects-legacy launch check are DGX-gated CUDA
// cases in tests/vt/test_ops_gdn.cpp. This suite pins the portable default-ON /
// '0'-off parse so the rollback contract is regression-covered on every
// platform, not just DGX.
#include <doctest/doctest.h>

#include "vt/cuda/gdn_packed_reg_tile.h"

using vt::cuda::GdnPackedRegTileFlagIsOn;

TEST_CASE(
    "VT_GDN_PACKED_REG_TILE defaults OFF; only a '1'-leading value opts in") {
  // Default (unset) is OFF: the legacy shared-memory kernel ships after the
  // 2026-07-16 DGX proof failed (oracle boundary FAIL; c16 -12%).
  CHECK_FALSE(GdnPackedRegTileFlagIsOn(nullptr));
  // Non-'1'-leading values stay OFF.
  CHECK_FALSE(GdnPackedRegTileFlagIsOn(""));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("on"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("true"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("2"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("0"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn(" 1"));
  // Experimental opt-in: FIRST character '1'.
  CHECK(GdnPackedRegTileFlagIsOn("1"));
  CHECK(GdnPackedRegTileFlagIsOn("10"));
  CHECK(GdnPackedRegTileFlagIsOn("1abc"));
}
