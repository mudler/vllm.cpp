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
    "VT_GDN_PACKED_REG_TILE defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: register-resident tiling is the new default.
  CHECK(GdnPackedRegTileFlagIsOn(nullptr));
  // Explicit enable spellings stay ON.
  CHECK(GdnPackedRegTileFlagIsOn("1"));
  CHECK(GdnPackedRegTileFlagIsOn(""));       // empty string -> ON (first char is not '0')
  CHECK(GdnPackedRegTileFlagIsOn("on"));
  CHECK(GdnPackedRegTileFlagIsOn("true"));
  CHECK(GdnPackedRegTileFlagIsOn("2"));
  // Rollback: any value whose FIRST character is '0' selects the legacy kernel
  // (mirrors the house default-ON GDN A/B convention GdnTritonEnvOn).
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("0"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("0 "));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("00"));
  CHECK_FALSE(GdnPackedRegTileFlagIsOn("0abc"));
  // A '0' that is NOT the first character does not roll back.
  CHECK(GdnPackedRegTileFlagIsOn("10"));
  CHECK(GdnPackedRegTileFlagIsOn(" 0"));
}
