// SPEC-DFLASH2 #2202. The DFlash block attention's query-block grid, gated on
// the CPU because the property that matters — every query row owned by exactly
// one block — is invisible from inside a CUDA kernel and a wrong mapping shows
// up as silently missing output rows rather than as a crash.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vt/dflash_attn_grid.h"

namespace {

// Every query row must be covered EXACTLY once by the live blocks, and no live
// block may cross a request boundary — which is the whole point of the change.
void CheckCoverage(const std::vector<int32_t>& qcu, int64_t block_rows) {
  const int num_reqs = static_cast<int>(qcu.size()) - 1;
  const int64_t tiles = vt::DFlashQueryTilesPerReq(qcu.data(), num_reqs, block_rows);
  const int64_t total = qcu.back();
  std::vector<int> covered(static_cast<size_t>(total), 0);
  const int64_t blocks = static_cast<int64_t>(num_reqs) * tiles;
  for (int64_t bx = 0; bx < blocks; ++bx) {
    const vt::DFlashQueryBlock b =
        vt::DFlashResolveQueryBlock(qcu.data(), num_reqs, tiles, block_rows, bx);
    if (!b.live) continue;
    INFO("bx=", bx, " req=", b.req, " [", b.qblk, ",", b.qend, ")");
    CHECK(b.qblk >= qcu[static_cast<size_t>(b.req)]);
    CHECK(b.qend <= qcu[static_cast<size_t>(b.req) + 1]);
    CHECK(b.qend > b.qblk);
    CHECK(b.qend - b.qblk <= block_rows);
    for (int64_t i = b.qblk; i < b.qend; ++i) covered[static_cast<size_t>(i)] += 1;
  }
  for (int64_t i = 0; i < total; ++i) {
    INFO("query row ", i);
    CHECK(covered[static_cast<size_t>(i)] == 1);
  }
}

}  // namespace

TEST_CASE("dflash query grid: the production c=8 shape, 8 requests of 1+k=9") {
  // Tq = 72 in ONE tile per request, against the old global tiling's two blocks
  // where block 0 spanned every request.
  std::vector<int32_t> qcu{0, 9, 18, 27, 36, 45, 54, 63, 72};
  CHECK(vt::DFlashQueryTilesPerReq(qcu.data(), 8, vt::kDFlashQueryBlockRows) == 1);
  CheckCoverage(qcu, vt::kDFlashQueryBlockRows);
}

TEST_CASE("dflash query grid: a single request still works, and is one block") {
  std::vector<int32_t> qcu{0, 9};
  CHECK(vt::DFlashQueryTilesPerReq(qcu.data(), 1, vt::kDFlashQueryBlockRows) == 1);
  CheckCoverage(qcu, vt::kDFlashQueryBlockRows);
}

TEST_CASE("dflash query grid: RAGGED requests pad to the widest, and the padding is dead") {
  // 5, 64 and 70 rows: the widest needs two tiles, so the two narrow requests
  // each get a second tile that owns nothing. Those blocks must report !live
  // rather than aliasing another request's rows.
  std::vector<int32_t> qcu{0, 5, 69, 139};
  const int64_t tiles = vt::DFlashQueryTilesPerReq(qcu.data(), 3, vt::kDFlashQueryBlockRows);
  CHECK(tiles == 2);
  CheckCoverage(qcu, vt::kDFlashQueryBlockRows);
  // The dead tile of request 0 is block 1.
  const vt::DFlashQueryBlock dead =
      vt::DFlashResolveQueryBlock(qcu.data(), 3, tiles, vt::kDFlashQueryBlockRows, 1);
  CHECK_FALSE(dead.live);
}

TEST_CASE("dflash query grid: a request WIDER than one block splits across tiles") {
  // A prefill-shaped draft block: 200 rows needs four tiles of 64.
  std::vector<int32_t> qcu{0, 200};
  CHECK(vt::DFlashQueryTilesPerReq(qcu.data(), 1, vt::kDFlashQueryBlockRows) == 4);
  CheckCoverage(qcu, vt::kDFlashQueryBlockRows);
}

TEST_CASE("dflash query grid: an EMPTY request is skipped without stealing rows") {
  // ctx_cu/cu can carry a zero-width request; it must own no rows and must not
  // shift its neighbours.
  std::vector<int32_t> qcu{0, 9, 9, 18};
  CheckCoverage(qcu, vt::kDFlashQueryBlockRows);
  const int64_t tiles = vt::DFlashQueryTilesPerReq(qcu.data(), 3, vt::kDFlashQueryBlockRows);
  const vt::DFlashQueryBlock empty =
      vt::DFlashResolveQueryBlock(qcu.data(), 3, tiles, vt::kDFlashQueryBlockRows, 1);
  CHECK_FALSE(empty.live);
}

TEST_CASE("dflash query grid: no rows at all") {
  std::vector<int32_t> qcu{0, 0};
  CHECK(vt::DFlashQueryTilesPerReq(qcu.data(), 1, vt::kDFlashQueryBlockRows) == 0);
}
