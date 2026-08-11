// vllm.cpp original. Portable contract for the opt-in fused GDN decode BV16
// value tile; the CUDA recurrence remains in cuda_gdn.cu.
#include <doctest/doctest.h>

#include <initializer_list>

#include "vt/cuda/gdn_decode_fused.h"

using vt::cuda::DispatchGdnDecodeValueTile;
using vt::cuda::DispatchGdnDecodeStateStorage;
using vt::cuda::GdnDecodeLaunchContractFor;
using vt::cuda::GdnDecodeRegisterLogicalColumn;
using vt::cuda::GdnDecodeRegisterSharedColumn;
using vt::cuda::GdnDecodeRegstateFlagIsOn;
using vt::cuda::GdnDecodeSharedColumn;
using vt::cuda::GdnDecodeStateStride;
using vt::cuda::GdnDecodeValueTile;
using vt::cuda::GdnDecodeValueTileFromEnv;
using vt::cuda::GdnDecodeSwizzleFlagIsOn;

static_assert(GdnDecodeRegisterSharedColumn(0, 0) == 0);
static_assert(GdnDecodeRegisterSharedColumn(7, 15) == 127);

TEST_CASE("VT_GDN_DECODE_BV selects only exact BV16") {
  CHECK(GdnDecodeValueTileFromEnv("8") == GdnDecodeValueTile::kBv32);
  CHECK(GdnDecodeValueTileFromEnv("16") == GdnDecodeValueTile::kBv16);
  CHECK(GdnDecodeValueTileFromEnv("24") == GdnDecodeValueTile::kBv32);
  CHECK(GdnDecodeValueTileFromEnv("32") == GdnDecodeValueTile::kBv32);
  constexpr const char* invalid[] = {
      nullptr, "",     "0",   "08",  "016", "024", "032", "+8",
      "-8",    " 8",  "8 ",  "8x",  "16x", "24x", "32x", "64",
      "on",    "999999999999999999999999999999999999999999999999999999999999"};
  for (const char* value : invalid) {
    CAPTURE(value == nullptr ? "<unset>" : value);
    CHECK(GdnDecodeValueTileFromEnv(value) == GdnDecodeValueTile::kBv32);
  }
}

TEST_CASE("fused GDN decode value-tile geometry preserves row groups") {
  const auto bv32 = GdnDecodeLaunchContractFor(nullptr, 128, 128, 8);
  const auto bv16 = GdnDecodeLaunchContractFor("16", 128, 128, 8);
  CHECK(bv32.value_tile == 32);
  CHECK(bv32.value_tiles == 4);
  CHECK(bv32.lanes_per_row == 8);
  CHECK(bv32.block_threads == 256);
  CHECK(bv32.shared_bytes == 17536);
  CHECK(bv16.value_tile == 16);
  CHECK(bv16.value_tiles == 8);
  CHECK(bv16.lanes_per_row == 8);
  CHECK(bv16.block_threads == 128);
  CHECK(bv16.shared_bytes == 9280);

  // Removed sweep values preserve the shipped BV32 launch contract.
  for (const char* removed : {"8", "24"}) {
    const auto fallback = GdnDecodeLaunchContractFor(removed, 128, 128, 8);
    CHECK(fallback.selected_tile == GdnDecodeValueTile::kBv32);
    CHECK(fallback.value_tile == 32);
    CHECK(fallback.value_tiles == 4);
    CHECK(fallback.block_threads == 256);
    CHECK(fallback.shared_bytes == 17536);
  }

  // A partial last tile is retained rather than dropped.
  CHECK(GdnDecodeLaunchContractFor(nullptr, 33, 65, 8).value_tiles == 2);
  CHECK(GdnDecodeLaunchContractFor("16", 33, 65, 8).value_tiles == 3);

  // Existing corner contract: Dv<32 forces one Dk lane per value row.
  const auto small32 = GdnDecodeLaunchContractFor(nullptr, 17, 65, 8);
  const auto small16 = GdnDecodeLaunchContractFor("16", 17, 65, 8);
  CHECK(small32.lanes_per_row == 1);
  CHECK(small32.value_tile == 17);
  CHECK(small32.value_tiles == 1);
  CHECK(small32.block_threads == 17);
  CHECK(small16.lanes_per_row == 1);
  CHECK(small16.value_tile == 16);
  CHECK(small16.value_tiles == 2);
  CHECK(small16.block_threads == 16);
  for (const auto zero : {GdnDecodeLaunchContractFor(nullptr, 0, 128, 8),
                          GdnDecodeLaunchContractFor("16", 128, 0, 8)}) {
    CHECK_FALSE(zero.should_launch);
    CHECK(zero.value_tiles == 0);
    CHECK(zero.block_threads == 0);
    CHECK(zero.shared_bytes == 0);
  }
}

TEST_CASE("fused GDN decode shared dispatcher invokes exactly one tile arm") {
  auto select = [](const char* value) {
    int bv16_calls = 0;
    int bv32_calls = 0;
    const auto selected = DispatchGdnDecodeValueTile(
        value, 128, 128, 8,
        [&](const auto& contract) {
          ++bv16_calls;
          CHECK(contract.value_tile == 16);
          return 16;
        },
        [&](const auto& contract) {
          ++bv32_calls;
          CHECK(contract.value_tile == 32);
          return 32;
        });
    CHECK(bv16_calls + bv32_calls == 1);
    return selected;
  };
  CHECK(select("8") == 32);
  CHECK(select("16") == 16);
  CHECK(select("24") == 32);
  CHECK(select("32") == 32);
  CHECK(select(nullptr) == 32);
  CHECK(select("16x") == 32);
  CHECK(select("24x") == 32);
}

TEST_CASE("VT_GDN_DECODE_SWIZZLE selects only exact one") {
  CHECK(GdnDecodeSwizzleFlagIsOn("1"));
  constexpr const char* invalid[] = {
      nullptr, "",   "0",  "01", "1 ", " 1", "+1", "1x",
      "2",    "on", "true", "999999999999999999999999999999999999"};
  for (const char* value : invalid) {
    CAPTURE(value == nullptr ? "<unset>" : value);
    CHECK_FALSE(GdnDecodeSwizzleFlagIsOn(value));
  }
}

TEST_CASE("VT_GDN_DECODE_REGSTATE selects only exact one") {
  CHECK(GdnDecodeRegstateFlagIsOn("1"));
  constexpr const char* invalid[] = {
      nullptr, "",   "0",  "01", "1 ", " 1", "+1", "1x",
      "2",    "on", "true", "yes", "999999999999999999999999999999999999"};
  for (const char* value : invalid) {
    CAPTURE(value == nullptr ? "<unset>" : value);
    CHECK_FALSE(GdnDecodeRegstateFlagIsOn(value));
  }
}

TEST_CASE("fused GDN decode swizzle contract is production-shape only") {
  const auto production =
      GdnDecodeLaunchContractFor("16", "1", 128, 128, 8);
  CHECK(production.swizzled);
  CHECK(production.value_tile == 16);
  CHECK(production.value_tiles == 8);
  CHECK(production.lanes_per_row == 8);
  CHECK(production.block_threads == 128);
  CHECK(production.shared_bytes == 9728);
  CHECK(GdnDecodeStateStride(true, 128, 8) == 136);
  CHECK(GdnDecodeStateStride(false, 128, 8) == 129);

  struct FallbackCase {
    vt::cuda::GdnDecodeLaunchContract contract;
    size_t shared_bytes;
  };
  for (const auto& fallback : {
           FallbackCase{GdnDecodeLaunchContractFor("16", nullptr, 128, 128, 8), 9280},
           FallbackCase{GdnDecodeLaunchContractFor("16", "0", 128, 128, 8), 9280},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", 127, 128, 8), 9280},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", 128, 127, 8), 9208},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", 128, 128, 4), 9280},
           FallbackCase{GdnDecodeLaunchContractFor(nullptr, "1", 128, 128, 8), 17536},
           FallbackCase{GdnDecodeLaunchContractFor("32", "1", 128, 128, 8), 17536},
       }) {
    CHECK_FALSE(fallback.contract.swizzled);
    CHECK(fallback.contract.shared_bytes == fallback.shared_bytes);
  }
}

TEST_CASE("fused GDN decode register-state contract is production-shape only") {
  const auto production =
      GdnDecodeLaunchContractFor("16", "1", "1", 128, 128, 8);
  CHECK(production.regstate);
  CHECK(production.swizzled);
  CHECK(production.value_tile == 16);
  CHECK(production.value_tiles == 8);
  CHECK(production.lanes_per_row == 8);
  CHECK(production.block_threads == 128);
  CHECK(production.shared_bytes == 9728);

  struct FallbackCase {
    vt::cuda::GdnDecodeLaunchContract contract;
    const char* failed_predicate;
  };
  for (const auto& fallback : {
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", nullptr, 128, 128, 8),
                        "regstate unset"},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", "0", 128, 128, 8),
                        "regstate disabled"},
           FallbackCase{GdnDecodeLaunchContractFor("32", "1", "1", 128, 128, 8),
                        "BV16"},
           FallbackCase{GdnDecodeLaunchContractFor("16", "0", "1", 128, 128, 8),
                        "swizzle"},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", "1", 127, 128, 8),
                        "Dv"},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", "1", 128, 127, 8),
                        "Dk"},
           FallbackCase{GdnDecodeLaunchContractFor("16", "1", "1", 128, 128, 4),
                        "NW"},
       }) {
    CAPTURE(fallback.failed_predicate);
    CHECK_FALSE(fallback.contract.regstate);
  }
}

TEST_CASE("fused GDN decode register-state dispatcher invokes exactly one storage arm") {
  auto select = [](const char* bv, const char* swizzle, const char* regstate,
                   int64_t dv, int64_t dk, int nw) {
    const auto contract =
        GdnDecodeLaunchContractFor(bv, swizzle, regstate, dv, dk, nw);
    int shared_calls = 0;
    int register_calls = 0;
    const auto selected = DispatchGdnDecodeStateStorage(
        contract,
        [&](const auto& resolved) {
          ++shared_calls;
          CHECK_FALSE(resolved.regstate);
          return 0;
        },
        [&](const auto& resolved) {
          ++register_calls;
          CHECK(resolved.regstate);
          return 1;
        });
    CHECK(shared_calls + register_calls == 1);
    return selected;
  };

  CHECK(select("16", "1", "1", 128, 128, 8) == 1);
  CHECK(select("16", "1", nullptr, 128, 128, 8) == 0);
  CHECK(select("16", "0", "1", 128, 128, 8) == 0);
  CHECK(select("32", "1", "1", 128, 128, 8) == 0);
  CHECK(select("16", "1", "1", 127, 128, 8) == 0);
  CHECK(select("16", "1", "1", 128, 127, 8) == 0);
  CHECK(select("16", "1", "1", 128, 128, 4) == 0);
}

TEST_CASE("fused GDN decode swizzle is a 128-column bijection") {
  CHECK(GdnDecodeSharedColumn(true, 0, 128, 8) == 0);
  CHECK(GdnDecodeSharedColumn(true, 15, 128, 8) == 120);
  CHECK(GdnDecodeSharedColumn(true, 16, 128, 8) == 1);
  CHECK(GdnDecodeSharedColumn(true, 127, 128, 8) == 127);
  CHECK(GdnDecodeSharedColumn(false, 127, 128, 8) == 127);

  bool seen[128] = {};
  for (int c = 0; c < 128; ++c) {
    const int sw = static_cast<int>(GdnDecodeSharedColumn(true, c, 128, 8));
    REQUIRE(sw >= 0);
    REQUIRE(sw < 128);
    CHECK_FALSE(seen[sw]);
    seen[sw] = true;
  }
  for (int wk = 0; wk < 8; ++wk)
    for (int j = 0; j < 16; ++j)
      CHECK(GdnDecodeSharedColumn(true, wk * 16 + j, 128, 8) ==
            j * 8 + wk);
}

TEST_CASE("fused GDN decode register slots map bijectively to logical and shared columns") {
  bool logical_seen[128] = {};
  bool shared_seen[128] = {};
  for (int wk = 0; wk < 8; ++wk) {
    for (int j = 0; j < 16; ++j) {
      const int logical =
          static_cast<int>(GdnDecodeRegisterLogicalColumn(wk, j));
      const int shared =
          static_cast<int>(GdnDecodeRegisterSharedColumn(wk, j));
      CHECK(logical == wk * 16 + j);
      CHECK(shared == j * 8 + wk);
      REQUIRE(logical >= 0);
      REQUIRE(logical < 128);
      REQUIRE(shared >= 0);
      REQUIRE(shared < 128);
      CHECK_FALSE(logical_seen[logical]);
      CHECK_FALSE(shared_seen[shared]);
      logical_seen[logical] = true;
      shared_seen[shared] = true;
    }
  }
  for (int c = 0; c < 128; ++c) {
    CHECK(logical_seen[c]);
    CHECK(shared_seen[c]);
  }
}
