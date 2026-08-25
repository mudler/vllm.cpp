// vllm.cpp — the HOST-SIDE contract of the PER-TENSOR FP8 CUTLASS GEMM's sm120
// tile ladder (src/vt/cuda/fp8_per_tensor_dispatch.h).
//
// PERF-FP8-SMALL-M-DISPATCH (.agents/specs/perf-fp8-small-m-dispatch.md),
// issue #1866, owning row `KERNEL-GEMM-FP8`. Pinned oracle: vLLM
// `5559679229bc961848b121ccdeaa8fa5d79bec98`; every ladder value below was read
// at that revision from
// `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//  scaled_mm_sm120_fp8_dispatch.cuh:155-179`.
//
// WHY THIS FILE EXISTS AT ALL. A wrong tile is a SLOW answer, not a wrong one.
// Every token gate in this tree stayed green while the per-tensor fp8 GEMM ran
// a 64-row tile for a 1-row decode step, because there is nothing about the
// value of the output that a 64-row tile gets wrong. So the ladder cannot be
// gated by any of the correctness suites; it has to be gated BY VALUE, on the
// tier that has a test runner, which is this one. The device-tier case that
// asserts the three small-M rungs agree bit-for-bit with the M64 rung needs a
// GPU and is OWED (spec `## Owed`).
//
// The four cases:
//   G1  the ladder, at every upstream boundary and one value either side.
//   G2  the rollback flag's parse table, including the mangled spellings.
//   G3  the rollback arm reproduces the PRE-#1866 two-way ladder exactly.
//   G4  the names, and the dispatch counters' accounting.
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vt/cuda/fp8_per_tensor_dispatch.h"

using vt::cuda::Fp8CutlassSmallMFlagIsOn;
using vt::cuda::Fp8PerTensorConfig;
using vt::cuda::Fp8PerTensorConfigName;
using vt::cuda::Fp8PerTensorCountDispatch;
using vt::cuda::Fp8PerTensorDispatchCount;
using vt::cuda::Fp8PerTensorResetDispatchCounts;
using vt::cuda::Fp8Sm120ConfigForM;

namespace {

// The upstream ladder, transcribed once from
// `cutlass_gemm_sm120_fp8_dispatch` (:155-179):
//
//   if (M <= 16)  -> Cutlass3xGemmM16
//   if (M <= 32)  -> Cutlass3xGemmM32
//   if (M <= 256) -> Cutlass3xGemmM64
//   else          -> Cutlass3xGemmDefault
//
// Written as an explicit TABLE rather than as a second `if`-chain on purpose:
// a chain here would be the same program as the one under test, and a test
// that re-derives its expectation from the implementation's own shape checks
// that the code is self-consistent, not that it matches upstream.
struct LadderCase {
  int64_t m;
  Fp8PerTensorConfig expect;
  const char* why;
};

const std::vector<LadderCase>& UpstreamLadder() {
  static const std::vector<LadderCase> cases = {
      {1, Fp8PerTensorConfig::kM16, "batch-1 decode, the single most common M in the tree"},
      {9, Fp8PerTensorConfig::kM16, "the DFlash2 spec-verify's 9 query rows (#1857)"},
      {16, Fp8PerTensorConfig::kM16, "the M16 boundary, inclusive"},
      {17, Fp8PerTensorConfig::kM32, "one past the M16 boundary"},
      {31, Fp8PerTensorConfig::kM32, "inside the M32 rung"},
      {32, Fp8PerTensorConfig::kM32, "the M32 boundary, inclusive"},
      {33, Fp8PerTensorConfig::kM64, "one past the M32 boundary"},
      {64, Fp8PerTensorConfig::kM64, "the config's namesake M, which is NOT its boundary"},
      {256, Fp8PerTensorConfig::kM64, "the M64 boundary, inclusive"},
      {257, Fp8PerTensorConfig::kDefault, "one past the M64 boundary"},
      {4096, Fp8PerTensorConfig::kDefault, "a real prefill T"},
  };
  return cases;
}

}  // namespace

// --- G1: the ladder, by value, at every boundary ---------------------------
TEST_CASE("Fp8Sm120ConfigForM mirrors upstream's four-way sm120 fp8 M ladder") {
  for (const LadderCase& c : UpstreamLadder()) {
    CAPTURE(c.m);
    CAPTURE(std::string(c.why));
    CHECK(Fp8Sm120ConfigForM(c.m, /*small_m=*/true) == c.expect);
  }
}

// The two rungs this tree dropped are exactly the decode regime, so they get
// their own case: a reader who deletes the M16 arm and reruns must see a
// failure that NAMES decode, not one that names "case 3 of 11".
TEST_CASE("the two rungs this tree dropped are the ones decode lands on") {
  // #1857's c1 profile: the target verify batches 9 query rows and the draft
  // runs at M=1..K. Nothing in that range may reach the 64-row tile.
  for (int64_t m = 1; m <= 16; ++m) {
    CAPTURE(m);
    CHECK(Fp8Sm120ConfigForM(m, /*small_m=*/true) == Fp8PerTensorConfig::kM16);
  }
  for (int64_t m = 17; m <= 32; ++m) {
    CAPTURE(m);
    CHECK(Fp8Sm120ConfigForM(m, /*small_m=*/true) == Fp8PerTensorConfig::kM32);
  }
}

// Degenerate and defensive inputs. `m <= 0` cannot reach the kernel (the
// caller returns early on m == 0 and a negative extent is impossible), but the
// ladder must still be TOTAL: the first comparison is `<=`, so a zero or
// negative M lands on the smallest rung rather than falling through to a tile
// sized for 4096 rows.
TEST_CASE("Fp8Sm120ConfigForM is total: a non-positive M takes the smallest rung") {
  CHECK(Fp8Sm120ConfigForM(0, /*small_m=*/true) == Fp8PerTensorConfig::kM16);
  CHECK(Fp8Sm120ConfigForM(-1, /*small_m=*/true) == Fp8PerTensorConfig::kM16);
  CHECK(Fp8Sm120ConfigForM(0, /*small_m=*/false) == Fp8PerTensorConfig::kM64);
}

// --- G2: the rollback flag's parse -----------------------------------------
TEST_CASE("VT_FP8_CUTLASS_SMALL_M is ON by default; OFF only for exactly \"0\"") {
  // ON: unset, and every spelling that is not exactly "0".
  CHECK(Fp8CutlassSmallMFlagIsOn(nullptr));
  CHECK(Fp8CutlassSmallMFlagIsOn(""));
  CHECK(Fp8CutlassSmallMFlagIsOn("1"));
  CHECK(Fp8CutlassSmallMFlagIsOn("on"));
  // A MANGLED rollback must fail toward the mirror, never away from it. "00",
  // " 0" and "false" are the spellings a hurried operator reaches for, and
  // each of them leaving the flag ON is the same polarity #1843 chose for
  // VT_FP8_PLAN_CACHE after a typo could have resurrected a capture bug.
  CHECK(Fp8CutlassSmallMFlagIsOn("00"));
  CHECK(Fp8CutlassSmallMFlagIsOn(" 0"));
  CHECK(Fp8CutlassSmallMFlagIsOn("0 "));
  CHECK(Fp8CutlassSmallMFlagIsOn("false"));

  // OFF: exactly one string.
  CHECK_FALSE(Fp8CutlassSmallMFlagIsOn("0"));
}

// --- G3: the rollback arm IS the pre-#1866 ladder ---------------------------
TEST_CASE("VT_FP8_CUTLASS_SMALL_M=0 reproduces the pre-#1866 two-way ladder") {
  // What this tree shipped before: `if (m <= 256) M64; else default;`. The A/B
  // is only an A/B if the OFF arm is the old program, so the OFF arm is pinned
  // by value over the whole small-M range, not merely asserted to differ.
  for (int64_t m : {int64_t{0}, int64_t{1}, int64_t{9}, int64_t{16}, int64_t{17}, int64_t{32},
                    int64_t{33}, int64_t{64}, int64_t{256}}) {
    CAPTURE(m);
    CHECK(Fp8Sm120ConfigForM(m, /*small_m=*/false) == Fp8PerTensorConfig::kM64);
  }
  CHECK(Fp8Sm120ConfigForM(257, /*small_m=*/false) == Fp8PerTensorConfig::kDefault);
  CHECK(Fp8Sm120ConfigForM(4096, /*small_m=*/false) == Fp8PerTensorConfig::kDefault);

  // And the flag must actually MOVE something, or "0" would be a no-op switch
  // wearing a rollback's name.
  CHECK(Fp8Sm120ConfigForM(9, /*small_m=*/true) != Fp8Sm120ConfigForM(9, /*small_m=*/false));
  CHECK(Fp8Sm120ConfigForM(20, /*small_m=*/true) != Fp8Sm120ConfigForM(20, /*small_m=*/false));
  // Above the small-M rungs the two arms must be IDENTICAL: the flag is scoped
  // to the two ported rungs and may not perturb prefill.
  CHECK(Fp8Sm120ConfigForM(64, /*small_m=*/true) == Fp8Sm120ConfigForM(64, /*small_m=*/false));
  CHECK(Fp8Sm120ConfigForM(4096, /*small_m=*/true) ==
        Fp8Sm120ConfigForM(4096, /*small_m=*/false));
}

// --- G4: names and counters -------------------------------------------------
TEST_CASE("every config has a distinct, tile-naming diagnostic string") {
  const std::vector<Fp8PerTensorConfig> all = {
      Fp8PerTensorConfig::kM16, Fp8PerTensorConfig::kM32, Fp8PerTensorConfig::kM64,
      Fp8PerTensorConfig::kDefault};
  std::vector<std::string> names;
  for (Fp8PerTensorConfig c : all) {
    std::string name = Fp8PerTensorConfigName(c);
    CHECK(name != "unknown");
    // The name carries the TILE, because the whole point of the diagnostic is
    // to be comparable against an nsys kernel name.
    CHECK(name.find("x128") != std::string::npos);
    names.push_back(name);
  }
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = i + 1; j < names.size(); ++j) {
      CAPTURE(names[i]);
      CAPTURE(names[j]);
      CHECK(names[i] != names[j]);
    }
  }
  CHECK(std::string(Fp8PerTensorConfigName(Fp8PerTensorConfig::kCount)) == "unknown");
}

TEST_CASE("the dispatch counters attribute each call to exactly one config") {
  Fp8PerTensorResetDispatchCounts();
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM16) == 0);

  Fp8PerTensorCountDispatch(Fp8Sm120ConfigForM(1, /*small_m=*/true));
  Fp8PerTensorCountDispatch(Fp8Sm120ConfigForM(9, /*small_m=*/true));
  Fp8PerTensorCountDispatch(Fp8Sm120ConfigForM(20, /*small_m=*/true));
  Fp8PerTensorCountDispatch(Fp8Sm120ConfigForM(4096, /*small_m=*/true));

  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM16) == 2);
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM32) == 1);
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM64) == 0);
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kDefault) == 1);

  // An out-of-range value is dropped rather than corrupting a neighbour's slot.
  Fp8PerTensorCountDispatch(Fp8PerTensorConfig::kCount);
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kCount) == 0);
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM16) == 2);

  Fp8PerTensorResetDispatchCounts();
  CHECK(Fp8PerTensorDispatchCount(Fp8PerTensorConfig::kM32) == 0);
}
