// vllm.cpp — the HOST-SIDE contract of the block-scaled FP8 CUTLASS GEMM's CUDA
// arm (src/vt/cuda/fp8_block_scaled_dispatch.h).
//
// VT-MATMUL-FP8-BLOCK-CUDA (.agents/specs/vt-matmul-fp8-block-cuda.md), issue
// #1189 milestone M5. Pinned oracle: vLLM
// 5559679229bc961848b121ccdeaa8fa5d79bec98, asserted as the local checkout's
// HEAD before every anchor below was read.
//
// WHAT THIS FILE IS FOR, AND WHAT IT IS NOT. The load-bearing gate of M5 is a
// value comparison of the CUDA kernel against `vt::MatmulFp8BlockScaled`'s CPU
// reference arm, and it needs a device: it lives in
// `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` and it is OWED, not done. This
// file is the half that can be gated WITHOUT hardware, and it is not a
// consolation prize — it holds every decision the kernel makes on the host,
// which is every part of the arm that can be wrong in a way a successful
// compile cannot see:
//
//   G1  the tile-config heuristic, at every boundary, BY VALUE.
//   G3  the two scale-layout index formulas cutlass DEDUCES from the problem
//       shape, against a hand-derived table.
//   G4  the refusals, each by name and by message, and — the other half, which
//       is the one a refusal test usually forgets — the shapes that must NOT be
//       refused, including the ragged N=576 that upstream's own CUTLASS test
//       runs.
//   G5  the dispatch counter's accounting.
//
// (G2 and G6-G9 are the device-tier cases in the sibling file. The numbering is
// the spec's and is deliberately not compacted, so a reader comparing the two
// files against `## Tests` finds every letter accounted for.)
//
// THE EXPECTATIONS ARE WRITTEN OUT, NOT COMPUTED. #911 burned this repository
// with a checker that read its expectation out of the thing it was checking and
// reported 27/27 FRESH while five anchors were wrong. Every table below is
// transcribed from upstream or from CUTLASS 4.5.0 by hand and asserted against
// the implementation, never derived from it.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vt/cuda/fp8_block_scaled_dispatch.h"

namespace {

using vt::cuda::Fp8BlockScaledActScaleIndex;
using vt::cuda::Fp8BlockScaledConfig;
using vt::cuda::Fp8BlockScaledConfigFor;
using vt::cuda::Fp8BlockScaledConfigName;
using vt::cuda::Fp8BlockScaledCountDispatch;
using vt::cuda::Fp8BlockScaledCountRefusal;
using vt::cuda::Fp8BlockScaledRefusal;
using vt::cuda::Fp8BlockScaledRefusalFor;
using vt::cuda::Fp8BlockScaledRefusalMessage;
using vt::cuda::Fp8BlockScaledStats;
using vt::cuda::Fp8BlockScaledStatsSnapshot;
using vt::cuda::Fp8BlockScaledWeightScaleIndex;

bool Contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// G1 — the tile-config heuristic
// ---------------------------------------------------------------------------
//
// upstream `cutlass_gemm_blockwise_sm120_fp8_dispatch`
// (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//  scaled_mm_blockwise_sm120_fp8_dispatch.cuh):
//
//     bool swap_ab = (M <= 64) || (M % 4 != 0);
//     if (!swap_ab) { if (M <= 256) pingpong; else default; }
//     else swapab;
//
// Both clauses of the OR are exercised separately, because a heuristic written
// as `M <= 64 && M % 4 != 0` — one character apart, and the wrong operator —
// agrees with this one on every M below 65 and on every M divisible by 4. Only
// an M above 64 that is NOT divisible by 4 tells them apart, which is why 65,
// 66, 67 and 4098 are here.
TEST_CASE("G1 the sm120 blockwise tile config mirrors upstream's M heuristic") {
  // M <= 64: swapped regardless of divisibility. Decode is M = 1 and lives here.
  CHECK(Fp8BlockScaledConfigFor(1) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(2) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(3) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(4) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(63) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(64) == Fp8BlockScaledConfig::kSwapAb);

  // M > 64 and M % 4 != 0: still swapped, and this is the clause a `&&` would
  // lose. It is a correctness condition, not a tuning knob — see the header.
  CHECK(Fp8BlockScaledConfigFor(65) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(66) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(67) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(4098) == Fp8BlockScaledConfig::kSwapAb);

  // M > 64, M % 4 == 0, M <= 256: pingpong.
  CHECK(Fp8BlockScaledConfigFor(68) == Fp8BlockScaledConfig::kPingpong);
  CHECK(Fp8BlockScaledConfigFor(128) == Fp8BlockScaledConfig::kPingpong);
  CHECK(Fp8BlockScaledConfigFor(256) == Fp8BlockScaledConfig::kPingpong);

  // M > 256 and M % 4 == 0: the default 128x128x128 tile.
  CHECK(Fp8BlockScaledConfigFor(260) == Fp8BlockScaledConfig::kDefault);
  CHECK(Fp8BlockScaledConfigFor(4096) == Fp8BlockScaledConfig::kDefault);

  // 257, 258 and 259 are all above 256 AND indivisible by 4, so they swap. A
  // reading that put the `M <= 256` test first would answer pingpong.
  CHECK(Fp8BlockScaledConfigFor(257) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(258) == Fp8BlockScaledConfig::kSwapAb);
  CHECK(Fp8BlockScaledConfigFor(259) == Fp8BlockScaledConfig::kSwapAb);

  // The names are what the counter reports and what a reader sees in a trace,
  // so a config that fell off the switch reads as "unknown" rather than as its
  // neighbour.
  CHECK(std::string(Fp8BlockScaledConfigName(Fp8BlockScaledConfig::kSwapAb)) ==
        "swapab_128x32x128");
  CHECK(std::string(Fp8BlockScaledConfigName(Fp8BlockScaledConfig::kPingpong)) ==
        "pingpong_64x128x128");
  CHECK(std::string(Fp8BlockScaledConfigName(Fp8BlockScaledConfig::kDefault)) ==
        "default_128x128x128");
  CHECK(std::string(Fp8BlockScaledConfigName(Fp8BlockScaledConfig::kCount)) == "unknown");
}

// ---------------------------------------------------------------------------
// G3 — the scale layouts CUTLASS deduces
// ---------------------------------------------------------------------------
//
// `cutlass_gemm_caller_blockwise` never reads the scale tensors' strides; it
// builds `layout_SFA`/`layout_SFB` from the PROBLEM SHAPE. So if these two
// formulas are wrong, every element after the first K-block is multiplied by
// the wrong scale and the GEMM is silently wrong — on hardware only, which is
// exactly why the formulas are pulled out here.
//
// Derived by hand from CUTLASS 4.5.0
// `include/cutlass/detail/blockwise_scale_layout.hpp`,
// `Sm1xxBlockwiseScaleConfig::tile_atom_to_shape_SFA` / `..._SFB`:
//
//     majorSF == UMMA::Major::MN -> stride ((0,1),(0, ceil_div(extent, vec)))
//     majorSF == UMMA::Major::K  -> stride ((0, ceil_div(K, vecK)),(0,1))
//
// The unswapped config is <1,128,128, MN, K>; the swapped one is
// <128,1,128, K, MN> over the problem (N,M,K). Working both through, the
// ACTIVATION scale is MN-major in both and the WEIGHT scale is K-major in both.
TEST_CASE("G3 the deduced scale layouts match a hand-derived index table") {
  // The activation scale is COLUMN-major [M, k_tiles]: consecutive rows are
  // adjacent, consecutive k-tiles are M apart. For M=3, k_tiles=2 the indices
  // visited in ROW-major order — (0,0),(0,1),(1,0),(1,1),(2,0),(2,1) — are
  // 0,3,1,4,2,5. A row-major implementation would give 0,1,2,3,4,5, which is
  // the whole bug this table exists to catch.
  const std::vector<int64_t> want_act = {0, 3, 1, 4, 2, 5};
  std::vector<int64_t> got_act;
  for (int64_t row = 0; row < 3; ++row)
    for (int64_t kt = 0; kt < 2; ++kt) got_act.push_back(Fp8BlockScaledActScaleIndex(row, kt, 3));
  CHECK(got_act == want_act);

  // The weight scale is ROW-major [n_tiles, k_tiles] — the checkpoint's own
  // `weight_scale_inv` layout, which is why upstream can hand cutlass `Bs.T`, a
  // transposed VIEW whose data pointer is `Bs`'s. For n_tiles=3, k_tiles=2 the
  // indices in row-major visiting order are 0..5.
  const std::vector<int64_t> want_wgt = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> got_wgt;
  for (int64_t nb = 0; nb < 3; ++nb)
    for (int64_t kt = 0; kt < 2; ++kt)
      got_wgt.push_back(Fp8BlockScaledWeightScaleIndex(nb, kt, 2));
  CHECK(got_wgt == want_wgt);

  // THE TWO LAYOUTS COINCIDE AT k_tiles == 1, so a transpose that was skipped
  // for the single-K-block case would be invisible everywhere except here. Both
  // formulas are asserted at k_tiles == 1 AND at k_tiles > 1, and the kernel has
  // no k_tiles == 1 fast path to skip.
  for (int64_t row = 0; row < 5; ++row) {
    CHECK(Fp8BlockScaledActScaleIndex(row, 0, 5) == row);
    CHECK(Fp8BlockScaledWeightScaleIndex(row, 0, 1) == row);
  }

  // A real shape from the target checkpoint: q_proj is [12288, 5120], so
  // n_tiles = 96 and k_tiles = 40, and a decode step is M = 1. The last element
  // of each grid pins the total extent, which a stride/extent swap gets wrong.
  CHECK(Fp8BlockScaledActScaleIndex(0, 39, 1) == 39);
  CHECK(Fp8BlockScaledWeightScaleIndex(95, 39, 40) == 96 * 40 - 1);
  // ... and a prefill of 32 tokens over the same weight.
  CHECK(Fp8BlockScaledActScaleIndex(31, 39, 32) == 32 * 40 - 1);
  CHECK(Fp8BlockScaledActScaleIndex(31, 0, 32) == 31);
  CHECK(Fp8BlockScaledActScaleIndex(0, 1, 32) == 32);
}

// ---------------------------------------------------------------------------
// G4 — the refusals, and the shapes that must NOT be refused
// ---------------------------------------------------------------------------
TEST_CASE("G4 the CUDA arm refuses exactly what cutlass cannot implement") {
  // --- accepted, and this half is load-bearing --------------------------------
  // A RAGGED 128-BLOCK IS NOT A REFUSAL. Upstream's own CUTLASS test is
  // M=32, N=576, K=7168 — 576 is 4*128 + 64 — chosen because DSV3's
  // kv_a_proj_with_mqa has that shape
  // (tests/kernels/quantization/test_block_fp8.py,
  // test_w8a8_block_fp8_cutlass_matmul). A refusal predicate that keyed on
  // `N % 128` instead of `N % 16` would reject the one shape upstream gates.
  CHECK(Fp8BlockScaledRefusalFor(576, 7168, 128, 128) == Fp8BlockScaledRefusal::kNone);
  // The target checkpoint's ten projections are all round; q_proj is the
  // largest.
  CHECK(Fp8BlockScaledRefusalFor(12288, 5120, 128, 128) == Fp8BlockScaledRefusal::kNone);
  CHECK(Fp8BlockScaledRefusalFor(5120, 12288, 128, 128) == Fp8BlockScaledRefusal::kNone);
  // Ragged on both axes at once, still aligned to 16.
  CHECK(Fp8BlockScaledRefusalFor(576, 4096 + 64, 128, 128) == Fp8BlockScaledRefusal::kNone);
  // Exactly 16 is aligned; the boundary is `% 16`, not `>= 128`.
  CHECK(Fp8BlockScaledRefusalFor(16, 16, 128, 128) == Fp8BlockScaledRefusal::kNone);

  // --- refused ----------------------------------------------------------------
  // K = 3884 is the other non-round shape in upstream's grid
  // (test_block_fp8.py, K = [256, 3884, 4096, 13824, 16384]) and 3884 % 16 is
  // 12, so cutlass cannot implement it and upstream reroutes it to triton one
  // rung higher (vllm/_custom_ops.py, cutlass_compatible_b). There is no triton
  // block arm here, so it is refused — while the CPU arm of the SAME op runs it,
  // which is upstream's situation exactly.
  CHECK(Fp8BlockScaledRefusalFor(512, 3884, 128, 128) == Fp8BlockScaledRefusal::kAlignK);
  CHECK(Fp8BlockScaledRefusalFor(512, 129, 128, 128) == Fp8BlockScaledRefusal::kAlignK);
  // N misaligned.
  CHECK(Fp8BlockScaledRefusalFor(578, 4096, 128, 128) == Fp8BlockScaledRefusal::kAlignN);
  CHECK(Fp8BlockScaledRefusalFor(1, 4096, 128, 128) == Fp8BlockScaledRefusal::kAlignN);
  // The block geometry is a compile-time template parameter of the collective.
  CHECK(Fp8BlockScaledRefusalFor(512, 4096, 64, 128) == Fp8BlockScaledRefusal::kBlockN);
  CHECK(Fp8BlockScaledRefusalFor(512, 4096, 256, 128) == Fp8BlockScaledRefusal::kBlockN);
  CHECK(Fp8BlockScaledRefusalFor(512, 4096, 128, 64) == Fp8BlockScaledRefusal::kBlockK);
  CHECK(Fp8BlockScaledRefusalFor(512, 4096, 128, 512) == Fp8BlockScaledRefusal::kBlockK);
  // The block geometry is asked BEFORE the alignment, because a checkpoint that
  // declares a different block size is a different question from a shape cutlass
  // cannot tile, and answering the second would misdescribe the first.
  CHECK(Fp8BlockScaledRefusalFor(578, 3884, 64, 64) == Fp8BlockScaledRefusal::kBlockN);

  // --- the messages -----------------------------------------------------------
  // Each names the dimension, its remainder, and where upstream draws the same
  // line, because a reader who hits one is being told the two arms of ONE op
  // have different domains and needs to know that is not our invention.
  CHECK(Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal::kNone, 512, 4096, 128, 128).empty());

  const std::string k_msg =
      Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal::kAlignK, 512, 3884, 128, 128);
  CHECK(Contains(k_msg, "3884"));
  CHECK(Contains(k_msg, "remainder of 12"));
  CHECK(Contains(k_msg, "16"));
  // The WHOLE clause, not the bare word: "triton" also appears in this message's
  // next sentence ("there is no triton block arm here"), so a bare-word check
  // passes with the reroute sentence deleted. Measured: mutation M10 removed the
  // reroute clause and the bare-word form stayed green.
  CHECK(Contains(k_msg, "reroutes it to triton"));
  CHECK(Contains(k_msg, "cutlass_compatible_b"));
  CHECK(Contains(k_msg, "CPU reference arm runs this shape"));

  const std::string n_msg =
      Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal::kAlignN, 578, 4096, 128, 128);
  CHECK(Contains(n_msg, "578"));
  CHECK(Contains(n_msg, "remainder of 2"));
  // The N message says out loud that a ragged 128-block is a DIFFERENT thing, so
  // the next reader does not "fix" the predicate into `% 128`.
  CHECK(Contains(n_msg, "576"));
  CHECK(Contains(n_msg, "RAGGED"));

  const std::string bn_msg =
      Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal::kBlockN, 512, 4096, 64, 128);
  CHECK(Contains(bn_msg, "block_n is 64"));
  CHECK(Contains(bn_msg, "128"));

  const std::string bk_msg =
      Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal::kBlockK, 512, 4096, 128, 64);
  CHECK(Contains(bk_msg, "block_k is 64"));
  CHECK(Contains(bk_msg, "128"));

  // Every message names the op, so the refusal is greppable from a log that
  // carries nothing else.
  for (const std::string& m : {k_msg, n_msg, bn_msg, bk_msg})
    CHECK(Contains(m, "matmul_fp8_block_scaled"));
}

// ---------------------------------------------------------------------------
// G5 — the dispatch counter's accounting
// ---------------------------------------------------------------------------
//
// #1189's gate design records why the counter exists: a x1.02 and a x1.10 scale
// perturbation on the per-tensor fp8 tower were demonstrably REACHED and still
// produced 16/16 identical tokens
// (tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp). A token gate cannot
// say whether anything ran here, and a silent dequant to bf16 is numerically
// BETTER than the quantized path, so no value comparison in this tree can see
// it either.
//
// Differenced, never read absolutely: another case in this binary may have
// dispatched, and a counter asserted at an absolute value would be a test of the
// run order rather than of the accounting.
TEST_CASE("G5 the block-scaled dispatch counter accounts per config and per refusal") {
  const Fp8BlockScaledStats before = Fp8BlockScaledStatsSnapshot();

  Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig::kSwapAb);
  Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig::kSwapAb);
  Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig::kPingpong);
  Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig::kDefault);
  Fp8BlockScaledCountRefusal();

  const Fp8BlockScaledStats after = Fp8BlockScaledStatsSnapshot();
  CHECK(after.swap_ab - before.swap_ab == 2);
  CHECK(after.pingpong - before.pingpong == 1);
  CHECK(after.deflt - before.deflt == 1);
  CHECK(after.refused - before.refused == 1);

  // A REFUSAL IS NOT A DISPATCH. If it were counted as one, a build whose every
  // call was refused would look identical to a build whose every call ran, which
  // is the precise question the counter exists to answer.
  CHECK(after.dispatched() - before.dispatched() == 4);

  // `kCount` is a sentinel, not a config: counting it would write past the
  // array. The guard is asserted rather than assumed.
  Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig::kCount);
  const Fp8BlockScaledStats sentinel = Fp8BlockScaledStatsSnapshot();
  CHECK(sentinel.dispatched() == after.dispatched());
  CHECK(sentinel.refused == after.refused);
}
