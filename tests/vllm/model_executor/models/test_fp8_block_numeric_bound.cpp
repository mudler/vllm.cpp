// GATE-FP8-NUMERIC-BOUND — the second and third layers of issue #1189's
// `## Gate design`, spec `.agents/specs/gate-fp8-numeric-bound.md`.
//
// WHY THIS FILE EXISTS, and it is a measurement rather than an opinion.
// `tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp` records a RED-mutation
// sweep on the per-tensor fp8 tower in which a x1.02 AND a x1.10 scale
// perturbation were both demonstrably REACHED and both still produced 16/16
// identical tokens; only x2.00 failed. So a token gate — including the one
// `GATE-QWEN38-27B-FP8-BLOCK` passed on 2026-08-23 against the pinned oracle —
// cannot see a wrong-but-close scale, and a silent dequant is numerically
// BETTER than the quantized path and therefore invisible to every value
// comparison in the tree.
//
// #1189 asks for two things a token gate cannot fake, and this file is both:
//
//   N1  a numerical LOWER BOUND on per-projection outputs, tight enough to fail
//       a x1.10 scale perturbation. Asserted in BOTH directions in the same
//       case: the unperturbed arm must be UNDER the bound and the perturbed arm
//       must be OVER it. A bound asserted in one direction only stops biting
//       the moment somebody widens it, and nothing would say so.
//   N2  the per-block scale-variance probe, whose degenerate per-tensor reading
//       is exactly 1.0 (`dense_fp8_block::Fp8BlockScaleSpread`), plus the
//       `Fp8BlockStats` snapshot that carries it beside the GEMM counter.
//   N3  the split path's GEMM-boundary operand assertions, which the merged
//       path has carried since M6 and the split path did not.
//
// CPU TIER, and that is the point: `vt::MatmulFp8BlockScaled` has a CPU
// reference arm (#1189 M2), so every assertion here runs on the `build-test-cpu`
// and `sanitize-cpu` lanes on every pull request, with no device and no
// checkpoint. The CUDA arm's own numerical comparison against this same
// reference is `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` G2/G7 and needs a
// lease; this file does not substitute for it and does not claim to.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/fp8_block_fixture.h"
#include "vllm/model_executor/layers/quantization/fp8.h"
#include "vllm/model_executor/layers/quantization/fp8_block.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using namespace fp8_block_fixture;  // NOLINT(build/namespaces)

namespace layers = vllm::layers;
namespace block_gemm = vllm::dense_fp8_block;
using vllm::Fp8BlockWeight;
using vt::DType;

namespace {

// ---------------------------------------------------------------------------
// The bound, and the two constants that make it two-sided
// ---------------------------------------------------------------------------
//
// THE STATISTIC. Per output element, `|got - ref| / max(|ref|, floor)`, worst
// over the projection, where `floor` is 1% of the projection's own largest
// reference magnitude and `ref` is the fixture's independent `double`
// `RefBlockGemm` — a different loop nest, a `double` accumulator and a
// bit-unpacked e4m3 decode, not the kernel under test.
//
// THE FLOOR IS NOT COSMETIC AND IT WAS MEASURED. With a bare `max(|ref|, 1e-6)`
// denominator, the same seven shapes read a worst UNPERTURBED relative error of
// 6.58e-3 at f32 output on `{M=32, N=512, K=7168}` — 835x the floored reading
// of 7.88e-6 on the identical data — because one output element cancels to near
// zero and its ratio says nothing about any scale. That statistic cannot carry
// a bound: it is dominated by whichever element happened to cancel hardest, and
// under the x1.10 perturbation it reads 609, so it is not even wrong in a
// useful direction. The floored statistic separates the two regimes by a factor
// this file asserts.
//
// THE TWO CONSTANTS, both measured on this tree before they were written down
// (`.agents/specs/gate-fp8-numeric-bound.md` `## Evidence` carries the table):
//
//   out_dtype   unperturbed worst   x1.10 worst   bound   headroom / bite
//   bf16         3.82e-3             1.034e-1      2e-2    5.24x  / 5.17x
//   f32          6.20e-6             1.000e-1      1e-4    16.1x  / 1000x
//
// Worst and tightest are taken over the WHOLE grid, so both columns are the
// hardest reading each regime produced rather than a representative one. The
// bf16 margin is 5.2x in each direction, a factor of 27 between the regimes;
// that is the number to argue with if this bound is ever called too loose or
// too tight, and it is not enormous. The f32 margin is.
//
// bf16 is the MODEL dtype — upstream's `out_dtype` on this path is
// `torch.get_default_dtype()` — so `kBoundBf16` is the load-bearing one and
// `kBoundF32` measures the arithmetic without the store's rounding on top.
// The bf16 reading is pinned from below by the store itself: this tree's
// `F32ToBF16` truncates, so 2^-8 = 3.906e-3 is a STRUCTURAL floor that the
// measured 3.82e-3 sits just under, and not a number that can drift up under an
// unrelated change. The 4x margin assertions at the end of N1 therefore leave
// 1.28x of slack on the bf16 side against that floor, which is stated here so
// that a future reader tightening `kBoundBf16` knows exactly what they will
// hit.
//
// LOOSE ENOUGH FOR THE ONE LEGITIMATE DIFFERENCE. Our CUTLASS collective
// associates the two scale multiplies left to right where the reference forms
// their product first, so the two arms may differ by up to one f32 ULP per
// K-block (`.agents/specs/vt-matmul-fp8-block-cuda.md`). At the widest shape
// here that is 56 K-blocks x 2^-24, about 3.3e-6 relative — under `kBoundF32`
// by 30x and under `kBoundBf16` by four orders. That difference is a DEVICE
// difference and does not arise on this CPU tier at all; it is written down
// because the same constants are what a device-side comparison must clear.
// Upstream applies `rel_diff < 0.001` to exactly this pair
// (`test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul`), which sits between
// the two constants below, as it should.
constexpr double kBoundBf16 = 2e-2;
constexpr double kBoundF32 = 1e-4;

// The perturbation the bound has to reject: upstream-plausible, one block, and
// the exact factor `test_qwen27n_fp8_tower_paged_engine.cpp` measured passing a
// token gate 16/16.
constexpr float kPerturbation = 1.10F;

struct Shape {
  int64_t m, n, k, block_n, block_k;
  const char* what;
};

// The shapes this model family actually presents, plus upstream's own
// CUTLASS-test K at the nearest N sm120 can serve. Every `K` is a multiple of
// `block_k`, which the dynamic per-token per-group activation quant requires
// (`fp8_utils.py:596-599`), and every entry says what it is in the grid for so
// that removing one has to argue against a sentence.
const Shape* Grid(size_t* count) {
  static const Shape g[] = {
      {1, 128, 128, 128, 128, "decode M=1, the shape the 27B gate run dispatched 2736 times"},
      {4, 192, 128, 128, 128, "GDN in_proj_qkv: a RAGGED N, cdiv leaves a short final block"},
      {4, 256, 128, 128, 128, "the merged attention QKV"},
      {3, 192, 256, 64, 128, "block_n != block_k, so a swapped pair misindexes the grid"},
      {8, 512, 1024, 128, 128, "8 K-tiles, so a per-K-block scale has somewhere to be wrong"},
      {8, 512, 7168, 128, 128, "upstream's own CUTLASS-test K at the nearest servable N"},
  };
  *count = sizeof(g) / sizeof(g[0]);
  return g;
}

// `|got - ref| / max(|ref|, 0.01 * max|ref|)`, worst over the projection.
double WorstRelative(const std::vector<float>& got,
                     const std::vector<double>& ref) {
  REQUIRE(got.size() == ref.size());
  double biggest = 0.0;
  for (const double r : ref) biggest = std::max(biggest, std::abs(r));
  const double floor_v = 0.01 * biggest;
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst,
                     std::abs(got[i] - ref[i]) / std::max(std::abs(ref[i]), floor_v));
  return worst;
}

// Multiply ONE cell of the grid: the LAST block row, first K-column. Never
// (0,0), so a per-tensor collapse to the first cell, an epilogue-folded scalar
// and a transposed index are each blind to it.
void BumpOneScaleBlock(Fp8BlockWeight* w, float factor) {
  auto* s = reinterpret_cast<float*>(w->scale.bytes.data());
  const int64_t rows = w->scale.shape[0], cols = w->scale.shape[1];
  s[(rows - 1) * cols + 0] *= factor;
}

// Replace every cell with the grid's own first value: a PER-TENSOR weight
// wearing a block-wise grid, which is what `Fp8BlockScaleSpread` exists to see.
void CollapseScaleGrid(Fp8BlockWeight* w) {
  auto* s = reinterpret_cast<float*>(w->scale.bytes.data());
  const size_t cells = w->scale.bytes.size() / sizeof(float);
  for (size_t i = 1; i < cells; ++i) s[i] = s[0];
}

// One projection through the PRODUCTION per-projection entry point,
// `layers::Fp8BlockLinearMethod::Apply` — the same call
// `ModelRegistry::Forward` makes at every block-wise projection, which
// `test_fp8_block_linear.cpp` G3 pins by dispatch count. `arm_bump` perturbs
// the scales the ARM sees; the reference is always computed from the
// UNPERTURBED weight, so the returned number is the error a kernel applying a
// wrong scale would produce.
double ProjectionError(const Shape& sh, DType out_dtype, float arm_bump) {
  const Fp8BlockWeight truth =
      MakeFp8Block(sh.n, sh.k, sh.block_n, sh.block_k, 4242);
  Fp8BlockWeight arm = truth;
  arm.d_packed.reset();
  arm.d_scale.reset();
  if (arm_bump != 1.0F) BumpOneScaleBlock(&arm, arm_bump);

  vllm::OwnedTensor xw = MakeOwned(DType::kBF16, {sh.m, sh.k}, 7);
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {sh.m, sh.k}, xw.bytes.data());

  auto method =
      layers::MakeLinearMethod(MakeOwned(DType::kBF16, {sh.n, sh.k}, 1), arm);
  REQUIRE(std::string(method->Name()) == "fp8-w8a8-block");
  vllm::dense_attn::DBuf out = method->Apply(d, x.t(), out_dtype);

  // The activation half through the gated M1 op, so what is measured below is
  // the GEMM and the composition rather than a second e4m3 encoder.
  vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {sh.m, sh.k});
  vllm::dense_attn::DBuf a_scale(d, DType::kF32, {sh.m, sh.k / sh.block_k});
  vt::QuantFp8Group(q, a_fp8.t(), a_scale.t(), x.t(),
                    static_cast<int>(sh.block_k));
  std::vector<uint8_t> a_bytes(static_cast<size_t>(sh.m * sh.k));
  std::memcpy(a_bytes.data(), a_fp8.t().data, a_bytes.size());
  std::vector<float> a_s(static_cast<size_t>(sh.m * (sh.k / sh.block_k)));
  std::memcpy(a_s.data(), a_scale.t().data, a_s.size() * 4);

  const std::vector<double> ref = RefBlockGemm(a_bytes, a_s, sh.m, sh.k, truth);
  const std::vector<float> got = ToF32(out.t(), sh.m * sh.n);
  // A vacuity guard: an all-zero output satisfies any ratio whose reference is
  // also small, and a bound over an empty signal bounds nothing.
  CHECK(CountNonZero(got) == static_cast<int64_t>(got.size()));
  return WorstRelative(got, ref);
}

}  // namespace

// N1 -------------------------------------------------------------------------
TEST_CASE("fp8 block numeric bound: every projection is under the bound, and a x1.10 scale is over it") {
  size_t count = 0;
  const Shape* grid = Grid(&count);
  REQUIRE(count == 6);

  double worst_clean_bf16 = 0.0, worst_clean_f32 = 0.0;
  double tightest_bump_bf16 = std::numeric_limits<double>::infinity();
  double tightest_bump_f32 = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < count; ++i) {
    const Shape& sh = grid[i];
    CAPTURE(sh.m);
    CAPTURE(sh.n);
    CAPTURE(sh.k);
    const std::string what(sh.what);
    CAPTURE(what);

    // bf16 out — the MODEL dtype, and what every wired call site passes.
    const double clean_bf16 = ProjectionError(sh, DType::kBF16, 1.0F);
    const double bump_bf16 = ProjectionError(sh, DType::kBF16, kPerturbation);
    CAPTURE(clean_bf16);
    CAPTURE(bump_bf16);
    CHECK(clean_bf16 < kBoundBf16);
    CHECK(bump_bf16 > kBoundBf16);
    worst_clean_bf16 = std::max(worst_clean_bf16, clean_bf16);
    tightest_bump_bf16 = std::min(tightest_bump_bf16, bump_bf16);

    // f32 out — the same arithmetic without the store's rounding on top, so the
    // bound can be two orders tighter and the margin is enormous.
    const double clean_f32 = ProjectionError(sh, DType::kF32, 1.0F);
    const double bump_f32 = ProjectionError(sh, DType::kF32, kPerturbation);
    CAPTURE(clean_f32);
    CAPTURE(bump_f32);
    CHECK(clean_f32 < kBoundF32);
    CHECK(bump_f32 > kBoundF32);
    worst_clean_f32 = std::max(worst_clean_f32, clean_f32);
    tightest_bump_f32 = std::min(tightest_bump_f32, bump_f32);
  }

  // THE MARGIN, asserted rather than described. A bound that both regimes clear
  // is useless and a bound neither clears is broken; these two say the gap is
  // wide in both directions at once, over the whole grid, so a later edit that
  // narrows either side fails HERE rather than quietly.
  CAPTURE(worst_clean_bf16);
  CAPTURE(tightest_bump_bf16);
  CHECK_MESSAGE(worst_clean_bf16 * 4.0 < kBoundBf16,
                "the clean bf16 arm must clear the bound by 4x, worst was "
                    << worst_clean_bf16);
  CHECK_MESSAGE(tightest_bump_bf16 > kBoundBf16 * 4.0,
                "the x1.10 bf16 arm must exceed the bound by 4x, tightest was "
                    << tightest_bump_bf16);
  CAPTURE(worst_clean_f32);
  CAPTURE(tightest_bump_f32);
  CHECK_MESSAGE(worst_clean_f32 * 4.0 < kBoundF32,
                "the clean f32 arm must clear the bound by 4x, worst was "
                    << worst_clean_f32);
  CHECK_MESSAGE(tightest_bump_f32 > kBoundF32 * 4.0,
                "the x1.10 f32 arm must exceed the bound by 4x, tightest was "
                    << tightest_bump_f32);
}

// N2 -------------------------------------------------------------------------
TEST_CASE("fp8 block numeric bound: the scale-variance probe reads exactly 1.0 on a per-tensor collapse") {
  SUBCASE("a genuine per-block grid is not 1.0, and a collapsed one is EXACTLY 1.0") {
    Fp8BlockWeight w = MakeFp8Block(256, 256, 128, 128, 11);
    REQUIRE(w.scale.shape[0] == 2);
    REQUIRE(w.scale.shape[1] == 2);
    const float genuine = block_gemm::Fp8BlockScaleSpread(w);
    CAPTURE(genuine);
    CHECK(genuine > 1.0F);

    Fp8BlockWeight collapsed = w;
    CollapseScaleGrid(&collapsed);
    // `==`, not `approx`. The issue pins the degenerate reading at exactly 1.0
    // and a tolerance here would accept a grid that is merely nearly uniform,
    // which is a different (and unasserted) claim.
    CHECK(block_gemm::Fp8BlockScaleSpread(collapsed) == 1.0F);
  }

  SUBCASE("a single-cell grid is 1.0 too, because there is nothing per-block about it") {
    const Fp8BlockWeight one = MakeFp8Block(128, 128, 128, 128, 3);
    REQUIRE(one.scale.bytes.size() == sizeof(float));
    CHECK(block_gemm::Fp8BlockScaleSpread(one) == 1.0F);
  }

  SUBCASE("a non-positive minimum is INFINITY, never the collapse reading") {
    Fp8BlockWeight z = MakeFp8Block(256, 256, 128, 128, 5);
    reinterpret_cast<float*>(z.scale.bytes.data())[3] = 0.0F;
    const float spread = block_gemm::Fp8BlockScaleSpread(z);
    CHECK(std::isinf(spread));
    CHECK(spread != 1.0F);
  }

  SUBCASE("ONE realistically-sized projection, healthy then collapsed, through Apply") {
    // The model fixture below is 128 wide almost everywhere, so 11 of its 13
    // grids hold a single cell and cannot discriminate anything. This subcase
    // carries the discrimination at a grid size a real checkpoint has: N=512,
    // K=1024 in [128,128] blocks is 4 x 8 = 32 cells.
    vt::Queue q = Q();
    vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
    vllm::dense_attn::Dev d{b, q};
    vllm::OwnedTensor xw = MakeOwned(DType::kBF16, {4, 1024}, 7);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {4, 1024}, xw.bytes.data());

    const auto run = [&](const Fp8BlockWeight& w) {
      auto m = layers::MakeLinearMethod(MakeOwned(DType::kBF16, {512, 1024}, 1), w);
      return ToF32(m->Apply(d, x.t(), DType::kBF16).t(), 4 * 512);
    };

    const Fp8BlockWeight healthy = MakeFp8Block(512, 1024, 128, 128, 21);
    REQUIRE(healthy.scale.bytes.size() / sizeof(float) == 32);
    Fp8BlockWeight collapsed = healthy;
    collapsed.d_packed.reset();
    collapsed.d_scale.reset();
    CollapseScaleGrid(&collapsed);

    const block_gemm::Fp8BlockStats s0 = block_gemm::ReadFp8BlockStats();
    const std::vector<float> a = run(healthy);
    const block_gemm::Fp8BlockStats s1 = block_gemm::ReadFp8BlockStats();
    const std::vector<float> bvals = run(collapsed);
    const block_gemm::Fp8BlockStats s2 = block_gemm::ReadFp8BlockStats();

    CHECK(s1.scale_grids - s0.scale_grids == 1);
    CHECK(s1.collapsed_scale_grids - s0.collapsed_scale_grids == 0);
    CHECK(s1.single_cell_scale_grids - s0.single_cell_scale_grids == 0);
    CHECK(s2.scale_grids - s1.scale_grids == 1);
    CHECK(s2.collapsed_scale_grids - s1.collapsed_scale_grids == 1);
    CHECK(s2.gemms - s0.gemms == 2);

    // BOTH projections are finite, fully populated and dispatched one GEMM
    // each. Nothing else in the tree separates them, which is the whole reason
    // the counter exists.
    CHECK(CountNonZero(a) == static_cast<int64_t>(a.size()));
    CHECK(CountNonZero(bvals) == static_cast<int64_t>(bvals.size()));
  }

  SUBCASE("a forward over the model records every grid, and NONE collapsed") {
    const vllm::HfConfig c = BlockConfig();
    const block_gemm::Fp8BlockStats before = block_gemm::ReadFp8BlockStats();
    uint64_t dispatched = 0;
    const std::vector<float> logits =
        RegistryForward(c, BlockWeights(c, true), &dispatched);
    const block_gemm::Fp8BlockStats after = block_gemm::ReadFp8BlockStats();

    // The snapshot carries the GEMM counter too, and it agrees with the count
    // `test_fp8_block_linear.cpp` G3 pins independently.
    CHECK(after.gemms - before.gemms == dispatched);
    CHECK(dispatched == static_cast<uint64_t>(kBlockProjectionsPerForward));

    // 13 quantized projections over two layers, each with its own grid: the
    // three GDN projections and three MLP, twice, plus the attention layer's
    // q/k/v and o_proj. The merged groups upload one operand but record their
    // SHARDS, which is why this is 13 where the GEMM count is 9.
    const uint64_t grids = after.scale_grids - before.scale_grids;
    CAPTURE(grids);
    CHECK(grids == 13);
    CHECK(after.collapsed_scale_grids - before.collapsed_scale_grids == 0);
    // Eleven of the thirteen are ONE cell, because this config's hidden size is
    // 128 and a [128,128] block covers a whole projection. Only `in_proj_qkv`
    // (N=192) and `q_proj` (N=256) have a grid with two rows. The count is
    // asserted rather than glossed, so that the two the model-level half can
    // actually discriminate are visible as two.
    CHECK(after.single_cell_scale_grids - before.single_cell_scale_grids == 11);
    CHECK(AllFinite(logits));
  }

  SUBCASE("the same forward over COLLAPSED grids records every one it can see") {
    const vllm::HfConfig c = BlockConfig();
    vllm::Qwen3_5DenseWeights w = BlockWeights(c, true);
    uint64_t collapsed_built = 0;
    for (auto& lw : w.layers) {
      Fp8BlockWeight* all[] = {
          &lw.gdn.in_proj_qkv_fp8_block, &lw.gdn.in_proj_z_fp8_block,
          &lw.gdn.out_proj_fp8_block,    &lw.attn.q_proj_fp8_block,
          &lw.attn.k_proj_fp8_block,     &lw.attn.v_proj_fp8_block,
          &lw.attn.o_proj_fp8_block,     &lw.mlp.gate_proj_fp8_block,
          &lw.mlp.up_proj_fp8_block,     &lw.mlp.down_proj_fp8_block};
      for (Fp8BlockWeight* p : all) {
        if (p->Empty()) continue;
        CollapseScaleGrid(p);
        ++collapsed_built;
      }
    }
    REQUIRE(collapsed_built == 13);

    const block_gemm::Fp8BlockStats before = block_gemm::ReadFp8BlockStats();
    const std::vector<float> logits = RegistryForward(c, w, nullptr);
    const block_gemm::Fp8BlockStats after = block_gemm::ReadFp8BlockStats();
    CHECK(after.scale_grids - before.scale_grids == 13);
    // THE DISCRIMINATION. The tokens this forward would sample, the bytes it
    // moves and the GEMMs it dispatches are all unchanged by the collapse; this
    // counter is the only instrument in the tree that moves. It moves by TWO
    // and not by thirteen, and the reason is stated rather than rounded off:
    // the other eleven grids hold one cell, so there was no per-block structure
    // in them for the collapse to remove. The subcase above carries the same
    // discrimination on a 32-cell grid.
    CHECK(after.collapsed_scale_grids - before.collapsed_scale_grids == 2);
    CHECK(after.single_cell_scale_grids - before.single_cell_scale_grids == 11);
    CHECK(AllFinite(logits));
  }
}

// N3 -------------------------------------------------------------------------
TEST_CASE("fp8 block numeric bound: the GEMM boundary refuses a malformed operand by name") {
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  const int64_t M = 4, N = 256, K = 256;
  vllm::OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 7);
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  const auto apply = [&](const Fp8BlockWeight& w) {
    auto m = layers::MakeLinearMethod(MakeOwned(DType::kBF16, {N, K}, 1), w);
    return m->Apply(d, x.t(), DType::kBF16);
  };

  SUBCASE("the well-formed weight is served, so the refusals below are not vacuous") {
    CHECK_NOTHROW(apply(MakeFp8Block(N, K, 128, 128, 9)));
  }

  SUBCASE("a packed buffer short of one fp8 byte per element") {
    // Without this check the [N,K] tensor view runs off the end of the buffer
    // and the first GEMM reads unowned memory — a wrong answer on a good day
    // and a sanitizer report on a bad one.
    Fp8BlockWeight w = MakeFp8Block(N, K, 128, 128, 9);
    w.packed.bytes.resize(w.packed.bytes.size() - 128);
    CHECK_THROWS_AS(apply(w), std::runtime_error);
    try {
      apply(w);
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      CHECK(msg.find("one fp8-e4m3fn byte per element") != std::string::npos);
      CHECK(msg.find(std::to_string(N * K)) != std::string::npos);
    }
  }

  SUBCASE("a scale grid tiled by FLOOR instead of cdiv") {
    // The ragged case: N = 192 in [128,128] blocks needs 2 rows and a floor
    // tiling gives 1, which drops the short final block's scale entirely.
    Fp8BlockWeight w = MakeFp8Block(192, K, 128, 128, 9);
    REQUIRE(w.scale.shape[0] == 2);
    w.scale.shape[0] = 1;
    w.scale.bytes.resize(static_cast<size_t>(w.scale.shape[1]) * sizeof(float));
    vllm::dense_attn::DBuf xr(d, DType::kBF16, {M, K}, xw.bytes.data());
    auto m = layers::MakeLinearMethod(MakeOwned(DType::kBF16, {192, K}, 1), w);
    CHECK_THROWS_AS(m->Apply(d, xr.t(), DType::kBF16), std::runtime_error);
    try {
      m->Apply(d, xr.t(), DType::kBF16);
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      CHECK(msg.find("Both axes round UP") != std::string::npos);
      CHECK(msg.find("FLOOR tiling silently drops one") != std::string::npos);
    }
  }

  SUBCASE("a scale grid whose bytes disagree with its declared shape") {
    Fp8BlockWeight w = MakeFp8Block(N, K, 128, 128, 9);
    w.scale.bytes.resize(w.scale.bytes.size() - sizeof(float));
    CHECK_THROWS_AS(apply(w), std::runtime_error);
  }
}
