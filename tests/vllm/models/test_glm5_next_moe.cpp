// GLM-5.3-Flash W5 gate — the 288 routed + 1 shared expert MoE block, its
// grouped `noaux_tc` router and the clamped-SwiGLU epilogue (#2223, row
// MODEL-MM-glm5-next-glm5-next-for-conditional-generation,
// `.agents/specs/glm5-next-flash.md` §W5).
//
// THE ORACLE IS RUN, NOT TRANSCRIBED. Every golden in
// `fixtures/glm5_next_moe_goldens.inc` is the return value of an unmodified
// `transformers` v5.16.1 module — `Glm5NextTextTopkRouter.forward`,
// `Glm5NextTextExperts._apply_gate`, `Glm5NextTextMLP.forward` and
// `Glm5NextTextMoE.forward` — captured by
// `fixtures/gen_glm5_next_moe_goldens.py`. vLLM registers no `glm5_next` at any
// revision, so under AGENTS.md "When vLLM has no implementation" transformers is
// the reference for this surface; W0 (#2096) recorded the lane revision.
//
// EVERY GOLDEN THIS FILE'S FIXTURE EMITS IS CONSUMED BY AN ASSERTION BELOW.
// W3's review found a captured `kIndexScores` golden that nothing read, which
// let two real scale defects pass 1602 of 1602 assertions. `kHidden` and
// `kRouterWeight` are the router's INPUTS and `kRouterLogits` is asserted as its
// fp32 intermediate, rather than the logits being fed in as data.
//
// THE SELECTION IS ASSERTED AS A SET, NEVER AS A TOLERANCE. Top-k error is
// BIMODAL: the selection is either the oracle's set or a different set, and a
// different set can carry values that are numerically close. A tolerance on
// `topk_weights` therefore passes a wrong routing. This file asserts SET
// equality of the ids, asserts the weight AT each id, and prints the separation
// margin the oracle's own scores give, so a reader can see how far the
// selection was from flipping.
//
// ORDER. `vt::MoeRouterTopK` emits the selected ids in descending
// selection-score order with the lowest expert index winning an exact tie;
// upstream calls `torch.topk(..., sorted=False)`, whose order is unspecified.
// Positional comparison would therefore gate this tree's determinism convention
// against torch's implementation detail. The SET and the per-id weight are what
// the combine reads, and those are what is asserted.
#include "vllm/model_executor/models/glm5_next_moe.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "vllm/model_executor/models/deepseek_v4_moe.h"
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"  // W9a: AdmitMoeQuantBanks
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

#include "glm5_next_moe_goldens.inc"

namespace g = glm5_next_moe_goldens;
namespace gn = vllm::glm5_next;

namespace {

// Our f32 host reduction against the oracle's f32 torch reduction. Both are
// float32 and only the summation ORDER differs, so the gap is accumulated
// rounding over a short dot product rather than an algorithmic gap.
constexpr float kTol = 3e-6f;
// The composed block runs three chained GEMMs and a weighted combine, so its
// rounding accumulates further; still far below any value the block emits.
constexpr float kBlockTol = 2e-5f;

vt::Queue CpuQueue() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The PUBLISHED router geometry. Every field is the checkpoint's, and the
// fixture asserts on the python side that the reference config kept them.
gn::MoeDims BigDims() {
  gn::MoeDims d;
  d.hidden_size = g::kHiddenSize;
  d.n_routed_experts = g::kNumRoutedExperts;
  d.n_shared_experts = g::kNSharedExperts;
  d.num_experts_per_tok = g::kNumExpertsPerTok;
  d.moe_intermediate_size = g::kMoeIntermediate;
  d.n_group = g::kNGroup;
  d.topk_group = g::kTopkGroup;
  d.routed_scaling_factor = g::kRoutedScalingFactor;
  d.norm_topk_prob = g::kNormTopkProb;
  d.swiglu_limit = g::kSwigluLimit;
  return d;
}

gn::MoeDims SmallDims() {
  gn::MoeDims d = BigDims();
  d.n_routed_experts = g::kSmallExperts;
  d.num_experts_per_tok = g::kSmallTopK;
  return d;
}

template <typename T, size_t N>
std::vector<T> Vec(const T (&a)[N]) {
  return std::vector<T>(std::begin(a), std::end(a));
}

std::vector<float> Row(const float* base, int64_t t, int64_t width) {
  return std::vector<float>(base + t * width, base + (t + 1) * width);
}

std::set<int32_t> SetOf(const std::vector<int32_t>& v, int64_t t, int64_t k) {
  std::set<int32_t> s;
  for (int64_t j = 0; j < k; ++j) s.insert(v[static_cast<size_t>(t * k + j)]);
  return s;
}

std::set<int32_t> SetOf(const int32_t* v, int64_t t, int64_t k) {
  std::set<int32_t> s;
  for (int64_t j = 0; j < k; ++j) s.insert(v[t * k + j]);
  return s;
}

// The weight the router assigned to expert `id` for token `t`, or NaN when the
// router did not select it. Comparing by ID rather than by position is what
// makes the order deviation above harmless.
float WeightOf(const std::vector<int32_t>& ids, const std::vector<float>& w, int64_t t,
               int64_t k, int32_t id) {
  for (int64_t j = 0; j < k; ++j) {
    if (ids[static_cast<size_t>(t * k + j)] == id) return w[static_cast<size_t>(t * k + j)];
  }
  return std::numeric_limits<float>::quiet_NaN();
}

float WeightOf(const int32_t* ids, const float* w, int64_t t, int64_t k, int32_t id) {
  for (int64_t j = 0; j < k; ++j) {
    if (ids[t * k + j] == id) return w[t * k + j];
  }
  return std::numeric_limits<float>::quiet_NaN();
}

gn::MoeLayerWeights BigWeights(bool with_bias) {
  gn::MoeLayerWeights w;
  w.router_weight = Vec(g::kRouterWeight);
  if (with_bias) {
    w.e_score_correction_bias = Vec(g::kBias);
  } else {
    // The reference's buffer at its CONSTRUCTOR value, which is zeros and not
    // absent (`modeling_glm5_next.py:156`). Passing zeros rather than an empty
    // vector is what mirrors the oracle: an empty vector selects our seam's
    // no-bias arm, whose GROUP SCORE is the group max instead of the sum of the
    // top two. At `n_group == 1` the two masks agree, and the case below pins
    // that they do rather than assuming it.
    w.e_score_correction_bias.assign(static_cast<size_t>(g::kNumRoutedExperts), 0.0f);
  }
  return w;
}

gn::MoeLayerWeights SmallWeights() {
  gn::MoeLayerWeights w;
  w.router_weight = Vec(g::kSmallRouterWeight);
  w.e_score_correction_bias = Vec(g::kSmallBias);
  w.expert_gate_up = Vec(g::kSmallGateUp);
  w.expert_down = Vec(g::kSmallDown);
  w.shared.gate_proj = Vec(g::kSmallShGate);
  w.shared.up_proj = Vec(g::kSmallShUp);
  w.shared.down_proj = Vec(g::kSmallShDown);
  return w;
}

vllm::HfConfig PublishedConfig() {
  const std::string path = std::string(GLM5_NEXT_CKPT_FIXTURE_DIR) + "/config.json";
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "missing fixture: " << path);
  const nlohmann::json doc = nlohmann::json::parse(in);
  return vllm::ParseHfConfig(doc, path);
}

}  // namespace

TEST_CASE("glm5_next moe: the PUBLISHED config resolves to the published router") {
  const vllm::Glm5NextParams p = vllm::ParseGlm5NextParams(PublishedConfig());
  const gn::MoeDims d = gn::MoeDimsFrom(p);

  CHECK(d.hidden_size == 4096);
  CHECK(d.n_routed_experts == 288);
  CHECK(d.n_shared_experts == 1);
  CHECK(d.num_experts_per_tok == 8);
  CHECK(d.moe_intermediate_size == 2048);
  CHECK(d.n_group == 1);
  CHECK(d.topk_group == 1);
  CHECK(d.routed_scaling_factor == doctest::Approx(2.5));
  CHECK(d.norm_topk_prob);
  CHECK(d.swiglu_limit == doctest::Approx(10.0f));

  // The shared expert is `moe_intermediate_size * n_shared_experts`
  // (`modeling_glm5_next.py:196-198`), NOT `intermediate_size`. On this
  // checkpoint those are 2048 and 12288, so a port that reads the wrong field
  // builds a shared expert six times too wide and still runs.
  CHECK(d.shared_intermediate_size() == 2048);
  CHECK(p.intermediate_size == 12288);
  CHECK(d.shared_intermediate_size() != p.intermediate_size);
}

TEST_CASE("glm5_next moe: an incoherent routing geometry is REFUSED by name") {
  const gn::MoeDims ok = BigDims();
  CHECK_NOTHROW(ok.Validate());

  auto refuses = [](gn::MoeDims d) { CHECK_THROWS_AS(d.Validate(), std::exception); };

  { gn::MoeDims d = ok; d.hidden_size = 0; refuses(d); }
  { gn::MoeDims d = ok; d.n_routed_experts = 0; refuses(d); }
  { gn::MoeDims d = ok; d.num_experts_per_tok = 0; refuses(d); }
  { gn::MoeDims d = ok; d.moe_intermediate_size = 0; refuses(d); }
  { gn::MoeDims d = ok; d.n_group = 0; refuses(d); }
  { gn::MoeDims d = ok; d.topk_group = 0; refuses(d); }
  { gn::MoeDims d = ok; d.n_shared_experts = -1; refuses(d); }
  // top_k above the expert count would make the router select an id it cannot
  // fill; upstream's `torch.topk` raises, ours names the model.
  { gn::MoeDims d = ok; d.num_experts_per_tok = d.n_routed_experts + 1; refuses(d); }
  // A group count that does not divide the experts leaves a ragged last group.
  { gn::MoeDims d = ok; d.n_group = 7; refuses(d); }
  { gn::MoeDims d = ok; d.n_group = 2; d.topk_group = 3; refuses(d); }
  // `clamp(max=0)` is not "no clamp": it zeroes every positive gate, so the
  // layer emits zeros through silu and never crashes.
  { gn::MoeDims d = ok; d.swiglu_limit = 0.0f; refuses(d); }
  { gn::MoeDims d = ok; d.swiglu_limit = -10.0f; refuses(d); }
}

TEST_CASE("glm5_next moe: the router GEMM reproduces the oracle's fp32 logits") {
  // An oracle whose identity is not asserted is an oracle nobody can reproduce,
  // and every number below is the return value of THIS revision. `5.16.1` is
  // this row's lane pin (W0/#2096, `.agents/oracles/transformers.md`); the
  // registry pin is `5.14.1`, which does not carry `models/glm5_next` at all.
  // Regenerating the fixture against a different revision must move this line,
  // not pass in silence. Mirrors `test_glm5_next_mhc.cpp:92`.
  CHECK(std::string(g::kOracle) == "transformers 5.16.1");

  const gn::MoeDims d = BigDims();
  const std::vector<float> hidden = Vec(g::kHidden);
  const std::vector<float> logits =
      gn::RouterLogits(d, hidden, Vec(g::kRouterWeight), g::kSeq);

  REQUIRE(static_cast<int64_t>(logits.size()) == g::kSeq * g::kNumRoutedExperts);
  for (size_t i = 0; i < logits.size(); ++i) {
    CHECK(logits[i] == doctest::Approx(g::kRouterLogits[i]).epsilon(kTol));
  }
}

TEST_CASE("glm5_next moe: the grouped noaux_tc selection is the oracle's SET") {
  const gn::MoeDims d = BigDims();
  const std::vector<float> hidden = Vec(g::kHidden);
  vt::Queue q = CpuQueue();
  const int64_t K = g::kNumExpertsPerTok;

  SUBCASE("bias buffer at its constructor ZEROS") {
    const gn::MoeRouting r = gn::RouteTopk(d, BigWeights(/*with_bias=*/false), hidden,
                                           g::kSeq, q);
    for (int64_t t = 0; t < g::kSeq; ++t) {
      const std::set<int32_t> ours = SetOf(r.topk_ids, t, K);
      const std::set<int32_t> theirs = SetOf(g::kTopkIndicesNoBias, t, K);
      INFO("token " << t);
      CHECK(ours == theirs);
      for (int32_t id : theirs) {
        CHECK(WeightOf(r.topk_ids, r.topk_weights, t, K, id) ==
              doctest::Approx(WeightOf(g::kTopkIndicesNoBias, g::kTopkWeightsNoBias, t,
                                       K, id))
                  .epsilon(kTol));
      }
    }
  }

  SUBCASE("bias buffer FILLED — the bias SELECTS, the unbiased score WEIGHTS") {
    const gn::MoeRouting r = gn::RouteTopk(d, BigWeights(/*with_bias=*/true), hidden,
                                           g::kSeq, q);
    int moved = 0;
    for (int64_t t = 0; t < g::kSeq; ++t) {
      const std::set<int32_t> ours = SetOf(r.topk_ids, t, K);
      const std::set<int32_t> theirs = SetOf(g::kTopkIndicesBias, t, K);
      INFO("token " << t);
      CHECK(ours == theirs);
      for (int32_t id : theirs) {
        CHECK(WeightOf(r.topk_ids, r.topk_weights, t, K, id) ==
              doctest::Approx(
                  WeightOf(g::kTopkIndicesBias, g::kTopkWeightsBias, t, K, id))
                  .epsilon(kTol));
      }
      if (theirs != SetOf(g::kTopkIndicesNoBias, t, K)) ++moved;
    }
    // WITHOUT this the case above and this one could both pass on a router that
    // ignores the bias entirely. The fixture asserts the same fact on the python
    // side; this is the C++ half of it.
    CHECK(moved > 0);
  }

  SUBCASE("the SEPARATION MARGIN, printed rather than tolerated") {
    // A discrete selection has bimodal error, so the number that says how
    // trustworthy the SET assertion is, is the gap between the last accepted and
    // the best rejected SELECTION score. Recomputed here from our own logits and
    // checked against the oracle's, so the fixture's `kSelectionMargin` is
    // consumed rather than decorative.
    const gn::MoeRouting r = gn::RouteTopk(d, BigWeights(/*with_bias=*/true), hidden,
                                           g::kSeq, q);
    const std::vector<float> bias = Vec(g::kBias);
    for (int64_t t = 0; t < g::kSeq; ++t) {
      const std::set<int32_t> sel = SetOf(r.topk_ids, t, K);
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (int64_t e = 0; e < g::kNumRoutedExperts; ++e) {
        const float logit = r.router_logits[static_cast<size_t>(t * g::kNumRoutedExperts + e)];
        const float score = 1.0f / (1.0f + std::exp(-logit)) + bias[static_cast<size_t>(e)];
        if (sel.count(static_cast<int32_t>(e)) != 0) {
          lo = std::min(lo, score);
        } else {
          hi = std::max(hi, score);
        }
      }
      const float margin = lo - hi;
      MESSAGE("token " << t << ": selection margin (last accepted - best rejected) = "
                       << margin << ", oracle " << g::kSelectionMargin[t]);
      CHECK(margin > 0.0f);
      CHECK(margin == doctest::Approx(g::kSelectionMargin[t]).epsilon(kTol));
    }
  }

  SUBCASE("at n_group == 1 an ABSENT bias and a ZERO bias agree") {
    // Documented in `glm5_next_moe.h`: an empty vector selects the seam's
    // no-bias arm, whose group score is the group MAX rather than the sum of the
    // top two. At one group the mask is all-ones either way, so the selection is
    // identical -- a fact this pins rather than assumes, because a checkpoint
    // that sets `n_group > 1` would make the two arms genuinely different.
    gn::MoeLayerWeights absent = BigWeights(/*with_bias=*/false);
    absent.e_score_correction_bias.clear();
    const gn::MoeRouting a = gn::RouteTopk(d, absent, hidden, g::kSeq, q);
    const gn::MoeRouting z = gn::RouteTopk(d, BigWeights(/*with_bias=*/false), hidden,
                                           g::kSeq, q);
    CHECK(a.topk_ids == z.topk_ids);
    for (size_t i = 0; i < a.topk_weights.size(); ++i) {
      CHECK(a.topk_weights[i] == doctest::Approx(z.topk_weights[i]).epsilon(kTol));
    }
  }
}

TEST_CASE("glm5_next moe: the router refuses a bias of the wrong length and a device queue") {
  const gn::MoeDims d = BigDims();
  const std::vector<float> hidden = Vec(g::kHidden);
  vt::Queue q = CpuQueue();

  gn::MoeLayerWeights w = BigWeights(/*with_bias=*/true);
  w.e_score_correction_bias.pop_back();
  CHECK_THROWS_AS(gn::RouteTopk(d, w, hidden, g::kSeq, q), std::exception);

  // A CUDA queue with host pointers is a crash rather than a fallback, so the
  // host reference refuses by name. The device arm is the assembled forward's.
  vt::Queue cuda{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  CHECK_THROWS_AS(gn::RouteTopk(d, BigWeights(true), hidden, g::kSeq, cuda),
                  std::exception);
}

TEST_CASE("glm5_next moe: the clamped SwiGLU epilogue, INCLUDING a row that clamps") {
  const int64_t I = g::kMoeIntermediate;
  const float limit = g::kSwigluLimit;

  // The instrument states what it is measuring: row 1 of the fixture is the
  // discriminating one and this asserts that it actually is, rather than
  // trusting the generator's comment.
  bool gate_over = false, gate_under = false, up_over = false, up_under = false;
  for (int64_t i = 0; i < I; ++i) {
    const float gate = g::kGateUp[2 * I + i];
    const float up = g::kGateUp[2 * I + I + i];
    gate_over = gate_over || gate > limit;
    gate_under = gate_under || gate < -limit;
    up_over = up_over || up > limit;
    up_under = up_under || up < -limit;
  }
  const bool discriminating = gate_over && gate_under && up_over && up_under;
  REQUIRE_MESSAGE(discriminating,
                  "fixture row 1 must exceed +-limit in BOTH halves or this case "
                  "cannot separate a max-only gate clamp from a two-sided one");

  for (int64_t r = 0; r < 2; ++r) {
    const std::vector<float> row = Row(g::kGateUp, r, 2 * I);
    const std::vector<float> out = gn::ExpertGate(row, I, limit);
    REQUIRE(static_cast<int64_t>(out.size()) == I);
    for (int64_t i = 0; i < I; ++i) {
      INFO("row " << r << " channel " << i);
      CHECK(out[static_cast<size_t>(i)] ==
            doctest::Approx(g::kGateOut[r * I + i]).epsilon(kTol));
    }
  }

  SUBCASE("a SYMMETRIC gate clamp is a different function, and row 1 shows it") {
    // The asymmetry is upstream's (`:139-140`: the gate takes `min=None`, the up
    // takes both bounds) and is the single easiest thing to get wrong here. A
    // port that clamps the gate on both sides is smooth, plausible, and wrong on
    // exactly the strongly-negative channels.
    const std::vector<float> row = Row(g::kGateUp, 1, 2 * I);
    std::vector<float> sym = row;
    for (int64_t i = 0; i < I; ++i) {
      sym[static_cast<size_t>(i)] = std::max(sym[static_cast<size_t>(i)], -limit);
    }
    const std::vector<float> a = gn::ExpertGate(row, I, limit);
    const std::vector<float> b = gn::ExpertGate(sym, I, limit);
    bool differs = false;
    for (int64_t i = 0; i < I; ++i) {
      differs = differs || std::fabs(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]) > 1e-4f;
    }
    CHECK(differs);
  }

  SUBCASE("a misshaped fused row and a non-positive limit are REFUSED") {
    std::vector<float> row = Row(g::kGateUp, 0, 2 * I);
    row.pop_back();
    CHECK_THROWS_AS(gn::ExpertGate(row, I, limit), std::exception);
    CHECK_THROWS_AS(gn::ExpertGate(Row(g::kGateUp, 0, 2 * I), I, 0.0f), std::exception);
    CHECK_THROWS_AS(gn::ExpertGate(Row(g::kGateUp, 0, 2 * I), 0, limit), std::exception);
  }
}

TEST_CASE("glm5_next moe: the DENSE feed-forward matches the oracle, clamp included") {
  gn::DenseMlpWeights w;
  w.gate_proj = Vec(g::kDenseGate);
  w.up_proj = Vec(g::kDenseUp);
  w.down_proj = Vec(g::kDenseDown);
  const std::vector<float> out =
      gn::DenseMlpForward(w, Vec(g::kDenseIn), g::kHiddenSize, g::kIntermediate,
                          g::kSeq, g::kSwigluLimit);
  REQUIRE(static_cast<int64_t>(out.size()) == g::kSeq * g::kHiddenSize);
  for (size_t i = 0; i < out.size(); ++i) {
    CHECK(out[i] == doctest::Approx(g::kDenseOut[i]).epsilon(kBlockTol));
  }

  SUBCASE("the clamp is REACHED — an unclamped dense MLP gives a different answer") {
    // The generator asserts on the python side that the pre-activations exceed
    // the limit. This is the C++ half: at a limit large enough never to bite,
    // the layer emits something else, so the golden is gating the clamp and not
    // merely the two GEMMs.
    const std::vector<float> loose =
        gn::DenseMlpForward(w, Vec(g::kDenseIn), g::kHiddenSize, g::kIntermediate,
                            g::kSeq, 1.0e9f);
    bool differs = false;
    for (size_t i = 0; i < out.size(); ++i) {
      differs = differs || std::fabs(out[i] - loose[i]) > 1e-3f;
    }
    CHECK(differs);
  }
}

TEST_CASE("glm5_next moe: the composed block is routed + shared, the shared UNSCALED") {
  const gn::MoeDims d = SmallDims();
  const gn::MoeLayerWeights w = SmallWeights();
  const std::vector<float> in = Vec(g::kMoeIn);
  vt::Queue q = CpuQueue();

  const std::vector<float> out = gn::MoeForward(d, w, in, g::kSeq, q);
  REQUIRE(static_cast<int64_t>(out.size()) == g::kSeq * g::kHiddenSize);
  for (size_t i = 0; i < out.size(); ++i) {
    INFO("element " << i);
    CHECK(out[i] == doctest::Approx(g::kMoeOut[i]).epsilon(kBlockTol));
  }

  SUBCASE("the SMALL router's selection is the oracle's set too") {
    const gn::MoeRouting r = gn::RouteTopk(d, w, in, g::kSeq, q);
    for (int64_t t = 0; t < g::kSeq; ++t) {
      INFO("token " << t);
      CHECK(SetOf(r.topk_ids, t, g::kSmallTopK) ==
            SetOf(g::kSmallTopkIndices, t, g::kSmallTopK));
      for (int32_t id : SetOf(g::kSmallTopkIndices, t, g::kSmallTopK)) {
        CHECK(WeightOf(r.topk_ids, r.topk_weights, t, g::kSmallTopK, id) ==
              doctest::Approx(WeightOf(g::kSmallTopkIndices, g::kSmallTopkWeights, t,
                                       g::kSmallTopK, id))
                  .epsilon(kTol));
      }
    }
  }

  SUBCASE("the SHARED term is the oracle's, at moe_intermediate * n_shared") {
    const std::vector<float> shared =
        gn::DenseMlpForward(w.shared, in, g::kHiddenSize, d.shared_intermediate_size(),
                            g::kSeq, d.swiglu_limit);
    for (size_t i = 0; i < shared.size(); ++i) {
      CHECK(shared[i] == doctest::Approx(g::kSharedOut[i]).epsilon(kBlockTol));
    }
    // And the ROUTED half separately, so a block that scaled the shared term by
    // `routed_scaling_factor` -- or folded the factor in twice -- cannot hide
    // inside the sum.
    for (size_t i = 0; i < out.size(); ++i) {
      CHECK(out[i] - shared[i] == doctest::Approx(g::kRoutedOut[i]).epsilon(kBlockTol));
    }
  }

  SUBCASE("applying routed_scaling_factor TWICE is a different, plausible answer") {
    // `routed_scaling_factor` is folded into the router weights
    // (`modeling_glm5_next.py:182`), so `vt::MoeCombine` keeps its default
    // `routed_scale = 1.0f`. Passing it in both places squares it: the block
    // still runs, still routes to the same experts, and is wrong by 2.5x on the
    // routed half only.
    gn::MoeDims twice = d;
    twice.routed_scaling_factor = d.routed_scaling_factor * d.routed_scaling_factor;
    const std::vector<float> bad = gn::MoeForward(twice, w, in, g::kSeq, q);
    bool differs = false;
    for (size_t i = 0; i < out.size(); ++i) {
      differs = differs || std::fabs(out[i] - bad[i]) > 1e-3f;
    }
    CHECK(differs);
  }

  SUBCASE("a shared expert sized from intermediate_size instead is REFUSED") {
    // The width mismatch is caught as a weight-shape refusal rather than being
    // absorbed, because `DenseMlpForward` checks the projection against the
    // width it was told.
    CHECK_THROWS_AS(gn::DenseMlpForward(w.shared, in, g::kHiddenSize, g::kIntermediate,
                                        g::kSeq, d.swiglu_limit),
                    std::exception);
  }

  SUBCASE("a STACKED expert tower of the wrong shape is REFUSED by name") {
    gn::MoeLayerWeights bad = w;
    bad.expert_gate_up.pop_back();
    CHECK_THROWS_AS(gn::MoeForward(d, bad, in, g::kSeq, q), std::exception);
    gn::MoeLayerWeights bad2 = w;
    bad2.expert_down.pop_back();
    CHECK_THROWS_AS(gn::MoeForward(d, bad2, in, g::kSeq, q), std::exception);
  }
}

// ─── W9a: the routed-expert GEMM on the shared keep-quant seam ───────────────
//
// The arm this section gates replaces a hand-written per-expert host-f32 matvec
// pair with `vt::MoeGateUpSwiGLUGrouped` + `vt::MatmulBTQuantGrouped`. The
// argument for it is in `MoeQuantBanks` (glm5_next_moe.h) and §W9a of the spec.
//
// WHY THESE CASES COMPARE TWO OF OUR OWN ARMS AND NOT AN ORACLE. The goldens
// above are the transformers reference and they gate the BLOCK. What is new
// here is not the function computed, it is which seam computes it, and the two
// seams differ by exactly one thing: the keep-quant arm quantizes the
// ACTIVATION to Q8_K once per call and the f32 arm does not. So the operand
// this section needs is our own f32 arm on the SAME weights, and the oracle
// comparison stays where it already is.
//
// AND A BAND ON ITS OWN WOULD BE A MUTE SWITCH. Every wrong answer available
// here — the wrong expert row, the wrong token's activation, the two towers
// swapped — is an O(1) error, while the activation quantization is O(1e-3). A
// tolerance that admits the second admits nothing about the first unless the
// separation is measured, so `the band DISCRIMINATES` below measures it rather
// than asserting the band and hoping.
namespace {

// `block_q2_K` (llama.cpp b10451 ggml-common.h): scales[16], qs[64], d at 80,
// dmin at 82, 84 bytes for 256 elements. Q2_K is one of the four
// (gate, up, down) encoding triples the published UD-Q2_K_XL arm actually
// contains — blk.45 is (Q2_K, Q2_K, Q3_K) — so this is the checkpoint's own
// encoding and not a convenient stand-in.
constexpr int64_t kQ2KBlockElems = 256;
constexpr int64_t kQ2KBlockBytes = 84;
constexpr int kQ2KDOff = 80;
constexpr int kQ2KDminOff = 82;

// Random block payload with SANE super-block scales, exactly as
// `tests/vt/test_cuda_quant_dot.cpp` builds its weight cases: uniformly random
// bits in the quant and scale nibbles, but `d`/`dmin` written as real fp16
// magnitudes. Random bits in an fp16 scale field are a NaN or a 6e4 half the
// time, and that would measure the fixture rather than the seam.
std::vector<uint8_t> Q2KBlocks(int64_t nblocks, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * kQ2KBlockBytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFFU);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + static_cast<size_t>(i * kQ2KBlockBytes);
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    const uint16_t d = vt::F32ToF16(0.0125F * jitter);
    const uint16_t dmin = vt::F32ToF16(0.0075F * jitter);
    std::memcpy(blk + kQ2KDOff, &d, sizeof(d));
    std::memcpy(blk + kQ2KDminOff, &dmin, sizeof(dmin));
  }
  return bytes;
}

vllm::OwnedTensor Q2KBank(int64_t e, int64_t n, int64_t k, uint32_t seed) {
  REQUIRE(k % kQ2KBlockElems == 0);
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kQ2_K;
  t.rank = 3;
  t.shape[0] = e;
  t.shape[1] = n;
  t.shape[2] = k;
  t.nk = true;
  const std::vector<uint8_t> src =
      Q2KBlocks(e * n * (k / kQ2KBlockElems), seed);
  t.bytes.assign(src.size(), 0U);
  std::memcpy(t.bytes.data(), src.data(), src.size());
  return t;
}

vllm::OwnedTensor F32Tensor(std::initializer_list<int64_t> shape, uint32_t seed,
                            float scale) {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kF32;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  int i = 0;
  for (int64_t s : shape) {
    t.shape[i++] = s;
    n *= s;
  }
  std::mt19937 rng(seed);
  std::vector<float> v(static_cast<size_t>(n));
  for (float& x : v)
    x = scale * (static_cast<float>(rng() % 2001) / 1000.0F - 1.0F);
  t.bytes.assign(v.size() * sizeof(float), 0U);
  std::memcpy(t.bytes.data(), v.data(), v.size() * sizeof(float));
  return t;
}

// Small, and every dimension chosen for a reason. K on both GEMMs must be a
// whole number of 256-element Q8_K super-blocks, so `hidden_size` and
// `moe_intermediate_size` are both 256. `num_experts_per_tok` 2 of 4 keeps more
// than one slot per token, which is what makes the `p / K` vs `p % K` gather
// observable at all. `n_group`/`topk_group` stay 1 as the published config has
// them.
gn::MoeDims KqDims() {
  gn::MoeDims d;
  d.hidden_size = 256;
  d.n_routed_experts = 4;
  d.n_shared_experts = 1;
  d.num_experts_per_tok = 2;
  d.moe_intermediate_size = 256;
  d.n_group = 1;
  d.topk_group = 1;
  d.routed_scaling_factor = 2.5;
  d.norm_topk_prob = true;
  d.swiglu_limit = 10.0F;
  d.Validate();
  return d;
}

vllm::Glm5NextMoeWeights KqSource(const gn::MoeDims& d) {
  vllm::Glm5NextMoeWeights s;
  s.router = F32Tensor({d.n_routed_experts, d.hidden_size}, 0x8017EU, 0.05F);
  s.e_score_correction_bias = F32Tensor({d.n_routed_experts}, 0xB1A5U, 0.01F);
  s.gate_exps = Q2KBank(d.n_routed_experts, d.moe_intermediate_size,
                        d.hidden_size, 0x6A7EU);
  s.up_exps = Q2KBank(d.n_routed_experts, d.moe_intermediate_size,
                      d.hidden_size, 0x0F00U);
  s.down_exps = Q2KBank(d.n_routed_experts, d.hidden_size,
                        d.moe_intermediate_size, 0xD0002U);
  const int64_t si = d.shared_intermediate_size();
  s.shared.gate_proj = F32Tensor({si, d.hidden_size}, 0x5111U, 0.03F);
  s.shared.up_proj = F32Tensor({si, d.hidden_size}, 0x5222U, 0.03F);
  s.shared.down_proj = F32Tensor({d.hidden_size, si}, 0x5333U, 0.03F);
  return s;
}

// Normalized mean squared error between two runs of the same block.
double Nmse(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::isfinite(a[i]));
    REQUIRE(std::isfinite(b[i]));
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += diff * diff;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return den > 0.0 ? num / den : num;
}

// The activation-quantization band. It is NOT the discriminating assertion;
// `the band DISCRIMINATES` below is. Q8_K gives the activation 8 bits and one
// scale per 256 elements, and the block chains two such GEMMs, so a few parts
// per thousand is the expected agreement between an f32-activation dot and an
// integer block dot over the identical weights.
constexpr double kKeepQuantBand = 5e-3;

}  // namespace

TEST_CASE("glm5_next moe keep-quant: the seam arm agrees with the f32 arm") {
  const gn::MoeDims d = KqDims();
  const vllm::Glm5NextMoeWeights src = KqSource(d);
  vt::Queue q = CpuQueue();

  gn::MoeLayerWeights w = gn::BridgeMoeLayer(src, d, "moe");
  // The bridge ADMITTED the banks. If this ever reads false the two arms below
  // are the same arm and every comparison in this file becomes a tautology, so
  // it is asserted rather than assumed.
  REQUIRE(w.has_quant_banks);
  gn::GgufExpertSource source(src, d, "moe");
  w.expert_source = &source;

  // The f32 operand: the SAME struct with the keep-quant arm switched off, so
  // the only difference between the two runs is which seam ran.
  gn::MoeLayerWeights wf = w;
  wf.has_quant_banks = false;
  wf.expert_source = &source;

  constexpr int64_t kT = 3;
  std::mt19937 rng(0xBEEF);
  std::vector<float> hidden(static_cast<size_t>(kT * d.hidden_size));
  for (float& x : hidden)
    x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F;

  // THE KEEP-QUANT ARM IS ASSERTED TO HAVE RUN, not assumed. Without this the
  // whole case is a tautology under the one mutation that matters: disable the
  // branch and BOTH runs become the f32 arm, the NMSE is exactly 0, and every
  // band below passes while the seam is never called.
  //
  // `GgufExpertSource::decoded()` is the instrument and it is a direct one. The
  // keep-quant arm never consults the source, so an empty log after the first
  // run IS the statement "no expert was decoded to f32 this step" — which is
  // the whole point of the wave — and a non-empty log after the second is the
  // statement that the operand it is compared against was really produced the
  // other way.
  const std::vector<float> got = gn::MoeForward(d, w, hidden, kT, q);
  CHECK(source.decoded().empty());
  const std::vector<float> ref = gn::MoeForward(d, wf, hidden, kT, q);
  CHECK_FALSE(source.decoded().empty());
  REQUIRE(got.size() == ref.size());
  REQUIRE(static_cast<int64_t>(got.size()) == kT * d.hidden_size);

  const double nmse = Nmse(got, ref);
  CAPTURE(nmse);
  CHECK(nmse <= kKeepQuantBand);
  for (float x : got) CHECK(std::isfinite(x));
  // The f32 arm was actually driven — a zeroed `ref` would make any band pass.
  double ref_energy = 0.0;
  for (float x : ref) ref_energy += static_cast<double>(x) * x;
  CHECK(ref_energy > 0.0);
}

TEST_CASE("glm5_next moe keep-quant: the band DISCRIMINATES") {
  // The band above admits the activation quantization. This case measures what
  // it REFUSES, because a band nobody has separated from a real defect is a
  // mute switch. The defect injected is the one the arm is most likely to have:
  // the routed slots read the wrong expert row.
  const gn::MoeDims d = KqDims();
  const vllm::Glm5NextMoeWeights src = KqSource(d);
  vt::Queue q = CpuQueue();

  gn::MoeLayerWeights w = gn::BridgeMoeLayer(src, d, "moe");
  REQUIRE(w.has_quant_banks);
  gn::GgufExpertSource source(src, d, "moe");
  w.expert_source = &source;
  gn::MoeLayerWeights wf = w;
  wf.has_quant_banks = false;

  constexpr int64_t kT = 3;
  std::mt19937 rng(0xBEEF);
  std::vector<float> hidden(static_cast<size_t>(kT * d.hidden_size));
  for (float& x : hidden)
    x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F;

  const std::vector<float> ref = gn::MoeForward(d, wf, hidden, kT, q);
  const std::vector<float> good = gn::MoeForward(d, w, hidden, kT, q);

  // ROTATE THE EXPERT TOWERS BY ONE EXPERT. Every shape, dtype, byte count and
  // block boundary is unchanged; only which expert a slot reads moves. This is
  // the shape of a real indexing defect, and it is what the band has to be able
  // to see.
  vllm::Glm5NextMoeWeights rotated = src;
  const size_t stride = static_cast<size_t>(
      d.moe_intermediate_size * (d.hidden_size / kQ2KBlockElems) *
      kQ2KBlockBytes);
  auto rotate = [&](vllm::OwnedTensor& t) {
    std::vector<uint8_t> b(t.bytes.size());
    std::memcpy(b.data(), t.bytes.data(), t.bytes.size());
    std::rotate(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(stride),
                b.end());
    std::memcpy(t.bytes.data(), b.data(), b.size());
  };
  rotate(rotated.gate_exps);
  rotate(rotated.up_exps);
  rotate(rotated.down_exps);

  gn::MoeLayerWeights wr = gn::BridgeMoeLayer(rotated, d, "moe");
  REQUIRE(wr.has_quant_banks);
  const std::vector<float> wrong = gn::MoeForward(d, wr, hidden, kT, q);

  const double nmse_good = Nmse(good, ref);
  const double nmse_wrong = Nmse(wrong, ref);
  CAPTURE(nmse_good);
  CAPTURE(nmse_wrong);
  // The separation, printed rather than implied. If a later change narrows this
  // to single digits the band has stopped gating and the reader can see it.
  const double separation = nmse_wrong / std::max(nmse_good, 1e-12);
  CAPTURE(separation);
  CHECK(nmse_wrong > kKeepQuantBand * 20.0);
  CHECK(separation > 20.0);
}

TEST_CASE("glm5_next moe keep-quant: a slot reads ITS OWN token's hidden row") {
  // `act[p] = hidden[p / top_k]`. Writing `p % top_k` indexes a real token for
  // every slot, so the result stays finite, correctly shaped and fluent — the
  // one defect in this arm that no shape, dtype or finiteness assertion can
  // see. The invariant that catches it: the block is a per-token function, so
  // row t of a multi-token call must equal the single-token call on that row.
  const gn::MoeDims d = KqDims();
  const vllm::Glm5NextMoeWeights src = KqSource(d);
  vt::Queue q = CpuQueue();

  gn::MoeLayerWeights w = gn::BridgeMoeLayer(src, d, "moe");
  REQUIRE(w.has_quant_banks);
  // Attached only as an instrument. If the keep-quant branch stops being taken,
  // this case would otherwise silently measure the f32 arm instead.
  gn::GgufExpertSource source(src, d, "moe");
  w.expert_source = &source;

  constexpr int64_t kT = 4;
  std::mt19937 rng(0x5107);
  std::vector<float> hidden(static_cast<size_t>(kT * d.hidden_size));
  for (float& x : hidden)
    x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F;

  const std::vector<float> many = gn::MoeForward(d, w, hidden, kT, q);
  REQUIRE(static_cast<int64_t>(many.size()) == kT * d.hidden_size);

  int64_t rows_checked = 0;
  for (int64_t t = 0; t < kT; ++t) {
    const std::vector<float> one(
        hidden.begin() + static_cast<std::ptrdiff_t>(t * d.hidden_size),
        hidden.begin() + static_cast<std::ptrdiff_t>((t + 1) * d.hidden_size));
    const std::vector<float> got = gn::MoeForward(d, w, one, 1, q);
    REQUIRE(static_cast<int64_t>(got.size()) == d.hidden_size);
    for (int64_t h = 0; h < d.hidden_size; ++h) {
      CAPTURE(t);
      CAPTURE(h);
      CHECK(many[static_cast<size_t>(t * d.hidden_size + h)] ==
            doctest::Approx(got[static_cast<size_t>(h)]).epsilon(1e-5));
    }
    ++rows_checked;
  }
  CHECK(rows_checked == kT);

  // The rows are DIFFERENT from one another, so the case above cannot pass by
  // the block emitting one constant row. Without this, a forward that ignored
  // its input entirely would satisfy every assertion in this case.
  double spread = 0.0;
  for (int64_t h = 0; h < d.hidden_size; ++h) {
    spread += std::fabs(many[static_cast<size_t>(h)] -
                        many[static_cast<size_t>(d.hidden_size + h)]);
  }
  CAPTURE(spread);
  CHECK(spread > 0.0);
  // Every call above went through the seam and none of them decoded an expert.
  CHECK(source.decoded().empty());
}

TEST_CASE("glm5_next moe keep-quant: the swiglu_limit REACHES the seam") {
  // MEASURED, then fixed here rather than disclosed. Mutation M4 replaces
  // `d.swiglu_limit` with +infinity at the seam call site, and every case above
  // SURVIVED it: this fixture's weights and activations never drive a pre-clamp
  // value past a limit of 10, so a comparison that never sees the clamp engage
  // cannot gate whether the clamp was passed at all. A band is only a gate for
  // the defects its data actually exercises.
  //
  // THE GATE IS A DIFFERENCE, NOT A VALUE. Run identical inputs at two limits,
  // one tight enough to bite and one loose, and require the outputs to differ.
  // A call site that drops the argument returns the same numbers twice whatever
  // the epilogue computes, and no golden is needed to see that.
  //
  // THE SHARED EXPERT IS ZEROED, and that is the load-bearing part of the
  // fixture. `DenseMlpForward` takes the SAME limit, so with a live shared term
  // the two runs would differ through the shared path alone and the case would
  // pass even under M4 -- gating nothing while looking strict. With the shared
  // projections zero its contribution is zero at any limit, so every difference
  // below is the routed seam's.
  vllm::Glm5NextMoeWeights src = KqSource(KqDims());
  const int64_t si = KqDims().shared_intermediate_size();
  const int64_t H = KqDims().hidden_size;
  src.shared.gate_proj = F32Tensor({si, H}, 0x2001U, 0.0F);
  src.shared.up_proj = F32Tensor({si, H}, 0x2002U, 0.0F);
  src.shared.down_proj = F32Tensor({H, si}, 0x2003U, 0.0F);

  gn::MoeDims tight = KqDims();
  tight.swiglu_limit = 0.05F;
  gn::MoeDims loose = KqDims();
  loose.swiglu_limit = 64.0F;

  vt::Queue q = CpuQueue();
  gn::MoeLayerWeights wt = gn::BridgeMoeLayer(src, tight, "moe");
  gn::MoeLayerWeights wl = gn::BridgeMoeLayer(src, loose, "moe");
  REQUIRE(wt.has_quant_banks);
  REQUIRE(wl.has_quant_banks);

  constexpr int64_t kT = 3;
  std::mt19937 rng(0xC1A3DU);
  std::vector<float> hidden(static_cast<size_t>(kT * H));
  for (float& x : hidden)
    x = static_cast<float>(rng() % 2001) / 1000.0F - 1.0F;

  const std::vector<float> a = gn::MoeForward(tight, wt, hidden, kT, q);
  const std::vector<float> b = gn::MoeForward(loose, wl, hidden, kT, q);
  REQUIRE(a.size() == b.size());

  double diff = 0.0;
  double energy = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::isfinite(a[i]));
    REQUIRE(std::isfinite(b[i]));
    diff += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    energy += std::fabs(static_cast<double>(b[i]));
  }
  CAPTURE(diff);
  CAPTURE(energy);
  // The loose arm is non-trivial, so "they differ" is not satisfied by both
  // being zero.
  CHECK(energy > 0.0);
  // A clamp at 0.05 against a loose 64 must move the block by a visible
  // fraction of its own magnitude, not by a rounding step.
  CHECK(diff > 0.01 * energy);
}

TEST_CASE("glm5_next moe keep-quant: admission declines, refuses, and admits") {
  const gn::MoeDims d = KqDims();

  SUBCASE("f32 banks DECLINE, and the f32 arm keeps running") {
    vllm::Glm5NextMoeWeights src = KqSource(d);
    src.gate_exps = F32Tensor({d.n_routed_experts, d.moe_intermediate_size,
                               d.hidden_size}, 0x1U, 0.02F);
    src.up_exps = F32Tensor({d.n_routed_experts, d.moe_intermediate_size,
                             d.hidden_size}, 0x2U, 0.02F);
    src.down_exps = F32Tensor({d.n_routed_experts, d.hidden_size,
                               d.moe_intermediate_size}, 0x3U, 0.02F);
    gn::MoeQuantBanks banks;
    CHECK_FALSE(gn::AdmitMoeQuantBanks(src, d, "moe", &banks));
  }

  SUBCASE("a gate/up dtype MISMATCH is refused BY NAME") {
    vllm::Glm5NextMoeWeights src = KqSource(d);
    src.up_exps.dtype = vt::DType::kQ6_K;  // both are block-quant, only differ
    gn::MoeQuantBanks banks;
    CHECK_THROWS_WITH_AS(gn::AdmitMoeQuantBanks(src, d, "moe", &banks),
                         doctest::Contains("requires one dtype for the pair"),
                         std::runtime_error);
  }

  SUBCASE("a REPACKED bank is refused BY NAME") {
    // The transform this refuses preserves the dtype AND the byte count, so no
    // other assertion in the bridge can see it.
    vllm::Glm5NextMoeWeights src = KqSource(d);
    src.gate_exps.repacked = true;
    gn::MoeQuantBanks banks;
    CHECK_THROWS_WITH_AS(gn::AdmitMoeQuantBanks(src, d, "moe", &banks),
                         doctest::Contains("REPACK marker"), std::runtime_error);
  }

  SUBCASE("a PARTIALLY quantized bank set is refused BY NAME") {
    vllm::Glm5NextMoeWeights src = KqSource(d);
    src.down_exps = F32Tensor({d.n_routed_experts, d.hidden_size,
                               d.moe_intermediate_size}, 0x9U, 0.02F);
    gn::MoeQuantBanks banks;
    CHECK_THROWS_WITH_AS(gn::AdmitMoeQuantBanks(src, d, "moe", &banks),
                         doctest::Contains("takes all three or none"),
                         std::runtime_error);
  }

  SUBCASE("the admitted views are the STACKED [E*N, K] towers") {
    const vllm::Glm5NextMoeWeights src = KqSource(d);
    gn::MoeQuantBanks banks;
    REQUIRE(gn::AdmitMoeQuantBanks(src, d, "moe", &banks));
    CHECK(banks.gate.rank == 2);
    CHECK(banks.gate.shape[0] == d.n_routed_experts * d.moe_intermediate_size);
    CHECK(banks.gate.shape[1] == d.hidden_size);
    CHECK(banks.down.shape[0] == d.n_routed_experts * d.hidden_size);
    CHECK(banks.down.shape[1] == d.moe_intermediate_size);
    CHECK(banks.gate.dtype == vt::DType::kQ2_K);
    CHECK(banks.up.dtype == vt::DType::kQ2_K);
    CHECK(banks.down.dtype == vt::DType::kQ2_K);
    // The view aims at the bank's own bytes: no copy, no decode.
    CHECK(banks.gate.data == static_cast<const void*>(src.gate_exps.bytes.data()));
  }
}
