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
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "vllm/model_executor/models/deepseek_v4_moe.h"
#include "vllm/model_executor/models/glm5_next.h"
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
