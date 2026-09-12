// DeepSeek-V4-Flash W6 UNIT GATE — the sqrtsoftplus + hash-routed MoE: the V4
// router score function (sqrt∘softplus), the noaux_tc bias-for-selection top-k,
// the hash `tid2eid` route that bypasses top-k, and the clamped SwiGLU expert
// activation.
//
// HONEST scope (mirrors W3/W4/W5): the full-model gate is multi-Spark-blocked
// (the V4 checkpoint is 156.7 GiB, does not fit one GB10; the forward also needs
// the device kernels + assembly (W7)). So W6 gates the MATH per-primitive against
// (a) HAND-DERIVED small cases with literal expected numbers I verify by hand from
// the vLLM eager reference, and (b) from-first-principles DOUBLE-PRECISION
// references on randomized shapes (rel-L2 / independent recompute). This is the
// "host-reference + hand-case + structural-review" bar named in the W6 brief, NOT
// a dumped-oracle rel-L2 (the fixed-config 167B arch cannot be constructed at a
// tiny shape).
//
// The three load-bearing nuances, each pinned by a dedicated case that FAILS
// under the obvious wrong implementation (RED-first proven):
//   - the score is sqrt(softplus(·)) — the COMPOSITION (dropping sqrt diverges);
//   - the bias affects SELECTION ONLY, weights come from the UNBIASED scores
//     (gathering from biased scores diverges);
//   - the SwiGLU clamp is ASYMMETRIC (gate max-only, up both-sided; symmetric-
//     clamping the gate diverges);
//   - the hash route BYPASSES top-k (a top-k impl picks different experts).
//
// Ported from: vllm/model_executor/layers/fused_moe/router/
// fused_topk_bias_router.py:75-118 (`_topk_softplus_sqrt_torch`) + :88 (score),
// activation.py:197-201 (`SiluAndMulWithClamp.forward_native`); upstream tests
// tests/kernels/moe/test_topk_softplus_sqrt.py + tests/kernels/core/
// test_activation.py; cross-checked SGLang v0.5.15
// python/sglang/srt/layers/moe/{topk.py:1013-1014, hash_topk.py:137-180}. All @
// vLLM pin 555967922.
#include "vllm/model_executor/models/deepseek_v4_moe.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace vllm::deepseek_v4;

namespace {
// Relative L2: ||a-b||_2 / max(||b||_2, eps). Reference (b) in double.
double RelL2(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

// Independent DOUBLE-PRECISION references (from-first-principles, NOT by calling
// the f32 impl) so an agreeing rel-L2 proves the port is not a transcription bug.
double SqrtSoftplusD(double x) {
  const double sp = std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
  return std::sqrt(sp);
}
double SigmoidD(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Logit x such that softplus(x) == s exactly (so sqrt(softplus(x)) == sqrt(s)).
// softplus(x) = log(1+exp(x)) = s  ⇒  x = log(exp(s) - 1).
float LogitForSoftplus(double s) { return static_cast<float>(std::log(std::expm1(s))); }
}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// (1) sqrtsoftplus score function
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-moe: SqrtSoftplus at x=0 is sqrt(ln2)") {
  // softplus(0) = ln(1+1) = ln2 ; sqrt(ln2) = 0.8325546...
  CHECK(SqrtSoftplus(0.0f) == doctest::Approx(std::sqrt(std::log(2.0))));
}

TEST_CASE("dsv4-moe: SqrtSoftplus COMPOSITION is load-bearing (sqrt∘softplus)") {
  // x chosen so softplus(x) == 4 exactly ⇒ sqrt(4) == 2. This pins BOTH halves:
  // softplus alone would give 4.0 (not 2.0), sqrt-of-raw-logit would give
  // sqrt(log(exp4-1)) ≈ 1.9953 (not 2.0). Dropping the sqrt in the impl (the
  // RED-first lever) makes this case fail.
  const float x4 = LogitForSoftplus(4.0);  // log(exp(4)-1) ≈ 3.98143
  CHECK(SqrtSoftplus(x4) == doctest::Approx(2.0));
  // And softplus(x)==1 ⇒ sqrt(1)==1.
  CHECK(SqrtSoftplus(LogitForSoftplus(1.0)) == doctest::Approx(1.0));
  // Distinct from the raw logit sqrt (guards against "no softplus" impl too).
  CHECK(SqrtSoftplus(x4) != doctest::Approx(std::sqrt(x4)));
}

TEST_CASE("dsv4-moe: SqrtSoftplus is monotone increasing and matches f64") {
  std::mt19937 rng(0x59F1);
  std::uniform_real_distribution<float> U(-8.0f, 8.0f);
  float prev = SqrtSoftplus(-30.0f);
  for (float x = -29.5f; x <= 30.0f; x += 0.5f) {
    const float s = SqrtSoftplus(x);
    CHECK(s >= prev - 1e-6f);  // non-decreasing
    prev = s;
  }
  for (int i = 0; i < 200; ++i) {
    const float x = U(rng);
    CHECK(static_cast<double>(SqrtSoftplus(x)) == doctest::Approx(SqrtSoftplusD(x)).epsilon(1e-6));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (2) the sqrtsoftplus / hash router
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-moe: bias affects SELECTION only — weights come from UNBIASED scores") {
  // E=3, topk=1. scores = [2, 1, ~0] (softplus=4/1/tiny). bias = [0, 2, 0] flips
  // the choice to expert1 (scores_for_choice = [2, 3, ~0]); the returned weight
  // is the UNBIASED score of expert1 = 1.0 (NOT 3.0). Gathering from the biased
  // scores (the RED-first lever) would return 3.0 and FAIL here.
  const std::vector<float> gating = {LogitForSoftplus(4.0), LogitForSoftplus(1.0), -20.0f};
  const std::vector<float> bias = {0.0f, 2.0f, 0.0f};
  const MoeRouteResult r = SqrtSoftplusRouteTopk(gating, 1, 3, 1, bias, /*renorm=*/false,
                                                 /*scale=*/1.0f, {}, {}, 0);
  REQUIRE(r.topk_ids.size() == 1);
  CHECK(r.topk_ids[0] == 1);                        // bias flipped the selection
  CHECK(r.topk_weights[0] == doctest::Approx(1.0)); // UNBIASED score, not 3.0

  // Without the bias, the unbiased argmax (expert0, score 2.0) is chosen instead —
  // proving the bias genuinely changed the selection above.
  const MoeRouteResult r0 = SqrtSoftplusRouteTopk(gating, 1, 3, 1, {}, false, 1.0f, {}, {}, 0);
  CHECK(r0.topk_ids[0] == 0);
  CHECK(r0.topk_weights[0] == doctest::Approx(2.0));
}

TEST_CASE("dsv4-moe: router renormalize divides by the UNBIASED weight sum") {
  // E=2, topk=2, no bias. scores = [3, 1]; descending selection [0,1]; renorm ⇒
  // [3/4, 1/4] = [0.75, 0.25].
  const std::vector<float> gating = {LogitForSoftplus(9.0), LogitForSoftplus(1.0)};
  const MoeRouteResult r = SqrtSoftplusRouteTopk(gating, 1, 2, 2, {}, /*renorm=*/true,
                                                 1.0f, {}, {}, 0);
  REQUIRE(r.topk_ids.size() == 2);
  CHECK(r.topk_ids[0] == 0);
  CHECK(r.topk_ids[1] == 1);
  CHECK(r.topk_weights[0] == doctest::Approx(0.75));
  CHECK(r.topk_weights[1] == doctest::Approx(0.25));
}

TEST_CASE("dsv4-moe: router routed_scaling_factor multiplies the weights") {
  // Same [3,1] scores, no renorm, scale=2 ⇒ weights [6, 2].
  const std::vector<float> gating = {LogitForSoftplus(9.0), LogitForSoftplus(1.0)};
  const MoeRouteResult r = SqrtSoftplusRouteTopk(gating, 1, 2, 2, {}, false, /*scale=*/2.0f,
                                                 {}, {}, 0);
  CHECK(r.topk_weights[0] == doctest::Approx(6.0));
  CHECK(r.topk_weights[1] == doctest::Approx(2.0));
}

TEST_CASE("dsv4-moe: HASH route bypasses top-k (tid2eid lookup selects experts)") {
  // vocab=4, E=4, topk=2. scores = [2, 1, 3, 1.5]. The learned top-2 would be
  // experts {2, 0}; the hash table row for token 2 forces {3, 1} instead —
  // proving the hash route BYPASSES the score-based selection. Weights are
  // gathered from the UNBIASED scores of the HASH-chosen experts = [1.5, 1.0].
  const std::vector<float> gating = {LogitForSoftplus(4.0), LogitForSoftplus(1.0),
                                     LogitForSoftplus(9.0), LogitForSoftplus(2.25)};
  // hash_indices_table [vocab=4, topk=2] row-major; only row 2 is exercised.
  std::vector<int32_t> table(8, 0);
  table[2 * 2 + 0] = 3;
  table[2 * 2 + 1] = 1;
  const std::vector<int64_t> tokens = {2};
  const MoeRouteResult r = SqrtSoftplusRouteTopk(gating, 1, 4, 2, {}, /*renorm=*/false,
                                                 1.0f, tokens, table, /*vocab=*/4);
  REQUIRE(r.topk_ids.size() == 2);
  CHECK(r.topk_ids[0] == 3);
  CHECK(r.topk_ids[1] == 1);
  CHECK(r.topk_weights[0] == doctest::Approx(1.5));  // score of expert 3
  CHECK(r.topk_weights[1] == doctest::Approx(1.0));  // score of expert 1

  // Renormalized: [1.5, 1.0] / 2.5 = [0.6, 0.4].
  const MoeRouteResult rn = SqrtSoftplusRouteTopk(gating, 1, 4, 2, {}, /*renorm=*/true,
                                                  1.0f, tokens, table, 4);
  CHECK(rn.topk_weights[0] == doctest::Approx(0.6));
  CHECK(rn.topk_weights[1] == doctest::Approx(0.4));
}

// MODEL-MM-deepseek-v4 W4 (#2411): the VISION routing bias, selected PER TOKEN.
//
// THE DECISION, and it is a deliberate divergence. At `llama-cpp-dsv4vision` the
// selection is per UBATCH -- `const bool is_media = ubatch.embd != nullptr;`,
// `pr28154.diff` @ `@@ -1275,7 +1280,14 @@` -- and when it is set EVERY layer
// takes `ffn_exp_probs_b_vl` and the `il < hparams.dsv4_hash_layer_count` branch
// is skipped WHOLESALE, so `ffn_gate_tid2eid` is never consulted.
//
// Per token is chosen, on three grounds:
//
//   1. It AGREES with the oracle on every input the oracle can express. A media
//      ubatch carries no text rows, so "every row is media" and "this row is
//      media" select identically there.
//   2. Our step is not a ubatch. This engine batches continuously, and one step
//      mixes an image request's prefill rows with other requests' decode rows.
//      A whole-step flag would route another request's TEXT tokens on the vision
//      bias, which the oracle never does on any batch it can build.
//   3. The hash question has a per-row answer, and it is the SAME answer the
//      oracle gives wholesale. A hash layer carries `exp_probs_b_vl` and no
//      `exp_probs_b`; an image row has no identifier worth hashing, so it takes
//      the vision bias and the learned route, and a text row in the same step
//      still hashes. The oracle skips the hash branch for the whole ubatch only
//      because no text row is there to keep it.
//
// THESE ARE SELECTION TESTS, not token tests. Both cases below are built so the
// vision bias and the text bias choose DIFFERENT experts, and so the hash table
// names a third set again -- a bias that changed the weights but not the choice
// would be invisible to a downstream token comparison.
TEST_CASE("dsv4-moe: the vision bias is selected PER TOKEN, not per step") {
  // E=4, topk=1. scores = [2, 1, 3, 1.5], so the UNBIASED top-1 is expert 2.
  const std::vector<float> row = {LogitForSoftplus(4.0), LogitForSoftplus(1.0),
                                  LogitForSoftplus(9.0), LogitForSoftplus(2.25)};
  std::vector<float> gating;
  for (int t = 0; t < 3; ++t) gating.insert(gating.end(), row.begin(), row.end());

  // The text bias hands expert 0 the win; the vision bias hands it to expert 1.
  // Neither is the unbiased winner, so a router that ignored the bias entirely,
  // or applied the wrong one, lands on a DIFFERENT expert in each case.
  const std::vector<float> text_bias = {5.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> vision_bias = {0.0f, 9.0f, 0.0f, 0.0f};
  // Rows 0 and 2 are text, row 1 is an image row.
  const std::vector<char> is_media = {0, 1, 0};

  const MoeRouteResult r = SqrtSoftplusRouteTopk(
      gating, 3, 4, 1, text_bias, /*renorm=*/false, 1.0f, {}, {}, /*vocab=*/4,
      vision_bias, is_media);
  REQUIRE(r.topk_ids.size() == 3);
  CHECK(r.topk_ids[0] == 0);  // text  -> text bias
  CHECK(r.topk_ids[1] == 1);  // IMAGE -> vision bias
  CHECK(r.topk_ids[2] == 0);  // text  -> text bias
  // The WEIGHT still comes from the UNBIASED scores, on both arms.
  CHECK(r.topk_weights[0] == doctest::Approx(2.0));
  CHECK(r.topk_weights[1] == doctest::Approx(1.0));

  // An EMPTY mask is every text step, and must be byte-identical to the call
  // that has no vision bias at all.
  const MoeRouteResult text_only = SqrtSoftplusRouteTopk(
      gating, 3, 4, 1, text_bias, false, 1.0f, {}, {}, 4, vision_bias, {});
  const MoeRouteResult before = SqrtSoftplusRouteTopk(
      gating, 3, 4, 1, text_bias, false, 1.0f, {}, {}, 4);
  CHECK(text_only.topk_ids == before.topk_ids);
  CHECK(text_only.topk_weights == before.topk_weights);
}

TEST_CASE("dsv4-moe: on a HASH layer an image row leaves the hash route, a text row keeps it") {
  // vocab=4, E=4, topk=1. The hash table sends every token to expert 3.
  const std::vector<float> row = {LogitForSoftplus(4.0), LogitForSoftplus(1.0),
                                  LogitForSoftplus(9.0), LogitForSoftplus(2.25)};
  std::vector<float> gating;
  for (int t = 0; t < 2; ++t) gating.insert(gating.end(), row.begin(), row.end());
  std::vector<int32_t> table(4 * 1, 3);
  // A hash layer carries NO text bias, which is why the converter drops
  // `ffn.gate.bias` there. The vision bias is present on every layer.
  const std::vector<float> vision_bias = {0.0f, 9.0f, 0.0f, 0.0f};
  const std::vector<int64_t> tokens = {2, 0};
  const std::vector<char> is_media = {0, 1};

  const MoeRouteResult r = SqrtSoftplusRouteTopk(
      gating, 2, 4, 1, /*e_score_correction_bias=*/{}, /*renorm=*/false, 1.0f,
      tokens, table, /*vocab=*/4, vision_bias, is_media);
  REQUIRE(r.topk_ids.size() == 2);
  // Row 0 is text: the hash table decides, and it names expert 3 -- which is
  // neither the unbiased winner (2) nor the vision-biased one (1).
  CHECK(r.topk_ids[0] == 3);
  CHECK(r.topk_weights[0] == doctest::Approx(1.5));
  // Row 1 is an image row: the hash route is REPLACED, and the vision bias
  // selects expert 1.
  CHECK(r.topk_ids[1] == 1);
  CHECK(r.topk_weights[1] == doctest::Approx(1.0));

  // Without the mask the SAME call hashes both rows, which is the behaviour
  // every text step keeps.
  const MoeRouteResult text_only = SqrtSoftplusRouteTopk(
      gating, 2, 4, 1, {}, false, 1.0f, tokens, table, 4, vision_bias, {});
  CHECK(text_only.topk_ids[0] == 3);
  CHECK(text_only.topk_ids[1] == 3);
}

TEST_CASE("dsv4-moe: router f32 == independent f64 reference (randomized top-k + bias)") {
  std::mt19937 rng(0x7A6E);
  std::uniform_real_distribution<float> G(-3.0f, 3.0f);
  std::uniform_real_distribution<float> B(-0.5f, 0.5f);
  for (int64_t E : {8, 16, 32}) {
    for (int64_t topk : {2, 4, 6}) {
      const int64_t M = 5;
      std::vector<float> gating(static_cast<size_t>(M * E)), bias(static_cast<size_t>(E));
      for (auto& v : gating) v = G(rng);
      for (auto& v : bias) v = B(rng);
      const float scale = 1.5f;
      const MoeRouteResult r =
          SqrtSoftplusRouteTopk(gating, M, E, topk, bias, /*renorm=*/true, scale, {}, {}, 0);

      // Independent f64 recompute per token.
      for (int64_t t = 0; t < M; ++t) {
        std::vector<double> sc(static_cast<size_t>(E)), scc(static_cast<size_t>(E));
        for (int64_t e = 0; e < E; ++e) {
          sc[e] = SqrtSoftplusD(gating[t * E + e]);
          scc[e] = sc[e] + bias[e];
        }
        std::vector<int64_t> ord(static_cast<size_t>(E));
        for (int64_t e = 0; e < E; ++e) ord[e] = e;
        std::partial_sort(ord.begin(), ord.begin() + topk, ord.end(),
                          [&](int64_t a, int64_t b) {
                            if (scc[a] != scc[b]) return scc[a] > scc[b];
                            return a < b;
                          });
        double sum = 0.0;
        std::vector<double> wref(static_cast<size_t>(topk));
        for (int64_t j = 0; j < topk; ++j) {
          wref[j] = sc[ord[j]];
          sum += wref[j];
        }
        for (int64_t j = 0; j < topk; ++j) {
          CHECK(r.topk_ids[t * topk + j] == static_cast<int32_t>(ord[j]));
          const double expect = wref[j] / std::max(sum, 1e-20) * scale;
          CHECK(static_cast<double>(r.topk_weights[t * topk + j]) ==
                doctest::Approx(expect).epsilon(1e-5));
        }
      }
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (3) clamped SwiGLU expert activation
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-moe: ClampedSwiGLU clamp is ASYMMETRIC (gate max-only, up both-sided)") {
  // d=1, limit=2, alpha=1, beta=0, gate_up=[-5, -5].
  //   gate = min(-5, 2) = -5   (NOT clamped low — max only)
  //   up   = clamp(-5, -2, 2) = -2
  //   out  = -5 * sigmoid(-5) * (-2) = 10 * 0.00669285 = 0.0669285
  // A symmetric gate clamp (gate → -2, the RED-first lever) would give
  //   -2 * sigmoid(-2) * -2 = 4 * 0.11920292 = 0.4768 — a DIFFERENT value.
  const std::vector<float> gate_up = {-5.0f, -5.0f};
  const std::vector<float> out = ClampedSwiGLU(gate_up, 1, /*limit=*/2.0f, 1.0f, 0.0f);
  REQUIRE(out.size() == 1);
  const double expect = -5.0 * SigmoidD(-5.0) * (-2.0);
  CHECK(out[0] == doctest::Approx(expect));
  CHECK(out[0] != doctest::Approx(-2.0 * SigmoidD(-2.0) * -2.0));  // != symmetric-clamp
}

TEST_CASE("dsv4-moe: ClampedSwiGLU gate max-clamp and up upper-clamp hand cases") {
  // gate over the limit: gate_up=[5,1], limit=2 ⇒ gate=2, up=1, out=2*sigmoid(2)*1.
  {
    const std::vector<float> out = ClampedSwiGLU({5.0f, 1.0f}, 1, 2.0f, 1.0f, 0.0f);
    CHECK(out[0] == doctest::Approx(2.0 * SigmoidD(2.0) * 1.0));
  }
  // up over the limit: gate_up=[1,5], limit=2 ⇒ gate=1, up=2, out=1*sigmoid(1)*2.
  {
    const std::vector<float> out = ClampedSwiGLU({1.0f, 5.0f}, 1, 2.0f, 1.0f, 0.0f);
    CHECK(out[0] == doctest::Approx(1.0 * SigmoidD(1.0) * 2.0));
  }
}

TEST_CASE("dsv4-moe: ClampedSwiGLU alpha (sigmoid scale) and beta (up bias)") {
  // No clamping (limit huge). alpha=2, beta=0: out = 1*sigmoid(2)*1.
  {
    const std::vector<float> out = ClampedSwiGLU({1.0f, 1.0f}, 1, 100.0f, 2.0f, 0.0f);
    CHECK(out[0] == doctest::Approx(1.0 * SigmoidD(2.0) * 1.0));
  }
  // alpha=1, beta=1: out = 1*sigmoid(1)*(1+1).
  {
    const std::vector<float> out = ClampedSwiGLU({1.0f, 1.0f}, 1, 100.0f, 1.0f, 1.0f);
    CHECK(out[0] == doctest::Approx(1.0 * SigmoidD(1.0) * 2.0));
  }
}

TEST_CASE("dsv4-moe: ClampedSwiGLU f32 == independent f64 reference (randomized)") {
  std::mt19937 rng(0x0AC7);
  std::uniform_real_distribution<float> U(-4.0f, 4.0f);
  for (int64_t d : {3, 8, 16}) {
    for (float limit : {1.5f, 3.0f}) {
      std::vector<float> gate_up(static_cast<size_t>(2 * d));
      for (auto& v : gate_up) v = U(rng);
      const float alpha = 1.0f, beta = 0.0f;
      const std::vector<float> out = ClampedSwiGLU(gate_up, d, limit, alpha, beta);
      std::vector<double> ref(static_cast<size_t>(d));
      for (int64_t i = 0; i < d; ++i) {
        const double gate = std::min(static_cast<double>(gate_up[i]), (double)limit);
        const double up = std::min(std::max(static_cast<double>(gate_up[d + i]), -(double)limit),
                                   (double)limit);
        ref[i] = gate * SigmoidD(alpha * gate) * (up + beta);
      }
      CHECK(RelL2(out, ref) < 1e-6);
    }
  }
}
