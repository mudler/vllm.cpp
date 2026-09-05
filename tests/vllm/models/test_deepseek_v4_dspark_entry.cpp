// DSV4-DSPARK-DRAFTER W-2 — the drafter's ENTRY, gated against HAND-COMPUTED
// values rather than against the helpers the code calls.
//
// Spec: `.agents/specs/dsv4-dspark-drafter.md`. Every expectation below is worked
// out from the definitions: the tap is a stream MEAN
// (`exllamav3/modules/transformer.py:198-203`), the concatenation is on the last
// axis, and `main_norm` applies to `main_proj`'s output.
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_dspark.h"

namespace {

// A kPlain f32 view, so the gate exercises the real signature without needing a
// quantized payload -- the dequant path has its own gate.
vllm::DeepseekV4MtpTensorView PlainView(const std::vector<float>& data, int64_t out_dim,
                                        int64_t in_dim) {
  vllm::DeepseekV4MtpTensorView v;
  v.format = vllm::DeepseekV4MtpFormat::kPlain;
  v.dtype = "F32";
  v.shape = {out_dim, in_dim};
  v.data = reinterpret_cast<const uint8_t*>(data.data());
  v.out_dim = out_dim;
  v.in_dim = in_dim;
  return v;
}

}  // namespace

TEST_CASE("W-2: the tap is the STREAM MEAN, hand-computed") {
  // 2 tokens, hc_mult 4, hidden 3. Token 0 dim 0 holds {1,2,3,4} across streams,
  // whose mean is 2.5 -- a value no other reduction produces (sum 10, first 1,
  // last 4, max 4).
  const int64_t T = 2, hc = 4, H = 3;
  std::vector<float> x(static_cast<size_t>(T * hc * H), 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t s = 0; s < hc; ++s)
      for (int64_t d = 0; d < H; ++d)
        x[static_cast<size_t>((t * hc + s) * H + d)] =
            static_cast<float>((s + 1) + 10 * d + 100 * t);

  const std::vector<float> got = vllm::dspark::StreamMeanTap(x, T, hc, H);
  REQUIRE(got.size() == static_cast<size_t>(T * H));
  CHECK(got[0] == doctest::Approx(2.5f));    // (1+2+3+4)/4
  CHECK(got[1] == doctest::Approx(12.5f));   // +10
  CHECK(got[2] == doctest::Approx(22.5f));   // +20
  CHECK(got[3] == doctest::Approx(102.5f));  // token 1
}

TEST_CASE("W-2: the stream mean is NOT the sum, the first stream, or the last") {
  // Guards the reductions a reader might substitute. Each would be a drafter that
  // runs and emits correct tokens while drafting badly.
  const int64_t T = 1, hc = 4, H = 1;
  const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> got = vllm::dspark::StreamMeanTap(x, T, hc, H);
  REQUIRE(got.size() == 1u);
  CHECK(got[0] == doctest::Approx(2.5f));
  CHECK(got[0] != doctest::Approx(10.0f));  // sum
  CHECK(got[0] != doctest::Approx(1.0f));   // first stream
  CHECK(got[0] != doctest::Approx(4.0f));   // last stream
}

TEST_CASE("W-2: ProjectTaps concatenates on the LAST axis, then norms") {
  // hidden 2, n_taps 3 => main_proj is [2, 6]. Identity-ish rows chosen so the
  // pre-norm product is computable by inspection.
  const int64_t T = 1, H = 2, N = 3;
  const std::vector<std::vector<float>> taps{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  // row 0 selects tap0[0], tap1[0], tap2[0]; row 1 selects tap0[1] only.
  const std::vector<float> w{
      1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,   // -> 1 + 3 + 5 = 9
      0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // -> 2
  };
  const auto view = PlainView(w, H, N * H);
  const std::vector<float> gamma{1.0f, 1.0f};
  const float eps = 0.0f;

  const std::vector<float> got =
      vllm::dspark::ProjectTaps(taps, view, gamma, eps, T, H);
  REQUIRE(got.size() == static_cast<size_t>(T * H));

  // Pre-norm [9, 2]; rms = sqrt((81+4)/2) = sqrt(42.5).
  const double rms = std::sqrt((81.0 + 4.0) / 2.0);
  CHECK(got[0] == doctest::Approx(static_cast<float>(9.0 / rms)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(2.0 / rms)));
}

TEST_CASE("W-2: main_norm's gamma is applied per channel") {
  const int64_t T = 1, H = 2, N = 1;
  const std::vector<std::vector<float>> taps{{3.0f, 4.0f}};
  const std::vector<float> w{1.0f, 0.0f, 0.0f, 1.0f};  // identity [2,2]
  const auto view = PlainView(w, H, N * H);
  const std::vector<float> gamma{2.0f, 10.0f};
  const std::vector<float> got =
      vllm::dspark::ProjectTaps(taps, view, gamma, 0.0f, T, H);
  const double rms = std::sqrt((9.0 + 16.0) / 2.0);
  CHECK(got[0] == doctest::Approx(static_cast<float>(3.0 / rms * 2.0)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(4.0 / rms * 10.0)));
}

TEST_CASE("W-2: a tap count that disagrees with main_proj REFUSES") {
  // `dspark_target_layer_ids` has 3 entries and `main_proj` is [H, 3H]. If they
  // ever disagree the projection would read past the taps, so it refuses rather
  // than producing a number.
  const int64_t T = 1, H = 2;
  const std::vector<std::vector<float>> two_taps{{1.0f, 2.0f}, {3.0f, 4.0f}};
  const std::vector<float> w(static_cast<size_t>(H * 3 * H), 0.0f);
  const auto view = PlainView(w, H, 3 * H);  // built for THREE taps
  const std::vector<float> gamma{1.0f, 1.0f};
  CHECK_THROWS(vllm::dspark::ProjectTaps(two_taps, view, gamma, 0.0f, T, H));
}

TEST_CASE("W-2: a tap that is still a STREAM STACK refuses") {
  // The most likely wiring error: handing over `[T, hc, H]` instead of its mean.
  // It has the wrong element count, so it is catchable here -- unlike a wrong
  // REDUCTION, which is not, and which is why the tap has its own gate above.
  const int64_t T = 1, H = 2;
  const std::vector<std::vector<float>> stacked{
      std::vector<float>(static_cast<size_t>(4 * H), 1.0f)};
  const std::vector<float> w(static_cast<size_t>(H * H), 0.0f);
  const auto view = PlainView(w, H, H);
  const std::vector<float> gamma{1.0f, 1.0f};
  CHECK_THROWS(vllm::dspark::ProjectTaps(stacked, view, gamma, 0.0f, T, H));
}

// ── W-4: the sequential draft loop with the factorized-bigram bias ───────────
//
// `sample_from_state` (`exllamav3/architecture/deepseek_v4_mtp.py:311-340`). The
// bias is what makes the chain a BIGRAM: position i's logits are corrected by a
// term conditioned on the id sampled at i-1, so the loop cannot be reordered or
// run in parallel, and dropping the bias yields a drafter that still emits valid
// tokens and drafts worse.

TEST_CASE("W-4: the bias is conditioned on the PREVIOUS sampled id") {
  // vocab 3, rank 1, block 2. w1 maps each id to a 1-wide code; w2 turns that
  // code into a per-vocab bias. Chosen so the chain is forced:
  //   seed 0 -> emb 1.0 -> bias favours id 2 -> out[1] = 2
  //   id 2   -> emb 0.0 -> bias vanishes     -> out[2] = argmax(logits[1]) = 1
  const int64_t V = 3, R = 1, B = 2;
  const std::vector<float> w1{1.0f, 0.0f, 0.0f};       // [V, R]: only id 0 codes 1
  const std::vector<float> w2{0.0f, 0.0f, 10.0f};      // [V, R]: bias id 2 by 10*emb
  const std::vector<float> logits{
      0.0f, 1.0f, 0.0f,   // position 0: without bias id 1 would win
      0.0f, 1.0f, 0.0f,   // position 1: id 1 wins
  };
  const auto out = vllm::dspark::MarkovDraftLoop(logits, /*seed=*/0, w1, w2, B, V, R);
  REQUIRE(out.size() == static_cast<size_t>(B) + 1);
  CHECK(out[0] == 0);  // the seed is carried through, uncropped
  CHECK(out[1] == 2);  // the bias OVERRODE the unbiased argmax of 1
  CHECK(out[2] == 1);  // and vanished once the chain moved to id 2
}

TEST_CASE("W-4: with a zero bias the loop is plain per-position argmax") {
  const int64_t V = 3, R = 2, B = 3;
  const std::vector<float> w1(static_cast<size_t>(V * R), 0.5f);
  const std::vector<float> w2(static_cast<size_t>(V * R), 0.0f);  // no bias at all
  const std::vector<float> logits{
      0.1f, 0.9f, 0.2f,   // -> 1
      0.7f, 0.3f, 0.2f,   // -> 0
      0.0f, 0.1f, 0.8f,   // -> 2
  };
  const auto out = vllm::dspark::MarkovDraftLoop(logits, /*seed=*/1, w1, w2, B, V, R);
  REQUIRE(out.size() == 4u);
  CHECK(out[0] == 1);
  CHECK(out[1] == 1);
  CHECK(out[2] == 0);
  CHECK(out[3] == 2);
}

TEST_CASE("W-4: a tie goes to the LOWEST id, as torch.argmax does") {
  // A draft that breaks ties the other way is still valid text and diverges from
  // the oracle, which is exactly the class of difference acceptance cannot
  // explain after the fact.
  const int64_t V = 3, R = 1, B = 1;
  const std::vector<float> w1{0.0f, 0.0f, 0.0f};
  const std::vector<float> w2{0.0f, 0.0f, 0.0f};
  const std::vector<float> logits{5.0f, 5.0f, 5.0f};
  const auto out = vllm::dspark::MarkovDraftLoop(logits, /*seed=*/2, w1, w2, B, V, R);
  CHECK(out[1] == 0);
}

TEST_CASE("W-4: the bias is a rank-R inner product, hand-computed") {
  // rank 2, so the GEMV actually sums two terms. emb(seed=1) = {2, 3};
  // w2 row 0 = {1, 0} -> 2 ; row 1 = {0, 1} -> 3 ; row 2 = {1, 1} -> 5.
  // logits {0, 1, 0} + bias {2, 3, 5} = {2, 4, 5} -> id 2.
  const int64_t V = 3, R = 2, B = 1;
  const std::vector<float> w1{0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f};
  const std::vector<float> w2{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<float> logits{0.0f, 1.0f, 0.0f};
  const auto out = vllm::dspark::MarkovDraftLoop(logits, /*seed=*/1, w1, w2, B, V, R);
  CHECK(out[1] == 2);
}

TEST_CASE("W-4: a seed outside the vocabulary REFUSES") {
  const int64_t V = 3, R = 1, B = 1;
  const std::vector<float> w1(static_cast<size_t>(V * R), 0.0f);
  const std::vector<float> w2(static_cast<size_t>(V * R), 0.0f);
  const std::vector<float> logits{0.0f, 0.0f, 0.0f};
  CHECK_THROWS(vllm::dspark::MarkovDraftLoop(logits, /*seed=*/V, w1, w2, B, V, R));
  CHECK_THROWS(vllm::dspark::MarkovDraftLoop(logits, /*seed=*/-1, w1, w2, B, V, R));
}

// ── W-3's first brick: the RoPE seam, now shared ─────────────────────────────
//
// `RopeInplaceLayer` was file-local in `deepseek_v4.cpp` until the drafter needed
// it: a DSpark block derives its KV rows from the projected taps and applies the
// SAME partial rotation the trunk applies to its own KV. Exported rather than
// re-implemented, because DeepSeek-V4's RoPE is DUAL and a second copy would be a
// second place for that split to drift.
//
// These cases pin the contract the drafter depends on. The trunk already calls
// the function, so they are not a reachability claim; they are the guard on what
// moving it out of an anonymous namespace must not have changed.

#include "vllm/model_executor/models/deepseek_v4_rope.h"

TEST_CASE("W-3 seam: the dense arm is a ROTATION, so it preserves pair norms") {
  // Dense layers rotate with ext_factor == 0. A rotation of adjacent pairs leaves
  // each pair's magnitude alone; anything that scales instead would pass a
  // finiteness check and quietly change attention logits.
  std::vector<float> v{1.0f, 2.0f, -3.0f, 0.5f, 0.25f, -1.5f};
  const std::vector<float> before = v;
  vllm::deepseek_v4::RopeInplaceLayer(v.data(), /*r=*/6, /*pos=*/7, /*base=*/10000.0,
                                      /*freq_scale=*/1.0, /*ext_factor=*/0.0,
                                      /*n_ctx_orig=*/4096, /*beta_fast=*/32.0,
                                      /*beta_slow=*/1.0);
  bool moved = false;
  for (size_t i = 0; i < v.size(); i += 2) {
    const double n0 = std::hypot(before[i], before[i + 1]);
    const double n1 = std::hypot(v[i], v[i + 1]);
    CHECK(n1 == doctest::Approx(n0).epsilon(1e-6));
    if (std::abs(v[i] - before[i]) > 1e-6) moved = true;
  }
  CHECK(moved);  // pos 7 must actually rotate; an all-no-op would satisfy the norms
}

TEST_CASE("W-3 seam: `inverse` un-rotates exactly") {
  std::vector<float> v{0.3f, -0.7f, 1.1f, 2.0f};
  const std::vector<float> before = v;
  for (int pass = 0; pass < 2; ++pass) {
    vllm::deepseek_v4::RopeInplaceLayer(v.data(), 4, /*pos=*/11, 10000.0, 1.0, 0.0, 4096,
                                        32.0, 1.0, /*inverse=*/pass == 1);
  }
  for (size_t i = 0; i < v.size(); ++i)
    CHECK(v[i] == doctest::Approx(before[i]).epsilon(1e-5));
}

TEST_CASE("W-3 seam: `ext_factor` alone changes the rotation") {
  // 41 of 43 trunk layers take the YaRN arm. The FIRST version of this case varied
  // `freq_scale` as well, so the two calls differed for that reason alone and a
  // mutation disabling the ext_factor ramp passed it. Everything but `ext_factor`
  // is held equal here, which is what makes the ramp observable.
  const std::vector<float> src{1.0f, 0.0f, 0.0f, 1.0f, 0.5f, -0.5f, 2.0f, 0.25f};
  std::vector<float> off = src, on = src;
  const double kFreq = 1.0 / 16.0;
  vllm::deepseek_v4::RopeInplaceLayer(off.data(), 8, 64, 160000.0, kFreq,
                                      /*ext_factor=*/0.0, 4096, 32.0, 1.0);
  vllm::deepseek_v4::RopeInplaceLayer(on.data(), 8, 64, 160000.0, kFreq,
                                      /*ext_factor=*/1.0, 4096, 32.0, 1.0);
  double diff = 0.0;
  for (size_t i = 0; i < src.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(off[i] - on[i])));
  CHECK(diff > 1e-3);
}

TEST_CASE("W-3 seam: `freq_scale` alone changes the rotation") {
  // The other half of the dual-theta split, held to the same standard: only
  // `freq_scale` moves.
  const std::vector<float> src{1.0f, 0.0f, 0.0f, 1.0f, 0.5f, -0.5f, 2.0f, 0.25f};
  std::vector<float> one = src, scaled = src;
  vllm::deepseek_v4::RopeInplaceLayer(one.data(), 8, 64, 160000.0, /*freq_scale=*/1.0,
                                      0.0, 4096, 32.0, 1.0);
  vllm::deepseek_v4::RopeInplaceLayer(scaled.data(), 8, 64, 160000.0,
                                      /*freq_scale=*/1.0 / 16.0, 0.0, 4096, 32.0, 1.0);
  double diff = 0.0;
  for (size_t i = 0; i < src.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(one[i] - scaled[i])));
  CHECK(diff > 1e-3);
}

// ── W-3: one block's KV rows from the projected taps ─────────────────────────

TEST_CASE("W-3: the norm covers the WHOLE head, then the tail rotates") {
  // The distinction this case exists for: DeepSeek-V2/V3 MLA norms only the nope
  // half and leaves the decoupled rope part unnormed (`vt::kFusedNormRope`), while
  // this path norms all `head_dim` and rotates afterwards
  // (`exllamav3_ext/rope.cu:207,248-251,379`). The artifact settles which: its
  // `kv_norm.weight` is `[512]`, the full head, not `[448]`.
  //
  // Built so the two are distinguishable: gamma is 1 on the nope half and 3 on the
  // rope half. Under the V2 convention the rope half would be untouched by gamma.
  const int64_t T = 1, H = 2, D = 4, RD = 2, NOPE = D - RD;
  const std::vector<float> main_x{1.0f, 0.0f};
  // wkv row d selects main_x[0] scaled, so the pre-norm row is {1,2,3,4}.
  std::vector<float> wkv(static_cast<size_t>(D * H), 0.0f);
  for (int64_t d = 0; d < D; ++d) wkv[static_cast<size_t>(d * H)] = static_cast<float>(d + 1);
  std::vector<float> gamma{1.0f, 1.0f, 3.0f, 3.0f};
  const std::vector<int32_t> pos{0};  // position 0 => the rotation is identity

  const auto got = vllm::dspark::BlockKvRows(main_x, wkv, gamma, /*eps=*/0.0f, pos,
                                             /*rope_theta=*/10000.0, T, H, D, RD);
  REQUIRE(got.size() == static_cast<size_t>(D));
  // RMS over ALL FOUR values: sqrt((1+4+9+16)/4) = sqrt(7.5).
  const double rms = std::sqrt(30.0 / 4.0);
  CHECK(got[0] == doctest::Approx(static_cast<float>(1.0 / rms * 1.0)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(2.0 / rms * 1.0)));
  // The rope half carries gamma too -- this is what the V2 convention would not do.
  CHECK(got[2] == doctest::Approx(static_cast<float>(3.0 / rms * 3.0)));
  CHECK(got[3] == doctest::Approx(static_cast<float>(4.0 / rms * 3.0)));
  (void)NOPE;
}

TEST_CASE("W-3: a non-zero position ROTATES only the rope tail") {
  const int64_t T = 2, H = 1, D = 4, RD = 2;
  const std::vector<float> main_x{1.0f, 1.0f};
  std::vector<float> wkv(static_cast<size_t>(D * H), 0.0f);
  for (int64_t d = 0; d < D; ++d) wkv[static_cast<size_t>(d)] = static_cast<float>(d + 1);
  const std::vector<float> gamma(static_cast<size_t>(D), 1.0f);
  // Token 0 at position 0 (identity), token 1 at position 5 (rotates).
  const std::vector<int32_t> pos{0, 5};
  const auto got = vllm::dspark::BlockKvRows(main_x, wkv, gamma, 0.0f, pos, 10000.0, T, H,
                                             D, RD);
  REQUIRE(got.size() == static_cast<size_t>(T * D));
  // The NOPE half is position-independent, so both tokens must agree there.
  CHECK(got[0] == doctest::Approx(got[4]));
  CHECK(got[1] == doctest::Approx(got[5]));
  // The rope half must NOT agree, or the position never entered the row.
  const double dr = std::max(std::abs(static_cast<double>(got[2] - got[6])),
                             std::abs(static_cast<double>(got[3] - got[7])));
  CHECK(dr > 1e-4);
  // ...and the rotation preserves the pair's magnitude.
  CHECK(std::hypot(got[6], got[7]) ==
        doctest::Approx(std::hypot(got[2], got[3])).epsilon(1e-5));
}

TEST_CASE("W-3: a kv_norm sized for the NOPE half REFUSES") {
  // The V2-convention mistake, caught at the boundary instead of producing a
  // plausible row.
  const int64_t T = 1, H = 1, D = 4, RD = 2;
  const std::vector<float> main_x{1.0f};
  const std::vector<float> wkv(static_cast<size_t>(D * H), 1.0f);
  const std::vector<float> nope_gamma(static_cast<size_t>(D - RD), 1.0f);
  const std::vector<int32_t> pos{0};
  CHECK_THROWS(vllm::dspark::BlockKvRows(main_x, wkv, nope_gamma, 0.0f, pos, 10000.0, T,
                                         H, D, RD));
}

TEST_CASE("W-3: an ODD rope span REFUSES, because the rotation is pairwise") {
  const int64_t T = 1, H = 1, D = 4, RD = 3;
  const std::vector<float> main_x{1.0f};
  const std::vector<float> wkv(static_cast<size_t>(D * H), 1.0f);
  const std::vector<float> gamma(static_cast<size_t>(D), 1.0f);
  const std::vector<int32_t> pos{0};
  CHECK_THROWS(
      vllm::dspark::BlockKvRows(main_x, wkv, gamma, 0.0f, pos, 10000.0, T, H, D, RD));
}

TEST_CASE("W-3: norm comes BEFORE rope, where the two do not commute") {
  // The order is only OBSERVABLE under two conditions at once, and the first
  // version of this file met neither, so a mutation that roped before norming
  // passed it untouched:
  //
  //   1. a NON-ZERO position, or the rotation is the identity; and
  //   2. a gamma that DIFFERS WITHIN a rotated pair. RMSNorm's scale is a scalar
  //      and rotation preserves the sum of squares, so with g[i] == g[i+1] the two
  //      operations commute exactly and no test can tell them apart.
  const int64_t T = 1, H = 1, D = 4, RD = 2, NOPE = D - RD;
  const std::vector<float> main_x{1.0f};
  std::vector<float> wkv(static_cast<size_t>(D * H), 0.0f);
  for (int64_t d = 0; d < D; ++d) wkv[static_cast<size_t>(d)] = static_cast<float>(d + 1);
  const std::vector<float> gamma{1.0f, 1.0f, 2.0f, 5.0f};  // 2 != 5 inside the pair
  const std::vector<int32_t> pos{1};

  const auto got = vllm::dspark::BlockKvRows(main_x, wkv, gamma, /*eps=*/0.0f, pos,
                                             /*rope_theta=*/10000.0, T, H, D, RD);
  REQUIRE(got.size() == static_cast<size_t>(D));

  // Hand-derived NORM-THEN-ROPE. Pre-norm row is {1,2,3,4}; rms = sqrt(30/4).
  // Normed = {1r, 2r, 6r, 20r}. The rope span is one pair at theta = pos = 1.
  const double r = 1.0 / std::sqrt(30.0 / 4.0);
  const double c = std::cos(1.0), s = std::sin(1.0);
  const double n2 = 6.0 * r, n3 = 20.0 * r;
  CHECK(got[0] == doctest::Approx(static_cast<float>(1.0 * r)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(2.0 * r)));
  CHECK(got[2] == doctest::Approx(static_cast<float>(n2 * c - n3 * s)));
  CHECK(got[3] == doctest::Approx(static_cast<float>(n2 * s + n3 * c)));
  (void)NOPE;
}

// ── W-3: the block's weights, assembled onto the trunk's own layer struct ────

namespace {

// A head at a tiny but SHAPE-CONSISTENT geometry: every tensor the assembly
// requires, at the widths `DeepseekV4Params` declares.
vllm::DeepseekV4MtpHead TinyHead(const vllm::DeepseekV4Params& p,
                                 std::vector<std::vector<float>>* storage) {
  vllm::DeepseekV4MtpHead h;
  const int64_t H = p.hidden_size, hc = p.hc_mult, D = p.head_dim;
  const auto add = [&](const std::string& key, int64_t out_dim, int64_t in_dim) {
    storage->push_back(std::vector<float>(
        static_cast<size_t>(out_dim * in_dim), 0.125f));
    vllm::DeepseekV4MtpTensorView v;
    v.format = vllm::DeepseekV4MtpFormat::kPlain;
    v.dtype = "F32";
    v.shape = {out_dim, in_dim};
    v.data = reinterpret_cast<const uint8_t*>(storage->back().data());
    v.out_dim = out_dim;
    v.in_dim = in_dim;
    h.tensors.emplace(key, v);
  };
  add("attn_norm.weight", H, 1);
  add("ffn_norm.weight", H, 1);
  add("attn.q_norm.weight", p.q_lora_rank, 1);
  add("attn.kv_norm.weight", D, 1);
  add("attn.attn_sink", p.num_attention_heads, 1);
  add("attn.wq_a.weight", p.q_lora_rank, H);
  add("attn.wq_b.weight", p.num_attention_heads * D, p.q_lora_rank);
  add("attn.wkv.weight", D, H);
  // `[n_groups, o_lora_rank, n_heads*head_dim/n_groups]`, so the element count is
  // `o_lora_rank * n_heads * head_dim` -- NOT `o_groups*o_lora_rank x hidden`.
  add("attn.wo_a.weight", p.o_lora_rank * p.num_attention_heads, D);
  add("attn.wo_b.weight", H, p.o_groups * p.o_lora_rank);
  add("ffn.gate.weight", p.n_routed_experts, H);
  add("ffn.gate.bias", p.n_routed_experts, 1);
  add("ffn.shared_experts.w1.weight", p.moe_intermediate_size, H);
  add("ffn.shared_experts.w2.weight", H, p.moe_intermediate_size);
  add("ffn.shared_experts.w3.weight", p.moe_intermediate_size, H);
  add("hc_attn_fn", (2 + hc) * hc, hc * H);
  add("hc_attn_base", (2 + hc) * hc, 1);
  add("hc_attn_scale", hc - 1, 1);
  add("hc_ffn_fn", (2 + hc) * hc, hc * H);
  add("hc_ffn_base", (2 + hc) * hc, 1);
  add("hc_ffn_scale", hc - 1, 1);
  return h;
}

vllm::DeepseekV4Params TinyBlockParams() {
  vllm::DeepseekV4Params p;
  p.hidden_size = 8;
  p.hc_mult = 4;
  p.head_dim = 6;
  p.num_attention_heads = 2;
  p.q_lora_rank = 4;
  p.o_lora_rank = 4;
  p.o_groups = 2;
  p.n_routed_experts = 4;
  p.moe_intermediate_size = 6;
  return p;
}

}  // namespace

TEST_CASE("W-3: a complete head assembles onto the trunk's layer struct") {
  const vllm::DeepseekV4Params p = TinyBlockParams();
  std::vector<std::vector<float>> storage;
  const vllm::DeepseekV4MtpHead h = TinyHead(p, &storage);
  vllm::DeepseekV4LayerHostWeights L;
  bool missing_experts = false;
  const std::string r = vllm::dspark::AssembleBlockWeights(h, p, &L, &missing_experts);
  CHECK(r.empty());
  CHECK(static_cast<int64_t>(L.attn_norm_weight.size()) == p.hidden_size);
  CHECK(static_cast<int64_t>(L.kv_norm_weight.size()) == p.head_dim);
  CHECK(static_cast<int64_t>(L.wkv.size()) == p.head_dim * p.hidden_size);
  CHECK(static_cast<int64_t>(L.gate_weight.size()) == p.n_routed_experts * p.hidden_size);
  // The blocks are compressor-less and indexer-less; those fields must stay EMPTY,
  // or a later forward would take the DSA path on a drafter block.
  CHECK(L.comp_wgate.empty());
  CHECK(L.comp_ape.empty());
  CHECK(L.idx_wk.empty());
  // The routed experts are deliberately absent -- 20.2 GiB per block as host f32 --
  // and the caller is TOLD rather than left to meet an empty vector later.
  CHECK(missing_experts);
  CHECK(L.exp_w1.empty());
}

TEST_CASE("W-3: a head missing ONE tensor is refused BY NAME") {
  const vllm::DeepseekV4Params p = TinyBlockParams();
  for (const char* drop : {"attn.wkv.weight", "hc_ffn_base", "ffn.gate.bias"}) {
    std::vector<std::vector<float>> storage;
    vllm::DeepseekV4MtpHead h = TinyHead(p, &storage);
    h.tensors.erase(drop);
    vllm::DeepseekV4LayerHostWeights L;
    bool me = false;
    const std::string r = vllm::dspark::AssembleBlockWeights(h, p, &L, &me);
    CAPTURE(drop);
    CHECK(!r.empty());
    CHECK(r.find(drop) != std::string::npos);
  }
}

TEST_CASE("W-3: a tensor at the WRONG width is refused, not silently accepted") {
  // The failure this guards: a shape mismatch that surfaces deep inside a block
  // forward as an anonymous MatVec size error naming no tensor and no layer.
  const vllm::DeepseekV4Params p = TinyBlockParams();
  std::vector<std::vector<float>> storage;
  vllm::DeepseekV4MtpHead h = TinyHead(p, &storage);
  auto it = h.tensors.find("attn.kv_norm.weight");
  REQUIRE(it != h.tensors.end());
  it->second.out_dim = p.head_dim - 2;  // a nope-sized norm, the V2-convention slip
  vllm::DeepseekV4LayerHostWeights L;
  bool me = false;
  const std::string r = vllm::dspark::AssembleBlockWeights(h, p, &L, &me);
  CHECK(!r.empty());
  CHECK(r.find("kv_norm") != std::string::npos);
}

// ── W-4b: the confidence-capped draft length ─────────────────────────────────

TEST_CASE("W-4b: the length is the longest CONTIGUOUS prefix, not a count") {
  // The distinction `cumprod(keep).sum()` encodes. With confidence high, low,
  // high the answer is 1 -- counting confident positions would say 2 and let the
  // drafter propose past a position the model flagged.
  const int64_t B = 3, H = 1, R = 1;
  // proj selects xpre only; markov_emb contributes nothing.
  const std::vector<float> proj{1.0f, 0.0f};
  const std::vector<float> emb(static_cast<size_t>(B * R), 0.0f);
  const std::vector<float> xpre{5.0f, -5.0f, 5.0f};  // sigmoid: ~1, ~0, ~1
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, emb, proj, 0.5f, B, H, R) == 1);
}

TEST_CASE("W-4b: an all-confident block drafts its full length") {
  const int64_t B = 4, H = 1, R = 1;
  const std::vector<float> proj{1.0f, 0.0f};
  const std::vector<float> emb(static_cast<size_t>(B * R), 0.0f);
  const std::vector<float> xpre{6.0f, 6.0f, 6.0f, 6.0f};
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, emb, proj, 0.5f, B, H, R) == 4);
}

TEST_CASE("W-4b: an unconfident FIRST position yields 0, i.e. skip drafting") {
  const int64_t B = 3, H = 1, R = 1;
  const std::vector<float> proj{1.0f, 0.0f};
  const std::vector<float> emb(static_cast<size_t>(B * R), 0.0f);
  const std::vector<float> xpre{-6.0f, 6.0f, 6.0f};
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, emb, proj, 0.5f, B, H, R) == 0);
}

TEST_CASE("W-4b: the markov embedding half of the input is LOAD-BEARING") {
  // The projection reads `cat(xpre, markov_emb)`. If only the hidden half were
  // consumed the head would be blind to which token the chain actually sampled.
  const int64_t B = 1, H = 1, R = 1;
  const std::vector<float> xpre{0.0f};
  const std::vector<float> proj{1.0f, 4.0f};  // all the signal is in the emb half
  const std::vector<float> hot{2.0f};   // 8.0 -> confident
  const std::vector<float> cold{-2.0f}; // -8.0 -> not
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, hot, proj, 0.5f, B, H, R) == 1);
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, cold, proj, 0.5f, B, H, R) == 0);
}

TEST_CASE("W-4b: the threshold is applied to SIGMOID(conf), not to conf") {
  // conf = 0.4 is below a 0.5 threshold as a raw logit but sigmoid(0.4) = 0.599,
  // which is above it. Comparing the raw logit would truncate healthy drafts.
  const int64_t B = 1, H = 1, R = 1;
  const std::vector<float> proj{1.0f, 0.0f};
  const std::vector<float> emb{0.0f};
  const std::vector<float> xpre{0.4f};
  CHECK(vllm::dspark::ConfidenceDraftLength(xpre, emb, proj, 0.5f, B, H, R) == 1);
}

TEST_CASE("W-4b: a projection sized for the hidden half alone REFUSES") {
  const int64_t B = 1, H = 2, R = 2;
  const std::vector<float> xpre{1.0f, 1.0f};
  const std::vector<float> emb{1.0f, 1.0f};
  const std::vector<float> short_proj{1.0f, 1.0f};  // H only, missing the rank half
  CHECK_THROWS(
      vllm::dspark::ConfidenceDraftLength(xpre, emb, short_proj, 0.5f, B, H, R));
}

// ── W-3: one DSpark block's attention half ──────────────────────────────────

TEST_CASE("W-3: a block attends the rows BlockKvRows wrote, without rewriting them") {
  // The whole point of the `kv_prewritten` seam. A DSpark block's KV comes from
  // the TARGET's taps, so the block must attend rows it did not compute -- and
  // must not overwrite them with rows derived from its own hidden state.
  vllm::DeepseekV4Params p = TinyBlockParams();
  p.num_hidden_layers = 1;
  p.num_attention_heads = 2;
  p.qk_rope_head_dim = 2;
  p.rms_norm_eps = 1e-6f;
  p.rope_theta = 10000.0;
  p.sliding_window = 0;

  std::vector<std::vector<float>> storage;
  const vllm::DeepseekV4MtpHead h = TinyHead(p, &storage);
  vllm::DeepseekV4LayerHostWeights L;
  bool missing = false;
  REQUIRE(vllm::dspark::AssembleBlockWeights(h, p, &L, &missing).empty());
  // The block forward needs the routed experts, which assembly deliberately does
  // not fill (20.2 GiB at the real geometry). Fill them HERE, at tiny shape, so
  // the attention half can be exercised.
  const int64_t E = p.n_routed_experts, mi = p.moe_intermediate_size,
                H = p.hidden_size;
  L.exp_w1.assign(static_cast<size_t>(E * mi * H), 0.02f);
  L.exp_w3.assign(static_cast<size_t>(E * mi * H), 0.02f);
  L.exp_w2.assign(static_cast<size_t>(E * H * mi), 0.02f);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = p.head_dim, bs = 8, nb = 2, T = 1, kv_base = 3;
  std::vector<float> cache(static_cast<size_t>(nb * bs * hd), 0.0f);
  // A RECOGNISABLE pattern: these are the rows the taps wrote.
  for (size_t i = 0; i < cache.size(); ++i)
    cache[i] = 0.01f * static_cast<float>((i * 5) % 29) - 0.1f;
  const std::vector<float> before = cache;
  std::vector<vt::Tensor> pages{vt::Tensor::Contiguous(
      cache.data(), vt::DType::kF32, q.device, {nb, bs, hd})};

  const std::vector<float> x(static_cast<size_t>(T * H), 0.05f);
  const std::vector<int32_t> pos{static_cast<int32_t>(kv_base)};
  const std::vector<float> o = vllm::DsparkBlockAttentionHost(q, L, p, x, pos, pages,
                                                              /*layer=*/0, kv_base);
  REQUIRE(o.size() == static_cast<size_t>(T * H));
  for (const float v : o) REQUIRE(std::isfinite(v));

  // THE CLAIM: the tap-written rows survive byte for byte.
  for (size_t i = 0; i < before.size(); ++i) {
    if (before[i] != cache[i]) {
      CAPTURE(i);
      REQUIRE(before[i] == cache[i]);
    }
  }
  // ...and the attention actually READ them: zeroing the cache must change the
  // output, or the block would be attending nothing and still returning finitely.
  std::vector<float> zeroed(cache.size(), 0.0f);
  std::vector<vt::Tensor> zpages{vt::Tensor::Contiguous(
      zeroed.data(), vt::DType::kF32, q.device, {nb, bs, hd})};
  const std::vector<float> oz =
      vllm::DsparkBlockAttentionHost(q, L, p, x, pos, zpages, 0, kv_base);
  double diff = 0.0;
  for (size_t i = 0; i < o.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(o[i] - oz[i])));
  CHECK(diff > 1e-9);
}

TEST_CASE("W-3: a block carrying a compressor is REFUSED") {
  // `mtp_layer_types` is asserted "sliding" (`deepseek_v4_mtp.py:61-63`): a
  // DSpark block is compressor-less and has no indexer. A layer carrying either
  // is not a DSpark block, and running it as one would take the DSA path.
  vllm::DeepseekV4Params p = TinyBlockParams();
  p.num_hidden_layers = 1;
  std::vector<std::vector<float>> storage;
  const vllm::DeepseekV4MtpHead h = TinyHead(p, &storage);
  vllm::DeepseekV4LayerHostWeights L;
  bool missing = false;
  REQUIRE(vllm::dspark::AssembleBlockWeights(h, p, &L, &missing).empty());
  L.comp_wgate.assign(4, 1.0f);  // not a DSpark block any more

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<float> cache(static_cast<size_t>(16 * p.head_dim), 0.0f);
  std::vector<vt::Tensor> pages{vt::Tensor::Contiguous(
      cache.data(), vt::DType::kF32, q.device, {2, 8, p.head_dim})};
  const std::vector<float> x(static_cast<size_t>(p.hidden_size), 0.05f);
  // It must refuse BY NAME. Something downstream throws anyway on the widened
  // tensor -- an anonymous size error -- so asserting only that it throws proves
  // nothing about THIS guard, and a mutation removing the guard survived exactly
  // that weaker assertion.
  std::string msg;
  try {
    (void)vllm::DsparkBlockAttentionHost(q, L, p, x, {0}, pages, 0, 0);
    FAIL("expected a refusal");
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CAPTURE(msg);
  CHECK(msg.find("COMPRESSOR-LESS") != std::string::npos);
  CHECK(msg.find("sliding") != std::string::npos);
}

// ── W-5: the propose side's bookkeeping ─────────────────────────────────────

TEST_CASE("W-5: cu_num_logits is 1 + the draft length, per request") {
  // The sampler derives each request's length as `cu[r+1] - cu[r]`
  // (`rejection_sampler.cpp:85-86`), and row `cu[r]` is the previous token,
  // uncompared. `MarkovDraftLoop` returns `[seed, drafts...]`, so the seed IS
  // that row.
  const std::vector<std::vector<int32_t>> drafted{
      {7, 11, 12, 13, 14, 15},  // seed 7, five drafts
      {9, 21, 22, 23, 24, 25},
  };
  const std::vector<int64_t> lengths{5, 2};  // the second request was capped
  std::vector<int32_t> cu;
  const auto flat = vllm::dspark::ProposeToVerifyInputs(drafted, lengths, &cu);

  REQUIRE(cu.size() == 3u);
  CHECK(cu[0] == 0);
  CHECK(cu[1] == 6);   // 1 + 5
  CHECK(cu[2] == 9);   // + 1 + 2
  REQUIRE(flat.size() == 9u);
  // Request 0: seed then all five drafts.
  CHECK(flat[0] == 7);
  CHECK(flat[5] == 15);
  // Request 1: seed then only the two the confidence cap kept.
  CHECK(flat[6] == 9);
  CHECK(flat[7] == 21);
  CHECK(flat[8] == 22);
}

TEST_CASE("W-5: a length of 0 is a request that SKIPS drafting, not an error") {
  // One row carrying the previous token yields the bonus token alone, which is a
  // valid sampling row. Treating it as empty would give the request zero rows and
  // desynchronise every later offset.
  const std::vector<std::vector<int32_t>> drafted{{4, 5, 6}, {8, 9, 10}};
  const std::vector<int64_t> lengths{0, 2};
  std::vector<int32_t> cu;
  const auto flat = vllm::dspark::ProposeToVerifyInputs(drafted, lengths, &cu);
  REQUIRE(cu.size() == 3u);
  CHECK(cu[0] == 0);
  CHECK(cu[1] == 1);  // exactly the previous token
  CHECK(cu[2] == 4);
  REQUIRE(flat.size() == 4u);
  CHECK(flat[0] == 4);   // the skipped request's seed survives
  CHECK(flat[1] == 8);
}

TEST_CASE("W-5: cu is CUMULATIVE, so a short request does not shift the rest") {
  // Three requests at different caps. The offsets must accumulate, since the
  // sampler indexes rows by them and an off-by-one would compare one request's
  // drafts against another's logits.
  const std::vector<std::vector<int32_t>> drafted{
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  const std::vector<int64_t> lengths{3, 0, 1};
  std::vector<int32_t> cu;
  const auto flat = vllm::dspark::ProposeToVerifyInputs(drafted, lengths, &cu);
  REQUIRE(cu.size() == 4u);
  CHECK(cu[0] == 0);
  CHECK(cu[1] == 4);
  CHECK(cu[2] == 5);
  CHECK(cu[3] == 7);
  CHECK(static_cast<int32_t>(flat.size()) == cu.back());
}

TEST_CASE("W-5: a length past the drafted block REFUSES") {
  // The confidence cap cannot exceed the block the loop produced; a length that
  // does would name drafts nobody sampled.
  const std::vector<std::vector<int32_t>> drafted{{1, 2, 3}};
  std::vector<int32_t> cu;
  CHECK_THROWS(vllm::dspark::ProposeToVerifyInputs(drafted, {3}, &cu));
  CHECK_THROWS(vllm::dspark::ProposeToVerifyInputs(drafted, {-1}, &cu));
}
