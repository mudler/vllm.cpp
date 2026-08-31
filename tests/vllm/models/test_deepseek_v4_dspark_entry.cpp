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
