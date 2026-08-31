// DeepSeek-V4-Flash W4 UNIT GATE — the DSA COMPRESSOR forward + the fp8_ds_mla
// KV-cache state read/write layout.
// HONEST scope: the full-model gate is multi-Spark-blocked (156.7 GiB, does not
// fit one GB10; forward also needs MHC/MoE, not ported). So W4 gates the MATH
// per-primitive against HAND-DERIVED small cases (literal expected numbers I can
// verify by hand from the vLLM source) PLUS from-first-principles double-
// precision references on randomized shapes (rel-L2 / independent recompute).
// This is the "hand-case + structural review" bar named in the W4 brief, NOT a
// dumped-oracle rel-L2 (the arch cannot be constructed at a tiny shape — it is a
// fixed-config 167B). The eventual GPU forward (W7) ports the same math into a
// CUDA kernel; these tests are its portable oracle.
//
// Grounded in: common/ops/save_partial_states.py:92-101 (APE add),
// common/ops/fused_compress_quant_cache.py:198-297 (pool+RMSNorm + fp8_ds_mla
// store), compressor.py:307-309 (layout), cross-checked against SGLang v0.5.15
// dsv4/fused_compress_triton.py + dsv4/quant_k_cache.py + dsv4/dequant_k_cache.py.
#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vllm/model_executor/models/deepseek_v4_rope.h"

#include <doctest/doctest.h>

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
}  // namespace

// ── (A1) Save-time APE fusion ────────────────────────────────────────────────

TEST_CASE("dsv4-compressor: save-time score += ape[position % compress_ratio]") {
  // T=2, width=2, compress_ratio=2. positions [0,1] -> ape rows [0,1].
  const std::vector<float> score = {1, 2, 3, 4};
  const std::vector<float> ape = {10, 20, 30, 40};  // [2 rows, 2 wide]
  const std::vector<int64_t> pos = {0, 1};
  const std::vector<float> out = CompressorSaveScoreApe(score, ape, pos, 2, 2, 2);
  REQUIRE(out.size() == 4);
  CHECK(out[0] == doctest::Approx(11));  // 1 + ape[0][0]=10
  CHECK(out[1] == doctest::Approx(22));  // 2 + ape[0][1]=20
  CHECK(out[2] == doctest::Approx(33));  // 3 + ape[1][0]=30
  CHECK(out[3] == doctest::Approx(44));  // 4 + ape[1][1]=40
}

TEST_CASE("dsv4-compressor: APE row wraps with position modulo compress_ratio") {
  // position 3, compress_ratio 2 -> ape row 1 (not 3).
  const std::vector<float> score = {0, 0};
  const std::vector<float> ape = {10, 20, 30, 40};
  const std::vector<int64_t> pos = {3};
  const std::vector<float> out = CompressorSaveScoreApe(score, ape, pos, 1, 2, 2);
  CHECK(out[0] == doctest::Approx(30));  // ape[3 % 2 = 1][0]
  CHECK(out[1] == doctest::Approx(40));  // ape[1][1]
}

// ── (A2) Compressor POOL + RMSNorm ───────────────────────────────────────────

TEST_CASE("dsv4-compressor: pool = softmax(score,dim=0) . kv, then RMSNorm — hand case") {
  // window=2, head_dim=2. score all 0 -> softmax = [0.5,0.5] per column.
  //   compressed[0] = 1*.5 + 3*.5 = 2 ; compressed[1] = 2*.5 + 4*.5 = 3
  //   var = (4+9)/2 = 6.5 ; rrms = 1/sqrt(6.5)
  const std::vector<float> kv = {1, 2, 3, 4};
  const std::vector<float> score = {0, 0, 0, 0};
  const std::vector<uint8_t> valid = {1, 1};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  const double rrms = 1.0 / std::sqrt(6.5);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(2.0 * rrms));
  CHECK(out[1] == doctest::Approx(3.0 * rrms));
}

TEST_CASE("dsv4-compressor: masked window rows are excluded from the pool") {
  // valid=[1,0] -> only row0 pools; softmax over one row = 1.0.
  //   compressed = kv[0] = [1,2] ; var = (1+4)/2 = 2.5
  const std::vector<float> kv = {1, 2, 999, 999};  // row1 must NOT leak
  const std::vector<float> score = {0, 0, 0, 0};
  const std::vector<uint8_t> valid = {1, 0};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  const double rrms = 1.0 / std::sqrt(2.5);
  CHECK(out[0] == doctest::Approx(1.0 * rrms));
  CHECK(out[1] == doctest::Approx(2.0 * rrms));
}

TEST_CASE("dsv4-compressor: softmax is PER-COLUMN (dim=0) — the load-bearing nuance") {
  // If pooling used one shared weight per row, both columns would pool
  // identically. Here column 1's score strongly favors row 0 while column 0 is
  // uniform, so the compressed columns differ in a way only per-column softmax
  // produces. kv=[[10,10],[0,0]], score=[[0,100],[0,0]].
  //   col0: softmax([0,0])=[.5,.5] -> 5 ; col1: softmax([100,0])~[1,0] -> 10
  // RMSNorm scales both columns by the same rrms, so the RATIO 10/5 = 2 survives.
  const std::vector<float> kv = {10, 10, 0, 0};
  const std::vector<float> score = {0, 100, 0, 0};
  const std::vector<uint8_t> valid = {1, 1};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  CHECK(out[1] == doctest::Approx(2.0 * out[0]));
  CHECK(out[0] > 0.0f);
}

// ── (B1) fp8_ds_mla layout geometry ──────────────────────────────────────────

TEST_CASE("dsv4-fp8_ds_mla: V4 layout is 448 fp8 + 64 bf16, 576B stride, 7+1 scales") {
  const Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 64, 64);
  CHECK(L.nope_head_dim == 448);
  CHECK(L.rope_head_dim == 64);
  CHECK(L.quant_block == 64);
  CHECK(L.n_nope_blocks == 7);         // 448 / 64
  CHECK(L.token_stride_bytes == 576);  // 448*1 + 64*2
  CHECK(L.scale_dim == 8);             // 7 real + 1 pad
}

// ── (B2) fp8_ds_mla UE8M0 scale-byte encoding (hand-derived) ─────────────────

TEST_CASE("dsv4-fp8_ds_mla: all-ones block -> UE8M0 scale byte 119, exact round-trip") {
  // absmax=1, raw=1/448 -> exponent=ceil(log2(1/448))=-8 -> byte=-8+127=119.
  // decode: fp8(256) * 2^(119-127)=2^-8 -> 1.0 (256 is e4m3-exact).
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(64, 0, 64);  // one nope block, no rope
  std::vector<float> head(64, 1.0f);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  REQUIRE(t.scale_ue8m0.size() == 1);
  CHECK(static_cast<int>(t.scale_ue8m0[0]) == 119);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  REQUIRE(dec.size() == 64);
  for (float v : dec) CHECK(v == doctest::Approx(1.0));
}

TEST_CASE("dsv4-fp8_ds_mla: value 3.0 block -> UE8M0 scale byte 120, exact round-trip") {
  // absmax=3, raw=3/448 -> exponent=ceil(log2(3/448))=-7 -> byte=120.
  // decode: fp8(384=1.5*2^8) * 2^-7 -> 3.0.
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(64, 0, 64);
  std::vector<float> head(64, 3.0f);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  CHECK(static_cast<int>(t.scale_ue8m0[0]) == 120);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  for (float v : dec) CHECK(v == doctest::Approx(3.0));
}

TEST_CASE("dsv4-fp8_ds_mla: RoPE part is stored/read bf16 verbatim") {
  // 448 nope (zeros) + rope [1.5, 0.0, -2.0, 0.25] — all bf16-exact.
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 4, 64);
  std::vector<float> head(452, 0.0f);
  head[448] = 1.5f;
  head[449] = 0.0f;
  head[450] = -2.0f;
  head[451] = 0.25f;
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  REQUIRE(t.rope_bf16.size() == 4);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  CHECK(dec[448] == doctest::Approx(1.5));
  CHECK(dec[449] == doctest::Approx(0.0));
  CHECK(dec[450] == doctest::Approx(-2.0));
  CHECK(dec[451] == doctest::Approx(0.25));
}

// ── from-first-principles double-precision references (randomized) ────────────

TEST_CASE("dsv4-compressor: pool+norm vs independent double reference (rel-L2)") {
  std::mt19937 rng(20260729);
  std::uniform_real_distribution<float> u(-1.5f, 1.5f);
  const int64_t W = 5, D = 12;
  const float eps = 1e-6f;
  std::vector<float> kv(W * D), score(W * D), rms(D);
  std::vector<uint8_t> valid(W, 1);
  for (auto& x : kv) x = u(rng);
  for (auto& x : score) x = u(rng);
  for (auto& x : rms) x = u(rng);
  valid[1] = 0;  // exercise the mask
  valid[4] = 0;

  const std::vector<float> got = CompressorPoolNorm(kv, score, valid, rms, eps, W, D);

  // Independent double reference (per-column softmax pool, then RMSNorm).
  std::vector<double> comp(static_cast<size_t>(D), 0.0);
  for (int64_t d = 0; d < D; ++d) {
    double m = -1e300;
    for (int64_t i = 0; i < W; ++i)
      if (valid[i]) m = std::max(m, static_cast<double>(score[i * D + d]));
    double denom = 0.0, acc = 0.0;
    for (int64_t i = 0; i < W; ++i) {
      if (!valid[i]) continue;
      const double e = std::exp(static_cast<double>(score[i * D + d]) - m);
      denom += e;
      acc += static_cast<double>(kv[i * D + d]) * e;
    }
    comp[d] = acc / denom;
  }
  double var = 0.0;
  for (int64_t d = 0; d < D; ++d) var += comp[d] * comp[d];
  var /= D;
  const double rrms = 1.0 / std::sqrt(var + eps);
  std::vector<double> ref(static_cast<size_t>(D));
  for (int64_t d = 0; d < D; ++d) ref[d] = comp[d] * rrms * rms[d];

  CHECK(RelL2(got, ref) < 1e-6);
}

TEST_CASE("dsv4-fp8_ds_mla: UE8M0 scale bytes match an independent recompute") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-6.0f, 6.0f);
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(256, 0, 64);  // 4 nope blocks
  std::vector<float> head(256);
  for (auto& x : head) x = u(rng);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);

  // Independent recompute of each block's UE8M0 exponent byte.
  REQUIRE(t.scale_ue8m0.size() == 4);
  for (int64_t b = 0; b < 4; ++b) {
    double absmax = 0.0;
    for (int64_t j = 0; j < 64; ++j)
      absmax = std::max(absmax, std::fabs(static_cast<double>(head[b * 64 + j])));
    absmax = std::max(absmax, 1e-4);
    const double exponent = std::ceil(std::log2(absmax / 448.0));
    int expected = static_cast<int>(exponent) + 127;
    expected = std::max(0, std::min(255, expected));
    CHECK(static_cast<int>(t.scale_ue8m0[static_cast<size_t>(b)]) == expected);
  }
}

TEST_CASE("dsv4-fp8_ds_mla: decode(encode(x)) round-trips within fp8 granularity") {
  std::mt19937 rng(101);
  std::uniform_real_distribution<float> u(-4.0f, 4.0f);
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 64, 64);
  std::vector<float> head(512);
  for (auto& x : head) x = u(rng);

  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);

  // e4m3 carries 3 mantissa bits -> worst-case per-element relative step ~2^-4;
  // with per-64-block power-of-two scaling the magnitude-weighted rel-L2 sits
  // well under that. bf16 rope is near-exact. Honest granularity bound, not 1e-6.
  std::vector<double> ref(head.begin(), head.end());
  CHECK(RelL2(dec, ref) < 0.05);
}

TEST_CASE("W1: the compressor CYCLE emits at boundaries only, and pools the closed window") {
  // `MODEL-DSV4-DSA-COMPOSE` W1 (#2286). The individual ops above are gated; this
  // gates the STATE MACHINE that drives them, which is where a compressor
  // actually goes wrong -- it is stateful across steps, so an error surfaces
  // several tokens later as a plausible value rather than immediately.
  //
  // Driven the way a forward drives it: a multi-token prefill step, then
  // single-token decode steps, with the state carried across.
  const int64_t hd = 4, cr = 3;
  const float eps = 1e-6f;
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  std::vector<float> ape(static_cast<size_t>(cr * hd));
  for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.01f * static_cast<float>(i + 1);

  auto row = [hd](float base) {
    std::vector<float> v(static_cast<size_t>(hd));
    for (int64_t d = 0; d < hd; ++d) v[static_cast<size_t>(d)] = base + 0.1f * static_cast<float>(d);
    return v;
  };

  std::vector<float> st_kv, st_sc;
  std::vector<float> all_emitted;
  int64_t emitted_steps = 0;

  // Step 1: a 4-token prefill at positions 0..3. Only position 2 is a boundary
  // ((2+1) % 3 == 0), so exactly ONE row must come out of a four-token step.
  {
    std::vector<float> kv, sc;
    std::vector<int64_t> pos;
    for (int64_t t = 0; t < 4; ++t) {
      const auto k = row(1.0f + static_cast<float>(t));
      const auto c = row(0.5f - 0.05f * static_cast<float>(t));
      kv.insert(kv.end(), k.begin(), k.end());
      sc.insert(sc.end(), c.begin(), c.end());
      pos.push_back(t);
    }
    const auto out = vllm::deepseek_v4::CompressorStepCycle(&st_kv, &st_sc, kv, sc, ape, pos,
                                                            rms, eps, cr, hd);
    CHECK(out.size() == static_cast<size_t>(hd));  // exactly one boundary crossed
    all_emitted.insert(all_emitted.end(), out.begin(), out.end());
    if (!out.empty()) ++emitted_steps;
  }

  // Steps 2-3: single tokens at positions 4 then 5. Position 5 is the boundary.
  for (int64_t p = 4; p <= 5; ++p) {
    const auto k = row(1.0f + static_cast<float>(p));
    const auto c = row(0.5f - 0.05f * static_cast<float>(p));
    const auto out = vllm::deepseek_v4::CompressorStepCycle(&st_kv, &st_sc, k, c, ape, {p},
                                                            rms, eps, cr, hd);
    if (p == 5) {
      CHECK(out.size() == static_cast<size_t>(hd));
      all_emitted.insert(all_emitted.end(), out.begin(), out.end());
      ++emitted_steps;
    } else {
      // NOT a boundary: nothing is emitted. A cycle that emitted every step
      // would still produce plausible compressed rows, just too many of them.
      CHECK(out.empty());
    }
  }

  CHECK(emitted_steps == 2);
  REQUIRE(all_emitted.size() == static_cast<size_t>(2 * hd));
  CHECK(st_kv.size() == static_cast<size_t>(6 * hd));  // every token retained

  // THE SECOND EMISSION IS THE WINDOW THAT JUST CLOSED -- positions 3,4,5 -- and
  // NOT positions 0..2 again. Computed here from the same gated helpers over the
  // hand-built window, so this is an independent expectation rather than a
  // second call to the function under test.
  {
    std::vector<float> wkv, wsc;
    for (int64_t p = 3; p <= 5; ++p) {
      const auto k = row(1.0f + static_cast<float>(p));
      const auto c = row(0.5f - 0.05f * static_cast<float>(p));
      const auto scored = vllm::deepseek_v4::CompressorSaveScoreApe(c, ape, {p}, 1, hd, cr);
      wkv.insert(wkv.end(), k.begin(), k.end());
      wsc.insert(wsc.end(), scored.begin(), scored.end());
    }
    const std::vector<uint8_t> valid(static_cast<size_t>(cr), 1);
    const auto want =
        vllm::deepseek_v4::CompressorPoolNorm(wkv, wsc, valid, rms, eps, cr, hd);
    REQUIRE(want.size() == static_cast<size_t>(hd));
    double worst = 0.0, mag = 0.0;
    for (int64_t d = 0; d < hd; ++d) {
      const double got = all_emitted[static_cast<size_t>(hd + d)];
      mag = std::max(mag, std::abs(static_cast<double>(want[static_cast<size_t>(d)])));
      worst = std::max(worst, std::abs(got - want[static_cast<size_t>(d)]));
    }
    REQUIRE(mag > 1e-4);
    CHECK(worst <= 1e-6 * mag);
    // And it DIFFERS from the first emission -- a cycle that re-pooled the same
    // window every time would satisfy every count above.
    double vs_first = 0.0;
    for (int64_t d = 0; d < hd; ++d)
      vs_first = std::max(vs_first, std::abs(static_cast<double>(
                                        all_emitted[static_cast<size_t>(d)] -
                                        all_emitted[static_cast<size_t>(hd + d)])));
    CHECK(vs_first > 1e-4 * mag);
  }
}

// ── W3 (#2286): the coff == 2 OVERLAPPED gathering window ───────────────────
//
// `fused_compress_quant_cache.py:169-183`, verbatim:
//     start  = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
//     tokens = arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
//     head_offset = (tokens >= COMPRESS_RATIO) * HEAD_SIZE
// A row's ROLE is its index within the gathering window, not a property of the
// row, which is why the tensors are doubled and why the refusal message says the
// role "is never recoverable from the tensor alone".

TEST_CASE("W3: coff == 2 gathers coff*ratio rows and splits their ROLE by index") {
  const int64_t cr = 4, hd = 2, coff = 2, W = coff * hd;
  // Each token's state row is [low_half | high_half]. Make the two halves
  // DISTINGUISHABLE so a wrong head_offset cannot look right: low = +1, high = -1
  // scaled by the token index.
  const int64_t T = 8;
  std::vector<float> kv(static_cast<size_t>(T * W), 0.0f), score(kv.size(), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < hd; ++d) {
      kv[static_cast<size_t>(t * W + d)] = 1.0f + static_cast<float>(t);        // low
      kv[static_cast<size_t>(t * W + hd + d)] = -(1.0f + static_cast<float>(t)); // high
    }
  }
  // A flat score makes the pool a plain mean over the gathered rows, so the
  // expected value is computable by hand.
  for (auto& v : score) v = 0.0f;
  const std::vector<float> ape(static_cast<size_t>(cr * W), 0.0f);
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  std::vector<int64_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = t;

  std::vector<float> st_kv, st_sc;
  const std::vector<float> emitted = vllm::deepseek_v4::CompressorStepCycle(
      &st_kv, &st_sc, kv, score, ape, pos, rms, /*eps=*/0.0f, cr, hd, /*rope_dim=*/0, /*rope_theta=*/10000.0, coff);

  // Boundaries at positions 3 and 7, so TWO rows are emitted.
  REQUIRE(emitted.size() == static_cast<size_t>(2 * hd));
  // The state carries the FULL doubled row per token.
  CHECK(st_kv.size() == static_cast<size_t>(T * W));

  // At position 7 the window is tokens 0..7. Indices 0..3 read the LOW half
  // (values +1..+4), indices 4..7 read the HIGH half (values -5..-8). With a flat
  // score the pool is the mean of those eight: (1+2+3+4-5-6-7-8)/8 = -2.0.
  // A gather that ignored head_offset would average +1..+8 = +4.5 instead, and a
  // gather that inverted the roles would give +2.0 -- all three are distinct.
  const double mean_second = (1.0 + 2.0 + 3.0 + 4.0 - 5.0 - 6.0 - 7.0 - 8.0) / 8.0;
  // RMSNorm with unit gamma divides by the row's own RMS; every channel of this
  // pooled row is identical, so the norm maps it to its own sign.
  const float got = emitted[static_cast<size_t>(hd)];
  CHECK(got == doctest::Approx(mean_second < 0 ? -1.0f : 1.0f).epsilon(1e-5));
  CHECK(mean_second < 0.0);  // the ROLE split is what makes it negative at all
}

TEST_CASE("W3: coff == 1 is unchanged by the overlap parameter") {
  // The default must stay byte-identical, or every landed coff==1 gate is
  // measuring a different function than it was written against.
  const int64_t cr = 2, hd = 3, T = 4;
  std::vector<float> kv(static_cast<size_t>(T * hd)), score(static_cast<size_t>(T * hd));
  for (size_t i = 0; i < kv.size(); ++i) {
    kv[i] = 0.1f * static_cast<float>(i + 1);
    score[i] = 0.05f * static_cast<float>((i * 7) % 11);
  }
  const std::vector<float> ape(static_cast<size_t>(cr * hd), 0.01f);
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  std::vector<int64_t> pos{0, 1, 2, 3};

  std::vector<float> a_kv, a_sc, b_kv, b_sc;
  const auto def = vllm::deepseek_v4::CompressorStepCycle(&a_kv, &a_sc, kv, score, ape,
                                                          pos, rms, 1e-6f, cr, hd);
  const auto one = vllm::deepseek_v4::CompressorStepCycle(&b_kv, &b_sc, kv, score, ape,
                                                          pos, rms, 1e-6f, cr, hd, /*rope_dim=*/0, /*rope_theta=*/10000.0, 1);
  REQUIRE(def.size() == one.size());
  for (size_t i = 0; i < def.size(); ++i) CHECK(def[i] == doctest::Approx(one[i]));
  CHECK(a_kv == b_kv);
}

TEST_CASE("W3: a coff outside {1, 2} REFUSES") {
  const int64_t cr = 4, hd = 2;
  const std::vector<float> kv(static_cast<size_t>(hd), 0.0f), score(kv.size(), 0.0f);
  const std::vector<float> ape(static_cast<size_t>(cr * hd), 0.0f);
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  std::vector<float> st_kv, st_sc;
  CHECK_THROWS(vllm::deepseek_v4::CompressorStepCycle(&st_kv, &st_sc, kv, score, ape,
                                                      {0}, rms, 0.0f, cr, hd, /*rope_dim=*/0, /*rope_theta=*/10000.0, 3));
}

// ── W3: the pooled row is ROTATED, at the WINDOW'S base position ─────────────
//
// `fused_compress_quant_cache.py:272-297` applies GPT-J RoPE to the rope tail of
// the pooled row, unconditionally -- the `rotate` constructor flag is dead and
// both compressors pass it. The position is
// `compressed_pos = (position / COMPRESS_RATIO) * COMPRESS_RATIO`, the window's
// BASE and not the emitting token's, so every window shares one phase.

TEST_CASE("W3: the pooled row's ROPE TAIL is rotated and its nope half is not") {
  const int64_t cr = 2, hd = 4, rd = 2, T = 2;
  std::vector<float> kv(static_cast<size_t>(T * hd), 0.0f), score(kv.size(), 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t d = 0; d < hd; ++d)
      kv[static_cast<size_t>(t * hd + d)] = 1.0f + static_cast<float>(d);
  const std::vector<float> ape(static_cast<size_t>(cr * hd), 0.0f);
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  const std::vector<int64_t> pos{0, 1};  // boundary closes at position 1

  std::vector<float> a_kv, a_sc, b_kv, b_sc;
  const auto unrot = vllm::deepseek_v4::CompressorStepCycle(
      &a_kv, &a_sc, kv, score, ape, pos, rms, 0.0f, cr, hd, /*rope_dim=*/0);
  const auto rot = vllm::deepseek_v4::CompressorStepCycle(
      &b_kv, &b_sc, kv, score, ape, pos, rms, 0.0f, cr, hd, /*rope_dim=*/rd,
      /*rope_theta=*/10000.0);
  REQUIRE(unrot.size() == static_cast<size_t>(hd));
  REQUIRE(rot.size() == unrot.size());

  // The NOPE half is untouched by the rotation.
  for (int64_t d = 0; d < hd - rd; ++d)
    CHECK(rot[static_cast<size_t>(d)] == doctest::Approx(unrot[static_cast<size_t>(d)]));

  // The rope tail IS touched. `compressed_pos = (1 / 2) * 2 = 0`, and RoPE at
  // position 0 is the identity -- so at THIS boundary the two must still agree,
  // which is the window-base rule showing itself.
  for (int64_t d = hd - rd; d < hd; ++d)
    CHECK(rot[static_cast<size_t>(d)] == doctest::Approx(unrot[static_cast<size_t>(d)]));
}

TEST_CASE("W3: the rotation uses the WINDOW's base position, not the token's") {
  // The distinguishing case. At `cr == 2`, a boundary at position 3 has
  // `compressed_pos = (3 / 2) * 2 = 2`, NOT 3. Rotating at 3 would be a different
  // phase, and both are finite and plausible.
  const int64_t cr = 2, hd = 4, rd = 2, T = 4;
  std::vector<float> kv(static_cast<size_t>(T * hd), 0.0f), score(kv.size(), 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t d = 0; d < hd; ++d)
      kv[static_cast<size_t>(t * hd + d)] = 1.0f + static_cast<float>(d);
  const std::vector<float> ape(static_cast<size_t>(cr * hd), 0.0f);
  const std::vector<float> rms(static_cast<size_t>(hd), 1.0f);
  const std::vector<int64_t> pos{0, 1, 2, 3};

  std::vector<float> st_kv, st_sc;
  const auto got = vllm::deepseek_v4::CompressorStepCycle(
      &st_kv, &st_sc, kv, score, ape, pos, rms, 0.0f, cr, hd, rd, 10000.0);
  REQUIRE(got.size() == static_cast<size_t>(2 * hd));  // boundaries at 1 and 3

  // Build the SECOND row's expectation by hand: pool (unrotated) then rotate at
  // compressed_pos = 2.
  std::vector<float> u_kv, u_sc;
  const auto unrot = vllm::deepseek_v4::CompressorStepCycle(
      &u_kv, &u_sc, kv, score, ape, pos, rms, 0.0f, cr, hd, /*rope_dim=*/0);
  std::vector<float> expect(unrot.begin() + hd, unrot.end());
  vllm::deepseek_v4::RopeInplaceLayer(expect.data() + (hd - rd), rd, /*pos=*/2,
                                      10000.0, 1.0, 0.0, 0, 0.0, 0.0);
  for (int64_t d = 0; d < hd; ++d)
    CHECK(got[static_cast<size_t>(hd + d)] ==
          doctest::Approx(expect[static_cast<size_t>(d)]).epsilon(1e-5));

  // And rotating at the TOKEN's position (3) would differ, or the case proves
  // nothing about which position was used.
  std::vector<float> wrong(unrot.begin() + hd, unrot.end());
  vllm::deepseek_v4::RopeInplaceLayer(wrong.data() + (hd - rd), rd, /*pos=*/3,
                                      10000.0, 1.0, 0.0, 0, 0.0, 0.0);
  double sep = 0.0;
  for (int64_t d = 0; d < hd; ++d)
    sep = std::max(sep, std::abs(static_cast<double>(expect[static_cast<size_t>(d)] -
                                                     wrong[static_cast<size_t>(d)])));
  CHECK(sep > 1e-4);
}
