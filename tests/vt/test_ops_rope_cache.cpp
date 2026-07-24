// Ported from:
//   tests/kernels/core/test_pos_encoding.py:54-193
//   tests/kernels/core/test_rotary_embedding.py:14-79
//   tests/kernels/core/test_mrope.py:16-126
// and the reference selection in
//   vllm/model_executor/layers/rotary_embedding/mrope.py:14-198
// @ e24d1b24fe96.
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

int AxisForPair(int64_t pair, const vt::RopeArgs& args) {
  if (args.mrope_interleaved) {
    if (pair % 3 == 1 && pair <= 3LL * args.mrope_section[1]) return 1;
    if (pair % 3 == 2 && pair <= 3LL * args.mrope_section[2]) return 2;
    return 0;
  }
  if (pair < args.mrope_section[0]) return 0;
  if (pair < static_cast<int64_t>(args.mrope_section[0]) +
                 args.mrope_section[1]) {
    return 1;
  }
  return 2;
}

void Reference(std::vector<float>& states, int64_t tokens, int64_t heads,
               int64_t head_dim, const std::vector<int64_t>& positions,
               bool mrope, const std::vector<float>& cache, int64_t cache_rows,
               const vt::RopeArgs& args) {
  const int64_t half = args.rotary_dim / 2;
  REQUIRE(static_cast<int64_t>(cache.size()) == cache_rows * args.rotary_dim);
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t pair = 0; pair < half; ++pair) {
      const int axis = mrope ? AxisForPair(pair, args) : 0;
      const int64_t position =
          positions[static_cast<size_t>(axis * tokens + token)];
      REQUIRE(position >= 0);
      REQUIRE(position < cache_rows);
      const float c = cache[static_cast<size_t>(position * args.rotary_dim + pair)];
      const float s =
          cache[static_cast<size_t>(position * args.rotary_dim + half + pair)];
      const int64_t first = args.is_neox_style ? pair : pair * 2;
      const int64_t second =
          args.is_neox_style ? pair + half : pair * 2 + 1;
      for (int64_t head = 0; head < heads; ++head) {
        const int64_t row = (token * heads + head) * head_dim;
        const float x = states[static_cast<size_t>(row + first)];
        const float y = states[static_cast<size_t>(row + second)];
        states[static_cast<size_t>(row + first)] = x * c - y * s;
        states[static_cast<size_t>(row + second)] = x * s + y * c;
      }
    }
  }
}

void CheckClose(const std::vector<float>& got, const std::vector<float>& want,
                double epsilon = 1e-6) {
  REQUIRE(got.size() == want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(want[i]).epsilon(epsilon));
  }
}

}  // namespace

TEST_CASE("rope_from_cache applies NeoX cache and rotates optional key") {
  constexpr int64_t kTokens = 2;
  constexpr int64_t kHeadDim = 6;
  constexpr int kRotaryDim = 4;
  std::vector<float> cache = {
      1.0F, 1.0F, 0.0F, 0.0F,
      0.5F, 0.25F, std::sqrt(0.75F), std::sqrt(0.9375F),
  };
  std::vector<int64_t> positions = {1, 0};
  std::vector<float> query = {
      1, 2, 3, 4, 50, 60,
      -1, -2, -3, -4, 70, 80,
  };
  std::vector<float> key = {
      4, 3, 2, 1, 90, 100,
      -4, -3, -2, -1, 110, 120,
  };
  std::vector<float> query_want = query;
  std::vector<float> key_want = key;
  vt::RopeArgs args{10000.0F, kRotaryDim};
  Reference(query_want, kTokens, 1, kHeadDim, positions, false, cache, 2,
            args);
  Reference(key_want, kTokens, 1, kHeadDim, positions, false, cache, 2,
            args);

  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32, Cpu(),
                                         {kTokens, 1, kHeadDim});
  vt::Tensor tk = vt::Tensor::Contiguous(key.data(), vt::DType::kF32, Cpu(),
                                         {kTokens, 1, kHeadDim});
  vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI64,
                                         Cpu(), {kTokens});
  vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(),
                                         {2, kRotaryDim});
  vt::Queue queue{Cpu(), nullptr};
  vt::RopeFromCache(queue, tq, &tk, tp, tc, args);
  CheckClose(query, query_want);
  CheckClose(key, key_want);
  CHECK(query[4] == 50.0F);
  CHECK(query[5] == 60.0F);
}

TEST_CASE("rope_from_cache applies GPT-J pairs and permits a missing key") {
  std::vector<float> cache = {
      1.0F, 1.0F, 0.0F, 0.0F,
      0.0F, -1.0F, 1.0F, 0.0F,
  };
  std::vector<int32_t> positions = {1};
  std::vector<int64_t> reference_positions = {1};
  std::vector<float> query = {1, 2, 3, 4, 99, 100};
  std::vector<float> want = query;
  vt::RopeArgs args{10000.0F, 4};
  args.is_neox_style = false;
  Reference(want, 1, 1, 6, reference_positions, false, cache, 2, args);

  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32, Cpu(),
                                         {1, 1, 6});
  vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI32,
                                         Cpu(), {1});
  vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(),
                                         {2, 4});
  vt::Queue queue{Cpu(), nullptr};
  vt::RopeFromCache(queue, tq, nullptr, tp, tc, args);
  CheckClose(query, want);
  CHECK(query[4] == 99.0F);
  CHECK(query[5] == 100.0F);
}

// GLM-4-9B-0414 partial rotary: head_dim=128, partial_rotary_factor=0.5 ->
// rotary_dim=64, in BOTH NeoX and interleaved (is_neox_style=false, GLM's actual
// layout) forms. Asserts (a) the rotated leading 64 dims match the CPU reference
// off the PRODUCTION RopeCosSinCache, and (b) the trailing [64,128) dims pass
// through BIT-EXACTLY. This is the new-ops unit gate for the GLM partial-rotary
// primitive (spike G2). GQA shape (Hq=4 q-heads, Hk=1 k-head — GLM ratio 32/2).
TEST_CASE("rope_from_cache GLM partial rotary passes the tail through (both layouts)") {
  constexpr int64_t kTokens = 3;
  constexpr int64_t kHeadDim = 128;
  constexpr int kRotaryDim = 64;  // 0.5 * 128
  constexpr int64_t kHq = 4;
  constexpr int64_t kHk = 1;
  for (bool neox : {true, false}) {
    CAPTURE(neox);
    std::vector<int32_t> positions = {0, 5, 131};
    std::vector<int64_t> ref_positions(positions.begin(), positions.end());

    // Build the cos|sin cache via the production op (theta 1e4, rotary_dim 64).
    vt::RopeArgs args{10000.0F, kRotaryDim};
    args.is_neox_style = neox;
    const int64_t cache_rows = 200;
    std::vector<float> cache(static_cast<size_t>(cache_rows) * kRotaryDim, 0.0F);
    {
      std::vector<int64_t> cpos(static_cast<size_t>(cache_rows));
      for (int64_t i = 0; i < cache_rows; ++i) cpos[static_cast<size_t>(i)] = i;
      vt::Tensor tcp = vt::Tensor::Contiguous(cpos.data(), vt::DType::kI64, Cpu(),
                                              {cache_rows});
      vt::Tensor tcache = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32,
                                                 Cpu(), {cache_rows, kRotaryDim});
      vt::Queue cq{Cpu(), nullptr};
      vt::RopeCosSinCache(cq, tcache, tcp, args);
    }

    std::vector<float> query(static_cast<size_t>(kTokens * kHq * kHeadDim));
    std::vector<float> key(static_cast<size_t>(kTokens * kHk * kHeadDim));
    for (size_t i = 0; i < query.size(); ++i)
      query[i] = 0.01F * static_cast<float>((i % 37) - 18);
    for (size_t i = 0; i < key.size(); ++i)
      key[i] = 0.01F * static_cast<float>((i % 29) - 14);
    const std::vector<float> query0 = query;  // originals for the tail check
    const std::vector<float> key0 = key;

    std::vector<float> query_want = query;
    std::vector<float> key_want = key;
    Reference(query_want, kTokens, kHq, kHeadDim, ref_positions, false, cache,
              cache_rows, args);
    Reference(key_want, kTokens, kHk, kHeadDim, ref_positions, false, cache,
              cache_rows, args);

    vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32, Cpu(),
                                           {kTokens, kHq, kHeadDim});
    vt::Tensor tk = vt::Tensor::Contiguous(key.data(), vt::DType::kF32, Cpu(),
                                           {kTokens, kHk, kHeadDim});
    vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI32,
                                           Cpu(), {kTokens});
    vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(),
                                           {cache_rows, kRotaryDim});
    vt::Queue queue{Cpu(), nullptr};
    vt::RopeFromCache(queue, tq, &tk, tp, tc, args);
    CheckClose(query, query_want);
    CheckClose(key, key_want);

    // Tail [rotary_dim, head_dim) passes through BIT-EXACTLY (== not Approx).
    for (int64_t t = 0; t < kTokens; ++t) {
      for (int64_t h = 0; h < kHq; ++h) {
        const int64_t row = (t * kHq + h) * kHeadDim;
        for (int64_t dctx = kRotaryDim; dctx < kHeadDim; ++dctx) {
          const size_t idx = static_cast<size_t>(row + dctx);
          CHECK(query[idx] == query0[idx]);
        }
      }
      const int64_t krow = t * kHk * kHeadDim;
      for (int64_t dctx = kRotaryDim; dctx < kHeadDim; ++dctx) {
        const size_t idx = static_cast<size_t>(krow + dctx);
        CHECK(key[idx] == key0[idx]);
      }
    }
  }
}

TEST_CASE("rope_from_cache MRoPE selects contiguous and interleaved T/H/W") {
  constexpr int64_t kRows = 4;
  constexpr int kRotaryDim = 8;
  constexpr int64_t kHeadDim = 10;
  // Each row is one deliberately distinct angle shared by all four pairs.
  const std::array<float, kRows> angle = {
      0.0F, 0.35F, 0.9F, 1.7F};
  std::vector<float> cache(static_cast<size_t>(kRows * kRotaryDim));
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t pair = 0; pair < kRotaryDim / 2; ++pair) {
      cache[static_cast<size_t>(row * kRotaryDim + pair)] =
          std::cos(angle[static_cast<size_t>(row)]);
      cache[static_cast<size_t>(row * kRotaryDim + kRotaryDim / 2 + pair)] =
          std::sin(angle[static_cast<size_t>(row)]);
    }
  }
  // T=1, H=2, W=3 for the single token.
  std::vector<int64_t> positions = {1, 2, 3};
  const std::vector<float> initial = {1, 2, 3, 4, 5, 6, 7, 8, 90, 91};

  for (bool interleaved : {false, true}) {
    CAPTURE(interleaved);
    std::vector<float> query = initial;
    std::vector<float> key = initial;
    std::vector<float> query_want = initial;
    std::vector<float> key_want = initial;
    vt::RopeArgs args{10000.0F, kRotaryDim};
    args.mrope_section = {2, 1, 1};
    args.mrope_interleaved = interleaved;
    Reference(query_want, 1, 1, kHeadDim, positions, true, cache, kRows,
              args);
    Reference(key_want, 1, 1, kHeadDim, positions, true, cache, kRows,
              args);

    vt::Tensor tq = vt::Tensor::Contiguous(
        query.data(), vt::DType::kF32, Cpu(), {1, 1, kHeadDim});
    vt::Tensor tk = vt::Tensor::Contiguous(
        key.data(), vt::DType::kF32, Cpu(), {1, 1, kHeadDim});
    vt::Tensor tp = vt::Tensor::Contiguous(
        positions.data(), vt::DType::kI64, Cpu(), {3, 1});
    vt::Tensor tc = vt::Tensor::Contiguous(
        cache.data(), vt::DType::kF32, Cpu(), {kRows, kRotaryDim});
    vt::Queue queue{Cpu(), nullptr};
    vt::RopeFromCache(queue, tq, &tk, tp, tc, args);
    CheckClose(query, query_want);
    CheckClose(key, key_want);
    CHECK(query[8] == 90.0F);
    CHECK(query[9] == 91.0F);
  }
}

TEST_CASE("rope_from_cache bf16 matches bf16-rounded reference") {
  std::vector<float> cache_f32 = {0.5F, 0.25F, std::sqrt(0.75F),
                                  std::sqrt(0.9375F)};
  std::vector<uint16_t> cache(4);
  for (size_t i = 0; i < cache.size(); ++i) {
    cache[i] = vt::F32ToBF16(cache_f32[i]);
    cache_f32[i] = vt::BF16ToF32(cache[i]);
  }
  std::vector<float> query_f32 = {1.25F, -0.75F, 0.5F, 2.0F};
  std::vector<uint16_t> query(4);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = vt::F32ToBF16(query_f32[i]);
    query_f32[i] = vt::BF16ToF32(query[i]);
  }
  std::vector<int32_t> positions = {0};
  vt::RopeArgs args{10000.0F, 4};
  Reference(query_f32, 1, 1, 4, std::vector<int64_t>{0}, false,
            cache_f32, 1, args);
  for (float& value : query_f32) {
    value = vt::BF16ToF32(vt::F32ToBF16(value));
  }

  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kBF16,
                                         Cpu(), {1, 1, 4});
  vt::Tensor tp = vt::Tensor::Contiguous(positions.data(), vt::DType::kI32,
                                         Cpu(), {1});
  vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kBF16,
                                         Cpu(), {1, 4});
  vt::Queue queue{Cpu(), nullptr};
  vt::RopeFromCache(queue, tq, nullptr, tp, tc, args);
  for (size_t i = 0; i < query.size(); ++i) {
    CHECK(vt::BF16ToF32(query[i]) == query_f32[i]);
  }
}

TEST_CASE("rope_from_cache validates MRoPE sections, dtype, and CPU bounds") {
  std::vector<float> query(8, 1.0F);
  std::vector<float> cache(8, 0.0F);
  std::vector<uint16_t> cache_bf16(8, 0);
  std::vector<int64_t> mpositions = {0, 0, 0};
  std::vector<int64_t> bad_position = {2};
  vt::Tensor tq = vt::Tensor::Contiguous(query.data(), vt::DType::kF32, Cpu(),
                                         {1, 1, 8});
  vt::Tensor tm = vt::Tensor::Contiguous(mpositions.data(), vt::DType::kI64,
                                         Cpu(), {3, 1});
  vt::Tensor tb = vt::Tensor::Contiguous(bad_position.data(), vt::DType::kI64,
                                         Cpu(), {1});
  vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(),
                                         {1, 8});
  vt::Tensor tcb = vt::Tensor::Contiguous(
      cache_bf16.data(), vt::DType::kBF16, Cpu(), {1, 8});
  vt::Queue queue{Cpu(), nullptr};
  vt::RopeArgs args{10000.0F, 8};
  args.mrope_section = {1, 1, 1};  // sum 3, must be 4
  CHECK_THROWS_WITH_AS(vt::RopeFromCache(queue, tq, nullptr, tm, tc, args),
                       doctest::Contains("must sum"), std::runtime_error);
  args.mrope_section = {2, 1, 1};
  CHECK_THROWS_WITH_AS(vt::RopeFromCache(queue, tq, nullptr, tm, tcb, args),
                       doctest::Contains("share f32 or bf16"),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(vt::RopeFromCache(queue, tq, nullptr, tb, tc, args),
                       doctest::Contains("outside cache"),
                       std::runtime_error);
}
