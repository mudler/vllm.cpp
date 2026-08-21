// BIT-IDENTITY gate for the row-wise parallelisation of the host-reference
// kernels (#672, spec `.agents/specs/minimax-music3.md` §11.4).
//
// `music3::LinearNoBias` and `vocoder1d::ConvTranspose1d` were sequential
// scalar loops. They now partition their OUTPUT elements across the `vt::cpu`
// threadpool. The claim that makes that a refactor rather than a numeric change
// is that no reduction order moves, and this file is the instrument for exactly
// that claim.
//
// WHAT THE ORACLE IS, AND WHY IT IS NOT THE SHIPPED CODE. Each case compares
// the shipped function against `SerialLinearNoBias` / `SerialConv1d` /
// `SerialConvTranspose1d` below. Comparing the shipped function to itself at
// two thread counts would prove only that it is deterministic; it would pass
// just as happily if the pivot had reassociated every sum, because a
// consistently reassociated sum is still consistent. A separate scalar
// transcription of the declared order is the only oracle that can see that, so
// it is carried here rather than referenced.
//
// WHAT THOSE REFERENCES NO LONGER ARE, said plainly. The two convolution ones
// were VERBATIM copies of the loops as they stood at `d9441ef3`, and that is
// what let them assert agreement with the pre-op host loop itself. #1474
// narrowed their accumulators from `double` to `float` in lockstep with the
// kernels, so they are transcriptions of the declared ORDER at the current
// WIDTH, and nothing here still compares against `d9441ef3`'s arithmetic. What
// they gate is unchanged in kind — that no reduction order moved, at any thread
// count — but the standing is weaker and pretending otherwise would be the
// defect. `SerialLinearNoBias` is untouched and is still the `d9441ef3` copy.
//
// A self-referential gate cannot see its own width, either: widening a kernel
// and its reference together leaves every case here green. That is what the two
// `accumulates in f32, which is what torch does` cases at the end of this file
// are for, and they assert torch's own measured answer rather than ours.
//
// EQUALITY IS BITWISE, NOT `Approx`. A tolerance would defeat the purpose:
// every defect this file exists to catch — a split accumulator, a reordered
// scatter, an atomic — lands well inside any epsilon anyone would write, and
// doctest's default `Approx` scale puts a 1.19e-5 absolute floor under one
// anyway.
//
// AND THE PARALLEL PATH IS ASSERTED, NOT ASSUMED. `host_parallel::ForOutputRows`
// runs the body inline below `kMinParallelWork`, so a case sized under the
// guard tests the serial path twice and reports a green that means nothing.
// Every case states its work product and REQUIREs it over the threshold.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/host_parallel.h"  // via -I src: kMinParallelWork
#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting
#include "vt/dtype.h"

namespace {

using vllm::models::music3::ArCompute;

// A deterministic, reproducible spread of values with a mix of magnitudes and
// signs — a plain LCG, so the case is identical on every box and architecture.
std::vector<float> Spread(size_t n, uint32_t seed, bool with_zeros = false) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525U + 1013904223U;
    // Magnitudes across ~6 decades, so a reassociated sum loses different bits
    // than the sequential one rather than cancelling into agreement.
    const double mantissa = static_cast<double>(s >> 8) / 16777216.0 - 0.5;
    const int exponent = static_cast<int>((s >> 4) & 0xFU) - 8;
    v[i] = static_cast<float>(mantissa * std::pow(2.0, exponent));
    // ConvTranspose1d skips zero inputs; the skip must survive the pivot, so
    // some cases must actually contain zeros.
    if (with_zeros && ((s >> 20) & 0x7U) == 0U) v[i] = 0.0F;
  }
  return v;
}

// `Store` as minimax_music3_ar.cpp:33-36 defines it (TU-private there).
float SerialStore(double value, ArCompute compute) {
  const float narrowed = static_cast<float>(value);
  return compute == ArCompute::kBFloat16 ? vt::BF16ToF32(vt::F32ToBF16(narrowed)) : narrowed;
}

// VERBATIM minimax_music3_ar.cpp @ d9441ef3 :563-572.
std::vector<float> SerialLinearNoBias(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                                      const std::vector<float>& weight, int64_t out_dim,
                                      ArCompute compute) {
  std::vector<float> out(static_cast<size_t>(rows * out_dim));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      const float* xr = x.data() + r * in_dim;
      const float* wo = weight.data() + o * in_dim;
      for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(xr[i]) * wo[i];
      out[static_cast<size_t>(r * out_dim + o)] = SerialStore(acc, compute);
    }
  }
  return out;
}

// vocoder1d.cpp @ d9441ef3 :70-92, verbatim EXCEPT the accumulator, which
// #1474 narrowed from `double` to `float` in lockstep with the kernel.
std::vector<float> SerialConv1d(const std::vector<float>& in, int64_t in_channels, int64_t in_len,
                                const std::vector<float>& weight, const std::vector<float>* bias,
                                int64_t out_channels, int64_t kernel, int64_t stride,
                                int64_t dilation, int64_t groups, int64_t* out_len) {
  const int64_t effective = dilation * (kernel - 1) + 1;
  const int64_t length = (in_len - effective) / stride + 1;
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<float> out(static_cast<size_t>(out_channels * length), 0.0f);
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    const int64_t g = oc / out_per_group;
    for (int64_t t = 0; t < length; ++t) {
      float acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0F;
      for (int64_t ic = 0; ic < in_per_group; ++ic) {
        const int64_t src_c = g * in_per_group + ic;
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t pos = t * stride + k * dilation;
          acc += in[static_cast<size_t>(src_c * in_len + pos)] *
                 weight[static_cast<size_t>((oc * in_per_group + ic) * kernel + k)];
        }
      }
      out[static_cast<size_t>(oc * length + t)] = acc;
    }
  }
  *out_len = length;
  return out;
}

// vocoder1d.cpp @ d9441ef3 :100-129, verbatim EXCEPT the accumulator, which
// #1474 narrowed from `double` to `float` in lockstep with the kernel.
std::vector<float> SerialConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                         int64_t in_len, const std::vector<float>& weight,
                                         const std::vector<float>* bias, int64_t out_channels,
                                         int64_t kernel, int64_t stride, int64_t padding,
                                         int64_t groups, int64_t* out_len) {
  const int64_t full = (in_len - 1) * stride + kernel;
  const int64_t length = full - 2 * padding;
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<float> acc(static_cast<size_t>(out_channels * full), 0.0F);
  for (int64_t ic = 0; ic < in_channels; ++ic) {
    const int64_t g = ic / in_per_group;
    for (int64_t t = 0; t < in_len; ++t) {
      const float value = in[static_cast<size_t>(ic * in_len + t)];
      if (value == 0.0F) continue;
      for (int64_t oc = 0; oc < out_per_group; ++oc) {
        const int64_t dst_c = g * out_per_group + oc;
        for (int64_t k = 0; k < kernel; ++k) {
          acc[static_cast<size_t>(dst_c * full + t * stride + k)] +=
              value * weight[static_cast<size_t>((ic * out_per_group + oc) * kernel + k)];
        }
      }
    }
  }
  std::vector<float> out(static_cast<size_t>(out_channels * length));
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      float value = acc[static_cast<size_t>(c * full + t + padding)];
      if (bias != nullptr) value += (*bias)[static_cast<size_t>(c)];
      out[static_cast<size_t>(c * length + t)] = value;
    }
  }
  *out_len = length;
  return out;
}

// Bitwise equality, reported with the first offending index rather than as a
// bare boolean — a count of mismatches is what tells a reordered scatter (a
// handful) from a wrong index map (nearly all of them).
void RequireBitIdentical(const std::vector<float>& got, const std::vector<float>& want,
                         const std::string& what) {
  REQUIRE_MESSAGE(got.size() == want.size(), what << ": size " << got.size() << " vs " << want.size());
  size_t first_bad = want.size();
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  INFO(what << ": " << bad << " of " << want.size() << " values differ; first at index "
             << first_bad);
  CHECK(bad == 0);
}

// Thread counts the determinism cases sweep. 1 proves the guard's inline path;
// the rest straddle the chunk grid (`ParallelForRows` oversubscribes 4x), and
// counts that do not divide the row count are the ones that catch an off-by-one
// in a partition.
const int kThreadCounts[] = {1, 2, 3, 7, 13};

}  // namespace

TEST_CASE("music3 LinearNoBias is bit-identical to the pre-parallel serial loop") {
  // Depth-decoder-shaped: rows 16 (its position window), in 512, out 384.
  const int64_t rows = 16, in_dim = 512, out_dim = 384;
  const int64_t work = rows * out_dim * in_dim;
  REQUIRE_MESSAGE(rows * out_dim * in_dim >= vllm::host_parallel::kMinParallelWork,
                  "case is under the size guard and would test the inline path twice; work="
                      << work);

  const std::vector<float> x = Spread(static_cast<size_t>(rows * in_dim), 0x1234u);
  const std::vector<float> w = Spread(static_cast<size_t>(out_dim * in_dim), 0x9E37u);

  for (const ArCompute compute : {ArCompute::kFloat32, ArCompute::kBFloat16}) {
    const std::vector<float> want = SerialLinearNoBias(x, rows, in_dim, w, out_dim, compute);
    for (const int threads : kThreadCounts) {
      vt::cpu::Threadpool pool(threads);
      vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
      const std::vector<float> got =
          vllm::models::music3::LinearNoBias(x, rows, in_dim, w, out_dim, compute);
      vt::cpu::Threadpool::SwapForTesting(previous);
      RequireBitIdentical(got, want,
                          std::string("LinearNoBias threads=") + std::to_string(threads) +
                              " compute=" +
                              (compute == ArCompute::kBFloat16 ? "bf16" : "f32"));
    }
  }
}

TEST_CASE("music3 LinearNoBias is bit-identical under CATASTROPHIC CANCELLATION") {
  // WHY THIS CASE EXISTS, and it is the finding of this change rather than a
  // flourish. With ordinary well-scaled inputs the case above CANNOT see a
  // reduction-order change at all: the accumulator is a `double` and the result
  // is stored as a `float`, so a reassociated sum differs by ~2^-53 relative
  // while the store rounds at 2^-24 — the narrowing swallows it. Mutating the
  // dot product into two interleaved accumulators left every one of those
  // assertions GREEN (mutation M1). A gate that stays green under the exact
  // defect it exists to catch is not a gate.
  //
  // What restores its teeth is engineering the cancellation the accumulator
  // otherwise hides: taps 0 and 1 carry +A and -A with A = 2^30, so the serial
  // order cancels them IMMEDIATELY and accumulates the small remainder exactly,
  // while ANY split that separates them accumulates the remainder against a
  // 2^30 magnitude and loses ~30 bits of it before the two halves meet. The
  // difference then lands far above the float store's ULP, and M1 fires.
  const int64_t rows = 8, in_dim = 1024, out_dim = 96;
  REQUIRE(rows * out_dim * in_dim >= vllm::host_parallel::kMinParallelWork);

  std::vector<float> x = Spread(static_cast<size_t>(rows * in_dim), 0x2A2Au);
  std::vector<float> w = Spread(static_cast<size_t>(out_dim * in_dim), 0x3B3Bu);
  const float kBig = 1073741824.0F;  // 2^30, exactly representable
  for (int64_t r = 0; r < rows; ++r) {
    x[static_cast<size_t>(r * in_dim + 0)] = 1.0F;
    x[static_cast<size_t>(r * in_dim + 1)] = 1.0F;
  }
  for (int64_t o = 0; o < out_dim; ++o) {
    w[static_cast<size_t>(o * in_dim + 0)] = kBig;
    w[static_cast<size_t>(o * in_dim + 1)] = -kBig;
  }

  for (const ArCompute compute : {ArCompute::kFloat32, ArCompute::kBFloat16}) {
    const std::vector<float> want = SerialLinearNoBias(x, rows, in_dim, w, out_dim, compute);
    for (const int threads : kThreadCounts) {
      vt::cpu::Threadpool pool(threads);
      vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
      const std::vector<float> got =
          vllm::models::music3::LinearNoBias(x, rows, in_dim, w, out_dim, compute);
      vt::cpu::Threadpool::SwapForTesting(previous);
      RequireBitIdentical(got, want,
                          std::string("LinearNoBias cancellation threads=") +
                              std::to_string(threads) + " compute=" +
                              (compute == ArCompute::kBFloat16 ? "bf16" : "f32"));
    }
  }
}

TEST_CASE("music3 LinearNoBias keeps a single row bit-identical") {
  // rows == 1 is the decode-step shape and the one the flat index must still
  // map correctly: `e / out_dim` is 0 for every element.
  const int64_t rows = 1, in_dim = 1024, out_dim = 1024;
  const std::vector<float> x = Spread(static_cast<size_t>(rows * in_dim), 0x51ABu);
  const std::vector<float> w = Spread(static_cast<size_t>(out_dim * in_dim), 0x77C1u);
  const std::vector<float> want =
      SerialLinearNoBias(x, rows, in_dim, w, out_dim, ArCompute::kBFloat16);
  for (const int threads : kThreadCounts) {
    vt::cpu::Threadpool pool(threads);
    vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
    const std::vector<float> got =
        vllm::models::music3::LinearNoBias(x, rows, in_dim, w, out_dim, ArCompute::kBFloat16);
    vt::cpu::Threadpool::SwapForTesting(previous);
    RequireBitIdentical(got, want,
                        std::string("LinearNoBias rows=1 threads=") + std::to_string(threads));
  }
}

TEST_CASE("vocoder1d ConvTranspose1d is bit-identical to the pre-parallel serial scatter") {
  struct Shape {
    const char* name;
    int64_t in_channels, in_len, out_channels, kernel, stride, padding, groups;
    bool with_bias, with_zeros;
  };
  // The first row is the MiniMax-Music3 vocoder's own first upsample stage in
  // miniature (stride 8, kernel 16 = 2*stride, padding (K-stride)/2), which is
  // the geometry `minimax_music3_loader.cpp:228` records. The rest exercise the
  // axes the pivot could plausibly get wrong: groups > 1 (the destination
  // channel's input range), zero padding vs a trimmed one, a kernel shorter
  // than the stride (so the scatter leaves gaps), and inputs containing exact
  // zeros (so the `value == 0.0` skip is on the path).
  const Shape shapes[] = {
      {"music3-up0", 96, 64, 48, 16, 8, 4, 1, true, false},
      {"groups=4", 64, 48, 96, 6, 3, 1, 4, true, false},
      {"kernel<stride", 48, 96, 64, 3, 5, 0, 1, false, false},
      {"zeros+no-bias", 80, 80, 80, 7, 2, 2, 1, false, true},
      {"depthwise-ish", 128, 512, 128, 4, 2, 0, 128, true, true},
  };

  for (const Shape& s : shapes) {
    const std::string shape_name(s.name);
    CAPTURE(shape_name);
    const int64_t in_per_group = s.in_channels / s.groups;
    const int64_t work = s.out_channels * in_per_group * s.in_len * s.kernel;
    REQUIRE_MESSAGE(work >= vllm::host_parallel::kMinParallelWork,
                    "shape is under the size guard and would test the inline path twice; work="
                        << work);

    const std::vector<float> in =
        Spread(static_cast<size_t>(s.in_channels * s.in_len), 0xC0FFu + s.kernel, s.with_zeros);
    const std::vector<float> weight = Spread(
        static_cast<size_t>(s.in_channels * (s.out_channels / s.groups) * s.kernel), 0xBEEFu);
    const std::vector<float> bias = Spread(static_cast<size_t>(s.out_channels), 0x0B1Au);
    const std::vector<float>* bias_ptr = s.with_bias ? &bias : nullptr;

    int64_t want_len = 0;
    const std::vector<float> want =
        SerialConvTranspose1d(in, s.in_channels, s.in_len, weight, bias_ptr, s.out_channels,
                              s.kernel, s.stride, s.padding, s.groups, &want_len);

    for (const int threads : kThreadCounts) {
      vt::cpu::Threadpool pool(threads);
      vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
      int64_t got_len = 0;
      const std::vector<float> got = vllm::vocoder1d::ConvTranspose1d(
          in, s.in_channels, s.in_len, weight, bias_ptr, s.out_channels, s.kernel, s.stride,
          s.padding, s.groups, &got_len);
      vt::cpu::Threadpool::SwapForTesting(previous);
      CHECK(got_len == want_len);
      RequireBitIdentical(got, want,
                          std::string("ConvTranspose1d ") + s.name + " threads=" +
                              std::to_string(threads));
    }
  }
}

TEST_CASE("vocoder1d Conv1d is bit-identical to the pre-parallel serial loop") {
  struct Shape {
    const char* name;
    int64_t in_channels, in_len, out_channels, kernel, stride, dilation, groups;
    bool with_bias;
  };
  const Shape shapes[] = {
      {"dilated", 64, 512, 64, 7, 1, 3, 1, true},
      {"strided", 96, 256, 128, 5, 2, 1, 1, true},
      {"grouped", 64, 384, 64, 3, 1, 1, 4, false},
      {"pointwise", 256, 128, 256, 1, 1, 1, 1, true},
  };
  for (const Shape& s : shapes) {
    const std::string shape_name(s.name);
    CAPTURE(shape_name);
    const int64_t effective = s.dilation * (s.kernel - 1) + 1;
    const int64_t length = (s.in_len - effective) / s.stride + 1;
    const int64_t work = s.out_channels * length * (s.in_channels / s.groups) * s.kernel;
    REQUIRE_MESSAGE(work >= vllm::host_parallel::kMinParallelWork,
                    "shape is under the size guard; work=" << work);

    const std::vector<float> in =
        Spread(static_cast<size_t>(s.in_channels * s.in_len), 0xD1CEu + s.kernel);
    const std::vector<float> weight = Spread(
        static_cast<size_t>(s.out_channels * (s.in_channels / s.groups) * s.kernel), 0xF00Du);
    const std::vector<float> bias = Spread(static_cast<size_t>(s.out_channels), 0x4E2Bu);
    const std::vector<float>* bias_ptr = s.with_bias ? &bias : nullptr;

    int64_t want_len = 0;
    const std::vector<float> want =
        SerialConv1d(in, s.in_channels, s.in_len, weight, bias_ptr, s.out_channels, s.kernel,
                     s.stride, s.dilation, s.groups, &want_len);
    for (const int threads : kThreadCounts) {
      vt::cpu::Threadpool pool(threads);
      vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
      int64_t got_len = 0;
      const std::vector<float> got =
          vllm::vocoder1d::Conv1d(in, s.in_channels, s.in_len, weight, bias_ptr, s.out_channels,
                                  s.kernel, s.stride, s.dilation, s.groups, &got_len);
      vt::cpu::Threadpool::SwapForTesting(previous);
      CHECK(got_len == want_len);
      RequireBitIdentical(got, want,
                          std::string("Conv1d ") + s.name + " threads=" + std::to_string(threads));
    }
  }
}

TEST_CASE("vocoder1d Conv1d is bit-identical under CATASTROPHIC CANCELLATION") {
  // The companion to the LinearNoBias cancellation case, and it exists for the
  // same measured reason: reversing Conv1d's `ic` walk left the four ordinary
  // shapes above entirely GREEN (mutation M7), because the accumulator was a
  // `double` stored through a `float` and could not show a ~2^-53 relative
  // change. #1474 narrowed that accumulator, which makes the ordinary shapes
  // MORE sensitive to an order change rather than less — but the case stays,
  // because a gate is not retired on the argument that it is now redundant.
  //
  // Here the BIAS is -2^40 and input channel 0 is all ones with a +2^40 tap, so
  // the serial (ic, k) walk cancels the two on its FIRST tap and accumulates the
  // 255 small channels exactly. Any walk that reaches that tap later carries
  // 2^40 through the small terms and quantises them at ~1.2e-4, far above the
  // float store's ULP for a result of that size. M7 then fires.
  const int64_t in_channels = 256, in_len = 512, out_channels = 256;
  const int64_t kernel = 1, stride = 1, dilation = 1, groups = 1;
  const float kBig = 1099511627776.0F;  // 2^40, exactly representable
  std::vector<float> in = Spread(static_cast<size_t>(in_channels * in_len), 0x6C6Cu);
  std::vector<float> weight = Spread(static_cast<size_t>(out_channels * in_channels), 0x7D7Du);
  std::vector<float> bias(static_cast<size_t>(out_channels), -kBig);
  for (int64_t t = 0; t < in_len; ++t) in[static_cast<size_t>(t)] = 1.0F;
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    weight[static_cast<size_t>(oc * in_channels)] = kBig;
  }

  int64_t want_len = 0;
  const std::vector<float> want = SerialConv1d(in, in_channels, in_len, weight, &bias,
                                               out_channels, kernel, stride, dilation, groups,
                                               &want_len);
  for (const int threads : kThreadCounts) {
    vt::cpu::Threadpool pool(threads);
    vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
    int64_t got_len = 0;
    const std::vector<float> got =
        vllm::vocoder1d::Conv1d(in, in_channels, in_len, weight, &bias, out_channels, kernel,
                                stride, dilation, groups, &got_len);
    vt::cpu::Threadpool::SwapForTesting(previous);
    CHECK(got_len == want_len);
    RequireBitIdentical(got, want,
                        std::string("Conv1d cancellation threads=") + std::to_string(threads));
  }
}

TEST_CASE("vocoder1d ConvTranspose1d is bit-identical under CATASTROPHIC CANCELLATION") {
  // The transposed op had no cancellation case until #672 moved its body behind
  // `vt::ConvTranspose1d`, and it needs one for the same measured reason the
  // other two do: with well-scaled taps a `double` accumulator stored through a
  // `float` could not show a reduction-order change at all, so the five ordinary
  // shapes above stayed green under a reassociated sweep. The accumulator is
  // `float` since #1474; the case is kept for the reason given on the Conv1d
  // one.
  //
  // The engineered cancellation is on the INPUT-CHANNEL axis, because that is
  // the axis a GATHER transcription of the scatter has to get right — and the
  // CUDA provider this row adds is exactly such a transcription. Input channels
  // 0 and 1 carry +2^40 and -2^40 at every position and share a weight row, so
  // the serial `ic` walk cancels them on its first two visits and accumulates
  // the remaining 30 channels exactly; any order that separates them carries
  // 2^40 through the small terms and quantises them at ~1.2e-4, far above the
  // float store's ULP at that magnitude.
  const int64_t in_channels = 32, in_len = 96, out_channels = 32;
  const int64_t kernel = 8, stride = 4, padding = 2, groups = 1;
  const float kBig = 1099511627776.0F;  // 2^40, exactly representable
  std::vector<float> in = Spread(static_cast<size_t>(in_channels * in_len), 0x5A5Au);
  std::vector<float> w =
      Spread(static_cast<size_t>(in_channels * out_channels * kernel), 0xA5A5u);
  for (int64_t t = 0; t < in_len; ++t) {
    in[static_cast<size_t>(0 * in_len + t)] = kBig;
    in[static_cast<size_t>(1 * in_len + t)] = -kBig;
  }
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t k = 0; k < kernel; ++k) {
      w[static_cast<size_t>((1 * out_channels + oc) * kernel + k)] =
          w[static_cast<size_t>((0 * out_channels + oc) * kernel + k)];
    }
  }

  const int64_t work = out_channels * in_channels * in_len * kernel;
  REQUIRE_MESSAGE(work >= vllm::host_parallel::kMinParallelWork,
                  "case is under the size guard; work=" << work);

  int64_t want_len = 0;
  const std::vector<float> want =
      SerialConvTranspose1d(in, in_channels, in_len, w, /*bias=*/nullptr, out_channels, kernel,
                            stride, padding, groups, &want_len);
  for (const int threads : kThreadCounts) {
    vt::cpu::Threadpool pool(threads);
    vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);
    int64_t got_len = 0;
    const std::vector<float> got =
        vllm::vocoder1d::ConvTranspose1d(in, in_channels, in_len, w, /*bias=*/nullptr,
                                         out_channels, kernel, stride, padding, groups, &got_len);
    vt::cpu::Threadpool::SwapForTesting(previous);
    CHECK(got_len == want_len);
    RequireBitIdentical(got, want, std::string("ConvTranspose1d cancellation threads=") +
                                       std::to_string(threads));
  }
}

TEST_CASE("host_parallel size guard runs the body inline below the threshold") {
  // The guard is a scheduling decision, so what is gated is that it partitions
  // the SAME rows exactly once each either side of it — not which side it
  // chooses. Below the threshold the body must see one whole range on the
  // calling thread; above it, every row still exactly once.
  for (const int threads : kThreadCounts) {
    vt::cpu::Threadpool pool(threads);
    vt::cpu::Threadpool* previous = vt::cpu::Threadpool::SwapForTesting(&pool);

    std::vector<int> visits(64, 0);
    int ranges_small = 0;
    vllm::host_parallel::ForOutputRows(64, 1, [&](int64_t r0, int64_t r1) {
      ++ranges_small;
      for (int64_t r = r0; r < r1; ++r) ++visits[static_cast<size_t>(r)];
    });
    CHECK(ranges_small == 1);  // inline: 64 * 1 << kMinParallelWork

    // Above the threshold: every row exactly once, AND — the part that keeps
    // the bit-identity cases from being a serial path testing itself — the body
    // must actually have run on more than one thread when the pool has more
    // than one. Without this the whole file passes with the helper hard-wired
    // to run inline, which is a green that means nothing.
    std::vector<int> big(64, 0);
    std::mutex seen_mu;
    std::set<std::thread::id> seen;
    vllm::host_parallel::ForOutputRows(64, vllm::host_parallel::kMinParallelWork,
                                       [&](int64_t r0, int64_t r1) {
                                         {
                                           const std::lock_guard<std::mutex> lock(seen_mu);
                                           seen.insert(std::this_thread::get_id());
                                         }
                                         for (int64_t r = r0; r < r1; ++r)
                                           ++big[static_cast<size_t>(r)];
                                       });
    vt::cpu::Threadpool::SwapForTesting(previous);

    CAPTURE(threads);
    const size_t distinct_threads = seen.size();
    CAPTURE(distinct_threads);
    // `ParallelForRows` seeds worker `ith` with chunk `ith` and the grid is
    // 4x-oversubscribed, so for every count here nchunk > nth and each worker
    // takes at least its own chunk. This is deterministic, not a race that
    // usually wins.
    if (threads == 1) {
      CHECK(distinct_threads == 1U);
    } else {
      CHECK(distinct_threads > 1U);
    }
    for (size_t i = 0; i < visits.size(); ++i) {
      CAPTURE(i);
      CHECK(visits[i] == 1);
      CHECK(big[i] == 1);
    }
  }
}

// --- The ACCUMULATOR WIDTH, gated against torch rather than against us -------
//
// The two cases below are the instrument for `VT-CONV1D-F32-ACC` (#1474), and
// they exist because every other case in this file is SELF-REFERENTIAL. Each
// one compares the shipped kernel to a serial reference carried in this file,
// so widening BOTH back to `double` in one edit leaves the whole suite green.
// That is inherent to a bitwise order gate and is not a defect in it — but it
// means the WIDTH needs a gate that does not read the width off ourselves.
//
// THE VALUE ASSERTED IS torch's, MEASURED. 27 taps over a uniform-1.0 input
// with weights `[+1e8, 0.1 x 25, -1e8]`. `1e8` is exactly representable in f32,
// and every partial sum `1e8 + j*0.1` for `j <= 25` sits below half an ulp of
// it (an ulp at 1e8 is 8.0), so an f32 accumulator holds exactly `1e8` until
// the final `-1e8` and lands on EXACTLY zero — in any summation order, which is
// what makes this a width probe and not an order probe. An f64 accumulator
// lands near 2.5, three decimal orders away. Run against torch 2.11.0+cu130:
// `F.conv1d` f32 -> 0.0, `F.conv1d` bf16 -> 0.0, `F.conv1d` f64 -> 2.4999998510,
// `F.conv_transpose1d` f32 -> 0.0, f64 -> 2.4999998510. So `0.0` is upstream's
// own answer for this reduction at the dtype its checkpoints carry, and the
// case gates a MIRRORED property rather than a local convention.
//
// vLLM owns neither op at the parity pin `555967922` — it drops the vocoder it
// would otherwise own (`qwen3_omni_moe_thinker.py:1975`) — and the one
// convolution it does own says the same thing in a comment
// (`csrc/cpu/mamba_kernels.hpp`, "Accumulate in float32 for precision").
//
// Both cases enter through `vllm::vocoder1d::*`, the production entry point the
// four audio consumers call, not through `vt::` directly.
namespace {

// [+1e8, 0.1 x 25, -1e8], in the order the accumulator receives them.
constexpr int64_t kWidthTaps = 27;

std::vector<float> WidthProbeWeights() {
  std::vector<float> w(static_cast<size_t>(kWidthTaps), 0.1F);
  w.front() = 1e8F;
  w.back() = -1e8F;
  return w;
}

}  // namespace

TEST_CASE("vocoder1d Conv1d accumulates in f32, which is what torch does") {
  const int64_t in_channels = 1, in_len = 64, out_channels = 1;
  const std::vector<float> in(static_cast<size_t>(in_channels * in_len), 1.0F);
  const std::vector<float> weight = WidthProbeWeights();

  int64_t out_len = 0;
  const std::vector<float> got =
      vllm::vocoder1d::Conv1d(in, in_channels, in_len, weight, /*bias=*/nullptr, out_channels,
                              kWidthTaps, /*stride=*/1, /*dilation=*/1, /*groups=*/1, &out_len);

  // 64 - 27 + 1 = 38, which spans one WHOLE tile of 32 and a 6-wide remainder,
  // so both the constant-trip fast path and the chunked tail are covered.
  REQUIRE(out_len == 38);
  REQUIRE(got.size() == static_cast<size_t>(out_len));
  for (size_t i = 0; i < got.size(); ++i) {
    CAPTURE(i);
    // EXACTLY zero, not Approx: an f64 accumulator lands on ~2.5 here and any
    // tolerance wide enough to be written by hand would admit it.
    CHECK(got[i] == 0.0F);
  }
}

TEST_CASE("vocoder1d ConvTranspose1d accumulates in f32, which is what torch does") {
  // The transposed op walks its taps in the OPPOSITE sense: destination cell
  // `p` receives `x[t] * w[p - t]` for t ascending, so the tap index descends.
  // The probe weights are therefore reversed, which puts `+1e8` first and
  // `-1e8` last in the sequence the accumulator actually sees.
  const int64_t in_channels = 1, in_len = kWidthTaps, out_channels = 1;
  const std::vector<float> in(static_cast<size_t>(in_channels * in_len), 1.0F);
  std::vector<float> weight = WidthProbeWeights();
  std::reverse(weight.begin(), weight.end());

  int64_t out_len = 0;
  const std::vector<float> got = vllm::vocoder1d::ConvTranspose1d(
      in, in_channels, in_len, weight, /*bias=*/nullptr, out_channels, kWidthTaps, /*stride=*/1,
      /*padding=*/0, /*groups=*/1, &out_len);

  REQUIRE(out_len == (in_len - 1) + kWidthTaps);
  REQUIRE(got.size() == static_cast<size_t>(out_len));
  // Cell 26 is the only one every one of the 27 taps reaches; the cells around
  // it receive a prefix or a suffix of the sequence and are not the probe.
  CHECK(got[static_cast<size_t>(kWidthTaps - 1)] == 0.0F);
}
