// Byte-exactness gate for `vt::Conv1d` and `vt::ConvTranspose1d` — the BigVGAN
// / DAC vocoder convolutions (#672, .agents/specs/minimax-music3.md §11.4;
// kernels src/vt/cpu/cpu_conv1d_general.cpp and
// src/vt/cuda/cuda_conv1d_general.cu).
//
// THREE CLAIMS, THREE INSTRUMENTS. Each is stated here because a green that
// cannot say which claim it covers is the failure mode this file exists inside.
//
// (1) THE CPU PROVIDER WALKS THE DECLARED ORDER. The oracle is `SerialConv1d`
//     / `SerialConvTranspose1d` below. They were VERBATIM copies of
//     `vllm::vocoder1d::Conv1d` / `ConvTranspose1d` as they stood at 8fa405bb7,
//     which let this clause read "IS THE PRE-OP HOST LOOP". #1474 narrowed
//     their accumulators from `double` to `float` in lockstep with the kernels,
//     so they no longer transcribe that loop's arithmetic and this clause is
//     weaker than it was: what it asserts is that the ORDER did not move, not
//     that the values match a historical implementation. Comparing the op
//     against itself at two thread counts would still prove only determinism —
//     it would pass just as happily if the move into the seam had reassociated
//     every sum — so a separate scalar transcription is still the only oracle
//     that can see an order change, and it is carried here rather than
//     referenced. The WIDTH is gated separately and NOT against ourselves:
//     `tests/vllm/models/test_host_parallel.cpp`, `accumulates in f32, which is
//     what torch does`, asserts torch's own measured answer for a reduction
//     engineered to separate the two widths.
//     `tests/vllm/models/test_host_parallel.cpp` makes the same comparison
//     through the four models' own entry point; this file covers the shapes
//     that entry point cannot express — padding != 0, dilation on the transposed
//     op, output_padding, and batch > 1.
//
// (2) THE CUDA PROVIDER IS BYTE-IDENTICAL TO THE CPU ONE. Not within a
//     tolerance. Both walk one f32 accumulator per output element in the same
//     order; the host is compiled `-ffp-contract=off` (CMakeLists.txt:40-56) and
//     the device kernel pins itself with `__fmul_rn`/`__fadd_rn`, so every
//     operation on both arms is an IEEE single multiply or add with
//     round-to-nearest-even. NOT RUN at that width: #1474 narrowed both arms on
//     a box with no CUDA toolkit and no lease, so these cases have skipped ever
//     since — .agents/specs/vt-conv1d-f32-accumulator.md §7. `memcmp` is therefore the right instrument and a
//     tolerance would be the wrong one: a transposed weight axis, a dropped
//     zero-skip or a reassociated sweep all land well inside any epsilon anyone
//     would write. Skips cleanly with a LOUD message when no GPU is present.
//
// (3) EQUALITY IS TESTED WHERE IT CAN ACTUALLY FAIL. An f64 accumulator stored
//     through an f32 could not show a ~2^-53 relative change, so an order defect
//     on well-scaled data was INVISIBLE — measured, not assumed: see the
//     mutation record in tests/vllm/models/test_host_parallel.cpp. #1474
//     narrowed the accumulator, which can only make well-scaled data more
//     sensitive; the engineered cases stay, because a gate is not retired on the
//     argument that something else now covers it. Every claim above is
//     therefore also exercised on engineered catastrophic cancellation, taps of
//     +2^40 and -2^40 against small remainders.
//
// The shape formulas are additionally checked against torch's own arithmetic
// written out longhand, because `Conv1dOutLength` is the single definition both
// the op's validation and its callers use — an instrument agreeing with itself.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

using vt::Conv1dArgs;
using vt::ConvTranspose1dArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// Deterministic, reproducible, and spread across ~6 decades of magnitude so a
// reassociated sum loses different bits than the sequential one rather than
// cancelling into agreement. A plain LCG, identical on every box.
std::vector<float> Spread(size_t n, uint32_t seed, bool with_zeros = false) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525U + 1013904223U;
    const double mantissa = static_cast<double>(s >> 8) / 16777216.0 - 0.5;
    const int exponent = static_cast<int>((s >> 4) & 0xFU) - 8;
    v[i] = static_cast<float>(mantissa * std::pow(2.0, exponent));
    // ConvTranspose1d skips inputs equal to zero; the skip must be on the path.
    if (with_zeros && ((s >> 20) & 0x7U) == 0U) v[i] = 0.0F;
  }
  return v;
}

Tensor View(const float* data, std::initializer_list<int64_t> shape, Device dev) {
  return Tensor::Contiguous(const_cast<float*>(data), DType::kF32, dev, shape);
}

void RequireBitIdentical(const std::vector<float>& got, const std::vector<float>& want,
                         const std::string& what) {
  REQUIRE_MESSAGE(got.size() == want.size(),
                  what << ": size " << got.size() << " vs " << want.size());
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

// --- The oracles -----------------------------------------------------------
// Written as torch documents `conv1d` / `conv_transpose1d`, with the ACCUMULATOR
// WIDTH and VISIT ORDER of the vocoder1d host loop at 8fa405bb7 — which is what
// the op contract pins (include/vt/ops.h at vt::Conv1d). The `padding`,
// `output_padding` and transposed-`dilation` arms are the torch generalisation
// of loops that only ever ran with them at their defaults; they are here because
// the op accepts them and an accepted parameter that nothing checks is a
// parameter that is wrong.

std::vector<float> SerialConv1d(const std::vector<float>& in, int64_t batch, int64_t in_channels,
                                int64_t in_len, const std::vector<float>& weight,
                                const std::vector<float>* bias, int64_t out_channels,
                                int64_t kernel, const Conv1dArgs& a, int64_t length,
                                bool reverse_ic = false, bool reverse_k = false) {
  const int64_t in_per_group = in_channels / a.groups;
  const int64_t out_per_group = out_channels / a.groups;
  std::vector<float> out(static_cast<size_t>(batch * out_channels * length), 0.0F);
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t oc = 0; oc < out_channels; ++oc) {
      const int64_t g = oc / out_per_group;
      for (int64_t t = 0; t < length; ++t) {
        float acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0F;
        for (int64_t step = 0; step < in_per_group; ++step) {
          const int64_t ic = reverse_ic ? in_per_group - 1 - step : step;
          const int64_t src_c = g * in_per_group + ic;
          for (int64_t tap = 0; tap < kernel; ++tap) {
            const int64_t k = reverse_k ? kernel - 1 - tap : tap;
            const int64_t pos = t * a.stride - a.padding + k * a.dilation;
            if (pos < 0 || pos >= in_len) continue;
            acc += in[static_cast<size_t>((n * in_channels + src_c) * in_len + pos)] *
                   weight[static_cast<size_t>((oc * in_per_group + ic) * kernel + k)];
          }
        }
        out[static_cast<size_t>((n * out_channels + oc) * length + t)] = acc;
      }
    }
  }
  return out;
}

// `reverse_ic` and `reverse_k` walk the input channels / the taps DESCENDING
// instead of ascending. Neither is a mode the op has; each is the ORDER MUTATION
// a cancellation case uses to prove it has teeth before any agreement is
// believed. Same multiset of products into every destination cell, different
// sequence of additions.
//
// THERE ARE TWO AXES AND EACH NEEDS ITS OWN CASE, which is a finding rather than
// a symmetry (#1334, review 2026-08-19). The sweep the CPU kernel hoists out of
// its accumulators is `(ic ascending, k ascending)`, and reversing EITHER is a
// genuine reassociation. Data engineered to cancel across `ic` does not bite
// along `k`: with the kernel's `k` loop reversed and nothing else changed,
// `test_ops_conv1d_general` 9/375, `test_host_parallel` 8/877,
// `test_vocoder1d` 10/58 and `test_bigvgan` 6/65 all reported `SUCCESS!` —
// measured on this tree, binary sha256 `c4bb0e76...` against the baseline
// `760061c5...`, so the mutation was in the binary that ran. A case per axis is
// therefore what the guarantee costs.
std::vector<float> SerialConvTranspose1d(const std::vector<float>& in, int64_t batch,
                                         int64_t in_channels, int64_t in_len,
                                         const std::vector<float>& weight,
                                         const std::vector<float>* bias, int64_t out_channels,
                                         int64_t kernel, const ConvTranspose1dArgs& a,
                                         int64_t length, bool reverse_ic = false) {
  const int64_t in_per_group = in_channels / a.groups;
  const int64_t out_per_group = out_channels / a.groups;
  const int64_t full = (in_len - 1) * a.stride + a.dilation * (kernel - 1) + 1;
  std::vector<float> out(static_cast<size_t>(batch * out_channels * length));
  std::vector<float> acc(static_cast<size_t>(out_channels * full));
  for (int64_t n = 0; n < batch; ++n) {
    std::fill(acc.begin(), acc.end(), 0.0F);
    for (int64_t step = 0; step < in_channels; ++step) {
      const int64_t ic = reverse_ic ? in_channels - 1 - step : step;
      const int64_t g = ic / in_per_group;
      for (int64_t t = 0; t < in_len; ++t) {
        const float value = in[static_cast<size_t>((n * in_channels + ic) * in_len + t)];
        if (value == 0.0F) continue;
        for (int64_t oc = 0; oc < out_per_group; ++oc) {
          const int64_t dst_c = g * out_per_group + oc;
          for (int64_t k = 0; k < kernel; ++k) {
            acc[static_cast<size_t>(dst_c * full + t * a.stride + k * a.dilation)] +=
                value * weight[static_cast<size_t>((ic * out_per_group + oc) * kernel + k)];
          }
        }
      }
    }
    for (int64_t c = 0; c < out_channels; ++c) {
      for (int64_t t = 0; t < length; ++t) {
        const int64_t p = t + a.padding;
        float value = p < full ? acc[static_cast<size_t>(c * full + p)] : 0.0F;
        if (bias != nullptr) value += (*bias)[static_cast<size_t>(c)];
        out[static_cast<size_t>((n * out_channels + c) * length + t)] = value;
      }
    }
  }
  return out;
}

// --- The case table --------------------------------------------------------
// Every axis the two providers could plausibly get wrong, plus the geometries
// the four consumers actually run.

struct FwdCase {
  const char* name;
  int64_t batch, in_channels, in_len, out_channels, kernel;
  Conv1dArgs args;
  bool with_bias;
};

const FwdCase kFwdCases[] = {
    // The BigVGAN conv_pre/conv_post shape (K=7, already-padded input), which is
    // what MiniMax-Music3, LTX-2.5 and MiniMax-H3 all run.
    {"bigvgan-k7", 1, 64, 256, 64, 7, {1, 0, 1, 1}, true},
    // The alias-free low-pass DOWNSAMPLE: depthwise, strided by the ratio.
    {"lowpass-depthwise", 1, 96, 300, 96, 12, {2, 0, 1, 96}, false},
    // Dilated residual unit (dilations 1/3/9 in the DAC decoder).
    {"dilated-9", 1, 48, 512, 48, 7, {1, 0, 9, 1}, true},
    // 1x1 projection (dec_in_proj, attn_proj).
    {"pointwise", 1, 128, 64, 256, 1, {1, 0, 1, 1}, true},
    // Grouped, and a group count that does not divide the channel count evenly
    // per output — catches a wrong `g` derivation.
    {"grouped-4", 1, 64, 128, 96, 3, {1, 0, 1, 4}, false},
    // Torch `padding=` — the arm no vocoder1d caller reaches (they pad through
    // Pad1d, which can replicate) and therefore the arm nothing else checks.
    {"padded-same", 1, 32, 64, 32, 5, {1, 2, 1, 1}, true},
    // padding + stride + dilation at once, where the window falls off both ends.
    {"padded-strided-dilated", 1, 16, 41, 24, 4, {3, 5, 2, 1}, true},
    // batch > 1: the vocoder always passes N=1, so the batch stride is otherwise
    // an untested index.
    {"batch3", 3, 24, 96, 24, 3, {1, 1, 1, 1}, false},
    // Kernel longer than the input once padded in.
    {"k-gt-l", 1, 8, 5, 8, 9, {1, 4, 1, 1}, true},
    // --- THE TILE GEOMETRY (#1334) -----------------------------------------
    // The CPU forward kernel walks output positions in tiles of 32, so the tile
    // boundary is an edge the cases above only meet by coincidence. These pick
    // lengths AGAINST it: exactly one tile, one tile plus a 1-wide tail, a
    // length below one tile, and a length whose left clamp falls in the first
    // tile while its right clamp falls in a partial LAST tile — the arrangement
    // a per-position skip cannot get wrong and a per-tile clamp can.
    {"tile-exact-32", 1, 24, 38, 24, 7, {1, 0, 1, 1}, true},
    {"tile-plus-one", 1, 24, 39, 24, 7, {1, 0, 1, 1}, false},
    {"sub-tile-padded", 1, 16, 20, 16, 7, {1, 3, 1, 1}, true},
    {"multi-tile-padded", 1, 12, 100, 12, 9, {1, 4, 1, 1}, true},
    {"tile-boundary-dilated", 1, 16, 96, 16, 5, {1, 6, 4, 1}, false},
    // The strided arm keeps the shipped gather, and its clamp is the one with a
    // ceiling division in it: `lo = ceil(-base/stride)`, `hi = ceil(room/stride)`.
    {"tile-strided-padded", 1, 16, 200, 16, 5, {3, 4, 2, 1}, true},
};

struct BwdCase {
  const char* name;
  int64_t batch, in_channels, in_len, out_channels, kernel;
  ConvTranspose1dArgs args;
  bool with_bias, with_zeros;
};

const BwdCase kBwdCases[] = {
    // MiniMax-Music3's first upsample stage in miniature: K = 2*stride,
    // padding = (K-stride)/2, the geometry minimax_music3_loader.cpp:228 records.
    {"music3-up0", 1, 96, 64, 48, 16, {8, 4, 0, 1, 1}, true, false},
    // The alias-free UPSAMPLE: depthwise, stride == ratio, no crop.
    {"upsample-depthwise", 1, 128, 200, 128, 12, {2, 0, 0, 1, 128}, false, true},
    {"groups=4", 1, 64, 48, 96, 6, {3, 1, 0, 1, 4}, true, false},
    // Kernel shorter than the stride, so the scatter leaves GAPS the gather
    // form has to reproduce as exact zeros.
    {"kernel<stride", 1, 48, 64, 64, 3, {5, 0, 0, 1, 1}, false, false},
    // Exact zeros in the input, so the `value == 0.0` skip is on the path.
    {"zeros+no-bias", 1, 80, 80, 80, 7, {2, 2, 0, 1, 1}, false, true},
    // output_padding — torch's odd-length escape hatch, and an arm no consumer
    // reaches today.
    {"output-padding", 1, 32, 33, 32, 4, {3, 1, 2, 1, 1}, true, false},
    // dilation on the transposed op: another accepted-but-unreached parameter.
    {"dilated", 1, 24, 40, 24, 5, {2, 3, 0, 3, 1}, true, true},
    {"batch3", 3, 16, 32, 24, 6, {2, 1, 1, 1, 1}, true, false},
};

// Runs the op on `dev`, staging through device memory when that is not the CPU.
std::vector<float> RunFwd(Device dev, const std::vector<float>& in,
                          std::initializer_list<int64_t> in_shape,
                          const std::vector<float>& weight,
                          std::initializer_list<int64_t> w_shape, const std::vector<float>* bias,
                          std::initializer_list<int64_t> out_shape, const Conv1dArgs& args);
std::vector<float> RunBwd(Device dev, const std::vector<float>& in,
                          std::initializer_list<int64_t> in_shape,
                          const std::vector<float>& weight,
                          std::initializer_list<int64_t> w_shape, const std::vector<float>* bias,
                          std::initializer_list<int64_t> out_shape,
                          const ConvTranspose1dArgs& args);

// One staging helper; the two Run* wrappers differ only in which op they call.
template <typename Launch>
std::vector<float> Stage(Device dev, const std::vector<float>& in,
                         std::initializer_list<int64_t> in_shape, const std::vector<float>& weight,
                         std::initializer_list<int64_t> w_shape, const std::vector<float>* bias,
                         std::initializer_list<int64_t> out_shape, size_t out_n,
                         const Launch& launch) {
  std::vector<float> out(out_n, 0.0F);
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor xt = View(in.data(), in_shape, dev);
    Tensor wt = View(weight.data(), w_shape, dev);
    Tensor ot = View(out.data(), out_shape, dev);
    Tensor bt;
    if (bias != nullptr) bt = View(bias->data(), {static_cast<int64_t>(bias->size())}, dev);
    launch(q, ot, xt, wt, bias != nullptr ? &bt : nullptr);
    return out;
  }
  vt::Backend& backend = vt::GetBackend(dev.type);
  Queue q = backend.CreateQueue();
  void* xd = backend.Alloc(in.size() * sizeof(float));
  void* wd = backend.Alloc(weight.size() * sizeof(float));
  void* od = backend.Alloc(out.size() * sizeof(float));
  void* bd = bias != nullptr ? backend.Alloc(bias->size() * sizeof(float)) : nullptr;
  backend.Copy(q, xd, in.data(), in.size() * sizeof(float));
  backend.Copy(q, wd, weight.data(), weight.size() * sizeof(float));
  if (bd != nullptr) backend.Copy(q, bd, bias->data(), bias->size() * sizeof(float));
  Tensor xt = Tensor::Contiguous(xd, DType::kF32, dev, in_shape);
  Tensor wt = Tensor::Contiguous(wd, DType::kF32, dev, w_shape);
  Tensor ot = Tensor::Contiguous(od, DType::kF32, dev, out_shape);
  Tensor bt;
  if (bd != nullptr) bt = Tensor::Contiguous(bd, DType::kF32, dev, {static_cast<int64_t>(bias->size())});
  launch(q, ot, xt, wt, bd != nullptr ? &bt : nullptr);
  backend.Copy(q, out.data(), od, out.size() * sizeof(float));
  backend.Synchronize(q);
  backend.Free(xd);
  backend.Free(wd);
  backend.Free(od);
  if (bd != nullptr) backend.Free(bd);
  backend.DestroyQueue(q);
  return out;
}

std::vector<float> RunFwd(Device dev, const std::vector<float>& in,
                          std::initializer_list<int64_t> in_shape,
                          const std::vector<float>& weight,
                          std::initializer_list<int64_t> w_shape, const std::vector<float>* bias,
                          std::initializer_list<int64_t> out_shape, const Conv1dArgs& args) {
  size_t n = 1;
  for (const int64_t d : out_shape) n *= static_cast<size_t>(d);
  return Stage(dev, in, in_shape, weight, w_shape, bias, out_shape, n,
               [&](Queue& q, Tensor& o, const Tensor& x, const Tensor& w, const Tensor* b) {
                 vt::Conv1d(q, o, x, w, b, args);
               });
}

std::vector<float> RunBwd(Device dev, const std::vector<float>& in,
                          std::initializer_list<int64_t> in_shape,
                          const std::vector<float>& weight,
                          std::initializer_list<int64_t> w_shape, const std::vector<float>* bias,
                          std::initializer_list<int64_t> out_shape,
                          const ConvTranspose1dArgs& args) {
  size_t n = 1;
  for (const int64_t d : out_shape) n *= static_cast<size_t>(d);
  return Stage(dev, in, in_shape, weight, w_shape, bias, out_shape, n,
               [&](Queue& q, Tensor& o, const Tensor& x, const Tensor& w, const Tensor* b) {
                 vt::ConvTranspose1d(q, o, x, w, b, args);
               });
}

}  // namespace

TEST_CASE("vt::Conv1d output length matches torch's formula longhand") {
  for (const FwdCase& c : kFwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t want = (c.in_len + 2 * c.args.padding - c.args.dilation * (c.kernel - 1) - 1) /
                             c.args.stride +
                         1;
    CHECK(vt::Conv1dOutLength(c.in_len, c.kernel, c.args) == want);
  }
}

TEST_CASE("vt::ConvTranspose1d output length matches torch's formula longhand") {
  for (const BwdCase& c : kBwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t want = (c.in_len - 1) * c.args.stride - 2 * c.args.padding +
                         c.args.dilation * (c.kernel - 1) + 1 + c.args.output_padding;
    CHECK(vt::ConvTranspose1dOutLength(c.in_len, c.kernel, c.args) == want);
  }
}

TEST_CASE("vt::Conv1d CPU provider is byte-identical to the pre-op host loop") {
  for (const FwdCase& c : kFwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t lout = vt::Conv1dOutLength(c.in_len, c.kernel, c.args);
    REQUIRE(lout > 0);
    const int64_t cin_g = c.in_channels / c.args.groups;
    const std::vector<float> in =
        Spread(static_cast<size_t>(c.batch * c.in_channels * c.in_len), 0xD1CEu + c.kernel);
    const std::vector<float> w =
        Spread(static_cast<size_t>(c.out_channels * cin_g * c.kernel), 0xF00Du + c.kernel);
    const std::vector<float> bias = Spread(static_cast<size_t>(c.out_channels), 0x4E2Bu);
    const std::vector<float>* bp = c.with_bias ? &bias : nullptr;

    const std::vector<float> want = SerialConv1d(in, c.batch, c.in_channels, c.in_len, w, bp,
                                                 c.out_channels, c.kernel, c.args, lout);
    const std::vector<float> got =
        RunFwd(Cpu(), in, {c.batch, c.in_channels, c.in_len}, w, {c.out_channels, cin_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    RequireBitIdentical(got, want, std::string("Conv1d cpu ") + c.name);
  }
}

TEST_CASE("vt::ConvTranspose1d CPU provider is byte-identical to the pre-op host loop") {
  for (const BwdCase& c : kBwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t lout = vt::ConvTranspose1dOutLength(c.in_len, c.kernel, c.args);
    REQUIRE(lout > 0);
    const int64_t cout_g = c.out_channels / c.args.groups;
    const std::vector<float> in = Spread(
        static_cast<size_t>(c.batch * c.in_channels * c.in_len), 0xC0FFu + c.kernel, c.with_zeros);
    const std::vector<float> w =
        Spread(static_cast<size_t>(c.in_channels * cout_g * c.kernel), 0xBEEFu + c.kernel);
    const std::vector<float> bias = Spread(static_cast<size_t>(c.out_channels), 0x0B1Au);
    const std::vector<float>* bp = c.with_bias ? &bias : nullptr;

    const std::vector<float> want = SerialConvTranspose1d(
        in, c.batch, c.in_channels, c.in_len, w, bp, c.out_channels, c.kernel, c.args, lout);
    const std::vector<float> got =
        RunBwd(Cpu(), in, {c.batch, c.in_channels, c.in_len}, w, {c.in_channels, cout_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    RequireBitIdentical(got, want, std::string("ConvTranspose1d cpu ") + c.name);
  }
}

TEST_CASE("vt::ConvTranspose1d reproduces the host loop's ZERO-SKIP exactly") {
  // Not a flourish. The host scatter skips an input that compares equal to 0.0
  // BEFORE touching the destination, and a gather form that instead adds
  // `0.0 * w` produces a DIFFERENT bit pattern for a cell whose running sum is
  // -0.0, because (-0.0) + (+0.0) == +0.0 while -0.0 left alone stays -0.0. An
  // all-zero input drives every output cell down that path at once, so the case
  // is a direct assertion on the sign bit rather than on a magnitude.
  const int64_t cin = 8, lin = 16, cout = 8, kernel = 4;
  ConvTranspose1dArgs args;
  args.stride = 2;
  const int64_t lout = vt::ConvTranspose1dOutLength(lin, kernel, args);
  const std::vector<float> in(static_cast<size_t>(cin * lin), 0.0F);
  // Negative weights so a `0.0 * w` term would be -0.0 rather than +0.0.
  const std::vector<float> w(static_cast<size_t>(cin * cout * kernel), -1.0F);
  const std::vector<float> want =
      SerialConvTranspose1d(in, 1, cin, lin, w, nullptr, cout, kernel, args, lout);
  const std::vector<float> got = RunBwd(Cpu(), in, {1, cin, lin}, w, {cin, cout, kernel}, nullptr,
                                        {1, cout, lout}, args);
  RequireBitIdentical(got, want, "ConvTranspose1d zero-skip");
  // And state what the oracle itself produced, so a future change to the skip
  // cannot quietly agree with a co-mutated oracle.
  for (const float v : want) CHECK(std::signbit(v) == false);
}

TEST_CASE("vt::Conv1d CPU holds its sweep ORDER under cancellation, across tile boundaries") {
  // THE case for #1334 along the `ic` AXIS, and a hole this file had before it.
  // The `k` axis is the case below, and the two are not interchangeable: this
  // one pairs input CHANNELS and its TEETH check reverses `ic`, so a `k` sweep
  // reversed in the kernel walks straight through it. The cancellation case at
  // the bottom compares Conv1d CPU-vs-CUDA only, so on a CPU-only build
  // NOTHING here held the forward sweep's ORDER against the pre-op host loop —
  // every other forward assertion runs on well-scaled data, where an f64
  // accumulator stored through an f32 hides a reduction-order change completely
  // (measured, not assumed: tests/vllm/models/test_host_parallel.cpp).
  //
  // That is exactly the guarantee the tiled kernel turns on. It interleaves the
  // (ic, k) sweep across 32 independent accumulators instead of serialising it
  // into one, which must leave the order each individual cell sees UNCHANGED. On
  // benign data a kernel that got that wrong would still pass everything above.
  //
  // The shape is chosen against the tile: length 197 is six whole tiles plus a
  // 5-wide tail, padding 3 puts the left clamp inside tile 0 and the right clamp
  // inside that tail, and dilation 2 makes the two clamps land at different taps.
  const int64_t cin = 32, lin = 205, cout = 24, kernel = 8;
  const float kBig = 1099511627776.0F;  // 2^40
  std::vector<float> in = Spread(static_cast<size_t>(cin * lin), 0x3C3Cu);
  std::vector<float> w = Spread(static_cast<size_t>(cout * cin * kernel), 0xC3C3u);
  for (int64_t t = 0; t < lin; ++t) {
    in[static_cast<size_t>(0 * lin + t)] = kBig;
    in[static_cast<size_t>(1 * lin + t)] = -kBig;
  }
  // Channels 0 and 1 share a weight row, so the sequential (ic ascending) sweep
  // cancels them at once and keeps the small remainder exactly; any other order
  // carries 2^40 through the accumulator and quantises at ~1.2e-4.
  for (int64_t oc = 0; oc < cout; ++oc) {
    for (int64_t k = 0; k < kernel; ++k) {
      w[static_cast<size_t>((oc * cin + 1) * kernel + k)] =
          w[static_cast<size_t>((oc * cin + 0) * kernel + k)];
    }
  }

  Conv1dArgs args;
  args.stride = 1;
  args.padding = 3;
  args.dilation = 2;
  const int64_t lout = vt::Conv1dOutLength(lin, kernel, args);
  REQUIRE(lout == 197);
  const std::vector<float> want =
      SerialConv1d(in, 1, cin, lin, w, nullptr, cout, kernel, args, lout);
  const std::vector<float> got = RunFwd(Cpu(), in, {1, cin, lin}, w, {cout, cin, kernel}, nullptr,
                                        {1, cout, lout}, args);
  RequireBitIdentical(got, want, "Conv1d cancellation cpu-vs-oracle");

  // TEETH, before the agreement above is believed. Reversing the input-channel
  // sweep is a genuine reassociation — the same multiset of products into every
  // cell, added in the opposite order. If this reads 0 the cancellation has
  // stopped biting and the equality above is vacuous, which is the one state a
  // passing suite cannot otherwise report.
  {
    const std::vector<float> other =
        SerialConv1d(in, 1, cin, lin, w, nullptr, cout, kernel, args, lout, /*reverse_ic=*/true);
    size_t differing = 0;
    for (size_t i = 0; i < other.size(); ++i) {
      if (std::memcmp(&other[i], &want[i], sizeof(float)) != 0) ++differing;
    }
    INFO("order sensitivity: " << differing << " of " << want.size()
                               << " cells change when the ic sweep is reversed");
    CHECK(differing > 0);
  }

  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: Conv1d tiled-cancellation CPU-vs-CUDA NOT exercised\n");
    return;
  }
  const std::vector<float> cuda = RunFwd(Device{DeviceType::kCUDA, 0}, in, {1, cin, lin}, w,
                                         {cout, cin, kernel}, nullptr, {1, cout, lout}, args);
  RequireBitIdentical(cuda, got, "Conv1d cancellation cuda-vs-cpu across tiles");
}

TEST_CASE("vt::Conv1d CPU holds its TAP order under cancellation, across tile boundaries") {
  // The SECOND axis, and the reason it is a separate case (#1334, review
  // 2026-08-19). The kernel hoists `(ic ascending, k ascending)` out of 32
  // accumulators, so reversing EITHER loop reassociates every cell — but the
  // case above is engineered along `ic` alone, and its own teeth check reverses
  // `ic` alone. Reversing only the kernel's `k` loop left the whole gate green:
  // 9/375, 8/877, 10/58, 6/65, every one `SUCCESS!`. A guarantee gated on one of
  // its two axes is gated on neither, so this case pairs TAPS instead.
  //
  // The construction, and why it bites. Input channel 0 is CONSTANT at 1.0, so
  // the taps of that channel all read the same value whatever `dilation` does
  // with their positions. Its first two taps are +2^40 and -2^40 and its
  // remaining six are O(1), and every other channel is O(1) throughout. Sweeping
  // `k` ascending cancels 2^40 against 2^40 at once and then adds the O(1)
  // remainder into an accumulator of magnitude ~0, exactly. Sweeping it
  // descending accumulates that remainder FIRST and then puts +-2^40 through it,
  // which quantises it at 2^-12 ~ 2.4e-4 — five orders of magnitude above the
  // f32 store's ~2e-6 ULP at this result's scale, so the difference cannot hide
  // in the round-trip through `float` the way a well-scaled reordering does.
  //
  // The shape is the one the `ic` case uses, for the same reason: `lout == 197`
  // is six whole 32-wide tiles plus a 5-wide tail, `padding == 3` puts the left
  // clamp inside tile 0, and `dilation == 2` lands the two big taps at different
  // positions. It also drives the clamp THROUGH the cancelling pair: at t=0..1
  // both big taps are out of range, at t=2 only the negative one is in range,
  // and from t=3 on both are — so a clamp that keeps the wrong tap produces 2^40
  // rather than a rounding difference.
  const int64_t cin = 4, lin = 205, cout = 8, kernel = 8;
  const float kBig = 1099511627776.0F;  // 2^40
  std::vector<float> in(static_cast<size_t>(cin * lin));
  std::vector<float> w(static_cast<size_t>(cout * cin * kernel));
  // O(1) and deterministic: the same LCG `Spread` uses, mapped into [0.5, 1.5)
  // instead of across six decades. The magnitude is the point — a small term
  // that is itself huge is not a remainder.
  uint32_t s = 0x5EEDu;
  const auto next = [&s]() {
    s = s * 1664525U + 1013904223U;
    return 0.5F + static_cast<float>(s >> 8) / 33554432.0F;  // [0.5, 1.5)
  };
  for (float& v : in) v = next();
  for (float& v : w) v = next();
  for (int64_t t = 0; t < lin; ++t) in[static_cast<size_t>(0 * lin + t)] = 1.0F;
  for (int64_t oc = 0; oc < cout; ++oc) {
    w[static_cast<size_t>((oc * cin + 0) * kernel + 0)] = kBig;
    w[static_cast<size_t>((oc * cin + 0) * kernel + 1)] = -kBig;
  }

  Conv1dArgs args;
  args.stride = 1;
  args.padding = 3;
  args.dilation = 2;
  const int64_t lout = vt::Conv1dOutLength(lin, kernel, args);
  REQUIRE(lout == 197);
  const std::vector<float> want =
      SerialConv1d(in, 1, cin, lin, w, nullptr, cout, kernel, args, lout);
  const std::vector<float> got = RunFwd(Cpu(), in, {1, cin, lin}, w, {cout, cin, kernel}, nullptr,
                                        {1, cout, lout}, args);
  RequireBitIdentical(got, want, "Conv1d tap-order cancellation cpu-vs-oracle");

  // TEETH, on THIS axis, before the agreement above is believed. Reversing the
  // tap sweep is the mutation this case exists to detect; if it changes nothing
  // the equality above is vacuous.
  {
    const std::vector<float> other =
        SerialConv1d(in, 1, cin, lin, w, nullptr, cout, kernel, args, lout,
                     /*reverse_ic=*/false, /*reverse_k=*/true);
    size_t differing = 0;
    for (size_t i = 0; i < other.size(); ++i) {
      if (std::memcmp(&other[i], &want[i], sizeof(float)) != 0) ++differing;
    }
    INFO("tap order sensitivity: " << differing << " of " << want.size()
                                   << " cells change when the k sweep is reversed");
    CHECK(differing > 0);
  }

  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: Conv1d tap-order cancellation CPU-vs-CUDA NOT exercised\n");
    return;
  }
  const std::vector<float> cuda = RunFwd(Device{DeviceType::kCUDA, 0}, in, {1, cin, lin}, w,
                                         {cout, cin, kernel}, nullptr, {1, cout, lout}, args);
  RequireBitIdentical(cuda, got, "Conv1d tap-order cancellation cuda-vs-cpu across tiles");
}

TEST_CASE("vt conv1d ops REFUSE a narrow dtype by name rather than widening it") {
  // f16/bf16 arms are not implemented. AGENTS.md requires an unimplemented arm
  // to be refused with a message naming the missing piece, not to be silently
  // promoted — a silent promotion here would put a shipped audio model on an
  // accumulator nothing gated.
  std::vector<uint16_t> half(64, 0);
  std::vector<float> f32(64, 0.0F);
  Queue q{Cpu(), nullptr};
  Tensor x = Tensor::Contiguous(half.data(), DType::kBF16, Cpu(), {1, 2, 8});
  Tensor w = Tensor::Contiguous(f32.data(), DType::kF32, Cpu(), {2, 2, 3});
  Tensor o = Tensor::Contiguous(f32.data(), DType::kF32, Cpu(), {1, 2, 6});
  Conv1dArgs fwd;
  CHECK_THROWS_WITH_AS(vt::Conv1d(q, o, x, w, nullptr, fwd),
                       doctest::Contains("must be f32"), std::runtime_error);
  ConvTranspose1dArgs bwd;
  Tensor ot = Tensor::Contiguous(f32.data(), DType::kF32, Cpu(), {1, 2, 10});
  CHECK_THROWS_WITH_AS(vt::ConvTranspose1d(q, ot, x, w, nullptr, bwd),
                       doctest::Contains("must be f32"), std::runtime_error);
}

TEST_CASE("vt conv1d ops CUDA provider is byte-identical to the CPU provider") {
  if (!HasCuda()) {
    // LOUD, because a silent skip on a CPU box is how a device arm goes
    // un-gated for a release.
    std::printf("[SKIP] no CUDA backend: vt::Conv1d/ConvTranspose1d device arm NOT exercised\n");
    return;
  }
  const Device gpu{DeviceType::kCUDA, 0};

  for (const FwdCase& c : kFwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t lout = vt::Conv1dOutLength(c.in_len, c.kernel, c.args);
    const int64_t cin_g = c.in_channels / c.args.groups;
    const std::vector<float> in =
        Spread(static_cast<size_t>(c.batch * c.in_channels * c.in_len), 0xD1CEu + c.kernel);
    const std::vector<float> w =
        Spread(static_cast<size_t>(c.out_channels * cin_g * c.kernel), 0xF00Du + c.kernel);
    const std::vector<float> bias = Spread(static_cast<size_t>(c.out_channels), 0x4E2Bu);
    const std::vector<float>* bp = c.with_bias ? &bias : nullptr;
    const std::vector<float> want =
        RunFwd(Cpu(), in, {c.batch, c.in_channels, c.in_len}, w, {c.out_channels, cin_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    const std::vector<float> got =
        RunFwd(gpu, in, {c.batch, c.in_channels, c.in_len}, w, {c.out_channels, cin_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    RequireBitIdentical(got, want, std::string("Conv1d cuda-vs-cpu ") + c.name);
  }

  for (const BwdCase& c : kBwdCases) {
    CAPTURE(std::string(c.name));
    const int64_t lout = vt::ConvTranspose1dOutLength(c.in_len, c.kernel, c.args);
    const int64_t cout_g = c.out_channels / c.args.groups;
    const std::vector<float> in = Spread(
        static_cast<size_t>(c.batch * c.in_channels * c.in_len), 0xC0FFu + c.kernel, c.with_zeros);
    const std::vector<float> w =
        Spread(static_cast<size_t>(c.in_channels * cout_g * c.kernel), 0xBEEFu + c.kernel);
    const std::vector<float> bias = Spread(static_cast<size_t>(c.out_channels), 0x0B1Au);
    const std::vector<float>* bp = c.with_bias ? &bias : nullptr;
    const std::vector<float> want =
        RunBwd(Cpu(), in, {c.batch, c.in_channels, c.in_len}, w, {c.in_channels, cout_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    const std::vector<float> got =
        RunBwd(gpu, in, {c.batch, c.in_channels, c.in_len}, w, {c.in_channels, cout_g, c.kernel},
               bp, {c.batch, c.out_channels, lout}, c.args);
    RequireBitIdentical(got, want, std::string("ConvTranspose1d cuda-vs-cpu ") + c.name);
  }
}

TEST_CASE("vt conv1d ops agree CPU-vs-CUDA under CATASTROPHIC CANCELLATION") {
  // THE case that can actually fail. Everything above runs on well-scaled data,
  // where an f64 accumulator narrowed to f32 hides a reduction-order change
  // completely — so a green there is compatible with the CUDA gather having
  // reassociated the sweep. Here input channels 0 and 1 carry +2^40 and -2^40
  // through a shared weight row, so the sequential order cancels them
  // immediately and keeps the small remainder exactly, while any other order
  // carries 2^40 through it and quantises at ~1.2e-4.
  const int64_t cin = 32, lin = 96, cout = 32, kernel = 8;
  const float kBig = 1099511627776.0F;  // 2^40
  std::vector<float> in = Spread(static_cast<size_t>(cin * lin), 0x5A5Au);
  std::vector<float> w = Spread(static_cast<size_t>(cin * cout * kernel), 0xA5A5u);
  for (int64_t t = 0; t < lin; ++t) {
    in[static_cast<size_t>(0 * lin + t)] = kBig;
    in[static_cast<size_t>(1 * lin + t)] = -kBig;
  }
  for (int64_t oc = 0; oc < cout; ++oc) {
    for (int64_t k = 0; k < kernel; ++k) {
      w[static_cast<size_t>((1 * cout + oc) * kernel + k)] =
          w[static_cast<size_t>((0 * cout + oc) * kernel + k)];
    }
  }

  ConvTranspose1dArgs bwd;
  bwd.stride = 4;
  bwd.padding = 2;
  const int64_t lout = vt::ConvTranspose1dOutLength(lin, kernel, bwd);
  const std::vector<float> want_cpu =
      SerialConvTranspose1d(in, 1, cin, lin, w, nullptr, cout, kernel, bwd, lout);
  const std::vector<float> cpu =
      RunBwd(Cpu(), in, {1, cin, lin}, w, {cin, cout, kernel}, nullptr, {1, cout, lout}, bwd);
  RequireBitIdentical(cpu, want_cpu, "ConvTranspose1d cancellation cpu-vs-oracle");

  // PROVE THE CASE HAS TEETH before believing any agreement. Reversing the
  // input-channel sweep is a genuine reassociation — the same multiset of
  // products into every cell, added in the opposite order — and with the
  // cancelling pair at ic 0/1 it must change the answer. If this ever reads
  // zero, the cancellation has stopped biting and every equality in this file is
  // vacuous, which is exactly the state a passing suite otherwise cannot report.
  //
  // (A weaker mutation was tried first and correctly read 0: swapping WHICH of
  // the two channels carries +2^40 leaves the partial sums the same magnitude at
  // the same step, so it is not an order change at all. Recorded so it is not
  // re-derived.)
  {
    const std::vector<float> other = SerialConvTranspose1d(
        in, 1, cin, lin, w, nullptr, cout, kernel, bwd, lout, /*reverse_ic=*/true);
    size_t differing = 0;
    for (size_t i = 0; i < other.size(); ++i) {
      if (std::memcmp(&other[i], &want_cpu[i], sizeof(float)) != 0) ++differing;
    }
    INFO("order sensitivity: " << differing << " of " << want_cpu.size()
                               << " cells change when the ic sweep is reversed");
    CHECK(differing > 0);
  }

  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: cancellation CPU-vs-CUDA agreement NOT exercised\n");
    return;
  }
  const Device gpu{DeviceType::kCUDA, 0};
  const std::vector<float> cuda =
      RunBwd(gpu, in, {1, cin, lin}, w, {cin, cout, kernel}, nullptr, {1, cout, lout}, bwd);
  RequireBitIdentical(cuda, cpu, "ConvTranspose1d cancellation cuda-vs-cpu");

  Conv1dArgs fwd;
  fwd.dilation = 2;
  const int64_t flout = vt::Conv1dOutLength(lin, kernel, fwd);
  const std::vector<float> fcpu =
      RunFwd(Cpu(), in, {1, cin, lin}, w, {cout, cin, kernel}, nullptr, {1, cout, flout}, fwd);
  const std::vector<float> fcuda =
      RunFwd(gpu, in, {1, cin, lin}, w, {cout, cin, kernel}, nullptr, {1, cout, flout}, fwd);
  RequireBitIdentical(fcuda, fcpu, "Conv1d cancellation cuda-vs-cpu");
}
