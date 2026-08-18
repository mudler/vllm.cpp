// Definitions for the shared 1-D BigVGAN vocoder core. See vocoder1d.h.
#include "vllm/model_executor/models/vocoder1d.h"

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace vocoder1d {

namespace {

// TU-private helpers for KaiserSincFilter1d, moved with their only caller.

// Zeroth-order modified Bessel function of the first kind, matching the series
// torch.kaiser_window uses.
double BesselI0(double x) {
  double sum = 1.0, term = 1.0;
  const double half_x_sq = (x / 2.0) * (x / 2.0);
  for (int k = 1; k < 64; ++k) {
    term *= half_x_sq / (static_cast<double>(k) * static_cast<double>(k));
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return sum;
}

double Sinc(double x) {
  if (x == 0.0) return 1.0;
  const double pix = std::numbers::pi_v<double> * x;
  return std::sin(pix) / pix;
}

// torch.kaiser_window(n, periodic=false, beta).
std::vector<double> KaiserWindow(int64_t length, double beta) {
  std::vector<double> window(static_cast<size_t>(length));
  const double denom = BesselI0(beta);
  // periodic=false => the window spans [0, length-1] inclusive.
  const double n_minus_1 = static_cast<double>(length - 1);
  for (int64_t i = 0; i < length; ++i) {
    const double ratio = (2.0 * static_cast<double>(i) - n_minus_1) / n_minus_1;
    window[static_cast<size_t>(i)] = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
  }
  return window;
}

// --- The convolution seam (#672) -------------------------------------------
//
// `Conv1d` and `ConvTranspose1d` below are no longer loops. They are thin
// callers of `vt::Conv1d` / `vt::ConvTranspose1d`, whose CPU providers
// (src/vt/cpu/cpu_conv1d_general.cpp) ARE those loops, moved statement for
// statement. Two things follow, and both are the point of the change:
//
//   * on the CPU device nothing moved. Same accumulator width, same visit
//     order, same zero-skip, same output-channel partition over the same
//     threadpool, and the tensors are views over the caller's `std::vector`
//     rather than copies. `tests/vllm/models/test_host_parallel.cpp` proves it
//     against a verbatim copy of the pre-op loop.
//   * a device provider now EXISTS to route to. Before this, the transposed
//     convolution — 88.5 % of MiniMax-Music3's acoustic-half profile — had no
//     `vt` op of any kind behind it, so there was nothing to offload to and
//     hand-rolling a kernel outside the shared seam is what AGENTS.md forbids.

vt::DeviceType ResolveConvDevice() {
  // OPT-IN, and default CPU. Turning the device arm on by default would move
  // the numerics of FOUR shipped models at once — MiniMax-Music3, MiniMax-H3's
  // audio VAE, LTX-2.5's audio VAE and IndexTTS-2.5 all decode through here,
  // and their goldens were taken on the host loop. The CUDA provider is written
  // to reproduce the host reduction order exactly (see cuda_conv1d_general.cu),
  // but "written to" is not "measured on every consumer's goldens", and a
  // default that silently re-gates four models is not a default this row is
  // entitled to set. `.agents/specs/minimax-music3.md` §13.6 names what is not
  // reached, the row that owns the wiring, and the issue (#672).
  //
  // AND IT NAMES NO DEVICE. The first draft spelled `kCUDA` here and
  // `check-device-leakage.py` refused it — correctly, because this is the
  // device-agnostic shared layer, and its advice is to ask the op/provider
  // table the question instead. Doing that turned out BETTER than the narrower
  // spelling: the knob now accepts any device whose name `vt` knows and that
  // actually carries both providers, so a Metal, Vulkan or ROCm provider becomes
  // reachable by registering it and touching nothing here. The name->enum walk
  // lives in `vt::DeviceTypeFromName` (include/vt/device.h) rather than here,
  // because enumerating the device list is the seam's job and an
  // `static_cast<DeviceType>(i)` in this file is the same leak wearing a
  // different hat.
  static const vt::DeviceType resolved = [] {
    const char* env = std::getenv("VLLM_CPP_VOCODER_DEVICE");
    if (env == nullptr || env[0] == '\0') return vt::DeviceType::kCPU;
    vt::DeviceType device = vt::DeviceType::kCPU;
    VT_CHECK(vt::DeviceTypeFromName(env, &device),
             "vocoder1d: VLLM_CPP_VOCODER_DEVICE names no device vt knows: '" + std::string(env) +
                 "'");
    // Refused BY NAME rather than silently falling back to the host. A fallback
    // would post a plausible set of timings that mean nothing, and an operator
    // who asked for a device would never learn they did not get one.
    VT_CHECK(vt::OpRegistered(vt::OpId::kConv1d, device) &&
                 vt::OpRegistered(vt::OpId::kConvTranspose1d, device),
             "vocoder1d: VLLM_CPP_VOCODER_DEVICE='" + std::string(env) +
                 "' has no registered vt::Conv1d / vt::ConvTranspose1d provider in this build");
    return device;
  }();
  return resolved;
}

vt::Tensor HostView(const float* data, std::initializer_list<int64_t> shape) {
  return vt::Tensor::Contiguous(const_cast<float*>(data), vt::DType::kF32,
                                vt::Device{vt::DeviceType::kCPU, 0}, shape);
}

// Runs `launch` on the resolved device. On CPU the tensors are views over the
// caller's own buffers and nothing is copied; on a device the inputs are staged
// across, the op runs, and the output comes back. `launch(q, out, x, w, bias)`
// receives tensors already resident on `q.device`.
//
// The device arm allocates, uploads, downloads and frees PER CALL, and creates
// a queue per call with it. That is deliberately literal for a first landing:
// `cuda` means cuda, with no size threshold quietly sending small shapes back to
// the host — a threshold would make the gate below report on a state it was not
// given, which is the exact failure this project keeps re-learning. The cost is
// real and is OWED rather than hidden: device-resident weights (they are
// loop-invariant and re-uploaded every call), one persistent queue, and a
// chain that stays on the device between stages instead of round-tripping. See
// `.agents/specs/minimax-music3.md` §13.
template <typename Launch>
void RunConv(const std::vector<float>& in, std::initializer_list<int64_t> in_shape,
             const std::vector<float>& weight, std::initializer_list<int64_t> weight_shape,
             const std::vector<float>* bias, std::vector<float>& out,
             std::initializer_list<int64_t> out_shape, const Launch& launch) {
  const vt::DeviceType type = ResolveConvDevice();
  if (type == vt::DeviceType::kCPU) {
    vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    vt::Tensor xt = HostView(in.data(), in_shape);
    vt::Tensor wt = HostView(weight.data(), weight_shape);
    vt::Tensor ot = HostView(out.data(), out_shape);
    vt::Tensor bt;
    if (bias != nullptr) bt = HostView(bias->data(), {static_cast<int64_t>(bias->size())});
    launch(q, ot, xt, wt, bias != nullptr ? &bt : nullptr);
    return;
  }

  vt::Backend& backend = vt::GetBackend(type);
  vt::Queue q = backend.CreateQueue();
  const vt::Device dev{type, 0};
  void* xd = backend.Alloc(in.size() * sizeof(float));
  void* wd = backend.Alloc(weight.size() * sizeof(float));
  void* od = backend.Alloc(out.size() * sizeof(float));
  void* bd = bias != nullptr ? backend.Alloc(bias->size() * sizeof(float)) : nullptr;
  try {
    backend.Copy(q, xd, in.data(), in.size() * sizeof(float));
    backend.Copy(q, wd, weight.data(), weight.size() * sizeof(float));
    if (bd != nullptr) backend.Copy(q, bd, bias->data(), bias->size() * sizeof(float));
    vt::Tensor xt = vt::Tensor::Contiguous(xd, vt::DType::kF32, dev, in_shape);
    vt::Tensor wt = vt::Tensor::Contiguous(wd, vt::DType::kF32, dev, weight_shape);
    vt::Tensor ot = vt::Tensor::Contiguous(od, vt::DType::kF32, dev, out_shape);
    vt::Tensor bt;
    if (bd != nullptr) {
      bt = vt::Tensor::Contiguous(bd, vt::DType::kF32, dev,
                                  {static_cast<int64_t>(bias->size())});
    }
    launch(q, ot, xt, wt, bd != nullptr ? &bt : nullptr);
    backend.Copy(q, out.data(), od, out.size() * sizeof(float));
    backend.Synchronize(q);
  } catch (...) {
    backend.Free(xd);
    backend.Free(wd);
    backend.Free(od);
    if (bd != nullptr) backend.Free(bd);
    backend.DestroyQueue(q);
    throw;
  }
  backend.Free(xd);
  backend.Free(wd);
  backend.Free(od);
  if (bd != nullptr) backend.Free(bd);
  backend.DestroyQueue(q);
}

}  // namespace

// ---------------------------------------------------------------------------
// The shared 1-D BigVGAN primitives (declared in vocoder1d.h). They were private
// to this translation unit until LTX-2.5's audio VAE — the same BigVGAN lineage —
// needed them and copied them instead; see the header for why one implementation
// gated by two suites beats two implementations each with its own green gate.
// ---------------------------------------------------------------------------

// One 1-D convolution over [C_in, T] with dilation/stride/groups.
// Weight is [C_out, C_in/groups, K]; input is assumed ALREADY padded.
std::vector<float> Conv1d(const std::vector<float>& in, int64_t in_channels,
                                   int64_t in_len, const std::vector<float>& weight,
                                   const std::vector<float>* bias, int64_t out_channels,
                                   int64_t kernel, int64_t stride, int64_t dilation,
                                   int64_t groups, int64_t* out_len) {
  // The input arrives ALREADY padded (callers pad through `Pad1d`, which can
  // also replicate), so the op's `padding` is 0 and its shape arithmetic
  // collapses to the one this function has always used.
  vt::Conv1dArgs args;
  args.stride = stride;
  args.padding = 0;
  args.dilation = dilation;
  args.groups = groups;
  const int64_t length = vt::Conv1dOutLength(in_len, kernel, args);
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv1d output length is empty");
  std::vector<float> out(static_cast<size_t>(out_channels * length), 0.0f);
  RunConv(in, {1, in_channels, in_len}, weight, {out_channels, in_channels / groups, kernel}, bias,
          out, {1, out_channels, length},
          [&](vt::Queue& q, vt::Tensor& o, const vt::Tensor& x, const vt::Tensor& w,
              const vt::Tensor* b) { vt::Conv1d(q, o, x, w, b, args); });
  *out_len = length;
  return out;
}

// torch.nn.functional.conv_transpose1d over [C_in, T].
// Weight is [C_in, C_out/groups, K]; output length = (T-1)*stride - 2*padding + K.
std::vector<float> ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                            int64_t in_len, const std::vector<float>& weight,
                                            const std::vector<float>* bias, int64_t out_channels,
                                            int64_t kernel, int64_t stride, int64_t padding,
                                            int64_t groups, int64_t* out_len) {
  vt::ConvTranspose1dArgs args;
  args.stride = stride;
  args.padding = padding;
  args.output_padding = 0;
  args.dilation = 1;
  args.groups = groups;
  const int64_t length = vt::ConvTranspose1dOutLength(in_len, kernel, args);
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv_transpose1d output length is empty");
  std::vector<float> out(static_cast<size_t>(out_channels * length));
  // torch's ConvTranspose1d parameter is [Cin, Cout/groups, K] — dim 0 is the
  // INPUT channel, the opposite of nn.Conv1d, which is the same trap
  // `MaterializeWeightNorm`'s `dim0` naming exists for (see vocoder1d.h).
  RunConv(in, {1, in_channels, in_len}, weight, {in_channels, out_channels / groups, kernel}, bias,
          out, {1, out_channels, length},
          [&](vt::Queue& q, vt::Tensor& o, const vt::Tensor& x, const vt::Tensor& w,
              const vt::Tensor* b) { vt::ConvTranspose1d(q, o, x, w, b, args); });
  *out_len = length;
  return out;
}

// F.pad along the time axis: mode="replicate", or the zero pad an ordinary
// nn.Conv1d `padding=` argument performs.
std::vector<float> Pad1d(const std::vector<float>& in, int64_t channels, int64_t in_len,
                                  int64_t left, int64_t right, bool replicate, int64_t* out_len) {
  const int64_t length = in_len + left + right;
  std::vector<float> out(static_cast<size_t>(channels * length), 0.0f);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      int64_t src = t - left;
      if (src < 0 || src >= in_len) {
        if (!replicate) continue;  // already zero
        src = std::max<int64_t>(0, std::min<int64_t>(in_len - 1, src));
      }
      out[static_cast<size_t>(c * length + t)] = in[static_cast<size_t>(c * in_len + src)];
    }
  }
  *out_len = length;
  return out;
}

// Snake / SnakeBeta: x + (b + kSnakeEps)^-1 * sin^2(a * x). A null `beta`
// is plain Snake, which reuses ALPHA as the reciprocal scale (LTX-2.5
// vocoder.py:198); a non-null one is SnakeBeta (vocoder.py:221), which is what
// every MiniMax-H3 checkpoint carries. Both are exponentiated when the checkpoint
// stores them in log scale.
void SnakeActivation(std::vector<float>& x, int64_t channels, int64_t length,
                              const std::vector<float>& alpha, const std::vector<float>* beta,
                              bool logscale) {
  for (int64_t c = 0; c < channels; ++c) {
    double a = alpha[static_cast<size_t>(c)];
    double b = beta != nullptr ? (*beta)[static_cast<size_t>(c)] : a;
    if (logscale) {
      a = std::exp(a);
      b = std::exp(b);
    }
    const double inv_beta = 1.0 / (b + kSnakeEps);
    for (int64_t t = 0; t < length; ++t) {
      const double v = x[static_cast<size_t>(c * length + t)];
      const double s = std::sin(a * v);
      x[static_cast<size_t>(c * length + t)] = static_cast<float>(v + inv_beta * s * s);
    }
  }
}

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60). Returns [kernel_size].
std::vector<float> KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size) {
  VT_CHECK(kernel_size > 0, "minimax_h3 audio vae: kernel_size must be positive");
  VT_CHECK(cutoff >= 0.0 && cutoff <= 0.5, "minimax_h3 audio vae: cutoff must be in [0, 0.5]");
  const bool even = (kernel_size % 2) == 0;
  const int64_t half_size = kernel_size / 2;

  const double delta_f = 4.0 * half_width;
  const double a = 2.285 * (static_cast<double>(half_size) - 1.0) *
                       std::numbers::pi_v<double> * delta_f +
                   7.95;
  double beta = 0.0;
  if (a > 50.0) {
    beta = 0.1102 * (a - 8.7);
  } else if (a >= 21.0) {
    beta = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  }
  const std::vector<double> window = KaiserWindow(kernel_size, beta);

  std::vector<double> time(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    time[static_cast<size_t>(i)] = even ? (static_cast<double>(-half_size + i) + 0.5)
                                        : static_cast<double>(i - half_size);
  }

  std::vector<double> filter(static_cast<size_t>(kernel_size), 0.0);
  if (cutoff == 0.0) {
    return std::vector<float>(static_cast<size_t>(kernel_size), 0.0f);
  }
  double sum = 0.0;
  for (int64_t i = 0; i < kernel_size; ++i) {
    filter[static_cast<size_t>(i)] =
        2.0 * cutoff * window[static_cast<size_t>(i)] * Sinc(2.0 * cutoff * time[static_cast<size_t>(i)]);
    sum += filter[static_cast<size_t>(i)];
  }
  // Normalized to sum 1 so a constant input does not leak.
  std::vector<float> out(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(filter[static_cast<size_t>(i)] / sum);
  }
  return out;
}

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0.

void AliasFreeActivation1d::Build() {
  // Up and down use the same cutoff/half_width/kernel, so one window serves both.
  filter = KaiserSincFilter1d(0.5 / static_cast<double>(ratio),
                                       0.6 / static_cast<double>(ratio), kernel_size);
}

std::vector<float> AliasFreeActivation1d::Apply(const std::vector<float>& in,
                                                         int64_t channels, int64_t in_len,
                                                         const std::vector<float>& alpha,
                                                         const std::vector<float>* beta,
                                                         bool logscale, int64_t* out_len) const {
  // --- UpSample1d ---
  const int64_t pad = kernel_size / ratio - 1;
  const int64_t pad_left = pad * ratio + (kernel_size - ratio) / 2;
  const int64_t pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
  int64_t padded_len = 0;
  const std::vector<float> padded =
      Pad1d(in, channels, in_len, pad, pad, /*replicate=*/true, &padded_len);
  // Depthwise transposed conv: filter.expand(C, -1, -1) => weight [C, 1, K].
  std::vector<float> depthwise(static_cast<size_t>(channels * kernel_size));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t k = 0; k < kernel_size; ++k) {
      depthwise[static_cast<size_t>(c * kernel_size + k)] = filter[static_cast<size_t>(k)];
    }
  }
  int64_t up_len = 0;
  std::vector<float> up =
      ConvTranspose1d(padded, channels, padded_len, depthwise, nullptr, channels,
                               kernel_size, ratio, /*padding=*/0, /*groups=*/channels, &up_len);
  for (float& value : up) value *= static_cast<float>(ratio);
  // x[..., pad_left : -pad_right]
  const int64_t trimmed_len = up_len - pad_left - pad_right;
  VT_CHECK(trimmed_len > 0, "minimax_h3 audio vae: upsample trim emptied the signal");
  std::vector<float> trimmed(static_cast<size_t>(channels * trimmed_len));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < trimmed_len; ++t) {
      trimmed[static_cast<size_t>(c * trimmed_len + t)] =
          up[static_cast<size_t>(c * up_len + pad_left + t)];
    }
  }

  // --- Snake / SnakeBeta ---
  SnakeActivation(trimmed, channels, trimmed_len, alpha, beta, logscale);

  // --- DownSample1d (LowPassFilter1d, stride = ratio, replicate padding) ---
  const bool even = (kernel_size % 2) == 0;
  const int64_t lp_left = kernel_size / 2 - (even ? 1 : 0);
  const int64_t lp_right = kernel_size / 2;
  int64_t lp_padded_len = 0;
  const std::vector<float> lp_padded = Pad1d(trimmed, channels, trimmed_len, lp_left,
                                                      lp_right, /*replicate=*/true, &lp_padded_len);
  return Conv1d(lp_padded, channels, lp_padded_len, depthwise, nullptr, channels,
                         kernel_size, /*stride=*/ratio, /*dilation=*/1, /*groups=*/channels,
                         out_len);
}

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0.
// Moved here from `minimax_h3_audio_vae.cpp` when MiniMax-Music3's vocoder
// became its second consumer; see the declaration for why the axis is named
// `dim0` and not `out_channels`.
std::vector<float> MaterializeWeightNorm(const std::vector<float>& g,
                                         const std::vector<float>& v, int64_t dim0) {
  VT_CHECK(dim0 > 0 && v.size() % static_cast<size_t>(dim0) == 0,
           "vocoder1d: weight-norm direction does not divide by dim 0");
  const int64_t per_slice = static_cast<int64_t>(v.size()) / dim0;
  VT_CHECK(static_cast<int64_t>(g.size()) == dim0,
           "vocoder1d: weight-norm magnitude must have one value per dim-0 slice");
  std::vector<float> out(v.size());
  for (int64_t c = 0; c < dim0; ++c) {
    double norm = 0.0;
    for (int64_t i = 0; i < per_slice; ++i) {
      const double value = v[static_cast<size_t>(c * per_slice + i)];
      norm += value * value;
    }
    norm = std::sqrt(norm);
    const double scale = norm > 0.0 ? static_cast<double>(g[static_cast<size_t>(c)]) / norm : 0.0;
    for (int64_t i = 0; i < per_slice; ++i) {
      out[static_cast<size_t>(c * per_slice + i)] =
          static_cast<float>(v[static_cast<size_t>(c * per_slice + i)] * scale);
    }
  }
  return out;
}

}  // namespace vocoder1d
}  // namespace vllm
