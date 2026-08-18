// CPU providers for `vt::Conv1d` and `vt::ConvTranspose1d` — the BigVGAN / DAC
// vocoder convolutions (#672, .agents/specs/minimax-music3.md §11.4).
//
// PROVENANCE, and why this is a MOVE rather than a rewrite. Both kernel bodies
// below are the `vllm::vocoder1d::Conv1d` / `vllm::vocoder1d::ConvTranspose1d`
// host loops as they stood at 8fa405bb7
// (src/vllm/model_executor/models/vocoder1d.cpp:65-161), carried here
// statement for statement. The accumulator width, the visit order, the seeding
// of the bias, the `value == 0.0` skip and the output-channel partition are all
// unchanged — only the buffer type changed, from `std::vector<float>` to a
// `vt::Tensor` view over the same bytes. That is what lets the four models that
// decode through `vocoder1d` (MiniMax-Music3, MiniMax-H3's audio VAE, LTX-2.5's
// audio VAE, IndexTTS-2.5) keep every committed golden they already had.
//
// Upstream semantics: `torch.nn.functional.conv1d` / `conv_transpose1d` as the
// checkpoints instantiate them — minimax_music3_vocoder.py:42,44,89,98
// (`nn.Conv1d`) and :55 (`nn.ConvTranspose1d`); LTX-2.5
// audio_vae/vocoder.py:104-184 for the alias-free resample pair.
//
// WHY f64 AND NOT f32. torch accumulates an f32 conv in f32. This deliberately
// does not, because f64 is what the host reference used and therefore what every
// golden was taken with; see include/vt/ops.h at vt::Conv1d for the full
// argument and the byte cost (none — only the register width differs).
//
// WHY NOT A MODE OF `vt::DepthwiseConv1d`. That op is f32-accumulate and its
// byte-exactness gate (tests/vt/test_ops_conv1d_depthwise.cpp) pins that width.
// Widening it would move the conformer encoders; narrowing these would move four
// audio models. They are siblings, and the depthwise op is untouched — the same
// call the depthwise op itself made against `vt::CausalConv1dFwd`.
//
// SELF-REGISTERING translation unit (the src/vt/cpu/cpu_ops.cpp Registrar
// idiom), like src/vt/cpu/cpu_conv1d_depthwise.cpp.
//
// DETERMINISM CONTRACT (gates: tests/vt/test_ops_conv1d_general.cpp and
// tests/vllm/models/test_host_parallel.cpp). Parallelism partitions OUTPUT
// (batch, channel) rows only. Every output element is produced by exactly one
// worker running the same instruction sequence, in the same order, as the
// single-thread code — no atomic accumulation, no split reduction, no
// reassociation. Bit-identical across thread counts BY CONSTRUCTION.
#include <algorithm>
#include <functional>
#include <vector>

#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// Multiply-accumulates below which a pool kick costs more than the work it
// distributes. Carried over from `vllm::host_parallel::kMinParallelWork`
// (src/vllm/model_executor/models/host_parallel.h), which is where these loops
// were dispatched from before the op existed, so the SCHEDULING behaviour of the
// vocoder is unchanged along with its arithmetic. It only ever moves WHERE a
// body runs: below the threshold the identical body runs inline on the caller.
constexpr int64_t kMinParallelWork = 1 << 16;

void ForOutputRows(int64_t rows, int64_t work_per_row,
                   const std::function<void(int64_t, int64_t)>& body) {
  if (rows <= 0) return;
  if (rows == 1 || rows * work_per_row < kMinParallelWork) {
    body(0, rows);
    return;
  }
  ParallelForRows(CurrentThreadpool(), rows, body);
}

// out[n, oc, t] = bias[oc] + Sum_{ic,k} x[n, g*Cin/g + ic, t*stride - padding +
// k*dilation] * w[oc, ic, k], taps outside [0, Lin) SKIPPED (zero padding).
//
// The accumulator is seeded with the bias and walked in (ic ascending, k
// ascending) order — vocoder1d.cpp:90-100 @ 8fa405bb7, verbatim. Every existing
// caller passes padding == 0 (the vocoder pads explicitly through
// `vocoder1d::Pad1d`, which can also replicate); the skip below is therefore
// unreachable on those shapes and exists for torch parity.
void Conv1dKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                  const Conv1dArgs& args) {
  const int64_t batch = x.shape[0], in_channels = x.shape[1], in_len = x.shape[2];
  const int64_t out_channels = w.shape[0], kernel = w.shape[2];
  const int64_t length = out.shape[2];
  const int64_t in_per_group = in_channels / args.groups;
  const int64_t out_per_group = out_channels / args.groups;
  const int64_t stride = args.stride, pad = args.padding, dilation = args.dilation;
  const float* xp = x.Ptr<float>();
  const float* wp = w.Ptr<float>();
  const float* bp = bias != nullptr ? bias->Ptr<float>() : nullptr;
  float* op = out.Ptr<float>();

  const int64_t rows = batch * out_channels;
  ForOutputRows(rows, length * in_per_group * kernel, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / out_channels;
      const int64_t oc = r - n * out_channels;
      const int64_t g = oc / out_per_group;
      const float* xn = xp + n * in_channels * in_len;
      float* on = op + (n * out_channels + oc) * length;
      for (int64_t t = 0; t < length; ++t) {
        double acc = bp != nullptr ? bp[oc] : 0.0;
        for (int64_t ic = 0; ic < in_per_group; ++ic) {
          const int64_t src_c = g * in_per_group + ic;
          for (int64_t k = 0; k < kernel; ++k) {
            const int64_t pos = t * stride - pad + k * dilation;
            if (pos < 0 || pos >= in_len) continue;
            acc += static_cast<double>(xn[src_c * in_len + pos]) *
                   static_cast<double>(wp[(oc * in_per_group + ic) * kernel + k]);
          }
        }
        on[t] = static_cast<float>(acc);
      }
    }
  });
}

// torch.nn.functional.conv_transpose1d. Weight is [Cin, Cout/groups, K].
//
// The SCATTER form, verbatim from vocoder1d.cpp:136-158 @ 8fa405bb7: one
// destination channel at a time, a scratch f64 line of `full` cells, inputs
// visited (ic ascending, t ascending), the `value == 0.0` skip intact, the bias
// added LAST on the way out. Both the skip and the ordering are load-bearing —
// see include/vt/ops.h at vt::ConvTranspose1d for why the skip decides the sign
// of a zero cell.
void ConvTranspose1dKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, const ConvTranspose1dArgs& args) {
  const int64_t batch = x.shape[0], in_channels = x.shape[1], in_len = x.shape[2];
  const int64_t out_per_group = w.shape[1], kernel = w.shape[2];
  const int64_t out_channels = out.shape[1];
  const int64_t length = out.shape[2];
  const int64_t in_per_group = in_channels / args.groups;
  const int64_t stride = args.stride, pad = args.padding, dilation = args.dilation;
  // The un-cropped scatter extent. `output_padding` extends `length` past it;
  // those trailing cells are zero (plus bias), exactly as torch leaves them.
  const int64_t full = (in_len - 1) * stride + dilation * (kernel - 1) + 1;
  const float* xp = x.Ptr<float>();
  const float* wp = w.Ptr<float>();
  const float* bp = bias != nullptr ? bias->Ptr<float>() : nullptr;
  float* op = out.Ptr<float>();

  const int64_t rows = batch * out_channels;
  ForOutputRows(rows, in_per_group * in_len * kernel, [&](int64_t r0, int64_t r1) {
    std::vector<double> acc(static_cast<size_t>(full));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / out_channels;
      const int64_t dst_c = r - n * out_channels;
      std::fill(acc.begin(), acc.end(), 0.0);
      const int64_t g = dst_c / out_per_group;
      const int64_t oc = dst_c - g * out_per_group;
      const float* xn = xp + n * in_channels * in_len;
      for (int64_t ic = g * in_per_group; ic < (g + 1) * in_per_group; ++ic) {
        const float* wc = wp + (ic * out_per_group + oc) * kernel;
        for (int64_t t = 0; t < in_len; ++t) {
          const double value = xn[ic * in_len + t];
          if (value == 0.0) continue;
          double* dst = acc.data() + t * stride;
          for (int64_t k = 0; k < kernel; ++k) dst[k * dilation] += value * static_cast<double>(wc[k]);
        }
      }
      float* on = op + (n * out_channels + dst_c) * length;
      for (int64_t t = 0; t < length; ++t) {
        const int64_t p = t + pad;
        double value = p < full ? acc[static_cast<size_t>(p)] : 0.0;
        if (bp != nullptr) value += bp[dst_c];
        on[t] = static_cast<float>(value);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kConv1d, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Conv1dFn>(&Conv1dKernel)));
    RegisterOp(OpId::kConvTranspose1d, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ConvTranspose1dFn>(&ConvTranspose1dKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
