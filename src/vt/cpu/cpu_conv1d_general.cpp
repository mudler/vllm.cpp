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

// How many OUTPUT POSITIONS share one sweep of the (ic, k) weights. It is not a
// blocking heuristic and it is not tunable at run time: it is the number of
// INDEPENDENT f64 accumulator chains the kernel offers the machine, and the
// whole of #1334 is that the shipped loop offered exactly one. 32 doubles is
// four AVX-512 registers or sixteen NEON ones, and it measured best of {8, 16,
// 32, 64} at the vocoder's shapes on both -O2 and -O3.
constexpr int64_t kConv1dPosTile = 32;

// The fixed width the in-range part of a tile is chunked into when the tile is
// not whole. GCC's -O2 vector cost model is `very-cheap` and takes only a loop
// whose trip count is a known multiple of the vector width, so a runtime bound
// is the difference between 1.1x and 5.2x on the SAME source. Release is -O3
// and CI builds it, but scripts/dgx-bringup.sh builds RelWithDebInfo, and a
// kernel that is fast only under the flag the next measurement happens to pick
// is a kernel nobody can compare.
constexpr int64_t kConv1dChunk = 8;

// out[n, oc, t] = bias[oc] + Sum_{ic,k} x[n, g*Cin/g + ic, t*stride - padding +
// k*dilation] * w[oc, ic, k], taps outside [0, Lin) SKIPPED (zero padding).
//
// THE ARITHMETIC IS THE ONE vocoder1d.cpp:90-100 @ 8fa405bb7 PERFORMED, AND
// THE ORDER IS TOO. What changed (#1334) is only which cells are in flight at
// once. The shipped form seeded one accumulator with the bias and swept
// (ic ascending, k ascending) into it, which for a MiniMax-Music3 residual unit
// is in_per_group * kernel = 384 * 7 = 2688 STRICTLY DEPENDENT f64 additions per
// output element: no instruction-level parallelism, no vectorisation at any
// width, and a measured ~2.8-3.0 cycles per multiply-accumulate, which is what
// an `fadd` latency of 3 predicts and nothing else does.
//
// This form holds kConv1dPosTile accumulators, one per output position, and
// hoists the (ic, k) sweep outside them. Fix ANY single output cell and read
// what it receives, in order: the bias, then (ic=0,k=0), (ic=0,k=1), ... — the
// identical sequence of IEEE-754 double additions of the identical double
// products, in the identical order. The additions are INTERLEAVED across
// independent cells rather than serialised into one. That is a scheduling
// change and not an arithmetic one, so `memcmp` equality with the pre-change
// loop, with the CUDA provider, and with all four consumers' goldens survives BY
// CONSTRUCTION rather than within a tolerance.
//
// The zero-padding skip moves from a per-position test to a CLAMP on the tile,
// which is the same set: for a fixed k the positions with 0 <= t*stride - pad +
// k*dilation < Lin form one contiguous run of t, so the (t, ic, k) triples this
// skips are exactly the triples the shipped loop skipped. Every existing caller
// passes padding == 0, so the clamp's edges are reached only by the op's torch
// arm — which is why tests/vt/test_ops_conv1d_general.cpp gates them and why
// this row added the tile-geometry cases that do.
//
// stride > 1 keeps the shipped gather (`acc[i] += x[base + i*stride] * w`),
// because a strided load does not vectorise and a gather instruction is slower
// than the scalar it replaces here. The MiniMax-Music3 vocoder's Conv1d calls
// are all stride 1; the strided caller is the alias-free DOWNSAMPLE in
// vocoder1d::AliasFreeActivation1d::Apply, which is depthwise (in_per_group ==
// 1) and therefore has a chain one tap deep — it is not the shape this is
// about. .agents/specs/minimax-music3.md §18.4.
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
    alignas(64) double acc[kConv1dPosTile];
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / out_channels;
      const int64_t oc = r - n * out_channels;
      const int64_t g = oc / out_per_group;
      const float* xn = xp + n * in_channels * in_len;
      float* on = op + (n * out_channels + oc) * length;
      const double seed = bp != nullptr ? bp[oc] : 0.0;
      for (int64_t t0 = 0; t0 < length; t0 += kConv1dPosTile) {
        const int64_t tile = std::min<int64_t>(kConv1dPosTile, length - t0);
        for (int64_t i = 0; i < tile; ++i) acc[i] = seed;
        for (int64_t ic = 0; ic < in_per_group; ++ic) {
          const float* xc = xn + (g * in_per_group + ic) * in_len;
          const float* wc = wp + (oc * in_per_group + ic) * kernel;
          for (int64_t k = 0; k < kernel; ++k) {
            const double wv = static_cast<double>(wc[k]);
            // Position of tile slot i is `base + i * stride`; it contributes iff
            // that lies in [0, in_len), which bounds i to [lo, hi).
            const int64_t base = t0 * stride - pad + k * dilation;
            int64_t lo = base < 0 ? (-base + stride - 1) / stride : 0;
            const int64_t room = in_len - base;
            int64_t hi = room <= 0 ? 0 : std::min<int64_t>(tile, (room + stride - 1) / stride);
            if (lo >= hi) continue;
            // Formed only after the clamp, so it never points before `xc`.
            const float* xs = xc + base + lo * stride;
            double* ap = acc + lo;
            const int64_t span = hi - lo;
            if (stride != 1) {
              for (int64_t i = 0; i < span; ++i)
                ap[i] += static_cast<double>(xs[i * stride]) * wv;
            } else if (span == kConv1dPosTile) {
              // The whole-tile case, and the only loop here with a compile-time
              // trip count. It is what the vocoder's own shapes take.
              for (int64_t i = 0; i < kConv1dPosTile; ++i)
                ap[i] += static_cast<double>(xs[i]) * wv;
            } else {
              int64_t i = 0;
              for (; i + kConv1dChunk <= span; i += kConv1dChunk) {
                double* a8 = ap + i;
                const float* x8 = xs + i;
                for (int64_t j = 0; j < kConv1dChunk; ++j)
                  a8[j] += static_cast<double>(x8[j]) * wv;
              }
              for (; i < span; ++i) ap[i] += static_cast<double>(xs[i]) * wv;
            }
          }
        }
        for (int64_t i = 0; i < tile; ++i) on[t0 + i] = static_cast<float>(acc[i]);
      }
    }
  });
}

// Fixed tap width for the contiguous (dilation == 1) scatter; 4 divides every
// kernel the four consumers run (4, 8, 12, 16) with no tail.
constexpr int64_t kConvTranspose1dTapChunk = 4;

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
          // The taps of ONE input value land in `kernel` DISTINCT cells, so this
        // scatter already has instruction-level parallelism and chunking it
        // reorders nothing: no cell receives two of these adds. What the chunk
        // buys is the compile-time trip count -O2's `very-cheap` vector model
        // needs (2.7-2.9x there on the three kernels of 8 taps or more; -O3
        // already finds it, which is why this op is ~6 % of the chain's wall and
        // Conv1d is ~94 %). §18.5.
        if (dilation == 1) {
          int64_t k = 0;
          for (; k + kConvTranspose1dTapChunk <= kernel; k += kConvTranspose1dTapChunk) {
            double* dk = dst + k;
            const float* wk = wc + k;
            for (int64_t j = 0; j < kConvTranspose1dTapChunk; ++j)
              dk[j] += value * static_cast<double>(wk[j]);
          }
          for (; k < kernel; ++k) dst[k] += value * static_cast<double>(wc[k]);
        } else {
          for (int64_t k = 0; k < kernel; ++k)
            dst[k * dilation] += value * static_cast<double>(wc[k]);
        }
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
