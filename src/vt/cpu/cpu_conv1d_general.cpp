// CPU providers for `vt::Conv1d` and `vt::ConvTranspose1d` — the BigVGAN / DAC
// vocoder convolutions (#672, .agents/specs/minimax-music3.md §11.4).
//
// PROVENANCE. Both kernel bodies below descend from the
// `vllm::vocoder1d::Conv1d` / `vllm::vocoder1d::ConvTranspose1d` host loops as
// they stood at 8fa405bb7 (src/vllm/model_executor/models/vocoder1d.cpp:65-161).
// The visit order, the seeding of the bias, the `value == 0.0` skip and the
// output-channel partition are unchanged from that transcription. The
// ACCUMULATOR WIDTH is not: it was f64 there and it is f32 here, narrowed by
// VT-CONV1D-F32-ACC (#1474) to the width torch accumulates a float convolution
// in. The four models that decode through `vocoder1d` (MiniMax-Music3,
// MiniMax-H3's audio VAE, LTX-2.5's audio VAE, IndexTTS-2.5) keep every
// committed golden they already had, MEASURED rather than argued: over the 194
// golden arms in their four suites, 182 did not move at all, 10 moved toward
// the golden, and 2 moved away from it by a single unit in the last place —
// MiniMax-H3's audio-VAE waveform from 4.19095e-09 to 4.65661e-09 against a
// 1e-05 bound (0.047 % of it), and MiniMax-Music3's vocoder arm from
// 2.98023e-08 to 5.96046e-08 against a 1e-06 absolute floor (5.96 % of it).
// Both are reported rather than absorbed; neither is near anything.
//
// Upstream semantics: `torch.nn.functional.conv1d` / `conv_transpose1d` as the
// checkpoints instantiate them — minimax_music3_vocoder.py:42,44,89,98
// (`nn.Conv1d`) and :55 (`nn.ConvTranspose1d`); LTX-2.5
// audio_vae/vocoder.py:104-184 for the alias-free resample pair.
//
// WHY f32. It is what torch accumulates a float convolution in, MEASURED on a
// 27-tap `[+1e8, 0.1 x 25, -1e8]` probe where an f32 accumulator lands on
// exactly 0.0 in any summation order and an f64 one lands near 2.5: `F.conv1d`
// and `F.conv_transpose1d` both return 0.0 at f32, and `F.conv1d` returns 0.0
// at bf16 as well. vLLM owns neither op at the parity pin `555967922`, so torch
// is the reference through the per-consumer secondary oracles — and it is the
// same oracle all four consumers' goldens came from, because every one of those
// generators casts to f32. The width this replaced was recorded as "what every
// golden was taken with", which was not true. See include/vt/ops.h at
// vt::Conv1d and .agents/specs/vt-conv1d-f32-accumulator.md.
//
// WHY NOT A MODE OF `vt::DepthwiseConv1d`. A transposed convolution is not a
// parameterisation of a forward one, and these two ops differ from the
// depthwise one in the bias seeding and the zero-skip. The WIDTHS now agree —
// that op accumulates f32 and so do these — so the sibling relationship is
// about expressiveness, and is no longer also about a divergence in width.
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

#include "cpu_conv1d_block.h"
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
// is in_per_group * kernel = 384 * 7 = 2688 STRICTLY DEPENDENT additions per
// output element: no instruction-level parallelism, no vectorisation at any
// width, and a measured ~2.8-3.0 cycles per multiply-accumulate, which is what
// an `fadd` latency of 3 predicts and nothing else does.
//
// This form holds kConv1dPosTile accumulators, one per output position, and
// hoists the (ic, k) sweep outside them. Fix ANY single output cell and read
// what it receives, in order: the bias, then (ic=0,k=0), (ic=0,k=1), ... — the
// identical sequence of IEEE-754 additions of the identical products, in the
// identical order. The additions are INTERLEAVED across independent cells
// rather than serialised into one. That is a scheduling change and not an
// arithmetic one, so `memcmp` equality with the CUDA provider survives BY
// CONSTRUCTION rather than within a tolerance. That is equality with the CUDA
// provider at the SAME width: #1474 narrowed both arms in one change, so this
// is no longer also equality with the f64 host loop at 8fa405bb7.
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
  // THE DECOMPOSITION. One unit of work is a (time block, output row) PAIR, and
  // the pair is flattened TIME-BLOCK-MAJOR so that consecutive units share a
  // block. `ParallelForRows` hands a thread a contiguous run of units
  // (cpu_threadpool.cpp:441-457), so a thread walks all of one block's output
  // rows against ONE resident activation slice before it moves on — which is
  // the reuse the row-only partition never took. The seam is unchanged and is
  // reached exactly as `cpu_paged_attn.cpp:211` and `cpu_attn_relpos.cpp:89`
  // reach it, with a flattened pair rather than a new primitive.
  //
  // BIT-IDENTITY IS BY CONSTRUCTION AND IS THE SAME ARGUMENT AS THE TILING'S.
  // Fix any single output cell: it is produced by exactly one unit, which seeds
  // it with the bias and sweeps (ic ascending, k ascending) into it — the
  // identical sequence of IEEE-754 additions of the identical products, in the
  // identical order, in an accumulator of the identical width. No cell is
  // touched twice, nothing is accumulated atomically, and no reduction is
  // split. Which THREAD computes a cell moves; what the cell receives does not.
  //
  // AND IT REACHES A SHAPE THE ROW PARTITION COULD NOT. `conv_out` has ONE
  // output row, so `rows == 1` ran the whole convolution INLINE on the caller
  // at every thread count; `blocks * rows` gives it `blocks` units.
  const int64_t block_len =
      Conv1dTimeBlock(in_per_group, out_channels, kernel, stride, dilation, in_len, length);
  const int64_t blocks = (length + block_len - 1) / block_len;
  const int64_t units = blocks * rows;
  // The size guard sees the same total work it saw before: `units * per_unit`
  // is `rows * length * in_per_group * kernel` up to the last block's
  // remainder, so the inline-vs-pooled decision for a small shape is unchanged.
  ForOutputRows(units, block_len * in_per_group * kernel, [&](int64_t u0, int64_t u1) {
    alignas(64) float acc[kConv1dPosTile];
    for (int64_t u = u0; u < u1; ++u) {
      const int64_t blk = u / rows;
      const int64_t r = u - blk * rows;
      const int64_t n = r / out_channels;
      const int64_t oc = r - n * out_channels;
      const int64_t g = oc / out_per_group;
      const float* xn = xp + n * in_channels * in_len;
      float* on = op + (n * out_channels + oc) * length;
      const float seed = bp != nullptr ? bp[oc] : 0.0F;
      const int64_t t_begin = blk * block_len;
      const int64_t t_end = std::min<int64_t>(length, t_begin + block_len);
      for (int64_t t0 = t_begin; t0 < t_end; t0 += kConv1dPosTile) {
        const int64_t tile = std::min<int64_t>(kConv1dPosTile, t_end - t0);
        for (int64_t i = 0; i < tile; ++i) acc[i] = seed;
        for (int64_t ic = 0; ic < in_per_group; ++ic) {
          const float* xc = xn + (g * in_per_group + ic) * in_len;
          const float* wc = wp + (oc * in_per_group + ic) * kernel;
          for (int64_t k = 0; k < kernel; ++k) {
            const float wv = wc[k];
            // Position of tile slot i is `base + i * stride`; it contributes iff
            // that lies in [0, in_len), which bounds i to [lo, hi).
            const int64_t base = t0 * stride - pad + k * dilation;
            int64_t lo = base < 0 ? (-base + stride - 1) / stride : 0;
            const int64_t room = in_len - base;
            int64_t hi = room <= 0 ? 0 : std::min<int64_t>(tile, (room + stride - 1) / stride);
            if (lo >= hi) continue;
            // Formed only after the clamp, so it never points before `xc`.
            const float* xs = xc + base + lo * stride;
            float* ap = acc + lo;
            const int64_t span = hi - lo;
            if (stride != 1) {
              for (int64_t i = 0; i < span; ++i) ap[i] += xs[i * stride] * wv;
            } else if (span == kConv1dPosTile) {
              // The whole-tile case, and the only loop here with a compile-time
              // trip count. It is what the vocoder's own shapes take.
              for (int64_t i = 0; i < kConv1dPosTile; ++i) ap[i] += xs[i] * wv;
            } else {
              int64_t i = 0;
              for (; i + kConv1dChunk <= span; i += kConv1dChunk) {
                float* a8 = ap + i;
                const float* x8 = xs + i;
                for (int64_t j = 0; j < kConv1dChunk; ++j) a8[j] += x8[j] * wv;
              }
              for (; i < span; ++i) ap[i] += xs[i] * wv;
            }
          }
        }
        for (int64_t i = 0; i < tile; ++i) on[t0 + i] = acc[i];
      }
    }
  });
}

// Fixed tap width for the contiguous (dilation == 1) scatter; 4 divides every
// kernel the four consumers run (4, 8, 12, 16) with no tail.
constexpr int64_t kConvTranspose1dTapChunk = 4;

// torch.nn.functional.conv_transpose1d. Weight is [Cin, Cout/groups, K].
//
// The SCATTER form of vocoder1d.cpp:136-158 @ 8fa405bb7, at f32 rather than
// that loop's f64 (#1474): one destination channel at a time, a scratch line of
// `full` cells, inputs
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
    std::vector<float> acc(static_cast<size_t>(full));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t n = r / out_channels;
      const int64_t dst_c = r - n * out_channels;
      std::fill(acc.begin(), acc.end(), 0.0F);
      const int64_t g = dst_c / out_per_group;
      const int64_t oc = dst_c - g * out_per_group;
      const float* xn = xp + n * in_channels * in_len;
      for (int64_t ic = g * in_per_group; ic < (g + 1) * in_per_group; ++ic) {
        const float* wc = wp + (ic * out_per_group + oc) * kernel;
        for (int64_t t = 0; t < in_len; ++t) {
          const float value = xn[ic * in_len + t];
          if (value == 0.0F) continue;
          float* dst = acc.data() + t * stride;
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
            float* dk = dst + k;
            const float* wk = wc + k;
            for (int64_t j = 0; j < kConvTranspose1dTapChunk; ++j) dk[j] += value * wk[j];
          }
          for (; k < kernel; ++k) dst[k] += value * wc[k];
        } else {
          for (int64_t k = 0; k < kernel; ++k) dst[k * dilation] += value * wc[k];
        }
        }
      }
      float* on = op + (n * out_channels + dst_c) * length;
      for (int64_t t = 0; t < length; ++t) {
        const int64_t p = t + pad;
        float value = p < full ? acc[static_cast<size_t>(p)] : 0.0F;
        if (bp != nullptr) value += bp[dst_c];
        on[t] = value;
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
