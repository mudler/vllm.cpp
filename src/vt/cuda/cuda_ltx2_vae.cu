// CUDA half of the LTX-2.5 conv video VAE device-resident kernel table
// (vt::OpId::kLtx2Vae).
//
// Row: LTX25-VAE-DEVICE-RESIDENCY. Spec:
// .agents/specs/ltx25-vae-device-residency.md. Issue #1451.
//
// The CPU sibling (src/vt/cpu/cpu_ltx2_vae.cpp) is the numeric reference, and it
// is the host loop every committed tests/vllm/models/ltx2_vae_goldens.inc entry
// was taken through. Each kernel here is the same expression, on the same
// values, in the same association order, so the two arms are BIT-IDENTICAL
// rather than "within a tolerance". That is what keeps the goldens a regression
// gate on the MOVE onto the device rather than a re-baselined gate.
//
// FMA CONTRACTION IS THE ONLY WAY THE TWO ARMS COULD DISAGREE, and it is the
// same argument src/vt/cuda/cuda_conv3d.cu makes at its own header: contraction
// fuses `acc + v * w` into one rounding and drops the intermediate rounding of
// the product. The host side is pinned against it project-wide by
// `-ffp-contract=off` (CMakeLists.txt); nvcc has no such pin, its `-fmad`
// default is on, and it contracts DOUBLE arithmetic as readily as float. So this
// file pins itself locally and visibly with `__fmul_rn`/`__fadd_rn` and, in
// group_norm, `__dmul_rn`/`__dadd_rn`. The divides and square roots are written
// as `__fdiv_rn`/`__fsqrt_rn`/`__ddiv_rn`/`__dsqrt_rn` for the same reason one
// step further out: `-use_fast_math` is source-scoped in this build today
// (CMakeLists.txt:1899, the NVFP4 FlashInfer tactics) and a later scope change
// must not be able to silently swap a correctly-rounded divide for an
// approximation under this table.
//
// F32 ONLY, REFUSED BY NAME, in the message shape the CPU arm uses. The whole
// LTX-2.5 conv video VAE decode is f32 in this tree after #1008 measured it, and
// ltx2_video_vae_kernels.h states that polarity under its DTYPE heading. A bf16
// storage arm is owed with the CUDA measurement (#1452) rather than guessed at.
//
// NEVER COMPILED, NEVER EXECUTED, ANYWHERE IN THIS PROJECT'S REACH. There is no
// nvcc on the box that wrote this file and no CI job here has a GPU runner, so
// nothing has built this translation unit and nothing has run a single one of
// these kernels. That is exactly what #1452 already records for
// src/vt/cuda/cuda_conv3d.cu. Read the bit-identity paragraph above as a DESIGN
// ARGUMENT and not as a measurement: it is a claim about what the arithmetic
// says, made by a reader of both arms, and it is owed a hardware run against the
// CPU arm on the same volume before anything cites it as a result.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (src/vt/cuda/cuda_ltx2.cu, src/vt/cuda/cuda_conv3d.cu): no existing kernel TU
// and no shared op array is edited.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/ltx2_video_vae_kernels.h"
#include "vt/ops.h"

namespace vllm::ltx2_vae {
namespace {

using vt::DType;
using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 1 ? 1 : (blocks < 65535 ? blocks : 65535));
}

// One refusal, spelled the same way at every entry and worded the same way as
// the CPU arm's (cpu_ltx2_vae.cpp's `RequireF32`), so a caller that reaches this table
// with the wrong storage gets the dtype and the op back rather than a
// reinterpreted buffer.
void RequireF32(DType dtype, const char* what) {
  VT_CHECK(dtype == DType::kF32,
           std::string("ltx2 vae ") + what +
               ": this arm serves f32 storage only; f16/bf16 is owed with the CUDA "
               "measurement (#1452)");
}

// `std::max`/`std::min` are host-only here; the pad's two clamps are written
// once rather than as four ternaries at the call sites.
__device__ inline int64_t Clamp(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// --- pixel_norm (cpu_ltx2_vae.cpp's `PixelNorm`) -----------------------------
//
// ONE THREAD PER PIXEL, and the reduction over channels stays serial inside that
// thread in ascending `c`. The channel count here is a VAE width (128 to 512),
// not a sequence length, so there is nothing to win by splitting it, and a tree
// reduction would be a different sum.
__global__ void PixelNormK(float* __restrict__ x, int64_t channels, int64_t spatial, float eps) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < spatial;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float mean_sq = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      const float v = x[c * spatial + i];
      mean_sq = __fadd_rn(mean_sq, __fmul_rn(v, v));
    }
    mean_sq = __fdiv_rn(mean_sq, static_cast<float>(channels));
    // ONE reciprocal per pixel, then a multiply per channel. Dividing per
    // channel instead is a different number in the last ulp and the committed
    // goldens are held to the first (see `pixel_norm`'s note in ltx2_video_vae_kernels.h).
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean_sq, eps)));
    for (int64_t c = 0; c < channels; ++c) {
      x[c * spatial + i] = __fmul_rn(x[c * spatial + i], inv);
    }
  }
}

void PixelNormCuda(Queue& q, void* xv, int64_t channels, int64_t spatial, float eps,
                   DType dtype) {
  RequireF32(dtype, "pixel_norm");
  if (channels <= 0 || spatial <= 0) return;
  PixelNormK<<<GridFor(spatial), kBlock, 0, AsStream(q)>>>(static_cast<float*>(xv), channels,
                                                           spatial, eps);
  Check(cudaGetLastError(), "ltx2 vae pixel_norm launch");
}

// --- group_norm (cpu_ltx2_vae.cpp's `GroupNorm`) -----------------------------
//
// THE ACCUMULATORS ARE f64 ON THIS ARM TOO, for the reason the CPU arm gives at
// its `GroupNorm` f64 note: `MiniMaxH3GroupNorm3d` sums the mean and the variance in double
// and forms `inv` in double, and every committed LTX-2.5 and MiniMax-H3 golden
// was taken through that. Narrowing them here would silently re-baseline four
// call sites in the video VAE alone.
//
// ONE THREAD PER GROUP, AND THAT IS A CHOICE OF CORRECTNESS OVER OCCUPANCY. The
// CPU arm sums the whole group serially, channel-major then spatial, in one
// order. A parallel tree reduction over (per_group * spatial) elements produces
// a DIFFERENT number: floating-point addition is not associative, and the arms
// would then agree only to a tolerance nobody has measured. A group is typically
// four to thirty-two groups per call, so this kernel leaves most of the device
// idle. The fast version, a two-pass block reduction with a recorded and
// separately gated summation order, is OWED WITH THE MEASUREMENT (#1452): it
// cannot be written blind, because the thing that would justify it is a
// hardware A/B against these numbers and no GPU has run either.
__global__ void GroupNormK(float* __restrict__ x, int64_t channels, int64_t spatial,
                           int64_t groups, const float* __restrict__ weight,
                           const float* __restrict__ bias, double eps) {
  const int64_t per_group = channels / groups;
  const int64_t count = per_group * spatial;
  for (int64_t g = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; g < groups;
       g += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t begin = g * per_group;
    double mean = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i)
        mean = __dadd_rn(mean, static_cast<double>(x[c * spatial + i]));
    }
    mean = __ddiv_rn(mean, static_cast<double>(count));
    double var = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = static_cast<double>(x[c * spatial + i]) - mean;
        var = __dadd_rn(var, __dmul_rn(d, d));
      }
    }
    var = __ddiv_rn(var, static_cast<double>(count));
    const double inv = __ddiv_rn(1.0, __dsqrt_rn(__dadd_rn(var, eps)));
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double normed = __dmul_rn(static_cast<double>(x[c * spatial + i]) - mean, inv);
        // The scale and the shift are applied in DOUBLE and narrowed once on
        // store, which is where the CPU arm's `static_cast<float>` sits.
        x[c * spatial + i] = static_cast<float>(__dadd_rn(
            __dmul_rn(normed, static_cast<double>(weight[c])), static_cast<double>(bias[c])));
      }
    }
  }
}

void GroupNormCuda(Queue& q, void* xv, int64_t channels, int64_t spatial, int64_t groups,
                   const void* wv, const void* bv, double eps, DType dtype) {
  RequireF32(dtype, "group_norm");
  VT_CHECK(groups > 0 && channels % groups == 0,
           "ltx2 vae group_norm: channels must be divisible by num_groups");
  if (channels <= 0 || spatial <= 0) return;
  GroupNormK<<<GridFor(groups), kBlock, 0, AsStream(q)>>>(
      static_cast<float*>(xv), channels, spatial, groups, static_cast<const float*>(wv),
      static_cast<const float*>(bv), eps);
  Check(cudaGetLastError(), "ltx2 vae group_norm launch");
}

// --- ada_ln (cpu_ltx2_vae.cpp's `AdaLn`) -------------------------------------
//
// One thread per element. `shift` and `scale` are re-formed per element rather
// than once per channel: they are the same two adds on the same two operands, so
// the value is identical, and hoisting them would need a per-channel launch
// shape this table has no reason to carry.
__global__ void AdaLnK(float* __restrict__ x, const float* __restrict__ table,
                       const float* __restrict__ embed, int64_t channels, int64_t spatial,
                       int64_t shift_row, int64_t scale_row) {
  const int64_t n = channels * spatial;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ch = idx / spatial;
    // table + embed FIRST, then the affine. Upstream forms `ada_values =
    // scale_shift_table + timestep` and only then modulates (resnet.py:135-148);
    // folding the two the other way round rounds differently.
    const float shift =
        __fadd_rn(table[shift_row * channels + ch], embed[shift_row * channels + ch]);
    const float scale =
        __fadd_rn(table[scale_row * channels + ch], embed[scale_row * channels + ch]);
    x[idx] = __fadd_rn(__fmul_rn(x[idx], __fadd_rn(1.0f, scale)), shift);
  }
}

void AdaLnCuda(Queue& q, void* xv, const void* tv, const void* ev, int64_t channels,
               int64_t spatial, int64_t rows, int64_t shift_row, int64_t scale_row,
               DType dtype) {
  RequireF32(dtype, "ada_ln");
  VT_CHECK(shift_row >= 0 && shift_row < rows && scale_row >= 0 && scale_row < rows,
           "ltx2 vae ada_ln: the shift/scale rows must lie inside the table");
  const int64_t n = channels * spatial;
  if (n <= 0) return;
  AdaLnK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(static_cast<float*>(xv),
                                                 static_cast<const float*>(tv),
                                                 static_cast<const float*>(ev), channels, spatial,
                                                 shift_row, scale_row);
  Check(cudaGetLastError(), "ltx2 vae ada_ln launch");
}

// --- spatial_noise (cpu_ltx2_vae.cpp's `SpatialNoise`) -----------------------
//
// The plane is an INPUT and is never drawn here. `Ltx2NoiseStream::Draw` is the
// reproducibility seam and it stays on the host (see `spatial_noise`'s note in ltx2_video_vae_kernels.h):
// a device-side generator would be a different stream, and the renders this
// project has captured are keyed to that one.
__global__ void SpatialNoiseK(float* __restrict__ x, const float* __restrict__ plane,
                              const float* __restrict__ scale, int64_t channels, int64_t t,
                              int64_t h, int64_t w) {
  const int64_t n = channels * t * h * w;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t wi = idx % w;
    const int64_t hi = (idx / w) % h;
    const int64_t c = idx / (w * h * t);
    x[idx] = __fadd_rn(x[idx], __fmul_rn(plane[hi * w + wi], scale[c]));
  }
}

void SpatialNoiseCuda(Queue& q, void* xv, const void* pv, const void* sv, int64_t channels,
                      int64_t t, int64_t h, int64_t w, DType dtype) {
  RequireF32(dtype, "spatial_noise");
  const int64_t n = channels * t * h * w;
  if (n <= 0) return;
  SpatialNoiseK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      static_cast<float*>(xv), static_cast<const float*>(pv), static_cast<const float*>(sv),
      channels, t, h, w);
  Check(cudaGetLastError(), "ltx2 vae spatial_noise launch");
}

// --- depth_to_space (cpu_ltx2_vae.cpp's `DepthToSpace`) ----------------------
//
// A pure GATHER, one thread per OUTPUT element: the host loop's scatter is a
// bijection onto the output, so inverting it touches every destination exactly
// once and moves no arithmetic. p1 IS THE OUTER FACTOR of the packed channel
// index; reading the unpack in any other order transposes every patch and still
// produces a plausible clip (see `depth_to_space`'s note in ltx2_video_vae_kernels.h).
__global__ void DepthToSpaceK(float* __restrict__ out, const float* __restrict__ x,
                              int64_t out_channels, int64_t t, int64_t h, int64_t w, int64_t st,
                              int64_t sh, int64_t sw) {
  const int64_t ot = t * st, oh = h * sh, ow = w * sw;
  const int64_t n = out_channels * ot * oh * ow;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ow_i = idx % ow;
    const int64_t oh_i = (idx / ow) % oh;
    const int64_t ot_i = (idx / (ow * oh)) % ot;
    const int64_t c = idx / (ow * oh * ot);
    const int64_t wi = ow_i / sw, p3 = ow_i % sw;
    const int64_t hi = oh_i / sh, p2 = oh_i % sh;
    const int64_t ti = ot_i / st, p1 = ot_i % st;
    const int64_t src_c = ((c * st + p1) * sh + p2) * sw + p3;
    out[idx] = x[((src_c * t + ti) * h + hi) * w + wi];
  }
}

void DepthToSpaceCuda(Queue& q, void* ov, const void* xv, int64_t out_channels, int64_t t,
                      int64_t h, int64_t w, int64_t st, int64_t sh, int64_t sw, DType dtype) {
  RequireF32(dtype, "depth_to_space");
  const int64_t n = out_channels * (t * st) * (h * sh) * (w * sw);
  if (n <= 0) return;
  DepthToSpaceK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(static_cast<float*>(ov),
                                                        static_cast<const float*>(xv),
                                                        out_channels, t, h, w, st, sh, sw);
  Check(cudaGetLastError(), "ltx2 vae depth_to_space launch");
}

// --- frame_slice (cpu_ltx2_vae.cpp's `FrameSlice`) ---------------------------
__global__ void FrameSliceK(float* __restrict__ out, const float* __restrict__ x,
                            int64_t channels, int64_t t, int64_t h, int64_t w, int64_t drop) {
  const int64_t ot = t - drop;
  const int64_t n = channels * ot * h * w;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t wi = idx % w;
    const int64_t hi = (idx / w) % h;
    const int64_t ti = (idx / (w * h)) % ot;
    const int64_t c = idx / (w * h * ot);
    out[idx] = x[((c * t + ti + drop) * h + hi) * w + wi];
  }
}

void FrameSliceCuda(Queue& q, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                    int64_t w, int64_t drop, DType dtype) {
  RequireF32(dtype, "frame_slice");
  VT_CHECK(drop >= 0 && drop < t, "ltx2 vae frame_slice: the slice must leave a frame");
  const int64_t n = channels * (t - drop) * h * w;
  if (n <= 0) return;
  FrameSliceK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      static_cast<float*>(ov), static_cast<const float*>(xv), channels, t, h, w, drop);
  Check(cudaGetLastError(), "ltx2 vae frame_slice launch");
}

// --- channel_repeat (cpu_ltx2_vae.cpp's `ChannelRepeat`) ---------------------
//
// torch's `repeat` TILES the whole tensor, so the block index is the OUTER axis
// and `out[r * block + j] = x[j]`. `repeat_interleave` would put the block index
// inner and is a different tensor (see `channel_repeat`'s note in ltx2_video_vae_kernels.h).
__global__ void ChannelRepeatK(float* __restrict__ out, const float* __restrict__ x,
                               int64_t block, int64_t repeat) {
  const int64_t n = block * repeat;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[idx] = x[idx % block];
  }
}

void ChannelRepeatCuda(Queue& q, void* ov, const void* xv, int64_t channels, int64_t spatial,
                       int64_t repeat, DType dtype) {
  RequireF32(dtype, "channel_repeat");
  const int64_t block = channels * spatial;
  const int64_t n = block * repeat;
  if (n <= 0) return;
  ChannelRepeatK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(static_cast<float*>(ov),
                                                         static_cast<const float*>(xv), block,
                                                         repeat);
  Check(cudaGetLastError(), "ltx2 vae channel_repeat launch");
}

// --- linear_cn (cpu_ltx2_vae.cpp's `LinearCn`) -------------------------------
//
// One thread is one (oc, i) pair, which is the CPU arm's row partition exactly:
// no two threads touch one accumulator, so the partition cannot change the
// arithmetic. The f32 accumulator is SEEDED WITH THE BIAS and takes one partial
// per input channel in ascending `ic`, because this is a 1x1x1 nn.Conv3d
// upstream (make_linear_nd, convolution.py:84-85) and so it takes vt::Conv3d's
// published accumulation contract (`vt::Conv3d`'s ACCUMULATION ORDER contract in ops.h).
__global__ void LinearCnK(float* __restrict__ out, const float* __restrict__ x,
                          const float* __restrict__ weight, const float* __restrict__ bias,
                          int64_t out_channels, int64_t in_channels, int64_t n) {
  const int64_t total = out_channels * n;
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < total;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t oc = r / n;
    const int64_t i = r % n;
    float acc = bias[oc];
    for (int64_t ic = 0; ic < in_channels; ++ic) {
      // __fmul_rn/__fadd_rn, never `acc += a * b`: see the file header.
      acc = __fadd_rn(acc, __fmul_rn(x[ic * n + i], weight[oc * in_channels + ic]));
    }
    out[oc * n + i] = acc;
  }
}

void LinearCnCuda(Queue& q, void* ov, const void* xv, const void* wv, const void* bv,
                  int64_t out_channels, int64_t in_channels, int64_t n, DType dtype) {
  RequireF32(dtype, "linear_cn");
  const int64_t total = out_channels * n;
  if (total <= 0) return;
  LinearCnK<<<GridFor(total), kBlock, 0, AsStream(q)>>>(
      static_cast<float*>(ov), static_cast<const float*>(xv), static_cast<const float*>(wv),
      static_cast<const float*>(bv), out_channels, in_channels, n);
  Check(cudaGetLastError(), "ltx2 vae linear_cn launch");
}

// --- unpatchify (cpu_ltx2_vae.cpp's `Unpatchify`) ----------------------------
//
// H TAKES q AND W TAKES r. Swapping them transposes every patch, which a
// shape-valid gate cannot see (see `unpatchify`'s note in ltx2_video_vae_kernels.h). Gather form
// again: the host scatter is a bijection onto the output.
__global__ void UnpatchifyK(float* __restrict__ out, const float* __restrict__ x,
                            int64_t channels, int64_t t, int64_t h, int64_t w, int64_t q,
                            int64_t r_stride) {
  const int64_t oh = h * q, ow = w * r_stride;
  const int64_t n = channels * t * oh * ow;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ow_i = idx % ow;
    const int64_t oh_i = (idx / ow) % oh;
    const int64_t f = (idx / (ow * oh)) % t;
    const int64_t c = idx / (ow * oh * t);
    const int64_t wi = ow_i / r_stride, ri = ow_i % r_stride;
    const int64_t hi = oh_i / q, qi = oh_i % q;
    const int64_t src_c = (c * r_stride + ri) * q + qi;
    out[idx] = x[((src_c * t + f) * h + hi) * w + wi];
  }
}

void UnpatchifyCuda(Queue& q, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                    int64_t w, int64_t q_stride, int64_t r_stride, DType dtype) {
  RequireF32(dtype, "unpatchify");
  const int64_t n = channels * t * (h * q_stride) * (w * r_stride);
  if (n <= 0) return;
  UnpatchifyK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(static_cast<float*>(ov),
                                                      static_cast<const float*>(xv), channels, t,
                                                      h, w, q_stride, r_stride);
  Check(cudaGetLastError(), "ltx2 vae unpatchify launch");
}

// --- pad (cpu_ltx2_vae.cpp's `Pad`) ------------------------------------------
//
// torch's "reflect" EXCLUDES the edge sample: [a b c] -> b a b c b. The loop
// form is the host helper's, character for character, because the alternative
// closed forms disagree with it once `index` is more than one period out.
__device__ int64_t ReflectIndex(int64_t index, int64_t size) {
  if (size == 1) return 0;
  while (index < 0 || index >= size) {
    if (index < 0) index = -index;
    if (index >= size) index = 2 * (size - 1) - index;
  }
  return index;
}

__device__ int64_t SpatialIndex(int64_t index, int64_t size, int mode, bool* zero) {
  *zero = false;
  if (index >= 0 && index < size) return index;
  switch (mode) {
    case vllm::ltx2_vae::kLtx2VaePadZeros:
      *zero = true;
      return 0;
    case vllm::ltx2_vae::kLtx2VaePadReflect:
      return ReflectIndex(index, size);
    case vllm::ltx2_vae::kLtx2VaePadReplicate:
      return Clamp(index, 0, size - 1);
    default:
      break;
  }
  *zero = true;
  return 0;
}

// A pure GATHER, one source element per destination element and no reduction, so
// the partition cannot change any arithmetic. The zero-mode taps are SKIPPED
// rather than written, exactly as the host loop skips them, which is why the
// launcher zeroes the whole buffer first.
__global__ void PadK(float* __restrict__ out, const float* __restrict__ x, int64_t channels,
                     int64_t t, int64_t h, int64_t w, int64_t pad_t_leading,
                     int64_t pad_t_trailing, int64_t pad_hw, int mode) {
  const int64_t pt = t + pad_t_leading + pad_t_trailing;
  const int64_t ph = h + 2 * pad_hw;
  const int64_t pw = w + 2 * pad_hw;
  const int64_t n = channels * pt * ph * pw;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t wi = idx % pw;
    const int64_t hi = (idx / pw) % ph;
    const int64_t ti = (idx / (pw * ph)) % pt;
    const int64_t c = idx / (pw * ph * pt);
    // Temporal padding REPLICATES the edge frame, never zeros: upstream
    // concatenates k_t - 1 copies of frame 0 (convolution.py:305-311), which a
    // clamp at 0 reproduces.
    const int64_t st = Clamp(ti - pad_t_leading, 0, t - 1);
    bool zero_h = false;
    const int64_t sh = SpatialIndex(hi - pad_hw, h, mode, &zero_h);
    bool zero_w = false;
    const int64_t sw = SpatialIndex(wi - pad_hw, w, mode, &zero_w);
    if (zero_h || zero_w) continue;
    out[idx] = x[((c * t + st) * h + sh) * w + sw];
  }
}

void PadCuda(Queue& q, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
             int64_t w, int64_t pad_t_leading, int64_t pad_t_trailing, int64_t pad_hw, int mode,
             DType dtype) {
  RequireF32(dtype, "pad");
  const int64_t pt = t + pad_t_leading + pad_t_trailing;
  const int64_t ph = h + 2 * pad_hw;
  const int64_t pw = w + 2 * pad_hw;
  const int64_t n = channels * pt * ph * pw;
  if (n <= 0) return;
  // The CPU arm's `std::fill(out, out + n, 0.0f)`. It is not decoration: the
  // kZeros mode SKIPS its taps instead of writing them, so without this the
  // padded border would be whatever the allocator last left there. An all-zero
  // byte pattern is exactly +0.0f in IEEE-754 binary32, so a memset is the same
  // buffer the host fill produces.
  Check(cudaMemsetAsync(ov, 0, static_cast<size_t>(n) * sizeof(float), AsStream(q)),
        "ltx2 vae pad zero-fill");
  PadK<<<GridFor(n), kBlock, 0, AsStream(q)>>>(static_cast<float*>(ov),
                                               static_cast<const float*>(xv), channels, t, h, w,
                                               pad_t_leading, pad_t_trailing, pad_hw, mode);
  Check(cudaGetLastError(), "ltx2 vae pad launch");
}

// The ten pointers in ltx2_video_vae_kernels.h's DECLARATION ORDER
// (pixel_norm, group_norm, ada_ln, spatial_noise, depth_to_space, frame_slice,
// channel_repeat, linear_cn, unpatchify, pad). A permutation here compiles
// cleanly whenever two entries have compatible signatures and then calls the
// wrong kernel, so the order is written out above rather than trusted to the
// diff.
const Ltx2VaeDeviceKernels kKernels{
    &PixelNormCuda,  &GroupNormCuda,     &AdaLnCuda,    &SpatialNoiseCuda, &DepthToSpaceCuda,
    &FrameSliceCuda, &ChannelRepeatCuda, &LinearCnCuda, &UnpatchifyCuda,   &PadCuda,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLtx2Vae, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vllm::ltx2_vae
