// LTX-2.5 CONV VIDEO VAE — the CPU arm of vt::OpId::kLtx2Vae.
//
// Row: LTX25-VAE-DEVICE-RESIDENCY. Spec:
// .agents/specs/ltx25-vae-device-residency.md. Issue #1451.
//
// EVERY KERNEL HERE IS A HOST LOOP THAT ALREADY EXISTED, MOVED. The arithmetic,
// its association order and its accumulator widths are transcribed from
// src/vllm/model_executor/models/ltx2_video_vae.cpp without change, which is
// what makes tests/vllm/models/ltx2_vae_goldens.inc a regression gate on the
// MOVE rather than a re-baselined one. Where an accumulator is f64 it is f64
// here too, and the header beside each entry says why.
//
// The kernels are f32-STORAGE only and refuse anything else by name, for the
// reason ltx2_video_vae_kernels.h gives under its DTYPE heading: the whole
// LTX-2.5 conv video VAE decode is f32 in this tree after #1008 measured it, and
// a bf16 storage arm is owed with the CUDA measurement (#1452) rather than
// guessed at.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "vllm/model_executor/models/ltx2_video_vae_kernels.h"
#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// One refusal, spelled the same way at every entry, so a caller that reaches
// this table with the wrong storage gets the dtype and the op back rather than a
// reinterpreted buffer. Mirrors src/vt/cuda/cuda_conv3d.cu:154-156.
void RequireF32(DType dtype, const char* what) {
  VT_CHECK(dtype == DType::kF32,
           std::string("ltx2 vae ") + what +
               ": this arm serves f32 storage only; f16/bf16 is owed with the CUDA "
               "measurement (#1452)");
}

// --- PixelNorm (ltx2_video_vae.cpp:381-400) ---------------------------------
void PixelNorm(Queue&, void* xv, int64_t channels, int64_t spatial, float eps, DType dtype) {
  RequireF32(dtype, "pixel_norm");
  float* x = static_cast<float*>(xv);
  for (int64_t i = 0; i < spatial; ++i) {
    float mean_sq = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      const float v = x[c * spatial + i];
      mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(channels);
    // ONE reciprocal per pixel, then a multiply per channel: the host loop's
    // shape, kept so the move changes nothing. NO GATE HOLDS IT -- a per-channel
    // divide leaves test_ltx2_vae fully green, because the difference is the last
    // ulp and the golden tolerance absorbs it. See the header.
    const float inv = 1.0f / std::sqrt(mean_sq + eps);
    for (int64_t c = 0; c < channels; ++c) x[c * spatial + i] = x[c * spatial + i] * inv;
  }
}

// --- GroupNorm over [C, T, H, W] (minimax_h3_vae_cnn.cpp:117-147) -----------
//
// THE ACCUMULATORS ARE f64 AND NOTHING GATES THAT. `MiniMaxH3GroupNorm3d` sums
// the mean and the variance in double and forms `inv` in double, and every
// committed LTX-2.5 and MiniMax-H3 golden was taken through that, so f64 is what
// this transcription must keep to leave the numbers alone.
//
// A fresh review MEASURED the protection and there is none: mutating `double
// mean` to `float mean` builds clean and leaves test_ltx2_vae at 45/45 and
// 3152/3152 GREEN. The goldens cannot see the width -- the double-accumulator
// -through-float trap -- while the same review's SCALE mutation on this kernel
// reds five cases. So the width is a deliberate choice with no gate behind it,
// and a later edit can narrow it silently. That is recorded under `## Owed` in
// .agents/specs/ltx25-vae-device-residency.md rather than left as a comment
// asserting a safety that does not exist.
//
// Whether f64 is itself the right mirror of torch is a separate question, and
// this row does not reopen it.
void GroupNorm(Queue&, void* xv, int64_t channels, int64_t spatial, int64_t groups,
               const void* wv, const void* bv, double eps, DType dtype) {
  RequireF32(dtype, "group_norm");
  VT_CHECK(groups > 0 && channels % groups == 0,
           "ltx2 vae group_norm: channels must be divisible by num_groups");
  float* x = static_cast<float*>(xv);
  const float* weight = static_cast<const float*>(wv);
  const float* bias = static_cast<const float*>(bv);
  const int64_t per_group = channels / groups;
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t begin = g * per_group;
    const int64_t count = per_group * spatial;
    double mean = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) mean += x[c * spatial + i];
    }
    mean /= static_cast<double>(count);
    double var = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = x[c * spatial + i] - mean;
        var += d * d;
      }
    }
    var /= static_cast<double>(count);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double normed = (x[c * spatial + i] - mean) * inv;
        x[c * spatial + i] = static_cast<float>(normed * weight[c] + bias[c]);
      }
    }
  }
}

// --- ApplyAdaLn (ltx2_video_vae.cpp:529-556) --------------------------------
void AdaLn(Queue&, void* xv, const void* tv, const void* ev, int64_t channels, int64_t spatial,
           int64_t rows, int64_t shift_row, int64_t scale_row, DType dtype) {
  RequireF32(dtype, "ada_ln");
  VT_CHECK(shift_row >= 0 && shift_row < rows && scale_row >= 0 && scale_row < rows,
           "ltx2 vae ada_ln: the shift/scale rows must lie inside the table");
  float* x = static_cast<float*>(xv);
  const float* table = static_cast<const float*>(tv);
  const float* embed = static_cast<const float*>(ev);
  for (int64_t ch = 0; ch < channels; ++ch) {
    // table + embed FIRST, then the affine — upstream's order (resnet.py:135-148).
    const float shift = table[shift_row * channels + ch] + embed[shift_row * channels + ch];
    const float scale = table[scale_row * channels + ch] + embed[scale_row * channels + ch];
    for (int64_t i = 0; i < spatial; ++i) {
      x[ch * spatial + i] = x[ch * spatial + i] * (1.0f + scale) + shift;
    }
  }
}

// --- FeedSpatialNoise (ltx2_video_vae.cpp:507-527) --------------------------
void SpatialNoise(Queue&, void* xv, const void* pv, const void* sv, int64_t channels, int64_t t,
                  int64_t h, int64_t w, DType dtype) {
  RequireF32(dtype, "spatial_noise");
  float* x = static_cast<float*>(xv);
  const float* plane = static_cast<const float*>(pv);
  const float* scale = static_cast<const float*>(sv);
  for (int64_t c = 0; c < channels; ++c) {
    const float s = scale[c];
    for (int64_t ti = 0; ti < t; ++ti) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          x[((c * t + ti) * h + hi) * w + wi] += plane[hi * w + wi] * s;
        }
      }
    }
  }
}

// --- DepthToSpaceUpsample's expand (ltx2_video_vae.cpp:605-631) -------------
void DepthToSpace(Queue&, void* ov, const void* xv, int64_t out_channels, int64_t t, int64_t h,
                  int64_t w, int64_t st, int64_t sh, int64_t sw, DType dtype) {
  RequireF32(dtype, "depth_to_space");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const int64_t ot = t * st, oh = h * sh, ow = w * sw;
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t p1 = 0; p1 < st; ++p1) {
      for (int64_t p2 = 0; p2 < sh; ++p2) {
        for (int64_t p3 = 0; p3 < sw; ++p3) {
          // p1 is the OUTER factor of the packed channel index; any other order
          // transposes every patch and still renders.
          const int64_t src_c = ((c * st + p1) * sh + p2) * sw + p3;
          for (int64_t ti = 0; ti < t; ++ti) {
            for (int64_t hi = 0; hi < h; ++hi) {
              for (int64_t wi = 0; wi < w; ++wi) {
                out[((c * ot + ti * st + p1) * oh + hi * sh + p2) * ow + wi * sw + p3] =
                    x[((src_c * t + ti) * h + hi) * w + wi];
              }
            }
          }
        }
      }
    }
  }
}

// --- drop_first_frame (ltx2_video_vae.cpp:632-654) --------------------------
void FrameSlice(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                int64_t w, int64_t drop, DType dtype) {
  RequireF32(dtype, "frame_slice");
  VT_CHECK(drop >= 0 && drop < t, "ltx2 vae frame_slice: the slice must leave a frame");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const int64_t ot = t - drop;
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t ti = 0; ti < ot; ++ti) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          out[((c * ot + ti) * h + hi) * w + wi] = x[((c * t + ti + drop) * h + hi) * w + wi];
        }
      }
    }
  }
}

// --- the residual's channel repeat (ltx2_video_vae.cpp:660-670) -------------
void ChannelRepeat(Queue&, void* ov, const void* xv, int64_t channels, int64_t spatial,
                   int64_t repeat, DType dtype) {
  RequireF32(dtype, "channel_repeat");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const int64_t block = channels * spatial;
  // torch's `repeat` TILES: the block index is the OUTER axis.
  for (int64_t r = 0; r < repeat; ++r) std::copy(x, x + block, out + r * block);
}

// --- Linear3d (ltx2_video_vae.cpp:339-368) ----------------------------------
void LinearCn(Queue&, void* ov, const void* xv, const void* wv, const void* bv,
              int64_t out_channels, int64_t in_channels, int64_t n, DType dtype) {
  RequireF32(dtype, "linear_cn");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const float* weight = static_cast<const float*>(wv);
  const float* bias = static_cast<const float*>(bv);
  // The same partition-the-outputs discipline the host helper used: one row is
  // one (oc, i) pair, so no two workers touch one accumulator and the partition
  // cannot change the arithmetic.
  ParallelForRows(CurrentThreadpool(), out_channels * n, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t oc = r / n;
      const int64_t i = r % n;
      // f32 accumulator SEEDED WITH THE BIAS — this is an nn.Conv3d upstream
      // (make_linear_nd, convolution.py:84-85), so it takes vt::Conv3d's
      // published accumulation contract.
      float acc = bias[oc];
      for (int64_t ic = 0; ic < in_channels; ++ic) {
        acc += x[ic * n + i] * weight[oc * in_channels + ic];
      }
      out[oc * n + i] = acc;
    }
  });
}

// --- unpatchify (ltx2_video_vae.cpp:945-970) --------------------------------
void Unpatchify(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                int64_t w, int64_t q, int64_t r_stride, DType dtype) {
  RequireF32(dtype, "unpatchify");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const int64_t oh = h * q, ow = w * r_stride;
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t ri = 0; ri < r_stride; ++ri) {
      for (int64_t qi = 0; qi < q; ++qi) {
        // H takes q and W takes r. Swapping them transposes every patch.
        const int64_t src_c = (c * r_stride + ri) * q + qi;
        for (int64_t f = 0; f < t; ++f) {
          for (int64_t hi = 0; hi < h; ++hi) {
            for (int64_t wi = 0; wi < w; ++wi) {
              out[((c * t + f) * oh + hi * q + qi) * ow + wi * r_stride + ri] =
                  x[((src_c * t + f) * h + hi) * w + wi];
            }
          }
        }
      }
    }
  }
}

// --- the CausalConv3d pad (ltx2_video_vae.cpp:262-300) ----------------------
//
// torch's "reflect" EXCLUDES the edge sample: [a b c] -> b a b c b.
int64_t ReflectIndex(int64_t index, int64_t size) {
  if (size == 1) return 0;
  while (index < 0 || index >= size) {
    if (index < 0) index = -index;
    if (index >= size) index = 2 * (size - 1) - index;
  }
  return index;
}

int64_t SpatialIndex(int64_t index, int64_t size, int mode, bool* zero) {
  *zero = false;
  if (index >= 0 && index < size) return index;
  switch (mode) {
    case vllm::ltx2_vae::kLtx2VaePadZeros:
      *zero = true;
      return 0;
    case vllm::ltx2_vae::kLtx2VaePadReflect:
      return ReflectIndex(index, size);
    case vllm::ltx2_vae::kLtx2VaePadReplicate:
      return std::max<int64_t>(0, std::min<int64_t>(size - 1, index));
    default:
      break;
  }
  *zero = true;
  return 0;
}

void Pad(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h, int64_t w,
         int64_t pad_t_leading, int64_t pad_t_trailing, int64_t pad_hw, int mode, DType dtype) {
  RequireF32(dtype, "pad");
  float* out = static_cast<float*>(ov);
  const float* x = static_cast<const float*>(xv);
  const int64_t pt = t + pad_t_leading + pad_t_trailing;
  const int64_t ph = h + 2 * pad_hw;
  const int64_t pw = w + 2 * pad_hw;
  std::fill(out, out + channels * pt * ph * pw, 0.0f);
  // One row is one padded line (c, ti, hi). A pure GATHER — one source element
  // per destination element, no reduction — so the partition cannot change any
  // arithmetic. Parallel for the Amdahl reason #1009 records.
  ParallelForRows(CurrentThreadpool(), channels * pt * ph, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t hi = r % ph;
      const int64_t ti = (r / ph) % pt;
      const int64_t c = r / (ph * pt);
      // Temporal padding REPLICATES the edge frame, never zeros.
      const int64_t st = std::max<int64_t>(0, std::min<int64_t>(t - 1, ti - pad_t_leading));
      bool zero_h = false;
      const int64_t sh = SpatialIndex(hi - pad_hw, h, mode, &zero_h);
      for (int64_t wi = 0; wi < pw; ++wi) {
        bool zero_w = false;
        const int64_t sw = SpatialIndex(wi - pad_hw, w, mode, &zero_w);
        if (zero_h || zero_w) continue;
        out[((c * pt + ti) * ph + hi) * pw + wi] = x[((c * t + st) * h + sh) * w + sw];
      }
    }
  });
}

const vllm::ltx2_vae::Ltx2VaeDeviceKernels kKernels{
    &PixelNorm, &GroupNorm,     &AdaLn,     &SpatialNoise, &DepthToSpace, &FrameSlice,
    &ChannelRepeat, &LinearCn,  &Unpatchify, &Pad,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLtx2Vae, DeviceType::kCPU,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
