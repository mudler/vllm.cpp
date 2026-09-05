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
// ─── THE BF16 ARM (A24 wave 3, row LTX25-A24-VIDEO-VAE-BF16, issue #2786) ───
//
// Upstream constructs the decoder in the pipeline's ONE dtype
// (Lightricks/LTX-2 @ fd4ded7f, ltx-pipelines/.../distilled.py:109, handed to
// `VideoDecoder` at :148), so every kernel here now serves BOTH widths behind the
// `DType` parameter each entry already carried. The f32 arm is byte-for-byte what
// it was -- it is the parity arm every committed
// tests/vllm/models/ltx2_vae_goldens.inc entry is measured against, and the bf16
// branches are additive.
//
// EVERY ROUNDING POINT BELOW WAS EXECUTED AGAINST UPSTREAM, not read off the
// source. torch's eager bf16 ops round at each expression boundary and widen
// INSIDE a reduction, and the two are different statements: `torch.mean` on bf16
// squares accumulates wider than bf16 (a sequential bf16 accumulate is wrong on
// 20 of 32 values at C=32 and 28 of 32 at C=128) while `x**2`, `+ eps`, `sqrt`
// and `/` each round. So "a fully bf16 chain" names the rounding POINTS and never
// the accumulator width, and each kernel below says which of the two it is
// applying and what the alternative measured.
//
// Five of the ten entries -- `depth_to_space`, `frame_slice`, `channel_repeat`,
// `unpatchify` and `pad` -- are pure SHAPE MOVEMENT. They gather one source
// element per destination element and round nowhere, so their bf16 arm moves
// 16-bit WORDS and performs no conversion at all.
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
// this table with a width it does not serve gets the dtype and the op back rather
// than a reinterpreted buffer.
void RequireWidth(DType dtype, const char* what) {
  VT_CHECK(dtype == DType::kF32 || dtype == DType::kBF16,
           std::string("ltx2 vae ") + what +
               ": this arm serves f32 (the parity arm) and bf16 (upstream's own model dtype, "
               "distilled.py:109) storage. The FP8 and NVFP4 arms are A22 and are not "
               "implemented; it was handed " +
               Name(dtype));
}

// LOAD WIDENS, STORE ROUNDS -- the `LoadF32At`/`StoreF32At` contract every other
// CPU kernel in this tree uses, spelled locally because this header may not pull
// in a tensor type.
inline float Load(const void* base, int64_t i, bool bf16) {
  return bf16 ? BF16ToF32(static_cast<const uint16_t*>(base)[i])
              : static_cast<const float*>(base)[i];
}
inline void Store(void* base, int64_t i, float v, bool bf16) {
  if (bf16) {
    static_cast<uint16_t*>(base)[i] = F32ToBF16(v);
  } else {
    static_cast<float*>(base)[i] = v;
  }
}
// A MID-CHAIN rounding, which is a different thing from a store. `Store` puts a
// value in memory; this puts an intermediate back on the bf16 grid because
// upstream materialized a bf16 tensor at that point. At f32 it is the identity,
// which is what keeps the parity arm byte-identical.
inline float Round(float v, bool bf16) { return bf16 ? BF16ToF32(F32ToBF16(v)) : v; }

// The shape movers' element, which is a WORD and never a number: they gather and
// convert nothing, so the only thing their bf16 arm needs is the width.

// --- PixelNorm (ltx2_video_vae.cpp:381-400) ---------------------------------
void PixelNorm(Queue&, void* xv, int64_t channels, int64_t spatial, float eps, DType dtype) {
  RequireWidth(dtype, "pixel_norm");
  const bool bf16 = dtype == DType::kBF16;
  // THE EPSILON'S OWN WIDTH IS PART OF THE ARITHMETIC AND IT IS THE bf16 ONE.
  // `PixelNorm.forward` is raw eager ops on the activation tensor
  // (model/common/normalization.py:37-40), so `mean_sq + self.eps` promotes the
  // Python float to the TENSOR's dtype: what reaches the add is
  // `bf16(1e-8) = 1.0011717677116394e-08`, not `1e-8`.
  //
  // AND THAT SEPARATES ONLY AT 2^-14, which is why it is a comment here and a
  // dedicated probe in the goldens rather than something the shipped fixture
  // would catch. Holding the chain fixed and varying ONLY the width of this add,
  // against upstream on [2,32,2,8,8]: 0 of 8192 at row scales 2^-0, 2^-6, 2^-10
  // and 2^-12, and 842 of 8192 at 2^-14. Removing the epsilon entirely separates
  // from 2^-10 (5579 of 8192) -- a different question, and the reason the
  // generator lays both scales. A probe at ordinary magnitude is a mute switch
  // for both.
  const float eps_w = Round(eps, bf16);
  for (int64_t i = 0; i < spatial; ++i) {
    // THE ACCUMULATOR IS f32 ON BOTH ARMS AND THE SQUARE ROUNDS ON THE bf16 ONE.
    // `torch.mean(x**2, dim=1)` materializes `x**2` as a bf16 tensor and then
    // reduces it with a WIDER accumulator: measured against upstream, an f32
    // accumulate of bf16 squares is 0 of 32 at C=32 and C=128, an f64 accumulate
    // is also 0 of 32, and a sequential bf16 accumulate is 20 and 28 of 32. So
    // the width of the accumulator does not separate and is NOT gated here; the
    // rounding of the square does.
    float mean_sq = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      const float v = Load(xv, c * spatial + i, bf16);
      mean_sq += Round(v * v, bf16);
    }
    mean_sq = Round(mean_sq / static_cast<float>(channels), bf16);
    if (bf16) {
      // A DIVIDE PER CHANNEL, NOT ONE RECIPROCAL AND A MULTIPLY. Upstream is
      // `x / rms` (normalization.py:40), and at bf16 the two are not the same
      // number: the reciprocal rounds to the bf16 grid first and the product then
      // rounds again, where the divide rounds once. The f32 arm keeps the
      // reciprocal because that is what its committed goldens were taken through.
      const float rms = Round(std::sqrt(Round(mean_sq + eps_w, bf16)), bf16);
      for (int64_t c = 0; c < channels; ++c) {
        Store(xv, c * spatial + i, Round(Load(xv, c * spatial + i, true) / rms, true), true);
      }
      continue;
    }
    float* x = static_cast<float*>(xv);
    // ONE reciprocal per pixel, then a multiply per channel: the host loop's
    // shape, kept so the move changes nothing. NO GATE HOLDS IT -- a per-channel
    // divide leaves test_ltx2_vae fully green, because the difference is the last
    // ulp and the golden tolerance absorbs it. See the header.
    const float inv = 1.0f / std::sqrt(mean_sq + eps_w);
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
  RequireWidth(dtype, "group_norm");
  VT_CHECK(groups > 0 && channels % groups == 0,
           "ltx2 vae group_norm: channels must be divisible by num_groups");
  const bool bf16 = dtype == DType::kBF16;
  const int64_t per_group = channels / groups;
  // NOTHING IN THE ARITHMETIC CHANGES AT bf16, AND THAT IS A MEASUREMENT.
  // Upstream's `nn.GroupNorm` at bf16 differs from the f32 module rounded once on
  // 463 of 1600 values (C=32, G=4), 6664 of 24576 (C=128, G=32) and 461 of 1600
  // (G=1, which is `norm3`) -- and the ENTIRE difference is that the module's
  // `weight` and `bias` are bf16. With the affine narrowed and the statistics
  // left in f64, this loop is 0 of 1600, 0 of 24576 and 0 of 1600 against it. The
  // statistics' own width does not separate at all: f32 and f64 agree everywhere.
  //
  // So the affine narrowing does NOT happen here -- it happens at LOAD, because
  // the weights bag itself is bf16 (`Ltx2LoadVaeWeights` at `kBF16`), and `Load`
  // below returns the already-narrowed value. Narrowing again here would round a
  // rounded number, which is a no-op, but writing it would hide where the
  // property actually comes from.
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t begin = g * per_group;
    const int64_t count = per_group * spatial;
    double mean = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) mean += Load(xv, c * spatial + i, bf16);
    }
    mean /= static_cast<double>(count);
    double var = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = Load(xv, c * spatial + i, bf16) - mean;
        var += d * d;
      }
    }
    var /= static_cast<double>(count);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = begin; c < begin + per_group; ++c) {
      const double w = Load(wv, c, bf16);
      const double b = Load(bv, c, bf16);
      for (int64_t i = 0; i < spatial; ++i) {
        const double normed = (Load(xv, c * spatial + i, bf16) - mean) * inv;
        Store(xv, c * spatial + i, static_cast<float>(normed * w + b), bf16);
      }
    }
  }
}

// --- ApplyAdaLn (ltx2_video_vae.cpp:529-556) --------------------------------
void AdaLn(Queue&, void* xv, const void* tv, const void* ev, int64_t channels, int64_t spatial,
           int64_t rows, int64_t shift_row, int64_t scale_row, DType dtype) {
  RequireWidth(dtype, "ada_ln");
  VT_CHECK(shift_row >= 0 && shift_row < rows && scale_row >= 0 && scale_row < rows,
           "ltx2 vae ada_ln: the shift/scale rows must lie inside the table");
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t ch = 0; ch < channels; ++ch) {
    // table + embed FIRST, then the affine — upstream's order (resnet.py:135-148).
    const float shift = Round(Load(tv, shift_row * channels + ch, bf16) +
                                  Load(ev, shift_row * channels + ch, bf16),
                              bf16);
    const float scale = Round(Load(tv, scale_row * channels + ch, bf16) +
                                  Load(ev, scale_row * channels + ch, bf16),
                              bf16);
    // THREE ROUNDINGS, NOT ONE. `hidden_states * (1 + scale1) + shift1`
    // (resnet.py:148, conv_video_decoder.py:347) is three eager ops on bf16
    // tensors, so `1 + scale` materializes, the multiply materializes and the add
    // materializes. Measured against upstream on 512 values: three roundings is
    // 0, an all-f32 expression rounded once is 198, and `(1+sc)` in bf16 followed
    // by a fused f32 mul-add is 162. Both alternatives separate.
    const float one_plus = Round(1.0f + scale, bf16);
    for (int64_t i = 0; i < spatial; ++i) {
      const float v = Load(xv, ch * spatial + i, bf16);
      Store(xv, ch * spatial + i, Round(Round(v * one_plus, bf16) + shift, bf16), bf16);
    }
  }
}

// --- FeedSpatialNoise (ltx2_video_vae.cpp:507-527) --------------------------
void SpatialNoise(Queue&, void* xv, const void* pv, const void* sv, int64_t channels, int64_t t,
                  int64_t h, int64_t w, DType dtype) {
  RequireWidth(dtype, "spatial_noise");
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t c = 0; c < channels; ++c) {
    const float s = Load(sv, c, bf16);
    for (int64_t ti = 0; ti < t; ++ti) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          // THE PRODUCT ROUNDS, THEN THE ADD ROUNDS. `_feed_spatial_noise` forms
          // `scaled_noise = spatial_noise * per_channel_scale` as its own tensor
          // and adds it (resnet.py:114-117), so it is two eager ops and not one
          // fused expression: measured 0 of 144 against upstream, where the fused
          // form is 8 of 144.
          const int64_t at = ((c * t + ti) * h + hi) * w + wi;
          const float prod = Round(Load(pv, hi * w + wi, bf16) * s, bf16);
          Store(xv, at, Round(Load(xv, at, bf16) + prod, bf16), bf16);
        }
      }
    }
  }
}

// --- DepthToSpaceUpsample's expand (ltx2_video_vae.cpp:605-631) -------------
template <typename W>
void DepthToSpaceT(void* ov, const void* xv, int64_t out_channels, int64_t t, int64_t h, int64_t w,
                   int64_t st, int64_t sh, int64_t sw) {
  W* out = static_cast<W*>(ov);
  const W* x = static_cast<const W*>(xv);
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

void DepthToSpace(Queue&, void* ov, const void* xv, int64_t out_channels, int64_t t, int64_t h,
                  int64_t w, int64_t st, int64_t sh, int64_t sw, DType dtype) {
  RequireWidth(dtype, "depth_to_space");
  if (dtype == DType::kBF16) {
    DepthToSpaceT<uint16_t>(ov, xv, out_channels, t, h, w, st, sh, sw);
  } else {
    DepthToSpaceT<float>(ov, xv, out_channels, t, h, w, st, sh, sw);
  }
}

// --- drop_first_frame (ltx2_video_vae.cpp:632-654) --------------------------
template <typename W>
void FrameSliceT(void* ov, const void* xv, int64_t channels, int64_t t, int64_t h, int64_t w,
                 int64_t drop) {
  W* out = static_cast<W*>(ov);
  const W* x = static_cast<const W*>(xv);
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

void FrameSlice(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                int64_t w, int64_t drop, DType dtype) {
  RequireWidth(dtype, "frame_slice");
  VT_CHECK(drop >= 0 && drop < t, "ltx2 vae frame_slice: the slice must leave a frame");
  if (dtype == DType::kBF16) {
    FrameSliceT<uint16_t>(ov, xv, channels, t, h, w, drop);
  } else {
    FrameSliceT<float>(ov, xv, channels, t, h, w, drop);
  }
}

// --- the residual's channel repeat (ltx2_video_vae.cpp:660-670) -------------
void ChannelRepeat(Queue&, void* ov, const void* xv, int64_t channels, int64_t spatial,
                   int64_t repeat, DType dtype) {
  RequireWidth(dtype, "channel_repeat");
  const int64_t block = channels * spatial;
  // torch's `repeat` TILES: the block index is the OUTER axis.
  if (dtype == DType::kBF16) {
    const uint16_t* x = static_cast<const uint16_t*>(xv);
    uint16_t* out = static_cast<uint16_t*>(ov);
    for (int64_t r = 0; r < repeat; ++r) std::copy(x, x + block, out + r * block);
    return;
  }
  const float* x = static_cast<const float*>(xv);
  float* out = static_cast<float*>(ov);
  for (int64_t r = 0; r < repeat; ++r) std::copy(x, x + block, out + r * block);
}

// --- Linear3d (ltx2_video_vae.cpp:339-368) ----------------------------------
void LinearCn(Queue&, void* ov, const void* xv, const void* wv, const void* bv,
              int64_t out_channels, int64_t in_channels, int64_t n, DType dtype) {
  RequireWidth(dtype, "linear_cn");
  const bool bf16 = dtype == DType::kBF16;
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
      //
      // AND IT IS BIT-EXACT AT bf16, measured: a 1x1x1 reduction has no
      // association order to differ in, so this loop is 0 of 4096 against torch's
      // own bf16 `F.conv3d` over four seeds, where the 3x3x3 case leaves 3 of
      // 18816. Seeding with the bias is what the width buys: adding the bias
      // AFTER the store rounding is 5211 of 18816 wrong on the 3x3x3 case.
      float acc = Load(bv, oc, bf16);
      for (int64_t ic = 0; ic < in_channels; ++ic) {
        acc += Load(xv, ic * n + i, bf16) * Load(wv, oc * in_channels + ic, bf16);
      }
      Store(ov, oc * n + i, acc, bf16);
    }
  });
}

// --- unpatchify (ltx2_video_vae.cpp:945-970) --------------------------------
template <typename W>
void UnpatchifyT(void* ov, const void* xv, int64_t channels, int64_t t, int64_t h, int64_t w,
                 int64_t q, int64_t r_stride) {
  W* out = static_cast<W*>(ov);
  const W* x = static_cast<const W*>(xv);
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

void Unpatchify(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h,
                int64_t w, int64_t q, int64_t r_stride, DType dtype) {
  RequireWidth(dtype, "unpatchify");
  if (dtype == DType::kBF16) {
    UnpatchifyT<uint16_t>(ov, xv, channels, t, h, w, q, r_stride);
  } else {
    UnpatchifyT<float>(ov, xv, channels, t, h, w, q, r_stride);
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

template <typename W>
void PadT(void* ov, const void* xv, int64_t channels, int64_t t, int64_t h, int64_t w,
          int64_t pad_t_leading, int64_t pad_t_trailing, int64_t pad_hw, int mode) {
  W* out = static_cast<W*>(ov);
  const W* x = static_cast<const W*>(xv);
  const int64_t pt = t + pad_t_leading + pad_t_trailing;
  const int64_t ph = h + 2 * pad_hw;
  const int64_t pw = w + 2 * pad_hw;
  // ZERO IS ZERO IN BOTH FORMATS: the bf16 encoding of +0.0 is the all-zero word,
  // so the kZeros pad writes the same bits either way and needs no conversion.
  std::fill(out, out + channels * pt * ph * pw, W{});
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

void Pad(Queue&, void* ov, const void* xv, int64_t channels, int64_t t, int64_t h, int64_t w,
         int64_t pad_t_leading, int64_t pad_t_trailing, int64_t pad_hw, int mode, DType dtype) {
  RequireWidth(dtype, "pad");
  if (dtype == DType::kBF16) {
    PadT<uint16_t>(ov, xv, channels, t, h, w, pad_t_leading, pad_t_trailing, pad_hw, mode);
  } else {
    PadT<float>(ov, xv, channels, t, h, w, pad_t_leading, pad_t_trailing, pad_hw, mode);
  }
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
