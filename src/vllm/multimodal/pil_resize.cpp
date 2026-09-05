// Pillow's BICUBIC resize. See the header for the provenance and for the two
// properties that separate it from a textbook four-tap cubic.
#include "vllm/multimodal/pil_resize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vllm::multimodal {

namespace {

// `bicubic_filter` (Resample.c, `#define a -0.5`, `support = 2.0`). The
// coefficient is upstream's own literal and not a tuning knob: PIL's BICUBIC is
// the Keys kernel AT a = -0.5, and every other value is a different filter.
double BicubicFilter(double x) {
  constexpr double kA = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((kA + 2.0) * x - (kA + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * kA;
  return 0.0;
}

// `#define PRECISION_BITS (32 - 8 - 2)`: 8 bits of result plus two for the
// overshoot a filter with negative lobes produces on both sides of the range.
constexpr int kPrecisionBits = 22;
constexpr int64_t kOne = static_cast<int64_t>(1) << kPrecisionBits;
constexpr int64_t kHalf = static_cast<int64_t>(1) << (kPrecisionBits - 1);

// One axis of `precompute_coeffs`. `kk` is `out * ksize` INTEGER coefficients
// already through `normalize_coeffs_8bpc`; `bounds` is `out` pairs of
// (first source index, count).
struct Coeffs {
  int ksize = 0;
  std::vector<int64_t> kk;
  std::vector<int64_t> bounds;
};

Coeffs PrecomputeCoeffs(int64_t in_size, int64_t out_size) {
  Coeffs c;
  // `filterscale = scale = (in1 - in0) / outSize` with the default box
  // (in0 = 0, in1 = inSize), then `filterscale = max(1.0, filterscale)`. THIS
  // IS THE DOWNSCALE ARM: above 1.0 the support widens and the filter turns
  // into a weighted area average. Dropping it leaves a four-tap cubic that
  // aliases, and no shape check or token count can see the difference.
  const double scale =
      static_cast<double>(in_size) / static_cast<double>(out_size);
  const double filterscale = std::max(1.0, scale);
  const double support = 2.0 * filterscale;  // `filterp->support * filterscale`
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.kk.assign(static_cast<size_t>(out_size) * static_cast<size_t>(c.ksize), 0);
  c.bounds.assign(static_cast<size_t>(out_size) * 2, 0);

  std::vector<double> k(static_cast<size_t>(c.ksize), 0.0);
  const double ss = 1.0 / filterscale;
  for (int64_t xx = 0; xx < out_size; ++xx) {
    // The SAMPLE CENTRE. Both halves of the `+ 0.5` here and in the filter
    // argument below are load-bearing: dropping either shifts the whole image
    // by half an output pixel while leaving every shape valid and the mean
    // error small.
    const double center = (static_cast<double>(xx) + 0.5) * scale;
    // `(int)` truncates toward zero. Below zero the clamp makes truncation and
    // floor agree, and `center + support + 0.5` is never negative, so this is
    // faithful to upstream rather than a simplification of it.
    int64_t xmin = static_cast<int64_t>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int64_t xmax = static_cast<int64_t>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;  // upstream stores a COUNT, not an end index
    if (xmax < 0) xmax = 0;

    double ww = 0.0;
    for (int64_t x = 0; x < xmax; ++x) {
      const double w = BicubicFilter(
          (static_cast<double>(x + xmin) - center + 0.5) * ss);
      k[static_cast<size_t>(x)] = w;
      ww += w;
    }
    // PER-OUTPUT NORMALIZATION. This is what keeps a clipped edge window and a
    // widened downscale window from changing the brightness; without it the
    // error is a near-uniform scale, which a relative tolerance absorbs.
    for (int64_t x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[static_cast<size_t>(x)] /= ww;
    }
    for (int64_t x = xmax; x < c.ksize; ++x) k[static_cast<size_t>(x)] = 0.0;

    // `normalize_coeffs_8bpc`: round half AWAY FROM ZERO into 22-bit fixed
    // point. `(int)(0.5 + v)` and `(int)(-0.5 + v)` both truncate toward zero,
    // which is what makes the two branches one rounding rule.
    int64_t* dst = &c.kk[static_cast<size_t>(xx) * static_cast<size_t>(c.ksize)];
    for (int i = 0; i < c.ksize; ++i) {
      const double v = k[static_cast<size_t>(i)] * static_cast<double>(kOne);
      dst[i] = static_cast<int64_t>(v < 0 ? v - 0.5 : v + 0.5);
    }
    c.bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return c;
}

// `clip8`: an ARITHMETIC right shift (floor division by 2^22) into a saturating
// 0..255 lookup. The `1 << (PRECISION_BITS - 1)` the caller seeds the
// accumulator with is the rounding half, so this is round-half-up with
// saturation at both ends.
uint8_t Clip8(int64_t acc) {
  const int64_t v = acc >> kPrecisionBits;
  if (v <= 0) return 0;
  if (v >= 255) return 255;
  return static_cast<uint8_t>(v);
}

}  // namespace

std::vector<uint8_t> PilResizeBicubicRgb(const uint8_t* src, int64_t in_h,
                                         int64_t in_w, int64_t out_h,
                                         int64_t out_w) {
  if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
    throw std::runtime_error(
        "PilResizeBicubicRgb: every extent must be positive, got " +
        std::to_string(in_w) + "x" + std::to_string(in_h) + " -> " +
        std::to_string(out_w) + "x" + std::to_string(out_h));
  }
  const size_t in_bytes =
      static_cast<size_t>(in_h) * static_cast<size_t>(in_w) * 3;
  if (out_h == in_h && out_w == in_w) {
    // `Image.resize`'s own `size == self.size` short circuit: a COPY, not a
    // filtered identity. (The filtered identity agrees byte for byte here —
    // the weights collapse to a single 1.0 — but upstream does not run it and
    // neither does this.)
    return std::vector<uint8_t>(src, src + in_bytes);
  }

  // `ImagingResampleInner`: horizontal pass into a uint8 intermediate, then
  // vertical over THAT. The intermediate is quantized, so the answer is not the
  // separable float computation rounded once, and each pass is skipped when its
  // axis does not change size.
  //
  // ONE DEVIATION, and it is an identity: upstream computes the horizontal pass
  // only over the source rows the vertical pass will read (`ybox_first ..
  // ybox_last`) and shifts `bounds_vert` to match. The horizontal pass reads
  // exactly one source row per output row, so computing all `in_h` rows and
  // indexing absolutely produces the same bytes on every row either version
  // computes. The deviation costs work, never a value.
  const bool need_horizontal = out_w != in_w;
  const bool need_vertical = out_h != in_h;

  std::vector<uint8_t> mid;
  const uint8_t* stage = src;
  int64_t stage_w = in_w;
  if (need_horizontal) {
    const Coeffs c = PrecomputeCoeffs(in_w, out_w);
    mid.assign(static_cast<size_t>(in_h) * static_cast<size_t>(out_w) * 3, 0);
    for (int64_t yy = 0; yy < in_h; ++yy) {
      const uint8_t* row = src + static_cast<size_t>(yy) *
                                     static_cast<size_t>(in_w) * 3;
      uint8_t* orow =
          mid.data() + static_cast<size_t>(yy) * static_cast<size_t>(out_w) * 3;
      for (int64_t xx = 0; xx < out_w; ++xx) {
        const int64_t xmin = c.bounds[static_cast<size_t>(xx) * 2 + 0];
        const int64_t xmax = c.bounds[static_cast<size_t>(xx) * 2 + 1];
        const int64_t* k =
            &c.kk[static_cast<size_t>(xx) * static_cast<size_t>(c.ksize)];
        // Seeded with the rounding half, exactly as upstream does. The
        // accumulator is int64 where upstream uses int, and the widening is
        // safe because `sum |k|` is bounded even though the weights are only
        // normalized to sum ONE. 1.25 is the four-tap phase-0.5 value
        // (`2*0.5625 + 2*0.0625`) and it is NOT the bound: once
        // `filterscale > 1` the raw weights no longer sum to 1, and dividing a
        // truncated window by a small row sum amplifies its magnitudes past
        // that. Swept over every `(in, out)` in `1..800 x 1..800` (640,000
        // pairs): the supremum is 1.268771, at `in = 208`, `out = 193`, output
        // index 96 — an INTERIOR window — with normalized taps
        // (-0.067193, +0.567193, +0.567193, -0.067193). At `sum |k| <= 1.27`,
        // `|acc| <= 255 * 1.27 * 2^22 + 2^21`, about 1.36e9: inside int32's
        // 2.15e9, and inside PIL's own `clip8` table, which covers `acc >> 22`
        // over `[-640, 639]` against a worst index of 324. The wider type
        // removes an overflow that cannot occur; it does not change an answer.
        int64_t s0 = kHalf, s1 = kHalf, s2 = kHalf;
        for (int64_t x = 0; x < xmax; ++x) {
          const uint8_t* p = row + static_cast<size_t>(x + xmin) * 3;
          s0 += static_cast<int64_t>(p[0]) * k[x];
          s1 += static_cast<int64_t>(p[1]) * k[x];
          s2 += static_cast<int64_t>(p[2]) * k[x];
        }
        uint8_t* o = orow + static_cast<size_t>(xx) * 3;
        o[0] = Clip8(s0);
        o[1] = Clip8(s1);
        o[2] = Clip8(s2);
      }
    }
    stage = mid.data();
    stage_w = out_w;
  }

  if (!need_vertical) {
    // Horizontal only: the intermediate IS the answer.
    return mid;
  }

  const Coeffs c = PrecomputeCoeffs(in_h, out_h);
  std::vector<uint8_t> out(
      static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * 3, 0);
  for (int64_t yy = 0; yy < out_h; ++yy) {
    const int64_t ymin = c.bounds[static_cast<size_t>(yy) * 2 + 0];
    const int64_t ymax = c.bounds[static_cast<size_t>(yy) * 2 + 1];
    const int64_t* k =
        &c.kk[static_cast<size_t>(yy) * static_cast<size_t>(c.ksize)];
    uint8_t* orow =
        out.data() + static_cast<size_t>(yy) * static_cast<size_t>(out_w) * 3;
    for (int64_t xx = 0; xx < out_w; ++xx) {
      int64_t s0 = kHalf, s1 = kHalf, s2 = kHalf;
      for (int64_t y = 0; y < ymax; ++y) {
        const uint8_t* p = stage + (static_cast<size_t>(y + ymin) *
                                        static_cast<size_t>(stage_w) +
                                    static_cast<size_t>(xx)) *
                                       3;
        s0 += static_cast<int64_t>(p[0]) * k[y];
        s1 += static_cast<int64_t>(p[1]) * k[y];
        s2 += static_cast<int64_t>(p[2]) * k[y];
      }
      uint8_t* o = orow + static_cast<size_t>(xx) * 3;
      o[0] = Clip8(s0);
      o[1] = Clip8(s1);
      o[2] = Clip8(s2);
    }
  }
  return out;
}

}  // namespace vllm::multimodal
