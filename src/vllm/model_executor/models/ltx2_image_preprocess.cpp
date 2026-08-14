// LTX-2.5 image conditioning input. Row LTX25-IMAGE-COND, issue #644.
// See ltx2_image_preprocess.h for the upstream anchors and the residuals.
#include "vllm/model_executor/models/ltx2_image_preprocess.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) { throw std::runtime_error(why); }

// ── PyTorch's `align_corners=False` index map, in f32 ───────────────────────
// aten/src/ATen/native/UpSample.h `area_pixel_compute_scale` /
// `area_pixel_compute_source_index`, and
// aten/src/ATen/native/cpu/UpSampleKernel.cpp
// `HelperInterpLinear::compute_indices_weights` + `guard_index_and_lambda`.
//
// The arithmetic is f32 because that is the `opmath_t` those helpers are
// dispatched at for a float tensor. Computing it in double here would be MORE
// accurate and would still be wrong: the goldens come from torch, so this port
// has to reproduce torch's rounding, not improve on it.
struct LinearTap {
  int64_t lo = 0, hi = 0;
  float w_lo = 1.0F, w_hi = 0.0F;
};

std::vector<LinearTap> LinearTaps(int64_t src, int64_t dst) {
  const float scale = static_cast<float>(src) / static_cast<float>(dst);
  std::vector<LinearTap> taps(static_cast<size_t>(dst));
  for (int64_t i = 0; i < dst; ++i) {
    float real = scale * (static_cast<float>(i) + 0.5F) - 0.5F;
    if (real < 0.0F) real = 0.0F;  // the `!cubic && src_idx < 0` clamp
    int64_t index = std::min(static_cast<int64_t>(std::floor(real)), src - 1);
    float lambda = std::min(std::max(real - static_cast<float>(index), 0.0F), 1.0F);
    LinearTap& tap = taps[static_cast<size_t>(i)];
    tap.lo = index;
    tap.hi = std::min(index + 1, src - 1);
    tap.w_lo = 1.0F - lambda;
    tap.w_hi = lambda;
  }
  return taps;
}

int NextPpmInt(std::istringstream& in, const std::string& field) {
  // PPM comments (`#` to end of line) may appear between any two header tokens.
  while (true) {
    in >> std::ws;
    if (in.peek() != '#') break;
    std::string skip;
    std::getline(in, skip);
  }
  int value = 0;
  if (!(in >> value)) Fail(field + ": bad PPM header");
  return value;
}

}  // namespace

std::vector<uint8_t> Ltx2DecodePpmRgb(const std::string& field, const std::string& bytes,
                                      int64_t* out_height, int64_t* out_width) {
  std::istringstream in(bytes, std::ios::binary);
  std::string magic;
  in >> magic;
  if (magic != "P6") {
    Fail(field +
         ": not a binary PPM (P6). No PNG/JPEG/EXR codec is vendored in this tree, so an image "
         "conditioning must be supplied as binary PPM — the same residual the MiniMax-H3 video "
         "seam carries (minimax_h3_video.cpp:84-86). Upstream's own decoder is PIL's "
         "(media_io/decode.py:139-170) and reads every format PIL does.");
  }
  const int width = NextPpmInt(in, field);
  const int height = NextPpmInt(in, field);
  const int maxval = NextPpmInt(in, field);
  if (width <= 0 || height <= 0) {
    Fail(field + ": PPM declares a " + std::to_string(width) + "x" + std::to_string(height) +
         " image");
  }
  if (maxval != 255) {
    // PIL rescales a non-255 maxval on the way to uint8; which rounding it uses
    // is a property of PIL's PPM plugin, not of LTX. Refusing rather than
    // inventing one — a rescale nobody checked against PIL would shift every
    // pixel of a conditioning image by a fraction no gate here can see.
    Fail(field + ": PPM maxval is " + std::to_string(maxval) +
         "; only 255 is read. `decode_image` (media_io/decode.py:139-170) returns "
         "`np.array(image, dtype=np.uint8)` out of PIL, and mirroring PIL's rescale for a "
         "narrower or wider maxval is a separate port. Re-save the image at maxval 255.");
  }
  in.get();  // the single whitespace byte between the header and the payload
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!in) Fail(field + ": truncated PPM payload");
  if (out_height != nullptr) *out_height = height;
  if (out_width != nullptr) *out_width = width;
  return rgb;
}

void Ltx2PreprocessImageCrf(int64_t crf) {
  if (crf == 0) return;  // decode.py:425-426 — `if crf == 0: return image`
  Fail(
      "image conditioning at CRF " + std::to_string(crf) +
      " is not ported: the H.264 round trip `preprocess` performs at a non-zero CRF "
      "(media_io/decode.py:430-434 -> encode_single_frame:386-400, libx264 preset=veryfast, "
      "rgb24 -> yuv420p, dimensions truncated to even, then decode_single_frame:403-410) needs a "
      "codec, and none is vendored in this tree. CRF 0 IS supported and is the only supported "
      "value: upstream short-circuits it at decode.py:425-426 and documents an explicit 0 as "
      "\"skip re-compression entirely\" (utils/args.py:58-59). Say so explicitly — an LTX-2.5 "
      "checkpoint RESOLVES 18 when the caller leaves the CRF unset (ImageConditioner.resolve_crf, "
      "blocks.py:977-983, over constants.py:37/124/130-133), so CRF 0 conditions on pixels this "
      "model generation was not trained against. That is a quality cost, not a correctness one, "
      "and it is stated rather than rendered silently.");
}

std::vector<float> Ltx2ResizeAndCenterCrop(const float* hwc, int64_t src_height,
                                           int64_t src_width, int64_t channels, int64_t height,
                                           int64_t width) {
  if (hwc == nullptr) Fail("ltx2 resize: null input");
  if (src_height <= 0 || src_width <= 0 || channels <= 0 || height <= 0 || width <= 0) {
    Fail("ltx2 resize: every dimension must be positive");
  }

  // resize.py:60-63. `scale = max(...)` is the aspect-FILL choice — the crop
  // happens after, so the SHORT side is what the target must be covered by.
  // `ceil` is upstream's own guard against a float rounding that would make
  // new_h/new_w land just under the target and give a negative crop offset.
  const double scale = std::max(static_cast<double>(height) / static_cast<double>(src_height),
                                static_cast<double>(width) / static_cast<double>(src_width));
  const int64_t new_h = static_cast<int64_t>(std::ceil(static_cast<double>(src_height) * scale));
  const int64_t new_w = static_cast<int64_t>(std::ceil(static_cast<double>(src_width) * scale));
  if (new_h < height || new_w < width) {
    Fail("ltx2 resize: the aspect-fill resize produced " + std::to_string(new_h) + "x" +
         std::to_string(new_w) + ", which cannot be cropped to " + std::to_string(height) + "x" +
         std::to_string(width));
  }

  const std::vector<LinearTap> rows = LinearTaps(src_height, new_h);
  const std::vector<LinearTap> cols = LinearTaps(src_width, new_w);

  // resize.py:71 — `crop_top = (new_h - height) // 2`, floor division.
  const int64_t crop_top = (new_h - height) / 2;
  const int64_t crop_left = (new_w - width) / 2;

  std::vector<float> out(static_cast<size_t>(channels * height * width));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t y = 0; y < height; ++y) {
      const LinearTap& r = rows[static_cast<size_t>(crop_top + y)];
      for (int64_t x = 0; x < width; ++x) {
        const LinearTap& k = cols[static_cast<size_t>(crop_left + x)];
        // WIDTH is the inner sum and HEIGHT the outer one, and each partial sum
        // is rounded to f32 before the next multiply — `Interpolate<n>::eval`
        // recurses over the dimensions in the order
        // `upsample_generic_Nd_kernel_impl` appends them (height, then width),
        // so the row combination happens first. Folding this into one
        // four-term dot product gives a different last bit.
        auto at = [&](int64_t yy, int64_t xx) {
          return hwc[static_cast<size_t>((yy * src_width + xx) * channels + c)];
        };
        const float lo = at(r.lo, k.lo) * k.w_lo + at(r.lo, k.hi) * k.w_hi;
        const float hi = at(r.hi, k.lo) * k.w_lo + at(r.hi, k.hi) * k.w_hi;
        out[static_cast<size_t>((c * height + y) * width + x)] = lo * r.w_lo + hi * r.w_hi;
      }
    }
  }
  return out;
}

std::vector<float> Ltx2LoadImageAndPreprocess(const std::string& field, const std::string& bytes,
                                              int64_t height, int64_t width, int64_t crf) {
  // decode.py:74-78, step for step and IN THIS ORDER.
  int64_t src_h = 0, src_w = 0;
  const std::vector<uint8_t> rgb = Ltx2DecodePpmRgb(field, bytes, &src_h, &src_w);  // :74
  Ltx2PreprocessImageCrf(crf);                                                      // :75

  // :76 — `torch.tensor(image, dtype=torch.float32)`. Values are still 0..255.
  std::vector<float> as_float(rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i) as_float[i] = static_cast<float>(rgb[i]);

  // :77 — the resize runs in 0..255 space.
  std::vector<float> chw = Ltx2ResizeAndCenterCrop(as_float.data(), src_h, src_w, 3, height, width);

  // :78 — `normalize_images` (range_map.py:8-9), AFTER the resize.
  for (float& v : chw) v = v / 127.5F - 1.0F;
  return chw;
}

int64_t Ltx2ResolveDefaultImageCrf(const std::vector<int64_t>& version_components) {
  // `_PARAMS_SINCE_VERSION` (constants.py:130-133), newest first, and
  // `detect_params`'s "the newest generation this version is at or above"
  // (:166-177). Only `default_image_crf` differs between the rows, so only that
  // is resolved here; a row that moved another knob would have to grow this.
  struct Row {
    std::vector<int64_t> since;
    int64_t crf;
  };
  static const std::vector<Row> kRows = {
      {{2, 4}, 18},  // LTX_2_4_PARAMS -> LTX_2_4_IMAGE_CRF (:37, :124)
      {{2, 3}, 33},  // LTX_2_3_PARAMS inherits DEFAULT_IMAGE_CRF (:36, :83-88)
  };
  for (const Row& row : kRows) {
    // Tuple comparison, which is what `parsed >= since` is in Python: element by
    // element, and a SHORTER tuple compares below a longer one that agrees on
    // the shared prefix. An empty `parsed` therefore falls through every row,
    // which is `detect_model_version`'s documented "compares below every real
    // version" (:138-139).
    if (!std::lexicographical_compare(version_components.begin(), version_components.end(),
                                      row.since.begin(), row.since.end())) {
      return row.crf;
    }
  }
  return 33;  // DEFAULT_IMAGE_CRF, via LTX_2_PARAMS (:36, :48, :80)
}

}  // namespace vllm
