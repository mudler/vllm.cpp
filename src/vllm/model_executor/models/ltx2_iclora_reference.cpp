// LTX-2.5 IC-LoRA reference conditioning — see ltx2_iclora_reference.h for the
// port map and for the four failure modes this file exists to make impossible.
//
// Row LTX25-IC-LORA-REF-VIDEO (#3020). Upstream: Lightricks/LTX-2 @ fd4ded7f.
#include "vllm/model_executor/models/ltx2_iclora_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("ltx2 ic-lora reference: " + why);
}

// `torch.nn.functional.interpolate(..., mode="area")` dispatches to
// `adaptive_avg_pool2d`, whose window for output index `i` over input extent `I`
// and output extent `O` is `[floor(i*I/O), ceil((i+1)*I/O))`. That is NOT the
// same as an integer-stride box filter whenever `O` does not divide `I`, and the
// difference is invisible on the divisible shapes a reduced fixture reaches — so
// the general form is written here rather than the special case.
int64_t PoolStart(int64_t index, int64_t in_extent, int64_t out_extent) {
  return (index * in_extent) / out_extent;
}
int64_t PoolEnd(int64_t index, int64_t in_extent, int64_t out_extent) {
  return ((index + 1) * in_extent + out_extent - 1) / out_extent;
}

}  // namespace

Ltx2IcLoraReferenceGeometry Ltx2ResolveIcLoraReferenceGeometry(int64_t height, int64_t width,
                                                               int64_t downscale_factor) {
  if (downscale_factor < 1) {
    Fail("reference_downscale_factor must be at least 1, got " +
         std::to_string(downscale_factor) +
         ". Upstream's absent-metadata default is 1 (iclora_utils.py:35)");
  }
  // `:112` — the guard is `scale != 1 AND (h % scale or w % scale)`. At scale 1
  // NOTHING is ever refused, which is what keeps every adapter-less render
  // working, and a port that dropped the `scale != 1` half would refuse none of
  // them differently (1 divides everything) while a port that dropped the
  // divisibility half would refuse all of them.
  if (downscale_factor != 1 && (height % downscale_factor != 0 || width % downscale_factor != 0)) {
    Fail("Output dimensions (" + std::to_string(height) + "x" + std::to_string(width) +
         ") must be divisible by reference_downscale_factor (" +
         std::to_string(downscale_factor) +
         "). This is upstream's own refusal (iclora_utils.py:112-115): the reference clip is read "
         "at height // factor by width // factor and a truncating division would place the "
         "reference tokens on a grid the target does not have");
  }
  Ltx2IcLoraReferenceGeometry out;
  out.height = height / downscale_factor;  // :116
  out.width = width / downscale_factor;    // :117
  return out;
}

std::vector<int64_t> Ltx2TemporalSubsampleIndices(int64_t frames, int64_t temporal_scale_factor) {
  if (frames < 1) Fail("a reference clip needs at least one frame");
  if (temporal_scale_factor < 1) {
    Fail("reference_temporal_scale_factor must be at least 1, got " +
         std::to_string(temporal_scale_factor));
  }
  // `indices = [0, *list(range(1, video.shape[2], temporal_scale_factor))]` (:89).
  //
  // INDEX 1 IS ALWAYS KEPT when it exists, because the range starts AT 1 rather
  // than stepping from 0. At factor 2 over 5 frames this is {0, 1, 3}; the
  // plausible `range(0, F, N)` gives {0, 2, 4} — the same COUNT, so no shape
  // check and no token count can tell them apart, and the reference simply
  // describes different moments of the clip. Gated by `kLtx2TemporalSubsampleKept`
  // against exactly that hypothesis.
  std::vector<int64_t> indices;
  indices.push_back(0);
  for (int64_t i = 1; i < frames; i += temporal_scale_factor) indices.push_back(i);
  return indices;
}

std::vector<float> Ltx2TemporalSubsample(const std::vector<float>& clip, int64_t channels,
                                         int64_t frames, int64_t plane,
                                         int64_t temporal_scale_factor) {
  const std::vector<int64_t> keep = Ltx2TemporalSubsampleIndices(frames, temporal_scale_factor);
  const size_t expect = static_cast<size_t>(channels) * static_cast<size_t>(frames) *
                        static_cast<size_t>(plane);
  if (clip.size() != expect) {
    Fail("the reference clip holds " + std::to_string(clip.size()) + " values but " +
         std::to_string(channels) + " x " + std::to_string(frames) + " x " +
         std::to_string(plane) + " is " + std::to_string(expect));
  }
  const int64_t kept = static_cast<int64_t>(keep.size());
  std::vector<float> out(static_cast<size_t>(channels) * static_cast<size_t>(kept) *
                         static_cast<size_t>(plane));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < kept; ++t) {
      const size_t src = (static_cast<size_t>(c) * static_cast<size_t>(frames) +
                          static_cast<size_t>(keep[static_cast<size_t>(t)])) *
                         static_cast<size_t>(plane);
      const size_t dst = (static_cast<size_t>(c) * static_cast<size_t>(kept) +
                          static_cast<size_t>(t)) *
                         static_cast<size_t>(plane);
      std::copy(clip.begin() + static_cast<std::ptrdiff_t>(src),
                clip.begin() + static_cast<std::ptrdiff_t>(src + static_cast<size_t>(plane)),
                out.begin() + static_cast<std::ptrdiff_t>(dst));
    }
  }
  return out;
}

std::vector<float> Ltx2MaskVideoFromPixels(const std::vector<float>& pixels, int64_t channels,
                                           int64_t frames, int64_t plane) {
  if (channels < 1) Fail("a mask video needs at least one channel");
  const size_t expect = static_cast<size_t>(channels) * static_cast<size_t>(frames) *
                        static_cast<size_t>(plane);
  if (pixels.size() != expect) {
    Fail("the mask video holds " + std::to_string(pixels.size()) + " values but " +
         std::to_string(channels) + " x " + std::to_string(frames) + " x " +
         std::to_string(plane) + " is " + std::to_string(expect));
  }
  std::vector<float> out(static_cast<size_t>(frames) * static_cast<size_t>(plane));
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t p = 0; p < plane; ++p) {
      // `mask_video.mean(dim=1, keepdim=True)` (:531). The pixels arrive
      // CHANNEL-major, which is the layout the encoder takes, so the three
      // samples of one pixel are a plane apart rather than adjacent.
      double sum = 0.0;
      for (int64_t c = 0; c < channels; ++c) {
        const size_t at = (static_cast<size_t>(c) * static_cast<size_t>(frames) +
                           static_cast<size_t>(t)) *
                              static_cast<size_t>(plane) +
                          static_cast<size_t>(p);
        sum += static_cast<double>(pixels[at]);
      }
      const double mean = sum / static_cast<double>(channels);
      // `(mask + 1.0) / 2.0` then `.clamp(0.0, 1.0)` (:534-536). The remap
      // undoes `normalize_images`' `/127.5 - 1`, and the clamp is upstream's and
      // not defensive: a mask frame that decoded above the range would otherwise
      // become an attention weight above 1, which `_prepare_self_attention_mask`
      // turns into a POSITIVE log-space bias — an amplifier, not an attenuator.
      out[static_cast<size_t>(t) * static_cast<size_t>(plane) + static_cast<size_t>(p)] =
          static_cast<float>(std::min(1.0, std::max(0.0, (mean + 1.0) / 2.0)));
    }
  }
  return out;
}

std::vector<float> Ltx2DownsampleMaskVideoToLatent(const std::vector<float>& mask, int64_t f_pix,
                                                   int64_t h_pix, int64_t w_pix, int64_t f_lat,
                                                   int64_t h_lat, int64_t w_lat) {
  if (f_pix < 1 || h_pix < 1 || w_pix < 1 || f_lat < 1 || h_lat < 1 || w_lat < 1) {
    Fail("every mask dimension must be positive");
  }
  const size_t expect =
      static_cast<size_t>(f_pix) * static_cast<size_t>(h_pix) * static_cast<size_t>(w_pix);
  if (mask.size() != expect) {
    Fail("the mask holds " + std::to_string(mask.size()) + " values but " +
         std::to_string(f_pix) + " x " + std::to_string(h_pix) + " x " + std::to_string(w_pix) +
         " is " + std::to_string(expect));
  }

  // :63-67 — AREA interpolation, per pixel frame, to the LATENT spatial grid.
  const size_t lat_plane = static_cast<size_t>(h_lat) * static_cast<size_t>(w_lat);
  std::vector<double> spatial(static_cast<size_t>(f_pix) * lat_plane);
  for (int64_t t = 0; t < f_pix; ++t) {
    for (int64_t oh = 0; oh < h_lat; ++oh) {
      const int64_t h0 = PoolStart(oh, h_pix, h_lat), h1 = PoolEnd(oh, h_pix, h_lat);
      for (int64_t ow = 0; ow < w_lat; ++ow) {
        const int64_t w0 = PoolStart(ow, w_pix, w_lat), w1 = PoolEnd(ow, w_pix, w_lat);
        double sum = 0.0;
        for (int64_t ih = h0; ih < h1; ++ih) {
          for (int64_t iw = w0; iw < w1; ++iw) {
            sum += static_cast<double>(
                mask[(static_cast<size_t>(t) * static_cast<size_t>(h_pix) +
                      static_cast<size_t>(ih)) *
                         static_cast<size_t>(w_pix) +
                     static_cast<size_t>(iw)]);
          }
        }
        spatial[static_cast<size_t>(t) * lat_plane + static_cast<size_t>(oh) *
                                                        static_cast<size_t>(w_lat) +
                static_cast<size_t>(ow)] = sum / static_cast<double>((h1 - h0) * (w1 - w0));
      }
    }
  }

  // :70 — the FIRST latent frame is pixel frame 0 ALONE. This is the causal
  // carve-out, and it is the whole content of the temporal half: pooling all
  // `f_pix` frames uniformly into `f_lat` groups produces a correctly shaped
  // mask of entirely plausible values. Gated against exactly that hypothesis by
  // `kLtx2MaskLatentUniformPool`.
  std::vector<float> out;
  out.reserve(static_cast<size_t>(f_lat) * lat_plane);
  for (size_t i = 0; i < lat_plane; ++i) {
    out.push_back(static_cast<float>(spatial[i]));
  }
  // :72 — and the rest ONLY when both axes have more than one frame; otherwise
  // the first frame alone is the whole answer (:81-82).
  if (f_pix > 1 && f_lat > 1) {
    const int64_t t = (f_pix - 1) / (f_lat - 1);  // :73
    if ((f_pix - 1) % (f_lat - 1) != 0) {         // :74-77, upstream's assertion
      Fail("Pixel frames (" + std::to_string(f_pix) + ") not compatible with latent frames (" +
           std::to_string(f_lat) + "): (f_pix - 1) must be divisible by (f_lat - 1). The mask "
           "video and the reference clip must describe the same moments, and a truncating "
           "group size would silently drop the tail of the mask");
    }
    for (int64_t lf = 1; lf < f_lat; ++lf) {
      for (size_t p = 0; p < lat_plane; ++p) {
        double sum = 0.0;
        for (int64_t k = 0; k < t; ++k) {
          // :78-79 — the remaining frames reshape to [f_lat - 1, t] and mean
          // over the group axis. Frame `1 + (lf - 1) * t + k`, because the
          // reshape happens AFTER frame 0 has been taken out.
          const int64_t src = 1 + (lf - 1) * t + k;
          sum += spatial[static_cast<size_t>(src) * lat_plane + p];
        }
        out.push_back(static_cast<float>(sum / static_cast<double>(t)));
      }
    }
  }
  return out;
}

std::vector<float> Ltx2ResolveCrossMask(const std::vector<float>& values, double scalar,
                                        int64_t num_new_tokens) {
  if (num_new_tokens < 1) Fail("a cross mask needs at least one new token");
  if (values.empty()) {
    // :31-37 — the scalar arm fills every new token with the same weight.
    return std::vector<float>(static_cast<size_t>(num_new_tokens), static_cast<float>(scalar));
  }
  // :49-53 — the 1-D `(M,)` form, whose length upstream refuses to broadcast.
  if (static_cast<int64_t>(values.size()) != num_new_tokens) {
    Fail("1-D attention_mask length must equal num_new_tokens (" +
         std::to_string(num_new_tokens) + "), got " + std::to_string(values.size()) +
         ". Upstream refuses rather than broadcasting (mask_utils.py:50-53), because a mask one "
         "token short of the sequence would attenuate the wrong tokens and still render");
  }
  return values;
}

std::vector<float> Ltx2BuildAttentionMask(const std::vector<float>& existing,
                                          int64_t num_noisy_tokens, int64_t num_new_tokens,
                                          int64_t num_existing_tokens,
                                          const std::vector<float>& cross_mask) {
  if (num_noisy_tokens < 1 || num_new_tokens < 1 || num_existing_tokens < num_noisy_tokens) {
    Fail("the attention mask's block structure needs 1 <= num_noisy <= num_existing and at least "
         "one new token; got noisy=" +
         std::to_string(num_noisy_tokens) + " new=" + std::to_string(num_new_tokens) +
         " existing=" + std::to_string(num_existing_tokens));
  }
  if (static_cast<int64_t>(cross_mask.size()) != num_new_tokens) {
    Fail("the cross mask must hold one weight per new token (" + std::to_string(num_new_tokens) +
         "), got " + std::to_string(cross_mask.size()));
  }
  const int64_t total = num_existing_tokens + num_new_tokens;  // :217
  if (!existing.empty() &&
      static_cast<int64_t>(existing.size()) != num_existing_tokens * num_existing_tokens) {
    Fail("the existing mask must be [" + std::to_string(num_existing_tokens) + ", " +
         std::to_string(num_existing_tokens) + "], got " + std::to_string(existing.size()) +
         " values");
  }

  std::vector<float> out(static_cast<size_t>(total) * static_cast<size_t>(total), 0.0F);  // :220
  const auto At = [&out, total](int64_t r, int64_t c) -> float& {
    return out[static_cast<size_t>(r) * static_cast<size_t>(total) + static_cast<size_t>(c)];
  };

  // :223-226 — the top-left N x N block is the existing mask, or ones when there
  // was none. NOT ones-plus-existing and not zeros: an absent mask means every
  // existing token attends to every other, INCLUDING prior reference tokens that
  // carried no mask of their own (`:204-205`).
  for (int64_t r = 0; r < num_existing_tokens; ++r) {
    for (int64_t c = 0; c < num_existing_tokens; ++c) {
      At(r, c) = existing.empty()
                     ? 1.0F
                     : existing[static_cast<size_t>(r) * static_cast<size_t>(num_existing_tokens) +
                                static_cast<size_t>(c)];
    }
  }
  // :229 — the new reference tokens fully attend to THEMSELVES.
  for (int64_t r = num_existing_tokens; r < total; ++r) {
    for (int64_t c = num_existing_tokens; c < total; ++c) At(r, c) = 1.0F;
  }
  // :236 — noisy rows attend to the new reference tokens at the cross weight,
  // one weight per COLUMN. `num_noisy_tokens` and NOT `num_existing_tokens`:
  // rows between the two are PRIOR reference tokens and they stay at the zero
  // `:242` leaves them at. On a fixture with no prior item the two readings are
  // elementwise equal, which is why the golden's fixture carries one.
  for (int64_t r = 0; r < num_noisy_tokens; ++r) {
    for (int64_t j = 0; j < num_new_tokens; ++j) {
      At(r, num_existing_tokens + j) = cross_mask[static_cast<size_t>(j)];
    }
  }
  // :240 — and the transpose: new reference ROWS attend to the noisy tokens, one
  // weight per ROW.
  for (int64_t i = 0; i < num_new_tokens; ++i) {
    for (int64_t c = 0; c < num_noisy_tokens; ++c) {
      At(num_existing_tokens + i, c) = cross_mask[static_cast<size_t>(i)];
    }
  }
  return out;
}

std::vector<float> Ltx2PadAttentionMaskForUnmaskedTokens(const std::vector<float>& existing,
                                                         int64_t num_noisy_tokens,
                                                         int64_t num_new_tokens,
                                                         int64_t num_existing_tokens) {
  // :142-143 — no mask requested and none present is upstream's `None`, and an
  // empty vector is how this engine spells it.
  if (existing.empty()) return {};
  // :147 — otherwise the new tokens get FULL attention, so the mask keeps
  // describing the whole sequence.
  const std::vector<float> ones(static_cast<size_t>(num_new_tokens), 1.0F);
  return Ltx2BuildAttentionMask(existing, num_noisy_tokens, num_new_tokens, num_existing_tokens,
                                ones);
}

}  // namespace vllm
