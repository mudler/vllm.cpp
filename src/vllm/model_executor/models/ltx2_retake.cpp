// LTX-2.5 RETAKE — see ltx2_retake.h for the port map and for the four failure
// modes this file exists to make impossible.
//
// Row LTX25-RETAKE (#924). Upstream: Lightricks/LTX-2 @ fd4ded7f.
#include "vllm/model_executor/models/ltx2_retake.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_image_preprocess.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) { throw std::runtime_error("ltx2 retake: " + why); }

// The layout the C ABI, `/v1/videos` and `minimax-h3-gen` already agree on
// (include/vllm.h:912, src/vllm/entrypoints/openai/video_api.cpp:115-121), so
// one run's frames chain into the next request.
std::string FramePath(const std::string& dir, int64_t index) {
  char name[32];
  std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(index));
  return dir + name;
}

// Empty when the file is absent, which is how the walk finds the end of the
// sequence — the same probe-until-missing shape the reference-clip reader in
// `minimax_h3_video.cpp:139-144` uses, because a directory listing would also
// have to sort and would accept a gap in the middle.
bool ReadWholeFile(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return true;
}

// `(t_end > start_time) & (t_start < end_time)` (noise_mask_cond.py:39). Both
// comparisons are STRICT and they point in opposite directions, which is what
// makes the window half-open `[start, end)` at the TOKEN level while still
// selecting a token that merely overlaps either edge. Written once so the two
// branches below cannot drift apart.
bool InRegion(double t_start, double t_end, double start_time, double end_time) {
  return t_end > start_time && t_start < end_time;
}

}  // namespace

std::vector<float> Ltx2TemporalRegionMaskVideo(const Ltx2VideoLatentShape& shape,
                                               int64_t patch_size,
                                               const Ltx2ScaleFactors& factors, double fps,
                                               double start_time, double end_time,
                                               bool causal_fix) {
  if (fps <= 0.0) {
    Fail("the temporal region mask divides the pixel bounds by fps "
         "(noise_mask_cond.py:35), so fps must be positive; got " +
         std::to_string(fps));
  }
  const int64_t tokens = Ltx2VideoTokenCount(shape, patch_size);
  // get_patch_grid_bounds -> get_pixel_coords (noise_mask_cond.py:31-33). The
  // temporal patch size is fixed at 1 (patchifiers.py:15), so latent frame `f`
  // enters as [f, f+1) and leaves as the causal-corrected pixel span.
  const std::vector<int64_t> bounds = Ltx2VideoPatchBounds(shape, patch_size);
  const std::vector<int64_t> pixels =
      Ltx2PixelCoords(bounds, shape.batch, tokens, factors, causal_fix);

  std::vector<float> mask(static_cast<size_t>(tokens));
  for (int64_t token = 0; token < tokens; ++token) {
    // Axis 0 of [batch, 3, tokens, 2] at batch 0 — the TEMPORAL axis. Axes 1 and
    // 2 are height and width and take no part in a temporal mask.
    const size_t base = static_cast<size_t>(token * 2);
    // `t_boundaries = pixel_bounds[:, 0] / self.fps` (noise_mask_cond.py:35).
    // Upstream divides an int64 tensor by a Python float, which torch promotes
    // to the default float32 dtype, so the division is done in float here and
    // widened only for the comparison.
    const double t_start =
        static_cast<double>(static_cast<float>(pixels[base]) / static_cast<float>(fps));
    const double t_end =
        static_cast<double>(static_cast<float>(pixels[base + 1]) / static_cast<float>(fps));
    mask[static_cast<size_t>(token)] =
        InRegion(t_start, t_end, start_time, end_time) ? 1.0F : 0.0F;
  }
  return mask;
}

std::vector<float> Ltx2TemporalRegionMaskAudio(const Ltx2AudioLatentShape& shape,
                                               const Ltx2AudioPatchifierParams& params,
                                               double start_time, double end_time) {
  // "Audio: patchifier get_patch_grid_bounds returns seconds"
  // (noise_mask_cond.py:28). No scale factors, no fps — see the header banner
  // for what happens when this side is scaled a second time.
  const std::vector<float> timings = Ltx2AudioPatchTimings(shape, params);
  std::vector<float> mask(static_cast<size_t>(shape.frames));
  for (int64_t token = 0; token < shape.frames; ++token) {
    const size_t base = static_cast<size_t>(token * 2);
    mask[static_cast<size_t>(token)] =
        InRegion(static_cast<double>(timings[base]), static_cast<double>(timings[base + 1]),
                 start_time, end_time)
            ? 1.0F
            : 0.0F;
  }
  return mask;
}

std::vector<float> Ltx2ConformLatentLength(const std::vector<float>& latent, int64_t channels,
                                           int64_t frames, int64_t per_frame,
                                           int64_t expected_frames) {
  if (channels < 1 || frames < 0 || per_frame < 1 || expected_frames < 1) {
    Fail("conform: channels, per-frame extent and the expected frame count must be positive and "
         "the actual frame count non-negative; got channels=" +
         std::to_string(channels) + " frames=" + std::to_string(frames) +
         " per_frame=" + std::to_string(per_frame) +
         " expected_frames=" + std::to_string(expected_frames));
  }
  const size_t want = static_cast<size_t>(channels) * static_cast<size_t>(frames) *
                      static_cast<size_t>(per_frame);
  if (latent.size() != want) {
    Fail("conform: the latent holds " + std::to_string(latent.size()) + " values but " +
         std::to_string(channels) + " channels x " + std::to_string(frames) + " frames x " +
         std::to_string(per_frame) + " per frame is " + std::to_string(want));
  }
  if (frames == expected_frames) return latent;

  // helpers.py:151-152 — `latent[:, :, :expected]`, so the LEADING frames
  // survive. A tail slice would keep the wrong end of the clip and is invisible
  // in every shape check.
  //
  // helpers.py:154-161 — `torch.cat([latent, pad], dim=2)`, so the zeros go on
  // the END. This is the polarity the audio-to-video path does NOT have; the
  // header banner says why the two cannot share a function.
  std::vector<float> out(static_cast<size_t>(channels) * static_cast<size_t>(expected_frames) *
                         static_cast<size_t>(per_frame));
  const int64_t copy_frames = std::min(frames, expected_frames);
  for (int64_t c = 0; c < channels; ++c) {
    const size_t src = static_cast<size_t>(c) * static_cast<size_t>(frames) *
                       static_cast<size_t>(per_frame);
    const size_t dst = static_cast<size_t>(c) * static_cast<size_t>(expected_frames) *
                       static_cast<size_t>(per_frame);
    const size_t n = static_cast<size_t>(copy_frames) * static_cast<size_t>(per_frame);
    std::copy(latent.begin() + static_cast<std::ptrdiff_t>(src),
              latent.begin() + static_cast<std::ptrdiff_t>(src + n),
              out.begin() + static_cast<std::ptrdiff_t>(dst));
  }
  return out;
}

void Ltx2RetakeAssertWindow(double start_time, double end_time) {
  // retake.py:211-212, verbatim in structure and in what it reports.
  if (start_time >= end_time) {
    Fail("start_time (" + std::to_string(start_time) + ") must be less than end_time (" +
         std::to_string(end_time) + ")");
  }
}

void Ltx2RetakeAssertSourceGeometry(int64_t frames, int64_t height, int64_t width,
                                    int64_t time_factor) {
  if (time_factor < 1) Fail("the VAE temporal scale factor must be positive");
  if (frames < 1) Fail("the source clip carries no frames");
  // retake.py:347-351. The snapped value is part of the message upstream, and it
  // is the part that turns a refusal into an instruction.
  if ((frames - 1) % time_factor != 0) {
    const int64_t snapped = ((frames - 1) / time_factor) * time_factor + 1;
    Fail("Video frame count must satisfy " + std::to_string(time_factor) +
         "k+1 (e.g. 97, 193). Got " + std::to_string(frames) + "; use a video with " +
         std::to_string(snapped) + " frames.");
  }
  // retake.py:352-353. One divisor, both axes, and both values reported.
  if (width % 32 != 0 || height % 32 != 0) {
    Fail("Video width and height must be multiples of 32. Got " + std::to_string(width) + "x" +
         std::to_string(height) + ".");
  }
}

Ltx2RetakeSourceGeometry Ltx2ProbeFrameDirectory(const std::string& dir) {
  Ltx2RetakeSourceGeometry geometry;
  for (int64_t index = 0;; ++index) {
    std::string bytes;
    if (!ReadWholeFile(FramePath(dir, index), &bytes)) break;
    int64_t h = 0, w = 0;
    // Decoding rather than sniffing the header: `Ltx2DecodePpmRgb` is the one
    // PPM reader on this side and it also refuses a non-P6 magic and a `maxval`
    // other than 255. A header-only probe would accept a file the read below
    // then rejects, and would report the geometry of something unreadable.
    (void)Ltx2DecodePpmRgb("retake source frame " + std::to_string(index), bytes, &h, &w);
    if (index == 0) {
      geometry.height = h;
      geometry.width = w;
    } else if (h != geometry.height || w != geometry.width) {
      Fail("frame_" + std::to_string(index) + " of '" + dir + "' is " + std::to_string(w) + "x" +
           std::to_string(h) + " but frame 0 is " + std::to_string(geometry.width) + "x" +
           std::to_string(geometry.height) +
           ". Upstream reads ONE spec for the whole folder (decode.py:217-223), so a ragged "
           "folder would silently reinterpret the later frames");
    }
    geometry.frames = index + 1;
  }
  if (geometry.frames == 0) {
    Fail("no frame_%06d.ppm files in '" + dir +
         "'. The retake source is a DIRECTORY of frames, numbered from 000000, which is the "
         "layout this ABI already documents for a reference video and the layout "
         "`minimax-h3-gen` writes");
  }
  return geometry;
}

std::vector<float> Ltx2ReadFrameDirectory(const std::string& dir, int64_t height, int64_t width) {
  if (height < 1 || width < 1) Fail("the target frame size must be positive");
  constexpr int64_t kChannels = 3;
  const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);

  std::vector<std::vector<float>> frames;  // each [3, height, width]
  for (int64_t index = 0;; ++index) {
    std::string bytes;
    if (!ReadWholeFile(FramePath(dir, index), &bytes)) break;
    // `crf = 0` is the identity arm of `preprocess` (decode.py:425-426), which
    // is what upstream's VIDEO ingestion applies: CRF is a per-image
    // conditioning knob and `video_preprocess` never takes one.
    frames.push_back(Ltx2LoadImageAndPreprocess("retake source frame " + std::to_string(index),
                                                bytes, height, width, /*crf=*/0));
  }
  if (frames.empty()) Fail("no frame_%06d.ppm files in '" + dir + "'");

  // Frame-major [T][C, H, W] to channel-major [C, T, H, W]: `Ltx2ConvVideoEncode`
  // takes the volume with the FRAME axis inside the channel axis, and reading
  // the two the other way round is a transposition that still type-checks.
  const int64_t count = static_cast<int64_t>(frames.size());
  std::vector<float> out(static_cast<size_t>(kChannels) * static_cast<size_t>(count) * plane);
  for (int64_t c = 0; c < kChannels; ++c) {
    for (int64_t t = 0; t < count; ++t) {
      const std::vector<float>& frame = frames[static_cast<size_t>(t)];
      const size_t src = static_cast<size_t>(c) * plane;
      const size_t dst = (static_cast<size_t>(c) * static_cast<size_t>(count) +
                          static_cast<size_t>(t)) *
                         plane;
      std::copy(frame.begin() + static_cast<std::ptrdiff_t>(src),
                frame.begin() + static_cast<std::ptrdiff_t>(src + plane),
                out.begin() + static_cast<std::ptrdiff_t>(dst));
    }
  }
  return out;
}

Ltx2RetakePlan Ltx2RetakePlanModalities(bool regenerate_video, bool regenerate_audio,
                                        bool has_audio_latent) {
  Ltx2RetakePlan plan;
  // retake.py:270-274. Neither video predicate consults the audio latent, and
  // the two are exact complements: the mask is applied when regenerating, and
  // the stream is frozen when not.
  plan.video_conditioned = regenerate_video;
  plan.video_frozen = !regenerate_video;
  // retake.py:278-282. BOTH audio predicates are conjunctions with
  // `initial_audio_latent is not None`, so with no audio latent the stream is
  // neither conditioned nor frozen whatever `regenerate_audio` says. See the
  // header for why that is mirrored rather than repaired.
  plan.audio_conditioned = has_audio_latent && regenerate_audio;
  plan.audio_frozen = has_audio_latent && !regenerate_audio;
  return plan;
}

}  // namespace vllm
