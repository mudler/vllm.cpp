// LTX-2.5 IMAGE CONDITIONING INPUT — pixels to the VAE encoder's [-1, 1] space.
//
// Row: LTX25-IMAGE-COND. Spec: .agents/specs/ltx25-image-conditioning.md §3.2.
// Issue #644.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f,
// packages/ltx-pipelines/src/ltx_pipelines/utils/
//   OURS                             <-  UPSTREAM
//   Ltx2LoadImageAndPreprocess       <-  media_io/decode.py:46-79
//   Ltx2PreprocessImageCrf           <-  media_io/decode.py:413-435 (`preprocess`)
//   Ltx2DecodePpmRgb                 <-  media_io/decode.py:139-170 (`decode_image`)
//   Ltx2ResizeAndCenterCrop          <-  media_io/resize.py:41-73
//   (the /127.5 - 1 map)             <-  media_io/range_map.py:8-9
//   Ltx2ResolveDefaultImageCrf       <-  constants.py:36-37, 124, 126-133
//                                        + blocks.py:966-983 (`ImageConditioner`)
//
// ─── THE ORDER IS LOAD-BEARING ───────────────────────────────────────────────
// `load_image_and_preprocess` resizes in 0..255 SPACE and normalizes AFTER
// (decode.py:76-78). Bilinear interpolation is affine, so the two commute in
// exact arithmetic and NOT in floating point. This project's other PPM reader
// (`minimax_h3_video.cpp:87-120`) normalizes at decode time, which is why this
// file has its own decoder rather than calling that one: reusing it would put
// the affine map first, and no shape or finiteness check could see it.
//
// ─── WHAT IS DELIBERATELY NOT PORTED, AND WHAT THAT COSTS ────────────────────
//  * THE H.264 ROUND TRIP. `preprocess` re-compresses at the checkpoint's
//    `default_image_crf` to match the compression the model was trained against
//    (decode.py:413-435 -> encode_single_frame:386-400, libx264 preset=veryfast,
//    rgb24 -> yuv420p, dimensions truncated to even). No codec is vendored here,
//    so a non-zero CRF is REFUSED BY NAME. `crf == 0` short-circuits upstream at
//    :425-426 and is served — it is upstream-legal ("including ``0`` to skip
//    re-compression entirely", args.py:58-59) and OUT OF DISTRIBUTION for a 2.5
//    checkpoint, whose resolved default is 18. Both halves of that are said out
//    loud rather than one of them.
//  * PNG / JPEG / EXR. Only binary PPM (P6) is read — the same NAMED residual
//    minimax_h3_video.cpp:84-86 already carries for this tree. EXIF orientation
//    and ICC -> sRGB conversion (decode.py:143-168) have nothing to act on in a
//    PPM, so they are absent rather than dropped.
//  * A `maxval` other than 255. PIL rescales those; mirroring its rescale is a
//    separate port, so they are refused rather than scaled by a rule nobody
//    checked against PIL.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

// `decode_image` (decode.py:139-170) for the one container this tree reads:
// binary PPM (P6), `maxval == 255`, into a uint8 [H, W, 3] buffer — the exact
// shape and dtype `np.array(image, dtype=np.uint8)` returns at :170.
std::vector<uint8_t> Ltx2DecodePpmRgb(const std::string& field, const std::string& bytes,
                                      int64_t* out_height, int64_t* out_width);

// `preprocess` (decode.py:413-435). Identity at `crf == 0` (:425-426); THROWS by
// name at any other value, naming the unported codec round trip. `crf < 0` is
// refused too: upstream's only unset spelling is `None`, which
// `ImageConditioner.resolve_crf` fills in before `preprocess` ever sees it
// (blocks.py:977-983), so a negative value is not "unset" in any upstream sense.
void Ltx2PreprocessImageCrf(int64_t crf);

// `resize_and_center_crop` (resize.py:41-73) for a single image. Takes HWC
// f32 in ANY value space and returns [channels, height, width] — the
// `1 c f h w` upstream emits at f = 1, with the two singleton axes dropped.
//
// Aspect FILL then centre crop: `scale = max(height/src_h, width/src_w)`, then
// `ceil` (upstream's own comment: avoids a negative crop offset from
// floating-point rounding), then bilinear with `align_corners=False`, then
// `crop_top = (new_h - height) // 2`.
//
// The bilinear kernel mirrors PyTorch's index map in f32, which is the dtype
// `HelperInterpLinear::compute_indices_weights` dispatches at for a float
// tensor: `real = scale * (i + 0.5) - 0.5` clamped to >= 0, `idx = floor(real)`
// capped at `src - 1`, `lambda = clamp(real - idx, 0, 1)`, right tap
// `min(idx + 1, src - 1)`. HEIGHT is the OUTER sum and WIDTH the inner one,
// matching `Interpolate<n>::eval`'s recursion order over the dimensions
// `upsample_generic_Nd_kernel_impl` appends in.
std::vector<float> Ltx2ResizeAndCenterCrop(const float* hwc, int64_t src_height,
                                           int64_t src_width, int64_t channels, int64_t height,
                                           int64_t width);

// `load_image_and_preprocess` (decode.py:46-79) end to end: decode -> CRF ->
// f32 0..255 -> resize+crop -> `/127.5 - 1.0`. Returns [3, height, width] in
// [-1, 1], which is what `Ltx2ConvVideoEncode` takes at `frame_count = 1`.
std::vector<float> Ltx2LoadImageAndPreprocess(const std::string& field, const std::string& bytes,
                                              int64_t height, int64_t width, int64_t crf);

// `PipelineParams.default_image_crf` as `detect_params` resolves it
// (constants.py:126-133): the newest generation row at or below the checkpoint's
// `model_version`, so an unrecognised NEWER version inherits the closest known
// one rather than falling back to 2.0's. Today that is `(2, 4) -> 18`
// (LTX_2_4_IMAGE_CRF, :37, :124) and everything older -> 33 (DEFAULT_IMAGE_CRF,
// :36). `components` is `Ltx2ParseModelVersion`'s output; an EMPTY one compares
// below every row, exactly as `detect_model_version` documents at :137-140.
//
// This is what makes the CRF refusal a real one rather than a formality: an
// LTX-2.5 request that does not name a CRF resolves 18 and is refused, and the
// caller has to ask for 0 knowingly.
int64_t Ltx2ResolveDefaultImageCrf(const std::vector<int64_t>& version_components);

}  // namespace vllm
