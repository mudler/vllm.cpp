// MiniMax-H3 t2va pipeline assembly — the path from prompt embeddings to frames
// and a waveform.
//
// Everything this file calls was ported and gated separately; W6 is the WIRING:
//
//   prompt_embeds ──► packed sequence (fl2va/t2va layout)
//                     │
//                     ├─► denoise loop: 50 x DiT forward + euler-eta0 step
//                     │      (sigmas from the rectified-flow time-shift schedule)
//                     ▼
//              video rows / audio rows
//                     │
//        unpatchify ──┤── unpack audio
//                     ▼
//        video latent ──► video VAE ViT3D decoder ──► frames  [C, T, H, W]
//        audio latent ──► audio VAE BigVGAN       ──► waveform per channel
//
// Latents are DENORMALIZED before decode (`latent * std + mean`), mirroring
// vae.py:252-270 / :341-357.
//
// NOISE IS AN INPUT, deliberately. Upstream seeds a torch CPU generator
// (pipeline_minimax_h3.py:813-843); reproducing torch's RNG bit-exactly is a
// separate concern that decides WHICH sample you get, not whether the pipeline is
// correct, so the caller supplies the initial rows. See the spec's open items.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/model_executor/models/device_pool.h"  // ActivePool()/DevicePool::Drain
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {

// Turn supplied KEYFRAME IMAGES into the packed conditioning rows the denoise loop
// pins. This is the one step that stands between "the fl2va primitives are ported"
// and "a caller can pass a first frame".
//
// Each image is [3, H, W] in [-1, 1] and is encoded as a ONE-FRAME clip: the 3D CNN
// is causal in time, so a single frame is a valid input and yields latent_t 1.
// Rows are then patchified with the DiT's own patch sizes, which is what makes them
// interchangeable with the video rows the loop carries.
//
// `noise_aug` blends toward the supplied noise (1.0 pins the frame exactly). Noise
// is an INPUT for the same reason it is in the t2va path: reproducing upstream's RNG
// decides WHICH sample you get, not whether the pipeline is right.
void MiniMaxH3VideoDenormalizePixels(std::vector<float>& frames, int64_t channels,
                                     int64_t per_channel) {
  VT_CHECK(channels == 3, "minimax_h3: ImageNet de-normalization expects 3 channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * per_channel,
           "minimax_h3: frame buffer does not match [C, ...]");
  for (int64_t c = 0; c < channels; ++c) {
    const float mean = kMiniMaxH3ImagenetMean[c], std_dev = kMiniMaxH3ImagenetStd[c];
    for (int64_t i = 0; i < per_channel; ++i) {
      float& v = frames[static_cast<size_t>(c * per_channel + i)];
      // clamp in [0,1] BEFORE the [-1,1] map, matching vae.py:693's order --
      // clamping afterwards would let out-of-gamut values survive rescaled.
      const float pixel = std::min(1.0F, std::max(0.0F, v * std_dev + mean));
      v = pixel * 2.0F - 1.0F;
    }
  }
}

void MiniMaxH3VideoNormalizePixels(std::vector<float>& frames, int64_t channels,
                                   int64_t per_channel) {
  VT_CHECK(channels == 3, "minimax_h3: ImageNet normalization expects 3 channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * per_channel,
           "minimax_h3: frame buffer does not match [C, ...]");
  for (int64_t c = 0; c < channels; ++c) {
    const float mean = kMiniMaxH3ImagenetMean[c], std_dev = kMiniMaxH3ImagenetStd[c];
    for (int64_t i = 0; i < per_channel; ++i) {
      float& v = frames[static_cast<size_t>(c * per_channel + i)];
      v = ((v + 1.0F) * 0.5F - mean) / std_dev;  // vae.py:659
    }
  }
}

std::vector<float> MiniMaxH3EncodeReferenceVideo(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<float>& frames, int64_t frame_count, int64_t frame_h, int64_t frame_w,
    MiniMaxH3RefBlock* out_block) {
  VT_CHECK(frame_count > 0, "minimax_h3 ref2va: a video reference needs at least one frame");
  VT_CHECK(static_cast<int64_t>(frames.size()) ==
               encoder_config.in_channels * frame_count * frame_h * frame_w,
           "minimax_h3 ref2va: reference video is not [in_channels, T, H, W]");

  // The 3D CNN is CAUSAL in time, so a clip encodes in one call -- this is the same
  // encoder the single-image path uses, just with t > 1, which is why a video
  // reference needed no new porting once the image path existed.
  // Same ImageNet normalization the image path needs (vae.py:659).
  std::vector<float> norm = frames;
  MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, frame_count * frame_h * frame_w);
  MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
  cfg.t = frame_count;
  cfg.h = frame_h;
  cfg.w = frame_w;
  MiniMaxH3VideoFrameShape ls{};
  const std::vector<float> latent =
      MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
  std::vector<float> rows = MiniMaxH3PatchifyVideoLatent(
      latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
      dit_params.patch_size_h, dit_params.patch_size_w);

  if (out_block != nullptr) {
    // kVideoAudio is the only kind that carries a temporal extent -- kImage counts
    // exactly one frame regardless of latent_t. ref_audio_t stays 0: silent.
    // Dims are the RAW VAE latent (t, h, w). BuildMiniMaxH3PackedSequenceRef2va
    // applies the DiT [1,2,2] patch division ITSELF (it mirrors upstream
    // minimax_h3_packed_sequence_ref2va_blocks, which takes the unpatched latent and
    // divides by _PATCH_{H,W} once) -- pre-dividing here would double-count the patch.
    MiniMaxH3RefBlock b;
    b.kind = MiniMaxH3RefBlock::Kind::kVideoAudio;
    b.ref_audio_t = 0;
    b.latent_t = ls.t;
    b.latent_h = ls.h;
    b.latent_w = ls.w;
    *out_block = b;
  }
  return rows;
}

// Turn a supplied REFERENCE WAVEFORM into the packed audio conditioning rows, the
// audio counterpart of MiniMaxH3EncodeReferenceImages. This is what makes a
// `kAudio` block (and a `kVideoAudio` block with sound) expressible at all.
//
// `waveform` is CHANNEL-MAJOR and already at kMiniMaxH3AudioSampleRate; a mono
// source is REPEATED up to kMiniMaxH3AudioChannels, exactly as upstream does
// before it encodes (vae.py:305-313). Rows come back normalized by the audio VAE's
// own latent statistics, because that is the space the DiT's rows live in.
//
// `noise_aug` blends toward the supplied noise the way the visual side does
// (1.0 pins the reference exactly); noise is an INPUT for the same reason it is
// everywhere else in this pipeline.
std::vector<float> MiniMaxH3EncodeReferenceAudio(
    const MiniMaxH3AudioVaeEncoderConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const std::vector<float>& waveform,
    int64_t channels, int64_t samples_per_channel, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std, double noise_aug,
    const std::vector<float>& noise_rows, MiniMaxH3RefBlock* out_block) {
  VT_CHECK(channels == kMiniMaxH3AudioChannels,
           "minimax_h3 ref2va: reference audio must be stereo (mono is repeated by the caller)");
  int64_t audio_t = 0;
  std::vector<float> rows = MiniMaxH3AudioVaeEncodeToRows(
      encoder_config, encoder_weights, waveform, channels, samples_per_channel, latents_mean,
      latents_std, &audio_t);
  VT_CHECK(audio_t > 0, "minimax_h3 ref2va: the reference waveform encoded to zero latent frames");

  if (out_block != nullptr) {
    // A bare audio reference is a kAudio block: no visual extent, `ref_audio_t`
    // latent frames per channel.
    MiniMaxH3RefBlock b;
    b.kind = MiniMaxH3RefBlock::Kind::kAudio;
    b.ref_audio_t = audio_t;
    *out_block = b;
  }
  if (noise_aug >= 1.0 || noise_rows.empty()) return rows;
  return MiniMaxH3AudioCondNoiseAug(rows, {audio_t}, noise_aug, noise_rows);
}

std::vector<float> MiniMaxH3EncodeReferenceImages(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<std::vector<float>>& images, int64_t image_h, int64_t image_w,
    std::vector<MiniMaxH3RefBlock>* out_blocks) {
  VT_CHECK(!images.empty(), "minimax_h3 ref2va: no reference images supplied");
  std::vector<float> rows;
  if (out_blocks != nullptr) out_blocks->clear();
  for (const std::vector<float>& img : images) {
    VT_CHECK(static_cast<int64_t>(img.size()) == encoder_config.in_channels * image_h * image_w,
             "minimax_h3 ref2va: reference image is not [in_channels, H, W]");
    // The encoder consumes IMAGENET-NORMALIZED pixels (vae.py:659), not [-1, 1].
    std::vector<float> norm = img;
    MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, image_h * image_w);
    MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
    cfg.t = 1;
    cfg.h = image_h;
    cfg.w = image_w;
    MiniMaxH3VideoFrameShape ls{};
    const std::vector<float> latent =
        MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
    const std::vector<float> patched = MiniMaxH3PatchifyVideoLatent(
        latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
        dit_params.patch_size_h, dit_params.patch_size_w);
    rows.insert(rows.end(), patched.begin(), patched.end());
    if (out_blocks != nullptr) {
      // The block declares the RAW VAE-latent grid (t, h, w).
      // BuildMiniMaxH3PackedSequenceRef2va applies the DiT [1,2,2] patch division
      // ITSELF, mirroring upstream minimax_h3_packed_sequence_ref2va_blocks, which
      // takes the unpatched latent and divides by _PATCH_{H,W} once
      // (pipeline_minimax_h3.py:1141-1145 sets visual_shape = (1, height//16,
      // width//16), the RAW latent, NOT the patched grid). Pre-dividing here
      // double-counted the patch: it under-allocated the reference rows by
      // patch_h*patch_w (=4), silently truncating the pinned reference to its first
      // quarter and positioning it on a shrunken grid -- the ref2va patch grid
      // (spec section 8.9). The encoded row COUNT is invariant-gated in
      // test_minimax_h3.cpp so this convention cannot silently drift again.
      MiniMaxH3RefBlock b;
      b.kind = MiniMaxH3RefBlock::Kind::kImage;
      b.latent_t = ls.t;
      b.latent_h = ls.h;
      b.latent_w = ls.w;
      out_blocks->push_back(b);
    }
  }
  return rows;
}

std::vector<float> MiniMaxH3EncodeKeyframeCondRows(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config, const MiniMaxH3AudioVaeWeights& encoder_weights,
    const MiniMaxH3DitParams& dit_params, const std::vector<std::vector<float>>& images,
    int64_t image_h, int64_t image_w, int64_t target_latent_t, double noise_aug,
    const std::vector<float>& noise_rows) {
  VT_CHECK(!images.empty(), "minimax_h3 keyframe: no images supplied");
  std::vector<float> rows;
  std::vector<int64_t> condition_shapes;
  for (const std::vector<float>& img : images) {
    VT_CHECK(static_cast<int64_t>(img.size()) == encoder_config.in_channels * image_h * image_w,
             "minimax_h3 keyframe: image is not [in_channels, H, W]");
    // The encoder consumes IMAGENET-NORMALIZED pixels (vae.py:659), not [-1, 1].
    std::vector<float> norm = img;
    MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, image_h * image_w);
    MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
    cfg.t = 1;  // a single frame is a valid causal clip
    cfg.h = image_h;
    cfg.w = image_w;
    MiniMaxH3VideoFrameShape ls{};
    const std::vector<float> latent =
        MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
    // [C, t, h, w] -> packed rows with the DiT's patch volume.
    std::vector<float> patched = MiniMaxH3PatchifyVideoLatent(
        latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
        dit_params.patch_size_h, dit_params.patch_size_w);
    rows.insert(rows.end(), patched.begin(), patched.end());
    condition_shapes.push_back(ls.t);
    condition_shapes.push_back(ls.h);
    condition_shapes.push_back(ls.w);
  }
  if (noise_aug >= 1.0 || noise_rows.empty()) return rows;
  return MiniMaxH3ImgvidCondNoiseAug(rows, condition_shapes, target_latent_t,
                                     static_cast<int64_t>(images.size()), noise_aug, noise_rows);
}

MiniMaxH3DenoiseResult MiniMaxH3DenoiseT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                            const MiniMaxH3DitParams& dit_params,
                                            const MiniMaxH3DitWeights& dit_weights,
                                            const std::vector<float>& prompt_embeds,
                                            const std::vector<float>& initial_video_rows,
                                            const std::vector<float>& initial_audio_rows,
                                            vt::DType compute_dtype,
                                            const MiniMaxH3DitDeviceWeights* prestaged) {
  VT_CHECK(request.text_len > 0, "minimax_h3 t2va: text_len must be positive");
  VT_CHECK(request.num_steps >= 1, "minimax_h3 t2va: num_steps must be >= 1");
  VT_CHECK(static_cast<int64_t>(prompt_embeds.size()) == request.text_len * dit_params.text_dim,
           "minimax_h3 t2va: prompt_embeds must be [text_len, text_dim]");

  // --- 1. packed layout. fl2va is the SAME layout with keyframe conditioning
  // switched on, so t2va is just the empty-keyframe case rather than a separate
  // path -- upstream models it the same way (packed_sequence.py:116-239).
  const bool has_keyframes = !request.keyframe_frame_indices.empty();
  const bool has_refs = !request.ref_blocks.empty();
  VT_CHECK(!(has_keyframes && has_refs),
           "minimax_h3: keyframe (fl2va) and reference (ref2va) conditioning are exclusive");
  MiniMaxH3DenoiseBranch branch;
  int64_t ref_audio_rows = 0;
  if (has_refs) {
    // AUDIO-bearing reference blocks claim packed audio rows, and every claimed row
    // must be BACKED by an encoded reference -- otherwise the layout would grow and
    // the loop would pin whatever happened to be in the buffer. The rows come from
    // MiniMaxH3EncodeReferenceAudio (the audio VAE's encoder half).
    for (const MiniMaxH3RefBlock& b : request.ref_blocks) {
      if (b.kind == MiniMaxH3RefBlock::Kind::kImage) continue;
      VT_CHECK(b.ref_audio_t >= 0, "minimax_h3 ref2va: ref_audio_t cannot be negative");
      ref_audio_rows += b.ref_audio_t * request.audio_channel;
    }
    VT_CHECK(static_cast<int64_t>(request.audio_ref_rows.size()) ==
                 ref_audio_rows * dit_params.audio_latents_dim,
             "minimax_h3 ref2va: audio_ref_rows must supply exactly the rows the reference blocks "
             "claim (ref_audio_t * audio_channel per block) -- see "
             "MiniMaxH3EncodeReferenceAudio");
    branch.packed = BuildMiniMaxH3PackedSequenceRef2va(
        request.text_len, request.latent_t, request.latent_h, request.latent_w, request.audio_t,
        request.ref_blocks, request.audio_channel);
  } else {
    branch.packed = BuildMiniMaxH3PackedSequence(
        request.text_len, request.latent_t, request.latent_h, request.latent_w, request.audio_t,
        request.audio_channel, has_keyframes, request.keyframe_frame_indices,
        has_keyframes ? request.num_frames : 0);
  }
  branch.text_embeddings = prompt_embeds;
  branch.token_tags = branch.packed.token_tags;

  // --- 2. the rectified-flow sigma schedules (video and audio shift differently) ---
  const std::vector<double> sigmas_video =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.video_shift);
  const std::vector<double> sigmas_audio =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.audio_shift);
  VT_CHECK(sigmas_video.size() == sigmas_audio.size(),
           "minimax_h3 t2va: the two sigma schedules must have equal length");

  // --- 3. the denoise loop (one DiT forward per step) ---
  // Keyframe conditioning ADDS condition rows to the packed layout, and the loop
  // wants initial rows for every img position. Callers supply noise for the TARGET
  // rows -- that is what they can know -- so the condition slots are filled in
  // here. Their contents do not matter: the loop pins them to the keyframe anchors
  // before the first step.
  const int64_t video_width = dit_params.video_row_width();
  const int64_t num_img = static_cast<int64_t>(branch.packed.img_pos.size());
  std::vector<float> video_rows = initial_video_rows;
  if (static_cast<int64_t>(video_rows.size()) != num_img * video_width) {
    VT_CHECK(has_keyframes || has_refs,
             "minimax_h3 t2va: initial video rows do not match the packed layout");
    std::vector<float> full(static_cast<size_t>(num_img * video_width), 0.0f);
    int64_t src = 0;
    for (int64_t r = 0; r < num_img; ++r) {
      if (!branch.packed.update_mask[static_cast<size_t>(r)]) continue;  // pinned anchor
      VT_CHECK((src + 1) * video_width <= static_cast<int64_t>(initial_video_rows.size()),
               "minimax_h3 t2va: too few initial video rows for the denoise targets");
      std::copy(initial_video_rows.begin() + src * video_width,
                initial_video_rows.begin() + (src + 1) * video_width,
                full.begin() + r * video_width);
      ++src;
    }
    video_rows = std::move(full);
  }

  // The audio side has the SAME shape of problem once a reference carries sound:
  // reference audio adds PINNED rows to the layout, and callers supply noise only
  // for the denoise TARGETS (pipeline_minimax_h3.py:937-955). Scatter theirs into
  // the updating slots and leave the pinned ones to the anchors below.
  const int64_t audio_width = dit_params.audio_latents_dim;
  const int64_t num_audio = static_cast<int64_t>(branch.packed.audio_pos.size());
  std::vector<float> audio_rows = initial_audio_rows;
  if (static_cast<int64_t>(audio_rows.size()) != num_audio * audio_width) {
    VT_CHECK(ref_audio_rows > 0,
             "minimax_h3 t2va: initial audio rows do not match the packed layout");
    std::vector<float> full(static_cast<size_t>(num_audio * audio_width), 0.0f);
    int64_t src = 0;
    for (int64_t r = 0; r < num_audio; ++r) {
      if (!branch.packed.audio_update_mask.empty() &&
          !branch.packed.audio_update_mask[static_cast<size_t>(r)]) {
        continue;  // a pinned reference row
      }
      VT_CHECK((src + 1) * audio_width <= static_cast<int64_t>(initial_audio_rows.size()),
               "minimax_h3 t2va: too few initial audio rows for the denoise targets");
      std::copy(initial_audio_rows.begin() + src * audio_width,
                initial_audio_rows.begin() + (src + 1) * audio_width,
                full.begin() + r * audio_width);
      ++src;
    }
    audio_rows = std::move(full);
  }

  // Both the keyframe rows and the reference-audio rows are PINNED: the loop
  // resets them to these anchors every step, so the supplied conditioning stays
  // put instead of being denoised away.
  return MiniMaxH3DenoiseLoop(device, dit_params, dit_weights, branch, video_rows, audio_rows,
                              request.keyframe_cond_rows, request.audio_ref_rows, sigmas_video,
                              sigmas_audio, compute_dtype, prestaged);
}

MiniMaxH3T2vaResult MiniMaxH3GenerateT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                          const MiniMaxH3DitParams& dit_params,
                                          const MiniMaxH3DitWeights& dit_weights,
                                          const MiniMaxH3VideoVaeDecoderConfig& video_config,
                                          const MiniMaxH3AudioVaeWeights& video_weights,
                                          const MiniMaxH3AudioVaeConfig& audio_config,
                                          const MiniMaxH3AudioVaeWeights& audio_weights,
                                          const std::vector<float>& prompt_embeds,
                                          const std::vector<float>& initial_video_rows,
                                          const std::vector<float>& initial_audio_rows,
                                          vt::DType compute_dtype,
                                          const MiniMaxH3DitDeviceWeights* prestaged) {
  // The task/partition guard — the raise half of `_resolve_task`
  // (pipeline_minimax_h3.py:374-391), which the #70/#74 white render bypassed by
  // running t2va on the Ref2VA NVFP4 checkpoint. The task is what the request
  // ENCODES (ref_blocks => ref2va, keyframes => fl2va, else t2va); a checkpoint that
  // declared its partition (driver --partition / server model_index.json) refuses a
  // task it does not serve. A request that never set `partition` (declared=false)
  // leaves the guard inactive, so the pipeline-math tests are unaffected.
  MiniMaxH3CheckTaskPartition(MiniMaxH3TaskOfRequest(request), request.partition);

  const MiniMaxH3DenoiseResult denoised =
      MiniMaxH3DenoiseT2va(device, request, dit_params, dit_weights, prompt_embeds,
                           initial_video_rows, initial_audio_rows, compute_dtype, prestaged);

  // --- 4. rows -> latents ---
  // ref2va PREPENDS pinned reference rows (encoded image/video/audio) to the
  // packed layout; the DiT zeroes them in its output (skip_mask_out_condition),
  // and only the TRAILING target rows are the generated clip. t2va/fl2va have no
  // such prefix, so the tail is the whole buffer -- this is a no-op there. Without
  // this, unpatchify sees (ref + target) rows and rejects a non-divisible count.
  const int64_t ph = request.latent_h / dit_params.patch_size_h;
  const int64_t pw = request.latent_w / dit_params.patch_size_w;
  const int64_t video_row_width = dit_params.video_row_width();
  const int64_t target_video_rows = request.latent_t * ph * pw;
  const int64_t have_video_rows =
      video_row_width > 0 ? static_cast<int64_t>(denoised.video_rows.size()) / video_row_width : 0;
  VT_CHECK(have_video_rows >= target_video_rows,
           "minimax_h3 t2va: denoise produced fewer video rows than the target clip needs");
  const std::vector<float> video_target_rows(
      denoised.video_rows.end() - target_video_rows * video_row_width, denoised.video_rows.end());
  std::vector<float> video_latent = MiniMaxH3UnpatchifyVideoTokens(
      video_target_rows, request.latent_t, ph, pw, dit_params.latents_dim,
      dit_params.patch_size_t, dit_params.patch_size_h, dit_params.patch_size_w);
  const int64_t audio_width = dit_params.audio_latents_dim;
  const int64_t target_audio_rows = request.audio_t * request.audio_channel;
  const int64_t have_audio_rows =
      audio_width > 0 ? static_cast<int64_t>(denoised.audio_rows.size()) / audio_width : 0;
  VT_CHECK(have_audio_rows >= target_audio_rows,
           "minimax_h3 t2va: denoise produced fewer audio rows than the target clip needs");
  const std::vector<float> audio_target_rows(
      denoised.audio_rows.end() - target_audio_rows * audio_width, denoised.audio_rows.end());
  std::vector<float> audio_latent = MiniMaxH3UnpackAudioTokens(
      audio_target_rows, target_audio_rows, request.audio_channel, dit_params.audio_latents_dim);

  // --- 5. denormalize (vae.py:252-270, :341-357) ---
  auto denormalize = [](std::vector<float>& latent, int64_t channels, int64_t per_channel,
                        const std::vector<float>& mean, const std::vector<float>& std_dev) {
    if (mean.empty() && std_dev.empty()) return;
    VT_CHECK(static_cast<int64_t>(mean.size()) == channels &&
                 static_cast<int64_t>(std_dev.size()) == channels,
             "minimax_h3 t2va: latents_mean/std must have one value per channel");
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t i = 0; i < per_channel; ++i) {
        float& value = latent[static_cast<size_t>(c * per_channel + i)];
        value = value * std_dev[static_cast<size_t>(c)] + mean[static_cast<size_t>(c)];
      }
    }
  };
  const int64_t video_per_channel = request.latent_t * request.latent_h * request.latent_w;
  denormalize(video_latent, dit_params.latents_dim, video_per_channel, request.video_latents_mean,
              request.video_latents_std);
  // `audio_t` is the PER-CHANNEL latent length (the planner sets it from
  // 40 Hz * duration), and the packed layout carries `audio_t * audio_channel`
  // ROWS -- one per (channel, step). Dividing by the channel count here halved
  // the decoded audio, and because the muxer passes `-shortest` that silently
  // truncated the VIDEO to half its frames too: a 124-frame render muxed 61.
  const int64_t audio_steps = request.audio_t;
  // DIAGNOSTIC (env-gated): the audio rows as the DiT EMITS them, before
  // denormalize. §8.17 measured the POST-denormalize latent at per-channel std
  // 1.0774 where the config's `latents_std` implies 1.8983 -- ~57% -- with the
  // MEAN landing correctly. Since denormalize is `value * std + mean`, that is
  // either a DiT whose normalized audio output is already under-dispersed, or a
  // denormalize that is not applying `std`. Those two have the SAME post-hoc
  // signature and only this dump separates them: if these rows are already
  // std ~0.57 the DiT (or the audio sigma schedule) owns it; if they are unit
  // and the shortfall appears after, the denormalize path does.
  if (const char* dump_dir = std::getenv("VT_H3_DUMP_DIR")) {
    const std::string ppath = std::string(dump_dir) + "/dit_audio_rows_prenorm.f32";
    if (std::FILE* pf = std::fopen(ppath.c_str(), "wb")) {
      std::fwrite(audio_latent.data(), sizeof(float), audio_latent.size(), pf);
      std::fclose(pf);
      std::fprintf(stderr, "[h3-dump] wrote %s (%zu floats, PRE-denormalize)\n", ppath.c_str(),
                   audio_latent.size());
    }
  }
  // The audio latent is NOT [latent_dim][...] like the video one. MiniMaxH3Unpack-
  // AudioTokens writes `[(c * latent_dim + d) * steps + t]`, i.e. the STEREO
  // CHANNEL is outermost -- [audio_channel][latent_dim][steps] -- and the decode
  // below reads it back at exactly that stride. Handing that buffer to the
  // video-shaped `denormalize` (which walks `latent[c * per_channel + i]` with c
  // over latents_dim) made block c cover latent dims {2c, 2c+1} of ONE stereo
  // channel, so every dim got another dim's mean/std: dim d must take mean[d] /
  // std[d] in BOTH stereo channels. The video arm is unaffected -- its latent
  // really is [C][T][H][W] with C outermost.
  //
  // Measured cost of the mismatch (§8.17/§8.18): the latent came out with roughly
  // HALF the per-channel |mean| of real speech (0.2406 against 0.4388-0.5008,
  // i.e. sitting on the CORPUS mean 0.194 instead of carrying clip identity) and
  // raised variance -- the averaging signature of statistics applied across the
  // wrong axis -- and it did not round-trip (encode(decode(z)) cosine +0.016).
  // That is the metallic voice.
  if (!request.audio_latents_mean.empty() || !request.audio_latents_std.empty()) {
    const int64_t adim = dit_params.audio_latents_dim;
    VT_CHECK(static_cast<int64_t>(request.audio_latents_mean.size()) == adim &&
                 static_cast<int64_t>(request.audio_latents_std.size()) == adim,
             "minimax_h3 t2va: audio latents_mean/std must have one value per latent dim");
    VT_CHECK(static_cast<int64_t>(audio_latent.size()) ==
                 request.audio_channel * adim * audio_steps,
             "minimax_h3 t2va: audio latent is not [audio_channel, latents_dim, audio_t]");
    for (int64_t ac = 0; ac < request.audio_channel; ++ac) {
      for (int64_t d = 0; d < adim; ++d) {
        const float mu = request.audio_latents_mean[static_cast<size_t>(d)];
        const float sd = request.audio_latents_std[static_cast<size_t>(d)];
        float* row = audio_latent.data() + static_cast<size_t>((ac * adim + d) * audio_steps);
        for (int64_t t = 0; t < audio_steps; ++t) row[t] = row[t] * sd + mu;
      }
    }
  }

  // --- 6. decode ---
  MiniMaxH3T2vaResult result;
  // post_quant_conv (the AutoencoderKL WRAPPER's Conv3d 1x1x1 channel mix) runs on
  // the latent BEFORE the decoder. It sits OUTSIDE ViT3DDecoder, which is why the
  // decoder's own 8.9e-8 gate never covered it. Applied only when the weights
  // carry it, so a synthetic/reduced-dimension weight set without it still runs --
  // the structural t2va test does not ship one.
  if (video_weights.Has("post_quant_conv.weight")) {
    video_latent = MiniMaxH3VideoVaePostQuantConv(video_weights, video_latent,
                                                  dit_params.latents_dim, video_per_channel);
  }
  // DIAGNOSTIC (env-gated, byte-identical when unset): dump the exact latent that
  // enters the video VAE (post unpatchify + denormalize + post_quant_conv) as raw
  // f32, so the VAE decode can be replayed on a KNOWN latent and runs byte-compared
  // (H3 render-coherence bisection at the VAE boundary).
  if (const char* dump_dir = std::getenv("VT_H3_DUMP_DIR")) {
    std::string path = std::string(dump_dir) + "/vae_input_video_latent.f32";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(video_latent.data(), sizeof(float), video_latent.size(), f);
      std::fclose(f);
      std::fprintf(stderr, "[h3-dump] wrote %s (%zu floats, channels=%lld per_channel=%lld)\n",
                   path.c_str(), video_latent.size(),
                   static_cast<long long>(dit_params.latents_dim),
                   static_cast<long long>(video_per_channel));
    }
    // The AUDIO arm of the same boundary. Without this the audio latent is the one
    // stage of the pipeline that can only be reasoned about, never measured: the
    // video arm has been bisected repeatedly (§8.16) while every audio claim has
    // been inferred from the two modalities sharing one attention sequence.
    // Layout matches the video dump's convention: [C][audio_t * audio_channel],
    // channel-major, exactly what `denormalize` above indexes.
    const std::string apath = std::string(dump_dir) + "/vae_input_audio_latent.f32";
    if (std::FILE* af = std::fopen(apath.c_str(), "wb")) {
      std::fwrite(audio_latent.data(), sizeof(float), audio_latent.size(), af);
      std::fclose(af);
      std::fprintf(stderr,
                   "[h3-dump] wrote %s (%zu floats, channels=%lld per_channel=%lld "
                   "audio_t=%lld audio_channel=%lld)\n",
                   apath.c_str(), audio_latent.size(),
                   static_cast<long long>(dit_params.audio_latents_dim),
                   static_cast<long long>(audio_steps * request.audio_channel),
                   static_cast<long long>(audio_steps),
                   static_cast<long long>(request.audio_channel));
    }
  }

  // On a device, run the ViT3D decoder device-resident. The portable decoder is a
  // scalar reference; at real resolutions it is the stage that does not finish. It
  // stays the CPU path, and stays the thing the device path is gated against.
  if (device.type != vt::DeviceType::kCPU) {
    vt::Backend& vae_backend = vt::GetBackend(device.type);
    // PHASE CHANGE: denoise is done, the VAE decode is next. The denoise left the
    // scratch pool holding every activation size class it touched, and on an
    // UNCAPPED pool (GB10/Thor: `device_pool_cap_bytes == 0`) those blocks are
    // never returned to the driver. The decode allocates DIFFERENT classes, so it
    // cannot reuse any of them -- they are pure headroom loss at the one moment
    // the decode needs its own working set. At the REF canvas (1344x768/124f)
    // that is the difference between a decode that fits and one that takes the
    // BOX DOWN: measured 85 GiB resident at this point against a ~18 GiB decode
    // in a 122 GiB unified pool, and the driver OOM (NV_ERR_NO_MEMORY) rebooted
    // the machine. Draining costs one cudaFree per retained block, once.
    const size_t drained = ActivePool(vae_backend).Drain(vae_backend);
    if (std::getenv("VT_POOL_STATS") != nullptr) {
      std::fprintf(stderr, "[h3] drained %.2f GiB of denoise scratch before VAE decode\n",
                   static_cast<double>(drained) / (1024.0 * 1024.0 * 1024.0));
    }
    vt::Queue vq = vae_backend.CreateQueue();
    const MiniMaxH3VideoVaeDeviceWeights staged_vae =
        StageMiniMaxH3VideoVaeWeights(vq, video_config, video_weights);
    // Upstream's video path is decode_base -> decode_temporal: chunked in TIME.
    // decode_temporal then composes SPATIAL tiling per chunk whenever
    // `decoder_tiling` is set, which it is by DEFAULT (minimax_h3.h: `bool
    // decoder_tiling = true`) -- required, not optional, because the ViT3D's RoPE
    // is length-normalized over the grid it is handed. A real canvas therefore
    // decodes as 256-px tiles inside each temporal chunk.
    result.frames = MiniMaxH3VideoVaeDecodeTemporalDevice(
        device, video_config, staged_vae, video_latent, request.latent_t, request.latent_h,
        request.latent_w, request.num_frames, &result.frame_shape);

    // DIAGNOSTIC (env-gated): VAE receptive-field probe. Perturb ONE interior spatial
    // latent cell across all channels + temporal frames, re-decode, and report the
    // per-16px-block RMS change on output frame 0. If ONLY the perturbed cell's block
    // moves, the decoder is not mixing tokens spatially (render-coherence bisection).
    if (std::getenv("VT_H3_VAE_PROBE")) {
      const int64_t lh = request.latent_h, lw = request.latent_w, lt = request.latent_t;
      const int64_t C = dit_params.latents_dim;
      const int64_t per = lt * lh * lw;
      const int64_t ch = lh / 2, cw = lw / 2;  // center latent cell
      std::vector<float> pert = video_latent;
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t t = 0; t < lt; ++t) {
          const int64_t idx = c * per + (t * lh + ch) * lw + cw;
          pert[static_cast<size_t>(idx)] += 8.0f;  // large, unambiguous impulse
        }
      }
      MiniMaxH3VideoFrameShape ps{};
      std::vector<float> pf = MiniMaxH3VideoVaeDecodeTemporalDevice(
          device, video_config, staged_vae, pert, request.latent_t, request.latent_h,
          request.latent_w, request.num_frames, &ps);
      const int64_t oh = result.frame_shape.h, ow = result.frame_shape.w, oc = result.frame_shape.channels;
      const int64_t ratio = oh / lh;  // pixels per latent cell (== vae spatial ratio)
      // block-diff map over the lh x lw grid, output frame 0
      std::fprintf(stderr, "[h3-vae-probe] latent %lldx%lldx%lld -> frame %lldx%lld, ratio=%lld, "
                   "perturbed cell (h=%lld,w=%lld). Per-block RMS |delta| (x1000):\n",
                   (long long)lt, (long long)lh, (long long)lw, (long long)oh, (long long)ow,
                   (long long)ratio, (long long)ch, (long long)cw);
      const int64_t plane = oh * ow;
      for (int64_t bh = 0; bh < lh; ++bh) {
        std::string line;
        for (int64_t bw = 0; bw < lw; ++bw) {
          double s2 = 0.0; int64_t n = 0;
          for (int64_t c = 0; c < oc; ++c) {
            for (int64_t py = 0; py < ratio; ++py) {
              for (int64_t px = 0; px < ratio; ++px) {
                const int64_t oy = bh * ratio + py, ox = bw * ratio + px;
                if (oy >= oh || ox >= ow) continue;
                const int64_t k = c * plane + oy * ow + ox;
                const double dd = static_cast<double>(pf[static_cast<size_t>(k)]) -
                                  static_cast<double>(result.frames[static_cast<size_t>(k)]);
                s2 += dd * dd; ++n;
              }
            }
          }
          const int v = static_cast<int>(1000.0 * std::sqrt(s2 / (n > 0 ? n : 1)));
          char buf[16]; std::snprintf(buf, sizeof(buf), "%5d", v); line += buf;
        }
        std::fprintf(stderr, "[h3-vae-probe] %s\n", line.c_str());
      }
      std::fflush(stderr);
    }
  } else {
    result.frames = MiniMaxH3VideoVaeDecode(video_config, video_weights, video_latent,
                                            request.latent_t, request.latent_h, request.latent_w,
                                            &result.frame_shape);
  }
  // The ViT decoder emits IMAGENET-NORMALIZED values; the wrapper de-normalizes
  // them (vae.py:693). Outside the decoder, so its own gate never covered it --
  // the same shape of omission as post_quant_conv.
  MiniMaxH3VideoDenormalizePixels(
      result.frames, result.frame_shape.channels,
      result.frame_shape.t * result.frame_shape.h * result.frame_shape.w);

  // The audio VAE decodes ONE channel at a time; the packed rows are channel-major.
  result.audio_channels = request.audio_channel;
  for (int64_t c = 0; c < request.audio_channel; ++c) {
    std::vector<float> channel(static_cast<size_t>(dit_params.audio_latents_dim * audio_steps));
    for (int64_t d = 0; d < dit_params.audio_latents_dim; ++d) {
      for (int64_t t = 0; t < audio_steps; ++t) {
        channel[static_cast<size_t>(d * audio_steps + t)] = audio_latent[static_cast<size_t>(
            (c * dit_params.audio_latents_dim + d) * audio_steps + t)];
      }
    }
    int64_t samples = 0;
    const std::vector<float> wave =
        MiniMaxH3AudioVaeDecode(audio_config, audio_weights, channel, audio_steps, &samples);
    if (c == 0) {
      result.audio_samples_per_channel = samples;
      result.waveform.reserve(static_cast<size_t>(samples * request.audio_channel));
    } else {
      VT_CHECK(samples == result.audio_samples_per_channel,
               "minimax_h3 t2va: audio channels decoded to different lengths");
    }
    result.waveform.insert(result.waveform.end(), wave.begin(), wave.end());
  }
  result.sample_rate = kMiniMaxH3AudioSampleRate;
  return result;
}

}  // namespace vllm
