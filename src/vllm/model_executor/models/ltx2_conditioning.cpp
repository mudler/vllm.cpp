// LTX-2.5 CONDITIONING ITEMS — see ltx2_conditioning.h for the upstream map and
// for the four things that fail silently. Gated by
// scripts/gen-ltx2-vae-goldens.py section 9, which executes the upstream
// `ConditioningItem` classes through the real `VideoLatentTools` /
// `AudioLatentTools` and freezes every field they touch.
//
// f32 throughout, matching the reference arms next door: the masks and positions
// upstream builds are float32 regardless of the model dtype
// (tools.py:158-176, keyframe_cond.py:57-65), so this one is not a widening.
#include "vllm/model_executor/models/ltx2_conditioning.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {

// The per-token positions upstream derives for a VIDEO latent of `shape`:
// patch bounds -> get_pixel_coords -> float32 -> temporal axis divided by fps.
// Returned as [3, tokens, 2]; batch is 1 throughout this file.
std::vector<float> VideoPositions(const Ltx2VideoLatentShape& shape, int64_t patch_size,
                                  const Ltx2ScaleFactors& factors, bool causal_fix,
                                  int64_t tokens) {
  const std::vector<int64_t> bounds = Ltx2VideoPatchBounds(shape, patch_size);
  const std::vector<int64_t> coords =
      Ltx2PixelCoords(bounds, /*batch=*/1, tokens, factors, causal_fix);
  VT_CHECK(static_cast<int64_t>(coords.size()) == 3 * tokens * 2,
           "ltx2 conditioning: pixel coords do not match [3, tokens, 2]");
  std::vector<float> out(coords.size());
  for (size_t i = 0; i < coords.size(); ++i) out[i] = static_cast<float>(coords[i]);
  return out;
}

void DivideTemporalByFps(std::vector<float>* positions, int64_t tokens, double fps) {
  VT_CHECK(fps > 0.0, "ltx2 conditioning: fps must be positive");
  for (int64_t i = 0; i < tokens * 2; ++i) {
    (*positions)[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<double>((*positions)[static_cast<size_t>(i)]) / fps);
  }
}

// The shared tail of every APPENDING item (keyframe_cond.py:77-89,
// reference_video_cond.py:96-107, reference_audio_cond.py:55-64): the noisy
// tensor grows with ZEROS, the clean one with the tokens, the mask with
// `1 - strength`, and the positions with the item's own.
void AppendTokens(Ltx2LatentState* state, const std::vector<float>& tokens, int64_t token_count,
                  const std::vector<float>& positions, double strength) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(static_cast<int64_t>(tokens.size()) == token_count * state->width,
           "ltx2 conditioning: appended tokens do not match [tokens, width]");
  VT_CHECK(static_cast<int64_t>(positions.size()) == state->pos_dims * token_count * 2,
           "ltx2 conditioning: appended positions do not match [pos_dims, tokens, 2]");

  const int64_t before = state->tokens;
  const int64_t after = before + token_count;

  // BEFORE anything else, because upstream hands `extend_keyframes_mask` the
  // PRE-append state (keyframe_cond.py:85, reference_video_cond.py:103) and the
  // None-and-marked branch sizes its fresh mask from that state's denoise mask.
  // Both video items ported here pass `marked=False`; the one construct that
  // passes true is `VideoGeneratedKeyframeSlots` (keyframe_slots.py:121), which
  // is not ported.
  //
  // It lives INSIDE this helper rather than at the three call sites because
  // upstream's docstring makes the call an obligation of appending itself
  // (mask_utils.py:83-85), and an obligation spread over three sites is one the
  // fourth site forgets. The desynchronisation it prevents is invisible to every
  // shape check: the render stays the right size and simply applies a trained
  // term to the wrong tokens.
  Ltx2ExtendKeyframesMask(state, token_count, /*marked=*/false);

  state->latent.resize(static_cast<size_t>(after * state->width), 0.0f);
  state->clean.insert(state->clean.end(), tokens.begin(), tokens.end());
  state->mask.insert(state->mask.end(), static_cast<size_t>(token_count),
                     static_cast<float>(1.0 - strength));

  // Positions are [pos_dims, tokens, 2], so the concatenation is per DIMENSION,
  // not a plain append — writing it as one `insert` at the end interleaves the
  // reference's height axis into the target's time axis and still type-checks.
  std::vector<float> grown(static_cast<size_t>(state->pos_dims * after * 2));
  for (int64_t d = 0; d < state->pos_dims; ++d) {
    std::copy(state->positions.begin() + static_cast<ptrdiff_t>(d * before * 2),
              state->positions.begin() + static_cast<ptrdiff_t>((d + 1) * before * 2),
              grown.begin() + static_cast<ptrdiff_t>(d * after * 2));
    std::copy(positions.begin() + static_cast<ptrdiff_t>(d * token_count * 2),
              positions.begin() + static_cast<ptrdiff_t>((d + 1) * token_count * 2),
              grown.begin() + static_cast<ptrdiff_t>(d * after * 2 + before * 2));
  }
  state->positions.swap(grown);
  state->tokens = after;
}

}  // namespace

Ltx2LatentState Ltx2CreateVideoLatentState(const Ltx2VideoLatentShape& shape, int64_t patch_size,
                                           const Ltx2ScaleFactors& factors, double fps,
                                           bool causal_fix, const float* initial_latent,
                                           std::vector<float>* out_keyframes_mask) {
  VT_CHECK(shape.batch == 1, "ltx2 conditioning: batch 1 only");
  const int64_t tokens = Ltx2VideoTokenCount(shape, patch_size);
  Ltx2LatentState state;
  state.tokens = tokens;
  state.width = shape.channels * patch_size * patch_size;
  state.pos_dims = 3;

  const std::vector<float> zeros(
      static_cast<size_t>(shape.channels * shape.frames * shape.height * shape.width), 0.0f);
  const float* source = initial_latent != nullptr ? initial_latent : zeros.data();
  state.latent = Ltx2VideoPatchify(source, shape, patch_size);
  state.clean = state.latent;
  state.mask.assign(static_cast<size_t>(tokens), 1.0f);
  state.positions = VideoPositions(shape, patch_size, factors, causal_fix, tokens);
  DivideTemporalByFps(&state.positions, tokens, fps);

  // tools.py:184 — `create_initial_state` returns
  // `replace(state, keyframes_mask=self._first_frame_keyframes_mask(state))` on
  // the same line that builds the state, unconditionally. Carried ON the state
  // so that an append can extend it; `out_keyframes_mask` stays for the callers
  // that only want the vector.
  state.keyframes_mask = Ltx2FirstFrameKeyframesMask(shape, patch_size);
  if (out_keyframes_mask != nullptr) {
    *out_keyframes_mask = state.keyframes_mask;
  }
  return state;
}

void Ltx2ExtendKeyframesMask(Ltx2LatentState* state, int64_t num_new_tokens, bool marked) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(num_new_tokens >= 0, "ltx2 conditioning: cannot extend the keyframes mask by a "
                                "negative token count");
  // `existing is None and not marked` -> `return None` (mask_utils.py:98-99).
  // The empty vector IS None here, so an audio state and an unmarked append onto
  // an unmarked state both stay empty rather than materialising a zero mask that
  // the DiT would then read as "a marker was supplied".
  if (state->keyframes_mask.empty() && !marked) return;
  // `existing is None` and marked -> `zeros_like(denoise_mask)` first
  // (mask_utils.py:100-101), sized by the state as it stands BEFORE the append.
  if (state->keyframes_mask.empty()) {
    state->keyframes_mask.assign(static_cast<size_t>(state->tokens), 0.0f);
  }
  VT_CHECK(static_cast<int64_t>(state->keyframes_mask.size()) == state->tokens,
           "ltx2 conditioning: the keyframes mask must have one value per token BEFORE the "
           "append — a mask that already disagrees with the token count has been extended by "
           "something that did not go through AppendTokens");
  state->keyframes_mask.insert(state->keyframes_mask.end(),
                               static_cast<size_t>(num_new_tokens), marked ? 1.0f : 0.0f);
}

void Ltx2ClearConditioning(Ltx2LatentState* state, int64_t target_tokens) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(target_tokens >= 0 && target_tokens <= state->tokens,
           "ltx2 conditioning: clear_conditioning TRUNCATES to the target token count "
           "(tools.py:101-105), so the target cannot exceed what the state carries. A state "
           "smaller than its own target means an appending item wrote somewhere other than the "
           "END, which is the one thing clear_conditioning's docstring forbids");

  state->latent.resize(static_cast<size_t>(target_tokens * state->width));
  state->clean.resize(static_cast<size_t>(target_tokens * state->width));

  // ALL ONES, not the conditioned mask sliced (tools.py:104 —
  // `torch.ones_like(latent_state.denoise_mask)[:, :num_tokens]`). The returned
  // state describes a FINISHED latent, in which every target token is denoised;
  // slicing the conditioned mask instead would carry `1 - strength` on the
  // conditioned tokens into whatever reads the state next, and on the two-stage
  // recipe that next reader is the following phase's initial latent.
  state->mask.assign(static_cast<size_t>(target_tokens), 1.0f);

  // Positions are [pos_dims, tokens, 2], so the truncation is per DIMENSION
  // (`positions[:, :, :num_tokens]`, tools.py:105). A plain resize would keep
  // the first dimension's appended tokens and drop the last dimension's real
  // ones, and the result still type-checks.
  std::vector<float> trimmed(static_cast<size_t>(state->pos_dims * target_tokens * 2));
  for (int64_t d = 0; d < state->pos_dims; ++d) {
    std::copy(state->positions.begin() + static_cast<ptrdiff_t>(d * state->tokens * 2),
              state->positions.begin() +
                  static_cast<ptrdiff_t>(d * state->tokens * 2 + target_tokens * 2),
              trimmed.begin() + static_cast<ptrdiff_t>(d * target_tokens * 2));
  }
  state->positions.swap(trimmed);

  // `keyframes_mask=None` (tools.py:113). The marker described a sequence that
  // no longer exists, and a sliced one would claim the trimmed state still
  // carries markers for tokens that were removed.
  state->keyframes_mask.clear();

  state->tokens = target_tokens;
}

std::vector<float> Ltx2FirstFrameKeyframesMask(const Ltx2VideoLatentShape& shape,
                                               int64_t patch_size) {
  // tools.py:194-195 — `zeros_like(denoise_mask)` then
  // `mask[:, :tokens_per_latent_frame] = 1.0`, with NO branch on whether a
  // keyframe exists. `tokens_per_latent_frame` is
  // `get_token_count(target._replace(frames=1))` (:198-201), so it is derived the
  // same way rather than as `tokens / frames`.
  const int64_t tokens = Ltx2VideoTokenCount(shape, patch_size);
  Ltx2VideoLatentShape one = shape;
  one.frames = 1;
  const int64_t per_frame = Ltx2VideoTokenCount(one, patch_size);
  VT_CHECK(per_frame <= tokens,
           "ltx2 conditioning: one latent frame cannot hold more tokens than the whole target");
  std::vector<float> mask(static_cast<size_t>(tokens), 0.0f);
  for (int64_t i = 0; i < per_frame; ++i) mask[static_cast<size_t>(i)] = 1.0f;
  return mask;
}

Ltx2LatentState Ltx2CreateAudioLatentState(const Ltx2AudioLatentShape& shape,
                                           const Ltx2AudioPatchifierParams& params,
                                           const float* initial_latent) {
  VT_CHECK(shape.batch == 1, "ltx2 conditioning: batch 1 only");
  Ltx2LatentState state;
  state.tokens = shape.frames;
  state.width = shape.channels * shape.mel_bins;
  state.pos_dims = 1;

  const std::vector<float> zeros(
      static_cast<size_t>(shape.channels * shape.frames * shape.mel_bins), 0.0f);
  const float* source = initial_latent != nullptr ? initial_latent : zeros.data();
  state.latent = Ltx2AudioPatchify(source, shape);
  state.clean = state.latent;
  state.mask.assign(static_cast<size_t>(shape.frames), 1.0f);
  // The audio positions ARE the patch grid bounds, in seconds — there is no
  // get_pixel_coords and no fps division on this side (tools.py:271-279).
  state.positions = Ltx2AudioPatchTimings(shape, params);
  return state;
}

void Ltx2ConditionVideoByLatentIndex(Ltx2LatentState* state, const Ltx2VideoLatentShape& target,
                                     int64_t patch_size, const Ltx2LatentVolume& conditioning,
                                     double strength, int64_t latent_idx) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(conditioning.batch == target.batch && conditioning.channels == target.channels &&
               conditioning.height == target.height && conditioning.width == target.width,
           "ltx2 conditioning: cannot apply an image conditioning whose (batch, channels, height, "
           "width) differs from the target latent's — upstream raises ConditioningError here "
           "(conditioning/types/latent_cond.py:28-34). Make sure the image and latent have the "
           "same spatial shape");
  VT_CHECK(latent_idx >= 0 && latent_idx + conditioning.frames <= target.frames,
           "ltx2 conditioning: the conditioning does not fit at that latent index");

  Ltx2VideoLatentShape cond_shape = target;
  cond_shape.frames = conditioning.frames;
  const std::vector<float> tokens =
      Ltx2VideoPatchify(conditioning.data.data(), cond_shape, patch_size);

  // start_token = get_token_count(target with frames = latent_idx) — the token
  // count of everything BEFORE the conditioned frame (latent_cond.py:36-38).
  Ltx2VideoLatentShape prefix = target;
  prefix.frames = latent_idx;
  const int64_t start = Ltx2VideoTokenCount(prefix, patch_size);
  const int64_t count = Ltx2VideoTokenCount(cond_shape, patch_size);
  VT_CHECK(static_cast<int64_t>(tokens.size()) == count * state->width,
           "ltx2 conditioning: patchified conditioning does not match [tokens, width]");
  VT_CHECK(start + count <= state->tokens, "ltx2 conditioning: token range runs past the state");

  // ONLY the clean latent and the mask. The noisy tensor is deliberately left
  // alone; the noiser composes the two (components/noisers.py:31-34).
  std::copy(tokens.begin(), tokens.end(),
            state->clean.begin() + static_cast<ptrdiff_t>(start * state->width));
  for (int64_t i = start; i < start + count; ++i) {
    state->mask[static_cast<size_t>(i)] = static_cast<float>(1.0 - strength);
  }
}

void Ltx2ConditionVideoByKeyframe(Ltx2LatentState* state, const Ltx2LatentVolume& keyframes,
                                  int64_t patch_size, const Ltx2ScaleFactors& factors, double fps,
                                  int64_t frame_idx, double strength, int64_t num_pixel_frames,
                                  bool causal_fix) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  Ltx2VideoLatentShape shape;
  shape.batch = keyframes.batch;
  shape.channels = keyframes.channels;
  shape.frames = keyframes.frames;
  shape.height = keyframes.height;
  shape.width = keyframes.width;
  const int64_t count = Ltx2VideoTokenCount(shape, patch_size);
  const std::vector<float> tokens = Ltx2VideoPatchify(keyframes.data.data(), shape, patch_size);

  // The causal fix is DISABLED unless the keyframe sits at pixel frame 0
  // (keyframe_cond.py:45-50): it exists for the target's own first latent frame,
  // and a keyframe placed later has no such frame to correct.
  std::vector<float> positions =
      VideoPositions(shape, patch_size, factors, frame_idx == 0 ? causal_fix : false, count);
  // The offset and the single-pixel-frame clamp happen in INTEGER pixel space,
  // BEFORE the division by fps (keyframe_cond.py:52-58).
  for (int64_t i = 0; i < count * 2; ++i) {
    positions[static_cast<size_t>(i)] += static_cast<float>(frame_idx);
  }
  if (num_pixel_frames == 1) {
    for (int64_t t = 0; t < count; ++t) {
      positions[static_cast<size_t>(t * 2 + 1)] = positions[static_cast<size_t>(t * 2)] + 1.0f;
    }
  }
  DivideTemporalByFps(&positions, count, fps);

  AppendTokens(state, tokens, count, positions, strength);
}

void Ltx2ConditionVideoByReference(Ltx2LatentState* state, const Ltx2LatentVolume& reference,
                                   int64_t patch_size, const Ltx2ScaleFactors& factors, double fps,
                                   int64_t downscale_factor, int64_t temporal_scale_factor,
                                   double strength, bool causal_fix) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(downscale_factor >= 1 && temporal_scale_factor >= 1,
           "ltx2 conditioning: the reference scale factors must be at least 1");
  VT_CHECK(state->tokens > 0,
           "ltx2 conditioning: the reference item reads the TARGET's first token period out of the "
           "state, so it cannot be applied to an empty one");

  Ltx2VideoLatentShape shape;
  shape.batch = reference.batch;
  shape.channels = reference.channels;
  shape.frames = reference.frames;
  shape.height = reference.height;
  shape.width = reference.width;
  const int64_t count = Ltx2VideoTokenCount(shape, patch_size);
  const std::vector<float> tokens = Ltx2VideoPatchify(reference.data.data(), shape, patch_size);

  // Unlike the keyframe item this one keeps the tools' own causal_fix
  // (reference_video_cond.py:60-65), and it converts to float BEFORE the
  // temporal rescale.
  std::vector<float> positions = VideoPositions(shape, patch_size, factors, causal_fix, count);
  // The reference sits on its OWN time spacing, target_fps / S
  // (reference_video_cond.py:71-72).
  DivideTemporalByFps(&positions, count, fps / static_cast<double>(temporal_scale_factor));

  if (temporal_scale_factor != 1) {
    // `t_target` is the TARGET's first token temporal END, i.e. 1 / target_fps,
    // read out of the state rather than recomputed — which is what makes this
    // item depend on the state it is applied to (reference_video_cond.py:74-79).
    const double t_target = static_cast<double>(state->positions[1]);
    const double shift = (static_cast<double>(temporal_scale_factor) - 1.0) * t_target;
    for (int64_t i = 0; i < count * 2; ++i) {
      positions[static_cast<size_t>(i)] = static_cast<float>(
          std::max(0.0, static_cast<double>(positions[static_cast<size_t>(i)]) - shift));
    }
  }
  if (downscale_factor != 1) {
    // Height is dim 1 and width is dim 2 (reference_video_cond.py:81-82).
    for (int64_t d = 1; d < 3; ++d) {
      for (int64_t i = 0; i < count * 2; ++i) {
        positions[static_cast<size_t>(d * count * 2 + i)] *= static_cast<float>(downscale_factor);
      }
    }
  }

  AppendTokens(state, tokens, count, positions, strength);
}

void Ltx2ConditionAudioByReference(Ltx2LatentState* state, const std::vector<float>& tokens,
                                   int64_t token_count, int64_t width,
                                   const std::vector<float>& positions, double strength) {
  VT_CHECK(state != nullptr, "ltx2 conditioning: null state");
  VT_CHECK(width == state->width,
           "ltx2 conditioning: the reference audio's token width must match the target's");
  AppendTokens(state, tokens, token_count, positions, strength);
}

}  // namespace vllm
