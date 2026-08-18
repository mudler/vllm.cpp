// LTX-2.5 DFR CANVAS LAYOUT — see
// include/vllm/model_executor/models/ltx2_dfr.h for the upstream mapping and for
// the five failure modes this file exists to make gateable.
//
// NO FLOATING POINT REACHES AN INDEX HERE, with exactly one annotated exception
// (`SlotSeedIndex`, which mirrors upstream's `round`). Everything else is
// integer arithmetic, because the quantities are token counts and frame indices
// and a rounding difference in one of them moves a keyframe rather than
// perturbing a value. That is the whole reason this is a separate translation
// unit from `ltx2_pipeline.cpp`, whose convention is the opposite — see that
// file's header on accumulating reductions in double.
//
// UPSTREAM RAISES; SO DO WE. Every `ValueError` and `RuntimeError` in
// `dfr_layout.py` and in the three `dfr_pipeline.py` helpers is mirrored as a
// throw with the same trigger. None of them is a defensive assertion added here:
// a canvas that fails one of these checks would otherwise produce a correctly
// shaped latent describing the wrong frames.
#include "vllm/model_executor/models/ltx2_dfr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) {
  throw std::runtime_error("ltx2 dfr: " + message);
}

std::string Join(const std::vector<int64_t>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(values[i]);
  }
  return out + "]";
}

// `(segment - content % segment) % segment` (dfr_layout.py:47-48). The outer
// modulo is what makes an exact multiple pad ZERO rather than a whole segment,
// and dropping it is the mutation that pads every already-aligned canvas.
int64_t PadFor(int64_t segment, int64_t content_frames) {
  return (segment - content_frames % segment) % segment;
}

// `min(max(round(position / temporal_scale), 0), T - 1)` (dfr_pipeline.py:109).
//
// THE ONLY FLOATING-POINT EXPRESSION IN THIS FILE, and it is here because
// upstream's is. `round` in Python 3 is half-to-EVEN, which `std::round` is not
// — `std::round` is half-away-from-zero — so `std::nearbyint` under the default
// FE_TONEAREST is the mirror and `std::round` is not. On the shipped
// SEGMENT_CANDIDATES the quotient is always exact (every slot position is an odd
// multiple of S, S is 24 or 32, both divisible by the temporal scale 8), so no
// canvas this engine can build observes the difference. Mirrored exactly anyway:
// "unobservable on the values we ship" is a statement about SEGMENT_CANDIDATES,
// not about this function, and a candidate added upstream would silently change
// which latent frame seeds every slot.
int64_t SlotSeedIndex(int64_t position, int64_t temporal_scale, int64_t frames) {
  const double quotient =
      static_cast<double>(position) / static_cast<double>(temporal_scale);
  int64_t index = static_cast<int64_t>(std::nearbyint(quotient));
  index = std::max<int64_t>(index, 0);
  index = std::min<int64_t>(index, frames - 1);
  return index;
}

// `_owned_segment_counts` (dfr_layout.py:93-100): `divmod` then `base + (1 if
// index < remainder else 0)` — the LARGEST runs first, which is what makes the
// partition deterministic rather than merely balanced.
std::vector<int64_t> OwnedSegmentCounts(int64_t n_segments, int64_t num_tiles) {
  if (n_segments < 1) {
    Refuse("n_segments must be >= 1, got " + std::to_string(n_segments) +
           " (dfr_layout.py:95-96)");
  }
  if (num_tiles < 1) {
    Refuse("num_tiles must be >= 1, got " + std::to_string(num_tiles) +
           " (dfr_layout.py:97-98)");
  }
  const int64_t base = n_segments / num_tiles;
  const int64_t remainder = n_segments % num_tiles;
  std::vector<int64_t> counts;
  counts.reserve(static_cast<size_t>(num_tiles));
  for (int64_t index = 0; index < num_tiles; ++index) {
    counts.push_back(base + (index < remainder ? 1 : 0));
  }
  return counts;
}

// `_build_tile` (dfr_layout.py:103-134). One window owning segments
// `[own_lo, own_hi)`, preceded by `lead_segments` of lead-in.
Ltx2DfrTileRange BuildTile(const std::vector<int64_t>& boundaries, int64_t own_lo, int64_t own_hi,
                           int64_t lead_segments, int64_t temporal_scale) {
  const int64_t window_lo = std::max<int64_t>(0, own_lo - lead_segments);

  Ltx2DfrTileRange tile;
  tile.pixel_start = boundaries[static_cast<size_t>(window_lo)];
  tile.pixel_end = boundaries[static_cast<size_t>(own_hi)];
  tile.latent_start = Ltx2DfrPixelToLatentIndex(tile.pixel_start, temporal_scale);
  tile.latent_end_exclusive = Ltx2DfrPixelToLatentIndex(tile.pixel_end, temporal_scale) + 1;

  // dfr_layout.py:118-122. Handover is exactly at the shared keyframe: the
  // PREVIOUS tile keeps the seam latent — the 8-pixel-frame token that ends on
  // the KF mark — and this tile resumes strictly after it, so the prefix drops
  // the lead-in PLUS that seam latent. The `+1` is the whole of the seam
  // handover and it is conditioned on `own_lo > 0` rather than on the lead-in
  // being non-zero, which are the same thing today and would not be if a future
  // caller passed `lead_segments = 0` for a middle tile.
  tile.drop_latent_prefix =
      Ltx2DfrPixelToLatentIndex(boundaries[static_cast<size_t>(own_lo)], temporal_scale) -
      tile.latent_start;
  if (own_lo > 0) tile.drop_latent_prefix += 1;

  // dfr_layout.py:129. `if boundaries[index]` drops the leading ZERO and only
  // that: frame 0 is not a keyframe, so the first tile's window start
  // contributes no anchor. Every other boundary is a seam position and is
  // non-zero by the strictly-increasing check in `Ltx2DfrTileRanges`.
  for (int64_t index = window_lo; index <= own_hi; ++index) {
    const int64_t position = boundaries[static_cast<size_t>(index)];
    if (position != 0) tile.anchor_kf_global.push_back(position);
  }

  // dfr_layout.py:130-132. The mid-segment positions this window INVENTS, one
  // per owned-or-lead-in segment: the integer midpoint of each consecutive
  // boundary pair. Note the range stops at `own_hi` exclusive, so a window with
  // a lead-in invents a slot for the lead-in segment too — which is why the
  // caller de-duplicates by position and lets the EARLIER tile's version win
  // (dfr_pipeline.py:518-525).
  for (int64_t index = window_lo; index < own_hi; ++index) {
    tile.slot_kf_global.push_back((boundaries[static_cast<size_t>(index)] +
                                   boundaries[static_cast<size_t>(index + 1)]) /
                                  2);
  }
  return tile;
}

}  // namespace

int64_t Ltx2DfrChooseSegmentLength(int64_t content_frames) {
  if (content_frames < 1) {
    Refuse("content_frames must be >= 1, got " + std::to_string(content_frames) +
           " (dfr_layout.py:44-45)");
  }

  // dfr_layout.py:50-56. `best` is seeded with candidate ZERO and displaced only
  // by a STRICTLY smaller pad, or by an equal pad on a LARGER segment. Both
  // halves are load-bearing and neither is a tie-break of convenience: 120
  // content frames pads 0 against 24 and 0 against 32, and the two answers place
  // 5 keyframes or 4.
  int64_t best = kLtx2DfrSegmentCandidates[0];
  int64_t best_pad = PadFor(best, content_frames);
  constexpr size_t kCount = sizeof(kLtx2DfrSegmentCandidates) / sizeof(int64_t);
  for (size_t i = 1; i < kCount; ++i) {
    const int64_t segment = kLtx2DfrSegmentCandidates[i];
    const int64_t pad = PadFor(segment, content_frames);
    if (pad < best_pad || (pad == best_pad && segment > best)) {
      best = segment;
      best_pad = pad;
    }
  }
  return best;
}

Ltx2DfrCanvas Ltx2DfrResolveCanvas(int64_t num_frames, int64_t temporal_scale) {
  if (num_frames < 1) {
    Refuse("num_frames must be >= 1, got " + std::to_string(num_frames) +
           " (dfr_layout.py:69-70)");
  }
  if (temporal_scale < 1) {
    Refuse("temporal_scale must be >= 1, got " + std::to_string(temporal_scale));
  }
  if ((num_frames - 1) % temporal_scale != 0) {
    Refuse("num_frames must satisfy (num_frames - 1) % " + std::to_string(temporal_scale) +
           " == 0 (got " + std::to_string(num_frames) +
           ") — dfr_layout.py:71-72. Every keyframe position DFR emits has to land on a latent "
           "border, and a count that does not satisfy this moves the terminal position off one. "
           "This is a REFUSAL where the engine's ordinary frames axis only FLOORS (#919): the "
           "envelope documents the rounding because `resolve_num_frames` returns an explicit "
           "count verbatim, and DFR is the one path that cannot live with it");
  }

  const int64_t content = num_frames - 1;
  if (content == 0) Refuse("the canvas needs at least 2 pixel frames (dfr_layout.py:75-76)");

  Ltx2DfrCanvas canvas;
  canvas.segment = Ltx2DfrChooseSegmentLength(content);
  const int64_t content_padded = content + PadFor(canvas.segment, content);

  // dfr_layout.py:80 — `range(1, content_padded // segment + 1)`, so positions
  // are `[S, 2S, ..., content_padded]` and the LAST one is `N' - 1`. Frame 0 is
  // excluded because it is already a single-pixel-frame token under causal
  // encoding, and the terminal frame is INCLUDED because `tile_ranges` requires
  // the last seam to be exactly `num_frames - 1` (:153-154).
  for (int64_t index = 1; index <= content_padded / canvas.segment; ++index) {
    canvas.positions.push_back(canvas.segment * index);
  }
  canvas.num_frames = content_padded + 1;
  return canvas;
}

int64_t Ltx2DfrPixelToLatentIndex(int64_t pixel_frame, int64_t temporal_scale) {
  if (pixel_frame < 0) {
    Refuse("pixel_frame must be >= 0, got " + std::to_string(pixel_frame) +
           " (dfr_layout.py:86-87)");
  }
  if (pixel_frame != 0 && pixel_frame % temporal_scale != 0) {
    Refuse("pixel_frame " + std::to_string(pixel_frame) + " is not on the x" +
           std::to_string(temporal_scale) + " latent border (dfr_layout.py:88-89)");
  }
  return pixel_frame / temporal_scale;
}

std::vector<Ltx2DfrTileRange> Ltx2DfrTileRanges(const std::vector<int64_t>& seam_positions,
                                                int64_t num_frames, int64_t num_tiles,
                                                int64_t temporal_scale, int64_t lead_segments) {
  if (num_frames < 2) {
    Refuse("num_frames must be >= 2, got " + std::to_string(num_frames) +
           " (dfr_layout.py:149-150)");
  }
  if (seam_positions.empty()) Refuse("seam_positions must be non-empty (dfr_layout.py:151-152)");
  if (seam_positions.back() != num_frames - 1) {
    Refuse("last seam must be the terminal frame " + std::to_string(num_frames - 1) + ", got " +
           std::to_string(seam_positions.back()) + " (dfr_layout.py:153-154)");
  }
  if (lead_segments < 1) {
    Refuse("lead_segments must be >= 1, got " + std::to_string(lead_segments) +
           " (dfr_layout.py:155-156)");
  }

  // dfr_layout.py:158 — the boundary run is `[0, *seam_positions]`, so index 0
  // is frame 0 and every later index is a seam.
  std::vector<int64_t> boundaries;
  boundaries.reserve(seam_positions.size() + 1);
  boundaries.push_back(0);
  for (int64_t position : seam_positions) boundaries.push_back(position);

  // dfr_layout.py:159-166. Three checks over each span, and the third is the one
  // a reader is most likely to drop: a segment under TWO latent frames cannot
  // carry a tile lead-in, because the lead-in's whole job is to put the window's
  // first two latents inside the discarded prefix.
  for (size_t index = 1; index < boundaries.size(); ++index) {
    const int64_t span = boundaries[index] - boundaries[index - 1];
    if (span <= 0) {
      Refuse("seam_positions must be strictly increasing, got " + Join(seam_positions) +
             " (dfr_layout.py:161-162)");
    }
    if (span % temporal_scale != 0) {
      Refuse("segment span " + std::to_string(span) + " is not a multiple of temporal scale " +
             std::to_string(temporal_scale) + " (dfr_layout.py:163-164)");
    }
    if (span / temporal_scale < 2) {
      Refuse("segment span " + std::to_string(span) +
             " is under 2 latent frames, too short to carry a tile lead-in "
             "(dfr_layout.py:165-166)");
    }
  }

  const int64_t n_segments = static_cast<int64_t>(boundaries.size()) - 1;
  // dfr_layout.py:171 — `min(num_tiles, n_segments)`. CLAMPED rather than
  // refused, which is what makes a 2-segment canvas legal at round 2 (which asks
  // for 4 tiles). A port that refused here would fail every short DFR render at
  // the second round and nowhere else.
  const std::vector<int64_t> counts =
      OwnedSegmentCounts(n_segments, std::min<int64_t>(num_tiles, n_segments));

  std::vector<Ltx2DfrTileRange> tiles;
  tiles.reserve(counts.size());
  int64_t own_lo = 0;
  for (size_t index = 0; index < counts.size(); ++index) {
    // dfr_layout.py:177 — the FIRST tile takes no lead-in. `max(0, own_lo -
    // lead)` would clamp its window to the same place, but its
    // `drop_latent_prefix` would stop being 0 and the head of the clip would be
    // truncated.
    tiles.push_back(BuildTile(boundaries, own_lo, own_lo + counts[index],
                              index > 0 ? lead_segments : 0, temporal_scale));
    own_lo += counts[index];
  }
  return tiles;
}

Ltx2LatentVolume Ltx2DfrStitchTileLatents(const std::vector<Ltx2LatentVolume>& tile_latents,
                                          const std::vector<Ltx2DfrTileRange>& ranges) {
  if (tile_latents.size() != ranges.size()) {
    Refuse("expected " + std::to_string(ranges.size()) + " tile latents, got " +
           std::to_string(tile_latents.size()) + " (dfr_layout.py:190-191)");
  }
  if (tile_latents.empty()) Refuse("tile_latents must be non-empty (dfr_layout.py:192-193)");

  Ltx2LatentVolume out;
  out.batch = tile_latents.front().batch;
  out.channels = tile_latents.front().channels;
  out.height = tile_latents.front().height;
  out.width = tile_latents.front().width;
  out.frames = 0;

  for (size_t i = 0; i < tile_latents.size(); ++i) {
    const Ltx2LatentVolume& latent = tile_latents[i];
    const Ltx2DfrTileRange& tile = ranges[i];
    const int64_t expected_t = tile.latent_end_exclusive - tile.latent_start;
    if (latent.frames != expected_t) {
      Refuse("tile latent T=" + std::to_string(latent.frames) + " != expected " +
             std::to_string(expected_t) + " for range [" + std::to_string(tile.latent_start) +
             ", " + std::to_string(tile.latent_end_exclusive) + ") (dfr_layout.py:199-204)");
    }
    if (tile.drop_latent_prefix < 0 || tile.drop_latent_prefix >= latent.frames) {
      Refuse("drop_latent_prefix=" + std::to_string(tile.drop_latent_prefix) +
             " invalid for tile T=" + std::to_string(latent.frames) + " (dfr_layout.py:205-206)");
    }
    // Upstream gets this from `torch.cat`, which raises on a mismatched
    // non-concatenated dimension. Here the volume is a flat buffer plus a
    // header, so a tile at the wrong spatial size would be memcpy'd into a
    // volume whose header describes something else — a correctly sized buffer
    // holding a different picture, which nothing downstream can see.
    if (latent.batch != out.batch || latent.channels != out.channels ||
        latent.height != out.height || latent.width != out.width) {
      Refuse("tile " + std::to_string(i) + " is " + std::to_string(latent.batch) + "x" +
             std::to_string(latent.channels) + "x*x" + std::to_string(latent.height) + "x" +
             std::to_string(latent.width) + " but tile 0 is " + std::to_string(out.batch) + "x" +
             std::to_string(out.channels) + "x*x" + std::to_string(out.height) + "x" +
             std::to_string(out.width) +
             "; `torch.cat` (dfr_layout.py:208) concatenates along T only and raises on any "
             "other axis");
    }
    if (static_cast<int64_t>(latent.data.size()) != latent.elems()) {
      Refuse("tile " + std::to_string(i) + " carries " + std::to_string(latent.data.size()) +
             " values for a " + std::to_string(latent.elems()) + "-element volume");
    }
    out.frames += latent.frames - tile.drop_latent_prefix;
  }

  // [B, C, T, H, W] row-major, so the per-(batch, channel) plane of one tile is
  // contiguous and the concatenation along T is a per-plane copy rather than a
  // single append. Writing it as one append would produce a correctly sized
  // buffer with the channels interleaved wrong.
  out.data.assign(static_cast<size_t>(out.elems()), 0.0F);
  const int64_t plane = out.height * out.width;
  int64_t frame_cursor = 0;
  for (size_t i = 0; i < tile_latents.size(); ++i) {
    const Ltx2LatentVolume& latent = tile_latents[i];
    const int64_t drop = ranges[i].drop_latent_prefix;
    const int64_t kept = latent.frames - drop;
    for (int64_t b = 0; b < out.batch; ++b) {
      for (int64_t c = 0; c < out.channels; ++c) {
        const size_t src = static_cast<size_t>(((b * out.channels + c) * latent.frames + drop) *
                                               plane);
        const size_t dst = static_cast<size_t>(
            ((b * out.channels + c) * out.frames + frame_cursor) * plane);
        std::copy(latent.data.begin() + static_cast<ptrdiff_t>(src),
                  latent.data.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(kept * plane)),
                  out.data.begin() + static_cast<ptrdiff_t>(dst));
      }
    }
    frame_cursor += kept;
  }
  return out;
}

std::vector<int64_t> Ltx2DfrRemapPositionsToLocal(const std::vector<int64_t>& positions,
                                                  int64_t pixel_start) {
  std::vector<int64_t> local;
  local.reserve(positions.size());
  for (int64_t position : positions) local.push_back(position - pixel_start);
  return local;
}

Ltx2LatentVolume Ltx2DfrSlotInitialsFromVideo(const Ltx2LatentVolume& video,
                                              const std::vector<int64_t>& positions,
                                              int64_t temporal_scale) {
  if (positions.empty()) Refuse("slot initials need at least one position");
  if (video.frames < 1) Refuse("the tile latent carries no frames to seed slots from");
  if (temporal_scale < 1) {
    Refuse("temporal_scale must be >= 1, got " + std::to_string(temporal_scale));
  }
  if (static_cast<int64_t>(video.data.size()) != video.elems()) {
    Refuse("the tile latent carries " + std::to_string(video.data.size()) + " values for a " +
           std::to_string(video.elems()) + "-element volume");
  }

  Ltx2LatentVolume out;
  out.batch = video.batch;
  out.channels = video.channels;
  out.frames = static_cast<int64_t>(positions.size());
  out.height = video.height;
  out.width = video.width;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0F);

  const int64_t plane = video.height * video.width;
  for (size_t k = 0; k < positions.size(); ++k) {
    const int64_t index = SlotSeedIndex(positions[k], temporal_scale, video.frames);
    for (int64_t b = 0; b < out.batch; ++b) {
      for (int64_t c = 0; c < out.channels; ++c) {
        const size_t src =
            static_cast<size_t>(((b * video.channels + c) * video.frames + index) * plane);
        const size_t dst = static_cast<size_t>(
            ((b * out.channels + c) * out.frames + static_cast<int64_t>(k)) * plane);
        std::copy(video.data.begin() + static_cast<ptrdiff_t>(src),
                  video.data.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(plane)),
                  out.data.begin() + static_cast<ptrdiff_t>(dst));
      }
    }
  }
  return out;
}

Ltx2DfrCarryForward Ltx2DfrMergeCarryForwardKeyframes(const std::vector<int64_t>& anchor_positions,
                                                      const Ltx2LatentVolume* anchor_latents,
                                                      const std::vector<int64_t>& slot_positions,
                                                      const Ltx2LatentVolume* slot_latents) {
  // dfr_pipeline.py:123-135. A `std::map` rather than a hash map because
  // upstream sorts the keys at :138 and an ordered container makes that the
  // container's property instead of a step a later edit can drop.
  std::map<int64_t, std::pair<const Ltx2LatentVolume*, int64_t>> by_position;

  // ANCHORS FIRST, SLOTS SECOND, and that order is upstream's (:125-126). It is
  // not an arbitrary tie-break: at a position carrying both, the SLOT was
  // denoised this round and the anchor is a still carried from the last one, so
  // the slot must win. Reversing the two loops keeps every shape and every count
  // identical and silently carries stale content forward.
  const std::pair<const std::vector<int64_t>*, const Ltx2LatentVolume*> sources[] = {
      {&anchor_positions, anchor_latents}, {&slot_positions, slot_latents}};
  const char* labels[] = {"anchor", "slot"};

  for (int which = 0; which < 2; ++which) {
    const std::vector<int64_t>& positions = *sources[which].first;
    const Ltx2LatentVolume* latents = sources[which].second;
    if (positions.empty()) continue;
    if (latents == nullptr) {
      Refuse(std::string("missing ") + labels[which] +
             " keyframe latents for carry-forward merge (dfr_pipeline.py:130-131)");
    }
    if (latents->frames != static_cast<int64_t>(positions.size())) {
      Refuse(std::string(labels[which]) + " latents K=" + std::to_string(latents->frames) +
             " != " + std::to_string(positions.size()) + " positions (dfr_pipeline.py:132-133)");
    }
    for (size_t index = 0; index < positions.size(); ++index) {
      by_position[positions[index]] = {latents, static_cast<int64_t>(index)};
    }
  }

  if (by_position.empty()) {
    Refuse("carry-forward keyframe bag is empty (dfr_pipeline.py:136-137)");
  }

  const Ltx2LatentVolume* any = by_position.begin()->second.first;
  Ltx2DfrCarryForward out;
  out.keyframes.batch = any->batch;
  out.keyframes.channels = any->channels;
  out.keyframes.height = any->height;
  out.keyframes.width = any->width;
  out.keyframes.frames = static_cast<int64_t>(by_position.size());
  out.keyframes.data.assign(static_cast<size_t>(out.keyframes.elems()), 0.0F);

  const int64_t plane = out.keyframes.height * out.keyframes.width;
  int64_t slot = 0;
  for (const auto& entry : by_position) {
    out.positions.push_back(entry.first);
    const Ltx2LatentVolume* source = entry.second.first;
    const int64_t index = entry.second.second;
    if (source->channels != out.keyframes.channels || source->height != out.keyframes.height ||
        source->width != out.keyframes.width || source->batch != out.keyframes.batch) {
      Refuse("carry-forward keyframes disagree on shape; `torch.cat` (dfr_pipeline.py:139) "
             "concatenates along K only");
    }
    for (int64_t b = 0; b < out.keyframes.batch; ++b) {
      for (int64_t c = 0; c < out.keyframes.channels; ++c) {
        const size_t src =
            static_cast<size_t>(((b * source->channels + c) * source->frames + index) * plane);
        const size_t dst = static_cast<size_t>(
            ((b * out.keyframes.channels + c) * out.keyframes.frames + slot) * plane);
        std::copy(source->data.begin() + static_cast<ptrdiff_t>(src),
                  source->data.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(plane)),
                  out.keyframes.data.begin() + static_cast<ptrdiff_t>(dst));
      }
    }
    ++slot;
  }
  return out;
}

int64_t Ltx2DfrTargetFrames(int64_t requested_frames, int64_t rounds) {
  if (rounds < 0 || rounds > kLtx2DfrMaxTemporalRounds) {
    Refuse("temporal_upsample_rounds must be 0, 1, or 2, got " + std::to_string(rounds) +
           " (dfr_pipeline.py:284-285)");
  }
  if (requested_frames < 1) {
    Refuse("requested_frames must be >= 1, got " + std::to_string(requested_frames));
  }
  return (requested_frames - 1) * (int64_t{1} << rounds) + 1;
}

}  // namespace vllm
