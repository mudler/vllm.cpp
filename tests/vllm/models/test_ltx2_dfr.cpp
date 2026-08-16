// LTX-2.5 DFR CANVAS LAYOUT — gated against EXECUTED upstream.
//
// Row LTX25-DFR-PIPELINE, issue #986, spec .agents/specs/ltx25-dfr-pipeline.md.
// Goldens: tests/vllm/models/ltx2_dfr_goldens.inc, produced by
// scripts/gen-ltx2-dfr-goldens.py running upstream's own `dfr_layout` module and
// the three `dfr_pipeline` helpers.
//
// WHY THIS SUITE COMPARES VECTORS AND NOT PROPERTIES. Every quantity here is an
// index. A port that resolves the segment tie the other way, drops the seam
// handover, or keeps boundary 0 as an anchor still returns a well-formed
// TileRange, still stitches to a plausible latent, and still renders a clip of
// the right length. There is no NaN to catch, no shape to disagree, and no
// magnitude to bound. So each case asserts the EXACT values upstream returned,
// element by element, and the few properties that are checked are checked BESIDE
// the values rather than instead of them.
//
// THE LOCAL HALF. `.agents/specs/ltx25-generated-keyframes.md` section 4a records
// the trap this campaign paid for: a suite that asserts only upstream symbol
// names stays green through the exact event that falsifies what it is checking,
// because no change to this tree can move an upstream name. The goldens here are
// upstream VALUES rather than names, and the two cases at the bottom pin LOCAL
// facts — the header's own constants, and the refusals this port raises — so a
// local edit that diverges from the golden set is separated from one that does
// not.
#include "vllm/model_executor/models/ltx2_dfr.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ltx2_dfr_goldens.inc"
#include "vllm/model_executor/models/ltx2_conditioning.h"

using namespace vllm;

namespace {

// The goldens are flattened with per-case counts, because the bags have
// different lengths per tile and one interleaved stream would let an off-by-one
// in one bag read as a correct value in another.
std::vector<int64_t> Slice(const int64_t* flat, int64_t offset, int64_t count) {
  return std::vector<int64_t>(flat + offset, flat + offset + count);
}

// A latent volume whose every element encodes its own coordinates, so a
// readback names the frame it came from instead of digesting it. Mirrors the
// generator's fill exactly: `global_latent_index * 1000 + channel * 10 + (h * W + w)`.
Ltx2LatentVolume IdentityVolume(int64_t channels, int64_t frames, int64_t height, int64_t width,
                                int64_t first_global_frame) {
  Ltx2LatentVolume v;
  v.batch = 1;
  v.channels = channels;
  v.frames = frames;
  v.height = height;
  v.width = width;
  v.data.assign(static_cast<size_t>(v.elems()), 0.0F);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t f = 0; f < frames; ++f) {
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const size_t index =
              static_cast<size_t>(((c * frames + f) * height + h) * width + w);
          v.data[index] = static_cast<float>((first_global_frame + f) * 1000 + c * 10 +
                                             (h * width + w));
        }
      }
    }
  }
  return v;
}

// A target small enough to read by hand: 2 channels, 3 latent frames, 1x2
// spatial. One latent frame is therefore TWO tokens, which is deliberately not
// one — at one token per frame every wrong `tokens_per_keyframe` is also one.
Ltx2VideoLatentShape SlotTarget() {
  Ltx2VideoLatentShape shape;
  shape.batch = 1;
  shape.channels = 2;
  shape.frames = 3;
  shape.height = 1;
  shape.width = 2;
  return shape;
}

std::string ThrownMessage(void (*body)()) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("ltx2 dfr the oracle is the pinned upstream tree") {
  // Asserted rather than assumed: a regeneration against a different checkout
  // would otherwise replace the oracle silently, and every value below would
  // still agree with itself.
  CHECK(std::string(vllm_test::kLtx2DfrUpstreamRevision) ==
        "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca");
  // VIDEO_SCALE_FACTORS.time, read off the executed module. Our default comes
  // from `Ltx2ScaleFactors`, and the two must agree or every latent index below
  // is computed on a different grid.
  CHECK(vllm_test::kLtx2DfrTemporalScale == Ltx2ScaleFactors{}.time);
}

TEST_CASE("ltx2 dfr the segment grid reproduces upstream") {
  SUBCASE("SEGMENT_CANDIDATES and TILE_LEAD_SEGMENTS match the executed module") {
    // A LOCAL fact checked against an upstream one. Our candidates live in
    // `ltx2_dfr.h`; upstream's were read off its own tuple. A candidate added
    // upstream fails here rather than silently changing every canvas.
    constexpr int64_t kOurs = sizeof(kLtx2DfrSegmentCandidates) / sizeof(int64_t);
    REQUIRE(kOurs == vllm_test::kLtx2DfrSegmentCandidatesGoldenCount);
    for (int64_t i = 0; i < kOurs; ++i) {
      CHECK(kLtx2DfrSegmentCandidates[i] == vllm_test::kLtx2DfrSegmentCandidatesGolden[i]);
    }
    CHECK(kLtx2DfrTileLeadSegments == vllm_test::kLtx2DfrTileLeadSegmentsGolden);
  }

  SUBCASE("the golden table actually CONTAINS a tie") {
    // The tie is the whole reason this table exists, and "we included one" is an
    // intent rather than a fact. Derived from the emitted pads, so a future edit
    // to the case list that drops the tie fails HERE, where the reason is
    // legible, instead of leaving the tie-break arm untested and green.
    int64_t ties = 0;
    const int64_t cands = vllm_test::kLtx2DfrSegmentCandidateCount;
    for (int64_t c = 0; c < vllm_test::kLtx2DfrSegmentContentCount; ++c) {
      bool all_equal = true;
      for (int64_t k = 1; k < cands; ++k) {
        if (vllm_test::kLtx2DfrSegmentPads[c * cands + k] !=
            vllm_test::kLtx2DfrSegmentPads[c * cands]) {
          all_equal = false;
        }
      }
      if (all_equal) ++ties;
    }
    CHECK(ties > 0);
  }

  SUBCASE("choose_segment_length matches upstream on every case") {
    REQUIRE(vllm_test::kLtx2DfrSegmentContentCount == vllm_test::kLtx2DfrSegmentChosenCount);
    for (int64_t i = 0; i < vllm_test::kLtx2DfrSegmentContentCount; ++i) {
      const int64_t content = vllm_test::kLtx2DfrSegmentContent[i];
      INFO("content_frames = ", content);
      CHECK(Ltx2DfrChooseSegmentLength(content) == vllm_test::kLtx2DfrSegmentChosen[i]);
    }
  }

  SUBCASE("a content count below 1 is refused, as upstream refuses it") {
    CHECK_THROWS_AS(Ltx2DfrChooseSegmentLength(0), std::exception);
    CHECK_THROWS_AS(Ltx2DfrChooseSegmentLength(-1), std::exception);
  }
}

TEST_CASE("ltx2 dfr resolve_canvas reproduces upstream") {
  SUBCASE("every case matches, frames, segment AND positions") {
    int64_t cursor = 0;
    for (int64_t i = 0; i < vllm_test::kLtx2DfrCanvasRequestCount; ++i) {
      const int64_t request = vllm_test::kLtx2DfrCanvasRequest[i];
      INFO("num_frames = ", request);
      const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(request);
      CHECK(canvas.num_frames == vllm_test::kLtx2DfrCanvasFrames[i]);
      CHECK(canvas.segment == vllm_test::kLtx2DfrCanvasSegment[i]);

      const int64_t count = vllm_test::kLtx2DfrCanvasPositionCount[i];
      REQUIRE(static_cast<int64_t>(canvas.positions.size()) == count);
      CHECK(canvas.positions == Slice(vllm_test::kLtx2DfrCanvasPositions, cursor, count));
      cursor += count;
    }
    // The flattened stream was fully consumed. Without this, a case whose count
    // is short leaves a tail nothing reads and every CHECK above still passes.
    CHECK(cursor == vllm_test::kLtx2DfrCanvasPositionsCount);
  }

  SUBCASE("frame 0 is excluded and the terminal frame is included") {
    // The two halves of upstream's docstring at dfr_layout.py:66-68, asserted as
    // properties BESIDE the exact vectors above rather than instead of them.
    // `tile_ranges` refuses a seam run whose last element is not `N' - 1`, so a
    // port that dropped the terminal position would fail later and elsewhere.
    for (int64_t i = 0; i < vllm_test::kLtx2DfrCanvasRequestCount; ++i) {
      const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(vllm_test::kLtx2DfrCanvasRequest[i]);
      REQUIRE(!canvas.positions.empty());
      CHECK(canvas.positions.front() != 0);
      CHECK(canvas.positions.back() == canvas.num_frames - 1);
    }
  }

  SUBCASE("a frame count off the x8 border is REFUSED, not floored") {
    // The asymmetry with the resolution envelope (#919) is deliberate and it is
    // the reason this subcase exists: the engine's ordinary frames axis FLOORS,
    // documented as such, while DFR refuses. A port that reused the engine's
    // polarity here would move every keyframe position off its latent border and
    // still render.
    const std::string message = ThrownMessage([] { Ltx2DfrResolveCanvas(100); });
    CHECK(message.find("dfr_layout.py:71-72") != std::string::npos);
    CHECK(message.find("#919") != std::string::npos);
    CHECK_THROWS_AS(Ltx2DfrResolveCanvas(0), std::exception);
    CHECK_THROWS_AS(Ltx2DfrResolveCanvas(1), std::exception);  // content == 0
  }
}

TEST_CASE("ltx2 dfr pixel_to_latent_index reproduces upstream") {
  CHECK(Ltx2DfrPixelToLatentIndex(0) == 0);
  CHECK(Ltx2DfrPixelToLatentIndex(8) == 1);
  CHECK(Ltx2DfrPixelToLatentIndex(120) == 15);
  // Zero is legal and every OTHER value must sit on the border. Upstream's own
  // asymmetry (dfr_layout.py:88), and a port that dropped the `pixel_frame != 0`
  // guard would refuse the first tile of every canvas.
  CHECK_THROWS_AS(Ltx2DfrPixelToLatentIndex(1), std::exception);
  CHECK_THROWS_AS(Ltx2DfrPixelToLatentIndex(-8), std::exception);
}

TEST_CASE("ltx2 dfr tile_ranges reproduces upstream") {
  SUBCASE("every tile of every case matches on all five scalars and both bags") {
    int64_t tile_cursor = 0, seam_cursor = 0, anchor_cursor = 0, slot_cursor = 0;
    for (int64_t c = 0; c < vllm_test::kLtx2DfrTileCaseRequestCount; ++c) {
      const int64_t frames = vllm_test::kLtx2DfrTileCaseFrames[c];
      const int64_t num_tiles = vllm_test::kLtx2DfrTileCaseNumTiles[c];
      const int64_t seam_count = vllm_test::kLtx2DfrTileCaseSeamCount[c];
      const std::vector<int64_t> seams =
          Slice(vllm_test::kLtx2DfrTileCaseSeams, seam_cursor, seam_count);
      seam_cursor += seam_count;

      INFO("request = ", vllm_test::kLtx2DfrTileCaseRequest[c],
           " rounds = ", vllm_test::kLtx2DfrTileCaseRounds[c], " frames = ", frames,
           " num_tiles = ", num_tiles);

      const std::vector<Ltx2DfrTileRange> tiles = Ltx2DfrTileRanges(seams, frames, num_tiles);
      REQUIRE(static_cast<int64_t>(tiles.size()) == vllm_test::kLtx2DfrTileCaseTileCount[c]);

      for (const Ltx2DfrTileRange& tile : tiles) {
        INFO("tile index = ", tile_cursor);
        CHECK(tile.pixel_start == vllm_test::kLtx2DfrTilePixelStart[tile_cursor]);
        CHECK(tile.pixel_end == vllm_test::kLtx2DfrTilePixelEnd[tile_cursor]);
        CHECK(tile.latent_start == vllm_test::kLtx2DfrTileLatentStart[tile_cursor]);
        CHECK(tile.latent_end_exclusive ==
              vllm_test::kLtx2DfrTileLatentEndExclusive[tile_cursor]);
        CHECK(tile.drop_latent_prefix ==
              vllm_test::kLtx2DfrTileDropLatentPrefix[tile_cursor]);

        const int64_t anchors = vllm_test::kLtx2DfrTileAnchorCount[tile_cursor];
        REQUIRE(static_cast<int64_t>(tile.anchor_kf_global.size()) == anchors);
        CHECK(tile.anchor_kf_global ==
              Slice(vllm_test::kLtx2DfrTileAnchors, anchor_cursor, anchors));
        anchor_cursor += anchors;

        const int64_t slots = vllm_test::kLtx2DfrTileSlotCount[tile_cursor];
        REQUIRE(static_cast<int64_t>(tile.slot_kf_global.size()) == slots);
        CHECK(tile.slot_kf_global == Slice(vllm_test::kLtx2DfrTileSlots, slot_cursor, slots));
        slot_cursor += slots;
        ++tile_cursor;
      }
    }
    // Every flattened stream fully consumed — the floor that stops a short count
    // from reporting a pass over a prefix.
    CHECK(tile_cursor == vllm_test::kLtx2DfrTilePixelStartCount);
    CHECK(seam_cursor == vllm_test::kLtx2DfrTileCaseSeamsCount);
    CHECK(anchor_cursor == vllm_test::kLtx2DfrTileAnchorsCount);
    CHECK(slot_cursor == vllm_test::kLtx2DfrTileSlotsCount);
  }

  SUBCASE("frame 0 is never an anchor, and the first tile drops no prefix") {
    // The two mutations M2 and M4 target exactly these, and the exact vectors
    // above already carry them. Stated separately because the vectors say WHAT
    // and this says WHY, and a later edit to the case list must not be able to
    // remove the property along with the case.
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(121);
    std::vector<int64_t> seams;
    for (int64_t p : canvas.positions) seams.push_back(2 * p);
    const int64_t frames = 2 * (canvas.num_frames - 1) + 1;
    const std::vector<Ltx2DfrTileRange> tiles = Ltx2DfrTileRanges(seams, frames, 2);
    REQUIRE(tiles.size() >= 2);
    for (const Ltx2DfrTileRange& tile : tiles) {
      for (int64_t anchor : tile.anchor_kf_global) CHECK(anchor != 0);
    }
    CHECK(tiles.front().pixel_start == 0);
    CHECK(tiles.front().drop_latent_prefix == 0);
    // Every non-first tile drops at least the seam latent.
    for (size_t i = 1; i < tiles.size(); ++i) CHECK(tiles[i].drop_latent_prefix >= 1);
  }

  SUBCASE("the first tile's window is lead-in INVARIANT, which is why M4 stays green") {
    // This subcase exists because the header claimed the opposite and a mutation
    // refuted it. Removing `index > 0 ? lead : 0` from `Ltx2DfrTileRanges` left
    // this suite at 10/10, 562 assertions, exit 0 — and the reason is not a
    // coverage hole. For the first tile `own_lo == 0`, so `max(0, 0 - lead)` is 0
    // for every lead and `drop_latent_prefix` is `0 - 0` with no `+1`. The
    // conditional is REDUNDANT with the clamp beside it.
    //
    // Verified the same way against executed upstream: `_build_tile` on the
    // first tile with lead 0, 1 and 5 returns three TileRanges that compare
    // equal. Gated here so the header's corrected paragraph is a checked claim
    // rather than a second confident sentence, and so a future change that makes
    // the lead-in matter for tile 0 fails HERE with the reason attached.
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(121);
    std::vector<int64_t> seams;
    for (int64_t p : canvas.positions) seams.push_back(2 * p);
    const int64_t frames = 2 * (canvas.num_frames - 1) + 1;

    const std::vector<Ltx2DfrTileRange> lead1 = Ltx2DfrTileRanges(seams, frames, 2, 8, 1);
    const std::vector<Ltx2DfrTileRange> lead5 = Ltx2DfrTileRanges(seams, frames, 2, 8, 5);
    REQUIRE(!lead1.empty());
    REQUIRE(!lead5.empty());
    CHECK(lead1.front().pixel_start == lead5.front().pixel_start);
    CHECK(lead1.front().latent_start == lead5.front().latent_start);
    CHECK(lead1.front().drop_latent_prefix == lead5.front().drop_latent_prefix);
    CHECK(lead1.front().drop_latent_prefix == 0);
    // The positive control, in the same walk: the lead-in DOES move a non-first
    // tile, so the equality above is a property of tile 0 and not of a parameter
    // this port silently ignores. Without this, a `Ltx2DfrTileRanges` that
    // dropped `lead_segments` entirely would satisfy every CHECK above.
    REQUIRE(lead1.size() >= 2);
    REQUIRE(lead5.size() >= 2);
    CHECK(lead1[1].pixel_start != lead5[1].pixel_start);
  }

  SUBCASE("the partition is GAPLESS and covers the canvas exactly") {
    // What `stitch_tile_latents` depends on and what no single tile can show. A
    // tile set that overlapped or left a hole would still stitch to a volume;
    // it would simply be the wrong length, or the right length holding a
    // duplicated span.
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(193);
    std::vector<int64_t> seams;
    for (int64_t p : canvas.positions) seams.push_back(2 * p);
    const int64_t frames = 2 * (canvas.num_frames - 1) + 1;
    const std::vector<Ltx2DfrTileRange> tiles = Ltx2DfrTileRanges(seams, frames, 4);

    int64_t stitched = 0;
    for (const Ltx2DfrTileRange& tile : tiles) {
      stitched += (tile.latent_end_exclusive - tile.latent_start) - tile.drop_latent_prefix;
    }
    CHECK(stitched == (frames - 1) / Ltx2ScaleFactors{}.time + 1);
  }

  SUBCASE("num_tiles is CLAMPED to the segment count, not refused") {
    // dfr_layout.py:171. This is what makes a short canvas legal at round 2,
    // which asks for 4 tiles. A port that refused instead would fail every short
    // DFR render at the second round and nowhere else.
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(49);
    std::vector<int64_t> seams;
    for (int64_t p : canvas.positions) seams.push_back(2 * p);
    const int64_t frames = 2 * (canvas.num_frames - 1) + 1;
    const std::vector<Ltx2DfrTileRange> many = Ltx2DfrTileRanges(seams, frames, 64);
    CHECK(static_cast<int64_t>(many.size()) == static_cast<int64_t>(seams.size()));
  }

  SUBCASE("upstream's four refusals are mirrored") {
    const std::vector<int64_t> ok = {16, 32};
    CHECK_THROWS_AS(Ltx2DfrTileRanges(ok, 1, 2), std::exception);          // :149-150
    CHECK_THROWS_AS(Ltx2DfrTileRanges({}, 33, 2), std::exception);         // :151-152
    CHECK_THROWS_AS(Ltx2DfrTileRanges(ok, 99, 2), std::exception);         // :153-154
    CHECK_THROWS_AS(Ltx2DfrTileRanges(ok, 33, 2, 8, 0), std::exception);   // :155-156
    // Strictly increasing (:161-162), on the border (:163-164), and at least two
    // latent frames per segment (:165-166). The last one is the check a reader
    // is most likely to drop, because a 8-pixel-frame segment is legal-looking.
    CHECK_THROWS_AS(Ltx2DfrTileRanges({32, 16, 32}, 33, 2), std::exception);
    CHECK_THROWS_AS(Ltx2DfrTileRanges({20, 32}, 33, 2), std::exception);
    CHECK_THROWS_AS(Ltx2DfrTileRanges({8, 32}, 33, 2), std::exception);
  }
}

TEST_CASE("ltx2 dfr stitch_tile_latents reproduces upstream") {
  SUBCASE("the stitched volume is the canvas's own latent frames, in order") {
    const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(vllm_test::kLtx2DfrStitchRequest);
    std::vector<int64_t> seams;
    for (int64_t p : canvas.positions) seams.push_back(2 * p);
    const int64_t frames = 2 * (canvas.num_frames - 1) + 1;
    REQUIRE(frames == vllm_test::kLtx2DfrStitchFrames);
    const std::vector<Ltx2DfrTileRange> tiles = Ltx2DfrTileRanges(seams, frames, 2);

    std::vector<Ltx2LatentVolume> volumes;
    for (const Ltx2DfrTileRange& tile : tiles) {
      volumes.push_back(IdentityVolume(vllm_test::kLtx2DfrStitchChannels,
                                       tile.latent_end_exclusive - tile.latent_start,
                                       vllm_test::kLtx2DfrStitchHeight,
                                       vllm_test::kLtx2DfrStitchWidth, tile.latent_start));
    }

    const Ltx2LatentVolume stitched = Ltx2DfrStitchTileLatents(volumes, tiles);
    CHECK(stitched.frames == vllm_test::kLtx2DfrStitchOutFrames);
    CHECK(stitched.channels == vllm_test::kLtx2DfrStitchChannels);
    CHECK(stitched.height == vllm_test::kLtx2DfrStitchHeight);
    CHECK(stitched.width == vllm_test::kLtx2DfrStitchWidth);
    REQUIRE(static_cast<int64_t>(stitched.data.size()) ==
            vllm_test::kLtx2DfrStitchExpectedCount);
    // Element-wise and EXACT. The fill is an identity, so a value that differs
    // names the latent frame that landed in the wrong place rather than moving a
    // digest by an amount nobody can interpret.
    for (int64_t i = 0; i < vllm_test::kLtx2DfrStitchExpectedCount; ++i) {
      INFO("element ", i);
      CHECK(static_cast<int64_t>(stitched.data[static_cast<size_t>(i)]) ==
            vllm_test::kLtx2DfrStitchExpected[i]);
    }
  }

  SUBCASE("upstream's three refusals are mirrored, plus the one torch gives for free") {
    const std::vector<int64_t> seams = {16, 32};
    const std::vector<Ltx2DfrTileRange> tiles = Ltx2DfrTileRanges(seams, 33, 2);
    REQUIRE(tiles.size() == 2);

    CHECK_THROWS_AS(Ltx2DfrStitchTileLatents({}, tiles), std::exception);
    std::vector<Ltx2LatentVolume> one;
    one.push_back(IdentityVolume(1, tiles[0].latent_end_exclusive - tiles[0].latent_start, 1, 1,
                                 tiles[0].latent_start));
    CHECK_THROWS_AS(Ltx2DfrStitchTileLatents(one, tiles), std::exception);

    // A tile whose T disagrees with its own range (:199-204).
    std::vector<Ltx2LatentVolume> wrong_t;
    wrong_t.push_back(IdentityVolume(1, 1, 1, 1, 0));
    wrong_t.push_back(IdentityVolume(1, 1, 1, 1, 0));
    CHECK_THROWS_AS(Ltx2DfrStitchTileLatents(wrong_t, tiles), std::exception);

    // A tile at a different SPATIAL size. Upstream gets this refusal from
    // `torch.cat`, which concatenates along T only; here the volume is a flat
    // buffer plus a header, so without the check the bytes would be copied into
    // a volume whose header describes a different picture and nothing
    // downstream could see it.
    std::vector<Ltx2LatentVolume> mixed;
    for (size_t i = 0; i < tiles.size(); ++i) {
      mixed.push_back(IdentityVolume(1, tiles[i].latent_end_exclusive - tiles[i].latent_start,
                                     i == 0 ? 1 : 2, 1, tiles[i].latent_start));
    }
    CHECK_THROWS_AS(Ltx2DfrStitchTileLatents(mixed, tiles), std::exception);
  }
}

TEST_CASE("ltx2 dfr the slot seed picks the frame upstream picks") {
  SUBCASE("every position, including both clamps and all three half quotients") {
    Ltx2LatentVolume video;
    video.batch = 1;
    video.channels = 1;
    video.frames = vllm_test::kLtx2DfrSeedFrames;
    video.height = 1;
    video.width = 1;
    video.data.assign(static_cast<size_t>(video.frames), 0.0F);
    for (int64_t f = 0; f < video.frames; ++f) {
      video.data[static_cast<size_t>(f)] = static_cast<float>(f);
    }

    const std::vector<int64_t> positions(
        vllm_test::kLtx2DfrSeedPositions,
        vllm_test::kLtx2DfrSeedPositions + vllm_test::kLtx2DfrSeedPositionsCount);
    const Ltx2LatentVolume seeds =
        Ltx2DfrSlotInitialsFromVideo(video, positions, Ltx2ScaleFactors{}.time);

    REQUIRE(seeds.frames == vllm_test::kLtx2DfrSeedSelectedCount);
    for (int64_t k = 0; k < seeds.frames; ++k) {
      INFO("position ", positions[static_cast<size_t>(k)]);
      // The fill is the frame index itself, so this reads back WHICH latent
      // frame seeded the slot. 52, 60 and 68 have quotients 6.5, 7.5 and 8.5,
      // and the three of them separate half-to-even from half-away-from-zero
      // from a floor: no two of those rules agree on all three.
      CHECK(static_cast<int64_t>(seeds.data[static_cast<size_t>(k)]) ==
            vllm_test::kLtx2DfrSeedSelected[k]);
    }
  }
}

TEST_CASE("ltx2 dfr the carry-forward bag merges the way upstream merges it") {
  SUBCASE("positions are sorted and the SLOT wins a shared position") {
    const std::vector<int64_t> anchor_positions(
        vllm_test::kLtx2DfrMergeAnchorPositions,
        vllm_test::kLtx2DfrMergeAnchorPositions + vllm_test::kLtx2DfrMergeAnchorPositionsCount);
    const std::vector<int64_t> slot_positions(
        vllm_test::kLtx2DfrMergeSlotPositions,
        vllm_test::kLtx2DfrMergeSlotPositions + vllm_test::kLtx2DfrMergeSlotPositionsCount);

    Ltx2LatentVolume anchors;
    anchors.channels = 1;
    anchors.frames = static_cast<int64_t>(anchor_positions.size());
    anchors.height = 1;
    anchors.width = 1;
    for (int64_t p : anchor_positions) anchors.data.push_back(static_cast<float>(1000 + p));

    Ltx2LatentVolume slots;
    slots.channels = 1;
    slots.frames = static_cast<int64_t>(slot_positions.size());
    slots.height = 1;
    slots.width = 1;
    for (int64_t p : slot_positions) slots.data.push_back(static_cast<float>(2000 + p));

    const Ltx2DfrCarryForward merged = Ltx2DfrMergeCarryForwardKeyframes(
        anchor_positions, &anchors, slot_positions, &slots);

    REQUIRE(static_cast<int64_t>(merged.positions.size()) ==
            vllm_test::kLtx2DfrMergePositionsCount);
    for (int64_t i = 0; i < vllm_test::kLtx2DfrMergePositionsCount; ++i) {
      CHECK(merged.positions[static_cast<size_t>(i)] == vllm_test::kLtx2DfrMergePositions[i]);
      // Anchors are tagged `1000 + position` and slots `2000 + position`, so the
      // value at the SHARED position says which side won. Every shape and every
      // count is identical either way, and this is the only thing that moves.
      INFO("position ", merged.positions[static_cast<size_t>(i)]);
      CHECK(static_cast<int64_t>(merged.keyframes.data[static_cast<size_t>(i)]) ==
            vllm_test::kLtx2DfrMergeValues[i]);
    }
  }

  SUBCASE("an empty bag and a missing latent are refused") {
    Ltx2LatentVolume empty;
    CHECK_THROWS_AS(Ltx2DfrMergeCarryForwardKeyframes({}, nullptr, {}, nullptr), std::exception);
    CHECK_THROWS_AS(Ltx2DfrMergeCarryForwardKeyframes({8}, nullptr, {}, nullptr), std::exception);
    // K disagreeing with the position count (dfr_pipeline.py:132-133): the
    // merge would otherwise index past the latent it was handed.
    Ltx2LatentVolume one;
    one.channels = 1;
    one.frames = 1;
    one.height = 1;
    one.width = 1;
    one.data = {1.0F};
    CHECK_THROWS_AS(Ltx2DfrMergeCarryForwardKeyframes({8, 16}, &one, {}, nullptr),
                    std::exception);
  }
}

TEST_CASE("ltx2 dfr the caller's frame contract survives the padded canvas") {
  SUBCASE("every (requested, rounds) pair matches upstream's expression") {
    for (int64_t i = 0; i < vllm_test::kLtx2DfrTrimRequestedCount; ++i) {
      INFO("requested = ", vllm_test::kLtx2DfrTrimRequested[i],
           " rounds = ", vllm_test::kLtx2DfrTrimRounds[i]);
      CHECK(Ltx2DfrTargetFrames(vllm_test::kLtx2DfrTrimRequested[i],
                                vllm_test::kLtx2DfrTrimRounds[i]) ==
            vllm_test::kLtx2DfrTrimTarget[i]);
    }
  }

  SUBCASE("the target never exceeds the canvas the rounds actually produce") {
    // dfr_pipeline.py:535-536 raises when it does, and that raise is a statement
    // about this arithmetic rather than a defensive check: the canvas pads UP,
    // so the target must always be reachable. Derived here across the whole case
    // list so the invariant is measured rather than asserted once.
    for (int64_t i = 0; i < vllm_test::kLtx2DfrTrimRequestedCount; ++i) {
      const int64_t requested = vllm_test::kLtx2DfrTrimRequested[i];
      const int64_t rounds = vllm_test::kLtx2DfrTrimRounds[i];
      const Ltx2DfrCanvas canvas = Ltx2DfrResolveCanvas(requested);
      int64_t frames = canvas.num_frames;
      for (int64_t r = 0; r < rounds; ++r) frames = 2 * (frames - 1) + 1;
      INFO("requested = ", requested, " rounds = ", rounds, " canvas = ", frames);
      CHECK(Ltx2DfrTargetFrames(requested, rounds) <= frames);
      // And the trim lands on a latent boundary, which is what makes it a slice
      // rather than a resample.
      CHECK((Ltx2DfrTargetFrames(requested, rounds) - 1) % Ltx2ScaleFactors{}.time == 0);
    }
  }

  SUBCASE("a rounds count outside {0, 1, 2} is refused by name") {
    CHECK_THROWS_AS(Ltx2DfrTargetFrames(121, 3), std::exception);
    CHECK_THROWS_AS(Ltx2DfrTargetFrames(121, -1), std::exception);
    const std::string message = ThrownMessage([] { Ltx2DfrTargetFrames(121, 3); });
    CHECK(message.find("dfr_pipeline.py:284-285") != std::string::npos);
  }
}

TEST_CASE("ltx2 dfr the header's constants are the ones upstream carries") {
  // LOCAL facts, checked against upstream VALUES rather than upstream names.
  // Section 4a of .agents/specs/ltx25-generated-keyframes.md records why that
  // distinction is the difference between a gate and a decoration: an upstream
  // symbol name cannot move when this tree changes, so a suite built only on
  // names stays green through the event that falsifies it.
  CHECK(kLtx2DfrAnchorKeyframeStrength == doctest::Approx(vllm_test::kLtx2DfrAnchorStrengthGolden));
  CHECK(kLtx2DfrTemporalAncestralEta == doctest::Approx(vllm_test::kLtx2DfrTemporalEtaGolden));
  CHECK(kLtx2DfrMaxConditioningFps ==
        doctest::Approx(vllm_test::kLtx2DfrMaxConditioningFpsGolden));
  // The DFR rounds sampler is NOT the distilled stage-1 sampler. Both are
  // "ancestral Euler" and they differ by a factor of two in eta, which is a
  // difference no shape and no token count can see.
  CHECK(kLtx2DfrTemporalAncestralEta != kLtx2AncestralEta);
  CHECK(kLtx2DfrMaxTemporalRounds == 2);
}

// ───────────────────────────────────────────────────────────────────────────
// The generated keyframe SLOT items themselves (row LTX25-DFR-PIPELINE, #986).
//
// These sit in this file rather than in `test_ltx2_video` for one reason: the
// configuration they gate is NOT REACHABLE from a production entry point in this
// port yet, and dressing it up as an engine test would hide that.
//
// `test_ltx2_video`'s DFR case is the reachability proof and it enters through
// `LoadVideoEngine` / `Generate`. Two properties it CANNOT reach are gated here,
// and both greens below were MEASURED as mutations against that suite rather
// than assumed:
//
//   1. THE SEED. Slot content is supplied only from phase 1 onward — stage 2
//      seeds its slots with stage 1's, spatially upsampled
//      (dfr_pipeline.py:348, :364) — and the engine case runs phase 0 only,
//      where `initial_keyframes` is null and the slot tokens are zeros either
//      way. The mutation that stops the seed reaching `latent` leaves that suite
//      at 52 cases / 52 passed, exit 0, because on phase 0 it is a no-op.
//   2. THE LAYOUT vs THE TRAILING ASSUMPTION. The two readings differ only when
//      another appending item lands AFTER the slot item, and upstream has
//      exactly one: `VideoConditionByReferenceLatent` at
//      dfr_pipeline.py:366-373, appended after the slots when a detailing LoRA
//      is present. That arm is refused here and owed by #975, so today's engine
//      ALWAYS has the slots trailing and the two readings agree. The mutation
//      that replaces `layout.first_token` with "the last `num_tokens` tokens"
//      also leaves that suite at 52 / 52, exit 0.
//
// Neither green is a coverage hole in `test_ltx2_video`; both are statements
// about what this port can currently reach. They are gated here so the
// properties are checked NOW rather than discovered when #975 lands and quietly
// changes which tokens the readback returns.

TEST_CASE("ltx2 dfr generated keyframe slots append, mark, seed and read back") {
  const Ltx2VideoLatentShape target = SlotTarget();
  const Ltx2ScaleFactors factors;
  const double kFps = 24.0;
  const int64_t target_tokens = Ltx2VideoTokenCount(target, 1);
  Ltx2VideoLatentShape one = target;
  one.frames = 1;
  const int64_t per_frame = Ltx2VideoTokenCount(one, 1);
  // Deliberately NOT 1: a per-frame token count of 1 makes an off-by-one in
  // `tokens_per_keyframe` invisible, because every wrong answer is also 1.
  REQUIRE(per_frame == 2);

  auto fresh = [&]() {
    return Ltx2CreateVideoLatentState(target, 1, factors, kFps, /*causal_fix=*/true);
  };

  SUBCASE("the slot tokens are MARKED and the layout locates them") {
    Ltx2LatentState state = fresh();
    const std::vector<int64_t> positions = {8, 16};
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, positions,
                                               nullptr);

    CHECK(state.tokens == target_tokens + 2 * per_frame);
    CHECK(state.generated_keyframe_layout.applied());
    CHECK(state.generated_keyframe_layout.first_token == target_tokens);
    CHECK(state.generated_keyframe_layout.tokens_per_keyframe == per_frame);
    CHECK(state.generated_keyframe_layout.pixel_frame_indices == positions);

    // `denoise_mask = 1` on every new token (keyframe_slots.py:118-119), NOT
    // `1 - strength`. At mask 1 the noiser lerps from the slot `latent` and
    // IGNORES `clean_latent`, which is what makes a slot generated output rather
    // than guidance.
    for (int64_t t = target_tokens; t < state.tokens; ++t) {
      CHECK(state.mask[static_cast<size_t>(t)] == 1.0F);
    }
    // And `clean_latent` is zeros for them (`torch.zeros_like`, :139).
    for (int64_t t = target_tokens; t < state.tokens; ++t) {
      for (int64_t w = 0; w < state.width; ++w) {
        CHECK(state.clean[static_cast<size_t>(t * state.width + w)] == 0.0F);
      }
    }
    // MARKED — upstream's single `marked=True` call site (:121), and the only
    // thing separating a slot from an ordinary append.
    REQUIRE(static_cast<int64_t>(state.keyframes_mask.size()) == state.tokens);
    for (int64_t t = target_tokens; t < state.tokens; ++t) {
      CHECK(state.keyframes_mask[static_cast<size_t>(t)] == 1.0F);
    }
    // The target's own first latent frame stays marked and nothing between it
    // and the slots does. `_first_frame_keyframes_mask` marks it
    // unconditionally (tools.py:184-196), so a port that REBUILT the mask on
    // append instead of extending it would lose that and nothing about the
    // token count would show it.
    for (int64_t t = 0; t < per_frame; ++t) {
      CHECK(state.keyframes_mask[static_cast<size_t>(t)] == 1.0F);
    }
    for (int64_t t = per_frame; t < target_tokens; ++t) {
      CHECK(state.keyframes_mask[static_cast<size_t>(t)] == 0.0F);
    }
  }

  SUBCASE("the slot's temporal span is exactly [t, t+1) over fps") {
    // keyframe_slots.py:152-174. ONE pixel frame, which is what distinguishes a
    // slot from a regular latent frame spanning `factors.time` in RoPE space.
    // `causal_fix` is off because the span is written explicitly; applying both
    // would shift the slot at index 0.
    Ltx2LatentState state = fresh();
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, {8, 16},
                                               nullptr);
    // Positions are [3, tokens, 2] concatenated PER DIMENSION, so the temporal
    // axis is dimension 0 and sits at the front of the buffer.
    for (int64_t k = 0; k < 2; ++k) {
      const double expect_start =
          static_cast<double>(static_cast<float>((8.0 + 8.0 * static_cast<double>(k)) / kFps));
      const double expect_end =
          static_cast<double>(static_cast<float>((9.0 + 8.0 * static_cast<double>(k)) / kFps));
      for (int64_t i = 0; i < per_frame; ++i) {
        const size_t at = static_cast<size_t>((target_tokens + k * per_frame + i) * 2);
        INFO("slot ", k, " token ", i);
        CHECK(state.positions[at] == doctest::Approx(expect_start));
        CHECK(state.positions[at + 1] == doctest::Approx(expect_end));
        // The span is ONE pixel frame wide, stated as a relation so a mutation
        // of the two literals above cannot move both sides together.
        CHECK((state.positions[at + 1] - state.positions[at]) ==
              doctest::Approx(1.0 / kFps).epsilon(1e-4));
      }
    }
  }

  SUBCASE("an initial_keyframes seed reaches `latent` and not `clean`") {
    // THE PROPERTY `test_ltx2_video` CANNOT REACH — see the note above this
    // case. Upstream's stage 2 passes the spatially upsampled stage-1 slots here
    // (dfr_pipeline.py:364), and a port that wrote them into `clean_latent`
    // instead would place them where the noiser never reads at mask 1,
    // producing a finite, correctly shaped, EMPTY slot.
    Ltx2LatentState state = fresh();
    Ltx2LatentVolume seed;
    seed.batch = 1;
    seed.channels = target.channels;
    seed.frames = 2;
    seed.height = target.height;
    seed.width = target.width;
    seed.data.assign(static_cast<size_t>(seed.elems()), 0.0F);
    for (size_t i = 0; i < seed.data.size(); ++i) {
      seed.data[i] = static_cast<float>(i + 1);  // every element distinct and non-zero
    }
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, {8, 16}, &seed);

    double latent_absmax = 0.0, clean_absmax = 0.0;
    for (int64_t t = target_tokens; t < state.tokens; ++t) {
      for (int64_t w = 0; w < state.width; ++w) {
        const size_t at = static_cast<size_t>(t * state.width + w);
        latent_absmax = std::max(latent_absmax, std::abs(static_cast<double>(state.latent[at])));
        clean_absmax = std::max(clean_absmax, std::abs(static_cast<double>(state.clean[at])));
      }
    }
    CHECK(latent_absmax > 0.0);
    CHECK(clean_absmax == 0.0);

    // A seed whose K, spatial size or channel count disagrees is REFUSED
    // (keyframe_slots.py:65-68, :107-112) rather than broadcast into place.
    Ltx2LatentState other = fresh();
    Ltx2LatentVolume wrong_k = seed;
    wrong_k.frames = 1;
    wrong_k.data.resize(static_cast<size_t>(wrong_k.elems()));
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&other, target, 1, factors, kFps,
                                                               {8, 16}, &wrong_k),
                    std::exception);
    Ltx2LatentState third = fresh();
    Ltx2LatentVolume wrong_hw = seed;
    wrong_hw.height = target.height + 1;
    wrong_hw.data.resize(static_cast<size_t>(wrong_hw.elems()));
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&third, target, 1, factors, kFps,
                                                               {8, 16}, &wrong_hw),
                    std::exception);
  }

  SUBCASE("the readback reads the LAYOUT, not the trailing tokens") {
    // THE SECOND PROPERTY `test_ltx2_video` CANNOT REACH. The two readings
    // differ only when another appending item lands after the slot item, and
    // upstream has exactly one — `VideoConditionByReferenceLatent`
    // (dfr_pipeline.py:366-373), whose arm #975 owes. So the configuration is
    // built here BY HAND, deliberately, and this comment is why.
    //
    // The seed makes the slots identifiable and the keyframe appended AFTER them
    // carries a different value. A readback that took the last `num_tokens`
    // tokens returns the KEYFRAME's values — in the right shape, at the right
    // count, and completely wrong.
    Ltx2LatentState state = fresh();
    Ltx2LatentVolume seed;
    seed.batch = 1;
    seed.channels = target.channels;
    seed.frames = 2;
    seed.height = target.height;
    seed.width = target.width;
    seed.data.assign(static_cast<size_t>(seed.elems()), 3.0F);
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, {8, 16}, &seed);
    const int64_t after_slots = state.tokens;

    Ltx2LatentVolume keyframe;
    keyframe.batch = 1;
    keyframe.channels = target.channels;
    keyframe.frames = 1;
    keyframe.height = target.height;
    keyframe.width = target.width;
    keyframe.data.assign(static_cast<size_t>(keyframe.elems()), 99.0F);
    Ltx2ConditionVideoByKeyframe(&state, keyframe, 1, factors, kFps, /*frame_idx=*/16,
                                 /*strength=*/1.0, /*num_pixel_frames=*/1, /*causal_fix=*/false);
    REQUIRE(state.tokens > after_slots);

    const Ltx2LatentVolume slots = Ltx2ExtractGeneratedKeyframes(state, target, 1);
    REQUIRE(slots.frames == 2);
    REQUIRE(!slots.data.empty());
    // 3.0 is the SEED; 99.0 is the keyframe that now trails. A trailing-token
    // readback returns 99 for every element.
    for (float value : slots.data) {
      CHECK(value == doctest::Approx(3.0F));
    }
  }

  SUBCASE("clear_conditioning extracts BEFORE it trims") {
    // tools.py:97 against :115. The slots live outside the target grid, so the
    // trim is exactly what destroys them, and the order is the whole content.
    Ltx2LatentState state = fresh();
    Ltx2LatentVolume seed;
    seed.batch = 1;
    seed.channels = target.channels;
    seed.frames = 2;
    seed.height = target.height;
    seed.width = target.width;
    seed.data.assign(static_cast<size_t>(seed.elems()), 7.0F);
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, {8, 16}, &seed);

    Ltx2LatentVolume out;
    Ltx2ClearConditioning(&state, target_tokens, &target, 1, &out);
    CHECK(state.tokens == target_tokens);
    REQUIRE(out.frames == 2);
    for (float value : out.data) CHECK(value == doctest::Approx(7.0F));
    // Upstream KEEPS the layout on the returned state (tools.py:114), so a
    // second read after the trim finds a layout pointing past the end and
    // refuses rather than returning whatever is there. That refusal is the
    // reason the order cannot be reversed silently.
    CHECK_THROWS_AS(Ltx2ExtractGeneratedKeyframes(state, target, 1), std::exception);
  }

  SUBCASE("upstream's own validation is mirrored") {
    Ltx2LatentState state = fresh();
    // Non-empty, non-negative, strictly increasing (keyframe_slots.py:52-57).
    CHECK_THROWS_AS(
        Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps, {}, nullptr),
        std::exception);
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps,
                                                               {-1}, nullptr),
                    std::exception);
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&state, target, 1, factors, kFps,
                                                               {16, 8}, nullptr),
                    std::exception);
    // At or beyond the target's PIXEL frame count (:77-81). The target is 3
    // latent frames at temporal factor 8, so 24 pixel frames: 23 is legal and 24
    // is not. Both sides of the bound are checked, because an off-by-one here is
    // invisible to every shape.
    Ltx2LatentState ok = fresh();
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&ok, target, 1, factors, kFps, {23}, nullptr);
    CHECK(ok.generated_keyframe_layout.applied());
    Ltx2LatentState over = fresh();
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&over, target, 1, factors, kFps,
                                                               {24}, nullptr),
                    std::exception);
    // A SECOND application is refused rather than merged (:133-134): the layout
    // is one contiguous range, so a second item would orphan the first one's
    // tokens while leaving the state perfectly well-formed.
    Ltx2LatentState twice = fresh();
    Ltx2ConditionVideoByGeneratedKeyframeSlots(&twice, target, 1, factors, kFps, {8}, nullptr);
    CHECK_THROWS_AS(Ltx2ConditionVideoByGeneratedKeyframeSlots(&twice, target, 1, factors, kFps,
                                                               {16}, nullptr),
                    std::exception);
  }

  SUBCASE("a state with no slots extracts NOTHING, and that is not an error") {
    // `if layout is None: return None` (tools.py:205-206). The ordinary case:
    // every render that asks for no slots takes it, so an exception here would
    // fire on the default path.
    Ltx2LatentState state = fresh();
    const Ltx2LatentVolume none = Ltx2ExtractGeneratedKeyframes(state, target, 1);
    CHECK(none.frames == 0);
    CHECK(none.data.empty());
  }
}
