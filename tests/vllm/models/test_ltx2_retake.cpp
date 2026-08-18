// LTX-2.5 RETAKE — the ported pieces, at value level. Row LTX25-RETAKE (#924),
// spec .agents/specs/ltx25-retake.md. Upstream: Lightricks/LTX-2 @ fd4ded7f.
//
// These are UNIT tests and they are not the reachability proof. That proof lives
// in `test_ltx2_video`, which drives a retake through `LoadVideoEngine` and
// `Generate` and goes red when the production call site is deleted. These cases
// localize a failure once that one has caught it — which is what
// .agents/reachability.md asks for, in that order.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_retake.h"

TEST_CASE("ltx2 retake: the video and audio branches use DIFFERENT coordinates") {
  // The two conventions, asserted separately. One shared assertion would pass
  // whichever of the two were used for both, which is the trap
  // `noise_mask_cond.py:27-35` sets by branching on the shape type.
  const vllm::Ltx2ScaleFactors factors;  // SpatioTemporalScaleFactors.default() = (8, 32, 32)
  vllm::Ltx2VideoLatentShape vshape;
  vshape.channels = 4;
  vshape.frames = 3;
  vshape.height = 1;
  vshape.width = 1;

  SUBCASE("video: latent bounds through get_pixel_coords, then divided by fps") {
    // `causal_fix` TRUE: latent frame f spans [max(8f - 7, 0), 8f + 1) pixel
    // frames (patchifiers.py:169), so at 24 fps frame 0 is [0, 0.0417) s,
    // frame 1 is [0.0417, 0.375) s and frame 2 is [0.375, 0.708) s.
    const std::vector<float> mask = vllm::Ltx2TemporalRegionMaskVideo(
        vshape, /*patch_size=*/1, factors, /*fps=*/24.0, /*start_time=*/0.05,
        /*end_time=*/0.10, /*causal_fix=*/true);
    REQUIRE(mask.size() == 3);
    CHECK(mask[0] == 0.0F);
    CHECK(mask[1] == 1.0F);
    CHECK(mask[2] == 0.0F);
  }

  SUBCASE("video: the test is OVERLAP, not containment") {
    // A window strictly INSIDE frame 1's span still selects frame 1
    // (noise_mask_cond.py:39 is `t_end > start && t_start < end`). Containment
    // would select nothing here and would leave a seam at each end of every
    // regenerated window.
    const std::vector<float> mask = vllm::Ltx2TemporalRegionMaskVideo(
        vshape, 1, factors, 24.0, /*start_time=*/0.10, /*end_time=*/0.11, /*causal_fix=*/true);
    REQUIRE(mask.size() == 3);
    CHECK(mask[0] == 0.0F);
    CHECK(mask[1] == 1.0F);
    CHECK(mask[2] == 0.0F);
  }

  SUBCASE("video: causal_fix TRUE and FALSE are distinguishable") {
    // The two upstream defaults DISAGREE — `get_pixel_coords` declares False
    // (patchifiers.py:140) and `TemporalRegionMask` calls it with True
    // (noise_mask_cond.py:33) — so this pins which one the port takes. Without
    // the fix frame f spans [8f, 8f + 8): frame 0 is [0, 0.333) s and frame 1 is
    // [0.333, 0.667) s, so the window below selects BOTH instead of one.
    const std::vector<float> with_fix =
        vllm::Ltx2TemporalRegionMaskVideo(vshape, 1, factors, 24.0, 0.30, 0.40, true);
    const std::vector<float> without =
        vllm::Ltx2TemporalRegionMaskVideo(vshape, 1, factors, 24.0, 0.30, 0.40, false);
    REQUIRE(with_fix.size() == 3);
    REQUIRE(without.size() == 3);
    CHECK(with_fix[0] == 0.0F);
    CHECK(with_fix[1] == 1.0F);
    CHECK(without[0] == 1.0F);
    CHECK(without[1] == 1.0F);
  }

  SUBCASE("audio: the bounds are ALREADY seconds and are not scaled again") {
    // `_get_audio_latent_time_in_sec` (patchifiers.py:216-249) at the
    // constructor defaults: latent frame t spans [max(4t - 3, 0), 4t + 1) mel
    // frames times 160/16000 s, so frame 0 is [0, 0.01) s, frame 1 is
    // [0.01, 0.05) s and frame 2 is [0.05, 0.09) s. Dividing these by an fps —
    // the video branch's last step — would put every boundary 24x too early.
    vllm::Ltx2AudioLatentShape ashape;
    ashape.channels = 8;
    ashape.mel_bins = 16;
    ashape.frames = 3;
    const vllm::Ltx2AudioPatchifierParams params;
    const std::vector<float> mask =
        vllm::Ltx2TemporalRegionMaskAudio(ashape, params, /*start_time=*/0.02, /*end_time=*/0.03);
    REQUIRE(mask.size() == 3);
    CHECK(mask[0] == 0.0F);
    CHECK(mask[1] == 1.0F);
    CHECK(mask[2] == 0.0F);
  }

  SUBCASE("the mask is neither all ones nor all zeros for a partial window") {
    // The floor a count-shaped assertion cannot make on its own: a build that
    // returned ones everywhere and a build that returned zeros everywhere both
    // produce a mask of the right length.
    const std::vector<float> mask =
        vllm::Ltx2TemporalRegionMaskVideo(vshape, 1, factors, 24.0, 0.05, 0.10, true);
    int64_t inside = 0;
    for (const float value : mask) {
      if (value != 0.0F) ++inside;
    }
    CHECK(inside > 0);
    CHECK(inside < static_cast<int64_t>(mask.size()));
  }
}

TEST_CASE("ltx2 retake: the conform TRUNCATES from the front and ZERO-PADS at the tail") {
  // This is the half a reader who knows only the audio-to-video path gets
  // backwards. `_conform_latent_length` (utils/helpers.py:149-162) pads;
  // `a2vid_two_stage.py:202` never does, and row LTX25-A2V-AUDIO-INPUT mirrored
  // that as a REFUSAL because `create_initial_state` asserts the shape
  // (ltx-core/tools.py:146-148). Both polarities are upstream's, on two callers,
  // which is why these are two functions and not one with a flag.
  const std::vector<float> latent = {1, 2, 3, 4, 10, 20, 30, 40};  // 2 channels x 4 frames x 1

  SUBCASE("longer keeps the LEADING frames") {
    const std::vector<float> out =
        vllm::Ltx2ConformLatentLength(latent, /*channels=*/2, /*frames=*/4, /*per_frame=*/1,
                                      /*expected_frames=*/2);
    REQUIRE(out.size() == 4);
    // A tail slice would give {3, 4, 30, 40}, which every shape check accepts.
    CHECK(out[0] == 1.0F);
    CHECK(out[1] == 2.0F);
    CHECK(out[2] == 10.0F);
    CHECK(out[3] == 20.0F);
  }

  SUBCASE("shorter appends zeros at the END, per channel") {
    const std::vector<float> out = vllm::Ltx2ConformLatentLength(latent, 2, 4, 1, 6);
    REQUIRE(out.size() == 12);
    CHECK(out[0] == 1.0F);
    CHECK(out[3] == 4.0F);
    CHECK(out[4] == 0.0F);
    CHECK(out[5] == 0.0F);
    // Channel 1 starts at its OWN offset. A flat append — pad the whole buffer
    // once at the end — would put channel 1's first value here and shift the
    // entire second channel by two frames.
    CHECK(out[6] == 10.0F);
    CHECK(out[9] == 40.0F);
    CHECK(out[10] == 0.0F);
    CHECK(out[11] == 0.0F);
  }

  SUBCASE("equal is returned unchanged") {
    const std::vector<float> out = vllm::Ltx2ConformLatentLength(latent, 2, 4, 1, 4);
    CHECK(out == latent);
  }
}

TEST_CASE("ltx2 retake: the four-way modality plan, and what a frame folder decides") {
  SUBCASE("with an audio latent, all four combinations are distinct") {
    // retake.py:268-283, read as four independent predicates rather than two.
    for (const bool video : {true, false}) {
      for (const bool audio : {true, false}) {
        const vllm::Ltx2RetakePlan plan =
            vllm::Ltx2RetakePlanModalities(video, audio, /*has_audio_latent=*/true);
        CAPTURE(video);
        CAPTURE(audio);
        CHECK(plan.video_conditioned == video);
        CHECK(plan.video_frozen == !video);
        CHECK(plan.audio_conditioned == audio);
        CHECK(plan.audio_frozen == !audio);
      }
    }
  }
  SUBCASE("with NO audio latent, regenerate_audio has no observable effect") {
    // BOTH audio predicates are conjunctions with `initial_audio_latent is not
    // None` (retake.py:279, :282), and a frame folder never produces one
    // (utils/helpers.py:261-262). So the soundtrack is generated fresh whichever
    // way the knob is set. That is surprising, it is upstream's, and it is
    // pinned here so the next reader finds it asserted rather than re-derives it.
    const vllm::Ltx2RetakePlan on = vllm::Ltx2RetakePlanModalities(true, true, false);
    const vllm::Ltx2RetakePlan off = vllm::Ltx2RetakePlanModalities(true, false, false);
    CHECK_FALSE(on.audio_conditioned);
    CHECK_FALSE(on.audio_frozen);
    CHECK_FALSE(off.audio_conditioned);
    CHECK_FALSE(off.audio_frozen);
    // The VIDEO half is unaffected by the missing audio, because neither video
    // predicate consults it (retake.py:271, :274).
    CHECK(on.video_conditioned);
    CHECK(off.video_conditioned);
  }
}

TEST_CASE("ltx2 retake: the CLI-stage validations name what would have worked") {
  SUBCASE("an inverted or empty window reports BOTH values") {
    // retake.py:211-212, and upstream's own message carries both numbers.
    CHECK_THROWS_AS(vllm::Ltx2RetakeAssertWindow(1.0, 1.0), std::exception);
    CHECK_THROWS_AS(vllm::Ltx2RetakeAssertWindow(2.0, 1.0), std::exception);
    CHECK_NOTHROW(vllm::Ltx2RetakeAssertWindow(1.0, 2.0));
    try {
      vllm::Ltx2RetakeAssertWindow(2.0, 1.0);
      FAIL("an inverted window must refuse");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("must be less than end_time") != std::string::npos);
    }
  }

  SUBCASE("a frame count off the grid NAMES the snapped value") {
    // retake.py:348-351 computes the snapped count and puts it in the message.
    CHECK_NOTHROW(vllm::Ltx2RetakeAssertSourceGeometry(97, 64, 64, 8));
    CHECK_NOTHROW(vllm::Ltx2RetakeAssertSourceGeometry(9, 64, 64, 8));
    try {
      vllm::Ltx2RetakeAssertSourceGeometry(10, 64, 64, 8);
      FAIL("10 frames is not 8k+1");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("8k+1") != std::string::npos);
      // 10 -> 9, not 17: upstream floors (`(frames - 1) // time * time + 1`).
      CHECK(msg.find("use a video with 9 frames") != std::string::npos);
      CHECK(msg.find("use a video with 17 frames") == std::string::npos);
    }
  }

  SUBCASE("a resolution off the 32 grid names BOTH axes, width first") {
    try {
      vllm::Ltx2RetakeAssertSourceGeometry(9, 64, 48, 8);
      FAIL("48 is not a multiple of 32");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("multiples of 32") != std::string::npos);
      // `Got {width}x{height}` (retake.py:353) — that order, not the other one.
      CHECK(msg.find("48x64") != std::string::npos);
    }
    // The height axis is checked too, and by the same divisor.
    CHECK_THROWS_AS(vllm::Ltx2RetakeAssertSourceGeometry(9, 48, 64, 8), std::exception);
  }
}
