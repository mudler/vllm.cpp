// LTX-2.5 IC-LoRA reference conditioning — VALUE gates against the EXECUTED
// pinned module. Row LTX25-IC-LORA-REF-VIDEO (#3020),
// spec .agents/specs/ltx25-ic-lora-ref-video.md.
//
// THESE ARE NOT THE PROOF THAT ANYTHING IS REACHED. Per .agents/reachability.md
// they localize a failure: every case here stays green when the production call
// site in `ltx2_video.cpp` is deleted. The reachability cases are
// `ltx2 ic-lora: ...` in `test_ltx2_video`, which render through `Generate`.
//
// Every golden is what upstream's own code RETURNED, and every case carries the
// hypothesis it rejects. `scripts/gen-ltx2-iclora-reference-goldens.py` refuses
// to emit a case whose separation is zero, and the separation table it writes at
// the bottom of the .inc is the record of what each case can actually see.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_conditioning.h"
#include "vllm/model_executor/models/ltx2_iclora_reference.h"

#include "ltx2_iclora_reference_goldens.inc"

namespace {

// The pin the goldens were generated against. Asserted rather than assumed: a
// regeneration against a different checkout must fail this gate instead of
// silently replacing the oracle.
constexpr const char* kExpectedUpstream = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";

std::string Message(void (*run)()) {
  try {
    run();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return std::string();
}

}  // namespace

TEST_CASE("ltx2 ic-lora goldens: the oracle pin is the one this suite claims") {
  CHECK(std::string(vllm_test::kLtx2IcLoraRefUpstreamRevision) == kExpectedUpstream);
}

TEST_CASE("ltx2 ic-lora: temporal_subsample keeps index 1, not every Nth from 0") {
  // The whole content of `temporal_subsample` (iclora_utils.py:87-90) is WHICH
  // indices it keeps, and the plausible wrong reading `range(0, F, N)` returns
  // the SAME COUNT — so no shape check, no token count and no rendered frame
  // count can tell them apart. The reference simply describes different moments.
  size_t at = 0;
  for (int c = 0; c < vllm_test::kLtx2TemporalSubsampleCases; ++c) {
    const int64_t frames = vllm_test::kLtx2TemporalSubsampleFrames[c];
    const int64_t factor = vllm_test::kLtx2TemporalSubsampleFactor[c];
    const int64_t count = vllm_test::kLtx2TemporalSubsampleCount[c];
    INFO("frames = " << frames << ", factor = " << factor);
    const std::vector<int64_t> got = vllm::Ltx2TemporalSubsampleIndices(frames, factor);
    REQUIRE(static_cast<int64_t>(got.size()) == count);
    for (int64_t i = 0; i < count; ++i) {
      CHECK(got[static_cast<size_t>(i)] == vllm_test::kLtx2TemporalSubsampleKept[at + i]);
    }
    // ...and it is NOT the rejected hypothesis, asserted case by case rather
    // than once over the whole set: the two agree at factor 1 and on a
    // single-frame clip, and a golden that only checked the aggregate would pass
    // on a build that got the two live cases wrong and the degenerate ones right.
    std::vector<int64_t> rejected;
    for (int64_t i = 0; i < frames; i += factor) rejected.push_back(i);
    if (frames > 1 && factor > 1) {
      CHECK_MESSAGE(got != rejected,
                    "at frames = " << frames << " factor = " << factor
                                   << " the port agrees with `range(0, F, N)`, which upstream is "
                                      "not");
    }
    at += static_cast<size_t>(count);
  }
}

TEST_CASE("ltx2 ic-lora: the gather follows the indices on the FRAME axis") {
  // `Ltx2TemporalSubsampleIndices` says which frames; this says the copy takes
  // them out of a CHANNEL-MAJOR volume. A prefix of the buffer would keep whole
  // channels and drop others, and the result is the right size either way.
  const int64_t channels = 3, frames = 5, plane = 2;
  std::vector<float> clip(static_cast<size_t>(channels * frames * plane));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      for (int64_t p = 0; p < plane; ++p) {
        clip[static_cast<size_t>((c * frames + t) * plane + p)] =
            static_cast<float>(c * 100 + t * 10 + p);
      }
    }
  }
  const std::vector<float> got = vllm::Ltx2TemporalSubsample(clip, channels, frames, plane, 2);
  // Kept indices are {0, 1, 3}.
  REQUIRE(got.size() == static_cast<size_t>(channels * 3 * plane));
  const int64_t keep[] = {0, 1, 3};
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < 3; ++t) {
      for (int64_t p = 0; p < plane; ++p) {
        CHECK(got[static_cast<size_t>((c * 3 + t) * plane + p)] ==
              doctest::Approx(static_cast<float>(c * 100 + keep[t] * 10 + p)));
      }
    }
  }
}

TEST_CASE("ltx2 ic-lora: the reference geometry divides, and refuses only when scale != 1") {
  // `iclora_utils.py:111-117`, executed. The refusal's guard is `scale != 1`
  // AND indivisible; dropping the first half refuses nothing extra (1 divides
  // everything) while dropping the second refuses every render.
  const vllm::Ltx2IcLoraReferenceGeometry geom =
      vllm::Ltx2ResolveIcLoraReferenceGeometry(448, 768, 4);
  CHECK(geom.height == vllm_test::kLtx2RefGeomHeight);
  CHECK(geom.width == vllm_test::kLtx2RefGeomWidth);
  // ...and NOT the phase's own grid, which is the plausible wrong answer.
  CHECK(geom.height != 448);
  CHECK(geom.width != 768);

  // Scale 1 never refuses, whatever the dimensions. Measured on the oracle:
  // 449 at scale 1 comes back as 449.
  CHECK(vllm::Ltx2ResolveIcLoraReferenceGeometry(449, 771, 1).height ==
        vllm_test::kLtx2RefScaleOneHeight);

  const std::string refusal =
      Message([] { (void)vllm::Ltx2ResolveIcLoraReferenceGeometry(448, 770, 4); });
  INFO(refusal);
  // Upstream's own sentence, from the raised ValueError rather than from a
  // reading of the f-string.
  CHECK(refusal.find(vllm_test::kLtx2RefDivisibilityRefusal) != std::string::npos);
}

TEST_CASE("ltx2 ic-lora: the mask downsample carves out the first latent frame") {
  // `downsample_mask_video_to_latent` (iclora_utils.py:52-84). The INPUT comes
  // out of the .inc so both sides read identical bytes rather than agreeing
  // about a random stream.
  const int64_t f_pix = vllm_test::kLtx2MaskPixFrames;
  const int64_t h_pix = vllm_test::kLtx2MaskPixHeight;
  const int64_t w_pix = vllm_test::kLtx2MaskPixWidth;
  const int64_t f_lat = vllm_test::kLtx2MaskLatFrames;
  const int64_t h_lat = vllm_test::kLtx2MaskLatHeight;
  const int64_t w_lat = vllm_test::kLtx2MaskLatWidth;
  const std::vector<float> mask(
      vllm_test::kLtx2MaskPixels,
      vllm_test::kLtx2MaskPixels + static_cast<size_t>(f_pix * h_pix * w_pix));

  const std::vector<float> got =
      vllm::Ltx2DownsampleMaskVideoToLatent(mask, f_pix, h_pix, w_pix, f_lat, h_lat, w_lat);
  REQUIRE(got.size() == static_cast<size_t>(f_lat * h_lat * w_lat));

  double max_diff = 0.0, bilinear_diff = 0.0, uniform_diff = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(static_cast<double>(
                                      got[i] - vllm_test::kLtx2MaskLatentWeights[i])));
    bilinear_diff = std::max(bilinear_diff,
                             std::fabs(static_cast<double>(
                                 got[i] - vllm_test::kLtx2MaskLatentBilinear[i])));
    uniform_diff = std::max(uniform_diff,
                            std::fabs(static_cast<double>(
                                got[i] - vllm_test::kLtx2MaskLatentUniformPool[i])));
  }
  // f32 area pooling accumulated in f64 against torch's own f32 reduction.
  CHECK_MESSAGE(max_diff < 1e-6, "max |diff| against the executed oracle is " << max_diff);
  // AND IT IS NOT EITHER REJECTED ANSWER. Both produce a correctly shaped mask
  // of entirely plausible values, and nothing about a render's shape can see the
  // difference — which is why they are asserted rather than described.
  CHECK_MESSAGE(bilinear_diff > 1e-3,
                "the port agrees with BILINEAR spatial interpolation; `mode=\"area\"` is what "
                "upstream passes (iclora_utils.py:63-67)");
  CHECK_MESSAGE(uniform_diff > 1e-3,
                "the port agrees with uniform temporal pooling, so the causal first-frame "
                "carve-out (iclora_utils.py:70, :80) is not happening");
}

TEST_CASE("ltx2 ic-lora: the area window is the general form, not an integer stride") {
  // `mode="area"` dispatches to `adaptive_avg_pool2d`, whose window for output
  // index `i` is `[floor(i*I/O), ceil((i+1)*I/O))`. An integer-stride box filter
  // `[i*(I//O), (i+1)*(I//O))` is elementwise EQUAL to it on every divisible
  // shape, so the 8 -> 2 fixture above cannot see the difference and neither can
  // the e2e case, which reads a 64x64 mask and pools to 2x2. This fixture is
  // 9 -> 2, where the two readings separate.
  const int64_t f_pix = vllm_test::kLtx2MaskNdPixFrames;
  const int64_t h_pix = vllm_test::kLtx2MaskNdPixHeight;
  const int64_t w_pix = vllm_test::kLtx2MaskNdPixWidth;
  const int64_t f_lat = vllm_test::kLtx2MaskNdLatFrames;
  const int64_t h_lat = vllm_test::kLtx2MaskNdLatHeight;
  const int64_t w_lat = vllm_test::kLtx2MaskNdLatWidth;
  REQUIRE(h_pix % h_lat != 0);  // the fixture is only a gate while it stays indivisible
  REQUIRE(w_pix % w_lat != 0);
  const std::vector<float> mask(
      vllm_test::kLtx2MaskNdPixels,
      vllm_test::kLtx2MaskNdPixels + static_cast<size_t>(f_pix * h_pix * w_pix));

  const std::vector<float> got =
      vllm::Ltx2DownsampleMaskVideoToLatent(mask, f_pix, h_pix, w_pix, f_lat, h_lat, w_lat);
  REQUIRE(got.size() == static_cast<size_t>(f_lat * h_lat * w_lat));

  double max_diff = 0.0, stride_diff = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(static_cast<double>(
                                      got[i] - vllm_test::kLtx2MaskNdLatentWeights[i])));
    stride_diff = std::max(stride_diff,
                           std::fabs(static_cast<double>(
                               got[i] - vllm_test::kLtx2MaskNdLatentStrideBox[i])));
  }
  CHECK_MESSAGE(max_diff < 1e-6, "max |diff| against the executed oracle is " << max_diff);
  CHECK_MESSAGE(stride_diff > 1e-3,
                "the port agrees with an integer-stride box filter, so the `area` window is "
                "the divisible special case rather than the general form "
                "(iclora_utils.py:63-67, ltx2_iclora_reference.cpp PoolStart/PoolEnd)");
}

TEST_CASE("ltx2 ic-lora: the mask downsample refuses an incompatible frame pair") {
  // Upstream's own assertion (`:74-77`). A truncating group size would silently
  // drop the tail of the mask and still produce the right number of weights.
  const std::string refusal = Message([] {
    const std::vector<float> mask(8 * 1 * 1, 0.5F);
    (void)vllm::Ltx2DownsampleMaskVideoToLatent(mask, 8, 1, 1, 4, 1, 1);
  });
  INFO(refusal);
  CHECK(refusal.find("must be divisible by") != std::string::npos);

  // ...and the DEGENERATE arms return the first frame alone (`:81-82`) rather
  // than refusing, which is what keeps a single-frame reference working.
  const std::vector<float> single(1 * 2 * 2, 0.25F);
  CHECK(vllm::Ltx2DownsampleMaskVideoToLatent(single, 1, 2, 2, 1, 2, 2).size() == 4U);
}

TEST_CASE("ltx2 ic-lora: the mask video's arithmetic remaps [-1, 1] to [0, 1]") {
  // `_load_mask_video` (ic_lora.py:530-536) after the read: mean over channels,
  // `(x + 1) / 2`, clamp. The pixels arrive CHANNEL-major, so the three samples
  // of one pixel are a plane apart — reading them as adjacent would average
  // three different pixels and still produce a plausible mask.
  const int64_t channels = 3, frames = 1, plane = 2;
  const std::vector<float> pixels = {
      -1.0F, 1.0F,   // channel 0
      -1.0F, 1.0F,   // channel 1
      -1.0F, 1.0F};  // channel 2
  const std::vector<float> got = vllm::Ltx2MaskVideoFromPixels(pixels, channels, frames, plane);
  REQUIRE(got.size() == 2U);
  CHECK(got[0] == doctest::Approx(0.0));
  CHECK(got[1] == doctest::Approx(1.0));

  // A pixel whose channels DISAGREE separates the channel-major read from the
  // interleaved one: the mean of {-1, 0, 1} is 0 and maps to 0.5, while reading
  // the first three values as one pixel would average {-1, 1, -1}.
  const std::vector<float> mixed = {-1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> mixed_got = vllm::Ltx2MaskVideoFromPixels(mixed, channels, frames, plane);
  CHECK(mixed_got[0] == doctest::Approx(0.5));

  // The clamp is upstream's and it is not defensive: a value above 1 becomes a
  // POSITIVE log-space bias, which amplifies attention rather than attenuating.
  const std::vector<float> hot(6, 4.0F);
  CHECK(vllm::Ltx2MaskVideoFromPixels(hot, channels, frames, plane)[0] == doctest::Approx(1.0));
}

TEST_CASE("ltx2 ic-lora: build_attention_mask puts the cross block on the NOISY rows only") {
  // THE FIXTURE CARRIES A PRIOR REFERENCE TOKEN, and that is the whole reason
  // this case can fail. With `num_existing == num_noisy` the true block
  // structure and the plausible reading `cross on ALL existing rows` are
  // ELEMENTWISE EQUAL — measured 0 separating elements on the oracle — and the
  // golden would be a mute switch. The generator refuses that fixture.
  const int64_t noisy = vllm_test::kLtx2AmNoisy;
  const int64_t added = vllm_test::kLtx2AmNew;
  const int64_t existing = vllm_test::kLtx2AmExisting;
  const std::vector<float> cross(vllm_test::kLtx2AmCross,
                                 vllm_test::kLtx2AmCross + static_cast<size_t>(added));
  const std::vector<float> got =
      vllm::Ltx2BuildAttentionMask({}, noisy, added, existing, cross);
  const int64_t total = existing + added;
  REQUIRE(got.size() == static_cast<size_t>(total * total));

  size_t oracle_diff = 0, rejected_same = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != vllm_test::kLtx2AmMask[i]) ++oracle_diff;
    if (got[i] == vllm_test::kLtx2AmMaskCrossOnAllRows[i]) ++rejected_same;
  }
  CHECK_MESSAGE(oracle_diff == 0, oracle_diff << " elements differ from the executed oracle");
  CHECK_MESSAGE(rejected_same < got.size(),
                "the port is elementwise equal to `cross on ALL existing rows`, which is the "
                "reading this fixture exists to separate");
}

TEST_CASE("ltx2 ic-lora: an existing mask is PRESERVED under a second item") {
  // `:223-226`. Refilling the top-left block with ones would discard the first
  // item's attenuation entirely, and the result is still a legal mask of the
  // right shape whose values are all in [0, 1].
  const int64_t noisy = vllm_test::kLtx2AmNoisy;
  const int64_t first_new = vllm_test::kLtx2AmNew;
  const int64_t existing = vllm_test::kLtx2AmExisting;
  const std::vector<float> cross(vllm_test::kLtx2AmCross,
                                 vllm_test::kLtx2AmCross + static_cast<size_t>(first_new));
  const std::vector<float> first =
      vllm::Ltx2BuildAttentionMask({}, noisy, first_new, existing, cross);
  const std::vector<float> cross2(vllm_test::kLtx2AmCross2, vllm_test::kLtx2AmCross2 + 1);
  const std::vector<float> second =
      vllm::Ltx2BuildAttentionMask(first, noisy, 1, existing + first_new, cross2);

  size_t diff = 0, matches_refilled = 0;
  for (size_t i = 0; i < second.size(); ++i) {
    if (second[i] != vllm_test::kLtx2AmMask2[i]) ++diff;
    if (second[i] == vllm_test::kLtx2AmMask2Refilled[i]) ++matches_refilled;
  }
  CHECK_MESSAGE(diff == 0, diff << " elements differ from the executed oracle");
  CHECK_MESSAGE(matches_refilled < second.size(),
                "the port discarded the existing mask and refilled the block with ones");
}

TEST_CASE("ltx2 ic-lora: resolve_cross_mask's scalar and 1-D arms, and its refusal") {
  const std::vector<float> scalar = vllm::Ltx2ResolveCrossMask({}, 0.5, 4);
  REQUIRE(scalar.size() == 4U);
  for (size_t i = 0; i < scalar.size(); ++i) CHECK(scalar[i] == vllm_test::kLtx2CrossScalar[i]);

  const std::vector<float> values = {0.1F, 0.2F, 0.3F, 0.4F};
  const std::vector<float> oned = vllm::Ltx2ResolveCrossMask(values, 1.0, 4);
  REQUIRE(oned.size() == 4U);
  for (size_t i = 0; i < oned.size(); ++i) CHECK(oned[i] == vllm_test::kLtx2CrossOneD[i]);
  // ...and the 1-D arm is not the scalar fill, which is the failure a per-token
  // mask that silently collapsed to a constant would produce.
  CHECK(oned[0] != oned[3]);

  // `:50-53` refuses rather than broadcasting: a mask one token short of the
  // sequence would attenuate the wrong tokens and still render.
  const std::string refusal = Message([] {
    const std::vector<float> three = {0.1F, 0.2F, 0.3F};
    (void)vllm::Ltx2ResolveCrossMask(three, 1.0, 4);
  });
  INFO(refusal);
  CHECK(refusal.find("must equal num_new_tokens") != std::string::npos);
}

TEST_CASE("ltx2 ic-lora: an unmasked append PADS the mask instead of leaving it short") {
  // `update_attention_mask`'s `attention_mask is None` arm (`:141-156`).
  const int64_t noisy = vllm_test::kLtx2AmNoisy;
  const int64_t first_new = vllm_test::kLtx2AmNew;
  const int64_t existing = vllm_test::kLtx2AmExisting;
  const std::vector<float> cross(vllm_test::kLtx2AmCross,
                                 vllm_test::kLtx2AmCross + static_cast<size_t>(first_new));
  const std::vector<float> first =
      vllm::Ltx2BuildAttentionMask({}, noisy, first_new, existing, cross);

  // No mask and none present is upstream's None.
  CHECK(vllm::Ltx2PadAttentionMaskForUnmaskedTokens({}, noisy, 1, existing).empty());

  const std::vector<float> padded =
      vllm::Ltx2PadAttentionMaskForUnmaskedTokens(first, noisy, 1, existing + first_new);
  REQUIRE(padded.size() == static_cast<size_t>((existing + first_new + 1) *
                                               (existing + first_new + 1)));
  size_t diff = 0;
  for (size_t i = 0; i < padded.size(); ++i) {
    if (padded[i] != vllm_test::kLtx2AmMaskPaddedOnes[i]) ++diff;
  }
  CHECK_MESSAGE(diff == 0, diff << " elements differ from the executed oracle");
  // GREW WITH THE SEQUENCE. A mask left at its pre-append size is not a shape
  // error downstream — `Ltx2ModalityInput` accepts the key-only broadcast form
  // too, so it is read as a legal mask over a different axis.
  CHECK(padded.size() > first.size());
}

TEST_CASE("ltx2 ic-lora: an appending item extends the state's attention mask") {
  // THE OBLIGATION LIVES IN `AppendTokens`, which is what upstream's docstring
  // makes it (mask_utils.py:83-85 for the keyframes mask; every appending item
  // calls `update_attention_mask` with a literal None at keyframe_cond.py:68-76
  // and reference_video_cond.py:86-94). Before this row no state could carry a
  // mask at all, so this branch was unreachable.
  vllm::Ltx2VideoLatentShape shape;
  shape.batch = 1;
  shape.channels = 2;
  shape.frames = 1;
  shape.height = 2;
  shape.width = 2;
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(shape, /*patch_size=*/1, factors, /*fps=*/24.0,
                                       /*causal_fix=*/true, nullptr, nullptr);
  REQUIRE(state.tokens == 4);
  CHECK_MESSAGE(state.noisy_tokens == 4,
                "the state must record `latent_tools.target_shape.token_count()` at construction; "
                "after the first append `tokens` is no longer it");
  vllm::Ltx2LatentVolume ref;
  ref.batch = 1;
  ref.channels = 2;
  ref.frames = 1;
  ref.height = 2;
  ref.width = 2;
  ref.data.assign(8, 0.5F);
  const auto apply = [&] {
    vllm::Ltx2ConditionVideoByReference(&state, ref, /*patch_size=*/1, factors, /*fps=*/24.0,
                                        /*downscale_factor=*/1, /*temporal_scale_factor=*/1,
                                        /*strength=*/1.0, /*causal_fix=*/true);
  };

  // FIRST ITEM, no mask anywhere. Upstream's `update_attention_mask` returns
  // None when its argument is None and the state carries none (`:142-143`), so
  // an append on an unmasked state must leave the field empty rather than
  // inventing an all-ones mask — which would be the identity and would make
  // every later assertion vacuous.
  apply();
  REQUIRE(state.tokens == 8);
  CHECK_MESSAGE(state.attention_mask.empty(),
                "an append on an unmasked state invented a mask; upstream returns None");

  // Now the wrapper writes one, over the 8 tokens that exist.
  state.attention_mask = vllm::Ltx2BuildAttentionMask({}, state.noisy_tokens, 4, 4,
                                                      {0.25F, 0.75F, 0.5F, 1.0F});
  REQUIRE(state.attention_mask.size() == 64U);

  // SECOND ITEM, and this is the branch this row made reachable.
  const int64_t before = state.tokens;
  apply();
  const int64_t added = state.tokens - before;
  REQUIRE(added == 4);
  // The mask covered 8 tokens and the state is now 12. A build that left it at
  // 64 entries would hand the DiT a legal, differently shaped mask.
  CHECK_MESSAGE(static_cast<int64_t>(state.attention_mask.size()) == 12 * 12,
                "the appending item did not extend the attention mask; it holds "
                    << state.attention_mask.size() << " entries for " << state.tokens
                    << " tokens");
  // ...and it PADDED with full attention rather than refilling with ones: the
  // first item's 0.25 is still there.
  bool has_quarter = false;
  for (const float v : state.attention_mask) {
    if (v == 0.25F) has_quarter = true;
  }
  CHECK_MESSAGE(has_quarter,
                "the earlier item's attenuation was discarded when the mask grew "
                "(mask_utils.py:224 preserves the existing block)");
}

TEST_CASE("ltx2 ic-lora: the consumption seam converts [0,1] to an additive log-space bias") {
  // `_prepare_self_attention_mask` (transformer_args.py:208-237), already ported
  // as `Ltx2PrepareSelfAttentionMask`. Pinned HERE because nothing in `src/`
  // assigned `Ltx2ModalityInput::attention_mask` before this row, so these
  // numbers were previously gated only through a hand-built test struct.
  const std::vector<float> probe(vllm_test::kLtx2SelfMaskProbe,
                                 vllm_test::kLtx2SelfMaskProbe + 4);
  const std::vector<float> bias = vllm::Ltx2PrepareSelfAttentionMask(probe.data(), 4);
  REQUIRE(bias.size() == 4U);
  for (size_t i = 0; i < bias.size(); ++i) {
    INFO("i = " << i);
    CHECK(bias[i] == vllm_test::kLtx2SelfMaskBias[i]);
  }
  // The two values a naive `log(x)` gets wrong, named rather than left to the
  // element compare: a zero is `finfo.min` and NOT -inf, and a subnormal is
  // clamped to `finfo.tiny` before the log rather than logged as-is.
  CHECK(std::isfinite(bias[2]));
  CHECK(bias[3] > -100.0F);
}
