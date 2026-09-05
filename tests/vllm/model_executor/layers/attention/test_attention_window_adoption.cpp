// `ENG-ATTENTION-WINDOW` W1 (#2388) — the resolver must compute, for every site
// that adopts it, EXACTLY what that site inlined before.
//
// WHY THE EXPECTATION IS THE OLD EXPRESSION AND NOT A SHARED CONSTANT. The whole
// risk of this row is that one site's guard differs from the others in a way the
// spec missed; adopting the resolver would then change behaviour only for inputs
// the old guard excluded, which is silent by construction. Asserting against a
// shared expectation would encode the reading being tested. Each case below
// therefore recomputes the ORIGINAL `{w - 1, 0}` locally and compares.
//
// WHAT THIS DOES NOT PROVE, stated because a value test on an attention
// parameter invites the assumption: it pins the WINDOW, not the attention. These
// sites feed paged attention, and nothing on a CPU-only build runs that path.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/v1/attention/backend.h"
#include "vt/ops.h"

namespace {

// The expression all five model sites inlined, verbatim.
vt::AttentionWindow Inlined(int64_t sliding_window) {
  return vt::AttentionWindow{static_cast<int32_t>(sliding_window - 1), 0};
}

}  // namespace

TEST_CASE("attention window: the resolver equals the expression it replaces") {
  // Includes 1, where the radius is 0 and an off-by-one would be invisible in
  // any larger value, and a window far past any real context.
  for (const int64_t w : {int64_t{1}, int64_t{2}, int64_t{256}, int64_t{4096},
                          int64_t{131072}, int64_t{1} << 30}) {
    CAPTURE(w);
    const std::optional<vt::AttentionWindow> got = vllm::ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, /*model=*/w,
        vllm::v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/false);
    REQUIRE(got.has_value());
    const vt::AttentionWindow want = Inlined(w);
    CHECK(got->left == want.left);
    CHECK(got->right == want.right);
  }
}

TEST_CASE("attention window: a per-layer window overrides the model window") {
  // The precedence no inline site implements, because each had only one window
  // to hand. Pinned here so adopting the resolver cannot silently change which
  // window wins on a model that later grows a per-layer one.
  const std::optional<vt::AttentionWindow> got = vllm::ResolveAttentionWindow(
      /*per_layer=*/512, /*model=*/4096, vllm::v1::AttentionType::kDecoder,
      false);
  REQUIRE(got.has_value());
  CHECK(got->left == 511);
  CHECK(got->right == 0);
}

TEST_CASE("attention window: the model-level disable yields NO window") {
  // `SlidingWindowEnabled()` in gemma2.cpp / gemma3.cpp is this parameter, under
  // a different name and duplicated. #2388 claimed no such flag existed; it does.
  CHECK_FALSE(vllm::ResolveAttentionWindow(std::nullopt, 4096,
                                           vllm::v1::AttentionType::kDecoder,
                                           /*disable=*/true)
                  .has_value());
  // A PER-LAYER window is not disabled by the model-level flag: the flag turns
  // off the model default, not an explicit per-layer instruction.
  CHECK(vllm::ResolveAttentionWindow(512, 4096,
                                     vllm::v1::AttentionType::kDecoder,
                                     /*disable=*/true)
            .has_value());
}

TEST_CASE("attention window: no window configured resolves to nothing, not zero") {
  // The difference that matters: `nullopt` leaves attention unwindowed, while a
  // window of 0 would be a radius of -1 and mask everything.
  CHECK_FALSE(vllm::ResolveAttentionWindow(std::nullopt, std::nullopt,
                                           vllm::v1::AttentionType::kDecoder,
                                           false)
                  .has_value());
}

TEST_CASE("attention window: out-of-range is REFUSED, not silently clamped") {
  // Every adopting site already guards `> 0` by hand, so the lower bound cannot
  // change behaviour — it is pinned because that guard is what makes the
  // adoption safe, and a future site that forgets it must fail loudly.
  CHECK_THROWS_AS(vllm::ResolveAttentionWindow(std::nullopt, 0,
                                               vllm::v1::AttentionType::kDecoder,
                                               false),
                  std::invalid_argument);
  CHECK_THROWS_AS(vllm::ResolveAttentionWindow(std::nullopt, -1,
                                               vllm::v1::AttentionType::kDecoder,
                                               false),
                  std::invalid_argument);
  // The upper bound is enforced NOWHERE in the tree today.
  CHECK_THROWS_AS(
      vllm::ResolveAttentionWindow(std::nullopt, int64_t{1} << 32,
                                   vllm::v1::AttentionType::kDecoder, false),
      std::invalid_argument);
}
