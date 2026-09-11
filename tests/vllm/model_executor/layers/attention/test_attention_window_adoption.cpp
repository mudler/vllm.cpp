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

// The expression all five model sites inlined, verbatim. W2 (#2388) found the
// SAME expression at three more, which is why one helper covers both waves:
// `v1/attention/backend.cpp`, `mla_chunked_context.h` and `deepseek_v4_dsa.cpp`.
vt::AttentionWindow Inlined(int64_t sliding_window) {
  return vt::AttentionWindow{static_cast<int32_t>(sliding_window - 1), 0};
}

// W2's three sites hold an `int64_t`, not an `optional`, and each guarded the
// inline expression with `if (sliding_window > 0)`. Adoption turns that guard
// into the optional the resolver takes, so this helper is the ONE thing W2
// actually changed about those sites and is what the case below pins.
std::optional<int64_t> W2Guard(int64_t sliding_window) {
  return sliding_window > 0 ? std::optional<int64_t>(sliding_window)
                            : std::nullopt;
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
  // W3 retired the two duplicated local spellings this comment used to name.
  // `SlidingWindowEnabled()` in gemma2.cpp / gemma3.cpp read `VT_GEMMA2_SLIDING`
  // and `VT_GEMMA3_SLIDING`; all five call sites now pass
  // `DisableSlidingWindowActive()`, the one switch mirroring
  // `ModelConfig.disable_sliding_window`. See test_disable_sliding_window.cpp.
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


// W2 (#2388) — the three sites that are NOT model code.
//
// WHY THIS NEEDED NO DEVICE, which is the finding rather than the code. W2 sat
// deferred because those sites "sit under paths no CPU-only build exercises, and
// moving them without a device gate would be the unverified change this work has
// been closing". That conflated two different questions. Proving the resolver
// returns the SAME VALUE is pure arithmetic and needs no device — it is exactly
// what W1 shipped for the five model sites, in the case at the top of this file.
// Proving the site is REACHED is the other question, and it is unchanged by W2:
// no gate here executes those paths, before or after.
TEST_CASE("attention window: W2's guard translation preserves the old behaviour") {
  // Every site guarded `if (sliding_window > 0)`. A 0 or a negative left
  // `window_size` as `std::nullopt` and the kernel on its full-prefix loop, and
  // that must still be true now the resolver decides it.
  for (const int64_t w : {int64_t{-1}, int64_t{0}}) {
    CAPTURE(w);
    CHECK_FALSE(vllm::ResolveAttentionWindow(std::nullopt, W2Guard(w),
                                             vllm::v1::AttentionType::kDecoder,
                                             false)
                    .has_value());
  }
  // And a real window resolves to precisely what the three sites inlined.
  for (const int64_t w : {int64_t{1}, int64_t{513}, int64_t{4096}}) {
    CAPTURE(w);
    const std::optional<vt::AttentionWindow> got = vllm::ResolveAttentionWindow(
        std::nullopt, W2Guard(w), vllm::v1::AttentionType::kDecoder, false);
    REQUIRE(got.has_value());
    CHECK(got->left == Inlined(w).left);
    CHECK(got->right == Inlined(w).right);
  }
}

TEST_CASE("attention window: W2's sites now REFUSE a window they used to truncate") {
  // This is a deliberate behaviour CHANGE and the only one W2 makes.
  // `static_cast<int32_t>(sliding_window - 1)` on a window past INT32_MAX wraps
  // and hands the kernel a plausible small — or negative — radius. The three
  // sites did that silently; the resolver refuses by name.
  CHECK_THROWS_AS(vllm::ResolveAttentionWindow(
                      std::nullopt, W2Guard(int64_t{1} << 40),
                      vllm::v1::AttentionType::kDecoder, false),
                  std::invalid_argument);
}

TEST_CASE("attention window: W2's sites OBEY --disable-sliding-window") {
  // The gap W3 left and W2 closes. After W3 the flag reached five of the eight
  // sites that carry a window; these three ignored it, so the same engine would
  // have honoured the switch on Gemma and quietly kept the window on dots3-note's
  // windowed decode, DeepSeek-V4's DSA and the MLA chunked-prefill context.
  vllm::ResetDisableSlidingWindowForTesting();
  vllm::SetDisableSlidingWindow(true);
  CHECK_FALSE(vllm::ResolveAttentionWindow(std::nullopt, W2Guard(4096),
                                           vllm::v1::AttentionType::kDecoder,
                                           vllm::DisableSlidingWindowActive())
                  .has_value());
  vllm::ResetDisableSlidingWindowForTesting();
  REQUIRE(vllm::ResolveAttentionWindow(std::nullopt, W2Guard(4096),
                                       vllm::v1::AttentionType::kDecoder,
                                       vllm::DisableSlidingWindowActive())
              .has_value());
}
