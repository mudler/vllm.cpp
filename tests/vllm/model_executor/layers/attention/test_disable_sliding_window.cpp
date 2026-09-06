// `ENG-ATTENTION-WINDOW` W3 (issue #2388) — the model-level sliding-window kill
// switch, which had no test in either of its previous spellings.
//
// WHAT W3 FOUND, and it is why this file exists rather than a rename. The switch
// was `VT_GEMMA2_SLIDING` and `VT_GEMMA3_SLIDING`: two environment knobs, on the
// kernel-internal allowlist, reaching two of the five model families that have a
// sliding window and none of the other three. W1's mutation pass proved nothing
// held either of them — rewiring gemma2's site to ignore `SlidingWindowEnabled()`
// compiled cleanly and `test_gemma2_forward` still passed 1003 assertions with
// `VT_GEMMA2_SLIDING=0`.
//
// WHY IT WAS NOT A DECISION TO ESCALATE. The row recorded W3 as a product call
// — give every model a switch, or take it from the two that have one. vLLM
// settles it: `ModelConfig.disable_sliding_window` (`vllm/config/model.py:248` @
// pin `5559679229`) is ONE model-agnostic field, and its docstring answers the
// asymmetry directly — "If the model does not support sliding window, this
// argument is ignored."
//
// WHY THE SWITCH IS NOT A CONFIG FIELD, which is the shape upstream uses.
// `config/model.py:766-769` disables the window by setting
// `hf_text_config.sliding_window = None`. That mechanism cannot work here, and
// the obstacle is a bypass rather than a missing field: `gemma2.cpp:103`,
// `gemma3.cpp:101`, `gemma4.cpp:139` and `laguna_weights.cpp:117` each read
//
//   cfg.sliding_window.value_or(RawInt(cfg.raw, "sliding_window", 0))
//
// and `cfg.raw` is the FULL UNTOUCHED checkpoint document (`hf_config.cpp:598`).
// Nulling the typed field leaves the raw one, so a mirrored implementation would
// have looked right and left the window on. The switch is therefore consumed
// where the window is USED, through `ResolveAttentionWindow`'s
// `disable_model_sliding_window` parameter.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>

#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/v1/attention/backend.h"
#include "vt/ops.h"

namespace {

constexpr int64_t kWindow = 4096;

// What every one of the five call sites computes, spelled once here so a case
// asserts the SITE's expression rather than a restatement of the resolver.
std::optional<vt::AttentionWindow> AtCallSite(std::optional<int64_t> window) {
  if (!window.has_value() || *window <= 0) return std::nullopt;
  return vllm::ResolveAttentionWindow(/*per_layer=*/std::nullopt, window,
                                      vllm::v1::AttentionType::kDecoder,
                                      vllm::DisableSlidingWindowActive());
}

}  // namespace

TEST_CASE("disable_sliding_window: the default is ENABLED, as upstream's is") {
  vllm::ResetDisableSlidingWindowForTesting();

  // Upstream's field is `= False`, so an engine that sets nothing must behave
  // exactly as every release before this switch existed.
  CHECK_FALSE(vllm::DisableSlidingWindowActive());

  const std::optional<vt::AttentionWindow> w = AtCallSite(kWindow);
  REQUIRE(w.has_value());
  CHECK(w->left == static_cast<int32_t>(kWindow - 1));
  CHECK(w->right == 0);
}

TEST_CASE("disable_sliding_window: set, the model-level window is GONE") {
  vllm::ResetDisableSlidingWindowForTesting();
  vllm::SetDisableSlidingWindow(true);
  CHECK(vllm::DisableSlidingWindowActive());

  // Not "a wider window" and not "a window of zero": no window at all, which is
  // what `hf_text_config.sliding_window = None` produces upstream. A zero-width
  // window would be a different and much worse thing -- it would mask every key.
  CHECK_FALSE(AtCallSite(kWindow).has_value());

  vllm::ResetDisableSlidingWindowForTesting();
}

TEST_CASE("disable_sliding_window: a PER-LAYER window still wins") {
  vllm::ResetDisableSlidingWindowForTesting();
  vllm::SetDisableSlidingWindow(true);

  // Upstream precedence (`attention.py:204-236`): the flag clears the MODEL-level
  // value, and an explicit per-layer value is not a model-level value. None of
  // the five sites passes one today, so this pins the resolver's contract before
  // a sixth site needs it rather than after.
  const std::optional<vt::AttentionWindow> w = vllm::ResolveAttentionWindow(
      /*per_layer=*/std::optional<int64_t>(256), /*model=*/kWindow,
      vllm::v1::AttentionType::kDecoder,
      /*disable_model_sliding_window=*/true);
  REQUIRE(w.has_value());
  CHECK(w->left == 255);

  vllm::ResetDisableSlidingWindowForTesting();
}

TEST_CASE("disable_sliding_window: a model with NO window is unaffected") {
  // Upstream: "If the model does not support sliding window, this argument is
  // ignored." This is the assertion that makes covering all five families free,
  // and it must hold in BOTH switch positions rather than only when set.
  for (const bool disabled : {false, true}) {
    CAPTURE(disabled);
    vllm::ResetDisableSlidingWindowForTesting();
    vllm::SetDisableSlidingWindow(disabled);
    CHECK_FALSE(AtCallSite(std::nullopt).has_value());
    // A checkpoint that spells "disabled" as 0 rather than null takes the same
    // path, which `hf_config.cpp:80-87` normalises and the call sites guard.
    CHECK_FALSE(AtCallSite(0).has_value());
  }
  vllm::ResetDisableSlidingWindowForTesting();
}

TEST_CASE("disable_sliding_window: the switch is LAST WRITE, not sticky") {
  // `LoadedEngine` installs it unconditionally at construction, so a second load
  // in one process must not inherit the first model's answer. That is the hazard
  // the MoE placement plan carried until #2382, in the same shape.
  vllm::ResetDisableSlidingWindowForTesting();
  vllm::SetDisableSlidingWindow(true);
  CHECK(vllm::DisableSlidingWindowActive());
  vllm::SetDisableSlidingWindow(false);
  CHECK_FALSE(vllm::DisableSlidingWindowActive());
  REQUIRE(AtCallSite(kWindow).has_value());
}
