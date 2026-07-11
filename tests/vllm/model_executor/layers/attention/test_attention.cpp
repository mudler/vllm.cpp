// Ported from:
//   vllm/model_executor/layers/attention/attention.py:204-236,387-400
//   vllm/v1/attention/backends/flash_attn.py:255-300
// Tests carried from tests/v1/attention/test_attention_backends.py:745-867,
// tests/kernels/attention/test_flashinfer.py:296-482, and
// tests/v1/e2e/general/test_correctness_sliding_window.py:19-78.
// @ e24d1b24fe96.
#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "vllm/model_executor/layers/attention/attention.h"

using vllm::MakePagedAttentionArgs;
using vllm::ResolveAttentionWindow;
using vllm::v1::AttentionLayer;
using vllm::v1::AttentionType;

TEST_CASE("Attention sliding_window resolves per-layer precedence and backend pair") {
  SUBCASE("missing model and per-layer values preserve full attention") {
    CHECK_FALSE(ResolveAttentionWindow(std::nullopt, std::nullopt,
                                       AttentionType::kDecoder)
                    .has_value());
  }

  SUBCASE("decoder maps W to (W-1, 0)") {
    const auto window = ResolveAttentionWindow(std::nullopt, 4096,
                                               AttentionType::kDecoder);
    REQUIRE(window.has_value());
    CHECK(window->left == 4095);
    CHECK(window->right == 0);
  }

  SUBCASE("encoder-only symmetrizes to (W-1, W-1)") {
    const auto window = ResolveAttentionWindow(std::nullopt, 128,
                                               AttentionType::kEncoderOnly);
    REQUIRE(window.has_value());
    CHECK(window->left == 127);
    CHECK(window->right == 127);
  }

  SUBCASE("per-layer value overrides model-level value") {
    const auto window = ResolveAttentionWindow(8, 4096,
                                               AttentionType::kDecoder);
    REQUIRE(window.has_value());
    CHECK(window->left == 7);
    CHECK(window->right == 0);
  }

  SUBCASE("disable-sliding-window clears the model value but not an explicit per-layer value") {
    CHECK_FALSE(ResolveAttentionWindow(std::nullopt, 4096,
                                       AttentionType::kDecoder,
                                       /*disable_model_sliding_window=*/true)
                    .has_value());
    const auto window = ResolveAttentionWindow(
        8, 4096, AttentionType::kDecoder,
        /*disable_model_sliding_window=*/true);
    REQUIRE(window.has_value());
    CHECK(window->left == 7);
    CHECK(window->right == 0);
  }
}

TEST_CASE("Attention sliding_window rejects non-positive and unrepresentable widths") {
  CHECK_THROWS_AS(ResolveAttentionWindow(0, std::nullopt,
                                         AttentionType::kDecoder),
                  std::invalid_argument);
  CHECK_THROWS_AS(ResolveAttentionWindow(-1, std::nullopt,
                                         AttentionType::kDecoder),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      ResolveAttentionWindow(
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
          std::nullopt, AttentionType::kDecoder),
      std::invalid_argument);
}

TEST_CASE("AttentionLayer carries one window into PagedAttentionArgs") {
  AttentionLayer layer;
  layer.window_size = ResolveAttentionWindow(17, std::nullopt,
                                             AttentionType::kDecoder);
  const vt::PagedAttentionArgs args = MakePagedAttentionArgs(0.125f, true, layer);
  CHECK(args.scale == doctest::Approx(0.125f));
  CHECK(args.causal);
  REQUIRE(args.window_size.has_value());
  CHECK(args.window_size->left == 16);
  CHECK(args.window_size->right == 0);
}

TEST_CASE("sliding-window backend tensor-parallel variants are tracked by PAR-TP" *
          doctest::skip(true) *
          doctest::description(
              "tests/v1/attention/test_attention_backends.py TP=2/4 requires PAR-TP")) {
  MESSAGE("SKIP PAR-TP: W2 ports the TP=1 operator semantics; distributed heads remain unported");
}

TEST_CASE("sliding-window FlashInfer fallback is tracked by KERNEL-ATTN-FLASHINFER-TRTLLM" *
          doctest::skip(true) *
          doctest::description(
              "tests/kernels/attention/test_flashinfer.py requires the FlashInfer backend row")) {
  MESSAGE("SKIP KERNEL-ATTN-FLASHINFER-TRTLLM: no FlashInfer runtime dependency is shipped");
}

TEST_CASE("sliding-window StarCoder2 feature e2e is tracked by the Transformers model alias" *
          doctest::skip(true) *
          doctest::description(
              "test_correctness_sliding_window.py requires "
              "MODEL-HFALIAS-transformers-transformers-for-causal-lm")) {
  MESSAGE("SKIP MODEL-HFALIAS-transformers-transformers-for-causal-lm: StarCoder2 model port absent");
}

TEST_CASE("sliding-window Gemma3 hybrid feature e2e is tracked by the Gemma3 model row" *
          doctest::skip(true) *
          doctest::description(
              "test_correctness_sliding_window.py requires MODEL-TEXT-gemma3-gemma3-for-causal-lm")) {
  MESSAGE("SKIP MODEL-TEXT-gemma3-gemma3-for-causal-lm: hybrid Gemma3 model port absent");
}
