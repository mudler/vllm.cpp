// SPEC-MTP-K-GT-1 (#81) — the MTP speculation-depth CONFIG contract.
//
// Upstream anchors, all @ 5559679229bc961848b121ccdeaa8fa5d79bec98:
//   * vllm/config/speculative.py:967-991 — an MTP-style draft resolves
//     num_speculative_tokens to n_predict when the user gives none, and a user
//     value above n_predict must be divisible by it for MTP module reuse.
//     Upstream imposes NO other bound on depth here.
//   * vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:236-238 —
//     `num_speculative_tokens == 1` EARLY-EXITS after the single draft prefill.
//     Anything above 1 falls through to prepare_decode_inputs (:597-671) and the
//     k-1 draft decode steps (:374-419).
//
// These cases pin the CONFIG TYPE, which is a faithful mirror of upstream's
// resolution and therefore carries any divisible depth. Whether THIS ENGINE can
// serve that depth is a capability question about the propose path, decided one
// level up in LoadedEngine::ResolveSpecConfig and gated by
// tests/vllm/v1/spec_decode/test_mtp_depth.cpp. Keeping the two apart is what
// makes it visible that the engine gate moved without the mirrored config type
// drifting from upstream.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "vllm/config/speculative.h"

using vllm::ParseSpeculativeConfigJson;
using vllm::SpeculativeConfig;

TEST_CASE("ResolveMtp defaults k to the head depth") {
  // speculative.py:977-979 — no user value defaults to n_predict, which is 1 on
  // both gate checkpoints. This is the DEFAULT and no depth work moves it.
  const SpeculativeConfig def = SpeculativeConfig::ResolveMtp(
      /*mtp_num_hidden_layers=*/1, /*user_num_speculative_tokens=*/std::nullopt);
  CHECK(def.method == "mtp");
  CHECK(def.n_predict == 1);
  REQUIRE(def.num_speculative_tokens.has_value());
  CHECK(*def.num_speculative_tokens == 1);
  CHECK(def.ResolvedNumSpeculativeTokens() == 1);

  const SpeculativeConfig explicit_one =
      SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1, /*user_k=*/1);
  REQUIRE(explicit_one.num_speculative_tokens.has_value());
  CHECK(*explicit_one.num_speculative_tokens == 1);
}

TEST_CASE("ResolveMtp keeps the upstream divisibility rule") {
  // speculative.py:980-991 — k above n_predict must be divisible by it. With
  // n_predict == 2, k == 3 is rejected for that reason, independently of depth.
  CHECK_THROWS_AS(
      SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/2, /*user_k=*/3),
      std::invalid_argument);
  // ... and k == 4 is accepted, which is what shows the rejection above is the
  // divisibility rule and not a depth bound smuggled into the mirrored type.
  const SpeculativeConfig four =
      SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/2, /*user_k=*/4);
  REQUIRE(four.num_speculative_tokens.has_value());
  CHECK(*four.num_speculative_tokens == 4);
}

TEST_CASE("the config type CARRIES a depth above 1, it never clamps it") {
  // The silent clamp this row exists to remove was never here: ResolveMtp always
  // carried the user's k, and the propose path ignored it. Pinning that the
  // config carries the value is what makes a future clamp visible HERE rather
  // than only as a throughput number nobody measured.
  const SpeculativeConfig three =
      SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1, /*user_k=*/3);
  REQUIRE(three.num_speculative_tokens.has_value());
  CHECK(*three.num_speculative_tokens == 3);
  CHECK(three.ResolvedNumSpeculativeTokens() == 3);

  // The user-facing JSON seam (what --speculative-config and the C ABI parse) is
  // a pure shape step: it carries the user k without knowing the checkpoint's
  // n_predict. The loader hands exactly this value to ResolveMtp against the
  // checkpoint's mtp_num_hidden_layers (model_loader.cpp:831).
  const SpeculativeConfig parsed = ParseSpeculativeConfigJson(
      R"({"method":"mtp","num_speculative_tokens":4})");
  CHECK(parsed.method == "mtp");
  REQUIRE(parsed.num_speculative_tokens.has_value());
  CHECK(*parsed.num_speculative_tokens == 4);
}

TEST_CASE("the other proposers keep their own depth resolution") {
  // These genuinely serve k>1 today (runner.cpp:2273-2282 moves the whole
  // per-request n-gram vector; :2196 gives DFlash a 1+k block), and they resolve
  // through their OWN entry points, so no MTP-side depth gate can reach them.
  const SpeculativeConfig ngram = ParseSpeculativeConfigJson(
      R"({"method":"ngram","num_speculative_tokens":5})");
  REQUIRE(ngram.num_speculative_tokens.has_value());
  CHECK(*ngram.num_speculative_tokens == 5);

  const SpeculativeConfig dflash = SpeculativeConfig::ResolveDflash(3);
  REQUIRE(dflash.num_speculative_tokens.has_value());
  CHECK(*dflash.num_speculative_tokens == 3);
}
