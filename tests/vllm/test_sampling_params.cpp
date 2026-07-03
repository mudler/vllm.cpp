// Tests for the SamplingParams port (vllm/sampling_params.py @ e24d1b24).
#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>

#include "vllm/sampling_params.h"

using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SamplingType;

TEST_CASE("SamplingParams defaults match upstream") {
  SamplingParams p;
  CHECK(p.n == 1);
  CHECK(p.presence_penalty == doctest::Approx(0.0));
  CHECK(p.frequency_penalty == doctest::Approx(0.0));
  CHECK(p.repetition_penalty == doctest::Approx(1.0));
  CHECK(p.temperature == doctest::Approx(1.0));
  CHECK(p.top_p == doctest::Approx(1.0));
  CHECK(p.top_k == 0);
  CHECK(p.min_p == doctest::Approx(0.0));
  CHECK_FALSE(p.seed.has_value());
  CHECK(p.stop.empty());
  CHECK(p.stop_token_ids.empty());
  CHECK_FALSE(p.ignore_eos);
  REQUIRE(p.max_tokens.has_value());
  CHECK(*p.max_tokens == 16);
  CHECK(p.min_tokens == 0);
  CHECK_FALSE(p.logprobs.has_value());
  CHECK_FALSE(p.prompt_logprobs.has_value());
  CHECK(p.detokenize);
  CHECK(p.skip_special_tokens);
  CHECK(p.spaces_between_special_tokens);
  CHECK_FALSE(p.include_stop_str_in_output);
  CHECK(p.output_kind == RequestOutputKind::kCumulative);
}

TEST_CASE("eos_token_id lives on SamplingParams (upstream _eos_token_id)") {
  // Engine-populated field mirroring SamplingParams._eos_token_id (exposed
  // upstream via the eos_token_id property that check_stop reads).
  SamplingParams p;
  // Defaults to unset (upstream default None).
  CHECK_FALSE(p.eos_token_id.has_value());
  // Settable by the engine.
  p.eos_token_id = 2;
  REQUIRE(p.eos_token_id.has_value());
  CHECK(*p.eos_token_id == 2);
  // Verify()/PostInit() ignore it (upstream does not validate _eos_token_id).
  CHECK_NOTHROW(p.Verify());
  CHECK_NOTHROW(p.PostInit());
  CHECK(*p.eos_token_id == 2);
}

TEST_CASE("enum int values are load-bearing and match upstream") {
  CHECK(static_cast<int>(SamplingType::kGreedy) == 0);
  CHECK(static_cast<int>(SamplingType::kRandom) == 1);
  CHECK(static_cast<int>(SamplingType::kRandomSeed) == 2);
  CHECK(static_cast<int>(RequestOutputKind::kCumulative) == 0);
  CHECK(static_cast<int>(RequestOutputKind::kDelta) == 1);
  CHECK(static_cast<int>(RequestOutputKind::kFinalOnly) == 2);
}

TEST_CASE("Type() derivation mirrors sampling_type property") {
  SUBCASE("temperature 0 => greedy") {
    SamplingParams p;
    p.temperature = 0.0;
    CHECK(p.Type() == SamplingType::kGreedy);
  }
  SUBCASE("near-zero temperature (< _SAMPLING_EPS) => greedy") {
    SamplingParams p;
    p.temperature = 1e-6;
    CHECK(p.Type() == SamplingType::kGreedy);
  }
  SUBCASE("positive temperature, no seed => random") {
    SamplingParams p;
    p.temperature = 0.8;
    CHECK(p.Type() == SamplingType::kRandom);
  }
  SUBCASE("positive temperature + seed => random_seed") {
    SamplingParams p;
    p.temperature = 0.8;
    p.seed = 42;
    CHECK(p.Type() == SamplingType::kRandomSeed);
  }
  SUBCASE("greedy takes precedence over seed") {
    SamplingParams p;
    p.temperature = 0.0;
    p.seed = 42;
    CHECK(p.Type() == SamplingType::kGreedy);
  }
}

TEST_CASE("valid greedy and random params verify cleanly") {
  SUBCASE("greedy") {
    SamplingParams p;
    p.temperature = 0.0;
    CHECK_NOTHROW(p.Verify());
    CHECK_NOTHROW(p.PostInit());
  }
  SUBCASE("random with all knobs") {
    SamplingParams p;
    p.temperature = 0.9;
    p.top_p = 0.9;
    p.top_k = 50;
    p.min_p = 0.1;
    p.presence_penalty = 0.5;
    p.frequency_penalty = 0.5;
    p.repetition_penalty = 1.2;
    p.min_tokens = 2;
    p.max_tokens = 128;
    p.logprobs = 5;
    p.prompt_logprobs = 1;
    CHECK_NOTHROW(p.Verify());
    CHECK_NOTHROW(p.PostInit());
  }
}

TEST_CASE("_verify_args failures throw") {
  SUBCASE("n < 1") {
    SamplingParams p;
    p.n = 0;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("n above VLLM_MAX_N_SEQUENCES") {
    SamplingParams p;
    p.n = vllm::kMaxNSequences + 1;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("presence_penalty out of [-2, 2]") {
    SamplingParams p;
    p.presence_penalty = 2.5;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("frequency_penalty out of [-2, 2]") {
    SamplingParams p;
    p.frequency_penalty = -2.5;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("repetition_penalty <= 0") {
    SamplingParams p;
    p.repetition_penalty = 0.0;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("non-finite repetition_penalty") {
    SamplingParams p;
    p.repetition_penalty = std::nan("");
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("non-finite temperature") {
    SamplingParams p;
    p.temperature = std::nan("");
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("negative temperature") {
    SamplingParams p;
    p.temperature = -0.1;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("temperature > 2") {
    SamplingParams p;
    p.temperature = 2.5;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("top_p out of (0, 1]") {
    SamplingParams p;
    p.top_p = 0.0;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
    p.top_p = 1.5;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("top_k < -1") {
    SamplingParams p;
    p.top_k = -2;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("top_k == -1 (disabled) is accepted") {
    SamplingParams p;
    p.top_k = -1;
    CHECK_NOTHROW(p.Verify());
  }
  SUBCASE("min_p out of [0, 1]") {
    SamplingParams p;
    p.min_p = 1.5;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("max_tokens < 1") {
    SamplingParams p;
    p.max_tokens = 0;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("min_tokens < 0") {
    SamplingParams p;
    p.min_tokens = -1;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("min_tokens > max_tokens") {
    SamplingParams p;
    p.max_tokens = 4;
    p.min_tokens = 8;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("logprobs negative but not -1") {
    SamplingParams p;
    p.logprobs = -2;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("logprobs -1 (all) accepted") {
    SamplingParams p;
    p.logprobs = -1;
    CHECK_NOTHROW(p.Verify());
  }
  SUBCASE("prompt_logprobs negative but not -1") {
    SamplingParams p;
    p.prompt_logprobs = -3;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("empty stop string") {
    SamplingParams p;
    p.stop = {""};
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("stop without detokenize") {
    SamplingParams p;
    p.stop = {"foo"};
    p.detokenize = false;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
}

TEST_CASE("PostInit normalizes like __post_init__") {
  SUBCASE("temperature in (0, _MAX_TEMP) is clamped up") {
    SamplingParams p;
    p.temperature = 0.001;
    p.PostInit();
    CHECK(p.temperature == doctest::Approx(vllm::kMaxTemp));
  }
  SUBCASE("seed == -1 becomes unset") {
    SamplingParams p;
    p.seed = -1;
    p.PostInit();
    CHECK_FALSE(p.seed.has_value());
  }
  SUBCASE("temperature 0 forces greedy sub-params") {
    SamplingParams p;
    p.temperature = 0.0;
    p.top_p = 0.5;
    p.top_k = 20;
    p.min_p = 0.3;
    p.PostInit();
    CHECK(p.top_p == doctest::Approx(1.0));
    CHECK(p.top_k == 0);
    CHECK(p.min_p == doctest::Approx(0.0));
    CHECK(p.Type() == SamplingType::kGreedy);
  }
  SUBCASE("greedy with n > 1 is rejected") {
    SamplingParams p;
    p.temperature = 0.0;
    p.n = 2;
    CHECK_THROWS_AS(p.PostInit(), std::runtime_error);
  }
}

// Locks the Verify()/PostInit() split so a future refactor cannot silently
// drop the mandatory __post_init__ normalization + greedy n-check. Upstream
// always runs __post_init__ at construction; Verify() alone is NOT a
// substitute (it neither normalizes fields nor enforces the greedy n==1 rule).
TEST_CASE("PostInit contract: mandatory __post_init__ equivalent") {
  SUBCASE("Verify() alone accepts a state upstream rejects (greedy, n>1)") {
    // temperature=0 (greedy) with n=2 is invalid upstream, but Verify() does
    // not run the greedy n-check, so it must NOT throw here.
    SamplingParams p;
    p.temperature = 0.0;
    p.n = 2;
    CHECK_NOTHROW(p.Verify());
    // The SAME object, run through PostInit(), IS rejected — proving Verify()
    // by itself is insufficient and PostInit() is the enforcing path.
    CHECK_THROWS_AS(p.PostInit(), std::runtime_error);
  }
  SUBCASE("PostInit() forces greedy sub-params at temperature 0") {
    SamplingParams p;
    p.temperature = 0.0;
    p.top_p = 0.5;
    p.top_k = 20;
    p.min_p = 0.3;
    p.PostInit();
    CHECK(p.top_p == doctest::Approx(1.0));
    CHECK(p.top_k == 0);
    CHECK(p.min_p == doctest::Approx(0.0));
    CHECK(p.Type() == SamplingType::kGreedy);
  }
  SUBCASE("PostInit() clamp precedes greedy check: 1e-6 is NOT greedy") {
    // Locks the ORDER in __post_init__: the (0, _MAX_TEMP) clamp raises a
    // near-zero positive temperature to _MAX_TEMP (1e-2) BEFORE the
    // < _SAMPLING_EPS (1e-5) greedy test runs. So temperature=1e-6 ends up
    // clamped and random, and greedy sub-params are NOT forced.
    SamplingParams p;
    p.temperature = 1e-6;
    p.top_p = 0.5;
    p.top_k = 20;
    p.min_p = 0.3;
    p.PostInit();
    CHECK(p.temperature == doctest::Approx(vllm::kMaxTemp));
    CHECK(p.top_p == doctest::Approx(0.5));
    CHECK(p.top_k == 20);
    CHECK(p.min_p == doctest::Approx(0.3));
    CHECK(p.Type() == SamplingType::kRandom);
  }
  SUBCASE("PostInit() clamps temperature in (0, _MAX_TEMP) up to _MAX_TEMP") {
    SamplingParams p;
    p.temperature = 0.005;  // in (0, 1e-2)
    p.PostInit();
    CHECK(p.temperature == doctest::Approx(vllm::kMaxTemp));
  }
  SUBCASE("PostInit() drops seed == -1") {
    SamplingParams p;
    p.seed = -1;
    p.PostInit();
    CHECK_FALSE(p.seed.has_value());
  }
}
