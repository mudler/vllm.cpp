// Tests for the V1 InputProcessor text path
// (vllm/v1/engine/input_processor.py::process_inputs @ e24d1b24).
//
// M1.8 Task 2 is a pure host-logic port (no goldens). The InputProcessor is the
// "constructing unit" that runs the SamplingParams __post_init__ (PostInit) our
// M1.1 SamplingParams deferred. Cases:
//   (a) text prompt   -> EngineCoreRequest.prompt_token_ids == tokenizer.Encode
//                        (the byte-level BPE path); request_id + arrival_time set.
//   (b) PostInit ran  -> a near-zero temperature is clamped to _MAX_TEMP; a
//                        greedy (temperature 0) params has its top_p/top_k/min_p
//                        normalized. (Verify() alone would not normalize these.)
//   (c) invalid params-> Verify (run inside PostInit) throws on an out-of-range
//                        field (e.g. top_p > 1).
//   (d) max_tokens    -> unset (nullopt) defaults to max_model_len - prompt_len.
//   (e) eos/stop      -> primary eos is written to sampling_params.eos_token_id
//                        unless ignore_eos; a secondary eos id (a list config)
//                        is merged into stop_token_ids unless ignore_eos.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/types.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::SamplingParams;
using vllm::SamplingType;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCoreRequest;
using vllm::v1::InputProcessor;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36";

// One tokenizer load for the whole binary (the golden tokenizer.json is ~20MB).
const Tokenizer& GoldenTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJson(kGoldenDir + "/tokenizer.json");
  return tok;
}

// Builds a minimal HfConfig carrying just what the InputProcessor reads:
// max_position_embeddings (-> max_model_len) and raw["eos_token_id"].
HfConfig MakeConfig(int64_t max_len, const json& eos_token_id) {
  HfConfig cfg;
  cfg.model_type = "qwen3_5_moe";
  cfg.hidden_size = 8;
  cfg.num_hidden_layers = 1;
  cfg.max_position_embeddings = max_len;
  cfg.raw = json::object();
  if (!eos_token_id.is_null()) {
    cfg.raw["eos_token_id"] = eos_token_id;
  }
  return cfg;
}

}  // namespace

TEST_CASE("process_inputs tokenizes the text prompt (byte-level BPE path)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, /*eos=*/json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  SamplingParams params;  // defaults: max_tokens=16
  EngineCoreRequest req = proc.process_inputs("req-1", prompt, params,
                                              /*arrival_time=*/123.0);

  CHECK(req.request_id == "req-1");
  CHECK(req.arrival_time == doctest::Approx(123.0));
  CHECK(req.prompt_token_ids == tok.Encode(prompt));
  CHECK_FALSE(req.prompt_token_ids.empty());
  // Default max_tokens is preserved when the caller sets it.
  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens == 16);
}

TEST_CASE("process_inputs runs SamplingParams PostInit (normalization)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SUBCASE("near-zero temperature is clamped up to _MAX_TEMP") {
    SamplingParams params;
    params.temperature = 0.005;  // in (0, _MAX_TEMP=1e-2)
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(vllm::kMaxTemp));
  }

  SUBCASE("greedy (temperature 0) normalizes top_p/top_k/min_p") {
    SamplingParams params;
    params.temperature = 0.0;
    params.top_p = 0.5;
    params.top_k = 20;
    params.min_p = 0.3;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(0.0));
    CHECK(req.sampling_params.top_p == doctest::Approx(1.0));
    CHECK(req.sampling_params.top_k == 0);
    CHECK(req.sampling_params.min_p == doctest::Approx(0.0));
    CHECK(req.sampling_params.Type() == SamplingType::kGreedy);
  }
}

TEST_CASE("process_inputs throws on invalid SamplingParams (Verify)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams bad;
  bad.top_p = 2.0;  // must be in (0, 1]
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", bad), std::runtime_error);

  SamplingParams greedy_multi;
  greedy_multi.temperature = 0.0;
  greedy_multi.n = 2;  // n must be 1 under greedy sampling
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", greedy_multi),
                  std::runtime_error);
}

TEST_CASE("process_inputs defaults max_tokens from max_model_len - prompt_len") {
  const Tokenizer& tok = GoldenTokenizer();
  const int64_t kMaxLen = 4096;
  HfConfig cfg = MakeConfig(kMaxLen, json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  const int prompt_len = static_cast<int>(tok.Encode(prompt).size());

  SamplingParams params;
  params.max_tokens = std::nullopt;  // "unset" -> generate up to max_model_len
  EngineCoreRequest req = proc.process_inputs("r", prompt, params);

  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens ==
        static_cast<int>(kMaxLen) - prompt_len);
}

TEST_CASE("process_inputs wires eos/stop token ids") {
  const Tokenizer& tok = GoldenTokenizer();

  SUBCASE("scalar eos is written to sampling_params.eos_token_id") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 151643);
    // A single eos id has no secondary ids -> stop_token_ids untouched.
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("ignore_eos leaves eos_token_id unset and stop_token_ids untouched") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("list eos: primary is eos_token_id, secondary merged into stop ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.stop_token_ids = {200, 999};  // 200 dedups against the eos ids
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 100);  // first element = primary
    // secondary eos ids {200, 300} unioned with existing {200, 999}.
    std::vector<int32_t> expected = {200, 300, 999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }

  SUBCASE("list eos with ignore_eos does not touch stop_token_ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    params.stop_token_ids = {999};
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    std::vector<int32_t> expected = {999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }
}
