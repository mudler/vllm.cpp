// Ported from: vllm/config/model.py::ModelConfig.get_diff_sampling_param
// @ 5559679229bc961848b121ccdeaa8fa5d79bec98, and the --generation-config
// selector it reads (config/model.py:298-304).
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "vllm/config/generation.h"

namespace {

// A checkpoint directory carrying config.json and, optionally, its own
// generation_config.json.
class TempModelDir {
 public:
  TempModelDir(const std::string& gen_body, const char* tag) {
    static int counter = 0;
    dir_ = (std::filesystem::temp_directory_path() /
            ("vllm_gen_cfg_test_" + std::string(tag) + "_" +
             std::to_string(counter++)))
               .string();
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ + "/config.json", std::ios::binary) << R"({
      "model_type": "llama",
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 8,
      "num_hidden_layers": 1,
      "num_attention_heads": 2,
      "vocab_size": 32,
      "max_position_embeddings": 128
    })";
    if (!gen_body.empty()) {
      std::ofstream(dir_ + "/generation_config.json", std::ios::binary) << gen_body;
    }
  }
  ~TempModelDir() { std::filesystem::remove_all(dir_); }
  std::string config_path() const { return dir_ + "/config.json"; }
  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
};

// The Qwen3.8-27B file, read live 2026-08-26 from
// https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/generation_config.json
constexpr const char* kQwen27B = R"({
  "bos_token_id": 248044,
  "do_sample": true,
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95
})";

}  // namespace

TEST_CASE("GetDiffSamplingParam: auto takes the checkpoint's own file") {
  TempModelDir model(kQwen27B, "auto");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());

  const vllm::DefaultSamplingParams d = vllm::GetDiffSamplingParam(cfg, "auto");
  CHECK_FALSE(d.empty());
  REQUIRE(d.temperature.has_value());
  CHECK(*d.temperature == doctest::Approx(1.0));
  REQUIRE(d.top_k.has_value());
  CHECK(*d.top_k == 20);
  REQUIRE(d.top_p.has_value());
  CHECK(*d.top_p == doctest::Approx(0.95));
  CHECK_FALSE(d.min_p.has_value());
  CHECK_FALSE(d.repetition_penalty.has_value());
  CHECK_FALSE(d.max_tokens.has_value());

  // "auto" is the DEFAULT selector (config/model.py:298), so calling without
  // one has to give the same answer.
  const vllm::DefaultSamplingParams implicit = vllm::GetDiffSamplingParam(cfg);
  CHECK(implicit.top_k == d.top_k);
  CHECK(implicit.top_p == d.top_p);
  CHECK(implicit.temperature == d.temperature);
}

TEST_CASE("GetDiffSamplingParam: vllm discards the file entirely") {
  // `src == "vllm"` -> `config = {}`. This is the documented escape hatch the
  // upstream warning tells a user to reach for, and it has to restore the
  // pre-#1985 behaviour exactly: nothing set, so every knob falls to neutral.
  TempModelDir model(kQwen27B, "vllm");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
  const vllm::DefaultSamplingParams d = vllm::GetDiffSamplingParam(cfg, "vllm");
  CHECK(d.empty());
  CHECK_FALSE(d.top_k.has_value());
  CHECK_FALSE(d.top_p.has_value());
  CHECK_FALSE(d.temperature.has_value());
}

TEST_CASE("GetDiffSamplingParam: a directory selector reads THAT directory") {
  TempModelDir checkpoint(kQwen27B, "ckpt");
  TempModelDir elsewhere(R"({"temperature": 0.3, "top_k": 5})", "elsewhere");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(checkpoint.config_path());

  const vllm::DefaultSamplingParams d =
      vllm::GetDiffSamplingParam(cfg, elsewhere.dir());
  // The named folder wins over the checkpoint's own sibling file.
  REQUIRE(d.temperature.has_value());
  CHECK(*d.temperature == doctest::Approx(0.3));
  REQUIRE(d.top_k.has_value());
  CHECK(*d.top_k == 5);
  CHECK_FALSE(d.top_p.has_value());  // the checkpoint's 0.95 must NOT leak in

  // A trailing separator names the same folder.
  const vllm::DefaultSamplingParams slashed =
      vllm::GetDiffSamplingParam(cfg, elsewhere.dir() + "/");
  CHECK(slashed.top_k == d.top_k);

  // A folder that does not exist yields {} rather than throwing, because
  // try_get_generation_config returns None and get_diff_sampling_param then
  // returns {}.
  vllm::DefaultSamplingParams missing;
  REQUIRE_NOTHROW(missing = vllm::GetDiffSamplingParam(cfg, "/nonexistent/xyz"));
  CHECK(missing.empty());
}

TEST_CASE("GetDiffSamplingParam: max_new_tokens is renamed to max_tokens") {
  // "Huggingface definition of max_new_tokens is equivalent to vLLM's
  // max_tokens" -- get_diff_sampling_param pops one and sets the other, so the
  // HF spelling must NOT survive into the resolved defaults.
  TempModelDir model(R"({"max_new_tokens": 512, "repetition_penalty": 1.05,
                         "min_p": 0.05})", "rename");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
  const vllm::DefaultSamplingParams d = vllm::GetDiffSamplingParam(cfg, "auto");
  REQUIRE(d.max_tokens.has_value());
  CHECK(*d.max_tokens == 512);
  REQUIRE(d.repetition_penalty.has_value());
  CHECK(*d.repetition_penalty == doctest::Approx(1.05));
  REQUIRE(d.min_p.has_value());
  CHECK(*d.min_p == doctest::Approx(0.05));
}

TEST_CASE("GetDiffSamplingParam: a checkpoint with no file resolves to empty") {
  TempModelDir model("", "nofile");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(model.config_path());
  CHECK(vllm::GetDiffSamplingParam(cfg, "auto").empty());
}

TEST_CASE("DefaultSamplingParams::ToString reports what was resolved") {
  // The startup line has to say WHICH values were taken from the checkpoint;
  // a log that only says "defaults applied" cannot be audited against the
  // oracle's own "Default vLLM sampling parameters have been overridden" line.
  vllm::DefaultSamplingParams d;
  CHECK(d.ToString().empty());
  d.temperature = 1.0;
  d.top_k = 20;
  d.top_p = 0.95;
  const std::string s = d.ToString();
  CHECK(s.find("'temperature': 1") != std::string::npos);
  CHECK(s.find("'top_k': 20") != std::string::npos);
  CHECK(s.find("'top_p': 0.95") != std::string::npos);
  CHECK(s.find("min_p") == std::string::npos);
}
