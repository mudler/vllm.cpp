#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "vllm/transformers_utils/hf_config.h"

namespace {

// Writes `body` to a unique file under the system temp dir and returns its
// path. Removed in the destructor so test runs don't accumulate files.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_hf_config_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Qwen3-Next-like hybrid MoE config (key names per upstream
// vllm/transformers_utils/configs/qwen3_next.py).
constexpr const char* kHybridJson = R"({
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3NextForCausalLM"],
  "hidden_size": 2048,
  "num_hidden_layers": 4,
  "vocab_size": 151936,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                  "full_attention"],
  "intermediate_size": 5120,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 512,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "rope_theta": 5000000.0,
  "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "max_position_embeddings": 262144,
  "torch_dtype": "bfloat16"
})";

// Minimal llama-like config: no MoE/GDN keys, no explicit head_dim, no
// partial_rotary_factor, no layer_types.
constexpr const char* kLlamaJson = R"({
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "vocab_size": 32000,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "rms_norm_eps": 1e-05,
  "max_position_embeddings": 4096,
  "torch_dtype": "float16"
})";

}  // namespace

TEST_CASE("LoadHfConfig parses a Qwen3-Next-like hybrid MoE config") {
  TempJson f(kHybridJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  CHECK(cfg.model_type == "qwen3_5_moe");
  REQUIRE(cfg.architectures.size() == 1);
  CHECK(cfg.architectures[0] == "Qwen3NextForCausalLM");
  CHECK(cfg.hidden_size == 2048);
  CHECK(cfg.num_hidden_layers == 4);
  CHECK(cfg.vocab_size == 151936);
  CHECK(cfg.num_attention_heads == 16);
  CHECK(cfg.num_key_value_heads == 2);
  CHECK(cfg.head_dim == 256);  // explicit key wins over hidden/heads = 128

  REQUIRE(cfg.layer_types.size() == 4);
  CHECK(cfg.layer_types[0] == "linear_attention");
  CHECK(cfg.layer_types[3] == "full_attention");

  CHECK(cfg.intermediate_size == 5120);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.num_experts_per_tok == 8);
  CHECK(cfg.moe_intermediate_size == 512);
  CHECK(cfg.shared_expert_intermediate_size == 512);

  CHECK(cfg.linear_num_key_heads == 16);
  CHECK(cfg.linear_num_value_heads == 32);
  CHECK(cfg.linear_key_head_dim == 128);
  CHECK(cfg.linear_value_head_dim == 128);
  CHECK(cfg.linear_conv_kernel_dim == 4);

  CHECK(cfg.rope_theta == doctest::Approx(5000000.0));
  CHECK(cfg.rotary_dim == 64);  // 0.25 * 256
  CHECK(cfg.rms_norm_eps == doctest::Approx(1e-6));
  CHECK(cfg.max_position_embeddings == 262144);
  CHECK(cfg.torch_dtype == "bfloat16");

  // Raw doc keeps untyped fields accessible.
  CHECK(cfg.raw.at("partial_rotary_factor").get<double>() ==
        doctest::Approx(0.25));
}

TEST_CASE("LoadHfConfig parses a minimal llama-like config with defaults") {
  TempJson f(kLlamaJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  CHECK(cfg.model_type == "llama");
  CHECK(cfg.hidden_size == 4096);
  CHECK(cfg.num_hidden_layers == 32);
  CHECK(cfg.num_attention_heads == 32);
  // num_key_value_heads absent -> num_attention_heads (MHA).
  CHECK(cfg.num_key_value_heads == 32);
  // head_dim absent -> hidden_size / num_attention_heads.
  CHECK(cfg.head_dim == 128);
  // No partial_rotary_factor -> full rotary.
  CHECK(cfg.rotary_dim == 128);
  // rope_theta absent -> upstream default 10000.0.
  CHECK(cfg.rope_theta == doctest::Approx(10000.0));

  CHECK(cfg.layer_types.empty());  // empty = all full attention

  // MoE / GDN keys absent -> zeros.
  CHECK(cfg.num_experts == 0);
  CHECK(cfg.num_experts_per_tok == 0);
  CHECK(cfg.moe_intermediate_size == 0);
  CHECK(cfg.shared_expert_intermediate_size == 0);
  CHECK(cfg.linear_num_key_heads == 0);
  CHECK(cfg.linear_num_value_heads == 0);
  CHECK(cfg.linear_key_head_dim == 0);
  CHECK(cfg.linear_value_head_dim == 0);
  CHECK(cfg.linear_conv_kernel_dim == 0);

  CHECK(cfg.torch_dtype == "float16");
}

TEST_CASE("LoadHfConfig throws when required fields are missing") {
  SUBCASE("missing model_type") {
    TempJson f(R"({"hidden_size": 64, "num_hidden_layers": 2})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
  SUBCASE("missing hidden_size") {
    TempJson f(R"({"model_type": "llama", "num_hidden_layers": 2})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
  SUBCASE("missing num_hidden_layers") {
    TempJson f(R"({"model_type": "llama", "hidden_size": 64})");
    CHECK_THROWS_AS(vllm::LoadHfConfig(f.path()), std::runtime_error);
  }
}

TEST_CASE("LoadHfConfig error paths mention the file path") {
  SUBCASE("file not found") {
    const std::string missing = "/nonexistent/vllm_cpp_no_such_config.json";
    try {
      vllm::LoadHfConfig(missing);
      FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
      CHECK(std::string(e.what()).find(missing) != std::string::npos);
    }
  }
  SUBCASE("invalid JSON") {
    TempJson f("{ not json !!");
    try {
      vllm::LoadHfConfig(f.path());
      FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
      CHECK(std::string(e.what()).find(f.path()) != std::string::npos);
    }
  }
}
