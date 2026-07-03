#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
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

// Multimodal wrapper config (Qwen3_5MoeForConditionalGeneration): the
// text-model fields are nested under `text_config`; only architectures,
// model_type and (here) torch_dtype live at the top level. Mirrors the real 35B
// config.json structure that LoadHfConfig must resolve via the upstream
// get_text_config() path.
constexpr const char* kNestedWrapperJson = R"({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "torch_dtype": "bfloat16",
  "vision_config": {
    "model_type": "qwen3_5_moe",
    "depth": 27,
    "hidden_size": 1152
  },
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention",
                    "linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention"],
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
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 32768
  }
})";

}  // namespace

TEST_CASE("LoadHfConfig resolves nested text_config for a wrapper config") {
  TempJson f(kNestedWrapperJson);
  vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());

  // architectures + model_type come from the top-level wrapper.
  REQUIRE(cfg.architectures.size() == 1);
  CHECK(cfg.architectures[0] == "Qwen3_5MoeForConditionalGeneration");
  CHECK(cfg.model_type == "qwen3_5_moe");

  // Text-model fields resolved from the nested text_config, not the wrapper.
  CHECK(cfg.num_hidden_layers == 40);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.num_experts_per_tok == 8);
  REQUIRE(cfg.layer_types.size() == 40);
  CHECK(cfg.layer_types[0] == "linear_attention");
  CHECK(cfg.layer_types[3] == "full_attention");
  CHECK(cfg.layer_types[39] == "full_attention");
  CHECK(cfg.hidden_size == 2048);
  CHECK(cfg.vocab_size == 248320);
  CHECK(cfg.num_attention_heads == 16);
  CHECK(cfg.num_key_value_heads == 2);
  CHECK(cfg.head_dim == 256);
  CHECK(cfg.moe_intermediate_size == 512);
  CHECK(cfg.shared_expert_intermediate_size == 512);
  CHECK(cfg.linear_num_key_heads == 16);
  CHECK(cfg.linear_num_value_heads == 32);
  CHECK(cfg.linear_key_head_dim == 128);
  CHECK(cfg.linear_value_head_dim == 128);
  CHECK(cfg.linear_conv_kernel_dim == 4);
  CHECK(cfg.rope_theta == doctest::Approx(5000000.0));
  CHECK(cfg.rms_norm_eps == doctest::Approx(1e-6));
  CHECK(cfg.max_position_embeddings == 32768);

  // partial_rotary_factor absent -> qwen family default 0.25 applies from the
  // resolved config (0.25 * 256 = 64).
  CHECK(cfg.rotary_dim == 64);

  // torch_dtype lives only at the top level here; fall back to the wrapper.
  CHECK(cfg.torch_dtype == "bfloat16");

  // raw keeps the full wrapper doc, including nested sub-configs.
  CHECK(cfg.raw.at("text_config").at("num_hidden_layers").get<int64_t>() == 40);
}

TEST_CASE("LoadHfConfig resolves llm_config alias and thinker_config.text_config") {
  SUBCASE("llm_config alias (upstream _CONFIG_ATTRS_MAPPING)") {
    TempJson f(R"({
      "architectures": ["SomeWrapperForCausalLM"],
      "model_type": "wrapper",
      "llm_config": {
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28
      }
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.model_type == "wrapper");
    CHECK(cfg.hidden_size == 3584);
    CHECK(cfg.num_hidden_layers == 28);
  }
  SUBCASE("thinker_config.text_config") {
    TempJson f(R"({
      "architectures": ["ThinkerForConditionalGeneration"],
      "model_type": "thinker_wrapper",
      "thinker_config": {
        "text_config": {
          "hidden_size": 4096,
          "num_hidden_layers": 36,
          "num_attention_heads": 32
        }
      }
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.model_type == "thinker_wrapper");
    CHECK(cfg.hidden_size == 4096);
    CHECK(cfg.num_hidden_layers == 36);
  }
}

TEST_CASE("LoadHfConfig on the real 35B config.json (skipped if absent)") {
  // Point VLLM_CPP_QWEN35_CONFIG at a real
  // Qwen3_5MoeForConditionalGeneration config.json to exercise this on dgx.
  const char* env = std::getenv("VLLM_CPP_QWEN35_CONFIG");
  if (env == nullptr || !std::filesystem::exists(env)) {
    MESSAGE("skipping: VLLM_CPP_QWEN35_CONFIG not set to an existing file");
    return;
  }
  vllm::HfConfig cfg = vllm::LoadHfConfig(env);
  CHECK(cfg.num_hidden_layers == 40);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.layer_types.size() == 40);
}

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

TEST_CASE("LoadHfConfig applies upstream rotary and head_dim defaults") {
  SUBCASE("qwen family without partial_rotary_factor defaults to 0.25") {
    // Upstream Qwen config classes default partial_rotary_factor to 0.25
    // (qwen3_next.py:240, qwen3_5_moe.py:92).
    TempJson f(R"({
      "model_type": "qwen3_5_moe",
      "hidden_size": 2048,
      "num_hidden_layers": 4,
      "num_attention_heads": 16,
      "head_dim": 256
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.rotary_dim == 64);  // head_dim / 4
  }
  SUBCASE("non-qwen model without partial_rotary_factor uses full rotary") {
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 4096,
      "num_hidden_layers": 2,
      "num_attention_heads": 32
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 128);
    CHECK(cfg.rotary_dim == 128);
  }
  SUBCASE("explicit head_dim of 0 falls back to derived value") {
    // Upstream honors explicit head_dim only when > 0
    // (model_arch_config_convertor.py:61-75).
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 4096,
      "num_hidden_layers": 2,
      "num_attention_heads": 32,
      "head_dim": 0
    })");
    vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.head_dim == 128);   // hidden_size / num_attention_heads
    CHECK(cfg.rotary_dim == 128);
  }
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
