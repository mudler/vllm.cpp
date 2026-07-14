// Sliding-window cases port ModelConfig normalization from
// vllm/config/model.py:542-559,654-660,723-726,1232-1234. RoPE cases port
// vllm/transformers_utils/config.py:439-509 and rotary_embedding/__init__.py:
// 33-112,243-283 @ e24d1b24fe96.
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
  "mamba_ssm_dtype": "float32",
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
    "mamba_ssm_dtype": "float32",
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
  CHECK(cfg.mamba_ssm_dtype == "float32");
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
  CHECK(cfg.mamba_ssm_dtype == "float32");

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

TEST_CASE("LoadHfConfig mirrors sliding_window normalization from the text config") {
  SUBCASE("positive top-level window is retained") {
    TempJson f(R"({
      "model_type": "starcoder2",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "sliding_window": 4096
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.sliding_window.has_value());
    CHECK(*cfg.sliding_window == 4096);
  }

  SUBCASE("zero and null mean disabled") {
    for (const char* value : {"0", "null"}) {
      TempJson f(std::string(R"({
        "model_type": "local",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "sliding_window": )") + value + "}");
      CHECK_FALSE(vllm::LoadHfConfig(f.path()).sliding_window.has_value());
    }
  }

  SUBCASE("nested text_config is the source") {
    TempJson f(R"({
      "model_type": "wrapper",
      "architectures": ["WrapperForCausalLM"],
      "sliding_window": 999,
      "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "sliding_window": 128
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.sliding_window.has_value());
    CHECK(*cfg.sliding_window == 128);
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

TEST_CASE("LoadHfConfig retains the complete typed YaRN parameter slice") {
  TempJson f(R"({
    "model_type": "nomic_bert",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 8192,
    "rope_parameters": {
      "rope_type": "yarn",
      "rope_theta": 1000.0,
      "rope_dim": 12,
      "factor": 4.0,
      "original_max_position_embeddings": 2048,
      "extrapolation_factor": 1.25,
      "attn_factor": 0.75,
      "beta_fast": 24,
      "beta_slow": 2,
      "apply_yarn_scaling": false,
      "truncate": false,
      "mrope_section": [2, 2, 2],
      "mrope_interleaved": true
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_theta == doctest::Approx(1000.0));
  CHECK(cfg.rotary_dim == 12);
  CHECK(cfg.rope_parameters.rope_type == "yarn");
  REQUIRE(cfg.rope_parameters.factor.has_value());
  CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 2048);
  CHECK(cfg.rope_parameters.extrapolation_factor == doctest::Approx(1.25));
  CHECK(cfg.rope_parameters.attn_factor == doctest::Approx(0.75));
  CHECK(cfg.rope_parameters.beta_fast == 24);
  CHECK(cfg.rope_parameters.beta_slow == 2);
  CHECK_FALSE(cfg.rope_parameters.apply_yarn_scaling);
  CHECK_FALSE(cfg.rope_parameters.truncate);
  CHECK(cfg.rope_parameters.mrope_section ==
        std::vector<int64_t>{2, 2, 2});
  CHECK(cfg.rope_parameters.mrope_interleaved);
}

TEST_CASE("LoadHfConfig retains the complete typed Llama 3 parameter slice") {
  TempJson f(R"({
    "model_type": "llama",
    "hidden_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "rope_parameters": {
      "rope_type": "llama3",
      "rope_theta": 10000.0,
      "rope_dim": 16,
      "factor": 4.0,
      "low_freq_factor": 1.0,
      "high_freq_factor": 4.0,
      "original_max_position_embeddings": 32
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_parameters.rope_type == "llama3");
  CHECK(cfg.rotary_dim == 16);
  REQUIRE(cfg.rope_parameters.factor.has_value());
  REQUIRE(cfg.rope_parameters.low_freq_factor.has_value());
  REQUIRE(cfg.rope_parameters.high_freq_factor.has_value());
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
  CHECK(*cfg.rope_parameters.low_freq_factor == doctest::Approx(1.0));
  CHECK(*cfg.rope_parameters.high_freq_factor == doctest::Approx(4.0));
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 32);
}

TEST_CASE("LoadHfConfig retains and normalizes Phi-3 LongRoPE parameters") {
  TempJson f(R"({
    "model_type": "phi3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 128,
    "rope_scaling": {
      "type": "su",
      "rope_theta": 10000.0,
      "rope_dim": 8,
      "original_max_position_embeddings": 32,
      "short_factor": [1.0, 1.1, 1.2, 1.3],
      "long_factor": [2.0, 2.2, 2.4, 2.6],
      "short_mscale": 0.75,
      "long_mscale": 1.25
    }
  })");
  const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
  REQUIRE(cfg.has_rope_parameters);
  CHECK(cfg.rope_parameters.rope_type == "longrope");
  CHECK(cfg.rotary_dim == 8);
  CHECK(cfg.rope_parameters.short_factor ==
        std::vector<double>{1.0, 1.1, 1.2, 1.3});
  CHECK(cfg.rope_parameters.long_factor ==
        std::vector<double>{2.0, 2.2, 2.4, 2.6});
  REQUIRE(cfg.rope_parameters.short_mscale.has_value());
  REQUIRE(cfg.rope_parameters.long_mscale.has_value());
  CHECK(*cfg.rope_parameters.short_mscale == doctest::Approx(0.75));
  CHECK(*cfg.rope_parameters.long_mscale == doctest::Approx(1.25));
  REQUIRE(cfg.rope_parameters.original_max_position_embeddings.has_value());
  CHECK(*cfg.rope_parameters.original_max_position_embeddings == 32);
}

TEST_CASE("LoadHfConfig retains both dynamic NTK dispatch forms") {
  SUBCASE("factor mode retains trained context") {
    TempJson f(R"({
      "model_type": "llama",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "max_position_embeddings": 128,
      "rope_parameters": {
        "rope_type": "dynamic",
        "rope_theta": 10000.0,
        "rope_dim": 8,
        "factor": 4.0,
        "max_trained_positions": 32
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "dynamic");
    REQUIRE(cfg.rope_parameters.factor.has_value());
    REQUIRE(cfg.rope_parameters.max_trained_positions.has_value());
    CHECK(*cfg.rope_parameters.factor == doctest::Approx(4.0));
    CHECK(*cfg.rope_parameters.max_trained_positions == 32);
  }

  SUBCASE("alpha mode is typed independently") {
    TempJson f(R"({
      "model_type": "hunyuan",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "max_position_embeddings": 128,
      "rope_parameters": {
        "rope_type": "dynamic",
        "rope_dim": 8,
        "alpha": 2.0
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    REQUIRE(cfg.rope_parameters.alpha.has_value());
    CHECK(*cfg.rope_parameters.alpha == doctest::Approx(2.0));
    CHECK_FALSE(cfg.rope_parameters.factor.has_value());
  }
}

TEST_CASE("LoadHfConfig normalizes legacy and default MRoPE dictionaries") {
  SUBCASE("legacy rope_scaling type yarn is standardized") {
    TempJson f(R"({
      "model_type": "legacy",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "rope_theta": 5000.0,
      "rope_scaling": {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "yarn");
    CHECK(cfg.rope_parameters.rope_theta == doctest::Approx(5000.0));
    CHECK(*cfg.rope_parameters.factor == doctest::Approx(2.0));
  }

  SUBCASE("legacy mrope type becomes default and retains sections") {
    TempJson f(R"({
      "model_type": "qwen_vl",
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "rope_parameters": {
        "type": "mrope",
        "rope_theta": 1000000.0,
        "mrope_section": [2, 3, 3],
        "mrope_interleaved": true
      }
    })");
    const vllm::HfConfig cfg = vllm::LoadHfConfig(f.path());
    CHECK(cfg.rope_parameters.rope_type == "default");
    CHECK(cfg.rope_parameters.mrope_section ==
          std::vector<int64_t>{2, 3, 3});
    CHECK(cfg.rope_parameters.mrope_interleaved);
  }
}

TEST_CASE("LoadHfConfig keeps unsupported or malformed RoPE loud") {
  const std::string prefix = R"({
    "model_type": "rope_test",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "rope_parameters": )";
  SUBCASE("future formula family remains rejected until its leaf lands") {
    TempJson f(prefix + R"({"rope_type":"linear"}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("does not implement yet"),
                         std::runtime_error);
  }
  SUBCASE("YaRN requires both dispatch fields") {
    TempJson f(prefix + R"({"rope_type":"yarn","factor":4.0}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("requires factor"),
                         std::runtime_error);
  }
  SUBCASE("Llama 3 requires every frequency-band field") {
    TempJson f(prefix +
               R"({"rope_type":"llama3","factor":4.0,"low_freq_factor":1.0}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("llama3 rope requires"),
                         std::runtime_error);
  }
  SUBCASE("LongRoPE requires both factor arrays and original context") {
    TempJson f(prefix +
               R"({"rope_type":"longrope","short_factor":[1.0,1.0]}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("longrope requires"),
                         std::runtime_error);
  }
  SUBCASE("dynamic NTK requires alpha or factor") {
    TempJson f(prefix + R"({"rope_type":"dynamic"}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("requires either alpha or factor"),
                         std::runtime_error);
  }
  SUBCASE("nested per-layer parameters cannot silently become default") {
    TempJson f(prefix +
               R"({"full_attention":{"rope_type":"default"},"sliding_attention":{"rope_type":"yarn"}}})");
    CHECK_THROWS_WITH_AS(vllm::LoadHfConfig(f.path()),
                         doctest::Contains("nested per-layer"),
                         std::runtime_error);
  }
}
