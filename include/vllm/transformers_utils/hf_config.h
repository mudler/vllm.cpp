// vllm.cpp original (container reader); no upstream mirror.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {

// Typed view over a HuggingFace config.json. Key names and defaults follow
// the pinned upstream config classes (vllm/transformers_utils/configs/
// qwen3_next.py and friends). Missing optional fields default to 0/empty;
// missing required fields (model_type, hidden_size, num_hidden_layers) throw.
struct HfConfig {
  std::string model_type;
  std::vector<std::string> architectures;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;  // absent -> num_attention_heads
  int64_t head_dim = 0;             // absent -> hidden_size / num_attention_heads
  // Hybrid models: "linear_attention"/"full_attention" per layer; empty = all
  // full attention.
  std::vector<std::string> layer_types;
  int64_t intermediate_size = 0;
  // MoE (0 when absent):
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t shared_expert_intermediate_size = 0;
  // Gated DeltaNet (0 when absent):
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;
  // RoPE:
  double rope_theta = 10000.0;
  int64_t rotary_dim = 0;  // partial_rotary_factor * head_dim, rounded
  double rms_norm_eps = 0.0;
  int64_t max_position_embeddings = 0;
  std::string torch_dtype;
  nlohmann::json raw;  // full doc for fields we don't type yet
};

// Loads and parses `path`. Throws std::runtime_error (message includes the
// path) on missing file, malformed JSON, or missing required fields.
HfConfig LoadHfConfig(const std::string& path);

}  // namespace vllm
