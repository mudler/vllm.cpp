// vllm.cpp original. BACKEND-ROCM-BF16-MOE production fixture (#3094).
// Adapted from test_moe_async_device_ids.cpp at 6db4bef906859e864c82523c01107473f7dcca29.
// Tensor order, RNG wraparound, conversion, and scales are preserved.
#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "vt/dtype.h"
namespace rocm_moe_fixture {
struct Fx {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i)
    s[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}
int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}
std::string Bf16Bytes(size_t n, int seed, float scale) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
    const uint16_t bf = vt::F32ToBF16((u - 0.5f) * 2.0f * scale);
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}
Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.08f) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
}
std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype},
                   {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

constexpr int64_t kH = 128, kL = 2, kHq = 1, kHkv = 1, kDh = 128, kV = 128;
constexpr int64_t kE = 4, kTopK = 2, kI = 128;

std::string ConfigJson() {
  nlohmann::json j;
  j["architectures"] = std::vector<std::string>{"Qwen3MoeForCausalLM"};
  j["model_type"] = "qwen3_moe";
  j["hidden_size"] = kH;
  j["num_hidden_layers"] = kL;
  j["num_attention_heads"] = kHq;
  j["num_key_value_heads"] = kHkv;
  j["head_dim"] = kDh;
  j["intermediate_size"] = kI;
  j["moe_intermediate_size"] = kI;
  j["shared_expert_intermediate_size"] = 0;
  j["num_experts"] = kE;
  j["num_experts_per_tok"] = kTopK;
  j["vocab_size"] = kV;
  j["max_position_embeddings"] = 256;
  j["rms_norm_eps"] = 1e-6;
  j["rope_theta"] = 10000000.0;
  j["tie_word_embeddings"] = false;
  j["attention_bias"] = false;
  j["torch_dtype"] = "bfloat16";
  j["norm_topk_prob"] = true;
  j["decoder_sparse_step"] = 1;
  j["mlp_only_layers"] = nlohmann::json::array();
  j["hidden_act"] = "silu";
  j["bos_token_id"] = 1;
  j["eos_token_id"] = 127;
  j["pad_token_id"] = 0;
  return j.dump(2);
}

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 7;
  v.push_back(Bf16("model.embed_tokens.weight", {kV, kH}, s++));
  v.push_back(Bf16("model.norm.weight", {kH}, s++, 0.5f));
  v.push_back(Bf16("lm_head.weight", {kV, kH}, s++));
  for (int64_t l = 0; l < kL; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    const std::string sa = b + "self_attn.";
    const std::string mlp = b + "mlp.";
    v.push_back(Bf16(b + "input_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(b + "post_attention_layernorm.weight", {kH}, s++, 0.5f));
    v.push_back(Bf16(sa + "q_proj.weight", {kHq * kDh, kH}, s++));
    v.push_back(Bf16(sa + "k_proj.weight", {kHkv * kDh, kH}, s++));
    v.push_back(Bf16(sa + "v_proj.weight", {kHkv * kDh, kH}, s++));
    v.push_back(Bf16(sa + "o_proj.weight", {kH, kHq * kDh}, s++));
    v.push_back(Bf16(sa + "q_norm.weight", {kDh}, s++, 0.5f));
    v.push_back(Bf16(sa + "k_norm.weight", {kDh}, s++, 0.5f));
    v.push_back(Bf16(mlp + "gate.weight", {kE, kH}, s++));
    for (int64_t e = 0; e < kE; ++e) {
      const std::string ex = mlp + "experts." + std::to_string(e) + ".";
      v.push_back(Bf16(ex + "gate_proj.weight", {kI, kH}, s++));
      v.push_back(Bf16(ex + "up_proj.weight", {kI, kH}, s++));
      v.push_back(Bf16(ex + "down_proj.weight", {kH, kI}, s++));
    }
  }
  return v;
}

inline void Export(const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  const std::string weights = BuildSt(BuildTensors());
  std::ofstream(directory / "model.safetensors", std::ios::binary).write(
      weights.data(), static_cast<std::streamsize>(weights.size()));
  std::ofstream(directory / "config.json") << ConfigJson() << '\n';
}
inline std::vector<int32_t> Prompt(int length, int request) {
  std::vector<int32_t> result;
  for (int i = 0; i < length; ++i) result.push_back(1 + ((11 + 29 * request + 17 * i) % 126));
  return result;
}
}  // namespace rocm_moe_fixture
