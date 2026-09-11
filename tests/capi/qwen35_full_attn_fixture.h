// ENG-QWEN35-FULL-ATTN-STATE (#3098): one deterministic, all-dense GGUF.
// Extracted from rocm_quant_gather_fixture.h, SHA256
// 339414f93e59e6be9a4ca545d5e762bb2816714343f466e218dbc2b69483bdfd.
// Keep the dense Q4_0 control byte-identical to that experiment. Its name and
// inactive SSM loader keys are part of the frozen artifact, not active layers.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace qwen35_full_attn_test {

inline constexpr uint32_t kSeed = 0x524f434d;
inline constexpr int64_t kHidden = 256;
inline constexpr int64_t kVocab = 128;
inline constexpr size_t kArtifactBytes = 991296;
inline constexpr const char* kArtifactSha256 =
    "0e6554ba521edfde00d4d025a3058eabaaacdce6b24344f959e83d8dde35df7f";

inline std::string DenseEmbedding() {
  // Q4_0 is only the deterministic source of these BF16 values. The resulting
  // GGUF contains a dense embedding and no quantized projection.
  constexpr int64_t kBlockBytes = 18;
  constexpr int64_t kBlockElems = 32;
  constexpr int64_t kBlocks = kVocab * kHidden / kBlockElems;
  std::string packed(static_cast<size_t>(kBlocks * kBlockBytes), '\0');
  std::mt19937 rng(kSeed);
  for (int64_t block = 0; block < kBlocks; ++block) {
    const size_t base = static_cast<size_t>(block * kBlockBytes);
    for (int64_t byte = 0; byte < kBlockBytes; ++byte)
      packed[base + static_cast<size_t>(byte)] = static_cast<char>(rng() & 255);
    packed[base] = 0;
    packed[base + 1] = 0x20;  // Half d = 1/128, exactly as the frozen control.
  }
  std::vector<float> decoded(static_cast<size_t>(kVocab * kHidden));
  const auto decode = vt::cpu::BlockToFloat(vt::DType::kQ4_0);
  if (decode == nullptr) throw std::runtime_error("fixture Q4_0 decoder absent");
  decode(packed.data(), decoded.data(), kVocab * kHidden);
  std::string dense;
  dense.reserve(decoded.size() * 2);
  for (float value : decoded) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture block");
    const uint16_t word = vt::F32ToBF16(value);
    dense.push_back(static_cast<char>(word & 255));
    dense.push_back(static_cast<char>(word >> 8));
  }
  return dense;
}

inline std::string BuildModel() {
  using namespace gguf_test;
  GgufModelBuilder builder;
  builder.AddKv(StrKv("general.architecture", "qwen35"));
  builder.AddKv(StrKv("general.name", "rocm-quant-gather-synthetic"));
  builder.AddKv(U32Kv("general.quantization_version", 2));
  builder.AddKv(U32Kv("qwen35.embedding_length", 256));
  builder.AddKv(U32Kv("qwen35.block_count", 1));
  builder.AddKv(U32Kv("qwen35.attention.head_count", 4));
  builder.AddKv(U32Kv("qwen35.attention.head_count_kv", 1));
  builder.AddKv(U32Kv("qwen35.attention.key_length", 64));
  builder.AddKv(U32Kv("qwen35.attention.value_length", 64));
  builder.AddKv(U32Kv("qwen35.feed_forward_length", 256));
  builder.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  builder.AddKv(U32Kv("qwen35.context_length", 64));
  builder.AddKv(U32Kv("qwen35.vocab_size", 128));
  builder.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  builder.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  builder.AddKv(U32Kv("qwen35.rope.dimension_count", 64));
  builder.AddKv(I32ArrayKv("qwen35.rope.dimension_sections", {16, 8, 8, 0}));
  builder.AddKv(U32Kv("qwen35.ssm.conv_kernel", 4));
  builder.AddKv(U32Kv("qwen35.ssm.inner_size", 64));
  builder.AddKv(U32Kv("qwen35.ssm.state_size", 64));
  builder.AddKv(U32Kv("qwen35.ssm.time_step_rank", 1));
  builder.AddKv(U32Kv("qwen35.ssm.group_count", 1));
  builder.AddKv(StrKv("tokenizer.ggml.model", "gpt2"));
  builder.AddKv(StrKv("tokenizer.ggml.pre", "qwen2"));
  builder.AddKv(U32Kv("tokenizer.ggml.unknown_token_id", 0));
  builder.AddKv(U32Kv("tokenizer.ggml.bos_token_id", 1));
  builder.AddKv(U32Kv("tokenizer.ggml.eos_token_id", 2));
  builder.AddKv(BoolKv("tokenizer.ggml.add_bos_token", false));
  std::vector<std::string> vocabulary = {"<unk>", "<s>", "</s>"};
  for (int id = 3; id < kVocab; ++id) vocabulary.push_back("token" + std::to_string(id));
  builder.AddKv(StrArrayKv("tokenizer.ggml.tokens", vocabulary));
  builder.AddKv(StrArrayKv("tokenizer.ggml.merges", {}));
  std::vector<int32_t> types(128, 1);
  types[0] = 2;
  types[1] = types[2] = 3;
  builder.AddKv(I32ArrayKv("tokenizer.ggml.token_type", types));
  builder.AddTensor("token_embd.weight", {256, 128}, 30, DenseEmbedding());

  std::mt19937 rng(kSeed);
  const auto projection = [&](const char* name, uint64_t input, uint64_t output) {
    std::string bytes;
    bytes.reserve(static_cast<size_t>(input * output * 2));
    for (uint64_t i = 0; i < input * output; ++i) {
      const int32_t integer = static_cast<int32_t>(rng() & 65535) - 32768;
      const float value = static_cast<float>(integer) / 2097152.0F;
      const uint16_t word = vt::F32ToBF16(value);
      bytes.push_back(static_cast<char>(word & 255));
      bytes.push_back(static_cast<char>(word >> 8));
    }
    builder.AddTensor(name, {input, output}, 30, bytes);
  };
  const auto norm = [&](const char* name, uint64_t width) {
    std::string bytes;
    for (uint64_t i = 0; i < width; ++i) bytes += U32Le(0x3f800000);
    builder.AddTensor(name, {width}, 0, bytes);
  };
  projection("output.weight", 256, 128);
  projection("blk.0.attn_q.weight", 256, 512);
  projection("blk.0.attn_k.weight", 256, 64);
  projection("blk.0.attn_v.weight", 256, 64);
  projection("blk.0.attn_output.weight", 256, 256);
  projection("blk.0.ffn_gate.weight", 256, 256);
  projection("blk.0.ffn_up.weight", 256, 256);
  projection("blk.0.ffn_down.weight", 256, 256);
  norm("output_norm.weight", 256);
  norm("blk.0.attn_norm.weight", 256);
  norm("blk.0.post_attention_norm.weight", 256);
  norm("blk.0.attn_q_norm.weight", 64);
  norm("blk.0.attn_k_norm.weight", 64);
  return builder.Build();
}

}  // namespace qwen35_full_attn_test
