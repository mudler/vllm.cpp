// Ported from vllm/model_executor/models/qwen3_5_mtp.py @
// e24d1b24fe96a56ba8b0d653efa076d03eb95d6c.
// Weight mapping also mirrors Qwen3_5Model.hf_to_vllm_mapper in
// vllm/model_executor/models/qwen3_5.py and qwen3_next.py at that pin.
#include "vllm/model_executor/models/qwen3_5_mtp.h"

#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

OwnedTensor MakeOwned(vt::DType dtype, const std::vector<int64_t>& shape) {
  OwnedTensor out;
  out.dtype = dtype;
  out.rank = static_cast<int>(shape.size());
  VT_CHECK(out.rank <= vt::kMaxRank, "qwen3_5 MTP: rank exceeds kMaxRank");
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    VT_CHECK(shape[static_cast<size_t>(i)] >= 0,
             "qwen3_5 MTP: negative tensor dimension");
    out.shape[i] = shape[static_cast<size_t>(i)];
    numel *= out.shape[i];
  }
  out.bytes.resize(static_cast<size_t>(numel) * vt::SizeOf(dtype));
  return out;
}

OwnedTensor LoadBf16Direct(const TensorResolver& get,
                           const std::string& name) {
  const StTensor& tensor = get(name);
  VT_CHECK(tensor.dtype == "BF16",
           "qwen3_5 MTP: expected BF16 for " + name);
  OwnedTensor out = MakeOwned(vt::DType::kBF16, tensor.shape);
  VT_CHECK(tensor.nbytes == out.bytes.size(),
           "qwen3_5 MTP: byte-size mismatch for " + name);
  std::memcpy(out.bytes.data(), tensor.data, tensor.nbytes);
  return out;
}

// Keep a torch Linear [N=out,K=in] raw and select vt::MatmulBT. This mirrors
// F.linear's layout and avoids a host transpose for the all-BF16 MTP head.
OwnedTensor LoadBf16RawNK(const TensorResolver& get,
                          const std::string& name) {
  OwnedTensor out = LoadBf16Direct(get, name);
  VT_CHECK(out.rank == 2,
           "qwen3_5 MTP: expected a 2-D Linear weight for " + name);
  out.nk = true;
  return out;
}

// Copy one contiguous [N,K] BF16 matrix out of a stacked tensor. `offset`
// counts BF16 elements, not bytes.
OwnedTensor CopyRawNK(const StTensor& source, int64_t offset, int64_t n,
                      int64_t k, const std::string& name) {
  VT_CHECK(source.dtype == "BF16",
           "qwen3_5 MTP: expected BF16 for " + name);
  VT_CHECK(offset >= 0 && n >= 0 && k >= 0,
           "qwen3_5 MTP: invalid stacked slice for " + name);
  const int64_t source_numel = static_cast<int64_t>(source.nbytes / sizeof(uint16_t));
  VT_CHECK(offset <= source_numel && n * k <= source_numel - offset,
           "qwen3_5 MTP: stacked slice out of bounds for " + name);
  OwnedTensor out = MakeOwned(vt::DType::kBF16, {n, k});
  out.nk = true;
  const auto* begin = reinterpret_cast<const uint16_t*>(source.data) + offset;
  std::memcpy(out.bytes.data(), begin, out.bytes.size());
  return out;
}

FullAttnLayerWeights LoadFullAttention(const TensorResolver& get,
                                       const std::string& base) {
  const std::string attn = base + "self_attn.";
  FullAttnLayerWeights out;
  out.q_proj = LoadBf16RawNK(get, attn + "q_proj.weight");
  out.k_proj = LoadBf16RawNK(get, attn + "k_proj.weight");
  out.v_proj = LoadBf16RawNK(get, attn + "v_proj.weight");
  out.o_proj = LoadBf16RawNK(get, attn + "o_proj.weight");
  out.q_norm = LoadBf16Direct(get, attn + "q_norm.weight");
  out.k_norm = LoadBf16Direct(get, attn + "k_norm.weight");
  return out;
}

DenseMlpWeights LoadDenseMlp(const TensorResolver& get,
                             const std::string& base) {
  const std::string mlp = base + "mlp.";
  DenseMlpWeights out;
  out.gate_proj = LoadBf16RawNK(get, mlp + "gate_proj.weight");
  out.up_proj = LoadBf16RawNK(get, mlp + "up_proj.weight");
  out.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
  return out;
}

MoeBlockWeights LoadMoe(const TensorResolver& get, const std::string& base,
                        const HfConfig& config) {
  const std::string mlp = base + "mlp.";
  const int64_t experts = config.num_experts;
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.moe_intermediate_size;
  VT_CHECK(experts > 0 && hidden > 0 && intermediate > 0,
           "qwen3_5 MTP: invalid MoE dimensions");

  MoeBlockWeights out;
  out.router_gate = LoadBf16RawNK(get, mlp + "gate.weight");
  out.shared_gate =
      LoadBf16RawNK(get, mlp + "shared_expert_gate.weight");

  // AutoWeightsLoader maps the checkpoint's fused expert stacks into the
  // per-expert gate/up/down parameters. The disk layouts are
  // gate_up[E,2I,H] and down[E,H,I].
  const StTensor& gate_up = get(mlp + "experts.gate_up_proj");
  const StTensor& down = get(mlp + "experts.down_proj");
  VT_CHECK(gate_up.shape == std::vector<int64_t>({experts, 2 * intermediate, hidden}),
           "qwen3_5 MTP: unexpected experts.gate_up_proj shape");
  VT_CHECK(down.shape == std::vector<int64_t>({experts, hidden, intermediate}),
           "qwen3_5 MTP: unexpected experts.down_proj shape");
  out.expert_gate.reserve(static_cast<size_t>(experts));
  out.expert_up.reserve(static_cast<size_t>(experts));
  out.expert_down.reserve(static_cast<size_t>(experts));
  const int64_t gate_up_stride = 2 * intermediate * hidden;
  const int64_t down_stride = hidden * intermediate;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const int64_t gu_base = expert * gate_up_stride;
    out.expert_gate.push_back(CopyRawNK(
        gate_up, gu_base, intermediate, hidden, "experts.gate_up_proj"));
    out.expert_up.push_back(CopyRawNK(
        gate_up, gu_base + intermediate * hidden, intermediate, hidden,
        "experts.gate_up_proj"));
    out.expert_down.push_back(CopyRawNK(
        down, expert * down_stride, hidden, intermediate,
        "experts.down_proj"));
  }

  const std::string shared = mlp + "shared_expert.";
  out.shared_gate_proj =
      LoadBf16RawNK(get, shared + "gate_proj.weight");
  out.shared_up_proj =
      LoadBf16RawNK(get, shared + "up_proj.weight");
  out.shared_down_proj =
      LoadBf16RawNK(get, shared + "down_proj.weight");
  return out;
}

const nlohmann::json& TextConfig(const HfConfig& config) {
  if (config.raw.is_object() && config.raw.contains("text_config") &&
      config.raw.at("text_config").is_object()) {
    return config.raw.at("text_config");
  }
  return config.raw;
}

int64_t NumMtpLayers(const HfConfig& config) {
  const nlohmann::json& text = TextConfig(config);
  if (!text.is_object()) return 1;
  return text.value("mtp_num_hidden_layers", int64_t{1});
}

bool UsesDedicatedEmbeddings(const HfConfig& config) {
  const nlohmann::json& text = TextConfig(config);
  return text.is_object() &&
         text.value("mtp_use_dedicated_embeddings", false);
}

void RequireShape(const OwnedTensor& tensor,
                  std::initializer_list<int64_t> expected,
                  const std::string& name) {
  VT_CHECK(tensor.rank == static_cast<int>(expected.size()),
           "qwen3_5 MTP: unexpected rank for " + name);
  size_t dim = 0;
  for (const int64_t want : expected) {
    VT_CHECK(tensor.shape[dim] == want,
             "qwen3_5 MTP: unexpected shape for " + name);
    ++dim;
  }
}

void ValidateAttention(const FullAttnLayerWeights& attention,
                       const HfConfig& config, const std::string& base) {
  const int64_t hidden = config.hidden_size;
  const int64_t head_dim = config.head_dim;
  const int64_t query = config.num_attention_heads * head_dim;
  const int64_t key_value = config.num_key_value_heads * head_dim;
  VT_CHECK(config.num_attention_heads > 0 &&
               config.num_key_value_heads > 0 && head_dim > 0,
           "qwen3_5 MTP: invalid full-attention dimensions");
  // Qwen3.5 full attention is output-gated: q_proj packs q|gate.
  RequireShape(attention.q_proj, {2 * query, hidden}, base + "q_proj.weight");
  RequireShape(attention.k_proj, {key_value, hidden}, base + "k_proj.weight");
  RequireShape(attention.v_proj, {key_value, hidden}, base + "v_proj.weight");
  RequireShape(attention.o_proj, {hidden, query}, base + "o_proj.weight");
  RequireShape(attention.q_norm, {head_dim}, base + "q_norm.weight");
  RequireShape(attention.k_norm, {head_dim}, base + "k_norm.weight");
}

void ValidateDenseLayer(const Qwen3_5DenseLayerWeights& layer,
                        const HfConfig& config, const std::string& base) {
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.intermediate_size;
  VT_CHECK(intermediate > 0,
           "qwen3_5 MTP: dense intermediate_size must be > 0");
  VT_CHECK(!layer.is_linear_attention,
           "qwen3_5 MTP: draft layer must use full attention");
  RequireShape(layer.input_layernorm, {hidden},
               base + "input_layernorm.weight");
  RequireShape(layer.post_attention_layernorm, {hidden},
               base + "post_attention_layernorm.weight");
  ValidateAttention(layer.attn, config, base + "self_attn.");
  RequireShape(layer.mlp.gate_proj, {intermediate, hidden},
               base + "mlp.gate_proj.weight");
  RequireShape(layer.mlp.up_proj, {intermediate, hidden},
               base + "mlp.up_proj.weight");
  RequireShape(layer.mlp.down_proj, {hidden, intermediate},
               base + "mlp.down_proj.weight");
}

void ValidateMoeLayer(const Qwen3_5MoeLayerWeights& layer,
                      const HfConfig& config, const std::string& base) {
  const int64_t hidden = config.hidden_size;
  const int64_t experts = config.num_experts;
  const int64_t intermediate = config.moe_intermediate_size;
  const int64_t shared = config.shared_expert_intermediate_size;
  VT_CHECK(experts > 0 && config.num_experts_per_tok > 0 &&
               config.num_experts_per_tok <= experts &&
               intermediate > 0 && shared > 0,
           "qwen3_5 MTP: invalid MoE dimensions");
  VT_CHECK(!layer.is_linear_attention,
           "qwen3_5 MTP: draft layer must use full attention");
  RequireShape(layer.input_layernorm, {hidden},
               base + "input_layernorm.weight");
  RequireShape(layer.post_attention_layernorm, {hidden},
               base + "post_attention_layernorm.weight");
  ValidateAttention(layer.attn, config, base + "self_attn.");
  RequireShape(layer.moe.router_gate, {experts, hidden},
               base + "mlp.gate.weight");
  RequireShape(layer.moe.shared_gate, {1, hidden},
               base + "mlp.shared_expert_gate.weight");
  VT_CHECK(layer.moe.expert_gate.size() == static_cast<size_t>(experts) &&
               layer.moe.expert_up.size() == static_cast<size_t>(experts) &&
               layer.moe.expert_down.size() == static_cast<size_t>(experts),
           "qwen3_5 MTP: unexpected routed-expert count");
  for (int64_t expert = 0; expert < experts; ++expert) {
    const std::string expert_name =
        base + "mlp.experts." + std::to_string(expert) + ".";
    RequireShape(layer.moe.expert_gate[static_cast<size_t>(expert)],
                 {intermediate, hidden}, expert_name + "gate_proj");
    RequireShape(layer.moe.expert_up[static_cast<size_t>(expert)],
                 {intermediate, hidden}, expert_name + "up_proj");
    RequireShape(layer.moe.expert_down[static_cast<size_t>(expert)],
                 {hidden, intermediate}, expert_name + "down_proj");
  }
  RequireShape(layer.moe.shared_gate_proj, {shared, hidden},
               base + "mlp.shared_expert.gate_proj.weight");
  RequireShape(layer.moe.shared_up_proj, {shared, hidden},
               base + "mlp.shared_expert.up_proj.weight");
  RequireShape(layer.moe.shared_down_proj, {hidden, shared},
               base + "mlp.shared_expert.down_proj.weight");
}

}  // namespace

int64_t Qwen3_5MTPWeights::NumLayers() const {
  return kind == Qwen3_5MTPKind::kDense
             ? static_cast<int64_t>(dense_layers.size())
             : static_cast<int64_t>(moe_layers.size());
}

Qwen3_5MTPWeights LoadQwen3_5MTP(const TensorResolver& get,
                                 const HfConfig& config,
                                 Qwen3_5MTPKind kind) {
  VT_CHECK(config.hidden_size > 0, "qwen3_5 MTP: hidden_size must be > 0");
  VT_CHECK(!UsesDedicatedEmbeddings(config),
           "qwen3_5 MTP: dedicated embeddings are not supported by M-mtp-0");
  const int64_t num_layers = NumMtpLayers(config);
  VT_CHECK(num_layers > 0, "qwen3_5 MTP: mtp_num_hidden_layers must be > 0");

  Qwen3_5MTPWeights out;
  out.kind = kind;
  out.fc = LoadBf16RawNK(get, "mtp.fc.weight");
  out.pre_fc_norm_embedding =
      LoadBf16Direct(get, "mtp.pre_fc_norm_embedding.weight");
  out.pre_fc_norm_hidden =
      LoadBf16Direct(get, "mtp.pre_fc_norm_hidden.weight");
  out.final_norm = LoadBf16Direct(get, "mtp.norm.weight");

  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    const std::string base =
        "mtp.layers." + std::to_string(layer_index) + ".";
    if (kind == Qwen3_5MTPKind::kDense) {
      Qwen3_5DenseLayerWeights layer;
      layer.is_linear_attention = false;
      layer.input_layernorm =
          LoadBf16Direct(get, base + "input_layernorm.weight");
      layer.post_attention_layernorm =
          LoadBf16Direct(get, base + "post_attention_layernorm.weight");
      layer.attn = LoadFullAttention(get, base);
      layer.mlp = LoadDenseMlp(get, base);
      out.dense_layers.push_back(std::move(layer));
    } else {
      Qwen3_5MoeLayerWeights layer;
      layer.is_linear_attention = false;
      layer.input_layernorm =
          LoadBf16Direct(get, base + "input_layernorm.weight");
      layer.post_attention_layernorm =
          LoadBf16Direct(get, base + "post_attention_layernorm.weight");
      layer.attn = LoadFullAttention(get, base);
      layer.moe = LoadMoe(get, base, config);
      out.moe_layers.push_back(std::move(layer));
    }
  }

  const int64_t hidden = config.hidden_size;
  RequireShape(out.fc, {hidden, 2 * hidden}, "mtp.fc.weight");
  RequireShape(out.pre_fc_norm_embedding, {hidden},
               "mtp.pre_fc_norm_embedding.weight");
  RequireShape(out.pre_fc_norm_hidden, {hidden},
               "mtp.pre_fc_norm_hidden.weight");
  RequireShape(out.final_norm, {hidden}, "mtp.norm.weight");
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    const std::string base =
        "mtp.layers." + std::to_string(layer_index) + ".";
    if (kind == Qwen3_5MTPKind::kDense) {
      ValidateDenseLayer(out.dense_layers[static_cast<size_t>(layer_index)],
                         config, base);
    } else {
      ValidateMoeLayer(out.moe_layers[static_cast<size_t>(layer_index)],
                       config, base);
    }
  }
  return out;
}

Qwen3_5MTPWeights LoadQwen3_5MTP(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    Qwen3_5MTPKind kind) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) where[name] = &shard;
  }
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    const auto it = where.find(name);
    VT_CHECK(it != where.end(),
             "qwen3_5 MTP: tensor not found: " + name);
    return it->second->Get(name);
  };
  return LoadQwen3_5MTP(get, config, kind);
}

}  // namespace vllm
