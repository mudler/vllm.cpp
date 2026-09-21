// ModernBERT-large encoder — the backbone of Laya (MODEL-LAYA).
//
// Ported from vllm/model_executor/models/modernbert.py @ pin e126687a9a:
//   ModernBertEmbeddings          modernbert.py:35-61
//   ModernBertAttention           modernbert.py:64-145
//   ModernBertMLP                 modernbert.py:148-165
//   ModernBertLayer               modernbert.py:168-206
//   ModernBertModel               modernbert.py:236-300
//
// This is the HOST REFERENCE forward, the same shape the DeBERTa v2 and
// GPT-2 lanes started from: a portable f32 implementation gated against the
// vLLM eager forward before any device path exists. It is not wired to the
// runner, the ABI or the server; that is Phase 4 of .agents/specs/laya.md.
//
// THE THINGS THIS ARCHITECTURE GETS WRONG QUIETLY are: (1) swapping the
// local and global RoPE theta — the model still runs and emits plausible
// hidden states; (2) omitting the sliding window — the local layers silently
// attend to all positions; (3) applying LayerNorm before attention on layer 0
// where the upstream uses Identity. Each is gated by a perturbation test.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {
namespace modernbert {

// The subset of ModernBertConfig the encoder reads.
struct Params {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t intermediate_size = 0;
  int64_t head_dim = 0;            // 0 → hidden_size / num_attention_heads
  int64_t local_attention = 0;     // sliding window size (128 for large)
  int64_t global_attn_every_n_layers = 0;  // 3: layers 0,3,6,... are global
  double local_rope_theta = 10000.0;
  double global_rope_theta = 160000.0;
  double layer_norm_eps = 1e-5;
  bool norm_bias = false;          // false: LayerNorm without bias
  bool mlp_bias = false;
  bool attention_bias = false;

  int64_t resolved_head_dim() const {
    return head_dim > 0 ? head_dim : hidden_size / num_attention_heads;
  }
  // layer_id is global if it divides evenly into global_attn_every_n_layers.
  bool is_global_layer(int64_t layer_id) const {
    return global_attn_every_n_layers > 0 &&
           layer_id % global_attn_every_n_layers == 0;
  }
  // Sliding window for local layers = local_attention / 2.
  int64_t sliding_window() const {
    return local_attention > 0 ? local_attention / 2 : 0;
  }
};

// Raw checkpoint tensors, in the orientation the file carries them.
struct CheckpointTensors {
  std::map<std::string, std::vector<float>> values;
  std::map<std::string, std::vector<int64_t>> shapes;

  void Set(const std::string& name, std::vector<int64_t> shape, std::vector<float> data);
  const std::vector<float>& Get(const std::string& name) const;
  const std::vector<int64_t>& Shape(const std::string& name) const;
  bool Has(const std::string& name) const { return values.count(name) != 0; }
};

// One ModernBERT layer's weights. nn.Linear stores weight as [out, in].
struct LayerWeights {
  // attention: combined Wqkv [3*H, H], output Wo [H, H] (no bias)
  std::vector<float> qkv_weight;       // [3*H, H]
  std::vector<float> attn_out_weight;  // [H, H]
  // attn_norm: LayerNorm weight [H] (absent for layer 0 → Identity)
  std::vector<float> attn_norm_weight;  // [H], empty when layer 0
  // MLP: gated GeGLU. Wi [2*I, H] (gate-up combined), Wo [H, I] (no bias)
  std::vector<float> mlp_gate_up_weight;  // [2*I, H]
  std::vector<float> mlp_down_weight;     // [H, I]
  // mlp_norm: LayerNorm weight [H]
  std::vector<float> mlp_norm_weight;  // [H]
};

struct Weights {
  std::vector<float> tok_embeddings;  // [vocab, H]
  std::vector<float> emb_norm_weight;  // [H]
  std::vector<float> final_norm_weight;  // [H]
  std::vector<LayerWeights> layers;
};

// Materialize `Weights` from a checkpoint. Throws BY NAME on a missing tensor.
Weights Load(const Params& params, const CheckpointTensors& tensors);

// Full encoder forward: embeddings -> N layers -> hidden states [seq, hidden].
// ModernBERT is bidirectional (no causal mask). Positional information comes
// from RoPE (different theta per local/global layer). Local layers use a
// sliding window; global layers attend to all positions.
std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                                const std::vector<int64_t>& input_ids);

}  // namespace modernbert
}  // namespace vllm
