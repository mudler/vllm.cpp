// DeBERTa v2 encoder — the backbone of GLiNER2.5-multi-v1 (MODEL-GLINER25).
//
// Ported from transformers/models/deberta_v2/modeling_deberta_v2.py
// @ v4.44.2 (the secondary oracle):
//   DebertaV2Embeddings.forward          modeling_deberta_v2.py:DebertaV2Embeddings
//   DisentangledSelfAttention.forward    modeling_deberta_v2.py:DisentangledSelfAttention
//   DisentangledSelfAttention.disentangled_attention_bias  (same file)
//   DebertaV2SelfOutput                  (dense -> LayerNorm + residual)
//   DebertaV2Intermediate                (dense -> gelu)
//   DebertaV2Output                      (dense -> LayerNorm + residual)
//   DebertaV2Encoder.get_rel_embedding   (rel_embeddings -> LayerNorm)
//   make_log_bucket_position             (log-scale relative position buckets)
//   build_relative_position              (relative position matrix)
//
// This is the HOST REFERENCE forward, the same shape the GPT-2 and w2vbert
// lanes started from: a portable f32 implementation gated against the
// HuggingFace reference before any device path exists. It is not wired to the
// runner, the ABI or the server; that is Phase 3 of .agents/specs/gliner2.5.md.
//
// THE THING THIS ARCHITECTURE GETS WRONG QUIETLY is dropping the disentangled
// attention bias. Without c2p and p2c the model still runs and emits plausible
// hidden states, so the bias is gated by a perturbation test of its own rather
// than left to be implied by the forward.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {
namespace deberta_v2 {

// The subset of DebertaV2Config the encoder reads.
struct Params {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t intermediate_size = 0;
  int64_t max_position_embeddings = 0;
  int64_t position_buckets = 0;
  double layer_norm_eps = 1e-7;
  bool position_biased_input = false;  // false for mdeberta-v3-base
  int64_t type_vocab_size = 0;          // 0 for mdeberta-v3-base
  bool norm_rel_ebd = true;             // "layer_norm"
  bool share_att_key = true;
  bool use_c2p = true;
  bool use_p2c = true;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
  int64_t max_rel_pos() const { return max_position_embeddings; }
  // Encoder: rel_embeddings table size = position_buckets * 2
  int64_t pos_ebd_size() const {
    return position_buckets > 0 ? position_buckets * 2 : max_rel_pos() * 2;
  }
  // Attention: att_span = position_buckets (or max_rel_pos)
  int64_t att_span() const {
    return position_buckets > 0 ? position_buckets : max_rel_pos();
  }
  int64_t scale_factor() const {
    int64_t sf = 1;
    if (use_c2p) sf += 1;
    if (use_p2c) sf += 1;
    return sf;
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

// One transformer layer's weights. nn.Linear stores weight as [out, in].
struct LayerWeights {
  // attention.self
  std::vector<float> query_weight, query_bias;   // [H, H], [H]
  std::vector<float> key_weight, key_bias;        // [H, H], [H]
  std::vector<float> value_weight, value_bias;   // [H, H], [H]
  // attention.output (DebertaV2SelfOutput)
  std::vector<float> attn_dense_weight, attn_dense_bias;       // [H, H], [H]
  std::vector<float> attn_ln_weight, attn_ln_bias;              // [H], [H]
  // intermediate
  std::vector<float> inter_weight, inter_bias;    // [I, H], [I]
  // output (DebertaV2Output)
  std::vector<float> out_dense_weight, out_dense_bias;  // [H, I], [H]
  std::vector<float> out_ln_weight, out_ln_bias;        // [H], [H]
};

struct Weights {
  std::vector<float> word_embeddings;  // [vocab, H]
  std::vector<float> emb_ln_weight, emb_ln_bias;  // [H], [H]
  std::vector<float> rel_embeddings;  // [pos_ebd_size, H]
  std::vector<float> rel_ln_weight, rel_ln_bias;  // [H], [H]
  std::vector<LayerWeights> layers;
};

// Materialize `Weights` from a checkpoint. Throws BY NAME on a missing tensor.
Weights Load(const Params& params, const CheckpointTensors& tensors);

// Full encoder forward: embeddings -> N layers -> hidden states [seq, hidden].
// DeBERTa is bidirectional (no causal mask). Positional information comes
// entirely from the disentangled attention relative position bias.
std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                               const std::vector<int64_t>& input_ids);

}  // namespace deberta_v2
}  // namespace vllm
