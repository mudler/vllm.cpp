// Laya decision head — the System 1 decision model on top of ModernBERT (MODEL-LAYA).
//
// Ported from laya/common.py DecisionModel (NandhaKishorM/laya):
//   type_emb: Embedding(3, d) — choice=0, score=1, noul=2; added to all positions.
//   head: nn.TransformerEncoder, head_layers layers (d, nhead, dim_ff=4*d,
//         norm_first=True, ReLU). Standard pre-norm transformer encoder layers
//         (NOT ModernBERT layers). Uses src_key_padding_mask = ~attention_mask.
//   scorer: Sequential(LayerNorm(d), Linear(d,d), GELU(), Linear(d,1)).
//   act_head: Sequential(Linear(d+4, 256), GELU(), Linear(256, n_act)).
//         Input is [CLS_pooled(d) + 4 confidence_features].
//   temperature: buffer [3] (one per question type). Applied at inference time
//         in the agent, NOT inside the forward.
//
// The encoder (ModernBERT) is Phase 1. This head takes encoder hidden states
// and produces (logits, act_logits). It is not wired to the runner, the ABI or
// the server; that is Phase 4 of .agents/specs/laya.md.
//
// THE THINGS THIS HEAD GETS WRONG QUIETLY are: (1) skipping the type_emb
// addition — the model still runs and emits plausible logits; (2) using
// post-norm instead of pre-norm in the head layers — plausible but wrong;
// (3) using GELU instead of ReLU in the head FFN — plausible but wrong; (4)
// skipping the key_padding_mask in head attention — padded positions silently
// contribute. Each is gated by a perturbation test.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/modernbert.h"

namespace vllm {
namespace laya {

// The subset of DecisionModel config the head reads.
struct Params {
  int64_t hidden_size = 0;       // = encoder hidden_size (1024 for large)
  int64_t num_heads = 0;        // = max(1, hidden_size // 64) (16 for large)
  int64_t head_layers = 0;      // 2 for Laya
  int64_t dim_ff = 0;           // = 4 * hidden_size (4096 for large)
  int64_t n_act = 2;            // escalate decision output dim
  int64_t act_hidden = 256;     // act_head intermediate size
  double layer_norm_eps = 1e-5;

  int64_t resolved_head_dim() const {
    return hidden_size / num_heads;
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

// One standard nn.TransformerEncoderLayer (norm_first=True, ReLU, batch_first).
struct HeadLayerWeights {
  // self_attn: combined in_proj [3*d, d] + bias, out_proj [d, d] + bias
  std::vector<float> in_proj_weight;   // [3*d, d]
  std::vector<float> in_proj_bias;    // [3*d]
  std::vector<float> out_proj_weight;  // [d, d]
  std::vector<float> out_proj_bias;   // [d]
  // FFN: linear1 [dim_ff, d] + bias, linear2 [d, dim_ff] + bias
  std::vector<float> linear1_weight;  // [dim_ff, d]
  std::vector<float> linear1_bias;    // [dim_ff]
  std::vector<float> linear2_weight;  // [d, dim_ff]
  std::vector<float> linear2_bias;     // [d]
  // LayerNorms (with bias, unlike ModernBERT)
  std::vector<float> norm1_weight;    // [d]
  std::vector<float> norm1_bias;      // [d]
  std::vector<float> norm2_weight;    // [d]
  std::vector<float> norm2_bias;      // [d]
};

// scorer: Sequential(LayerNorm(d), Linear(d,d), GELU(), Linear(d,1))
struct ScorerWeights {
  std::vector<float> norm_weight;      // [d]
  std::vector<float> norm_bias;        // [d]
  std::vector<float> linear1_weight;   // [d, d]
  std::vector<float> linear1_bias;     // [d]
  std::vector<float> linear2_weight;   // [1, d]
  std::vector<float> linear2_bias;     // [1]
};

// act_head: Sequential(Linear(d+4, act_hidden), GELU(), Linear(act_hidden, n_act))
struct ActHeadWeights {
  std::vector<float> linear1_weight;  // [act_hidden, d+4]
  std::vector<float> linear1_bias;    // [act_hidden]
  std::vector<float> linear2_weight;  // [n_act, act_hidden]
  std::vector<float> linear2_bias;     // [n_act]
};

struct Weights {
  std::vector<float> type_emb;              // [3, d]
  std::vector<HeadLayerWeights> head_layers;
  ScorerWeights scorer;
  ActHeadWeights act_head;
  std::vector<float> temperature;           // [3] — not used in forward, stored for inference
};

// Materialize `Weights` from a checkpoint. Throws BY NAME on a missing tensor.
Weights Load(const Params& params, const CheckpointTensors& tensors);

// Decision head forward (batch=1).
//
//   hidden_states: [seq, hidden] (encoder output, flat row-major)
//   attention_mask: [seq] (1=valid, 0=padding)
//   marker_pos: [k_max] (positions of MASK tokens; padded entries clamped to 0)
//   marker_mask: [k_max] (1=valid marker, 0=padding)
//   qtype: 0=choice, 1=score, 2=noul
//
// Returns {logits [k_max], act_logits [n_act]}.
struct ForwardOutput {
  std::vector<float> logits;      // [k_max]
  std::vector<float> act_logits;  // [n_act]
};

ForwardOutput ForwardHost(
    const Params& params, const Weights& weights,
    const std::vector<float>& hidden_states,
    const std::vector<int64_t>& attention_mask,
    const std::vector<int64_t>& marker_pos,
    const std::vector<int64_t>& marker_mask,
    int64_t qtype);

// Resolve the temperature for a given question type and option count.
//
// Mirrors reference temp_bucket(): builds a bucket key like "choice:3-5"
// from the qtype name and the option count k, looks it up in
// temperature_by_options, and falls back to temperature[qtype] when no
// bucket matches. The returned value divides logits before softmax.
//
//   qtype: 0=choice, 1=score, 2=noul
//   k:     number of options (logits)
float TemperatureFor(
    int qtype, int k,
    const std::vector<float>& temperature,
    const std::map<std::string, float>& temperature_by_options);

}  // namespace laya

// ── Production model (Phase 4: registration) ────────────────────────────
// Forward declarations to keep this header free of the safetensors and
// config includes.
class SafetensorsFile;
struct HfConfig;

// Combined model weights: the ModernBERT encoder + the Laya decision head,
// with their parsed configs. Materialized from a safetensors checkpoint by
// LoadLayaWeights and owned by LayaLoadedModel.
struct LayaModelWeights {
  modernbert::Params encoder_params;
  modernbert::Weights encoder_weights;
  laya::Params head_params;
  laya::Weights head_weights;
  // Per-cardinality temperature map from rl_agent_config.json
  // (keys like "choice:3-5", "noul:2"). Used to scale logits before softmax,
  // matching reference temp_bucket() lookup.
  std::map<std::string, float> temperature_by_options;
};

// Production weight loader: reads all F32 tensors from the safetensors
// shards, infers the ModernBERT encoder config from weight shapes, reads
// the Laya head config from config.raw (rl_agent_config.json fields), and
// calls modernbert::Load + laya::Load.
LayaModelWeights LoadLayaWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

}  // namespace vllm
