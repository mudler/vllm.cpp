// cua-s1-forms TinyTransformerScorer — option scorer (MODEL-CUA-S1-FORMS).
//
// Ported from trycua/cua libs/cua-s1/python/src/cua_s1/model.py @ 9bbfa7dd:
//   ByteCollator          model.py:52-99
//   AttentionHead          model.py:102-136
//   TinyTransformerScorer  model.py:166-236
//
// This is the HOST REFERENCE forward: a portable f32 implementation gated
// against the PyTorch model before any device path exists. It is not wired
// to the runner, the ABI or the server; that is Phase 3-4 of
// .agents/specs/cua-s1-forms.md.
//
// THE THINGS THIS ARCHITECTURE GETS WRONG QUIETLY are: (1) swapping the Q/K
// split order in the packed in_proj_weight — attention still produces a result
// but it is numerically wrong; (2) using -inf instead of finfo.min for masking
// — produces NaN when an entire row is masked, which the safe-mask prevents
// but the wrong fill value hides the mutation test; (3) skipping the safe mask
// force at position 0 — the model still runs on typical inputs. Each is gated
// by a perturbation test.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {
namespace cua_s1 {

// Config fields from cua-s1-forms.json (inside the "config" envelope).
struct Params {
  int64_t width = 128;
  int64_t rank = 128;
  int64_t context_tokens = 224;
  int64_t option_tokens = 96;
  int64_t layers = 2;
  int64_t heads = 4;

  int64_t head_dim() const { return width / heads; }
  int64_t dim_ff() const { return width * 4; }
  int64_t max_pos() const {
    return context_tokens > option_tokens ? context_tokens : option_tokens;
  }
};

// Name → shape + flat data, matching the safetensors checkpoint.
class CheckpointTensors {
 public:
  void Set(const std::string& name, std::vector<int64_t> shape,
           std::vector<float> data);
  const std::vector<float>& Get(const std::string& name) const;
  const std::vector<int64_t>& Shape(const std::string& name) const;

 private:
  std::map<std::string, std::vector<int64_t>> shapes;
  std::map<std::string, std::vector<float>> values;
};

// Per-layer weights for nn.TransformerEncoderLayer (norm_first=True).
struct EncoderLayerWeights {
  std::vector<float> in_proj_weight;    // (3*width, width) packed QKV
  std::vector<float> in_proj_bias;     // (3*width,)
  std::vector<float> out_proj_weight;   // (width, width)
  std::vector<float> out_proj_bias;     // (width,)
  std::vector<float> linear1_weight;    // (dim_ff, width)
  std::vector<float> linear1_bias;      // (dim_ff,)
  std::vector<float> linear2_weight;    // (width, dim_ff)
  std::vector<float> linear2_bias;      // (width,)
  std::vector<float> norm1_weight;      // (width,)
  std::vector<float> norm1_bias;        // (width,)
  std::vector<float> norm2_weight;      // (width,)
  std::vector<float> norm2_bias;        // (width,)
};

struct AttentionHeadWeights {
  std::vector<float> context_norm_weight;  // (width,)
  std::vector<float> context_norm_bias;      // (width,)
  std::vector<float> option_norm_weight;     // (width,)
  std::vector<float> option_norm_bias;       // (width,)
  std::vector<float> query_weight;           // (rank, width) no bias
  std::vector<float> key_weight;             // (rank, width) no bias
  std::vector<float> value_weight;           // (rank, width) no bias
};

struct Weights {
  std::vector<float> embedding_weight;  // (257, width) padding_idx=0
  std::vector<float> position_weight;   // (max_pos, width)
  std::vector<EncoderLayerWeights> encoder_layers;       // context encoder
  std::vector<EncoderLayerWeights> option_encoder_layers; // option encoder
  AttentionHeadWeights head;
};

Weights Load(const Params& params, const CheckpointTensors& tensors);

// Host-only f32 forward. Returns logits (n_opt values).
// batch is implicitly 1.
//
// context_ids:    (ctx_len,) byte token ids
// context_mask:   (ctx_len,) bool — True = valid
// option_ids:     (n_opt * opt_len,) flattened, row-major
// option_tok_mask:(n_opt * opt_len,) bool — True = valid
// option_mask:    (n_opt,) bool — True = valid option
std::vector<float> ForwardHost(
    const Params& params, const Weights& weights,
    const std::vector<int64_t>& context_ids,
    const std::vector<uint8_t>& context_mask,
    const std::vector<int64_t>& option_ids,
    const std::vector<uint8_t>& option_tok_mask,
    const std::vector<uint8_t>& option_mask);

}  // namespace cua_s1

// Forward declarations for the production weight loader (defined in
// cua_s1_weights.cpp). The full types live in safetensors_reader.h and
// hf_config.h, which are not included here to keep this header lightweight.
class SafetensorsFile;
struct HfConfig;

// Combined model weights: parsed params + loaded weight tensors. Materialized
// from a safetensors checkpoint by LoadCuaS1Weights and owned by
// CuaS1LoadedModel.
struct CuaS1ModelWeights {
  cua_s1::Params params;
  cua_s1::Weights weights;
};

// Production weight loader: reads all F32 tensors from the safetensors
// shards, parses the model params from config.raw (the unwrapped
// cua-s1-forms.json config envelope), and calls cua_s1::Load.
CuaS1ModelWeights LoadCuaS1Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

}  // namespace vllm
