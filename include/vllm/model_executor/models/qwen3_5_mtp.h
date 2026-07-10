// Ported from vllm/model_executor/models/qwen3_5_mtp.py @
// e24d1b24fe96a56ba8b0d653efa076d03eb95d6c.
//
// Qwen3.5/3.6 MTP draft model used by the k=1 speculative-decoding path. The
// checkpoint-owned `mtp.*` tensors are loaded separately from the target model,
// while embed_tokens and lm_head remain references to the target weights exactly
// as load_eagle_model does upstream (v1/worker/gpu/spec_decode/eagle/utils.py).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_dense.h"

namespace vllm {

enum class Qwen3_5MTPKind : uint8_t { kDense, kMoe };

// The checkpoint-owned portion of Qwen3_5MultiTokenPredictor. Both gate
// checkpoints have one layer today, but the vector mirrors upstream's
// `mtp_num_hidden_layers` and spec_step_idx modulo selection.
struct Qwen3_5MTPWeights {
  Qwen3_5MTPKind kind = Qwen3_5MTPKind::kDense;
  OwnedTensor fc;  // bf16 raw torch Linear [H,2H], nk=true
  OwnedTensor pre_fc_norm_embedding;  // bf16 [H], Gemma RMSNorm
  OwnedTensor pre_fc_norm_hidden;     // bf16 [H], Gemma RMSNorm
  OwnedTensor final_norm;             // bf16 [H], Gemma RMSNorm
  std::vector<Qwen3_5DenseLayerWeights> dense_layers;
  std::vector<Qwen3_5MoeLayerWeights> moe_layers;

  int64_t NumLayers() const;
};

// Load only `mtp.*` tensors through an existing checkpoint resolver. Every MTP
// tensor is required to be BF16 (the NVFP4 checkpoint exclusion mirrored from
// qwen3_5_mtp.py:86-103). Dedicated embeddings are rejected for this bounded
// Qwen3.6 leaf; both gate checkpoints set mtp_use_dedicated_embeddings=false.
Qwen3_5MTPWeights LoadQwen3_5MTP(const TensorResolver& get,
                                 const HfConfig& config,
                                 Qwen3_5MTPKind kind);

// Multi-shard convenience overload. It indexes the shard headers, then calls
// the resolver overload. Normal target-model loaders intentionally do not call
// this: vLLM loads the draft only when speculative decoding is enabled.
Qwen3_5MTPWeights LoadQwen3_5MTP(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    Qwen3_5MTPKind kind);

// Owning device/CPU buffer for the MTP forward's direct hidden-state return.
// This mirrors Qwen3_5MTP.forward returning hidden states (not a tuple); the
// caller may then apply the target-shared lm_head through ComputeLogits().
struct Qwen3_5MTPHiddenStates {
  std::shared_ptr<void> storage;
  vt::Tensor tensor;  // bf16 [T,H]
};

class Qwen3_5MTPModel {
 public:
  // The target and config must outlive this lightweight sharing wrapper.
  Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                  const Qwen3_5DenseWeights& target,
                  const HfConfig& config);
  Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                  const Qwen3_5MoeWeights& target,
                  const HfConfig& config);

  // Upstream load_eagle_model shares these because Qwen3.5 MTP has no own copy.
  bool has_own_embed_tokens() const { return false; }
  bool has_own_lm_head() const { return false; }
  const OwnedTensor& embed_tokens() const { return *embed_tokens_; }
  const OwnedTensor* lm_head() const { return lm_head_; }
  const Nvfp4Weight* lm_head_fp4() const { return lm_head_fp4_; }

  // input_ids/positions and target_hidden_states all have T rows. The target
  // hidden states are the target model's post-final-norm bf16 output, matching
  // qwen3_5_mtp.py:129-165. `spec_step_idx` selects one MTP layer modulo depth.
  Qwen3_5MTPHiddenStates Forward(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states, vt::Queue& queue,
      int64_t spec_step_idx = 0) const;

  // Apply the shared target lm_head to a direct MTP hidden-state return. Logits
  // remain device-resident in ForwardLogits, matching the target hot-path API.
  ForwardLogits ComputeLogits(const vt::Tensor& hidden_states,
                              vt::Queue& queue) const;

  // Standalone parity convenience: Forward + shared lm_head + one host download.
  std::vector<float> ForwardLogitsHost(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states, vt::Queue& queue,
      int64_t spec_step_idx = 0) const;

 private:
  const Qwen3_5MTPWeights* weights_ = nullptr;
  const HfConfig* config_ = nullptr;
  const OwnedTensor* embed_tokens_ = nullptr;
  const OwnedTensor* lm_head_ = nullptr;
  const Nvfp4Weight* lm_head_fp4_ = nullptr;
};

}  // namespace vllm
