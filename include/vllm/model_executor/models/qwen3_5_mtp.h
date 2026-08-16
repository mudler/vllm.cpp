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

// The MTP head depth, read from the checkpoint config (`mtp_num_hidden_layers`,
// under `text_config` when the config nests it). Defaults to 1, which is what
// both Qwen3.5/3.6 gate checkpoints ship. Exported because the GGUF head loader
// (`SPEC-MTP-GGUF`) resolves the same value from the same place, after
// HfConfigFromGguf republishes it from `<arch>.nextn_predict_layers`.
int64_t NumMtpLayers(const HfConfig& config);

// Whether the checkpoint gives the MTP head its OWN embedding table rather than
// sharing the target's (`mtp_use_dedicated_embeddings`). Both gate checkpoints
// set it false, and the loaders reject the true case.
bool UsesDedicatedEmbeddings(const HfConfig& config);

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

  // PAGED MTP draft forward (SPEC-MTP I5c). Same head math as Forward (the
  // fc-cat-norm over embed(input_ids) + target_hidden), but the single
  // full_attention decoder layer reads/writes the DRAFT KV-cache layer through
  // the paged attention path (vt::ReshapeAndCache + vt::PagedAttention) using the
  // target's block table / slot mapping — mirroring qwen3_5_mtp.py:129-165 driven
  // over the paged attention backend the target layers already use, exactly what
  // AutoRegressiveSpeculator._prefill (speculator.py:332-370) runs each step. The
  // MTP head is layer_type="full_attention" (qwen3_5_mtp.py:105-112) so no GDN
  // path is ever taken; `attn_meta` carries slot_mapping / block_table / seq_lens
  // / query_start_loc and `draft_kv` is the head's own paged K/V layer (index
  // num_hidden_layers upstream). Over a single-request 1-block KV with a trivial
  // slot map this reproduces Forward's dense result (the paged==dense anchor).
  Qwen3_5MTPHiddenStates ForwardPaged(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states,
      const v1::CommonAttentionMetadata& attn_meta, PagedKvCache& draft_kv,
      vt::Queue& queue, int64_t spec_step_idx = 0) const;

  // Gather `rows` of a [T,H] bf16 device hidden-state tensor into a fresh,
  // owning [rows.size(), H] bf16 device buffer (SPEC-MTP-K-GT-1, #81).
  //
  // The mirror of `self.hidden_states[:num_reqs] = hidden_states[
  // last_token_indices]` (autoregressive/speculator.py:367-371 @ 555967922),
  // which is how the draft PREFILL hands its per-request hidden state to the
  // first multi-step draft DECODE. It lives on the model, not in the speculator,
  // because the device-buffer pool and the row-copy idiom are private to
  // qwen3_5.cpp; a copy in the speculator would be a second, parallel path to
  // the same allocator. `hidden_states` must be contiguous bf16 [T,H] on the
  // queue's device and every row index must be in [0, T).
  Qwen3_5MTPHiddenStates GatherHiddenRows(const vt::Tensor& hidden_states,
                                          const std::vector<int64_t>& rows,
                                          vt::Queue& queue) const;

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
