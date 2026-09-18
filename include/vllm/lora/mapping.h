// LoRA metadata + mapping layer — the index arithmetic that feeds the punica
// kernels.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/layers/utils.py:27-42          LoRAMappingType, LoRAMapping
//   vllm/lora/punica_wrapper/utils.py:15-50  compute_meta (sgmv segment
//                                             clustering via unique_consecutive)
//   vllm/lora/punica_wrapper/utils.py:54-160 convert_mapping (LoRAMapping →
//                                             4 index tensors)
//   vllm/lora/punica_wrapper/punica_base.py:124-299  PunicaWrapperBase state
//                                             machine (update_metadata +
//                                             property accessors)
//
// W1+W2 own the kernel apply (shrink/expand GEMM in punica.h). W3 owns the
// index arithmetic that FEEDS those kernels: how a scheduler batch maps to
// per-token LoRA slot indices, per-request sampler indices, and the sgmv
// prefill segment structure.
//
// torch tensors become std::vector<int64_t> (the index arrays are integer
// metadata, not float compute — matching vLLM's torch.long dtype). The
// pre-allocated buffer pattern (torch.empty(max_num_batched_tokens)) becomes a
// vector sized at construction with a length field tracking the live prefix.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace lora {

// LoRAMappingType (layers/utils.py:27-30).
enum class LoRAMappingType : int {
  kLanguage = 1,
  kTower = 2,
  kConnector = 3,
};

// LoRAMapping (layers/utils.py:33-42). Maps rows in a batch to LoRA ids.
//   index_mapping: per-token LoRA id (0 or negative = no adapter).
//   prompt_mapping: per-request LoRA id (0 or negative = no adapter).
//   is_prefill: whether this is a prefill batch (triggers sgmv segmentation).
struct LoRAMapping {
  std::vector<int64_t> index_mapping;
  std::vector<int64_t> prompt_mapping;
  bool is_prefill = false;
  LoRAMappingType type = LoRAMappingType::kLanguage;
};

// Result of compute_meta (punica_wrapper/utils.py:15-50). Clusters consecutive
// identical LoRA ids in the token-lora array into sgmv segments.
struct SgmvMeta {
  // b_seq_start_tensor: start position of each segment in the token array.
  std::vector<int64_t> b_seq_start;
  // seq_length_tensor: length of each segment.
  std::vector<int64_t> seq_lengths;
  // lora_indices_tensor: the LoRA id of each segment (unique consecutive).
  std::vector<int64_t> lora_indices;
  int64_t batch_size = 0;   // number of segments after clustering.
  int64_t max_length = 0;   // max segment length.
  int64_t token_nums = 0;   // total tokens (sum of segment lengths).
  bool no_lora = false;     // true when the whole batch is a single -1 segment.
};

// compute_meta (punica_wrapper/utils.py:15-50). Mirrors
// torch.unique_consecutive(token_lora, return_counts=True) + cumulative sum.
SgmvMeta ComputeMeta(const std::vector<int64_t>& token_lora);

// Result of convert_mapping (punica_wrapper/utils.py:54-160).
struct ConvertedMapping {
  // base_indices [batch_size]: per-token LoRA slot index (-1 = no adapter).
  std::vector<int64_t> base_indices;
  // sampler_indices [n_requests]: per-request LoRA slot index (-1 = none).
  std::vector<int64_t> sampler_indices;
  // sampler_indices_padded [n_requests]: scatter index = i + slot * n_requests.
  std::vector<int64_t> sampler_indices_padded;
  // embeddings_indices [2 * batch_size] flat (row 0 at [0, batch_size),
  // row 1 at [batch_size, 2*batch_size)). Row 0 = slot * extra_vocab_size,
  // row 1 = slot * (vocab_size + extra_vocab_size), -1 clamped to max_loras-1.
  std::vector<int64_t> embeddings_indices;
  // Lengths of the above tensors (base, sampler, sampler_padded, embeddings).
  // embeddings_indices_len is the column count (batch_size), not the flat size.
  int64_t indices_len[4] = {};
};

// convert_mapping (punica_wrapper/utils.py:54-160). Converts a LoRAMapping to
// the four index tensors the punica kernels consume.
//   lora_index_to_id: slot → LoRA id mapping (-1 = unused slot, mirroring
//                      Python's None).
//   extra_vocab_size: always 0 (punica_base.py:178); kept for signature parity.
ConvertedMapping ConvertMapping(
    const LoRAMapping& mapping,
    const std::vector<int64_t>& lora_index_to_id,
    int64_t max_loras,
    int64_t vocab_size,
    int64_t extra_vocab_size);

// PunicaWrapperBase (punica_base.py:124-299). The metadata state machine.
// Owns the four pre-allocated index buffers and the sgmv prefill tensors.
// update_metadata converts a LoRAMapping into the index arrays and, for
// prefill, clusters them into sgmv segments.
//
// DEVIATION from upstream: the abstract apply methods (add_shrink, add_expand,
// add_lora_linear, add_lora_logits at punica_base.py:301-449) are NOT part of
// this class. They remain as the free functions from W1+W2 (punica.h). W5 will
// wire this wrapper's token_lora_indices accessor into those calls.
class PunicaWrapperBase {
 public:
  // punica_base.py:131-166. Pre-allocates the index buffers.
  PunicaWrapperBase(int64_t max_num_batched_tokens, int64_t max_batches);

  // punica_base.py:284-299. Converts mapping → index arrays, then (for
  // prefill) clusters them into sgmv segments.
  void UpdateMetadata(const LoRAMapping& mapping,
                      const std::vector<int64_t>& lora_index_to_id,
                      int64_t max_loras, int64_t vocab_size);

  // --- Property accessors (punica_base.py:225-282) ---

  // token_lora_indices (punica_base.py:249-256). Returns the live prefix.
  const std::vector<int64_t>& token_lora_indices() const { return token_lora_indices_; }
  int64_t token_lora_indices_len() const { return indices_len_[0]; }

  // sampler_indices (punica_base.py:258-265).
  const std::vector<int64_t>& sampler_indices() const { return sampler_indices_; }
  int64_t sampler_indices_len() const { return indices_len_[1]; }

  // sampler_indices_padded (punica_base.py:267-273).
  const std::vector<int64_t>& sampler_indices_padded() const { return sampler_indices_padded_; }
  int64_t sampler_indices_padded_len() const { return indices_len_[2]; }

  // embeddings_indices (punica_base.py:275-282). Flat [2, max_num_batched_tokens].
  // Row 0 at [0, max_num_batched_tokens), row 1 at [max_num_batched_tokens, ...].
  const std::vector<int64_t>& embeddings_indices() const { return embeddings_indices_; }
  int64_t embeddings_indices_len() const { return indices_len_[3]; }

  // prefill_metadata (punica_base.py:225-247).
  const std::vector<int64_t>& seq_start_locs() const { return seq_start_locs_; }
  const std::vector<int64_t>& seq_lengths() const { return seq_lengths_; }
  const std::vector<int64_t>& lora_indices_per_batch() const { return lora_indices_per_batch_; }
  int64_t batch_size() const { return batch_size_; }
  int64_t max_length() const { return max_length_; }
  int64_t token_nums() const { return token_nums_; }
  bool is_prefill() const { return is_prefill_; }
  bool no_lora() const { return no_lora_; }

 private:
  // punica_base.py:168-202. Converts mapping → index arrays, copies into
  // the pre-allocated buffers.
  void UpdateBaseMetadata(const LoRAMapping& mapping,
                          const std::vector<int64_t>& lora_index_to_id,
                          int64_t max_loras, int64_t vocab_size);

  // punica_base.py:204-223. Clusters token_lora_indices into sgmv segments.
  void UpdatePrefillMetadata(const std::vector<int64_t>& token_lora,
                             int64_t token_lora_len);

  // Pre-allocated index buffers (punica_base.py:138-149).
  std::vector<int64_t> token_lora_indices_;     // [max_num_batched_tokens]
  std::vector<int64_t> sampler_indices_;        // [max_num_batched_tokens]
  std::vector<int64_t> sampler_indices_padded_; // [max_num_batched_tokens]
  std::vector<int64_t> embeddings_indices_;     // [2 * max_num_batched_tokens]

  // 4 index lengths (punica_base.py:154).
  int64_t indices_len_[4] = {};

  // sgmv prefill metadata (punica_base.py:156-166).
  std::vector<int64_t> seq_start_locs_;         // [max_batches]
  std::vector<int64_t> seq_lengths_;             // [max_batches]
  std::vector<int64_t> lora_indices_per_batch_; // [max_batches]

  int64_t max_length_ = 0;
  int64_t token_nums_ = 0;
  int64_t batch_size_ = -1;
  bool is_prefill_ = false;
  bool no_lora_ = false;
};

}  // namespace lora
}  // namespace vllm
