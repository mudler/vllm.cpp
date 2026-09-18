// LoRA metadata + mapping layer implementation.
//
// UPSTREAM (ported FROM, ${VLLM_SOURCE} @ 555967922):
//   vllm/lora/punica_wrapper/utils.py:15-50   compute_meta
//   vllm/lora/punica_wrapper/utils.py:54-160  convert_mapping
//   vllm/lora/punica_wrapper/punica_base.py:168-299  _update_base_metadata,
//                                                  _update_prefill_metadata,
//                                                  update_metadata
#include "vllm/lora/mapping.h"

#include <algorithm>
#include <unordered_map>

namespace vllm {
namespace lora {

// compute_meta (punica_wrapper/utils.py:15-50).
// Mirrors torch.unique_consecutive(token_lora, return_counts=True) + cumsum.
SgmvMeta ComputeMeta(const std::vector<int64_t>& token_lora) {
  SgmvMeta meta;

  if (token_lora.empty()) {
    return meta;
  }

  // unique_consecutive: group consecutive identical values.
  int64_t i = 0;
  const int64_t n = static_cast<int64_t>(token_lora.size());
  while (i < n) {
    int64_t j = i;
    while (j < n && token_lora[static_cast<size_t>(j)] == token_lora[static_cast<size_t>(i)]) {
      ++j;
    }
    meta.lora_indices.push_back(token_lora[static_cast<size_t>(i)]);
    meta.seq_lengths.push_back(j - i);
    i = j;
  }

  // cumsum → b_seq_start (starts at 0, each subsequent start = prefix sum).
  const int64_t num_segments = static_cast<int64_t>(meta.seq_lengths.size());
  meta.b_seq_start.resize(static_cast<size_t>(num_segments));
  int64_t acc = 0;
  for (int64_t k = 0; k < num_segments; ++k) {
    meta.b_seq_start[static_cast<size_t>(k)] = acc;
    acc += meta.seq_lengths[static_cast<size_t>(k)];
  }

  // max_length = max(seq_lengths).
  meta.max_length = 0;
  for (int64_t k = 0; k < num_segments; ++k) {
    meta.max_length = std::max(meta.max_length, meta.seq_lengths[static_cast<size_t>(k)]);
  }

  meta.token_nums = acc;  // sum(seq_lengths) == total tokens.
  meta.batch_size = num_segments;

  // no_lora: true when the whole batch is one segment with lora_id == -1
  // (punica_wrapper/utils.py:40-41).
  meta.no_lora = (num_segments == 1 && meta.lora_indices[0] == -1);

  return meta;
}

// convert_mapping (punica_wrapper/utils.py:54-160).
ConvertedMapping ConvertMapping(
    const LoRAMapping& mapping,
    const std::vector<int64_t>& lora_index_to_id,
    int64_t max_loras,
    int64_t vocab_size,
    int64_t extra_vocab_size) {
  ConvertedMapping result;

  // Build reverse lookup: LoRA id → slot index (punica_wrapper/utils.py:98-102).
  // -1 in lora_index_to_id means None (unused slot).
  std::unordered_map<int64_t, int64_t> lora_id_to_index;
  for (int64_t i = 0; i < static_cast<int64_t>(lora_index_to_id.size()); ++i) {
    int64_t id = lora_index_to_id[static_cast<size_t>(i)];
    if (id != -1) {
      lora_id_to_index[id] = i;
    }
  }

  // prompt_mapping: lora_id → slot index, -1 if no adapter
  // (punica_wrapper/utils.py:104-106).
  const int64_t prompt_len = static_cast<int64_t>(mapping.prompt_mapping.size());
  result.sampler_indices.resize(static_cast<size_t>(prompt_len));
  for (int64_t i = 0; i < prompt_len; ++i) {
    int64_t x = mapping.prompt_mapping[static_cast<size_t>(i)];
    auto it = lora_id_to_index.find(x);
    result.sampler_indices[static_cast<size_t>(i)] = (x > 0 && it != lora_id_to_index.end())
                                                          ? it->second
                                                          : -1;
  }

  // lora_indices (base_indices) and embedding_indices
  // (punica_wrapper/utils.py:108-115).
  const int64_t batch_len = static_cast<int64_t>(mapping.index_mapping.size());
  result.base_indices.resize(static_cast<size_t>(batch_len));
  std::vector<int64_t> embedding_indices(static_cast<size_t>(batch_len));
  for (int64_t i = 0; i < batch_len; ++i) {
    int64_t idx_val = mapping.index_mapping[static_cast<size_t>(i)];
    auto it = lora_id_to_index.find(idx_val);
    int64_t lora_idx = (idx_val > 0 && it != lora_id_to_index.end()) ? it->second : -1;
    result.base_indices[static_cast<size_t>(i)] = lora_idx;
    // embedding_indices: slot index if adapter, 0 if none (never -1).
    embedding_indices[static_cast<size_t>(i)] = (idx_val > 0 && it != lora_id_to_index.end())
                                                    ? lora_idx
                                                    : 0;
  }

  // embeddings_indices = stack([embedding_indices * extra_vocab_size,
  //                              embedding_indices * (vocab_size + extra_vocab_size)])
  // then torch.where(== -1, max_loras - 1, ...) (punica_wrapper/utils.py:127-135).
  // embedding_indices is never -1 (0 or slot index), so the clamp is a no-op,
  // but we implement it for correctness.
  result.embeddings_indices.resize(static_cast<size_t>(batch_len * 2));
  for (int64_t i = 0; i < batch_len; ++i) {
    int64_t ei = embedding_indices[static_cast<size_t>(i)];
    int64_t row0 = ei * extra_vocab_size;
    int64_t row1 = ei * (vocab_size + extra_vocab_size);
    result.embeddings_indices[static_cast<size_t>(i)] = (row0 == -1) ? max_loras - 1 : row0;
    result.embeddings_indices[static_cast<size_t>(batch_len + i)] =
        (row1 == -1) ? max_loras - 1 : row1;
  }

  // sampler_indices_padded (punica_wrapper/utils.py:138-144):
  //   replace -1 with max_loras - 1, then arange(0, n) + padded * n.
  const int64_t n = prompt_len;
  result.sampler_indices_padded.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    int64_t si = result.sampler_indices[static_cast<size_t>(i)];
    int64_t padded = (si == -1) ? max_loras - 1 : si;
    result.sampler_indices_padded[static_cast<size_t>(i)] = i + padded * n;
  }

  // indices_len (punica_wrapper/utils.py:147-152).
  result.indices_len[0] = batch_len;   // base_indices
  result.indices_len[1] = prompt_len;  // sampler_indices
  result.indices_len[2] = n;            // sampler_indices_padded
  result.indices_len[3] = batch_len;    // embeddings_indices (column count)

  return result;
}

// --- PunicaWrapperBase ---

// punica_base.py:131-166.
PunicaWrapperBase::PunicaWrapperBase(int64_t max_num_batched_tokens,
                                     int64_t max_batches)
    : token_lora_indices_(static_cast<size_t>(max_num_batched_tokens), 0),
      sampler_indices_(static_cast<size_t>(max_num_batched_tokens), 0),
      sampler_indices_padded_(static_cast<size_t>(max_num_batched_tokens), 0),
      embeddings_indices_(static_cast<size_t>(max_num_batched_tokens * 2), 0),
      seq_start_locs_(static_cast<size_t>(max_batches), 0),
      seq_lengths_(static_cast<size_t>(max_batches), 0),
      lora_indices_per_batch_(static_cast<size_t>(max_batches), 0) {}

// punica_base.py:168-202.
void PunicaWrapperBase::UpdateBaseMetadata(
    const LoRAMapping& mapping,
    const std::vector<int64_t>& lora_index_to_id,
    int64_t max_loras, int64_t vocab_size) {
  // extra_vocab_size is always 0 (punica_base.py:178).
  constexpr int64_t extra_vocab_size = 0;

  ConvertedMapping conv = ConvertMapping(
      mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size);

  // Copy into pre-allocated buffers (punica_base.py:193-202).
  std::copy(conv.base_indices.begin(), conv.base_indices.end(),
            token_lora_indices_.begin());
  std::copy(conv.sampler_indices.begin(), conv.sampler_indices.end(),
            sampler_indices_.begin());
  std::copy(conv.sampler_indices_padded.begin(), conv.sampler_indices_padded.end(),
            sampler_indices_padded_.begin());

  // embeddings_indices is [2, batch_len] from ConvertMapping (row-major flat:
  // row 0 at [0, batch_len), row 1 at [batch_len, 2*batch_len)). The
  // pre-allocated buffer is [2, max_num_batched_tokens] row-major (row 1 at
  // offset max_num_batched_tokens). Copy each row to the correct offset,
  // mirroring self._embeddings_indices[:shape0, :shape1].copy_(embeddings).
  const int64_t batch_len = conv.indices_len[0];
  const int64_t stride = static_cast<int64_t>(embeddings_indices_.size()) / 2;
  std::copy_n(conv.embeddings_indices.begin(), batch_len,
              embeddings_indices_.begin());
  std::copy_n(conv.embeddings_indices.begin() + batch_len, batch_len,
              embeddings_indices_.begin() + stride);

  for (int i = 0; i < 4; ++i) {
    indices_len_[i] = conv.indices_len[i];
  }
}

// punica_base.py:204-223.
void PunicaWrapperBase::UpdatePrefillMetadata(
    const std::vector<int64_t>& token_lora, int64_t token_lora_len) {
  // Pass only the live prefix to ComputeMeta (punica_base.py:296 passes
  // self.token_lora_indices which is the [:token_lora_len] view).
  std::vector<int64_t> live(token_lora.begin(),
                            token_lora.begin() + token_lora_len);

  SgmvMeta meta = ComputeMeta(live);

  std::copy(meta.b_seq_start.begin(), meta.b_seq_start.end(),
            seq_start_locs_.begin());
  std::copy(meta.seq_lengths.begin(), meta.seq_lengths.end(),
            seq_lengths_.begin());
  std::copy(meta.lora_indices.begin(), meta.lora_indices.end(),
            lora_indices_per_batch_.begin());

  batch_size_ = meta.batch_size;
  max_length_ = meta.max_length;
  token_nums_ = meta.token_nums;
  no_lora_ = meta.no_lora;
}

// punica_base.py:284-299.
void PunicaWrapperBase::UpdateMetadata(
    const LoRAMapping& mapping,
    const std::vector<int64_t>& lora_index_to_id,
    int64_t max_loras, int64_t vocab_size) {
  UpdateBaseMetadata(mapping, lora_index_to_id, max_loras, vocab_size);

  if (mapping.is_prefill) {
    UpdatePrefillMetadata(token_lora_indices_, indices_len_[0]);
    is_prefill_ = true;
  } else {
    is_prefill_ = false;
  }
}

}  // namespace lora
}  // namespace vllm
