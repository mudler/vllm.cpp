// Ported from: vllm/v1/attention/backends/utils.py:225-420 @ e24d1b24
//
// Chunked-local attention is executed by splitting each real request into
// chunk-aligned virtual requests, then reusing an ordinary causal attention
// backend. The project currently represents device metadata with host arrays;
// the reusable gather plan below is the host equivalent of upstream's eagerly
// uploaded batch/block index tensors and update_block_table callback.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/v1/attention/backend.h"

namespace vllm::v1 {

struct LocalAttentionVirtualBatches {
  CommonAttentionMetadata common_attn_metadata;

  // One original request index and absolute K start per virtual request.
  std::vector<int32_t> virtual_batch_request_indices;
  std::vector<int32_t> k_seqstarts_absolute;

  // Flattened [num_virtual_batches, pages_per_local_batch] source-column
  // indices. They are clipped against the source table exactly once and then
  // reused when another KV group updates the block table.
  std::vector<int32_t> block_indices;
  int actual_batch_size = 0;
  int pages_per_local_batch = 0;
  int source_block_table_num_cols = 0;

  std::vector<int32_t> make_virtual_batches_block_table(
      const std::vector<int32_t>& block_table,
      int block_table_num_cols) const;
};

// Split common attention metadata into independent chunk-aligned causal
// virtual requests. Every input request must schedule at least one query token;
// attn_chunk_size must be a positive multiple of block_size.
LocalAttentionVirtualBatches MakeLocalAttentionVirtualBatches(
    int attn_chunk_size,
    const CommonAttentionMetadata& common_attn_metadata,
    int block_size);

}  // namespace vllm::v1
