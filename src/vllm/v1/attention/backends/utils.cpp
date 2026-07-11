// Ported from: vllm/v1/attention/backends/utils.py:225-420 @ e24d1b24
#include "vllm/v1/attention/backends/utils.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vllm::v1 {
namespace {

int Cdiv(int value, int divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

int CheckedInt64ToInt(int64_t value, const char* field) {
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::overflow_error(std::string(field) + " exceeds int range");
  }
  return static_cast<int>(value);
}

}  // namespace

std::vector<int32_t>
LocalAttentionVirtualBatches::make_virtual_batches_block_table(
    const std::vector<int32_t>& block_table,
    int block_table_num_cols) const {
  if (block_table_num_cols <= 0) {
    throw std::invalid_argument("block table must have at least one column");
  }
  const size_t required = static_cast<size_t>(actual_batch_size) *
                          static_cast<size_t>(block_table_num_cols);
  if (block_table.size() < required) {
    throw std::invalid_argument("block table is smaller than its declared shape");
  }
  const size_t expected_indices =
      virtual_batch_request_indices.size() *
      static_cast<size_t>(pages_per_local_batch);
  if (block_indices.size() != expected_indices) {
    throw std::logic_error("invalid chunked-local block-table gather plan");
  }

  std::vector<int32_t> local_block_table;
  local_block_table.reserve(block_indices.size());
  for (size_t virtual_batch = 0;
       virtual_batch < virtual_batch_request_indices.size(); ++virtual_batch) {
    const int request = virtual_batch_request_indices[virtual_batch];
    if (request < 0 || request >= actual_batch_size) {
      throw std::logic_error("invalid original request in gather plan");
    }
    for (int page = 0; page < pages_per_local_batch; ++page) {
      const size_t plan_index =
          virtual_batch * static_cast<size_t>(pages_per_local_batch) +
          static_cast<size_t>(page);
      const int column = block_indices[plan_index];
      if (column < 0 || column >= block_table_num_cols) {
        throw std::invalid_argument(
            "updated block table is incompatible with the gather plan");
      }
      const size_t source_index =
          static_cast<size_t>(request) *
              static_cast<size_t>(block_table_num_cols) +
          static_cast<size_t>(column);
      local_block_table.push_back(block_table[source_index]);
    }
  }
  return local_block_table;
}

LocalAttentionVirtualBatches MakeLocalAttentionVirtualBatches(
    int attn_chunk_size,
    const CommonAttentionMetadata& common_attn_metadata,
    int block_size) {
  if (attn_chunk_size <= 0 || block_size <= 0) {
    throw std::invalid_argument(
        "attention chunk size and block size must be positive");
  }
  if (attn_chunk_size % block_size != 0) {
    throw std::invalid_argument("attention chunk size must divide into blocks");
  }
  const int actual_batch_size = common_attn_metadata.num_reqs;
  if (actual_batch_size <= 0 ||
      common_attn_metadata.query_start_loc_cpu.size() !=
          static_cast<size_t>(actual_batch_size + 1) ||
      common_attn_metadata.seq_lens_cpu.size() !=
          static_cast<size_t>(actual_batch_size)) {
    throw std::invalid_argument("invalid common attention metadata shape");
  }
  const int source_num_cols = common_attn_metadata.block_table_num_cols;
  if (source_num_cols <= 0 ||
      common_attn_metadata.block_table_tensor.size() <
          static_cast<size_t>(actual_batch_size) *
              static_cast<size_t>(source_num_cols)) {
    throw std::invalid_argument("invalid common attention block-table shape");
  }

  const std::vector<int32_t> query_lens =
      common_attn_metadata.naive_query_lens();
  std::vector<int> first_block_query_tokens(
      static_cast<size_t>(actual_batch_size));
  std::vector<int> last_block_tokens(static_cast<size_t>(actual_batch_size));
  std::vector<int> local_blocks(static_cast<size_t>(actual_batch_size));
  int virtual_batch_count = 0;

  for (int request = 0; request < actual_batch_size; ++request) {
    const int query_len = query_lens[static_cast<size_t>(request)];
    const int seq_len =
        common_attn_metadata.seq_lens_cpu[static_cast<size_t>(request)];
    const int computed_tokens = seq_len - query_len;
    if (query_len <= 0 || computed_tokens < 0) {
      throw std::invalid_argument(
          "chunked-local attention requires positive queries and valid sequence lengths");
    }
    const int first =
        std::min(attn_chunk_size - computed_tokens % attn_chunk_size,
                 query_len);
    const int remainder = seq_len % attn_chunk_size;
    const int last = remainder == 0 ? attn_chunk_size : remainder;
    const int blocks = 1 + Cdiv(query_len - first, attn_chunk_size);
    first_block_query_tokens[static_cast<size_t>(request)] = first;
    last_block_tokens[static_cast<size_t>(request)] = last;
    local_blocks[static_cast<size_t>(request)] = blocks;
    virtual_batch_count += blocks;
  }

  LocalAttentionVirtualBatches result;
  result.actual_batch_size = actual_batch_size;
  result.pages_per_local_batch = attn_chunk_size / block_size;
  result.source_block_table_num_cols = source_num_cols;
  result.virtual_batch_request_indices.reserve(
      static_cast<size_t>(virtual_batch_count));
  result.k_seqstarts_absolute.reserve(static_cast<size_t>(virtual_batch_count));
  result.block_indices.reserve(
      static_cast<size_t>(virtual_batch_count) *
      static_cast<size_t>(result.pages_per_local_batch));

  CommonAttentionMetadata local;
  local.query_start_loc_cpu.reserve(
      static_cast<size_t>(virtual_batch_count + 1));
  local.query_start_loc_cpu.push_back(0);
  local.seq_lens_cpu.reserve(static_cast<size_t>(virtual_batch_count));
  local.num_computed_tokens_cpu.reserve(
      static_cast<size_t>(virtual_batch_count));

  int64_t cumulative_query_tokens = 0;
  for (int request = 0; request < actual_batch_size; ++request) {
    const int query_len = query_lens[static_cast<size_t>(request)];
    const int seq_len =
        common_attn_metadata.seq_lens_cpu[static_cast<size_t>(request)];
    const int first = first_block_query_tokens[static_cast<size_t>(request)];
    const int last = last_block_tokens[static_cast<size_t>(request)];
    const int blocks = local_blocks[static_cast<size_t>(request)];

    for (int local_block = 0; local_block < blocks; ++local_block) {
      int local_query_len = first;
      if (local_block > 0) {
        local_query_len = std::min(
            query_len - first - attn_chunk_size * (local_block - 1),
            attn_chunk_size);
      }
      const int reverse_block = blocks - local_block - 1;
      const int local_k_len = local_block == blocks - 1 ? last : attn_chunk_size;
      const int k_start =
          seq_len - (reverse_block * attn_chunk_size + last);
      const int local_computed_tokens = local_k_len - local_query_len;
      if (local_query_len <= 0 || local_computed_tokens < 0 || k_start < 0) {
        throw std::logic_error("invalid chunked-local virtual batch");
      }

      cumulative_query_tokens += local_query_len;
      local.query_start_loc_cpu.push_back(CheckedInt64ToInt(
          cumulative_query_tokens, "virtual query start"));
      local.seq_lens_cpu.push_back(local_k_len);
      local.num_computed_tokens_cpu.push_back(local_computed_tokens);
      result.virtual_batch_request_indices.push_back(request);
      result.k_seqstarts_absolute.push_back(k_start);

      const int block_start = k_start / block_size;
      for (int page = 0; page < result.pages_per_local_batch; ++page) {
        result.block_indices.push_back(
            std::min(block_start + page, source_num_cols - 1));
      }
    }
  }

  if (cumulative_query_tokens != common_attn_metadata.num_actual_tokens) {
    throw std::invalid_argument(
        "num_actual_tokens does not match the query lengths");
  }

  local.query_start_loc = local.query_start_loc_cpu;
  local.seq_lens = local.seq_lens_cpu;
  local.num_reqs = virtual_batch_count;
  local.num_actual_tokens = common_attn_metadata.num_actual_tokens;
  local.max_query_len = 0;
  for (size_t i = 1; i < local.query_start_loc_cpu.size(); ++i) {
    local.max_query_len =
        std::max(local.max_query_len,
                 local.query_start_loc_cpu[i] -
                     local.query_start_loc_cpu[i - 1]);
  }
  local.max_seq_len =
      *std::max_element(local.seq_lens_cpu.begin(), local.seq_lens_cpu.end());
  local.block_table_num_cols = result.pages_per_local_batch;
  local.block_table_tensor = result.make_virtual_batches_block_table(
      common_attn_metadata.block_table_tensor, source_num_cols);
  local.slot_mapping = common_attn_metadata.slot_mapping;
  local.causal = true;

  result.common_attn_metadata = std::move(local);
  return result;
}

}  // namespace vllm::v1
