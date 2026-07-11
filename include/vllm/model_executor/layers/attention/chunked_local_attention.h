// Ported from:
//   vllm/model_executor/layers/attention/chunked_local_attention.py:30-128
// @ e24d1b24
//
// Chunked-local attention reuses an ordinary attention backend. Its metadata
// builder first turns real requests into chunk-aligned virtual requests, then
// delegates build/update calls to the underlying builder. The backend wrapper
// is cached per underlying backend type and chunk size, matching the Python
// lru_cache. Device index arrays remain host vectors under the repository-wide
// CommonAttentionMetadata deviation.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/utils.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

template <typename Metadata>
struct ChunkedLocalAttentionBuiltMetadata {
  Metadata metadata;
  v1::LocalAttentionVirtualBatches virtual_batches;
};

// Typed adapter over an ordinary metadata builder. UnderlyingBuilder must
// provide build(common_prefix_len, CommonAttentionMetadata, fast_build) and,
// when update_block_table is called, update_block_table(metadata, flat table,
// num_cols, slot_mapping).
template <typename Metadata, typename UnderlyingBuilder>
class ChunkedLocalAttentionBuilder {
 public:
  ChunkedLocalAttentionBuilder(int attention_chunk_size, int block_size,
                               UnderlyingBuilder& underlying_builder)
      : attention_chunk_size_(attention_chunk_size),
        block_size_(block_size),
        underlying_builder_(underlying_builder) {}

  static constexpr v1::AttentionCGSupport get_cudagraph_support() {
    return v1::AttentionCGSupport::kNever;
  }

  ChunkedLocalAttentionBuiltMetadata<Metadata> build(
      int common_prefix_len,
      const v1::CommonAttentionMetadata& common_attn_metadata,
      bool fast_build = false) {
    v1::LocalAttentionVirtualBatches virtual_batches =
        v1::MakeLocalAttentionVirtualBatches(
            attention_chunk_size_, common_attn_metadata, block_size_);
    Metadata metadata = underlying_builder_.build(
        common_prefix_len, virtual_batches.common_attn_metadata, fast_build);
    return {std::move(metadata), std::move(virtual_batches)};
  }

  void update_block_table(
      ChunkedLocalAttentionBuiltMetadata<Metadata>& built,
      const std::vector<int32_t>& block_table, int block_table_num_cols,
      const std::vector<int64_t>& slot_mapping) {
    std::vector<int32_t> local_block_table =
        built.virtual_batches.make_virtual_batches_block_table(
            block_table, block_table_num_cols);
    underlying_builder_.update_block_table(
        built.metadata, local_block_table,
        built.virtual_batches.pages_per_local_batch, slot_mapping);
  }

 private:
  int attention_chunk_size_;
  int block_size_;
  UnderlyingBuilder& underlying_builder_;
};

// Backend identity/shape/implementation adapter. Metadata construction uses
// ChunkedLocalAttentionBuilder above; all compute remains in the ordinary
// backend returned by underlying_backend().
class ChunkedLocalAttentionBackend final : public v1::AttentionBackend {
 public:
  ChunkedLocalAttentionBackend(
      std::shared_ptr<const v1::AttentionBackend> underlying_backend,
      int attention_chunk_size);

  std::string get_name() const override;
  std::vector<int64_t> get_kv_cache_shape(
      int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
      int64_t head_size,
      const std::string& cache_dtype_str = "auto") const override;
  std::unique_ptr<v1::AttentionImpl> get_impl_cls() const override;

  const std::shared_ptr<const v1::AttentionBackend>& underlying_backend() const {
    return underlying_backend_;
  }
  int attention_chunk_size() const { return attention_chunk_size_; }

 private:
  std::shared_ptr<const v1::AttentionBackend> underlying_backend_;
  int attention_chunk_size_;
};

std::shared_ptr<const ChunkedLocalAttentionBackend>
CreateChunkedLocalAttentionBackend(
    std::shared_ptr<const v1::AttentionBackend> underlying_backend,
    int attention_chunk_size);

// Generic layer configuration/spec seam used by future model ports. It selects
// the cached wrapped backend supplied by the ordinary backend selector and
// emits the W3 ChunkedLocalAttentionSpec without model-specific reinterpretation.
class ChunkedLocalAttention {
 public:
  ChunkedLocalAttention(
      int num_heads, int head_size, float scale, int attention_chunk_size,
      std::optional<int> num_kv_heads,
      std::shared_ptr<const v1::AttentionBackend> underlying_backend);

  std::shared_ptr<v1::ChunkedLocalAttentionSpec> get_kv_cache_spec(
      int block_size, vt::DType dtype,
      v1::KVQuantMode kv_quant_mode = v1::KVQuantMode::kNone,
      std::optional<int64_t> page_size_padded = std::nullopt,
      bool indexes_kv_by_block_stride = false) const;

  int num_heads;
  int head_size;
  float scale;
  int attention_chunk_size;
  int num_kv_heads;
  std::shared_ptr<const ChunkedLocalAttentionBackend> backend;
};

}  // namespace vllm
