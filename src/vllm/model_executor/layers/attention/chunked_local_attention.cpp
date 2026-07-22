// Ported from:
//   vllm/model_executor/layers/attention/chunked_local_attention.py:30-128
// @ e24d1b24
#include "vllm/model_executor/layers/attention/chunked_local_attention.h"

#include <functional>
#include <mutex>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace vllm {
namespace {

struct BackendCacheKey {
  std::type_index underlying_type;
  int attention_chunk_size;

  bool operator==(const BackendCacheKey&) const = default;
};

struct BackendCacheKeyHash {
  size_t operator()(const BackendCacheKey& key) const {
    const size_t type_hash = key.underlying_type.hash_code();
    const size_t chunk_hash = std::hash<int>{}(key.attention_chunk_size);
    return type_hash ^ (chunk_hash + 0x9e3779b9U + (type_hash << 6U) +
                        (type_hash >> 2U));
  }
};

using BackendCache =
    std::unordered_map<BackendCacheKey,
                       std::weak_ptr<const ChunkedLocalAttentionBackend>,
                       BackendCacheKeyHash>;

BackendCache& backend_cache() {
  static BackendCache cache;
  return cache;
}

std::mutex& backend_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace

ChunkedLocalAttentionBackend::ChunkedLocalAttentionBackend(
    std::shared_ptr<const v1::AttentionBackend> underlying_backend,
    int attention_chunk_size)
    : underlying_backend_(std::move(underlying_backend)),
      attention_chunk_size_(attention_chunk_size) {
  if (underlying_backend_ == nullptr) {
    throw std::invalid_argument(
        "chunked-local attention requires an underlying backend");
  }
  if (attention_chunk_size_ <= 0) {
    throw std::invalid_argument("attention chunk size must be positive");
  }
}

std::string ChunkedLocalAttentionBackend::get_name() const {
  return "ChunkedLocalAttention_" +
         std::to_string(attention_chunk_size_) + "_" +
         underlying_backend_->get_name();
}

std::vector<int64_t> ChunkedLocalAttentionBackend::get_kv_cache_shape(
    int64_t num_blocks, int64_t block_size, int64_t num_kv_heads,
    int64_t head_size, const std::string& cache_dtype_str) const {
  return underlying_backend_->get_kv_cache_shape(
      num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str);
}

std::unique_ptr<v1::AttentionImpl>
ChunkedLocalAttentionBackend::get_impl_cls() const {
  return underlying_backend_->get_impl_cls();
}

std::shared_ptr<const ChunkedLocalAttentionBackend>
CreateChunkedLocalAttentionBackend(
    std::shared_ptr<const v1::AttentionBackend> underlying_backend,
    int attention_chunk_size) {
  if (underlying_backend == nullptr) {
    throw std::invalid_argument(
        "chunked-local attention requires an underlying backend");
  }
  if (attention_chunk_size <= 0) {
    throw std::invalid_argument("attention chunk size must be positive");
  }
  // The DYNAMIC type of the underlying backend is deliberately part of the cache
  // key, so this typeid IS meant to be evaluated at run time. Binding the
  // dereference to a named reference first makes the typeid operand a plain
  // glvalue instead of a `shared_ptr::operator*` call expression — Clang's
  // -Wpotentially-evaluated-expression fires on the latter ("expression with
  // side effects will be evaluated despite being used as an operand to
  // 'typeid'"), which on macOS is a -Werror build break (BACKEND-METAL-MLX W0
  // item 3). Behaviour is unchanged: same polymorphic operand, same dynamic
  // type, still evaluated.
  const v1::AttentionBackend& underlying_ref = *underlying_backend;
  const BackendCacheKey key{std::type_index(typeid(underlying_ref)),
                            attention_chunk_size};
  std::lock_guard<std::mutex> lock(backend_cache_mutex());
  auto& cache = backend_cache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (auto existing = it->second.lock()) {
      return existing;
    }
  }
  auto backend = std::make_shared<const ChunkedLocalAttentionBackend>(
      std::move(underlying_backend), attention_chunk_size);
  cache[key] = backend;
  return backend;
}

ChunkedLocalAttention::ChunkedLocalAttention(
    int num_heads, int head_size, float scale, int attention_chunk_size,
    std::optional<int> num_kv_heads,
    std::shared_ptr<const v1::AttentionBackend> underlying_backend)
    : num_heads(num_heads),
      head_size(head_size),
      scale(scale),
      attention_chunk_size(attention_chunk_size),
      num_kv_heads(num_kv_heads.value_or(num_heads)),
      backend(CreateChunkedLocalAttentionBackend(
          std::move(underlying_backend), attention_chunk_size)) {
  if (num_heads <= 0 || head_size <= 0 || this->num_kv_heads <= 0) {
    throw std::invalid_argument(
        "chunked-local attention head counts and size must be positive");
  }
}

std::shared_ptr<v1::ChunkedLocalAttentionSpec>
ChunkedLocalAttention::get_kv_cache_spec(
    int block_size, vt::DType dtype, v1::KVQuantMode kv_quant_mode,
    std::optional<int64_t> page_size_padded,
    bool indexes_kv_by_block_stride) const {
  if (block_size <= 0) {
    throw std::invalid_argument("KV cache block size must be positive");
  }
  return std::make_shared<v1::ChunkedLocalAttentionSpec>(
      block_size, num_kv_heads, head_size, dtype, attention_chunk_size,
      kv_quant_mode, page_size_padded, indexes_kv_by_block_stride);
}

}  // namespace vllm
