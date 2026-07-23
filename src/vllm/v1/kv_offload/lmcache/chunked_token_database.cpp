// Ported from: lmcache/v1/token_database.py:34-56,234-449 @ LMCache 8570aad.
// See include/vllm/v1/kv_offload/lmcache/chunked_token_database.h for the exact
// derivation and its citations.
#include "vllm/v1/kv_offload/lmcache/chunked_token_database.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "vllm/v1/core/kv_cache_utils.h"  // CborValue, sha256_cbor (byte-exact)

namespace vllm::v1::kv_offload::lmcache {
namespace {

// int.from_bytes(digest[:8], "big") (token_database.py:54-55).
uint64_t Fold8(const std::string& digest) {
  uint64_t v = 0;
  const std::size_t n = std::min<std::size_t>(digest.size(), 8);
  for (std::size_t i = 0; i < n; ++i) {
    v = (v << 8) | static_cast<uint8_t>(digest[i]);
  }
  return v;
}

}  // namespace

ChunkedTokenDatabase::ChunkedTokenDatabase(int chunk_size, uint64_t none_hash,
                                           bool save_unfull_chunk,
                                           PreCachingHash hash)
    : chunk_size_(chunk_size),
      none_hash_(none_hash),
      save_unfull_chunk_(save_unfull_chunk),
      hash_(hash) {
  if (chunk_size_ <= 0) {
    throw std::runtime_error("ChunkedTokenDatabase: chunk_size must be > 0");
  }
}

uint64_t ChunkedTokenDatabase::NoneHashFromSeed(const std::string& seed) {
  // init_none_hash(sha256_cbor): NONE_HASH = sha256_cbor(str(seed))  (32 bytes);
  // then _normalize_hash_to_int folds the first 8 big-endian bytes to a uint64.
  return Fold8(::vllm::v1::sha256_cbor(::vllm::v1::CborValue::Text(seed)));
}

uint64_t ChunkedTokenDatabase::HashTokens(
    const std::vector<int32_t>& chunk_tokens, uint64_t prefix) const {
  // hash_func((canon_prefix, canon_tokens, canon_extra)) where the tuple is the
  // CBOR "Any" input: (prefix:int, tuple(tokens), ()) (token_database.py:263-295).
  std::vector<::vllm::v1::CborValue> tok_items;
  tok_items.reserve(chunk_tokens.size());
  for (int32_t t : chunk_tokens) {
    // Token ids are non-negative; cbor2 canonical encodes them as minimal-width
    // unsigned ints, which CborValue::Int(>=0) -> UInt reproduces exactly.
    tok_items.push_back(::vllm::v1::CborValue::Int(t));
  }
  // The prefix is a folded uint64 (may exceed int64 max), so it MUST use the
  // unsigned CBOR major type — cbor2 encodes a Python int in [0, 2^64) as major 0.
  ::vllm::v1::CborValue input = ::vllm::v1::CborValue::Array({
      ::vllm::v1::CborValue::UInt(prefix),
      ::vllm::v1::CborValue::Array(std::move(tok_items)),
      ::vllm::v1::CborValue::Array({}),  // empty extra_keys tuple -> CBOR 0x80
  });
  switch (hash_) {
    case PreCachingHash::kSha256Cbor:
      return Fold8(::vllm::v1::sha256_cbor(input));
  }
  return 0;  // unreachable
}

std::vector<ChunkedTokenDatabase::Entry> ChunkedTokenDatabase::ProcessTokens(
    const std::vector<int32_t>& tokens) const {
  // _chunk_tokens (token_database.py:334-356): stop at the last full chunk unless
  // save_unfull_chunk keeps the trailing partial.
  const int n = static_cast<int>(tokens.size());
  const int end_limit = save_unfull_chunk_ ? n : (n - n % chunk_size_);
  std::vector<Entry> out;
  uint64_t prefix = none_hash_;
  for (int i = 0; i < end_limit; i += chunk_size_) {
    const int start = i;
    const int end = std::min(i + chunk_size_, n);
    const std::vector<int32_t> chunk(tokens.begin() + start,
                                     tokens.begin() + end);
    prefix = HashTokens(chunk, prefix);  // rolling
    out.push_back(Entry{start, end, prefix});
  }
  return out;
}

std::vector<std::string> ChunkedTokenDatabase::ProcessKeys(
    const std::vector<int32_t>& tokens, const std::string& model_name,
    int64_t world_size, int64_t worker_id, Dtype dtype) const {
  std::vector<std::string> keys;
  for (const Entry& e : ProcessTokens(tokens)) {
    CacheEngineKey key;
    key.model_name = model_name;
    key.world_size = world_size;
    key.worker_id = worker_id;
    key.chunk_hash = e.chunk_hash;
    key.dtype = dtype;
    keys.push_back(key.ToString());
  }
  return keys;
}

}  // namespace vllm::v1::kv_offload::lmcache
