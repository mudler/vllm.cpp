// Ported from: lmcache/v1/token_database.py:298-449 (ChunkedTokenDatabase) and
//              lmcache/v1/token_database.py:34-56,234-295 (the hash chain) @
//              LMCache 8570aad.
//
// THE PEER-AGREEING KV KEY DERIVATION for the lm:// remote store (MODE 1) and the
// in-process TokenDatabase path. This is the hash a REAL vLLM+LMCache peer keys
// chunks on, and it is NOT the blake3 MP `TokenHasher` (token_hasher.h, a
// different subsystem — the ZMQ MP server). A single divergent bit = zero cache
// hits against the peer, so this must be BYTE-EXACT.
//
// The derivation (token_database.py:358-431, _hash_tokens :269-295):
//   chunk_size            := config.chunk_size (LMCache default 256)
//   hash_func             := vLLM's own hash (pre_caching_hash_algorithm);
//                            the portable interop choice is `sha256_cbor`
//                            (vllm/utils/hashing.py:43 — cbor2 canonical + SHA-256)
//   NONE_HASH (int)       := _normalize_hash_to_int(init_none_hash(hash_func))
//                            = fold8( sha256_cbor(str(PYTHONHASHSEED)) )
//                            (kv_cache_utils.py:99-114 + token_database.py:34-56,99)
//   prefix := NONE_HASH
//   for each chunk of `chunk_size` tokens (a trailing partial chunk is kept iff
//   save_unfull_chunk, token_database.py:334-356):
//     prefix := fold8( hash_func( (prefix, tuple(chunk_tokens), ()) ) )
//     emit (start, end, chunk_hash = prefix)
//   where fold8(bytes) = int.from_bytes(bytes[:8], "big") (token_database.py:54-55)
//   and the CBOR "Any" input is the 3-tuple (prefix:int, tokens:tuple[int], ()).
//
// The CacheEngineKey per chunk is
//   model_name@world_size@worker_id@f"{chunk_hash:x}"@dtype  (utils.py:449-457),
// built by CacheEngineKey (cache_engine_key.h). extra_keys is empty for text-only
// requests (token_database.py:282-284); multimodal mm-hash injection would add
// non-empty extra_keys, out of scope for this text-only increment (see the spec).
//
// We reuse the project's byte-exact CBOR + SHA-256 (vllm/v1/core/kv_cache_utils.h:
// CborValue + sha256_cbor), the same primitives that already match Python's
// cbor2 + hashlib for vLLM's own block hashes.
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_CHUNKED_TOKEN_DATABASE_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_CHUNKED_TOKEN_DATABASE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/v1/kv_offload/lmcache/cache_engine_key.h"
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

namespace vllm::v1::kv_offload::lmcache {

// The chunk hash function a peer keys on. Only sha256_cbor is byte-exact-ported
// (it is the documented portable/interop hash). The default "builtin" (CPython
// hash) and xxhash_cbor are enumerated for parity but not implemented here (the
// former is not cleanly reproducible across implementations; see the spec).
enum class PreCachingHash {
  kSha256Cbor,
};

// Mirrors ChunkedTokenDatabase (token_database.py:298-449). Text-only: no mask,
// no multimodal extra_keys. Pure function of (tokens, chunk_size, none_hash,
// save_unfull_chunk); needs no GPU, no server, no torch.
class ChunkedTokenDatabase {
 public:
  struct Entry {
    int start = 0;          // start token index (inclusive)
    int end = 0;            // end token index (exclusive)
    uint64_t chunk_hash = 0;  // the folded uint64 CacheEngineKey.chunk_hash
  };

  // `none_hash` is the folded uint64 NONE_HASH (use NoneHashFromSeed). Default
  // chunk_size 256 and save_unfull_chunk match the LMCache config defaults
  // (config.py:90 chunk_size=256; save_unfull_chunk default False).
  explicit ChunkedTokenDatabase(int chunk_size = 256, uint64_t none_hash = 0,
                                bool save_unfull_chunk = false,
                                PreCachingHash hash = PreCachingHash::kSha256Cbor);

  int chunk_size() const { return chunk_size_; }
  uint64_t none_hash() const { return none_hash_; }
  bool save_unfull_chunk() const { return save_unfull_chunk_; }

  // fold8( sha256_cbor(str(seed)) ) — the reproducible NONE_HASH for a given
  // PYTHONHASHSEED string (kv_cache_utils.py:113-114 + token_database.py:99).
  // The interop peer uses seed "0".
  static uint64_t NoneHashFromSeed(const std::string& seed);

  // hash one chunk with a rolling `prefix`: fold8(hash_func((prefix, tokens, ())))
  // (token_database.py:269-295). `prefix` is the folded uint64 of the previous
  // chunk (or none_hash for the first).
  uint64_t HashTokens(const std::vector<int32_t>& chunk_tokens,
                      uint64_t prefix) const;

  // The full chunked prefix-hash walk (token_database.py:414-431). Each Entry's
  // chunk_hash is the CacheEngineKey.chunk_hash for [start, end).
  std::vector<Entry> ProcessTokens(const std::vector<int32_t>& tokens) const;

  // Build the CacheEngineKey string per chunk (utils.py:449-457), given the
  // identity fields. Convenience over ProcessTokens + CacheEngineKey.
  std::vector<std::string> ProcessKeys(const std::vector<int32_t>& tokens,
                                       const std::string& model_name,
                                       int64_t world_size, int64_t worker_id,
                                       Dtype dtype) const;

 private:
  int chunk_size_;
  uint64_t none_hash_;
  bool save_unfull_chunk_;
  PreCachingHash hash_;
};

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_CHUNKED_TOKEN_DATABASE_H_
