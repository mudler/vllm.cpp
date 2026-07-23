// Ported from: lmcache/utils.py:398-561 (CacheEngineKey) @ LMCache 8570aad.
//
// LMCache MODE-1 (lm://) cache key.  The wire key string is
//   model_name@world_size@worker_id@chunk_hash_hex@dtype[@tag%value...]
// (utils.py:449-457), constrained to <= 150 bytes by the protocol header.
//
// `chunk_hash` is LMCache's OWN token hash (NOT a vLLM block hash).  In the
// lm:// / in-process TokenDatabase path it is stored as an int and folded from
// the blake3 digest via _normalize_hash_to_int (token_database.py:34-56):
// int.from_bytes(digest[:8], "big").  chunk_hash_hex is `f"{chunk_hash:x}"`
// (utils.py:557-561) — lowercase, minimal digits, no "0x".
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_CACHE_ENGINE_KEY_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_CACHE_ENGINE_KEY_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

namespace vllm::v1::kv_offload::lmcache {

struct CacheEngineKey {
  std::string model_name;
  int64_t world_size = 0;
  int64_t worker_id = 0;
  uint64_t chunk_hash = 0;  // folded uint64 (see header note)
  Dtype dtype = Dtype::kNone;
  // Optional lmcache.tag.* entries, in insertion order, as (name, value).
  std::vector<std::pair<std::string, std::string>> tags;

  // utils.py:557-561.
  std::string ChunkHashHex() const;
  // utils.py:449-457.
  std::string ToString() const;
  // utils.py:489-509.
  static CacheEngineKey FromString(std::string_view s);
};

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_CACHE_ENGINE_KEY_H_
