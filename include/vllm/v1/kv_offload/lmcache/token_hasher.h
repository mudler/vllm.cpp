// Ported from: lmcache/v1/multiprocess/token_hasher.py:28-240 @ LMCache 8570aad.
//
// LMCache's rolling prefix token hash (blake3, chunk_size 256).  This is the
// hash LMCache keys chunks on; a single-bit divergence = zero cache hits, so it
// must be BYTE-EXACT against the Python `blake3` package.
//
// The hash of one chunk (token_hasher.py:37-49):
//     h = blake3()
//     h.update(prefix_bytes)                 # 32-byte digest, or the initial
//                                            # prefix int 0 -> 8 big-endian
//                                            # signed bytes
//     h.update(struct.pack(">{N}I", tokens)) # tokens as big-endian uint32
//     return h.digest()                      # 32 bytes
//
// none_hash = blake3_hash((0, (0,), None)) (token_hasher.py:179) — i.e. the
// digest of 8 zero bytes (int 0) followed by one big-endian uint32 0, hence
// blake3 of twelve zero bytes.
//
// Rolling chunk hashes (token_hasher.py:192-230): prefix starts at none_hash;
// for each COMPLETE chunk of chunk_size tokens, prefix = hash(chunk, prefix)
// and the digest is emitted.  A trailing partial chunk is discarded.
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_TOKEN_HASHER_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_TOKEN_HASHER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace vllm::v1::kv_offload::lmcache {

class TokenHasher {
 public:
  explicit TokenHasher(int chunk_size = 256);

  int chunk_size() const { return chunk_size_; }

  // The 32-byte none_hash digest (token_hasher.py:179).
  const std::string& none_hash() const { return none_hash_; }

  // Hash one chunk of tokens with a rolling `prefix` (a 32-byte digest, or
  // none_hash for the first chunk).  Returns a 32-byte digest.
  std::string HashTokens(const std::vector<uint32_t>& tokens,
                         const std::string& prefix) const;
  // Same, using none_hash as the prefix.
  std::string HashTokens(const std::vector<uint32_t>& tokens) const;

  // Rolling prefix hashes for every COMPLETE chunk in `token_ids`
  // (token_hasher.py:192-230).  Each element is a 32-byte digest.
  std::vector<std::string> ComputeChunkHashes(
      const std::vector<uint32_t>& token_ids) const;

  // Lowercase-hex of a digest (matches Python bytes.hex()).
  static std::string ToHex(const std::string& digest);

  // Fold a 32-byte digest to the uint64 chunk_hash LMCache stores in
  // CacheEngineKey (token_database.py:34-56 _normalize_hash_to_int):
  // int.from_bytes(digest[:8], "big").
  static uint64_t FoldToUint64(const std::string& digest);

 private:
  int chunk_size_;
  std::string none_hash_;
};

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_TOKEN_HASHER_H_
