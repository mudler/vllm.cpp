// Ported from: lmcache/v1/multiprocess/token_hasher.py:28-240 @ LMCache 8570aad.
//
// Uses the vendored official BLAKE3 C implementation (third_party/blake3,
// upstream tag 1.5.5 / commit 81f772a), which is the same algorithm the
// `blake3` PyPI package binds.
#include "vllm/v1/kv_offload/lmcache/token_hasher.h"

#include <array>
#include <cstddef>

#include <blake3/blake3.h>

namespace vllm::v1::kv_offload::lmcache {
namespace {

// struct.pack(">{N}I", *tokens): each token as a big-endian uint32.
std::string EncodeTokensBigEndian(const std::vector<uint32_t>& tokens) {
  std::string out;
  out.reserve(tokens.size() * 4);
  for (uint32_t t : tokens) {
    out.push_back(static_cast<char>((t >> 24) & 0xFF));
    out.push_back(static_cast<char>((t >> 16) & 0xFF));
    out.push_back(static_cast<char>((t >> 8) & 0xFF));
    out.push_back(static_cast<char>(t & 0xFF));
  }
  return out;
}

// blake3(prefix_bytes || tokens_big_endian) -> 32-byte digest
// (token_hasher.py:37-49).  `prefix_bytes` is fed raw: either a prior 32-byte
// digest (rolling) or the 8-byte big-endian-signed encoding of the initial
// prefix int (none_hash bootstrap).
std::string Blake3Digest(const std::string& prefix_bytes,
                         const std::vector<uint32_t>& tokens) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, prefix_bytes.data(), prefix_bytes.size());
  const std::string tok = EncodeTokensBigEndian(tokens);
  blake3_hasher_update(&hasher, tok.data(), tok.size());
  std::array<uint8_t, BLAKE3_OUT_LEN> out{};
  blake3_hasher_finalize(&hasher, out.data(), out.size());
  return std::string(reinterpret_cast<const char*>(out.data()), out.size());
}

}  // namespace

TokenHasher::TokenHasher(int chunk_size) : chunk_size_(chunk_size) {
  // none_hash = blake3_hash((0, (0,), None)) (token_hasher.py:179):
  // prefix int 0 -> to_bytes(8, "big", signed=True) = eight 0x00 bytes;
  // tokens (0,) -> pack(">1I", 0) = four 0x00 bytes.
  const std::string prefix_zero_int(8, '\0');
  none_hash_ = Blake3Digest(prefix_zero_int, std::vector<uint32_t>{0});
}

std::string TokenHasher::HashTokens(const std::vector<uint32_t>& tokens,
                                    const std::string& prefix) const {
  return Blake3Digest(prefix, tokens);
}

std::string TokenHasher::HashTokens(
    const std::vector<uint32_t>& tokens) const {
  return Blake3Digest(none_hash_, tokens);
}

std::vector<std::string> TokenHasher::ComputeChunkHashes(
    const std::vector<uint32_t>& token_ids) const {
  // token_hasher.py:220-230.
  std::vector<std::string> hashes;
  std::string prefix = none_hash_;
  const std::size_t n = token_ids.size();
  const auto cs = static_cast<std::size_t>(chunk_size_);
  const std::size_t num_complete = n - (n % cs);
  for (std::size_t i = 0; i < num_complete; i += cs) {
    std::vector<uint32_t> chunk(token_ids.begin() + static_cast<long>(i),
                                token_ids.begin() + static_cast<long>(i + cs));
    prefix = Blake3Digest(prefix, chunk);
    hashes.push_back(prefix);
  }
  return hashes;
}

std::string TokenHasher::ToHex(const std::string& digest) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(digest.size() * 2);
  for (unsigned char c : digest) {
    out.push_back(kHex[c >> 4]);
    out.push_back(kHex[c & 0xF]);
  }
  return out;
}

uint64_t TokenHasher::FoldToUint64(const std::string& digest) {
  // int.from_bytes(digest[:8], "big").
  uint64_t v = 0;
  const std::size_t n = digest.size() < 8 ? digest.size() : 8;
  for (std::size_t i = 0; i < n; ++i) {
    v = (v << 8) | static_cast<uint8_t>(digest[i]);
  }
  return v;
}

}  // namespace vllm::v1::kv_offload::lmcache
