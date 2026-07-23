// Ported from: lmcache/v1/memory_management.py:79-133 (MemoryFormat) @ LMCache
//              8570aad.
//
// LMCache MODE-1 (lm:// remote store) — KV chunk memory formats.  This W1
// deliverable covers KV_2LTD, the [2, num_layers, num_tokens, hidden_dim]
// contiguous row-major chunk layout the remote-store server expects
// (memory_management.py:81-84; the docstring for that shape sits one member
// above KV_2LTD in the upstream Enum, an upstream comment-placement quirk we do
// NOT replicate — the SHAPE is what binds, confirmed by
// MemoryFormat.token_dim()==2 for KV_2LTD, memory_management.py:116-118).
//
// The enum integer VALUES are on the wire (protocol.py header `fmt` field), so
// they must match exactly: UNDEFINED=0 then auto() from 1.
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_MEMORY_FORMAT_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_MEMORY_FORMAT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vllm::v1::kv_offload::lmcache {

// Mirrors lmcache/v1/memory_management.py:79-114.  Values are wire-visible.
enum class MemoryFormat : int32_t {
  kUndefined = 0,
  kKV2LTD = 1,    // [2, num_layers, num_tokens, hidden_dim]
  kKVT2D = 2,     // [num_tokens, 2, hidden_dim]
  kKV2TD = 3,     // [2, num_tokens, hidden_dim]
  kBinary = 4,
  kBinaryBuffer = 5,
  kKVMLAFmt = 6,  // [1, num_layers, num_tokens, aligned_head_size]
  kECTD = 7,      // [num_tokens, hidden_dim]
  kHSTD = 8,      // [num_tokens, hidden_dim]
};

// Mirrors MemoryFormat.token_dim() (memory_management.py:116-133): the axis
// index that indexes tokens, used by LMCache to slice chunks.
int TokenDim(MemoryFormat fmt);

// Descriptor for a KV_2LTD chunk = [2, num_layers, num_tokens, hidden_dim],
// row-major C-contiguous.  Element (kv, l, t, d), with kv in {0=K, 1=V}, lives
// at flat element index ((kv * L + l) * T + t) * D + d.  The whole raison
// d'etre of this helper is to get that axis order and those strides byte-exact.
struct Kv2ltdLayout {
  int num_layers = 0;
  int num_tokens = 0;
  int hidden_dim = 0;
  std::size_t elem_size = 0;  // bytes per element (dtype width)

  std::size_t NumElements() const {
    return static_cast<std::size_t>(2) * static_cast<std::size_t>(num_layers) *
           static_cast<std::size_t>(num_tokens) *
           static_cast<std::size_t>(hidden_dim);
  }
  std::size_t NumBytes() const { return NumElements() * elem_size; }

  // Element strides (in elements) for axes [kv, layer, token, dim].
  std::size_t KvStride() const {
    return static_cast<std::size_t>(num_layers) *
           static_cast<std::size_t>(num_tokens) *
           static_cast<std::size_t>(hidden_dim);
  }
  std::size_t LayerStride() const {
    return static_cast<std::size_t>(num_tokens) *
           static_cast<std::size_t>(hidden_dim);
  }
  std::size_t TokenStride() const {
    return static_cast<std::size_t>(hidden_dim);
  }

  // Byte offset of element (kv, layer, token, dim).
  std::size_t ByteOffset(int kv, int layer, int token, int dim) const;

  // The 4-D shape as it goes on the protocol header (already 4-D, no padding).
  std::vector<int32_t> Shape() const {
    return {2, num_layers, num_tokens, hidden_dim};
  }
};

// Repack per-layer K and V planes into a KV_2LTD contiguous byte buffer.
//
// `k_planes[l]` / `v_planes[l]` are each the contiguous bytes of a
// [num_tokens, hidden_dim] plane for layer `l` (row-major, `elem_size` bytes
// per element).  The output is the [2, L, T, D] chunk: all K layers first
// (kv=0), then all V layers (kv=1).  This is pure host-side memory reshaping;
// the device->host copy that fills the planes is W2.
std::string PackKv2ltd(const Kv2ltdLayout& layout,
                       const std::vector<std::string>& k_planes,
                       const std::vector<std::string>& v_planes);

// Inverse of PackKv2ltd: split a KV_2LTD buffer back into per-layer K/V planes.
void UnpackKv2ltd(const Kv2ltdLayout& layout, const std::string& packed,
                  std::vector<std::string>* k_planes,
                  std::vector<std::string>* v_planes);

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_MEMORY_FORMAT_H_
