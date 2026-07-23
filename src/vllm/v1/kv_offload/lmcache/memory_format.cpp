// Ported from: lmcache/v1/memory_management.py:79-133 @ LMCache 8570aad.
#include "vllm/v1/kv_offload/lmcache/memory_format.h"

#include <cstring>
#include <stdexcept>

namespace vllm::v1::kv_offload::lmcache {

int TokenDim(MemoryFormat fmt) {
  // memory_management.py:116-133.
  switch (fmt) {
    case MemoryFormat::kKV2LTD:
      return 2;
    case MemoryFormat::kKVT2D:
      return 1;
    case MemoryFormat::kKV2TD:
      return 0;
    case MemoryFormat::kBinary:
      return 0;
    case MemoryFormat::kBinaryBuffer:
      return 0;
    case MemoryFormat::kKVMLAFmt:
      return 2;
    case MemoryFormat::kECTD:
      return 0;
    case MemoryFormat::kHSTD:
      return 0;
    case MemoryFormat::kUndefined:
      return 0;
  }
  return 0;
}

std::size_t Kv2ltdLayout::ByteOffset(int kv, int layer, int token,
                                     int dim) const {
  const std::size_t idx =
      (static_cast<std::size_t>(kv) * static_cast<std::size_t>(num_layers) +
       static_cast<std::size_t>(layer)) *
          static_cast<std::size_t>(num_tokens) *
          static_cast<std::size_t>(hidden_dim) +
      static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden_dim) +
      static_cast<std::size_t>(dim);
  return idx * elem_size;
}

std::string PackKv2ltd(const Kv2ltdLayout& layout,
                       const std::vector<std::string>& k_planes,
                       const std::vector<std::string>& v_planes) {
  const auto num_layers = static_cast<std::size_t>(layout.num_layers);
  if (k_planes.size() != num_layers || v_planes.size() != num_layers) {
    throw std::invalid_argument(
        "PackKv2ltd: plane count does not match num_layers");
  }
  const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;
  std::string out;
  out.resize(layout.NumBytes());
  char* dst = out.data();
  // kv=0: all K layers, then kv=1: all V layers — matches [2, L, T, D].
  std::size_t off = 0;
  for (std::size_t l = 0; l < num_layers; ++l) {
    if (k_planes[l].size() != plane_bytes) {
      throw std::invalid_argument("PackKv2ltd: K plane wrong size");
    }
    std::memcpy(dst + off, k_planes[l].data(), plane_bytes);
    off += plane_bytes;
  }
  for (std::size_t l = 0; l < num_layers; ++l) {
    if (v_planes[l].size() != plane_bytes) {
      throw std::invalid_argument("PackKv2ltd: V plane wrong size");
    }
    std::memcpy(dst + off, v_planes[l].data(), plane_bytes);
    off += plane_bytes;
  }
  return out;
}

void UnpackKv2ltd(const Kv2ltdLayout& layout, const std::string& packed,
                  std::vector<std::string>* k_planes,
                  std::vector<std::string>* v_planes) {
  if (packed.size() != layout.NumBytes()) {
    throw std::invalid_argument("UnpackKv2ltd: buffer size mismatch");
  }
  const auto num_layers = static_cast<std::size_t>(layout.num_layers);
  const std::size_t plane_bytes = layout.LayerStride() * layout.elem_size;
  k_planes->clear();
  v_planes->clear();
  k_planes->reserve(num_layers);
  v_planes->reserve(num_layers);
  const char* src = packed.data();
  std::size_t off = 0;
  for (std::size_t l = 0; l < num_layers; ++l) {
    k_planes->emplace_back(src + off, plane_bytes);
    off += plane_bytes;
  }
  for (std::size_t l = 0; l < num_layers; ++l) {
    v_planes->emplace_back(src + off, plane_bytes);
    off += plane_bytes;
  }
}

}  // namespace vllm::v1::kv_offload::lmcache
