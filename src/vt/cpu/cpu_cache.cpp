// Ported from: vllm/csrc/libtorch_stable/cache_kernels.cu @ e24d1b24
//   (reshape_and_cache_flash — the "auto"/contiguous-heads write semantics
//    only; the NHD cache layout is FlashAttentionBackend::get_kv_cache_shape's
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT the HND cpu_attn
//    layout — see .agents/discipline.md and the M1.6 Task-2 layout trap note).
#include <cstdint>
#include <cstring>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// Write each new per-token K/V into the paged NHD cache at its slot id. The
// "auto" cache path is a raw element copy (cache dtype == k/v dtype); we copy
// bytes so f32/f16/bf16 are all bit-exact (upstream KV_T == CACHE_T in auto).
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;  // one token's page (NHD)
  const int64_t page_stride = n_elems;               // stride over block_size (dim 1)
  const int64_t block_stride = block_size * n_elems; // stride over num_blocks  (dim 0)
  const size_t elem = SizeOf(k.dtype);

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream NOTE: slot can be -1)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t dst = block * block_stride + offset * page_stride;  // element offset
    const int64_t src = t * n_elems;
    const size_t bytes = static_cast<size_t>(n_elems) * elem;
    std::memcpy(kdst + static_cast<size_t>(dst) * elem, ksrc + static_cast<size_t>(src) * elem,
                bytes);
    std::memcpy(vdst + static_cast<size_t>(dst) * elem, vsrc + static_cast<size_t>(src) * elem,
                bytes);
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
