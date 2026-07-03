// Ported from: vllm/csrc/libtorch_stable/cache_kernels.cu @ e24d1b24
//   (reshape_and_cache_flash — the "auto"/contiguous-heads write semantics
//    only; the NHD cache layout is FlashAttentionBackend::get_kv_cache_shape's
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT the HND cpu_attn
//    layout — see the M1.6 Task-2 layout trap note).
// Correctness-grade (M1.6): one block per token, threads stride over the page
// (num_kv_heads*head_size). The perf kernel (vectorized / fp8) is M2.4.
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Word is the raw storage type (uint32_t for f32, uint16_t for f16/bf16): the
// auto cache path is a bit-exact copy, so no dtype-aware conversion is needed.
//
// CRITICAL (M1.6 Task-2): destination is indexed from TENSOR STRIDES, not from
// k_cache.shape. K/V are the two dim-1 unbind slices of one (num_blocks, 2,
// block_size, H, D) allocation, so their block stride is 2*bs*H*D (NOT bs*H*D)
// and they are NON-contiguous rank-4 views. Mirrors pinned cache_kernels.cu @
// e24d1b24 (host ~L797-801 sources key_stride/block_stride/page_stride/
// head_stride from the tensor strides; kernel ~L337-347 does key_dst =
// key_cache + block_idx*block_stride + block_offset*page_stride). The wrapper
// guarantees head_stride == head_size && elem stride == 1 (the NHD unbind
// slice), so the per-token page is one dense run of n_elems inside the block —
// pinned's is_contiguous_heads fast path; here the threads stride that run.
template <typename Word>
__global__ void ReshapeAndCacheKernel(
    const Word* __restrict__ key, const Word* __restrict__ value,
    Word* __restrict__ key_cache, Word* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int64_t block_size, int64_t n_elems,
    int64_t k_block_stride, int64_t k_page_stride, int64_t v_block_stride,
    int64_t v_page_stride, int64_t k_tok_stride, int64_t v_tok_stride) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;  // padded token → skip
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t kdst = block * k_block_stride + offset * k_page_stride;  // element offset
  const int64_t vdst = block * v_block_stride + offset * v_page_stride;
  const int64_t ksrc = token * k_tok_stride;
  const int64_t vsrc = token * v_tok_stride;
  for (int64_t e = threadIdx.x; e < n_elems; e += blockDim.x) {
    key_cache[kdst + e] = key[ksrc + e];
    value_cache[vdst + e] = value[vsrc + e];
  }
}

void ReshapeAndCacheKernelCuda(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];
  if (num_slots == 0 || n_elems == 0) return;
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const unsigned grid = static_cast<unsigned>(num_slots);
  const unsigned block = static_cast<unsigned>(n_elems < 512 ? n_elems : 512);
  const cudaStream_t s = AsStream(q);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  switch (SizeOf(k.dtype)) {
    case 4:
      ReshapeAndCacheKernel<uint32_t><<<grid, block, 0, s>>>(
          k.Ptr<uint32_t>(), v.Ptr<uint32_t>(), k_cache.Ptr<uint32_t>(), v_cache.Ptr<uint32_t>(),
          slots, block_size, n_elems, k_block_stride, k_page_stride, v_block_stride,
          v_page_stride, k_tok_stride, v_tok_stride);
      break;
    case 2:
      ReshapeAndCacheKernel<uint16_t><<<grid, block, 0, s>>>(
          k.Ptr<uint16_t>(), v.Ptr<uint16_t>(), k_cache.Ptr<uint16_t>(), v_cache.Ptr<uint16_t>(),
          slots, block_size, n_elems, k_block_stride, k_page_stride, v_block_stride,
          v_page_stride, k_tok_stride, v_tok_stride);
      break;
    default: VT_CHECK(false, "cuda reshape_and_cache: unsupported dtype element size");
  }
  Check(cudaGetLastError(), "reshape_and_cache launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kReshapeAndCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
