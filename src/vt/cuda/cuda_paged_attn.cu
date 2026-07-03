// Ported from: vllm/v1/attention/backends/flash_attn.py @ e24d1b24
//   (FlashAttentionImpl.forward SEMANTICS: causal GQA softmax over the paged K/V,
//    softmax_scale = self.scale, cu_seqlens_q = query_start_loc, seqused_k =
//    seq_lens, block_table = block_table_tensor). Cache READ is the NHD layout
//    FlashAttentionBackend::get_kv_cache_shape allocates, indexed by TENSOR
//    STRIDES (the two dim-1 unbind slices; block stride 2*bs*H*D) — NOT cpu_attn's
//    HND arithmetic (M1.6 Task-3 layout trap).
//
// Correctness-grade (M1.6): one block per (query token, q-head); block threads
// cooperate over head_size and stream the keys with an online (flash-style)
// softmax — algebraically identical to the CPU two-pass reference, f32
// accumulation. The perf kernel (FlashInfer-class) is M2.4. This is NOT the
// paged fp8/vectorized kernel — it reads the "auto" cache (cache dtype == q).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kPagedBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// One block per (query token t = blockIdx.x, q-head h = blockIdx.y). The block
// scans query_start_loc to find token t's request r, derives the absolute
// position p, and streams keys 0..p (causal) from the paged cache.
template <typename Tin, typename Tout>
__global__ void PagedAttentionKernel(Tout* out, const Tin* query, const Tin* k_cache,
                                     const Tin* v_cache, const int32_t* block_table,
                                     const int32_t* seq_lens, const int32_t* query_start_loc,
                                     int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d,
                                     int64_t block_size, int64_t bt_row, int64_t bt_col,
                                     int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                     int64_t vc_pg, int64_t vc_hd, float scale, bool causal) {
  const int64_t t = blockIdx.x;  // global query-token index
  const int64_t h = blockIdx.y;  // q-head
  // Find request r with query_start_loc[r] <= t < query_start_loc[r+1].
  int64_t r = -1, q0 = 0, q1 = 0;
  for (int64_t rr = 0; rr < num_reqs; ++rr) {
    const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
    if (t >= a && t < b) {
      r = rr;
      q0 = a;
      q1 = b;
      break;
    }
  }
  if (r < 0) return;  // padding token beyond the last request

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);
  const int64_t jmax = causal ? p : seqlen - 1;
  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;             // [d] running output accumulator
  float* red = smem + d;         // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t j = 0; j <= jmax; ++j) {
    const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
    const int64_t off = j % block_size;
    const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(k_cache, kbase + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);  // 0 on the first key (s_m == -inf)
    const float pw = expf(s - m_new);
    const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + pw * Load(v_cache, vbase + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + pw;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

template <typename Tin>
void LaunchPaged(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                 const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                 const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  const int64_t num_tokens = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t num_kv_heads = k_cache.shape[2], block_size = k_cache.shape[1];
  if (num_tokens == 0 || hq == 0 || d == 0) return;
  const dim3 grid(static_cast<unsigned>(num_tokens), static_cast<unsigned>(hq));
  const size_t shmem = (static_cast<size_t>(d) + kPagedBlock) * sizeof(float);
  const int32_t* bt = block_table.Ptr<int32_t>();
  const int32_t* sl = seq_lens.Ptr<int32_t>();
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  switch (out.dtype) {
    case DType::kF32:
      PagedAttentionKernel<Tin, float><<<grid, kPagedBlock, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), k_cache.Ptr<Tin>(), v_cache.Ptr<Tin>(), bt, sl, qsl,
          num_reqs, hq, num_kv_heads, d, block_size, block_table.stride[0], block_table.stride[1],
          k_cache.stride[0], k_cache.stride[1], k_cache.stride[2], v_cache.stride[0],
          v_cache.stride[1], v_cache.stride[2], args.scale, args.causal);
      break;
    case DType::kBF16:
      PagedAttentionKernel<Tin, __nv_bfloat16><<<grid, kPagedBlock, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), k_cache.Ptr<Tin>(), v_cache.Ptr<Tin>(), bt,
          sl, qsl, num_reqs, hq, num_kv_heads, d, block_size, block_table.stride[0],
          block_table.stride[1], k_cache.stride[0], k_cache.stride[1], k_cache.stride[2],
          v_cache.stride[0], v_cache.stride[1], v_cache.stride[2], args.scale, args.causal);
      break;
    default: VT_CHECK(false, "cuda paged_attention: unsupported out dtype");
  }
  Check(cudaGetLastError(), "paged_attention launch");
}

void PagedAttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchPaged<float>(AsStream(q), out, query, k_cache, v_cache, block_table, seq_lens,
                         query_start_loc, args);
      break;
    case DType::kBF16:
      LaunchPaged<__nv_bfloat16>(AsStream(q), out, query, k_cache, v_cache, block_table, seq_lens,
                                 query_start_loc, args);
      break;
    default: VT_CHECK(false, "cuda paged_attention: unsupported input dtype (f32/bf16 only)");
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kPagedAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
