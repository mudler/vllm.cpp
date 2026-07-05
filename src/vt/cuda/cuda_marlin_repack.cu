// vllm.cpp — load-time Marlin NVFP4 repack (torch-free lift of vLLM's
// marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin @ e24d1b24).
//
// Turns the 35B's resident modelopt W4A16 experts (packed fp4 [N,K/2] uint8 +
// fp8-e4m3 block scales [N,K/16] + per-tensor scale2) into Marlin's interleaved
// representation ONCE at load:
//   * weight  -> gptq_marlin_repack (num_bits=4, no perm) : [K/16, N*2] int32
//   * scales  -> marlin_permute_scales + nvfp4_marlin_process_scales (S0E5M3)
//   * global  -> nvfp4_marlin_process_global_scale / combined_scale_factor
//
// The weight kernel is the vendored gptq_marlin_repack_kernel (verbatim). The
// scale processing is reproduced as a device kernel whose index maps are the
// exact torch reshape/permute chain (marlin_permute_scales' 64-wide scale_perm,
// then the within-4 [0,2,1,3], then the int16<<1 / [:,1::2] byte-extract). All
// value math uses hardware __nv_fp8_e4m3 / __half intrinsics (bit-identical to
// torch's .to(dtype)); the multipliers (scale_factor, 128) are powers of two so
// the products are exact. Proven bit-exact vs vLLM in tools/marlin/repack_*.
//
// Gated by VT_MARLIN_NVFP4 (built only on sm_12xa with the vendored slice).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtorch_stable/quantization/marlin/gptq_marlin_repack.cuh"

#include "vt/cuda/marlin_repack.h"

namespace vt::cuda {
namespace {

void RCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("vt cuda: marlin_repack: ") + what + ": " +
                             cudaGetErrorString(err));
}

// modelopt packed [N, K/2] uint8, viewed int32 [N, K/8], transposed+contiguous
// to [K/8, N] int32 — the b_q_weight the repack kernel expects (size_k=K,
// size_n=N, pack_factor=8). One thread per (kp, n).
__global__ void TransposeToInt32Kernel(uint32_t* __restrict__ out,
                                       const uint8_t* __restrict__ packed, int size_k,
                                       int size_n) {
  const int kp = blockIdx.y * blockDim.y + threadIdx.y;  // [0, K/8)
  const int n = blockIdx.x * blockDim.x + threadIdx.x;   // [0, N)
  const int kpacks = size_k / 8;
  if (kp >= kpacks || n >= size_n) return;
  // int32 at (n, kp): 4 bytes packed[n][kp*4 .. kp*4+3] = bytes at n*(K/2)+kp*4.
  const uint8_t* src = packed + static_cast<size_t>(n) * (size_k / 2) + kp * 4;
  uint32_t v = static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) |
               (static_cast<uint32_t>(src[2]) << 16) | (static_cast<uint32_t>(src[3]) << 24);
  out[static_cast<size_t>(kp) * size_n + n] = v;
}

// vLLM get_scale_perms(): 64-wide scale_perm for group_size < size_k.
__constant__ int kScalePerm[64];
static const int kScalePermHost[64] = {
    0,  8,  16, 24, 32, 40, 48, 56, 1,  9,  17, 25, 33, 41, 49, 57,
    2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59,
    4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61,
    6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63};

// One thread per output scale element (r in [0,SK16), c in [0,SN)).
// output[r][c] = highbyte( (half_bits(v) << 1) ), where v is the source fp8
// scale after marlin_permute_scales + within-4 [0,2,1,3], times scale_factor*128
// (clamped to 0 if < 2). scalebuf is our [N, K/16] fp8 block-scale (row-major).
__global__ void ProcessScalesKernel(uint8_t* __restrict__ out,
                                    const uint8_t* __restrict__ scalebuf, int size_k,
                                    int size_n, float scale_factor) {
  const int SN = size_n;
  const int SK16 = size_k / 16;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= SK16 || c >= SN) return;

  const int i = r * SN + c;  // flat index into the [SK16, SN] processed tensor
  // within-4 [0,2,1,3] permute (view(-1,4)[:,[0,2,1,3]])
  const int perm4[4] = {0, 2, 1, 3};
  const int mi = (i / 4) * 4 + perm4[i % 4];  // flat index into M (permute_scales out)
  // marlin_permute_scales inverse: M_flat[idx] = S_T_flat[(idx/64)*64 + scale_perm[idx%64]]
  const int j = (mi / 64) * 64 + kScalePerm[mi % 64];  // flat index into S_T [SK16, SN]
  // S_T_flat[j] = scale[n][kr] with kr=j/SN, n=j%SN → scalebuf[n*SK16 + kr]
  const int kr = j / SN;
  const int n = j % SN;
  const uint8_t fp8b = scalebuf[static_cast<size_t>(n) * SK16 + kr];

  // decode fp8-e4m3 -> float (exact), *scale_factor*128 (exact powers of 2)
  __nv_fp8_e4m3 fp8v;
  fp8v.__x = fp8b;
  float v = static_cast<float>(fp8v) * scale_factor * 128.0f;

  uint8_t obyte;
  if (!(v >= 2.0f)) {
    obyte = 0;  // marlin_scales[marlin_scales < 2] = 0  (NaN handled: !>= is false-safe)
  } else {
    __half h = __float2half_rn(v);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    bits = static_cast<uint16_t>(bits << 1);  // view(int16) << 1
    obyte = static_cast<uint8_t>(bits >> 8);  // reinterpret fp8, take [:,1::2] (high byte)
  }
  out[static_cast<size_t>(r) * SN + c] = obyte;
}

// fp8-e4m3 byte -> float (host, for the combined_scale_factor max reduce).
float DecodeFp8E4m3(uint8_t b) {
  const int sign = (b >> 7) & 1;
  const int exp = (b >> 3) & 0xF;
  const int mant = b & 0x7;
  float val;
  if (exp == 0) {
    val = static_cast<float>(mant) * std::ldexp(1.0f, -9);  // 2^(1-7) / 8
  } else if (exp == 0xF && mant == 0x7) {
    val = 0.0f;  // e4m3fn NaN — treat as 0 (won't occur in weight scales)
  } else {
    val = (1.0f + static_cast<float>(mant) / 8.0f) * std::ldexp(1.0f, exp - 7);
  }
  return sign ? -val : val;
}

}  // namespace

void MarlinRepackExpertWeight(void* stream, int device, uint32_t* out_weight,
                              const uint8_t* packed_nk2, int size_k, int size_n) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int kpacks = size_k / 8;
  // int32-transpose to [K/8, N]
  uint32_t* bqt = nullptr;
  RCheck(cudaMallocAsync(&bqt, static_cast<size_t>(kpacks) * size_n * sizeof(uint32_t), s),
         "malloc bqt");
  dim3 tb(32, 8);
  dim3 tg((size_n + tb.x - 1) / tb.x, (kpacks + tb.y - 1) / tb.y);
  TransposeToInt32Kernel<<<tg, tb, 0, s>>>(bqt, packed_nk2, size_k, size_n);
  RCheck(cudaGetLastError(), "transpose launch");

  // repack (num_bits=4, has_perm=false, is_a_8bit=false)
  int blocks = 0;
  RCheck(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device), "sms");
  int max_shared_mem = 0;
  RCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
         "smem");
  auto kern = MARLIN_NAMESPACE_NAME::gptq_marlin_repack_kernel<MARLIN_NAMESPACE_NAME::repack_threads,
                                                               4, false, false>;
  RCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem),
         "func attr");
  uint32_t* empty_perm = nullptr;
  kern<<<blocks, MARLIN_NAMESPACE_NAME::repack_threads, max_shared_mem, s>>>(
      bqt, empty_perm, out_weight, size_k, size_n);
  RCheck(cudaGetLastError(), "repack launch");
  RCheck(cudaFreeAsync(bqt, s), "free bqt");
}

void MarlinProcessExpertScales(void* stream, const uint8_t* scale_nk16, uint8_t* out_scale,
                               int size_k, int size_n, float scale_factor) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  static bool perm_uploaded = false;
  if (!perm_uploaded) {
    RCheck(cudaMemcpyToSymbol(kScalePerm, kScalePermHost, sizeof(kScalePermHost)),
           "upload scale_perm");
    perm_uploaded = true;
  }
  const int SN = size_n;
  const int SK16 = size_k / 16;
  dim3 tb(32, 8);
  dim3 tg((SN + tb.x - 1) / tb.x, (SK16 + tb.y - 1) / tb.y);
  ProcessScalesKernel<<<tg, tb, 0, s>>>(out_scale, scale_nk16, size_k, size_n, scale_factor);
  RCheck(cudaGetLastError(), "process scales launch");
}

float MarlinNvfp4CombinedScaleFactor(const std::vector<const uint8_t*>& host_scale_bufs,
                                     const std::vector<size_t>& lens) {
  // _nvfp4_compute_scale_factor: ws = max(scale)*2^7; if ws < 448*2^7:
  //   sf = 2^floor(log2(448*2^7 / ws)); else 1.  (bf16 activation path.)
  float maxscale = 0.0f;
  for (size_t t = 0; t < host_scale_bufs.size(); ++t) {
    const uint8_t* buf = host_scale_bufs[t];
    for (size_t i = 0; i < lens[t]; ++i) {
      float v = DecodeFp8E4m3(buf[i]);
      if (v > maxscale) maxscale = v;
    }
  }
  const float ws = maxscale * 128.0f;
  const float limit = 448.0f * 128.0f;
  if (maxscale > 0.0f && ws < limit) {
    const float sf = std::exp2(std::floor(std::log2(limit / ws)));
    return sf;
  }
  return 1.0f;
}

float MarlinNvfp4ProcessGlobalScale(float scale2, float combined_sf) {
  // nvfp4_marlin_process_global_scale (bf16): g * 2^(126-7) = g * 2^119, /sf.
  return scale2 * std::exp2(119.0f) / combined_sf;
}

namespace {
// Torch-free moe_align_block_size (mirror of csrc/moe/moe_align_sum_kernels.cu
// semantics; the ordering within an expert is irrelevant to Marlin since each
// sorted row is gathered independently and padding rows are masked). Single
// block; dynamic shared holds counts[E], block-start[E], fill-counter[E]. Fills
// sorted_ids with the pair-flat index (t*top_k+k), padding-slots = numel.
__global__ void MoeAlignKernel(const int32_t* __restrict__ topk_ids, int numel, int top_k,
                               int num_experts, int block_size, int32_t* __restrict__ sorted_ids,
                               int32_t* __restrict__ expert_ids,
                               int32_t* __restrict__ num_tokens_post_pad,
                               int max_num_tokens_padded, int max_num_blocks) {
  extern __shared__ int sh_align[];
  int* counts = sh_align;                    // [E]
  int* bstart = sh_align + num_experts;      // [E]
  int* fillc = sh_align + 2 * num_experts;   // [E]
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  for (int i = tid; i < max_num_tokens_padded; i += nthreads) sorted_ids[i] = numel;
  for (int i = tid; i < max_num_blocks; i += nthreads) expert_ids[i] = 0;
  for (int e = tid; e < num_experts; e += nthreads) counts[e] = 0;
  __syncthreads();

  for (int p = tid; p < numel; p += nthreads) {
    int e = topk_ids[p];
    if (e >= 0 && e < num_experts) atomicAdd(&counts[e], 1);
  }
  __syncthreads();

  if (tid == 0) {
    int off = 0, blk = 0;
    for (int e = 0; e < num_experts; ++e) {
      bstart[e] = off;
      fillc[e] = off;
      int nb = (counts[e] + block_size - 1) / block_size;
      for (int b = 0; b < nb; ++b) expert_ids[blk++] = e;
      off += nb * block_size;
    }
    num_tokens_post_pad[0] = off;
  }
  __syncthreads();

  for (int p = tid; p < numel; p += nthreads) {
    int e = topk_ids[p];
    if (e >= 0 && e < num_experts) {
      int pos = atomicAdd(&fillc[e], 1);
      sorted_ids[pos] = p;
    }
  }
}
}  // namespace

int MarlinDeviceSms(int device) {
  int sms = 0;
  RCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "sms");
  return sms;
}

int MarlinMoeAlignBlockSizeSelect(int num_tokens, int top_k, int num_experts) {
  // vLLM marlin_moe.py block_size_m selection (bf16 path), clamped to >=16 to
  // stay off the m_block_size_8 special kernel.
  int bs = 8;
  for (int cand : {8, 16, 32, 48, 64}) {
    bs = cand;
    if (static_cast<double>(num_tokens) * top_k / num_experts / cand < 0.9) break;
  }
  return bs < 16 ? 16 : bs;
}

void MarlinMoeAlignSizes(int num_tokens, int top_k, int num_experts, int block_size,
                         int* max_num_tokens_padded, int* max_num_blocks) {
  const int numel = num_tokens * top_k;
  int m = numel + num_experts * (block_size - 1);
  if (numel < num_experts) {
    int clamp = numel * block_size;
    if (clamp < m) m = clamp;
  }
  *max_num_tokens_padded = m;
  *max_num_blocks = (m + block_size - 1) / block_size;
}

void MarlinMoeAlignBlockSize(void* stream, const int32_t* topk_ids, int num_tokens, int top_k,
                             int num_experts, int block_size, int32_t* sorted_ids,
                             int32_t* expert_ids, int32_t* num_tokens_post_pad) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  int max_tok = 0, max_blk = 0;
  MarlinMoeAlignSizes(num_tokens, top_k, num_experts, block_size, &max_tok, &max_blk);
  const int numel = num_tokens * top_k;
  const size_t shmem = static_cast<size_t>(3) * num_experts * sizeof(int);
  MoeAlignKernel<<<1, 256, shmem, s>>>(topk_ids, numel, top_k, num_experts, block_size, sorted_ids,
                                       expert_ids, num_tokens_post_pad, max_tok, max_blk);
  RCheck(cudaGetLastError(), "moe_align launch");
}

}  // namespace vt::cuda
