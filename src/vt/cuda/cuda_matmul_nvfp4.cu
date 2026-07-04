// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// NVFP4 W4A16 dequant-GEMM for the CUDA backend (M2.2a, correctness-grade).
//
// out[M,N] = act[M,K] (f32/bf16) @ dequant(w).T, where w is a modelopt
// W4A16_NVFP4 weight [N=out_features, K=in_features] read DIRECTLY from device
// memory and dequantized on the fly (no host bf16 weight tensor). The decode is
// bit-for-bit vllm::DequantNvfp4ToBf16 (the authoritative reference):
//   * two 4-bit E2M1 codes per packed byte, low nibble = input elem 2i, high
//     nibble = 2i+1; nibble bit 3 = sign, bits 0..2 index the magnitude LUT
//     {0, .5, 1, 1.5, 2, 3, 4, 6};
//   * one IEEE fp8-e4m3fn scale per 16-element input group;
//   * group scale = f32(fp8_scale) * weight_scale_2 (per-tensor, multiplied);
//   * the dequanted value is ROUNDED TO BF16 before use — exactly the byte
//     DequantNvfp4ToBf16 stores — so this GEMM and Matmul(act, dequant(w).T)
//     differ only in K-reduction order, not in the per-element product.
// These are standard IEEE fp8-e4m3fn scales: the GGUF killgate fork's UE4M3
// x0.5 LUT trap (.agents/gguf-nvfp4-notes.md) does NOT apply to modelopt NVFP4.
//
// Correctness-grade layout: one thread per output element (grid.x over N tiles,
// grid.y over M rows), f32 accumulation. MMA / shared-memory tiling is the
// M2.2a perf follow-up — this delivers + validates the on-device fp4 decode.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_nvfp4: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// IEEE fp8-e4m3fn -> f32, mirroring vllm::F8E4M3ToF32 (1 sign, 4 exp, 3 mant,
// bias 7, no inf, NaN only at 0x7F/0xFF).
__device__ inline float F8E4M3ToF32Dev(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1u;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFu;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7u;
  const float sign_mul = sign ? -1.0f : 1.0f;
  if (exp == 0xFu && mant == 0x7u) return CUDART_NAN_F;
  if (exp == 0u) return sign_mul * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sign_mul * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// out[m,n] = sum_k act[m,k] * bf16(e2m1(w[n,k]) * group_scale[n,k/16]).
template <typename Tin, typename Tout>
__global__ void MatmulNvfp4Kernel(Tout* out, const Tin* act, const uint8_t* packed,
                                  const uint8_t* scale, float scale2, int64_t m_rows,
                                  int64_t n_cols, int64_t k_dim) {
  // E2M1 (fp4) magnitude LUT, indexed by the low 3 bits of the nibble; bit 3 is
  // the sign. 1x-scaled — modelopt NVFP4, NOT the 2x GGUF kvalues_mxfp4 LUT.
  const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;

  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* prow = packed + n * packed_cols;
  const uint8_t* srow = scale + n * groups;
  const Tin* arow = act + m * k_dim;

  float acc = 0.0f;
  for (int64_t g = 0; g < groups; ++g) {
    const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
    const int64_t base = g * 16;
    for (int j = 0; j < 8; ++j) {
      const uint8_t b = prow[g * 8 + j];
      const uint8_t low = b & 0x0Fu;   // input elem base + 2j
      const uint8_t high = b >> 4;     // input elem base + 2j + 1
      const float lo_mag = e2m1[low & 0x7u] * ((low & 0x8u) ? -1.0f : 1.0f);
      const float hi_mag = e2m1[high & 0x7u] * ((high & 0x8u) ? -1.0f : 1.0f);
      // Round the dequanted weight through bf16 exactly like DequantNvfp4ToBf16.
      const float w_lo = __bfloat162float(__float2bfloat16(lo_mag * gs));
      const float w_hi = __bfloat162float(__float2bfloat16(hi_mag * gs));
      acc += Load(arow, base + 2 * j) * w_lo;
      acc += Load(arow, base + 2 * j + 1) * w_hi;
    }
  }
  Store(out, m * n_cols + n, acc);
}

template <typename Tin, typename Tout>
void Launch(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
            const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
  MatmulNvfp4Kernel<Tin, Tout><<<grid, kBlock, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(), scale2, m, n,
      k);
  Check(cudaGetLastError(), "kernel launch");
}

template <typename Tin>
void LaunchByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
                 const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32:
      Launch<Tin, float>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    case DType::kBF16:
      Launch<Tin, __nv_bfloat16>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

// --- Fused MoE grouped NVFP4 GEMM (M2.4) --------------------------------------
// out[p, n] = sum_k act[row(p), k] * bf16(e2m1(W_e[n,k]) * gscale_e[n,k/16]),
// where e = expert_ids[p], W_e = packed_ptrs[e]/scale_ptrs[e]/scale2s[e], and
// row(p) = row_map ? row_map[p] : p. One block over (p rows, N tiles) replaces
// the per-expert loop of MatmulNvfp4 launches — the launch-count reduction is
// the win (GB10 is launch-overhead bound). The per-element decode is byte-for-
// byte MatmulNvfp4Kernel's, so each output row is bit-identical to running the
// single-expert kernel for that pair's expert.
template <typename Tin, typename Tout>
__global__ void MoeGroupedGemmNvfp4Kernel(Tout* out, const Tin* act, const int32_t* expert_ids,
                                          const int32_t* row_map, const int64_t* packed_ptrs,
                                          const int64_t* scale_ptrs, const float* scale2s,
                                          int64_t p_rows, int64_t n_cols, int64_t k_dim) {
  const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_cols) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  // Grid-stride over the pair rows: P = T*top_k can exceed the gridDim.y max
  // (65535), so blockIdx.y strides rather than indexing p directly (a long
  // prefill chunk with a large token count would otherwise blow the y-grid cap).
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
    const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
    const float scale2 = scale2s[e];

    const uint8_t* prow = packed + n * packed_cols;
    const uint8_t* srow = scale + n * groups;
    const Tin* arow = act + r * k_dim;

    float acc = 0.0f;
    for (int64_t g = 0; g < groups; ++g) {
      const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
      const int64_t base = g * 16;
      for (int j = 0; j < 8; ++j) {
        const uint8_t b = prow[g * 8 + j];
        const uint8_t low = b & 0x0Fu;
        const uint8_t high = b >> 4;
        const float lo_mag = e2m1[low & 0x7u] * ((low & 0x8u) ? -1.0f : 1.0f);
        const float hi_mag = e2m1[high & 0x7u] * ((high & 0x8u) ? -1.0f : 1.0f);
        const float w_lo = __bfloat162float(__float2bfloat16(lo_mag * gs));
        const float w_hi = __bfloat162float(__float2bfloat16(hi_mag * gs));
        acc += Load(arow, base + 2 * j) * w_lo;
        acc += Load(arow, base + 2 * j + 1) * w_hi;
      }
    }
    Store(out, p * n_cols + n, acc);
  }
}

template <typename Tin, typename Tout>
void LaunchGrouped(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                   const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                   const Tensor& scale2s, int64_t p, int64_t n, int64_t k) {
  const int64_t y = p < 65535 ? p : 65535;  // grid.y max; kernel strides over p
  const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
  MoeGroupedGemmNvfp4Kernel<Tin, Tout><<<grid, kBlock, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
      row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
      scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch");
}

template <typename Tin>
void LaunchGroupedByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                        const Tensor& scale2s, int64_t p, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32:
      LaunchGrouped<Tin, float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k);
      break;
    case DType::kBF16:
      LaunchGrouped<Tin, __nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

void MoeGroupedGemmNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                   const Tensor& expert_ids, const Tensor* row_map,
                                   const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                                   const Tensor& scale2s) {
  const int64_t p = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (p == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchGroupedByOut<float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k);
      break;
    case DType::kBF16:
      LaunchGroupedByOut<__nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

void MatmulNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                           const Tensor& weight_scale, float weight_scale_2) {
  const int64_t m = act.shape[0], k = act.shape[1], n = weight_packed.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchByOut<float>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    case DType::kBF16:
      LaunchByOut<__nv_bfloat16>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

// Registers the CUDA NVFP4 dequant-GEMM during static init (table fill only, no
// CUDA calls — see cuda_ops.cu for the pre-main discipline rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fn>(&MatmulNvfp4KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4Fn>(&MoeGroupedGemmNvfp4KernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
