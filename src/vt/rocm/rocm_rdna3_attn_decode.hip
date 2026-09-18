// Single-query Gemma attention, matching the pinned ROCm paged_attention_2d.
// Two query heads share one KV head. Keep its key tiles and WMMA accumulation.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

#include "rocm_rdna3_attn_decode.h"

#if defined(__gfx1100__)
#include <rocwmma/rocwmma.hpp>

#include "rocm_rdna3_wmma.h"
#endif

namespace vt::rocm {
namespace {
template <int Tile, int Physical>
__global__ void Rdna3DecodeWmma(__hip_bfloat16* out, const __hip_bfloat16* query,
                                const __hip_bfloat16* key, const __hip_bfloat16* value,
                                const int32_t* blocks, const int32_t* lengths, int64_t block_size,
                                int64_t bt_col, int64_t kb, int64_t kp, int64_t kh, int64_t vb,
                                int64_t vp, int64_t vh, float scale, float softcap, int window_left,
                                int window_right) {
#if defined(__gfx1100__)
  using namespace rocwmma;
  constexpr int D = 256, LD = D + 8, Threads = 128, Waves = Threads / 32;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
  const int head = blockIdx.x;
  const int64_t physical = Physical ? Physical : block_size;
  const int length = lengths[0], pos = length - 1;
  const int first = window_left >= 0 ? max(0, pos - window_left) : 0;
  const int last = window_right >= 0 ? min(pos, pos + window_right) : pos;
  __shared__ bfloat16_t q[16 * LD], kv[Tile * LD];
  __shared__ float scores[Tile * 16], factors[16 * 16], scratch[Waves * 16 * 16];
  __shared__ bfloat16_t probabilities[Tile * 16];
  __shared__ float alpha[2], denominator[2];
  for (int i = tid; i < 16 * D; i += Threads)
    q[(i / D) * LD + i % D] =
        i < 2 * D ? reinterpret_cast<const bfloat16_t*>(query)[head * 2 * D + i] : bfloat16_t{};
  for (int i = tid; i < Tile * 16; i += Threads) probabilities[i] = bfloat16_t{};
  for (int i = tid; i < 256; i += Threads) factors[i] = i / 16 == 0 ? 1.f : 0.f;
  using Acc = fragment<accumulator, 16, 16, 16, float>;
  Acc output[D / 16 / Waves];
#pragma unroll
  for (auto& acc : output) fill_fragment(acc, 0.f);
  float m = -INFINITY, l = 0.f;
  __syncthreads();
  Acc first_head;
  load_matrix_sync(first_head, factors, 16, mem_col_major);
  uint4 values[Tile <= 32 ? Tile * D / 8 / Threads : 1];
  for (int start = (first / Tile) * Tile; start <= last; start += Tile) {
#pragma unroll Tile <= 32 ? Tile * D / 8 / Threads : 1
    for (int f = 0; f < Tile * D / 8 / Threads; ++f) {
      const int i = tid + f * Threads;
      const int k = i / (D / 8), d = (i % (D / 8)) * 8, j = start + k;
      const int64_t offset = j < length
                                 ? static_cast<int64_t>(blocks[(j / physical) * bt_col]) * kb +
                                       (j % physical) * kp + head * kh + d
                                 : 0;
      *reinterpret_cast<uint4*>(kv + k * LD + d) =
          j < length ? *reinterpret_cast<const uint4*>(key + offset) : uint4{0, 0, 0, 0};
      const int64_t value_offset =
          j < length ? static_cast<int64_t>(blocks[(j / physical) * bt_col]) * vb +
                           (j % physical) * vp + head * vh + d
                     : 0;
      if constexpr (Tile <= 32)
        values[f] =
            j < length ? *reinterpret_cast<const uint4*>(value + value_offset) : uint4{0, 0, 0, 0};
    }
    __syncthreads();
    if (wave < Tile / 16) {
      fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a;
      fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major> b;
      Acc score;
      fill_fragment(score, 0.f);
#pragma unroll 1
      for (int d = 0; d < D; d += 16) {
        load_matrix_sync(a, kv + wave * 16 * LD + d, LD);
        load_matrix_sync(b, q + d, LD);
        AttentionMmaSync(score, a, b, score);
      }
      store_matrix_sync(scores + wave * 256, score, 16, mem_row_major);
    }
    __syncthreads();
    float p[2] = {};
    if (wave < 2) {
      float s[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int k = lane + half * 32, j = start + k;
        if (k < Tile && j >= first && j <= last) {
          const float raw = scores[k * 16 + wave] * scale;
          s[half] = softcap > 0.f ? softcap * tanhf(raw / softcap) : raw;
        }
      }
      float next = fmaxf(s[0], s[1]);
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) next = fmaxf(next, __shfl_xor(next, offset));
      next = fmaxf(m, next);
      const float a =
          m > -INFINITY ? __builtin_amdgcn_exp2f((m - next) * 1.4426950408889634f) : 0.f;
#pragma unroll
      for (int half = 0; half < 2; ++half)
        p[half] =
            next > -INFINITY ? __builtin_amdgcn_exp2f((s[half] - next) * 1.4426950408889634f) : 0.f;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < Tile / 2; ++i) {
        const int k = 2 * i + (lane & 1);
        sum += __shfl(p[k / 32], k % 32);
      }
      sum += __shfl_xor(sum, 1);
      l = fmaf(l, a, sum);
      m = next;
      if (lane == 0) {
        alpha[wave] = a;
        denominator[wave] = l + 1e-10f;
      }
    }
    if (wave < 2)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int k = lane + 32 * half;
        if (k < Tile) probabilities[k * 16 + wave] = static_cast<bfloat16_t>(p[half]);
      }
#pragma unroll Tile <= 32 ? Tile * D / 8 / Threads : 1
    for (int f = 0; f < Tile * D / 8 / Threads; ++f) {
      const int i = tid + f * Threads, k = i / (D / 8), d = (i % (D / 8)) * 8;
      if constexpr (Tile <= 32) {
        *reinterpret_cast<uint4*>(kv + k * LD + d) = values[f];
      } else {
        const int j = start + k;
        const int64_t offset = j < length
                                   ? static_cast<int64_t>(blocks[(j / physical) * bt_col]) * vb +
                                         (j % physical) * vp + head * vh + d
                                   : 0;
        *reinterpret_cast<uint4*>(kv + k * LD + d) =
            j < length ? *reinterpret_cast<const uint4*>(value + offset) : uint4{0, 0, 0, 0};
      }
    }
    __syncthreads();
    // A public accumulator load supplies the head selector. Reuse that fixed
    // layout in registers instead of materializing a scale matrix per tile.
    const float a0 = alpha[0], a1 = alpha[1];
#pragma unroll
    for (int f = 0; f < D / 16 / Waves; ++f) {
#pragma unroll
      for (unsigned i = 0; i < Acc::num_elements; ++i)
        output[f].x[i] *= first_head.x[i] > 0.f ? a0 : a1;
      fragment<matrix_a, 16, 16, 16, bfloat16_t, col_major> a;
      fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major> b;
#pragma unroll 1
      for (int k = 0; k < Tile; k += 16) {
        load_matrix_sync(a, kv + k * LD + (wave + f * Waves) * 16, LD);
        load_matrix_sync(b, probabilities + k * 16, 16);
        AttentionMmaSync(output[f], a, b, output[f]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int f = 0; f < D / 16 / Waves; ++f) {
    float* tile = scratch + wave * 256;
    store_matrix_sync(tile, output[f], 16, mem_col_major);
    __syncthreads();
    out[(head * 2 + lane / 16) * D + (wave + f * Waves) * 16 + lane % 16] =
        __float2bfloat16(tile[lane] / denominator[lane / 16]);
    __syncthreads();
  }
#else
  (void)out;
  (void)query;
  (void)key;
  (void)value;
  (void)blocks;
  (void)lengths;
  (void)block_size;
  (void)bt_col;
  (void)kb;
  (void)kp;
  (void)kh;
  (void)vb;
  (void)vp;
  (void)vh;
  (void)scale;
  (void)softcap;
  (void)window_left;
  (void)window_right;
#endif
}
}  // namespace

void Rdna3AttentionDecode(Queue& queue, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor& blocks, const Tensor& lengths,
                          const PagedAttentionArgs& args, int window_left, int window_right) {
  const int64_t block = key.shape[1];
  auto launch = [&]<int Tile, int Physical>() {
    Rdna3DecodeWmma<Tile, Physical>
        <<<static_cast<unsigned>(key.shape[2]), 128, 0, static_cast<hipStream_t>(queue.handle)>>>(
            out.Ptr<__hip_bfloat16>(), query.Ptr<__hip_bfloat16>(), key.Ptr<__hip_bfloat16>(),
            value.Ptr<__hip_bfloat16>(), blocks.Ptr<int32_t>(), lengths.Ptr<int32_t>(), block,
            blocks.stride[1], key.stride[0], key.stride[1], key.stride[2], value.stride[0],
            value.stride[1], value.stride[2], args.scale, args.logits_soft_cap, window_left,
            window_right);
  };
  if (block == 16)
    launch.template operator()<16, 16>();
  else if (block == 64)
    launch.template operator()<64, 64>();
  else if (block == 32)
    launch.template operator()<32, 32>();
  else
    launch.template operator()<32, 0>();
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("ROCm RDNA3 decode launch failed");
}
}  // namespace vt::rocm
