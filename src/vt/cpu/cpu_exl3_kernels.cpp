// EXL3 (exllamav3 trellis) device kernels, CPU arm — MODEL-DSV4-EXL3 W2a/W2b.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/hadamard.cu:88-173          had_r_128, the host launcher
//   exllamav3_ext/quant/hadamard_inner.cuh:17-44    shuffle_had_f4x32
//   exllamav3_ext/quant/hadamard_inner.cuh:93-279   the hf / ff / fh inners
//   exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-80   the fused chain the GEMM runs
//   exllamav3_ext/quant/exl3_dq.cuh + codebook.cuh  reached through W1a's
//                                                   Exl3DecodeTile (cpu_exl3_dequant.cpp)
//
// WHY THE OPERATION ORDER IS COPIED AND NOT SIMPLIFIED. The obvious CPU
// Hadamard is a two-line loop nest. This one instead reproduces upstream's warp
// decomposition exactly: levels 1-2 on the four values a "lane" holds, levels
// 4-64 as five xor-partner steps over 32 lanes, the sign flip performed by
// XORing the f32 SIGN BIT, every add in f32, one multiply by `r_scale` at the
// end and one round at the store. That is what makes the claim in
// `.agents/specs/model-dsv4-exl3.md` `## W2 design` §1 tier 2 a BYTE claim
// rather than a tolerance: the CUDA arm and this arm perform the same f32
// operations on the same values in the same order, so a device-vs-host gate can
// require equality and a defect cannot hide inside a bound. Written as a plain
// loop it would be close, and close is not checkable.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// 1/sqrt(128), spelled as upstream spells it (hadamard.cu:107). Kept as the
// literal rather than computed, because the literal is what the device rounds
// to and a recomputed 1/sqrt(128) can differ in the last f32 bit.
constexpr float kInvSqrt128 = 0.088388347648f;

inline float NegBySignBit(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  u ^= 0x80000000u;
  float r;
  std::memcpy(&r, &u, sizeof(r));
  return r;
}

// fp16 multiply, the CPU spelling of __hmul2. The exact product of two fp16
// significands is 22 bits and fits f32 exactly, so rounding once from f32 is
// the same value an fp16 multiply produces.
inline uint16_t MulF16(uint16_t a, uint16_t b) {
  return F32ToF16(F16ToF32(a) * F16ToF32(b));
}

// `shuffle_had_f4x32` (hadamard_inner.cuh:17-44) over a whole 32-"lane" warp.
// `h[j][t]` is lane t's j-th value; the element it stands for is column 4*t + j.
// Lane bit `i` therefore carries Hadamard level 4*i, which is what makes levels
// 4..64 exactly the five steps below.
void ShuffleHadWarp(float h[4][32]) {
  float next[4][32];
  for (int i = 1; i < 32; i <<= 1) {
    for (int t = 0; t < 32; ++t) {
      const int p = t ^ i;
      const bool flip = (t & i) != 0;
      for (int j = 0; j < 4; ++j) {
        const float own = flip ? NegBySignBit(h[j][t]) : h[j][t];
        next[j][t] = own + h[j][p];
      }
    }
    std::memcpy(h, next, sizeof(next));
  }
}

// The 128-element transform of ONE block, in upstream's order. `in` holds the
// 128 f32 values ALREADY pre-scaled (or not); `out` receives them transformed
// and multiplied by `r_scale`, before any post-scale.
void HadBlock128(const float* in, float* out, float r_scale) {
  float h[4][32];
  for (int t = 0; t < 32; ++t) {
    // hadamard_inner.cuh:118-129, the level-1 and level-2 butterflies.
    const float v0 = in[4 * t + 0];
    const float v1 = in[4 * t + 1];
    const float v2 = in[4 * t + 2];
    const float v3 = in[4 * t + 3];
    const float s0 = v0 + v1;
    const float d0 = v0 - v1;
    const float s1 = v2 + v3;
    const float d1 = v2 - v3;
    h[0][t] = s0 + s1;
    h[1][t] = d0 + d1;
    h[2][t] = s0 - s1;
    h[3][t] = d0 - d1;
  }
  ShuffleHadWarp(h);
  for (int t = 0; t < 32; ++t)
    for (int j = 0; j < 4; ++j) out[4 * t + j] = h[j][t] * r_scale;
}

// `had_hf_r_128_inner` (fp16 in, fp16 out), `had_ff_r_128_inner` (f32 in, f32
// out) and `had_fh_r_128_inner` (f32 in, fp16 out) differ in exactly three
// places: how the value is loaded, how a pre-scale multiplies it (fp16 multiply
// for the half input, f32 multiply for the float input — hadamard_inner.cuh:109
// vs :167), and how the post-scale applies after the store rounding. Everything
// between is the same f32 arithmetic, so the three share this body.
enum class HadIo { kHalfHalf, kFloatFloat, kFloatHalf };

void HadRowBlock(HadIo io, const void* in, void* out, const uint16_t* pre, const uint16_t* post,
                 float r_scale, int64_t block_base) {
  float buf[128];
  if (io == HadIo::kHalfHalf) {
    const uint16_t* p = static_cast<const uint16_t*>(in);
    for (int i = 0; i < 128; ++i) {
      // pre_scale rides an fp16 multiply BEFORE the widen (hadamard_inner.cuh:112-114).
      const uint16_t v = pre != nullptr ? MulF16(p[i], pre[block_base + i]) : p[i];
      buf[i] = F16ToF32(v);
    }
  } else {
    const float* p = static_cast<const float*>(in);
    for (int i = 0; i < 128; ++i) {
      // the float inners widen the fp16 scale and multiply in f32
      // (hadamard_inner.cuh:171-174).
      buf[i] = pre != nullptr ? p[i] * F16ToF32(pre[block_base + i]) : p[i];
    }
  }

  float res[128];
  HadBlock128(buf, res, r_scale);

  if (io == HadIo::kFloatFloat) {
    float* o = static_cast<float*>(out);
    for (int i = 0; i < 128; ++i)
      o[i] = post != nullptr ? res[i] * F16ToF32(post[block_base + i]) : res[i];
  } else {
    // Both half-output inners round FIRST and apply the post-scale as an fp16
    // multiply afterwards (hadamard_inner.cuh:137-146 and :264-278).
    uint16_t* o = static_cast<uint16_t*>(out);
    for (int i = 0; i < 128; ++i) {
      const uint16_t r = F32ToF16(res[i]);
      o[i] = post != nullptr ? MulF16(r, post[block_base + i]) : r;
    }
  }
}

void HadRows(HadIo io, const void* in, void* out, const uint16_t* pre, const uint16_t* post,
             float r_scale, int64_t rows, int64_t cols) {
  const bool half_in = (io == HadIo::kHalfHalf);
  const bool half_out = (io != HadIo::kFloatFloat);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < cols; b += 128) {
      const void* ip = half_in ? static_cast<const void*>(static_cast<const uint16_t*>(in) + r * cols + b)
                               : static_cast<const void*>(static_cast<const float*>(in) + r * cols + b);
      void* op = half_out ? static_cast<void*>(static_cast<uint16_t*>(out) + r * cols + b)
                          : static_cast<void*>(static_cast<float*>(out) + r * cols + b);
      HadRowBlock(io, ip, op, pre, post, r_scale, b);
    }
  }
}

void Exl3HadR128KernelCpu(Queue& q, Tensor& out, const Tensor& in, const Exl3HadArgs& args) {
  (void)q;
  const int64_t rows = in.shape[0];
  const int64_t cols = in.shape[1];
  if (rows == 0 || cols == 0) return;
  const uint16_t* pre = args.pre_scale != nullptr ? args.pre_scale->Ptr<uint16_t>() : nullptr;
  const uint16_t* post = args.post_scale != nullptr ? args.post_scale->Ptr<uint16_t>() : nullptr;
  const float r_scale = args.scale * kInvSqrt128;  // hadamard.cu:107
  HadRows(in.dtype == DType::kF16 ? HadIo::kHalfHalf : HadIo::kFloatFloat, in.data, out.data, pre,
          post, r_scale, rows, cols);
}

// The fused chain `exl3_gemm_kernel` runs (exl3_gemm_kernel.cuh:14-50 plus the
// output transform at exl3_gemm_inner.cuh:456-480):
//   A_had = had_r_128(A, pre_scale = suh)
//   C_raw = A_had @ reconstruct(trellis)          [f32 accumulation]
//   C     = had_r_128(C_raw, post_scale = svh)
// The device does the first and third INSIDE the GEMM launch; here they are the
// same three steps, and the middle one accumulates in f32 exactly as the mma
// accumulators do. The trellis is decoded a 16x16 TILE AT A TIME rather than
// materialised as a [k, n] matrix, because a real expert is k = 4096, n = 2048
// and the materialised f32 weight would be 32 MiB per projection.
void Exl3GemmKernelCpu(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                       const Tensor& suh, const Tensor& svh, Tensor& a_had,
                       const Exl3GemmArgs& args) {
  (void)q;
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;

  // 1. the input transform, into the caller's scratch (which may alias A).
  HadRows(HadIo::kHalfHalf, a.data, a_had.data, suh.Ptr<uint16_t>(), nullptr, kInvSqrt128, m, k);

  // 2. the matmul against the decoded trellis, f32 accumulators.
  const uint16_t* ah = a_had.Ptr<uint16_t>();
  const uint16_t* tw = trellis.Ptr<uint16_t>();
  const int64_t tiles_n = n / 16;
  const int64_t tile_words = 16 * static_cast<int64_t>(args.bits);
  std::vector<float> raw(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
  float tile[256];
  for (int64_t ti = 0; ti < k / 16; ++ti) {
    for (int64_t tj = 0; tj < tiles_n; ++tj) {
      Exl3DecodeTile(tw + (ti * tiles_n + tj) * tile_words, args.bits, tile);
      for (int64_t r = 0; r < m; ++r) {
        float* orow = &raw[static_cast<size_t>(r * n + tj * 16)];
        for (int rr = 0; rr < 16; ++rr) {
          const float xv = F16ToF32(ah[r * k + ti * 16 + rr]);
          if (xv == 0.0f) continue;
          const float* wrow = tile + rr * 16;
          for (int cc = 0; cc < 16; ++cc) orow[cc] += xv * wrow[cc];
        }
      }
    }
  }

  // 3. the output transform. The device holds this tile in f32 shared memory and
  // finishes with had_ff (f32 C) or had_fh (fp16 C) — the same two arms here.
  HadRows(c.dtype == DType::kF32 ? HadIo::kFloatFloat : HadIo::kFloatHalf, raw.data(), c.data,
          nullptr, svh.Ptr<uint16_t>(), kInvSqrt128, m, n);
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kExl3HadR128, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Exl3HadR128Fn>(&Exl3HadR128KernelCpu)));
    RegisterOp(OpId::kExl3Gemm, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelCpu)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
