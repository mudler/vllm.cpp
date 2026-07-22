// Elementwise CPU GEMM micro-kernels — see cpu_matmul_elem.h for the upstream
// anchors and the recorded "vectorize across OUTPUT columns, not along K"
// deviation that keeps every result bit-identical to the scalar reference.
#include "cpu_matmul_elem.h"

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vt::cpu {
namespace {

template <ElemKind K>
struct Elem;
template <>
struct Elem<ElemKind::kF32> {
  using T = float;
  static inline float Cvt(T v) { return v; }
};
template <>
struct Elem<ElemKind::kF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return F16ToF32(v); }
};
template <>
struct Elem<ElemKind::kBF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return BF16ToF32(v); }
};

// ---------------------------------------------------------------------------
// Tier 0 — portable C++, always built, what CI exercises.
//
// The ONE structural change versus the historical kernel: `kElemLanes`
// independent accumulators instead of one. The old loop's `acc += a*b` is a
// loop-carried dependency on a single f32 adder, so it ran at one MAC per
// FP-add latency (~4 cycles) — which is exactly the 0.77-0.84 GFLOP/s/thread
// the floor re-measurement recorded. Sixteen chains saturate the pipes while
// each chain keeps its own strictly sequential order over p.
// ---------------------------------------------------------------------------

template <ElemKind K>
void Bt16Portable(const float* af, const void* bv, int64_t k, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  float s[kElemLanes];
  for (int l = 0; l < kElemLanes; ++l) s[l] = 0.0f;
  for (int64_t p = 0; p < k; ++p) {
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      s[l] += av * E::Cvt(b[static_cast<int64_t>(l) * k + p]);
    }
  }
  for (int l = 0; l < kElemLanes; ++l) acc[l] = s[l];
}

template <ElemKind K>
void Nk16Portable(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  float s[kElemLanes];
  for (int l = 0; l < kElemLanes; ++l) s[l] = 0.0f;
  for (int64_t p = 0; p < k; ++p) {
    const float av = af[p];
    const typename E::T* row = b + p * n;
    for (int l = 0; l < kElemLanes; ++l) {
      s[l] += av * E::Cvt(row[l]);
    }
  }
  for (int l = 0; l < kElemLanes; ++l) acc[l] = s[l];
}

// ---------------------------------------------------------------------------
// Tier 1a — AArch64 NEON (the dgx.casa / GB10 tier).
//
// 4 accumulator vectors x 4 lanes = the same 16 output columns. For the [N,K]
// weight the four rows of a group are contiguous along p, so a 4-element load
// per row plus a 4x4 transpose yields, for each p, the vector of that p across
// the four output columns — which is what lets the lane-wise accumulation stay
// in p order. Products are vmulq + vaddq, NEVER vfmaq: the scalar reference
// rounds the product before the add (-ffp-contract=off, CMakeLists.txt:21).
//
// bf16 widening is ggml's shift-left-16 (vec.cpp:172, the AVX2 `LOAD` macro),
// here `vshll_n_u16(v, 16)`; f16 widening is the hardware convert ggml reaches
// through GGML_F16_VEC_LOAD (simd-mappings.h), here `vcvt_f32_f16`. Both are
// asserted bit-identical to vt::BF16ToF32 / vt::F16ToF32 over the exhaustive
// 65,536-value domain in tests/vt/test_ops_matmul_elem.cpp.
// ---------------------------------------------------------------------------
#if defined(__aarch64__)

template <ElemKind K>
inline float32x4_t LoadV4(const typename Elem<K>::T* p);
template <>
inline float32x4_t LoadV4<ElemKind::kF32>(const float* p) {
  return vld1q_f32(p);
}
template <>
inline float32x4_t LoadV4<ElemKind::kBF16>(const uint16_t* p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
template <>
inline float32x4_t LoadV4<ElemKind::kF16>(const uint16_t* p) {
  return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
}

inline void Transpose4(float32x4_t& r0, float32x4_t& r1, float32x4_t& r2, float32x4_t& r3) {
  const float32x4x2_t t01 = vtrnq_f32(r0, r1);
  const float32x4x2_t t23 = vtrnq_f32(r2, r3);
  r0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
  r1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
  r2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
  r3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
}

template <ElemKind K>
void Bt16Neon(const float* af, const void* bv, int64_t k, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
  float32x4_t* A[4] = {&a0, &a1, &a2, &a3};
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    const float32x4_t av = vld1q_f32(af + p);
    for (int g = 0; g < 4; ++g) {
      const typename E::T* br = b + static_cast<int64_t>(4 * g) * k + p;
      float32x4_t r0 = LoadV4<K>(br);
      float32x4_t r1 = LoadV4<K>(br + k);
      float32x4_t r2 = LoadV4<K>(br + 2 * k);
      float32x4_t r3 = LoadV4<K>(br + 3 * k);
      Transpose4(r0, r1, r2, r3);
      float32x4_t s = *A[g];
      s = vaddq_f32(s, vmulq_f32(r0, vdupq_laneq_f32(av, 0)));
      s = vaddq_f32(s, vmulq_f32(r1, vdupq_laneq_f32(av, 1)));
      s = vaddq_f32(s, vmulq_f32(r2, vdupq_laneq_f32(av, 2)));
      s = vaddq_f32(s, vmulq_f32(r3, vdupq_laneq_f32(av, 3)));
      *A[g] = s;
    }
  }
  vst1q_f32(acc + 0, a0);
  vst1q_f32(acc + 4, a1);
  vst1q_f32(acc + 8, a2);
  vst1q_f32(acc + 12, a3);
  for (; p < k; ++p) {  // K tail, still in p order per lane
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      acc[l] += av * E::Cvt(b[static_cast<int64_t>(l) * k + p]);
    }
  }
}

// M-blocked [N,K] kernel: kMrNeon activation rows share one weight load +
// transpose. 16 accumulator vectors + 4 weight vectors + the broadcast row
// values fit AArch64's 32 SIMD registers.
constexpr int kMrNeon = 4;

template <ElemKind K>
void BtM4Neon(const float* af, int64_t a_stride, const void* bv, int64_t k, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  float32x4_t A[kMrNeon][4];
  for (int r = 0; r < kMrNeon; ++r) {
    for (int g = 0; g < 4; ++g) A[r][g] = vdupq_n_f32(0.0f);
  }
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    for (int g = 0; g < 4; ++g) {
      const typename E::T* br = b + static_cast<int64_t>(4 * g) * k + p;
      float32x4_t w0 = LoadV4<K>(br);
      float32x4_t w1 = LoadV4<K>(br + k);
      float32x4_t w2 = LoadV4<K>(br + 2 * k);
      float32x4_t w3 = LoadV4<K>(br + 3 * k);
      Transpose4(w0, w1, w2, w3);
      for (int r = 0; r < kMrNeon; ++r) {
        const float32x4_t av = vld1q_f32(af + r * a_stride + p);
        float32x4_t s = A[r][g];
        s = vaddq_f32(s, vmulq_f32(w0, vdupq_laneq_f32(av, 0)));
        s = vaddq_f32(s, vmulq_f32(w1, vdupq_laneq_f32(av, 1)));
        s = vaddq_f32(s, vmulq_f32(w2, vdupq_laneq_f32(av, 2)));
        s = vaddq_f32(s, vmulq_f32(w3, vdupq_laneq_f32(av, 3)));
        A[r][g] = s;
      }
    }
  }
  for (int r = 0; r < kMrNeon; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_f32(acc + r * kElemLanes + 4 * g, A[r][g]);
  }
  for (int64_t pt = p; pt < k; ++pt) {  // K tail, still in p order per lane
    for (int r = 0; r < kMrNeon; ++r) {
      const float av = af[r * a_stride + pt];
      for (int l = 0; l < kElemLanes; ++l) {
        acc[r * kElemLanes + l] += av * E::Cvt(b[static_cast<int64_t>(l) * k + pt]);
      }
    }
  }
}

template <ElemKind K>
void Nk16Neon(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  const typename Elem<K>::T* b = static_cast<const typename Elem<K>::T*>(bv);
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
  for (int64_t p = 0; p < k; ++p) {
    const float32x4_t av = vdupq_n_f32(af[p]);
    const typename Elem<K>::T* row = b + p * n;
    a0 = vaddq_f32(a0, vmulq_f32(LoadV4<K>(row + 0), av));
    a1 = vaddq_f32(a1, vmulq_f32(LoadV4<K>(row + 4), av));
    a2 = vaddq_f32(a2, vmulq_f32(LoadV4<K>(row + 8), av));
    a3 = vaddq_f32(a3, vmulq_f32(LoadV4<K>(row + 12), av));
  }
  vst1q_f32(acc + 0, a0);
  vst1q_f32(acc + 4, a1);
  vst1q_f32(acc + 8, a2);
  vst1q_f32(acc + 12, a3);
}

#endif  // __aarch64__

// ---------------------------------------------------------------------------
// Tier 1b — x86-64 SSE2 (baseline on x86-64, so no runtime probe needed) for
// f32/bf16, plus an F16C-gated f16 pair selected by __builtin_cpu_supports.
// Same 4x4-transpose / lane-per-output-column structure as the NEON tier, so
// the bit-exactness argument is identical. bf16 widening is
// `_mm_unpacklo_epi16(zero, v)` — the SSE2 spelling of ggml's shift-left-16
// (vec.cpp:172). The wider AVX2/AVX512 quant tier remains work row G5.
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)

template <ElemKind K>
inline __m128 LoadX4(const typename Elem<K>::T* p);
template <>
inline __m128 LoadX4<ElemKind::kF32>(const float* p) {
  return _mm_loadu_ps(p);
}
template <>
inline __m128 LoadX4<ElemKind::kBF16>(const uint16_t* p) {
  __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  return _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), v));
}

template <ElemKind K>
void Bt16Sse2(const float* af, const void* bv, int64_t k, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  __m128 A[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    const __m128 av = _mm_loadu_ps(af + p);
    for (int g = 0; g < 4; ++g) {
      const typename E::T* br = b + static_cast<int64_t>(4 * g) * k + p;
      __m128 r0 = LoadX4<K>(br);
      __m128 r1 = LoadX4<K>(br + k);
      __m128 r2 = LoadX4<K>(br + 2 * k);
      __m128 r3 = LoadX4<K>(br + 3 * k);
      _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
      __m128 s = A[g];
      s = _mm_add_ps(s, _mm_mul_ps(r0, _mm_shuffle_ps(av, av, 0x00)));
      s = _mm_add_ps(s, _mm_mul_ps(r1, _mm_shuffle_ps(av, av, 0x55)));
      s = _mm_add_ps(s, _mm_mul_ps(r2, _mm_shuffle_ps(av, av, 0xAA)));
      s = _mm_add_ps(s, _mm_mul_ps(r3, _mm_shuffle_ps(av, av, 0xFF)));
      A[g] = s;
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, A[g]);
  for (; p < k; ++p) {
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      acc[l] += av * E::Cvt(b[static_cast<int64_t>(l) * k + p]);
    }
  }
}

// M-blocked SSE2 [N,K] kernel. MR=2 (not 4): x86-64 SSE2 has only 16 XMM
// registers, so 2 rows x 4 groups of accumulators plus the 4 transposed weight
// vectors is what fits without spilling.
constexpr int kMrSse2 = 2;

template <ElemKind K>
void BtM2Sse2(const float* af, int64_t a_stride, const void* bv, int64_t k, float* acc) {
  using E = Elem<K>;
  const typename E::T* b = static_cast<const typename E::T*>(bv);
  __m128 A[kMrSse2][4];
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) A[r][g] = _mm_setzero_ps();
  }
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    for (int g = 0; g < 4; ++g) {
      const typename E::T* br = b + static_cast<int64_t>(4 * g) * k + p;
      __m128 w0 = LoadX4<K>(br);
      __m128 w1 = LoadX4<K>(br + k);
      __m128 w2 = LoadX4<K>(br + 2 * k);
      __m128 w3 = LoadX4<K>(br + 3 * k);
      _MM_TRANSPOSE4_PS(w0, w1, w2, w3);
      for (int r = 0; r < kMrSse2; ++r) {
        const __m128 av = _mm_loadu_ps(af + r * a_stride + p);
        __m128 s = A[r][g];
        s = _mm_add_ps(s, _mm_mul_ps(w0, _mm_shuffle_ps(av, av, 0x00)));
        s = _mm_add_ps(s, _mm_mul_ps(w1, _mm_shuffle_ps(av, av, 0x55)));
        s = _mm_add_ps(s, _mm_mul_ps(w2, _mm_shuffle_ps(av, av, 0xAA)));
        s = _mm_add_ps(s, _mm_mul_ps(w3, _mm_shuffle_ps(av, av, 0xFF)));
        A[r][g] = s;
      }
    }
  }
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + r * kElemLanes + 4 * g, A[r][g]);
  }
  for (int64_t pt = p; pt < k; ++pt) {
    for (int r = 0; r < kMrSse2; ++r) {
      const float av = af[r * a_stride + pt];
      for (int l = 0; l < kElemLanes; ++l) {
        acc[r * kElemLanes + l] += av * E::Cvt(b[static_cast<int64_t>(l) * k + pt]);
      }
    }
  }
}

__attribute__((target("f16c"))) void BtM2F16c(const float* af, int64_t a_stride, const void* bv,
                                              int64_t k, float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 A[kMrSse2][4];
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) A[r][g] = _mm_setzero_ps();
  }
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    for (int g = 0; g < 4; ++g) {
      const uint16_t* br = b + static_cast<int64_t>(4 * g) * k + p;
      __m128 w0 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br)));
      __m128 w1 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + k)));
      __m128 w2 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + 2 * k)));
      __m128 w3 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + 3 * k)));
      _MM_TRANSPOSE4_PS(w0, w1, w2, w3);
      for (int r = 0; r < kMrSse2; ++r) {
        const __m128 av = _mm_loadu_ps(af + r * a_stride + p);
        __m128 s = A[r][g];
        s = _mm_add_ps(s, _mm_mul_ps(w0, _mm_shuffle_ps(av, av, 0x00)));
        s = _mm_add_ps(s, _mm_mul_ps(w1, _mm_shuffle_ps(av, av, 0x55)));
        s = _mm_add_ps(s, _mm_mul_ps(w2, _mm_shuffle_ps(av, av, 0xAA)));
        s = _mm_add_ps(s, _mm_mul_ps(w3, _mm_shuffle_ps(av, av, 0xFF)));
        A[r][g] = s;
      }
    }
  }
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + r * kElemLanes + 4 * g, A[r][g]);
  }
  for (int64_t pt = p; pt < k; ++pt) {
    for (int r = 0; r < kMrSse2; ++r) {
      const float av = af[r * a_stride + pt];
      for (int l = 0; l < kElemLanes; ++l) {
        acc[r * kElemLanes + l] += av * F16ToF32(b[static_cast<int64_t>(l) * k + pt]);
      }
    }
  }
}

template <ElemKind K>
void Nk16Sse2(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  const typename Elem<K>::T* b = static_cast<const typename Elem<K>::T*>(bv);
  __m128 A[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
  for (int64_t p = 0; p < k; ++p) {
    const __m128 av = _mm_set1_ps(af[p]);
    const typename Elem<K>::T* row = b + p * n;
    for (int g = 0; g < 4; ++g) {
      A[g] = _mm_add_ps(A[g], _mm_mul_ps(LoadX4<K>(row + 4 * g), av));
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, A[g]);
}

__attribute__((target("f16c"))) void Bt16F16c(const float* af, const void* bv, int64_t k,
                                              float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 A[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    const __m128 av = _mm_loadu_ps(af + p);
    for (int g = 0; g < 4; ++g) {
      const uint16_t* br = b + static_cast<int64_t>(4 * g) * k + p;
      __m128 r0 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br)));
      __m128 r1 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + k)));
      __m128 r2 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + 2 * k)));
      __m128 r3 = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(br + 3 * k)));
      _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
      __m128 s = A[g];
      s = _mm_add_ps(s, _mm_mul_ps(r0, _mm_shuffle_ps(av, av, 0x00)));
      s = _mm_add_ps(s, _mm_mul_ps(r1, _mm_shuffle_ps(av, av, 0x55)));
      s = _mm_add_ps(s, _mm_mul_ps(r2, _mm_shuffle_ps(av, av, 0xAA)));
      s = _mm_add_ps(s, _mm_mul_ps(r3, _mm_shuffle_ps(av, av, 0xFF)));
      A[g] = s;
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, A[g]);
  for (; p < k; ++p) {
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      acc[l] += av * F16ToF32(b[static_cast<int64_t>(l) * k + p]);
    }
  }
}

__attribute__((target("f16c"))) void Nk16F16c(const float* af, const void* bv, int64_t k,
                                              int64_t n, float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 A[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
  for (int64_t p = 0; p < k; ++p) {
    const __m128 av = _mm_set1_ps(af[p]);
    const uint16_t* row = b + p * n;
    for (int g = 0; g < 4; ++g) {
      const __m128 v =
          _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + 4 * g)));
      A[g] = _mm_add_ps(A[g], _mm_mul_ps(v, av));
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, A[g]);
}

#endif  // x86-64

constexpr int kF32 = static_cast<int>(ElemKind::kF32);
constexpr int kF16 = static_cast<int>(ElemKind::kF16);
constexpr int kBF16 = static_cast<int>(ElemKind::kBF16);

ElemGemmTierTable BuildPortableTier() {
  ElemGemmTierTable t{};
  t.bt[kF32] = &Bt16Portable<ElemKind::kF32>;
  t.bt[kF16] = &Bt16Portable<ElemKind::kF16>;
  t.bt[kBF16] = &Bt16Portable<ElemKind::kBF16>;
  t.nk[kF32] = &Nk16Portable<ElemKind::kF32>;
  t.nk[kF16] = &Nk16Portable<ElemKind::kF16>;
  t.nk[kBF16] = &Nk16Portable<ElemKind::kBF16>;
  t.mr = 1;  // no M blocking: the portable tier has no transpose to amortize
  t.name = "portable";
  return t;
}

std::string ForcedTier() {
  const char* forced = std::getenv("VT_CPU_MATMUL_TIER");
  return forced == nullptr ? std::string() : std::string(forced);
}

ElemGemmTierTable BuildTier() {
  const std::string forced = ForcedTier();
  ElemGemmTierTable t = BuildPortableTier();
  if (forced == "portable") return t;
  if (forced == "ref") {
    t.name = "ref";
    return t;
  }
#if defined(__aarch64__)
  t.bt[kF32] = &Bt16Neon<ElemKind::kF32>;
  t.bt[kF16] = &Bt16Neon<ElemKind::kF16>;
  t.bt[kBF16] = &Bt16Neon<ElemKind::kBF16>;
  t.nk[kF32] = &Nk16Neon<ElemKind::kF32>;
  t.nk[kF16] = &Nk16Neon<ElemKind::kF16>;
  t.nk[kBF16] = &Nk16Neon<ElemKind::kBF16>;
  t.btm[kF32] = &BtM4Neon<ElemKind::kF32>;
  t.btm[kF16] = &BtM4Neon<ElemKind::kF16>;
  t.btm[kBF16] = &BtM4Neon<ElemKind::kBF16>;
  t.mr = kMrNeon;
  t.name = "neon";
#elif defined(__x86_64__) || defined(_M_X64)
  t.bt[kF32] = &Bt16Sse2<ElemKind::kF32>;
  t.bt[kBF16] = &Bt16Sse2<ElemKind::kBF16>;
  t.nk[kF32] = &Nk16Sse2<ElemKind::kF32>;
  t.nk[kBF16] = &Nk16Sse2<ElemKind::kBF16>;
  t.btm[kF32] = &BtM2Sse2<ElemKind::kF32>;
  t.btm[kBF16] = &BtM2Sse2<ElemKind::kBF16>;
  t.mr = kMrSse2;
  t.name = "sse2";
  if (__builtin_cpu_supports("f16c")) {
    t.bt[kF16] = &Bt16F16c;
    t.nk[kF16] = &Nk16F16c;
    t.btm[kF16] = &BtM2F16c;
    t.name = "sse2+f16c";
  } else {
    // No f16 M-blocked kernel without F16C: fall back to the 1-row path for
    // f16 only, which the caller selects on a null btm entry.
    t.btm[kF16] = nullptr;
  }
#endif
  return t;
}

}  // namespace

bool ElemKindOf(DType dt, ElemKind* out) {
  switch (dt) {
    case DType::kF32: *out = ElemKind::kF32; return true;
    case DType::kF16: *out = ElemKind::kF16; return true;
    case DType::kBF16: *out = ElemKind::kBF16; return true;
    default: return false;
  }
}

const ElemGemmTierTable& ElemGemmTier() {
  static const ElemGemmTierTable t = BuildTier();
  return t;
}

const char* ElemGemmTierName() { return ElemGemmTier().name; }

bool ElemGemmUseRef() {
  static const bool v = std::strcmp(ElemGemmTier().name, "ref") == 0;
  return v;
}

void WidenRowToF32(DType dt, const void* src, int64_t n, float* dst) {
  switch (dt) {
    case DType::kF32:
      std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
      break;
    case DType::kF16: {
      const uint16_t* s = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) dst[i] = F16ToF32(s[i]);
      break;
    }
    case DType::kBF16: {
      const uint16_t* s = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) dst[i] = BF16ToF32(s[i]);
      break;
    }
    default:
      VT_CHECK(false, "matmul: unsupported elementwise dtype");
  }
}

}  // namespace vt::cpu
