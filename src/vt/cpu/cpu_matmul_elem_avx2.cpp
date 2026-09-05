// vllm.cpp original (KERNEL-GEMM-CPU-ELEM-X86WIDE work row W1); build style
// mirrors llama.cpp's per-ISA CPU build (ggml/src/CMakeLists.txt:371-401
// GGML_CPU_ALL_VARIANTS @ 237ad9b96): the same kernel compiled at a wider ISA
// level and selected at runtime. We take that MODEL but not its MECHANISM (N
// shared libraries behind GGML_BACKEND_DL), because we ship one static library;
// instead this TU alone is built with -mavx2 -mf16c, exactly as
// CMakeLists.txt:744-760 already does for the Arm i8mm tier.
//
// Tier 1c — x86-64 AVX2. The SSE2 tier (cpu_matmul_elem.cpp:226) is 128-bit,
// so it produces 4 output columns per vector and needs 4 groups to cover
// kElemLanes=16. AVX2 is 256-bit: 8 output columns per vector, 2 groups.
//
// THE INVARIANT THAT MAKES THIS BIT-EXACT (E1-E4, and the reason the existing
// memcmp gate remains sufficient): widening adds OUTPUT LANES, never a split of
// the K reduction. Each lane still accumulates its own output over p in strict
// increasing order, so every output's addition sequence is identical to the
// portable tier's. Two consequences that are easy to get wrong:
//
//   1. Products are _mm256_mul_ps + _mm256_add_ps, NEVER _mm256_fmadd_ps. The
//      scalar reference rounds the product before the add (-ffp-contract=off,
//      CMakeLists.txt:21), and an FMA keeps the full-width product. Using FMA
//      here would be faster and WRONG, and the byte-identity test would catch
//      it. The AVX2 win is width, not FMA.
//   2. The 8x8 transpose must be a pure permutation. It moves values between
//      lanes; it must never combine them.
//
// bf16 widening is the shift-left-16 ggml uses (vec.cpp:172); f16 widening is
// the F16C hardware convert (_mm256_cvtph_ps), the same instruction the
// existing sse2+f16c tier already asserts bit-identical to vt::F16ToF32 over
// all 65,536 patterns.

#include <immintrin.h>

#include <cstdint>

#include "vt/dtype.h"
#include "cpu_matmul_elem.h"

namespace vt::cpu {
namespace {

// Mirrors Elem<> in cpu_matmul_elem.cpp: same types, same scalar converters,
// so the ragged tail below is bit-identical to the portable tier's tail.
template <ElemKind K>
struct ElemA;
template <>
struct ElemA<ElemKind::kF32> {
  using T = float;
  static inline float Cvt(T v) { return v; }
};
template <>
struct ElemA<ElemKind::kF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return F16ToF32(v); }
};
template <>
struct ElemA<ElemKind::kBF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return BF16ToF32(v); }
};

// Load 8 consecutive weight values of one output row, widened to f32.
template <ElemKind K>
inline __m256 LoadX8(const typename ElemA<K>::T* p);
template <>
inline __m256 LoadX8<ElemKind::kF32>(const float* p) {
  return _mm256_loadu_ps(p);
}
template <>
inline __m256 LoadX8<ElemKind::kBF16>(const uint16_t* p) {
  // shift-left-16 into the f32 exponent/mantissa position (ggml vec.cpp:172)
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}
template <>
inline __m256 LoadX8<ElemKind::kF16>(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// 8x8 transpose, pure lane permutation. After it, r[j] holds weight element
// p+j across the 8 output rows, which is what keeps each lane's accumulation
// in p order.
inline void Transpose8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4,
                       __m256& r5, __m256& r6, __m256& r7) {
  __m256 t0 = _mm256_unpacklo_ps(r0, r1);
  __m256 t1 = _mm256_unpackhi_ps(r0, r1);
  __m256 t2 = _mm256_unpacklo_ps(r2, r3);
  __m256 t3 = _mm256_unpackhi_ps(r2, r3);
  __m256 t4 = _mm256_unpacklo_ps(r4, r5);
  __m256 t5 = _mm256_unpackhi_ps(r4, r5);
  __m256 t6 = _mm256_unpacklo_ps(r6, r7);
  __m256 t7 = _mm256_unpackhi_ps(r6, r7);
  __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
  __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
  __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
  __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
  __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
  r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
  r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
  r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
  r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
  r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
  r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
  r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// Broadcast activation element p+j of an 8-wide chunk. `j` is a compile-time
// index so the shuffle/permute pair folds to constants.
template <int J>
inline __m256 BroadcastLane(__m256 v) {
  const __m256 half = _mm256_permute2f128_ps(v, v, (J < 4) ? 0x00 : 0x11);
  return _mm256_permute_ps(half, _MM_SHUFFLE(J % 4, J % 4, J % 4, J % 4));
}

// acc[l] = sum_p af[p] * B[l*k + p], 16 outputs as 2 groups of 8.
template <ElemKind K>
void Bt16Avx2(const float* af, const void* bv, int64_t k, float* acc) {
  using T = typename ElemA<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m256 A[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
  int64_t p = 0;
  for (; p + 8 <= k; p += 8) {
    const __m256 av = _mm256_loadu_ps(af + p);
    for (int g = 0; g < 2; ++g) {
      const T* br = b + static_cast<int64_t>(8 * g) * k + p;
      __m256 r0 = LoadX8<K>(br);
      __m256 r1 = LoadX8<K>(br + k);
      __m256 r2 = LoadX8<K>(br + 2 * k);
      __m256 r3 = LoadX8<K>(br + 3 * k);
      __m256 r4 = LoadX8<K>(br + 4 * k);
      __m256 r5 = LoadX8<K>(br + 5 * k);
      __m256 r6 = LoadX8<K>(br + 6 * k);
      __m256 r7 = LoadX8<K>(br + 7 * k);
      Transpose8(r0, r1, r2, r3, r4, r5, r6, r7);
      __m256 s = A[g];
      // Strict p order, mul-then-add. See the invariant note at the top.
      s = _mm256_add_ps(s, _mm256_mul_ps(r0, BroadcastLane<0>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r1, BroadcastLane<1>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r2, BroadcastLane<2>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r3, BroadcastLane<3>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r4, BroadcastLane<4>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r5, BroadcastLane<5>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r6, BroadcastLane<6>(av)));
      s = _mm256_add_ps(s, _mm256_mul_ps(r7, BroadcastLane<7>(av)));
      A[g] = s;
    }
  }
  for (int g = 0; g < 2; ++g) _mm256_storeu_ps(acc + 8 * g, A[g]);
  // Ragged tail, scalar and in the same p order.
  for (; p < k; ++p) {
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      acc[l] += av * ElemA<K>::Cvt(b[static_cast<int64_t>(l) * k + p]);
    }
  }
}

// acc[l] = sum_p af[p] * B[p*n + l]; weight rows are strided, so the 16 outputs
// are already contiguous per p and no transpose is needed.
template <ElemKind K>
void Nk16Avx2(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  using T = typename ElemA<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m256 A[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
  for (int64_t p = 0; p < k; ++p) {
    const __m256 av = _mm256_set1_ps(af[p]);
    const T* row = b + p * n;
    A[0] = _mm256_add_ps(A[0], _mm256_mul_ps(LoadX8<K>(row), av));
    A[1] = _mm256_add_ps(A[1], _mm256_mul_ps(LoadX8<K>(row + 8), av));
  }
  for (int g = 0; g < 2; ++g) _mm256_storeu_ps(acc + 8 * g, A[g]);
}

// M-blocked [N,K] kernel. MR=4: AVX2 has 16 YMM registers, and 4 rows x 2
// accumulator groups plus the 8 transposed weight vectors fits without
// spilling (SSE2 could only afford MR=2 with its 16 XMM).
constexpr int kMrAvx2 = 4;
static_assert(kMrAvx2 <= kElemMaxMr, "cpu_ops.cpp's accumulator tile bounds mr");

template <ElemKind K>
void BtM4Avx2(const float* af, int64_t a_stride, const void* bv, int64_t k, float* acc) {
  using T = typename ElemA<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m256 A[kMrAvx2][2];
  for (int r = 0; r < kMrAvx2; ++r) {
    A[r][0] = _mm256_setzero_ps();
    A[r][1] = _mm256_setzero_ps();
  }
  int64_t p = 0;
  for (; p + 8 <= k; p += 8) {
    for (int g = 0; g < 2; ++g) {
      const T* br = b + static_cast<int64_t>(8 * g) * k + p;
      __m256 w0 = LoadX8<K>(br);
      __m256 w1 = LoadX8<K>(br + k);
      __m256 w2 = LoadX8<K>(br + 2 * k);
      __m256 w3 = LoadX8<K>(br + 3 * k);
      __m256 w4 = LoadX8<K>(br + 4 * k);
      __m256 w5 = LoadX8<K>(br + 5 * k);
      __m256 w6 = LoadX8<K>(br + 6 * k);
      __m256 w7 = LoadX8<K>(br + 7 * k);
      Transpose8(w0, w1, w2, w3, w4, w5, w6, w7);
      // The transposed weights are shared by every activation row: this is the
      // whole point of M blocking, and it changes no output's order.
      for (int r = 0; r < kMrAvx2; ++r) {
        const __m256 av = _mm256_loadu_ps(af + r * a_stride + p);
        __m256 s = A[r][g];
        s = _mm256_add_ps(s, _mm256_mul_ps(w0, BroadcastLane<0>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w1, BroadcastLane<1>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w2, BroadcastLane<2>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w3, BroadcastLane<3>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w4, BroadcastLane<4>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w5, BroadcastLane<5>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w6, BroadcastLane<6>(av)));
        s = _mm256_add_ps(s, _mm256_mul_ps(w7, BroadcastLane<7>(av)));
        A[r][g] = s;
      }
    }
  }
  for (int r = 0; r < kMrAvx2; ++r) {
    for (int g = 0; g < 2; ++g) _mm256_storeu_ps(acc + r * kElemLanes + 8 * g, A[r][g]);
  }
  for (; p < k; ++p) {
    for (int r = 0; r < kMrAvx2; ++r) {
      const float av = af[r * a_stride + p];
      for (int l = 0; l < kElemLanes; ++l) {
        acc[r * kElemLanes + l] +=
            av * ElemA<K>::Cvt(b[static_cast<int64_t>(l) * k + p]);
      }
    }
  }
}

// M-blocked [K,N]. No transpose, so the 2 weight vectors per p are simply
// reused across kMrAvx2 activation rows. Same per-output p order.
template <ElemKind K>
void NkM4Avx2(const float* af, int64_t a_stride, const void* bv, int64_t k, int64_t n,
              float* acc) {
  using T = typename ElemA<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m256 A[kMrAvx2][2];
  for (int r = 0; r < kMrAvx2; ++r) {
    A[r][0] = _mm256_setzero_ps();
    A[r][1] = _mm256_setzero_ps();
  }
  for (int64_t p = 0; p < k; ++p) {
    const T* row = b + p * n;
    const __m256 w0 = LoadX8<K>(row);
    const __m256 w1 = LoadX8<K>(row + 8);
    for (int r = 0; r < kMrAvx2; ++r) {
      const __m256 av = _mm256_set1_ps(af[r * a_stride + p]);
      A[r][0] = _mm256_add_ps(A[r][0], _mm256_mul_ps(w0, av));
      A[r][1] = _mm256_add_ps(A[r][1], _mm256_mul_ps(w1, av));
    }
  }
  for (int r = 0; r < kMrAvx2; ++r) {
    _mm256_storeu_ps(acc + r * kElemLanes, A[r][0]);
    _mm256_storeu_ps(acc + r * kElemLanes + 8, A[r][1]);
  }
}

}  // namespace

void FillAvx2Tier(ElemGemmTierTable* t) {
  constexpr int kF32i = static_cast<int>(ElemKind::kF32);
  constexpr int kF16i = static_cast<int>(ElemKind::kF16);
  constexpr int kBF16i = static_cast<int>(ElemKind::kBF16);
  t->bt[kF32i] = &Bt16Avx2<ElemKind::kF32>;
  t->bt[kF16i] = &Bt16Avx2<ElemKind::kF16>;
  t->bt[kBF16i] = &Bt16Avx2<ElemKind::kBF16>;
  t->nk[kF32i] = &Nk16Avx2<ElemKind::kF32>;
  t->nk[kF16i] = &Nk16Avx2<ElemKind::kF16>;
  t->nk[kBF16i] = &Nk16Avx2<ElemKind::kBF16>;
  t->btm[kF32i] = &BtM4Avx2<ElemKind::kF32>;
  t->btm[kF16i] = &BtM4Avx2<ElemKind::kF16>;
  t->btm[kBF16i] = &BtM4Avx2<ElemKind::kBF16>;
  t->nkm[kF32i] = &NkM4Avx2<ElemKind::kF32>;
  t->nkm[kF16i] = &NkM4Avx2<ElemKind::kF16>;
  t->nkm[kBF16i] = &NkM4Avx2<ElemKind::kBF16>;
  t->mr = kMrAvx2;
  t->name = "avx2";
}

}  // namespace vt::cpu
