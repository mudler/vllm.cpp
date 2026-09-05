// vllm.cpp original (KERNEL-GEMM-CPU-ELEM-X86WIDE work row W2). Sibling of
// cpu_matmul_elem_avx2.cpp; same build model (this TU alone is compiled at the
// wider ISA, entry gated on a runtime probe), same bit-exactness invariant.
//
// Tier 1d — x86-64 AVX-512F. kElemLanes is 16 and a ZMM holds exactly 16 f32,
// so the 16 outputs collapse into ONE accumulator and one 16x16 transpose,
// against AVX2's two groups of 8 and SSE2's four groups of 4.
//
// SAME INVARIANT AS EVERY OTHER TIER: widening adds OUTPUT LANES, never a split
// of the K reduction. Lane l accumulates output l over p in strict increasing
// order, so the addition sequence is identical to the portable tier's and the
// memcmp byte-identity gate stays valid. In particular products are
// _mm512_mul_ps + _mm512_add_ps and NEVER _mm512_fmadd_ps: the scalar reference
// rounds the product before the add (-ffp-contract=off, CMakeLists.txt:21).
// llamafile's sgemm does use FMA (sgemm.cpp:147), which is part of why it is
// faster and is exactly the trade we are not making here.

#include <immintrin.h>

#include <cstdint>

#include "vt/dtype.h"
#include "cpu_matmul_elem.h"

namespace vt::cpu {
namespace {

template <ElemKind K>
struct ElemZ;
template <>
struct ElemZ<ElemKind::kF32> {
  using T = float;
  static inline float Cvt(T v) { return v; }
};
template <>
struct ElemZ<ElemKind::kF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return F16ToF32(v); }
};
template <>
struct ElemZ<ElemKind::kBF16> {
  using T = uint16_t;
  static inline float Cvt(T v) { return BF16ToF32(v); }
};

// Load 16 consecutive weight values of one output row, widened to f32.
template <ElemKind K>
inline __m512 LoadX16(const typename ElemZ<K>::T* p);
template <>
inline __m512 LoadX16<ElemKind::kF32>(const float* p) {
  return _mm512_loadu_ps(p);
}
template <>
inline __m512 LoadX16<ElemKind::kBF16>(const uint16_t* p) {
  // shift-left-16, the same widening ggml uses (vec.cpp:172)
  const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
template <>
inline __m512 LoadX16<ElemKind::kF16>(const uint16_t* p) {
  return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

// 16x16 f32 transpose, pure permutation (moves values between lanes, never
// combines them). Four stages: unpack pairs, shuffle 64-bit halves, then two
// rounds of 128-bit lane shuffles. After it, r[j] holds weight element p+j
// across the 16 output rows, which is what keeps each lane in p order.
inline void Transpose16(__m512* r) {
  __m512 t[16];
  for (int i = 0; i < 8; ++i) {
    t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  __m512 s[16];
  for (int i = 0; i < 4; ++i) {
    s[4 * i + 0] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0x44);
    s[4 * i + 1] = _mm512_shuffle_ps(t[4 * i + 0], t[4 * i + 2], 0xEE);
    s[4 * i + 2] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0x44);
    s[4 * i + 3] = _mm512_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0xEE);
  }
  // 128-bit lane gather: combine quads 0/1 and 2/3, then the halves.
  for (int i = 0; i < 8; ++i) {
    const int lo = (i < 4) ? i : (i - 4);
    const int a = (i < 4) ? lo : lo + 8;
    t[i] = _mm512_shuffle_f32x4(s[a], s[a + 4], 0x88);
    t[i + 8] = _mm512_shuffle_f32x4(s[a], s[a + 4], 0xDD);
  }
  for (int i = 0; i < 4; ++i) {
    r[i] = _mm512_shuffle_f32x4(t[i], t[i + 4], 0x88);
    r[i + 4] = _mm512_shuffle_f32x4(t[i + 8], t[i + 12], 0x88);
    r[i + 8] = _mm512_shuffle_f32x4(t[i], t[i + 4], 0xDD);
    r[i + 12] = _mm512_shuffle_f32x4(t[i + 8], t[i + 12], 0xDD);
  }
}

// Broadcast activation element p+J of a 16-wide chunk.
template <int J>
inline __m512 BroadcastLane16(const float* af) {
  return _mm512_set1_ps(af[J]);
}

template <ElemKind K>
void Bt16Avx512(const float* af, const void* bv, int64_t k, float* acc) {
  using T = typename ElemZ<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m512 A = _mm512_setzero_ps();
  int64_t p = 0;
  for (; p + 16 <= k; p += 16) {
    __m512 r[16];
    for (int l = 0; l < 16; ++l) r[l] = LoadX16<K>(b + static_cast<int64_t>(l) * k + p);
    Transpose16(r);
    // Strict p order, mul-then-add. See the invariant note at the top.
    for (int j = 0; j < 16; ++j) {
      A = _mm512_add_ps(A, _mm512_mul_ps(r[j], _mm512_set1_ps(af[p + j])));
    }
  }
  _mm512_storeu_ps(acc, A);
  for (; p < k; ++p) {
    const float av = af[p];
    for (int l = 0; l < kElemLanes; ++l) {
      acc[l] += av * ElemZ<K>::Cvt(b[static_cast<int64_t>(l) * k + p]);
    }
  }
}

// [K,N] orientation: the 16 outputs are already contiguous per p, no transpose.
template <ElemKind K>
void Nk16Avx512(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  using T = typename ElemZ<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m512 A = _mm512_setzero_ps();
  for (int64_t p = 0; p < k; ++p) {
    A = _mm512_add_ps(A, _mm512_mul_ps(LoadX16<K>(b + p * n), _mm512_set1_ps(af[p])));
  }
  _mm512_storeu_ps(acc, A);
}

// M-blocked [N,K]. MR=6: AVX-512 has 32 ZMM, and the 16 transposed weight
// vectors plus 6 accumulators plus a broadcast still fits without spilling.
// The transpose is the dominant cost and is shared across all MR rows, which
// is the whole point of M blocking and changes no output's order.
// MR = 8 rather than 6 so a full 16-row activation tile is exactly TWO
// M-blocked calls. At 6 the same tile needed three, and the four rows the third
// call could not cover fell through to the one-row kernel -- one whole weight
// load and one whole 16x16 transpose each, six passes over the weight block
// where the blocking factor allows two. AVX-512 has 32 ZMM and this kernel
// holds 16 transposed weight vectors plus MR accumulators plus one broadcast:
// 25 at MR = 8, against 23 at MR = 6.
constexpr int kMrAvx512 = 8;
static_assert(kMrAvx512 <= kElemMaxMr, "cpu_ops.cpp's accumulator tile bounds mr");

template <ElemKind K>
void BtM6Avx512(const float* af, int64_t a_stride, const void* bv, int64_t k, float* acc) {
  using T = typename ElemZ<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m512 A[kMrAvx512];
  for (int r0 = 0; r0 < kMrAvx512; ++r0) A[r0] = _mm512_setzero_ps();
  int64_t p = 0;
  for (; p + 16 <= k; p += 16) {
    __m512 w[16];
    for (int l = 0; l < 16; ++l) w[l] = LoadX16<K>(b + static_cast<int64_t>(l) * k + p);
    Transpose16(w);
    for (int r0 = 0; r0 < kMrAvx512; ++r0) {
      const float* ar = af + r0 * a_stride;
      __m512 s = A[r0];
      for (int j = 0; j < 16; ++j) {
        s = _mm512_add_ps(s, _mm512_mul_ps(w[j], _mm512_set1_ps(ar[p + j])));
      }
      A[r0] = s;
    }
  }
  for (int r0 = 0; r0 < kMrAvx512; ++r0) _mm512_storeu_ps(acc + r0 * kElemLanes, A[r0]);
  for (; p < k; ++p) {
    for (int r0 = 0; r0 < kMrAvx512; ++r0) {
      const float av = af[r0 * a_stride + p];
      for (int l = 0; l < kElemLanes; ++l) {
        acc[r0 * kElemLanes + l] += av * ElemZ<K>::Cvt(b[static_cast<int64_t>(l) * k + p]);
      }
    }
  }
}

// M-blocked [K,N]. One 16-wide weight load per p, reused across kMrAvx512 rows,
// no transpose. Same per-output p order.
template <ElemKind K>
void NkM6Avx512(const float* af, int64_t a_stride, const void* bv, int64_t k, int64_t n,
                float* acc) {
  using T = typename ElemZ<K>::T;
  const T* b = static_cast<const T*>(bv);
  __m512 A[kMrAvx512];
  for (int r = 0; r < kMrAvx512; ++r) A[r] = _mm512_setzero_ps();
  for (int64_t p = 0; p < k; ++p) {
    const __m512 w = LoadX16<K>(b + p * n);
    for (int r = 0; r < kMrAvx512; ++r) {
      A[r] = _mm512_add_ps(A[r], _mm512_mul_ps(w, _mm512_set1_ps(af[r * a_stride + p])));
    }
  }
  for (int r = 0; r < kMrAvx512; ++r) _mm512_storeu_ps(acc + r * kElemLanes, A[r]);
}

}  // namespace

void FillAvx512Tier(ElemGemmTierTable* t) {
  constexpr int kF32i = static_cast<int>(ElemKind::kF32);
  constexpr int kF16i = static_cast<int>(ElemKind::kF16);
  constexpr int kBF16i = static_cast<int>(ElemKind::kBF16);
  t->bt[kF32i] = &Bt16Avx512<ElemKind::kF32>;
  t->bt[kF16i] = &Bt16Avx512<ElemKind::kF16>;
  t->bt[kBF16i] = &Bt16Avx512<ElemKind::kBF16>;
  t->nk[kF32i] = &Nk16Avx512<ElemKind::kF32>;
  t->nk[kF16i] = &Nk16Avx512<ElemKind::kF16>;
  t->nk[kBF16i] = &Nk16Avx512<ElemKind::kBF16>;
  t->btm[kF32i] = &BtM6Avx512<ElemKind::kF32>;
  t->btm[kF16i] = &BtM6Avx512<ElemKind::kF16>;
  t->btm[kBF16i] = &BtM6Avx512<ElemKind::kBF16>;
  t->nkm[kF32i] = &NkM6Avx512<ElemKind::kF32>;
  t->nkm[kF16i] = &NkM6Avx512<ElemKind::kF16>;
  t->nkm[kBF16i] = &NkM6Avx512<ElemKind::kBF16>;
  t->mr = kMrAvx512;
  t->name = "avx512";
}

}  // namespace vt::cpu
