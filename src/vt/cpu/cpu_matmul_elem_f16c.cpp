#include "vt/cpu/cpu_matmul_elem_f16c.h"

#include "vt/quant.h"
#include "vt/unaligned.h"

#include <immintrin.h>

#include <cstdint>

namespace vt::cpu {
namespace {

constexpr int kMrSse2 = 2;
constexpr int kF16 = static_cast<int>(ElemKind::kF16);

void BtM2F16c(const float* af, int64_t a_stride, const void* bv, int64_t k,
              float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 accum[kMrSse2][4];
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) accum[r][g] = _mm_setzero_ps();
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
        __m128 sum = accum[r][g];
        sum = _mm_add_ps(sum, _mm_mul_ps(w0, _mm_shuffle_ps(av, av, 0x00)));
        sum = _mm_add_ps(sum, _mm_mul_ps(w1, _mm_shuffle_ps(av, av, 0x55)));
        sum = _mm_add_ps(sum, _mm_mul_ps(w2, _mm_shuffle_ps(av, av, 0xAA)));
        sum = _mm_add_ps(sum, _mm_mul_ps(w3, _mm_shuffle_ps(av, av, 0xFF)));
        accum[r][g] = sum;
      }
    }
  }
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) {
      _mm_storeu_ps(acc + r * kElemLanes + 4 * g, accum[r][g]);
    }
  }
  for (int64_t pt = p; pt < k; ++pt) {
    for (int r = 0; r < kMrSse2; ++r) {
      const float av = af[r * a_stride + pt];
      for (int lane = 0; lane < kElemLanes; ++lane) {
        acc[r * kElemLanes + lane] +=
            av * F16ToF32(vt::LoadUnaligned<uint16_t>(b + static_cast<int64_t>(lane) * k + pt));
      }
    }
  }
}

void Bt16F16c(const float* af, const void* bv, int64_t k, float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 accum[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(),
                     _mm_setzero_ps()};
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
      __m128 sum = accum[g];
      sum = _mm_add_ps(sum, _mm_mul_ps(r0, _mm_shuffle_ps(av, av, 0x00)));
      sum = _mm_add_ps(sum, _mm_mul_ps(r1, _mm_shuffle_ps(av, av, 0x55)));
      sum = _mm_add_ps(sum, _mm_mul_ps(r2, _mm_shuffle_ps(av, av, 0xAA)));
      sum = _mm_add_ps(sum, _mm_mul_ps(r3, _mm_shuffle_ps(av, av, 0xFF)));
      accum[g] = sum;
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, accum[g]);
  for (; p < k; ++p) {
    const float av = af[p];
    for (int lane = 0; lane < kElemLanes; ++lane) {
      acc[lane] += av * F16ToF32(vt::LoadUnaligned<uint16_t>(b + static_cast<int64_t>(lane) * k + p));
    }
  }
}

void Nk16F16c(const float* af, const void* bv, int64_t k, int64_t n, float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 accum[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(),
                     _mm_setzero_ps()};
  for (int64_t p = 0; p < k; ++p) {
    const __m128 av = _mm_set1_ps(af[p]);
    const uint16_t* row = b + p * n;
    for (int g = 0; g < 4; ++g) {
      const __m128 values =
          _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + 4 * g)));
      accum[g] = _mm_add_ps(accum[g], _mm_mul_ps(values, av));
    }
  }
  for (int g = 0; g < 4; ++g) _mm_storeu_ps(acc + 4 * g, accum[g]);
}

void NkM2F16c(const float* af, int64_t a_stride, const void* bv, int64_t k,
              int64_t n, float* acc) {
  const uint16_t* b = static_cast<const uint16_t*>(bv);
  __m128 accum[kMrSse2][4];
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) accum[r][g] = _mm_setzero_ps();
  }
  for (int64_t p = 0; p < k; ++p) {
    const uint16_t* row = b + p * n;
    __m128 weights[4];
    for (int g = 0; g < 4; ++g) {
      weights[g] =
          _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + 4 * g)));
    }
    for (int r = 0; r < kMrSse2; ++r) {
      const __m128 av = _mm_set1_ps(af[r * a_stride + p]);
      for (int g = 0; g < 4; ++g) {
        accum[r][g] = _mm_add_ps(accum[r][g], _mm_mul_ps(weights[g], av));
      }
    }
  }
  for (int r = 0; r < kMrSse2; ++r) {
    for (int g = 0; g < 4; ++g) {
      _mm_storeu_ps(acc + r * kElemLanes + 4 * g, accum[r][g]);
    }
  }
}

}  // namespace

void FillF16cTier(ElemGemmTierTable* table) {
  table->bt[kF16] = &Bt16F16c;
  table->nk[kF16] = &Nk16F16c;
  table->btm[kF16] = &BtM2F16c;
  table->nkm[kF16] = &NkM2F16c;
  table->name = "sse2+f16c";
}

}  // namespace vt::cpu
