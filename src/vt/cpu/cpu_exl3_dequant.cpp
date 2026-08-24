// EXL3 (exllamav3 trellis) reference dequant — CPU tier. Row MODEL-DSV4-EXL3 W1a.
//
// PORTED 1:1 FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT).
// vLLM implements no EXL3 at the parity pin (`layers/quantization/` carries no
// exl3/exllamav3/trellis), so exllamav3 is the secondary oracle for this format
// per AGENTS.md "When vLLM has no implementation"; see
// `.agents/specs/model-dsv4-exl3.md`.
//
//   exllamav3_ext/quant/exl3_dq.cuh:15-31   `dq<bits, cb>` — the 16-bit
//     tail-biting window read. Weight t's codeword starts at bit
//     `t*bits + bits - 16` modulo `256*bits` inside the tile, and the two
//     uint32 words that straddle it are funnel-shifted into place.
//   exllamav3_ext/quant/pack.cu:29-57       the ENCODE side, which is what
//     makes that read work: 16 spans of 16 weights, each span written
//     MSB-first into a 32-bit buffer and flushed 16 bits at a time, then
//     `SWAP16` (`:56`) on every uint32 pair. The swap is exactly what turns
//     the stored little-endian int16 array into a BIG-ENDIAN bit stream when
//     read through a uint32 view, which is the view `dq` uses.
//   exllamav3_ext/quant/codebook.cuh:67-75  `decode_3inst<1>` — the MCG
//     codebook. `lop3.b32 ... 0x6a` is `(a & b) ^ c` (immLut 0x6a with the
//     canonical a=0xF0, b=0xCC, c=0xAA), so the three instructions are
//     `x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` and the two fp16
//     halves are then summed in fp16 (`__hadd`).
//   exllamav3/modules/quant/exl3_lib/quantize.py:22-42  `tensor_core_perm`,
//     the row-major -> codeword-order permutation the quantizer applies to each
//     16x16 tile (`quantize.py:574,976`). It is the inverse of the tensor-core
//     fragment shuffle `reconstruct.cu:46-74` performs on the way out.
//   exllamav3/modules/quant/exl3.py:222-237 `get_inner_weight_tensor` and
//     `get_weight_tensor`, the two functions this file mirrors.
//   exllamav3/modules/quant/exl3_lib/quantize.py:15,340-358  `had_k = had_n =
//     128` and `preapply_had_l` / `preapply_had_r`, both scaled by
//     `1/sqrt(had_dim)` and both BLOCKWISE over the 128-wide axis.
//   exllamav3/util/hadamard.py:34-42,107-123  the Hadamard itself: no
//     `hadamard_128.txt` ships, so `get_hadamard(128)` recurses Sylvester down
//     to `hadamard_1.txt` = "+". A Sylvester Hadamard in natural order is what
//     the fast Walsh-Hadamard butterfly below computes, exactly.
//   exllamav3_ext/quant/hadamard.cu:88-110 + quant/hadamard_inner.cuh
//     (`shuffle_had_f4x32`)  the INFERENCE Hadamard `had_r_128`, which row
//     MODEL-DSV4-EXL3 W2a ports. Not a different transform — see the note in
//     `Exl3DequantLinear` below.
//
// The mcg int32 tensor each linear also stores is a codebook MARKER and is
// never read at inference (`quantize.py:1414-1424`); the caller checks it and
// selects the codebook, it is not an argument here.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt {
namespace {

// `had_k = had_n = 128` (quantize.py:15). Not a tunable: the quantizer baked
// this block size into the stored weights.
constexpr int kHadDim = 128;

// The tile's uint32 view (exl3_dq.cuh:25-26 `ptr[...]`), assembled by hand so
// the trellis may sit at any alignment inside a safetensors mmap.
inline uint32_t TileWord32(const uint16_t* tile, int index) {
  return static_cast<uint32_t>(tile[2 * index]) |
         (static_cast<uint32_t>(tile[2 * index + 1]) << 16);
}

inline float RoundHalf(float v) { return F16ToF32(F32ToF16(v)); }

// Fast Walsh-Hadamard transform over `kHadDim` elements strided by `stride`
// floats, repeated for `lanes` contiguous lanes at each element. Computes
// H @ x for the natural-order Sylvester H (hadamard.py:34-42) exactly.
void Fwht128(float* base, int64_t stride, int64_t lanes) {
  for (int len = 1; len < kHadDim; len <<= 1) {
    for (int i = 0; i < kHadDim; i += (len << 1)) {
      for (int j = 0; j < len; ++j) {
        float* a = base + static_cast<int64_t>(i + j) * stride;
        float* b = base + static_cast<int64_t>(i + j + len) * stride;
        for (int64_t c = 0; c < lanes; ++c) {
          const float u = a[c];
          const float v = b[c];
          a[c] = u + v;
          b[c] = u - v;
        }
      }
    }
  }
}

}  // namespace

uint16_t Exl3TileCodeword(const uint16_t* tile, int bits, int t) {
  // exl3_dq.cuh:18-29, verbatim. `+ 256*bits` is upstream's way of keeping the
  // tail-biting wrap non-negative; the `% (bits*256/32)` on the word index is
  // the wrap itself.
  const int words32 = bits * 256 / 32;
  const int b0 = t * bits + bits - 16 + 256 * bits;
  const int b1 = b0 + 16;
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  const uint32_t a = TileWord32(tile, i0 % words32);
  const uint32_t b = TileWord32(tile, i1 % words32);
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint16_t>((merged >> s0) & 0xffffu);
}

float Exl3DecodeMcg(uint16_t codeword) {
  // codebook.cuh:67-75, cb == 1.
  uint32_t x = static_cast<uint32_t>(codeword) * 0xCBAC1FEDu;
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = F16ToF32(static_cast<uint16_t>(x & 0xffffu));
  const float hi = F16ToF32(static_cast<uint16_t>(x >> 16));
  return RoundHalf(lo + hi);  // __hadd: the sum is taken in fp16
}

int Exl3TileRowMajorIndex(int t) {
  // quantize.py:28-42. `t / 8` is the tensor-core lane, `t % 8` its eight
  // fragment slots.
  const int lane = t >> 3;
  const int sub = t & 7;
  static const int kRowOffset[8] = {0, 1, 8, 9, 0, 1, 8, 9};
  const int r = (lane % 4) * 2 + kRowOffset[sub];
  const int c = lane / 4 + (sub < 4 ? 0 : 8);
  return r * 16 + c;
}

void Exl3DecodeTile(const uint16_t* tile, int bits, float* out256) {
  VT_CHECK(bits >= 1 && bits <= 8,
           "exl3: bits must be in [1, 8]; got " + std::to_string(bits));
  for (int t = 0; t < 256; ++t) {
    out256[Exl3TileRowMajorIndex(t)] = Exl3DecodeMcg(Exl3TileCodeword(tile, bits, t));
  }
}

void Exl3ReconstructInner(const uint16_t* trellis, int64_t k, int64_t n, int bits,
                          float* out) {
  VT_CHECK(bits >= 1 && bits <= 8,
           "exl3: bits must be in [1, 8]; got " + std::to_string(bits));
  VT_CHECK(k > 0 && k % 16 == 0 && n > 0 && n % 16 == 0,
           "exl3: in/out features must be positive multiples of 16 (the trellis "
           "tile is 16x16); got k=" + std::to_string(k) + " n=" + std::to_string(n));
  const int64_t tiles_k = k / 16;
  const int64_t tiles_n = n / 16;
  const int64_t tile_words = 16 * bits;
  float tile_out[256];
  for (int64_t i = 0; i < tiles_k; ++i) {
    for (int64_t j = 0; j < tiles_n; ++j) {
      Exl3DecodeTile(trellis + (i * tiles_n + j) * tile_words, bits, tile_out);
      for (int r = 0; r < 16; ++r) {
        std::memcpy(out + (i * 16 + r) * n + j * 16, tile_out + r * 16,
                    16 * sizeof(float));
      }
    }
  }
}

void Exl3DequantLinear(const uint16_t* trellis, const uint16_t* suh,
                       const uint16_t* svh, int64_t k, int64_t n, int bits,
                       float* out) {
  VT_CHECK(k % kHadDim == 0 && n % kHadDim == 0,
           "exl3: both features must be multiples of 128 (each side was "
           "Hadamard-128 transformed at quantization time, "
           "exl3_lib/quantize.py:15); got k=" + std::to_string(k) +
               " n=" + std::to_string(n));
  Exl3ReconstructInner(trellis, k, n, bits, out);

  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHadDim)));

  // FOR W2a, WHICH PORTS THE INFERENCE HADAMARD — read this before assuming the
  // two sides diverge. What follows is upstream's QUANTIZE-time
  // `preapply_had_l`/`preapply_had_r` (quantize.py:340-356): a per-128-block
  // Sylvester butterfly whose result is rounded back to fp16 at each stage by
  // `.to(half)`. Upstream's INFERENCE Hadamard is the SAME butterfly:
  // `had_r_128` (quant/hadamard.cu:88-110, `shuffle_had_f4x32` in
  // quant/hadamard_inner.cuh) runs it and scales once at the END by
  // `r_scale = scale * 0.088388347648f` — and 0.088388347648 IS 1/sqrt(128),
  // the `scale` above. So the structure of this reference CONVERGES on the
  // kernel W2a will port rather than diverging from it.
  //
  // Two differences are left to reconcile, and both are numerical rather than
  // structural. WHERE the scale lands: per 128-block with an fp16 round here,
  // once at the end there. WHAT it is applied to: the WEIGHTS here, the
  // ACTIVATIONS there (`had_r_128(x, suh)` in, `had_r_128(y, svh)` out,
  // exl3.py:183-214) — which is upstream's whole point, since the two are
  // equivalent only up to summation order. Anything that SUMS therefore needs
  // W2's parity gate to decide an ulp bound in its own spec, not to discover
  // one when the gate first reds.
  //
  // The `hadamard.cu`/`hadamard_inner.cuh` reading is this row's fresh review
  // of 2026-08-24 at the anchors named; W2a re-reads them as it ports.

  // preapply_had_l(w, 128) (quantize.py:340-347): blockwise over the k axis,
  // every column transformed together, then rounded back to fp16 by `.to(half)`.
  for (int64_t b = 0; b < k; b += kHadDim) {
    Fwht128(out + b * n, /*stride=*/n, /*lanes=*/n);
    for (int64_t i = 0; i < kHadDim; ++i) {
      float* row = out + (b + i) * n;
      for (int64_t j = 0; j < n; ++j) row[j] = RoundHalf(row[j] * scale);
    }
  }

  // w *= suh[:, None] (exl3.py:233) — an fp16 multiply.
  for (int64_t i = 0; i < k; ++i) {
    const float s = F16ToF32(suh[i]);
    float* row = out + i * n;
    for (int64_t j = 0; j < n; ++j) row[j] = RoundHalf(row[j] * s);
  }

  // preapply_had_r(w, 128) (quantize.py:349-356): blockwise over the n axis.
  // The Sylvester Hadamard is symmetric, so `x @ H` down a row is the same
  // butterfly as `H @ x` down a column.
  for (int64_t i = 0; i < k; ++i) {
    float* row = out + i * n;
    for (int64_t b = 0; b < n; b += kHadDim) {
      Fwht128(row + b, /*stride=*/1, /*lanes=*/1);
      for (int64_t j = 0; j < kHadDim; ++j)
        row[b + j] = RoundHalf(row[b + j] * scale);
    }
  }

  // w *= svh[None, :] (exl3.py:235).
  for (int64_t i = 0; i < k; ++i) {
    float* row = out + i * n;
    for (int64_t j = 0; j < n; ++j) row[j] = RoundHalf(row[j] * F16ToF32(svh[j]));
  }
}

}  // namespace vt
