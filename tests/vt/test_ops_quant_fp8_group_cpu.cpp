// vllm.cpp — dynamic per-token, per-group FP8 (e4m3fn) activation quant.
//
// VT-QUANT-FP8-GROUP (.agents/specs/vt-quant-fp8-group.md), issue #1189
// milestone M1. Pinned oracle: vLLM 5559679229bc961848b121ccdeaa8fa5d79bec98.
//
// WHICH UPSTREAM ARM THIS MIRRORS, because there are two and they disagree.
// `per_token_group_quant_fp8` reads like a Triton kernel with a C++ fast path.
// It is the other way round: on a CUDA-alike platform with a contiguous input
// it calls the C++ custom op and RETURNS
// (vllm/model_executor/layers/quantization/utils/fp8_utils.py:635-650), so the
// Triton kernel below it never executes there. The executing kernel is
//   csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu
//     :47  float local_absmax = eps            (eps SEEDS the reduction)
//     :53  fmaxf(local_absmax, fabsf((float)src))
//     :68  float y_s = local_absmax / max_8bit                  <- a DIVIDE
//     :85  fminf(fmaxf((float)src / y_s, min_8bit), max_8bit)   <- a DIVIDE
//     :86  DST_DTYPE(q)                        (hardware e4m3 RNE, saturating)
// The Triton fallback instead forms `scale_raw = _absmax * (1.0 / fp8_max)`
// (fp8_utils.py:145) under a comment that names the 1-ULP difference. One f32
// ULP before an e4m3 round changes the emitted byte near a tie.
//
// WHY G1 EXISTS ON TOP OF THE PORTED CASE, and this is MEASURED rather than
// argued. Upstream's own case compares values at rtol=0.15
// (tests/kernels/quantization/test_block_fp8.py:112-114) and the scale at
// torch.allclose's default rtol=1e-5 (:115). Two mutations of the CPU kernel to
// the Triton arm's form, each built (compile_rc=0) and run:
//   * `y_s = amax * (1.0f/448)` instead of `amax / 448`
//       G1 fails 49 of 146 assertions. G2 fails 6 of its 48 shape checks, ALL of
//       them the VALUE check and ALL at num_tokens=2050; the scale check never
//       fires, and no shape at num_tokens=7 fires at all.
//   * `x * (1.0f/y_s)` instead of `x / y_s`
//       G1 fails 14 of 146. G2 passes ENTIRELY, 50 of 50.
// So upstream's tolerances see one of the two forms, on the large shapes only,
// by luck of which elements land on an e4m3 boundary. They do not see the other
// at all. G1 sees both, on every shape, because it compares BYTES.
//
//   G1  BITWISE, ZERO TOLERANCE, against an INDEPENDENTLY WRITTEN reference.
//   G2  the ported upstream case (test_block_fp8.py:82-118) with its grid.
//   G3  the scale itself: amax/448 by construction, and the all-zero group.
//   G4  the shape and dtype contract of out_scale.
//   G5  the refusals, each by name.
//   G6  CPU vs CUDA byte identity. CUDA-gated; OWED on a host with no GPU.
//
// THE G1 REFERENCE IS DERIVED FROM THE FORMAT, never from a codec in src/,
// otherwise it would be a tautology dressed as a gate. `RefEncodeRne`
// enumerates all 128 finite e4m3fn magnitudes, decodes each to an exact double
// from the field layout, and picks the nearest with an even-significand
// tie-break by scanning. That is a different ALGORITHM from `F32ToFp8`
// (frexp + std::nearbyint). G2 needs the same encode 290M times, where an
// exhaustive scan is not affordable, so it uses `FastEncodeRne`, a binary
// search over the same table -- and G1 proves the two agree on every input
// class before G2 is allowed to rely on the fast one.
//
// Byte comparisons, not doctest Approx: Approx carries a `scale` term
// defaulting to 1.0 and therefore a ~1.19e-5 absolute floor, meaningless here.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

// --- the independent e4m3fn reference -------------------------------------
// Written from the format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa
// bits, no infinities, and 0x7F/0xFF the only NaN encodings ("fn").
constexpr float kFp8Max = 448.0f;   // = 1.75 * 2^8, encoding 0x7E
constexpr float kFp8Min = -448.0f;  // quant_utils.py:27-35 finfo(e4m3fn)
constexpr float kEps = 1e-10f;      // fp8_utils.py:570 default, the only value
                                    // any upstream call site passes

// Exact value of one finite e4m3fn magnitude encoding.
double E4m3Exact(unsigned exp_field, unsigned mant) {
  if (exp_field == 0) return std::ldexp(static_cast<double>(mant), -9);  // mant / 512
  return std::ldexp(1.0 + static_cast<double>(mant) / 8.0, static_cast<int>(exp_field) - 7);
}

// Round-to-nearest-EVEN encode by exhaustive nearest-value scan over the 128
// finite magnitudes. `r` must already be clamped to [-448, 448] and finite.
uint8_t RefEncodeRne(float r) {
  const auto sign = static_cast<uint8_t>(std::signbit(r) ? 0x80u : 0x00u);
  const double a = std::fabs(static_cast<double>(r));
  unsigned best_e = 0, best_m = 0;
  double best_d = std::numeric_limits<double>::infinity();
  for (unsigned e = 0; e <= 15; ++e) {
    for (unsigned m = 0; m <= 7; ++m) {
      if (e == 15 && m == 7) continue;  // the NaN encoding is not a value
      const double d = std::fabs(a - E4m3Exact(e, m));
      // Strictly nearer wins. On an EXACT tie prefer the even significand; the
      // tie-break carries across an exponent step too, because mant 7 (odd) at
      // exponent e is adjacent to mant 0 (even) at e+1.
      if (d < best_d || (d == best_d && (m & 1u) == 0u && (best_m & 1u) != 0u)) {
        best_d = d;
        best_e = e;
        best_m = m;
      }
    }
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(best_e << 3) |
                              static_cast<uint8_t>(best_m));
}

// The same encode in O(log n), for the 290M-element ported grid where the
// exhaustive scan is not affordable. The 127 finite magnitudes 0x00..0x7E are
// MONOTONIC in the byte value, so a binary search over their midpoints picks the
// nearest, and an exact midpoint hit resolves to the even mantissa. G1 proves
// this agrees with RefEncodeRne on every input class before G2 uses it.
const std::vector<double>& Magnitudes() {
  static const std::vector<double> table = [] {
    std::vector<double> v;
    v.reserve(127);
    for (unsigned e = 0; e <= 15; ++e)
      for (unsigned m = 0; m <= 7; ++m) {
        if (e == 15 && m == 7) continue;
        v.push_back(E4m3Exact(e, m));
      }
    return v;
  }();
  return table;
}

uint8_t FastEncodeRne(float r) {
  const auto sign = static_cast<uint8_t>(std::signbit(r) ? 0x80u : 0x00u);
  const double a = std::fabs(static_cast<double>(r));
  const std::vector<double>& mag = Magnitudes();
  // First index whose magnitude is >= a.
  const auto it = std::lower_bound(mag.begin(), mag.end(), a);
  size_t hi = static_cast<size_t>(it - mag.begin());
  if (hi == 0) return sign;                                    // a == 0
  if (hi >= mag.size()) return static_cast<uint8_t>(sign | 0x7Eu);  // a == 448
  const size_t lo = hi - 1;
  const double dlo = a - mag[lo], dhi = mag[hi] - a;
  size_t pick = 0;
  if (dlo < dhi) {
    pick = lo;
  } else if (dhi < dlo) {
    pick = hi;
  } else {
    // Table index == byte value, so `lo & 1` IS the mantissa's low bit.
    pick = (lo & 1u) == 0u ? lo : hi;  // even significand wins the tie
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(pick));
}

// Decode a raw e4m3fn byte back to an exact double, for the value comparison
// upstream's ported case makes. Table-driven because G2 decodes ~145M bytes
// twice each and an ldexp per call dominated the case's run time.
const std::vector<double>& DequantTable() {
  static const std::vector<double> table = [] {
    std::vector<double> v(256);
    for (unsigned b = 0; b < 256; ++b) {
      const double m = E4m3Exact((b >> 3) & 0xFu, b & 0x7u);
      v[b] = (b & 0x80u) != 0 ? -m : m;
    }
    return v;
  }();
  return table;
}

double RefDequant(uint8_t byte) { return DequantTable()[byte]; }

// bf16 round-trip, so a bf16 input reaches the reference at the SAME width the
// kernel loads it at. Written here rather than taken from vt/, for the same
// independence reason as RefEncodeRne: round-to-nearest-even on the low 16 bits.
float RoundToBf16(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)); }

// The whole upstream expression, in upstream's order and upstream's f32 width.
// `enc` selects the exhaustive or the binary-search encoder; the arithmetic
// above it is identical either way.
void RefQuantGroup(const float* x, int64_t n, uint8_t* out, float* out_scale,
                   uint8_t (*enc)(float)) {
  float amax = kEps;                                       // :47 eps SEEDS it
  for (int64_t i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(x[i]));  // :53
  const float y_s = amax / kFp8Max;                        // :68 a DIVIDE
  *out_scale = y_s;
  for (int64_t i = 0; i < n; ++i) {
    const float q = std::fmin(std::fmax(x[i] / y_s, kFp8Min), kFp8Max);  // :85
    out[i] = enc(q);                                                     // :86
  }
}

// --- inputs ----------------------------------------------------------------
// One ROW of G1 input. Every value class whose handling a mutation can break,
// sized so that the group's amax is known and the scaled values land on
// saturation, exact e4m3 ties, the subnormal ladder and both zeros.
std::vector<float> G1Row(int64_t k, uint32_t seed, float span) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(k));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-span, span);
  // The scaled value x/y_s spans [-448, 448] by construction, so these are the
  // ties and ladder points of the e4m3 grid, expressed in x units.
  const float unit = span / kFp8Max;
  for (int e = -4; e <= 8; ++e)
    for (int m = 0; m < 7; ++m) {
      const double lo = std::ldexp(1.0 + m / 8.0, e);
      const double hi = std::ldexp(1.0 + (m + 1) / 8.0, e);
      v.push_back(static_cast<float>((lo + hi) / 2.0) * unit);
      v.push_back(-static_cast<float>((lo + hi) / 2.0) * unit);
    }
  for (int m = 0; m <= 8; ++m) {
    v.push_back(static_cast<float>(m / 512.0) * unit);
    v.push_back(static_cast<float>((2 * m + 1) / 1024.0) * unit);
    v.push_back(-static_cast<float>((2 * m + 1) / 1024.0) * unit);
  }
  v.push_back(0.0f);
  v.push_back(-0.0f);
  v.push_back(span);   // the amax itself: this element must quantize to 0x7E
  v.push_back(-span);
  while (static_cast<int64_t>(v.size()) < k) v.push_back(ux(rng));
  v.resize(static_cast<size_t>(k));
  return v;
}

// Runs the CPU op and returns the number of differing OUTPUT bytes plus the
// number of differing scale words. `x_dtype` selects the input width; the
// reference consumes the same width, so this compares the codec and not the
// store width.
struct ByteDiff {
  size_t bytes = 0;
  size_t scales = 0;
  size_t nonzero_out = 0;
};

ByteDiff RunBitwise(const std::vector<float>& x_row, int64_t m, int64_t k, int group_size,
                    DType x_dtype, uint8_t (*enc)(float)) {
  REQUIRE(static_cast<int64_t>(x_row.size()) == k);
  const int64_t groups = k / group_size;
  Queue q{Cpu(), nullptr};

  std::vector<float> xf32(static_cast<size_t>(m * k));
  std::vector<uint16_t> xbf16(static_cast<size_t>(m * k));
  std::vector<float> ref_in(static_cast<size_t>(m * k));
  for (int64_t r = 0; r < m; ++r)
    for (int64_t c = 0; c < k; ++c) {
      // Each row is the same population rotated, so every row has its own amax.
      const float v = x_row[static_cast<size_t>((c + r * 37) % k)] *
                      (1.0f + 0.25f * static_cast<float>(r));
      xf32[static_cast<size_t>(r * k + c)] = v;
      xbf16[static_cast<size_t>(r * k + c)] = vt::F32ToBF16(v);
      ref_in[static_cast<size_t>(r * k + c)] = x_dtype == DType::kBF16 ? RoundToBf16(v) : v;
    }

  void* xp = x_dtype == DType::kBF16 ? static_cast<void*>(xbf16.data())
                                     : static_cast<void*>(xf32.data());
  Tensor tx = MakeTensor(xp, x_dtype, Cpu(), {m, k});
  std::vector<uint8_t> got(static_cast<size_t>(m * k), 0xFFu);
  std::vector<float> got_s(static_cast<size_t>(m * groups), -1.0f);
  Tensor tq = MakeTensor(got.data(), DType::kI8, Cpu(), {m, k});
  Tensor ts = MakeTensor(got_s.data(), DType::kF32, Cpu(), {m, groups});
  vt::QuantFp8Group(q, tq, ts, tx, group_size);

  std::vector<uint8_t> want(static_cast<size_t>(m * k));
  std::vector<float> want_s(static_cast<size_t>(m * groups));
  for (int64_t r = 0; r < m; ++r)
    for (int64_t g = 0; g < groups; ++g)
      RefQuantGroup(&ref_in[static_cast<size_t>(r * k + g * group_size)], group_size,
                    &want[static_cast<size_t>(r * k + g * group_size)],
                    &want_s[static_cast<size_t>(r * groups + g)], enc);

  ByteDiff d;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      if (d.bytes == 0) {
        CAPTURE(i);
        CAPTURE(static_cast<unsigned>(got[i]));
        CAPTURE(static_cast<unsigned>(want[i]));
      }
      ++d.bytes;
    }
    if ((want[i] & 0x7Fu) != 0u) ++d.nonzero_out;
  }
  for (size_t i = 0; i < want_s.size(); ++i) {
    // BITWISE on the scale too: an f32 compare here, not an Approx, because a
    // 1-ULP scale difference is exactly what distinguishes the two upstream
    // arms and an Approx cannot see it.
    if (got_s[i] != want_s[i]) {
      if (d.scales == 0) {
        CAPTURE(i);
        CAPTURE(got_s[i]);
        CAPTURE(want_s[i]);
      }
      ++d.scales;
    }
  }
  return d;
}

}  // namespace

// ===========================================================================
// G1 — bitwise, zero tolerance, against the exhaustive reference. This is the
// only arm that can distinguish the executing CUDA arm's two divides from the
// Triton fallback's reciprocal multiply.
TEST_CASE("G1: CPU QuantFp8Group equals an independent e4m3 reference byte for byte") {
  REQUIRE(vt::OpRegistered(vt::OpId::kQuantFp8Group, DeviceType::kCPU));
  // The fast encoder is validated against the exhaustive one here, so G2 may
  // rely on it. A disagreement is a defect in the TEST and must not be widened.
  size_t enc_mismatch = 0;
  for (int e = -12; e <= 9; ++e)
    for (int m = 0; m < 16; ++m) {
      const auto v = static_cast<float>(std::ldexp(1.0 + m / 16.0, e));
      for (float s : {1.0f, -1.0f}) {
        const float r = std::fmin(std::fmax(v * s, kFp8Min), kFp8Max);
        if (RefEncodeRne(r) != FastEncodeRne(r)) ++enc_mismatch;
      }
    }
  CHECK(enc_mismatch == 0u);

  // group_size 64/128/512 over K that each divide, both input widths, and M > 1
  // so a per-ROW amax that leaked across rows is visible.
  for (int group_size : {64, 128, 512}) {
    for (int64_t k : {512, 1024}) {
      for (float span : {1.0f, 0.0037f, 91.5f}) {
        CAPTURE(group_size);
        CAPTURE(k);
        CAPTURE(span);
        const auto row = G1Row(k, 1234u + static_cast<uint32_t>(k), span);
        for (DType dt : {DType::kF32, DType::kBF16}) {
          const ByteDiff d = RunBitwise(row, 3, k, group_size, dt, &RefEncodeRne);
          CHECK(d.bytes == 0u);
          CHECK(d.scales == 0u);
          // VACUITY GUARD: an all-zero reference would make any implementation
          // pass. Most of this population is nonzero by construction.
          CHECK(d.nonzero_out > static_cast<size_t>(3 * k) / 2);
        }
      }
    }
  }
}

// ===========================================================================
// G2 — the ported upstream case.
//
// PORT OF tests/kernels/quantization/test_block_fp8.py:82-118 at vLLM
// 5559679229bc961848b121ccdeaa8fa5d79bec98, with its grid at :42-46:
//   NUM_TOKENS = [7, 2050]     D = [512, 4096, 5120, 13824]
//   GROUP_SIZE = [64, 128, 512]                     SEEDS = [0]
// and its tolerances: values compared after dequant at rtol=0.15 (:112-114),
// the scale at torch.allclose's defaults rtol=1e-5 atol=1e-8 (:115).
//
// HARNESS ADAPTATIONS, and nothing else changed. (1) Upstream draws `x` from
// torch.rand on the device; there is no bit-compatible host RNG, so this draws
// the same shape from std::mt19937 over the same [0,1) support. (2) Upstream's
// DTYPES list ships only bfloat16, with float32 commented out at :40, and this
// runs bfloat16 only for the same reason. f32 input coverage is not lost: G1
// compares BOTH widths bitwise, which is the stronger statement.
// (3) COLUMN_MAJOR_SCALES and TMA_ALIGNED_SCALES (:45-46, :117-120) are
// dropped: those layouts are owed to #1189 M5 and this op emits only the
// row-major one. (4) The ROCm 1-ULP branch (:97-110) is not applicable.
TEST_CASE("G2: ported test_per_token_group_quant_fp8 grid matches the native reference") {
  REQUIRE(vt::OpRegistered(vt::OpId::kQuantFp8Group, DeviceType::kCPU));
  Queue q{Cpu(), nullptr};
  size_t total_nonzero = 0;
  size_t total_elems = 0;

  for (int64_t num_tokens : {7, 2050}) {
    for (int64_t d : {512, 4096, 5120, 13824}) {
      for (int group_size : {64, 128, 512}) {
        for (DType dt : {DType::kBF16}) {  // DTYPES = [torch.bfloat16] at :40
          CAPTURE(num_tokens);
          CAPTURE(d);
          CAPTURE(group_size);
          const int64_t groups = d / group_size;
          const auto n = static_cast<size_t>(num_tokens * d);

          std::mt19937 rng(0u);  // SEEDS = [0]
          std::uniform_real_distribution<float> ux(0.0f, 1.0f);  // torch.rand
          std::vector<float> xf32(n);
          std::vector<uint16_t> xbf16(n);
          std::vector<float> ref_in(n);
          for (size_t i = 0; i < n; ++i) {
            const float v = ux(rng);
            xf32[i] = v;
            xbf16[i] = vt::F32ToBF16(v);
            ref_in[i] = dt == DType::kBF16 ? RoundToBf16(v) : v;
          }
          void* xp = dt == DType::kBF16 ? static_cast<void*>(xbf16.data())
                                        : static_cast<void*>(xf32.data());
          Tensor tx = MakeTensor(xp, dt, Cpu(), {num_tokens, d});
          std::vector<uint8_t> got(n, 0xFFu);
          std::vector<float> got_s(static_cast<size_t>(num_tokens * groups), -1.0f);
          Tensor tq = MakeTensor(got.data(), DType::kI8, Cpu(), {num_tokens, d});
          Tensor ts = MakeTensor(got_s.data(), DType::kF32, Cpu(), {num_tokens, groups});
          vt::QuantFp8Group(q, tq, ts, tx, group_size);

          std::vector<uint8_t> want(n);
          std::vector<float> want_s(static_cast<size_t>(num_tokens * groups));
          for (int64_t r = 0; r < num_tokens; ++r)
            for (int64_t g = 0; g < groups; ++g)
              RefQuantGroup(&ref_in[static_cast<size_t>(r * d + g * group_size)], group_size,
                            &want[static_cast<size_t>(r * d + g * group_size)],
                            &want_s[static_cast<size_t>(r * groups + g)], &FastEncodeRne);

          // :112-114  assert allclose(out.float(), ref_out.float(), rtol=0.15)
          size_t bad = 0, nonzero = 0;
          for (size_t i = 0; i < n; ++i) {
            const double a = RefDequant(got[i]), b = RefDequant(want[i]);
            if (!(std::fabs(a - b) <= 0.15 * std::fabs(b))) ++bad;
            if (b != 0.0) ++nonzero;
          }
          CHECK(bad == 0u);
          // :115  assert allclose(scale, ref_scale)   rtol=1e-5 atol=1e-8
          size_t bad_s = 0;
          for (size_t i = 0; i < want_s.size(); ++i)
            if (!(std::fabs(static_cast<double>(got_s[i]) - want_s[i]) <=
                  1e-8 + 1e-5 * std::fabs(static_cast<double>(want_s[i]))))
              ++bad_s;
          CHECK(bad_s == 0u);
          total_nonzero += nonzero;
          total_elems += n;
        }
      }
    }
  }
  // VACUITY GUARD: torch.rand draws from [0,1), so essentially every reference
  // element is nonzero. An all-zero reference would make any implementation
  // pass every comparison above.
  CAPTURE(total_nonzero);
  CAPTURE(total_elems);
  CHECK(total_nonzero > total_elems - total_elems / 1000);
}

// ===========================================================================
// G3 — the scale itself. A token gate cannot see a scale that collapsed to
// per-tensor or that lost its eps floor; an exact value can.
TEST_CASE("G3: the group scale is amax/448 and an all-zero group uses the eps floor") {
  Queue q{Cpu(), nullptr};
  constexpr int64_t kM = 2, kK = 256;
  constexpr int kG = 64;
  const int64_t groups = kK / kG;
  std::vector<float> x(static_cast<size_t>(kM * kK), 0.0f);
  // Row 0: each group gets its OWN known amax. A per-tensor collapse makes all
  // four scales equal, which this detects.
  const float amax[4] = {1.0f, 8.0f, 0.125f, 300.0f};
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t i = 0; i < kG; ++i)
      x[static_cast<size_t>(g * kG + i)] = amax[g] * 0.5f * ((i % 3 == 0) ? -1.0f : 1.0f);
    x[static_cast<size_t>(g * kG + 7)] = -amax[g];  // the amax, negative
  }
  // Row 1 stays all zero: y_s must be eps/448 and every output byte must be 0.
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {kM, kK});
  std::vector<uint8_t> got(static_cast<size_t>(kM * kK), 0xFFu);
  std::vector<float> got_s(static_cast<size_t>(kM * groups), -1.0f);
  Tensor tq = MakeTensor(got.data(), DType::kI8, Cpu(), {kM, kK});
  Tensor ts = MakeTensor(got_s.data(), DType::kF32, Cpu(), {kM, groups});
  vt::QuantFp8Group(q, tq, ts, tx, kG);

  for (int64_t g = 0; g < groups; ++g) {
    CAPTURE(g);
    CHECK(got_s[static_cast<size_t>(g)] == amax[g] / kFp8Max);
    // The amax element saturates the grid: |x|/y_s == 448 exactly -> 0x7E.
    CHECK(got[static_cast<size_t>(g * kG + 7)] == 0xFEu);  // sign bit set
  }
  // A per-tensor collapse would make these equal.
  CHECK(got_s[0] != got_s[1]);
  for (int64_t g = 0; g < groups; ++g) {
    CAPTURE(g);
    CHECK(got_s[static_cast<size_t>(groups + g)] == kEps / kFp8Max);
    for (int64_t i = 0; i < kG; ++i)
      CHECK(got[static_cast<size_t>(kK + g * kG + i)] == 0x00u);
  }
}

// ===========================================================================
// G4 — the scale's dtype and shape are part of the contract. Upstream allocates
// float32 [M, K/group_size] (fp8_utils.py:629-631). A scale that silently
// narrowed to bf16 would still produce plausible tokens.
TEST_CASE("G4: out_scale must be f32 and shaped M by K over group_size") {
  Queue q{Cpu(), nullptr};
  constexpr int64_t kM = 4, kK = 256;
  constexpr int kG = 128;
  std::vector<float> x(static_cast<size_t>(kM * kK), 0.5f);
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {kM, kK});
  std::vector<uint8_t> outq(static_cast<size_t>(kM * kK));
  Tensor tq = MakeTensor(outq.data(), DType::kI8, Cpu(), {kM, kK});
  std::vector<float> s(static_cast<size_t>(kM * (kK / kG)));

  Tensor ok = MakeTensor(s.data(), DType::kF32, Cpu(), {kM, kK / kG});
  CHECK_NOTHROW(vt::QuantFp8Group(q, tq, ok, tx, kG));

  Tensor narrow = MakeTensor(s.data(), DType::kBF16, Cpu(), {kM, kK / kG});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq, narrow, tx, kG), std::runtime_error);

  Tensor wrong_groups = MakeTensor(s.data(), DType::kF32, Cpu(), {kM, kK / kG + 1});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq, wrong_groups, tx, kG), std::runtime_error);

  Tensor wrong_rows = MakeTensor(s.data(), DType::kF32, Cpu(), {kM - 1, kK / kG});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq, wrong_rows, tx, kG), std::runtime_error);

  Tensor wrong_out = MakeTensor(outq.data(), DType::kF32, Cpu(), {kM, kK});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, wrong_out, ok, tx, kG), std::runtime_error);
}

// ===========================================================================
// G5 — the refusals. Upstream asserts divisibility at fp8_utils.py:596-599 and
// contiguity at :600. A ragged K accepted silently reads past the row.
TEST_CASE("G5: a K that the group size does not divide is refused by name") {
  Queue q{Cpu(), nullptr};
  constexpr int64_t kM = 2, kK = 200;  // 200 % 128 != 0 and 200 % 64 != 0
  std::vector<float> x(static_cast<size_t>(kM * kK), 0.5f);
  std::vector<uint8_t> outq(static_cast<size_t>(kM * kK));
  std::vector<float> s(static_cast<size_t>(kM * 2));
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {kM, kK});
  Tensor tq = MakeTensor(outq.data(), DType::kI8, Cpu(), {kM, kK});
  Tensor ts = MakeTensor(s.data(), DType::kF32, Cpu(), {kM, 2});
  CHECK_THROWS_WITH_AS(vt::QuantFp8Group(q, tq, ts, tx, 128),
                       doctest::Contains("must be divisible"), std::runtime_error);
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq, ts, tx, 0), std::runtime_error);
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq, ts, tx, -128), std::runtime_error);

  // A non-contiguous x: fp8_utils.py:600 `x.stride(-1) == 1`.
  constexpr int64_t kK2 = 256;
  std::vector<float> x2(static_cast<size_t>(kM * kK2), 0.5f);
  Tensor gappy = MakeTensor(x2.data(), DType::kF32, Cpu(), {kM, kK2});
  gappy.stride[1] = 2;
  std::vector<uint8_t> outq2(static_cast<size_t>(kM * kK2));
  std::vector<float> s2(static_cast<size_t>(kM * 2));
  Tensor tq2 = MakeTensor(outq2.data(), DType::kI8, Cpu(), {kM, kK2});
  Tensor ts2 = MakeTensor(s2.data(), DType::kF32, Cpu(), {kM, 2});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq2, ts2, gappy, 128), std::runtime_error);

  // A device mismatch between the queue and the tensors.
  Tensor onGpu = MakeTensor(x2.data(), DType::kF32, Device{DeviceType::kCUDA, 0}, {kM, kK2});
  CHECK_THROWS_AS(vt::QuantFp8Group(q, tq2, ts2, onGpu, 128), std::runtime_error);
}

// ===========================================================================
// G6 — CPU vs CUDA, bitwise, on the identical input. The arm that says the CPU
// path mirrors what ships rather than merely being self-consistent.
// NAME THIS CASE WITHOUT A COMMA: doctest splits `-tc=` on commas, so a comma
// makes the name unselectable and the binary reports `0 cases ran ... SUCCESS!`
// with exit 0 (measured on tests/vt/test_ops_fp8_cpu.cpp, #468 review F6).
TEST_CASE("G6: CPU QuantFp8Group equals CUDA QuantFp8Group byte for byte") {
  if (!HasCuda()) {
    // NOT a silent skip. The CPU registration is still asserted so the case can
    // never be vacuous, and the banner names what is owed. This is the state
    // .agents/specs/vt-quant-fp8-group.md records under `## Owed`: the row took
    // no GPU lease by design, so the CUDA arm compiles and does not run.
    MESSAGE("G6 PENDING: no CUDA device on this host, CPU-vs-CUDA byte agreement "
            "was NOT measured (owed by #1189 milestone M5)");
    CHECK(vt::OpRegistered(vt::OpId::kQuantFp8Group, DeviceType::kCPU));
    return;
  }
  vt::Backend& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  Queue cq{Cpu(), nullptr};
  const Device gpu{DeviceType::kCUDA, 0};

  for (int group_size : {64, 128, 512}) {
    for (int64_t k : {512, 1024}) {
      CAPTURE(group_size);
      CAPTURE(k);
      constexpr int64_t kM = 3;
      const int64_t groups = k / group_size;
      const auto row = G1Row(k, 4321u, 1.0f);
      std::vector<float> x(static_cast<size_t>(kM * k));
      for (int64_t r = 0; r < kM; ++r)
        for (int64_t c = 0; c < k; ++c)
          x[static_cast<size_t>(r * k + c)] =
              row[static_cast<size_t>((c + r * 37) % k)] * (1.0f + 0.25f * static_cast<float>(r));

      std::vector<uint8_t> cpu_out(static_cast<size_t>(kM * k));
      std::vector<float> cpu_s(static_cast<size_t>(kM * groups));
      Tensor tx_cpu = MakeTensor(x.data(), DType::kF32, Cpu(), {kM, k});
      Tensor tq_cpu = MakeTensor(cpu_out.data(), DType::kI8, Cpu(), {kM, k});
      Tensor ts_cpu = MakeTensor(cpu_s.data(), DType::kF32, Cpu(), {kM, groups});
      vt::QuantFp8Group(cq, tq_cpu, ts_cpu, tx_cpu, group_size);

      void* dx = b.Alloc(static_cast<size_t>(kM * k) * sizeof(float));
      void* dq = b.Alloc(static_cast<size_t>(kM * k));
      void* ds = b.Alloc(static_cast<size_t>(kM * groups) * sizeof(float));
      b.Copy(gq, dx, x.data(), static_cast<size_t>(kM * k) * sizeof(float));
      Tensor tx_gpu = MakeTensor(dx, DType::kF32, gpu, {kM, k});
      Tensor tq_gpu = MakeTensor(dq, DType::kI8, gpu, {kM, k});
      Tensor ts_gpu = MakeTensor(ds, DType::kF32, gpu, {kM, groups});
      vt::QuantFp8Group(gq, tq_gpu, ts_gpu, tx_gpu, group_size);
      std::vector<uint8_t> gpu_out(static_cast<size_t>(kM * k));
      std::vector<float> gpu_s(static_cast<size_t>(kM * groups));
      b.Copy(gq, gpu_out.data(), dq, gpu_out.size());
      b.Copy(gq, gpu_s.data(), ds, gpu_s.size() * sizeof(float));
      b.Synchronize(gq);

      size_t bad = 0, bad_s = 0, nonzero = 0;
      for (size_t i = 0; i < cpu_out.size(); ++i) {
        if (cpu_out[i] != gpu_out[i]) ++bad;
        if ((cpu_out[i] & 0x7Fu) != 0u) ++nonzero;
      }
      for (size_t i = 0; i < cpu_s.size(); ++i)
        if (cpu_s[i] != gpu_s[i]) ++bad_s;
      CHECK(bad == 0u);
      CHECK(bad_s == 0u);
      CHECK(nonzero > cpu_out.size() / 2);  // vacuity guard
      b.Free(dx);
      b.Free(dq);
      b.Free(ds);
    }
  }
  b.DestroyQueue(gq);
}
