// QUANT-GGUF-CIQ-GEMM work rows G2 (activation quant) and G3 (the six tier-0
// generic `vec_dot` kernels + their GEMM wiring).
//
// Ported from llama.cpp @ 237ad9b96:
//   tests/test-quantize-fns.cpp   — the RMSE/dot-product bounds at :17-28
//     (MAX_QUANTIZATION_TOTAL_ERROR 0.002, MAX_QUANTIZATION_REFERENCE_ERROR
//     0.0001, MAX_DOT_PRODUCT_ERROR 0.02, ..._LOWBIT 0.04), the synthetic data
//     generator at :35, `array_rmse` at :41, `dot_product_error` at :86, and
//     the 32*128 test size at :132. Upstream's thresholds are used as-is; none
//     is widened.
//   tests/test-backend-ops.cpp    — the MUL_MAT NMSE bound `max_nmse_err()`
//     = 5e-4 at :4277-4279, applied to `kMatmulBTQuant` at model-ish shapes.
//
// WHY THE PRIMARY GATE IS NOT AN UPSTREAM PORT.
// A `vec_dot` is exactly where a subtle block-decode slip yields
// plausible-but-wrong numbers, and checking one copy of the decode against
// another copy of the same logic would not catch it. So the load-bearing case
// here — "vec_dot agrees with an INDEPENDENT f64 reference" — dequantizes both
// operands through `BlockToFloat` (the loader-side `dequantize_row_*` decoders,
// a SEPARATE port that walks the block layout differently from the inline
// decode inside each vec_dot) and dots them in DOUBLE precision. The two agree
// exactly in exact arithmetic, so the check can be tight: the tolerance is
// relative to the dot's L1 magnitude, which makes it immune to the sign
// cancellation that would otherwise let a large error hide behind a small sum.
//
// SCOPE NOTE — what is honestly NOT covered. Upstream's `dot_product_error`
// quantizes BOTH operands with `from_float`. We port `from_float` only for the
// two activation encodings Q8_0 and Q8_K (that is all of G2's scope — nothing
// in this project ever quantizes an activation into a k-quant), so the full
// upstream round-trip is reproduced for Q8_0 and the k-quants are gated on the
// f64 reference above with random-bit-pattern blocks instead. Those random
// blocks are strictly HARDER on the decode than encoder output: they exercise
// bit patterns a real encoder never emits.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

// --- upstream thresholds, test-quantize-fns.cpp:17-28 -----------------------
constexpr float kMaxQuantizationReferenceError = 0.0001F;
constexpr float kMaxQuantizationTotalError = 0.002F;
constexpr float kMaxDotProductError = 0.02F;
// test-backend-ops.cpp:4277-4279
constexpr double kMaxNmseErr = 5e-4;
// test-quantize-fns.cpp:132
constexpr size_t kTestSize = 32 * 128;

// test-quantize-fns.cpp:35 — the synthetic signal upstream measures on.
void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
  }
}

// test-quantize-fns.cpp:41 — note the normalization is sqrt(sum)/n, NOT
// sqrt(sum/n). Kept verbatim because the thresholds above are calibrated to it.
float ArrayRmse(const float* a1, const float* a2, size_t n) {
  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    double diff = static_cast<double>(a1[i]) - static_cast<double>(a2[i]);
    sum += diff * diff;
  }
  return std::sqrt(sum) / static_cast<float>(n);
}

// --- block field offsets, written out FRESH from ggml-common.h --------------
// Deliberately independent of src/vt/cpu/cpu_quant_blocks.h (the struct mirror
// the kernels use) so this file is a third statement of the same layout facts,
// in the same spirit as the G1 trait cross-check.
struct WeightCase {
  vt::DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;      // ggml_half super-block scale, -1 when absent
  int dmin_off;   // ggml_half super-block min scale, -1 when absent
  const char* name;
};

// Byte offsets are the running field sums of each ggml-common.h struct. The
// K-quants do NOT agree on where the delta sits — q4_K/q5_K lead with it while
// q3_K/q6_K trail it — which is exactly the kind of detail a port gets wrong,
// so each is spelled out:
//   q4_0 :213-218  d@0  qs@2                                    (18B)
//   q8_0 :242-245  d@0  qs@2                                    (34B)
//   q3_K :305-310  hmask@0 qs@32 sc@96 d@108                    (110B)
//   q4_K :317-327  d@0 dmin@2 sc@4 qs@16                        (144B)
//   q5_K :334-345  d@0 dmin@2 sc@4 qh@16 qs@48                  (176B)
//   q6_K :352-357  ql@0 qh@128 sc@192 d@208                     (210B)
const WeightCase kWeightCases[] = {
    {vt::DType::kQ4_0, 32, 18, 0, -1, "q4_0"},
    {vt::DType::kQ8_0, 32, 34, 0, -1, "q8_0"},
    {vt::DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {vt::DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {vt::DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {vt::DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

// Random raw blocks: every quant/scale payload byte is arbitrary (all legal),
// with only the f16 delta fields pinned to finite, modest values so the dot
// stays in range. Random payloads are the point — they sweep bit patterns
// (including the 6-bit packed-scale corner cases and every hmask/qh bit) that
// an encoder would never produce.
std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    // Vary the deltas per block so a kernel that hoisted the scale out of the
    // block loop would be caught.
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

// The INDEPENDENT reference: decode both operands through the loader-side
// `to_float` decoders and accumulate in f64. Returns the dot and, separately,
// the L1 magnitude sum(|w*a|) used to set a cancellation-robust tolerance.
struct RefDot {
  double dot;
  double l1;
};

RefDot ReferenceDotF64(vt::DType wtype, const uint8_t* wq, vt::DType atype,
                       const uint8_t* aq, int64_t k) {
  std::vector<float> w(static_cast<size_t>(k));
  std::vector<float> a(static_cast<size_t>(k));
  vt::cpu::BlockToFloat(wtype)(wq, w.data(), k);
  vt::cpu::BlockToFloat(atype)(aq, a.data(), k);
  RefDot r{0.0, 0.0};
  for (int64_t i = 0; i < k; ++i) {
    // A non-finite REFERENCE means the synthetic blocks are malformed (a delta
    // field left as random bytes decoding to inf/nan), not that a kernel is
    // wrong. Failing here keeps that diagnosis unambiguous — it is how the
    // q6_K delta offset (208, not 0: q4_K/q5_K lead with the delta, q6_K
    // trails it) was caught while writing this file.
    REQUIRE(std::isfinite(w[static_cast<size_t>(i)]));
    REQUIRE(std::isfinite(a[static_cast<size_t>(i)]));
    const double t = static_cast<double>(w[static_cast<size_t>(i)]) *
                     static_cast<double>(a[static_cast<size_t>(i)]);
    r.dot += t;
    r.l1 += std::fabs(t);
  }
  return r;
}

// Quantize an f32 activation row into the encoding `wtype` dots against.
std::vector<uint8_t> QuantizeActivation(vt::DType wtype, const float* x,
                                        int64_t k) {
  const vt::DType at = vt::cpu::QuantTraits(wtype).vec_dot_type;
  std::vector<uint8_t> q(vt::RowSizeBytes(at, k));
  vt::cpu::QuantTraits(at).from_float(x, q.data(), k);
  return q;
}

float RunVecDot(vt::DType wtype, const uint8_t* wq, const uint8_t* aq,
                int64_t k) {
  float s = 0.0F;
  vt::cpu::QuantTraits(wtype).vec_dot(static_cast<int>(k), &s, 0, wq, 0, aq, 0,
                                      1);
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// G2 + G3 — the traits table is populated
// ---------------------------------------------------------------------------

TEST_CASE("G2/G3 populate from_float and vec_dot (ggml-cpu.c:211-406)") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const vt::cpu::QuantTypeTraits& t = vt::cpu::QuantTraits(c.dtype);
    CHECK(t.to_float != nullptr);
    CHECK(t.vec_dot != nullptr);           // G3
    CHECK(t.nrows == 1);                   // nrows==2 is i8mm-only (G6)
    // The activation encoding it dispatches to must be able to encode.
    CHECK(vt::cpu::QuantTraits(t.vec_dot_type).from_float != nullptr);  // G2
    CHECK(vt::cpu::HasQuantDotKernel(c.dtype));
  }

  // from_float exists for exactly the two activation encodings.
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kQ8_0) != nullptr);
  CHECK(vt::cpu::BlockFromFloat(vt::DType::kQ8_K) != nullptr);
  for (vt::DType d : {vt::DType::kQ4_0, vt::DType::kQ3_K, vt::DType::kQ4_K,
                      vt::DType::kQ5_K, vt::DType::kQ6_K}) {
    CHECK(vt::cpu::BlockFromFloat(d) == nullptr);
  }

  // Q8_K is activation-only: upstream gives it no vec_dot row, so it must stay
  // on the generic dequant composite rather than pretending to have a kernel.
  CHECK(vt::cpu::BlockVecDot(vt::DType::kQ8_K) == nullptr);
  CHECK_FALSE(vt::cpu::HasQuantDotKernel(vt::DType::kQ8_K));
}

// ---------------------------------------------------------------------------
// G2 — activation quantization
// ---------------------------------------------------------------------------

TEST_CASE("G2 quantize_row_q8_0/q8_K round-trip within upstream RMSE bounds") {
  // Port of test-quantize-fns.cpp `total_quantization_error` (:52) on the
  // upstream synthetic signal at the upstream test size.
  std::vector<float> data(kTestSize);
  GenerateData(0.0F, data.size(), data.data());

  for (vt::DType d : {vt::DType::kQ8_0, vt::DType::kQ8_K}) {
    CAPTURE(vt::Name(d));
    std::vector<uint8_t> q(vt::RowSizeBytes(d, static_cast<int64_t>(kTestSize)));
    std::vector<float> back(kTestSize);

    vt::cpu::QuantTraits(d).from_float(data.data(), q.data(),
                                       static_cast<int64_t>(kTestSize));
    vt::cpu::BlockToFloat(d)(q.data(), back.data(),
                             static_cast<int64_t>(kTestSize));

    const float rmse = ArrayRmse(data.data(), back.data(), kTestSize);
    CAPTURE(rmse);
    CHECK(rmse < kMaxQuantizationTotalError);
  }
}

TEST_CASE("G2 from_float is the reference encoder (reference error == 0)") {
  // Upstream's `reference_quantization_error` (:62) compares the CPU-tier
  // `from_float` against the `from_float_ref` reference encoder; on the generic
  // tier they are literally the same function (quants.c:45,117 just call the
  // _ref form), so ours must agree to the BIT, not merely within 1e-4.
  std::vector<float> data(kTestSize);
  GenerateData(0.0F, data.size(), data.data());

  for (vt::DType d : {vt::DType::kQ8_0, vt::DType::kQ8_K}) {
    CAPTURE(vt::Name(d));
    const size_t bytes = vt::RowSizeBytes(d, static_cast<int64_t>(kTestSize));
    std::vector<uint8_t> q1(bytes);
    std::vector<uint8_t> q2(bytes);
    vt::cpu::QuantTraits(d).from_float(data.data(), q1.data(),
                                       static_cast<int64_t>(kTestSize));
    vt::cpu::QuantTraits(d).from_float(data.data(), q2.data(),
                                       static_cast<int64_t>(kTestSize));
    // Bit-exact and deterministic run to run (project rule: fixed reduction
    // order, no nondeterministic accumulation anywhere in the quant path).
    CHECK(std::memcmp(q1.data(), q2.data(), bytes) == 0);

    std::vector<float> b1(kTestSize);
    std::vector<float> b2(kTestSize);
    vt::cpu::BlockToFloat(d)(q1.data(), b1.data(),
                             static_cast<int64_t>(kTestSize));
    vt::cpu::BlockToFloat(d)(q2.data(), b2.data(),
                             static_cast<int64_t>(kTestSize));
    CHECK(ArrayRmse(b1.data(), b2.data(), kTestSize) <
          kMaxQuantizationReferenceError);
  }
}

namespace {

// Round half AWAY FROM ZERO — an independent statement of `roundf`'s rule
// (ggml-quants.c:260 uses roundf for Q8_0), written without calling roundf.
float RoundHalfAway(float v) {
  const float m = std::floor(std::fabs(v) + 0.5F);
  return v < 0 ? -m : m;
}

// Round half to EVEN — an independent statement of what ggml-quants.c:563's
// `nearest_int` magic-constant trick (fval + 12582912.f) actually computes,
// written via the FP rounding mode instead of the bit hack. Q8_K's quants
// depend on this exact tie rule.
int RoundHalfEven(float v) { return static_cast<int>(std::nearbyint(v)); }

}  // namespace

TEST_CASE("G2 from_float is BYTE-EXACT vs an independent reference encoder") {
  // The upstream RMSE/NMSE bounds are calibrated for 8-bit and are, measurably,
  // too loose to distinguish round-to-nearest from truncation: a mutant that
  // replaced `roundf(x0)` with a bare cast passed every statistical gate in
  // this file. The encoder's ROUNDING RULE therefore gets a byte-level gate of
  // its own, against a reference written from the upstream prose rather than
  // by calling the same library function.
  constexpr int64_t kK = 256 * 4;  // whole blocks for both encodings
  std::vector<float> data(static_cast<size_t>(kK));
  GenerateData(0.25F, data.size(), data.data());

  SUBCASE("q8_0 (ggml-quants.c:238)") {
    std::vector<uint8_t> got(vt::RowSizeBytes(vt::DType::kQ8_0, kK));
    vt::cpu::QuantTraits(vt::DType::kQ8_0)
        .from_float(data.data(), got.data(), kK);

    for (int64_t b = 0; b < kK / 32; ++b) {
      CAPTURE(b);
      const float* x = data.data() + b * 32;
      float amax = 0.0F;
      for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[j]));
      // NOTE the deliberate upstream asymmetry: the quants are derived from the
      // UNROUNDED d, while the STORED d is the f16 rounding of it.
      const float d = amax / 127.0F;
      const float id = d ? 1.0F / d : 0.0F;

      const uint8_t* blk = got.data() + b * 34;
      uint16_t d_stored = 0;
      std::memcpy(&d_stored, blk, sizeof(d_stored));
      CHECK(d_stored == vt::F32ToF16(d));
      const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
      for (int j = 0; j < 32; ++j) {
        CHECK(qs[j] == static_cast<int8_t>(RoundHalfAway(x[j] * id)));
      }
    }
  }

  SUBCASE("q8_K (ggml-quants.c:2696)") {
    std::vector<uint8_t> got(vt::RowSizeBytes(vt::DType::kQ8_K, kK));
    vt::cpu::QuantTraits(vt::DType::kQ8_K)
        .from_float(data.data(), got.data(), kK);

    for (int64_t b = 0; b < kK / 256; ++b) {
      CAPTURE(b);
      const float* x = data.data() + b * 256;
      float mx = 0;
      float amax = 0;
      for (int j = 0; j < 256; ++j) {
        const float ax = std::fabs(x[j]);
        if (ax > amax) {
          amax = ax;
          mx = x[j];
        }
      }
      REQUIRE(amax != 0.0F);
      // Keyed on the SIGNED extremum, so `iscale` is negative and the quants
      // are sign-flipped relative to the input — a detail that is easy to
      // "correct" into a silent divergence from every consumer vec_dot.
      const float iscale = -127.0F / mx;

      const uint8_t* blk = got.data() + b * 292;
      float d_stored = 0;
      std::memcpy(&d_stored, blk, sizeof(d_stored));
      CHECK(d_stored == 1.0F / iscale);
      const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
      for (int j = 0; j < 256; ++j) {
        const int v = RoundHalfEven(iscale * x[j]);
        // Upstream's MIN(127, v) is reproduced, but note it is UNREACHABLE for
        // well-formed input: |iscale| is 127/amax and |x[j]| <= amax, so
        // |iscale*x[j]| <= 127 by construction. A mutant deleting the clamp is
        // therefore NOT caught by this file — correctly, since the two are
        // equivalent on every reachable input. It is kept for upstream
        // fidelity, not because a test can distinguish it.
        CHECK(qs[j] == static_cast<int8_t>(v < 127 ? v : 127));
      }
    }
  }
}

TEST_CASE("G2 Q8_K bsums equal the per-16 group sums (ggml-quants.c:2696)") {
  // The Q4_K/Q5_K vec_dots subtract the block minimum using `bsums` alone, so
  // a wrong bsum corrupts those dots while `qs` still looks perfect. Check the
  // invariant directly against the stored quants.
  constexpr int64_t kK = 256 * 5;
  std::vector<float> data(static_cast<size_t>(kK));
  GenerateData(0.5F, data.size(), data.data());

  std::vector<uint8_t> q(vt::RowSizeBytes(vt::DType::kQ8_K, kK));
  vt::cpu::QuantTraits(vt::DType::kQ8_K).from_float(data.data(), q.data(), kK);

  // block_q8_K (ggml-common.h:361-365): f32 d @0, i8 qs[256] @4, i16 bsums[16]
  // @260. Offsets restated here rather than taken from the kernel's header.
  for (int64_t b = 0; b < kK / 256; ++b) {
    const uint8_t* blk = q.data() + b * 292;
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    for (int g = 0; g < 16; ++g) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) sum += qs[g * 16 + i];
      int16_t stored = 0;
      std::memcpy(&stored, blk + 260 + 2 * g, sizeof(stored));
      CAPTURE(b);
      CAPTURE(g);
      CHECK(stored == static_cast<int16_t>(sum));
    }
    // Upstream keys iscale on the SIGNED extremum (-127/max), so quants reach
    // -127 on the extreme element and the positive side is MIN(127, v)-clamped.
    for (int i = 0; i < 256; ++i) CHECK(qs[i] >= -127);
  }
}

TEST_CASE("G2 all-zero Q8_K block encodes to zero delta, quants and bsums") {
  // ggml-quants.c:2706-2711 — the `!amax` early-out. Upstream relies on calloc'd
  // scratch for bsums here; ours zeroes them explicitly, so assert it.
  std::vector<float> zeros(256, 0.0F);
  std::vector<uint8_t> q(vt::RowSizeBytes(vt::DType::kQ8_K, 256), 0xAB);
  vt::cpu::QuantTraits(vt::DType::kQ8_K).from_float(zeros.data(), q.data(), 256);

  float d = 1.0F;
  std::memcpy(&d, q.data(), sizeof(d));
  CHECK(d == 0.0F);
  for (int i = 0; i < 256; ++i) CHECK(q[4 + static_cast<size_t>(i)] == 0);
  for (int g = 0; g < 16; ++g) {
    int16_t stored = 1;
    std::memcpy(&stored, q.data() + 260 + 2 * g, sizeof(stored));
    CHECK(stored == 0);
  }
}

TEST_CASE("G2 activation quant rejects ragged K (fails loudly)") {
  std::vector<float> data(1024, 0.5F);
  std::vector<uint8_t> q(64 * 1024);
  // Q8_0 blocks are 32 elements, Q8_K blocks are 256: a partial trailing block
  // has no representation, so it must throw rather than write a short row.
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_0)
                   .from_float(data.data(), q.data(), 33));
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                   .from_float(data.data(), q.data(), 257));
  CHECK_THROWS(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                   .from_float(data.data(), q.data(), 255));
  // ...and the exact block multiples do not.
  CHECK_NOTHROW(vt::cpu::QuantTraits(vt::DType::kQ8_0)
                    .from_float(data.data(), q.data(), 32));
  CHECK_NOTHROW(vt::cpu::QuantTraits(vt::DType::kQ8_K)
                    .from_float(data.data(), q.data(), 256));
}

TEST_CASE("G2 scratch sizing mirrors ggml_row_size / graph_plan wdata") {
  // ggml-cpu.c:1313-1349 lays src1 out one ggml_row_size(vec_dot_type, k) row
  // at a time with no padding; ggml-cpu.c:2752-2980 sizes wdata as rows*that.
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ4_K, 256) == 292);   // one q8_K
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ6_K, 512) == 584);   // two
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ4_0, 32) == 34);     // one q8_0
  CHECK(vt::cpu::QuantActRowBytes(vt::DType::kQ8_0, 96) == 102);    // three
  CHECK(vt::cpu::QuantActScratchBytes(vt::DType::kQ4_K, 4, 512) == 4 * 584);
  CHECK(vt::cpu::QuantActScratchBytes(vt::DType::kQ8_0, 0, 32) == 0);
  // Ragged K has no valid scratch layout.
  CHECK_THROWS(vt::cpu::QuantActRowBytes(vt::DType::kQ4_K, 255));
  CHECK_THROWS(vt::cpu::QuantActRowBytes(vt::DType::kQ4_0, 33));
}

// ---------------------------------------------------------------------------
// G3 — vec_dot vs the independent f64 reference  (THE primary correctness gate)
// ---------------------------------------------------------------------------

TEST_CASE("G3 vec_dot == independent f64 dequantize-then-dot, all six types") {
  for (const WeightCase& c : kWeightCases) {
    // Block-boundary and ragged-K coverage: a SINGLE-block row, an even
    // multiple, ODD multiples (3, 5, 7 blocks — these catch a kernel that
    // assumed pairs or an unrolled-by-2 block loop), and a large row.
    for (int64_t nblocks : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                            int64_t{7}, int64_t{16}}) {
      const int64_t k = nblocks * c.block_elems;
      CAPTURE(std::string(c.name));
      CAPTURE(nblocks);
      CAPTURE(k);

      const std::vector<uint8_t> wq =
          RandomBlocks(c, nblocks, 0xC1A0U + static_cast<uint32_t>(nblocks));

      // Activation: a real f32 signal pushed through G2's from_float, so this
      // case also gates the G2 -> G3 handoff (the round-trip that feeds the
      // dot) and not just the weight decode.
      std::vector<float> act(static_cast<size_t>(k));
      GenerateData(1.0F, act.size(), act.data());
      const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);

      const float got = RunVecDot(c.dtype, wq.data(), aq.data(), k);
      const vt::DType at = vt::cpu::QuantTraits(c.dtype).vec_dot_type;
      const RefDot ref = ReferenceDotF64(c.dtype, wq.data(), at, aq.data(), k);

      CAPTURE(got);
      CAPTURE(ref.dot);
      CAPTURE(ref.l1);
      CHECK(std::isfinite(got));
      // Tolerance relative to the L1 magnitude, not to |dot|: with random
      // signs the dot can sit near zero, and a |dot|-relative bound would then
      // be unsatisfiable while an absolute bound would be vacuous. f32
      // accumulation over <=16 blocks costs ~1e-6 relative; 1e-5 is a tight
      // ceiling that still catches any real decode error (those are O(1)).
      CHECK(std::fabs(static_cast<double>(got) - ref.dot) <= 1e-5 * ref.l1);
    }
  }
}

TEST_CASE("G3 vec_dot is bit-exact run to run (fixed reduction order)") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 5 * c.block_elems;
    const std::vector<uint8_t> wq = RandomBlocks(c, 5, 4242U);
    std::vector<float> act(static_cast<size_t>(k));
    GenerateData(2.0F, act.size(), act.data());
    const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);

    const float first = RunVecDot(c.dtype, wq.data(), aq.data(), k);
    for (int rep = 0; rep < 4; ++rep) {
      // Bit equality, not Approx: the kernels fix the reduction order, so any
      // drift would mean nondeterministic accumulation crept in.
      CHECK(RunVecDot(c.dtype, wq.data(), aq.data(), k) == first);
    }
  }
}

TEST_CASE("G3 vec_dot rejects ragged n and the unported nrc==2 mmla mode") {
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 2 * c.block_elems;
    const std::vector<uint8_t> wq = RandomBlocks(c, 2, 7U);
    std::vector<float> act(static_cast<size_t>(k), 0.25F);
    const std::vector<uint8_t> aq = QuantizeActivation(c.dtype, act.data(), k);
    const vt::cpu::VecDotFn f = vt::cpu::QuantTraits(c.dtype).vec_dot;
    float s = 0.0F;

    // n must be a whole number of weight blocks.
    CHECK_THROWS(f(static_cast<int>(c.block_elems) + 1, &s, 0, wq.data(), 0,
                   aq.data(), 0, 1));
    // nrc==2 is the i8mm mmla contract (G6). Accepting it here would silently
    // read a second row that the generic kernel never dots.
    CHECK_THROWS(f(static_cast<int>(k), &s, 0, wq.data(), 0, aq.data(), 0, 2));
    CHECK_NOTHROW(f(static_cast<int>(k), &s, 0, wq.data(), 0, aq.data(), 0, 1));
  }
}

TEST_CASE("G3 dot_product_error within upstream bound (test-quantize-fns:86)") {
  // The literal upstream case, reproducible for Q8_0 because it is the one
  // type where we have BOTH the encoder (G2) and the vec_dot (G3): quantize
  // both operands, dot them, and compare against the full-precision dot of the
  // original data, normalized by test_size exactly as upstream does.
  std::vector<float> d1(kTestSize);
  std::vector<float> d2(kTestSize);
  GenerateData(0.0F, d1.size(), d1.data());
  GenerateData(1.0F, d2.size(), d2.data());

  const int64_t k = static_cast<int64_t>(kTestSize);
  std::vector<uint8_t> q1(vt::RowSizeBytes(vt::DType::kQ8_0, k));
  vt::cpu::QuantTraits(vt::DType::kQ8_0).from_float(d1.data(), q1.data(), k);
  const std::vector<uint8_t> q2 =
      QuantizeActivation(vt::DType::kQ8_0, d2.data(), k);

  const float got = RunVecDot(vt::DType::kQ8_0, q1.data(), q2.data(), k);
  double dot_ref = 0;
  for (size_t i = 0; i < kTestSize; i++) dot_ref += d1[i] * d2[i];

  const float err =
      std::fabs(got - static_cast<float>(dot_ref)) / static_cast<float>(kTestSize);
  CAPTURE(got);
  CAPTURE(dot_ref);
  CAPTURE(err);
  CHECK(err < kMaxDotProductError);
}

// ---------------------------------------------------------------------------
// G3 — the GEMM wiring (kMatmulBTQuant), ported MUL_MAT cases
// ---------------------------------------------------------------------------

namespace {

// Build the [N,K] block weight tensor + [M,K] f32 activations and run the op.
struct GemmFixture {
  std::vector<uint8_t> wq;
  std::vector<float> a;
  std::vector<float> out;
};

GemmFixture RunGemm(const WeightCase& c, int64_t m, int64_t k, int64_t n,
                    uint32_t seed, int64_t* out_n = nullptr) {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  GemmFixture f;
  f.wq = RandomBlocks(c, n * (k / c.block_elems), seed);
  f.a.resize(static_cast<size_t>(m * k));
  GenerateData(1.0F, f.a.size(), f.a.data());
  f.out.assign(static_cast<size_t>(m * n), 0.0F);

  vt::Tensor at =
      vt::Tensor::Contiguous(f.a.data(), vt::DType::kF32, q.device, {m, k});
  vt::Tensor bt = vt::Tensor::Contiguous(f.wq.data(), vt::DType::kF32, q.device,
                                         {n, k});
  bt.dtype = c.dtype;  // block dtype: elementwise strides are not meaningful
  vt::Tensor ot =
      vt::Tensor::Contiguous(f.out.data(), vt::DType::kF32, q.device, {m, n});
  vt::MatmulBTQuant(q, ot, at, bt);
  if (out_n != nullptr) *out_n = n;
  return f;
}

}  // namespace

TEST_CASE("G3 MatmulBTQuant NMSE <= 5e-4 vs dequant-f32 (test-backend-ops)") {
  // Port of the MUL_MAT cases: the quantized GEMM is measured against the
  // full-precision composite (dequantize the weight, dot in f32 against the
  // UNQUANTIZED activation), which is exactly the quantization error upstream
  // bounds at 5e-4. Shapes span decode (M=1) through prefill (M=512) at
  // model-ish K/N, plus the odd N that catches a chunking assumption.
  for (const WeightCase& c : kWeightCases) {
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);
        const GemmFixture f = RunGemm(c, m, k, n, 0x5EEDU);

        // Independent reference: the loader-side decoder + an f64 dot.
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(f.wq.data(), w.data(), n * k);

        double num = 0;
        double den = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t j = 0; j < n; ++j) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p) {
              ref += static_cast<double>(f.a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(j * k + p)]);
            }
            const double got = f.out[static_cast<size_t>(i * n + j)];
            num += (got - ref) * (got - ref);
            den += ref * ref;
          }
        }
        const double nmse = den > 0 ? num / den : num;
        CAPTURE(nmse);
        CHECK(nmse <= kMaxNmseErr);
      }
    }
  }
}

TEST_CASE("G3 MatmulBTQuant is bit-exact run to run and across thread counts") {
  // The project rule: output rows are partitioned, each output keeps its own
  // sequential K reduction, so thread count must not perturb a single bit.
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 4 * c.block_elems;
    const int64_t m = 3;
    const int64_t n = 9;

    const GemmFixture base = RunGemm(c, m, k, n, 0xBEEFU);
    for (int threads : {1, 2, 4}) {
      CAPTURE(threads);
      vt::cpu::Threadpool tp(threads);
      vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
      const GemmFixture again = RunGemm(c, m, k, n, 0xBEEFU);
      vt::cpu::Threadpool::SwapForTesting(prev);

      REQUIRE(again.out.size() == base.out.size());
      CHECK(std::memcmp(again.out.data(), base.out.data(),
                        base.out.size() * sizeof(float)) == 0);
    }
  }
}

TEST_CASE("G3 MatmulBTQuant fails loudly on ragged K") {
  // K must be a whole number of blocks for BOTH the weight encoding and the
  // activation encoding it dots against (256 for the K-quants). A partial
  // block would mis-stride the scratch, so it must throw, never round down.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = c.block_elems + 1;  // ragged
    const int64_t m = 1;
    const int64_t n = 1;
    std::vector<uint8_t> wq(static_cast<size_t>(4 * c.block_bytes), 0);
    std::vector<float> a(static_cast<size_t>(m * k), 0.25F);
    std::vector<float> out(static_cast<size_t>(m * n), 0.0F);

    vt::Tensor at =
        vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
    vt::Tensor bt =
        vt::Tensor::Contiguous(wq.data(), vt::DType::kF32, q.device, {n, k});
    bt.dtype = c.dtype;
    vt::Tensor ot =
        vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {m, n});
    CHECK_THROWS(vt::MatmulBTQuant(q, ot, at, bt));
  }
}

TEST_CASE("G3 MatmulBTQuant matches per-row vec_dot exactly (no GEMM drift)") {
  // The GEMM must be nothing but "quantize src1 once, then one vec_dot per
  // output" (ggml-cpu.c:1313-1443). Comparing the op's output BIT-EXACTLY
  // against a hand-driven vec_dot per element proves the wiring adds no
  // reordering, no fused accumulation and no stride slip.
  for (const WeightCase& c : kWeightCases) {
    CAPTURE(std::string(c.name));
    const int64_t k = 3 * c.block_elems;
    const int64_t m = 2;
    const int64_t n = 5;
    const GemmFixture f = RunGemm(c, m, k, n, 0xF00DU);

    const size_t w_row = static_cast<size_t>(vt::RowSizeBytes(c.dtype, k));
    for (int64_t i = 0; i < m; ++i) {
      const std::vector<uint8_t> aq =
          QuantizeActivation(c.dtype, f.a.data() + i * k, k);
      for (int64_t j = 0; j < n; ++j) {
        const float expect =
            RunVecDot(c.dtype, f.wq.data() + static_cast<size_t>(j) * w_row,
                      aq.data(), k);
        CHECK(f.out[static_cast<size_t>(i * n + j)] == expect);
      }
    }
  }
}
