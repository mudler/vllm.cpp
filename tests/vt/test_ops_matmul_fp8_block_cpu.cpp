// vllm.cpp — the 128x128 block-scaled FP8 GEMM, CPU reference arm.
//
// VT-MATMUL-FP8-BLOCK-REF (.agents/specs/vt-matmul-fp8-block-ref.md), issue
// #1189 milestone M2. Pinned oracle: vLLM 5559679229bc961848b121ccdeaa8fa5d79bec98,
// asserted as the local checkout's HEAD before every anchor below was read.
//
// THE CONSTRAINT THIS FILE EXISTS TO HOLD. The scales apply in the GEMM
// MAINLOOP, once per K-block, into an f32 accumulator — not in the epilogue:
//
//     accumulator = zeros(f32)
//     for k_block:
//         accumulator += dot(a_tile, b_tile) * a_s[m,k_block] * b_s[n_block,k_block]
//
// (vllm/model_executor/layers/quantization/utils/fp8_utils.py:826-836, the
// Triton arm spelling out the structure). Our per-tensor FP8 path folds ONE
// scalar alpha into the epilogue (`MatmulFp8CutlassKernel`, `alpha * acc` after
// the whole K reduction). An epilogue has exactly one degree of freedom per
// output element and the block scheme has cdiv(K, block_k) of them, so an
// epilogue-only application CANNOT EXPRESS a per-K-block scale at all. That is a
// correctness constraint, not an optimisation choice. G4 is the instrument for
// it and no single-alpha implementation can pass G4.
//
// WHICH IMPLEMENTATION ACTUALLY RUNS, asked because M1 found upstream's Triton
// source is not what executes and that the two arms DISAGREE in polarity. Here
// the chain lands differently — the executing kernel is CUTLASS and it AGREES
// with the reference:
//   vllm/model_executor/kernels/linear/__init__.py:355-377   CUDA priority list
//   vllm/utils/deep_gemm.py:27-46                            DeepGEMM off for
//                                                            qwen3_5_text on family 120
//   .../scaled_mm/cutlass.py:312-326      ops.cutlass_scaled_mm(A, B.T, As, Bs.T)
//   .../cutlass/scaled_mm_entry.cu:220-226            every sm>=120 -> sm120
//   .../cutlass/scaled_mm_c3x_sm120.cu:13-20          -> blockwise_sm120_fp8
//   .../c3x/scaled_mm_helper.hpp:15-18,39-55          blockwise branch: both
//                                                     scales f32 and 2-D,
//                                                     ceil_div shapes, no bias
//   .../c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56-58,218-235
//                                                     ElementAccumulator=float,
//                                                     both scale pointers are
//                                                     MAINLOOP arguments
//   cutlass 4.5.0 include/cutlass/gemm/collective/
//     sm120_mma_tma_blockwise_scaling.hpp:714-717
//       accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i)
// which is native_w8a8_block_matmul's `c += matmul(a, b.t()) * s`
// (tests/kernels/quant_utils.py:145-151) with the two scale multiplies
// associated left to right instead of as a product. The two differ by at most
// one f32 ULP per K-block, and upstream's own gate is what admits it: it
// compares the CUTLASS arm against this very reference at rel_diff < 0.001
// (tests/kernels/quantization/test_block_fp8.py:194-200). We mirror the
// reference's association, because this op IS the reference port.
//
//   G1  the registration itself, and OpName totality.
//   G2  the ported upstream case (test_block_fp8.py:123-153) over an adapted
//       grid, against an INDEPENDENTLY WRITTEN double reference, carrying
//       upstream's rel_diff < 0.001 verbatim AND a tighter f32 forward-error
//       bound. Vacuity guard.
//   G3  the ragged edges, called out separately because they are why the grid
//       is what it is: N=576 (4*128+64), K=3884 (30*128+44), and both at once.
//   G4  THE MAINLOOP CONSTRAINT, by exact hand-computed values plus a scale
//       SWAP that no epilogue-folded alpha can distinguish.
//   G5  the refusals, each by name.
//   G6  the M1 seam: vt::QuantFp8Group -> vt::MatmulFp8BlockScaled end to end
//       on a CPU queue. A COMPOSITION test, not a reachability claim: nothing
//       in production dispatches either op yet and the spec's `## Owed` says so.
//
// THE REFERENCE IS INDEPENDENT, or the gate is a tautology. It accumulates in
// `double` with a DIFFERENT loop nest — upstream's, k-tile outer and n-block
// inner over whole tiles — while the kernel walks output elements with the
// k-tiles inside. Its fp8 decode is derived from the e4m3fn field layout
// (1 sign, 4 exponent bias 7, 3 mantissa, no infinities), not taken from any
// codec in src/. Reduction-order ULPs are exactly what the f32 forward-error
// bound admits; nothing here claims bit-exactness against a double.
//
// THE UPSTREAM GRID IS ADAPTED, and the adaptation is stated rather than
// hidden. Upstream runs itertools.product(M, N, K) with M=[1,7,8,83,4096],
// N=[128,512,576,7168,13824], K=[256,3884,4096,13824,16384]
// (test_block_fp8.py:48-55) — 125 combinations whose largest is 4096x13824x16384,
// about 9.3e11 multiply-accumulates. That is a GPU grid; a naive CPU reference
// nest, run twice, cannot execute it. Every axis VALUE below appears at least
// once and the ragged values appear separately and together; only the PAIRING
// is dropped. Parameters, dtypes, tolerance and failure criterion are preserved
// exactly.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// --- the independent e4m3fn decode ----------------------------------------
// From the format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, no
// infinities, 0x7F/0xFF the only NaN encodings ("fn"). Not taken from src/.
constexpr float kFp8Max = 448.0f;   // quant_utils.py:27-35 finfo(e4m3fn).max
constexpr float kFp8Min = -448.0f;

double E4m3Exact(unsigned exp_field, unsigned mant) {
  if (exp_field == 0) return std::ldexp(static_cast<double>(mant), -9);  // mant / 512
  return std::ldexp(1.0 + static_cast<double>(mant) / 8.0, static_cast<int>(exp_field) - 7);
}

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

// The 127 finite magnitudes 0x00..0x7E are MONOTONIC in the byte value, so the
// nearest-value encode is a binary search over their midpoints, with an exact
// midpoint resolving to the even mantissa (table index == byte value, so
// `lo & 1` IS the mantissa's low bit).
const std::vector<double>& Magnitudes() {
  static const std::vector<double> table = [] {
    std::vector<double> v;
    v.reserve(127);
    for (unsigned e = 0; e <= 15; ++e)
      for (unsigned m = 0; m <= 7; ++m) {
        if (e == 15 && m == 7) continue;  // the NaN encoding is not a value
        v.push_back(E4m3Exact(e, m));
      }
    return v;
  }();
  return table;
}

uint8_t EncodeRne(float r) {
  const auto sign = static_cast<uint8_t>(std::signbit(r) ? 0x80u : 0x00u);
  const double a = std::fabs(static_cast<double>(r));
  const std::vector<double>& mag = Magnitudes();
  const auto it = std::lower_bound(mag.begin(), mag.end(), a);
  size_t hi = static_cast<size_t>(it - mag.begin());
  if (hi == 0) return sign;                                        // a == 0
  if (hi >= mag.size()) return static_cast<uint8_t>(sign | 0x7Eu);  // a >= 448
  const size_t lo = hi - 1;
  const double dlo = a - mag[lo], dhi = mag[hi] - a;
  size_t pick = 0;
  if (dlo < dhi) {
    pick = lo;
  } else if (dhi < dlo) {
    pick = hi;
  } else {
    pick = (lo & 1u) == 0u ? lo : hi;
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(pick));
}

// --- operands ---------------------------------------------------------------
// Upstream builds each operand as
//   A_fp32 = (rand(M,K) - 0.5) * 2 * fp8_max
//   A_fp8  = A_fp32.clamp(min=fp8_min, max=fp8_max).to(fp8)
// (test_block_fp8.py:130-136), i.e. the fp8 bytes induced by a value that is
// UNIFORM on [-448, 448]. The transform is reproduced; the RNG is not torch's,
// which is a harness fact rather than a parameter — the values are random on
// both sides and nothing here depends on a particular draw.
//
// The encode is hoisted into a 2^20-entry table of uniformly spaced values
// because the grid below needs ~4e7 operand bytes and a binary search per byte,
// at the -O0 the CPU lane builds with, dominated the case's run time. The
// resulting byte distribution is upstream's to a value resolution of 8.5e-4,
// and a uniform draw lands below that magnitude with probability ~2e-6.
const std::vector<uint8_t>& UniformByteTable() {
  static const std::vector<uint8_t> table = [] {
    constexpr int kN = 1 << 20;
    std::vector<uint8_t> v(static_cast<size_t>(kN));
    for (int i = 0; i < kN; ++i) {
      const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(kN);  // (0,1)
      const float x = (u - 0.5f) * 2.0f * kFp8Max;
      v[static_cast<size_t>(i)] = EncodeRne(std::fmin(std::fmax(x, kFp8Min), kFp8Max));
    }
    return v;
  }();
  return table;
}

std::vector<uint8_t> RandomFp8(int64_t n, uint32_t seed) {
  const std::vector<uint8_t>& tab = UniformByteTable();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> idx(0, static_cast<uint32_t>(tab.size() - 1));
  std::vector<uint8_t> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = tab[idx(rng)];
  return out;
}

// As = rand(M, k_tiles) * 1e-2 ; Bs = rand(n_tiles, k_tiles) * 1e-2
// (test_block_fp8.py:143-144, factor_for_scale = 1e-2 at :127).
std::vector<float> RandomScales(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = u(rng) * 1e-2f;
  return out;
}

// --- the independent reference ---------------------------------------------
// native_w8a8_block_matmul (tests/kernels/quant_utils.py:91-154), transcribed:
// k-tile outer, n-block inner, whole tiles, f32 semantics widened to double so
// that the kernel's own f32 rounding is what the bound measures. `abs_sum`
// carries the same nest and is the forward-error bound's magnitude term.
struct RefResult {
  std::vector<double> c;        // [M,N]
  std::vector<double> abs_sum;  // [M,N] sum over k-tiles of |partial| * |s|
};

RefResult RefBlockMatmul(const std::vector<double>& ad, const std::vector<double>& bd,
                         const std::vector<float>& as, const std::vector<float>& bs, int64_t m,
                         int64_t n, int64_t k, int64_t block_n, int64_t block_k) {
  const int64_t n_tiles = CDiv(n, block_n);  // quant_utils.py:123-124, CEIL
  const int64_t k_tiles = CDiv(k, block_k);
  RefResult r;
  r.c.assign(static_cast<size_t>(m * n), 0.0);
  r.abs_sum.assign(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < k_tiles; ++i) {            // quant_utils.py:145
    const int64_t k0 = i * block_k;
    const int64_t k1 = std::min((i + 1) * block_k, k);
    for (int64_t j = 0; j < n_tiles; ++j) {          // quant_utils.py:146
      const int64_t n0 = j * block_n;
      const int64_t n1 = std::min((j + 1) * block_n, n);
      const double b_s = static_cast<double>(bs[static_cast<size_t>(j * k_tiles + i)]);
      for (int64_t row = 0; row < m; ++row) {
        const double a_s = static_cast<double>(as[static_cast<size_t>(row * k_tiles + i)]);
        const double s = a_s * b_s;                  // quant_utils.py:150, PRODUCT first
        for (int64_t col = n0; col < n1; ++col) {
          double part = 0.0, part_abs = 0.0;
          for (int64_t kk = k0; kk < k1; ++kk) {
            const double p = ad[static_cast<size_t>(row * k + kk)] *
                             bd[static_cast<size_t>(col * k + kk)];
            part += p;
            part_abs += std::fabs(p);
          }
          r.c[static_cast<size_t>(row * n + col)] += part * s;   // quant_utils.py:151
          r.abs_sum[static_cast<size_t>(row * n + col)] += part_abs * std::fabs(s);
        }
      }
    }
  }
  return r;
}

std::vector<double> Decode(const std::vector<uint8_t>& bytes) {
  const std::vector<double>& tab = DequantTable();
  std::vector<double> out(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) out[i] = tab[bytes[i]];
  return out;
}

double ReadOut(const std::vector<uint8_t>& buf, DType dt, int64_t i) {
  if (dt == DType::kF32) {
    float v = 0.0f;
    std::memcpy(&v, buf.data() + static_cast<size_t>(i) * 4, sizeof(v));
    return static_cast<double>(v);
  }
  uint16_t v = 0;
  std::memcpy(&v, buf.data() + static_cast<size_t>(i) * 2, sizeof(v));
  return static_cast<double>(vt::BF16ToF32(v));
}

// One shape of the ported case. `out_dtype` is the store width; upstream runs
// bfloat16 (test_block_fp8.py:54) and the op admits f32 as well.
void RunPorted(int64_t m, int64_t n, int64_t k, DType out_dtype, uint32_t seed) {
  constexpr int64_t kBlockN = 128, kBlockK = 128;  // test_block_fp8.py:53
  const int64_t n_tiles = CDiv(n, kBlockN), k_tiles = CDiv(k, kBlockK);
  CAPTURE(m);
  CAPTURE(n);
  CAPTURE(k);

  const std::vector<uint8_t> a = RandomFp8(m * k, seed);
  const std::vector<uint8_t> b = RandomFp8(n * k, seed + 1u);
  const std::vector<float> as = RandomScales(m * k_tiles, seed + 2u);
  const std::vector<float> bs = RandomScales(n_tiles * k_tiles, seed + 3u);

  const int64_t out_elems = m * n;
  std::vector<uint8_t> out(static_cast<size_t>(out_elems) *
                           (out_dtype == DType::kF32 ? 4u : 2u));

  Queue q;
  q.device = Cpu();
  Tensor ta = MakeTensor(const_cast<uint8_t*>(a.data()), DType::kI8, Cpu(), {m, k});
  Tensor tas = MakeTensor(const_cast<float*>(as.data()), DType::kF32, Cpu(), {m, k_tiles});
  Tensor tb = MakeTensor(const_cast<uint8_t*>(b.data()), DType::kI8, Cpu(), {n, k});
  Tensor tbs =
      MakeTensor(const_cast<float*>(bs.data()), DType::kF32, Cpu(), {n_tiles, k_tiles});
  Tensor tout = MakeTensor(out.data(), out_dtype, Cpu(), {m, n});
  vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, static_cast<int>(kBlockN),
                           static_cast<int>(kBlockK));

  const RefResult ref = RefBlockMatmul(Decode(a), Decode(b), as, bs, m, n, k, kBlockN, kBlockK);

  // (1) upstream's own criterion, verbatim (test_block_fp8.py:150-153):
  //     mean|out - ref| / mean|ref| < 0.001, with BOTH sides at the store width
  //     so that a bf16 store is common-mode exactly as it is upstream.
  // (2) a tighter per-element f32 forward-error bound, because our arm IS the
  //     reference rather than a kernel measured against it: the f32 recursive-sum
  //     bound over K terms, x4 margin, plus the store's own half-ulp for bf16.
  double sum_abs_diff = 0.0, sum_abs_ref = 0.0, worst_ratio = 0.0;
  int64_t bad = 0, first_bad = -1, nonzero = 0;
  const double eps_f32 = static_cast<double>(std::numeric_limits<float>::epsilon());
  for (int64_t i = 0; i < out_elems; ++i) {
    const double got = ReadOut(out, out_dtype, i);
    const double want = ref.c[static_cast<size_t>(i)];
    const double want_stored =
        out_dtype == DType::kF32
            ? static_cast<double>(static_cast<float>(want))
            : static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(want))));
    sum_abs_diff += std::fabs(got - want_stored);
    sum_abs_ref += std::fabs(want_stored);
    const double tol =
        4.0 * static_cast<double>(k) * eps_f32 * ref.abs_sum[static_cast<size_t>(i)] +
        (out_dtype == DType::kBF16 ? std::fabs(want) * 0.004 : 0.0);
    const double diff = std::fabs(got - want);
    if (tol > 0.0 && diff / tol > worst_ratio) worst_ratio = diff / tol;
    if (!(diff <= tol)) {
      if (bad == 0) {
        first_bad = i;
        CAPTURE(got);
        CAPTURE(want);
        CAPTURE(tol);
      }
      ++bad;
    }
    if (std::fabs(want) > 0.0) ++nonzero;
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
  }
  CAPTURE(worst_ratio);
  CHECK(bad == 0);
  const double rel_diff = sum_abs_diff / sum_abs_ref;
  CAPTURE(rel_diff);
  CHECK(rel_diff < 0.001);  // test_block_fp8.py:153
  // VACUITY GUARD: an all-zero reference would make any implementation pass.
  CHECK(nonzero == out_elems);
}

}  // namespace

// ===========================================================================
// G1 — THE REGISTRATION ITSELF.
//
// A refusal test cannot stand in for this and that is measured, not argued: M1
// deleted its CPU registration and every refusal case still passed, because the
// refusals live in the wrapper's validation and fire before dispatch. Only an
// OpRegistered assertion caught it (.agents/specs/vt-quant-fp8-group.md,
// "A refusal test cannot stand in for a registration test").
TEST_CASE("G1: MatmulFp8BlockScaled is registered on the CPU backend and named") {
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCPU));
  CHECK(std::string(vt::OpName(vt::OpId::kMatmulFp8BlockScaled)) == "MatmulFp8BlockScaled");
}

// ===========================================================================
// G2/G3 — the ported upstream case over the adapted grid.
//
// Every axis value of upstream's M/N/K lists appears at least once. The three
// ragged entries are G3 and are marked: N=576 is 4*128 + 64, K=3884 is
// 30*128 + 44, and one case carries both at once, which is where a floor
// division and a ceil division give different answers in different places.
TEST_CASE("G2: MatmulFp8BlockScaled matches native_w8a8_block_matmul over the ported grid") {
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCPU));
  //         M     N      K      out            seed
  RunPorted(4096, 128,   256,   DType::kBF16, 1101);  // M=4096
  RunPorted(83,   128,   4096,  DType::kBF16, 1201);  // M=83, K=4096
  RunPorted(8,    7168,  256,   DType::kBF16, 1301);  // N=7168
  RunPorted(7,    13824, 256,   DType::kBF16, 1401);  // M=7, N=13824
  RunPorted(1,    512,   13824, DType::kBF16, 1501);  // M=1 decode, K=13824
  RunPorted(1,    128,   16384, DType::kBF16, 1601);  // K=16384
  // the f32 store arm, which upstream lists but leaves commented out at :54
  RunPorted(83,   128,   4096,  DType::kF32,  1701);
}

TEST_CASE("G3: MatmulFp8BlockScaled handles a ragged final N-block and K-block") {
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCPU));
  RunPorted(8,  576, 4096, DType::kBF16, 2101);  // N=576 = 4*128 + 64, round K
  RunPorted(32, 576, 3884, DType::kBF16, 2201);  // BOTH ragged: K=3884 = 30*128 + 44
  // Upstream's own dedicated ragged case, "weight.shape % 128 != 0, like in DSV3
  // kv_a_proj_with_mqa" (test_block_fp8.py:156-200).
  RunPorted(32, 576, 7168, DType::kBF16, 2301);
  RunPorted(32, 576, 3884, DType::kF32,  2401);
}

// ===========================================================================
// G4 — THE MAINLOOP CONSTRAINT, and it is constructed so that no epilogue-folded
// alpha can pass it.
//
// Every value here is exact in bf16 and in f32, so these are equalities and not
// tolerances. Written as bare `==` rather than doctest's Approx: Approx carries
// a `scale` term and a strict `<`, so `epsilon(0.0)` compares `|a-b| < 0` and
// fails on values that ARE equal, while the default epsilon admits ~1.19e-5. Case A has two K-blocks whose partial products differ (128 and
// 256) and whose scales differ (0.25 and 0.5). SWAPPING the two K-block scales
// leaves every per-tensor summary of the scale tensor identical — same set,
// same sum, same product, same max — and changes the correct answer from 160 to
// 128. A kernel that reduces the scales to one epilogue alpha returns the same
// number for both and fails here.
TEST_CASE("G4: the scales apply per K-BLOCK in the mainloop, not once in the epilogue") {
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCPU));
  constexpr uint8_t kOne = 0x38;  // e4m3fn 1.0 = exponent field 7, mantissa 0
  constexpr uint8_t kTwo = 0x40;  // e4m3fn 2.0 = exponent field 8, mantissa 0
  REQUIRE(DequantTable()[kOne] == 1.0);
  REQUIRE(DequantTable()[kTwo] == 2.0);

  Queue q;
  q.device = Cpu();

  SUBCASE("A: swapping the two K-block scales changes the answer") {
    constexpr int64_t kM = 1, kN = 1, kK = 256;  // 2 K-blocks of 128, 1 N-block
    std::vector<uint8_t> a(static_cast<size_t>(kM * kK), kOne);
    std::vector<uint8_t> b(static_cast<size_t>(kN * kK), kOne);
    for (int64_t i = 128; i < kK; ++i) b[static_cast<size_t>(i)] = kTwo;  // block 1: b = 2
    // partial(block 0) = 128 * 1 * 1 = 128 ; partial(block 1) = 128 * 1 * 2 = 256
    std::vector<float> as = {1.0f, 1.0f};
    std::vector<float> bs = {0.25f, 0.5f};

    Tensor ta = MakeTensor(a.data(), DType::kI8, Cpu(), {kM, kK});
    Tensor tb = MakeTensor(b.data(), DType::kI8, Cpu(), {kN, kK});
    Tensor tas = MakeTensor(as.data(), DType::kF32, Cpu(), {kM, 2});
    Tensor tbs = MakeTensor(bs.data(), DType::kF32, Cpu(), {1, 2});

    float got = 0.0f;
    Tensor tout = MakeTensor(&got, DType::kF32, Cpu(), {kM, kN});
    vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, 128, 128);
    // 128*0.25 + 256*0.5 = 32 + 128
    CHECK(static_cast<double>(got) == 160.0);

    std::swap(bs[0], bs[1]);
    float got_swapped = 0.0f;
    Tensor tout2 = MakeTensor(&got_swapped, DType::kF32, Cpu(), {kM, kN});
    vt::MatmulFp8BlockScaled(q, tout2, ta, tas, tb, tbs, 128, 128);
    // 128*0.5 + 256*0.25 = 64 + 64
    CHECK(static_cast<double>(got_swapped) == 128.0);
    // The load-bearing statement: an epilogue alpha cannot tell these apart.
    CHECK(got != got_swapped);
    // VACUITY GUARD.
    CHECK(got != 0.0f);
    CHECK(got_swapped != 0.0f);
  }

  SUBCASE("B: the b-scale row is indexed by OUTPUT COLUMN / block_n, ragged block included") {
    // N = 129 -> 2 N-blocks, the second of them one column wide. a = b = 1.0
    // everywhere, so every K-block partial is exactly 128 and the output is
    // decided entirely by which scale pair the kernel picked.
    constexpr int64_t kM = 2, kN = 129, kK = 256;
    std::vector<uint8_t> a(static_cast<size_t>(kM * kK), kOne);
    std::vector<uint8_t> b(static_cast<size_t>(kN * kK), kOne);
    std::vector<float> as = {1.0f, 1.0f, 2.0f, 2.0f};              // [2,2]
    std::vector<float> bs = {0.25f, 0.5f, 1.0f, 0.125f};           // [2,2]

    Tensor ta = MakeTensor(a.data(), DType::kI8, Cpu(), {kM, kK});
    Tensor tb = MakeTensor(b.data(), DType::kI8, Cpu(), {kN, kK});
    Tensor tas = MakeTensor(as.data(), DType::kF32, Cpu(), {kM, 2});
    Tensor tbs = MakeTensor(bs.data(), DType::kF32, Cpu(), {2, 2});

    std::vector<float> out(static_cast<size_t>(kM * kN), 0.0f);
    Tensor tout = MakeTensor(out.data(), DType::kF32, Cpu(), {kM, kN});
    vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, 128, 128);

    int64_t nonzero = 0;
    for (int64_t row = 0; row < kM; ++row) {
      const double a_row = row == 0 ? 1.0 : 2.0;
      for (int64_t col = 0; col < kN; ++col) {
        const size_t nb = col < 128 ? 0u : 1u;  // OUTPUT COLUMN / block_n
        const double want = 128.0 * (a_row * static_cast<double>(bs[nb * 2 + 0])) +
                            128.0 * (a_row * static_cast<double>(bs[nb * 2 + 1]));
        const double got = static_cast<double>(out[static_cast<size_t>(row * kN + col)]);
        if (got != want) {
          CAPTURE(row);
          CAPTURE(col);
          CAPTURE(got);
          CAPTURE(want);
          REQUIRE(got == want);
        }
        if (got != 0.0) ++nonzero;
      }
    }
    // VACUITY GUARD.
    CHECK(nonzero == kM * kN);
    // Named explicitly so the ragged column is not merely covered by the loop:
    // column 128 is the one-wide second N-block and must read bs row 1.
    CHECK(static_cast<double>(out[static_cast<size_t>(128)]) == 144.0);
    CHECK(static_cast<double>(out[static_cast<size_t>(kN + 128)]) == 288.0);
  }
}

// ===========================================================================
// G5 — the refusals, each by name.
TEST_CASE("G5: MatmulFp8BlockScaled refuses a malformed call by name") {
  constexpr int64_t kM = 2, kN = 256, kK = 256;
  constexpr int kBn = 128, kBk = 128;
  const int64_t nt = 2, kt = 2;
  std::vector<uint8_t> a(static_cast<size_t>(kM * kK), 0x38);
  std::vector<uint8_t> b(static_cast<size_t>(kN * kK), 0x38);
  std::vector<float> as(static_cast<size_t>(kM * kt), 1.0f);
  std::vector<float> bs(static_cast<size_t>(nt * kt), 1.0f);
  std::vector<float> out(static_cast<size_t>(kM * kN), 0.0f);

  Queue q;
  q.device = Cpu();
  Tensor ta = MakeTensor(a.data(), DType::kI8, Cpu(), {kM, kK});
  Tensor tb = MakeTensor(b.data(), DType::kI8, Cpu(), {kN, kK});
  Tensor tas = MakeTensor(as.data(), DType::kF32, Cpu(), {kM, kt});
  Tensor tbs = MakeTensor(bs.data(), DType::kF32, Cpu(), {nt, kt});
  Tensor tout = MakeTensor(out.data(), DType::kF32, Cpu(), {kM, kN});
  // The well-formed call this case perturbs.
  REQUIRE_NOTHROW(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, kBn, kBk));

  SUBCASE("a zero or negative block size, validated BEFORE it divides anything") {
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, 0, kBk),
                    std::runtime_error);
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, kBn, 0),
                    std::runtime_error);
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, -128, kBk),
                    std::runtime_error);
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs, kBn, -128),
                    std::runtime_error);
  }
  SUBCASE("a b_scale sized by FLOOR instead of cdiv, on N and on K") {
    // N = 200 -> cdiv = 2, floor = 1. A floor-tiled kernel would accept this.
    constexpr int64_t kNr = 200;
    std::vector<uint8_t> br(static_cast<size_t>(kNr * kK), 0x38);
    std::vector<float> bsf(static_cast<size_t>(1 * kt), 1.0f);
    std::vector<float> outr(static_cast<size_t>(kM * kNr), 0.0f);
    Tensor tbr = MakeTensor(br.data(), DType::kI8, Cpu(), {kNr, kK});
    Tensor tbsf = MakeTensor(bsf.data(), DType::kF32, Cpu(), {1, kt});
    Tensor toutr = MakeTensor(outr.data(), DType::kF32, Cpu(), {kM, kNr});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, toutr, ta, tas, tbr, tbsf, kBn, kBk),
                    std::runtime_error);
    // K = 200 -> cdiv = 2, floor = 1, on the K axis of b_scale.
    constexpr int64_t kKr = 200;
    std::vector<uint8_t> ar2(static_cast<size_t>(kM * kKr), 0x38);
    std::vector<uint8_t> br2(static_cast<size_t>(kN * kKr), 0x38);
    std::vector<float> as2(static_cast<size_t>(kM * 2), 1.0f);
    std::vector<float> bs2(static_cast<size_t>(nt * 1), 1.0f);
    Tensor ta2 = MakeTensor(ar2.data(), DType::kI8, Cpu(), {kM, kKr});
    Tensor tb2 = MakeTensor(br2.data(), DType::kI8, Cpu(), {kN, kKr});
    Tensor tas2 = MakeTensor(as2.data(), DType::kF32, Cpu(), {kM, 2});
    Tensor tbs2 = MakeTensor(bs2.data(), DType::kF32, Cpu(), {nt, 1});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta2, tas2, tb2, tbs2, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("an a_scale with the wrong group count or the wrong row count") {
    std::vector<float> narrow(static_cast<size_t>(kM * 1), 1.0f);
    Tensor tnarrow = MakeTensor(narrow.data(), DType::kF32, Cpu(), {kM, 1});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tnarrow, tb, tbs, kBn, kBk),
                    std::runtime_error);
    std::vector<float> tall(static_cast<size_t>((kM + 1) * kt), 1.0f);
    Tensor ttall = MakeTensor(tall.data(), DType::kF32, Cpu(), {kM + 1, kt});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, ttall, tb, tbs, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("a scale that is not f32") {
    std::vector<uint16_t> as16(static_cast<size_t>(kM * kt), 0);
    Tensor tas16 = MakeTensor(as16.data(), DType::kBF16, Cpu(), {kM, kt});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas16, tb, tbs, kBn, kBk),
                    std::runtime_error);
    std::vector<uint16_t> bs16(static_cast<size_t>(nt * kt), 0);
    Tensor tbs16 = MakeTensor(bs16.data(), DType::kBF16, Cpu(), {nt, kt});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tb, tbs16, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("a packed operand that is not i8, and a mismatched inner dimension") {
    std::vector<uint16_t> a16(static_cast<size_t>(kM * kK), 0);
    Tensor ta16 = MakeTensor(a16.data(), DType::kBF16, Cpu(), {kM, kK});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta16, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
    std::vector<uint8_t> bk(static_cast<size_t>(kN * 128), 0x38);
    Tensor tbk = MakeTensor(bk.data(), DType::kI8, Cpu(), {kN, 128});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, ta, tas, tbk, tbs, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("an out that is the wrong shape, the wrong rank, or an unsupported dtype") {
    std::vector<float> small(static_cast<size_t>(kM * 8), 0.0f);
    Tensor tsmall = MakeTensor(small.data(), DType::kF32, Cpu(), {kM, 8});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tsmall, ta, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
    Tensor t3d = MakeTensor(out.data(), DType::kF32, Cpu(), {1, kM, kN});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, t3d, ta, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
    std::vector<uint8_t> i8out(static_cast<size_t>(kM * kN), 0);
    Tensor ti8 = MakeTensor(i8out.data(), DType::kI8, Cpu(), {kM, kN});
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, ti8, ta, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("a non-contiguous operand") {
    Tensor gappy = ta;
    gappy.stride[0] = kK + 8;  // a row gap: the K run is no longer contiguous
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, gappy, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
  }
  SUBCASE("a device mismatch") {
    Tensor foreign = ta;
    foreign.device = Device{DeviceType::kCUDA, 0};
    CHECK_THROWS_AS(vt::MatmulFp8BlockScaled(q, tout, foreign, tas, tb, tbs, kBn, kBk),
                    std::runtime_error);
  }
}

// ===========================================================================
// G6 — the M1 seam, end to end on a CPU queue.
//
// vt::QuantFp8Group produces exactly the a_fp8/a_scale pair this GEMM consumes,
// and this is the composition a block-FP8 linear method will run. It is NOT a
// reachability claim: nothing in production dispatches either op at this merge
// commit, milestone M4 owns the wiring, and the spec's `## Owed` records it.
TEST_CASE("G6: QuantFp8Group feeds MatmulFp8BlockScaled on a CPU queue") {
  REQUIRE(vt::OpRegistered(vt::OpId::kQuantFp8Group, DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCPU));
  constexpr int64_t kM = 5, kN = 320, kK = 256, kG = 128;
  const int64_t nt = CDiv(kN, 128), kt = kK / kG;

  std::mt19937 rng(9001);
  std::uniform_real_distribution<float> ux(-3.0f, 3.0f);
  std::vector<float> x(static_cast<size_t>(kM * kK));
  for (auto& v : x) v = ux(rng);

  std::vector<uint8_t> aq(static_cast<size_t>(kM * kK));
  std::vector<float> aqs(static_cast<size_t>(kM * kt));
  Queue q;
  q.device = Cpu();
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {kM, kK});
  Tensor taq = MakeTensor(aq.data(), DType::kI8, Cpu(), {kM, kK});
  Tensor taqs = MakeTensor(aqs.data(), DType::kF32, Cpu(), {kM, kt});
  vt::QuantFp8Group(q, taq, taqs, tx, static_cast<int>(kG));

  const std::vector<uint8_t> b = RandomFp8(kN * kK, 7001);
  const std::vector<float> bs = RandomScales(nt * kt, 7002);
  std::vector<uint16_t> out(static_cast<size_t>(kM * kN), 0);
  Tensor tb = MakeTensor(const_cast<uint8_t*>(b.data()), DType::kI8, Cpu(), {kN, kK});
  Tensor tbs = MakeTensor(const_cast<float*>(bs.data()), DType::kF32, Cpu(), {nt, kt});
  Tensor tout = MakeTensor(out.data(), DType::kBF16, Cpu(), {kM, kN});
  vt::MatmulFp8BlockScaled(q, tout, taq, taqs, tb, tbs, 128, static_cast<int>(kG));

  const RefResult ref = RefBlockMatmul(Decode(aq), Decode(b), aqs, bs, kM, kN, kK, 128, kG);
  const double eps_f32 = static_cast<double>(std::numeric_limits<float>::epsilon());
  int64_t bad = 0, nonzero = 0;
  for (int64_t i = 0; i < kM * kN; ++i) {
    const double got = static_cast<double>(vt::BF16ToF32(out[static_cast<size_t>(i)]));
    const double want = ref.c[static_cast<size_t>(i)];
    const double tol =
        4.0 * static_cast<double>(kK) * eps_f32 * ref.abs_sum[static_cast<size_t>(i)] +
        std::fabs(want) * 0.004;
    if (!(std::fabs(got - want) <= tol)) ++bad;
    if (std::fabs(want) > 0.0) ++nonzero;
  }
  CHECK(bad == 0);
  // VACUITY GUARD.
  CHECK(nonzero == kM * kN);
}
