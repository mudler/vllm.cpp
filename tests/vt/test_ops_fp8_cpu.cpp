// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror
// of the TEST, but the reference math below is transcribed from upstream.
//
// VT-FP8-W8A8-CPU-ARM (.agents/specs/vt-fp8-w8a8-cpu-arm.md), issue #468.
//
// The CPU arm of the static per-tensor FP8 W8A8 path — vt::QuantFp8Static and
// vt::MatmulFp8Cutlass registered on DeviceType::kCPU — is what makes the fp8
// seam reachable, and therefore gateable, without a GPU. This file is that gate.
//
//   G1  BITWISE, ZERO TOLERANCE. vt::QuantFp8Static on CPU must equal an
//       INDEPENDENTLY WRITTEN reference quantizer, byte for byte.
//   G2  CPU vs CUDA, bitwise, on the identical input. CUDA-gated.
//   G3  vt::MatmulFp8Cutlass on CPU against a `double` reference that reproduces
//       upstream's LOSSY pipeline (clamp, e4m3 RNE, dequant) before accumulating.
//
// WHY THE G1 REFERENCE IS WRITTEN THE WAY IT IS. It is derived from the FORMAT
// and from upstream's formula, never from a codec in this tree — otherwise it
// would be a tautology dressed as a gate. `RefEncodeRne` enumerates all 128
// finite e4m3fn magnitudes, decodes each to an exact double from the field
// layout, and picks the nearest with an even-significand tie-break by scanning.
// That is a different ALGORITHM from `F32ToFp8` (frexp + std::nearbyint) and
// from `vllm::F32ToF8E4M3`, so agreement between them is evidence.
//
// Upstream chain (pinned oracle @ 5559679229bc961848b121ccdeaa8fa5d79bec98):
//   csrc/quantization/w8a8/fp8/common.cuh:58-77  scaled_fp8_conversion
//     :62  x = val * scale                 (is_scale_inverted == true)
//     :68  fmaxf(-448, fminf(x, 448))
//     :71  hardware RNE convert
//   csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31  1.0f / scale[...]
//     — the reciprocal is formed ONCE, outside the elementwise math
//   csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:204-210  scale.numel()==1
//     ⇒ ONE group over the whole tensor (per-tensor, not per-token)
//   vllm/model_executor/layers/quantization/modelopt.py:510-513 / :528
//     — the method is STATIC and hard-coded; input_scale collapses to a scalar
//
// So the scale is applied as `x * (1/s)`, NOT `x / s`. The two differ by up to
// one f32 ulp before the fp8 round, and near an e4m3 tie that ulp changes the
// emitted byte, which is why G1 compares BYTES and not an Approx: doctest's
// Approx carries a `scale` term defaulting to 1.0 and therefore a ~1.19e-5
// absolute floor, meaningless for a byte compare.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// The INDEPENDENT e4m3fn reference. Written from the format: 1 sign bit, 4
// exponent bits (bias 7), 3 mantissa bits, NO infinities, and 0x7F/0xFF the only
// NaN encodings (that is what the "fn" in e4m3fn means). Nothing here reads any
// codec in src/.
constexpr float kE4m3MaxFinite = 448.0f;  // = 1.75 * 2^8, encoding 0x7E

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
      // Strictly nearer wins. On an EXACT tie prefer the even significand — the
      // tie-break also carries across an exponent step, because mant 7 (odd) at
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

// The whole upstream expression, in upstream's order and upstream's f32 width:
// form the reciprocal ONCE, multiply, clamp, convert RNE.
uint8_t RefQuantFp8Static(float x, float input_scale) {
  const float inv = 1.0f / input_scale;                                   // common.cu:31
  const float scaled = x * inv;                                           // common.cuh:62
  const float r = std::fmax(-kE4m3MaxFinite, std::fmin(scaled, kE4m3MaxFinite));  // :68
  return RefEncodeRne(r);                                                 // :71 RNE cvt
}

// Dequant for the GEMM reference: the same format decode, sign restored.
double RefDequant(uint8_t byte) {
  const double m = E4m3Exact(static_cast<unsigned>(byte >> 3) & 0xFu,
                             static_cast<unsigned>(byte) & 0x7u);
  return (byte & 0x80u) != 0 ? -m : m;
}

// The G1 input population. Deliberately not just "random in a nice range":
// overflow in BOTH signs, the subnormal ladder, exact ties at several exponents,
// and both zeros — every input class whose handling a mutation can break.
std::vector<float> G1Inputs(float input_scale, uint32_t seed) {
  std::vector<float> v;
  // Random bulk.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  for (int i = 0; i < 4096; ++i) v.push_back(ux(rng));
  // Saturation, both signs: these scale past +-448 and MUST clamp.
  for (float s : {600.0f, 448.5f, 1e4f, 1e30f}) {
    v.push_back(s * input_scale);
    v.push_back(-s * input_scale);
  }
  // Exactly the largest finite value, and just inside it.
  v.push_back(448.0f * input_scale);
  v.push_back(-448.0f * input_scale);
  v.push_back(447.0f * input_scale);
  // Exact ties in the NORMAL range: midpoints between adjacent e4m3 values at
  // several exponents. RNE must pick the even mantissa.
  for (int e = -4; e <= 8; ++e) {
    for (int m = 0; m < 7; ++m) {
      const double lo = std::ldexp(1.0 + m / 8.0, e);
      const double hi = std::ldexp(1.0 + (m + 1) / 8.0, e);
      const auto mid = static_cast<float>((lo + hi) / 2.0);
      v.push_back(mid * input_scale);
      v.push_back(-mid * input_scale);
    }
  }
  // The SUBNORMAL ladder and its midpoints (values m/512 and (2m+1)/1024).
  for (int m = 0; m <= 8; ++m) {
    v.push_back(static_cast<float>(m / 512.0) * input_scale);
    v.push_back(static_cast<float>((2 * m + 1) / 1024.0) * input_scale);
    v.push_back(-static_cast<float>((2 * m + 1) / 1024.0) * input_scale);
  }
  // Both zeros (e4m3fn has a signed zero) and a value that rounds to zero.
  v.push_back(0.0f);
  v.push_back(-0.0f);
  v.push_back(static_cast<float>(1.0 / 4096.0) * input_scale);
  return v;
}

// Runs the CPU op over `x` and compares BYTE FOR BYTE against the reference.
// Returns the number of differing bytes so a caller can report a rate.
size_t RunG1(const std::vector<float>& x, float input_scale, DType x_dtype) {
  const auto n = static_cast<int64_t>(x.size());
  Queue q{Cpu(), nullptr};

  // Materialize x at the requested width. For bf16 the REFERENCE consumes the
  // bf16-rounded value too, so this compares the codec and not the store width.
  std::vector<float> xf32(x);
  std::vector<uint16_t> xbf16(x.size());
  for (size_t i = 0; i < x.size(); ++i) xbf16[i] = vt::F32ToBF16(x[i]);
  std::vector<float> ref_in(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    ref_in[i] = x_dtype == DType::kBF16 ? vt::BF16ToF32(xbf16[i]) : xf32[i];

  void* xp = x_dtype == DType::kBF16 ? static_cast<void*>(xbf16.data())
                                     : static_cast<void*>(xf32.data());
  Tensor tx = MakeTensor(xp, x_dtype, Cpu(), {1, n});
  std::vector<uint8_t> got(x.size());
  Tensor tout = MakeTensor(got.data(), DType::kI8, Cpu(), {1, n});
  vt::QuantFp8Static(q, tout, tx, input_scale);

  std::vector<uint8_t> want(x.size());
  for (size_t i = 0; i < x.size(); ++i) want[i] = RefQuantFp8Static(ref_in[i], input_scale);

  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(ref_in[first_bad]);
    CAPTURE(static_cast<int>(got[first_bad]));
    CAPTURE(static_cast<int>(want[first_bad]));
  }
  CHECK(bad == 0);
  // VACUITY GUARD: a kernel that wrote nothing, or a population that is all
  // zeros, would compare equal to a reference that also produced zeros. Demand
  // that the run actually produced a spread of distinct non-zero encodings.
  size_t nonzero = 0;
  for (auto b : want) {
    if ((b & 0x7Fu) != 0u) ++nonzero;
  }
  CHECK(nonzero > want.size() / 2);
  return bad;
}

}  // namespace

// ===========================================================================
// G1 — bitwise, zero tolerance.
TEST_CASE("G1: CPU QuantFp8Static is BYTE-identical to an independent e4m3 reference") {
  // The registration itself is the thing under test; a missing one must fail
  // here rather than surface as a confusing throw inside the helper.
  REQUIRE(vt::OpRegistered(vt::OpId::kQuantFp8Static, DeviceType::kCPU));

  // THE SCALE SET IS PART OF THE GATE, not decoration. 1.0 and 0.5 make the tie
  // inputs above land EXACTLY on e4m3 midpoints, so RNE alone decides every one
  // of them. 0.035 / 0.0092 are production-shaped per-tensor scales.
  //
  // 0.13 and 0.77 are here for ONE measured reason: they are what makes the
  // `x/s` vs `x*(1/s)` defect visible. Both forms agree on almost every input —
  // over 20000 random values in [-2,2] they NEVER disagree at any scale tried —
  // so a gate can only see the difference where an input lands on an e4m3 tie
  // after scaling, which is exactly what the tie population above constructs.
  // Even then it is scale-dependent: measured over the structured population,
  // 10 of 18 candidate scales expose it at all, and of {1.0, 0.5, 0.035, 0.0092,
  // 7.25} only 0.0092 does, at 24 of 209 words. 0.13 (78/209) and 0.77 (82/209)
  // are the strongest detectors found, so the mutation dies by a wide margin
  // rather than by luck. DO NOT prune this list: removing the last detecting
  // scale would silently disarm the assertion that keeps the reciprocal form.
  for (float s : {1.0f, 0.5f, 0.035f, 0.0092f, 7.25f, 0.13f, 0.77f}) {
    CAPTURE(s);
    const auto x = G1Inputs(s, 1234u);
    CHECK(RunG1(x, s, DType::kF32) == 0u);
    CHECK(RunG1(x, s, DType::kBF16) == 0u);
  }
}

// ===========================================================================
// G2 — the CPU registration must agree with the CUDA kernel BIT for BIT on the
// same input. This is the arm that says the CPU path is a mirror of what ships,
// not merely self-consistent with a host reference.
// NAME THIS CASE WITHOUT A COMMA. doctest splits `-tc=` on commas, so a comma in
// a case name makes the name unselectable: the filter becomes two patterns that
// each match nothing, and the binary then reports
//   test cases: 0 | 0 passed | 0 failed | 4 skipped ... Status: SUCCESS!
// with exit 0. That is the worst failure mode available to a gate -- the ONE
// arm this row still owes would have selected nothing and reported success.
// Measured on this file before the rename (#468 review F6).
TEST_CASE("G2: CPU QuantFp8Static equals CUDA QuantFp8Static byte for byte") {
  if (!HasCuda()) {
    // NOT a silent skip. The CPU registration is still asserted so the case can
    // never be vacuous, and the banner names what is owed.
    MESSAGE("G2 PENDING: no CUDA device on this host, CPU-vs-CUDA byte agreement "
            "was NOT measured (gate hosts: dgx.casa GB10/sm_121, 192.168.68.23 Thor/sm_110)");
    CHECK(vt::OpRegistered(vt::OpId::kQuantFp8Static, DeviceType::kCPU));
    return;
  }
  vt::Backend& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  Queue cq{Cpu(), nullptr};
  const Device gpu{DeviceType::kCUDA, 0};

  for (float s : {1.0f, 0.5f, 0.035f, 0.0092f, 7.25f}) {
    CAPTURE(s);
    const auto x = G1Inputs(s, 4321u);
    const auto n = static_cast<int64_t>(x.size());

    std::vector<uint8_t> cpu_out(x.size());
    std::vector<float> xf(x);
    Tensor tx_cpu = MakeTensor(xf.data(), DType::kF32, Cpu(), {1, n});
    Tensor to_cpu = MakeTensor(cpu_out.data(), DType::kI8, Cpu(), {1, n});
    vt::QuantFp8Static(cq, to_cpu, tx_cpu, s);

    void* dx = b.Alloc(xf.size() * sizeof(float));
    void* dout = b.Alloc(xf.size());
    b.Copy(gq, dx, xf.data(), xf.size() * sizeof(float));
    Tensor tx_gpu = MakeTensor(dx, DType::kF32, gpu, {1, n});
    Tensor to_gpu = MakeTensor(dout, DType::kI8, gpu, {1, n});
    vt::QuantFp8Static(gq, to_gpu, tx_gpu, s);
    std::vector<uint8_t> gpu_out(x.size());
    b.Copy(gq, gpu_out.data(), dout, gpu_out.size());
    b.Synchronize(gq);
    b.Free(dx);
    b.Free(dout);

    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
      if (cpu_out[i] != gpu_out[i]) {
        if (bad == 0) first_bad = i;
        ++bad;
      }
    }
    if (bad != 0) {
      CAPTURE(bad);
      CAPTURE(first_bad);
      CAPTURE(x[first_bad]);
      CAPTURE(static_cast<int>(cpu_out[first_bad]));
      CAPTURE(static_cast<int>(gpu_out[first_bad]));
    }
    CHECK(bad == 0);
  }
  b.DestroyQueue(gq);
}

// ===========================================================================
// G3 — the GEMM, against a reference that is LOSSY in exactly the places
// upstream is lossy.
//
// The reference quantizes the activation through the SAME clamp + e4m3 RNE the
// hardware path uses and then dequantizes both operands back, so it computes
// `alpha * Sum_k f8val(a) * f8val(b)`. A reference that instead accumulated the
// pre-quant f32 activations would be a DIFFERENT, more accurate computation, and
// a wrong implementation could sit closer to it than the correct one does — the
// gate would reward being unlike upstream. Only the accumulation WIDTH differs
// here (double vs the kernel's f32), and the tolerance bounds exactly that:
// `4 * K * FLT_EPSILON * alpha * Sum|terms|` is the standard forward bound on a
// K-term f32 recursive sum, with a factor-4 margin. It scales with the terms,
// not with the (possibly cancelled) result, which is what makes it tight.
namespace {
void RunG3(int M, int N, int K, uint32_t seed, float input_scale, float weight_scale,
           DType out_dtype) {
  CAPTURE(M);
  CAPTURE(N);
  CAPTURE(K);
  CAPTURE(seed);
  const float alpha = input_scale * weight_scale;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_int_distribution<int> ub(0, 255);

  std::vector<float> x(static_cast<size_t>(M) * K);
  for (auto& v : x) v = ux(rng);
  std::vector<uint8_t> b_fp8(static_cast<size_t>(N) * K);
  for (auto& v : b_fp8) {
    int byte = ub(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;  // avoid the NaN encodings
    v = static_cast<uint8_t>(byte);
  }

  Queue q{Cpu(), nullptr};
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {M, K});
  std::vector<uint8_t> a_fp8(x.size());
  Tensor ta = MakeTensor(a_fp8.data(), DType::kI8, Cpu(), {M, K});
  vt::QuantFp8Static(q, ta, tx, input_scale);

  Tensor tb = MakeTensor(b_fp8.data(), DType::kI8, Cpu(), {N, K});
  const size_t out_n = static_cast<size_t>(M) * N;
  std::vector<float> out_f32(out_dtype == DType::kF32 ? out_n : 0);
  std::vector<uint16_t> out_bf16(out_dtype == DType::kBF16 ? out_n : 0);
  void* outp = out_dtype == DType::kF32 ? static_cast<void*>(out_f32.data())
                                        : static_cast<void*>(out_bf16.data());
  Tensor tout = MakeTensor(outp, out_dtype, Cpu(), {M, N});
  vt::MatmulFp8Cutlass(q, tout, ta, tb, alpha);

  size_t bad = 0, first_bad = 0;
  double worst_ratio = 0.0;
  size_t nonzero = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0, abs_sum = 0.0;
      for (int k = 0; k < K; ++k) {
        const double t = RefDequant(a_fp8[static_cast<size_t>(m) * K + k]) *
                         RefDequant(b_fp8[static_cast<size_t>(n) * K + k]);
        acc += t;
        abs_sum += std::fabs(t);
      }
      const double want = static_cast<double>(alpha) * acc;
      const size_t i = static_cast<size_t>(m) * N + n;
      const double got = out_dtype == DType::kF32 ? static_cast<double>(out_f32[i])
                                                  : static_cast<double>(vt::BF16ToF32(out_bf16[i]));
      // f32 recursive-sum forward bound, x4 margin; a bf16 store adds its own
      // half-ulp (2^-9 relative), which is what the second term admits.
      const double tol = 4.0 * K * static_cast<double>(std::numeric_limits<float>::epsilon()) *
                             std::fabs(static_cast<double>(alpha)) * abs_sum +
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
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
  }
  CAPTURE(worst_ratio);
  CHECK(bad == 0);
  // VACUITY GUARD: an all-zero reference would make any implementation pass.
  CHECK(nonzero == out_n);
}
}  // namespace

TEST_CASE("G3: CPU MatmulFp8Cutlass matches a LOSSY double W8A8 reference") {
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8Cutlass, DeviceType::kCPU));
  // Decode (M=1) and small-prefill shapes; both output dtypes the op admits.
  RunG3(1, 64, 128, 101, 0.035f, 0.017f, DType::kF32);
  RunG3(4, 32, 256, 102, 0.035f, 0.017f, DType::kF32);
  RunG3(8, 48, 128, 103, 0.041f, 0.0092f, DType::kBF16);
  RunG3(3, 16, 64, 104, 1.0f, 1.0f, DType::kF32);
}

// ===========================================================================
// The seam the whole row exists for: QuantFp8Static -> MatmulFp8Cutlass, the
// exact pair vLLM's ModelOptFp8LinearMethod runs (modelopt.py:510-513), now
// executing end-to-end on a CPU queue. Without both registrations this case
// cannot even be written, which is the coverage debt #468 recorded.
TEST_CASE("the static fp8 W8A8 pair resolves and runs end-to-end on a CPU queue") {
  CHECK(vt::OpRegistered(vt::OpId::kQuantFp8Static, DeviceType::kCPU));
  CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8Cutlass, DeviceType::kCPU));
  // The cuBLASLt fp8 op deliberately stays CUDA-only (a "cuBLASLt" kernel on the
  // host would be a lie in the name), which is why the MODEL-layer predicate at
  // qwen3_5.cpp `MatmulFp8CutlassD` still refuses on CPU. Pinned here so the
  // residual gap recorded in the spec is visible rather than assumed closed.
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, DeviceType::kCPU));
}

// ===========================================================================
// G4 (issue #960) — THE REGISTRATION ITSELF, on every CUDA build.
//
// `QuantFp8Static`'s CUDA kernel is `x * (1/s)` plus a hardware e4m3 convert and
// has no cutlass dependency of any kind, but it USED TO SHARE a translation unit
// with the cutlass sm120 fp8 GEMM — and that TU is compiled only when
// `VT_CUTLASS_FP8_ARCHS` resolves non-empty. So on every CUDA arch outside that
// set (sm_110 is the measured one) `OpId::kQuantFp8Static` was not registered for
// `DeviceType::kCUDA` at all, the resolver installed the portable CPU reference
// tier for a CUDA queue, and the first real call dereferenced device pointers and
// took the process down. G2 above cannot state that: on a host WITHOUT the native
// kernel it crashes before it can report, and on a host WITH it the condition
// never arises. This case is the one that reads the same on both.
//
// It deliberately does NOT need a CUDA DEVICE — `OpRegistered` is a table lookup
// over registrars that ran before main, so it answers on any CUDA BUILD, which is
// exactly the axis the defect lived on.
#if defined(VLLM_CPP_CUDA)
TEST_CASE("G4: QuantFp8Static is registered for CUDA independent of cutlass-fp8") {
  CHECK(vt::OpRegistered(vt::OpId::kQuantFp8Static, DeviceType::kCUDA));
  // NON-VACUITY, and the actual claim: the quant registration is INDEPENDENT of
  // the cutlass one. Asserting only the line above would pass on a cutlass-fp8
  // host for the old reason as well as the new one. Here the cutlass GEMM is
  // required to track its own feature macro, so on a build where it is ABSENT
  // (Thor/sm_110) this case still proves the quant survived the arch gate, and on
  // a build where it is PRESENT (GB10/sm_121a) it proves nothing regressed.
#if defined(VT_CUTLASS_FP8)
  CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8Cutlass, DeviceType::kCUDA));
#else
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kMatmulFp8Cutlass, DeviceType::kCUDA));
#endif
}
#endif  // VLLM_CPP_CUDA
