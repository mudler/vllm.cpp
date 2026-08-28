// The EXL3 decode, against REAL exllamav3-produced data — QUANT-EXL3 (#2181).
//
// This is the gate the tree did not have, and its absence is what let a wrong
// codebook ship. Every other EXL3 fixture here is random bytes, where any
// codebook and any tile permutation is self-consistent: our decode agrees with
// our reference because both read the same wrong constant. Only data produced
// by exllamav3, checked against the weight it approximates, can falsify it.
//
// The reference is the UNQUANTIZED tensor, so this is a CORRELATION gate rather
// than an equality one — 3 bits per weight is a real approximation and the
// elementwise error is large. It is decisive anyway, because the failure mode
// it exists for is not a small error. A wrong codebook multiplier produces a
// codebook with the SAME DISTRIBUTION and no relation to the right one, so the
// score is ~0 rather than ~0.9: measured -0.0006 for codebook 1 against +0.9896
// for codebook 0 on the full tensor.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#include "exl3_real_corner.inc"

namespace {

double Cosine(const std::vector<float>& a, const std::vector<uint16_t>& ref_bf16) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = a[i];
    const double y = vt::BF16ToF32(ref_bf16[i]);
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return (na == 0.0 || nb == 0.0) ? 0.0 : dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<float> Decode(int codebook) {
  std::vector<float> w(static_cast<size_t>(exl3_real::kK) * exl3_real::kN, 0.0f);
  vt::Exl3DequantLinear(exl3_real::kTrellis, exl3_real::kSuh, exl3_real::kSvh, exl3_real::kK,
                        exl3_real::kN, exl3_real::kBits, codebook, w.data());
  return w;
}

}  // namespace

TEST_CASE("exl3 real data: codebook 0 reconstructs the unquantized weight") {
  const std::vector<float> w = Decode(exl3_real::kCodebook);
  const std::vector<uint16_t> ref(exl3_real::kRefBf16,
                                  exl3_real::kRefBf16 + (exl3_real::kK * exl3_real::kN));
  REQUIRE(w.size() == ref.size());

  const double cos = Cosine(w, ref);
  MESSAGE("cosine(decoded cb0, unquantized) = ", cos);
  // The full tensor scores 0.9896. A 128x128 corner is a smaller sample of the
  // same distribution, so the bound is loose enough not to be a near-tie and
  // far above anything a wrong constant can reach.
  CHECK(cos > 0.90);

  // Scale, independently of direction: a decode that got the codebook right but
  // the Hadamard or the sign vectors wrong can still correlate while sitting at
  // the wrong magnitude.
  double s2 = 0.0, r2 = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    s2 += static_cast<double>(w[i]) * w[i];
    const double r = vt::BF16ToF32(ref[i]);
    r2 += r * r;
  }
  const double rms_ours = std::sqrt(s2 / w.size());
  const double rms_ref = std::sqrt(r2 / ref.size());
  MESSAGE("rms ours=", rms_ours, " ref=", rms_ref);
  CHECK(rms_ours == doctest::Approx(rms_ref).epsilon(0.15));
}

TEST_CASE("exl3 real data: the WRONG codebook scores ~0, which is why absence must not mean MCG") {
  // The whole point. Decoding this artifact as MCG is not a small error and not
  // a loud one: it is a different, identically-distributed weight. If this case
  // ever starts passing at a high score, the two codebooks have stopped being
  // distinguishable and the selection gate above has stopped meaning anything.
  const std::vector<float> wrong = Decode(1);
  const std::vector<uint16_t> ref(exl3_real::kRefBf16,
                                  exl3_real::kRefBf16 + (exl3_real::kK * exl3_real::kN));
  const double cos = Cosine(wrong, ref);
  MESSAGE("cosine(decoded cb1 == WRONG here, unquantized) = ", cos);
  CHECK(std::abs(cos) < 0.10);

  // And it is NOT distinguishable by magnitude, which is the trap: the wrong
  // codebook lands within a few percent of the right RMS.
  double s2 = 0.0, r2 = 0.0;
  for (size_t i = 0; i < wrong.size(); ++i) {
    s2 += static_cast<double>(wrong[i]) * wrong[i];
    const double r = vt::BF16ToF32(ref[i]);
    r2 += r * r;
  }
  CHECK(std::sqrt(s2 / wrong.size()) == doctest::Approx(std::sqrt(r2 / ref.size())).epsilon(0.25));
}

TEST_CASE("exl3 real data: the fused GEMM agrees with the decode on the same real weights") {
  // `Exl3Gemm` computes the same linear with the Hadamards riding the
  // ACTIVATIONS instead of the weights (`exl3.py:183-214` vs `:227-237`). The
  // synthetic suites already gate that identity; doing it here as well is what
  // ties the GEMM arm to real data rather than to random bytes.
  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  const int64_t m = 2, k = exl3_real::kK, n = exl3_real::kN;
  std::vector<uint16_t> a(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = vt::F32ToF16(0.05f * static_cast<float>((i * 37) % 19) - 0.45f);

  std::vector<uint16_t> a_had(a.size(), 0);
  std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
  vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF16, q.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, q.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(c.data(), vt::DType::kF32, q.device, {m, n});
  vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(exl3_real::kTrellis),
                                         vt::DType::kI8, q.device,
                                         {k / 16, n / 16, 32 * exl3_real::kBits});
  vt::Tensor tsuh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(exl3_real::kSuh), vt::DType::kF16, q.device, {k});
  vt::Tensor tsvh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(exl3_real::kSvh), vt::DType::kF16, q.device, {n});
  vt::Exl3GemmArgs args;
  args.bits = exl3_real::kBits;
  args.codebook = exl3_real::kCodebook;
  vt::Exl3Gemm(q, tc, ta, tb, tsuh, tsvh, tah, args);

  const std::vector<float> w = Decode(exl3_real::kCodebook);
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk)
        acc += static_cast<double>(vt::F16ToF32(a[static_cast<size_t>(i * k + kk)])) *
               w[static_cast<size_t>(kk * n + j)];
      const double d = static_cast<double>(c[static_cast<size_t>(i * n + j)]) - acc;
      num += d * d;
      den += acc * acc;
    }
  const double rel = std::sqrt(num / den);
  MESSAGE("exl3_gemm vs weight-side decode on REAL data: rel_rms = ", rel);
  CHECK(rel <= 2.0e-3);
  REQUIRE(den > 0.0);
  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}
