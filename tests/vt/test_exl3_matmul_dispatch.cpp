// The EXL3 reconstruct dispatch through the production seam — QUANT-EXL3,
// ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2 item 1.
//
// #3150 ported exllamav3's `AUTO_RECONSTRUCT_THRESHOLD` (exl3.py:10,135) into
// `dense_attn::Exl3MatmulD`: M > 144 routes to `vt::Exl3ReconstructGemm`. That
// op is registered for CUDA only (`src/vt/cuda/cuda_exl3.cu`), while
// `vt::Exl3Gemm` is registered for CPU, ROCm and Vulkan too. A device-blind
// threshold therefore refused every EXL3 prefill above 144 tokens on those
// backends, which served every M through `Exl3Gemm` before #3150. Upstream's
// reconstruct path is CUDA-only as well, so the correct mirror on a backend with
// no reconstruct kernel is the cooperative kernel at every M.
//
// This suite enters through `Exl3MatmulD` on a CPU queue, which is the seam
// `layers::Exl3LinearMethod::Apply` delegates to, so it measures the dispatch a
// model forward reaches and not the kernels by hand. CPU-only; runs in CI.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3ChainF64;
using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rms;
using exl3_test::Rng;
using exl3_test::UlpF16;

vllm::Exl3Weight WrapFixture(const Exl3Fixture& f) {
  vllm::Exl3Weight w;
  w.codebook = 1;  // the codebook Exl3ChainF64 decodes by default
  const auto bytes_of = [](const std::vector<uint16_t>& v) {
    return vllm::OwnedBytes(std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(v.data()),
        reinterpret_cast<const uint8_t*>(v.data()) + v.size() * 2));
  };
  w.trellis.dtype = vt::DType::kI8;
  w.trellis.rank = 3;
  w.trellis.shape[0] = f.k / 16;
  w.trellis.shape[1] = f.n / 16;
  w.trellis.shape[2] = 32 * f.bits;
  w.trellis.bytes = bytes_of(f.trellis);
  w.suh.dtype = vt::DType::kF16;
  w.suh.rank = 1;
  w.suh.shape[0] = f.k;
  w.suh.bytes = bytes_of(f.suh);
  w.svh.dtype = vt::DType::kF16;
  w.svh.rank = 1;
  w.svh.shape[0] = f.n;
  w.svh.bytes = bytes_of(f.svh);
  return w;
}

}  // namespace

TEST_CASE("exl3 dispatch: M > 144 on a backend with no reconstruct kernel serves through Exl3Gemm") {
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();
  vllm::dense_attn::Dev d{b, q};

  // The premise, asserted rather than assumed: if a CPU reconstruct kernel is
  // ever registered, this case no longer tests the fallback and must be revised.
  REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU));

  // 145 is the first M upstream sends to reconstruct_hgemm (`rows <= 144` keeps
  // the cooperative kernel, exl3.py:135).
  const int64_t m = 145, k = 128, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150A11u);
  const vllm::Exl3Weight w = WrapFixture(f);

  Rng rng;
  rng.s = 0x0C0FFEEu;
  std::vector<float> x(static_cast<size_t>(m * k));
  // Through fp16 first: the seam stages the activation to fp16, and the
  // reference must read the same values.
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));

  vt::EnableOpProviderCallStats(true);
  const unsigned long long before =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;

  vllm::dense_attn::DBuf xb(d, vt::DType::kF32, {m, k}, x.data());
  std::vector<float> got(static_cast<size_t>(m * n), 0.0f);
  std::string refusal;
  try {
    vllm::dense_attn::DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF32);
    out.Download(d, got.data());
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  INFO("Exl3MatmulD refusal: " << refusal);
  REQUIRE(refusal.empty());

  // Positive signal that the cooperative kernel served the call, not merely
  // that something returned numbers.
  const unsigned long long after =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;
  vt::EnableOpProviderCallStats(false);
  CHECK(after > before);

  const std::vector<double> ref = Exl3ChainF64(f, x, m);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0, worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dlt = static_cast<double>(got[i]) - ref[i];
    sq += dlt * dlt;
    worst = std::max(worst, std::fabs(dlt));
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  MESSAGE("Exl3MatmulD M=145 cpu vs f64: rel_rms=", rel_rms, " worst=", worst);
  // The CPU arm's own tier-3 bound (test_exl3_gemm.cpp, spec `## W2 design` §1).
  CHECK(rel_rms <= 1.0e-3);
  CHECK(worst <= 8.0 * UlpF16(rms));

  b.DestroyQueue(q);
}
