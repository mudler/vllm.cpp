// LinearMethod / QuantizationConfig seam — scheme×device selection + bf16 apply.
//
// Ports vLLM's scheme-parameterized create_weights/apply coverage
// (tests/kernels/quantization/**) to the vt::-native seam (work row S4 of
// .agents/specs/accelerator-seam-audit.md): the scheme is chosen ONCE by the
// factory from the checkpoint's populated weights, and the bf16 UnquantizedLinear
// apply runs the exact vt::MatmulBT the inline model path did.
//
// CPU-only (no checkpoint), runs in CI. The NVFP4 numeric path is gated on dgx
// via the paged-engine model tests; here we assert the FACTORY SELECTS the right
// method per scheme (the S4 policy decision) and that the bf16 apply is correct.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vt::DType;
namespace layers = vllm::layers;

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, uint32_t seed) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = true;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  uint32_t s = seed;
  for (int64_t i = 0; i < numel; ++i) {
    s = s * 1664525u + 1013904223u;
    const float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
    p[i] = vt::F32ToBF16(v);
  }
  return o;
}

// A minimal non-empty W4A16 NVFP4 weight (alpha == 0): enough for the factory to
// select the quantized scheme. Its numeric path is exercised on dgx, not here.
Nvfp4Weight MakeNvfp4W4A16(int64_t N, int64_t K) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.scale2 = 1.0f;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * (K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 16;
  w.scale.bytes.resize(static_cast<size_t>(N) * (K / 16), 0);
  return w;
}

}  // namespace

TEST_CASE("linear_method: factory selects bf16 vs nvfp4-w4a16 by weight presence") {
  OwnedTensor bf16 = MakeBf16({4, 16}, 1);
  Nvfp4Weight empty_fp4;      // Empty() == true
  Nvfp4Weight fp4 = MakeNvfp4W4A16(4, 16);
  REQUIRE(empty_fp4.Empty());
  REQUIRE_FALSE(fp4.Empty());

  // get_quant_method analogue: a bf16 checkpoint => UnquantizedLinearMethod.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty_fp4);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  // An NVFP4-packed checkpoint => the compressed-tensors W4A16 method, chosen
  // ONCE here (not by a per-call IsNvfp4() probe in the model forward).
  auto m_fp4 = layers::MakeLinearMethod(bf16, fp4);
  CHECK(std::string(m_fp4->Name()) == "compressed-tensors-nvfp4-w4a16");
}

TEST_CASE("linear_method: gate_up factory selects scheme by weight presence") {
  OwnedTensor gate_up = MakeBf16({2 * 16, 8}, 2);
  Nvfp4Weight empty;
  Nvfp4Weight gate = MakeNvfp4W4A16(16, 8);
  Nvfp4Weight up = MakeNvfp4W4A16(16, 8);

  auto g_bf16 = layers::MakeMlpGateUpMethod(gate_up, empty, empty, 16);
  CHECK(std::string(g_bf16->Name()) == "bf16-gate-up");

  auto g_fp4 = layers::MakeMlpGateUpMethod(gate_up, gate, up, 16);
  CHECK(std::string(g_fp4->Name()) == "compressed-tensors-nvfp4-w4a16-gate-up");
}

TEST_CASE("linear_method: bf16 UnquantizedLinearMethod apply == reference MatmulBT") {
  const int64_t M = 2, K = 16, N = 4;
  OwnedTensor w = MakeBf16({N, K}, 7);  // raw-NK [N=out, K=in]
  OwnedTensor xw = MakeBf16({M, K}, 9);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());
  auto method = layers::MakeLinearMethod(w, Nvfp4Weight{});
  REQUIRE(std::string(method->Name()) == "bf16-unquantized");
  vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kF32);

  std::vector<float> got(static_cast<size_t>(M) * N);
  out.Download(d, got.data());

  const auto* wp = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const auto* xp = reinterpret_cast<const uint16_t*>(xw.bytes.data());
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += vt::BF16ToF32(xp[m * K + k]) * vt::BF16ToF32(wp[n * K + k]);
      CHECK(got[static_cast<size_t>(m) * N + n] == doctest::Approx(acc).epsilon(0.02));
    }
  }
}
