// MODEL-FP8-BLOCK-LINEAR — #1189 milestone M4, spec
// `.agents/specs/model-fp8-block-linear.md`.
//
// A block-wise (fine-grained 128x128) FP8 checkpoint RUNS: the linear method,
// the shared compute body it wraps, and the ten Qwen3.5 dense projections that
// reach it.
//
// The load half is M3's and is gated by `test_fp8_block_weight_load.cpp`. What
// is new here is the CONSUMPTION, so the cases are built around three questions
// a numeric comparison alone cannot answer:
//
//   1. does a PRODUCTION entry point reach the arm at all (G3, the dispatch
//      counter through `ModelRegistry::Forward`);
//   2. is the weight still `N*K` fp8 bytes at the GEMM boundary, or did
//      something dequantize it to bf16 on the way — which is numerically BETTER
//      and therefore invisible to every correctness gate (G4);
//   3. is the scale really per BLOCK, or did it collapse to one number (G5,
//      which perturbs ONE block and requires the logits to move).
//
// No checkpoint download, no GPU, no snapshot. `vt::MatmulFp8BlockScaled` is a
// CPU correctness reference (M2) and that is what makes this gateable here; the
// CUDA kernel is #1189 M5 and until it lands a CUDA device refuses this
// checkpoint by name at `Prepare`, which G6 asserts.
#include <doctest/doctest.h>
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/fp8_block_fixture.h"
#include "vllm/model_executor/layers/quantization/fp8.h"
#include "vllm/model_executor/layers/quantization/fp8_block.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

// The synthetic model, the fp8 weight builder and the independent `double`
// GEMM reference live in `tests/support/fp8_block_fixture.h`, because #1189 M6
// asserts the MERGED block-GEMM count over the SAME model this file asserts the
// split count over. Two copies of that fixture would let exactly that pair
// drift.
using namespace fp8_block_fixture;  // NOLINT(build/namespaces)

namespace layers = vllm::layers;


// G1 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the factory selects the block method by weight presence") {
  OwnedTensor bf16 = MakeOwned(DType::kBF16, {128, 128}, 3, /*nk=*/true);
  const Fp8BlockWeight empty;
  const Fp8BlockWeight block = MakeFp8Block(128, 128, kBlock, kBlock, 31);
  REQUIRE(empty.Empty());
  REQUIRE_FALSE(block.Empty());

  // get_quant_method analogue (`fp8.py:297-298`): `weight_block_size is not
  // None` is the whole dispatch, resolved ONCE from the populated weight.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  auto m_block = layers::MakeLinearMethod(bf16, block);
  CHECK(std::string(m_block->Name()) == "fp8-w8a8-block");
  // The NAME alone would pass for a method that returned it from the wrong
  // class, so the concrete type is pinned too.
  CHECK(dynamic_cast<const layers::Fp8BlockLinearMethod*>(m_block.get()) !=
        nullptr);

  // The overloads coexist and are chosen by the weight TYPE, not by a runtime
  // probe: the per-tensor factory is in scope here and an `Fp8Weight` argument
  // still reaches it rather than this one.
  const vllm::Fp8Weight per_tensor;
  auto m_pt = layers::MakeLinearMethod(bf16, per_tensor);
  CHECK(std::string(m_pt->Name()) == "bf16-unquantized");
}

// G2 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: Apply is upstream's quant-then-block-GEMM, in f32, stored to out_dtype") {
  // block_n != block_k on purpose. Both are 128 in every checkpoint in play, so
  // a body that swapped the two arguments would be invisible at 128x128; here
  // the scale grid is [cdiv(N,64), cdiv(K,128)] and a swap misindexes it.
  const int64_t M = 3, K = 256, N = 192;
  const int64_t bn = 64, bk = 128;
  const Fp8BlockWeight w = MakeFp8Block(N, K, bn, bk, 77);
  REQUIRE(w.scale.shape[0] == CDiv(N, bn));
  REQUIRE(w.scale.shape[1] == CDiv(K, bk));

  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 5);
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  auto method = layers::MakeLinearMethod(MakeOwned(DType::kBF16, {N, K}, 1), w);
  REQUIRE(std::string(method->Name()) == "fp8-w8a8-block");

  SUBCASE("bf16 out, which is upstream's torch.get_default_dtype()") {
    vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kBF16);
    REQUIRE(out.t().dtype == DType::kBF16);
    REQUIRE(out.t().shape[0] == M);
    REQUIRE(out.t().shape[1] == N);

    // The activation half, run through the gated M1 op, so the reference below
    // measures the GEMM and the composition rather than re-deriving the e4m3
    // encoder a byte-exact gate already owns.
    vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {M, K});
    vllm::dense_attn::DBuf a_scale(d, DType::kF32, {M, K / bk});
    vt::QuantFp8Group(q, a_fp8.t(), a_scale.t(), x.t(), static_cast<int>(bk));
    std::vector<uint8_t> a_bytes(static_cast<size_t>(M * K));
    std::memcpy(a_bytes.data(), a_fp8.t().data, a_bytes.size());
    std::vector<float> a_s(static_cast<size_t>(M * (K / bk)));
    std::memcpy(a_s.data(), a_scale.t().data, a_s.size() * 4);

    const std::vector<double> ref = RefBlockGemm(a_bytes, a_s, M, K, w);
    const std::vector<float> got = ToF32(out.t(), M * N);

    // bf16 carries 8 significand bits, so the store rounds at ~2^-9 relative.
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double denom = std::max(std::abs(ref[i]), 1e-6);
      worst = std::max(worst, std::abs(got[i] - ref[i]) / denom);
    }
    CHECK(worst < 8e-3);
    // A vacuity guard: an all-zero output satisfies every relative comparison
    // whose reference is also zero, and this one would not be.
    CHECK(CountNonZero(got) == static_cast<int64_t>(got.size()));
  }

  SUBCASE("f32 out is the same value, stored wider") {
    vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kF32);
    REQUIRE(out.t().dtype == DType::kF32);
    const std::vector<float> got = ToF32(out.t(), M * N);
    CHECK(CountNonZero(got) == static_cast<int64_t>(got.size()));

    vllm::dense_attn::DBuf a_fp8(d, DType::kI8, {M, K});
    vllm::dense_attn::DBuf a_scale(d, DType::kF32, {M, K / bk});
    vt::QuantFp8Group(q, a_fp8.t(), a_scale.t(), x.t(), static_cast<int>(bk));
    std::vector<uint8_t> a_bytes(static_cast<size_t>(M * K));
    std::memcpy(a_bytes.data(), a_fp8.t().data, a_bytes.size());
    std::vector<float> a_s(static_cast<size_t>(M * (K / bk)));
    std::memcpy(a_s.data(), a_scale.t().data, a_s.size() * 4);
    const std::vector<double> ref = RefBlockGemm(a_bytes, a_s, M, K, w);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double denom = std::max(std::abs(ref[i]), 1e-6);
      worst = std::max(worst, std::abs(got[i] - ref[i]) / denom);
    }
    // f32 accumulation against a double reference: two orders tighter than the
    // bf16 store above, which is the whole point of asserting both.
    CHECK(worst < 5e-5);
  }
}

// G3 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: ModelRegistry::Forward REACHES the block arm at every projection") {
  // The reachability case (`.agents/reachability.md`). It enters at
  // `ModelRegistry::Forward` — a production entry point — and not at the linear
  // method, so deleting a production call site in the dense forward reds it.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);

  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);

  // The instrument a token gate cannot fake: the forward dispatched exactly one
  // block-scaled GEMM per quantized projection GROUP. A count of 0 means
  // nothing reached the arm; a count below the expected one means a projection
  // silently took another. The expected value dropped from 13 to 9 when #1189
  // M6 merged `gate_up` and QKV, which is what a merged topology looks like
  // from here; the fixture's own comment derives the figure.
  CHECK(dispatched == static_cast<uint64_t>(kBlockProjectionsPerForward));
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));

  // The f32 KV cache (`VT_KV_CACHE_F32=1`) is NOT exercised here and cannot be:
  // this arm emits a bf16 V at `v_proj`, which is upstream's `out_dtype` and the
  // model dtype, and `vt::ReshapeAndCache`'s auto path requires one shared
  // dtype exactly as upstream's `reshape_and_cache_flash` does. That refusal
  // hits every bf16 arm in the tree rather than this row's, so it is issue
  // #1249 and the row's spec lists it under `## Owed` instead of being widened
  // into M4.
}

// G4 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the weight stays fp8 bytes plus a scale grid, not a bf16 expansion") {
  // A silent dequant to bf16 is numerically BETTER than the quantized path and
  // therefore invisible to every value comparison in this file. Only the byte
  // counts can see it.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);
  const Fp8BlockWeight& q = w.layers[1].attn.q_proj_fp8_block;
  const int64_t N = q.n, K = q.k;
  REQUIRE(N == 256);
  REQUIRE(K == 128);

  CHECK_MESSAGE(q.packed.bytes.size() == static_cast<size_t>(N * K),
                "one fp8 byte per element; a bf16 expansion would be "
                    << (2 * N * K));
  CHECK(q.packed.dtype == DType::kI8);
  CHECK(q.scale.dtype == DType::kF32);
  CHECK(q.scale.bytes.size() ==
        static_cast<size_t>(CDiv(N, kBlock) * CDiv(K, kBlock)) * 4);
  // The ragged axis: the GDN `in_proj_qkv` is 192 wide against a 128 block, so
  // its scale grid has TWO rows and the second one is short. A floor tiling
  // would give one row and index out of the grid on every row above 127.
  const Fp8BlockWeight& g = w.layers[0].gdn.in_proj_qkv_fp8_block;
  CHECK(g.n == 192);
  CHECK(g.scale.shape[0] == 2);
  CHECK(g.scale.bytes.size() == static_cast<size_t>(2 * CDiv(g.k, kBlock)) * 4);

  // And the resident upload keeps both, at the same widths, on the device the
  // GEMM reads them from.
  vt::Queue qq = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, qq};
  const vt::Tensor packed = block_gemm::ResidentFp8BlockPacked(d, q);
  const vt::Tensor scale = block_gemm::ResidentFp8BlockScale(d, q);
  CHECK(packed.dtype == DType::kI8);
  CHECK(packed.shape[0] == N);
  CHECK(packed.shape[1] == K);
  CHECK(scale.dtype == DType::kF32);
  CHECK(scale.shape[0] == CDiv(N, kBlock));
  CHECK(scale.shape[1] == CDiv(K, kBlock));
}

// G5 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: ONE perturbed block scale moves the model's logits") {
  // The issue's gate design, point 2: a token gate cannot see a scale
  // perturbation (`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp:43-45`
  // records x1.02 and x1.10 both producing 16/16 identical tokens). This is the
  // instrument that can. The bumped element is row 1, column 0 of ONE
  // projection's grid, so a per-tensor collapse to (0,0), an epilogue-folded
  // alpha and a transposed scale index are each blind to it.
  const HfConfig c = BlockConfig();
  const std::vector<float> base =
      RegistryForward(c, BlockWeights(c, true, 1.0F), nullptr);
  const std::vector<float> bumped =
      RegistryForward(c, BlockWeights(c, true, 1.10F), nullptr);
  REQUIRE(base.size() == bumped.size());

  double worst = 0.0;
  for (size_t i = 0; i < base.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(base[i]) - bumped[i]));
  CHECK_MESSAGE(worst > 1e-4,
                "a x1.10 bump on one 128x128 block moved the logits by "
                    << worst);

  // The control: the SAME construction with no bump is bit-reproducible, so the
  // difference above is the scale and not run-to-run noise.
  const std::vector<float> again =
      RegistryForward(c, BlockWeights(c, true, 1.0F), nullptr);
  CHECK(std::memcmp(base.data(), again.data(), base.size() * 4) == 0);
}

// G6 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: a device with no block-scaled GEMM refuses at Prepare, by name") {
  // M3 refused every LOADED block weight because nothing could read one. That
  // is narrowed, not deleted: the refusal now asks the PREPARE queue's device
  // whether it has a kernel. CUDA does not until #1189 M5, and a user finding
  // that out at the first GEMM — after a graph capture — is the failure this
  // prevents.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);

  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, vt::DeviceType::kCPU));
  REQUIRE_FALSE(
      vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, vt::DeviceType::kCUDA));

  std::string message;
  try {
    vllm::RefuseUnrunnableQwen3_5DenseFp8Block(w, vt::DeviceType::kCUDA);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("block-wise") != std::string::npos);
  CHECK(message.find("in_proj_qkv") != std::string::npos);
  CHECK(message.find("cuda") != std::string::npos);
  CHECK(message.find("1189") != std::string::npos);

  // On a device that HAS the kernel it is inert. Without this the case passes
  // for a refusal that never stopped refusing.
  CHECK_NOTHROW(
      vllm::RefuseUnrunnableQwen3_5DenseFp8Block(w, vt::DeviceType::kCPU));
  // And a checkpoint with no block weight is inert on every device.
  const Qwen3_5DenseWeights plain = BlockWeights(c, /*block_arm=*/false);
  CHECK_NOTHROW(
      vllm::RefuseUnrunnableQwen3_5DenseFp8Block(plain, vt::DeviceType::kCUDA));

  // Through the PRODUCTION call site, not just the function: deleting the call
  // in `PrepareQwen3_5Dense` must red this. The queue names a CUDA device and
  // carries no handle, which is safe precisely BECAUSE the refusal runs first
  // in that function — before `PrepareLmHeadResident` or anything else touches
  // a backend. If the refusal is ever moved below them, this stops being a
  // clean throw and says so.
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  vt::Queue cuda_q{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  std::string routed;
  try {
    ModelRegistry::Prepare(*model, c, cuda_q);
  } catch (const std::exception& e) {
    routed = e.what();
  }
  REQUIRE_FALSE(routed.empty());
  CHECK(routed.find("block-wise") != std::string::npos);
  CHECK(routed.find("cuda") != std::string::npos);
  CHECK(routed.find("1189") != std::string::npos);

  // The same call site on a device that CAN run it does not refuse.
  std::unique_ptr<vllm::LoadedModel> cpu_model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  vt::Queue cpu_q = Q();
  CHECK_NOTHROW(ModelRegistry::Prepare(*cpu_model, c, cpu_q));
}

// G7 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the bf16 arm is unchanged and dispatches no block GEMM") {
  // Without this the gate passes for a forward that routed EVERY projection
  // through the block arm.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/false);
  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);
  CHECK(dispatched == 0);
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));
}

// G8 -----------------------------------------------------------------------
TEST_CASE("fp8 block linear: the body refuses what it cannot compute, by name") {
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  SUBCASE("a K the activation quant cannot group") {
    // Upstream asserts the same divisibility (`fp8_utils.py:596-599`). The
    // refusal is the BODY's, so it names the shape and the block rather than
    // surfacing from inside vt one frame deeper.
    const int64_t M = 2, K = 192, N = 128;
    const Fp8BlockWeight w = MakeFp8Block(N, K, kBlock, kBlock, 9);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K});
    auto method = layers::MakeLinearMethod(OwnedTensor{}, w);
    CHECK_THROWS_WITH_AS(method->Apply(d, x.t(), DType::kBF16),
                         doctest::Contains("192"), std::runtime_error);
  }

  SUBCASE("an activation whose K disagrees with the weight's") {
    const int64_t M = 2, K = 256, N = 128;
    const Fp8BlockWeight w = MakeFp8Block(N, 128, kBlock, kBlock, 9);
    vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K});
    auto method = layers::MakeLinearMethod(OwnedTensor{}, w);
    CHECK_THROWS_AS(method->Apply(d, x.t(), DType::kBF16), std::runtime_error);
  }
}
