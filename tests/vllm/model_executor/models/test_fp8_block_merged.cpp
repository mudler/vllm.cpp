// MODEL-FP8-BLOCK-MERGED — #1189 milestone M6, spec
// `.agents/specs/model-fp8-block-merged.md`.
//
// The block-wise (fine-grained 128x128) FP8 arm runs ONE `gate_up` GEMM and ONE
// QKV GEMM, as vLLM does, instead of the five separate ones M4 landed.
//
// BYTE-IDENTITY IS THE LOAD-BEARING GATE HERE, not a tolerance. The whole claim
// of this row is that N-concatenating block-scaled shards is EXACT, so a
// merged output that merely agrees to some epsilon with the split one is a
// mis-concatenation — a scale row read one index off produces small errors, not
// large ones, and a tolerance would hide exactly the defect the guard exists
// for. Every comparison below is `memcmp`.
//
// The second thing a value comparison cannot answer is whether anything merged
// at all: the merged and split GEMMs produce the SAME bytes, so only the
// dispatch counter can tell the two topologies apart. G5 reads it through
// `ModelRegistry::Forward`.
//
// No checkpoint download, no GPU, no snapshot.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/fp8_block_fixture.h"
#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"
#include "vllm/model_executor/layers/quantization/fp8_block.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/merged_gemm.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using namespace fp8_block_fixture;  // NOLINT(build/namespaces)

namespace {

namespace layers = vllm::layers;
using vllm::Fp8BlockMergedResident;
using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vllm::dense_fp8_block::Fp8BlockMergedView;
using vllm::dense_fp8_block::Fp8BlockShard;

// The block GEMMs one forward dispatches once `gate_up` and QKV are merged.
// Derived in `tests/support/fp8_block_fixture.h`, which owns the model.
constexpr uint64_t kMergedBlockGemmsPerForward =
    static_cast<uint64_t>(kBlockProjectionsPerForward);

// The raw bytes of a device tensor, for a comparison that admits no epsilon.
std::vector<uint8_t> Bytes(const vt::Tensor& t, int64_t numel) {
  std::vector<uint8_t> out(static_cast<size_t>(numel) * vt::SizeOf(t.dtype));
  std::memcpy(out.data(), t.data, out.size());
  return out;
}

// One column range of a [M,N] tensor, row by row, as raw bytes. The merged
// output's shards are column ranges of one buffer while the split outputs are
// separate contiguous buffers, so the comparison has to walk rows.
std::vector<uint8_t> ColumnBytes(const vt::Tensor& t, int64_t M, int64_t N,
                                 int64_t c0, int64_t c1) {
  const size_t esz = vt::SizeOf(t.dtype);
  std::vector<uint8_t> out(static_cast<size_t>(M * (c1 - c0)) * esz);
  const auto* src = static_cast<const uint8_t*>(t.data);
  for (int64_t m = 0; m < M; ++m) {
    std::memcpy(out.data() + static_cast<size_t>(m * (c1 - c0)) * esz,
                src + (static_cast<size_t>(m * N) + static_cast<size_t>(c0)) * esz,
                static_cast<size_t>(c1 - c0) * esz);
  }
  return out;
}

// Give a shard's scale grid values of its OWN, and prove it did.
//
// A MEASURED weakness in the shared fixture, not a decoration.
// `MakeFp8Block`'s grid is `0.0625 + 0.125*((r*3 + c*5) % 5)` — a function of
// the POSITION only, so two shards of the same shape get byte-identical grids
// whatever their seeds. Under that fixture a merged operand that concatenated
// the scale grids in the WRONG ORDER produced the right answer, and the
// gate_up byte-identity case passed a mutation that reversed the copy loop.
// The scale values here are per-shard, so that mutation reds.
void VaryScale(vllm::Fp8BlockWeight& w, uint64_t seed) {
  auto* s = reinterpret_cast<float*>(w.scale.bytes.data());
  const int64_t n = w.scale.shape[0] * w.scale.shape[1];
  for (int64_t i = 0; i < n; ++i) {
    s[i] = 0.05F + 0.03F * static_cast<float>(Mix(seed + static_cast<uint64_t>(i)) % 17);
  }
}

// The instrument's own precondition: two shards' grids must actually differ,
// or a wrong-order concatenation is invisible to every comparison below.
void RequireScalesDiffer(const vllm::Fp8BlockWeight& a,
                         const vllm::Fp8BlockWeight& b) {
  REQUIRE(a.scale.bytes.size() == b.scale.bytes.size());
  REQUIRE(std::memcmp(a.scale.bytes.data(), b.scale.bytes.data(),
                      a.scale.bytes.size()) != 0);
}

}  // namespace

// G1 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: gate_up is BYTE-IDENTICAL to the split gate and up GEMMs") {
  const int64_t M = 3, K = 256, I = 128;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};

  // Different seeds, so the two shards' scale grids differ along BOTH axes and
  // a merged operand that read one shard's grid for both would fail.
  vllm::Fp8BlockWeight gate = MakeFp8Block(I, K, kBlock, kBlock, 4001);
  vllm::Fp8BlockWeight up = MakeFp8Block(I, K, kBlock, kBlock, 4002);
  VaryScale(gate, 4001);
  VaryScale(up, 4002);
  RequireScalesDiffer(gate, up);
  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 4003);
  DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  // The SPLIT reference: exactly what M4's dense MLP ran.
  DBuf sg = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), gate,
                                                               DType::kBF16);
  DBuf su = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), up,
                                                               DType::kBF16);
  DBuf split(d, DType::kBF16, {M, I});
  vt::MoeSiluMul(q, split.t(), sg.t(), su.t());

  // The MERGED arm.
  const Fp8BlockShard shards[2] = {{&gate, "gate_proj"}, {&up, "up_proj"}};
  Fp8BlockMergedResident resident;
  const Fp8BlockMergedView view = vllm::dense_fp8_block::ResidentFp8BlockMerged(
      d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, resident);
  REQUIRE(view.n_total == 2 * I);
  REQUIRE(view.scale.shape[0] == CDiv(2 * I, kBlock));
  DBuf merged = vllm::dense_fp8_block::Fp8BlockGateUpSwiGLUD<DBuf>(
      d, x.t(), view, DType::kBF16);
  REQUIRE(merged.t().dtype == DType::kBF16);
  REQUIRE(merged.t().shape[0] == M);
  REQUIRE(merged.t().shape[1] == I);

  CHECK(Bytes(merged.t(), M * I) == Bytes(split.t(), M * I));
  // A vacuity guard: two all-zero buffers are byte-identical too.
  CHECK(CountNonZero(ToF32(merged.t(), M * I)) == M * I);
}

// G2 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: qkv is BYTE-IDENTICAL to the three split GEMMs") {
  const int64_t M = 3, K = 128;
  const int64_t qn = 256, kn = 128, vn = 128;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};

  vllm::Fp8BlockWeight wq = MakeFp8Block(qn, K, kBlock, kBlock, 5001);
  vllm::Fp8BlockWeight wk = MakeFp8Block(kn, K, kBlock, kBlock, 5002);
  vllm::Fp8BlockWeight wv = MakeFp8Block(vn, K, kBlock, kBlock, 5003);
  VaryScale(wq, 5001);
  VaryScale(wk, 5002);
  VaryScale(wv, 5003);
  RequireScalesDiffer(wk, wv);
  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 5004);
  DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  DBuf sq = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wq,
                                                               DType::kBF16);
  DBuf sk = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wk,
                                                               DType::kBF16);
  DBuf sv = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wv,
                                                               DType::kBF16);

  const Fp8BlockShard shards[3] = {
      {&wq, "q_proj"}, {&wk, "k_proj"}, {&wv, "v_proj"}};
  Fp8BlockMergedResident resident;
  const Fp8BlockMergedView view = vllm::dense_fp8_block::ResidentFp8BlockMerged(
      d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);
  REQUIRE(view.n_total == qn + kn + vn);
  DBuf merged = vllm::dense_fp8_block::MatmulFp8BlockMergedD<DBuf>(
      d, x.t(), view, DType::kBF16, vt::kFp8BlockQkv);
  const int64_t N = qn + kn + vn;
  REQUIRE(merged.t().shape[1] == N);

  CHECK(ColumnBytes(merged.t(), M, N, 0, qn) == Bytes(sq.t(), M * qn));
  CHECK(ColumnBytes(merged.t(), M, N, qn, qn + kn) == Bytes(sk.t(), M * kn));
  CHECK(ColumnBytes(merged.t(), M, N, qn + kn, N) == Bytes(sv.t(), M * vn));
  CHECK(CountNonZero(ToF32(merged.t(), M * N)) == M * N);
}

// G3 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: a ragged NON-final shard is refused by name") {
  // Upstream's `validate_fp8_block_shape` (`fp8_utils.py:1229-1244`) exempts
  // only the LAST partition, and its escape hatch (`linear.py:532-557`) does
  // not rescue an earlier one: running upstream's own `adjust_block_scale_shard`
  // over q=256, k=64, v=64 at block_n=128 slices rows [3,4) of a 3-row scale
  // parameter. So this geometry has no correct merged GEMM anywhere, and we
  // refuse it rather than split or mis-slice in silence.
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};
  Fp8BlockMergedResident resident;

  SUBCASE("qkv, the exact geometry upstream slices off the end of its scale") {
    const vllm::Fp8BlockWeight wq = MakeFp8Block(256, 128, kBlock, kBlock, 6001);
    const vllm::Fp8BlockWeight wk = MakeFp8Block(64, 128, kBlock, kBlock, 6002);
    const vllm::Fp8BlockWeight wv = MakeFp8Block(64, 128, kBlock, kBlock, 6003);
    const Fp8BlockShard shards[3] = {
        {&wq, "q_proj"}, {&wk, "k_proj"}, {&wv, "v_proj"}};
    std::string message;
    try {
      vllm::dense_fp8_block::ResidentFp8BlockMerged(
          d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);
    } catch (const std::exception& e) {
      message = e.what();
    }
    REQUIRE_FALSE(message.empty());
    CHECK(message.find("qkv_proj") != std::string::npos);
    CHECK(message.find("k_proj") != std::string::npos);
    CHECK(message.find("64") != std::string::npos);
    CHECK(message.find("128") != std::string::npos);
    CHECK(message.find("LAST") != std::string::npos);
    // The refusal must fire BEFORE anything is allocated, so a caught throw
    // leaves no half-built resident behind for the next call to trust.
    CHECK(resident.d_packed == nullptr);
    CHECK(resident.d_scale == nullptr);
  }

  SUBCASE("gate_up, where the ragged shard is the gate") {
    const vllm::Fp8BlockWeight g = MakeFp8Block(100, 128, kBlock, kBlock, 6011);
    const vllm::Fp8BlockWeight u = MakeFp8Block(100, 128, kBlock, kBlock, 6012);
    const Fp8BlockShard shards[2] = {{&g, "gate_proj"}, {&u, "up_proj"}};
    std::string message;
    try {
      vllm::dense_fp8_block::ResidentFp8BlockMerged(
          d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, resident);
    } catch (const std::exception& e) {
      message = e.what();
    }
    REQUIRE_FALSE(message.empty());
    CHECK(message.find("gate_up_proj") != std::string::npos);
    CHECK(message.find("gate_proj") != std::string::npos);
    CHECK(message.find("100") != std::string::npos);
  }

  SUBCASE("a gated group whose two halves are unequal") {
    // vt::SiluAndMul splits [M,2D] at D, so an unequal pair mis-splits without
    // erroring. Upstream's gate_up is `output_sizes = [I, I]`.
    const vllm::Fp8BlockWeight g = MakeFp8Block(256, 128, kBlock, kBlock, 6021);
    const vllm::Fp8BlockWeight u = MakeFp8Block(128, 128, kBlock, kBlock, 6022);
    const Fp8BlockShard shards[2] = {{&g, "gate_proj"}, {&u, "up_proj"}};
    CHECK_THROWS_WITH_AS(
        vllm::dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, resident),
        doctest::Contains("EQUAL"), std::runtime_error);
  }

  SUBCASE("shards that disagree about K") {
    const vllm::Fp8BlockWeight g = MakeFp8Block(128, 128, kBlock, kBlock, 6031);
    const vllm::Fp8BlockWeight u = MakeFp8Block(128, 256, kBlock, kBlock, 6032);
    const Fp8BlockShard shards[2] = {{&g, "gate_proj"}, {&u, "up_proj"}};
    CHECK_THROWS_WITH_AS(
        vllm::dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, resident),
        doctest::Contains("ONE input"), std::runtime_error);
  }
}

// G4 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: a ragged FINAL shard merges, and is byte-identical") {
  // Upstream allows exactly this case, and its slicing produces exactly the
  // row-concatenation built here: for q=256, k=128, v=100 the shard grids land
  // at rows [0,2), [2,3), [3,4) of a 4-row parameter, and
  // sum(cdiv(N_i,128)) == 4 == cdiv(484,128).
  const int64_t M = 2, K = 128;
  const int64_t qn = 256, kn = 128, vn = 100;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};

  vllm::Fp8BlockWeight wq = MakeFp8Block(qn, K, kBlock, kBlock, 7001);
  vllm::Fp8BlockWeight wk = MakeFp8Block(kn, K, kBlock, kBlock, 7002);
  vllm::Fp8BlockWeight wv = MakeFp8Block(vn, K, kBlock, kBlock, 7003);
  VaryScale(wq, 7001);
  VaryScale(wk, 7002);
  VaryScale(wv, 7003);
  RequireScalesDiffer(wk, wv);
  REQUIRE(wv.scale.shape[0] == 1);  // the short final block
  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 7004);
  DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());

  DBuf sq = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wq,
                                                               DType::kBF16);
  DBuf sk = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wk,
                                                               DType::kBF16);
  DBuf sv = vllm::dense_fp8_block::MatmulFp8BlockScaledD<DBuf>(d, x.t(), wv,
                                                               DType::kBF16);

  const Fp8BlockShard shards[3] = {
      {&wq, "q_proj"}, {&wk, "k_proj"}, {&wv, "v_proj"}};
  Fp8BlockMergedResident resident;
  const Fp8BlockMergedView view = vllm::dense_fp8_block::ResidentFp8BlockMerged(
      d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);
  CHECK(view.n_total == qn + kn + vn);
  CHECK(view.scale.shape[0] == 4);
  CHECK(view.scale.shape[0] == CDiv(qn + kn + vn, kBlock));

  DBuf merged = vllm::dense_fp8_block::MatmulFp8BlockMergedD<DBuf>(
      d, x.t(), view, DType::kBF16, vt::kFp8BlockQkv);
  const int64_t N = qn + kn + vn;
  CHECK(ColumnBytes(merged.t(), M, N, 0, qn) == Bytes(sq.t(), M * qn));
  CHECK(ColumnBytes(merged.t(), M, N, qn, qn + kn) == Bytes(sk.t(), M * kn));
  CHECK(ColumnBytes(merged.t(), M, N, qn + kn, N) == Bytes(sv.t(), M * vn));
  CHECK(CountNonZero(ToF32(merged.t(), M * N)) == M * N);
}

// G5 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: ModelRegistry::Forward runs ONE gate_up and ONE qkv GEMM") {
  // The reachability case (`.agents/reachability.md`). It enters at
  // `ModelRegistry::Forward` — a production entry point — and not at the merge
  // helper, so deleting either production merge call site reds it.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/true);

  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);

  CHECK_MESSAGE(dispatched == kMergedBlockGemmsPerForward,
                "block GEMMs per forward: " << dispatched << ", merged expects "
                                            << kMergedBlockGemmsPerForward);
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));

  // Bit-reproducible across runs. The merged operand is a NEW device buffer
  // built from a pool, and pooled storage is not zeroed, so a concatenation
  // that left any byte unwritten would read whatever the previous tenant left
  // and would differ run to run while staying finite and non-vacuous.
  uint64_t again_dispatched = 0;
  const std::vector<float> again = RegistryForward(c, w, &again_dispatched);
  REQUIRE(again.size() == logits.size());
  CHECK(again_dispatched == dispatched);
  CHECK(std::memcmp(logits.data(), again.data(), logits.size() * 4) == 0);

  // What this case does NOT establish, said here rather than implied: it
  // asserts the TOPOLOGY (one GEMM per merged group) and the forward's health.
  // The merged GEMM's VALUES are gated at the GEMM boundary by G1, G2 and G4,
  // byte-for-byte against the split arm. The row-strided q/k/v views the merged
  // QKV hands downstream are the same bf16 [T, N] views the NVFP4 merged arm
  // already produces at this site, so nothing new consumes them.
}

// G6 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: the merged resident is fp8 bytes plus one scale grid, built once") {
  // A silent dequant to bf16 is numerically BETTER than the quantized path and
  // therefore invisible to every value comparison above; only byte counts see
  // it. And a merge that ALSO built the per-shard residents would double the
  // device bytes while every value stayed right.
  const int64_t K = 128, qn = 256, kn = 128, vn = 128;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};

  const vllm::Fp8BlockWeight wq = MakeFp8Block(qn, K, kBlock, kBlock, 8001);
  const vllm::Fp8BlockWeight wk = MakeFp8Block(kn, K, kBlock, kBlock, 8002);
  const vllm::Fp8BlockWeight wv = MakeFp8Block(vn, K, kBlock, kBlock, 8003);
  const Fp8BlockShard shards[3] = {
      {&wq, "q_proj"}, {&wk, "k_proj"}, {&wv, "v_proj"}};
  Fp8BlockMergedResident resident;
  const Fp8BlockMergedView view = vllm::dense_fp8_block::ResidentFp8BlockMerged(
      d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);

  const int64_t N = qn + kn + vn;
  CHECK(view.packed.dtype == DType::kI8);
  CHECK(view.packed.shape[0] == N);
  CHECK(view.packed.shape[1] == K);
  CHECK_MESSAGE(wq.packed.bytes.size() + wk.packed.bytes.size() +
                        wv.packed.bytes.size() ==
                    static_cast<size_t>(N * K),
                "one fp8 byte per element; a bf16 expansion would be "
                    << (2 * N * K));
  CHECK(view.scale.dtype == DType::kF32);
  CHECK(view.scale.shape[0] == CDiv(N, kBlock));
  CHECK(view.scale.shape[1] == CDiv(K, kBlock));

  // The shards' OWN residents were never built, so the merge is not additive.
  CHECK(wq.d_packed == nullptr);
  CHECK(wk.d_packed == nullptr);
  CHECK(wv.d_packed == nullptr);
  CHECK(wq.d_scale == nullptr);

  // Built once: a second call reuses the same device buffers.
  const void* first_packed = view.packed.data;
  const void* first_scale = view.scale.data;
  const Fp8BlockMergedView again =
      vllm::dense_fp8_block::ResidentFp8BlockMerged(
          d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);
  CHECK(again.packed.data == first_packed);
  CHECK(again.scale.data == first_scale);
}

// G7 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: a perturbed scale row moves ONLY its own shard's rows") {
  // The concatenation's row mapping, asserted directly. A merged grid that read
  // the wrong shard's rows, or that indexed with a floor where a cdiv belongs,
  // moves output columns that belong to another projection.
  const int64_t M = 2, K = 128, qn = 256, kn = 128, vn = 128;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};
  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 9004);
  DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());
  const int64_t N = qn + kn + vn;

  const auto run = [&](float bump) {
    vllm::Fp8BlockWeight wq = MakeFp8Block(qn, K, kBlock, kBlock, 9001);
    vllm::Fp8BlockWeight wk = MakeFp8Block(kn, K, kBlock, kBlock, 9002);
    vllm::Fp8BlockWeight wv = MakeFp8Block(vn, K, kBlock, kBlock, 9003);
    VaryScale(wq, 9001);
    VaryScale(wk, 9002);
    VaryScale(wv, 9003);
    // `k_proj` owns exactly one merged scale row — row 2 of the 4-row grid.
    reinterpret_cast<float*>(wk.scale.bytes.data())[0] *= bump;
    const Fp8BlockShard shards[3] = {
        {&wq, "q_proj"}, {&wk, "k_proj"}, {&wv, "v_proj"}};
    Fp8BlockMergedResident resident;
    const Fp8BlockMergedView view =
        vllm::dense_fp8_block::ResidentFp8BlockMerged(
            d, vt::kFp8BlockQkv, "qkv_proj", shards, 3, resident);
    DBuf merged = vllm::dense_fp8_block::MatmulFp8BlockMergedD<DBuf>(
        d, x.t(), view, DType::kBF16, vt::kFp8BlockQkv);
    return ToF32(merged.t(), M * N);
  };

  const std::vector<float> base = run(1.0F);
  const std::vector<float> bumped = run(1.10F);
  REQUIRE(base.size() == bumped.size());

  int64_t moved_in_k = 0;
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(m * N + n);
      const bool is_k = n >= qn && n < qn + kn;
      if (is_k) {
        if (base[i] != bumped[i]) ++moved_in_k;
      } else {
        CHECK_MESSAGE(base[i] == bumped[i],
                      "column " << n << " is not k_proj's and must not move");
      }
    }
  }
  CHECK_MESSAGE(moved_in_k > 0,
                "a x1.10 bump on k_proj's only scale row moved nothing");
}

// G8 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: the gate_up factory selects the block method by weight presence") {
  OwnedTensor bf16 = MakeOwned(DType::kBF16, {256, 128}, 3, /*nk=*/true);
  const vllm::Fp8BlockWeight empty;
  vllm::Fp8BlockWeight gate = MakeFp8Block(128, 128, kBlock, kBlock, 3101);
  vllm::Fp8BlockWeight up = MakeFp8Block(128, 128, kBlock, kBlock, 3102);
  VaryScale(gate, 3101);
  VaryScale(up, 3102);
  RequireScalesDiffer(gate, up);
  Fp8BlockMergedResident resident;

  auto m_bf16 = layers::MakeMlpGateUpMethod(bf16, empty, empty, resident, 128);
  CHECK(std::string(m_bf16->Name()) == "bf16-gate-up");

  auto m_block = layers::MakeMlpGateUpMethod(bf16, gate, up, resident, 128);
  CHECK(std::string(m_block->Name()) == "fp8-w8a8-block-gate-up");
  // The NAME alone would pass for a method that returned it from the wrong
  // class, so the concrete type is pinned too.
  CHECK(dynamic_cast<const layers::Fp8BlockMlpGateUpMethod*>(m_block.get()) !=
        nullptr);

  // The overloads coexist and are chosen by the weight TYPE: the NVFP4 factory
  // is in scope here and an `Nvfp4Weight` argument still reaches it.
  const vllm::Nvfp4Weight fp4_empty;
  auto m_fp4 =
      layers::MakeMlpGateUpMethod(bf16, fp4_empty, fp4_empty, 128);
  CHECK(std::string(m_fp4->Name()) == "bf16-gate-up");

  // And the seam runs: the method's own Apply is the merged body, byte for
  // byte, so a caller that binds the seam gets what the model forward gets.
  const int64_t M = 2, K = 128, I = 128;
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Dev d{b, q};
  OwnedTensor xw = MakeOwned(DType::kBF16, {M, K}, 3103);
  DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());
  DBuf via_seam = m_block->Apply(d, x.t());
  REQUIRE(via_seam.t().dtype == DType::kBF16);
  REQUIRE(via_seam.t().shape[1] == I);

  const Fp8BlockShard shards[2] = {{&gate, "gate_proj"}, {&up, "up_proj"}};
  Fp8BlockMergedResident direct;
  const Fp8BlockMergedView view = vllm::dense_fp8_block::ResidentFp8BlockMerged(
      d, vt::kFp8BlockGateUpSwiGLU, "gate_up_proj", shards, 2, direct);
  DBuf via_body = vllm::dense_fp8_block::Fp8BlockGateUpSwiGLUD<DBuf>(
      d, x.t(), view, DType::kBF16);
  CHECK(Bytes(via_seam.t(), M * I) == Bytes(via_body.t(), M * I));
  CHECK(CountNonZero(ToF32(via_seam.t(), M * I)) == M * I);
}

// G9 -----------------------------------------------------------------------
TEST_CASE("fp8 block merged: the bf16 arm is unchanged and dispatches no block GEMM") {
  // Without this the suite passes for a forward that routed EVERY projection
  // through the block arm.
  const HfConfig c = BlockConfig();
  const Qwen3_5DenseWeights w = BlockWeights(c, /*block_arm=*/false);
  uint64_t dispatched = 0;
  const std::vector<float> logits = RegistryForward(c, w, &dispatched);
  CHECK(dispatched == 0);
  CHECK(AllFinite(logits));
  CHECK(CountNonZero(logits) == static_cast<int64_t>(logits.size()));
}
