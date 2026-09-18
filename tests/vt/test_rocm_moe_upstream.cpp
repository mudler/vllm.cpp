// Port of vLLM tests/kernels/moe/test_moe.py:test_fused_moe at
// e126687a9a828d513c01a07cd69f025f27d63280 (#3094).
// rocm_moe_upstream.py preserves the upstream fixtures, reference, modes,
// seed 7, atol 0.02, and rtol 0. This half consumes those exact exported bytes.
#include "rocm_moe_test_helpers.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
using namespace rocm_moe_test;
using vt::DType;
template <typename T>
std::vector<T> Read(const std::filesystem::path& path, int64_t count) {
  REQUIRE(std::filesystem::file_size(path) == static_cast<uint64_t>(count) * sizeof(T));
  std::vector<T> result(static_cast<size_t>(count));
  std::ifstream source(path, std::ios::binary);
  source.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size() * sizeof(T)));
  REQUIRE(source.good());
  return result;
}
float Compare(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected) {
  REQUIRE(actual.size() == expected.size());
  size_t bad = 0, first = 0;
  float maximum = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float delta = std::abs(vt::BF16ToF32(actual[i]) - vt::BF16ToF32(expected[i]));
    if (!(delta <= 0.02f)) {
      if (bad == 0) first = i;
      ++bad;
    }
    maximum = std::max(maximum, delta);
  }
  CAPTURE(bad);
  CAPTURE(first);
  CAPTURE(maximum);
  CHECK(bad == 0);
  return maximum;
}
}  // namespace

TEST_CASE("ROCm native BF16 MoE matches every exported upstream case") {
  const char* directory = std::getenv("VT_ROCM_MOE_COMPONENT_CASE");
  if (directory == nullptr) {
    MESSAGE("Set VT_ROCM_MOE_COMPONENT_CASE to a pinned upstream export");
    std::exit(77);
  }
  RequireDevice();
  const std::filesystem::path path(directory);
  nlohmann::json record;
  std::ifstream(path / "case.json") >> record;
  REQUIRE(record["pin"] == "e126687a9a828d513c01a07cd69f025f27d63280");
  REQUIRE(record["seed"] == 7);
  REQUIRE(record["renormalize"] == false);
  REQUIRE(record["atol"] == 0.02);
  REQUIRE(record["rtol"] == 0);
  const int64_t m = record["M"], n = record["N"], k = record["K"];
  const int64_t experts = record["experts"], top_k = record["top_k"], pairs = m * top_k;
  const int64_t pad = record["padding"].get<bool>() ? 128 : 0;
  for (const auto& call : record["upstream_calls"]) {
    REQUIRE(call["w1_stride"] == std::vector<int64_t>{2 * n * (k + pad), k + pad, 1});
    REQUIRE(call["w2_stride"] == std::vector<int64_t>{k * (n + pad), n + pad, 1});
  }
  // The loader's shared Matmul-B contract materializes [K,N] matrices. Padded
  // cases retain and compare the actual padded oracle, not a native stride mode.
  REQUIRE(record["files"]["gate.bf16"]["export_stride"] == std::vector<int64_t>{k * n, n, 1});
  REQUIRE(record["files"]["up.bf16"]["export_stride"] == std::vector<int64_t>{k * n, n, 1});
  REQUIRE(record["files"]["down.bf16"]["export_stride"] == std::vector<int64_t>{n * k, k, 1});
  auto a = Read<uint16_t>(path / "a.bf16", m * k);
  auto gate = Read<uint16_t>(path / "gate.bf16", experts * k * n);
  auto up = Read<uint16_t>(path / "up.bf16", experts * k * n);
  auto down = Read<uint16_t>(path / "down.bf16", experts * n * k);
  auto ids = Read<int32_t>(path / "ids.i32", pairs);
  auto routes = Read<float>(path / "routes.f32", pairs);
  const auto reference = Read<uint16_t>(path / "reference.bf16", m * k);
  const auto triton = Read<uint16_t>(path / "triton.bf16", m * k);
  std::vector<int32_t> map(static_cast<size_t>(pairs));
  for (int64_t pair = 0; pair < pairs; ++pair)
    map[static_cast<size_t>(pair)] = static_cast<int32_t>(pair / top_k);
  Queue queue;
  auto& q = queue.value;
  Buffer da(q, DType::kBF16, {m, k}, a.data());
  Buffer dg(q, DType::kBF16, {experts, k, n}, gate.data());
  Buffer du(q, DType::kBF16, {experts, k, n}, up.data());
  Buffer dd(q, DType::kBF16, {experts, n, k}, down.data());
  Buffer di(q, DType::kI32, {pairs}, ids.data()), dm(q, DType::kI32, {pairs}, map.data());
  Buffer dr(q, DType::kF32, {pairs}, routes.data());
  std::vector<int64_t> gp, uptr, dp;
  for (int64_t expert = 0; expert < experts; ++expert) {
    gp.push_back(reinterpret_cast<int64_t>(dg.tensor().Ptr<uint16_t>() + expert * k * n));
    uptr.push_back(reinterpret_cast<int64_t>(du.tensor().Ptr<uint16_t>() + expert * k * n));
    dp.push_back(reinterpret_cast<int64_t>(dd.tensor().Ptr<uint16_t>() + expert * n * k));
  }
  Buffer dgp(q, DType::kI64, {experts}, gp.data()), dup(q, DType::kI64, {experts}, uptr.data());
  Buffer ddp(q, DType::kI64, {experts}, dp.data());
  Buffer activated(q, DType::kBF16, {pairs, n}), weighted(q, DType::kBF16, {pairs, k});
  Buffer result(q, DType::kBF16, {m, k});
  const auto weighted_view = View(weighted.tensor().data, DType::kBF16, q.device, {m, top_k, k});
  auto run = [&] {
    vt::MoeGroupedGemmBf16GateUpSiluNative(q, activated.tensor(), da.tensor(), di.tensor(),
                                          &dm.tensor(), dgp.tensor(), dup.tensor());
    vt::MoeGroupedGemmBf16Weighted(q, weighted.tensor(), activated.tensor(), di.tensor(),
                                   nullptr, ddp.tensor(), dr.tensor());
    vt::MoeCombinePreweighted(q, result.tensor(), weighted_view);
  };
  run();
  const auto first = result.Download<uint16_t>(q);
  const float reference_error = Compare(first, reference);
  const float triton_error = Compare(first, triton);
  REQUIRE(activated.tensor().dtype == DType::kBF16);
  REQUIRE(weighted.tensor().dtype == DType::kBF16);
  if (record["use_cudagraph"].get<bool>()) {
    auto& backend = vt::GetBackend(q.device);
    REQUIRE(backend.SupportsGraphCapture());
    backend.BeginCapture(q);
    run();
    void* graph = backend.EndCaptureGraph(q);
    REQUIRE(graph != nullptr);
    for (int repeat = 0; repeat < 3; ++repeat) {
      backend.Memset(q, result.tensor().data, 0, static_cast<size_t>(m * k) * 2);
      backend.ReplayGraph(q, graph);
      CHECK(result.Download<uint16_t>(q) == first);
    }
    backend.DestroyGraph(graph);
  }
  activated.CheckGuard(q);
  weighted.CheckGuard(q);
  result.CheckGuard(q);
  record["native_max_abs_vs_reference"] = reference_error;
  record["native_max_abs_vs_triton"] = triton_error;
  std::ofstream(path / "native-result.json") << record.dump(2) << '\n';
}
