// vllm.cpp original. Typed shared BF16 MoE contracts (#3094).
#include <doctest/doctest.h>
#include <array>
#include <type_traits>
#include "rocm_moe_test_helpers.h"
#include "vt/op_provider.h"

// Adding native modes must not change any existing caller's function signature.
static_assert(std::is_same_v<vt::MoeGroupedGemmBf16Fn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, const vt::Tensor&)>);
static_assert(std::is_same_v<vt::MoeGroupedGemmBf16GateUpSiluFn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, const vt::Tensor&, const vt::Tensor&)>);
static_assert(std::is_same_v<vt::MoeCombineFn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, float)>);

TEST_CASE("shared native BF16 MoE validates routes and physical tensor formats") {
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::array<uint16_t, 16> values{};
  std::array<float, 4> weights{};
  std::array<int32_t, 4> ids{};
  std::array<int64_t, 4> ptrs{};
  auto a = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto out = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto expert = rocm_moe_test::View(ids.data(), vt::DType::kI32, cpu, {2});
  auto pointers = rocm_moe_test::View(ptrs.data(), vt::DType::kI64, cpu, {4});
  auto route = rocm_moe_test::View(weights.data(), vt::DType::kF32, cpu, {1});
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr, pointers, route),
      doctest::Contains("route_weights must be contiguous f32 [P]"));
  route.shape[0] = 2;
  route.dtype = vt::DType::kBF16;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr, pointers, route),
      doctest::Contains("route_weights must be contiguous f32 [P]"));
  a.dtype = vt::DType::kF32;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, nullptr,
                                                        pointers, pointers),
                    doctest::Contains("act must be bf16"));
  a.dtype = vt::DType::kBF16;
  out.dtype = vt::DType::kF32;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, nullptr,
                                                        pointers, pointers),
                    doctest::Contains("out must be bf16"));
  auto rows = expert;
  rows.stride[0] = 2;
  out.dtype = vt::DType::kBF16;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, &rows,
                                                        pointers, pointers),
                    doctest::Contains("row_map must be contiguous i32 [P]"));
  auto combined = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {1, 4});
  auto weighted = rocm_moe_test::View(values.data(), vt::DType::kF32, cpu, {1, 2, 4});
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted),
                    doctest::Contains("expert_out must be bf16"));
  weighted.dtype = vt::DType::kBF16;
  weighted.shape[1] = 4;
  weighted.shape[2] = 2;
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted),
                    doctest::Contains("must match out"));
  weighted = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {1, 2, 4});
  auto shared = combined;
  shared.dtype = vt::DType::kF16;
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted, &shared),
                    doctest::Contains("shared must be f32/bf16"));
  CHECK_FALSE(vt::MoeGroupedBf16NativeAvailable(vt::DeviceType::kCPU));
}

TEST_CASE("shared native BF16 MoE requires all five providers in every subset") {
  // XPU has no backend registrations. Unique names let each subset use the
  // real provider seam without changing any production device's registrations.
  constexpr vt::DeviceType device = vt::DeviceType::kXPU;
  constexpr std::array operations = {
      vt::OpId::kMoeGroupedGemmBf16, vt::OpId::kMoeGroupedGemmBf16GateUpSilu,
      vt::OpId::kMoeGroupedGemmBf16GateUpSiluNative,
      vt::OpId::kMoeGroupedGemmBf16Weighted, vt::OpId::kMoeCombinePreweighted};
  static constexpr std::array names = {
      "moe-contract-grouped", "moe-contract-gate-up", "moe-contract-native",
      "moe-contract-weighted", "moe-contract-combine"};
  struct DisableOnExit {
    ~DisableOnExit() {
      for (const char* name : names) vt::DisableOpProvider(name, true);
    }
  } cleanup;
  for (size_t i = 0; i < operations.size(); ++i) {
    REQUIRE_FALSE(vt::OpRegistered(operations[i], device));
    // The stub is never dispatched. Only native availability is inspected.
    vt::RegisterOpProvider(operations[i], device,
        {names[i], 0, nullptr, reinterpret_cast<void*>(+[] {})});
  }
  // Mask 3 is the legacy-only set. Masks 31^(1<<i) omit each provider in turn.
  for (unsigned mask = 0; mask < 32; ++mask) {
    CAPTURE(mask);
    for (size_t i = 0; i < operations.size(); ++i) {
      const bool present = (mask & (1u << i)) != 0;
      vt::DisableOpProvider(names[i], !present);
      REQUIRE(vt::OpRegistered(operations[i], device) == present);
    }
    CHECK(vt::MoeGroupedBf16NativeAvailable(device) == (mask == 31));
  }
}

TEST_CASE("shared native BF16 MoE rejects weighted route metadata before dispatch") {
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::array<uint16_t, 16> values{};
  std::array<float, 2> weights{};
  std::array<int32_t, 2> ids{};
  std::array<int64_t, 4> ptrs{};
  auto a = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto out = a;
  auto expert = rocm_moe_test::View(ids.data(), vt::DType::kI32, cpu, {2});
  auto pointers = rocm_moe_test::View(ptrs.data(), vt::DType::kI64, cpu, {4});
  auto route = rocm_moe_test::View(weights.data(), vt::DType::kF32, cpu, {2});
  SUBCASE("route rank with the correct element count") {
    route = rocm_moe_test::View(weights.data(), vt::DType::kF32, cpu, {1, 2});
  }
  SUBCASE("route stride") { route.stride[0] = 2; }
  SUBCASE("route device") { route.device.index = 1; }
  SUBCASE("weighted down retains the common grouped validator") {
    a.dtype = vt::DType::kF32;
    CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr,
                                                    pointers, route),
                      doctest::Contains("act must be bf16"));
    return;
  }
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr,
                                                  pointers, route),
                    doctest::Contains("route_weights must be contiguous f32 [P]"));
}

TEST_CASE("shared native BF16 MoE rejects combine metadata before dispatch") {
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::array<uint16_t, 32> values{};
  auto out = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto weighted = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 2, 4});
  SUBCASE("expert output stride") { weighted.stride[0] += 1; }
  SUBCASE("combined output stride") { out.stride[0] += 1; }
  SUBCASE("expert output device") { weighted.device.index = 1; }
  SUBCASE("combined output device") { out.device.index = 1; }
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, out, weighted),
                    doctest::Contains("contiguous tensors on the queue device required"));
}

TEST_CASE("shared native BF16 MoE rejects shared metadata before dispatch") {
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::array<uint16_t, 32> values{};
  auto out = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto weighted = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 2, 4});
  auto shared = out;
  SUBCASE("shared rank with matching leading dimensions") {
    shared = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4, 1});
  }
  SUBCASE("shared token count") {
    shared = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {3, 4});
  }
  SUBCASE("shared hidden width") {
    shared = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 5});
  }
  SUBCASE("shared stride") { shared.stride[0] += 1; }
  SUBCASE("shared device") { shared.device.index = 1; }
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, out, weighted, &shared),
                    doctest::Contains("shared must be f32/bf16 [T,H] on the queue device"));
}
