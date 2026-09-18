// vllm.cpp original. Native BF16 boundaries and lifetime gates (#3094).
// Numeric boundaries mirror vLLM e126687a9a, fused_moe.py:593-610,
// triton_moe.py:388-527, activation_kernels.cu:44,165-177.
#include "rocm_moe_test_helpers.h"
#include <array>
#include <cstring>
#include <future>
#include <memory>
#include <random>
#include <thread>
#include "vt/op_provider.h"

namespace {
using namespace rocm_moe_test;
using vt::DType;

// Exact BF16 witnesses, found with seed 7 then frozen. Each distinguishes one
// required narrowing point independently of dot-order and token tolerances.
struct Boundary {
  std::array<uint16_t, 3> x, gate, up;
  uint16_t expected;
};
constexpr std::array<Boundary, 3> kBoundaries = {{
    {{{0xbffa, 0x401d, 0xbe9d}}, {{0x3e95, 0x4028, 0x3e10}},
      {{0x3fdf, 0xc00d, 0xbf31}}, 0xc247},
    {{{0x3e91, 0x3f72, 0x4022}}, {{0xbf62, 0xbfa7, 0xbe7b}},
      {{0xbf92, 0xbf47, 0xbecd}}, 0x3ef3},
    {{{0xbf93, 0x3eb0, 0x4033}}, {{0x3f65, 0xbfcb, 0xc009}},
      {{0x3f0e, 0x400e, 0x402b}}, 0xbcf5},
}};

void CheckBoundary(vt::Queue& q, const Boundary& fixture) {
  Buffer x(q, DType::kBF16, {1, 3}, fixture.x.data());
  Buffer gate(q, DType::kBF16, {3, 1}, fixture.gate.data());
  Buffer up(q, DType::kBF16, {3, 1}, fixture.up.data());
  const int64_t gate_ptr = reinterpret_cast<int64_t>(gate.tensor().data);
  const int64_t up_ptr = reinterpret_cast<int64_t>(up.tensor().data);
  const int32_t expert = 0;
  Buffer gp(q, DType::kI64, {1}, &gate_ptr), uptr(q, DType::kI64, {1}, &up_ptr);
  Buffer ids(q, DType::kI32, {1}, &expert), output(q, DType::kBF16, {1, 1});
  vt::MoeGroupedGemmBf16GateUpSiluNative(q, output.tensor(), x.tensor(), ids.tensor(),
                                       nullptr, gp.tensor(), uptr.tensor());
  CHECK(output.Download<uint16_t>(q) == std::vector<uint16_t>{fixture.expected});
  output.CheckGuard(q);
}

std::vector<uint16_t> Random(int64_t count, uint32_t seed) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
  std::vector<uint16_t> result(static_cast<size_t>(count));
  for (auto& value : result) value = vt::F32ToBF16(distribution(engine));
  return result;
}

// Explicit BF16 gate/up storage is the local native decomposition. The separate
// pinned engine fixture and upstream component gate supply the external oracle.
struct NativeCase {
  vt::Queue& q;
  int64_t p, n;
  std::unique_ptr<Buffer> x, ids, map, gp, up, output, reference;
  std::vector<std::unique_ptr<Buffer>> experts;
  NativeCase(vt::Queue& queue, int64_t pairs, int64_t columns, int64_t inner,
              bool row_map, uint32_t seed)
      : q(queue), p(pairs), n(columns) {
    HostDeviceScope setup_device(q.device.index);
    const int64_t rows = row_map ? (pairs + 2) / 3 : pairs;
    const auto activation = Random(rows * inner, seed);
    x = std::make_unique<Buffer>(q, DType::kBF16, std::vector<int64_t>{rows, inner}, activation.data());
    std::vector<int32_t> selections(static_cast<size_t>(pairs)), mapping(static_cast<size_t>(pairs));
    for (int64_t pair = 0; pair < pairs; ++pair) {
      // Four experts, expert 2 empty. The repeated, unordered choices and
      // repeated/reversed rows forbid contiguous-expert or token-order shortcuts.
      selections[static_cast<size_t>(pair)] = std::array<int32_t, 5>{3, 0, 3, 1, 0}[pair % 5];
      mapping[static_cast<size_t>(pair)] = static_cast<int32_t>(rows - 1 - (pair / 3));
    }
    ids = std::make_unique<Buffer>(q, DType::kI32, std::vector<int64_t>{pairs}, selections.data());
    if (row_map)
      map = std::make_unique<Buffer>(q, DType::kI32, std::vector<int64_t>{pairs}, mapping.data());
    std::vector<int64_t> gate_ptrs, up_ptrs;
    for (int expert = 0; expert < 4; ++expert) {
      for (int tower = 0; tower < 2; ++tower) {
        const auto weight = Random(inner * columns, seed + 1 + expert * 2 + tower);
        experts.push_back(std::make_unique<Buffer>(q, DType::kBF16,
                            std::vector<int64_t>{inner, columns}, weight.data()));
        (tower == 0 ? gate_ptrs : up_ptrs).push_back(
            reinterpret_cast<int64_t>(experts.back()->tensor().data));
      }
    }
    gp = std::make_unique<Buffer>(q, DType::kI64, std::vector<int64_t>{4}, gate_ptrs.data());
    up = std::make_unique<Buffer>(q, DType::kI64, std::vector<int64_t>{4}, up_ptrs.data());
    output = std::make_unique<Buffer>(q, DType::kBF16, std::vector<int64_t>{pairs, columns});
    reference = std::make_unique<Buffer>(q, DType::kBF16, std::vector<int64_t>{pairs, columns});
    Buffer gate(q, DType::kBF16, {pairs, columns}), upper(q, DType::kBF16, {pairs, columns});
    vt::MoeGroupedGemmBf16(q, gate.tensor(), x->tensor(), ids->tensor(),
                          map == nullptr ? nullptr : &map->tensor(), gp->tensor());
    vt::MoeGroupedGemmBf16(q, upper.tensor(), x->tensor(), ids->tensor(),
                          map == nullptr ? nullptr : &map->tensor(), up->tensor());
    vt::MoeSiluMul(q, reference->tensor(), gate.tensor(), upper.tensor());
    vt::GetBackend(q.device).Synchronize(q);
  }
  void Launch() {
    vt::MoeGroupedGemmBf16GateUpSiluNative(q, output->tensor(), x->tensor(), ids->tensor(),
                                         map == nullptr ? nullptr : &map->tensor(),
                                         gp->tensor(), up->tensor());
  }
  void Check() {
    CHECK(output->tensor().dtype == DType::kBF16);
    CHECK(output->Download<uint16_t>(q) == reference->Download<uint16_t>(q));
    output->CheckGuard(q);
  }
};
}  // namespace

TEST_CASE("ROCm native MoE preserves gate, up, and SiLU BF16 boundaries") {
  RequireDevice();
  Queue queue;
  for (const auto& fixture : kBoundaries) CheckBoundary(queue.value, fixture);
}

TEST_CASE("ROCm native MoE weights before down narrowing and combines preweighted values") {
  RequireDevice();
  Queue queue;
  auto& q = queue.value;
  const std::array<uint16_t, 3> x = {0x3df4, 0x3fe4, 0xc016};
  const std::array<uint16_t, 3> weight = {0x3fd3, 0xbe7a, 0xbe72};
  const float route = 0.761520803f;
  const int32_t expert = 0;
  Buffer a(q, DType::kBF16, {1, 3}, x.data()), w(q, DType::kBF16, {3, 1}, weight.data());
  const int64_t ptr = reinterpret_cast<int64_t>(w.tensor().data);
  Buffer wp(q, DType::kI64, {1}, &ptr), ids(q, DType::kI32, {1}, &expert);
  Buffer routes(q, DType::kF32, {1}, &route), result(q, DType::kBF16, {1, 1});
  vt::MoeGroupedGemmBf16Weighted(q, result.tensor(), a.tensor(), ids.tensor(),
                                nullptr, wp.tensor(), routes.tensor());
  CHECK(result.Download<uint16_t>(q) == std::vector<uint16_t>{0x3e76});
  result.CheckGuard(q);
  const std::array<uint16_t, 6> weighted = {0x3e76, 0xbf40, 0x3e00, 0x3f00, 0x3f80, 0xbf00};
  Buffer values(q, DType::kBF16, {2, 3, 1}, weighted.data());
  Buffer combined(q, DType::kBF16, {2, 1});
  vt::MoeCombinePreweighted(q, combined.tensor(), values.tensor());
  CHECK(combined.Download<uint16_t>(q) == std::vector<uint16_t>{0xbec5, 0x3f80});
  combined.CheckGuard(q);
  const std::array<float, 2> shared = {0.25f, -0.125f};
  Buffer share(q, DType::kF32, {2, 1}, shared.data()), fp32(q, DType::kF32, {2, 1});
  vt::MoeCombinePreweighted(q, fp32.tensor(), values.tensor(), &share.tensor(), 0.5f);
  const auto sum = fp32.Download<float>(q);
  CHECK(sum[0] == 0.0576171875f);
  CHECK(sum[1] == 0.375f);
  fp32.CheckGuard(q);
}

TEST_CASE("ROCm native MoE preserves pair order, empty experts, maps, tails, and repeated launches") {
  RequireDevice();
  Queue queue;
  for (const auto& shape : std::array<std::array<int64_t, 3>, 5>{{
           {6, 8, 1024}, {40, 130, 80}, {1024, 70, 96}, {128, 256, 264}, {65539, 3, 5}}}) {
    for (bool map : {false, true}) {
      NativeCase fixture(queue.value, shape[0], shape[1], shape[2], map, 7);
      for (int repeat = 0; repeat < 3; ++repeat) {
        fixture.Launch();
        fixture.Check();
      }
    }
  }
}

TEST_CASE("ROCm native MoE capture replays after larger shapes on independent streams") {
  RequireDevice();
  Queue first, second;
  NativeCase small(first.value, 6, 17, 35, true, 7);
  NativeCase other(second.value, 15, 33, 65, false, 41);
  small.Launch();
  other.Launch();
  small.Check();
  other.Check();
  auto& backend = vt::GetBackend(Device());
  REQUIRE(backend.SupportsGraphCapture());
  backend.BeginCapture(first.value);
  small.Launch();
  void* graph = backend.EndCaptureGraph(first.value);
  REQUIRE(graph != nullptr);
  NativeCase large(first.value, 1024, 129, 257, true, 91);
  large.Launch();
  large.Check();
  for (int repeat = 0; repeat < 3; ++repeat) {
    backend.Memset(first.value, small.output->tensor().data, 0,
                    static_cast<size_t>(small.p * small.n) * 2);
    backend.ReplayGraph(first.value, graph);
    other.Launch();
    small.Check();
    other.Check();
  }
  backend.DestroyGraph(graph);
}

TEST_CASE("ROCm native MoE concurrent stream launches keep independent results") {
  RequireDevice();
  Queue first, second;
  NativeCase a(first.value, 97, 130, 257, true, 7);
  NativeCase b(second.value, 513, 65, 129, false, 99);
  std::promise<void> release;
  const auto ready = release.get_future().share();
  auto run = [&](NativeCase& fixture) {
    ready.wait();
    for (int repeat = 0; repeat < 3; ++repeat) fixture.Launch();
    return fixture.output->Download<uint16_t>(fixture.q);
  };
  auto left = std::async(std::launch::async, [&] { return run(a); });
  auto right = std::async(std::launch::async, [&] { return run(b); });
  release.set_value();
  CHECK(left.get() == a.reference->Download<uint16_t>(first.value));
  CHECK(right.get() == b.reference->Download<uint16_t>(second.value));
  a.output->CheckGuard(first.value);
  b.output->CheckGuard(second.value);
}

TEST_CASE("ROCm native MoE keeps null default streams on separate devices") {
  if (std::getenv("VT_ROCM_MOE_TWO_DEVICES") == nullptr) {
    MESSAGE("PENDING: set VT_ROCM_MOE_TWO_DEVICES with two visible devices for this separate gate");
    return;
  }
  RequireDevice(0);
  RequireDevice(1);
  vt::Queue q0{Device(0), nullptr}, q1{Device(1), nullptr};
  NativeCase first(q0, 12, 31, 35, true, 7);
  NativeCase second(q1, 17, 65, 129, false, 99);
  for (int repeat = 0; repeat < 3; ++repeat) {
    {
      HostDeviceScope opposite(1);
      first.Launch();
      CHECK(HostDeviceScope::Current() == 1);
      first.Check();
    }
    {
      HostDeviceScope opposite(0);
      second.Launch();
      CHECK(HostDeviceScope::Current() == 0);
      second.Check();
    }
  }
}

TEST_CASE("ROCm native MoE zero pairs and output width do not launch") {
  RequireDevice();
  Queue queue;
  auto& q = queue.value;
  const int64_t pointer = 0;
  for (auto shape : {std::array<int64_t, 2>{0, 7}, std::array<int64_t, 2>{3, 0}}) {
    const int64_t p = shape[0], n = shape[1];
    Buffer a(q, DType::kBF16, {p, 3}), out(q, DType::kBF16, {p, n});
    Buffer ids(q, DType::kI32, {p}), ptr(q, DType::kI64, {1}, &pointer);
    Buffer weights(q, DType::kF32, {p});
    vt::MoeGroupedGemmBf16(q, out.tensor(), a.tensor(), ids.tensor(), nullptr, ptr.tensor());
    vt::MoeGroupedGemmBf16GateUpSilu(q, out.tensor(), a.tensor(), ids.tensor(), nullptr,
                                    ptr.tensor(), ptr.tensor());
    vt::MoeGroupedGemmBf16GateUpSiluNative(q, out.tensor(), a.tensor(), ids.tensor(), nullptr,
                                          ptr.tensor(), ptr.tensor());
    vt::MoeGroupedGemmBf16Weighted(q, out.tensor(), a.tensor(), ids.tensor(), nullptr,
                                   ptr.tensor(), weights.tensor());
    auto expert_out = View(out.tensor().data, DType::kBF16, q.device, {p, 1, n});
    vt::MoeCombinePreweighted(q, out.tensor(), expert_out);
    out.CheckGuard(q);
  }
}

TEST_CASE("ROCm weighted down and preweighted combine isolate null streams by device") {
  if (std::getenv("VT_ROCM_MOE_TWO_DEVICES") == nullptr) {
    MESSAGE("PENDING: set VT_ROCM_MOE_TWO_DEVICES with two visible devices for this separate gate");
    return;
  }
  RequireDevice(0);
  RequireDevice(1);
  for (int device : {0, 1}) {
    vt::Queue q{Device(device), nullptr};
    const std::array<uint16_t, 3> x = {0x3df4, 0x3fe4, 0xc016};
    const std::array<uint16_t, 3> w = {0x3fd3, 0xbe7a, 0xbe72};
    const float route = 0.761520803f;
    const int32_t expert = 0;
    Buffer activation(q, DType::kBF16, {1, 3}, x.data());
    Buffer weight(q, DType::kBF16, {3, 1}, w.data());
    const int64_t pointer = reinterpret_cast<int64_t>(weight.tensor().data);
    Buffer ptr(q, DType::kI64, {1}, &pointer), ids(q, DType::kI32, {1}, &expert);
    Buffer routes(q, DType::kF32, {1}, &route), down(q, DType::kBF16, {1, 1});
    const std::array<uint16_t, 6> values = {0x3e76, 0xbf40, 0x3e00, 0x3f00, 0x3f80, 0xbf00};
    Buffer weighted(q, DType::kBF16, {2, 3, 1}, values.data());
    Buffer combined(q, DType::kBF16, {2, 1});
    for (int repeat = 0; repeat < 3; ++repeat) {
      HostDeviceScope opposite(1 - device);
      vt::MoeGroupedGemmBf16Weighted(q, down.tensor(), activation.tensor(), ids.tensor(),
                                     nullptr, ptr.tensor(), routes.tensor());
      CHECK(HostDeviceScope::Current() == 1 - device);
      CHECK(down.Download<uint16_t>(q) == std::vector<uint16_t>{0x3e76});
      vt::MoeCombinePreweighted(q, combined.tensor(), weighted.tensor());
      CHECK(HostDeviceScope::Current() == 1 - device);
      CHECK(combined.Download<uint16_t>(q) == std::vector<uint16_t>{0xbec5, 0x3f80});
      down.CheckGuard(q);
      combined.CheckGuard(q);
    }
  }
}

TEST_CASE("ROCm native MoE weighted down preserves F32 output and mapped per-pair routes") {
  RequireDevice();
  Queue queue;
  auto& q = queue.value;
  // Rows: [1,2], [3,-1], [-2,4], [5,-2]. Expert matrices are
  // [[2,1],[-1,3]] and [[1,-2],[4,1]]. The FP32 arithmetic is exact.
  const std::array<uint16_t, 8> x = {
      0x3f80, 0x4000, 0x4040, 0xbf80, 0xc000, 0x4080, 0x40a0, 0xc000};
  const std::array<uint16_t, 4> first = {0x4000, 0x3f80, 0xbf80, 0x4040};
  const std::array<uint16_t, 4> second = {0x3f80, 0xc000, 0x4080, 0x3f80};
  const std::array<int32_t, 4> selections = {1, 0, 0, 1}, mapping = {2, 0, 2, 1};
  const std::array<float, 4> routes = {0.501953125f, 0.25f, 0.125f, 0.75f};
  Buffer a(q, DType::kBF16, {4, 2}, x.data());
  Buffer w0(q, DType::kBF16, {2, 2}, first.data()), w1(q, DType::kBF16, {2, 2}, second.data());
  const std::array<int64_t, 2> pointers = {
      reinterpret_cast<int64_t>(w0.tensor().data), reinterpret_cast<int64_t>(w1.tensor().data)};
  Buffer wp(q, DType::kI64, {2}, pointers.data()), ids(q, DType::kI32, {4}, selections.data());
  Buffer map(q, DType::kI32, {4}, mapping.data()), route(q, DType::kF32, {4}, routes.data());
  // Mapped expert results [14,8], [0,7], [-8,10], [-1,-7], multiplied
  // by each pair's route. Route 0 is 257/512, so its FP32 results must retain
  // bits that BF16 output rounds away. No expected value calls another MoE operation.
  const std::vector<float> expected = {7.02734375f, 4.015625f, 0.0f, 1.75f,
                                       -1.0f, 1.25f, -0.75f, -5.25f};
  const std::vector<uint16_t> expected_bf16 = {
      0x40e1, 0x4080, 0x0000, 0x3fe0, 0xbf80, 0x3fa0, 0xbf40, 0xc0a8};
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    CAPTURE(dtype);
    Buffer output(q, dtype, {4, 2});
    vt::MoeGroupedGemmBf16Weighted(q, output.tensor(), a.tensor(), ids.tensor(),
                                  &map.tensor(), wp.tensor(), route.tensor());
    if (dtype == DType::kF32)
      CHECK(output.Download<float>(q) == expected);
    else
      CHECK(output.Download<uint16_t>(q) == expected_bf16);
    output.CheckGuard(q);
  }
}

TEST_CASE("ROCm native MoE combines BF16 shared inputs into both output types") {
  RequireDevice();
  Queue queue;
  auto& q = queue.value;
  // Expert sums: [0.5,2.25] and [3.5,1]. Scale by 0.5, then add
  // shared rows [129/512,-1/8] and [-1/2,2], giving [257/512,1] and [1.25,2.5].
  // FP32 retains 257/512. BF16 rounds that halfway value to the even neighbor 0.5.
  const std::array<uint16_t, 8> weighted = {
      0x3f80, 0x4000, 0xbf00, 0x3e80, 0x4040, 0xbf80, 0x3f00, 0x4000};
  const std::array<uint16_t, 4> shared = {0x3e81, 0xbe00, 0xbf00, 0x4000};
  Buffer values(q, DType::kBF16, {2, 2, 2}, weighted.data());
  Buffer share(q, DType::kBF16, {2, 2}, shared.data());
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    CAPTURE(dtype);
    Buffer output(q, dtype, {2, 2});
    vt::MoeCombinePreweighted(q, output.tensor(), values.tensor(), &share.tensor(), 0.5f);
    if (dtype == DType::kF32)
      CHECK(output.Download<float>(q) == std::vector<float>{0.501953125f, 1.0f, 1.25f, 2.5f});
    else
      CHECK(output.Download<uint16_t>(q) == std::vector<uint16_t>{0x3f00, 0x3f80, 0x3fa0, 0x4020});
    output.CheckGuard(q);
  }
}
