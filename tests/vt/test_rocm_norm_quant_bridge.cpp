#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>

#include "vt/rocm/rocm_norm_quant_bridge.h"

namespace {

using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::rocm::detail::NormQuantKey;
using vt::rocm::detail::NormQuantCaptureState;
using vt::rocm::detail::NormQuantScratchPool;
using vt::rocm::detail::NormQuantToken;
using vt::rocm::detail::NormQuantTokenRegistry;

NormQuantKey Key() {
  return NormQuantKey{reinterpret_cast<void*>(0x1000),
                      2,
                      512,
                      544,
                      DType::kBF16,
                      Device{DeviceType::kROCM, 1},
                      77,
                      reinterpret_cast<void*>(0x2000),
                      std::this_thread::get_id()};
}

NormQuantToken Token() {
  return NormQuantToken{Key(), reinterpret_cast<void*>(0x3000), 1168, 9};
}

TEST_CASE("ROCm norm-quant tokens bind every producer identity field") {
  using Mutate = void (*)(NormQuantKey&);
  const std::array<Mutate, 8> mutations = {
      [](NormQuantKey& key) { key.activation = reinterpret_cast<void*>(0x1001); },
      [](NormQuantKey& key) { ++key.rows; },
      [](NormQuantKey& key) { key.hidden += 256; },
      [](NormQuantKey& key) { ++key.row_stride; },
      [](NormQuantKey& key) { key.dtype = DType::kF16; },
      [](NormQuantKey& key) { ++key.device.index; },
      [](NormQuantKey& key) { ++key.queue_id; },
      [](NormQuantKey& key) { key.stream = reinterpret_cast<void*>(0x2001); },
  };

  for (Mutate mutate : mutations) {
    NormQuantTokenRegistry registry;
    registry.Record(Token());
    NormQuantKey mismatch = Key();
    mutate(mismatch);
    CHECK_FALSE(registry.Take(mismatch).has_value());
    CHECK_FALSE(registry.Take(Key()).has_value());
  }
}

TEST_CASE("ROCm norm-quant tokens are one-shot and carry scratch lifetime") {
  NormQuantTokenRegistry registry;
  registry.Record(Token());
  const auto taken = registry.Take(Key());
  REQUIRE(taken.has_value());
  CHECK(taken->scratch == reinterpret_cast<void*>(0x3000));
  CHECK(taken->scratch_bytes == 1168);
  CHECK(taken->scratch_generation == 9);
  CHECK_FALSE(registry.Take(Key()).has_value());
}

TEST_CASE("ROCm norm-quant tokens reject a concurrent host thread") {
  NormQuantTokenRegistry registry;
  registry.Record(Token());
  const NormQuantKey key = Key();
  std::optional<NormQuantToken> taken = Token();
  std::thread other([&] {
    NormQuantKey concurrent = key;
    concurrent.host_thread = std::this_thread::get_id();
    taken = registry.Take(concurrent);
  });
  other.join();
  CHECK_FALSE(taken.has_value());
  CHECK_FALSE(registry.Take(key).has_value());
}

TEST_CASE("ROCm norm-quant scratch is queue-owned and capture-safe") {
  NormQuantScratchPool<uint64_t, int> pool;
  NormQuantCaptureState capture = NormQuantCaptureState::kNone;
  int queries = 0;
  int allocations = 0;
  std::array<std::byte, 128> first{};
  std::array<std::byte, 128> second{};
  auto query = [&](int) {
    ++queries;
    return capture;
  };
  auto allocate = [&](size_t, int) -> void* {
    return allocations++ == 0 ? first.data() : second.data();
  };

  CHECK(pool.Ensure(7, 70, 64, query, allocate) == first.data());
  CHECK(queries == 1);
  capture = NormQuantCaptureState::kActive;
  CHECK(pool.Ensure(7, 70, 64, query, allocate) == first.data());
  CHECK(queries == 1);
  CHECK_THROWS_WITH_AS(pool.Ensure(7, 70, 96, query, allocate),
                       "vt rocm: norm-quant scratch: pre-warm RmsNorm on this queue before graph capture",
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(pool.Ensure(8, 70, 64, query, allocate),
                       "vt rocm: norm-quant scratch: pre-warm RmsNorm on this queue before graph capture",
                       std::runtime_error);
  CHECK(allocations == 1);
}

}  // namespace
