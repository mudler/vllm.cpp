// vllm.cpp runtime tests. Core W4A4 references mirror the pinned quant kernels;
// the W3-D packed-QKV case also ports QKVParallelLinear's loader topology from
// tests/model_executor/model_loader/test_reload.py:150 and the logical shard
// mapping exercised by tests/models/test_adapters.py:44-60.
// TRUE W4A4 (fp4 activations x fp4 weights) ops — the 27B path (notes §7). These
// validate the CPU kernels vt::ScaledFp4Quant + vt::MatmulNvfp4Fp4 against the
// pinned vLLM CPU truth:
//   - ScaledFp4Quant must be BYTE-EXACT vs vllm::RefScaledFp4Quant, and its
//     decode must reproduce vllm::RefNvfp4QuantDequant's x_dq.
//   - MatmulNvfp4Fp4( ScaledFp4Quant(x), W ) must equal vllm::RunNvfp4Emulation
//     (the emulated true-W4A4 linear) up to K-reduction float order.
// CPU-only (also the reference the future CUDA kernels validate against).
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32, kE2M1Lut
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/backend.h"
#ifdef VLLM_CPP_CUDA
#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"
#endif
#include "vt/cuda/nvfp4_autotune.h"
#include "vt/cuda/nvfp4_plan_cache.h"
#include "vt/cuda/nvfp4_tactic_ids.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

// Decode a fp4 activation row-trip from the ScaledFp4Quant outputs, mirroring the
// GEMM's a-operand: a_fp4 * f8(a_scale) / input_global_scale (== x_dq).
float DecodeActElem(const uint8_t* packed, const uint8_t* scale, int64_t k, int64_t row,
                    int64_t col, float input_global_scale) {
  const int64_t groups = k / 16;
  const uint8_t nib = (col % 2 == 0) ? (packed[(row * k + col) / 2] & 0x0FU)
                                     : (packed[(row * k + col) / 2] >> 4);
  const float mag = vllm::kE2M1Lut[nib & 0x7U] * ((nib & 0x8U) ? -1.0F : 1.0F);
  const float sf = vllm::F8E4M3ToF32(scale[row * groups + col / 16]);
  return mag * sf / input_global_scale;  // block_scale = sf/global
}

int64_t RoundUpTo(int64_t value, int64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

int64_t CutlassScaleOffset(int64_t row, int64_t col, int64_t padded_cols) {
  const int64_t m_tile = row / 128;
  const int64_t outer_m = row % 32;
  const int64_t inner_m = (row % 128) / 32;
  const int64_t k_tile = col / 4;
  const int64_t inner_k = col % 4;
  return ((((m_tile * (padded_cols / 4) + k_tile) * 32 + outer_m) * 4 +
            inner_m) *
               4 +
           inner_k);
}

std::vector<uint8_t> SwizzleScaleReference(const std::vector<uint8_t>& linear,
                                           int64_t rows, int64_t cols) {
  const int64_t padded_rows = RoundUpTo(rows, 128);
  const int64_t padded_cols = RoundUpTo(cols, 4);
  std::vector<uint8_t> swizzled(
      static_cast<size_t>(padded_rows * padded_cols), uint8_t{0});
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      swizzled[static_cast<size_t>(
          CutlassScaleOffset(row, col, padded_cols))] =
          linear[static_cast<size_t>(row * cols + col)];
    }
  }
  return swizzled;
}
}  // namespace

TEST_CASE("FlashInfer NVFP4 hybrid M buckets preserve small decode shapes") {
  using vt::cuda::nvfp4::HybridMBucket;
  const std::vector<std::pair<uint32_t, uint32_t>> cases = {
      {0, 1},       {1, 1},       {2, 2},       {3, 4},       {4, 4},
      {8, 8},       {16, 16},     {255, 256},   {256, 256},   {257, 512},
      {2048, 2048}, {2049, 2560}, {4096, 4096}, {4097, 8192}, {32768, 32768},
  };
  for (const auto& [value, expected] : cases) {
    CAPTURE(value);
    CHECK(HybridMBucket(value) == expected);
  }

  using vt::cuda::nvfp4::LegacyMBucket;
  CHECK(LegacyMBucket(0) == 16);
  CHECK(LegacyMBucket(1) == 16);
  CHECK(LegacyMBucket(2) == 16);
  CHECK(LegacyMBucket(8) == 16);
  CHECK(LegacyMBucket(16) == 16);
  CHECK(LegacyMBucket(17) == 32);
  CHECK_THROWS_AS(vt::cuda::nvfp4::NextPositivePowerOfTwo(0x80000001U),
                  std::overflow_error);

  using vt::cuda::nvfp4::HybridMTuningBuckets;
  CHECK(HybridMTuningBuckets(2048) ==
        std::vector<uint32_t>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                              768, 1024, 1280, 1536, 1792, 2048});
  CHECK(HybridMTuningBuckets(4096) ==
        std::vector<uint32_t>{1,    2,    4,    8,    16,   32,   64,
                              128,  256,  512,  768,  1024, 1280, 1536,
                              1792, 2048, 2560, 3072, 3584, 4096});
  CHECK(HybridMTuningBuckets(5000).back() == 5000);
}

TEST_CASE("FlashInfer SM12 NVFP4 tactics retain exact stable descriptor order") {
  using vt::cuda::nvfp4::TacticDescriptorForId;
  using vt::cuda::nvfp4::kFullTacticDescriptors;
  using vt::cuda::nvfp4::kFullTacticSetVersion;
  using vt::cuda::nvfp4::kW1TacticIds;
  using vt::cuda::nvfp4::kW1TacticSetVersion;

  constexpr std::array<std::array<int, 3>, 8> tiles{{
      {128, 32, 128},   {128, 32, 256},  {128, 64, 128},  {128, 64, 256},
      {128, 128, 128},  {128, 128, 256}, {256, 128, 128}, {128, 256, 128},
  }};
  REQUIRE(kFullTacticDescriptors.size() == tiles.size() * 4);
  for (size_t tile = 0; tile < tiles.size(); ++tile) {
    for (size_t variant = 0; variant < 4; ++variant) {
      const size_t id = tile * 4 + variant;
      const auto& descriptor = kFullTacticDescriptors[id];
      CAPTURE(id);
      CHECK(descriptor.id == static_cast<int>(id));
      CHECK(descriptor.tile_m == tiles[tile][0]);
      CHECK(descriptor.tile_n == tiles[tile][1]);
      CHECK(descriptor.tile_k == tiles[tile][2]);
      CHECK(descriptor.swap_ab == (variant == 0 || variant == 2));
      CHECK(descriptor.stream_k == (variant >= 2));
      CHECK(std::string(descriptor.name).empty() == false);
      CHECK(TacticDescriptorForId(static_cast<int>(id)) == &descriptor);
    }
  }
  CHECK(TacticDescriptorForId(-1) == nullptr);
  CHECK(TacticDescriptorForId(32) == nullptr);
  CHECK((kW1TacticIds == std::array<int, 4>{17, 25, 21, 29}));
  CHECK(kW1TacticSetVersion == 1);
  CHECK(kFullTacticSetVersion == 2);
}

TEST_CASE("NVFP4 plan key covers device architecture dtype shape and tactic ABI") {
  using vt::cuda::nvfp4::PlanKey;
  using vt::cuda::nvfp4::PlanKeyHash;
  const PlanKey base{16, 5120, 17408, 0, 121, 2, 1};
  std::unordered_set<PlanKey, PlanKeyHash> keys;
  keys.insert(base);
  auto key = base;
  key.m_bucket = 8;
  keys.insert(key);
  key = base;
  key.n = 1024;
  keys.insert(key);
  key = base;
  key.k = 5120;
  keys.insert(key);
  key = base;
  key.device_ordinal = 1;
  keys.insert(key);
  key = base;
  key.architecture = 120;
  keys.insert(key);
  key = base;
  key.output_dtype = 1;
  keys.insert(key);
  key = base;
  key.tactic_set_version = 2;
  keys.insert(key);
  CHECK(keys.size() == 8);
  CHECK(keys.contains(base));
}

TEST_CASE("NVFP4 plan tuning is single-flight per key and independent across keys") {
  using vt::cuda::nvfp4::PlanKey;
  using vt::cuda::nvfp4::ResolvePlan;
  using vt::cuda::nvfp4::SingleFlightPlanCache;
  SingleFlightPlanCache<int> cache;
  const PlanKey key{16, 5120, 17408, 0, 121, 2, 1};

  std::promise<void> owner_started;
  auto owner_started_future = owner_started.get_future();
  std::promise<void> release_owner;
  const auto release_future = release_owner.get_future().share();
  std::atomic<int> tune_count{0};
  std::vector<int> results(16, -1);
  std::vector<std::exception_ptr> errors(16);

  std::thread owner([&] {
    try {
      results[0] = ResolvePlan(
          cache, key, [] { return true; }, [&] {
            ++tune_count;
            owner_started.set_value();
            release_future.wait();
            return 7;
          });
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  owner_started_future.wait();

  std::vector<std::thread> waiters;
  for (size_t i = 1; i < results.size(); ++i) {
    waiters.emplace_back([&, i] {
      try {
        results[i] = ResolvePlan(cache, key, [] { return true; }, [&] {
          ++tune_count;
          return 99;
        });
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (cache.WaiterCountForTesting(key) != 15) std::this_thread::yield();
  release_owner.set_value();
  owner.join();
  for (auto& waiter : waiters) waiter.join();

  CHECK(tune_count.load() == 1);
  CHECK(cache.SizeForTesting() == 1);
  for (size_t i = 0; i < results.size(); ++i) {
    CAPTURE(i);
    CHECK(errors[i] == nullptr);
    CHECK(results[i] == 7);
  }

  // A tuner blocked on one shape never owns the global map mutex.
  const PlanKey blocked_key{8, 5120, 5120, 0, 121, 2, 1};
  const PlanKey independent_key{8, 1024, 5120, 0, 121, 2, 1};
  std::promise<void> blocked_started;
  auto blocked_started_future = blocked_started.get_future();
  std::promise<void> release_blocked;
  const auto blocked_release_future = release_blocked.get_future().share();
  int blocked_result = -1;
  std::thread blocked([&] {
    blocked_result = ResolvePlan(
        cache, blocked_key, [] { return true; }, [&] {
          blocked_started.set_value();
          blocked_release_future.wait();
          return 11;
        });
  });
  blocked_started_future.wait();
  CHECK(ResolvePlan(cache, independent_key, [] { return true; }, [] { return 13; }) == 13);
  release_blocked.set_value();
  blocked.join();
  CHECK(blocked_result == 11);
}

TEST_CASE("NVFP4 plan cache rejects capture misses and retries failed tuning") {
  using vt::cuda::nvfp4::PlanKey;
  using vt::cuda::nvfp4::ResolvePlan;
  using vt::cuda::nvfp4::SingleFlightPlanCache;
  SingleFlightPlanCache<int> cache;
  const PlanKey key{4, 5120, 17408, 0, 121, 2, 1};
  int capture_queries = 0;
  std::atomic<int> tune_count{0};

  CHECK_THROWS_WITH_AS(
      ResolvePlan(
          cache, key,
          [&] {
            ++capture_queries;
            return false;
          },
          [&] {
            ++tune_count;
            return 1;
          }),
      "NVFP4 plan cache miss while tuning is disallowed", std::runtime_error);
  CHECK(capture_queries == 1);
  CHECK(tune_count.load() == 0);
  CHECK(cache.SizeForTesting() == 0);

  std::promise<void> failed_owner_started;
  auto failed_owner_started_future = failed_owner_started.get_future();
  std::promise<void> release_failed_owner;
  const auto failed_release_future = release_failed_owner.get_future().share();
  std::vector<std::exception_ptr> failures(8);
  std::thread failed_owner([&] {
    try {
      static_cast<void>(ResolvePlan(cache, key, [] { return true; }, [&]() -> int {
        ++tune_count;
        failed_owner_started.set_value();
        failed_release_future.wait();
        throw std::runtime_error("tune failed");
      }));
    } catch (...) {
      failures[0] = std::current_exception();
    }
  });
  failed_owner_started_future.wait();
  std::vector<std::thread> failed_waiters;
  for (size_t i = 1; i < failures.size(); ++i) {
    failed_waiters.emplace_back([&, i] {
      try {
        static_cast<void>(ResolvePlan(cache, key, [] { return true; }, [&] {
          ++tune_count;
          return 99;
        }));
      } catch (...) {
        failures[i] = std::current_exception();
      }
    });
  }
  while (cache.WaiterCountForTesting(key) != 7) std::this_thread::yield();
  release_failed_owner.set_value();
  failed_owner.join();
  for (auto& waiter : failed_waiters) waiter.join();

  CHECK(tune_count.load() == 1);
  for (size_t i = 0; i < failures.size(); ++i) {
    CAPTURE(i);
    REQUIRE(failures[i] != nullptr);
    try {
      std::rethrow_exception(failures[i]);
    } catch (const std::runtime_error& error) {
      CHECK(std::string(error.what()) == "tune failed");
    }
  }
  CHECK(cache.SizeForTesting() == 0);
  CHECK(ResolvePlan(cache, key, [] { return true; }, [&] {
          ++tune_count;
          return 17;
        }) == 17);
  CHECK(tune_count.load() == 2);

  // A ready graph-capture lookup never evaluates the capture predicate.
  capture_queries = 0;
  CHECK(ResolvePlan(
            cache, key,
            [&] {
              ++capture_queries;
              return false;
            },
            [] { return 23; }) == 17);
  CHECK(capture_queries == 0);
}

TEST_CASE("scaled_fp4_quant CPU == vllm::RefScaledFp4Quant (byte-exact) + decode") {
  const int64_t M = 5, K = 64;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0F, 3.0F);
  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  const float input_global_scale = 7.3F;  // on-disk divisor (used directly)

  std::vector<uint8_t> ref_packed(static_cast<size_t>(M * K / 2));
  std::vector<uint8_t> ref_scale(static_cast<size_t>(M * K / 16));
  vllm::RefScaledFp4Quant(x.data(), M, K, input_global_scale, ref_packed.data(),
                          ref_scale.data());

  std::vector<uint8_t> op_packed(static_cast<size_t>(M * K / 2), 0);
  std::vector<uint8_t> op_scale(static_cast<size_t>(M * K / 16), 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
  Tensor tp = Tensor::Contiguous(op_packed.data(), DType::kI8, Cpu(), {M, K / 2});
  Tensor ts = Tensor::Contiguous(op_scale.data(), DType::kI8, Cpu(), {M, K / 16});
  Queue q = CpuQueue();
  vt::ScaledFp4Quant(q, tp, ts, tx, input_global_scale);

  for (size_t i = 0; i < ref_packed.size(); ++i) CHECK(op_packed[i] == ref_packed[i]);
  for (size_t i = 0; i < ref_scale.size(); ++i) CHECK(op_scale[i] == ref_scale[i]);

  // Decode == RefNvfp4QuantDequant x_dq (the emulation activation round-trip).
  std::vector<float> x_dq(static_cast<size_t>(M * K));
  vllm::RefNvfp4QuantDequant(x.data(), M, K, input_global_scale, x_dq.data());
  for (int64_t r = 0; r < M; ++r)
    for (int64_t c = 0; c < K; ++c) {
      const float got = DecodeActElem(op_packed.data(), op_scale.data(), K, r, c, input_global_scale);
      CHECK(got == doctest::Approx(x_dq[static_cast<size_t>(r * K + c)]).epsilon(1e-6));
    }
}

// Ports vllm@v0.25.0
// tests/kernels/quantization/test_nvfp4_quant.py::
//   test_python_util_matches_cpp_allocation,
//   test_quantize_to_fp4_with_padded_output, and
//   test_quantize_to_fp4_padded{,_no_sf_swizzled}.
// The direct producer must be byte-identical to linear production followed by
// the standalone swizzle, including every zero-filled padded scale slot.
TEST_CASE("scaled_fp4_quant direct CUTLASS scales match linear swizzle on CPU") {
  Queue queue = CpuQueue();
  constexpr float kInputGlobalScale = 7.3F;
  const std::array<std::pair<int64_t, int64_t>, 6> shapes{{
      {1, 64}, {32, 4096}, {127, 1024}, {128, 4096},
      {256, 16384}, {32, 14336},
  }};
  for (const auto& [m, k] : shapes) {
    CAPTURE(m);
    CAPTURE(k);
    std::vector<float> input(static_cast<size_t>(m * k));
    for (int64_t index = 0; index < m * k; ++index) {
      input[static_cast<size_t>(index)] =
          static_cast<float>((index * 37) % 257 - 128) / 19.0F;
    }
    std::vector<uint8_t> linear_packed(static_cast<size_t>(m * k / 2));
    std::vector<uint8_t> direct_packed(linear_packed.size(), uint8_t{0xA5});
    std::vector<uint8_t> linear_scale(static_cast<size_t>(m * k / 16));
    const int64_t padded_rows = RoundUpTo(m, 128);
    const int64_t padded_cols = RoundUpTo(k / 16, 4);
    std::vector<uint8_t> direct_scale(
        static_cast<size_t>(padded_rows * padded_cols), uint8_t{0xA5});

    Tensor x = Tensor::Contiguous(input.data(), DType::kF32, Cpu(), {m, k});
    Tensor linear_packed_tensor = Tensor::Contiguous(
        linear_packed.data(), DType::kI8, Cpu(), {m, k / 2});
    Tensor direct_packed_tensor = Tensor::Contiguous(
        direct_packed.data(), DType::kI8, Cpu(), {m, k / 2});
    Tensor linear_scale_tensor = Tensor::Contiguous(
        linear_scale.data(), DType::kI8, Cpu(), {m, k / 16});
    Tensor direct_scale_tensor = Tensor::Contiguous(
        direct_scale.data(), DType::kI8, Cpu(), {padded_rows, padded_cols});

    vt::ScaledFp4Quant(queue, linear_packed_tensor, linear_scale_tensor, x,
                       kInputGlobalScale);
    vt::ScaledFp4Quant(queue, direct_packed_tensor, direct_scale_tensor, x,
                       kInputGlobalScale,
                       vt::Fp4ScaleLayout::kCutlassSwizzled);

    CHECK(direct_packed == linear_packed);
    CHECK(direct_scale ==
          SwizzleScaleReference(linear_scale, m, k / 16));
  }
}

TEST_CASE("direct CUTLASS scale layout is explicit and validates exact shape") {
  Queue queue = CpuQueue();
  constexpr int64_t kM = 1;
  constexpr int64_t kK = 64;
  std::vector<float> input(static_cast<size_t>(kM * kK), 1.0F);
  std::vector<uint8_t> packed(static_cast<size_t>(kM * kK / 2));
  std::vector<uint8_t> ambiguous_scale(static_cast<size_t>(kM * kK / 16));
  Tensor x = Tensor::Contiguous(input.data(), DType::kF32, Cpu(), {kM, kK});
  Tensor packed_tensor = Tensor::Contiguous(
      packed.data(), DType::kI8, Cpu(), {kM, kK / 2});
  Tensor ambiguous_tensor = Tensor::Contiguous(
      ambiguous_scale.data(), DType::kI8, Cpu(), {kM, kK / 16});
  CHECK_THROWS_AS(
      vt::ScaledFp4Quant(queue, packed_tensor, ambiguous_tensor, x, 7.3F,
                         vt::Fp4ScaleLayout::kCutlassSwizzled),
      std::runtime_error);
}

TEST_CASE("matmul_nvfp4_fp4 CPU == vllm::RunNvfp4Emulation") {
  const int64_t M = 4, K = 96, N = 7;
  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);

  // Random bf16-ish activations (f32).
  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);

  // Random fp4 weight bytes [N, K/2] + fp8 group scales [N, K/16] (sane, no NaN)
  // + a global divisor.
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& b : w_packed) b = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& s : w_scale) s = vllm::F32ToF8E4M3(scale_d(rng));
  const float weight_global_scale_disk = 6.1F;   // divisor
  const float input_global_scale_disk = 9.4F;    // divisor

  // Reference: the emulated true-W4A4 linear.
  std::vector<float> ref(static_cast<size_t>(M * N));
  vllm::RunNvfp4Emulation(x.data(), M, K, w_packed.data(), w_scale.data(),
                          weight_global_scale_disk, input_global_scale_disk, N, ref.data());

  // Op path: quantize activations, then fp4xfp4 GEMM with the folded alpha.
  std::vector<uint8_t> a_packed(static_cast<size_t>(M * K / 2), 0);
  std::vector<uint8_t> a_scale(static_cast<size_t>(M * K / 16), 0);
  Queue q = CpuQueue();
  {
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
    Tensor tp = Tensor::Contiguous(a_packed.data(), DType::kI8, Cpu(), {M, K / 2});
    Tensor ts = Tensor::Contiguous(a_scale.data(), DType::kI8, Cpu(), {M, K / 16});
    vt::ScaledFp4Quant(q, tp, ts, tx, input_global_scale_disk);
  }
  const float alpha = (1.0F / input_global_scale_disk) * (1.0F / weight_global_scale_disk);
  std::vector<float> out(static_cast<size_t>(M * N), -1.0F);
  Tensor tap = Tensor::Contiguous(a_packed.data(), DType::kI8, Cpu(), {M, K / 2});
  Tensor tas = Tensor::Contiguous(a_scale.data(), DType::kI8, Cpu(), {M, K / 16});
  Tensor tbp = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {N, K / 2});
  Tensor tbs = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {N, K / 16});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {M, N});
  vt::MatmulNvfp4Fp4(q, to, tap, tas, tbp, tbs, alpha);

  for (int64_t i = 0; i < M * N; ++i) {
    // Real-arithmetic identical to RunNvfp4Emulation; only float K-order differs.
    CHECK(out[static_cast<size_t>(i)] ==
          doctest::Approx(ref[static_cast<size_t>(i)]).epsilon(1e-5).scale(1.0));
  }
}

// --- CUDA device-vs-CPU cross-check (GB10; skips cleanly without a GPU) --------
namespace {
bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor GpuTensor(const std::vector<int64_t>& shape) {
  Tensor t;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}
}  // namespace

TEST_CASE("scaled_fp4_quant + matmul_nvfp4_fp4 CUDA == CPU") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();

  const int64_t M = 40, K = 128, N = 33;  // M >= 32 exercises the WMMA path
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);

  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& v : w_packed) v = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& v : w_scale) v = vllm::F32ToF8E4M3(scale_d(rng));
  const float input_global_scale = 8.2F, weight_global_scale = 5.5F;
  const float alpha = (1.0F / input_global_scale) * (1.0F / weight_global_scale);

  // CPU reference (already validated vs vllm::RunNvfp4Emulation above).
  std::vector<uint8_t> cpu_ap(static_cast<size_t>(M * K / 2), 0), cpu_as(static_cast<size_t>(M * K / 16), 0);
  std::vector<float> cpu_out(static_cast<size_t>(M * N), 0);
  {
    Queue cq = CpuQueue();
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
    Tensor tp = Tensor::Contiguous(cpu_ap.data(), DType::kI8, Cpu(), {M, K / 2});
    Tensor ts = Tensor::Contiguous(cpu_as.data(), DType::kI8, Cpu(), {M, K / 16});
    vt::ScaledFp4Quant(cq, tp, ts, tx, input_global_scale);
    Tensor tbp = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {N, K / 2});
    Tensor tbs = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {N, K / 16});
    Tensor to = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {M, N});
    vt::MatmulNvfp4Fp4(cq, to, tp, ts, tbp, tbs, alpha);
  }

  // Device path.
  auto up = [&](const void* h, size_t nb) { void* p = b.Alloc(nb); b.Copy(gq, p, h, nb); return p; };
  void* dx = up(x.data(), x.size() * sizeof(float));
  void* dbp = up(w_packed.data(), w_packed.size());
  void* dbs = up(w_scale.data(), w_scale.size());
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dout = b.Alloc(static_cast<size_t>(M * N) * sizeof(float));
  Tensor tx = GpuTensor({M, K}); tx.data = dx; tx.dtype = DType::kF32; tx.device = Gpu();
  Tensor tap = GpuTensor({M, K / 2}); tap.data = dap; tap.dtype = DType::kI8; tap.device = Gpu();
  Tensor tas = GpuTensor({M, K / 16}); tas.data = das; tas.dtype = DType::kI8; tas.device = Gpu();
  Tensor tbp = GpuTensor({N, K / 2}); tbp.data = dbp; tbp.dtype = DType::kI8; tbp.device = Gpu();
  Tensor tbs = GpuTensor({N, K / 16}); tbs.data = dbs; tbs.dtype = DType::kI8; tbs.device = Gpu();
  Tensor to = GpuTensor({M, N}); to.data = dout; to.dtype = DType::kF32; to.device = Gpu();
  vt::ScaledFp4Quant(gq, tap, tas, tx, input_global_scale);
  vt::MatmulNvfp4Fp4(gq, to, tap, tas, tbp, tbs, alpha);

  std::vector<uint8_t> g_ap(static_cast<size_t>(M * K / 2)), g_as(static_cast<size_t>(M * K / 16));
  std::vector<float> g_out(static_cast<size_t>(M * N));
  b.Copy(gq, g_ap.data(), dap, g_ap.size());
  b.Copy(gq, g_as.data(), das, g_as.size());
  b.Copy(gq, g_out.data(), dout, g_out.size() * sizeof(float));
  b.Synchronize(gq);

  // Quant: the device kernel uses the hardware fp8 cast (matches the real vLLM
  // kernel), the CPU op the emulation codec — expect byte-exact for almost all
  // groups; a handful of fp8-tie edge cases are benign (allow a small fraction).
  size_t scale_mismatch = 0;
  for (size_t i = 0; i < g_as.size(); ++i) scale_mismatch += (g_as[i] != cpu_as[i]);
  CHECK(scale_mismatch <= g_as.size() / 50 + 1);  // <= ~2% fp8 ties
  // GEMM close device-vs-CPU (bf16 tensor-core dequant vs f32 CPU): matmul tol.
  for (size_t i = 0; i < g_out.size(); ++i)
    CHECK(g_out[i] == doctest::Approx(cpu_out[i]).epsilon(0.02).scale(1.0));

  for (void* p : {dx, dbp, dbs, dap, das, dout}) b.Free(p);
  b.DestroyQueue(gq);
}

TEST_CASE("scaled_fp4_quant CUDA direct scales match linear swizzle") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue queue = b.CreateQueue();
  constexpr float kInputGlobalScale = 8.2F;

  auto run_check = [&](int64_t m, int64_t k, DType dtype) {
    CAPTURE(m);
    CAPTURE(k);
    CAPTURE(static_cast<int>(dtype));
    std::vector<float> input_f32(static_cast<size_t>(m * k));
    for (int64_t index = 0; index < m * k; ++index) {
      input_f32[static_cast<size_t>(index)] =
          static_cast<float>((index * 37) % 257 - 128) / 19.0F;
    }
    std::vector<uint16_t> input_bf16;
    const void* input_host = input_f32.data();
    size_t input_bytes = input_f32.size() * sizeof(float);
    if (dtype == DType::kBF16) {
      input_bf16.resize(input_f32.size());
      std::transform(input_f32.begin(), input_f32.end(), input_bf16.begin(),
                     FloatToBf16);
      input_host = input_bf16.data();
      input_bytes = input_bf16.size() * sizeof(uint16_t);
    }
    const int64_t padded_rows = RoundUpTo(m, 128);
    const int64_t padded_cols = RoundUpTo(k / 16, 4);
    auto allocate = [&](size_t bytes) { return b.Alloc(bytes); };
    void* input_device = allocate(input_bytes);
    b.Copy(queue, input_device, input_host, input_bytes);
    void* linear_packed_device =
        allocate(static_cast<size_t>(m * k / 2));
    void* direct_packed_device =
        allocate(static_cast<size_t>(m * k / 2));
    void* linear_scale_device =
        allocate(static_cast<size_t>(m * k / 16));
    void* direct_scale_device =
        allocate(static_cast<size_t>(padded_rows * padded_cols));
    b.Memset(queue, direct_scale_device, 0xA5,
             static_cast<size_t>(padded_rows * padded_cols));

    Tensor input = GpuTensor({m, k});
    input.data = input_device;
    input.dtype = dtype;
    input.device = Gpu();
    Tensor linear_packed = GpuTensor({m, k / 2});
    linear_packed.data = linear_packed_device;
    linear_packed.dtype = DType::kI8;
    linear_packed.device = Gpu();
    Tensor direct_packed = GpuTensor({m, k / 2});
    direct_packed.data = direct_packed_device;
    direct_packed.dtype = DType::kI8;
    direct_packed.device = Gpu();
    Tensor linear_scale = GpuTensor({m, k / 16});
    linear_scale.data = linear_scale_device;
    linear_scale.dtype = DType::kI8;
    linear_scale.device = Gpu();
    Tensor direct_scale = GpuTensor({padded_rows, padded_cols});
    direct_scale.data = direct_scale_device;
    direct_scale.dtype = DType::kI8;
    direct_scale.device = Gpu();

    vt::ScaledFp4Quant(queue, linear_packed, linear_scale, input,
                       kInputGlobalScale);
    vt::ScaledFp4Quant(queue, direct_packed, direct_scale, input,
                       kInputGlobalScale,
                       vt::Fp4ScaleLayout::kCutlassSwizzled);

    std::vector<uint8_t> linear_packed_host(
        static_cast<size_t>(m * k / 2));
    std::vector<uint8_t> direct_packed_host(linear_packed_host.size());
    std::vector<uint8_t> linear_scale_host(
        static_cast<size_t>(m * k / 16));
    std::vector<uint8_t> direct_scale_host(
        static_cast<size_t>(padded_rows * padded_cols));
    b.Copy(queue, linear_packed_host.data(), linear_packed_device,
           linear_packed_host.size());
    b.Copy(queue, direct_packed_host.data(), direct_packed_device,
           direct_packed_host.size());
    b.Copy(queue, linear_scale_host.data(), linear_scale_device,
           linear_scale_host.size());
    b.Copy(queue, direct_scale_host.data(), direct_scale_device,
           direct_scale_host.size());
    b.Synchronize(queue);

    CHECK(direct_packed_host == linear_packed_host);
    CHECK(direct_scale_host ==
          SwizzleScaleReference(linear_scale_host, m, k / 16));
    for (void* pointer : {input_device, linear_packed_device,
                          direct_packed_device, linear_scale_device,
                          direct_scale_device}) {
      b.Free(pointer);
    }
  };

  run_check(1, 64, DType::kF32);
  run_check(127, 1024, DType::kF32);
  run_check(1, 5120, DType::kBF16);
  run_check(32, 14336, DType::kBF16);
  run_check(9, 17408, DType::kBF16);
  b.DestroyQueue(queue);
}

TEST_CASE("sigmoid_gate_fp4_quant CPU == SigmoidGateBf16 + ScaledFp4Quant (BYTE-EXACT)") {
  Queue queue = CpuQueue();
  constexpr float kInputGlobalScale = 6.7F;
  for (DType attn_dtype : {DType::kF32, DType::kBF16}) {
    for (const auto& [m, k] :
         {std::pair<int64_t, int64_t>{1, 64}, {37, 128}}) {
      CAPTURE(m);
      CAPTURE(k);
      CAPTURE(static_cast<int>(attn_dtype));
      std::mt19937 rng(static_cast<unsigned>(701 + m * 131 + k));
      std::normal_distribution<float> normal(0.0F, 2.0F);
      std::vector<float> attn_f32(static_cast<size_t>(m * k));
      std::vector<float> gate_f32(static_cast<size_t>(m * k));
      for (float& value : attn_f32) value = normal(rng);
      for (float& value : gate_f32) value = normal(rng);
      std::vector<uint16_t> attn_bf16;
      void* attn_data = attn_f32.data();
      if (attn_dtype == DType::kBF16) {
        attn_bf16.resize(attn_f32.size());
        std::transform(attn_f32.begin(), attn_f32.end(), attn_bf16.begin(),
                       FloatToBf16);
        attn_data = attn_bf16.data();
      }

      std::vector<uint16_t> gated(static_cast<size_t>(m * k));
      std::vector<uint8_t> reference_packed(static_cast<size_t>(m * k / 2));
      std::vector<uint8_t> reference_scale(static_cast<size_t>(m * k / 16));
      std::vector<uint8_t> fused_packed(reference_packed.size());
      std::vector<uint8_t> fused_scale(reference_scale.size());
      Tensor attn = Tensor::Contiguous(attn_data, attn_dtype, Cpu(), {m, k});
      Tensor gate = Tensor::Contiguous(gate_f32.data(), DType::kF32, Cpu(), {m, k});
      Tensor gated_t =
          Tensor::Contiguous(gated.data(), DType::kBF16, Cpu(), {m, k});
      Tensor ref_packed = Tensor::Contiguous(
          reference_packed.data(), DType::kI8, Cpu(), {m, k / 2});
      Tensor ref_scale = Tensor::Contiguous(
          reference_scale.data(), DType::kI8, Cpu(), {m, k / 16});
      Tensor got_packed = Tensor::Contiguous(
          fused_packed.data(), DType::kI8, Cpu(), {m, k / 2});
      Tensor got_scale = Tensor::Contiguous(
          fused_scale.data(), DType::kI8, Cpu(), {m, k / 16});

      vt::SigmoidGateBf16(queue, gated_t, attn, gate);
      vt::ScaledFp4Quant(queue, ref_packed, ref_scale, gated_t, kInputGlobalScale);
      vt::SigmoidGateFp4Quant(queue, got_packed, got_scale, attn, gate,
                              kInputGlobalScale);
      CHECK(fused_packed == reference_packed);
      CHECK(fused_scale == reference_scale);
    }
  }
}

TEST_CASE("sigmoid_gate_fp4_quant CUDA == SigmoidGateBf16 + ScaledFp4Quant (BYTE-EXACT)") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float input_global_scale = 6.7F;

  auto run_check = [&](int64_t M, int64_t K) {
    std::mt19937 rng(static_cast<unsigned>(707 + M * 131 + K));
    std::normal_distribution<float> nd(0.0F, 2.0F);
    std::vector<float> attn(static_cast<size_t>(M * K)), gate(static_cast<size_t>(M * K));
    for (auto& v : attn) v = nd(rng);
    for (auto& v : gate) v = nd(rng);

    auto upl = [&](const void* h, size_t nb) { void* p = b.Alloc(nb); b.Copy(gq, p, h, nb); return p; };
    void* dattn = upl(attn.data(), attn.size() * sizeof(float));
    void* dgate = upl(gate.data(), gate.size() * sizeof(float));
    Tensor tattn = GpuTensor({M, K}); tattn.data = dattn; tattn.dtype = DType::kF32; tattn.device = Gpu();
    Tensor tgate = GpuTensor({M, K}); tgate.data = dgate; tgate.dtype = DType::kF32; tgate.device = Gpu();

    // Unfused reference: SigmoidGateBf16 -> bf16 gated, then ScaledFp4Quant.
    void* dgated = b.Alloc(static_cast<size_t>(M * K) * sizeof(uint16_t));
    void* dref_ap = b.Alloc(static_cast<size_t>(M * K / 2));
    void* dref_as = b.Alloc(static_cast<size_t>(M * K / 16));
    Tensor tgated = GpuTensor({M, K}); tgated.data = dgated; tgated.dtype = DType::kBF16; tgated.device = Gpu();
    Tensor tref_ap = GpuTensor({M, K / 2}); tref_ap.data = dref_ap; tref_ap.dtype = DType::kI8; tref_ap.device = Gpu();
    Tensor tref_as = GpuTensor({M, K / 16}); tref_as.data = dref_as; tref_as.dtype = DType::kI8; tref_as.device = Gpu();
    vt::SigmoidGateBf16(gq, tgated, tattn, tgate);
    vt::ScaledFp4Quant(gq, tref_ap, tref_as, tgated, input_global_scale);

    // Fused (linear scale layout).
    void* dfu_ap = b.Alloc(static_cast<size_t>(M * K / 2));
    void* dfu_as = b.Alloc(static_cast<size_t>(M * K / 16));
    Tensor tfu_ap = GpuTensor({M, K / 2}); tfu_ap.data = dfu_ap; tfu_ap.dtype = DType::kI8; tfu_ap.device = Gpu();
    Tensor tfu_as = GpuTensor({M, K / 16}); tfu_as.data = dfu_as; tfu_as.dtype = DType::kI8; tfu_as.device = Gpu();
    vt::SigmoidGateFp4Quant(gq, tfu_ap, tfu_as, tattn, tgate, input_global_scale);
    b.Synchronize(gq);

    std::vector<uint8_t> ref_ap(static_cast<size_t>(M * K / 2)), fu_ap(ref_ap.size());
    std::vector<uint8_t> ref_as(static_cast<size_t>(M * K / 16)), fu_as(ref_as.size());
    b.Copy(gq, ref_ap.data(), dref_ap, ref_ap.size());
    b.Copy(gq, fu_ap.data(), dfu_ap, fu_ap.size());
    b.Copy(gq, ref_as.data(), dref_as, ref_as.size());
    b.Copy(gq, fu_as.data(), dfu_as, fu_as.size());
    b.Synchronize(gq);
    CHECK(fu_ap == ref_ap);
    CHECK(fu_as == ref_as);

    for (void* p : {dattn, dgate, dgated, dref_ap, dref_as, dfu_ap, dfu_as}) b.Free(p);
  };
  run_check(1, 64);
  run_check(37, 128);
  run_check(128, 512);
}

TEST_CASE("silu_mul_fp4_quant CUDA == MoeSiluMul + ScaledFp4Quant (BYTE-EXACT)") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float input_global_scale = 6.7F;

  auto run_check = [&](int64_t M, int64_t I) {
    std::mt19937 rng(static_cast<unsigned>(99 + M * 131 + I));
    std::normal_distribution<float> nd(0.0F, 2.0F);
    std::vector<float> gate(static_cast<size_t>(M * I)), up(static_cast<size_t>(M * I));
    for (auto& v : gate) v = nd(rng);
    for (auto& v : up) v = nd(rng);

    auto upl = [&](const void* h, size_t nb) { void* p = b.Alloc(nb); b.Copy(gq, p, h, nb); return p; };
    void* dgate = upl(gate.data(), gate.size() * sizeof(float));
    void* dup = upl(up.data(), up.size() * sizeof(float));
    Tensor tgate = GpuTensor({M, I}); tgate.data = dgate; tgate.dtype = DType::kF32; tgate.device = Gpu();
    Tensor tup = GpuTensor({M, I}); tup.data = dup; tup.dtype = DType::kF32; tup.device = Gpu();

    // Unfused reference: MoeSiluMul -> bf16 act, then ScaledFp4Quant.
    void* dact = b.Alloc(static_cast<size_t>(M * I) * sizeof(uint16_t));
    void* dref_ap = b.Alloc(static_cast<size_t>(M * I / 2));
    void* dref_as = b.Alloc(static_cast<size_t>(M * I / 16));
    Tensor tact = GpuTensor({M, I}); tact.data = dact; tact.dtype = DType::kBF16; tact.device = Gpu();
    Tensor tref_ap = GpuTensor({M, I / 2}); tref_ap.data = dref_ap; tref_ap.dtype = DType::kI8; tref_ap.device = Gpu();
    Tensor tref_as = GpuTensor({M, I / 16}); tref_as.data = dref_as; tref_as.dtype = DType::kI8; tref_as.device = Gpu();
    vt::MoeSiluMul(gq, tact, tgate, tup);
    vt::ScaledFp4Quant(gq, tref_ap, tref_as, tact, input_global_scale);

    // Fused.
    void* dfu_ap = b.Alloc(static_cast<size_t>(M * I / 2));
    void* dfu_as = b.Alloc(static_cast<size_t>(M * I / 16));
    Tensor tfu_ap = GpuTensor({M, I / 2}); tfu_ap.data = dfu_ap; tfu_ap.dtype = DType::kI8; tfu_ap.device = Gpu();
    Tensor tfu_as = GpuTensor({M, I / 16}); tfu_as.data = dfu_as; tfu_as.dtype = DType::kI8; tfu_as.device = Gpu();
    vt::SiluMulFp4Quant(gq, tfu_ap, tfu_as, tgate, tup, input_global_scale);

    const int64_t padded_rows = RoundUpTo(M, 128);
    const int64_t padded_cols = RoundUpTo(I / 16, 4);
    void* ddir_ap = b.Alloc(static_cast<size_t>(M * I / 2));
    void* ddir_as =
        b.Alloc(static_cast<size_t>(padded_rows * padded_cols));
    b.Memset(gq, ddir_as, 0xA5,
             static_cast<size_t>(padded_rows * padded_cols));
    Tensor tdir_ap = GpuTensor({M, I / 2});
    tdir_ap.data = ddir_ap;
    tdir_ap.dtype = DType::kI8;
    tdir_ap.device = Gpu();
    Tensor tdir_as = GpuTensor({padded_rows, padded_cols});
    tdir_as.data = ddir_as;
    tdir_as.dtype = DType::kI8;
    tdir_as.device = Gpu();
    vt::SiluMulFp4Quant(gq, tdir_ap, tdir_as, tgate, tup,
                        input_global_scale,
                        vt::Fp4ScaleLayout::kCutlassSwizzled);

    std::vector<uint8_t> ref_ap(static_cast<size_t>(M * I / 2)), ref_as(static_cast<size_t>(M * I / 16));
    std::vector<uint8_t> fu_ap(static_cast<size_t>(M * I / 2)), fu_as(static_cast<size_t>(M * I / 16));
    std::vector<uint8_t> dir_ap(fu_ap.size());
    std::vector<uint8_t> dir_as(
        static_cast<size_t>(padded_rows * padded_cols));
    b.Copy(gq, ref_ap.data(), dref_ap, ref_ap.size());
    b.Copy(gq, ref_as.data(), dref_as, ref_as.size());
    b.Copy(gq, fu_ap.data(), dfu_ap, fu_ap.size());
    b.Copy(gq, fu_as.data(), dfu_as, fu_as.size());
    b.Copy(gq, dir_ap.data(), ddir_ap, dir_ap.size());
    b.Copy(gq, dir_as.data(), ddir_as, dir_as.size());
    b.Synchronize(gq);

    size_t pmis = 0, smis = 0;
    for (size_t i = 0; i < ref_ap.size(); ++i) pmis += (fu_ap[i] != ref_ap[i]);
    for (size_t i = 0; i < ref_as.size(); ++i) smis += (fu_as[i] != ref_as[i]);
    CHECK(pmis == 0);  // fused packed fp4 == unfused, byte-for-byte
    CHECK(smis == 0);  // fused fp8 block scales == unfused, byte-for-byte
    CHECK(dir_ap == fu_ap);
    CHECK(dir_as == SwizzleScaleReference(fu_as, M, I / 16));

    for (void* p : {dgate, dup, dact, dref_ap, dref_as, dfu_ap, dfu_as,
                    ddir_ap, ddir_as}) {
      b.Free(p);
    }
  };

  run_check(1, 64);      // decode single-row
  run_check(8, 256);     // small batch, many groups/row
  run_check(40, 128);    // prefill-ish
  run_check(37, 2048);   // the real 27B intermediate_size class, non-32-aligned M
  b.DestroyQueue(gq);
}

// Port of vllm@e24d1b24
// tests/kernels/quantization/test_silu_mul_nvfp4_quant.py::
// test_silu_mul_nvfp4_quant. The local gate is intentionally stronger than the
// upstream dequantized tolerance: the fused producer must preserve the eager
// BF16 SiluAndMul -> ScaledFp4Quant boundary byte-for-byte.
TEST_CASE("silu_and_mul_nvfp4_quant one-input CPU is BYTE-EXACT") {
  Queue queue = CpuQueue();
  constexpr float kInputGlobalScale = 6.7F;
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    for (const auto& [m, i] :
         {std::pair<int64_t, int64_t>{1, 64}, {37, 128}}) {
      CAPTURE(m);
      CAPTURE(i);
      CAPTURE(static_cast<int>(dtype));
      std::mt19937 rng(static_cast<unsigned>(401 + m * 131 + i));
      std::normal_distribution<float> normal(0.0F, 2.0F);
      std::vector<float> input_f32(static_cast<size_t>(m * 2 * i));
      for (float& value : input_f32) value = normal(rng);
      std::vector<uint16_t> input_bf16;
      void* input_data = input_f32.data();
      if (dtype == DType::kBF16) {
        input_bf16.resize(input_f32.size());
        std::transform(input_f32.begin(), input_f32.end(), input_bf16.begin(),
                       FloatToBf16);
        input_data = input_bf16.data();
      }

      std::vector<uint16_t> activation(static_cast<size_t>(m * i));
      std::vector<uint8_t> reference_packed(static_cast<size_t>(m * i / 2));
      std::vector<uint8_t> reference_scale(static_cast<size_t>(m * i / 16));
      std::vector<uint8_t> fused_packed(reference_packed.size());
      std::vector<uint8_t> fused_scale(reference_scale.size());
      Tensor input = Tensor::Contiguous(input_data, dtype, Cpu(), {m, 2 * i});
      Tensor act = Tensor::Contiguous(activation.data(), DType::kBF16, Cpu(),
                                      {m, i});
      Tensor ref_packed = Tensor::Contiguous(
          reference_packed.data(), DType::kI8, Cpu(), {m, i / 2});
      Tensor ref_scale = Tensor::Contiguous(
          reference_scale.data(), DType::kI8, Cpu(), {m, i / 16});
      Tensor got_packed = Tensor::Contiguous(
          fused_packed.data(), DType::kI8, Cpu(), {m, i / 2});
      Tensor got_scale = Tensor::Contiguous(
          fused_scale.data(), DType::kI8, Cpu(), {m, i / 16});

      vt::SiluAndMul(queue, act, input);
      vt::ScaledFp4Quant(queue, ref_packed, ref_scale, act,
                         kInputGlobalScale);
      vt::SiluAndMulFp4Quant(queue, got_packed, got_scale, input,
                             kInputGlobalScale);
      CHECK(fused_packed == reference_packed);
      CHECK(fused_scale == reference_scale);
    }
  }
}

TEST_CASE("fused SiLU NVFP4 producers emit direct CUTLASS scales on CPU") {
  Queue queue = CpuQueue();
  constexpr int64_t kM = 37;
  constexpr int64_t kI = 128;
  constexpr float kInputGlobalScale = 6.7F;
  std::vector<float> gate(static_cast<size_t>(kM * kI));
  std::vector<float> up(gate.size());
  for (int64_t index = 0; index < kM * kI; ++index) {
    gate[static_cast<size_t>(index)] =
        static_cast<float>((index * 17) % 193 - 96) / 23.0F;
    up[static_cast<size_t>(index)] =
        static_cast<float>((index * 29) % 181 - 90) / 17.0F;
  }
  std::vector<float> gate_up(static_cast<size_t>(kM * 2 * kI));
  for (int64_t row = 0; row < kM; ++row) {
    std::copy_n(gate.data() + row * kI, kI,
                gate_up.data() + row * 2 * kI);
    std::copy_n(up.data() + row * kI, kI,
                gate_up.data() + row * 2 * kI + kI);
  }

  const int64_t padded_rows = RoundUpTo(kM, 128);
  const int64_t padded_cols = RoundUpTo(kI / 16, 4);
  auto check_scale = [&](const std::vector<uint8_t>& linear_scale,
                         const std::vector<uint8_t>& direct_scale) {
    CHECK(direct_scale ==
          SwizzleScaleReference(linear_scale, kM, kI / 16));
  };

  std::vector<uint8_t> two_linear_packed(static_cast<size_t>(kM * kI / 2));
  std::vector<uint8_t> two_direct_packed(two_linear_packed.size());
  std::vector<uint8_t> two_linear_scale(static_cast<size_t>(kM * kI / 16));
  std::vector<uint8_t> two_direct_scale(
      static_cast<size_t>(padded_rows * padded_cols), uint8_t{0xA5});
  Tensor gate_tensor =
      Tensor::Contiguous(gate.data(), DType::kF32, Cpu(), {kM, kI});
  Tensor up_tensor =
      Tensor::Contiguous(up.data(), DType::kF32, Cpu(), {kM, kI});
  Tensor two_linear_packed_tensor = Tensor::Contiguous(
      two_linear_packed.data(), DType::kI8, Cpu(), {kM, kI / 2});
  Tensor two_direct_packed_tensor = Tensor::Contiguous(
      two_direct_packed.data(), DType::kI8, Cpu(), {kM, kI / 2});
  Tensor two_linear_scale_tensor = Tensor::Contiguous(
      two_linear_scale.data(), DType::kI8, Cpu(), {kM, kI / 16});
  Tensor two_direct_scale_tensor = Tensor::Contiguous(
      two_direct_scale.data(), DType::kI8, Cpu(),
      {padded_rows, padded_cols});
  vt::SiluMulFp4Quant(queue, two_linear_packed_tensor,
                      two_linear_scale_tensor, gate_tensor, up_tensor,
                      kInputGlobalScale);
  vt::SiluMulFp4Quant(queue, two_direct_packed_tensor,
                      two_direct_scale_tensor, gate_tensor, up_tensor,
                      kInputGlobalScale,
                      vt::Fp4ScaleLayout::kCutlassSwizzled);
  CHECK(two_direct_packed == two_linear_packed);
  check_scale(two_linear_scale, two_direct_scale);

  std::vector<uint8_t> one_linear_packed(two_linear_packed.size());
  std::vector<uint8_t> one_direct_packed(two_linear_packed.size());
  std::vector<uint8_t> one_linear_scale(two_linear_scale.size());
  std::vector<uint8_t> one_direct_scale(
      static_cast<size_t>(padded_rows * padded_cols), uint8_t{0xA5});
  Tensor gate_up_tensor = Tensor::Contiguous(
      gate_up.data(), DType::kF32, Cpu(), {kM, 2 * kI});
  Tensor one_linear_packed_tensor = Tensor::Contiguous(
      one_linear_packed.data(), DType::kI8, Cpu(), {kM, kI / 2});
  Tensor one_direct_packed_tensor = Tensor::Contiguous(
      one_direct_packed.data(), DType::kI8, Cpu(), {kM, kI / 2});
  Tensor one_linear_scale_tensor = Tensor::Contiguous(
      one_linear_scale.data(), DType::kI8, Cpu(), {kM, kI / 16});
  Tensor one_direct_scale_tensor = Tensor::Contiguous(
      one_direct_scale.data(), DType::kI8, Cpu(),
      {padded_rows, padded_cols});
  vt::SiluAndMulFp4Quant(queue, one_linear_packed_tensor,
                         one_linear_scale_tensor, gate_up_tensor,
                         kInputGlobalScale);
  vt::SiluAndMulFp4Quant(queue, one_direct_packed_tensor,
                         one_direct_scale_tensor, gate_up_tensor,
                         kInputGlobalScale,
                         vt::Fp4ScaleLayout::kCutlassSwizzled);
  CHECK(one_direct_packed == one_linear_packed);
  check_scale(one_linear_scale, one_direct_scale);
}

TEST_CASE("silu_and_mul_nvfp4_quant one-input CUDA is BYTE-EXACT") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float input_global_scale = 6.7F;

  auto run_check = [&](int64_t m, int64_t i, DType dtype,
                       bool misaligned_input = false) {
    CAPTURE(m);
    CAPTURE(i);
    CAPTURE(static_cast<int>(dtype));
    CAPTURE(misaligned_input);
    std::mt19937 rng(static_cast<unsigned>(211 + m * 131 + i));
    std::normal_distribution<float> nd(0.0F, 2.0F);
    std::vector<float> input_f32(static_cast<size_t>(m * 2 * i));
    for (auto& value : input_f32) value = nd(rng);
    std::vector<uint16_t> input_bf16;
    const void* input_host = input_f32.data();
    size_t input_bytes = input_f32.size() * sizeof(float);
    if (dtype == DType::kBF16) {
      input_bf16.resize(input_f32.size());
      std::transform(input_f32.begin(), input_f32.end(), input_bf16.begin(),
                     FloatToBf16);
      input_host = input_bf16.data();
      input_bytes = input_bf16.size() * sizeof(uint16_t);
    }

    void* dinput_allocation =
        b.Alloc(input_bytes + (misaligned_input ? size_t{2} : size_t{0}));
    void* dinput =
        misaligned_input
            ? static_cast<void*>(
                  static_cast<uint8_t*>(dinput_allocation) + 2)
            : dinput_allocation;
    b.Copy(gq, dinput, input_host, input_bytes);
    Tensor input = GpuTensor({m, 2 * i});
    input.data = dinput;
    input.dtype = dtype;
    input.device = Gpu();

    void* dact = b.Alloc(static_cast<size_t>(m * i) * sizeof(uint16_t));
    void* dref_packed = b.Alloc(static_cast<size_t>(m * i / 2));
    void* dref_scale = b.Alloc(static_cast<size_t>(m * i / 16));
    Tensor act = GpuTensor({m, i});
    act.data = dact;
    act.dtype = DType::kBF16;
    act.device = Gpu();
    Tensor ref_packed = GpuTensor({m, i / 2});
    ref_packed.data = dref_packed;
    ref_packed.dtype = DType::kI8;
    ref_packed.device = Gpu();
    Tensor ref_scale = GpuTensor({m, i / 16});
    ref_scale.data = dref_scale;
    ref_scale.dtype = DType::kI8;
    ref_scale.device = Gpu();
    vt::SiluAndMul(gq, act, input);
    vt::ScaledFp4Quant(gq, ref_packed, ref_scale, act,
                       input_global_scale);

    void* dfused_packed = b.Alloc(static_cast<size_t>(m * i / 2));
    void* dfused_scale = b.Alloc(static_cast<size_t>(m * i / 16));
    Tensor fused_packed = GpuTensor({m, i / 2});
    fused_packed.data = dfused_packed;
    fused_packed.dtype = DType::kI8;
    fused_packed.device = Gpu();
    Tensor fused_scale = GpuTensor({m, i / 16});
    fused_scale.data = dfused_scale;
    fused_scale.dtype = DType::kI8;
    fused_scale.device = Gpu();
    vt::SiluAndMulFp4Quant(gq, fused_packed, fused_scale, input,
                           input_global_scale);

    const int64_t padded_rows = RoundUpTo(m, 128);
    const int64_t padded_cols = RoundUpTo(i / 16, 4);
    void* ddirect_packed = b.Alloc(static_cast<size_t>(m * i / 2));
    void* ddirect_scale =
        b.Alloc(static_cast<size_t>(padded_rows * padded_cols));
    b.Memset(gq, ddirect_scale, 0xA5,
             static_cast<size_t>(padded_rows * padded_cols));
    Tensor direct_packed = GpuTensor({m, i / 2});
    direct_packed.data = ddirect_packed;
    direct_packed.dtype = DType::kI8;
    direct_packed.device = Gpu();
    Tensor direct_scale = GpuTensor({padded_rows, padded_cols});
    direct_scale.data = ddirect_scale;
    direct_scale.dtype = DType::kI8;
    direct_scale.device = Gpu();
    vt::SiluAndMulFp4Quant(gq, direct_packed, direct_scale, input,
                           input_global_scale,
                           vt::Fp4ScaleLayout::kCutlassSwizzled);

    std::vector<uint8_t> reference_packed(static_cast<size_t>(m * i / 2));
    std::vector<uint8_t> reference_scale(static_cast<size_t>(m * i / 16));
    std::vector<uint8_t> fused_packed_host(reference_packed.size());
    std::vector<uint8_t> fused_scale_host(reference_scale.size());
    std::vector<uint8_t> direct_packed_host(reference_packed.size());
    std::vector<uint8_t> direct_scale_host(
        static_cast<size_t>(padded_rows * padded_cols));
    b.Copy(gq, reference_packed.data(), dref_packed,
           reference_packed.size());
    b.Copy(gq, reference_scale.data(), dref_scale, reference_scale.size());
    b.Copy(gq, fused_packed_host.data(), dfused_packed,
           fused_packed_host.size());
    b.Copy(gq, fused_scale_host.data(), dfused_scale,
           fused_scale_host.size());
    b.Copy(gq, direct_packed_host.data(), ddirect_packed,
           direct_packed_host.size());
    b.Copy(gq, direct_scale_host.data(), ddirect_scale,
           direct_scale_host.size());
    b.Synchronize(gq);

    CHECK(fused_packed_host == reference_packed);
    CHECK(fused_scale_host == reference_scale);
    CHECK(direct_packed_host == fused_packed_host);
    CHECK(direct_scale_host ==
          SwizzleScaleReference(fused_scale_host, m, i / 16));

    // Port the upstream functionalized two-output contract through our CUDA
    // graph API. The first eager call above also initializes the packed
    // candidate's cached occupancy before capture. Poisoning before each replay
    // proves that the captured operation owns every padded scale byte rather
    // than relying on allocator contents or one-time initialization.
    if (dtype == DType::kBF16 && m == 1 && i == 64 && !misaligned_input) {
      b.Memset(gq, ddirect_scale, 0xA5,
               static_cast<size_t>(padded_rows * padded_cols));
      b.Synchronize(gq);
      b.BeginCapture(gq);
      vt::SiluAndMulFp4Quant(gq, direct_packed, direct_scale, input,
                             input_global_scale,
                             vt::Fp4ScaleLayout::kCutlassSwizzled);
      void* graph = b.EndCaptureGraph(gq);
      for (int replay = 0; replay < 2; ++replay) {
        b.Memset(gq, ddirect_scale, 0xA5,
                 static_cast<size_t>(padded_rows * padded_cols));
        b.ReplayGraph(gq, graph);
        b.Copy(gq, direct_packed_host.data(), ddirect_packed,
               direct_packed_host.size());
        b.Copy(gq, direct_scale_host.data(), ddirect_scale,
               direct_scale_host.size());
        b.Synchronize(gq);
        CHECK(direct_packed_host == fused_packed_host);
        CHECK(direct_scale_host ==
              SwizzleScaleReference(fused_scale_host, m, i / 16));
      }
      b.DestroyGraph(graph);
    }

    for (void* pointer : {dinput_allocation, dact, dref_packed, dref_scale,
                          dfused_packed, dfused_scale, ddirect_packed,
                          ddirect_scale}) {
      b.Free(pointer);
    }
  };

  for (DType dtype : {DType::kF32, DType::kBF16}) {
    run_check(1, 64, dtype);       // decode
    run_check(128, 128, dtype);    // upstream shape
    run_check(37, 2048, dtype);    // padded-M prefill
    run_check(9, 17408, dtype);    // exact 27B intermediate class
  }
  for (int64_t m : {2, 4, 8, 16, 32, 48}) {
    run_check(m, 128, DType::kBF16);  // exact decode-graph row classes
  }
  run_check(1, 64, DType::kBF16, true);  // packed path must fall back safely
  b.DestroyQueue(gq);
}

// --- NUMERICS-NEUTRAL vectorized-load fast-path bit-identity ----------------
// The VT_FP4_QUANT_FAST / VT_SILU_FP4_FAST fast kernels change ONLY the memory
// access pattern (one 16-byte vectorized global load per 16-element group vs 16
// scalar loads; upstream vLLM nvfp4_quant_kernels.cu:56-80 ld256/ld128 pattern)
// and keep the exact per-element math. Every fp4 nibble AND per-group fp8 scale
// byte MUST equal the shipped scalar kernel on adversarial inputs — proved here
// by running the SAME op with the flag ROLLED BACK (=0, scalar) then DEFAULT
// (fast) and asserting byte-exact packed + scale. Covers both dtypes (f32/bf16),
// both scale
// layouts (linear + swizzled, incl. padded rows/cols), decode + prefill shapes,
// and a deliberately misaligned input that forces the fast kernel's scalar
// fallback branch.
namespace {
// Adversarial fp4/fp8 content: random-normal background plus injected boundary
// probes (exact +/-0, the +/-6.0 fp4 clamp edge and just past it, signed tiny
// values that round to fp4 zero, magnitudes that clamp the group scale to 448),
// and a forced all-zero leading group (amax==0 -> out_scale==0 branch).
void FillAdversarialFp4(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0F, 3.0F);
  for (auto& x : v) x = nd(rng);
  const std::array<float, 12> probes = {0.0F,   -0.0F,   6.0F,    -6.0F,
                                        6.5F,   -7.25F,  1e-4F,   -1e-4F,
                                        448.0F, -600.0F, 3.0F,    -1.5F};
  for (size_t i = 0; i + probes.size() <= v.size(); i += 97) {
    for (size_t j = 0; j < probes.size(); ++j) v[i + j] = probes[j];
  }
  for (size_t j = 0; j < std::min<size_t>(16, v.size()); ++j) v[j] = 0.0F;
}

// Scoped setenv/restore so the per-call getenv in the launcher selects the fast
// or scalar kernel deterministically and the environment is restored after.
struct ScopedEnv {
  const char* name;
  std::string saved;
  bool had;
  ScopedEnv(const char* n, const char* value) : name(n) {
    const char* cur = std::getenv(n);
    had = cur != nullptr;
    if (had) saved = cur;
    if (value)
      setenv(n, value, 1);
    else
      unsetenv(n);
  }
  ~ScopedEnv() {
    if (had)
      setenv(name, saved.c_str(), 1);
    else
      unsetenv(name);
  }
};

int64_t EnvInt(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  return v ? std::strtoll(v, nullptr, 10) : fallback;
}
}  // namespace

TEST_CASE("scaled_fp4_quant CUDA fast vectorized-load == scalar (BYTE-EXACT)") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float input_global_scale = 8.2F;

  auto run = [&](int64_t m, int64_t k, DType dtype, vt::Fp4ScaleLayout layout,
                 bool misaligned) {
    CAPTURE(m);
    CAPTURE(k);
    CAPTURE(static_cast<int>(dtype));
    CAPTURE(static_cast<int>(layout));
    CAPTURE(misaligned);
    std::vector<float> in_f32(static_cast<size_t>(m * k));
    FillAdversarialFp4(in_f32, static_cast<unsigned>(17 + m * 7 + k));
    std::vector<uint16_t> in_bf16;
    const void* host = in_f32.data();
    size_t bytes = in_f32.size() * sizeof(float);
    if (dtype == DType::kBF16) {
      in_bf16.resize(in_f32.size());
      std::transform(in_f32.begin(), in_f32.end(), in_bf16.begin(),
                     FloatToBf16);
      host = in_bf16.data();
      bytes = in_bf16.size() * sizeof(uint16_t);
    }
    // A 2-byte-offset base is bf16-load-legal but never 16-byte aligned, so the
    // fast kernel takes its scalar fallback (still byte-exact).
    void* dx_alloc = b.Alloc(bytes + (misaligned ? size_t{2} : size_t{0}));
    void* dx = misaligned
                   ? static_cast<void*>(static_cast<uint8_t*>(dx_alloc) + 2)
                   : dx_alloc;
    b.Copy(gq, dx, host, bytes);
    Tensor tx = GpuTensor({m, k});
    tx.data = dx;
    tx.dtype = dtype;
    tx.device = Gpu();

    const bool swizzled = layout == vt::Fp4ScaleLayout::kCutlassSwizzled;
    const int64_t srows = swizzled ? RoundUpTo(m, 128) : m;
    const int64_t scols = swizzled ? RoundUpTo(k / 16, 4) : k / 16;
    auto quant = [&](const char* flag) {
      void* dp = b.Alloc(static_cast<size_t>(m * k / 2));
      void* ds = b.Alloc(static_cast<size_t>(srows * scols));
      b.Memset(gq, dp, 0x5A, static_cast<size_t>(m * k / 2));
      b.Memset(gq, ds, 0xA5, static_cast<size_t>(srows * scols));
      Tensor tp = GpuTensor({m, k / 2});
      tp.data = dp;
      tp.dtype = DType::kI8;
      tp.device = Gpu();
      Tensor ts = GpuTensor({srows, scols});
      ts.data = ds;
      ts.dtype = DType::kI8;
      ts.device = Gpu();
      {
        ScopedEnv env("VT_FP4_QUANT_FAST", flag);
        vt::ScaledFp4Quant(gq, tp, ts, tx, input_global_scale, layout);
      }
      std::vector<uint8_t> hp(static_cast<size_t>(m * k / 2));
      std::vector<uint8_t> hs(static_cast<size_t>(srows * scols));
      b.Copy(gq, hp.data(), dp, hp.size());
      b.Copy(gq, hs.data(), ds, hs.size());
      b.Synchronize(gq);
      b.Free(dp);
      b.Free(ds);
      return std::make_pair(hp, hs);
    };
    const auto scalar = quant("0");      // rollback (shipped scalar kernel)
    const auto fast = quant(nullptr);    // default ON (vectorized-load kernel)
    CHECK(fast.first == scalar.first);    // fp4 nibbles byte-exact
    CHECK(fast.second == scalar.second);  // fp8 group scales byte-exact
    b.Free(dx_alloc);
  };

  for (DType dtype : {DType::kF32, DType::kBF16}) {
    for (vt::Fp4ScaleLayout layout : {vt::Fp4ScaleLayout::kLinear,
                                      vt::Fp4ScaleLayout::kCutlassSwizzled}) {
      run(1, 128, dtype, layout, false);    // decode m=1
      run(1, 256, dtype, layout, false);
      run(5, 128, dtype, layout, false);    // several rows + swizzled padding
      run(40, 512, dtype, layout, false);   // larger prefill class
    }
  }
  // Force the fast kernel's scalar fallback branch (unaligned bf16 base).
  run(3, 128, DType::kBF16, vt::Fp4ScaleLayout::kCutlassSwizzled, true);
  b.DestroyQueue(gq);
}

TEST_CASE("silu_and_mul_fp4_quant CUDA fast vectorized-load == scalar (BYTE-EXACT)") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float input_global_scale = 6.7F;

  auto run = [&](int64_t m, int64_t i, DType dtype, vt::Fp4ScaleLayout layout,
                 bool misaligned) {
    CAPTURE(m);
    CAPTURE(i);
    CAPTURE(static_cast<int>(dtype));
    CAPTURE(static_cast<int>(layout));
    CAPTURE(misaligned);
    std::vector<float> in_f32(static_cast<size_t>(m * 2 * i));
    FillAdversarialFp4(in_f32, static_cast<unsigned>(211 + m * 131 + i));
    std::vector<uint16_t> in_bf16;
    const void* host = in_f32.data();
    size_t bytes = in_f32.size() * sizeof(float);
    if (dtype == DType::kBF16) {
      in_bf16.resize(in_f32.size());
      std::transform(in_f32.begin(), in_f32.end(), in_bf16.begin(),
                     FloatToBf16);
      host = in_bf16.data();
      bytes = in_bf16.size() * sizeof(uint16_t);
    }
    void* dx_alloc = b.Alloc(bytes + (misaligned ? size_t{2} : size_t{0}));
    void* dx = misaligned
                   ? static_cast<void*>(static_cast<uint8_t*>(dx_alloc) + 2)
                   : dx_alloc;
    b.Copy(gq, dx, host, bytes);
    Tensor tx = GpuTensor({m, 2 * i});
    tx.data = dx;
    tx.dtype = dtype;
    tx.device = Gpu();

    const bool swizzled = layout == vt::Fp4ScaleLayout::kCutlassSwizzled;
    const int64_t srows = swizzled ? RoundUpTo(m, 128) : m;
    const int64_t scols = swizzled ? RoundUpTo(i / 16, 4) : i / 16;
    auto quant = [&](const char* flag) {
      void* dp = b.Alloc(static_cast<size_t>(m * i / 2));
      void* ds = b.Alloc(static_cast<size_t>(srows * scols));
      b.Memset(gq, dp, 0x5A, static_cast<size_t>(m * i / 2));
      b.Memset(gq, ds, 0xA5, static_cast<size_t>(srows * scols));
      Tensor tp = GpuTensor({m, i / 2});
      tp.data = dp;
      tp.dtype = DType::kI8;
      tp.device = Gpu();
      Tensor ts = GpuTensor({srows, scols});
      ts.data = ds;
      ts.dtype = DType::kI8;
      ts.device = Gpu();
      {
        ScopedEnv env("VT_SILU_FP4_FAST", flag);
        vt::SiluAndMulFp4Quant(gq, tp, ts, tx, input_global_scale, layout);
      }
      std::vector<uint8_t> hp(static_cast<size_t>(m * i / 2));
      std::vector<uint8_t> hs(static_cast<size_t>(srows * scols));
      b.Copy(gq, hp.data(), dp, hp.size());
      b.Copy(gq, hs.data(), ds, hs.size());
      b.Synchronize(gq);
      b.Free(dp);
      b.Free(ds);
      return std::make_pair(hp, hs);
    };
    const auto scalar = quant("0");      // rollback (shipped scalar kernel)
    const auto fast = quant(nullptr);    // default ON (vectorized-load kernel)
    CHECK(fast.first == scalar.first);    // fp4 nibbles byte-exact
    CHECK(fast.second == scalar.second);  // fp8 group scales byte-exact
    b.Free(dx_alloc);
  };

  for (DType dtype : {DType::kF32, DType::kBF16}) {
    for (vt::Fp4ScaleLayout layout : {vt::Fp4ScaleLayout::kLinear,
                                      vt::Fp4ScaleLayout::kCutlassSwizzled}) {
      run(1, 64, dtype, layout, false);     // decode
      run(4, 128, dtype, layout, false);    // several rows + swizzled padding
      run(37, 512, dtype, layout, false);   // padded-M prefill class
    }
  }
  run(3, 64, DType::kBF16, vt::Fp4ScaleLayout::kCutlassSwizzled, true);
  b.DestroyQueue(gq);
}

// Env-gated ISOLATED per-launch microbench for the two vectorized-load fast
// kernels at the real 27B swizzled decode shapes (hidden 5120, intermediate
// 17408; M = decode batch). Runs a warmup + timed loop of the scalar kernel then
// the fast kernel at ONE fixed (kernel, M, dim) so an nsys `cuda_gpu_kern_sum`
// capture reports the pure-kernel average duration for each distinct kernel name
// (…FastKernel vs its scalar sibling). Off by default (needs VT_FP4_MB_RUN=1);
// select with VT_FP4_MB_KERNEL=scaled|silu, VT_FP4_MB_M, VT_FP4_MB_K (scaled) /
// VT_FP4_MB_I (silu), VT_FP4_MB_ITERS.
TEST_CASE("fp4 quant fast microbench (VT_FP4_MB_RUN)") {
  if (!HasCuda()) return;
  if (std::getenv("VT_FP4_MB_RUN") == nullptr) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();
  const float igs = 8.2F;
  const int iters = static_cast<int>(EnvInt("VT_FP4_MB_ITERS", 500));
  const int64_t m = EnvInt("VT_FP4_MB_M", 16);
  const char* kernel = std::getenv("VT_FP4_MB_KERNEL");
  const bool silu = kernel != nullptr && std::string(kernel) == "silu";
  const auto layout = vt::Fp4ScaleLayout::kCutlassSwizzled;

  // Input width: scaled = K; silu = 2*I (gate then up), fp4 output width I.
  const int64_t inner = silu ? EnvInt("VT_FP4_MB_I", 17408)
                             : EnvInt("VT_FP4_MB_K", 5120);
  const int64_t in_cols = silu ? 2 * inner : inner;

  std::vector<float> in_f32(static_cast<size_t>(m * in_cols));
  FillAdversarialFp4(in_f32, 909);
  std::vector<uint16_t> in_bf16(in_f32.size());
  std::transform(in_f32.begin(), in_f32.end(), in_bf16.begin(), FloatToBf16);
  void* dx = b.Alloc(in_bf16.size() * sizeof(uint16_t));
  b.Copy(gq, dx, in_bf16.data(), in_bf16.size() * sizeof(uint16_t));
  Tensor tx = GpuTensor({m, in_cols});
  tx.data = dx;
  tx.dtype = DType::kBF16;
  tx.device = Gpu();

  const int64_t srows = RoundUpTo(m, 128);
  const int64_t scols = RoundUpTo(inner / 16, 4);
  void* dp = b.Alloc(static_cast<size_t>(m * inner / 2));
  void* ds = b.Alloc(static_cast<size_t>(srows * scols));
  Tensor tp = GpuTensor({m, inner / 2});
  tp.data = dp;
  tp.dtype = DType::kI8;
  tp.device = Gpu();
  Tensor ts = GpuTensor({srows, scols});
  ts.data = ds;
  ts.dtype = DType::kI8;
  ts.device = Gpu();

  const char* flag_name = silu ? "VT_SILU_FP4_FAST" : "VT_FP4_QUANT_FAST";
  auto loop = [&](const char* flag) {
    ScopedEnv env(flag_name, flag);
    for (int w = 0; w < 30; ++w) {
      if (silu)
        vt::SiluAndMulFp4Quant(gq, tp, ts, tx, igs, layout);
      else
        vt::ScaledFp4Quant(gq, tp, ts, tx, igs, layout);
    }
    b.Synchronize(gq);
    for (int it = 0; it < iters; ++it) {
      if (silu)
        vt::SiluAndMulFp4Quant(gq, tp, ts, tx, igs, layout);
      else
        vt::ScaledFp4Quant(gq, tp, ts, tx, igs, layout);
    }
    b.Synchronize(gq);
  };
  MESSAGE("microbench kernel=" << (silu ? "silu" : "scaled") << " M=" << m
                               << " inner=" << inner << " iters=" << iters);
  loop("0");      // scalar sibling (rollback)
  loop(nullptr);  // vectorized-load fast kernel (default ON)
  CHECK(true);
  b.Free(dx);
  b.Free(dp);
  b.Free(ds);
  b.DestroyQueue(gq);
}

TEST_CASE("matmul_nvfp4_fp4 validates shapes loudly") {
  std::vector<uint8_t> buf(64, 0);
  std::vector<float> ob(16, 0);
  Tensor ap = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {2, 8});   // K=16
  Tensor as = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {2, 1});
  Tensor bp = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {3, 4});   // K=8 mismatch
  Tensor bs = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {3, 1});
  Tensor o = Tensor::Contiguous(ob.data(), DType::kF32, Cpu(), {2, 3});
  Queue q = CpuQueue();
  CHECK_THROWS_AS(vt::MatmulNvfp4Fp4(q, o, ap, as, bp, bs, 1.0F), std::runtime_error);
}

// Port of vLLM's stable NVFP4 op input contract and
// tests/kernels/quantization/test_nvfp4_scaled_mm.py device-alpha fixture. The
// scalar is validated before backend dispatch, so these malformed cases remain
// CPU-runnable even though the CUTLASS operation itself is CUDA-only.
TEST_CASE("matmul_nvfp4_cutlass validates device alpha loudly") {
  std::vector<uint8_t> packed(32 * 16, 0);
  std::vector<uint8_t> scale(128 * 4, 0);
  std::vector<uint16_t> output(32, 0);
  std::array<float, 2> alpha_values = {0.125F, 0.25F};
  Queue queue = CpuQueue();
  Tensor a = Tensor::Contiguous(packed.data(), DType::kI8, Cpu(), {1, 16});
  Tensor b = Tensor::Contiguous(packed.data(), DType::kI8, Cpu(), {32, 16});
  Tensor a_scale =
      Tensor::Contiguous(scale.data(), DType::kI8, Cpu(), {128, 4});
  Tensor b_scale =
      Tensor::Contiguous(scale.data(), DType::kI8, Cpu(), {128, 4});
  Tensor out =
      Tensor::Contiguous(output.data(), DType::kBF16, Cpu(), {1, 32});
  Tensor alpha = Tensor::Contiguous(alpha_values.data(), DType::kF32, Cpu(), {1});
  const auto check_alpha_error = [&](const Tensor& candidate,
                                     const char* expected) {
    try {
      vt::MatmulNvfp4Cutlass(queue, out, a, a_scale, b, b_scale,
                             candidate);
      FAIL_CHECK("invalid alpha reached backend dispatch");
    } catch (const std::runtime_error& error) {
      CHECK(std::string(error.what()).find(expected) != std::string::npos);
    }
  };

  Tensor wrong_rank = alpha;
  wrong_rank.rank = 2;
  wrong_rank.shape[0] = 1;
  wrong_rank.shape[1] = 1;
  wrong_rank.stride[0] = 1;
  wrong_rank.stride[1] = 1;
  check_alpha_error(
      wrong_rank,
      "matmul_nvfp4_cutlass: alpha must be a rank-0 or rank-1 scalar tensor");

  Tensor multiple =
      Tensor::Contiguous(alpha_values.data(), DType::kF32, Cpu(), {2});
  check_alpha_error(multiple,
                    "matmul_nvfp4_cutlass: alpha must contain exactly one element");

  Tensor wrong_dtype = alpha;
  wrong_dtype.dtype = DType::kBF16;
  check_alpha_error(wrong_dtype, "matmul_nvfp4_cutlass: alpha must be f32");

  Tensor null_alpha = alpha;
  null_alpha.data = nullptr;
  check_alpha_error(null_alpha,
                    "matmul_nvfp4_cutlass: alpha must have non-null storage");

  Tensor noncontiguous = alpha;
  noncontiguous.stride[0] = 2;
  check_alpha_error(noncontiguous,
                    "matmul_nvfp4_cutlass: alpha must be contiguous");

  Tensor cross_device = alpha;
  cross_device.device = Device{DeviceType::kCUDA, 0};
  check_alpha_error(cross_device,
                    "matmul_nvfp4_cutlass: alpha device mismatch");
}

#ifdef VT_CUTLASS_NVFP4
namespace {
float Bf16ToFloat(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
int64_t RoundUp(int64_t x, int64_t y) { return (x + y - 1) / y * y; }
}  // namespace

// The cutlass sm120a fp4xfp4 GEMM (MatmulNvfp4Cutlass, with SwizzleBlockscale on
// both scale streams) must reproduce the emulation truth (MatmulNvfp4Fp4 with
// LINEAR scales, itself proven == vllm::RunNvfp4Emulation) within GEMM tol. This
// is the load-bearing drop-in validation: the swizzle layout + the lifted cutlass
// kernel + the bf16 epilogue all agree with the reference.
TEST_CASE("matmul_nvfp4_cutlass (swizzled) == emulation reference CUDA") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();

  const int64_t M = 96, K = 512, N = 256;  // K,N % 32 == 0
  std::mt19937 rng(202);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);

  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& v : w_packed) v = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& v : w_scale) v = vllm::F32ToF8E4M3(scale_d(rng));
  const float input_global_scale = 8.2F, weight_global_scale = 5.5F;
  const float alpha = (1.0F / input_global_scale) * (1.0F / weight_global_scale);

  // Emulation reference (CPU, f32) — the truth.
  std::vector<uint8_t> cpu_ap(static_cast<size_t>(M * K / 2), 0), cpu_as(static_cast<size_t>(M * K / 16), 0);
  std::vector<float> cpu_out(static_cast<size_t>(M * N), 0);
  {
    Queue cq = CpuQueue();
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
    Tensor tp = Tensor::Contiguous(cpu_ap.data(), DType::kI8, Cpu(), {M, K / 2});
    Tensor ts = Tensor::Contiguous(cpu_as.data(), DType::kI8, Cpu(), {M, K / 16});
    vt::ScaledFp4Quant(cq, tp, ts, tx, input_global_scale);
    Tensor tbp = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {N, K / 2});
    Tensor tbs = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {N, K / 16});
    Tensor to = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {M, N});
    vt::MatmulNvfp4Fp4(cq, to, tp, ts, tbp, tbs, alpha);
  }

  const int64_t Mp = RoundUp(M, 128), Np = RoundUp(N, 128), Kp = RoundUp(K / 16, 4);
  auto up = [&](const void* h, size_t nb) { void* p = b.Alloc(nb); b.Copy(gq, p, h, nb); return p; };
  void* dx = up(x.data(), x.size() * sizeof(float));
  void* dbp = up(w_packed.data(), w_packed.size());
  void* dbs = up(w_scale.data(), w_scale.size());
  void* dalpha = up(&alpha, sizeof(alpha));
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* dap_direct = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dasw = b.Alloc(static_cast<size_t>(Mp * Kp));
  void* das_direct = b.Alloc(static_cast<size_t>(Mp * Kp));
  void* dbsw = b.Alloc(static_cast<size_t>(Np * Kp));
  void* dout = b.Alloc(static_cast<size_t>(M * N) * sizeof(uint16_t));  // bf16
  void* dout_direct =
      b.Alloc(static_cast<size_t>(M * N) * sizeof(uint16_t));
  void* dout_device_alpha =
      b.Alloc(static_cast<size_t>(M * N) * sizeof(uint16_t));
  void* dout_scalar_alpha =
      b.Alloc(static_cast<size_t>(M * N) * sizeof(uint16_t));
  b.Memset(gq, dasw, 0, static_cast<size_t>(Mp * Kp));
  b.Memset(gq, dbsw, 0, static_cast<size_t>(Np * Kp));

  Tensor tx = GpuTensor({M, K}); tx.data = dx; tx.dtype = DType::kF32; tx.device = Gpu();
  Tensor tap = GpuTensor({M, K / 2}); tap.data = dap; tap.dtype = DType::kI8; tap.device = Gpu();
  Tensor tap_direct = GpuTensor({M, K / 2}); tap_direct.data = dap_direct; tap_direct.dtype = DType::kI8; tap_direct.device = Gpu();
  Tensor tas = GpuTensor({M, K / 16}); tas.data = das; tas.dtype = DType::kI8; tas.device = Gpu();
  Tensor tasw = GpuTensor({Mp, Kp}); tasw.data = dasw; tasw.dtype = DType::kI8; tasw.device = Gpu();
  Tensor tas_direct = GpuTensor({Mp, Kp}); tas_direct.data = das_direct; tas_direct.dtype = DType::kI8; tas_direct.device = Gpu();
  Tensor tbp = GpuTensor({N, K / 2}); tbp.data = dbp; tbp.dtype = DType::kI8; tbp.device = Gpu();
  Tensor tbs = GpuTensor({N, K / 16}); tbs.data = dbs; tbs.dtype = DType::kI8; tbs.device = Gpu();
  Tensor tbsw = GpuTensor({Np, Kp}); tbsw.data = dbsw; tbsw.dtype = DType::kI8; tbsw.device = Gpu();
  Tensor to = GpuTensor({M, N}); to.data = dout; to.dtype = DType::kBF16; to.device = Gpu();
  Tensor to_direct = GpuTensor({M, N}); to_direct.data = dout_direct; to_direct.dtype = DType::kBF16; to_direct.device = Gpu();
  Tensor to_device_alpha = GpuTensor({M, N}); to_device_alpha.data = dout_device_alpha; to_device_alpha.dtype = DType::kBF16; to_device_alpha.device = Gpu();
  Tensor to_scalar_alpha = GpuTensor({M, N}); to_scalar_alpha.data = dout_scalar_alpha; to_scalar_alpha.dtype = DType::kBF16; to_scalar_alpha.device = Gpu();
  Tensor talpha = GpuTensor({1}); talpha.data = dalpha; talpha.dtype = DType::kF32; talpha.device = Gpu();
  Tensor talpha_scalar = talpha;
  talpha_scalar.rank = 0;

  vt::ScaledFp4Quant(gq, tap, tas, tx, input_global_scale);
  vt::ScaledFp4Quant(gq, tap_direct, tas_direct, tx, input_global_scale,
                     vt::Fp4ScaleLayout::kCutlassSwizzled);
  vt::SwizzleBlockscale(gq, tasw, tas);
  vt::SwizzleBlockscale(gq, tbsw, tbs);
  vt::MatmulNvfp4Cutlass(gq, to, tap, tasw, tbp, tbsw, alpha);
  vt::MatmulNvfp4Cutlass(gq, to_direct, tap_direct, tas_direct, tbp, tbsw,
                         alpha);
  vt::MatmulNvfp4Cutlass(gq, to_device_alpha, tap, tasw, tbp, tbsw,
                         talpha);
  vt::MatmulNvfp4Cutlass(gq, to_scalar_alpha, tap, tasw, tbp, tbsw,
                         talpha_scalar);

  std::vector<uint16_t> g_out(static_cast<size_t>(M * N));
  std::vector<uint16_t> direct_out(g_out.size());
  std::vector<uint16_t> device_alpha_out(g_out.size());
  std::vector<uint16_t> scalar_alpha_out(g_out.size());
  std::vector<uint8_t> composed_packed(static_cast<size_t>(M * K / 2));
  std::vector<uint8_t> direct_packed(composed_packed.size());
  std::vector<uint8_t> composed_scale(static_cast<size_t>(Mp * Kp));
  std::vector<uint8_t> direct_scale(composed_scale.size());
  b.Copy(gq, g_out.data(), dout, g_out.size() * sizeof(uint16_t));
  b.Copy(gq, direct_out.data(), dout_direct,
         direct_out.size() * sizeof(uint16_t));
  b.Copy(gq, device_alpha_out.data(), dout_device_alpha,
         device_alpha_out.size() * sizeof(uint16_t));
  b.Copy(gq, scalar_alpha_out.data(), dout_scalar_alpha,
         scalar_alpha_out.size() * sizeof(uint16_t));
  b.Copy(gq, composed_packed.data(), dap, composed_packed.size());
  b.Copy(gq, direct_packed.data(), dap_direct, direct_packed.size());
  b.Copy(gq, composed_scale.data(), dasw, composed_scale.size());
  b.Copy(gq, direct_scale.data(), das_direct, direct_scale.size());
  b.Synchronize(gq);
  CHECK(direct_packed == composed_packed);
  CHECK(direct_scale == composed_scale);
  CHECK(direct_out == g_out);
  CHECK(device_alpha_out == g_out);
  CHECK(scalar_alpha_out == g_out);

  double max_abs = 0.0, ref_absmax = 0.0;
  for (size_t i = 0; i < g_out.size(); ++i) {
    const float got = Bf16ToFloat(g_out[i]);
    max_abs = std::max(max_abs, std::abs(static_cast<double>(got) - cpu_out[i]));
    ref_absmax = std::max(ref_absmax, std::abs(static_cast<double>(cpu_out[i])));
  }
  // bf16 output (8-bit mantissa) vs f32 emulation: relative tol ~ bf16 ULP + the
  // fp4-MMA vs emulation reduction-order delta. Assert small relative error.
  MESSAGE("cutlass max_abs=" << max_abs << " ref_absmax=" << ref_absmax);
  CHECK(max_abs <= 0.03 * ref_absmax + 1e-3);

  // A warmed plan must be a pure cache hit during CUDA graph capture. Replay
  // the same fixed-pointer GEMM and require byte-identical bf16 output.
  b.Memset(gq, dout, 0, g_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  b.BeginCapture(gq);
  vt::MatmulNvfp4Cutlass(gq, to, tap, tasw, tbp, tbsw, alpha);
  void* ready_graph = b.EndCaptureGraph(gq);
  REQUIRE(ready_graph != nullptr);
  b.ReplayGraph(gq, ready_graph);
  b.Synchronize(gq);
  std::vector<uint16_t> replay_out(g_out.size());
  b.Copy(gq, replay_out.data(), dout, replay_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  CHECK(replay_out == g_out);
  b.DestroyGraph(ready_graph);

  // The upstream-shaped resident alpha pointer also remains fixed and valid
  // across capture/replay, without a scalar-staging kernel in the graph.
  b.Memset(gq, dout_device_alpha, 0,
           device_alpha_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  b.BeginCapture(gq);
  vt::MatmulNvfp4Cutlass(gq, to_device_alpha, tap, tasw, tbp, tbsw,
                         talpha);
  void* device_alpha_graph = b.EndCaptureGraph(gq);
  REQUIRE(device_alpha_graph != nullptr);
  b.ReplayGraph(gq, device_alpha_graph);
  b.Synchronize(gq);
  std::vector<uint16_t> device_alpha_replay(g_out.size());
  b.Copy(gq, device_alpha_replay.data(), dout_device_alpha,
         device_alpha_replay.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  CHECK(device_alpha_replay == g_out);
  b.DestroyGraph(device_alpha_graph);

  // M=64 has a different plan key from the warmed M=96 (bucket 64 vs 128).
  // The call records its already-warmed scalar write, then rejects the missing
  // plan before creating tuner events or launching a GEMM. Ending the capture
  // must still produce a valid graph, proving the rejection itself did not
  // invalidate the stream. An eager retry must tune normally (no partial plan).
  Tensor tap_miss = tap;
  tap_miss.shape[0] = 64;
  Tensor to_miss = to;
  to_miss.shape[0] = 64;
  b.Memset(gq, dout, 0, g_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  b.BeginCapture(gq);
  CHECK_THROWS_WITH_AS(
      vt::MatmulNvfp4Cutlass(gq, to_miss, tap_miss, tasw, tbp, tbsw, alpha),
      "NVFP4 plan cache miss while tuning is disallowed", std::runtime_error);
  void* miss_graph = b.EndCaptureGraph(gq);
  REQUIRE(miss_graph != nullptr);
  b.DestroyGraph(miss_graph);

  vt::MatmulNvfp4Cutlass(gq, to_miss, tap_miss, tasw, tbp, tbsw, alpha);
  std::vector<uint16_t> retry_out(static_cast<size_t>(64 * N));
  b.Copy(gq, retry_out.data(), dout, retry_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  for (size_t i = 0; i < retry_out.size(); ++i) {
    const float got = Bf16ToFloat(retry_out[i]);
    CHECK(got == doctest::Approx(cpu_out[i]).epsilon(0.03).scale(1.0));
  }

  // Port of kernel_warmup.py::flashinfer_autotune plus FlashInfer's
  // get_hybrid_num_tokens_buckets profile generation. One maximum-M call under
  // the pre-serve scope materializes every missing bucket for this N/K. M=32
  // must therefore become a capture-safe ready hit before any real request.
  const vt::cuda::Nvfp4AutotuneWarmupStats stats_before =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  {
    vt::cuda::Nvfp4AutotuneWarmupScope warmup(static_cast<uint32_t>(M));
    vt::MatmulNvfp4Cutlass(gq, to, tap, tasw, tbp, tbsw, talpha);
    warmup.Complete();
  }
  const vt::cuda::Nvfp4AutotuneWarmupStats stats_after =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  CHECK(stats_after.scopes_started == stats_before.scopes_started + 1);
  CHECK(stats_after.scopes_completed == stats_before.scopes_completed + 1);
  CHECK(stats_after.profiles_requested > stats_before.profiles_requested);
  CHECK(stats_after.profiles_tuned - stats_before.profiles_tuned ==
        stats_after.profiles_requested - stats_before.profiles_requested);

  Tensor tap_prewarmed = tap;
  tap_prewarmed.shape[0] = 32;
  Tensor to_prewarmed = to;
  to_prewarmed.shape[0] = 32;
  b.Memset(gq, dout, 0, g_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  b.BeginCapture(gq);
  vt::MatmulNvfp4Cutlass(gq, to_prewarmed, tap_prewarmed, tasw, tbp, tbsw,
                         talpha);
  void* prewarmed_graph = b.EndCaptureGraph(gq);
  REQUIRE(prewarmed_graph != nullptr);
  b.ReplayGraph(gq, prewarmed_graph);
  b.Synchronize(gq);
  std::vector<uint16_t> prewarmed_out(static_cast<size_t>(32 * N));
  b.Copy(gq, prewarmed_out.data(), dout,
         prewarmed_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);
  for (size_t i = 0; i < prewarmed_out.size(); ++i) {
    const float got = Bf16ToFloat(prewarmed_out[i]);
    CHECK(got == doctest::Approx(cpu_out[i]).epsilon(0.03).scale(1.0));
  }
  b.DestroyGraph(prewarmed_graph);

  for (void* p : {dx, dbp, dbs, dalpha, dap, dap_direct, das, dasw,
                  das_direct, dbsw, dout, dout_direct, dout_device_alpha,
                  dout_scalar_alpha}) {
    b.Free(p);
  }
  b.DestroyQueue(gq);
}

// Ported from QKVParallelLinear's one-physical-weight contract
// (`linear.py:942-1050`) and the CT max-logical-shard scalar rule. Concatenating
// Q/K/V packed rows and scales, quantizing once and launching one GEMM must
// reproduce the same logical BF16 outputs as three views using the one merged
// alpha. The packed output is split with its real Q+K+V row stride.
TEST_CASE("matmul_nvfp4_cutlass packed QKV matches logical shard views CUDA") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue q = b.CreateQueue();
  const int64_t M = 16, K = 512;
  const std::array<int64_t, 3> ns = {256, 128, 128};
  const int64_t total_n = ns[0] + ns[1] + ns[2];
  const int64_t mp = RoundUp(M, 128), kp = RoundUp(K / 16, 4);

  std::mt19937 rng(12025);
  std::normal_distribution<float> activation(0.0F, 2.0F);
  std::uniform_int_distribution<int> packed_byte(0, 255);
  std::uniform_real_distribution<float> block_scale(0.05F, 4.0F);
  std::vector<float> x(static_cast<size_t>(M * K));
  for (float& value : x) value = activation(rng);
  std::vector<uint8_t> weights(static_cast<size_t>(total_n * K / 2));
  for (uint8_t& value : weights)
    value = static_cast<uint8_t>(packed_byte(rng));
  std::vector<uint8_t> scales(static_cast<size_t>(total_n * K / 16));
  for (uint8_t& value : scales)
    value = vllm::F32ToF8E4M3(block_scale(rng));

  vllm::Nvfp4Weight qw, kw, vw;
  qw.input_global_scale_inv = 72.0F;
  kw.input_global_scale_inv = 80.0F;
  vw.input_global_scale_inv = 96.0F;
  qw.weight_global_scale_inv = 256.0F;
  kw.weight_global_scale_inv = 512.0F;
  vw.weight_global_scale_inv = 384.0F;
  const vllm::FullAttnQkvGlobals globals =
      vllm::MergeFullAttnQkvGlobals(qw, kw, vw);

  const auto upload = [&](const void* host, size_t bytes) {
    void* device = b.Alloc(bytes);
    b.Copy(q, device, host, bytes);
    return device;
  };
  void* dx = upload(x.data(), x.size() * sizeof(float));
  void* dw = upload(weights.data(), weights.size());
  void* ds = upload(scales.data(), scales.size());
  void* dalpha = upload(&globals.alpha, sizeof(globals.alpha));
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dasw = b.Alloc(static_cast<size_t>(mp * kp));
  void* dsw = b.Alloc(static_cast<size_t>(RoundUp(total_n, 128) * kp));
  void* dout = b.Alloc(static_cast<size_t>(M * total_n) * sizeof(uint16_t));

  Tensor tx = GpuTensor({M, K});
  tx.data = dx;
  tx.dtype = DType::kF32;
  Tensor tap = GpuTensor({M, K / 2});
  tap.data = dap;
  tap.dtype = DType::kI8;
  Tensor tas = GpuTensor({M, K / 16});
  tas.data = das;
  tas.dtype = DType::kI8;
  Tensor tasw = GpuTensor({mp, kp});
  tasw.data = dasw;
  tasw.dtype = DType::kI8;
  Tensor tw = GpuTensor({total_n, K / 2});
  tw.data = dw;
  tw.dtype = DType::kI8;
  Tensor ts = GpuTensor({total_n, K / 16});
  ts.data = ds;
  ts.dtype = DType::kI8;
  Tensor tsw = GpuTensor({RoundUp(total_n, 128), kp});
  tsw.data = dsw;
  tsw.dtype = DType::kI8;
  Tensor tout = GpuTensor({M, total_n});
  tout.data = dout;
  tout.dtype = DType::kBF16;
  Tensor talpha = GpuTensor({1});
  talpha.data = dalpha;
  talpha.dtype = DType::kF32;
  vt::ScaledFp4Quant(q, tap, tas, tx, globals.input_global_scale_inv);
  vt::SwizzleBlockscale(q, tasw, tas);
  vt::SwizzleBlockscale(q, tsw, ts);
  vt::MatmulNvfp4Cutlass(q, tout, tap, tasw, tw, tsw, talpha);

  std::array<void*, 3> shard_scale_sw{};
  std::array<void*, 3> shard_out{};
  int64_t n_offset = 0;
  for (size_t shard = 0; shard < ns.size(); ++shard) {
    const int64_t n = ns[shard];
    shard_scale_sw[shard] =
        b.Alloc(static_cast<size_t>(RoundUp(n, 128) * kp));
    shard_out[shard] =
        b.Alloc(static_cast<size_t>(M * n) * sizeof(uint16_t));
    Tensor shard_w = GpuTensor({n, K / 2});
    shard_w.data = static_cast<uint8_t*>(dw) + n_offset * K / 2;
    shard_w.dtype = DType::kI8;
    Tensor shard_s = GpuTensor({n, K / 16});
    shard_s.data = static_cast<uint8_t*>(ds) + n_offset * K / 16;
    shard_s.dtype = DType::kI8;
    Tensor shard_sw = GpuTensor({RoundUp(n, 128), kp});
    shard_sw.data = shard_scale_sw[shard];
    shard_sw.dtype = DType::kI8;
    Tensor shard_o = GpuTensor({M, n});
    shard_o.data = shard_out[shard];
    shard_o.dtype = DType::kBF16;
    vt::SwizzleBlockscale(q, shard_sw, shard_s);
    vt::MatmulNvfp4Cutlass(q, shard_o, tap, tasw, shard_w, shard_sw,
                           talpha);
    n_offset += n;
  }

  std::vector<uint16_t> packed_out(static_cast<size_t>(M * total_n));
  b.Copy(q, packed_out.data(), dout, packed_out.size() * sizeof(uint16_t));
  std::array<std::vector<uint16_t>, 3> shard_outputs;
  for (size_t shard = 0; shard < ns.size(); ++shard) {
    shard_outputs[shard].resize(static_cast<size_t>(M * ns[shard]));
    b.Copy(q, shard_outputs[shard].data(), shard_out[shard],
           shard_outputs[shard].size() * sizeof(uint16_t));
  }
  b.Synchronize(q);

  n_offset = 0;
  double max_abs = 0.0;
  double ref_absmax = 0.0;
  for (size_t shard = 0; shard < ns.size(); ++shard) {
    const int64_t n = ns[shard];
    for (int64_t row = 0; row < M; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        const float got = Bf16ToFloat(
            packed_out[static_cast<size_t>(row * total_n + n_offset + col)]);
        const float ref = Bf16ToFloat(
            shard_outputs[shard][static_cast<size_t>(row * n + col)]);
        max_abs = std::max(max_abs,
                           std::abs(static_cast<double>(got) - ref));
        ref_absmax = std::max(ref_absmax, std::abs(static_cast<double>(ref)));
      }
    }
    n_offset += n;
  }
  MESSAGE("packed-QKV max_abs=" << max_abs << " ref_absmax=" << ref_absmax);
  CHECK(max_abs <= 0.03 * ref_absmax + 1e-3);

  for (void* pointer : {dx, dw, ds, dalpha, dap, das, dasw, dsw,
                        dout}) {
    b.Free(pointer);
  }
  for (void* pointer : shard_scale_sw) b.Free(pointer);
  for (void* pointer : shard_out) b.Free(pointer);
  b.DestroyQueue(q);
}

// Port of the forced-tactic coverage in
// vLLM `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py:26-168`
// plus FlashInfer's 32-entry getConfigs contract. CMake launches this case in
// 32 fresh processes so the process-stable VT_FP4_FORCE_TACTIC selects every ID.
TEST_CASE("matmul_nvfp4_cutlass every forced SM12 tactic matches reference and captures") {
  const char* forced_text = std::getenv("VT_FP4_FORCE_TACTIC");
  if (forced_text == nullptr || !HasCuda()) return;
  const int tactic = std::stoi(forced_text);
  REQUIRE(tactic >= 0);
  REQUIRE(tactic < 32);

  auto& backend = vt::GetBackend(DeviceType::kCUDA);
  Queue queue = backend.CreateQueue();
  const auto dimension = [](const char* name, int64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr) return fallback;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    REQUIRE(end != text);
    REQUIRE(*end == '\0');
    REQUIRE(value > 0);
    return static_cast<int64_t>(value);
  };
  const bool real_projection = std::getenv("VT_FP4_TEST_REAL_SHAPE") != nullptr;
  const bool direct_scale =
      std::getenv("VT_FP4_TEST_DIRECT_SF") != nullptr;
  const int64_t m = dimension("VT_FP4_TEST_M", 1 + tactic % 3);
  const int64_t n = dimension("VT_FP4_TEST_N", real_projection ? 1024 : 256);
  const int64_t k = dimension("VT_FP4_TEST_K", real_projection ? 5120 : 512);
  const unsigned seed = static_cast<unsigned>(
      dimension("VT_FP4_TEST_SEED", static_cast<int64_t>(9202 + tactic)));
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 2.0F);
  std::uniform_int_distribution<int> packed_value(0, 255);
  std::uniform_real_distribution<float> scale_value(0.05F, 4.0F);

  std::vector<float> x(static_cast<size_t>(m * k));
  for (float& value : x) value = normal(rng);
  const bool bf16_input = std::getenv("VT_FP4_TEST_BF16_INPUT") != nullptr;
  std::vector<uint16_t> x_bf16;
  if (bf16_input) {
    x_bf16.resize(x.size());
    for (size_t index = 0; index < x.size(); ++index) {
      x_bf16[index] = vt::F32ToBF16(x[index]);
    }
  }
  std::vector<uint8_t> w_packed(static_cast<size_t>(n * k / 2));
  for (uint8_t& value : w_packed) value = static_cast<uint8_t>(packed_value(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(n * k / 16));
  for (uint8_t& value : w_scale) value = vllm::F32ToF8E4M3(scale_value(rng));
  constexpr float kInputGlobalScale = 8.2F;
  constexpr float kWeightGlobalScale = 5.5F;
  constexpr float kAlpha = (1.0F / kInputGlobalScale) * (1.0F / kWeightGlobalScale);

  std::vector<uint8_t> cpu_a_packed(static_cast<size_t>(m * k / 2), 0);
  std::vector<uint8_t> cpu_a_scale(static_cast<size_t>(m * k / 16), 0);
  std::vector<float> reference(static_cast<size_t>(m * n), 0.0F);
  {
    Queue cpu_queue = CpuQueue();
    Tensor tx = bf16_input
                    ? Tensor::Contiguous(x_bf16.data(), DType::kBF16, Cpu(), {m, k})
                    : Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {m, k});
    Tensor ta = Tensor::Contiguous(cpu_a_packed.data(), DType::kI8, Cpu(), {m, k / 2});
    Tensor tas = Tensor::Contiguous(cpu_a_scale.data(), DType::kI8, Cpu(), {m, k / 16});
    Tensor tw = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {n, k / 2});
    Tensor tws = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {n, k / 16});
    Tensor tout = Tensor::Contiguous(reference.data(), DType::kF32, Cpu(), {m, n});
    vt::ScaledFp4Quant(cpu_queue, ta, tas, tx, kInputGlobalScale);
    vt::MatmulNvfp4Fp4(cpu_queue, tout, ta, tas, tw, tws, kAlpha);
  }

  const int64_t mp = RoundUp(m, 128);
  const int64_t np = RoundUp(n, 128);
  const int64_t kp = RoundUp(k / 16, 4);
  const auto upload = [&](const void* host, size_t bytes) {
    void* device = backend.Alloc(bytes);
    backend.Copy(queue, device, host, bytes);
    return device;
  };
  void* dx = bf16_input ? upload(x_bf16.data(), x_bf16.size() * sizeof(uint16_t))
                        : upload(x.data(), x.size() * sizeof(float));
  void* dw = upload(w_packed.data(), w_packed.size());
  void* dws = upload(w_scale.data(), w_scale.size());
  void* dalpha = upload(&kAlpha, sizeof(kAlpha));
  void* da = backend.Alloc(static_cast<size_t>(m * k / 2));
  void* das = backend.Alloc(static_cast<size_t>(m * k / 16));
  void* dasw = backend.Alloc(static_cast<size_t>(mp * kp));
  void* dwsw = backend.Alloc(static_cast<size_t>(np * kp));
  void* dout = backend.Alloc(static_cast<size_t>(m * n) * sizeof(uint16_t));
  backend.Memset(queue, dasw, 0, static_cast<size_t>(mp * kp));
  backend.Memset(queue, dwsw, 0, static_cast<size_t>(np * kp));

  Tensor tx = GpuTensor({m, k});
  tx.data = dx;
  tx.dtype = bf16_input ? DType::kBF16 : DType::kF32;
  tx.device = Gpu();
  Tensor ta = GpuTensor({m, k / 2});
  ta.data = da;
  ta.dtype = DType::kI8;
  ta.device = Gpu();
  Tensor tas = GpuTensor({m, k / 16});
  tas.data = das;
  tas.dtype = DType::kI8;
  tas.device = Gpu();
  Tensor tasw = GpuTensor({mp, kp});
  tasw.data = dasw;
  tasw.dtype = DType::kI8;
  tasw.device = Gpu();
  Tensor tw = GpuTensor({n, k / 2});
  tw.data = dw;
  tw.dtype = DType::kI8;
  tw.device = Gpu();
  Tensor tws = GpuTensor({n, k / 16});
  tws.data = dws;
  tws.dtype = DType::kI8;
  tws.device = Gpu();
  Tensor twsw = GpuTensor({np, kp});
  twsw.data = dwsw;
  twsw.dtype = DType::kI8;
  twsw.device = Gpu();
  Tensor tout = GpuTensor({m, n});
  tout.data = dout;
  tout.dtype = DType::kBF16;
  tout.device = Gpu();
  Tensor talpha = GpuTensor({1});
  talpha.data = dalpha;
  talpha.dtype = DType::kF32;
  talpha.device = Gpu();

  if (direct_scale) {
    vt::ScaledFp4Quant(queue, ta, tasw, tx, kInputGlobalScale,
                       vt::Fp4ScaleLayout::kCutlassSwizzled);
  } else {
    vt::ScaledFp4Quant(queue, ta, tas, tx, kInputGlobalScale);
    vt::SwizzleBlockscale(queue, tasw, tas);
  }
  vt::SwizzleBlockscale(queue, twsw, tws);
  vt::MatmulNvfp4Cutlass(queue, tout, ta, tasw, tw, twsw, talpha);
  std::vector<uint16_t> first(static_cast<size_t>(m * n));
  backend.Copy(queue, first.data(), dout, first.size() * sizeof(uint16_t));
  backend.Synchronize(queue);

  double max_abs = 0.0;
  double ref_absmax = 0.0;
  for (size_t i = 0; i < first.size(); ++i) {
    const double got = Bf16ToFloat(first[i]);
    max_abs = std::max(max_abs, std::abs(got - reference[i]));
    ref_absmax = std::max(ref_absmax, std::abs(static_cast<double>(reference[i])));
  }
  CAPTURE(tactic);
  CAPTURE(m);
  CAPTURE(max_abs);
  CAPTURE(ref_absmax);
  if (std::getenv("VT_FP4_TEST_VERBOSE") != nullptr) {
    std::fprintf(stderr,
                 "NVFP4 forced tactic=%d M=%lld N=%lld K=%lld max_abs=%.9g "
                 "ref_absmax=%.9g relative=%.9g\n",
                 tactic, static_cast<long long>(m), static_cast<long long>(n),
                 static_cast<long long>(k), max_abs, ref_absmax,
                 ref_absmax == 0.0 ? 0.0 : max_abs / ref_absmax);
  }
  CHECK(max_abs <= 0.03 * ref_absmax + 1e-3);

  backend.Memset(queue, dout, 0, first.size() * sizeof(uint16_t));
  backend.Synchronize(queue);
  backend.BeginCapture(queue);
  vt::MatmulNvfp4Cutlass(queue, tout, ta, tasw, tw, twsw, talpha);
  void* graph = backend.EndCaptureGraph(queue);
  REQUIRE(graph != nullptr);
  backend.ReplayGraph(queue, graph);
  backend.Synchronize(queue);
  std::vector<uint16_t> replay(first.size());
  backend.Copy(queue, replay.data(), dout, replay.size() * sizeof(uint16_t));
  backend.Synchronize(queue);
  CHECK(replay == first);
  backend.DestroyGraph(graph);

  if (const char* dump_dir = std::getenv("VT_FP4_TEST_DUMP_DIR"); dump_dir != nullptr) {
    std::vector<uint8_t> gpu_a_packed(static_cast<size_t>(m * k / 2));
    std::vector<uint8_t> gpu_a_sf(static_cast<size_t>(mp * kp));
    std::vector<uint8_t> gpu_b_sf(static_cast<size_t>(np * kp));
    backend.Copy(queue, gpu_a_packed.data(), da, gpu_a_packed.size());
    backend.Copy(queue, gpu_a_sf.data(), dasw, gpu_a_sf.size());
    backend.Copy(queue, gpu_b_sf.data(), dwsw, gpu_b_sf.size());
    backend.Synchronize(queue);
    const auto write = [dump_dir](const char* name, const void* data, size_t bytes) {
      std::ofstream stream(std::string(dump_dir) + "/" + name,
                           std::ios::binary | std::ios::trunc);
      REQUIRE(stream.good());
      stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
      REQUIRE(stream.good());
    };
    write("a_packed.bin", gpu_a_packed.data(), gpu_a_packed.size());
    write("a_sf_sw.bin", gpu_a_sf.data(), gpu_a_sf.size());
    write("b_packed.bin", w_packed.data(), w_packed.size());
    write("b_sf_sw.bin", gpu_b_sf.data(), gpu_b_sf.size());
    write("alpha.bin", &kAlpha, sizeof(kAlpha));
    write("input_global_scale.bin", &kInputGlobalScale,
          sizeof(kInputGlobalScale));
    if (bf16_input) {
      write("input_bf16.bin", x_bf16.data(), x_bf16.size() * sizeof(uint16_t));
    }
    write("out_bf16.bin", first.data(), first.size() * sizeof(uint16_t));
    std::ofstream metadata(std::string(dump_dir) + "/shape.txt", std::ios::trunc);
    REQUIRE(metadata.good());
    metadata << m << ' ' << n << ' ' << k << ' ' << mp << ' ' << np << ' ' << kp << '\n';
    REQUIRE(metadata.good());
  }

  for (void* pointer : {dx, dw, dws, dalpha, da, das, dasw, dwsw,
                        dout}) {
    backend.Free(pointer);
  }
  backend.DestroyQueue(queue);
}

// Port of vLLM v0.25 kernel_warmup.py:133-213 plus FlashInfer 0.6.13's
// user-loaded-config priority and tune_mode=False miss contract. CMake runs
// this case alone in a fresh process against the immutable 64-plan GB10 cache.
TEST_CASE("NVFP4 persistent runtime loads frozen plans before warmup") {
  if (std::getenv("VT_FP4_TEST_PERSISTENT_RUNTIME") == nullptr || !HasCuda()) {
    return;
  }
  const char* native_path = std::getenv("VT_FP4_AUTOTUNE_CACHE_PATH");
  REQUIRE(native_path != nullptr);
  std::error_code remove_error;
  std::filesystem::remove(native_path, remove_error);
  REQUIRE_FALSE(remove_error);

  auto& backend = vt::GetBackend(DeviceType::kCUDA);
  Queue queue = backend.CreateQueue();
  constexpr int64_t kM = 16;
  constexpr int64_t kN = 5120;
  constexpr int64_t kK = 6144;
  constexpr int64_t kMp = 128;
  constexpr int64_t kNp = 5120;
  constexpr int64_t kKp = 384;
  constexpr float kAlpha = 1.0F;

  void* a = backend.Alloc(static_cast<size_t>(kM * kK / 2));
  void* a_sf = backend.Alloc(static_cast<size_t>(kMp * kKp));
  void* b = backend.Alloc(static_cast<size_t>(kN * kK / 2));
  void* b_sf = backend.Alloc(static_cast<size_t>(kNp * kKp));
  void* out = backend.Alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
  for (const auto& [pointer, bytes] :
       std::array<std::pair<void*, size_t>, 5>{
           std::pair{a, static_cast<size_t>(kM * kK / 2)},
           std::pair{a_sf, static_cast<size_t>(kMp * kKp)},
           std::pair{b, static_cast<size_t>(kN * kK / 2)},
           std::pair{b_sf, static_cast<size_t>(kNp * kKp)},
           std::pair{out,
                     static_cast<size_t>(kM * kN) * sizeof(uint16_t)}}) {
    backend.Memset(queue, pointer, 0, bytes);
  }
  backend.Synchronize(queue);

  Tensor ta = GpuTensor({kM, kK / 2});
  ta.data = a;
  ta.dtype = DType::kI8;
  Tensor ta_sf = GpuTensor({kMp, kKp});
  ta_sf.data = a_sf;
  ta_sf.dtype = DType::kI8;
  Tensor tb = GpuTensor({kN, kK / 2});
  tb.data = b;
  tb.dtype = DType::kI8;
  Tensor tb_sf = GpuTensor({kNp, kKp});
  tb_sf.data = b_sf;
  tb_sf.dtype = DType::kI8;
  Tensor tout = GpuTensor({kM, kN});
  tout.data = out;
  tout.dtype = DType::kBF16;

  const vt::cuda::Nvfp4AutotuneWarmupStats before =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  {
    vt::cuda::Nvfp4AutotuneWarmupScope warmup(
        static_cast<uint32_t>(kM), queue.device.index);
    const vt::cuda::Nvfp4AutotuneWarmupStats loaded =
        vt::cuda::GetNvfp4AutotuneWarmupStats();
    CHECK(loaded.persistent_cache_enabled);
    CHECK(loaded.read_only);
    CHECK(loaded.mode == "read-only");
    CHECK(loaded.delay_microseconds == 5000);
    CHECK(loaded.profiles_loaded == before.profiles_loaded + 64);
    CHECK(loaded.selected_plans.size() == 64);
    vt::MatmulNvfp4Cutlass(queue, tout, ta, ta_sf, tb, tb_sf, kAlpha);
    warmup.Complete();
  }
  backend.Synchronize(queue);

  const vt::cuda::Nvfp4AutotuneWarmupStats complete =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  CHECK(complete.scopes_completed == before.scopes_completed + 1);
  CHECK(complete.profiles_requested == before.profiles_requested);
  CHECK(complete.profiles_tuned == before.profiles_tuned);
  CHECK(complete.profiles_saved == before.profiles_saved);
  CHECK(complete.lazy_misses == before.lazy_misses);
  CHECK(complete.selected_plans.size() == 64);
  CHECK_FALSE(std::filesystem::exists(native_path));
  bool found_binding_plan = false;
  for (const vt::cuda::Nvfp4AutotuneSelectedPlan& plan :
       complete.selected_plans) {
    if (plan.m_bucket == 16 && plan.n == 5120 && plan.k == 6144) {
      CHECK(plan.tactic_id == 0);
      found_binding_plan = true;
    }
  }
  CHECK(found_binding_plan);

  Tensor ta_hit = ta;
  ta_hit.shape[0] = 8;
  Tensor tout_hit = tout;
  tout_hit.shape[0] = 8;
  backend.BeginCapture(queue);
  vt::MatmulNvfp4Cutlass(queue, tout_hit, ta_hit, ta_sf, tb, tb_sf, kAlpha);
  void* graph = backend.EndCaptureGraph(queue);
  REQUIRE(graph != nullptr);
  backend.ReplayGraph(queue, graph);
  backend.Synchronize(queue);
  backend.DestroyGraph(graph);

  Tensor ta_miss = GpuTensor({1, 256});
  ta_miss.data = a;
  ta_miss.dtype = DType::kI8;
  Tensor ta_sf_miss = GpuTensor({128, 32});
  ta_sf_miss.data = a_sf;
  ta_sf_miss.dtype = DType::kI8;
  Tensor tb_miss = GpuTensor({256, 256});
  tb_miss.data = b;
  tb_miss.dtype = DType::kI8;
  Tensor tb_sf_miss = GpuTensor({256, 32});
  tb_sf_miss.data = b_sf;
  tb_sf_miss.dtype = DType::kI8;
  Tensor tout_miss = GpuTensor({1, 256});
  tout_miss.data = out;
  tout_miss.dtype = DType::kBF16;
  CHECK_THROWS_WITH_AS(
      vt::MatmulNvfp4Cutlass(queue, tout_miss, ta_miss, ta_sf_miss,
                             tb_miss, tb_sf_miss, kAlpha),
      "NVFP4 frozen persistent cache miss before readiness: M=1 bucket=1 "
      "N=256 K=512 device=0 sm=121",
      std::runtime_error);
  backend.Synchronize(queue);
  CHECK(vt::cuda::GetNvfp4AutotuneWarmupStats().lazy_misses ==
        before.lazy_misses);

  for (void* pointer : {a, a_sf, b, b_sf, out}) backend.Free(pointer);
  backend.DestroyQueue(queue);
}

TEST_CASE("NVFP4 persistent runtime publishes only completed warmup") {
  if (std::getenv("VT_FP4_TEST_PERSISTENT_SAVE") == nullptr || !HasCuda()) {
    return;
  }
  const char* native_path = std::getenv("VT_FP4_AUTOTUNE_CACHE_PATH");
  REQUIRE(native_path != nullptr);
  std::error_code file_error;
  std::filesystem::remove(native_path, file_error);
  REQUIRE_FALSE(file_error);

  auto& backend = vt::GetBackend(DeviceType::kCUDA);
  Queue queue = backend.CreateQueue();
  constexpr int64_t kM = 16;
  constexpr int64_t kN = 256;
  constexpr int64_t kK = 512;
  constexpr int64_t kMp = 128;
  constexpr int64_t kNp = 256;
  constexpr int64_t kKp = 32;
  void* a = backend.Alloc(static_cast<size_t>(kM * kK / 2));
  void* a_sf = backend.Alloc(static_cast<size_t>(kMp * kKp));
  void* b = backend.Alloc(static_cast<size_t>(kN * kK / 2));
  void* b_sf = backend.Alloc(static_cast<size_t>(kNp * kKp));
  void* out = backend.Alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
  for (const auto& [pointer, bytes] :
       std::array<std::pair<void*, size_t>, 5>{
           std::pair{a, static_cast<size_t>(kM * kK / 2)},
           std::pair{a_sf, static_cast<size_t>(kMp * kKp)},
           std::pair{b, static_cast<size_t>(kN * kK / 2)},
           std::pair{b_sf, static_cast<size_t>(kNp * kKp)},
           std::pair{out,
                     static_cast<size_t>(kM * kN) * sizeof(uint16_t)}}) {
    backend.Memset(queue, pointer, 0, bytes);
  }
  backend.Synchronize(queue);

  Tensor ta = GpuTensor({kM, kK / 2});
  ta.data = a;
  ta.dtype = DType::kI8;
  Tensor ta_sf = GpuTensor({kMp, kKp});
  ta_sf.data = a_sf;
  ta_sf.dtype = DType::kI8;
  Tensor tb = GpuTensor({kN, kK / 2});
  tb.data = b;
  tb.dtype = DType::kI8;
  Tensor tb_sf = GpuTensor({kNp, kKp});
  tb_sf.data = b_sf;
  tb_sf.dtype = DType::kI8;
  Tensor tout = GpuTensor({kM, kN});
  tout.data = out;
  tout.dtype = DType::kBF16;

  const vt::cuda::Nvfp4AutotuneWarmupStats before =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  {
    vt::cuda::Nvfp4AutotuneWarmupScope cancelled(
        static_cast<uint32_t>(kM), queue.device.index);
    vt::MatmulNvfp4Cutlass(queue, tout, ta, ta_sf, tb, tb_sf, 1.0F);
  }
  backend.Synchronize(queue);
  CHECK_FALSE(std::filesystem::exists(native_path));
  const vt::cuda::Nvfp4AutotuneWarmupStats after_cancel =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  CHECK(after_cancel.scopes_started == before.scopes_started + 1);
  CHECK(after_cancel.scopes_completed == before.scopes_completed);
  CHECK(after_cancel.profiles_tuned == before.profiles_tuned + 5);
  CHECK(after_cancel.profiles_saved == before.profiles_saved);

  {
    vt::cuda::Nvfp4AutotuneWarmupScope completed(
        static_cast<uint32_t>(kM), queue.device.index);
    vt::MatmulNvfp4Cutlass(queue, tout, ta, ta_sf, tb, tb_sf, 1.0F);
    completed.Complete();
  }
  backend.Synchronize(queue);
  const vt::cuda::Nvfp4AutotuneWarmupStats after_complete =
      vt::cuda::GetNvfp4AutotuneWarmupStats();
  CHECK(after_complete.scopes_completed == before.scopes_completed + 1);
  CHECK(after_complete.profiles_tuned == after_cancel.profiles_tuned);
  CHECK(after_complete.profiles_saved == before.profiles_saved + 5);
  CHECK(after_complete.selected_plans.size() == 5);
  CHECK(std::filesystem::is_regular_file(native_path));
  CHECK(std::filesystem::file_size(native_path) > 0);

  std::filesystem::remove(native_path, file_error);
  CHECK_FALSE(file_error);
  for (void* pointer : {a, a_sf, b, b_sf, out}) backend.Free(pointer);
  backend.DestroyQueue(queue);
}
#endif  // VT_CUTLASS_NVFP4

// ── BACKEND-CUDA-ARCH-ADDITIVITY: the runtime SM-dispatch seam is EXERCISED ──
// A passing correctness gate does NOT prove a new code path ran (the W7 decode
// graph taught us that: graph-ON and graph-OFF logs were byte-identical until a
// stats counter proved capture happened). So this case asserts on the tactic
// registry's OWN counters, which only move when SelectArchTactic actually runs:
//
//   * the capability probe (seam-gap #4) returns a REAL architecture, not zeros,
//     and agrees with the backend's newly-carried DeviceCapabilityMajor/Minor;
//   * exactly one tactic is registered for the nvfp4 fp4xfp4 family (seam-gap #2)
//     — the additivity counter: it rises by one per architecture brought up;
//   * with VT_NVFP4_FP4_NATIVE unset (the production default, the arm the gate
//     models run) the launcher consults the registry, no tactic supports the
//     device, and `fallbacks` advances -> the portable path ran, as before;
//   * with VT_NVFP4_FP4_NATIVE=1 the sm_12x tactic IS selected on a sm_12x
//     device and `selections` advances with `last_selected` naming it.
// The A/B is same-binary.
#ifdef VLLM_CPP_CUDA
TEST_CASE("CUDA arch tactic registry is exercised by the fp4xfp4 launcher") {
  if (!HasCuda()) return;
  const vt::cuda::DeviceCaps& caps = vt::cuda::GetDeviceCaps();
  REQUIRE(caps.valid);
  CHECK(caps.sm_major > 0);
  CHECK(caps.max_shared_memory_per_block_optin >= 48 * 1024);
  CHECK(caps.multiprocessor_count > 0);

  auto& b = vt::GetBackend(DeviceType::kCUDA);
  // seam-gap #4: the kernel-layer backend now CARRIES the capability, from the
  // same cached probe the platform seam reports.
  CHECK(b.DeviceCapabilityMajor() == caps.sm_major);
  CHECK(b.DeviceCapabilityMinor() == caps.sm_minor);

  const int registered =
      vt::cuda::RegisteredArchTacticCount(vt::cuda::TacticFamily::kNvfp4Fp4Mma);
#ifdef VT_FP4_MMA_SM120A
  CHECK(registered == 1);  // exactly the one tactic this change registers
#else
  CHECK(registered == 0);
#endif

  Queue gq = b.CreateQueue();
  const int64_t M = 40, K = 128, N = 33;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);
  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& v : w_packed) v = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& v : w_scale) v = vllm::F32ToF8E4M3(scale_d(rng));
  const float input_global_scale = 8.2F, weight_global_scale = 5.5F;
  const float alpha = (1.0F / input_global_scale) * (1.0F / weight_global_scale);

  auto up = [&](const void* h, size_t nb) {
    void* p = b.Alloc(nb);
    b.Copy(gq, p, h, nb);
    return p;
  };
  void* dx = up(x.data(), x.size() * sizeof(float));
  void* dbp = up(w_packed.data(), w_packed.size());
  void* dbs = up(w_scale.data(), w_scale.size());
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dout = b.Alloc(static_cast<size_t>(M * N) * sizeof(float));
  Tensor tx = GpuTensor({M, K}); tx.data = dx; tx.dtype = DType::kF32; tx.device = Gpu();
  Tensor tap = GpuTensor({M, K / 2}); tap.data = dap; tap.dtype = DType::kI8; tap.device = Gpu();
  Tensor tas = GpuTensor({M, K / 16}); tas.data = das; tas.dtype = DType::kI8; tas.device = Gpu();
  Tensor tbp = GpuTensor({N, K / 2}); tbp.data = dbp; tbp.dtype = DType::kI8; tbp.device = Gpu();
  Tensor tbs = GpuTensor({N, K / 16}); tbs.data = dbs; tbs.dtype = DType::kI8; tbs.device = Gpu();
  Tensor to = GpuTensor({M, N}); to.data = dout; to.dtype = DType::kF32; to.device = Gpu();
  vt::ScaledFp4Quant(gq, tap, tas, tx, input_global_scale);

  // ARM A — production default (VT_NVFP4_FP4_NATIVE unset).
  const vt::cuda::ArchTacticStats before =
      vt::cuda::GetArchTacticStats(vt::cuda::TacticFamily::kNvfp4Fp4Mma);
  {
    ScopedEnv off("VT_NVFP4_FP4_NATIVE", nullptr);
    vt::MatmulNvfp4Fp4(gq, to, tap, tas, tbp, tbs, alpha);
    b.Synchronize(gq);
  }
  const vt::cuda::ArchTacticStats after_default =
      vt::cuda::GetArchTacticStats(vt::cuda::TacticFamily::kNvfp4Fp4Mma);
  CHECK(after_default.fallbacks > before.fallbacks);     // the seam RAN
  CHECK(after_default.selections == before.selections);  // and chose nothing

#ifdef VT_FP4_MMA_SM120A
  // ARM B — the sm_12x tactic is selected on a sm_12x device.
  {
    ScopedEnv on("VT_NVFP4_FP4_NATIVE", "1");
    vt::MatmulNvfp4Fp4(gq, to, tap, tas, tbp, tbs, alpha);
    b.Synchronize(gq);
  }
  const vt::cuda::ArchTacticStats after_native =
      vt::cuda::GetArchTacticStats(vt::cuda::TacticFamily::kNvfp4Fp4Mma);
  if (caps.sm_major == 12) {
    CHECK(after_native.selections == after_default.selections + 1);
    REQUIRE(after_native.last_selected != nullptr);
    CHECK(std::string(after_native.last_selected) == "nvfp4-fp4-mma/sm12x");
  } else {
    CHECK(after_native.fallbacks > after_default.fallbacks);
  }
#endif  // VT_FP4_MMA_SM120A

  for (void* p : {dx, dbp, dbs, dap, das, dout}) b.Free(p);
  b.DestroyQueue(gq);
}

// ── BACKEND-CUDA-SM090 (arch additivity §W9): the SM-dispatch seam is CROSS-
// FAMILY additive, not merely cross-arch within the sm_12x family ─────────────
// sm_120a was near-free because it SHARES the sm_12x kernel bodies (same fp4
// `mma.sync kind::mxf4nvf4` PTX). A genuinely different family (Hopper sm_90,
// datacenter Blackwell sm_100, Ampere sm_80) has NO body here, so bringing one
// up must be: register its tactic from its own TU, and the launcher selects it
// by capability with ZERO launcher edits. This case proves the SELECTOR half of
// that contract WITHOUT fabricating a kernel we cannot test: it feeds the
// registry a synthetic Hopper capability and asserts the ONE registered tactic
// (sm_12x) correctly DECLINES it, so a Hopper device falls through to the
// portable path today and a real Hopper tactic would be additively selectable
// tomorrow. Pure table logic — no GPU, no device dispatch — so it runs anywhere.
TEST_CASE("CUDA arch tactic registry is cross-family additive (Hopper sm_90 declines to sm_12x)") {
  using vt::cuda::ArchTactic;
  using vt::cuda::DeviceCaps;
  using vt::cuda::SelectArchTactic;
  using vt::cuda::TacticFamily;

  // A synthetic, VALID Hopper capability. No CUDA device is required: the
  // registry's selection is a pure predicate over DeviceCaps.
  DeviceCaps hopper;
  hopper.valid = true;
  hopper.sm_major = 9;  // Hopper H100/H200 — a different FAMILY from sm_12x
  hopper.sm_minor = 0;
  hopper.max_shared_memory_per_block_optin = 227 * 1024;  // H100 opt-in ceiling
  hopper.multiprocessor_count = 132;

  // The fp4 family must NOT select the sm_12x tactic for a Hopper device: its
  // `mma.sync kind::mxf4nvf4` is consumer-Blackwell-only. A null return is the
  // portable-fallback signal the launcher already handles.
  const ArchTactic* fp4 = SelectArchTactic(TacticFamily::kNvfp4Fp4Mma, hopper);
  CHECK(fp4 == nullptr);

  // Same for a synthetic datacenter-Blackwell (sm_100) and an Ampere (sm_80):
  // neither shares the sm_12x fp4 body, so both fall back.
  for (int major : {10, 8}) {
    DeviceCaps other = hopper;
    other.sm_major = major;
    CHECK(SelectArchTactic(TacticFamily::kNvfp4Fp4Mma, other) == nullptr);
  }

  // The additivity counter is still exactly the sm_12x population — registering
  // a Hopper tactic would raise it by one and this selection would then return
  // it, with no edit to LaunchFp4Fp4. That the registry ACCEPTS a major!=12 entry
  // is the whole point of the seam.
  const int registered =
      vt::cuda::RegisteredArchTacticCount(TacticFamily::kNvfp4Fp4Mma);
#ifdef VT_FP4_MMA_SM120A
  CHECK(registered == 1);
#else
  CHECK(registered == 0);
#endif
}
#endif  // VLLM_CPP_CUDA
