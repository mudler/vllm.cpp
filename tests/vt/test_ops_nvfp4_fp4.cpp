// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32, kE2M1Lut
#include "vt/backend.h"
#include "vt/cuda/nvfp4_plan_cache.h"
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

    std::vector<uint8_t> ref_ap(static_cast<size_t>(M * I / 2)), ref_as(static_cast<size_t>(M * I / 16));
    std::vector<uint8_t> fu_ap(static_cast<size_t>(M * I / 2)), fu_as(static_cast<size_t>(M * I / 16));
    b.Copy(gq, ref_ap.data(), dref_ap, ref_ap.size());
    b.Copy(gq, ref_as.data(), dref_as, ref_as.size());
    b.Copy(gq, fu_ap.data(), dfu_ap, fu_ap.size());
    b.Copy(gq, fu_as.data(), dfu_as, fu_as.size());
    b.Synchronize(gq);

    size_t pmis = 0, smis = 0;
    for (size_t i = 0; i < ref_ap.size(); ++i) pmis += (fu_ap[i] != ref_ap[i]);
    for (size_t i = 0; i < ref_as.size(); ++i) smis += (fu_as[i] != ref_as[i]);
    CHECK(pmis == 0);  // fused packed fp4 == unfused, byte-for-byte
    CHECK(smis == 0);  // fused fp8 block scales == unfused, byte-for-byte

    for (void* p : {dgate, dup, dact, dref_ap, dref_as, dfu_ap, dfu_as}) b.Free(p);
  };

  run_check(1, 64);      // decode single-row
  run_check(8, 256);     // small batch, many groups/row
  run_check(40, 128);    // prefill-ish
  run_check(37, 2048);   // the real 27B intermediate_size class, non-32-aligned M
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
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dasw = b.Alloc(static_cast<size_t>(Mp * Kp));
  void* dbsw = b.Alloc(static_cast<size_t>(Np * Kp));
  void* dout = b.Alloc(static_cast<size_t>(M * N) * sizeof(uint16_t));  // bf16
  b.Memset(gq, dasw, 0, static_cast<size_t>(Mp * Kp));
  b.Memset(gq, dbsw, 0, static_cast<size_t>(Np * Kp));

  Tensor tx = GpuTensor({M, K}); tx.data = dx; tx.dtype = DType::kF32; tx.device = Gpu();
  Tensor tap = GpuTensor({M, K / 2}); tap.data = dap; tap.dtype = DType::kI8; tap.device = Gpu();
  Tensor tas = GpuTensor({M, K / 16}); tas.data = das; tas.dtype = DType::kI8; tas.device = Gpu();
  Tensor tasw = GpuTensor({Mp, Kp}); tasw.data = dasw; tasw.dtype = DType::kI8; tasw.device = Gpu();
  Tensor tbp = GpuTensor({N, K / 2}); tbp.data = dbp; tbp.dtype = DType::kI8; tbp.device = Gpu();
  Tensor tbs = GpuTensor({N, K / 16}); tbs.data = dbs; tbs.dtype = DType::kI8; tbs.device = Gpu();
  Tensor tbsw = GpuTensor({Np, Kp}); tbsw.data = dbsw; tbsw.dtype = DType::kI8; tbsw.device = Gpu();
  Tensor to = GpuTensor({M, N}); to.data = dout; to.dtype = DType::kBF16; to.device = Gpu();

  vt::ScaledFp4Quant(gq, tap, tas, tx, input_global_scale);
  vt::SwizzleBlockscale(gq, tasw, tas);
  vt::SwizzleBlockscale(gq, tbsw, tbs);
  vt::MatmulNvfp4Cutlass(gq, to, tap, tasw, tbp, tbsw, alpha);

  std::vector<uint16_t> g_out(static_cast<size_t>(M * N));
  b.Copy(gq, g_out.data(), dout, g_out.size() * sizeof(uint16_t));
  b.Synchronize(gq);

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

  for (void* p : {dx, dbp, dbs, dap, das, dasw, dbsw, dout}) b.Free(p);
  b.DestroyQueue(gq);
}
#endif  // VT_CUTLASS_NVFP4
