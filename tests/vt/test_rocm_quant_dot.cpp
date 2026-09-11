// ROCm keep-quant GEMM gate (KERNEL-QUANT-CIQ-GEMM-ROCM W1). The kROCM
// provider for `OpId::kMatmulBTQuant` / `kMatmulBTQuantGrouped`
// (src/vt/rocm/rocm_quant_dot.hip) is measured against the LANDED CPU
// keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle) and an
// INDEPENDENT f64 dequantize-then-dot, on the ten Q8_K-family encodings the
// CUDA sibling serves (test_cuda_quant_dot.cpp's WeightCase table).
//
// THE GATE mirrors the CUDA file: the Q8_K activation quant and the whole
// INTEGER dot are bit-identical to the CPU reference by construction, so
// ROCm-vs-CPU is asserted at a TIGHT NMSE (1e-6, f32 out) — only the per-
// super-block float scale sum is reassociated (warp reduction vs the CPU's
// sequential add). ROCm-vs-f64-dequant uses the same 5e-4 band
// test_ops_quant_dot.cpp applies. A wrong codebook index / scale unpack /
// sign blows both bands (RED-first).
//
// Skips cleanly when no AMD GPU is present, so CPU-only CI stays green.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/cpu/cpu_quant_blocks.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/rocm/rocm_mmvq_policy.h"
#include "vt/rocm/rocm_norm_quant_bridge.h"
#include "vt/tensor.h"

// Compile the exact scratch policy used by the HIP provider with host fakes.
// The provider file excludes kernels in this mode, so CPU-only CI can gate
// capture ownership without a ROCm runtime.
#define VT_ROCM_QUANT_DOT_SCRATCH_TEST_ONLY
#include "../../src/vt/rocm/rocm_quant_dot.hip"
#undef VT_ROCM_QUANT_DOT_SCRATCH_TEST_ONLY

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

#if defined(VLLM_CPP_HIP)
namespace vt::rocm {
void Q8KResetRouteDispatchCountsForTest();
uint64_t Q8KRouteDispatchCountForTest(bool grouped, bool candidate);
void Q8KQuantizeForTest(Queue& q, void* scratch, const void* act, DType dtype,
                        int64_t row_stride, int64_t rows, int64_t nsb,
                        bool candidate);
}  // namespace vt::rocm
#endif

namespace {

using ScratchCaptureState = vt::rocm::detail::ScratchCaptureState;

struct FakeScratchRuntime {
  ScratchCaptureState capture = ScratchCaptureState::kNone;
  int query_result = 0;
  int queries = 0;
  int allocations = 0;
  std::vector<std::vector<uint8_t>> blocks;

  ScratchCaptureState Query(int) {
    ++queries;
    if (query_result != 0) throw std::runtime_error("capture query failed");
    return capture;
  }

  void* Allocate(size_t bytes, int) {
    ++allocations;
    blocks.emplace_back(bytes);
    return blocks.back().data();
  }
};

constexpr double kMaxNmseErr = 5e-4;      // test-backend-ops.cpp:4277 band
constexpr double kMaxNmseVsCpu = 1e-6;    // integer core exact; scale sum only

bool HasRocm() {
  try {
    vt::GetBackend(DeviceType::kROCM);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kROCM, 0}; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
  // f64-dequant ceiling override (0 = kMaxNmseErr); see the CUDA table for why
  // the IQ1 family needs a wider ACTIVATION-error band while the ROCm-vs-CPU
  // bound below stays shared and unrelaxed.
  double nmse_ref_max = 0.0;
};

const WeightCase kCases[] = {
    {DType::kIQ2_XXS, 256, 66, 0, -1, "iq2_xxs"},
    {DType::kIQ3_XXS, 256, 98, 0, -1, "iq3_xxs"},
    {DType::kIQ2_S, 256, 82, 0, -1, "iq2_s"},
    {DType::kIQ1_S, 256, 50, 0, -1, "iq1_s", 2e-3},
    {DType::kIQ1_XXXS, 256, 38, 0, -1, "iq1_xxxs", 2e-3},
    {DType::kQ2_K, 256, 84, 80, 82, "q2_K"},
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
    // Q8_0 is NOT a Q8_K-superblock encoding: it dots a Q8_0 activation and is
    // served by the *Gdn kernels this file delegates to.
    {DType::kQ8_0, 32, 34, 0, -1, "q8_0"},
};

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
    // IQ1 sub-block scales live INSIDE the weight (qh bits 12-14 / sc nibbles):
    // narrow them to encoder-plausible values exactly as the CUDA table does.
    if (c.dtype == DType::kIQ1_S) {
      for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh = 0;
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
        const uint16_t ls = static_cast<uint16_t>(2 + ((i + ib) % 3));
        qh = static_cast<uint16_t>((qh & 0x8FFFU) | (ls << 12));
        std::memcpy(blk + 34 + 2 * ib, &qh, sizeof(qh));
      }
    }
    if (c.dtype == DType::kIQ1_XXXS) {
      for (int ib = 0; ib < 8; ++ib) {
        uint8_t& byte = blk[34 + ib / 2];
        const int shift = 4 * (ib & 1);
        const uint8_t ls = static_cast<uint8_t>(2 + ((i + ib) % 3));
        const uint8_t keep_sign = static_cast<uint8_t>((byte >> shift) & 0x8);
        byte = static_cast<uint8_t>((byte & ~(0xFU << shift)) |
                                    ((keep_sign | ls) << shift));
      }
    }
  }
  return bytes;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
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

#if defined(VLLM_CPP_HIP)
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    if (value == nullptr) {
      ::unsetenv(name);
    } else {
      ::setenv(name, value, 1);
    }
  }

  ~ScopedEnv() {
    if (had_old_) {
      ::setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

std::vector<uint8_t> EncodeActivation(DType dtype, int64_t elements) {
  std::vector<float> values(static_cast<size_t>(elements));
  GenerateData(2.0F, values.size(), values.data());
  std::vector<uint8_t> encoded(static_cast<size_t>(elements) * sizeof(uint16_t));
  for (int64_t i = 0; i < elements; ++i) {
    const uint16_t value =
        dtype == DType::kBF16 ? vt::F32ToBF16(values[static_cast<size_t>(i)])
                              : vt::F32ToF16(values[static_cast<size_t>(i)]);
    std::memcpy(encoded.data() + static_cast<size_t>(i) * sizeof(value), &value,
                sizeof(value));
  }
  return encoded;
}

std::vector<uint8_t> RunDenseQuant(Backend& gpu, Queue& queue,
                                   const std::vector<uint8_t>& activation,
                                   DType activation_dtype,
                                   const std::vector<uint8_t>& weight,
                                   DType weight_dtype, DType output_dtype,
                                   int64_t m, int64_t n, int64_t k) {
  void* device_activation = gpu.Alloc(activation.size());
  void* device_weight = gpu.Alloc(weight.size());
  const size_t output_bytes =
      static_cast<size_t>(m * n) * (output_dtype == DType::kF32 ? 4 : 2);
  void* device_output = gpu.Alloc(output_bytes);
  std::vector<uint8_t> output(output_bytes, 0xA5);
  gpu.Copy(queue, device_activation, activation.data(), activation.size());
  gpu.Copy(queue, device_weight, weight.data(), weight.size());
  gpu.Copy(queue, device_output, output.data(), output.size());
  Tensor at = DevTensor(device_activation, activation_dtype, {m, k});
  Tensor wt = DevTensor(device_weight, weight_dtype, {n, k});
  Tensor ot = DevTensor(device_output, output_dtype, {m, n});
  vt::MatmulBTQuant(queue, ot, at, wt);
  gpu.Copy(queue, output.data(), device_output, output.size());
  gpu.Synchronize(queue);
  gpu.Free(device_activation);
  gpu.Free(device_weight);
  gpu.Free(device_output);
  return output;
}

namespace vt_rocm_test_api {
using vt::rocm::MmvqRouteCounts;
void ResetMmvq() { vt::rocm::MmvqResetRouteCountsForTesting(); }
MmvqRouteCounts MmvqCounts() { return vt::rocm::MmvqRouteCountsForTesting(); }
}  // namespace vt_rocm_test_api

std::vector<uint8_t> EncodeNormValues(DType dtype,
                                      const std::vector<float>& values) {
  const size_t element_bytes = dtype == DType::kF32 ? sizeof(float)
                                                     : sizeof(uint16_t);
  std::vector<uint8_t> encoded(values.size() * element_bytes);
  for (size_t i = 0; i < values.size(); ++i) {
    if (dtype == DType::kF32) {
      std::memcpy(encoded.data() + i * element_bytes, &values[i],
                  sizeof(float));
    } else {
      const uint16_t value =
          dtype == DType::kBF16 ? vt::F32ToBF16(values[i])
                                : vt::F32ToF16(values[i]);
      std::memcpy(encoded.data() + i * element_bytes, &value, sizeof(value));
    }
  }
  return encoded;
}

float DecodeNormValue(DType dtype, const uint8_t* encoded, size_t index) {
  if (dtype == DType::kF32) {
    float value = 0.0F;
    std::memcpy(&value, encoded + index * sizeof(float), sizeof(value));
    return value;
  }
  uint16_t value = 0;
  std::memcpy(&value, encoded + index * sizeof(value), sizeof(value));
  return vt::BF16ToF32(value);
}
#endif

}  // namespace

TEST_CASE("ROCm MMVQ route policy keeps the default and gates the fold") {
  using vt::rocm::detail::MmvqRoute;
  using vt::rocm::detail::ParseMmvqFoldMaxRows;
  using vt::rocm::detail::SelectMmvqRoute;

  CHECK(ParseMmvqFoldMaxRows(nullptr) == 512);
  CHECK(ParseMmvqFoldMaxRows("") == 512);
  CHECK(ParseMmvqFoldMaxRows("0") == 512);
  CHECK(ParseMmvqFoldMaxRows("-1") == 512);
  CHECK(ParseMmvqFoldMaxRows("512x") == 512);
  CHECK(ParseMmvqFoldMaxRows("999999999999999999999999") == 512);
  CHECK(ParseMmvqFoldMaxRows("128") == 128);

  CHECK(SelectMmvqRoute(nullptr, nullptr, 1, 256, 4096) ==
        MmvqRoute::kBaseline);
  CHECK(SelectMmvqRoute("0", nullptr, 1, 256, 4096) ==
        MmvqRoute::kBaseline);
  CHECK(SelectMmvqRoute("1", nullptr, 2, 256, 4096) ==
        MmvqRoute::kBaseline);
  CHECK(SelectMmvqRoute("1", nullptr, 1, 256, 4096) ==
        MmvqRoute::kFused);
  CHECK(SelectMmvqRoute("1", nullptr, 1, 513, 4096) ==
        MmvqRoute::kGemv);
  CHECK(SelectMmvqRoute("1", "128", 1, 128, 4096) ==
        MmvqRoute::kFused);
  CHECK(SelectMmvqRoute("1", "128", 1, 129, 4096) ==
        MmvqRoute::kGemv);
  CHECK(SelectMmvqRoute("1", nullptr, 1, 256, 32 * 1024) ==
        MmvqRoute::kFused);
  CHECK(SelectMmvqRoute("1", nullptr, 1, 256, 32 * 1024 + 1) ==
        MmvqRoute::kGemv);
}

#if defined(VLLM_CPP_HIP)
TEST_CASE("ROCm MMVQ production route is byte-exact for decode dtypes") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue queue = gpu.CreateQueue();
  constexpr int64_t k = 8 * 256;
  constexpr int64_t nsb = k / 256;
  constexpr int64_t n = 7;

  for (int case_index : {7, 8, 9}) {
    const WeightCase& c = kCases[case_index];
    std::vector<uint8_t> weight = RandomBlocks(c, n * nsb, 0x4D4D5651U);
    for (DType activation_dtype : {DType::kBF16, DType::kF16}) {
      std::vector<uint8_t> activation = EncodeActivation(activation_dtype, k);
      for (DType output_dtype : {DType::kBF16, DType::kF32}) {
        CAPTURE(std::string(c.name));
        CAPTURE(static_cast<int>(activation_dtype));
        CAPTURE(static_cast<int>(output_dtype));
        std::vector<uint8_t> baseline;
        {
          ScopedEnv mmvq("VT_GEMV_MMVQ", nullptr);
          ScopedEnv fold("VT_GEMV_MMVQ_FOLD_MAX", nullptr);
          vt_rocm_test_api::ResetMmvq();
          baseline = RunDenseQuant(gpu, queue, activation, activation_dtype,
                                   weight, c.dtype, output_dtype, 1, n, k);
          const auto counts = vt_rocm_test_api::MmvqCounts();
          CHECK(counts.baseline == 1);
          CHECK(counts.gemv == 0);
          CHECK(counts.fused == 0);
        }
        {
          ScopedEnv mmvq("VT_GEMV_MMVQ", "1");
          ScopedEnv fold("VT_GEMV_MMVQ_FOLD_MAX", nullptr);
          vt_rocm_test_api::ResetMmvq();
          const std::vector<uint8_t> candidate =
              RunDenseQuant(gpu, queue, activation, activation_dtype, weight,
                            c.dtype, output_dtype, 1, n, k);
          CHECK(candidate == baseline);
          const auto counts = vt_rocm_test_api::MmvqCounts();
          CHECK(counts.baseline == 0);
          CHECK(counts.gemv == 0);
          CHECK(counts.fused == 1);
        }
      }
    }
  }

  gpu.DestroyQueue(queue);
}

TEST_CASE("ROCm MMVQ production route observes M and fold gates") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue queue = gpu.CreateQueue();
  constexpr int64_t k = 8 * 256;
  constexpr int64_t nsb = k / 256;
  const WeightCase& c = kCases[7];

  {
    constexpr int64_t n = 513;
    std::vector<uint8_t> activation = EncodeActivation(DType::kBF16, k);
    std::vector<uint8_t> weight = RandomBlocks(c, n * nsb, 0xF01DU);
    ScopedEnv mmvq("VT_GEMV_MMVQ", "1");
    ScopedEnv fold("VT_GEMV_MMVQ_FOLD_MAX", "128");
    vt_rocm_test_api::ResetMmvq();
    RunDenseQuant(gpu, queue, activation, DType::kBF16, weight, c.dtype,
                  DType::kF32, 1, n, k);
    const auto counts = vt_rocm_test_api::MmvqCounts();
    CHECK(counts.baseline == 0);
    CHECK(counts.gemv == 1);
    CHECK(counts.fused == 0);
  }

  {
    constexpr int64_t m = 3;
    constexpr int64_t n = 7;
    std::vector<uint8_t> activation = EncodeActivation(DType::kF16, m * k);
    std::vector<uint8_t> weight = RandomBlocks(c, n * nsb, 0x4D474154U);
    ScopedEnv mmvq("VT_GEMV_MMVQ", "1");
    ScopedEnv fold("VT_GEMV_MMVQ_FOLD_MAX", nullptr);
    vt_rocm_test_api::ResetMmvq();
    RunDenseQuant(gpu, queue, activation, DType::kF16, weight, c.dtype,
                  DType::kBF16, m, n, k);
    const auto counts = vt_rocm_test_api::MmvqCounts();
    CHECK(counts.baseline == 1);
    CHECK(counts.gemv == 0);
    CHECK(counts.fused == 0);
  }

  gpu.DestroyQueue(queue);
}

TEST_CASE("ROCm fused norm quant matches standalone bytes and CPU Q8_K") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue queue = gpu.CreateQueue();
  constexpr int64_t rows = 3;
  constexpr int64_t k = 2 * 256;
  constexpr int64_t nsb = k / 256;
  const size_t scratch_bytes =
      static_cast<size_t>(rows * nsb) * sizeof(vt::cpu::BlockQ8_K);

  std::vector<float> input(static_cast<size_t>(rows * k));
  GenerateData(0.75F, static_cast<size_t>(k), input.data());
  input[static_cast<size_t>(k)] = 3.5F;
  input[static_cast<size_t>(k + 17)] = -3.5F;
  std::fill(input.begin() + 2 * k, input.end(), 0.0F);
  std::vector<float> gamma(static_cast<size_t>(k), 1.0F);
  const std::vector<uint8_t> encoded_input =
      EncodeNormValues(DType::kBF16, input);
  const std::vector<uint8_t> encoded_gamma =
      EncodeNormValues(DType::kBF16, gamma);

  void* device_input = gpu.Alloc(encoded_input.size());
  void* device_gamma = gpu.Alloc(encoded_gamma.size());
  gpu.Copy(queue, device_input, encoded_input.data(), encoded_input.size());
  gpu.Copy(queue, device_gamma, encoded_gamma.data(), encoded_gamma.size());

  ScopedEnv enabled("VT_NORM_QUANT_FUSED", "1");
  ScopedEnv mmvq("VT_GEMV_MMVQ", nullptr);
  for (DType output_dtype : {DType::kBF16, DType::kF32}) {
    CAPTURE(static_cast<int>(output_dtype));
    const size_t output_bytes = static_cast<size_t>(rows * k) *
                                (output_dtype == DType::kF32 ? 4 : 2);
    void* device_output = gpu.Alloc(output_bytes);
    void* device_reference = gpu.Alloc(scratch_bytes);
    Tensor input_tensor = DevTensor(device_input, DType::kBF16, {rows, k});
    Tensor gamma_tensor = DevTensor(device_gamma, DType::kBF16, {k});
    Tensor output_tensor =
        DevTensor(device_output, output_dtype, {rows, k});

    vt::rocm::NormQuantResetForTesting();
    vt::RmsNorm(queue, output_tensor, input_tensor, gamma_tensor,
                vt::RmsNormArgs{1e-6F, false});
    const void* fused_scratch =
        vt::rocm::NormQuantLastScratchForTesting();
    REQUIRE(fused_scratch != nullptr);
    vt::rocm::Q8KQuantizeForTest(queue, device_reference, device_output,
                                 output_dtype, k, rows, nsb, false);

    std::vector<uint8_t> fused(scratch_bytes);
    std::vector<uint8_t> standalone(scratch_bytes);
    std::vector<uint8_t> normalized(output_bytes);
    gpu.Copy(queue, fused.data(), fused_scratch, fused.size());
    gpu.Copy(queue, standalone.data(), device_reference, standalone.size());
    gpu.Copy(queue, normalized.data(), device_output, normalized.size());
    gpu.Synchronize(queue);
    CHECK(fused == standalone);

    const auto quantize_cpu = vt::cpu::BlockFromFloat(DType::kQ8_K);
    REQUIRE(quantize_cpu != nullptr);
    for (int64_t row = 0; row < rows; ++row) {
      std::vector<float> widened(static_cast<size_t>(k));
      for (int64_t column = 0; column < k; ++column) {
        widened[static_cast<size_t>(column)] = DecodeNormValue(
            output_dtype, normalized.data(),
            static_cast<size_t>(row * k + column));
      }
      std::vector<uint8_t> expected(
          static_cast<size_t>(nsb) * sizeof(vt::cpu::BlockQ8_K));
      quantize_cpu(widened.data(), expected.data(), k);
      CHECK(std::memcmp(fused.data() + static_cast<size_t>(row * nsb) *
                                         sizeof(vt::cpu::BlockQ8_K),
                        expected.data(), expected.size()) == 0);
    }
    const auto counts = vt::rocm::NormQuantCountsForTesting();
    CHECK(counts.producers == 1);
    CHECK(counts.consumers_reused == 0);
    CHECK(counts.consumers_standalone == 0);
    gpu.Free(device_reference);
    gpu.Free(device_output);
  }

  gpu.Free(device_input);
  gpu.Free(device_gamma);
  gpu.DestroyQueue(queue);
}

TEST_CASE("ROCm fused norm quant is opt-in one-shot and rejects mismatches") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue producer_queue = gpu.CreateQueue();
  Queue other_queue = gpu.CreateQueue();
  constexpr int64_t rows = 2;
  constexpr int64_t k = 2 * 256;
  constexpr int64_t n = 7;
  const WeightCase& mismatch_weights_case = kCases[7];
  std::vector<float> input(static_cast<size_t>(rows * k));
  GenerateData(1.25F, input.size(), input.data());
  std::vector<float> gamma(static_cast<size_t>(k), 1.0F);
  const std::vector<uint8_t> encoded_input =
      EncodeNormValues(DType::kBF16, input);
  const std::vector<uint8_t> encoded_gamma =
      EncodeNormValues(DType::kBF16, gamma);
  const std::vector<uint8_t> mismatch_weights =
      RandomBlocks(mismatch_weights_case, n * (k / 256), 0x2791U);
  void* device_input = gpu.Alloc(encoded_input.size());
  void* device_gamma = gpu.Alloc(encoded_gamma.size());
  void* device_norm = gpu.Alloc(encoded_input.size() + sizeof(uint16_t));
  void* device_weights = gpu.Alloc(mismatch_weights.size());
  void* device_output = gpu.Alloc(static_cast<size_t>(rows * n) * sizeof(float));
  gpu.Copy(producer_queue, device_input, encoded_input.data(),
           encoded_input.size());
  gpu.Copy(producer_queue, device_gamma, encoded_gamma.data(),
           encoded_gamma.size());
  gpu.Copy(producer_queue, device_weights, mismatch_weights.data(),
           mismatch_weights.size());
  Tensor input_tensor = DevTensor(device_input, DType::kBF16, {rows, k});
  Tensor gamma_tensor = DevTensor(device_gamma, DType::kBF16, {k});
  Tensor norm_tensor = DevTensor(device_norm, DType::kBF16, {rows, k});
  Tensor weights_tensor =
      DevTensor(device_weights, DType::kQ4_K, {n, k});
  Tensor output_tensor =
      DevTensor(device_output, DType::kF32, {rows, n});
  ScopedEnv mmvq("VT_GEMV_MMVQ", nullptr);

  {
    ScopedEnv disabled("VT_NORM_QUANT_FUSED", nullptr);
    vt::rocm::NormQuantResetForTesting();
    vt::RmsNorm(producer_queue, norm_tensor, input_tensor, gamma_tensor,
                vt::RmsNormArgs{1e-6F, false});
    vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor,
                      weights_tensor);
    const auto counts = vt::rocm::NormQuantCountsForTesting();
    CHECK(counts.producers == 0);
    CHECK(counts.consumers_reused == 0);
    CHECK(counts.consumers_standalone == 1);
  }
  {
    ScopedEnv disabled("VT_NORM_QUANT_FUSED", "0");
    vt::rocm::NormQuantResetForTesting();
    vt::RmsNorm(producer_queue, norm_tensor, input_tensor, gamma_tensor,
                vt::RmsNormArgs{1e-6F, false});
    CHECK(vt::rocm::NormQuantCountsForTesting().producers == 0);
  }

  ScopedEnv enabled("VT_NORM_QUANT_FUSED", "1");
  auto produce = [&] {
    vt::RmsNorm(producer_queue, norm_tensor, input_tensor, gamma_tensor,
                vt::RmsNormArgs{1e-6F, false});
  };

  for (size_t case_index : {size_t{7}, size_t{8}, size_t{9}}) {
    const WeightCase& weights_case = kCases[case_index];
    const std::string weights_name(weights_case.name);
    CAPTURE(weights_name);
    const std::vector<uint8_t> weights = RandomBlocks(
        weights_case, n * (k / 256), 0x2791U + case_index);
    void* format_weights = gpu.Alloc(weights.size());
    gpu.Copy(producer_queue, format_weights, weights.data(), weights.size());
    Tensor format_weights_tensor =
        DevTensor(format_weights, weights_case.dtype, {n, k});

    produce();
    vt::rocm::NormQuantResetForTesting();
    vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor,
                      format_weights_tensor);
    std::vector<uint8_t> standalone(static_cast<size_t>(rows * n) *
                                    sizeof(float));
    gpu.Copy(producer_queue, standalone.data(), device_output,
             standalone.size());
    gpu.Synchronize(producer_queue);

    vt::rocm::NormQuantResetForTesting();
    produce();
    vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor,
                      format_weights_tensor);
    std::vector<uint8_t> reused(standalone.size());
    gpu.Copy(producer_queue, reused.data(), device_output, reused.size());
    gpu.Synchronize(producer_queue);
    CHECK(reused == standalone);
    const auto format_counts = vt::rocm::NormQuantCountsForTesting();
    CHECK(format_counts.producers == 1);
    CHECK(format_counts.consumers_reused == 1);
    CHECK(format_counts.consumers_standalone == 0);
    gpu.Free(format_weights);
  }

  vt::rocm::NormQuantResetForTesting();
  produce();
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  auto counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.producers == 1);
  CHECK(counts.consumers_reused == 1);
  CHECK(counts.consumers_standalone == 1);

  vt::rocm::NormQuantResetForTesting();
  Tensor one_row_input = DevTensor(device_input, DType::kBF16, {1, k});
  Tensor one_row_norm = DevTensor(device_norm, DType::kBF16, {1, k});
  Tensor one_row_output = DevTensor(device_output, DType::kF32, {1, n});
  vt::RmsNorm(producer_queue, one_row_norm, one_row_input, gamma_tensor,
              vt::RmsNormArgs{1e-6F, false});
  {
    ScopedEnv fused_mmvq("VT_GEMV_MMVQ", "1");
    vt_rocm_test_api::ResetMmvq();
    vt::MatmulBTQuant(producer_queue, one_row_output, one_row_norm,
                      weights_tensor);
    CHECK(vt_rocm_test_api::MmvqCounts().fused == 1);
  }
  vt::MatmulBTQuant(producer_queue, one_row_output, one_row_norm,
                    weights_tensor);
  counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.consumers_reused == 0);
  CHECK(counts.consumers_standalone == 1);

  vt::rocm::NormQuantResetForTesting();
  produce();
  gpu.Synchronize(producer_queue);
  vt::MatmulBTQuant(other_queue, output_tensor, norm_tensor, weights_tensor);
  gpu.Synchronize(other_queue);
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.consumers_reused == 0);
  CHECK(counts.consumers_standalone == 2);

  vt::rocm::NormQuantResetForTesting();
  produce();
  vt::MatmulBTQuant(producer_queue, one_row_output, one_row_norm,
                    weights_tensor);
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.consumers_reused == 0);
  CHECK(counts.consumers_standalone == 2);

  vt::rocm::NormQuantResetForTesting();
  produce();
  Tensor wrong_stride = norm_tensor;
  wrong_stride.stride[0] = k + 1;
  vt::MatmulBTQuant(producer_queue, output_tensor, wrong_stride,
                    weights_tensor);
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.consumers_reused == 0);
  CHECK(counts.consumers_standalone == 2);

  vt::rocm::NormQuantResetForTesting();
  produce();
  Tensor wrong_dtype = norm_tensor;
  wrong_dtype.dtype = DType::kF16;
  vt::MatmulBTQuant(producer_queue, output_tensor, wrong_dtype,
                    weights_tensor);
  vt::MatmulBTQuant(producer_queue, output_tensor, norm_tensor, weights_tensor);
  counts = vt::rocm::NormQuantCountsForTesting();
  CHECK(counts.consumers_reused == 0);
  CHECK(counts.consumers_standalone == 2);

  gpu.Synchronize(producer_queue);
  gpu.Synchronize(other_queue);
  gpu.Free(device_input);
  gpu.Free(device_gamma);
  gpu.Free(device_norm);
  gpu.Free(device_weights);
  gpu.Free(device_output);
  gpu.DestroyQueue(other_queue);
  gpu.DestroyQueue(producer_queue);
}

TEST_CASE("ROCm fused norm quant scratch supports warm graph capture") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  REQUIRE(gpu.SupportsGraphCapture());
  Queue queue = gpu.CreateQueue();
  constexpr int64_t k = 2 * 256;
  constexpr int64_t n = 7;
  std::vector<float> input(static_cast<size_t>(k));
  GenerateData(2.0F, input.size(), input.data());
  std::vector<float> gamma(static_cast<size_t>(k), 1.0F);
  const std::vector<uint8_t> encoded_input =
      EncodeNormValues(DType::kBF16, input);
  const std::vector<uint8_t> encoded_gamma =
      EncodeNormValues(DType::kBF16, gamma);
  void* device_input = gpu.Alloc(encoded_input.size());
  void* device_gamma = gpu.Alloc(encoded_gamma.size());
  void* device_norm = gpu.Alloc(encoded_input.size());
  void* device_output = gpu.Alloc(static_cast<size_t>(n) * sizeof(float));
  gpu.Copy(queue, device_input, encoded_input.data(), encoded_input.size());
  gpu.Copy(queue, device_gamma, encoded_gamma.data(), encoded_gamma.size());
  gpu.Synchronize(queue);
  Tensor input_tensor = DevTensor(device_input, DType::kBF16, {1, k});
  Tensor gamma_tensor = DevTensor(device_gamma, DType::kBF16, {k});
  Tensor norm_tensor = DevTensor(device_norm, DType::kBF16, {1, k});
  Tensor output_tensor = DevTensor(device_output, DType::kF32, {1, n});
  ScopedEnv enabled("VT_NORM_QUANT_FUSED", "1");
  ScopedEnv mmvq("VT_GEMV_MMVQ", nullptr);

  gpu.BeginCapture(queue);
  CHECK_THROWS_WITH_AS(
      vt::RmsNorm(queue, norm_tensor, input_tensor, gamma_tensor,
                  vt::RmsNormArgs{1e-6F, false}),
      doctest::Contains("pre-warm"), std::runtime_error);
  void* empty_graph = gpu.EndCaptureGraph(queue);
  gpu.DestroyGraph(empty_graph);

  vt::RmsNorm(queue, norm_tensor, input_tensor, gamma_tensor,
              vt::RmsNormArgs{1e-6F, false});
  for (size_t case_index : {size_t{7}, size_t{8}, size_t{9}}) {
    const WeightCase& weights_case = kCases[case_index];
    const std::string weights_name(weights_case.name);
    CAPTURE(weights_name);
    const std::vector<uint8_t> weights = RandomBlocks(
        weights_case, n * (k / 256), 0xCA2791U + case_index);
    void* device_weights = gpu.Alloc(weights.size());
    gpu.Copy(queue, device_weights, weights.data(), weights.size());
    Tensor weights_tensor =
        DevTensor(device_weights, weights_case.dtype, {n, k});

    vt::rocm::NormQuantResetForTesting();
    vt::MatmulBTQuant(queue, output_tensor, norm_tensor, weights_tensor);
    std::vector<uint8_t> standalone(static_cast<size_t>(n) * sizeof(float));
    gpu.Copy(queue, standalone.data(), device_output, standalone.size());
    gpu.Synchronize(queue);

    vt::rocm::NormQuantResetForTesting();
    gpu.BeginCapture(queue);
    vt::RmsNorm(queue, norm_tensor, input_tensor, gamma_tensor,
                vt::RmsNormArgs{1e-6F, false});
    vt::MatmulBTQuant(queue, output_tensor, norm_tensor, weights_tensor);
    void* graph = gpu.EndCaptureGraph(queue);
    const auto counts = vt::rocm::NormQuantCountsForTesting();
    CHECK(counts.producers == 1);
    CHECK(counts.consumers_reused == 1);
    CHECK(counts.consumers_standalone == 0);
    gpu.ReplayGraph(queue, graph);
    std::vector<uint8_t> captured(standalone.size());
    gpu.Copy(queue, captured.data(), device_output, captured.size());
    gpu.Synchronize(queue);
    CHECK(captured == standalone);
    gpu.DestroyGraph(graph);
    gpu.Free(device_weights);
  }

  gpu.Free(device_input);
  gpu.Free(device_gamma);
  gpu.Free(device_norm);
  gpu.Free(device_output);
  gpu.DestroyQueue(queue);
}
#endif

TEST_CASE("ROCm quant-dot scratch refuses capture misses and keeps queue ownership") {
  vt::rocm::detail::QuantDotScratchPool<uint64_t, int> pool;
  FakeScratchRuntime runtime;
  auto ensure = [&](uint64_t queue, int stream, size_t bytes) {
    return pool.Ensure(
        queue, stream, bytes,
        [&](int value) { return runtime.Query(value); },
        [&](size_t value, int stream_value) {
          return runtime.Allocate(value, stream_value);
        });
  };

  runtime.capture = ScratchCaptureState::kActive;
  CHECK_THROWS_WITH_AS(ensure(11, 7, 16), doctest::Contains("pre-warm"),
                       std::runtime_error);
  CHECK(runtime.allocations == 0);
  CHECK(pool.CapacityFor(11) == 0);

  runtime.capture = ScratchCaptureState::kNone;
  void* warm = ensure(11, 7, 16);
  REQUIRE(warm != nullptr);
  CHECK(runtime.allocations == 1);
  const int queries_after_warm = runtime.queries;

  runtime.capture = ScratchCaptureState::kActive;
  CHECK(ensure(11, 7, 16) == warm);
  CHECK(runtime.queries == queries_after_warm);
  CHECK_THROWS_WITH_AS(ensure(11, 7, 32), doctest::Contains("pre-warm"),
                       std::runtime_error);
  CHECK(runtime.allocations == 1);
  CHECK(pool.BlockFor(11) == warm);
  CHECK(pool.CapacityFor(11) == 16);

  runtime.capture = ScratchCaptureState::kNone;
  CHECK(ensure(11, 7, 16) == warm);

  runtime.query_result = 1;
  CHECK_THROWS_WITH_AS(ensure(13, 8, 16), doctest::Contains("capture query failed"),
                       std::runtime_error);
  CHECK(pool.CapacityFor(13) == 0);
  runtime.query_result = 0;

  // Queue identity owns the allocation. A recycled native stream does not
  // expose the first queue's graph pointer to a later queue.
  void* second_queue = ensure(12, 7, 16);
  CHECK(second_queue != warm);
  CHECK(pool.BlockFor(11) == warm);
  CHECK(pool.BlockFor(12) == second_queue);
}

TEST_CASE("ROCm quant-dot provider preserves scratch across capture and queue reuse") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend registered; quant-dot capture gate PENDING");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  REQUIRE(gpu.SupportsGraphCapture());
  Queue cq{Cpu(), nullptr};
  const WeightCase& c = kCases[5];  // Q2_K exercises this provider, not *Gdn.
  constexpr int64_t n = 2;
  const int64_t k = 8 * c.block_elems;
  std::vector<uint8_t> weight =
      RandomBlocks(c, n * (k / c.block_elems), 0xCAFEU);

  auto cpu_golden = [&](const std::vector<float>& act, int64_t rows) {
    std::vector<float> out(static_cast<size_t>(rows * n), 0.0F);
    Tensor at = Tensor::Contiguous(const_cast<float*>(act.data()), DType::kF32,
                                   Cpu(), {rows, k});
    Tensor wt = Tensor::Contiguous(weight.data(), DType::kF32, Cpu(), {n, k});
    wt.dtype = c.dtype;
    Tensor ot = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {rows, n});
    vt::MatmulBTQuant(cq, ot, at, wt);
    return out;
  };
  auto check_output = [&](Queue& queue, void* device_out,
                          const std::vector<float>& expected) {
    std::vector<float> got(expected.size(), 0.0F);
    gpu.Copy(queue, got.data(), device_out, got.size() * sizeof(float));
    gpu.Synchronize(queue);
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      num += (got[i] - expected[i]) * (got[i] - expected[i]);
      den += expected[i] * expected[i];
    }
    CHECK((den > 0.0 ? num / den : num) <= kMaxNmseVsCpu);
  };

  auto exercise_queue = [&](Queue& queue, bool run_replay) {
    std::vector<float> act(static_cast<size_t>(k));
    GenerateData(1.0F, act.size(), act.data());
    void* device_act = gpu.Alloc(act.size() * sizeof(float));
    void* device_weight = gpu.Alloc(weight.size());
    void* device_out = gpu.Alloc(static_cast<size_t>(n) * sizeof(float));
    gpu.Copy(queue, device_act, act.data(), act.size() * sizeof(float));
    gpu.Copy(queue, device_weight, weight.data(), weight.size());
    gpu.Synchronize(queue);
    auto invoke = [&](int64_t rows, void* act_ptr, void* out_ptr) {
      Tensor at = DevTensor(act_ptr, DType::kF32, {rows, k});
      Tensor wt = DevTensor(device_weight, c.dtype, {n, k});
      Tensor ot = DevTensor(out_ptr, DType::kF32, {rows, n});
      vt::MatmulBTQuant(queue, ot, at, wt);
    };
    auto capture_refusal = [&](int64_t rows, void* act_ptr, void* out_ptr) {
      gpu.BeginCapture(queue);
      std::string refusal;
      try {
        invoke(rows, act_ptr, out_ptr);
      } catch (const std::runtime_error& error) {
        refusal = error.what();
      }
      void* empty = gpu.EndCaptureGraph(queue);
      gpu.DestroyGraph(empty);
      return refusal;
    };

    CHECK(capture_refusal(1, device_act, device_out).find("pre-warm") !=
          std::string::npos);
    if (run_replay) {
      invoke(1, device_act, device_out);
      check_output(queue, device_out, cpu_golden(act, 1));

      gpu.BeginCapture(queue);
      invoke(1, device_act, device_out);
      void* graph = gpu.EndCaptureGraph(queue);

      GenerateData(2.0F, act.size(), act.data());
      gpu.Copy(queue, device_act, act.data(), act.size() * sizeof(float));
      invoke(1, device_act, device_out);  // Eager reuse before first replay.
      check_output(queue, device_out, cpu_golden(act, 1));

      GenerateData(3.0F, act.size(), act.data());
      gpu.Copy(queue, device_act, act.data(), act.size() * sizeof(float));
      gpu.ReplayGraph(queue, graph);
      check_output(queue, device_out, cpu_golden(act, 1));

      std::vector<float> grown_act(static_cast<size_t>(2 * k));
      GenerateData(4.0F, grown_act.size(), grown_act.data());
      void* grown_device_act = gpu.Alloc(grown_act.size() * sizeof(float));
      void* grown_device_out = gpu.Alloc(static_cast<size_t>(2 * n) * sizeof(float));
      gpu.Copy(queue, grown_device_act, grown_act.data(),
               grown_act.size() * sizeof(float));
      gpu.Synchronize(queue);
      CHECK(capture_refusal(2, grown_device_act, grown_device_out).find("pre-warm") !=
            std::string::npos);
      gpu.Free(grown_device_act);
      gpu.Free(grown_device_out);
      gpu.DestroyGraph(graph);
    }
    gpu.Free(device_act);
    gpu.Free(device_weight);
    gpu.Free(device_out);
  };

  Queue first = gpu.CreateQueue();
  exercise_queue(first, true);
  gpu.DestroyQueue(first);
  Queue replacement = gpu.CreateQueue();
  exercise_queue(replacement, false);
  gpu.DestroyQueue(replacement);
}

TEST_CASE("ROCm keep-quant GEMM == CPU reference and f64 dequant (Q8_K family)") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq =
            RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
          Tensor bt =
              Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
          bt.dtype = c.dtype;
          Tensor ot =
              Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }

        // --- ROCm path (device tensors; discrete card, so real staging) ------
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> rocm_out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // --- f64 independent reference --------------------------------------
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got =
                rocm_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        const double ref_ceiling =
            c.nmse_ref_max > 0 ? c.nmse_ref_max : kMaxNmseErr;
        CHECK(nmse_ref <= ref_ceiling);     // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);   // matches the CPU oracle (int core exact)
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("ROCm keep-quant registers the native kROCM providers") {
  // The registration flips the GGUF loader's keep-quant default ON on a ROCm
  // device (GgufQuantComputeAvailable -> OpRegistered(kMatmulBTQuant,kROCM)).
  // Present only in a HIP build.
  if (!HasRocm()) return;
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuant, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, DeviceType::kROCM));
}

TEST_CASE(
    "ROCm grouped keep-quant GEMM == CPU grouped golden and it WRITES the "
    "output") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};
#if defined(VLLM_CPP_HIP)
  ScopedEnv mmvq("VT_GEMV_MMVQ", "1");
  ScopedEnv fold("VT_GEMV_MMVQ_FOLD_MAX", nullptr);
  vt_rocm_test_api::ResetMmvq();
  vt::rocm::Q8KResetRouteDispatchCountsForTest();
#endif

  // All ten encodings, decode + prefill shapes, broadcast and per-row arms —
  // the same matrix the CUDA grouped gate runs, over a POISONED output buffer.
  struct GroupedShape {
    int64_t P;
    int64_t n;
    int64_t E;
    bool bcast;
  };
  const GroupedShape kGroupedShapes[] = {
      {6, 3, 4, false}, {32, 7, 8, false}, {16, 5, 2, true}};
  int64_t combos = 0;
  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (const GroupedShape& g : kGroupedShapes) {
      CAPTURE(std::string(c.name));
      CAPTURE(g.P);
      CAPTURE(g.n);
      CAPTURE(g.E);
      CAPTURE(g.bcast);
      const int64_t arows = g.bcast ? 1 : g.P;
      std::vector<uint8_t> wq =
          RandomBlocks(c, g.E * g.n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> af(static_cast<size_t>(arows * k));
      GenerateData(1.0F, af.size(), af.data());
      std::vector<int32_t> ids(g.P);
      for (int64_t p = 0; p < g.P; ++p) ids[static_cast<size_t>(p)] = p % g.E;
      const size_t outn = static_cast<size_t>(g.P * g.n);

      // --- CPU golden (the landed grouped keep-quant kernel over host tensors)
      std::vector<float> cpu_out(outn, 1337.0F);
      {
        Tensor at =
            Tensor::Contiguous(af.data(), DType::kF32, Cpu(), {arows, k});
        Tensor wt =
            Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {g.E * g.n, k});
        wt.dtype = c.dtype;
        Tensor et =
            Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {g.P});
        Tensor ot =
            Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {g.P, g.n});
        vt::MatmulBTQuantGrouped(cq, ot, at, wt, et);
      }

      // --- ROCm path over a POISONED output buffer -------------------------
      void* d_a = gpu.Alloc(af.size() * sizeof(float));
      void* d_w = gpu.Alloc(wq.size());
      void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
      void* d_o = gpu.Alloc(outn * sizeof(float));
      std::vector<float> poison(outn, 1337.0F);
      gpu.Copy(gq, d_a, af.data(), af.size() * sizeof(float));
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
      gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
      gpu.Synchronize(gq);
      Tensor at = DevTensor(d_a, DType::kF32, {arows, k});
      Tensor wt = DevTensor(d_w, c.dtype, {g.E * g.n, k});
      Tensor et = DevTensor(d_e, DType::kI32, {g.P});
      Tensor ot = DevTensor(d_o, DType::kF32, {g.P, g.n});
      vt::MatmulBTQuantGrouped(gq, ot, at, wt, et);
      std::vector<float> got(outn, 0.0F);
      gpu.Copy(gq, got.data(), d_o, got.size() * sizeof(float));
      gpu.Synchronize(gq);
      gpu.Free(d_a);
      gpu.Free(d_w);
      gpu.Free(d_e);
      gpu.Free(d_o);

      int poisoned = 0;
      int nonfinite = 0;
      double num = 0, den = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == 1337.0F) ++poisoned;
        if (!std::isfinite(got[i])) ++nonfinite;
        num += (got[i] - cpu_out[i]) * (got[i] - cpu_out[i]);
        den += cpu_out[i] * cpu_out[i];
      }
      const double nmse = den > 0 ? num / den : num;
      CAPTURE(nmse);
      CHECK(poisoned == 0);   // a dispatch that launches nothing lands HERE
      CHECK(nonfinite == 0);
      CHECK(nmse <= kMaxNmseVsCpu);
      ++combos;
    }
  }
  // doctest prints "SUCCESS!" for a loop that never ran. Say how many it ran.
  CAPTURE(combos);
  CHECK(combos ==
        static_cast<int64_t>(std::size(kCases) * std::size(kGroupedShapes)));
  CHECK(combos > 0);
#if defined(VLLM_CPP_HIP)
  const auto mmvq_counts = vt_rocm_test_api::MmvqCounts();
  CHECK(mmvq_counts.baseline == 0);
  CHECK(mmvq_counts.gemv == 0);
  CHECK(mmvq_counts.fused == 0);
  const uint64_t grouped_q8k_routes =
      vt::rocm::Q8KRouteDispatchCountForTest(true, false) +
      vt::rocm::Q8KRouteDispatchCountForTest(true, true);
  CHECK(grouped_q8k_routes == 9);
#endif
  gpu.DestroyQueue(gq);
}
