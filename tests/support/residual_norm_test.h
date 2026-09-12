// vllm.cpp original test harness for the shared residual-expression contract.
#pragma once
#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"
#if defined(VLLM_CPP_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace residual_norm_test {
inline std::vector<vt::Device> Devices() {
  std::vector<vt::Device> result{{vt::DeviceType::kCPU, 0}};
#if defined(VLLM_CPP_HIP)
  REQUIRE(vt::TryGetBackend({vt::DeviceType::kROCM, 0}) != nullptr);
  result.push_back({vt::DeviceType::kROCM, 0});
#endif
  return result;
}
struct DeviceScope {
  int previous = 0;
  bool rocm;
  explicit DeviceScope(vt::Device device) : rocm(device.type == vt::DeviceType::kROCM) {
#if defined(VLLM_CPP_HIP)
    if (rocm) { REQUIRE(hipGetDevice(&previous) == hipSuccess); REQUIRE(hipSetDevice(device.index) == hipSuccess); }
#endif
  }
  ~DeviceScope() {
#if defined(VLLM_CPP_HIP)
    if (rocm) (void)hipSetDevice(previous);
#endif
  }
};
struct Queue {
  vt::Queue q;
  explicit Queue(vt::Device device) { DeviceScope scope(device); q = vt::CreateQueue(device); }
  ~Queue() { DeviceScope scope(q.device); vt::DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor t;
  size_t count;
  static constexpr uint16_t kGuard = 0x3555;
  Buffer(vt::Queue& queue, const std::vector<uint16_t>& values,
         int64_t rows, int64_t width, int64_t stride = 0) : q(queue), count(values.size()) {
    DeviceScope scope(q.device);
    auto& backend = vt::GetBackend(q.device);
    void* data = backend.Alloc((count + 64) * sizeof(uint16_t));
    t = rows < 0 ? vt::Tensor::Contiguous(data, vt::DType::kBF16, q.device, {width})
                 : vt::Tensor::Contiguous(data, vt::DType::kBF16, q.device, {rows, width});
    if (rows >= 0) t.stride[0] = stride == 0 ? width : stride;
    std::vector<uint16_t> storage(values);
    storage.resize(count + 64, kGuard);
    backend.Copy(q, data, storage.data(), storage.size() * sizeof(uint16_t));
    backend.Synchronize(q);
  }
  ~Buffer() { DeviceScope scope(q.device); vt::GetBackend(q.device).Free(t.data); }
  std::vector<uint16_t> Read(bool guard = false) const {
    DeviceScope scope(q.device);
    std::vector<uint16_t> result(count + (guard ? 64 : 0));
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, result.data(), t.data, result.size() * sizeof(uint16_t));
    backend.Synchronize(q);
    return result;
  }
  void CheckGuard() const {
    const auto values = Read(true);
    CHECK(std::all_of(values.begin() + static_cast<ptrdiff_t>(count), values.end(),
                      [](uint16_t value) { return value == kGuard; }));
  }
};
// Independent scalar reference: FP64 variance accumulation deliberately does
// not duplicate the native 256-lane tree. Exact boundary witnesses use values
// whose squares and sums are exact; general inputs use the pinned tolerance.
inline std::vector<uint16_t> Reference(const std::vector<uint16_t>& a,
    const std::vector<uint16_t>& base, const std::vector<uint16_t>* delta,
    const std::vector<uint16_t>& gamma, int64_t rows, int64_t width, int64_t stride,
    float epsilon, std::vector<uint16_t>* residual = nullptr) {
  std::vector<uint16_t> out(static_cast<size_t>(rows * stride), Buffer::kGuard);
  if (residual != nullptr) residual->assign(out.size(), Buffer::kGuard);
  for (int64_t row = 0; row < rows; ++row) {
    std::vector<float> sum(static_cast<size_t>(width));
    double variance = 0;
    for (int64_t j = 0; j < width; ++j) {
      const size_t at = static_cast<size_t>(row * stride + j);
      const float pair = vt::BF16ToF32(a[at]) + vt::BF16ToF32(base[at]);
      const float value = delta == nullptr ? pair : vt::BF16ToF32((*delta)[at]) + pair;
      sum[j] = value;
      variance += static_cast<double>(value) * value;
    }
    const float inverse = 1.0f / std::sqrt(static_cast<float>(variance / width) + epsilon);
    for (int64_t j = 0; j < width; ++j) {
      const size_t at = static_cast<size_t>(row * stride + j);
      out[at] = vt::F32ToBF16((sum[j] * inverse) * vt::BF16ToF32(gamma[j]));
      if (residual != nullptr) (*residual)[at] = vt::F32ToBF16(sum[j]);
    }
  }
  return out;
}
inline void Run(int mode, vt::Queue& q, vt::Tensor& out, const vt::Tensor& a,
                const vt::Tensor& base, const vt::Tensor* delta, const vt::Tensor& gamma,
                const vt::ResidualRmsNormArgs& args, vt::Tensor* residual) {
  if (mode == 0) { vt::ResidualRmsNorm(q, out, a, base, delta, gamma, args, residual); return; }
  if (mode == 1) { vt::FusedChain(q, out, a, base, delta, gamma, args, residual); return; }
  vt::FusedBinding binding{};
  binding.n = 6;
  binding.op[0] = const_cast<vt::Tensor*>(&a);
  binding.op[1] = const_cast<vt::Tensor*>(&base);
  binding.op[2] = const_cast<vt::Tensor*>(delta);
  binding.op[3] = const_cast<vt::Tensor*>(&gamma);
  binding.op[4] = &out;
  binding.op[5] = residual;
  vt::FusedParams params{}; params.eps = args.eps;
  vt::FusedChainComposite(q, vt::ResidualRmsNormRecipe(args.descriptor), binding, params);
}
}  // namespace residual_norm_test
