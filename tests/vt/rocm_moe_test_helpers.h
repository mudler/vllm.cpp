// vllm.cpp original. Scoped device ownership for the BF16 MoE gates (#3094).
#pragma once
#include <doctest/doctest.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <vector>
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#if defined(VLLM_CPP_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace rocm_moe_test {
// The existing Backend allocation/copy helpers use the calling thread's HIP
// device. Scope test resources explicitly so the device-isolation gate measures
// the new operation, with real allocations on each declared device.
class HostDeviceScope {
 public:
  explicit HostDeviceScope(int device) {
#if defined(VLLM_CPP_HIP)
    REQUIRE(hipGetDevice(&previous_) == hipSuccess);
    REQUIRE(hipSetDevice(device) == hipSuccess);
#else
    (void)device;
#endif
  }
  ~HostDeviceScope() {
#if defined(VLLM_CPP_HIP)
    (void)hipSetDevice(previous_);
#endif
  }
  static int Current() {
    int device = 0;
#if defined(VLLM_CPP_HIP)
    REQUIRE(hipGetDevice(&device) == hipSuccess);
#endif
    return device;
  }
 private:
#if defined(VLLM_CPP_HIP)
  int previous_ = 0;
#endif
};
inline vt::Device Device(int index = 0) { return {vt::DeviceType::kROCM, index}; }
inline void RequireDevice(int index = 0) {
  if (vt::TryGetBackend(Device(index)) == nullptr) {
    MESSAGE("required ROCm device is unavailable");
    std::exit(77);
  }
}
inline vt::Tensor View(void* data, vt::DType dtype, vt::Device device,
                       const std::vector<int64_t>& shape) {
  vt::Tensor result;
  result.data = data;
  result.dtype = dtype;
  result.device = device;
  result.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = result.rank - 1; i >= 0; --i) {
    result.shape[i] = shape[static_cast<size_t>(i)];
    result.stride[i] = stride;
    stride *= result.shape[i];
  }
  return result;
}
struct Queue {
  vt::Queue value;
  static vt::Queue Create(int index) {
    HostDeviceScope device(index);
    return vt::CreateQueue(Device(index));
  }
  explicit Queue(int index = 0) : value(Create(index)) {}
  ~Queue() { HostDeviceScope device(value.device.index); vt::DestroyQueue(value); }
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
};
class Buffer {
 public:
  Buffer(vt::Queue& q, vt::DType dtype, const std::vector<int64_t>& shape,
          const void* host = nullptr) : backend_(vt::GetBackend(q.device)) {
    HostDeviceScope device(q.device.index);
    tensor_ = View(nullptr, dtype, q.device, shape);
    bytes_ = static_cast<size_t>(tensor_.Numel()) * vt::SizeOf(dtype);
    tensor_.data = vt::Alloc(q.device, bytes_ + kGuard);
#if defined(VLLM_CPP_HIP)
    hipPointerAttribute_t attributes{};
    REQUIRE(hipPointerGetAttributes(&attributes, tensor_.data) == hipSuccess);
    REQUIRE(attributes.device == q.device.index);
#endif
    // Positive guard bytes make a removed K mask observable even when the
    // allocator otherwise returns zero-filled pages beyond the logical matrix.
    backend_.Memset(q, tensor_.data, 0x55, bytes_ + kGuard);
    if (host != nullptr) backend_.Copy(q, tensor_.data, host, bytes_);
    backend_.Synchronize(q);
  }
  ~Buffer() { HostDeviceScope device(tensor_.device.index); vt::Free(tensor_.device, tensor_.data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  vt::Tensor& tensor() { return tensor_; }
  const vt::Tensor& tensor() const { return tensor_; }
  template <typename T> std::vector<T> Download(vt::Queue& q) const {
    HostDeviceScope device(q.device.index);
    REQUIRE(bytes_ % sizeof(T) == 0);
    std::vector<T> host(bytes_ / sizeof(T));
    backend_.Copy(q, host.data(), tensor_.data, bytes_);
    backend_.Synchronize(q);
    return host;
  }
  void CheckGuard(vt::Queue& q) const {
    HostDeviceScope device(q.device.index);
    std::vector<uint8_t> guard(kGuard);
    backend_.Copy(q, guard.data(), static_cast<const uint8_t*>(tensor_.data) + bytes_, kGuard);
    backend_.Synchronize(q);
    CHECK(std::all_of(guard.begin(), guard.end(), [](uint8_t b) { return b == 0x55; }));
  }
 private:
  static constexpr size_t kGuard = 1024;
  vt::Backend& backend_;
  vt::Tensor tensor_;
  size_t bytes_;
};
}  // namespace rocm_moe_test
