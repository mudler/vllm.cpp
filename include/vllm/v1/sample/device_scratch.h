// vllm.cpp original (no upstream mirror) — a vt-runtime support helper for the
// M1.7 Task 3 sampling-op wrappers.
//
// The V1 penalty / mask / builtin-proc entry points read ragged per-request
// state from the host SamplingMetadata (token-id lists, sparse bias maps) and
// build the small derived tensors (bin-count / mask matrices, (req, token)
// scatter pair-lists) the vt ops consume. Those derived tensors must live on the
// SAME device as the logits. Upstream builds them with torch (async_tensor_h2d /
// scatter_add on the device); here DeviceScratch owns that materialization:
//   - unified-memory backends (CPU, GB10) wrap the host buffer in place (0-copy);
//   - discrete backends alloc device memory and copy the host buffer up,
//     freeing it in the destructor.
#ifndef VLLM_V1_SAMPLE_DEVICE_SCRATCH_H_
#define VLLM_V1_SAMPLE_DEVICE_SCRATCH_H_

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// Materializes a host buffer as a contiguous Tensor on `device`. The Tensor
// (view via tensor()) is valid for the DeviceScratch's lifetime.
class DeviceScratch {
 public:
  DeviceScratch(vt::Device device, vt::Queue& q, const void* host, vt::DType dtype,
                std::initializer_list<int64_t> shape)
      : backend_(&vt::GetBackend(device.type)) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dtype);
    if (backend_->UnifiedMemory()) {
      // Host and device share one address space: point straight at the host
      // buffer (const_cast is safe — the ops treat inputs as read-only).
      tensor_ = vt::Tensor::Contiguous(const_cast<void*>(host), dtype, device, shape);
    } else {
      owned_ = backend_->Alloc(bytes_ == 0 ? 1 : bytes_);
      if (bytes_ != 0) backend_->Copy(q, owned_, host, bytes_);
      tensor_ = vt::Tensor::Contiguous(owned_, dtype, device, shape);
    }
  }
  ~DeviceScratch() {
    if (owned_ != nullptr) backend_->Free(owned_);
  }
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;

  vt::Tensor& tensor() { return tensor_; }

 private:
  vt::Backend* backend_ = nullptr;
  void* owned_ = nullptr;
  size_t bytes_ = 0;
  vt::Tensor tensor_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_DEVICE_SCRATCH_H_
