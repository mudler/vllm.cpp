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

// Stages a HOST buffer so a DEVICE kernel may dereference it, reusing ONE
// grow-only device allocation across steps.
//
// Same residency contract as DeviceScratch above, and for the same reason —
// but for the biggest tensor in the sampling path rather than the small derived
// ones, so it must not alloc/free per step. The runner assembles the
// [rows, vocab] logits the on-device sampler runs on from a host buffer
// whenever the forward returned ForwardLogits.host (nemotron_h, laguna,
// qwen3_vl — scripts/runner-routing-allowlist.txt), and every sampling op
// dereferences that pointer ON DEVICE: vt::GreedyArgmax's CUDA arm passes it
// straight to a kernel (src/vt/cuda/cuda_sample.cu:199,207), and so do
// ApplyTemperature / ApplyTopKTopP / ComputeProbs / ComputeLogprobs.
//
// On a unified-memory backend a host address is a valid device address, so the
// wrap is in place and free — this is the GB10 path, and it is byte-for-byte
// the expression it replaces. On a DISCRETE backend it is not, and the bytes
// must be copied up first (#1313).
class HostBufferStaging {
 public:
  HostBufferStaging() = default;
  ~HostBufferStaging() { release(); }
  HostBufferStaging(const HostBufferStaging&) = delete;
  HostBufferStaging& operator=(const HostBufferStaging&) = delete;

  // A [shape] view of `host`, dereferenceable by `device`. The returned Tensor
  // is valid until the next Stage() on THIS object, or its destruction.
  vt::Tensor Stage(vt::Device device, vt::Queue& q, const void* host, vt::DType dtype,
                   std::initializer_list<int64_t> shape) {
    vt::Backend& b = vt::GetBackend(device.type);
    if (b.UnifiedMemory()) {
      // Host and device share one address space (CPU, GB10, integrated Vulkan):
      // point straight at the host buffer. const_cast is safe on exactly the
      // terms DeviceScratch states — and note the sampling ops DO mutate this
      // one (temperature / top-k / top-p / the grammar bitmask), in place,
      // which is what the host path has always done. This branch is the
      // expression it replaces, character for character, so GB10 does not move.
      return vt::Tensor::Contiguous(const_cast<void*>(host), dtype, device, shape);
    }

    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(dtype);

    // A backend swap invalidates the allocation: free it against the backend
    // that made it, never against the new one.
    if (backend_ != nullptr && backend_ != &b) release();
    backend_ = &b;
    if (bytes > capacity_) {
      if (owned_ != nullptr) backend_->Free(owned_);
      capacity_ = bytes == 0 ? 1 : bytes;
      owned_ = backend_->Alloc(capacity_);
    }
    if (bytes != 0) backend_->Copy(q, owned_, host, bytes);
    return vt::Tensor::Contiguous(owned_, dtype, device, shape);
  }

  // Drop the staging allocation. Only meaningful on a discrete backend.
  void release() {
    if (owned_ != nullptr) backend_->Free(owned_);
    owned_ = nullptr;
    capacity_ = 0;
    backend_ = nullptr;
  }

 private:
  vt::Backend* backend_ = nullptr;
  void* owned_ = nullptr;
  size_t capacity_ = 0;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_DEVICE_SCRATCH_H_
