// Ported from: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py +
// rejection_sampler_utils.py::rejection_sample @ e24d1b24. See
// include/vllm/v1/spec_decode/rejection_sampler.h for the exact greedy accept
// rule (with upstream file:line for every clause) and the deferred seams.
#include "vllm/v1/spec_decode/rejection_sampler.h"

#include <cstddef>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

namespace {

// Owns a device-side output buffer and downloads it to host. Same shape as the
// sampler's DeviceBuffer (src/vllm/v1/sample/sampler.cpp): on unified backends
// Alloc is host-addressable and Copy is a memcpy, so this works on CPU and CUDA.
class OutBuffer {
 public:
  OutBuffer(vt::Device device, vt::Queue& q, vt::DType dtype,
            std::initializer_list<int64_t> shape)
      : backend_(&vt::GetBackend(device.type)), q_(q) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dtype);
    owned_ = backend_->Alloc(bytes_ == 0 ? 1 : bytes_);
    tensor_ = vt::Tensor::Contiguous(owned_, dtype, device, shape);
  }
  ~OutBuffer() { backend_->Free(owned_); }
  OutBuffer(const OutBuffer&) = delete;
  OutBuffer& operator=(const OutBuffer&) = delete;

  vt::Tensor& tensor() { return tensor_; }
  void download(void* dst) {
    if (bytes_ != 0) backend_->Copy(q_, dst, owned_, bytes_);
    backend_->Synchronize(q_);
  }

 private:
  vt::Backend* backend_ = nullptr;
  vt::Queue& q_;
  void* owned_ = nullptr;
  size_t bytes_ = 0;
  vt::Tensor tensor_;
};

}  // namespace

RejectionSamplerOutput RejectionSampler::forward(
    vt::Queue& q, const vt::Tensor& logits, const std::vector<int32_t>& draft_sampled,
    const std::vector<int32_t>& cu_num_logits,
    const std::vector<char>& is_chunked_prefilling) const {
  VT_CHECK(logits.rank == 2, "rejection_sampler: logits must be [num_logits, vocab]");
  VT_CHECK(logits.dtype == vt::DType::kF32, "rejection_sampler: logits must be f32");
  VT_CHECK(cu_num_logits.size() >= 1,
           "rejection_sampler: cu_num_logits must have num_reqs+1 entries");
  const int64_t num_reqs = static_cast<int64_t>(cu_num_logits.size()) - 1;
  const int64_t num_logits = logits.shape[0];
  VT_CHECK(num_reqs == 0 || cu_num_logits.back() == static_cast<int32_t>(num_logits),
           "rejection_sampler: cu_num_logits.back() must equal the expanded logits rows");
  VT_CHECK(draft_sampled.size() == static_cast<size_t>(num_logits),
           "rejection_sampler: draft_sampled must have one entry per expanded logits row");

  RejectionSamplerOutput out;
  out.sampled_token_ids.resize(static_cast<size_t>(num_reqs));
  out.num_sampled.assign(static_cast<size_t>(num_reqs), 0);
  out.num_rejected.assign(static_cast<size_t>(num_reqs), 0);
  if (num_reqs == 0) return out;

  // `sampled` row width: upstream sizes it num_speculative_steps + 1
  // (rejection_sampler_utils.py:1026-1028). Widen if a request somehow carries
  // more expanded rows than the configured k (defensive; the scheduler clamps).
  int64_t width = static_cast<int64_t>(num_speculative_steps_) + 1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t n = cu_num_logits[static_cast<size_t>(r) + 1] -
                      cu_num_logits[static_cast<size_t>(r)];
    VT_CHECK(n >= 1, "rejection_sampler: every request needs at least the bonus logit row");
    if (n > width) width = n;
  }

  const vt::Device dev = logits.device;
  DeviceScratch draft(dev, q, draft_sampled.data(), vt::DType::kI32, {num_logits});
  DeviceScratch cu(dev, q, cu_num_logits.data(), vt::DType::kI32, {num_reqs + 1});
  OutBuffer sampled(dev, q, vt::DType::kI32, {num_reqs, width});
  OutBuffer num_sampled(dev, q, vt::DType::kI32, {num_reqs});

  vt::GreedyRejectionSample(q, sampled.tensor(), num_sampled.tensor(), logits, draft.tensor(),
                            cu.tensor());

  std::vector<int32_t> host_sampled(static_cast<size_t>(num_reqs * width));
  std::vector<int32_t> host_num_sampled(static_cast<size_t>(num_reqs));
  sampled.download(host_sampled.data());
  num_sampled.download(host_num_sampled.data());

  // get_num_sampled_and_rejected (gpu/input_batch.py:408-453): num_rejected =
  // num_logits - num_sampled; a still-chunked-prefilling row samples nothing and
  // rejects nothing.
  for (int64_t r = 0; r < num_reqs; ++r) {
    const size_t ur = static_cast<size_t>(r);
    const int32_t row_logits = cu_num_logits[ur + 1] - cu_num_logits[ur];
    int32_t ns = host_num_sampled[ur];
    const bool prefilling = ur < is_chunked_prefilling.size() && is_chunked_prefilling[ur] != 0;
    if (prefilling) {
      out.num_sampled[ur] = 0;
      out.num_rejected[ur] = 0;
      continue;
    }
    out.num_sampled[ur] = ns;
    out.num_rejected[ur] = row_logits - ns;
    out.sampled_token_ids[ur].reserve(static_cast<size_t>(ns));
    for (int32_t j = 0; j < ns; ++j) {
      out.sampled_token_ids[ur].push_back(host_sampled[ur * static_cast<size_t>(width) +
                                                       static_cast<size_t>(j)]);
    }
  }
  return out;
}

}  // namespace vllm::v1
