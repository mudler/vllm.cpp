// Ported from: vllm/v1/worker/gpu/async_utils.py:12-70 +
// vllm/v1/worker/gpu/gpu_model_runner.py:242-332 @ e24d1b24. See async_output.h
// for scope, device-neutrality, and the token-exactness argument.
#include "vllm/v1/worker/gpu/async_output.h"

#include <cstdint>
#include <utility>

namespace vllm::v1 {

AsyncGPUModelRunnerOutput::AsyncGPUModelRunnerOutput(
    ModelRunnerOutput skeleton, vt::Device device, void* device_sampled_ids,
    int num_reqs, vt::Queue& main_q, vt::Queue& copy_q)
    : output_(std::move(skeleton)),
      backend_(&vt::GetBackend(device.type)),
      device_(device),
      device_sampled_ids_(device_sampled_ids),
      pinned_host_(nullptr),
      num_reqs_(num_reqs) {
  // NOTE(async_utils.py:20-23): we retain the device tensor reference (owned
  // here) because the copy runs on a DIFFERENT queue than the one that produced
  // it — it must outlive the copy.
  const size_t bytes = static_cast<size_t>(num_reqs_) * sizeof(int64_t);
  pinned_host_ =
      static_cast<int64_t*>(backend_->AllocPinned(bytes == 0 ? 1 : bytes));

  // copy_stream.wait_stream(default_stream) (async_utils.py:29): make the copy
  // queue wait for the sampling work on the main queue, WITHOUT blocking the
  // host. Implemented as record-on-main + queue-wait-on-copy (the building block
  // torch's wait_stream compiles to). No-ops on a synchronous backend.
  vt::Event fork = backend_->CreateEvent();
  backend_->RecordEvent(fork, main_q);
  backend_->QueueWaitEvent(copy_q, fork);
  backend_->DestroyEvent(fork);

  // async_copy_to_np (async_utils.py:32,108-109): the NON-BLOCKING D2H of the
  // sampled ids into pinned host memory. The main queue is never synchronized.
  if (bytes != 0) {
    backend_->Copy(copy_q, pinned_host_, device_sampled_ids_, bytes);
  }

  // copy_event.record(copy_stream) (async_utils.py:44): the completion event the
  // host will later wait on (and only this — never the main queue).
  ready_event_ = backend_->CreateEvent();
  backend_->RecordEvent(ready_event_, copy_q);
}

AsyncGPUModelRunnerOutput::~AsyncGPUModelRunnerOutput() {
  backend_->DestroyEvent(ready_event_);
  if (pinned_host_ != nullptr) backend_->FreePinned(pinned_host_);
  if (device_sampled_ids_ != nullptr) vt::Free(device_, device_sampled_ids_);
}

ModelRunnerOutput AsyncGPUModelRunnerOutput::get_output() {
  // async_utils.py:48 self.copy_event.synchronize(): the ONE blocking wait, and
  // it waits the COPY queue's event — the main queue was never blocked.
  backend_->SynchronizeEvent(ready_event_);

  // gpu_model_runner.py:298-300: release the device snapshot once the copy has
  // landed (safe now — the copy is complete).
  if (device_sampled_ids_ != nullptr) {
    vt::Free(device_, device_sampled_ids_);
    device_sampled_ids_ = nullptr;
  }

  // async_utils.py:53-58 / gpu_model_runner.py:301-304: materialize the ragged
  // per-request sampled_token_ids from the (now host-resident) ids. T0 non-spec
  // decode is exactly one bonus token per request ([num_reqs, 1]).
  output_.sampled_token_ids.assign(static_cast<size_t>(num_reqs_), {});
  for (int i = 0; i < num_reqs_; ++i) {
    output_.sampled_token_ids[static_cast<size_t>(i)] = {
        static_cast<int32_t>(pinned_host_[i])};
  }
  consumed_ = true;
  return std::move(output_);
}

}  // namespace vllm::v1
