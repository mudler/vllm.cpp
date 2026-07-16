// Ported from: vllm/v1/worker/gpu/async_utils.py:12-70 (AsyncOutput) +
// vllm/v1/worker/gpu/gpu_model_runner.py:242-332 (AsyncGPUModelRunnerOutput) +
// vllm/v1/outputs.py:298-307 (AsyncModelRunnerOutput.get_output contract) @
// e24d1b24.
//
// Scope (ENG-ASYNC-SCHED W3, sampler-OUTPUT half): the deferred-output seam that
// captures the measured ~3.25 ms/step GPU idle. Under async/overlap scheduling
// the sampled token ids stay GPU-resident (the Sampler writes them into a device
// tensor — see sampler.h `sampled_ids_out`), and instead of the runner blocking
// the MAIN queue on a D2H, an AsyncGPUModelRunnerOutput issues the single copy on
// a dedicated COPY queue with a completion event. The engine calls get_output()
// during OUTPUT PROCESSING — off the model's critical path — where it blocks ONLY
// on that copy event; the main queue proceeds to the next step's forward while
// the copy overlaps.
//
// Device-neutral: on a synchronous/unified backend (CPU, GB10) the copy is a
// memcpy and the events are no-ops (vt::Backend base impls), so get_output()
// returns the byte-identical ModelRunnerOutput the synchronous path would have —
// token-exact by construction. On CUDA the copy queue + cudaEvent overlap the
// forward. The copy runs OUTSIDE any CUDA-graph capture (issued after sampling,
// before the next graph replay), mirroring how upstream keeps AsyncOutput out of
// graphed regions (async-serving spec D5).
#ifndef VLLM_V1_WORKER_GPU_ASYNC_OUTPUT_H_
#define VLLM_V1_WORKER_GPU_ASYNC_OUTPUT_H_

#include <utility>

#include "vllm/v1/engine/types.h"  // ModelRunnerOutput
#include "vt/backend.h"
#include "vt/device.h"

namespace vllm::v1 {

// AsyncModelRunnerOutput (vllm/v1/outputs.py:298-307). The abstract deferred
// result: execute_model/sample_tokens may hand one back instead of a
// materialized ModelRunnerOutput, and the engine calls get_output() (exactly
// once) during output processing to block on the copy and read the tokens.
class AsyncModelRunnerOutput {
 public:
  virtual ~AsyncModelRunnerOutput() = default;
  // Block until the sampled-id copy has completed, then return the fully
  // materialized ModelRunnerOutput. Idempotent-unsafe (call once).
  virtual ModelRunnerOutput get_output() = 0;
};

// ReadyModelRunnerOutput: an AsyncModelRunnerOutput wrapping an ALREADY
// materialized ModelRunnerOutput — get_output() returns it with no wait. This is
// the sync/degenerate case of the async seam: a runner that samples synchronously
// (the production default, or the test stub) wraps its result in one of these, so
// EngineCore::step_with_batch_queue can hold a uniform AsyncModelRunnerOutput and
// call get_output() at consume time without branching. Mirrors uniproc_executor's
// non_block path handing back an immediately-ready future (uniproc_executor.py:101).
class ReadyModelRunnerOutput final : public AsyncModelRunnerOutput {
 public:
  explicit ReadyModelRunnerOutput(ModelRunnerOutput output)
      : output_(std::move(output)) {}
  ModelRunnerOutput get_output() override { return std::move(output_); }

 private:
  ModelRunnerOutput output_;
};

// AsyncGPUModelRunnerOutput (gpu_model_runner.py:242-332 + async_utils.py:12-70).
// Owns the device-resident sampled-id snapshot for the in-flight window (so a
// depth-2 next step may overwrite the runner's working buffers without tearing
// this copy), the pinned host destination, and the completion event.
class AsyncGPUModelRunnerOutput final : public AsyncModelRunnerOutput {
 public:
  // Mirrors AsyncOutput.__init__ (async_utils.py:12-46). `device_sampled_ids` is
  // a backend allocation on `device` holding `num_reqs` int64 sampled ids
  // produced on `main_q` — this object TAKES OWNERSHIP and frees it in the dtor.
  // `skeleton` carries the ordering fields (req_ids / req_id_to_index), known
  // before sampling; get_output() fills in sampled_token_ids. The constructor:
  //   1. records a fork event on main_q and makes copy_q wait it
  //      (copy_stream.wait_stream(default_stream), async_utils.py:29),
  //   2. issues the NON-BLOCKING D2H into an owned pinned host buffer
  //      (async_copy_to_np, :32),
  //   3. records the ready event on copy_q (copy_event.record, :44).
  // The MAIN queue is never synchronized here.
  AsyncGPUModelRunnerOutput(ModelRunnerOutput skeleton, vt::Device device,
                            void* device_sampled_ids, int num_reqs,
                            vt::Queue& main_q, vt::Queue& copy_q);
  ~AsyncGPUModelRunnerOutput() override;

  AsyncGPUModelRunnerOutput(const AsyncGPUModelRunnerOutput&) = delete;
  AsyncGPUModelRunnerOutput& operator=(const AsyncGPUModelRunnerOutput&) = delete;

  // get_output (gpu_model_runner.py:290-332 + async_utils.py:47-70). Blocks the
  // HOST on the ready event, releases the device snapshot, materializes each
  // request's sampled_token_ids [1] from the pinned host buffer (int64 -> int32),
  // and returns the ModelRunnerOutput. Call exactly once.
  ModelRunnerOutput get_output() override;

  // Number of requests in this batch (for the runner's last_sampled scatter and
  // the tests). The pinned host buffer is only valid after get_output()'s event
  // synchronize; before that the device snapshot is authoritative.
  int num_reqs() const { return num_reqs_; }

 private:
  ModelRunnerOutput output_;   // ordering fields set at construction
  vt::Backend* backend_;
  vt::Device device_;
  void* device_sampled_ids_;   // owned: backend allocation on `device_`
  int64_t* pinned_host_;       // owned: AllocPinned host buffer [num_reqs]
  int num_reqs_;
  vt::Event ready_event_;      // completion of the D2H on the copy queue
  bool consumed_ = false;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_ASYNC_OUTPUT_H_
