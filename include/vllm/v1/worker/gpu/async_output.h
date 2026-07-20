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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/v1/engine/types.h"  // ModelRunnerOutput
#include "vt/backend.h"
#include "vt/device.h"

namespace vllm::v1 {

// AsyncOutputSlot — one reusable set of the per-step overlap resources: the
// device sampled-id buffer the sampler's argmax writes, the pinned host buffer
// the D2H lands in, and the two CUDA events (fork = copy-queue-waits-main, ready
// = D2H completion). Allocated ONCE (pool ctor / grow) and reused across steps;
// NEVER a per-step cudaMalloc/cudaHostAlloc/cudaEventCreate.
//
// WHY (ENG-ASYNC-SCHED W3 throughput lever): the original AsyncGPUModelRunner
// Output OWNED a fresh device buffer + pinned host buffer + two events per step
// and freed them in get_output()/dtor. On CUDA cudaMalloc/cudaFree/cudaHostAlloc/
// cudaFreeHost each SYNCHRONIZE the whole device (block the host until every
// launched kernel — including the NEXT step's forward — finishes), which
// serialized the depth-2 overlap the copy stream was built to deliver
// (measured: throughput-neutral W3 despite a −5.4 ms/step TPOT visibility gain).
// vLLM never pays this: its sampled ids come from torch's caching device
// allocator and its pinned D2H destination from torch's caching pinned allocator
// (both pool hits, no raw driver sync), and torch.Event objects are reused
// (async_utils.py:12-70 + the persistent runner buffers). This pool mirrors that
// "no per-step raw alloc/free" property.
struct AsyncOutputSlot {
  void* device_sampled_ids = nullptr;  // capacity int64 elements on `device`
  int64_t* pinned_host = nullptr;      // capacity int64 pinned host elements
  vt::Event fork_event{};              // main->copy ordering (created once)
  vt::Event ready_event{};             // D2H completion (created once)
  int capacity = 0;                    // element capacity (>= max_num_reqs)
  bool in_use = false;
};

// AsyncOutputPool — owns the persistent AsyncOutputSlots. Acquire() hands out a
// free slot (growing by one on the rare miss); Release() returns it. Everything
// is freed ONCE in the dtor (runner teardown), never on the hot path. Single-
// threaded by contract (the one engine thread owns the runner). Device-neutral:
// on the CPU/synchronous backend the buffers are plain host allocations and the
// events are null-handle no-ops (vt::Backend base impls), so the pool is inert
// but structurally identical.
class AsyncOutputPool {
 public:
  // `capacity_elems` sizes every slot's buffers (== max_num_reqs, the batch
  // bound); `initial_slots` are pre-allocated (>= the max concurrent batches in
  // flight = depth-2, plus grammar-deferral headroom).
  AsyncOutputPool(vt::Device device, int capacity_elems, int initial_slots);
  ~AsyncOutputPool();
  AsyncOutputPool(const AsyncOutputPool&) = delete;
  AsyncOutputPool& operator=(const AsyncOutputPool&) = delete;

  AsyncOutputSlot* Acquire();
  void Release(AsyncOutputSlot* slot);

  int capacity_elems() const { return capacity_elems_; }
  int num_slots() const { return static_cast<int>(slots_.size()); }

 private:
  std::unique_ptr<AsyncOutputSlot> MakeSlot();

  vt::Device device_;
  vt::Backend* backend_;
  int capacity_elems_;
  std::vector<std::unique_ptr<AsyncOutputSlot>> slots_;
};

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
// BORROWS a persistent AsyncOutputSlot from the runner's AsyncOutputPool for the
// in-flight window (device sampled-id snapshot + pinned host destination + the
// two events) and RELEASES it back on consume — it does NOT own or free any
// device/pinned/event resource, so the hot path has NO per-step
// cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost/cudaEventCreate/Destroy. The
// slot's device buffer holds the sampler's argmax ids (written on `main_q`
// before construction); the pool guarantees it is not reused until this object
// releases it, so a depth-2 next step can never tear this copy.
class AsyncGPUModelRunnerOutput final : public AsyncModelRunnerOutput {
 public:
  // Mirrors AsyncOutput.__init__ (async_utils.py:12-46). `slot` is a pool slot
  // whose `device_sampled_ids` already holds this batch's `num_reqs` int64 argmax
  // ids (produced on `main_q`); `pool` receives the slot back on consume/destroy.
  // `skeleton` carries the ordering fields (req_ids / req_id_to_index), known
  // before sampling; get_output() fills in sampled_token_ids. The constructor:
  //   1. records the slot's fork event on main_q and makes copy_q wait it
  //      (copy_stream.wait_stream(default_stream), async_utils.py:29),
  //   2. issues the NON-BLOCKING D2H into the slot's pinned host buffer
  //      (async_copy_to_np, :32),
  //   3. records the slot's ready event on copy_q (copy_event.record, :44).
  // The MAIN queue is never synchronized here.
  AsyncGPUModelRunnerOutput(ModelRunnerOutput skeleton, vt::Device device,
                            AsyncOutputPool& pool, AsyncOutputSlot* slot,
                            int num_reqs, vt::Queue& main_q, vt::Queue& copy_q,
                            std::vector<int32_t> invalid_req_indices = {});
  ~AsyncGPUModelRunnerOutput() override;

  AsyncGPUModelRunnerOutput(const AsyncGPUModelRunnerOutput&) = delete;
  AsyncGPUModelRunnerOutput& operator=(const AsyncGPUModelRunnerOutput&) = delete;

  // get_output (gpu_model_runner.py:290-332 + async_utils.py:47-70). Blocks the
  // HOST on the ready event, materializes each request's sampled_token_ids [1]
  // from the pinned host buffer (int64 -> int32), RELEASES the slot back to the
  // pool, and returns the ModelRunnerOutput. Call exactly once.
  ModelRunnerOutput get_output() override;

  // Number of requests in this batch (for the runner's last_sampled scatter and
  // the tests). The pinned host buffer is only valid after get_output()'s event
  // synchronize; before that the device snapshot is authoritative.
  int num_reqs() const { return num_reqs_; }

 private:
  void ReleaseSlot();

  ModelRunnerOutput output_;   // ordering fields set at construction
  vt::Backend* backend_;
  vt::Device device_;
  AsyncOutputPool* pool_;      // borrowed: returns `slot_` on consume/destroy
  AsyncOutputSlot* slot_;      // borrowed: persistent device+pinned buffers+events
  int num_reqs_;
  // discard_request_mask indices (gpu_model_runner.py:3625-3628): the scheduled
  // requests still consuming known prefill tokens this step, whose sampled tokens
  // get_output() must CLEAR to empty (vllm/v1/outputs.py:303-304
  // valid_sampled_token_ids[i].clear()).
  std::vector<int32_t> invalid_req_indices_;
  bool consumed_ = false;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_ASYNC_OUTPUT_H_
