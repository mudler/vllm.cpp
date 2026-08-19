// THE DECODE-GRAPH SLOT'S TOKEN-ID INPUT, ON THE SHARED SEAM.
//
// Row `ENG-CUDAGRAPH-BREAK`, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1305, with #1179 and #323 as the standing trackers for the decline
// this exists to make unnecessary.
//
// WHAT IT IS. One decode-graph size slot's per-step input identifiers, held in a
// DEVICE buffer whose address does not move for the life of that slot, refreshed
// once per step from the padded host vector and then, when the asynchronous
// device mirror is live, RE-READ over the real prefix from the mirror's own
// device buffer. `vt::PersistentStepInput` (`include/vt/persistent_step_input.h`)
// owns the address-stability rule, the pinned staging block and the counted
// choice of source; this type owns the one thing that seam deliberately does not
// — the destination itself, drawn from the caller's pooled allocator.
//
// WHY IT EXISTS. Before it, three shipped registrations
// (`qwen3_moe_registry.cpp`, `deepseek_v2_registry.cpp`,
// `glm4_moe_lite_registry.cpp`) routed a pure-decode step into a driver that
// embedded from `ModelForwardInput::token_ids` and never looked at
// `device_token_ids`. On the asynchronous serving path the runner's combine
// splices each decode row's sampled token into the DEVICE identifiers on the
// main queue and leaves the host vector deliberately stale
// (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm), so those models
// generated from stale identifiers for every row past the first. Two other
// families already consumed the mirror by hand
// (`qwen3.cpp::ApplyDeviceTokenIdsOverride`, `qwen3_5.cpp`); a third and fourth
// hand-rolled copy is what this header refuses to be.
//
// WHAT IT DOES NOT CLAIM. The embed still sits OUTSIDE the captured region in
// every driver that uses this, because `vt::Embedding` allocates a device
// bounds-check flag and synchronizes the stream — both illegal under capture. So
// the identifiers are read once per step from a stable device address, not from
// inside the replay. The difference matters for exactly one future change
// (moving the embed inside the capture), and this type is the destination that
// change needs; it is not that change. The `qwen3.cpp` decline stays where it
// is: `ENG-CUDAGRAPH-BREAK` W4 (#1307) measured its recorded CAUSE false, and a
// mitigation whose failure mode is unexplained is not retired by a refactor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, MakeTensor
#include "vt/dtype.h"                                      // DType, VT_CHECK
#include "vt/persistent_step_input.h"
#include "vt/tensor.h"

namespace vllm {

class StepTokenIds {
 public:
  // Allocate and BIND a device destination for `capacity` int32 identifiers. A
  // slot calls this once per padded size; a capacity change reallocates, which is
  // why a driver must invalidate the captured graph in the same step (the same
  // obligation the block-table column count already carries).
  void Ensure(dense_attn::Dev d, int64_t capacity) {
    if (buf_ != nullptr && capacity_ == capacity) return;
    // Construct the new block BEFORE releasing the old one, so the pool cannot
    // hand back the address the previous capture baked while it is still bound.
    auto next = std::make_unique<dense_attn::DBuf>(d, vt::DType::kI32,
                                                   std::vector<int64_t>{capacity});
    buf_ = std::move(next);
    capacity_ = capacity;
    view_ = dense_attn::MakeTensor(buf_->ptr(), vt::DType::kI32, d.q.device,
                                   {capacity});
    cell_.Bind(d.b, buf_->ptr(),
               static_cast<size_t>(capacity) * sizeof(int32_t), /*staged=*/true);
  }

  // Refresh this step's identifiers. `padded_host_ids` is the slot's persistent
  // padded host vector, which is authoritative for the inert padding rows and,
  // when no mirror is live, for every row. `device_ids` is the runner's device
  // buffer for THIS step's real prefix, or null when the mirror is off; when it
  // is present the real prefix is re-read from it ON THE QUEUE, so the copy is
  // ordered after the combine that produced it instead of racing it.
  //
  // BOTH copies run every step, in this order, and that is not redundancy: the
  // host arm is what fills the padding rows and the prefill rows the combine
  // never touches, and the device arm is what corrects the decode rows the host
  // vector is stale for.
  void Refresh(dense_attn::Dev d, const std::vector<int32_t>& padded_host_ids,
               const int32_t* device_ids, int64_t device_count) {
    VT_CHECK(cell_.bound(),
             "StepTokenIds::Refresh on an unbound cell; call Ensure() for this "
             "step's padded size first");
    const int64_t T = static_cast<int64_t>(padded_host_ids.size());
    VT_CHECK(T <= capacity_,
             "StepTokenIds::Refresh: the padded host ids are longer than the bound "
             "device destination, whose address a captured graph has baked");
    view_ = dense_attn::MakeTensor(buf_->ptr(), vt::DType::kI32, d.q.device, {T});
    cell_.RefreshFromHost(d.q, padded_host_ids.data(),
                          static_cast<size_t>(T) * sizeof(int32_t));
    if (device_ids == nullptr) return;
    // A device buffer LONGER than this step's input would run past the end, which
    // can only mean the runner and the model disagree about the step's shape.
    // Fail loudly rather than embed past the padding.
    VT_CHECK(device_count <= T,
             "StepTokenIds::Refresh: the mirror's device ids are longer than this "
             "step's padded input");
    cell_.RefreshFromDevice(d.q, device_ids,
                            static_cast<size_t>(device_count) * sizeof(int32_t));
  }

  bool bound() const { return cell_.bound(); }
  int64_t capacity() const { return capacity_; }
  // The tensor to embed from: [T] for the step most recently refreshed.
  const vt::Tensor& t() const { return view_; }
  // WHICH ARM last refreshed this slot. The whole point of routing through the
  // seam rather than writing a fifth private copy: a step that re-read the device
  // mirror and one that uploaded a stale host vector leave the same bytes-shaped
  // destination and the same token count, and no token gate can separate them.
  //
  // WHO READS THESE, stated rather than assumed. Nothing does, yet: `bound()`,
  // `capacity()`, `last_source()`, `device_refreshes()` and `host_refreshes()`
  // have no caller in `src/` or `tests/` — `t()` is the only accessor a caller
  // uses. What `tests/vllm/models/test_moe_async_device_ids.cpp` asserts is the
  // PROCESS-WIDE `vt::GetStepInputStats()`, which the same refreshes move. These
  // five are the per-slot form of the same answer and are kept because a
  // multi-slot driver needs the per-slot one, not because a reader exists.
  vt::StepInputSource last_source() const { return cell_.last_source(); }
  int64_t device_refreshes() const { return cell_.device_refreshes(); }
  int64_t host_refreshes() const { return cell_.host_refreshes(); }

 private:
  std::unique_ptr<dense_attn::DBuf> buf_;  // the destination this type owns
  vt::PersistentStepInput cell_;
  vt::Tensor view_{};
  int64_t capacity_ = 0;
};

}  // namespace vllm
