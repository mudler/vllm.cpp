// Implementation of the persistent device input path as a seam capability. See
// `include/vt/persistent_step_input.h` for what it owns and what it deliberately
// does not, and `.agents/specs/eng-cudagraph-break.md` for the design.
//
// Row ENG-CUDAGRAPH-BREAK W4, issue #1307, parent #1163.
#include "vt/persistent_step_input.h"

#include <atomic>
#include <cstring>

#include "vt/dtype.h"  // VT_CHECK

namespace vt {
namespace {

std::atomic<int64_t> g_binds{0};
std::atomic<int64_t> g_host_refreshes{0};
std::atomic<int64_t> g_device_refreshes{0};

}  // namespace

StepInputStats GetStepInputStats() {
  StepInputStats s;
  s.binds = g_binds.load(std::memory_order_relaxed);
  s.host_refreshes = g_host_refreshes.load(std::memory_order_relaxed);
  s.device_refreshes = g_device_refreshes.load(std::memory_order_relaxed);
  return s;
}

void ResetStepInputStats() {
  g_binds.store(0, std::memory_order_relaxed);
  g_host_refreshes.store(0, std::memory_order_relaxed);
  g_device_refreshes.store(0, std::memory_order_relaxed);
}

PersistentStepInput::~PersistentStepInput() { Unbind(); }

PersistentStepInput::PersistentStepInput(PersistentStepInput&& o) noexcept
    : b_(o.b_),
      device_(o.device_),
      staging_(o.staging_),
      capacity_(o.capacity_),
      host_refreshes_(o.host_refreshes_),
      device_refreshes_(o.device_refreshes_),
      last_source_(o.last_source_) {
  // The moved-from cell must not free the staging block the new owner now holds.
  o.staging_ = nullptr;
  o.device_ = nullptr;
  o.capacity_ = 0;
}

PersistentStepInput& PersistentStepInput::operator=(PersistentStepInput&& o) noexcept {
  if (this != &o) {
    Unbind();
    b_ = o.b_;
    device_ = o.device_;
    staging_ = o.staging_;
    capacity_ = o.capacity_;
    host_refreshes_ = o.host_refreshes_;
    device_refreshes_ = o.device_refreshes_;
    last_source_ = o.last_source_;
    o.staging_ = nullptr;
    o.device_ = nullptr;
    o.capacity_ = 0;
  }
  return *this;
}

void PersistentStepInput::Bind(Backend& b, void* device, size_t capacity, bool staged) {
  VT_CHECK(device != nullptr,
           "PersistentStepInput::Bind: the device destination must not be null; the "
           "cell binds a buffer the CALLER owns and keeps alive across every replay");
  // Release any previous staging block FIRST: a rebind after a shape change must
  // not leak the block the old shape allocated.
  Unbind();
  b_ = &b;
  device_ = device;
  capacity_ = capacity;
  if (staged) staging_ = b.AllocPinned(capacity);
  // A count that outlived the graph it described is a number nobody can trust
  // twice, so the per-cell counters start over. The PROCESS-WIDE ones do not:
  // they answer "did anything in this run use the capability at all", which a
  // rebind does not make untrue.
  host_refreshes_ = 0;
  device_refreshes_ = 0;
  last_source_ = StepInputSource::kUnset;
  g_binds.fetch_add(1, std::memory_order_relaxed);
}

void PersistentStepInput::Unbind() {
  if (staging_ != nullptr && b_ != nullptr) b_->FreePinned(staging_);
  staging_ = nullptr;
  device_ = nullptr;
  capacity_ = 0;
}

void PersistentStepInput::Copy(Queue& q, const void* src, size_t bytes,
                               StepInputSource from) {
  VT_CHECK(bound(),
           "PersistentStepInput: refresh on an UNBOUND cell. A dropped refresh leaves "
           "the previous step's values in whatever the graph reads, which is silently "
           "wrong input rather than a fault");
  // THE ADDRESS-STABILITY RULE. Growing to fit would have to reallocate, and the
  // captured graph goes on reading the address it baked; nothing faults and the
  // tokens still parse. Refuse at the call that would cause it.
  VT_CHECK(bytes <= capacity_,
           "PersistentStepInput: refresh is longer than the bound capacity. The "
           "destination address is baked into a captured graph and cannot grow; "
           "rebind and recapture for the new shape");
  // A ZERO-BYTE REFRESH SKIPS THE COPY AND STILL COUNTS. The counters and
  // `last_source_` below answer "which arm refreshed this cell", which is the
  // observable this capability adds; returning early from here would leave a
  // step that DID choose an arm reporting the previous one, or `kUnset` on the
  // first step, and `host_refreshes` would under-count. The copy is what has
  // nothing to do, not the decision.
  if (bytes > 0) {
    if (from == StepInputSource::kHost && staging_ != nullptr) {
      // Through the PINNED block, so the upload is a true asynchronous DMA rather
      // than an effectively host-synchronous pageable copy — and so the source
      // address a capture bakes belongs to this cell rather than to the caller's
      // frame.
      std::memcpy(staging_, src, bytes);
      b_->Copy(q, device_, staging_, bytes);
    } else {
      b_->Copy(q, device_, src, bytes);
    }
  }
  if (from == StepInputSource::kHost) {
    ++host_refreshes_;
    g_host_refreshes.fetch_add(1, std::memory_order_relaxed);
  } else {
    ++device_refreshes_;
    g_device_refreshes.fetch_add(1, std::memory_order_relaxed);
  }
  last_source_ = from;
}

void PersistentStepInput::RefreshFromHost(Queue& q, const void* src, size_t bytes) {
  VT_CHECK(src != nullptr || bytes == 0,
           "PersistentStepInput::RefreshFromHost: null host source");
  Copy(q, src, bytes, StepInputSource::kHost);
}

void PersistentStepInput::RefreshFromDevice(Queue& q, const void* src, size_t bytes) {
  // A NULL SOURCE IS A REFUSAL AND NOT A NO-OP. A driver reaches this arm because
  // it decided the asynchronous device mirror is live; a silent no-op would leave
  // the PREVIOUS step's identifiers in the destination and the replay would
  // generate from them, which is the exact defect this capability exists to
  // remove (#323: depth-2 graph ON FAIL, slots 1-3 degenerate). Ask the mirror
  // first and take the host arm when it is absent.
  VT_CHECK(src != nullptr,
           "PersistentStepInput::RefreshFromDevice: null device source. Test the "
           "mirror before choosing this arm; a dropped refresh replays the previous "
           "step's values");
  Copy(q, src, bytes, StepInputSource::kDevice);
}

}  // namespace vt
