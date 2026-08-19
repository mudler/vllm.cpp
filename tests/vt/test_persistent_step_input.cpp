// THE W4 SEAM-CAPABILITY GATE — `vt::PersistentStepInput`.
//
// Row ENG-CUDAGRAPH-BREAK W4, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1307, parent #1163; it also owns #1305 and #1179.
//
// vllm.cpp original; no upstream mirror. SGLang's BCG suite has no counterpart
// because upstream's persistent step inputs are torch tensors the runner
// refreshes with `copy_`, and the guarantee this file gates — that a per-step
// input's DEVICE ADDRESS does not move, and that the arm which refreshed it is
// observable — is a property of our allocator discipline rather than of theirs.
// vLLM's own equivalent is the persistent input batch
// (`gpu_model_runner.py` `self.input_batch.{positions,slot_mapping,seq_lens}`),
// refreshed once per step into fixed device tensors the captured graph reads;
// that is the STRUCTURE mirrored here, and it has no unit test upstream either.
//
// THE CASE THIS FILE EXISTS FOR is "a device refresh recorded inside a capture
// re-reads the source on every replay, and a host refresh does not". That is the
// difference between the two arms, it is invisible to a token gate and to a
// segment count, and it is the exact defect `qwen3.cpp`'s decline was measured
// against (#323: depth-2 graph ON FAIL, slots 1-3 degenerate). Every other case
// here is a precondition for that one.
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away. The recording
// backend has no real device memory and no asynchrony, so it cannot exhibit the
// RACE between a host read and a device write — only the STALENESS that race
// resolves to. It also cannot show a `cudaMemcpyAsync` from pageable memory
// behaving host-synchronously, which is why the pinned staging block is gated
// here as an allocation and not as a timing.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/persistent_step_input.h"
#include "vt/recording_capture_backend.h"

namespace {

using vt::PersistentStepInput;
using vt::StepInputSource;
using vt_test::RecordingCaptureBackend;

// A device-resident int32 vector on the recording backend.
int32_t* DevVec(RecordingCaptureBackend& b, const std::vector<int32_t>& v) {
  auto* p = static_cast<int32_t*>(b.Alloc(v.size() * sizeof(int32_t)));
  std::memcpy(p, v.data(), v.size() * sizeof(int32_t));
  return p;
}

std::vector<int32_t> Read(const int32_t* p, size_t n) {
  return std::vector<int32_t>(p, p + n);
}

constexpr size_t kN = 4;
constexpr size_t kBytes = kN * sizeof(int32_t);

}  // namespace

TEST_CASE("Bind names a destination the cell does not own, and allocates staging") {
  RecordingCaptureBackend b;
  int32_t* dst = DevVec(b, {0, 0, 0, 0});

  PersistentStepInput in;
  CHECK_FALSE(in.bound());

  in.Bind(b, dst, kBytes, /*staged=*/true);
  CHECK(in.bound());
  CHECK(in.device() == dst);
  CHECK(in.capacity() == kBytes);
  CHECK(in.staged());
  CHECK(in.staging() != nullptr);
  // The staging block is the cell's OWN host memory, never the destination.
  CHECK(in.staging() != static_cast<void*>(dst));
  CHECK(in.host_refreshes() == 0);
  CHECK(in.device_refreshes() == 0);
  CHECK(in.last_source() == StepInputSource::kUnset);

  // An UNSTAGED cell allocates nothing and copies straight from the caller's
  // host address — which is legal, and which bakes that address under capture.
  PersistentStepInput bare;
  bare.Bind(b, dst, kBytes, /*staged=*/false);
  CHECK(bare.bound());
  CHECK_FALSE(bare.staged());
  CHECK(bare.staging() == nullptr);

  in.Unbind();
  CHECK_FALSE(in.bound());
  CHECK(in.staging() == nullptr);
}

TEST_CASE("A refresh moves BYTES and never the destination ADDRESS") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {0, 0, 0, 0});
  int32_t* dev_src = DevVec(b, {7, 7, 7, 7});
  const std::vector<int32_t> host_src = {5, 5, 5, 5};

  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  void* const baked = in.device();

  in.RefreshFromHost(q, host_src.data(), kBytes);
  CHECK(Read(dst, kN) == std::vector<int32_t>{5, 5, 5, 5});
  CHECK(in.device() == baked);  // THE address a captured graph would have baked
  CHECK(in.host_refreshes() == 1);
  CHECK(in.device_refreshes() == 0);
  CHECK(in.last_source() == StepInputSource::kHost);

  in.RefreshFromDevice(q, dev_src, kBytes);
  CHECK(Read(dst, kN) == std::vector<int32_t>{7, 7, 7, 7});
  CHECK(in.device() == baked);
  CHECK(in.host_refreshes() == 1);
  CHECK(in.device_refreshes() == 1);
  CHECK(in.last_source() == StepInputSource::kDevice);
}

// THE ADDRESS-STABILITY RULE, AS A REFUSAL. A cell that grew to fit a longer
// step would have to reallocate, and the captured graph reads the OLD address
// forever. Nothing faults and the tokens still parse, so this has to be loud at
// the call that would cause it rather than discovered on replay N.
TEST_CASE("A refresh LONGER than the bound capacity is refused, on both arms") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {0, 0, 0, 0});
  const std::vector<int32_t> too_long = {1, 2, 3, 4, 5, 6};
  int32_t* dev_too_long = DevVec(b, too_long);

  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  CHECK_THROWS_AS(in.RefreshFromHost(q, too_long.data(), too_long.size() * 4),
                  std::exception);
  CHECK_THROWS_AS(in.RefreshFromDevice(q, dev_too_long, too_long.size() * 4),
                  std::exception);
  // The refusal fires BEFORE any byte moves, so the destination is untouched and
  // no counter advanced.
  CHECK(Read(dst, kN) == std::vector<int32_t>{0, 0, 0, 0});
  CHECK(in.host_refreshes() == 0);
  CHECK(in.device_refreshes() == 0);
}

// The PADDED-PREFIX case. A batched decode driver pads a real batch of B up to a
// captured size S and refreshes only the real prefix, leaving the inert padding
// rows as the capture left them. `ApplyDeviceTokenIdsOverride`
// (`qwen3.cpp:208`) already expresses exactly this as `ov.count <= T`.
TEST_CASE("A SHORTER refresh is legal and leaves the padding tail alone") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {9, 9, 9, 9});
  const std::vector<int32_t> real_prefix = {1, 2};

  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  in.RefreshFromHost(q, real_prefix.data(), real_prefix.size() * sizeof(int32_t));
  CHECK(Read(dst, kN) == std::vector<int32_t>{1, 2, 9, 9});
  CHECK(in.host_refreshes() == 1);
}

// A ZERO-BYTE REFRESH IS STILL A REFRESH, and the counters are the reason this
// matters. `host_refreshes`/`device_refreshes` and `last_source()` exist to
// answer "which arm refreshed this cell", which no token gate and no segment
// count can see. A zero-byte call that returned before touching them would leave
// the cell reporting the PREVIOUS arm — `kUnset` on the first step — over a step
// that did choose an arm, so the observable would under-count exactly the shape
// it was added to make visible. No driver passes zero today (every byte count is
// a positive product and `gdn_state_idx` is guarded by `has_idx`), which is why
// this is a counter defect and not a numerics one: the copy has nothing to move
// either way.
TEST_CASE("A ZERO-BYTE refresh is counted and records its arm") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {7, 7, 7, 7});
  const std::vector<int32_t> src = {1, 2, 3, 4};

  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  vt::ResetStepInputStats();

  in.RefreshFromHost(q, src.data(), 0);
  CHECK(in.host_refreshes() == 1);
  CHECK(in.last_source() == StepInputSource::kHost);
  CHECK(vt::GetStepInputStats().host_refreshes == 1);

  in.RefreshFromDevice(q, dst, 0);
  CHECK(in.device_refreshes() == 1);
  CHECK(in.last_source() == StepInputSource::kDevice);
  CHECK(vt::GetStepInputStats().device_refreshes == 1);

  // And it moved NOTHING, which is the half that must stay true.
  CHECK(Read(dst, kN) == std::vector<int32_t>{7, 7, 7, 7});
}

TEST_CASE("An UNBOUND cell refuses a refresh rather than dropping it") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  const std::vector<int32_t> src = {1, 2, 3, 4};
  PersistentStepInput in;
  CHECK_THROWS_AS(in.RefreshFromHost(q, src.data(), kBytes), std::exception);
  CHECK_THROWS_AS(in.RefreshFromDevice(q, src.data(), kBytes), std::exception);
}

// A NULL DEVICE SOURCE IS A REFUSAL AND NOT A NO-OP, and the distinction is the
// whole hazard. `ModelForwardInput::device_token_ids` is null on every path
// except the asynchronous CUDA runner, so a driver asks "is the mirror live?"
// and takes one arm or the other. A `RefreshFromDevice(nullptr)` that quietly
// did nothing would leave the PREVIOUS step's identifiers in the destination —
// stale input, no fault, correct-looking tokens for one row and garbage after —
// which is the failure this capability exists to remove, reintroduced at the
// seam.
TEST_CASE("RefreshFromDevice refuses a null source") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {3, 3, 3, 3});
  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  CHECK_THROWS_AS(in.RefreshFromDevice(q, nullptr, kBytes), std::exception);
  CHECK(Read(dst, kN) == std::vector<int32_t>{3, 3, 3, 3});
  CHECK(in.device_refreshes() == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// THE CASE THE CAPABILITY EXISTS FOR.
// ───────────────────────────────────────────────────────────────────────────
//
// A refresh issued INSIDE a capture is recorded, so it runs on every REPLAY,
// re-reading whatever its source holds at that moment. That is the decline's own
// wording for the fix — "read the identifiers at REPLAY time from a stable
// device buffer" (`qwen3.cpp`'s `DenseDecodeGraphForward`) — made executable.
//
// Both arms are asserted in ONE case, because asserting only the device arm
// would stay green if the two arms had collapsed into each other. The host arm
// copies through the cell's PINNED STAGING BLOCK, whose contents were fixed when
// the refresh ran; a later write to the caller's own host vector therefore does
// NOT reach a replay. The device arm bakes the SOURCE ADDRESS; a later write to
// that device buffer DOES. Stale versus fresh, on the same captured graph, in
// the same process.
TEST_CASE("A DEVICE refresh inside a capture re-reads the source on replay; a HOST one replays stale bytes") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();

  int32_t* host_dst = DevVec(b, {0, 0, 0, 0});
  int32_t* dev_dst = DevVec(b, {0, 0, 0, 0});
  int32_t* dev_src = DevVec(b, {10, 10, 10, 10});
  std::vector<int32_t> host_src = {10, 10, 10, 10};

  PersistentStepInput from_host;
  PersistentStepInput from_dev;
  from_host.Bind(b, host_dst, kBytes, /*staged=*/true);
  from_dev.Bind(b, dev_dst, kBytes, /*staged=*/true);

  // CAPTURE. Both refreshes are issued inside the capture, so neither has moved
  // a byte yet — exactly as a real `cudaMemcpyAsync` inside a stream capture
  // records the copy without performing it.
  b.BeginCapture(q);
  from_host.RefreshFromHost(q, host_src.data(), kBytes);
  from_dev.RefreshFromDevice(q, dev_src, kBytes);
  void* graph = b.EndCaptureGraph(q);
  CHECK(Read(host_dst, kN) == std::vector<int32_t>{0, 0, 0, 0});
  CHECK(Read(dev_dst, kN) == std::vector<int32_t>{0, 0, 0, 0});

  // REPLAY 1: both destinations receive the capture-time values.
  b.ReplayGraph(q, graph);
  CHECK(Read(host_dst, kN) == std::vector<int32_t>{10, 10, 10, 10});
  CHECK(Read(dev_dst, kN) == std::vector<int32_t>{10, 10, 10, 10});

  // THE NEXT STEP arrives. The asynchronous combine patches the DEVICE ids on
  // the queue; the caller's host vector is deliberately left stale, because
  // materializing it on the host is the synchronize the async path removes.
  for (size_t i = 0; i < kN; ++i) dev_src[i] = 42;
  host_src.assign(kN, 42);

  // REPLAY 2. The device arm sees 42, because the capture baked the SOURCE
  // ADDRESS and re-reads it. The host arm still sees 10, because the bytes it
  // replays were fixed in the staging block at capture time — which is precisely
  // the "replays against stale HOST token ids" defect, reproduced here.
  b.ReplayGraph(q, graph);
  CHECK(Read(dev_dst, kN) == std::vector<int32_t>{42, 42, 42, 42});
  CHECK(Read(host_dst, kN) == std::vector<int32_t>{10, 10, 10, 10});

  // REPLAY 3, because a single replay cannot distinguish a correct read from one
  // that happens to find a buffer nothing has overwritten yet. Spec `## Gates`
  // G1 requires more than one replay for the same reason.
  for (size_t i = 0; i < kN; ++i) dev_src[i] = 77;
  b.ReplayGraph(q, graph);
  CHECK(Read(dev_dst, kN) == std::vector<int32_t>{77, 77, 77, 77});
  CHECK(Read(host_dst, kN) == std::vector<int32_t>{10, 10, 10, 10});

  CHECK(from_dev.device_refreshes() == 1);   // ONE issue, three replays
  CHECK(from_host.host_refreshes() == 1);
  b.DestroyGraph(graph);
}

// AN UNSTAGED HOST REFRESH BAKES THE CALLER'S ADDRESS — spec `## Risks/decisions`
// D2, stated at the declaration and gated here so the two host arms cannot be
// confused. This is not a second way to be fresh: it is the reason the staging
// block exists, because a caller whose host vector reallocates between capture
// and replay reads freed host memory instead of stale bytes.
TEST_CASE("An UNSTAGED host refresh bakes the caller's own host address") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {0, 0, 0, 0});
  std::vector<int32_t> host_src = {1, 1, 1, 1};

  PersistentStepInput in;
  in.Bind(b, dst, kBytes, /*staged=*/false);
  b.BeginCapture(q);
  in.RefreshFromHost(q, host_src.data(), kBytes);
  void* graph = b.EndCaptureGraph(q);

  b.ReplayGraph(q, graph);
  CHECK(Read(dst, kN) == std::vector<int32_t>{1, 1, 1, 1});
  host_src.assign(kN, 2);  // mutated IN PLACE: the address is unchanged
  b.ReplayGraph(q, graph);
  CHECK(Read(dst, kN) == std::vector<int32_t>{2, 2, 2, 2});
  b.DestroyGraph(graph);
}

TEST_CASE("The process-wide counters see both arms, and a rebind clears the cell's own") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  int32_t* dst = DevVec(b, {0, 0, 0, 0});
  int32_t* dev_src = DevVec(b, {1, 1, 1, 1});
  const std::vector<int32_t> host_src = {2, 2, 2, 2};

  vt::ResetStepInputStats();
  CHECK(vt::GetStepInputStats().binds == 0);

  PersistentStepInput in;
  in.Bind(b, dst, kBytes);
  in.RefreshFromHost(q, host_src.data(), kBytes);
  in.RefreshFromDevice(q, dev_src, kBytes);
  in.RefreshFromDevice(q, dev_src, kBytes);

  const vt::StepInputStats s = vt::GetStepInputStats();
  CHECK(s.binds == 1);
  CHECK(s.host_refreshes == 1);
  CHECK(s.device_refreshes == 2);
  CHECK(in.host_refreshes() == 1);
  CHECK(in.device_refreshes() == 2);

  // A shape change forces a recapture, so the driver rebinds. A count that
  // outlived the graph it described is a number nobody can trust twice.
  in.Bind(b, dst, kBytes);
  CHECK(in.host_refreshes() == 0);
  CHECK(in.device_refreshes() == 0);
  CHECK(in.last_source() == StepInputSource::kUnset);
  CHECK(vt::GetStepInputStats().binds == 2);  // process-wide, NOT reset by a rebind
}
