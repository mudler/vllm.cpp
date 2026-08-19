// THE PERSISTENT DEVICE INPUT PATH, AS A SEAM CAPABILITY.
//
// Row ENG-CUDAGRAPH-BREAK W4, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1307, parent #1163; it also owns #1305 and #1179.
//
// WHAT THIS IS. One capture slot's per-step input, held in a device buffer whose
// ADDRESS does not move for the life of the captured graph, refreshed IN PLACE
// once per step from either a persistent host staging block or a DEVICE source.
// A captured graph bakes the addresses it read at capture time; a per-step input
// that is reallocated each step therefore cannot be read by a replay at all, and
// one that is refreshed from a host vector is only ever as fresh as that vector.
//
// WHY IT IS HERE AND NOT IN A MODEL. The capability existed in the tree already,
// as `StepDevInputs` in `src/vllm/model_executor/models/qwen3_5.cpp`, and in
// exactly one driver. Measured with `grep -c StepDevInputs <file>` at this
// commit: 44 in `qwen3_5.cpp` (41 before this row added its own comments), 0 in
// `qwen3_moe.cpp`, `deepseek_v2.cpp`, `voxtral.cpp` and `qwen3_dflash.cpp`, and
// 1 in `qwen3.cpp` — which is a COMMENT naming the fix it does not have, not a
// use. That last number is the instrument's limit rather than a correction:
// `grep -c` counts matching LINES, prose included, so it can only ever bound
// the answer from above. That asymmetry is not tidiness: the four
// drivers without it replay against HOST vectors, and one of them,
// `qwen3.cpp`'s `DenseDecodeGraphForward`, DECLINES its decode graph outright
// whenever the asynchronous device-token mirror is live, on a measured battery
// (depth-1 graph ON PASS 78/78, depth-2 graph OFF PASS 82/82, depth-2 graph ON
// FAIL with slots 1-3 degenerate; #323, #1179). A shipped model lost graphed
// decode because a capability written once was unavailable to it. Writing an
// eighth copy would be the defect, so the capability moves to the seam.
//
// WHAT IT DELIBERATELY DOES NOT OWN. Not the device allocation. Every driver
// already draws its persistent inputs from an allocator with its own discipline
// — `Qwen3_5DecodeGraph` uses a DEDICATED `DevicePool` precisely so a retained
// buffer never pops a block the captured forward's own scratch then needs, and a
// seam that took that allocation over would silently move nine drivers onto one
// pool. It BINDS a destination the caller owns. Nor does it own WHICH inputs a
// model has, their shapes, or their dtypes: that is model knowledge, and a seam
// that interpreted it would become a second dispatcher with a per-model branch
// (spec `## Risks/decisions` D3).
//
// WHAT IT DOES OWN, which is the part every driver re-derives:
//
//   * the address-stability rule, as a REFUSAL rather than a comment;
//   * the pinned host staging block that makes the upload a true async DMA
//     rather than a host-synchronous pageable copy;
//   * the choice of SOURCE, host or device, as an observable rather than an
//     inference from which line the driver happened to call.
//
// COVERAGE AND CORRECTNESS, NEVER SPEED. This row claims no throughput and
// measures none. GB10 measured prefill idle between launches at 3.8% with
// GPU-busy above 96%, and the 27B prefill gap at 92.5% non-GEMM glue GPU work;
// decode is already captured and already banked its launch-overhead win.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/backend.h"

namespace vt {

// WHICH SOURCE last refreshed a cell. This is an OBSERVABLE and not a
// convenience, and W3 is why it exists: that stage shipped a `kFull`/`kPiecewise`
// distinction that no gate could see, and flipping one token left a whole driver
// gate GREEN at 226/226. The host and device arms here are the same shape of
// difference — both leave the destination holding bytes, both leave every
// pointer unmoved, and only one of them is correct while an asynchronous device
// mirror is live. A guard nobody can read is a mute switch, so the arm is
// counted where it is decided.
enum class StepInputSource {
  kUnset,   // bound, never refreshed
  kHost,    // RefreshFromHost: as fresh as the host vector, and no fresher
  kDevice,  // RefreshFromDevice: re-reads the live device source
};

// Process-wide counters, the `vt::GraphBreakStats` sibling for this capability.
// Segment counts tell a two-segment capture from an eager step; these tell a
// step that re-read the device mirror from one that uploaded a stale host
// vector, which no token gate and no segment count can distinguish.
//
// A ZERO-BYTE refresh counts, and sets `last_source()`. What it skips is the
// COPY; the arm was still chosen, and a counter that missed it would under-count
// exactly the thing it exists to make visible.
struct StepInputStats {
  int64_t binds = 0;
  int64_t host_refreshes = 0;
  int64_t device_refreshes = 0;
};
StepInputStats GetStepInputStats();
void ResetStepInputStats();

// ---------------------------------------------------------------------------
// vt::PersistentStepInput — one capture-stable per-step device input.
// ---------------------------------------------------------------------------
//
// LIFETIME. The bound device destination must outlive every replay of the graph
// captured over it, and this object must not outlive that destination. It is the
// same contract `vt::BreakSlot` enforces for a break point's output, applied to
// a step's INPUT, and it is stated rather than enforced by the type for one
// reason: the destination is the driver's own pooled buffer, and taking
// ownership of it is exactly what the paragraph above refuses to do.
//
// THE TWO REFRESH ARMS ARE NOT INTERCHANGEABLE, and the difference is the whole
// point of this type.
//
//   * `RefreshFromHost` copies through a pinned staging block into the bound
//     device address. Correct whenever the host vector is authoritative. Under
//     an asynchronous scheduler it is NOT: the runner's device combine splices
//     each decode row's sampled token into the DEVICE ids on the main queue
//     while the host vector stays deliberately stale, because materializing it
//     on the host is the synchronize the async path exists to remove.
//   * `RefreshFromDevice` copies from a device source into the same bound
//     address, on the queue, so it is ordered AFTER whatever produced that
//     source. This is the arm the `qwen3.cpp` decline's own comment names as the
//     real fix: read the identifiers from a stable device buffer instead of
//     racing a host read against a device write.
//
//     **THIS ARM LANDS WITH NO PRODUCTION CALLER, named rather than implied.**
//     `grep -rn RefreshFromDevice src/ include/` returns its definition and
//     nothing else, and `last_source()`/`StepInputSource` have no production
//     reader either. It is a staged slice under AGENTS.md's "Nothing lands
//     dead": the capability the decline needs is the DESTINATION — a device
//     token-id buffer the captured graph reads — and no driver has one (see the
//     decline in `qwen3.cpp`'s `DenseDecodeGraphForward`). Writing this arm's
//     caller before that destination exists would be the tenth hand-rolled copy
//     this row removes. Owner: row `ENG-CUDAGRAPH-BREAK`, the stage that gets a
//     `dgx` window WITH the Qwen3-0.6B/4B checkpoints the battery needs; listed
//     under `## Owed` in `.agents/specs/eng-cudagraph-break.md` and tracked by
//     #1179 and #323. The HOST arm IS reached, and its own reach is bounded
//     rather than claimed whole: `qwen3_5.cpp`'s `StageStepInputs` routes both
//     Qwen3.5 decode drivers through it whenever `dbuf` is set, which is
//     `VT_ASYNC_EXECUTOR=1` (default OFF, `docs/ENVIRONMENT.md`) or a
//     speculative verify step. `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`
//     holds that call site with a mutation that reds only the reachability case.
//
// WHERE THE CALL SITE SITS decides what a REPLAY sees, and both placements are
// legal:
//
//   * OUTSIDE the captured region, once per step, before the replay. The graph
//     reads the bound address, and the refresh that ran this step is what it
//     finds there. This is what `StageStepInputs` (`qwen3_5.cpp`) does today and
//     it is the placement a batched decode driver wants, because the copy is
//     then not part of the graph and the input-staged event can be recorded
//     right after it.
//   * INSIDE the captured region. Capture then bakes the SOURCE address as well
//     as the destination, and every replay re-reads whatever that source holds
//     at replay time. For `RefreshFromDevice` that is the strongest form of the
//     fix — the identifiers are read at REPLAY time, which is the decline's own
//     wording. For `RefreshFromHost` it is spec `## Risks/decisions` D2, and the
//     host block must then be persistent, which is exactly why the staging block
//     below belongs to this object and not to the caller's frame.
class PersistentStepInput {
 public:
  PersistentStepInput() = default;
  ~PersistentStepInput();
  PersistentStepInput(const PersistentStepInput&) = delete;
  PersistentStepInput& operator=(const PersistentStepInput&) = delete;
  PersistentStepInput(PersistentStepInput&& o) noexcept;
  PersistentStepInput& operator=(PersistentStepInput&& o) noexcept;

  // Bind a device destination the CALLER owns and keeps alive. `capacity` is the
  // size the captured graph will read, in bytes; no refresh may exceed it, which
  // is what makes the address stable by construction rather than by convention.
  //
  // `staged` allocates a pinned host block of the same size, so
  // `RefreshFromHost` is a true asynchronous DMA. A pageable host source makes
  // `cudaMemcpyAsync` effectively host-synchronous, which is why the driver this
  // capability comes from allocates one per slot — and why a poison test that
  // corrupted a pageable source could never reproduce its own hazard.
  //
  // Re-binding is legal and is what a driver does when a shape change forces a
  // recapture. It resets the per-cell counters, because a count that outlived
  // the graph it described is a number nobody can trust twice.
  void Bind(Backend& b, void* device, size_t capacity, bool staged = true);

  // Release the staging block and forget the destination. The destination itself
  // is the caller's and is not freed here.
  void Unbind();

  bool bound() const { return device_ != nullptr; }
  void* device() const { return device_; }
  size_t capacity() const { return capacity_; }
  bool staged() const { return staging_ != nullptr; }
  void* staging() const { return staging_; }

  // Copy `bytes` from a HOST source into the bound destination, through the
  // pinned staging block when this cell is staged. `bytes` may be SHORTER than
  // the capacity: a padded decode step refreshes only its real prefix and leaves
  // the inert padding rows as the capture left them, which is what
  // `ApplyDeviceTokenIdsOverride`'s own `ov.count <= T` already expresses.
  void RefreshFromHost(Queue& q, const void* src, size_t bytes);

  // Copy `bytes` from a DEVICE source into the bound destination, on `q`, so the
  // copy is ordered after whatever produced the source. See the two arms above.
  void RefreshFromDevice(Queue& q, const void* src, size_t bytes);

  int64_t host_refreshes() const { return host_refreshes_; }
  int64_t device_refreshes() const { return device_refreshes_; }
  StepInputSource last_source() const { return last_source_; }

 private:
  void Copy(Queue& q, const void* src, size_t bytes, StepInputSource from);

  Backend* b_ = nullptr;
  void* device_ = nullptr;   // NOT owned: the caller's pooled buffer
  void* staging_ = nullptr;  // owned pinned host block, or null
  size_t capacity_ = 0;
  int64_t host_refreshes_ = 0;
  int64_t device_refreshes_ = 0;
  StepInputSource last_source_ = StepInputSource::kUnset;
};

}  // namespace vt
