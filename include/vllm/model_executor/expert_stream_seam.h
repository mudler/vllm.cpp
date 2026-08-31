// MODEL-TEXT-GLM-MOE-DSA W3 (issue #2214, spec
// .agents/specs/glm-dsa-latest-deepseek.md §3.7).
//
// The expert-streaming WIRING, lifted out of `qwen3_5.cpp` so a second model's
// translation unit can include it.
//
// WHAT WAS WRONG BEFORE THIS FILE EXISTED. `expert_streamer.{h,cpp}`,
// `expert_slot_cache.{h,cpp}` and `host_expert_slot_store.h` were already
// shared: the policy, the fill and the destination are all reusable. What was
// NOT shared is everything that turns those three into a lane a forward can
// take — the process-lifetime store, the step boundary, the slot-sized
// reservation, the platform predicate and the slice seam itself. All of that
// lived inside `qwen3_5.cpp`'s unnamed namespace, and it was the ONLY model TU
// that constructed a `HostExpertSlotStore`. A new architecture could include
// every one of those headers and still not stream, because the mechanism that
// uses them was not reachable from anywhere else (spec §3.2 gap 1, `## Owed`
// O8).
//
// WHY THE RESIDENT FALLBACK IS INJECTED RATHER THAN CALLED. `ExpertSlice`
// falls back to the model's own resident-tower view whenever streaming is off
// or the cache cannot serve the slice, and that fallback is NOT common code.
// `qwen3_5.cpp` defines its own `ResidentWeight` in its unnamed namespace which
// SHADOWS `dense_attn_block.h`'s, and the two differ materially: the local one
// carries the `expert_streamed`, `elem_kn_repacked` and `repacked` staging
// refusals and the `MakeHostBytesDeviceAliasable` / `StageOwnedWeightsToDevice`
// host-alias arm, none of which the header's version has. Calling
// `ResidentWeight` from THIS translation unit would therefore bind to the
// header's definition and silently change which guarantees Qwen3.5's streamed
// lane carries — a behaviour change to a gated model, which §3.7 makes a stop
// condition rather than a judgement call. So the caller passes its own
// resident-slice function and byte-identity holds by construction.
//
// WHAT THIS DELIBERATELY DOES NOT CHANGE. The policy is still the hotness-decayed
// LFU in `expert_slot_cache.cpp`; there is still no prefetch, no double
// buffering and no asynchronous I/O; the destination is still the concrete host
// store. Those are `ENG-EXPERT-STREAM` W6 and `ENG-EXPERT-STREAM-DEVICE` W2 and
// this file takes none of them. The lift is a move, plus the one refusal §3.3
// argues for.
#ifndef VLLM_MODEL_EXECUTOR_EXPERT_STREAM_SEAM_H_
#define VLLM_MODEL_EXECUTOR_EXPERT_STREAM_SEAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "vllm/model_executor/expert_slot_cache.h"
#include "vllm/model_executor/expert_streamer.h"
#include "vllm/model_executor/host_expert_slot_store.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

namespace vllm {
namespace expert_stream {

using ::vllm::dense_attn::Dev;

// Whether the operator ASKED for streaming, independent of whether a store has
// been built yet. The grouped-MoE gate needs this before any expert is touched.
bool StreamRequested();

// ENG-EXPERT-STREAM W4: serve an expert slice from a BOUNDED slot cache instead
// of reading it straight out of the mmap'd tower.
//
// Default OFF (`VT_MOE_EXPERT_STREAM=1` turns it on, `VT_MOE_EXPERT_STREAM_SLOTS`
// sets the budget), so every existing path keeps its bytes and its numbers.
//
// Why this exists, measured rather than assumed. On `Qwen3.8-2.4T-A95B UD-Q1_0`
// the borrowed tower already makes a 370 GiB model FIT in 119 GiB, because the
// weights are never copied. What it does not do is make it fast: each token
// needs ~6.7 GB of expert bytes and the kernel serves them as 4 KiB faults in
// ROUTER order, measured near 100 MB/s against an NVMe that sustains ~5 GB/s.
// A slot is filled by ONE contiguous copy of a whole slice, so the read is
// sequential, and a slice already resident costs nothing at all.
//
// The cache is keyed by (tower, expert). A tower's identity is `TowerUid()`, a
// process-unique counter, NOT the buffer's address: the store outlives any one
// model, and the allocator hands a freed tower's address to the next one (#1066,
// and the note on TowerId in the .cpp).
class ExpertStreamLane {
 public:
  // The store, or nullptr when nothing has built one. NEVER constructs, which
  // is the whole reason it is separate from Get: the once-per-step hook and the
  // stats reader must be able to ask "is this lane live?" without allocating an
  // 18 GiB slot array behind a model that is not streaming at all.
  static ExpertStreamLane* Existing();

  // The largest slice a caller is about to take, declared BEFORE it takes any.
  //
  // gate/up and down are not the same size whenever a dynamic (UD) quant keeps
  // `down_proj` at a higher precision than the gate/up pair, which is exactly
  // what the checkpoints this row targets do. Sizing the store from whichever
  // slice happened to arrive first then makes the FIRST down slice exceed the
  // slot and trip the check in `Get` in the middle of decode. Declaring the
  // maximum up front sizes the store once, correctly, before anything is
  // stored, so the only remaining way to trip that check is a genuinely
  // unforeseen slice.
  static void Reserve(size_t slot_bytes);

  // The store, constructing it on first use. Refuses BY NAME when `slot_bytes`
  // exceeds the slot budget the store was built with, rather than falling back
  // to the mmap path and letting a streaming benchmark quietly measure it.
  static ExpertStreamLane* Get(size_t slot_bytes);

  // The decode step boundary. Calling it is what clears per-step eviction
  // protection and advances the hotness clock; see the note on the RAII guard
  // below for what happens when nobody does.
  static void EndStepIfActive();

  // Force the cache-exhaustion branch, which makes every Slice return nullptr
  // and every caller fall back to the resident tower view. This is a REAL
  // production state (a budget smaller than one step's working set reaches it),
  // and having a switch for it is what lets a gate prove the streamed and
  // unstreamed arms produce the same bytes inside one process.
  static void SetForceFallback(bool on);

  // The final statistics line, printed exactly ONCE per process. See the .cpp
  // for why the destructor and this entry share a once-flag rather than a call.
  static void FlushFinalStats();

  // Returns the slot's bytes for `expert` of the tower based at `base`, or
  // nullptr when the cache is exhausted for this step (the caller then reads
  // the tower directly, which is correct but slow, and is counted).
  uint8_t* Slice(const uint8_t* base, uint64_t tower_uid, int64_t expert,
                 size_t offset, size_t bytes, int fd, size_t file_offset);

  void EndStep();
  void ReportStats(bool final) const;

  const ExpertStreamer& streamer() const { return *streamer_; }
  const ExpertSlotCache& cache() const { return *cache_; }
  int64_t exhausted() const { return exhausted_; }
  int64_t forced() const { return forced_; }
  int64_t advised() const { return advised_; }

  ~ExpertStreamLane();

 private:
  explicit ExpertStreamLane(size_t slot_bytes);

  static std::unique_ptr<ExpertStreamLane>& Slot();
  static std::mutex& Mutex();
  static size_t& Reserved();
  static bool& ForceFallback();
  static bool& FinalReported();
  static void PrintStatsLine(int64_t steps, int64_t hits, int64_t misses,
                             int64_t evictions, int64_t fills, int64_t bytes,
                             int64_t exhausted, int64_t advised);

  int32_t TowerId(uint64_t uid);

  std::unique_ptr<HostExpertSlotStore> store_;
  std::unique_ptr<ExpertSlotCache> cache_;
  std::unique_ptr<ExpertStreamer> streamer_;
  std::unordered_map<uint64_t, int32_t> tower_ids_;
  int32_t next_tower_id_ = 0;
  int64_t exhausted_ = 0;
  // Slices the FORCED-fallback switch refused. Separate from `exhausted_`
  // because that one is an operator-facing budget diagnosis and this one is a
  // gate asking for the unstreamed arm; see the note at the ForceFallback
  // branch in Slice.
  int64_t forced_ = 0;
  int64_t advised_ = 0;
  int64_t stats_every_ = 16;
};

// The decode step boundary, as a scope guard.
//
// WHY THIS EXISTS AT ALL. `ExpertSlotCache::Acquire` marks every entry it
// serves `protected_this_step`, because evicting a slot the current step is
// about to read would hand the kernel bytes that are being overwritten. ONLY
// EndStep clears that mark. Without a caller, the protection is permanent: once
// the cache fills, `ColdestEvictable` finds every entry protected and returns
// -1, `Acquire` returns slot -1, `Slice` returns nullptr, and every later slice
// falls back to the mmap path. The lane switches itself off and says nothing,
// the step clock never advances, and so the hotness decay, the LFU score, the
// LRU tiebreak and eviction never run in production at all. On the live
// configuration (8000 slots, ~2790 slices per token) that happens partway
// through the third token.
//
// A guard rather than a call at the end of the body: a forward has several
// returns and can throw, and a step that ended by throwing still ended.
//
// ONE FORWARD IS ONE STEP, AND THE GUARD REFUSES TO NEST (#1091 finding 3).
// Five forwards in `qwen3_5.cpp` take expert slices — `ForwardLayers`,
// `Qwen3_5Model::ForwardDense`, both MTP forwards and `Qwen3_5ReplayLayer` —
// and each is a complete forward that no other one contains. A nested guard
// would end the step twice, which advances the hotness clock for a step that
// never happened and decays every resident entry an extra tick; that is a
// quieter defect than the missing boundary and it is the one adding guards
// invites. So the precondition is stated rather than handled, the same way
// `MatmulF32Slice` states `expert >= 0`: the flag is per-thread because a
// decode step runs on one host thread, which is the assumption the store's own
// locking already makes.
//
// THE REFUSAL IS NOT GATED ON `StreamRequested()`, deliberately.
// "One forward is one step" is a property of the CALL GRAPH, not of the
// streaming lane: a nest is a defect whether or not a store exists, and the
// streamed run is the rare configuration. Arming it only there would let the
// default-on path establish a nest that nobody sees until someone turns
// streaming on, which is the shape this row keeps finding. The cost is that a
// nest reds every forward rather than only the streamed ones, and that
// is the intended polarity: loud on the default path is what makes it a gate.
//
// `Begin`/`End` are named rather than living only in the constructor and
// destructor bodies so that `detail::ExpertStreamStepScope` can hold THE SAME
// boundary. A gate that re-implemented the refusal would prove its own copy;
// this way deleting the `VT_CHECK` in `Begin` is one edit that both changes
// production and takes the gate red.
struct ExpertStreamStepGuard {
  ExpertStreamStepGuard() { Begin(); }
  ~ExpertStreamStepGuard() { End(); }
  ExpertStreamStepGuard(const ExpertStreamStepGuard&) = delete;
  ExpertStreamStepGuard& operator=(const ExpertStreamStepGuard&) = delete;

  // Open the step. Throws when one is already open on this thread; the flag is
  // then left as it was, so the outer guard's `End` still closes exactly one.
  static void Begin();
  static void End();

 private:
  static bool& Open();
};

// A [N,K] weight tensor over HOST-resident slice bytes — a slot, or the tower's
// own mapping — carrying exactly the markers `ResidentWeight`'s ALIASING branch
// carries and nothing else.
//
// WHY THIS IS NOT `ResidentWeight` WITH THE POINTER OVERWRITTEN, which is what
// the slot arm used to do (ENG-EXPERT-STREAM-DEVICE W0c, issue #1124). That call
// existed solely to inherit dtype/device/repack markers, and on CPU it aliases
// and costs nothing. On a WEIGHT-STAGING platform it takes the other branch and
// runs `d.b.Alloc(w.bytes.size())` on the whole `[E*N,K]` tower — 1.1875 GiB —
// and memoizes it on `w.d_dev`. So the very first streamed slice allocated the
// thing streaming exists to avoid, and lifting the `is_cpu()` guard in
// `ExpertSlice` without this helper reproduces issue #1123 instead of fixing it.
//
// The marker set is inherited by TRANSCRIPTION rather than by call, so it is
// worth saying what it is and what it deliberately omits. `repacked` and
// `elem_kn_repacked` are copied, exactly as the aliasing branch copies them.
// `q8_0_aligned` is NOT, also exactly as that branch does not: `MakeTensor`
// drops it and `ResidentWeight` never restores it on either branch, so copying
// it here would change the CPU lane's bytes rather than preserve them.
vt::Tensor HostSliceView(Dev d, const OwnedTensor& w, uint8_t* data, int64_t N,
                         int64_t K);

// The model's own resident-tower slice, called when streaming is off or the
// cache cannot serve the slice. See the file header for why this is a parameter
// rather than a call: the two `ResidentWeight` definitions in this tree are not
// interchangeable, and binding to the wrong one changes a gated model's
// behaviour without changing a single byte of this file.
using ResidentSliceFn = vt::Tensor (*)(Dev d, const OwnedTensor& w, int64_t N,
                                       int64_t K, int64_t row_off);

// The expert-slice seam. Identical to `resident_fallback` except that, when
// streaming is on, the returned tensor points at a SLOT holding a contiguous
// copy of the slice rather than into the tower itself.
vt::Tensor ExpertSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                       int64_t row_off, int64_t expert,
                       ResidentSliceFn resident_fallback);

// ─────────────────────────────────────────────────────────────────────────────
// The capacity refusal (spec §3.3).
//
// WHAT GOES WRONG WITHOUT IT. Every `Acquire` marks its entry protected for the
// step and only `EndStep` clears it (`expert_slot_cache.cpp:91`, `:142-145`), so
// one decode step's working set is every slice it touches, all resident at once:
// one per streamed tower per routed expert, i.e. `towers * experts_per_tok`
// where `towers` is `moe_layers * 3`. Below that budget `Acquire` returns -1,
// `Slice` returns nullptr, and the caller reads the tower IN PLACE out of the
// mmap. That is counted on stderr and is not an error.
//
// On the model this row targets it is fatal to any measurement: the default is
// 64 slots against a 1800-slice working set, and the silent fallback is a
// 187 GiB random read per token through the page cache. A benchmark that
// reports a streaming number measured on the mmap path is the exact shape of
// result this repository has been burned by, and the check costs one
// comparison at load.
//
// WHY THIS IS A REFUSAL AND NOT A CLAMP. Raising the budget silently would
// allocate an arena the operator did not ask for and did not size their device
// for; degrading silently is what this replaces. The operator is told the
// number they need and which knob sets it.

// One decode step's slice working set: `streamed_tower_count * experts_per_tok`.
// `streamed_tower_count` is the number of stacked `*_exps.weight` towers the
// lane will serve, which for a gate/up/down MoE is `moe_layers * 3`.
int64_t WorkingSetSlots(int64_t streamed_tower_count, int64_t experts_per_tok);

// Refuse, by name, a configured slot budget below that working set. `slots` is
// the resolved budget; `model_name` appears in the message so an operator with
// several models loaded knows which one refused. Inert when either input is
// non-positive, because a caller that could not determine the geometry must not
// synthesise a refusal out of a zero.
void RequireSlotCapacity(const std::string& model_name,
                         int64_t streamed_tower_count, int64_t experts_per_tok,
                         int64_t slots);

}  // namespace expert_stream
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_EXPERT_STREAM_SEAM_H_
