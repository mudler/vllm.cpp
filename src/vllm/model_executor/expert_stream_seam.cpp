// MODEL-TEXT-GLM-MOE-DSA W3 (issue #2214). See expert_stream_seam.h for what
// this is and why it stopped living in `qwen3_5.cpp`.
//
// Every body below is the one that translation unit ran, moved rather than
// rewritten, so the streamed lane's bytes and its stderr lines are unchanged.
// The three deliberate edits are named where they occur: the resident fallback
// is a parameter, the step guard's refusal loses its `qwen3_5:` prefix because
// the guard is no longer that model's, and `RequireSlotCapacity` is new.
#include "vllm/model_executor/expert_stream_seam.h"

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>  // ::sysconf(_SC_PAGESIZE) in the readahead hint below
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "vllm/config/weight_residency.h"
#include "vllm/platforms/interface.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace expert_stream {

bool StreamRequested() {
  // ENG-RESIDENCY-CONFIG (#1110): the answer now comes from
  // `ResolveExpertStreamRequested()`, which holds `VT_MOE_EXPERT_STREAM` >
  // `--offload-config`'s `vllm_cpp.expert_stream.enabled` > OFF, and which keeps
  // this knob's ODD environment rule verbatim: only the FIRST CHARACTER is
  // examined, so `VT_MOE_EXPERT_STREAM=false` is ON. That is what
  // docs/ENVIRONMENT.md documents, so it is transcribed rather than normalised
  // onto the tree's whole-value polarity.
  //
  // The answer is still cached on first call, and that is deliberate — it decides
  // whether an ~18 GiB slot store is built and whether the default-on grouped-MoE
  // path is disabled, and those two must not be able to disagree later in the same
  // process. The function-local static lives in `ResolveExpertStreamRequested`, not
  // here; this is a pure delegation.
  //
  // That cached answer is why `SetWeightResidencyConfig` refuses a config that would
  // CHANGE it — not one that arrives late. A document that omits `expert_stream`, or
  // asks for exactly what was decided, or that the environment overrides anyway, is
  // accepted; only one that would make this function's answer differ from the value
  // already returned is refused, because recording that would publish a
  // configuration the engine is not running. The loader installs in `FromModelDir`'s
  // first block, ahead of all weight I/O, so the ordering holds by construction.
  return ResolveExpertStreamRequested();
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertStreamLane

std::unique_ptr<ExpertStreamLane>& ExpertStreamLane::Slot() {
  // The single instance, and the lock that makes creating it safe. Both are
  // function-local statics so their own initialisation is the thread-safe magic
  // static this class failed to use for the instance itself.
  static std::unique_ptr<ExpertStreamLane> inst;
  return inst;
}

std::mutex& ExpertStreamLane::Mutex() {
  static std::mutex m;
  return m;
}

size_t& ExpertStreamLane::Reserved() {
  static size_t bytes = 0;
  return bytes;
}

bool& ExpertStreamLane::ForceFallback() {
  static bool on = false;
  return on;
}

// Constant-initialised and destructor-free, so the "has the final line been
// printed" answer survives every other static's destruction.
bool& ExpertStreamLane::FinalReported() {
  static bool done = false;
  return done;
}

ExpertStreamLane* ExpertStreamLane::Existing() { return Slot().get(); }

void ExpertStreamLane::Reserve(size_t slot_bytes) {
  if (!StreamRequested()) return;
  std::lock_guard<std::mutex> lk(Mutex());
  if (slot_bytes > Reserved()) Reserved() = slot_bytes;
}

ExpertStreamLane* ExpertStreamLane::Get(size_t slot_bytes) {
  if (!StreamRequested()) return nullptr;
  ExpertStreamLane* inst = nullptr;
  {
    // Construction only. The lock does NOT cover Slice: a decode step runs on
    // one host thread, and holding a process-wide mutex across every expert
    // slice would serialise the lane it exists to speed up.
    //
    // It does have to cover construction. The previous `static T* inst =
    // nullptr; if (inst == nullptr) inst = new T(...)` is not the magic-static
    // idiom used a few lines above and carries none of its guarantees: two
    // concurrent first calls both see null, both construct, and one ~18 GiB
    // store is leaked while the two halves of the model disagree about which
    // cache they are using.
    std::lock_guard<std::mutex> lk(Mutex());
    std::unique_ptr<ExpertStreamLane>& slot = Slot();
    if (slot == nullptr) {
      slot.reset(new ExpertStreamLane(std::max(slot_bytes, Reserved())));
    }
    inst = slot.get();
  }
  // A tower larger than the slot cannot be served by THIS store. Refuse by
  // name rather than silently falling back, which would make a streaming
  // benchmark quietly measure the mmap path instead. Both sizes are named,
  // because "too big" without the budget it exceeded is not actionable.
  VT_CHECK(slot_bytes <= inst->store_->slot_bytes(),
           "expert stream: a slice of " + std::to_string(slot_bytes) +
               " bytes exceeds the slot budget of " +
               std::to_string(inst->store_->slot_bytes()) +
               "; raise VT_MOE_EXPERT_STREAM_SLOT_BYTES");
  return inst;
}

void ExpertStreamLane::EndStepIfActive() {
  if (ExpertStreamLane* s = Existing()) s->EndStep();
}

void ExpertStreamLane::SetForceFallback(bool on) { ForceFallback() = on; }

// `fd`/`file_offset` describe where the slice lives on DISK. When they are
// valid the slot is filled by pread, which is the form the design specified
// and the only one that changes the I/O: a memcpy from the mapping still
// traps every 4 KiB page of its source, which is why W4 measured no decode
// gain. The mapping copy stays as the fallback for a weight with no
// descriptor (an expanded or repacked tensor owns its bytes outright).
uint8_t* ExpertStreamLane::Slice(const uint8_t* base, uint64_t tower_uid,
                                 int64_t expert, size_t offset, size_t bytes,
                                 int fd, size_t file_offset) {
  if (ForceFallback()) {
    // COUNTED SEPARATELY FROM `exhausted_`, on purpose (#1091 finding 6).
    // `exhausted` is the operator-facing number and both docs define it as
    // "the budget is smaller than one step's working set; raise
    // VT_MOE_EXPERT_STREAM_SLOTS". This switch has no production caller at
    // all, so charging it to that counter told an operator to raise a budget
    // that was never the reason. It stays out of the stderr line for the same
    // reason: in a production process it is always zero.
    ++forced_;
    return nullptr;
  }
  const int32_t tower = TowerId(tower_uid);
  const ExpertKey key{tower, static_cast<int32_t>(expert)};
  if (fd >= 0) {
    const ExpertStreamer::Result r =
        streamer_->EnsureFile(key, fd, file_offset + offset, bytes);
    if (r.slot < 0) {
      ++exhausted_;
      return nullptr;
    }
    return store_->Slot(r.slot);
  }
  // Ask the kernel for the WHOLE slice up front, before the copy touches it.
  //
  // This is the difference between one readahead and 608 demand faults. The
  // W4 measurement showed that filling a slot by memcpy from the mapping
  // inherits the fault path streaming exists to bypass: the copy is
  // sequential, but each 4 KiB page still traps. MADV_WILLNEED hands the
  // range to the kernel's readahead in one call, which is the same lever
  // `PrefaultBorrowedSpan` already uses at load, applied per slice at decode.
  //
  // Advisory and read-only, so it cannot change a byte. Skipped on a hit,
  // where nothing will be read at all.
  //
  // THE ADDRESS MUST BE PAGE-ALIGNED. madvise(2) returns EINVAL when it is
  // not, and a GGUF tensor is aligned to `general.alignment`, which defaults
  // to 32 (gguf_reader.cpp:401) — so the slice address is essentially never a
  // page boundary and the unaligned form was a no-op that reported nothing,
  // because the return value was discarded too. Round the start DOWN and the
  // end UP: the extra bytes belong to a neighbouring expert that this step is
  // very likely to want as well, and MADV_WILLNEED cannot harm them either
  // way. `advised_` counts the calls that were actually accepted, so a run
  // can tell a working hint from a silently rejected one.
  //
  // ROUNDING THE END UP CAN LEAVE THE ALLOCATION, and that is the one way
  // this call still fails: madvise(2) returns ENOMEM when any page in the
  // range is unmapped, so a tower whose last byte sits near the end of its
  // final mapped page would not be counted. In production the tower is a
  // borrowed view into a file mapping many pages larger than one slice, so
  // the trailing page is mapped. On the heap-backed towers the gates build it
  // holds because the allocator's arena page is mapped, not because the
  // allocation reaches it — which is why `advised == fills` is asserted
  // against a measured run rather than assumed from the arithmetic.
  //
  // NO SPEEDUP IS CLAIMED HERE. This makes the call well-formed; whether
  // readahead moves decode is a measurement the spec records as owed.
#if defined(__unix__)
  if (!cache_->IsResident(key)) {
    const long ps_l = ::sysconf(_SC_PAGESIZE);
    const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
    const auto begin = reinterpret_cast<uintptr_t>(base + offset);
    const uintptr_t page_begin = begin & ~(ps - 1);
    const uintptr_t page_end = (begin + bytes + ps - 1) & ~(ps - 1);
    if (::madvise(reinterpret_cast<void*>(page_begin),
                  static_cast<size_t>(page_end - page_begin),
                  MADV_WILLNEED) == 0) {
      ++advised_;
    }
  }
#endif
  const ExpertStreamer::Result r =
      streamer_->EnsureSpan(key, base + offset, bytes);
  if (r.slot < 0) {
    ++exhausted_;
    return nullptr;
  }
  return store_->Slot(r.slot);
}

void ExpertStreamLane::EndStep() {
  streamer_->EndStep();
  ReportStats(/*final=*/false);
}

// ONE line a benchmark can read to prove the lane stayed live.
//
// This exists because of how the row's published decode number went wrong.
// The run printed `[expert-stream] ON ...` once at startup and then nothing,
// so a cache that switched itself off partway through token 3 looked exactly
// like one that worked for the whole run, and "streaming ON: no decode gain"
// was measured against a dead lane. The two numbers that would have caught it
// immediately are `steps` and `exhausted`: steps==0 means the step clock never
// advanced, and exhausted>0 means slices were refused and silently served from
// the mapping instead. Both are on this line, and either is wrong at a glance.
//
// `final` IS THE WHOLE POINT AND IT USED TO HAVE NO CALLER (#1091 finding 1).
// The periodic report is skipped on `steps == 0` — so the one run that most
// needs the line, the one where the step boundary is never reached, printed
// nothing at all, and both docs told an operator to read a zero off a line
// that could not exist. It is skipped again whenever `stats_every_` does not
// divide the step count, and the default is 16, so a healthy five-token run
// printed nothing either and a benchmark reading absence as failure reported
// VOID on a working lane. The final report crosses both early returns.
void ExpertStreamLane::ReportStats(bool final) const {
  const int64_t steps = cache_->steps();
  if (!final) {
    if (stats_every_ <= 0) return;
    if (steps == 0 || steps % stats_every_ != 0) return;
  }
  PrintStatsLine(steps, cache_->hits(), cache_->misses(), cache_->evictions(),
                 streamer_->fills(), streamer_->bytes_filled(), exhausted_,
                 advised_);
}

// The final line, printed exactly ONCE per process.
//
// NOTHING IN PRODUCTION CALLS THIS, and read the destructor below before you
// conclude otherwise. Teardown produces the LINE but does not route through
// here: the store is a function-local static, so `~ExpertStreamLane` runs on
// the normal exit path and calls `ReportStats` itself, for the reason stated
// there. The two share the once-flag, not a call, so exactly one of them
// prints. This entry exists so a GATE can observe the same guarantee from
// inside a running process, because a static destructor fires after main
// returns and nothing in the process can assert on it.
//
// A once-flag rather than two independent prints, so "one line" is a property
// of the process and not of which caller happened to win. The flag is a plain
// bool with constant initialisation and no destructor of its own, so it cannot
// itself be lost to static-destruction ordering.
//
// No store means no line, and that is not a gap: a store that exists always
// announced itself with `[expert-stream] ON ...` first, so banner-without-line
// is a process that died, and no-banner is a lane nothing ever reached.
void ExpertStreamLane::FlushFinalStats() {
  if (FinalReported()) return;
  ExpertStreamLane* s = Existing();
  if (s == nullptr) return;
  FinalReported() = true;
  s->ReportStats(/*final=*/true);
}

ExpertStreamLane::~ExpertStreamLane() {
  // The store holds the numbers, so it prints them before it goes away. Not
  // routed through FlushFinalStats: that reads `Existing()`, and the unique_ptr
  // this object lives in does not clear itself before running this destructor.
  if (!FinalReported()) {
    FinalReported() = true;
    ReportStats(/*final=*/true);
  }
}

// The one place the statistics line's format lives, so the final report and
// the periodic report cannot drift apart into two shapes a parser has to
// know about.
void ExpertStreamLane::PrintStatsLine(int64_t steps, int64_t hits,
                                      int64_t misses, int64_t evictions,
                                      int64_t fills, int64_t bytes,
                                      int64_t exhausted, int64_t advised) {
  std::fprintf(stderr,
               "[expert-stream] steps=%lld hits=%lld misses=%lld "
               "evictions=%lld fills=%lld bytes=%lld exhausted=%lld "
               "advised=%lld\n",
               static_cast<long long>(steps), static_cast<long long>(hits),
               static_cast<long long>(misses),
               static_cast<long long>(evictions),
               static_cast<long long>(fills), static_cast<long long>(bytes),
               static_cast<long long>(exhausted),
               static_cast<long long>(advised));
}

ExpertStreamLane::ExpertStreamLane(size_t slot_bytes) {
  // ENG-RESIDENCY-CONFIG (#1110): both sizes resolve through the shared
  // resolvers, which hold env var > `--offload-config`'s
  // `vllm_cpp.expert_stream.{slots,slot_bytes}` > the default, and which keep
  // the tolerant integer parsing this constructor already had (atoll, then
  // ignore anything non-positive) so an environment-only run is unchanged. The
  // CONFIG side is stricter: a zero or negative value is refused at startup,
  // where the operator can still read the message, rather than silently becoming
  // the default.
  //
  // `slot_bytes`' default is the caller's computed maximum, not a constant, so it
  // is passed in rather than duplicated here.
  slot_bytes = static_cast<size_t>(
      ResolveExpertStreamSlotBytes(static_cast<int64_t>(slot_bytes)));
  const int32_t slots = static_cast<int32_t>(ResolveExpertStreamSlots());
  // STATS_EVERY stays environment-only, and that is a decision rather than an
  // oversight: it changes only how often the line below is printed, so it is the
  // instrument and not the configuration. `--offload-config` refuses it as an
  // unknown key rather than accepting and dropping it.
  const char* se = std::getenv("VT_MOE_EXPERT_STREAM_STATS_EVERY");
  if (se != nullptr && *se != '\0') {
    const long v = std::atol(se);
    if (v >= 0) stats_every_ = static_cast<int64_t>(v);
  }
  // Record the geometry this store was actually built with. It is the only way
  // a test can tell that the two resolvers above were consulted: the values are
  // otherwise visible only on a stderr line, and a site that hardcoded the
  // default would leave every existing suite green.
  NoteExpertStreamGeometry(static_cast<int64_t>(slots),
                           static_cast<int64_t>(slot_bytes));
  store_ = std::make_unique<HostExpertSlotStore>(slots, slot_bytes);
  cache_ = std::make_unique<ExpertSlotCache>(slots);
  streamer_ = std::make_unique<ExpertStreamer>(*cache_, *store_);
  std::fprintf(stderr,
               "[expert-stream] ON slots=%d slot_bytes=%zu resident=%.2f GiB\n",
               slots, slot_bytes, store_->resident_bytes() / 1073741824.0);
}

// A tower's cache identity, compacted into the int32 the key carries.
//
// The argument is the tensor's PROCESS-UNIQUE uid, not its base pointer. A
// pointer was wrong here in a way no single-model test could see. This store
// is a process-lifetime singleton, so it outlives any one model, and the
// allocator hands out an address again as soon as the first model is freed.
// The second model's tower then hit the FIRST model's entries and was served
// another checkpoint's weights, as a HIT, which by contract moves no bytes and
// so leaves nothing downstream to notice. Measured on two synthetic models in
// one process: 24 towers occupied 21 distinct addresses, and 20 of 222 slices
// came back wrong. The comment this replaces asserted the opposite, and its
// premise ("stable for the model's life") was true; the CACHE is simply not
// scoped to one model's life.
int32_t ExpertStreamLane::TowerId(uint64_t uid) {
  auto it = tower_ids_.find(uid);
  if (it != tower_ids_.end()) return it->second;
  const int32_t id = next_tower_id_++;
  tower_ids_.emplace(uid, id);
  return id;
}

// ─────────────────────────────────────────────────────────────────────────────
// The step guard

bool& ExpertStreamStepGuard::Open() {
  static thread_local bool open = false;
  return open;
}

void ExpertStreamStepGuard::Begin() {
  // THE ONE TEXT CHANGE IN THIS LIFT. The message read `qwen3_5: a decode step
  // is already open; the expert-stream step guard marks ONE forward and must
  // not nest`. The guard is no longer that model's, so the prefix is now the
  // seam's. `test_expert_stream_steps.cpp` matches on `must not nest` rather
  // than on the whole sentence — deliberately, because VT_CHECK appends
  // " at <file>:<line>" — so the substring it reads is unchanged.
  VT_CHECK(!Open(), "expert stream: a decode step is already open; the "
                    "expert-stream step guard marks ONE forward and must not "
                    "nest");
  Open() = true;
}

void ExpertStreamStepGuard::End() {
  Open() = false;
  ExpertStreamLane::EndStepIfActive();
}

// ─────────────────────────────────────────────────────────────────────────────
// The slice seam

vt::Tensor HostSliceView(Dev d, const OwnedTensor& w, uint8_t* data, int64_t N,
                         int64_t K) {
  vt::Tensor wt =
      dense_attn::MakeTensor(static_cast<void*>(data), w.dtype, d.q.device,
                             {N, K});
  wt.repacked = w.repacked;
  wt.elem_kn_repacked = w.elem_kn_repacked;
  return wt;
}

vt::Tensor ExpertSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                       int64_t row_off, int64_t expert,
                       ResidentSliceFn resident_fallback) {
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  const size_t bytes = static_cast<size_t>(N) * row_bytes;
  const vllm::platforms::Platform& p =
      vllm::platforms::GetPlatform(d.q.device.type);
  const bool cpu = p.is_cpu();
  // ENG-EXPERT-STREAM-DEVICE W0b/W0c (issue #1124). The lane serves slices out
  // of HOST storage — `HostExpertSlotStore`'s arena is a `std::vector<uint8_t>`
  // — so the question is not "is this the CPU" but "can this platform's kernels
  // READ host storage". `is_cpu()` is one answer to that;
  // `host_memory_is_device_addressable()` is the probed answer for the rest, and
  // it is what lets a GB10 serve a 369.96 GiB checkpoint out of a 119.631 GiB
  // pool instead of refusing the load.
  //
  // A DISCRETE device answers false, keeps falling through to the caller's
  // resident slice, and therefore keeps hitting the #1123 load-time refusal.
  // That is correct for it: a slot store it cannot read is not a lane, and
  // giving it one is `ENG-EXPERT-STREAM-DEVICE` W2's job, not this branch's.
  if (cpu || p.host_memory_is_device_addressable()) {
    if (ExpertStreamLane* st = ExpertStreamLane::Get(bytes)) {
      // Claim the tower for the lane BEFORE taking a slice, so the refusal in
      // `ResidentWeight` covers the whole tower rather than only the slices
      // that happened to be served. A tower the lane touched at all must never
      // be staged.
      w.expert_streamed = true;
      const uint8_t* base = w.bytes.data();
      if (uint8_t* slot = st->Slice(base, w.TowerUid(), expert,
                                    static_cast<size_t>(row_off) * row_bytes,
                                    bytes, w.mmap_fd, w.mmap_file_offset)) {
        return HostSliceView(d, w, slot, N, K);
      }
      // The cache could not serve this slice. On CPU that falls through to the
      // caller's unchanged resident slice below, which aliases the tower and is
      // what every existing arm-comparison gate measures.
      //
      // On a host-addressable DEVICE it must NOT: the resident slice would
      // stage the whole 1.1875 GiB tower, and this branch is not rare. Prefill
      // takes it thousands of times by construction — the peak protected set
      // for a T-token prompt is `93 x 3 x min(512, 10*T)` slices, which
      // saturates at 331 GiB for any T >= 52, so no slot budget makes prefill
      // fit. Reading the tower's own host bytes in place is exactly what the
      // CPU arm does, costs nothing, and is correct precisely because this
      // platform said its kernels can follow a host pointer.
      if (!cpu) {
        return HostSliceView(
            d, w,
            const_cast<uint8_t*>(base) + static_cast<size_t>(row_off) * row_bytes,
            N, K);
      }
    }
  }
  return resident_fallback(d, w, N, K, row_off);
}

// ─────────────────────────────────────────────────────────────────────────────
// The capacity refusal (spec §3.3)

int64_t WorkingSetSlots(int64_t streamed_tower_count, int64_t experts_per_tok) {
  if (streamed_tower_count <= 0 || experts_per_tok <= 0) return 0;
  return streamed_tower_count * experts_per_tok;
}

void RequireSlotCapacity(const std::string& model_name,
                         int64_t streamed_tower_count, int64_t experts_per_tok,
                         int64_t slots) {
  const int64_t need = WorkingSetSlots(streamed_tower_count, experts_per_tok);
  // Inert when the geometry could not be determined. A caller that reads a zero
  // out of a file it did not understand must not have that zero turn into a
  // refusal of every budget, and it must not turn into a silent pass either —
  // which is why the caller, not this function, decides whether a missing
  // geometry is itself an error.
  if (need <= 0 || slots <= 0) return;
  // The message names the number, the knob and the model, because "too small"
  // without the budget it needed is not actionable — the same standard the
  // slot-bytes refusal in `Get` already meets.
  VT_CHECK(slots >= need,
           "expert stream: " + model_name + " needs at least " +
               std::to_string(need) + " slots (" +
               std::to_string(streamed_tower_count) + " streamed expert towers x " +
               std::to_string(experts_per_tok) +
               " experts per token, all protected until the step ends) but the "
               "configured budget is " +
               std::to_string(slots) +
               "; raise VT_MOE_EXPERT_STREAM_SLOTS or "
               "vllm_cpp.expert_stream.slots. A budget below one step's working "
               "set does not fail: every refused slice is read in place out of "
               "the mmap'd tower, which silently measures the page-cache path "
               "instead of the streaming one");
}

}  // namespace expert_stream
}  // namespace vllm
