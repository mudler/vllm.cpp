// WHERE A RENDER SPENDS ITS WALL, and how many bytes it held while it did.
//
// Row LTX25-DEVICE-RESIDENCY stage W0, issue #1010. Spec:
// `.agents/specs/ltx25-device-residency.md`.
//
// WHY THIS EXISTS. The LTX-2.5 render path emitted exactly one line per render
// — the example's `wrote N frames` — and every attempt to act on that render's
// profile has since failed on the same three defects, all of them measurement
// defects and all of them recorded in the row's spec: the evidence is
// unretrievable (#1040: the sampler CSVs exist only on a host that stopped
// answering), the subject is unnamed (#1087 measures a 1731 s phase and says in
// its own text "Do not guess it from the duration"; #1024's GPU-zero window is
// now known to be neither the denoise nor the decode), and the ranking the
// campaign inherited is stale (#1009 and #1208 both landed after it). None of
// those is fixable by reading the code harder. They are fixable by a render
// that says what it did.
//
// WHAT IT IS. A flat, non-overlapping timeline of NAMED leaves, each carrying a
// monotone timestamp measured from the instrument's origin, a peak host byte
// count and a peak device byte count. It is written to a FILE beside the frames
// it explains, because a number that exists only on a console somebody had to
// be watching is the evidence class #1040 is made of.
//
// WHAT IT IS NOT. It is not a sampling profiler and it is not a benchmark
// harness. The leaves are placed by hand at the boundaries the driver already
// has, so the table is exactly as honest as where the scopes were put — which
// is why `unaccounted_seconds` is emitted as its own quantity rather than
// smeared over the leaves. Time inside no leaf is a phase nobody named, and the
// spec's stop condition for W0 is that naming it is the work, not rounding it
// away. A partial phase table is worse than none, because it invites exactly
// the interpolation the sampler CSVs already invited.
//
// ON BY DEFAULT, and that is the point. An instrument behind a flag nobody sets
// measures nothing on the runs that matter — the long ones, on a leased box, by
// somebody who did not know they would need the number until afterwards. The
// cost is a `/proc/self/statm` read at each boundary (~30 per render) and one
// 100 ms sampler thread, against renders measured in minutes.
//
// NESTING IS RECORDED, NEVER SUMMED. A leaf opened while another leaf is open is
// marked `nested` and excluded from the sum, in the shape `music3_profile.h`'s
// span/leaf split already ships: a table whose parts sum past its whole is a
// table nobody can read, and silently double counting would make the residue
// negative instead of visible.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vllm {
namespace multimodal {
namespace phase {

// One closed leaf (or span) of the timeline.
struct Record {
  std::string name;
  int64_t render = 0;   // 0 = the load; 1..N = the Nth generation in this process
  double start = 0.0;   // seconds since the instrument's origin
  double end = 0.0;
  int64_t peak_host_bytes = -1;    // max over the samples taken inside this scope
  int64_t peak_device_bytes = -1;  // -1 => no device probe was installed on this arm
  bool span = false;    // printed for context, never summed
  bool nested = false;  // opened while another leaf was open; excluded from the sum
};

// Resident set size in bytes, or -1 where the platform publishes none.
int64_t HostResidentBytes();

// The device-byte probe a family installs for its device arm. Returns bytes in
// use on the device, or -1 when the probe cannot answer. Called from the sampler
// thread as well as from the render thread, so an implementation must be safe to
// call concurrently with itself.
using DeviceByteProbe = std::function<int64_t()>;

// The process-wide timeline. Serialized internally: the sampler thread and the
// render thread both write it.
class PhaseLog {
 public:
  static PhaseLog& Instance();

  // Start a timeline, DISCARDING any earlier one.
  //
  // A LOAD IS WHAT STARTS A SESSION, and that is the whole reason this resets.
  // The load's own phases belong in the render's table — the spike measures ~7.5
  // minutes of DiT staging paid at the front of every render — so the origin has
  // to sit at the load and not at the generation. A process that loads a second
  // engine is measuring the second one, and carrying the first one's timeline
  // forward would report the time between two unrelated loads as this render's
  // unaccounted residue. The measurement shape the campaign takes is one load
  // and one render per process; a process that interleaves several gets a table
  // about the last load, which is stated here rather than left to be discovered
  // from a wall that does not match a stopwatch.
  void Begin();

  // What the device column means on this arm. Passing an empty probe restores
  // the "no device probe" sentinel, which is what the CPU arm reports.
  void SetDeviceProbe(DeviceByteProbe probe);

  // Which generation the leaves that follow belong to. The load is render 0.
  void SetRender(int64_t render);

  double Elapsed() const;

  // Open a leaf (or a span) and return its handle. `Close` on that handle
  // finishes it; closing twice is a no-op.
  size_t Open(const std::string& name, bool span);
  void Close(size_t handle);

  // Take one host/device sample and attribute it to every open scope. Called at
  // every Open and Close, by the sampler thread, and by hand inside a loop whose
  // interior peak the boundaries would miss.
  void Sample();

  // The live lane's counter line. See the free `Tick` at the bottom of this
  // header for what it prints and why it is called before the work rather than
  // after it.
  void Tick(const std::string& unit, int64_t index, const std::string& detail);

  std::vector<Record> Records() const;
  int64_t Samples() const;

  // Write the table as JSON. Returns false with *why set on an IO failure — a
  // render must not fail because its instrument could not write.
  bool WriteJson(const std::string& path, const std::string& family,
                 const std::string& device, std::string* why) const;

  // The same table as a fixed-width block, for `VLLM_RENDER_PHASE_LOG_STDERR`.
  std::string RenderText(const std::string& family, const std::string& device) const;

  // Drop every record and stop the sampler. For tests, and for a process that
  // wants a second independent timeline.
  void Reset();

 private:
  PhaseLog();
  ~PhaseLog();
  PhaseLog(const PhaseLog&) = delete;
  PhaseLog& operator=(const PhaseLog&) = delete;

  struct Impl;
  Impl* impl_;
};

// RAII around one leaf. `Close()` finishes it early, which is what the long
// linear render driver needs: its regions are sequential statements in one
// function, not nested blocks.
class Scope {
 public:
  explicit Scope(const std::string& name, bool span = false);
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  void Close();

 private:
  size_t handle_;
  bool closed_ = false;
};

// One sample, from inside a loop. Cheap enough for a per-step call.
void SampleNow();

// ─── THE LIVE LANE (issue #1413) ─────────────────────────────────────────────
//
// The table above is written by the SUCCESS PATH ONLY — `WritePhaseLog` sits
// after `im.trace.completed = true` — so a render that is killed, aborted by a
// lease governor, or still running writes nothing at all. That is not a corner
// case here: #1375 is `ABORT[92] PROJECTED OVERRUN` / `child exit=-15` / 0
// frames, and `ltx25-decode-speed.md`'s two rungs are `EXIT=137` / 0 frames and
// `EXIT=1` / 0 frames. An instrument that reports on completion reported on none
// of the runs this campaign has.
//
// So these functions emit as the render RUNS, to stderr, on the shipped default.
// `PhaseLog::Open` and `PhaseLog::Close` print a boundary line each, and `Tick`
// prints one line per repeated unit of work inside a phase.
//
// THE OPEN LINE IS THE LOAD-BEARING ONE. It means the LAST LINE PRINTED names
// the phase that is currently running, which is the entire difference between a
// working render and a hung one — two states that were byte-identical from
// outside for 2.5 hours at a time. A close-only emitter would still have printed
// nothing for the 3002 s conditioning stretch, because that stretch never
// closed.
//
// NOT BEHIND A FLAG. `VT_H3_PROGRESS` is the existing shape for this in the tree
// (`minimax_h3.cpp:776-793`) and it is opt-in, which is exactly why no LTX-2.5
// run has one: the runs whose profile is needed are the long ones, on a leased
// box, by somebody who did not know they would need the number until afterwards.
// `VLLM_RENDER_PROGRESS=0` silences it, in the measurement-lane shape
// `VLLM_RENDER_PHASE_SAMPLER` and `VLLM_LTX2_POOL_DRAIN` already take here — it
// exists so an A/B over what the emitter itself costs runs on ONE binary, and
// nothing in this tree sets it.

// Whether the live lane is on for this process. Read once, from
// `VLLM_RENDER_PROGRESS`, and cached.
//
// PUBLIC BECAUSE A CALLER HAS TO BUILD `detail` BEFORE IT CAN PASS IT. `Tick`
// returns immediately when the lane is off, but the argument is already built by
// then — a `std::string` and three `std::to_string`s per DiT forward at the
// LTX-2.5 call site. That is negligible against a 162 s forward and it is not
// nothing, so the claim "`VLLM_RENDER_PROGRESS=0` costs one `getenv` per
// process" is only true if the call site skips the formatting too. This is how
// it does that.
bool ProgressEnabled();

// One progress line for the `index`-th occurrence of a repeating unit of work.
// The emitter appends the elapsed clock and `last=`, the seconds since the
// previous tick of the SAME unit, which is the per-forward cost #1375 could
// only obtain as a wall-clock interval between GPU busy/idle edges. `detail` is
// free text printed after the counter.
//
// CALL IT BEFORE THE WORK, NOT AFTER. After is the obvious placement and it is
// wrong: the line a reader most needs is the one naming the unit that was in
// flight when the run stopped. The cost is that the last unit of a completed
// run has no line of its own, and its duration is inside the enclosing phase's
// close line.
//
// AND SAY WHAT THE GATE CAN SEE OF THAT, because the choice is presented above
// as though it were checked. On a render that COMPLETES the two placements are
// indistinguishable end to end: same ticks, same intervals, same ordering
// relative to the phase boundaries. Moving the production call below the forward
// leaves `test_ltx2_video` green. The two differ only on a run that dies inside
// a unit, which no gate here can stage through the ABI. What IS gated is this
// function's own contract — that the line reaches fd 2, FLUSHED, before the
// caller's next statement runs, so that a `SIGKILL` between the two leaves the
// announcement behind (`a live tick is FLUSHED BEFORE the work it announces`).
// The production placement rests on that contract plus a reading of the call
// site, and this comment is the whole of the evidence for it.
//
// `last=` IS KEYED BY `unit`, and the key's lifetime is the RENDER. The map is
// cleared by `Begin`, by `Reset` and by `SetRender`, so the first tick of each
// generation has no `last=` rather than one spanning the gap since the previous
// generation ended. Two units ticking in the same phase keep independent clocks.
//
// THE LINE IS WRITTEN UNDER THE PROCESS-WIDE PHASE MUTEX — the same one the
// 100 ms sampler thread takes. So the cost of a tick is a held global lock plus
// a flushed `fwrite`, not a bare `fprintf`, and a tick can briefly block a
// sample. That is the price of emitting the interval and the elapsed clock from
// the same guarded state the table is built from; at this lane's cadence
// (~110 lines per render) it is not a contention story, and it is written down
// here rather than left to be discovered from a profile.
void Tick(const std::string& unit, int64_t index, const std::string& detail);


}  // namespace phase
}  // namespace multimodal
}  // namespace vllm
