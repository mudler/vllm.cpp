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

}  // namespace phase
}  // namespace multimodal
}  // namespace vllm
