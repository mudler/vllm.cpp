#include "vllm/multimodal/render_phase_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

// `sysconf(_SC_PAGESIZE)` rather than a hardcoded 4096: `statm` counts PAGES,
// and the boxes this instrument exists for are aarch64 — GB10 and Thor — where a
// 64 KiB page would make every byte count in the table 16x too small. The guard
// names `_WIN32` as well as `__linux__` because `check-windows-portability.py`
// reads only the `_WIN32` form, and a guard a checker cannot see is one nobody
// can rely on.
#if defined(__linux__) && !defined(_WIN32)
#include <unistd.h>
#endif

namespace vllm {
namespace multimodal {
namespace phase {

// `/proc/self/statm` field 2 is the resident page count, which is what `ps`
// reports as RSS. Reading `statm` rather than `status` is deliberate and the
// reason is `music3_profile.h`'s: `status` is ~50 formatted lines and this runs
// ten times a second beside a multi-gigabyte load.
int64_t HostResidentBytes() {
#if defined(__linux__) && !defined(_WIN32)
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr) return -1;
  long long total_pages = 0;
  long long resident_pages = 0;
  const int read = std::fscanf(f, "%lld %lld", &total_pages, &resident_pages);
  std::fclose(f);
  if (read != 2) return -1;
  const long page = ::sysconf(_SC_PAGESIZE);
  if (page <= 0) return -1;
  return static_cast<int64_t>(resident_pages) * static_cast<int64_t>(page);
#else
  return -1;
#endif
}

namespace {

// The sampler is a MEASUREMENT LANE, in the same shape `VT_POOL_BYPASS` and
// `VLLM_LTX2_POOL_DRAIN` already take: it exists so an A/B over what the sampler
// itself costs runs on ONE binary. It is never a configuration — the boundary
// samples are taken either way, so turning it off narrows the peaks rather than
// removing the table.
bool SamplerEnabled() {
  const char* off = std::getenv("VLLM_RENDER_PHASE_SAMPLER");
  return off == nullptr || off[0] != '0';
}

bool StderrEnabled() {
  const char* on = std::getenv("VLLM_RENDER_PHASE_LOG_STDERR");
  return on != nullptr && on[0] != '0';
}

// THE LIVE LANE'S OWN SWITCH (#1413), and it is ON unless somebody turns it off.
//
// A measurement lane in the shape `VLLM_RENDER_PHASE_SAMPLER` and
// `VLLM_LTX2_POOL_DRAIN` already take here: it exists so an A/B over what the
// emitter itself costs runs on ONE binary. It is never a configuration, and
// nothing in this tree sets it. The polarity is the point — `VT_H3_PROGRESS` is
// the same instrument for MiniMax-H3 and it is opt-in, which is exactly why no
// LTX-2.5 run has ever had one.
//
// READ ONCE. This is consulted at every phase boundary and every DiT forward, so
// a `getenv` per call would put a process-environment lookup inside the
// instrument whose cost this row claims is negligible.
}  // namespace

// Defined outside the anonymous namespace because the LTX-2.5 call site has to
// ask BEFORE it formats `detail`; see the declaration in the header.
bool ProgressEnabled() {
  static const bool on = []() {
    const char* off = std::getenv("VLLM_RENDER_PROGRESS");
    return off == nullptr || off[0] != '0';
  }();
  return on;
}

struct PhaseLog::Impl {
  mutable std::mutex mu;

  // An OPEN scope. `depth` at open time is what decides `nested`: a leaf that
  // finds another leaf already open did not measure a disjoint interval, and
  // adding it to the sum would make the residue negative rather than visible.
  struct Open {
    size_t handle = 0;
    // THE INSTRUMENT'S OWN WALL, charged to this record. See `ChargeLocked`.
    double instrument = 0.0;
    // The end of the last interval charged to this record, so two overlapping
    // charges cannot be counted twice. Seeded with the record's own `start`, so
    // a charge whose clock read happened BEFORE this record existed cannot reach
    // back past it either. See `ChargeLocked`.
    double instrument_charged_to = 0.0;
    std::string name;
    int64_t render = 0;
    double start = 0.0;
    int64_t peak_host = -1;
    int64_t peak_device = -1;
    bool span = false;
    bool nested = false;
    bool live = false;
  };

  std::chrono::steady_clock::time_point origin{};
  bool running = false;
  int64_t render = 0;
  size_t next_handle = 1;
  int64_t samples = 0;
  // The instrument's own wall that no LEAF absorbed, i.e. the part of
  // `unaccounted_seconds` this instrument spent rather than the render. See
  // `ChargeLocked`.
  double instrument_gap = 0.0;
  // The same high-water mark for the table's own share. See `ChargeLocked`.
  double instrument_gap_charged_to = 0.0;
  std::vector<Open> open;
  // Per-unit tick clock for the live lane, so `last=` is the interval between
  // two occurrences of the SAME unit rather than since any other line.
  std::map<std::string, double> last_tick;
  std::vector<Record> records;
  DeviceByteProbe device_probe;

  std::thread sampler;
  std::condition_variable stop_cv;
  // ONE STOP FLAG PER WORKER, not one per PhaseLog, and the difference is a
  // hang. A single `stop` member is written true by `StopSampler` and back to
  // false by the next `StartSamplerLocked`; between the moment `StopSampler`
  // hands the thread object out under `mu` and the moment it joins outside it,
  // an `Open` on another thread can run that `StartSamplerLocked` and clear the
  // flag the OLD worker is still reading. The old worker then never leaves its
  // loop, the join blocks forever, and the process ends with two samplers live.
  // Reachable through `Reset()` racing `Open`. A flag OWNED by the worker cannot
  // be cleared by anybody else's start.
  std::shared_ptr<bool> stop_flag;

  double Now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - origin).count();
  }

  // Caller holds `mu`. FLUSHED ON EVERY LINE, because the whole population this
  // lane exists for is runs that are killed: a line still sitting in a stdio
  // buffer when the process takes a SIGKILL is a line nobody gets, and it is the
  // last one — the one naming the phase that was running.
  void EmitLocked(const char* text) {
    std::fputs(text, stderr);
    std::fflush(stderr);
  }

  // ── THE INSTRUMENT'S OWN WALL (row LTX25-PHASE-INSTRUMENT, #1668) ─────────
  //
  // WHY THIS EXISTS. `unaccounted_seconds` and the uncovered part of a leaf both
  // contain a term this instrument creates and never reported: the wall it
  // spends inside its own entry points while no record — or no CHILD of the
  // record — is open. `Open` stamps `start` after taking this mutex, so the
  // mutex wait is before the new record begins; `Close` stamps `end` before it
  // prints its progress line and erases the entry, so that tail is after the
  // record ends. Both land outside every record, and until this row nothing
  // could tell them from a phase nobody named. Two gates were comparing that
  // mixture against a share of the render's wall, which is why both decided by
  // box load at fixture scale (#1439, #1494, #1470, #1536).
  //
  // THE RULE IS ONE SENTENCE: every interval of the instrument's own wall is
  // charged to the innermost live NON-SPAN record at the moment it is spent, and
  // to the table when none is live. A span is excluded because `Sum` excludes
  // spans, so time inside a span but outside a leaf is exactly the residue —
  // charging it to the enclosing `load` or `generate` span would hide it in a
  // number nothing adds up.
  //
  // "INNERMOST" IS THE LAST LIVE NON-SPAN ENTRY, because `open` is pushed in
  // open order: a nested sub-scope is appended after the leaf that contains it.
  // Caller holds `mu`.
  void ChargeLocked(double from, double to) {
    // A NEGATIVE `from` IS REFUSED RATHER THAN CLAMPED, and the polarity is the
    // reason. `Open` reads its clock BEFORE it takes this mutex, so a `Begin` on
    // another thread between those two points moves the origin under it and the
    // offset comes out negative. Clamping to zero would then charge the whole
    // timeline so far, and a charge that grows LOOSENS every bound that has this
    // quantity in a denominator — a defect that makes a gate pass is the one
    // nobody finds. The interval is not attributable to this timeline, so it is
    // dropped.
    if (from < 0.0) return;
    if (!(to > from)) return;
    // ── WHY EACH TARGET CARRIES A HIGH-WATER MARK ────────────────────────────
    //
    // A fresh review broke the invariant `instrument_seconds <= duration_seconds`
    // and it took no race to do it. `Open` and `Tick` read their clock BEFORE
    // they take this mutex, so the interval they charge to a record spans a
    // window in which ANOTHER thread — the 100 ms sampler, or a second caller —
    // can hold the mutex and charge the SAME record. Both charges are correct
    // individually and they OVERLAP, so their sum exceeded the record's own
    // single-threaded duration: 24 threads calling `SampleNow()` inside one live
    // leaf drove that ratio to 1.914, red in 3 runs of 5.
    //
    // Clamping `from` to the end of the last interval already charged to this
    // target makes the charges disjoint, so their sum is at most the UNION of
    // the intervals. The mark starts at the record's OWN `start`, so the union
    // also cannot reach back before the record began -- which a `Tick` whose
    // clock read predates the record and whose lock acquisition follows it would
    // otherwise do. Every charged interval then lies inside `[start, end]` and
    // no two of them overlap, so `instrument_seconds <= duration_seconds` holds
    // by construction rather than by hoping the box stays quiet.
    //
    // IT CAN UNDER-COUNT, and that direction is deliberate. Two charges that
    // arrive out of order — a later `to` first, then an earlier interval — lose
    // the earlier one entirely. Under-counting the instrument is the safe
    // direction here BECAUSE this quantity is in no denominator anywhere: a
    // smaller charge makes every gate that reads it stricter, never looser, and
    // `.agents/specs/ltx25-phase-residue.md` `## Design` 3 is why it is in no
    // denominator. The opposite choice — counting the overlap — makes a bound
    // pass, and a defect that makes a gate pass is the one nobody finds.
    for (size_t i = open.size(); i > 0; --i) {
      Open& o = open[i - 1];
      if (!o.live || o.span) continue;
      const double start = from > o.instrument_charged_to ? from : o.instrument_charged_to;
      if (to > start) {
        o.instrument += to - start;
        o.instrument_charged_to = to;
      }
      return;
    }
    const double start =
        from > instrument_gap_charged_to ? from : instrument_gap_charged_to;
    if (to > start) {
      instrument_gap += to - start;
      instrument_gap_charged_to = to;
    }
  }

  // Caller holds `mu`. Reads both counters once and folds them into every open
  // scope, so a nested span sees the peak its children reached.
  //
  // IT CHARGES ITSELF. A sample is taken at every boundary, by the 100 ms
  // worker, and by hand from inside the denoise loop; the last two land inside
  // the innermost record and outside its children, which is uncovered time this
  // instrument produced. `Open` and `Close` call it with the record they are
  // opening or closing already innermost, so those two charge to themselves and
  // the charge is inside that record's own duration.
  void SampleLocked() {
    const double entered = running ? Now() : 0.0;
    SampleUnchargedLocked();
    if (running) ChargeLocked(entered, Now());
  }

  void SampleUnchargedLocked() {
    const int64_t host = HostResidentBytes();
    int64_t device = -1;
    if (device_probe) {
      device = device_probe();
    }
    ++samples;
    for (Open& o : open) {
      if (!o.live) continue;
      if (host > o.peak_host) o.peak_host = host;
      if (device > o.peak_device) o.peak_device = device;
    }
  }

  void StartSamplerLocked() {
    if (sampler.joinable() || !SamplerEnabled()) return;
    stop_flag = std::make_shared<bool>(false);
    std::shared_ptr<bool> mine = stop_flag;
    sampler = std::thread([this, mine]() {
      std::unique_lock<std::mutex> lock(mu);
      while (!*mine) {
        stop_cv.wait_for(lock, std::chrono::milliseconds(100));
        if (*mine) break;
        SampleLocked();
      }
    });
  }

  // Is anything open that a sample could be attributed to? A sample taken with
  // no live scope updates no peak and is charged to nobody, so it is a
  // `/proc/self/statm` read under the process-wide mutex that buys nothing.
  bool AnythingLive() const {
    for (const Open& o : open) {
      if (o.live) return true;
    }
    return false;
  }

  // The thread object is HANDED OUT under `mu` and joined outside it. Reading
  // `sampler.joinable()` with the lock released would race a concurrent
  // `StartSamplerLocked`, which writes that same object; joining while holding
  // `mu` would deadlock, because the worker needs `mu` to finish its wait. A
  // second mutex for the thread's lifetime would invert the lock order against
  // `Open`, which already holds `mu` when it starts the sampler. Moving the
  // handle out is the one shape that has neither problem.
  // Caller holds `mu`. Marks the running worker stopped and hands its thread
  // object out; the caller joins it after releasing the lock, because the worker
  // needs `mu` to finish its wait.
  std::thread TakeSamplerLocked() {
    if (stop_flag) *stop_flag = true;
    stop_flag.reset();
    return std::move(sampler);
  }

  void StopSampler() {
    std::thread victim;
    {
      std::lock_guard<std::mutex> lock(mu);
      victim = TakeSamplerLocked();
    }
    stop_cv.notify_all();
    if (victim.joinable()) victim.join();
  }
};

PhaseLog::PhaseLog() : impl_(new Impl()) {}

PhaseLog::~PhaseLog() {
  impl_->StopSampler();
  delete impl_;
}

PhaseLog& PhaseLog::Instance() {
  static PhaseLog log;
  return log;
}

void PhaseLog::Begin() {
  // The previous timeline's sampler is stopped BEFORE this one starts. `Begin`
  // discards the records and the open scopes, so a surviving worker would sample
  // into nothing, and its 100 ms `/proc/self/statm` read under the process-wide
  // mutex would outlive the render that asked for it.
  impl_->StopSampler();
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->records.clear();
  impl_->open.clear();
  impl_->last_tick.clear();
  impl_->origin = std::chrono::steady_clock::now();
  impl_->running = true;
  impl_->render = 0;
  impl_->samples = 0;
  impl_->instrument_gap = 0.0;
  impl_->instrument_gap_charged_to = 0.0;
  impl_->device_probe = DeviceByteProbe();
}

void PhaseLog::SetDeviceProbe(DeviceByteProbe probe) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->device_probe = std::move(probe);
}

void PhaseLog::SetRender(int64_t render) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->render = render;
  // The per-unit tick clock belongs to ONE generation. A process that renders
  // twice calls this between them, and carrying `last_tick` across would give
  // the second render's first DiT forward a `last=` measuring the gap between
  // two renders — a number that looks exactly like a very slow forward. `Begin`
  // and `Reset` clear it for the same reason; this is the third door into a new
  // timeline and it was the one left open.
  impl_->last_tick.clear();
}

double PhaseLog::Elapsed() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (!impl_->running) return 0.0;
  return impl_->Now();
}

size_t PhaseLog::Open(const std::string& name, bool span) {
  // THE HEAD OF THE BOUNDARY, taken before the process-wide mutex. Everything
  // between here and `o.start` below — the lock wait, which the 100 ms worker
  // can hold, and the sampler start — is wall this instrument spends BEFORE the
  // new record begins, so it lands in the gap before it. Row
  // LTX25-PHASE-INSTRUMENT charges it to whatever encloses that gap.
  const std::chrono::steady_clock::time_point entered = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(impl_->mu);
  const bool was_running = impl_->running;
  if (!impl_->running) {
    impl_->origin = std::chrono::steady_clock::now();
    impl_->running = true;
  }
  impl_->StartSamplerLocked();
  Impl::Open o;
  o.handle = impl_->next_handle++;
  o.name = name;
  o.render = impl_->render;
  o.start = impl_->Now();
  o.instrument_charged_to = o.start;
  o.span = span;
  o.live = true;
  bool leaf_already_open = false;
  for (const Impl::Open& other : impl_->open) {
    if (other.live && !other.span) leaf_already_open = true;
  }
  o.nested = leaf_already_open && !span;
  const double opened_at = o.start;
  const std::string opened_name = o.name;
  // Charged BEFORE the new entry is pushed, so `ChargeLocked` resolves the
  // innermost live leaf to this record's PARENT — which is where the head of
  // this boundary was actually spent. Skipped when this `Open` started the
  // timeline, because then the origin IS `o.start` and there is no gap.
  if (was_running) {
    impl_->ChargeLocked(
        std::chrono::duration<double>(entered - impl_->origin).count(), opened_at);
  }
  impl_->open.push_back(std::move(o));
  impl_->SampleLocked();
  // THE TAIL OF THIS BOUNDARY. Everything from here to the return runs INSIDE
  // the record just opened and BEFORE any child of it, so it is uncovered time
  // this instrument produced — the same quantity `Close`'s tail is, on the other
  // side of the boundary. `SampleLocked` above charges itself; the flushed
  // progress line below is the most expensive statement in the function.
  const double after_sample = impl_->Now();
  // W0-live (#1413): the OPEN line, which is the load-bearing half. It means the
  // last line printed names the phase that is CURRENTLY RUNNING, and that is the
  // whole difference between a working render and a hung one. A close-only
  // emitter would have printed nothing at all for the 3002 s conditioning
  // stretch, because that stretch never closed.
  if (ProgressEnabled()) {
    char text[256];
    std::snprintf(text, sizeof(text), "[render] + %-24s t=%.3fs\n", opened_name.c_str(),
                  opened_at);
    impl_->EmitLocked(text);
  }
  // Charged to the record just opened, which `ChargeLocked` resolves as the
  // innermost live leaf. It is inside that record's own duration and outside
  // every child of it, which is exactly where the coverage bound looks.
  impl_->ChargeLocked(after_sample, impl_->Now());
  return impl_->open.back().handle;
}

void PhaseLog::Close(size_t handle) {
  // THE LAST CLOSE STOPS THE SAMPLER, and the reason is a process this
  // instrument does not own. `Begin` starts the timeline and `Open` starts the
  // worker, but nothing except `Reset` and the destructor ever stopped it, so a
  // server that rendered one clip kept a 100 ms `/proc/self/statm` read under
  // the process-wide mutex for the rest of its life and accumulated that idle
  // time into the NEXT table's sample count. Restarting it costs one thread
  // creation, and on this driver that happens twice per process: the `load` span
  // and the `generate` span each stay open across everything beneath them, so
  // the scope stack is empty only BETWEEN a load and a generation.
  std::thread victim;
  // The clock at the end of the locked block, kept so the sampler JOIN below can
  // be charged too. See the note beside it.
  double left_lock_at = -1.0;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->SampleLocked();
    // WHERE THIS RECORD ENDED, kept outside the loop so the TAIL of the boundary
    // can be charged after the entry is erased. Row LTX25-PHASE-INSTRUMENT: the
    // progress line, the record push and the vector erase all run after `r.end`
    // is stamped, so they are wall this instrument spends AFTER the record ends
    // and they land in the gap after it. Erasing first is what makes
    // `ChargeLocked` resolve the innermost live leaf to this record's PARENT.
    double closed_at = -1.0;
    for (size_t i = 0; i < impl_->open.size(); ++i) {
      Impl::Open& o = impl_->open[i];
      if (o.handle != handle || !o.live) continue;
      Record r;
      r.name = o.name;
      r.render = o.render;
      r.start = o.start;
      r.end = impl_->Now();
      closed_at = r.end;
      r.peak_host_bytes = o.peak_host;
      r.peak_device_bytes = o.peak_device;
      r.span = o.span;
      r.nested = o.nested;
      r.instrument_seconds = o.instrument;
      // W0-live (#1413): what the phase COST, on the line, at the moment it ends.
      // A reader of a killed run's log takes every completed phase's duration off
      // this without waiting for a table that will never be written.
      if (ProgressEnabled()) {
        const double kGiB = 1024.0 * 1024.0 * 1024.0;
        const double host = r.peak_host_bytes < 0 ? -1.0
                                                  : static_cast<double>(r.peak_host_bytes) / kGiB;
        char text[320];
        if (r.peak_device_bytes >= 0) {
          std::snprintf(text, sizeof(text),
                        "[render] - %-24s t=%.3fs dur=%.3fs host=%.2fGiB dev=%.2fGiB\n",
                        r.name.c_str(), r.end, r.end - r.start, host,
                        static_cast<double>(r.peak_device_bytes) / kGiB);
        } else {
          std::snprintf(text, sizeof(text), "[render] - %-24s t=%.3fs dur=%.3fs host=%.2fGiB\n",
                        r.name.c_str(), r.end, r.end - r.start, host);
        }
        impl_->EmitLocked(text);
      }
      impl_->records.push_back(std::move(r));
      impl_->open.erase(impl_->open.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
    if (!impl_->AnythingLive()) victim = impl_->TakeSamplerLocked();
    if (closed_at >= 0.0) {
      left_lock_at = impl_->Now();
      impl_->ChargeLocked(closed_at, left_lock_at);
    }
  }
  if (victim.joinable()) {
    impl_->stop_cv.notify_all();
    victim.join();
    // AND THE JOIN IS CHARGED TOO, which costs a second lock acquisition and is
    // worth it. This is the LAST close of a timeline, so nothing is live and the
    // whole notify-and-join lands in `unaccounted_seconds` — uncharged, it reads
    // as time nobody named. Measured at about 117 us per join on a contended x86
    // box against a residue of 346 us: leaving it out made a two-scope timeline
    // whose gaps contain NOTHING report a residue three times the instrument's
    // own charge, which is exactly the reading a real un-named phase produces.
    // The LTX-2.5 driver pays it twice per process, once when the `load` span
    // closes and once when `generate` does.
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (left_lock_at >= 0.0 && impl_->running) impl_->ChargeLocked(left_lock_at, impl_->Now());
  }
}

void PhaseLog::Sample() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->SampleLocked();
}

void PhaseLog::Tick(const std::string& unit, int64_t index, const std::string& detail) {
  if (!ProgressEnabled()) return;
  // Charged like a boundary (row LTX25-PHASE-INSTRUMENT): a tick is a held
  // global lock plus a FLUSHED `fwrite`, it runs ~110 times per render from
  // inside the denoise loop, and it lands inside the innermost record and
  // outside its children — i.e. it is uncovered time this instrument produced.
  const std::chrono::steady_clock::time_point entered = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(impl_->mu);
  const bool charge = impl_->running;
  const double entered_at =
      charge ? std::chrono::duration<double>(entered - impl_->origin).count() : 0.0;
  // A tick before any scope opened starts the timeline, exactly as `Open` does.
  // Returning silently instead would make the first unit of work of a render
  // that took no scope disappear, which is the failure this lane exists to stop.
  if (!impl_->running) {
    impl_->origin = std::chrono::steady_clock::now();
    impl_->running = true;
  }
  const double now = impl_->Now();
  const std::map<std::string, double>::const_iterator previous = impl_->last_tick.find(unit);
  char text[384];
  if (previous == impl_->last_tick.end()) {
    std::snprintf(text, sizeof(text), "[render]   %s %lld  %s  t=%.3fs\n", unit.c_str(),
                  static_cast<long long>(index), detail.c_str(), now);
  } else {
    // `last=` IS THE DELIVERABLE. #1375 could only obtain the per-forward cost as
    // a wall-clock interval between GPU busy/idle edges from outside the process,
    // because `eu-stack` unwinds zero frames inside the `rc` worker container.
    // This is the same quantity, measured by the thing doing the work, on every
    // run including the ones that die.
    std::snprintf(text, sizeof(text), "[render]   %s %lld  %s  t=%.3fs last=%.3fs\n",
                  unit.c_str(), static_cast<long long>(index), detail.c_str(), now,
                  now - previous->second);
  }
  impl_->last_tick[unit] = now;
  impl_->EmitLocked(text);
  if (charge) impl_->ChargeLocked(entered_at, impl_->Now());
}

std::vector<Record> PhaseLog::Records() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->records;
}

double PhaseLog::Instrument() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->instrument_gap;
}

int64_t PhaseLog::Samples() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->samples;
}

void PhaseLog::Reset() {
  impl_->StopSampler();
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->records.clear();
  impl_->open.clear();
  impl_->last_tick.clear();
  impl_->running = false;
  impl_->render = 0;
  impl_->samples = 0;
  impl_->instrument_gap = 0.0;
  impl_->instrument_gap_charged_to = 0.0;
  impl_->device_probe = DeviceByteProbe();
}

namespace {

// The arithmetic every consumer of this table needs, in ONE place: the wall, the
// sum of the leaves that partition it, and the difference between them. Leaves
// only — a span encloses leaves and a nested leaf overlaps one, and adding
// either would make `unaccounted` the residue of double counting instead of a
// real quantity.
struct Totals {
  double wall = 0.0;
  double leaves = 0.0;
  double unaccounted = 0.0;
};

// The records are APPENDED in completion order, because that is when a scope
// knows its own end. A reader wants a TIMELINE, so the emitted order is by
// start — stably, so two scopes that opened in the same tick keep the order they
// were opened in. Sorting here rather than at insertion keeps the append cheap
// and keeps the appended order recoverable from `end_seconds`.
std::vector<Record> ByStart(std::vector<Record> records) {
  std::stable_sort(records.begin(), records.end(),
                   [](const Record& a, const Record& b) { return a.start < b.start; });
  return records;
}

Totals Sum(const std::vector<Record>& records, double wall) {
  Totals t;
  t.wall = wall;
  for (const Record& r : records) {
    if (r.span || r.nested) continue;
    t.leaves += r.end - r.start;
  }
  t.unaccounted = wall - t.leaves;
  return t;
}

// ── WHERE THE RESIDUE ACTUALLY IS (row LTX25-PHASE-INSTRUMENT, #1571) ───────
//
// WHY THIS IS IN THE FILE AND NOT IN A SCRIPT. `unaccounted_seconds` shipped as
// an AGGREGATE, and four issues -- #1439, #1470, #1494, #1536 -- argued about
// whether its 95% floor was the right tolerance without anyone splitting it into
// the gaps between consecutive leaves. Splitting it took ONE pass over the table
// the render already writes and settled the question immediately: 92% of the
// residue was a single gap, the load's prologue from the timeline's origin to
// `Open("load.dit")`, 17.661 ms of 19.178 ms, while the sixteen gaps between
// adjacent named phases held 6.8 us each. That pass was a scratch script that
// was never shipped, so a reader of `phase-log.json` still could not see it and
// the same investigation would be re-derived the next time the residue moved.
//
// IT IS ALSO AN ACCOUNTING IDENTITY, which is the half a gate can hold. `Sum`
// adds the records with `span == false && nested == false`, and `Open` marks a
// leaf `nested` whenever another leaf is already live, so those records are
// non-overlapping and -- after `ByStart` -- ordered. The complement of their
// union inside `[0, wall]` is therefore exactly `wall - sum_leaf_seconds`, which
// is `unaccounted_seconds`. The gaps below add to it by construction rather than
// by tolerance, so a gate over that sum is arithmetic and cannot move with box
// load. That is the difference between this and every bound this table has
// carried.
//
// A NEGATIVE GAP IS EMITTED RATHER THAN CLAMPED. It cannot arise while the
// non-overlap invariant above holds, so clamping it would hide a broken
// instrument inside a number that still adds up.
nlohmann::json GapsBetweenLeaves(const std::vector<Record>& records, double wall) {
  nlohmann::json gaps = nlohmann::json::array();
  double cursor = 0.0;
  std::string previous = "<origin>";
  for (const Record& r : records) {
    if (r.span || r.nested) continue;
    nlohmann::json g;
    g["after"] = previous;
    g["before"] = r.name;
    g["start_seconds"] = cursor;
    g["end_seconds"] = r.start;
    g["seconds"] = r.start - cursor;
    gaps.push_back(std::move(g));
    cursor = r.end;
    previous = r.name;
  }
  nlohmann::json tail;
  tail["after"] = previous;
  tail["before"] = "<end>";
  tail["start_seconds"] = cursor;
  tail["end_seconds"] = wall;
  tail["seconds"] = wall - cursor;
  gaps.push_back(std::move(tail));
  return gaps;
}

}  // namespace

bool PhaseLog::WriteJson(const std::string& path, const std::string& family,
                         const std::string& device, std::string* why) const {
  // THE CLOCK IS READ FIRST, AND THE ORDER IS THE WHOLE CONTENT OF THIS LINE.
  // `Records()` copies the record vector under the process-wide mutex and
  // `ByStart` stable-sorts the copy. Reading `Elapsed()` after them charged this
  // WRITER's own serialization to the RENDER's wall, and therefore to
  // `unaccounted_seconds` — a residue the render did not produce. This table
  // measures the render, so the writer's clock stops before the writer works.
  // Row LTX25-PHASE-INSTRUMENT, issue #1569: the gate that holds this ordering
  // is `the emitter reads its clock BEFORE it serialises the table` in
  // `tests/vllm/multimodal/test_render_phase_log.cpp`, and it needs a table of
  // thousands of records to see the difference at all.
  //
  // AND THE COST OF THAT ORDER, WHICH A FRESH REVIEW ASKED FOR IN WRITING. The
  // wall below and the records on the next line are taken under TWO separate
  // acquisitions of the process-wide mutex, so they are no longer one snapshot.
  // Before this order they were effectively one in the direction that matters:
  // the clock was read last, so `wall >= max(end_seconds)` held by
  // construction. It no longer does, and the observable if it broke is a
  // NEGATIVE tail gap, which `gaps` reports and the unit case refuses.
  //
  // IT IS UNREACHABLE ON THE SHIPPED PATH, and that is a property of the CALL
  // SITE rather than of this function. Both `WritePhaseLog` calls in
  // `ltx2_video.cpp` run after `generate_span.Close()`, that span is the last
  // live scope, and `PhaseLog::Close` stops and JOINS the sampler before it
  // returns when nothing is left live. So no thread can close a scope between
  // these two lines on any path this project ships. A fresh review also failed
  // to stage an inversion adversarially: 27,471 probes of a churn thread
  // against a replica of these two statements produced zero. Making the pair a
  // single locked snapshot is the real repair and it is a public API change;
  // it is recorded as owed rather than smuggled in here.
  const double wall = Elapsed();

  // THE CONSOLE COPY GOES FIRST, and "first" now means BEFORE THIS WRITER DOES
  // ANY WORK AT ALL rather than merely before the two failure returns below.
  //
  // `VLLM_RENDER_PHASE_LOG_STDERR` exists for the run whose table cannot reach a
  // file: an unwritable `--output-dir`, a read-only mount, a full disk. Emitted
  // after those returns it was silent in exactly that case, and
  // `docs/ENVIRONMENT.md`'s "also prints" described something the code did not
  // do. Emitting it here also means a process that dies during the write has
  // still said what it measured.
  //
  // AND IT SITS ABOVE THE COPY, THE SORT AND THE WHOLE JSON BUILD BECAUSE THOSE
  // WERE BEING CHARGED TO IT (issue #1755). `RenderText` reads the clock ITSELF,
  // so wherever this block stands is the instant the console's `WALL` reports.
  // Standing after `out` was assembled, the console copy of a render quoted a
  // wall that contained this writer's own serialization: on the 8001-record unit
  // timeline `sum(leaf)` held at 0.189 s across five calls while the console's
  // `unaccounted` climbed 0.065 -> 0.134 -> 0.200 -> 0.265 -> 0.329 s, about
  // 66 ms of writer work per call charged to the render. That is #1569's defect
  // on #1569's own sibling emitter: the file copy was repaired and the console
  // copy a reader watches was not. Here the two clock reads are one `getenv`
  // apart, so the console copy and the file copy describe the same instant.
  if (StderrEnabled()) {
    const std::string block = RenderText(family, device);
    std::fwrite(block.data(), 1, block.size(), stderr);
    std::fflush(stderr);
  }

  const std::vector<Record> records = ByStart(Records());
  const Totals totals = Sum(records, wall);

  nlohmann::json out;
  out["schema"] = "vllm.cpp render phase log v1";
  out["family"] = family;
  out["device"] = device;
  out["wall_seconds"] = totals.wall;
  out["sum_leaf_seconds"] = totals.leaves;
  out["unaccounted_seconds"] = totals.unaccounted;
  // HOW MUCH OF `unaccounted_seconds` THIS INSTRUMENT SPENT (row
  // LTX25-PHASE-INSTRUMENT, #1668). Without it the residue can only be compared
  // against a SHARE of the render's wall, and that share is a property of the
  // fixture: #1439 measured a 95% floor deciding by box load at 64x64x9 while
  // the same residue is invisible on the 21 B render this table exists for. With
  // it, a reader subtracts the cost of naming the phases before calling what is
  // left a phase nobody named.
  out["instrument_seconds"] = Instrument();
  out["host_bytes_source"] =
#if defined(__linux__)
      "/proc/self/statm";
#else
      "none";
#endif
  out["device_bytes_source"] = device == "cpu" ? "none" : "vt::Backend::DeviceMemoryInfo";
  out["samples"] = Samples();
  // WHAT THIS FILE IS NOT, WRITTEN INTO THE FILE. A phase log is evidence, and
  // evidence separated from its context is the failure this whole row exists to
  // stop (#1040's sampler CSVs, #1087's unnamed 1731 s phase). The first
  // artifact this row committed landed under `benchmarks/` carrying a `denoise`
  // of 8.13 s and a `decode.audio` of 3.06 s and NOTHING that said the host was
  // contended, the checkpoint was a two-block fixture, or that the rank of those
  // two phases had reversed between two runs of the same binary. A reader who
  // opens one of these files months later gets the caveat from the file rather
  // than from a document they would have to know to look for.
  // NO MEASURED NUMBER IN THIS STRING, and that is a repair rather than a
  // shortening. The sentence used to name three wall times from one contended
  // box. A measurement baked into library source is a number that goes stale
  // where nobody looks for it: it is not a projection document, no gate reads
  // it, and the next run that refutes it edits a file in `src/`. The counts and
  // the ratios belong to the artifact, whose `_caveat` carries them beside the
  // run they came from.
  out["notice"] =
      "NOT A BENCHMARK. Every duration here is wall clock on whatever host ran this render, "
      "under whatever else that host was doing at the time, and this file records neither. "
      "On a contended box the same binary at the same geometry has moved by more than an "
      "order of magnitude in wall, and the RANK of its two largest phases has reversed "
      "between two such runs. What this table supports is the SHAPE of a render and the "
      "ratio sum_leaf_seconds/wall_seconds. Quote a duration only with the host, the "
      "checkpoint and the contention state beside it. "
      "See .agents/specs/ltx25-device-residency.md.";
  out["sum_rule"] =
      "sum_leaf_seconds adds only records with span=false and nested=false. A span encloses "
      "leaves and a nested record decomposes one, so adding either would make "
      "unaccounted_seconds the residue of double counting instead of time nobody named.";
  // Whether the 100 ms sampler ran. With it off the peaks are what the phase
  // BOUNDARIES saw, which is a different measurement rather than a worse one.
  out["sampler_enabled"] = SamplerEnabled();
  nlohmann::json phases = nlohmann::json::array();
  for (const Record& r : records) {
    nlohmann::json e;
    e["name"] = r.name;
    e["render"] = r.render;
    e["start_seconds"] = r.start;
    e["end_seconds"] = r.end;
    e["duration_seconds"] = r.end - r.start;
    e["peak_host_bytes"] = r.peak_host_bytes;
    e["peak_device_bytes"] = r.peak_device_bytes;
    e["span"] = r.span;
    e["nested"] = r.nested;
    // The other half of `instrument_seconds` above: how much of THIS record's
    // own duration the instrument spent, outside every child of it. It is what
    // separates "this leaf encloses a phase nobody named" from "this leaf paid
    // for its own sub-scope boundaries" — a distinction the coverage gate had no
    // way to make (#1494).
    e["instrument_seconds"] = r.instrument_seconds;
    phases.push_back(std::move(e));
  }
  out["phases"] = std::move(phases);
  out["gaps"] = GapsBetweenLeaves(records, totals.wall);
  out["gap_rule"] =
      "gaps decomposes unaccounted_seconds. The leaves that sum_leaf_seconds adds are "
      "non-overlapping and start-ordered, so the complement of their union inside "
      "[0, wall_seconds] is exactly the residue: there is one gap before each leaf and one "
      "after the last, and their seconds add to unaccounted_seconds. `after` and `before` "
      "name the leaves a gap lies between; <origin> and <end> are the ends of the timeline.";

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.good()) {
    if (why != nullptr) *why = "cannot open " + path;
    return false;
  }
  const std::string text = out.dump(2);
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
  f.put('\n');
  if (!f.good()) {
    if (why != nullptr) *why = "cannot write " + path;
    return false;
  }
  return true;
}

std::string PhaseLog::RenderText(const std::string& family, const std::string& device) const {
  // THE CLOCK IS READ FIRST HERE TOO, FOR THE REASON `WriteJson` GIVES ABOVE
  // (issue #1755). `Records()` copies the record vector under the process-wide
  // mutex and `ByStart` stable-sorts the copy; reading `Elapsed()` after them
  // charged THIS emitter's own serialization to the RENDER's wall, and so to the
  // `unaccounted` line printed three lines below it. #1569 repaired that
  // ordering in the file emitter and left its sibling alone, and nothing noticed
  // because the console prints every total with `%10.3f` while a copy and a sort
  // of a few thousand records are tenths of a millisecond.
  //
  // The cost of the order is the one `WriteJson` records: the wall and the
  // records below are two acquisitions of the mutex rather than one snapshot.
  const double wall = Elapsed();
  const std::vector<Record> records = ByStart(Records());
  const Totals totals = Sum(records, wall);
  const double kGiB = 1024.0 * 1024.0 * 1024.0;
  std::string out = "\nRENDER PHASE LOG family=" + family + " device=" + device + "\n";
  char line[256];
  std::snprintf(line, sizeof(line), "  %-4s %-30s %10s %10s %10s %10s\n", "r", "phase", "start_s",
                "dur_s", "host_GiB", "dev_GiB");
  out += line;
  for (const Record& r : records) {
    const char* kind = r.span ? "span" : (r.nested ? "nest" : "leaf");
    std::snprintf(line, sizeof(line), "  %-4lld %-30s %10.3f %10.3f %10.3f %10.3f %s\n",
                  static_cast<long long>(r.render), r.name.c_str(), r.start, r.end - r.start,
                  r.peak_host_bytes < 0 ? -1.0 : static_cast<double>(r.peak_host_bytes) / kGiB,
                  r.peak_device_bytes < 0 ? -1.0 : static_cast<double>(r.peak_device_bytes) / kGiB,
                  kind);
    out += line;
  }
  std::snprintf(line, sizeof(line), "  %-4s %-30s %10s %10.3f\n", "", "sum(leaf)", "",
                totals.leaves);
  out += line;
  std::snprintf(line, sizeof(line), "  %-4s %-30s %10s %10.3f\n", "", "unaccounted", "",
                totals.unaccounted);
  out += line;
  std::snprintf(line, sizeof(line), "  %-4s %-30s %10s %10.3f\n", "", "WALL", "", totals.wall);
  out += line;
  // AND WHAT THE INSTRUMENT CHARGED ITSELF, so the console copy and the file
  // copy answer the same question. A fresh review found the two had diverged:
  // `phase-log.json` carried `instrument_seconds` and this block did not, so a
  // reader watching a terminal saw a residue with no way to subtract the cost of
  // naming the phases from it -- which is the whole reason that number exists.
  std::snprintf(line, sizeof(line), "  %-4s %-30s %10s %10.3f\n", "", "instrument", "",
                Instrument());
  out += line;
  return out;
}

Scope::Scope(const std::string& name, bool span)
    : handle_(PhaseLog::Instance().Open(name, span)) {}

Scope::~Scope() { Close(); }

void Scope::Close() {
  if (closed_) return;
  closed_ = true;
  PhaseLog::Instance().Close(handle_);
}

void SampleNow() { PhaseLog::Instance().Sample(); }

void Tick(const std::string& unit, int64_t index, const std::string& detail) {
  PhaseLog::Instance().Tick(unit, index, detail);
}

}  // namespace phase
}  // namespace multimodal
}  // namespace vllm
