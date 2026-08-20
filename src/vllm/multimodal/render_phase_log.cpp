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
  std::vector<Open> open;
  // Per-unit tick clock for the live lane, so `last=` is the interval between
  // two occurrences of the SAME unit rather than since any other line.
  std::map<std::string, double> last_tick;
  std::vector<Record> records;
  DeviceByteProbe device_probe;

  std::thread sampler;
  std::condition_variable stop_cv;
  bool stop = false;

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

  // Caller holds `mu`. Reads both counters once and folds them into every open
  // scope, so a nested span sees the peak its children reached.
  void SampleLocked() {
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
    stop = false;
    sampler = std::thread([this]() {
      std::unique_lock<std::mutex> lock(mu);
      while (!stop) {
        stop_cv.wait_for(lock, std::chrono::milliseconds(100));
        if (stop) break;
        SampleLocked();
      }
    });
  }

  // The thread object is HANDED OUT under `mu` and joined outside it. Reading
  // `sampler.joinable()` with the lock released would race a concurrent
  // `StartSamplerLocked`, which writes that same object; joining while holding
  // `mu` would deadlock, because the worker needs `mu` to finish its wait. A
  // second mutex for the thread's lifetime would invert the lock order against
  // `Open`, which already holds `mu` when it starts the sampler. Moving the
  // handle out is the one shape that has neither problem.
  void StopSampler() {
    std::thread victim;
    {
      std::lock_guard<std::mutex> lock(mu);
      stop = true;
      victim = std::move(sampler);
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
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->records.clear();
  impl_->open.clear();
  impl_->last_tick.clear();
  impl_->origin = std::chrono::steady_clock::now();
  impl_->running = true;
  impl_->render = 0;
  impl_->samples = 0;
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
  std::lock_guard<std::mutex> lock(impl_->mu);
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
  o.span = span;
  o.live = true;
  bool leaf_already_open = false;
  for (const Impl::Open& other : impl_->open) {
    if (other.live && !other.span) leaf_already_open = true;
  }
  o.nested = leaf_already_open && !span;
  const double opened_at = o.start;
  const std::string opened_name = o.name;
  impl_->open.push_back(std::move(o));
  impl_->SampleLocked();
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
  return impl_->open.back().handle;
}

void PhaseLog::Close(size_t handle) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->SampleLocked();
  for (size_t i = 0; i < impl_->open.size(); ++i) {
    Impl::Open& o = impl_->open[i];
    if (o.handle != handle || !o.live) continue;
    Record r;
    r.name = o.name;
    r.render = o.render;
    r.start = o.start;
    r.end = impl_->Now();
    r.peak_host_bytes = o.peak_host;
    r.peak_device_bytes = o.peak_device;
    r.span = o.span;
    r.nested = o.nested;
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
    return;
  }
}

void PhaseLog::Sample() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->SampleLocked();
}

void PhaseLog::Tick(const std::string& unit, int64_t index, const std::string& detail) {
  if (!ProgressEnabled()) return;
  std::lock_guard<std::mutex> lock(impl_->mu);
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
}

std::vector<Record> PhaseLog::Records() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->records;
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

}  // namespace

bool PhaseLog::WriteJson(const std::string& path, const std::string& family,
                         const std::string& device, std::string* why) const {
  const std::vector<Record> records = ByStart(Records());
  const Totals totals = Sum(records, Elapsed());

  nlohmann::json out;
  out["schema"] = "vllm.cpp render phase log v1";
  out["family"] = family;
  out["device"] = device;
  out["wall_seconds"] = totals.wall;
  out["sum_leaf_seconds"] = totals.leaves;
  out["unaccounted_seconds"] = totals.unaccounted;
  out["host_bytes_source"] =
#if defined(__linux__)
      "/proc/self/statm";
#else
      "none";
#endif
  out["device_bytes_source"] = device == "cpu" ? "none" : "vt::Backend::DeviceMemoryInfo";
  out["samples"] = Samples();
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
    phases.push_back(std::move(e));
  }
  out["phases"] = std::move(phases);

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
  if (StderrEnabled()) {
    const std::string block = RenderText(family, device);
    std::fwrite(block.data(), 1, block.size(), stderr);
    std::fflush(stderr);
  }
  return true;
}

std::string PhaseLog::RenderText(const std::string& family, const std::string& device) const {
  const std::vector<Record> records = ByStart(Records());
  const Totals totals = Sum(records, Elapsed());
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
