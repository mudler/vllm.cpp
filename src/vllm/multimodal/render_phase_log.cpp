#include "vllm/multimodal/render_phase_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

}  // namespace

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
  impl_->open.push_back(std::move(o));
  impl_->SampleLocked();
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
  {
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
      impl_->records.push_back(std::move(r));
      impl_->open.erase(impl_->open.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
    if (!impl_->AnythingLive()) victim = impl_->TakeSamplerLocked();
  }
  if (victim.joinable()) {
    impl_->stop_cv.notify_all();
    victim.join();
  }
}

void PhaseLog::Sample() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->SampleLocked();
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
    phases.push_back(std::move(e));
  }
  out["phases"] = std::move(phases);

  // THE CONSOLE COPY GOES FIRST, and the order is the whole point of it.
  //
  // `VLLM_RENDER_PHASE_LOG_STDERR` exists for the run whose table cannot reach a
  // file: an unwritable `--output-dir`, a read-only mount, a full disk. Emitted
  // after the two failure returns below it was silent in exactly that case, and
  // `docs/ENVIRONMENT.md`'s "also prints" described something the code did not
  // do. Hoisting it also means a process that dies during the write has still
  // said what it measured.
  if (StderrEnabled()) {
    const std::string block = RenderText(family, device);
    std::fwrite(block.data(), 1, block.size(), stderr);
    std::fflush(stderr);
  }

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

}  // namespace phase
}  // namespace multimodal
}  // namespace vllm
