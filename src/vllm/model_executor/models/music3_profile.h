// Per-STAGE wall-clock and RSS attribution for MiniMax-Music3 (#672).
//
// WHY THIS EXISTS. `examples/minimax_music3_gen/main.cpp:164-206` times exactly
// two things: the checkpoint load and the whole of `vllm_synthesize`. A 20 s
// clip at the shipped 30 steps takes tens of minutes on the device arm, and
// those two numbers cannot say which of the SIX stages spent them — the spec
// (§11.1, §14.2) records that four of the six are still host reference loops,
// but "records" is not "measured on this run at this duration". Every attempt
// to reason about the split from FLOP counts alone has to guess a host scalar
// rate, and the guesses have been an order of magnitude apart.
//
// WHAT IT IS NOT. It is not a benchmark harness and it takes no clock window
// (.agents/benchmarking.md §The clock is part of the measurement): a stage
// SPLIT is a within-run ratio, so it survives a clock it did not assert, and
// nothing here may be quoted as a per-kernel or cross-box figure. It is not a
// sampling profiler either — the brackets are placed by hand at the six stage
// boundaries the pipeline already has, so what it reports is exactly as honest
// as where the calls were put, and the `unattributed` line below exists so that
// a badly placed bracket shows up as a number rather than as silence.
//
// OFF BY DEFAULT, and the check is a cached env read: with
// `VLLM_CPP_MUSIC3_PROFILE` unset every `Timer` constructor takes one predicted
// branch and reads no clock, so a production synthesis pays nothing measurable.
//
// SINGLE-THREADED BY CONTRACT. Every bracket sits on the request thread — the
// pipeline's stage loops are serial and the parallelism lives INSIDE them, in
// `host_parallel::ForOutputRows` (host_parallel.h). Nothing here is a mutex, so
// a caller that brackets from a pool worker would corrupt the table. There is
// no such caller and there must not be one.
#pragma once

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace music3 {
namespace profile {

// `VLLM_CPP_MUSIC3_PROFILE` in {1, true, on, yes} (case-insensitive) turns the
// instrument on. Anything else — including unset — leaves it off, because a
// typo'd value must not silently produce a run with no numbers that the
// operator believes was profiled.
inline bool ParseEnabled(const char* value) {
  if (value == nullptr || value[0] == '\0') return false;
  std::string lowered(value);
  for (char& c : lowered) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return lowered == "1" || lowered == "true" || lowered == "on" || lowered == "yes";
}

// The env is read ONCE, at first use. The flag is a reference rather than a
// `const bool` so a test can drive both arms in one process; nothing on the
// synthesis path writes it, and a caller that did would be changing what the
// numbers below mean halfway through a run.
inline bool& EnabledFlag() {
  static bool enabled = ParseEnabled(std::getenv("VLLM_CPP_MUSIC3_PROFILE"));
  return enabled;
}

inline bool Enabled() { return EnabledFlag(); }

// A SECOND opt-in, for the intra-forward spans inside the device DiT
// (`minimax_music3_device.cpp`, spec §21.3). It is separate from the flag above
// and that separation is a property of the measurement rather than caution.
//
// Attributing time INSIDE one device forward needs a `Backend::Synchronize` at
// every bracket, because the ops are asynchronous on one stream and an
// un-drained bracket times the LAUNCH: it would report every GEMM as free and
// charge its whole cost to whichever bracket happened to contain the next
// synchronize. The drain is therefore mandatory for the split to mean anything,
// and it perturbs the very total it is splitting.
//
// So `VLLM_CPP_MUSIC3_PROFILE=1` alone leaves the forward byte for byte the path
// spec §20 timed — `denoise.dit_device` stays comparable to §15.7's and §20's
// tables value for value — and `VLLM_CPP_MUSIC3_DIT_SPANS=1` opts additionally
// into a perturbed run whose perturbation is MEASURED by taking both arms rather
// than asserted to be small.
//
// A reference for the same reason `EnabledFlag` is one: a test drives both arms
// in one process. Nothing on the synthesis path writes it.
inline bool& DitSpansFlag() {
  static bool enabled = ParseEnabled(std::getenv("VLLM_CPP_MUSIC3_DIT_SPANS"));
  return enabled;
}

// The spans are a REFINEMENT of the profile, never a way to turn it on: with the
// instrument off there is no table for a span to land in, so asking for spans
// alone is a no-op rather than a partial arming.
inline bool DitSpans() { return Enabled() && DitSpansFlag(); }

// A bucket is a LEAF or a SPAN. Leaves partition the run and are summed; spans
// enclose leaves and are printed for context but never added, because a table
// whose parts sum past its whole is a table nobody can read. `unattributed` is
// then a real quantity — the glue between the leaves — rather than the
// arithmetic residue of double counting.
struct Bucket {
  std::string name;
  double seconds = 0.0;
  int64_t calls = 0;
  bool span = false;
};

struct Marker {
  std::string label;
  double at_seconds = 0.0;
  int64_t rss_kb = -1;
};

struct State {
  std::vector<Bucket> buckets;
  std::vector<Marker> markers;
  std::chrono::steady_clock::time_point origin{};
  bool running = false;
};

inline State& Table() {
  static State state;
  return state;
}

// Resident set size in KiB, or -1 where the platform does not publish one.
// `/proc/self/statm` field 2 is the resident page count; multiplying by the
// page size is what `ps` reports as RSS. Reading `statm` rather than `status`
// is deliberate: `status` is ~50 lines of formatted text and this is called at
// six stage boundaries, one of which sits between two multi-gigabyte loads.
inline int64_t RssKb() {
#if defined(__linux__)
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr) return -1;
  long long total_pages = 0, resident_pages = 0;
  const int read = std::fscanf(f, "%lld %lld", &total_pages, &resident_pages);
  std::fclose(f);
  if (read != 2) return -1;
  return static_cast<int64_t>(resident_pages) * 4;  // 4 KiB pages on every box this runs on
#else
  return -1;
#endif
}

// Read-only view for a gate. There is no setter: the only way a bucket changes
// is through `Add` / `Count`, which is what the accounting rules live in.
inline const std::vector<Bucket>& Buckets() { return Table().buckets; }
inline const std::vector<Marker>& Markers() { return Table().markers; }

inline void Begin() {
  if (!Enabled()) return;
  State& state = Table();
  state.buckets.clear();
  state.markers.clear();
  state.origin = std::chrono::steady_clock::now();
  state.running = true;
}

inline double Elapsed() {
  const State& state = Table();
  if (!state.running) return 0.0;
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - state.origin).count();
}

inline void Add(const char* name, double seconds, bool span = false) {
  if (!Enabled()) return;
  State& state = Table();
  for (Bucket& bucket : state.buckets) {
    if (bucket.name == name) {
      bucket.seconds += seconds;
      ++bucket.calls;
      return;
    }
  }
  Bucket bucket;
  bucket.name = name;
  bucket.seconds = seconds;
  bucket.calls = 1;
  bucket.span = span;
  state.buckets.push_back(std::move(bucket));
}

// A COUNT with no time of its own — for a quantity the split has to state to be
// readable (frames, windows) but that no bracket owns.
inline void Count(const char* name, int64_t n) {
  if (!Enabled()) return;
  State& state = Table();
  for (Bucket& bucket : state.buckets) {
    if (bucket.name == name) {
      bucket.calls += n;
      return;
    }
  }
  Bucket bucket;
  bucket.name = name;
  bucket.seconds = -1.0;  // sentinel: a pure counter, excluded from the sum
  bucket.calls = n;
  state.buckets.push_back(std::move(bucket));
}

inline void Mark(const char* label) {
  if (!Enabled()) return;
  Marker marker;
  marker.label = label;
  marker.at_seconds = Elapsed();
  marker.rss_kb = RssKb();
  Table().markers.push_back(std::move(marker));
}

// The non-RAII form, for a bracket whose subject is a `const` object built by
// its own initializer — wrapping one of those in a scope would force it onto
// the heap, and moving a 18 GB weight set to satisfy an instrument is exactly
// the kind of change an instrument must not make.
inline std::chrono::steady_clock::time_point Now() {
  return std::chrono::steady_clock::now();
}

inline void AddSince(const char* name, std::chrono::steady_clock::time_point t0,
                     bool span = false) {
  if (!Enabled()) return;
  Add(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), span);
}

class Timer {
 public:
  explicit Timer(const char* name, bool span = false) : name_(name), span_(span) {
    if (!Enabled()) return;
    started_ = true;
    t0_ = std::chrono::steady_clock::now();
  }
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;
  ~Timer() {
    if (!started_) return;
    Add(name_, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(),
        span_);
  }

 private:
  const char* name_;
  bool span_ = false;
  std::chrono::steady_clock::time_point t0_{};
  bool started_ = false;
};

// The report. Prints to stderr, because stdout on this path may be a WAV.
//
// It prints the SUM of the timed buckets and the difference against the total
// beside it. A stage split whose parts do not add up to the whole is the shape
// of a missing bracket, and printing `unattributed` is how that becomes visible
// instead of being distributed silently over the stages that were measured.
inline void Report(const char* banner) {
  if (!Enabled()) return;
  State& state = Table();
  const double total = Elapsed();
  double attributed = 0.0;
  for (const Bucket& bucket : state.buckets) {
    if (bucket.seconds >= 0.0 && !bucket.span) attributed += bucket.seconds;
  }
  std::string out = "\nMUSIC3_PROFILE ";
  out += banner;
  out += "\n";
  char line[256];
  std::snprintf(line, sizeof(line), "  %-4s %-28s %12s %10s %8s\n", "kind", "stage", "seconds",
                "calls", "%");
  out += line;
  for (const Bucket& bucket : state.buckets) {
    const char* kind = bucket.seconds < 0.0 ? "cnt" : (bucket.span ? "span" : "leaf");
    if (bucket.seconds < 0.0) {
      std::snprintf(line, sizeof(line), "  %-4s %-28s %12s %10lld %8s\n", kind,
                    bucket.name.c_str(), "-", static_cast<long long>(bucket.calls), "-");
    } else {
      std::snprintf(line, sizeof(line), "  %-4s %-28s %12.3f %10lld %7.2f%%\n", kind,
                    bucket.name.c_str(), bucket.seconds, static_cast<long long>(bucket.calls),
                    total > 0.0 ? 100.0 * bucket.seconds / total : 0.0);
    }
    out += line;
  }
  std::snprintf(line, sizeof(line), "  %-4s %-28s %12.3f %10s %7.2f%%\n", "", "sum(leaf)",
                attributed, "-", total > 0.0 ? 100.0 * attributed / total : 0.0);
  out += line;
  std::snprintf(line, sizeof(line), "  %-4s %-28s %12.3f %10s %7.2f%%\n", "", "unattributed",
                total - attributed, "-",
                total > 0.0 ? 100.0 * (total - attributed) / total : 0.0);
  out += line;
  std::snprintf(line, sizeof(line), "  %-4s %-28s %12.3f %10s %7.2f%%\n", "", "TOTAL", total, "-",
                100.0);
  out += line;
  out += "  rss at stage boundaries (MiB):\n";
  for (const Marker& marker : state.markers) {
    std::snprintf(line, sizeof(line), "    %-30s t=%10.3f s  rss=%9.1f\n", marker.label.c_str(),
                  marker.at_seconds,
                  marker.rss_kb < 0 ? -1.0 : static_cast<double>(marker.rss_kb) / 1024.0);
    out += line;
  }
  // ONE fwrite, not a `<<` chain: the arm banner defect this row already
  // recorded twice (spec §14.7) came from assembling a report piecewise.
  std::fwrite(out.data(), 1, out.size(), stderr);
  std::fflush(stderr);
  state.running = false;
}

}  // namespace profile
}  // namespace music3
}  // namespace models
}  // namespace vllm
