// BACKEND-CPU developer harness. See
// .agents/specs/rpi5-cortex-a76-cpu-optimization.md. This is deliberately an
// op-level tool, not a production API: later reached-loop fixtures plug into
// the same CLI/result schema without putting benchmark machinery in libvllm.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string op = "matmul-bt-quant";
  std::string dtype = "q8_0";
  int64_t m = 1;
  int64_t n = 3072;
  int64_t k = 2048;
  int threads = 1;
  std::string variant = "auto";
  int warmup = 2;
  int iterations = 10;
  std::string cache = "hot";
  std::string counters = "auto";
  std::string format = "text";
  uint32_t seed = 0x6a11U;
  bool pin = true;
};

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::string ReadText(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream out;
  out << in.rdbuf();
  std::string value = out.str();
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
  return value;
}

template <typename T>
T ParseInteger(std::string_view value, std::string_view flag) {
  T parsed{};
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    Fail("invalid integer for --" + std::string(flag) + ": " + std::string(value));
  }
  return parsed;
}

void Usage(std::ostream& out) {
  out << "vllm-cpu-kernel-bench, PMU-backed CPU operation harness\n\n"
      << "  --op matmul-bt-quant  --dtype q8_0|q4_k|q6_k\n"
      << "  --m N --n N --k N --threads N\n"
      << "  --variant auto|portable|sdot|a76-asm|repacked|mmla\n"
      << "  --warmup N --iterations N --cache hot|l2|l3|stream\n"
      << "  --counters auto|off|generic|a76|all --format text|json\n"
      << "  --seed N --no-pin\n\n"
      << "Cache names are reproducible pressure profiles, not residency claims: l2\n"
      << "evicts L1, l3 evicts private L2, and stream exceeds the last-level cache.\n";
}

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      Usage(std::cout);
      std::exit(0);
    }
    if (arg == "--no-pin") {
      o.pin = false;
      continue;
    }
    if (!arg.starts_with("--")) Fail("unexpected positional argument: " + arg);
    std::string key;
    std::string value;
    const size_t eq = arg.find('=');
    if (eq != std::string::npos) {
      key = arg.substr(2, eq - 2);
      value = arg.substr(eq + 1);
    } else {
      key = arg.substr(2);
      if (++i >= argc) Fail("missing value for --" + key);
      value = argv[i];
    }
    if (key == "op")
      o.op = value;
    else if (key == "dtype")
      o.dtype = value;
    else if (key == "m")
      o.m = ParseInteger<int64_t>(value, key);
    else if (key == "n")
      o.n = ParseInteger<int64_t>(value, key);
    else if (key == "k")
      o.k = ParseInteger<int64_t>(value, key);
    else if (key == "threads")
      o.threads = ParseInteger<int>(value, key);
    else if (key == "variant")
      o.variant = value;
    else if (key == "warmup")
      o.warmup = ParseInteger<int>(value, key);
    else if (key == "iterations")
      o.iterations = ParseInteger<int>(value, key);
    else if (key == "cache")
      o.cache = value;
    else if (key == "counters")
      o.counters = value;
    else if (key == "format")
      o.format = value;
    else if (key == "seed")
      o.seed = ParseInteger<uint32_t>(value, key);
    else
      Fail("unknown option --" + key);
  }
  if (o.op != "matmul-bt-quant") Fail("unsupported --op: " + o.op);
  if (o.m <= 0 || o.n <= 0 || o.k <= 0) Fail("M, N and K must be positive");
  if (o.threads <= 0 || o.threads > 512) Fail("--threads must be in [1,512]");
  if (o.warmup < 0 || o.iterations <= 0) Fail("warmup must be >=0 and iterations >0");
  const std::array<std::string_view, 6> variants = {"auto",    "portable", "sdot",
                                                    "a76-asm", "repacked", "mmla"};
  const std::array<std::string_view, 4> caches = {"hot", "l2", "l3", "stream"};
  const std::array<std::string_view, 5> counters = {"auto", "off", "generic", "a76", "all"};
  if (std::find(variants.begin(), variants.end(), o.variant) == variants.end())
    Fail("bad --variant");
  if (std::find(caches.begin(), caches.end(), o.cache) == caches.end()) Fail("bad --cache");
  if (std::find(counters.begin(), counters.end(), o.counters) == counters.end())
    Fail("bad --counters");
  if (o.format != "text" && o.format != "json") Fail("--format must be text or json");
  return o;
}

void SetEnv(const char* name, const std::string& value) {
#if defined(_WIN32)
  if (_putenv_s(name, value.c_str()) != 0) Fail(std::string("cannot set ") + name);
#else
  if (setenv(name, value.c_str(), 1) != 0) Fail(std::string("cannot set ") + name);
#endif
}

std::string CompilerIdentity() {
#if defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

std::string CompileFeatures() {
  std::string s;
#if defined(NDEBUG)
  s += "release";
#else
  s += "debug";
#endif
#if defined(__ARM_FEATURE_DOTPROD)
  s += ",dotprod";
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
  s += ",i8mm";
#endif
#if defined(__AVX512F__)
  s += ",avx512f";
#elif defined(__AVX2__)
  s += ",avx2";
#endif
  return s;
}

std::string CpuIdentity() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    for (const std::string_view key : {"model name", "Hardware"}) {
      if (line.starts_with(key)) {
        const size_t colon = line.find(':');
        if (colon != std::string::npos) return line.substr(colon + 2);
      }
    }
  }
  return "unavailable";
}

std::string Affinity() {
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return "unavailable";
  std::ostringstream out;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    if (!first) out << ',';
    out << cpu;
    first = false;
  }
  return out.str();
#else
  return "unavailable";
#endif
}

void PinToFirstCpus(int count) {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) Fail("sched_getaffinity failed");
  cpu_set_t chosen;
  CPU_ZERO(&chosen);
  int found = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE && found < count; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      CPU_SET(cpu, &chosen);
      ++found;
    }
  }
  if (found != count) Fail("affinity contains fewer CPUs than --threads");
  if (sched_setaffinity(0, sizeof(chosen), &chosen) != 0) Fail("sched_setaffinity failed");
#else
  (void)count;
#endif
}

std::optional<double> ReadNumber(const std::string& path, double scale) {
  const std::string text = ReadText(path);
  if (text.empty()) return std::nullopt;
  try {
    return std::stod(text) * scale;
  } catch (...) {
    return std::nullopt;
  }
}

std::string ThrottlingStatus() {
#if defined(__linux__)
  FILE* pipe = popen("vcgencmd get_throttled 2>/dev/null", "r");
  if (pipe == nullptr) return "unavailable";
  std::array<char, 128> buffer{};
  const char* read = std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe);
  const int status = pclose(pipe);
  if (read == nullptr || status != 0) return "unavailable";
  std::string value(buffer.data());
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
  return value;
#else
  return "unavailable";
#endif
}

struct EventSpec {
  std::string name;
  uint32_t type = 0;
  uint64_t config = 0;
};

struct CounterValue {
  std::string name;
  double value = 0.0;
};

struct CounterPass {
  std::string name;
  std::string status = "unsupported";
  std::string error;
  double running_ratio = 0.0;
  std::vector<CounterValue> values;
};

#if defined(__linux__)
class PerfGroup {
 public:
  PerfGroup(std::string name, const std::vector<EventSpec>& specs) : name_(std::move(name)) {
    if (specs.empty()) error_ = "no matching PMU events";
    for (const EventSpec& spec : specs) Open(spec);
  }
  ~PerfGroup() {
    for (int fd : fds_) close(fd);
  }
  PerfGroup(const PerfGroup&) = delete;
  PerfGroup& operator=(const PerfGroup&) = delete;
  PerfGroup(PerfGroup&& other) noexcept { *this = std::move(other); }
  PerfGroup& operator=(PerfGroup&& other) noexcept {
    if (this == &other) return *this;
    for (int fd : fds_) close(fd);
    name_ = std::move(other.name_);
    error_ = std::move(other.error_);
    fds_ = std::move(other.fds_);
    names_ = std::move(other.names_);
    ids_ = std::move(other.ids_);
    other.fds_.clear();
    return *this;
  }
  bool Ready() const { return !fds_.empty(); }
  const std::string& Name() const { return name_; }
  const std::string& Error() const { return error_; }
  void Start() {
    if (!Ready()) return;
    if (ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
      error_ = std::strerror(errno);
    }
  }
  CounterPass Stop() {
    CounterPass pass;
    pass.name = name_;
    pass.error = error_;
    if (!Ready()) return pass;
    if (ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
      pass.error = std::strerror(errno);
      return pass;
    }
    std::vector<uint64_t> data(3 + 2 * fds_.size());
    const ssize_t wanted = static_cast<ssize_t>(data.size() * sizeof(uint64_t));
    if (read(fds_[0], data.data(), static_cast<size_t>(wanted)) != wanted) {
      pass.error = std::strerror(errno);
      return pass;
    }
    const uint64_t nr = data[0];
    const uint64_t enabled = data[1];
    const uint64_t running = data[2];
    if (nr != fds_.size() || enabled == 0 || running == 0) {
      pass.status = "unscheduled";
      pass.error = "zero PMU running time or incomplete group read";
      return pass;
    }
    pass.running_ratio = static_cast<double>(running) / static_cast<double>(enabled);
    const double scale = static_cast<double>(enabled) / static_cast<double>(running);
    for (size_t i = 0; i < static_cast<size_t>(nr); ++i) {
      const uint64_t raw = data[3 + 2 * i];
      const uint64_t id = data[4 + 2 * i];
      const auto it = std::find(ids_.begin(), ids_.end(), id);
      if (it == ids_.end()) continue;
      const size_t index = static_cast<size_t>(it - ids_.begin());
      pass.values.push_back({names_[index], static_cast<double>(raw) * scale});
    }
    if (!error_.empty())
      pass.status = "partial";
    else
      pass.status = pass.running_ratio < 0.999 ? "multiplexed" : "ok";
    return pass;
  }

 private:
  void Open(const EventSpec& spec) {
    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = spec.type;
    attr.config = spec.config;
    attr.disabled = fds_.empty() ? 1U : 0U;
    attr.inherit = 1U;
    attr.inherit_stat = 1U;
    attr.exclude_kernel = 1U;
    attr.exclude_hv = 1U;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;
    const int leader = fds_.empty() ? -1 : fds_[0];
    const int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, leader, 0));
    if (fd < 0) {
      if (!error_.empty()) error_ += "; ";
      error_ += spec.name + ": " + std::strerror(errno);
      return;
    }
    uint64_t id = 0;
    if (ioctl(fd, PERF_EVENT_IOC_ID, &id) != 0) {
      if (error_.empty()) error_ = spec.name + ": cannot read event id";
      close(fd);
      return;
    }
    fds_.push_back(fd);
    names_.push_back(spec.name);
    ids_.push_back(id);
  }
  std::string name_;
  std::string error_;
  std::vector<int> fds_;
  std::vector<std::string> names_;
  std::vector<uint64_t> ids_;
};
#else
class PerfGroup {
 public:
  PerfGroup(std::string name, const std::vector<EventSpec>&) : name_(std::move(name)) {}
  bool Ready() const { return false; }
  const std::string& Name() const { return name_; }
  const std::string& Error() const { return error_; }
  void Start() {}
  CounterPass Stop() { return CounterPass{name_, "unsupported", error_, 0.0, {}}; }

 private:
  std::string name_;
  std::string error_ = "perf_event_open is Linux-only";
};
#endif

// The Cortex-A76 PMU lookup reads /sys/bus/event_source, so it is called only
// from the __linux__ half of MakeCounterGroups below. It carries the same guard
// as its caller: without it the pair is dead code everywhere else, and
// -Werror,-Wunused-function makes that a hard build failure (macOS/clang).
#if defined(__linux__)
std::optional<uint64_t> ParsePerfEvent(std::string_view descriptor) {
  const size_t at = descriptor.find("event=");
  if (at == std::string_view::npos) return std::nullopt;
  const size_t start = at + 6;
  size_t end = start;
  while (end < descriptor.size() && descriptor[end] != ',' && descriptor[end] != '\n') ++end;
  uint64_t value = 0;
  std::string text(descriptor.substr(start, end - start));
  char* parsed_end = nullptr;
  value = std::strtoull(text.c_str(), &parsed_end, 0);
  if (parsed_end == text.c_str() || *parsed_end != '\0') return std::nullopt;
  return value;
}

std::optional<EventSpec> A76Event(const std::string& name) {
  const std::string root = "/sys/bus/event_source/devices/armv8_cortex_a76/";
  const std::string type_text = ReadText(root + "type");
  const std::string descriptor = ReadText(root + "events/" + name);
  if (type_text.empty() || descriptor.empty()) return std::nullopt;
  const auto config = ParsePerfEvent(descriptor);
  if (!config) return std::nullopt;
  return EventSpec{name, ParseInteger<uint32_t>(type_text, "pmu-type"), *config};
}
#endif

std::vector<PerfGroup> MakeCounterGroups(const std::string& selection) {
  std::vector<PerfGroup> groups;
  if (selection == "off") return groups;
#if defined(__linux__)
  const bool generic = selection == "auto" || selection == "generic" || selection == "all";
  const bool a76 = selection == "auto" || selection == "a76" || selection == "all";
  if (generic) {
    groups.emplace_back(
        "generic-core",
        std::vector<EventSpec>{
            {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
            {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
            {"frontend_stalls", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
            {"backend_stalls", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND}});
    groups.emplace_back(
        "generic-memory-branch",
        std::vector<EventSpec>{{"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
                               {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
                               {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
                               {"branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES}});
  }
  if (a76) {
    const size_t before = groups.size();
    for (const auto& names : std::array<std::array<const char*, 4>, 4>{
             {{{"cpu_cycles", "inst_retired", "stall_frontend", "stall_backend"}},
              {{"cpu_cycles", "inst_retired", "l1d_cache_refill", "l2d_cache_refill"}},
              {{"cpu_cycles", "inst_retired", "ll_cache_rd", "ll_cache_miss_rd"}},
              {{"cpu_cycles", "inst_retired", "dtlb_walk", "br_mis_pred"}}}}) {
      std::vector<EventSpec> specs;
      for (const char* name : names) {
        if (auto event = A76Event(name)) specs.push_back(*event);
      }
      if (specs.size() == names.size()) groups.emplace_back(std::string("a76-") + names[2], specs);
    }
    if (groups.size() == before) groups.emplace_back("a76-pmu", std::vector<EventSpec>{});
  }
#else
  (void)selection;
#endif
  return groups;
}

vt::DType ParseDType(const std::string& name) {
  if (name == "q8_0") return vt::DType::kQ8_0;
  if (name == "q4_k") return vt::DType::kQ4_K;
  if (name == "q6_k") return vt::DType::kQ6_K;
  Fail("unsupported --dtype: " + name);
}

int64_t BlockElements(vt::DType dtype) {
  return dtype == vt::DType::kQ8_0 ? 32 : 256;
}

std::vector<uint8_t> RandomBlocks(vt::DType dtype, int64_t nblocks, uint32_t seed) {
  const int64_t be = BlockElements(dtype);
  const size_t block_bytes = vt::RowSizeBytes(dtype, be);
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks) * block_bytes);
  for (uint8_t& byte : bytes) byte = static_cast<uint8_t>(rng() & 0xffU);
  const int d_offset = dtype == vt::DType::kQ6_K ? 208 : 0;
  const int dmin_offset = dtype == vt::DType::kQ4_K ? 2 : -1;
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* block = bytes.data() + static_cast<size_t>(i) * block_bytes;
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    auto put = [&](int offset, float value) {
      const uint16_t half = vt::F32ToF16(value);
      std::memcpy(block + offset, &half, sizeof(half));
    };
    put(d_offset, 0.0125F * jitter);
    if (dmin_offset >= 0) put(dmin_offset, 0.0075F * jitter);
  }
  return bytes;
}

class QuantMatmulFixture {
 public:
  explicit QuantMatmulFixture(const Options& o)
      : dtype_(ParseDType(o.dtype)), q_({vt::DeviceType::kCPU, 0}, nullptr) {
    const int64_t be = BlockElements(dtype_);
    if (o.k % be != 0) Fail("K must be divisible by the quant block size");
    if (o.m > std::numeric_limits<int64_t>::max() / o.k ||
        o.n > std::numeric_limits<int64_t>::max() / o.k ||
        o.m > std::numeric_limits<int64_t>::max() / o.n)
      Fail("shape overflow");
    weights_ = RandomBlocks(dtype_, o.n * (o.k / be), o.seed);
    std::mt19937 rng(o.seed ^ 0x51f15eU);
    activations_.resize(static_cast<size_t>(o.m * o.k));
    for (float& x : activations_) {
      const int centered = static_cast<int>(rng() % 2001U) - 1000;
      x = 0.1F + 0.001F * static_cast<float>(centered);
    }
    output_.resize(static_cast<size_t>(o.m * o.n));
    if (o.variant == "repacked") {
      if (dtype_ != vt::DType::kQ8_0 || !vt::cpu::QuantRepackActive()) {
        Fail("repacked requires q8_0 and an active repack provider");
      }
      vt::cpu::QuantRepackWeight(dtype_, weights_.data(), o.n, o.k);
      repacked_ = true;
    }
    if (o.variant == "mmla" && !vt::cpu::QuantMmlaActive()) Fail("mmla is unavailable");
    if ((o.variant == "sdot" || o.variant == "a76-asm") && dtype_ != vt::DType::kQ8_0)
      Fail("sdot and a76-asm require q8_0");
    if (o.variant == "sdot" && !vt::cpu::QuantQ8SdotActive()) Fail("sdot is unavailable");
    if (o.variant == "a76-asm" && vt::cpu::QuantQ8A76AsmVecDot() == nullptr)
      Fail("a76-asm is unavailable");
    at_ = vt::Tensor::Contiguous(activations_.data(), vt::DType::kF32, q_.device, {o.m, o.k});
    bt_ = vt::Tensor::Contiguous(weights_.data(), vt::DType::kF32, q_.device, {o.n, o.k});
    bt_.dtype = dtype_;
    bt_.repacked = repacked_;
    ot_ = vt::Tensor::Contiguous(output_.data(), vt::DType::kF32, q_.device, {o.m, o.n});
  }
  void Run() { vt::MatmulBTQuant(q_, ot_, at_, bt_); }
  uint64_t Checksum() const {
    uint64_t hash = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const uint8_t*>(output_.data());
    for (size_t i = 0; i < output_.size() * sizeof(float); ++i) {
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
  }
  uint64_t BytesPerCall() const {
    return static_cast<uint64_t>(weights_.size()) +
           static_cast<uint64_t>((activations_.size() + output_.size()) * sizeof(float));
  }

 private:
  vt::DType dtype_;
  vt::Queue q_;
  std::vector<uint8_t> weights_;
  std::vector<float> activations_;
  std::vector<float> output_;
  vt::Tensor at_;
  vt::Tensor bt_;
  vt::Tensor ot_;
  bool repacked_ = false;
};

size_t CachePressureBytes(const std::string& mode) {
  if (mode == "hot") return 0;
  if (mode == "l2") return 128U * 1024U;
  if (mode == "l3") return 1024U * 1024U;
  return 16U * 1024U * 1024U;
}

void Touch(std::vector<uint8_t>& buffer) {
  static volatile uint64_t sink = 0;
  uint64_t sum = 0;
  for (size_t i = 0; i < buffer.size(); i += 64) {
    buffer[i] = static_cast<uint8_t>(buffer[i] + 1U);
    sum += buffer[i];
  }
  sink = sink ^ sum;
}

double TimerOverheadNs() {
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 1000; ++i) {
    const auto begin = Clock::now();
    const auto end = Clock::now();
    best = std::min(best, std::chrono::duration<double, std::nano>(end - begin).count());
  }
  return best;
}

struct Result {
  size_t calls_per_sample = 1;
  std::vector<double> sample_ns;
  double median_ns = 0.0;
  double min_ns = 0.0;
  double mean_ns = 0.0;
  double timer_overhead_ratio = 0.0;
  uint64_t checksum = 0;
  uint64_t bytes_per_call = 0;
  std::vector<CounterPass> counters;
  int cpu_before = -1;
  int cpu_after = -1;
  std::optional<double> freq_before_mhz;
  std::optional<double> freq_after_mhz;
  std::optional<double> temp_before_c;
  std::optional<double> temp_after_c;
  std::string governor = "unavailable";
  std::string throttle_before = "unavailable";
  std::string throttle_after = "unavailable";
};

int CurrentCpu() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

Result Benchmark(const Options& o, QuantMatmulFixture& fixture, std::vector<PerfGroup>& groups) {
  Result result;
  result.bytes_per_call = fixture.BytesPerCall();
  std::vector<uint8_t> pressure(CachePressureBytes(o.cache));
  for (int i = 0; i < o.warmup; ++i) fixture.Run();
  const double overhead = TimerOverheadNs();
  if (o.cache == "hot") {
    for (;;) {
      const auto begin = Clock::now();
      for (size_t i = 0; i < result.calls_per_sample; ++i) fixture.Run();
      const auto end = Clock::now();
      const double ns = std::chrono::duration<double, std::nano>(end - begin).count();
      if (ns >= std::max(2.0e6, 1000.0 * overhead) || result.calls_per_sample >= (1U << 20)) break;
      result.calls_per_sample *= 2;
    }
  }
  result.cpu_before = CurrentCpu();
  if (result.cpu_before >= 0) {
    const std::string root =
        "/sys/devices/system/cpu/cpu" + std::to_string(result.cpu_before) + "/cpufreq/";
    result.freq_before_mhz = ReadNumber(root + "scaling_cur_freq", 0.001);
    const std::string governor = ReadText(root + "scaling_governor");
    if (!governor.empty()) result.governor = governor;
  }
  result.temp_before_c = ReadNumber("/sys/class/thermal/thermal_zone0/temp", 0.001);
  result.throttle_before = ThrottlingStatus();
  for (int sample = 0; sample < o.iterations; ++sample) {
    Touch(pressure);
    const auto begin = Clock::now();
    for (size_t call = 0; call < result.calls_per_sample; ++call) fixture.Run();
    const auto end = Clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - begin).count() /
                      static_cast<double>(result.calls_per_sample);
    result.sample_ns.push_back(ns);
  }
  result.cpu_after = CurrentCpu();
  if (result.cpu_after >= 0) {
    const std::string root =
        "/sys/devices/system/cpu/cpu" + std::to_string(result.cpu_after) + "/cpufreq/";
    result.freq_after_mhz = ReadNumber(root + "scaling_cur_freq", 0.001);
  }
  result.temp_after_c = ReadNumber("/sys/class/thermal/thermal_zone0/temp", 0.001);
  result.throttle_after = ThrottlingStatus();
  std::vector<double> sorted = result.sample_ns;
  std::sort(sorted.begin(), sorted.end());
  result.min_ns = sorted.front();
  result.median_ns = sorted[sorted.size() / 2];
  double total = 0.0;
  for (double ns : result.sample_ns) total += ns;
  result.mean_ns = total / static_cast<double>(result.sample_ns.size());
  result.timer_overhead_ratio =
      overhead / (result.median_ns * static_cast<double>(result.calls_per_sample));
  result.checksum = fixture.Checksum();
  for (PerfGroup& group : groups) {
    Touch(pressure);
    group.Start();
    for (size_t call = 0; call < result.calls_per_sample; ++call) fixture.Run();
    CounterPass pass = group.Stop();
    if (fixture.Checksum() != result.checksum) Fail("output checksum changed during PMU pass");
    for (CounterValue& value : pass.values)
      value.value /= static_cast<double>(result.calls_per_sample);
    result.counters.push_back(std::move(pass));
  }
  return result;
}

void JsonString(std::ostream& out, std::string_view value) {
  out << '"';
  for (const char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c == '\n')
      out << "\\n";
    else
      out << c;
  }
  out << '"';
}

void JsonOptional(std::ostream& out, const std::optional<double>& value) {
  if (value)
    out << *value;
  else
    out << "null";
}

std::optional<double> Counter(const CounterPass& pass, std::string_view name) {
  for (const CounterValue& value : pass.values) {
    if (value.name == name) return value.value;
  }
  return std::nullopt;
}

void PrintJson(const Options& o, const Result& r) {
  const double seconds = r.median_ns * 1e-9;
  const double gflops = 2.0 * static_cast<double>(o.m) * static_cast<double>(o.n) *
                        static_cast<double>(o.k) / seconds / 1e9;
  const double bandwidth = static_cast<double>(r.bytes_per_call) / seconds / 1e9;
  const bool timer_valid = r.timer_overhead_ratio < 0.001;
  const bool migration_valid = r.cpu_before < 0 || r.cpu_after < 0 || r.cpu_before == r.cpu_after;
  std::cout << std::setprecision(10) << '{';
  std::cout << "\"schema\":\"vllm-cpu-kernel-bench/v1\",\"scope\":\"vt-op\",\"op\":";
  JsonString(std::cout, o.op);
  std::cout << ",\"dtype\":";
  JsonString(std::cout, o.dtype);
  std::cout << ",\"shape\":{\"m\":" << o.m << ",\"n\":" << o.n << ",\"k\":" << o.k << '}';
  std::cout << ",\"threads\":" << o.threads << ",\"variant\":";
  JsonString(std::cout, o.variant);
  std::cout << ",\"cache\":";
  JsonString(std::cout, o.cache);
  std::cout << ",\"timing\":{\"samples\":" << o.iterations
            << ",\"calls_per_sample\":" << r.calls_per_sample << ",\"min_ns\":" << r.min_ns
            << ",\"median_ns\":" << r.median_ns << ",\"mean_ns\":" << r.mean_ns
            << ",\"timer_overhead_ratio\":" << r.timer_overhead_ratio
            << ",\"valid\":" << (timer_valid ? "true" : "false") << '}';
  std::cout << ",\"work\":{\"bytes_per_call\":" << r.bytes_per_call
            << ",\"effective_gb_s\":" << bandwidth << ",\"gflop_s\":" << gflops << '}';
  std::ostringstream checksum;
  checksum << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.checksum;
  std::cout << ",\"checksum\":";
  JsonString(std::cout, checksum.str());
  std::cout << ",\"metadata\":{\"compiler\":";
  JsonString(std::cout, CompilerIdentity());
  std::cout << ",\"compile_features\":";
  JsonString(std::cout, CompileFeatures());
  std::cout << ",\"mmla_active\":" << (vt::cpu::QuantMmlaActive() ? "true" : "false")
            << ",\"repack_active\":" << (vt::cpu::QuantRepackActive() ? "true" : "false");
  std::cout << ",\"cpu\":";
  JsonString(std::cout, CpuIdentity());
  std::cout << ",\"affinity\":";
  JsonString(std::cout, Affinity());
  std::cout << ",\"cpu_before\":" << r.cpu_before << ",\"cpu_after\":" << r.cpu_after;
  std::cout << ",\"frequency_before_mhz\":";
  JsonOptional(std::cout, r.freq_before_mhz);
  std::cout << ",\"frequency_after_mhz\":";
  JsonOptional(std::cout, r.freq_after_mhz);
  std::cout << ",\"temperature_before_c\":";
  JsonOptional(std::cout, r.temp_before_c);
  std::cout << ",\"temperature_after_c\":";
  JsonOptional(std::cout, r.temp_after_c);
  std::cout << ",\"governor\":";
  JsonString(std::cout, r.governor);
  std::cout << ",\"throttle_before\":";
  JsonString(std::cout, r.throttle_before);
  std::cout << ",\"throttle_after\":";
  JsonString(std::cout, r.throttle_after);
  std::cout << ",\"migration_valid\":" << (migration_valid ? "true" : "false") << '}';
  std::cout << ",\"counter_passes\":[";
  for (size_t i = 0; i < r.counters.size(); ++i) {
    const CounterPass& pass = r.counters[i];
    if (i != 0) std::cout << ',';
    std::cout << "{\"name\":";
    JsonString(std::cout, pass.name);
    std::cout << ",\"status\":";
    JsonString(std::cout, pass.status);
    std::cout << ",\"error\":";
    JsonString(std::cout, pass.error);
    std::cout << ",\"running_ratio\":" << pass.running_ratio << ",\"ipc\":";
    const auto cycles = Counter(pass, pass.name.starts_with("a76-") ? "cpu_cycles" : "cycles");
    const auto instructions =
        Counter(pass, pass.name.starts_with("a76-") ? "inst_retired" : "instructions");
    if (cycles && instructions && *cycles > 0.0)
      std::cout << *instructions / *cycles;
    else
      std::cout << "null";
    std::cout << ",\"events\":{";
    for (size_t j = 0; j < pass.values.size(); ++j) {
      if (j != 0) std::cout << ',';
      JsonString(std::cout, pass.values[j].name);
      std::cout << ':' << pass.values[j].value;
    }
    std::cout << "}}";
  }
  std::cout << "]}\n";
}

void PrintText(const Options& o, const Result& r) {
  const double seconds = r.median_ns * 1e-9;
  const double gflops = 2.0 * static_cast<double>(o.m) * static_cast<double>(o.n) *
                        static_cast<double>(o.k) / seconds / 1e9;
  std::cout << "vllm-cpu-kernel-bench/v1 " << o.op << ' ' << o.dtype << " M=" << o.m << " N=" << o.n
            << " K=" << o.k << " threads=" << o.threads << " variant=" << o.variant
            << " cache=" << o.cache << '\n';
  std::cout << std::fixed << std::setprecision(3) << "median " << r.median_ns / 1e6 << " ms, min "
            << r.min_ns / 1e6 << " ms, " << gflops << " GFLOP/s, timer ratio "
            << 100.0 * r.timer_overhead_ratio << "%\n";
  std::cout << "affinity " << Affinity() << ", CPU " << r.cpu_before << " -> " << r.cpu_after
            << ", checksum 0x" << std::hex << r.checksum << std::dec << '\n';
  for (const CounterPass& pass : r.counters) {
    std::cout << pass.name << ": " << pass.status << " running=" << 100.0 * pass.running_ratio
              << '%';
    if (!pass.error.empty()) std::cout << " (" << pass.error << ')';
    for (const CounterValue& value : pass.values) {
      std::cout << "  " << value.name << '=' << std::setprecision(0) << value.value;
    }
    const auto cycles = Counter(pass, pass.name.starts_with("a76-") ? "cpu_cycles" : "cycles");
    const auto instructions =
        Counter(pass, pass.name.starts_with("a76-") ? "inst_retired" : "instructions");
    if (cycles && instructions && *cycles > 0.0) {
      std::cout << "  IPC=" << std::setprecision(3) << *instructions / *cycles;
    }
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = ParseArgs(argc, argv);
    SetEnv("VLLM_CPP_CPU_THREADS", std::to_string(options.threads));
    if (options.variant == "portable" || options.variant == "sdot" ||
        options.variant == "a76-asm") {
      SetEnv("VT_CPU_QUANT_MMLA", "0");
      SetEnv("VT_CPU_QUANT_REPACK", "0");
    }
    if (options.variant == "portable" || options.variant == "sdot" ||
        options.variant == "a76-asm") {
      SetEnv("VT_CPU_Q8_DOT", options.variant);
    }
    if (options.pin) PinToFirstCpus(options.threads);
    QuantMatmulFixture fixture(options);
    // Open before the warmup creates persistent workers. inherit=1 then makes
    // each pass process-wide rather than silently counting only worker zero.
    std::vector<PerfGroup> groups = MakeCounterGroups(options.counters);
    Result result = Benchmark(options, fixture, groups);
    if (options.format == "json")
      PrintJson(options, result);
    else
      PrintText(options, result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vllm-cpu-kernel-bench: " << error.what() << '\n';
    return 2;
  }
}
