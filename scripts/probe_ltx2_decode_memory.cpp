// Where does the memory go? — the LTX-2.5 Conv VAE decode, instrumented.
//
// Loads the REAL shipped conv VAE and calls Ltx2ConvVideoDecode at the exact
// 448x256/25f latent, while a sampler thread reads /proc/self/status VmRSS/VmHWM
// and /proc/meminfo MemAvailable every 50 ms. Touches NO product code, so it
// cannot perturb what it measures.
//
// ─── BUILD AND RUN (there is no CMake target; this is the recorded recipe) ───
// It is deliberately NOT a CMake target: the global `operator new` replacement
// below would be linked into anything sharing the target's objects, and adding it
// to the default build would charge every configure for a probe. So the recipe is
// written down instead of implied — a reviewer cannot re-run a probe whose compile
// line was never recorded, which is exactly the finding this comment closes.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   ninja -C build vllm
//   g++ -O2 -std=c++20 -Iinclude -Ithird_party \
//       scripts/probe_ltx2_decode_memory.cpp build/libvllm.a -o /tmp/ltx2_mem -pthread
//   /tmp/ltx2_mem <vae.safetensors> <lat_t> <lat_h> <lat_w>
//
// e.g. 448x256/25f on the shipped conv VAE is `... 4 8 14`. The sibling
// scripts/probe_ltx2_tiled_equivalence.cpp builds with the identical line.
#include <malloc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

// ── EXACT heap accounting ───────────────────────────────────────────────────
// Replacing the global operator new/delete in the final link makes every
// std::vector inside libvllm.a route through this counter, so the peak is the
// decoder's own allocations rather than an allocator-retention artefact of RSS.
// Both numbers are reported; they answer different questions.
namespace probe {
std::atomic<long long> live_bytes{0};
std::atomic<long long> peak_bytes{0};
std::atomic<bool> arm{false};
std::atomic<long long> biggest_single{0};
inline void Note(std::size_t n) {
  if (!arm.load(std::memory_order_relaxed)) return;
  const long long now = live_bytes.fetch_add(static_cast<long long>(n)) + static_cast<long long>(n);
  long long seen = peak_bytes.load(std::memory_order_relaxed);
  while (now > seen && !peak_bytes.compare_exchange_weak(seen, now)) {
  }
  long long big = biggest_single.load(std::memory_order_relaxed);
  while (static_cast<long long>(n) > big &&
         !biggest_single.compare_exchange_weak(big, static_cast<long long>(n))) {
  }
}
}  // namespace probe

void* operator new(std::size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  probe::Note(malloc_usable_size(p));
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept {
  if (p == nullptr) return;
  if (probe::arm.load(std::memory_order_relaxed)) {
    probe::live_bytes.fetch_sub(static_cast<long long>(malloc_usable_size(p)));
  }
  std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {

long ReadKb(const char* path, const char* key) {
  std::ifstream in(path);
  std::string line;
  const std::string want(key);
  while (std::getline(in, line)) {
    if (line.rfind(want, 0) == 0) {
      const size_t colon = line.find(':');
      return std::strtol(line.c_str() + colon + 1, nullptr, 10);
    }
  }
  return -1;
}

struct Sampler {
  std::atomic<bool> stop{false};
  std::atomic<long> peak_rss_kb{0};
  std::atomic<long> min_avail_kb{1L << 62};
  std::thread th;

  void Start() {
    th = std::thread([this] {
      while (!stop.load()) {
        const long rss = ReadKb("/proc/self/status", "VmRSS");
        const long avail = ReadKb("/proc/meminfo", "MemAvailable");
        if (rss > peak_rss_kb.load()) peak_rss_kb.store(rss);
        if (avail >= 0 && avail < min_avail_kb.load()) min_avail_kb.store(avail);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  void Stop() {
    stop.store(true);
    th.join();
  }
};

void Mark(const char* what) {
  std::fprintf(stderr, "[probe] %-28s VmRSS=%7.2f MiB VmHWM=%7.2f MiB MemAvailable=%8.2f MiB\n",
               what, ReadKb("/proc/self/status", "VmRSS") / 1024.0,
               ReadKb("/proc/self/status", "VmHWM") / 1024.0,
               ReadKb("/proc/meminfo", "MemAvailable") / 1024.0);
}

// The decoder never uses noise on the shipped config (timestep_conditioning is
// false), but the seam requires a stream when it is on. Deterministic.
class ZeroNoise : public vllm::Ltx2NoiseStream {
 public:
  std::vector<float> Draw(int64_t count) override {
    return std::vector<float>(static_cast<size_t>(count), 0.0f);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: decode_mem_probe <vae.safetensors> <lat_t> <lat_h> <lat_w>\n");
    return 2;
  }
  const std::string path = argv[1];
  const int64_t lt = std::strtoll(argv[2], nullptr, 10);
  const int64_t lh = std::strtoll(argv[3], nullptr, 10);
  const int64_t lw = std::strtoll(argv[4], nullptr, 10);

  Mark("start");
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  vllm::Ltx2VideoDecoderKind kind = vllm::Ltx2VideoDecoderKind::kConv;
  const vllm::Ltx2ConvVideoDecoderConfig cfg =
      vllm::Ltx2ParseConvVideoDecoderConfig(vllm::Ltx2ReadCheckpointConfig(file), &kind);
  Mark("config parsed");
  const vllm::Ltx2VaeWeights weights =
      vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeDecoderKeyRules());
  Mark("weights loaded (f32)");

  const int64_t lc = cfg.in_channels;
  std::vector<float> latent(static_cast<size_t>(lc * lt * lh * lw));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < latent.size(); ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    latent[i] = static_cast<float>(static_cast<int32_t>(s >> 33)) / 2147483648.0f - 0.5f;
  }
  Mark("latent built");

  const long rss_before = ReadKb("/proc/self/status", "VmRSS");
  const long avail_before = ReadKb("/proc/meminfo", "MemAvailable");
  Sampler sampler;
  sampler.Start();
  probe::arm.store(true);
  const auto t0 = std::chrono::steady_clock::now();
  ZeroNoise noise;
  const vllm::Ltx2VideoFrames out =
      vllm::Ltx2ConvVideoDecode(cfg, weights, latent, lc, lt, lh, lw, &noise);
  const auto t1 = std::chrono::steady_clock::now();
  probe::arm.store(false);
  sampler.Stop();
  Mark("decode done");

  std::fprintf(stderr,
               "\n[probe] RESULT latent=[%lld,%lld,%lld,%lld] -> frames=[%lld,%lld,%lld,%lld]\n",
               (long long)lc, (long long)lt, (long long)lh, (long long)lw,
               (long long)out.channels, (long long)out.frames, (long long)out.height,
               (long long)out.width);
  std::fprintf(stderr, "[probe] decode wall            = %.2f s\n",
               std::chrono::duration<double>(t1 - t0).count());
  std::fprintf(stderr, "[probe] VmRSS before decode    = %.2f MiB\n", rss_before / 1024.0);
  std::fprintf(stderr, "[probe] VmRSS peak (sampled)   = %.2f MiB\n",
               sampler.peak_rss_kb.load() / 1024.0);
  std::fprintf(stderr, "[probe] decode-attributed RSS  = %.2f MiB\n",
               (sampler.peak_rss_kb.load() - rss_before) / 1024.0);
  std::fprintf(stderr, "[probe] VmHWM (process peak)   = %.2f MiB\n",
               ReadKb("/proc/self/status", "VmHWM") / 1024.0);
  std::fprintf(stderr, "[probe] MemAvailable before    = %.2f MiB\n", avail_before / 1024.0);
  std::fprintf(stderr, "[probe] MemAvailable min       = %.2f MiB  (delta %.2f MiB)\n",
               sampler.min_avail_kb.load() / 1024.0,
               (avail_before - sampler.min_avail_kb.load()) / 1024.0);
  std::fprintf(stderr, "[probe] HEAP peak live (exact)  = %.2f MiB\n",
               probe::peak_bytes.load() / 1048576.0);
  std::fprintf(stderr, "[probe] HEAP largest single all = %.2f MiB\n",
               probe::biggest_single.load() / 1048576.0);
  std::fprintf(stderr, "[probe] HEAP still live at exit = %.2f MiB\n",
               probe::live_bytes.load() / 1048576.0);
  double checksum = 0.0;
  for (float v : out.data) checksum += v;
  std::fprintf(stderr, "[probe] output checksum        = %.6f (keeps the buffer live)\n", checksum);
  return 0;
}
