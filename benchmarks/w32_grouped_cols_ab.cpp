// Grouped 32-block GEMV output-width A/B: N output COLUMNS per warp against the
// shipped one-warp-per-output kernel.
// MODEL-MM-QWEN4-EXP, ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q.
//
// WHY AN ISOLATED SWEEP EXISTS AT ALL. The decode bandwidth attribution ranked
// the qwen4_exp kernels ascending and the floor is this one: 41.6 GB/s, 15.2% of
// peak, against cuBLAS `gemvx` at 162.3 GB/s in the same capture. The issue owes
// an N sweep over 1/2/4/8 BEFORE a width is chosen, because register pressure
// turns the curve over somewhere and the turning point is the answer. Deciding
// that from an e2e decode run means paying a model load per arm and reading the
// width off a number that three other kernels also move.
//
// SAME BINARY, ONE VARIABLE. Every arm runs this executable; the arms differ only
// in VT_V4_W32_COLS and VT_V4_W32_WARPS, which LaunchGrouped32 reads per call.
// Comparing two BUILDS is what .agents/benchmark-protocol.md rules out.
//
// IT PROVES BIT IDENTITY IN THE SAME RUN, with an FNV-1a hash over the raw bits
// of the whole output. Each output element keeps its own accumulator and its own
// ascending block order at every width, so every arm must print the SAME hash. A
// spot-checked relative error cannot tell "bit-identical" from "close"; a hash of
// every output bit can, and a divergent arm's timing is then void rather than
// interesting.
//
// SHAPE: the released Qwen3.8-Flash-Next routed gate/up at decode. hidden 2560
// (nb = 80 blocks of 32), moe_intermediate 640, num_experts 512,
// num_experts_per_tok 10, one token -> P = 10 and a broadcast activation. That is
// the 1,600-block grid the attribution measured.
//
// NOT WIRED INTO CI. It is an executable under examples/CMakeLists.txt guarded on
// VLLM_CPP_CUDA, like vulkan-gemv-ab, and nothing runs it automatically.
//
// TIMING INCLUDES THE LAUNCH. Each iteration is one enqueue plus a queue
// synchronize on the host clock, not a CUDA event pair around the kernel, so the
// figure carries ~5-10 us of launch and sync overhead against a ~200 us kernel.
// That is about 3% and it is charged EQUALLY to every arm, so the ratios are
// clean; the absolute GB/s is therefore a slight UNDER-estimate of what the
// kernel achieved, in the conservative direction.
//
// RUN IT INSIDE A LEASE. `thor:gpu0` and `dgx:gpu0` are fleet devices; take them
// with `rc run`/`rc hold` and never by ssh. Absolute timings taken under
// contention are upper bounds.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

double MedianMs(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

// FNV-1a over the RAW BITS of the output. The width axis is supposed to be
// BIT-identical, so a hash is the cheapest evidence that the claim is a fact
// rather than an argument about the source. NaN-safe by construction: it hashes
// bit patterns, so an all-NaN arm gets a different hash instead of comparing
// equal to everything.
uint64_t BitHash(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  for (float f : v) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    for (int i = 0; i < 4; ++i) {
      h ^= static_cast<uint64_t>((b >> (8 * i)) & 0xFFu);
      h *= 1099511628211ull;
    }
  }
  return h;
}

// A weight tower of VALID blocks: a finite f16 scale plus a pseudorandom payload.
// Random BYTES would put a NaN or an inf in the scale field, and every arm would
// then agree on a hash of NaNs -- a green that measured nothing.
std::vector<uint8_t> MakeTower(DType dt, int64_t rows, int64_t nb, uint32_t seed) {
  const size_t bb = static_cast<size_t>(vt::BlockBytes(dt));
  std::vector<uint8_t> w(static_cast<size_t>(rows) * static_cast<size_t>(nb) * bb);
  // Three small powers of two as f16 bits: 0.25, 0.5, 1.0. Exactly representable,
  // so the scale contributes no rounding of its own.
  const uint16_t scales[3] = {0x3400u, 0x3800u, 0x3C00u};
  uint32_t s = seed | 1u;
  for (size_t blk = 0; blk * bb < w.size(); ++blk) {
    uint8_t* p = w.data() + blk * bb;
    const uint16_t d = scales[blk % 3];
    std::memcpy(p, &d, 2);
    for (size_t i = 2; i < bb; ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;  // xorshift32
      p[i] = static_cast<uint8_t>(s & 0xFFu);
    }
  }
  return w;
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct Arm {
  const char* cols;
  const char* warps;
};

}  // namespace

int main(int argc, char** argv) {
  // The released decode shape, overridable so the sweep can be repeated at
  // another geometry without a rebuild.
  int64_t H = 2560, I = 640, E = 512, P = 10;
  int iters = 200, warmup = 20;
  const char* dt_name = "iq4_nl";
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--hidden=", 9)) H = std::atoll(argv[i] + 9);
    else if (!std::strncmp(argv[i], "--inter=", 8)) I = std::atoll(argv[i] + 8);
    else if (!std::strncmp(argv[i], "--experts=", 10)) E = std::atoll(argv[i] + 10);
    else if (!std::strncmp(argv[i], "--rows=", 7)) P = std::atoll(argv[i] + 7);
    else if (!std::strncmp(argv[i], "--iters=", 8)) iters = std::atoi(argv[i] + 8);
    else if (!std::strncmp(argv[i], "--dtype=", 8)) dt_name = argv[i] + 8;
  }
  DType dt = DType::kIQ4_NL;
  if (!std::strcmp(dt_name, "q5_0")) dt = DType::kQ5_0;
  else if (!std::strcmp(dt_name, "q4_0")) dt = DType::kQ4_0;
  else if (std::strcmp(dt_name, "iq4_nl")) { std::fprintf(stderr, "bad --dtype\n"); return 2; }

  const int64_t nb = H / 32;
  if (nb * 32 != H) { std::fprintf(stderr, "hidden must be a multiple of 32\n"); return 2; }
  const size_t bb = static_cast<size_t>(vt::BlockBytes(dt));

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue q = gpu.CreateQueue();

  std::vector<uint8_t> w = MakeTower(dt, E * I, nb, 0xA5A5C015u);
  // ONE activation row: a single decoded token routed to P experts, which is the
  // `bcast` polarity the released decode step takes.
  std::vector<float> a(static_cast<size_t>(H));
  {
    uint32_t s = 0x1234567u;
    for (size_t i = 0; i < a.size(); ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      a[i] = static_cast<float>(static_cast<int32_t>(s % 2001) - 1000) / 1000.0f;
    }
  }
  std::vector<int32_t> ids(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p)
    ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 47 + 3) % E);

  const size_t outn = static_cast<size_t>(P * I);
  void* d_w = gpu.Alloc(w.size());
  void* d_a = gpu.Alloc(a.size() * sizeof(float));
  void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
  void* d_o = gpu.Alloc(outn * sizeof(float));
  gpu.Copy(q, d_w, w.data(), w.size());
  gpu.Copy(q, d_a, a.data(), a.size() * sizeof(float));
  gpu.Copy(q, d_e, ids.data(), ids.size() * sizeof(int32_t));
  gpu.Synchronize(q);

  Tensor wt = DevTensor(d_w, dt, {E * I, H});
  Tensor at = DevTensor(d_a, DType::kF32, {1, H});
  Tensor et = DevTensor(d_e, DType::kI32, {P});
  Tensor ot = DevTensor(d_o, DType::kF32, {P, I});

  // The bytes one launch MUST move: the P routed weight rows, the broadcast
  // activation once, and the output once. Counted, not assumed -- an arm that
  // re-reads the activation per output element moves more than this and its
  // achieved figure is therefore a LOWER bound on what the hardware did.
  const double wbytes = static_cast<double>(P) * static_cast<double>(I) *
                        static_cast<double>(nb) * static_cast<double>(bb);
  const double abytes = static_cast<double>(nb) * 34.0;  // Q8_0: f16 scale + 32 int8
  const double obytes = static_cast<double>(outn) * 4.0;
  const double bytes = wbytes + abytes + obytes;

  std::printf("shape hidden=%lld inter=%lld experts=%lld rows=%lld nb=%lld dtype=%s\n",
              (long long)H, (long long)I, (long long)E, (long long)P, (long long)nb, dt_name);
  std::printf("tower=%.1f MiB  per-launch bytes=%.3f MiB (weights %.3f + act %.0f B + out %.0f B)\n",
              (double)w.size() / 1048576.0, bytes / 1048576.0, wbytes / 1048576.0, abytes, obytes);
  std::printf("%-6s %-6s %-7s %10s %10s %10s %20s\n", "cols", "warps", "blocks", "median_us",
              "GB/s", "vs_cols1", "bithash");

  const Arm arms[] = {
      {"1", "4"}, {"2", "4"}, {"4", "4"}, {"8", "4"},
      {"1", "1"}, {"1", "2"}, {"1", "8"},
      {"4", "1"}, {"4", "2"}, {"4", "8"},
  };
  double base_gbs = 0.0;
  uint64_t base_hash = 0;
  int rc = 0;
  for (const Arm& arm : arms) {
    setenv("VT_V4_W32_COLS", arm.cols, 1);
    setenv("VT_V4_W32_WARPS", arm.warps, 1);
    const int cols = std::atoi(arm.cols), warps = std::atoi(arm.warps);
    const long long njg = (I + cols - 1) / cols;
    const long long blocks = (P * njg + warps - 1) / warps;

    for (int i = 0; i < warmup; ++i) vt::MatmulBTQuantGrouped(q, ot, at, wt, et);
    gpu.Synchronize(q);

    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      vt::MatmulBTQuantGrouped(q, ot, at, wt, et);
      gpu.Synchronize(q);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double med = MedianMs(ms);
    const double gbs = bytes / (med * 1.0e-3) / 1.0e9;

    std::vector<float> out(outn, 0.0f);
    gpu.Copy(q, out.data(), d_o, out.size() * sizeof(float));
    gpu.Synchronize(q);
    const uint64_t h = BitHash(out);
    if (base_gbs == 0.0) { base_gbs = gbs; base_hash = h; }
    // A width whose hash moved is not this change. Say so on the line, so the
    // table cannot be read for speed without reading the verdict beside it.
    const bool same = (h == base_hash);
    if (!same) rc = 1;
    std::printf("%-6s %-6s %-7lld %10.1f %10.1f %10.3f %#20llx%s\n", arm.cols, arm.warps, blocks,
                med * 1000.0, gbs, gbs / base_gbs, (unsigned long long)h,
                same ? "" : "   <<< BIT MISMATCH");
  }
  unsetenv("VT_V4_W32_COLS");
  unsetenv("VT_V4_W32_WARPS");

  gpu.Free(d_w); gpu.Free(d_a); gpu.Free(d_e); gpu.Free(d_o);
  gpu.DestroyQueue(q);
  if (rc) std::printf("FAIL: at least one arm changed the output bits.\n");
  else std::printf("OK: every arm produced byte-identical output.\n");
  return rc;
}
