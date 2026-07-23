// Op-level GFLOP/s microbench for kMatmulBTQuant — QUANT-GGUF-CIQ-GEMM G6
// evidence tool (NOT a ctest gate). Times the quant GEMM at this model's
// prefill/decode shapes so the portable nrc==1 tier and the Arm i8mm mmla tier
// can be compared apples-to-apples, decoupled from the mixed-file Amdahl
// dilution an end-to-end run mixes in.
//
//   VT_CPU_QUANT_MMLA=0 env  -> portable tier (nrc==1)
//   (unset, i8mm host)       -> i8mm mmla tier (nrc==2) at even M,N
//
// GFLOP/s = 2*M*N*K / best-of-R seconds. Run the two arms as two processes
// (the tier is picked once per process from the env).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

struct Shape {
  int64_t m, n, k;
  const char* name;
};

std::vector<uint8_t> RandomBlocks(vt::DType dt, int64_t nblocks, int d_off, int dmin_off,
                                  uint32_t seed) {
  const int64_t be = (dt == vt::DType::kQ4_0 || dt == vt::DType::kQ8_0) ? 32 : 256;
  const size_t block_bytes = vt::RowSizeBytes(dt, be);
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks) * block_bytes);
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + static_cast<size_t>(i) * block_bytes;
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    auto put = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    if (d_off >= 0) put(d_off, 0.0125F * jitter);
    if (dmin_off >= 0) put(dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

void Bench(vt::DType dt, int d_off, int dmin_off, const char* type_name) {
  const int64_t be = (dt == vt::DType::kQ4_0 || dt == vt::DType::kQ8_0) ? 32 : 256;
  const Shape shapes[] = {
      {128, 3072, 2048, "prefill qkv"},
      {128, 12288, 2048, "prefill gate_up"},
      {128, 2048, 6144, "prefill down"},
      {1, 3072, 2048, "decode qkv"},
  };
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  for (const Shape& s : shapes) {
    if (s.k % be != 0) continue;
    const int64_t nblk = s.n * (s.k / be);
    std::vector<uint8_t> wq = RandomBlocks(dt, nblk, d_off, dmin_off, 0x6A11U);
    std::vector<float> a(static_cast<size_t>(s.m * s.k));
    std::mt19937 rng(1);
    for (float& x : a) x = 0.1F + 0.001F * static_cast<float>(rng() % 2000 - 1000);
    std::vector<float> out(static_cast<size_t>(s.m * s.n));

    vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {s.m, s.k});
    vt::Tensor bt = vt::Tensor::Contiguous(wq.data(), vt::DType::kF32, q.device, {s.n, s.k});
    bt.dtype = dt;
    vt::Tensor ot = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {s.m, s.n});

    auto time_best = [&](vt::Tensor& b) {
      vt::MatmulBTQuant(q, ot, at, b);  // warm
      double best = 1e30;
      const int reps = s.m == 1 ? 20 : 6;
      for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        vt::MatmulBTQuant(q, ot, at, b);
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        if (sec < best) best = sec;
      }
      return 2.0 * static_cast<double>(s.m * s.n * s.k) / best / 1e9;
    };

    const double gflops = time_best(bt);
    std::printf("%-9s %-16s M=%-4lld N=%-6lld K=%-5lld  %8.2f GFLOP/s\n", type_name,
                s.name, (long long)s.m, (long long)s.n, (long long)s.k, gflops);

    // CIQ G7: repacked-weight arm for q8_0 (i8mm interleave). Repack a copy of
    // the weight, mark it, and time the repacked gemm/gemv against the same
    // shapes — the tier-0/mmla vs repacked op-level A/B.
    if (dt == vt::DType::kQ8_0 && vt::cpu::QuantRepackActive()) {
      std::vector<uint8_t> wrp = wq;
      vt::cpu::QuantRepackWeight(dt, wrp.data(), s.n, s.k);
      vt::Tensor brp =
          vt::Tensor::Contiguous(wrp.data(), vt::DType::kF32, q.device, {s.n, s.k});
      brp.dtype = dt;
      brp.repacked = true;
      const double g2 = time_best(brp);
      std::printf("%-9s %-16s M=%-4lld N=%-6lld K=%-5lld  %8.2f GFLOP/s\n",
                  "q8_0-rp", s.name, (long long)s.m, (long long)s.n,
                  (long long)s.k, g2);
    }
  }
}

}  // namespace

int main() {
  std::printf("== kMatmulBTQuant op-level bench — mmla tier %s ==\n",
              vt::cpu::QuantMmlaActive() ? "ON (i8mm)" : "OFF (portable)");
  Bench(vt::DType::kQ8_0, 0, -1, "q8_0");
  Bench(vt::DType::kQ4_K, 0, 2, "q4_K");
  Bench(vt::DType::kQ6_K, 208, -1, "q6_K");
  return 0;
}
