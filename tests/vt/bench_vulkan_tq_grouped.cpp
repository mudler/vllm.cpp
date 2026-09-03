// bench_vulkan_tq_grouped: microbenchmark for TQ1_0/TQ2_0 grouped dev and MoE
// fused gate+up+SwiGLU shaders on Vulkan.
//
// Measures throughput of:
//   1. vt::MatmulBTQuantGrouped (the down_proj GEMM — grouped dev shader)
//   2. vt::MoeGateUpSwiGLUGrouped (the fused gate+up+SwiGLU — MoE shader)
//
// Reports GB/s of weight bytes read and GFLOP/s of effective MACs.
//
// Usage: ./bench_vulkan_tq_grouped [--iters N] [--p P] [--n N] [--k K] [--e E]
//
// Defaults: --iters 100 --p 1 --n 4096 --k 4096 --e 8 (decode shape)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"
#include "vt/vulkan/vulkan_context.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

struct B_ {
  Backend& b;
  void* p_;
  B_(Backend& x, size_t e, size_t eb) : b(x), p_(x.Alloc(e * eb)) {}
  ~B_() { b.Free(p_); }
  void* p() const { return p_; }
};

std::vector<float> SeededFloats(size_t n, float scale, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    v[i] = scale * (static_cast<float>(s % 1000u) / 500.0f - 1.0f);
  }
  return v;
}

std::vector<int32_t> SeededExpertIds(size_t p, int e, uint32_t seed) {
  std::vector<int32_t> ids(p);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < p; ++i) {
    s = s * 1103515245u + 12345u;
    ids[i] = static_cast<int32_t>(s % static_cast<uint32_t>(e));
  }
  return ids;
}

// Build TQ2_0 weight blocks: { u8 qs[64]; u16 d; } = 66 bytes per block.
std::vector<uint8_t> BuildTQ2_0(int64_t rows, int64_t k) {
  const int64_t nb = k / 256;
  std::vector<uint8_t> wq(rows * nb * 66, 0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t b = 0; b < nb; ++b) {
      uint8_t* blk = wq.data() + (row * nb + b) * 66;
      uint32_t seed = 71u + static_cast<uint32_t>(row) * 131u +
                      static_cast<uint32_t>(b);
      for (int64_t e = 0; e < 256; ++e) {
        const int64_t j = (e >= 128) ? 32 : 0;
        const int64_t l = (e % 128) / 32, kk = e % 32;
        seed = seed * 1103515245u + 12345u;
        const int code = static_cast<int>(seed % 3u);
        if (code != 0) {
          blk[j + kk] |= static_cast<uint8_t>(code << (l * 2));
        }
      }
      const uint16_t dhalf = 0x3800;  // f16(0.5)
      std::memcpy(blk + 64, &dhalf, 2);
    }
  }
  return wq;
}

// Build TQ1_0 weight blocks: { u8 qs[48]; u8 qh[4]; u16 d; } = 54 bytes.
std::vector<uint8_t> BuildTQ1_0(int64_t rows, int64_t k) {
  const int64_t nb = k / 256;
  std::vector<uint8_t> wq(rows * nb * 54, 0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t b = 0; b < nb; ++b) {
      uint8_t* blk = wq.data() + (row * nb + b) * 54;
      uint32_t seed = 13u + static_cast<uint32_t>(row) * 257u +
                      static_cast<uint32_t>(b);
      for (int64_t e = 0; e < 256; ++e) {
        seed = seed * 1103515245u + 12345u;
        const int code = static_cast<int>(seed % 3u) - 1;
        if (e < 160) {
          int64_t m = e % 32;
          if (e / 32 == 0) {
            blk[m] = static_cast<uint8_t>((code + 1) & 0xFF);
          }
        } else if (e < 240) {
          int64_t m = (e - 160) % 16;
          if ((e - 160) / 16 == 0) {
            blk[32 + m] = static_cast<uint8_t>((code + 1) & 0xFF);
          }
        } else {
          int64_t j = (e - 240) % 4;
          if ((e - 240) / 4 == 0) {
            blk[48 + j] = static_cast<uint8_t>((code + 1) & 0xFF);
          }
        }
      }
      const uint16_t dhalf = 0x3800;  // f16(0.5)
      std::memcpy(blk + 52, &dhalf, 2);
    }
  }
  return wq;
}

struct BenchResult {
  double ms_per_call;
  double gbps;    // weight GB/s
  double gflops;  // effective MAC GFLOP/s
};

// Benchmark kMatmulBTQuantGrouped (down_proj GEMM).
BenchResult RunGroupedBench(Backend& vk, Queue& vq, const Device& vd,
                            DType wdt, int64_t p, int64_t n, int64_t k,
                            int64_t e, int iters) {
  const size_t wbytes_per_row = vt::cpu::QuantActRowBytes(wdt, k);
  const int64_t wrows = e * n;
  std::vector<uint8_t> wq;
  if (wdt == DType::kTQ2_0) wq = BuildTQ2_0(wrows, k);
  else wq = BuildTQ1_0(wrows, k);

  std::vector<float> act = SeededFloats(p * k, 1.5f, 29u);
  std::vector<int32_t> eids = SeededExpertIds(p, static_cast<int>(e), 37u);

  B_ vb(vk, wrows * wbytes_per_row, 1);
  B_ va(vk, p * k, 4);
  B_ vo(vk, p * n, 4);
  B_ ve(vk, p, 4);
  std::memcpy(vb.p(), wq.data(), wq.size());
  std::memcpy(va.p(), act.data(), p * k * 4);
  std::memcpy(ve.p(), eids.data(), p * 4);
  vk.Copy(vq, va.p(), act.data(), p * k * 4);
  vk.Copy(vq, vb.p(), wq.data(), wq.size());
  vk.Copy(vq, ve.p(), eids.data(), p * 4);
  vk.Synchronize(vq);

  Tensor vbt = Tensor::Contiguous(vb.p(), wdt, vd, {wrows, k});
  Tensor vat = Tensor::Contiguous(va.p(), DType::kF32, vd, {p, k});
  Tensor vot = Tensor::Contiguous(vo.p(), DType::kF32, vd, {p, n});
  Tensor vet = Tensor::Contiguous(ve.p(), DType::kI32, vd, {p});

  // Warmup
  for (int i = 0; i < 5; ++i) {
    vt::MatmulBTQuantGrouped(vq, vot, vat, vbt, vet);
  }
  vk.Synchronize(vq);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    vt::MatmulBTQuantGrouped(vq, vot, vat, vbt, vet);
  }
  vk.Synchronize(vq);
  auto t1 = std::chrono::high_resolution_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_per_call = total_ms / iters;
  // Each of P rows reads N weight rows (one per expert), each wbytes_per_row.
  const double weight_bytes = static_cast<double>(p * n * wbytes_per_row);
  const double macs = 2.0 * p * n * k;
  const double gbps = weight_bytes / (ms_per_call * 1e-3) / 1e9;
  const double gflops = macs / (ms_per_call * 1e-3) / 1e9;
  return {ms_per_call, gbps, gflops};
}

// Benchmark kMoeGateUpSwiGLUGrouped (fused gate+up+SwiGLU).
BenchResult RunMoeBench(Backend& vk, Queue& vq, const Device& vd,
                        DType wdt, int64_t p, int64_t n, int64_t k,
                        int64_t e, int iters) {
  const size_t wbytes_per_row = vt::cpu::QuantActRowBytes(wdt, k);
  const int64_t wrows = e * n;
  std::vector<uint8_t> gate_q, up_q;
  if (wdt == DType::kTQ2_0) {
    gate_q = BuildTQ2_0(wrows, k);
    up_q = BuildTQ2_0(wrows, k);
  } else {
    gate_q = BuildTQ1_0(wrows, k);
    up_q = BuildTQ1_0(wrows, k);
  }

  // MoE bcast: act is [1, K] (single token, broadcast to P rows).
  std::vector<float> act = SeededFloats(k, 1.5f, 29u);
  std::vector<int32_t> eids = SeededExpertIds(p, static_cast<int>(e), 37u);

  B_ vgw(vk, wrows * wbytes_per_row, 1);
  B_ vuw(vk, wrows * wbytes_per_row, 1);
  B_ va(vk, k, 4);
  B_ vo(vk, p * n, 4);
  B_ ve(vk, p, 4);
  std::memcpy(vgw.p(), gate_q.data(), gate_q.size());
  std::memcpy(vuw.p(), up_q.data(), up_q.size());
  std::memcpy(va.p(), act.data(), k * 4);
  std::memcpy(ve.p(), eids.data(), p * 4);
  vk.Copy(vq, va.p(), act.data(), k * 4);
  vk.Copy(vq, vgw.p(), gate_q.data(), gate_q.size());
  vk.Copy(vq, vuw.p(), up_q.data(), up_q.size());
  vk.Copy(vq, ve.p(), eids.data(), p * 4);
  vk.Synchronize(vq);

  Tensor vgwt = Tensor::Contiguous(vgw.p(), wdt, vd, {wrows, k});
  Tensor vuwt = Tensor::Contiguous(vuw.p(), wdt, vd, {wrows, k});
  Tensor vat = Tensor::Contiguous(va.p(), DType::kF32, vd, {1, k});
  Tensor vot = Tensor::Contiguous(vo.p(), DType::kF32, vd, {p, n});
  Tensor vet = Tensor::Contiguous(ve.p(), DType::kI32, vd, {p});

  const float limit = 1.0f;

  // Warmup
  for (int i = 0; i < 5; ++i) {
    vt::MoeGateUpSwiGLUGrouped(vq, vot, vat, vgwt, vuwt, vet, limit);
  }
  vk.Synchronize(vq);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    vt::MoeGateUpSwiGLUGrouped(vq, vot, vat, vgwt, vuwt, vet, limit);
  }
  vk.Synchronize(vq);
  auto t1 = std::chrono::high_resolution_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_per_call = total_ms / iters;
  // Fused gate+up: 2x weight reads (gate + up), each P*N*wbytes_per_row.
  const double weight_bytes = 2.0 * static_cast<double>(p * n * wbytes_per_row);
  const double macs = 2.0 * 2.0 * p * n * k;  // 2 GEMMs
  const double gbps = weight_bytes / (ms_per_call * 1e-3) / 1e9;
  const double gflops = macs / (ms_per_call * 1e-3) / 1e9;
  return {ms_per_call, gbps, gflops};
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 100;
  int64_t p = 1, n = 4096, k = 4096, e = 8;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--p" && i + 1 < argc) p = std::atoll(argv[++i]);
    else if (a == "--n" && i + 1 < argc) n = std::atoll(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoll(argv[++i]);
    else if (a == "--e" && i + 1 < argc) e = std::atoll(argv[++i]);
    else if (a == "--help") {
      std::printf("usage: %s [--iters N] [--p P] [--n N] [--k K] [--e E]\n", argv[0]);
      return 0;
    }
  }

  if (!vt::vulkan::VulkanDeviceAvailable()) {
    std::fprintf(stderr, "No Vulkan device available\n");
    return 1;
  }

  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue vq = vk.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};

  std::printf("=== TQ grouped + MoE Vulkan microbenchmark ===\n");
  std::printf("Shape: P=%lld N=%lld K=%lld E=%lld, iters=%d\n\n",
              (long long)p, (long long)n, (long long)k, (long long)e, iters);

  for (DType wdt : {DType::kTQ1_0, DType::kTQ2_0}) {
    const char* name = (wdt == DType::kTQ2_0) ? "TQ2_0" : "TQ1_0";
    const size_t wb = vt::cpu::QuantActRowBytes(wdt, k);

    // Grouped dev (down_proj)
    {
      auto r = RunGroupedBench(vk, vq, vd, wdt, p, n, k, e, iters);
      std::printf("%s grouped-dev: %.3f ms/call | %.1f GB/s weight | %.1f GFLOP/s | "
                  "wbytes=%zu\n",
                  name, r.ms_per_call, r.gbps, r.gflops, p * n * wb);
    }
    // MoE fused gate+up+SwiGLU
    {
      auto r = RunMoeBench(vk, vq, vd, wdt, p, n, k, e, iters);
      std::printf("%s moe-gate-up:  %.3f ms/call | %.1f GB/s weight | %.1f GFLOP/s | "
                  "wbytes=%zu\n",
                  name, r.ms_per_call, r.gbps, r.gflops, 2 * p * n * wb);
    }
    std::printf("\n");
  }

  vk.DestroyQueue(vq);
  return 0;
}
