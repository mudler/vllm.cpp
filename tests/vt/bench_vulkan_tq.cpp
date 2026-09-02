// bench_vulkan_tq: microbenchmark for TQ1_0/TQ2_0 keep-quant matmul on Vulkan.
//
// Measures the throughput of vt::MatmulBTQuant for TQ1_0 and TQ2_0 weight
// blocks against f32 activations, at realistic decode (M=1) and prefill
// (M=128) shapes. Reports GB/s of weight bytes read and GFLOP/s of effective
// MACs (2*K*N*M per call).
//
// Usage: ./bench_vulkan_tq [--iters N] [--m M] [--n N] [--k K]
//
// Defaults: --iters 200 --m 1 --n 4096 --k 4096 (decode shape)

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

// Build TQ2_0 weight blocks from a seeded pattern. BlockTQ2_0 = { u8 qs[64]; u16 d; } = 66 bytes.
std::vector<uint8_t> BuildTQ2_0(int64_t n, int64_t k) {
  const int64_t nb = k / 256;
  std::vector<uint8_t> wq(n * nb * 66, 0);
  for (int64_t row = 0; row < n; ++row) {
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

// Build TQ1_0 weight blocks from a seeded pattern. BlockTQ1_0 = { u8 qs[48]; u8 qh[4]; u16 d; } = 54 bytes.
std::vector<uint8_t> BuildTQ1_0(int64_t n, int64_t k) {
  const int64_t nb = k / 256;
  std::vector<uint8_t> wq(n * nb * 54, 0);
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t b = 0; b < nb; ++b) {
      uint8_t* blk = wq.data() + (row * nb + b) * 54;
      uint32_t seed = 13u + static_cast<uint32_t>(row) * 257u +
                      static_cast<uint32_t>(b);
      for (int64_t e = 0; e < 256; ++e) {
        seed = seed * 1103515245u + 12345u;
        const int code = static_cast<int>(seed % 3u) - 1;  // -1, 0, +1
        // TQ1_0 packs 5 trits per byte in qs[0..31] (elements 0..159),
        // 5 trits per byte in qs[32..47] (elements 160..239),
        // 4 trits per byte in qh[0..3] (elements 240..255).
        // We use the trit value = code+1 (0,1,2) and pack via the
        // pow3 encoding: trit_value = ((byte * pow3) & 0xFF) * 3 >> 8.
        // For simplicity, set each byte to encode the same trit for all
        // its lanes. byte = sum of trit_code * pow3^l for l in 0..4 (or 0..3).
        // We just set each byte to a deterministic value.
        if (e < 160) {
          int64_t m = e % 32;
          if (e / 32 == 0) {  // first trit lane
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

BenchResult RunBench(Backend& vk, Queue& vq, const Device& vd,
                     DType wdt, int64_t m, int64_t n, int64_t k,
                     int iters) {
  const size_t wbytes_per_row = vt::cpu::QuantActRowBytes(wdt, k);
  std::vector<uint8_t> wq;
  if (wdt == DType::kTQ2_0) wq = BuildTQ2_0(n, k);
  else wq = BuildTQ1_0(n, k);

  std::vector<float> act = SeededFloats(m * k, 1.5f, 29u);

  B_ vb(vk, n * wbytes_per_row, 1);
  B_ va(vk, m * k, 4);
  B_ vo(vk, m * n, 4);
  std::memcpy(vb.p(), wq.data(), wq.size());
  std::memcpy(va.p(), act.data(), m * k * 4);
  vk.Copy(vq, va.p(), act.data(), m * k * 4);
  vk.Copy(vq, vb.p(), wq.data(), wq.size());
  vk.Synchronize(vq);

  Tensor vbt = Tensor::Contiguous(vb.p(), wdt, vd, {n, k});
  Tensor vat = Tensor::Contiguous(va.p(), DType::kF32, vd, {m, k});
  Tensor vot = Tensor::Contiguous(vo.p(), DType::kF32, vd, {m, n});

  // Warmup
  for (int i = 0; i < 5; ++i) {
    vt::MatmulBTQuant(vq, vot, vat, vbt);
  }
  vk.Synchronize(vq);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    vt::MatmulBTQuant(vq, vot, vat, vbt);
  }
  vk.Synchronize(vq);
  auto t1 = std::chrono::high_resolution_clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_per_call = total_ms / iters;
  const double weight_bytes = static_cast<double>(n * wbytes_per_row);
  const double macs = 2.0 * m * n * k;
  const double gbps = weight_bytes / (ms_per_call * 1e-3) / 1e9;
  const double gflops = macs / (ms_per_call * 1e-3) / 1e9;

  return {ms_per_call, gbps, gflops};
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 200;
  int64_t m = 1, n = 4096, k = 4096;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--m" && i + 1 < argc) m = std::atoll(argv[++i]);
    else if (a == "--n" && i + 1 < argc) n = std::atoll(argv[++i]);
    else if (a == "--k" && i + 1 < argc) k = std::atoll(argv[++i]);
    else if (a == "--help") {
      std::printf("usage: %s [--iters N] [--m M] [--n N] [--k K]\n", argv[0]);
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

  std::printf("=== TQ keep-quant Vulkan microbenchmark ===\n");
  std::printf("Intel Arc Pro B60 (BMG G21), Mesa 26.1.7\n");
  std::printf("Shape: M=%lld N=%lld K=%lld, iters=%d\n\n",
              (long long)m, (long long)n, (long long)k, iters);

  // TQ1_0
  {
    auto r = RunBench(vk, vq, vd, DType::kTQ1_0, m, n, k, iters);
    const size_t wb = vt::cpu::QuantActRowBytes(DType::kTQ1_0, k);
    std::printf("TQ1_0:  %.3f ms/call  |  %.1f GB/s weight  |  %.1f GFLOP/s  |  "
                "wbytes=%zu\n",
                r.ms_per_call, r.gbps, r.gflops, n * wb);
  }

  // TQ2_0
  {
    auto r = RunBench(vk, vq, vd, DType::kTQ2_0, m, n, k, iters);
    const size_t wb = vt::cpu::QuantActRowBytes(DType::kTQ2_0, k);
    std::printf("TQ2_0:  %.3f ms/call  |  %.1f GB/s weight  |  %.1f GFLOP/s  |  "
                "wbytes=%zu\n",
                r.ms_per_call, r.gbps, r.gflops, n * wb);
  }

  vk.DestroyQueue(vq);
  return 0;
}
