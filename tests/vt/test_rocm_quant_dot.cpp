// ROCm keep-quant GEMM gate (KERNEL-QUANT-CIQ-GEMM-ROCM W1). The kROCM
// provider for `OpId::kMatmulBTQuant` / `kMatmulBTQuantGrouped`
// (src/vt/rocm/rocm_quant_dot.hip) is measured against the LANDED CPU
// keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle) and an
// INDEPENDENT f64 dequantize-then-dot, on the ten Q8_K-family encodings the
// CUDA sibling serves (test_cuda_quant_dot.cpp's WeightCase table).
//
// THE GATE mirrors the CUDA file: the Q8_K activation quant and the whole
// INTEGER dot are bit-identical to the CPU reference by construction, so
// ROCm-vs-CPU is asserted at a TIGHT NMSE (1e-6, f32 out) — only the per-
// super-block float scale sum is reassociated (warp reduction vs the CPU's
// sequential add). ROCm-vs-f64-dequant uses the same 5e-4 band
// test_ops_quant_dot.cpp applies. A wrong codebook index / scale unpack /
// sign blows both bands (RED-first).
//
// Skips cleanly when no AMD GPU is present, so CPU-only CI stays green.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

constexpr double kMaxNmseErr = 5e-4;      // test-backend-ops.cpp:4277 band
constexpr double kMaxNmseVsCpu = 1e-6;    // integer core exact; scale sum only

bool HasRocm() {
  try {
    vt::GetBackend(DeviceType::kROCM);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kROCM, 0}; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
  // f64-dequant ceiling override (0 = kMaxNmseErr); see the CUDA table for why
  // the IQ1 family needs a wider ACTIVATION-error band while the ROCm-vs-CPU
  // bound below stays shared and unrelaxed.
  double nmse_ref_max = 0.0;
};

const WeightCase kCases[] = {
    {DType::kIQ2_XXS, 256, 66, 0, -1, "iq2_xxs"},
    {DType::kIQ3_XXS, 256, 98, 0, -1, "iq3_xxs"},
    {DType::kIQ2_S, 256, 82, 0, -1, "iq2_s"},
    {DType::kIQ1_S, 256, 50, 0, -1, "iq1_s", 2e-3},
    {DType::kIQ1_XXXS, 256, 38, 0, -1, "iq1_xxxs", 2e-3},
    {DType::kQ2_K, 256, 84, 80, 82, "q2_K"},
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
    // IQ1 sub-block scales live INSIDE the weight (qh bits 12-14 / sc nibbles):
    // narrow them to encoder-plausible values exactly as the CUDA table does.
    if (c.dtype == DType::kIQ1_S) {
      for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh = 0;
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
        const uint16_t ls = static_cast<uint16_t>(2 + ((i + ib) % 3));
        qh = static_cast<uint16_t>((qh & 0x8FFFU) | (ls << 12));
        std::memcpy(blk + 34 + 2 * ib, &qh, sizeof(qh));
      }
    }
    if (c.dtype == DType::kIQ1_XXXS) {
      for (int ib = 0; ib < 8; ++ib) {
        uint8_t& byte = blk[34 + ib / 2];
        const int shift = 4 * (ib & 1);
        const uint8_t ls = static_cast<uint8_t>(2 + ((i + ib) % 3));
        const uint8_t keep_sign = static_cast<uint8_t>((byte >> shift) & 0x8);
        byte = static_cast<uint8_t>((byte & ~(0xFU << shift)) |
                                    ((keep_sign | ls) << shift));
      }
    }
  }
  return bytes;
}

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

}  // namespace

TEST_CASE("ROCm keep-quant GEMM == CPU reference and f64 dequant (Q8_K family)") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq =
            RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
          Tensor bt =
              Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
          bt.dtype = c.dtype;
          Tensor ot =
              Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }

        // --- ROCm path (device tensors; discrete card, so real staging) ------
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> rocm_out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // --- f64 independent reference --------------------------------------
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got =
                rocm_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        const double ref_ceiling =
            c.nmse_ref_max > 0 ? c.nmse_ref_max : kMaxNmseErr;
        CHECK(nmse_ref <= ref_ceiling);     // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);   // matches the CPU oracle (int core exact)
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("ROCm keep-quant registers the native kROCM providers") {
  // The registration flips the GGUF loader's keep-quant default ON on a ROCm
  // device (GgufQuantComputeAvailable -> OpRegistered(kMatmulBTQuant,kROCM)).
  // Present only in a HIP build.
  if (!HasRocm()) return;
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuant, DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, DeviceType::kROCM));
}

TEST_CASE(
    "ROCm grouped keep-quant GEMM == CPU grouped golden and it WRITES the "
    "output") {
  if (!HasRocm()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // All ten encodings, decode + prefill shapes, broadcast and per-row arms —
  // the same matrix the CUDA grouped gate runs, over a POISONED output buffer.
  struct GroupedShape {
    int64_t P;
    int64_t n;
    int64_t E;
    bool bcast;
  };
  const GroupedShape kGroupedShapes[] = {
      {6, 3, 4, false}, {32, 7, 8, false}, {16, 5, 2, true}};
  int64_t combos = 0;
  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (const GroupedShape& g : kGroupedShapes) {
      CAPTURE(std::string(c.name));
      CAPTURE(g.P);
      CAPTURE(g.n);
      CAPTURE(g.E);
      CAPTURE(g.bcast);
      const int64_t arows = g.bcast ? 1 : g.P;
      std::vector<uint8_t> wq =
          RandomBlocks(c, g.E * g.n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> af(static_cast<size_t>(arows * k));
      GenerateData(1.0F, af.size(), af.data());
      std::vector<int32_t> ids(g.P);
      for (int64_t p = 0; p < g.P; ++p) ids[static_cast<size_t>(p)] = p % g.E;
      const size_t outn = static_cast<size_t>(g.P * g.n);

      // --- CPU golden (the landed grouped keep-quant kernel over host tensors)
      std::vector<float> cpu_out(outn, 1337.0F);
      {
        Tensor at =
            Tensor::Contiguous(af.data(), DType::kF32, Cpu(), {arows, k});
        Tensor wt =
            Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {g.E * g.n, k});
        wt.dtype = c.dtype;
        Tensor et =
            Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {g.P});
        Tensor ot =
            Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {g.P, g.n});
        vt::MatmulBTQuantGrouped(cq, ot, at, wt, et);
      }

      // --- ROCm path over a POISONED output buffer -------------------------
      void* d_a = gpu.Alloc(af.size() * sizeof(float));
      void* d_w = gpu.Alloc(wq.size());
      void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
      void* d_o = gpu.Alloc(outn * sizeof(float));
      std::vector<float> poison(outn, 1337.0F);
      gpu.Copy(gq, d_a, af.data(), af.size() * sizeof(float));
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
      gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
      gpu.Synchronize(gq);
      Tensor at = DevTensor(d_a, DType::kF32, {arows, k});
      Tensor wt = DevTensor(d_w, c.dtype, {g.E * g.n, k});
      Tensor et = DevTensor(d_e, DType::kI32, {g.P});
      Tensor ot = DevTensor(d_o, DType::kF32, {g.P, g.n});
      vt::MatmulBTQuantGrouped(gq, ot, at, wt, et);
      std::vector<float> got(outn, 0.0F);
      gpu.Copy(gq, got.data(), d_o, got.size() * sizeof(float));
      gpu.Synchronize(gq);
      gpu.Free(d_a);
      gpu.Free(d_w);
      gpu.Free(d_e);
      gpu.Free(d_o);

      int poisoned = 0;
      int nonfinite = 0;
      double num = 0, den = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == 1337.0F) ++poisoned;
        if (!std::isfinite(got[i])) ++nonfinite;
        num += (got[i] - cpu_out[i]) * (got[i] - cpu_out[i]);
        den += cpu_out[i] * cpu_out[i];
      }
      const double nmse = den > 0 ? num / den : num;
      CAPTURE(nmse);
      CHECK(poisoned == 0);   // a dispatch that launches nothing lands HERE
      CHECK(nonfinite == 0);
      CHECK(nmse <= kMaxNmseVsCpu);
      ++combos;
    }
  }
  // doctest prints "SUCCESS!" for a loop that never ran. Say how many it ran.
  CAPTURE(combos);
  CHECK(combos ==
        static_cast<int64_t>(std::size(kCases) * std::size(kGroupedShapes)));
  CHECK(combos > 0);
  gpu.DestroyQueue(gq);
}
