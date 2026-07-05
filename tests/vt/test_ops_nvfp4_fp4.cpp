// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// TRUE W4A4 (fp4 activations x fp4 weights) ops — the 27B path (notes §7). These
// validate the CPU kernels vt::ScaledFp4Quant + vt::MatmulNvfp4Fp4 against the
// pinned vLLM CPU truth:
//   - ScaledFp4Quant must be BYTE-EXACT vs vllm::RefScaledFp4Quant, and its
//     decode must reproduce vllm::RefNvfp4QuantDequant's x_dq.
//   - MatmulNvfp4Fp4( ScaledFp4Quant(x), W ) must equal vllm::RunNvfp4Emulation
//     (the emulated true-W4A4 linear) up to K-reduction float order.
// CPU-only (also the reference the future CUDA kernels validate against).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32, kE2M1Lut
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

// Decode a fp4 activation row-trip from the ScaledFp4Quant outputs, mirroring the
// GEMM's a-operand: a_fp4 * f8(a_scale) / input_global_scale (== x_dq).
float DecodeActElem(const uint8_t* packed, const uint8_t* scale, int64_t k, int64_t row,
                    int64_t col, float input_global_scale) {
  const int64_t groups = k / 16;
  const uint8_t nib = (col % 2 == 0) ? (packed[(row * k + col) / 2] & 0x0FU)
                                     : (packed[(row * k + col) / 2] >> 4);
  const float mag = vllm::kE2M1Lut[nib & 0x7U] * ((nib & 0x8U) ? -1.0F : 1.0F);
  const float sf = vllm::F8E4M3ToF32(scale[row * groups + col / 16]);
  return mag * sf / input_global_scale;  // block_scale = sf/global
}
}  // namespace

TEST_CASE("scaled_fp4_quant CPU == vllm::RefScaledFp4Quant (byte-exact) + decode") {
  const int64_t M = 5, K = 64;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0F, 3.0F);
  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  const float input_global_scale = 7.3F;  // on-disk divisor (used directly)

  std::vector<uint8_t> ref_packed(static_cast<size_t>(M * K / 2));
  std::vector<uint8_t> ref_scale(static_cast<size_t>(M * K / 16));
  vllm::RefScaledFp4Quant(x.data(), M, K, input_global_scale, ref_packed.data(),
                          ref_scale.data());

  std::vector<uint8_t> op_packed(static_cast<size_t>(M * K / 2), 0);
  std::vector<uint8_t> op_scale(static_cast<size_t>(M * K / 16), 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
  Tensor tp = Tensor::Contiguous(op_packed.data(), DType::kI8, Cpu(), {M, K / 2});
  Tensor ts = Tensor::Contiguous(op_scale.data(), DType::kI8, Cpu(), {M, K / 16});
  Queue q = CpuQueue();
  vt::ScaledFp4Quant(q, tp, ts, tx, input_global_scale);

  for (size_t i = 0; i < ref_packed.size(); ++i) CHECK(op_packed[i] == ref_packed[i]);
  for (size_t i = 0; i < ref_scale.size(); ++i) CHECK(op_scale[i] == ref_scale[i]);

  // Decode == RefNvfp4QuantDequant x_dq (the emulation activation round-trip).
  std::vector<float> x_dq(static_cast<size_t>(M * K));
  vllm::RefNvfp4QuantDequant(x.data(), M, K, input_global_scale, x_dq.data());
  for (int64_t r = 0; r < M; ++r)
    for (int64_t c = 0; c < K; ++c) {
      const float got = DecodeActElem(op_packed.data(), op_scale.data(), K, r, c, input_global_scale);
      CHECK(got == doctest::Approx(x_dq[static_cast<size_t>(r * K + c)]).epsilon(1e-6));
    }
}

TEST_CASE("matmul_nvfp4_fp4 CPU == vllm::RunNvfp4Emulation") {
  const int64_t M = 4, K = 96, N = 7;
  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);

  // Random bf16-ish activations (f32).
  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);

  // Random fp4 weight bytes [N, K/2] + fp8 group scales [N, K/16] (sane, no NaN)
  // + a global divisor.
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& b : w_packed) b = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& s : w_scale) s = vllm::F32ToF8E4M3(scale_d(rng));
  const float weight_global_scale_disk = 6.1F;   // divisor
  const float input_global_scale_disk = 9.4F;    // divisor

  // Reference: the emulated true-W4A4 linear.
  std::vector<float> ref(static_cast<size_t>(M * N));
  vllm::RunNvfp4Emulation(x.data(), M, K, w_packed.data(), w_scale.data(),
                          weight_global_scale_disk, input_global_scale_disk, N, ref.data());

  // Op path: quantize activations, then fp4xfp4 GEMM with the folded alpha.
  std::vector<uint8_t> a_packed(static_cast<size_t>(M * K / 2), 0);
  std::vector<uint8_t> a_scale(static_cast<size_t>(M * K / 16), 0);
  Queue q = CpuQueue();
  {
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
    Tensor tp = Tensor::Contiguous(a_packed.data(), DType::kI8, Cpu(), {M, K / 2});
    Tensor ts = Tensor::Contiguous(a_scale.data(), DType::kI8, Cpu(), {M, K / 16});
    vt::ScaledFp4Quant(q, tp, ts, tx, input_global_scale_disk);
  }
  const float alpha = (1.0F / input_global_scale_disk) * (1.0F / weight_global_scale_disk);
  std::vector<float> out(static_cast<size_t>(M * N), -1.0F);
  Tensor tap = Tensor::Contiguous(a_packed.data(), DType::kI8, Cpu(), {M, K / 2});
  Tensor tas = Tensor::Contiguous(a_scale.data(), DType::kI8, Cpu(), {M, K / 16});
  Tensor tbp = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {N, K / 2});
  Tensor tbs = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {N, K / 16});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {M, N});
  vt::MatmulNvfp4Fp4(q, to, tap, tas, tbp, tbs, alpha);

  for (int64_t i = 0; i < M * N; ++i) {
    // Real-arithmetic identical to RunNvfp4Emulation; only float K-order differs.
    CHECK(out[static_cast<size_t>(i)] ==
          doctest::Approx(ref[static_cast<size_t>(i)]).epsilon(1e-5).scale(1.0));
  }
}

// --- CUDA device-vs-CPU cross-check (GB10; skips cleanly without a GPU) --------
namespace {
bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor GpuTensor(const std::vector<int64_t>& shape) {
  Tensor t;
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

TEST_CASE("scaled_fp4_quant + matmul_nvfp4_fp4 CUDA == CPU") {
  if (!HasCuda()) return;
  auto& b = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = b.CreateQueue();

  const int64_t M = 40, K = 128, N = 33;  // M >= 32 exercises the WMMA path
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0F, 2.0F);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> scale_d(0.05F, 4.0F);

  std::vector<float> x(static_cast<size_t>(M * K));
  for (auto& v : x) v = nd(rng);
  std::vector<uint8_t> w_packed(static_cast<size_t>(N * K / 2));
  for (auto& v : w_packed) v = static_cast<uint8_t>(byte_d(rng));
  std::vector<uint8_t> w_scale(static_cast<size_t>(N * K / 16));
  for (auto& v : w_scale) v = vllm::F32ToF8E4M3(scale_d(rng));
  const float input_global_scale = 8.2F, weight_global_scale = 5.5F;
  const float alpha = (1.0F / input_global_scale) * (1.0F / weight_global_scale);

  // CPU reference (already validated vs vllm::RunNvfp4Emulation above).
  std::vector<uint8_t> cpu_ap(static_cast<size_t>(M * K / 2), 0), cpu_as(static_cast<size_t>(M * K / 16), 0);
  std::vector<float> cpu_out(static_cast<size_t>(M * N), 0);
  {
    Queue cq = CpuQueue();
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {M, K});
    Tensor tp = Tensor::Contiguous(cpu_ap.data(), DType::kI8, Cpu(), {M, K / 2});
    Tensor ts = Tensor::Contiguous(cpu_as.data(), DType::kI8, Cpu(), {M, K / 16});
    vt::ScaledFp4Quant(cq, tp, ts, tx, input_global_scale);
    Tensor tbp = Tensor::Contiguous(w_packed.data(), DType::kI8, Cpu(), {N, K / 2});
    Tensor tbs = Tensor::Contiguous(w_scale.data(), DType::kI8, Cpu(), {N, K / 16});
    Tensor to = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {M, N});
    vt::MatmulNvfp4Fp4(cq, to, tp, ts, tbp, tbs, alpha);
  }

  // Device path.
  auto up = [&](const void* h, size_t nb) { void* p = b.Alloc(nb); b.Copy(gq, p, h, nb); return p; };
  void* dx = up(x.data(), x.size() * sizeof(float));
  void* dbp = up(w_packed.data(), w_packed.size());
  void* dbs = up(w_scale.data(), w_scale.size());
  void* dap = b.Alloc(static_cast<size_t>(M * K / 2));
  void* das = b.Alloc(static_cast<size_t>(M * K / 16));
  void* dout = b.Alloc(static_cast<size_t>(M * N) * sizeof(float));
  Tensor tx = GpuTensor({M, K}); tx.data = dx; tx.dtype = DType::kF32; tx.device = Gpu();
  Tensor tap = GpuTensor({M, K / 2}); tap.data = dap; tap.dtype = DType::kI8; tap.device = Gpu();
  Tensor tas = GpuTensor({M, K / 16}); tas.data = das; tas.dtype = DType::kI8; tas.device = Gpu();
  Tensor tbp = GpuTensor({N, K / 2}); tbp.data = dbp; tbp.dtype = DType::kI8; tbp.device = Gpu();
  Tensor tbs = GpuTensor({N, K / 16}); tbs.data = dbs; tbs.dtype = DType::kI8; tbs.device = Gpu();
  Tensor to = GpuTensor({M, N}); to.data = dout; to.dtype = DType::kF32; to.device = Gpu();
  vt::ScaledFp4Quant(gq, tap, tas, tx, input_global_scale);
  vt::MatmulNvfp4Fp4(gq, to, tap, tas, tbp, tbs, alpha);

  std::vector<uint8_t> g_ap(static_cast<size_t>(M * K / 2)), g_as(static_cast<size_t>(M * K / 16));
  std::vector<float> g_out(static_cast<size_t>(M * N));
  b.Copy(gq, g_ap.data(), dap, g_ap.size());
  b.Copy(gq, g_as.data(), das, g_as.size());
  b.Copy(gq, g_out.data(), dout, g_out.size() * sizeof(float));
  b.Synchronize(gq);

  // Quant: the device kernel uses the hardware fp8 cast (matches the real vLLM
  // kernel), the CPU op the emulation codec — expect byte-exact for almost all
  // groups; a handful of fp8-tie edge cases are benign (allow a small fraction).
  size_t scale_mismatch = 0;
  for (size_t i = 0; i < g_as.size(); ++i) scale_mismatch += (g_as[i] != cpu_as[i]);
  CHECK(scale_mismatch <= g_as.size() / 50 + 1);  // <= ~2% fp8 ties
  // GEMM close device-vs-CPU (bf16 tensor-core dequant vs f32 CPU): matmul tol.
  for (size_t i = 0; i < g_out.size(); ++i)
    CHECK(g_out[i] == doctest::Approx(cpu_out[i]).epsilon(0.02).scale(1.0));

  for (void* p : {dx, dbp, dbs, dap, das, dout}) b.Free(p);
  b.DestroyQueue(gq);
}

TEST_CASE("matmul_nvfp4_fp4 validates shapes loudly") {
  std::vector<uint8_t> buf(64, 0);
  std::vector<float> ob(16, 0);
  Tensor ap = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {2, 8});   // K=16
  Tensor as = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {2, 1});
  Tensor bp = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {3, 4});   // K=8 mismatch
  Tensor bs = Tensor::Contiguous(buf.data(), DType::kI8, Cpu(), {3, 1});
  Tensor o = Tensor::Contiguous(ob.data(), DType::kF32, Cpu(), {2, 3});
  Queue q = CpuQueue();
  CHECK_THROWS_AS(vt::MatmulNvfp4Fp4(q, o, ap, as, bp, bs, 1.0F), std::runtime_error);
}
