// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Per-tensor FP8 (W8A16) dequant-GEMM. Validates that vt::MatmulFp8, which reads
// the fp8 weight directly from device memory and dequantizes on the fly
// (bit-for-bit vllm::DequantFp8ToBf16), matches the reference path
// Matmul(act, DequantFp8ToBf16(w).T) within matmul tolerance — on BOTH the CPU
// kernel (bit-identical dequant, direct check) and the CUDA kernel (skipped
// cleanly when no GPU is present).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
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

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
  }
  CHECK(bad == 0);
}

// Synthetic per-tensor FP8 weight [N, K]: random fp8-e4m3fn bytes, excluding the
// NaN encodings (0x7F / 0xFF) so the reference stays finite.
struct Fp8Weight {
  std::vector<uint8_t> bytes;  // [N, K]
  float scale;
};

Fp8Weight MakeFp8Weight(int64_t n, int64_t k, uint32_t seed) {
  Fp8Weight w;
  w.bytes.resize(static_cast<size_t>(n * k));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (auto& b : w.bytes) {
    int v = byte_dist(rng);
    if (v == 0x7F || v == 0xFF) v = 0x00;  // avoid NaN
    b = static_cast<uint8_t>(v);
  }
  w.scale = 0.375f;  // any finite positive per-tensor scale
  return w;
}

std::vector<float> RandomF32(size_t numel, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> v(numel);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}

// Reference: dequant fp8 weight to bf16 [N, K] (bit-for-bit DequantFp8ToBf16),
// transpose to [K, N] (Matmul-B layout), then out[M,N] = act @ W_deq_T on CPU.
std::vector<float> ReferenceMatmul(const Fp8Weight& w, const std::vector<uint16_t>& act_bf16,
                                   int64_t m, int64_t k, int64_t n) {
  std::vector<uint16_t> w_deq(static_cast<size_t>(n * k));
  vllm::DequantFp8ToBf16(w.bytes.data(), w.scale, n * k, w_deq.data());
  std::vector<uint16_t> w_deq_t(static_cast<size_t>(k * n));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j)
      w_deq_t[static_cast<size_t>(j * n + i)] = w_deq[static_cast<size_t>(i * k + j)];

  std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
  Tensor ta = MakeTensor(const_cast<uint16_t*>(act_bf16.data()), DType::kBF16, Cpu(), {m, k});
  Tensor tb = MakeTensor(w_deq_t.data(), DType::kBF16, Cpu(), {k, n});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {m, n});
  Queue cq{Cpu(), nullptr};
  vt::Matmul(cq, to, ta, tb);
  return out;
}

const struct {
  int64_t m, k, n;
} kDims[] = {{1, 16, 1},   {5, 64, 13},   {17, 128, 32}, {3, 100, 7},
             {64, 128, 70}, {33, 80, 40}, {96, 48, 129}, {8, 40, 5}};

}  // namespace

TEST_CASE("CPU matmul_fp8 matches Matmul(act, dequant(w).T)") {
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  uint32_t seed = 3000;
  for (const auto& d : kDims) {
    CAPTURE(d.m);
    CAPTURE(d.k);
    CAPTURE(d.n);
    const Fp8Weight w = MakeFp8Weight(d.n, d.k, seed);
    const auto act_f = RandomF32(static_cast<size_t>(d.m * d.k), seed + 1);
    const auto act_bf16 = ToBf16(act_f);
    const std::vector<float> ref = ReferenceMatmul(w, act_bf16, d.m, d.k, d.n);

    QueueGuard cq(cpu);
    std::vector<float> got(static_cast<size_t>(d.m * d.n), 0.0f);
    Tensor tact = MakeTensor(const_cast<uint16_t*>(act_bf16.data()), DType::kBF16, Cpu(), {d.m, d.k});
    Tensor tw = MakeTensor(const_cast<uint8_t*>(w.bytes.data()), DType::kI8, Cpu(), {d.n, d.k});
    Tensor tout = MakeTensor(got.data(), DType::kF32, Cpu(), {d.m, d.n});
    vt::MatmulFp8(cq.q, tout, tact, tw, w.scale);
    // CPU kernel: identical dequant AND identical reduction order as the
    // reference Matmul — expect bit-exact (tight tolerance).
    CheckClose(got, ref, 1e-5f, 1e-5f);
    seed += 100;
  }
}

TEST_CASE("CUDA matmul_fp8 matches Matmul(act, dequant(w).T) on odd shapes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  uint32_t seed = 5000;
  for (const auto& d : kDims) {
    CAPTURE(d.m);
    CAPTURE(d.k);
    CAPTURE(d.n);
    const Fp8Weight w = MakeFp8Weight(d.n, d.k, seed);
    const auto act_f = RandomF32(static_cast<size_t>(d.m * d.k), seed + 1);
    const auto act_bf16 = ToBf16(act_f);
    const std::vector<float> ref = ReferenceMatmul(w, act_bf16, d.m, d.k, d.n);

    {  // f32 output (bf16 activations): matmul tolerance (reduction order only).
      QueueGuard gq(gpu);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {d.m, d.k}, act_bf16.data());
      DeviceTensor dw(gpu, gq.q, DType::kI8, {d.n, d.k}, w.bytes.data());
      DeviceTensor dout(gpu, gq.q, DType::kF32, {d.m, d.n});
      vt::MatmulFp8(gq.q, dout.tensor(), dact.tensor(), dw.tensor(), w.scale);
      std::vector<float> got(static_cast<size_t>(d.m * d.n));
      dout.Download(gq.q, got.data());
      CheckClose(got, ref, 2e-3f, 2e-3f);
    }
    {  // bf16 output: allow one output bf16 ulp on top.
      QueueGuard gq(gpu);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {d.m, d.k}, act_bf16.data());
      DeviceTensor dw(gpu, gq.q, DType::kI8, {d.n, d.k}, w.bytes.data());
      DeviceTensor dout(gpu, gq.q, DType::kBF16, {d.m, d.n});
      vt::MatmulFp8(gq.q, dout.tensor(), dact.tensor(), dw.tensor(), w.scale);
      std::vector<uint16_t> got_bf16(static_cast<size_t>(d.m * d.n));
      dout.Download(gq.q, got_bf16.data());
      std::vector<float> got(got_bf16.size());
      for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(got_bf16[i]);
      CheckClose(got, ref, 4e-3f, 8e-3f);
    }
    seed += 100;
  }
}

TEST_CASE("matmul_fp8 validates shapes loudly (CPU dispatch)") {
  std::vector<uint16_t> act(4 * 16, 0);
  std::vector<uint8_t> weight(3 * 8, 0);  // wrong: inner should be K=16
  std::vector<float> out(4 * 3, 0.0f);
  Tensor tact = MakeTensor(act.data(), DType::kBF16, Cpu(), {4, 16});
  Tensor tw = MakeTensor(weight.data(), DType::kI8, Cpu(), {3, 8});
  Tensor tout = MakeTensor(out.data(), DType::kF32, Cpu(), {4, 3});
  Queue cq{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::MatmulFp8(cq, tout, tact, tw, 1.0f), std::runtime_error);
}
