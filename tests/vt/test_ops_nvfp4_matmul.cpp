// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// M2.2a — the NVFP4 W4A16 dequant-GEMM. Validates that the CUDA kernel
// (vt::MatmulNvfp4), which reads the fp4 weight directly from device memory and
// dequantizes on the fly, matches the reference path
// Matmul(act, DequantNvfp4ToBf16(w).T) within matmul tolerance. Guarded like
// test_cuda_ops.cpp: skips cleanly when no GPU is present (the op is CUDA-only).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
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

// Device buffer + tensor view; uploads on construction when host data given.
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
    if (!(std::fabs(got[i] - want[i]) <= tol)) {  // catches NaN too
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

// Builds a synthetic modelopt W4A16_NVFP4 weight [N, K] (K % 16 == 0):
//   packed[N, K/2] random fp4 bytes; scale[N, K/16] random POSITIVE finite
//   fp8-e4m3fn bytes (sign 0, exp in [1,13], mant in [0,7] — no NaN, no
//   subnormal-zero rows so the reference stays well-scaled).
struct Nvfp4Weight {
  std::vector<uint8_t> packed;  // [N, K/2]
  std::vector<uint8_t> scale;   // [N, K/16]
  float scale2;
};

Nvfp4Weight MakeNvfp4Weight(int64_t n, int64_t k, uint32_t seed) {
  Nvfp4Weight w;
  w.packed.resize(static_cast<size_t>(n * (k / 2)));
  w.scale.resize(static_cast<size_t>(n * (k / 16)));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> exp_dist(1, 13);
  std::uniform_int_distribution<int> mant_dist(0, 7);
  for (auto& b : w.packed) b = static_cast<uint8_t>(byte_dist(rng));
  for (auto& s : w.scale) {
    const int exp = exp_dist(rng);
    const int mant = mant_dist(rng);
    s = static_cast<uint8_t>((exp << 3) | mant);  // sign=0, positive normal
  }
  w.scale2 = 0.375f;  // stand-in for amax/2688; any finite positive value works
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

// Reference: dequant the NVFP4 weight to bf16 [N, K], transpose to [K, N]
// (Matmul-B layout), then out_ref[M,N] = act[M,K] @ W_deq_T on the CPU.
std::vector<float> ReferenceMatmul(const Nvfp4Weight& w, const std::vector<uint16_t>& act_bf16,
                                   int64_t m, int64_t k, int64_t n) {
  std::vector<uint16_t> w_deq(static_cast<size_t>(n * k));  // [N, K]
  vllm::DequantNvfp4ToBf16(w.packed.data(), w.scale.data(), w.scale2, n, k, w_deq.data());
  std::vector<uint16_t> w_deq_t(static_cast<size_t>(k * n));  // [K, N]
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

// Same dequant-then-matmul reference, but with f32 activations: dequant to bf16
// [N, K], then out[m,n] = sum_k act_f32[m,k] * f32(bf16_weight[n,k]) in f32 — an
// apples-to-apples reference for the f32-activation kernel path (same weight
// bytes, same f32 accumulation, only the reduction order differs).
std::vector<float> ReferenceMatmulF32Act(const Nvfp4Weight& w, const std::vector<float>& act_f32,
                                         int64_t m, int64_t k, int64_t n) {
  std::vector<uint16_t> w_deq(static_cast<size_t>(n * k));  // [N, K] bf16
  vllm::DequantNvfp4ToBf16(w.packed.data(), w.scale.data(), w.scale2, n, k, w_deq.data());
  std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p)
        acc += act_f32[static_cast<size_t>(i * k + p)] *
               vt::BF16ToF32(w_deq[static_cast<size_t>(j * k + p)]);
      out[static_cast<size_t>(i * n + j)] = acc;
    }
  return out;
}

}  // namespace

TEST_CASE("CUDA matmul_nvfp4 matches Matmul(act, dequant(w).T) on odd shapes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  struct Dims {
    int64_t m, k, n;
  };
  // K is a multiple of 16 (one group at 16); odd M/N exercise the grid tails.
  const Dims dims[] = {{1, 16, 1}, {5, 64, 13}, {17, 128, 32}, {3, 256, 7}};
  uint32_t seed = 4000;
  for (const Dims& d : dims) {
    CAPTURE(d.m);
    CAPTURE(d.k);
    CAPTURE(d.n);
    const Nvfp4Weight w = MakeNvfp4Weight(d.n, d.k, seed);
    const auto act_f = RandomF32(static_cast<size_t>(d.m * d.k), seed + 1);
    const auto act_bf16 = ToBf16(act_f);
    const std::vector<float> ref = ReferenceMatmul(w, act_bf16, d.m, d.k, d.n);

    // f32 output: products are bit-identical to the reference (same bf16 act,
    // same bf16 weight), only the K-reduction order differs — matmul tolerance.
    {
      QueueGuard gq(gpu);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {d.m, d.k}, act_bf16.data());
      DeviceTensor dpacked(gpu, gq.q, DType::kI8, {d.n, d.k / 2}, w.packed.data());
      DeviceTensor dscale(gpu, gq.q, DType::kI8, {d.n, d.k / 16}, w.scale.data());
      DeviceTensor dout(gpu, gq.q, DType::kF32, {d.m, d.n});
      vt::MatmulNvfp4(gq.q, dout.tensor(), dact.tensor(), dpacked.tensor(), dscale.tensor(),
                      w.scale2);
      std::vector<float> got(static_cast<size_t>(d.m * d.n));
      dout.Download(gq.q, got.data());
      CheckClose(got, ref, 2e-3f, 2e-3f);
    }

    // bf16 output: compare after BF16ToF32; allow one output bf16 ulp on top.
    {
      QueueGuard gq(gpu);
      DeviceTensor dact(gpu, gq.q, DType::kBF16, {d.m, d.k}, act_bf16.data());
      DeviceTensor dpacked(gpu, gq.q, DType::kI8, {d.n, d.k / 2}, w.packed.data());
      DeviceTensor dscale(gpu, gq.q, DType::kI8, {d.n, d.k / 16}, w.scale.data());
      DeviceTensor dout(gpu, gq.q, DType::kBF16, {d.m, d.n});
      vt::MatmulNvfp4(gq.q, dout.tensor(), dact.tensor(), dpacked.tensor(), dscale.tensor(),
                      w.scale2);
      std::vector<uint16_t> got_bf16(static_cast<size_t>(d.m * d.n));
      dout.Download(gq.q, got_bf16.data());
      std::vector<float> got(got_bf16.size());
      for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(got_bf16[i]);
      CheckClose(got, ref, 4e-3f, 8e-3f);
    }
    seed += 100;
  }
}

TEST_CASE("CUDA matmul_nvfp4: f32 activations also match the reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t m = 4, k = 128, n = 9;
  const Nvfp4Weight w = MakeNvfp4Weight(n, k, 7000);
  const auto act_f = RandomF32(static_cast<size_t>(m * k), 7001);
  const std::vector<float> ref = ReferenceMatmulF32Act(w, act_f, m, k, n);

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kF32, {m, k}, act_f.data());
  DeviceTensor dpacked(gpu, gq.q, DType::kI8, {n, k / 2}, w.packed.data());
  DeviceTensor dscale(gpu, gq.q, DType::kI8, {n, k / 16}, w.scale.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {m, n});
  vt::MatmulNvfp4(gq.q, dout.tensor(), dact.tensor(), dpacked.tensor(), dscale.tensor(), w.scale2);
  std::vector<float> got(static_cast<size_t>(m * n));
  dout.Download(gq.q, got.data());
  // Same weight bytes + f32 accumulation on both sides; only reduction order
  // differs — matmul tolerance.
  CheckClose(got, ref, 2e-3f, 2e-3f);
}

TEST_CASE("matmul_nvfp4 validates shapes loudly (CPU dispatch)") {
  // Shape validation happens in the public op before device dispatch, so this
  // runs everywhere. A weight_packed inner dim != K/2 must throw.
  std::vector<uint16_t> act(4 * 16, 0);
  std::vector<uint8_t> packed(3 * 4, 0);  // wrong: inner should be 16/2 = 8
  std::vector<uint8_t> scale(3 * 1, 0);
  std::vector<float> out(4 * 3, 0.0f);
  Tensor tact = MakeTensor(act.data(), DType::kBF16, Cpu(), {4, 16});
  Tensor tpacked = MakeTensor(packed.data(), DType::kI8, Cpu(), {3, 4});
  Tensor tscale = MakeTensor(scale.data(), DType::kI8, Cpu(), {3, 1});
  Tensor tout = MakeTensor(out.data(), DType::kF32, Cpu(), {4, 3});
  Queue cq{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::MatmulNvfp4(cq, tout, tact, tpacked, tscale, 1.0f), std::runtime_error);
}
