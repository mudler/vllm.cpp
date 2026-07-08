// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// A/B token-exact check for the Triton AOT fast-path: the AOT-compiled Triton
// GemmaRMSNorm (triton_kernels/rmsnorm.py, selected by VT_TRITON_RMSNORM=1) must
// match BOTH the CPU reference and the hand CUDA RmsNorm (RmsNormRowKernel with
// gemma=true) bit-close. Because rmsnorm reduces in f32 and Triton's reduction
// order (tl.sum) differs from the hand shared-memory tree reduction, we assert
// max|Δ| within a tight tolerance rather than exact equality.
//
// Layers of gating:
//   * No CUDA backend registered  -> skip (CPU/CI).
//   * CUDA present but built WITHOUT VLLM_CPP_TRITON -> the VT_TRITON_RMSNORM
//     toggle is a no-op, so both GPU runs take the hand path; the test still
//     validates gemma f32 RmsNorm on the GPU, and the "triton == hand" check is
//     trivially exact. The A/B only exercises the Triton cubin when compiled ON.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
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

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {  // catches NaN
      if (bad == 0) first = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first);
    CAPTURE(got[first]);
    CAPTURE(want[first]);
  }
  CHECK(bad == 0);
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

// Runs gemma-RMSNorm(f32) on the GPU with VT_TRITON_RMSNORM forced to `triton`.
std::vector<float> GpuRmsNorm(Backend& gpu, const std::vector<float>& xf,
                              const std::vector<float>& wf, int64_t t, int64_t h,
                              float eps, bool triton) {
  if (triton) {
    setenv("VT_TRITON_RMSNORM", "1", 1);
  } else {
    unsetenv("VT_TRITON_RMSNORM");
  }
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, DType::kF32, {t, h}, xf.data());
  DeviceTensor dw(gpu, gq.q, DType::kF32, {h}, wf.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {t, h});
  vt::RmsNorm(gq.q, dout.tensor(), dx.tensor(), dw.tensor(), RmsNormArgs{eps, /*gemma=*/true},
              nullptr);
  std::vector<float> out(static_cast<size_t>(t * h));
  dout.Download(gq.q, out.data());
  unsetenv("VT_TRITON_RMSNORM");
  return out;
}

}  // namespace

TEST_CASE("Triton AOT RmsNorm (gemma, f32) matches CPU and the hand CUDA kernel") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping Triton AOT A/B");
    return;
  }
#ifdef VLLM_CPP_TRITON
  MESSAGE("built with VLLM_CPP_TRITON: VT_TRITON_RMSNORM=1 exercises the AOT cubin");
#else
  MESSAGE("built WITHOUT VLLM_CPP_TRITON: toggle is a no-op, both runs take the hand path");
#endif
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);

  // Row widths must be <= the kernel's BLOCK_SIZE (4096). Odd sizes exercise the
  // masked reduction tail.
  const int64_t widths[] = {1, 128, 129, 256, 1024, 2048, 4096};
  const int64_t rows = 5;
  const float eps = 1e-6f;
  uint32_t seed = 4242;

  for (int64_t h : widths) {
    CAPTURE(h);
    const auto xf = RandomF32(static_cast<size_t>(rows * h), seed);
    const auto wf = RandomF32(static_cast<size_t>(h), seed + 1);
    seed += 10;

    // CPU reference (gemma, f32, no residual).
    std::vector<float> cpu(static_cast<size_t>(rows * h), 0.0f);
    Tensor tx = MakeTensor(const_cast<float*>(xf.data()), DType::kF32, Cpu(), {rows, h});
    Tensor tw = MakeTensor(const_cast<float*>(wf.data()), DType::kF32, Cpu(), {h});
    Tensor to = MakeTensor(cpu.data(), DType::kF32, Cpu(), {rows, h});
    Queue cq{Cpu(), nullptr};
    vt::RmsNorm(cq, to, tx, tw, RmsNormArgs{eps, /*gemma=*/true}, nullptr);

    const auto hand = GpuRmsNorm(gpu, xf, wf, rows, h, eps, /*triton=*/false);
    const auto triton = GpuRmsNorm(gpu, xf, wf, rows, h, eps, /*triton=*/true);

    // Triton vs CPU: both f32, reduction orders differ -> tight-but-not-exact.
    CheckClose(triton, cpu, /*atol=*/1e-4f, /*rtol=*/1e-4f);
    // Triton vs hand CUDA: two GPU realizations of the same op.
    CheckClose(triton, hand, /*atol=*/1e-4f, /*rtol=*/1e-4f);
    CAPTURE(MaxAbsDiff(triton, hand));
    CAPTURE(MaxAbsDiff(triton, cpu));
  }
}
