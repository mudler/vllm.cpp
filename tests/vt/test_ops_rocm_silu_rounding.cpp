// BACKEND-ROCM silu-gate dtype rounding repair (#2889 silu item, #1954).
//
// The ROCm SiluAndMul (dense, packed [T,2D]) and MoeSiluMul (MoE, separate
// gate/up) kernels compute silu(gate) in f32 and multiply by up WITHOUT first
// narrowing the silu result to the gate tensor's dtype. The CPU oracle
// (cpu_ops.cpp:669,733) narrows via RoundThrough(in_dt, ...) before the
// multiply, and upstream vLLM's silu_kernel (activation_kernels.cu:158) casts
// the intermediate to T before compute multiplies. On exact-equality checks
// the bf16 arm diverges.
//
// This test runs both ops on the ROCm backend and compares against the CPU
// oracle (run through the same vt:: entry points) with EXACT equality on every
// dtype arm: raw uint16 bits for bf16, exact f32 for f32. The f32 path is
// NarrowTo<float> = identity, so it is bit-identical to before; the bf16 path
// is the one the repair fixes.
//
// Self-skipping without a ROCm device, mirroring test_rocm_backend.cpp's guard.
// Linked into a test binary only in a HIP build (tests/CMakeLists.txt gates it
// on VLLM_CPP_HIP) but COMPILES everywhere as a bit-rot guard.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

bool NoDevice() { return !vt::rocm::DeviceAvailable(); }

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

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
    t_ = MakeTensor(p_, dt, q.device, shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void* ptr() { return p_; }
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

std::vector<float> RandomVec(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint16_t> F32VecToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}

}  // namespace

// ── Dense SiluAndMul: out[T,D] = silu(x[:,:D]) * x[:,D:] ────────────────────

// f32 exactness is NOT the contract here: the GPU evaluates silu with the
// device libdevice expf and the CPU oracle with the host expf, and those two
// implementations differ by a few ULPs on arbitrary inputs. The repair's
// contract is the NARROWING: bf16/f16 must be bit-exact (the narrow dtype
// absorbs expf ULP differences), and f32 must agree within a small ULP band.
// 4 ULPs on the silu product covers expf's documented cross-implementation
// spread with margin; a real association or formula divergence reds it.
int UlpsApart(float a, float b) {
  if (a == b) return 0;
  if (std::isnan(a) || std::isnan(b)) return 1 << 20;
  int32_t ia, ib;
  std::memcpy(&ia, &a, 4);
  std::memcpy(&ib, &b, 4);
  const int32_t bias = 0x80000000;
  const int32_t sa = ia < 0 ? bias - ia : ia;
  const int32_t sb = ib < 0 ? bias - ib : ib;
  return sa > sb ? sa - sb : sb - sa;
}

TEST_CASE("ROCm SiluAndMul gate-dtype narrowing matches CPU oracle exactly") {
  if (NoDevice()) return;

  // Shapes: small + a real-width row.
  struct Shape { int64_t T, D; };
  const Shape shapes[] = {{1, 4}, {3, 128}, {2, 1024}};

  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  QueueGuard cqg(cpu), rqg(rocm);

  for (const auto& sh : shapes) {
    const int64_t T = sh.T, D = sh.D;
    const size_t xn = static_cast<size_t>(T * 2 * D);
    const size_t on = static_cast<size_t>(T * D);
    const std::vector<float> x_f = RandomVec(xn, 1001 + static_cast<uint32_t>(T * D));
    const std::vector<uint16_t> x_bf = F32VecToBf16(x_f);

    for (bool bf16 : {false, true}) {
      CAPTURE(bf16);
      CAPTURE(T);
      CAPTURE(D);

      // CPU oracle
      std::vector<float> ref_f(on, 0.0f);
      std::vector<uint16_t> ref_b(on, 0);
      if (bf16) {
        std::vector<uint16_t> cx = x_bf;
        Tensor tx = MakeTensor(cx.data(), DType::kBF16, Cpu(), {T, 2 * D});
        Tensor to = MakeTensor(ref_b.data(), DType::kBF16, Cpu(), {T, D});
        vt::SiluAndMul(cqg.q, to, tx);
      } else {
        std::vector<float> cx = x_f;
        Tensor tx = MakeTensor(cx.data(), DType::kF32, Cpu(), {T, 2 * D});
        Tensor to = MakeTensor(ref_f.data(), DType::kF32, Cpu(), {T, D});
        vt::SiluAndMul(cqg.q, to, tx);
      }

      // ROCm
      const DType dt = bf16 ? DType::kBF16 : DType::kF32;
      DeviceTensor dx(rocm, rqg.q, dt, {T, 2 * D}, bf16 ? static_cast<const void*>(x_bf.data())
                                                        : static_cast<const void*>(x_f.data()));
      DeviceTensor dout(rocm, rqg.q, dt, {T, D});
      vt::SiluAndMul(rqg.q, dout.tensor(), dx.tensor());
      rocm.Synchronize(rqg.q);

      if (bf16) {
        std::vector<uint16_t> got(on);
        dout.Download(rqg.q, got.data());
        CHECK(got == ref_b);  // exact: both sides narrow silu to bf16 then multiply
      } else {
        std::vector<float> got(on);
        dout.Download(rqg.q, got.data());
        for (size_t i = 0; i < on; ++i) {
          CAPTURE(i);
          CHECK(UlpsApart(got[i], ref_f[i]) <= 4);  // device vs host expf band
        }
      }
    }
  }
}

// ── GeluAndMul: the same defect, same file, same helper ─────────────────────
// The CPU oracle rounds the GELU intermediate through in_dt before the multiply
// (cpu_ops.cpp:644) exactly as it does the silu one, so GeluMulK needs the same
// narrowing. Without it the bf16 arm multiplies an f32 gelu where the oracle
// multiplies a bf16-rounded one, and this case is red. Bounds are the SiluAndMul
// case's: exact bytes on bf16 (which is what the narrowing buys), a 4-ULP band
// on f32 for the device-vs-host transcendental.
TEST_CASE("ROCm GeluAndMul gate-dtype narrowing matches CPU oracle exactly") {
  if (NoDevice()) return;

  struct Shape { int64_t T, D; };
  const Shape shapes[] = {{1, 4}, {3, 128}, {2, 1024}};

  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  QueueGuard cqg(cpu), rqg(rocm);

  // bf16 ONLY, and that is the point rather than a gap. NarrowTo<float> is the
  // identity, so an f32 arm cannot witness this change at all -- it would be
  // measuring device tanhf against host std::tanh, a property this change does
  // not touch. Measured on gfx1151: an f32 arm held to the SiluAndMul case's
  // 4-ULP band fails ~0.6% of elements with the fix correctly IN PLACE, because
  // the gelu polynomial amplifies that divergence further than silu's expf
  // does. Widening the band to fit would be choosing a number to green an
  // assertion that discriminates nothing here.
  for (const auto& sh : shapes) {
    const int64_t T = sh.T, D = sh.D;
    const size_t xn = static_cast<size_t>(T * 2 * D);
    const size_t on = static_cast<size_t>(T * D);
    CAPTURE(T);
    CAPTURE(D);
    const std::vector<float> x_f = RandomVec(xn, 2003 + static_cast<uint32_t>(T * D));
    const std::vector<uint16_t> x_bf = F32VecToBf16(x_f);

    std::vector<uint16_t> ref_b(on, 0);
    {
      std::vector<uint16_t> cx = x_bf;
      Tensor tx = MakeTensor(cx.data(), DType::kBF16, Cpu(), {T, 2 * D});
      Tensor to = MakeTensor(ref_b.data(), DType::kBF16, Cpu(), {T, D});
      vt::GeluAndMul(cqg.q, to, tx);
    }

    DeviceTensor dx(rocm, rqg.q, DType::kBF16, {T, 2 * D},
                    static_cast<const void*>(x_bf.data()));
    DeviceTensor dout(rocm, rqg.q, DType::kBF16, {T, D});
    vt::GeluAndMul(rqg.q, dout.tensor(), dx.tensor());
    rocm.Synchronize(rqg.q);

    std::vector<uint16_t> got(on);
    dout.Download(rqg.q, got.data());
    CHECK(got == ref_b);  // exact: both sides narrow gelu to bf16 then multiply
  }
}

// ── MoE MoeSiluMul: out[N] = silu(gate[N]) * up[N] ──────────────────────────
TEST_CASE("ROCm MoeSiluMul gate-dtype narrowing matches CPU oracle exactly") {
  if (NoDevice()) return;

  const int64_t Ns[] = {8, 512, 4096};

  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  QueueGuard cqg(cpu), rqg(rocm);

  for (int64_t N : Ns) {
    const size_t n = static_cast<size_t>(N);
    const std::vector<float> gate_f = RandomVec(n, 2001 + static_cast<uint32_t>(N));
    const std::vector<float> up_f = RandomVec(n, 2002 + static_cast<uint32_t>(N));
    const std::vector<uint16_t> gate_bf = F32VecToBf16(gate_f);
    const std::vector<uint16_t> up_bf = F32VecToBf16(up_f);

    for (bool bf16 : {false, true}) {
      CAPTURE(bf16);
      CAPTURE(N);

      // CPU oracle
      std::vector<float> ref_f(n, 0.0f);
      std::vector<uint16_t> ref_b(n, 0);
      if (bf16) {
        std::vector<uint16_t> cg = gate_bf, cu = up_bf;
        Tensor tg = MakeTensor(cg.data(), DType::kBF16, Cpu(), {N});
        Tensor tu = MakeTensor(cu.data(), DType::kBF16, Cpu(), {N});
        Tensor to = MakeTensor(ref_b.data(), DType::kBF16, Cpu(), {N});
        vt::MoeSiluMul(cqg.q, to, tg, tu);
      } else {
        std::vector<float> cg = gate_f, cu = up_f;
        Tensor tg = MakeTensor(cg.data(), DType::kF32, Cpu(), {N});
        Tensor tu = MakeTensor(cu.data(), DType::kF32, Cpu(), {N});
        Tensor to = MakeTensor(ref_f.data(), DType::kF32, Cpu(), {N});
        vt::MoeSiluMul(cqg.q, to, tg, tu);
      }

      // ROCm
      const DType dt = bf16 ? DType::kBF16 : DType::kF32;
      DeviceTensor dg(rocm, rqg.q, dt, {N},
                      bf16 ? static_cast<const void*>(gate_bf.data())
                           : static_cast<const void*>(gate_f.data()));
      DeviceTensor du(rocm, rqg.q, dt, {N},
                      bf16 ? static_cast<const void*>(up_bf.data())
                           : static_cast<const void*>(up_f.data()));
      DeviceTensor dout(rocm, rqg.q, dt, {N});
      vt::MoeSiluMul(rqg.q, dout.tensor(), dg.tensor(), du.tensor());
      rocm.Synchronize(rqg.q);

      if (bf16) {
        std::vector<uint16_t> got(n);
        dout.Download(rqg.q, got.data());
        CHECK(got == ref_b);  // exact: both sides narrow silu to bf16 then multiply
      } else {
        std::vector<float> got(n);
        dout.Download(rqg.q, got.data());
        for (size_t i = 0; i < n; ++i) {
          CAPTURE(i);
          CHECK(UlpsApart(got[i], ref_f[i]) <= 4);  // device vs host expf band
        }
      }
    }
  }
}
