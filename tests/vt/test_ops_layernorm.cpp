// Unit tests for the cross-family dense primitives added by the OPT
// (`OPTForCausalLM`) bring-up: vt::LayerNorm, vt::Relu, vt::Add.
//
// Upstream spec ported FROM:
//   * LayerNorm — torch `nn.LayerNorm` / ATen `native_layer_norm`, which
//     `vllm/model_executor/models/opt.py:146-148,164-166,248-251` construct with
//     the default eps=1e-5. The reference below is the documented formula
//     `y = (x - E[x]) / sqrt(Var[x] + eps) * w + b` with the BIASED (1/N)
//     variance torch uses.
//   * Relu — `vllm/model_executor/layers/activation.py::get_act_fn("relu")`
//     (opt.py:156); upstream kernel-test analog `tests/kernels/core/
//     test_activation.py`.
//   * Add — `torch.add`, in the two shapes OPT needs (elementwise residual join
//     opt.py:178,191,279; rank-1 `nn.Linear` bias row-broadcast opt.py:90-104).
//
// CPU is the primary gate (runs in CI). The CUDA section runs on dgx and skips
// cleanly when no GPU backend is registered; it checks the CUDA kernels against
// the SAME f32 reference within the bf16/f32 rounding tolerance.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::LayerNormArgs;
using vt::Queue;
using vt::Tensor;

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

std::vector<float> RandF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// The torch `nn.LayerNorm` reference: biased variance, f32 accumulation, affine.
std::vector<float> RefLayerNorm(const std::vector<float>& x, int64_t rows, int64_t d,
                                const std::vector<float>* w, const std::vector<float>* b,
                                float eps) {
  std::vector<float> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t base = r * d;
    float sum = 0.0f;
    for (int64_t i = 0; i < d; ++i) sum += x[static_cast<size_t>(base + i)];
    const float mean = sum / static_cast<float>(d);
    float sq = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
      const float dv = x[static_cast<size_t>(base + i)] - mean;
      sq += dv * dv;
    }
    const float rstd = 1.0f / std::sqrt(sq / static_cast<float>(d) + eps);
    for (int64_t i = 0; i < d; ++i) {
      float v = (x[static_cast<size_t>(base + i)] - mean) * rstd;
      if (w != nullptr) v *= (*w)[static_cast<size_t>(i)];
      if (b != nullptr) v += (*b)[static_cast<size_t>(i)];
      out[static_cast<size_t>(base + i)] = v;
    }
  }
  return out;
}

Backend* MaybeCuda() {
  try {
    return &vt::GetBackend(DeviceType::kCUDA);
  } catch (const std::exception&) {
    return nullptr;
  }
}

}  // namespace

TEST_CASE("layer_norm matches the torch nn.LayerNorm reference (f32, affine)") {
  constexpr int64_t kRows = 7;
  constexpr int64_t kD = 768;  // OPT-125m hidden_size
  const float eps = 1e-5f;
  std::vector<float> x = RandF32(kRows * kD, 11);
  std::vector<float> w = RandF32(kD, 22, 0.5f, 1.5f);
  std::vector<float> b = RandF32(kD, 33, -0.5f, 0.5f);
  std::vector<float> out(x.size(), 0.0f);

  Queue q;
  q.device = Cpu();
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {kRows, kD});
  Tensor tw = MakeTensor(w.data(), DType::kF32, Cpu(), {kD});
  Tensor tb = MakeTensor(b.data(), DType::kF32, Cpu(), {kD});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {kRows, kD});
  vt::LayerNorm(q, to, tx, &tw, &tb, LayerNormArgs{eps});

  const std::vector<float> ref = RefLayerNorm(x, kRows, kD, &w, &b, eps);
  for (size_t i = 0; i < ref.size(); ++i) CHECK(out[i] == doctest::Approx(ref[i]).epsilon(1e-5));

  // Sanity: a normalized row (before affine) has zero mean and unit variance.
  std::vector<float> plain(x.size(), 0.0f);
  Tensor tp = MakeTensor(plain.data(), DType::kF32, Cpu(), {kRows, kD});
  vt::LayerNorm(q, tp, tx, nullptr, nullptr, LayerNormArgs{eps});
  for (int64_t r = 0; r < kRows; ++r) {
    double m = 0.0;
    double v = 0.0;
    for (int64_t i = 0; i < kD; ++i) m += plain[static_cast<size_t>(r * kD + i)];
    m /= static_cast<double>(kD);
    for (int64_t i = 0; i < kD; ++i) {
      const double dv = plain[static_cast<size_t>(r * kD + i)] - m;
      v += dv * dv;
    }
    v /= static_cast<double>(kD);
    CHECK(m == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(v == doctest::Approx(1.0).epsilon(1e-3));
  }
}

TEST_CASE("layer_norm bf16 in/out rounds once on store") {
  constexpr int64_t kRows = 5;
  constexpr int64_t kD = 128;
  const float eps = 1e-5f;
  const std::vector<float> xf = RandF32(kRows * kD, 44);
  const std::vector<float> wf = RandF32(kD, 55, 0.5f, 1.5f);
  const std::vector<float> bf = RandF32(kD, 66, -0.5f, 0.5f);

  // Round the inputs to bf16 first so the reference sees EXACTLY what the kernel
  // reads back (otherwise the comparison would fold in the input rounding).
  std::vector<uint16_t> x16(xf.size());
  std::vector<uint16_t> w16(wf.size());
  std::vector<uint16_t> b16(bf.size());
  std::vector<float> xr(xf.size());
  std::vector<float> wr(wf.size());
  std::vector<float> br(bf.size());
  for (size_t i = 0; i < xf.size(); ++i) {
    x16[i] = vt::F32ToBF16(xf[i]);
    xr[i] = vt::BF16ToF32(x16[i]);
  }
  for (size_t i = 0; i < wf.size(); ++i) {
    w16[i] = vt::F32ToBF16(wf[i]);
    wr[i] = vt::BF16ToF32(w16[i]);
    b16[i] = vt::F32ToBF16(bf[i]);
    br[i] = vt::BF16ToF32(b16[i]);
  }
  std::vector<uint16_t> out16(xf.size(), 0);

  Queue q;
  q.device = Cpu();
  Tensor tx = MakeTensor(x16.data(), DType::kBF16, Cpu(), {kRows, kD});
  Tensor tw = MakeTensor(w16.data(), DType::kBF16, Cpu(), {kD});
  Tensor tb = MakeTensor(b16.data(), DType::kBF16, Cpu(), {kD});
  Tensor to = MakeTensor(out16.data(), DType::kBF16, Cpu(), {kRows, kD});
  vt::LayerNorm(q, to, tx, &tw, &tb, LayerNormArgs{eps});

  // The kernel reads bf16, computes in f32 and rounds ONCE on store, so the
  // result must land within one bf16 ULP (~2^-8 relative) of the f32 reference
  // computed on the same rounded inputs.
  const std::vector<float> ref = RefLayerNorm(xr, kRows, kD, &wr, &br, eps);
  for (size_t i = 0; i < ref.size(); ++i) {
    const float got = vt::BF16ToF32(out16[i]);
    const float tol = 0.008f * std::fabs(ref[i]) + 1e-3f;
    CHECK(std::fabs(got - ref[i]) <= tol);
  }
}

TEST_CASE("relu clamps negatives and preserves positives (f32 + bf16, in place)") {
  const std::vector<float> x = RandF32(1024, 77);
  std::vector<float> out(x.size(), 0.0f);
  Queue q;
  q.device = Cpu();
  Tensor tx = MakeTensor(const_cast<float*>(x.data()), DType::kF32, Cpu(), {8, 128});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {8, 128});
  vt::Relu(q, to, tx);
  for (size_t i = 0; i < x.size(); ++i) CHECK(out[i] == (x[i] > 0.0f ? x[i] : 0.0f));

  // In-place (out aliases x) — the form the OPT MLP uses.
  std::vector<float> inplace = x;
  Tensor ti = MakeTensor(inplace.data(), DType::kF32, Cpu(), {8, 128});
  vt::Relu(q, ti, ti);
  for (size_t i = 0; i < x.size(); ++i) CHECK(inplace[i] == (x[i] > 0.0f ? x[i] : 0.0f));
}

TEST_CASE("add: elementwise and rank-1 bias row-broadcast") {
  constexpr int64_t kRows = 6;
  constexpr int64_t kD = 64;
  const std::vector<float> a = RandF32(kRows * kD, 88);
  const std::vector<float> b = RandF32(kRows * kD, 99);
  const std::vector<float> bias = RandF32(kD, 111);
  std::vector<float> out(a.size(), 0.0f);

  Queue q;
  q.device = Cpu();
  Tensor ta = MakeTensor(const_cast<float*>(a.data()), DType::kF32, Cpu(), {kRows, kD});
  Tensor tb = MakeTensor(const_cast<float*>(b.data()), DType::kF32, Cpu(), {kRows, kD});
  Tensor tbias = MakeTensor(const_cast<float*>(bias.data()), DType::kF32, Cpu(), {kD});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {kRows, kD});

  vt::Add(q, to, ta, tb);
  for (size_t i = 0; i < a.size(); ++i) CHECK(out[i] == doctest::Approx(a[i] + b[i]));

  vt::Add(q, to, ta, tbias);
  for (int64_t r = 0; r < kRows; ++r)
    for (int64_t c = 0; c < kD; ++c) {
      const size_t i = static_cast<size_t>(r * kD + c);
      CHECK(out[i] == doctest::Approx(a[i] + bias[static_cast<size_t>(c)]));
    }

  // In-place bias add — exactly what the biased OPT projections do.
  std::vector<float> inplace = a;
  Tensor ti = MakeTensor(inplace.data(), DType::kF32, Cpu(), {kRows, kD});
  vt::Add(q, ti, ti, tbias);
  for (int64_t r = 0; r < kRows; ++r)
    for (int64_t c = 0; c < kD; ++c) {
      const size_t i = static_cast<size_t>(r * kD + c);
      CHECK(inplace[i] == doctest::Approx(a[i] + bias[static_cast<size_t>(c)]));
    }
}

TEST_CASE("layer_norm/relu/add on CUDA match the CPU reference") {
  Backend* cuda = MaybeCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  constexpr int64_t kRows = 9;
  constexpr int64_t kD = 768;
  const float eps = 1e-5f;
  const std::vector<float> x = RandF32(kRows * kD, 123);
  const std::vector<float> w = RandF32(kD, 234, 0.5f, 1.5f);
  const std::vector<float> b = RandF32(kD, 345, -0.5f, 0.5f);

  Queue q = cuda->CreateQueue();
  auto upload = [&](const std::vector<float>& h) {
    void* p = cuda->Alloc(h.size() * sizeof(float));
    cuda->Copy(q, p, h.data(), h.size() * sizeof(float));
    return p;
  };
  void* dx = upload(x);
  void* dw = upload(w);
  void* db = upload(b);
  void* dout = cuda->Alloc(x.size() * sizeof(float));

  Device dev{DeviceType::kCUDA, 0};
  Tensor tx = MakeTensor(dx, DType::kF32, dev, {kRows, kD});
  Tensor tw = MakeTensor(dw, DType::kF32, dev, {kD});
  Tensor tb = MakeTensor(db, DType::kF32, dev, {kD});
  Tensor to = MakeTensor(dout, DType::kF32, dev, {kRows, kD});

  vt::LayerNorm(q, to, tx, &tw, &tb, LayerNormArgs{eps});
  std::vector<float> got(x.size());
  cuda->Copy(q, got.data(), dout, got.size() * sizeof(float));
  cuda->Synchronize(q);
  const std::vector<float> ref = RefLayerNorm(x, kRows, kD, &w, &b, eps);
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));

  vt::Relu(q, to, tx);
  cuda->Copy(q, got.data(), dout, got.size() * sizeof(float));
  cuda->Synchronize(q);
  for (size_t i = 0; i < x.size(); ++i) CHECK(got[i] == doctest::Approx(x[i] > 0 ? x[i] : 0.0f));

  vt::Add(q, to, tx, tw);  // rank-1 [kD] bias broadcast over kRows rows
  cuda->Copy(q, got.data(), dout, got.size() * sizeof(float));
  cuda->Synchronize(q);
  for (int64_t r = 0; r < kRows; ++r)
    for (int64_t c = 0; c < kD; ++c) {
      const size_t i = static_cast<size_t>(r * kD + c);
      CHECK(got[i] == doctest::Approx(x[i] + w[static_cast<size_t>(c)]));
    }

  cuda->Free(dx);
  cuda->Free(dw);
  cuda->Free(db);
  cuda->Free(dout);
}
