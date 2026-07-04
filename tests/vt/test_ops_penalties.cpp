// Ported from: vllm/model_executor/layers/utils.py + vllm/_custom_ops.py +
// vllm/v1/sample/ops/bad_words.py + vllm/v1/sample/logits_processor/builtin.py
// @ e24d1b24. vt-op-level tests for the M1.7 Task 3 penalty / mask / builtin
// primitives: CPU hand cases plus CPU-vs-CUDA parity. The CUDA kernels mirror
// the CPU reference element for element, so masked positions / kept values must
// match exactly (softmax to 1e-5). Guarded by HasCuda (dgx-pending on CPU-only
// CI). Harness mirrors test_ops_sample.cpp.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

Tensor T(void* d, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = d;
  t.dtype = dt;
  t.device = Cpu();
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

// ---------------------------------------------------------------------------
// vt::ApplyPenalties — the fused elementwise formula on pre-built masks/counts.
TEST_CASE("vt::ApplyPenalties: repetition sign branch + freq + presence") {
  std::vector<float> logits = {2.0f, -2.0f, 1.0f, -1.0f};
  std::vector<int8_t> pmask = {1, 1, 0, 0};   // prompt {0,1}
  std::vector<int8_t> omask = {0, 0, 1, 0};   // output {2}
  std::vector<int32_t> ocnt = {0, 0, 2, 0};   // token2 twice
  std::vector<float> freq = {0.5f}, pres = {1.0f}, rep = {2.0f};
  Queue q = Q();
  Tensor tl = T(logits.data(), DType::kF32, {1, 4});
  Tensor tpm = T(pmask.data(), DType::kI8, {1, 4});
  Tensor tom = T(omask.data(), DType::kI8, {1, 4});
  Tensor toc = T(ocnt.data(), DType::kI32, {1, 4});
  Tensor tf = T(freq.data(), DType::kF32, {1});
  Tensor tp = T(pres.data(), DType::kF32, {1});
  Tensor tr = T(rep.data(), DType::kF32, {1});
  vt::ApplyPenalties(q, tl, tpm, toc, tom, tf, tp, tr);
  CHECK(logits[0] == doctest::Approx(1.0f));
  CHECK(logits[1] == doctest::Approx(-4.0f));
  CHECK(logits[2] == doctest::Approx(-1.5f));
  CHECK(logits[3] == doctest::Approx(-1.0f));
}

TEST_CASE("vt::ApplyTokenMask: scatters -inf at (row, col) pairs") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2 x 3
  std::vector<int32_t> rows = {0, 1};
  std::vector<int32_t> cols = {2, 0};
  Queue q = Q();
  Tensor tl = T(logits.data(), DType::kF32, {2, 3});
  Tensor tr = T(rows.data(), DType::kI32, {2});
  Tensor tc = T(cols.data(), DType::kI32, {2});
  vt::ApplyTokenMask(q, tl, tr, tc);
  CHECK(logits[2] == kNegInf);  // (0,2)
  CHECK(logits[3] == kNegInf);  // (1,0)
  CHECK(logits[0] == doctest::Approx(1.0f));
}

TEST_CASE("vt::ApplyAllowedTokenIds: masked_fill where mask != 0") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int8_t> mask = {1, 0, 0, 1};
  Queue q = Q();
  Tensor tl = T(logits.data(), DType::kF32, {1, 4});
  Tensor tm = T(mask.data(), DType::kI8, {1, 4});
  vt::ApplyAllowedTokenIds(q, tl, tm);
  CHECK(logits[0] == kNegInf);
  CHECK(logits[1] == doctest::Approx(2.0f));
  CHECK(logits[3] == kNegInf);
}

// ===========================================================================
// CPU-vs-CUDA parity (dgx-pending on CPU-only CI). Mirrors test_ops_sample.cpp.
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

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeT(p_, dt, Gpu(), shape);
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

std::vector<float> RandomLogits(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed ? seed : 1;
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = (static_cast<float>(s) / static_cast<float>(UINT32_MAX)) * 10.0f - 5.0f;
  }
  return v;
}

}  // namespace

TEST_CASE("CUDA ApplyPenalties / ApplyMinP / scatter ops match CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping (dgx-pending)");
    return;
  }
  const int64_t N = 5, V = 300;
  const auto logits = RandomLogits(static_cast<size_t>(N * V), 7);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  Queue cq = Q();

  // Build deterministic masks / counts / penalty vectors.
  std::vector<int8_t> pmask(static_cast<size_t>(N * V), 0), omask(static_cast<size_t>(N * V), 0);
  std::vector<int32_t> ocnt(static_cast<size_t>(N * V), 0);
  std::vector<float> freq(static_cast<size_t>(N)), pres(static_cast<size_t>(N)),
      rep(static_cast<size_t>(N)), minp(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < V; ++j) {
      if ((i + j) % 7 == 0) pmask[static_cast<size_t>(i * V + j)] = 1;
      if ((i * 3 + j) % 5 == 0) {
        omask[static_cast<size_t>(i * V + j)] = 1;
        ocnt[static_cast<size_t>(i * V + j)] = static_cast<int32_t>(1 + (j % 3));
      }
    }
    freq[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    pres[static_cast<size_t>(i)] = 0.2f;
    rep[static_cast<size_t>(i)] = 1.0f + 0.25f * static_cast<float>(i);
    minp[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i);  // row 0 = 0 (no-op)
  }

  // ApplyPenalties parity.
  {
    std::vector<float> lc = logits;
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tpm = MakeT(pmask.data(), DType::kI8, Cpu(), {N, V});
    Tensor tom = MakeT(omask.data(), DType::kI8, Cpu(), {N, V});
    Tensor toc = MakeT(ocnt.data(), DType::kI32, Cpu(), {N, V});
    Tensor tf = MakeT(freq.data(), DType::kF32, Cpu(), {N});
    Tensor tp = MakeT(pres.data(), DType::kF32, Cpu(), {N});
    Tensor tr = MakeT(rep.data(), DType::kF32, Cpu(), {N});
    vt::ApplyPenalties(cq, tl, tpm, toc, tom, tf, tp, tr);

    DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    DeviceTensor dpm(gpu, gq.q, DType::kI8, {N, V}, pmask.data());
    DeviceTensor dom(gpu, gq.q, DType::kI8, {N, V}, omask.data());
    DeviceTensor doc(gpu, gq.q, DType::kI32, {N, V}, ocnt.data());
    DeviceTensor df(gpu, gq.q, DType::kF32, {N}, freq.data());
    DeviceTensor dp(gpu, gq.q, DType::kF32, {N}, pres.data());
    DeviceTensor dr(gpu, gq.q, DType::kF32, {N}, rep.data());
    vt::ApplyPenalties(gq.q, dl.tensor(), dpm.tensor(), doc.tensor(), dom.tensor(), df.tensor(),
                       dp.tensor(), dr.tensor());
    std::vector<float> lg(lc.size());
    dl.Download(gq.q, lg.data());
    for (size_t i = 0; i < lc.size(); ++i)
      CHECK(lg[i] == doctest::Approx(lc[i]).epsilon(1e-5));
  }

  // ApplyMinP parity (masked positions must match; kept values exact).
  {
    std::vector<float> lc = logits;
    Tensor tl = MakeT(lc.data(), DType::kF32, Cpu(), {N, V});
    Tensor tmp = MakeT(minp.data(), DType::kF32, Cpu(), {N});
    vt::ApplyMinP(cq, tl, tmp);
    DeviceTensor dl(gpu, gq.q, DType::kF32, {N, V}, logits.data());
    DeviceTensor dmp(gpu, gq.q, DType::kF32, {N}, minp.data());
    vt::ApplyMinP(gq.q, dl.tensor(), dmp.tensor());
    std::vector<float> lg(lc.size());
    dl.Download(gq.q, lg.data());
    for (size_t i = 0; i < lc.size(); ++i) {
      if (std::isinf(lc[i])) {
        CHECK(std::isinf(lg[i]));
      } else {
        CHECK(lg[i] == doctest::Approx(lc[i]).epsilon(1e-5));
      }
    }
  }
}
