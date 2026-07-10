// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// MoE CPU reference op unit tests. Formula reference: .agents/specs/moe-semantics.md
// (§3 router softmax/top-k/renormalize, §4 expert MLP + weighted combine, §5
// shared-expert sigmoid gate, §6 combine order). Golden coverage lives in
// tests/parity/test_op_parity.cpp; these tests pin hand-computed values and the
// corners the tie-free random goldens do not cover (exact-tie tie-break,
// renormalize on/off, top-8-of-256 shape, f32 accumulation, sigmoid-on-output).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::DeviceType;
using vt::Device;
using vt::DType;
using vt::MoeRouterTopKArgs;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor F32_2(std::vector<float>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b});
}
Tensor F32_3(std::vector<float>& v, int64_t a, int64_t b, int64_t c) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b, c});
}
Tensor I32_2(std::vector<int32_t>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kI32, Cpu(), {a, b});
}
Tensor Bf16_3(std::vector<uint16_t>& v, int64_t a, int64_t b, int64_t c) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {a, b, c});
}
}  // namespace

// ---------------------------------------------------------------------------
// Router tie-break: exact-equal logits, lowest expert index wins
// (moe-semantics.md §3, .cu:530-537 "We want lower indices to win"). The random
// goldens are tie-free by construction, so this rule is pinned only here.
//
//   E=4, logits=[2,2,1,1], top_k=2, renormalize=true.
//   p = softmax([2,2,1,1]): p0=p1 (equal maxes), p2=p3.
//   greedy round 0: strict-`>` scan picks idx 0 (first max).
//   greedy round 1: idx 0 taken -> picks idx 1 (next equal max).
//   ids = [0,1]; renorm: w = p0/(p0+p1) = 0.5 each.
TEST_CASE("moe router: exact-tie tie-break picks lowest index, renorm halves") {
  std::vector<float> logits = {2, 2, 1, 1};
  std::vector<float> w(2, 0.0f);
  std::vector<int32_t> ids(2, -1);
  Tensor tl = F32_2(logits, 1, 4), tw = F32_2(w, 1, 2);
  Tensor ti = I32_2(ids, 1, 2);
  Queue q = Q();
  vt::MoeRouterTopK(q, tw, ti, tl, MoeRouterTopKArgs{2, true});
  CHECK(ids[0] == 0);
  CHECK(ids[1] == 1);
  CHECK(w[0] == doctest::Approx(0.5f).epsilon(1e-6));
  CHECK(w[1] == doctest::Approx(0.5f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// Renormalize on/off from the softmax probabilities (moe-semantics.md §3).
//
//   E=3, logits=[1,0,0], top_k=2. e = exp(1) = 2.7182818.
//   denom = e + 1 + 1 = 4.7182818.
//   p = [e/denom, 1/denom, 1/denom] = [0.5761169, 0.2119416, 0.2119416].
//   top-2: idx0 (max), then idx1 (tie with idx2 -> lowest index) => ids=[0,1].
//   renormalize=false: weights = [0.5761169, 0.2119416].
//   renormalize=true : s = 0.5761169+0.2119416 = 0.7880585,
//                      weights = [0.7310586, 0.2689414]  (= sigmoid(+/-1)).
TEST_CASE("moe router: renormalize on/off matches hand-computed softmax") {
  std::vector<float> logits = {1, 0, 0};
  Queue q = Q();

  std::vector<float> w_off(2, 0.0f);
  std::vector<int32_t> id_off(2, -1);
  Tensor tl = F32_2(logits, 1, 3), tw_off = F32_2(w_off, 1, 2);
  Tensor ti_off = I32_2(id_off, 1, 2);
  vt::MoeRouterTopK(q, tw_off, ti_off, tl, MoeRouterTopKArgs{2, false});
  CHECK(id_off[0] == 0);
  CHECK(id_off[1] == 1);
  CHECK(w_off[0] == doctest::Approx(0.5761169f).epsilon(1e-5));
  CHECK(w_off[1] == doctest::Approx(0.2119416f).epsilon(1e-5));

  std::vector<float> w_on(2, 0.0f);
  std::vector<int32_t> id_on(2, -1);
  Tensor tw_on = F32_2(w_on, 1, 2);
  Tensor ti_on = I32_2(id_on, 1, 2);
  vt::MoeRouterTopK(q, tw_on, ti_on, tl, MoeRouterTopKArgs{2, true});
  CHECK(id_on[0] == 0);
  CHECK(id_on[1] == 1);
  CHECK(w_on[0] == doctest::Approx(0.7310586f).epsilon(1e-5));
  CHECK(w_on[1] == doctest::Approx(0.2689414f).epsilon(1e-5));
  CHECK(w_on[0] + w_on[1] == doctest::Approx(1.0f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// Real-ratio routing shape: top-8 of 256 (moe-semantics.md §1 Qwen3.6 config).
// Distinct descending logits -> ids must be the 8 highest indices in order and
// weights must be sorted descending and (renorm) sum to 1.
TEST_CASE("moe router: top-8 of 256 shape, descending, sums to one") {
  const int64_t E = 256, K = 8, T = 3;
  std::vector<float> logits(static_cast<size_t>(T * E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t e = 0; e < E; ++e)
      logits[static_cast<size_t>(t * E + e)] = static_cast<float>(e) * 0.01f + static_cast<float>(t);
  std::vector<float> w(static_cast<size_t>(T * K), 0.0f);
  std::vector<int32_t> ids(static_cast<size_t>(T * K), -1);
  Tensor tl = F32_2(logits, T, E), tw = F32_2(w, T, K);
  Tensor ti = I32_2(ids, T, K);
  Queue q = Q();
  vt::MoeRouterTopK(q, tw, ti, tl, MoeRouterTopKArgs{static_cast<int>(K), true});
  for (int64_t t = 0; t < T; ++t) {
    float sum = 0.0f;
    for (int64_t j = 0; j < K; ++j) {
      // logits increase with e, so the 8 highest are E-1..E-8 in order.
      CHECK(ids[static_cast<size_t>(t * K + j)] == static_cast<int32_t>(E - 1 - j));
      if (j > 0)
        CHECK(w[static_cast<size_t>(t * K + j)] <= w[static_cast<size_t>(t * K + j - 1)] + 1e-6f);
      sum += w[static_cast<size_t>(t * K + j)];
    }
    CHECK(sum == doctest::Approx(1.0f).epsilon(1e-6));
  }
}

// ---------------------------------------------------------------------------
// Weighted combine, f32 accumulation over bf16 expert outputs
// (moe-semantics.md §4/§6). Two experts, H=2, no shared term.
//   expert_out (bf16): slot0 = [1.0, 2.0], slot1 = [4.0, 8.0]
//   weights (f32):     [0.25, 0.5]
//   out = 0.25*[1,2] + 0.5*[4,8] = [2.25, 4.5]  (accumulated in f32)
TEST_CASE("moe combine: f32-accumulated weighted sum over bf16 expert outputs") {
  std::vector<uint16_t> eo = {vt::F32ToBF16(1.0f), vt::F32ToBF16(2.0f), vt::F32ToBF16(4.0f),
                              vt::F32ToBF16(8.0f)};
  std::vector<float> w = {0.25f, 0.5f};
  std::vector<float> out(2, 0.0f);
  Tensor teo = Bf16_3(eo, 1, 2, 2);
  Tensor tw = F32_2(w, 1, 2);
  Tensor to = F32_2(out, 1, 2);
  Queue q = Q();
  vt::MoeCombine(q, to, teo, tw, nullptr);
  CHECK(out[0] == doctest::Approx(2.25f).epsilon(1e-6));
  CHECK(out[1] == doctest::Approx(4.5f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// Combine order: shared term is ADDED to the routed sum (moe-semantics.md §6,
// moe_runner.py:717 shared_output + fused_output). The shared-expert sigmoid
// gate (§5) is applied to the MLP OUTPUT, not the input — the test supplies a
// pre-gated shared vector and checks it passes through unchanged.
//   expert_out slot0 = [10, 20], weight = 1.0
//   shared = sigmoid(0) * [4, 8] = 0.5*[4,8] = [2, 4]   (gate logit 0 -> 0.5)
//   out = [10,20] + [2,4] = [12, 24]
TEST_CASE("moe combine: shared term (sigmoid-gated output) added after routed") {
  std::vector<float> eo = {10.0f, 20.0f};
  std::vector<float> w = {1.0f};
  const float gate = 1.0f / (1.0f + std::exp(-0.0f));  // sigmoid(0) = 0.5
  std::vector<float> shared = {gate * 4.0f, gate * 8.0f};
  std::vector<float> out(2, 0.0f);
  Tensor teo = F32_3(eo, 1, 1, 2);
  Tensor tw = F32_2(w, 1, 1);
  Tensor tsh = F32_2(shared, 1, 2);
  Tensor to = F32_2(out, 1, 2);
  Queue q = Q();
  vt::MoeCombine(q, to, teo, tw, &tsh);
  CHECK(out[0] == doctest::Approx(12.0f).epsilon(1e-6));
  CHECK(out[1] == doctest::Approx(24.0f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// Tiny end-to-end block composed from the §-cited ops (router -> per-expert
// silu-mul MLP via Matmul+SiluAndMul+Matmul -> MoeCombine), one expert selected,
// no shared term. Hand table (moe-semantics.md §3/§4):
//
//   E=2, H=2, I=2, top_k=1, renorm=true.  x = [1, 0].
//   gate_w = [[1,0],[0,1]]  -> logits = [1, 0] -> softmax [0.7311,0.2689]
//     top-1 -> id=0, weight (renorm, k=1) = 1.0.
//   expert 0: w13[0]=[[1,0],[0,1],[1,0],[0,1]] (gate rows 0-1, up rows 2-3)
//     h1 = x @ w13[0].T = [1, 0, 1, 0]   (gate=[1,0], up=[1,0])
//     a  = silu(gate)*up = [silu(1)*1, silu(0)*0] = [0.7310586, 0]
//     w2[0] = [[1,0],[0,1]] -> y = a @ w2[0].T = [0.7310586, 0]
//   routed = 1.0 * y = [0.7310586, 0].  (expert 1 not selected)
TEST_CASE("moe block: tiny hand-computed router -> MLP -> combine") {
  Queue q = Q();
  // Router.
  std::vector<float> x = {1.0f, 0.0f};
  std::vector<float> gate_wT = {1, 0, 0, 1};  // gate_w.T ([H,E]) = identity
  std::vector<float> logits(2, 0.0f);
  Tensor tx = F32_2(x, 1, 2), tgwT = F32_2(gate_wT, 2, 2), tlog = F32_2(logits, 1, 2);
  vt::Matmul(q, tlog, tx, tgwT);
  std::vector<float> w(1, 0.0f);
  std::vector<int32_t> ids(1, -1);
  Tensor tw = F32_2(w, 1, 1), ti = I32_2(ids, 1, 1);
  vt::MoeRouterTopK(q, tw, ti, tlog, MoeRouterTopKArgs{1, true});
  CHECK(ids[0] == 0);
  CHECK(w[0] == doctest::Approx(1.0f).epsilon(1e-6));

  // Expert 0 MLP (w13[0].T is [H,2I], w2[0].T is [I,H]).
  std::vector<float> w13T = {1, 0, 1, 0, 0, 1, 0, 1};  // [H=2, 2I=4]
  std::vector<float> w2T = {1, 0, 0, 1};               // [I=2, H=2]
  std::vector<float> h1(4, 0.0f), a(2, 0.0f), y(2, 0.0f);
  Tensor tw13T = F32_2(w13T, 2, 4), th1 = F32_2(h1, 1, 4);
  vt::Matmul(q, th1, tx, tw13T);
  Tensor ta = F32_2(a, 1, 2);
  vt::SiluAndMul(q, ta, th1);
  Tensor tw2T = F32_2(w2T, 2, 2), ty = F32_2(y, 1, 2);
  vt::Matmul(q, ty, ta, tw2T);

  // Combine (single slot).
  std::vector<float> eo = {y[0], y[1]};
  std::vector<float> out(2, 0.0f);
  Tensor teo = F32_3(eo, 1, 1, 2), to = F32_2(out, 1, 2);
  vt::MoeCombine(q, to, teo, tw, nullptr);
  CHECK(out[0] == doctest::Approx(0.7310586f).epsilon(1e-5));
  CHECK(out[1] == doctest::Approx(0.0f).epsilon(1e-6));
}

// ===========================================================================
// CUDA parity: the CUDA kernels must match the CPU reference bit-for-bit on the
// tie-break (indices EXACT) and to 1e-5 on the f32 weights/outputs. Guarded by
// HasCuda so the CPU-only build skips these. Harness mirrors test_ops_gdn.cpp.
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

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint8_t> Pack(const std::vector<float>& f, DType dt) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), f.data(), out.size());
  } else {
    REQUIRE(dt == DType::kBF16);
    auto* p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < f.size(); ++i) p[i] = vt::F32ToBF16(f[i]);
  }
  return out;
}

std::vector<float> UnpackF32(const std::vector<uint8_t>& b) {
  std::vector<float> out(b.size() / sizeof(float));
  std::memcpy(out.data(), b.data(), b.size());
  return out;
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

// Runs moe_router_topk CPU-vs-CUDA on one logits matrix. Weights toleranced to
// 1e-5 (f32), ids required EXACT (the lowest-index tie-break must match).
void RunRouterCudaCase(const std::vector<float>& logits_f32, int64_t t, int64_t e, int k,
                       bool renorm, DType logit_dt) {
  const auto lb = Pack(logits_f32, logit_dt);
  const MoeRouterTopKArgs args{k, renorm};

  std::vector<float> w_cpu(static_cast<size_t>(t * k), 0.0f);
  std::vector<int32_t> id_cpu(static_cast<size_t>(t * k), -1);
  Tensor tl = MakeT(const_cast<uint8_t*>(lb.data()), logit_dt, Cpu(), {t, e});
  Tensor tw = MakeT(w_cpu.data(), DType::kF32, Cpu(), {t, k});
  Tensor ti = MakeT(id_cpu.data(), DType::kI32, Cpu(), {t, k});
  Queue cq = Q();
  vt::MoeRouterTopK(cq, tw, ti, tl, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dl(gpu, gq.q, logit_dt, {t, e}, lb.data());
  DeviceTensor dw(gpu, gq.q, DType::kF32, {t, k});
  DeviceTensor did(gpu, gq.q, DType::kI32, {t, k});
  vt::MoeRouterTopK(gq.q, dw.tensor(), did.tensor(), dl.tensor(), args);
  std::vector<float> w_gpu(w_cpu.size());
  std::vector<int32_t> id_gpu(id_cpu.size());
  dw.Download(gq.q, w_gpu.data());
  did.Download(gq.q, id_gpu.data());

  for (size_t i = 0; i < w_cpu.size(); ++i) {
    CHECK(id_gpu[i] == id_cpu[i]);  // tie-break must match bit-for-bit
    CHECK(w_gpu[i] == doctest::Approx(w_cpu[i]).epsilon(1e-5));
  }
}

// Runs moe_combine CPU-vs-CUDA. Expert outputs are the same bytes on both sides
// (identical f32 accumulation), so out compares at 1e-5.
void RunCombineCudaCase(int64_t t, int64_t h, int k, DType eo_dt, bool with_shared,
                        uint32_t seed) {
  const auto eof = RandomF32(static_cast<size_t>(t * k * h), seed);
  const auto wf = RandomF32(static_cast<size_t>(t * k), seed + 1, 0.0f, 1.0f);
  const auto shf = RandomF32(static_cast<size_t>(t * h), seed + 2);
  const auto eob = Pack(eof, eo_dt);
  const auto shb = Pack(shf, eo_dt);

  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * h) * sizeof(float));
  std::vector<float> w = wf;
  Tensor teo = MakeT(const_cast<uint8_t*>(eob.data()), eo_dt, Cpu(), {t, k, h});
  Tensor tw = MakeT(w.data(), DType::kF32, Cpu(), {t, k});
  Tensor tsh = MakeT(const_cast<uint8_t*>(shb.data()), eo_dt, Cpu(), {t, h});
  Tensor to = MakeT(out_cpu.data(), DType::kF32, Cpu(), {t, h});
  Queue cq = Q();
  vt::MoeCombine(cq, to, teo, tw, with_shared ? &tsh : nullptr);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor deo(gpu, gq.q, eo_dt, {t, k, h}, eob.data());
  DeviceTensor dw(gpu, gq.q, DType::kF32, {t, k}, wf.data());
  DeviceTensor dsh(gpu, gq.q, eo_dt, {t, h}, shb.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {t, h});
  vt::MoeCombine(gq.q, dout.tensor(), deo.tensor(), dw.tensor(),
                 with_shared ? &dsh.tensor() : nullptr);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  dout.Download(gq.q, out_gpu.data());

  const auto oc = UnpackF32(out_cpu), og = UnpackF32(out_gpu);
  for (size_t i = 0; i < oc.size(); ++i)
    CHECK(og[i] == doctest::Approx(oc[i]).epsilon(1e-5));
}

}  // namespace

TEST_CASE("CUDA moe router: top-8 of 256 matches CPU (f32 and bf16 logits)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const int64_t T = 5, E = 256;
  const int K = 8;
  const auto logits = RandomF32(static_cast<size_t>(T * E), 7000, -4.0f, 4.0f);
  for (DType dt : {DType::kF32, DType::kBF16}) {
    for (bool renorm : {false, true}) {
      CAPTURE(static_cast<int>(dt));
      CAPTURE(renorm);
      RunRouterCudaCase(logits, T, E, K, renorm, dt);
    }
  }
}

TEST_CASE("CUDA moe router: exact-tie tie-break picks lowest index on GPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // All-equal logits over E=256: the greedy top-8 must pick indices 0..7 in
  // order on the GPU, exactly as the CPU reference does.
  const int64_t T = 2, E = 256;
  const int K = 8;
  std::vector<float> logits(static_cast<size_t>(T * E), 1.5f);
  RunRouterCudaCase(logits, T, E, K, /*renorm=*/true, DType::kF32);
  RunRouterCudaCase(logits, T, E, K, /*renorm=*/false, DType::kF32);

  // Partial tie: a block of equal maxes then descending — winners are the tied
  // block in ascending index order, then the next-highest.
  std::vector<float> mixed(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) mixed[static_cast<size_t>(e)] = e < 4 ? 3.0f : -0.01f * static_cast<float>(e);
  RunRouterCudaCase(mixed, 1, E, K, /*renorm=*/true, DType::kF32);
}

TEST_CASE("CUDA moe combine: weighted sum (+shared) matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  for (DType eo_dt : {DType::kF32, DType::kBF16}) {
    for (bool shared : {false, true}) {
      CAPTURE(static_cast<int>(eo_dt));
      CAPTURE(shared);
      RunCombineCudaCase(4, 2048, 8, eo_dt, shared, 7100);   // real H, top-8
      RunCombineCudaCase(3, 17, 2, eo_dt, shared, 7150);     // small odd H
    }
  }
}
