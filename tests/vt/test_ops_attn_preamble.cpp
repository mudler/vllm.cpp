// vllm.cpp original (vt runtime). CUDA op-level bit-identity check for the fused
// full-attention preamble (AttnQkNormRopeGate + RopeCosSinCache) vs the unfused
// four-op sequence (AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox). Mirrors
// the composition used by qwen3_5.cpp FullAttnBlockPaged. The fused kernel claims
// byte-identical intermediates for f32 out; this test measures max|diff|.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::RopeArgs;
using vt::Tensor;

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

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class Dev {
 public:
  Dev(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape, const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~Dev() { b_.Free(p_); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  Tensor& t() { return t_; }
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

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t* first = nullptr) {
  REQUIRE(a.size() == b.size());
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > m) {
      m = d;
      if (first != nullptr) *first = i;
    }
  }
  return m;
}

}  // namespace

TEST_CASE("attn preamble fused == unfused (f32, gemma) bit-identity") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  Queue& q = g.q;

  const int64_t T = 8, Hq = 16, Hkv = 2, Dh = 128;
  const int rot = 64;
  const float base = 1.0e7f, eps = 1.0e-6f;

  auto qgate_h = RandomF32(static_cast<size_t>(T * Hq * 2 * Dh), 11);
  auto kf_h = RandomF32(static_cast<size_t>(T * Hkv * Dh), 22);
  auto qn_h = RandomF32(static_cast<size_t>(Dh), 33, -0.5f, 0.5f);
  auto kn_h = RandomF32(static_cast<size_t>(Dh), 44, -0.5f, 0.5f);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos_h[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  Dev qgate(b, q, DType::kF32, {T, Hq * 2 * Dh}, qgate_h.data());
  Dev kf(b, q, DType::kF32, {T, Hkv * Dh}, kf_h.data());
  Dev qn(b, q, DType::kF32, {Dh}, qn_h.data());
  Dev kn(b, q, DType::kF32, {Dh}, kn_h.data());
  Dev pos(b, q, DType::kI32, {T}, pos_h.data());

  // ---- unfused: AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox ----
  Dev qf(b, q, DType::kF32, {T, Hq, Dh});
  Dev gate_u(b, q, DType::kF32, {T, Hq, Dh});
  Dev dq_u(b, q, DType::kF32, {T, Hq, Dh});
  Dev dk_u(b, q, DType::kF32, {T, Hkv, Dh});
  vt::AttnGateSplit(q, qf.t(), gate_u.t(), qgate.t());
  Tensor dqn2d = MakeT(dq_u.t().data, DType::kF32, Gpu(), {T * Hq, Dh});
  Tensor qf2d = MakeT(qf.t().data, DType::kF32, Gpu(), {T * Hq, Dh});
  vt::RmsNorm(q, dqn2d, qf2d, qn.t(), RmsNormArgs{eps, true});
  Tensor dkn2d = MakeT(dk_u.t().data, DType::kF32, Gpu(), {T * Hkv, Dh});
  Tensor kf2d = MakeT(kf.t().data, DType::kF32, Gpu(), {T * Hkv, Dh});
  vt::RmsNorm(q, dkn2d, kf2d, kn.t(), RmsNormArgs{eps, true});
  vt::RopeNeox(q, dq_u.t(), dk_u.t(), pos.t(), RopeArgs{base, rot});

  std::vector<float> qout_u(static_cast<size_t>(T * Hq * Dh));
  std::vector<float> kout_u(static_cast<size_t>(T * Hkv * Dh));
  std::vector<float> gate_uo(static_cast<size_t>(T * Hq * Dh));
  dq_u.Download(q, qout_u.data());
  dk_u.Download(q, kout_u.data());
  gate_u.Download(q, gate_uo.data());

  // ---- fused: RopeCosSinCache + AttnQkNormRopeGate ----
  Dev cos_sin(b, q, DType::kF32, {T, rot});
  vt::RopeCosSinCache(q, cos_sin.t(), pos.t(), RopeArgs{base, rot});
  Dev dq_f(b, q, DType::kF32, {T, Hq, Dh});
  Dev dk_f(b, q, DType::kF32, {T, Hkv, Dh});
  Dev gate_f(b, q, DType::kF32, {T, Hq, Dh});
  Tensor kf_view = MakeT(kf.t().data, DType::kF32, Gpu(), {T, Hkv * Dh});
  vt::AttnQkNormRopeGate(q, dq_f.t(), dk_f.t(), gate_f.t(), qgate.t(), kf_view, qn.t(), kn.t(),
                         cos_sin.t(), RmsNormArgs{eps, true}, RopeArgs{base, rot});

  std::vector<float> qout_f(static_cast<size_t>(T * Hq * Dh));
  std::vector<float> kout_f(static_cast<size_t>(T * Hkv * Dh));
  std::vector<float> gate_fo(static_cast<size_t>(T * Hq * Dh));
  dq_f.Download(q, qout_f.data());
  dk_f.Download(q, kout_f.data());
  gate_f.Download(q, gate_fo.data());

  // Port of QKVParallelLinear's torch.split-style output contract: Q and K
  // rows are inner-contiguous views whose token stride remains Q+K+V. The
  // fused preamble must consume those views without split-copy kernels.
  const int64_t q_width = Hq * 2 * Dh;
  const int64_t k_width = Hkv * Dh;
  const int64_t v_width = Hkv * Dh;
  const int64_t packed_width = q_width + k_width + v_width;
  std::vector<float> packed_h(static_cast<size_t>(T * packed_width), -777.0F);
  for (int64_t tok = 0; tok < T; ++tok) {
    std::memcpy(packed_h.data() + tok * packed_width,
                qgate_h.data() + tok * q_width,
                static_cast<size_t>(q_width) * sizeof(float));
    std::memcpy(packed_h.data() + tok * packed_width + q_width,
                kf_h.data() + tok * k_width,
                static_cast<size_t>(k_width) * sizeof(float));
  }
  Dev packed(b, q, DType::kF32, {T, packed_width}, packed_h.data());
  Tensor packed_q = MakeT(packed.t().data, DType::kF32, Gpu(), {T, q_width});
  packed_q.stride[0] = packed_width;
  auto* packed_k_data = static_cast<float*>(packed.t().data) + q_width;
  Tensor packed_k = MakeT(packed_k_data, DType::kF32, Gpu(), {T, k_width});
  packed_k.stride[0] = packed_width;
  Dev dq_s(b, q, DType::kF32, {T, Hq, Dh});
  Dev dk_s(b, q, DType::kF32, {T, Hkv, Dh});
  Dev gate_s(b, q, DType::kF32, {T, Hq, Dh});
  vt::AttnQkNormRopeGate(q, dq_s.t(), dk_s.t(), gate_s.t(), packed_q,
                         packed_k, qn.t(), kn.t(), cos_sin.t(),
                         RmsNormArgs{eps, true}, RopeArgs{base, rot});
  std::vector<float> qout_s(qout_f.size());
  std::vector<float> kout_s(kout_f.size());
  std::vector<float> gate_so(gate_fo.size());
  dq_s.Download(q, qout_s.data());
  dk_s.Download(q, kout_s.data());
  gate_s.Download(q, gate_so.data());
  CHECK(qout_s == qout_f);
  CHECK(kout_s == kout_f);
  CHECK(gate_so == gate_fo);

  size_t fq = 0, fk = 0, fg = 0;
  const float dq = MaxAbsDiff(qout_u, qout_f, &fq);
  const float dk = MaxAbsDiff(kout_u, kout_f, &fk);
  const float dg = MaxAbsDiff(gate_uo, gate_fo, &fg);
  MESSAGE("max|diff| q=", dq, " (idx ", fq, " u=", qout_u[fq], " f=", qout_f[fq], ")");
  MESSAGE("max|diff| k=", dk, " (idx ", fk, ")");
  MESSAGE("max|diff| gate=", dg, " (idx ", fg, ")");
  // MEASURED (GB10, 2026-07-09): the gate passthrough is BIT-IDENTICAL (pure copy),
  // but the RoPE'd q/k differ by ONE f32 ULP (~2.4e-7 at magnitude ~2). The fused
  // `ni*cs - nih*cs` contracts to a DIFFERENT FMA than the unfused RmsNorm-store +
  // RopeNeox `x*c - y*sn`, so the fusion is NOT bit-identical. On the 35B (fp8) the
  // deterministic greedy decode is 1-ULP-sensitive and DIVERGES within 16 tokens, so
  // VT_FUSE_ATTN_PREAMBLE stays DEFAULT-OFF (it fails the token-exact gate). This
  // test PINS that relationship (gate exact; q/k within 1 ULP) so a future agent
  // does not flip the default on the (false) assumption of bit-identity.
  CHECK(dg == 0.0f);                 // gate: bit-identical
  CHECK(dq <= 3.0e-7f);              // q: within ~1 f32 ULP, but NOT zero
  CHECK(dk <= 3.0e-7f);              // k: within ~1 f32 ULP, but NOT zero
  CHECK(dq > 0.0f);                  // documents the non-bit-identity (the NO-GO cause)
}

// The FA-2 prefill combo: bf16 q_out/k_out + f32 gate_out. After the tightening
// (RoundToStore<Tqk> rounds the normed q/k to the STORE dtype BEFORE the RoPE
// multiply — cuda_ops.cu, mirroring vLLM fused_qk_norm_rope.py:67,89-90), the
// fused bf16 preamble is BIT-IDENTICAL to the UNFUSED bf16 four-op sequence that
// FullAttnBlockPaged's `else` branch runs on the FA-2 path: AttnGateSplit +
// RmsNorm(->bf16) + RmsNorm(->bf16) + RopeNeox(bf16). Both round the normed q/k
// to bf16 and then feed RoPE the bf16 value. The f32 gate stays the exact
// passthrough. This pins the numerics FA-2 prefill actually runs on the 27B AND
// the 35B (2026-07-18 flip) — the razor-tie the old test documented is removed.
TEST_CASE("attn preamble fused-bf16 == unfused-bf16 (FA-2 prefill, bit-identity)") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  Queue& q = g.q;

  const int64_t T = 8, Hq = 16, Hkv = 2, Dh = 256;  // the 35B gate head shape (ratio-8)
  const int rot = 64;
  const float base = 1.0e7f, eps = 1.0e-6f;

  auto qgate_h = RandomF32(static_cast<size_t>(T * Hq * 2 * Dh), 55);
  auto kf_h = RandomF32(static_cast<size_t>(T * Hkv * Dh), 66);
  auto qn_h = RandomF32(static_cast<size_t>(Dh), 77, -0.5f, 0.5f);
  auto kn_h = RandomF32(static_cast<size_t>(Dh), 88, -0.5f, 0.5f);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos_h[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  Dev qgate(b, q, DType::kF32, {T, Hq * 2 * Dh}, qgate_h.data());
  Dev kf(b, q, DType::kF32, {T, Hkv * Dh}, kf_h.data());
  Dev qn(b, q, DType::kF32, {Dh}, qn_h.data());
  Dev kn(b, q, DType::kF32, {Dh}, kn_h.data());
  Dev pos(b, q, DType::kI32, {T}, pos_h.data());

  // ---- unfused bf16: AttnGateSplit + RmsNorm(->bf16) + RmsNorm(->bf16) +
  // RopeNeox(bf16) — the exact sequence FullAttnBlockPaged runs when the fused
  // preamble is OFF but the FA-2 attn dtype is bf16 (dq3/dk3 allocated bf16). ----
  Dev qf(b, q, DType::kF32, {T, Hq, Dh});
  Dev gate_u(b, q, DType::kF32, {T, Hq, Dh});
  Dev dq_u(b, q, DType::kBF16, {T, Hq, Dh});
  Dev dk_u(b, q, DType::kBF16, {T, Hkv, Dh});
  vt::AttnGateSplit(q, qf.t(), gate_u.t(), qgate.t());
  Tensor dqn2d = MakeT(dq_u.t().data, DType::kBF16, Gpu(), {T * Hq, Dh});
  Tensor qf2d = MakeT(qf.t().data, DType::kF32, Gpu(), {T * Hq, Dh});
  vt::RmsNorm(q, dqn2d, qf2d, qn.t(), RmsNormArgs{eps, true});
  Tensor dkn2d = MakeT(dk_u.t().data, DType::kBF16, Gpu(), {T * Hkv, Dh});
  Tensor kf2d = MakeT(kf.t().data, DType::kF32, Gpu(), {T * Hkv, Dh});
  vt::RmsNorm(q, dkn2d, kf2d, kn.t(), RmsNormArgs{eps, true});
  vt::RopeNeox(q, dq_u.t(), dk_u.t(), pos.t(), RopeArgs{base, rot});
  std::vector<uint16_t> qu(static_cast<size_t>(T * Hq * Dh));
  std::vector<uint16_t> ku(static_cast<size_t>(T * Hkv * Dh));
  std::vector<float> gu(static_cast<size_t>(T * Hq * Dh));
  dq_u.Download(q, qu.data());
  dk_u.Download(q, ku.data());
  gate_u.Download(q, gu.data());

  // ---- fused bf16: RopeCosSinCache + AttnQkNormRopeGate (bf16 q/k, f32 gate) ----
  Dev cos_sin(b, q, DType::kF32, {T, rot});
  vt::RopeCosSinCache(q, cos_sin.t(), pos.t(), RopeArgs{base, rot});
  Dev dq_b(b, q, DType::kBF16, {T, Hq, Dh});
  Dev dk_b(b, q, DType::kBF16, {T, Hkv, Dh});
  Dev gate_b(b, q, DType::kF32, {T, Hq, Dh});
  Tensor kf_view = MakeT(kf.t().data, DType::kF32, Gpu(), {T, Hkv * Dh});
  vt::AttnQkNormRopeGate(q, dq_b.t(), dk_b.t(), gate_b.t(), qgate.t(), kf_view, qn.t(), kn.t(),
                         cos_sin.t(), RmsNormArgs{eps, true}, RopeArgs{base, rot});
  std::vector<uint16_t> qb(static_cast<size_t>(T * Hq * Dh));
  std::vector<uint16_t> kb(static_cast<size_t>(T * Hkv * Dh));
  std::vector<float> gb(static_cast<size_t>(T * Hq * Dh));
  dq_b.Download(q, qb.data());
  dk_b.Download(q, kb.data());
  gate_b.Download(q, gb.data());

  size_t qmis = 0, kmis = 0;
  for (size_t i = 0; i < qu.size(); ++i)
    if (qb[i] != qu[i]) ++qmis;
  for (size_t i = 0; i < ku.size(); ++i)
    if (kb[i] != ku[i]) ++kmis;
  size_t fg = 0;
  const float dg = MaxAbsDiff(gu, gb, &fg);
  MESSAGE("fused-bf16 vs unfused-bf16 mismatches: q=", qmis, "/", qu.size(), " k=", kmis, "/",
          ku.size(), " gate max|diff|=", dg);
  CHECK(qmis == 0);   // q: fused bf16 == unfused bf16, bit-identical (round-before-RoPE)
  CHECK(kmis == 0);   // k: same
  CHECK(dg == 0.0f);  // gate: exact f32 passthrough
}
