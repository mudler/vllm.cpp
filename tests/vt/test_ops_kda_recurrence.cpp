// KDA per-K-channel-decay gated-delta recurrence (vt::KdaGatedDeltaRule) — UNIT GATE.
//
// The genuinely-net-new-vs-GDN device primitive for Kimi-Linear-48B: plain GDN
// decays the [Dv,Dk] recurrent state by ONE scalar per value head
// (`b_h *= exp(b_g)`); KDA decays it PER-K-CHANNEL (`b_h *= exp(b_gk[None,:])`).
// Ported 1:1 from FLA fused_recurrent_gated_delta_rule_fwd_kernel IS_KDA=True
// (third_party/flash_linear_attention/ops/fused_recurrent.py:88-175, wrapped by
// ops/kda.py:109-146 fused_recurrent_kda @ pin 555967922).
//
// ─── WHY THESE ARE THE CORRECTNESS EVIDENCE ────────────────────────────────
// (1) EQUIVALENCE gate: with g broadcast from a per-head scalar, the per-channel
//     op MUST reduce BIT-FOR-BIT to the landed+gated vt::GdnPrefill (Qwen3.6
//     27B/35B production kernel) — the KDA op is GDN's recurrence with a decay
//     VECTOR that happens to be constant. This ties the net-new op to a proven
//     reference in the degenerate case with zero new numerics.
// (2) PER-CHANNEL gate: with distinct per-channel g the op is checked against a
//     from-first-principles f64 reference (the exact island recurrence,
//     kimi_linear_device.cpp KdaRecurrenceIsland) at a documented f32 tolerance —
//     this is the ONLY place the per-channel column-wise decay is exercised.
// (3) CPU<->CUDA parity: the CUDA scan kernel matches the CPU kernel (f32 both).
#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <cstdlib>
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
using vt::GdnArgs;
using vt::Queue;
using vt::Tensor;

namespace {
// Restores the previous value at scope exit, so a pinned flag cannot leak into
// a later case in the same binary (KERNEL-GDN-CHUNKED-MIRROR T9/R10).
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) { had_old_ = true; old_ = old; }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_) setenv(name_.c_str(), old_.c_str(), 1);
    else unsetenv(name_.c_str());
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};


Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

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

std::vector<float> RandF32(size_t n, uint32_t seed, float lo = -1.5f, float hi = 1.5f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// L2-normalize each [dim] row over its last axis (sum, not mean; eps 1e-6) — the
// caller's preprocessing contract (upstream USE_QK_L2NORM_IN_KERNEL, ops/l2norm.py).
std::vector<float> L2NormRows(const std::vector<float>& x, size_t rows, size_t dim) {
  std::vector<float> y(x.size());
  for (size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (size_t d = 0; d < dim; ++d) ss += static_cast<double>(x[r * dim + d]) * x[r * dim + d];
    const double inv = 1.0 / std::sqrt(ss + 1e-6);
    for (size_t d = 0; d < dim; ++d) y[r * dim + d] = static_cast<float>(x[r * dim + d] * inv);
  }
  return y;
}

// From-first-principles f64 reference for the per-channel recurrence, single
// sequence, fresh zero state. MIRRORS kimi_linear_device.cpp KdaRecurrenceIsland
// exactly: S[hv][vd][k] *= exp(g[t,hv,k]); v'=(v - S@k)*beta; S += outer(v',k);
// out = S @ (q*scale).  q/k already l2-normalized by the caller.
std::vector<double> KdaRefF64(const std::vector<float>& qn, const std::vector<float>& kn,
                              const std::vector<float>& v, const std::vector<float>& g,
                              const std::vector<float>& beta, int64_t T, int64_t H, int64_t D,
                              double scale) {
  const int64_t proj = H * D;
  std::vector<double> S(static_cast<size_t>(H) * D * D, 0.0);
  std::vector<double> out(static_cast<size_t>(T) * proj, 0.0);
  std::vector<double> u(static_cast<size_t>(D));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t base = t * proj + h * D;
      const float* qp = &qn[static_cast<size_t>(base)];
      const float* kp = &kn[static_cast<size_t>(base)];
      const float* vp = &v[static_cast<size_t>(base)];
      const float* gp = &g[static_cast<size_t>(base)];
      const double b = beta[static_cast<size_t>(t * H + h)];
      double* Sp = &S[static_cast<size_t>(h) * D * D];
      for (int64_t vd = 0; vd < D; ++vd)
        for (int64_t k = 0; k < D; ++k) Sp[vd * D + k] *= std::exp(static_cast<double>(gp[k]));
      for (int64_t vd = 0; vd < D; ++vd) {
        double dot = 0.0;
        for (int64_t k = 0; k < D; ++k) dot += Sp[vd * D + k] * kp[k];
        u[static_cast<size_t>(vd)] = (static_cast<double>(vp[vd]) - dot) * b;
      }
      for (int64_t vd = 0; vd < D; ++vd)
        for (int64_t k = 0; k < D; ++k) Sp[vd * D + k] += u[static_cast<size_t>(vd)] * kp[k];
      for (int64_t vd = 0; vd < D; ++vd) {
        double o = 0.0;
        for (int64_t k = 0; k < D; ++k) o += Sp[vd * D + k] * (qp[k] * scale);
        out[static_cast<size_t>(base + vd)] = o;
      }
    }
  }
  return out;
}

void CheckCloseF64(const std::vector<float>& got, const std::vector<double>& want, float atol,
                   float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(static_cast<double>(got[i]) - want[i]) <= tol)) {
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

}  // namespace

// ── (1) EQUIVALENCE: broadcast per-channel g == the landed GdnPrefill ─────────
TEST_CASE("kda recurrence: g broadcast from per-head scalar == vt::GdnPrefill (bit-identical)") {
  const int64_t T = 5, H = 3, D = 8;  // Hk==Hv==H (KDA has no GQA)
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 1), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 2), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto v = RandF32(static_cast<size_t>(T) * proj, 3);
  auto beta = RandF32(static_cast<size_t>(T) * H, 4, 0.1f, 0.9f);
  // per-head log-decay in (-1, 0] like real gates (g = -exp(A_log)*softplus(.) < 0)
  auto ghead = RandF32(static_cast<size_t>(T) * H, 5, -1.0f, -0.01f);
  // broadcast ghead[t,h] across the D k-channels
  std::vector<float> gchan(static_cast<size_t>(T) * proj);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t d = 0; d < D; ++d)
        gchan[static_cast<size_t>((t * H + h) * D + d)] = ghead[static_cast<size_t>(t * H + h)];

  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const float scale = 0.35355339f;
  Queue cq = CpuQ();

  std::vector<float> out_kda(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<float> st_kda(static_cast<size_t>(H) * D * D, 0.0f);
  {
    Tensor to = MakeT(out_kda.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tq = MakeT(q.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tk = MakeT(k.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tv = MakeT(v.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tg = MakeT(gchan.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
    Tensor ts = MakeT(st_kda.data(), DType::kF32, Cpu(), {1, H, D, D});
    Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
    vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{scale});
  }

  std::vector<float> out_gdn(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<float> st_gdn(static_cast<size_t>(H) * D * D, 0.0f);
  {
    Tensor to = MakeT(out_gdn.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tq = MakeT(q.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tk = MakeT(k.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tv = MakeT(v.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tg = MakeT(ghead.data(), DType::kF32, Cpu(), {T, H});
    Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
    Tensor ts = MakeT(st_gdn.data(), DType::kF32, Cpu(), {1, H, D, D});
    Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
    vt::GdnPrefill(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{scale});
  }

  // Bit-identical: the per-channel decay vector is constant == the scalar decay,
  // so the fused f32 arithmetic is the same op stream (exact float equality).
  size_t out_diff = 0, st_diff = 0;
  for (size_t i = 0; i < out_kda.size(); ++i)
    if (out_kda[i] != out_gdn[i]) ++out_diff;
  for (size_t i = 0; i < st_kda.size(); ++i)
    if (st_kda[i] != st_gdn[i]) ++st_diff;
  CHECK(out_diff == 0);
  CHECK(st_diff == 0);
}

// ── (1b) THE ARM THAT CLAIM HOLDS ON — KERNEL-GDN-CHUNKED-MIRROR T7/G6 ────────
// The case above is entirely f32 (every tensor in it is kF32), so D0's dtype
// predicate leaves it green after this row: at f32 vt::GdnPrefill still runs the
// sequential recurrence, which is what vt::KdaGatedDeltaRule reduces to. That
// green is TRUE, and it is also NARROWER than it reads. Left alone it would be a
// true assertion standing in for one that stopped being true, which is worse
// than a red.
//
// The spec's resolution, taken here rather than in the test: the KDA row's claim
// is about the RECURRENCE, not about which evaluation order our GDN default
// happens to take, so the reduction is restated as holding on the SEQUENTIAL
// arm — and this case pins the other half explicitly. At bf16, vt::GdnPrefill
// takes vLLM's chunked decomposition while vt::KdaGatedDeltaRule has no chunked
// arm at all, so the two MUST differ. Asserting that is what stops the narrowing
// from being silent. The `== 0` above is NOT weakened to a tolerance
// (stop condition 6); it is scoped, and this is the scope.
TEST_CASE("kda recurrence: at bf16 vt::GdnPrefill takes the chunked arm and KDA does not") {
  // PINNED ON: this case is about what the CHUNKED arm does, so it must not
  // depend on the ambient VT_GDN_CHUNKED.
  ScopedEnv chunked_on("VT_GDN_CHUNKED", "1");
  const int64_t T = 70, H = 2, D = 64;  // > one chunk, so the state carry runs
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 31), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 32), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto v = RandF32(static_cast<size_t>(T) * proj, 33);
  auto beta = RandF32(static_cast<size_t>(T) * H, 34, 0.1f, 0.9f);
  auto ghead = RandF32(static_cast<size_t>(T) * H, 35, -1.0f, -0.01f);
  std::vector<float> gchan(static_cast<size_t>(T) * proj);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t d = 0; d < D; ++d)
        gchan[static_cast<size_t>((t * H + h) * D + d)] = ghead[static_cast<size_t>(t * H + h)];
  std::vector<uint16_t> qb(q.size()), kb(k.size()), vb(v.size());
  for (size_t i = 0; i < q.size(); ++i) qb[i] = vt::F32ToBF16(q[i]);
  for (size_t i = 0; i < k.size(); ++i) kb[i] = vt::F32ToBF16(k[i]);
  for (size_t i = 0; i < v.size(); ++i) vb[i] = vt::F32ToBF16(v[i]);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const float scale = 0.125f;
  Queue cq = CpuQ();

  std::vector<uint16_t> o_kda(static_cast<size_t>(T) * proj, 0), o_gdn(o_kda.size(), 0);
  std::vector<float> st_kda(static_cast<size_t>(H) * D * D, 0.0f), st_gdn(st_kda.size(), 0.0f);
  Tensor tq = MakeT(qb.data(), DType::kBF16, Cpu(), {T, H, D});
  Tensor tk = MakeT(kb.data(), DType::kBF16, Cpu(), {T, H, D});
  Tensor tv = MakeT(vb.data(), DType::kBF16, Cpu(), {T, H, D});
  Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
  Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
  {
    Tensor to = MakeT(o_kda.data(), DType::kBF16, Cpu(), {T, H, D});
    Tensor tg = MakeT(gchan.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor ts = MakeT(st_kda.data(), DType::kF32, Cpu(), {1, H, D, D});
    vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{scale});
  }
  {
    Tensor to = MakeT(o_gdn.data(), DType::kBF16, Cpu(), {T, H, D});
    Tensor tg = MakeT(ghead.data(), DType::kF32, Cpu(), {T, H});
    Tensor ts = MakeT(st_gdn.data(), DType::kF32, Cpu(), {1, H, D, D});
    vt::GdnPrefill(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{scale});
  }
  size_t out_diff = 0;
  double max_st = 0.0;
  for (size_t i = 0; i < o_kda.size(); ++i) out_diff += o_kda[i] != o_gdn[i] ? 1 : 0;
  for (size_t i = 0; i < st_kda.size(); ++i)
    max_st = std::max(max_st, std::abs(static_cast<double>(st_kda[i]) - st_gdn[i]));
  MESSAGE("bf16: KDA(sequential) vs GdnPrefill(chunked) out elements differing: "
          << out_diff << " / " << o_kda.size() << ", state max|d|=" << max_st);
  // They are two evaluation orders of ONE recurrence, so they must differ...
  CHECK(out_diff > 0);
  CHECK(max_st > 0.0);
  // ...and still agree to the chunk-vs-sequential scale, not arbitrarily. If
  // this half fails, the two are not computing the same recurrence any more.
  CHECK(max_st < 5e-2);
}

// ── (2) PER-CHANNEL: distinct per-channel decay vs the f64 island reference ───
TEST_CASE("kda recurrence: distinct per-channel decay matches the f64 reference") {
  const int64_t T = 8, H = 4, D = 16;
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 11), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 12), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto v = RandF32(static_cast<size_t>(T) * proj, 13);
  auto beta = RandF32(static_cast<size_t>(T) * H, 14, 0.1f, 0.9f);
  auto gchan = RandF32(static_cast<size_t>(T) * proj, 15, -0.8f, -0.01f);  // distinct per channel
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const double scale = std::pow(static_cast<double>(D), -0.5);

  std::vector<float> out(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<float> st(static_cast<size_t>(H) * D * D, 0.0f);
  Queue cq = CpuQ();
  Tensor to = MakeT(out.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tq = MakeT(q.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tk = MakeT(k.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tv = MakeT(v.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tg = MakeT(gchan.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
  Tensor ts = MakeT(st.data(), DType::kF32, Cpu(), {1, H, D, D});
  Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
  vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{static_cast<float>(scale)});

  const std::vector<double> ref = KdaRefF64(q, k, v, gchan, beta, T, H, D, scale);
  // f32 recurrence over 8 tokens vs f64 reference: documented tolerance.
  CheckCloseF64(out, ref, /*atol=*/1e-4f, /*rtol=*/3e-3f);
}

// ── (3) validation: g must be per-channel [T,Hv,Dk], scale must be set ────────
TEST_CASE("kda recurrence: validation rejects per-head g and unset scale") {
  const int64_t T = 2, H = 1, D = 2;
  std::vector<float> buf(static_cast<size_t>(T) * H * D, 0.1f);
  std::vector<float> beta(static_cast<size_t>(T) * H, 0.5f);
  std::vector<float> st(static_cast<size_t>(H) * D * D, 0.0f);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  Queue cq = CpuQ();
  Tensor to = MakeT(buf.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tq = MakeT(buf.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tk = MakeT(buf.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tv = MakeT(buf.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
  Tensor ts = MakeT(st.data(), DType::kF32, Cpu(), {1, H, D, D});
  Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
  // per-head g [T,H] is rank-2 -> rejected (kda wants [T,H,D])
  std::vector<float> ghead(static_cast<size_t>(T) * H, -0.1f);
  Tensor tg_head = MakeT(ghead.data(), DType::kF32, Cpu(), {T, H});
  CHECK_THROWS(vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg_head, tb, ts, tqsl, GdnArgs{0.5f}));
  // unset scale
  Tensor tg = MakeT(buf.data(), DType::kF32, Cpu(), {T, H, D});
  CHECK_THROWS(vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{}));
}

// ── (4) CPU<->CUDA parity ─────────────────────────────────────────────────────
#ifdef VLLM_CPP_CUDA
namespace {
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}
struct QGuard {
  Backend& b;
  Queue q;
  explicit QGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QGuard() { b.DestroyQueue(q); }
  QGuard(const QGuard&) = delete;
  QGuard& operator=(const QGuard&) = delete;
};
class DTensor {
 public:
  DTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
          const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~DTensor() { b_.Free(p_); }
  DTensor(const DTensor&) = delete;
  DTensor& operator=(const DTensor&) = delete;
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
}  // namespace

TEST_CASE("kda recurrence: CUDA scan matches the CPU kernel (f32)") {
  if (!HasCuda()) return;
  const int64_t T = 7, H = 4, D = 32;
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 21), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 22), static_cast<size_t>(T) * H,
                      static_cast<size_t>(D));
  auto v = RandF32(static_cast<size_t>(T) * proj, 23);
  auto beta = RandF32(static_cast<size_t>(T) * H, 24, 0.1f, 0.9f);
  auto gchan = RandF32(static_cast<size_t>(T) * proj, 25, -0.8f, -0.01f);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const float scale = std::pow(static_cast<float>(D), -0.5f);

  // CPU
  std::vector<float> out_cpu(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<float> st_cpu(static_cast<size_t>(H) * D * D, 0.0f);
  {
    Queue cq = CpuQ();
    Tensor to = MakeT(out_cpu.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tq = MakeT(q.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tk = MakeT(k.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tv = MakeT(v.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tg = MakeT(gchan.data(), DType::kF32, Cpu(), {T, H, D});
    Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});
    Tensor ts = MakeT(st_cpu.data(), DType::kF32, Cpu(), {1, H, D, D});
    Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
    vt::KdaGatedDeltaRule(cq, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{scale});
  }

  // CUDA
  std::vector<float> out_gpu(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<float> st_gpu(static_cast<size_t>(H) * D * D, 0.0f);
  {
    Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
    QGuard g(gpu);
    std::vector<float> st_zero(static_cast<size_t>(H) * D * D, 0.0f);
    DTensor to(gpu, g.q, DType::kF32, {T, H, D});
    DTensor tq(gpu, g.q, DType::kF32, {T, H, D}, q.data());
    DTensor tk(gpu, g.q, DType::kF32, {T, H, D}, k.data());
    DTensor tv(gpu, g.q, DType::kF32, {T, H, D}, v.data());
    DTensor tg(gpu, g.q, DType::kF32, {T, H, D}, gchan.data());
    DTensor tb(gpu, g.q, DType::kF32, {T, H}, beta.data());
    DTensor ts(gpu, g.q, DType::kF32, {1, H, D, D}, st_zero.data());
    DTensor tqsl(gpu, g.q, DType::kI32, {2}, qsl);
    vt::KdaGatedDeltaRule(g.q, to.tensor(), tq.tensor(), tk.tensor(), tv.tensor(), tg.tensor(),
                          tb.tensor(), ts.tensor(), tqsl.tensor(), GdnArgs{scale});
    to.Download(g.q, out_gpu.data());
    ts.Download(g.q, st_gpu.data());
  }

  // Same f32 math; different exp libm / FMA contraction -> tiny arithmetic-order gap.
  size_t bad = 0;
  for (size_t i = 0; i < out_cpu.size(); ++i)
    if (std::fabs(out_gpu[i] - out_cpu[i]) > 1e-4f + 2e-3f * std::fabs(out_cpu[i])) ++bad;
  CHECK(bad == 0);
  size_t bad_s = 0;
  for (size_t i = 0; i < st_cpu.size(); ++i)
    if (std::fabs(st_gpu[i] - st_cpu[i]) > 1e-4f + 2e-3f * std::fabs(st_cpu[i])) ++bad_s;
  CHECK(bad_s == 0);
}
#endif  // VLLM_CPP_CUDA
