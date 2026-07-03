// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// GDN CPU reference op unit tests. Formula reference: .agents/gdn-semantics.md
// (§2 conv fwd, §3 conv update, §4 l2norm, §5 gated rmsnorm, §7 delta rule).
// Golden coverage lives in tests/parity/test_op_parity.cpp; these tests pin
// hand-computed values and the corners the goldens do not cover (conv bias,
// GQA ratio 3, tiny hand tables).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::CausalConv1dArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::GdnArgs;
using vt::L2NormArgs;
using vt::Queue;
using vt::RmsNormGatedArgs;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor T2(std::vector<float>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b});
}
Tensor T3(std::vector<float>& v, int64_t a, int64_t b, int64_t c) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b, c});
}
Tensor T4(std::vector<float>& v, int64_t a, int64_t b, int64_t c, int64_t d) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b, c, d});
}
Tensor Ti(std::vector<int32_t>& v) {
  return Tensor::Contiguous(v.data(), DType::kI32, Cpu(), {static_cast<int64_t>(v.size())});
}
}  // namespace

// ---------------------------------------------------------------------------
// GdnDecode: hand-computed single step (gdn-semantics.md §7 pseudocode).
// Hk=Hv=1, Dk=Dv=2, scale=0.5. S rows index v, cols index k.
//
//   S0 = [[1, 2], [3, 4]]   q=[1,0] k=[0,1] v=[2,1] g=ln(0.5) beta=0.5
//
//   step              | value
//   ------------------+------------------------------------------
//   decay = exp(g)    | 0.5
//   S *= decay        | [[0.5, 1.0], [1.5, 2.0]]
//   S @ k             | [1.0, 2.0]
//   v' = (v-S@k)*beta | [(2-1)*0.5, (1-2)*0.5] = [0.5, -0.5]
//   S += outer(v', k) | [[0.5, 1.5], [1.5, 1.5]]
//   q' = q * scale    | [0.5, 0.0]
//   o  = S @ q'       | [0.25, 0.75]
//
// All values are exact in binary floating point.
TEST_CASE("gdn decode: hand-computed single step, state updated in place") {
  std::vector<float> qv = {1.0f, 0.0f}, kv = {0.0f, 1.0f}, vv = {2.0f, 1.0f};
  std::vector<float> g = {std::log(0.5f)}, beta = {0.5f};
  std::vector<float> state = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tq = T3(qv, 1, 1, 2), tk = T3(kv, 1, 1, 2), tv = T3(vv, 1, 1, 2);
  Tensor tg = T2(g, 1, 1), tb = T2(beta, 1, 1), ts = T4(state, 1, 1, 2, 2);
  Tensor to = T3(out, 1, 1, 2);
  Queue q = Q();
  vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{0.5f});
  CHECK(out[0] == doctest::Approx(0.25f).epsilon(1e-6));
  CHECK(out[1] == doctest::Approx(0.75f).epsilon(1e-6));
  CHECK(state[0] == doctest::Approx(0.5f).epsilon(1e-6));
  CHECK(state[1] == doctest::Approx(1.5f).epsilon(1e-6));
  CHECK(state[2] == doctest::Approx(1.5f).epsilon(1e-6));
  CHECK(state[3] == doctest::Approx(1.5f).epsilon(1e-6));
}

// GdnPrefill: hand-computed 3-token table (same §7 recurrence, one sequence).
// Hk=Hv=1, Dk=Dv=2, scale=0.5, S0=[[1,2],[3,4]]. Derived step by step from the
// pseudocode (f64 simulation; every value below is exact in binary fp):
//
//   t | q      k        v       g       beta | decay S@k        v'            S after                      o
//   --+--------------------------------------+-------------------------------------------------------------------------
//   0 | [1,0]  [0,1]    [2,1]   0       0.5  | 1.0  [2, 4]      [0, -1.5]     [[1, 2], [3, 2.5]]           [0.5, 1.5]
//   1 | [0,1]  [1,0]    [1,-1]  ln(0.5) 1.0  | 0.5  [0.5, 1.5]  [0.5, -2.5]   [[1, 1], [-1, 1.25]]         [0.5, 0.625]
//   2 | [1,1]  [.5,.5]  [0,2]   0       0.5  | 1.0  [1, 0.125]  [-0.5,0.9375] [[.75,.75],[-.53125,1.71875]] [0.75, 0.59375]
TEST_CASE("gdn prefill: hand-computed 3-token recurrence table") {
  std::vector<float> qv = {1, 0, 0, 1, 1, 1};
  std::vector<float> kv = {0, 1, 1, 0, 0.5f, 0.5f};
  std::vector<float> vv = {2, 1, 1, -1, 0, 2};
  std::vector<float> g = {0.0f, std::log(0.5f), 0.0f};
  std::vector<float> beta = {0.5f, 1.0f, 0.5f};
  std::vector<float> state = {1, 2, 3, 4};
  std::vector<float> out(6, 0.0f);
  std::vector<int32_t> qsl = {0, 3};
  Tensor tq = T3(qv, 3, 1, 2), tk = T3(kv, 3, 1, 2), tv = T3(vv, 3, 1, 2);
  Tensor tg = T2(g, 3, 1), tb = T2(beta, 3, 1), ts = T4(state, 1, 1, 2, 2);
  Tensor to = T3(out, 3, 1, 2), tqsl = Ti(qsl);
  Queue q = Q();
  vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{0.5f});
  const float want_out[6] = {0.5f, 1.5f, 0.5f, 0.625f, 0.75f, 0.59375f};
  const float want_state[4] = {0.75f, 0.75f, -0.53125f, 1.71875f};
  for (int i = 0; i < 6; ++i) CHECK(out[i] == doctest::Approx(want_out[i]).epsilon(1e-6));
  for (int i = 0; i < 4; ++i) CHECK(state[i] == doctest::Approx(want_state[i]).epsilon(1e-6));
}

// GQA ratio 3 (the 27B gate model is Hk16/Hv48; goldens only cover ratio 2).
// Broadcast contract: v-head hv reads q/k head hv / (Hv/Hk) — heads only
// replicate the math, so a Hk=2/Hv=6 decode step must equal six independent
// single-head steps fed the mapped q/k head.
TEST_CASE("gdn decode: GQA ratio 3 broadcast (Hk=2, Hv=6)") {
  constexpr int64_t kHk = 2, kHv = 6, kDk = 2, kDv = 2;
  std::vector<float> qv = {0.6f, -0.8f, 0.1f, 0.9f};        // [1,Hk,Dk]
  std::vector<float> kv = {0.5f, 0.5f, -0.25f, 1.0f};       // [1,Hk,Dk]
  std::vector<float> vv(kHv * kDv), g(kHv), beta(kHv), state(kHv * kDv * kDk);
  for (int64_t h = 0; h < kHv; ++h) {
    vv[h * 2] = 0.5f + 0.25f * static_cast<float>(h);
    vv[h * 2 + 1] = -1.0f + 0.5f * static_cast<float>(h);
    g[h] = -0.1f * static_cast<float>(h + 1);
    beta[h] = 0.1f + 0.1f * static_cast<float>(h);
    for (int64_t e = 0; e < kDv * kDk; ++e)
      state[h * kDv * kDk + e] = 0.125f * static_cast<float>(h) - 0.0625f * static_cast<float>(e);
  }
  std::vector<float> state_full = state, out_full(kHv * kDv, 0.0f);
  Tensor tq = T3(qv, 1, kHk, kDk), tk = T3(kv, 1, kHk, kDk), tv = T3(vv, 1, kHv, kDv);
  Tensor tg = T2(g, 1, kHv), tb = T2(beta, 1, kHv);
  Tensor ts = T4(state_full, 1, kHv, kDv, kDk), to = T3(out_full, 1, kHv, kDv);
  Queue q = Q();
  vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{0.70710678f});
  for (int64_t hv = 0; hv < kHv; ++hv) {
    const int64_t hk = hv / (kHv / kHk);  // ratio 3: v-heads 0-2 -> head 0, 3-5 -> head 1
    std::vector<float> q1(qv.begin() + hk * kDk, qv.begin() + (hk + 1) * kDk);
    std::vector<float> k1(kv.begin() + hk * kDk, kv.begin() + (hk + 1) * kDk);
    std::vector<float> v1(vv.begin() + hv * kDv, vv.begin() + (hv + 1) * kDv);
    std::vector<float> g1 = {g[static_cast<size_t>(hv)]}, b1 = {beta[static_cast<size_t>(hv)]};
    std::vector<float> s1(state.begin() + hv * kDv * kDk, state.begin() + (hv + 1) * kDv * kDk);
    std::vector<float> o1(kDv, 0.0f);
    Tensor tq1 = T3(q1, 1, 1, kDk), tk1 = T3(k1, 1, 1, kDk), tv1 = T3(v1, 1, 1, kDv);
    Tensor tg1 = T2(g1, 1, 1), tb1 = T2(b1, 1, 1), ts1 = T4(s1, 1, 1, kDv, kDk);
    Tensor to1 = T3(o1, 1, 1, kDv);
    vt::GdnDecode(q, to1, tq1, tk1, tv1, tg1, tb1, ts1, GdnArgs{0.70710678f});
    for (int64_t e = 0; e < kDv; ++e)
      CHECK(out_full[static_cast<size_t>(hv * kDv + e)] ==
            doctest::Approx(o1[static_cast<size_t>(e)]).epsilon(1e-6));
    for (int64_t e = 0; e < kDv * kDk; ++e)
      CHECK(state_full[static_cast<size_t>(hv * kDv * kDk + e)] ==
            doctest::Approx(s1[static_cast<size_t>(e)]).epsilon(1e-6));
  }
}

// Varlen contract (mirrors how the goldens encode sequences: packed tokens +
// query_start_loc [N+1] + per-sequence state slices): a two-sequence prefill
// must equal two independent single-sequence prefills — no state leaks across
// the qsl boundary.
TEST_CASE("gdn prefill: multi-sequence varlen equals independent runs") {
  constexpr int64_t kT = 5, kDk = 2, kDv = 2;
  std::vector<int32_t> qsl = {0, 2, 5};
  std::vector<float> qv(kT * kDk), kv(kT * kDk), vv(kT * kDv), g(kT), beta(kT);
  for (int64_t t = 0; t < kT; ++t) {
    qv[t * 2] = 0.3f * static_cast<float>(t) - 0.5f;
    qv[t * 2 + 1] = 0.9f - 0.2f * static_cast<float>(t);
    kv[t * 2] = 0.7f - 0.3f * static_cast<float>(t);
    kv[t * 2 + 1] = 0.1f * static_cast<float>(t + 1);
    vv[t * 2] = 1.0f - 0.4f * static_cast<float>(t);
    vv[t * 2 + 1] = -0.5f + 0.3f * static_cast<float>(t);
    g[t] = -0.05f * static_cast<float>(t + 1);
    beta[t] = 0.2f + 0.15f * static_cast<float>(t);
  }
  std::vector<float> state = {0.5f, -0.25f, 0.125f, 1.0f, -1.0f, 0.75f, 0.0f, 0.5f};
  std::vector<float> state_joint = state, out_joint(kT * kDv, 0.0f);
  Tensor tq = T3(qv, kT, 1, kDk), tk = T3(kv, kT, 1, kDk), tv = T3(vv, kT, 1, kDv);
  Tensor tg = T2(g, kT, 1), tb = T2(beta, kT, 1);
  Tensor ts = T4(state_joint, 2, 1, kDv, kDk), to = T3(out_joint, kT, 1, kDv);
  Tensor tqsl = Ti(qsl);
  Queue q = Q();
  vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, ts, tqsl, GdnArgs{0.70710678f});
  for (int s = 0; s < 2; ++s) {
    const int64_t begin = qsl[static_cast<size_t>(s)], len = qsl[static_cast<size_t>(s) + 1] - begin;
    std::vector<float> q1(qv.begin() + begin * kDk, qv.begin() + (begin + len) * kDk);
    std::vector<float> k1(kv.begin() + begin * kDk, kv.begin() + (begin + len) * kDk);
    std::vector<float> v1(vv.begin() + begin * kDv, vv.begin() + (begin + len) * kDv);
    std::vector<float> g1(g.begin() + begin, g.begin() + begin + len);
    std::vector<float> b1(beta.begin() + begin, beta.begin() + begin + len);
    std::vector<float> s1(state.begin() + s * kDv * kDk, state.begin() + (s + 1) * kDv * kDk);
    std::vector<float> o1(static_cast<size_t>(len * kDv), 0.0f);
    std::vector<int32_t> qsl1 = {0, static_cast<int32_t>(len)};
    Tensor tq1 = T3(q1, len, 1, kDk), tk1 = T3(k1, len, 1, kDk), tv1 = T3(v1, len, 1, kDv);
    Tensor tg1 = T2(g1, len, 1), tb1 = T2(b1, len, 1), ts1 = T4(s1, 1, 1, kDv, kDk);
    Tensor to1 = T3(o1, len, 1, kDv), tqsl1 = Ti(qsl1);
    vt::GdnPrefill(q, to1, tq1, tk1, tv1, tg1, tb1, ts1, tqsl1, GdnArgs{0.70710678f});
    for (size_t i = 0; i < o1.size(); ++i)
      CHECK(out_joint[static_cast<size_t>(begin * kDv) + i] == doctest::Approx(o1[i]).epsilon(1e-6));
    for (size_t i = 0; i < s1.size(); ++i)
      CHECK(state_joint[static_cast<size_t>(s * kDv * kDk) + i] ==
            doctest::Approx(s1[i]).epsilon(1e-6));
  }
}

// A 1-token prefill and a decode step are the same recurrence.
TEST_CASE("gdn prefill T=1 equals gdn decode") {
  std::vector<float> qv = {0.6f, -0.8f}, kv = {0.28f, 0.96f}, vv = {1.5f, -0.5f};
  std::vector<float> g = {-0.25f}, beta = {0.4f};
  std::vector<float> s_prefill = {0.5f, -1.0f, 0.25f, 0.75f}, s_decode = s_prefill;
  std::vector<float> o_prefill(2, 0.0f), o_decode(2, 0.0f);
  std::vector<int32_t> qsl = {0, 1};
  Tensor tq = T3(qv, 1, 1, 2), tk = T3(kv, 1, 1, 2), tv = T3(vv, 1, 1, 2);
  Tensor tg = T2(g, 1, 1), tb = T2(beta, 1, 1);
  Tensor tsp = T4(s_prefill, 1, 1, 2, 2), tsd = T4(s_decode, 1, 1, 2, 2);
  Tensor top = T3(o_prefill, 1, 1, 2), tod = T3(o_decode, 1, 1, 2), tqsl = Ti(qsl);
  Queue q = Q();
  vt::GdnPrefill(q, top, tq, tk, tv, tg, tb, tsp, tqsl, GdnArgs{0.70710678f});
  vt::GdnDecode(q, tod, tq, tk, tv, tg, tb, tsd, GdnArgs{0.70710678f});
  for (int i = 0; i < 2; ++i) CHECK(o_prefill[i] == doctest::Approx(o_decode[i]));
  for (int i = 0; i < 4; ++i) CHECK(s_prefill[i] == doctest::Approx(s_decode[i]));
}

TEST_CASE("gdn validation: GQA requires Hv multiple of Hk; scale must be set") {
  std::vector<float> qv(1 * 2 * 2, 0.1f), kv(1 * 2 * 2, 0.1f), vv(1 * 3 * 2, 0.1f);
  std::vector<float> g(3, -0.1f), beta(3, 0.5f), state(3 * 2 * 2, 0.0f), out(3 * 2, 0.0f);
  Tensor tq = T3(qv, 1, 2, 2), tk = T3(kv, 1, 2, 2), tv = T3(vv, 1, 3, 2);
  Tensor tg = T2(g, 1, 3), tb = T2(beta, 1, 3), ts = T4(state, 1, 3, 2, 2);
  Tensor to = T3(out, 1, 3, 2);
  Queue q = Q();
  // Hv=3 not a multiple of Hk=2.
  CHECK_THROWS_AS(vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{0.5f}),
                  std::runtime_error);
  // Default (unset) scale throws.
  std::vector<float> vv2(1 * 2 * 2, 0.1f), g2(2, -0.1f), b2(2, 0.5f), st2(2 * 2 * 2, 0.0f),
      out2(2 * 2, 0.0f);
  Tensor tv2 = T3(vv2, 1, 2, 2), tg2 = T2(g2, 1, 2), tb2 = T2(b2, 1, 2);
  Tensor ts2 = T4(st2, 1, 2, 2, 2), to2 = T3(out2, 1, 2, 2);
  CHECK_THROWS_AS(vt::GdnDecode(q, to2, tq, tk, tv2, tg2, tb2, ts2, GdnArgs{}),
                  std::runtime_error);
}

// ---------------------------------------------------------------------------
// CausalConv1dFwd (gdn-semantics.md §2). C=1, K=3 (W=2), bias=0.5 — no golden
// has a bias (Qwen GDN conv is bias=False), so this pins the bias formula:
// acc = bias + sum_j w[j]*window[j], w[:,K-1] multiplies the current token.
//   weight = [0.25, 0.5, 1.0], init state = [3, 4], x = [1, 2]
//   t0: acc = 0.5 + 0.25*3 + 0.5*4 + 1.0*1 = 4.25
//   t1: acc = 0.5 + 0.25*4 + 0.5*1 + 1.0*2 = 4.0
//   write-back (T=2 >= W): state <- last 2 raw x = [1, 2]
TEST_CASE("causal_conv1d_fwd: bias + initial state, hand-computed") {
  std::vector<float> x = {1.0f, 2.0f}, w = {0.25f, 0.5f, 1.0f}, bias = {0.5f};
  std::vector<float> state = {3.0f, 4.0f}, out(2, 0.0f);
  std::vector<int32_t> qsl = {0, 2}, his = {1};
  Tensor tx = T2(x, 2, 1), tw = T2(w, 1, 3), tb = Tensor::Contiguous(bias.data(), DType::kF32,
                                                                     Cpu(), {1});
  Tensor ts = T3(state, 1, 1, 2), to = T2(out, 2, 1), tqsl = Ti(qsl), this_ = Ti(his);
  Queue q = Q();
  SUBCASE("silu activation") {
    vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tqsl, this_, CausalConv1dArgs{true});
    CHECK(out[0] == doctest::Approx(4.190229585f));  // silu(4.25)
    CHECK(out[1] == doctest::Approx(3.928055160f));  // silu(4.0)
  }
  SUBCASE("no activation (linear)") {
    vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tqsl, this_, CausalConv1dArgs{false});
    CHECK(out[0] == doctest::Approx(4.25f));
    CHECK(out[1] == doctest::Approx(4.0f));
  }
  CHECK(state[0] == doctest::Approx(1.0f));  // raw x, pre-activation
  CHECK(state[1] == doctest::Approx(2.0f));
}

// T < K-1 write-back: with an initial state the old state shifts left; without
// one the left slots are zero (gdn-semantics.md §2 state write-back).
TEST_CASE("causal_conv1d_fwd: short-sequence state write-back (T < K-1)") {
  std::vector<float> x = {7.0f}, w = {0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> out(1, 0.0f);
  std::vector<int32_t> qsl = {0, 1};
  Tensor tx = T2(x, 1, 1), tw = T2(w, 1, 4), to = T2(out, 1, 1), tqsl = Ti(qsl);
  Queue q = Q();
  SUBCASE("with initial state: shifted old state") {
    std::vector<float> state = {1.0f, 2.0f, 3.0f};
    std::vector<int32_t> his = {1};
    Tensor ts = T3(state, 1, 1, 3), this_ = Ti(his);
    vt::CausalConv1dFwd(q, to, tx, tw, nullptr, ts, tqsl, this_, CausalConv1dArgs{false});
    CHECK(out[0] == doctest::Approx(7.0f));  // w picks only the current token
    CHECK(state[0] == doctest::Approx(2.0f));
    CHECK(state[1] == doctest::Approx(3.0f));
    CHECK(state[2] == doctest::Approx(7.0f));
  }
  SUBCASE("without initial state: zero left-pad") {
    std::vector<float> state = {1.0f, 2.0f, 3.0f};  // stale garbage; must be ignored
    std::vector<int32_t> his = {0};
    Tensor ts = T3(state, 1, 1, 3), this_ = Ti(his);
    vt::CausalConv1dFwd(q, to, tx, tw, nullptr, ts, tqsl, this_, CausalConv1dArgs{false});
    CHECK(out[0] == doctest::Approx(7.0f));
    CHECK(state[0] == doctest::Approx(0.0f));
    CHECK(state[1] == doctest::Approx(0.0f));
    CHECK(state[2] == doctest::Approx(7.0f));
  }
}

// CausalConv1dUpdate (gdn-semantics.md §3): read-old-then-roll, with bias.
//   state = [1, 2], x = 5, w = [0.25, 0.5, 1.0], bias = 0.5
//   acc = 0.5 + 0.25*1 + 0.5*2 + 1.0*5 = 6.75; state <- [2, 5]
TEST_CASE("causal_conv1d_update: bias + rolling state, hand-computed") {
  std::vector<float> x = {5.0f}, w = {0.25f, 0.5f, 1.0f}, bias = {0.5f};
  std::vector<float> state = {1.0f, 2.0f}, out(1, 0.0f);
  Tensor tx = T2(x, 1, 1), tw = T2(w, 1, 3);
  Tensor tb = Tensor::Contiguous(bias.data(), DType::kF32, Cpu(), {1});
  Tensor ts = T3(state, 1, 1, 2), to = T2(out, 1, 1);
  Queue q = Q();
  vt::CausalConv1dUpdate(q, to, tx, tw, &tb, ts, CausalConv1dArgs{true});
  CHECK(out[0] == doctest::Approx(6.742105806f));  // silu(6.75)
  CHECK(state[0] == doctest::Approx(2.0f));        // rolled left
  CHECK(state[1] == doctest::Approx(5.0f));        // raw x appended
}

// Conv fwd/update consistency: feeding tokens one at a time through the update
// op must match a prefill over the same tokens (same state trajectory).
TEST_CASE("causal_conv1d update chain equals fwd over the same tokens") {
  constexpr int64_t kT = 4, kC = 2, kK = 4;
  std::vector<float> x = {0.5f, -1.0f, 1.5f, 0.25f, -0.75f, 2.0f, 1.0f, -0.5f};  // [T,C]
  std::vector<float> w = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, 0.25f, -0.1f, 0.2f};    // [C,K]
  std::vector<float> st_fwd = {0.2f, -0.3f, 0.7f, 0.1f, 0.4f, -0.6f};            // [1,C,K-1]
  std::vector<float> st_upd = st_fwd;
  std::vector<float> out_fwd(kT * kC, 0.0f);
  std::vector<int32_t> qsl = {0, kT}, his = {1};
  Tensor tx = T2(x, kT, kC), tw = T2(w, kC, kK), tsf = T3(st_fwd, 1, kC, kK - 1);
  Tensor tof = T2(out_fwd, kT, kC), tqsl = Ti(qsl), this_ = Ti(his);
  Queue q = Q();
  vt::CausalConv1dFwd(q, tof, tx, tw, nullptr, tsf, tqsl, this_, CausalConv1dArgs{true});
  Tensor tsu = T3(st_upd, 1, kC, kK - 1);
  for (int64_t t = 0; t < kT; ++t) {
    std::vector<float> xt(x.begin() + t * kC, x.begin() + (t + 1) * kC), ot(kC, 0.0f);
    Tensor txt = T2(xt, 1, kC), tot = T2(ot, 1, kC);
    vt::CausalConv1dUpdate(q, tot, txt, tw, nullptr, tsu, CausalConv1dArgs{true});
    for (int64_t c = 0; c < kC; ++c)
      CHECK(out_fwd[static_cast<size_t>(t * kC + c)] ==
            doctest::Approx(ot[static_cast<size_t>(c)]).epsilon(1e-6));
  }
  for (size_t i = 0; i < st_fwd.size(); ++i)
    CHECK(st_fwd[i] == doctest::Approx(st_upd[i]).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// L2Norm (gdn-semantics.md §4): plain SUM of squares, not mean — x=[3,4] with
// eps=0 gives exactly [0.6, 0.8] (an rmsnorm would give [3,4]/sqrt(12.5)).
TEST_CASE("l2norm: plain sum of squares over the last dim") {
  std::vector<float> x = {3.0f, 4.0f}, out(2, 0.0f);
  Tensor tx = T2(x, 1, 2), to = T2(out, 1, 2);
  Queue q = Q();
  vt::L2Norm(q, to, tx, L2NormArgs{0.0f});
  CHECK(out[0] == doctest::Approx(0.6f));
  CHECK(out[1] == doctest::Approx(0.8f));
}

TEST_CASE("l2norm: rank-3 [T,H,D] normalizes each (token, head) row") {
  std::vector<float> x = {3.0f, 4.0f, 6.0f, 8.0f}, out(4, 0.0f);
  Tensor tx = T3(x, 1, 2, 2), to = T3(out, 1, 2, 2);
  Queue q = Q();
  vt::L2Norm(q, to, tx, L2NormArgs{0.0f});
  CHECK(out[0] == doctest::Approx(0.6f));
  CHECK(out[1] == doctest::Approx(0.8f));
  CHECK(out[2] == doctest::Approx(0.6f));  // head 1 normalized independently
  CHECK(out[3] == doctest::Approx(0.8f));
}

TEST_CASE("l2norm: eps keeps zero rows finite") {
  std::vector<float> x = {0.0f, 0.0f}, out(2, -1.0f);
  Tensor tx = T2(x, 1, 2), to = T2(out, 1, 2);
  Queue q = Q();
  vt::L2Norm(q, to, tx, L2NormArgs{1e-6f});
  CHECK(out[0] == 0.0f);
  CHECK(out[1] == 0.0f);
}

// ---------------------------------------------------------------------------
// RmsNormGated (gdn-semantics.md §5): var is a MEAN (unlike §4), norm first,
// then gate. x=[3,4], w=[2,0.5]: normed = [1.697056, 0.565685] (same as the
// plain rmsnorm golden); gate z=[1,-1]:
//   silu:    [1.697056*silu(1), 0.565685*silu(-1)] = [1.2406475, -0.1521362]
//   sigmoid: [1.697056*sig(1),  0.565685*sig(-1)]  = [1.2406475,  0.1521362]
TEST_CASE("rmsnorm_gated: norm-then-gate, silu and sigmoid activations") {
  std::vector<float> x = {3.0f, 4.0f}, z = {1.0f, -1.0f}, w = {2.0f, 0.5f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = T2(x, 1, 2), tz = T2(z, 1, 2);
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = T2(out, 1, 2);
  Queue q = Q();
  SUBCASE("silu gate (default)") {
    vt::RmsNormGated(q, to, tx, tz, tw, RmsNormGatedArgs{0.0f, false});
    CHECK(out[0] == doctest::Approx(1.2406475f));
    CHECK(out[1] == doctest::Approx(-0.1521362f));
  }
  SUBCASE("sigmoid gate") {
    vt::RmsNormGated(q, to, tx, tz, tw, RmsNormGatedArgs{0.0f, true});
    CHECK(out[0] == doctest::Approx(1.2406475f));
    CHECK(out[1] == doctest::Approx(0.1521362f));
  }
}

// ---------------------------------------------------------------------------
// bf16 dtype rules: bf16 inputs load via f32 conversion; bf16 outputs round on
// store; states stay f32 (a bf16 state must throw).
TEST_CASE("gdn decode: bf16 q/k/v in, bf16 out, f32 state enforced") {
  // Same hand case as the f32 decode test; all inputs exact in bf16.
  std::vector<uint16_t> qv = {vt::F32ToBF16(1.0f), vt::F32ToBF16(0.0f)};
  std::vector<uint16_t> kv = {vt::F32ToBF16(0.0f), vt::F32ToBF16(1.0f)};
  std::vector<uint16_t> vv = {vt::F32ToBF16(2.0f), vt::F32ToBF16(1.0f)};
  std::vector<float> g = {std::log(0.5f)}, beta = {0.5f};
  std::vector<float> state = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> out(2, 0);
  Tensor tq = Tensor::Contiguous(qv.data(), DType::kBF16, Cpu(), {1, 1, 2});
  Tensor tk = Tensor::Contiguous(kv.data(), DType::kBF16, Cpu(), {1, 1, 2});
  Tensor tv = Tensor::Contiguous(vv.data(), DType::kBF16, Cpu(), {1, 1, 2});
  Tensor tg = T2(g, 1, 1), tb = T2(beta, 1, 1), ts = T4(state, 1, 1, 2, 2);
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {1, 1, 2});
  Queue q = Q();
  vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{0.5f});
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(0.25f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(0.75f).epsilon(0.01));
  CHECK(state[0] == doctest::Approx(0.5f));  // state math stays f32
  // bf16 state must throw loudly.
  std::vector<uint16_t> bad_state(4, 0);
  Tensor tbad = Tensor::Contiguous(bad_state.data(), DType::kBF16, Cpu(), {1, 1, 2, 2});
  CHECK_THROWS_AS(vt::GdnDecode(q, to, tq, tk, tv, tg, tb, tbad, GdnArgs{0.5f}),
                  std::runtime_error);
}
