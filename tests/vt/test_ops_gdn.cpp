// vllm.cpp original vt runtime with GDN cases ported from pinned upstream
// tests/kernels/mamba/test_gdn_prefill_cutedsl.py @ e24d1b24 (multi-sequence
// varlen, bf16 output-buffer parity, state parity) and vLLM v0.25.0
// tests/kernels/test_fused_recurrent_packed_decode.py @ 702f4814 (FP16/BF16/
// F32, row-strided packed input, indexed persistent state). Formula reference:
// .agents/specs/gdn-semantics.md (§2-§7 and Tests to port).
// Golden coverage lives in tests/parity/test_op_parity.cpp; these tests pin
// hand-computed values and the corners the goldens do not cover (conv bias,
// GQA ratio 3, tiny hand tables).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_CUDA
// Packed-decode debug counters (reg-tile vs legacy launch sub-counts) plus the
// Triton AOT stats used by the chunked cases. Defined in cuda_gdn.cu, available
// whenever the CUDA backend is built (the packed counters do not require Triton).
#include "vt/cuda/cuda_gdn_internal.h"
#endif

using vt::Backend;
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
  SUBCASE("upstream bool mask stored as i8") {
    std::vector<int8_t> his_i8 = {1};
    Tensor mask = Tensor::Contiguous(his_i8.data(), DType::kI8, Cpu(), {1});
    vt::CausalConv1dFwd(q, to, tx, tw, &tb, ts, tqsl, mask,
                        CausalConv1dArgs{false});
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

// Indexed in-place path (fla fused_recurrent ssm_state_indices / mamba
// causal_conv1d_update conv_state_indices): operating on SCATTERED slots of a
// larger persistent cache must equal the compact op then gather/scatter. This is
// the decode memcpy-tax elimination path (no per-request state gather/scatter).
TEST_CASE("gdn decode + conv update: indexed in-place == compact reference") {
  Queue q = Q();
  const float scale = 0.70710678f;
  // Two single-token sequences (Hk=Hv=1, Dk=Dv=2), scattered onto cache slots
  // {2, 0} of a 3-slot cache — slot 1 is a sentinel that must stay untouched.
  std::vector<int32_t> idx = {2, 0};

  SUBCASE("gdn decode") {
    std::vector<float> qv = {0.3f, -0.7f, 0.5f, 0.1f}, kv = {0.2f, 0.9f, -0.4f, 0.6f};
    std::vector<float> vv = {1.0f, -0.5f, 0.2f, 0.8f}, gv = {-0.1f, -0.3f}, bv = {0.5f, 0.9f};
    std::vector<float> s0 = {1.0f, 2.0f, 3.0f, 4.0f}, s1 = {-1.0f, 0.5f, 0.25f, -0.75f};
    // Compact reference: state rows [seq0; seq1].
    std::vector<float> ref(s0);
    ref.insert(ref.end(), s1.begin(), s1.end());
    std::vector<float> refout(4, 0.0f);
    {
      Tensor tq = T3(qv, 2, 1, 2), tk = T3(kv, 2, 1, 2), tv = T3(vv, 2, 1, 2);
      Tensor tg = T2(gv, 2, 1), tb = T2(bv, 2, 1), ts = T4(ref, 2, 1, 2, 2), to = T3(refout, 2, 1, 2);
      vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{scale});
    }
    // Indexed: 3-slot cache, seq0->slot2, seq1->slot0, slot1 = junk sentinel.
    std::vector<float> cache(3 * 4, 999.0f);
    for (int i = 0; i < 4; ++i) { cache[2 * 4 + i] = s0[i]; cache[0 * 4 + i] = s1[i]; }
    std::vector<float> out(4, 0.0f);
    {
      Tensor tq = T3(qv, 2, 1, 2), tk = T3(kv, 2, 1, 2), tv = T3(vv, 2, 1, 2);
      Tensor tg = T2(gv, 2, 1), tb = T2(bv, 2, 1), ts = T4(cache, 3, 1, 2, 2), to = T3(out, 2, 1, 2);
      Tensor tidx = Ti(idx);
      vt::GdnDecode(q, to, tq, tk, tv, tg, tb, ts, GdnArgs{scale}, &tidx);
    }
    for (int i = 0; i < 4; ++i) CHECK(out[i] == doctest::Approx(refout[i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[2 * 4 + i] == doctest::Approx(ref[i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[0 * 4 + i] == doctest::Approx(ref[4 + i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[1 * 4 + i] == doctest::Approx(999.0f));  // untouched
  }

  SUBCASE("causal_conv1d_update") {
    // x[B=2, C=2], w[C=2, K=3], state row [C, K-1=2] per sequence.
    std::vector<float> x = {5.0f, -1.0f, 2.0f, 0.5f};
    std::vector<float> w = {0.25f, 0.5f, 1.0f, -0.5f, 0.3f, 0.8f};
    std::vector<float> st0 = {1.0f, 2.0f, -1.0f, 0.5f}, st1 = {0.2f, -0.3f, 0.7f, 1.1f};
    std::vector<float> ref(st0);
    ref.insert(ref.end(), st1.begin(), st1.end());
    std::vector<float> refout(4, 0.0f);
    {
      Tensor tx = T2(x, 2, 2), tw = T2(w, 2, 3), ts = T3(ref, 2, 2, 2), to = T2(refout, 2, 2);
      vt::CausalConv1dUpdate(q, to, tx, tw, nullptr, ts, CausalConv1dArgs{true});
    }
    std::vector<float> cache(3 * 2 * 2, 999.0f);
    for (int i = 0; i < 4; ++i) { cache[2 * 4 + i] = st0[i]; cache[0 * 4 + i] = st1[i]; }
    std::vector<float> out(4, 0.0f);
    {
      Tensor tx = T2(x, 2, 2), tw = T2(w, 2, 3), ts = T3(cache, 3, 2, 2), to = T2(out, 2, 2);
      Tensor tidx = Ti(idx);
      vt::CausalConv1dUpdate(q, to, tx, tw, nullptr, ts, CausalConv1dArgs{true}, &tidx);
    }
    for (int i = 0; i < 4; ++i) CHECK(out[i] == doctest::Approx(refout[i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[2 * 4 + i] == doctest::Approx(ref[i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[0 * 4 + i] == doctest::Approx(ref[4 + i]).epsilon(1e-6));
    for (int i = 0; i < 4; ++i) CHECK(cache[1 * 4 + i] == doctest::Approx(999.0f));  // untouched
  }
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

  // Direct prefill has an F32-state kernel ABI on every backend. Persistent
  // compressed cache rows are gathered to F32 by the model boundary before
  // entering this op; accepting the cache tensor directly would make CUDA use
  // four-byte state accesses over two-byte storage.
  std::vector<int32_t> qsl = {0, 1};
  Tensor tqsl = Ti(qsl);
  CHECK_THROWS_WITH_AS(
      vt::GdnPrefill(q, to, tq, tk, tv, tg, tb, tbad, tqsl,
                     GdnArgs{0.5f}),
      doctest::Contains(
          "gdn_prefill: state must be f32; gather compressed cache rows first"),
      std::runtime_error);
}

// ===========================================================================
// CUDA sections: the CUDA GDN kernels (src/vt/cuda/cuda_gdn.cu) vs the CPU
// reference on the SAME inputs (fixed seeds). Guarded like test_cuda_ops.cpp:
// skip cleanly when no GPU is present. Tolerances (M0.6 rope precedent — same
// f32 math, different libm exp and FMA contraction): f32-in 1e-5; bf16-in
// 2e-3 (the input BYTES are identical on both sides, so diffs are arithmetic
// order only, but keep the M0.6 combo values); bf16-out one bf16 ulp (8e-3).
// f32 states compare at 1e-5 in every combo (state math is f32 on both sides)
// and conv states at 1e-6 (the write-back is a raw copy, no arithmetic).

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

// Packs an f32 master vector into the byte representation of dt.
std::vector<uint8_t> Pack(const std::vector<float>& f, DType dt) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), f.data(), out.size());
  } else {
    REQUIRE((dt == DType::kF16 || dt == DType::kBF16));
    auto* p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < f.size(); ++i)
      p[i] = dt == DType::kF16 ? vt::F32ToF16(f[i]) : vt::F32ToBF16(f[i]);
  }
  return out;
}

std::vector<float> Unpack(const std::vector<uint8_t>& b, DType dt) {
  const size_t n = b.size() / vt::SizeOf(dt);
  std::vector<float> out(n);
  if (dt == DType::kF32) {
    std::memcpy(out.data(), b.data(), b.size());
  } else {
    REQUIRE((dt == DType::kF16 || dt == DType::kBF16));
    const auto* p = reinterpret_cast<const uint16_t*>(b.data());
    for (size_t i = 0; i < n; ++i)
      out[i] = dt == DType::kF16 ? vt::F16ToF32(p[i]) : vt::BF16ToF32(p[i]);
  }
  return out;
}

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  size_t first_bad = 0;
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

struct Combo {
  DType in;
  DType out;
  float atol;
  float rtol;
};
constexpr Combo kCudaCombos[] = {
    {DType::kF32, DType::kF32, 1e-5f, 1e-5f},
    {DType::kBF16, DType::kF32, 2e-3f, 2e-3f},
    {DType::kBF16, DType::kBF16, 4e-3f, 8e-3f},
};

// ---------------------------------------------------------------------------
// Packed pure-decode port of vLLM v0.25.0
// tests/kernels/test_fused_recurrent_packed_decode.py. The production op keeps
// q/k normalization in f32, rounds sigmoid(b) through b.dtype, and reads/writes
// persistent state in its declared FP16/BF16/F32 storage dtype.

Tensor RowView(void* data, DType dtype, Device device, int64_t rows,
               int64_t cols, int64_t row_stride) {
  Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = row_stride;
  t.stride[1] = 1;
  return t;
}

float ReadPacked(const Tensor& t, int64_t offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[offset];
    case DType::kF16: return vt::F16ToF32(t.Ptr<uint16_t>()[offset]);
    case DType::kBF16: return vt::BF16ToF32(t.Ptr<uint16_t>()[offset]);
    default: throw std::runtime_error("packed reference requires a float tensor");
  }
}

void WritePacked(const Tensor& t, int64_t offset, float value) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[offset] = value; break;
    case DType::kF16: t.Ptr<uint16_t>()[offset] = vt::F32ToF16(value); break;
    case DType::kBF16: t.Ptr<uint16_t>()[offset] = vt::F32ToBF16(value); break;
    default: throw std::runtime_error("packed reference requires a float tensor");
  }
}

float RoundPacked(float value, DType dtype) {
  if (dtype == DType::kF16) return vt::F16ToF32(vt::F32ToF16(value));
  if (dtype == DType::kBF16) return vt::BF16ToF32(vt::F32ToBF16(value));
  return value;
}

void PackedDecodeReference(Tensor& out, const Tensor& mixed_qkv,
                           const Tensor& a, const Tensor& b,
                           const Tensor& a_log, const Tensor& dt_bias,
                           Tensor& state, const Tensor& state_idx,
                           const GdnArgs& args) {
  const int64_t batch = mixed_qkv.shape[0];
  const int64_t hv_n = state.shape[1];
  const int64_t dv = state.shape[2];
  const int64_t dk = state.shape[3];
  const int64_t qk_dim = mixed_qkv.shape[1] - hv_n * dv;
  const int64_t hk_n = qk_dim / (2 * dk);
  const int64_t ratio = hv_n / hk_n;
  const auto* indices = state_idx.Ptr<int32_t>();
  std::vector<float> qv(static_cast<size_t>(dk));
  std::vector<float> kv(static_cast<size_t>(dk));
  std::vector<float> state_row(static_cast<size_t>(dk));

  for (int64_t bt = 0; bt < batch; ++bt) {
    const int32_t slot = indices[bt];
    if (slot < 0) {
      for (int64_t i = 0; i < hv_n * dv; ++i)
        WritePacked(out, bt * hv_n * dv + i, 0.0f);
      continue;
    }
    for (int64_t hv = 0; hv < hv_n; ++hv) {
      const int64_t hk = hv / ratio;
      float q_sumsq = 0.0f;
      float k_sumsq = 0.0f;
      for (int64_t ki = 0; ki < dk; ++ki) {
        qv[static_cast<size_t>(ki)] =
            ReadPacked(mixed_qkv, bt * mixed_qkv.stride[0] + hk * dk + ki);
        kv[static_cast<size_t>(ki)] = ReadPacked(
            mixed_qkv, bt * mixed_qkv.stride[0] + hk_n * dk + hk * dk + ki);
        q_sumsq += qv[static_cast<size_t>(ki)] * qv[static_cast<size_t>(ki)];
        k_sumsq += kv[static_cast<size_t>(ki)] * kv[static_cast<size_t>(ki)];
      }
      const float q_inv = 1.0f / std::sqrt(q_sumsq + 1e-6f);
      const float k_inv = 1.0f / std::sqrt(k_sumsq + 1e-6f);
      for (int64_t ki = 0; ki < dk; ++ki) {
        qv[static_cast<size_t>(ki)] *= q_inv * args.scale;
        kv[static_cast<size_t>(ki)] *= k_inv;
      }

      const float av = ReadPacked(a, bt * a.stride[0] + hv);
      const float bv = ReadPacked(b, bt * b.stride[0] + hv);
      const float x = av + ReadPacked(dt_bias, hv);
      const float softplus =
          x <= 20.0f ? std::log1p(std::exp(std::min(x, 20.0f))) : x;
      const float decay =
          std::exp(-std::exp(ReadPacked(a_log, hv)) * softplus);
      const float beta =
          RoundPacked(1.0f / (1.0f + std::exp(-bv)), b.dtype);

      for (int64_t vi = 0; vi < dv; ++vi) {
        const int64_t state_base =
            ((static_cast<int64_t>(slot) * hv_n + hv) * dv + vi) * dk;
        float dot = 0.0f;
        for (int64_t ki = 0; ki < dk; ++ki) {
          state_row[static_cast<size_t>(ki)] =
              ReadPacked(state, state_base + ki) * decay;
          dot += state_row[static_cast<size_t>(ki)] * kv[static_cast<size_t>(ki)];
        }
        const int64_t v_offset = bt * mixed_qkv.stride[0] + 2 * hk_n * dk + hv * dv + vi;
        const float vp = (ReadPacked(mixed_qkv, v_offset) - dot) * beta;
        float output = 0.0f;
        for (int64_t ki = 0; ki < dk; ++ki) {
          const float updated = state_row[static_cast<size_t>(ki)] +
                                vp * kv[static_cast<size_t>(ki)];
          WritePacked(state, state_base + ki, updated);
          output += updated * qv[static_cast<size_t>(ki)];
        }
        WritePacked(out, (bt * hv_n + hv) * dv + vi, output);
      }
    }
  }
}

struct PackedCase {
  int64_t batch;
  int64_t hk;
  int64_t hv;
  int64_t dk;
  int64_t dv;
  int64_t slots;
  int64_t mixed_cols;
  int64_t mixed_stride;
  int64_t gate_stride;
  DType dtype;
  std::vector<uint8_t> mixed;
  std::vector<uint8_t> a;
  std::vector<uint8_t> b;
  std::vector<uint8_t> a_log;
  std::vector<uint8_t> dt_bias;
  std::vector<uint8_t> state;
  std::vector<int32_t> indices;
};

PackedCase MakePackedCase(DType dtype, bool strided, int64_t batch,
                          int64_t hk, int64_t hv, int64_t dk, int64_t dv,
                          uint32_t seed) {
  const int64_t mixed_cols = 2 * hk * dk + hv * dv;
  const int64_t mixed_stride = mixed_cols + (strided ? 64 : 0);
  const int64_t gate_stride = hv + (strided ? 5 : 0);
  const int64_t slots = batch;
  auto mixed_f = RandomF32(static_cast<size_t>(batch * mixed_stride), seed);
  auto a_f = RandomF32(static_cast<size_t>(batch * gate_stride), seed + 1);
  auto b_f = RandomF32(static_cast<size_t>(batch * gate_stride), seed + 2);
  auto a_log_f = RandomF32(static_cast<size_t>(hv), seed + 3, -1.0f, 0.5f);
  auto dt_bias_f = RandomF32(static_cast<size_t>(hv), seed + 4, -0.5f, 0.5f);
  auto state_f =
      RandomF32(static_cast<size_t>(slots * hv * dv * dk), seed + 5, -0.25f, 0.25f);
  std::vector<int32_t> indices(static_cast<size_t>(batch));
  for (int64_t i = 0; i < batch; ++i) indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  if (batch >= 3) {
    indices[static_cast<size_t>(batch - 3)] = -1;
    indices[static_cast<size_t>(batch - 2)] = -1;
    indices[static_cast<size_t>(batch - 1)] = -1;
  }
  return PackedCase{batch,
                    hk,
                    hv,
                    dk,
                    dv,
                    slots,
                    mixed_cols,
                    mixed_stride,
                    gate_stride,
                    dtype,
                    Pack(mixed_f, dtype),
                    Pack(a_f, dtype),
                    Pack(b_f, dtype),
                    Pack(a_log_f, dtype),
                    Pack(dt_bias_f, dtype),
                    Pack(state_f, dtype),
                    std::move(indices)};
}

void RunPackedCpuCase(DType dtype, bool strided,
                      DType state_dtype = DType::kI8,
                      DType a_log_dtype = DType::kI8,
                      DType dt_bias_dtype = DType::kI8) {
  PackedCase c = MakePackedCase(dtype, strided, 6, 2, 4, 8, 8, 8400);
  if (state_dtype == DType::kI8) state_dtype = dtype;
  if (a_log_dtype == DType::kI8) a_log_dtype = dtype;
  if (dt_bias_dtype == DType::kI8) dt_bias_dtype = dtype;
  std::vector<uint8_t> a_log =
      Pack(Unpack(c.a_log, dtype), a_log_dtype);
  std::vector<uint8_t> dt_bias =
      Pack(Unpack(c.dt_bias, dtype), dt_bias_dtype);
  const std::vector<uint8_t> initial_state =
      Pack(Unpack(c.state, dtype), state_dtype);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(c.dk))};
  std::vector<uint8_t> state_want = initial_state;
  std::vector<uint8_t> state_got = initial_state;
  std::vector<uint8_t> out_want(
      static_cast<size_t>(c.batch * c.hv * c.dv) * vt::SizeOf(dtype));
  std::vector<uint8_t> out_got(out_want.size());
  Tensor mixed = RowView(c.mixed.data(), dtype, Cpu(), c.batch,
                         c.mixed_cols, c.mixed_stride);
  Tensor ta = RowView(c.a.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor tb = RowView(c.b.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor tal = MakeT(a_log.data(), a_log_dtype, Cpu(), {c.hv});
  Tensor tdt = MakeT(dt_bias.data(), dt_bias_dtype, Cpu(), {c.hv});
  Tensor tidx = MakeT(c.indices.data(), DType::kI32, Cpu(), {c.batch});
  Tensor sw = MakeT(state_want.data(), state_dtype, Cpu(),
                    {c.slots, c.hv, c.dv, c.dk});
  Tensor sg = MakeT(state_got.data(), state_dtype, Cpu(),
                    {c.slots, c.hv, c.dv, c.dk});
  Tensor ow = MakeT(out_want.data(), dtype, Cpu(), {c.batch, c.hv, c.dv});
  Tensor og = MakeT(out_got.data(), dtype, Cpu(), {c.batch, c.hv, c.dv});
  PackedDecodeReference(ow, mixed, ta, tb, tal, tdt, sw, tidx, args);
  Queue queue = Q();
  vt::GdnPackedDecode(queue, og, mixed, ta, tb, tal, tdt, sg, tidx, args);
  const float tol = dtype == DType::kF32 ? 1e-5f : 2e-2f;
  CheckClose(Unpack(out_got, dtype), Unpack(out_want, dtype), tol, tol);
  CheckClose(Unpack(state_got, state_dtype), Unpack(state_want, state_dtype),
             tol, tol);
}

void RunPackedCudaCase(DType dtype, bool strided,
                       DType state_dtype = DType::kI8,
                       DType a_log_dtype = DType::kI8,
                       DType dt_bias_dtype = DType::kI8) {
  PackedCase c = MakePackedCase(dtype, strided, 32, 4, 8, 128, 128, 8500);
  if (state_dtype == DType::kI8) state_dtype = dtype;
  if (a_log_dtype == DType::kI8) a_log_dtype = dtype;
  if (dt_bias_dtype == DType::kI8) dt_bias_dtype = dtype;
  std::vector<uint8_t> a_log =
      Pack(Unpack(c.a_log, dtype), a_log_dtype);
  std::vector<uint8_t> dt_bias =
      Pack(Unpack(c.dt_bias, dtype), dt_bias_dtype);
  const std::vector<uint8_t> initial_state =
      Pack(Unpack(c.state, dtype), state_dtype);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(c.dk))};

  // The portable CPU operation is the explicit reference for the full upstream
  // B=32/H=4/HV=8/K=V=128 matrix.
  std::vector<uint8_t> state_cpu = initial_state;
  std::vector<uint8_t> out_cpu(
      static_cast<size_t>(c.batch * c.hv * c.dv) * vt::SizeOf(dtype));
  Tensor mixed_cpu = RowView(c.mixed.data(), dtype, Cpu(), c.batch,
                             c.mixed_cols, c.mixed_stride);
  Tensor a_cpu = RowView(c.a.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor b_cpu = RowView(c.b.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor al_cpu = MakeT(a_log.data(), a_log_dtype, Cpu(), {c.hv});
  Tensor dt_cpu =
      MakeT(dt_bias.data(), dt_bias_dtype, Cpu(), {c.hv});
  Tensor idx_cpu = MakeT(c.indices.data(), DType::kI32, Cpu(), {c.batch});
  Tensor state_cpu_t =
      MakeT(state_cpu.data(), state_dtype, Cpu(),
            {c.slots, c.hv, c.dv, c.dk});
  Tensor out_cpu_t = MakeT(out_cpu.data(), dtype, Cpu(), {c.batch, c.hv, c.dv});
  Queue cpu_q = Q();
  vt::GdnPackedDecode(cpu_q, out_cpu_t, mixed_cpu, a_cpu, b_cpu, al_cpu,
                      dt_cpu, state_cpu_t, idx_cpu, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor d_mixed(gpu, guard.q, dtype, {c.batch, c.mixed_stride}, c.mixed.data());
  DeviceTensor d_a(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.a.data());
  DeviceTensor d_b(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.b.data());
  DeviceTensor d_al(gpu, guard.q, a_log_dtype, {c.hv}, a_log.data());
  DeviceTensor d_dt(gpu, guard.q, dt_bias_dtype, {c.hv}, dt_bias.data());
  DeviceTensor d_idx(gpu, guard.q, DType::kI32, {c.batch}, c.indices.data());
  DeviceTensor d_state(gpu, guard.q, state_dtype,
                       {c.slots, c.hv, c.dv, c.dk}, initial_state.data());
  DeviceTensor d_out(gpu, guard.q, dtype, {c.batch, c.hv, c.dv});
  Tensor mixed_gpu = d_mixed.tensor();
  mixed_gpu.shape[1] = c.mixed_cols;
  Tensor a_gpu = d_a.tensor();
  a_gpu.shape[1] = c.hv;
  Tensor b_gpu = d_b.tensor();
  b_gpu.shape[1] = c.hv;
  vt::GdnPackedDecode(guard.q, d_out.tensor(), mixed_gpu, a_gpu, b_gpu,
                      d_al.tensor(), d_dt.tensor(), d_state.tensor(),
                      d_idx.tensor(), args);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  std::vector<uint8_t> state_gpu(initial_state.size());
  d_out.Download(guard.q, out_gpu.data());
  d_state.Download(guard.q, state_gpu.data());
  const float tol = dtype == DType::kF32 ? 1e-4f : 2e-2f;
  const float rtol = dtype == DType::kF32 ? 1e-4f : 1e-2f;
  CheckClose(Unpack(out_gpu, dtype), Unpack(out_cpu, dtype), tol, rtol);
  CheckClose(Unpack(state_gpu, state_dtype), Unpack(state_cpu, state_dtype),
             tol, rtol);
}

TEST_CASE("gdn packed decode CPU reference matches explicit recurrence") {
  for (DType dtype : {DType::kF16, DType::kBF16, DType::kF32}) {
    RunPackedCpuCase(dtype, false);
    RunPackedCpuCase(dtype, true);
  }
  // Real Qwen3.6 boundary: BF16 activations/output, FP32 SSM cache and A_log,
  // BF16 dt_bias. These types are independent in upstream's Triton kernel.
  RunPackedCpuCase(DType::kBF16, true, DType::kF32, DType::kF32,
                   DType::kBF16);
}

TEST_CASE("gdn packed decode rejects duplicate and out-of-range live slots") {
  PackedCase c = MakePackedCase(DType::kF32, false, 2, 1, 1, 4, 4, 8600);
  const GdnArgs args{0.5f};
  std::vector<float> out(8);
  Tensor mixed = RowView(c.mixed.data(), c.dtype, Cpu(), 2, c.mixed_cols,
                         c.mixed_stride);
  Tensor ta = RowView(c.a.data(), c.dtype, Cpu(), 2, 1, c.gate_stride);
  Tensor tb = RowView(c.b.data(), c.dtype, Cpu(), 2, 1, c.gate_stride);
  Tensor tal = MakeT(c.a_log.data(), c.dtype, Cpu(), {1});
  Tensor tdt = MakeT(c.dt_bias.data(), c.dtype, Cpu(), {1});
  Tensor state = MakeT(c.state.data(), c.dtype, Cpu(), {2, 1, 4, 4});
  Tensor tout = MakeT(out.data(), DType::kF32, Cpu(), {2, 1, 4});
  Queue queue = Q();
  std::vector<int32_t> duplicate = {0, 0};
  Tensor dup = MakeT(duplicate.data(), DType::kI32, Cpu(), {2});
  CHECK_THROWS_AS(vt::GdnPackedDecode(queue, tout, mixed, ta, tb, tal, tdt,
                                     state, dup, args),
                  std::runtime_error);
  std::vector<int32_t> out_of_range = {0, 2};
  Tensor oor = MakeT(out_of_range.data(), DType::kI32, Cpu(), {2});
  CHECK_THROWS_AS(vt::GdnPackedDecode(queue, tout, mixed, ta, tb, tal, tdt,
                                     state, oor, args),
                  std::runtime_error);
}

TEST_CASE("gdn packed decode permits fewer live state slots than padded batch rows") {
  PackedCase c = MakePackedCase(DType::kF32, false, 3, 1, 1, 4, 4, 8650);
  c.slots = 2;
  c.state.resize(static_cast<size_t>(c.slots * c.hv * c.dv * c.dk) *
                 vt::SizeOf(c.dtype));
  c.indices = {0, 1, -1};
  const GdnArgs args{0.5f};
  std::vector<uint8_t> state_want = c.state;
  std::vector<uint8_t> state_got = c.state;
  std::vector<float> out_want(static_cast<size_t>(c.batch * c.hv * c.dv));
  std::vector<float> out_got(out_want.size());
  Tensor mixed = RowView(c.mixed.data(), c.dtype, Cpu(), c.batch,
                         c.mixed_cols, c.mixed_stride);
  Tensor ta = RowView(c.a.data(), c.dtype, Cpu(), c.batch, c.hv,
                      c.gate_stride);
  Tensor tb = RowView(c.b.data(), c.dtype, Cpu(), c.batch, c.hv,
                      c.gate_stride);
  Tensor tal = MakeT(c.a_log.data(), c.dtype, Cpu(), {c.hv});
  Tensor tdt = MakeT(c.dt_bias.data(), c.dtype, Cpu(), {c.hv});
  Tensor tidx = MakeT(c.indices.data(), DType::kI32, Cpu(), {c.batch});
  Tensor sw = MakeT(state_want.data(), c.dtype, Cpu(),
                    {c.slots, c.hv, c.dv, c.dk});
  Tensor sg = MakeT(state_got.data(), c.dtype, Cpu(),
                    {c.slots, c.hv, c.dv, c.dk});
  Tensor ow = MakeT(out_want.data(), c.dtype, Cpu(),
                    {c.batch, c.hv, c.dv});
  Tensor og = MakeT(out_got.data(), c.dtype, Cpu(),
                    {c.batch, c.hv, c.dv});
  PackedDecodeReference(ow, mixed, ta, tb, tal, tdt, sw, tidx, args);
  Queue queue = Q();
  vt::GdnPackedDecode(queue, og, mixed, ta, tb, tal, tdt, sg, tidx, args);
  CheckClose(out_got, out_want, 1e-5f, 1e-5f);
  CheckClose(Unpack(state_got, c.dtype), Unpack(state_want, c.dtype), 1e-5f,
             1e-5f);
}

TEST_CASE("CUDA gdn packed decode matches CPU across upstream dtype and stride matrix") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  for (DType dtype : {DType::kF16, DType::kBF16, DType::kF32}) {
    RunPackedCudaCase(dtype, false);
    RunPackedCudaCase(dtype, true);
  }
  RunPackedCudaCase(DType::kBF16, true, DType::kF32, DType::kF32,
                    DType::kBF16);
}

TEST_CASE("CUDA gdn packed decode capture replays preserve padding state and canaries") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  constexpr DType dtype = DType::kBF16;
  PackedCase c = MakePackedCase(dtype, true, 6, 2, 4, 32, 32, 8700);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(c.dk))};

  std::vector<uint8_t> state_cpu = c.state;
  std::vector<uint8_t> out_cpu(
      static_cast<size_t>(c.batch * c.hv * c.dv) * vt::SizeOf(dtype));
  Tensor mixed_cpu = RowView(c.mixed.data(), dtype, Cpu(), c.batch,
                             c.mixed_cols, c.mixed_stride);
  Tensor a_cpu =
      RowView(c.a.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor b_cpu =
      RowView(c.b.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor al_cpu = MakeT(c.a_log.data(), dtype, Cpu(), {c.hv});
  Tensor dt_cpu = MakeT(c.dt_bias.data(), dtype, Cpu(), {c.hv});
  Tensor idx_cpu = MakeT(c.indices.data(), DType::kI32, Cpu(), {c.batch});
  Tensor state_cpu_t =
      MakeT(state_cpu.data(), dtype, Cpu(), {c.slots, c.hv, c.dv, c.dk});
  Tensor out_cpu_t = MakeT(out_cpu.data(), dtype, Cpu(), {c.batch, c.hv, c.dv});
  Queue cpu_q = Q();
  vt::GdnPackedDecode(cpu_q, out_cpu_t, mixed_cpu, a_cpu, b_cpu, al_cpu,
                      dt_cpu, state_cpu_t, idx_cpu, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor d_mixed(gpu, guard.q, dtype, {c.batch, c.mixed_stride},
                       c.mixed.data());
  DeviceTensor d_a(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.a.data());
  DeviceTensor d_b(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.b.data());
  DeviceTensor d_al(gpu, guard.q, dtype, {c.hv}, c.a_log.data());
  DeviceTensor d_dt(gpu, guard.q, dtype, {c.hv}, c.dt_bias.data());
  DeviceTensor d_idx(gpu, guard.q, DType::kI32, {c.batch}, c.indices.data());
  constexpr size_t kGuardBytes = 256;
  std::vector<uint8_t> state_storage(2 * kGuardBytes + c.state.size(), 0xA5);
  std::copy(c.state.begin(), c.state.end(),
            state_storage.begin() + static_cast<ptrdiff_t>(kGuardBytes));
  std::vector<uint8_t> out_storage(2 * kGuardBytes + out_cpu.size(), 0xA5);
  DeviceTensor d_state_storage(
      gpu, guard.q, DType::kI8,
      {static_cast<int64_t>(state_storage.size())}, state_storage.data());
  DeviceTensor d_out_storage(
      gpu, guard.q, DType::kI8,
      {static_cast<int64_t>(out_storage.size())}, out_storage.data());
  Tensor state_gpu = MakeT(
      static_cast<uint8_t*>(d_state_storage.tensor().data) + kGuardBytes,
      dtype, Gpu(), {c.slots, c.hv, c.dv, c.dk});
  Tensor out_gpu = MakeT(
      static_cast<uint8_t*>(d_out_storage.tensor().data) + kGuardBytes,
      dtype, Gpu(), {c.batch, c.hv, c.dv});
  Tensor mixed_gpu = d_mixed.tensor();
  mixed_gpu.shape[1] = c.mixed_cols;
  Tensor a_gpu = d_a.tensor();
  a_gpu.shape[1] = c.hv;
  Tensor b_gpu = d_b.tensor();
  b_gpu.shape[1] = c.hv;

  gpu.BeginCapture(guard.q);
  vt::GdnPackedDecode(guard.q, out_gpu, mixed_gpu, a_gpu, b_gpu,
                      d_al.tensor(), d_dt.tensor(), state_gpu,
                      d_idx.tensor(), args);
  void* graph = gpu.EndCaptureGraph(guard.q);
  std::vector<uint8_t> out_result(out_cpu.size());
  std::vector<uint8_t> state_result(c.state.size());
  std::vector<uint8_t> out_storage_after(out_storage.size());
  std::vector<uint8_t> state_storage_after(state_storage.size());
  for (int replay = 0; replay < 2; ++replay) {
    gpu.Copy(guard.q, state_gpu.data, c.state.data(), c.state.size());
    gpu.Memset(guard.q, out_gpu.data, 0xA5, out_result.size());
    gpu.ReplayGraph(guard.q, graph);
    d_out_storage.Download(guard.q, out_storage_after.data());
    d_state_storage.Download(guard.q, state_storage_after.data());
    std::copy_n(out_storage_after.begin() +
                    static_cast<ptrdiff_t>(kGuardBytes),
                out_result.size(), out_result.begin());
    std::copy_n(state_storage_after.begin() +
                    static_cast<ptrdiff_t>(kGuardBytes),
                state_result.size(), state_result.begin());
    CheckClose(Unpack(out_result, dtype), Unpack(out_cpu, dtype), 2e-2f,
               1e-2f);
    CheckClose(Unpack(state_result, dtype), Unpack(state_cpu, dtype), 2e-2f,
               1e-2f);
    CHECK(std::all_of(out_storage_after.begin(),
                      out_storage_after.begin() +
                          static_cast<ptrdiff_t>(kGuardBytes),
                      [](uint8_t v) { return v == 0xA5; }));
    CHECK(std::all_of(out_storage_after.end() -
                          static_cast<ptrdiff_t>(kGuardBytes),
                      out_storage_after.end(),
                      [](uint8_t v) { return v == 0xA5; }));
    CHECK(std::all_of(state_storage_after.begin(),
                      state_storage_after.begin() +
                          static_cast<ptrdiff_t>(kGuardBytes),
                      [](uint8_t v) { return v == 0xA5; }));
    CHECK(std::all_of(state_storage_after.end() -
                          static_cast<ptrdiff_t>(kGuardBytes),
                      state_storage_after.end(),
                      [](uint8_t v) { return v == 0xA5; }));
  }
  gpu.DestroyGraph(graph);

  std::vector<uint8_t> mixed_after(c.mixed.size());
  std::vector<uint8_t> a_after(c.a.size());
  std::vector<uint8_t> b_after(c.b.size());
  d_mixed.Download(guard.q, mixed_after.data());
  d_a.Download(guard.q, a_after.data());
  d_b.Download(guard.q, b_after.data());
  CHECK(mixed_after == c.mixed);
  CHECK(a_after == c.a);
  CHECK(b_after == c.b);
}

// VT_GDN_PACKED_REG_TILE opt-in contract (default OFF): the legacy kernel is the
// default; "=0" restores the legacy shared-memory kernel in the SAME binary.
// Uses the packed-decode host-dispatch debug counters (reg-tile vs legacy
// sub-counts) to prove which kernel each arm actually launched, and checks BOTH
// arms match the portable CPU reference (loose tol; reduction orders differ).
// Guarded on VLLM_CPP_CUDA because it references the CUDA-only debug counters
// (defined in cuda_gdn.cu); the portable flag parse is covered on every platform
// by tests/vt/test_gdn_packed_reg_tile.cpp.
#ifdef VLLM_CPP_CUDA
TEST_CASE(
    "CUDA gdn packed decode defaults legacy; VT_GDN_PACKED_REG_TILE=1 opts in "
    "selects reg-tile") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  constexpr DType dtype = DType::kBF16;
  // Real 27B gate geometry (dk == dv == 128 => bv == 32) so the reg-tile path is
  // eligible (the dk==32 capture case exercises the BK==32 instantiation).
  PackedCase c = MakePackedCase(dtype, false, 32, 4, 8, 128, 128, 8900);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(c.dk))};

  // Portable CPU reference for the correctness of BOTH arms.
  std::vector<uint8_t> state_cpu = c.state;
  std::vector<uint8_t> out_cpu(
      static_cast<size_t>(c.batch * c.hv * c.dv) * vt::SizeOf(dtype));
  Tensor mixed_cpu = RowView(c.mixed.data(), dtype, Cpu(), c.batch,
                             c.mixed_cols, c.mixed_stride);
  Tensor a_cpu = RowView(c.a.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor b_cpu = RowView(c.b.data(), dtype, Cpu(), c.batch, c.hv, c.gate_stride);
  Tensor al_cpu = MakeT(c.a_log.data(), dtype, Cpu(), {c.hv});
  Tensor dt_cpu = MakeT(c.dt_bias.data(), dtype, Cpu(), {c.hv});
  Tensor idx_cpu = MakeT(c.indices.data(), DType::kI32, Cpu(), {c.batch});
  Tensor state_cpu_t =
      MakeT(state_cpu.data(), dtype, Cpu(), {c.slots, c.hv, c.dv, c.dk});
  Tensor out_cpu_t = MakeT(out_cpu.data(), dtype, Cpu(), {c.batch, c.hv, c.dv});
  Queue cpu_q = Q();
  vt::GdnPackedDecode(cpu_q, out_cpu_t, mixed_cpu, a_cpu, b_cpu, al_cpu, dt_cpu,
                      state_cpu_t, idx_cpu, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor d_mixed(gpu, guard.q, dtype, {c.batch, c.mixed_stride},
                       c.mixed.data());
  DeviceTensor d_a(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.a.data());
  DeviceTensor d_b(gpu, guard.q, dtype, {c.batch, c.gate_stride}, c.b.data());
  DeviceTensor d_al(gpu, guard.q, dtype, {c.hv}, c.a_log.data());
  DeviceTensor d_dt(gpu, guard.q, dtype, {c.hv}, c.dt_bias.data());
  DeviceTensor d_idx(gpu, guard.q, DType::kI32, {c.batch}, c.indices.data());
  Tensor mixed_gpu = d_mixed.tensor();
  mixed_gpu.shape[1] = c.mixed_cols;
  Tensor a_gpu = d_a.tensor();
  a_gpu.shape[1] = c.hv;
  Tensor b_gpu = d_b.tensor();
  b_gpu.shape[1] = c.hv;

  auto run = [&](const char* env_value, uint64_t& reg_tile, uint64_t& legacy,
                 std::vector<uint8_t>& out_host) {
    if (env_value == nullptr)
      unsetenv("VT_GDN_PACKED_REG_TILE");
    else
      setenv("VT_GDN_PACKED_REG_TILE", env_value, 1);
    DeviceTensor d_state(gpu, guard.q, dtype, {c.slots, c.hv, c.dv, c.dk},
                         c.state.data());
    DeviceTensor d_out(gpu, guard.q, dtype, {c.batch, c.hv, c.dv});
    vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
    vt::GdnPackedDecode(guard.q, d_out.tensor(), mixed_gpu, a_gpu, b_gpu,
                        d_al.tensor(), d_dt.tensor(), d_state.tensor(),
                        d_idx.tensor(), args);
    const auto stats = vt::cuda::testing::GetGdnPackedDecodeDebugStats();
    vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
    reg_tile = stats.reg_tile_launches;
    legacy = stats.legacy_launches;
    out_host.resize(out_cpu.size());
    d_out.Download(guard.q, out_host.data());
  };

  // Default (unset) selects the LEGACY kernel (reg-tile default OFF after the
  // failed 2026-07-16 DGX proof: oracle boundary FAIL + c16 -12%).
  uint64_t reg0 = 0, legacy0 = 0;
  std::vector<uint8_t> out_legacy;
  run(nullptr, reg0, legacy0, out_legacy);
  CHECK(legacy0 == 1);
  CHECK(reg0 == 0);
  CheckClose(Unpack(out_legacy, dtype), Unpack(out_cpu, dtype), 2e-2f, 1e-2f);

  // Experimental opt-in: VT_GDN_PACKED_REG_TILE=1 selects the register-resident
  // kernel (kept for a corrected future port attempt).
  uint64_t reg1 = 0, legacy1 = 0;
  std::vector<uint8_t> out_reg;
  run("1", reg1, legacy1, out_reg);
  CHECK(reg1 == 1);
  CHECK(legacy1 == 0);
  CheckClose(Unpack(out_reg, dtype), Unpack(out_cpu, dtype), 2e-2f, 1e-2f);

  unsetenv("VT_GDN_PACKED_REG_TILE");
}
#endif  // VLLM_CPP_CUDA

// Vendored Triton AOT packed-decode fast-path (VT_GDN_PACKED_DECODE_TRITON,
// default ON; =0 rolls back; 27B-only). MEASURED codegen-bound (dgx phase1
// 2026-07-16): the identical register-resident [BV=32,BK=128] fp32 state tile is
// REG:205/0-spill under Triton but REG:255+STACK:48 (spills) as hand-CUDA. Builds
// the EXACT 27B packed-decode call site the single cubin was baked to (bf16
// mixed_qkv/a/b/out, f32 state/A_log/dt_bias, merged-BA a/b views of row stride
// 2*Hv=96, Hk=16, Hv=48, Dk=Dv=128) so the launcher guards accept it, and proves
// (a) the DEFAULT now fires the vendored cubin (it is vLLM's exact kernel), (b)
// "=0" rolls back to the hand legacy kernel, and (c) both the AOT and legacy
// outputs/state match the portable CPU reference within the BF16 tolerance.
// Guarded on VLLM_CPP_TRITON (the cubin only exists in that build).
#if defined(VLLM_CPP_CUDA) && defined(VLLM_CPP_TRITON)
TEST_CASE(
    "CUDA gdn packed decode: VT_GDN_PACKED_DECODE_TRITON defaults ON (fires the "
    "vendored 27B cubin), =0 rolls back, both match the CPU reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  constexpr DType dtype = DType::kBF16;
  constexpr int64_t batch = 16, hk = 16, hv = 48, dk = 128, dv = 128;
  PackedCase c = MakePackedCase(dtype, false, batch, hk, hv, dk, dv, 9310);
  REQUIRE(c.mixed_stride == 2 * hk * dk + hv * dv);  // 10240 (baked)
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(dk))};

  // Merged-BA buffer [batch, 2*Hv] bf16: cols [0,Hv)=b, cols [Hv,2*Hv)=a (mirrors
  // ProjectGdnBA: out.b=Slice(0,Hv), out.a=Slice(Hv,2*Hv)); a/b are stride-96
  // views into it, exactly what the model passes to GdnPackedDecode.
  const int64_t ba_stride = 2 * hv;  // 96 (the baked stride_a/b_tok)
  std::vector<uint8_t> ba(static_cast<size_t>(batch) * ba_stride *
                          vt::SizeOf(dtype));
  const size_t half = static_cast<size_t>(hv) * vt::SizeOf(dtype);
  for (int64_t r = 0; r < batch; ++r) {
    std::memcpy(ba.data() + static_cast<size_t>(r * ba_stride) * vt::SizeOf(dtype),
                c.b.data() + static_cast<size_t>(r * hv) * vt::SizeOf(dtype), half);
    std::memcpy(ba.data() + (static_cast<size_t>(r * ba_stride) + hv) * vt::SizeOf(dtype),
                c.a.data() + static_cast<size_t>(r * hv) * vt::SizeOf(dtype), half);
  }
  // f32 SSM state, A_log and dt_bias (the real 27B resident dtypes).
  std::vector<uint8_t> state_f32 = Pack(Unpack(c.state, dtype), DType::kF32);
  std::vector<uint8_t> a_log_f32 = Pack(Unpack(c.a_log, dtype), DType::kF32);
  std::vector<uint8_t> dt_bias_f32 = Pack(Unpack(c.dt_bias, dtype), DType::kF32);

  // Portable CPU reference for the exact config.
  std::vector<uint8_t> state_cpu = state_f32;
  std::vector<uint8_t> out_cpu(
      static_cast<size_t>(batch * hv * dv) * vt::SizeOf(dtype));
  Tensor mixed_cpu = RowView(c.mixed.data(), dtype, Cpu(), batch, c.mixed_cols,
                             c.mixed_stride);
  Tensor ba_cpu = RowView(ba.data(), dtype, Cpu(), batch, ba_stride, ba_stride);
  Tensor b_cpu = ba_cpu.Slice(1, 0, hv);       // stride 96
  Tensor a_cpu = ba_cpu.Slice(1, hv, 2 * hv);  // stride 96
  Tensor al_cpu = MakeT(a_log_f32.data(), DType::kF32, Cpu(), {hv});
  Tensor dt_cpu = MakeT(dt_bias_f32.data(), DType::kF32, Cpu(), {hv});
  Tensor idx_cpu = MakeT(c.indices.data(), DType::kI32, Cpu(), {batch});
  Tensor state_cpu_t =
      MakeT(state_cpu.data(), DType::kF32, Cpu(), {c.slots, hv, dv, dk});
  Tensor out_cpu_t = MakeT(out_cpu.data(), dtype, Cpu(), {batch, hv, dv});
  Queue cpu_q = Q();
  vt::GdnPackedDecode(cpu_q, out_cpu_t, mixed_cpu, a_cpu, b_cpu, al_cpu, dt_cpu,
                      state_cpu_t, idx_cpu, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor d_mixed(gpu, guard.q, dtype, {batch, c.mixed_stride},
                       c.mixed.data());
  DeviceTensor d_ba(gpu, guard.q, dtype, {batch, ba_stride}, ba.data());
  DeviceTensor d_al(gpu, guard.q, DType::kF32, {hv}, a_log_f32.data());
  DeviceTensor d_dt(gpu, guard.q, DType::kF32, {hv}, dt_bias_f32.data());
  DeviceTensor d_idx(gpu, guard.q, DType::kI32, {batch}, c.indices.data());
  Tensor mixed_gpu = d_mixed.tensor();
  mixed_gpu.shape[1] = c.mixed_cols;
  Tensor b_gpu = d_ba.tensor().Slice(1, 0, hv);       // stride 96 view
  Tensor a_gpu = d_ba.tensor().Slice(1, hv, 2 * hv);  // stride 96 view

  auto run = [&](const char* env_value, uint64_t& triton, uint64_t& legacy,
                 std::vector<uint8_t>& out_host, std::vector<uint8_t>& state_host) {
    if (env_value == nullptr)
      unsetenv("VT_GDN_PACKED_DECODE_TRITON");
    else
      setenv("VT_GDN_PACKED_DECODE_TRITON", env_value, 1);
    DeviceTensor d_state(gpu, guard.q, DType::kF32, {c.slots, hv, dv, dk},
                         state_f32.data());
    DeviceTensor d_out(gpu, guard.q, dtype, {batch, hv, dv});
    vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
    vt::GdnPackedDecode(guard.q, d_out.tensor(), mixed_gpu, a_gpu, b_gpu,
                        d_al.tensor(), d_dt.tensor(), d_state.tensor(),
                        d_idx.tensor(), args);
    const auto stats = vt::cuda::testing::GetGdnPackedDecodeDebugStats();
    vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
    triton = stats.triton_launches;
    legacy = stats.legacy_launches;
    out_host.resize(out_cpu.size());
    state_host.resize(state_f32.size());
    d_out.Download(guard.q, out_host.data());
    d_state.Download(guard.q, state_host.data());
  };

  // Rollback "=0": the hand kernel handles it, the vendored cubin does not fire.
  uint64_t t0 = 0, l0 = 0;
  std::vector<uint8_t> out_legacy, state_legacy;
  run("0", t0, l0, out_legacy, state_legacy);
  CHECK(t0 == 0);
  CHECK(l0 == 1);
  CheckClose(Unpack(out_legacy, dtype), Unpack(out_cpu, dtype), 2e-2f, 1e-2f);

  // Default (unset): the vendored Triton cubin fires (it is vLLM's exact kernel)
  // and is bit-close to the oracle (sequential per-lane reduction), so it matches
  // the CPU reference tightly.
  uint64_t t1 = 0, l1 = 0;
  std::vector<uint8_t> out_aot, state_aot;
  run(nullptr, t1, l1, out_aot, state_aot);
  CHECK(t1 == 1);
  CHECK(l1 == 0);
  CheckClose(Unpack(out_aot, dtype), Unpack(out_cpu, dtype), 2e-2f, 1e-2f);
  CheckClose(Unpack(state_aot, DType::kF32), Unpack(state_cpu, DType::kF32),
             2e-2f, 1e-2f);
  // The AOT and legacy kernels implement the same semantics; agree closely.
  CheckClose(Unpack(out_aot, dtype), Unpack(out_legacy, dtype), 2e-2f, 1e-2f);

  unsetenv("VT_GDN_PACKED_DECODE_TRITON");
}
#endif  // VLLM_CPP_CUDA && VLLM_CPP_TRITON

// causal_conv1d_fwd CPU-vs-CUDA on one varlen batch.
void RunConvFwdCudaCase(const std::vector<int32_t>& qsl, const std::vector<int32_t>& his,
                        int64_t c, int64_t k, bool with_bias, bool silu, const Combo& cb,
                        uint32_t seed, bool i8_mask = false) {
  const int64_t n = static_cast<int64_t>(qsl.size()) - 1;
  const int64_t t = qsl.back();
  const auto xf = RandomF32(static_cast<size_t>(t * c), seed);
  const auto wf = RandomF32(static_cast<size_t>(c * k), seed + 1, -1.0f, 1.0f);
  const auto bf = RandomF32(static_cast<size_t>(c), seed + 2, -1.0f, 1.0f);
  const auto stf = RandomF32(static_cast<size_t>(n * c * (k - 1)), seed + 3);
  const auto xb = Pack(xf, cb.in);
  const auto wb = Pack(wf, cb.in);
  const auto bb = Pack(bf, cb.in);
  const CausalConv1dArgs args{silu};

  // CPU reference (state mutated in place on a copy).
  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * c) * vt::SizeOf(cb.out));
  std::vector<float> st_cpu = stf;
  std::vector<int32_t> qsl_cpu = qsl, his_cpu = his;
  std::vector<int8_t> his_i8(his.begin(), his.end());
  Tensor tx = MakeT(const_cast<uint8_t*>(xb.data()), cb.in, Cpu(), {t, c});
  Tensor tw = MakeT(const_cast<uint8_t*>(wb.data()), cb.in, Cpu(), {c, k});
  Tensor tb = MakeT(const_cast<uint8_t*>(bb.data()), cb.in, Cpu(), {c});
  Tensor ts = MakeT(st_cpu.data(), DType::kF32, Cpu(), {n, c, k - 1});
  Tensor tqsl = MakeT(qsl_cpu.data(), DType::kI32, Cpu(), {n + 1});
  Tensor this_ = i8_mask ? MakeT(his_i8.data(), DType::kI8, Cpu(), {n})
                         : MakeT(his_cpu.data(), DType::kI32, Cpu(), {n});
  Tensor to = MakeT(out_cpu.data(), cb.out, Cpu(), {t, c});
  Queue cq = Q();
  vt::CausalConv1dFwd(cq, to, tx, tw, with_bias ? &tb : nullptr, ts, tqsl, this_, args);

  // CUDA on the same packed inputs; qsl/has_initial_state live on the DEVICE
  // (the kernels read them device-side — M0.7 metadata contract).
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, cb.in, {t, c}, xb.data());
  DeviceTensor dw(gpu, gq.q, cb.in, {c, k}, wb.data());
  DeviceTensor db(gpu, gq.q, cb.in, {c}, bb.data());
  DeviceTensor dst(gpu, gq.q, DType::kF32, {n, c, k - 1}, stf.data());
  DeviceTensor dqsl(gpu, gq.q, DType::kI32, {n + 1}, qsl.data());
  DeviceTensor dhis(gpu, gq.q, i8_mask ? DType::kI8 : DType::kI32, {n},
                    i8_mask ? static_cast<const void*>(his_i8.data())
                            : static_cast<const void*>(his.data()));
  DeviceTensor dout(gpu, gq.q, cb.out, {t, c});
  vt::CausalConv1dFwd(gq.q, dout.tensor(), dx.tensor(), dw.tensor(),
                      with_bias ? &db.tensor() : nullptr, dst.tensor(), dqsl.tensor(),
                      dhis.tensor(), args);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  std::vector<float> st_gpu(st_cpu.size());
  dout.Download(gq.q, out_gpu.data());
  dst.Download(gq.q, st_gpu.data());

  CheckClose(Unpack(out_gpu, cb.out), Unpack(out_cpu, cb.out), cb.atol, cb.rtol);
  CheckClose(st_gpu, st_cpu, 1e-6f, 0.0f);  // raw-copy write-back: exact
}

// causal_conv1d_update CPU-vs-CUDA on one single-token batch.
void RunConvUpdateCudaCase(int64_t batch, int64_t c, int64_t k, bool with_bias, bool silu,
                           const Combo& cb, uint32_t seed) {
  const auto xf = RandomF32(static_cast<size_t>(batch * c), seed);
  const auto wf = RandomF32(static_cast<size_t>(c * k), seed + 1, -1.0f, 1.0f);
  const auto bf = RandomF32(static_cast<size_t>(c), seed + 2, -1.0f, 1.0f);
  const auto stf = RandomF32(static_cast<size_t>(batch * c * (k - 1)), seed + 3);
  const auto xb = Pack(xf, cb.in);
  const auto wb = Pack(wf, cb.in);
  const auto bb = Pack(bf, cb.in);
  const CausalConv1dArgs args{silu};

  std::vector<uint8_t> out_cpu(static_cast<size_t>(batch * c) * vt::SizeOf(cb.out));
  std::vector<float> st_cpu = stf;
  Tensor tx = MakeT(const_cast<uint8_t*>(xb.data()), cb.in, Cpu(), {batch, c});
  Tensor tw = MakeT(const_cast<uint8_t*>(wb.data()), cb.in, Cpu(), {c, k});
  Tensor tb = MakeT(const_cast<uint8_t*>(bb.data()), cb.in, Cpu(), {c});
  Tensor ts = MakeT(st_cpu.data(), DType::kF32, Cpu(), {batch, c, k - 1});
  Tensor to = MakeT(out_cpu.data(), cb.out, Cpu(), {batch, c});
  Queue cq = Q();
  vt::CausalConv1dUpdate(cq, to, tx, tw, with_bias ? &tb : nullptr, ts, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, cb.in, {batch, c}, xb.data());
  DeviceTensor dw(gpu, gq.q, cb.in, {c, k}, wb.data());
  DeviceTensor db(gpu, gq.q, cb.in, {c}, bb.data());
  DeviceTensor dst(gpu, gq.q, DType::kF32, {batch, c, k - 1}, stf.data());
  DeviceTensor dout(gpu, gq.q, cb.out, {batch, c});
  vt::CausalConv1dUpdate(gq.q, dout.tensor(), dx.tensor(), dw.tensor(),
                         with_bias ? &db.tensor() : nullptr, dst.tensor(), args);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  std::vector<float> st_gpu(st_cpu.size());
  dout.Download(gq.q, out_gpu.data());
  dst.Download(gq.q, st_gpu.data());

  CheckClose(Unpack(out_gpu, cb.out), Unpack(out_cpu, cb.out), cb.atol, cb.rtol);
  CheckClose(st_gpu, st_cpu, 1e-6f, 0.0f);  // rolled raw values: exact
}

// l2norm CPU-vs-CUDA on one shape (rank 2 or 3).
void RunL2NormCudaCase(const std::vector<int64_t>& shape, const Combo& cb, uint32_t seed) {
  size_t numel = 1;
  for (int64_t s : shape) numel *= static_cast<size_t>(s);
  const auto xf = RandomF32(numel, seed);
  const auto xb = Pack(xf, cb.in);
  const vt::L2NormArgs args{1e-6f};

  std::vector<uint8_t> out_cpu(numel * vt::SizeOf(cb.out));
  Tensor tx = MakeT(const_cast<uint8_t*>(xb.data()), cb.in, Cpu(), shape);
  Tensor to = MakeT(out_cpu.data(), cb.out, Cpu(), shape);
  Queue cq = Q();
  vt::L2Norm(cq, to, tx, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, cb.in, shape, xb.data());
  DeviceTensor dout(gpu, gq.q, cb.out, shape);
  vt::L2Norm(gq.q, dout.tensor(), dx.tensor(), args);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  dout.Download(gq.q, out_gpu.data());

  CheckClose(Unpack(out_gpu, cb.out), Unpack(out_cpu, cb.out), cb.atol, cb.rtol);
}

// rmsnorm_gated CPU-vs-CUDA on one [T,D] shape.
void RunRmsNormGatedCudaCase(int64_t t, int64_t d, bool sigmoid_gate, const Combo& cb,
                             uint32_t seed) {
  const auto xf = RandomF32(static_cast<size_t>(t * d), seed);
  const auto zf = RandomF32(static_cast<size_t>(t * d), seed + 1);
  const auto wf = RandomF32(static_cast<size_t>(d), seed + 2, -1.0f, 1.0f);
  const auto xb = Pack(xf, cb.in);
  const auto zb = Pack(zf, cb.in);
  const auto wb = Pack(wf, cb.in);
  const RmsNormGatedArgs args{1e-6f, sigmoid_gate};

  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * d) * vt::SizeOf(cb.out));
  Tensor tx = MakeT(const_cast<uint8_t*>(xb.data()), cb.in, Cpu(), {t, d});
  Tensor tz = MakeT(const_cast<uint8_t*>(zb.data()), cb.in, Cpu(), {t, d});
  Tensor tw = MakeT(const_cast<uint8_t*>(wb.data()), cb.in, Cpu(), {d});
  Tensor to = MakeT(out_cpu.data(), cb.out, Cpu(), {t, d});
  Queue cq = Q();
  vt::RmsNormGated(cq, to, tx, tz, tw, args);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, cb.in, {t, d}, xb.data());
  DeviceTensor dz(gpu, gq.q, cb.in, {t, d}, zb.data());
  DeviceTensor dw(gpu, gq.q, cb.in, {d}, wb.data());
  DeviceTensor dout(gpu, gq.q, cb.out, {t, d});
  vt::RmsNormGated(gq.q, dout.tensor(), dx.tensor(), dz.tensor(), dw.tensor(), args);
  std::vector<uint8_t> out_gpu(out_cpu.size());
  dout.Download(gq.q, out_gpu.data());

  CheckClose(Unpack(out_gpu, cb.out), Unpack(out_cpu, cb.out), cb.atol, cb.rtol);
}

// VT_RMSNORM_GATED_FAST: the fast gated-RMSNorm decode kernel
// (RmsNormGatedRowFastKernel) is BIT-IDENTICAL (0-ulp) to the shipped
// RmsNormGatedRowKernel. It reproduces shipped's variance ORDER (per-element f32
// square + shared-memory binary tree; for d==128 each thread owns one element so the
// 128-thread tree == shipped's 256-thread tree byte-for-byte), inv=1.0f/sqrtf, and
// the same normalize/act multiply order; only the launch geometry (128 vs 256
// threads) and a single register-cached x load differ. So at the real GDN decode
// shape ([T*Hv, Dv=128], bf16, silu/sigmoid gate) the output is BIT-EXACT (0-ulp) vs
// shipped, guaranteeing fast-gated+fast-RMSNorm+cubin == the passing baseline on the
// 27B greedy near-tie (token-exactness adjudicated on DGX by test_qwen27_paged_engine
// 235/235 + test_qwen36_paged_engine 315/315). Runs both arms with EXPLICIT env
// values ("1" fast / "0" rollback; the launcher reads getenv per call) so the
// comparison is default-independent. Covers the contiguous rank-2 gate (gate_group=1)
// AND the padded rank-3 [T,Hv,Dv] strided gate (gate_group=Hv). Skips w/o CUDA.
void RunRmsNormGatedFastContiguous(int64_t rows, bool sigmoid_gate, uint32_t seed) {
  constexpr int64_t d = 128;  // Dv, the only production gated-norm shape
  // Adversarial: wide x/z ranges exercise variance magnitude and silu/sigmoid
  // saturation; w spans [-1,1] like the real norm weight.
  const auto xf = RandomF32(static_cast<size_t>(rows * d), seed, -6.0f, 6.0f);
  const auto zf = RandomF32(static_cast<size_t>(rows * d), seed + 1, -6.0f, 6.0f);
  const auto wf = RandomF32(static_cast<size_t>(d), seed + 2, -1.0f, 1.0f);
  const auto xb = Pack(xf, DType::kBF16);
  const auto zb = Pack(zf, DType::kBF16);
  const auto wb = Pack(wf, DType::kBF16);
  const RmsNormGatedArgs args{1e-6f, sigmoid_gate};

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, DType::kBF16, {rows, d}, xb.data());
  DeviceTensor dz(gpu, gq.q, DType::kBF16, {rows, d}, zb.data());
  DeviceTensor dw(gpu, gq.q, DType::kBF16, {d}, wb.data());

  auto run = [&](bool fast, std::vector<uint8_t>& out_bytes) {
    ::setenv("VT_RMSNORM_GATED_FAST", fast ? "1" : "0", 1);
    DeviceTensor dout(gpu, gq.q, DType::kBF16, {rows, d});
    vt::RmsNormGated(gq.q, dout.tensor(), dx.tensor(), dz.tensor(), dw.tensor(), args);
    out_bytes.resize(static_cast<size_t>(rows * d) * vt::SizeOf(DType::kBF16));
    dout.Download(gq.q, out_bytes.data());
  };
  std::vector<uint8_t> out_ref, out_fast;
  run(/*fast=*/false, out_ref);
  run(/*fast=*/true, out_fast);
  ::unsetenv("VT_RMSNORM_GATED_FAST");
  // BIT-EXACT (0-ulp): fast==shipped by construction.
  CheckClose(Unpack(out_fast, DType::kBF16), Unpack(out_ref, DType::kBF16), 0.0f, 0.0f);
}

// Gdn prefill/decode CPU-vs-CUDA. qsl empty => decode (batch single-token
// sequences); otherwise varlen prefill with batch ignored.
void RunGdnCudaCase(const std::vector<int32_t>& qsl, int64_t batch,
                    int64_t hk, int64_t hv, int64_t dk, int64_t dv,
                    const Combo& cb, uint32_t seed,
                    DType state_dtype = DType::kF32) {
  // Pin this CPU-vs-CUDA case to the SEQUENTIAL prefill scan (tight tol): the
  // chunk-parallel path reassociates the accumulation and is validated
  // separately against the sequential kernel below. Decode is always sequential.
  setenv("VT_GDN_CHUNKED", "0", 1);
  const bool decode = qsl.empty();
  const int64_t t = decode ? batch : qsl.back();
  const int64_t n = decode ? batch : static_cast<int64_t>(qsl.size()) - 1;
  const auto qf = RandomF32(static_cast<size_t>(t * hk * dk), seed, -1.0f, 1.0f);
  const auto kf = RandomF32(static_cast<size_t>(t * hk * dk), seed + 1, -1.0f, 1.0f);
  const auto vf = RandomF32(static_cast<size_t>(t * hv * dv), seed + 2, -1.0f, 1.0f);
  const auto gf = RandomF32(static_cast<size_t>(t * hv), seed + 3, -1.0f, 0.0f);
  const auto betaf = RandomF32(static_cast<size_t>(t * hv), seed + 4, 0.05f, 0.95f);
  const auto stf = RandomF32(static_cast<size_t>(n * hv * dv * dk), seed + 5, -0.5f, 0.5f);
  const auto qb = Pack(qf, cb.in);
  const auto kb = Pack(kf, cb.in);
  const auto vb = Pack(vf, cb.in);
  REQUIRE((state_dtype == DType::kF32 ||
           (decode && (state_dtype == DType::kF16 ||
                       state_dtype == DType::kBF16))));
  const std::vector<uint8_t> state_initial_bytes = Pack(stf, state_dtype);
  const std::vector<float> state_initial =
      Unpack(state_initial_bytes, state_dtype);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(dk))};

  std::vector<uint8_t> out_cpu(static_cast<size_t>(t * hv * dv) * vt::SizeOf(cb.out));
  std::vector<float> st_cpu = state_initial, g_cpu = gf, beta_cpu = betaf;
  std::vector<int32_t> qsl_cpu = qsl;
  Tensor tq = MakeT(const_cast<uint8_t*>(qb.data()), cb.in, Cpu(), {t, hk, dk});
  Tensor tk = MakeT(const_cast<uint8_t*>(kb.data()), cb.in, Cpu(), {t, hk, dk});
  Tensor tv = MakeT(const_cast<uint8_t*>(vb.data()), cb.in, Cpu(), {t, hv, dv});
  Tensor tg = MakeT(g_cpu.data(), DType::kF32, Cpu(), {t, hv});
  Tensor tbeta = MakeT(beta_cpu.data(), DType::kF32, Cpu(), {t, hv});
  Tensor ts = MakeT(st_cpu.data(), DType::kF32, Cpu(), {n, hv, dv, dk});
  Tensor to = MakeT(out_cpu.data(), cb.out, Cpu(), {t, hv, dv});
  Queue cq = Q();
  if (decode) {
    vt::GdnDecode(cq, to, tq, tk, tv, tg, tbeta, ts, args);
  } else {
    Tensor tqsl = MakeT(qsl_cpu.data(), DType::kI32, Cpu(), {n + 1});
    vt::GdnPrefill(cq, to, tq, tk, tv, tg, tbeta, ts, tqsl, args);
  }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dq(gpu, gq.q, cb.in, {t, hk, dk}, qb.data());
  DeviceTensor dkt(gpu, gq.q, cb.in, {t, hk, dk}, kb.data());
  DeviceTensor dvt(gpu, gq.q, cb.in, {t, hv, dv}, vb.data());
  DeviceTensor dg(gpu, gq.q, DType::kF32, {t, hv}, gf.data());
  DeviceTensor dbeta(gpu, gq.q, DType::kF32, {t, hv}, betaf.data());
  DeviceTensor dst(gpu, gq.q, state_dtype, {n, hv, dv, dk},
                   state_initial_bytes.data());
  DeviceTensor dout(gpu, gq.q, cb.out, {t, hv, dv});
  if (decode) {
    vt::GdnDecode(gq.q, dout.tensor(), dq.tensor(), dkt.tensor(), dvt.tensor(), dg.tensor(),
                  dbeta.tensor(), dst.tensor(), args);
  } else {
    DeviceTensor dqsl(gpu, gq.q, DType::kI32, {n + 1}, qsl.data());
    vt::GdnPrefill(gq.q, dout.tensor(), dq.tensor(), dkt.tensor(), dvt.tensor(), dg.tensor(),
                   dbeta.tensor(), dst.tensor(), dqsl.tensor(), args);
  }
  std::vector<uint8_t> out_gpu(out_cpu.size());
  std::vector<uint8_t> st_gpu_bytes(state_initial_bytes.size());
  dout.Download(gq.q, out_gpu.data());
  dst.Download(gq.q, st_gpu_bytes.data());
  const std::vector<float> st_gpu = Unpack(st_gpu_bytes, state_dtype);

  const float state_tol =
      state_dtype == DType::kF32 ? 1e-5f : 3e-2f;
  CheckClose(Unpack(out_gpu, cb.out), Unpack(out_cpu, cb.out),
             std::max(cb.atol, state_tol), std::max(cb.rtol, state_tol));
  CheckClose(st_gpu, st_cpu, state_tol, state_tol);
}

// Chunk-parallel prefill (VT_GDN_CHUNKED, default) vs the sequential scan
// (VT_GDN_CHUNKED=0), same inputs, same binary. The chunked path is the M2
// prefill perf kernel (cuda_gdn.cu GdnChunk*); it must reproduce the sequential
// recurrence within GDN tolerance (fp reassociation only). Covers >=2 chunks +
// a partial tail and multiple heads / varlen sequences.
struct GdnDiffStats {
  float output_max = 0.0f;
  double output_mean = 0.0;
  float state_max = 0.0f;
  double state_mean = 0.0;
};

std::pair<float, double> AbsoluteDiffStats(const std::vector<float>& got,
                                           const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  float max_error = 0.0f;
  double sum_error = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float error = std::fabs(got[i] - want[i]);
    max_error = std::max(max_error, error);
    sum_error += static_cast<double>(error);
  }
  return {max_error, got.empty() ? 0.0 : sum_error / static_cast<double>(got.size())};
}

// ---------------------------------------------------------------------------
// W2 merged-qkvz strided consumers. vLLM's Qwen3.5/3.6 GDN issues ONE
// in_proj_qkvz GEMM and splits its [T, key*2+value*2] output into mixed_qkv and
// z last-dimension slices (qwen_gdn_linear_attn.py:929-936). Our merge feeds the
// conv (mixed_qkv) and the gated RMSNorm (z) padded-row (inner-contiguous, outer
// stride > width) views of that single output without any copy. These pin that
// the strided views are bit-identical to the contiguous split-projection inputs.

// A rank-3 [T,Hv,Dv] gate view with a padded per-token (outer) stride; the inner
// [Hv,Dv] block stays contiguous, exactly as z = mixed_qkvz[:, conv_dim:] views.
Tensor Rank3RowView(void* data, DType dtype, Device device, int64_t t, int64_t hv,
                    int64_t dv, int64_t row_stride) {
  Tensor x;
  x.data = data;
  x.dtype = dtype;
  x.device = device;
  x.rank = 3;
  x.shape[0] = t;
  x.shape[1] = hv;
  x.shape[2] = dv;
  x.stride[0] = row_stride;
  x.stride[1] = dv;
  x.stride[2] = 1;
  return x;
}

// Padded rank-3 [T,Hv,Dv=128] strided gate (the real z-slice geometry: the gate is a
// padded-row view of the merged [T, qkvz_width] projection, gate_group=Hv). Proves
// the VT_RMSNORM_GATED_FAST fast kernel's gate addressing is bit-identical to shipped
// under a non-unit gate_group / padded gate_outer. fast==shipped 0-ulp.
void RunRmsNormGatedFastPaddedRank3(int64_t t, int64_t hv, bool sigmoid_gate, uint32_t seed) {
  constexpr int64_t d = 128;  // Dv
  const int64_t rows = t * hv, gate_cols = hv * d, stride = gate_cols + 11;  // pad canary tail
  const auto xf = RandomF32(static_cast<size_t>(rows * d), seed, -6.0f, 6.0f);
  const auto zf = RandomF32(static_cast<size_t>(t * gate_cols), seed + 1, -6.0f, 6.0f);
  const auto wf = RandomF32(static_cast<size_t>(d), seed + 2, -1.0f, 1.0f);
  const auto xb = Pack(xf, DType::kBF16);
  const auto wb = Pack(wf, DType::kBF16);
  // Padded z: each token's Hv*Dv gate lands in the first gate_cols of a wider row.
  std::vector<float> z_pad(static_cast<size_t>(t) * stride, -3.5f);
  for (int64_t tok = 0; tok < t; ++tok)
    for (int64_t j = 0; j < gate_cols; ++j)
      z_pad[tok * stride + j] = zf[static_cast<size_t>(tok * gate_cols + j)];
  const auto zpb = Pack(z_pad, DType::kBF16);
  const RmsNormGatedArgs args{1e-6f, sigmoid_gate};

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dx(gpu, gq.q, DType::kBF16, {t, hv, d}, xb.data());
  DeviceTensor dz(gpu, gq.q, DType::kBF16, {t, stride}, zpb.data());
  DeviceTensor dw(gpu, gq.q, DType::kBF16, {d}, wb.data());
  Tensor tz3 = Rank3RowView(dz.tensor().data, DType::kBF16, Gpu(), t, hv, d, stride);

  auto run = [&](bool fast, std::vector<uint8_t>& out_bytes) {
    ::setenv("VT_RMSNORM_GATED_FAST", fast ? "1" : "0", 1);
    DeviceTensor dout(gpu, gq.q, DType::kBF16, {t, hv, d});
    vt::RmsNormGated(gq.q, dout.tensor(), dx.tensor(), tz3, dw.tensor(), args);
    out_bytes.resize(static_cast<size_t>(rows * d) * vt::SizeOf(DType::kBF16));
    dout.Download(gq.q, out_bytes.data());
  };
  std::vector<uint8_t> out_ref, out_fast;
  run(/*fast=*/false, out_ref);
  run(/*fast=*/true, out_fast);
  ::unsetenv("VT_RMSNORM_GATED_FAST");
  CheckClose(Unpack(out_fast, DType::kBF16), Unpack(out_ref, DType::kBF16), 0.0f, 0.0f);
}

TEST_CASE("causal_conv1d_fwd: padded-row strided x matches contiguous") {
  constexpr int64_t kT = 5, kC = 6, kK = 4, kPad = 7;  // stride = kC + kPad
  const auto xf = RandomF32(static_cast<size_t>(kT * kC), 4242);
  std::vector<float> w = RandomF32(static_cast<size_t>(kC * kK), 4243);
  std::vector<float> st0 = RandomF32(static_cast<size_t>(kC * (kK - 1)), 4244);
  std::vector<int32_t> qsl = {0, kT}, his = {1};
  Queue q = Q();
  // Contiguous reference.
  std::vector<float> x_c(xf), st_c(st0), out_c(static_cast<size_t>(kT * kC), 0.0f);
  Tensor txc = T2(x_c, kT, kC), tw = T2(w, kC, kK), tsc = T3(st_c, 1, kC, kK - 1);
  Tensor toc = T2(out_c, kT, kC), tqsl = Ti(qsl), this_ = Ti(his);
  vt::CausalConv1dFwd(q, toc, txc, tw, nullptr, tsc, tqsl, this_, CausalConv1dArgs{true});
  // Padded-row strided x: each token row lands in the first kC cols of a wider
  // buffer; the tail columns are canaries that must stay untouched.
  const int64_t stride = kC + kPad;
  std::vector<float> x_p(static_cast<size_t>(kT) * stride, -12345.0f);
  for (int64_t t = 0; t < kT; ++t)
    for (int64_t c = 0; c < kC; ++c) x_p[t * stride + c] = xf[static_cast<size_t>(t * kC + c)];
  std::vector<float> st_p(st0), out_p(static_cast<size_t>(kT * kC), 0.0f);
  Tensor txp = RowView(x_p.data(), DType::kF32, Cpu(), kT, kC, stride);
  Tensor tsp = T3(st_p, 1, kC, kK - 1), top = T2(out_p, kT, kC);
  vt::CausalConv1dFwd(q, top, txp, tw, nullptr, tsp, tqsl, this_, CausalConv1dArgs{true});
  for (size_t i = 0; i < out_c.size(); ++i) CHECK(out_p[i] == out_c[i]);
  for (size_t i = 0; i < st_c.size(); ++i) CHECK(st_p[i] == st_c[i]);
  for (int64_t t = 0; t < kT; ++t)
    for (int64_t c = kC; c < stride; ++c) CHECK(x_p[t * stride + c] == -12345.0f);
}

TEST_CASE("causal_conv1d_update: padded-row strided x matches contiguous") {
  // B=4 single-token sequences, C=8, K=4, padded-row x view.
  constexpr int64_t kB = 4, kC = 8, kK = 4, kPad = 5;
  const auto xf = RandomF32(static_cast<size_t>(kB * kC), 7001);
  std::vector<float> w = RandomF32(static_cast<size_t>(kC * kK), 7002);
  std::vector<float> st0 = RandomF32(static_cast<size_t>(kB * kC * (kK - 1)), 7003);
  Queue q = Q();
  std::vector<float> x_c(xf), st_c(st0), out_c(static_cast<size_t>(kB * kC), 0.0f);
  Tensor txc = T2(x_c, kB, kC), tw = T2(w, kC, kK), tsc = T3(st_c, kB, kC, kK - 1);
  Tensor toc = T2(out_c, kB, kC);
  vt::CausalConv1dUpdate(q, toc, txc, tw, nullptr, tsc, CausalConv1dArgs{true});
  const int64_t stride = kC + kPad;
  std::vector<float> x_p(static_cast<size_t>(kB) * stride, -777.0f);
  for (int64_t b = 0; b < kB; ++b)
    for (int64_t c = 0; c < kC; ++c) x_p[b * stride + c] = xf[static_cast<size_t>(b * kC + c)];
  std::vector<float> st_p(st0), out_p(static_cast<size_t>(kB * kC), 0.0f);
  Tensor txp = RowView(x_p.data(), DType::kF32, Cpu(), kB, kC, stride);
  Tensor tsp = T3(st_p, kB, kC, kK - 1), top = T2(out_p, kB, kC);
  vt::CausalConv1dUpdate(q, top, txp, tw, nullptr, tsp, CausalConv1dArgs{true});
  for (size_t i = 0; i < out_c.size(); ++i) CHECK(out_p[i] == out_c[i]);
  for (size_t i = 0; i < st_c.size(); ++i) CHECK(st_p[i] == st_c[i]);
  for (int64_t b = 0; b < kB; ++b)
    for (int64_t c = kC; c < stride; ++c) CHECK(x_p[b * stride + c] == -777.0f);
}

TEST_CASE("rmsnorm_gated: rank-3 padded-row strided gate matches contiguous") {
  // Real z geometry: T tokens, Hv heads, Dv per head; the gate is the padded-row
  // z slice of the merged [T, qkvz_width] projection output.
  constexpr int64_t kT = 4, kHv = 3, kDv = 5, kPad = 11;
  const int64_t rows = kT * kHv, gate_cols = kHv * kDv;
  const auto xf = RandomF32(static_cast<size_t>(rows * kDv), 9101);
  const auto zf = RandomF32(static_cast<size_t>(kT * gate_cols), 9102);
  std::vector<float> w = RandomF32(static_cast<size_t>(kDv), 9103);
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {kDv});
  Queue q = Q();
  // Contiguous rank-2 [rows, Dv] reference.
  std::vector<float> x_c(xf), z_c(zf), out_c(static_cast<size_t>(rows * kDv), 0.0f);
  Tensor txc = T2(x_c, rows, kDv), tzc = T2(z_c, rows, kDv), toc = T2(out_c, rows, kDv);
  vt::RmsNormGated(q, toc, txc, tzc, tw, RmsNormGatedArgs{0.0f, false});
  // Rank-3 [T,Hv,Dv] with x/out contiguous and the gate a padded-token view.
  const int64_t stride = gate_cols + kPad;
  std::vector<float> z_p(static_cast<size_t>(kT) * stride, -3.5f);
  for (int64_t t = 0; t < kT; ++t)
    for (int64_t j = 0; j < gate_cols; ++j)
      z_p[t * stride + j] = zf[static_cast<size_t>(t * gate_cols + j)];
  std::vector<float> x_p(xf), out_p(static_cast<size_t>(rows * kDv), 0.0f);
  Tensor txp = T3(x_p, kT, kHv, kDv), top = T3(out_p, kT, kHv, kDv);
  Tensor tzp = Rank3RowView(z_p.data(), DType::kF32, Cpu(), kT, kHv, kDv, stride);
  vt::RmsNormGated(q, top, txp, tzp, tw, RmsNormGatedArgs{0.0f, false});
  for (size_t i = 0; i < out_c.size(); ++i) CHECK(out_p[i] == out_c[i]);
  for (int64_t t = 0; t < kT; ++t)
    for (int64_t j = gate_cols; j < stride; ++j) CHECK(z_p[t * stride + j] == -3.5f);
}

#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
void CheckUpstreamCutedslTolerances(const GdnDiffStats& stats) {
  CAPTURE(stats.output_max);
  CAPTURE(stats.output_mean);
  CAPTURE(stats.state_max);
  CAPTURE(stats.state_mean);
  // Pinned upstream executable contract:
  // tests/kernels/mamba/test_gdn_prefill_cutedsl.py:163-166 @ e24d1b24.
  CHECK(stats.output_max < 2e-3f);
  CHECK(stats.output_mean < 6e-5);
  CHECK(stats.state_max < 2e-2f);
  CHECK(stats.state_mean < 6e-4);
}
#endif

GdnDiffStats RunGdnChunkedVsSequentialOnQueue(Backend& gpu, Queue& queue,
                                               const std::vector<int32_t>& qsl, int64_t hk,
                                               int64_t hv, int64_t dk, int64_t dv,
                                               const Combo& cb, uint32_t seed, float atol,
                                               float rtol) {
  const int64_t t = qsl.back();
  const int64_t n = static_cast<int64_t>(qsl.size()) - 1;
  auto qf = RandomF32(static_cast<size_t>(t * hk * dk), seed, -1.0f, 1.0f);
  auto kf = RandomF32(static_cast<size_t>(t * hk * dk), seed + 1, -1.0f, 1.0f);
  // L2-normalize q/k per (token, head) over Dk, exactly as the real GDN path
  // does before the scan. Un-normalized k makes the delta-rule recurrence
  // (S += outer(v - S@k, k)) explode exponentially — an ill-conditioned input
  // both scans blow up on identically; normalization keeps it contractive so
  // the chunked-vs-sequential comparison measures the kernel, not fp chaos.
  auto l2norm = [&](std::vector<float>& x) {
    for (int64_t r = 0; r < t * hk; ++r) {
      double ss = 0.0;
      for (int64_t d = 0; d < dk; ++d) ss += static_cast<double>(x[r * dk + d]) * x[r * dk + d];
      const float inv = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));
      for (int64_t d = 0; d < dk; ++d) x[r * dk + d] *= inv;
    }
  };
  l2norm(qf);
  l2norm(kf);
  const auto vf = RandomF32(static_cast<size_t>(t * hv * dv), seed + 2, -1.0f, 1.0f);
  // g in [-0.5, 0]: a realistic log-decay range; cumulative over a chunk stays
  // well within f32 exp range so both paths are numerically sane.
  const auto gf = RandomF32(static_cast<size_t>(t * hv), seed + 3, -0.5f, 0.0f);
  const auto betaf = RandomF32(static_cast<size_t>(t * hv), seed + 4, 0.05f, 0.95f);
  const auto stf = RandomF32(static_cast<size_t>(n * hv * dv * dk), seed + 5, -0.5f, 0.5f);
  const auto qb = Pack(qf, cb.in);
  const auto kb = Pack(kf, cb.in);
  const auto vb = Pack(vf, cb.in);
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(dk))};

  std::vector<uint8_t> out_seq(static_cast<size_t>(t * hv * dv) * vt::SizeOf(cb.out));
  std::vector<uint8_t> out_chunk(out_seq.size());
  std::vector<float> st_seq(stf.size()), st_chunk(stf.size());

  auto run = [&](const char* toggle, std::vector<uint8_t>& out_bytes, std::vector<float>& st_out) {
    setenv("VT_GDN_CHUNKED", toggle, 1);
    DeviceTensor dq(gpu, queue, cb.in, {t, hk, dk}, qb.data());
    DeviceTensor dkt(gpu, queue, cb.in, {t, hk, dk}, kb.data());
    DeviceTensor dvt(gpu, queue, cb.in, {t, hv, dv}, vb.data());
    DeviceTensor dg(gpu, queue, DType::kF32, {t, hv}, gf.data());
    DeviceTensor dbeta(gpu, queue, DType::kF32, {t, hv}, betaf.data());
    DeviceTensor dst(gpu, queue, DType::kF32, {n, hv, dv, dk}, stf.data());
    DeviceTensor dout(gpu, queue, cb.out, {t, hv, dv});
    DeviceTensor dqsl(gpu, queue, DType::kI32, {n + 1}, qsl.data());
    vt::GdnPrefill(queue, dout.tensor(), dq.tensor(), dkt.tensor(), dvt.tensor(), dg.tensor(),
                   dbeta.tensor(), dst.tensor(), dqsl.tensor(), args);
    dout.Download(queue, out_bytes.data());
    dst.Download(queue, st_out.data());
  };
  run("0", out_seq, st_seq);
  run("1", out_chunk, st_chunk);

  const auto out_chunk_f = Unpack(out_chunk, cb.out);
  const auto out_seq_f = Unpack(out_seq, cb.out);
  CheckClose(out_chunk_f, out_seq_f, atol, rtol);
  CheckClose(st_chunk, st_seq, atol, rtol);
  const auto [output_max, output_mean] = AbsoluteDiffStats(out_chunk_f, out_seq_f);
  const auto [state_max, state_mean] = AbsoluteDiffStats(st_chunk, st_seq);
  return GdnDiffStats{output_max, output_mean, state_max, state_mean};
}

void RunGdnChunkedVsSequential(const std::vector<int32_t>& qsl, int64_t hk, int64_t hv,
                               int64_t dk, int64_t dv, const Combo& cb, uint32_t seed,
                               float atol, float rtol) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  (void)RunGdnChunkedVsSequentialOnQueue(gpu, gq.q, qsl, hk, hv, dk, dv, cb, seed, atol,
                                         rtol);
}

}  // namespace

// Ported behavior from tests/v1/worker/test_mamba_utils.py:342-358: gather
// non-contiguous persistent cache rows, zero fresh requests, then scatter only
// the selected rows back. Both cache dtypes exercise the exact cache boundary;
// i8 and i32 masks cover upstream bool metadata and standalone callers.
TEST_CASE("gdn indexed state I/O: dtype conversion, fresh zero, untouched slots") {
  constexpr int64_t kSlots = 5, kRows = 3, kRowElems = 6;
  const std::vector<int32_t> idx = {3, 1, 4};
  const std::vector<int8_t> has_i8 = {1, 0, 1};
  const std::vector<int32_t> has_i32 = {1, 0, 1};
  std::vector<float> master(static_cast<size_t>(kSlots * kRowElems));
  for (int64_t i = 0; i < static_cast<int64_t>(master.size()); ++i)
    master[static_cast<size_t>(i)] = static_cast<float>(i) * 0.125f - 2.0f;

  for (DType cache_dtype : {DType::kF32, DType::kF16, DType::kBF16}) {
    CAPTURE(static_cast<int>(cache_dtype));
    std::vector<uint8_t> cache_bytes = Pack(master, cache_dtype);
    const std::vector<float> cache_initial = Unpack(cache_bytes, cache_dtype);
    std::vector<float> working(static_cast<size_t>(kRows * kRowElems), -99.0f);
    std::vector<int32_t> idx_copy = idx;
    std::vector<int8_t> has8 = has_i8;
    std::vector<int32_t> has32 = has_i32;
    Tensor cache = MakeT(cache_bytes.data(), cache_dtype, Cpu(),
                         {kSlots, 2, 3});
    Tensor work = MakeT(working.data(), DType::kF32, Cpu(), {kRows, 2, 3});
    Tensor tidx = MakeT(idx_copy.data(), DType::kI32, Cpu(), {kRows});
    Tensor mask = cache_dtype == DType::kBF16
                      ? MakeT(has8.data(), DType::kI8, Cpu(), {kRows})
                      : MakeT(has32.data(), DType::kI32, Cpu(), {kRows});
    Queue q = Q();
    vt::GdnStateGather(q, work, cache, tidx, &mask);

    std::vector<float> want_work(working.size(), 0.0f);
    for (int64_t r = 0; r < kRows; ++r) {
      if (has_i8[static_cast<size_t>(r)] == 0) continue;
      for (int64_t e = 0; e < kRowElems; ++e) {
        want_work[static_cast<size_t>(r * kRowElems + e)] =
            cache_initial[static_cast<size_t>(idx[static_cast<size_t>(r)] *
                                              kRowElems + e)];
      }
    }
    CHECK(working == want_work);

    for (int64_t i = 0; i < static_cast<int64_t>(working.size()); ++i)
      working[static_cast<size_t>(i)] = 10.0f + static_cast<float>(i) * 0.25f;
    vt::GdnStateScatter(q, cache, work, tidx);
    std::vector<float> want_cache = cache_initial;
    const std::vector<float> stored = Unpack(Pack(working, cache_dtype), cache_dtype);
    for (int64_t r = 0; r < kRows; ++r) {
      for (int64_t e = 0; e < kRowElems; ++e) {
        want_cache[static_cast<size_t>(idx[static_cast<size_t>(r)] *
                                       kRowElems + e)] =
            stored[static_cast<size_t>(r * kRowElems + e)];
      }
    }
    CHECK(Unpack(cache_bytes, cache_dtype) == want_cache);
  }

  std::vector<float> cache(12, 0.0f), working(2, 0.0f);
  std::vector<int32_t> bad_idx = {6};
  Tensor tc = MakeT(cache.data(), DType::kF32, Cpu(), {6, 2});
  Tensor tw = MakeT(working.data(), DType::kF32, Cpu(), {1, 2});
  Tensor ti = MakeT(bad_idx.data(), DType::kI32, Cpu(), {1});
  Queue q = Q();
  CHECK_THROWS_AS(vt::GdnStateGather(q, tw, tc, ti), std::runtime_error);
  CHECK_THROWS_AS(vt::GdnStateScatter(q, tc, tw, ti), std::runtime_error);
}

TEST_CASE("CUDA gdn indexed state I/O matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  constexpr int64_t kSlots = 7, kRows = 3, kRowElems = 24;
  const std::vector<int32_t> idx = {5, 1, 6};
  const std::vector<int8_t> has = {1, 0, 1};
  const std::vector<float> master =
      RandomF32(static_cast<size_t>(kSlots * kRowElems), 8120);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  for (DType cache_dtype : {DType::kF16, DType::kBF16}) {
    CAPTURE(static_cast<int>(cache_dtype));
    const std::vector<uint8_t> packed = Pack(master, cache_dtype);
    const std::vector<float> initial = Unpack(packed, cache_dtype);
    std::vector<float> want_work(static_cast<size_t>(kRows * kRowElems),
                                 0.0f);
    for (int64_t r = 0; r < kRows; ++r) {
      if (has[static_cast<size_t>(r)] == 0) continue;
      for (int64_t e = 0; e < kRowElems; ++e)
        want_work[static_cast<size_t>(r * kRowElems + e)] =
            initial[static_cast<size_t>(
                idx[static_cast<size_t>(r)] * kRowElems + e)];
    }

    DeviceTensor dcache(gpu, gq.q, cache_dtype, {kSlots, 2, 3, 4},
                        packed.data());
    DeviceTensor didx(gpu, gq.q, DType::kI32, {kRows}, idx.data());
    DeviceTensor dhas(gpu, gq.q, DType::kI8, {kRows}, has.data());
    DeviceTensor dwork(gpu, gq.q, DType::kF32, {kRows, 2, 3, 4});
    vt::GdnStateGather(gq.q, dwork.tensor(), dcache.tensor(), didx.tensor(),
                       &dhas.tensor());
    std::vector<float> got_work(want_work.size());
    dwork.Download(gq.q, got_work.data());
    CHECK(got_work == want_work);

    const std::vector<float> replacement = RandomF32(want_work.size(), 8130);
    DeviceTensor dreplacement(gpu, gq.q, DType::kF32, {kRows, 2, 3, 4},
                              replacement.data());
    vt::GdnStateScatter(gq.q, dcache.tensor(), dreplacement.tensor(),
                        didx.tensor());
    std::vector<uint8_t> got_cache(packed.size());
    dcache.Download(gq.q, got_cache.data());
    std::vector<float> want_cache = initial;
    const std::vector<float> stored =
        Unpack(Pack(replacement, cache_dtype), cache_dtype);
    for (int64_t r = 0; r < kRows; ++r) {
      for (int64_t e = 0; e < kRowElems; ++e)
        want_cache[static_cast<size_t>(
            idx[static_cast<size_t>(r)] * kRowElems + e)] =
            stored[static_cast<size_t>(r * kRowElems + e)];
    }
    CHECK(Unpack(got_cache, cache_dtype) == want_cache);
  }
}

TEST_CASE("CUDA gdn decode corner fallback supports indexed compressed state") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }

  constexpr int64_t n = 2, slots = 3, hk = 1, hv = 1, dk = 512,
                    dv = 32;
  constexpr float scale = 0.0441941738f;  // 1/sqrt(512)
  const std::vector<int32_t> idx = {2, 0};
  const std::vector<float> qv = RandomF32(n * hk * dk, 8191, -0.2f, 0.2f);
  const std::vector<float> kv = RandomF32(n * hk * dk, 8192, -0.2f, 0.2f);
  const std::vector<float> vv = RandomF32(n * hv * dv, 8193, -0.5f, 0.5f);
  const std::vector<float> gv = RandomF32(n * hv, 8194, -0.3f, -0.01f);
  const std::vector<float> bv = RandomF32(n * hv, 8195, 0.1f, 0.9f);
  const std::vector<float> state_master =
      RandomF32(slots * hv * dv * dk, 8196, -0.25f, 0.25f);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  for (DType state_dtype : {DType::kF16, DType::kBF16}) {
    CAPTURE(static_cast<int>(state_dtype));
    const std::vector<uint8_t> state_bytes = Pack(state_master, state_dtype);
    const std::vector<float> state_initial = Unpack(state_bytes, state_dtype);
    const int64_t row_elems = hv * dv * dk;

    std::vector<float> compact(static_cast<size_t>(n * row_elems));
    for (int64_t row = 0; row < n; ++row) {
      const int64_t slot = idx[static_cast<size_t>(row)];
      std::copy_n(state_initial.begin() + slot * row_elems, row_elems,
                  compact.begin() + row * row_elems);
    }
    std::vector<float> ref_out(static_cast<size_t>(n * hv * dv), 0.0f);
    std::vector<float> q_copy = qv, k_copy = kv, v_copy = vv, g_copy = gv,
                       b_copy = bv;
    Tensor tq = T3(q_copy, n, hk, dk), tk = T3(k_copy, n, hk, dk),
           tv = T3(v_copy, n, hv, dv), tg = T2(g_copy, n, hv),
           tb = T2(b_copy, n, hv), ts = T4(compact, n, hv, dv, dk),
           to = T3(ref_out, n, hv, dv);
    Queue cpu_q = Q();
    vt::GdnDecode(cpu_q, to, tq, tk, tv, tg, tb, ts, GdnArgs{scale});

    DeviceTensor dq(gpu, gq.q, DType::kF32, {n, hk, dk}, qv.data());
    DeviceTensor dkv(gpu, gq.q, DType::kF32, {n, hk, dk}, kv.data());
    DeviceTensor dvv(gpu, gq.q, DType::kF32, {n, hv, dv}, vv.data());
    DeviceTensor dg(gpu, gq.q, DType::kF32, {n, hv}, gv.data());
    DeviceTensor db(gpu, gq.q, DType::kF32, {n, hv}, bv.data());
    DeviceTensor ds(gpu, gq.q, state_dtype, {slots, hv, dv, dk},
                    state_bytes.data());
    DeviceTensor dout(gpu, gq.q, DType::kF32, {n, hv, dv});
    DeviceTensor didx(gpu, gq.q, DType::kI32, {n}, idx.data());
    vt::GdnDecode(gq.q, dout.tensor(), dq.tensor(), dkv.tensor(), dvv.tensor(),
                  dg.tensor(), db.tensor(), ds.tensor(), GdnArgs{scale},
                  &didx.tensor());

    std::vector<float> got_out(ref_out.size());
    std::vector<uint8_t> got_state(state_bytes.size());
    dout.Download(gq.q, got_out.data());
    ds.Download(gq.q, got_state.data());
    CheckClose(got_out, ref_out, 3e-2f, 3e-2f);

    std::vector<float> want_state = state_initial;
    const std::vector<float> rounded_compact =
        Unpack(Pack(compact, state_dtype), state_dtype);
    for (int64_t row = 0; row < n; ++row) {
      const int64_t slot = idx[static_cast<size_t>(row)];
      std::copy_n(rounded_compact.begin() + row * row_elems, row_elems,
                  want_state.begin() + slot * row_elems);
    }
    CheckClose(Unpack(got_state, state_dtype), want_state, 3e-2f, 3e-2f);
  }
}

TEST_CASE("CUDA causal_conv1d_fwd matches CPU (varlen, bias, dtypes)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 5000;
  for (const Combo& cb : kCudaCombos) {
    CAPTURE(static_cast<int>(cb.in));
    CAPTURE(static_cast<int>(cb.out));
    RunConvFwdCudaCase({0, 5, 6, 9}, {1, 0, 1}, 10, 4, /*with_bias=*/true, /*silu=*/true, cb,
                       seed);
    seed += 10;
  }
  // T < K-1 write-back paths (shifted old state / zero left-pad), linear.
  RunConvFwdCudaCase({0, 1, 3}, {1, 0}, 3, 4, /*with_bias=*/false, /*silu=*/false,
                     kCudaCombos[0], seed);
  // Real conv dim (512 channels spans multiple thread blocks per sequence).
  RunConvFwdCudaCase({0, 9, 16}, {1, 1}, 512, 4, /*with_bias=*/false, /*silu=*/true,
                     kCudaCombos[0], seed + 10);
  // W1 persistent metadata uploads the upstream bool mask as i8.
  RunConvFwdCudaCase({0, 5, 9}, {1, 0}, 16, 4, /*with_bias=*/false,
                     /*silu=*/true, kCudaCombos[0], seed + 20,
                     /*i8_mask=*/true);
}

TEST_CASE("CUDA causal_conv1d_update matches CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 5200;
  for (const Combo& cb : kCudaCombos) {
    CAPTURE(static_cast<int>(cb.in));
    CAPTURE(static_cast<int>(cb.out));
    RunConvUpdateCudaCase(4, 7, 4, /*with_bias=*/true, /*silu=*/true, cb, seed);
    seed += 10;
  }
  RunConvUpdateCudaCase(3, 512, 4, /*with_bias=*/false, /*silu=*/true, kCudaCombos[0], seed);
}

TEST_CASE("CUDA l2norm matches CPU (rank 2 and 3)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 5400;
  for (const Combo& cb : kCudaCombos) {
    CAPTURE(static_cast<int>(cb.in));
    CAPTURE(static_cast<int>(cb.out));
    RunL2NormCudaCase({5, 127}, cb, seed);
    RunL2NormCudaCase({3, 2, 128}, cb, seed + 1);
    seed += 10;
  }
  // Row longer than the block (300 > 256) exercises the strided reduction.
  RunL2NormCudaCase({2, 300}, kCudaCombos[0], seed);
}

// gdn_post_conv: (1) CPU fused == CPU composed (GdnConvSplit + GdnGBeta +
// L2Norm x2), bit-exact; (2) CUDA fused == CPU fused, within tol. Baseline f32.
void RunGdnPostConvCase(int64_t t, int64_t hk, int64_t hv, int64_t dk, int64_t dv, uint32_t seed) {
  const int64_t key_dim = hk * dk, value_dim = hv * dv, conv_dim = 2 * key_dim + value_dim;
  const auto conv = RandomF32(static_cast<size_t>(t * conv_dim), seed, -1.5f, 1.5f);
  const auto araw = RandomF32(static_cast<size_t>(t * hv), seed + 1, -1.0f, 1.0f);
  const auto braw = RandomF32(static_cast<size_t>(t * hv), seed + 2, -1.0f, 1.0f);
  const auto alog = RandomF32(static_cast<size_t>(hv), seed + 3, -1.0f, 1.0f);
  const auto dtb = RandomF32(static_cast<size_t>(hv), seed + 4, -1.0f, 1.0f);
  const vt::L2NormArgs args{1e-6f};
  Queue cq = Q();
  Tensor tconv = MakeT(const_cast<float*>(conv.data()), DType::kF32, Cpu(), {t, conv_dim});
  Tensor taraw = MakeT(const_cast<float*>(araw.data()), DType::kF32, Cpu(), {t, hv});
  Tensor tbraw = MakeT(const_cast<float*>(braw.data()), DType::kF32, Cpu(), {t, hv});
  Tensor talog = MakeT(const_cast<float*>(alog.data()), DType::kF32, Cpu(), {hv});
  Tensor tdtb = MakeT(const_cast<float*>(dtb.data()), DType::kF32, Cpu(), {hv});

  // Composed CPU reference.
  std::vector<float> rq(t * hk * dk), rk(t * hk * dk), rv(t * value_dim), rg(t * hv), rb(t * hv);
  {
    std::vector<float> sq(t * key_dim), sk(t * key_dim);
    Tensor tsq = MakeT(sq.data(), DType::kF32, Cpu(), {t, key_dim});
    Tensor tsk = MakeT(sk.data(), DType::kF32, Cpu(), {t, key_dim});
    Tensor tv = MakeT(rv.data(), DType::kF32, Cpu(), {t, value_dim});
    vt::GdnConvSplit(cq, tsq, tsk, tv, tconv);
    Tensor tg = MakeT(rg.data(), DType::kF32, Cpu(), {t, hv});
    Tensor tbo = MakeT(rb.data(), DType::kF32, Cpu(), {t, hv});
    vt::GdnGBeta(cq, tg, tbo, taraw, tbraw, talog, tdtb);
    Tensor tq3 = MakeT(sq.data(), DType::kF32, Cpu(), {t, hk, dk});
    Tensor tk3 = MakeT(sk.data(), DType::kF32, Cpu(), {t, hk, dk});
    Tensor trq = MakeT(rq.data(), DType::kF32, Cpu(), {t, hk, dk});
    Tensor trk = MakeT(rk.data(), DType::kF32, Cpu(), {t, hk, dk});
    vt::L2Norm(cq, trq, tq3, args);
    vt::L2Norm(cq, trk, tk3, args);
  }

  // Fused CPU.
  std::vector<float> fq(t * hk * dk), fk(t * hk * dk), fv(t * value_dim), fg(t * hv), fb(t * hv);
  Tensor tfq = MakeT(fq.data(), DType::kF32, Cpu(), {t, hk, dk});
  Tensor tfk = MakeT(fk.data(), DType::kF32, Cpu(), {t, hk, dk});
  Tensor tfv = MakeT(fv.data(), DType::kF32, Cpu(), {t, hv, dv});
  Tensor tfg = MakeT(fg.data(), DType::kF32, Cpu(), {t, hv});
  Tensor tfb = MakeT(fb.data(), DType::kF32, Cpu(), {t, hv});
  vt::GdnPostConv(cq, tfq, tfk, tfv, tfg, tfb, tconv, taraw, tbraw, talog, tdtb, args);
  CheckClose(fq, rq, 0.0f, 0.0f);  // bit-exact vs composed
  CheckClose(fk, rk, 0.0f, 0.0f);
  CheckClose(fv, rv, 0.0f, 0.0f);
  CheckClose(fg, rg, 0.0f, 0.0f);
  CheckClose(fb, rb, 0.0f, 0.0f);

  if (!HasCuda()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dconv(gpu, gq.q, DType::kF32, {t, conv_dim}, conv.data());
  DeviceTensor daraw(gpu, gq.q, DType::kF32, {t, hv}, araw.data());
  DeviceTensor dbraw(gpu, gq.q, DType::kF32, {t, hv}, braw.data());
  DeviceTensor dalog(gpu, gq.q, DType::kF32, {hv}, alog.data());
  DeviceTensor ddtb(gpu, gq.q, DType::kF32, {hv}, dtb.data());
  DeviceTensor dgq(gpu, gq.q, DType::kF32, {t, hk, dk});
  DeviceTensor dgk(gpu, gq.q, DType::kF32, {t, hk, dk});
  DeviceTensor dgv(gpu, gq.q, DType::kF32, {t, hv, dv});
  DeviceTensor dgg(gpu, gq.q, DType::kF32, {t, hv});
  DeviceTensor dgb(gpu, gq.q, DType::kF32, {t, hv});
  vt::GdnPostConv(gq.q, dgq.tensor(), dgk.tensor(), dgv.tensor(), dgg.tensor(), dgb.tensor(),
                  dconv.tensor(), daraw.tensor(), dbraw.tensor(), dalog.tensor(), ddtb.tensor(),
                  args);
  std::vector<float> gq_q(fq.size()), gq_k(fk.size()), gq_v(fv.size()), gq_g(fg.size()),
      gq_b(fb.size());
  dgq.Download(gq.q, gq_q.data());
  dgk.Download(gq.q, gq_k.data());
  dgv.Download(gq.q, gq_v.data());
  dgg.Download(gq.q, gq_g.data());
  dgb.Download(gq.q, gq_b.data());
  CheckClose(gq_q, fq, 1e-5f, 1e-5f);
  CheckClose(gq_k, fk, 1e-5f, 1e-5f);
  CheckClose(gq_v, fv, 1e-6f, 0.0f);  // raw copy: exact
  CheckClose(gq_g, fg, 1e-5f, 1e-5f);
  CheckClose(gq_b, fb, 1e-5f, 1e-5f);
}

TEST_CASE("gdn_post_conv fused == composed (CPU) and CUDA matches (real dims, striding)") {
  RunGdnPostConvCase(5, 2, 6, 128, 128, 7100);   // GQA ratio 3, Dk==blockDim/2
  RunGdnPostConvCase(3, 4, 4, 300, 128, 7200);   // Dk 300 > blockDim(256): strided
  RunGdnPostConvCase(1, 1, 2, 64, 64, 7300);     // T==1 (decode-shaped)
}

// Ported from tests/kernels/test_fused_gdn_post_conv.py and the packed
// `in_proj_ba` slicing in qwen_gdn_linear_attn.py:908-943. vLLM projects BA in
// model dtype, then b/a are logical views with a wider parent row stride. Pin
// both the accepted local F32 gate arithmetic and upstream BF16 load/upcast
// semantics, non-zero view offsets, padding canaries and the decode/online
// batch sizes used by the W1 gate.
void RunGdnPackedBaCase(int64_t t, DType gate_dtype, uint32_t seed) {
  constexpr int64_t kHk = 2;
  constexpr int64_t kHv = 6;
  constexpr int64_t kDk = 8;
  constexpr int64_t kDv = 8;
  constexpr int64_t kPrefix = 1;
  constexpr int64_t kSuffix = 3;
  constexpr int64_t kParentWidth = kPrefix + 2 * kHv + kSuffix;
  const int64_t key_dim = kHk * kDk;
  const int64_t value_dim = kHv * kDv;
  const int64_t conv_dim = 2 * key_dim + value_dim;

  const auto conv = RandomF32(static_cast<size_t>(t * conv_dim), seed, -1.5f, 1.5f);
  const auto araw_src = RandomF32(static_cast<size_t>(t * kHv), seed + 1, -1.0f, 1.0f);
  const auto braw_src = RandomF32(static_cast<size_t>(t * kHv), seed + 2, -1.0f, 1.0f);
  const auto alog = RandomF32(static_cast<size_t>(kHv), seed + 3, -1.0f, 1.0f);
  const auto dtb = RandomF32(static_cast<size_t>(kHv), seed + 4, -1.0f, 1.0f);

  std::vector<float> parent(static_cast<size_t>(t * kParentWidth), 7.5f);
  for (int64_t row = 0; row < t; ++row) {
    for (int64_t h = 0; h < kHv; ++h) {
      parent[static_cast<size_t>(row * kParentWidth + kPrefix + h)] =
          braw_src[static_cast<size_t>(row * kHv + h)];
      parent[static_cast<size_t>(row * kParentWidth + kPrefix + kHv + h)] =
          araw_src[static_cast<size_t>(row * kHv + h)];
    }
  }
  const std::vector<uint8_t> parent_bytes = Pack(parent, gate_dtype);
  const std::vector<float> parent_rounded = Unpack(parent_bytes, gate_dtype);
  std::vector<float> araw(static_cast<size_t>(t * kHv));
  std::vector<float> braw(static_cast<size_t>(t * kHv));
  for (int64_t row = 0; row < t; ++row) {
    for (int64_t h = 0; h < kHv; ++h) {
      braw[static_cast<size_t>(row * kHv + h)] =
          parent_rounded[static_cast<size_t>(row * kParentWidth + kPrefix + h)];
      araw[static_cast<size_t>(row * kHv + h)] =
          parent_rounded[static_cast<size_t>(row * kParentWidth + kPrefix + kHv + h)];
    }
  }

  const vt::L2NormArgs args{1e-6f};
  Queue cq = Q();
  Tensor tconv = MakeT(const_cast<float*>(conv.data()), DType::kF32, Cpu(),
                       {t, conv_dim});
  Tensor talog = MakeT(const_cast<float*>(alog.data()), DType::kF32, Cpu(), {kHv});
  Tensor tdtb = MakeT(const_cast<float*>(dtb.data()), DType::kF32, Cpu(), {kHv});
  Tensor taraw = MakeT(araw.data(), DType::kF32, Cpu(), {t, kHv});
  Tensor tbraw = MakeT(braw.data(), DType::kF32, Cpu(), {t, kHv});

  std::vector<float> rq(static_cast<size_t>(t * key_dim));
  std::vector<float> rk(static_cast<size_t>(t * key_dim));
  std::vector<float> rv(static_cast<size_t>(t * value_dim));
  std::vector<float> rg(static_cast<size_t>(t * kHv));
  std::vector<float> rb(static_cast<size_t>(t * kHv));
  Tensor trq = MakeT(rq.data(), DType::kF32, Cpu(), {t, kHk, kDk});
  Tensor trk = MakeT(rk.data(), DType::kF32, Cpu(), {t, kHk, kDk});
  Tensor trv = MakeT(rv.data(), DType::kF32, Cpu(), {t, kHv, kDv});
  Tensor trg = MakeT(rg.data(), DType::kF32, Cpu(), {t, kHv});
  Tensor trb = MakeT(rb.data(), DType::kF32, Cpu(), {t, kHv});
  vt::GdnPostConv(cq, trq, trk, trv, trg, trb, tconv, taraw, tbraw,
                  talog, tdtb, args);

  std::vector<uint8_t> cpu_parent = parent_bytes;
  Tensor tcp = MakeT(cpu_parent.data(), gate_dtype, Cpu(), {t, kParentWidth});
  Tensor tcb = tcp.Slice(1, kPrefix, kPrefix + kHv);
  Tensor tca = tcp.Slice(1, kPrefix + kHv, kPrefix + 2 * kHv);
  std::vector<float> cqv(rq.size()), ckv(rk.size()), cvv(rv.size());
  std::vector<float> cgv(rg.size()), cbv(rb.size());
  Tensor tcq = MakeT(cqv.data(), DType::kF32, Cpu(), {t, kHk, kDk});
  Tensor tck = MakeT(ckv.data(), DType::kF32, Cpu(), {t, kHk, kDk});
  Tensor tcv = MakeT(cvv.data(), DType::kF32, Cpu(), {t, kHv, kDv});
  Tensor tcg = MakeT(cgv.data(), DType::kF32, Cpu(), {t, kHv});
  Tensor tcbeta = MakeT(cbv.data(), DType::kF32, Cpu(), {t, kHv});
  vt::GdnPostConv(cq, tcq, tck, tcv, tcg, tcbeta, tconv, tca, tcb,
                  talog, tdtb, args);
  CheckClose(cqv, rq, 0.0f, 0.0f);
  CheckClose(ckv, rk, 0.0f, 0.0f);
  CheckClose(cvv, rv, 0.0f, 0.0f);
  CheckClose(cgv, rg, 0.0f, 0.0f);
  CheckClose(cbv, rb, 0.0f, 0.0f);
  CHECK(cpu_parent == parent_bytes);  // const inputs and padding canaries untouched

  std::vector<float> direct_g(rg.size()), direct_b(rb.size());
  Tensor tdg = MakeT(direct_g.data(), DType::kF32, Cpu(), {t, kHv});
  Tensor tdb = MakeT(direct_b.data(), DType::kF32, Cpu(), {t, kHv});
  vt::GdnGBeta(cq, tdg, tdb, tca, tcb, talog, tdtb);
  CheckClose(direct_g, rg, 0.0f, 0.0f);
  CheckClose(direct_b, rb, 0.0f, 0.0f);

  if (!HasCuda()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dconv(gpu, gq.q, DType::kF32, {t, conv_dim}, conv.data());
  DeviceTensor dparent(gpu, gq.q, gate_dtype, {t, kParentWidth},
                       parent_bytes.data());
  Tensor dbview = dparent.tensor().Slice(1, kPrefix, kPrefix + kHv);
  Tensor daview = dparent.tensor().Slice(1, kPrefix + kHv,
                                         kPrefix + 2 * kHv);
  DeviceTensor dalog(gpu, gq.q, DType::kF32, {kHv}, alog.data());
  DeviceTensor ddtb(gpu, gq.q, DType::kF32, {kHv}, dtb.data());
  DeviceTensor dq(gpu, gq.q, DType::kF32, {t, kHk, kDk});
  DeviceTensor dk(gpu, gq.q, DType::kF32, {t, kHk, kDk});
  DeviceTensor dv(gpu, gq.q, DType::kF32, {t, kHv, kDv});
  DeviceTensor dg(gpu, gq.q, DType::kF32, {t, kHv});
  DeviceTensor dbeta(gpu, gq.q, DType::kF32, {t, kHv});
  vt::GdnPostConv(gq.q, dq.tensor(), dk.tensor(), dv.tensor(), dg.tensor(),
                  dbeta.tensor(), dconv.tensor(), daview, dbview,
                  dalog.tensor(), ddtb.tensor(), args);
  std::vector<float> gqv(rq.size()), gkv(rk.size()), gvv(rv.size());
  std::vector<float> ggv(rg.size()), gbv(rb.size());
  dq.Download(gq.q, gqv.data());
  dk.Download(gq.q, gkv.data());
  dv.Download(gq.q, gvv.data());
  dg.Download(gq.q, ggv.data());
  dbeta.Download(gq.q, gbv.data());
  CheckClose(gqv, rq, 1e-5f, 1e-5f);
  CheckClose(gkv, rk, 1e-5f, 1e-5f);
  CheckClose(gvv, rv, 1e-6f, 0.0f);
  CheckClose(ggv, rg, 1e-5f, 1e-5f);
  CheckClose(gbv, rb, 1e-5f, 1e-5f);
  DeviceTensor ddg(gpu, gq.q, DType::kF32, {t, kHv});
  DeviceTensor ddb(gpu, gq.q, DType::kF32, {t, kHv});
  vt::GdnGBeta(gq.q, ddg.tensor(), ddb.tensor(), daview, dbview,
               dalog.tensor(), ddtb.tensor());
  std::vector<float> dggv(rg.size()), dgbv(rb.size());
  ddg.Download(gq.q, dggv.data());
  ddb.Download(gq.q, dgbv.data());
  CheckClose(dggv, rg, 1e-5f, 1e-5f);
  CheckClose(dgbv, rb, 1e-5f, 1e-5f);

  // Both production consumers must remain allocation-free and preserve the
  // packed views through graph capture. Poison every output before both
  // replays so an eager result cannot make this pass accidentally.
  gpu.BeginCapture(gq.q);
  vt::GdnPostConv(gq.q, dq.tensor(), dk.tensor(), dv.tensor(), dg.tensor(),
                  dbeta.tensor(), dconv.tensor(), daview, dbview,
                  dalog.tensor(), ddtb.tensor(), args);
  vt::GdnGBeta(gq.q, ddg.tensor(), ddb.tensor(), daview, dbview,
               dalog.tensor(), ddtb.tensor());
  void* graph = gpu.EndCaptureGraph(gq.q);
  for (int replay = 0; replay < 2; ++replay) {
    gpu.Memset(gq.q, dq.tensor().data, 0xA5, rq.size() * sizeof(float));
    gpu.Memset(gq.q, dk.tensor().data, 0xA5, rk.size() * sizeof(float));
    gpu.Memset(gq.q, dv.tensor().data, 0xA5, rv.size() * sizeof(float));
    gpu.Memset(gq.q, dg.tensor().data, 0xA5, rg.size() * sizeof(float));
    gpu.Memset(gq.q, dbeta.tensor().data, 0xA5,
               rb.size() * sizeof(float));
    gpu.Memset(gq.q, ddg.tensor().data, 0xA5, rg.size() * sizeof(float));
    gpu.Memset(gq.q, ddb.tensor().data, 0xA5, rb.size() * sizeof(float));
    gpu.ReplayGraph(gq.q, graph);
    dq.Download(gq.q, gqv.data());
    dk.Download(gq.q, gkv.data());
    dv.Download(gq.q, gvv.data());
    dg.Download(gq.q, ggv.data());
    dbeta.Download(gq.q, gbv.data());
    ddg.Download(gq.q, dggv.data());
    ddb.Download(gq.q, dgbv.data());
    CheckClose(gqv, rq, 1e-5f, 1e-5f);
    CheckClose(gkv, rk, 1e-5f, 1e-5f);
    CheckClose(gvv, rv, 1e-6f, 0.0f);
    CheckClose(ggv, rg, 1e-5f, 1e-5f);
    CheckClose(gbv, rb, 1e-5f, 1e-5f);
    CheckClose(dggv, rg, 1e-5f, 1e-5f);
    CheckClose(dgbv, rb, 1e-5f, 1e-5f);
  }
  gpu.DestroyGraph(graph);

  std::vector<uint8_t> got_parent(parent_bytes.size());
  dparent.Download(gq.q, got_parent.data());
  CHECK(got_parent == parent_bytes);
}

TEST_CASE("gdn BA accepts F32/BF16 row-strided packed views on CPU and CUDA") {
  uint32_t seed = 7400;
  for (DType gate_dtype : {DType::kF32, DType::kBF16}) {
    for (int64_t batch : {1, 2, 4, 16, 32}) {
      CAPTURE(gate_dtype);
      CAPTURE(batch);
      RunGdnPackedBaCase(batch, gate_dtype, seed);
      seed += 10;
    }
  }
}

// Ported from the packed `in_proj_qkvz` slicing in qwen_gdn_linear_attn.py:
// 929-936: vLLM projects qkvz in model dtype once, then mixed_qkv and z are
// logical last-dim views with the packed parent row stride. The two W2
// consumers (causal conv over mixed_qkv, gated RMSNorm over z as [T,Hv,Dv])
// must produce bit-identical results from the strided views and the contiguous
// copies, leave the parent (canary prefix/suffix included) untouched, and
// survive CUDA graph capture/replay without materialization.
void RunGdnMergedQkvzCase(int64_t t, DType dtype, uint32_t seed) {
  constexpr int64_t kHk = 2;
  constexpr int64_t kHv = 6;
  constexpr int64_t kDk = 8;
  constexpr int64_t kDv = 8;
  constexpr int64_t kKw = 4;
  constexpr int64_t kPrefix = 1;
  constexpr int64_t kSuffix = 3;
  const int64_t key_dim = kHk * kDk;
  const int64_t value_dim = kHv * kDv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t parent_w = kPrefix + conv_dim + value_dim + kSuffix;

  const auto parent_f = RandomF32(static_cast<size_t>(t * parent_w), seed, -1.5f, 1.5f);
  const std::vector<uint8_t> parent_bytes = Pack(parent_f, dtype);
  const std::vector<float> parent_rounded = Unpack(parent_bytes, dtype);
  // Contiguous copies of the two logical slices (identical rounded values).
  std::vector<float> mixed_f(static_cast<size_t>(t * conv_dim));
  std::vector<float> z_f(static_cast<size_t>(t * value_dim));
  for (int64_t row = 0; row < t; ++row) {
    for (int64_t c = 0; c < conv_dim; ++c)
      mixed_f[static_cast<size_t>(row * conv_dim + c)] =
          parent_rounded[static_cast<size_t>(row * parent_w + kPrefix + c)];
    for (int64_t c = 0; c < value_dim; ++c)
      z_f[static_cast<size_t>(row * value_dim + c)] =
          parent_rounded[static_cast<size_t>(row * parent_w + kPrefix + conv_dim + c)];
  }
  const std::vector<uint8_t> mixed_bytes = Pack(mixed_f, dtype);
  const std::vector<uint8_t> z_bytes = Pack(z_f, dtype);
  const auto wconv_f = RandomF32(static_cast<size_t>(conv_dim * kKw), seed + 1);
  const std::vector<uint8_t> wconv_bytes = Pack(wconv_f, dtype);
  const auto st_fwd0 = RandomF32(static_cast<size_t>(conv_dim * (kKw - 1)), seed + 2);
  const auto st_upd0 = RandomF32(static_cast<size_t>(t * conv_dim * (kKw - 1)), seed + 3);
  const auto core_f = RandomF32(static_cast<size_t>(t * kHv * kDv), seed + 4);
  const std::vector<uint8_t> core_bytes = Pack(core_f, dtype);
  const auto nw_f = RandomF32(static_cast<size_t>(kDv), seed + 5, -1.0f, 1.0f);
  const std::vector<uint8_t> nw_bytes = Pack(nw_f, dtype);
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(t)}, his = {1};
  const CausalConv1dArgs conv_args{true};
  const RmsNormGatedArgs norm_args{1e-6f, false};
  const size_t out_conv_bytes = static_cast<size_t>(t * conv_dim) * vt::SizeOf(dtype);
  const size_t out_norm_bytes = static_cast<size_t>(t * value_dim) * vt::SizeOf(dtype);

  Queue cq = Q();
  Tensor tw = MakeT(const_cast<uint8_t*>(wconv_bytes.data()), dtype, Cpu(), {conv_dim, kKw});
  Tensor tnw = MakeT(const_cast<uint8_t*>(nw_bytes.data()), dtype, Cpu(), {kDv});
  Tensor tqsl = Ti(qsl), this_ = Ti(his);

  // --- CPU contiguous reference. ---
  std::vector<uint8_t> conv_ref(out_conv_bytes);
  std::vector<float> st_fwd_ref = st_fwd0;
  {
    Tensor tx = MakeT(const_cast<uint8_t*>(mixed_bytes.data()), dtype, Cpu(), {t, conv_dim});
    Tensor ts = MakeT(st_fwd_ref.data(), DType::kF32, Cpu(), {1, conv_dim, kKw - 1});
    Tensor to = MakeT(conv_ref.data(), dtype, Cpu(), {t, conv_dim});
    vt::CausalConv1dFwd(cq, to, tx, tw, nullptr, ts, tqsl, this_, conv_args);
  }
  std::vector<uint8_t> upd_ref(out_conv_bytes);
  std::vector<float> st_upd_ref = st_upd0;
  {
    Tensor tx = MakeT(const_cast<uint8_t*>(mixed_bytes.data()), dtype, Cpu(), {t, conv_dim});
    Tensor ts = MakeT(st_upd_ref.data(), DType::kF32, Cpu(), {t, conv_dim, kKw - 1});
    Tensor to = MakeT(upd_ref.data(), dtype, Cpu(), {t, conv_dim});
    vt::CausalConv1dUpdate(cq, to, tx, tw, nullptr, ts, conv_args);
  }
  std::vector<uint8_t> norm_ref(out_norm_bytes);
  {
    Tensor txc = MakeT(const_cast<uint8_t*>(core_bytes.data()), dtype, Cpu(),
                       {t * kHv, kDv});
    Tensor tz = MakeT(const_cast<uint8_t*>(z_bytes.data()), dtype, Cpu(), {t * kHv, kDv});
    Tensor to = MakeT(norm_ref.data(), dtype, Cpu(), {t * kHv, kDv});
    vt::RmsNormGated(cq, to, txc, tz, tnw, norm_args);
  }

  // --- CPU strided views over the packed parent: bit-identical, no writes. ---
  std::vector<uint8_t> cpu_parent = parent_bytes;
  Tensor tparent = MakeT(cpu_parent.data(), dtype, Cpu(), {t, parent_w});
  Tensor tmixed = tparent.Slice(1, kPrefix, kPrefix + conv_dim);
  Tensor tzs = tparent.Slice(1, kPrefix + conv_dim, kPrefix + conv_dim + value_dim);
  Tensor tz3 = Rank3RowView(tzs.data, dtype, Cpu(), t, kHv, kDv, parent_w);
  {
    std::vector<uint8_t> got(out_conv_bytes);
    std::vector<float> st = st_fwd0;
    Tensor ts = MakeT(st.data(), DType::kF32, Cpu(), {1, conv_dim, kKw - 1});
    Tensor to = MakeT(got.data(), dtype, Cpu(), {t, conv_dim});
    vt::CausalConv1dFwd(cq, to, tmixed, tw, nullptr, ts, tqsl, this_, conv_args);
    CHECK(got == conv_ref);
    CHECK(st == st_fwd_ref);
  }
  {
    std::vector<uint8_t> got(out_conv_bytes);
    std::vector<float> st = st_upd0;
    Tensor ts = MakeT(st.data(), DType::kF32, Cpu(), {t, conv_dim, kKw - 1});
    Tensor to = MakeT(got.data(), dtype, Cpu(), {t, conv_dim});
    vt::CausalConv1dUpdate(cq, to, tmixed, tw, nullptr, ts, conv_args);
    CHECK(got == upd_ref);
    CHECK(st == st_upd_ref);
  }
  {
    std::vector<uint8_t> got(out_norm_bytes);
    std::vector<uint8_t> core3 = core_bytes;
    Tensor txc = MakeT(core3.data(), dtype, Cpu(), {t, kHv, kDv});
    Tensor to = MakeT(got.data(), dtype, Cpu(), {t, kHv, kDv});
    vt::RmsNormGated(cq, to, txc, tz3, tnw, norm_args);
    CHECK(got == norm_ref);
  }
  CHECK(cpu_parent == parent_bytes);  // canaries + const packed input untouched

  if (!HasCuda()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceTensor dparent(gpu, gq.q, dtype, {t, parent_w}, parent_bytes.data());
  DeviceTensor dmixed_c(gpu, gq.q, dtype, {t, conv_dim}, mixed_bytes.data());
  DeviceTensor dz_c(gpu, gq.q, dtype, {t * kHv, kDv}, z_bytes.data());
  DeviceTensor dw(gpu, gq.q, dtype, {conv_dim, kKw}, wconv_bytes.data());
  DeviceTensor dnw(gpu, gq.q, dtype, {kDv}, nw_bytes.data());
  DeviceTensor dcore2(gpu, gq.q, dtype, {t * kHv, kDv}, core_bytes.data());
  DeviceTensor dcore3(gpu, gq.q, dtype, {t, kHv, kDv}, core_bytes.data());
  DeviceTensor dqsl(gpu, gq.q, DType::kI32, {2}, qsl.data());
  DeviceTensor dhis(gpu, gq.q, DType::kI32, {1}, his.data());
  Tensor dmix_v = dparent.tensor().Slice(1, kPrefix, kPrefix + conv_dim);
  Tensor dz_slice =
      dparent.tensor().Slice(1, kPrefix + conv_dim, kPrefix + conv_dim + value_dim);
  Tensor dz3 = Rank3RowView(dz_slice.data, dtype, Gpu(), t, kHv, kDv, parent_w);

  // CUDA contiguous reference arms.
  DeviceTensor dconv_ref(gpu, gq.q, dtype, {t, conv_dim});
  DeviceTensor dst_fwd_ref(gpu, gq.q, DType::kF32, {1, conv_dim, kKw - 1}, st_fwd0.data());
  vt::CausalConv1dFwd(gq.q, dconv_ref.tensor(), dmixed_c.tensor(), dw.tensor(), nullptr,
                      dst_fwd_ref.tensor(), dqsl.tensor(), dhis.tensor(), conv_args);
  DeviceTensor dupd_ref(gpu, gq.q, dtype, {t, conv_dim});
  DeviceTensor dst_upd_ref(gpu, gq.q, DType::kF32, {t, conv_dim, kKw - 1}, st_upd0.data());
  vt::CausalConv1dUpdate(gq.q, dupd_ref.tensor(), dmixed_c.tensor(), dw.tensor(), nullptr,
                         dst_upd_ref.tensor(), conv_args);
  DeviceTensor dnorm_ref(gpu, gq.q, dtype, {t * kHv, kDv});
  vt::RmsNormGated(gq.q, dnorm_ref.tensor(), dcore2.tensor(), dz_c.tensor(), dnw.tensor(),
                   norm_args);
  std::vector<uint8_t> conv_ref_gpu(out_conv_bytes), upd_ref_gpu(out_conv_bytes),
      norm_ref_gpu(out_norm_bytes);
  std::vector<float> st_fwd_ref_gpu(st_fwd0.size()), st_upd_ref_gpu(st_upd0.size());
  dconv_ref.Download(gq.q, conv_ref_gpu.data());
  dst_fwd_ref.Download(gq.q, st_fwd_ref_gpu.data());
  dupd_ref.Download(gq.q, upd_ref_gpu.data());
  dst_upd_ref.Download(gq.q, st_upd_ref_gpu.data());
  dnorm_ref.Download(gq.q, norm_ref_gpu.data());

  // CUDA strided arms: bit-identical to the CUDA contiguous arms.
  DeviceTensor dconv(gpu, gq.q, dtype, {t, conv_dim});
  DeviceTensor dst_fwd(gpu, gq.q, DType::kF32, {1, conv_dim, kKw - 1}, st_fwd0.data());
  DeviceTensor dupd(gpu, gq.q, dtype, {t, conv_dim});
  DeviceTensor dst_upd(gpu, gq.q, DType::kF32, {t, conv_dim, kKw - 1}, st_upd0.data());
  DeviceTensor dnorm(gpu, gq.q, dtype, {t, kHv, kDv});
  Tensor dnorm3 = dnorm.tensor();
  auto run_strided = [&] {
    vt::CausalConv1dFwd(gq.q, dconv.tensor(), dmix_v, dw.tensor(), nullptr,
                        dst_fwd.tensor(), dqsl.tensor(), dhis.tensor(), conv_args);
    vt::CausalConv1dUpdate(gq.q, dupd.tensor(), dmix_v, dw.tensor(), nullptr,
                           dst_upd.tensor(), conv_args);
    vt::RmsNormGated(gq.q, dnorm3, dcore3.tensor(), dz3, dnw.tensor(), norm_args);
  };
  auto check_strided = [&] {
    std::vector<uint8_t> got_conv(out_conv_bytes), got_upd(out_conv_bytes),
        got_norm(out_norm_bytes);
    std::vector<float> got_st_fwd(st_fwd0.size()), got_st_upd(st_upd0.size());
    dconv.Download(gq.q, got_conv.data());
    dst_fwd.Download(gq.q, got_st_fwd.data());
    dupd.Download(gq.q, got_upd.data());
    dst_upd.Download(gq.q, got_st_upd.data());
    dnorm.Download(gq.q, got_norm.data());
    CHECK(got_conv == conv_ref_gpu);
    CHECK(got_st_fwd == st_fwd_ref_gpu);
    CHECK(got_upd == upd_ref_gpu);
    CHECK(got_st_upd == st_upd_ref_gpu);
    CHECK(got_norm == norm_ref_gpu);
  };
  auto reset_states = [&] {
    gpu.Copy(gq.q, dst_fwd.tensor().data, st_fwd0.data(),
             st_fwd0.size() * sizeof(float));
    gpu.Copy(gq.q, dst_upd.tensor().data, st_upd0.data(),
             st_upd0.size() * sizeof(float));
  };
  run_strided();
  check_strided();

  // Capture the three strided consumers; poison outputs and restore the conv
  // states before each replay so an eager result cannot pass accidentally.
  reset_states();
  gpu.BeginCapture(gq.q);
  run_strided();
  void* graph = gpu.EndCaptureGraph(gq.q);
  for (int replay = 0; replay < 2; ++replay) {
    gpu.Memset(gq.q, dconv.tensor().data, 0xA5, out_conv_bytes);
    gpu.Memset(gq.q, dupd.tensor().data, 0xA5, out_conv_bytes);
    gpu.Memset(gq.q, dnorm.tensor().data, 0xA5, out_norm_bytes);
    reset_states();
    gpu.ReplayGraph(gq.q, graph);
    check_strided();
  }
  gpu.DestroyGraph(graph);

  std::vector<uint8_t> got_parent(parent_bytes.size());
  dparent.Download(gq.q, got_parent.data());
  CHECK(got_parent == parent_bytes);
}

TEST_CASE("gdn merged qkvz strided mixed/z views feed conv + gated norm exactly") {
  uint32_t seed = 8600;
  for (DType dtype : {DType::kF32, DType::kBF16}) {
    for (int64_t batch : {1, 2, 4, 16, 32}) {
      CAPTURE(dtype);
      CAPTURE(batch);
      RunGdnMergedQkvzCase(batch, dtype, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA rmsnorm_gated matches CPU (silu and sigmoid gates)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 5600;
  for (bool sigmoid : {false, true}) {
    for (const Combo& cb : kCudaCombos) {
      CAPTURE(sigmoid);
      CAPTURE(static_cast<int>(cb.in));
      CAPTURE(static_cast<int>(cb.out));
      RunRmsNormGatedCudaCase(4, 129, sigmoid, cb, seed);
      seed += 10;
    }
  }
}

TEST_CASE("CUDA rmsnorm_gated decode-fast (VT_RMSNORM_GATED_FAST) matches rollback kernel 0-ulp") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // Real GDN decode shapes: [T*Hv, Dv=128] (27B Hv=48, 35B Hv=32) at c1-c16, both
  // silu (sigmoid_gate=false, the production gate) and sigmoid. fast==shipped 0-ulp.
  uint32_t seed = 7100;
  for (bool sigmoid : {false, true}) {
    // Contiguous rank-2 gate (gate_group=1): sweep the token count via rows=T*Hv.
    for (int64_t rows : {48, 96, 512, 768}) {  // 27B c1/c2, 35B c16, 27B c16
      CAPTURE(sigmoid);
      CAPTURE(rows);
      RunRmsNormGatedFastContiguous(rows, sigmoid, seed);
      seed += 10;
    }
    // Padded rank-3 [T,Hv,Dv] strided gate (gate_group=Hv): the real z-slice geometry.
    for (int64_t t : {1, 2, 16}) {
      for (int64_t hv : {48, 32}) {  // 27B, 35B
        CAPTURE(sigmoid);
        CAPTURE(t);
        CAPTURE(hv);
        RunRmsNormGatedFastPaddedRank3(t, hv, sigmoid, seed);
        seed += 10;
      }
    }
  }
}

TEST_CASE("CUDA gdn prefill matches CPU (varlen, GQA ratio 3)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 5800;
  for (const Combo& cb : kCudaCombos) {
    CAPTURE(static_cast<int>(cb.in));
    CAPTURE(static_cast<int>(cb.out));
    RunGdnCudaCase({0, 4, 7}, 0, 2, 6, 16, 8, cb, seed);
    seed += 10;
  }
}

TEST_CASE("CUDA gdn prefill matches CPU (real dims and > blockDim striding)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // Real head dims (Dk=Dv=128, GQA ratio 2 as in the 35B gate).
  RunGdnCudaCase({0, 5}, 0, 1, 2, 128, 128, kCudaCombos[0], 6000);
  // Dv=300 > 256 threads exercises the row-stride loop; Dk=300 the shared
  // q'/k staging stride.
  RunGdnCudaCase({0, 3}, 0, 1, 2, 300, 300, kCudaCombos[0], 6010);
}

TEST_CASE("CUDA gdn decode matches CPU (GQA ratio 3)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 6200;
  for (const Combo& cb : kCudaCombos) {
    CAPTURE(static_cast<int>(cb.in));
    CAPTURE(static_cast<int>(cb.out));
    RunGdnCudaCase({}, 3, 2, 6, 16, 8, cb, seed);
    seed += 10;
  }
}

// Batched decode at the REAL gate head dims (Dk=Dv=128, GQA ratio 2), multiple
// sequences: exercises the fused (batch × Hv × value-tile) decode grid — each
// (seq, v-head) fans out over NV=ceil(Dv/BV)=4 value tiles, and the batch of
// 8 single-token sequences advances 8 independent [Dv,Dk] states in one launch.
// This is the path the paged decode step drives; it must match the sequential
// per-sequence CPU reference token-for-token (state included).
TEST_CASE("CUDA gdn decode matches CPU (real dims, batched multi-seq)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGdnCudaCase({}, 8, 16, 32, 128, 128, kCudaCombos[0], 6300);  // f32 (the real GDN dtype)
  RunGdnCudaCase({}, 8, 16, 32, 128, 128, kCudaCombos[2], 6310);  // bf16 in/out
  RunGdnCudaCase({}, 5, 2, 2, 128, 128, kCudaCombos[0], 6320);    // GQA ratio 1, odd batch
}

// vLLM's public mamba_ssm_cache_dtype accepts float16 independently of the
// model activation dtype. Preserve the decomposed rollback path for that
// configuration as well as the packed path; otherwise disabling packed decode
// would turn an accepted cache configuration into a runtime validation error.
TEST_CASE("CUDA gdn decode accepts independent fp16 SSM cache") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGdnCudaCase({}, 4, 2, 4, 16, 8, kCudaCombos[2], 6330,
                 DType::kF16);
}

// bf16 persistent state cache (an accepted explicit cache dtype) vs the f32
// gate-model cache, SAME f32 q/k/v/g/beta and SAME initial state (the f32 master
// packed to bf16). The decode kernel reads bf16 → f32 registers → writes bf16
// (mirrors fla fused_recurrent). This quantifies the one-step
// read/compute/write-back drift the bf16 cache introduces relative to the f32
// cache. Real gate dims (Dk=Dv=128, Hv=32, GQA 2). Nsim decode steps chained to
// expose accumulation.
TEST_CASE("CUDA gdn decode: bf16 state cache drift vs f32 (real dims, chained steps)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const int64_t batch = 8, hk = 16, hv = 32, dk = 128, dv = 128;
  const int64_t t = batch, n = batch;
  const GdnArgs args{1.0f / std::sqrt(static_cast<float>(dk))};
  const int steps = 16;  // chain 16 decode steps to accumulate the bf16 round-trip

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);

  // f32 initial state master (identical for both caches).
  const auto stf = RandomF32(static_cast<size_t>(n * hv * dv * dk), 7005, -0.5f, 0.5f);
  const auto st_bf = Pack(stf, DType::kBF16);

  DeviceTensor dst_f32(gpu, gq.q, DType::kF32, {n, hv, dv, dk}, stf.data());
  DeviceTensor dst_bf(gpu, gq.q, DType::kBF16, {n, hv, dv, dk}, st_bf.data());
  DeviceTensor dout_ref(gpu, gq.q, DType::kF32, {t, hv, dv});
  DeviceTensor dout_bf(gpu, gq.q, DType::kF32, {t, hv, dv});

  float max_out = 0.0f, max_st = 0.0f, sum2_out = 0.0f;
  size_t cnt_out = 0;
  for (int s = 0; s < steps; ++s) {
    // Fresh per-step q/k/v/g/beta (a distinct token each step). The real decode
    // feeds L2-NORMALIZED q,k (dql2/dkl2, unit norm over Dk) — normalize here so
    // the recurrence is in the model's stable regime (unnormalized 128-dim k
    // makes the rank-1 delta updates blow up, which is not the real workload).
    auto qf = RandomF32(static_cast<size_t>(t * hk * dk), 7100u + s, -1.0f, 1.0f);
    auto kf = RandomF32(static_cast<size_t>(t * hk * dk), 7200u + s, -1.0f, 1.0f);
    auto l2norm_rows = [dk](std::vector<float>& x) {
      for (size_t r = 0; r + static_cast<size_t>(dk) <= x.size(); r += static_cast<size_t>(dk)) {
        float ss = 0.0f;
        for (int64_t c = 0; c < dk; ++c) ss += x[r + c] * x[r + c];
        const float inv = 1.0f / std::sqrt(ss + 1e-6f);
        for (int64_t c = 0; c < dk; ++c) x[r + c] *= inv;
      }
    };
    l2norm_rows(qf);
    l2norm_rows(kf);
    const auto vf = RandomF32(static_cast<size_t>(t * hv * dv), 7300u + s, -1.0f, 1.0f);
    const auto gf = RandomF32(static_cast<size_t>(t * hv), 7400u + s, -1.0f, 0.0f);
    const auto betaf = RandomF32(static_cast<size_t>(t * hv), 7500u + s, 0.05f, 0.95f);
    DeviceTensor dq(gpu, gq.q, DType::kF32, {t, hk, dk}, qf.data());
    DeviceTensor dk_(gpu, gq.q, DType::kF32, {t, hk, dk}, kf.data());
    DeviceTensor dv_(gpu, gq.q, DType::kF32, {t, hv, dv}, vf.data());
    DeviceTensor dg(gpu, gq.q, DType::kF32, {t, hv}, gf.data());
    DeviceTensor dbeta(gpu, gq.q, DType::kF32, {t, hv}, betaf.data());

    vt::GdnDecode(gq.q, dout_ref.tensor(), dq.tensor(), dk_.tensor(), dv_.tensor(), dg.tensor(),
                  dbeta.tensor(), dst_f32.tensor(), args);
    vt::GdnDecode(gq.q, dout_bf.tensor(), dq.tensor(), dk_.tensor(), dv_.tensor(), dg.tensor(),
                  dbeta.tensor(), dst_bf.tensor(), args);

    std::vector<float> o_ref(static_cast<size_t>(t * hv * dv));
    std::vector<float> o_bf(static_cast<size_t>(t * hv * dv));
    dout_ref.Download(gq.q, o_ref.data());
    dout_bf.Download(gq.q, o_bf.data());
    for (size_t i = 0; i < o_ref.size(); ++i) {
      const float diff = std::fabs(o_bf[i] - o_ref[i]);
      max_out = std::max(max_out, diff);
      sum2_out += diff * diff;
      ++cnt_out;
    }
  }
  // Final-state drift after the 16 chained steps (worst-case accumulation).
  std::vector<uint8_t> st_bf_out(st_bf.size());
  std::vector<float> st_ref_out(stf.size());
  dst_bf.Download(gq.q, st_bf_out.data());
  dst_f32.Download(gq.q, st_ref_out.data());
  const auto st_bf_f = Unpack(st_bf_out, DType::kBF16);
  for (size_t i = 0; i < st_ref_out.size(); ++i)
    max_st = std::max(max_st, std::fabs(st_bf_f[i] - st_ref_out[i]));

  const float rms_out = std::sqrt(sum2_out / static_cast<float>(cnt_out));
  MESSAGE("bf16-state decode drift over " << steps << " chained steps: out max|Δ|=" << max_out
          << " rms=" << rms_out << " ; final-state max|Δ|=" << max_st);
  // bf16 carries ~8 mantissa bits (~4e-3 relative). One rank-1 update per step
  // keeps the state O(1); after 16 steps the drift stays well within a loose
  // bf16-scale bound. This is the faithful mirror of vLLM's bf16 cache.
  CHECK(max_out < 5e-2f);
  CHECK(max_st < 5e-2f);
}

// The M2 chunk-parallel prefill scan vs the sequential scan (same binary),
// real head dims Dk=Dv=128 (the gate model), GQA ratio 2, multiple heads.
// 150 tokens = 2 full chunks (64) + a partial tail (22) exercises the chunk
// boundary and the partial-tail masking.
TEST_CASE("CUDA gdn prefill chunked matches sequential (2+ chunks, partial tail)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // f32 in/out: reassociation-only diffs. bf16 in: bf16-rounding scale.
  const Combo f32 = {DType::kF32, DType::kF32, 5e-3f, 5e-3f};
  const Combo bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  // Isolation ladder: single partial chunk (intra only), one exact chunk, two
  // chunks (first cross-chunk state pass), then the partial-tail cases.
  RunGdnChunkedVsSequential({0, 40}, 1, 1, 128, 128, f32, 6900, 5e-3f, 5e-3f);
  RunGdnChunkedVsSequential({0, 64}, 1, 1, 128, 128, f32, 6910, 5e-3f, 5e-3f);
  RunGdnChunkedVsSequential({0, 128}, 1, 1, 128, 128, f32, 6920, 5e-3f, 5e-3f);
  RunGdnChunkedVsSequential({0, 150}, 2, 4, 128, 128, f32, 7000, 5e-3f, 5e-3f);
  RunGdnChunkedVsSequential({0, 150}, 2, 4, 128, 128, bf16, 7010, 3e-2f, 3e-2f);
  // Multi-sequence varlen: two sequences of differing chunk counts packed
  // together (chunk-offset / per-seq state layout).
  RunGdnChunkedVsSequential({0, 150, 200}, 2, 4, 128, 128, f32, 7020, 5e-3f, 5e-3f);
  // GQA ratio 2 at the real gate shape (Hk=16/Hv=32 sliced to Hk=4/Hv=8).
  RunGdnChunkedVsSequential({0, 130}, 4, 8, 128, 128, f32, 7030, 5e-3f, 5e-3f);
  // Restore default for any later cases.
  setenv("VT_GDN_CHUNKED", "1", 1);
  (void)f32;
  (void)bf16;
}

#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
TEST_CASE("CUDA gdn Triton AOT concurrent first load is safe across two queues") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard q0(gpu);
  QueueGuard q1(gpu);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::exception_ptr failures[2];
  auto warm = [&](size_t index, Queue& queue) {
    ready.fetch_add(1, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    try {
      vt::cuda::testing::WarmGdnTritonAotModules(queue.device.index);
    } catch (...) {
      failures[index] = std::current_exception();
    }
  };
  std::thread first(warm, 0, std::ref(q0.q));
  std::thread second(warm, 1, std::ref(q1.q));
  while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (failures[0]) std::rethrow_exception(failures[0]);
  if (failures[1]) std::rethrow_exception(failures[1]);

  setenv("VT_GDN_CHUNKED", "1", 1);
  setenv("VT_GDN_DELTAH_TRITON", "1", 1);
  setenv("VT_GDN_CHUNKO_TRITON", "1", 1);
  setenv("VT_GDN_WU_TRITON", "1", 1);
  setenv("VT_GDN_TRITON_CHUNK_POOL", "1", 1);
  setenv("VT_GDN_TRITON_WU_POOL", "1", 1);
  const Combo bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  vt::cuda::testing::ResetGdnTritonDebugStats();
  const auto h48 = RunGdnChunkedVsSequentialOnQueue(
      gpu, q0.q, {0, 150}, 16, 48, 128, 128, bf16, 7220, 3e-2f, 3e-2f);
  const auto h32 = RunGdnChunkedVsSequentialOnQueue(
      gpu, q1.q, {0, 150}, 16, 32, 128, 128, bf16, 7230, 3e-2f, 3e-2f);
  CheckUpstreamCutedslTolerances(h48);
  CheckUpstreamCutedslTolerances(h32);
  const auto stats = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(stats.chunk_o_bf16_launches == 2);
  CHECK(stats.chunk_o_hand_launches == 0);
  vt::cuda::testing::DisableGdnTritonDebugStats();
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  unsetenv("VT_GDN_TRITON_CHUNK_POOL");
  unsetenv("VT_GDN_TRITON_WU_POOL");
  setenv("VT_GDN_CHUNKED", "1", 1);
}
#endif

// SANCTIONED Triton AOT delta_h fast-path, token-exact at the EXACT gate-model GDN
// shapes (K=V=128, Hk=16, Hv in {48,32} = Qwen3.6-27B / 35B). The chunked path with
// VT_GDN_DELTAH_TRITON=1 routes delta_h through the AOT cubin; it is compared vs the
// sequential scan exactly like the hand path above. In a build WITHOUT VLLM_CPP_TRITON
// the env is inert (the chunked path stays hand-C++), so this case is valid in both
// builds — it only exercises Triton in the VLLM_CPP_TRITON build. bf16 tolerance
// matches the hand chunked-vs-sequential cases (same per-head delta-rule math).
TEST_CASE("CUDA gdn prefill chunked (Triton delta_h) matches sequential at gate shape") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const Combo bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  setenv("VT_GDN_DELTAH_TRITON", "1", 1);  // opt into the Triton path (inert if not built)
  setenv("VT_GDN_CHUNKO_TRITON", "0", 1);
  setenv("VT_GDN_WU_TRITON", "0", 1);
  // Two chunks + partial tail (150 tok) exercises the cross-chunk state recurrence,
  // which is the delta_h kernel. 27B shape (Hk16/Hv48) then 35B shape (Hk16/Hv32).
  RunGdnChunkedVsSequential({0, 150}, 16, 48, 128, 128, bf16, 7100, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16, 7110, 3e-2f, 3e-2f);
  // Multi-sequence varlen at the 27B shape (differing chunk counts + chunk offsets).
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 48, 128, 128, bf16, 7120, 3e-2f, 3e-2f);
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  setenv("VT_GDN_CHUNKED", "1", 1);
  (void)bf16;
}

// SANCTIONED Triton AOT chunk_o + WU (WY) fast-paths, token-exact at the EXACT
// gate-model GDN shapes (K=V=128, Hk=16, Hv in {48,32} = Qwen3.6-27B / 35B). The
// chunked path with VT_GDN_CHUNKO_TRITON=1 / VT_GDN_WU_TRITON=1 routes chunk_o /
// the WY pipeline through the AOT cubins; both are compared vs the sequential scan
// exactly like the hand path above. chunk_o always has the f32-output AOT spec;
// the bf16-output AOT spec engages only when the vendored artifacts were regenerated
// and VLLM_CPP_TRITON_CHUNKO_BF16 is compiled. The bf16-in/bf16-out cases still
// guard the fallback path and become Triton coverage in that build. WU Triton is
// dtype-agnostic on out, so a bf16-in/bf16-out case also exercises it. In a build
// WITHOUT VLLM_CPP_TRITON the envs are inert (chunked stays hand-C++), so this case
// is valid in both builds. Tolerance matches the hand
// chunked-vs-sequential cases (same per-head WY + delta-rule + output math).
TEST_CASE("CUDA gdn prefill chunked (Triton chunk_o only) matches sequential at gate shape") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const Combo bf16_f32 = {DType::kBF16, DType::kF32, 3e-2f, 3e-2f};   // default GDN out dtype
  const Combo bf16_bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  setenv("VT_GDN_DELTAH_TRITON", "0", 1);
  setenv("VT_GDN_CHUNKO_TRITON", "1", 1);
  setenv("VT_GDN_WU_TRITON", "0", 1);
  RunGdnChunkedVsSequential({0, 150}, 16, 48, 128, 128, bf16_f32, 7200, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16_f32, 7210, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 48, 128, 128, bf16_f32, 7215, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 48, 128, 128, bf16_bf16, 7216, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16_bf16, 7217, 3e-2f, 3e-2f);
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  setenv("VT_GDN_CHUNKED", "1", 1);
}

TEST_CASE("CUDA gdn prefill chunked (Triton WU only) matches sequential at gate shape") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const Combo bf16_bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  setenv("VT_GDN_DELTAH_TRITON", "0", 1);
  setenv("VT_GDN_CHUNKO_TRITON", "0", 1);
  setenv("VT_GDN_WU_TRITON", "1", 1);
  RunGdnChunkedVsSequential({0, 150}, 16, 48, 128, 128, bf16_bf16, 7220, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16_bf16, 7230, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 48, 128, 128, bf16_bf16, 7235, 3e-2f, 3e-2f);
  // H=32 (35B) multi-sequence — the batched-graph gate config. Several seqs with
  // partial tails at different offsets, exercising chunk_indices at H=32.
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 32, 128, 128, bf16_bf16, 7236, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 100, 150, 264, 300, 401}, 16, 32, 128, 128, bf16_bf16, 7237,
                            3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 100, 150, 264, 300, 401}, 16, 48, 128, 128, bf16_bf16, 7238,
                            3e-2f, 3e-2f);
  // Very short seqs (T < 16, single partial sub-block) — the batched 35B prompt
  // shape (~6 concurrent ~10-token prompts). Exercises solve_tril's T<16 corner.
  RunGdnChunkedVsSequential({0, 10, 22, 31, 40, 52, 61}, 16, 32, 128, 128, bf16_bf16, 7239,
                            3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 10}, 16, 32, 128, 128, bf16_bf16, 7241, 3e-2f, 3e-2f);
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  setenv("VT_GDN_CHUNKED", "1", 1);
}

TEST_CASE("CUDA gdn prefill chunked (Triton WU + delta_h + chunk_o) matches sequential") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  const Combo bf16_f32 = {DType::kBF16, DType::kF32, 3e-2f, 3e-2f};
  const Combo bf16_bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  setenv("VT_GDN_WU_TRITON", "1", 1);
  setenv("VT_GDN_CHUNKO_TRITON", "1", 1);
  setenv("VT_GDN_DELTAH_TRITON", "1", 1);
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 48, 128, 128, bf16_f32, 7240, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16_f32, 7250, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150, 200}, 16, 48, 128, 128, bf16_bf16, 7251, 3e-2f, 3e-2f);
  RunGdnChunkedVsSequential({0, 150}, 16, 32, 128, 128, bf16_bf16, 7252, 3e-2f, 3e-2f);
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  setenv("VT_GDN_CHUNKED", "1", 1);
}

#ifdef VLLM_CPP_TRITON_CHUNKO_BF16
// Ported coverage from pinned vLLM
// tests/kernels/mamba/test_gdn_prefill_cutedsl.py::test_gdn_chunk_cutedsl_correctness:
// bf16 output-buffer parity across varlen batches. The vllm.cpp-specific
// additions prove the AOT dispatcher and queue-owned allocator behavior rather
// than allowing a hand-kernel fallback to satisfy the numerical assertion.
TEST_CASE("CUDA gdn Triton bf16 chunk_o dispatch and same-stream pools reuse dirty scratch") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }

  setenv("VT_GDN_CHUNKED", "1", 1);
  setenv("VT_GDN_DELTAH_TRITON", "1", 1);
  setenv("VT_GDN_CHUNKO_TRITON", "1", 1);
  setenv("VT_GDN_WU_TRITON", "1", 1);
  setenv("VT_GDN_TRITON_CHUNK_POOL", "1", 1);
  setenv("VT_GDN_TRITON_WU_POOL", "1", 1);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  const Combo bf16 = {DType::kBF16, DType::kBF16, 3e-2f, 3e-2f};
  vt::cuda::testing::ResetGdnTritonDebugStats();

  // First call owns the initial allocations and must dispatch bf16 chunk_o.
  const auto first_diff = RunGdnChunkedVsSequentialOnQueue(
      gpu, gq.q, {0, 150}, 16, 48, 128, 128, bf16, 7300, 3e-2f, 3e-2f);
  CheckUpstreamCutedslTolerances(first_diff);
  auto first = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(first.chunk_o_bf16_launches == 1);
  CHECK(first.chunk_o_hand_launches == 0);
  CHECK(first.chunk_pool_allocations == 9);
  CHECK(first.chunk_pool_growths == 0);
  CHECK(first.wu_pool_allocations == 2);
  CHECK(first.wu_pool_growths == 0);

  // Poison every pooled allocation with 0xff: this is NaN for float/bf16
  // payloads and invalid for the metadata arrays. A correct repeated launch on
  // the SAME stream must overwrite/zero every byte it may consume. Fresh-zero
  // allocation behavior cannot accidentally satisfy this assertion.
  CHECK(vt::cuda::testing::PoisonGdnTritonScratch(gq.q, 0xff) == 11);

  // Different data at the same shape reuses those explicitly dirty buffers.
  const auto reused_diff = RunGdnChunkedVsSequentialOnQueue(
      gpu, gq.q, {0, 150}, 16, 48, 128, 128, bf16, 7310, 3e-2f, 3e-2f);
  CheckUpstreamCutedslTolerances(reused_diff);
  auto reused = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(reused.chunk_o_bf16_launches == 2);
  CHECK(reused.chunk_o_hand_launches == 0);
  CHECK(reused.chunk_pool_allocations == first.chunk_pool_allocations);
  CHECK(reused.chunk_pool_reuses >= first.chunk_pool_reuses + 9);
  CHECK(reused.wu_pool_allocations == first.wu_pool_allocations);
  CHECK(reused.wu_pool_reuses >= first.wu_pool_reuses + 2);

  // A larger varlen batch grows every high-water buffer once. A following
  // smaller H=32 call must reuse without shrinking or allocating again.
  const auto grown_diff = RunGdnChunkedVsSequentialOnQueue(
      gpu, gq.q, {0, 100, 150, 264, 300, 401}, 16, 48, 128, 128, bf16, 7320, 3e-2f,
      3e-2f);
  CheckUpstreamCutedslTolerances(grown_diff);
  auto grown = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(grown.chunk_o_bf16_launches == 3);
  CHECK(grown.chunk_o_hand_launches == 0);
  CHECK(grown.chunk_pool_growths >= 9);
  CHECK(grown.wu_pool_growths >= 2);

  const auto smaller_diff = RunGdnChunkedVsSequentialOnQueue(
      gpu, gq.q, {0, 150}, 16, 32, 128, 128, bf16, 7330, 3e-2f, 3e-2f);
  CheckUpstreamCutedslTolerances(smaller_diff);
  auto smaller = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(smaller.chunk_o_bf16_launches == 4);
  CHECK(smaller.chunk_o_hand_launches == 0);
  CHECK(smaller.chunk_pool_allocations == grown.chunk_pool_allocations);
  CHECK(smaller.chunk_pool_growths == grown.chunk_pool_growths);
  CHECK(smaller.chunk_pool_reuses >= grown.chunk_pool_reuses + 9);
  CHECK(smaller.wu_pool_allocations == grown.wu_pool_allocations);
  CHECK(smaller.wu_pool_growths == grown.wu_pool_growths);
  CHECK(smaller.wu_pool_reuses >= grown.wu_pool_reuses + 2);

  // Same binary, pools disabled: numerical parity and Triton dispatch remain,
  // while the persistent-pool counters stay unchanged.
  setenv("VT_GDN_TRITON_CHUNK_POOL", "0", 1);
  setenv("VT_GDN_TRITON_WU_POOL", "0", 1);
  const auto disabled_diff = RunGdnChunkedVsSequentialOnQueue(
      gpu, gq.q, {0, 150}, 16, 48, 128, 128, bf16, 7340, 3e-2f, 3e-2f);
  CheckUpstreamCutedslTolerances(disabled_diff);
  auto disabled = vt::cuda::testing::GetGdnTritonDebugStats();
  CHECK(disabled.chunk_o_bf16_launches == 5);
  CHECK(disabled.chunk_o_hand_launches == 0);
  CHECK(disabled.chunk_pool_allocations == smaller.chunk_pool_allocations);
  CHECK(disabled.chunk_pool_growths == smaller.chunk_pool_growths);
  CHECK(disabled.chunk_pool_reuses == smaller.chunk_pool_reuses);
  CHECK(disabled.wu_pool_allocations == smaller.wu_pool_allocations);
  CHECK(disabled.wu_pool_growths == smaller.wu_pool_growths);
  CHECK(disabled.wu_pool_reuses == smaller.wu_pool_reuses);

  vt::cuda::testing::DisableGdnTritonDebugStats();
  unsetenv("VT_GDN_DELTAH_TRITON");
  unsetenv("VT_GDN_CHUNKO_TRITON");
  unsetenv("VT_GDN_WU_TRITON");
  unsetenv("VT_GDN_TRITON_CHUNK_POOL");
  unsetenv("VT_GDN_TRITON_WU_POOL");
  setenv("VT_GDN_CHUNKED", "1", 1);
}
#endif
