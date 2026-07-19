// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Qwen3.6 elementwise "glue" op CPU reference tests (M0.9 forward). Each op is
// exercised on the CPU backend against hand-computed reference values; these
// fuse the small host-side reshape/split/activation loops between the big decode
// ops so the whole decode step stays on-device. All math is f32; output bf16
// values are compared after the f32 -> bf16 round (via vt::F32ToBF16).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor F32(std::vector<float>& v, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), shape);
}
Tensor Bf16(std::vector<uint16_t>& v, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), shape);
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float Bf(float x) { return vt::BF16ToF32(vt::F32ToBF16(x)); }  // f32 -> bf16 -> f32
}  // namespace

// ---------------------------------------------------------------------------
// cast_bf16: out[i] = bf16(in[i]).
TEST_CASE("cast_bf16: f32 -> bf16 rounds each element") {
  std::vector<float> in = {1.0f, -2.5f, 0.333333f, 100.0f, -0.0f};
  std::vector<uint16_t> out(in.size(), 0);
  Tensor ti = F32(in, {1, static_cast<int64_t>(in.size())});
  Tensor to = Bf16(out, {1, static_cast<int64_t>(out.size())});
  Queue q = Q();
  vt::CastBf16(q, to, ti);
  for (size_t i = 0; i < in.size(); ++i)
    CHECK(vt::BF16ToF32(out[i]) == doctest::Approx(Bf(in[i])));
}

// ---------------------------------------------------------------------------
// cast_f32: out[i] = f32(in[i]) (bf16 -> f32 widen). CPU sibling of the CUDA
// CastF32 kernel; mirrors cast_bf16 so both cast directions are backend-complete
// on CPU. Widening bf16 -> f32 is exact (no rounding), so out == BF16ToF32(in).
TEST_CASE("cast_f32: bf16 -> f32 widens each element") {
  std::vector<float> src = {1.0f, -2.5f, 0.333333f, 100.0f, -0.0f};
  std::vector<uint16_t> in(src.size(), 0);
  for (size_t i = 0; i < src.size(); ++i) in[i] = vt::F32ToBF16(src[i]);
  std::vector<float> out(in.size(), 0.0f);
  Tensor ti = Bf16(in, {1, static_cast<int64_t>(in.size())});
  Tensor to = F32(out, {1, static_cast<int64_t>(out.size())});
  Queue q = Q();
  vt::CastF32(q, to, ti);
  for (size_t i = 0; i < in.size(); ++i)
    CHECK(out[i] == doctest::Approx(vt::BF16ToF32(in[i])));
}

// ---------------------------------------------------------------------------
// mul_col_vec_f32: x[m,n] *= col[n]. The per-column dequant for the merged fp8
// QKV projection (one GEMM at alpha=1, then each shard's folded scalar applied
// per output column). Exact f32 multiply, so byte-identical to scaling each
// slice by its scalar.
TEST_CASE("mul_col_vec_f32: scales each column by its scalar") {
  const int64_t M = 3, N = 4;
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> col = {0.5f, 2.0f, -1.0f, 0.25f};
  std::vector<float> ref(x.size());
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) ref[m * N + n] = x[m * N + n] * col[n];
  Tensor tx = F32(x, {M, N});
  Tensor tc = F32(col, {N});
  Queue q = Q();
  vt::MulColVecF32(q, tx, tc);
  for (size_t i = 0; i < x.size(); ++i) CHECK(x[i] == ref[i]);  // exact
}

// mul_col_vec_f32 on a ROW-STRIDED view (the real merged-QKV case: x is a
// [M,Ntot] buffer whose logical N is a shard slice, but here we exercise a padded
// row stride to prove the kernel honors stride[0] and only touches [0,N) cols).
TEST_CASE("mul_col_vec_f32: honors padded row stride") {
  const int64_t M = 2, N = 3, RS = 5;  // 5-wide rows, only first 3 cols scaled
  std::vector<float> buf(static_cast<size_t>(M) * RS, -7.0f);  // sentinel pad
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) buf[m * RS + n] = static_cast<float>(m * 10 + n + 1);
  std::vector<float> col = {3.0f, 0.5f, -2.0f};
  Tensor tx = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {M, RS});
  tx.shape[1] = N;  // logical N < row stride (strided packed view)
  Tensor tc = F32(col, {N});
  Queue q = Q();
  vt::MulColVecF32(q, tx, tc);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n)
      CHECK(buf[m * RS + n] == static_cast<float>(m * 10 + n + 1) * col[n]);
    for (int64_t n = N; n < RS; ++n) CHECK(buf[m * RS + n] == -7.0f);  // pad intact
  }
}

// ---------------------------------------------------------------------------
// attn_gate_split: qgate [T, Hq*2*Dh] -> q_out/gate_out [T,Hq,Dh].
//   T=2, Hq=2, Dh=2. Row layout per (t,hq): [q0 q1 | g0 g1].
TEST_CASE("attn_gate_split: splits [q|gate] per head") {
  const int64_t T = 2, Hq = 2, Dh = 2;
  std::vector<float> qgate;  // [T, Hq*2*Dh] = [2, 8]
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < Hq; ++h) {
      qgate.push_back(static_cast<float>(t * 100 + h * 10 + 0));  // q0
      qgate.push_back(static_cast<float>(t * 100 + h * 10 + 1));  // q1
      qgate.push_back(static_cast<float>(t * 100 + h * 10 + 5));  // g0
      qgate.push_back(static_cast<float>(t * 100 + h * 10 + 6));  // g1
    }
  std::vector<float> qo(static_cast<size_t>(T * Hq * Dh), 0.0f), go(qo.size(), 0.0f);
  Tensor tqg = F32(qgate, {T, Hq * 2 * Dh});
  Tensor tqo = F32(qo, {T, Hq, Dh}), tgo = F32(go, {T, Hq, Dh});
  Queue q = Q();
  vt::AttnGateSplit(q, tqo, tgo, tqg);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < Hq; ++h) {
      const size_t o = static_cast<size_t>((t * Hq + h) * Dh);
      CHECK(qo[o + 0] == doctest::Approx(static_cast<float>(t * 100 + h * 10 + 0)));
      CHECK(qo[o + 1] == doctest::Approx(static_cast<float>(t * 100 + h * 10 + 1)));
      CHECK(go[o + 0] == doctest::Approx(static_cast<float>(t * 100 + h * 10 + 5)));
      CHECK(go[o + 1] == doctest::Approx(static_cast<float>(t * 100 + h * 10 + 6)));
    }
}

// ---------------------------------------------------------------------------
// sigmoid_gate_bf16: out[i] = bf16(attn[i] * sigmoid(gate[i])).
TEST_CASE("sigmoid_gate_bf16: attn * sigmoid(gate), rounded to bf16") {
  std::vector<float> attn = {2.0f, -1.0f, 4.0f, 0.5f};
  std::vector<float> gate = {0.0f, 1.0f, -2.0f, 3.0f};
  std::vector<uint16_t> out(attn.size(), 0);
  Tensor ta = F32(attn, {2, 2}), tg = F32(gate, {2, 2});
  Tensor to = Bf16(out, {2, 2});
  Queue q = Q();
  vt::SigmoidGateBf16(q, to, ta, tg);
  for (size_t i = 0; i < attn.size(); ++i)
    CHECK(vt::BF16ToF32(out[i]) == doctest::Approx(Bf(attn[i] * Sigmoid(gate[i]))));
}

// bf16-attn variant (the FA-2 prefill path hands a bf16 attention out): the
// upcast is exact, so the result must be BIT-IDENTICAL to the f32-attn run over
// the same (bf16-representable) values. The gate stays f32.
TEST_CASE("sigmoid_gate_bf16: bf16 attn == f32 attn over the same values") {
  std::vector<float> attn_f = {2.0f, -1.0f, 4.0f, 0.5f};  // exactly bf16-representable
  std::vector<float> gate = {0.0f, 1.0f, -2.0f, 3.0f};
  std::vector<uint16_t> attn_b(attn_f.size(), 0);
  for (size_t i = 0; i < attn_f.size(); ++i) attn_b[i] = vt::F32ToBF16(attn_f[i]);
  std::vector<uint16_t> out_f(attn_f.size(), 0), out_b(attn_f.size(), 0);
  Tensor taf = F32(attn_f, {2, 2}), tab = Bf16(attn_b, {2, 2}), tg = F32(gate, {2, 2});
  Tensor tof = Bf16(out_f, {2, 2}), tob = Bf16(out_b, {2, 2});
  Queue q = Q();
  vt::SigmoidGateBf16(q, tof, taf, tg);
  vt::SigmoidGateBf16(q, tob, tab, tg);
  for (size_t i = 0; i < out_f.size(); ++i) CHECK(out_b[i] == out_f[i]);
}

// ---------------------------------------------------------------------------
// gdn_g_beta (gdn-semantics.md §6):
//   x = araw + dt_bias[hv];  sp = softplus(x);
//   g = -exp(a_log[hv]) * sp;  beta = sigmoid(braw).
// Also checks the softplus > 20 identity branch.
TEST_CASE("gdn_g_beta: softplus decay and sigmoid gate per head") {
  const int64_t T = 2, Hv = 2;
  std::vector<float> araw = {0.0f, 1.0f, 30.0f, -1.0f};  // [T,Hv], (1,0)=30 hits x>20
  std::vector<float> braw = {0.5f, -0.5f, 2.0f, -2.0f};
  std::vector<float> a_log = {0.0f, 0.5f};   // [Hv]
  std::vector<float> dt_bias = {0.0f, 0.5f}; // [Hv]
  std::vector<float> g(static_cast<size_t>(T * Hv), 0.0f), beta(g.size(), 0.0f);
  Tensor tar = F32(araw, {T, Hv}), tbr = F32(braw, {T, Hv});
  Tensor tal = F32(a_log, {Hv}), tdt = F32(dt_bias, {Hv});
  Tensor tg = F32(g, {T, Hv}), tbeta = F32(beta, {T, Hv});
  Queue q = Q();
  vt::GdnGBeta(q, tg, tbeta, tar, tbr, tal, tdt);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < Hv; ++h) {
      const size_t idx = static_cast<size_t>(t * Hv + h);
      const float x = araw[idx] + dt_bias[static_cast<size_t>(h)];
      const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
      const float g_ref = -std::exp(a_log[static_cast<size_t>(h)]) * sp;
      CHECK(g[idx] == doctest::Approx(g_ref));
      CHECK(beta[idx] == doctest::Approx(Sigmoid(braw[idx])));
    }
  // (t=1,h=0): x = 30 + 0 = 30 > 20, so softplus == x == 30.
  CHECK(g[2] == doctest::Approx(-std::exp(0.0f) * 30.0f));
}

// ---------------------------------------------------------------------------
// gdn_conv_split: conv [T, 2*key_dim+value_dim] -> q/k [T,key_dim], v [T,value_dim].
//   T=2, key_dim=2, value_dim=3.  Row = [q0 q1 | k0 k1 | v0 v1 v2].
TEST_CASE("gdn_conv_split: splits mixed qkv conv row") {
  const int64_t T = 2, key_dim = 2, value_dim = 3, conv_dim = 2 * key_dim + value_dim;
  std::vector<float> conv;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < conv_dim; ++j) conv.push_back(static_cast<float>(t * 100 + j));
  std::vector<float> qo(static_cast<size_t>(T * key_dim), 0.0f), ko(qo.size(), 0.0f);
  std::vector<float> vo(static_cast<size_t>(T * value_dim), 0.0f);
  Tensor tc = F32(conv, {T, conv_dim});
  Tensor tqo = F32(qo, {T, key_dim}), tko = F32(ko, {T, key_dim}), tvo = F32(vo, {T, value_dim});
  Queue q = Q();
  vt::GdnConvSplit(q, tqo, tko, tvo, tc);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < key_dim; ++j) {
      CHECK(qo[static_cast<size_t>(t * key_dim + j)] ==
            doctest::Approx(static_cast<float>(t * 100 + j)));
      CHECK(ko[static_cast<size_t>(t * key_dim + j)] ==
            doctest::Approx(static_cast<float>(t * 100 + key_dim + j)));
    }
    for (int64_t j = 0; j < value_dim; ++j)
      CHECK(vo[static_cast<size_t>(t * value_dim + j)] ==
            doctest::Approx(static_cast<float>(t * 100 + 2 * key_dim + j)));
  }
}

// ---------------------------------------------------------------------------
// shared_expert_gate: out[t,c] = bf16(sigmoid(gl[t]) * sd[t*H+c]). gl is [T]
// (one gate per token). Checks the gl [T,1] shape too (same Numel == T).
TEST_CASE("shared_expert_gate: per-token sigmoid gate over the shared MLP output") {
  const int64_t T = 2, H = 3;
  std::vector<float> sd = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // [T,H]
  std::vector<float> gl = {0.0f, 1.0f};                          // [T]
  std::vector<uint16_t> out(static_cast<size_t>(T * H), 0);
  Tensor tsd = F32(sd, {T, H}), tgl = F32(gl, {T});
  Tensor to = Bf16(out, {T, H});
  Queue q = Q();
  vt::SharedExpertGate(q, to, tsd, tgl);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t c = 0; c < H; ++c) {
      const size_t idx = static_cast<size_t>(t * H + c);
      const float ref = Sigmoid(gl[static_cast<size_t>(t)]) * sd[idx];
      CHECK(vt::BF16ToF32(out[idx]) == doctest::Approx(Bf(ref)));
    }

  // gl as [T,1] must behave identically (Numel == T).
  std::vector<float> gl2 = {0.0f, 1.0f};
  std::vector<uint16_t> out2(static_cast<size_t>(T * H), 0);
  Tensor tgl2 = F32(gl2, {T, 1});
  Tensor to2 = Bf16(out2, {T, H});
  vt::SharedExpertGate(q, to2, tsd, tgl2);
  for (size_t i = 0; i < out.size(); ++i) CHECK(out2[i] == out[i]);
}
