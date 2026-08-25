// The fused EXL3 mixture-of-experts MLP — MODEL-DSV4-EXL3 W2d.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_moe.cu:99-301             the launcher and its checks
//   exllamav3_ext/quant/exl3_moe_kernel.cuh:17-283     the fused kernel
//   exllamav3_ext/quant/exl3_moe_common.cuh:6-15       the act ids and the geometry
//   exllamav3_ext/quant/hadamard_inner.cuh:284-473     the guad and d epilogues
//   exllamav3/modules/block_sparse_mlp.py:1079-1141    the host-side batching
//
// WHAT IS GATED HERE.
//
//   * The BATCHING is host arithmetic — a bincount, a grouping and a count of
//     active experts — so it is gated on any machine, against expectations
//     computed here from upstream's own description rather than from the
//     implementation.
//   * The CPU arm of `Exl3MoeMlp` is gated against the PER-EXPERT LOOP the W2b
//     wave landed, at tier 4 (2.0e-2 relative RMS,
//     `.agents/specs/model-dsv4-exl3.md` `## W2cd design` W2d-7). The loop is
//     built here out of `vt::Exl3Gemm` calls and a transcription of vLLM's
//     `SiluAndMulWithClamp` formula, so the reference shares no code with the
//     fused arm's activation.
//   * The ACTIVATION CHOICE (W2d-2) is pinned by two directed cases, because it
//     is the one place where mirroring the kernel oracle would mirror the wrong
//     model behavior, and no shape gate can see it.
//   * The device arm SKIPS loudly and still asserts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

bool HasCudaMoe() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3MoeMlp, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

// A whole synthetic expert tower: `E` experts, each with gate (w1), up (w3) and
// down (w2) EXL3 linears at the DeepSeek-V4 orientation — gate/up are [H -> I],
// down is [I -> H].
struct MoeFixture {
  int64_t hidden = 0;
  int64_t interm = 0;
  int64_t experts = 0;
  int bits = 3;
  std::vector<Exl3Fixture> gate, up, down;
};

MoeFixture MakeMoeFixture(int64_t hidden, int64_t interm, int64_t experts, uint32_t seed) {
  MoeFixture m;
  m.hidden = hidden;
  m.interm = interm;
  m.experts = experts;
  for (int64_t e = 0; e < experts; ++e) {
    m.gate.push_back(MakeFixture(hidden, interm, 3, seed + static_cast<uint32_t>(e) * 7u + 1u));
    m.up.push_back(MakeFixture(hidden, interm, 3, seed + static_cast<uint32_t>(e) * 7u + 2u));
    m.down.push_back(MakeFixture(interm, hidden, 3, seed + static_cast<uint32_t>(e) * 7u + 3u));
  }
  return m;
}

// One EXL3 linear through the op the W2b wave landed. This is the LOOP arm, and
// it is the tier-4 reference.
std::vector<float> LoopLinear(vt::Queue& q, const Exl3Fixture& f, const float* xin) {
  const int64_t k = f.k, n = f.n;
  std::vector<uint16_t> a(static_cast<size_t>(k)), ah(static_cast<size_t>(k), 0),
      c(static_cast<size_t>(n), 0);
  for (int64_t i = 0; i < k; ++i) a[static_cast<size_t>(i)] = vt::F32ToF16(xin[i]);
  vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF16, q.device, {1, k});
  vt::Tensor tah = vt::Tensor::Contiguous(ah.data(), vt::DType::kF16, q.device, {1, k});
  vt::Tensor tc = vt::Tensor::Contiguous(c.data(), vt::DType::kF16, q.device, {1, n});
  vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8,
                                         q.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, q.device, {k});
  vt::Tensor tsvh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, q.device, {n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = 1;
  vt::Exl3Gemm(q, tc, ta, tb, tsuh, tsvh, tah, args);
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::F16ToF32(c[static_cast<size_t>(i)]);
  return out;
}

// vLLM's `SiluAndMulWithClamp` (model_executor/layers/activation.py:197-201),
// transcribed from the formula, with alpha = 1 and beta = 0 as
// `DeepseekV4MLP` passes them (nvidia/model.py:131):
//   gate = min(g, limit)                  (MAX only)
//   up   = clamp(u, -limit, limit)        (BOTH sides)
//   out  = gate * sigmoid(gate) * up
std::vector<float> VllmSiluAndMulClamp(const std::vector<float>& g, const std::vector<float>& u,
                                       float limit) {
  std::vector<float> out(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    const float gate = std::min(g[i], limit);
    const float up = std::min(std::max(u[i], -limit), limit);
    out[i] = gate / (1.0f + std::exp(-gate)) * up;
  }
  return out;
}

// Upstream's own order (`hadamard_inner.cuh:110-122`), for the directed case
// that pins WHICH of the two this port implements: the clamp lands AFTER the
// silu, and a zero limit means NO limit.
std::vector<float> Exl3SiluClampAfter(const std::vector<float>& g, const std::vector<float>& u,
                                      float limit) {
  std::vector<float> out(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    float vg = g[i] / (1.0f + std::exp(-g[i]));
    float vu = u[i];
    if (limit != 0.0f) {
      vu = std::min(std::max(vu, -limit), limit);
      vg = std::min(vg, limit);
    }
    out[i] = vg * vu;
  }
  return out;
}

// The whole routed-expert contribution, computed the way `MoeBlock`'s W2b arm
// computes it: one `Exl3Gemm` per (token, expert, projection), the activation in
// f32 on the host, the weighted sum accumulated in f32.
std::vector<float> LoopReference(vt::Queue& q, const MoeFixture& m, const std::vector<float>& x,
                                 int64_t tokens, const std::vector<int64_t>& ids,
                                 const std::vector<float>& weights, int64_t topk, float limit) {
  std::vector<float> out(static_cast<size_t>(tokens * m.hidden), 0.0f);
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t e = ids[static_cast<size_t>(t * topk + j)];
      const float w = weights[static_cast<size_t>(t * topk + j)];
      const std::vector<float> g = LoopLinear(q, m.gate[static_cast<size_t>(e)], &x[t * m.hidden]);
      const std::vector<float> u = LoopLinear(q, m.up[static_cast<size_t>(e)], &x[t * m.hidden]);
      const std::vector<float> act = VllmSiluAndMulClamp(g, u, limit);
      const std::vector<float> eo = LoopLinear(q, m.down[static_cast<size_t>(e)], act.data());
      for (int64_t h = 0; h < m.hidden; ++h)
        out[static_cast<size_t>(t * m.hidden + h)] += w * eo[static_cast<size_t>(h)];
    }
  }
  return out;
}

// Everything the fused op needs, kept alive for the duration of a call.
struct MoeCall {
  std::vector<int64_t> gate_tr, gate_su, gate_sv, up_tr, up_su, up_sv, down_tr, down_su, down_sv;
  std::vector<int64_t> expert_count, token_sorted;
  std::vector<uint16_t> weight_sorted;
  std::vector<uint16_t> hidden;
  std::vector<uint16_t> state_g, state_u, interm_g, interm_u;
  int num_active = 0;
};

std::vector<float> FusedCall(vt::Queue& q, const MoeFixture& m, const std::vector<float>& x,
                             int64_t tokens, const std::vector<int64_t>& ids,
                             const std::vector<float>& weights, int64_t topk, float limit,
                             vt::Exl3MoeAct act, int64_t max_rows, MoeCall* keep) {
  MoeCall& c = *keep;
  for (int64_t e = 0; e < m.experts; ++e) {
    const size_t i = static_cast<size_t>(e);
    c.gate_tr.push_back(reinterpret_cast<int64_t>(m.gate[i].trellis.data()));
    c.gate_su.push_back(reinterpret_cast<int64_t>(m.gate[i].suh.data()));
    c.gate_sv.push_back(reinterpret_cast<int64_t>(m.gate[i].svh.data()));
    c.up_tr.push_back(reinterpret_cast<int64_t>(m.up[i].trellis.data()));
    c.up_su.push_back(reinterpret_cast<int64_t>(m.up[i].suh.data()));
    c.up_sv.push_back(reinterpret_cast<int64_t>(m.up[i].svh.data()));
    c.down_tr.push_back(reinterpret_cast<int64_t>(m.down[i].trellis.data()));
    c.down_su.push_back(reinterpret_cast<int64_t>(m.down[i].suh.data()));
    c.down_sv.push_back(reinterpret_cast<int64_t>(m.down[i].svh.data()));
  }
  const int64_t assignments = tokens * topk;
  c.expert_count.assign(static_cast<size_t>(m.experts + 1), 0);
  c.token_sorted.assign(static_cast<size_t>(assignments), 0);
  c.weight_sorted.assign(static_cast<size_t>(assignments), 0);
  c.num_active = vt::Exl3MoeSortTokensByExpert(ids.data(), weights.data(), tokens, topk, m.experts,
                                               max_rows, c.expert_count.data(),
                                               c.token_sorted.data(), c.weight_sorted.data());

  c.hidden.assign(static_cast<size_t>(tokens * m.hidden), 0);
  for (size_t i = 0; i < c.hidden.size(); ++i) c.hidden[i] = vt::F32ToF16(x[i]);
  c.state_g.assign(static_cast<size_t>(max_rows * m.hidden), 0);
  c.state_u.assign(static_cast<size_t>(max_rows * m.hidden), 0);
  c.interm_g.assign(static_cast<size_t>(max_rows * m.interm), 0);
  c.interm_u.assign(static_cast<size_t>(max_rows * m.interm), 0);

  std::vector<float> out(static_cast<size_t>(tokens * m.hidden), 0.0f);
  const vt::Device d = q.device;
  vt::Tensor t_out = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, d, {tokens, m.hidden});
  vt::Tensor t_hid = vt::Tensor::Contiguous(c.hidden.data(), vt::DType::kF16, d, {tokens, m.hidden});
  auto ptr_tensor = [&](std::vector<int64_t>& v) {
    return vt::Tensor::Contiguous(v.data(), vt::DType::kI64, d, {m.experts});
  };
  vt::Tensor gt = ptr_tensor(c.gate_tr), gs = ptr_tensor(c.gate_su), gv = ptr_tensor(c.gate_sv);
  vt::Tensor ut = ptr_tensor(c.up_tr), us = ptr_tensor(c.up_su), uv = ptr_tensor(c.up_sv);
  vt::Tensor dt = ptr_tensor(c.down_tr), ds = ptr_tensor(c.down_su), dv = ptr_tensor(c.down_sv);
  vt::Exl3MoeExpertTables tables;
  tables.gate_trellis = &gt;
  tables.gate_suh = &gs;
  tables.gate_svh = &gv;
  tables.up_trellis = &ut;
  tables.up_suh = &us;
  tables.up_svh = &uv;
  tables.down_trellis = &dt;
  tables.down_suh = &ds;
  tables.down_svh = &dv;

  vt::Tensor t_cnt =
      vt::Tensor::Contiguous(c.expert_count.data(), vt::DType::kI64, d, {m.experts + 1});
  vt::Tensor t_tok = vt::Tensor::Contiguous(c.token_sorted.data(), vt::DType::kI64, d, {assignments});
  vt::Tensor t_wgt =
      vt::Tensor::Contiguous(c.weight_sorted.data(), vt::DType::kF16, d, {assignments});
  vt::Exl3MoeRouting routing;
  routing.expert_count = &t_cnt;
  routing.token_sorted = &t_tok;
  routing.weight_sorted = &t_wgt;

  vt::Tensor sg = vt::Tensor::Contiguous(c.state_g.data(), vt::DType::kF16, d, {1, max_rows, m.hidden});
  vt::Tensor su = vt::Tensor::Contiguous(c.state_u.data(), vt::DType::kF16, d, {1, max_rows, m.hidden});
  vt::Tensor ig = vt::Tensor::Contiguous(c.interm_g.data(), vt::DType::kF16, d, {1, max_rows, m.interm});
  vt::Tensor iu = vt::Tensor::Contiguous(c.interm_u.data(), vt::DType::kF16, d, {1, max_rows, m.interm});
  vt::Exl3MoeTemps temps;
  temps.state_g = &sg;
  temps.state_u = &su;
  temps.intermediate_g = &ig;
  temps.intermediate_u = &iu;

  vt::Exl3MoeArgs args;
  args.bits_gate = args.bits_up = args.bits_down = m.bits;
  args.codebook = 1;
  args.act = act;
  args.act_limit = limit;
  args.num_active = c.num_active;
  vt::Exl3MoeMlp(q, t_out, t_hid, tables, routing, temps, args);
  return out;
}

double RelRms(const std::vector<float>& a, const std::vector<float>& b) {
  double sq = 0.0, rq = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    sq += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    rq += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(sq / static_cast<double>(a.size())) /
         std::sqrt(rq / static_cast<double>(a.size()));
}

}  // namespace

// ─── W2d-3: the batching is upstream's, and it is host arithmetic ────────────

TEST_CASE("exl3 moe: the token sort reproduces upstream's bincount and grouping") {
  // 3 tokens, top-2 of 4 experts. Assignments, in token-major order:
  //   t0 -> e2, e0 ; t1 -> e2, e2 ; t2 -> e1, e2
  const std::vector<int64_t> ids = {2, 0, 2, 2, 1, 2};
  const std::vector<float> w = {0.5f, 0.25f, 0.125f, 0.0625f, 1.0f, 2.0f};
  const int64_t tokens = 3, topk = 2, experts = 4;
  std::vector<int64_t> count(static_cast<size_t>(experts + 1), -1);
  std::vector<int64_t> tok(static_cast<size_t>(tokens * topk), -1);
  std::vector<uint16_t> wgt(static_cast<size_t>(tokens * topk), 0);
  const int active = vt::Exl3MoeSortTokensByExpert(ids.data(), w.data(), tokens, topk, experts,
                                                   /*max_tokens_per_expert=*/128, count.data(),
                                                   tok.data(), wgt.data());

  // `bincount(flat_expert, minlength = E + 1)` (block_sparse_mlp.py:1100). The
  // trailing slot is upstream's sentinel for an out-of-shard assignment and a
  // single-shard load never fills it.
  CHECK(count[0] == 1);
  CHECK(count[1] == 1);
  CHECK(count[2] == 4);
  CHECK(count[3] == 0);
  CHECK(count[4] == 0);

  // The assignments grouped by expert, and inside a group in the order the
  // tokens appear — a stable sort where upstream calls the unstable
  // `Tensor.argsort` (`## W2cd design` W2d-3 records the deviation).
  const std::vector<int64_t> want_tok = {0, 2, 0, 1, 1, 2};
  CHECK(tok == want_tok);
  const std::vector<float> want_w = {0.25f, 1.0f, 0.5f, 0.125f, 0.0625f, 2.0f};
  for (size_t i = 0; i < want_w.size(); ++i)
    CHECK(vt::F16ToF32(wgt[i]) == doctest::Approx(want_w[i]));

  // `num_active` = experts with 0 < count <= max_tokens_per_expert (:1105).
  CHECK(active == 3);
}

TEST_CASE("exl3 moe: an expert over max_tokens_per_expert is NOT counted active") {
  // All six assignments land on e0. With a cut of 5 the fused arm declines it,
  // and the caller's per-expert loop is what covers it (W2d-4).
  const std::vector<int64_t> ids = {0, 0, 0, 0, 0, 0};
  const std::vector<float> w(6, 1.0f);
  std::vector<int64_t> count(3, -1), tok(6, -1);
  std::vector<uint16_t> wgt(6, 0);
  CHECK(vt::Exl3MoeSortTokensByExpert(ids.data(), w.data(), 3, 2, 2, 5, count.data(), tok.data(),
                                      wgt.data()) == 0);
  CHECK(count[0] == 6);
  // One more than the cut is out; exactly the cut is in.
  CHECK(vt::Exl3MoeSortTokensByExpert(ids.data(), w.data(), 3, 2, 2, 6, count.data(), tok.data(),
                                      wgt.data()) == 1);
  // An expert with NO tokens is not active either (`0 < c`).
  CHECK(count[1] == 0);
}

TEST_CASE("exl3 moe: the concurrency is upstream's SM division, clamped") {
  // exl3_moe.cu:14-18, num_sms / MOE_SMS_PER_EXPERT, capped at MOE_MAX_GROUPS
  // (exl3_devctx.cuh:13) and never below one.
  CHECK(vt::kExl3MoeSmsPerExpert == 8);
  CHECK(vt::kExl3MoeMaxGroups == 64);
  CHECK(vt::kExl3MoeTempRowsFused == 128);
  CHECK(vt::Exl3MoeMaxConcurrency(80) == 10);
  CHECK(vt::Exl3MoeMaxConcurrency(8) == 1);
  CHECK(vt::Exl3MoeMaxConcurrency(1) == 1);   // never zero groups
  CHECK(vt::Exl3MoeMaxConcurrency(0) == 1);
  CHECK(vt::Exl3MoeMaxConcurrency(4096) == 64);  // capped
}

// ─── W2d-7: tier 4, the fused arm against the per-expert loop arm ────────────

TEST_CASE("exl3 moe: the fused CPU arm agrees with the per-expert loop within tier 4") {
  vt::Queue q = CpuQueue();
  const int64_t H = 256, I = 128, E = 4, T = 3, topk = 2;
  const MoeFixture m = MakeMoeFixture(H, I, E, 0xA11CEu);
  Rng rng;
  std::vector<float> x(static_cast<size_t>(T * H));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(0.5f)));
  const std::vector<int64_t> ids = {2, 0, 3, 2, 1, 2};
  const std::vector<float> w = {0.5f, 0.25f, 0.75f, 0.125f, 0.4f, 0.6f};
  const float limit = 10.0f;

  const std::vector<float> ref = LoopReference(q, m, x, T, ids, w, topk, limit);
  MoeCall keep;
  const std::vector<float> got =
      FusedCall(q, m, x, T, ids, w, topk, limit, vt::Exl3MoeAct::kSiluAndMulClamp, 128, &keep);

  const double rel = RelRms(got, ref);
  MESSAGE("tier 4: fused vs loop relative RMS ", rel);
  CHECK(rel <= 2.0e-2);
  // The bound has to DISCRIMINATE, so the same comparison against an unrelated
  // routing must blow through it. Without this the case would pass on a fused
  // arm that ignored `token_sorted` entirely.
  const std::vector<int64_t> other = {0, 1, 1, 3, 3, 0};
  const std::vector<float> wrong = LoopReference(q, m, x, T, other, w, topk, limit);
  const double discrim = RelRms(wrong, ref);
  MESSAGE("tier 4 discrimination: an unrelated routing is ", discrim, " away");
  CHECK(discrim > 1.0e-1);
}

// ─── W2d-2: the activation is vLLM's, and these two cases say so ─────────────

TEST_CASE("exl3 moe: the clamp lands BEFORE the silu, as vLLM does, not after") {
  // A limit small enough that it binds on this fixture, so the two orders
  // genuinely disagree. `## W2cd design` W2d-2 predicts the gap: for a gate
  // value above the limit upstream yields `limit` and vLLM yields
  // `silu(limit) = limit * sigmoid(limit)`.
  const std::vector<float> g = {0.0f, 0.25f, 3.0f, 12.0f};
  const std::vector<float> u = {1.0f, 1.0f, 1.0f, 1.0f};
  const float limit = 0.5f;
  const std::vector<float> vllm = VllmSiluAndMulClamp(g, u, limit);
  const std::vector<float> exl3 = Exl3SiluClampAfter(g, u, limit);
  // They agree below the limit and diverge above it, which is what makes the
  // choice observable at all.
  CHECK(vllm[0] == doctest::Approx(exl3[0]));
  CHECK(vllm[3] != doctest::Approx(exl3[3]));
  // `up` is clamped to the limit too (BOTH sides), so each product carries a
  // factor of `limit`: upstream's gate saturates at `limit`, vLLM's at
  // `silu(limit)`.
  CHECK(exl3[3] == doctest::Approx(limit * limit));
  CHECK(vllm[3] == doctest::Approx(limit / (1.0f + std::exp(-limit)) * limit));

  // At DeepSeek-V4-Flash's own limit the gap is the number the spec states.
  const std::vector<float> big_g = {50.0f};
  const std::vector<float> one = {1.0f};
  const std::vector<float> a = VllmSiluAndMulClamp(big_g, one, 10.0f);
  const std::vector<float> b = Exl3SiluClampAfter(big_g, one, 10.0f);
  CHECK((b[0] - a[0]) == doctest::Approx(10.0f / (1.0f + std::exp(10.0f))).epsilon(1e-3));

  // And the fused arm is on vLLM's side of that. `kSilu` is kept as upstream's
  // own arm and must land on the other side.
  vt::Queue q = CpuQueue();
  const int64_t H = 256, I = 128, E = 2, T = 1, topk = 1;
  const MoeFixture m = MakeMoeFixture(H, I, E, 0xBEE5u);
  Rng rng;
  std::vector<float> x(static_cast<size_t>(T * H));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(0.5f)));
  const std::vector<int64_t> ids = {1};
  const std::vector<float> w = {1.0f};

  MoeCall k1, k2;
  const std::vector<float> as_vllm =
      FusedCall(q, m, x, T, ids, w, topk, limit, vt::Exl3MoeAct::kSiluAndMulClamp, 128, &k1);
  const std::vector<float> as_exl3 =
      FusedCall(q, m, x, T, ids, w, topk, limit, vt::Exl3MoeAct::kSilu, 128, &k2);
  const std::vector<float> ref = LoopReference(q, m, x, T, ids, w, topk, limit);
  CHECK(RelRms(as_vllm, ref) <= 2.0e-2);
  CHECK(RelRms(as_exl3, ref) > 2.0e-2);
}

TEST_CASE("exl3 moe: a zero limit means ZERO for vLLM's arm and NO LIMIT for upstream's") {
  // The two ports read the value 0 differently and that is not a bug in either:
  // upstream guards its clamp with `act_limit != 0.0f`
  // (`hadamard_inner.cuh:110`), while vLLM's `SiluAndMulWithClamp` always
  // applies the limit, so a limit of 0 zeroes the `up` factor
  // (`deepseek_v4_weights.cpp:1034` already records that degenerate case).
  vt::Queue q = CpuQueue();
  const int64_t H = 256, I = 128, E = 2, T = 1, topk = 1;
  const MoeFixture m = MakeMoeFixture(H, I, E, 0xC0FFEEu);
  Rng rng;
  std::vector<float> x(static_cast<size_t>(T * H));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(0.5f)));
  const std::vector<int64_t> ids = {0};
  const std::vector<float> w = {1.0f};

  MoeCall k1, k2;
  const std::vector<float> zeroed =
      FusedCall(q, m, x, T, ids, w, topk, 0.0f, vt::Exl3MoeAct::kSiluAndMulClamp, 128, &k1);
  double mx = 0.0;
  for (float v : zeroed) mx = std::max(mx, std::fabs(static_cast<double>(v)));
  CHECK(mx == 0.0);

  const std::vector<float> unlimited =
      FusedCall(q, m, x, T, ids, w, topk, 0.0f, vt::Exl3MoeAct::kSilu, 128, &k2);
  double mx2 = 0.0;
  for (float v : unlimited) mx2 = std::max(mx2, std::fabs(static_cast<double>(v)));
  CHECK(mx2 > 0.0);
}

// ─── the refusals ────────────────────────────────────────────────────────────

TEST_CASE("exl3 moe: unrepresentable inputs REFUSE BY NAME") {
  vt::Queue q = CpuQueue();
  const int64_t H = 256, I = 128, E = 2, T = 1, topk = 1;
  const MoeFixture m = MakeMoeFixture(H, I, E, 0xD00Du);
  std::vector<float> x(static_cast<size_t>(T * H), 0.25f);
  const std::vector<int64_t> ids = {0};
  const std::vector<float> w = {1.0f};

  auto refusal = [&](vt::Exl3MoeAct act, int codebook, bool bad_out_dtype) -> std::string {
    MoeCall keep;
    try {
      if (codebook != 1 || bad_out_dtype) {
        // Reach the checks through a hand-built call so one field at a time is
        // wrong; FusedCall always builds a valid one.
        MoeCall c;
        std::vector<float> out(static_cast<size_t>(T * H), 0.0f);
        std::vector<uint16_t> out16(static_cast<size_t>(T * H), 0);
        c.hidden.assign(static_cast<size_t>(T * H), 0);
        for (size_t i = 0; i < c.hidden.size(); ++i) c.hidden[i] = vt::F32ToF16(x[i]);
        c.expert_count.assign(static_cast<size_t>(E + 1), 0);
        c.token_sorted.assign(1, 0);
        c.weight_sorted.assign(1, vt::F32ToF16(1.0f));
        c.expert_count[0] = 1;
        c.state_g.assign(static_cast<size_t>(128 * H), 0);
        c.state_u.assign(static_cast<size_t>(128 * H), 0);
        c.interm_g.assign(static_cast<size_t>(128 * I), 0);
        c.interm_u.assign(static_cast<size_t>(128 * I), 0);
        for (int64_t e = 0; e < E; ++e) {
          const size_t i = static_cast<size_t>(e);
          c.gate_tr.push_back(reinterpret_cast<int64_t>(m.gate[i].trellis.data()));
          c.gate_su.push_back(reinterpret_cast<int64_t>(m.gate[i].suh.data()));
          c.gate_sv.push_back(reinterpret_cast<int64_t>(m.gate[i].svh.data()));
          c.up_tr.push_back(reinterpret_cast<int64_t>(m.up[i].trellis.data()));
          c.up_su.push_back(reinterpret_cast<int64_t>(m.up[i].suh.data()));
          c.up_sv.push_back(reinterpret_cast<int64_t>(m.up[i].svh.data()));
          c.down_tr.push_back(reinterpret_cast<int64_t>(m.down[i].trellis.data()));
          c.down_su.push_back(reinterpret_cast<int64_t>(m.down[i].suh.data()));
          c.down_sv.push_back(reinterpret_cast<int64_t>(m.down[i].svh.data()));
        }
        const vt::Device d = q.device;
        auto pt = [&](std::vector<int64_t>& v) {
          return vt::Tensor::Contiguous(v.data(), vt::DType::kI64, d, {E});
        };
        vt::Tensor gt = pt(c.gate_tr), gs = pt(c.gate_su), gv = pt(c.gate_sv);
        vt::Tensor ut = pt(c.up_tr), us = pt(c.up_su), uv = pt(c.up_sv);
        vt::Tensor dt = pt(c.down_tr), ds = pt(c.down_su), dv = pt(c.down_sv);
        vt::Exl3MoeExpertTables tb{&gt, &gs, &gv, &ut, &us, &uv, &dt, &ds, &dv};
        vt::Tensor tc = vt::Tensor::Contiguous(c.expert_count.data(), vt::DType::kI64, d, {E + 1});
        vt::Tensor tt = vt::Tensor::Contiguous(c.token_sorted.data(), vt::DType::kI64, d, {1});
        vt::Tensor tw = vt::Tensor::Contiguous(c.weight_sorted.data(), vt::DType::kF16, d, {1});
        vt::Exl3MoeRouting rt{&tc, &tt, &tw};
        vt::Tensor sg =
            vt::Tensor::Contiguous(c.state_g.data(), vt::DType::kF16, d, {1, 128, H});
        vt::Tensor su =
            vt::Tensor::Contiguous(c.state_u.data(), vt::DType::kF16, d, {1, 128, H});
        vt::Tensor ig =
            vt::Tensor::Contiguous(c.interm_g.data(), vt::DType::kF16, d, {1, 128, I});
        vt::Tensor iu =
            vt::Tensor::Contiguous(c.interm_u.data(), vt::DType::kF16, d, {1, 128, I});
        vt::Exl3MoeTemps tp{&sg, &su, &ig, &iu};
        vt::Tensor t_hid =
            vt::Tensor::Contiguous(c.hidden.data(), vt::DType::kF16, d, {T, H});
        vt::Tensor t_out =
            bad_out_dtype
                ? vt::Tensor::Contiguous(out16.data(), vt::DType::kF16, d, {T, H})
                : vt::Tensor::Contiguous(out.data(), vt::DType::kF32, d, {T, H});
        vt::Exl3MoeArgs a;
        a.bits_gate = a.bits_up = a.bits_down = 3;
        a.codebook = codebook;
        a.act = act;
        a.num_active = 1;
        vt::Exl3MoeMlp(q, t_out, t_hid, tb, rt, tp, a);
      } else {
        (void)FusedCall(q, m, x, T, ids, w, topk, 10.0f, act, 128, &keep);
      }
    } catch (const std::exception& e) {
      return e.what();
    }
    return "";
  };

  // A codebook this row does not decode. The artifact is mcg (cb 1).
  const std::string cb = refusal(vt::Exl3MoeAct::kSiluAndMulClamp, 2, false);
  CHECK(cb.find("exl3_moe") != std::string::npos);
  CHECK(cb.find("codebook") != std::string::npos);

  // The output accumulator is f32 and that is upstream's own width
  // (exl3_moe.cu:151); an fp16 destination is not a narrowing we may take
  // silently, because the scatter-add sums one contribution per active expert.
  const std::string od = refusal(vt::Exl3MoeAct::kSiluAndMulClamp, 1, true);
  CHECK(od.find("exl3_moe") != std::string::npos);
  CHECK(od.find("f32") != std::string::npos);
}

// ─── the device arm ──────────────────────────────────────────────────────────

TEST_CASE("exl3 device: the fused MoE arm agrees with the CPU arm within tier 4") {
  if (!HasCudaMoe()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2d's device parity is PENDING, and so is "
        "the launch-count claim (3 * topk * T -> 1 per layer). dgx.casa is flapping. "
        "Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_moe -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3MoeMlp, vt::DeviceType::kCUDA));
    return;
  }
  // The fused kernel dereferences the per-expert pointer tables ON THE DEVICE,
  // so this case needs a device-resident tower, which is the row's own owed
  // "Real-checkpoint residency for the coalesced tower". Until it lands the
  // device arm is reachable only where host pointers are dereferenceable.
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  if (!cb.DeviceMemoryIsHostAddressable()) {
    MESSAGE(
        "SKIPPED, CUDA present but host pointers are not dereferenceable: the coalesced EXL3 "
        "tower is host-resident (MODEL-DSV4-EXL3 `## Owed`, device residency).");
    CHECK_FALSE(cb.DeviceMemoryIsHostAddressable());
    return;
  }
  vt::Queue dq = cb.CreateQueue();
  vt::Queue hq = CpuQueue();
  const int64_t H = 256, I = 128, E = 4, T = 3, topk = 2;
  const MoeFixture m = MakeMoeFixture(H, I, E, 0xA11CEu);
  Rng rng;
  std::vector<float> x(static_cast<size_t>(T * H));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(0.5f)));
  const std::vector<int64_t> ids = {2, 0, 3, 2, 1, 2};
  const std::vector<float> w = {0.5f, 0.25f, 0.75f, 0.125f, 0.4f, 0.6f};

  MoeCall kh, kd;
  const std::vector<float> host =
      FusedCall(hq, m, x, T, ids, w, topk, 10.0f, vt::Exl3MoeAct::kSiluAndMulClamp, 128, &kh);
  const std::vector<float> dev =
      FusedCall(dq, m, x, T, ids, w, topk, 10.0f, vt::Exl3MoeAct::kSiluAndMulClamp, 128, &kd);
  cb.Synchronize(dq);
  const double rel = RelRms(dev, host);
  MESSAGE("tier 4, device vs CPU: relative RMS ", rel);
  CHECK(rel <= 2.0e-2);
  cb.DestroyQueue(dq);
}
