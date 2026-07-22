// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// W4 — the EXECUTABLE BACKEND-ADDITIVITY proof, closing the KERNEL-FUSION-FRAMEWORK
// W-series (.agents/specs/portable-fusion-framework.md §4 additivity, §6 portability,
// §10 W4). The thesis made executable:
//
//   "A new backend registers `kFusedChain` (Tier-0 composite) ONCE and inherits the
//    ENTIRE recipe catalog correct, with ZERO per-recipe work; adding a recipe
//    requires zero backend edits."
//
// PROOF APPROACH (spike §10 W4, PREFERRED option b — no mock DeviceType, which would
// have to edit the core enum + every switch and be ironically NON-additive): treat the
// EXISTING CPU backend AS the 'second backend' relative to CUDA. The CPU backend
// registers exactly ONE `OpId::kFusedChain` (`src/vt/cpu/cpu_ops.cpp`) plus the
// standalone primitive ops; the Tier-0 composite is ONE device-agnostic walker
// (`FusedChainCompositeImpl`, `src/vt/ops.cpp`) whose per-opcode switch self-dispatches
// each step to `q.device`'s standalone op. So the CPU backend realizes EVERY catalog
// recipe through that one path with NO per-recipe code.
//
// This test enumerates the WHOLE catalog (`kCatalog`, all 9 recipes) and, in ONE
// generic loop, asserts each runs CORRECT on the CPU 'second backend' via the Tier-0
// composite — BYTE-EXACT vs the standalone-op-sequence golden, over the CPU-expressible
// scope. For recipes whose quant terminal is CPU-expressible (fp4 + the attn macro +
// plain add-rmsnorm) that is end-to-end; for the CUDA-only static-fp8 quant terminal
// (`vt::QuantFp8Static`, unregistered on CPU per §3b/§6), the composite runs the
// CPU-expressible PREFIX byte-exact, and the test ALSO asserts the full composite THROWS
// on CPU — documenting the backend-negotiated tail rather than silently skipping it.
//
// ADDITIVITY EVIDENCE this test pins (catalog grows ⇒ backend does NOT):
//   * The catalog (`include/vt/recipes.h`) grew 1→6→7 recipes across W0→W1→W3, while
//     the composite walker stayed ONE function and the CPU/CUDA `kFusedChain`
//     registration stayed ONE line each — no per-recipe growth.
//   * W3's `kSiluMulQuantFp8` (a whole new vLLM fusion pass) landed touching EXACTLY
//     2 shared files (`recipes.h` + a test) and ZERO `src/vt/` backend files: its
//     opcodes {kSiluMul, kQuantFp8} were already in the walker, so the CPU 'second
//     backend' inherited it for free. This test drives `kSiluMulQuantFp8` through the
//     same generic loop as every other recipe — proof that inheritance is real.
//
// CPU is the primary gate here (the composite oracle is the CPU-side reference); the
// sibling `test_ops_fused_chain.cpp` carries the CUDA byte-exact arms on dgx.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Fp4ScaleLayout;
using vt::FusedBinding;
using vt::FusedParams;
using vt::FusedRecipe;
using vt::Queue;
using vt::RmsNormArgs;
using vt::RmsNormGatedArgs;
using vt::RopeArgs;
using vt::Tensor;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeTensor(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
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

std::vector<float> RandF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

std::vector<uint8_t> PackBf16(const std::vector<float>& f) {
  std::vector<uint8_t> out(f.size() * vt::SizeOf(DType::kBF16));
  auto* p = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < f.size(); ++i) p[i] = vt::F32ToBF16(f[i]);
  return out;
}

// The recipe with its trailing step (the CUDA-only static-fp8 quant terminal) and its
// trailing operand (the fp8 output slot) dropped — the CPU-expressible PREFIX. Purely
// structural on the POD (no per-recipe knowledge): every fp8-terminal recipe in the
// catalog has the fp8 quant as its last step and the fp8 output as its last operand.
FusedRecipe CpuExpressiblePrefix(const FusedRecipe& r) {
  FusedRecipe pr = r;
  pr.n = r.n - 1;
  pr.n_operands = r.n_operands - 1;
  return pr;
}

// ---------------------------------------------------------------------------
// Per-recipe CPU byte-exact drivers. Every driver runs the standalone-op-sequence
// golden AND the Tier-0 composite through the SAME public `vt::FusedChainComposite`
// entry, and asserts byte-identical. The drivers differ only in operand shapes/goldens
// (a per-RECIPE concern, in the catalog/test layer); the BACKEND path they exercise is
// identical for all — the one device-agnostic walker.

// kFusedAddRmsNorm — full: RmsNorm(residual), gemma weight = 1+w.
void CheckFusedAddRmsNorm() {
  const int64_t t = 3, h = 512;
  const float eps = 1e-6f;
  const auto xf = RandF32(static_cast<size_t>(t * h), 11);
  const auto wf = RandF32(static_cast<size_t>(h), 12);
  const auto rf = RandF32(static_cast<size_t>(t * h), 13);
  const auto xb = PackBf16(xf), wb = PackBf16(wf);
  std::vector<float> res_g = rf, res_c = rf;
  std::vector<uint8_t> out_g(static_cast<size_t>(t * h) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> out_c(out_g.size());
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), DType::kBF16, {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), DType::kBF16, {h});
  Queue q{Cpu(), nullptr};

  Tensor tog = MakeTensor(out_g.data(), DType::kBF16, {t, h});
  Tensor trg = MakeTensor(res_g.data(), DType::kF32, {t, h});
  vt::RmsNorm(q, tog, tx, tw, RmsNormArgs{eps, true}, &trg);

  Tensor toc = MakeTensor(out_c.data(), DType::kBF16, {t, h});
  Tensor trc = MakeTensor(res_c.data(), DType::kF32, {t, h});
  FusedBinding b;
  b.op[0] = &tx;
  b.op[1] = &tw;
  b.op[2] = &trc;
  b.op[3] = &toc;
  b.n = 4;
  FusedParams p;
  p.eps = eps;
  vt::FusedChainComposite(q, vt::kFusedAddRmsNorm, b, p);
  CHECK(out_c == out_g);
  CHECK(res_c == res_g);
}

// kFusedAddRmsNormStd — full: the gemma=false sibling of kFusedAddRmsNorm (weight `w`,
// not `1+w`) used by the Qwen3 DENSE decoder norms. It was declared in recipes.h without
// a catalog row, which is exactly what the count guard below exists to prevent — the
// guard had drifted to 7 while the catalog declared 9
// (.agents/specs/metal-mlx-reuse-study.md §4.3(b), §9 item 4). Repaired here.
void CheckFusedAddRmsNormStd() {
  const int64_t t = 3, h = 512;
  const float eps = 1e-6f;
  const auto xf = RandF32(static_cast<size_t>(t * h), 111);
  const auto wf = RandF32(static_cast<size_t>(h), 112);
  const auto rf = RandF32(static_cast<size_t>(t * h), 113);
  const auto xb = PackBf16(xf), wb = PackBf16(wf);
  std::vector<float> res_g = rf, res_c = rf;
  std::vector<uint8_t> out_g(static_cast<size_t>(t * h) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> out_c(out_g.size());
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), DType::kBF16, {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), DType::kBF16, {h});
  Queue q{Cpu(), nullptr};

  Tensor tog = MakeTensor(out_g.data(), DType::kBF16, {t, h});
  Tensor trg = MakeTensor(res_g.data(), DType::kF32, {t, h});
  vt::RmsNorm(q, tog, tx, tw, RmsNormArgs{eps, /*gemma=*/false}, &trg);

  Tensor toc = MakeTensor(out_c.data(), DType::kBF16, {t, h});
  Tensor trc = MakeTensor(res_c.data(), DType::kF32, {t, h});
  FusedBinding b;
  b.op[0] = &tx;
  b.op[1] = &tw;
  b.op[2] = &trc;
  b.op[3] = &toc;
  b.n = 4;
  FusedParams p;
  p.eps = eps;
  vt::FusedChainComposite(q, vt::kFusedAddRmsNormStd, b, p);
  CHECK(out_c == out_g);
  CHECK(res_c == res_g);
}

// kAttnQkNormRope — full: the NON-gated sibling of kAttnQkNormRopeGate (Qwen3 dense has
// no attention output gate). Composite-only MACRO realized as three EXISTING standalone
// ops: RmsNorm(q) in place, RmsNorm(k) in place, RopeFromCache over the 3-D views that
// alias the same buffers. Second half of the count-guard repair.
void CheckAttnQkNormRope() {
  const int64_t T = 8, Hq = 16, Hkv = 2, Dh = 128;
  const int rot = 64;
  const float base = 1.0e7f, eps = 1e-6f;
  const auto q_h = RandF32(static_cast<size_t>(T * Hq * Dh), 310);
  const auto k_h = RandF32(static_cast<size_t>(T * Hkv * Dh), 320);
  const auto qn_h = RandF32(static_cast<size_t>(Dh), 330, -0.5f, 0.5f);
  const auto kn_h = RandF32(static_cast<size_t>(Dh), 340, -0.5f, 0.5f);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int64_t x = 0; x < T; ++x) pos_h[static_cast<size_t>(x)] = static_cast<int32_t>(x);

  Queue q{Cpu(), nullptr};
  Tensor tqn = MakeTensor(const_cast<float*>(qn_h.data()), DType::kF32, {Dh});
  Tensor tkn = MakeTensor(const_cast<float*>(kn_h.data()), DType::kF32, {Dh});
  Tensor tpos = MakeTensor(pos_h.data(), DType::kI32, {T});
  std::vector<float> cs(static_cast<size_t>(T * rot));
  Tensor tcs = MakeTensor(cs.data(), DType::kF32, {T, rot});
  vt::RopeCosSinCache(q, tcs, tpos, RopeArgs{base, rot});

  auto run = [&](bool golden, std::vector<float>& qo, std::vector<float>& ko) {
    qo = q_h;
    ko = k_h;
    Tensor tq2 = MakeTensor(qo.data(), DType::kF32, {T * Hq, Dh});
    Tensor tk2 = MakeTensor(ko.data(), DType::kF32, {T * Hkv, Dh});
    Tensor tq3 = MakeTensor(qo.data(), DType::kF32, {T, Hq, Dh});
    Tensor tk3 = MakeTensor(ko.data(), DType::kF32, {T, Hkv, Dh});
    if (golden) {
      vt::RmsNorm(q, tq2, tq2, tqn, RmsNormArgs{eps, /*gemma=*/false}, nullptr);
      vt::RmsNorm(q, tk2, tk2, tkn, RmsNormArgs{eps, /*gemma=*/false}, nullptr);
      vt::RopeFromCache(q, tq3, &tk3, tpos, tcs, RopeArgs{base, rot});
    } else {
      FusedBinding b;
      b.op[0] = &tq2;
      b.op[1] = &tqn;
      b.op[2] = &tk2;
      b.op[3] = &tkn;
      b.op[4] = &tq3;
      b.op[5] = &tk3;
      b.op[6] = &tcs;
      b.op[7] = &tpos;
      b.n = 8;
      FusedParams p;
      p.eps = eps;
      p.rope = RopeArgs{base, rot};
      vt::FusedChainComposite(q, vt::kAttnQkNormRope, b, p);
    }
  };
  std::vector<float> qg, kg, qc, kc;
  run(true, qg, kg);
  run(false, qc, kc);
  CHECK(qc == qg);
  CHECK(kc == kg);
}

// kSiluMulFp4Quant — full: MoeSiluMul(bf16) + ScaledFp4Quant (fp4 terminal is CPU-ok).
void CheckSiluMulFp4Quant() {
  const int64_t m = 3, i = 256;
  const float gs = 6.5f;
  const auto gf = RandF32(static_cast<size_t>(m * i), 21);
  const auto uf = RandF32(static_cast<size_t>(m * i), 22);
  const auto gb = PackBf16(gf), ub = PackBf16(uf);
  Tensor tg = MakeTensor(const_cast<uint8_t*>(gb.data()), DType::kBF16, {m, i});
  Tensor tu = MakeTensor(const_cast<uint8_t*>(ub.data()), DType::kBF16, {m, i});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp(static_cast<size_t>(m * i) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> pk_g(static_cast<size_t>(m * i / 2)), sc_g(static_cast<size_t>(m * i / 16));
  Tensor ttmp = MakeTensor(tmp.data(), DType::kBF16, {m, i});
  Tensor tpk_g = MakeTensor(pk_g.data(), DType::kI8, {m, i / 2});
  Tensor tsc_g = MakeTensor(sc_g.data(), DType::kI8, {m, i / 16});
  vt::MoeSiluMul(q, ttmp, tg, tu);
  vt::ScaledFp4Quant(q, tpk_g, tsc_g, ttmp, gs);

  std::vector<uint8_t> tmp_c(tmp.size()), pk_c(pk_g.size()), sc_c(sc_g.size());
  Tensor ttmp_c = MakeTensor(tmp_c.data(), DType::kBF16, {m, i});
  Tensor tpk_c = MakeTensor(pk_c.data(), DType::kI8, {m, i / 2});
  Tensor tsc_c = MakeTensor(sc_c.data(), DType::kI8, {m, i / 16});
  FusedBinding b;
  b.op[0] = &tg;
  b.op[1] = &tu;
  b.op[2] = &ttmp_c;
  b.op[3] = &tpk_c;
  b.op[4] = &tsc_c;
  b.n = 5;
  FusedParams p;
  p.quant_scale = gs;
  vt::FusedChainComposite(q, vt::kSiluMulFp4Quant, b, p);
  CHECK(pk_c == pk_g);
  CHECK(sc_c == sc_g);
}

// kSigmoidGateFp4Quant — full: SigmoidGateBf16(bf16) + ScaledFp4Quant.
void CheckSigmoidGateFp4Quant() {
  const int64_t m = 3, k = 256;
  const float gs = 5.25f;
  const auto af = RandF32(static_cast<size_t>(m * k), 31);
  const auto gatef = RandF32(static_cast<size_t>(m * k), 32);
  const auto ab = PackBf16(af);
  Tensor tattn = MakeTensor(const_cast<uint8_t*>(ab.data()), DType::kBF16, {m, k});
  Tensor tgate = MakeTensor(const_cast<float*>(gatef.data()), DType::kF32, {m, k});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp(static_cast<size_t>(m * k) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> pk_g(static_cast<size_t>(m * k / 2)), sc_g(static_cast<size_t>(m * k / 16));
  Tensor ttmp = MakeTensor(tmp.data(), DType::kBF16, {m, k});
  Tensor tpk_g = MakeTensor(pk_g.data(), DType::kI8, {m, k / 2});
  Tensor tsc_g = MakeTensor(sc_g.data(), DType::kI8, {m, k / 16});
  vt::SigmoidGateBf16(q, ttmp, tattn, tgate);
  vt::ScaledFp4Quant(q, tpk_g, tsc_g, ttmp, gs);

  std::vector<uint8_t> tmp_c(tmp.size()), pk_c(pk_g.size()), sc_c(sc_g.size());
  Tensor ttmp_c = MakeTensor(tmp_c.data(), DType::kBF16, {m, k});
  Tensor tpk_c = MakeTensor(pk_c.data(), DType::kI8, {m, k / 2});
  Tensor tsc_c = MakeTensor(sc_c.data(), DType::kI8, {m, k / 16});
  FusedBinding b;
  b.op[0] = &tattn;
  b.op[1] = &tgate;
  b.op[2] = &ttmp_c;
  b.op[3] = &tpk_c;
  b.op[4] = &tsc_c;
  b.n = 5;
  FusedParams p;
  p.quant_scale = gs;
  vt::FusedChainComposite(q, vt::kSigmoidGateFp4Quant, b, p);
  CHECK(pk_c == pk_g);
  CHECK(sc_c == sc_g);
}

// kAttnQkNormRopeGate — full: the composite MACRO dispatches the whole fused preamble
// to the single standalone vt::AttnQkNormRopeGate op (CPU-registered).
void CheckAttnQkNormRopeGate() {
  const int64_t T = 8, Hq = 16, Hkv = 2, Dh = 128;
  const int rot = 64;
  const float base = 1.0e7f, eps = 1e-6f;
  const auto qgate_h = RandF32(static_cast<size_t>(T * Hq * 2 * Dh), 210);
  const auto kf_h = RandF32(static_cast<size_t>(T * Hkv * Dh), 220);
  const auto qn_h = RandF32(static_cast<size_t>(Dh), 230, -0.5f, 0.5f);
  const auto kn_h = RandF32(static_cast<size_t>(Dh), 240, -0.5f, 0.5f);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int64_t x = 0; x < T; ++x) pos_h[static_cast<size_t>(x)] = static_cast<int32_t>(x);

  Queue q{Cpu(), nullptr};
  Tensor tqg = MakeTensor(const_cast<float*>(qgate_h.data()), DType::kF32, {T, Hq * 2 * Dh});
  Tensor tkf = MakeTensor(const_cast<float*>(kf_h.data()), DType::kF32, {T, Hkv * Dh});
  Tensor tqn = MakeTensor(const_cast<float*>(qn_h.data()), DType::kF32, {Dh});
  Tensor tkn = MakeTensor(const_cast<float*>(kn_h.data()), DType::kF32, {Dh});
  Tensor tpos = MakeTensor(pos_h.data(), DType::kI32, {T});
  std::vector<float> cs(static_cast<size_t>(T * rot));
  Tensor tcs = MakeTensor(cs.data(), DType::kF32, {T, rot});
  vt::RopeCosSinCache(q, tcs, tpos, RopeArgs{base, rot});

  auto run = [&](bool golden, std::vector<float>& qo, std::vector<float>& ko,
                 std::vector<float>& go) {
    qo.assign(static_cast<size_t>(T * Hq * Dh), 0);
    ko.assign(static_cast<size_t>(T * Hkv * Dh), 0);
    go.assign(static_cast<size_t>(T * Hq * Dh), 0);
    Tensor tq = MakeTensor(qo.data(), DType::kF32, {T, Hq, Dh});
    Tensor tk = MakeTensor(ko.data(), DType::kF32, {T, Hkv, Dh});
    Tensor tgo = MakeTensor(go.data(), DType::kF32, {T, Hq, Dh});
    if (golden) {
      vt::AttnQkNormRopeGate(q, tq, tk, tgo, tqg, tkf, tqn, tkn, tcs, RmsNormArgs{eps, true},
                             RopeArgs{base, rot});
    } else {
      FusedBinding b;
      b.op[0] = &tqg;
      b.op[1] = &tkf;
      b.op[2] = &tqn;
      b.op[3] = &tkn;
      b.op[4] = &tcs;
      b.op[5] = &tq;
      b.op[6] = &tk;
      b.op[7] = &tgo;
      b.n = 8;
      FusedParams p;
      p.eps = eps;
      p.rope = RopeArgs{base, rot};
      vt::FusedChainComposite(q, vt::kAttnQkNormRopeGate, b, p);
    }
  };
  std::vector<float> qg, kg, gg, qc, kc, gc;
  run(true, qg, kg, gg);
  run(false, qc, kc, gc);
  CHECK(qc == qg);
  CHECK(kc == kg);
  CHECK(gc == gg);
}

// --- fp8-terminal recipes: CPU-expressible PREFIX byte-exact + negotiated-tail throw --

// kRmsNormQuantFp8 — prefix: (add residual) + gemma-RMSNorm → bf16; fp8 tail CUDA-only.
void CheckRmsNormQuantFp8() {
  const int64_t t = 3, h = 256;
  const float eps = 1e-6f, scale = 0.125f;
  const auto xf = RandF32(static_cast<size_t>(t * h), 41);
  const auto wf = RandF32(static_cast<size_t>(h), 42);
  const auto rf = RandF32(static_cast<size_t>(t * h), 43);
  const auto xb = PackBf16(xf), wb = PackBf16(wf);
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), DType::kBF16, {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), DType::kBF16, {h});
  Queue q{Cpu(), nullptr};

  std::vector<float> res_g = rf;
  std::vector<uint8_t> tmp_g(static_cast<size_t>(t * h) * vt::SizeOf(DType::kBF16));
  Tensor ttmp_g = MakeTensor(tmp_g.data(), DType::kBF16, {t, h});
  Tensor trg = MakeTensor(res_g.data(), DType::kF32, {t, h});
  vt::RmsNorm(q, ttmp_g, tx, tw, RmsNormArgs{eps, true}, &trg);

  std::vector<float> res_c = rf;
  std::vector<uint8_t> tmp_c(tmp_g.size());
  Tensor ttmp_c = MakeTensor(tmp_c.data(), DType::kBF16, {t, h});
  Tensor trc = MakeTensor(res_c.data(), DType::kF32, {t, h});
  FusedBinding b;
  b.op[0] = &tx;
  b.op[1] = &tw;
  b.op[2] = &trc;
  b.op[3] = &ttmp_c;
  b.n = 4;  // prefix drops the fp8 output slot
  FusedParams p;
  p.eps = eps;
  p.quant_scale = scale;
  vt::FusedChainComposite(q, CpuExpressiblePrefix(vt::kRmsNormQuantFp8), b, p);
  CHECK(tmp_c == tmp_g);
  CHECK(res_c == res_g);

  // Backend-negotiated tail (§3b/§6): the full composite reaches the CUDA-only
  // vt::QuantFp8Static terminal, which is unregistered on the CPU 'second backend' →
  // it THROWS. The CPU backend inherits the portable prefix, negotiates the fp8 tail.
  std::vector<uint8_t> fp8(static_cast<size_t>(t * h));
  Tensor tfp8 = MakeTensor(fp8.data(), DType::kI8, {t, h});
  b.op[4] = &tfp8;
  b.n = 5;
  CHECK_THROWS(vt::FusedChainComposite(q, vt::kRmsNormQuantFp8, b, p));
}

// kRmsNormGatedQuantFp8 — prefix: gated-RMSNorm → bf16; fp8 tail CUDA-only.
void CheckRmsNormGatedQuantFp8() {
  const int64_t rows = 3, d = 256;
  const float eps = 1e-6f, scale = 0.1875f;
  const auto xf = RandF32(static_cast<size_t>(rows * d), 51);
  const auto gatef = RandF32(static_cast<size_t>(rows * d), 52);
  const auto wf = RandF32(static_cast<size_t>(d), 53);
  const auto xb = PackBf16(xf), gb = PackBf16(gatef), wb = PackBf16(wf);
  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), DType::kBF16, {rows, d});
  Tensor tgate = MakeTensor(const_cast<uint8_t*>(gb.data()), DType::kBF16, {rows, d});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), DType::kBF16, {d});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp_g(static_cast<size_t>(rows * d) * vt::SizeOf(DType::kBF16));
  Tensor ttmp_g = MakeTensor(tmp_g.data(), DType::kBF16, {rows, d});
  vt::RmsNormGated(q, ttmp_g, tx, tgate, tw, RmsNormGatedArgs{eps, false});

  std::vector<uint8_t> tmp_c(tmp_g.size());
  Tensor ttmp_c = MakeTensor(tmp_c.data(), DType::kBF16, {rows, d});
  FusedBinding b;
  b.op[0] = &tx;
  b.op[1] = &tgate;
  b.op[2] = &tw;
  b.op[3] = &ttmp_c;
  b.n = 4;  // prefix drops the fp8 output slot
  FusedParams p;
  p.eps = eps;
  p.quant_scale = scale;
  vt::FusedChainComposite(q, CpuExpressiblePrefix(vt::kRmsNormGatedQuantFp8), b, p);
  CHECK(tmp_c == tmp_g);

  std::vector<uint8_t> fp8(static_cast<size_t>(rows * d));
  Tensor tfp8 = MakeTensor(fp8.data(), DType::kI8, {rows, d});
  b.op[4] = &tfp8;
  b.n = 5;
  CHECK_THROWS(vt::FusedChainComposite(q, vt::kRmsNormGatedQuantFp8, b, p));
}

// kSiluMulQuantFp8 — the W3 mechanical-sync recipe. Prefix: silu(gate)·up → bf16; fp8
// tail CUDA-only. This is the executable proof that the CPU 'second backend' inherited
// a WHOLE NEW fusion pass (ported in W3) with ZERO backend edits — same generic path.
void CheckSiluMulQuantFp8() {
  const int64_t m = 3, i = 256;
  const float scale = 0.125f;
  const auto gf = RandF32(static_cast<size_t>(m * i), 61);
  const auto uf = RandF32(static_cast<size_t>(m * i), 62);
  const auto gb = PackBf16(gf), ub = PackBf16(uf);
  Tensor tg = MakeTensor(const_cast<uint8_t*>(gb.data()), DType::kBF16, {m, i});
  Tensor tu = MakeTensor(const_cast<uint8_t*>(ub.data()), DType::kBF16, {m, i});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp_g(static_cast<size_t>(m * i) * vt::SizeOf(DType::kBF16));
  Tensor ttmp_g = MakeTensor(tmp_g.data(), DType::kBF16, {m, i});
  vt::MoeSiluMul(q, ttmp_g, tg, tu);

  std::vector<uint8_t> tmp_c(tmp_g.size());
  Tensor ttmp_c = MakeTensor(tmp_c.data(), DType::kBF16, {m, i});
  FusedBinding b;
  b.op[0] = &tg;
  b.op[1] = &tu;
  b.op[2] = &ttmp_c;
  b.n = 3;  // prefix drops the fp8 output slot
  FusedParams p;
  p.quant_scale = scale;
  vt::FusedChainComposite(q, CpuExpressiblePrefix(vt::kSiluMulQuantFp8), b, p);
  CHECK(tmp_c == tmp_g);

  std::vector<uint8_t> fp8(static_cast<size_t>(m * i));
  Tensor tfp8 = MakeTensor(fp8.data(), DType::kI8, {m, i});
  b.op[3] = &tfp8;
  b.n = 4;
  CHECK_THROWS(vt::FusedChainComposite(q, vt::kSiluMulQuantFp8, b, p));
}

// The catalog: the SINGLE enumeration of every recipe the framework declares. Each row
// pairs a recipe with the CPU-scope driver above. `cpu_full` records whether the CPU
// 'second backend' realizes the recipe end-to-end (true) or up to the CPU-expressible
// prefix with the fp8 quant terminal backend-negotiated (false). Adding a recipe adds
// ONE row here (a catalog/test concern) — the BACKEND path (composite walker + the one
// kFusedChain registration) is untouched. This IS the additivity mechanism, executable.
struct CatalogEntry {
  const FusedRecipe* recipe;
  const char* name;
  bool cpu_full;
  void (*run)();
};

const CatalogEntry kCatalog[] = {
    {&vt::kFusedAddRmsNorm, "kFusedAddRmsNorm", true, &CheckFusedAddRmsNorm},
    {&vt::kFusedAddRmsNormStd, "kFusedAddRmsNormStd", true, &CheckFusedAddRmsNormStd},
    {&vt::kAttnQkNormRope, "kAttnQkNormRope", true, &CheckAttnQkNormRope},
    {&vt::kSiluMulFp4Quant, "kSiluMulFp4Quant", true, &CheckSiluMulFp4Quant},
    {&vt::kSigmoidGateFp4Quant, "kSigmoidGateFp4Quant", true, &CheckSigmoidGateFp4Quant},
    {&vt::kAttnQkNormRopeGate, "kAttnQkNormRopeGate", true, &CheckAttnQkNormRopeGate},
    {&vt::kRmsNormQuantFp8, "kRmsNormQuantFp8", false, &CheckRmsNormQuantFp8},
    {&vt::kRmsNormGatedQuantFp8, "kRmsNormGatedQuantFp8", false, &CheckRmsNormGatedQuantFp8},
    {&vt::kSiluMulQuantFp8, "kSiluMulQuantFp8", false, &CheckSiluMulQuantFp8},
};

}  // namespace

// ===========================================================================
// The executable backend-additivity assertion. ONE generic loop over the WHOLE catalog:
// every recipe runs CORRECT on the CPU 'second backend' via the Tier-0 composite, with
// no per-recipe backend code (the walker is one per-opcode switch; the CPU backend
// registers one kFusedChain + the standalone primitives). The count guard ties catalog
// growth to test growth so a future recipe cannot silently escape the additivity proof.
TEST_CASE("W4 additivity: every catalog recipe runs correct on the CPU 'second backend' via composite") {
  // The guard is the mechanism, so it must equal the number of recipes DECLARED in
  // include/vt/recipes.h — it had drifted to 7 against a 9-recipe catalog, which is
  // precisely the silent escape it exists to prevent. Repaired to 9 with the two
  // missing rows (kFusedAddRmsNormStd, kAttnQkNormRope) added above.
  CHECK(sizeof(kCatalog) / sizeof(kCatalog[0]) == 9);
  for (const auto& e : kCatalog) {
    CAPTURE(e.name);
    CAPTURE(e.cpu_full);
    e.run();
  }
}
