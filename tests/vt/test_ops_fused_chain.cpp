// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
//
// TDR Phase 0 parity gate. Asserts the kFusedAddRmsNorm recipe realized three
// ways is BIT-IDENTICAL:
//   Tier-0 composite (VT_FUSED_TIER=0) == Tier-1 interpreter (VT_FUSED_TIER=1)
//   == the existing vt::RmsNorm(residual) golden (the fused_add_rms_norm path at
//   src/vllm/model_executor/models/qwen3_5.cpp:2322).
// The recipe is the single source of truth; every tier must agree to the bit.
// CPU is the primary bit-identical gate; the CUDA section runs on dgx (skips
// cleanly when no GPU backend is registered) and checks the same equalities
// among the GPU realizations.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
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

void SetTier(int tier) { setenv("VT_FUSED_TIER", tier == 1 ? "1" : "0", 1); }

// Runs the golden RmsNorm(residual) and both FusedChain tiers on CPU with the
// SAME inputs, and asserts all three outputs (and residual streams) are byte-
// identical. dtypes cover the shapes the real call-site hits (bf16 x/weight with
// an f32 or bf16 residual) plus the all-f32 reference.
void RunCpuCase(int64_t t, int64_t h, DType xdt, DType outdt, DType resdt, uint32_t seed) {
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, xdt);
  const auto wb = Pack(wf, xdt);  // weight follows x (the model materializes it so)
  const auto rb = Pack(rf, resdt);
  const float eps = 1e-6f;
  const size_t obytes = static_cast<size_t>(t * h) * vt::SizeOf(outdt);
  const size_t rbytes = rb.size();

  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), xdt, Cpu(), {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), xdt, Cpu(), {h});
  Queue q{Cpu(), nullptr};

  // Golden: the existing fused_add_rms_norm path (gemma weight = 1+w).
  std::vector<uint8_t> out_g(obytes), res_g = rb;
  Tensor tog = MakeTensor(out_g.data(), outdt, Cpu(), {t, h});
  Tensor trg = MakeTensor(res_g.data(), resdt, Cpu(), {t, h});
  vt::RmsNorm(q, tog, tx, tw, RmsNormArgs{eps, /*gemma=*/true}, &trg);

  // Tier-0 composite.
  SetTier(0);
  std::vector<uint8_t> out_0(obytes), res_0 = rb;
  Tensor to0 = MakeTensor(out_0.data(), outdt, Cpu(), {t, h});
  Tensor tr0 = MakeTensor(res_0.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to0, tx, tw, &tr0, vt::kFusedAddRmsNorm, eps);

  // Tier-1 interpreter.
  SetTier(1);
  std::vector<uint8_t> out_1(obytes), res_1 = rb;
  Tensor to1 = MakeTensor(out_1.data(), outdt, Cpu(), {t, h});
  Tensor tr1 = MakeTensor(res_1.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to1, tx, tw, &tr1, vt::kFusedAddRmsNorm, eps);
  SetTier(0);

  // Bit-identical: composite == golden, interpreter == golden (hence all three).
  CHECK(std::memcmp(out_0.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(out_1.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(res_0.data(), res_g.data(), rbytes) == 0);
  CHECK(std::memcmp(res_1.data(), res_g.data(), rbytes) == 0);
}

}  // namespace

TEST_CASE("fused_chain kFusedAddRmsNorm: Tier-0 == Tier-1 == RmsNorm(residual), bit-identical") {
  // 2048 = the 35B/27B hidden_size the W0 adoption site (RunLayerPaged
  // post_attention_layernorm) actually hits in production — the shape the
  // vt::FusedChain(kFusedAddRmsNorm) seam runs at token-exact.
  const int64_t sizes[] = {1, 7, 8, 127, 128, 129, 512, 2048};
  uint32_t seed = 20;
  for (int64_t h : sizes) {
    CAPTURE(h);
    // Primary gate: all-f32 (full-precision bit-identical).
    RunCpuCase(3, h, DType::kF32, DType::kF32, DType::kF32, seed);
    seed += 7;
    // The real call-site: bf16 x/weight, f32 residual, bf16 output.
    RunCpuCase(3, h, DType::kBF16, DType::kBF16, DType::kF32, seed);
    seed += 7;
    // bf16 residual (vLLM model-dtype residual): rounding is identical per tier.
    RunCpuCase(3, h, DType::kBF16, DType::kBF16, DType::kBF16, seed);
    seed += 7;
  }
}

namespace {

// --- kFusedAddRmsNormStd (ADDITIVE-MODEL W3): the gemma=false sibling. Same
// three-way bit-identity as above but the golden RmsNorm is STANDARD (weight
// `w`, not `1+w`) — the Qwen3 dense input/post/final norm.
void RunCpuCaseStd(int64_t t, int64_t h, DType xdt, DType outdt, DType resdt, uint32_t seed) {
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, xdt);
  const auto wb = Pack(wf, xdt);
  const auto rb = Pack(rf, resdt);
  const float eps = 1e-6f;
  const size_t obytes = static_cast<size_t>(t * h) * vt::SizeOf(outdt);
  const size_t rbytes = rb.size();

  Tensor tx = MakeTensor(const_cast<uint8_t*>(xb.data()), xdt, Cpu(), {t, h});
  Tensor tw = MakeTensor(const_cast<uint8_t*>(wb.data()), xdt, Cpu(), {h});
  Queue q{Cpu(), nullptr};

  // Golden: standard (non-gemma) add+RMSNorm.
  std::vector<uint8_t> out_g(obytes), res_g = rb;
  Tensor tog = MakeTensor(out_g.data(), outdt, Cpu(), {t, h});
  Tensor trg = MakeTensor(res_g.data(), resdt, Cpu(), {t, h});
  vt::RmsNorm(q, tog, tx, tw, RmsNormArgs{eps, /*gemma=*/false}, &trg);

  SetTier(0);
  std::vector<uint8_t> out_0(obytes), res_0 = rb;
  Tensor to0 = MakeTensor(out_0.data(), outdt, Cpu(), {t, h});
  Tensor tr0 = MakeTensor(res_0.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to0, tx, tw, &tr0, vt::kFusedAddRmsNormStd, eps);

  SetTier(1);
  std::vector<uint8_t> out_1(obytes), res_1 = rb;
  Tensor to1 = MakeTensor(out_1.data(), outdt, Cpu(), {t, h});
  Tensor tr1 = MakeTensor(res_1.data(), resdt, Cpu(), {t, h});
  vt::FusedChain(q, to1, tx, tw, &tr1, vt::kFusedAddRmsNormStd, eps);
  SetTier(0);

  CHECK(std::memcmp(out_0.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(out_1.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(res_0.data(), res_g.data(), rbytes) == 0);
  CHECK(std::memcmp(res_1.data(), res_g.data(), rbytes) == 0);
}

}  // namespace

TEST_CASE("fused_chain kFusedAddRmsNormStd: Tier-0 == Tier-1 == RmsNorm(std,residual), bit-identical") {
  // 1024 = the Qwen3-0.6B hidden_size the W3 adoption site (input/post/final
  // norm) actually hits — the vt::FusedChain(kFusedAddRmsNormStd) seam runs at
  // token-exact.
  const int64_t sizes[] = {1, 7, 8, 127, 128, 129, 512, 1024};
  uint32_t seed = 120;
  for (int64_t h : sizes) {
    CAPTURE(h);
    RunCpuCaseStd(3, h, DType::kF32, DType::kF32, DType::kF32, seed);
    seed += 7;
    RunCpuCaseStd(3, h, DType::kBF16, DType::kBF16, DType::kF32, seed);
    seed += 7;
    RunCpuCaseStd(3, h, DType::kBF16, DType::kBF16, DType::kBF16, seed);
    seed += 7;
  }
}

namespace {

// kAttnQkNormRope composite == standalone RmsNorm(q)+RmsNorm(k)+RopeFromCache,
// byte-exact. The non-gated per-head preamble the Qwen3 dense attention uses
// (qwen3.py::Qwen3Attention.forward). The two q/k norms run IN PLACE over the
// 2-D [T*H,Dh] view; kRope rotates the same buffers viewed 3-D [T,H,Dh].
void RunCpuAttnQkNormRope(int64_t t, int64_t hq, int64_t hk, int64_t dh, uint32_t seed) {
  const int rot = static_cast<int>(dh);
  const float eps = 1e-6f;
  const auto qf = RandF32(static_cast<size_t>(t * hq * dh), seed);
  const auto kf = RandF32(static_cast<size_t>(t * hk * dh), seed + 1);
  const auto qnf = RandF32(static_cast<size_t>(dh), seed + 2);
  const auto knf = RandF32(static_cast<size_t>(dh), seed + 3);
  std::vector<int32_t> pos(static_cast<size_t>(t));
  for (int64_t i = 0; i < t; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  const vt::RopeArgs rope{1000000.0f, rot, /*is_neox_style=*/true};

  Queue q{Cpu(), nullptr};
  Tensor tpos = MakeTensor(pos.data(), DType::kI32, Cpu(), {t});
  Tensor tqn = MakeTensor(const_cast<float*>(qnf.data()), DType::kF32, Cpu(), {dh});
  Tensor tkn = MakeTensor(const_cast<float*>(knf.data()), DType::kF32, Cpu(), {dh});
  std::vector<float> cs(static_cast<size_t>(t) * rot);
  Tensor tcs = MakeTensor(cs.data(), DType::kF32, Cpu(), {t, rot});
  vt::RopeCosSinCache(q, tcs, tpos, rope);

  // Golden: standalone in-place norms + RopeFromCache.
  std::vector<float> qg = qf, kg = kf;
  Tensor qg2 = MakeTensor(qg.data(), DType::kF32, Cpu(), {t * hq, dh});
  Tensor kg2 = MakeTensor(kg.data(), DType::kF32, Cpu(), {t * hk, dh});
  vt::RmsNorm(q, qg2, qg2, tqn, RmsNormArgs{eps, false});
  vt::RmsNorm(q, kg2, kg2, tkn, RmsNormArgs{eps, false});
  Tensor qg3 = MakeTensor(qg.data(), DType::kF32, Cpu(), {t, hq, dh});
  Tensor kg3 = MakeTensor(kg.data(), DType::kF32, Cpu(), {t, hk, dh});
  vt::RopeFromCache(q, qg3, &kg3, tpos, tcs, rope);

  // Composite via the declared recipe.
  std::vector<float> qc = qf, kc = kf;
  Tensor qc2 = MakeTensor(qc.data(), DType::kF32, Cpu(), {t * hq, dh});
  Tensor kc2 = MakeTensor(kc.data(), DType::kF32, Cpu(), {t * hk, dh});
  Tensor qc3 = MakeTensor(qc.data(), DType::kF32, Cpu(), {t, hq, dh});
  Tensor kc3 = MakeTensor(kc.data(), DType::kF32, Cpu(), {t, hk, dh});
  vt::FusedBinding b;
  b.op[0] = &qc2;
  b.op[1] = &tqn;
  b.op[2] = &kc2;
  b.op[3] = &tkn;
  b.op[4] = &qc3;
  b.op[5] = &kc3;
  b.op[6] = &tcs;
  b.op[7] = &tpos;
  b.n = 8;
  vt::FusedParams p;
  p.eps = eps;
  p.rope = rope;
  SetTier(0);
  vt::FusedChain(q, vt::kAttnQkNormRope, b, p);

  const size_t qb = qc.size() * sizeof(float), kb = kc.size() * sizeof(float);
  CHECK(std::memcmp(qc.data(), qg.data(), qb) == 0);
  CHECK(std::memcmp(kc.data(), kg.data(), kb) == 0);
}

}  // namespace

TEST_CASE("fused_chain kAttnQkNormRope composite == RmsNorm(q)+RmsNorm(k)+RopeFromCache (CPU, byte-exact)") {
  // Qwen3-0.6B attention shape: 16 q / 8 kv heads, head_dim 128.
  RunCpuAttnQkNormRope(/*t=*/5, /*hq=*/16, /*hk=*/8, /*dh=*/128, 300);
  RunCpuAttnQkNormRope(/*t=*/1, /*hq=*/16, /*hk=*/8, /*dh=*/128, 311);
  RunCpuAttnQkNormRope(/*t=*/3, /*hq=*/4, /*hk=*/2, /*dh=*/64, 322);
}

namespace {

}  // namespace

TEST_CASE("fused_chain validates operands at the chokepoint") {
  Queue q{Cpu(), nullptr};
  std::vector<float> x(4, 1.0f), w(2, 1.0f), out(4, 0.0f);
  Tensor tx = MakeTensor(x.data(), DType::kF32, Cpu(), {2, 2});
  Tensor to = MakeTensor(out.data(), DType::kF32, Cpu(), {2, 2});
  // weight size mismatch ([1] != H=2) → wrapper throws before dispatch.
  Tensor tbad = MakeTensor(w.data(), DType::kF32, Cpu(), {1});
  CHECK_THROWS_AS(vt::FusedChain(q, to, tx, tbad, nullptr, vt::kFusedAddRmsNorm, 1e-6f),
                  std::runtime_error);
}

// ---------------------------------------------------------------------------
// CUDA section: the same three realizations must agree bit-for-bit on the GPU
// (all use the identical f32 tree reduction). Skips when no CUDA backend exists.

namespace {

using vt::GetBackend;

bool HasCuda() {
  try {
    GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

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
    t_ = MakeTensor(p_, dt, Gpu(), shape);
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

void RunCudaCase(int64_t t, int64_t h, DType xdt, DType outdt, DType resdt, uint32_t seed) {
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, xdt);
  const auto wb = Pack(wf, xdt);
  const auto rb = Pack(rf, resdt);
  const float eps = 1e-6f;
  const size_t obytes = static_cast<size_t>(t * h) * vt::SizeOf(outdt);

  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);

  DeviceTensor dx(gpu, g.q, xdt, {t, h}, xb.data());
  DeviceTensor dw(gpu, g.q, xdt, {h}, wb.data());

  auto run = [&](int tier, bool golden, std::vector<uint8_t>& out_bytes,
                 std::vector<uint8_t>& res_bytes) {
    SetTier(tier);
    DeviceTensor dout(gpu, g.q, outdt, {t, h});
    DeviceTensor dres(gpu, g.q, resdt, {t, h}, rb.data());
    if (golden) {
      vt::RmsNorm(g.q, dout.tensor(), dx.tensor(), dw.tensor(), RmsNormArgs{eps, true},
                  &dres.tensor());
    } else {
      vt::FusedChain(g.q, dout.tensor(), dx.tensor(), dw.tensor(), &dres.tensor(),
                     vt::kFusedAddRmsNorm, eps);
    }
    out_bytes.assign(obytes, 0);
    res_bytes.assign(rb.size(), 0);
    dout.Download(g.q, out_bytes.data());
    dres.Download(g.q, res_bytes.data());
  };

  std::vector<uint8_t> out_g, res_g, out_0, res_0, out_1, res_1;
  run(0, /*golden=*/true, out_g, res_g);
  run(0, /*golden=*/false, out_0, res_0);
  run(1, /*golden=*/false, out_1, res_1);
  SetTier(0);

  CHECK(std::memcmp(out_0.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(out_1.data(), out_g.data(), obytes) == 0);
  CHECK(std::memcmp(res_0.data(), res_g.data(), rb.size()) == 0);
  CHECK(std::memcmp(res_1.data(), res_g.data(), rb.size()) == 0);
}

}  // namespace

TEST_CASE("CUDA fused_chain: Tier-0 == Tier-1 == RmsNorm(residual), bit-identical") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 4000;
  for (int64_t h : {1, 127, 128, 129, 512, 2048}) {
    CAPTURE(h);
    RunCudaCase(3, h, DType::kF32, DType::kF32, DType::kF32, seed);
    seed += 7;
    RunCudaCase(3, h, DType::kBF16, DType::kBF16, DType::kF32, seed);
    seed += 7;
    RunCudaCase(3, h, DType::kBF16, DType::kBF16, DType::kBF16, seed);
    seed += 7;
  }
}

// ===========================================================================
// W1: the generalized POD's new recipes. Each new recipe's Tier-0 COMPOSITE must
// equal its standalone-op-sequence golden BYTE-EXACT (the composite literally
// dispatches to those standalone ops, so equality is by construction — this pins
// the operand binding / dispatch plumbing of the generalized POD). Composite runs
// regardless of VT_FUSED_TIER (these recipes are not Tier-1-able), so we assert
// tier-invariance too. CPU where the constituent ops have a CPU kernel; CUDA for
// all (the fp8 quant terminal is CUDA-only, §3b/§6).

namespace {

using vt::Fp4ScaleLayout;
using vt::FusedBinding;
using vt::FusedParams;
using vt::RmsNormGatedArgs;
using vt::RopeArgs;

// --- CPU byte-exact drivers (fp4 chains + attn preamble; fp8 is CUDA-only) -----

// kSiluMulFp4Quant: MoeSiluMul(gate,up->bf16) then ScaledFp4Quant.
void RunSiluMulFp4Cpu(int64_t m, int64_t i, uint32_t seed, int tier) {
  const float gs = 6.5f;
  const auto gf = RandF32(static_cast<size_t>(m * i), seed);
  const auto uf = RandF32(static_cast<size_t>(m * i), seed + 1);
  const auto gb = Pack(gf, DType::kBF16);
  const auto ub = Pack(uf, DType::kBF16);
  Tensor tg = MakeTensor(const_cast<uint8_t*>(gb.data()), DType::kBF16, Cpu(), {m, i});
  Tensor tu = MakeTensor(const_cast<uint8_t*>(ub.data()), DType::kBF16, Cpu(), {m, i});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp_g(static_cast<size_t>(m * i) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> pk_g(static_cast<size_t>(m * i / 2)), sc_g(static_cast<size_t>(m * i / 16));
  Tensor ttmp_g = MakeTensor(tmp_g.data(), DType::kBF16, Cpu(), {m, i});
  Tensor tpk_g = MakeTensor(pk_g.data(), DType::kI8, Cpu(), {m, i / 2});
  Tensor tsc_g = MakeTensor(sc_g.data(), DType::kI8, Cpu(), {m, i / 16});
  vt::MoeSiluMul(q, ttmp_g, tg, tu);
  vt::ScaledFp4Quant(q, tpk_g, tsc_g, ttmp_g, gs);

  SetTier(tier);
  std::vector<uint8_t> tmp_f(tmp_g.size());
  std::vector<uint8_t> pk_f(pk_g.size()), sc_f(sc_g.size());
  Tensor ttmp_f = MakeTensor(tmp_f.data(), DType::kBF16, Cpu(), {m, i});
  Tensor tpk_f = MakeTensor(pk_f.data(), DType::kI8, Cpu(), {m, i / 2});
  Tensor tsc_f = MakeTensor(sc_f.data(), DType::kI8, Cpu(), {m, i / 16});
  FusedBinding bind;
  bind.op[0] = &tg;
  bind.op[1] = &tu;
  bind.op[2] = &ttmp_f;
  bind.op[3] = &tpk_f;
  bind.op[4] = &tsc_f;
  bind.n = 5;
  FusedParams p;
  p.quant_scale = gs;
  // W2: FusedChain now dispatches to the FAST realization (the bespoke
  // SiluMulFp4Quant kernel, recipe.fast_op) — must equal the standalone-op golden.
  vt::FusedChain(q, vt::kSiluMulFp4Quant, bind, p);
  SetTier(0);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
  // ...and the Tier-0 composite oracle (bind carries tmp_bf16 = op[2]) must ALSO
  // equal the golden, byte-for-byte (fast == composite == unfused sequence, §5).
  vt::FusedChainComposite(q, vt::kSiluMulFp4Quant, bind, p);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
}

// kSigmoidGateFp4Quant: SigmoidGateBf16(attn,gate->bf16) then ScaledFp4Quant.
void RunSigmoidGateFp4Cpu(int64_t m, int64_t k, uint32_t seed, int tier) {
  const float gs = 5.25f;
  const auto af = RandF32(static_cast<size_t>(m * k), seed);
  const auto gatef = RandF32(static_cast<size_t>(m * k), seed + 1);
  const auto ab = Pack(af, DType::kBF16);  // attn bf16
  Tensor tattn = MakeTensor(const_cast<uint8_t*>(ab.data()), DType::kBF16, Cpu(), {m, k});
  Tensor tgate = MakeTensor(const_cast<float*>(gatef.data()), DType::kF32, Cpu(), {m, k});
  Queue q{Cpu(), nullptr};

  std::vector<uint8_t> tmp_g(static_cast<size_t>(m * k) * vt::SizeOf(DType::kBF16));
  std::vector<uint8_t> pk_g(static_cast<size_t>(m * k / 2)), sc_g(static_cast<size_t>(m * k / 16));
  Tensor ttmp_g = MakeTensor(tmp_g.data(), DType::kBF16, Cpu(), {m, k});
  Tensor tpk_g = MakeTensor(pk_g.data(), DType::kI8, Cpu(), {m, k / 2});
  Tensor tsc_g = MakeTensor(sc_g.data(), DType::kI8, Cpu(), {m, k / 16});
  vt::SigmoidGateBf16(q, ttmp_g, tattn, tgate);
  vt::ScaledFp4Quant(q, tpk_g, tsc_g, ttmp_g, gs);

  SetTier(tier);
  std::vector<uint8_t> tmp_f(tmp_g.size());
  std::vector<uint8_t> pk_f(pk_g.size()), sc_f(sc_g.size());
  Tensor ttmp_f = MakeTensor(tmp_f.data(), DType::kBF16, Cpu(), {m, k});
  Tensor tpk_f = MakeTensor(pk_f.data(), DType::kI8, Cpu(), {m, k / 2});
  Tensor tsc_f = MakeTensor(sc_f.data(), DType::kI8, Cpu(), {m, k / 16});
  FusedBinding bind;
  bind.op[0] = &tattn;
  bind.op[1] = &tgate;
  bind.op[2] = &ttmp_f;
  bind.op[3] = &tpk_f;
  bind.op[4] = &tsc_f;
  bind.n = 5;
  FusedParams p;
  p.quant_scale = gs;
  // W2: FusedChain dispatches the FAST realization (bespoke SigmoidGateFp4Quant).
  vt::FusedChain(q, vt::kSigmoidGateFp4Quant, bind, p);
  SetTier(0);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
  // Tier-0 composite oracle must also match byte-for-byte.
  vt::FusedChainComposite(q, vt::kSigmoidGateFp4Quant, bind, p);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
}

}  // namespace

TEST_CASE("fused_chain kSiluMulFp4Quant composite == MoeSiluMul + ScaledFp4Quant (CPU, byte-exact)") {
  uint32_t seed = 900;
  for (int tier : {0, 1}) {
    for (int64_t i : {16, 32, 64, 256}) {
      CAPTURE(i);
      CAPTURE(tier);
      RunSiluMulFp4Cpu(3, i, seed, tier);
      seed += 7;
    }
  }
}

TEST_CASE("fused_chain kSigmoidGateFp4Quant composite == SigmoidGateBf16 + ScaledFp4Quant (CPU, byte-exact)") {
  uint32_t seed = 1300;
  for (int tier : {0, 1}) {
    for (int64_t k : {16, 32, 64, 256}) {
      CAPTURE(k);
      CAPTURE(tier);
      RunSigmoidGateFp4Cpu(3, k, seed, tier);
      seed += 7;
    }
  }
}

// --- CUDA byte-exact drivers for all five new recipes -------------------------

namespace {

// kSiluMulFp4Quant on CUDA.
void RunSiluMulFp4Cuda(int64_t m, int64_t i, uint32_t seed) {
  const float gs = 6.5f;
  const auto gf = RandF32(static_cast<size_t>(m * i), seed);
  const auto uf = RandF32(static_cast<size_t>(m * i), seed + 1);
  const auto gb = Pack(gf, DType::kBF16);
  const auto ub = Pack(uf, DType::kBF16);
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dg(gpu, g.q, DType::kBF16, {m, i}, gb.data());
  DeviceTensor du(gpu, g.q, DType::kBF16, {m, i}, ub.data());

  // mode 0 = golden standalone sequence, 1 = FusedChain FAST realization, 2 = Tier-0
  // composite. All three must produce byte-identical packed/scale streams.
  auto run = [&](int mode, std::vector<uint8_t>& pk, std::vector<uint8_t>& sc) {
    DeviceTensor dtmp(gpu, g.q, DType::kBF16, {m, i});
    DeviceTensor dpk(gpu, g.q, DType::kI8, {m, i / 2});
    DeviceTensor dsc(gpu, g.q, DType::kI8, {m, i / 16});
    if (mode == 0) {
      vt::MoeSiluMul(g.q, dtmp.tensor(), dg.tensor(), du.tensor());
      vt::ScaledFp4Quant(g.q, dpk.tensor(), dsc.tensor(), dtmp.tensor(), gs);
    } else {
      FusedBinding b;
      b.op[0] = &dg.tensor();
      b.op[1] = &du.tensor();
      b.op[2] = &dtmp.tensor();
      b.op[3] = &dpk.tensor();
      b.op[4] = &dsc.tensor();
      b.n = 5;
      FusedParams p;
      p.quant_scale = gs;
      if (mode == 1) {
        vt::FusedChain(g.q, vt::kSiluMulFp4Quant, b, p);
      } else {
        vt::FusedChainComposite(g.q, vt::kSiluMulFp4Quant, b, p);
      }
    }
    pk.assign(static_cast<size_t>(m * i / 2), 0);
    sc.assign(static_cast<size_t>(m * i / 16), 0);
    dpk.Download(g.q, pk.data());
    dsc.Download(g.q, sc.data());
  };
  std::vector<uint8_t> pk_g, sc_g, pk_f, sc_f, pk_c, sc_c;
  run(0, pk_g, sc_g);
  run(1, pk_f, sc_f);
  run(2, pk_c, sc_c);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
  CHECK(pk_c == pk_g);
  CHECK(sc_c == sc_g);
}

// kSigmoidGateFp4Quant on CUDA.
void RunSigmoidGateFp4Cuda(int64_t m, int64_t k, uint32_t seed) {
  const float gs = 5.25f;
  const auto af = RandF32(static_cast<size_t>(m * k), seed);
  const auto gatef = RandF32(static_cast<size_t>(m * k), seed + 1);
  const auto ab = Pack(af, DType::kBF16);
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dattn(gpu, g.q, DType::kBF16, {m, k}, ab.data());
  DeviceTensor dgate(gpu, g.q, DType::kF32, {m, k}, gatef.data());

  auto run = [&](int mode, std::vector<uint8_t>& pk, std::vector<uint8_t>& sc) {
    DeviceTensor dtmp(gpu, g.q, DType::kBF16, {m, k});
    DeviceTensor dpk(gpu, g.q, DType::kI8, {m, k / 2});
    DeviceTensor dsc(gpu, g.q, DType::kI8, {m, k / 16});
    if (mode == 0) {
      vt::SigmoidGateBf16(g.q, dtmp.tensor(), dattn.tensor(), dgate.tensor());
      vt::ScaledFp4Quant(g.q, dpk.tensor(), dsc.tensor(), dtmp.tensor(), gs);
    } else {
      FusedBinding b;
      b.op[0] = &dattn.tensor();
      b.op[1] = &dgate.tensor();
      b.op[2] = &dtmp.tensor();
      b.op[3] = &dpk.tensor();
      b.op[4] = &dsc.tensor();
      b.n = 5;
      FusedParams p;
      p.quant_scale = gs;
      if (mode == 1) {
        vt::FusedChain(g.q, vt::kSigmoidGateFp4Quant, b, p);
      } else {
        vt::FusedChainComposite(g.q, vt::kSigmoidGateFp4Quant, b, p);
      }
    }
    pk.assign(static_cast<size_t>(m * k / 2), 0);
    sc.assign(static_cast<size_t>(m * k / 16), 0);
    dpk.Download(g.q, pk.data());
    dsc.Download(g.q, sc.data());
  };
  std::vector<uint8_t> pk_g, sc_g, pk_f, sc_f, pk_c, sc_c;
  run(0, pk_g, sc_g);
  run(1, pk_f, sc_f);
  run(2, pk_c, sc_c);
  CHECK(pk_f == pk_g);
  CHECK(sc_f == sc_g);
  CHECK(pk_c == pk_g);
  CHECK(sc_c == sc_g);
}

// kSiluMulQuantFp8 (W3 mechanical-sync proof) on CUDA: MoeSiluMul(gate,up->bf16)
// then QuantFp8Static. Composite-only (no fast_op) — mode 1 (FusedChain) falls
// through to the Tier-0 composite; mode 2 calls it explicitly. Both must equal the
// standalone-op-sequence golden byte-for-byte (the fp8 terminal is CUDA-only).
void RunSiluMulQuantFp8Cuda(int64_t m, int64_t i, uint32_t seed) {
  const float scale = 0.125f;
  const auto gf = RandF32(static_cast<size_t>(m * i), seed);
  const auto uf = RandF32(static_cast<size_t>(m * i), seed + 1);
  const auto gb = Pack(gf, DType::kBF16);
  const auto ub = Pack(uf, DType::kBF16);
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dg(gpu, g.q, DType::kBF16, {m, i}, gb.data());
  DeviceTensor du(gpu, g.q, DType::kBF16, {m, i}, ub.data());

  auto run = [&](int mode, std::vector<uint8_t>& fp8) {
    DeviceTensor dtmp(gpu, g.q, DType::kBF16, {m, i});
    DeviceTensor dfp8(gpu, g.q, DType::kI8, {m, i});
    if (mode == 0) {
      vt::MoeSiluMul(g.q, dtmp.tensor(), dg.tensor(), du.tensor());
      vt::QuantFp8Static(g.q, dfp8.tensor(), dtmp.tensor(), scale);
    } else {
      FusedBinding b;
      b.op[0] = &dg.tensor();
      b.op[1] = &du.tensor();
      b.op[2] = &dtmp.tensor();
      b.op[3] = &dfp8.tensor();
      b.n = 4;
      FusedParams p;
      p.quant_scale = scale;
      if (mode == 1) {
        vt::FusedChain(g.q, vt::kSiluMulQuantFp8, b, p);
      } else {
        vt::FusedChainComposite(g.q, vt::kSiluMulQuantFp8, b, p);
      }
    }
    fp8.assign(static_cast<size_t>(m * i), 0);
    dfp8.Download(g.q, fp8.data());
  };
  std::vector<uint8_t> fp8_g, fp8_f, fp8_c;
  run(0, fp8_g);
  run(1, fp8_f);
  run(2, fp8_c);
  CHECK(fp8_f == fp8_g);
  CHECK(fp8_c == fp8_g);
}

// kRmsNormQuantFp8 on CUDA: RmsNorm(bf16,+residual) then QuantFp8Static.
void RunRmsNormQuantFp8Cuda(int64_t t, int64_t h, DType resdt, uint32_t seed) {
  const float scale = 0.125f, eps = 1e-6f;
  const auto xf = RandF32(static_cast<size_t>(t * h), seed);
  const auto wf = RandF32(static_cast<size_t>(h), seed + 1);
  const auto rf = RandF32(static_cast<size_t>(t * h), seed + 2);
  const auto xb = Pack(xf, DType::kBF16);
  const auto wb = Pack(wf, DType::kBF16);
  const auto rb = Pack(rf, resdt);
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dx(gpu, g.q, DType::kBF16, {t, h}, xb.data());
  DeviceTensor dw(gpu, g.q, DType::kBF16, {h}, wb.data());

  auto run = [&](int mode, std::vector<uint8_t>& fp8, std::vector<uint8_t>& res) {
    DeviceTensor dres(gpu, g.q, resdt, {t, h}, rb.data());
    DeviceTensor dtmp(gpu, g.q, DType::kBF16, {t, h});
    DeviceTensor dfp8(gpu, g.q, DType::kI8, {t, h});
    if (mode == 0) {
      vt::RmsNorm(g.q, dtmp.tensor(), dx.tensor(), dw.tensor(), RmsNormArgs{eps, true},
                  &dres.tensor());
      vt::QuantFp8Static(g.q, dfp8.tensor(), dtmp.tensor(), scale);
    } else {
      FusedBinding b;
      b.op[0] = &dx.tensor();
      b.op[1] = &dw.tensor();
      b.op[2] = &dres.tensor();
      b.op[3] = &dtmp.tensor();
      b.op[4] = &dfp8.tensor();
      b.n = 5;
      FusedParams p;
      p.eps = eps;
      p.quant_scale = scale;
      if (mode == 1) {
        vt::FusedChain(g.q, vt::kRmsNormQuantFp8, b, p);
      } else {
        vt::FusedChainComposite(g.q, vt::kRmsNormQuantFp8, b, p);
      }
    }
    fp8.assign(static_cast<size_t>(t * h), 0);
    res.assign(rb.size(), 0);
    dfp8.Download(g.q, fp8.data());
    dres.Download(g.q, res.data());
  };
  std::vector<uint8_t> fp8_g, res_g, fp8_f, res_f, fp8_c, res_c;
  run(0, fp8_g, res_g);
  run(1, fp8_f, res_f);
  run(2, fp8_c, res_c);
  CHECK(fp8_f == fp8_g);
  CHECK(res_f == res_g);
  CHECK(fp8_c == fp8_g);
  CHECK(res_c == res_g);
}

// kRmsNormGatedQuantFp8 on CUDA: RmsNormGated(bf16) then QuantFp8Static.
void RunRmsNormGatedQuantFp8Cuda(int64_t rows, int64_t d, uint32_t seed) {
  const float scale = 0.1875f, eps = 1e-6f;
  const auto xf = RandF32(static_cast<size_t>(rows * d), seed);
  const auto gatef = RandF32(static_cast<size_t>(rows * d), seed + 1);
  const auto wf = RandF32(static_cast<size_t>(d), seed + 2);
  const auto xb = Pack(xf, DType::kBF16);
  const auto gb = Pack(gatef, DType::kBF16);
  const auto wb = Pack(wf, DType::kBF16);
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dx(gpu, g.q, DType::kBF16, {rows, d}, xb.data());
  DeviceTensor dgate(gpu, g.q, DType::kBF16, {rows, d}, gb.data());
  DeviceTensor dw(gpu, g.q, DType::kBF16, {d}, wb.data());

  auto run = [&](int mode, std::vector<uint8_t>& fp8) {
    DeviceTensor dtmp(gpu, g.q, DType::kBF16, {rows, d});
    DeviceTensor dfp8(gpu, g.q, DType::kI8, {rows, d});
    if (mode == 0) {
      vt::RmsNormGated(g.q, dtmp.tensor(), dx.tensor(), dgate.tensor(), dw.tensor(),
                       RmsNormGatedArgs{eps, false});
      vt::QuantFp8Static(g.q, dfp8.tensor(), dtmp.tensor(), scale);
    } else {
      FusedBinding b;
      b.op[0] = &dx.tensor();
      b.op[1] = &dgate.tensor();
      b.op[2] = &dw.tensor();
      b.op[3] = &dtmp.tensor();
      b.op[4] = &dfp8.tensor();
      b.n = 5;
      FusedParams p;
      p.eps = eps;
      p.quant_scale = scale;
      if (mode == 1) {
        vt::FusedChain(g.q, vt::kRmsNormGatedQuantFp8, b, p);
      } else {
        vt::FusedChainComposite(g.q, vt::kRmsNormGatedQuantFp8, b, p);
      }
    }
    fp8.assign(static_cast<size_t>(rows * d), 0);
    dfp8.Download(g.q, fp8.data());
  };
  std::vector<uint8_t> fp8_g, fp8_f, fp8_c;
  run(0, fp8_g);
  run(1, fp8_f);
  run(2, fp8_c);
  CHECK(fp8_f == fp8_g);
  CHECK(fp8_c == fp8_g);
}

}  // namespace

TEST_CASE("CUDA fused_chain new recipes composite == standalone-op-sequence (byte-exact)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  uint32_t seed = 7000;
  for (int64_t i : {16, 64, 256}) {
    CAPTURE(i);
    RunSiluMulFp4Cuda(4, i, seed);
    seed += 7;
    RunSigmoidGateFp4Cuda(4, i, seed);
    seed += 7;
    // W3: the newly-ported kSiluMulQuantFp8 (silu·mul -> static per-tensor fp8).
    RunSiluMulQuantFp8Cuda(4, i, seed);
    seed += 7;
  }
  for (int64_t h : {128, 256, 2048}) {
    CAPTURE(h);
    RunRmsNormQuantFp8Cuda(3, h, DType::kF32, seed);
    seed += 7;
    RunRmsNormQuantFp8Cuda(3, h, DType::kBF16, seed);
    seed += 7;
    RunRmsNormGatedQuantFp8Cuda(3, h, seed);
    seed += 7;
  }
}

// kAttnQkNormRopeGate composite MACRO == the standalone vt::AttnQkNormRopeGate the
// model hand-calls today (the composite dispatches the whole fused preamble to it).
TEST_CASE("fused_chain kAttnQkNormRopeGate composite == AttnQkNormRopeGate (CPU + CUDA)") {
  const int64_t T = 8, Hq = 16, Hkv = 2, Dh = 128;
  const int rot = 64;
  const float base = 1.0e7f, eps = 1e-6f;
  const auto qgate_h = RandF32(static_cast<size_t>(T * Hq * 2 * Dh), 210);
  const auto kf_h = RandF32(static_cast<size_t>(T * Hkv * Dh), 220);
  const auto qn_h = RandF32(static_cast<size_t>(Dh), 230, -0.5f, 0.5f);
  const auto kn_h = RandF32(static_cast<size_t>(Dh), 240, -0.5f, 0.5f);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int64_t x = 0; x < T; ++x) pos_h[static_cast<size_t>(x)] = static_cast<int32_t>(x);

  // ---- CPU ----
  {
    Queue q{Cpu(), nullptr};
    Tensor tqg = MakeTensor(const_cast<float*>(qgate_h.data()), DType::kF32, Cpu(),
                            {T, Hq * 2 * Dh});
    Tensor tkf = MakeTensor(const_cast<float*>(kf_h.data()), DType::kF32, Cpu(), {T, Hkv * Dh});
    Tensor tqn = MakeTensor(const_cast<float*>(qn_h.data()), DType::kF32, Cpu(), {Dh});
    Tensor tkn = MakeTensor(const_cast<float*>(kn_h.data()), DType::kF32, Cpu(), {Dh});
    Tensor tpos = MakeTensor(pos_h.data(), DType::kI32, Cpu(), {T});
    std::vector<float> cs(static_cast<size_t>(T * rot));
    Tensor tcs = MakeTensor(cs.data(), DType::kF32, Cpu(), {T, rot});
    vt::RopeCosSinCache(q, tcs, tpos, RopeArgs{base, rot});

    auto run = [&](bool golden, std::vector<float>& qo, std::vector<float>& ko,
                   std::vector<float>& go) {
      qo.assign(static_cast<size_t>(T * Hq * Dh), 0);
      ko.assign(static_cast<size_t>(T * Hkv * Dh), 0);
      go.assign(static_cast<size_t>(T * Hq * Dh), 0);
      Tensor tq = MakeTensor(qo.data(), DType::kF32, Cpu(), {T, Hq, Dh});
      Tensor tk = MakeTensor(ko.data(), DType::kF32, Cpu(), {T, Hkv, Dh});
      Tensor tgo = MakeTensor(go.data(), DType::kF32, Cpu(), {T, Hq, Dh});
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
        vt::FusedChain(q, vt::kAttnQkNormRopeGate, b, p);
      }
    };
    std::vector<float> qg, kg, gg, qf, kf2, gf;
    run(true, qg, kg, gg);
    run(false, qf, kf2, gf);
    CHECK(qf == qg);
    CHECK(kf2 == kg);
    CHECK(gf == gg);
  }

  // ---- CUDA ----
  if (!HasCuda()) return;
  Backend& gpu = GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dqg(gpu, g.q, DType::kF32, {T, Hq * 2 * Dh}, qgate_h.data());
  DeviceTensor dkf(gpu, g.q, DType::kF32, {T, Hkv * Dh}, kf_h.data());
  DeviceTensor dqn(gpu, g.q, DType::kF32, {Dh}, qn_h.data());
  DeviceTensor dkn(gpu, g.q, DType::kF32, {Dh}, kn_h.data());
  DeviceTensor dpos(gpu, g.q, DType::kI32, {T}, pos_h.data());
  DeviceTensor dcs(gpu, g.q, DType::kF32, {T, rot});
  vt::RopeCosSinCache(g.q, dcs.tensor(), dpos.tensor(), RopeArgs{base, rot});

  auto run = [&](bool golden, std::vector<float>& qo, std::vector<float>& ko,
                 std::vector<float>& go) {
    DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, Dh});
    DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hkv, Dh});
    DeviceTensor dgo(gpu, g.q, DType::kF32, {T, Hq, Dh});
    if (golden) {
      vt::AttnQkNormRopeGate(g.q, dq.tensor(), dk.tensor(), dgo.tensor(), dqg.tensor(),
                             dkf.tensor(), dqn.tensor(), dkn.tensor(), dcs.tensor(),
                             RmsNormArgs{eps, true}, RopeArgs{base, rot});
    } else {
      FusedBinding b;
      b.op[0] = &dqg.tensor();
      b.op[1] = &dkf.tensor();
      b.op[2] = &dqn.tensor();
      b.op[3] = &dkn.tensor();
      b.op[4] = &dcs.tensor();
      b.op[5] = &dq.tensor();
      b.op[6] = &dk.tensor();
      b.op[7] = &dgo.tensor();
      b.n = 8;
      FusedParams p;
      p.eps = eps;
      p.rope = RopeArgs{base, rot};
      vt::FusedChain(g.q, vt::kAttnQkNormRopeGate, b, p);
    }
    qo.assign(static_cast<size_t>(T * Hq * Dh), 0);
    ko.assign(static_cast<size_t>(T * Hkv * Dh), 0);
    go.assign(static_cast<size_t>(T * Hq * Dh), 0);
    dq.Download(g.q, qo.data());
    dk.Download(g.q, ko.data());
    dgo.Download(g.q, go.data());
  };
  std::vector<float> qg, kg, gg, qf, kf2, gf;
  run(true, qg, kg, gg);
  run(false, qf, kf2, gf);
  CHECK(qf == qg);
  CHECK(kf2 == kg);
  CHECK(gf == gg);
}
