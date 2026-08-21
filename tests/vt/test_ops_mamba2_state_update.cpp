// Mamba2 single-token selective state update (vt::Mamba2StateUpdate) — UNIT GATE.
// .agents/specs/mamba2-ssd.md W1, issue #496.
//
// Ported from tests/kernels/mamba/test_mamba_ssm.py @ pin 555967922
// (vLLM 0.26.0.dev0) — `test_selective_state_update` (:344-389),
// `test_selective_state_update_with_batch_indices` (:706-807) and
// `test_selective_state_update_with_heads_with_batch_indices` (:810-889) —
// preserving their shapes, parameters and tolerances. The op under test mirrors
// `selective_state_update` (ops/mamba_ssm.py:497+) as mamba_mixer2.py:1087 calls
// it on the decode path.
//
// HARNESS ADAPTATION (documented, per porting.md). Upstream's three tests run
// with A/dt/D either per (head, headdim) or `tie_hdim`-broadcast per head. Mamba2
// only ever uses the SCALAR-PER-HEAD form (`A_d`/`dt_d`/`D_d` come from
// `A_log`/`dt`/`D` of shape (nheads,), mamba_mixer2.py:1055-1104), which is also
// the only form upstream's OWN CPU kernel implements — the three tests above are
// all `skipif(current_platform.is_cpu())` with the reason "CPU kernel for
// selective_state_update only supports Mamba 2 (scalar A/dt), not Mamba 1"
// (:348-355). This op takes that form; the Mamba-1 per-channel `A` is explicitly
// out of scope (mamba2-ssd.md §0). Upstream's reference
// (tests/kernels/mamba/utils.py:9-77 `selective_state_update_ref`) is
// re-expressed here in `double` against that shape.
//
// ─── WHY THESE ARE THE CORRECTNESS EVIDENCE ───────────────────────────────────
// (1) A `double` reference derived from utils.py:59-75, not from the op.
// (2) SCATTERED CACHE SLOTS and the NULL row: the local ABI (index < 0) that
//     GdnDecode already models — the null row's cache slot must be BYTE-identical
//     afterwards, and every non-selected slot in the cache untouched. Upstream's
//     equivalent assertion is `torch.equal(state_before[unused], state[unused])`
//     (test_mamba_ssm.py:800).
// (3) DECODE == PREFILL: N single-token updates must reproduce what
//     vt::Mamba2ChunkScan produces for the same N tokens. Two entirely separate
//     code paths (sequential vs 5-stage chunked), so agreement is evidence, not
//     a shared-helper tautology
//     ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]).
// (4) An F32 OUTPUT ARM alongside every bf16 arm
//     ([[bf16-store-absorbs-reduction-order-defects]]), and the SSM cache dtype
//     exercised independently of the activation dtype (mamba_utils.py:73-81).
//
// TOLERANCES: explicit `torch.testing.assert_close` arithmetic, never
// `doctest::Approx` — its `scale` defaults to 1.0 and floors every comparison at
// ~1.19e-5 absolute ([[doctest-approx-scale-term-floor]]).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Mamba2Args;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
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

// NOTE — doctest 2.5.2 `INFO` prints a `const char*` VARIABLE as `1` (it binds
// the bool overload; only a string LITERAL prints as text), so every label here
// is a `std::string`. A `const char*` label silently turns a failure message
// into "1: worst element ...".
void ExpectClose(const std::string& what, const std::vector<float>& got,
                 const std::vector<double>& want, double atol, double rtol) {
  REQUIRE(got.size() == want.size());
  REQUIRE(!got.empty());
  double worst_slack = -std::numeric_limits<double>::infinity();
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = want[i];
    const double slack = std::abs(g - w) - (atol + rtol * std::abs(w));
    if (!std::isfinite(g) || slack > worst_slack) {
      worst_slack = slack;
      worst_i = i;
      if (!std::isfinite(g)) break;
    }
  }
  INFO(what << ": worst element [" << worst_i << "] got=" << got[worst_i]
            << " want=" << want[worst_i] << " |diff|="
            << std::abs(static_cast<double>(got[worst_i]) - want[worst_i])
            << " budget=" << (atol + rtol * std::abs(want[worst_i])));
  CHECK(std::isfinite(static_cast<double>(got[worst_i])));
  CHECK(worst_slack <= 0.0);
}

void ExpectCloseF(const std::string& what, const std::vector<float>& got,
                  const std::vector<float>& want, double atol, double rtol) {
  const std::vector<double> w(want.begin(), want.end());
  ExpectClose(what, got, w, atol, rtol);
}

std::vector<uint8_t> Pack(const std::vector<float>& src, DType dt) {
  std::vector<uint8_t> raw(src.size() * vt::SizeOf(dt));
  for (size_t i = 0; i < src.size(); ++i) {
    if (dt == DType::kF32) {
      std::memcpy(raw.data() + i * 4, &src[i], 4);
    } else {
      const uint16_t v = vt::F32ToBF16(src[i]);
      std::memcpy(raw.data() + i * 2, &v, 2);
    }
  }
  return raw;
}

std::vector<float> Unpack(const std::vector<uint8_t>& raw, size_t n, DType dt) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    if (dt == DType::kF32) {
      std::memcpy(&out[i], raw.data() + i * 4, 4);
    } else {
      uint16_t v;
      std::memcpy(&v, raw.data() + i * 2, 2);
      out[i] = vt::BF16ToF32(v);
    }
  }
  return out;
}

// ─── the `double` reference ──────────────────────────────────────────────────
// `selective_state_update_ref` (tests/kernels/mamba/utils.py:9-77) at the Mamba2
// scalar-per-head shape, restated in double:
//   dt = softplus(dt[b,h] + dt_bias[h])         (:58-60)
//   s  = s*exp(A[h]*dt) + (dt*B[b,g,:]) * x     (:61-70)
//   y  = sum_n s[n]*C[b,g,n] (+ D[h]*x) (* silu(z))   (:71-74)
struct RefStep {
  std::vector<double> out;    // [Nb,H,P]
  std::vector<double> state;  // [Nb,H,P,N] — the updated rows, in batch order
};

RefStep SelectiveStateUpdateRef(const std::vector<double>& state_in, const std::vector<float>& x,
                                const std::vector<float>& dt, const std::vector<float>& A,
                                const std::vector<float>& B, const std::vector<float>& C,
                                const std::vector<float>* D, const std::vector<float>* z,
                                const std::vector<float>* dt_bias, bool dt_softplus, int64_t Nb,
                                int64_t H, int64_t P, int64_t G, int64_t N) {
  RefStep r;
  r.out.assign(static_cast<size_t>(Nb * H * P), 0.0);
  r.state = state_in;
  const int64_t heads_per_group = H / G;
  for (int64_t b = 0; b < Nb; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t g = h / heads_per_group;
      double d = dt[static_cast<size_t>(b * H + h)];
      if (dt_bias != nullptr) d += (*dt_bias)[static_cast<size_t>(h)];
      if (dt_softplus) d = d <= 20.0 ? std::log1p(std::exp(d)) : d;
      const double dA = std::exp(static_cast<double>(A[static_cast<size_t>(h)]) * d);
      for (int64_t p = 0; p < P; ++p) {
        const double xv = x[static_cast<size_t>((b * H + h) * P + p)];
        double y = 0.0;
        for (int64_t n = 0; n < N; ++n) {
          double& s = r.state[static_cast<size_t>(((b * H + h) * P + p) * N + n)];
          s = s * dA + d * B[static_cast<size_t>((b * G + g) * N + n)] * xv;
          y += s * C[static_cast<size_t>((b * G + g) * N + n)];
        }
        if (D != nullptr) y += (*D)[static_cast<size_t>(h)] * xv;
        if (z != nullptr) {
          const double zv = (*z)[static_cast<size_t>((b * H + h) * P + p)];
          y *= zv / (1.0 + std::exp(-zv));
        }
        r.out[static_cast<size_t>((b * H + h) * P + p)] = y;
      }
    }
  }
  return r;
}

struct StepInputs {
  std::vector<float> x, dt, B, C, z;
  std::vector<float> A, D, dt_bias;
};

StepInputs GenerateStep(int64_t Nb, int64_t H, int64_t P, int64_t G, int64_t N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  StepInputs s;
  // A = -rand(nheads) - 1.0 (test_mamba_ssm.py:852-854, the tie_hdim arm)
  s.A.resize(static_cast<size_t>(H));
  for (auto& v : s.A) v = -ud(rng) - 1.0f;
  s.dt_bias.resize(static_cast<size_t>(H));
  for (auto& v : s.dt_bias) v = ud(rng) - 4.0f;  // rand(nheads) - 4.0 (:851)
  s.D.resize(static_cast<size_t>(H));
  for (auto& v : s.D) v = nd(rng);
  s.x.resize(static_cast<size_t>(Nb * H * P));
  for (auto& v : s.x) v = nd(rng);
  s.dt.resize(static_cast<size_t>(Nb * H));
  for (auto& v : s.dt) v = nd(rng);
  s.B.resize(static_cast<size_t>(Nb * G * N));
  for (auto& v : s.B) v = nd(rng);
  s.C.resize(static_cast<size_t>(Nb * G * N));
  for (auto& v : s.C) v = nd(rng);
  s.z.resize(static_cast<size_t>(Nb * H * P));
  for (auto& v : s.z) v = nd(rng);
  return s;
}

// Upstream generates every activation tensor directly in `itype`
// (test_mamba_ssm.py:846-864), so the reference and the kernel see the SAME
// reduced-precision values and the comparison measures the KERNEL, not the
// input quantization. Rounding the inputs first reproduces that.
void RoundActivationsTo(StepInputs& s, DType dt) {
  const auto round = [dt](std::vector<float>& v) {
    if (dt == DType::kBF16)
      for (auto& e : v) e = vt::BF16ToF32(vt::F32ToBF16(e));
  };
  round(s.x);
  round(s.dt);
  round(s.B);
  round(s.C);
  round(s.z);
}

struct StepCfg {
  DType act_dtype = DType::kF32;
  DType state_dtype = DType::kF32;
  bool dt_softplus = true;  // mamba_mixer2.py:1097 always passes True
  bool use_D = true;
  bool use_z = false;
  bool use_dt_bias = true;
  int64_t tp_world_size = 1;
};

// Runs the op in place on a raw state cache. `state_raw` is [S,H,P,N] in
// cfg.state_dtype; `slots` is null for the compact one-row-per-token form.
std::vector<float> RunStateUpdate(std::vector<uint8_t>& state_raw, int64_t S,
                                  const StepInputs& in, const std::vector<int32_t>* slots,
                                  int64_t Nb, int64_t H, int64_t P, int64_t G, int64_t N,
                                  const StepCfg& cfg) {
  Queue q = CpuQ();
  std::vector<uint8_t> xb = Pack(in.x, cfg.act_dtype);
  std::vector<uint8_t> dtb = Pack(in.dt, cfg.act_dtype);
  std::vector<uint8_t> Bb = Pack(in.B, cfg.act_dtype);
  std::vector<uint8_t> Cb = Pack(in.C, cfg.act_dtype);
  std::vector<uint8_t> zb = Pack(in.z, cfg.act_dtype);
  std::vector<float> Ac = in.A, Dc = in.D, dbc = in.dt_bias;
  std::vector<uint8_t> outb(static_cast<size_t>(Nb * H * P) * vt::SizeOf(cfg.act_dtype), 0);

  Tensor st = MakeT(state_raw.data(), cfg.state_dtype, {S, H, P, N});
  Tensor xt = MakeT(xb.data(), cfg.act_dtype, {Nb, H, P});
  Tensor dtt = MakeT(dtb.data(), cfg.act_dtype, {Nb, H});
  Tensor At = MakeT(Ac.data(), DType::kF32, {H});
  Tensor Bt = MakeT(Bb.data(), cfg.act_dtype, {Nb, G, N});
  Tensor Ct = MakeT(Cb.data(), cfg.act_dtype, {Nb, G, N});
  Tensor Dt = MakeT(Dc.data(), DType::kF32, {H});
  Tensor zt = MakeT(zb.data(), cfg.act_dtype, {Nb, H, P});
  Tensor dbt = MakeT(dbc.data(), DType::kF32, {H});
  Tensor outt = MakeT(outb.data(), cfg.act_dtype, {Nb, H, P});

  std::vector<int32_t> idx;
  Tensor idxt;
  if (slots != nullptr) {
    idx = *slots;
    idxt = MakeT(idx.data(), DType::kI32, {Nb});
  }

  Mamba2Args args;
  args.dt_softplus = cfg.dt_softplus;
  args.tp_world_size = cfg.tp_world_size;
  vt::Mamba2StateUpdate(q, outt, st, xt, dtt, At, Bt, Ct, cfg.use_D ? &Dt : nullptr,
                        cfg.use_z ? &zt : nullptr, cfg.use_dt_bias ? &dbt : nullptr,
                        slots != nullptr ? &idxt : nullptr, args);
  return Unpack(outb, static_cast<size_t>(Nb * H * P), cfg.act_dtype);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// (1) The plain decode step, compact state. Shapes from
// test_selective_state_update_with_heads_with_batch_indices (:810-889):
// headdim = 64, nheads = dim/64 for dim in {2048, 4096} scaled down to keep the
// host reference cheap, dstate in {16, 64}, ngroups in {1, 4}, has_z in
// {False, True}. Upstream f32 tolerance is rtol 3e-4 / atol 1e-3; bf16 is
// rtol 5e-3 / atol 3e-2 (:828-831).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update matches the reference recurrence") {
  const int64_t Nb = 3, H = 16, P = 64;
  for (int64_t N : {16, 64}) {
    for (int64_t G : {1, 4}) {
      for (bool has_z : {false, true}) {
        const StepInputs in = GenerateStep(Nb, H, P, G, N, 0x51A7Eu + static_cast<uint32_t>(N));
        std::mt19937 rng(9u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> state0(static_cast<size_t>(Nb * H * P * N));
        for (auto& v : state0) v = nd(rng);
        const std::vector<double> state0d(state0.begin(), state0.end());

        StepCfg cfg;
        cfg.use_z = has_z;
        const RefStep ref =
            SelectiveStateUpdateRef(state0d, in.x, in.dt, in.A, in.B, in.C, &in.D,
                                    has_z ? &in.z : nullptr, &in.dt_bias, true, Nb, H, P, G, N);

        INFO("N=" << N << " G=" << G << " has_z=" << has_z);
        // f32 arm.
        {
          std::vector<uint8_t> raw = Pack(state0, DType::kF32);
          const std::vector<float> out =
              RunStateUpdate(raw, Nb, in, nullptr, Nb, H, P, G, N, cfg);
          ExpectClose("out f32", out, ref.out, 1e-3, 3e-4);
          ExpectClose("state f32", Unpack(raw, state0.size(), DType::kF32), ref.state, 1e-3,
                      3e-4);
        }
        // bf16 activation arm, f32 cache — Nemotron-H's shipped combination
        // (mamba_ssm_cache_dtype=float32 with a bf16 model dtype). Upstream's
        // bf16 tolerance is rtol 1e-1 / atol 1e-1 (test_mamba_ssm.py:829-831).
        {
          StepCfg bcfg = cfg;
          bcfg.act_dtype = DType::kBF16;
          StepInputs bin = in;
          RoundActivationsTo(bin, DType::kBF16);
          const RefStep bref = SelectiveStateUpdateRef(
              state0d, bin.x, bin.dt, bin.A, bin.B, bin.C, &bin.D,
              has_z ? &bin.z : nullptr, &bin.dt_bias, true, Nb, H, P, G, N);
          std::vector<uint8_t> raw = Pack(state0, DType::kF32);
          const std::vector<float> out =
              RunStateUpdate(raw, Nb, bin, nullptr, Nb, H, P, G, N, bcfg);
          ExpectClose("out bf16", out, bref.out, 1e-1, 1e-1);
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) SCATTERED CACHE SLOTS and the NULL row.
// Ported from test_selective_state_update_with_batch_indices (:706-807):
// `total_entries = 10 * batch_size`, slot indices drawn from a permutation, and
// a padded tail. Upstream's padding sentinel is `NULL_BLOCK_ID = 0`
// (v1/attention/backends/utils.py:46); the LOCAL ABI is index < 0, exactly as
// GdnDecode/CausalConv1dSpecUpdate already model it.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update honours scattered slots and the NULL row") {
  const int64_t H = 8, P = 32, N = 16, G = 2;
  const int64_t real = 3, padding = 5, Nb = real + padding;
  const int64_t S = 30;

  const StepInputs in = GenerateStep(Nb, H, P, G, N, 0xFACEu);
  std::mt19937 rng(11u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> cache(static_cast<size_t>(S * H * P * N));
  for (auto& v : cache) v = nd(rng);

  // Three distinct, deliberately non-contiguous and out-of-order slots.
  const std::vector<int32_t> chosen{17, 2, 25};
  std::vector<int32_t> slots = chosen;
  for (int64_t i = 0; i < padding; ++i) slots.push_back(-1);  // the NULL rows

  // Reference: only the three real rows advance, from their own slots.
  std::vector<double> state0d(static_cast<size_t>(real * H * P * N));
  for (int64_t b = 0; b < real; ++b) {
    const size_t src = static_cast<size_t>(chosen[static_cast<size_t>(b)]) *
                       static_cast<size_t>(H * P * N);
    for (int64_t i = 0; i < H * P * N; ++i)
      state0d[static_cast<size_t>(b * H * P * N + i)] = cache[src + static_cast<size_t>(i)];
  }
  StepInputs real_in = in;
  real_in.x.resize(static_cast<size_t>(real * H * P));
  real_in.dt.resize(static_cast<size_t>(real * H));
  real_in.B.resize(static_cast<size_t>(real * G * N));
  real_in.C.resize(static_cast<size_t>(real * G * N));
  real_in.z.resize(static_cast<size_t>(real * H * P));
  const RefStep ref =
      SelectiveStateUpdateRef(state0d, real_in.x, real_in.dt, real_in.A, real_in.B, real_in.C,
                              &real_in.D, &real_in.z, &real_in.dt_bias, true, real, H, P, G, N);

  StepCfg cfg;
  cfg.use_z = true;
  std::vector<uint8_t> raw = Pack(cache, DType::kF32);
  const std::vector<uint8_t> before = raw;
  const std::vector<float> out = RunStateUpdate(raw, S, in, &slots, Nb, H, P, G, N, cfg);
  const std::vector<float> after = Unpack(raw, cache.size(), DType::kF32);

  // (a) the three selected slots hold the advanced state.
  std::vector<float> got_state(static_cast<size_t>(real * H * P * N));
  for (int64_t b = 0; b < real; ++b) {
    const size_t src = static_cast<size_t>(chosen[static_cast<size_t>(b)]) *
                       static_cast<size_t>(H * P * N);
    for (int64_t i = 0; i < H * P * N; ++i)
      got_state[static_cast<size_t>(b * H * P * N + i)] = after[src + static_cast<size_t>(i)];
  }
  ExpectClose("scattered state", got_state, ref.state, 1e-3, 3e-4);

  // (b) the three real output rows match.
  std::vector<float> got_out(out.begin(), out.begin() + static_cast<size_t>(real * H * P));
  ExpectClose("scattered out", got_out, ref.out, 1e-3, 3e-4);

  // (c) EVERY slot that was not selected is BYTE-identical — upstream's
  //     `torch.equal(state_before[unused_states_bool], state[unused_states_bool])`
  //     (:800). A kernel that wrote through row index instead of slot index
  //     passes (a) and (b) only by accident; it cannot pass this.
  size_t untouched_slots = 0;
  for (int64_t s = 0; s < S; ++s) {
    if (std::find(chosen.begin(), chosen.end(), static_cast<int32_t>(s)) != chosen.end())
      continue;
    ++untouched_slots;
    const size_t off = static_cast<size_t>(s) * static_cast<size_t>(H * P * N) * 4;
    CHECK(std::memcmp(before.data() + off, raw.data() + off,
                      static_cast<size_t>(H * P * N) * 4) == 0);
  }
  CHECK(untouched_slots == static_cast<size_t>(S - real));

  // (d) the NULL rows write a ZEROED output row and touch no cache slot (the
  //     latter is already covered by (c), since -1 is in no slot list).
  for (int64_t b = real; b < Nb; ++b)
    for (int64_t i = 0; i < H * P; ++i)
      CHECK(out[static_cast<size_t>(b * H * P + i)] == 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) DECODE == PREFILL. Running T single-token updates must reproduce what
// vt::Mamba2ChunkScan produces for the same T tokens of one sequence. The two
// go through entirely different code (sequential vs the 5-stage chunked
// pipeline), which is what makes the agreement evidence.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update reproduces the chunked prefill token by token") {
  const int64_t T = 40, H = 4, P = 8, G = 2, N = 16, chunk = 8;
  std::mt19937 rng(0x1234u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);

  std::vector<float> A(static_cast<size_t>(H));
  for (auto& v : A) v = -ud(rng) - 1.0f;
  std::vector<float> D(static_cast<size_t>(H));
  for (auto& v : D) v = nd(rng);
  std::vector<float> dt_bias(static_cast<size_t>(H));
  for (auto& v : dt_bias) v = ud(rng) - 4.0f;
  std::vector<float> x(static_cast<size_t>(T * H * P));
  for (auto& v : x) v = nd(rng);
  std::vector<float> dt(static_cast<size_t>(T * H));
  for (auto& v : dt) v = nd(rng);
  std::vector<float> B(static_cast<size_t>(T * G * N));
  for (auto& v : B) v = nd(rng);
  std::vector<float> C(static_cast<size_t>(T * G * N));
  for (auto& v : C) v = nd(rng);

  // ── prefill arm ──
  std::vector<float> y_prefill(static_cast<size_t>(T * H * P), 0.0f);
  std::vector<float> fs(static_cast<size_t>(H * P * N), 0.0f);
  {
    Queue q = CpuQ();
    // One sequence, chunked at `chunk`: logical chunks are [0,8) [8,16) ...
    std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
    std::vector<int32_t> ccs{0};
    std::vector<int32_t> sidx;
    for (int64_t pos = 0; pos < T; pos += chunk) {
      ccs.push_back(static_cast<int32_t>(std::min(pos + chunk, T)));
      sidx.push_back(0);
    }
    std::vector<int32_t> lci{static_cast<int32_t>(sidx.size()) - 1};
    Tensor xt = MakeT(x.data(), DType::kF32, {T, H, P});
    Tensor dtt = MakeT(dt.data(), DType::kF32, {T, H});
    Tensor At = MakeT(A.data(), DType::kF32, {H});
    Tensor Bt = MakeT(B.data(), DType::kF32, {T, G, N});
    Tensor Ct = MakeT(C.data(), DType::kF32, {T, G, N});
    Tensor Dt = MakeT(D.data(), DType::kF32, {H});
    Tensor dbt = MakeT(dt_bias.data(), DType::kF32, {H});
    Tensor outt = MakeT(y_prefill.data(), DType::kF32, {T, H, P});
    Tensor fst = MakeT(fs.data(), DType::kF32, {1, H, P, N});
    Tensor cust = MakeT(cu.data(), DType::kI32, {2});
    Tensor ccst = MakeT(ccs.data(), DType::kI32, {static_cast<int64_t>(ccs.size())});
    Tensor lcit = MakeT(lci.data(), DType::kI32, {1});
    Tensor sit = MakeT(sidx.data(), DType::kI32, {static_cast<int64_t>(sidx.size())});
    Mamba2Args args;
    args.chunk_size = chunk;
    args.dt_softplus = true;
    vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, &Dt, nullptr, &dbt, nullptr, cust,
                        ccst, lcit, sit, args);
  }

  // ── decode arm: T single-token steps on one cache slot ──
  std::vector<uint8_t> raw(static_cast<size_t>(H * P * N) * 4, 0);
  std::vector<float> y_decode(static_cast<size_t>(T * H * P), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    StepInputs step;
    step.A = A;
    step.D = D;
    step.dt_bias = dt_bias;
    step.x.assign(x.begin() + static_cast<size_t>(t * H * P),
                  x.begin() + static_cast<size_t>((t + 1) * H * P));
    step.dt.assign(dt.begin() + static_cast<size_t>(t * H),
                   dt.begin() + static_cast<size_t>((t + 1) * H));
    step.B.assign(B.begin() + static_cast<size_t>(t * G * N),
                  B.begin() + static_cast<size_t>((t + 1) * G * N));
    step.C.assign(C.begin() + static_cast<size_t>(t * G * N),
                  C.begin() + static_cast<size_t>((t + 1) * G * N));
    step.z.assign(static_cast<size_t>(H * P), 0.0f);
    StepCfg cfg;
    const std::vector<float> o = RunStateUpdate(raw, 1, step, nullptr, 1, H, P, G, N, cfg);
    std::copy(o.begin(), o.end(), y_decode.begin() + static_cast<size_t>(t * H * P));
  }

  ExpectCloseF("decode vs chunked prefill", y_decode, y_prefill, 5e-3, 5e-3);
  ExpectCloseF("final state", Unpack(raw, static_cast<size_t>(H * P * N), DType::kF32), fs,
               5e-3, 5e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) The SSM cache dtype is its own knob (`mamba2_state_dtype`,
// mamba_utils.py:73-81) — a bf16 cache with f32 activations must run and must
// round exactly where the knob says.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update keeps the cache dtype independent of the activation dtype") {
  const int64_t Nb = 2, H = 4, P = 8, G = 2, N = 16;
  const StepInputs in = GenerateStep(Nb, H, P, G, N, 0xB16D7u);
  std::mt19937 rng(5u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> state0(static_cast<size_t>(Nb * H * P * N));
  for (auto& v : state0) v = nd(rng);
  // Start from a bf16-representable cache so the only rounding measured is the
  // op's own store.
  for (auto& v : state0) v = vt::BF16ToF32(vt::F32ToBF16(v));
  const std::vector<double> state0d(state0.begin(), state0.end());
  const RefStep ref = SelectiveStateUpdateRef(state0d, in.x, in.dt, in.A, in.B, in.C, &in.D,
                                              nullptr, &in.dt_bias, true, Nb, H, P, G, N);

  StepCfg cfg;
  cfg.state_dtype = DType::kBF16;
  std::vector<uint8_t> raw = Pack(state0, DType::kBF16);
  const std::vector<float> out = RunStateUpdate(raw, Nb, in, nullptr, Nb, H, P, G, N, cfg);
  const std::vector<float> after = Unpack(raw, state0.size(), DType::kBF16);
  ExpectClose("bf16 cache state", after, ref.state, 5e-2, 5e-2);
  ExpectClose("out with bf16 cache", out, ref.out, 5e-2, 5e-2);
  for (float v : after) CHECK(vt::BF16ToF32(vt::F32ToBF16(v)) == v);
}

// ─────────────────────────────────────────────────────────────────────────────
// (4b) THE READOUT USES THE F32 STATE, NOT THE VALUE RE-READ FROM THE CACHE.
// The Triton kernel holds `state` in registers and computes `out = sum(state*C)`
// from those registers, storing the cache-width copy separately
// (mamba_ssm.py:433,451); upstream's own CPU kernel does the same with `s_new`
// (csrc/cpu/mamba_kernels.hpp:225-228).
//
// Test (4) above cannot see this: its only bf16-cache arm compares at
// atol/rtol 5e-2, and one bf16 rounding of the state disappears inside that
// budget ([[token-gates-cannot-see-dequant-fallbacks]] — a tolerance wide enough
// for the dtype is wide enough for the defect). The exact statement instead:
//
//   `out` does not depend on the CACHE dtype at all, so starting from a
//   bf16-representable state, the bf16-cache run and the f32-cache run must
//   produce BIT-IDENTICAL outputs.
//
// A kernel that read `y` back out of the cache would round once per (p, n) and
// break it. The stored states, of course, DO differ — that is checked too, so a
// no-op comparison cannot masquerade as a pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update reads out the f32 state, not the rounded cache") {
  const int64_t Nb = 2, H = 4, P = 8, G = 2, N = 16;
  const StepInputs in = GenerateStep(Nb, H, P, G, N, 0xDEC0DEu);
  std::mt19937 rng(0x99u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> state0(static_cast<size_t>(Nb * H * P * N));
  // bf16-representable, so the two runs START from the same numbers and the only
  // difference between them is where the op's OWN store rounds.
  for (auto& v : state0) v = vt::BF16ToF32(vt::F32ToBF16(nd(rng)));

  for (bool has_z : {false, true}) {
    StepCfg f32cfg;
    f32cfg.use_z = has_z;
    StepCfg bf16cfg = f32cfg;
    bf16cfg.state_dtype = DType::kBF16;

    std::vector<uint8_t> raw_f32 = Pack(state0, DType::kF32);
    const std::vector<float> out_f32 =
        RunStateUpdate(raw_f32, Nb, in, nullptr, Nb, H, P, G, N, f32cfg);
    std::vector<uint8_t> raw_bf16 = Pack(state0, DType::kBF16);
    const std::vector<float> out_bf16 =
        RunStateUpdate(raw_bf16, Nb, in, nullptr, Nb, H, P, G, N, bf16cfg);

    REQUIRE(out_f32.size() == out_bf16.size());
    size_t differing = 0;
    double worst = 0.0;
    for (size_t i = 0; i < out_f32.size(); ++i) {
      if (out_f32[i] != out_bf16[i]) {
        ++differing;
        worst = std::max(worst, std::abs(static_cast<double>(out_f32[i]) - out_bf16[i]));
      }
    }
    INFO("has_z=" << has_z << ": " << differing << " of " << out_f32.size()
                  << " outputs move with the CACHE dtype, max|diff| = " << worst);
    CHECK(differing == 0);

    // The cache rounding really is live — otherwise the check above compares two
    // identical computations and proves nothing.
    const std::vector<float> after_f32 = Unpack(raw_f32, state0.size(), DType::kF32);
    const std::vector<float> after_bf16 = Unpack(raw_bf16, state0.size(), DType::kBF16);
    size_t rounded = 0;
    for (size_t i = 0; i < after_f32.size(); ++i)
      if (after_f32[i] != after_bf16[i]) ++rounded;
    INFO("stored states that the bf16 cache rounded: " << rounded);
    REQUIRE(rounded > 0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) REFUSALS.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 state update refuses the arms it does not implement") {
  const int64_t Nb = 2, H = 4, P = 8, G = 2, N = 8;
  const StepInputs in = GenerateStep(Nb, H, P, G, N, 1u);

  SUBCASE("tp_world_size > 1 names extra_groups_for_head_shards") {
    StepCfg cfg;
    cfg.tp_world_size = 4;
    std::vector<uint8_t> raw(static_cast<size_t>(Nb * H * P * N) * 4, 0);
    bool threw = false;
    std::string msg;
    try {
      RunStateUpdate(raw, Nb, in, nullptr, Nb, H, P, G, N, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    CHECK(threw);
    INFO(msg);
    CHECK(msg.find("extra_groups_for_head_shards") != std::string::npos);
  }

  // ── the precondition the decay rests on ───────────────────────────────────
  // `A = -exp(A_log)` (mamba_mixer2.py:456) is strictly negative for every finite
  // `A_log`, and the decode step's `exp(A[h]*dt)` is a DECAY only while it is.
  // Fed `A > 0` the step returns a silently GROWING recurrence where the caller
  // asked for a decaying one, so the op refuses — `CheckMamba2ANegative`,
  // src/vt/cpu/cpu_ops.cpp, the decode twin of the chunk-scan guard this case
  // mirrors (test_ops_mamba2_ssd.cpp:900).
  //
  // WHY THIS SUBCASE EXISTS (mamba2-ssd.md §8.2, round-2 review correction). The
  // guard was present, correct and reachable, and pinned by NOTHING: deleting the
  // call left this whole suite green, while the same deletion on the chunk-scan
  // twin reds. A guard no test can see is a guard the next refactor removes.
  SUBCASE("A must be negative (A = -exp(A_log))") {
    StepCfg cfg;
    StepInputs bad = in;
    bad.A[static_cast<size_t>(H - 1)] = 1.0f;
    std::vector<uint8_t> raw(static_cast<size_t>(Nb * H * P * N) * 4, 0);
    bool threw = false;
    std::string msg;
    try {
      RunStateUpdate(raw, Nb, bad, nullptr, Nb, H, P, G, N, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    INFO(msg);
    CHECK(threw);
    CHECK(msg.find("A_log") != std::string::npos);

    // Zero is refused too: `-exp(A_log)` is strictly negative for every finite
    // A_log, and a flat (non-decaying) state is not what the layer asked for.
    StepInputs zero = in;
    zero.A[0] = 0.0f;
    std::vector<uint8_t> raw_zero(static_cast<size_t>(Nb * H * P * N) * 4, 0);
    CHECK_THROWS(RunStateUpdate(raw_zero, Nb, zero, nullptr, Nb, H, P, G, N, cfg));

    // ... and a legitimately tiny negative A is ACCEPTED, so the guard is a
    // sign test and not an accidental magnitude floor.
    StepInputs tiny = in;
    for (auto& v : tiny.A) v = -1e-30f;
    std::vector<uint8_t> raw_tiny(static_cast<size_t>(Nb * H * P * N) * 4, 0);
    CHECK_NOTHROW(RunStateUpdate(raw_tiny, Nb, tiny, nullptr, Nb, H, P, G, N, cfg));
  }

  SUBCASE("a compact state must have one row per token") {
    StepCfg cfg;
    std::vector<uint8_t> raw(static_cast<size_t>((Nb + 1) * H * P * N) * 4, 0);
    CHECK_THROWS(RunStateUpdate(raw, Nb + 1, in, nullptr, Nb, H, P, G, N, cfg));
  }

  // Each row owns ONE cache slot: the kernel is a parallel-for over rows
  // (`ForRows`, src/vt/cpu/cpu_ops.cpp), so two rows pointing at the same slot
  // race on the same read-modify-write and the result depends on the thread
  // schedule. Upstream's CPU kernel is sequential and stays deterministic under
  // duplicates, so this is a LOCAL precondition — which means it has to be
  // enforced, not just documented.
  SUBCASE("duplicate state_indices are refused, not raced") {
    StepCfg cfg;
    const int64_t S = 8;
    std::vector<uint8_t> raw(static_cast<size_t>(S * H * P * N) * 4, 0);
    const std::vector<int32_t> dup{3, 3};
    bool threw = false;
    std::string msg;
    try {
      RunStateUpdate(raw, S, in, &dup, Nb, H, P, G, N, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    INFO(msg);
    CHECK(threw);
    CHECK(msg.find("state_indices") != std::string::npos);

    // Distinct slots are fine, and repeated NULL rows (index < 0) are NOT
    // duplicates — they touch no slot at all.
    const std::vector<int32_t> ok{3, 5};
    std::vector<uint8_t> raw_ok(static_cast<size_t>(S * H * P * N) * 4, 0);
    CHECK_NOTHROW(RunStateUpdate(raw_ok, S, in, &ok, Nb, H, P, G, N, cfg));
    const std::vector<int32_t> nulls{-1, -1};
    std::vector<uint8_t> raw_null(static_cast<size_t>(S * H * P * N) * 4, 0);
    CHECK_NOTHROW(RunStateUpdate(raw_null, S, in, &nulls, Nb, H, P, G, N, cfg));
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// (6) THE CUDA ARM — .agents/specs/mamba2-ssd.md W2, issue #496.
// The declared equivalence contract is stated in full at the head of the CUDA
// section of tests/vt/test_ops_mamba2_ssd.cpp and in
// src/vt/cuda/cuda_mamba2_ssd.cuh: f32 accumulation throughout, no tile
// downcasts, identical memory format, and the same per-element accumulation
// order — leaving TWO admitted sources of divergence, the libm difference and
// nvcc's default `--fmad=true` contraction of `y += sn * cv`, both of which
// `DerivedRtol` bounds without a tuned number.
//
// The decode step's evidence is mostly EXACT rather than approximate, and
// deliberately so: which cache slot was written, which was left alone, and what a
// NULL row produced are byte facts, not tolerances.
// ═════════════════════════════════════════════════════════════════════════════
#ifdef VLLM_CPP_CUDA

#include <memory>
#include <stdexcept>

namespace {

using vt::Backend;

Backend* MaybeCuda() {
  try {
    return &vt::GetBackend(DeviceType::kCUDA);
  } catch (const std::exception&) {
    return nullptr;
  }
}

// A GREEN TEST DOES NOT PROVE THE DEVICE RAN IT. GB10 is
// `integrated && pageable_memory_access`, so `Backend::UnifiedMemory()` is TRUE,
// and the reference tier USED TO gate on that: absent a native kernel, `GetOp`
// did not throw — it installed the CPU HOST kernel as a `kReferenceProviderName`
// provider and ran THAT over the device pointers (op_provider.h, "portable
// reference tier"), so every assertion below would pass while nothing ran on the
// GPU. Every CUDA case therefore asserts the SELECTED provider is native. These
// are EAGER dispatches, so the counters are populated
// ([[graph-replay-does-no-host-dispatch-counters-read-zero]]).
//
// SINCE #844 / #1435 that specific false-green is CLOSED, and the guard stays.
// `ReferenceTierEligible` now reads `Backend::DeviceMemoryIsHostAddressable()`,
// which CUDA answers FALSE because `CudaBackend::Alloc` calls `cudaMalloc`, so a
// missing kernel is a named refusal rather than a silent host run. The paragraph
// above is kept as the REASON this guard exists, not as current behaviour. Do not
// delete the guard on the strength of the fix: it also catches a provider that
// DECLINES at run time, and a future backend that answers the narrow predicate
// true would restore the original hazard exactly.
void RequireNativeCudaProvider(vt::OpId op, const std::string& what) {
  const vt::OpProviderStats st = vt::GetOpProviderStats(op, DeviceType::kCUDA);
  INFO(what << ": selected CUDA provider = "
            << (st.last_selected != nullptr ? st.last_selected : "<none>")
            << "; process-wide reference-tier hits = " << vt::GetReferenceTierHits());
  REQUIRE(st.last_selected != nullptr);
  CHECK(std::string(st.last_selected) != std::string(vt::kReferenceProviderName));
}

// `5*(K + 2)*u` — the bound derived at the head of the CUDA section of
// tests/vt/test_ops_mamba2_ssd.cpp: 2.5 ulp of glibc-vs-CUDA libm disagreement
// per decay factor through a product of at most K, plus the standard (K-1)*u
// forward error of a length-K f32 summation, plus K*u for the K product
// roundings the host arm's `-ffp-contract=off` keeps and the device arm's
// nvcc-`fmad` `fma` does not (`y += sn * cv`, cuda_mamba2_ssd.cuh).
constexpr double kUnitRoundoff = 5.9604644775390625e-08;  // 2^-24
double DerivedRtol(int64_t K) { return 5.0 * static_cast<double>(K + 2) * kUnitRoundoff; }

void ExpectDeviceMatchesHost(const std::string& what, const std::vector<float>& dev,
                             const std::vector<float>& host, int64_t K) {
  REQUIRE(dev.size() == host.size());
  REQUIRE(!dev.empty());
  double scale = 0.0;
  for (float v : host) scale = std::max(scale, std::abs(static_cast<double>(v)));
  const double rtol = DerivedRtol(K);
  const double atol = rtol * scale;
  size_t bit_differing = 0, worst_i = 0;
  double worst_ratio = -1.0, worst_diff = 0.0;
  for (size_t i = 0; i < dev.size(); ++i) {
    if (dev[i] != host[i]) ++bit_differing;
    const double d = std::abs(static_cast<double>(dev[i]) - static_cast<double>(host[i]));
    const double budget = atol + rtol * std::abs(static_cast<double>(host[i]));
    const double ratio = budget > 0.0 ? d / budget : (d > 0.0 ? 1e30 : 0.0);
    if (!std::isfinite(static_cast<double>(dev[i])) || ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_i = i;
      worst_diff = d;
      if (!std::isfinite(static_cast<double>(dev[i]))) break;
    }
  }
  // MESSAGE, not INFO: doctest prints an INFO context only when an assertion in
  // its scope FAILS, so the used slack has to be logged unconditionally for the
  // derived bar to be auditable on the green run that matters.
  MESSAGE(what << ": K=" << K << " rtol=" << rtol << " scale=" << scale << "; "
               << bit_differing << " of " << dev.size()
               << " elements differ in any bit; worst element [" << worst_i
               << "] dev=" << dev[worst_i] << " host=" << host[worst_i]
               << " |diff|=" << worst_diff << " used " << (worst_ratio * 100.0)
               << "% of its derived budget");
  CHECK(std::isfinite(static_cast<double>(dev[worst_i])));
  CHECK(worst_ratio <= 1.0);
}

Tensor MakeTDev(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

class DBuf {
 public:
  DBuf(Backend& b, Queue& q, const void* host, size_t bytes) : b_(&b), bytes_(bytes) {
    p_ = b.Alloc(bytes == 0 ? 1 : bytes);
    if (host != nullptr && bytes > 0) b.Copy(q, p_, host, bytes);
  }
  ~DBuf() {
    if (p_ != nullptr) b_->Free(p_);
  }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  void* get() const { return p_; }
  void Download(Queue& q, void* dst) const {
    if (bytes_ > 0) b_->Copy(q, dst, p_, bytes_);
    b_->Synchronize(q);
  }

 private:
  Backend* b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
};

// The CUDA twin of RunStateUpdate. `state_raw` is updated IN PLACE, as on host.
std::vector<float> RunStateUpdateCuda(Backend& gpu, std::vector<uint8_t>& state_raw, int64_t S,
                                      const StepInputs& in, const std::vector<int32_t>* slots,
                                      int64_t Nb, int64_t H, int64_t P, int64_t G, int64_t N,
                                      const StepCfg& cfg) {
  Queue q = gpu.CreateQueue();
  const Device dev{DeviceType::kCUDA, 0};
  const std::vector<uint8_t> xb = Pack(in.x, cfg.act_dtype);
  const std::vector<uint8_t> dtb = Pack(in.dt, cfg.act_dtype);
  const std::vector<uint8_t> Bb = Pack(in.B, cfg.act_dtype);
  const std::vector<uint8_t> Cb = Pack(in.C, cfg.act_dtype);
  const std::vector<uint8_t> zb = Pack(in.z, cfg.act_dtype);
  const size_t out_bytes = static_cast<size_t>(Nb * H * P) * vt::SizeOf(cfg.act_dtype);

  DBuf dstate(gpu, q, state_raw.data(), state_raw.size());
  DBuf dx(gpu, q, xb.data(), xb.size());
  DBuf ddt(gpu, q, dtb.data(), dtb.size());
  DBuf dA(gpu, q, in.A.data(), in.A.size() * sizeof(float));
  DBuf dB(gpu, q, Bb.data(), Bb.size());
  DBuf dC(gpu, q, Cb.data(), Cb.size());
  DBuf dD(gpu, q, in.D.data(), in.D.size() * sizeof(float));
  DBuf dz(gpu, q, zb.data(), zb.size());
  DBuf ddb(gpu, q, in.dt_bias.data(), in.dt_bias.size() * sizeof(float));
  DBuf dout(gpu, q, nullptr, out_bytes);

  Tensor st = MakeTDev(dstate.get(), cfg.state_dtype, dev, {S, H, P, N});
  Tensor xt = MakeTDev(dx.get(), cfg.act_dtype, dev, {Nb, H, P});
  Tensor dtt = MakeTDev(ddt.get(), cfg.act_dtype, dev, {Nb, H});
  Tensor At = MakeTDev(dA.get(), DType::kF32, dev, {H});
  Tensor Bt = MakeTDev(dB.get(), cfg.act_dtype, dev, {Nb, G, N});
  Tensor Ct = MakeTDev(dC.get(), cfg.act_dtype, dev, {Nb, G, N});
  Tensor Dt = MakeTDev(dD.get(), DType::kF32, dev, {H});
  Tensor zt = MakeTDev(dz.get(), cfg.act_dtype, dev, {Nb, H, P});
  Tensor dbt = MakeTDev(ddb.get(), DType::kF32, dev, {H});
  Tensor outt = MakeTDev(dout.get(), cfg.act_dtype, dev, {Nb, H, P});

  std::vector<int32_t> idx;
  std::unique_ptr<DBuf> didx;
  Tensor idxt;
  if (slots != nullptr) {
    idx = *slots;
    didx = std::make_unique<DBuf>(gpu, q, idx.data(), idx.size() * sizeof(int32_t));
    idxt = MakeTDev(didx->get(), DType::kI32, dev, {Nb});
  }

  Mamba2Args args;
  args.dt_softplus = cfg.dt_softplus;
  args.tp_world_size = cfg.tp_world_size;
  vt::Mamba2StateUpdate(q, outt, st, xt, dtt, At, Bt, Ct, cfg.use_D ? &Dt : nullptr,
                        cfg.use_z ? &zt : nullptr, cfg.use_dt_bias ? &dbt : nullptr,
                        slots != nullptr ? &idxt : nullptr, args);

  std::vector<uint8_t> outb(out_bytes);
  dout.Download(q, outb.data());
  dstate.Download(q, state_raw.data());
  gpu.Synchronize(q);
  gpu.DestroyQueue(q);
  return Unpack(outb, static_cast<size_t>(Nb * H * P), cfg.act_dtype);
}

}  // namespace

TEST_CASE("mamba2 state update CUDA arm matches the reference recurrence") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  // The driver shapes: nheads 64, headdim 64, dstate 128, ngroups 8
  // (mamba2-ssd.md §1.4), plus the small sweep the host arm runs.
  struct Shape {
    int64_t Nb, H, P, G, N;
  };
  const std::vector<Shape> shapes{{4, 64, 64, 8, 128}, {3, 16, 64, 1, 16}, {3, 16, 64, 4, 64}};
  for (const Shape& sh : shapes) {
    for (bool has_z : {false, true}) {
      const StepInputs in =
          GenerateStep(sh.Nb, sh.H, sh.P, sh.G, sh.N, 0x51A7Eu + static_cast<uint32_t>(sh.N));
      std::mt19937 rng(9u);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<float> state0(static_cast<size_t>(sh.Nb * sh.H * sh.P * sh.N));
      for (auto& v : state0) v = nd(rng);
      const std::vector<double> state0d(state0.begin(), state0.end());

      StepCfg cfg;
      cfg.use_z = has_z;
      const RefStep ref =
          SelectiveStateUpdateRef(state0d, in.x, in.dt, in.A, in.B, in.C, &in.D,
                                  has_z ? &in.z : nullptr, &in.dt_bias, true, sh.Nb, sh.H, sh.P,
                                  sh.G, sh.N);
      INFO("Nb=" << sh.Nb << " H=" << sh.H << " P=" << sh.P << " G=" << sh.G << " N=" << sh.N
                 << " has_z=" << has_z);

      std::vector<uint8_t> raw_dev = Pack(state0, DType::kF32);
      const std::vector<float> out_dev =
          RunStateUpdateCuda(*gpu, raw_dev, sh.Nb, in, nullptr, sh.Nb, sh.H, sh.P, sh.G, sh.N,
                             cfg);
      RequireNativeCudaProvider(vt::OpId::kMamba2StateUpdate, "plain decode step");
      std::vector<uint8_t> raw_host = Pack(state0, DType::kF32);
      const std::vector<float> out_host =
          RunStateUpdate(raw_host, sh.Nb, in, nullptr, sh.Nb, sh.H, sh.P, sh.G, sh.N, cfg);

      // G1 — the device arm against the INDEPENDENT double reference, at
      // upstream's own f32 tolerance (rtol 3e-4 / atol 1e-3, test_mamba_ssm.py:828).
      ExpectClose("device out", out_dev, ref.out, 1e-3, 3e-4);
      ExpectClose("device state", Unpack(raw_dev, state0.size(), DType::kF32), ref.state, 1e-3,
                  3e-4);
      // G2 — device vs host. A decode step runs exactly ONE decay factor and one
      // length-N summation over dstate, so K is N and the bound is at its
      // tightest anywhere in this row.
      ExpectDeviceMatchesHost("out device vs host", out_dev, out_host, sh.N);
      ExpectDeviceMatchesHost("state device vs host",
                              Unpack(raw_dev, state0.size(), DType::kF32),
                              Unpack(raw_host, state0.size(), DType::kF32), sh.N);
    }
  }
}

// SCATTERED CACHE SLOTS and the NULL row, on device. This is the case that pins
// `state_indices` being honoured at all: a kernel that wrote through the ROW
// index instead of the SLOT index, or that treated the NULL row as slot 0, passes
// nothing here. The evidence is BYTE-EXACT — no tolerance can hide a slot that
// was written when it should not have been.
TEST_CASE("mamba2 state update CUDA arm honours scattered slots and the NULL row") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  const int64_t H = 8, P = 32, N = 16, G = 2;
  const int64_t real = 3, padding = 5, Nb = real + padding;
  const int64_t S = 30;

  const StepInputs in = GenerateStep(Nb, H, P, G, N, 0xFACEu);
  std::mt19937 rng(11u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> cache(static_cast<size_t>(S * H * P * N));
  for (auto& v : cache) v = nd(rng);

  const std::vector<int32_t> chosen{17, 2, 25};
  std::vector<int32_t> slots = chosen;
  for (int64_t i = 0; i < padding; ++i) slots.push_back(-1);

  std::vector<double> state0d(static_cast<size_t>(real * H * P * N));
  for (int64_t b = 0; b < real; ++b) {
    const size_t src =
        static_cast<size_t>(chosen[static_cast<size_t>(b)]) * static_cast<size_t>(H * P * N);
    for (int64_t i = 0; i < H * P * N; ++i)
      state0d[static_cast<size_t>(b * H * P * N + i)] = cache[src + static_cast<size_t>(i)];
  }
  StepInputs real_in = in;
  real_in.x.resize(static_cast<size_t>(real * H * P));
  real_in.dt.resize(static_cast<size_t>(real * H));
  real_in.B.resize(static_cast<size_t>(real * G * N));
  real_in.C.resize(static_cast<size_t>(real * G * N));
  real_in.z.resize(static_cast<size_t>(real * H * P));
  const RefStep ref =
      SelectiveStateUpdateRef(state0d, real_in.x, real_in.dt, real_in.A, real_in.B, real_in.C,
                              &real_in.D, &real_in.z, &real_in.dt_bias, true, real, H, P, G, N);

  StepCfg cfg;
  cfg.use_z = true;
  std::vector<uint8_t> raw = Pack(cache, DType::kF32);
  const std::vector<uint8_t> before = raw;
  const std::vector<float> out = RunStateUpdateCuda(*gpu, raw, S, in, &slots, Nb, H, P, G, N, cfg);
  RequireNativeCudaProvider(vt::OpId::kMamba2StateUpdate, "scattered slots");
  const std::vector<float> after = Unpack(raw, cache.size(), DType::kF32);

  // (a) the three selected slots hold the advanced state.
  std::vector<float> got_state(static_cast<size_t>(real * H * P * N));
  for (int64_t b = 0; b < real; ++b) {
    const size_t src =
        static_cast<size_t>(chosen[static_cast<size_t>(b)]) * static_cast<size_t>(H * P * N);
    for (int64_t i = 0; i < H * P * N; ++i)
      got_state[static_cast<size_t>(b * H * P * N + i)] = after[src + static_cast<size_t>(i)];
  }
  ExpectClose("device scattered state", got_state, ref.state, 1e-3, 3e-4);

  // (b) the three real output rows match.
  std::vector<float> got_out(out.begin(), out.begin() + static_cast<size_t>(real * H * P));
  ExpectClose("device scattered out", got_out, ref.out, 1e-3, 3e-4);

  // (c) EVERY slot that was not selected is BYTE-identical — upstream's
  //     `torch.equal(state_before[unused], state[unused])` (test_mamba_ssm.py:800).
  //     Slot 0 is among them, so a kernel that mapped the NULL row onto slot 0
  //     fails here and nowhere else.
  size_t untouched_slots = 0;
  for (int64_t s = 0; s < S; ++s) {
    if (std::find(chosen.begin(), chosen.end(), static_cast<int32_t>(s)) != chosen.end()) continue;
    ++untouched_slots;
    const size_t off = static_cast<size_t>(s) * static_cast<size_t>(H * P * N) * 4;
    CHECK(std::memcmp(before.data() + off, raw.data() + off,
                      static_cast<size_t>(H * P * N) * 4) == 0);
  }
  CHECK(untouched_slots == static_cast<size_t>(S - real));

  // (d) the NULL rows write an EXACTLY zeroed output row.
  for (int64_t b = real; b < Nb; ++b)
    for (int64_t i = 0; i < H * P; ++i)
      CHECK(out[static_cast<size_t>(b * H * P + i)] == 0.0f);
}

// DECODE == PREFILL, both arms on device. The two go through entirely different
// device code (a single-token step vs the 5-stage chunked pipeline), which is
// what makes the agreement evidence rather than a shared-helper tautology
// ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]).
TEST_CASE("mamba2 state update CUDA arm reproduces the CUDA chunked prefill") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  const int64_t T = 40, H = 4, P = 8, G = 2, N = 16, chunk = 8;
  std::mt19937 rng(0x1234u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);

  std::vector<float> A(static_cast<size_t>(H));
  for (auto& v : A) v = -ud(rng) - 1.0f;
  std::vector<float> D(static_cast<size_t>(H));
  for (auto& v : D) v = nd(rng);
  std::vector<float> dt_bias(static_cast<size_t>(H));
  for (auto& v : dt_bias) v = ud(rng) - 4.0f;
  std::vector<float> x(static_cast<size_t>(T * H * P));
  for (auto& v : x) v = nd(rng);
  std::vector<float> dt(static_cast<size_t>(T * H));
  for (auto& v : dt) v = nd(rng);
  std::vector<float> B(static_cast<size_t>(T * G * N));
  for (auto& v : B) v = nd(rng);
  std::vector<float> C(static_cast<size_t>(T * G * N));
  for (auto& v : C) v = nd(rng);

  // ── prefill arm, on device ──
  std::vector<float> y_prefill(static_cast<size_t>(T * H * P), 0.0f);
  std::vector<float> fs(static_cast<size_t>(H * P * N), 0.0f);
  {
    Queue q = gpu->CreateQueue();
    const Device dev{DeviceType::kCUDA, 0};
    std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
    std::vector<int32_t> ccs{0};
    std::vector<int32_t> sidx;
    for (int64_t pos = 0; pos < T; pos += chunk) {
      ccs.push_back(static_cast<int32_t>(std::min(pos + chunk, T)));
      sidx.push_back(0);
    }
    std::vector<int32_t> lci{static_cast<int32_t>(sidx.size()) - 1};

    DBuf dx(*gpu, q, x.data(), x.size() * sizeof(float));
    DBuf ddt(*gpu, q, dt.data(), dt.size() * sizeof(float));
    DBuf dA(*gpu, q, A.data(), A.size() * sizeof(float));
    DBuf dB(*gpu, q, B.data(), B.size() * sizeof(float));
    DBuf dC(*gpu, q, C.data(), C.size() * sizeof(float));
    DBuf dD(*gpu, q, D.data(), D.size() * sizeof(float));
    DBuf ddb(*gpu, q, dt_bias.data(), dt_bias.size() * sizeof(float));
    DBuf dout(*gpu, q, nullptr, y_prefill.size() * sizeof(float));
    DBuf dfs(*gpu, q, nullptr, fs.size() * sizeof(float));
    DBuf dcu(*gpu, q, cu.data(), cu.size() * sizeof(int32_t));
    DBuf dccs(*gpu, q, ccs.data(), ccs.size() * sizeof(int32_t));
    DBuf dlci(*gpu, q, lci.data(), lci.size() * sizeof(int32_t));
    DBuf dsi(*gpu, q, sidx.data(), sidx.size() * sizeof(int32_t));

    Tensor xt = MakeTDev(dx.get(), DType::kF32, dev, {T, H, P});
    Tensor dtt = MakeTDev(ddt.get(), DType::kF32, dev, {T, H});
    Tensor At = MakeTDev(dA.get(), DType::kF32, dev, {H});
    Tensor Bt = MakeTDev(dB.get(), DType::kF32, dev, {T, G, N});
    Tensor Ct = MakeTDev(dC.get(), DType::kF32, dev, {T, G, N});
    Tensor Dt = MakeTDev(dD.get(), DType::kF32, dev, {H});
    Tensor dbt = MakeTDev(ddb.get(), DType::kF32, dev, {H});
    Tensor outt = MakeTDev(dout.get(), DType::kF32, dev, {T, H, P});
    Tensor fst = MakeTDev(dfs.get(), DType::kF32, dev, {1, H, P, N});
    Tensor cust = MakeTDev(dcu.get(), DType::kI32, dev, {2});
    Tensor ccst = MakeTDev(dccs.get(), DType::kI32, dev, {static_cast<int64_t>(ccs.size())});
    Tensor lcit = MakeTDev(dlci.get(), DType::kI32, dev, {1});
    Tensor sit = MakeTDev(dsi.get(), DType::kI32, dev, {static_cast<int64_t>(sidx.size())});
    Mamba2Args args;
    args.chunk_size = chunk;
    args.dt_softplus = true;
    vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, &Dt, nullptr, &dbt, nullptr, cust,
                        ccst, lcit, sit, args);
    dout.Download(q, y_prefill.data());
    dfs.Download(q, fs.data());
    gpu->Synchronize(q);
    gpu->DestroyQueue(q);
  }
  RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "decode-vs-prefill: prefill arm");

  // ── decode arm: T single-token device steps on one cache slot ──
  std::vector<uint8_t> raw(static_cast<size_t>(H * P * N) * 4, 0);
  std::vector<float> y_decode(static_cast<size_t>(T * H * P), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    StepInputs step;
    step.A = A;
    step.D = D;
    step.dt_bias = dt_bias;
    step.x.assign(x.begin() + static_cast<size_t>(t * H * P),
                  x.begin() + static_cast<size_t>((t + 1) * H * P));
    step.dt.assign(dt.begin() + static_cast<size_t>(t * H),
                   dt.begin() + static_cast<size_t>((t + 1) * H));
    step.B.assign(B.begin() + static_cast<size_t>(t * G * N),
                  B.begin() + static_cast<size_t>((t + 1) * G * N));
    step.C.assign(C.begin() + static_cast<size_t>(t * G * N),
                  C.begin() + static_cast<size_t>((t + 1) * G * N));
    step.z.assign(static_cast<size_t>(H * P), 0.0f);
    StepCfg cfg;
    const std::vector<float> o =
        RunStateUpdateCuda(*gpu, raw, 1, step, nullptr, 1, H, P, G, N, cfg);
    std::copy(o.begin(), o.end(), y_decode.begin() + static_cast<size_t>(t * H * P));
  }

  RequireNativeCudaProvider(vt::OpId::kMamba2StateUpdate, "decode-vs-prefill: decode arm");
  ExpectCloseF("device decode vs device chunked prefill", y_decode, y_prefill, 5e-3, 5e-3);
  ExpectCloseF("device final state", Unpack(raw, static_cast<size_t>(H * P * N), DType::kF32), fs,
               5e-3, 5e-3);
}

// The SSM cache dtype is its own knob (mamba_utils.py:73-81), and `out` does not
// depend on it at all: the readout uses the f32 state held in registers, not the
// value re-read from the cache (mamba_ssm.py:433,451). On device that is the
// EXACT, tolerance-free statement it is on host.
TEST_CASE("mamba2 state update CUDA arm reads out the f32 state, not the rounded cache") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  const int64_t Nb = 2, H = 4, P = 8, G = 2, N = 16;
  const StepInputs in = GenerateStep(Nb, H, P, G, N, 0xDEC0DEu);
  std::mt19937 rng(0x99u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> state0(static_cast<size_t>(Nb * H * P * N));
  for (auto& v : state0) v = vt::BF16ToF32(vt::F32ToBF16(nd(rng)));

  for (bool has_z : {false, true}) {
    StepCfg f32cfg;
    f32cfg.use_z = has_z;
    StepCfg bf16cfg = f32cfg;
    bf16cfg.state_dtype = DType::kBF16;

    std::vector<uint8_t> raw_f32 = Pack(state0, DType::kF32);
    const std::vector<float> out_f32 =
        RunStateUpdateCuda(*gpu, raw_f32, Nb, in, nullptr, Nb, H, P, G, N, f32cfg);
    std::vector<uint8_t> raw_bf16 = Pack(state0, DType::kBF16);
    const std::vector<float> out_bf16 =
        RunStateUpdateCuda(*gpu, raw_bf16, Nb, in, nullptr, Nb, H, P, G, N, bf16cfg);
    RequireNativeCudaProvider(vt::OpId::kMamba2StateUpdate, "cache-dtype independence");

    REQUIRE(out_f32.size() == out_bf16.size());
    size_t differing = 0;
    for (size_t i = 0; i < out_f32.size(); ++i)
      if (out_f32[i] != out_bf16[i]) ++differing;
    INFO("has_z=" << has_z << ": " << differing << " of " << out_f32.size()
                  << " device outputs move with the CACHE dtype");
    CHECK(differing == 0);

    // The cache rounding really is live, so the check above is not comparing two
    // identical computations.
    const std::vector<float> after_f32 = Unpack(raw_f32, state0.size(), DType::kF32);
    const std::vector<float> after_bf16 = Unpack(raw_bf16, state0.size(), DType::kBF16);
    for (float v : after_bf16) CHECK(vt::BF16ToF32(vt::F32ToBF16(v)) == v);
    size_t rounded = 0;
    for (size_t i = 0; i < after_f32.size(); ++i)
      if (after_f32[i] != after_bf16[i]) ++rounded;
    INFO("stored states the bf16 cache rounded: " << rounded);
    REQUIRE(rounded > 0);
  }
}

#endif  // VLLM_CPP_CUDA
