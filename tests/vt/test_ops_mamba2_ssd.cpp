// Mamba2 chunked SSD prefill scan (vt::Mamba2ChunkScan) — UNIT GATE.
// .agents/specs/mamba2-ssd.md W1, issue #496.
//
// Ported from tests/kernels/mamba/test_mamba_ssm_ssd.py @ pin 555967922
// (vLLM 0.26.0.dev0), preserving its shapes, parameters and tolerances.
// The op under test mirrors `mamba_chunk_scan_combined_varlen`
// (vllm/model_executor/layers/mamba/ops/ssd_combined.py:157-235) over the
// 5-stage pipeline `_mamba_chunk_scan_combined_fwd` (:27-156).
//
// HARNESS ADAPTATION (documented, per porting.md). Upstream's reference is
// `ssd_minimal_discrete` (test_mamba_ssm_ssd.py:42-90), a *second chunked*
// implementation written in einsum. Comparing one chunked implementation to
// another cannot see a defect both share
// ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]), so the
// reference here is the SEQUENTIAL per-token recurrence written independently in
// `double` (`SequentialSsdRef` below) — the same recurrence upstream's OWN CPU
// kernel runs token-by-token (csrc/cpu/mamba_kernels.hpp:279-382,
// `mamba_chunk_scan_fwd_kernel`, reached via
// `_mamba_chunk_scan_combined_fwd_cpu`, ops/cpu/mamba_ssm.py:9-78). The chunked
// scan is an algebraic identity for that recurrence, so it is a strictly
// stronger gate than upstream's, and it is the one that catches a wrong
// `dA_cumsum` or a dropped inter-chunk term.
//
// ─── WHY THESE ARE THE CORRECTNESS EVIDENCE ───────────────────────────────────
// (1) CHUNKED == SEQUENTIAL. The whole point of the SSD factorisation. A wrong
//     per-chunk cumulative decay, a dropped state-passing term, or an off-by-one
//     in `states[c-1]` all break this and nothing else catches them.
// (2) CHUNK-BOUNDARY INVARIANCE. The same sequence at chunk_size in
//     {8,16,32,64,128} must agree. A state-passing defect is invisible at ONE
//     chunk size and loud across five — the failure shape of
//     [[h3-video-decode-temporal-and-tiling-compose]], where the gates ran below
//     one chunk and saw nothing.
// (3) SEQUENCE BOUNDARIES INSIDE A PHYSICAL CHUNK, with and without
//     `initial_states` — the `seq_idx[c] != seq_idx[c-1]` branch
//     (ssd_chunk_scan.py:236-250) that continuous batching lives on.
// (4) CHUNKED PREFILL EQUIVALENCE: a sequence fed in two calls, carrying
//     `final_states` across, equals the same sequence fed whole.
// (5) An F32 OUTPUT ARM is swept alongside every bf16 arm — a bf16 store absorbs
//     real reduction-order defects ([[bf16-store-absorbs-reduction-order-defects]]).
// (6) The SSM state dtype is a SEPARATE knob from the activation dtype
//     (`state_dtype`, ssd_combined.py:46,119,176); a test pins that it is not
//     derived from the activation dtype in either direction.
//
// TOLERANCES. `doctest::Approx` is deliberately NOT used: its `scale` defaults to
// 1.0, which floors every comparison at ~1.19e-5 absolute, so `Approx(1e-6)`
// accepts 1e-5, 1e-7 AND 0 ([[doctest-approx-scale-term-floor]]). Every check
// below goes through `ExpectClose`, which implements `torch.testing.assert_close`
// arithmetic (`|got-want| <= atol + rtol*|want|`) explicitly and reports the
// worst element.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

// ─── comparison ──────────────────────────────────────────────────────────────
// torch.testing.assert_close arithmetic, stated explicitly. Reports the worst
// offending element so a failure names a number, not just "false".
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
    if (!(std::isfinite(g)) || slack > worst_slack) {
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
  std::vector<double> w(want.begin(), want.end());
  ExpectClose(what, got, w, atol, rtol);
}

// ─── random inputs, mirroring test_mamba_ssm_ssd.py:93-104 ───────────────────
//   A  = -exp(rand(nheads))            dt = softplus(randn(T, nheads) - 4)
//   X/B/C = randn(...)
struct Inputs {
  std::vector<float> x, dt, B, C;  // x [T,H,P]; dt [T,H]; B/C [T,G,N]
  std::vector<float> A;            // [H] f32, negative
};

Inputs GenerateInputs(int64_t T, int64_t H, int64_t P, int64_t G, int64_t N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  Inputs in;
  in.A.resize(static_cast<size_t>(H));
  for (auto& a : in.A) a = -std::exp(ud(rng));
  in.dt.resize(static_cast<size_t>(T * H));
  for (auto& d : in.dt) {
    const float v = nd(rng) - 4.0f;
    d = std::log1p(std::exp(v));  // softplus, already applied (dt_softplus=False)
  }
  in.x.resize(static_cast<size_t>(T * H * P));
  for (auto& v : in.x) v = nd(rng);
  in.B.resize(static_cast<size_t>(T * G * N));
  for (auto& v : in.B) v = nd(rng);
  in.C.resize(static_cast<size_t>(T * G * N));
  for (auto& v : in.C) v = nd(rng);
  return in;
}

// Upstream generates every tensor directly in `itype`
// (generate_random_inputs, test_mamba_ssm_ssd.py:93-104), so the reference and
// the kernel see the SAME reduced-precision values and the comparison measures
// the KERNEL, not the input quantization. Rounding the inputs first reproduces
// that. `A` stays an f32 tensor by the vt contract but is rounded to the same
// grid, because upstream's `A` is generated in `itype` too (:95).
void RoundInputsTo(Inputs& in, DType dt) {
  if (dt != DType::kBF16) return;
  const auto round = [](std::vector<float>& v) {
    for (auto& e : v) e = vt::BF16ToF32(vt::F32ToBF16(e));
  };
  round(in.x);
  round(in.dt);
  round(in.B);
  round(in.C);
  round(in.A);
}

// ─── varlen chunk metadata ───────────────────────────────────────────────────
// Ported 1:1 from `compute_varlen_chunk_metadata`
// (vllm/v1/attention/backends/mamba2_attn.py:22-88). Logical chunks never cross
// a sequence boundary OR a physical chunk boundary, and the physical boundary is
// taken on the GLOBAL token position (`pos % chunk_size`, :62).
struct ChunkMeta {
  std::vector<int32_t> cu_chunk_seqlens;    // [nchunks+1]
  std::vector<int32_t> last_chunk_indices;  // [S]
  std::vector<int32_t> seq_idx;             // [nchunks] — PER CHUNK (:60-61)
};

ChunkMeta ComputeVarlenChunkMetadata(const std::vector<int32_t>& cu_seqlens,
                                     int64_t chunk_size) {
  ChunkMeta m;
  const size_t nseq = cu_seqlens.size() - 1;
  m.last_chunk_indices.assign(nseq, -1);
  std::vector<int32_t> chunk_lens;
  for (size_t b = 0; b < nseq; ++b) {
    const int64_t s = cu_seqlens[b], e = cu_seqlens[b + 1];
    if (e <= s) continue;
    int64_t pos = s;
    while (pos < e) {
      const int64_t room = chunk_size - (pos % chunk_size);
      const int64_t take = std::min(room, e - pos);
      chunk_lens.push_back(static_cast<int32_t>(take));
      m.seq_idx.push_back(static_cast<int32_t>(b));
      m.last_chunk_indices[b] = static_cast<int32_t>(chunk_lens.size()) - 1;
      pos += take;
    }
  }
  m.cu_chunk_seqlens.push_back(0);
  for (int32_t len : chunk_lens)
    m.cu_chunk_seqlens.push_back(m.cu_chunk_seqlens.back() + len);
  return m;
}

// ─── the SEQUENTIAL double-precision reference ───────────────────────────────
// Independent statement of the Mamba2 selective-scan recurrence, in `double`.
// Structurally the same loop upstream's own CPU prefill kernel runs
// (csrc/cpu/mamba_kernels.hpp:279-382): dt preprocessing (bias -> softplus ->
// dt_limit clamp) is `_mamba_chunk_scan_combined_fwd_cpu` ops/cpu/mamba_ssm.py:37-43
// and `_chunk_cumsum_fwd_kernel` ssd_chunk_state.py:87-96; the per-token update is
//   s = s*exp(A[h]*dt) + B[t,g,:]*x[t,h,p]*dt
//   y = sum_n s[n]*C[t,g,n] + D*x        (then y *= z*sigmoid(z))
// Nothing here is shared with the op under test.
struct SeqRefOut {
  std::vector<double> y;             // [T,H,P]
  std::vector<double> final_states;  // [S,H,P,N]
};

struct RefCfg {
  bool dt_softplus = false;
  double dt_min = 0.0;
  double dt_max = std::numeric_limits<double>::infinity();
};

SeqRefOut SequentialSsdRef(const Inputs& in, int64_t T, int64_t H, int64_t P, int64_t G,
                           int64_t N, const std::vector<int32_t>& cu_seqlens,
                           const std::vector<float>* D, bool d_has_hdim,
                           const std::vector<float>* z, const std::vector<float>* dt_bias,
                           const std::vector<double>* initial_states, const RefCfg& cfg) {
  const size_t nseq = cu_seqlens.size() - 1;
  SeqRefOut out;
  out.y.assign(static_cast<size_t>(T * H * P), 0.0);
  out.final_states.assign(static_cast<size_t>(static_cast<int64_t>(nseq) * H * P * N), 0.0);
  const int64_t heads_per_group = H / G;
  for (size_t b = 0; b < nseq; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t g = h / heads_per_group;
      const double a = in.A[static_cast<size_t>(h)];
      std::vector<double> s(static_cast<size_t>(P * N), 0.0);
      if (initial_states != nullptr) {
        const size_t base = static_cast<size_t>((static_cast<int64_t>(b) * H + h) * P * N);
        for (int64_t i = 0; i < P * N; ++i)
          s[static_cast<size_t>(i)] = (*initial_states)[base + static_cast<size_t>(i)];
      }
      for (int64_t t = cu_seqlens[b]; t < cu_seqlens[b + 1]; ++t) {
        double dt = in.dt[static_cast<size_t>(t * H + h)];
        if (dt_bias != nullptr) dt += (*dt_bias)[static_cast<size_t>(h)];
        if (cfg.dt_softplus) dt = dt <= 20.0 ? std::log1p(std::exp(dt)) : dt;
        dt = std::min(std::max(dt, cfg.dt_min), cfg.dt_max);
        const double dA = std::exp(a * dt);
        for (int64_t p = 0; p < P; ++p) {
          const double xv = in.x[static_cast<size_t>((t * H + h) * P + p)];
          double y = 0.0;
          for (int64_t n = 0; n < N; ++n) {
            const double Bv = in.B[static_cast<size_t>((t * G + g) * N + n)];
            const double Cv = in.C[static_cast<size_t>((t * G + g) * N + n)];
            double& sv = s[static_cast<size_t>(p * N + n)];
            sv = sv * dA + Bv * xv * dt;
            y += sv * Cv;
          }
          if (D != nullptr) {
            const double dval =
                d_has_hdim ? (*D)[static_cast<size_t>(h * P + p)] : (*D)[static_cast<size_t>(h)];
            y += dval * xv;
          }
          if (z != nullptr) {
            const double zv = (*z)[static_cast<size_t>((t * H + h) * P + p)];
            y *= zv / (1.0 + std::exp(-zv));
          }
          out.y[static_cast<size_t>((t * H + h) * P + p)] = y;
        }
      }
      const size_t base = static_cast<size_t>((static_cast<int64_t>(b) * H + h) * P * N);
      for (int64_t i = 0; i < P * N; ++i)
        out.final_states[base + static_cast<size_t>(i)] = s[static_cast<size_t>(i)];
    }
  }
  return out;
}

// ─── op driver ───────────────────────────────────────────────────────────────
struct RunCfg {
  int64_t chunk_size = 0;
  DType act_dtype = DType::kF32;    // x / dt / B / C / z / out
  DType state_dtype = DType::kF32;  // final_states / initial_states
  bool dt_softplus = false;
  float dt_min = 0.0f;
  float dt_max = std::numeric_limits<float>::infinity();
  bool d_has_hdim = false;  // D is [H,P] rather than [H] (ssd_combined.py:56-57)
};

struct RunOut {
  std::vector<float> y;             // [T,H,P] read back as f32
  std::vector<float> final_states;  // [S,H,P,N] read back as f32
};

// Bytes of `src` in `dt`; the tensor's declared dtype is what LoadF32/StoreF32 use.
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

RunOut RunChunkScan(const Inputs& in, int64_t T, int64_t H, int64_t P, int64_t G, int64_t N,
                    const std::vector<int32_t>& cu_seqlens, const std::vector<float>* D,
                    const std::vector<float>* z, const std::vector<float>* dt_bias,
                    const std::vector<float>* initial_states, const RunCfg& cfg) {
  Queue q = CpuQ();
  const int64_t S = static_cast<int64_t>(cu_seqlens.size()) - 1;
  const ChunkMeta meta = ComputeVarlenChunkMetadata(cu_seqlens, cfg.chunk_size);
  const int64_t nchunks = static_cast<int64_t>(meta.seq_idx.size());

  std::vector<uint8_t> xb = Pack(in.x, cfg.act_dtype);
  std::vector<uint8_t> dtb = Pack(in.dt, cfg.act_dtype);
  std::vector<uint8_t> Bb = Pack(in.B, cfg.act_dtype);
  std::vector<uint8_t> Cb = Pack(in.C, cfg.act_dtype);
  std::vector<float> Ac = in.A;
  std::vector<uint8_t> outb(static_cast<size_t>(T * H * P) * vt::SizeOf(cfg.act_dtype), 0);
  std::vector<uint8_t> fsb(static_cast<size_t>(S * H * P * N) * vt::SizeOf(cfg.state_dtype), 0);

  Tensor xt = MakeT(xb.data(), cfg.act_dtype, {T, H, P});
  Tensor dtt = MakeT(dtb.data(), cfg.act_dtype, {T, H});
  Tensor At = MakeT(Ac.data(), DType::kF32, {H});
  Tensor Bt = MakeT(Bb.data(), cfg.act_dtype, {T, G, N});
  Tensor Ct = MakeT(Cb.data(), cfg.act_dtype, {T, G, N});
  Tensor outt = MakeT(outb.data(), cfg.act_dtype, {T, H, P});
  Tensor fst = MakeT(fsb.data(), cfg.state_dtype, {S, H, P, N});

  std::vector<float> Dc;
  Tensor Dt;
  if (D != nullptr) {
    Dc = *D;
    Dt = cfg.d_has_hdim ? MakeT(Dc.data(), DType::kF32, {H, P})
                        : MakeT(Dc.data(), DType::kF32, {H});
  }
  std::vector<uint8_t> zb;
  Tensor zt;
  if (z != nullptr) {
    zb = Pack(*z, cfg.act_dtype);
    zt = MakeT(zb.data(), cfg.act_dtype, {T, H, P});
  }
  std::vector<float> dbc;
  Tensor dbt;
  if (dt_bias != nullptr) {
    dbc = *dt_bias;
    dbt = MakeT(dbc.data(), DType::kF32, {H});
  }
  std::vector<uint8_t> isb;
  Tensor ist;
  if (initial_states != nullptr) {
    isb = Pack(*initial_states, cfg.state_dtype);
    ist = MakeT(isb.data(), cfg.state_dtype, {S, H, P, N});
  }

  std::vector<int32_t> cus = cu_seqlens;
  ChunkMeta m = meta;
  Tensor cust = MakeT(cus.data(), DType::kI32, {S + 1});
  Tensor ccst = MakeT(m.cu_chunk_seqlens.data(), DType::kI32, {nchunks + 1});
  Tensor lcit = MakeT(m.last_chunk_indices.data(), DType::kI32, {S});
  Tensor sit = MakeT(m.seq_idx.data(), DType::kI32, {nchunks});

  Mamba2Args args;
  args.chunk_size = cfg.chunk_size;
  args.dt_softplus = cfg.dt_softplus;
  args.dt_min = cfg.dt_min;
  args.dt_max = cfg.dt_max;

  vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, D != nullptr ? &Dt : nullptr,
                      z != nullptr ? &zt : nullptr, dt_bias != nullptr ? &dbt : nullptr,
                      initial_states != nullptr ? &ist : nullptr, cust, ccst, lcit, sit, args);

  RunOut r;
  r.y = Unpack(outb, static_cast<size_t>(T * H * P), cfg.act_dtype);
  r.final_states = Unpack(fsb, static_cast<size_t>(S * H * P * N), cfg.state_dtype);
  return r;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// (1) CHUNKED == SEQUENTIAL, single example.
// Shapes and tolerances from test_mamba_chunk_scan_single_example
// (test_mamba_ssm_ssd.py:200-263): d_head in {5,8,32,128}, n_heads in {4,16,32},
// (seqlen, chunk_size) in {(112,16),(128,32)}, itype in {f32, bf16};
// f32 atol/rtol = 8e-3 / 5e-3, bf16 = 5e-2 / 5e-2.
// Upstream sets ngroups == nheads and dstate == d_head
// (generate_random_inputs, :93-104); the ngroups < nheads sharing is covered
// separately below.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan equals the sequential recurrence (single example)") {
  for (int64_t d_head : {5, 8, 32, 128}) {
    for (int64_t n_heads : {4, 16, 32}) {
      for (auto sc : {std::pair<int64_t, int64_t>{112, 16}, std::pair<int64_t, int64_t>{128, 32}}) {
        const int64_t T = sc.first, chunk = sc.second;
        const int64_t H = n_heads, P = d_head, G = n_heads, N = d_head;
        const Inputs in = GenerateInputs(T, H, P, G, N, 0xA11CEu);
        const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
        const SeqRefOut ref =
            SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});

        // f32 arm — the one a bf16 store cannot hide a reduction-order defect in.
        RunCfg cfg;
        cfg.chunk_size = chunk;
        const RunOut got = RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr,
                                        nullptr, cfg);
        INFO("f32 d_head=" << d_head << " n_heads=" << n_heads << " T=" << T
                           << " chunk=" << chunk);
        ExpectClose("y", got.y, ref.y, 8e-3, 5e-3);
        ExpectClose("final_states", got.final_states, ref.final_states, 8e-3, 5e-3);

        // bf16 activation arm, upstream's own looser threshold (5e-2 / 5e-2,
        // test_mamba_ssm_ssd.py:210-213), against a reference that sees the
        // SAME bf16-rounded inputs.
        RunCfg bcfg = cfg;
        bcfg.act_dtype = DType::kBF16;
        Inputs bin = in;
        RoundInputsTo(bin, DType::kBF16);
        const SeqRefOut bref = SequentialSsdRef(bin, T, H, P, G, N, cu, nullptr, false, nullptr,
                                                nullptr, nullptr, {});
        const RunOut gotb = RunChunkScan(bin, T, H, P, G, N, cu, nullptr, nullptr, nullptr,
                                         nullptr, bcfg);
        INFO("bf16 d_head=" << d_head << " n_heads=" << n_heads);
        ExpectClose("y bf16", gotb.y, bref.y, 5e-2, 5e-2);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) CHUNK-BOUNDARY INVARIANCE — NOT an upstream test.
// The same sequence scanned at chunk_size in {8,16,32,64,128} must produce the
// same output. A state-passing defect is invisible at one chunk size.
// ─────────────────────────────────────────────────────────────────────────────
// T IS 300, NOT 128. With T == 128 the largest chunk size in the sweep leaves
// `nchunks == 1`, so that arm ran ZERO state passing — and 128 is exactly the
// `chunk_size` Nemotron-3.5 ships (.agents/specs/nemotron-h-model.md), i.e. the
// one arm that most needed the coverage was the degenerate one. That is the
// failure shape of [[h3-video-decode-temporal-and-tiling-compose]] again. The
// test now ASSERTS `nchunks > 1` for every chunk size rather than trusting the
// shape. Upstream's own tolerance loosens to atol 1e-2 past 256 tokens
// (test_mamba_ssm_ssd.py:266-355), and this sequence is longer than that.
TEST_CASE("mamba2 chunk scan is invariant to chunk_size") {
  const int64_t T = 300, H = 8, P = 16, G = 2, N = 32;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0xC0FFEEu);
  const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
  const SeqRefOut ref =
      SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});

  std::vector<float> first_y, first_state;
  for (int64_t chunk : {8, 16, 32, 64, 128}) {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    // The arm must actually EXERCISE inter-chunk state passing.
    const int64_t nchunks =
        static_cast<int64_t>(ComputeVarlenChunkMetadata(cu, chunk).seq_idx.size());
    INFO("chunk_size=" << chunk << " nchunks=" << nchunks);
    REQUIRE(nchunks > 1);
    const RunOut got = RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr,
                                    cfg);
    ExpectClose("y vs sequential", got.y, ref.y, 1e-2, 5e-3);
    ExpectClose("final_states vs sequential", got.final_states, ref.final_states, 1e-2, 5e-3);
    if (first_y.empty()) {
      first_y = got.y;
      first_state = got.final_states;
    } else {
      ExpectCloseF("y vs chunk_size=8", got.y, first_y, 1e-2, 5e-3);
      ExpectCloseF("final_states vs chunk_size=8", got.final_states, first_state, 1e-2, 5e-3);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) CONTINUOUS BATCHING — sequences that start and end INSIDE a physical
// chunk, with and without initial_states. Cases from
// test_mamba_chunk_scan_cont_batch (test_mamba_ssm_ssd.py:266-355) and
// test_mamba_chunk_scan_cont_batch_prefill_chunking (:358+): chunk_size in
// {8, 256}, irregular sequence lengths, and a long-sequence case that catches
// "errors with init states decay". Upstream f32 tolerance is 5e-3/5e-3, or
// 1e-2/5e-3 when max seqlen > 256.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan handles continuous batches and sequence boundaries") {
  struct Case {
    std::vector<int32_t> lens;
    int64_t chunk;
  };
  const std::vector<Case> cases{
      {{64, 32}, 8},          // chunk size boundary
      {{8, 8, 8}, 8},         // exactly one chunk each
      {{4, 4, 4, 4}, 8},      // chunk_size larger than the sequences
      {{64, 32, 16, 8, 8}, 8},
      {{5, 1, 1, 1}, 256},    // irregular, all far below one chunk
      {{5, 30, 1, 2}, 256},
      {{138, 225}, 128},      // long enough to catch init-state decay errors
      {{270, 88}, 8},
      {{16, 20}, 8},
  };
  const int64_t H = 8, P = 16, G = 2, N = 16;
  for (const Case& c : cases) {
    std::vector<int32_t> cu{0};
    for (int32_t l : c.lens) cu.push_back(cu.back() + l);
    const int64_t T = cu.back();
    const int64_t S = static_cast<int64_t>(c.lens.size());
    const int64_t maxlen = *std::max_element(c.lens.begin(), c.lens.end());
    const double atol = maxlen > 256 ? 1e-2 : 5e-3;
    const Inputs in = GenerateInputs(T, H, P, G, N, 0xBEEF01u + static_cast<uint32_t>(c.chunk));

    INFO("chunk=" << c.chunk << " nseq=" << S << " T=" << T);

    // (a) fresh sequences: the `seq_idx[c] != seq_idx[c-1]` branch must take
    //     ZEROS as the previous state (ssd_chunk_scan.py:271-274).
    {
      const SeqRefOut ref =
          SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});
      RunCfg cfg;
      cfg.chunk_size = c.chunk;
      const RunOut got =
          RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
      ExpectClose("y (fresh)", got.y, ref.y, atol, 5e-3);
      ExpectClose("final_states (fresh)", got.final_states, ref.final_states, atol, 5e-3);
    }

    // (b) with initial_states: the same branch must instead take
    //     initial_states[seq_idx[c]] (ssd_chunk_scan.py:236-243).
    {
      std::mt19937 rng(1234u);
      std::normal_distribution<float> nd(0.0f, 0.5f);
      std::vector<float> init(static_cast<size_t>(S * H * P * N));
      for (auto& v : init) v = nd(rng);
      const std::vector<double> initd(init.begin(), init.end());
      const SeqRefOut ref =
          SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, &initd, {});
      RunCfg cfg;
      cfg.chunk_size = c.chunk;
      const RunOut got =
          RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, &init, cfg);
      ExpectClose("y (init states)", got.y, ref.y, atol, 5e-3);
      ExpectClose("final_states (init states)", got.final_states, ref.final_states, atol, 5e-3);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) CHUNKED PREFILL EQUIVALENCE. Feeding a sequence in two calls and carrying
// `final_states` across must equal feeding it whole — upstream's
// test_mamba_chunk_scan_cont_batch_prefill_chunking (:358-...), and the actual
// operational contract of chunked prefill.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan splits a prefill without drift") {
  const int64_t H = 8, P = 16, G = 4, N = 16;
  for (int64_t chunk : {8, 32, 256}) {
    for (int64_t split : {1, 7, 64, 100}) {
      const int64_t T = 137;
      if (split >= T) continue;
      const Inputs in = GenerateInputs(T, H, P, G, N, 0x5EED77u);
      const std::vector<int32_t> whole{0, static_cast<int32_t>(T)};
      RunCfg cfg;
      cfg.chunk_size = chunk;
      const RunOut ref = RunChunkScan(in, T, H, P, G, N, whole, nullptr, nullptr, nullptr,
                                      nullptr, cfg);

      // First leg: tokens [0, split).
      Inputs leg1 = in;
      leg1.x.assign(in.x.begin(), in.x.begin() + static_cast<size_t>(split * H * P));
      leg1.dt.assign(in.dt.begin(), in.dt.begin() + static_cast<size_t>(split * H));
      leg1.B.assign(in.B.begin(), in.B.begin() + static_cast<size_t>(split * G * N));
      leg1.C.assign(in.C.begin(), in.C.begin() + static_cast<size_t>(split * G * N));
      const std::vector<int32_t> cu1{0, static_cast<int32_t>(split)};
      const RunOut r1 =
          RunChunkScan(leg1, split, H, P, G, N, cu1, nullptr, nullptr, nullptr, nullptr, cfg);

      // Second leg: tokens [split, T), seeded with leg 1's final state.
      const int64_t rest = T - split;
      Inputs leg2 = in;
      leg2.x.assign(in.x.begin() + static_cast<size_t>(split * H * P), in.x.end());
      leg2.dt.assign(in.dt.begin() + static_cast<size_t>(split * H), in.dt.end());
      leg2.B.assign(in.B.begin() + static_cast<size_t>(split * G * N), in.B.end());
      leg2.C.assign(in.C.begin() + static_cast<size_t>(split * G * N), in.C.end());
      const std::vector<int32_t> cu2{0, static_cast<int32_t>(rest)};
      const RunOut r2 = RunChunkScan(leg2, rest, H, P, G, N, cu2, nullptr, nullptr, nullptr,
                                     &r1.final_states, cfg);

      std::vector<float> joined = r1.y;
      joined.insert(joined.end(), r2.y.begin(), r2.y.end());
      INFO("chunk=" << chunk << " split=" << split);
      ExpectCloseF("y split vs whole", joined, ref.y, 5e-3, 5e-3);
      ExpectCloseF("final_states split vs whole", r2.final_states, ref.final_states, 5e-3, 5e-3);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) The optional arms: D as [H] and [H,P], the z silu gate, dt_bias and
// dt_softplus, and the dt_limit clamp — every branch of `_chunk_cumsum_fwd`
// (ssd_chunk_state.py:87-110) and of the `_chunk_scan_fwd` epilogue
// (ssd_chunk_scan.py:376-406).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan covers D, z, dt_bias, dt_softplus and dt_limit") {
  const int64_t T = 100, H = 8, P = 12, G = 2, N = 16, chunk = 32;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0xD00D42u);
  const std::vector<int32_t> cu{0, 40, static_cast<int32_t>(T)};

  std::mt19937 rng(7u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  std::vector<float> d_head_scalar(static_cast<size_t>(H));
  for (auto& v : d_head_scalar) v = nd(rng);
  std::vector<float> d_hdim(static_cast<size_t>(H * P));
  for (auto& v : d_hdim) v = nd(rng);
  std::vector<float> z(static_cast<size_t>(T * H * P));
  for (auto& v : z) v = nd(rng);
  std::vector<float> dt_bias(static_cast<size_t>(H));
  for (auto& v : dt_bias) v = ud(rng) - 4.0f;  // upstream's `torch.rand(...) - 4.0`

  SUBCASE("D as [H]") {
    const SeqRefOut ref = SequentialSsdRef(in, T, H, P, G, N, cu, &d_head_scalar, false, nullptr,
                                           nullptr, nullptr, {});
    RunCfg cfg;
    cfg.chunk_size = chunk;
    const RunOut got =
        RunChunkScan(in, T, H, P, G, N, cu, &d_head_scalar, nullptr, nullptr, nullptr, cfg);
    ExpectClose("y", got.y, ref.y, 5e-3, 5e-3);
  }
  SUBCASE("D as [H,P]") {
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, &d_hdim, true, nullptr, nullptr, nullptr, {});
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.d_has_hdim = true;
    const RunOut got =
        RunChunkScan(in, T, H, P, G, N, cu, &d_hdim, nullptr, nullptr, nullptr, cfg);
    ExpectClose("y", got.y, ref.y, 5e-3, 5e-3);
  }
  SUBCASE("z silu gate") {
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, &z, nullptr, nullptr, {});
    RunCfg cfg;
    cfg.chunk_size = chunk;
    const RunOut got = RunChunkScan(in, T, H, P, G, N, cu, nullptr, &z, nullptr, nullptr, cfg);
    ExpectClose("y", got.y, ref.y, 5e-3, 5e-3);
  }
  SUBCASE("dt_bias + dt_softplus") {
    RefCfg rc;
    rc.dt_softplus = true;
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, &dt_bias, nullptr, rc);
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.dt_softplus = true;
    const RunOut got =
        RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, &dt_bias, nullptr, cfg);
    ExpectClose("y", got.y, ref.y, 5e-3, 5e-3);
    ExpectClose("final_states", got.final_states, ref.final_states, 5e-3, 5e-3);
  }
  SUBCASE("dt_limit clamp") {
    RefCfg rc;
    rc.dt_softplus = true;
    rc.dt_min = 0.05;
    rc.dt_max = 0.10;
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, &dt_bias, nullptr, rc);
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.dt_softplus = true;
    cfg.dt_min = 0.05f;
    cfg.dt_max = 0.10f;
    const RunOut got =
        RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, &dt_bias, nullptr, cfg);
    ExpectClose("y", got.y, ref.y, 5e-3, 5e-3);
    // The clamp must actually BITE: without it the reference and the op would
    // agree trivially. Re-running the reference unclamped must NOT match.
    RefCfg unclamped;
    unclamped.dt_softplus = true;
    const SeqRefOut loose = SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr,
                                             &dt_bias, nullptr, unclamped);
    double max_diff = 0.0;
    for (size_t i = 0; i < loose.y.size(); ++i)
      max_diff = std::max(max_diff, std::abs(loose.y[i] - ref.y[i]));
    CHECK(max_diff > 1e-2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) The SSM state dtype is a SEPARATE knob from the activation dtype
// (`state_dtype`, ssd_combined.py:46,119,176). Neither may be derived from the
// other: a bf16 state with f32 activations, and an f32 state with bf16
// activations, must both run and must round exactly where the knob says.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan keeps the state dtype independent of the activation dtype") {
  const int64_t T = 64, H = 4, P = 8, G = 2, N = 16, chunk = 16;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0x51A7Eu);
  const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
  const SeqRefOut ref =
      SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});

  SUBCASE("f32 activations, bf16 state") {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.act_dtype = DType::kF32;
    cfg.state_dtype = DType::kBF16;
    const RunOut got =
        RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    // The ACTIVATION output stays f32-accurate; only the state rounds.
    ExpectClose("y", got.y, ref.y, 8e-3, 5e-3);
    ExpectClose("final_states (bf16 rounded)", got.final_states, ref.final_states, 5e-2, 5e-2);
    // ... and it really is bf16-rounded: every value must survive a bf16
    // round-trip unchanged.
    for (float v : got.final_states) CHECK(vt::BF16ToF32(vt::F32ToBF16(v)) == v);

    // `state_dtype` is not cosmetic on the RETURNED state: `_state_passing_fwd`
    // stores the inter-chunk state at that width and `_chunk_scan_fwd` READS IT
    // BACK (ssd_chunk_scan.py:249-250, :266-269), so a bf16 state must move `out`
    // too. Without that store-side rounding the two runs would be identical.
    RunCfg f32cfg = cfg;
    f32cfg.state_dtype = DType::kF32;
    const RunOut f32state =
        RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, f32cfg);
    double max_out_diff = 0.0;
    for (size_t i = 0; i < got.y.size(); ++i)
      max_out_diff =
          std::max(max_out_diff, std::abs(static_cast<double>(got.y[i]) - f32state.y[i]));
    INFO("max|out(bf16 state) - out(f32 state)| = " << max_out_diff);
    CHECK(max_out_diff > 1e-6);
  }
  SUBCASE("bf16 activations, f32 state") {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.act_dtype = DType::kBF16;
    cfg.state_dtype = DType::kF32;
    Inputs bin = in;
    RoundInputsTo(bin, DType::kBF16);
    const SeqRefOut bref =
        SequentialSsdRef(bin, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});
    const RunOut got =
        RunChunkScan(bin, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    ExpectClose("final_states", got.final_states, bref.final_states, 5e-2, 5e-2);
    // The f32 state must NOT have been silently narrowed to the activation
    // dtype: at least one value must fail a bf16 round-trip.
    bool any_finer_than_bf16 = false;
    for (float v : got.final_states)
      if (vt::BF16ToF32(vt::F32ToBF16(v)) != v) any_finer_than_bf16 = true;
    CHECK(any_finer_than_bf16);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (6b) THE RUNNING STATE DOES NOT COMPOUND ITS OWN ROUNDED STORE.
// `_state_passing_fwd` keeps `states` in f32 REGISTERS across the chunk loop and
// stores a `state_dtype` copy per chunk (ssd_state_passing.py:88-97). It never
// reads that store back into the recurrence — the rounding is observable in
// `out` (via `_chunk_scan_fwd`, which DOES read the stored copy) but must not
// accumulate chunk over chunk.
//
// Test (6) above pins the STORE side: dropping the rounding at the store makes
// `out(bf16 state) == out(f32 state)` and reds. It cannot see the other half,
// because a kernel that ALSO fed the rounded value back still stores rounded
// values. This is the exact, tolerance-free complement:
//
//   the f32 recurrence does not depend on `state_dtype`, so the bf16 run's
//   `final_states` must be BIT-FOR-BIT the bf16 rounding of the f32 run's.
//
// A kernel that compounded would drift from chunk 2 onwards and fail it. The
// sequences below are deliberately several chunks long — one chunk cannot
// compound anything.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan keeps the passed state in f32, never compounding its store") {
  const int64_t H = 4, P = 8, G = 2, N = 16, chunk = 16;
  // Two sequences, 6 and 4 logical chunks — continuous batching, and enough
  // chunks that a per-chunk rounding has somewhere to accumulate.
  const std::vector<int32_t> cu{0, 96, 160};
  const int64_t T = cu.back(), S = static_cast<int64_t>(cu.size()) - 1;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0x5A17Eu);
  const ChunkMeta meta = ComputeVarlenChunkMetadata(cu, chunk);
  REQUIRE(meta.seq_idx.size() > 2);

  for (bool with_init : {false, true}) {
    std::mt19937 rng(0x2468u);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> init(static_cast<size_t>(S * H * P * N));
    for (auto& v : init) v = vt::BF16ToF32(vt::F32ToBF16(nd(rng)));

    RunCfg f32cfg;
    f32cfg.chunk_size = chunk;
    f32cfg.state_dtype = DType::kF32;
    RunCfg bf16cfg = f32cfg;
    bf16cfg.state_dtype = DType::kBF16;
    const std::vector<float>* ip = with_init ? &init : nullptr;
    const RunOut a = RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, ip, f32cfg);
    const RunOut b = RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, ip, bf16cfg);

    REQUIRE(a.final_states.size() == b.final_states.size());
    size_t drifted = 0, rounding_bit = 0;
    double worst = 0.0;
    for (size_t i = 0; i < a.final_states.size(); ++i) {
      const float want = vt::BF16ToF32(vt::F32ToBF16(a.final_states[i]));
      if (want != a.final_states[i]) ++rounding_bit;  // the rounding is non-trivial here
      if (b.final_states[i] != want) {
        ++drifted;
        worst = std::max(worst, std::abs(static_cast<double>(b.final_states[i]) - want));
      }
    }
    INFO("with_init=" << with_init << " nchunks=" << meta.seq_idx.size() << ": " << drifted
                      << " of " << a.final_states.size()
                      << " final_states differ from bf16(f32-run), max|diff| = " << worst);
    // The comparison is only meaningful if bf16 actually rounds these values.
    REQUIRE(rounding_bit > 0);
    CHECK(drifted == 0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (7) REFUSALS. An unimplemented or ill-formed arm is refused with the missing
// piece named, never silently mis-computed (mamba2-ssd.md §7).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 chunk scan refuses the arms it does not implement") {
  const int64_t T = 32, H = 4, P = 8, G = 2, N = 8;
  const Inputs in = GenerateInputs(T, H, P, G, N, 3u);
  const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};

  SUBCASE("tp_world_size > 1 names extra_groups_for_head_shards") {
    Queue q = CpuQ();
    const ChunkMeta meta = ComputeVarlenChunkMetadata(cu, 16);
    std::vector<float> x = in.x, dt = in.dt, A = in.A, B = in.B, C = in.C;
    std::vector<float> out(static_cast<size_t>(T * H * P), 0.0f);
    std::vector<float> fs(static_cast<size_t>(H * P * N), 0.0f);
    std::vector<int32_t> cus = cu;
    ChunkMeta m = meta;
    Tensor xt = MakeT(x.data(), DType::kF32, {T, H, P});
    Tensor dtt = MakeT(dt.data(), DType::kF32, {T, H});
    Tensor At = MakeT(A.data(), DType::kF32, {H});
    Tensor Bt = MakeT(B.data(), DType::kF32, {T, G, N});
    Tensor Ct = MakeT(C.data(), DType::kF32, {T, G, N});
    Tensor outt = MakeT(out.data(), DType::kF32, {T, H, P});
    Tensor fst = MakeT(fs.data(), DType::kF32, {1, H, P, N});
    Tensor cust = MakeT(cus.data(), DType::kI32, {2});
    Tensor ccst = MakeT(m.cu_chunk_seqlens.data(), DType::kI32,
                        {static_cast<int64_t>(m.cu_chunk_seqlens.size())});
    Tensor lcit = MakeT(m.last_chunk_indices.data(), DType::kI32, {1});
    Tensor sit =
        MakeT(m.seq_idx.data(), DType::kI32, {static_cast<int64_t>(m.seq_idx.size())});
    Mamba2Args args;
    args.chunk_size = 16;
    args.tp_world_size = 2;
    std::string msg;
    try {
      vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, nullptr, nullptr, nullptr, nullptr,
                          cust, ccst, lcit, sit, args);
      FAIL("expected a refusal");
    } catch (const std::exception& e) {
      msg = e.what();
    }
    INFO(msg);
    CHECK(msg.find("extra_groups_for_head_shards") != std::string::npos);
  }

  SUBCASE("chunk_size must be a power of two") {
    RunCfg cfg;
    cfg.chunk_size = 24;
    CHECK_THROWS(RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg));
  }

  // ── the two PRECONDITIONS the intra-chunk clamp rests on ──────────────────
  // `exp(min(dA_i - dA_j, 0))` (ssd_chunk_scan.py:339-341) and
  // `exp(min(dA_last - dA_i, 0))` (ssd_chunk_state.py:283-285) are no-ops only
  // while `dA_cumsum` is non-increasing, which needs `A < 0` AND `dt >= 0`.
  // Upstream never has anything else — `A = -exp(A_log)` (mamba_mixer2.py:456)
  // and `dt_limit` defaults to `(0.0, inf)` (ssd_combined.py:180) — but neither
  // is derivable from the arguments, so the op has to state them. Fed `A > 0`
  // the clamp silently truncates a genuinely growing recurrence: the reviewer's
  // `A = +1.0` case returned `0.5 1 1.5 2 3.79744 ...` where the unclamped
  // recurrence gives `0.5 1.32436 2.6835 4.92435 ...`. An arm that is not
  // implemented is REFUSED, never silently mis-computed.
  SUBCASE("A must be negative (A = -exp(A_log))") {
    Inputs bad = in;
    bad.A[static_cast<size_t>(H - 1)] = 1.0f;  // the reviewer's escaping value
    RunCfg cfg;
    cfg.chunk_size = 16;
    bool threw = false;
    std::string msg;
    try {
      RunChunkScan(bad, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    INFO(msg);
    CHECK(threw);
    CHECK(msg.find("A_log") != std::string::npos);

    // Zero is refused too: `-exp(A_log)` is strictly negative for every finite
    // A_log, and dA_cumsum would be flat rather than decaying.
    Inputs zero = in;
    zero.A[0] = 0.0f;
    CHECK_THROWS(RunChunkScan(zero, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg));
  }

  SUBCASE("dt_limit must not admit a negative dt") {
    RunCfg cfg;
    cfg.chunk_size = 16;
    cfg.dt_min = -1.0f;  // `dt_limit=(0.0, inf)` upstream (ssd_combined.py:180)
    bool threw = false;
    std::string msg;
    try {
      RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    INFO(msg);
    CHECK(threw);
    CHECK(msg.find("dt_min") != std::string::npos);

    RunCfg inverted;
    inverted.chunk_size = 16;
    inverted.dt_min = 0.5f;
    inverted.dt_max = 0.1f;
    CHECK_THROWS(
        RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, inverted));
  }

  // ── empty sequences ────────────────────────────────────────────────────────
  // `compute_varlen_chunk_metadata` leaves `last_chunk_indices[b] == -1` for a
  // sequence with no tokens (mamba2_attn.py:56-74 never pushes a chunk for it),
  // and upstream's `varlen_states = states[last_chunk_indices]`
  // (ssd_combined.py:154) is then a torch NEGATIVE index: it silently returns
  // the last chunk of the WHOLE BATCH — some other sequence's state. vLLM never
  // schedules an empty sequence, so that is an indexing quirk, not a behaviour
  // to mirror; this op refuses instead of deviating quietly.
  SUBCASE("an empty sequence is refused, not silently given someone else's state") {
    const std::vector<int32_t> with_empty{0, 16, 16, static_cast<int32_t>(T)};
    const ChunkMeta m = ComputeVarlenChunkMetadata(with_empty, 16);
    REQUIRE(m.last_chunk_indices[1] == -1);
    RunCfg cfg;
    cfg.chunk_size = 16;
    bool threw = false;
    std::string msg;
    try {
      RunChunkScan(in, T, H, P, G, N, with_empty, nullptr, nullptr, nullptr, nullptr, cfg);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    INFO(msg);
    CHECK(threw);
    CHECK(msg.find("empty") != std::string::npos);
  }

  SUBCASE("seq_idx is per chunk, not per token") {
    Queue q = CpuQ();
    const ChunkMeta meta = ComputeVarlenChunkMetadata(cu, 16);
    std::vector<float> x = in.x, dt = in.dt, A = in.A, B = in.B, C = in.C;
    std::vector<float> out(static_cast<size_t>(T * H * P), 0.0f);
    std::vector<float> fs(static_cast<size_t>(H * P * N), 0.0f);
    std::vector<int32_t> cus = cu;
    ChunkMeta m = meta;
    std::vector<int32_t> per_token(static_cast<size_t>(T), 0);  // the WRONG length
    Tensor xt = MakeT(x.data(), DType::kF32, {T, H, P});
    Tensor dtt = MakeT(dt.data(), DType::kF32, {T, H});
    Tensor At = MakeT(A.data(), DType::kF32, {H});
    Tensor Bt = MakeT(B.data(), DType::kF32, {T, G, N});
    Tensor Ct = MakeT(C.data(), DType::kF32, {T, G, N});
    Tensor outt = MakeT(out.data(), DType::kF32, {T, H, P});
    Tensor fst = MakeT(fs.data(), DType::kF32, {1, H, P, N});
    Tensor cust = MakeT(cus.data(), DType::kI32, {2});
    Tensor ccst = MakeT(m.cu_chunk_seqlens.data(), DType::kI32,
                        {static_cast<int64_t>(m.cu_chunk_seqlens.size())});
    Tensor lcit = MakeT(m.last_chunk_indices.data(), DType::kI32, {1});
    Tensor sit = MakeT(per_token.data(), DType::kI32, {T});
    Mamba2Args args;
    args.chunk_size = 16;
    CHECK_THROWS(vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, nullptr, nullptr,
                                     nullptr, nullptr, cust, ccst, lcit, sit, args));
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// (8) THE CUDA ARM — .agents/specs/mamba2-ssd.md W2, issue #496.
//
// ─── THE DECLARED EQUIVALENCE CONTRACT ──────────────────────────────────────
// The CUDA kernels keep f32 accumulation THROUGHOUT and do NOT mirror the tile
// downcasts inside upstream's Triton dots (`b.to(x_ptr.dtype.element_ty)`
// ssd_chunk_state.py:283-285; `cb.to(...)` / `prev_states.to(C_ptr.dtype...)`
// ssd_chunk_scan.py:266-269, :359-363). Those casts are the input-precision
// requirement of `tl.dot`, i.e. of a tensor-core MMA — every one of those tiles is
// loaded `.to(tl.float32)` and computed in f32 right up to the MMA. Our kernels
// are scalar-FMA kernels with no MMA, so the bar here is NOT a downcast-derived
// tolerance. The memory format is unchanged: every load and store goes through the
// operand's own dtype, and the inter-chunk `passed` buffer is allocated at
// `state_dtype`, not at the host reference's f32 working width (§8.2 F9).
//
// Two gates, in order of authority:
//
//   G1 (PRIMARY, INDEPENDENT). The device outputs are held to the SAME
//   double-precision sequential reference (`SequentialSsdRef`) at the SAME
//   upstream-ported tolerances as the host arm. Every structural defect — a
//   dropped inter-chunk term, `states[c]` for `states[c-1]`, ignored
//   `initial_states`, a missing `D` skip — is an O(1) error and fails here,
//   against a reference the kernel was not written from.
//
//   G2 (DEVICE vs HOST, DERIVED). A BYTE COMPARE IS NOT REACHABLE, for TWO
//   named reasons and no others:
//
//     (a) LIBM. The two arms call different ones. CUDA's `expf` is documented to
//         <= 2 ulp and glibc's to <= 0.5, and they disagree in the last ulp on
//         some inputs, so `exp`/`log1p` alone put the two arms off each other by
//         ulps that the recurrence then amplifies.
//     (b) FMA CONTRACTION. Host C++ is pinned `-ffp-contract=off`
//         (CMakeLists.txt:41-56), so `a*b + c` keeps two roundings; nothing
//         passes `--fmad=false` to nvcc, so the device arm compiles at the
//         DEFAULT `--fmad=true` and every `acc += a*b` is a single-rounding
//         `fma` whose host twin is not. `-fmad=false` was weighed and rejected
//         at src/vt/cuda/cuda_mamba2_ssd.cuh — it is a per-TU flag on a header
//         included by a hot GDN TU — so the bound carries the term.
//
//   Summation ORDER is not a third source: the device kernels accumulate every
//   output element in ONE thread, over the same index range in the same
//   direction as the host arm. `DerivedRtol` below propagates (a) and (b), and
//   only those. Nothing in it was tuned, and the slack actually USED is reported
//   on every comparison — if it ever approached the bar, the bar would have
//   stopped being a statement about libm and contraction and the finding would
//   be a NEEDS_DECISION, not a wider tolerance.
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

// A GREEN TEST DOES NOT PROVE THE DEVICE RAN IT — and on THIS box it very nearly
// proves the opposite. GB10 is `integrated && pageable_memory_access`, so
// `Backend::UnifiedMemory()` is TRUE (cuda_backend.cu Registrar) and therefore
// `ReferenceTierEligible(kCUDA)` is TRUE. Absent a native kernel, `GetOp` does
// not throw: it installs the CPU HOST kernel as a `kReferenceProviderName`
// provider and runs THAT over the device pointers (op_provider.h, "portable
// reference tier"). Every numeric assertion below would then pass — the device
// arm would be gated by running the host arm twice, the exact false-green shape
// of [[absent-hook-looks-like-armed-instrument]] and
// [[gate-comparing-shared-helper-proves-consistency-not-correctness]].
//
// So every CUDA case asserts the SELECTED provider is native. These are EAGER
// dispatches rather than a captured graph, so the counters are genuinely
// populated ([[graph-replay-does-no-host-dispatch-counters-read-zero]]).
void RequireNativeCudaProvider(vt::OpId op, const std::string& what) {
  const vt::OpProviderStats st = vt::GetOpProviderStats(op, DeviceType::kCUDA);
  INFO(what << ": selected CUDA provider = "
            << (st.last_selected != nullptr ? st.last_selected : "<none>")
            << "; process-wide reference-tier hits = " << vt::GetReferenceTierHits());
  REQUIRE(st.last_selected != nullptr);
  CHECK(std::string(st.last_selected) != std::string(vt::kReferenceProviderName));
}

// f32 unit roundoff, and the bound the two arms are held to.
//
// A value that has run through a product of at most K decay factors and a
// length-K f32 sum of products carries, between the two arms, THREE terms:
//
//   libm         <= 2.5 ulp PER FACTOR — CUDA `expf` <= 2 ulp, glibc <= 0.5 —
//                through a product of at most K, i.e. <= 2.5*K*u;
//   summation    the standard (K-1)*u forward error of a length-K f32 sum,
//                which is what AMPLIFIES the libm difference in the inputs;
//   contraction  the device arm's `acc += a*b` is ONE nvcc-`fmad` rounding and
//                the host arm's is TWO (`-ffp-contract=off`, CMakeLists.txt:55),
//                so the host carries K product roundings the device does not:
//                <= K*u. See the FMA-contraction note in
//                src/vt/cuda/cuda_mamba2_ssd.cuh for why the flag is not simply
//                turned off instead.
//
// Total <= 2.5*K*u + (K-1)*u + K*u = 4.5*K*u - u. `5*(K + 2)*u` is that,
// rounded up to integers, and it covers the model for every K >= 0 because
// 5K + 10 >= 4.5K - 1 reduces to 0.5K + 11 >= 0.
//
// THE OLD `4*(K + 2)*u` DID NOT. It omitted the contraction term, and
// 4.5K - 1 <= 4K + 8 holds only for K <= 18 — while the driver-shapes case runs
// at K = T = 200. The constant moved because the DERIVATION gained a term the
// build actually emits, not because a run needed slack: at 4*(K+2) the worst of
// the 55 audited comparisons used 7.66% of budget, so at 5*(K+2) it uses 6.13%,
// and mutant M3's 962173% becomes 769738% — still caught by four orders of
// magnitude (§8.4).
//
// K is the case's own recurrence length; nothing here is fitted, and the slack
// actually USED is reported on every comparison so a bar that had stopped doing
// work would be visible rather than silently absorbing a defect.
constexpr double kUnitRoundoff = 5.9604644775390625e-08;  // 2^-24
double DerivedRtol(int64_t K) { return 5.0 * static_cast<double>(K + 2) * kUnitRoundoff; }

// atol is `rtol * max|host|` rather than 0: a bound proportional to |want| alone
// is vacuous for an element that is near zero through cancellation of O(max)
// terms, which this recurrence produces routinely.
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
  // its scope FAILS, so a claim that the used slack "is reported" would be false
  // on the green run that matters. This line is emitted unconditionally, which is
  // what makes the derived bar auditable rather than merely asserted.
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

// Owning device allocation, uploaded from host bytes (or left zero-sized).
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

// The CUDA twin of RunChunkScan, argument for argument.
//
// `lci_override` / `sidx_override` replace the metadata `ComputeVarlenChunkMetadata`
// derived, and exist for ONE caller: the clamp case below, which must reach the
// device kernels with metadata the host arm refuses. Every other caller passes
// nullptr and gets the derived metadata unchanged.
RunOut RunChunkScanCuda(Backend& gpu, const Inputs& in, int64_t T, int64_t H, int64_t P,
                        int64_t G, int64_t N, const std::vector<int32_t>& cu_seqlens,
                        const std::vector<float>* D, const std::vector<float>* z,
                        const std::vector<float>* dt_bias,
                        const std::vector<float>* initial_states, const RunCfg& cfg,
                        const std::vector<int32_t>* lci_override = nullptr,
                        const std::vector<int32_t>* sidx_override = nullptr) {
  Queue q = gpu.CreateQueue();
  const Device dev{DeviceType::kCUDA, 0};
  const int64_t S = static_cast<int64_t>(cu_seqlens.size()) - 1;
  ChunkMeta meta = ComputeVarlenChunkMetadata(cu_seqlens, cfg.chunk_size);
  const int64_t nchunks = static_cast<int64_t>(meta.seq_idx.size());
  if (lci_override != nullptr) meta.last_chunk_indices = *lci_override;
  if (sidx_override != nullptr) meta.seq_idx = *sidx_override;

  const std::vector<uint8_t> xb = Pack(in.x, cfg.act_dtype);
  const std::vector<uint8_t> dtb = Pack(in.dt, cfg.act_dtype);
  const std::vector<uint8_t> Bb = Pack(in.B, cfg.act_dtype);
  const std::vector<uint8_t> Cb = Pack(in.C, cfg.act_dtype);
  const size_t out_bytes = static_cast<size_t>(T * H * P) * vt::SizeOf(cfg.act_dtype);
  const size_t fs_bytes = static_cast<size_t>(S * H * P * N) * vt::SizeOf(cfg.state_dtype);
  std::vector<int32_t> cus = cu_seqlens;

  DBuf dx(gpu, q, xb.data(), xb.size());
  DBuf ddt(gpu, q, dtb.data(), dtb.size());
  DBuf dA(gpu, q, in.A.data(), in.A.size() * sizeof(float));
  DBuf dB(gpu, q, Bb.data(), Bb.size());
  DBuf dC(gpu, q, Cb.data(), Cb.size());
  DBuf dout(gpu, q, nullptr, out_bytes);
  DBuf dfs(gpu, q, nullptr, fs_bytes);
  DBuf dcus(gpu, q, cus.data(), cus.size() * sizeof(int32_t));
  DBuf dccs(gpu, q, meta.cu_chunk_seqlens.data(), meta.cu_chunk_seqlens.size() * sizeof(int32_t));
  DBuf dlci(gpu, q, meta.last_chunk_indices.data(),
            meta.last_chunk_indices.size() * sizeof(int32_t));
  DBuf dsi(gpu, q, meta.seq_idx.data(), meta.seq_idx.size() * sizeof(int32_t));

  Tensor xt = MakeTDev(dx.get(), cfg.act_dtype, dev, {T, H, P});
  Tensor dtt = MakeTDev(ddt.get(), cfg.act_dtype, dev, {T, H});
  Tensor At = MakeTDev(dA.get(), DType::kF32, dev, {H});
  Tensor Bt = MakeTDev(dB.get(), cfg.act_dtype, dev, {T, G, N});
  Tensor Ct = MakeTDev(dC.get(), cfg.act_dtype, dev, {T, G, N});
  Tensor outt = MakeTDev(dout.get(), cfg.act_dtype, dev, {T, H, P});
  Tensor fst = MakeTDev(dfs.get(), cfg.state_dtype, dev, {S, H, P, N});
  Tensor cust = MakeTDev(dcus.get(), DType::kI32, dev, {S + 1});
  Tensor ccst = MakeTDev(dccs.get(), DType::kI32, dev, {nchunks + 1});
  Tensor lcit = MakeTDev(dlci.get(), DType::kI32, dev, {S});
  Tensor sit = MakeTDev(dsi.get(), DType::kI32, dev, {nchunks});

  std::vector<float> Dc;
  std::unique_ptr<DBuf> dD;
  Tensor Dt;
  if (D != nullptr) {
    Dc = *D;
    dD = std::make_unique<DBuf>(gpu, q, Dc.data(), Dc.size() * sizeof(float));
    Dt = cfg.d_has_hdim ? MakeTDev(dD->get(), DType::kF32, dev, {H, P})
                        : MakeTDev(dD->get(), DType::kF32, dev, {H});
  }
  std::vector<uint8_t> zb;
  std::unique_ptr<DBuf> dz;
  Tensor zt;
  if (z != nullptr) {
    zb = Pack(*z, cfg.act_dtype);
    dz = std::make_unique<DBuf>(gpu, q, zb.data(), zb.size());
    zt = MakeTDev(dz->get(), cfg.act_dtype, dev, {T, H, P});
  }
  std::vector<float> dbc;
  std::unique_ptr<DBuf> ddb;
  Tensor dbt;
  if (dt_bias != nullptr) {
    dbc = *dt_bias;
    ddb = std::make_unique<DBuf>(gpu, q, dbc.data(), dbc.size() * sizeof(float));
    dbt = MakeTDev(ddb->get(), DType::kF32, dev, {H});
  }
  std::vector<uint8_t> isb;
  std::unique_ptr<DBuf> dis;
  Tensor ist;
  if (initial_states != nullptr) {
    isb = Pack(*initial_states, cfg.state_dtype);
    dis = std::make_unique<DBuf>(gpu, q, isb.data(), isb.size());
    ist = MakeTDev(dis->get(), cfg.state_dtype, dev, {S, H, P, N});
  }

  Mamba2Args args;
  args.chunk_size = cfg.chunk_size;
  args.dt_softplus = cfg.dt_softplus;
  args.dt_min = cfg.dt_min;
  args.dt_max = cfg.dt_max;

  vt::Mamba2ChunkScan(q, outt, fst, xt, dtt, At, Bt, Ct, D != nullptr ? &Dt : nullptr,
                      z != nullptr ? &zt : nullptr, dt_bias != nullptr ? &dbt : nullptr,
                      initial_states != nullptr ? &ist : nullptr, cust, ccst, lcit, sit, args);

  std::vector<uint8_t> outb(out_bytes), fsb(fs_bytes);
  dout.Download(q, outb.data());
  dfs.Download(q, fsb.data());
  gpu.Synchronize(q);

  RunOut r;
  r.y = Unpack(outb, static_cast<size_t>(T * H * P), cfg.act_dtype);
  r.final_states = Unpack(fsb, static_cast<size_t>(S * H * P * N), cfg.state_dtype);
  gpu.DestroyQueue(q);
  return r;
}

}  // namespace

// G1 + G2 on the shapes the row exists for: Nemotron-3.5-Lightning-30B-A3B's
// mamba layer — nheads 64, headdim 64, dstate 128, ngroups 8, chunk_size 128,
// mamba_ssm_cache_dtype float32 (mamba2-ssd.md §1.4). T is 200 so `nchunks == 2`
// and the arm actually exercises inter-chunk state passing — the failure shape of
// [[h3-video-decode-temporal-and-tiling-compose]], and of §8.2 F6.
TEST_CASE("mamba2 chunk scan CUDA arm on the driver shapes") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  const int64_t T = 200, H = 64, P = 64, G = 8, N = 128, chunk = 128;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0x4E33Au);
  const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
  REQUIRE(ComputeVarlenChunkMetadata(cu, chunk).seq_idx.size() > 1);

  std::mt19937 rng(0x77u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> D(static_cast<size_t>(H));
  for (auto& v : D) v = nd(rng);

  RunCfg cfg;
  cfg.chunk_size = chunk;
  const SeqRefOut ref =
      SequentialSsdRef(in, T, H, P, G, N, cu, &D, false, nullptr, nullptr, nullptr, {});
  const RunOut host = RunChunkScan(in, T, H, P, G, N, cu, &D, nullptr, nullptr, nullptr, cfg);
  const RunOut dev =
      RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, &D, nullptr, nullptr, nullptr, cfg);
  RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "driver shapes");

  // G1 — the device arm against the INDEPENDENT double reference, at upstream's
  // own f32 threshold for the chunked single-example test (atol 8e-3 / rtol 5e-3,
  // test_mamba_ssm_ssd.py:210-213). THE HOST ARM IS HELD TO THE SAME NUMBERS on
  // the same inputs, so a failure separates cleanly: device-only means a device
  // defect, both means the cited threshold does not cover this shape and the
  // finding is a NEEDS_DECISION rather than a wider tolerance.
  ExpectClose("host y vs sequential double", host.y, ref.y, 8e-3, 5e-3);
  ExpectClose("host final_states vs sequential double", host.final_states, ref.final_states,
              8e-3, 5e-3);
  ExpectClose("device y vs sequential double", dev.y, ref.y, 8e-3, 5e-3);
  ExpectClose("device final_states vs sequential double", dev.final_states, ref.final_states,
              8e-3, 5e-3);
  // G2 — device vs host, at the derived libm bound.
  ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, T);
  ExpectDeviceMatchesHost("final_states device vs host", dev.final_states, host.final_states, T);
}

// The structural properties the chunked factorisation lives on, on device:
// chunk-size invariance, sequence boundaries INSIDE a physical chunk, and
// `initial_states`. Small shapes so the sweep stays cheap; the driver shapes are
// covered above.
TEST_CASE("mamba2 chunk scan CUDA arm holds the chunked-factorisation properties") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }

  SUBCASE("invariant to chunk_size") {
    const int64_t T = 300, H = 8, P = 16, G = 2, N = 32;
    const Inputs in = GenerateInputs(T, H, P, G, N, 0xC0FFEEu);
    const std::vector<int32_t> cu{0, static_cast<int32_t>(T)};
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});
    std::vector<float> first_y, first_state;
    for (int64_t chunk : {8, 16, 32, 64, 128}) {
      RunCfg cfg;
      cfg.chunk_size = chunk;
      const int64_t nchunks =
          static_cast<int64_t>(ComputeVarlenChunkMetadata(cu, chunk).seq_idx.size());
      INFO("chunk_size=" << chunk << " nchunks=" << nchunks);
      REQUIRE(nchunks > 1);
      const RunOut dev =
          RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
      RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "chunk_size invariance");
      const RunOut host =
          RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
      ExpectClose("device y vs sequential", dev.y, ref.y, 1e-2, 5e-3);
      ExpectClose("device final_states vs sequential", dev.final_states, ref.final_states, 1e-2,
                  5e-3);
      ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, T);
      ExpectDeviceMatchesHost("final_states device vs host", dev.final_states, host.final_states,
                              T);
      if (first_y.empty()) {
        first_y = dev.y;
        first_state = dev.final_states;
      } else {
        ExpectCloseF("device y vs chunk_size=8", dev.y, first_y, 1e-2, 5e-3);
        ExpectCloseF("device final_states vs chunk_size=8", dev.final_states, first_state, 1e-2,
                     5e-3);
      }
    }
  }

  SUBCASE("continuous batches, with and without initial_states") {
    struct Case {
      std::vector<int32_t> lens;
      int64_t chunk;
    };
    const std::vector<Case> cases{
        {{64, 32}, 8},    {{4, 4, 4, 4}, 8}, {{5, 30, 1, 2}, 256},
        {{138, 225}, 128}, {{270, 88}, 8},
    };
    const int64_t H = 8, P = 16, G = 2, N = 16;
    for (const Case& c : cases) {
      std::vector<int32_t> cu{0};
      for (int32_t l : c.lens) cu.push_back(cu.back() + l);
      const int64_t T = cu.back();
      const int64_t S = static_cast<int64_t>(c.lens.size());
      const int64_t maxlen = *std::max_element(c.lens.begin(), c.lens.end());
      const double atol = maxlen > 256 ? 1e-2 : 5e-3;
      const Inputs in = GenerateInputs(T, H, P, G, N, 0xBEEF01u + static_cast<uint32_t>(c.chunk));
      INFO("chunk=" << c.chunk << " nseq=" << S << " T=" << T);
      RunCfg cfg;
      cfg.chunk_size = c.chunk;

      // (a) fresh sequences — the `seq_idx[c] != seq_idx[c-1]` branch must take
      //     ZEROS as the previous state (ssd_chunk_scan.py:271-274).
      {
        const SeqRefOut ref = SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr,
                                               nullptr, nullptr, {});
        const RunOut dev = RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr,
                                            nullptr, nullptr, cfg);
        RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "continuous batch (fresh)");
        const RunOut host =
            RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
        ExpectClose("device y (fresh)", dev.y, ref.y, atol, 5e-3);
        ExpectClose("device final_states (fresh)", dev.final_states, ref.final_states, atol,
                    5e-3);
        ExpectDeviceMatchesHost("y (fresh) device vs host", dev.y, host.y, maxlen);
        ExpectDeviceMatchesHost("final_states (fresh) device vs host", dev.final_states,
                                host.final_states, maxlen);
      }
      // (b) with initial_states — the same branch must instead take
      //     initial_states[seq_idx[c]] (ssd_chunk_scan.py:236-243).
      {
        std::mt19937 rng(1234u);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        std::vector<float> init(static_cast<size_t>(S * H * P * N));
        for (auto& v : init) v = nd(rng);
        const std::vector<double> initd(init.begin(), init.end());
        const SeqRefOut ref = SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr,
                                               nullptr, &initd, {});
        const RunOut dev =
            RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, &init, cfg);
        const RunOut host =
            RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, &init, cfg);
        ExpectClose("device y (init states)", dev.y, ref.y, atol, 5e-3);
        ExpectClose("device final_states (init states)", dev.final_states, ref.final_states, atol,
                    5e-3);
        ExpectDeviceMatchesHost("y (init) device vs host", dev.y, host.y, maxlen);
        ExpectDeviceMatchesHost("final_states (init) device vs host", dev.final_states,
                                host.final_states, maxlen);
      }
    }
  }
}

// The optional arms and the dtype knobs, on device: D as [H] and [H,P], the z
// silu gate, dt_bias + dt_softplus, the dt_limit clamp, a bf16 activation stream
// and a bf16 SSM state.
TEST_CASE("mamba2 chunk scan CUDA arm covers the optional arms and the dtype knobs") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  const int64_t T = 100, H = 8, P = 12, G = 2, N = 16, chunk = 32;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0xD00D42u);
  const std::vector<int32_t> cu{0, 40, static_cast<int32_t>(T)};
  const int64_t maxlen = 60;  // the longer of the two sequences

  std::mt19937 rng(7u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  std::vector<float> d_head_scalar(static_cast<size_t>(H));
  for (auto& v : d_head_scalar) v = nd(rng);
  std::vector<float> d_hdim(static_cast<size_t>(H * P));
  for (auto& v : d_hdim) v = nd(rng);
  std::vector<float> z(static_cast<size_t>(T * H * P));
  for (auto& v : z) v = nd(rng);
  std::vector<float> dt_bias(static_cast<size_t>(H));
  for (auto& v : dt_bias) v = ud(rng) - 4.0f;

  SUBCASE("D as [H]") {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    const SeqRefOut ref = SequentialSsdRef(in, T, H, P, G, N, cu, &d_head_scalar, false, nullptr,
                                           nullptr, nullptr, {});
    const RunOut dev = RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, &d_head_scalar, nullptr,
                                        nullptr, nullptr, cfg);
    RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "D as [H]");
    const RunOut host =
        RunChunkScan(in, T, H, P, G, N, cu, &d_head_scalar, nullptr, nullptr, nullptr, cfg);
    ExpectClose("device y", dev.y, ref.y, 5e-3, 5e-3);
    ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, maxlen);
  }
  SUBCASE("D as [H,P]") {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.d_has_hdim = true;
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, &d_hdim, true, nullptr, nullptr, nullptr, {});
    const RunOut dev =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, &d_hdim, nullptr, nullptr, nullptr, cfg);
    const RunOut host =
        RunChunkScan(in, T, H, P, G, N, cu, &d_hdim, nullptr, nullptr, nullptr, cfg);
    ExpectClose("device y", dev.y, ref.y, 5e-3, 5e-3);
    ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, maxlen);
  }
  SUBCASE("z silu gate") {
    RunCfg cfg;
    cfg.chunk_size = chunk;
    const SeqRefOut ref =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, &z, nullptr, nullptr, {});
    const RunOut dev =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, &z, nullptr, nullptr, cfg);
    const RunOut host = RunChunkScan(in, T, H, P, G, N, cu, nullptr, &z, nullptr, nullptr, cfg);
    ExpectClose("device y", dev.y, ref.y, 5e-3, 5e-3);
    ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, maxlen);
  }
  SUBCASE("dt_bias + dt_softplus, then the dt_limit clamp") {
    RefCfg rc;
    rc.dt_softplus = true;
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.dt_softplus = true;
    {
      const SeqRefOut ref =
          SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, &dt_bias, nullptr, rc);
      const RunOut dev =
          RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, &dt_bias, nullptr, cfg);
      const RunOut host =
          RunChunkScan(in, T, H, P, G, N, cu, nullptr, nullptr, &dt_bias, nullptr, cfg);
      ExpectClose("device y", dev.y, ref.y, 5e-3, 5e-3);
      ExpectClose("device final_states", dev.final_states, ref.final_states, 5e-3, 5e-3);
      ExpectDeviceMatchesHost("y device vs host", dev.y, host.y, maxlen);
    }
    RefCfg clamped = rc;
    clamped.dt_min = 0.05;
    clamped.dt_max = 0.10;
    RunCfg ccfg = cfg;
    ccfg.dt_min = 0.05f;
    ccfg.dt_max = 0.10f;
    const SeqRefOut cref = SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr,
                                            &dt_bias, nullptr, clamped);
    const RunOut cdev =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, &dt_bias, nullptr, ccfg);
    ExpectClose("device y (clamped)", cdev.y, cref.y, 5e-3, 5e-3);
    // The clamp must actually BITE, or the comparison is trivial.
    const SeqRefOut loose =
        SequentialSsdRef(in, T, H, P, G, N, cu, nullptr, false, nullptr, &dt_bias, nullptr, rc);
    double max_diff = 0.0;
    for (size_t i = 0; i < loose.y.size(); ++i)
      max_diff = std::max(max_diff, std::abs(loose.y[i] - cref.y[i]));
    CHECK(max_diff > 1e-2);
  }
  SUBCASE("bf16 activations") {
    Inputs bin = in;
    RoundInputsTo(bin, DType::kBF16);
    RunCfg cfg;
    cfg.chunk_size = chunk;
    cfg.act_dtype = DType::kBF16;
    const SeqRefOut bref =
        SequentialSsdRef(bin, T, H, P, G, N, cu, nullptr, false, nullptr, nullptr, nullptr, {});
    const RunOut dev =
        RunChunkScanCuda(*gpu, bin, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    ExpectClose("device y bf16", dev.y, bref.y, 5e-2, 5e-2);
  }
  // `state_dtype` is a SEPARATE knob (ssd_combined.py:46,119,176), and it moves
  // `out` as well as `final_states` because `_chunk_scan_fwd` reads the stored
  // copy back (:249-250, :266-269). The device arm allocates its inter-chunk
  // buffer at that width — NOT at the host reference's f32 working width (§8.2 F9)
  // — so the two arms must agree on BOTH outputs.
  SUBCASE("bf16 SSM state with f32 activations") {
    const int64_t T2 = 96, H2 = 4, P2 = 8, G2 = 2, N2 = 16, chunk2 = 16;
    const Inputs in2 = GenerateInputs(T2, H2, P2, G2, N2, 0x51A7Eu);
    const std::vector<int32_t> cu2{0, static_cast<int32_t>(T2)};
    RunCfg cfg;
    cfg.chunk_size = chunk2;
    cfg.state_dtype = DType::kBF16;
    const RunOut dev = RunChunkScanCuda(*gpu, in2, T2, H2, P2, G2, N2, cu2, nullptr, nullptr,
                                        nullptr, nullptr, cfg);
    RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "bf16 SSM state");
    const RunOut host =
        RunChunkScan(in2, T2, H2, P2, G2, N2, cu2, nullptr, nullptr, nullptr, nullptr, cfg);
    for (float v : dev.final_states) CHECK(vt::BF16ToF32(vt::F32ToBF16(v)) == v);
    ExpectDeviceMatchesHost("y (bf16 state) device vs host", dev.y, host.y, T2);
    ExpectDeviceMatchesHost("final_states (bf16 state) device vs host", dev.final_states,
                            host.final_states, T2);
    // ... and the bf16 state really does move `out` on the device arm too.
    RunCfg f32cfg = cfg;
    f32cfg.state_dtype = DType::kF32;
    const RunOut f32dev = RunChunkScanCuda(*gpu, in2, T2, H2, P2, G2, N2, cu2, nullptr, nullptr,
                                           nullptr, nullptr, f32cfg);
    double max_out_diff = 0.0;
    for (size_t i = 0; i < dev.y.size(); ++i)
      max_out_diff =
          std::max(max_out_diff, std::abs(static_cast<double>(dev.y[i]) - f32dev.y[i]));
    INFO("device max|out(bf16 state) - out(f32 state)| = " << max_out_diff);
    CHECK(max_out_diff > 1e-6);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// OUT-OF-CONTRACT METADATA, DEVICE ONLY.
//
// The shared validator checks metadata SHAPE / DTYPE / DEVICE only
// (`CheckI32Meta`, ops.cpp); every VALUE check lives in the HOST kernel, which
// reads the tensors (cpu_ops.cpp). The device arm therefore runs with NONE of
// them. src/vt/cuda/cuda_mamba2_ssd.cuh enumerates all six and says which are
// memory-unsafe; this case covers the TWO that are both memory-unsafe and
// bounded by a register-local clamp:
//   * `last_chunk_indices[b] >= nchunks` makes the `passed` store run past its
//     `cudaMallocAsync` allocation, and
//   * a `seq_idx[c]` outside `[0,S)` reads `initial_states` out of bounds, with
//     `seq_idx[0] < 0` additionally indexing `passed` at chunk -1.
// It PINS the clamped behaviour so it is a test, not a sentence in a header.
// The `cu_chunk_seqlens` checks are NOT clamped and are NOT covered here; the
// header states that as an open memory-safety gap rather than hiding it.
//
// THIS CASE HAS NO HOST TWIN, deliberately: the host kernel REFUSES both inputs,
// which is exactly why the device arm needed the clamp. Each clamp is pinned
// against an IN-CONTRACT reference run whose result the clamp is defined to
// reproduce, so the assertions are EXACT — a tolerance here would be wide enough
// to hide the defect it is meant to catch.
//
// What this case does NOT establish is memory safety itself. An out-of-bounds
// write into a `cudaMallocAsync` pool very often does not fault, so a green run
// is necessary and not sufficient; `compute-sanitizer memcheck` on this case is
// what proves it, and is recorded as owed in mamba2-ssd.md §8.5.
TEST_CASE("mamba2 chunk scan CUDA arm clamps out-of-contract metadata in registers") {
  Backend* gpu = MaybeCuda();
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (CPU-only build/box)");
    return;
  }
  // Four one-chunk sequences, so `nchunks == S == 4` and every chunk opens a new
  // sequence. That makes the in-contract `seq_idx` {0,1,2,3} and the in-contract
  // `last_chunk_indices` {0,1,2,3}.
  const int64_t H = 4, P = 8, G = 2, N = 16, chunk = 16;
  const std::vector<int32_t> cu{0, 16, 32, 48, 64};
  const int64_t T = cu.back(), S = static_cast<int64_t>(cu.size()) - 1;
  const Inputs in = GenerateInputs(T, H, P, G, N, 0xC1A47u);
  const ChunkMeta meta = ComputeVarlenChunkMetadata(cu, chunk);
  const int64_t nchunks = static_cast<int64_t>(meta.seq_idx.size());
  REQUIRE(nchunks == S);
  // A braced init-list cannot appear inside a doctest macro — the preprocessor
  // splits it on the commas — so the expectation is named first.
  const std::vector<int32_t> in_contract{0, 1, 2, 3};
  REQUIRE(meta.last_chunk_indices == in_contract);
  REQUIRE(meta.seq_idx == in_contract);

  std::mt19937 rng(0xB0B0u);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::vector<float> init(static_cast<size_t>(S * H * P * N));
  for (auto& v : init) v = nd(rng);

  RunCfg cfg;
  cfg.chunk_size = chunk;

  auto max_abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return worst;
  };
  auto count_differing = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i]) ++n;
    return n;
  };

  SUBCASE("last_chunk_indices past the end stops the chunk loop at nchunks") {
    // In contract, the last sequence's `lci` is `nchunks - 1`. The clamp is
    // defined to make anything >= nchunks behave as nchunks - 1, so these two
    // runs must agree BIT FOR BIT — in `final_states` as much as in `y`, since
    // the clamp is inside the state-passing kernel that produces both.
    const RunOut ref =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, &init, cfg);
    RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "lci clamp reference");
    for (int32_t past : {static_cast<int32_t>(nchunks), static_cast<int32_t>(nchunks + 41)}) {
      std::vector<int32_t> lci = meta.last_chunk_indices;
      lci.back() = past;
      const RunOut got = RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr,
                                          &init, cfg, &lci, nullptr);
      INFO("last_chunk_indices.back() = " << past << " (nchunks = " << nchunks << ")");
      CHECK(count_differing(got.y, ref.y) == 0);
      CHECK(count_differing(got.final_states, ref.final_states) == 0);
    }
  }

  SUBCASE("a seq_idx outside [0,S) opens the chunk with a ZERO previous state") {
    // The reference is an IN-CONTRACT run with NO initial states: `seq_idx`
    // {0,1,2,3} makes every chunk open a new sequence, and with `initial_states
    // == nullptr` every one of them therefore opens from zero. That is exactly
    // what the clamp is defined to do for an out-of-range index, so `y` must
    // match bit for bit. `final_states` is NOT compared: the state-passing
    // kernel reads `initial_states` by `b`, not by `seq_idx`, so it legitimately
    // differs between a run that was given initial states and one that was not.
    const RunOut zero_ref =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, nullptr, cfg);
    RequireNativeCudaProvider(vt::OpId::kMamba2ChunkScan, "seq_idx clamp reference");
    // ... and the pin is not vacuous: with the SAME metadata in contract, the
    // initial states genuinely move `y`, so "matches the zero-init run" is a
    // real statement rather than two identical computations
    // ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]).
    const RunOut init_ref =
        RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr, &init, cfg);
    INFO("max|y(initial_states) - y(none)| = " << max_abs_diff(init_ref.y, zero_ref.y));
    REQUIRE(max_abs_diff(init_ref.y, zero_ref.y) > 1e-6);

    // Over the top: every index >= S, which unclamped indexes `initial_states`
    // out of bounds. Under the bottom: every index negative, which unclamped
    // ALSO makes `si == si_prev` at c == 0 and indexes `passed` at chunk -1.
    const std::vector<std::vector<int32_t>> bad{
        {static_cast<int32_t>(S), static_cast<int32_t>(S + 1), static_cast<int32_t>(S + 2),
         static_cast<int32_t>(S + 900)},
        {-1, -1, -1, -1},
    };
    for (const std::vector<int32_t>& sidx : bad) {
      REQUIRE(static_cast<int64_t>(sidx.size()) == nchunks);
      const RunOut got = RunChunkScanCuda(*gpu, in, T, H, P, G, N, cu, nullptr, nullptr, nullptr,
                                          &init, cfg, nullptr, &sidx);
      INFO("seq_idx = {" << sidx[0] << ", " << sidx[1] << ", " << sidx[2] << ", " << sidx[3]
                         << "}, S = " << S);
      CHECK(count_differing(got.y, zero_ref.y) == 0);
    }
  }
}

#endif  // VLLM_CPP_CUDA
