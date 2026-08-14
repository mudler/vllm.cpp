// Nemotron-H W4 — the FORWARD gate.
// Spec: .agents/specs/nemotron-h-model.md §4 W4. Issue #517.
//
// ─── WHY THIS GATE LOOKS LIKE THIS ──────────────────────────────────────────
//
// `porting-a-model.md` §3: "Numeric bounds, not just token equality. A mechanism
// can be missing while the argmax is unchanged." So every block is compared
// ELEMENTWISE against a reference written independently in `double` from the
// upstream formulas — never against a second call of the code under test, and
// never through a helper both arms share
// ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]).
//
// The references below are transcriptions of, all @ pin 5559679229bc961848b121ccdeaa8fa5d79bec98:
//   models/nemotron_h.py:126-256 (MoE), :473-486 (attention), :86-123 (MLP),
//   :301-313 / :604-660 (the single-branch residual stream),
//   layers/mamba/mamba_mixer2.py:100-149 (Mixer2RMSNormGated), :548-586 +
//   :687-696 + :830-891 (the mixer), and the SSD recurrence upstream's own CPU
//   kernel runs token-by-token (csrc/cpu/mamba_kernels.hpp:279-382).
//
// FOUR TRAPS THIS ARCHITECTURE HAS ALREADY SPRUNG, each with its own case:
//  1. `routed_scaling_factor` is applied to the OUTPUT, not folded into the
//     router logits or weights (nemotron_h.py:234, layer.py:291-300). Laguna
//     legitimately folds the same factor (laguna_ops.h:48) because it passes no
//     shared expert; here the fold is a DIFFERENT answer on every token whose
//     shared term is non-zero. Gated with a shared expert present.
//  2. The router runs in f32 (`force_fp32_compute=True`, :150-156) — mirrored,
//     not inherited from the model dtype.
//  3. The SSM state dtype is resolved INDEPENDENTLY of the conv dtype. It is
//     UNOBSERVABLE with fresh state (the scan computes in f32 and only STORES
//     `final_states` at that dtype), so the two-leg case is what makes it
//     observable — the same thing the W6 paged decode does every step.
//  4. Only 6 of 52 layers hold attention, so an attention-side defect is diluted
//     across 46 non-attention layers. The whole-stack cases therefore run a
//     LONG-prompt arm, not only a short one, and the attention block is gated
//     directly rather than only through the stack.
//
// Plus the one this port found: NemotronH attention has NO RoPE AT ALL —
// `models/nemotron_h.py` contains zero occurrences of `rope`/`rotary`. The
// released config.json still ships `rope_theta` and `partial_rotary_factor`, and
// applying them changes no SHAPE, so nothing but a numeric gate can see it.
//
// TOLERANCES: `doctest::Approx` is deliberately NOT used — its `scale` defaults
// to 1.0, flooring every comparison at ~1.19e-5 absolute, so `Approx(1e-6)`
// accepts 1e-5, 1e-7 AND 0 ([[doctest-approx-scale-term-floor]]). Every check
// goes through `ExpectClose`, stated explicitly, which reports the worst element.
// Every label is a `std::string`: doctest 2.5.2 prints a `const char*` VARIABLE
// as `1`.
//
// EVERY ARM IS SWEPT AT F32 AS WELL AS BF16. A bf16 store absorbs real
// reduction-order defects ([[bf16-store-absorbs-reduction-order-defects]]), and
// bf16 is the released checkpoint's model dtype, so both have to run.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vllm::NemotronHAttentionWeights;
using vllm::NemotronHBlock;
using vllm::NemotronHExpertWeights;
using vllm::NemotronHHostWeights;
using vllm::NemotronHLayerWeights;
using vllm::NemotronHMambaState;
using vllm::NemotronHMambaWeights;
using vllm::NemotronHMlpWeights;
using vllm::NemotronHMoeWeights;
using vllm::NemotronHOwned;
using vllm::NemotronHParams;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

// ─── comparison ─────────────────────────────────────────────────────────────
//
// THE BAND IS TIED TO THE REFERENCE'S OWN SCALE, and every comparison CERTIFIES
// ITSELF. Both halves of that are repairs to the gate this file shipped with,
// and the reason is measured rather than stylistic.
//
// As inherited, the tolerance was a flat pair — `{2e-4, 2e-4}` at f32 and
// `{6e-2, 6e-2}` at bf16 — applied to references whose own magnitude nobody had
// looked at. Measured against each comparison's actual scale, the bf16 arm was
// judging a mamba2 mixer whose mean |reference| is 1.69e-2 with an absolute band
// of 6e-2 (3.55x the signal), and an attention block whose mean |reference| is
// 3.55e-4 with the same 6e-2 (169x). A block that returned ALL ZEROS passed
// those arms. bf16 is the RELEASED checkpoint's model dtype, so the arm that
// mattered most was the one that could not fail.
//
// So: `band = rel * max|want| + rel * |want[i]|`, which cannot be loose relative
// to a signal it is derived from, and `ExpectCloseRel` REQUIREs that the same
// band REJECTS an all-zeros answer before it accepts the real one. A gate that
// cannot fail is now itself a test failure — the property is asserted at run
// time rather than left to the next reader to re-derive.
// [[gate-comparing-shared-helper-proves-consistency-not-correctness]]

double MaxAbs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

// True when the two series differ somewhere by more than the band. Used to prove
// a gate is not vacuous — that the thing it distinguishes really is different.
bool AnyDiffers(const std::vector<float>& a, const std::vector<double>& b, double atol,
                double rtol) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(static_cast<double>(a[i]) - b[i]) > atol + rtol * std::abs(b[i])) {
      return true;
    }
  }
  return false;
}

// The worst RELATIVE separation between two series, normalised by the reference's
// own peak. Absolute separations are meaningless across blocks whose outputs span
// four orders of magnitude here, and using one is what made the inherited
// mis-port guards vacuous: the bf16-SSM-state defect separates by 2.4% of the
// signal, which is 1.4e-6 in absolute terms against a guard band of 2e-4.
double RelSeparation(const std::vector<float>& a, const std::vector<double>& b) {
  REQUIRE(a.size() == b.size());
  const double scale = MaxAbs(b);
  REQUIRE(scale > 0.0);
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
  }
  return worst / scale;
}

// torch.testing.assert_close arithmetic over a band derived from `want`'s own
// peak, plus the self-certification described above.
void ExpectCloseRel(const std::string& what, const std::vector<float>& got,
                    const std::vector<double>& want, double rel) {
  REQUIRE(got.size() == want.size());
  REQUIRE(!got.empty());
  const double scale = MaxAbs(want);
  // A reference that is identically zero cannot gate anything.
  REQUIRE(scale > 0.0);
  const double atol = rel * scale;

  // SELF-CERTIFICATION: this exact band must REJECT an all-zeros answer. If it
  // does not, the comparison below proves nothing and the failure is the gate's,
  // not the code's.
  const std::vector<float> zeros(got.size(), 0.0f);
  INFO(what << ": non-vacuity — band atol=" << atol << " rel=" << rel
            << " against max|want|=" << scale);
  REQUIRE(AnyDiffers(zeros, want, atol, rel));

  double worst = -std::numeric_limits<double>::infinity();
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double slack = std::abs(g - want[i]) - (atol + rel * std::abs(want[i]));
    if (!std::isfinite(g) || slack > worst) {
      worst = slack;
      worst_i = i;
    }
  }
  INFO(what << ": worst element " << worst_i << " got=" << got[worst_i]
            << " want=" << want[worst_i] << " slack=" << worst
            << " (band atol=" << atol << " rel=" << rel << ")");
  CHECK(worst <= 0.0);
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Silu(double x) { return x * Sigmoid(x); }
double Softplus(double x) { return x <= 20.0 ? std::log1p(std::exp(x)) : x; }

// ─── the tiny model ─────────────────────────────────────────────────────────
// Structurally the released checkpoint: all three block kinds, a real GQA ratio,
// grouped B/C sharing (n_groups < num_heads), a shared expert, and enough tokens
// to cross several SSD chunk boundaries. Small enough that a double reference is
// cheap and a failure names a number.
NemotronHParams TinyParams() {
  NemotronHParams p;
  p.hidden_size = 24;
  p.vocab_size = 32;
  p.max_position_embeddings = 4096;
  p.layer_norm_epsilon = 1e-5;
  p.tie_word_embeddings = false;

  p.layers_block_type = {NemotronHBlock::kMamba, NemotronHBlock::kMoe,
                         NemotronHBlock::kAttention, NemotronHBlock::kMamba,
                         NemotronHBlock::kMoe};
  p.num_nextn_predict_layers = 0;

  p.num_attention_heads = 4;
  p.num_key_value_heads = 2;  // GQA 2:1, as the real 32/2 is
  p.head_dim = 6;
  p.rope_theta = 10000.0;      // present in config.json and INERT here
  p.partial_rotary_factor = 1.0;  // ditto
  p.attention_bias = false;

  p.mamba_num_heads = 4;
  p.mamba_head_dim = 6;   // intermediate = 24 == hidden, as 64*64 == 4096 there
  p.n_groups = 2;         // < num_heads, so B/C are SHARED across head groups
  p.ssm_state_size = 8;
  p.conv_kernel = 4;
  p.chunk_size = 8;  // power of two, and T below crosses it several times
  p.expand = 2;
  p.mamba_hidden_act = "silu";
  p.mamba_ssm_cache_dtype = "float32";
  p.use_conv_bias = true;
  p.use_bias = false;
  p.mamba_proj_bias = false;

  p.n_routed_experts = 8;
  p.num_experts_per_tok = 3;
  p.moe_intermediate_size = 10;
  p.n_shared_experts = 1;
  p.moe_shared_expert_intermediate_size = 12;
  p.n_group = 1;
  p.topk_group = 1;
  p.routed_scaling_factor = 2.5;  // the released value
  p.norm_topk_prob = true;
  p.moe_shared_expert_overlap = true;

  p.intermediate_size = 14;
  p.mlp_bias = false;
  p.mlp_hidden_act = "relu2";
  return p;
}

// Deterministic, spread over several octaves so no term can hide under another.
float Synth(int64_t a, int64_t b, float k) {
  const double v = std::sin(0.7 * static_cast<double>(a) + 1.3 * static_cast<double>(b) +
                            0.21 * static_cast<double>(a * b));
  return static_cast<float>(k * v);
}

std::vector<float> SynthVec(size_t n, int64_t salt, float k) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = Synth(salt, static_cast<int64_t>(i), k);
  }
  return v;
}

NemotronHOwned Own(const std::vector<float>& v, DType dt, std::vector<int64_t> shape) {
  return NemotronHOwned::FromF32(v, dt, std::move(shape));
}

// Read a packed owned tensor back out at its DECLARED dtype. Used to inspect the
// carried recurrent state directly, which is the only place its dtype is
// observable at all — see the SSM-cache-dtype subcase.
std::vector<float> OwnedToFloat(const NemotronHOwned& o) {
  std::vector<float> out(static_cast<size_t>(o.Numel()));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = o.dtype == DType::kF32
                 ? reinterpret_cast<const float*>(o.bytes.data())[i]
                 : vt::BF16ToF32(reinterpret_cast<const uint16_t*>(o.bytes.data())[i]);
  }
  return out;
}

std::vector<double> OwnedToDouble(const NemotronHOwned& o) {
  const std::vector<float> f = OwnedToFloat(o);
  return std::vector<double>(f.begin(), f.end());
}

// The BF16-rounded value of every input, so a bf16 arm's reference sees the same
// operands the kernel does rather than a more precise pair.
std::vector<float> RoundTo(const std::vector<float>& v, DType dt) {
  if (dt == DType::kF32) return v;
  std::vector<float> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = vt::BF16ToF32(vt::F32ToBF16(v[i]));
  return r;
}

// ─── independent double references ──────────────────────────────────────────

// y[t,o] = sum_i x[t,i] * w[o,i]   (torch Linear, row-major [out,in]).
std::vector<double> RefLinear(const std::vector<float>& x, const std::vector<float>& w,
                              int64_t T, int64_t K, int64_t N) {
  std::vector<double> y(static_cast<size_t>(T * N), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int64_t i = 0; i < K; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(t * K + i)]) *
               static_cast<double>(w[static_cast<size_t>(o * K + i)]);
      }
      y[static_cast<size_t>(t * N + o)] = acc;
    }
  }
  return y;
}

std::vector<double> RefRmsNorm(const std::vector<double>& x, const std::vector<float>& w,
                               int64_t rows, int64_t dim, double eps) {
  std::vector<double> y(x.size(), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    double var = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      const double v = x[static_cast<size_t>(r * dim + d)];
      var += v * v;
    }
    var /= static_cast<double>(dim);
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int64_t d = 0; d < dim; ++d) {
      y[static_cast<size_t>(r * dim + d)] =
          x[static_cast<size_t>(r * dim + d)] * rstd * static_cast<double>(w[static_cast<size_t>(d)]);
    }
  }
  return y;
}

// Mixer2RMSNormGated (mamba_mixer2.py:100-149): SILU-gate first, then normalize
// over `group_size = hidden / n_groups` SLICES — not over the whole row.
std::vector<double> RefGatedGroupNorm(const std::vector<double>& x,
                                      const std::vector<double>& gate,
                                      const std::vector<float>& w, int64_t rows,
                                      int64_t hidden, int64_t n_groups, double eps) {
  const int64_t gs = hidden / n_groups;
  std::vector<double> y(x.size(), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<double> v(static_cast<size_t>(hidden));
    for (int64_t d = 0; d < hidden; ++d) {
      v[static_cast<size_t>(d)] =
          x[static_cast<size_t>(r * hidden + d)] * Silu(gate[static_cast<size_t>(r * hidden + d)]);
    }
    for (int64_t g = 0; g < n_groups; ++g) {
      double var = 0.0;
      for (int64_t d = 0; d < gs; ++d) {
        const double s = v[static_cast<size_t>(g * gs + d)];
        var += s * s;
      }
      var /= static_cast<double>(gs);
      const double rstd = 1.0 / std::sqrt(var + eps);
      for (int64_t d = 0; d < gs; ++d) {
        y[static_cast<size_t>(r * hidden + g * gs + d)] =
            v[static_cast<size_t>(g * gs + d)] * rstd *
            static_cast<double>(w[static_cast<size_t>(g * gs + d)]);
      }
    }
  }
  return y;
}

// The mamba weights as plain f32, so the reference never reads the packed bytes.
struct MambaRefWeights {
  std::vector<float> in_proj, out_proj, conv_w, conv_b, A_log, D, dt_bias, norm_w;
};

// The Mamba2 mixer, written straight from mamba_mixer2.py + the token-by-token
// SSD recurrence. Optionally seeded with a carried conv/SSM state, which is what
// makes the two-leg case a real test of the state, not of the scan.
struct MambaRefState {
  std::vector<double> conv;  // [conv_dim, K-1]
  std::vector<double> ssm;   // [Hh, P, N]
};

std::vector<double> RefMamba2Mixer(const MambaRefWeights& w, const NemotronHParams& p,
                                   const std::vector<float>& hidden, int64_t T,
                                   MambaRefState* state) {
  const int64_t H = p.hidden_size;
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t Hh = p.mamba_num_heads;
  const int64_t P = p.mamba_head_dim;
  const int64_t G = p.n_groups;
  const int64_t N = p.ssm_state_size;
  const int64_t K = p.conv_kernel;
  const int64_t proj = p.in_proj_out_features();

  const std::vector<double> zxbcdt = RefLinear(hidden, w.in_proj, T, H, proj);

  // conv over the xBC slice, silu activation, seeded from the carried state.
  std::vector<double> xbc(static_cast<size_t>(T * Cd), 0.0);
  for (int64_t c = 0; c < Cd; ++c) {
    for (int64_t t = 0; t < T; ++t) {
      double acc = static_cast<double>(w.conv_b[static_cast<size_t>(c)]);
      for (int64_t j = 0; j < K; ++j) {
        const int64_t tt = t - (K - 1 - j);
        double xv = 0.0;
        if (tt >= 0) {
          xv = zxbcdt[static_cast<size_t>(tt * proj + I + c)];
        } else if (state != nullptr) {
          // conv_state[c, :] holds the last K-1 RAW tokens, oldest first.
          const int64_t si = (K - 1) + tt;
          if (si >= 0) xv = state->conv[static_cast<size_t>(c * (K - 1) + si)];
        }
        acc += static_cast<double>(w.conv_w[static_cast<size_t>(c * K + j)]) * xv;
      }
      xbc[static_cast<size_t>(t * Cd + c)] = Silu(acc);
    }
  }
  if (state != nullptr) {
    // Write back the last K-1 RAW (pre-activation) tokens.
    std::vector<double> nc(static_cast<size_t>(Cd * (K - 1)), 0.0);
    for (int64_t c = 0; c < Cd; ++c) {
      for (int64_t s = 0; s < K - 1; ++s) {
        const int64_t tt = T - (K - 1) + s;
        if (tt >= 0) {
          nc[static_cast<size_t>(c * (K - 1) + s)] =
              zxbcdt[static_cast<size_t>(tt * proj + I + c)];
        } else {
          const int64_t si = (K - 1) + tt;
          nc[static_cast<size_t>(c * (K - 1) + s)] =
              state->conv[static_cast<size_t>(c * (K - 1) + si)];
        }
      }
    }
    state->conv = nc;
  }

  // The selective scan, token by token.
  std::vector<double> S(static_cast<size_t>(Hh * P * N), 0.0);
  if (state != nullptr) S = state->ssm;
  std::vector<double> y(static_cast<size_t>(T * I), 0.0);
  const int64_t heads_per_group = Hh / G;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < Hh; ++h) {
      const int64_t g = h / heads_per_group;
      // dt: bias, softplus, dt_limit(0, inf) — mamba_mixer2.py:888-889.
      double dt = zxbcdt[static_cast<size_t>(t * proj + I + Cd + h)] +
                  static_cast<double>(w.dt_bias[static_cast<size_t>(h)]);
      dt = Softplus(dt);
      const double A = -std::exp(static_cast<double>(w.A_log[static_cast<size_t>(h)]));
      const double decay = std::exp(A * dt);
      for (int64_t pp = 0; pp < P; ++pp) {
        const double xv = xbc[static_cast<size_t>(t * Cd + h * P + pp)];
        double out = 0.0;
        for (int64_t n = 0; n < N; ++n) {
          const double Bv = xbc[static_cast<size_t>(t * Cd + I + g * N + n)];
          const double Cv = xbc[static_cast<size_t>(t * Cd + I + G * N + g * N + n)];
          double& s = S[static_cast<size_t>((h * P + pp) * N + n)];
          s = s * decay + Bv * xv * dt;
          out += s * Cv;
        }
        out += static_cast<double>(w.D[static_cast<size_t>(h)]) * xv;
        y[static_cast<size_t>(t * I + h * P + pp)] = out;
      }
    }
  }
  if (state != nullptr) state->ssm = S;

  std::vector<double> z(static_cast<size_t>(T * I), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t i = 0; i < I; ++i) {
      z[static_cast<size_t>(t * I + i)] = zxbcdt[static_cast<size_t>(t * proj + i)];
    }
  }
  const std::vector<double> normed =
      RefGatedGroupNorm(y, z, w.norm_w, T, I, G, p.layer_norm_epsilon);

  std::vector<float> nf(normed.size());
  for (size_t i = 0; i < normed.size(); ++i) nf[i] = static_cast<float>(normed[i]);
  return RefLinear(nf, w.out_proj, T, I, H);
}

struct AttnRefWeights {
  std::vector<float> q, k, v, o;
};

// nemotron_h.py:473-486 — causal GQA SDPA, scale head_dim**-0.5, and NO
// POSITIONAL EMBEDDING of any kind.
std::vector<double> RefAttention(const AttnRefWeights& w, const NemotronHParams& p,
                                 const std::vector<float>& hidden, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t Hq = p.num_attention_heads;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t D = p.head_dim;
  const std::vector<double> q = RefLinear(hidden, w.q, T, H, Hq * D);
  const std::vector<double> k = RefLinear(hidden, w.k, T, H, Hkv * D);
  const std::vector<double> v = RefLinear(hidden, w.v, T, H, Hkv * D);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  const int64_t rep = Hq / Hkv;

  std::vector<double> ctx(static_cast<size_t>(T * Hq * D), 0.0);
  for (int64_t h = 0; h < Hq; ++h) {
    const int64_t g = h / rep;
    for (int64_t i = 0; i < T; ++i) {
      std::vector<double> s(static_cast<size_t>(i + 1));
      double mx = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j <= i; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < D; ++d) {
          dot += q[static_cast<size_t>(i * Hq * D + h * D + d)] *
                 k[static_cast<size_t>(j * Hkv * D + g * D + d)];
        }
        s[static_cast<size_t>(j)] = dot * scale;
        mx = std::max(mx, s[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int64_t j = 0; j <= i; ++j) {
        s[static_cast<size_t>(j)] = std::exp(s[static_cast<size_t>(j)] - mx);
        sum += s[static_cast<size_t>(j)];
      }
      for (int64_t d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          acc += (s[static_cast<size_t>(j)] / sum) *
                 v[static_cast<size_t>(j * Hkv * D + g * D + d)];
        }
        ctx[static_cast<size_t>(i * Hq * D + h * D + d)] = acc;
      }
    }
  }
  std::vector<float> cf(ctx.size());
  for (size_t i = 0; i < ctx.size(); ++i) cf[i] = static_cast<float>(ctx[i]);
  return RefLinear(cf, w.o, T, Hq * D, H);
}

struct MoeRefWeights {
  std::vector<float> gate, bias;
  std::vector<std::vector<float>> up, down;  // [E][I*H], [E][H*I]
  std::vector<float> shared_up, shared_down;
  bool has_shared = false;
};

// nemotron_h.py:126-256 with the grouped-topk router
// (grouped_topk_router.py:80-161, n_group == 1 so the group mask is trivial).
//
// THREE MIS-PORTS are expressible here, and they are NOT equally observable —
// which is the whole point of the case below, and a correction to what this file
// originally claimed:
//   * `fold_scale_into_weights` — multiply each router weight by the factor
//     instead of the assembled routed sum. Because the factor is applied AFTER
//     renormalisation, this is EXACT-ARITHMETIC-EQUAL to the shipped form;
//     measured separation on this fixture is 3.7e-5 relative, pure
//     floating-point association (245 of 288 elements differ BITWISE, none
//     numerically). It is a real defect and it is real work to catch — but the
//     instrument for it is a BITWISE comparison on engineered data, which lives
//     at the op level in tests/vt/test_ops_moe_nongated_relu2.cpp (spec §6a M6).
//     A tolerance-based model-level gate cannot see it, and this file used to
//     claim otherwise.
//   * `scale_logits` — multiply the router LOGITS. sigmoid is non-linear, so
//     this moves the weights and can move the SELECTION. Measured separation
//     25.3x the signal.
//   * `scale_shared_too` — apply the factor to the assembled output INCLUDING
//     the shared-expert term instead of the routed sum only. This is the trap
//     `apply_routed_scale_to_output=True` exists to name
//     (moe_runner.py:402-406 scales `fused_output` alone, then :722-725 adds
//     `shared_output`), it is the one a shared-expert-free architecture like
//     Laguna cannot expose, and unlike the fold it is a LARGE numeric difference.
std::vector<double> RefMoe(const MoeRefWeights& w, const NemotronHParams& p,
                           const std::vector<float>& hidden, int64_t T,
                           bool fold_scale_into_weights = false,
                           bool scale_logits = false,
                           bool scale_shared_too = false) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  const int64_t Kk = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size;
  const std::vector<double> logits = RefLinear(hidden, w.gate, T, H, E);

  std::vector<double> out(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> score(static_cast<size_t>(E)), sel(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
      double l = logits[static_cast<size_t>(t * E + e)];
      if (scale_logits) l *= p.routed_scaling_factor;
      score[static_cast<size_t>(e)] = Sigmoid(l);
      sel[static_cast<size_t>(e)] =
          score[static_cast<size_t>(e)] + static_cast<double>(w.bias[static_cast<size_t>(e)]);
    }
    // Greedy top-k with a strict `>` ascending scan: lowest index wins a tie,
    // the house convention both vt router kernels use.
    std::vector<int64_t> pick;
    std::vector<bool> taken(static_cast<size_t>(E), false);
    for (int64_t j = 0; j < Kk; ++j) {
      int64_t best = -1;
      for (int64_t e = 0; e < E; ++e) {
        if (taken[static_cast<size_t>(e)]) continue;
        if (best < 0 || sel[static_cast<size_t>(e)] > sel[static_cast<size_t>(best)]) best = e;
      }
      taken[static_cast<size_t>(best)] = true;
      pick.push_back(best);
    }
    // The WEIGHT is the UNBIASED score at the selected id (:147-150).
    std::vector<double> ww(pick.size());
    double denom = 0.0;
    for (size_t j = 0; j < pick.size(); ++j) {
      ww[j] = score[static_cast<size_t>(pick[j])];
      denom += ww[j];
    }
    if (p.norm_topk_prob) {
      if (!(denom > 0.0)) denom = 1.0;
      for (double& v : ww) v /= denom;
    }
    if (fold_scale_into_weights) {
      for (double& v : ww) v *= p.routed_scaling_factor;
    }

    std::vector<double> routed(static_cast<size_t>(H), 0.0);
    for (size_t j = 0; j < pick.size(); ++j) {
      const int64_t e = pick[j];
      std::vector<double> h(static_cast<size_t>(I), 0.0);
      for (int64_t i = 0; i < I; ++i) {
        double acc = 0.0;
        for (int64_t d = 0; d < H; ++d) {
          acc += static_cast<double>(hidden[static_cast<size_t>(t * H + d)]) *
                 static_cast<double>(w.up[static_cast<size_t>(e)][static_cast<size_t>(i * H + d)]);
        }
        const double r = std::max(0.0, acc);
        h[static_cast<size_t>(i)] = r * r;  // relu^2
      }
      for (int64_t d = 0; d < H; ++d) {
        double acc = 0.0;
        for (int64_t i = 0; i < I; ++i) {
          acc += h[static_cast<size_t>(i)] *
                 static_cast<double>(w.down[static_cast<size_t>(e)][static_cast<size_t>(d * I + i)]);
        }
        routed[static_cast<size_t>(d)] += ww[j] * acc;
      }
    }
    // The scale multiplies the ASSEMBLED routed sum; the shared term is added
    // AFTER and UNSCALED (moe_runner.py:402-406 then :722-725).
    if (!fold_scale_into_weights) {
      for (double& v : routed) v *= p.routed_scaling_factor;
    }
    for (int64_t d = 0; d < H; ++d) out[static_cast<size_t>(t * H + d)] = routed[static_cast<size_t>(d)];
  }

  if (w.has_shared) {
    const int64_t Is = p.moe_shared_expert_intermediate_size * p.n_shared_experts;
    for (int64_t t = 0; t < T; ++t) {
      std::vector<double> h(static_cast<size_t>(Is), 0.0);
      for (int64_t i = 0; i < Is; ++i) {
        double acc = 0.0;
        for (int64_t d = 0; d < H; ++d) {
          acc += static_cast<double>(hidden[static_cast<size_t>(t * H + d)]) *
                 static_cast<double>(w.shared_up[static_cast<size_t>(i * H + d)]);
        }
        const double r = std::max(0.0, acc);
        h[static_cast<size_t>(i)] = r * r;
      }
      for (int64_t d = 0; d < H; ++d) {
        double acc = 0.0;
        for (int64_t i = 0; i < Is; ++i) {
          acc += h[static_cast<size_t>(i)] *
                 static_cast<double>(w.shared_down[static_cast<size_t>(d * Is + i)]);
        }
        // The shipped form adds the shared term UNSCALED (moe_runner.py:722-725).
        // The mis-port scales it along with the routed sum.
        out[static_cast<size_t>(t * H + d)] +=
            scale_shared_too ? acc * p.routed_scaling_factor : acc;
      }
    }
  }
  return out;
}

// ─── weight builders (packed for the code under test, raw f32 for the ref) ───

struct TinyWeights {
  NemotronHHostWeights host;
  // raw f32 mirrors, rounded to the arm's dtype so the reference sees the same
  // operands the kernel does.
  std::vector<MambaRefWeights> mamba;   // by layer index (empty for non-mamba)
  std::vector<AttnRefWeights> attn;
  std::vector<MoeRefWeights> moe;
  std::vector<std::vector<float>> norm;
  std::vector<float> embed, norm_f, lm_head;
};

MambaRefWeights BuildMambaRef(const NemotronHParams& p, int64_t salt, DType dt) {
  MambaRefWeights r;
  r.in_proj = RoundTo(SynthVec(static_cast<size_t>(p.in_proj_out_features() * p.hidden_size),
                               salt + 1, 0.12f), dt);
  r.out_proj = RoundTo(SynthVec(static_cast<size_t>(p.hidden_size * p.mamba_intermediate_size()),
                                salt + 2, 0.15f), dt);
  r.conv_w = RoundTo(SynthVec(static_cast<size_t>(p.conv_dim() * p.conv_kernel), salt + 3, 0.4f), dt);
  r.conv_b = RoundTo(SynthVec(static_cast<size_t>(p.conv_dim()), salt + 4, 0.2f), dt);
  // A_log/D/dt_bias are f32 upstream whatever the model dtype, so they are NOT
  // rounded here — that asymmetry is the port, not an oversight.
  r.A_log = SynthVec(static_cast<size_t>(p.mamba_num_heads), salt + 5, 0.5f);
  r.D = SynthVec(static_cast<size_t>(p.mamba_num_heads), salt + 6, 0.3f);
  r.dt_bias = SynthVec(static_cast<size_t>(p.mamba_num_heads), salt + 7, 0.25f);
  r.norm_w = RoundTo(SynthVec(static_cast<size_t>(p.mamba_intermediate_size()), salt + 8, 0.9f), dt);
  return r;
}

NemotronHMambaWeights PackMamba(const MambaRefWeights& r, const NemotronHParams& p, DType dt) {
  NemotronHMambaWeights w;
  w.in_proj = Own(r.in_proj, dt, {p.in_proj_out_features(), p.hidden_size});
  w.out_proj = Own(r.out_proj, dt, {p.hidden_size, p.mamba_intermediate_size()});
  w.conv1d_weight = Own(r.conv_w, dt, {p.conv_dim(), p.conv_kernel});
  w.conv1d_bias = Own(r.conv_b, dt, {p.conv_dim()});
  w.A_log = Own(r.A_log, DType::kF32, {p.mamba_num_heads});
  w.D = Own(r.D, DType::kF32, {p.mamba_num_heads});
  w.dt_bias = Own(r.dt_bias, DType::kF32, {p.mamba_num_heads});
  w.norm_weight = Own(r.norm_w, dt, {p.mamba_intermediate_size()});
  return w;
}

// THE q/k SCALE IS LOAD-BEARING, and it was measured rather than chosen. At the
// inherited 0.2 the tiny model's attention logits came out at q.k*head_dim^-0.5
// ~ 0.09, so the softmax was NEAR-UNIFORM and the block degenerated into an
// unweighted mean of v. Two consequences, both measured: the output collapsed by
// cancellation to a mean |value| of 3.5e-4 at T=48, and the block became almost
// BLIND to anything that only moves attention WEIGHTS. Rotating q and k by a
// measured max_abs of 0.288 (RopeNeox, 360 of 384 elements moved) shifted the
// block's output by 9.4e-6 — which is why the no-RoPE guard below could not see
// a defect it was written to catch.
//
// The real checkpoint does not have this problem: head_dim 128 and hidden 2688
// put its logits in the selective regime by construction. 0.95 restores that
// regime here (logits O(2), a genuinely peaked softmax), so q/k defects — RoPE,
// the scale factor, causality — actually move the answer.
AttnRefWeights BuildAttnRef(const NemotronHParams& p, int64_t salt, DType dt) {
  AttnRefWeights r;
  const int64_t qd = p.q_proj_out_features(), kd = p.kv_proj_out_features();
  r.q = RoundTo(SynthVec(static_cast<size_t>(qd * p.hidden_size), salt + 1, 0.95f), dt);
  r.k = RoundTo(SynthVec(static_cast<size_t>(kd * p.hidden_size), salt + 2, 0.95f), dt);
  r.v = RoundTo(SynthVec(static_cast<size_t>(kd * p.hidden_size), salt + 3, 0.2f), dt);
  r.o = RoundTo(SynthVec(static_cast<size_t>(p.hidden_size * qd), salt + 4, 0.2f), dt);
  return r;
}

NemotronHAttentionWeights PackAttn(const AttnRefWeights& r, const NemotronHParams& p, DType dt) {
  NemotronHAttentionWeights w;
  const int64_t qd = p.q_proj_out_features(), kd = p.kv_proj_out_features();
  w.q_proj = Own(r.q, dt, {qd, p.hidden_size});
  w.k_proj = Own(r.k, dt, {kd, p.hidden_size});
  w.v_proj = Own(r.v, dt, {kd, p.hidden_size});
  w.o_proj = Own(r.o, dt, {p.hidden_size, qd});
  return w;
}

MoeRefWeights BuildMoeRef(const NemotronHParams& p, int64_t salt, DType dt, bool shared) {
  MoeRefWeights r;
  // The router weight is f32 on BOTH sides: force_fp32_compute=True upstream.
  r.gate = SynthVec(static_cast<size_t>(p.n_routed_experts * p.hidden_size), salt + 1, 0.35f);
  r.bias = SynthVec(static_cast<size_t>(p.n_routed_experts), salt + 2, 0.4f);
  for (int64_t e = 0; e < p.n_routed_experts; ++e) {
    r.up.push_back(RoundTo(
        SynthVec(static_cast<size_t>(p.moe_intermediate_size * p.hidden_size), salt + 10 + e, 0.3f),
        dt));
    r.down.push_back(RoundTo(
        SynthVec(static_cast<size_t>(p.hidden_size * p.moe_intermediate_size), salt + 40 + e, 0.3f),
        dt));
  }
  r.has_shared = shared;
  if (shared) {
    const int64_t Is = p.moe_shared_expert_intermediate_size * p.n_shared_experts;
    r.shared_up = RoundTo(SynthVec(static_cast<size_t>(Is * p.hidden_size), salt + 90, 0.3f), dt);
    r.shared_down = RoundTo(SynthVec(static_cast<size_t>(p.hidden_size * Is), salt + 91, 0.3f), dt);
  }
  return r;
}

NemotronHMoeWeights PackMoe(const MoeRefWeights& r, const NemotronHParams& p, DType dt) {
  NemotronHMoeWeights w;
  w.gate = Own(r.gate, DType::kF32, {p.n_routed_experts, p.hidden_size});
  w.e_score_correction_bias = Own(r.bias, DType::kF32, {p.n_routed_experts});
  for (int64_t e = 0; e < p.n_routed_experts; ++e) {
    NemotronHExpertWeights ew;
    ew.up_proj = Own(r.up[static_cast<size_t>(e)], dt, {p.moe_intermediate_size, p.hidden_size});
    ew.down_proj = Own(r.down[static_cast<size_t>(e)], dt, {p.hidden_size, p.moe_intermediate_size});
    w.experts.push_back(std::move(ew));
  }
  w.has_shared = r.has_shared;
  if (r.has_shared) {
    const int64_t Is = p.moe_shared_expert_intermediate_size * p.n_shared_experts;
    w.shared.up_proj = Own(r.shared_up, dt, {Is, p.hidden_size});
    w.shared.down_proj = Own(r.shared_down, dt, {p.hidden_size, Is});
  }
  return w;
}

TinyWeights BuildTiny(const NemotronHParams& p, DType dt, bool shared = true) {
  TinyWeights tw;
  tw.host.act_dtype = dt;
  const int64_t L = p.num_hidden_layers();
  tw.embed = RoundTo(SynthVec(static_cast<size_t>(p.vocab_size * p.hidden_size), 3, 0.5f), dt);
  tw.norm_f = RoundTo(SynthVec(static_cast<size_t>(p.hidden_size), 5, 0.8f), dt);
  tw.lm_head = RoundTo(SynthVec(static_cast<size_t>(p.vocab_size * p.hidden_size), 7, 0.25f), dt);
  tw.host.embeddings = Own(tw.embed, dt, {p.vocab_size, p.hidden_size});
  tw.host.norm_f = Own(tw.norm_f, dt, {p.hidden_size});
  tw.host.lm_head = Own(tw.lm_head, dt, {p.vocab_size, p.hidden_size});
  tw.mamba.resize(static_cast<size_t>(L));
  tw.attn.resize(static_cast<size_t>(L));
  tw.moe.resize(static_cast<size_t>(L));
  tw.norm.resize(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    NemotronHLayerWeights lw;
    lw.block = p.layers_block_type[static_cast<size_t>(l)];
    tw.norm[static_cast<size_t>(l)] =
        RoundTo(SynthVec(static_cast<size_t>(p.hidden_size), 100 + l, 0.9f), dt);
    lw.norm = Own(tw.norm[static_cast<size_t>(l)], dt, {p.hidden_size});
    switch (lw.block) {
      case NemotronHBlock::kMamba:
        tw.mamba[static_cast<size_t>(l)] = BuildMambaRef(p, 1000 + 100 * l, dt);
        lw.mamba = PackMamba(tw.mamba[static_cast<size_t>(l)], p, dt);
        break;
      case NemotronHBlock::kAttention:
        tw.attn[static_cast<size_t>(l)] = BuildAttnRef(p, 2000 + 100 * l, dt);
        lw.attn = PackAttn(tw.attn[static_cast<size_t>(l)], p, dt);
        break;
      case NemotronHBlock::kMoe:
        tw.moe[static_cast<size_t>(l)] = BuildMoeRef(p, 3000 + 100 * l, dt, shared);
        lw.moe = PackMoe(tw.moe[static_cast<size_t>(l)], p, dt);
        break;
      case NemotronHBlock::kMlp:
        break;
    }
    tw.host.layers.push_back(std::move(lw));
  }
  tw.host.materialized = true;
  return tw;
}

// The whole decoder in double: the SINGLE-branch pre-norm stream
// (nemotron_h.py:625-641). Returns the final normed hidden [T,H].
std::vector<double> RefStack(const TinyWeights& tw, const NemotronHParams& p,
                             const std::vector<int32_t>& ids) {
  const int64_t H = p.hidden_size;
  const int64_t T = static_cast<int64_t>(ids.size());
  const int64_t L = p.num_hidden_layers();
  std::vector<double> residual(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < H; ++d) {
      residual[static_cast<size_t>(t * H + d)] =
          tw.embed[static_cast<size_t>(ids[static_cast<size_t>(t)] * H + d)];
    }
  }
  std::vector<double> carry;
  for (int64_t l = 0; l < L; ++l) {
    std::vector<double> normed;
    if (l == 0) {
      normed = RefRmsNorm(residual, tw.norm[static_cast<size_t>(l)], T, H, p.layer_norm_epsilon);
    } else {
      for (size_t i = 0; i < residual.size(); ++i) residual[i] += carry[i];
      normed = RefRmsNorm(residual, tw.norm[static_cast<size_t>(l)], T, H, p.layer_norm_epsilon);
    }
    std::vector<float> nf(normed.size());
    for (size_t i = 0; i < normed.size(); ++i) nf[i] = static_cast<float>(normed[i]);
    switch (p.layers_block_type[static_cast<size_t>(l)]) {
      case NemotronHBlock::kMamba:
        carry = RefMamba2Mixer(tw.mamba[static_cast<size_t>(l)], p, nf, T, nullptr);
        break;
      case NemotronHBlock::kAttention:
        carry = RefAttention(tw.attn[static_cast<size_t>(l)], p, nf, T);
        break;
      case NemotronHBlock::kMoe:
        carry = RefMoe(tw.moe[static_cast<size_t>(l)], p, nf, T);
        break;
      case NemotronHBlock::kMlp:
        break;
    }
  }
  for (size_t i = 0; i < residual.size(); ++i) residual[i] += carry[i];
  return RefRmsNorm(residual, tw.norm_f, T, H, p.layer_norm_epsilon);
}

// The RELATIVE band per arm, as a fraction of the reference's own peak. The f32
// arm is the one that cannot hide a reduction-order defect
// ([[bf16-store-absorbs-reduction-order-defects]]); the bf16 arm is the released
// checkpoint's model dtype and carries bf16's ~2^-8 relative resolution through
// however many layers the case runs.
double RelFor(DType dt) { return dt == DType::kF32 ? 2e-4 : 3e-2; }

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. The Mamba2 mixer.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH mamba2 mixer equals an independent sequential reference") {
  const NemotronHParams p = TinyParams();
  // 20 tokens at chunk_size 8 => THREE logical chunks, so the inter-chunk state
  // passing runs twice. A T below one chunk would gate the degenerate case only.
  const int64_t T = 20;
  Queue q = CpuQ();
  for (DType dt : {DType::kF32, DType::kBF16}) {
    const MambaRefWeights r = BuildMambaRef(p, 1000, dt);
    const NemotronHMambaWeights w = PackMamba(r, p, dt);
    const std::vector<float> hidden =
        RoundTo(SynthVec(static_cast<size_t>(T * p.hidden_size), 11, 0.6f), dt);
    const std::vector<float> got =
        vllm::NemotronHMamba2Mixer(w, p, hidden, T, dt, q, nullptr);
    const std::vector<double> want = RefMamba2Mixer(r, p, hidden, T, nullptr);
    ExpectCloseRel(std::string("mamba2 mixer ") + (dt == DType::kF32 ? "f32" : "bf16"), got,
                   want, RelFor(dt));
  }
}

// A chunk-boundary defect (a dropped inter-chunk term, a wrong dA_cumsum) is
// invisible at ONE chunk size. The mixer's answer must not depend on it.
TEST_CASE("NemotronH mamba2 mixer is invariant to the SSD chunk size") {
  NemotronHParams p = TinyParams();
  const int64_t T = 20;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  const MambaRefWeights r = BuildMambaRef(p, 1000, dt);
  const NemotronHMambaWeights w = PackMamba(r, p, dt);
  const std::vector<float> hidden = SynthVec(static_cast<size_t>(T * p.hidden_size), 11, 0.6f);
  const std::vector<double> want = RefMamba2Mixer(r, p, hidden, T, nullptr);
  for (int64_t chunk : {4, 8, 16}) {
    NemotronHParams pc = p;
    pc.chunk_size = chunk;
    // Guard against the degenerate arm: at least two chunks must exist.
    REQUIRE(T > chunk);
    const std::vector<float> got =
        vllm::NemotronHMamba2Mixer(w, pc, hidden, T, dt, q, nullptr);
    ExpectCloseRel("mamba2 chunk " + std::to_string(chunk), got, want, 2e-4);
  }
}

// TRAP 3. The recurrent state's dtype is resolved from `mamba_ssm_cache_dtype`,
// INDEPENDENTLY of the conv/activation dtype. With FRESH state it is
// unobservable — the scan computes in f32 and only STORES `final_states` at that
// dtype — so this case carries a state across two legs, which is what the W6
// paged decode does every step.
TEST_CASE("NemotronH mamba2 state: the SSM dtype is independent, and it is CARRIED") {
  const NemotronHParams p = TinyParams();
  const int64_t T1 = 12, T2 = 8, T = T1 + T2;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  const MambaRefWeights r = BuildMambaRef(p, 1000, dt);
  const NemotronHMambaWeights w = PackMamba(r, p, dt);
  const std::vector<float> hidden = SynthVec(static_cast<size_t>(T * p.hidden_size), 11, 0.6f);

  SUBCASE("the resolver reads mamba_ssm_cache_dtype, not the activation dtype") {
    CHECK(vllm::NemotronHSsmCacheDType(p, DType::kBF16) == DType::kF32);
    CHECK(vllm::NemotronHSsmCacheDType(p, DType::kF32) == DType::kF32);
    NemotronHParams pb = p;
    pb.mamba_ssm_cache_dtype = "bfloat16";
    CHECK(vllm::NemotronHSsmCacheDType(pb, DType::kF32) == DType::kBF16);
  }

  SUBCASE("two legs carrying the state equal one whole-sequence leg") {
    const std::vector<float> whole =
        vllm::NemotronHMamba2Mixer(w, p, hidden, T, dt, q, nullptr);

    std::vector<float> h1(hidden.begin(),
                          hidden.begin() + static_cast<long>(T1 * p.hidden_size));
    std::vector<float> h2(hidden.begin() + static_cast<long>(T1 * p.hidden_size), hidden.end());
    NemotronHMambaState st;
    const std::vector<float> y1 = vllm::NemotronHMamba2Mixer(w, p, h1, T1, dt, q, &st);
    // The carried state is the RESOLVED ssm dtype, not the activation dtype.
    CHECK(st.has_initial);
    CHECK(st.ssm.dtype == vllm::NemotronHSsmCacheDType(p, dt));
    CHECK(st.ssm.Numel() == p.mamba_num_heads * p.mamba_head_dim * p.ssm_state_size);
    CHECK(st.conv.size() ==
          static_cast<size_t>(p.conv_dim() * (p.conv_kernel - 1)));
    const std::vector<float> y2 = vllm::NemotronHMamba2Mixer(w, p, h2, T2, dt, q, &st);

    std::vector<double> want(whole.begin(), whole.end());
    std::vector<float> joined = y1;
    joined.insert(joined.end(), y2.begin(), y2.end());
    ExpectCloseRel("two-leg == whole", joined, want, 2e-4);
  }

  // WHERE THE SSM CACHE DTYPE IS ACTUALLY OBSERVABLE, measured rather than
  // assumed. The state is what the dtype names, so the state is what this gates.
  //
  // The inherited subcase asserted the defect downstream, on leg 2's OUTPUT.
  // Measured, that separation is 3.16e-5 of the signal peak — BELOW the f32 arm's
  // own 2e-4 band, so no downstream assertion here can be both honest and
  // non-vacuous. The reason is the recurrence itself: A = -exp(A_log) puts the
  // per-token decay near exp(-1), so a carried state is forgotten within a few
  // tokens and its bf16 rounding never reaches most of leg 2. That is a property
  // of Mamba2, not a weakness of the port, and asserting through it would be
  // gating on noise.
  //
  // The STATE is where the dtype lives and where the difference is real: bf16
  // carries ~2^-8 relative resolution, so the stored recurrent state differs
  // materially from the f32 one on essentially every element. W6's paged decode
  // reads that state on EVERY step rather than once per leg, which is what makes
  // the dtype matter in production and what this pins.
  SUBCASE("the SSM cache dtype is load-bearing IN THE STORED STATE") {
    NemotronHParams pb = p;
    pb.mamba_ssm_cache_dtype = "bfloat16";
    REQUIRE(vllm::NemotronHSsmCacheDType(pb, dt) == DType::kBF16);
    std::vector<float> h1(hidden.begin(),
                          hidden.begin() + static_cast<long>(T1 * p.hidden_size));

    NemotronHMambaState st_f32;
    (void)vllm::NemotronHMamba2Mixer(w, p, h1, T1, dt, q, &st_f32);
    NemotronHMambaState st_bf16;
    (void)vllm::NemotronHMamba2Mixer(w, pb, h1, T1, dt, q, &st_bf16);

    // The MEMORY FORMAT itself, asserted rather than inferred from matching
    // numbers: a too-wide state is numerically correct and invisible downstream
    // ([[token-gates-cannot-see-dequant-fallbacks]]).
    CHECK(st_f32.ssm.dtype == DType::kF32);
    CHECK(st_bf16.ssm.dtype == DType::kBF16);
    REQUIRE(st_f32.ssm.Numel() == st_bf16.ssm.Numel());
    // ...and the byte counts differ by exactly the factor the dtypes name, which
    // is the assertion that caught the shared-resolver defect in W3 (spec §5c).
    CHECK(st_f32.ssm.bytes.size() == 2 * st_bf16.ssm.bytes.size());

    const std::vector<double> ref = OwnedToDouble(st_f32.ssm);
    const std::vector<float> got = OwnedToFloat(st_bf16.ssm);
    const double sep = RelSeparation(got, ref);
    INFO("bf16 vs f32 stored SSM state: relative separation " << sep);
    // bf16 resolves ~2^-8; anything at or below the f32 arm's 2e-4 band would
    // mean the store is not actually happening at the resolved dtype.
    CHECK(sep > 1e-3);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. The GQA attention block — and the NO-ROPE finding.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH attention equals an independent causal GQA reference") {
  const NemotronHParams p = TinyParams();
  Queue q = CpuQ();
  // A LONG arm as well as a short one: only 6 of 52 layers are attention, so an
  // attention defect is diluted 46:6 in the stack and a 6-token prompt can miss
  // it entirely (spec §6).
  for (int64_t T : {6, 48}) {
    for (DType dt : {DType::kF32, DType::kBF16}) {
      const AttnRefWeights r = BuildAttnRef(p, 2000, dt);
      const NemotronHAttentionWeights w = PackAttn(r, p, dt);
      const std::vector<float> hidden =
          RoundTo(SynthVec(static_cast<size_t>(T * p.hidden_size), 13, 0.6f), dt);
      const std::vector<float> got =
          vllm::NemotronHAttentionMixer(w, p, hidden, T, dt, q);
      const std::vector<double> want = RefAttention(r, p, hidden, T);
      ExpectCloseRel("attention T=" + std::to_string(T) +
                         (dt == DType::kF32 ? " f32" : " bf16"),
                     got, want, RelFor(dt));
    }
  }
}

// `models/nemotron_h.py` @ 555967922 contains ZERO occurrences of rope/rotary;
// NemotronHAttention.forward (:473-486) is qkv -> split -> attn -> o_proj. The
// config still ships rope_theta and partial_rotary_factor. Applying them changes
// no SHAPE and might not move a token on a short prompt, so this pins it
// numerically: the reference has no rotation, and a rotated q/k is a measurably
// different answer.
TEST_CASE("NemotronH attention applies NO positional embedding") {
  const NemotronHParams p = TinyParams();
  const int64_t T = 16;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  const AttnRefWeights r = BuildAttnRef(p, 2000, dt);
  const NemotronHAttentionWeights w = PackAttn(r, p, dt);
  const std::vector<float> hidden = SynthVec(static_cast<size_t>(T * p.hidden_size), 13, 0.6f);
  const std::vector<float> got = vllm::NemotronHAttentionMixer(w, p, hidden, T, dt, q);

  CHECK(vllm::kNemotronHAttentionHasNoRope);

  // A NeoX RoPE at the config's own rope_theta over the full head_dim, applied
  // to the reference's q/k, is a different answer at every position but 0.
  const std::vector<double> want_no_rope = RefAttention(r, p, hidden, T);
  ExpectCloseRel("attention without rope", got, want_no_rope, 2e-4);

  const int64_t H = p.hidden_size, Hq = p.num_attention_heads, Hkv = p.num_key_value_heads,
                D = p.head_dim;
  std::vector<float> qv(static_cast<size_t>(T * Hq * D)), kv(static_cast<size_t>(T * Hkv * D));
  {
    const std::vector<double> qd = RefLinear(hidden, r.q, T, H, Hq * D);
    const std::vector<double> kd = RefLinear(hidden, r.k, T, H, Hkv * D);
    for (size_t i = 0; i < qv.size(); ++i) qv[i] = static_cast<float>(qd[i]);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = static_cast<float>(kd[i]);
  }
  std::vector<int32_t> pos(static_cast<size_t>(T));
  std::iota(pos.begin(), pos.end(), 0);
  {
    vt::Tensor qt = vt::Tensor::Contiguous(qv.data(), DType::kF32, Cpu(), {T, Hq, D});
    vt::Tensor kt = vt::Tensor::Contiguous(kv.data(), DType::kF32, Cpu(), {T, Hkv, D});
    vt::Tensor pt = vt::Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {T});
    vt::RopeArgs ra;
    ra.base = static_cast<float>(p.rope_theta);
    ra.rotary_dim = static_cast<int>(p.head_dim * p.partial_rotary_factor);
    vt::RopeNeox(q, qt, kt, pt, ra);
  }
  // Re-run the causal core over the ROTATED q/k and confirm it disagrees. A
  // forward that applied RoPE would land here instead.
  std::vector<double> rotated(static_cast<size_t>(T * H), 0.0);
  {
    AttnRefWeights ident = r;
    // Reuse the reference's tail (attention core + o_proj) by feeding it q/k/v
    // through a hand-rolled repeat of the core; simplest is to inline it.
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    const int64_t rep = Hq / Hkv;
    const std::vector<double> vv = RefLinear(hidden, r.v, T, H, Hkv * D);
    std::vector<float> ctx(static_cast<size_t>(T * Hq * D), 0.0f);
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / rep;
      for (int64_t i = 0; i < T; ++i) {
        std::vector<double> s(static_cast<size_t>(i + 1));
        double mx = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j <= i; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < D; ++d) {
            dot += static_cast<double>(qv[static_cast<size_t>(i * Hq * D + h * D + d)]) *
                   static_cast<double>(kv[static_cast<size_t>(j * Hkv * D + g * D + d)]);
          }
          s[static_cast<size_t>(j)] = dot * scale;
          mx = std::max(mx, s[static_cast<size_t>(j)]);
        }
        double sum = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          s[static_cast<size_t>(j)] = std::exp(s[static_cast<size_t>(j)] - mx);
          sum += s[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j <= i; ++j) {
            acc += (s[static_cast<size_t>(j)] / sum) * vv[static_cast<size_t>(j * Hkv * D + g * D + d)];
          }
          ctx[static_cast<size_t>(i * Hq * D + h * D + d)] = static_cast<float>(acc);
        }
      }
    }
    rotated = RefLinear(ctx, ident.o, T, Hq * D, H);
  }
  // The instrument first: prove RopeNeox actually ROTATED q. A guard that fails
  // because its own rotation was a no-op would look exactly like a forward that
  // correctly applies none ([[absent-hook-looks-like-armed-instrument]]).
  {
    const std::vector<double> q_unrot = RefLinear(hidden, r.q, T, H, Hq * D);
    const double moved = RelSeparation(qv, q_unrot);
    INFO("RopeNeox moved q by relative " << moved);
    REQUIRE(moved > 1e-2);
  }
  // And now the finding: the shipped forward is NOT the rotated answer.
  const double sep = RelSeparation(got, rotated);
  INFO("no-rope forward vs a rope'd one: relative separation " << sep);
  CHECK(sep > 1e-2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. The non-gated relu² MoE block — the routed-scale POSITION and the f32 router.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH MoE equals an independent non-gated relu^2 reference") {
  const NemotronHParams p = TinyParams();
  const int64_t T = 12;
  Queue q = CpuQ();
  for (DType dt : {DType::kF32, DType::kBF16}) {
    const MoeRefWeights r = BuildMoeRef(p, 3000, dt, /*shared=*/true);
    const NemotronHMoeWeights w = PackMoe(r, p, dt);
    const std::vector<float> hidden =
        RoundTo(SynthVec(static_cast<size_t>(T * p.hidden_size), 17, 0.6f), dt);
    const std::vector<float> got = vllm::NemotronHMoeMixer(w, p, hidden, T, dt, q);
    const std::vector<double> want = RefMoe(r, p, hidden, T);
    ExpectCloseRel(std::string("moe ") + (dt == DType::kF32 ? "f32" : "bf16"), got, want,
                   RelFor(dt));
  }
}

// TRAP 1. `apply_routed_scale_to_output=True` (nemotron_h.py:234) — WHERE the
// 2.5 is applied.
//
// This case previously asserted that folding the factor into the ROUTER WEIGHTS
// "scales the shared term too and is a different answer on every token". That is
// FALSE, and measuring it is what showed why: the factor is applied AFTER the
// top-k renormalisation, so `scale * Σ w_j e_j` and `Σ (scale*w_j) e_j` are the
// same expression. Measured separation on this fixture: 3.7e-5 relative,
// 2.1e-8 absolute — pure floating-point association, with 245 of 288 elements
// differing BITWISE and none numerically. The guard could never have passed, and
// a tolerance-based model-level comparison is the wrong instrument for it; the
// bitwise one is at the op level (spec §6a M6,
// tests/vt/test_ops_moe_nongated_relu2.cpp). Recorded here so the next reader
// does not re-derive it from a failing assertion.
//
// What IS observable at this level, and what this case now gates:
//   * scaling the OUTPUT INCLUDING the shared term rather than the routed sum
//     alone — the defect `apply_routed_scale_to_output=True` exists to prevent,
//     and the one a shared-expert-free architecture cannot expose at all;
//   * scaling the router LOGITS — sigmoid is non-linear, so it moves the weights
//     and can move the SELECTION.
TEST_CASE("NemotronH MoE: the routed scale is on the routed sum, not the shared term or logits") {
  NemotronHParams p = TinyParams();
  REQUIRE(p.routed_scaling_factor != 1.0);
  const int64_t T = 12;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  const MoeRefWeights r = BuildMoeRef(p, 3000, dt, /*shared=*/true);
  const NemotronHMoeWeights w = PackMoe(r, p, dt);
  REQUIRE(r.has_shared);  // the shared term is what separates the mis-ports
  const std::vector<float> hidden = SynthVec(static_cast<size_t>(T * p.hidden_size), 17, 0.6f);
  const std::vector<float> got = vllm::NemotronHMoeMixer(w, p, hidden, T, dt, q);

  const std::vector<double> want = RefMoe(r, p, hidden, T);
  ExpectCloseRel("moe, scale on the routed sum only", got, want, 2e-4);

  // MIS-PORT A: the factor applied to routed + shared.
  const std::vector<double> shared_scaled =
      RefMoe(r, p, hidden, T, /*fold=*/false, /*scale_logits=*/false,
             /*scale_shared_too=*/true);
  const double sep_shared = RelSeparation(got, shared_scaled);
  INFO("scaling the shared term too: relative separation " << sep_shared);
  CHECK(sep_shared > 1e-2);

  // MIS-PORT B: the factor applied to the router logits.
  const std::vector<double> logit_scaled =
      RefMoe(r, p, hidden, T, /*fold=*/false, /*scale_logits=*/true);
  const double sep_logits = RelSeparation(got, logit_scaled);
  INFO("scaling the router logits: relative separation " << sep_logits);
  CHECK(sep_logits > 1e-2);

  // And the fold, pinned as what it actually is: arithmetically INDISTINGUISHABLE
  // here, with or without a shared expert. This is why Laguna may legitimately
  // fold the same factor (laguna_ops.h:48) and why copying that is a BITWISE
  // defect rather than a numeric one — the thing a tolerance gate must not be
  // claimed to catch.
  const std::vector<double> folded =
      RefMoe(r, p, hidden, T, /*fold_scale_into_weights=*/true);
  const double sep_fold = RelSeparation(got, folded);
  INFO("folding into the router weights: relative separation " << sep_fold);
  CHECK(sep_fold < 1e-3);
  ExpectCloseRel("shared present: the fold is arithmetically indistinguishable", got, folded,
                 2e-4);

  const MoeRefWeights rn = BuildMoeRef(p, 3000, dt, /*shared=*/false);
  const NemotronHMoeWeights wn = PackMoe(rn, p, dt);
  const std::vector<float> gotn = vllm::NemotronHMoeMixer(wn, p, hidden, T, dt, q);
  const std::vector<double> foldedn =
      RefMoe(rn, p, hidden, T, /*fold_scale_into_weights=*/true);
  ExpectCloseRel("no-shared: the fold is indistinguishable", gotn, foldedn, 2e-4);
}

// TRAP 2. `GateLinear(out_dtype=torch.float32, force_fp32_compute=True)`
// (nemotron_h.py:150-156). The router is f32 whatever the model dtype. On the
// bf16 arm the block's own tolerance is far too loose to see a bf16 router, so
// this compares the ROUTER STAGE directly: the f32 logits against a bf16 GEMM's,
// and asserts the weights the block was built with are f32.
TEST_CASE("NemotronH MoE: the router runs in f32, mirrored not inherited") {
  const NemotronHParams p = TinyParams();
  const int64_t T = 12;
  const int64_t H = p.hidden_size, E = p.n_routed_experts;
  const DType dt = DType::kBF16;  // the released model dtype
  const MoeRefWeights r = BuildMoeRef(p, 3000, dt, /*shared=*/true);
  const NemotronHMoeWeights w = PackMoe(r, p, dt);

  // The router's two parameters are f32 EVEN ON THE BF16 ARM.
  CHECK(w.gate.dtype == DType::kF32);
  CHECK(w.e_score_correction_bias.dtype == DType::kF32);

  // And the f32 GEMM is a different number from a bf16 one, so the dtype is
  // load-bearing rather than a cosmetic annotation.
  const std::vector<float> hidden =
      RoundTo(SynthVec(static_cast<size_t>(T * H), 17, 0.6f), dt);
  const std::vector<double> f32_logits = RefLinear(hidden, r.gate, T, H, E);
  std::vector<float> bf16_logits(static_cast<size_t>(T * E), 0.0f);
  {
    const std::vector<float> gb = RoundTo(r.gate, DType::kBF16);
    const std::vector<double> d = RefLinear(hidden, gb, T, H, E);
    for (size_t i = 0; i < bf16_logits.size(); ++i) {
      bf16_logits[i] = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(d[i])));
    }
  }
  CHECK(AnyDiffers(bf16_logits, f32_logits, 1e-6, 1e-6));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. The dense `-` MLP block.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH dense MLP is the same non-gated relu^2 shape") {
  const NemotronHParams p = TinyParams();
  const int64_t T = 7, H = p.hidden_size, I = p.intermediate_size;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  NemotronHMlpWeights w;
  const std::vector<float> up = SynthVec(static_cast<size_t>(I * H), 501, 0.3f);
  const std::vector<float> down = SynthVec(static_cast<size_t>(H * I), 502, 0.3f);
  w.up_proj = Own(up, dt, {I, H});
  w.down_proj = Own(down, dt, {H, I});
  const std::vector<float> hidden = SynthVec(static_cast<size_t>(T * H), 19, 0.6f);
  const std::vector<float> got = vllm::NemotronHMlpMixer(w, p, hidden, T, dt, q);

  std::vector<double> want(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> h(static_cast<size_t>(I), 0.0);
    for (int64_t i = 0; i < I; ++i) {
      double acc = 0.0;
      for (int64_t d = 0; d < H; ++d) {
        acc += static_cast<double>(hidden[static_cast<size_t>(t * H + d)]) *
               static_cast<double>(up[static_cast<size_t>(i * H + d)]);
      }
      const double rr = std::max(0.0, acc);
      h[static_cast<size_t>(i)] = rr * rr;
    }
    for (int64_t d = 0; d < H; ++d) {
      double acc = 0.0;
      for (int64_t i = 0; i < I; ++i) {
        acc += h[static_cast<size_t>(i)] * static_cast<double>(down[static_cast<size_t>(d * I + i)]);
      }
      want[static_cast<size_t>(t * H + d)] = acc;
    }
  }
  ExpectCloseRel("dense mlp", got, want, 2e-4);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. The whole hybrid stack — the thing W4 actually is.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH forward: the hybrid stack computes through all three block kinds") {
  const NemotronHParams p = TinyParams();
  Queue q = CpuQ();
  // SHORT and LONG. The long arm is not decoration: 6-of-52 attention means a
  // short prompt can leave an attention defect invisible (spec §6), and the SSD
  // scan only passes state once T exceeds one chunk.
  for (int64_t T : {6, 40}) {
    for (DType dt : {DType::kF32, DType::kBF16}) {
      const TinyWeights tw = BuildTiny(p, dt);
      std::vector<int32_t> ids(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) {
        ids[static_cast<size_t>(t)] = static_cast<int32_t>((t * 7 + 3) % p.vocab_size);
      }
      vllm::NemotronHTrace trace;
      trace.capture = true;
      const std::vector<float> logits =
          vllm::NemotronHForward(tw.host, p, ids, {}, q, &trace);
      REQUIRE(logits.size() == static_cast<size_t>(T * p.vocab_size));

      // (a) the FINAL NORMED HIDDEN, elementwise, against the double stack. This
      // is the per-layer-style numeric bound porting-a-model.md §3 asks for; a
      // tokens-only comparison could pass with a dropped mechanism.
      const std::vector<double> want_hidden = RefStack(tw, p, ids);
      ExpectCloseRel("final normed hidden T=" + std::to_string(T) +
                         (dt == DType::kF32 ? " f32" : " bf16"),
                     trace.final_normed, want_hidden, RelFor(dt));

      // (b) every block kind ran, and every layer moved the stream. A mixer that
      // silently returned zeros would still produce plausible logits.
      REQUIRE(trace.mixer.size() == static_cast<size_t>(p.num_hidden_layers()));
      for (size_t l = 0; l < trace.mixer.size(); ++l) {
        double mx = 0.0;
        for (float v : trace.mixer[l]) mx = std::max(mx, std::abs(static_cast<double>(v)));
        INFO("layer " << l << " is a "
                      << std::string(vllm::NemotronHBlockName(
                             p.layers_block_type[l])));
        CHECK(mx > 1e-6);
      }

      // (c) the logits themselves, off the same reference hidden.
      std::vector<float> hf(want_hidden.size());
      for (size_t i = 0; i < hf.size(); ++i) hf[i] = static_cast<float>(want_hidden[i]);
      const std::vector<double> want_logits =
          RefLinear(hf, tw.lm_head, T, p.hidden_size, p.vocab_size);
      ExpectCloseRel("logits T=" + std::to_string(T) + (dt == DType::kF32 ? " f32" : " bf16"),
                     logits, want_logits, RelFor(dt) * 10.0);
    }
  }
}

// Every layer is a (norm, mixer) PAIR — one norm, not two. A port that gave each
// layer an input_layernorm AND a post_attention_layernorm would double the depth
// of the norm chain and is a different answer.
TEST_CASE("NemotronH forward: the residual stream is SINGLE-branch per layer") {
  const NemotronHParams p = TinyParams();
  const int64_t T = 10;
  Queue q = CpuQ();
  const DType dt = DType::kF32;
  const TinyWeights tw = BuildTiny(p, dt);
  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) ids[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);

  vllm::NemotronHTrace trace;
  trace.capture = true;
  (void)vllm::NemotronHForward(tw.host, p, ids, {}, q, &trace);

  // Layer 0's norm input IS the embedding (residual is None at :627-631), and
  // layer l>0's residual is the running sum of every previous mixer output.
  const int64_t H = p.hidden_size;
  std::vector<double> residual(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < H; ++d) {
      residual[static_cast<size_t>(t * H + d)] =
          tw.embed[static_cast<size_t>(ids[static_cast<size_t>(t)] * H + d)];
    }
  }
  for (size_t l = 0; l < trace.mixer.size(); ++l) {
    for (size_t i = 0; i < residual.size(); ++i) residual[i] += trace.mixer[l][i];
    ExpectCloseRel("residual after layer " + std::to_string(l), trace.hidden[l], residual,
                   2e-4);
    // And the norm the NEXT layer sees is that residual, normed ONCE.
    if (l + 1 < trace.mixer.size()) {
      const std::vector<double> want =
          RefRmsNorm(residual, tw.norm[l + 1], T, H, p.layer_norm_epsilon);
      ExpectCloseRel("normed input of layer " + std::to_string(l + 1), trace.normed[l + 1],
                     want, 2e-4);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. What the forward still REFUSES, by name.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("NemotronH forward: the pieces still owed REFUSE BY NAME") {
  const NemotronHParams p = TinyParams();
  Queue q = CpuQ();

  SUBCASE("unmaterialized weights are refused, never computed on as zeros") {
    NemotronHHostWeights empty;
    empty.act_dtype = DType::kF32;
    CHECK_THROWS_AS(vllm::NemotronHForward(empty, p, {1, 2, 3}, {}, q, nullptr),
                    std::runtime_error);
    try {
      (void)vllm::NemotronHForward(empty, p, {1, 2, 3}, {}, q, nullptr);
      FAIL("expected a refusal");
    } catch (const std::runtime_error& e) {
      const std::string msg = e.what();
      CHECK(msg.find("not materialized") != std::string::npos);
      CHECK(msg.find("nemotron-h-model.md") != std::string::npos);
    }
  }

  SUBCASE("a non-CPU queue is refused: the device/paged path is W6") {
    const TinyWeights tw = BuildTiny(p, DType::kF32);
    Queue cuda{Device{DeviceType::kCUDA, 0}, nullptr};
    CHECK_THROWS_AS(vllm::NemotronHForward(tw.host, p, {1, 2, 3}, {}, cuda, nullptr),
                    std::runtime_error);
  }

  SUBCASE("a latent MoE is refused by name (spec §0 puts it out of scope)") {
    NemotronHParams pl = p;
    pl.moe_latent_size = 16;
    const MoeRefWeights r = BuildMoeRef(p, 3000, DType::kF32, true);
    const NemotronHMoeWeights w = PackMoe(r, p, DType::kF32);
    const std::vector<float> hidden = SynthVec(static_cast<size_t>(4 * p.hidden_size), 17, 0.6f);
    CHECK_THROWS_AS(vllm::NemotronHMoeMixer(w, pl, hidden, 4, DType::kF32, q),
                    std::runtime_error);
  }

  SUBCASE("an unported activation is refused rather than silently substituted") {
    NemotronHParams ps = p;
    ps.mamba_hidden_act = "gelu";
    const MambaRefWeights r = BuildMambaRef(p, 1000, DType::kF32);
    const NemotronHMambaWeights w = PackMamba(r, p, DType::kF32);
    const std::vector<float> hidden = SynthVec(static_cast<size_t>(8 * p.hidden_size), 11, 0.6f);
    CHECK_THROWS_AS(vllm::NemotronHMamba2Mixer(w, ps, hidden, 8, DType::kF32, q, nullptr),
                    std::runtime_error);
  }

  SUBCASE("an f16 model dtype is refused: no op on this path can store it") {
    CHECK_THROWS_AS(NemotronHOwned::FromF32({1.0f}, DType::kF16, {1}), std::runtime_error);
  }
}

// The greedy loop is what a token gate would ride on. It is exercised here only
// to prove it runs and is deterministic; the TOKEN gate against the pinned
// oracle's committed goldens is W6's, and needs the real checkpoint.
TEST_CASE("NemotronH greedy decode is deterministic and in range") {
  const NemotronHParams p = TinyParams();
  Queue q = CpuQ();
  const TinyWeights tw = BuildTiny(p, DType::kBF16);
  const std::vector<int32_t> prompt = {3, 9, 14, 2, 7, 21};
  const std::vector<int32_t> a = vllm::NemotronHGreedyDecode(tw.host, p, prompt, 4, q);
  const std::vector<int32_t> b = vllm::NemotronHGreedyDecode(tw.host, p, prompt, 4, q);
  REQUIRE(a.size() == 4);
  CHECK(a == b);
  for (int32_t t : a) {
    CHECK(t >= 0);
    CHECK(t < p.vocab_size);
  }
}
