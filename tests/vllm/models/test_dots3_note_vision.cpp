// dots3-note W6a — the DENSE vision tower, against an INDEPENDENT
// double-precision reference (#2512, `.agents/specs/dots3-note.md` §4.11).
//
// WHAT THIS GATE IS, IN ITS OWN WORDS. §6.4 of the spec records option B,
// decided 2026-08-15: the checkpoint is 298.67 GB fp8 / 576.89 GB bf16 against
// 119-122 GiB hosts, so vLLM cannot be run on it on any hardware this project
// owns and NO DENOMINATOR EXISTS. This is therefore a CONSISTENCY gate. It
// establishes that two implementations of the same formula agree. It does NOT
// establish that either matches vLLM, and no performance number is claimable on
// any axis while B holds. Nothing below claims otherwise.
//
// The reference is written from `vllm/models/dots3_note/nvidia/vision.py` and
// `nvidia/vision_attention.py` read in `~/_git/vllm` at **`9035151d6`** — the
// merge of vllm#51255 — and shares NO helper with the implementation. It is a
// scalar `double` loop with its own GEMM, its own softmax, its own rope and its
// own norms; the implementation is `vt::MatmulBT` / `vt::RmsNorm` /
// `vt::RopeFromCache` / `vt::AttentionDenseFlash` over bf16 device buffers
// through the shared seams. Every anchor names that SHA because upstream has
// already moved under this row: `vision_attention.py` is 477 lines at
// `9035151d6` and 494 at vLLM `main` `7a100bb61`.
//
// THE ONE FORMULA DIFFERENCE, and why the reference does not copy it. Upstream's
// `RMSNorm.forward` (`vision.py:114-116`) casts the normalized value back to the
// ACTIVATION dtype before multiplying by the weight; `vt::RmsNorm` keeps f32
// through that multiply. At infinite precision the two are the same function,
// so the reference — which is double throughout — is the algebra BOTH implement
// and the tolerance below covers our bf16 storage AND upstream's intermediate
// cast together. Copying the cast into the reference would make the reference
// agree with a rounding choice instead of with the maths.
#include "vllm/model_executor/models/dots3_note_vision.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <system_error>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dots3_note_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/multimodal/pil_resize.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using dots3_tiny::TinyCheckpoint;
using dots3_tiny::TinySpec;
using vllm::Dots3NoteVisionForward;
using vllm::Dots3NoteVisionParams;
using vllm::Dots3NoteVisionPosIds;
using vllm::Dots3NoteVisionRefusal;
using vllm::Dots3NoteVisionWeights;
using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ParseDots3NoteVisionParams;

std::string FixtureDir() { return DOTS3_NOTE_CKPT_FIXTURE_DIR; }

nlohmann::json ReleasedConfigDoc() {
  std::ifstream in(FixtureDir() + "/config.json");
  REQUIRE_MESSAGE(in.good(), "cannot open " << FixtureDir() << "/config.json");
  nlohmann::json j;
  in >> j;
  return j;
}

// A throwaway `config.json` holding an arbitrary document, so a case can drive
// the REAL `LoadHfConfig` -> `ParseDots3NoteVisionParams` path rather than
// building an `HfConfig` by hand.
class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    static int counter = 0;
    static const unsigned salt = std::random_device{}();
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_vision_cfg_" + std::to_string(salt) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json", std::ios::binary) << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

Dots3NoteVisionParams ParseDoc(const nlohmann::json& doc) {
  const TempConfig cfg(doc);
  return ParseDots3NoteVisionParams(LoadHfConfig(cfg.path()));
}

// ═══════════════════════════════════════════════════════════════════════════
// THE INDEPENDENT REFERENCE. Every line below is written from the upstream
// source at `9035151d6`; it calls nothing the implementation calls.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref {

// out[M,N] = x[M,K] @ w[N,K]^T (+ bias). Plain triple loop, double accumulator.
std::vector<double> Linear(const std::vector<double>& x,
                           const std::vector<double>& w,
                           const std::vector<double>* bias, int64_t M,
                           int64_t K, int64_t N) {
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(n)] : 0.0;
      for (int64_t k = 0; k < K; ++k) {
        acc += x[static_cast<size_t>(m * K + k)] *
               w[static_cast<size_t>(n * K + k)];
      }
      out[static_cast<size_t>(m * N + n)] = acc;
    }
  }
  return out;
}

// `RMSNorm.forward` (vision.py:108-122) and `_RMSNorm` (vision_attention.py:97-110),
// which are the same function: `x * rsqrt(mean(x^2) + eps) * weight`. The
// intermediate `.type_as(x)` is a no-op in double (see the file header).
std::vector<double> Rms(const std::vector<double>& x,
                        const std::vector<double>& w, int64_t rows, int64_t dim,
                        double eps) {
  std::vector<double> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double v = x[static_cast<size_t>(r * dim + c)];
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (int64_t c = 0; c < dim; ++c) {
      out[static_cast<size_t>(r * dim + c)] =
          x[static_cast<size_t>(r * dim + c)] * inv * w[static_cast<size_t>(c)];
    }
  }
  return out;
}

// `nn.LayerNorm` with weight and bias, the adapter's `ln_q` (vision.py:481).
std::vector<double> LayerNorm(const std::vector<double>& x,
                              const std::vector<double>& w,
                              const std::vector<double>& b, int64_t rows,
                              int64_t dim, double eps) {
  std::vector<double> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < dim; ++c) mean += x[static_cast<size_t>(r * dim + c)];
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double d = x[static_cast<size_t>(r * dim + c)] - mean;
      var += d * d;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = 0; c < dim; ++c) {
      out[static_cast<size_t>(r * dim + c)] =
          (x[static_cast<size_t>(r * dim + c)] - mean) * inv *
              w[static_cast<size_t>(c)] +
          b[static_cast<size_t>(c)];
    }
  }
  return out;
}

// `get_pos_ids_by_grid` (vision.py:565-599), written straight from the reshape /
// permute / flatten upstream spells, rather than from the loop the
// implementation collapsed it into.
std::vector<std::array<int64_t, 2>> PosIds(int64_t t, int64_t h, int64_t w,
                                           int64_t rope_merge) {
  // hpos[i][j] = i ; wpos[i][j] = j, both [h, w]
  std::vector<int64_t> hp(static_cast<size_t>(h * w));
  std::vector<int64_t> wp(static_cast<size_t>(h * w));
  for (int64_t i = 0; i < h; ++i) {
    for (int64_t j = 0; j < w; ++j) {
      hp[static_cast<size_t>(i * w + j)] = i;
      wp[static_cast<size_t>(i * w + j)] = j;
    }
  }
  // reshape(h/m, m, w/m, m) then permute(0, 2, 1, 3) then flatten.
  const int64_t m = rope_merge;
  const auto regroup = [&](const std::vector<int64_t>& src) {
    std::vector<int64_t> out;
    out.reserve(src.size());
    for (int64_t a = 0; a < h / m; ++a) {
      for (int64_t c = 0; c < w / m; ++c) {
        for (int64_t b = 0; b < m; ++b) {
          for (int64_t dd = 0; dd < m; ++dd) {
            // index into the [h/m, m, w/m, m] view: (a, b, c, dd)
            const int64_t row = a * m + b;
            const int64_t col = c * m + dd;
            out.push_back(src[static_cast<size_t>(row * w + col)]);
          }
        }
      }
    }
    return out;
  };
  const std::vector<int64_t> hf = regroup(hp);
  const std::vector<int64_t> wf = regroup(wp);
  std::vector<std::array<int64_t, 2>> out;
  for (int64_t f = 0; f < t; ++f) {
    for (size_t i = 0; i < hf.size(); ++i) out.push_back({hf[i], wf[i]});
  }
  return out;
}

// `apply_rotary_pos_emb_vision` (vision_attention.py:33-49) applied to ONE
// [L, nh, hd] tensor, with the frequency table built as
// `VisionRotaryEmbedding(hd // 2)` (vision_attention.py:60-89).
void ApplyRope(std::vector<double>* x,
               const std::vector<std::array<int64_t, 2>>& pos, int64_t L,
               int64_t nh, int64_t hd) {
  const int64_t dim = hd / 2;        // the table's own `dim`
  const int64_t nf = dim / 2;        // frequencies per spatial axis
  for (int64_t l = 0; l < L; ++l) {
    // freqs[l] is [2, nf] flattened to [dim]; cos/sin are then REPEATED to hd
    // as [f | f] (`.repeat(1, 1, 2)` at vision_attention.py:46-47).
    std::vector<double> c(static_cast<size_t>(hd)), s(static_cast<size_t>(hd));
    for (int64_t axis = 0; axis < 2; ++axis) {
      for (int64_t i = 0; i < nf; ++i) {
        const double invf =
            1.0 / std::pow(10000.0, static_cast<double>(2 * i) /
                                        static_cast<double>(dim));
        const double ang =
            static_cast<double>(pos[static_cast<size_t>(l)][
                static_cast<size_t>(axis)]) * invf;
        const int64_t k = axis * nf + i;
        c[static_cast<size_t>(k)] = std::cos(ang);
        c[static_cast<size_t>(dim + k)] = std::cos(ang);
        s[static_cast<size_t>(k)] = std::sin(ang);
        s[static_cast<size_t>(dim + k)] = std::sin(ang);
      }
    }
    for (int64_t h = 0; h < nh; ++h) {
      const size_t base = static_cast<size_t>((l * nh + h) * hd);
      std::vector<double> in(x->begin() + static_cast<ptrdiff_t>(base),
                             x->begin() + static_cast<ptrdiff_t>(base + hd));
      for (int64_t d = 0; d < hd; ++d) {
        // rotate_half: (-x2, x1) over the two halves (vision_attention.py:33-36)
        const double rot = d < dim ? -in[static_cast<size_t>(d + dim)]
                                   : in[static_cast<size_t>(d - dim)];
        (*x)[base + static_cast<size_t>(d)] =
            in[static_cast<size_t>(d)] * c[static_cast<size_t>(d)] +
            rot * s[static_cast<size_t>(d)];
      }
    }
  }
}

// `VisionAttentionV2.forward` (vision_attention.py:210-239) for ONE window,
// scaled by 1/sqrt(head_dim), softmax in f32 (here, double). `causal` is the
// FLASH family's `causal=self.is_causal` (vision_attention.py:265, :291, :302),
// which is the arm the released `attn_implementation = flash_attention_3`
// selects; the two eager classes store the flag and never read it, which is an
// upstream inconsistency this reference records rather than smooths over.
std::vector<double> Attention(const std::vector<double>& q,
                              const std::vector<double>& k,
                              const std::vector<double>& v, int64_t L,
                              int64_t nh, int64_t hd, bool causal) {
  std::vector<double> out(static_cast<size_t>(L * nh * hd), 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  for (int64_t h = 0; h < nh; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      const int64_t last = causal ? i : L - 1;
      std::vector<double> sc(static_cast<size_t>(L));
      double mx = -1e300;
      for (int64_t j = 0; j <= last; ++j) {
        double acc = 0.0;
        for (int64_t d = 0; d < hd; ++d) {
          acc += q[static_cast<size_t>((i * nh + h) * hd + d)] *
                 k[static_cast<size_t>((j * nh + h) * hd + d)];
        }
        sc[static_cast<size_t>(j)] = acc * scale;
        mx = std::max(mx, sc[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int64_t j = 0; j <= last; ++j) {
        sc[static_cast<size_t>(j)] = std::exp(sc[static_cast<size_t>(j)] - mx);
        sum += sc[static_cast<size_t>(j)];
      }
      for (int64_t j = 0; j <= last; ++j) {
        const double p = sc[static_cast<size_t>(j)] / sum;
        for (int64_t d = 0; d < hd; ++d) {
          out[static_cast<size_t>((i * nh + h) * hd + d)] +=
              p * v[static_cast<size_t>((j * nh + h) * hd + d)];
        }
      }
    }
  }
  return out;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// What the gate needs to say about a DISCRETE decision, computed on the
// reference's own side.
struct MoeRouteRef {
  int64_t block = 0;
  int64_t num_routed = 0;
  int64_t top_k = 0;
  // [L * top_k], each token's selected ids in ASCENDING order — a SET, because
  // `torch.topk(..., sorted=False)` leaves the order unspecified upstream and
  // the combine is a sum, so order carries no information and comparing it
  // would be comparing an artefact.
  std::vector<int64_t> ids;
  // min over tokens of (the k-th selected biased score) - (the best REJECTED
  // biased score). How much room the set assertion had.
  double min_margin = 1e300;
  // ...and the token it happened at, so a report can point at one row.
  int64_t min_margin_token = -1;
};

// `MoESwiGLUFFN.forward` (vision.py:170-218 @ 9035151d6), transcribed line by
// line. It calls nothing the implementation calls: its own sigmoid, its own
// selection scan, its own per-expert SwiGLU through `Linear` above, and its own
// self-normalizing accumulation.
//
// ON TIE-BREAKING. Upstream is `torch.topk`, whose order among equal scores is
// unspecified; this tree's house convention (`vt/ops.h`, the determinism
// deviation note above `MoeRouterTopK`) is a greedy strict-`>` scan over
// ascending index, so the LOWEST id wins an exact tie. The reference uses the
// same rule, and that is a statement about a case upstream does not define
// rather than the reference agreeing with the implementation about a case it
// does: at double precision over random weights no two biased scores here are
// equal, and the printed margin is what says so.
std::vector<double> MoeFfn(const TinySpec& s, const TinyCheckpoint& ck,
                           const std::string& pre, int64_t ne,
                           const std::vector<double>& x, int64_t L, int64_t E,
                           int64_t block, MoeRouteRef* route) {
  const int64_t Im = s.v_moe_inter;
  const double epsilon = 1e-9;
  // `topk = min(int(self.capacity_factor), self.num_routed)` (:190)
  const int64_t k = std::min<int64_t>(
      static_cast<int64_t>(s.v_capacity_factor), ne);
  // `gate_logits = F.linear(x_flat.float(), self.gate_weight.float())` (:180)
  const std::vector<double> gl =
      Linear(x, ck.value_of(pre + "mlp.gate_weight"), nullptr, L, E, ne);
  const std::vector<double>& rb = ck.value_of(pre + "mlp.router_bias");

  if (route != nullptr) {
    route->block = block;
    route->num_routed = ne;
    route->top_k = k;
    route->ids.assign(static_cast<size_t>(L * k), -1);
  }

  std::vector<double> agg_out(static_cast<size_t>(L * E), 0.0);
  std::vector<double> agg_gate(static_cast<size_t>(L), 0.0);
  for (int64_t t = 0; t < L; ++t) {
    std::vector<double> gp(static_cast<size_t>(ne));
    std::vector<double> gb(static_cast<size_t>(ne));
    for (int64_t j = 0; j < ne; ++j) {
      // `gating_prob = torch.sigmoid(gate_logits)` (:182-183) — ELEMENTWISE,
      // not normalized across experts.
      gp[static_cast<size_t>(j)] = Sigmoid(gl[static_cast<size_t>(t * ne + j)]);
      // `gating_with_bias = gating_prob + router_bias.float()` (:192)
      gb[static_cast<size_t>(j)] =
          gp[static_cast<size_t>(j)] + rb[static_cast<size_t>(j)];
    }
    // `torch.topk(gating_with_bias, k=topk, sorted=False)` (:193)
    std::vector<char> taken(static_cast<size_t>(ne), 0);
    std::vector<int64_t> sel;
    double last_selected = 1e300;
    for (int64_t r = 0; r < k; ++r) {
      int64_t best = -1;
      double best_v = -1e300;
      for (int64_t j = 0; j < ne; ++j) {
        if (taken[static_cast<size_t>(j)]) continue;
        if (best < 0 || gb[static_cast<size_t>(j)] > best_v) {
          best_v = gb[static_cast<size_t>(j)];
          best = j;
        }
      }
      taken[static_cast<size_t>(best)] = 1;
      sel.push_back(best);
      last_selected = best_v;
    }
    if (route != nullptr) {
      double best_rejected = -1e300;
      for (int64_t j = 0; j < ne; ++j) {
        if (taken[static_cast<size_t>(j)]) continue;
        best_rejected = std::max(best_rejected, gb[static_cast<size_t>(j)]);
      }
      // ne == k has no rejected expert and therefore no margin to report.
      if (best_rejected > -1e299) {
        const double m = last_selected - best_rejected;
        if (m < route->min_margin) {
          route->min_margin = m;
          route->min_margin_token = t;
        }
      }
      std::vector<int64_t> asc = sel;
      std::sort(asc.begin(), asc.end());
      for (int64_t r = 0; r < k; ++r)
        route->ids[static_cast<size_t>(t * k + r)] = asc[static_cast<size_t>(r)];
    }
    // `routed_weights = gating_prob.gather(1, topk_indices)` (:195) — the
    // UNBIASED score weights, the biased one only selected.
    std::vector<double> rw(static_cast<size_t>(k));
    double wsum = 0.0;
    for (int64_t r = 0; r < k; ++r) {
      rw[static_cast<size_t>(r)] = gp[static_cast<size_t>(sel[static_cast<size_t>(r)])];
      wsum += rw[static_cast<size_t>(r)];
    }
    // `if sigmoid and topk > 1: routed_weights /= (sum + epsilon)` (:196-199)
    if (s.v_router_scoring_func == "sigmoid" && k > 1) {
      for (int64_t r = 0; r < k; ++r) rw[static_cast<size_t>(r)] /= (wsum + epsilon);
    }
    // `routed_weights = routed_weights * self.router_scale` (:200)
    for (int64_t r = 0; r < k; ++r) rw[static_cast<size_t>(r)] *= s.v_router_scale;

    // `for expert_idx ...: aggregated_output[n] += expert(x[n]) * w;
    //  aggregated_gate[n] += w` (:202-213)
    for (int64_t r = 0; r < k; ++r) {
      const std::string ep =
          pre + "mlp.experts." + std::to_string(sel[static_cast<size_t>(r)]) + ".";
      const std::vector<double> row(x.begin() + static_cast<ptrdiff_t>(t * E),
                                    x.begin() + static_cast<ptrdiff_t>((t + 1) * E));
      // `DotsSwiGLUFFN.forward`: `fc2(F.silu(fc1(x)) * fc3(x))` (:136)
      const std::vector<double> g =
          Linear(row, ck.value_of(ep + "fc1.weight"), nullptr, 1, E, Im);
      const std::vector<double> u =
          Linear(row, ck.value_of(ep + "fc3.weight"), nullptr, 1, E, Im);
      std::vector<double> act(static_cast<size_t>(Im));
      for (int64_t c = 0; c < Im; ++c)
        act[static_cast<size_t>(c)] = Silu(g[static_cast<size_t>(c)]) *
                                      u[static_cast<size_t>(c)];
      const std::vector<double> o =
          Linear(act, ck.value_of(ep + "fc2.weight"), nullptr, 1, Im, E);
      for (int64_t c = 0; c < E; ++c)
        agg_out[static_cast<size_t>(t * E + c)] +=
            o[static_cast<size_t>(c)] * rw[static_cast<size_t>(r)];
      agg_gate[static_cast<size_t>(t)] += rw[static_cast<size_t>(r)];
    }
  }
  // `aggregated_output / (aggregated_gate.unsqueeze(-1) + epsilon)` (:215-217).
  // THE SELF-NORMALIZING DIVIDE, spelled as upstream spells it — by the SUMMED
  // gate rather than by the constant the implementation folds into
  // `vt::MoeCombine`'s `routed_scale`. Keeping the literal form here is what
  // makes the 1e-9 difference between the two a MEASUREMENT instead of a
  // definition.
  for (int64_t t = 0; t < L; ++t) {
    const double den = agg_gate[static_cast<size_t>(t)] + epsilon;
    for (int64_t c = 0; c < E; ++c) agg_out[static_cast<size_t>(t * E + c)] /= den;
  }
  return agg_out;
}

// `_pixel_shuffle(x, scale_factor=0.5)` (vision.py:401-416 @ 9035151d6) over a
// row-major [gh, gw, E] grid with BOTH sides even, written from the reshape /
// permute chain rather than from the closed form the implementation gathers by:
//   reshape(n,h,w/2,2c) -> permute(0,2,1,3) -> reshape(n,w/2,h/2,4c)
//   -> permute(0,2,1,3)
std::vector<double> PixelShuffle(const std::vector<double>& x, int64_t gh,
                                 int64_t gw, int64_t E) {
  // step A: [h, w/2, 2E]
  std::vector<double> a(static_cast<size_t>(gh * (gw / 2) * 2 * E));
  for (int64_t i = 0; i < gh; ++i)
    for (int64_t j = 0; j < gw / 2; ++j)
      for (int64_t c = 0; c < 2 * E; ++c)
        a[static_cast<size_t>((i * (gw / 2) + j) * 2 * E + c)] =
            x[static_cast<size_t>((i * gw + 2 * j + c / E) * E + c % E)];
  // step B: permute to [w/2, h, 2E]
  std::vector<double> b(a.size());
  for (int64_t i = 0; i < gh; ++i)
    for (int64_t j = 0; j < gw / 2; ++j)
      for (int64_t c = 0; c < 2 * E; ++c)
        b[static_cast<size_t>((j * gh + i) * 2 * E + c)] =
            a[static_cast<size_t>((i * (gw / 2) + j) * 2 * E + c)];
  // step C: reshape to [w/2, h/2, 4E]
  // step D: permute to [h/2, w/2, 4E]
  std::vector<double> out(b.size());
  for (int64_t j = 0; j < gw / 2; ++j)
    for (int64_t i = 0; i < gh / 2; ++i)
      for (int64_t c = 0; c < 4 * E; ++c)
        out[static_cast<size_t>((i * (gw / 2) + j) * 4 * E + c)] =
            b[static_cast<size_t>((j * gh + 2 * i + c / (2 * E)) * 2 * E +
                                  c % (2 * E))];
  return out;
}
// `nn.GELU()` with no `approximate=` is the EXACT erf gelu.
double GeluErf(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// `DotsMoEVitModel.forward` (vision.py:631-677), over BOTH block kinds and
// BOTH adapters. `routes`, when given, collects one entry per ROUTED block.
std::vector<double> Tower(const TinySpec& s, const TinyCheckpoint& ck,
                          const std::vector<double>& pixels, int64_t t,
                          int64_t gh, int64_t gw,
                          std::vector<MoeRouteRef>* routes = nullptr) {
  const int64_t E = s.v_embed, nh = s.v_heads, hd = s.v_head_dim();
  const int64_t L = t * gh * gw, P = s.v_patch_row(), VI = s.v_inter;
  const double eps = s.v_rms_eps;
  const std::string vp = "vision_encoder.";

  // patch_embed: the Conv2d over one non-overlapping patch IS a Linear over the
  // flattened patch row, then RMSNorm (vision.py:336-345).
  std::vector<double> hidden =
      Linear(pixels, ck.value_of(vp + "patch_embed.proj.weight"),
             &ck.value_of(vp + "patch_embed.proj.bias"), L, P, E);
  hidden = Rms(hidden, ck.value_of(vp + "patch_embed.norm.weight"), L, E, eps);

  const int64_t rope_merge =
      s.v_pre_pixel_shuffle ? (s.v_merge > 1 ? s.v_merge : 2) : 1;
  const std::vector<std::array<int64_t, 2>> pos = PosIds(t, gh, gw, rope_merge);

  for (int64_t b = 0; b < s.v_layers; ++b) {
    const std::string pre = vp + "blocks." + std::to_string(b) + ".";
    // `apply_vision_attention_residual`: hidden + attn(norm_1(hidden))
    const std::vector<double> n1 =
        Rms(hidden, ck.value_of(pre + "norm_1.weight"), L, E, eps);
    const std::vector<double> qkv =
        Linear(n1, ck.value_of(pre + "attn.qkv.weight"), nullptr, L, E, 3 * E);
    // `.reshape(L, 3, nh, -1).permute(1, 0, 2, 3).unbind(0)`
    std::vector<double> qh(static_cast<size_t>(L * E));
    std::vector<double> kh(static_cast<size_t>(L * E));
    std::vector<double> vh(static_cast<size_t>(L * E));
    for (int64_t l = 0; l < L; ++l) {
      for (int64_t c = 0; c < E; ++c) {
        qh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + c)];
        kh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + E + c)];
        vh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + 2 * E + c)];
      }
    }
    // Q/K NORM FIRST, ROPE SECOND (vision_attention.py:161-165). The order is
    // silent when swapped: same shapes, same magnitudes, different numbers.
    // `use_qk_norm` false builds no norm at all (:145-147).
    if (s.v_use_qk_norm) {
      qh = Rms(qh, ck.value_of(pre + "attn.q_norm.weight"), L * nh, hd, eps);
      kh = Rms(kh, ck.value_of(pre + "attn.k_norm.weight"), L * nh, hd, eps);
    }
    ApplyRope(&qh, pos, L, nh, hd);
    ApplyRope(&kh, pos, L, nh, hd);
    const std::vector<double> ao =
        Attention(qh, kh, vh, L, nh, hd, s.v_is_causal);
    const std::vector<double> proj =
        Linear(ao, ck.value_of(pre + "attn.proj.weight"), nullptr, L, E, E);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += proj[i];

    // hidden + mlp(norm_2(hidden)) (vision.py:394)
    const std::vector<double> n2 =
        Rms(hidden, ck.value_of(pre + "norm_2.weight"), L, E, eps);
    const bool moe = b < static_cast<int64_t>(s.v_pyramid.size()) &&
                     s.v_pyramid[static_cast<size_t>(b)] > 0;
    if (moe) {
      MoeRouteRef route;
      const std::vector<double> routed =
          MoeFfn(s, ck, pre, s.v_pyramid[static_cast<size_t>(b)], n2, L, E, b,
                 routes != nullptr ? &route : nullptr);
      if (routes != nullptr) routes->push_back(route);
      for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += routed[i];
    } else {
      // `fc2(silu(fc1(x)) * fc3(x))`
      const std::vector<double> g =
          Linear(n2, ck.value_of(pre + "mlp.fc1.weight"), nullptr, L, E, VI);
      const std::vector<double> u =
          Linear(n2, ck.value_of(pre + "mlp.fc3.weight"), nullptr, L, E, VI);
      std::vector<double> act(g.size());
      for (size_t i = 0; i < g.size(); ++i) act[i] = Silu(g[i]) * u[i];
      const std::vector<double> down =
          Linear(act, ck.value_of(pre + "mlp.fc2.weight"), nullptr, L, VI, E);
      for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += down[i];
    }
  }

  if (s.v_post_norm) {
    hidden = Rms(hidden, ck.value_of(vp + "post_trunk_norm.weight"), L, E, eps);
  }

  const int64_t M = s.v_merged_dim(), O = s.v_adapter_out();
  const int64_t Nm = L * E / M;
  if (s.v_adapter_type == "pixel_shuffle_mlp") {
    // `PixelShuffleAdapter.forward` (vision.py:439-461): shuffle FIRST, then a
    // LayerNorm over the MERGED width at torch's default eps of 1e-5, then
    // `Linear(M, O) / GELU / Linear(O, O)` (:432-437).
    const std::vector<double> sh = PixelShuffle(hidden, gh, gw, E);
    const std::vector<double> ln =
        LayerNorm(sh, ck.value_of(vp + "adapter.proj.0.weight"),
                  ck.value_of(vp + "adapter.proj.0.bias"), Nm, M, 1e-5);
    std::vector<double> p1 =
        Linear(ln, ck.value_of(vp + "adapter.proj.1.weight"),
               &ck.value_of(vp + "adapter.proj.1.bias"), Nm, M, O);
    for (double& x : p1) x = GeluErf(x);
    return Linear(p1, ck.value_of(vp + "adapter.proj.3.weight"),
                  &ck.value_of(vp + "adapter.proj.3.bias"), Nm, O, O);
  }

  // `PatchMergerAdapter.forward` (vision.py:488-496): ln_q over the per-token
  // dim at a HARD-CODED eps of 1e-6, then `reshape(-1, merged_dim)`, then the
  // two-layer MLP with an exact-erf GELU between.
  const std::vector<double> lnq =
      LayerNorm(hidden, ck.value_of(vp + "adapter.ln_q.weight"),
                ck.value_of(vp + "adapter.ln_q.bias"), L, E, 1e-6);
  std::vector<double> f1 =
      Linear(lnq, ck.value_of(vp + "adapter.mlp.0.weight"),
             &ck.value_of(vp + "adapter.mlp.0.bias"), Nm, M, M);
  for (double& x : f1) x = GeluErf(x);
  return Linear(f1, ck.value_of(vp + "adapter.mlp.2.weight"),
                &ck.value_of(vp + "adapter.mlp.2.bias"), Nm, M, O);
}

}  // namespace ref

// The whole loaded model, built once per case through the REAL registry over the
// REAL loader — never a hand-built weights struct.
struct Bench {
  TinySpec spec;
  TinyCheckpoint ckpt;
  HfConfig config;
  std::unique_ptr<vllm::LoadedModel> model;

  explicit Bench(TinySpec s = TinySpec{})
      : spec(s),
        ckpt(FixtureDir(), s),
        config(LoadHfConfig(ckpt.config_path())) {
    const std::vector<std::string> arch{"Dots3NoteForCausalLM"};
    const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.weights_path()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    model = reg.factory->load_weights(reg, config, source);
  }
};

// The bf16 patch rows the tower consumes, widened to double, so the reference
// and the implementation start from the SAME bytes and the comparison measures
// the forward rather than the processor.
std::vector<double> WidenBf16(const std::vector<uint16_t>& bits) {
  std::vector<double> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i)
    out[i] = static_cast<double>(vt::BF16ToF32(bits[i]));
  return out;
}

double MaxAbs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

// ONE tower run, both arms, off ONE fixture. Shared by every arithmetic case
// below so that a new config arm is one struct field rather than a fourth copy
// of the load / process / forward / reference sequence — four copies being how
// two of them end up measuring different models.
struct TowerRun {
  std::unique_ptr<Bench> bench;
  vllm::Dots3NoteVisionWeights weights;
  Dots3NoteVisionParams params;
  vllm::Dots3NoteVisionCapture capture;
  std::vector<float> ours;
  std::vector<double> want;
  std::vector<ref::MoeRouteRef> ref_routes;
  double rel = 0.0;
  double max_abs = 0.0;
  double scale = 0.0;
};

TowerRun RunTower(const TinySpec& spec, int image_variant = 0) {
  TowerRun r;
  r.bench = std::make_unique<Bench>(spec);
  // The processor is the PRODUCTION one, so the patch rows the tower sees are
  // the rows a served request would produce.
  const vllm::multimodal::Dots3NoteProcessorConfig pcfg =
      vllm::multimodal::LoadDots3NoteProcessorConfig(
          r.bench->ckpt.dir() + "/preprocessor_config.json",
          r.bench->ckpt.config_path(), "tiny-dots3");
  const vllm::multimodal::Dots3NoteImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb = dots3_tiny::FixtureImage(image_variant);
  const vllm::multimodal::ImageKwargs kw = proc.ProcessImage(
      rgb.data(), dots3_tiny::kImageSide, dots3_tiny::kImageSide);
  REQUIRE(kw.image_grid_thw[0] == 1);
  REQUIRE(kw.image_grid_thw[1] == 4);
  REQUIRE(kw.image_grid_thw[2] == 4);

  r.params = ParseDots3NoteVisionParams(r.bench->config);
  const std::string why = Dots3NoteVisionRefusal(r.params, "", {});
  REQUIRE_MESSAGE(why.empty(), "this fixture config refuses: " << why);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(r.bench->ckpt.weights_path()));
  r.weights = vllm::MaterializeDots3NoteVision(shards, r.params);
  REQUIRE(r.weights.present);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  r.ours = Dots3NoteVisionForward(kw.pixel_values_bf16, kw.image_grid_thw,
                                  r.weights, r.params, backend, &r.capture);
  r.want = ref::Tower(r.bench->spec, r.bench->ckpt,
                      WidenBf16(kw.pixel_values_bf16), 1, 4, 4, &r.ref_routes);
  REQUIRE(r.ours.size() == r.want.size());
  for (size_t i = 0; i < r.ours.size(); ++i)
    r.max_abs = std::max(r.max_abs,
                         std::abs(static_cast<double>(r.ours[i]) - r.want[i]));
  r.scale = MaxAbs(r.want);
  r.rel = r.scale > 0.0 ? r.max_abs / r.scale : r.max_abs;
  return r;
}

// A tiny tower whose SECOND block is a 4-expert pyramid block. One dense block
// ahead of it so a routed block that read a dense block's operand would move
// the answer, and 4 experts against top-2 so exactly half of them are rejected
// on every token — the smallest geometry in which a selection can be wrong.
TinySpec MoeSpec() {
  TinySpec s;
  s.v_pyramid = {-1, 4};
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE RELEASED `vision_config` resolves to the geometry the checkpoint's own
//    shard index carries. Every number here was read from the COMMITTED
//    fixture, not from the issue text.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the RELEASED vision_config resolves to the measured geometry") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  REQUIRE(v.present);
  CHECK(v.embed_dim == 1536);
  CHECK(v.num_attention_heads == 24);
  CHECK(v.head_dim() == 64);
  CHECK(v.num_hidden_layers == 42);
  CHECK(v.intermediate_size == 4224);
  CHECK(v.moe_intermediate_size == 2112);
  CHECK(v.patch_size == 14);
  CHECK(v.temporal_patch_size == 1);
  CHECK(v.spatial_merge_size == 2);
  CHECK(v.rms_norm_eps == doctest::Approx(1e-5));
  CHECK_FALSE(v.use_bias);
  CHECK(v.use_qk_norm);
  CHECK_FALSE(v.is_causal);
  CHECK(v.post_norm);

  // THE TWO FLAGS #2512's PROSE CONFLATES, asserted apart. `adapter_type` is
  // `patch_merger`, which is upstream's name for the arm that SKIPS the
  // pixel-shuffle permutation (vision.py:465-471 @ 9035151d6, its docstring); the 2x2
  // regrouping did not disappear, `pre_pixel_shuffle` moved it into the
  // PREPROCESSOR and into the RoPE. The issue's own tensor inventory —
  // `adapter.{ln_q, mlp.0, mlp.2}` — is `PatchMergerAdapter`'s state dict and
  // agrees with this; `PixelShuffleAdapter` spells its parameters
  // `proj.0`/`proj.1`/`proj.3` (vision.py:423, :432-437). See spec §4.11.1.
  CHECK(v.adapter_type == "patch_merger");
  CHECK(v.pre_pixel_shuffle);
  CHECK(v.adapter_in_dim == 1536);
  CHECK(v.adapter_out_dim == 5120);
  CHECK(v.adapter_merge_size == 2);
  CHECK(v.merged_dim() == 6144);  // 4 x 1536

  // 25 dense + 17 MoE, counted from `pyramid_num_routed` rather than assumed:
  // `is_moe` is `> 0` (vision.py:363-366), so the leading -1s are DENSE.
  REQUIRE(v.pyramid_num_routed.size() == 42u);
  CHECK(v.num_dense_blocks() == 25);
  CHECK(v.num_moe_blocks() == 17);
  CHECK(v.is_moe_block(24) == false);
  CHECK(v.is_moe_block(25) == true);
  CHECK(v.pyramid_num_routed[25] == 4);
  CHECK(v.pyramid_num_routed[41] == 64);
  int64_t experts = 0;
  for (int64_t i = 25; i < 42; ++i) experts += v.pyramid_num_routed[static_cast<size_t>(i)];
  // 1960 of the 2195 vision tensors: 17 x 8 block tensors + 608 x 3 experts.
  CHECK(experts == 608);
  CHECK(17 * 8 + experts * 3 == 1960);
}

// ---------------------------------------------------------------------------
// 2. THE RELEASED CHECKPOINT NO LONGER REFUSES (W6b, #2613).
//
//    W6a returned a message here naming block 25 and W6b. This case is the
//    inverse of that one and it is the headline of this brick: the released
//    `vision_config` — 25 dense blocks, 17 pyramid blocks, 608 routed experts,
//    sigmoid scoring, capacity factor 2 — is ACCEPTED, so its tower computes.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: the RELEASED vision tower is ACCEPTED, pyramid and all") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  const std::string why = Dots3NoteVisionRefusal(v, "", {});
  INFO("refusal: ", why);
  CHECK(why.empty());
  // The PREMISE, asserted rather than assumed: it really is a pyramid tower.
  // A config that had quietly become all-dense would also be accepted here and
  // would say nothing at all about W6b.
  REQUIRE(v.num_moe_blocks() == 17);
  CHECK(v.routed_top_k(25) == 2);   // min(int(2.0), 4)
  CHECK(v.routed_top_k(41) == 2);   // min(int(2.0), 64)
  CHECK(v.routed_top_k(24) == 0);   // dense blocks have no router
  CHECK(v.router_scoring_func == "sigmoid");
  CHECK(v.router_scale == doctest::Approx(1.0));
  CHECK(v.capacity_factor == doctest::Approx(2.0));
  CHECK(v.moe_intermediate_size == 2112);
}

TEST_CASE("dots3-note W6b: every unported vision shape refuses BY NAME with its brick") {
  const nlohmann::json released = ReleasedConfigDoc();

  SUBCASE("the RELEASED config is accepted — the premise of every case below") {
    CHECK(Dots3NoteVisionRefusal(ParseDoc(released), "", {}).empty());
  }
  SUBCASE("the BLOCKWISE-FP8 arm is W9, and it outranks everything") {
    const std::string why =
        Dots3NoteVisionRefusal(ParseDoc(released), "fp8", {128, 128});
    INFO(why);
    CHECK(why.find("W9") != std::string::npos);
    CHECK(why.find("weight_block_size") != std::string::npos);
  }
  SUBCASE("`pixel_shuffle_mlp` is now IMPLEMENTED and is accepted") {
    nlohmann::json d = released;
    d["vision_config"]["adapter_type"] = "pixel_shuffle_mlp";
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK(why.empty());
  }
  SUBCASE("...but only at merge size 2, because `_pixel_shuffle` hard-codes it") {
    nlohmann::json d = released;
    d["vision_config"]["adapter_type"] = "pixel_shuffle_mlp";
    d["vision_config"]["adapter_merge_size"] = 4;
    d["vision_config"]["spatial_merge_size"] = 4;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("scale_factor=0.5") != std::string::npos);
  }
  SUBCASE("an UNKNOWN adapter refuses at PARSE, not at load") {
    nlohmann::json d = released;
    d["vision_config"]["adapter_type"] = "something_else";
    CHECK_THROWS_AS((void)ParseDoc(d), std::runtime_error);
  }
  SUBCASE("`temporal_patch_size != 1` is the VIDEO arm and names W7") {
    nlohmann::json d = released;
    d["vision_config"]["temporal_patch_size"] = 2;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK(why.find("W7") != std::string::npos);
  }
  // THE THREE ARMS W6a DEFERRED AND W6b LIFTS. Each was a refusal message on
  // `main` at `3d045ba1b`; each is now a computed path, and the arithmetic
  // cases below measure them against the reference.
  SUBCASE("`post_norm`, `use_qk_norm` and `is_causal` are LIFTED, not refused") {
    for (const char* key : {"post_norm", "use_qk_norm", "is_causal"}) {
      nlohmann::json d = released;
      d["vision_config"][key] = !d["vision_config"][key].get<bool>();
      const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
      INFO("key ", key, " -> '", why, "'");
      CHECK_MESSAGE(why.empty(), "flipping " << key << " still refuses: " << why);
    }
  }
  // ...AND THE ONE IT DOES NOT. `use_bias` is refused with its reason and its
  // issue, because the shared `MlpGateUpMethodBase` seam has no bias operand
  // and no published dots3-note checkpoint sets the key.
  SUBCASE("`use_bias` still refuses, and names the issue that owns it") {
    nlohmann::json d = released;
    d["vision_config"]["use_bias"] = true;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("use_bias") != std::string::npos);
    CHECK(why.find("MlpGateUpMethodBase") != std::string::npos);
    CHECK(why.find("#2616") != std::string::npos);
  }
  // THE TWO ROUTER ARMS W6b DOES NOT SERVE, and why they are a pair: on both
  // of them upstream skips the weight renormalization, which leaves the
  // combine's `aggregated_gate` denominator per-token.
  SUBCASE("a SOFTMAX router refuses, naming issue #2615") {
    nlohmann::json d = released;
    d["vision_config"]["router_scoring_func"] = "softmax";
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("router_scoring_func") != std::string::npos);
    CHECK(why.find("#2615") != std::string::npos);
  }
  SUBCASE("a top-k below 2 refuses, naming issue #2615") {
    nlohmann::json d = released;
    d["vision_config"]["capacity_factor"] = 1;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("top-1") != std::string::npos);
    CHECK(why.find("#2615") != std::string::npos);
  }
  SUBCASE("...and an ALL-DENSE tower with the same capacity_factor does NOT") {
    // The refusal is per ROUTED block, so a tower with no router is untouched
    // by it. Without this case the top-k refusal could be a blanket one on
    // `capacity_factor` and read the same.
    nlohmann::json d = released;
    d["vision_config"]["capacity_factor"] = 1;
    for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
    CHECK(Dots3NoteVisionRefusal(ParseDoc(d), "", {}).empty());
  }
  // THE THREE CONDITIONS THE ENCODER ASSERTS ON, asked HERE too. Before the
  // fresh review of #2523 the refusal was a strict SUBSET of
  // `EncodeMmDots3NoteForCausalLM`'s `VT_CHECK`s, and the gap was reachable
  // from an all-dense config the seam ACCEPTED — after which the throw lands in
  // the engine's busy loop and stops `AsyncLLM` for the life of the process.
  // The refusal predicate and the route predicate must be the SAME predicate.
  SUBCASE("`adapter_out_dim` that is not the TEXT hidden_size refuses") {
    nlohmann::json d = released;
    d["vision_config"]["adapter_out_dim"] = d["hidden_size"].get<int64_t>() + 8;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("adapter_out_dim") != std::string::npos);
    CHECK(why.find("hidden_size") != std::string::npos);
  }
  SUBCASE("the TEXT hidden_size is what it compares against, not vision_config's") {
    // The two documents' copies of the number are made to DISAGREE. The
    // encoder's assert reads `config.hidden_size`, so a refusal that read
    // `vision_config.hidden_size` instead would accept the left case and refuse
    // the right one — both of them backwards.
    nlohmann::json d = released;
    d["vision_config"]["hidden_size"] = d["hidden_size"].get<int64_t>() + 8;
    CHECK(Dots3NoteVisionRefusal(ParseDoc(d), "", {}).empty());

    nlohmann::json e2 = released;
    e2["hidden_size"] = e2["hidden_size"].get<int64_t>() + 8;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(e2), "", {});
    INFO(why);
    CHECK_FALSE(why.empty());
    CHECK(why.find("adapter_out_dim") != std::string::npos);
  }
  SUBCASE("`adapter_merge_size` that is not `spatial_merge_size` refuses") {
    // BOTH directions, because they reach DIFFERENT asserts on a served
    // request: 1 makes the tower emit more rows than the placeholder span holds
    // (`rows == item.length`), and 3 makes the trunk length not group into
    // whole merger rows at all (`L % merge_unit == 0`). One refusal covers
    // both, because both are the same disagreement between two keys.
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      nlohmann::json d = released;
      d["vision_config"]["adapter_merge_size"] = m;
      const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
      INFO("adapter_merge_size ", m, " -> ", why);
      CHECK_FALSE(why.empty());
      CHECK(why.find("adapter_merge_size") != std::string::npos);
      CHECK(why.find("spatial_merge_size") != std::string::npos);
    }
  }
  SUBCASE("a checkpoint with NO vision_config refuses, naming the absence") {
    nlohmann::json d = released;
    d.erase("vision_config");
    const Dots3NoteVisionParams v = ParseDoc(d);
    CHECK_FALSE(v.present);
    const std::string why = Dots3NoteVisionRefusal(v, "", {});
    CHECK(why.find("vision_config") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// 3. ALL 2195 VISION TENSORS, counted against the COMMITTED shard index.
//    W6a claimed 235 of them and left 1960 deferred; W6b claims the lot. The
//    counts are asserted BY NUMBER and cross-checked against the released
//    index's own arithmetic, because "nothing was left over" is also true of a
//    map that claimed a pyramid block as dense.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: the tower claims ALL 2195 of the released tower's tensors") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteVisionTensors(v);
  CHECK(claimed.size() == 2195u);

  // 25 dense x 9 + 3 patch_embed + 1 post_trunk_norm + 6 adapter = W6a's 235.
  CHECK(25 * 9 + 3 + 1 + 6 == 235);
  // ...and the pyramid's 1960: 17 blocks x 8 block tensors (norm_1, norm_2,
  // qkv, proj, q_norm, k_norm, gate_weight, router_bias) + 608 routed experts
  // x 3. That is #2613's own arithmetic and the released index's.
  int64_t experts = 0;
  for (int64_t i = 25; i < 42; ++i)
    experts += v.pyramid_num_routed[static_cast<size_t>(i)];
  CHECK(experts == 608);
  CHECK(17 * 8 + experts * 3 == 1960);
  CHECK(235 + 1960 == 2195);

  std::set<std::string> names;
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK_MESSAGE(!t.consumer.empty(), t.name << " has no named consumer");
    CHECK_MESSAGE(names.insert(t.name).second, t.name << " is claimed twice");
    CHECK(t.name.rfind("vision_encoder.", 0) == 0);
  }

  // THE ROUTER SPELLING, asserted by name. `mlp.gate_weight` +
  // `mlp.router_bias` is the VISION router (vision.py:152-168 @ 9035151d6); the
  // LANGUAGE tower's is `mlp.gate.weight` + `mlp.gate.e_score_correction_bias`
  // (deepseek_v2.py:313-318). Claiming the language spelling here would find no
  // tensor on disk and refuse the load for the wrong reason, so the two are
  // asserted apart rather than assumed distinct.
  for (int64_t b = 25; b < 42; ++b) {
    const std::string pre = "vision_encoder.blocks." + std::to_string(b) + ".";
    CHECK_MESSAGE(names.count(pre + "mlp.gate_weight") == 1,
                  "block " << b << " is routed and its router is unclaimed");
    CHECK_MESSAGE(names.count(pre + "mlp.router_bias") == 1,
                  "block " << b << " is routed and its router bias is unclaimed");
    CHECK(names.count(pre + "mlp.gate.weight") == 0);
    CHECK(names.count(pre + "mlp.gate.e_score_correction_bias") == 0);
    // ...and NOT the dense spelling, whose tensors do not exist on a routed
    // block: 17 x 3 = 51 names that would refuse the load.
    CHECK(names.count(pre + "mlp.fc1.weight") == 0);
    CHECK(names.count(pre + "mlp.fc3.weight") == 0);
    // The LAST expert of the block, so a loop that stopped one short shows.
    const int64_t ne = v.pyramid_num_routed[static_cast<size_t>(b)];
    CHECK(names.count(pre + "mlp.experts." + std::to_string(ne - 1) +
                      ".fc2.weight") == 1);
    CHECK(names.count(pre + "mlp.experts." + std::to_string(ne) +
                      ".fc2.weight") == 0);
  }
  // ...and every DENSE block still carries the dense spelling and no router.
  for (int64_t b = 0; b < 25; ++b) {
    const std::string pre = "vision_encoder.blocks." + std::to_string(b) + ".";
    CHECK_MESSAGE(names.count(pre + "mlp.fc3.weight") == 1,
                  "block " << b << " is dense and its fc3 is unclaimed");
    CHECK(names.count(pre + "mlp.gate_weight") == 0);
  }
  CHECK(names.count("vision_encoder.adapter.ln_q.bias") == 1);
  CHECK(names.count("vision_encoder.adapter.mlp.2.weight") == 1);
  CHECK(names.count("vision_encoder.post_trunk_norm.weight") == 1);
  CHECK(names.count("vision_encoder.patch_embed.proj.bias") == 1);

  // THE CONFIG ARMS CHANGE WHAT IS CLAIMED, which is what makes them arms
  // rather than flags nothing reads.
  SUBCASE("`use_qk_norm` false drops two tensors PER BLOCK, all 42 of them") {
    nlohmann::json d = ReleasedConfigDoc();
    d["vision_config"]["use_qk_norm"] = false;
    const std::vector<vllm::Dots3NoteTensor> c2 =
        vllm::EnumerateDots3NoteVisionTensors(ParseDoc(d));
    CHECK(c2.size() == 2195u - 2u * 42u);
  }
  SUBCASE("`post_norm` false drops exactly one") {
    nlohmann::json d = ReleasedConfigDoc();
    d["vision_config"]["post_norm"] = false;
    const std::vector<vllm::Dots3NoteTensor> c2 =
        vllm::EnumerateDots3NoteVisionTensors(ParseDoc(d));
    CHECK(c2.size() == 2194u);
  }
  SUBCASE("`pixel_shuffle_mlp` claims proj.0/1/3 and NOT ln_q/mlp.0/mlp.2") {
    nlohmann::json d = ReleasedConfigDoc();
    d["vision_config"]["adapter_type"] = "pixel_shuffle_mlp";
    std::set<std::string> n2;
    for (const vllm::Dots3NoteTensor& t :
         vllm::EnumerateDots3NoteVisionTensors(ParseDoc(d)))
      n2.insert(t.name);
    CHECK(n2.size() == 2195u);
    CHECK(n2.count("vision_encoder.adapter.proj.0.weight") == 1);
    CHECK(n2.count("vision_encoder.adapter.proj.1.bias") == 1);
    CHECK(n2.count("vision_encoder.adapter.proj.3.weight") == 1);
    CHECK(n2.count("vision_encoder.adapter.ln_q.weight") == 0);
    CHECK(n2.count("vision_encoder.adapter.mlp.0.weight") == 0);
    CHECK(n2.count("vision_encoder.adapter.mlp.2.weight") == 0);
  }
}

// ---------------------------------------------------------------------------
// 4. THE POSITION GRID. `pre_pixel_shuffle` selects between two DIFFERENT token
//    orders, and the flag is read by the processor AND by the tower — so a case
//    that only checked one order could not see the two disagreeing.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: pre_pixel_shuffle regroups the RoPE positions, and NOT setting it does not") {
  TinySpec s;
  const std::array<int64_t, 3> grid{1, 4, 4};

  s.v_pre_pixel_shuffle = true;
  Dots3NoteVisionParams grouped = ParseDoc(dots3_tiny::TinyConfigDoc(FixtureDir(), s));
  s.v_pre_pixel_shuffle = false;
  Dots3NoteVisionParams flat = ParseDoc(dots3_tiny::TinyConfigDoc(FixtureDir(), s));

  const auto ours_grouped = Dots3NoteVisionPosIds(grid, grouped);
  const auto ours_flat = Dots3NoteVisionPosIds(grid, flat);
  const auto ref_grouped = ref::PosIds(1, 4, 4, /*rope_merge=*/2);
  const auto ref_flat = ref::PosIds(1, 4, 4, /*rope_merge=*/1);

  REQUIRE(ours_grouped.size() == 16u);
  CHECK(ours_grouped == ref_grouped);
  CHECK(ours_flat == ref_flat);
  // THE PREMISE, asserted rather than assumed: the two orders really differ, so
  // agreeing with the wrong one is a detectable defect rather than a tie.
  CHECK(ours_grouped != ours_flat);
  // Spot the grouped layout by hand: token 1 is the 2x2 block's top-RIGHT, so
  // it is row 0 column 1, where the flat order would put row 0 column 1 too but
  // token 2 apart — flat token 2 is (0, 2), grouped token 2 is (1, 0).
  CHECK(ours_grouped[2] == std::array<int64_t, 2>{1, 0});
  CHECK(ours_flat[2] == std::array<int64_t, 2>{0, 2});
}

// ---------------------------------------------------------------------------
// 5. THE TOWER, against the independent double-precision reference.
//    THE CONSISTENCY GATE. It says two implementations agree; it does not say
//    either matches vLLM, because vLLM cannot be run on this model here
//    (spec §6.4 option B). No performance number is claimable on any axis.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the DENSE tower agrees with an INDEPENDENT double reference") {
  TowerRun r = RunTower(TinySpec{});

  // THE MEMORY FORMAT (porting.md). A widened store passes every shape check
  // and every token gate while moving twice the bytes, and this row's W2 F1
  // fixture row already proves a re-typed tensor fires.
  CHECK(r.weights.patch_proj_w.dtype == vt::DType::kBF16);
  CHECK(r.weights.adapter_mlp2_w.dtype == vt::DType::kBF16);
  for (const auto& blk : r.weights.blocks) {
    CHECK(blk.qkv.dtype == vt::DType::kBF16);
    CHECK(blk.gate_up.dtype == vt::DType::kBF16);
  }
  // The merge is real: [2I, E], gate then up.
  REQUIRE(r.weights.blocks.size() == static_cast<size_t>(r.bench->spec.v_layers));
  CHECK(r.weights.blocks[0].gate_up.shape[0] == 2 * r.bench->spec.v_inter);
  CHECK(r.weights.blocks[0].gate_up.shape[1] == r.bench->spec.v_embed);
  // No routed block here, so nothing was captured — the premise of the case.
  CHECK(r.capture.moe_routes.empty());

  // FOUR merger rows (16 patches / 2x2), each in the TEXT hidden space.
  CHECK(r.ours.size() == static_cast<size_t>(dots3_tiny::kExpectedImageTokens *
                                             r.bench->spec.hidden));
  MESSAGE("dense tower vs double reference: max |diff| ", r.max_abs,
          " over a scale of ", r.scale, " => relative ", r.rel);
  // THE BOUND, and where it comes from. The implementation stores every
  // activation and every weight in bf16 (8 mantissa bits, ~3.9e-3 relative) and
  // runs two blocks plus a 64-wide adapter GEMM over it; the reference is
  // double throughout and also absorbs upstream's intermediate `.type_as(x)`
  // cast, which is a no-op at infinite precision. The bound is set at a
  // MEASURED multiple of the observed deviation rather than at a round number,
  // and the mutation evidence in the spec is what proves it is still tight
  // enough to see a defect — a bound above the real error is a mute switch.
  // MEASURED 2026-09-01 on this fixture: max |diff| 0.0533 over a scale of
  // 6.314, i.e. 8.44e-3 relative. The bound is 0.02 — a 2.4x margin over the
  // observation, wide enough that a different libm or a different GEMM
  // reduction order does not red it, and tight enough that the mutations
  // recorded in spec §4.11 all exceed it.
  CHECK(r.rel < 0.02);
  // ...and the two are not trivially equal, which would mean one of them is
  // reading the other's answer.
  CHECK(r.scale > 1e-3);
}

// ---------------------------------------------------------------------------
// 5b. THE PYRAMID TOWER (W6b, #2613), against the same reference — AND against
//     a gate shape the dense arm did not need.
//
//     A TOLERANCE ALONE IS A MUTE SWITCH HERE. Top-k expert selection is a
//     DISCRETE choice: the error it makes is bimodal, not continuous. Either
//     the same experts were chosen and the output error is the ordinary bf16
//     one, or a different expert was chosen and the output is a different
//     function — and in between there is nothing for a relative bound to
//     measure. Worse, a selection defect that happens NOT to flip on this
//     fixture leaves the tolerance green while saying nothing at all. So the
//     selection is asserted as a SET, per token, against the reference's own
//     independent scan, and the minimum decision MARGIN is printed so the
//     reader knows how much room that assertion had.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: the PYRAMID tower agrees with the reference, and its SELECTION does too") {
  TowerRun r = RunTower(MoeSpec());

  // ── the weights, and their MEMORY FORMAT ─────────────────────────────────
  REQUIRE(r.weights.blocks.size() == 2u);
  CHECK_FALSE(r.weights.blocks[0].is_moe);
  REQUIRE(r.weights.blocks[1].is_moe);
  const vllm::Dots3NoteVisionMoeWeights& m = r.weights.blocks[1].moe;
  CHECK(m.num_routed == 4);
  CHECK(m.top_k == 2);
  REQUIRE(m.expert_gate.size() == 4u);
  REQUIRE(m.expert_up.size() == 4u);
  REQUIRE(m.expert_down.size() == 4u);
  // The router weight is BF16 and the router BIAS is F32, which is upstream's
  // own asymmetry (`vision.py:152-154` against `:165-168` @ 9035151d6) and the
  // whole reason the released tower has 17 F32 tensors against 2178 BF16 ones.
  // A token gate cannot see a dtype that is too wide OR too narrow; this row's
  // W2 F1 fixture row proves a re-typed tensor fires.
  CHECK(m.gate_weight.dtype == vt::DType::kBF16);
  CHECK(m.router_bias.dtype == vt::DType::kF32);
  CHECK(m.gate_weight.shape[0] == 4);
  CHECK(m.gate_weight.shape[1] == r.bench->spec.v_embed);
  CHECK(m.router_bias.shape[0] == 4);
  for (size_t e = 0; e < 4u; ++e) {
    CHECK(m.expert_gate[e].dtype == vt::DType::kBF16);
    CHECK(m.expert_up[e].dtype == vt::DType::kBF16);
    CHECK(m.expert_down[e].dtype == vt::DType::kBF16);
    // SPLIT operands, [Im, E] each — not a merged [2Im, E]. The merge would
    // cost 7.9 GiB of resident copy on the released checkpoint.
    CHECK(m.expert_gate[e].shape[0] == r.bench->spec.v_moe_inter);
    CHECK(m.expert_gate[e].shape[1] == r.bench->spec.v_embed);
    CHECK(m.expert_down[e].shape[0] == r.bench->spec.v_embed);
    CHECK(m.expert_down[e].shape[1] == r.bench->spec.v_moe_inter);
  }

  // ── THE DISCRETE ASSERTION ───────────────────────────────────────────────
  REQUIRE(r.capture.moe_routes.size() == 1u);
  REQUIRE(r.ref_routes.size() == 1u);
  const vllm::Dots3NoteVisionMoeRoute& ours = r.capture.moe_routes[0];
  const ref::MoeRouteRef& want = r.ref_routes[0];
  CHECK(ours.block == 1);
  CHECK(want.block == 1);
  CHECK(ours.num_routed == want.num_routed);
  CHECK(ours.top_k == want.top_k);
  const int64_t L = 16, K = ours.top_k;
  REQUIRE(ours.ids.size() == static_cast<size_t>(L * K));
  REQUIRE(want.ids.size() == static_cast<size_t>(L * K));

  // Formats a selection of ANY width. The old form indexed `[0]` and `[1]`
  // directly while `top_k == 2` was only a `CHECK`, so a top-k that came back
  // 1 would have read out of bounds inside the failure message of the
  // assertion that was about to report it.
  const auto join = [](const std::vector<int64_t>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i != 0) out += ", ";
      out += std::to_string(v[i]);
    }
    return out + "}";
  };
  REQUIRE(K == 2);

  int64_t agreed = 0;
  std::set<int64_t> distinct;
  std::set<std::vector<int64_t>> distinct_sets;
  std::vector<int64_t> load(static_cast<size_t>(ours.num_routed), 0);
  for (int64_t t = 0; t < L; ++t) {
    std::vector<int64_t> mine;
    for (int64_t j = 0; j < K; ++j)
      mine.push_back(ours.ids[static_cast<size_t>(t * K + j)]);
    std::sort(mine.begin(), mine.end());
    std::vector<int64_t> theirs(
        want.ids.begin() + static_cast<ptrdiff_t>(t * K),
        want.ids.begin() + static_cast<ptrdiff_t>((t + 1) * K));
    for (int64_t e : mine) {
      distinct.insert(e);
      load[static_cast<size_t>(e)]++;
    }
    distinct_sets.insert(mine);
    // SET equality, per token. `torch.topk(..., sorted=False)` leaves the
    // ORDER unspecified upstream and the combine is a sum, so the set is the
    // decision and the order is an artefact.
    CHECK_MESSAGE(mine == theirs, "token " << t << " selected " << join(mine)
                                           << " against the reference's "
                                           << join(theirs));
    if (mine == theirs) ++agreed;
  }
  CHECK(agreed == L);

  // THE INSTRUMENT'S OWN PRECONDITION, and it is asserted on the two axes that
  // decide how much of the router the SET assertion above actually exercises.
  //
  // WHY IT IS NOT `distinct >= 3`, which is what this block used to say. On the
  // fixture as first landed, 13 of 16 tokens routed to {1, 2} and the other 3
  // to {2, 3}: expert 2 sat in EVERY set, **expert 0 was never selected at
  // all**, and only three of the sixteen tokens could distinguish one set from
  // another. `distinct >= 3` passed with ZERO slack against a spread of exactly
  // 3, so the weakest assertion in the file guarded the strongest one. Expert 0
  // is also the index an off-by-one lands on, so leaving it unrouted is the one
  // omission that matters most.
  //
  // The fixture's `v_router_seed_nudge` was searched until both hold, and both
  // are asserted rather than printed:
  //
  //   1. EVERY routed expert is selected by some token — a per-expert load
  //      floor of 1, not a count of distinct ids with a margin of nothing.
  //   2. More than two of the `C(4,2) = 6` possible pairs occur, so a router
  //      that collapsed onto one or two pairs regardless of its input could not
  //      reach this population.
  //
  // MEASURED 2026-09-03 at nudge 42: all 4 experts, per-expert loads
  // 7/4/9/12 over 32 slots, and all SIX pairs present.
  std::string loads;
  for (size_t e = 0; e < load.size(); ++e)
    loads += (e ? "/" : "") + std::to_string(load[e]);
  MESSAGE("routed experts touched over ", L, " tokens: ", distinct.size(),
          " of ", ours.num_routed, "; per-expert load ", loads,
          "; distinct selection SETS ", distinct_sets.size(), " of the ",
          ours.num_routed * (ours.num_routed - 1) / 2, " possible pairs");
  CHECK(distinct.size() == static_cast<size_t>(ours.num_routed));
  for (size_t e = 0; e < load.size(); ++e)
    CHECK_MESSAGE(load[e] >= 1, "expert " << e << " was never selected");
  CHECK(distinct_sets.size() > 2u);
  // THE MARGIN, printed rather than assumed. This is how much room the set
  // assertion had: the gap between the last SELECTED biased score and the best
  // REJECTED one, minimised over tokens. A margin at zero would mean the
  // fixture decides its routing by a tie and the agreement is luck.
  MESSAGE("minimum decision margin over ", L, " tokens: ", want.min_margin,
          " at token ", want.min_margin_token,
          " (biased-score gap between the last selected and the first "
          "rejected expert)");
  CHECK(want.min_margin > 0.0);
  // ...and above the drift the two routers can differ by. MEASURED 2026-09-03
  // at nudge 42: 1.26e-2 at token 8. The implementation's logits come from a
  // bf16-operand GEMM with an f32 accumulator over a 16-wide reduction, so they
  // sit within ~1e-3 relative of the reference's double ones; through the
  // sigmoid, whose slope is at most 1/4, that is ~2.5e-4 of score. The margin
  // is ~50x that, which is the number that says the agreement above is a
  // decision and not a coin toss.
  //
  // IT IS ALSO 3x THE 4.01e-3 THE FIXTURE USED TO REPORT, and that is a REAL
  // cost paid for the coverage asserted above, not an improvement. A SMALL
  // margin is the useful direction here: it means the fixture sits near the
  // decision boundary, so a selection defect has somewhere to show. The trade
  // was taken because the old fixture bought its 4.01e-3 by routing 13 of 16
  // tokens to one pair and never selecting expert 0 at all — a tight margin on
  // a population that could not discriminate. 1.26e-2 over all four experts and
  // all six pairs is the better instrument, and the number is recorded here so
  // the direction of the trade is visible rather than inferred.
  CHECK(want.min_margin > 1e-3);

  // ── the router weights, and the output ───────────────────────────────────
  // The weights are f32 as the op contract requires, and they sum to
  // `router_scale` per token, which is the renormalization at vision.py:196-200
  // and the reason the self-normalizing combine's denominator is a CONSTANT.
  REQUIRE(ours.weights.size() == static_cast<size_t>(L * K));
  for (int64_t t = 0; t < L; ++t) {
    double sum = 0.0;
    for (int64_t j = 0; j < K; ++j)
      sum += ours.weights[static_cast<size_t>(t * K + j)];
    CHECK(sum == doctest::Approx(r.bench->spec.v_router_scale).epsilon(1e-5));
  }

  MESSAGE("pyramid tower vs double reference: max |diff| ", r.max_abs,
          " over a scale of ", r.scale, " => relative ", r.rel);
  // MEASURED 2026-09-03 at nudge 42: max |diff| 0.0612 over a scale of
  // 6.045, i.e. 1.01e-2 relative — the same order as the dense tower's
  // 8.44e-3, which is what one expects when the selection agrees and the only
  // difference left is bf16 storage. The bound is the dense case's 0.02.
  CHECK(r.rel < 0.02);
  CHECK(r.scale > 1e-3);
  CHECK(r.ours.size() == static_cast<size_t>(dots3_tiny::kExpectedImageTokens *
                                             r.bench->spec.hidden));
}

// ---------------------------------------------------------------------------
// 5c. A PYRAMID BLOCK IS NOT A DENSE BLOCK. The case above proves the routed
//     arm agrees with the reference; this one proves the routed arm is a
//     DIFFERENT FUNCTION from the dense one, so that agreement is not an
//     accident of a tower where the branch did not matter.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: routing block 1 changes the tower's answer") {
  TowerRun dense = RunTower(TinySpec{});
  TowerRun moe = RunTower(MoeSpec());
  REQUIRE(dense.ours.size() == moe.ours.size());
  double d = 0.0;
  for (size_t i = 0; i < dense.ours.size(); ++i)
    d = std::max(d, std::abs(static_cast<double>(dense.ours[i]) -
                             static_cast<double>(moe.ours[i])));
  MESSAGE("dense block 1 against routed block 1: max |diff| ", d);
  CHECK(d > 1e-2);
}

// ---------------------------------------------------------------------------
// 5d. THE FOUR CONFIG ARMS W6a DEFERRED AND W6b LIFTED. Each one is run over
//     the SAME reference, on a tower that also has a pyramid block, so an arm
//     that only worked on a dense tower would show.
//
//     Each arm is also asserted to CHANGE the answer. An arm that computed the
//     same numbers as the default would agree with a reference that read the
//     same flag and prove nothing about either.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: post_norm, use_qk_norm, is_causal and pixel_shuffle_mlp all compute") {
  const TowerRun base = RunTower(MoeSpec());

  SUBCASE("`post_norm` false skips the trunk norm and ships no tensor for it") {
    TinySpec s = MoeSpec();
    s.v_post_norm = false;
    TowerRun r = RunTower(s);
    MESSAGE("post_norm=false vs reference: relative ", r.rel);
    // MEASURED 2026-09-03 at nudge 42: 6.59e-3. The bound follows the same
    // rule the dense case states — a measured multiple of the observation, not
    // a round number — at 2.4x, which is 0.017; 0.02 is the nearest value the
    // file already uses and is inside that.
    CHECK(r.rel < 0.02);
    CHECK(r.scale > 1e-3);
    CHECK(r.weights.post_trunk_norm.bytes.empty());
    double d = 0.0;
    for (size_t i = 0; i < r.ours.size(); ++i)
      d = std::max(d, std::abs(static_cast<double>(r.ours[i]) -
                               static_cast<double>(base.ours[i])));
    CHECK(d > 1e-2);
  }
  SUBCASE("`use_qk_norm` false drops the per-head norms and their tensors") {
    TinySpec s = MoeSpec();
    s.v_use_qk_norm = false;
    TowerRun r = RunTower(s);
    MESSAGE("use_qk_norm=false vs reference: relative ", r.rel);
    // MEASURED 2026-09-03 at nudge 42: 8.36e-3, which is about the dense
    // tower's own 8.4e-3. The 0.032 bound predates the fixture's router seed
    // change, when this arm read 1.35e-2, and it is kept rather than tightened
    // because the reason for the looser bound is unchanged: dropping the
    // per-head norm removes the one stage that bounds |q| and |k|, so the
    // attention logits grow and the softmax gets more sensitive to the bf16
    // store underneath it. A single bound shared across towers with different
    // conditioning would be the arbitrary choice, not this one.
    CHECK(r.rel < 0.032);
    CHECK(r.scale > 1e-3);
    for (const auto& blk : r.weights.blocks) {
      CHECK(blk.q_norm.bytes.empty());
      CHECK(blk.k_norm.bytes.empty());
    }
    double d = 0.0;
    for (size_t i = 0; i < r.ours.size(); ++i)
      d = std::max(d, std::abs(static_cast<double>(r.ours[i]) -
                               static_cast<double>(base.ours[i])));
    CHECK(d > 1e-2);
  }
  SUBCASE("`is_causal` true masks the attention, the FLASH arm's own behaviour") {
    TinySpec s = MoeSpec();
    s.v_is_causal = true;
    TowerRun r = RunTower(s);
    MESSAGE("is_causal=true vs reference: relative ", r.rel);
    // MEASURED 2026-09-03 at nudge 42: 1.53e-2, and again with a reason:
    // under a causal mask token 0 attends to ONE key, so its output is that
    // value verbatim and the early rows average far fewer terms — there is
    // less error cancellation left in them than in a bidirectional row. 2.4x
    // the observation, as above.
    CHECK(r.rel < 0.032);
    CHECK(r.scale > 1e-3);
    double d = 0.0;
    for (size_t i = 0; i < r.ours.size(); ++i)
      d = std::max(d, std::abs(static_cast<double>(r.ours[i]) -
                               static_cast<double>(base.ours[i])));
    CHECK(d > 1e-2);
  }
  SUBCASE("`pixel_shuffle_mlp` is a DIFFERENT adapter, on a DIFFERENT state dict") {
    TinySpec s = MoeSpec();
    s.v_adapter_type = "pixel_shuffle_mlp";
    // The shuffle assumes the trunk rows are a ROW-MAJOR grid, which is what
    // the preprocessor emits when `pre_pixel_shuffle` is off — the two flags
    // are independent switches and this arm is the one that needs the flat
    // order (spec §4.11.1, §4.12).
    s.v_pre_pixel_shuffle = false;
    TowerRun r = RunTower(s);
    MESSAGE("pixel_shuffle_mlp vs reference: relative ", r.rel);
    // MEASURED 2026-09-03 at nudge 42: 1.31e-2.
    CHECK(r.rel < 0.02);
    CHECK(r.scale > 1e-3);
    // The state dict really is the other one: `proj.1` is [O, M] where
    // `patch_merger`'s `mlp.0` is [M, M].
    CHECK(r.weights.adapter_ln_w.shape[0] == s.v_merged_dim());
    CHECK(r.weights.adapter_mlp0_w.shape[0] == s.v_adapter_out());
    CHECK(r.weights.adapter_mlp0_w.shape[1] == s.v_merged_dim());
    CHECK(r.weights.adapter_mlp2_w.shape[0] == s.v_adapter_out());
    CHECK(r.weights.adapter_mlp2_w.shape[1] == s.v_adapter_out());
    // ...and it still emits the placeholder span's four rows.
    CHECK(r.ours.size() ==
          static_cast<size_t>(dots3_tiny::kExpectedImageTokens * s.hidden));
  }
}

// ---------------------------------------------------------------------------
// 6. THE PROCESSOR. `resized_size` is NOT `smart_resize`, the normalization is
//    PER CHANNEL, and the placeholder count follows from the grid.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the image processor mirrors upstream's own resize and normalization") {
  using vllm::multimodal::Dots3NoteResizedSize;
  // `factor = patch * merge` = 28 on the released geometry.
  SUBCASE("a conformant size is the identity") {
    const auto rs = Dots3NoteResizedSize(56, 84, 28, 16, 1 << 22);
    CHECK(rs[0] == 56);
    CHECK(rs[1] == 84);
  }
  SUBCASE("each side rounds INDEPENDENTLY to a multiple of factor") {
    const auto rs = Dots3NoteResizedSize(57, 83, 28, 16, 1 << 22);
    CHECK(rs[0] == 56);
    CHECK(rs[1] == 84);
  }
  SUBCASE("a side under factor/4 refuses, naming the bound") {
    CHECK_THROWS_AS((void)Dots3NoteResizedSize(6, 84, 28, 16, 1 << 22),
                    std::runtime_error);
  }
  SUBCASE("an aspect ratio over 200 refuses") {
    CHECK_THROWS_AS((void)Dots3NoteResizedSize(28, 28 * 300, 28, 16, 1 << 30),
                    std::runtime_error);
  }
  SUBCASE("the max-pixel budget shrinks BOTH sides by the same beta") {
    const auto rs = Dots3NoteResizedSize(280, 280, 28, 16, 28 * 28 * 4);
    CHECK(rs[0] % 28 == 0);
    CHECK(rs[1] % 28 == 0);
    CHECK(rs[0] * rs[1] <= 28 * 28 * 4);
  }

  // PER-CHANNEL normalization, which is the second thing that separates this
  // processor from Qwen3-VL's. The fixture's mean/std differ per channel, so a
  // processor that read only `image_mean[0]` would put red's statistics on
  // green and blue.
  TinySpec s;
  TinyCheckpoint ck(FixtureDir(), s);
  const vllm::multimodal::Dots3NoteProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteProcessorConfig(
          ck.dir() + "/preprocessor_config.json", ck.config_path(), "tiny");
  CHECK(cfg.image_mean[0] == doctest::Approx(0.5));
  CHECK(cfg.image_mean[1] == doctest::Approx(0.45));
  CHECK(cfg.image_mean[2] == doctest::Approx(0.4));
  CHECK(cfg.image_std[0] == doctest::Approx(0.25));
  CHECK(cfg.image_std[2] == doctest::Approx(0.35));
  // The three marker ids resolved from `added_tokens.json`.
  CHECK(cfg.image_token_id == dots3_tiny::kImgPadId);
  CHECK(cfg.image_start_token_id == dots3_tiny::kImgStartId);
  CHECK(cfg.image_end_token_id == dots3_tiny::kImgEndId);

  const vllm::multimodal::Dots3NoteImageProcessor proc(cfg);
  const std::vector<uint8_t> rgb = dots3_tiny::FixtureImage(0);
  const vllm::multimodal::ImageKwargs kw = proc.ProcessImage(
      rgb.data(), dots3_tiny::kImageSide, dots3_tiny::kImageSide);
  CHECK(kw.num_patches == 16);
  CHECK(kw.patch_feature_dim == s.v_patch_row());
  // The placeholder run: `prod(grid) // merge**2` (multimodal.py:151-155).
  CHECK(kw.num_patches / (s.v_merge * s.v_merge) ==
        dots3_tiny::kExpectedImageTokens);

  // One patch row, computed by hand against the pre_pixel_shuffle layout:
  // row 0 is the 2x2 block (0,0)'s sub-patch (0,0), i.e. source pixels
  // [0..1] x [0..1], and its channel-c element (ph, pw) is
  // (raw - mean_c/rescale) / (std_c/rescale).
  const int64_t patch = s.v_patch;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t ph = 0; ph < patch; ++ph) {
      for (int64_t pw = 0; pw < patch; ++pw) {
        const uint8_t raw =
            rgb[static_cast<size_t>(ph * dots3_tiny::kImageSide * 3 + pw * 3 + c)];
        const double shift = cfg.image_mean[static_cast<size_t>(c)] /
                             cfg.rescale_factor;
        const double sc =
            cfg.image_std[static_cast<size_t>(c)] / cfg.rescale_factor;
        const double want = (static_cast<double>(raw) - shift) / sc;
        const int64_t k = (c * patch + ph) * patch + pw;
        CHECK(kw.pixel_values_f32[static_cast<size_t>(k)] ==
              doctest::Approx(want).epsilon(1e-6));
      }
    }
  }

  // A non-conformant image is RESIZED and patchified at the RESIZED grid.
  // W6a's version of this assertion was the refusal; W6c (#2537) inverted it,
  // and the two halves of the inversion are asserted together so a resize that
  // ran but kept the old geometry cannot pass: 9x9 resolves to 8x8, which is
  // FOUR 2x2 patches by four, and the placeholder run follows that grid and not
  // the 9x9 one.
  std::vector<uint8_t> odd(static_cast<size_t>(9 * 9 * 3), 128);
  const vllm::multimodal::ImageKwargs okw = proc.ProcessImage(odd.data(), 9, 9);
  const auto ors = Dots3NoteResizedSize(9, 9, proc.factor(), cfg.min_pixels,
                                        cfg.max_pixels);
  CHECK(ors[0] == 8);
  CHECK(ors[1] == 8);
  CHECK(okw.image_grid_thw[1] == ors[0] / s.v_patch);
  CHECK(okw.image_grid_thw[2] == ors[1] / s.v_patch);
  CHECK(okw.num_patches == okw.image_grid_thw[1] * okw.image_grid_thw[2]);
  // A CONSTANT image survives the resample, so every patch element is that
  // constant put through the per-channel normalization. This is the cheapest
  // assertion that the resized buffer — and not the original one read at the
  // wrong stride — is what got patchified.
  for (int64_t c = 0; c < 3; ++c) {
    const double want =
        (128.0 - cfg.image_mean[static_cast<size_t>(c)] / cfg.rescale_factor) /
        (cfg.image_std[static_cast<size_t>(c)] / cfg.rescale_factor);
    const int64_t k = (c * s.v_patch + 0) * s.v_patch + 0;
    CHECK(okw.pixel_values_f32[static_cast<size_t>(k)] ==
          doctest::Approx(want).epsilon(1e-6));
  }
}


// ═══════════════════════════════════════════════════════════════════════════
// THE INDEPENDENT RESAMPLER REFERENCE (W6c, #2537). Written from Pillow's
// `src/libImaging/Resample.c` at tag `12.1.1`; it calls nothing the
// implementation calls, and every qualified name below is `std::`.
//
// It is deliberately a DIFFERENT SHAPE from the implementation. The
// implementation carries upstream's `bounds` + `ksize` packing, one row of
// coefficients per output pixel with the window stored as (first, count). This
// reference builds a DENSE `out x in` weight matrix per axis and multiplies by
// it, so a transcription of the implementation's indexing cannot be mistaken
// for a second reading of the algorithm.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_resample {

// `bicubic_filter`, `a = -0.5`, support 2 (Resample.c). Written as the Keys
// piecewise polynomial rather than in upstream's factored form, so an
// arithmetic slip in one is not reproduced by the other.
double Cubic(double t) {
  const double a = -0.5;
  const double x = t < 0.0 ? -t : t;
  if (x < 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
  if (x < 2.0) return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
  return 0.0;
}

// One axis' DENSE weight matrix, `out` rows of `in` doubles, normalized per row.
// `precompute_coeffs`: `scale = in/out`, `filterscale = max(1, scale)`,
// `support = 2*filterscale`, `center = (o + 0.5)*scale`, weight of input index
// `i` is `cubic((i - center + 0.5)/filterscale)` over the truncated-and-clamped
// window, divided by the row sum.
std::vector<double> Weights(long in_size, long out_size) {
  std::vector<double> w(static_cast<std::size_t>(out_size * in_size), 0.0);
  const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
  const double fs = scale > 1.0 ? scale : 1.0;
  const double support = 2.0 * fs;
  for (long o = 0; o < out_size; ++o) {
    const double center = (static_cast<double>(o) + 0.5) * scale;
    long lo = static_cast<long>(center - support + 0.5);
    if (lo < 0) lo = 0;
    long hi = static_cast<long>(center + support + 0.5);
    if (hi > in_size) hi = in_size;
    double total = 0.0;
    for (long i = lo; i < hi; ++i) {
      const double v = Cubic((static_cast<double>(i) - center + 0.5) / fs);
      w[static_cast<std::size_t>(o * in_size + i)] = v;
      total += v;
    }
    if (total != 0.0) {
      for (long i = lo; i < hi; ++i) w[static_cast<std::size_t>(o * in_size + i)] /= total;
    }
  }
  return w;
}

// `normalize_coeffs_8bpc` + the accumulator seed + `clip8`, applied to one
// weighted sum of 8-bit samples. 22 fraction bits, weights rounded half away
// from zero, accumulator seeded with half an output level, floor shift, clamp.
long FixedPointSample(const std::vector<double>& w, std::size_t row_base,
                      long in_size, const std::vector<unsigned char>& src,
                      std::size_t src_base, std::size_t src_stride) {
  const double kUnit = 4194304.0;  // 2^22
  long acc = 2097152;              // 2^21
  for (long i = 0; i < in_size; ++i) {
    const double v = w[row_base + static_cast<std::size_t>(i)] * kUnit;
    const long q = static_cast<long>(v < 0.0 ? v - 0.5 : v + 0.5);
    acc += q * static_cast<long>(src[src_base + static_cast<std::size_t>(i) * src_stride]);
  }
  long out = acc >> 22;
  if (out < 0) out = 0;
  if (out > 255) out = 255;
  return out;
}

// The whole `ImagingResampleInner` 8bpc path: horizontal into a uint8
// intermediate, then vertical over it, each pass skipped when its axis is
// unchanged.
std::vector<unsigned char> ResizeExact(const std::vector<unsigned char>& src,
                                       long ih, long iw, long oh, long ow) {
  if (ih == oh && iw == ow) return src;
  std::vector<unsigned char> mid;
  const std::vector<unsigned char>* stage = &src;
  long sw = iw;
  if (ow != iw) {
    const std::vector<double> w = Weights(iw, ow);
    mid.assign(static_cast<std::size_t>(ih * ow * 3), 0);
    for (long y = 0; y < ih; ++y) {
      for (long x = 0; x < ow; ++x) {
        for (long c = 0; c < 3; ++c) {
          mid[static_cast<std::size_t>((y * ow + x) * 3 + c)] =
              static_cast<unsigned char>(FixedPointSample(
                  w, static_cast<std::size_t>(x * iw), iw, src,
                  static_cast<std::size_t>((y * iw) * 3 + c), 3));
        }
      }
    }
    stage = &mid;
    sw = ow;
  }
  if (oh == ih) return *stage;
  const std::vector<double> w = Weights(ih, oh);
  std::vector<unsigned char> out(static_cast<std::size_t>(oh * ow * 3), 0);
  for (long y = 0; y < oh; ++y) {
    for (long x = 0; x < ow; ++x) {
      for (long c = 0; c < 3; ++c) {
        out[static_cast<std::size_t>((y * ow + x) * 3 + c)] =
            static_cast<unsigned char>(FixedPointSample(
                w, static_cast<std::size_t>(y * ih), ih, *stage,
                static_cast<std::size_t>(x * 3 + c),
                static_cast<std::size_t>(sw * 3)));
      }
    }
  }
  return out;
}

// The same algorithm with the fixed-point ROUNDING removed but the
// intermediate's SATURATION kept: the uint8 intermediate really does clamp
// between the two passes, and dropping that is a different algorithm rather
// than an unrounded one — on a hard 0/255 edge the horizontal overshoot is
// clipped before the vertical pass ever sees it, which moved this arm by 21.7
// levels when it was first written without the clamp. What is left out is only
// step 6, so the residual against the implementation is the rounding alone.
//
// WHAT THIS ARM CROSS-CHECKS, AND WHAT IT DOES NOT. It exists because the exact
// reference above and the implementation could in principle share a MISREADING
// of the QUANTIZATION — steps 6 and 7 — and still agree. This arm computes
// neither, so such a misreading moves it.
//
// It supplies NO independence on the GEOMETRY, and it must not be read as if it
// did: `ResizeContinuous` calls `ref_resample::Weights`, which is the same
// function `ResizeExact` calls. The two arms differ by step 6 alone. Had the
// centre been read as `xx * scale` with no `+ 0.5` and written into BOTH
// `PrecomputeCoeffs` and `Weights`, every case here would report `maxe == 0`
// and `maxc == 0.0`, both CHECKs green, with every served image shifted half an
// output pixel. Steps 1-5 rest on ONE reading of `Resample.c` plus the Pillow
// cross-check recorded as evidence in spec §4.13.4, which is not a gate.
std::vector<double> ResizeContinuous(const std::vector<unsigned char>& src,
                                     long ih, long iw, long oh, long ow) {
  std::vector<double> cur(static_cast<std::size_t>(ih * iw * 3));
  for (std::size_t i = 0; i < cur.size(); ++i) cur[i] = static_cast<double>(src[i]);
  long cw = iw;
  if (ow != iw) {
    const std::vector<double> w = Weights(iw, ow);
    std::vector<double> nxt(static_cast<std::size_t>(ih * ow * 3), 0.0);
    for (long y = 0; y < ih; ++y)
      for (long x = 0; x < ow; ++x)
        for (long c = 0; c < 3; ++c) {
          double a = 0.0;
          for (long i = 0; i < iw; ++i)
            a += w[static_cast<std::size_t>(x * iw + i)] *
                 cur[static_cast<std::size_t>((y * iw + i) * 3 + c)];
          nxt[static_cast<std::size_t>((y * ow + x) * 3 + c)] = a;
        }
    for (double& v : nxt) {
      if (v < 0.0) v = 0.0;
      if (v > 255.0) v = 255.0;
    }
    cur.swap(nxt);
    cw = ow;
  }
  if (oh != ih) {
    const std::vector<double> w = Weights(ih, oh);
    std::vector<double> nxt(static_cast<std::size_t>(oh * cw * 3), 0.0);
    for (long y = 0; y < oh; ++y)
      for (long x = 0; x < cw; ++x)
        for (long c = 0; c < 3; ++c) {
          double a = 0.0;
          for (long i = 0; i < ih; ++i)
            a += w[static_cast<std::size_t>(y * ih + i)] *
                 cur[static_cast<std::size_t>((i * cw + x) * 3 + c)];
          nxt[static_cast<std::size_t>((y * cw + x) * 3 + c)] = a;
        }
    cur.swap(nxt);
  }
  for (double& v : cur) {
    if (v < 0.0) v = 0.0;
    if (v > 255.0) v = 255.0;
  }
  return cur;
}

}  // namespace ref_resample

// ---------------------------------------------------------------------------
// 6b. THE RESAMPLER (W6c, #2537). `Image.Resampling.BICUBIC` is not a four-tap
//     cubic, and the difference is not a rounding difference on a downscale.
//
//     A MEAN error bound is the wrong instrument here, so this case asserts the
//     PER-PIXEL MAXIMUM against two reference ARMS — one that reproduces
//     upstream's 22-bit fixed point, where the bound is EXACT equality (0
//     levels of 255), and one that stops before the fixed point, where the
//     bound is 2 levels because that is all the intermediate rounding can move
//     a pixel. Both are independent of the IMPLEMENTATION; they are not
//     independent of EACH OTHER, because both build their weights with
//     `ref_resample::Weights`, so what the pair cross-checks is the
//     quantization and not the geometry (see the note above
//     `ResizeContinuous`). Four defect shapes a tolerance alone cannot
//     see get their own arms: a half-pixel centre, an axis swap (every case is
//     non-square in and out), an unnormalized weight row (a near-uniform scale
//     a relative bound absorbs), and the 0/255 saturation ends.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6c: the bicubic resampler is PIL's, per-pixel and at both extremes") {
  using vllm::multimodal::PilResizeBicubicRgb;

  // A deterministic image whose value depends on x, y AND channel, so an axis
  // swap, a dropped channel and a half-pixel shift each change the bytes.
  const auto make = [](long h, long w, int variant) {
    std::vector<unsigned char> v(static_cast<std::size_t>(h * w * 3));
    for (long y = 0; y < h; ++y)
      for (long x = 0; x < w; ++x)
        for (long c = 0; c < 3; ++c) {
          const std::size_t i = static_cast<std::size_t>((y * w + x) * 3 + c);
          switch (variant) {
            case 0:  // a hard 0/255 checker: bicubic OVERSHOOTS past both ends
              v[i] = ((x / 3 + y / 2 + c) % 2) ? 255 : 0;
              break;
            case 1:  // a linear ramp: the half-pixel probe
              v[i] = static_cast<unsigned char>((x * 7 + y * 13 + c * 61) & 0xFF);
              break;
            default: {  // high-frequency noise: the aliasing probe
              unsigned long s = static_cast<unsigned long>(
                  (y * 1103515245UL + x * 12345UL + c * 7919UL) ^ 0x5DEECE66DUL);
              s = (s * 6364136223846793005UL + 1442695040888963407UL);
              v[i] = static_cast<unsigned char>((s >> 33) & 0xFF);
              break;
            }
          }
        }
    return v;
  };

  struct Case {
    long ih, iw, oh, ow;
    const char* what;
  };
  // Every case is NON-SQUARE in at least one of source and target, and the four
  // regimes are covered separately because the support scaling exists only on
  // the downscale side.
  const std::vector<Case> cases{
      {29, 45, 28, 44, "mild downscale, both axes, non-square"},
      {20, 28, 4, 8, "5x by 3.5x downscale — the support scaling dominates"},
      {6, 14, 8, 16, "the served fixture's own upscale, non-square"},
      {33, 17, 11, 34, "downscale one axis while UPSCALING the other"},
      {5, 7, 28, 28, "heavy upscale to a square target from a non-square source"},
      {24, 24, 24, 8, "one axis unchanged, so only ONE pass runs"},
  };

  long worst_exact = 0;
  double worst_cont = 0.0;
  bool saw_zero = false, saw_255 = false;
  for (const Case& k : cases) {
    for (int variant = 0; variant < 3; ++variant) {
      CAPTURE(std::string(k.what));
      CAPTURE(variant);
      const std::vector<unsigned char> src = make(k.ih, k.iw, variant);
      const std::vector<uint8_t> got =
          PilResizeBicubicRgb(src.data(), k.ih, k.iw, k.oh, k.ow);
      REQUIRE(got.size() == static_cast<std::size_t>(k.oh * k.ow * 3));

      const std::vector<unsigned char> want =
          ref_resample::ResizeExact(src, k.ih, k.iw, k.oh, k.ow);
      const std::vector<double> cont =
          ref_resample::ResizeContinuous(src, k.ih, k.iw, k.oh, k.ow);
      REQUIRE(want.size() == got.size());
      REQUIRE(cont.size() == got.size());

      long maxe = 0;
      double maxc = 0.0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        const long d = std::abs(static_cast<long>(got[i]) -
                                static_cast<long>(want[i]));
        if (d > maxe) maxe = d;
        const double dc = std::fabs(static_cast<double>(got[i]) - cont[i]);
        if (dc > maxc) maxc = dc;
        if (got[i] == 0) saw_zero = true;
        if (got[i] == 255) saw_255 = true;
      }
      // EXACT. Both sides compute the same 22-bit fixed point, so any
      // disagreement at all is a defect and not a tolerance question.
      CHECK(maxe == 0);
      // ...and the geometry alone, with no fixed point anywhere, agrees to
      // within what the intermediate rounding can move a pixel.
      CHECK(maxc <= 2.0);
      if (maxe > worst_exact) worst_exact = maxe;
      if (maxc > worst_cont) worst_cont = maxc;
    }
  }
  MESSAGE("W6c resampler: per-pixel max |impl - fixed-point ref| = ", worst_exact,
          " of 255 levels over ", cases.size() * 3, " cases");
  MESSAGE("W6c resampler: per-pixel max |impl - continuous ref| = ", worst_cont,
          " of 255 levels (bound 2.0)");
  // The saturating clamp was REACHED at both ends rather than assumed: the
  // 0/255 checker downscaled overshoots past both, and a clamp that only
  // guarded one end would have been invisible without this.
  CHECK(saw_zero);
  CHECK(saw_255);

  // THE PROCESSOR ACTUALLY USES IT, at the RESIZED stride. Everything above
  // measures `PilResizeBicubicRgb` in isolation; the two arms below measure
  // what `ProcessImage` patchified. They are the arms that catch a resize that
  // ran and was then read back at the SOURCE row stride — a shear that leaves
  // the grid, the patch count, the placeholder run and the served status all
  // valid.
  //
  // The check is a full RECONSTRUCTION: undo the per-channel normalization and
  // undo `pre_pixel_shuffle`'s transpose (`processor.py:185-197`), which puts
  // every element back at the resized-image coordinate it came from, then
  // compare the whole image to the reference resample byte for byte.
  //
  // It is one helper called twice because the two arms differ ONLY in the
  // geometry the processor resolves, and that difference is the whole point:
  // see the downscale arm below.
  const auto process_image_matches_reference =
      [&make](const TinySpec& ps, long ih, long iw, long want_rh,
              long want_rw) {
        const TinyCheckpoint ck(FixtureDir(), ps);
        const vllm::multimodal::Dots3NoteProcessorConfig pcfg =
            vllm::multimodal::LoadDots3NoteProcessorConfig(
                ck.dir() + "/preprocessor_config.json", ck.config_path(),
                "tiny");
        const vllm::multimodal::Dots3NoteImageProcessor proc(pcfg);
        REQUIRE(pcfg.pre_pixel_shuffle);

        const std::vector<unsigned char> src = make(ih, iw, 2);
        const auto rs = vllm::multimodal::Dots3NoteResizedSize(
            ih, iw, proc.factor(), pcfg.min_pixels, pcfg.max_pixels);
        const long rh = static_cast<long>(rs[0]), rw = static_cast<long>(rs[1]);
        REQUIRE(rh == want_rh);
        REQUIRE(rw == want_rw);
        const std::vector<unsigned char> want =
            ref_resample::ResizeExact(src, ih, iw, rh, rw);

        const vllm::multimodal::ImageKwargs kw =
            proc.ProcessImage(src.data(), ih, iw);
        const int64_t patch = pcfg.patch_size, m = pcfg.merge_size;
        const int64_t gh = rh / patch, gw = rw / patch;
        const int64_t Gw = gw / m;
        REQUIRE(kw.image_grid_thw[1] == gh);
        REQUIRE(kw.image_grid_thw[2] == gw);
        REQUIRE(kw.num_patches == gh * gw);

        long worst = 0;
        for (int64_t bh = 0; bh < gh / m; ++bh)
          for (int64_t bw = 0; bw < Gw; ++bw)
            for (int64_t mh = 0; mh < m; ++mh)
              for (int64_t mw = 0; mw < m; ++mw) {
                const int64_t r = ((bh * Gw + bw) * m + mh) * m + mw;
                for (int64_t c = 0; c < 3; ++c) {
                  const double shift =
                      pcfg.image_mean[static_cast<std::size_t>(c)] /
                      pcfg.rescale_factor;
                  const double sc =
                      pcfg.image_std[static_cast<std::size_t>(c)] /
                      pcfg.rescale_factor;
                  for (int64_t ph = 0; ph < patch; ++ph)
                    for (int64_t pw = 0; pw < patch; ++pw) {
                      const int64_t k = (c * patch + ph) * patch + pw;
                      const double got =
                          static_cast<double>(
                              kw.pixel_values_f32[static_cast<std::size_t>(
                                  r * kw.patch_feature_dim + k)]) *
                              sc +
                          shift;
                      const long y = (bh * m + mh) * patch + ph;
                      const long x = (bw * m + mw) * patch + pw;
                      const long ref = static_cast<long>(
                          want[static_cast<std::size_t>((y * rw + x) * 3 + c)]);
                      const long d =
                          std::abs(static_cast<long>(std::lround(got)) - ref);
                      if (d > worst) worst = d;
                    }
                }
              }
        return worst;
      };

  SUBCASE("ProcessImage patchifies the RESIZED image, at the resized stride") {
    // 6x14 is the served fixture's own non-conformant shape: NEITHER side is a
    // multiple of `factor` = 4, and the two sides differ, so an axis swap
    // cannot survive either.
    const long worst = process_image_matches_reference(
        TinySpec{}, dots3_tiny::kOddImageH, dots3_tiny::kOddImageW,
        dots3_tiny::kOddResizedH, dots3_tiny::kOddResizedW);
    MESSAGE("W6c ProcessImage vs reference resample: max |level| = ", worst);
    CHECK(worst == 0);
  }

  // THE DOWNSCALE ARM AT THE PROCESSOR, and the reason it is a separate case.
  //
  // Every other resize that reaches `ProcessImage` is an UPSCALE on both axes,
  // because `Dots3NoteResizedSize` only rounds each side to a multiple of
  // `factor`. On an upscale `filterscale = max(1, in/out)` is 1, the support
  // stays 2.0, and this resampler is bit-for-bit the textbook four-tap cubic:
  // 6x14 -> 8x16 is BYTE-IDENTICAL with the support scaling deleted, so the
  // case above cannot see that deletion at all. The single downscale that used
  // to reach here, 9x9 -> 8x8, carries a CONSTANT image, which every normalized
  // weight set preserves by construction.
  //
  // `factor` is 28 on the released checkpoint, so essentially every real
  // request DOWNSCALES. This arm puts that regime on the production path:
  // `kBudgetMaxPixels` makes `Dots3NoteResizedSize` resolve 24x96 to 4x16, a 6x
  // downscale on both axes with a 25-tap support-scaled window per output
  // pixel, over a TEXTURED image, and compares the patchified result to the
  // independent reference byte for byte.
  SUBCASE("ProcessImage DOWNSCALES through the support-scaled window") {
    TinySpec ps;
    ps.p_max_pixels = dots3_tiny::kBudgetMaxPixels;
    // The premise, asserted rather than assumed: this really is the regime the
    // upscale arm cannot reach. `filterscale` is `in/out` on both axes and it
    // is 6, so `support = 12` and `ksize = 25` where the arm above has 2 and 5.
    REQUIRE(dots3_tiny::kBigImageH / dots3_tiny::kBigResizedH == 6);
    REQUIRE(dots3_tiny::kBigImageW / dots3_tiny::kBigResizedW == 6);
    const long worst = process_image_matches_reference(
        ps, dots3_tiny::kBigImageH, dots3_tiny::kBigImageW,
        dots3_tiny::kBigResizedH, dots3_tiny::kBigResizedW);
    MESSAGE("W6c ProcessImage 6x DOWNSCALE vs reference resample: max |level| = ",
            worst);
    CHECK(worst == 0);
  }

  SUBCASE("an unchanged size is the identity, byte for byte") {
    const std::vector<unsigned char> src = make(9, 13, 2);
    const std::vector<uint8_t> got = PilResizeBicubicRgb(src.data(), 9, 13, 9, 13);
    REQUIRE(got.size() == src.size());
    CHECK(std::equal(src.begin(), src.end(), got.begin()));
  }

  SUBCASE("a non-positive extent refuses by name") {
    const std::vector<unsigned char> src = make(4, 4, 1);
    CHECK_THROWS_AS((void)PilResizeBicubicRgb(src.data(), 4, 4, 0, 4),
                    std::runtime_error);
    CHECK_THROWS_AS((void)PilResizeBicubicRgb(src.data(), 4, 4, 4, -1),
                    std::runtime_error);
  }

  // A CONSTANT image must survive any resize unchanged. This is the arm the
  // weight normalization owns on its own: unnormalized rows scale a flat field
  // away from its own value, at the borders first and everywhere on a
  // downscale, and a relative tolerance on a textured image can absorb that.
  SUBCASE("a constant field is preserved, which is what per-row normalization buys") {
    for (const Case& k : cases) {
      CAPTURE(std::string(k.what));
      for (int level : {0, 1, 137, 254, 255}) {
        const std::vector<unsigned char> src(
            static_cast<std::size_t>(k.ih * k.iw * 3),
            static_cast<unsigned char>(level));
        const std::vector<uint8_t> got =
            PilResizeBicubicRgb(src.data(), k.ih, k.iw, k.oh, k.ow);
        long maxe = 0;
        for (uint8_t v : got) {
          const long d = std::abs(static_cast<long>(v) - static_cast<long>(level));
          if (d > maxe) maxe = d;
        }
        CAPTURE(level);
        CHECK(maxe == 0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 7. THE LOADER BUILDS a MoE tower over a checkpoint that really ships one,
//    through the REAL registry loader, and the multimodal registration is
//    intact beside it. W6a's version of this case asserted the refusal; this
//    is the same checkpoint with the refusal lifted.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: a PYRAMID vision block LOADS through the real registry") {
  Bench bench(MoeSpec());
  REQUIRE(bench.model != nullptr);

  const Dots3NoteVisionParams v = ParseDots3NoteVisionParams(bench.config);
  const std::string why = Dots3NoteVisionRefusal(v, "", {});
  INFO("refusal: '", why, "'");
  CHECK(why.empty());
  REQUIRE(v.num_moe_blocks() == 1);
  CHECK(v.routed_top_k(1) == 2);

  const std::vector<std::string> arch{"Dots3NoteForCausalLM"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
  CHECK(reg.factory->encode_mm != nullptr);
  CHECK(reg.factory->embed_mm != nullptr);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(*bench.model));
  // ...and NOT an M-RoPE model.
  CHECK_FALSE(vllm::ModelRegistry::UsesMrope(*bench.model));
}

// ---------------------------------------------------------------------------
// 8. THE LOAD REFUSES A RE-TYPED `router_bias` BY NAME (porting.md).
//
//    The 17 F32 router biases are the one place upstream itself asks for a
//    dtype that is not the model's, and a narrowed one is invisible to every
//    token gate: the shapes match, the tower computes, and the top-k quietly
//    selects different experts. This case writes the tensor BF16 and asserts
//    the loader says so, which is the F1 fixture row this row already carries
//    for the language tower, pointed at the vision router.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: a BF16 router_bias is refused by name, not read") {
  const TinySpec s = MoeSpec();
  std::vector<dots3_tiny::StOut> entries = dots3_tiny::TinyEntries(s);
  bool retyped = false;
  for (dots3_tiny::StOut& e : entries) {
    if (e.name == "vision_encoder.blocks.1.mlp.router_bias") {
      REQUIRE(e.dtype == "F32");  // the premise
      e.dtype = "BF16";
      retyped = true;
    }
  }
  REQUIRE(retyped);

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("dots3_retyped_bias_" + std::to_string(std::random_device{}()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "config.json", std::ios::binary)
      << dots3_tiny::TinyConfigDoc(FixtureDir(), s).dump();
  dots3_tiny::WriteSafetensors(entries, (dir / "model.safetensors").string());

  const HfConfig cfg = LoadHfConfig((dir / "config.json").string());
  const Dots3NoteVisionParams v = ParseDots3NoteVisionParams(cfg);
  REQUIRE(Dots3NoteVisionRefusal(v, "", {}).empty());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(
      vllm::SafetensorsFile::Open((dir / "model.safetensors").string()));
  CHECK_THROWS_WITH_AS(
      (void)vllm::MaterializeDots3NoteVision(shards, v),
      doctest::Contains("router_bias"), std::runtime_error);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
