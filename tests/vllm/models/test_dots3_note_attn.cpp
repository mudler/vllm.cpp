// dots3-note W3 — the FULL-attention layer gate. CPU-only, checkpoint-free,
// no GPU. Issue #699, #1846; spec `.agents/specs/dots3-note.md` §7 W3.
//
// ─── WHAT THIS ESTABLISHES, AND WHAT IT CANNOT ───────────────────────────────
// `_forward_note_mla`'s full-attention arm is checked against an INDEPENDENT
// double-precision reference in the `ref` namespace below, transcribed from the
// upstream python at `origin/main` = `06ecec7a84` (2026-08-25):
//
//   `vllm/models/dots3_note/nvidia/model.py`      :135-201, :219-326
//   `vllm/model_executor/models/deepseek_v2.py`   :751-842 (Indexer.forward),
//                                                 :1026 (the softmax scale),
//                                                 :1104-1109, :1155-1159 (both ropes)
//   `vllm/ir/ops/layernorm.py`                    :10-21 (RMSNorm)
//   `vllm/model_executor/layers/rotary_embedding/base.py`   :80-103, :161-201
//   `vllm/model_executor/layers/rotary_embedding/common.py` :169-181
//
// The reference is written FROM THE PYTHON, not from `dots3_note_attn.cpp`, and
// it is deliberately a different algorithm wherever a different one exists: it
// rotates with a complex multiply instead of a cos/sin cache, it softmaxes
// WITHOUT the max subtraction in `long double`, it selects the top-k by a full
// stable sort instead of a partial selection, and it accumulates every dot
// product in `long double`. Two implementations of the same formula that share
// no code is the strongest statement this row can make, and it is still only
// that: spec §6.4 option B records that NO oracle for this model runs on any
// hardware this project owns, so NOTHING here says either arm matches vLLM.
// Every mechanism therefore ALSO carries a property assertion that a
// plausible-but-wrong port breaks, which is a statement about the mechanism
// rather than about the agreement of two files.
//
// ─── WHAT IS NOT REACHED ─────────────────────────────────────────────────────
// `Dots3NoteModel::ForwardDevice` still REFUSES BY NAME and this layer is NOT
// on the decode path: the full layers need the padded sparse MLA backend
// (`nvidia/attention.py`::Dots3NotePaddedSparseImpl) over a heterogeneous KV
// cache, which is W4's brick. The last case in this file asserts that refusal
// so the boundary is executable rather than a comment. The geometry the
// reference runs is still the RELEASED config's, resolved through
// `ModelRegistry` and `ParseDots3NoteParams`, never typed by hand.
//
// ─── THE MECHANISMS, AND THE PROPERTY THAT PINS EACH ─────────────────────────
//   * §4 trap 5, the two LoRA rescales (model.py:155, :159). RMSNorm is
//     INVARIANT to a rescale of its input, so a port that applied the scale
//     BEFORE the norm would be a silent no-op; applied AFTER it is not. Both
//     halves are asserted, so the assertion distinguishes the two placements
//     rather than only noticing a missing multiply.
//   * `k_rope_only_layernorm` (model.py:160). Because k_pe is normalised, the
//     whole layer is INVARIANT to a rescale of the `kv_a_proj_with_mqa` rows
//     that produce it. DeepSeek, which does not have this norm, is not. That
//     property needs no reference at all.
//   * the headwise gate (model.py:191-197). One sigmoid per HEAD, so
//     `gated[t,h,d] / attn_out[t,h,d]` is constant over `d` and equals
//     `sigmoid(logit[t,h])`. A lane-wise or transposed gate breaks it.
//   * §4 trap 2, `indexer_rope_interleave` (deepseek_v2.py:1159). GPT-J against
//     NeoX changes which learned coordinates rotate, which changes the SELECTED
//     keys, which changes the output.
//   * #1846, the LEADING rope slice (deepseek_v2.py:804-805). Which HALF of the
//     128-wide index head rotates. Independent of the pairing above, and
//     equally silent.
//   * `is_sparse` (model.py:171). The indexer runs only on the full arm; with
//     it off the layer is dense causal attention and the answer differs.
#include "vllm/model_executor/models/dots3_note_attn.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::Dots3NoteParams;
using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ModelRegistry;
using vllm::ParseDots3NoteParams;
using vllm::dots3_note::Dots3NoteFullAttnDimsFrom;
using vllm::dots3_note::ForwardFullAttention;
using vllm::dots3_note::FullAttnDims;
using vllm::dots3_note::FullAttnTrace;
using vllm::dots3_note::FullAttnWeights;
using vllm::dots3_note::IndexerRopeOffset;

namespace {

std::string FixtureDir() {
#ifdef DOTS3_NOTE_CKPT_FIXTURE_DIR
  return DOTS3_NOTE_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/dots3_note_prev";
#endif
}

// Process-unique, for the reason #1860 records: a bare `static int counter` is
// per-process state, so two concurrent runs of this same binary share one
// directory, each rewrites a file the other has open, and the reader takes
// SIGBUS and exits 135 with doctest's block-buffered summary lost. That reads
// as NO RESULT rather than as a failure, and under spec §6.4 this file is the
// only instrument this row has.
std::filesystem::path UniqueTempDir(const std::string& stem) {
  static const std::string kToken = [] {
    std::random_device rd;
    std::ostringstream os;
    os << std::hex << rd() << "_"
       << std::chrono::steady_clock::now().time_since_epoch().count();
    return os.str();
  }();
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         (stem + kToken + "_" + std::to_string(counter++));
}

class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    dir_ = UniqueTempDir("dots3_note_attn_cfg_");
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

nlohmann::json FixtureConfigDoc() {
  std::ifstream in(FixtureDir() + "/config.json");
  REQUIRE_MESSAGE(in.good(), "cannot open " << FixtureDir() << "/config.json");
  nlohmann::json j;
  in >> j;
  return j;
}

// ─────────────────────────────────────────────────────────────────────────────
// The tiny geometry. Every dimension is the smallest that still exercises the
// branch it stands for, and the numbers are chosen so no mechanism is inert:
//   * `index_topk` 3 against 8 tokens, so FIVE query rows really select rather
//     than taking the short-context all-candidate path (deepseek_v2 indexer /
//     sparse_attn_indexer.py:509-518). The gate asserts that count BY NUMBER —
//     a fixture where the top-k never bit would make both indexer cases pass
//     vacuously.
//   * `index_head_dim` 6 against `qk_rope_head_dim` 4, so the LEADING slice
//     [0,4) and the TAIL slice [2,6) are DIFFERENT sets of lanes (#1846). At
//     index_head_dim == qk_rope_head_dim they would coincide and the
//     assertion would prove nothing.
//   * `q_lora_rank` 6 and `kv_lora_rank` 4 against `hidden_size` 8, so the two
//     rescales sqrt(8/6) and sqrt(8/4) are DIFFERENT from each other and both
//     differ from 1 (§4 trap 5). Equal scales would hide a swap.
//   * `rms_norm_eps` 1e-3, far enough from zero that a port which dropped the
//     epsilon would show.
// ─────────────────────────────────────────────────────────────────────────────
struct TinySpec {
  int64_t hidden = 8;
  int64_t heads = 2;
  int64_t qk_nope = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 4;
  int64_t q_lora = 6;
  int64_t kv_lora = 4;
  int64_t swa_kv_lora = 5;  // > kv_lora, so the padded physical row is real
  int64_t index_n_heads = 2;
  int64_t index_head_dim = 6;
  int64_t index_topk = 3;
  double rope_theta = 137.0;
  double rms_eps = 1e-3;
  int64_t tokens = 8;
};

// The tiny config is the REAL released `config.json` with the geometry
// overridden, so it still travels through `ParseDots3NoteParams` with all 36
// required keys present and W1's whole validation applies to it.
nlohmann::json TinyConfigDoc(const TinySpec& s) {
  nlohmann::json d = FixtureConfigDoc();
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = 4;
  d["layer_types"] = nlohmann::json::array({"full_attention", "full_attention",
                                            "sliding_attention",
                                            "sliding_attention"});
  d["num_attention_heads"] = s.heads;
  d["num_key_value_heads"] = s.heads;
  d["qk_nope_head_dim"] = s.qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["index_n_heads"] = s.index_n_heads;
  d["index_head_dim"] = s.index_head_dim;
  d["index_topk"] = s.index_topk;
  d["swa_num_attention_heads"] = 1;
  d["swa_num_key_value_heads"] = 1;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.swa_kv_lora;
  d["swa_qk_nope_head_dim"] = s.qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["vocab_size"] = 32;
  d["intermediate_size"] = 12;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  return d;
}

Dots3NoteParams TinyParams(const TinySpec& s) {
  TempConfig cfg(TinyConfigDoc(s));
  return ParseDots3NoteParams(LoadHfConfig(cfg.path()));
}

// Deterministic pseudo-random fill in [-amp, amp]. `std::mt19937_64` with a
// fixed seed is reproducible across platforms and standard libraries, which a
// `std::uniform_real_distribution` is not — so the mapping is written out.
struct Rng {
  std::mt19937_64 gen;
  explicit Rng(uint64_t seed) : gen(seed) {}
  double next(double amp) {
    const double u = static_cast<double>(gen() >> 11) /
                     static_cast<double>(1ULL << 53);  // [0,1)
    return (2.0 * u - 1.0) * amp;
  }
  std::vector<double> fill(int64_t n, double amp) {
    std::vector<double> v(static_cast<size_t>(n));
    for (double& x : v) x = next(amp);
    return v;
  }
};

FullAttnWeights TinyWeights(const FullAttnDims& d, uint64_t seed) {
  Rng r(seed);
  const int64_t H = d.hidden_size;
  FullAttnWeights w;
  w.q_a_proj = r.fill(d.q_lora_rank * H, 0.5);
  w.kv_a_proj_with_mqa = r.fill((d.kv_lora_rank + d.qk_rope_head_dim) * H, 0.5);
  // Norm weights around 1.0: an all-zero weight would zero the branch and hide
  // everything downstream of it.
  w.q_a_layernorm = r.fill(d.q_lora_rank, 0.3);
  for (double& x : w.q_a_layernorm) x += 1.0;
  w.kv_a_layernorm = r.fill(d.kv_lora_rank, 0.3);
  for (double& x : w.kv_a_layernorm) x += 1.0;
  w.k_rope_only_layernorm = r.fill(d.qk_rope_head_dim, 0.3);
  for (double& x : w.k_rope_only_layernorm) x += 1.0;
  w.q_b_proj = r.fill(d.num_heads * d.qk_head_dim() * d.q_lora_rank, 0.5);
  w.kv_b_proj =
      r.fill(d.num_heads * (d.qk_nope_head_dim + d.v_head_dim) * d.kv_lora_rank,
             0.5);
  w.o_proj = r.fill(H * d.num_heads * d.v_head_dim, 0.5);
  w.g_proj = r.fill(d.num_heads * H, 0.7);
  w.indexer_wq_b = r.fill(d.index_n_heads * d.index_head_dim * d.q_lora_rank, 0.5);
  w.indexer_wk = r.fill(d.index_head_dim * H, 0.5);
  w.indexer_weights_proj = r.fill(d.index_n_heads * H, 0.5);
  w.indexer_k_norm_weight = r.fill(d.index_head_dim, 0.3);
  for (double& x : w.indexer_k_norm_weight) x += 1.0;
  w.indexer_k_norm_bias = r.fill(d.index_head_dim, 0.2);
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────
// THE INDEPENDENT REFERENCE. Transcribed from the python listed at the top of
// this file. It shares no line of arithmetic with `dots3_note_attn.cpp`:
// different rotation (complex multiply, angles recomputed per element),
// different softmax (no max subtraction, `long double`), different selection
// (full stable sort), different accumulation type. `RefOpts` names the
// plausible-but-wrong ports each mechanism protects against, so a property
// assertion can say WHICH defect it caught.
// ─────────────────────────────────────────────────────────────────────────────
namespace ref {

struct Opts {
  bool apply_lora_rescale = true;  // model.py:155,:159 — §4 trap 5
  bool k_rope_only_norm = true;    // model.py:160
  bool headwise_gate = true;       // model.py:191-197 (false => lane-wise)
  bool indexer_rope_neox = false;  // deepseek_v2.py:1159 — §4 trap 2
  bool indexer_rope_tail = false;  // deepseek_v2.py:804-805 — #1846
  bool run_indexer = true;         // model.py:171 — is_sparse
};

struct Out {
  std::vector<double> out;         // [T, hidden]
  std::vector<double> q_c;         // [T, q_lora_rank]
  std::vector<double> kv_c_normed; // [T, kv_lora_rank]
  std::vector<double> k_pe;        // [T, qk_rope]
  std::vector<double> q;           // [T, heads, qk_head_dim]
  std::vector<double> attn_out;    // [T, heads*v]
  std::vector<double> gated;       // [T, heads*v]
  std::vector<double> gate;        // [T, heads]
  std::vector<int64_t> topk;       // [T, index_topk]
  // Instrument self-reporting, so the gate can say what it actually measured
  // rather than assuming.
  //
  // `rows_where_topk_pruned` — how many query rows really select rather than
  // taking the short-context all-candidate path. If this were 0 the indexer
  // cases below would pass on an identical answer and prove nothing.
  //
  // The margin split matters and is not decoration. The indexer logit is
  // `sum_h w[t,h] * ReLU(dot)` (triton_fp8_mqa_logits.py:129-132), and the ReLU
  // makes an EXACT zero a common value: a key whose every head dots negative
  // scores exactly 0.0. So many rows are decided by a TIE at zero, resolved by
  // the smaller key index in both arms (sparse_attn_indexer.py:509-518). A tie
  // at exactly 0.0 is representable in float and in double alike, so it cannot
  // be flipped by the implementation's float-narrowed logits. What COULD be
  // flipped is a strict margin narrower than float epsilon, so that is the
  // quantity worth bounding, and it is bounded only over the rows that have one.
  int64_t rows_where_topk_pruned = 0;
  int64_t rows_decided_by_a_tie = 0;
  int64_t rows_decided_by_a_strict_margin = 0;
  double min_strict_margin = std::numeric_limits<double>::infinity();
  double max_abs_score = 0.0;
};

// torch `nn.Linear(bias=False)`: y = x @ W.T, W row-major [out, in].
std::vector<double> Dense(const std::vector<double>& x,
                          const std::vector<double>& w, int64_t rows,
                          int64_t in_f, int64_t out_f) {
  std::vector<double> y(static_cast<size_t>(rows) * static_cast<size_t>(out_f));
  for (int64_t o = 0; o < out_f; ++o) {
    for (int64_t r = 0; r < rows; ++r) {
      long double acc = 0.0L;
      for (int64_t i = 0; i < in_f; ++i) {
        acc += static_cast<long double>(x[static_cast<size_t>(r * in_f + i)]) *
               static_cast<long double>(w[static_cast<size_t>(o * in_f + i)]);
      }
      y[static_cast<size_t>(r * out_f + o)] = static_cast<double>(acc);
    }
  }
  return y;
}

// `vllm/ir/ops/layernorm.py`:14-21.
std::vector<double> Rms(const std::vector<double>& x,
                        const std::vector<double>& weight, int64_t rows,
                        int64_t cols, double eps) {
  std::vector<double> y(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    long double var = 0.0L;
    for (int64_t c = 0; c < cols; ++c) {
      const long double v = x[static_cast<size_t>(r * cols + c)];
      var += v * v;
    }
    var /= static_cast<long double>(cols);
    const long double rsqrt = 1.0L / std::sqrt(var + static_cast<long double>(eps));
    for (int64_t c = 0; c < cols; ++c) {
      y[static_cast<size_t>(r * cols + c)] = static_cast<double>(
          static_cast<long double>(x[static_cast<size_t>(r * cols + c)]) *
          rsqrt * static_cast<long double>(weight[static_cast<size_t>(c)]));
    }
  }
  return y;
}

// torch.nn.LayerNorm, elementwise_affine=True — the indexer's `k_norm`
// (deepseek_v2.py:708).
std::vector<double> Ln(const std::vector<double>& x,
                       const std::vector<double>& weight,
                       const std::vector<double>& bias, int64_t rows,
                       int64_t cols, double eps) {
  std::vector<double> y(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    long double mu = 0.0L;
    for (int64_t c = 0; c < cols; ++c) mu += x[static_cast<size_t>(r * cols + c)];
    mu /= static_cast<long double>(cols);
    long double var = 0.0L;
    for (int64_t c = 0; c < cols; ++c) {
      const long double dv = x[static_cast<size_t>(r * cols + c)] - mu;
      var += dv * dv;
    }
    var /= static_cast<long double>(cols);
    const long double denom = std::sqrt(var + static_cast<long double>(eps));
    for (int64_t c = 0; c < cols; ++c) {
      y[static_cast<size_t>(r * cols + c)] = static_cast<double>(
          ((x[static_cast<size_t>(r * cols + c)] - mu) / denom) *
              static_cast<long double>(weight[static_cast<size_t>(c)]) +
          static_cast<long double>(bias[static_cast<size_t>(c)]));
    }
  }
  return y;
}

// `common.py`:169-181 as a COMPLEX rotation, which is the same formula written
// the way the RoPE paper states it: for pair j, (o1 + i*o2) = (x1 + i*x2) *
// e^{i*theta_j*pos}. The cos/sin are recomputed per element rather than read
// from a cache, so nothing about the cache layout is shared with the
// implementation.
void Rotate(std::vector<double>& x, const std::vector<int32_t>& positions,
            double base, int64_t rows, int64_t heads, int64_t stride,
            int64_t offset, int64_t rotary_dim, bool neox) {
  const int64_t half = rotary_dim / 2;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      double* head = x.data() + (t * heads + h) * stride + offset;
      for (int64_t j = 0; j < half; ++j) {
        const double inv_freq =
            std::pow(base, -static_cast<double>(2 * j) /
                               static_cast<double>(rotary_dim));
        const std::complex<double> rot = std::polar(
            1.0, static_cast<double>(positions[static_cast<size_t>(t)]) *
                     inv_freq);
        const int64_t a = neox ? j : 2 * j;
        const int64_t b = neox ? half + j : 2 * j + 1;
        const std::complex<double> z(head[a], head[b]);
        const std::complex<double> zr = z * rot;
        head[a] = zr.real();
        head[b] = zr.imag();
      }
    }
  }
}

Out Forward(const FullAttnDims& d, const FullAttnWeights& w,
            const std::vector<double>& hidden,
            const std::vector<int32_t>& positions, int64_t T, const Opts& o) {
  const int64_t H = d.hidden_size;
  const int64_t N = d.num_heads;
  const int64_t P = d.qk_nope_head_dim;
  const int64_t R = d.qk_rope_head_dim;
  const int64_t V = d.v_head_dim;
  const int64_t QK = P + R;
  const int64_t L = d.kv_lora_rank;
  Out r;

  // model.py:147-155.
  std::vector<double> q_c = Dense(hidden, w.q_a_proj, T, H, d.q_lora_rank);
  q_c = Rms(q_c, w.q_a_layernorm, T, d.q_lora_rank, d.rms_norm_eps);
  if (o.apply_lora_rescale) {
    for (double& v : q_c) v *= d.q_lora_scale;
  }

  // model.py:156-160.
  const std::vector<double> kv_lora =
      Dense(hidden, w.kv_a_proj_with_mqa, T, H, L + R);
  std::vector<double> kv_c(static_cast<size_t>(T * L));
  std::vector<double> k_pe(static_cast<size_t>(T * R));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < L; ++c) {
      kv_c[static_cast<size_t>(t * L + c)] =
          kv_lora[static_cast<size_t>(t * (L + R) + c)];
    }
    for (int64_t c = 0; c < R; ++c) {
      k_pe[static_cast<size_t>(t * R + c)] =
          kv_lora[static_cast<size_t>(t * (L + R) + L + c)];
    }
  }
  std::vector<double> kv_c_normed = Rms(kv_c, w.kv_a_layernorm, T, L, d.rms_norm_eps);
  if (o.apply_lora_rescale) {
    for (double& v : kv_c_normed) v *= d.kv_lora_scale;
  }
  if (o.k_rope_only_norm) {
    k_pe = Rms(k_pe, w.k_rope_only_layernorm, T, R, d.rms_norm_eps);
  }

  // model.py:162-169.
  std::vector<double> q = Dense(q_c, w.q_b_proj, T, d.q_lora_rank, N * QK);
  Rotate(q, positions, d.rope_theta, T, N, QK, /*offset=*/P, R,
         d.rope_is_neox_style);
  Rotate(k_pe, positions, d.rope_theta, T, 1, R, /*offset=*/0, R,
         d.rope_is_neox_style);

  // model.py:171-172 -> deepseek_v2.py:751-842.
  const int64_t IH = d.index_n_heads;
  const int64_t ID = d.index_head_dim;
  std::vector<std::vector<int64_t>> selected(static_cast<size_t>(T));
  if (o.run_indexer) {
    std::vector<double> iq = Dense(q_c, w.indexer_wq_b, T, d.q_lora_rank, IH * ID);
    std::vector<double> ik = Dense(hidden, w.indexer_wk, T, H, ID);
    const std::vector<double> iw =
        Dense(hidden, w.indexer_weights_proj, T, H, IH);
    // The epsilon is the upstream LITERAL, not `d.indexer_k_norm_eps`. Every
    // other scalar this reference reads is pinned by its own assertion, and
    // this one was not: a reviewer moved it three orders of magnitude, the code
    // was reached (the min strict selection margin shifted 1.29e-3 -> 1.16e-3)
    // and the gate stayed green, because BOTH arms read the same wrong number.
    // Two arms drifting together is the shared-helper failure mode wearing a
    // different hat, and under spec §6.4 option B this gate is the only
    // correctness instrument the row has. Transcribed from
    // `deepseek_v2.py`:708 — `LayerNorm(head_dim, eps=1e-6)`.
    ik = Ln(ik, w.indexer_k_norm_weight, w.indexer_k_norm_bias, T, ID, 1e-6);
    const int64_t offset = o.indexer_rope_tail ? ID - R : 0;
    Rotate(iq, positions, d.rope_theta, T, IH, ID, offset, R,
           o.indexer_rope_neox);
    Rotate(ik, positions, d.rope_theta, T, 1, ID, offset, R,
           o.indexer_rope_neox);
    // sparse_attn_indexer.py:203-206 folds the gate weight by
    // softmax_scale * head_scale; triton_fp8_mqa_logits.py:129-132 is
    // sum_h weight[t,h] * relu(dot(q[t,h,:], k[s,:])).
    const double fold = (1.0 / std::sqrt(static_cast<double>(ID))) *
                        (1.0 / std::sqrt(static_cast<double>(IH)));
    for (int64_t t = 0; t < T; ++t) {
      std::vector<std::pair<double, int64_t>> cand;
      cand.reserve(static_cast<size_t>(t + 1));
      for (int64_t s = 0; s <= t; ++s) {
        long double logit = 0.0L;
        for (int64_t h = 0; h < IH; ++h) {
          long double dot = 0.0L;
          for (int64_t c = 0; c < ID; ++c) {
            dot += static_cast<long double>(
                       iq[static_cast<size_t>((t * IH + h) * ID + c)]) *
                   static_cast<long double>(ik[static_cast<size_t>(s * ID + c)]);
          }
          const long double relu = dot > 0.0L ? dot : 0.0L;
          logit += static_cast<long double>(
                       iw[static_cast<size_t>(t * IH + h)] * fold) *
                   relu;
        }
        cand.emplace_back(static_cast<double>(logit), s);
      }
      // Descending by logit, ties to the SMALLER key index: `cand` is already
      // in ascending index order, so a STABLE sort on the logit alone gives it.
      std::vector<std::pair<double, int64_t>> sorted = cand;
      std::stable_sort(sorted.begin(), sorted.end(),
                       [](const std::pair<double, int64_t>& a,
                          const std::pair<double, int64_t>& b) {
                         return a.first > b.first;
                       });
      const int64_t k = std::min<int64_t>(d.index_topk,
                                          static_cast<int64_t>(sorted.size()));
      if (static_cast<int64_t>(sorted.size()) > d.index_topk) {
        ++r.rows_where_topk_pruned;
        const double margin = sorted[static_cast<size_t>(k - 1)].first -
                              sorted[static_cast<size_t>(k)].first;
        if (margin > 0.0) {
          ++r.rows_decided_by_a_strict_margin;
          r.min_strict_margin = std::min(r.min_strict_margin, margin);
        } else {
          ++r.rows_decided_by_a_tie;
        }
      }
      std::vector<int64_t> keys;
      keys.reserve(static_cast<size_t>(k));
      for (int64_t i = 0; i < k; ++i) keys.push_back(sorted[static_cast<size_t>(i)].second);
      std::sort(keys.begin(), keys.end());
      selected[static_cast<size_t>(t)] = keys;
    }
  } else {
    // What a port that missed `is_sparse` would do: attend to the whole causal
    // window.
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t s = 0; s <= t; ++s) selected[static_cast<size_t>(t)].push_back(s);
    }
  }
  r.topk.assign(static_cast<size_t>(T * d.index_topk), -1);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<int64_t>& keys = selected[static_cast<size_t>(t)];
    for (size_t i = 0; i < keys.size() && static_cast<int64_t>(i) < d.index_topk; ++i) {
      r.topk[static_cast<size_t>(t * d.index_topk) + i] = keys[i];
    }
  }

  // model.py:179-188. The unabsorbed MLA: up-project the latent through
  // kv_b_proj and run plain MHA over the selected keys.
  const std::vector<double> kv = Dense(kv_c_normed, w.kv_b_proj, T, L, N * (P + V));
  const double scale = std::pow(static_cast<double>(QK), -0.5);
  std::vector<double> attn(static_cast<size_t>(T * N * V), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<int64_t>& keys = selected[static_cast<size_t>(t)];
    for (int64_t h = 0; h < N; ++h) {
      // No max subtraction: a genuinely different softmax, in long double.
      std::vector<long double> ex(keys.size());
      long double denom = 0.0L;
      for (size_t i = 0; i < keys.size(); ++i) {
        const int64_t s = keys[i];
        long double dot = 0.0L;
        for (int64_t c = 0; c < P; ++c) {
          dot += static_cast<long double>(q[static_cast<size_t>((t * N + h) * QK + c)]) *
                 static_cast<long double>(kv[static_cast<size_t>((s * N + h) * (P + V) + c)]);
        }
        for (int64_t c = 0; c < R; ++c) {
          dot += static_cast<long double>(q[static_cast<size_t>((t * N + h) * QK + P + c)]) *
                 static_cast<long double>(k_pe[static_cast<size_t>(s * R + c)]);
        }
        const long double score = dot * static_cast<long double>(scale);
        r.max_abs_score = std::max(r.max_abs_score,
                                   static_cast<double>(score < 0 ? -score : score));
        ex[i] = std::exp(score);
        denom += ex[i];
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        const long double p = ex[i] / denom;
        for (int64_t c = 0; c < V; ++c) {
          attn[static_cast<size_t>((t * N + h) * V + c)] += static_cast<double>(
              p * static_cast<long double>(
                      kv[static_cast<size_t>((keys[i] * N + h) * (P + V) + P + c)]));
        }
      }
    }
  }

  // model.py:190-201.
  const std::vector<double> logits = Dense(hidden, w.g_proj, T, H, N);
  std::vector<double> gated(attn.size());
  r.gate.assign(static_cast<size_t>(T * N), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < N; ++h) {
      const double g =
          1.0 / (1.0 + std::exp(-logits[static_cast<size_t>(t * N + h)]));
      r.gate[static_cast<size_t>(t * N + h)] = g;
      for (int64_t c = 0; c < V; ++c) {
        const size_t at = static_cast<size_t>((t * N + h) * V + c);
        // The wrong port this flag stands for reuses head 0's gate for every
        // head, which is what a `[T, v_head_dim]`-shaped read of a `[T, heads]`
        // tensor produces on this geometry.
        const double use =
            o.headwise_gate
                ? g
                : 1.0 / (1.0 + std::exp(-logits[static_cast<size_t>(t * N)]));
        gated[at] = attn[at] * use;
      }
    }
  }
  r.out = Dense(gated, w.o_proj, T, N * V, H);
  r.q_c = q_c;
  r.kv_c_normed = kv_c_normed;
  r.k_pe = k_pe;
  r.q = q;
  r.attn_out = attn;
  r.gated = gated;
  return r;
}

}  // namespace ref

// ── comparison, reporting what it compared ───────────────────────────────────
// `max_rel` is the largest ABSOLUTE disagreement divided by the largest
// MAGNITUDE in the two tensors, i.e. error relative to the tensor's own scale.
// It is deliberately not a per-element relative error: `o_proj` sums signed
// terms, so a single output element can land near zero by cancellation, and a
// per-element ratio there reports the reassociation noise of the terms that
// cancelled as if it were a defect. The scale-relative form still catches every
// mechanism difference in this file, because each of them moves the tensor's
// LARGEST entries by more than 1% — which the cases print rather than assume.
struct Diff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double max_mag = 0.0;
  size_t at = 0;
};

Diff Compare(const std::vector<double>& a, const std::vector<double>& b) {
  REQUIRE(a.size() == b.size());
  Diff d;
  for (size_t i = 0; i < a.size(); ++i) {
    const double abs_diff = std::abs(a[i] - b[i]);
    d.max_mag = std::max(d.max_mag, std::max(std::abs(a[i]), std::abs(b[i])));
    if (abs_diff > d.max_abs) {
      d.max_abs = abs_diff;
      d.at = i;
    }
  }
  d.max_rel = d.max_mag > 0.0 ? d.max_abs / d.max_mag : 0.0;
  return d;
}

// The whole tiny bench, built once per case so nothing leaks between them.
struct Bench {
  TinySpec spec;
  Dots3NoteParams params;
  FullAttnDims dims;
  FullAttnWeights w;
  std::vector<double> hidden;
  std::vector<int32_t> positions;

  explicit Bench(TinySpec s = TinySpec{})
      : spec(s), params(TinyParams(spec)),
        dims(Dots3NoteFullAttnDimsFrom(params)) {
    w = TinyWeights(dims, 0x9E3779B97F4A7C15ULL);
    Rng r(0xD1B54A32D192ED03ULL);
    hidden = r.fill(spec.tokens * dims.hidden_size, 1.0);
    positions.resize(static_cast<size_t>(spec.tokens));
    for (int64_t t = 0; t < spec.tokens; ++t) {
      positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    }
  }
  std::vector<double> Run(FullAttnTrace* tr) const {
    return ForwardFullAttention(dims, w, hidden, positions, spec.tokens, tr);
  }
  ref::Out Ref(const ref::Opts& o) const {
    return ref::Forward(dims, w, hidden, positions, spec.tokens, o);
  }
};

// The agreement bound. Both arms are double and the formulas are identical;
// only the summation order, the softmax form and the rotation spelling differ,
// so the residue is pure floating-point reassociation over dimensions of at
// most a few dozen. 1e-11 relative is roughly four orders above the ~1e-15 that
// reassociation at this size can produce and eleven orders below the
// mechanism differences below, every one of which lands above 1e-2. It is NOT
// `doctest::Approx`, whose `scale` term puts an absolute floor of about 1.19e-5
// under any comparison and would pass three of the six mutations in this file.
constexpr double kAgreeRel = 1e-11;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE(
    "dots3-note W3: the FULL geometry comes off the RELEASED config, not a "
    "hand-typed struct") {
  // The registry is the entry point, and `parse_config` is what the engine
  // calls; the dims are derived from THAT result.
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->parse_config != nullptr);
  CHECK_NOTHROW(reg.factory->parse_config(config));

  const Dots3NoteParams p = ParseDots3NoteParams(config);
  const FullAttnDims d = Dots3NoteFullAttnDimsFrom(p);

  CHECK(d.hidden_size == 5120);
  CHECK(d.num_heads == 128);
  CHECK(d.qk_nope_head_dim == 128);
  CHECK(d.qk_rope_head_dim == 64);
  CHECK(d.v_head_dim == 128);
  CHECK(d.q_lora_rank == 1024);
  CHECK(d.kv_lora_rank == 512);
  CHECK(d.qk_head_dim() == 192);

  // §4 trap 5, both scales, and they DIFFER from each other.
  CHECK(std::abs(d.q_lora_scale - std::sqrt(5120.0 / 1024.0)) < 1e-12);
  CHECK(std::abs(d.kv_lora_scale - std::sqrt(5120.0 / 512.0)) < 1e-12);
  CHECK(d.q_lora_scale != d.kv_lora_scale);

  // §4 item 6 as W1 corrected it (#1804): the FULL layers carry the model-level
  // theta and BOTH ropes are GPT-J.
  CHECK(d.rope_theta == doctest::Approx(8e7));
  CHECK_FALSE(d.rope_is_neox_style);
  // §4 trap 2: indexer_rope_interleave=True => the indexer rope is GPT-J too.
  CHECK_FALSE(d.indexer_rope_is_neox_style);

  // The softmax scale has NO YaRN mscale, because dots3 rebuilds
  // rope_parameters as rope_type="default" (model.py:230-238).
  CHECK(std::abs(d.softmax_scale() - std::pow(192.0, -0.5)) < 1e-15);

  // DSA geometry. `indexer_k_norm_eps` is pinned here because it is the one
  // scalar BOTH arms read, and the reference now carries the upstream literal
  // rather than this field, so a drift in either direction reds the gate
  // instead of cancelling out. torch.nn.LayerNorm's own default, hard-coded at
  // `deepseek_v2.py`:708 rather than read from any config.
  CHECK(d.indexer_k_norm_eps == 1e-6);
  CHECK(d.index_n_heads == 64);
  CHECK(d.index_head_dim == 128);
  CHECK(d.index_topk == 2048);

  // #1846. The released shard index declares `indexer_rope_layout: "leading"`,
  // W2 pinned the string, and this is the consumer of it. A TAIL layout would
  // be 128 - 64 = 64 and would rotate an entirely different set of learned
  // coordinates with no shape change and no error.
  CHECK(IndexerRopeOffset(d) == 0);
  CHECK(IndexerRopeOffset(d) != d.index_head_dim - d.qk_rope_head_dim);
}

TEST_CASE("dots3-note W3: the FULL geometry refuses what is not the full arm") {
  TinySpec s;
  Dots3NoteParams p = TinyParams(s);

  // A schedule with no full-attention layer at all. The sliding geometry is
  // W4's, and it differs in five places.
  Dots3NoteParams sliding_only = p;
  for (vllm::Dots3NoteLayerKind& k : sliding_only.layer_types) {
    k = vllm::Dots3NoteLayerKind::kSlidingAttention;
  }
  CHECK_THROWS_WITH_AS((void)Dots3NoteFullAttnDimsFrom(sliding_only),
                       doctest::Contains("no full_attention layer"),
                       std::runtime_error);

  // is_sparse is what makes a layer the full arm (model.py:171).
  Dots3NoteParams no_indexer = p;
  no_indexer.full.has_indexer = false;
  CHECK_THROWS_WITH_AS((void)Dots3NoteFullAttnDimsFrom(no_indexer),
                       doctest::Contains("no DSA indexer"), std::runtime_error);

  // The lane-wise gate arm is refused BY NAME rather than approximated, and the
  // message says why: W1's parse already refuses it, so porting it would put
  // production code behind an input nothing can produce (spec §4.4, W2's M12).
  FullAttnDims d = Dots3NoteFullAttnDimsFrom(p);
  d.attention_gate_type = "linear";
  CHECK_THROWS_WITH_AS(d.Validate(),
                       doctest::Contains("attention_gate_type='linear'"),
                       std::runtime_error);
}

TEST_CASE(
    "dots3-note W3: _forward_note_mla agrees with the independent reference") {
  const Bench b;
  FullAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);
  const ref::Out want = b.Ref(ref::Opts{});

  // FIRST: say what the instrument measured, in its own output. A fixture whose
  // top-k never prunes would make every indexer assertion below vacuous, and a
  // selection margin near float epsilon would make the implementation's
  // float-narrowed logits able to flip a choice.
  MESSAGE("tiny bench: rows where top-k pruned = "
          << want.rows_where_topk_pruned << " of " << b.spec.tokens
          << " (" << want.rows_decided_by_a_strict_margin
          << " by a strict margin, " << want.rows_decided_by_a_tie
          << " by a ReLU tie at exactly 0), min strict margin = "
          << want.min_strict_margin
          << ", max |attention score| = " << want.max_abs_score);
  // Tokens 0,1,2 have 1,2,3 causal candidates against index_topk 3, so they
  // take the short-context all-select path; tokens 3..7 really prune. FIVE.
  REQUIRE(want.rows_where_topk_pruned == 5);
  // At least some rows must be decided by a real margin, or the indexer cases
  // would be exercising nothing but the tie-break rule.
  REQUIRE(want.rows_decided_by_a_strict_margin >= 2);
  // And that margin must be far above the float epsilon at which the
  // implementation's narrowed logits could flip a choice. FLT_EPSILON times the
  // logit scale is order 1e-7 here; 1e-4 is three orders above it.
  REQUIRE(want.min_strict_margin > 1e-4);
  // The reference softmaxes WITHOUT the max subtraction, so its own validity
  // depends on the scores staying in expl()'s comfortable range.
  REQUIRE(want.max_abs_score < 50.0);

  // The selection agrees exactly, which is the part the implementation computes
  // in float.
  REQUIRE(tr.topk.size() == want.topk.size());
  int64_t topk_mismatches = 0;
  for (size_t i = 0; i < tr.topk.size(); ++i) {
    if (tr.topk[i] != want.topk[i]) ++topk_mismatches;
  }
  CHECK(topk_mismatches == 0);

  const Diff dq = Compare(tr.q_c, want.q_c);
  const Diff dkv = Compare(tr.kv_c_normed, want.kv_c_normed);
  const Diff dk = Compare(tr.k_pe, want.k_pe);
  const Diff dqq = Compare(tr.q, want.q);
  const Diff da = Compare(tr.attn_out, want.attn_out);
  const Diff dg = Compare(tr.gated, want.gated);
  const Diff dout = Compare(got, want.out);
  MESSAGE("max relative disagreement: q_c "
          << dq.max_rel << ", kv_c_normed " << dkv.max_rel << ", k_pe "
          << dk.max_rel << ", q " << dqq.max_rel << ", attn_out " << da.max_rel
          << ", gated " << dg.max_rel << ", out " << dout.max_rel);
  CHECK(dq.max_rel < kAgreeRel);
  CHECK(dkv.max_rel < kAgreeRel);
  CHECK(dk.max_rel < kAgreeRel);
  CHECK(dqq.max_rel < kAgreeRel);
  CHECK(da.max_rel < kAgreeRel);
  CHECK(dg.max_rel < kAgreeRel);
  CHECK(dout.max_rel < kAgreeRel);
  // A reference that returned zeros would pass a relative bound on nothing.
  REQUIRE(dout.max_mag > 1e-3);
}

TEST_CASE(
    "dots3-note W3 §4 TRAP 5: the two lora rescales land AFTER the norm, and "
    "before it they would be a no-op") {
  const Bench b;
  FullAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);

  REQUIRE(b.dims.q_lora_scale > 1.0);
  REQUIRE(b.dims.kv_lora_scale > 1.0);
  REQUIRE(b.dims.q_lora_scale != b.dims.kv_lora_scale);

  // (a) DROPPED. The port our DeepSeek MLA would give us has no scalar at all.
  ref::Opts none;
  none.apply_lora_rescale = false;
  const ref::Out unscaled = b.Ref(none);
  const Diff d_out = Compare(got, unscaled.out);
  const Diff d_qc = Compare(tr.q_c, unscaled.q_c);
  MESSAGE("lora rescale DROPPED: max relative output change " << d_out.max_rel
          << ", q_c " << d_qc.max_rel);
  CHECK(d_out.max_rel > 1e-2);
  // q_c is scaled by exactly q_lora_scale, so the relative change is exact.
  CHECK(std::abs(d_qc.max_rel - (b.dims.q_lora_scale - 1.0) /
                                    b.dims.q_lora_scale) < 1e-12);

  // (b) PLACED BEFORE THE NORM. RMSNorm is invariant to a positive rescale of
  //     its input, so a port that multiplied q_c BEFORE `q_a_layernorm` would
  //     compute the UNSCALED answer while looking like it had applied the
  //     scale. That is why (a) alone is not enough: it catches a MISSING
  //     multiply, not a MISPLACED one.
  //
  //     The invariance is exact only as eps -> 0, because
  //     `mean((s*x)^2) + eps` is not `s^2 * (mean(x^2) + eps)`
  //     (ir/ops/layernorm.py:17-18 puts the epsilon INSIDE the root). So this
  //     half runs on a second bench whose only difference is a negligible
  //     `rms_norm_eps`, and the released 1e-5 stays on the main bench where
  //     the epsilon's own placement is what the M18 mutation moves.
  TinySpec tiny_eps = b.spec;
  tiny_eps.rms_eps = 1e-14;
  const Bench be(tiny_eps);
  REQUIRE(be.dims.rms_norm_eps == 1e-14);
  FullAttnWeights pre = be.w;
  for (double& x : pre.q_a_proj) x *= be.dims.q_lora_scale;
  for (double& x : pre.kv_a_proj_with_mqa) x *= be.dims.kv_lora_scale;
  ref::Opts drop_after;
  drop_after.apply_lora_rescale = false;
  const ref::Out pre_scaled = ref::Forward(be.dims, pre, be.hidden, be.positions,
                                           be.spec.tokens, drop_after);
  const ref::Out plain_unscaled =
      ref::Forward(be.dims, be.w, be.hidden, be.positions, be.spec.tokens,
                   drop_after);
  const Diff d_pre = Compare(pre_scaled.q_c, plain_unscaled.q_c);
  const Diff d_post = Compare(be.Run(nullptr), plain_unscaled.out);
  MESSAGE("lora rescale BEFORE the norm moves q_c by "
          << d_pre.max_rel << " (a no-op, as RMSNorm's input invariance "
          << "predicts); AFTER the norm it moves the output by "
          << d_post.max_rel);
  CHECK(d_pre.max_rel < 1e-12);
  CHECK(d_post.max_rel > 1e-2);

  // And the scale a port must NOT confuse: swapping the two changes the answer.
  FullAttnDims swapped = b.dims;
  std::swap(swapped.q_lora_scale, swapped.kv_lora_scale);
  const ref::Out swapped_out =
      ref::Forward(swapped, b.w, b.hidden, b.positions, b.spec.tokens, ref::Opts{});
  CHECK(Compare(got, swapped_out.out).max_rel > 1e-2);
}

TEST_CASE(
    "dots3-note W3: k_rope_only_layernorm makes the layer INVARIANT to the "
    "scale of the k_pe projection, which DeepSeek is not") {
  // The invariance below is exact only as the RMSNorm epsilon goes to zero,
  // because `mean((s*x)^2) + eps` is not `s^2 * (mean(x^2) + eps)`
  // (ir/ops/layernorm.py:17-18). At the main bench's deliberately large
  // 1e-3 the 7.5x rescale still moves the output by 4.7e-3, which is a real
  // effect of the epsilon and not of the mechanism — so the property runs on
  // the negligible-epsilon bench, where it is a clean statement, and the case
  // reports the epsilon-limited number beside it rather than hiding it.
  TinySpec tiny_eps;
  tiny_eps.rms_eps = 1e-14;
  const Bench b(tiny_eps);
  const std::vector<double> got = b.Run(nullptr);

  // The rows of `kv_a_proj_with_mqa` at [kv_lora_rank, kv_lora_rank+qk_rope)
  // are the ones that produce k_pe (model.py:156-158). Because
  // `k_rope_only_layernorm` normalises k_pe row-wise, scaling them by any
  // positive constant CANNOT change the layer's output. This needs no
  // reference: it is a property of the mechanism, and it is FALSE for
  // DeepSeek-V2/V3, whose k_pe goes into the rope unnormalised.
  const int64_t H = b.dims.hidden_size;
  const int64_t L = b.dims.kv_lora_rank;
  const int64_t R = b.dims.qk_rope_head_dim;
  FullAttnWeights rescaled = b.w;
  for (int64_t r = L; r < L + R; ++r) {
    for (int64_t c = 0; c < H; ++c) {
      rescaled.kv_a_proj_with_mqa[static_cast<size_t>(r * H + c)] *= 7.5;
    }
  }
  const std::vector<double> invariant =
      ForwardFullAttention(b.dims, rescaled, b.hidden, b.positions,
                           b.spec.tokens, nullptr);
  const Diff d_inv = Compare(got, invariant);
  MESSAGE("k_pe projection scaled 7.5x: max relative output change "
          << d_inv.max_rel);
  CHECK(d_inv.max_rel < 1e-11);

  // The DeepSeek port — same layer without the extra norm — is NOT invariant,
  // and it does not agree with ours in the first place.
  ref::Opts no_norm;
  no_norm.k_rope_only_norm = false;
  const ref::Out plain = ref::Forward(b.dims, b.w, b.hidden, b.positions,
                                      b.spec.tokens, no_norm);
  const ref::Out plain_rescaled = ref::Forward(
      b.dims, rescaled, b.hidden, b.positions, b.spec.tokens, no_norm);
  const Diff d_drop = Compare(got, plain.out);
  const Diff d_plain_inv = Compare(plain.out, plain_rescaled.out);
  MESSAGE("k_rope_only_layernorm DROPPED: max relative output change "
          << d_drop.max_rel << "; the same 7.5x rescale then moves it by "
          << d_plain_inv.max_rel);
  CHECK(d_drop.max_rel > 1e-2);
  CHECK(d_plain_inv.max_rel > 1e-2);

  // And the epsilon-limited number, so the paragraph above is measured rather
  // than argued: at rms_norm_eps 1e-3 the same rescale moves the output by
  // ~5e-3 — an order and a half below the ~0.1 that dropping the norm costs,
  // but not zero, and a reader who saw only the clean assertion above would not
  // know why it needed its own bench.
  const Bench eps_bench;
  REQUIRE(eps_bench.dims.rms_norm_eps > 1e-4);
  FullAttnWeights eps_rescaled = eps_bench.w;
  for (int64_t r = L; r < L + R; ++r) {
    for (int64_t c = 0; c < H; ++c) {
      eps_rescaled.kv_a_proj_with_mqa[static_cast<size_t>(r * H + c)] *= 7.5;
    }
  }
  const Diff d_eps = Compare(
      eps_bench.Run(nullptr),
      ForwardFullAttention(eps_bench.dims, eps_rescaled, eps_bench.hidden,
                           eps_bench.positions, eps_bench.spec.tokens, nullptr));
  MESSAGE("the same rescale at rms_norm_eps "
          << eps_bench.dims.rms_norm_eps << ": " << d_eps.max_rel
          << " — the epsilon inside the root, not the mechanism");
  CHECK(d_eps.max_rel > 1e-6);
  CHECK(d_eps.max_rel < 0.1 * d_drop.max_rel);
}

TEST_CASE(
    "dots3-note W3: the attention gate is HEADWISE — one sigmoid per head, "
    "constant across that head's value lanes") {
  const Bench b;
  FullAttnTrace tr;
  (void)b.Run(&tr);
  const int64_t T = b.spec.tokens;
  const int64_t N = b.dims.num_heads;
  const int64_t V = b.dims.v_head_dim;

  REQUIRE(static_cast<int64_t>(tr.gate.size()) == T * N);
  REQUIRE(static_cast<int64_t>(tr.gated.size()) == T * N * V);

  // (a) every gate is a sigmoid, so strictly inside (0,1), and NOT all equal —
  //     an all-0.5 column would mean g_proj produced zeros and the test would
  //     be measuring nothing.
  double gmin = 1.0;
  double gmax = 0.0;
  for (const double g : tr.gate) {
    CHECK(g > 0.0);
    CHECK(g < 1.0);
    gmin = std::min(gmin, g);
    gmax = std::max(gmax, g);
  }
  MESSAGE("headwise gate range: [" << gmin << ", " << gmax << "]");
  REQUIRE(gmax - gmin > 0.1);

  // (b) THE property: gated / attn_out is constant over the v lanes of a head
  //     and equals that head's sigmoid. A lane-wise gate, a transposed gate, or
  //     a gate broadcast from the wrong axis all break it.
  double worst = 0.0;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < N; ++h) {
      for (int64_t c = 0; c < V; ++c) {
        const size_t at = static_cast<size_t>((t * N + h) * V + c);
        if (std::abs(tr.attn_out[at]) < 1e-9) continue;
        const double ratio = tr.gated[at] / tr.attn_out[at];
        worst = std::max(worst,
                         std::abs(ratio - tr.gate[static_cast<size_t>(t * N + h)]));
      }
    }
  }
  MESSAGE("worst |gated/attn_out - sigmoid(logit)| over all heads/lanes: "
          << worst);
  CHECK(worst < 1e-12);

  // (c) a port that reused head 0's gate for every head — which is what reading
  //     a [T, num_heads] tensor as if it were [T, v_head_dim] gives on this
  //     geometry — produces a different answer.
  ref::Opts flat;
  flat.headwise_gate = false;
  const ref::Out one_gate = b.Ref(flat);
  const Diff d_flat = Compare(tr.gated, one_gate.gated);
  MESSAGE("gate broadcast from head 0 only: max relative change "
          << d_flat.max_rel);
  CHECK(d_flat.max_rel > 1e-2);

  // (d) the gate reads `hidden_states`, not the attention output: perturbing
  //     `g_proj` alone must move the output.
  FullAttnWeights bumped = b.w;
  bumped.g_proj[0] += 1.0;
  const std::vector<double> moved = ForwardFullAttention(
      b.dims, bumped, b.hidden, b.positions, b.spec.tokens, nullptr);
  CHECK(Compare(b.Run(nullptr), moved).max_rel > 1e-6);
}

TEST_CASE(
    "dots3-note W3 §4 TRAP 2: the indexer RoPE is GPT-J, and NeoX selects "
    "different keys") {
  const Bench b;
  FullAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);

  REQUIRE_FALSE(b.dims.indexer_rope_is_neox_style);

  ref::Opts neox;
  neox.indexer_rope_neox = true;
  const ref::Out wrong = b.Ref(neox);

  // The pairing changes WHICH learned coordinates rotate, so it changes the
  // selection. Count the differing slots BY NUMBER: if a future fixture stopped
  // discriminating, this REQUIRE says so instead of the case passing on an
  // identical answer.
  int64_t differing = 0;
  for (size_t i = 0; i < tr.topk.size(); ++i) {
    if (tr.topk[i] != wrong.topk[i]) ++differing;
  }
  MESSAGE("indexer RoPE GPT-J vs NeoX: " << differing << " of "
          << tr.topk.size() << " selection slots differ");
  REQUIRE(differing > 0);

  const Diff d = Compare(got, wrong.out);
  MESSAGE("indexer RoPE NeoX: max relative output change " << d.max_rel);
  CHECK(d.max_rel > 1e-3);
}

TEST_CASE(
    "dots3-note W3 #1846: the indexer rotates the LEADING 64 lanes, and the "
    "tail slice selects different keys") {
  const Bench b;
  FullAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);

  // The layout the released shard index declares, and the lane it implies.
  CHECK(IndexerRopeOffset(b.dims) == 0);
  const int64_t tail = b.dims.index_head_dim - b.dims.qk_rope_head_dim;
  REQUIRE(tail > 0);  // otherwise leading and tail coincide and this proves nothing

  ref::Opts tail_opts;
  tail_opts.indexer_rope_tail = true;
  const ref::Out wrong = b.Ref(tail_opts);

  int64_t differing = 0;
  for (size_t i = 0; i < tr.topk.size(); ++i) {
    if (tr.topk[i] != wrong.topk[i]) ++differing;
  }
  MESSAGE("indexer rope slice LEADING [0," << b.dims.qk_rope_head_dim
          << ") vs TAIL [" << tail << "," << b.dims.index_head_dim << "): "
          << differing << " of " << tr.topk.size()
          << " selection slots differ");
  REQUIRE(differing > 0);

  const Diff d = Compare(got, wrong.out);
  MESSAGE("indexer TAIL slice: max relative output change " << d.max_rel);
  CHECK(d.max_rel > 1e-3);

  // The two questions are INDEPENDENT: the pairing (§4 trap 2) and the slice
  // (#1846) pick different keys as each other, so neither subsumes the other.
  ref::Opts neox;
  neox.indexer_rope_neox = true;
  const ref::Out neox_out = b.Ref(neox);
  int64_t between = 0;
  for (size_t i = 0; i < wrong.topk.size(); ++i) {
    if (wrong.topk[i] != neox_out.topk[i]) ++between;
  }
  CHECK(between > 0);
}

TEST_CASE(
    "dots3-note W3: a q_c rescale is INVISIBLE to the indexer's selection, so "
    "§4 trap 5 reaches the output only through the MLA scores") {
  // This case exists because a mutation came back GREEN. Feeding the indexer
  // the UNRESCALED `q_c` — i.e. applying §4 trap 5 after the indexer instead of
  // before it — changed nothing, and the reason is an invariance rather than a
  // hole: the logit is `sum_h w[t,h] * ReLU(dot(q[t,h,:], k[s,:]))`
  // (triton_fp8_mqa_logits.py:129-132), so a POSITIVE rescale of q_c multiplies
  // every logit in a row by one constant and the argmax does not move.
  //
  // The honest response to a green mutation is to state the guarantee the
  // mutation actually probed, not to keep a claim the gate cannot see. The
  // scale factor is 4.0 rather than 3.0 on purpose: a power of two rescales a
  // binary float exactly, so the logit ratio below is an EQUALITY and not a
  // tolerance.
  const Bench b;
  FullAttnTrace base;
  const std::vector<double> got = b.Run(&base);

  FullAttnWeights scaled = b.w;
  for (double& x : scaled.indexer_wq_b) x *= 4.0;
  FullAttnTrace bumped;
  const std::vector<double> out = ForwardFullAttention(
      b.dims, scaled, b.hidden, b.positions, b.spec.tokens, &bumped);

  REQUIRE(base.indexer_logits.size() == bumped.indexer_logits.size());
  int64_t finite_logits = 0;
  double worst_ratio_error = 0.0;
  for (size_t i = 0; i < base.indexer_logits.size(); ++i) {
    if (!std::isfinite(base.indexer_logits[i])) continue;
    ++finite_logits;
    if (base.indexer_logits[i] == 0.0) {
      CHECK(bumped.indexer_logits[i] == 0.0);
      continue;
    }
    worst_ratio_error = std::max(
        worst_ratio_error,
        std::abs(bumped.indexer_logits[i] / base.indexer_logits[i] - 4.0));
  }
  MESSAGE("indexer_wq_b scaled 4x: " << finite_logits
          << " finite logits, worst |ratio - 4| = " << worst_ratio_error);
  REQUIRE(finite_logits > 0);
  CHECK(worst_ratio_error == 0.0);

  // The selection is byte-identical, and so is the whole layer output: the
  // indexer's only product is the mask.
  CHECK(base.topk == bumped.topk);
  CHECK(Compare(got, out).max_abs == 0.0);
}

TEST_CASE(
    "dots3-note W3: the full arm is SPARSE — the indexer's top-k is the mask, "
    "and dense causal attention is a different answer") {
  const Bench b;
  FullAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);
  const int64_t K = b.dims.index_topk;

  // Every selected key is causal and unique, and the short-context rows really
  // do take everything (the short-context rule the shared DSA port owns;
  // the selector itself is sparse_attn_indexer.py:509-518).
  for (int64_t t = 0; t < b.spec.tokens; ++t) {
    std::vector<int64_t> row;
    for (int64_t j = 0; j < K; ++j) {
      const int64_t s = tr.topk[static_cast<size_t>(t * K + j)];
      if (s < 0) continue;
      CHECK(s <= t);
      row.push_back(s);
    }
    std::sort(row.begin(), row.end());
    CHECK(std::adjacent_find(row.begin(), row.end()) == row.end());
    const int64_t candidates = t + 1;
    CHECK(static_cast<int64_t>(row.size()) == std::min(candidates, K));
  }

  ref::Opts dense;
  dense.run_indexer = false;
  const ref::Out full_causal = b.Ref(dense);
  const Diff d = Compare(got, full_causal.out);
  MESSAGE("indexer OFF (dense causal): max relative output change " << d.max_rel);
  CHECK(d.max_rel > 1e-3);
}

TEST_CASE("dots3-note W3: the layer refuses a weight of the wrong size BY NAME") {
  const Bench b;
  FullAttnWeights bad = b.w;
  bad.k_rope_only_layernorm.pop_back();
  CHECK_THROWS_WITH_AS(
      (void)ForwardFullAttention(b.dims, bad, b.hidden, b.positions,
                                 b.spec.tokens, nullptr),
      doctest::Contains("k_rope_only_layernorm.weight"), std::runtime_error);

  FullAttnWeights bad_gate = b.w;
  bad_gate.g_proj.resize(bad_gate.g_proj.size() + b.dims.hidden_size);
  CHECK_THROWS_WITH_AS(
      (void)ForwardFullAttention(b.dims, bad_gate, b.hidden, b.positions,
                                 b.spec.tokens, nullptr),
      doctest::Contains("g_proj.weight"), std::runtime_error);

  FullAttnWeights bad_knorm = b.w;
  bad_knorm.indexer_k_norm_bias.clear();
  CHECK_THROWS_WITH_AS(
      (void)ForwardFullAttention(b.dims, bad_knorm, b.hidden, b.positions,
                                 b.spec.tokens, nullptr),
      doctest::Contains("indexer.k_norm.bias"), std::runtime_error);
}

TEST_CASE(
    "dots3-note W3: the DEVICE forward still refuses — this layer is NOT on "
    "the decode path") {
  // The honest boundary, made executable. W3 lands the reference and its gate;
  // the decode path needs the padded sparse MLA backend over a heterogeneous KV
  // cache, which is W4's brick (#699). Reaching the refusal through the REAL
  // model the factory returns, never a fabricated LoadedModel subclass, which
  // is undefined behaviour the moment the handle is opened (#730/#775).
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->forward != nullptr);

  vllm::Dots3NoteWeights weights;
  weights.params = ParseDots3NoteParams(config);
  weights.materialized = false;
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> ids{1};
  const std::vector<int32_t> pos{0};
  vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;
  const std::vector<int32_t> logits_indices{0};
  CHECK_THROWS_WITH_AS(
      (void)vllm::Dots3NoteModel::ForwardDevice(ids, pos, meta, kv, weights,
                                                queue, logits_indices),
      doctest::Contains("not ported"), std::runtime_error);
}
