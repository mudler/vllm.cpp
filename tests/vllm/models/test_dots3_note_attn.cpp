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
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_device_glue.h"  // W5: Dev/DBuf for the router probe
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

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
  // §4 trap 5, as TWO switches rather than one. Review finding F1: an arm that
  // neutralises BOTH at once cannot tell a port that dropped only the q scale
  // from one that carries both, and the q scale is the field most likely to
  // disturb DeepSeek-V3 because it is the only one on the q_lora branch.
  bool apply_q_lora_rescale = true;   // model.py:155
  bool apply_kv_lora_rescale = true;  // model.py:159
  bool k_rope_only_norm = true;    // model.py:160
  bool headwise_gate = true;       // model.py:191-197 (false => lane-wise)
  bool indexer_rope_neox = false;  // deepseek_v2.py:1159 — §4 trap 2
  bool indexer_rope_tail = false;  // deepseek_v2.py:804-805 — #1846
  bool run_indexer = true;         // model.py:171 — is_sparse
  // The WIDTH of the gate logit, which is a memory format and not an algorithm
  // (review finding F2). `g_proj` is built with no `params_dtype`
  // (model.py:292-297), so upstream's sigmoid input is a BF16 value that
  // `.float()` then widens. FALSE keeps this reference in pure double, which
  // is what the whole-model comparison uses; TRUE models upstream's width, and
  // the case that turns it on is how the device path's own width is GATED
  // rather than asserted.
  bool bf16_gate_logit = false;
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
  if (o.apply_q_lora_rescale) {
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
  if (o.apply_kv_lora_rescale) {
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
  std::vector<double> logits = Dense(hidden, w.g_proj, T, H, N);
  if (o.bf16_gate_logit) {
    for (double& v : logits) {
      v = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(v))));
    }
  }
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
  none.apply_q_lora_rescale = false;
  none.apply_kv_lora_rescale = false;
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
  drop_after.apply_q_lora_rescale = false;
  drop_after.apply_kv_lora_rescale = false;
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
    "dots3-note W3/W4a/W5: the RELEASED config is no longer refused, and an "
    "UNMATERIALIZED handle still is") {
  // The honest boundary, made executable, and its subject has MOVED TWICE.
  // W3 read "nothing is on the decode path". W4a put the FULL-attention layer
  // there and this case pinned the released config still being refused, at
  // layer 1 (MoE, W5) and layer 2 (sliding, W4b). **W5 lifted the last two
  // branches** — the MoE layer and the nextn tail (W5c, #2176) — so
  // `Dots3NoteDeviceRefusal` over the real released `config.json` is EMPTY, and
  // the assertion below is the `true -> false` flip that says so rather than a
  // deleted case.
  //
  // What the forward still refuses on THIS handle is the other guard, and it is
  // the reason the case survives: `weights.materialized` is false here, and a
  // forward over an unmaterialized tower would read uninitialised weights and
  // emit a plausible token. Reached through the REAL model the factory returns,
  // never a fabricated LoadedModel subclass, which is undefined behaviour the
  // moment the handle is opened (#730/#775).
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
  // THE W5 FLIP, asserted where a reader following W3/W4a's evidence lands.
  CHECK(vllm::Dots3NoteDeviceRefusal(weights.params).empty());
  CHECK_THROWS_WITH_AS(
      (void)vllm::Dots3NoteModel::ForwardDevice(ids, pos, meta, kv, weights,
                                                queue, logits_indices),
      doctest::Contains("not materialized"), std::runtime_error);
}

// ════════════════════════════════════════════════════════════════════════════
// W4a — the FULL-attention layer ON THE DECODE PATH (#699).
//
// W3 proved the maths against an independent double reference and said plainly
// that nothing reached it. This section is the other half: the SAME reference,
// now run against the DEVICE path, reached through `ModelRegistry::Forward` on
// a real `ModelRegistration` the registry resolved, over a real synthetic
// safetensors checkpoint the real loader read.
//
// Two things this can and cannot say, stated before the assertions:
//
//  * IT CAN say the device path computes what the reference computes. The
//    reference is unchanged from W3 — a different algorithm at every step —
//    and it is the only correctness instrument this row has (spec §6.4
//    option B). No vLLM oracle exists for this model on any host we own.
//  * IT CANNOT say either arm matches vLLM. It also cannot resolve the device
//    arm below bf16: upstream's activation dtype IS bf16 (`porting.md`: one
//    model dtype, `f32` an annotated escape), so the comparison bound below is
//    a bf16 bound and is printed rather than assumed. Every mutation in the
//    table lands one to three orders ABOVE it.
namespace {
namespace w4a {

using vllm::Dots3NoteDenseEquivalentMaxSeqLen;
using vllm::Dots3NoteDeviceRefusal;
using vllm::Dots3NoteFullAttnMlaDims;
using vllm::PagedKvCache;

// The stored width of every weight and of the whole activation stream.
double Bf16(double x) {
  return static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(x))));
}
std::vector<double> Bf16All(std::vector<double> v) {
  for (double& x : v) x = Bf16(x);
  return v;
}

// One tensor as it will be written to disk: the shape the loader checks, and
// since W5 the DTYPE too.
//
// `dtype` defaults to BF16 and every W3/W4 entry keeps that default, so nothing
// those bricks wrote changes. It exists because `mlp.gate.e_score_correction_bias`
// is F32 in the released checkpoint and F32 upstream (deepseek_v2.py:322-324) —
// the ONE dtype exception in this tower — and a fixture that wrote it BF16 would
// be testing a checkpoint the publisher does not ship, on the exact axis
// `porting.md`'s memory-format rule is about.
struct StOut {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<double> values;  // ALREADY rounded to `dtype`
  std::string dtype = "BF16";
};

void WriteSafetensorsBf16(const std::vector<StOut>& entries, const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StOut& e : entries) {
    size_t n = 1;
    for (int64_t s : e.shape) n *= static_cast<size_t>(s);
    REQUIRE(n == e.values.size());
    REQUIRE_MESSAGE((e.dtype == "BF16" || e.dtype == "F32"),
                    "unsupported fixture dtype " << e.dtype << " for " << e.name);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StOut& e : entries) {
    for (double v : e.values) {
      if (e.dtype == "F32") {
        const float f = static_cast<float>(v);
        out.write(reinterpret_cast<const char*>(&f), 4);
      } else {
        const uint16_t b = vt::F32ToBF16(static_cast<float>(v));
        out.write(reinterpret_cast<const char*>(&b), 2);
      }
    }
  }
}

class TempCheckpoint {
 public:
  explicit TempCheckpoint(const std::vector<StOut>& entries) {
    dir_ = UniqueTempDir("dots3_note_w4a_ckpt_");
    std::filesystem::create_directories(dir_);
    WriteSafetensorsBf16(entries, (dir_ / "model.safetensors").string());
  }
  ~TempCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string file() const { return (dir_ / "model.safetensors").string(); }

 private:
  std::filesystem::path dir_;
};

// ─────────────────────────────────────────────────────────────────────────────
// The device bench: a config the W4a device forward can actually run — every
// layer FULL attention with a DENSE MLP — built from the RELEASED config.json
// with the geometry overridden, so all 36 required keys and the whole W1
// validation still apply to it.
// The RANKS are chosen so the two §4-trap-5 scales land near the RELEASED
// model's, and that is review finding F1 rather than taste. At the first
// draft's `q_lora=6, kv_lora=4` over `hidden=8` the scales were 1.155 and
// 1.414 — nearly the identity — against the released 2.236 and 3.162, and a
// mutation dropping `q_lora_scale` alone reddened at only 1.05x the bound: one
// seed or one compiler from a false green, on the field most likely to disturb
// DeepSeek-V3. `q_lora=3, kv_lora=2` over `hidden=16` gives 2.309 and 2.828,
// which is the released ratio's neighbourhood, keeps the two DIFFERENT from
// each other so a swap cannot hide, and keeps `kv_lora >= 2` so the latent
// RMSNorm is not the degenerate 1-wide one. `heads * v_head == hidden` is kept
// deliberately: it is what lets a mutation feed the gate the POST-attention
// state and still compile, which is the silent defect M7 probes.
struct DeviceSpec {
  int64_t hidden = 16;
  int64_t heads = 2;
  int64_t qk_nope = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 8;
  int64_t q_lora = 3;
  int64_t kv_lora = 2;
  int64_t layers = 2;
  int64_t vocab = 12;
  int64_t inter = 10;
  int64_t max_pos = 16;
  // >= tokens, so the DSA top-k selects every causal candidate and dense
  // attention IS upstream's answer. The refusal past this bound has its own
  // case below.
  int64_t index_topk = 16;
  int64_t index_n_heads = 2;
  int64_t index_head_dim = 6;
  // §4 trap 2, `indexer_rope_interleave` (deepseek_v2.py:1159 @ `bc2d63e650`):
  // `is_neox_style = not indexer_rope_interleave`, so TRUE — the released
  // default this repository resolves — makes the INDEXER rope GPT-J, the same
  // pairing the main MLA rope uses. Exposed here so a case can set it FALSE and
  // drive the two flags APART, which is the only way to gate their independence
  // on the device path.
  bool indexer_rope_interleave = true;
  double rope_theta = 137.0;
  double rms_eps = 1e-3;
  int64_t tokens = 6;
  bool tie_word_embeddings = false;
};

nlohmann::json DeviceConfigDoc(const DeviceSpec& s) {
  nlohmann::json d = FixtureConfigDoc();
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = s.layers;
  nlohmann::json lt = nlohmann::json::array();
  for (int64_t i = 0; i < s.layers; ++i) lt.push_back("full_attention");
  d["layer_types"] = lt;
  d["num_attention_heads"] = s.heads;
  d["num_key_value_heads"] = s.heads;
  d["qk_nope_head_dim"] = s.qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["max_position_embeddings"] = s.max_pos;
  d["index_n_heads"] = s.index_n_heads;
  d["index_head_dim"] = s.index_head_dim;
  d["index_topk"] = s.index_topk;
  d["indexer_rope_interleave"] = s.indexer_rope_interleave;
  // The SWA geometry is required by the parse even with zero sliding layers.
  // `swa_kv_lora_rank == kv_lora_rank` makes the PHYSICAL latent row equal the
  // logical one, because narrowing a padded row is W4b and the forward refuses
  // it by name.
  d["swa_num_attention_heads"] = 1;
  d["swa_num_key_value_heads"] = 1;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.kv_lora;
  d["swa_qk_nope_head_dim"] = s.qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["vocab_size"] = s.vocab;
  d["intermediate_size"] = s.inter;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  // Every layer DENSE: W5 owns the MoE.
  d["first_k_dense_replace"] = s.layers;
  // No nextn tail: W10 owns it, and the loader would demand its tensors.
  d["num_nextn_predict_layers"] = 0;
  d["tie_word_embeddings"] = s.tie_word_embeddings;
  return d;
}

// The whole tiny model in double, with every weight ALREADY rounded to the
// bf16 the checkpoint stores — so the comparison measures the FORWARD, not the
// weights' storage width.
struct DeviceWeights {
  std::vector<double> embed;      // [vocab, hidden]
  std::vector<double> final_norm; // [hidden]
  std::vector<double> lm_head;    // [vocab, hidden]
  struct Layer {
    std::vector<double> input_ln;   // [hidden]
    std::vector<double> post_ln;    // [hidden]
    FullAttnWeights attn;
    std::vector<double> gate_proj;  // [inter, hidden]
    std::vector<double> up_proj;    // [inter, hidden]
    std::vector<double> down_proj;  // [hidden, inter]
  };
  std::vector<Layer> layers;
};

DeviceWeights MakeDeviceWeights(const DeviceSpec& s, const FullAttnDims& d,
                                uint64_t seed) {
  Rng r(seed);
  DeviceWeights w;
  w.embed = Bf16All(r.fill(s.vocab * s.hidden, 0.7));
  w.final_norm = Bf16All([&] {
    std::vector<double> v = r.fill(s.hidden, 0.3);
    for (double& x : v) x += 1.0;
    return v;
  }());
  w.lm_head = Bf16All(r.fill(s.vocab * s.hidden, 0.5));
  for (int64_t l = 0; l < s.layers; ++l) {
    DeviceWeights::Layer lw;
    lw.input_ln = Bf16All([&] {
      std::vector<double> v = r.fill(s.hidden, 0.3);
      for (double& x : v) x += 1.0;
      return v;
    }());
    lw.post_ln = Bf16All([&] {
      std::vector<double> v = r.fill(s.hidden, 0.3);
      for (double& x : v) x += 1.0;
      return v;
    }());
    // The SAME generator W3's gate uses, so the attention weights are the ones
    // the reference was written against.
    lw.attn = TinyWeights(d, seed + 0x1000ULL * static_cast<uint64_t>(l + 1));
    lw.attn.q_a_proj = Bf16All(lw.attn.q_a_proj);
    lw.attn.q_a_layernorm = Bf16All(lw.attn.q_a_layernorm);
    lw.attn.kv_a_layernorm = Bf16All(lw.attn.kv_a_layernorm);
    // `k_rope_only_layernorm` has to be MADE observable or the case that
    // drops it proves nothing. Two deliberate choices, both of them fixture
    // design rather than model behaviour: the rope rows of
    // `kv_a_proj_with_mqa` are amplified so k_pe arrives with an RMS far from
    // 1 (which is what the norm then removes), and the norm's own weights are
    // spread widely around 1 instead of hugging it. `TinyWeights`' defaults
    // put both within a few percent of the identity, and dropping the norm
    // there moved the LOGITS by only 2.7x the bf16 floor.
    {
      const int64_t R = d.qk_rope_head_dim, H = d.hidden_size;
      const int64_t off = d.kv_lora_rank * H;
      for (int64_t i = 0; i < R * H; ++i)
        lw.attn.kv_a_proj_with_mqa[static_cast<size_t>(off + i)] *= 6.0;
      // The weight ALTERNATES sharply within each rotated GPT-J pair (lanes
      // 2i, 2i+1), and that is not decoration. RoPE preserves the L2 norm of
      // every pair exactly, so `rms(rope(x)) == rms(x)` and the ONLY part of
      // `k_rope_only_layernorm` that does not commute with the rotation is the
      // per-lane weight. With weights hugging 1 the two ORDERS differ by a
      // hair: mutation M5 (norm applied AFTER the rope) moved the measurement
      // from 0.0135 to 0.0193 relative and slipped under a 2e-2 bound, i.e. it
      // read GREEN. Widening the bound would have hidden it; making the
      // fixture able to see it is the fix.
      for (int64_t i = 0; i < R; ++i)
        lw.attn.k_rope_only_layernorm[static_cast<size_t>(i)] =
            (i % 2 == 0) ? 2.5 : 0.3;
    }
    lw.attn.kv_a_proj_with_mqa = Bf16All(lw.attn.kv_a_proj_with_mqa);
    lw.attn.k_rope_only_layernorm = Bf16All(lw.attn.k_rope_only_layernorm);
    lw.attn.q_b_proj = Bf16All(lw.attn.q_b_proj);
    lw.attn.kv_b_proj = Bf16All(lw.attn.kv_b_proj);
    lw.attn.o_proj = Bf16All(lw.attn.o_proj);
    lw.attn.g_proj = Bf16All(lw.attn.g_proj);
    lw.attn.indexer_wq_b = Bf16All(lw.attn.indexer_wq_b);
    lw.attn.indexer_wk = Bf16All(lw.attn.indexer_wk);
    lw.attn.indexer_weights_proj = Bf16All(lw.attn.indexer_weights_proj);
    lw.attn.indexer_k_norm_weight = Bf16All(lw.attn.indexer_k_norm_weight);
    lw.attn.indexer_k_norm_bias = Bf16All(lw.attn.indexer_k_norm_bias);
    lw.gate_proj = Bf16All(r.fill(s.inter * s.hidden, 0.5));
    lw.up_proj = Bf16All(r.fill(s.inter * s.hidden, 0.5));
    lw.down_proj = Bf16All(r.fill(s.hidden * s.inter, 0.5));
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The on-disk entries, in the names `EnumerateDots3NoteTensors` claims. The
// five indexer tensors are written because the loader ACCOUNTS for them; the
// device path does not read them, which is what the `index_topk` refusal is
// about.
std::vector<StOut> CheckpointOf(const DeviceSpec& s, const FullAttnDims& d,
                                const DeviceWeights& w) {
  const int64_t H = s.hidden, N = d.num_heads, QK = d.qk_head_dim();
  std::vector<StOut> e;
  e.push_back({"model.embed_tokens.weight", {s.vocab, H}, w.embed});
  e.push_back({"model.norm.weight", {H}, w.final_norm});
  if (!s.tie_word_embeddings) e.push_back({"lm_head.weight", {s.vocab, H}, w.lm_head});
  for (int64_t l = 0; l < s.layers; ++l) {
    const DeviceWeights::Layer& lw = w.layers[static_cast<size_t>(l)];
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const std::string sa = p + "self_attn.";
    e.push_back({p + "input_layernorm.weight", {H}, lw.input_ln});
    e.push_back({p + "post_attention_layernorm.weight", {H}, lw.post_ln});
    e.push_back({sa + "q_a_proj.weight", {d.q_lora_rank, H}, lw.attn.q_a_proj});
    e.push_back({sa + "q_a_layernorm.weight", {d.q_lora_rank}, lw.attn.q_a_layernorm});
    e.push_back({sa + "q_b_proj.weight", {N * QK, d.q_lora_rank}, lw.attn.q_b_proj});
    e.push_back({sa + "kv_a_proj_with_mqa.weight",
                 {d.kv_lora_rank + d.qk_rope_head_dim, H},
                 lw.attn.kv_a_proj_with_mqa});
    e.push_back({sa + "kv_a_layernorm.weight", {d.kv_lora_rank}, lw.attn.kv_a_layernorm});
    e.push_back({sa + "kv_b_proj.weight",
                 {N * (d.qk_nope_head_dim + d.v_head_dim), d.kv_lora_rank},
                 lw.attn.kv_b_proj});
    e.push_back({sa + "o_proj.weight", {H, N * d.v_head_dim}, lw.attn.o_proj});
    e.push_back({sa + "g_proj.weight", {N, H}, lw.attn.g_proj});
    e.push_back({sa + "k_rope_only_layernorm.weight",
                 {d.qk_rope_head_dim},
                 lw.attn.k_rope_only_layernorm});
    e.push_back({sa + "indexer.wq_b.weight",
                 {d.index_n_heads * d.index_head_dim, d.q_lora_rank},
                 lw.attn.indexer_wq_b});
    e.push_back({sa + "indexer.wk.weight", {d.index_head_dim, H}, lw.attn.indexer_wk});
    e.push_back({sa + "indexer.k_norm.weight",
                 {d.index_head_dim},
                 lw.attn.indexer_k_norm_weight});
    e.push_back({sa + "indexer.k_norm.bias",
                 {d.index_head_dim},
                 lw.attn.indexer_k_norm_bias});
    e.push_back({sa + "indexer.weights_proj.weight",
                 {d.index_n_heads, H},
                 lw.attn.indexer_weights_proj});
    e.push_back({p + "mlp.gate_proj.weight", {s.inter, H}, lw.gate_proj});
    e.push_back({p + "mlp.up_proj.weight", {s.inter, H}, lw.up_proj});
    // torch `nn.Linear(intermediate, hidden).weight` is [hidden, intermediate].
    e.push_back({p + "mlp.down_proj.weight", {H, s.inter}, lw.down_proj});
  }
  return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// The WHOLE-MODEL reference: W3's `ref::Forward` for the attention, and the
// residual stream / MLP / lm_head around it, all in double. The attention half
// is untouched from W3 and is a different algorithm from ours at every step.
std::vector<double> RefModel(const DeviceSpec& s, const FullAttnDims& d,
                             const DeviceWeights& w,
                             const std::vector<int32_t>& tokens,
                             const std::vector<int32_t>& positions,
                             const ref::Opts& o) {
  const int64_t T = static_cast<int64_t>(tokens.size()), H = s.hidden;
  std::vector<double> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < H; ++c) {
      hidden[static_cast<size_t>(t * H + c)] =
          w.embed[static_cast<size_t>(tokens[static_cast<size_t>(t)] * H + c)];
    }
  }
  std::vector<double> res(static_cast<size_t>(T * H), 0.0);
  for (int64_t l = 0; l < s.layers; ++l) {
    const DeviceWeights::Layer& lw = w.layers[static_cast<size_t>(l)];
    for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
    const std::vector<double> x = ref::Rms(res, lw.input_ln, T, H, s.rms_eps);
    const ref::Out a = ref::Forward(d, lw.attn, x, positions, T, o);
    for (size_t i = 0; i < res.size(); ++i) res[i] += a.out[i];
    const std::vector<double> y = ref::Rms(res, lw.post_ln, T, H, s.rms_eps);
    // `Dots3NoteMLP` == DeepseekV2MLP: down(silu(gate(y)) * up(y)).
    const std::vector<double> g = ref::Dense(y, lw.gate_proj, T, H, s.inter);
    const std::vector<double> u = ref::Dense(y, lw.up_proj, T, H, s.inter);
    std::vector<double> act(g.size());
    for (size_t i = 0; i < g.size(); ++i) act[i] = (g[i] / (1.0 + std::exp(-g[i]))) * u[i];
    hidden = ref::Dense(act, lw.down_proj, T, s.inter, H);
  }
  for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
  const std::vector<double> z = ref::Rms(res, w.final_norm, T, H, s.rms_eps);
  return ref::Dense(z, w.lm_head, T, H, s.vocab);
}

// ─────────────────────────────────────────────────────────────────────────────
// The MLA caches the forward writes into: one per layer, [blocks, block, row].
struct MlaCachePool {
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  MlaCachePool(int64_t layers, int64_t head_size, int64_t num_blocks,
               int64_t block_size) {
    for (int64_t l = 0; l < layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * block_size * head_size), 0);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
  }
};

// The paged block size every device case in this file runs on. Named because a
// batch metadata builder has to agree with the pool the forward writes into,
// and two hand-typed 8s that must match are one edit away from not matching.
constexpr int64_t kDeviceBlockSize = 8;

vllm::v1::CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  vllm::v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

// Everything a device case needs, built once. The registration, the config and
// the loaded model all come from the REAL registry over the REAL loader.
struct DeviceBench {
  DeviceSpec spec;
  TempConfig cfg;
  HfConfig config;
  Dots3NoteParams params;
  FullAttnDims dims;
  DeviceWeights w;
  std::vector<StOut> entries;
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;

  explicit DeviceBench(DeviceSpec s = DeviceSpec{})
      : spec(s),
        cfg(DeviceConfigDoc(s)),
        config(LoadHfConfig(cfg.path())),
        params(ParseDots3NoteParams(config)),
        dims(Dots3NoteFullAttnDimsFrom(params)),
        w(MakeDeviceWeights(s, dims, 0x243F6A8885A308D3ULL)),
        entries(CheckpointOf(s, dims, w)) {
    for (int64_t t = 0; t < spec.tokens; ++t) {
      tokens.push_back(static_cast<int32_t>((t * 5 + 1) % spec.vocab));
      positions.push_back(static_cast<int32_t>(t));
    }
  }

  // Load through the REAL factory and forward through `ModelRegistry::Forward`.
  // Returns the [T, vocab] f32 logits.
  std::vector<double> RunDevice() const {
    return RunDeviceWithCacheRow(params.physical_latent_row());
  }

  // `cache_row` is the MLA cache head_size the engine allocated. It normally
  // equals `physical_latent_row()`; passing anything else is how the forward's
  // own cache-row assertion is reached, which the CONFIG-level refusal cannot
  // do because an engine allocates the cache separately from the config.
  std::vector<double> RunDeviceWithCacheRow(int64_t cache_row) const {
    return RunDeviceOn(tokens, positions, PrefillMeta(spec.tokens, kDeviceBlockSize),
                       /*num_reqs=*/1, cache_row);
  }

  // The GENERAL form: a caller-supplied BATCH. `RunDeviceWithCacheRow` above is
  // this with W4a's one-request prefill metadata, so nothing an existing case
  // measures moves; what it adds is the ability to drive `num_reqs > 1` through
  // the same registry, the same loader and the same `ModelRegistry::Forward`.
  // A continuous-batching step is the shape the W4b-3c review found unreached,
  // and a bench that can only build one request cannot reach it.
  std::vector<double> RunDeviceOn(const std::vector<int32_t>& tok,
                                  const std::vector<int32_t>& pos,
                                  const vllm::v1::CommonAttentionMetadata& am,
                                  int num_reqs, int64_t cache_row) const {
    const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
    TempCheckpoint ckpt(entries);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    std::unique_ptr<vllm::LoadedModel> model = reg.factory->load_weights(reg, config, source);
    REQUIRE(model != nullptr);

    MlaCachePool pool(spec.layers, cache_row, /*num_blocks=*/2, kDeviceBlockSize);
    std::vector<vllm::GdnStateCache> gdn_state;
    vllm::v1::GDNAttentionMetadata gdn_meta{};
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const std::vector<int32_t> logits_indices;
    const vllm::ModelForwardInput input{.token_ids = tok,
                                        .positions = pos,
                                        .attn_meta = am,
                                        .gdn_meta = gdn_meta,
                                        .attn_kv = pool.attn_kv,
                                        .gdn_state = gdn_state,
                                        .config = config,
                                        .queue = queue,
                                        .logits_indices = logits_indices,
                                        .num_reqs = num_reqs};
    const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, input);
    REQUIRE(fl.on_device());
    REQUIRE(fl.rows == static_cast<int64_t>(tok.size()));
    REQUIRE(fl.vocab == spec.vocab);
    // CPU backend: the pool block IS host memory, so the f32 logits are read
    // directly. Nothing here is CUDA.
    const auto* src = static_cast<const float*>(fl.device_tensor.data);
    std::vector<double> out(static_cast<size_t>(fl.rows * fl.vocab));
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(src[i]);
    return out;
  }

  std::vector<double> RunRef(const ref::Opts& o = ref::Opts{}) const {
    return RefModel(spec, dims, w, tokens, positions, o);
  }
};

// The hidden state layer 0 actually sees: the embedding lookup of the bench's
// own tokens, before any norm. Shared by the width probe and the gate case so
// neither invents a different input from the one the device runs.
std::vector<double> HiddenOfBench(const w4a::DeviceBench& b) {
  const int64_t H = b.dims.hidden_size;
  std::vector<double> h(static_cast<size_t>(b.spec.tokens * H));
  for (int64_t t = 0; t < b.spec.tokens; ++t)
    for (int64_t c = 0; c < H; ++c)
      h[static_cast<size_t>(t * H + c)] =
          b.w.embed[static_cast<size_t>(b.tokens[static_cast<size_t>(t)] * H + c)];
  return h;
}

// The bf16 agreement bound, chosen for SEPARATION rather than to hug the
// measurement. Both arms compute the same function; the device arm stores every
// activation in bf16 (upstream's model dtype) while the reference is double
// throughout, so the residue is bf16 quantisation compounded over two layers,
// not a mechanism difference.
//
// Review finding F1 is why this comment exists. The first draft set 2e-2 with
// the residue at 1.9e-2 and a seam mutation dropping `q_lora_scale` reddening
// at 2.1e-2 — a 5% margin, one seed or one compiler from a false green on the
// single field that touches the DeepSeek-V3 q_lora branch. The cause was the
// fixture, not the number: the bench's LoRA ranks put both scales within 15% of
// the identity. With the ranks at the released model's ratio (see DeviceSpec)
// the same mutation reds at 0.761 and its kv sibling at 1.006.
//
// THREE ratios, kept separate because collapsing them is how the first draft
// looked healthy — and because a later draft collapsed two of them the OTHER
// way and overstated the headroom by 2.8x:
//   bound / residue      = 0.05   / 0.0179 = 2.8x   headroom above the floor
//   mutation / bound     = 0.761  / 0.05   = 15.2x  headroom below the nearest
//                                                   mutation
//   mutation / residue   = 0.761  / 0.0179 = 42.6x  separation of the whole
//                                                   instrument
// The middle one is the number that says this bound cannot admit a missing
// `q_lora_scale`; the last one is a statement about the fixture, not about the
// bound, and reads as far more headroom than exists if the two are merged.
constexpr double kDeviceRel = 5e-2;

}  // namespace w4a
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE(
    "dots3-note W4a: the FULL-attention layer is REACHED through "
    "ModelRegistry::Forward and agrees with W3's independent reference") {
  const w4a::DeviceBench b;
  // The scope boundary is a property of the CONFIG, checked before any weight
  // is read, so the case says which shape it is running.
  CHECK(w4a::Dots3NoteDeviceRefusal(b.params).empty());
  REQUIRE(b.params.num_hidden_layers == 2);
  for (int64_t l = 0; l < b.params.num_hidden_layers; ++l) {
    CHECK(b.params.kind_of(l) == vllm::Dots3NoteLayerKind::kFullAttention);
    CHECK_FALSE(b.params.is_moe_layer(l));
  }
  // The seam carries the two §4-trap-5 scalars, and they are DIFFERENT from
  // each other and from 1 — an equal pair would hide a swap.
  const vllm::mla::MlaBlockDims md = w4a::Dots3NoteFullAttnMlaDims(b.params);
  CHECK(md.q_lora_scale == doctest::Approx(std::sqrt(16.0 / 3.0)));
  CHECK(md.kv_lora_scale == doctest::Approx(std::sqrt(16.0 / 2.0)));
  // Both well away from 1, and in the released model's neighbourhood
  // (sqrt(5120/1024) = 2.236, sqrt(5120/512) = 3.162) — see DeviceSpec.
  CHECK(md.q_lora_scale > 2.0);
  CHECK(md.kv_lora_scale > 2.5);
  CHECK(md.q_lora_scale != md.kv_lora_scale);
  CHECK_FALSE(md.is_neox_style);
  // No YaRN on the full layers (model.py:230-238), so the softmax scale is
  // exactly qk_head_dim**-0.5 with no mscale^2 factor.
  CHECK(static_cast<double>(md.scale) == doctest::Approx(b.dims.softmax_scale()));

  const std::vector<double> got = b.RunDevice();
  const std::vector<double> want = b.RunRef();
  REQUIRE(got.size() == want.size());
  for (double v : got) REQUIRE(std::isfinite(v));
  const Diff d = Compare(got, want);
  MESSAGE("W4a device-vs-reference: max|diff| " << d.max_abs << " over scale "
                                                << d.max_mag << " = " << d.max_rel
                                                << " relative");
  CHECK(d.max_rel < w4a::kDeviceRel);
}

TEST_CASE("dots3-note W4a: the device forward is DETERMINISTIC run to run") {
  const w4a::DeviceBench b;
  const std::vector<double> a = b.RunDevice();
  const std::vector<double> c = b.RunDevice();
  REQUIRE(a.size() == c.size());
  bool same = true;
  for (size_t i = 0; i < a.size(); ++i) same = same && (a[i] == c[i]);
  CHECK(same);
}

TEST_CASE(
    "dots3-note W4a: every one of the seam's four new fields is EXERCISED — "
    "dropping any one of them moves the logits") {
  const w4a::DeviceBench b;
  const std::vector<double> got = b.RunDevice();
  const std::vector<double> full = b.RunRef();
  const double base = Compare(got, full).max_rel;

  // Each arm is the reference with ONE delta neutralised. If the DEVICE path
  // did not carry that delta, the neutralised reference would be the CLOSER
  // one — which is exactly the shape of the defect this asserts against.
  struct Arm {
    std::string what;
    ref::Opts o;
  };
  std::vector<Arm> arms;
  {
    ref::Opts a;
    a.apply_q_lora_rescale = false;  // model.py:155, the q_lora branch ONLY
    arms.push_back({"the q LoRA rescale dropped", a});
  }
  {
    ref::Opts a;
    a.apply_kv_lora_rescale = false;  // model.py:159
    arms.push_back({"the kv LoRA rescale dropped", a});
  }
  {
    ref::Opts a;
    a.apply_q_lora_rescale = false;
    a.apply_kv_lora_rescale = false;
    arms.push_back({"both LoRA rescales dropped", a});
  }
  {
    ref::Opts a;
    a.k_rope_only_norm = false;  // model.py:160
    arms.push_back({"k_rope_only_layernorm dropped", a});
  }
  {
    ref::Opts a;
    a.headwise_gate = false;  // model.py:191-197 -> the lane-wise arm
    arms.push_back({"the headwise gate made lane-wise", a});
  }
  for (const Arm& a : arms) {
    CAPTURE(a.what);
    const Diff d = Compare(got, b.RunRef(a.o));
    MESSAGE("W4a " << a.what << ": device-vs-neutralised " << d.max_rel
                   << " relative, against " << base << " with it");
    CHECK(d.max_rel > 10.0 * base);
  }
}

TEST_CASE(
    "dots3-note W4a: the headwise gate's WIDTHS are gated — the logit is bf16 "
    "like upstream, and the one rounding step we do not mirror is measured") {
  // W3 recorded this as owed: its double reference could not see the dtype at
  // all. The device path can, and there are TWO widths to answer for, not one
  // (review finding F2 — the first draft claimed one and was wrong).
  //
  //   the LOGIT   upstream builds `g_proj` with no `params_dtype`
  //               (model.py:292-297), so it inherits the model dtype and the
  //               sigmoid input is BF16, widened by `.float()`. Ours is a bf16
  //               GEMM store widened by `vt::CastF32`. MIRRORED — and part (a)
  //               gates it rather than asserting it.
  //   the SIGMOID upstream rounds it to bf16 and multiplies in bf16
  //               (`.to(attn_out.dtype)`, model.py:196-197), so the product is
  //               rounded twice; `vt::SharedExpertGate` keeps the sigmoid in
  //               f32 and rounds only the product. NOT mirrored, one rounding
  //               fewer, and part (b) measures what it costs.
  const w4a::DeviceBench b;
  const FullAttnDims& d = b.dims;

  // (a) THE LOGIT WIDTH. It is mirrored, and — this is the part worth stating —
  //     no value gate on a bf16 output can confirm that, so it is checked
  //     against the SOURCE and held here by an analytic bound plus a
  //     measurement that says which of the two it is.
  //
  //     ANALYTIC, on the ABSOLUTE change in the gate. Rounding the logit moves
  //     it by |dsigma| = sigma(1-sigma)*|dx| <= max_x[sigma(1-sigma)*|x|] * 2^-9
  //     = 0.2239 * 2^-9 = 4.38e-4, since a bf16 half-ulp is 2^-9 RELATIVE and
  //     the peak of sigma(1-sigma)|x| is 0.2239 at |x| ~ 1.54. The gate is a
  //     multiplier in [0,1], so 4.38e-4 is also its worst absolute effect on
  //     the gated output's scale.
  //
  //     The RELATIVE change is deliberately NOT claimed as bounded, because it
  //     is not: at x -> -inf the gate vanishes while |dsigma/sigma| = (1-sigma)|x|
  //     grows without limit. A first draft asserted 0.2785*2^-9 for the
  //     relative form and was WRONG for exactly that reason — it scanned only
  //     positive logits. The measured relative figure is reported beside the
  //     store's own granularity instead, as an observation about this fixture.
  //
  //     A first draft also tried to resolve the width by asking whether the
  //     device sits closer to a bf16-logit reference than to a f64 one. It does
  //     not, and cannot at whole-model scale: the two differ by 0.0186 against
  //     0.0179, which is the bf16 residue reshuffling rather than a measurement
  //     of the width. That arm is deleted rather than tuned, because a
  //     comparison that cannot resolve its subject is not evidence. This is
  //     porting.md's "a token gate cannot catch a dtype that is too WIDE" with
  //     the "cannot" quantified instead of quoted.
  const ref::Out probe = ref::Forward(d, b.w.layers[0].attn, HiddenOfBench(b),
                                      b.positions, b.spec.tokens, ref::Opts{});
  double worst_abs = 0.0, worst_rel = 0.0;
  for (int64_t t = 0; t < b.spec.tokens; ++t) {
    for (int64_t h = 0; h < d.num_heads; ++h) {
      // Recover the logit from the gate the reference reported: a sigmoid is
      // invertible, so this needs no second forward.
      const double g = probe.gate[static_cast<size_t>(t * d.num_heads + h)];
      const double x = std::log(g / (1.0 - g));
      const double gx = 1.0 / (1.0 + std::exp(-w4a::Bf16(x)));
      worst_abs = std::max(worst_abs, std::abs(gx - g));
      worst_rel = std::max(worst_rel, std::abs(gx - g) / g);
    }
  }
  MESSAGE("W4a gate-logit width: rounding the logit to bf16 (upstream's width) "
          << "moves the gate by <= " << worst_abs << " absolute and "
          << worst_rel << " relative, against a bf16 STORE half-ulp of "
          << std::pow(2.0, -9));
  // ANALYTIC, holds for any fixture.
  CHECK(worst_abs <= 0.2239 * std::pow(2.0, -9));
  // MEASURED on this fixture: the relative move stays under the store's own
  // granularity, so the two widths are indistinguishable in the output here.
  // Not a bound — see the note above on why no relative bound exists.
  CHECK(worst_rel < std::pow(2.0, -9));

  // (b) THE SIGMOID ROUNDING, measured. `attn_out` and the gate come from a
  //     real forward of layer 0's attention on the real embedded hidden state.
  ref::Opts narrow;
  narrow.bf16_gate_logit = true;  // upstream's logit width (model.py:292-297)
  const ref::Out o = ref::Forward(d, b.w.layers[0].attn, HiddenOfBench(b),
                                  b.positions, b.spec.tokens, narrow);
  REQUIRE(!o.gate.empty());
  double worst_gate = 0.0, worst_prod = 0.0, scale = 0.0;
  for (int64_t t = 0; t < b.spec.tokens; ++t) {
    for (int64_t h = 0; h < d.num_heads; ++h) {
      const double g = o.gate[static_cast<size_t>(t * d.num_heads + h)];
      worst_gate = std::max(worst_gate, std::abs(w4a::Bf16(g) - g));
      for (int64_t v = 0; v < d.v_head_dim; ++v) {
        const double a =
            o.attn_out[static_cast<size_t>(t * d.num_heads * d.v_head_dim +
                                           h * d.v_head_dim + v)];
        const double ours = w4a::Bf16(g * a);               // one rounding
        const double theirs = w4a::Bf16(w4a::Bf16(g) * a);  // upstream's two
        worst_prod = std::max(worst_prod, std::abs(ours - theirs));
        scale = std::max(scale, std::abs(ours));
      }
    }
  }
  MESSAGE("W4a sigmoid rounding: |bf16(sigmoid)-sigmoid| <= "
          << worst_gate << "; the extra rounding upstream applies moves the "
          << "gated output by <= " << worst_prod << " over a scale of " << scale);
  // ANALYTIC: a sigmoid is in (0,1), so its bf16 rounding is at most half an
  // ulp at 1.0. This one holds for any fixture.
  CHECK(worst_gate <= std::pow(2.0, -9));
  // EMPIRICAL, and said so rather than dressed as a bound: this is what the
  // product difference MEASURED on this fixture, not a proof about every input.
  // Recorded so a future change that widens it becomes visible.
  CHECK(worst_prod <= scale * std::pow(2.0, -7));
}
TEST_CASE("dots3-note W4a: what the device path still REFUSES, by name") {
  // (a) a sliding layer — LIFTED at W4b-2. It is kept here as an ACCEPTANCE
  //     rather than deleted, because a reader following W4a's evidence lands
  //     on the answer instead of a gap: the same config W4a refused by name is
  //     now empty, and the W4b-2 cases below run it end to end.
  {
    w4a::DeviceSpec s;
    nlohmann::json doc = w4a::DeviceConfigDoc(s);
    doc["layer_types"] = nlohmann::json::array({"full_attention", "sliding_attention"});
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4a::Dots3NoteDeviceRefusal(p).empty());
  }
  // (b) a MoE layer — LIFTED at W5, and kept as an ACCEPTANCE for the same
  //     reason as (a): a reader following W4a's evidence lands on the answer.
  //     `Dots3NoteMoeBlock` over `vllm::RunMoePlaced` runs it, and the W5 cases
  //     at the bottom of this file gate it against an independent reference.
  {
    w4a::DeviceSpec s;
    nlohmann::json doc = w4a::DeviceConfigDoc(s);
    doc["first_k_dense_replace"] = 1;  // layer 1 becomes MoE
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4a::Dots3NoteDeviceRefusal(p).empty());
  }
  // (c) the RELEASED config — LIFTED at W5 (the MoE layer) and W5c (the nextn
  //     tail, #2176). It is EMPTY now, which is the row's headline and is
  //     asserted here as the `true -> false` flip rather than deleted.
  {
    TempConfig cfg(FixtureConfigDoc());
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4a::Dots3NoteDeviceRefusal(p).empty());
  }
  // (d) a sequence past `index_topk` — LIFTED at W4b-3c for a SINGLE-SHOT
  //     prefill, and kept here as an ACCEPTANCE for the same reason as (a) and
  //     (e): a reader following W4a's evidence lands on the answer instead of a
  //     gap. The step W4a refused by name now runs SPARSELY, and the W4b-3c
  //     cases below gate it against the reference that selects. What still
  //     refuses is a RESUMED request, which would need the indexer's own key
  //     cache (#1925) — its case is beside them.
  {
    w4a::DeviceSpec s;
    s.index_topk = 2;  // < tokens
    const w4a::DeviceBench b(s);
    CHECK(w4a::Dots3NoteDenseEquivalentMaxSeqLen(b.params) == 2);
    CHECK(w4a::Dots3NoteDeviceRefusal(b.params).empty());  // the CONFIG is fine
    CHECK_NOTHROW((void)b.RunDevice());
  }
  // (e) a PADDED physical latent row — LIFTED at W4b-2, and kept here for the
  //     same reason (a): the config W4a refused at CONFIG level now passes,
  //     and `Tensor::Slice(2, 0, logical)` is what narrows it on read.
  {
    w4a::DeviceSpec s;
    nlohmann::json doc = w4a::DeviceConfigDoc(s);
    doc["swa_kv_lora_rank"] = s.kv_lora + 3;  // physical row wider than logical
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    REQUIRE(p.physical_latent_row() > p.full.latent_row());
    CHECK(w4a::Dots3NoteDeviceRefusal(p).empty());
  }
  // (f) a nextn tail — LIFTED at W5c (#2176), and this one was a DEFECT rather
  //     than a gap: the refusal was STRICTER THAN UPSTREAM. vLLM does not turn
  //     a checkpoint with nextn weights away, it DROPS them from the main model
  //     (utils.py:542 -> deepseek_v2.py:1618-1620; model.py:624 @ bc2d63e650).
  //     They are a NAMED W10 deferral in the accounting now, which is what
  //     `Dots3NoteIsNextnTensor` is for.
  {
    w4a::DeviceSpec s;
    nlohmann::json doc = w4a::DeviceConfigDoc(s);
    doc["num_nextn_predict_layers"] = 1;
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4a::Dots3NoteDeviceRefusal(p).empty());
    CHECK(vllm::Dots3NoteIsNextnTensor(
        p, "model.layers." + std::to_string(p.num_hidden_layers) + ".enorm.weight"));
    CHECK(vllm::Dots3NoteIsNextnTensor(p, "model.mtp.embed_tokens.weight"));
  }
  // (g) a KV cache whose row disagrees with the config it was built from. The
  //     config-level check above cannot see this — an engine allocates the
  //     cache separately — so the forward keeps its own assertion, and this is
  //     what makes that assertion REACHED rather than defensive decoration.
  {
    const w4a::DeviceBench b;
    CHECK(w4a::Dots3NoteDeviceRefusal(b.params).empty());
    // W4b-2 changed what this check COMPARES AGAINST — the physical row rather
    // than the full arm's logical one — because a padded row is now legal and
    // the allocator is told the physical number. The message moved with it.
    CHECK_THROWS_WITH_AS(b.RunDeviceWithCacheRow(b.params.physical_latent_row() + 4),
                         doctest::Contains("PHYSICAL row"), std::runtime_error);
  }
}

TEST_CASE("dots3-note W4a: a weight of the WRONG shape refuses BY NAME at load") {
  w4a::DeviceSpec s;
  const w4a::DeviceBench b(s);
  std::vector<w4a::StOut> bad = b.entries;
  // Truncate `g_proj` to one head. A silently short gate is exactly the class
  // of defect this row has no oracle to catch.
  for (w4a::StOut& e : bad) {
    if (e.name == "model.layers.0.self_attn.g_proj.weight") {
      e.shape = {1, s.hidden};
      e.values.resize(static_cast<size_t>(s.hidden));
    }
  }
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(b.config);
  w4a::TempCheckpoint ckpt(bad);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  CHECK_THROWS_WITH_AS((void)reg.factory->load_weights(reg, b.config, source),
                       doctest::Contains("g_proj"), std::runtime_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// W4b-3c — the DSA lightning indexer's SELECTION, on the device path.
//
// ─── WHAT THESE CASES ESTABLISH ──────────────────────────────────────────────
// A prompt longer than `index_topk` is now served SPARSELY, through the
// production chain and nothing else: `ModelRegistry::Resolve` -> the real
// `load_weights` over a real `SafetensorsFile` -> `ModelRegistry::Forward` ->
// `Dots3NoteModel::ForwardDevice` -> `mla::ForwardMlaAttentionBlock` -> the
// indexer ops and the selected-slot MLA decode. No case here constructs the
// block, the dims or the indexer weights by hand; a unit test that did would
// prove the class works and never that anything reaches it.
//
// THE ORACLE is W3's independent double reference, which has selected since W3
// — `ref::Forward` runs the whole indexer (`wq_b`, the merged `wk`/
// `weights_proj`, the `LayerNorm` `k_norm` at the upstream literal 1e-6, the
// leading-slice GPT-J rope, the ReLU'd MQA logit and the top-k) and then
// attends over the SELECTED keys only. It is a different algorithm at every
// step: a full stable sort rather than a partial selection, `long double`
// accumulation, and a softmax with no max subtraction.
//
// THE CONTROL that says the selection BITES is `ref::Opts::run_indexer =
// false` — the same reference with the selection replaced by the whole causal
// window, i.e. exactly what a port that missed `is_sparse` would compute. The
// device must be far from it, or every agreement above would also hold for a
// forward that never selected anything.
namespace w4b3c {

// The same bench W4a uses, at an `index_topk` SHORTER than the prompt so the
// top-k really prunes. Nothing else about the geometry changes, which is what
// makes the comparison against W4a's own numbers meaningful.
w4a::DeviceSpec SparseSpec() {
  w4a::DeviceSpec s;
  s.index_topk = 2;  // < s.tokens (6), so rows 2..5 select rather than take all
  return s;
}

}  // namespace w4b3c

TEST_CASE(
    "dots3-note W4b-3c: a prompt past `index_topk` is served SPARSELY, reached "
    "through ModelRegistry::Forward, and agrees with the selecting reference") {
  const w4a::DeviceBench b(w4b3c::SparseSpec());
  REQUIRE(w4a::Dots3NoteDenseEquivalentMaxSeqLen(b.params) == 2);
  REQUIRE(b.spec.tokens > b.params.index_topk);
  // The CONFIG is runnable and the step is no longer refused — that is the lift.
  REQUIRE(w4a::Dots3NoteDeviceRefusal(b.params).empty());

  // THE INSTRUMENT SAYS WHAT IT MEASURED, before anything is asserted about it.
  // Below the pruning threshold every comparison in this case would pass on a
  // forward that performed no selection at all.
  const ref::Out probe = ref::Forward(b.dims, b.w.layers[0].attn, w4a::HiddenOfBench(b),
                                      b.positions, b.spec.tokens, ref::Opts{});
  MESSAGE("W4b-3c selection: " << probe.rows_where_topk_pruned << " of " << b.spec.tokens
                               << " rows really prune ("
                               << probe.rows_decided_by_a_strict_margin
                               << " by a strict margin, " << probe.rows_decided_by_a_tie
                               << " by an exact tie), min strict margin "
                               << probe.min_strict_margin);
  // Rows 0 and 1 have 1 and 2 causal candidates against `index_topk` 2, so they
  // take the short-context all-select path; rows 2..5 prune. Hand-derived from
  // the causal rule, not from the code under test.
  REQUIRE(probe.rows_where_topk_pruned == b.spec.tokens - b.params.index_topk);
  int64_t keys_dropped = 0;
  for (int64_t t = b.params.index_topk; t < b.spec.tokens; ++t) {
    keys_dropped += (t + 1) - b.params.index_topk;
  }
  MESSAGE("W4b-3c selection drops " << keys_dropped << " causal keys in total");
  // Rows 2..5 see 3..6 causal keys and keep `index_topk` 2, so they drop
  // 1, 2, 3 and 4 keys. Hand-derived from the causal rule.
  REQUIRE(keys_dropped == 1 + 2 + 3 + 4);

  const std::vector<double> got = b.RunDevice();
  const std::vector<double> want = b.RunRef();
  const Diff d = Compare(got, want);
  MESSAGE("W4b-3c sparse device vs the selecting reference: max|diff| "
          << d.max_abs << " over a scale of " << d.max_mag << " = " << d.max_rel
          << " relative; bound " << w4a::kDeviceRel << " = "
          << (w4a::kDeviceRel / d.max_rel) << "x the residue");
  CHECK(d.max_rel < w4a::kDeviceRel);

  // THE CONTROL. `run_indexer = false` is the whole causal window — what a port
  // that missed `is_sparse` computes, and what W4a served before the lift. The
  // device must be FAR from it, or the agreement above says nothing.
  ref::Opts dense;
  dense.run_indexer = false;
  const Diff dd = Compare(got, b.RunRef(dense));
  MESSAGE("W4b-3c against the NON-selecting reference: " << dd.max_rel
                                                         << " relative, against " << d.max_rel
                                                         << " with the selection");
  CHECK(dd.max_rel > 10.0 * d.max_rel);
}

TEST_CASE(
    "dots3-note W4b-3c: the indexer's rope POLARITY is independent of the main "
    "rope's, and the DEVICE honours the config flag") {
  // Spec section 4 trap 2. The main MLA rope is GPT-J on both dots3 geometries
  // (`is_neox_style=False`, deepseek_v2.py:1108), while the INDEXER rope
  // follows `is_neox_style = not indexer_rope_interleave` (`:1159`). On the
  // released config the flag resolves so that both are GPT-J, which is exactly
  // why a port can read one from the other and never notice — the mutation
  // "the indexer rope pairing follows the MAIN rope" SURVIVES on that config.
  //
  // This case drives the two APART. With `indexer_rope_interleave = false` the
  // indexer rope is NeoX while the main rope stays GPT-J, and the device must
  // follow the INDEXER's flag. W3 measured the difference at 7 of 24 selection
  // slots on its own bench; here it has to move the whole model's logits.
  w4a::DeviceSpec s = w4b3c::SparseSpec();
  s.indexer_rope_interleave = false;
  const w4a::DeviceBench b(s);
  REQUIRE(b.params.indexer_rope_is_neox_style());       // the INDEXER: NeoX
  REQUIRE_FALSE(b.params.full.rope_is_neox_style);      // the MAIN rope: GPT-J

  const std::vector<double> got = b.RunDevice();
  ref::Opts neox;
  neox.indexer_rope_neox = true;  // the reference follows the same flag
  const Diff d = Compare(got, b.RunRef(neox));
  ref::Opts gptj;  // what a port that read the MAIN rope's flag would compute
  const Diff wrong = Compare(got, b.RunRef(gptj));
  MESSAGE("W4b-3c indexer rope polarity: device-vs-NeoX-reference " << d.max_rel
          << " relative, device-vs-GPT-J-reference " << wrong.max_rel);
  CHECK(d.max_rel < w4a::kDeviceRel);
  CHECK(wrong.max_rel > 10.0 * d.max_rel);
}

TEST_CASE(
    "dots3-note W4b-3c: the sparse route is the STEP's decision, and a short "
    "prompt keeps the dense answer BYTE for byte") {
  // Upstream's own rule: `use_dense_mha = prefill_max_seq_len <= topk_tokens`
  // (sparse_mla_attention.py:296-299 @ `bc2d63e650`). At or below the threshold
  // the top-k selects every causal candidate, so dense attention IS upstream's
  // answer and the step must not move — which is what keeps W4a's and W4b-2's
  // gates valid across this change. Asserted BIT for bit rather than to a
  // tolerance, because "the route was not taken" is a statement about control
  // flow and a tolerance cannot make it.
  w4a::DeviceSpec s;
  s.index_topk = s.tokens;  // exactly at the threshold: nothing prunes
  const w4a::DeviceBench at_threshold(s);
  const std::vector<double> a = at_threshold.RunDevice();
  const std::vector<double> b2 = at_threshold.RunDevice();
  REQUIRE(a.size() == b2.size());
  for (size_t i = 0; i < a.size(); ++i) REQUIRE(a[i] == b2[i]);

  // And the DEFAULT bench, whose `index_topk` is comfortably above the prompt,
  // is byte-identical to the at-threshold one: neither takes the sparse route,
  // so the only thing that could differ is a route that fired when it must not.
  const w4a::DeviceBench wide;
  const std::vector<double> c = wide.RunDevice();
  REQUIRE(c.size() == a.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK(c[i] == a[i]);
}

// ═════════════════════════════════════════════════════════════════════════════
// W4b-1 — the SLIDING arm, the §2.3 machinery and the padded KV row.
//
// ─── WHAT THIS ESTABLISHES, AND WHAT IT CANNOT ───────────────────────────────
// Same instrument, same limits as W3/W4a and for the same reason: spec §6.4
// option B. There is NO vLLM oracle for `dots3_note` on any host this project
// owns, so the strongest available statement is that two independently written
// implementations of the upstream python agree, plus a PROPERTY per mechanism
// that a plausible-but-wrong port breaks. Neither says our answer is vLLM's.
//
// ─── WHY THE REFERENCE IS INDEPENDENT, CONCRETELY ────────────────────────────
// The implementation computes the sliding attention the way UPSTREAM does:
// the ABSORBED MQA of `_forward_swa_mqa` (attention.py:470-563) over a PAGED,
// PADDED latent cache, gathered by `GatherSwaKv` and windowed by
// `ApplySwaScoreMask` — i.e. a window derived from `gather_start`, `GATHER_LEN`
// and a per-slot `valid` flag.
//
// The reference below takes the other route at every level:
//   * MATERIALIZED MHA — `kv_b_proj` up-projects the latent into per-head K/V
//     and the attention is a plain dot product, with no `W_UK` / `W_UV` fold
//     and no latent-space intermediate at all;
//   * NO CACHE — key `s` is token `s`, so there is no paging, no block table,
//     no `slot_mapping` and no padded row;
//   * the window is the DIRECT positional predicate `s <= t && t - s < W`,
//     never `gather_start + slot` arithmetic;
//   * softmax WITHOUT the max subtraction, in `long double`;
//   * the rotation is a complex multiply with angles recomputed per element.
// So the two arms share the geometry and nothing else, and a defect in any of
// the four §2.3 mechanisms moves the implementation and not the reference.
//
// ─── WHAT IS NOT REACHED ─────────────────────────────────────────────────────
// The DEVICE path. `Dots3NoteModel::ForwardDevice` still refuses a
// `sliding_attention` layer and a PADDED physical row by name, and the last
// case here asserts both refusals so the boundary is executable rather than a
// comment. Putting the sliding arm on the decode path is W4b-2; the header
// argues why it is a separate brick and `## Owed` records it.
// ═════════════════════════════════════════════════════════════════════════════

namespace {
namespace w4b {

using vllm::dots3_note::ApplySwaScoreMask;
using vllm::dots3_note::BuildSlidingWindowMetadata;
using vllm::dots3_note::Dots3NoteSlidingAttnDimsFrom;
using vllm::dots3_note::ForwardSlidingAttention;
using vllm::dots3_note::GatherSwaKv;
using vllm::dots3_note::kSwaMaskedScore;
using vllm::dots3_note::NarrowLogicalCacheRows;
using vllm::dots3_note::PaddedMlaCacheSpec;
using vllm::dots3_note::SlidingAttnDims;
using vllm::dots3_note::SlidingAttnTrace;
using vllm::dots3_note::SlidingAttnWeights;
using vllm::dots3_note::SlidingPaging;
using vllm::dots3_note::SlidingWindowChunk;
using vllm::dots3_note::SwaGatherLen;
using vllm::dots3_note::SwaGatherResult;
using vllm::dots3_note::WritePaddedMlaCache;

// ─────────────────────────────────────────────────────────────────────────────
// The tiny SLIDING bench. Every dimension is the smallest that still makes its
// mechanism OBSERVABLE, which is spec §4.6's review finding F1 applied before
// the fact rather than after it:
//
//   * `window` 3 against `tokens` 8, so FIVE query rows really lose a key. At
//     window >= tokens the windowed answer IS the causal answer and every
//     assertion in this file would pass on a port with no window at all. The
//     count is asserted BY NUMBER.
//   * `SwaGatherLen(3, 8) == 16`, which is a REAL round-up from 10, so eight
//     gathered slots are past the sequence and must come back invalid. A
//     gather that ignored `valid` would read zeros and score them.
//   * `page_size` 3 against 8 tokens is THREE pages, and the block table is
//     SHUFFLED to {2, 0, 1}. A contiguous table makes a paged read and a flat
//     read the same answer, which would leave the block lookup unproven.
//   * `swa_kv_lora` 6 against the FULL arm's 4, so the physical row (6+4=10)
//     is genuinely WIDER than a full layer's logical row (4+4=8) and the
//     padding is two real lanes rather than zero.
//   * `swa_q_lora` 3 against `hidden` 16 gives rescales sqrt(16/3)=2.309 and
//     sqrt(16/6)=1.633 — both far from 1.0 and DIFFERENT from each other, so
//     neither a missing nor a swapped scale can hide. (Upstream's released
//     ranks make the two sliding scales EQUAL at 2.236; equal scales in a
//     fixture would hide a swap, so the fixture deliberately does not copy
//     them, and the geometry case pins the released values separately.)
//   * `swa_qk_nope` 8 against the full arm's 4, mirroring upstream's 192-vs-128
//     — so the two softmax scales differ (12^-0.5 vs 8^-0.5) and a layer that
//     used the wrong geometry's scale is visible. 8 rather than 6 because at 6
//     the sliding `qk_head_dim` (6+4) would EQUAL its `latent_row` (6+4), and
//     "scale by the latent row instead of qk_head_dim" — a real confusion, since
//     the absorbed MQA dots over 1088 lanes and scales by 256^-0.5 — would be a
//     numeric no-op. A mutation measured that: it reddened only the
//     released-config assertion, so the fixture changed rather than the row.
//   * `swa_heads` 3 against the full arm's 2, mirroring upstream's 64-vs-128.
//     Equal head counts would make "read the FULL arm's head count" a no-op,
//     which is the same disease one field over.
//   * `swa_rope_theta` 41 against the full arm's 137: the released model's two
//     thetas are three orders apart and the fixture keeps them distinct.
// ─────────────────────────────────────────────────────────────────────────────
struct SwaSpec {
  int64_t hidden = 16;
  int64_t full_heads = 2;
  int64_t full_qk_nope = 4;
  int64_t full_kv_lora = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 4;
  int64_t q_lora = 3;
  int64_t swa_heads = 3;
  int64_t swa_qk_nope = 8;
  int64_t swa_kv_lora = 6;
  int64_t window = 3;
  double rope_theta = 137.0;
  double swa_rope_theta = 41.0;
  double rms_eps = 1e-3;
  int64_t tokens = 8;
  int64_t page_size = 3;
};

nlohmann::json SwaConfigDoc(const SwaSpec& s) {
  nlohmann::json d = FixtureConfigDoc();
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = 4;
  d["layer_types"] = nlohmann::json::array({"full_attention", "full_attention",
                                            "sliding_attention",
                                            "sliding_attention"});
  d["num_attention_heads"] = s.full_heads;
  d["num_key_value_heads"] = s.full_heads;
  d["qk_nope_head_dim"] = s.full_qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.full_kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["index_n_heads"] = 2;
  d["index_head_dim"] = 6;
  d["index_topk"] = 3;
  d["swa_num_attention_heads"] = s.swa_heads;
  d["swa_num_key_value_heads"] = s.swa_heads;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.swa_kv_lora;
  d["swa_qk_nope_head_dim"] = s.swa_qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["swa_rope_theta"] = s.swa_rope_theta;
  d["sliding_window_size"] = s.window;
  d["vocab_size"] = 32;
  d["intermediate_size"] = 12;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  return d;
}

Dots3NoteParams SwaParams(const SwaSpec& s) {
  TempConfig cfg(SwaConfigDoc(s));
  return ParseDots3NoteParams(LoadHfConfig(cfg.path()));
}

SlidingAttnWeights SwaWeights(const SlidingAttnDims& d, uint64_t seed) {
  Rng r(seed);
  const int64_t H = d.hidden_size;
  SlidingAttnWeights w;
  w.q_a_proj = r.fill(d.q_lora_rank * H, 0.5);
  w.kv_a_proj_with_mqa = r.fill((d.kv_lora_rank + d.qk_rope_head_dim) * H, 0.5);
  w.q_a_layernorm = r.fill(d.q_lora_rank, 0.3);
  for (double& x : w.q_a_layernorm) x += 1.0;
  w.kv_a_layernorm = r.fill(d.kv_lora_rank, 0.3);
  for (double& x : w.kv_a_layernorm) x += 1.0;
  // ALTERNATING 2.5 / 0.3 within each rotated pair, for the reason spec §4.6
  // records: RoPE preserves the L2 norm of a rotated pair exactly, so the
  // norm's ORDER commutes with the rotation whenever `w[2i] == w[2i+1]`, and
  // at weights hugging 1.0 the "norm AFTER the rope" defect slips under any
  // sane bound. Only a per-lane weight that differs WITHIN the pair fails to
  // commute, which is what makes M-after-rope observable here.
  w.k_rope_only_layernorm.resize(static_cast<size_t>(d.qk_rope_head_dim));
  for (int64_t i = 0; i < d.qk_rope_head_dim; ++i) {
    w.k_rope_only_layernorm[static_cast<size_t>(i)] = (i % 2 == 0) ? 2.5 : 0.3;
  }
  w.q_b_proj = r.fill(d.num_heads * d.qk_head_dim() * d.q_lora_rank, 0.5);
  w.kv_b_proj = r.fill(
      d.num_heads * (d.qk_nope_head_dim + d.v_head_dim) * d.kv_lora_rank, 0.5);
  w.o_proj = r.fill(H * d.num_heads * d.v_head_dim, 0.5);
  w.g_proj = r.fill(d.num_heads * H, 0.7);
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────
// THE INDEPENDENT REFERENCE for the sliding arm. It reuses `ref::Dense`,
// `ref::Rms` and `ref::Rotate` — which are themselves the independent
// transcriptions W3 landed and a reviewer proved independent by mutating the
// shared helper the IMPLEMENTATION routes through (spec §4.5, R9) — and adds
// the one thing the sliding arm needs: a windowed materialized MHA.
// ─────────────────────────────────────────────────────────────────────────────
namespace sref {

struct Opts {
  bool apply_q_lora_rescale = true;   // model.py:155
  bool apply_kv_lora_rescale = true;  // model.py:159
  bool k_rope_only_norm = true;       // model.py:160
  bool headwise_gate = true;          // model.py:191-197
  // FALSE models a port that never noticed `sliding_window_size` and ran plain
  // causal attention — the defect the whole brick exists to prevent, and the
  // one a shape check cannot see.
  bool windowed = true;
  // The rope theta a port would use if it inherited the model-level value
  // instead of reading `swa_rope_theta` (model.py:406).
  double rope_theta_override = 0.0;
};

struct Out {
  std::vector<double> out;          // [T, hidden]
  std::vector<double> q_c;          // [T, q_lora_rank]
  std::vector<double> kv_c_normed;  // [T, kv_lora_rank]
  std::vector<double> k_pe;         // [T, qk_rope]
  std::vector<double> q;            // [T, heads, qk_head_dim]
  std::vector<double> attn_out;     // [T, heads*v]
  std::vector<double> gate;         // [T, heads]
  std::vector<double> gated;        // [T, heads*v]
  // Instrument self-reporting, so a case can say what it MEASURED rather than
  // assume the fixture bit. `queries_that_lost_a_key` is the whole reason this
  // bench uses window 3 against 8 tokens.
  int64_t queries_that_lost_a_key = 0;
  int64_t keys_dropped_by_the_window = 0;
  double max_abs_score = 0.0;
};

Out Forward(const SlidingAttnDims& d, const SlidingAttnWeights& w,
            const std::vector<double>& hidden,
            const std::vector<int32_t>& positions, int64_t T, const Opts& o) {
  const int64_t H = d.hidden_size;
  const int64_t N = d.num_heads;
  const int64_t P = d.qk_nope_head_dim;
  const int64_t R = d.qk_rope_head_dim;
  const int64_t V = d.v_head_dim;
  const int64_t QK = P + R;
  const int64_t L = d.kv_lora_rank;
  const double theta =
      o.rope_theta_override > 0.0 ? o.rope_theta_override : d.rope_theta;
  Out r;

  std::vector<double> q_c = ref::Dense(hidden, w.q_a_proj, T, H, d.q_lora_rank);
  q_c = ref::Rms(q_c, w.q_a_layernorm, T, d.q_lora_rank, d.rms_norm_eps);
  if (o.apply_q_lora_rescale) {
    for (double& v : q_c) v *= d.q_lora_scale;
  }

  const std::vector<double> kv_lora =
      ref::Dense(hidden, w.kv_a_proj_with_mqa, T, H, L + R);
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
  std::vector<double> kv_c_normed =
      ref::Rms(kv_c, w.kv_a_layernorm, T, L, d.rms_norm_eps);
  if (o.apply_kv_lora_rescale) {
    for (double& v : kv_c_normed) v *= d.kv_lora_scale;
  }
  if (o.k_rope_only_norm) {
    k_pe = ref::Rms(k_pe, w.k_rope_only_layernorm, T, R, d.rms_norm_eps);
  }

  std::vector<double> q = ref::Dense(q_c, w.q_b_proj, T, d.q_lora_rank, N * QK);
  ref::Rotate(q, positions, theta, T, N, QK, /*offset=*/P, R,
              d.rope_is_neox_style);
  ref::Rotate(k_pe, positions, theta, T, 1, R, /*offset=*/0, R,
              d.rope_is_neox_style);

  // The MATERIALIZED MHA. No absorption, no latent-space intermediate, no
  // cache, and the window is the direct positional predicate.
  const std::vector<double> kv =
      ref::Dense(kv_c_normed, w.kv_b_proj, T, L, N * (P + V));
  const double scale = std::pow(static_cast<double>(QK), -0.5);
  std::vector<double> attn(static_cast<size_t>(T * N * V), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<int64_t> keys;
    for (int64_t s = 0; s <= t; ++s) {
      // attention.py:151-152 as ONE predicate over token indices: with
      // seq_len == query_len == T the kv position IS `s` and the query
      // position IS `t`, so `kv_pos >= query_pos - W + 1` is `t - s < W`.
      if (o.windowed && t - s >= d.sliding_window) continue;
      keys.push_back(s);
    }
    if (static_cast<int64_t>(keys.size()) < t + 1) {
      ++r.queries_that_lost_a_key;
      r.keys_dropped_by_the_window += (t + 1) - static_cast<int64_t>(keys.size());
    }
    for (int64_t h = 0; h < N; ++h) {
      std::vector<long double> ex(keys.size());
      long double denom = 0.0L;
      for (size_t i = 0; i < keys.size(); ++i) {
        const int64_t s = keys[i];
        long double dot = 0.0L;
        for (int64_t c = 0; c < P; ++c) {
          dot += static_cast<long double>(q[static_cast<size_t>((t * N + h) * QK + c)]) *
                 static_cast<long double>(
                     kv[static_cast<size_t>((s * N + h) * (P + V) + c)]);
        }
        for (int64_t c = 0; c < R; ++c) {
          dot += static_cast<long double>(
                     q[static_cast<size_t>((t * N + h) * QK + P + c)]) *
                 static_cast<long double>(k_pe[static_cast<size_t>(s * R + c)]);
        }
        const long double score = dot * static_cast<long double>(scale);
        r.max_abs_score = std::max(
            r.max_abs_score, static_cast<double>(score < 0 ? -score : score));
        ex[i] = std::exp(score);  // no max subtraction, on purpose
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

  const std::vector<double> logits = ref::Dense(hidden, w.g_proj, T, H, N);
  std::vector<double> gated(attn.size());
  r.gate.assign(static_cast<size_t>(T * N), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < N; ++h) {
      const double g =
          1.0 / (1.0 + std::exp(-logits[static_cast<size_t>(t * N + h)]));
      r.gate[static_cast<size_t>(t * N + h)] = g;
      for (int64_t c = 0; c < V; ++c) {
        const size_t at = static_cast<size_t>((t * N + h) * V + c);
        const double use =
            o.headwise_gate
                ? g
                : 1.0 / (1.0 + std::exp(-logits[static_cast<size_t>(t * N)]));
        gated[at] = attn[at] * use;
      }
    }
  }
  r.out = ref::Dense(gated, w.o_proj, T, N * V, H);
  r.q_c = q_c;
  r.kv_c_normed = kv_c_normed;
  r.k_pe = k_pe;
  r.q = q;
  r.attn_out = attn;
  r.gated = gated;
  return r;
}

}  // namespace sref

struct SwaBench {
  SwaSpec spec;
  Dots3NoteParams params;
  SlidingAttnDims dims;
  SlidingAttnWeights w;
  std::vector<double> hidden;
  std::vector<int32_t> positions;
  SlidingPaging paging;

  explicit SwaBench(SwaSpec s = SwaSpec{})
      : spec(s), params(SwaParams(spec)),
        dims(Dots3NoteSlidingAttnDimsFrom(params)) {
    w = SwaWeights(dims, 0xB5026F5AA96619E9ULL);
    Rng r(0x27BB2EE687B0B0FDULL);
    hidden = r.fill(spec.tokens * dims.hidden_size, 1.0);
    positions.resize(static_cast<size_t>(spec.tokens));
    for (int64_t t = 0; t < spec.tokens; ++t) {
      positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    }
    paging.page_size = spec.page_size;
    // SHUFFLED on purpose: 8 tokens over pages of 3 is three logical pages,
    // mapped to physical 2, 0, 1. A contiguous table would make the paged read
    // and a flat read identical and prove nothing about the lookup.
    paging.block_table = {2, 0, 1};
  }
  std::vector<double> Run(SlidingAttnTrace* tr) const {
    return ForwardSlidingAttention(dims, w, hidden, positions, spec.tokens,
                                   paging, tr);
  }
  sref::Out Ref(const sref::Opts& o) const {
    return sref::Forward(dims, w, hidden, positions, spec.tokens, o);
  }
};

}  // namespace w4b
}  // namespace

// The W4b-1 surface, at file scope, so a case reads the way the W3/W4a ones do.
using vllm::dots3_note::ForwardSlidingAttention;
using vllm::dots3_note::kSwaMaskedScore;
using vllm::dots3_note::PaddedMlaCacheSpec;
using vllm::dots3_note::SlidingAttnDims;
using vllm::dots3_note::SlidingAttnTrace;
using vllm::dots3_note::SlidingAttnWeights;
using vllm::dots3_note::SlidingPaging;
using vllm::dots3_note::SlidingWindowChunk;
using vllm::dots3_note::SwaGatherResult;
using vllm::dots3_note::WritePaddedMlaCache;

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE(
    "dots3-note W4b-1: the SLIDING geometry comes off the RELEASED config, and "
    "it is NOT the full one") {
  const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(FixtureDir() + "/config.json"));
  const SlidingAttnDims s = w4b::Dots3NoteSlidingAttnDimsFrom(p);
  const FullAttnDims f = Dots3NoteFullAttnDimsFrom(p);

  // The released `dots-studio/dots3-note-prev` sliding geometry, §1.1.
  CHECK(s.hidden_size == 5120);
  CHECK(s.num_heads == 64);
  CHECK(s.qk_nope_head_dim == 192);
  CHECK(s.qk_rope_head_dim == 64);
  CHECK(s.v_head_dim == 128);
  CHECK(s.q_lora_rank == 1024);
  CHECK(s.kv_lora_rank == 1024);
  CHECK(s.sliding_window == 513);
  CHECK(s.rope_theta == doctest::Approx(5e4));
  CHECK_FALSE(s.rope_is_neox_style);  // model.py:408, passed LITERALLY
  CHECK(s.attention_gate_type == "headwise");

  // The KV-cache contract: ONE physical row for the whole model, which is the
  // SLIDING arm's own (model.py:283 -> :216). `get_supported_head_sizes`
  // returns exactly [1088] (attention.py:422-424).
  CHECK(s.latent_row() == 1088);
  CHECK(s.physical_latent_row == 1088);
  CHECK(p.physical_latent_row() == 1088);
  // ... and the FULL arm reads only the leading 576 of that row. This is the
  // whole reason `_logical_cache` exists.
  CHECK(f.kv_lora_rank + f.qk_rope_head_dim == 576);
  CHECK(s.physical_latent_row > f.kv_lora_rank + f.qk_rope_head_dim);

  // FIVE fields on which the two geometries DISAGREE. A port that ran one
  // struct for both would be silently wrong on 33 of the 46 layers.
  CHECK(s.num_heads != f.num_heads);                    // 64 vs 128
  CHECK(s.kv_lora_rank != f.kv_lora_rank);              // 1024 vs 512
  CHECK(s.qk_nope_head_dim != f.qk_nope_head_dim);      // 192 vs 128
  CHECK(s.rope_theta != doctest::Approx(f.rope_theta)); // 5e4 vs 8e7
  CHECK(s.sliding_window > 0);
  // ... and one on which they must AGREE: both ropes are GPT-J (§4 item 6,
  // corrected at W1 — #1804).
  CHECK(s.rope_is_neox_style == f.rope_is_neox_style);

  // The softmax scales therefore differ too: 256^-0.5 against 192^-0.5, with
  // NO YaRN mscale on either (both rebuild `rope_type="default"`).
  CHECK(s.softmax_scale() == doctest::Approx(std::pow(256.0, -0.5)).epsilon(1e-14));
  CHECK(f.softmax_scale() == doctest::Approx(std::pow(192.0, -0.5)).epsilon(1e-14));
  CHECK(s.softmax_scale() != doctest::Approx(f.softmax_scale()));

  // §4 trap 5 at the sliding ranks. Both swa ranks are 1024 on the released
  // model, so the two sliding scales are EQUAL — which is exactly why the
  // fixture below does NOT copy them.
  CHECK(s.q_lora_scale == doctest::Approx(std::sqrt(5120.0 / 1024.0)));
  CHECK(s.kv_lora_scale == doctest::Approx(std::sqrt(5120.0 / 1024.0)));
  CHECK(f.kv_lora_scale == doctest::Approx(std::sqrt(5120.0 / 512.0)));
  CHECK(s.kv_lora_scale != doctest::Approx(f.kv_lora_scale));
}

TEST_CASE("dots3-note W4b-1: the sliding geometry refuses what is not it") {
  w4b::SwaSpec s;
  {
    // A schedule with no sliding layer at all: the sliding geometry is then
    // not the one any layer runs, and returning a plausible struct would let a
    // caller compute a whole layer nothing in the model uses.
    nlohmann::json d = w4b::SwaConfigDoc(s);
    d["layer_types"] = nlohmann::json::array({"full_attention", "full_attention",
                                              "full_attention", "full_attention"});
    TempConfig cfg(d);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK_THROWS_WITH_AS((void)w4b::Dots3NoteSlidingAttnDimsFrom(p),
                         doctest::Contains("no sliding_attention layer"),
                         std::runtime_error);
  }
  {
    // `is_sparse == False` is what MAKES this the sliding arm (model.py:434).
    // A params object whose sliding arm claims an indexer is a contradiction,
    // and it is refused rather than quietly ignored.
    nlohmann::json d = w4b::SwaConfigDoc(s);
    TempConfig cfg(d);
    Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    REQUIRE_FALSE(p.swa.has_indexer);
    REQUIRE(p.full.has_indexer);
    p.swa.has_indexer = true;
    CHECK_THROWS_WITH_AS((void)w4b::Dots3NoteSlidingAttnDimsFrom(p),
                         doctest::Contains("is_sparse"), std::runtime_error);
  }
  {
    // The physical row must be at least the logical one — upstream asserts the
    // same at model.py:210.
    const w4b::SwaBench b(s);
    SlidingAttnDims bad = b.dims;
    bad.physical_latent_row = bad.latent_row() - 1;
    CHECK_THROWS_WITH_AS(bad.Validate(), doctest::Contains("physical MLA cache row"),
                         std::runtime_error);
  }
}

TEST_CASE(
    "dots3-note W4b-1: the sliding layer agrees with the independent "
    "reference") {
  const w4b::SwaBench b;
  SlidingAttnTrace tr;
  const std::vector<double> got = b.Run(&tr);
  const w4b::sref::Out want = b.Ref(w4b::sref::Opts{});

  // WHAT THE INSTRUMENT MEASURED, printed rather than assumed. If the window
  // never pruned, every window assertion in this file would pass on a plain
  // causal answer.
  MESSAGE("gather_len=" << tr.gather_len << " (window " << b.dims.sliding_window
                        << " + " << b.spec.tokens << " tokens, rounded to 8)");
  MESSAGE("queries that lost a key: " << want.queries_that_lost_a_key
                                      << ", keys dropped: "
                                      << want.keys_dropped_by_the_window);
  MESSAGE("max |score| = " << want.max_abs_score
                           << " (the reference's max-subtraction-free softmax "
                              "needs this inside exp's comfortable range)");
  CHECK(tr.gather_len == 16);
  CHECK(tr.rows_pruned_by_the_window == 5);
  CHECK(want.queries_that_lost_a_key == 5);
  CHECK(want.keys_dropped_by_the_window == 15);  // 1+2+3+4+5 for t=3..7
  CHECK(want.max_abs_score < 30.0);
  // Eight of the sixteen gathered slots are PAST the sequence and must come
  // back invalid — the round-up to 8 is real here, not the identity.
  int64_t valid_slots = 0;
  for (const char v : tr.gather_valid) valid_slots += v ? 1 : 0;
  CHECK(valid_slots == b.spec.tokens);
  CHECK(static_cast<int64_t>(tr.gather_valid.size()) == 16);

  // Every traced intermediate, so a mismatch attributes to ONE mechanism.
  const Diff d_qc = Compare(tr.q_c, want.q_c);
  const Diff d_kv = Compare(tr.kv_c_normed, want.kv_c_normed);
  const Diff d_kpe = Compare(tr.k_pe, want.k_pe);
  const Diff d_q = Compare(tr.q, want.q);
  const Diff d_attn = Compare(tr.attn_out, want.attn_out);
  const Diff d_gate = Compare(tr.gate, want.gate);
  const Diff d_gated = Compare(tr.gated, want.gated);
  const Diff d_out = Compare(got, want.out);
  MESSAGE("relative agreement: q_c " << d_qc.max_rel << ", kv_c_normed "
                                     << d_kv.max_rel << ", k_pe " << d_kpe.max_rel
                                     << ", q " << d_q.max_rel << ", attn_out "
                                     << d_attn.max_rel << ", gate "
                                     << d_gate.max_rel << ", gated "
                                     << d_gated.max_rel << ", out "
                                     << d_out.max_rel);
  CHECK(d_qc.max_rel < kAgreeRel);
  CHECK(d_kv.max_rel < kAgreeRel);
  CHECK(d_kpe.max_rel < kAgreeRel);
  CHECK(d_q.max_rel < kAgreeRel);
  CHECK(d_attn.max_rel < kAgreeRel);
  CHECK(d_gate.max_rel < kAgreeRel);
  CHECK(d_gated.max_rel < kAgreeRel);
  CHECK(d_out.max_rel < kAgreeRel);
  // The ABSORBED arm and the MATERIALIZED arm agree, which is the statement
  // worth making: `q_nope @ W_UK` then a latent dot product is the same
  // function as up-projecting K and dotting per head, and the sliding path
  // takes the first route while the reference takes the second.
  CHECK(d_attn.max_mag > 0.05);
}

TEST_CASE(
    "dots3-note W4b-1: the WINDOW is the mechanism — dense causal is a "
    "different answer, and a wide window is the same one") {
  const w4b::SwaBench b;
  const std::vector<double> got = b.Run(nullptr);

  // (a) A port that never noticed `sliding_window_size` computes plain causal
  //     attention. That is upstream's answer only while the window covers the
  //     whole sequence, and this fixture is deliberately past that point.
  w4b::sref::Opts dense;
  dense.windowed = false;
  const w4b::sref::Out unwindowed = b.Ref(dense);
  const Diff d_dense = Compare(got, unwindowed.out);
  MESSAGE("dense causal vs windowed: max|diff| " << d_dense.max_abs << " over a "
                                                 << "scale of " << d_dense.max_mag
                                                 << " = " << d_dense.max_rel
                                                 << " relative");
  CHECK(unwindowed.queries_that_lost_a_key == 0);
  CHECK(d_dense.max_rel > 1e-2);

  // (b) The converse, which is what says the window is a WINDOW and not just a
  //     perturbation: widen it past the sequence and the two answers converge
  //     to the SAME numbers the dense reference produces. A port with an
  //     off-by-one window bound fails this and (a) cannot see it.
  w4b::SwaSpec wide = b.spec;
  wide.window = wide.tokens;  // W == T: every causal key is inside the window
  const w4b::SwaBench bw(wide);
  const std::vector<double> got_wide = bw.Run(nullptr);
  const w4b::sref::Out ref_dense_wide = bw.Ref(dense);
  const Diff d_wide = Compare(got_wide, ref_dense_wide.out);
  MESSAGE("window == tokens vs dense causal: " << d_wide.max_rel << " relative");
  CHECK(ref_dense_wide.queries_that_lost_a_key == 0);
  CHECK(d_wide.max_rel < kAgreeRel);

  // (c) The off-by-one itself. At W == T - 1 exactly ONE query (the last) loses
  //     exactly ONE key, so the two windows differ by the smallest possible
  //     amount and the gate still sees it. This is the assertion that pins
  //     `kv_pos >= query_pos - W + 1` rather than `> query_pos - W`.
  w4b::SwaSpec off = b.spec;
  off.window = off.tokens - 1;
  const w4b::SwaBench bo(off);
  w4b::sref::Opts win;
  const w4b::sref::Out ref_off = bo.Ref(win);
  CHECK(ref_off.queries_that_lost_a_key == 1);
  CHECK(ref_off.keys_dropped_by_the_window == 1);
  const Diff d_off = Compare(bo.Run(nullptr), ref_off.out);
  CHECK(d_off.max_rel < kAgreeRel);
  const Diff d_off_vs_wide = Compare(bo.Run(nullptr), got_wide);
  MESSAGE("ONE dropped key moves the layer by " << d_off_vs_wide.max_rel
                                                << " relative");
  CHECK(d_off_vs_wide.max_rel > 1e-3);
}

TEST_CASE(
    "dots3-note W4b-1: `gather_len` is ONE formula, so the workspace and the "
    "gather cannot drift") {
  // attention.py:484 in `_forward_swa_mqa` and :321 in
  // `_reserve_attn_logits_workspace` are the SAME expression on purpose: the
  // decode gathers into the workspace the builder reserved. Ported as one
  // function so a future edit cannot move only one of them.
  CHECK(w4b::SwaGatherLen(513, 1) == 520);   // (513 + 0 + 7)/8*8
  CHECK(w4b::SwaGatherLen(513, 8) == 520);   // 520 exactly — no round-up needed
  CHECK(w4b::SwaGatherLen(513, 9) == 528);   // 521 -> 528
  CHECK(w4b::SwaGatherLen(3, 8) == 16);      // 10 -> 16, the bench's own case
  CHECK(w4b::SwaGatherLen(8, 1) == 8);       // already a multiple of 8
  CHECK(w4b::SwaGatherLen(1, 1) == 8);       // rounds UP, never to 0
  // It never returns less than the window, and never less than the query span.
  for (int64_t w = 1; w <= 40; ++w) {
    for (int64_t q = 1; q <= 12; ++q) {
      const int64_t g = w4b::SwaGatherLen(w, q);
      CHECK(g >= w + q - 1);
      CHECK(g % 8 == 0);
      CHECK(g - (w + q - 1) < 8);  // the SMALLEST such multiple
    }
  }
  CHECK_THROWS_AS((void)w4b::SwaGatherLen(0, 4), std::runtime_error);
  CHECK_THROWS_AS((void)w4b::SwaGatherLen(4, 0), std::runtime_error);
}

TEST_CASE(
    "dots3-note W4b-1: the KV gather reads PAGED rows at the PHYSICAL stride") {
  // A cache whose every element identifies its own (slot, lane), so a wrong
  // page, a wrong offset or a wrong stride is READABLE rather than merely
  // different.
  const int64_t num_blocks = 3, page_size = 3, physical = 10, logical = 8;
  const int64_t slots = num_blocks * page_size;
  std::vector<double> cache(static_cast<size_t>(slots * physical));
  for (int64_t s = 0; s < slots; ++s) {
    for (int64_t c = 0; c < physical; ++c) {
      cache[static_cast<size_t>(s * physical + c)] =
          static_cast<double>(s * 100 + c);
    }
  }
  const std::vector<int32_t> bt{2, 0, 1};  // logical page -> physical page
  const std::vector<int32_t> seq_lens{8};
  const int64_t gather_len = w4b::SwaGatherLen(3, 8);
  REQUIRE(gather_len == 16);

  const SwaGatherResult g =
      w4b::GatherSwaKv(cache, bt, seq_lens, 1, 3, num_blocks, page_size,
                       physical, logical, gather_len);
  // Hand-derived, not recomputed with the production formula: token 0 sits in
  // logical page 0 -> physical page 2, offset 0 -> slot 6; token 3 in logical
  // page 1 -> physical 0, offset 0 -> slot 0; token 7 in logical page 2 ->
  // physical 1, offset 1 -> slot 4.
  CHECK(g.kv[0] == doctest::Approx(600.0));
  CHECK(g.kv[static_cast<size_t>(3 * logical)] == doctest::Approx(0.0));
  CHECK(g.kv[static_cast<size_t>(7 * logical)] == doctest::Approx(400.0));
  CHECK(g.kv[static_cast<size_t>(7 * logical + 1)] == doctest::Approx(401.0));
  // The PADDING: only the leading `logical` lanes are read, and the row base
  // still advances by `physical`. Lane 7 of token 7 is 407, and 408/409 are
  // never read.
  CHECK(g.kv[static_cast<size_t>(7 * logical + 7)] == doctest::Approx(407.0));
  // Past the sequence: invalid and ZERO.
  for (int64_t slot = 8; slot < gather_len; ++slot) {
    CHECK(g.valid[static_cast<size_t>(slot)] == 0);
    CHECK(g.kv[static_cast<size_t>(slot * logical)] == doctest::Approx(0.0));
  }
  for (int64_t slot = 0; slot < 8; ++slot) CHECK(g.valid[static_cast<size_t>(slot)] == 1);

  // PAD INVARIANCE. The same logical rows in an UNPADDED cache gather to the
  // same bytes. This is the property `_logical_cache` guarantees, stated where
  // the read happens; a reader that kept the LOGICAL stride would find real
  // cached numbers from the wrong tokens and no shape would change.
  std::vector<double> unpadded(static_cast<size_t>(slots * logical));
  for (int64_t s = 0; s < slots; ++s) {
    for (int64_t c = 0; c < logical; ++c) {
      unpadded[static_cast<size_t>(s * logical + c)] =
          cache[static_cast<size_t>(s * physical + c)];
    }
  }
  const SwaGatherResult gu =
      w4b::GatherSwaKv(unpadded, bt, seq_lens, 1, 3, num_blocks, page_size,
                       logical, logical, gather_len);
  CHECK(Compare(g.kv, gu.kv).max_abs == 0.0);
  CHECK(g.valid == gu.valid);

  // A cache read at the LOGICAL stride when the rows are PHYSICAL is the
  // defect, and it is not the same answer — built by hand here so the probe
  // shares no arithmetic with the production gather it is contradicting.
  std::vector<double> wrong(g.kv.size(), 0.0);
  for (int64_t slot = 0; slot < 8; ++slot) {
    const int64_t page = bt[static_cast<size_t>(slot / page_size)];
    const int64_t off = slot % page_size;
    const int64_t src = (page * page_size + off) * logical;  // WRONG stride
    for (int64_t c = 0; c < logical; ++c) {
      wrong[static_cast<size_t>(slot * logical + c)] =
          cache[static_cast<size_t>(src + c)];
    }
  }
  const Diff d_stride = Compare(g.kv, wrong);
  MESSAGE("logical-stride gather vs physical-stride gather: max|diff| "
          << d_stride.max_abs);
  CHECK(d_stride.max_abs > 0.0);

  // An UNMAPPED page (a negative block-table entry, attention.py:86) yields
  // invalid, zero slots rather than a read of page -1.
  const std::vector<int32_t> bt_hole{2, -1, 1};
  const SwaGatherResult gh =
      w4b::GatherSwaKv(cache, bt_hole, seq_lens, 1, 3, num_blocks, page_size,
                       physical, logical, gather_len);
  for (int64_t slot = 3; slot < 6; ++slot) {
    CHECK(gh.valid[static_cast<size_t>(slot)] == 0);
    CHECK(gh.kv[static_cast<size_t>(slot * logical)] == doctest::Approx(0.0));
  }
  CHECK(gh.valid[2] == 1);
  CHECK(gh.valid[6] == 1);

  CHECK_THROWS_WITH_AS(
      (void)w4b::GatherSwaKv(cache, bt, seq_lens, 1, 3, num_blocks, page_size,
                             logical, physical, gather_len),
      doctest::Contains("must fit inside the physical row"), std::runtime_error);

  // A DECODE-shaped gather, which is the ONLY shape in which `gather_start` is
  // not 0 — and it exists because a mutation that pinned `gather_start = 0`
  // came back GREEN against everything above. In a PREFILL the gather covers
  // the whole sequence (`gather_len >= window + T - 1 >= T`), so the maximum is
  // the identity; a decode carries ONE query at the tail of a long context and
  // the gather has to skip the head of it. Spec §4.7 records the green and this
  // is the fixture that answers it rather than the note that excuses it.
  {
    const int64_t dec_blocks = 8, dec_pages = 7;
    std::vector<double> big(static_cast<size_t>(dec_blocks * page_size * physical));
    for (int64_t sl = 0; sl < dec_blocks * page_size; ++sl) {
      for (int64_t c = 0; c < physical; ++c) {
        big[static_cast<size_t>(sl * physical + c)] =
            static_cast<double>(sl * 100 + c);
      }
    }
    // Seven logical pages of 3, SHUFFLED, for a 20-token context.
    const std::vector<int32_t> dbt{5, 1, 7, 0, 6, 2, 4};
    const std::vector<int32_t> dseq{20};
    const int64_t dgather = w4b::SwaGatherLen(3, 1);
    REQUIRE(dgather == 8);
    REQUIRE(dgather < dseq[0]);  // the whole point: the gather CANNOT cover it
    const SwaGatherResult dg =
        w4b::GatherSwaKv(big, dbt, dseq, 1, dec_pages, dec_blocks, page_size,
                         physical, logical, dgather);
    // gather_start = max(20 - 8, 0) = 12, so slot g holds token 12 + g.
    // HAND-DERIVED: token 12 is logical page 4 -> physical 6, offset 0 ->
    // slot 18; token 19 is logical page 6 -> physical 4, offset 1 -> slot 13.
    CHECK(dg.kv[0] == doctest::Approx(1800.0));
    CHECK(dg.kv[static_cast<size_t>(7 * logical)] == doctest::Approx(1300.0));
    for (int64_t slot = 0; slot < dgather; ++slot) {
      CHECK(dg.valid[static_cast<size_t>(slot)] == 1);
      const int64_t token = 12 + slot;
      const int64_t page = dbt[static_cast<size_t>(token / page_size)];
      const int64_t off = token % page_size;
      CHECK(dg.kv[static_cast<size_t>(slot * logical)] ==
            doctest::Approx(static_cast<double>((page * page_size + off) * 100)));
    }
    // Every gathered token is inside [seq_len - gather_len, seq_len): the head
    // of the context is skipped, which is what a 513-wide window over a 524288
    // position model needs and what `gather_start = 0` would silently undo.
    CHECK(dg.kv[0] != doctest::Approx(0.0));
  }
}

TEST_CASE(
    "dots3-note W4b-1: the score mask is a POSITION predicate, not a "
    "token-order one") {
  // A DECODE-shaped batch: two requests, two queries each, whose queries are
  // the TAIL of much longer sequences. Nothing about the token's index in the
  // batch says which keys it may see — only `seq_len - query_len + q` does
  // (attention.py:142), and this is the case that pins it.
  const int64_t n_reqs = 2, heads = 2, query_len = 2, window = 3;
  const int64_t gather_len = w4b::SwaGatherLen(window, query_len);
  REQUIRE(gather_len == 8);
  const std::vector<int32_t> seq_lens{10, 4};
  std::vector<char> valid(static_cast<size_t>(n_reqs * gather_len), 1);
  // req0: seq_len 10 > gather_len 8, so gather_start is 2 and all 8 slots are
  // real. req1: seq_len 4 < 8, so only slots 0..3 are real.
  for (int64_t slot = 4; slot < gather_len; ++slot) {
    valid[static_cast<size_t>(1 * gather_len + slot)] = 0;
  }
  std::vector<double> scores(
      static_cast<size_t>(n_reqs * heads * query_len * gather_len), 1.0);
  w4b::ApplySwaScoreMask(scores, seq_lens, valid, n_reqs, heads, query_len,
                         gather_len, window);

  // HAND-DERIVED, not recomputed with the production predicate.
  //   req0 gather_start = max(10-8, 0) = 2, kv positions 2..9.
  //        query 0 sits at position 8 -> keeps kv 6,7,8   -> slots 4,5,6
  //        query 1 sits at position 9 -> keeps kv 7,8,9   -> slots 5,6,7
  //   req1 gather_start = 0, kv positions 0..7 but only 0..3 are real.
  //        query 0 sits at position 2 -> keeps kv 0,1,2   -> slots 0,1,2
  //        query 1 sits at position 3 -> keeps kv 1,2,3   -> slots 1,2,3
  const std::vector<std::vector<std::vector<int64_t>>> keep{
      {{4, 5, 6}, {5, 6, 7}},
      {{0, 1, 2}, {1, 2, 3}},
  };
  for (int64_t req = 0; req < n_reqs; ++req) {
    for (int64_t q = 0; q < query_len; ++q) {
      for (int64_t h = 0; h < heads; ++h) {
        for (int64_t slot = 0; slot < gather_len; ++slot) {
          const std::vector<int64_t>& k =
              keep[static_cast<size_t>(req)][static_cast<size_t>(q)];
          const bool want_kept =
              std::find(k.begin(), k.end(), slot) != k.end();
          const double v = scores[static_cast<size_t>(
              ((req * heads + h) * query_len + q) * gather_len + slot)];
          if (want_kept) {
            CHECK(v == doctest::Approx(1.0));
          } else {
            CHECK(v == doctest::Approx(kSwaMaskedScore));
          }
        }
      }
    }
  }

  // The masked value is `-FLT_MAX` and not `-inf`, which is upstream's literal
  // (attention.py:161). It matters: `exp(-FLT_MAX - max)` underflows to exactly
  // 0, while an all-`-inf` row softmaxes to NaN. The number is pinned so a
  // "tidy-up" to -inf is a red test rather than a silent change of failure mode.
  CHECK(kSwaMaskedScore == -3.4028234663852886e38);
  CHECK(std::isfinite(kSwaMaskedScore));
  CHECK(std::exp(kSwaMaskedScore - 1.0) == 0.0);

  // `valid` beats the window: a slot INSIDE the window whose page was never
  // mapped is still masked (attention.py:143-147, `page_valid`).
  std::vector<char> holed = valid;
  holed[static_cast<size_t>(5)] = 0;  // req0 slot 5, inside both queries' windows
  std::vector<double> scores2(scores.size(), 1.0);
  w4b::ApplySwaScoreMask(scores2, seq_lens, holed, n_reqs, heads, query_len,
                         gather_len, window);
  CHECK(scores2[static_cast<size_t>(((0 * heads + 0) * query_len + 0) * gather_len + 5)] ==
        doctest::Approx(kSwaMaskedScore));
  CHECK(scores2[static_cast<size_t>(((0 * heads + 0) * query_len + 0) * gather_len + 4)] ==
        doctest::Approx(1.0));

  CHECK_THROWS_WITH_AS(
      w4b::ApplySwaScoreMask(scores, seq_lens, valid, n_reqs, heads, query_len,
                             gather_len, 0),
      doctest::Contains("WINDOW_SIZE must be positive"), std::runtime_error);
}

TEST_CASE(
    "dots3-note W4b-1: `_logical_cache` narrows a PADDED row, and the logical "
    "stride does not") {
  PaddedMlaCacheSpec spec;
  spec.num_blocks = 3;
  spec.page_size = 3;
  spec.physical_row = 10;  // the SLIDING arm's row, shared by the whole model
  spec.logical_row = 8;    // what a FULL layer reads out of it
  std::vector<double> cache(static_cast<size_t>(spec.slots() * spec.physical_row), -7.0);

  const int64_t T = 5, kv_lora = 4, rope = 4;
  Rng r(0x2545F4914F6CDD1DULL);
  const std::vector<double> kv_c = r.fill(T * kv_lora, 1.0);
  const std::vector<double> k_pe = r.fill(T * rope, 1.0);
  const std::vector<int64_t> slot_mapping{6, 7, 8, 0, 1};
  WritePaddedMlaCache(cache, spec, kv_c, k_pe, kv_lora, rope, slot_mapping, T);

  const std::vector<double> narrowed = w4b::NarrowLogicalCacheRows(cache, spec);
  REQUIRE(static_cast<int64_t>(narrowed.size()) == spec.slots() * spec.logical_row);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = slot_mapping[static_cast<size_t>(t)];
    for (int64_t c = 0; c < kv_lora; ++c) {
      CHECK(narrowed[static_cast<size_t>(slot * spec.logical_row + c)] ==
            doctest::Approx(kv_c[static_cast<size_t>(t * kv_lora + c)]));
    }
    for (int64_t c = 0; c < rope; ++c) {
      CHECK(narrowed[static_cast<size_t>(slot * spec.logical_row + kv_lora + c)] ==
            doctest::Approx(k_pe[static_cast<size_t>(t * rope + c)]));
    }
  }

  // The TAIL of every physical row is untouched by a logical-width write. That
  // is what lets one block serve two logical widths: the sliding layers own
  // lanes [8, 10) of THEIR rows, and a full layer writing 8 lanes into the same
  // block shape must not tread on them.
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = slot_mapping[static_cast<size_t>(t)];
    for (int64_t c = spec.logical_row; c < spec.physical_row; ++c) {
      CHECK(cache[static_cast<size_t>(slot * spec.physical_row + c)] ==
            doctest::Approx(-7.0));
    }
  }

  // THE DEFECT, executed. A reader that keeps the LOGICAL stride over a
  // PHYSICAL cache finds finite, previously-cached numbers from the wrong
  // tokens — no shape changes, nothing throws, and under spec §6.4 there is no
  // oracle downstream that would notice.
  std::vector<double> wrong(static_cast<size_t>(spec.slots() * spec.logical_row));
  for (int64_t s = 0; s < spec.slots(); ++s) {
    for (int64_t c = 0; c < spec.logical_row; ++c) {
      wrong[static_cast<size_t>(s * spec.logical_row + c)] =
          cache[static_cast<size_t>(s * spec.logical_row + c)];
    }
  }
  const Diff d = Compare(narrowed, wrong);
  MESSAGE("logical-stride read vs `_logical_cache`: max|diff| " << d.max_abs);
  CHECK(d.max_abs > 0.0);

  // At physical == logical the narrowing is the IDENTITY, which is the sliding
  // arm's own case and must stay free.
  PaddedMlaCacheSpec flat = spec;
  flat.physical_row = flat.logical_row;
  std::vector<double> flat_cache(static_cast<size_t>(flat.slots() * flat.physical_row), 0.0);
  WritePaddedMlaCache(flat_cache, flat, kv_c, k_pe, kv_lora, rope, slot_mapping, T);
  CHECK(Compare(w4b::NarrowLogicalCacheRows(flat_cache, flat), flat_cache).max_abs == 0.0);

  PaddedMlaCacheSpec bad = spec;
  bad.physical_row = bad.logical_row - 1;
  CHECK_THROWS_WITH_AS(bad.Validate(), doctest::Contains("narrower than the logical row"),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(
      WritePaddedMlaCache(cache, spec, kv_c, k_pe, kv_lora, rope,
                               std::vector<int64_t>{6, 7, 8, 0, 99}, T),
      doctest::Contains("outside the"), std::runtime_error);
}

TEST_CASE(
    "dots3-note W4b-1: the windowed metadata caps every gather at the window "
    "and packs chunks that fit") {
  // Three requests with very different contexts, so `min(seq_len, query_len +
  // window - 1)` is the binding constraint on some and `seq_len` on others.
  const std::vector<int32_t> seq_lens{10, 4, 20};
  const std::vector<int32_t> qsl{0, 2, 4, 5};  // query lens 2, 2, 1
  const int64_t window = 3;

  const std::vector<SlidingWindowChunk> chunks =
      w4b::BuildSlidingWindowMetadata(seq_lens, qsl, window, /*workspace=*/8);
  // kv_lens = min(10, 2+2)=4, min(4, 2+2)=4, min(20, 1+2)=3 — HAND-DERIVED.
  // Packing at a workspace of 8: reqs 0 and 1 fill it exactly, req 2 starts a
  // second chunk.
  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0].req_start == 0);
  CHECK(chunks[0].req_end == 2);
  CHECK(chunks[0].query_start == 0);
  CHECK(chunks[0].query_end == 4);
  CHECK(chunks[0].num_kv_tokens == 8);
  CHECK(chunks[0].cu_seq_lens_q == std::vector<int32_t>{0, 2, 4});
  CHECK(chunks[0].cu_seq_lens_k == std::vector<int32_t>{0, 4, 8});
  // `starts = seq_len - kv_len`: request 0 begins its gather at position 6,
  // request 1 at 0. THIS is the line that makes a 524288-position model
  // affordable on a 513-wide layer.
  CHECK(chunks[0].starts == std::vector<int32_t>{6, 0});
  CHECK(chunks[0].token_to_seq == std::vector<int32_t>{0, 0, 0, 0, 1, 1, 1, 1});
  CHECK(chunks[0].max_seq_len_q == 2);
  CHECK(chunks[0].max_seq_len_k == 4);
  CHECK(chunks[1].req_start == 2);
  CHECK(chunks[1].req_end == 3);
  CHECK(chunks[1].query_start == 4);
  CHECK(chunks[1].query_end == 5);
  CHECK(chunks[1].num_kv_tokens == 3);
  CHECK(chunks[1].cu_seq_lens_k == std::vector<int32_t>{0, 3});
  CHECK(chunks[1].starts == std::vector<int32_t>{17});

  // The CAP is the mechanism. Request 2 has 20 cached positions and gathers 3.
  // A port that gathered `seq_len` would produce the SAME answer once the mask
  // ran — at almost 7x the workspace — so no value assertion anywhere can catch
  // it and this one has to.
  int64_t total = 0;
  for (const SlidingWindowChunk& c : chunks) total += c.num_kv_tokens;
  CHECK(total == 11);
  int64_t total_seq = 0;
  for (const int32_t s : seq_lens) total_seq += s;
  CHECK(total_seq == 34);
  MESSAGE("the window cap gathers " << total << " KV tokens where the full "
                                    << "contexts are " << total_seq);

  // A wide-enough workspace packs everything into ONE chunk; a narrow one
  // splits per request.
  CHECK(w4b::BuildSlidingWindowMetadata(seq_lens, qsl, window, 64).size() == 1);
  CHECK(w4b::BuildSlidingWindowMetadata(seq_lens, qsl, window, 4).size() == 3);

  // A request that cannot fit ALONE is a hard error, not a truncation
  // (attention.py:218-221). The message is upstream's own text.
  CHECK_THROWS_WITH_AS(
      (void)w4b::BuildSlidingWindowMetadata(seq_lens, qsl, window, 2),
      doctest::Contains("SWA prefill window exceeds the MLA workspace"),
      std::runtime_error);
  // The queries are the TAIL of the sequence, so seq_len < query_len is not
  // representable and is refused rather than producing a negative start.
  CHECK_THROWS_WITH_AS(
      (void)w4b::BuildSlidingWindowMetadata(std::vector<int32_t>{1}, std::vector<int32_t>{0, 3},
                                            window, 8),
      doctest::Contains("below its query_len"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      (void)w4b::BuildSlidingWindowMetadata(seq_lens, std::vector<int32_t>{2, 4, 6, 7}, window, 8),
      doctest::Contains("must start at 0"), std::runtime_error);
}

TEST_CASE(
    "dots3-note W4b-1: the four deltas `_forward_note_mla` shares are each "
    "observable on the SLIDING arm too") {
  // `_forward_note_mla` is ONE function upstream, but this tree has two
  // callers, so a delta can be present on the full arm and missing on the
  // sliding one with no shape change and no oracle to notice. Each is
  // neutralised in the REFERENCE and the implementation is measured drifting
  // AWAY — the same construction spec §4.6 uses, and it is a statement about
  // the MECHANISM rather than about the two files agreeing.
  const w4b::SwaBench b;
  const std::vector<double> got = b.Run(nullptr);
  const Diff base = Compare(got, b.Ref(w4b::sref::Opts{}).out);
  MESSAGE("agreement with every delta in place: " << base.max_rel);
  REQUIRE(base.max_rel < kAgreeRel);

  struct Arm {
    std::string what;
    w4b::sref::Opts o;
  };
  std::vector<Arm> arms;
  {
    Arm a{"the q LoRA rescale dropped", {}};
    a.o.apply_q_lora_rescale = false;
    arms.push_back(a);
  }
  {
    Arm a{"the kv LoRA rescale dropped", {}};
    a.o.apply_kv_lora_rescale = false;
    arms.push_back(a);
  }
  {
    Arm a{"`k_rope_only_layernorm` dropped", {}};
    a.o.k_rope_only_norm = false;
    arms.push_back(a);
  }
  {
    Arm a{"the headwise gate made lane-wise", {}};
    a.o.headwise_gate = false;
    arms.push_back(a);
  }
  {
    // The one delta that is SLIDING-ONLY: `swa_rope_theta` (model.py:406). A
    // port that inherited the model-level theta rotates 33 of the 46 layers at
    // 8e7 instead of 5e4, which is three orders of magnitude and completely
    // silent.
    Arm a{"the model-level rope theta instead of `swa_rope_theta`", {}};
    a.o.rope_theta_override = b.spec.rope_theta;
    arms.push_back(a);
  }
  for (const Arm& a : arms) {
    const Diff d = Compare(got, b.Ref(a.o).out);
    MESSAGE("with " << a.what << ": " << d.max_rel << " relative, i.e. "
                    << (d.max_rel / kAgreeRel) << "x the agreement bound");
    CHECK(d.max_rel > 1e-2);
  }
  // The two LoRA scales are neutralised SEPARATELY and are DIFFERENT numbers
  // on this fixture (sqrt(16/3) vs sqrt(16/6)), so an arm that dropped both at
  // once could not tell a port carrying both from one carrying only the q.
  CHECK(b.dims.q_lora_scale == doctest::Approx(std::sqrt(16.0 / 3.0)));
  CHECK(b.dims.kv_lora_scale == doctest::Approx(std::sqrt(16.0 / 6.0)));
  CHECK(b.dims.q_lora_scale != doctest::Approx(b.dims.kv_lora_scale));
  // ... and the FIVE fixture separations the header argues for, pinned so a
  // later edit cannot quietly make a mechanism unobservable again.
  //
  // The head-count pin was MISSING until the fresh review, and its absence was
  // measured rather than argued: setting `swa_heads` equal to `full_heads` and
  // changing nothing else left the whole gate green at 30 cases / 2417
  // assertions. Four of these five were written after a green mutation exposed
  // the geometry that hid it; this one is here because a reviewer looked for
  // the fifth and found no assertion behind it.
  CHECK(b.dims.qk_head_dim() != b.dims.latent_row());
  CHECK(b.dims.num_heads != b.spec.full_heads);
  CHECK(b.dims.physical_latent_row > b.spec.full_kv_lora + b.spec.qk_rope);
  CHECK(b.spec.window < b.spec.tokens);
  CHECK(b.spec.swa_rope_theta != doctest::Approx(b.spec.rope_theta));
}

TEST_CASE(
    "dots3-note W4b-1: the sliding layer refuses a weight of the wrong size BY "
    "NAME") {
  const w4b::SwaBench b;
  {
    SlidingAttnWeights bad = b.w;
    bad.q_b_proj.resize(bad.q_b_proj.size() - 1);
    CHECK_THROWS_WITH_AS(
        (void)ForwardSlidingAttention(b.dims, bad, b.hidden, b.positions,
                                      b.spec.tokens, b.paging, nullptr),
        doctest::Contains("q_b_proj.weight"), std::runtime_error);
  }
  {
    // The FULL arm's `q_b_proj` on a sliding layer: [heads*(128+64), q_lora] vs
    // [heads*(192+64), q_lora]. Two real shapes from the same checkpoint, and
    // the released index is what says they differ (spec §4.4 fact 2).
    SlidingAttnWeights bad = b.w;
    bad.q_b_proj.assign(
        static_cast<size_t>(b.dims.num_heads *
                            (b.spec.full_qk_nope + b.spec.qk_rope) *
                            b.dims.q_lora_rank),
        0.1);
    CHECK_THROWS_WITH_AS(
        (void)ForwardSlidingAttention(b.dims, bad, b.hidden, b.positions,
                                      b.spec.tokens, b.paging, nullptr),
        doctest::Contains("q_b_proj.weight"), std::runtime_error);
  }
  {
    SlidingAttnWeights bad = b.w;
    bad.g_proj.resize(static_cast<size_t>(b.dims.hidden_size));  // one head
    CHECK_THROWS_WITH_AS(
        (void)ForwardSlidingAttention(b.dims, bad, b.hidden, b.positions,
                                      b.spec.tokens, b.paging, nullptr),
        doctest::Contains("g_proj.weight"), std::runtime_error);
  }
  {
    // A block table too short for the request. Silently attending to fewer
    // tokens is the failure this prevents.
    SlidingPaging short_paging = b.paging;
    short_paging.block_table = {2, 0};
    CHECK_THROWS_WITH_AS(
        (void)ForwardSlidingAttention(b.dims, b.w, b.hidden, b.positions,
                                      b.spec.tokens, short_paging, nullptr),
        doctest::Contains("block table"), std::runtime_error);
  }
}

TEST_CASE(
    "dots3-note W4b-1/W4b-2/W5: this bench's config is refused by NOTHING any "
    "more, and each branch names the brick that lifted it") {
  // W4b-1 wrote HOST code and lifted nothing. This case recorded the boundary
  // it left: the sliding layer and the PADDED physical row were both refused
  // by `Dots3NoteDeviceRefusal`. **W4b-2 lifted both**, and **W5 lifted the MoE
  // layer while W5c lifted the nextn tail (#2176)**, so the assertions below
  // are the ACCEPTANCES that replaced them — kept in place rather than deleted,
  // so a reader following W4b-1's evidence lands on the answer.
  const w4b::SwaBench b;
  // The bench's own config is the released schedule's shape: MoE from layer 1
  // and sliding from layer 2, plus the §4-trap-3 nextn default of 1. Every one
  // of those was a refusal at some brick and none is one now.
  const std::string why = vllm::Dots3NoteDeviceRefusal(b.params);
  MESSAGE("device refusal, bench config: '" << why << "' (empty)");
  CHECK(why.empty());
  REQUIRE(b.params.is_moe_layer(1));
  REQUIRE(b.params.num_nextn_predict_layers == 1);

  // The same config with every layer DENSE is also empty, which says the MoE
  // acceptance above is not standing in for something else.
  nlohmann::json d = w4b::SwaConfigDoc(b.spec);
  d["first_k_dense_replace"] = 4;  // every layer dense
  TempConfig cfg_nextn(d);
  const Dots3NoteParams p_nextn = ParseDots3NoteParams(LoadHfConfig(cfg_nextn.path()));
  const std::string why_nextn = vllm::Dots3NoteDeviceRefusal(p_nextn);
  MESSAGE("with no MoE layer, the refusal is: '" << why_nextn << "' (empty)");
  CHECK(why_nextn.empty());
  d["num_nextn_predict_layers"] = 0;
  TempConfig cfg_swa(d);
  const Dots3NoteParams p_swa = ParseDots3NoteParams(LoadHfConfig(cfg_swa.path()));
  const std::string why_swa = vllm::Dots3NoteDeviceRefusal(p_swa);
  MESSAGE("with no MoE and no nextn tail, the refusal is now: '" << why_swa
                                                                 << "' (empty)");
  CHECK(why_swa.empty());
  CHECK(p_swa.physical_latent_row() == 10);
  CHECK(p_swa.full.latent_row() == 8);
  CHECK(p_swa.swa.latent_row() == 10);
  // The padded row is real on this config: two spare lanes on every physical
  // row a full layer writes into.
  CHECK(p_swa.physical_latent_row() - p_swa.full.latent_row() == 2);
}

// ═════════════════════════════════════════════════════════════════════════════
// W4b-2 — the SLIDING arm ON THE DECODE PATH, over a PADDED KV cache.
//
// ─── WHAT THIS ESTABLISHES, AND WHAT IT CANNOT ───────────────────────────────
// A MIXED config — layers `{full, sliding, full}`, every one with a
// dense MLP — is loaded through the REAL registry and run through
// `ModelRegistry::Forward` TWICE against one KV cache pool: a PREFILL of six
// tokens, then a DECODE of the seventh. Its logits are compared against a
// whole-model double reference that dispatches per layer kind — W3's
// `ref::Forward` for the full layers, W4b-1's `sref::Forward` for the sliding
// ones — with no vt op, no cache and no window arithmetic anywhere inside it.
//
// It CANNOT say the model matches vLLM. No oracle for `dots3_note` runs on any
// host this project owns (spec §6.2, §6.4 option B), so this is a consistency
// gate between two independent implementations of the same formula, one of
// them the reference W3/W4b-1 transcribed straight from the python.
//
// ─── WHAT THE PADDED ROW MEANS HERE, MEASURED ────────────────────────────────
// The physical cache row is 10 (`swa_kv_lora_rank` 6 + `swa_qk_rope_head_dim`
// 4); a FULL layer reads 6 (`kv_lora_rank` 2 + 4). Those four lanes are not
// decoration: the case reads the RAW cache bytes after the forward and asserts
// that every full layer's written slots still carry ZERO in lanes [6, 10),
// which is exactly upstream's `_logical_cache` narrowing
// (`vllm/models/dots3_note/nvidia/attention.py:700-702` @ `bc2d63e650`) and is
// FALSE for any port that writes at the physical stride.
//
// ─── WHAT IS NOT REACHED ─────────────────────────────────────────────────────
// The MoE layers (W5), the vision and audio towers (W6/W7), the nextn tail
// (W10), the DSA lightning indexer's SELECTION and a windowed prefill that
// also carries chunked CONTEXT (both W4b-3). Each is refused BY NAME and the
// refusals have their own case below.
// ═════════════════════════════════════════════════════════════════════════════
namespace {
namespace w4b2 {

using vllm::Dots3NoteDeviceRefusal;
using vllm::Dots3NoteFullAttnMlaDims;
using vllm::Dots3NoteLayerKind;
using vllm::Dots3NoteSlidingAttnMlaDims;
using vllm::PagedKvCache;
using vllm::dots3_note::Dots3NoteSlidingAttnDimsFrom;
using w4a::Bf16All;
using w4a::StOut;

// ─────────────────────────────────────────────────────────────────────────────
// The MIXED bench. Every number is chosen so a mechanism this brick added is
// OBSERVABLE, and the reasons are the ones W4a's finding F1 and W4b-1's two
// green mutations already paid for:
//
//   * `layer_types` ALTERNATES full/sliding/full. A block of full
//     layers followed by a block of sliding ones would let a per-layer field
//     that is never RESET — the impl's `sliding_window`, the rope cache, the
//     dims — still produce the right answer; alternating makes a leak in
//     either direction wrong.
//   * `window` 3 against a 6-token prompt, so THREE of the six prefill queries
//     really lose a key and the decode query at position 6 keeps 3 of its 7.
//     At `window >= tokens` the windowed answer IS the causal answer and every
//     assertion here would pass on a port with no window at all. Both counts
//     are printed by the case.
//   * `swa_kv_lora` 6 against the full arm's 2, so the physical row (10) is
//     genuinely WIDER than a full layer's logical row (6) and the padding is
//     four real lanes rather than zero.
//   * `swa_heads` 3 against the full arm's 2 and `swa_qk_nope` 8 against 4, so
//     the two geometries disagree in head count, latent rank, NoPE width AND
//     softmax scale — a layer that ran the wrong `MlaBlockDims` cannot produce
//     the right answer by coincidence.
//   * `swa_rope_theta` 3 against the full arm's 1300 — two thetas ORDERS
//     apart, mirroring the released 5e4 against 8e7. W4b-1's fixture used 41
//     against 137 and measured a shared rope cache at only 0.0300 relative on
//     one LAYER; over a whole bf16 model that sits UNDER the quantisation
//     residue, so this fixture separates the two thetas the way upstream does
//     rather than the way a tidy pair of close numbers would.
//   * `page_size` 4 with a SHUFFLED block table {1, 0}: logical page 0 maps to
//     physical page 1. A contiguous table makes a paged read and a flat read
//     the same answer, which would leave the block lookup unproven — and the
//     windowed decode's key range starts INSIDE logical page 1, so the lookup
//     is on the windowed path and not only the prefill one.
//   * `q_lora` 3 over `hidden` 16 gives rescales sqrt(16/3)=2.309 (q) and
//     sqrt(16/2)=2.828 / sqrt(16/6)=1.633 (kv, full / sliding) — all far from
//     1.0 and different from each other, so neither a missing nor a swapped
//     scale can hide on either arm.
// ─────────────────────────────────────────────────────────────────────────────
struct Spec {
  int64_t hidden = 16;
  int64_t vocab = 12;
  int64_t inter = 10;
  int64_t max_pos = 32;
  double rms_eps = 1e-3;
  // the FULL arm — W4a's geometry, unchanged, so the two bricks' fixtures agree
  int64_t full_heads = 2;
  int64_t full_qk_nope = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 8;
  int64_t q_lora = 3;
  int64_t full_kv_lora = 2;
  double rope_theta = 1300.0;
  // the SLIDING arm
  int64_t swa_heads = 3;
  int64_t swa_qk_nope = 8;
  int64_t swa_kv_lora = 6;
  double swa_rope_theta = 3.0;
  int64_t window = 3;
  // >= prompt + 1, so the DSA top-k selects every causal candidate on the full
  // layers and dense attention IS upstream's answer. The refusal past this
  // bound has its own case.
  int64_t index_topk = 32;
  int64_t index_n_heads = 2;
  int64_t index_head_dim = 6;
  int64_t prompt = 6;
  int64_t page_size = 4;
  bool tie_word_embeddings = false;
  // {full, sliding, full}
  std::vector<Dots3NoteLayerKind> kinds{Dots3NoteLayerKind::kFullAttention,
                                        Dots3NoteLayerKind::kSlidingAttention,
                                        Dots3NoteLayerKind::kFullAttention};
  int64_t layers() const { return static_cast<int64_t>(kinds.size()); }
};

nlohmann::json ConfigDoc(const Spec& s) {
  nlohmann::json d = FixtureConfigDoc();
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = s.layers();
  nlohmann::json lt = nlohmann::json::array();
  for (Dots3NoteLayerKind k : s.kinds) {
    lt.push_back(k == Dots3NoteLayerKind::kSlidingAttention ? "sliding_attention"
                                                            : "full_attention");
  }
  d["layer_types"] = lt;
  d["num_attention_heads"] = s.full_heads;
  d["num_key_value_heads"] = s.full_heads;
  d["qk_nope_head_dim"] = s.full_qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.full_kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["max_position_embeddings"] = s.max_pos;
  d["index_n_heads"] = s.index_n_heads;
  d["index_head_dim"] = s.index_head_dim;
  d["index_topk"] = s.index_topk;
  d["swa_num_attention_heads"] = s.swa_heads;
  d["swa_num_key_value_heads"] = s.swa_heads;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.swa_kv_lora;
  d["swa_qk_nope_head_dim"] = s.swa_qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["swa_rope_theta"] = s.swa_rope_theta;
  d["sliding_window_size"] = s.window;
  d["vocab_size"] = s.vocab;
  d["intermediate_size"] = s.inter;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  // Every layer DENSE: W5 owns the MoE.
  d["first_k_dense_replace"] = s.layers();
  d["num_nextn_predict_layers"] = 0;
  d["tie_word_embeddings"] = s.tie_word_embeddings;
  return d;
}

// The whole tiny mixed model in double, every weight ALREADY bf16-rounded so
// the comparison measures the FORWARD and not the weights' storage width.
struct Weights {
  std::vector<double> embed;
  std::vector<double> final_norm;
  std::vector<double> lm_head;
  struct Layer {
    Dots3NoteLayerKind kind = Dots3NoteLayerKind::kFullAttention;
    std::vector<double> input_ln;
    std::vector<double> post_ln;
    FullAttnWeights full;    // populated iff kind == full
    SlidingAttnWeights swa;  // populated iff kind == sliding
    std::vector<double> gate_proj;
    std::vector<double> up_proj;
    std::vector<double> down_proj;
  };
  std::vector<Layer> layers;
};

std::vector<double> Ln(Rng& r, int64_t n) {
  std::vector<double> v = r.fill(n, 0.3);
  for (double& x : v) x += 1.0;
  return Bf16All(v);
}

Weights MakeWeights(const Spec& s, const FullAttnDims& fd, const SlidingAttnDims& sd,
                    uint64_t seed) {
  Rng r(seed);
  Weights w;
  w.embed = Bf16All(r.fill(s.vocab * s.hidden, 0.7));
  w.final_norm = Ln(r, s.hidden);
  w.lm_head = Bf16All(r.fill(s.vocab * s.hidden, 0.5));
  for (int64_t l = 0; l < s.layers(); ++l) {
    Weights::Layer lw;
    lw.kind = s.kinds[static_cast<size_t>(l)];
    lw.input_ln = Ln(r, s.hidden);
    lw.post_ln = Ln(r, s.hidden);
    const uint64_t lseed = seed + 0x1000ULL * static_cast<uint64_t>(l + 1);
    if (lw.kind == Dots3NoteLayerKind::kFullAttention) {
      lw.full = TinyWeights(fd, lseed);
      // `k_rope_only_layernorm` is made OBSERVABLE the way W4a's bench makes it
      // observable, and for the reason spec §4.6 records: RoPE preserves the L2
      // norm of every rotated pair exactly, so the only part of the norm that
      // does not commute with the rotation is the PER-LANE weight. Weights
      // hugging 1.0 let "norm AFTER the rope" slip under any sane bound.
      {
        const int64_t R = fd.qk_rope_head_dim, H = fd.hidden_size;
        const int64_t off = fd.kv_lora_rank * H;
        for (int64_t i = 0; i < R * H; ++i)
          lw.full.kv_a_proj_with_mqa[static_cast<size_t>(off + i)] *= 6.0;
        for (int64_t i = 0; i < R; ++i)
          lw.full.k_rope_only_layernorm[static_cast<size_t>(i)] = (i % 2 == 0) ? 2.5 : 0.3;
      }
      lw.full.q_a_proj = Bf16All(lw.full.q_a_proj);
      lw.full.q_a_layernorm = Bf16All(lw.full.q_a_layernorm);
      lw.full.kv_a_layernorm = Bf16All(lw.full.kv_a_layernorm);
      lw.full.kv_a_proj_with_mqa = Bf16All(lw.full.kv_a_proj_with_mqa);
      lw.full.k_rope_only_layernorm = Bf16All(lw.full.k_rope_only_layernorm);
      lw.full.q_b_proj = Bf16All(lw.full.q_b_proj);
      lw.full.kv_b_proj = Bf16All(lw.full.kv_b_proj);
      lw.full.o_proj = Bf16All(lw.full.o_proj);
      lw.full.g_proj = Bf16All(lw.full.g_proj);
      lw.full.indexer_wq_b = Bf16All(lw.full.indexer_wq_b);
      lw.full.indexer_wk = Bf16All(lw.full.indexer_wk);
      lw.full.indexer_weights_proj = Bf16All(lw.full.indexer_weights_proj);
      lw.full.indexer_k_norm_weight = Bf16All(lw.full.indexer_k_norm_weight);
      lw.full.indexer_k_norm_bias = Bf16All(lw.full.indexer_k_norm_bias);
    } else {
      // `w4b::SwaWeights` already alternates the k_pe norm 2.5/0.3 within each
      // rotated pair, for the same reason, and it needs no extra amplification
      // of the rope rows: dropping the norm on this arm moves the logits by
      // more than 1.5 relative, which the case prints.
      lw.swa = w4b::SwaWeights(sd, lseed);
      lw.swa.q_a_proj = Bf16All(lw.swa.q_a_proj);
      lw.swa.q_a_layernorm = Bf16All(lw.swa.q_a_layernorm);
      lw.swa.kv_a_layernorm = Bf16All(lw.swa.kv_a_layernorm);
      lw.swa.kv_a_proj_with_mqa = Bf16All(lw.swa.kv_a_proj_with_mqa);
      lw.swa.k_rope_only_layernorm = Bf16All(lw.swa.k_rope_only_layernorm);
      lw.swa.q_b_proj = Bf16All(lw.swa.q_b_proj);
      lw.swa.kv_b_proj = Bf16All(lw.swa.kv_b_proj);
      lw.swa.o_proj = Bf16All(lw.swa.o_proj);
      lw.swa.g_proj = Bf16All(lw.swa.g_proj);
    }
    lw.gate_proj = Bf16All(r.fill(s.inter * s.hidden, 0.5));
    lw.up_proj = Bf16All(r.fill(s.inter * s.hidden, 0.5));
    lw.down_proj = Bf16All(r.fill(s.hidden * s.inter, 0.5));
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The on-disk entries in the names `EnumerateDots3NoteTensors` claims. The five
// indexer tensors exist ONLY on the full layers, which is upstream's own shape
// (`Dots3NoteSlidingAttention` sets `self.indexer = None`, model.py:432) and
// what the W1 enumerator already encodes.
std::vector<StOut> CheckpointOf(const Spec& s, const FullAttnDims& fd,
                                const SlidingAttnDims& sd, const Weights& w) {
  const int64_t H = s.hidden;
  std::vector<StOut> e;
  e.push_back({"model.embed_tokens.weight", {s.vocab, H}, w.embed});
  e.push_back({"model.norm.weight", {H}, w.final_norm});
  if (!s.tie_word_embeddings) e.push_back({"lm_head.weight", {s.vocab, H}, w.lm_head});
  for (int64_t l = 0; l < s.layers(); ++l) {
    const Weights::Layer& lw = w.layers[static_cast<size_t>(l)];
    const bool sliding = lw.kind == Dots3NoteLayerKind::kSlidingAttention;
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const std::string sa = p + "self_attn.";
    const int64_t N = sliding ? sd.num_heads : fd.num_heads;
    const int64_t QK = sliding ? sd.qk_head_dim() : fd.qk_head_dim();
    const int64_t P = sliding ? sd.qk_nope_head_dim : fd.qk_nope_head_dim;
    const int64_t V = sliding ? sd.v_head_dim : fd.v_head_dim;
    const int64_t L = sliding ? sd.kv_lora_rank : fd.kv_lora_rank;
    const int64_t R = sliding ? sd.qk_rope_head_dim : fd.qk_rope_head_dim;
    const int64_t QL = sliding ? sd.q_lora_rank : fd.q_lora_rank;
    const SlidingAttnWeights& sw = lw.swa;
    const FullAttnWeights& fw = lw.full;
    e.push_back({p + "input_layernorm.weight", {H}, lw.input_ln});
    e.push_back({p + "post_attention_layernorm.weight", {H}, lw.post_ln});
    e.push_back({sa + "q_a_proj.weight", {QL, H}, sliding ? sw.q_a_proj : fw.q_a_proj});
    e.push_back({sa + "q_a_layernorm.weight",
                 {QL},
                 sliding ? sw.q_a_layernorm : fw.q_a_layernorm});
    e.push_back({sa + "q_b_proj.weight",
                 {N * QK, QL},
                 sliding ? sw.q_b_proj : fw.q_b_proj});
    e.push_back({sa + "kv_a_proj_with_mqa.weight",
                 {L + R, H},
                 sliding ? sw.kv_a_proj_with_mqa : fw.kv_a_proj_with_mqa});
    e.push_back({sa + "kv_a_layernorm.weight",
                 {L},
                 sliding ? sw.kv_a_layernorm : fw.kv_a_layernorm});
    e.push_back({sa + "kv_b_proj.weight",
                 {N * (P + V), L},
                 sliding ? sw.kv_b_proj : fw.kv_b_proj});
    e.push_back({sa + "o_proj.weight", {H, N * V}, sliding ? sw.o_proj : fw.o_proj});
    e.push_back({sa + "g_proj.weight", {N, H}, sliding ? sw.g_proj : fw.g_proj});
    e.push_back({sa + "k_rope_only_layernorm.weight",
                 {R},
                 sliding ? sw.k_rope_only_layernorm : fw.k_rope_only_layernorm});
    if (!sliding) {
      e.push_back({sa + "indexer.wq_b.weight",
                   {fd.index_n_heads * fd.index_head_dim, fd.q_lora_rank},
                   fw.indexer_wq_b});
      e.push_back({sa + "indexer.wk.weight", {fd.index_head_dim, H}, fw.indexer_wk});
      e.push_back({sa + "indexer.k_norm.weight",
                   {fd.index_head_dim},
                   fw.indexer_k_norm_weight});
      e.push_back({sa + "indexer.k_norm.bias",
                   {fd.index_head_dim},
                   fw.indexer_k_norm_bias});
      e.push_back({sa + "indexer.weights_proj.weight",
                   {fd.index_n_heads, H},
                   fw.indexer_weights_proj});
    }
    e.push_back({p + "mlp.gate_proj.weight", {s.inter, H}, lw.gate_proj});
    e.push_back({p + "mlp.up_proj.weight", {s.inter, H}, lw.up_proj});
    e.push_back({p + "mlp.down_proj.weight", {H, s.inter}, lw.down_proj});
  }
  return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// The MIXED whole-model reference. The residual stream, the MLP and the lm_head
// are W4a's; the attention DISPATCHES on the layer kind, into W3's
// `ref::Forward` or W4b-1's `sref::Forward`. Neither of those knows about a
// cache, a block table, a slot mapping, a gather or a physical row: the sliding
// one takes the window as the direct positional predicate `s <= t && t - s < W`
// over a materialized MHA. That is the whole point — the device arm's answer
// has to survive being computed a completely different way.
std::vector<double> RefModel(const Spec& s, const FullAttnDims& fd,
                             const SlidingAttnDims& sd, const Weights& w,
                             const std::vector<int32_t>& tokens,
                             const std::vector<int32_t>& positions, const ref::Opts& fo,
                             const w4b::sref::Opts& so) {
  const int64_t T = static_cast<int64_t>(tokens.size()), H = s.hidden;
  std::vector<double> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < H; ++c) {
      hidden[static_cast<size_t>(t * H + c)] =
          w.embed[static_cast<size_t>(tokens[static_cast<size_t>(t)] * H + c)];
    }
  }
  std::vector<double> res(static_cast<size_t>(T * H), 0.0);
  for (int64_t l = 0; l < s.layers(); ++l) {
    const Weights::Layer& lw = w.layers[static_cast<size_t>(l)];
    for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
    const std::vector<double> x = ref::Rms(res, lw.input_ln, T, H, s.rms_eps);
    const std::vector<double> a =
        lw.kind == Dots3NoteLayerKind::kSlidingAttention
            ? w4b::sref::Forward(sd, lw.swa, x, positions, T, so).out
            : ref::Forward(fd, lw.full, x, positions, T, fo).out;
    for (size_t i = 0; i < res.size(); ++i) res[i] += a[i];
    const std::vector<double> y = ref::Rms(res, lw.post_ln, T, H, s.rms_eps);
    const std::vector<double> g = ref::Dense(y, lw.gate_proj, T, H, s.inter);
    const std::vector<double> u = ref::Dense(y, lw.up_proj, T, H, s.inter);
    std::vector<double> act(g.size());
    for (size_t i = 0; i < g.size(); ++i) act[i] = (g[i] / (1.0 + std::exp(-g[i]))) * u[i];
    hidden = ref::Dense(act, lw.down_proj, T, s.inter, H);
  }
  for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
  const std::vector<double> z = ref::Rms(res, w.final_norm, T, H, s.rms_eps);
  return ref::Dense(z, w.lm_head, T, H, s.vocab);
}

// The PHYSICAL slot a logical position lands in, through the SHUFFLED block
// table. Derived here by hand rather than taken from the forward, so the test
// and the code cannot share one arithmetic bug.
int64_t SlotOf(const Spec& s, const std::vector<int32_t>& block_table, int64_t pos) {
  const int64_t page = pos / s.page_size;
  REQUIRE(page < static_cast<int64_t>(block_table.size()));
  return static_cast<int64_t>(block_table[static_cast<size_t>(page)]) * s.page_size +
         pos % s.page_size;
}

// `Dots3NoteFullAttnDimsFrom` REFUSES a schedule with no `full_attention`
// layer, by design (spec §1.1) — and one of the cases below drives exactly that
// schedule, to show that a pure-SWA config is not refused for a DSA indexer it
// does not have. An all-sliding bench never reads these dims, so it gets a
// default-constructed struct rather than the refusal.
FullAttnDims FullDimsOrDefault(const Dots3NoteParams& p) {
  for (Dots3NoteLayerKind k : p.layer_types) {
    if (k == Dots3NoteLayerKind::kFullAttention) return Dots3NoteFullAttnDimsFrom(p);
  }
  return FullAttnDims{};
}

// The mixed bench. The registration, the config and the loaded model all come
// from the REAL registry over the REAL loader; the geometry comes from the
// released `config.json` with the fields above overridden, so every W1
// validation still applies to it.
struct Bench {
  Spec spec;
  TempConfig cfg;
  HfConfig config;
  Dots3NoteParams params;
  // The HOST reference geometries (W3 / W4b-1), which drive the weights and the
  // double reference...
  FullAttnDims fdims;
  SlidingAttnDims sdims;
  // ...and the DEVICE seam geometries the forward itself builds, which is where
  // the softmax scale and the window live.
  vllm::mla::MlaBlockDims mfull;
  vllm::mla::MlaBlockDims mswa;
  Weights w;
  std::vector<StOut> entries;
  std::vector<int32_t> tokens;  // the prompt PLUS the token the decode step runs
  std::vector<int32_t> positions;
  std::vector<int32_t> block_table{1, 0};  // SHUFFLED

  explicit Bench(Spec s = Spec{})
      : spec(s),
        cfg(ConfigDoc(s)),
        config(LoadHfConfig(cfg.path())),
        params(ParseDots3NoteParams(config)),
        fdims(FullDimsOrDefault(params)),
        sdims(Dots3NoteSlidingAttnDimsFrom(params)),
        mfull(Dots3NoteFullAttnMlaDims(params)),
        mswa(Dots3NoteSlidingAttnMlaDims(params)),
        w(MakeWeights(s, fdims, sdims, 0x243F6A8885A308D3ULL)),
        entries(CheckpointOf(s, fdims, sdims, w)) {
    for (int64_t t = 0; t <= spec.prompt; ++t) {
      tokens.push_back(static_cast<int32_t>((t * 5 + 1) % spec.vocab));
      positions.push_back(static_cast<int32_t>(t));
    }
  }

  // The whole point of this brick: PREFILL then DECODE, both through
  // `ModelRegistry::Forward`, against ONE cache pool. Returns the decode step's
  // [1, vocab] logits, and optionally the raw cache bytes afterwards.
  std::vector<double> RunPrefillThenDecode(
      std::vector<std::vector<uint16_t>>* cache_out = nullptr,
      int64_t cache_row_override = 0) const {
    const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
    w4a::TempCheckpoint ckpt(entries);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    std::unique_ptr<vllm::LoadedModel> model = reg.factory->load_weights(reg, config, source);
    REQUIRE(model != nullptr);

    const int64_t row =
        cache_row_override > 0 ? cache_row_override : params.physical_latent_row();
    w4a::MlaCachePool pool(spec.layers(), row, /*num_blocks=*/2, spec.page_size);
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const std::vector<int32_t> no_gather;
    std::vector<vllm::GdnStateCache> gdn_state;
    vllm::v1::GDNAttentionMetadata gdn_meta{};

    // ── step 1: PREFILL the prompt ─────────────────────────────────────────
    {
      vllm::v1::CommonAttentionMetadata m;
      m.num_reqs = 1;
      m.num_actual_tokens = static_cast<int>(spec.prompt);
      m.query_start_loc = {0, static_cast<int32_t>(spec.prompt)};
      m.query_start_loc_cpu = m.query_start_loc;
      m.seq_lens = {static_cast<int32_t>(spec.prompt)};
      m.seq_lens_cpu = m.seq_lens;
      m.max_query_len = static_cast<int>(spec.prompt);
      m.max_seq_len = static_cast<int>(spec.prompt);
      m.block_table_num_cols = static_cast<int>(block_table.size());
      m.block_table_tensor = block_table;
      for (int64_t t = 0; t < spec.prompt; ++t) {
        m.slot_mapping.push_back(SlotOf(spec, block_table, t));
      }
      m.causal = true;
      const std::vector<int32_t> ids(tokens.begin(), tokens.begin() + spec.prompt);
      const std::vector<int32_t> pos(positions.begin(), positions.begin() + spec.prompt);
      const vllm::ModelForwardInput in{.token_ids = ids,
                                       .positions = pos,
                                       .attn_meta = m,
                                       .gdn_meta = gdn_meta,
                                       .attn_kv = pool.attn_kv,
                                       .gdn_state = gdn_state,
                                       .config = config,
                                       .queue = queue,
                                       .logits_indices = no_gather,
                                       .num_reqs = 1};
      const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);
      REQUIRE(fl.on_device());
      REQUIRE(fl.rows == spec.prompt);
    }

    // ── step 2: DECODE the next token, against the cache step 1 wrote ──────
    std::vector<double> out;
    {
      vllm::v1::CommonAttentionMetadata m;
      m.num_reqs = 1;
      m.num_actual_tokens = 1;
      m.query_start_loc = {0, 1};
      m.query_start_loc_cpu = m.query_start_loc;
      m.seq_lens = {static_cast<int32_t>(spec.prompt + 1)};
      m.seq_lens_cpu = m.seq_lens;
      m.max_query_len = 1;
      m.max_seq_len = static_cast<int>(spec.prompt + 1);
      m.block_table_num_cols = static_cast<int>(block_table.size());
      m.block_table_tensor = block_table;
      m.slot_mapping = {SlotOf(spec, block_table, spec.prompt)};
      m.causal = true;
      const std::vector<int32_t> ids{tokens.back()};
      const std::vector<int32_t> pos{positions.back()};
      const vllm::ModelForwardInput in{.token_ids = ids,
                                       .positions = pos,
                                       .attn_meta = m,
                                       .gdn_meta = gdn_meta,
                                       .attn_kv = pool.attn_kv,
                                       .gdn_state = gdn_state,
                                       .config = config,
                                       .queue = queue,
                                       .logits_indices = no_gather,
                                       .num_reqs = 1};
      const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);
      REQUIRE(fl.on_device());
      REQUIRE(fl.rows == 1);
      REQUIRE(fl.vocab == spec.vocab);
      const auto* src = static_cast<const float*>(fl.device_tensor.data);
      out.assign(static_cast<size_t>(fl.vocab), 0.0);
      for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(src[i]);
    }
    if (cache_out != nullptr) *cache_out = pool.buf;
    return out;
  }

  // The reference's logits for the LAST position, i.e. the row the decode step
  // produces. The reference recomputes the whole 7-token sequence from scratch
  // in double and has no cache at all, so "what the decode step read out of the
  // cache" has to equal "what a fresh full-sequence forward computes".
  std::vector<double> RefLastRow(const ref::Opts& fo = ref::Opts{},
                                 const w4b::sref::Opts& so = w4b::sref::Opts{}) const {
    const std::vector<double> all =
        RefModel(spec, fdims, sdims, w, tokens, positions, fo, so);
    const int64_t T = static_cast<int64_t>(tokens.size());
    return std::vector<double>(all.begin() + static_cast<long>((T - 1) * spec.vocab),
                               all.end());
  }
};

// The bf16 agreement bound, chosen for SEPARATION and not to hug the residue —
// W4a's review finding F1 applied to a THREE-layer model whose activation
// stream is bf16 end to end while the reference is double throughout.
//
// THREE, not four. The first draft of this fixture ran `{full, sliding, full,
// sliding}`; the retune §4.8 records dropped it to `{full, sliding, full}`,
// which is still full/sliding/full so a per-layer field leaking in EITHER
// direction is wrong. `Spec::kinds` is the committed truth and the case
// `REQUIRE`s `num_hidden_layers == 3`; two comments kept saying four until the
// W4b-2 review read them against the code.
//
// THREE ratios, kept SEPARATE, because merging any two of them overstates the
// headroom: spec §4.6 records a draft that did exactly that and was wrong by
// 2.8x. The cases PRINT all three from the numbers they just measured, so this
// constant is the only thing written down in advance.
constexpr double kMixedRel = 6e-2;

}  // namespace w4b2
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE(
    "dots3-note W4b-2: the SLIDING layer is REACHED through "
    "ModelRegistry::Forward, over a PADDED cache, and agrees with the "
    "independent reference") {
  const w4b2::Bench b;
  // The scope boundary first: this config is one the device path ACCEPTS, and
  // it is genuinely mixed and genuinely padded.
  CHECK(w4b2::Dots3NoteDeviceRefusal(b.params).empty());
  REQUIRE(b.params.num_hidden_layers == 3);
  int64_t n_full = 0, n_sliding = 0;
  for (int64_t l = 0; l < b.params.num_hidden_layers; ++l) {
    if (b.params.kind_of(l) == vllm::Dots3NoteLayerKind::kSlidingAttention) {
      ++n_sliding;
    } else {
      ++n_full;
    }
    CHECK_FALSE(b.params.is_moe_layer(l));
  }
  CHECK(n_full == 2);
  CHECK(n_sliding == 1);
  // The PADDING is real: the physical row is wider than a full layer's logical
  // one. Without this the narrowing is the identity and proves nothing.
  REQUIRE(b.params.physical_latent_row() == 10);
  REQUIRE(b.params.full.latent_row() == 6);
  REQUIRE(b.params.swa.latent_row() == b.params.physical_latent_row());
  // The two geometries really differ, on every axis that could hide a mistake.
  CHECK(b.mswa.num_heads != b.mfull.num_heads);
  CHECK(b.mswa.kv_lora_rank != b.mfull.kv_lora_rank);
  CHECK(b.mswa.qk_nope_head_dim != b.mfull.qk_nope_head_dim);
  CHECK(b.mswa.scale != b.mfull.scale);
  CHECK(b.mswa.sliding_window == 3);
  CHECK(b.mfull.sliding_window == 0);
  CHECK(b.mswa.head_size() == b.params.physical_latent_row());
  CHECK(b.mfull.head_size() < b.params.physical_latent_row());

  // The WINDOW bites, COUNTED rather than assumed. Prefill: query t sees keys
  // [t - 2, t], so queries 3, 4 and 5 lose 1, 2 and 3 keys. Decode: the query
  // at position 6 keeps keys 4..6 of 0..6, i.e. it loses four.
  int64_t prefill_queries_that_lose = 0, prefill_keys_dropped = 0;
  for (int64_t t = 0; t < b.spec.prompt; ++t) {
    const int64_t lost = std::max<int64_t>(0, t - (b.spec.window - 1));
    if (lost > 0) ++prefill_queries_that_lose;
    prefill_keys_dropped += lost;
  }
  const int64_t decode_keys_dropped = b.spec.prompt + 1 - b.spec.window;
  MESSAGE("W4b-2 window bite: prefill "
          << prefill_queries_that_lose << " of " << b.spec.prompt
          << " queries lose a key (" << prefill_keys_dropped
          << " keys dropped); the decode query at position " << b.spec.prompt
          << " keeps " << b.spec.window << " of " << (b.spec.prompt + 1) << " keys");
  REQUIRE(prefill_queries_that_lose == 3);
  REQUIRE(prefill_keys_dropped == 6);
  REQUIRE(decode_keys_dropped == 4);

  std::vector<std::vector<uint16_t>> cache;
  const std::vector<double> got = b.RunPrefillThenDecode(&cache);
  const std::vector<double> want = b.RefLastRow();
  const Diff d = Compare(got, want);
  MESSAGE("W4b-2 mixed prefill+decode vs the independent reference: max|diff| "
          << d.max_abs << " over a scale of " << d.max_mag << " = " << d.max_rel
          << " relative; bound " << w4b2::kMixedRel << " = "
          << (w4b2::kMixedRel / d.max_rel) << "x the residue");
  CHECK(d.max_rel < w4b2::kMixedRel);

  // ── the PADDED row's spare lanes, read out of the RAW cache ──────────────
  // A FULL layer writes `full.latent_row()` lanes into a `physical_row`-wide
  // slot. Upstream narrows the cache view on read AND on write
  // (`_logical_cache`, attention.py:700-720), so the tail of every physical row
  // it touches stays exactly as the allocator left it — zero. A port that wrote
  // at the physical stride, or read at the logical one, breaks this.
  const int64_t phys = b.params.physical_latent_row();
  const int64_t logical = b.params.full.latent_row();
  int64_t full_slots_checked = 0, sliding_nonzero_pad_lanes = 0;
  for (int64_t l = 0; l < b.params.num_hidden_layers; ++l) {
    const bool sliding =
        b.params.kind_of(l) == vllm::Dots3NoteLayerKind::kSlidingAttention;
    const std::vector<uint16_t>& buf = cache[static_cast<size_t>(l)];
    for (int64_t t = 0; t <= b.spec.prompt; ++t) {
      const int64_t slot = w4b2::SlotOf(b.spec, b.block_table, t);
      for (int64_t c = logical; c < phys; ++c) {
        const uint16_t v = buf[static_cast<size_t>(slot * phys + c)];
        if (sliding) {
          if (v != 0) ++sliding_nonzero_pad_lanes;
        } else {
          CHECK(v == 0);
        }
      }
      if (!sliding) ++full_slots_checked;
    }
  }
  MESSAGE("W4b-2 padded row: " << full_slots_checked
                               << " full-layer slots checked, lanes [" << logical
                               << ", " << phys
                               << ") all ZERO; the same lanes on the sliding layers "
                                  "carry "
                               << sliding_nonzero_pad_lanes << " non-zero values");
  CHECK(full_slots_checked == 2 * (b.spec.prompt + 1));
  // The CONTROL: those lanes are not zero everywhere. A sliding layer's logical
  // row IS the physical one, so it writes them — which is what makes the
  // all-zero assertion above a statement about the NARROWING rather than about
  // the fixture happening to produce zeros.
  CHECK(sliding_nonzero_pad_lanes > 0);
}

TEST_CASE("dots3-note W4b-2: the mixed device forward is DETERMINISTIC run to run") {
  const w4b2::Bench b;
  const std::vector<double> a = b.RunPrefillThenDecode();
  const std::vector<double> c = b.RunPrefillThenDecode();
  REQUIRE(a.size() == c.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK(a[i] == c[i]);
}

TEST_CASE(
    "dots3-note W4b-2: the WINDOW is what makes the answer different, and each "
    "sliding-only mechanism is EXERCISED on the device path") {
  const w4b2::Bench b;
  const std::vector<double> got = b.RunPrefillThenDecode();
  const Diff base = Compare(got, b.RefLastRow());

  // The reference with `windowed = false` models a port that never noticed
  // `sliding_window_size` and ran plain causal attention on the 33 sliding
  // layers. That is THE defect this brick exists to prevent, and no shape check
  // can see it.
  w4b::sref::Opts unwindowed;
  unwindowed.windowed = false;
  const Diff no_win = Compare(got, b.RefLastRow(ref::Opts{}, unwindowed));
  MESSAGE("W4b-2 separation: residue " << base.max_rel << "; the bound "
                                       << w4b2::kMixedRel << " is "
                                       << (w4b2::kMixedRel / base.max_rel)
                                       << "x the residue");
  MESSAGE("W4b-2 with NO WINDOW at all: " << no_win.max_rel << " relative = "
                                          << (no_win.max_rel / w4b2::kMixedRel)
                                          << "x the BOUND, "
                                          << (no_win.max_rel / base.max_rel)
                                          << "x the RESIDUE");
  CHECK(no_win.max_rel > w4b2::kMixedRel);

  // Each sliding-only mechanism, neutralised in the REFERENCE, with the device
  // arm drifting AWAY. Ratios are given against both the bound and the residue,
  // LABELLED, because the two are different statements (spec §4.6 F1).
  struct Arm {
    std::string what;
    w4b::sref::Opts so;
  };
  std::vector<Arm> arms;
  {
    Arm a{"the sliding arm inheriting the MODEL-level rope theta", {}};
    a.so.rope_theta_override = b.spec.rope_theta;
    arms.push_back(a);
  }
  {
    Arm a{"the sliding arm's q LoRA rescale dropped", {}};
    a.so.apply_q_lora_rescale = false;
    arms.push_back(a);
  }
  {
    Arm a{"the sliding arm's kv LoRA rescale dropped", {}};
    a.so.apply_kv_lora_rescale = false;
    arms.push_back(a);
  }
  {
    Arm a{"the sliding arm's k_rope_only_layernorm dropped", {}};
    a.so.k_rope_only_norm = false;
    arms.push_back(a);
  }
  {
    Arm a{"the sliding arm's headwise gate made lane-wise", {}};
    a.so.headwise_gate = false;
    arms.push_back(a);
  }
  for (const Arm& a : arms) {
    const Diff dd = Compare(got, b.RefLastRow(ref::Opts{}, a.so));
    MESSAGE("W4b-2 with " << a.what << ": " << dd.max_rel << " relative = "
                          << (dd.max_rel / w4b2::kMixedRel) << "x the BOUND, "
                          << (dd.max_rel / base.max_rel) << "x the RESIDUE");
    CHECK(dd.max_rel > w4b2::kMixedRel);
  }
}

TEST_CASE(
    "dots3-note W4b-2: the SLIDING geometry the DEVICE forward runs comes off "
    "the RELEASED config") {
  // Not a hand-typed struct: the released `config.json`, through
  // `ParseDots3NoteParams` -> `Dots3NoteSlidingAttnMlaDims`, which is the
  // function `MaterializeDots3NoteDevice` and `ForwardDevice` both call.
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const Dots3NoteParams p = ParseDots3NoteParams(config);
  const vllm::mla::MlaBlockDims sd = vllm::Dots3NoteSlidingAttnMlaDims(p);
  const vllm::mla::MlaBlockDims fd = vllm::Dots3NoteFullAttnMlaDims(p);
  CHECK(sd.num_heads == 64);
  CHECK(sd.kv_lora_rank == 1024);
  CHECK(sd.qk_nope_head_dim == 192);
  CHECK(sd.qk_rope_head_dim == 64);
  CHECK(sd.v_head_dim == 128);
  CHECK(sd.q_lora_rank == 1024);
  CHECK(sd.head_size() == 1088);
  CHECK(sd.head_size() == p.physical_latent_row());
  // `sliding_window=config.sliding_window_size` (model.py:457).
  CHECK(sd.sliding_window == 513);
  CHECK(fd.sliding_window == 0);
  // `scale = qk_head_dim ** -0.5` (model.py:446) — 256^-0.5, NOT the full arm's
  // 192^-0.5. No YaRN and no mscale on either arm.
  CHECK(sd.scale == doctest::Approx(1.0 / std::sqrt(256.0)).epsilon(1e-6));
  CHECK(fd.scale == doctest::Approx(1.0 / std::sqrt(192.0)).epsilon(1e-6));
  // Both geometries are GPT-J (spec §4 item 6, #1804).
  CHECK_FALSE(sd.is_neox_style);
  CHECK_FALSE(fd.is_neox_style);
  // The released ranks make the two SLIDING rescales EQUAL at sqrt(5120/1024);
  // the full arm's kv rescale is sqrt(5120/512) and differs.
  CHECK(sd.q_lora_scale == doctest::Approx(std::sqrt(5120.0 / 1024.0)).epsilon(1e-12));
  CHECK(sd.kv_lora_scale == doctest::Approx(std::sqrt(5120.0 / 1024.0)).epsilon(1e-12));
  CHECK(fd.kv_lora_scale == doctest::Approx(std::sqrt(5120.0 / 512.0)).epsilon(1e-12));
  // The full layers read 576 out of the 1088-wide physical row: the padding is
  // 512 real lanes on 13 of the 46 layers.
  CHECK(fd.head_size() == 576);
  CHECK(p.physical_latent_row() - fd.head_size() == 512);
}

TEST_CASE("dots3-note W4b-2/W5: what the device path refuses, by name") {
  // (a) a MoE layer — LIFTED at W5. Kept as an ACCEPTANCE so a reader
  //     following W4b-2's evidence lands on the answer rather than a gap.
  {
    nlohmann::json doc = w4b2::ConfigDoc(w4b2::Spec{});
    doc["first_k_dense_replace"] = 1;
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4b2::Dots3NoteDeviceRefusal(p).empty());
  }
  // (b) the RELEASED config — LIFTED at W5 (the MoE layer) and W5c (the nextn
  //     tail, #2176). W4b-2 asserted `why.find("MoE") != npos` here; that
  //     `true -> false` flip is the row's headline and is asserted rather than
  //     deleted. It is NOT a claim that the model runs: the MoE is 545.82 GB of
  //     a 576.89 GB checkpoint (94.62%), and nothing this project owns holds it.
  {
    TempConfig cfg(FixtureConfigDoc());
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    const std::string why = w4b2::Dots3NoteDeviceRefusal(p);
    MESSAGE("W5 released-config refusal: '" << why << "' (empty)");
    CHECK(why.empty());
  }
  // (c) a sequence past `index_topk`, on a config that HAS a full layer. The
  //     DSA selection is still not on the device path (W4b-3).
  {
    w4b2::Spec s;
    s.index_topk = 2;  // < the prompt
    const w4b2::Bench b(s);
    CHECK(w4b2::Dots3NoteDeviceRefusal(b.params).empty());  // the CONFIG is fine
    CHECK_THROWS_WITH_AS(b.RunPrefillThenDecode(), doctest::Contains("index_topk"),
                         std::runtime_error);
  }
  // (d) the SAME `index_topk` on a config with NO full layer RUNS, because a
  //     sliding layer carries no indexer at all (`self.indexer = None`,
  //     model.py:432). This is the narrowing W4b-2 made, asserted rather than
  //     described — it is what stops a pure-SWA config being refused for a
  //     mechanism it does not have.
  {
    w4b2::Spec s;
    s.index_topk = 2;
    s.kinds = {vllm::Dots3NoteLayerKind::kSlidingAttention,
               vllm::Dots3NoteLayerKind::kSlidingAttention};
    const w4b2::Bench b(s);
    CHECK(w4b2::Dots3NoteDeviceRefusal(b.params).empty());
    CHECK_NOTHROW((void)b.RunPrefillThenDecode());
  }
  // (e) a nextn tail — LIFTED at W5c (#2176), because the refusal was STRICTER
  //     than upstream: vLLM drops those weights from the main model rather than
  //     turning the checkpoint away.
  {
    nlohmann::json doc = w4b2::ConfigDoc(w4b2::Spec{});
    doc["num_nextn_predict_layers"] = 1;
    TempConfig cfg(doc);
    const Dots3NoteParams p = ParseDots3NoteParams(LoadHfConfig(cfg.path()));
    CHECK(w4b2::Dots3NoteDeviceRefusal(p).empty());
  }
  // (f) a KV cache whose row disagrees with the config it was built from. The
  //     config-level checks cannot see this — an engine allocates the cache
  //     separately — so the PER-STEP assertion stays, and this is what makes it
  //     reached rather than defensive decoration.
  {
    const w4b2::Bench b;
    CHECK(w4b2::Dots3NoteDeviceRefusal(b.params).empty());
    CHECK_THROWS_WITH_AS(
        b.RunPrefillThenDecode(nullptr, b.params.physical_latent_row() + 3),
        doctest::Contains("PHYSICAL row"), std::runtime_error);
  }
}

TEST_CASE(
    "dots3-note W4b-2: a windowed PREFILL that also has chunked CONTEXT is "
    "refused BY NAME inside the shared seam") {
  // Upstream's windowed prefill caps its gather at
  // `min(seq_len, query_len + W - 1)` and runs one varlen call per request group
  // (attention.py:206, :594-654); it never merges context chunks under a window,
  // so there is no windowed form of `forward_mha`'s LSE merge to mirror. The
  // seam refuses rather than merging an UNwindowed context into a windowed
  // suffix, which would be silently wrong. W4b-3 owes it.
  //
  // The operands are deliberately EMPTY: the refusal is at the top of the
  // function, before anything is read, which is the same discipline W4a's
  // finding F6 applied to the gate's dtype check. The control below is what
  // makes that a statement about the WINDOW rather than about the empty
  // tensors — with `sliding_window == 0` the same call reaches the op and fails
  // there instead, with a different type and a different message.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::mla::MlaPrefillContextBuffers bufs{};
  std::vector<vllm::mla::MlaChunkDeviceMetadata> chunks(1);
  vt::Tensor empty{};
  vllm::mla::MlaUpProjectFn up;
  CHECK_THROWS_WITH_AS(
      vllm::mla::ForwardMlaPrefillMha(q, empty, empty, empty, empty, empty, empty, empty,
                                      chunks, up, 1.0f, 1, 0, bufs, empty, empty,
                                      /*sliding_window=*/513),
      doctest::Contains("SLIDING-WINDOW layer with chunked CONTEXT"),
      std::invalid_argument);
  // The CONTROL. Without a window the guard is a NOT-TAKEN branch, so the call
  // proceeds into `vt::MlaPrefillAttention` and fails on the empty operands —
  // a `std::runtime_error` from VT_CHECK, not the `std::invalid_argument`
  // above. Asserting the TYPE is what stops this control passing vacuously.
  CHECK_THROWS_AS(
      vllm::mla::ForwardMlaPrefillMha(q, empty, empty, empty, empty, empty, empty, empty,
                                      chunks, up, 1.0f, 1, 0, bufs, empty, empty,
                                      /*sliding_window=*/0),
      std::runtime_error);
  // And with a window but NO context the guard does not fire either: the chunk
  // list is what pairs with it, so an ordinary windowed prefill is unaffected.
  const std::vector<vllm::mla::MlaChunkDeviceMetadata> no_chunks;
  CHECK_THROWS_AS(
      vllm::mla::ForwardMlaPrefillMha(q, empty, empty, empty, empty, empty, empty, empty,
                                      no_chunks, up, 1.0f, 1, 0, bufs, empty, empty,
                                      /*sliding_window=*/513),
      std::runtime_error);
}

TEST_CASE(
    "dots3-note W4b-3c: a RESUMED request past `index_topk` still refuses BY "
    "NAME, and names the row that owns the index cache") {
  // The narrowed refusal. The discriminator is the indexer's own KEY CACHE, not
  // the sequence length: `wk_weights_proj` builds a token's index key from that
  // token's hidden state (deepseek_v2.py:808-810), so a single-shot prefill has
  // every key in hand and a resumed step does not. Carrying the indexer's
  // 128-wide cache is a SECOND attention group on the same layers, which is
  // `KV-DSV4-MULTICACHE` (#1925).
  w4b2::Spec s;
  s.index_topk = 2;  // < the prompt
  const w4b2::Bench b(s);
  CHECK(w4b2::Dots3NoteDeviceRefusal(b.params).empty());  // the CONFIG is fine
  // The PREFILL half of the same run is served; it is the DECODE step, which
  // resumes from the prompt, that refuses.
  CHECK_THROWS_WITH_AS(b.RunPrefillThenDecode(), doctest::Contains("index_topk"),
                       std::runtime_error);
  CHECK_THROWS_WITH_AS(b.RunPrefillThenDecode(), doctest::Contains("#1925"),
                       std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// W4b-3c REVIEW REPAIR (#699). The unit the sparse route is decided on is the
// STEP, and until this repair the refusal was decided PER REQUEST. The two
// predicates were therefore different predicates, and the gap between them was
// reachable by the most ordinary shape continuous batching produces:
//
//   the refusal   fired on   sl > topk  AND  computed > 0,  per request
//   the route     died on    ANY request with computed > 0,  for the step
//
// so a batch of {one resumed request at or under `index_topk`, one FRESH
// prompt past it} passed the refusal and took no sparse route: dense attention
// on a sparse model, silently, which is the outcome the refusal exists to
// prevent. Every dots3-note device case before this one is `num_reqs = 1`, so
// nothing in the brick could see it.
namespace w4b3cmix {

// One request of a batch. `seq_len` is the TOTAL number of keys the request
// has — its cached prefix INCLUDED — and `computed` is how many of them were
// produced on an earlier step, which is the discriminator the indexer turns on.
struct Req {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  int32_t seq_len = 0;
  int32_t computed = 0;
};

// A FRESH request of `n` tokens: nothing computed before this step, positions
// from 0, and `seq_len == n`.
Req Fresh(int64_t n, int32_t token_seed) {
  Req r;
  for (int64_t i = 0; i < n; ++i) {
    r.tokens.push_back(static_cast<int32_t>((i * 5 + token_seed) % 12));
    r.positions.push_back(static_cast<int32_t>(i));
  }
  r.seq_len = static_cast<int32_t>(n);
  return r;
}

// A RESUMED request: one query token at absolute position `computed`, over a
// prefix of `computed` keys already in the cache.
Req Resumed(int32_t computed, int32_t token_seed) {
  Req r;
  r.tokens.push_back(token_seed % 12);
  r.positions.push_back(computed);
  r.computed = computed;
  r.seq_len = computed + 1;
  return r;
}

// One paged block per request, so the block table stays one column wide and a
// request's slots are `req_index * block_size + position`.
vllm::v1::CommonAttentionMetadata BatchMeta(const std::vector<Req>& rs) {
  vllm::v1::CommonAttentionMetadata m;
  m.num_reqs = static_cast<int>(rs.size());
  int32_t total = 0;
  m.query_start_loc.push_back(0);
  for (size_t r = 0; r < rs.size(); ++r) {
    total += static_cast<int32_t>(rs[r].tokens.size());
    m.query_start_loc.push_back(total);
    m.seq_lens.push_back(rs[r].seq_len);
    m.num_computed_tokens_cpu.push_back(rs[r].computed);
    m.block_table_tensor.push_back(static_cast<int32_t>(r));
    m.max_query_len =
        std::max(m.max_query_len, static_cast<int>(rs[r].tokens.size()));
    m.max_seq_len = std::max(m.max_seq_len, static_cast<int>(rs[r].seq_len));
    for (const int32_t p : rs[r].positions) {
      m.slot_mapping.push_back(static_cast<int64_t>(r) * w4a::kDeviceBlockSize + p);
    }
  }
  m.num_actual_tokens = total;
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens_cpu = m.seq_lens;
  m.block_table_num_cols = 1;
  m.causal = true;
  return m;
}

std::vector<int32_t> CatTokens(const std::vector<Req>& rs) {
  std::vector<int32_t> v;
  for (const Req& r : rs) v.insert(v.end(), r.tokens.begin(), r.tokens.end());
  return v;
}

std::vector<int32_t> CatPositions(const std::vector<Req>& rs) {
  std::vector<int32_t> v;
  for (const Req& r : rs) v.insert(v.end(), r.positions.begin(), r.positions.end());
  return v;
}

// The reference for a BATCH of independent fresh requests: W3's whole-model
// reference run once PER REQUEST and concatenated, because the requests do not
// see each other. Only valid for requests with no cached prefix — a resumed one
// would need the earlier step's state, which is exactly what this row refuses.
std::vector<double> RefBatch(const w4a::DeviceBench& b, const std::vector<Req>& rs,
                             const ref::Opts& o) {
  std::vector<double> out;
  for (const Req& r : rs) {
    REQUIRE(r.computed == 0);
    const std::vector<double> one =
        w4a::RefModel(b.spec, b.dims, b.w, r.tokens, r.positions, o);
    out.insert(out.end(), one.begin(), one.end());
  }
  return out;
}

}  // namespace w4b3cmix

TEST_CASE(
    "dots3-note W4b-3c: TWO fresh prompts past `index_topk` in ONE step are "
    "each served SPARSELY, and agree with the selecting reference") {
  // `num_reqs = 2`, which no dots3-note device case had before. Both requests
  // are FRESH, so every index key is in hand and the step takes the sparse
  // route — per request, over its own key space, which is what
  // `indexer_cu_seqlens_q` carries.
  const w4a::DeviceBench b(w4b3c::SparseSpec());
  REQUIRE(b.params.index_topk == 2);
  const std::vector<w4b3cmix::Req> rs = {w4b3cmix::Fresh(6, 1), w4b3cmix::Fresh(5, 3)};
  const vllm::v1::CommonAttentionMetadata am = w4b3cmix::BatchMeta(rs);
  REQUIRE(am.num_reqs == 2);
  REQUIRE(am.num_actual_tokens == 11);
  REQUIRE(w4a::Dots3NoteDeviceRefusal(b.params).empty());

  // WHAT THE SELECTION ACTUALLY DOES on this batch, printed before anything is
  // asserted about it: request 0 prunes on rows 2..5 and request 1 on rows
  // 2..4, hand-derived from the causal rule against `index_topk` 2.
  int64_t pruning_rows = 0, keys_dropped = 0;
  for (const w4b3cmix::Req& r : rs) {
    for (int64_t t = b.params.index_topk; t < static_cast<int64_t>(r.tokens.size()); ++t) {
      ++pruning_rows;
      keys_dropped += (t + 1) - b.params.index_topk;
    }
  }
  MESSAGE("W4b-3c mixed-batch selection: " << pruning_rows << " of "
                                           << am.num_actual_tokens
                                           << " rows prune, " << keys_dropped
                                           << " causal keys dropped");
  REQUIRE(pruning_rows == 4 + 3);
  REQUIRE(keys_dropped == (1 + 2 + 3 + 4) + (1 + 2 + 3));

  const std::vector<double> got = b.RunDeviceOn(
      w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs), am, /*num_reqs=*/2,
      b.params.physical_latent_row());
  const Diff d = Compare(got, w4b3cmix::RefBatch(b, rs, ref::Opts{}));
  MESSAGE("W4b-3c two-request sparse device vs the selecting reference: "
          << d.max_rel << " relative; bound " << w4a::kDeviceRel);
  CHECK(d.max_rel < w4a::kDeviceRel);

  // THE CONTROL. `run_indexer = false` is the whole causal window — what a step
  // that took no sparse route computes. The device must be FAR from it, or the
  // agreement above holds for a forward that selected nothing.
  ref::Opts dense;
  dense.run_indexer = false;
  const Diff dd = Compare(got, w4b3cmix::RefBatch(b, rs, dense));
  MESSAGE("W4b-3c two-request against the NON-selecting reference: "
          << dd.max_rel << " relative, against " << d.max_rel << " with the selection");
  CHECK(dd.max_rel > w4a::kDeviceRel);
  CHECK(dd.max_rel > 10.0 * d.max_rel);
}

TEST_CASE(
    "dots3-note W4b-3c: a MIXED step — a RESUMED request beside a fresh prompt "
    "past `index_topk` — REFUSES rather than silently serving dense") {
  // THE DEFECT THIS CASE EXISTS FOR. The sparse route dies for the WHOLE step
  // as soon as any request resumes, because the indexer's key space is the
  // step's own tokens. The refusal has to be the complement of that, or the
  // shape below is served with no selection at all and no message: the fresh
  // request's rows past `index_topk` get the full causal window. MEASURED on
  // the pre-repair code with exactly this batch, comparing request 1's own
  // logit rows against BOTH references — 0.880079 relative from the SELECTING
  // one (17.6x outside the 0.05 bound) and 0.0103532 from the NON-selecting
  // one (inside it). The device agreed with the reference that selects
  // nothing, which is this file's own gate inverted.
  const w4a::DeviceBench b(w4b3c::SparseSpec());
  REQUIRE(w4a::Dots3NoteDenseEquivalentMaxSeqLen(b.params) == 2);
  REQUIRE(w4a::Dots3NoteDeviceRefusal(b.params).empty());  // the CONFIG is fine

  // Decode tokens are PACKED FIRST, which is the seam's own convention
  // (mla_attention.py:700-709). The resumed request is at or UNDER `index_topk`
  // and so passes the per-request refusal on its own; the fresh one is past it
  // and so needs a selection nothing in this step can produce for its
  // co-batched neighbour.
  const std::vector<w4b3cmix::Req> rs = {w4b3cmix::Resumed(1, 7), w4b3cmix::Fresh(5, 3)};
  REQUIRE(rs[0].seq_len <= b.params.index_topk);
  REQUIRE(rs[1].seq_len > b.params.index_topk);
  const vllm::v1::CommonAttentionMetadata am = w4b3cmix::BatchMeta(rs);
  CHECK_THROWS_WITH_AS(
      b.RunDeviceOn(w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs), am,
                    /*num_reqs=*/2, b.params.physical_latent_row()),
      doctest::Contains("index_topk"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      b.RunDeviceOn(w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs), am,
                    /*num_reqs=*/2, b.params.physical_latent_row()),
      doctest::Contains("#1925"), std::runtime_error);

  // THE CONTROL, and it is what stops this case passing on "any two-request
  // step refuses". Shorten the fresh request to `index_topk` keys and NOTHING
  // in the step prunes, so the top-k selects every causal candidate, dense
  // attention IS upstream's answer (`use_dense_mha`,
  // sparse_mla_attention.py:296-299) and the same resumed request runs.
  const std::vector<w4b3cmix::Req> ok = {w4b3cmix::Resumed(1, 7), w4b3cmix::Fresh(2, 3)};
  REQUIRE(ok[1].seq_len <= b.params.index_topk);
  CHECK_NOTHROW((void)b.RunDeviceOn(w4b3cmix::CatTokens(ok), w4b3cmix::CatPositions(ok),
                                    w4b3cmix::BatchMeta(ok), /*num_reqs=*/2,
                                    b.params.physical_latent_row()));

  // And the SECOND control: two FRESH requests, one of them past `index_topk`,
  // still run — the refusal turns on the CACHED PREFIX, not on the batch size.
  const std::vector<w4b3cmix::Req> fresh = {w4b3cmix::Fresh(6, 1), w4b3cmix::Fresh(5, 3)};
  CHECK_NOTHROW((void)b.RunDeviceOn(w4b3cmix::CatTokens(fresh),
                                    w4b3cmix::CatPositions(fresh),
                                    w4b3cmix::BatchMeta(fresh), /*num_reqs=*/2,
                                    b.params.physical_latent_row()));
}

TEST_CASE(
    "dots3-note W4b-3c: a step whose metadata is NOT SHAPED the way the sparse "
    "route reads it REFUSES by name, and says which half of the refusal fired") {
  // THE OTHER ARM, which nothing measured. The repair made the refusal the
  // exact complement of `Dots3NoteSparseEligibility::Active`, so it fires on
  // `prunes && (resumes || !well_formed)`. The `resumes` half is held by the
  // mixed-batch case above. The `!well_formed` half was held by nothing: the
  // message's ternary could lose its false arm entirely and every gate in this
  // tree stayed green — a predicate nothing measures, which is the shape of the
  // defect the whole brick spent two review rounds repairing.
  //
  // `well_formed` is `index_topk > 0 && num_reqs > 0 &&
  // query_start_loc.size() == num_reqs + 1 && seq_lens.size() >= num_reqs`. A
  // `query_start_loc` one entry too long is the cheapest violation and the one
  // that matters: `BuildDots3NoteSparseStep` reads exactly `num_reqs + 1`
  // entries off it to build `cu`, so a step it cannot read is a step nothing
  // may serve — and a step that PRUNES and is not eligible has no dense answer
  // to fall back to.
  const w4a::DeviceBench b(w4b3c::SparseSpec());
  REQUIRE(w4a::Dots3NoteDenseEquivalentMaxSeqLen(b.params) == 2);
  REQUIRE(w4a::Dots3NoteDeviceRefusal(b.params).empty());  // the CONFIG is fine

  // ONE FRESH request past `index_topk`. Nothing resumes, so the `#1925` arm of
  // the message cannot be what fires and the discriminator is unambiguous.
  const std::vector<w4b3cmix::Req> rs = {w4b3cmix::Fresh(5, 3)};
  REQUIRE(rs[0].computed == 0);
  REQUIRE(static_cast<int64_t>(rs[0].seq_len) > b.params.index_topk);

  // THE CONTROL, and it comes first. The very same request on WELL-FORMED
  // metadata takes the sparse route and runs, so whatever the refusal below
  // measures is the SHAPE of the metadata and not this request.
  const vllm::v1::CommonAttentionMetadata good = w4b3cmix::BatchMeta(rs);
  REQUIRE(good.query_start_loc.size() == static_cast<size_t>(good.num_reqs) + 1);
  CHECK_NOTHROW((void)b.RunDeviceOn(w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs),
                                    good, /*num_reqs=*/1,
                                    b.params.physical_latent_row()));

  // One entry too long, and NOTHING else changed.
  vllm::v1::CommonAttentionMetadata bad = good;
  bad.query_start_loc.push_back(bad.query_start_loc.back());
  bad.query_start_loc_cpu = bad.query_start_loc;
  REQUIRE(bad.query_start_loc.size() != static_cast<size_t>(bad.num_reqs) + 1);

  // It refuses; it names `index_topk`, because the step really does prune; and
  // it names the MALFORMATION rather than the resumed-request arm, which is the
  // half of the message that had no gate.
  CHECK_THROWS_WITH_AS(
      b.RunDeviceOn(w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs), bad,
                    /*num_reqs=*/1, b.params.physical_latent_row()),
      doctest::Contains("index_topk"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      b.RunDeviceOn(w4b3cmix::CatTokens(rs), w4b3cmix::CatPositions(rs), bad,
                    /*num_reqs=*/1, b.params.physical_latent_row()),
      doctest::Contains("not shaped"), std::runtime_error);
}

// ═════════════════════════════════════════════════════════════════════════════
// W5 — the MoE layer ON THE DECODE PATH, and the RELEASED config stops refusing.
//
// ─── WHAT THIS ESTABLISHES, AND WHAT IT CANNOT ───────────────────────────────
// A config whose layers are `{full+dense, sliding+MoE, full+MoE}` is loaded
// through the REAL registry over the REAL loader and run through
// `ModelRegistry::Forward` TWICE against one KV-cache pool — a PREFILL then a
// DECODE — and its logits are compared against a whole-model double reference
// whose MoE block was transcribed from `grouped_topk_router.py:80-161`,
// `deepseek_v2.py:406-429` and `nvidia/model.py:115-132` at `bc2d63e650`, and
// WITHOUT reading `src/vt/cpu/cpu_ops.cpp`. A reference that shares a helper
// with the code it gates measures consistency with itself.
//
// It CANNOT say the answer matches vLLM. vLLM cannot run this model on any
// hardware this project owns (spec §6.2), so there is no oracle and no token
// gate anywhere on this row. Two implementations of one formula agreeing is
// what is claimed, and nothing more (spec §6.4, option B).
//
// ─── WHY THE DISCRETE ASSERTION IS THE POINT ─────────────────────────────────
// Router logits are stored bf16, whose maximum relative rounding error is
// 2^-8 = 3.906e-3. Through `sigmoid' <= 0.25` that is a SCORE perturbation of
// order 1e-3. At the released E=256 the typical gap between the 8th and 9th
// order statistics of a uniform score spread is about 1/256 = 3.9e-3 — the same
// order — so a fixture sampled from noise is a coin flip on whether the
// SELECTED SET matches, and a relative-error bound cannot tell "the same 8
// experts, rounded" from "a different expert entirely". §4.9 records that exact
// defect one brick ago on the DSA indexer (7.43e-4 margin against a 1.28e-3
// ulp), and the repair there was the FIXTURE and never the threshold.
//
// So the fixture is DESIGNED rather than sampled, and the case prints its own
// discriminators: the minimum decision margin against the bf16 score ulp, the
// number of distinct experts the batch actually activates, and the tie.
// ═════════════════════════════════════════════════════════════════════════════
namespace {
namespace w5 {

using vllm::Dots3NoteDeviceRefusal;
using vllm::Dots3NoteLayerKind;
using vllm::PagedKvCache;
using vllm::dots3_note::Dots3NoteSlidingAttnDimsFrom;
using w4a::Bf16All;
using w4a::StOut;

// The stored width of the correction bias, which is the one F32 tensor in this
// tower. Rounding the designed values through float here is what makes the
// reference read the same number the checkpoint stores.
double F32(double x) { return static_cast<double>(static_cast<float>(x)); }
std::vector<double> F32All(std::vector<double> v) {
  for (double& x : v) x = F32(x);
  return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// THE FIXTURE, and every number in it is a decision.
//
// The ATTENTION geometry is W4b-2's, unchanged and reused rather than retyped:
// `{full, sliding, full}` with a 3-wide window against a 6-token prompt, a
// padded physical row, two rope thetas orders apart and a SHUFFLED block table.
// W5 changes the MLP and nothing else, so re-deriving the attention fixture
// would only risk weakening it.
//
// The MoE half:
//   * `first_k_dense_replace = 1`, so layer 0 is DENSE and layers 1-2 are MoE —
//     the released schedule's own shape (`first_k_dense_replace: 1` in the
//     committed config.json). A fixture with no dense layer would not test that
//     `MaterializeDots3NoteDevice` still reads `mlp.gate_proj.weight` where it
//     exists, and one with no MoE layer would test nothing at all.
//   * The MoE layers are one of EACH attention kind (layer 1 sliding, layer 2
//     full), so a MoE block that leaked a per-layer field across kinds is wrong.
//   * `E = 8`, `top_k = 3`. Small enough to enumerate by hand in a failure
//     message, large enough that 8-choose-3 is not a formality.
//   * THE BIAS IS THE SELECTION, and it is designed in three tiers:
//       - experts 0 and 1 carry bias +1.30 and BYTE-IDENTICAL router rows, so
//         their biased scores tie EXACTLY in double for every token and BOTH
//         are always selected. That is the deliberate exact tie, and it sits
//         OFF the k-th boundary on purpose: with both inside the set the
//         selection is unambiguous whatever the tie rule is, which is the only
//         claim upstream's `torch.topk(..., sorted=False)` actually supports
//         (grouped_topk_router.py:134 defines `use_sorted`; :148 is the
//         EXPERT topk that reads it). A tie ON the boundary would make
//         the correct answer genuinely undefined and gating it would pin our
//         kernel's accident as a contract.
//       - experts 2-5 carry bias 0.0, so the LOGITS decide the third slot and
//         the selection is genuinely token-dependent. The case prints how many
//         distinct experts the batch activates; a fixture where every token
//         picks the same three has not tested routing at all.
//       - experts 6 and 7 carry bias -1.50 and are never selected.
//     The selected experts therefore carry DIFFERENT biases (+1.30, +1.30, 0.0),
//     which is what makes the nearest mechanism — the bias applied to the
//     routing WEIGHT as well as to the selection — visible rather than absorbed
//     by the renormalisation.
//   * `router_amp` keeps the logits small, so the sigmoid scores sit in a band
//     narrow enough for the +1.30 / 0.0 / -1.50 tiers to dominate, while still
//     spreading enough inside the middle tier for the third slot to move. The
//     case MEASURES the resulting margin rather than trusting this paragraph.
//   * `moe_inter = 6` against the dense layers' `inter = 10`, and
//     `n_shared_experts = 1`, so the shared expert's width is
//     `moe_intermediate_size * n_shared_experts` = 6 and NOT `intermediate_size`
//     = 10. A port that reached for the wrong one builds a 10-wide MLP and the
//     load refuses BY NAME — which has its own case.
// ─────────────────────────────────────────────────────────────────────────────
struct Spec {
  w4b2::Spec attn;  // {full, sliding, full}, prompt 6, page 4, shuffled table
  int64_t n_experts = 8;
  int64_t top_k = 3;
  int64_t moe_inter = 6;
  int64_t n_shared = 1;
  int64_t first_k_dense = 1;
  double routed_scaling_factor = 1.0;
  bool norm_topk_prob = true;
  // CHOSEN BY MEASUREMENT over {0.09, 0.18, 0.30, 0.45, 0.60}, and the
  // discriminator was the SOFTMAX arm rather than the one this brick expected.
  // At 0.09 the sigmoid-versus-softmax defect moved the logits by only 0.0400 —
  // BELOW the 0.06 bound, so a port that wrote `kSoftmax` into the router args
  // would have passed. The scoring functions only separate once the logits are
  // large enough to leave sigmoid's near-linear region, and the arm scales
  // 0.0400 / 0.0770 / 0.1126 / 0.1515 / 0.2080 across that grid while the
  // minimum decision margin barely moves (19.4x / 19.3x / 18.7x / 18.1x /
  // 17.2x the ulp) and the residue FALLS (0.0230 -> 0.0119). 0.45 is the
  // SMALLEST value that clears the 0.15 fixture-quality floor at all, and it
  // clears it by one percent — a guard met by one percent is the hugged
  // threshold this project keeps naming, so the fixture takes 0.60 and the
  // nearest mechanism lands at 0.208 instead.
  double router_amp = 0.60;
  double expert_amp = 0.5;
  double shared_amp = 0.5;
  // A per-expert output GAIN of 3 on the CONTENDED tier (experts 2-5), and it
  // is what makes the nearest mechanism visible rather than a decoration. The
  // bias-in-the-weight defect moves routing weight AWAY from the contended
  // expert (0.333 -> 0.122 at this fixture's scores) and TOWARDS the two
  // always-selected ones, so unless the three differ in OUTPUT magnitude the
  // reshuffle largely cancels in the sum. MEASURED: at gain 1 the defect moves
  // the logits by 0.070-0.226 depending on the seed and at gain 3 by
  // 0.262-0.712. The first draft of this fixture had gain 1 and a different
  // seed, and the defect landed at 0.0183 — BELOW the 0.06 bound, i.e. a gate
  // that would have passed the single most likely port defect on this block.
  // The repair was the fixture, never the bound.
  double contended_gain = 3.0;
  // CHOSEN BY MEASUREMENT over six seeds x {gain 1, 3} x {shared amplitude
  // 0.5, 0.15}, on the two discriminators this gate turns on: the minimum
  // decision margin against the bf16 score ulp, and the distance to the nearest
  // mechanism. This one reads margin 19.4x the ulp (bar 4x), 6 distinct experts
  // activated, residue 0.0230 and nearest mechanism 0.3214. The scan's own
  // spread is the argument for measuring rather than assuming: the margin
  // ranged 1.51x to 25.8x across the grid, so one seed in twelve would have
  // shipped a fixture whose selection is a coin flip.
  uint64_t moe_seed = 0x1234567890ABCDEFULL;
  // The three tiers. F32 because that is how the checkpoint stores them.
  std::vector<double> bias{1.30, 1.30, 0.0, 0.0, 0.0, 0.0, -1.50, -1.50};
  int64_t layers() const { return attn.layers(); }
  bool is_moe(int64_t l) const { return l >= first_k_dense; }
  int64_t shared_inter() const { return moe_inter * n_shared; }
};

nlohmann::json ConfigDoc(const Spec& s) {
  nlohmann::json d = w4b2::ConfigDoc(s.attn);
  d["n_routed_experts"] = s.n_experts;
  d["num_experts_per_tok"] = s.top_k;
  d["moe_intermediate_size"] = s.moe_inter;
  d["n_shared_experts"] = s.n_shared;
  d["first_k_dense_replace"] = s.first_k_dense;
  d["moe_layer_freq"] = 1;
  d["norm_topk_prob"] = s.norm_topk_prob;
  d["routed_scaling_factor"] = s.routed_scaling_factor;
  // The released config.json carries NEITHER key (§4 trap 1) and
  // `ParseDots3NoteParams` defaults both to 1, which is what
  // `Dots3NoteConfig.__init__` does at configs/dots3_note.py:18-19. Left ABSENT
  // here for the same reason: writing them would test a config the publisher
  // does not ship.
  d.erase("n_group");
  d.erase("topk_group");
  return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// One MoE layer's weights, in the on-disk orientation, in double.
struct MoeW {
  std::vector<double> router;                  // [E, H]
  std::vector<double> bias;                    // [E], F32-rounded
  std::vector<std::vector<double>> egate;      // E x [I, H]
  std::vector<std::vector<double>> eup;        // E x [I, H]
  std::vector<std::vector<double>> edown;      // E x [H, I]
  std::vector<double> sgate, sup, sdown;       // shared, at [SI,H]/[SI,H]/[H,SI]
};

// W5 REUSES W4b-2's WEIGHTS WHOLESALE and adds one thing. `w4b2::MakeWeights`
// builds the embedding, the lm_head, the norms and BOTH attention arms from the
// same generator at the same seed, so W5's attention fixture is byte-for-byte
// W4b-2's rather than a re-derivation of it — including the dense gate/up/down
// on the MoE layers, which are generated and then not written, precisely so the
// RNG stream does not shift and the attention weights stay identical.
struct Weights {
  w4b2::Weights base;
  std::vector<MoeW> moe;  // one per layer; only the MoE layers' are populated
};

MoeW MakeMoe(const Spec& s, Rng& r) {
  const int64_t H = s.attn.hidden, E = s.n_experts, I = s.moe_inter;
  const int64_t SI = s.shared_inter();
  MoeW m;
  m.router = Bf16All(r.fill(E * H, s.router_amp));
  // THE EXACT TIE. Expert 1's router row is expert 0's, byte for byte, and
  // their biases are equal — so `sigmoid(logit) + bias` is bit-identical for
  // the two at every token and the tie is real rather than approximate. Both
  // sit inside the selected set (tier +1.30 against top_k 3), so the SET does
  // not depend on which side a tie-break picks, which is the only claim
  // upstream's `sorted=False` supports.
  for (int64_t c = 0; c < H; ++c) {
    m.router[static_cast<size_t>(1 * H + c)] = m.router[static_cast<size_t>(c)];
  }
  m.bias = F32All(s.bias);
  for (int64_t e = 0; e < E; ++e) {
    const double gain = e >= 2 ? s.contended_gain : 1.0;
    m.egate.push_back(Bf16All(r.fill(I * H, s.expert_amp)));
    m.eup.push_back(Bf16All(r.fill(I * H, s.expert_amp)));
    m.edown.push_back(Bf16All(r.fill(H * I, s.expert_amp * gain)));
  }
  m.sgate = Bf16All(r.fill(SI * H, s.shared_amp));
  m.sup = Bf16All(r.fill(SI * H, s.shared_amp));
  m.sdown = Bf16All(r.fill(H * SI, s.shared_amp));
  return m;
}

Weights MakeWeights(const Spec& s, const FullAttnDims& fd, const SlidingAttnDims& sd,
                    uint64_t seed) {
  Weights w;
  w.base = w4b2::MakeWeights(s.attn, fd, sd, seed);
  Rng r(s.moe_seed);
  for (int64_t l = 0; l < s.layers(); ++l) {
    w.moe.push_back(s.is_moe(l) ? MakeMoe(s, r) : MoeW{});
  }
  return w;
}

// W4b-2's checkpoint with the DENSE MLP entries of each MoE layer replaced by
// the MoE ones. Filtering rather than re-emitting keeps every attention tensor
// byte-identical to the fixture W4b-2 gated, and it is what makes the two
// bricks' benches comparable.
std::vector<StOut> CheckpointOf(const Spec& s, const FullAttnDims& fd,
                                const SlidingAttnDims& sd, const Weights& w) {
  const int64_t H = s.attn.hidden;
  const std::vector<StOut> dense = w4b2::CheckpointOf(s.attn, fd, sd, w.base);
  std::vector<StOut> e;
  for (const StOut& t : dense) {
    bool drop = false;
    for (int64_t l = 0; l < s.layers(); ++l) {
      if (!s.is_moe(l)) continue;
      const std::string p = "model.layers." + std::to_string(l) + ".mlp.";
      if (t.name.rfind(p, 0) == 0) drop = true;
    }
    if (!drop) e.push_back(t);
  }
  for (int64_t l = 0; l < s.layers(); ++l) {
    if (!s.is_moe(l)) continue;
    const MoeW& m = w.moe[static_cast<size_t>(l)];
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const int64_t E = s.n_experts, I = s.moe_inter, SI = s.shared_inter();
    e.push_back({p + "mlp.gate.weight", {E, H}, m.router});
    // F32, which is the ONE dtype exception in this tower and is upstream's own
    // (`torch.empty(n_routed_experts, dtype=torch.float32)`,
    // deepseek_v2.py:322-324). A fixture that stored it bf16 would be testing a
    // checkpoint the publisher does not ship.
    e.push_back({p + "mlp.gate.e_score_correction_bias", {E}, m.bias, "F32"});
    for (int64_t x = 0; x < E; ++x) {
      const std::string ex = p + "mlp.experts." + std::to_string(x) + ".";
      e.push_back({ex + "gate_proj.weight", {I, H}, m.egate[static_cast<size_t>(x)]});
      e.push_back({ex + "up_proj.weight", {I, H}, m.eup[static_cast<size_t>(x)]});
      e.push_back({ex + "down_proj.weight", {H, I}, m.edown[static_cast<size_t>(x)]});
    }
    e.push_back({p + "mlp.shared_experts.gate_proj.weight", {SI, H}, m.sgate});
    e.push_back({p + "mlp.shared_experts.up_proj.weight", {SI, H}, m.sup});
    e.push_back({p + "mlp.shared_experts.down_proj.weight", {H, SI}, m.sdown});
  }
  return e;
}

// ─────────────────────────────────────────────────────────────────────────────
// THE INDEPENDENT DOUBLE REFERENCE for the MoE block.
//
// Transcribed from the upstream Python and from nothing else:
//   grouped_topk_router.py:112-161   scoring, the correction bias, the group
//                                    stage, the top-k, the routing weights,
//                                    renormalise, routed_scaling_factor
//   deepseek_v2.py:406-429           `DeepseekV2MoE.forward` (routed only here,
//                                    because model.py:88-90 sets
//                                    `n_shared_experts` to None on the routed
//                                    config so deepseek_v2.py:354-355 leaves
//                                    `self.shared_experts = None`)
//   nvidia/model.py:115-132          `Dots3NoteMoE.forward` — the unfused
//                                    `+ self.shared_experts(hidden_states)`
//
// `src/vt/cpu/cpu_ops.cpp` was NOT read while writing it. Every switch on
// `Opts` is a MUTATION HANDLE: the case flips one and requires the answer to
// move, which is what turns "the two agree" into "the two agree BECAUSE the
// mechanism is there".
struct Opts {
  bool bias_in_selection = true;   // grouped_topk_router.py:120-124
  bool bias_in_weights = false;    // the NEAREST MECHANISM; :148-150 says NO
  bool renormalize = true;         // :156-157
  bool shared = true;              // model.py:125-127
  bool group_stage = true;         // :125-145 — INERT at n_group == 1
  bool softmax_scoring = false;    // :112-117 — sigmoid for this architecture
  double routed_scale = 1.0;       // :159-160
  int top_k_delta = 0;
};

// What the router decided for ONE token, kept so the case can assert the
// DISCRETE half rather than only a norm.
struct Decision {
  std::vector<int> selected;   // in the order the reference produced them
  std::vector<double> weight;  // the routing weight per slot
  double margin = 0.0;         // biased[k-th selected] - biased[best rejected]
  double tie_gap = 0.0;        // |biased[0] - biased[1]|, the deliberate tie
  double max_abs_logit = 0.0;
};

std::vector<double> Silu(const std::vector<double>& g, const std::vector<double>& u) {
  std::vector<double> a(g.size());
  for (size_t i = 0; i < g.size(); ++i) a[i] = (g[i] / (1.0 + std::exp(-g[i]))) * u[i];
  return a;
}

// One expert MLP over one row: `down(silu(gate(x)) * up(x))`.
std::vector<double> ExpertMlp(const std::vector<double>& x, const std::vector<double>& wg,
                              const std::vector<double>& wu,
                              const std::vector<double>& wd, int64_t H, int64_t I) {
  const std::vector<double> g = ref::Dense(x, wg, 1, H, I);
  const std::vector<double> u = ref::Dense(x, wu, 1, H, I);
  return ref::Dense(Silu(g, u), wd, 1, I, H);
}

std::vector<double> RefMoe(const Spec& s, const MoeW& w, const std::vector<double>& x,
                           int64_t T, const Opts& o,
                           std::vector<Decision>* trace = nullptr) {
  const int64_t H = s.attn.hidden, E = s.n_experts, I = s.moe_inter;
  const int64_t K = s.top_k + o.top_k_delta;
  const int64_t SI = s.shared_inter();
  REQUIRE(K >= 1);
  REQUIRE(K <= E);
  // `router_logits, _ = self.gate(hidden_states)` — a plain linear at the model
  // dtype, because `_get_moe_router_dtype` returns None for this model type
  // (deepseek_v2.py:131-141). The reference computes it in long double; the
  // WEIGHTS it reads are already bf16-rounded, which is what makes the
  // comparison measure the forward rather than the storage width.
  const std::vector<double> logits = ref::Dense(x, w.router, T, H, E);
  std::vector<double> out(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    Decision dec;
    std::vector<double> scores(static_cast<size_t>(E));
    if (o.softmax_scoring) {
      double m = -1e300;
      for (int64_t e = 0; e < E; ++e) {
        m = std::max(m, logits[static_cast<size_t>(t * E + e)]);
      }
      double z = 0.0;
      for (int64_t e = 0; e < E; ++e) {
        scores[static_cast<size_t>(e)] = std::exp(logits[static_cast<size_t>(t * E + e)] - m);
        z += scores[static_cast<size_t>(e)];
      }
      for (double& v : scores) v /= z;
    } else {
      for (int64_t e = 0; e < E; ++e) {
        scores[static_cast<size_t>(e)] =
            1.0 / (1.0 + std::exp(-logits[static_cast<size_t>(t * E + e)]));
      }
    }
    for (int64_t e = 0; e < E; ++e) {
      dec.max_abs_logit =
          std::max(dec.max_abs_logit, std::fabs(logits[static_cast<size_t>(t * E + e)]));
    }
    // `scores = scores + e_score_correction_bias.unsqueeze(0)` (:122). Used for
    // SELECTION; the routing weights read `original_scores` (:148-150).
    std::vector<double> biased = scores;
    if (o.bias_in_selection) {
      for (int64_t e = 0; e < E; ++e) biased[static_cast<size_t>(e)] += w.bias[static_cast<size_t>(e)];
    }
    // The GROUP stage (:125-145). At `n_group == 1` there is one group, the
    // `topk_group == 1` selection keeps it, and the mask is all-ones — so this
    // is written out and then is the identity. It is modelled explicitly rather
    // than skipped so the PREDICTED-GREEN mutation has a real thing to delete
    // and the inertness is demonstrated instead of asserted in prose.
    std::vector<bool> mask(static_cast<size_t>(E), true);
    if (o.group_stage) {
      const int64_t G = 1, TG = 1, per = E / G;
      std::vector<double> gs(static_cast<size_t>(G), 0.0);
      for (int64_t g = 0; g < G; ++g) {
        // `.topk(2, dim=-1)[0].sum(dim=-1)` over the group (:125-127)
        double a = -1e300, b = -1e300;
        for (int64_t j = 0; j < per; ++j) {
          const double v = biased[static_cast<size_t>(g * per + j)];
          if (v > a) { b = a; a = v; } else if (v > b) { b = v; }
        }
        gs[static_cast<size_t>(g)] = a + b;
      }
      std::vector<int64_t> order(static_cast<size_t>(G));
      for (int64_t g = 0; g < G; ++g) order[static_cast<size_t>(g)] = g;
      std::stable_sort(order.begin(), order.end(), [&](int64_t p, int64_t q) {
        return gs[static_cast<size_t>(p)] > gs[static_cast<size_t>(q)];
      });
      std::fill(mask.begin(), mask.end(), false);
      for (int64_t g = 0; g < TG; ++g) {
        const int64_t gi = order[static_cast<size_t>(g)];
        for (int64_t j = 0; j < per; ++j) mask[static_cast<size_t>(gi * per + j)] = true;
      }
    }
    // `topk_ids = torch.topk(tmp_scores, k=topk, ...)[1]` (:148). STABLE, so a
    // tie resolves to the lower index; the fixture keeps its deliberate tie OFF
    // the k-th boundary so the resulting SET does not depend on that choice.
    std::vector<int> order(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) order[static_cast<size_t>(e)] = static_cast<int>(e);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      const double va = mask[static_cast<size_t>(a)] ? biased[static_cast<size_t>(a)] : -1e300;
      const double vb = mask[static_cast<size_t>(b)] ? biased[static_cast<size_t>(b)] : -1e300;
      return va > vb;
    });
    dec.selected.assign(order.begin(), order.begin() + static_cast<long>(K));
    dec.margin = biased[static_cast<size_t>(order[static_cast<size_t>(K - 1)])] -
                 biased[static_cast<size_t>(order[static_cast<size_t>(K)])];
    dec.tie_gap = std::fabs(biased[0] - biased[1]);
    // `topk_weights = original_scores.gather(1, topk_ids)` (:150) — the
    // UNBIASED scores. `bias_in_weights` is the port defect this gate is tuned
    // against.
    double z = 0.0;
    for (int64_t j = 0; j < K; ++j) {
      const int e = dec.selected[static_cast<size_t>(j)];
      const double v = o.bias_in_weights ? biased[static_cast<size_t>(e)]
                                         : scores[static_cast<size_t>(e)];
      dec.weight.push_back(v);
      z += v;
    }
    if (o.renormalize && z != 0.0) {
      for (double& v : dec.weight) v /= z;
    }
    if (o.routed_scale != 1.0) {
      for (double& v : dec.weight) v *= o.routed_scale;
    }
    const std::vector<double> xt(x.begin() + static_cast<long>(t * H),
                                 x.begin() + static_cast<long>((t + 1) * H));
    for (int64_t j = 0; j < K; ++j) {
      const size_t e = static_cast<size_t>(dec.selected[static_cast<size_t>(j)]);
      const std::vector<double> y =
          ExpertMlp(xt, w.egate[e], w.eup[e], w.edown[e], H, I);
      for (int64_t c = 0; c < H; ++c) {
        out[static_cast<size_t>(t * H + c)] += dec.weight[static_cast<size_t>(j)] * y[static_cast<size_t>(c)];
      }
    }
    // `+ self.shared_experts(hidden_states)` (model.py:127) — a PLAIN MLP whose
    // output is ADDED, no gate.
    if (o.shared) {
      const std::vector<double> sh = ExpertMlp(xt, w.sgate, w.sup, w.sdown, H, SI);
      for (int64_t c = 0; c < H; ++c) out[static_cast<size_t>(t * H + c)] += sh[static_cast<size_t>(c)];
    }
    if (trace != nullptr) trace->push_back(std::move(dec));
  }
  return out;
}

// The whole-model reference: W4b-2's residual stream and attention dispatch,
// with the MLP dispatching on `is_moe`.
std::vector<double> RefModel(const Spec& s, const FullAttnDims& fd,
                             const SlidingAttnDims& sd, const Weights& w,
                             const std::vector<int32_t>& tokens,
                             const std::vector<int32_t>& positions, const Opts& mo,
                             const ref::Opts& fo = ref::Opts{},
                             const w4b::sref::Opts& so = w4b::sref::Opts{},
                             std::vector<Decision>* trace = nullptr,
                             std::vector<std::vector<double>>* moe_inputs = nullptr) {
  const int64_t T = static_cast<int64_t>(tokens.size()), H = s.attn.hidden;
  std::vector<double> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < H; ++c) {
      hidden[static_cast<size_t>(t * H + c)] =
          w.base.embed[static_cast<size_t>(tokens[static_cast<size_t>(t)] * H + c)];
    }
  }
  std::vector<double> res(static_cast<size_t>(T * H), 0.0);
  for (int64_t l = 0; l < s.layers(); ++l) {
    const w4b2::Weights::Layer& lw = w.base.layers[static_cast<size_t>(l)];
    for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
    const std::vector<double> x = ref::Rms(res, lw.input_ln, T, H, s.attn.rms_eps);
    const std::vector<double> a =
        lw.kind == Dots3NoteLayerKind::kSlidingAttention
            ? w4b::sref::Forward(sd, lw.swa, x, positions, T, so).out
            : ref::Forward(fd, lw.full, x, positions, T, fo).out;
    for (size_t i = 0; i < res.size(); ++i) res[i] += a[i];
    const std::vector<double> y = ref::Rms(res, lw.post_ln, T, H, s.attn.rms_eps);
    if (s.is_moe(l)) {
      if (moe_inputs != nullptr) moe_inputs->push_back(y);
      hidden = RefMoe(s, w.moe[static_cast<size_t>(l)], y, T, mo, trace);
    } else {
      const std::vector<double> g = ref::Dense(y, lw.gate_proj, T, H, s.attn.inter);
      const std::vector<double> u = ref::Dense(y, lw.up_proj, T, H, s.attn.inter);
      hidden = ref::Dense(Silu(g, u), lw.down_proj, T, s.attn.inter, H);
    }
  }
  for (size_t i = 0; i < res.size(); ++i) res[i] += hidden[i];
  const std::vector<double> z = ref::Rms(res, w.base.final_norm, T, H,
                                         s.attn.rms_eps);
  return ref::Dense(z, w.base.lm_head, T, H, s.attn.vocab);
}

// ─────────────────────────────────────────────────────────────────────────────
struct Bench {
  Spec spec;
  TempConfig cfg;
  HfConfig config;
  Dots3NoteParams params;
  FullAttnDims fdims;
  SlidingAttnDims sdims;
  Weights w;
  std::vector<StOut> entries;
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> block_table{1, 0};  // SHUFFLED, as W4b-2's

  explicit Bench(Spec s = Spec{})
      : spec(s),
        cfg(ConfigDoc(s)),
        config(LoadHfConfig(cfg.path())),
        params(ParseDots3NoteParams(config)),
        fdims(w4b2::FullDimsOrDefault(params)),
        sdims(Dots3NoteSlidingAttnDimsFrom(params)),
        w(MakeWeights(s, fdims, sdims, 0x243F6A8885A308D3ULL)),
        entries(CheckpointOf(s, fdims, sdims, w)) {
    for (int64_t t = 0; t <= spec.attn.prompt; ++t) {
      tokens.push_back(static_cast<int32_t>((t * 5 + 1) % spec.attn.vocab));
      positions.push_back(static_cast<int32_t>(t));
    }
  }

  // PREFILL then DECODE, both through `ModelRegistry::Forward`, against ONE
  // cache pool — the production entry point, entered at `ModelRegistry::Resolve`
  // over a real `SafetensorsFile`. Returns the decode step's [1, vocab] logits.
  std::vector<double> RunPrefillThenDecode() const {
    const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
    w4a::TempCheckpoint ckpt(entries);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    std::unique_ptr<vllm::LoadedModel> model =
        reg.factory->load_weights(reg, config, source);
    REQUIRE(model != nullptr);

    w4a::MlaCachePool pool(spec.layers(), params.physical_latent_row(),
                           /*num_blocks=*/2, spec.attn.page_size);
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const std::vector<int32_t> no_gather;
    std::vector<vllm::GdnStateCache> gdn_state;
    vllm::v1::GDNAttentionMetadata gdn_meta{};

    {
      vllm::v1::CommonAttentionMetadata m;
      m.num_reqs = 1;
      m.num_actual_tokens = static_cast<int>(spec.attn.prompt);
      m.query_start_loc = {0, static_cast<int32_t>(spec.attn.prompt)};
      m.query_start_loc_cpu = m.query_start_loc;
      m.seq_lens = {static_cast<int32_t>(spec.attn.prompt)};
      m.seq_lens_cpu = m.seq_lens;
      m.max_query_len = static_cast<int>(spec.attn.prompt);
      m.max_seq_len = static_cast<int>(spec.attn.prompt);
      m.block_table_num_cols = static_cast<int>(block_table.size());
      m.block_table_tensor = block_table;
      for (int64_t t = 0; t < spec.attn.prompt; ++t) {
        m.slot_mapping.push_back(w4b2::SlotOf(spec.attn, block_table, t));
      }
      m.causal = true;
      const std::vector<int32_t> ids(tokens.begin(),
                                     tokens.begin() + spec.attn.prompt);
      const std::vector<int32_t> pos(positions.begin(),
                                     positions.begin() + spec.attn.prompt);
      const vllm::ModelForwardInput in{.token_ids = ids,
                                       .positions = pos,
                                       .attn_meta = m,
                                       .gdn_meta = gdn_meta,
                                       .attn_kv = pool.attn_kv,
                                       .gdn_state = gdn_state,
                                       .config = config,
                                       .queue = queue,
                                       .logits_indices = no_gather,
                                       .num_reqs = 1};
      const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);
      REQUIRE(fl.on_device());
      REQUIRE(fl.rows == spec.attn.prompt);
    }

    std::vector<double> out;
    {
      vllm::v1::CommonAttentionMetadata m;
      m.num_reqs = 1;
      m.num_actual_tokens = 1;
      m.query_start_loc = {0, 1};
      m.query_start_loc_cpu = m.query_start_loc;
      m.seq_lens = {static_cast<int32_t>(spec.attn.prompt + 1)};
      m.seq_lens_cpu = m.seq_lens;
      m.max_query_len = 1;
      m.max_seq_len = static_cast<int>(spec.attn.prompt + 1);
      m.block_table_num_cols = static_cast<int>(block_table.size());
      m.block_table_tensor = block_table;
      m.slot_mapping = {w4b2::SlotOf(spec.attn, block_table, spec.attn.prompt)};
      m.causal = true;
      const std::vector<int32_t> ids{tokens.back()};
      const std::vector<int32_t> pos{positions.back()};
      const vllm::ModelForwardInput in{.token_ids = ids,
                                       .positions = pos,
                                       .attn_meta = m,
                                       .gdn_meta = gdn_meta,
                                       .attn_kv = pool.attn_kv,
                                       .gdn_state = gdn_state,
                                       .config = config,
                                       .queue = queue,
                                       .logits_indices = no_gather,
                                       .num_reqs = 1};
      const vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);
      REQUIRE(fl.on_device());
      REQUIRE(fl.rows == 1);
      REQUIRE(fl.vocab == spec.attn.vocab);
      const auto* src = static_cast<const float*>(fl.device_tensor.data);
      out.assign(static_cast<size_t>(fl.vocab), 0.0);
      for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(src[i]);
    }
    return out;
  }

  std::vector<double> RefLastRow(const Opts& mo = Opts{},
                                 std::vector<Decision>* trace = nullptr) const {
    const std::vector<double> all =
        RefModel(spec, fdims, sdims, w, tokens, positions, mo, ref::Opts{},
                 w4b::sref::Opts{}, trace);
    const int64_t T = static_cast<int64_t>(tokens.size());
    return std::vector<double>(all.begin() + static_cast<long>((T - 1) * spec.attn.vocab),
                               all.end());
  }

  // The reference's INPUT to the MoE block of layer `layer`: the post-attention
  // RMSNorm output. Captured by walking the SAME reference the gate runs rather
  // than by a second derivation, so the router probe is driven by the numbers
  // this model actually presents to its router.
  std::vector<double> RefMoeInput(int64_t layer) const {
    std::vector<std::vector<double>> inputs;
    (void)RefModel(spec, fdims, sdims, w, tokens, positions, Opts{}, ref::Opts{},
                   w4b::sref::Opts{}, nullptr, &inputs);
    int64_t which = 0;
    for (int64_t l = 0; l < spec.layers(); ++l) {
      if (!spec.is_moe(l)) continue;
      if (l == layer) return inputs[static_cast<size_t>(which)];
      ++which;
    }
    REQUIRE_MESSAGE(false, "layer " << layer << " is not a MoE layer");
    return {};
  }

  // ONE prefill through `ModelRegistry::Forward` on an already-loaded model, so
  // a case can drive a config whose FORWARD is expected to refuse by name.
  void RunPrefillOn(vllm::LoadedModel& model, const HfConfig& cfg_in) const {
    w4a::MlaCachePool pool(spec.layers(), params.physical_latent_row(),
                           /*num_blocks=*/2, spec.attn.page_size);
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const std::vector<int32_t> no_gather;
    std::vector<vllm::GdnStateCache> gdn_state;
    vllm::v1::GDNAttentionMetadata gdn_meta{};
    vllm::v1::CommonAttentionMetadata m;
    m.num_reqs = 1;
    m.num_actual_tokens = static_cast<int>(spec.attn.prompt);
    m.query_start_loc = {0, static_cast<int32_t>(spec.attn.prompt)};
    m.query_start_loc_cpu = m.query_start_loc;
    m.seq_lens = {static_cast<int32_t>(spec.attn.prompt)};
    m.seq_lens_cpu = m.seq_lens;
    m.max_query_len = static_cast<int>(spec.attn.prompt);
    m.max_seq_len = static_cast<int>(spec.attn.prompt);
    m.block_table_num_cols = static_cast<int>(block_table.size());
    m.block_table_tensor = block_table;
    for (int64_t t = 0; t < spec.attn.prompt; ++t) {
      m.slot_mapping.push_back(w4b2::SlotOf(spec.attn, block_table, t));
    }
    m.causal = true;
    const std::vector<int32_t> ids(tokens.begin(), tokens.begin() + spec.attn.prompt);
    const std::vector<int32_t> pos(positions.begin(),
                                   positions.begin() + spec.attn.prompt);
    const vllm::ModelForwardInput in{.token_ids = ids,
                                     .positions = pos,
                                     .attn_meta = m,
                                     .gdn_meta = gdn_meta,
                                     .attn_kv = pool.attn_kv,
                                     .gdn_state = gdn_state,
                                     .config = cfg_in,
                                     .queue = queue,
                                     .logits_indices = no_gather,
                                     .num_reqs = 1};
    (void)ModelRegistry::Forward(model, in);
  }
};

// The bf16 agreement bound. Chosen for SEPARATION rather than to hug the
// residue, and the case PRINTS all three ratios from the numbers it just
// measured so this constant is the only thing written down in advance.
constexpr double kMoeRel = 6e-2;

// The maximum RELATIVE rounding error of a bf16 store: 8 explicit mantissa
// bits, so half an ulp is 2^-8. This is the number the decision margin is
// measured against, scaled by `sigmoid' <= 0.25` and by the fixture's own
// largest router logit.
constexpr double kBf16HalfUlp = 3.90625e-3;

}  // namespace w5
}  // namespace

TEST_CASE(
    "dots3-note W5: the MoE layer runs THROUGH ModelRegistry::Forward, against "
    "an independent double reference") {
  const w5::Bench b;
  // The fixture states its own shape rather than trusting the comment above it.
  REQUIRE(b.params.num_hidden_layers == 3);
  REQUIRE(b.params.first_k_dense_replace == 1);
  CHECK_FALSE(b.params.is_moe_layer(0));  // DENSE, as the released layer 0 is
  CHECK(b.params.is_moe_layer(1));        // MoE + SLIDING attention
  CHECK(b.params.is_moe_layer(2));        // MoE + FULL attention
  CHECK(b.params.kind_of(1) == vllm::Dots3NoteLayerKind::kSlidingAttention);
  CHECK(b.params.kind_of(2) == vllm::Dots3NoteLayerKind::kFullAttention);
  // §4 trap 1, re-asserted rather than inherited: the two group keys are ABSENT
  // from this config (as they are from the released one) and resolve to 1, so
  // the group stage is inert and every group-stage mutation is PREDICTED GREEN.
  CHECK(b.params.n_group == 1);
  CHECK(b.params.topk_group == 1);
  CHECK(b.params.n_routed_experts == 8);
  CHECK(b.params.num_experts_per_tok == 3);
  CHECK(b.params.n_shared_experts == 1);
  CHECK(b.params.shared_intermediate_size() == 6);
  CHECK(b.params.shared_intermediate_size() != b.params.intermediate_size);
  // W5 lifted the MoE refusal; the config carries no nextn tail here, so
  // nothing at all is refused.
  CHECK(vllm::Dots3NoteDeviceRefusal(b.params).empty());

  std::vector<w5::Decision> trace;
  const std::vector<double> want = b.RefLastRow(w5::Opts{}, &trace);
  const std::vector<double> got = b.RunPrefillThenDecode();
  const Diff d = Compare(got, want);

  // ── the DISCRETE half, printed from the numbers just measured ────────────
  // `trace` is (MoE layer, token) in layer-major order: 2 MoE layers x 7 tokens.
  REQUIRE(trace.size() == 14u);
  double min_margin = 1e300;
  double max_logit = 0.0;
  double max_tie_gap = 0.0;
  std::set<int> distinct;
  for (const w5::Decision& dec : trace) {
    min_margin = std::min(min_margin, dec.margin);
    max_logit = std::max(max_logit, dec.max_abs_logit);
    max_tie_gap = std::max(max_tie_gap, dec.tie_gap);
    for (int e : dec.selected) distinct.insert(e);
    CHECK(dec.selected.size() == 3u);
  }
  // The bf16 SCORE ulp at THIS fixture's scale: a logit stored bf16 carries at
  // most `|L| * 2^-8` of error, and `sigmoid' <= 0.25` carries it into the
  // score the top-k compares. The bar is 4x that, which is the bar W4b-3c
  // stated and met at 48.8x for the DSA indexer.
  const double score_ulp = 0.25 * max_logit * w5::kBf16HalfUlp;
  MESSAGE("W5 decision margin: min " << min_margin << " against a bf16 score ulp of "
                                     << score_ulp << " (max|logit| " << max_logit
                                     << ") = " << (min_margin / score_ulp)
                                     << "x the ulp, bar is 4x");
  MESSAGE("W5 distinct experts activated across the batch: "
          << distinct.size() << " of " << b.params.n_routed_experts);
  MESSAGE("W5 deliberate tie, experts 0 and 1: |biased[0] - biased[1]| max over "
          "the batch = " << max_tie_gap << " (EXACT zero expected)");
  CHECK(min_margin > 4.0 * score_ulp);
  // A fixture in which every token picks the same three experts has not tested
  // routing. 2 always-selected + at least 2 of the contended tier.
  CHECK(distinct.size() >= 4u);
  // THE EXACT TIE, and it is exact rather than close: experts 0 and 1 share a
  // router row byte for byte and share a bias, so their biased scores are
  // bit-identical. Both are inside the selected set at every token, so the SET
  // is unambiguous whatever the tie rule is — which is the only thing upstream's
  // `torch.topk(..., sorted=False)` promises (grouped_topk_router.py:134 +
  // :148 — the EXPERT topk, NOT the group topk at :135-137).
  CHECK(max_tie_gap == 0.0);
  for (const w5::Decision& dec : trace) {
    const bool has0 = std::find(dec.selected.begin(), dec.selected.end(), 0) !=
                      dec.selected.end();
    const bool has1 = std::find(dec.selected.begin(), dec.selected.end(), 1) !=
                      dec.selected.end();
    CHECK(has0);
    CHECK(has1);
  }

  // ── the CONTINUOUS half ──────────────────────────────────────────────────
  MESSAGE("W5 mixed dense+MoE forward: " << d.max_rel << " relative ("
                                         << (d.max_rel / w5::kMoeRel)
                                         << "x the bound " << w5::kMoeRel << ")");
  CHECK(d.max_rel < w5::kMoeRel);
}

TEST_CASE(
    "dots3-note W5: every MoE mechanism moves the answer past the bound, and "
    "the NEAREST one is the bias applied to the routing weight") {
  const w5::Bench b;
  const std::vector<double> base = b.RefLastRow(w5::Opts{});
  const std::vector<double> got = b.RunPrefillThenDecode();
  const double residue = Compare(got, base).max_rel;

  struct Arm {
    const char* what;
    w5::Opts o;
  };
  std::vector<Arm> arms;
  {
    // THE NEAREST MECHANISM. Upstream is explicit that the correction bias
    // feeds the SELECTION and not the routing weight ("We use biased scores for
    // expert selection but original scores for routing weights",
    // grouped_topk_router.py:120-123 / :148-150). This fixture's selected
    // experts carry DIFFERENT biases (+1.30, +1.30, 0.0), which is what stops
    // the renormalisation absorbing the defect.
    w5::Opts o;
    o.bias_in_weights = true;
    arms.push_back({"the bias applied to the routing WEIGHT too", o});
  }
  {
    w5::Opts o;
    o.bias_in_selection = false;
    arms.push_back({"the correction bias dropped from the SELECTION", o});
  }
  {
    w5::Opts o;
    o.renormalize = false;
    arms.push_back({"norm_topk_prob dropped (:156-157)", o});
  }
  {
    w5::Opts o;
    o.shared = false;
    arms.push_back({"the shared expert dropped (model.py:127)", o});
  }
  {
    w5::Opts o;
    o.softmax_scoring = true;
    arms.push_back({"softmax scoring instead of sigmoid (:112-117)", o});
  }
  {
    w5::Opts o;
    o.top_k_delta = 1;
    arms.push_back({"top_k + 1", o});
  }
  {
    w5::Opts o;
    o.top_k_delta = -1;
    arms.push_back({"top_k - 1", o});
  }
  {
    w5::Opts o;
    o.routed_scale = 1.7;
    arms.push_back({"a routed_scaling_factor of 1.7 (this config's is 1.0)", o});
  }

  double nearest = 1e300;
  const char* nearest_what = "-";
  for (const Arm& a : arms) {
    const Diff dd = Compare(got, b.RefLastRow(a.o));
    MESSAGE("W5 with " << std::string(a.what) << ": " << dd.max_rel << " relative = "
                       << (dd.max_rel / w5::kMoeRel) << "x the BOUND, "
                       << (dd.max_rel / residue) << "x the RESIDUE");
    CHECK(dd.max_rel > w5::kMoeRel);
    if (dd.max_rel < nearest) {
      nearest = dd.max_rel;
      nearest_what = a.what;
    }
  }
  MESSAGE("W5 residue " << residue << ", bound " << w5::kMoeRel
                        << ", NEAREST mechanism " << nearest << " (" << std::string(nearest_what)
                        << ")");
  // The bound sits near the geometric mean of the residue and the nearest
  // mechanism, which is W4b-2's shape and the reason it is not a hugged
  // threshold. A mechanism under the residue would be a FIXTURE failure and is
  // repaired by retuning the fixture, never by widening the bound.
  CHECK(nearest >= 0.15);
  CHECK(residue < w5::kMoeRel);
}

TEST_CASE(
    "dots3-note W5: the SELECTION is set-equal to the reference's, measured on "
    "vt::MoeRouterTopK at the fixture's own scale") {
  // The DEVICE forward does not expose its top-k, so set equality is asserted
  // on the op the model routes to, driven by the reference's own activations
  // rounded to bf16 — which is the width the model's router GEMM writes. This
  // is the DISCRETE assertion the continuous bound above cannot make: at E=256
  // and a bf16 score ulp near 1/256 the two questions "the same experts,
  // rounded" and "a different expert" are not distinguishable by a norm.
  //
  // WHAT THIS IS AN ORACLE FOR, stated because it matters: `vt::MoeRouterTopK`
  // is itself gated by `tests/vt/test_ops_moe_router_grouped.cpp`. What THIS
  // case adds is that dots3-note's own arguments and its own fixture scale
  // produce the reference's selection, which is a property of the model wiring
  // rather than of the kernel.
  const w5::Bench b;
  const int64_t E = b.params.n_routed_experts;
  const int64_t K = b.params.num_experts_per_tok;
  const int64_t T = static_cast<int64_t>(b.tokens.size());

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};

  int64_t checked = 0;
  for (int64_t l = 0; l < b.spec.layers(); ++l) {
    if (!b.spec.is_moe(l)) continue;
    const w5::MoeW& m = b.w.moe[static_cast<size_t>(l)];
    // The reference's post-attention normed activations for this layer, which
    // is what the model feeds its router.
    const std::vector<double> y = b.RefMoeInput(l);
    std::vector<uint16_t> logits_bf16(static_cast<size_t>(T * E));
    const std::vector<double> logits =
        ref::Dense(y, m.router, T, b.spec.attn.hidden, E);
    for (size_t i = 0; i < logits.size(); ++i) {
      logits_bf16[i] = vt::F32ToBF16(static_cast<float>(logits[i]));
    }
    std::vector<float> bias_f32(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
      bias_f32[static_cast<size_t>(e)] =
          static_cast<float>(m.bias[static_cast<size_t>(e)]);
    }
    vllm::dense_attn::DBuf dlog(d, vt::DType::kBF16, {T, E}, logits_bf16.data());
    vllm::dense_attn::DBuf dbias(d, vt::DType::kF32, {E}, bias_f32.data());
    vllm::dense_attn::DBuf dtw(d, vt::DType::kF32, {T, K});
    vllm::dense_attn::DBuf dtid(d, vt::DType::kI32, {T, K});
    vt::MoeRouterTopKArgs args{};
    args.top_k = static_cast<int>(K);
    args.renormalize = b.params.norm_topk_prob;
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;
    args.num_expert_group = static_cast<int>(b.params.n_group);
    args.topk_group = static_cast<int>(b.params.topk_group);
    args.routed_scaling_factor = static_cast<float>(b.params.routed_scaling_factor);
    vt::Tensor tw = dtw.t(), tid = dtid.t(), tl = dlog.t(), tb = dbias.t();
    vt::MoeRouterTopK(queue, tw, tid, tl, args, &tb);
    std::vector<int32_t> ids(static_cast<size_t>(T * K));
    std::vector<float> wts(static_cast<size_t>(T * K));
    dtid.Download(d, ids.data());
    dtw.Download(d, wts.data());

    // The reference's decision on the SAME logits.
    std::vector<w5::Decision> ref_dec;
    (void)w5::RefMoe(b.spec, m, y, T, w5::Opts{}, &ref_dec);
    REQUIRE(ref_dec.size() == static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      const w5::Decision& dec = ref_dec[static_cast<size_t>(t)];
      // A SET, because `torch.topk(..., sorted=False)` leaves the order
      // unspecified and `vt::MoeCombine` sums over the slots; asserting the
      // order would pin something upstream does not promise.
      std::set<int> want(dec.selected.begin(), dec.selected.end());
      std::set<int> have;
      for (int64_t j = 0; j < K; ++j) {
        have.insert(ids[static_cast<size_t>(t * K + j)]);
      }
      CHECK_MESSAGE(have == want,
                    "layer " << l << " token " << t
                             << ": the op selected a different expert SET than "
                                "the reference");
      // The routing weights, matched per EXPERT rather than per slot for the
      // same reason.
      for (int64_t j = 0; j < K; ++j) {
        const int e = ids[static_cast<size_t>(t * K + j)];
        const auto it = std::find(dec.selected.begin(), dec.selected.end(), e);
        REQUIRE(it != dec.selected.end());
        const size_t rj = static_cast<size_t>(it - dec.selected.begin());
        CHECK(static_cast<double>(wts[static_cast<size_t>(t * K + j)]) ==
              doctest::Approx(dec.weight[rj]).epsilon(2e-3));
      }
      ++checked;
    }
  }
  MESSAGE("W5 selection-set equality checked on " << checked
                                                  << " (layer, token) decisions");
  CHECK(checked == 14);
}

TEST_CASE(
    "dots3-note W5: the group stage is INERT at n_group=1, and the "
    "UNGROUPED-ONLY refusal is what covers it") {
  // PREDICTED GREEN, named in advance and then demonstrated. With `n_group == 1`
  // the group stage builds one group, `topk_group == 1` keeps it, and the mask
  // is all-ones — so deleting the whole stage from the reference changes
  // NOTHING. Every group-stage mutation is therefore a coverage hole by
  // construction rather than by omission, and the thing that actually stands
  // between this port and a regrouped router is the parse-time refusal below.
  const w5::Bench b;
  w5::Opts no_group;
  no_group.group_stage = false;
  const std::vector<double> with = b.RefLastRow(w5::Opts{});
  const std::vector<double> without = b.RefLastRow(no_group);
  const Diff dd = Compare(with, without);
  MESSAGE("W5 group stage deleted from the reference: " << dd.max_abs
                                                        << " absolute (EXACT zero expected)");
  CHECK(dd.max_abs == 0.0);

  // THE STANDING COVERAGE, re-asserted here rather than inherited silently: a
  // config that really did carry a grouped router is refused BY NAME at parse
  // time (§4 trap 1, configs/dots3_note.py:18-19). Without this the inertness
  // above would be an untested claim about a path nothing guards.
  for (const auto& kv : std::vector<std::pair<const char*, int64_t>>{
           {"n_group", 2}, {"topk_group", 2}}) {
    nlohmann::json doc = w5::ConfigDoc(w5::Spec{});
    doc[kv.first] = kv.second;
    TempConfig cfg(doc);
    CHECK_THROWS_WITH_AS((void)ParseDots3NoteParams(LoadHfConfig(cfg.path())),
                         doctest::Contains("UNGROUPED"), std::runtime_error);
  }

  // PREDICTED GREEN #2: `routed_scaling_factor` is 1.0 on this config and on
  // the released one, so the multiply is the identity and a mutation that drops
  // it cannot be seen here. Stated so it is not discovered as a surprise.
  CHECK(b.params.routed_scaling_factor == 1.0);
  w5::Opts unit_scale;
  unit_scale.routed_scale = 1.0;
  CHECK(Compare(with, b.RefLastRow(unit_scale)).max_abs == 0.0);
}

TEST_CASE(
    "dots3-note W5: the SHARED expert's width is moe_intermediate_size * "
    "n_shared_experts, and the wrong one refuses BY NAME") {
  // `intermediate_size=_padded_mlp_size(config.moe_intermediate_size *
  // num_shared_experts, ...)` (model.py:103-107). On the RELEASED checkpoint
  // that is 1536 while `intermediate_size` is 13824, and the shipped
  // `mlp.shared_experts.gate_proj.weight` is [1536, 5120] — so a port that
  // reached for the wrong field builds a 9x-too-wide MLP. This is the case that
  // makes reaching for it a loud failure rather than a silent read past the end
  // of a weight.
  const w5::Bench b;
  REQUIRE(b.params.moe_intermediate_size == 6);
  REQUIRE(b.params.intermediate_size == 10);
  REQUIRE(b.params.shared_intermediate_size() == 6);

  std::vector<w4a::StOut> bad = b.entries;
  bool patched = false;
  for (w4a::StOut& e : bad) {
    if (e.name != "model.layers.1.mlp.shared_experts.gate_proj.weight") continue;
    // The DENSE layers' width, which is what a port reading `intermediate_size`
    // would allocate.
    e.shape = {b.params.intermediate_size, b.params.hidden_size};
    e.values.assign(static_cast<size_t>(b.params.intermediate_size *
                                        b.params.hidden_size),
                    0.125);
    patched = true;
  }
  REQUIRE(patched);
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(b.config);
  w4a::TempCheckpoint ckpt(bad);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  CHECK_THROWS_WITH_AS(reg.factory->load_weights(reg, b.config, source),
                       doctest::Contains("shared_experts"), std::runtime_error);
}

TEST_CASE(
    "dots3-note W5: a BLOCKWISE-quantized checkpoint refuses BY NAME, and the "
    "brick it names is W9") {
  // `dots-studio/dots3-note-prev-fp8` carries `quantization_config = {"fmt":
  // "e4m3", "quant_method": "fp8", "activation_scheme": "dynamic",
  // "weight_block_size": [128, 128]}` and ships a `weight_scale_inv` beside
  // every projection — at the routed experts' [1536, 5120] that scale is
  // [12, 40]. `dense_loaders::MaterializeBf16Source` looks up `<name>_scale`
  // and accepts only a per-tensor or per-output-ROW scale, so WITHOUT this
  // refusal the load throws a bare "tensor not found" that names neither fp8
  // nor the brick that owes it.
  nlohmann::json doc = w5::ConfigDoc(w5::Spec{});
  doc["quantization_config"] = {{"quant_method", "fp8"},
                                {"fmt", "e4m3"},
                                {"activation_scheme", "dynamic"},
                                {"weight_block_size", {128, 128}}};
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const Dots3NoteParams p = ParseDots3NoteParams(config);
  CHECK(p.quant_method == "fp8");
  REQUIRE(p.weight_block_size.size() == 2u);
  CHECK(p.weight_block_size[0] == 128);
  CHECK(p.weight_block_size[1] == 128);
  CHECK(p.has_blockwise_quant());
  const std::string why = vllm::Dots3NoteDeviceRefusal(p);
  MESSAGE("W5 blockwise-fp8 refusal: " << why);
  CHECK(why.find("BLOCKWISE") != std::string::npos);
  CHECK(why.find("weight_scale_inv") != std::string::npos);
  CHECK(why.find("W9") != std::string::npos);

  // ...and it fires BEFORE any bf16 loader runs, through the real registry, so
  // the message a user gets names fp8 rather than a missing tensor.
  const w5::Bench b;
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  w4a::TempCheckpoint ckpt(b.entries);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ckpt.file()));
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model =
      reg.factory->load_weights(reg, config, source);
  REQUIRE(model != nullptr);
  // The load itself succeeds (the accounting is a real production result) and
  // the FORWARD refuses by name, which is this port's standing shape.
  CHECK_THROWS_WITH_AS(b.RunPrefillOn(*model, config), doctest::Contains("BLOCKWISE"),
                       std::runtime_error);

  // The RELEASED bf16 checkpoint carries no `quantization_config` at all, so
  // nothing about it changed. Asserted rather than assumed, because a refusal
  // keyed on an absent key that is actually present would turn the released
  // config away.
  TempConfig released(FixtureConfigDoc());
  const Dots3NoteParams rp = ParseDots3NoteParams(LoadHfConfig(released.path()));
  CHECK_FALSE(rp.has_blockwise_quant());
  CHECK(rp.quant_method.empty());
}

TEST_CASE(
    "dots3-note W5c: the RELEASED config no longer refuses, and the nextn tail "
    "is a NAMED W10 deferral instead") {
  // THE HEADLINE. `Dots3NoteDeviceRefusal` over the real released
  // `config.json` was non-empty at every brick from W1 to W4b-3c — W4b-2's own
  // case asserts `why.find("MoE") != npos` — and it is empty now. Two branches
  // went: the MoE layer (W5) and the nextn tail (W5c, #2176).
  TempConfig cfg(FixtureConfigDoc());
  const HfConfig config = LoadHfConfig(cfg.path());
  const Dots3NoteParams p = ParseDots3NoteParams(config);
  // The released config really does have both features, so the flip is not the
  // fixture quietly losing them.
  REQUIRE(p.num_hidden_layers == 46);
  REQUIRE(p.n_routed_experts == 256);
  REQUIRE(p.num_experts_per_tok == 8);
  REQUIRE(p.first_k_dense_replace == 1);
  REQUIRE(p.is_moe_layer(1));
  // §4 trap 3: the key is ABSENT and defaults to 1, which is what made the
  // nextn branch fire on every released checkpoint.
  REQUIRE_FALSE(FixtureConfigDoc().contains("num_nextn_predict_layers"));
  REQUIRE(p.num_nextn_predict_layers == 1);

  const std::string why = vllm::Dots3NoteDeviceRefusal(p);
  MESSAGE("W5 released-config refusal is now: '" << why << "' (empty)");
  CHECK(why.empty());

  // The nextn tensors are DEFERRED, not silently dropped: the predicate names
  // exactly the 19 the release ships, and the accounting counts them.
  CHECK(vllm::Dots3NoteIsNextnTensor(p, "model.layers.46.input_layernorm.weight"));
  CHECK(vllm::Dots3NoteIsNextnTensor(p, "model.layers.46.eh_proj.weight"));
  CHECK(vllm::Dots3NoteIsNextnTensor(p, "model.mtp.embed_tokens.weight"));
  // ...and it claims NOTHING the backbone reads. 46 is the tail; 45 is the last
  // backbone layer and must not be swept up with it.
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(p, "model.layers.45.input_layernorm.weight"));
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(p, "model.layers.4.mlp.gate.weight"));
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(p, "model.embed_tokens.weight"));
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(p, "vision_encoder.blocks.0.attn.qkv.weight"));
  // A config with NO nextn tail claims nothing at all, so the predicate cannot
  // steal a backbone layer from a checkpoint that ships none.
  Dots3NoteParams none = p;
  none.num_nextn_predict_layers = 0;
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(none, "model.layers.46.input_layernorm.weight"));
  CHECK_FALSE(vllm::Dots3NoteIsNextnTensor(none, "model.mtp.embed_tokens.weight"));

  // What is STILL refused, so the flip above is not read as "the model runs".
  // The blockwise-fp8 arm has its own case; here it is the honest scale.
  MESSAGE("W5: the released config is REPRESENTABLE, which is not RUNNABLE — "
          "the MoE is 545.82 GB of a 576.89 GB checkpoint (94.62%), and the "
          "298.67 GB fp8 sibling is refused by name (W9)");
}

TEST_CASE("dots3-note W5: the mixed dense+MoE device forward is DETERMINISTIC") {
  const w5::Bench b;
  const std::vector<double> a = b.RunPrefillThenDecode();
  const std::vector<double> c = b.RunPrefillThenDecode();
  REQUIRE(a.size() == c.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK(a[i] == c[i]);
}


