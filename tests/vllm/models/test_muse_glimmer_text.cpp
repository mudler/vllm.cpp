// Muse Glimmer TEXT TOWER (W1) correctness gate. CPU-only, checkpoint-free.
//
// WHAT THIS ESTABLISHES. The whole text forward is checked against an INDEPENDENT
// fp32 reference transcribed directly from vllm#51655 head `075d645af`
// (`vllm/model_executor/models/muse_glimmer.py`, MuseGlimmerAttention.forward
// :1177-1215, MuseGlimmerDecoderLayer.forward :1249-1277, MuseGlimmerModel
// :1279-1345, compute_logits :1615-1621). The reference is written from the python,
// not from our C++, so it is a second opinion and not a restatement — and every
// mechanism additionally gets a PROPERTY test that a plausible-but-wrong port
// breaks:
//
//   * embed_norm is a WEIGHTLESS RMSNorm (:1286), so the forward is INVARIANT to a
//     power-of-two rescale of the whole embedding table — which Gemma's
//     sqrt(hidden) multiplier in the same slot is not.
//   * QK-norm (:1189-1196) makes the forward INVARIANT to a power-of-two rescale of
//     the q and k projections; without it every attention score would scale.
//   * the query pre-scale (:1192) is a CONFIG CONSTANT applied after QK-norm, so it
//     changes the output while the projection rescale above does not.
//   * iRoPE (:1114-1116, :1167-1168): an all-NoPE model is INVARIANT to a shift of
//     `positions` and to the sliding window; an all-RoPE model is sensitive to both.
//     RoPE and the window travel together.
//   * split eps (:1236-1247): `post_norm_eps` alone changes the output, which it
//     cannot do if the post-norms wrongly read `rms_norm_eps`.
//   * the FINAL norm (:1296) has NO `+1` offset, unlike all four sandwich norms:
//     a zeroed final-norm weight must zero the logits.
//   * `output_multiplier` is applied BEFORE the soft-cap (:1618-1621).
//
// WHAT THIS DOES NOT ESTABLISH. Nothing about tokens from the real 30B checkpoint,
// and nothing at all about speed. Muse Glimmer is BEYOND the parity pin
// `555967922` — the pinned oracle cannot load `muse_glimmer`, so there is neither a
// golden nor a throughput denominator (specs/muse-glimmer.md §0). Token-exact e2e
// vs the HF reference is W2; the perception encoder is W3.
#include "vllm/model_executor/models/muse_glimmer.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // FusedChainAdoptEnabled
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::HfConfig;
using vllm::MuseGlimmerLayerWeights;
using vllm::MuseGlimmerModel;
using vllm::MuseGlimmerParams;
using vllm::MuseGlimmerWeights;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

namespace {

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// ─────────────────────────── tiny synthetic model ───────────────────────────
// Real geometry scaled to the smallest shape that still exercises every branch:
// GQA (2 query heads / 1 KV head), head_dim 4 (so RoPE has 2 rotary pairs), two
// layers with ONE RoPE layer and ONE NoPE layer, a sliding window shorter than the
// sequence (so the window actually masks), an untied lm_head, a non-unit
// output_multiplier and a soft-cap.
struct TinySpec {
  int64_t vocab = 11, hidden = 8, inter = 6, layers = 2;
  int64_t heads = 2, kv_heads = 1, head_dim = 4;
  double rms_eps = 1e-2;   // deliberately FAR from post_eps so the split is visible
  double post_eps = 1e-6;
  // Modular schema: `qk_scale_factor` is already folded, and upstream keeps it as-is
  // when it is below sqrt(head_dim) (muse_glimmer.py:472-517). The released 30B ships
  // 3.87 against sqrt(128)=11.31; at THIS geometry sqrt(head_dim)=2, so the tiny model
  // uses a value below 2 to stay on the same branch. The magnitude disambiguation
  // itself is gated at the real head_dim 128 in test_muse_glimmer_scaffold.
  double qk_scale_factor = 1.75;
  int64_t sliding = 2;
  std::vector<int64_t> no_rope = {1, 0};  // 1 => RoPE + sliding, 0 => NoPE + full
  double out_mult = 0.75;
  double softcap = 20.0;
  double rope_theta = 500000.0;
};

HfConfig MakeConfig(const TinySpec& s) {
  nlohmann::json text{
      {"model_type", "muse_glimmer_text"},
      {"vocab_size", s.vocab},
      {"hidden_size", s.hidden},
      {"intermediate_size", s.inter},
      {"num_hidden_layers", s.layers},
      {"num_attention_heads", s.heads},
      {"num_key_value_heads", s.kv_heads},
      {"head_dim", s.head_dim},
      {"max_position_embeddings", 131072},
      {"sliding_window", s.sliding},
      {"rms_norm_eps", s.rms_eps},
      {"post_norm_eps", s.post_eps},
      {"hidden_activation", "silu"},
      {"qk_scale_factor", s.qk_scale_factor},
      {"no_rope_layers", s.no_rope},
      {"output_multiplier", s.out_mult},
      {"final_logit_softcapping", s.softcap},
      {"tie_word_embeddings", false},
      {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", s.rope_theta}}},
  };
  HfConfig c;
  c.architectures = {"MuseGlimmerForConditionalGeneration"};
  c.hidden_size = s.hidden;
  c.num_hidden_layers = s.layers;
  c.vocab_size = s.vocab;
  c.num_attention_heads = s.heads;
  c.raw = nlohmann::json{{"model_type", "muse_glimmer"}, {"text_config", text}};
  return c;
}

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                     float scale = 0.08f) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return o;
}

// Read an owned bf16 tensor back as f32 — the reference reads the SAME bytes the
// forward does, so any difference is arithmetic, never a different model.
std::vector<float> Host(const OwnedTensor& t) {
  const size_t n = t.bytes.size() / sizeof(uint16_t);
  const auto* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = vt::BF16ToF32(p[i]);
  return out;
}

MuseGlimmerWeights TinyWeights(const HfConfig& cfg) {
  MuseGlimmerWeights w;
  w.params = vllm::ParseMuseGlimmerParams(cfg);
  const vllm::MuseGlimmerTextParams& t = w.params.text;
  const int64_t H = t.hidden_size, I = t.intermediate_size, V = t.vocab_size;
  const int64_t qdim = t.num_attention_heads * t.head_dim;
  const int64_t kdim = t.num_key_value_heads * t.head_dim;

  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.3f);
  w.lm_head = MakeBf16({H, V}, /*nk=*/false, 3);  // Matmul-B [in,out]; UNTIED
  uint32_t seed = 100;
  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    MuseGlimmerLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.pre_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.output_gate_proj = MakeBf16({qdim, H}, /*nk=*/true, seed++);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
    w.layers.push_back(std::move(lw));
  }
  w.text_loaded = true;
  return w;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const MuseGlimmerParams& p, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = p.text.num_key_value_heads, Dh = p.text.head_dim;
    for (int64_t l = 0; l < p.text.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
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

const std::vector<int32_t>& Tokens() {
  static const std::vector<int32_t> t = {3, 7, 1, 9, 4};
  return t;
}
std::vector<int32_t> Positions(int32_t stride = 1) {
  std::vector<int32_t> p(Tokens().size());
  for (size_t i = 0; i < p.size(); ++i) p[i] = stride * static_cast<int32_t>(i);
  return p;
}

std::vector<float> RunForward(const MuseGlimmerWeights& w,
                              const std::vector<int32_t>& positions) {
  const int64_t T = static_cast<int64_t>(Tokens().size());
  CachePool pool(w.params, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  vt::Queue q = Qcpu();
  return MuseGlimmerModel::Forward(Tokens(), positions, am, pool.attn_kv, w, q);
}

// ───────────────────── the independent fp32 reference ─────────────────────
// Transcribed from muse_glimmer.py @ 075d645af. Written from the PYTHON, so it is a
// second implementation and not a restatement of the C++ under test.

// MuseGlimmerRMSNorm (:520-552): fp32 `_norm`, weight applied as (w + offset).
// `weight == nullptr` is the WEIGHTLESS form (with_scale=False) used by embed_norm
// and qk_norm.
void RefRmsNorm(std::vector<float>& row, const float* weight, float offset,
                float eps) {
  double sq = 0.0;
  for (float v : row) sq += static_cast<double>(v) * v;
  const double inv = 1.0 / std::sqrt(sq / static_cast<double>(row.size()) + eps);
  for (size_t j = 0; j < row.size(); ++j) {
    const double scaled = static_cast<double>(row[j]) * inv;
    row[j] = static_cast<float>(weight == nullptr ? scaled
                                                  : scaled * (weight[j] + offset));
  }
}

// torch F.linear against a raw-NK [N,K] weight: out[n] = sum_k in[k]*W[n,k].
std::vector<float> RefLinear(const std::vector<float>& in, const std::vector<float>& w,
                             int64_t N, int64_t K) {
  std::vector<float> out(static_cast<size_t>(N), 0.0f);
  for (int64_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (int64_t k = 0; k < K; ++k)
      acc += static_cast<double>(in[static_cast<size_t>(k)]) *
             w[static_cast<size_t>(n * K + k)];
    out[static_cast<size_t>(n)] = static_cast<float>(acc);
  }
  return out;
}

// NeoX RoPE over one head vector (get_rope(..., is_neox_style=True), :1163-1174).
void RefRope(float* head, int64_t Dh, double base, int64_t pos) {
  const int64_t half = Dh / 2;
  for (int64_t i = 0; i < half; ++i) {
    const double freq = std::pow(base, -2.0 * static_cast<double>(i) /
                                            static_cast<double>(Dh));
    const double angle = static_cast<double>(pos) * freq;
    const float c = static_cast<float>(std::cos(angle));
    const float s = static_cast<float>(std::sin(angle));
    const float x = head[i], y = head[i + half];
    head[i] = x * c - y * s;
    head[i + half] = x * s + y * c;
  }
}

std::vector<float> RefForward(const MuseGlimmerWeights& w,
                              const std::vector<int32_t>& positions) {
  const vllm::MuseGlimmerTextParams& t = w.params.text;
  const int64_t T = static_cast<int64_t>(Tokens().size());
  const int64_t H = t.hidden_size, I = t.intermediate_size, V = t.vocab_size;
  const int64_t Hq = t.num_attention_heads, Hkv = t.num_key_value_heads;
  const int64_t Dh = t.head_dim, qdim = Hq * Dh, kdim = Hkv * Dh;
  const float rms_eps = t.rms_norm_eps, post_eps = t.post_norm_eps;
  const float attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));

  const std::vector<float> embed = Host(w.embed_tokens);
  // embed_input_ids (:1298-1299): embed_tokens then the WEIGHTLESS embed_norm.
  std::vector<std::vector<float>> h(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    h[static_cast<size_t>(i)].assign(
        embed.begin() + static_cast<int64_t>(Tokens()[static_cast<size_t>(i)]) * H,
        embed.begin() + (static_cast<int64_t>(Tokens()[static_cast<size_t>(i)]) + 1) * H);
    RefRmsNorm(h[static_cast<size_t>(i)], nullptr, 0.0f, rms_eps);
  }

  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    const MuseGlimmerLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    const std::vector<float> w_in = Host(lw.input_layernorm);
    const std::vector<float> w_pa = Host(lw.post_attention_layernorm);
    const std::vector<float> w_pf = Host(lw.pre_feedforward_layernorm);
    const std::vector<float> w_pff = Host(lw.post_feedforward_layernorm);
    const std::vector<float> w_qkv = Host(lw.attn.qkv_proj);
    const std::vector<float> w_o = Host(lw.attn.o_proj);
    const std::vector<float> w_g = Host(lw.attn.output_gate_proj);
    const std::vector<float> w_gu = Host(lw.mlp.gate_up_proj);
    const std::vector<float> w_dn = Host(lw.mlp.down_proj);
    const bool use_rope = t.no_rope_layers[static_cast<size_t>(l)] == 1;

    // residual = hidden_states; hidden_states = input_layernorm(hidden_states)
    std::vector<std::vector<float>> res = h;
    std::vector<std::vector<float>> x = h;
    for (auto& row : x) RefRmsNorm(row, w_in.data(), 1.0f, rms_eps);

    // qkv_proj, then WEIGHTLESS per-head QK-norm BEFORE RoPE, then the query
    // pre-scale on q only (:1183-1196).
    std::vector<std::vector<float>> q(static_cast<size_t>(T)), k(static_cast<size_t>(T)),
        v(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
      const std::vector<float> qkv =
          RefLinear(x[static_cast<size_t>(i)], w_qkv, qdim + 2 * kdim, H);
      q[static_cast<size_t>(i)].assign(qkv.begin(), qkv.begin() + qdim);
      k[static_cast<size_t>(i)].assign(qkv.begin() + qdim, qkv.begin() + qdim + kdim);
      v[static_cast<size_t>(i)].assign(qkv.begin() + qdim + kdim, qkv.end());
      if (t.use_qk_norm) {
        for (int64_t hh = 0; hh < Hq; ++hh) {
          std::vector<float> head(q[static_cast<size_t>(i)].begin() + hh * Dh,
                                  q[static_cast<size_t>(i)].begin() + (hh + 1) * Dh);
          RefRmsNorm(head, nullptr, 0.0f, rms_eps);
          for (int64_t e = 0; e < Dh; ++e)
            q[static_cast<size_t>(i)][static_cast<size_t>(hh * Dh + e)] =
                static_cast<float>(head[static_cast<size_t>(e)] * t.scale_query_by);
        }
        for (int64_t hh = 0; hh < Hkv; ++hh) {
          std::vector<float> head(k[static_cast<size_t>(i)].begin() + hh * Dh,
                                  k[static_cast<size_t>(i)].begin() + (hh + 1) * Dh);
          RefRmsNorm(head, nullptr, 0.0f, rms_eps);
          for (int64_t e = 0; e < Dh; ++e)
            k[static_cast<size_t>(i)][static_cast<size_t>(hh * Dh + e)] =
                head[static_cast<size_t>(e)];
        }
      }
      if (use_rope) {
        for (int64_t hh = 0; hh < Hq; ++hh)
          RefRope(q[static_cast<size_t>(i)].data() + hh * Dh, Dh, t.rope_theta,
                  positions[static_cast<size_t>(i)]);
        for (int64_t hh = 0; hh < Hkv; ++hh)
          RefRope(k[static_cast<size_t>(i)].data() + hh * Dh, Dh, t.rope_theta,
                  positions[static_cast<size_t>(i)]);
      }
    }

    // Causal GQA attention, softmax scale head_dim**-0.5, sliding window ONLY on
    // RoPE layers (:1167-1168). Full prefill => key index == query position.
    std::vector<std::vector<float>> attn(static_cast<size_t>(T),
                                         std::vector<float>(static_cast<size_t>(qdim), 0.0f));
    const int64_t group = Hq / Hkv;
    for (int64_t i = 0; i < T; ++i) {
      const int64_t jmin =
          (use_rope && t.sliding_window > 0) ? std::max<int64_t>(0, i - (t.sliding_window - 1)) : 0;
      for (int64_t hh = 0; hh < Hq; ++hh) {
        const int64_t kvh = hh / group;
        std::vector<double> score;
        double mx = -1e30;
        for (int64_t j = jmin; j <= i; ++j) {
          double dot = 0.0;
          for (int64_t e = 0; e < Dh; ++e)
            dot += static_cast<double>(q[static_cast<size_t>(i)][static_cast<size_t>(hh * Dh + e)]) *
                   k[static_cast<size_t>(j)][static_cast<size_t>(kvh * Dh + e)];
          dot *= attn_scale;
          score.push_back(dot);
          mx = std::max(mx, dot);
        }
        double denom = 0.0;
        for (double& sdot : score) {
          sdot = std::exp(sdot - mx);
          denom += sdot;
        }
        for (size_t si = 0; si < score.size(); ++si) {
          const double p = score[si] / denom;
          const int64_t j = jmin + static_cast<int64_t>(si);
          for (int64_t e = 0; e < Dh; ++e)
            attn[static_cast<size_t>(i)][static_cast<size_t>(hh * Dh + e)] +=
                static_cast<float>(p * v[static_cast<size_t>(j)][static_cast<size_t>(kvh * Dh + e)]);
        }
      }
    }

    // Output gate (:1202-1206): sigmoid(output_gate_proj(x)) * attn, where x is the
    // NORMED LAYER INPUT — then o_proj.
    std::vector<std::vector<float>> h_attn(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> gated = attn[static_cast<size_t>(i)];
      if (t.use_attn_output_gate) {
        const std::vector<float> g = RefLinear(x[static_cast<size_t>(i)], w_g, qdim, H);
        for (int64_t e = 0; e < qdim; ++e)
          gated[static_cast<size_t>(e)] = static_cast<float>(
              gated[static_cast<size_t>(e)] /
              (1.0 + std::exp(-static_cast<double>(g[static_cast<size_t>(e)]))));
      }
      h_attn[static_cast<size_t>(i)] = RefLinear(gated, w_o, H, qdim);
      // post_attention_layernorm (post_norm_eps), then residual += .
      RefRmsNorm(h_attn[static_cast<size_t>(i)], w_pa.data(), 1.0f, post_eps);
      for (int64_t e = 0; e < H; ++e)
        h_attn[static_cast<size_t>(i)][static_cast<size_t>(e)] +=
            res[static_cast<size_t>(i)][static_cast<size_t>(e)];
    }
    h = h_attn;

    // residual = hidden; pre_feedforward_layernorm -> SwiGLU MLP ->
    // post_feedforward_layernorm -> residual += .
    res = h;
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> x2 = h[static_cast<size_t>(i)];
      RefRmsNorm(x2, w_pf.data(), 1.0f, rms_eps);
      const std::vector<float> gu = RefLinear(x2, w_gu, 2 * I, H);
      std::vector<float> act(static_cast<size_t>(I));
      for (int64_t e = 0; e < I; ++e) {
        const double g = gu[static_cast<size_t>(e)];
        act[static_cast<size_t>(e)] =
            static_cast<float>((g / (1.0 + std::exp(-g))) * gu[static_cast<size_t>(I + e)]);
      }
      std::vector<float> down = RefLinear(act, w_dn, H, I);
      RefRmsNorm(down, w_pff.data(), 1.0f, post_eps);
      for (int64_t e = 0; e < H; ++e)
        down[static_cast<size_t>(e)] += res[static_cast<size_t>(i)][static_cast<size_t>(e)];
      h[static_cast<size_t>(i)] = down;
    }
  }

  // Final norm (:1296) — weight as `w`, NO +1 offset — then the UNTIED lm_head,
  // the output multiplier and the tanh soft-cap (:1615-1621).
  const std::vector<float> w_fn = Host(w.final_norm);
  const std::vector<float> lm = Host(w.lm_head);  // [H, V]
  std::vector<float> logits(static_cast<size_t>(T * V));
  for (int64_t i = 0; i < T; ++i) {
    std::vector<float> row = h[static_cast<size_t>(i)];
    RefRmsNorm(row, w_fn.data(), 0.0f, rms_eps);
    for (int64_t vi = 0; vi < V; ++vi) {
      double acc = 0.0;
      for (int64_t e = 0; e < H; ++e)
        acc += static_cast<double>(row[static_cast<size_t>(e)]) *
               lm[static_cast<size_t>(e * V + vi)];
      acc *= t.output_multiplier;
      if (t.final_logit_softcapping > 0.0)
        acc = t.final_logit_softcapping * std::tanh(acc / t.final_logit_softcapping);
      logits[static_cast<size_t>(i * V + vi)] = static_cast<float>(acc);
    }
  }
  return logits;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

bool Differs(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() != b.size() ||
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0;
}

// Multiply a whole owned bf16 tensor by an exact power of two — bf16 stores the
// mantissa unchanged, so this is an EXACT rescale and any output difference is a
// real sensitivity, never rounding.
void ScaleRowsPow2(OwnedTensor* t, int64_t first_row, int64_t rows, int64_t cols,
                   float factor) {
  auto* p = reinterpret_cast<uint16_t*>(t->bytes.data());
  for (int64_t r = first_row; r < first_row + rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r * cols + c);
      p[idx] = vt::F32ToBF16(vt::BF16ToF32(p[idx]) * factor);
    }
}

void ZeroTensor(OwnedTensor* t) {
  auto* p = reinterpret_cast<uint16_t*>(t->bytes.data());
  const size_t n = t->bytes.size() / sizeof(uint16_t);
  for (size_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(0.0f);
}

}  // namespace

TEST_CASE("muse_glimmer text: forward runs, finite, deterministic") {
  const HfConfig cfg = MakeConfig(TinySpec{});
  const MuseGlimmerWeights w = TinyWeights(cfg);
  const std::vector<float> a = RunForward(w, Positions());
  REQUIRE(a.size() == static_cast<size_t>(Tokens().size()) *
                          static_cast<size_t>(w.params.text.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));
  CHECK_FALSE(Differs(a, RunForward(w, Positions())));
}

TEST_CASE("muse_glimmer text: refuses a params-only (unloaded) weight set BY NAME") {
  const HfConfig cfg = MakeConfig(TinySpec{});
  MuseGlimmerWeights w = TinyWeights(cfg);
  w.text_loaded = false;  // the W0 structural-accounting form
  CHECK_THROWS(RunForward(w, Positions()));
}

// THE GATE: the whole text forward vs the independent fp32 reference. Our path is
// bf16 per op (vLLM's own stores) against an f64-accumulating f32 reference, so the
// band is a bf16-DEPTH envelope, not an equality — but it is far tighter than any
// of the mechanism mutations below, each of which moves the output by O(1).
TEST_CASE("muse_glimmer text: matches the fp32 reference transcribed from upstream") {
  const HfConfig cfg = MakeConfig(TinySpec{});
  const MuseGlimmerWeights w = TinyWeights(cfg);
  const std::vector<float> got = RunForward(w, Positions());
  const std::vector<float> want = RefForward(w, Positions());
  double scale = 0.0;
  for (float x : want) scale = std::max(scale, std::abs(static_cast<double>(x)));
  const double diff = MaxAbsDiff(got, want);
  // This band is the same rounded-up W1 measurement class the biting case below
  // no longer carries, and `4712dac40` moved it from 0.242 to 0.687 of its
  // bound. Owed as #1466, with its own derivation; not re-tuned here.
  MESSAGE("muse_glimmer text vs fp32 reference: max|diff|=" << diff
          << " over logits of max|.|=" << scale);
  CHECK(diff <= 5e-4);

  // Same model with a BITING soft-cap, so the reference also pins the ORDER of the
  // output multiplier and the cap (:1618-1621). At the default cap of 20 the tanh is
  // linear over these logits and the order is unobservable; at 1e-3 it is not.
  //
  // WHERE THE ORDER IS VISIBLE, and it is not in a max|diff| band. At cap = 1e-3
  // the logits (max|.| = 4.88e-2) drive |x / cap| ~ 50, so the tanh is saturated
  // to within 1e-40 and the two candidate orders reach DIFFERENT saturation
  // magnitudes: `cap * tanh(mult * x / cap)` reaches `cap`, and the swapped
  // `mult * cap * tanh(x / cap)` reaches `out_mult * cap`. On this fixture that
  // is 1.0e-03 against 7.5e-04. So the order is an ALGEBRAIC property of the
  // output's range, and gating it there costs no band to tune: MEASURED, our
  // saturation and the reference's agree EXACTLY (delta 0), and the swap moves it
  // by |1 - out_mult| * cap = 2.5e-04 — 32x the bound below.
  //
  // The bound: our logits pass a bf16 store, so the saturation we can report
  // departs from `cap` by at most one bf16 unit roundoff (2^-8) on each side.
  //
  // WHAT THIS REPLACES, AND WHY. `bdiff <= 1e-5` was the W1 measurement (5.28e-06
  // at `3a54c4b7d`) rounded up, carrying 1.89x of headroom and no derivation.
  // `4712dac40` narrowed `act(gate)` to the input dtype — upstream's polarity, and
  // the only form that reproduces the committed `silu_and_mul_bf16_8x256` oracle
  // golden bit-exactly — which adds one legitimate bf16 rounding per activation
  // element and grew this envelope 2.8x, to 1.48e-05. A constant a CORRECT kernel
  // change consumes was never a bound. #1458.
  //
  // Two replacements were tried and REJECTED, recorded so nobody re-derives them:
  // `bdiff <= diff` is rigorous (the cap is 1-Lipschitz, so it cannot enlarge a
  // difference) but does NOT red the swap, because the uncapped envelope is
  // 3.4e-04 and the defect is 2.5e-04. A measured twin at `out_mult = 1`, where
  // the two orders coincide by algebra, gives 6.10e-06 — but it is a DIFFERENT
  // model, and its max|diff| sits at a different point on the tanh knee, so the
  // gated run reaches 2.4x it with nothing wrong. The band is knee-driven and is
  // kept only at its rigorous Lipschitz value.
  TinySpec biting;
  biting.softcap = 1e-3;
  const MuseGlimmerWeights wb = TinyWeights(MakeConfig(biting));
  const std::vector<float> bgot = RunForward(wb, Positions());
  const std::vector<float> bwant = RefForward(wb, Positions());
  const double bdiff = MaxAbsDiff(bgot, bwant);
  double got_sat = 0.0, want_sat = 0.0;
  for (float x : bgot) got_sat = std::max(got_sat, std::abs(static_cast<double>(x)));
  for (float x : bwant) want_sat = std::max(want_sat, std::abs(static_cast<double>(x)));
  // THE PRECONDITION, ASSERTED. Everything above turns on the tanh being
  // saturated at this cap; if the fixture ever drifts so that it is not, the two
  // orders stop having different ranges and this case degrades silently from an
  // order gate into a weak magnitude comparison that the swap no longer reds.
  // Saturated, the reference's own maximum IS the cap, and both sides are f32
  // values, so the check is an equality with nothing to tune.
  REQUIRE(want_sat > 0.0);
  REQUIRE(static_cast<float>(want_sat) == static_cast<float>(biting.softcap));
  // 2^-8 is bf16's UNIT ROUNDOFF -- the largest relative error a correctly
  // rounded bf16 store can carry. It is NOT the format's relative spacing
  // (machine epsilon), which is twice it at 2^-7. The unit roundoff is what a
  // stored value's departure is bounded by, which is what both uses want.
  constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
  MESSAGE("muse_glimmer text vs fp32 reference (biting soft-cap): max|diff|="
          << bdiff << " (uncapped envelope " << diff << "); saturation ours="
          << got_sat << " ref=" << want_sat << ", delta="
          << std::abs(got_sat - want_sat) << " against "
          << (2.0 * kBf16UnitRoundoff * want_sat));
  CHECK(std::abs(got_sat - want_sat) <= 2.0 * kBf16UnitRoundoff * want_sat);
  CHECK(bdiff <= diff);
}

TEST_CASE("muse_glimmer text: embed_norm is WEIGHTLESS RMSNorm, not a sqrt(H) scale") {
  // RMSNorm is scale-invariant, so rescaling the WHOLE embedding table by an exact
  // power of two must leave the logits bit-identical. Gemma's sqrt(hidden)
  // multiplier in the same slot would carry the factor straight through.
  TinySpec s;
  s.rms_eps = 0.0;  // eps would break exact scale invariance
  s.post_eps = 0.0;
  const HfConfig cfg = MakeConfig(s);
  const MuseGlimmerWeights base = TinyWeights(cfg);
  MuseGlimmerWeights scaled = base;
  ScaleRowsPow2(&scaled.embed_tokens, 0, s.vocab, s.hidden, 4.0f);
  CHECK_FALSE(Differs(RunForward(base, Positions()), RunForward(scaled, Positions())));
}

TEST_CASE("muse_glimmer text: QK-norm is applied (q/k projection rescale is inert)") {
  // The weightless per-head QK-norm removes the magnitude of q and k, so scaling
  // BOTH projections by an exact power of two cannot move a single logit. Drop the
  // QK-norm and every attention score scales by 16.
  TinySpec s;
  s.rms_eps = 0.0;
  s.post_eps = 0.0;
  const HfConfig cfg = MakeConfig(s);
  const MuseGlimmerWeights base = TinyWeights(cfg);
  const int64_t qdim = s.heads * s.head_dim, kdim = s.kv_heads * s.head_dim;
  MuseGlimmerWeights scaled = base;
  for (auto& lw : scaled.layers)
    ScaleRowsPow2(&lw.attn.qkv_proj, 0, qdim + kdim, s.hidden, 4.0f);  // q and k only
  CHECK_FALSE(Differs(RunForward(base, Positions()), RunForward(scaled, Positions())));

  // THE CONTROL, so the invariance above is not vacuous. Rescaling ONE q output row
  // changes the DIRECTION of a query head vector, which the per-head RMSNorm cannot
  // undo, so it MUST move the logits — proving this harness really does mutate the
  // weights the forward reads.
  //
  // (Note what does NOT work as a control: a uniform rescale of the V rows scales the
  // whole attention branch, and the sandwich post_attention_layernorm — being an
  // RMSNorm — erases exactly that. Measured inert, which is correct behaviour of the
  // sandwich, not an inert harness.)
  MuseGlimmerWeights qrow = base;
  for (auto& lw : qrow.layers) ScaleRowsPow2(&lw.attn.qkv_proj, 0, 1, s.hidden, 4.0f);
  CHECK(Differs(RunForward(base, Positions()), RunForward(qrow, Positions())));
}

TEST_CASE("muse_glimmer text: the query pre-scale is applied and is a config constant") {
  TinySpec a;
  TinySpec b = a;
  b.qk_scale_factor = 0.5;  // also below sqrt(head_dim) => used as-is
  const MuseGlimmerWeights wa = TinyWeights(MakeConfig(a));
  const MuseGlimmerWeights wb = TinyWeights(MakeConfig(b));
  REQUIRE(wa.params.text.scale_query_by == doctest::Approx(1.75));
  REQUIRE(wb.params.text.scale_query_by == doctest::Approx(0.5));
  CHECK(Differs(RunForward(wa, Positions()), RunForward(wb, Positions())));
}

TEST_CASE("muse_glimmer text: iRoPE — NoPE layers are position-invariant") {
  // An ALL-NoPE model has nothing position-dependent left, so REMAPPING `positions`
  // must be bit-inert. An ALL-RoPE model must be sensitive to the same remap.
  //
  // The remap SPREADS the positions (stride 2) rather than shifting them: RoPE's
  // q·k depends only on the position DIFFERENCE, so a uniform shift is inert for a
  // RoPE model too (measured — that is RoPE's defining property, not a bug), and
  // only a change in the relative spacing separates the two layer classes. Masking
  // is unaffected either way: paged attention bottom-right-aligns by token index,
  // not by this vector.
  TinySpec nope;
  nope.no_rope = {0, 0};
  const MuseGlimmerWeights wn = TinyWeights(MakeConfig(nope));
  CHECK_FALSE(Differs(RunForward(wn, Positions(1)), RunForward(wn, Positions(2))));

  TinySpec rope;
  rope.no_rope = {1, 1};
  const MuseGlimmerWeights wr = TinyWeights(MakeConfig(rope));
  CHECK(Differs(RunForward(wr, Positions(1)), RunForward(wr, Positions(2))));

  // And the MIXED default (layer 0 RoPE, layer 1 NoPE) is sensitive too — the NoPE
  // layer does not neutralise the RoPE one.
  const MuseGlimmerWeights wm = TinyWeights(MakeConfig(TinySpec{}));
  CHECK(Differs(RunForward(wm, Positions(1)), RunForward(wm, Positions(2))));
}

TEST_CASE("muse_glimmer text: the sliding window travels WITH RoPE") {
  // muse_glimmer.py:1167 — `sliding_window = None if not self.use_rope`. So the
  // window must bite on an all-RoPE model and be inert on an all-NoPE one.
  TinySpec rope_narrow;
  rope_narrow.no_rope = {1, 1};
  rope_narrow.sliding = 2;
  TinySpec rope_wide = rope_narrow;
  rope_wide.sliding = 4096;  // longer than the sequence => full attention
  CHECK(Differs(RunForward(TinyWeights(MakeConfig(rope_narrow)), Positions()),
                RunForward(TinyWeights(MakeConfig(rope_wide)), Positions())));

  TinySpec nope_narrow;
  nope_narrow.no_rope = {0, 0};
  nope_narrow.sliding = 2;
  TinySpec nope_wide = nope_narrow;
  nope_wide.sliding = 4096;
  CHECK_FALSE(Differs(RunForward(TinyWeights(MakeConfig(nope_narrow)), Positions()),
                      RunForward(TinyWeights(MakeConfig(nope_wide)), Positions())));
}

// ─────────────────────────────────────────────────────────────────────────────
// THE ROPE BASE, AT A REALISTIC MAGNITUDE.
//
// COVERAGE HOLE this closes (review of #279): every case above runs at positions
// 0..4 with head_dim 4, which has exactly ONE non-trivial rotary pair. At those
// positions the angle a base of 5e5 produces (~0.006 rad at pos 4) and the angle
// 1e4 produces (~0.04 rad) differ by less than the bf16-depth reference band, so
// substituting the 1e4 default that almost every OTHER model ships — the
// archetypal "forgot to read the config" bug — left the whole file GREEN.
// Only an absurd base (2.0) went red, which gates nothing anybody would write.
//
// So this case moves to positions in the thousands, where the two bases are
// separated by radians rather than milliradians, and pins the base three ways:
// our forward matches the reference at ITS OWN base, does NOT match a reference
// built at 1e4, and cannot produce the same logits from the two configs.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("muse_glimmer text: the RoPE base is READ from the config (5e5 vs 1e4)") {
  TinySpec real;
  real.no_rope = {1, 1};   // both layers RoPE, so no NoPE layer dilutes the effect
  real.sliding = 4096;     // window wider than the sequence => every pair attends
  TinySpec wrong = real;
  wrong.rope_theta = 10000.0;  // the default every other model ships

  const MuseGlimmerWeights w_real = TinyWeights(MakeConfig(real));
  const MuseGlimmerWeights w_wrong = TinyWeights(MakeConfig(wrong));
  REQUIRE(w_real.params.text.rope_theta == doctest::Approx(500000.0));
  REQUIRE(w_wrong.params.text.rope_theta == doctest::Approx(10000.0));

  // Positions a 131072-context model actually sees. Only the SPACING matters:
  // RoPE's q·k depends on the position difference, so the base is observable
  // exactly when that difference times the frequency is O(1) rather than O(1e-3).
  const std::vector<int32_t> pos = {0, 1024, 2048, 3072, 4096};
  const std::vector<float> got = RunForward(w_real, pos);
  for (float x : got) REQUIRE(std::isfinite(x));

  // The weights are seeded from the spec's shape, not its theta, so both arms read
  // BYTE-IDENTICAL tensors and the base is the only thing that differs.
  const double good = MaxAbsDiff(got, RefForward(w_real, pos));
  const double bad = MaxAbsDiff(got, RefForward(w_wrong, pos));
  MESSAGE("muse_glimmer RoPE base: max|diff| vs its own 5e5 reference " << good
          << ", vs a 1e4 reference " << bad);
  // The band here is 2e-3, not the 5e-4 the positions-0..4 case uses: at these
  // positions RoPE rotates through radians rather than milliradians, and the
  // bf16-per-op depth error grows with the rotation. MEASURED 6.9e-4 against a
  // wrong-base signal of 4.1e-2 — 59x the measured error and 20x the band — so the
  // widening buys the wrong base nothing.
  CHECK(good <= 2e-3);
  // A wrong-but-plausible base must land FAR outside the band the right one sits
  // in — not merely outside it, or the gate would ride on bf16 noise.
  CHECK(bad > 1e-2);
  CHECK(bad > 20.0 * good);

  // And end to end: two configs differing only in `rope_theta` cannot agree.
  CHECK(Differs(got, RunForward(w_wrong, pos)));
}

// ─────────────────────────────────────────────────────────────────────────────
// WHICH FUSION ARM THIS BINARY IS RUNNING.
//
// COVERAGE HOLE this closes (review of #279): `FusedChainAdoptEnabled()` defaults
// ON, so every `else vt::RmsNorm(...)` fallback in muse_glimmer.cpp (the input
// layernorm, the pre-feedforward layernorm and the final norm) was DEAD in every
// test — mutating one of them stayed green. That fallback is not decoration: it
// is what runs wherever vt::FusedChain's recipes are unavailable.
//
// tests/CMakeLists.txt therefore registers this binary TWICE, once at the default
// and once with VT_FUSED_CHAIN_ADOPT=0, so the whole file — including the fp32
// reference gate above — runs on BOTH arms. This case is what stops the second
// registration from silently re-running the first arm: the flag is read once per
// process into a function-local static, so a test that flipped the env at runtime
// would prove nothing at all.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("muse_glimmer text: this binary runs the fusion arm the environment selected") {
  const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
  const bool want_adopt = !(e != nullptr && e[0] == '0');
  const std::string env = e != nullptr ? std::string(e) : std::string("(unset)");
  const std::string arm = want_adopt ? "FusedChain ADOPT" : "hand-call FALLBACK";
  MESSAGE("VT_FUSED_CHAIN_ADOPT=" << env << " => " << arm << " arm");
  CHECK(vllm::dense_attn::FusedChainAdoptEnabled() == want_adopt);
}

TEST_CASE("muse_glimmer text: the post-norms read post_norm_eps, not rms_norm_eps") {
  // Changing ONLY post_norm_eps must change the output. If the post-norms wrongly
  // took rms_norm_eps, post_norm_eps would be dead config and this would be inert.
  TinySpec a;
  TinySpec b = a;
  b.post_eps = 0.5;
  CHECK(Differs(RunForward(TinyWeights(MakeConfig(a)), Positions()),
                RunForward(TinyWeights(MakeConfig(b)), Positions())));

  // The mirror: changing ONLY rms_norm_eps must also change the output, so neither
  // eps is the one that silently drives both.
  TinySpec c = a;
  c.rms_eps = 0.5;
  CHECK(Differs(RunForward(TinyWeights(MakeConfig(a)), Positions()),
                RunForward(TinyWeights(MakeConfig(c)), Positions())));
}

TEST_CASE("muse_glimmer text: the FINAL norm has NO +1 weight offset") {
  // model.norm is MuseGlimmerRMSNorm(hidden, eps) — weight_offset 0 (:1296), unlike
  // the four sandwich norms. Zeroing its weight must therefore zero the logits; a
  // (1+w) final norm would leave the model fully alive.
  TinySpec s;
  s.softcap = 0.0;
  const HfConfig cfg = MakeConfig(s);
  MuseGlimmerWeights w = TinyWeights(cfg);
  ZeroTensor(&w.final_norm);
  for (float x : RunForward(w, Positions())) CHECK(x == 0.0f);
}

TEST_CASE("muse_glimmer text: the attention OUTPUT GATE is applied") {
  // Zeroing output_gate_proj makes every gate sigmoid(0) = 0.5 exactly; a forward
  // that ignored the gate would be unchanged by that.
  const HfConfig cfg = MakeConfig(TinySpec{});
  const MuseGlimmerWeights base = TinyWeights(cfg);
  MuseGlimmerWeights zeroed = base;
  for (auto& lw : zeroed.layers) ZeroTensor(&lw.attn.output_gate_proj);
  CHECK(Differs(RunForward(base, Positions()), RunForward(zeroed, Positions())));
}

TEST_CASE("muse_glimmer text: output_multiplier scales the logits BEFORE the soft-cap") {
  // With the cap off the multiplier is exactly linear, so doubling it doubles every
  // logit. (With the cap on it would not — which is why the order matters.)
  TinySpec a;
  a.softcap = 0.0;
  a.out_mult = 0.5;
  TinySpec b = a;
  b.out_mult = 1.0;
  const std::vector<float> half = RunForward(TinyWeights(MakeConfig(a)), Positions());
  const std::vector<float> full = RunForward(TinyWeights(MakeConfig(b)), Positions());
  REQUIRE(half.size() == full.size());
  for (size_t i = 0; i < half.size(); ++i) CHECK(full[i] == doctest::Approx(2.0f * half[i]));
}

TEST_CASE("muse_glimmer text: the final soft-cap bounds every logit") {
  TinySpec s;
  s.softcap = 1e-3;  // small enough that the tanh actually saturates
  s.out_mult = 1.0;
  TinySpec off = s;
  off.softcap = 0.0;
  const std::vector<float> capped = RunForward(TinyWeights(MakeConfig(s)), Positions());
  const std::vector<float> raw = RunForward(TinyWeights(MakeConfig(off)), Positions());
  // `cap*tanh(y/cap)` is strictly inside the cap in exact arithmetic but ROUNDS TO the
  // cap in f32 once it saturates, so the bound is inclusive.
  for (float x : capped) CHECK(std::abs(x) <= 1e-3f);
  double raw_max = 0.0;
  for (float x : raw) raw_max = std::max(raw_max, std::abs(static_cast<double>(x)));
  CHECK(raw_max > 1e-3);  // the control: uncapped logits really do exceed the cap
}


// ─────────── synthetic checkpoint: the loader round-trip (W1 materialization) ───────────
// Mirrors the builder in test_kimi_linear_scaffold.cpp. The point is to drive the REAL
// loader — `LoadMuseGlimmerForConditionalGenerationWeights` — through
// `NormalizeMuseGlimmerWeightName` end to end, in BOTH checkpoint conventions, and to
// prove the two sandwich-norm rename hazards the header documents do not swap a tensor.
namespace {

struct Fx {
  std::string name;
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// bf16 bytes from SMALL finite floats — a raw byte fill would manufacture NaNs and the
// "loaded weights forward finite" check could not tell a load bug from a NaN input.
std::string Bf16Fill(size_t numel, int seed) {
  std::string s(numel * 2, '\0');
  auto* p = reinterpret_cast<uint16_t*>(s.data());
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
  for (size_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return s;
}

Fx Bf16Fx(const std::string& n, const std::vector<int64_t>& shape, int seed) {
  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;
  return {n, shape, Bf16Fill(static_cast<size_t>(numel), seed)};
}

std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("muse_glimmer_text_" + std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// NOTE the configs below are TEXT-ONLY (MakeConfig emits no vision_config), so the
// enumeration is exactly the text tower and `accounted == enumerated` is a meaningful
// equality rather than a partial count.

// The canonical `model.language_model.*` checkpoint. NOTE the attention gate ships as
// `.self_attn.gate_proj` on disk — the name that collides with the MLP gate by suffix
// and that the mapper must rename FIRST.
std::vector<Fx> CanonicalShards(const TinySpec& s) {
  const int64_t H = s.hidden, I = s.inter, V = s.vocab;
  const int64_t qdim = s.heads * s.head_dim, kdim = s.kv_heads * s.head_dim;
  const std::string L0 = "model.language_model.layers.0.";
  return {
      Bf16Fx("model.language_model.embed_tokens.weight", {V, H}, 1),
      Bf16Fx("model.language_model.norm.weight", {H}, 2),
      Bf16Fx("lm_head.weight", {V, H}, 3),
      Bf16Fx(L0 + "input_layernorm.weight", {H}, 11),
      Bf16Fx(L0 + "post_attention_layernorm.weight", {H}, 12),
      Bf16Fx(L0 + "pre_feedforward_layernorm.weight", {H}, 13),
      Bf16Fx(L0 + "post_feedforward_layernorm.weight", {H}, 14),
      Bf16Fx(L0 + "self_attn.q_proj.weight", {qdim, H}, 21),
      Bf16Fx(L0 + "self_attn.k_proj.weight", {kdim, H}, 22),
      Bf16Fx(L0 + "self_attn.v_proj.weight", {kdim, H}, 23),
      Bf16Fx(L0 + "self_attn.o_proj.weight", {H, qdim}, 24),
      Bf16Fx(L0 + "self_attn.gate_proj.weight", {qdim, H}, 25),
      Bf16Fx(L0 + "mlp.gate_proj.weight", {I, H}, 31),
      Bf16Fx(L0 + "mlp.up_proj.weight", {I, H}, 32),
      Bf16Fx(L0 + "mlp.down_proj.weight", {H, I}, 33),
  };
}

// The legacy `guac` checkpoint carrying the SAME bytes under the SAME seeds, but with
// the sandwich norms under their legacy names: legacy `post_attention_layernorm` is
// really the PRE-feedforward norm, legacy `post_attn_norm` the true post-attention one.
std::vector<Fx> LegacyShards(const TinySpec& s) {
  std::vector<Fx> out = CanonicalShards(s);
  for (Fx& f : out) {
    const std::string from = "model.language_model.";
    if (f.name.rfind(from, 0) == 0) f.name = "model." + f.name.substr(from.size());
  }
  for (Fx& f : out) {
    if (f.name == "model.layers.0.post_attention_layernorm.weight")
      f.name = "model.layers.0.post_attn_norm.weight";           // true post-attention
    else if (f.name == "model.layers.0.pre_feedforward_layernorm.weight")
      f.name = "model.layers.0.post_attention_layernorm.weight";  // legacy pre-FF name
    else if (f.name == "model.layers.0.post_feedforward_layernorm.weight")
      f.name = "model.layers.0.post_ffn_norm.weight";
  }
  return out;
}

const std::string& BytesOf(const std::vector<Fx>& fx, const std::string& name) {
  for (const Fx& f : fx)
    if (f.name == name) return f.bytes;
  REQUIRE_MESSAGE(false, "fixture missing " << name);
  return fx.front().bytes;
}

bool SameBytes(const OwnedTensor& t, const std::string& raw) {
  return t.bytes.size() == raw.size() &&
         std::memcmp(t.bytes.data(), raw.data(), raw.size()) == 0;
}

}  // namespace

TEST_CASE("muse_glimmer text: the loader materializes the text tower (canonical names)") {
  TinySpec s;
  s.layers = 1;
  s.no_rope = {1};
  const HfConfig cfg = MakeConfig(s);
  const std::vector<Fx> fx = CanonicalShards(s);
  const TempFile file(BuildSt(fx));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));

  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, cfg);
  REQUIRE(w.text_loaded);
  // Every enumerated tensor is present, and the enumeration is the text tower alone.
  CHECK(w.accounted_tensors == w.enumerated_tensors);
  REQUIRE(w.layers.size() == 1u);

  const int64_t H = s.hidden, I = s.inter, V = s.vocab;
  const int64_t qdim = s.heads * s.head_dim, kdim = s.kv_heads * s.head_dim;
  CHECK(w.embed_tokens.shape[0] == V);
  CHECK(w.embed_tokens.shape[1] == H);
  CHECK(w.final_norm.shape[0] == H);
  CHECK(w.lm_head.shape[0] == H);  // TRANSPOSED to Matmul-B [in,out]
  CHECK(w.lm_head.shape[1] == V);
  CHECK(w.layers[0].attn.qkv_proj.shape[0] == qdim + 2 * kdim);
  CHECK(w.layers[0].attn.o_proj.shape[0] == H);
  CHECK(w.layers[0].attn.output_gate_proj.shape[0] == qdim);
  CHECK(w.layers[0].mlp.gate_up_proj.shape[0] == 2 * I);
  CHECK(w.layers[0].mlp.down_proj.shape[0] == H);

  // The merged QKV owner carries the disk q|k|v shards IN THAT ROW ORDER. Without
  // this, swapping two same-width shards (k and v are both kdim rows) is a silent,
  // shape-valid defect: the model still runs and still produces fluent-looking text.
  const std::string L0 = "model.language_model.layers.0.";
  const std::string q_disk = BytesOf(fx, L0 + "self_attn.q_proj.weight");
  const std::string k_disk = BytesOf(fx, L0 + "self_attn.k_proj.weight");
  const std::string v_disk = BytesOf(fx, L0 + "self_attn.v_proj.weight");
  const auto* qkv = reinterpret_cast<const char*>(w.layers[0].attn.qkv_proj.bytes.data());
  CHECK(std::memcmp(qkv, q_disk.data(), q_disk.size()) == 0);
  CHECK(std::memcmp(qkv + q_disk.size(), k_disk.data(), k_disk.size()) == 0);
  CHECK(std::memcmp(qkv + q_disk.size() + k_disk.size(), v_disk.data(), v_disk.size()) == 0);

  // Same for the merged gate|up MLP owner (gate FIRST — SiluAndMul reads the first
  // half as the gate).
  const std::string up_disk = BytesOf(fx, L0 + "mlp.up_proj.weight");
  const std::string gate_disk = BytesOf(fx, L0 + "mlp.gate_proj.weight");
  const auto* gu = reinterpret_cast<const char*>(w.layers[0].mlp.gate_up_proj.bytes.data());
  CHECK(std::memcmp(gu, gate_disk.data(), gate_disk.size()) == 0);
  CHECK(std::memcmp(gu + gate_disk.size(), up_disk.data(), up_disk.size()) == 0);

  // The attention gate landed on output_gate_proj, NOT folded into the MLP gate_up:
  // its bytes are the disk `.self_attn.gate_proj` tensor, and the MLP's first half is
  // the disk `mlp.gate_proj` tensor.
  CHECK(SameBytes(w.layers[0].attn.output_gate_proj,
                  BytesOf(fx, "model.language_model.layers.0.self_attn.gate_proj.weight")));


  // And the loaded tower actually forwards.
  const std::vector<float> logits = RunForward(w, Positions());
  REQUIRE(logits.size() == static_cast<size_t>(Tokens().size()) * static_cast<size_t>(V));
  for (float x : logits) CHECK(std::isfinite(x));
}

TEST_CASE("muse_glimmer text: the legacy guac convention loads the SAME tower") {
  // The two conventions carry byte-identical tensors under different norm names. If the
  // legacy remap fired in the wrong order the post-attention and pre-feedforward norms
  // would silently swap — plausible output, wrong model.
  TinySpec s;
  s.layers = 1;
  s.no_rope = {1};
  const HfConfig cfg = MakeConfig(s);
  const std::vector<Fx> canon = CanonicalShards(s);
  const std::vector<Fx> legacy = LegacyShards(s);
  const TempFile cfile(BuildSt(canon));
  const TempFile lfile(BuildSt(legacy));
  std::vector<vllm::SafetensorsFile> cshards, lshards;
  cshards.push_back(vllm::SafetensorsFile::Open(cfile.path()));
  lshards.push_back(vllm::SafetensorsFile::Open(lfile.path()));

  const MuseGlimmerWeights cw =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(cshards, cfg);
  const MuseGlimmerWeights lw =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(lshards, cfg);
  CHECK(lw.accounted_tensors == lw.enumerated_tensors);

  // The legacy `post_attn_norm` (seed 12) must land on post_attention_layernorm, and the
  // legacy `post_attention_layernorm` (seed 13) on pre_feedforward_layernorm.
  CHECK(SameBytes(lw.layers[0].post_attention_layernorm,
                  BytesOf(legacy, "model.layers.0.post_attn_norm.weight")));
  CHECK(SameBytes(lw.layers[0].pre_feedforward_layernorm,
                  BytesOf(legacy, "model.layers.0.post_attention_layernorm.weight")));
  CHECK(SameBytes(lw.layers[0].post_attention_layernorm,
                  BytesOf(canon,
                          "model.language_model.layers.0.post_attention_layernorm.weight")));

  // End to end: the two conventions produce the same logits.
  CHECK_FALSE(Differs(RunForward(cw, Positions()), RunForward(lw, Positions())));
}

TEST_CASE("muse_glimmer text: the loader throws BY NAME on a missing tensor") {
  TinySpec s;
  s.layers = 1;
  s.no_rope = {1};
  const HfConfig cfg = MakeConfig(s);
  std::vector<Fx> fx = CanonicalShards(s);
  fx.erase(std::remove_if(fx.begin(), fx.end(),
                          [](const Fx& f) {
                            return f.name ==
                                   "model.language_model.layers.0.self_attn.k_proj.weight";
                          }),
           fx.end());
  const TempFile file(BuildSt(fx));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(file.path()));
  CHECK_THROWS(vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, cfg));
}


// ─────────── the RELEASED checkpoint's config, as a hardcoded fixture ───────────
// Transcribed from meta-models/Muse-Glimmer-30B `config.json` (read from the local
// download 2026-08-10). It is HARDCODED on purpose: the test must run in CI on a box
// with no checkpoint, and the point is to pin the field SPELLINGS and SHAPES the real
// artifact ships, which the upstream python alone does not tell you.
//
// Two things here were live defects until this fixture existed:
//   * `no_rope_layers` is ABSENT. The split is carried by `layer_rope_theta` (a 0
//     marks NoPE) and `layer_types` ("full_attention" marks the same layer). The
//     backward-counted default only HAPPENS to agree at L=52.
//   * the vision block ships `merge_size`, and NEITHER `output_dim` NOR
//     `adapter_dim` — those are top-level `out_hidden_size` / `projector_hidden_size`.
namespace {

nlohmann::json ReleasedConfigJson() {
  constexpr int64_t L = 52;
  nlohmann::json layer_types = nlohmann::json::array();
  nlohmann::json layer_rope_theta = nlohmann::json::array();
  for (int64_t i = 0; i < L; ++i) {
    const bool full = ((i + 1) % 4 == 0);  // 3, 7, 11, ... 51
    layer_types.push_back(full ? "full_attention" : "sliding_attention");
    layer_rope_theta.push_back(full ? 0.0 : 500000.0);
  }
  nlohmann::json vision_layer_types = nlohmann::json::array();
  for (int64_t i = 0; i < 50; ++i)
    vision_layer_types.push_back(((i + 1) % 4 == 0 || i == 49) ? "full_attention"
                                                               : "window_attention");
  return nlohmann::json{
      {"architectures", nlohmann::json::array({"MuseGlimmerForConditionalGeneration"})},
      {"dtype", "bfloat16"},
      {"image_token_id", 200092},
      {"video_token_id", 200091},
      {"model_type", "muse_glimmer"},
      {"out_hidden_size", 6144},
      {"projector_hidden_act", "gelu"},
      {"projector_hidden_size", 4096},
      {"text_config",
       {{"attention_bias", false},
        {"attention_dropout", 0.0},
        {"bos_token_id", 200000},
        {"eos_token_id", 200001},
        {"final_logit_softcapping", 20.0},
        {"head_dim", 128},
        {"hidden_activation", "silu"},
        {"hidden_size", 6656},
        {"initializer_range", 0.02},
        {"intermediate_size", 19968},
        {"layer_rope_theta", layer_rope_theta},
        {"layer_types", layer_types},
        {"max_position_embeddings", 131072},
        {"model_type", "muse_glimmer_text"},
        {"num_attention_heads", 32},
        {"num_hidden_layers", L},
        {"num_key_value_heads", 2},
        {"output_multiplier", 0.19611613513818404},
        {"post_norm_eps", 1e-08},
        {"qk_scale_factor", 3.87},
        {"rms_norm_eps", 1e-05},
        {"rope_parameters", {{"rope_theta", 500000.0}, {"rope_type", "default"}}},
        {"sliding_window", 2048},
        {"tie_word_embeddings", false},
        {"use_cache", true},
        {"vocab_size", 202048}}},
      {"vision_config",
       {{"hidden_act", "gelu"},
        {"hidden_size", 1536},
        {"intermediate_size", 8960},
        {"layer_norm_eps", 1e-05},
        {"layer_types", vision_layer_types},
        {"merge_size", 2},
        {"num_attention_heads", 16},
        {"num_hidden_layers", 50},
        {"patch_size", 14},
        {"patch_temporal", 2},
        {"pos_emb_height", 32},
        {"pos_emb_width", 32}}},
      {"transformers_version", "5.15.0.dev0"},
  };
}

HfConfig ReleasedConfig() {
  HfConfig c;
  c.architectures = {"MuseGlimmerForConditionalGeneration"};
  c.hidden_size = 6656;
  c.num_hidden_layers = 52;
  c.vocab_size = 202048;
  c.num_attention_heads = 32;
  c.raw = ReleasedConfigJson();
  return c;
}

}  // namespace

TEST_CASE("muse_glimmer: the RELEASED 30B config parses to the real geometry") {
  const MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(ReleasedConfig());
  const vllm::MuseGlimmerTextParams& t = p.text;

  CHECK(t.vocab_size == 202048);
  CHECK(t.hidden_size == 6656);
  CHECK(t.intermediate_size == 19968);
  CHECK(t.num_hidden_layers == 52);
  CHECK(t.num_attention_heads == 32);
  CHECK(t.num_key_value_heads == 2);
  CHECK(t.head_dim == 128);
  CHECK(t.max_position_embeddings == 131072);
  CHECK(t.sliding_window == 2048);
  CHECK(t.rope_theta == doctest::Approx(500000.0));
  CHECK_FALSE(t.tie_word_embeddings);
  CHECK(t.hidden_activation == "silu");
  CHECK(t.output_multiplier == doctest::Approx(0.19611613513818404));
  CHECK(t.final_logit_softcapping == doctest::Approx(20.0));

  // The split eps is REAL: three orders of magnitude apart in the shipped config.
  // `.scale(0)` is load-bearing on every epsilon assertion in this file: doctest's
  // Approx adds a scale term that defaults to 1.0, so a bare `Approx(1e-8)` accepts
  // 1e-5 — the exact wrong value #412 is about — as equal.
  CHECK(static_cast<double>(t.rms_norm_eps) == doctest::Approx(1e-5).scale(0.0));
  CHECK(static_cast<double>(t.post_norm_eps) == doctest::Approx(1e-8).scale(0.0));
  CHECK(t.post_norm_eps < t.rms_norm_eps);

  // The pre-folded schema at the REAL head_dim: 3.87 < sqrt(128) = 11.31, so it is
  // used as-is. The trap is dividing it again (or treating a native 43.784 as folded).
  CHECK(t.scale_query_by == doctest::Approx(3.87));

  // Both flags are ABSENT in the released config and MUST read as ON.
  CHECK(t.use_qk_norm);
  CHECK(t.use_attn_output_gate);
}

TEST_CASE("muse_glimmer: the released config's iRoPE split comes from the CHECKPOINT") {
  // `no_rope_layers` is absent; the split is derived from layer_rope_theta (0 => NoPE)
  // and layer_types ("full_attention" => NoPE), which must agree.
  const MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(ReleasedConfig());
  REQUIRE(p.text.no_rope_layers.size() == 52u);
  for (int64_t i = 0; i < 52; ++i) {
    const bool full = ((i + 1) % 4 == 0);
    CHECK(p.text.no_rope_layers[static_cast<size_t>(i)] == (full ? 0 : 1));
  }
  // It agrees with upstream's backward-counted default AT THIS DEPTH — which is why
  // the defect was invisible. The derivation is what makes that agreement a fact
  // rather than a coincidence we depend on.
  CHECK(p.text.no_rope_layers == vllm::DefaultMuseGlimmerNoRopeLayers(52));

  // A checkpoint whose two encodings DISAGREE is a config we do not understand.
  HfConfig bad = ReleasedConfig();
  bad.raw["text_config"]["layer_types"][0] = "full_attention";  // theta still 500000
  CHECK_THROWS(vllm::ParseMuseGlimmerParams(bad));

  // And a per-layer theta that disagrees with rope_parameters is rejected rather
  // than silently applied with the wrong base.
  HfConfig mixed = ReleasedConfig();
  mixed.raw["text_config"]["layer_rope_theta"][0] = 10000.0;
  CHECK_THROWS(vllm::ParseMuseGlimmerParams(mixed));

  // The derivation must SURVIVE a schedule that differs from the counted default:
  // make layer 0 NoPE in both encodings and require the mask to follow the file.
  HfConfig shifted = ReleasedConfig();
  for (int64_t i = 0; i < 52; ++i) {
    const bool full = (i % 4 == 0);  // 0, 4, 8, ... — NOT the backward default
    shifted.raw["text_config"]["layer_types"][static_cast<size_t>(i)] =
        full ? "full_attention" : "sliding_attention";
    shifted.raw["text_config"]["layer_rope_theta"][static_cast<size_t>(i)] =
        full ? 0.0 : 500000.0;
  }
  const MuseGlimmerParams sp = vllm::ParseMuseGlimmerParams(shifted);
  CHECK(sp.text.no_rope_layers[0] == 0);
  CHECK(sp.text.no_rope_layers[51] == 1);
  CHECK(sp.text.no_rope_layers != vllm::DefaultMuseGlimmerNoRopeLayers(52));
}

TEST_CASE("muse_glimmer: the released vision block's REAL field spellings resolve") {
  const MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(ReleasedConfig());
  const vllm::MuseGlimmerVisionParams& v = p.vision;
  REQUIRE(v.present);
  CHECK(v.hidden_size == 1536);
  CHECK(v.intermediate_size == 8960);
  CHECK(v.num_hidden_layers == 50);
  CHECK(v.num_attention_heads == 16);
  CHECK(v.patch_size == 14);
  CHECK(v.patch_temporal == 2);
  CHECK(v.pos_emb_height == 32);
  CHECK(v.pos_emb_width == 32);
  // `merge_size` (vision block), `out_hidden_size` and `projector_hidden_size`
  // (TOP level) are the real spellings. Reading the old names alone fell back to
  // defaults that happen to equal these, so this pins the read, not just the value.
  CHECK(v.merge_kernel_size == 2);
  CHECK(v.output_dim == 6144);
  CHECK(v.adapter_dim == 4096);
  REQUIRE(v.layer_types.size() == 50u);
  CHECK(v.layer_types[0] == "window_attention");
  CHECK(v.layer_types[3] == "full_attention");

  // The spelling read must come from the file, not from a default: change ONLY the
  // real field names and the parse must follow (and reject the mismatched shape).
  HfConfig moved = ReleasedConfig();
  moved.raw["out_hidden_size"] = 4096;  // != 1536 * 2 * 2
  CHECK_THROWS(vllm::ParseMuseGlimmerParams(moved));
  HfConfig merged = ReleasedConfig();
  merged.raw["vision_config"]["merge_size"] = 4;  // 1536*16 != 6144
  CHECK_THROWS(vllm::ParseMuseGlimmerParams(merged));
  // `adapter_dim` has no structural equation to cross-check it, so assert it TRACKS
  // the top-level field. Without this the 4096 above is equally consistent with the
  // field never being read at all (measured: it was, until this control existed).
  HfConfig proj = ReleasedConfig();
  proj.raw["projector_hidden_size"] = 2048;
  CHECK(vllm::ParseMuseGlimmerParams(proj).vision.adapter_dim == 2048);

  CHECK(p.image_token_id == 200092);
  CHECK(p.video_token_id == 200091);
}

// ── issue #412: an ABSENT key falls back to the ARCHITECTURE, not to neutral ──
// The released 30B `config.json` above carries every one of these six, which is
// exactly why their defaults went untested: the only checkpoints that hit a
// default are the ones that OMIT the key. Two of those exist and both are real
// artifacts — the released GGUF (no post-norm epsilon; see the GGUF gate) and
// the DFlash drafter's `config.json`, committed verbatim as a fixture below.
//
// Every value asserted here is a LITERAL, transcribed from the two independent
// references rather than from our own constants, so the assertion is a second
// opinion and not a restatement:
//   vLLM #51655 @ 075d645af  transformers_utils/configs/muse_glimmer.py:45,56,58,62,65,67
//   SGLang     @ 38a1bc5d2f  python/sglang/srt/configs/muse_glimmer.py:90,91,93,98,101,102
// The two AGREE on all six; there is nothing to adjudicate.
namespace {

std::string DraftFixtureDir() {
#ifdef MUSE_GLIMMER_DRAFT_FIXTURE_DIR
  return MUSE_GLIMMER_DRAFT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/muse_glimmer_30b_assistant";
#endif
}

// The DFlash drafter's config, byte-for-byte as released with
// `meta-models/Muse-Glimmer-30B-Assistant`. It is a FLAT config (no
// `text_config`) and it omits `qk_scale_factor`, `post_norm_eps`,
// `output_multiplier`, `final_logit_softcapping` and `vocab_size`.
HfConfig DrafterConfig() {
  const std::string path = DraftFixtureDir() + "/config.json";
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path);
  nlohmann::json raw;
  in >> raw;

  // The five omissions are the POINT of this fixture. If a future re-release
  // starts shipping them the defaults stop being exercised, so assert the shape
  // of the artifact before asserting anything about the parse.
  for (const char* absent : {"qk_scale_factor", "post_norm_eps", "output_multiplier",
                             "final_logit_softcapping", "vocab_size"})
    REQUIRE_MESSAGE(!raw.contains(absent), "fixture unexpectedly carries " << absent);

  HfConfig c;
  c.architectures = {"MuseGlimmerAssistantModel"};
  c.hidden_size = raw["hidden_size"].get<int64_t>();
  c.num_hidden_layers = raw["num_hidden_layers"].get<int64_t>();
  c.num_attention_heads = raw["num_attention_heads"].get<int64_t>();
  c.raw = std::move(raw);
  return c;
}

// The minimum a config can carry and still parse: the three required geometry
// fields plus head_dim. Everything else must come from the architecture.
HfConfig GeometryOnlyConfig() {
  HfConfig c;
  c.architectures = {"MuseGlimmerForCausalLM"};
  c.hidden_size = 512;
  c.num_hidden_layers = 4;
  c.num_attention_heads = 4;
  c.raw = nlohmann::json{{"model_type", "muse_glimmer"},
                         {"text_config",
                          {{"hidden_size", 512},
                           {"intermediate_size", 1024},
                           {"num_hidden_layers", 4},
                           {"num_attention_heads", 4},
                           {"num_key_value_heads", 2},
                           {"head_dim", 128}}}};
  return c;
}

}  // namespace

TEST_CASE("muse_glimmer #412: the DRAFTER config omits five keys and still gets"
          " the architecture's constants") {
  const MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(DrafterConfig());
  const vllm::MuseGlimmerTextParams& t = p.text;

  // Control — the keys the drafter DOES ship must come from the file.
  CHECK(t.hidden_size == 6656);
  CHECK(t.num_hidden_layers == 5);
  CHECK(t.num_key_value_heads == 8);  // 8, not the target's 2
  CHECK(t.head_dim == 128);
  CHECK(t.sliding_window == 2048);
  CHECK(static_cast<double>(t.rms_norm_eps) == doctest::Approx(1e-5).scale(0.0));
  CHECK(t.hidden_activation == "silu");  // via the flat `hidden_act` rename
  REQUIRE(t.no_rope_layers.size() == 5u);
  for (int64_t v : t.no_rope_layers) CHECK(v == 1);  // all "sliding_attention"

  // THE DEFECT. `use_qk_norm` is absent too and correctly reads TRUE, so the
  // drafter runs its QK-norm — and then, with a neutral 1.0 here, multiplies the
  // normed queries by nothing at all instead of by 3.87. No error, coherent
  // draft tokens, acceptance quietly collapses.
  CHECK(t.use_qk_norm);
  CHECK(t.scale_query_by == doctest::Approx(3.87).epsilon(1e-9));
  CHECK(t.scale_query_by != doctest::Approx(1.0));

  // 1e-8, NOT rms_norm_eps: the post-norms are a thousand times tighter.
  CHECK(static_cast<double>(t.post_norm_eps) == doctest::Approx(1e-8).scale(0.0));
  CHECK(static_cast<double>(t.post_norm_eps) !=
        doctest::Approx(static_cast<double>(t.rms_norm_eps)).scale(0.0));

  CHECK(t.output_multiplier == doctest::Approx(0.19611613513818404));
  CHECK(t.final_logit_softcapping == doctest::Approx(20.0));

  // `vocab_size` stays 0 DELIBERATELY. A DFlash draft has no head of its own and
  // borrows the target's, which is what SGLang encodes as
  // `MuseGlimmerAssistantConfig.vocab_size = None`
  // (srt/configs/muse_glimmer.py:28-32). 0 is the "ask the target" sentinel, not
  // a default that lost a value.
  CHECK(t.vocab_size == 0);
}

TEST_CASE("muse_glimmer #412: a config that omits EVERY defaulted key lands on"
          " the architecture, not on neutral values") {
  const MuseGlimmerParams p = vllm::ParseMuseGlimmerParams(GeometryOnlyConfig());
  const vllm::MuseGlimmerTextParams& t = p.text;

  // Each of the six, against the value the OLD neutral default produced.
  CHECK(t.scale_query_by == doctest::Approx(3.87).epsilon(1e-9));  // was 1.0
  CHECK(t.sliding_window == 2048);                                 // was 0 = none
  CHECK(static_cast<double>(t.rms_norm_eps) ==
        doctest::Approx(1e-5).scale(0.0));  // was 1e-6
  CHECK(static_cast<double>(t.post_norm_eps) ==
        doctest::Approx(1e-8).scale(0.0));  // was rms_norm_eps
  CHECK(t.output_multiplier == doctest::Approx(0.19611613513818404));  // was 1.0
  CHECK(t.final_logit_softcapping == doctest::Approx(20.0));           // was 0 = off

  // `sliding_window == 0` is the switch that turns the window OFF entirely
  // (muse_glimmer.cpp:229). Defaulting to it did not soften the window — it
  // deleted it, turning all 52 layers into global attention.
  CHECK(t.sliding_window > 0);
  // and 0 is likewise the "no soft-cap" sentinel, so the old default silently
  // dropped the tanh cap at :1618-1621.
  CHECK(t.final_logit_softcapping > 0.0);
}

TEST_CASE("muse_glimmer #412: an EXPLICIT null still means OFF") {
  // Both nullable fields are `X | None` upstream (configs/muse_glimmer.py:56,58),
  // so ABSENT and NULL are different states and only ABSENT takes the default.
  // Without this the new defaults would make a checkpoint that deliberately
  // disables the window or the soft-cap impossible to express.
  HfConfig off = GeometryOnlyConfig();
  off.raw["text_config"]["sliding_window"] = nullptr;
  off.raw["text_config"]["final_logit_softcapping"] = nullptr;
  const vllm::MuseGlimmerTextParams& t = vllm::ParseMuseGlimmerParams(off).text;
  CHECK(t.sliding_window == 0);
  CHECK(t.final_logit_softcapping == doctest::Approx(0.0));

  // The FLAT layout must agree — the hoist has to carry an explicit null through
  // rather than dropping the key and re-defaulting it.
  HfConfig flat;
  flat.architectures = {"MuseGlimmerForCausalLM"};
  flat.hidden_size = 512;
  flat.num_hidden_layers = 4;
  flat.num_attention_heads = 4;
  flat.raw = nlohmann::json{{"model_type", "muse_glimmer"},
                            {"hidden_size", 512},
                            {"intermediate_size", 1024},
                            {"num_hidden_layers", 4},
                            {"num_attention_heads", 4},
                            {"num_key_value_heads", 2},
                            {"head_dim", 128},
                            {"sliding_window", nullptr},
                            {"final_logit_softcapping", nullptr}};
  const vllm::MuseGlimmerTextParams& ft = vllm::ParseMuseGlimmerParams(flat).text;
  CHECK(ft.sliding_window == 0);
  CHECK(ft.final_logit_softcapping == doctest::Approx(0.0));
  // ...while the same flat config WITHOUT the keys takes the defaults.
  HfConfig flat_absent = flat;
  flat_absent.raw.erase("sliding_window");
  flat_absent.raw.erase("final_logit_softcapping");
  const vllm::MuseGlimmerTextParams& at =
      vllm::ParseMuseGlimmerParams(flat_absent).text;
  CHECK(at.sliding_window == 2048);
  CHECK(at.final_logit_softcapping == doctest::Approx(20.0));
}
