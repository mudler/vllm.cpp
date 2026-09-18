// GLiNER2 boundary head host reference. See gliner2.h for upstream anchors.
#include "vllm/model_executor/models/gliner2.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace gliner2 {

namespace {

// ── Shared math helpers (same conventions as deberta_v2.cpp) ──────────────

void LinearRow(const float* x, const std::vector<float>& w,
               const std::vector<float>& b, int64_t in_dim, int64_t out_dim,
               float* out) {
  for (int64_t o = 0; o < out_dim; ++o) {
    double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
    const float* row = w.data() + static_cast<size_t>(o * in_dim);
    for (int64_t i = 0; i < in_dim; ++i) {
      acc += static_cast<double>(row[i]) * static_cast<double>(x[i]);
    }
    out[o] = static_cast<float>(acc);
  }
}

void LayerNormRow(const float* x, const std::vector<float>& w,
                  const std::vector<float>& b, int64_t n, double eps,
                  float* out) {
  double mean = 0.0;
  for (int64_t i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  const double inv = 1.0 / std::sqrt(var + eps);
  for (int64_t i = 0; i < n; ++i) {
    const double normed = (static_cast<double>(x[i]) - mean) * inv;
    out[i] = static_cast<float>(
        normed * static_cast<double>(w[static_cast<size_t>(i)]) +
        static_cast<double>(b[static_cast<size_t>(i)]));
  }
}

// SiLU/Swish: x * sigmoid(x).
double Silu(double x) {
  return x / (1.0 + std::exp(-x));
}

}  // namespace

// ── Weight loading ──────────────────────────────────────────────────────────

BoundaryHeadWeights LoadBoundaryHead(const BoundaryParams& params,
                                     const deberta_v2::CheckpointTensors& tensors) {
  BoundaryHeadWeights hw;

  // BoundaryEncoder
  auto& enc = hw.encoder;
  enc.left_proj_weight = tensors.Get("boundary_head.boundary_encoder.left_projection.weight");
  enc.left_proj_bias = tensors.Get("boundary_head.boundary_encoder.left_projection.bias");
  enc.right_proj_weight = tensors.Get("boundary_head.boundary_encoder.right_projection.weight");
  enc.right_proj_bias = tensors.Get("boundary_head.boundary_encoder.right_projection.bias");
  enc.output_proj_weight = tensors.Get("boundary_head.boundary_encoder.output_projection.weight");
  enc.output_proj_bias = tensors.Get("boundary_head.boundary_encoder.output_projection.bias");
  enc.layer_norm_weight = tensors.Get("boundary_head.boundary_encoder.layer_norm.weight");
  enc.layer_norm_bias = tensors.Get("boundary_head.boundary_encoder.layer_norm.bias");
  enc.bos_state = tensors.Get("boundary_head.boundary_encoder.bos_state");
  enc.eos_state = tensors.Get("boundary_head.boundary_encoder.eos_state");

  enc.attention_blocks.resize(static_cast<size_t>(params.boundary_attention_layers));
  for (int64_t i = 0; i < params.boundary_attention_layers; ++i) {
    const std::string p = "boundary_head.boundary_encoder.attention_blocks." +
                           std::to_string(i) + ".";
    auto& blk = enc.attention_blocks[static_cast<size_t>(i)];
    blk.norm_weight = tensors.Get(p + "norm.weight");
    blk.norm_bias = tensors.Get(p + "norm.bias");
    blk.qkv_weight = tensors.Get(p + "qkv_projection.weight");
    blk.qkv_bias = tensors.Get(p + "qkv_projection.bias");
    blk.output_weight = tensors.Get(p + "output_projection.weight");
    blk.output_bias = tensors.Get(p + "output_projection.bias");
  }

  enc.refinement_blocks.resize(static_cast<size_t>(params.boundary_refinement_layers));
  for (int64_t i = 0; i < params.boundary_refinement_layers; ++i) {
    const std::string p = "boundary_head.boundary_encoder.refinement_blocks." +
                           std::to_string(i) + ".";
    auto& blk = enc.refinement_blocks[static_cast<size_t>(i)];
    blk.norm_weight = tensors.Get(p + "norm.weight");
    blk.norm_bias = tensors.Get(p + "norm.bias");
    blk.input_weight = tensors.Get(p + "input_projection.weight");
    blk.input_bias = tensors.Get(p + "input_projection.bias");
    blk.output_weight = tensors.Get(p + "output_projection.weight");
    blk.output_bias = tensors.Get(p + "output_projection.bias");
  }

  // BoundaryQueryHead
  auto& qh = hw.query_head;
  qh.start_boundary_weight = tensors.Get("boundary_head.boundary_query_head.start_boundary_projection.weight");
  qh.start_boundary_bias = tensors.Get("boundary_head.boundary_query_head.start_boundary_projection.bias");
  qh.end_boundary_weight = tensors.Get("boundary_head.boundary_query_head.end_boundary_projection.weight");
  qh.end_boundary_bias = tensors.Get("boundary_head.boundary_query_head.end_boundary_projection.bias");
  qh.start_query_weight = tensors.Get("boundary_head.boundary_query_head.start_query_projection.weight");
  qh.start_query_bias = tensors.Get("boundary_head.boundary_query_head.start_query_projection.bias");
  qh.end_query_weight = tensors.Get("boundary_head.boundary_query_head.end_query_projection.weight");
  qh.end_query_bias = tensors.Get("boundary_head.boundary_query_head.end_query_projection.bias");
  qh.inside_text_weight = tensors.Get("boundary_head.boundary_query_head.inside_text_projection.weight");
  qh.inside_text_bias = tensors.Get("boundary_head.boundary_query_head.inside_text_projection.bias");
  qh.inside_query_weight = tensors.Get("boundary_head.boundary_query_head.inside_query_projection.weight");
  qh.inside_query_bias = tensors.Get("boundary_head.boundary_query_head.inside_query_projection.bias");

  return hw;
}

// ── BoundaryEncoder forward ────────────────────────────────────────────────
//
// text_states: [L, H]  →  boundary_states: [L+1, d]
//
// For boundary position i (0..L):
//   left[i]  = bos_state      if i==0 else text_states[i-1]    [H]
//   right[i] = eos_state      if i==L else text_states[i]      [H]
//   left_proj  = Linear(left,  left_projection)                 [d]
//   right_proj = Linear(right, right_projection)                [d]
//   concat     = [left_proj | right_proj]                       [2d]
//   out        = Linear(concat, output_projection)              [d]
//   normed     = LayerNorm(out, layer_norm)                     [d]
//   → attention_blocks (pre-norm self-attention with sliding window)
//   → refinement_blocks (pre-norm ResidualSwiGLU)

std::vector<float> BoundaryEncoderForward(const BoundaryParams& params,
                                           const BoundaryEncoderWeights& w,
                                           const std::vector<float>& text_states,
                                           int64_t seq_len) {
  const int64_t d = params.boundary_dim;
  const int64_t H = params.hidden_size;
  const int64_t B = seq_len + 1;  // L+1 boundary positions
  const int64_t heads = params.boundary_attention_heads;
  const int64_t hd = params.head_dim();
  const int64_t window = params.boundary_attention_window;
  const int64_t ffn = params.ffn_dim();
  const double eps = params.layer_norm_eps;

  // ── Build left/right states [B, H] ────────────────────────────────────
  std::vector<float> left_states(static_cast<size_t>(B * H));
  std::vector<float> right_states(static_cast<size_t>(B * H));
  for (int64_t i = 0; i < B; ++i) {
    const float* lsrc = (i == 0) ? w.bos_state.data()
                                  : text_states.data() + static_cast<size_t>((i - 1) * H);
    const float* rsrc = (i == seq_len) ? w.eos_state.data()
                                        : text_states.data() + static_cast<size_t>(i * H);
    std::copy(lsrc, lsrc + H, left_states.data() + static_cast<size_t>(i * H));
    std::copy(rsrc, rsrc + H, right_states.data() + static_cast<size_t>(i * H));
  }

  // ── Projections ────────────────────────────────────────────────────────
  std::vector<float> left_proj(static_cast<size_t>(B * d));
  std::vector<float> right_proj(static_cast<size_t>(B * d));
  std::vector<float> concat(static_cast<size_t>(B * 2 * d));
  std::vector<float> out(static_cast<size_t>(B * d));

  for (int64_t i = 0; i < B; ++i) {
    LinearRow(&left_states[static_cast<size_t>(i * H)], w.left_proj_weight,
              w.left_proj_bias, H, d, &left_proj[static_cast<size_t>(i * d)]);
    LinearRow(&right_states[static_cast<size_t>(i * H)], w.right_proj_weight,
              w.right_proj_bias, H, d, &right_proj[static_cast<size_t>(i * d)]);
    // concat = [left_proj | right_proj]
    float* c = &concat[static_cast<size_t>(i * 2 * d)];
    std::copy(left_proj.data() + static_cast<size_t>(i * d),
              left_proj.data() + static_cast<size_t>(i * d + d), c);
    std::copy(right_proj.data() + static_cast<size_t>(i * d),
              right_proj.data() + static_cast<size_t>(i * d + d), c + d);
    // output projection: 2d → d
    LinearRow(c, w.output_proj_weight, w.output_proj_bias, 2 * d, d,
              &out[static_cast<size_t>(i * d)]);
    // LayerNorm
    LayerNormRow(&out[static_cast<size_t>(i * d)], w.layer_norm_weight,
                 w.layer_norm_bias, d, eps, &out[static_cast<size_t>(i * d)]);
  }

  // ── Attention blocks (pre-norm) ────────────────────────────────────────
  std::vector<float> normed(static_cast<size_t>(B * d));
  std::vector<float> qkv(static_cast<size_t>(B * 3 * d));
  std::vector<float> attn_out(static_cast<size_t>(B * d));
  std::vector<float> proj(static_cast<size_t>(B * d));

  for (const auto& blk : w.attention_blocks) {
    // Pre-norm
    for (int64_t i = 0; i < B; ++i) {
      LayerNormRow(&out[static_cast<size_t>(i * d)], blk.norm_weight, blk.norm_bias,
                   d, eps, &normed[static_cast<size_t>(i * d)]);
    }
    // QKV projection: [B, d] → [B, 3d]
    for (int64_t i = 0; i < B; ++i) {
      LinearRow(&normed[static_cast<size_t>(i * d)], blk.qkv_weight, blk.qkv_bias,
                d, 3 * d, &qkv[static_cast<size_t>(i * 3 * d)]);
    }
    // Multi-head self-attention with sliding window mask
    // Q = qkv[:, 0:d], K = qkv[:, d:2d], V = qkv[:, 2d:3d]
    const double attn_scale =
        1.0 / std::sqrt(static_cast<double>(hd));
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t i = 0; i < B; ++i) {
        const float* q = &qkv[static_cast<size_t>(i * 3 * d + head * hd)];
        // Compute attention scores for positions within the window
        std::vector<double> scores(static_cast<size_t>(B));
        double max_score = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < B; ++j) {
          // Sliding window mask: |i - j| <= window/2
          if (std::abs(i - j) > window / 2) {
            scores[static_cast<size_t>(j)] = -std::numeric_limits<double>::infinity();
            continue;
          }
          const float* k = &qkv[static_cast<size_t>(j * 3 * d + d + head * hd)];
          double dot = 0.0;
          for (int64_t dd = 0; dd < hd; ++dd) {
            dot += static_cast<double>(q[dd]) * static_cast<double>(k[dd]);
          }
          scores[static_cast<size_t>(j)] = dot * attn_scale;
          max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
        }
        // Softmax
        double denom = 0.0;
        std::vector<double> probs(static_cast<size_t>(B));
        for (int64_t j = 0; j < B; ++j) {
          if (scores[static_cast<size_t>(j)] ==
              -std::numeric_limits<double>::infinity()) {
            probs[static_cast<size_t>(j)] = 0.0;
          } else {
            probs[static_cast<size_t>(j)] =
                std::exp(scores[static_cast<size_t>(j)] - max_score);
            denom += probs[static_cast<size_t>(j)];
          }
        }
        // Weighted sum of V
        for (int64_t dd = 0; dd < hd; ++dd) {
          double acc = 0.0;
          for (int64_t j = 0; j < B; ++j) {
            const float* v =
                &qkv[static_cast<size_t>(j * 3 * d + 2 * d + head * hd)];
            acc += probs[static_cast<size_t>(j)] * static_cast<double>(v[dd]);
          }
          attn_out[static_cast<size_t>(i * d + head * hd + dd)] =
              static_cast<float>(acc / denom);
        }
      }
    }
    // Output projection + residual
    for (int64_t i = 0; i < B; ++i) {
      LinearRow(&attn_out[static_cast<size_t>(i * d)], blk.output_weight,
                blk.output_bias, d, d, &proj[static_cast<size_t>(i * d)]);
      for (int64_t dd = 0; dd < d; ++dd) {
        out[static_cast<size_t>(i * d + dd)] += proj[static_cast<size_t>(i * d + dd)];
      }
    }
  }

  // ── Refinement blocks (pre-norm ResidualSwiGLU) ────────────────────────
  // SwiGLU(x) = (Silu(xW_in[:, :ffn/2]) * xW_in[:, ffn/2:]) W_out
  const int64_t half_ffn = ffn / 2;
  std::vector<float> ffn_buf(static_cast<size_t>(B * ffn));
  std::vector<float> gated(static_cast<size_t>(B * half_ffn));

  for (const auto& blk : w.refinement_blocks) {
    // Pre-norm
    for (int64_t i = 0; i < B; ++i) {
      LayerNormRow(&out[static_cast<size_t>(i * d)], blk.norm_weight, blk.norm_bias,
                   d, eps, &normed[static_cast<size_t>(i * d)]);
    }
    // Input projection: [B, d] → [B, ffn]
    for (int64_t i = 0; i < B; ++i) {
      LinearRow(&normed[static_cast<size_t>(i * d)], blk.input_weight, blk.input_bias,
                d, ffn, &ffn_buf[static_cast<size_t>(i * ffn)]);
    }
    // SwiGLU gate: gate = Silu(ffn[:, :half]), value = ffn[:, half:]
    // gated = gate * value
    for (int64_t i = 0; i < B; ++i) {
      for (int64_t j = 0; j < half_ffn; ++j) {
        double gate = Silu(static_cast<double>(
            ffn_buf[static_cast<size_t>(i * ffn + j)]));
        double val = static_cast<double>(
            ffn_buf[static_cast<size_t>(i * ffn + half_ffn + j)]);
        gated[static_cast<size_t>(i * half_ffn + j)] =
            static_cast<float>(gate * val);
      }
    }
    // Output projection: [B, half_ffn] → [B, d] + residual
    for (int64_t i = 0; i < B; ++i) {
      LinearRow(&gated[static_cast<size_t>(i * half_ffn)], blk.output_weight,
                blk.output_bias, half_ffn, d, &proj[static_cast<size_t>(i * d)]);
      for (int64_t dd = 0; dd < d; ++dd) {
        out[static_cast<size_t>(i * d + dd)] += proj[static_cast<size_t>(i * d + dd)];
      }
    }
  }

  return out;  // [B, d] = [L+1, d]
}

// ── BoundaryQueryHead forward ───────────────────────────────────────────────
//
// boundary_states: [L+1, d]   (from BoundaryEncoderForward)
// query_states:    [Q, H]     (encoder hidden states for the entity-type queries)
//
// start_boundary = Linear(boundary_states, start_boundary_proj)   [L+1, d]
// end_boundary   = Linear(boundary_states, end_boundary_proj)     [L+1, d]
// start_query    = Linear(query_states, start_query_proj)         [Q, d]
// end_query      = Linear(query_states, end_query_proj)           [Q, d]
// inside_text    = Linear(text_states, inside_text_proj)          [L, d]
// inside_query   = Linear(query_states, inside_query_proj)        [Q, d]
//
// start_logits[q, i] = dot(start_boundary[i], start_query[q]) * scale   [Q, L+1]
// end_logits[q, i]   = dot(end_boundary[i], end_query[q]) * scale       [Q, L+1]
// inside_logits[q, t] = dot(inside_text[t], inside_query[q]) * scale     [Q, L]
// inside_prefix[q, i] = cumsum(inside_logits[q, :i])                      [Q, L+1]
//   computed in fp32, mean-centered per query

BoundaryMarginals BoundaryQueryHeadForward(
    const BoundaryParams& params, const BoundaryQueryHeadWeights& w,
    const std::vector<float>& boundary_states, int64_t seq_len,
    const std::vector<float>& text_states,
    const std::vector<float>& query_states, int64_t num_queries) {
  const int64_t d = params.boundary_dim;
  const int64_t H = params.hidden_size;
  const int64_t B = seq_len + 1;
  const double scale = params.scale();

  BoundaryMarginals m;
  m.start_logits.resize(static_cast<size_t>(num_queries * B));
  m.end_logits.resize(static_cast<size_t>(num_queries * B));
  m.inside_logits.resize(static_cast<size_t>(num_queries * seq_len));
  m.inside_prefix.resize(static_cast<size_t>(num_queries * B));

  // Project boundary states
  std::vector<float> start_boundary(static_cast<size_t>(B * d));
  std::vector<float> end_boundary(static_cast<size_t>(B * d));
  for (int64_t i = 0; i < B; ++i) {
    LinearRow(&boundary_states[static_cast<size_t>(i * d)],
              w.start_boundary_weight, w.start_boundary_bias, d, d,
              &start_boundary[static_cast<size_t>(i * d)]);
    LinearRow(&boundary_states[static_cast<size_t>(i * d)],
              w.end_boundary_weight, w.end_boundary_bias, d, d,
              &end_boundary[static_cast<size_t>(i * d)]);
  }

  // Project query states
  std::vector<float> start_query(static_cast<size_t>(num_queries * d));
  std::vector<float> end_query(static_cast<size_t>(num_queries * d));
  std::vector<float> inside_query(static_cast<size_t>(num_queries * d));
  for (int64_t q = 0; q < num_queries; ++q) {
    LinearRow(&query_states[static_cast<size_t>(q * H)],
              w.start_query_weight, w.start_query_bias, H, d,
              &start_query[static_cast<size_t>(q * d)]);
    LinearRow(&query_states[static_cast<size_t>(q * H)],
              w.end_query_weight, w.end_query_bias, H, d,
              &end_query[static_cast<size_t>(q * d)]);
    LinearRow(&query_states[static_cast<size_t>(q * H)],
              w.inside_query_weight, w.inside_query_bias, H, d,
              &inside_query[static_cast<size_t>(q * d)]);
  }

  // Start/end logits: dot product of boundary projections with query projections
  for (int64_t q = 0; q < num_queries; ++q) {
    const float* sq = &start_query[static_cast<size_t>(q * d)];
    const float* eq = &end_query[static_cast<size_t>(q * d)];
    for (int64_t i = 0; i < B; ++i) {
      const float* sb = &start_boundary[static_cast<size_t>(i * d)];
      const float* eb = &end_boundary[static_cast<size_t>(i * d)];
      double s_dot = 0.0, e_dot = 0.0;
      for (int64_t dd = 0; dd < d; ++dd) {
        s_dot += static_cast<double>(sb[dd]) * static_cast<double>(sq[dd]);
        e_dot += static_cast<double>(eb[dd]) * static_cast<double>(eq[dd]);
      }
      m.start_logits[static_cast<size_t>(q * B + i)] =
          static_cast<float>(s_dot * scale);
      m.end_logits[static_cast<size_t>(q * B + i)] =
          static_cast<float>(e_dot * scale);
    }
  }

  // Inside logits: project text_states [L, H] → inside_text [L, d], then
  // dot with inside_query [Q, d].
  std::vector<float> inside_text(static_cast<size_t>(seq_len * d));
  for (int64_t t = 0; t < seq_len; ++t) {
    LinearRow(&text_states[static_cast<size_t>(t * H)],
              w.inside_text_weight, w.inside_text_bias, H, d,
              &inside_text[static_cast<size_t>(t * d)]);
  }
  for (int64_t q = 0; q < num_queries; ++q) {
    const float* iq = &inside_query[static_cast<size_t>(q * d)];
    for (int64_t t = 0; t < seq_len; ++t) {
      const float* it = &inside_text[static_cast<size_t>(t * d)];
      double dot = 0.0;
      for (int64_t dd = 0; dd < d; ++dd) {
        dot += static_cast<double>(it[dd]) * static_cast<double>(iq[dd]);
      }
      m.inside_logits[static_cast<size_t>(q * seq_len + t)] =
          static_cast<float>(dot * scale);
    }
  }

  // Inside prefix: cumsum of inside_logits, fp32, mean-centered per query.
  // inside_prefix[q, 0] = 0, inside_prefix[q, i] = sum(inside_logits[q, 0:i]).
  for (int64_t q = 0; q < num_queries; ++q) {
    double prefix_sum = 0.0;
    m.inside_prefix[static_cast<size_t>(q * B + 0)] = 0.0F;
    for (int64_t t = 0; t < seq_len; ++t) {
      prefix_sum += static_cast<double>(
          m.inside_logits[static_cast<size_t>(q * seq_len + t)]);
      m.inside_prefix[static_cast<size_t>(q * B + t + 1)] =
          static_cast<float>(prefix_sum);
    }
    double mean = 0.0;
    for (int64_t i = 0; i < B; ++i) {
      mean += static_cast<double>(m.inside_prefix[static_cast<size_t>(q * B + i)]);
    }
    mean /= static_cast<double>(B);
    for (int64_t i = 0; i < B; ++i) {
      m.inside_prefix[static_cast<size_t>(q * B + i)] -= static_cast<float>(mean);
    }
  }

  return m;
}

}  // namespace gliner2
}  // namespace vllm
