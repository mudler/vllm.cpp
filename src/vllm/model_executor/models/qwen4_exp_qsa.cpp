// Qwen4-Exp W4 — Qwen Sparse Attention. Host reference implementations.
// See qwen4_exp_qsa.h for the full port map, file:line on BOTH sides.
#include "vllm/model_executor/models/qwen4_exp_qsa.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "vt/dtype.h"  // VT_CHECK, F32ToBF16, BF16ToF32

namespace vllm::qwen4_exp {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// One bf16 round trip. Upstream's `.type_as(x)` / `.to(raw_keys.dtype)` on a
// bf16 model path, and a bf16 elementwise op's store, both do exactly this.
// `vt::F32ToBF16` rounds to nearest even, which is what torch does.
inline float Bf16(float x) { return vt::BF16ToF32(vt::F32ToBF16(x)); }
inline float MaybeBf16(float x, bool round) { return round ? Bf16(x) : x; }

}  // namespace

// ── Config ───────────────────────────────────────────────────────────────────

void QsaValidateConfig(const QsaConfig& cfg) {
  // configuration_qwen4_exp.py:219-220 — every QSA field must be positive.
  VT_CHECK(cfg.index_n_heads > 0 && cfg.index_kv_heads > 0 &&
               cfg.index_head_dim > 0 && cfg.token_budget > 0 &&
               cfg.compress_ratio > 0,
           "QSA config values must be positive");
  // :221-222 — "Qwen4-Exp QSA requires indexer_kv_heads=1."
  VT_CHECK(cfg.index_kv_heads == 1, "Qwen4-Exp QSA requires indexer_kv_heads=1");
  // :223-224 — "indexer_budget must be divisible by indexer_compress_ratio."
  VT_CHECK(cfg.token_budget % cfg.compress_ratio == 0,
           "indexer_budget must be divisible by indexer_compress_ratio");
  // :225-231 — rotary_dim = int(head_dim * partial_rotary_factor) must FIT the
  // index head, because the indexer ropes the leading rotary_dim of an
  // indexer_head_dim-wide vector.
  VT_CHECK(cfg.rotary_dim >= 0 && cfg.rotary_dim <= cfg.index_head_dim,
           "Qwen4-Exp attention RoPE dimensions must fit the QSA index head");
  VT_CHECK(cfg.rotary_dim % 2 == 0, "rotary_dim must be even (rotate_half)");
}

// ── The side cache ───────────────────────────────────────────────────────────

int64_t QsaSideCacheSpec::StatesForTokens(int64_t num_tokens) const {
  VT_CHECK(tokens_per_state > 0, "tokens_per_state must be positive");
  VT_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  // FLOOR, not ceil: `(position + 1) % COMPRESS_RATIO != 0` early-exits the
  // compressor (fused_compress_quant_cache.py:729-731), so an incomplete
  // trailing block writes no state. It is attended from the raw KV instead.
  return num_tokens / tokens_per_state;
}

int64_t QsaSideCacheSpec::BytesPerTokenPerLayer() const {
  VT_CHECK(tokens_per_state > 0 && head_size > 0 && elem_bytes > 0 &&
               num_kv_heads > 0,
           "side cache spec must be positive");
  // KEY-ONLY: one vector per state, not 2x for K+V. That is what
  // MLAAttentionSpec buys here, and it is why this is a quarter of the cost of
  // a per-token index cache.
  const int64_t per_state = elem_bytes * num_kv_heads * head_size;
  VT_CHECK(per_state % tokens_per_state == 0,
           "side cache state size must divide evenly across its tokens");
  return per_state / tokens_per_state;
}

int64_t QsaSideCacheSpec::BytesForTokens(int64_t num_tokens) const {
  VT_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
  return BytesPerTokenPerLayer() * num_tokens;
}

QsaSideCacheSpec QsaMakeSideCacheSpec(const QsaConfig& cfg, int64_t elem_bytes) {
  QsaValidateConfig(cfg);
  VT_CHECK(elem_bytes > 0, "elem_bytes must be positive");
  QsaSideCacheSpec spec;
  spec.num_kv_heads = cfg.index_kv_heads;  // 1, and the config gate enforces it
  spec.head_size = cfg.index_head_dim;
  spec.tokens_per_state = cfg.compress_ratio;
  spec.elem_bytes = elem_bytes;
  return spec;
}

int64_t QsaCompressedSlot(int64_t position, int64_t compress_ratio) {
  VT_CHECK(compress_ratio > 0, "compress_ratio must be positive");
  VT_CHECK(position >= 0, "position must be non-negative");
  // compressor_utils.py:50,:61 — `is_valid = (pos + 1) % COMPRESS_RATIO == 0`,
  // and PAD_ID (-1) everywhere else.
  if ((position + 1) % compress_ratio != 0) return -1;
  return position / compress_ratio;  // :51 `pos_after_compress`
}

// ── Primitives ───────────────────────────────────────────────────────────────

std::vector<float> QsaRmsNorm(const std::vector<float>& x, int64_t rows,
                              int64_t dim, const std::vector<float>& weight,
                              float eps, bool round_to_bf16) {
  VT_CHECK(rows >= 0 && dim > 0, "bad rmsnorm shape");
  VT_CHECK(static_cast<int64_t>(x.size()) == rows * dim, "x size mismatch");
  VT_CHECK(static_cast<int64_t>(weight.size()) == dim, "weight size mismatch");
  std::vector<float> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = &x[r * dim];
    // `x.pow(2).mean(-1)`, in float32 as `_norm(x.float())` demands.
    float acc = 0.0f;
    for (int64_t d = 0; d < dim; ++d) acc += src[d] * src[d];
    const float rrms = 1.0f / std::sqrt(acc / static_cast<float>(dim) + eps);
    for (int64_t d = 0; d < dim; ++d) {
      // `out * (1.0 + weight)`, NOT vLLM's `out * weight`. The weight is
      // zero-initialised upstream, so dropping the +1 turns the norm off
      // entirely and is invisible on a freshly built module.
      const float v = src[d] * rrms * (1.0f + weight[d]);
      // A single rounding, at the `.type_as(x)` on the way out.
      out[r * dim + d] = MaybeBf16(v, round_to_bf16);
    }
  }
  return out;
}

std::vector<float> QsaApplyRotaryLeadingHalf(const std::vector<float>& x,
                                             int64_t rows, int64_t head_dim,
                                             const std::vector<float>& cos,
                                             const std::vector<float>& sin,
                                             int64_t rotary_dim,
                                             bool round_to_bf16) {
  VT_CHECK(rows >= 0 && head_dim > 0, "bad rope shape");
  VT_CHECK(rotary_dim >= 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0,
           "rotary_dim must be even and fit head_dim");
  VT_CHECK(static_cast<int64_t>(x.size()) == rows * head_dim, "x size mismatch");
  VT_CHECK(static_cast<int64_t>(cos.size()) == rows * rotary_dim &&
               static_cast<int64_t>(sin.size()) == rows * rotary_dim,
           "cos/sin size mismatch");
  const int64_t half = rotary_dim / 2;
  std::vector<float> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = &x[r * head_dim];
    const float* c = &cos[r * rotary_dim];
    const float* s = &sin[r * rotary_dim];
    for (int64_t d = 0; d < rotary_dim; ++d) {
      // `rotate_half`: cat(-x2, x1) over the ROTARY span only.
      const float rot = d < half ? -src[d + half] : src[d - half];
      // Two products then a sum. On a bf16 tensor each of those three ops
      // stores a bf16, so each rounds; folding them into one float32
      // expression would drift from the oracle.
      const float a = MaybeBf16(src[d] * c[d], round_to_bf16);
      const float b = MaybeBf16(rot * s[d], round_to_bf16);
      out[r * head_dim + d] = MaybeBf16(a + b, round_to_bf16);
    }
    // The NoPE dims trail and are concatenated back untouched. This is the half
    // that DeepSeek-V4's indexer puts FIRST — see the header note.
    for (int64_t d = rotary_dim; d < head_dim; ++d) {
      out[r * head_dim + d] = src[d];
    }
  }
  return out;
}

std::vector<float> QsaCompressNormRope(const std::vector<float>& raw_keys,
                                       int64_t num_keys,
                                       const std::vector<float>& k_norm_weight,
                                       const std::vector<float>& cos,
                                       const std::vector<float>& sin,
                                       const QsaConfig& cfg,
                                       bool round_to_bf16) {
  QsaValidateConfig(cfg);
  const int64_t D = cfg.index_head_dim;
  const int64_t CR = cfg.compress_ratio;
  VT_CHECK(num_keys >= 0, "num_keys must be non-negative");
  VT_CHECK(num_keys % CR == 0,
           "QsaCompressNormRope takes COMPLETE blocks only; the caller trims "
           "the ragged tail, which is attended from the raw KV instead");
  VT_CHECK(static_cast<int64_t>(raw_keys.size()) >= num_keys * D,
           "raw_keys size mismatch");
  VT_CHECK(static_cast<int64_t>(k_norm_weight.size()) == D,
           "k_layernorm weight size mismatch");
  const int64_t nb = num_keys / CR;
  VT_CHECK(static_cast<int64_t>(cos.size()) >= num_keys * cfg.rotary_dim &&
               static_cast<int64_t>(sin.size()) >= num_keys * cfg.rotary_dim,
           "cos/sin must cover every key position");

  // 1. Unweighted mean over a NON-OVERLAPPING window of compress_ratio, in
  //    float, then rounded back to the cache dtype. This is the one place the
  //    DeepSeek-V4 compressor must NOT be copied: its pool is a LEARNED softmax
  //    over an overlapping window, in the Triton
  //    `_fused_kv_compress_norm_rope_insert_indexer_attn`
  //    (fused_compress_quant_cache.py:677, softmax at :769) and again in the
  //    CuteDSL `SparseAttnCompressNormRopeStoreC4Kernel`
  //    (nvidia/ops/sparse_attn_compress_cutedsl.py:75, :1121-1130).
  std::vector<float> pooled(static_cast<size_t>(nb) * D);
  for (int64_t b = 0; b < nb; ++b) {
    for (int64_t d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int64_t i = 0; i < CR; ++i) acc += raw_keys[(b * CR + i) * D + d];
      pooled[b * D + d] =
          MaybeBf16(acc / static_cast<float>(CR), round_to_bf16);
    }
  }

  // 2. k_layernorm on the POOLED key, not on the raw ones.
  const std::vector<float> normed =
      QsaRmsNorm(pooled, nb, D, k_norm_weight, cfg.rms_norm_eps, round_to_bf16);

  // 3. RoPE at the position of the block's FIRST token. Upstream reads it as
  //    `group_starts = block_token_indices[:, 0]`; the Triton kernel computes
  //    the same value as `(position // CR) * CR`. Taking the block's LAST
  //    position instead is a silent one-block phase error.
  std::vector<float> bcos(static_cast<size_t>(nb) * cfg.rotary_dim);
  std::vector<float> bsin(bcos.size());
  for (int64_t b = 0; b < nb; ++b) {
    const int64_t start = b * CR;
    for (int64_t d = 0; d < cfg.rotary_dim; ++d) {
      bcos[b * cfg.rotary_dim + d] = cos[start * cfg.rotary_dim + d];
      bsin[b * cfg.rotary_dim + d] = sin[start * cfg.rotary_dim + d];
    }
  }
  return QsaApplyRotaryLeadingHalf(normed, nb, D, bcos, bsin, cfg.rotary_dim,
                                   round_to_bf16);
}

std::vector<float> QsaBlockScores(const std::vector<float>& q,
                                  const std::vector<float>& block_keys,
                                  int64_t num_blocks, const QsaConfig& cfg) {
  QsaValidateConfig(cfg);
  const int64_t H = cfg.index_n_heads, D = cfg.index_head_dim;
  VT_CHECK(static_cast<int64_t>(q.size()) == H * D, "q size mismatch");
  VT_CHECK(num_blocks >= 0, "num_blocks must be non-negative");
  VT_CHECK(static_cast<int64_t>(block_keys.size()) >= num_blocks * D,
           "block_keys size mismatch");
  // The scale is a CONSTANT. DeepSeek-V4 folds a learned per-(token,head)
  // `weights_proj` and an extra head_scale of n_heads**-0.5 in here; QSA has
  // neither tensor. Inheriting that fold would silently rescale every logit.
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
  std::vector<float> scores(static_cast<size_t>(num_blocks));
  for (int64_t b = 0; b < num_blocks; ++b) {
    const float* kp = &block_keys[b * D];
    float acc = 0.0f;
    for (int64_t h = 0; h < H; ++h) {
      const float* qp = &q[h * D];
      float dot = 0.0f;
      for (int64_t d = 0; d < D; ++d) dot += qp[d] * kp[d];
      // The ReLU is load-bearing: it lets a negatively-correlated head abstain
      // rather than veto.
      acc += dot > 0.0f ? dot : 0.0f;
    }
    // Divided AFTER the sum over heads, as upstream writes it.
    scores[b] = acc * inv_sqrt_d;
  }
  return scores;
}

std::vector<int64_t> QsaTopkBlocks(const std::vector<float>& scores,
                                   int64_t num_blocks, int64_t k) {
  VT_CHECK(num_blocks >= 0, "num_blocks must be non-negative");
  VT_CHECK(static_cast<int64_t>(scores.size()) >= num_blocks,
           "scores size mismatch");
  VT_CHECK(k >= 0 && k <= num_blocks, "k must be in [0, num_blocks]");

  std::vector<int64_t> idx(static_cast<size_t>(num_blocks));
  std::iota(idx.begin(), idx.end(), 0);
  if (num_blocks <= k) {
    // sampler.cu:391-402 — the all-select shortcut emits ASCENDING candidate
    // order, not score order. Inherited for consistency with vLLM's op, NOT
    // load-bearing: `QsaSelectedTokenIndices` sorts the expanded tokens anyway,
    // so reversing this order changes nothing observable and the suite stays
    // green. What makes the bit-identity oracle hold is that `std::sort`, not
    // this shortcut, sets the gather's reduction order.
    return idx;
  }
  // sampler.cu:515 — `logit < otherLogit || (logit == otherLogit && i < j)`:
  // a tie goes to the LOWER index. `std::stable_sort` on a strict
  // greater-than gives exactly that. Inherited for consistency and UNEXERCISED:
  // the golden fixtures produce no exact score tie, so flipping this rule leaves
  // the suite green. It is here so a real tie resolves the way the op it mirrors
  // resolves one, not because anything below can see it.
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int64_t a, int64_t b) { return scores[a] > scores[b]; });
  idx.resize(static_cast<size_t>(k));
  return idx;
}

std::vector<int32_t> QsaSelectedTokenIndices(
    const std::vector<int64_t>& selected_blocks, int64_t num_complete_blocks,
    int64_t kv_len, const QsaConfig& cfg) {
  QsaValidateConfig(cfg);
  const int64_t CR = cfg.compress_ratio;
  const int64_t width = cfg.index_width();
  VT_CHECK(kv_len >= 0, "kv_len must be non-negative");
  VT_CHECK(num_complete_blocks == kv_len / CR,
           "num_complete_blocks must be kv_len / compress_ratio");
  VT_CHECK(static_cast<int64_t>(selected_blocks.size()) <= cfg.block_topk(),
           "more blocks selected than the budget allows");

  std::vector<int32_t> tokens;
  tokens.reserve(static_cast<size_t>(width));
  for (int64_t b : selected_blocks) {
    VT_CHECK(b >= 0 && b < num_complete_blocks, "selected block out of range");
    // Block b IS tokens [CR*b, CR*b + CR). This expansion is what a mask-only
    // implementation never performs, and the reason the consumer can be a
    // gather at all.
    for (int64_t i = 0; i < CR; ++i) {
      tokens.push_back(static_cast<int32_t>(b * CR + i));
    }
  }
  // The incomplete trailing block is ALWAYS attended, whatever the scores said.
  // It is why the buffer is `budget + compress_ratio - 1` wide, not `budget`.
  for (int64_t t = num_complete_blocks * CR; t < kv_len; ++t) {
    tokens.push_back(static_cast<int32_t>(t));
  }
  VT_CHECK(static_cast<int64_t>(tokens.size()) <= width,
           "selected token count exceeds the index buffer width");

  // ASCENDING. Upstream's buffer is in score-rank order because it is only ever
  // scattered into a mask; a gather's order IS the softmax's reduction order,
  // and ascending is what makes a sub-budget gather reduce over exactly the
  // dense sequence.
  std::sort(tokens.begin(), tokens.end());
  std::vector<int32_t> out(static_cast<size_t>(width), -1);
  std::copy(tokens.begin(), tokens.end(), out.begin());
  return out;
}

// ── Consumers ────────────────────────────────────────────────────────────────

namespace {

// Shared shape check for both consumers, so they cannot disagree about what
// they were handed.
void CheckConsumerShapes(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, int64_t kv_len,
                         int64_t num_q_heads, int64_t num_kv_heads,
                         int64_t head_dim) {
  VT_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0,
           "bad attention shape");
  VT_CHECK(num_q_heads % num_kv_heads == 0,
           "GQA needs num_q_heads divisible by num_kv_heads");
  VT_CHECK(kv_len >= 0, "kv_len must be non-negative");
  VT_CHECK(static_cast<int64_t>(q.size()) == num_q_heads * head_dim,
           "q size mismatch (one query token)");
  VT_CHECK(static_cast<int64_t>(k.size()) >= kv_len * num_kv_heads * head_dim,
           "k size mismatch");
  VT_CHECK(static_cast<int64_t>(v.size()) >= kv_len * num_kv_heads * head_dim,
           "v size mismatch");
}

// The selected positions, ascending, with the -1 padding stripped.
std::vector<int64_t> Gathered(const std::vector<int32_t>& indices,
                              int64_t kv_len) {
  std::vector<int64_t> out;
  out.reserve(indices.size());
  for (int32_t idx : indices) {
    if (idx < 0) break;  // -1 terminates; the padding is not a position
    VT_CHECK(idx < kv_len, "selected index past the end of the cache");
    out.push_back(static_cast<int64_t>(idx));
  }
  return out;
}

float Dot(const float* a, const float* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

}  // namespace

std::vector<float> QsaGatherAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const std::vector<int32_t>& indices,
                                      int64_t kv_len, int64_t num_q_heads,
                                      int64_t num_kv_heads, int64_t head_dim,
                                      int64_t* keys_visited) {
  CheckConsumerShapes(q, k, v, kv_len, num_q_heads, num_kv_heads, head_dim);
  const std::vector<int64_t> sel = Gathered(indices, kv_len);
  VT_CHECK(!sel.empty() || kv_len == 0, "a causal query attends at least itself");

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t groups = num_q_heads / num_kv_heads;
  std::vector<float> out(static_cast<size_t>(num_q_heads) * head_dim, 0.0f);

  // The KEY-ROW READ COUNT, taken AT the read and nowhere else. An earlier
  // revision assigned `sel.size()` here, which restates the index buffer instead
  // of measuring the loop: a body that dot-products every one of the `kv_len`
  // cached rows and masks the rest with -inf — QsaMaskedAttention's cost wearing
  // this function's name — kept reporting the sparse number, and the suite
  // stayed green (W4 fresh review, mutation M22c). Counting at the `Dot` makes
  // the observable a function of the walk, so an implementation that does the
  // dense work cannot report the sparse figure.
  int64_t reads = 0;
  const auto key_dot = [&](const float* qp, int64_t t, int64_t kvh) {
    ++reads;
    return Dot(qp, &k[(t * num_kv_heads + kvh) * head_dim], head_dim);
  };

  for (int64_t h = 0; h < num_q_heads; ++h) {
    const int64_t kvh = h / groups;
    const float* qp = &q[h * head_dim];
    // Pass 1: the max, over the GATHERED rows only.
    float m = kNegInf;
    for (int64_t t : sel) {
      const float l = key_dot(qp, t, kvh) * scale;
      m = std::max(m, l);
    }
    // Pass 2: the softmax weights and the value reduction, ascending.
    float denom = 0.0f;
    float* dst = &out[h * head_dim];
    for (int64_t t : sel) {
      const float l = key_dot(qp, t, kvh) * scale;
      const float w = std::exp(l - m);
      denom += w;
      const float* vp = &v[(t * num_kv_heads + kvh) * head_dim];
      for (int64_t d = 0; d < head_dim; ++d) dst[d] += w * vp[d];
    }
    for (int64_t d = 0; d < head_dim; ++d) dst[d] /= denom;
  }
  if (keys_visited != nullptr) *keys_visited = reads;
  return out;
}

std::vector<float> QsaMaskedAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const std::vector<int32_t>& indices,
                                      int64_t kv_len, int64_t num_q_heads,
                                      int64_t num_kv_heads, int64_t head_dim,
                                      int64_t* keys_visited) {
  CheckConsumerShapes(q, k, v, kv_len, num_q_heads, num_kv_heads, head_dim);
  const std::vector<int64_t> sel = Gathered(indices, kv_len);
  std::vector<bool> keep(static_cast<size_t>(kv_len), false);
  for (int64_t t : sel) keep[static_cast<size_t>(t)] = true;

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t groups = num_q_heads / num_kv_heads;
  std::vector<float> out(static_cast<size_t>(num_q_heads) * head_dim, 0.0f);

  // Counted at the read, in the SAME unit as the gather's, so the two numbers
  // are comparable. This path reads every cached row whatever the mask says,
  // which is exactly what llama.cpp #27739 measured a sparse mask over a dense
  // cache doing under CUDA flash attention. It is a measured `kv_len *
  // num_q_heads * 2`, not an asserted `kv_len`.
  int64_t reads = 0;
  const auto key_dot = [&](const float* qp, int64_t t, int64_t kvh) {
    ++reads;
    return Dot(qp, &k[(t * num_kv_heads + kvh) * head_dim], head_dim);
  };

  for (int64_t h = 0; h < num_q_heads; ++h) {
    const int64_t kvh = h / groups;
    const float* qp = &q[h * head_dim];
    float m = kNegInf;
    for (int64_t t = 0; t < kv_len; ++t) {
      const float raw = key_dot(qp, t, kvh) * scale;
      const float l = keep[static_cast<size_t>(t)] ? raw : kNegInf;
      m = std::max(m, l);
    }
    float denom = 0.0f;
    float* dst = &out[h * head_dim];
    for (int64_t t = 0; t < kv_len; ++t) {
      const float raw = key_dot(qp, t, kvh) * scale;
      const float l = keep[static_cast<size_t>(t)] ? raw : kNegInf;
      // exp(-inf - m) is exactly +0, and adding an exact zero to a float
      // accumulator changes nothing. That is why this agrees with the gather
      // value for value while doing kv_len times the work.
      const float w = std::exp(l - m);
      denom += w;
      const float* vp = &v[(t * num_kv_heads + kvh) * head_dim];
      for (int64_t d = 0; d < head_dim; ++d) dst[d] += w * vp[d];
    }
    for (int64_t d = 0; d < head_dim; ++d) dst[d] /= denom;
  }
  if (keys_visited != nullptr) *keys_visited = reads;
  return out;
}

}  // namespace vllm::qwen4_exp
