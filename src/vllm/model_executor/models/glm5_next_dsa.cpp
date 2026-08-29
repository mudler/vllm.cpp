// GLM-5.3-Flash W3 — the DSA indexer's k-pool compression and its ragged tail.
// See glm5_next_dsa.h for the oracle, the port anchors on both sides, and why
// `deepseek_v4_dsa.cpp`'s raw-token selection is the wrong reuse.
#include "vllm/model_executor/models/glm5_next_dsa.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace vllm::glm5_next {

namespace {

void Require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("glm5_next DSA indexer: " + what);
}

// `torch.finfo(torch.float32).min` — upstream masks invalid candidates with THIS
// and not with `-inf` (`:839-842`). The difference is observable rather than
// cosmetic: a query row whose candidates are ALL invalid still produces finite
// scores and a well-defined (then discarded) top-k, where `-inf` would give NaN.
constexpr float kFinfoMin = std::numeric_limits<float>::lowest();

// `nn.Linear` — torch stores the weight as [out_features, in_features].
void Linear(const float* w, const float* x, int64_t out_features, int64_t in_features,
            float* out) {
  for (int64_t o = 0; o < out_features; ++o) {
    const float* row = w + o * in_features;
    double acc = 0.0;
    for (int64_t i = 0; i < in_features; ++i)
      acc += static_cast<double>(row[i]) * static_cast<double>(x[i]);
    out[o] = static_cast<float>(acc);
  }
}

int64_t Clamp(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

float IndexerDims::softmax_scale() const {
  Require(head_dim > 0, "`index_head_dim` must be > 0 to build the softmax scale");
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
}

int64_t IndexerDims::SelectK(int64_t num_pools) const {
  Validate();
  const int64_t budget = index_topk / index_kpool;
  return budget < num_pools ? budget : num_pools;
}

int64_t IndexerDims::OutputWidth() const {
  Validate();
  return always_select_tail ? index_topk + index_kpool - 1 : index_topk;
}

void IndexerDims::Validate() const {
  Require(hidden_size > 0 && q_lora_rank > 0 && n_heads > 0 && head_dim > 0,
          "`hidden_size`, `q_lora_rank`, `index_n_heads` and `index_head_dim` must "
          "all be > 0 — the indexer consumes `q_resid`, the q-LoRA latent "
          "(modular_glm5_next.py:1064-1070), so it cannot run without one");
  Require(index_topk > 0, "`index_topk` must be > 0");
  Require(index_kpool >= 1,
          "`index_kpool` must be >= 1 (configuration_glm5_next.py:216-217); it is 4 "
          "on the published checkpoint and 16 in the config class, so a defaulted "
          "value is wrong by a factor of four");
  Require(index_topk % index_kpool == 0,
          "`index_topk` must be divisible by `index_kpool` — the pool budget "
          "`index_topk // index_kpool` is exact upstream "
          "(configuration_glm5_next.py:219-220)");
}

IndexerDims IndexerDimsFrom(const Glm5NextParams& p) {
  IndexerDims d;
  d.hidden_size = p.hidden_size;
  d.q_lora_rank = p.mla.q_lora_rank;
  d.n_heads = p.indexer.n_heads;
  d.head_dim = p.indexer.head_dim;
  d.index_topk = p.indexer.topk;
  // READ, never defaulted. The published checkpoint says 4; the upstream config
  // class says 16; `Glm5NextParams` carries whichever the artifact declared.
  d.index_kpool = p.indexer.kpool;
  d.always_select_tail = p.indexer.kpool_always_select_tail;
  d.Validate();
  return d;
}

// `:795-801`. `k = self.k_norm(self.wk(hidden_states))`, `gate_scores =
// F.linear(hidden_states, self.index_kpool_compress_gate)`, `valid_channel =
// attention_mask.to(k.dtype)[..., None]`, then one `torch.cat` on the last axis.
std::vector<float> PackIndexerStates(const IndexerDims& d, const IndexerWeights& w,
                                     const std::vector<float>& hidden,
                                     const std::vector<uint8_t>& mask, int64_t batch,
                                     int64_t seq_len) {
  d.Validate();
  const int64_t H = d.hidden_size, D = d.head_dim;
  Require(w.wk != nullptr && w.k_norm_weight != nullptr && w.k_norm_bias != nullptr &&
              w.kpool_gate != nullptr,
          "`wk`, `k_norm.{weight,bias}` and `index_kpool_compress_gate` are required");
  Require(batch > 0 && seq_len > 0, "batch/seq_len must be > 0");
  Require(hidden.size() == static_cast<size_t>(batch * seq_len * H),
          "`hidden` must be [batch, seq_len, hidden_size]");
  Require(mask.size() == static_cast<size_t>(batch * seq_len),
          "`mask` must be [batch, seq_len]");

  const int64_t row = 2 * D + 1;
  std::vector<float> packed(static_cast<size_t>(batch * seq_len * row), 0.0f);
  std::vector<float> k(static_cast<size_t>(D));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      const float* x = hidden.data() + (b * seq_len + s) * H;
      float* dst = packed.data() + (b * seq_len + s) * row;
      Linear(w.wk, x, D, H, k.data());
      // `nn.LayerNorm(index_head_dim, eps=1e-6)` — mean subtraction, BIASED
      // variance over the last axis, then `weight * x_hat + bias`. NOT an
      // RMSNorm: both the mean subtraction and the bias are live, and the
      // checkpoint carrying `indexer.k_norm.bias` is what settles it.
      double mean = 0.0;
      for (int64_t i = 0; i < D; ++i) mean += static_cast<double>(k[static_cast<size_t>(i)]);
      mean /= static_cast<double>(D);
      double var = 0.0;
      for (int64_t i = 0; i < D; ++i) {
        const double c = static_cast<double>(k[static_cast<size_t>(i)]) - mean;
        var += c * c;
      }
      var /= static_cast<double>(D);
      const double inv = 1.0 / std::sqrt(var + static_cast<double>(kIndexerKNormEps));
      for (int64_t i = 0; i < D; ++i) {
        const double xh = (static_cast<double>(k[static_cast<size_t>(i)]) - mean) * inv;
        dst[i] = static_cast<float>(xh * static_cast<double>(w.k_norm_weight[i]) +
                                    static_cast<double>(w.k_norm_bias[i]));
      }
      Linear(w.kpool_gate, x, D, H, dst + D);
      dst[2 * D] = mask[static_cast<size_t>(b * seq_len + s)] != 0 ? 1.0f : 0.0f;
    }
  }
  return packed;
}

// `:877-895`. `causal & valid_keys`, with the query positions offset by
// `current_length - q_length` so a decode step's single query sees the whole
// cached prefix rather than only slot 0.
std::vector<uint8_t> GetVisibleTokens(const std::vector<uint8_t>& valid_keys, int64_t batch,
                                      int64_t kv_len, int64_t q_length,
                                      int64_t current_length) {
  Require(batch > 0 && kv_len > 0 && q_length > 0, "batch/kv_len/q_length must be > 0");
  Require(valid_keys.size() == static_cast<size_t>(batch * kv_len),
          "`valid_keys` must be [batch, kv_len]");
  std::vector<uint8_t> vis(static_cast<size_t>(batch * q_length * kv_len), 0);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < q_length; ++s) {
      const int64_t q_pos = current_length - q_length + s;
      for (int64_t j = 0; j < kv_len; ++j) {
        const bool causal = j <= q_pos;
        vis[static_cast<size_t>((b * q_length + s) * kv_len + j)] =
            (causal && valid_keys[static_cast<size_t>(b * kv_len + j)] != 0) ? 1 : 0;
      }
    }
  }
  return vis;
}

// `:897-970`.
PooledStates GetPooledStates(const IndexerDims& d, const IndexerWeights& w,
                             const std::vector<float>& packed, int64_t batch,
                             int64_t kv_len) {
  d.Validate();
  const int64_t D = d.head_dim, K = d.index_kpool;
  const int64_t row = 2 * D + 1;
  Require(w.kpool_ape != nullptr, "`index_kpool_compress_ape` is required");
  Require(batch > 0 && kv_len > 0, "batch/kv_len must be > 0");
  Require(packed.size() == static_cast<size_t>(batch * kv_len * row),
          "`packed` must be [batch, kv_len, 2 * index_head_dim + 1]");

  const int64_t np = (kv_len + K - 1) / K;  // `number_of_pools` (:928)

  // `first_key` — the index of the first NON-PAD token, or `seq_len` when the
  // row is entirely padding (`:938-942`). This is what makes a left-padded row
  // group differently from an unpadded one; pooling from slot 0 instead is the
  // defect that passes every unpadded fixture.
  std::vector<int64_t> first_key(static_cast<size_t>(batch), kv_len);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t j = 0; j < kv_len; ++j) {
      if (packed[static_cast<size_t>((b * kv_len + j) * row + 2 * D)] != 0.0f) {
        first_key[static_cast<size_t>(b)] = j;
        break;
      }
    }
  }

  std::vector<int64_t> raw_idx(static_cast<size_t>(batch * np * K), 0);
  std::vector<uint8_t> member_valid(static_cast<size_t>(batch * np * K), 0);
  std::vector<uint8_t> pool_valid(static_cast<size_t>(batch * np), 0);
  std::vector<float> pool_keys(static_cast<size_t>(batch * np * D), 0.0f);

  std::vector<double> logit(static_cast<size_t>(K));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t p = 0; p < np; ++p) {
      bool all_valid = true;
      for (int64_t j = 0; j < K; ++j) {
        const int64_t idx = first_key[static_cast<size_t>(b)] + p * K + j;
        const size_t o = static_cast<size_t>((b * np + p) * K + j);
        raw_idx[o] = idx;
        // `safe_indices = pool_indices.clamp(0, seq_len - 1)` (:948) makes the
        // gather legal; `grouped_valid_keys & (pool_indices < seq_len)` (:955)
        // then throws the out-of-range members away again.
        const int64_t safe = Clamp(idx, 0, kv_len - 1);
        const bool ok =
            idx < kv_len &&
            packed[static_cast<size_t>((b * kv_len + safe) * row + 2 * D)] != 0.0f;
        member_valid[o] = ok ? 1 : 0;
        all_valid = all_valid && ok;
      }
      pool_valid[static_cast<size_t>(b * np + p)] = all_valid ? 1 : 0;

      // The LEARNED pool weighting (`:959-965`): `index_head_dim` INDEPENDENT
      // softmaxes, one per channel, over the pool's `index_kpool` members, in
      // fp32, with the intra-pool absolute position embedding added to each
      // member's gate score. It is NOT a mean, and a mean passes every shape
      // check this file could carry.
      for (int64_t c = 0; c < D; ++c) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < K; ++j) {
          const size_t o = static_cast<size_t>((b * np + p) * K + j);
          if (member_valid[o] == 0) {
            logit[static_cast<size_t>(j)] = -std::numeric_limits<double>::infinity();
            continue;
          }
          const int64_t safe = Clamp(raw_idx[o], 0, kv_len - 1);
          const double g = static_cast<double>(
              packed[static_cast<size_t>((b * kv_len + safe) * row + D + c)]);
          const double ape = static_cast<double>(w.kpool_ape[j * D + c]);
          logit[static_cast<size_t>(j)] = g + ape;
          mx = std::max(mx, logit[static_cast<size_t>(j)]);
        }
        // `torch.nan_to_num(logits.softmax(dim=2))` — a pool with NO valid
        // member softmaxes to NaN and is then zeroed, so it contributes nothing
        // rather than poisoning the whole row (`:962-964`).
        if (!(mx > -std::numeric_limits<double>::infinity())) continue;
        double denom = 0.0;
        for (int64_t j = 0; j < K; ++j) {
          logit[static_cast<size_t>(j)] =
              logit[static_cast<size_t>(j)] == -std::numeric_limits<double>::infinity()
                  ? 0.0
                  : std::exp(logit[static_cast<size_t>(j)] - mx);
          denom += logit[static_cast<size_t>(j)];
        }
        double acc = 0.0;
        for (int64_t j = 0; j < K; ++j) {
          if (logit[static_cast<size_t>(j)] == 0.0) continue;
          const size_t o = static_cast<size_t>((b * np + p) * K + j);
          const int64_t safe = Clamp(raw_idx[o], 0, kv_len - 1);
          const double key =
              static_cast<double>(packed[static_cast<size_t>((b * kv_len + safe) * row + c)]);
          acc += (logit[static_cast<size_t>(j)] / denom) * key;
        }
        pool_keys[static_cast<size_t>((b * np + p) * D + c)] = static_cast<float>(acc);
      }
    }
  }

  // `keep = pool_valid.any(0)` then `[:, keep]` (`:967-970`) — a pool no row in
  // the batch can use is dropped, which also renumbers every pool after it.
  std::vector<int64_t> kept;
  kept.reserve(static_cast<size_t>(np));
  for (int64_t p = 0; p < np; ++p) {
    bool any = false;
    for (int64_t b = 0; b < batch && !any; ++b) {
      any = pool_valid[static_cast<size_t>(b * np + p)] != 0;
    }
    if (any) kept.push_back(p);
  }

  PooledStates out;
  out.num_pools = static_cast<int64_t>(kept.size());
  const int64_t P = out.num_pools;
  out.pool_keys.assign(static_cast<size_t>(batch * P * D), 0.0f);
  out.pool_indices.assign(static_cast<size_t>(batch * P * K), -1);
  out.pool_valid.assign(static_cast<size_t>(batch * P), 0);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t i = 0; i < P; ++i) {
      const int64_t p = kept[static_cast<size_t>(i)];
      out.pool_valid[static_cast<size_t>(b * P + i)] =
          pool_valid[static_cast<size_t>(b * np + p)];
      for (int64_t c = 0; c < D; ++c) {
        out.pool_keys[static_cast<size_t>((b * P + i) * D + c)] =
            pool_keys[static_cast<size_t>((b * np + p) * D + c)];
      }
      for (int64_t j = 0; j < K; ++j) {
        const size_t src = static_cast<size_t>((b * np + p) * K + j);
        // `pool_indices.masked_fill(~grouped_valid_keys, -1)` (:957) — the -1
        // is written BEFORE `pool_end` is read, so an invalid last member makes
        // the pool's visibility probe read slot 0 after the clamp at `:831`.
        out.pool_indices[static_cast<size_t>((b * P + i) * K + j)] =
            member_valid[src] != 0 ? static_cast<int32_t>(raw_idx[src]) : -1;
      }
    }
  }
  return out;
}

// `:972-1022`.
std::vector<int32_t> AppendVisibleTail(const IndexerDims& d, const std::vector<int32_t>& topk,
                                       int64_t in_width, const std::vector<uint8_t>& visible,
                                       const std::vector<uint8_t>& valid_keys, int64_t batch,
                                       int64_t q_length, int64_t kv_len) {
  d.Validate();
  const int64_t K = d.index_kpool;
  const int64_t tail_w = K - 1;  // `max_tail_width` (:985)
  Require(batch > 0 && q_length > 0 && kv_len > 0, "batch/q_length/kv_len must be > 0");
  Require(topk.size() == static_cast<size_t>(batch * q_length * in_width),
          "`topk` must be [batch, q_length, in_width]");
  Require(visible.size() == static_cast<size_t>(batch * q_length * kv_len),
          "`visible` must be [batch, q_length, kv_len]");
  Require(valid_keys.size() == static_cast<size_t>(batch * kv_len),
          "`valid_keys` must be [batch, kv_len]");
  // `if (max_tail_width := self.index_kpool - 1) == 0: return topk_indices`
  // (`:985-986`) — at `index_kpool == 1` there is no ragged tail to keep.
  if (tail_w == 0) return topk;

  std::vector<int64_t> first_key(static_cast<size_t>(batch), kv_len);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t j = 0; j < kv_len; ++j) {
      if (valid_keys[static_cast<size_t>(b * kv_len + j)] != 0) {
        first_key[static_cast<size_t>(b)] = j;
        break;
      }
    }
  }

  const int64_t out_w = in_width + tail_w;
  std::vector<int32_t> out(static_cast<size_t>(batch * q_length * out_w), -1);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < q_length; ++s) {
      const int32_t* src = topk.data() + (b * q_length + s) * in_width;
      int32_t* dst = out.data() + (b * q_length + s) * out_w;
      for (int64_t i = 0; i < in_width; ++i) dst[i] = src[i];
      int64_t visible_count = 0;
      for (int64_t j = 0; j < kv_len; ++j) {
        visible_count += visible[static_cast<size_t>((b * q_length + s) * kv_len + j)] != 0 ? 1 : 0;
      }
      const int64_t tail_count = visible_count % K;
      const int64_t tail_start =
          first_key[static_cast<size_t>(b)] + visible_count - tail_count;
      for (int64_t j = 0; j < tail_w; ++j) {
        const int64_t idx = tail_start + j;
        // `tail_valid` drops the fill positions and anything past the cache
        // (`:1013`); `tail_visible` then re-checks the padding mask (`:1016-1017`).
        const bool valid = j < tail_count && idx < kv_len;
        const int64_t safe = Clamp(idx, 0, kv_len - 1);
        const bool vis =
            visible[static_cast<size_t>((b * q_length + s) * kv_len + safe)] != 0;
        dst[in_width + j] = (valid && vis) ? static_cast<int32_t>(idx) : -1;
      }
    }
  }
  return out;
}

// `:771-875`.
IndexerSelection SelectIndexerTopk(const IndexerDims& d, const IndexerWeights& w,
                                   const std::vector<float>& hidden,
                                   const std::vector<float>& q_resid,
                                   const std::vector<uint8_t>& mask, int64_t batch,
                                   int64_t seq_len) {
  d.Validate();
  const int64_t H = d.hidden_size, D = d.head_dim, N = d.n_heads, K = d.index_kpool;
  const int64_t QL = d.q_lora_rank;
  Require(w.wq_b != nullptr && w.weights_proj != nullptr,
          "`wq_b` and `weights_proj` are required");
  Require(q_resid.size() == static_cast<size_t>(batch * seq_len * QL),
          "`q_resid` must be [batch, seq_len, q_lora_rank]");

  // `past_key_values is None`: `kv_len = current_length = seq_len` (`:803-811`).
  // The cached arm — where `kv_len` is the STATIC cache width and
  // `current_length` the live one — is W5's, and it is why every function above
  // takes those two lengths separately instead of assuming they agree.
  const int64_t kv_len = seq_len, current_length = seq_len;

  const std::vector<float> packed = PackIndexerStates(d, w, hidden, mask, batch, seq_len);
  const int64_t row = 2 * D + 1;

  std::vector<uint8_t> valid_keys(static_cast<size_t>(batch * kv_len), 0);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t j = 0; j < kv_len; ++j) {
      valid_keys[static_cast<size_t>(b * kv_len + j)] =
          packed[static_cast<size_t>((b * kv_len + j) * row + 2 * D)] != 0.0f ? 1 : 0;
    }
  }
  const std::vector<uint8_t> visible =
      GetVisibleTokens(valid_keys, batch, kv_len, seq_len, current_length);

  IndexerSelection sel;
  sel.pooled = GetPooledStates(d, w, packed, batch, kv_len);
  // P can legitimately be ZERO: below `index_kpool` valid tokens no pool is
  // complete, `keep = pool_valid.any(0)` is empty (`:967-970`) and upstream
  // carries an empty candidate dimension through the rest of the function
  // rather than refusing. `select_k` is then `min(index_topk // index_kpool, 0)`
  // = 0, `flat` is zero-width, both loops below do not run, and
  // `AppendVisibleTail` returns the raw visible tail on its own — which is the
  // selection upstream serves. Refusing here instead rejected the first
  // `index_kpool - 1` tokens of every prefill.
  const int64_t P = sel.pooled.num_pools;

  // `q = self.wq_b(q_resid).view(B, S, -1, head_dim)` (`:795`).
  std::vector<float> q(static_cast<size_t>(N * D));
  std::vector<float> weights(static_cast<size_t>(N));
  const double head_scale = 1.0 / std::sqrt(static_cast<double>(N));  // `n_heads**-0.5` (:827)
  const double sm_scale = static_cast<double>(d.softmax_scale());

  sel.index_scores.assign(static_cast<size_t>(batch * seq_len * P), 0.0f);
  std::vector<float> masked(static_cast<size_t>(P));
  std::vector<uint8_t> candidate(static_cast<size_t>(P));
  const int64_t select_k = d.SelectK(P);
  const int64_t flat_w = select_k * K;
  std::vector<int32_t> flat(static_cast<size_t>(batch * seq_len * flat_w), -1);
  std::vector<int64_t> order(static_cast<size_t>(P));

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      const float* x = hidden.data() + (b * seq_len + s) * H;
      Linear(w.wq_b, q_resid.data() + (b * seq_len + s) * QL, N * D, QL, q.data());
      Linear(w.weights_proj, x, N, H, weights.data());

      for (int64_t p = 0; p < P; ++p) {
        // `scores = relu(matmul(q, pool_keys^T) * softmax_scale)` per head
        // (`:823-824`), then the per-head weighted sum (`:827-828`). The ReLU is
        // BEFORE the head mix, so a head that dislikes a pool contributes zero
        // rather than a negative vote.
        double acc = 0.0;
        for (int64_t h = 0; h < N; ++h) {
          double dot = 0.0;
          const float* qh = q.data() + h * D;
          const float* pk = sel.pooled.pool_keys.data() + (b * P + p) * D;
          for (int64_t c = 0; c < D; ++c)
            dot += static_cast<double>(qh[c]) * static_cast<double>(pk[c]);
          const double relu = std::max(0.0, dot * sm_scale);
          acc += static_cast<double>(weights[static_cast<size_t>(h)]) * head_scale * relu;
        }
        sel.index_scores[static_cast<size_t>((b * seq_len + s) * P + p)] =
            static_cast<float>(acc);

        // `pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)` and
        // `pool_visible = visible_tokens.gather(-1, pool_end)` (`:831-835`):
        // a pool is selectable only if its LAST member is visible to the query.
        const int64_t last =
            Clamp(sel.pooled.pool_indices[static_cast<size_t>((b * P + p) * K + K - 1)], 0,
                  kv_len - 1);
        const bool vis = visible[static_cast<size_t>((b * seq_len + s) * kv_len + last)] != 0;
        candidate[static_cast<size_t>(p)] =
            (vis && sel.pooled.pool_valid[static_cast<size_t>(b * P + p)] != 0) ? 1 : 0;
        masked[static_cast<size_t>(p)] = candidate[static_cast<size_t>(p)] != 0
                                             ? static_cast<float>(acc)
                                             : kFinfoMin;
      }

      // `index_scores.topk(select_k, dim=-1).indices` (`:850`). Ties are broken
      // by the lower pool index, which is `torch.topk`'s CPU behaviour; the gate
      // fixture is deliberately tie-FREE and prints the margin, because a top-k
      // whose separation is zero is a coin flip on either side.
      std::iota(order.begin(), order.begin() + static_cast<long>(P), int64_t{0});
      std::stable_sort(order.begin(), order.begin() + static_cast<long>(P),
                       [&](int64_t a, int64_t c) {
                         return masked[static_cast<size_t>(a)] > masked[static_cast<size_t>(c)];
                       });

      int32_t* dst = flat.data() + (b * seq_len + s) * flat_w;
      for (int64_t j = 0; j < select_k; ++j) {
        const int64_t p = order[static_cast<size_t>(j)];
        // `selected_valid` masks the WHOLE expanded pool to -1 (`:853, :859-862`)
        // — a pool the top-k had to pick because nothing better existed does not
        // become a real selection.
        const bool keep = candidate[static_cast<size_t>(p)] != 0;
        for (int64_t m = 0; m < K; ++m) {
          dst[j * K + m] =
              keep ? sel.pooled.pool_indices[static_cast<size_t>((b * P + p) * K + m)] : -1;
        }
      }
    }
  }

  int64_t width = flat_w;
  std::vector<int32_t> outv = flat;
  if (d.always_select_tail) {
    outv = AppendVisibleTail(d, outv, width, visible, valid_keys, batch, seq_len, kv_len);
    width += K - 1;
  }

  // `F.pad(..., value=-1)` then `[..., :output_width]` (`:869-872`), then the
  // query-side padding mask (`:873`).
  const int64_t out_w = d.OutputWidth();
  sel.topk_indices.assign(static_cast<size_t>(batch * seq_len * out_w), -1);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      if (mask[static_cast<size_t>(b * seq_len + s)] == 0) continue;
      const int32_t* src = outv.data() + (b * seq_len + s) * width;
      int32_t* dst = sel.topk_indices.data() + (b * seq_len + s) * out_w;
      for (int64_t i = 0; i < out_w && i < width; ++i) dst[i] = src[i];
    }
  }
  return sel;
}

}  // namespace vllm::glm5_next
