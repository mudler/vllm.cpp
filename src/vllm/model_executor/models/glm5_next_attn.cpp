// GLM-5.3-Flash W5b-1 — `Glm5NextTextAttention`. See `glm5_next_attn.h` for the
// oracle, the port anchors and the three traps this file exists to avoid.
#include "vllm/model_executor/models/glm5_next_attn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm::glm5_next {
namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("glm5_next attention: " + what);
}

void RequireSize(const char* what, size_t got, int64_t want) {
  if (static_cast<int64_t>(got) != want) {
    Fail(std::string(what) + " must hold " + std::to_string(want) +
         " floats, got " + std::to_string(got));
  }
}

// `Glm5NextTextRMSNorm.forward` (`:75-80`): variance over the LAST axis, the
// eps INSIDE the rsqrt, and the gain applied after. Computed in f32, which is
// upstream's own `.to(torch.float32)` at `:77`.
void RmsNorm(const float* in, const float* gamma, int64_t n, double eps,
             float* out) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(in[i]) * in[i];
  const float inv =
      static_cast<float>(1.0 / std::sqrt(acc / static_cast<double>(n) + eps));
  for (int64_t i = 0; i < n; ++i) out[i] = gamma[i] * (in[i] * inv);
}

// Append `[batch, seq_len, width]` new rows to a `[batch, cached_len, width]`
// history, in place, producing `[batch, cached_len + seq_len, width]`.
//
// It is a per-batch-row SPLICE and not a `insert(end(), ...)`, because the batch
// axis is the OUTER one: appending flatly would put request 0's new tokens in
// front of request 1's history. Upstream never has to state this — its cache is
// a torch `cat(dim=-2)` on a tensor whose batch axis is separate — and getting
// it wrong at batch 1 is invisible, which is why the focused gate runs the cached
// case at batch 2 with a left-padded row.
void AppendRows(std::vector<float>* hist, const std::vector<float>& fresh,
                int64_t batch, int64_t cached_len, int64_t seq_len,
                int64_t width) {
  const int64_t total = cached_len + seq_len;
  std::vector<float> out(static_cast<size_t>(batch * total * width));
  for (int64_t b = 0; b < batch; ++b) {
    if (cached_len > 0) {
      std::copy_n(hist->data() + b * cached_len * width,
                  static_cast<size_t>(cached_len * width),
                  out.data() + b * total * width);
    }
    std::copy_n(fresh.data() + b * seq_len * width,
                static_cast<size_t>(seq_len * width),
                out.data() + (b * total + cached_len) * width);
  }
  *hist = std::move(out);
}

// `nn.Linear(bias=False)` on a row-major `[out, in]` weight: one dot per output.
void LinearNoBias(const float* x, const float* w, int64_t in_dim,
                  int64_t out_dim, float* out) {
  for (int64_t o = 0; o < out_dim; ++o) {
    const float* row = w + o * in_dim;
    double acc = 0.0;
    for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(x[i]) * row[i];
    out[o] = static_cast<float>(acc);
  }
}

}  // namespace

float MlaDims::scaling() const {
  return static_cast<float>(
      std::pow(static_cast<double>(qk_head_dim()), -0.5));
}

void MlaDims::Validate() const {
  const auto positive = [](const char* name, int64_t v) {
    if (v <= 0) {
      Fail(std::string("`") + name + "` must be > 0, got " + std::to_string(v));
    }
  };
  positive("hidden_size", hidden_size);
  positive("num_attention_heads", num_heads);
  positive("q_lora_rank", q_lora_rank);
  positive("kv_lora_rank", kv_lora_rank);
  positive("qk_nope_head_dim", qk_nope_head_dim);
  positive("v_head_dim", v_head_dim);
  if (!(rms_norm_eps > 0.0)) {
    Fail("`rms_norm_eps` must be > 0, got " + std::to_string(rms_norm_eps));
  }
  // Upstream's own clause, mirrored so the two refusals mean the same thing:
  // `validate_architecture` (`configuration_glm5_next.py:225-228`) raises
  // "Expecting NoPE for the DSA attention layers, but got {n} as RoPE dim."
  // for any positive value. This port therefore has no rope branch; a
  // positive width is refused rather than half-implemented.
  if (qk_rope_head_dim != 0) {
    Fail("Expecting NoPE for the DSA attention layers, but got " +
         std::to_string(qk_rope_head_dim) + " as RoPE dim.");
  }
}

MlaDims MlaDimsFrom(const Glm5NextParams& p) {
  MlaDims d;
  d.hidden_size = p.hidden_size;
  d.num_heads = p.num_attention_heads;
  d.q_lora_rank = p.mla.q_lora_rank;
  d.kv_lora_rank = p.mla.kv_lora_rank;
  d.qk_nope_head_dim = p.mla.qk_nope_head_dim;
  d.qk_rope_head_dim = p.mla.qk_rope_head_dim;
  d.v_head_dim = p.mla.v_head_dim;
  d.rms_norm_eps = p.rms_norm_eps;
  d.Validate();
  return d;
}

IndexerRole IndexerRoleFor(const Glm5NextParams& p, int64_t layer_idx) {
  const int64_t n = static_cast<int64_t>(p.indexer_types.size());
  if (n <= 0) Fail("`indexer_types` is empty; the schedule was never resolved");
  if (layer_idx < 0 || layer_idx >= n) {
    Fail("layer_idx " + std::to_string(layer_idx) + " is outside [0, " +
         std::to_string(n) + ")");
  }
  IndexerRole r;
  r.skip_topk = p.indexer_types[static_cast<size_t>(layer_idx)] ==
                Glm5NextIndexerKind::kShared;
  // `min(layer_idx + 1, len - 1)` (`:1133`) — the LAST layer looks at itself.
  const int64_t next = std::min(layer_idx + 1, n - 1);
  r.next_skip_topk =
      !r.skip_topk && p.indexer_types[static_cast<size_t>(next)] ==
                          Glm5NextIndexerKind::kShared;
  return r;
}

std::vector<float> QResid(const MlaDims& d, const MlaWeights& w,
                          const std::vector<float>& hidden, int64_t batch,
                          int64_t seq_len) {
  d.Validate();
  const int64_t tokens = batch * seq_len;
  RequireSize("hidden_states", hidden.size(), tokens * d.hidden_size);
  RequireSize("q_a_proj", w.q_a_proj.size(), d.q_lora_rank * d.hidden_size);
  RequireSize("q_a_layernorm", w.q_a_layernorm.size(), d.q_lora_rank);

  std::vector<float> out(static_cast<size_t>(tokens * d.q_lora_rank));
  std::vector<float> tmp(static_cast<size_t>(d.q_lora_rank));
  for (int64_t t = 0; t < tokens; ++t) {
    LinearNoBias(hidden.data() + t * d.hidden_size, w.q_a_proj.data(),
                 d.hidden_size, d.q_lora_rank, tmp.data());
    RmsNorm(tmp.data(), w.q_a_layernorm.data(), d.q_lora_rank, d.rms_norm_eps,
            out.data() + t * d.q_lora_rank);
  }
  return out;
}

std::vector<float> CompressKv(const MlaDims& d, const MlaWeights& w,
                              const std::vector<float>& hidden, int64_t batch,
                              int64_t seq_len) {
  d.Validate();
  const int64_t tokens = batch * seq_len;
  // `kv_lora_rank + qk_rope_head_dim`, and the rope half has no width, so the
  // split at `:1171` takes the whole projection and `k_rot` is empty.
  const int64_t proj = d.kv_lora_rank + d.qk_rope_head_dim;
  RequireSize("hidden_states", hidden.size(), tokens * d.hidden_size);
  RequireSize("kv_a_proj_with_mqa", w.kv_a_proj_with_mqa.size(),
              proj * d.hidden_size);
  RequireSize("kv_a_layernorm", w.kv_a_layernorm.size(), d.kv_lora_rank);

  std::vector<float> out(static_cast<size_t>(tokens * d.kv_lora_rank));
  std::vector<float> tmp(static_cast<size_t>(proj));
  for (int64_t t = 0; t < tokens; ++t) {
    LinearNoBias(hidden.data() + t * d.hidden_size, w.kv_a_proj_with_mqa.data(),
                 d.hidden_size, proj, tmp.data());
    RmsNorm(tmp.data(), w.kv_a_layernorm.data(), d.kv_lora_rank, d.rms_norm_eps,
            out.data() + t * d.kv_lora_rank);
  }
  return out;
}

ExpandedKv ExpandKv(const MlaDims& d, const MlaWeights& w,
                    const std::vector<float>& k_pass, int64_t batch,
                    int64_t seq_len) {
  d.Validate();
  const int64_t tokens = batch * seq_len;
  const int64_t r = d.kv_lora_rank;
  const int64_t nope = d.qk_nope_head_dim;
  const int64_t vhd = d.v_head_dim;
  const int64_t heads = d.num_heads;
  RequireSize("k_pass", k_pass.size(), tokens * r);
  RequireSize("k_b_proj", w.k_b_proj.size(), heads * r * nope);
  RequireSize("v_b_proj", w.v_b_proj.size(), heads * vhd * r);

  ExpandedKv out;
  // `key_states` is `qk_head_dim` wide; the rope half has no width, so it IS
  // `k_nope` and there is nothing to concatenate (`:1150-1152`).
  out.key_states.assign(
      static_cast<size_t>(batch * heads * seq_len * d.qk_head_dim()), 0.0F);
  out.value_states.assign(
      static_cast<size_t>(batch * heads * seq_len * vhd), 0.0F);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      // TRANSPOSED: `k_b_proj[h][latent][nope]`, so the contraction runs over
      // the FIRST inner axis and the stride over `nope` is 1.
      const float* kb = w.k_b_proj.data() + h * r * nope;
      // NOT transposed: `v_b_proj[h][v][latent]`, contraction over the SECOND.
      const float* vb = w.v_b_proj.data() + h * vhd * r;
      for (int64_t t = 0; t < seq_len; ++t) {
        const float* kp = k_pass.data() + (b * seq_len + t) * r;
        float* kdst = out.key_states.data() +
                      ((b * heads + h) * seq_len + t) * d.qk_head_dim();
        float* vdst =
            out.value_states.data() + ((b * heads + h) * seq_len + t) * vhd;
        for (int64_t dd = 0; dd < nope; ++dd) {
          double acc = 0.0;
          for (int64_t i = 0; i < r; ++i) {
            acc += static_cast<double>(kp[i]) * kb[i * nope + dd];
          }
          kdst[dd] = static_cast<float>(acc);
        }
        for (int64_t dd = 0; dd < vhd; ++dd) {
          const float* row = vb + dd * r;
          double acc = 0.0;
          for (int64_t i = 0; i < r; ++i) {
            acc += static_cast<double>(kp[i]) * row[i];
          }
          vdst[dd] = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

std::vector<uint8_t> BuildAttentionMaskFromTopk(const std::vector<int32_t>& topk,
                                                int64_t batch, int64_t q_length,
                                                int64_t width,
                                                int64_t kv_length) {
  if (batch <= 0 || q_length <= 0 || width <= 0 || kv_length <= 0) {
    Fail("build_attention_mask_from_topk needs positive batch, q_length, width "
         "and kv_length");
  }
  RequireSize("topk_indices", topk.size(), batch * q_length * width);

  // `selected_counts` is int32 upstream and only its `ne(0)` is read
  // (`:1236-1246`), so a saturating byte is the same predicate at 1/4 the
  // buffer. It is NOT a bool being or-ed: the scatter-add of a ZERO for an
  // invalid index must not turn a visible key off, and `|=` of 0 does not.
  std::vector<uint8_t> mask(
      static_cast<size_t>(batch * q_length * kv_length), 0U);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < q_length; ++q) {
      const int32_t* row = topk.data() + (b * q_length + q) * width;
      uint8_t* dst = mask.data() + (b * q_length + q) * kv_length;
      for (int64_t i = 0; i < width; ++i) {
        const int32_t idx = row[i];
        // `topk_indices.ge(0) & topk_indices.lt(kv_length)` (`:1232`).
        const bool valid = idx >= 0 && idx < static_cast<int32_t>(kv_length);
        // `.clamp(0, kv_length - 1)` "only so scatter has a legal index"
        // (`:1234-1235`); the clamped slot receives a 0 when the entry was invalid.
        const int64_t safe = std::min<int64_t>(
            std::max<int64_t>(idx, 0), kv_length - 1);
        if (valid) dst[safe] = 1U;
      }
    }
  }
  return mask;
}

AttentionResult Attention(const MlaDims& d, const MlaWeights& w,
                          const IndexerDims& id, const IndexerWeights* indexer,
                          const IndexerRole& role,
                          const std::vector<float>& hidden,
                          const std::vector<uint8_t>& mask,
                          const std::vector<int32_t>* prev_topk_indices,
                          int64_t prev_topk_width, int64_t batch,
                          int64_t seq_len, DsaCache* cache) {
  d.Validate();
  if (batch <= 0 || seq_len <= 0) Fail("batch and seq_len must be > 0");
  const int64_t tokens = batch * seq_len;
  RequireSize("hidden_states", hidden.size(), tokens * d.hidden_size);
  if (static_cast<int64_t>(mask.size()) != tokens) {
    Fail("attention_mask must hold " + std::to_string(tokens) +
         " entries, got " + std::to_string(mask.size()));
  }
  RequireSize("q_b_proj", w.q_b_proj.size(),
              d.num_heads * d.qk_head_dim() * d.q_lora_rank);
  RequireSize("o_proj", w.o_proj.size(),
              d.hidden_size * d.num_heads * d.v_head_dim);

  // `self.indexer = None if self.skip_topk else Glm5NextTextIndexer(...)`
  // (`:1131`). Both mismatches are refused: a shared layer handed an indexer
  // would let a caller re-enable the recomputation this model must not do, and
  // a full layer without one has nothing to select with.
  if (role.skip_topk && indexer != nullptr) {
    Fail("a `shared` layer has no indexer of its own (`:1131`); pass nullptr "
         "and supply `prev_topk_indices` instead");
  }
  if (!role.skip_topk && indexer == nullptr) {
    Fail("a `full` layer needs its own indexer weights (`:1131`)");
  }

  AttentionResult res;
  const int64_t heads = d.num_heads;
  const int64_t qk = d.qk_head_dim();
  const int64_t vhd = d.v_head_dim;

  // ── the projections (`:1163-1175`) ────────────────────────────────────────
  const std::vector<float> q_resid = QResid(d, w, hidden, batch, seq_len);
  std::vector<float> query(static_cast<size_t>(batch * heads * seq_len * qk));
  {
    std::vector<float> row(static_cast<size_t>(heads * qk));
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t t = 0; t < seq_len; ++t) {
        LinearNoBias(q_resid.data() + (b * seq_len + t) * d.q_lora_rank,
                     w.q_b_proj.data(), d.q_lora_rank, heads * qk, row.data());
        // `.view(B, S, -1, qk_head_dim).transpose(1, 2)` (`:1164`, `:1168`):
        // the flat projection is HEAD-MAJOR within a token.
        for (int64_t h = 0; h < heads; ++h) {
          float* dst = query.data() + ((b * heads + h) * seq_len + t) * qk;
          for (int64_t i = 0; i < qk; ++i) dst[i] = row[h * qk + i];
        }
      }
    }
  }
  // ── the cache update (`:1177-1179`) ───────────────────────────────────────
  // `k_pass` for THIS window, then the history it belongs to. Upstream appends
  // the EXPANDED K/V here; this port appends the 512-wide latent and expands the
  // history, which is value-for-value identical because `ExpandKv` is token-wise
  // under NoPE. The header states the argument; `test_glm5_next_layer.cpp`
  // asserts the split identity on real values.
  const std::vector<float> k_pass_new = CompressKv(d, w, hidden, batch, seq_len);
  const int64_t cached_len = cache != nullptr ? cache->cached_len : 0;
  const int64_t kv_length = cached_len + seq_len;
  if (cache != nullptr) {
    if (cached_len < 0) Fail("`DsaCache::cached_len` must not be negative");
    RequireSize("DsaCache::k_pass", cache->k_pass.size(),
                batch * cached_len * d.kv_lora_rank);
    AppendRows(&cache->k_pass, k_pass_new, batch, cached_len, seq_len,
               d.kv_lora_rank);
  }
  const std::vector<float>& k_pass_all =
      cache != nullptr ? cache->k_pass : k_pass_new;
  const ExpandedKv kv = ExpandKv(d, w, k_pass_all, batch, kv_length);

  // ── the selection (`:1181-1191`) ──────────────────────────────────────────
  if (role.skip_topk) {
    if (prev_topk_indices == nullptr) {
      // Upstream's own message (`:1190`), mirrored verbatim so a log line means
      // the same thing on both sides.
      Fail("Shared DSA layers require top-k indices from a previous full "
           "indexer layer.");
    }
    if (prev_topk_width <= 0) {
      Fail("`prev_topk_width` must be > 0 for a `shared` layer");
    }
    RequireSize("prev_topk_indices", prev_topk_indices->size(),
                tokens * prev_topk_width);
    res.topk_indices = *prev_topk_indices;
    res.topk_width = prev_topk_width;
  } else if (cache == nullptr) {
    const IndexerSelection sel =
        SelectIndexerTopk(id, *indexer, hidden, q_resid, mask, batch, seq_len);
    res.topk_width = id.OutputWidth();
    res.topk_indices = sel.topk_indices;
  } else {
    // `past_key_values.update_indexer(packed_states, self.layer_idx)` (`:810`):
    // the packed rows of THIS window are appended, and the selection runs over
    // the WHOLE history the call returns.
    const int64_t row = 2 * id.head_dim + 1;
    RequireSize("DsaCache::indexer_packed", cache->indexer_packed.size(),
                batch * cached_len * row);
    AppendRows(&cache->indexer_packed,
               PackIndexerStates(id, *indexer, hidden, mask, batch, seq_len),
               batch, cached_len, seq_len, row);
    const IndexerSelection sel = SelectIndexerTopkFromPacked(
        id, *indexer, hidden, q_resid, mask, cache->indexer_packed, batch,
        seq_len, kv_length);
    res.topk_width = id.OutputWidth();
    res.topk_indices = sel.topk_indices;
  }

  // ── the mask (`:1193-1197`) ───────────────────────────────────────────────
  // `kv_length` is `key_states.shape[2]`, which is the CACHE length. It equals
  // `seq_len` on a fresh prefill (`cache == nullptr`) and exceeds it on every
  // cached continuation, which is what W5b-2's binding made reachable.
  const std::vector<uint8_t> visible = BuildAttentionMaskFromTopk(
      res.topk_indices, batch, seq_len, res.topk_width, kv_length);

  // ── `eager_attention_forward` (`:1039-1061`) ──────────────────────────────
  // `repeat_kv` is the identity: `num_key_value_groups` is 1 for this model.
  const float scaling = d.scaling();
  // `torch.finfo(query_states.dtype).min` (`:1253`) and NOT `-inf`: a query row
  // whose every key is masked then has a UNIFORM softmax and a FINITE output.
  const float min_bias = std::numeric_limits<float>::lowest();

  std::vector<float> ctx(static_cast<size_t>(tokens * heads * vhd));
  std::vector<float> logits(static_cast<size_t>(kv_length));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      const float* qh = query.data() + (b * heads + h) * seq_len * qk;
      const float* kh = kv.key_states.data() + (b * heads + h) * kv_length * qk;
      const float* vh = kv.value_states.data() + (b * heads + h) * kv_length * vhd;
      for (int64_t t = 0; t < seq_len; ++t) {
        const uint8_t* vis = visible.data() + (b * seq_len + t) * kv_length;
        float maxv = -std::numeric_limits<float>::infinity();
        for (int64_t s = 0; s < kv_length; ++s) {
          double acc = 0.0;
          for (int64_t i = 0; i < qk; ++i) {
            acc += static_cast<double>(qh[t * qk + i]) * kh[s * qk + i];
          }
          float v = static_cast<float>(acc) * scaling;
          if (vis[s] == 0U) v += min_bias;
          logits[static_cast<size_t>(s)] = v;
          maxv = std::max(maxv, v);
        }
        double sum = 0.0;
        for (int64_t s = 0; s < kv_length; ++s) {
          const float e = std::exp(logits[static_cast<size_t>(s)] - maxv);
          logits[static_cast<size_t>(s)] = e;
          sum += e;
        }
        const float inv = static_cast<float>(1.0 / sum);
        // `attn_output.transpose(1, 2)` then `.reshape(B, S, -1)` (`:1059`, `:1214`):
        // head-major within a token, which is what `o_proj` expects.
        float* dst = ctx.data() + (b * seq_len + t) * heads * vhd + h * vhd;
        for (int64_t dv = 0; dv < vhd; ++dv) {
          double acc = 0.0;
          for (int64_t s = 0; s < kv_length; ++s) {
            acc += static_cast<double>(logits[static_cast<size_t>(s)]) * inv *
                   vh[s * vhd + dv];
          }
          dst[dv] = static_cast<float>(acc);
        }
      }
    }
  }

  // ── `o_proj` (`:1215`) ────────────────────────────────────────────────────
  res.attn_output.assign(static_cast<size_t>(tokens * d.hidden_size), 0.0F);
  for (int64_t t = 0; t < tokens; ++t) {
    LinearNoBias(ctx.data() + t * heads * vhd, w.o_proj.data(), heads * vhd,
                 d.hidden_size, res.attn_output.data() + t * d.hidden_size);
  }

  if (cache != nullptr) cache->cached_len = kv_length;
  res.propagates_topk = role.next_skip_topk;
  return res;
}

}  // namespace vllm::glm5_next
